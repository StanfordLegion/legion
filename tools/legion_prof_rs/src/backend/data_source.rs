use std::cmp::max;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use legion_prof_viewer::{
    data::{
        Color32, DataSource, DataSourceDescription, DataSourceInfo, EntryID, EntryInfo, Field,
        FieldID, FieldSchema, Item, ItemLink, ItemMeta, ItemUID, Rgba, SlotMetaTile,
        SlotMetaTileData, SlotTile, SlotTileData, SummaryTile, SummaryTileData, TileID, TileSet,
        UtilPoint,
    },
    timestamp as ts,
};

use slice_group_by::GroupBy;

use crate::backend::common::{
    ChanEntryFieldsPretty, ChanEntryShort, DimOrderPretty, FSpaceShort, FieldsPretty, ISpacePretty,
    InstShort, MemGroup, ProcGroup, SizePretty, StatePostprocess,
};
use crate::conditional_assert;
use crate::state::{
    ChanEntry, ChanID, ChanKind, Color, Config, Container, ContainerEntry, Copy, CopyInstInfo,
    Fill, FillInstInfo, Inst, InstUID, MemID, MemKind, NodeID, OpID, ProcEntryKind, ProcID,
    ProcKind, ProfUID, State, TimeRange, Timestamp,
};

impl Into<ts::Timestamp> for Timestamp {
    fn into(self) -> ts::Timestamp {
        ts::Timestamp(self.0.try_into().unwrap())
    }
}

impl Into<Timestamp> for ts::Timestamp {
    fn into(self) -> Timestamp {
        Timestamp(self.0.try_into().unwrap())
    }
}

impl Into<ts::Interval> for TimeRange {
    fn into(self) -> ts::Interval {
        ts::Interval::new(self.start.unwrap().into(), self.stop.unwrap().into())
    }
}

impl Into<ItemUID> for ProfUID {
    fn into(self) -> ItemUID {
        ItemUID(self.0)
    }
}

impl Into<Color32> for Color {
    fn into(self) -> Color32 {
        Color32::from_rgb(
            ((self.0 >> 16) & 0xFF) as u8,
            ((self.0 >> 8) & 0xFF) as u8,
            (self.0 & 0xFF) as u8,
        )
    }
}

#[derive(Debug, Clone)]
enum EntryKind {
    ProcKind(ProcGroup),
    Proc(ProcID),
    MemKind(MemGroup),
    Mem(MemID),
    ChanKind(Option<NodeID>),
    Chan(ChanID),
}

#[derive(Debug, Clone)]
struct ItemInfo {
    point_interval: ts::Interval,
    expand: bool,
}

#[derive(Debug, Clone)]
pub struct Fields {
    chan_reqs: FieldID,
    expanded_for_visibility: FieldID,
    operation: FieldID,
    insts: FieldID,
    inst_fields: FieldID,
    inst_fspace: FieldID,
    inst_ispace: FieldID,
    inst_layout: FieldID,
    size: FieldID,
    interval: FieldID,
    num_items: FieldID,
    provenance: FieldID,
    status_ready: FieldID,
    status_running: FieldID,
    status_waiting: FieldID,
    deferred_time: FieldID,
    delayed_time: FieldID,
    creator: FieldID,
    caller: FieldID,
}

#[derive(Debug)]
pub struct StateDataSource {
    state: State,
    field_schema: FieldSchema,
    fields: Fields,
    info: EntryInfo,
    entry_map: BTreeMap<EntryID, EntryKind>,
    proc_entries: BTreeMap<ProcID, EntryID>,
    proc_groups: BTreeMap<ProcGroup, Vec<ProcID>>,
    mem_entries: BTreeMap<MemID, EntryID>,
    mem_groups: BTreeMap<MemGroup, Vec<MemID>>,
    chan_groups: BTreeMap<Option<NodeID>, Vec<ChanID>>,
    step_utilization_cache: Mutex<BTreeMap<EntryID, Arc<Vec<(Timestamp, f64)>>>>,
}

impl StateDataSource {
    pub fn new(state: State) -> Self {
        let mut field_schema = FieldSchema::new();

        let fields = Fields {
            chan_reqs: field_schema.insert("Requirements".to_owned(), true),
            expanded_for_visibility: field_schema
                .insert("(Expanded for Visibility)".to_owned(), false),
            operation: field_schema.insert("Operation".to_owned(), true),
            insts: field_schema.insert("Instances".to_owned(), true),
            inst_fields: field_schema.insert("Fields".to_owned(), true),
            inst_fspace: field_schema.insert("Field Space".to_owned(), true),
            inst_ispace: field_schema.insert("Index Space".to_owned(), true),
            inst_layout: field_schema.insert("Layout".to_owned(), true),
            interval: field_schema.insert("Lifetime".to_owned(), false),
            num_items: field_schema.insert("Number of Items".to_owned(), false),
            provenance: field_schema.insert("Provenance".to_owned(), true),
            size: field_schema.insert("Size".to_owned(), true),
            status_ready: field_schema.insert("Ready".to_owned(), false),
            status_running: field_schema.insert("Running".to_owned(), false),
            status_waiting: field_schema.insert("Waiting".to_owned(), false),
            deferred_time: field_schema.insert("Deferred".to_owned(), false),
            delayed_time: field_schema.insert("Delayed".to_owned(), false),
            creator: field_schema.insert("Creator".to_owned(), false),
            caller: field_schema.insert("Caller".to_owned(), false),
        };

        let mut entry_map = BTreeMap::<EntryID, EntryKind>::new();
        let mut proc_entries = BTreeMap::new();
        let mut mem_entries = BTreeMap::new();

        let mut proc_groups = state.group_procs();
        let mem_groups = state.group_mems();
        let chan_groups = state.group_chans();

        let mut nodes: BTreeSet<_> = proc_groups.keys().map(|ProcGroup(n, _)| *n).collect();
        let proc_kinds: BTreeSet<_> = proc_groups.keys().map(|ProcGroup(_, k)| *k).collect();
        let mem_kinds: BTreeSet<_> = mem_groups.keys().map(|MemGroup(_, k)| *k).collect();

        if !state.has_multiple_nodes() {
            nodes.remove(&None);
        }

        let mut node_slots = Vec::new();
        let root_id = EntryID::root();
        for node in &nodes {
            let node_short_name;
            let node_long_name;
            match node {
                Some(node_id) => {
                    node_short_name = format!("n{}", node_id.0);
                    node_long_name = format!("Node {}", node_id.0);
                }
                None => {
                    node_short_name = "all".to_owned();
                    node_long_name = "All Nodes".to_owned();
                }
            }
            let node_index = node_slots.len() as u64;
            let node_id = root_id.child(node_index);

            let mut kind_slots = Vec::new();
            let mut kind_index = 0;
            let mut node_empty = node.is_some();
            // Processors
            for kind in &proc_kinds {
                let group = ProcGroup(*node, *kind);

                let procs = proc_groups.get(&group).unwrap();
                if node.is_some() {
                    // Don't render kind if all processors of the kind are empty
                    let empty = procs.iter().all(|p| state.procs.get(p).unwrap().is_empty());
                    node_empty = node_empty && empty;
                    if empty {
                        continue;
                    }
                }

                let kind_name = format!("{:?}", kind);
                let kind_first_letter = kind_name.chars().next().unwrap().to_lowercase();

                let kind_id = node_id.child(kind_index);
                kind_index += 1;

                let color = match kind {
                    ProcKind::GPU => Color::OLIVEDRAB,
                    ProcKind::CPU => Color::STEELBLUE,
                    ProcKind::Utility => Color::CRIMSON,
                    ProcKind::IO => Color::ORANGERED,
                    ProcKind::ProcGroup => Color::ORANGERED,
                    ProcKind::ProcSet => Color::ORANGERED,
                    ProcKind::OpenMP => Color::ORANGERED,
                    ProcKind::Python => Color::OLIVEDRAB,
                };
                let color: Color32 = color.into();

                let mut proc_slots = Vec::new();
                if node.is_some() {
                    for (proc_index, proc) in procs.iter().enumerate() {
                        let proc_id = kind_id.child(proc_index as u64);
                        entry_map.insert(proc_id.clone(), EntryKind::Proc(*proc));
                        proc_entries.insert(*proc, proc_id);

                        let rows = state.procs.get(proc).unwrap().max_levels as u64 + 1;
                        proc_slots.push(EntryInfo::Slot {
                            short_name: format!("{}{}", kind_first_letter, proc.proc_in_node()),
                            long_name: format!(
                                "{} {} {}",
                                node_long_name,
                                kind_name,
                                proc.proc_in_node()
                            ),
                            max_rows: rows,
                        });
                    }
                }

                let summary_id = kind_id.summary();
                entry_map.insert(summary_id, EntryKind::ProcKind(group));

                kind_slots.push(EntryInfo::Panel {
                    short_name: kind_name.to_lowercase(),
                    long_name: format!("{} {}", node_long_name, kind_name),
                    summary: Some(Box::new(EntryInfo::Summary { color })),
                    slots: proc_slots,
                });
            }

            // Don't render node if all processors of the node are empty
            if node_empty {
                // Remove this node's processors from the all nodes list to
                // avoid influencing global utilization
                for kind in &proc_kinds {
                    let group = ProcGroup(None, *kind);
                    proc_groups
                        .get_mut(&group)
                        .unwrap()
                        .retain(|p| p.node_id() != node.unwrap());
                }
                continue;
            }

            // Memories
            for kind in &mem_kinds {
                let group = MemGroup(*node, *kind);

                let Some(mems) = mem_groups.get(&group) else {
                    continue;
                };

                let kind_name = format!("{:?}", kind);
                let kind_first_letter = kind_name.chars().next().unwrap().to_lowercase();

                let kind_id = node_id.child(kind_index);
                kind_index += 1;

                let color = match kind {
                    MemKind::NoMemKind => unreachable!(),
                    MemKind::Global => Color::CRIMSON,
                    MemKind::System => Color::OLIVEDRAB,
                    MemKind::Registered => Color::DARKMAGENTA,
                    MemKind::Socket => Color::ORANGERED,
                    MemKind::ZeroCopy => Color::CRIMSON,
                    MemKind::Framebuffer => Color::BLUE,
                    MemKind::Disk => Color::DARKGOLDENROD,
                    MemKind::HDF5 => Color::OLIVEDRAB,
                    MemKind::File => Color::ORANGERED,
                    MemKind::L3Cache => Color::CRIMSON,
                    MemKind::L2Cache => Color::DARKMAGENTA,
                    MemKind::L1Cache => Color::OLIVEDRAB,
                    MemKind::GPUManaged => Color::DARKMAGENTA,
                    MemKind::GPUDynamic => Color::ORANGERED,
                };
                let color: Color32 = color.into();

                let mut mem_slots = Vec::new();
                if node.is_some() {
                    for (mem_index, mem) in mems.iter().enumerate() {
                        let mem_id = kind_id.child(mem_index as u64);
                        entry_map.insert(mem_id.clone(), EntryKind::Mem(*mem));
                        mem_entries.insert(*mem, mem_id);

                        let rows = state.mems.get(mem).unwrap().max_live_insts as u64 + 1;
                        mem_slots.push(EntryInfo::Slot {
                            short_name: format!("{}{}", kind_first_letter, mem.mem_in_node()),
                            long_name: format!(
                                "{} {} {}",
                                node_long_name,
                                kind_name,
                                mem.mem_in_node()
                            ),
                            max_rows: rows,
                        });
                    }
                }

                let summary_id = kind_id.summary();
                entry_map.insert(summary_id, EntryKind::MemKind(group));

                kind_slots.push(EntryInfo::Panel {
                    short_name: kind_name.to_lowercase(),
                    long_name: format!("{} {}", node_long_name, kind_name),
                    summary: Some(Box::new(EntryInfo::Summary { color })),
                    slots: mem_slots,
                });
            }

            // Channels
            loop {
                let Some(chans) = chan_groups.get(node) else {
                    break;
                };

                let kind_id = node_id.child(kind_index);

                let color: Color32 = Color::ORANGERED.into();

                let mut chan_slots = Vec::new();
                if node.is_some() {
                    for (chan_index, chan) in chans.iter().enumerate() {
                        let chan_id = kind_id.child(chan_index as u64);
                        entry_map.insert(chan_id, EntryKind::Chan(*chan));

                        let (src_name, src_short) = if let Some(mem) = chan.src {
                            let kind = state.mems.get(&mem).unwrap().kind;
                            let kind_first_letter =
                                format!("{:?}", kind).chars().next().unwrap().to_lowercase();
                            let src_node = mem.node_id().0;
                            (
                                Some(format!(
                                    "Node {} {:?} {}",
                                    src_node,
                                    kind,
                                    mem.mem_in_node()
                                )),
                                Some(format!(
                                    "n{}{}{}",
                                    src_node,
                                    kind_first_letter,
                                    mem.mem_in_node()
                                )),
                            )
                        } else {
                            (None, None)
                        };

                        let (dst_name, dst_short) = if let Some(mem) = chan.dst {
                            let kind = state.mems.get(&mem).unwrap().kind;
                            let kind_first_letter =
                                format!("{:?}", kind).chars().next().unwrap().to_lowercase();
                            let dst_node = mem.node_id().0;
                            (
                                Some(format!(
                                    "Node {} {:?} {}",
                                    dst_node,
                                    kind,
                                    mem.mem_in_node()
                                )),
                                Some(format!(
                                    "n{}{}{}",
                                    dst_node,
                                    kind_first_letter,
                                    mem.mem_in_node()
                                )),
                            )
                        } else {
                            (None, None)
                        };

                        let short_name = match chan.channel_kind {
                            ChanKind::Copy => {
                                format!("{}-{}", src_short.unwrap(), dst_short.unwrap())
                            }
                            ChanKind::Fill => format!("f {}", dst_short.unwrap()),
                            ChanKind::Gather => format!("g {}", dst_short.unwrap()),
                            ChanKind::Scatter => format!("s {}", src_short.unwrap()),
                            ChanKind::DepPart => "dp".to_owned(),
                        };

                        let long_name = match chan.channel_kind {
                            ChanKind::Copy => {
                                format!("{} to {}", src_name.unwrap(), dst_name.unwrap())
                            }
                            ChanKind::Fill => format!("Fill {}", dst_name.unwrap()),
                            ChanKind::Gather => format!("Gather to {}", dst_name.unwrap()),
                            ChanKind::Scatter => {
                                format!("Scatter from {}", src_name.unwrap())
                            }
                            ChanKind::DepPart => "Dependent Partitioning".to_owned(),
                        };

                        let rows = state.chans.get(chan).unwrap().max_levels as u64 + 1;
                        chan_slots.push(EntryInfo::Slot {
                            short_name,
                            long_name,
                            max_rows: rows,
                        });
                    }
                }

                let summary_id = kind_id.summary();
                entry_map.insert(summary_id, EntryKind::ChanKind(*node));

                kind_slots.push(EntryInfo::Panel {
                    short_name: "chan".to_owned(),
                    long_name: format!("{} Channel", node_long_name),
                    summary: Some(Box::new(EntryInfo::Summary { color })),
                    slots: chan_slots,
                });

                break;
            }
            node_slots.push(EntryInfo::Panel {
                short_name: node_short_name,
                long_name: node_long_name,
                summary: None,
                slots: kind_slots,
            });
        }

        let info = EntryInfo::Panel {
            short_name: "root".to_owned(),
            long_name: "root".to_owned(),
            summary: None,
            slots: node_slots,
        };

        Self {
            state,
            field_schema,
            fields,
            info,
            entry_map,
            proc_entries,
            proc_groups,
            mem_entries,
            mem_groups,
            chan_groups,
            step_utilization_cache: Mutex::new(BTreeMap::new()),
        }
    }
}

impl StateDataSource {
    /// A step utilization is a series of step functions. At time T, the
    /// utilization takes value U. That value continues until the next
    /// step. This is a good match for Legion's discrete execution model (a
    /// task is either running, or it is not), but doesn't play so well with
    /// interpolation and level of detail. We compute this first because it's
    /// how the profiler internally represents utilization, but we convert it
    /// to a more useful format below.
    fn generate_step_utilization(&self, entry_id: &EntryID) -> Arc<Vec<(Timestamp, f64)>> {
        // This is an INTENTIONAL race; if two requests for the same entry
        // arrive simultaneously, we'll miss in the cache on both and compute
        // the utilization twice. The result should be the same, so this is
        // mostly wasted computation (in exchange for enabling parallelism).

        let cache = &self.step_utilization_cache;
        if let Some(util) = cache.lock().unwrap().get(entry_id) {
            return util.clone();
        }

        let step_utilization = match self.entry_map.get(entry_id).unwrap() {
            EntryKind::ProcKind(group) => {
                let procs = self.proc_groups.get(group).unwrap();
                let points = self.state.proc_group_timepoints(procs);
                let count = procs.len() as u64;
                let owners: BTreeSet<_> = procs
                    .iter()
                    .zip(points.iter())
                    .filter(|(_, tp)| !tp.is_empty())
                    .map(|(proc_id, _)| *proc_id)
                    .collect();

                if owners.is_empty() {
                    Vec::new()
                } else {
                    let mut utilizations = Vec::new();
                    for tp in points {
                        if !tp.is_empty() {
                            self.state
                                .convert_points_to_utilization(tp, &mut utilizations);
                        }
                    }
                    utilizations.sort_by_key(|point| point.time_key());
                    self.state
                        .calculate_proc_utilization_data(utilizations, owners, count)
                }
            }
            EntryKind::MemKind(group) => {
                let mems = self.mem_groups.get(group).unwrap();
                let points = self.state.mem_group_timepoints(mems);
                let owners: BTreeSet<_> = mems
                    .iter()
                    .zip(points.iter())
                    .filter(|(_, tp)| !tp.is_empty())
                    .map(|(mem_id, _)| *mem_id)
                    .collect();

                if owners.is_empty() {
                    Vec::new()
                } else {
                    let mut utilizations: Vec<_> = points
                        .iter()
                        .filter(|tp| !tp.is_empty())
                        .flat_map(|tp| *tp)
                        .collect();
                    utilizations.sort_by_key(|point| point.time_key());
                    self.state
                        .calculate_mem_utilization_data(utilizations, owners)
                }
            }
            EntryKind::ChanKind(node) => {
                let chans = self.chan_groups.get(node).unwrap();
                let points = self.state.chan_group_timepoints(chans);
                let owners: BTreeSet<_> = chans
                    .iter()
                    .zip(points.iter())
                    .filter(|(_, tp)| !tp.is_empty())
                    .map(|(chan_id, _)| *chan_id)
                    .collect();

                if owners.is_empty() {
                    Vec::new()
                } else {
                    let mut utilizations = Vec::new();
                    for tp in points {
                        if !tp.is_empty() {
                            self.state
                                .convert_points_to_utilization(tp, &mut utilizations);
                        }
                    }
                    utilizations.sort_by_key(|point| point.time_key());
                    self.state
                        .calculate_chan_utilization_data(utilizations, owners)
                }
            }
            _ => unreachable!(),
        };
        let result = Arc::new(step_utilization);
        cache
            .lock()
            .unwrap()
            .insert(entry_id.clone(), result.clone());
        result
    }

    /// Converts the step utilization into a sample utilization, where each
    /// utilization point (sample) represents the average utilization over a
    /// certain time interval. The sample is located in the middle of the
    /// interval.
    fn compute_sample_utilization(
        step_utilization: &Vec<(Timestamp, f64)>,
        interval: ts::Interval,
        samples: u64,
    ) -> Vec<UtilPoint> {
        let start_time = interval.start.0 as u64;
        let duration = interval.duration_ns() as u64;

        let first_index = step_utilization
            .partition_point(|&(t, _)| {
                let t: ts::Timestamp = t.into();
                t < interval.start
            })
            .saturating_sub(1);

        let mut last_index = step_utilization.partition_point(|&(t, _)| {
            let t: ts::Timestamp = t.into();
            t < interval.stop
        });
        if last_index + 1 < step_utilization.len() {
            last_index = last_index + 1;
        }

        let mut utilization = Vec::new();
        let mut last_t = Timestamp(0);
        let mut last_u = 0.0;
        let mut step_it = step_utilization[first_index..last_index].iter().peekable();
        for sample in 0..samples {
            let sample_start = Timestamp(duration * sample / samples + start_time);
            let sample_stop = Timestamp(duration * (sample + 1) / samples + start_time);
            if sample_stop.0 - sample_start.0 == 0 {
                continue;
            }

            let mut sample_util = 0.0;
            while let Some((t, u)) = step_it.next_if(|(t, _)| *t < sample_stop) {
                if *t < sample_start {
                    (last_t, last_u) = (*t, *u);
                    continue;
                }

                // This is a step utilization. So utilization u begins on time
                // t. That means the previous utilization stop at time t-1.
                let last_duration = (t.0 - 1).saturating_sub(last_t.0.max(sample_start.0));
                sample_util += last_duration as f64 * last_u;

                (last_t, last_u) = (*t, *u);
            }
            if last_t < sample_stop {
                let last_duration = sample_stop.0 - last_t.0.max(sample_start.0);
                sample_util += last_duration as f64 * last_u;
            }

            sample_util = sample_util / (sample_stop.0 - sample_start.0) as f64;
            assert!(sample_util <= 1.0);
            utilization.push(UtilPoint {
                time: Timestamp((sample_start.0 + sample_stop.0) / 2).into(),
                util: sample_util as f32,
            });
        }
        utilization
    }

    /// Items smaller than this should be expanded (and merged, if suitable
    /// nearby items are found)
    const MAX_RATIO: f64 = 2000.0;

    /// Items larger than this should NOT be merged, even if nearby an expanded
    /// item
    const MIN_RATIO: f64 = 1000.0;

    /// Expand small items to improve visibility
    fn expand_item(
        interval: &mut ts::Interval,
        tile_id: TileID,
        last: Option<&Item>,
        merged: u64,
    ) -> bool {
        let view_ratio = tile_id.0.duration_ns() as f64 / interval.duration_ns() as f64;

        let expand = view_ratio > Self::MAX_RATIO;
        if expand {
            let min_duration = tile_id.0.duration_ns() as f64 / Self::MAX_RATIO;
            let center = (interval.start.0 + interval.stop.0) as f64 / 2.0;
            let start = ts::Timestamp((center - min_duration / 2.0) as i64);
            let stop = ts::Timestamp(start.0 + min_duration as i64);
            *interval = ts::Interval::new(start, stop);

            // If the previous task is large (and overlaps), shrink to avoid overlapping it
            if let Some(last) = last {
                let last_ratio =
                    tile_id.0.duration_ns() as f64 / last.interval.duration_ns() as f64;
                if interval.overlaps(last.interval) && last_ratio < Self::MIN_RATIO {
                    if merged > 0 {
                        // It's already a merged task, ok to keep merging
                    } else {
                        interval.start = last.interval.stop;
                    }
                }
            }
        }
        expand
    }

    /// Merge small tasks to reduce load on renderer
    fn merge_items(
        interval: ts::Interval,
        tile_id: TileID,
        last: &mut Item,
        last_meta: Option<&mut ItemMeta>,
        num_items_field: FieldID,
        merged: &mut u64,
    ) -> bool {
        // Check for overlap with previous task. If so, either one or the
        // other task was expanded (since tasks don't normally overlap)
        // and this is a good opportunity to combine them.
        if last.interval.overlaps(interval) {
            // If the current task is large, don't merge. Instead,
            // just modify the previous task so it doesn't overlap
            let view_ratio = tile_id.0.duration_ns() as f64 / interval.duration_ns() as f64;
            if view_ratio < Self::MIN_RATIO {
                last.interval.stop = interval.start;
            } else {
                last.interval.stop = interval.stop;
                last.color = Color::GRAY.into();
                if let Some(last_meta) = last_meta {
                    if let Some((_, Field::U64(value))) = last_meta.fields.get_mut(0) {
                        *value += 1;
                    } else {
                        last_meta.title = "Merged Tasks".to_owned();
                        last_meta.fields = vec![(num_items_field, Field::U64(2))];
                    }
                }
                *merged += 1;
                return true;
            }
        }
        *merged = 0;
        false
    }

    fn build_items<C>(
        &self,
        cont: &C,
        tile_id: TileID,
        full: bool,
        mut item_metas: Option<&mut Vec<Vec<ItemMeta>>>,
        get_meta: impl Fn(&C::Entry, ItemInfo) -> ItemMeta,
    ) -> Vec<Vec<Item>>
    where
        C: Container,
    {
        let mut items: Vec<Vec<Item>> = Vec::new();
        let mut merged = Vec::new();
        items.resize_with(cont.max_levels() + 1, Vec::new);
        if let Some(ref mut item_metas) = item_metas {
            item_metas.resize_with(cont.max_levels() + 1, Vec::new);
        }
        merged.resize(cont.max_levels() + 1, 0u64);
        let points = cont.time_points();

        for point in points {
            assert!(point.first);

            let entry = cont.entry(point.entry);
            let (base, time_range, waiters) = (&entry.base(), entry.time_range(), &entry.waiters());

            let point_interval: ts::Interval = time_range.into();
            if !point_interval.overlaps(tile_id.0) {
                continue;
            }
            let mut view_interval = point_interval.intersection(tile_id.0);

            let level = base.level.unwrap() as usize;

            let expand = !full
                && Self::expand_item(
                    &mut view_interval,
                    tile_id,
                    items[level].last(),
                    merged[level],
                );

            if let Some(last) = items[level].last_mut() {
                let last_meta = if let Some(ref mut item_metas) = item_metas {
                    item_metas[level].last_mut()
                } else {
                    None
                };
                if Self::merge_items(
                    view_interval,
                    tile_id,
                    last,
                    last_meta,
                    self.fields.num_items,
                    &mut merged[level],
                ) {
                    continue;
                }
            }

            let color = entry.color(&self.state);
            let color: Color32 = color.into();
            let color: Rgba = color.into();

            let item_meta = item_metas.as_ref().map(|_| {
                get_meta(
                    entry,
                    ItemInfo {
                        point_interval,
                        expand,
                    },
                )
            });

            let mut add_item = |interval: ts::Interval, opacity: f32, status: Option<FieldID>| {
                if !interval.overlaps(tile_id.0) {
                    return;
                }
                let view_interval = interval.intersection(tile_id.0);
                let color = (Rgba::WHITE.multiply(1.0 - opacity) + color.multiply(opacity)).into();
                let item = Item {
                    item_uid: base.prof_uid.into(),
                    interval: view_interval,
                    color,
                };
                items[level].push(item);
                if let Some(ref mut item_metas) = item_metas {
                    let mut item_meta = item_meta.clone().unwrap();
                    if let Some(status) = status {
                        item_meta
                            .fields
                            .insert(1, (status, Field::Interval(interval)));
                    }
                    item_metas[level].push(item_meta);
                }
            };
            if let Some(waiters) = waiters {
                let mut start = time_range.start.unwrap();
                for wait in &waiters.wait_intervals {
                    let running_interval = ts::Interval::new(start.into(), wait.start.into());
                    let waiting_interval = ts::Interval::new(wait.start.into(), wait.ready.into());
                    let ready_interval = ts::Interval::new(wait.ready.into(), wait.end.into());
                    add_item(running_interval, 1.0, Some(self.fields.status_running));
                    add_item(waiting_interval, 0.15, Some(self.fields.status_waiting));
                    add_item(ready_interval, 0.45, Some(self.fields.status_ready));
                    start = max(start, wait.end);
                }
                let stop = time_range.stop.unwrap();
                if start < stop {
                    let running_interval = ts::Interval::new(start.into(), stop.into());
                    add_item(running_interval, 1.0, Some(self.fields.status_running));
                }
            } else {
                add_item(view_interval, 1.0, None);
            }
        }
        items
    }

    fn generate_proc_slot_tile(
        &self,
        entry_id: &EntryID,
        proc_id: ProcID,
        tile_id: TileID,
        full: bool,
    ) -> SlotTile {
        let proc = self.state.procs.get(&proc_id).unwrap();
        let items = self.build_items(proc, tile_id, full, None, |_, _| unreachable!());
        SlotTile {
            entry_id: entry_id.clone(),
            tile_id,
            data: SlotTileData { items },
        }
    }

    fn generate_op_link(&self, op_id: OpID) -> Field {
        if let Some(proc_id) = self.state.tasks.get(&op_id) {
            if let Some(proc) = self.state.procs.get(proc_id) {
                let op = proc.find_task(op_id).unwrap();
                let op_name = op.name(&self.state);
                return Field::ItemLink(ItemLink {
                    item_uid: op.base().prof_uid.into(),
                    title: op_name,
                    interval: op.time_range().into(),
                    entry_id: self.proc_entries.get(proc_id).unwrap().clone(),
                });
            }
        }
        if let Some(task) = self.state.multi_tasks.get(&op_id) {
            if let Some(kind) = self.state.task_kinds.get(&task.task_id) {
                if let Some(name) = &kind.name {
                    return Field::String(format!("Task {}<{}>", name, op_id.0));
                }
            }
        }
        if let Some(op) = self.state.find_op(op_id) {
            if let Some(kind) = op.kind {
                return Field::String(format!(
                    "{} Operation<{}>",
                    self.state.op_kinds[&kind].name, op_id.0
                ));
            }
        }
        Field::U64(op_id.0)
    }

    fn generate_inst_link(&self, inst_uid: InstUID, prefix: &str) -> Option<Field> {
        let mem_id = self.state.insts.get(&inst_uid)?;
        let mem = self.state.mems.get(mem_id)?;
        let inst = mem.insts.get(&inst_uid)?;

        Some(Field::ItemLink(ItemLink {
            item_uid: inst.base().prof_uid.into(),
            title: format!("{}0x{:x}", prefix, inst.inst_id.unwrap().0),
            interval: inst.time_range().into(),
            entry_id: self.mem_entries.get(mem_id).unwrap().clone(),
        }))
    }

    fn generate_creator_link(&self, prof_uid: ProfUID, create_time: Timestamp) -> Field {
        let proc_id = self.state.prof_uid_proc.get(&prof_uid).unwrap();
        let proc = self.state.procs.get(&proc_id).unwrap();
        let mut entry = proc.find_entry(prof_uid).unwrap();
        // Check to see if we need to link one of the subcalls instead
        // of the main task that produced the this operation
        // Subcalls are sorted from smallest to largest so the first one we hit
        // is the one we know that that actually made this box
        for (call_uid, start_time, stop_time) in &entry.subcalls {
            if (*start_time <= create_time) && (create_time < *stop_time) {
                entry = proc.find_entry(*call_uid).unwrap();
                break;
            }
        }
        let op_name = entry.name(&self.state);
        Field::ItemLink(ItemLink {
            item_uid: entry.base().prof_uid.into(),
            title: op_name,
            interval: entry.time_range().into(),
            entry_id: self.proc_entries.get(proc_id).unwrap().clone(),
        })
    }

    fn generate_proc_slot_meta_tile(
        &self,
        entry_id: &EntryID,
        proc_id: ProcID,
        tile_id: TileID,
        full: bool,
    ) -> SlotMetaTile {
        let proc = self.state.procs.get(&proc_id).unwrap();
        let mut item_metas: Vec<Vec<ItemMeta>> = Vec::new();
        let items = self.build_items(proc, tile_id, full, Some(&mut item_metas), |entry, info| {
            let ItemInfo {
                point_interval,
                expand,
            } = info;

            let name = entry.name(&self.state);
            let provenance = entry.provenance(&self.state);

            let mut fields = Vec::new();
            if expand {
                fields.push((self.fields.expanded_for_visibility, Field::Empty));
            }
            fields.push((self.fields.interval, Field::Interval(point_interval)));
            if let Some(initiation_op) = entry.initiation_op {
                // FIXME: You might think that initiation_op is None rather than
                // needing this check with zero, but backwards compatibility is hard
                // You can remove this check once we stop needing to be compatible with Python
                if initiation_op.0 > 0 {
                    fields.push((self.fields.operation, self.generate_op_link(initiation_op)));
                }
            }
            if let Some(op_id) = entry.op_id {
                let op = self.state.find_op(op_id).unwrap();
                let inst_set: BTreeSet<_> =
                    op.operation_inst_infos.iter().map(|i| i.inst_uid).collect();

                let insts: Vec<_> = inst_set
                    .iter()
                    .flat_map(|i| {
                        let result = self.generate_inst_link(*i, "");
                        conditional_assert!(
                            result.is_some(),
                            Config::all_logs(),
                            "Cannot find instance 0x{:x}",
                            i.0
                        );
                        result
                    })
                    .collect();
                fields.push((self.fields.insts, Field::Vec(insts)));
            }
            if let Some(provenance) = provenance {
                fields.push((
                    self.fields.provenance,
                    Field::String(provenance.to_string()),
                ));
            }
            if let Some(creator) = self.state.fevents.get(&entry.creator) {
                // Check to see if these are function calls or tasks
                match entry.kind {
                    ProcEntryKind::MapperCall(_)
                    | ProcEntryKind::RuntimeCall(_)
                    | ProcEntryKind::GPUKernel(_, _) => {
                        if let Some(start_time) = entry.time_range.start {
                            fields.push((
                                self.fields.caller,
                                // Use the first tick before the start so it is outside
                                // of our box but hopefully in the caller's box
                                self.generate_creator_link(*creator, start_time - Timestamp(1)),
                            ));
                        }
                    }
                    _ => {
                        // Everything else can use the create time to find the creator
                        if let Some(create_time) = entry.time_range.create {
                            fields.push((
                                self.fields.creator,
                                self.generate_creator_link(*creator, create_time),
                            ));
                        }
                    }
                }
            }
            if let Some(ready) = entry.time_range.ready {
                if let Some(create) = entry.time_range.create {
                    fields.push((
                        self.fields.deferred_time,
                        Field::Interval(ts::Interval::new(create.into(), ready.into())),
                    ));
                }
                if let Some(start) = entry.time_range.start {
                    fields.push((
                        self.fields.delayed_time,
                        Field::Interval(ts::Interval::new(ready.into(), start.into())),
                    ));
                }
            }
            ItemMeta {
                item_uid: entry.base().prof_uid.into(),
                title: name,
                original_interval: point_interval,
                fields,
            }
        });
        assert_eq!(items.len(), item_metas.len());
        for (item_row, item_meta_row) in items.iter().zip(item_metas.iter()) {
            assert_eq!(item_row.len(), item_meta_row.len());
        }
        SlotMetaTile {
            entry_id: entry_id.clone(),
            tile_id,
            data: SlotMetaTileData { items: item_metas },
        }
    }

    fn generate_mem_slot_tile(
        &self,
        entry_id: &EntryID,
        mem_id: MemID,
        tile_id: TileID,
        full: bool,
    ) -> SlotTile {
        let mem = self.state.mems.get(&mem_id).unwrap();
        let items = self.build_items(mem, tile_id, full, None, |_, _| unreachable!());
        SlotTile {
            entry_id: entry_id.clone(),
            tile_id,
            data: SlotTileData { items },
        }
    }

    fn generate_inst_regions(&self, inst: &Inst, result: &mut Vec<(FieldID, Field)>) {
        for (ispace_id, fspace_id) in inst.ispace_ids.iter().zip(inst.fspace_ids.iter()) {
            let ispace = format!("{}", ISpacePretty(*ispace_id, &self.state),);
            result.push((self.fields.inst_ispace, Field::String(ispace)));

            let fspace = self.state.field_spaces.get(&fspace_id).unwrap();
            let fspace_name = format!("{}", FSpaceShort(&fspace));
            result.push((self.fields.inst_fspace, Field::String(fspace_name)));

            let fields = format!("{}", FieldsPretty(&fspace, inst));
            result.push((self.fields.inst_fields, Field::String(fields)));
        }
    }

    fn generate_inst_layout(&self, inst: &Inst, result: &mut Vec<(FieldID, Field)>) {
        let layout = format!("{}", DimOrderPretty(inst, false));
        result.push((self.fields.inst_layout, Field::String(layout)));
    }

    fn generate_inst_size(&self, inst: &Inst, result: &mut Vec<(FieldID, Field)>) {
        let size = format!("{}", SizePretty(inst.size.unwrap()));
        result.push((self.fields.size, Field::String(size)));
    }

    fn generate_mem_slot_meta_tile(
        &self,
        entry_id: &EntryID,
        mem_id: MemID,
        tile_id: TileID,
        full: bool,
    ) -> SlotMetaTile {
        let mem = self.state.mems.get(&mem_id).unwrap();
        let mut item_metas: Vec<Vec<ItemMeta>> = Vec::new();
        let items = self.build_items(mem, tile_id, full, Some(&mut item_metas), |entry, info| {
            let ItemInfo {
                point_interval,
                expand,
            } = info;

            let name = format!("Instance {}", InstShort(entry));
            let provenance = entry.provenance(&self.state);

            let mut fields = Vec::new();
            if expand {
                fields.push((self.fields.expanded_for_visibility, Field::Empty));
            }
            fields.push((self.fields.interval, Field::Interval(point_interval)));
            self.generate_inst_regions(entry, &mut fields);
            self.generate_inst_layout(entry, &mut fields);
            self.generate_inst_size(entry, &mut fields);
            if let Some(initiation_op) = entry.initiation() {
                // FIXME: You might think that initiation_op is None rather than
                // needing this check with zero, but backwards compatibility is hard
                // You can remove this check once we stop needing to be compatible with Python
                if initiation_op.0 > 0 {
                    fields.push((self.fields.operation, self.generate_op_link(initiation_op)));
                }
            }
            if let Some(provenance) = provenance {
                fields.push((
                    self.fields.provenance,
                    Field::String(provenance.to_string()),
                ));
            }
            if let Some(creator_event) = entry.creator {
                if let Some(creator) = self.state.fevents.get(&creator_event) {
                    if let Some(create_time) = entry.time_range.create {
                        fields.push((
                            self.fields.creator,
                            self.generate_creator_link(*creator, create_time),
                        ));
                    }
                }
            }
            ItemMeta {
                item_uid: entry.base().prof_uid.into(),
                title: name,
                original_interval: point_interval,
                fields,
            }
        });
        assert_eq!(items.len(), item_metas.len());
        for (item_row, item_meta_row) in items.iter().zip(item_metas.iter()) {
            assert_eq!(item_row.len(), item_meta_row.len());
        }
        SlotMetaTile {
            entry_id: entry_id.clone(),
            tile_id,
            data: SlotMetaTileData { items: item_metas },
        }
    }

    fn generate_chan_slot_tile(
        &self,
        entry_id: &EntryID,
        chan_id: ChanID,
        tile_id: TileID,
        full: bool,
    ) -> SlotTile {
        let chan = self.state.chans.get(&chan_id).unwrap();
        let items = self.build_items(chan, tile_id, full, None, |_, _| unreachable!());
        SlotTile {
            entry_id: entry_id.clone(),
            tile_id,
            data: SlotTileData { items },
        }
    }

    fn generate_copy_reqs(&self, copy: &Copy, result_reqs: &mut Vec<Field>) {
        let groups = copy.copy_inst_infos.linear_group_by(|a, b| {
            a.src_inst_uid == b.src_inst_uid
                && a.dst_inst_uid == b.dst_inst_uid
                && a.num_hops == b.num_hops
        });
        let mut i = 0;
        for group in groups {
            let req_nums = if group.len() == 1 {
                format!("Requirement {}", i)
            } else {
                format!("Requirements {}-{}", i, i + group.len() - 1)
            };
            result_reqs.push(Field::String(req_nums));

            let CopyInstInfo {
                src_inst_uid,
                dst_inst_uid,
                num_hops,
                ..
            } = group[0];

            let src_inst = self.state.find_inst(src_inst_uid);
            let dst_inst = self.state.find_inst(dst_inst_uid);

            let src_fids = group.iter().map(|x| x.src_fid).collect();
            let src_fields = format!(
                "Fields: {}",
                ChanEntryFieldsPretty(src_inst, &src_fids, &self.state)
            );

            let dst_fids = group.iter().map(|x| x.dst_fid).collect();
            let dst_fields = format!(
                "Fields: {}",
                ChanEntryFieldsPretty(dst_inst, &dst_fids, &self.state)
            );

            match (src_inst_uid.0, dst_inst_uid.0) {
                (0, 0) => unreachable!(),
                (0, _) => {
                    let prefix = "Scatter: destination indirect instance ";
                    if let Some(dst) = self.generate_inst_link(dst_inst_uid, prefix) {
                        result_reqs.push(dst);
                    } else {
                        result_reqs.push(Field::String(format!("{}<unknown instance>", prefix)));
                    }
                    result_reqs.push(Field::String(dst_fields));
                }
                (_, 0) => {
                    let prefix = "Gather: source indirect instance ";
                    if let Some(src) = self.generate_inst_link(src_inst_uid, prefix) {
                        result_reqs.push(src);
                    } else {
                        result_reqs.push(Field::String(format!("{}<unknown instance>", prefix)));
                    }
                    result_reqs.push(Field::String(src_fields));
                }
                (_, _) => {
                    let prefix = "Source: ";
                    if let Some(src) = self.generate_inst_link(src_inst_uid, prefix) {
                        result_reqs.push(src);
                    } else {
                        result_reqs.push(Field::String(format!("{}<unknown instance>", prefix)));
                    }
                    result_reqs.push(Field::String(src_fields));

                    let prefix = "Destination: ";
                    if let Some(dst) = self.generate_inst_link(dst_inst_uid, prefix) {
                        result_reqs.push(dst);
                    } else {
                        result_reqs.push(Field::String(format!("{}<unknown instance>", prefix)));
                    }
                    result_reqs.push(Field::String(dst_fields));
                }
            }

            result_reqs.push(Field::String(format!("Number of Hops: {}", num_hops)));

            i += group.len();
        }
    }

    fn generate_fill_reqs(&self, fill: &Fill, result_reqs: &mut Vec<Field>) {
        let groups = fill
            .fill_inst_infos
            .linear_group_by(|a, b| a.dst_inst_uid == b.dst_inst_uid);
        let mut i = 0;
        for group in groups {
            let req_nums = if group.len() == 1 {
                format!("Requirement {}", i)
            } else {
                format!("Requirements {}-{}", i, i + group.len() - 1)
            };
            result_reqs.push(Field::String(req_nums));

            let FillInstInfo { dst_inst_uid, .. } = group[0];

            let dst_inst = self.state.find_inst(dst_inst_uid);

            let dst_fids = group.iter().map(|x| x.fid).collect();
            let dst_fields = format!(
                "Fields: {}",
                ChanEntryFieldsPretty(dst_inst, &dst_fids, &self.state)
            );

            let prefix = "Destination: ";
            if let Some(dst) = self.generate_inst_link(dst_inst_uid, prefix) {
                result_reqs.push(dst);
            } else {
                result_reqs.push(Field::String(format!("{}<unknown instance>", prefix)));
            }
            result_reqs.push(Field::String(dst_fields));

            i += group.len();
        }
    }

    fn generate_chan_reqs(&self, entry: &ChanEntry, result: &mut Vec<(FieldID, Field)>) {
        let mut result_reqs = Vec::new();
        match entry {
            ChanEntry::Copy(copy) => {
                self.generate_copy_reqs(copy, &mut result_reqs);
            }
            ChanEntry::Fill(fill) => {
                self.generate_fill_reqs(fill, &mut result_reqs);
            }
            ChanEntry::DepPart(_) => {}
        }
        result.push((self.fields.chan_reqs, Field::Vec(result_reqs)));
    }

    fn generate_chan_size(&self, entry: &ChanEntry, result: &mut Vec<(FieldID, Field)>) {
        let size = match entry {
            ChanEntry::Copy(copy) => copy.size,
            ChanEntry::Fill(fill) => fill.size,
            ChanEntry::DepPart(_) => return,
        };
        let size = format!("{}", SizePretty(size));
        result.push((self.fields.size, Field::String(size)));
    }

    fn generate_chan_slot_meta_tile(
        &self,
        entry_id: &EntryID,
        chan_id: ChanID,
        tile_id: TileID,
        full: bool,
    ) -> SlotMetaTile {
        let chan = self.state.chans.get(&chan_id).unwrap();
        let mut item_metas: Vec<Vec<ItemMeta>> = Vec::new();
        let items = self.build_items(chan, tile_id, full, Some(&mut item_metas), |entry, info| {
            let ItemInfo {
                point_interval,
                expand,
            } = info;

            let name = format!("{}", ChanEntryShort(entry));
            let provenance = entry.provenance(&self.state);

            let mut fields = Vec::new();
            if expand {
                fields.push((self.fields.expanded_for_visibility, Field::Empty));
            }
            fields.push((self.fields.interval, Field::Interval(point_interval)));
            self.generate_chan_reqs(entry, &mut fields);
            self.generate_chan_size(entry, &mut fields);
            if let Some(initiation_op) = entry.initiation() {
                // FIXME: You might think that initiation_op is None rather than
                // needing this check with zero, but backwards compatibility is hard
                // You can remove this check once we stop needing to be compatible with Python
                if initiation_op.0 > 0 {
                    fields.push((self.fields.operation, self.generate_op_link(initiation_op)));
                }
            }
            if let Some(provenance) = provenance {
                fields.push((
                    self.fields.provenance,
                    Field::String(provenance.to_string()),
                ));
            }
            if let Some(creator) = entry.creator() {
                if let Some(creator_uid) = self.state.fevents.get(&creator) {
                    if let Some(create_time) = entry.time_range().create {
                        fields.push((
                            self.fields.creator,
                            self.generate_creator_link(*creator_uid, create_time),
                        ));
                    }
                }
            }
            ItemMeta {
                item_uid: entry.base().prof_uid.into(),
                title: name,
                original_interval: point_interval,
                fields,
            }
        });
        assert_eq!(items.len(), item_metas.len());
        for (item_row, item_meta_row) in items.iter().zip(item_metas.iter()) {
            assert_eq!(item_row.len(), item_meta_row.len());
        }
        SlotMetaTile {
            entry_id: entry_id.clone(),
            tile_id,
            data: SlotMetaTileData { items: item_metas },
        }
    }

    fn interval(&self) -> ts::Interval {
        let last_time = self.state.last_time;
        // Add a bit to the end of the timeline to make it more visible
        let last_time = last_time + Timestamp(last_time.0 / 200);
        ts::Interval::new(ts::Timestamp(0), last_time.into())
    }
}

impl DataSource for StateDataSource {
    fn fetch_description(&self) -> DataSourceDescription {
        DataSourceDescription {
            source_locator: self.state.source_locator.clone(),
        }
    }

    fn fetch_info(&self) -> DataSourceInfo {
        DataSourceInfo {
            entry_info: self.info.clone(),
            interval: self.interval(),
            tile_set: TileSet::default(),
            field_schema: self.field_schema.clone(),
        }
    }

    fn fetch_summary_tile(&self, entry_id: &EntryID, tile_id: TileID, full: bool) -> SummaryTile {
        // Pick this number to be approximately the number of pixels we expect
        // the user to have on their screen. If this is a full tile, increase
        // 10x so that we get more resolution when zoomed in.
        let samples = if full { 10_000 } else { 1_000 };

        let step_utilization = self.generate_step_utilization(entry_id);

        let utilization = Self::compute_sample_utilization(&step_utilization, tile_id.0, samples);

        SummaryTile {
            entry_id: entry_id.clone(),
            tile_id,
            data: SummaryTileData { utilization },
        }
    }

    fn fetch_slot_tile(&self, entry_id: &EntryID, tile_id: TileID, full: bool) -> SlotTile {
        let entry = self.entry_map.get(entry_id).unwrap();
        match entry {
            EntryKind::Proc(proc_id) => {
                self.generate_proc_slot_tile(entry_id, *proc_id, tile_id, full)
            }
            EntryKind::Mem(mem_id) => self.generate_mem_slot_tile(entry_id, *mem_id, tile_id, full),
            EntryKind::Chan(chan_id) => {
                self.generate_chan_slot_tile(entry_id, *chan_id, tile_id, full)
            }
            _ => unreachable!(),
        }
    }

    fn fetch_slot_meta_tile(
        &self,
        entry_id: &EntryID,
        tile_id: TileID,
        full: bool,
    ) -> SlotMetaTile {
        let entry = self.entry_map.get(entry_id).unwrap();
        match entry {
            EntryKind::Proc(proc_id) => {
                self.generate_proc_slot_meta_tile(entry_id, *proc_id, tile_id, full)
            }
            EntryKind::Mem(mem_id) => {
                self.generate_mem_slot_meta_tile(entry_id, *mem_id, tile_id, full)
            }
            EntryKind::Chan(chan_id) => {
                self.generate_chan_slot_meta_tile(entry_id, *chan_id, tile_id, full)
            }
            _ => unreachable!(),
        }
    }
}
