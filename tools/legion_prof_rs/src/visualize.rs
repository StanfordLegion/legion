use std::cmp::max;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{create_dir, remove_dir_all, File};
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::state::{Chan, ChanID, ChanPoint, NodeID, MemID, Proc, ProcEntry, ProcID, ProcKind, ProcPoint, State, Timestamp, Color};

static INDEX_HTML_CONTENT: &[u8] = include_bytes!("../../legion_prof_files/index.html");
static TIMELINE_JS_CONTENT: &[u8] = include_bytes!("../../legion_prof_files/js/timeline.js");
static UTIL_JS_CONTENT: &[u8] = include_bytes!("../../legion_prof_files/js/util.js");

#[derive(Serialize, Copy, Clone)]
struct DataRecord<'a> {
    level: u32,
    level_ready: Option<u32>,
    ready: &'a str,
    start: &'a str,
    end: &'a str,
    color: &'a str,
    opacity: f64,
    title: &'a str,
    initiation: &'a str,
    #[serde(rename = "in")]
    in_: &'a str,
    out: &'a str,
    children: &'a str,
    parents: &'a str,
    prof_uid: u64,
}

#[derive(Serialize, Copy, Clone)]
struct OpRecord<'a> {
    op_id: u64,
    desc: &'a str,
    proc: Option<&'a str>,
    level: Option<u32>,
}

#[derive(Serialize, Copy, Clone)]
struct UtilizationRecord<'a> {
    time: &'a str,
    count: &'a str,
}

#[derive(Serialize, Clone)]
struct ProcessorRecord {
    full_text: String,
    text: String,
    tsv: PathBuf,
    levels: u32,
}

#[derive(Serialize, Copy, Clone)]
struct ScaleRecord<'a> {
    start: &'a str,
    end: &'a str,
    stats_levels: u64,
    max_level: u32,
}

impl Proc {
    fn emit_tsv_point(
        &self,
        f: &mut csv::Writer<File>,
        point: &ProcPoint,
        state: &State,
        ) -> io::Result<()> {
        let (base, time_range, waiters) = self.entry(point.entry);
        let name = match point.entry {
            ProcEntry::Task(op_id) => {
                let task = &self.tasks.get(&op_id).unwrap();
                let task_name = &state.task_kinds.get(&task.task_id).unwrap().name;
                let variant_name = &state
                    .variants
                    .get(&(task.task_id, task.variant_id))
                    .unwrap()
                    .name;
                match task_name {
                    Some(task_name) => format!("{} [{}] <{}>", task_name, variant_name, op_id.0),
                    None => variant_name.clone(),
                }
            }
            ProcEntry::MetaTask(op_id, variant_id, idx) => {
                let task = &self.meta_tasks.get(&(op_id, variant_id)).unwrap()[idx];
                state
                    .meta_variants
                    .get(&task.variant_id)
                    .unwrap()
                    .name
                    .clone()
            }
            ProcEntry::MapperCall(idx) => state
                .mapper_call_kinds
                .get(&self.mapper_calls[idx].kind)
                .unwrap()
                .name
                .clone(),
            ProcEntry::RuntimeCall(idx) => state
                .runtime_call_kinds
                .get(&self.runtime_calls[idx].kind)
                .unwrap()
                .name
                .clone(),
            ProcEntry::ProfTask(idx) => format!("ProfTask <{:?}>", self.prof_tasks[idx].op_id.0)
        };

        let color = match point.entry {
            ProcEntry::Task(op_id) => {
                let task = &self.tasks.get(&op_id).unwrap();
                state
                    .variants
                    .get(&(task.task_id, task.variant_id))
                    .unwrap()
                    .color
                    .unwrap()
            }
            ProcEntry::MetaTask(op_id, variant_id, idx) => {
                let task = &self.meta_tasks.get(&(op_id, variant_id)).unwrap()[idx];
                state
                    .meta_variants
                    .get(&task.variant_id)
                    .unwrap()
                    .color
                    .unwrap()
            }
            ProcEntry::MapperCall(idx) => state
                .mapper_call_kinds
                .get(&self.mapper_calls[idx].kind)
                .unwrap()
                .color
                .unwrap(),
            ProcEntry::RuntimeCall(idx) => state
                .runtime_call_kinds
                .get(&self.runtime_calls[idx].kind)
                .unwrap()
                .color
                .unwrap(),
            ProcEntry::ProfTask(idx) => {
                // proftask color is hardcoded to
                // self.color = '#FFC0CB'  # Pink
                // FIXME don't hardcode this here
                Color(0xFFC0CB)
            }
        };
        let color = format!("#{:06x}", color);

        let level = self.max_levels + 1 - base.level.unwrap();
        let level_ready = base.level_ready.map(|l| self.max_levels_ready + 1 - l);

        let default = DataRecord {
            level,
            level_ready,
            ready: "0",
            start: "0",
            end: "0",
            color: &color,
            opacity: 1.0,
            title: &name,
            initiation: "",
            in_: "",
            out: "",
            children: "",
            parents: "",
            prof_uid: base.prof_uid.0,
        };

        let mut start = time_range.start.unwrap();
        if !waiters.wait_intervals.is_empty() {
            for wait in &waiters.wait_intervals {
                f.serialize(DataRecord {
                    ready: &format!("{}", start),
                    start: &format!("{}", start),
                    end: &format!("{}", wait.start),
                    opacity: 1.0,
                    title: &name,
                    ..default
                })?;
                f.serialize(DataRecord {
                    title: &format!("{} (waiting)", &name),
                    ready: &format!("{}", wait.start),
                    start: &format!("{}", wait.start),
                    end: &format!("{}", wait.ready),
                    opacity: 0.15,
                    ..default
                })?;
                f.serialize(DataRecord {
                    title: &format!("{} (ready)", &name),
                    ready: &format!("{}", wait.ready),
                    start: &format!("{}", wait.ready),
                    end: &format!("{}", wait.end),
                    opacity: 0.45,
                    ..default
                })?;
                start = max(start, wait.end);
            }
            if start < time_range.stop.unwrap() {
                f.serialize(DataRecord {
                    ready: &format!("{}", start),
                    start: &format!("{}", start),
                    end: &format!("{}", time_range.stop.unwrap()),
                    opacity: 1.0,
                    title: &name,
                    ..default
                })?;
            }
        } else {
            f.serialize(DataRecord {
                ready: &format!("{}", time_range.ready.unwrap_or(start)),
                start: &format!("{}", start),
                end: &format!("{}", time_range.stop.unwrap()),
                ..default
            })?;
        }

        Ok(())
    }

    fn emit_tsv<P: AsRef<Path>>(&self, path: P, state: &State) -> io::Result<ProcessorRecord> {
        let mut filename = PathBuf::new();
        filename.push("tsv");
        filename.push(format!("Proc_0x{:x}.tsv", self.proc_id));
        let mut f = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_path(path.as_ref().join(&filename))?;

        for point in &self.time_points {
            if point.first {
                self.emit_tsv_point(&mut f, point, state)?;
            }
        }

        let level = max(self.max_levels, 1) + 1;

        Ok(ProcessorRecord {
            full_text: format!("{:?} Processor 0x{:x}", self.kind, self.proc_id),
            text: format!("{:?} Proc {}", self.kind, self.proc_in_node),
            tsv: filename,
            levels: level,
        })
    }
}

impl State {
    fn group_node_proc_kind_timepoints(
        &self,
    ) -> (
        BTreeMap<(Option<NodeID>, ProcKind), Vec<(ProcID, &Vec<ProcPoint>)>>,
        BTreeMap<(Option<NodeID>, ProcKind), u64>,
    ) {
        let mut timepoint = BTreeMap::new();
        let mut proc_count = BTreeMap::new();

        for proc in self.procs.values() {
            if !proc.time_points.is_empty() {
                let nodes = vec![None, Some(proc.node_id)];
                for node in nodes {
                    let group = (node, proc.kind);
                    timepoint
                        .entry(group)
                        .or_insert_with(|| Vec::new())
                        .push((proc.proc_id, &proc.util_time_points));
                    *proc_count.entry(group).or_insert(0) += 1;
                }
            }
        }

        (timepoint, proc_count)
    }

    fn group_node_chan_kind_timepoints(&self) ->
        BTreeMap<Option<NodeID>, Vec<(ChanID, &Vec<ChanPoint>)>>
    {
        let mut result = BTreeMap::new();

        for (chan_id, chan) in &self.channels {
            if !chan.time_points.is_empty() {
                if chan_id.node_id().is_some() {
                    let mut nodes = vec![
                        None,
                        chan_id.src.map(|src| src.node_id()),
                        chan_id.dst.map(|dst| dst.node_id())
                    ];
                    &nodes.dedup();
                    for node in nodes {
                        result
                            .entry(node)
                            .or_insert_with(|| Vec::new())
                            .push((*chan_id, &chan.time_points))
                    }
                }
            }
        }

        result
    }

    fn convert_proc_points_to_utilization(&self, points: &Vec<ProcPoint>, proc_id: ProcID) -> Vec<ProcPoint> {
        let mut utilization = Vec::new();
        let mut count = 0;
        for point in points {
            if point.first {
                count += 1;
                if count == 1 {
                    utilization.push(*point);
                }
            } else {
                count -= 1;
                if count == 0 {
                    utilization.push(*point);
                }
            }
        }
        utilization
    }

    // This is character-for-character the same function as convert_proc_points_to_utilization(
    // TODO look into making a TimePoint trait so that we don't have to repeat method definitions
    fn convert_chan_points_to_utilization(&self, points: &Vec<ChanPoint>, chan_id: ChanID) -> Vec<ChanPoint> {
        let mut utilization = Vec::new();
        let mut count = 0;

        if chan_id.node_id().is_some() {
            for point in points {
                if point.first {
                    count += 1;
                    if count == 1 {
                        utilization.push(*point);
                    }
                } else {
                    count -= 1;
                    if count == 0 {
                        utilization.push(*point);
                    }
                }
            }
        }
        utilization
    }

    fn calculate_proc_utilization_data(
        &self,
        points: Vec<ProcPoint>,
        owners: BTreeSet<ProcID>,
        max_count: u64,
    ) -> Vec<(Timestamp, f64)> {
        // we assume that the timepoints are sorted before this step

        // loop through all the timepoints. Get the earliest. If it's first,
        // add to the count. if it's second, decrement the count. Store the
        // (time, count) pair.

        assert!(owners.len() > 0);

        let max_count = max_count as f64;

        let mut utilization = Vec::new();
        let mut count = 0;
        let mut last_time = None;

        for point in points {
            if point.first {
                count += 1;
            } else {
                count -= 1;
            }

            if last_time.map_or(false, |time| time == point.time) {
                *utilization.last_mut().unwrap() = (point.time, count as f64 / max_count);
            } else {
                utilization.push((point.time, count as f64 / max_count));
            }
            last_time = Some(point.time);
        }

        utilization
    }

    fn calculate_chan_utilization_data(
        &self,
        points: Vec<ChanPoint>,
        owners: BTreeSet<ChanID>
    ) -> Vec<(Timestamp, f64)> {
        // we assume that the timepoints are sorted before this step

        // loop through all the timepoints. Get the earliest. If it's first,
        // add to the count. if it's second, decrement the count. Store the
        // (time, count) pair.

        assert!(owners.len() > 0);

        let max_count = owners.len();

        let mut utilization = Vec::new();
        let mut last_time = None;
        let mut count = 0;

        for point in &points {
            if point.first {
                count += 1;
            } else {
                count -= 1;
            }

            let count = count as f64;
            let max_count = max_count as f64;
            if last_time.map_or(false, |time| time == point.time) {
                if count > 0.0 {
                    *utilization.last_mut().unwrap() = (point.time, 1.0);
                } else {
                    *utilization.last_mut().unwrap() = (point.time, count / max_count);
                }
            } else {
                if count > 0.0 {
                    utilization.push((point.time, 1.0));
                } else {
                    utilization.push((point.time, count / max_count));
                }
            }
            last_time = Some(point.time);
        }

        utilization
    }

    fn get_nodes(&self) -> BTreeSet<Option<NodeID>> {
        let mut nodes = BTreeSet::new();
        for proc in self.procs.values() {
            if !proc.time_points.is_empty() {
                nodes.insert(Some(proc.node_id));
            }
        }
        if nodes.len() > 1 {
            nodes.insert(None);
        }
        nodes
    }

    fn get_kinds(&self) -> BTreeSet<ProcKind> {
        let mut kinds = BTreeSet::new();
        for proc in self.procs.values() {
            if !proc.time_points.is_empty() {
                kinds.insert(proc.kind);
            }
        }
        kinds
    }

    fn emit_utilization_tsv<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let (timepoint_proc, proc_count) = self.group_node_proc_kind_timepoints();

        let nodes = self.get_nodes();
        let kinds = self.get_kinds();

        let mut stats = BTreeMap::new();

        for node in nodes {
            for kind in &kinds {
                let group = (node, *kind);
                let node_name = match node {
                    None => "all".to_owned(),
                    Some(node_id) => format!("{}", node_id.0),
                };
                let group_name = format!("{} ({:?})", &node_name, kind);
                if timepoint_proc.contains_key(&group) {
                    stats
                        .entry(node_name)
                        .or_insert_with(|| Vec::new())
                        .push(group_name);
                }
            }
        }

        {
            let mut filename = path.as_ref().join("json").join("utils.json");
            let mut file = File::create(filename)?;
            serde_json::to_writer(&file, &stats)?;
        }

        for (group, points) in timepoint_proc {
            let owners: BTreeSet<_> = points
                .iter()
                .filter(|(_, tp)| !tp.is_empty())
                .map(|(proc_id, tp)| *proc_id)
                .collect();

            let utilization = if owners.is_empty() {
                Vec::new()
            } else {
                let count = *proc_count.get(&group).unwrap_or(&(owners.len() as u64));
                let mut utilizations: Vec<_> = points
                    .iter()
                    .filter(|(_, tp)| !tp.is_empty())
                    .flat_map(|(proc_id, tp)| self.convert_proc_points_to_utilization(tp, *proc_id))
                    .collect();
                utilizations.sort_by_key(|point| point.time_key());
                self.calculate_proc_utilization_data(utilizations, owners, count)
            };

            let (node, kind) = group;
            let node_name = match node {
                None => "all".to_owned(),
                Some(node_id) => format!("{}", node_id.0),
            };
            let group_name = format!("{} ({:?})", &node_name, kind);
            let filename = path
                .as_ref()
                .join("tsv")
                .join(format!("{}_util.tsv", group_name));
            let mut f = csv::WriterBuilder::new()
                .delimiter(b'\t')
                .from_path(filename)?;
            f.serialize(UtilizationRecord {
                time: "0.000",
                count: "0.00",
            })?;
            for (time, count) in utilization {
                f.serialize(UtilizationRecord {
                    time: &format!("{}", time),
                    count: &format!("{:.2}", count),
                })?;
            }
        }

        let timepoint_chan = self.group_node_chan_kind_timepoints();
        for (node_id, points) in timepoint_chan {
            let owners: BTreeSet<_> = points
                .iter()
                .filter(|(_, tp)| !tp.is_empty())
                .map(|(chan_id, tp)| *chan_id)
                .collect();

            let utilization = if owners.is_empty() {
                Vec::new()
            } else {
                let mut utilizations: Vec<_> = points
                    .iter()
                    .filter(|(_, tp)| !tp.is_empty())
                    .flat_map(|(chan_id, tp)| self.convert_chan_points_to_utilization(tp, *chan_id))
                    .collect();
                utilizations.sort_by_key(|point| point.time_key());
                self.calculate_chan_utilization_data(utilizations, owners)
            };

            let group_name = if let Some(node_id) = node_id {
                format!("{} (Channel)", node_id.0)
            } else { "all (Channel)".to_owned() };

            let filename = path
                .as_ref()
                .join("tsv")
                .join(format!("{}_util.tsv", group_name));
            let mut f = csv::WriterBuilder::new()
                .delimiter(b'\t')
                .from_path(filename)?;
            f.serialize(UtilizationRecord {
                time: "0.000",
                count: "0.00",
            })?;
            for (time, count) in utilization {
                f.serialize(UtilizationRecord {
                    time: &format!("{}", time),
                    count: &format!("{:.2}", count),
                })?;
            }
        }

        Ok(())
    }
}

fn create_unique_dir<P: AsRef<Path>>(path: P, force: bool) -> io::Result<PathBuf> {
    let mut path = path.as_ref().to_owned();
    if force {
        println!("Removing previous contents of {:?}", &path);
        let _ = remove_dir_all(&path); // ignore failure, we'll catch it on create
        create_dir(&path)?;
    } else {
        if create_dir(&path).is_err() {
            let mut i = 1;
            let retry_limit = 100;
            loop {
                let mut f = path.file_name().unwrap().to_owned();
                f.push(format!(".{}", i));
                let p = path.with_file_name(f);
                let r = create_dir(&p);
                if r.is_ok() {
                    path = p.as_path().to_owned();
                    break;
                } else if i >= retry_limit {
                    // tried too many times, assume this is a permanent failure
                    r?;
                }
                i += 1;
            }
        }
    }
    Ok(path)
}

fn write_file<P: AsRef<Path>>(path: P, content: &[u8]) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(content)?;
    Ok(())
}

pub fn emit_interactive_visualization<P: AsRef<Path>>(
    state: &State,
    path: P,
    force: bool,
    show_procs: bool,
    show_channels: bool,
    show_instances: bool,
) -> io::Result<()> {
    let path = create_unique_dir(path, force)?;
    println!(
        "Generating interactive visualization files in directory {:?}",
        &path
    );

    // generate static files
    write_file(path.join("index.html"), INDEX_HTML_CONTENT)?;
    create_dir(path.join("js"))?;
    write_file(path.join("js").join("timeline.js"), TIMELINE_JS_CONTENT)?;
    write_file(path.join("js").join("util.js"), UTIL_JS_CONTENT)?;

    // create subdirectories
    create_dir(path.join("tsv"))?;
    create_dir(path.join("json"))?;

    // generate tsv data
    let mut base_level = 0;
    let mut proc_records = BTreeMap::new();

    for proc in state.procs.values() {
        if !proc.is_empty() {
            let record = proc.emit_tsv(&path, state)?;
            base_level += record.levels;
            proc_records.insert(proc.proc_id, record);
        }
    }

    state.emit_utilization_tsv(&path)?;

    {
        let mut filename = path.join("legion_prof_processor.tsv");
        let mut file = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_path(filename)?;
        for record in proc_records.values() {
            file.serialize(record)?;
        }
    }

    {
        let mut filename = path.join("legion_prof_ops.tsv");
        let mut file = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_path(filename)?;
        // FIXME: Generate other op types
        for (proc_id, proc_record) in &proc_records {
            let proc = state.procs.get(&proc_id).unwrap();
            for task in proc.tasks.values() {
                let task_name = &state.task_kinds.get(&task.task_id).unwrap().name;
                let variant_name = &state
                    .variants
                    .get(&(task.task_id, task.variant_id))
                    .unwrap()
                    .name;
                let name = match task_name {
                    Some(task_name) => format!("{} [{}]", task_name, variant_name),
                    None => variant_name.clone(),
                };

                file.serialize(OpRecord {
                    op_id: task.op_id.0,
                    desc: &format!("{} <{}>", name, task.op_id.0),
                    proc: Some(&proc_record.full_text),
                    level: task.base.level.map(|x| x + 1),
                })?;
            }
        }
    }

    {
        let stats_levels = 4;
        let scale_data = ScaleRecord {
            start: "0",
            end: &format!("{:.3}", state.last_time.0 as f64 * 1.01 / 1000.),
            stats_levels: stats_levels,
            max_level: base_level + 1,
        };

        let mut filename = path.join("json").join("scale.json");
        let mut file = File::create(filename)?;
        serde_json::to_writer(&file, &scale_data)?;
    }

    Ok(())
}
