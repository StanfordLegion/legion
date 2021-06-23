use std::cmp::max;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{create_dir, remove_dir_all, File};
use std::io;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};

use serde::{Serialize, Serializer};

use rayon::prelude::*;

use crate::state::{
    Chan, ChanEntry, ChanID, ChanPoint, Color, CopyInfo, MemID, MemKind, MemPoint, NodeID, Proc,
    ProcEntry, ProcID, ProcKind, ProcPoint, State, TimePoint, Timestamp,
};

static INDEX_HTML_CONTENT: &[u8] = include_bytes!("../../legion_prof_files/index.html");
static TIMELINE_JS_CONTENT: &[u8] = include_bytes!("../../legion_prof_files/js/timeline.js");
static UTIL_JS_CONTENT: &[u8] = include_bytes!("../../legion_prof_files/js/util.js");

impl Serialize for Timestamp {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut buf = [0u8; 64];
        let mut cursor = Cursor::new(&mut buf[..]);
        write!(cursor, "{}", self).unwrap();
        let len = cursor.position() as usize;
        serializer.serialize_bytes(&buf[..len])
    }
}

#[derive(Copy, Clone)]
struct Count(f64);

impl Serialize for Count {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut buf = [0u8; 64];
        let mut cursor = Cursor::new(&mut buf[..]);
        write!(cursor, "{:.2}", self.0).unwrap();
        let len = cursor.position() as usize;
        serializer.serialize_bytes(&buf[..len])
    }
}

#[derive(Serialize, Copy, Clone)]
struct DataRecord<'a> {
    level: u32,
    level_ready: Option<u32>,
    ready: Option<Timestamp>,
    start: Timestamp,
    end: Timestamp,
    color: &'a str,
    opacity: f64,
    title: &'a str,
    initiation: Option<u64>,
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
struct UtilizationRecord {
    time: Timestamp,
    count: Count,
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
                    Some(task_name) => {
                        if task_name != variant_name {
                            format!("{} [{}] <{}>", task_name, variant_name, op_id.0)
                        } else {
                            format!("{} <{}>", task_name, op_id.0)
                        }
                    }
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
            ProcEntry::MapperCall(idx) => {
                let mapper_call = &self.mapper_calls[idx];
                if mapper_call.deps.op_id.0 > 0 {
                    format!(
                        "Mapper Call {} for {}",
                        state.mapper_call_kinds.get(&mapper_call.kind).unwrap().name,
                        mapper_call.deps.op_id.0
                    )
                } else {
                    format!(
                        "Mapper Call {}",
                        state.mapper_call_kinds.get(&mapper_call.kind).unwrap().name
                    )
                }
            }
            ProcEntry::RuntimeCall(idx) => state
                .runtime_call_kinds
                .get(&self.runtime_calls[idx].kind)
                .unwrap()
                .name
                .clone(),
            ProcEntry::ProfTask(idx) => format!("ProfTask <{:?}>", self.prof_tasks[idx].op_id.0),
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
            ProcEntry::ProfTask(_) => {
                // proftask color is hardcoded to
                // self.color = '#FFC0CB'  # Pink
                // FIXME don't hardcode this here
                Color(0xFFC0CB)
            }
        };
        let color = format!("#{:06x}", color);

        let initiation = match point.entry {
            ProcEntry::Task(_) => None,
            ProcEntry::MetaTask(op_id, variant_id, idx) => {
                let task = &self.meta_tasks.get(&(op_id, variant_id)).unwrap()[idx];
                Some(task.deps.op_id.0)
            }
            ProcEntry::MapperCall(idx) => {
                let dep = self.mapper_calls[idx].deps.op_id.0;
                if dep > 0 {
                    Some(dep)
                } else {
                    None
                }
            }
            ProcEntry::RuntimeCall(_) => None,
            ProcEntry::ProfTask(_) => None,
        };

        let level = self.max_levels + 1 - base.level.unwrap();
        let level_ready = base.level_ready.map(|l| self.max_levels_ready + 1 - l);

        let default = DataRecord {
            level,
            level_ready,
            ready: None,
            start: Timestamp(0),
            end: Timestamp(0),
            color: &color,
            opacity: 1.0,
            title: &name,
            initiation,
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
                    ready: Some(start),
                    start: start,
                    end: wait.start,
                    opacity: 1.0,
                    title: &name,
                    ..default
                })?;
                f.serialize(DataRecord {
                    title: &format!("{} (waiting)", &name),
                    ready: Some(wait.start),
                    start: wait.start,
                    end: wait.ready,
                    opacity: 0.15,
                    ..default
                })?;
                f.serialize(DataRecord {
                    title: &format!("{} (ready)", &name),
                    ready: Some(wait.ready),
                    start: wait.ready,
                    end: wait.end,
                    opacity: 0.45,
                    ..default
                })?;
                start = max(start, wait.end);
            }
            if start < time_range.stop.unwrap() {
                f.serialize(DataRecord {
                    ready: Some(start),
                    start: start,
                    end: time_range.stop.unwrap(),
                    opacity: 1.0,
                    title: &name,
                    ..default
                })?;
            }
        } else {
            f.serialize(DataRecord {
                ready: Some(time_range.ready.unwrap_or(start)),
                start: start,
                end: time_range.stop.unwrap(),
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

        let level = max(self.max_levels, 1);

        Ok(ProcessorRecord {
            full_text: format!("{:?} Processor 0x{:x}", self.kind, self.proc_id),
            text: format!("{:?} Proc {}", self.kind, self.proc_id.proc_in_node()),
            tsv: filename,
            levels: level,
        })
    }
}

struct CopySize(u64);

impl fmt::Display for CopySize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= (1024 * 1024 * 1024) {
            // GBs
            write!(f, "{:.3} GiB", self.0 as f64 / (1024.0 * 1024.0 * 1024.0))
        } else if self.0 >= (1024 * 1024) {
            // MBs
            write!(f, "{:.3} MiB", self.0 as f64 / (1024.0 * 1024.0))
        } else if self.0 >= 1024 {
            // KBs
            write!(f, "{:.3} KiB", self.0 as f64 / 1024.0)
        } else {
            // Bytes
            write!(f, "{} B", self.0)
        }
    }
}

#[derive(Debug)]
pub struct CopyInfoVec<'a>(pub &'a Vec<CopyInfo>);

impl fmt::Display for CopyInfoVec<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, elt) in self.0.iter().enumerate() {
            write!(f, "$req[{}]: {}", i, elt)?;
        }
        Ok(())
    }
}

impl Chan {
    fn emit_tsv_point(
        &self,
        f: &mut csv::Writer<File>,
        point: &ChanPoint,
        state: &State,
    ) -> io::Result<()> {
        let (base, time_range, waiters) = self.entry(point.entry);
        let name = match point.entry {
            ChanEntry::Copy(idx) => {
                let copy = &self.copies[idx];
                let nreqs = copy.copy_info.len();
                if nreqs > 0 {
                    format!(
                        "size={}, num reqs={}, {}",
                        CopySize(copy.size),
                        nreqs,
                        CopyInfoVec(&copy.copy_info)
                    )
                } else {
                    format!("size={}, num reqs={}", CopySize(copy.size), nreqs)
                }
            }
            ChanEntry::Fill(_) => {
                format!("Fill")
            }
            ChanEntry::DepPart(idx) => {
                format!("{}", self.depparts[idx].part_op)
            }
        };

        let initiation = match point.entry {
            ChanEntry::Copy(idx) => self.copies[idx].deps.op_id,
            ChanEntry::Fill(idx) => self.fills[idx].deps.op_id,
            ChanEntry::DepPart(idx) => self.depparts[idx].deps.op_id,
        };

        let color = state.find_task(initiation).map_or_else(
            || {
                state.find_op(initiation).map_or(Color(0x000000), |op| {
                    op.kind.map_or(Color(0x000000), |kind| {
                        state.op_kinds.get(&kind).unwrap().color.unwrap()
                    })
                })
            },
            |task| {
                state
                    .variants
                    .get(&(task.task_id, task.variant_id))
                    .unwrap()
                    .color
                    .unwrap()
            },
        );
        let color = format!("#{:06x}", color);

        let level = max(self.max_levels + 1, 4) - base.level.unwrap();

        f.serialize(DataRecord {
            level,
            level_ready: None,
            ready: None,
            start: time_range.start.unwrap(),
            end: time_range.stop.unwrap(),
            color: &color,
            opacity: 1.0,
            title: &name,
            initiation: Some(initiation.0),
            in_: "",
            out: "",
            children: "",
            parents: "",
            prof_uid: base.prof_uid.0,
        })?;

        Ok(())
    }

    fn emit_tsv<P: AsRef<Path>>(&self, path: P, state: &State) -> io::Result<ProcessorRecord> {
        let mem_kind = |mem_id: MemID| state.mems.get(&mem_id).unwrap().kind;
        let slug = match (self.channel_id.src, self.channel_id.dst) {
            (Some(src), Some(dst)) => format!(
                "({}_Memory_0x{:x},_{}_Memory_0x{:x})",
                mem_kind(src),
                &src,
                mem_kind(dst),
                &dst
            ),
            (None, Some(dst)) => format!("{}_Memory_0x{:x}", mem_kind(dst), dst),
            (None, None) => format!("None"),
            _ => unreachable!(),
        };

        let long_name = match (self.channel_id.src, self.channel_id.dst) {
            (Some(src), Some(dst)) => format!(
                "{} Memory 0x{:x} to {} Memory 0x{:x} Channel",
                mem_kind(src),
                &src,
                mem_kind(dst),
                &dst
            ),
            (None, Some(dst)) => format!("Fill {} Memory 0x{:x} Channel", mem_kind(dst), dst),
            (None, None) => format!("Dependent Partition Channel"),
            _ => unreachable!(),
        };

        let short_name = match (self.channel_id.src, self.channel_id.dst) {
            (Some(src), Some(dst)) => format!(
                "[n{}] {} to [n{}] {}",
                src.node_id().0,
                mem_kind(src),
                dst.node_id().0,
                mem_kind(dst)
            ),
            (None, Some(dst)) => format!("[n{}] {}", dst.node_id().0, mem_kind(dst)),
            (None, None) => format!("Dependent Partition Channel"),
            _ => unreachable!(),
        };

        let mut filename = PathBuf::new();
        filename.push("tsv");
        filename.push(format!("{}.tsv", slug));
        let mut f = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_path(path.as_ref().join(&filename))?;

        for point in &self.time_points {
            if point.first {
                self.emit_tsv_point(&mut f, point, state)?;
            }
        }

        let level = max(self.max_levels, 4) - 1;

        Ok(ProcessorRecord {
            full_text: long_name,
            text: short_name,
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
            let nodes = vec![None, Some(proc.proc_id.node_id())];
            for node in nodes {
                let group = (node, proc.kind);
                proc_count.entry(group).and_modify(|i| *i += 1).or_insert(1);
                if !proc.is_empty() {
                    timepoint
                        .entry(group)
                        .or_insert_with(|| Vec::new())
                        .push((proc.proc_id, &proc.util_time_points));
                }
            }
        }

        (timepoint, proc_count)
    }

    fn group_node_mem_kind_timepoints(
        &self,
    ) -> BTreeMap<(Option<NodeID>, MemKind), Vec<(MemID, &Vec<MemPoint>)>> {
        let mut result = BTreeMap::new();
        for (mem_id, mem) in &self.mems {
            if !mem.time_points.is_empty() {
                let nodes = vec![None, Some(mem.mem_id.node_id())];
                for node in nodes {
                    let group = (node, mem.kind);
                    result
                        .entry(group)
                        .or_insert_with(|| Vec::new())
                        .push((*mem_id, &mem.time_points))
                }
            }
        }

        result
    }

    fn group_node_chan_kind_timepoints(
        &self,
    ) -> BTreeMap<Option<NodeID>, Vec<(ChanID, &Vec<ChanPoint>)>> {
        let mut result = BTreeMap::new();

        for (chan_id, chan) in &self.channels {
            if !chan.time_points.is_empty() {
                if chan_id.node_id().is_some() {
                    let mut nodes = vec![
                        None,
                        chan_id.src.map(|src| src.node_id()),
                        chan_id.dst.map(|dst| dst.node_id()),
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

    fn convert_points_to_utilization<Entry, Secondary>(
        &self,
        points: &Vec<TimePoint<Entry, Secondary>>,
        utilization: &mut Vec<TimePoint<Entry, Secondary>>,
    ) where
        Entry: Copy,
        Secondary: Copy,
    {
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

        let mut utilization = Vec::new();
        let mut last_time = None;
        let max_count = max_count as f64;
        let mut count = 0;

        for point in points {
            if point.first {
                count += 1;
            } else {
                count -= 1;
            }

            let ratio = count as f64 / max_count;

            if last_time.map_or(false, |time| time == point.time) {
                *utilization.last_mut().unwrap() = (point.time, ratio);
            } else {
                utilization.push((point.time, ratio));
            }
            last_time = Some(point.time);
        }

        utilization
    }

    fn calculate_mem_utilization_data(
        &self,
        points: Vec<&MemPoint>,
        owners: BTreeSet<MemID>,
    ) -> Vec<(Timestamp, f64)> {
        assert!(owners.len() > 0);

        let mut result = Vec::new();

        let mut max_count = 0;
        for mem_id in owners {
            let mem = self.mems.get(&mem_id).unwrap();
            max_count += mem.capacity;
        }

        let max_count = max_count as f64;
        let mut count = 0;
        let mut last_time = None;

        for point in points {
            let inst = self.instances.get(&point.entry).unwrap();
            if point.first {
                count += inst.size.unwrap();
            } else {
                count -= inst.size.unwrap();
            }

            let ratio = count as f64 / max_count;
            if last_time.map_or(false, |time| time == point.time) {
                *result.last_mut().unwrap() = (point.time, ratio);
            } else {
                result.push((point.time, ratio));
            }
            last_time = Some(point.time);
        }

        result
    }

    fn calculate_chan_utilization_data(
        &self,
        points: Vec<ChanPoint>,
        owners: BTreeSet<ChanID>,
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

    fn emit_utilization_tsv_proc<P: AsRef<Path>>(
        &self,
        path: P,
        group: (Option<NodeID>, ProcKind),
        points: Vec<(ProcID, &Vec<ProcPoint>)>,
        proc_count: &BTreeMap<(Option<NodeID>, ProcKind), u64>,
    ) -> io::Result<()> {
        let owners: BTreeSet<_> = points
            .iter()
            .filter(|(_, tp)| !tp.is_empty())
            .map(|(proc_id, _)| *proc_id)
            .collect();

        let utilization = if owners.is_empty() {
            Vec::new()
        } else {
            let count = *proc_count.get(&group).unwrap_or(&(owners.len() as u64));
            let mut utilizations = Vec::new();
            for (_, tp) in points {
                if !tp.is_empty() {
                    self.convert_points_to_utilization(tp, &mut utilizations);
                }
            }
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
            time: Timestamp(0),
            count: Count(0.0),
        })?;
        for (time, count) in utilization {
            f.serialize(UtilizationRecord {
                time: time,
                count: Count(count),
            })?;
        }

        Ok(())
    }

    fn emit_utilization_tsv_mem<P: AsRef<Path>>(
        &self,
        path: P,
        group: (Option<NodeID>, MemKind),
        points: Vec<(MemID, &Vec<MemPoint>)>,
    ) -> io::Result<()> {
        let owners: BTreeSet<_> = points
            .iter()
            .filter(|(_, tp)| !tp.is_empty())
            .map(|(mem_id, _)| *mem_id)
            .collect();

        let utilization = if owners.is_empty() {
            Vec::new()
        } else {
            let mut utilizations: Vec<_> = points
                .iter()
                .filter(|(_, tp)| !tp.is_empty())
                .flat_map(|(_, tp)| *tp)
                .collect();
            utilizations.sort_by_key(|point| point.time_key());
            self.calculate_mem_utilization_data(utilizations, owners)
        };

        let (node, kind) = group;
        let node_name = match node {
            None => "all".to_owned(),
            Some(node_id) => format!("{}", node_id.0),
        };
        let group_name = format!("{} ({} Memory)", &node_name, kind);
        let filename = path
            .as_ref()
            .join("tsv")
            .join(format!("{}_util.tsv", group_name));
        let mut f = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_path(filename)?;
        f.serialize(UtilizationRecord {
            time: Timestamp(0),
            count: Count(0.0),
        })?;
        for (time, count) in utilization {
            f.serialize(UtilizationRecord {
                time: time,
                count: Count(count),
            })?;
        }

        Ok(())
    }

    fn emit_utilization_tsv_chan<P: AsRef<Path>>(
        &self,
        path: P,
        node_id: Option<NodeID>,
        points: Vec<(ChanID, &Vec<ChanPoint>)>,
    ) -> io::Result<()> {
        let owners: BTreeSet<_> = points
            .iter()
            .filter(|(_, tp)| !tp.is_empty())
            .map(|(chan_id, _)| *chan_id)
            .collect();

        let utilization = if owners.is_empty() {
            Vec::new()
        } else {
            let mut utilizations = Vec::new();
            for (_, tp) in points {
                if !tp.is_empty() {
                    self.convert_points_to_utilization(tp, &mut utilizations);
                }
            }
            utilizations.sort_by_key(|point| point.time_key());
            self.calculate_chan_utilization_data(utilizations, owners)
        };

        let group_name = if let Some(node_id) = node_id {
            format!("{} (Channel)", node_id.0)
        } else {
            "all (Channel)".to_owned()
        };

        let filename = path
            .as_ref()
            .join("tsv")
            .join(format!("{}_util.tsv", group_name));
        let mut f = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_path(filename)?;
        f.serialize(UtilizationRecord {
            time: Timestamp(0),
            count: Count(0.0),
        })?;
        for (time, count) in utilization {
            f.serialize(UtilizationRecord {
                time: time,
                count: Count(count),
            })?;
        }

        Ok(())
    }

    fn emit_utilization_tsv<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let (timepoint_proc, proc_count) = self.group_node_proc_kind_timepoints();
        let timepoint_mem = self.group_node_mem_kind_timepoints();
        let timepoint_chan = self.group_node_chan_kind_timepoints();

        let mut stats = BTreeMap::new();

        for group in timepoint_proc.keys() {
            let (node, kind) = group;
            let node_name = match node {
                None => "all".to_owned(),
                Some(node_id) => format!("{}", node_id.0),
            };
            let group_name = format!("{} ({:?})", &node_name, kind);
            stats
                .entry(node_name)
                .or_insert_with(|| Vec::new())
                .push(group_name);
        }

        for group in timepoint_mem.keys() {
            let (node, kind) = group;
            let node_name = match node {
                None => "all".to_owned(),
                Some(node_id) => format!("{}", node_id.0),
            };
            let group_name = format!("{} ({} Memory)", &node_name, kind);
            stats
                .entry(node_name)
                .or_insert_with(|| Vec::new())
                .push(group_name);
        }

        for node in timepoint_chan.keys() {
            let node_name = match node {
                None => "all".to_owned(),
                Some(node_id) => format!("{}", node_id.0),
            };
            let group_name = format!("{} (Channel)", &node_name);
            stats
                .entry(node_name)
                .or_insert_with(|| Vec::new())
                .push(group_name);
        }

        let path = path.as_ref();
        {
            let filename = path.join("json").join("utils.json");
            let file = File::create(filename)?;
            serde_json::to_writer(&file, &stats)?;
        }

        let mut result_proc = Ok(());
        let mut result_mem = Ok(());
        let mut result_chan = Ok(());
        rayon::scope(|s| {
            s.spawn(|_| {
                result_proc = timepoint_proc
                    .into_par_iter()
                    .map(|(group, points)| {
                        self.emit_utilization_tsv_proc(path, group, points, &proc_count)
                    })
                    .collect::<io::Result<_>>()
            });

            s.spawn(|_| {
                result_mem = timepoint_mem
                    .into_par_iter()
                    .map(|(group, points)| self.emit_utilization_tsv_mem(path, group, points))
                    .collect::<io::Result<_>>()
            });

            s.spawn(|_| {
                result_chan = timepoint_chan
                    .into_par_iter()
                    .map(|(node_id, points)| self.emit_utilization_tsv_chan(path, node_id, points))
                    .collect::<io::Result<_>>()
            });
        });
        result_proc?;
        result_mem?;
        result_chan?;

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
    let procs = state.procs.values().collect::<Vec<_>>();
    let proc_records: BTreeMap<_, _> = procs
        .par_iter()
        .filter(|proc| !proc.is_empty())
        .map(|proc| {
            proc.emit_tsv(&path, state)
                .map(|record| (proc.proc_id, record))
        })
        .collect::<io::Result<_>>()?;

    let mut base_level = 0;
    for record in proc_records.values() {
        base_level += record.levels;
    }

    let channels = state.channels.values().collect::<Vec<_>>();
    let chan_records: BTreeMap<_, _> = channels
        .par_iter()
        .filter(|chan| !chan.is_empty())
        .map(|chan| {
            chan.emit_tsv(&path, state)
                .map(|record| (chan.channel_id, record))
        })
        .collect::<io::Result<_>>()?;

    state.emit_utilization_tsv(&path)?;

    {
        let filename = path.join("legion_prof_processor.tsv");
        let mut file = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_path(filename)?;
        for record in proc_records.values() {
            file.serialize(record)?;
        }
        for record in chan_records.values() {
            file.serialize(record)?;
        }
    }

    {
        let filename = path.join("legion_prof_ops.tsv");
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

        let filename = path.join("json").join("scale.json");
        let file = File::create(filename)?;
        serde_json::to_writer(&file, &scale_data)?;
    }

    Ok(())
}
