use std::cmp::max;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{create_dir, remove_dir_all, File};
use std::io;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};

use serde::{Serialize, Serializer};

use rayon::prelude::*;

use crate::backend::common::{
    CopyInstInfoDumpInstVec, FillInstInfoDumpInstVec, MemGroup, ProcGroup, StatePostprocess,
};
use crate::state::{
    Chan, ChanEntry, ChanID, ChanPoint, Config, Container, ContainerEntry, DeviceKind, Mem, MemID,
    MemKind, MemPoint, MemProcAffinity, NodeID, OpID, OperationInstInfo, Proc, ProcEntryKind,
    ProcID, ProcKind, ProcPoint, ProfUID, SpyState, State, Timestamp,
};

use crate::conditional_assert;

static INDEX_HTML_CONTENT: &[u8] = include_bytes!("../../../legion_prof_files/index.html");
static TIMELINE_JS_CONTENT: &[u8] = include_bytes!("../../../legion_prof_files/js/timeline.js");
static UTIL_JS_CONTENT: &[u8] = include_bytes!("../../../legion_prof_files/js/util.js");

#[derive(Debug, Copy, Clone)]
struct TimestampFormat(Timestamp);

impl Serialize for TimestampFormat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut buf = [0u8; 64];
        let mut cursor = Cursor::new(&mut buf[..]);
        write!(cursor, "{}", self.0).unwrap();
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
struct DependencyRecord(u64, u64, u64);

#[derive(Serialize, Copy, Clone)]
struct CriticalPathRecord {
    tuple: Option<DependencyRecord>,
    obj: Option<DependencyRecord>,
}

#[derive(Serialize, Copy, Clone)]
struct DataRecord<'a> {
    level: u32,
    level_ready: Option<u32>,
    ready: Option<TimestampFormat>,
    start: TimestampFormat,
    end: TimestampFormat,
    color: &'a str,
    opacity: f64,
    title: &'a str,
    initiation: Option<OpID>,
    #[serde(rename = "in")]
    in_: &'a str,
    out: &'a str,
    children: &'a str,
    parents: &'a str,
    prof_uid: u64,
    op_id: Option<OpID>,
    instances: &'a str,
}

#[derive(Serialize, Copy, Clone)]
struct OpRecord<'a> {
    op_id: OpID,
    parent_id: Option<OpID>,
    desc: &'a str,
    proc: Option<&'a str>,
    level: Option<u32>,
    provenance: Option<&'a str>,
}

#[derive(Serialize, Copy, Clone)]
struct UtilizationRecord {
    time: TimestampFormat,
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
struct ScaleRecord {
    start: f64,
    end: f64,
    stats_levels: u64,
    max_level: u32,
}

fn prof_uid_record(prof_uid: ProfUID, state: &State) -> Option<DependencyRecord> {
    let proc_id = state.prof_uid_proc.get(&prof_uid)?;
    Some(DependencyRecord(
        proc_id.node_id().0,
        proc_id.proc_in_node(),
        prof_uid.0,
    ))
}

#[derive(Debug)]
pub struct OperationInstInfoDumpInstVec<'a>(pub &'a Vec<OperationInstInfo>, pub &'a State);

impl fmt::Display for OperationInstInfoDumpInstVec<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // remove duplications
        let mut insts_set = BTreeSet::new();
        for elt in self.0.iter() {
            if let Some(inst) = self.1.find_inst(elt.inst_uid) {
                insts_set.insert(inst);
            } else {
                conditional_assert!(
                    false,
                    Config::all_logs(),
                    "Operation can not find inst:0x{:x}",
                    elt.inst_uid.0
                );
            }
        }
        write!(f, "[")?;
        for (i, inst) in insts_set.iter().enumerate() {
            write!(
                f,
                "[\"0x{:x}\",{}]",
                inst.inst_id.unwrap().0,
                inst.base.prof_uid.0
            )?;
            if i < insts_set.len() - 1 {
                write!(f, ",")?;
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl Proc {
    fn emit_tsv_point(
        &self,
        f: &mut csv::Writer<File>,
        device: Option<DeviceKind>,
        point: &ProcPoint,
        state: &State,
        spy_state: &SpyState,
    ) -> io::Result<()> {
        let entry = self.entry(point.entry);
        let (op_id, initiation_op) = (entry.op_id, entry.initiation_op);
        let (base, time_range, waiters) = (&entry.base, &entry.time_range, &entry.waiters);
        let name = entry.name(state);

        let color = entry.color(state);
        let color = format!("#{:06x}", color);

        let initiation = match entry.kind {
            // FIXME: Elliott: special case on ProfTask to match legion_prof.py behavior
            ProcEntryKind::ProfTask => None,
            // And another special case, because for MapperCalls only, we set default to 0 to match with python
            ProcEntryKind::MapperCall(..) => Some(initiation_op.unwrap_or(OpID::ZERO)),
            _ => initiation_op,
        };

        let op_id = match entry.kind {
            // FIXME: Elliott: special case on ProfTask to match legion_prof.py behavior
            ProcEntryKind::ProfTask => Some(initiation_op.unwrap()),
            _ => op_id,
        };

        let render_op = |prof_uid: &ProfUID| prof_uid_record(*prof_uid, state);

        let deps = spy_state.spy_op_deps.get(&base.prof_uid);

        let mut in_ = String::new();
        let mut out = String::new();
        let mut parent = String::new();
        let mut children = String::new();
        if let Some(deps) = deps {
            let deps_in: Vec<_> = deps.in_.iter().filter_map(render_op).collect();
            let deps_out: Vec<_> = deps.out.iter().filter_map(render_op).collect();
            let deps_parent: Vec<_> = deps.parent.iter().filter_map(render_op).collect();
            let deps_children: Vec<_> = deps.children.iter().filter_map(render_op).collect();
            if !deps_in.is_empty() {
                in_ = serde_json::to_string(&deps_in)?;
            }
            if !deps_out.is_empty() {
                out = serde_json::to_string(&deps_out)?;
            }
            if !deps_parent.is_empty() {
                parent = serde_json::to_string(&deps_parent)?;
            }
            if !deps_children.is_empty() {
                children = serde_json::to_string(&deps_children)?;
            }
        }

        let level = self.max_levels(device) - base.level.unwrap();
        let level_ready = base.level_ready.map(|l| self.max_levels_ready(device) - l);

        let instances = {
            // ProfTask has no op_id
            if let Some(op_id) = entry.op_id {
                let task = state.find_op(op_id).unwrap();
                format!(
                    "{}",
                    OperationInstInfoDumpInstVec(&task.operation_inst_infos, state)
                )
            } else {
                "".to_owned()
            }
        };

        let default = DataRecord {
            level,
            level_ready,
            ready: None,
            start: TimestampFormat(Timestamp::ZERO),
            end: TimestampFormat(Timestamp::ZERO),
            color: &color,
            opacity: 1.0,
            title: &name,
            initiation,
            in_: "",
            out: "",
            children: "",
            parents: "",
            prof_uid: base.prof_uid.0,
            op_id,
            instances: &instances,
        };

        let mut start = time_range.start.unwrap();
        if !waiters.wait_intervals.is_empty() {
            for wait in &waiters.wait_intervals {
                f.serialize(DataRecord {
                    ready: Some(TimestampFormat(start)),
                    start: TimestampFormat(start),
                    end: TimestampFormat(wait.start),
                    opacity: 1.0,
                    title: &name,
                    // Somehow, these are coming through backwards...
                    in_: &out, //&in_,
                    out: &in_, //&out,
                    children: &children,
                    parents: &parent,
                    ..default
                })?;
                // Only write dependencies once
                in_ = String::new();
                out = String::new();
                parent = String::new();
                children = String::new();
                f.serialize(DataRecord {
                    title: &format!("{} (waiting)", &name),
                    ready: Some(TimestampFormat(wait.start)),
                    start: TimestampFormat(wait.start),
                    end: TimestampFormat(wait.ready),
                    opacity: 0.15,
                    ..default
                })?;
                f.serialize(DataRecord {
                    title: &format!("{} (ready)", &name),
                    ready: Some(TimestampFormat(wait.ready)),
                    start: TimestampFormat(wait.ready),
                    end: TimestampFormat(wait.end),
                    opacity: 0.45,
                    ..default
                })?;
                start = max(start, wait.end);
            }
            if start < time_range.stop.unwrap() {
                f.serialize(DataRecord {
                    ready: Some(TimestampFormat(start)),
                    start: TimestampFormat(start),
                    end: TimestampFormat(time_range.stop.unwrap()),
                    opacity: 1.0,
                    title: &name,
                    ..default
                })?;
            }
        } else {
            f.serialize(DataRecord {
                ready: Some(TimestampFormat(time_range.ready.unwrap_or(start))),
                start: TimestampFormat(start),
                end: TimestampFormat(time_range.stop.unwrap()),
                // Somehow, these are coming through backwards...
                in_: &out, //&in_,
                out: &in_, //&out,
                children: &children,
                parents: &parent,
                ..default
            })?;
        }

        Ok(())
    }

    fn emit_tsv<P: AsRef<Path>>(
        &self,
        device: Option<DeviceKind>,
        path: P,
        state: &State,
        spy_state: &SpyState,
    ) -> io::Result<ProcessorRecord> {
        let suffix = match device {
            Some(DeviceKind::Device) => " Device",
            Some(DeviceKind::Host) => " Host",
            None => "",
        };
        let file_suffix = match device {
            Some(DeviceKind::Device) => "_Device",
            Some(DeviceKind::Host) => "_Host",
            None => "",
        };

        let mut filename = PathBuf::new();
        filename.push("tsv");
        filename.push(format!("Proc_0x{:x}{}.tsv", self.proc_id, file_suffix));
        let mut f = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_path(path.as_ref().join(&filename))?;

        for point in self.time_points(device) {
            assert!(point.first);
            self.emit_tsv_point(&mut f, device, point, state, spy_state)?;
        }

        let level = max(self.max_levels(device), 1);

        Ok(ProcessorRecord {
            full_text: format!("{:?}{} Processor 0x{:x}", self.kind, suffix, self.proc_id),
            text: format!(
                "{:?}{} Proc {}",
                self.kind,
                suffix,
                self.proc_id.proc_in_node(),
            ),
            tsv: filename,
            levels: level,
        })
    }
}

#[derive(Debug)]
pub struct MemKindShort(pub MemKind);

impl fmt::Display for MemKindShort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            MemKind::NoMemKind => write!(f, "none"),
            MemKind::Global => write!(f, "glob"),
            MemKind::System => write!(f, "sys"),
            MemKind::Registered => write!(f, "reg"),
            MemKind::Socket => write!(f, "sock"),
            MemKind::ZeroCopy => write!(f, "zcpy"),
            MemKind::Framebuffer => write!(f, "fb"),
            MemKind::Disk => write!(f, "disk"),
            MemKind::HDF5 => write!(f, "hdf5"),
            MemKind::File => write!(f, "file"),
            MemKind::L3Cache => write!(f, "l3"),
            MemKind::L2Cache => write!(f, "l2"),
            MemKind::L1Cache => write!(f, "l1"),
            MemKind::GPUManaged => write!(f, "uvm"),
            MemKind::GPUDynamic => write!(f, "gpu-dyn"),
        }
    }
}

#[derive(Debug)]
pub struct MemShort<'a>(
    pub MemKind,
    pub Option<&'a Mem>,
    pub Option<&'a MemProcAffinity>,
    pub &'a State,
);

impl fmt::Display for MemShort<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (mem_kind, mem, affinity, state) = (self.0, self.1, self.2, self.3);

        match mem_kind {
            MemKind::NoMemKind | MemKind::Global => write!(f, "[all n]"),
            MemKind::System
            | MemKind::Registered
            | MemKind::Socket
            | MemKind::ZeroCopy
            | MemKind::Disk
            | MemKind::HDF5
            | MemKind::File
            | MemKind::L3Cache
            | MemKind::GPUManaged => {
                let mem = mem.unwrap();
                write!(
                    f,
                    "[n{}] {}",
                    mem.mem_id.node_id().0,
                    MemKindShort(mem_kind)
                )
            }
            MemKind::Framebuffer | MemKind::GPUDynamic => {
                let affinity = affinity.unwrap();
                let proc = state.procs.get(&affinity.best_aff_proc).unwrap();
                write!(
                    f,
                    "[n{}][gpu{}] {}",
                    proc.proc_id.node_id().0,
                    proc.proc_id.proc_in_node(),
                    MemKindShort(mem_kind)
                )
            }
            MemKind::L2Cache | MemKind::L1Cache => {
                let affinity = affinity.unwrap();
                let proc = state.procs.get(&affinity.best_aff_proc).unwrap();
                write!(
                    f,
                    "[n{}][cpu{}] {}",
                    proc.proc_id.node_id().0,
                    proc.proc_id.proc_in_node(),
                    MemKindShort(mem_kind)
                )
            }
        }
    }
}

impl Chan {
    fn emit_tsv_point(
        &self,
        f: &mut csv::Writer<File>,
        point: &ChanPoint,
        state: &State,
    ) -> io::Result<()> {
        let entry = self.entry(point.entry);
        let (base, time_range) = (entry.base(), entry.time_range());
        let name = entry.name(state);
        let ready_timestamp = match entry {
            ChanEntry::Copy(_) => time_range.ready,
            ChanEntry::Fill(_) => time_range.ready,
            ChanEntry::DepPart(_) => None,
        };

        let initiation = entry.initiation();

        let color = format!("#{:06x}", entry.color(state));

        let level = max(self.max_levels(None) + 1, 4) - base.level.unwrap() - 1;

        let instances = match entry {
            ChanEntry::Copy(copy) => {
                format!("{}", CopyInstInfoDumpInstVec(&copy.copy_inst_infos, state))
            }
            ChanEntry::Fill(fill) => {
                format!("{}", FillInstInfoDumpInstVec(&fill.fill_inst_infos, state))
            }
            ChanEntry::DepPart(_deppart) => "".to_owned(),
        };

        f.serialize(DataRecord {
            level,
            level_ready: None,
            ready: ready_timestamp.map(TimestampFormat),
            start: TimestampFormat(time_range.start.unwrap()),
            end: TimestampFormat(time_range.stop.unwrap()),
            color: &color,
            opacity: 1.0,
            title: &name,
            initiation,
            in_: "",
            out: "",
            children: "",
            parents: "",
            prof_uid: base.prof_uid.0,
            op_id: None,
            instances: &instances,
        })?;

        Ok(())
    }

    fn emit_tsv<P: AsRef<Path>>(&self, path: P, state: &State) -> io::Result<ProcessorRecord> {
        let mem_kind = |mem_id: MemID| {
            state
                .mems
                .get(&mem_id)
                .map_or(MemKind::NoMemKind, |mem| mem.kind)
        };
        let slug = match self.chan_id {
            ChanID::Copy { src, dst } => format!(
                "({}_Memory_0x{:x},_{}_Memory_0x{:x},_Copy)",
                mem_kind(src),
                &src,
                mem_kind(dst),
                &dst
            ),
            ChanID::Fill { dst } => format!("(None,_{}_Memory_0x{:x},_Fill)", mem_kind(dst), dst),
            ChanID::Gather { dst } => {
                format!("(None,_{}_Memory_0x{:x},_Gather)", mem_kind(dst), dst)
            }
            ChanID::Scatter { src } => {
                format!("(None,_{}_Memory_0x{:x},_Scatter)", mem_kind(src), src)
            }
            ChanID::DepPart { node_id } => format!("(Node{},_DepPart)", node_id.0),
        };

        let long_name = match self.chan_id {
            ChanID::Copy { src, dst } => format!(
                "{} Memory 0x{:x} to {} Memory 0x{:x} Channel",
                mem_kind(src),
                &src,
                mem_kind(dst),
                &dst
            ),
            ChanID::Fill { dst } => format!("Fill {} Memory 0x{:x} Channel", mem_kind(dst), dst),
            ChanID::Gather { dst } => {
                format!("Gather {} Memory 0x{:x} Channel", mem_kind(dst), dst)
            }
            ChanID::Scatter { src } => {
                format!("Scatter {} Memory 0x{:x} Channel", mem_kind(src), src)
            }
            ChanID::DepPart { node_id } => format!("Dependent Partition {}", node_id.0),
        };

        let short_name = match self.chan_id {
            ChanID::Copy { src, dst } => format!(
                "{} to {}",
                MemShort(
                    mem_kind(src),
                    state.mems.get(&src),
                    state.mem_proc_affinity.get(&src),
                    state
                ),
                MemShort(
                    mem_kind(dst),
                    state.mems.get(&dst),
                    state.mem_proc_affinity.get(&dst),
                    state
                )
            ),
            ChanID::Fill { dst } => format!(
                "Fill {}",
                MemShort(
                    mem_kind(dst),
                    state.mems.get(&dst),
                    state.mem_proc_affinity.get(&dst),
                    state
                )
            ),
            ChanID::Gather { dst } => format!(
                "Gather {}",
                MemShort(
                    mem_kind(dst),
                    state.mems.get(&dst),
                    state.mem_proc_affinity.get(&dst),
                    state
                )
            ),
            ChanID::Scatter { src } => format!(
                "Scatter {}",
                MemShort(
                    mem_kind(src),
                    state.mems.get(&src),
                    state.mem_proc_affinity.get(&src),
                    state
                )
            ),
            ChanID::DepPart { node_id } => format!("Dependent Partition {}", node_id.0),
        };

        let mut filename = PathBuf::new();
        filename.push("tsv");
        filename.push(format!("{}.tsv", slug));
        let mut f = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_path(path.as_ref().join(&filename))?;

        for point in self.time_points(None) {
            assert!(point.first);
            self.emit_tsv_point(&mut f, point, state)?;
        }

        let level = max(self.max_levels(None) + 1, 4) - 1;

        Ok(ProcessorRecord {
            full_text: long_name,
            text: short_name,
            tsv: filename,
            levels: level,
        })
    }
}

impl Mem {
    fn emit_tsv_point(
        &self,
        f: &mut csv::Writer<File>,
        point: &MemPoint,
        state: &State,
    ) -> io::Result<()> {
        let inst = self.insts.get(&point.entry).unwrap();
        let (base, time_range) = (&inst.base, &inst.time_range);
        let name = inst.name(state);

        let initiation = inst.op_id;

        let color = format!("#{:06x}", inst.color(state));

        let level = max(self.max_levels(None) + 1, 4) - base.level.unwrap();

        f.serialize(DataRecord {
            level,
            level_ready: None,
            ready: None,
            start: TimestampFormat(time_range.create.unwrap()),
            end: TimestampFormat(time_range.ready.unwrap()),
            color: &color,
            opacity: 0.45,
            title: &format!("{} (deferred)", &name),
            initiation,
            in_: "",
            out: "",
            children: "",
            parents: "",
            prof_uid: base.prof_uid.0,
            op_id: None,
            instances: "",
        })?;

        f.serialize(DataRecord {
            level,
            level_ready: None,
            ready: None,
            start: TimestampFormat(time_range.start.unwrap()),
            end: TimestampFormat(time_range.stop.unwrap()),
            color: &color,
            opacity: 1.0,
            title: &name,
            initiation,
            in_: "",
            out: "",
            children: "",
            parents: "",
            prof_uid: base.prof_uid.0,
            op_id: None,
            instances: "",
        })?;

        Ok(())
    }

    fn emit_tsv<P: AsRef<Path>>(&self, path: P, state: &State) -> io::Result<ProcessorRecord> {
        let mem_kind = |mem_id: MemID| state.mems.get(&mem_id).unwrap().kind;
        let slug = format!("Mem_0x{:x}", &self.mem_id);

        let long_name = format!("{} Memory 0x{:x}", mem_kind(self.mem_id), &self.mem_id);

        let short_name = format!(
            "{}",
            MemShort(
                self.kind,
                Some(self),
                state.mem_proc_affinity.get(&self.mem_id),
                state
            )
        );

        let mut filename = PathBuf::new();
        filename.push("tsv");
        filename.push(format!("{}.tsv", slug));
        let mut f = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_path(path.as_ref().join(&filename))?;

        for point in self.time_points(None) {
            assert!(point.first);
            self.emit_tsv_point(&mut f, point, state)?;
        }

        let level = max(self.max_levels(None) + 1, 4) - 1;

        Ok(ProcessorRecord {
            full_text: long_name,
            text: short_name,
            tsv: filename,
            levels: level,
        })
    }
}

impl State {
    fn emit_utilization_tsv_proc<P: AsRef<Path>>(
        &self,
        path: P,
        group: ProcGroup,
        points: Vec<(ProcID, &Vec<ProcPoint>)>,
        proc_count: &BTreeMap<ProcGroup, u64>,
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

        let ProcGroup(node, kind, device) = group;
        let node_name = match node {
            None => "all".to_owned(),
            Some(node_id) => format!("{}", node_id.0),
        };
        let suffix = match device {
            Some(DeviceKind::Device) => " Device",
            Some(DeviceKind::Host) => " Host",
            None => "",
        };
        let group_name = format!("{} ({:?}{})", &node_name, kind, suffix);
        let filename = path
            .as_ref()
            .join("tsv")
            .join(format!("{}_util.tsv", group_name));
        let mut f = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_path(filename)?;
        f.serialize(UtilizationRecord {
            time: TimestampFormat(Timestamp::ZERO),
            count: Count(0.0),
        })?;
        for (time, count) in utilization {
            f.serialize(UtilizationRecord {
                time: TimestampFormat(time),
                count: Count(count),
            })?;
        }

        Ok(())
    }

    fn emit_utilization_tsv_mem<P: AsRef<Path>>(
        &self,
        path: P,
        group: MemGroup,
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

        let MemGroup(node, kind) = group;
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
            time: TimestampFormat(Timestamp::ZERO),
            count: Count(0.0),
        })?;
        for (time, count) in utilization {
            f.serialize(UtilizationRecord {
                time: TimestampFormat(time),
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
            time: TimestampFormat(Timestamp::ZERO),
            count: Count(0.0),
        })?;
        for (time, count) in utilization {
            f.serialize(UtilizationRecord {
                time: TimestampFormat(time),
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

        let multinode = self.has_multiple_nodes();
        for group in timepoint_proc.keys() {
            let ProcGroup(node, kind, device) = group;
            if node.is_some() || multinode {
                let node_name = match node {
                    None => "all".to_owned(),
                    Some(node_id) => format!("{}", node_id.0),
                };
                let suffix = match device {
                    Some(DeviceKind::Device) => " Device",
                    Some(DeviceKind::Host) => " Host",
                    None => "",
                };
                let group_name = format!("{} ({:?}{})", &node_name, kind, suffix);
                stats
                    .entry(node_name)
                    .or_insert_with(Vec::new)
                    .push(group_name);
            }
        }

        for group in timepoint_mem.keys() {
            let MemGroup(node, kind) = group;
            if node.is_some() || multinode {
                let node_name = match node {
                    None => "all".to_owned(),
                    Some(node_id) => format!("{}", node_id.0),
                };
                let group_name = format!("{} ({} Memory)", &node_name, kind);
                stats
                    .entry(node_name)
                    .or_insert_with(Vec::new)
                    .push(group_name);
            }
        }

        for node in timepoint_chan.keys() {
            if node.is_some() || multinode {
                let node_name = match node {
                    None => "all".to_owned(),
                    Some(node_id) => format!("{}", node_id.0),
                };
                let group_name = format!("{} (Channel)", &node_name);
                stats
                    .entry(node_name)
                    .or_insert_with(Vec::new)
                    .push(group_name);
            }
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
    } else if create_dir(&path).is_err() {
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
    Ok(path)
}

fn write_file<P: AsRef<Path>>(path: P, content: &[u8]) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(content)?;
    Ok(())
}

pub fn emit_interactive_visualization<P: AsRef<Path>>(
    state: &State,
    spy_state: &SpyState,
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
        .filter(|proc| !proc.is_empty() && proc.is_visible())
        .flat_map(|proc| match proc.kind.unwrap() {
            ProcKind::GPU => vec![
                (proc, Some(DeviceKind::Device)),
                (proc, Some(DeviceKind::Host)),
            ],
            _ => vec![(proc, None)],
        })
        .map(|(proc, device)| {
            proc.emit_tsv(device, &path, state, spy_state)
                .map(|record| ((proc.proc_id, device), record))
        })
        .collect::<io::Result<_>>()?;

    let mut base_level = 0;
    for record in proc_records.values() {
        base_level += record.levels + 1;
    }

    let chans = state.chans.values().collect::<Vec<_>>();
    let chan_records: BTreeMap<_, _> = chans
        .par_iter()
        .filter(|chan| !chan.is_empty() && chan.is_visible())
        .map(|chan| {
            chan.emit_tsv(&path, state)
                .map(|record| (chan.chan_id, record))
        })
        .collect::<io::Result<_>>()?;

    for record in chan_records.values() {
        base_level += record.levels + 1;
    }

    let mems = state.mems.values().collect::<Vec<_>>();
    let mem_records: BTreeMap<_, _> = mems
        .par_iter()
        .filter(|mem| !mem.is_empty() && mem.is_visible())
        .map(|mem| {
            mem.emit_tsv(&path, state)
                .map(|record| (mem.mem_id, record))
        })
        .collect::<io::Result<_>>()?;

    for record in mem_records.values() {
        base_level += record.levels + 1;
    }

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
        for record in mem_records.values() {
            file.serialize(record)?;
        }
    }

    {
        let filename = path.join("legion_prof_ops.tsv");
        let mut file = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_path(filename)?;
        for (op_id, op) in &state.operations {
            let parent_id = op.parent_id;
            let provenance = op.provenance.and_then(|pid| state.find_provenance(pid));
            if let Some(proc_id) = state.tasks.get(op_id) {
                let proc = state.procs.get(proc_id).unwrap();
                let proc_full_text = format!("{:?} Processor 0x{:x}", proc.kind, proc.proc_id);
                let task = proc.find_task(*op_id).unwrap();
                let (task_id, variant_id) = match task.kind {
                    ProcEntryKind::Task(task_id, variant_id) => (task_id, variant_id),
                    _ => unreachable!(),
                };
                let task_name = &state.task_kinds.get(&task_id).unwrap().name;
                let variant_name = &state.variants.get(&(task_id, variant_id)).unwrap().name;
                let desc = match task_name {
                    Some(task_name) => {
                        if task_name == variant_name {
                            format!("{} <{}>", task_name, op_id.0)
                        } else {
                            format!("{} [{}] <{}>", task_name, variant_name, op_id.0)
                        }
                    }
                    None => format!("{} <{}>", variant_name, op_id.0),
                };

                file.serialize(OpRecord {
                    op_id: *op_id,
                    parent_id,
                    desc: &desc,
                    proc: Some(&proc_full_text),
                    level: task.base.level.map(|x| x + 2),
                    provenance,
                })?;
            } else if let Some(task) = state.multi_tasks.get(op_id) {
                let task_name = state
                    .task_kinds
                    .get(&task.task_id)
                    .unwrap()
                    .name
                    .as_ref()
                    .unwrap();

                file.serialize(OpRecord {
                    op_id: *op_id,
                    parent_id,
                    desc: &format!("{} <{}>", task_name, op_id.0),
                    proc: None,
                    level: None,
                    provenance,
                })?;
            } else {
                let desc = op.kind.and_then(|k| state.op_kinds.get(&k)).map_or_else(
                    || format!("Operation <{}>", op_id.0),
                    |k| format!("{} Operation <{}>", k.name, op_id.0),
                );

                file.serialize(OpRecord {
                    op_id: *op_id,
                    parent_id,
                    desc: &desc,
                    proc: None,
                    level: None,
                    provenance,
                })?;
            }
        }
    }

    {
        let stats_levels = 4;
        let scale_data = ScaleRecord {
            start: 0.0,
            end: (state.last_time.to_ns() as f64 / 100. * 1.01).ceil() / 10.,
            stats_levels,
            max_level: base_level + 1,
        };

        let filename = path.join("json").join("scale.json");
        let file = File::create(filename)?;
        serde_json::to_writer(&file, &scale_data)?;
    }

    {
        let filename = path.join("json").join("critical_path.json");
        let file = File::create(filename)?;
        let render_op = |prof_uid: ProfUID| prof_uid_record(prof_uid, state);
        let mut critical_path = Vec::new();
        let mut last = None;
        for node in &spy_state.critical_path {
            critical_path.push(CriticalPathRecord {
                tuple: render_op(*node),
                obj: last.and_then(render_op),
            });
            last = Some(*node);
        }
        serde_json::to_writer(file, &critical_path)?;
    }

    Ok(())
}
