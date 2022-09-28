use std::cmp::max;
use std::collections::{BTreeMap, BTreeSet};
use std::convert::TryFrom;
use std::fmt;
use std::fs::{create_dir, remove_dir_all, File};
use std::io;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};

use serde::{Serialize, Serializer};

use rayon::prelude::*;

use crate::state::{
    Bounds, Chan, ChanEntry, ChanEntryRef, ChanID, ChanPoint, Color, CopyInfo, DimKind, FSpace,
    ISpaceID, Inst, Mem, MemID, MemKind, MemPoint, MemProcAffinity, NodeID, OpID, Proc,
    ProcEntryKind, ProcID, ProcKind, ProcPoint, ProfUID, State, TimePoint, Timestamp,
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
struct DependencyRecord(u64, u64, u64);

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
    op_id: Option<u64>,
}

#[derive(Serialize, Copy, Clone)]
struct OpRecord<'a> {
    op_id: u64,
    parent_id: u64,
    desc: &'a str,
    proc: Option<&'a str>,
    level: Option<u32>,
    provenance: Option<&'a str>,
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
struct ScaleRecord {
    start: f64,
    end: f64,
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
        let entry = self.entry(point.entry);
        let (op_id, initiation_op) = (entry.op_id, entry.initiation_op);
        let (base, time_range, waiters) = (&entry.base, &entry.time_range, &entry.waiters);
        let name = match entry.kind {
            ProcEntryKind::Task(task_id, variant_id) => {
                let task_name = &state.task_kinds.get(&task_id).unwrap().name;
                let variant_name = &state.variants.get(&(task_id, variant_id)).unwrap().name;
                match task_name {
                    Some(task_name) => {
                        if task_name != variant_name {
                            format!("{} [{}] <{}>", task_name, variant_name, op_id.unwrap().0)
                        } else {
                            format!("{} <{}>", task_name, op_id.unwrap().0)
                        }
                    }
                    None => variant_name.clone(),
                }
            }
            ProcEntryKind::MetaTask(variant_id) => {
                state.meta_variants.get(&variant_id).unwrap().name.clone()
            }
            ProcEntryKind::MapperCall(kind) => {
                let name = &state.mapper_call_kinds.get(&kind).unwrap().name;
                if let Some(initiation_op_id) = initiation_op {
                    format!("Mapper Call {} for {}", name, initiation_op_id.0)
                } else {
                    format!("Mapper Call {}", name)
                }
            }
            ProcEntryKind::RuntimeCall(kind) => {
                state.runtime_call_kinds.get(&kind).unwrap().name.clone()
            }
            ProcEntryKind::ProfTask => format!("ProfTask <{:?}>", initiation_op.unwrap().0),
        };

        let color = match entry.kind {
            ProcEntryKind::Task(task_id, variant_id) => state
                .variants
                .get(&(task_id, variant_id))
                .unwrap()
                .color
                .unwrap(),
            ProcEntryKind::MetaTask(variant_id) => {
                state.meta_variants.get(&variant_id).unwrap().color.unwrap()
            }
            ProcEntryKind::MapperCall(kind) => {
                state.mapper_call_kinds.get(&kind).unwrap().color.unwrap()
            }
            ProcEntryKind::RuntimeCall(kind) => {
                state.runtime_call_kinds.get(&kind).unwrap().color.unwrap()
            }
            ProcEntryKind::ProfTask => {
                // FIXME don't hardcode this here
                Color(0xFFC0CB)
            }
        };
        let color = format!("#{:06x}", color);

        let initiation = match entry.kind {
            // FIXME: Elliott: special case on ProfTask to match legion_prof.py behavior
            ProcEntryKind::ProfTask => None,
            // And another special case, because for MapperCalls only, we skip zero (!!)
            ProcEntryKind::MapperCall(_) => initiation_op.map(|op_id| op_id.0),
            _ => Some(initiation_op.map_or(0, |op_id| op_id.0)),
        };

        let op_id = match entry.kind {
            // FIXME: Elliott: special case on ProfTask to match legion_prof.py behavior
            ProcEntryKind::ProfTask => Some(initiation_op.unwrap().0),
            _ => op_id.map(|id| id.0),
        };

        let render_op = |prof_uid: &ProfUID| {
            state.prof_uid_proc.get(prof_uid).map(|proc_id| {
                DependencyRecord(proc_id.node_id().0, proc_id.proc_in_node(), prof_uid.0)
            })
        };

        let deps = state.spy_op_deps.get(&base.prof_uid);

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
            op_id: op_id,
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

struct SizePretty(u64);

impl fmt::Display for SizePretty {
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
        let name = match entry {
            ChanEntryRef::Copy(_, copy) => {
                let nreqs = copy.copy_info.len();
                if nreqs > 0 {
                    format!(
                        "size={}, num reqs={}{}",
                        SizePretty(copy.size),
                        nreqs,
                        CopyInfoVec(&copy.copy_info)
                    )
                } else {
                    format!("size={}, num reqs={}", SizePretty(copy.size), nreqs)
                }
            }
            ChanEntryRef::Fill(_, _) => format!("Fill"),
            ChanEntryRef::DepPart(_, deppart) => format!("{}", deppart.part_op),
        };
        let ready_timestamp = match point.entry {
            ChanEntry::Copy(_, _) => time_range.ready,
            ChanEntry::Fill(_, _) => None,
            ChanEntry::DepPart(_, _) => None,
        };

        let initiation = match point.entry {
            ChanEntry::Copy(op_id, _) => op_id,
            ChanEntry::Fill(op_id, _) => op_id,
            ChanEntry::DepPart(op_id, _) => op_id,
        };

        let color = format!("#{:06x}", state.get_op_color(initiation));

        let level = max(self.max_levels + 1, 4) - base.level.unwrap();

        f.serialize(DataRecord {
            level,
            level_ready: None,
            ready: ready_timestamp,
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
            op_id: None,
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
        let slug = match (self.chan_id.src, self.chan_id.dst) {
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

        let long_name = match (self.chan_id.src, self.chan_id.dst) {
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

        let short_name = match (self.chan_id.src, self.chan_id.dst) {
            (Some(src), Some(dst)) => format!(
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
            (None, Some(dst)) => format!(
                "{}",
                MemShort(
                    mem_kind(dst),
                    state.mems.get(&dst),
                    state.mem_proc_affinity.get(&dst),
                    state
                )
            ),
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

        let level = max(self.max_levels + 1, 4) - 1;

        Ok(ProcessorRecord {
            full_text: long_name,
            text: short_name,
            tsv: filename,
            levels: level,
        })
    }
}

#[derive(Debug)]
pub struct ISpacePretty<'a>(pub ISpaceID, pub &'a State);

impl fmt::Display for ISpacePretty<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ISpacePretty(ispace_id, state) = self;

        let ispace = state.index_spaces.get(&ispace_id);
        if ispace.is_none() {
            write!(f, "ispace:{}", ispace_id.0)?;
            return Ok(());
        }
        let ispace = ispace.unwrap();

        match &ispace.bounds {
            Bounds::Empty => {
                write!(f, "empty index space")?;
                return Ok(());
            }
            _ => {}
        }

        if let Some(name) = &ispace.name {
            write!(f, "{}", name)?;
        } else {
            let parent = ispace.parent.and_then(|p_id| {
                state.index_partitions.get(&p_id).and_then(|p| {
                    p.parent
                        .map(|gp_id| state.index_spaces.get(&gp_id).unwrap())
                })
            });
            if let Some(name) = parent.and_then(|p| p.name.as_ref()) {
                write!(f, "{}", name)?;
            } else if let Some(parent_id) = parent.map(|p| p.ispace_id) {
                write!(f, "ispace:{}", parent_id.0)?;
            } else {
                write!(f, "ispace:{}", ispace_id.0)?;
            }
        }
        if let Some(size) = &ispace.size {
            if size.is_sparse {
                write!(
                    f,
                    "[sparse:({} of {} points)]",
                    size.sparse_size, size.dense_size
                )?;
                return Ok(());
            }
        }
        match &ispace.bounds {
            Bounds::Point { point, dim } => {
                for x in &point[..*dim as usize] {
                    write!(f, "[{}]", x)?;
                }
            }
            Bounds::Rect { lo, hi, dim } => {
                for (l, h) in lo.iter().zip(hi.iter()).take(*dim as usize) {
                    write!(f, "[{}:{}]", l, h)?;
                }
            }
            Bounds::Empty => unreachable!(),
            Bounds::Unknown => {}
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct FSpacePretty<'a>(pub &'a FSpace, pub &'a Inst);

impl fmt::Display for FSpacePretty<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let FSpacePretty(fspace, inst) = self;

        if let Some(name) = &fspace.name {
            write!(f, "{}", name)?;
        } else {
            write!(f, "fspace:{}", fspace.fspace_id.0)?;
        }

        let align_desc = inst.align_desc.get(&fspace.fspace_id).unwrap();
        let fields = inst.fields.get(&fspace.fspace_id).unwrap();

        let mut fields = fields.iter().enumerate().peekable();
        if fields.peek().is_some() {
            write!(f, "$Fields: [")?;
            while let Some((i, field)) = fields.next() {
                let align = &align_desc[i];
                if let Some(fld) = fspace.fields.get(field) {
                    write!(f, "{}", fld.name)?;
                } else {
                    write!(f, "fid:{}", field.0)?;
                }
                if align.has_align {
                    write!(f, ":align={}", align.align_desc)?;
                }
                if fields.peek().is_some() {
                    write!(f, ",")?;
                    if i % 5 == 0 {
                        write!(f, "$")?;
                    }
                }
            }
            write!(f, "]")?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct DimOrderPretty<'a>(pub &'a Inst);

impl fmt::Display for DimOrderPretty<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inst = self.0;

        let mut aos = false;
        let mut soa = false;
        let mut cmpx_order = false;
        let mut column_major = 0;
        let mut row_major = 0;
        let dim_first = inst.dim_order.iter().next();
        let dim_last = inst.dim_order.iter().last();
        for (dim, dim_order) in &inst.dim_order {
            if dim.0 == 0 {
                if *dim_order == DimKind::DimF {
                    aos = true;
                }
            } else {
                if dim == dim_last.unwrap().0 {
                    if *dim_order == DimKind::DimF {
                        soa = true;
                    }
                } else {
                    if *dim_order == DimKind::DimF {
                        cmpx_order = true;
                    }
                }
            }

            // SOA + order -> DIM_X, DIM_Y,.. DIM_F-> column_major
            // or .. DIM_Y, DIM_X, DIM_F? -> row_major
            if *dim_last.unwrap().1 == DimKind::DimF {
                if *dim_order != DimKind::DimF {
                    if *dim_order == DimKind::try_from(dim.0).unwrap() {
                        column_major += 1;
                    }
                    if *dim_order == DimKind::try_from(dim_last.unwrap().0 .0 - dim.0 - 1).unwrap()
                    {
                        row_major += 1;
                    }
                }
            }

            // AOS + order -> DIM_F, DIM_X, DIM_Y -> column_major
            // or DIM_F, DIM_Y, DIM_X -> row_major?
            if *dim_first.unwrap().1 == DimKind::DimF {
                if *dim_order != DimKind::DimF {
                    if *dim_order == DimKind::try_from(dim.0 - 1).unwrap() {
                        column_major += 1;
                    }
                    if *dim_order == DimKind::try_from(dim_last.unwrap().0 .0 - dim.0).unwrap() {
                        row_major += 1;
                    }
                }
            }
        }
        if dim_last.map_or(false, |(d, _)| d.0 != 1) {
            if column_major == dim_last.unwrap().0 .0 && !cmpx_order {
                write!(f, "[Column Major]")?;
            } else if row_major == dim_last.unwrap().0 .0 && !cmpx_order {
                write!(f, "[Row Major]")?;
            }
        }
        if cmpx_order {
            for (dim, dim_order) in &inst.dim_order {
                write!(f, "[{:?}]", dim_order)?;
                if (dim.0 + 1) % 4 == 0 && dim != dim_last.unwrap().0 {
                    write!(f, "$")?;
                }
            }
        } else if aos {
            write!(f, "[Array-of-structs (AOS)]")?;
        } else if soa {
            write!(f, "[Struct-of-arrays (SOA)]")?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct InstPretty<'a>(pub &'a Inst, pub &'a State);

impl fmt::Display for InstPretty<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let InstPretty(inst, state) = self;

        let mut ispace_ids = inst.ispace_ids.iter().enumerate().peekable();
        while let Some((i, ispace_id)) = ispace_ids.next() {
            let fspace_id = inst.fspace_ids[i];
            let fspace = state.field_spaces.get(&fspace_id).unwrap();

            write!(
                f,
                "Region: {} x {}",
                ISpacePretty(*ispace_id, state),
                FSpacePretty(fspace, inst),
            )?;
            if ispace_ids.peek().is_some() {
                write!(f, "$")?;
            }
        }
        write!(
            f,
            "$Layout Order: {} $Inst: 0x{:x} $Size: {}",
            DimOrderPretty(inst),
            inst.inst_id.0,
            SizePretty(inst.size.unwrap())
        )?;

        Ok(())
    }
}

impl Mem {
    fn emit_tsv_point(
        &self,
        f: &mut csv::Writer<File>,
        point: &MemPoint,
        state: &State,
    ) -> io::Result<()> {
        let (_, op_id) = point.entry;
        let inst = self.insts.get(&point.entry).unwrap();
        let (base, time_range) = (&inst.base, &inst.time_range);
        let name = format!("{}", InstPretty(inst, state));

        let initiation = op_id;

        let color = format!("#{:06x}", state.get_op_color(initiation));

        let level = max(self.max_live_insts + 1, 4) - base.level.unwrap();

        f.serialize(DataRecord {
            level,
            level_ready: None,
            ready: None,
            start: time_range.create.unwrap(),
            end: time_range.ready.unwrap(),
            color: &color,
            opacity: 0.45,
            title: &format!("{} (deferred)", &name),
            initiation: Some(initiation.0),
            in_: "",
            out: "",
            children: "",
            parents: "",
            prof_uid: base.prof_uid.0,
            op_id: None,
        })?;

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
            op_id: None,
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

        for point in &self.time_points {
            if point.first {
                self.emit_tsv_point(&mut f, point, state)?;
            }
        }

        let level = max(self.max_live_insts + 1, 4) - 1;

        Ok(ProcessorRecord {
            full_text: long_name,
            text: short_name,
            tsv: filename,
            levels: level,
        })
    }
}

impl State {
    fn get_op_color(&self, op_id: OpID) -> Color {
        if let Some(task) = self.find_task(op_id) {
            match task.kind {
                ProcEntryKind::Task(task_id, variant_id) => {
                    return self
                        .variants
                        .get(&(task_id, variant_id))
                        .unwrap()
                        .color
                        .unwrap()
                }
                _ => unreachable!(),
            }
        }

        if let Some(op) = self.find_op(op_id) {
            if let Some(kind) = op.kind {
                return self.op_kinds.get(&kind).unwrap().color.unwrap();
            }
        }

        return Color(0x000000);
    }

    fn has_multiple_nodes(&self) -> bool {
        let mut node = None;
        for proc in self.procs.values() {
            match node {
                Some(n) => {
                    if n != proc.proc_id.node_id() {
                        return true;
                    }
                }
                None => {
                    node = Some(proc.proc_id.node_id());
                }
            }
        }
        false
    }

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

        for (chan_id, chan) in &self.chans {
            if !chan.time_points.is_empty() {
                if chan_id.node_id().is_some() {
                    // gathers/scatters
                    let mut nodes = vec![None];
                    if chan_id.dst.is_some() && chan_id.dst.unwrap() != MemID(0) {
                        nodes.push(chan_id.dst.map(|dst| dst.node_id()));
                    }
                    if chan_id.src.is_some() && chan_id.src.unwrap() != MemID(0) {
                        nodes.push(chan_id.src.map(|src| src.node_id()));
                    }
                    nodes.dedup();
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
            let mem_id = self.insts.get(&point.entry).unwrap();
            let mem = self.mems.get(&mem_id).unwrap();
            let inst = mem.insts.get(&point.entry).unwrap();
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

        let multinode = self.has_multiple_nodes();
        for group in timepoint_proc.keys() {
            let (node, kind) = group;
            if node.is_some() || multinode {
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
        }

        for group in timepoint_mem.keys() {
            let (node, kind) = group;
            if node.is_some() || multinode {
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
                    .or_insert_with(|| Vec::new())
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
        base_level += record.levels + 1;
    }

    let chans = state.chans.values().collect::<Vec<_>>();
    let chan_records: BTreeMap<_, _> = chans
        .par_iter()
        .filter(|chan| !chan.is_empty())
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
        .filter(|mem| !mem.is_empty())
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
            let provenance = Some(op.provenance.as_deref().unwrap_or(""));
            if let Some(proc_id) = state.tasks.get(&op_id) {
                let proc = state.procs.get(&proc_id).unwrap();
                let proc_record = proc_records.get(&proc_id).unwrap();
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
                    op_id: op_id.0,
                    parent_id: parent_id.map_or(0, |x| x.0),
                    desc: &desc,
                    proc: Some(&proc_record.full_text),
                    level: task.base.level.map(|x| x + 1),
                    provenance: provenance,
                })?;
            } else if let Some(task) = state.multi_tasks.get(&op_id) {
                let task_name = state
                    .task_kinds
                    .get(&task.task_id)
                    .unwrap()
                    .name
                    .as_ref()
                    .unwrap();

                file.serialize(OpRecord {
                    op_id: op_id.0,
                    parent_id: parent_id.map_or(0, |x| x.0),
                    desc: &format!("{} <{}>", task_name, op_id.0),
                    proc: None,
                    level: None,
                    provenance: provenance,
                })?;
            } else {
                let desc = op.kind.and_then(|k| state.op_kinds.get(&k)).map_or_else(
                    || format!("Operation <{}>", op_id.0),
                    |k| format!("{} Operation <{}>", k.name, op_id.0),
                );

                file.serialize(OpRecord {
                    op_id: op_id.0,
                    parent_id: parent_id.map_or(0, |x| x.0),
                    desc: &desc,
                    proc: None,
                    level: None,
                    provenance: provenance,
                })?;
            }
        }
    }

    {
        let stats_levels = 4;
        let scale_data = ScaleRecord {
            start: 0.0,
            end: (state.last_time.0 as f64 / 100. * 1.01).ceil() / 10.,
            stats_levels: stats_levels,
            max_level: base_level + 1,
        };

        let filename = path.join("json").join("scale.json");
        let file = File::create(filename)?;
        serde_json::to_writer(&file, &scale_data)?;
    }

    {
        let filename = path.join("json").join("critical_path.json");
        let mut file = File::create(filename)?;
        write!(file, "[]")?;
    }

    Ok(())
}
