use std::cmp::max;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::state::{
    Bounds, ChanEntryRef, ChanID, ChanPoint, Color, CopyInstInfo, DimKind, FSpace, FieldID,
    FillInstInfo, ISpaceID, Inst, InstUID, MemID, MemKind, MemPoint, NodeID, OpID, ProcID,
    ProcKind, ProcPoint, State, TimePoint, Timestamp,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProcGroup(pub Option<NodeID>, pub ProcKind);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MemGroup(pub Option<NodeID>, pub MemKind);

pub trait StatePostprocess {
    fn has_multiple_nodes(&self) -> bool;

    fn group_procs(&self) -> BTreeMap<ProcGroup, Vec<ProcID>>;
    fn group_mems(&self) -> BTreeMap<MemGroup, Vec<MemID>>;
    fn group_chans(&self) -> BTreeMap<Option<NodeID>, Vec<ChanID>>;

    fn proc_group_timepoints(&self, procs: &Vec<ProcID>) -> Vec<&Vec<ProcPoint>>;
    fn mem_group_timepoints(&self, mems: &Vec<MemID>) -> Vec<&Vec<MemPoint>>;
    fn chan_group_timepoints(&self, chans: &Vec<ChanID>) -> Vec<&Vec<ChanPoint>>;

    fn group_node_proc_kind_timepoints(
        &self,
    ) -> (
        BTreeMap<ProcGroup, Vec<(ProcID, &Vec<ProcPoint>)>>,
        BTreeMap<ProcGroup, u64>,
    );

    fn group_node_mem_kind_timepoints(&self) -> BTreeMap<MemGroup, Vec<(MemID, &Vec<MemPoint>)>>;

    fn group_node_chan_kind_timepoints(
        &self,
    ) -> BTreeMap<Option<NodeID>, Vec<(ChanID, &Vec<ChanPoint>)>>;

    fn convert_points_to_utilization<Entry, Secondary>(
        &self,
        points: &Vec<TimePoint<Entry, Secondary>>,
        utilization: &mut Vec<TimePoint<Entry, Secondary>>,
    ) where
        Entry: Copy,
        Secondary: Copy;

    fn calculate_proc_utilization_data(
        &self,
        points: Vec<ProcPoint>,
        owners: BTreeSet<ProcID>,
        max_count: u64,
    ) -> Vec<(Timestamp, f64)>;

    fn calculate_dynamic_memory_size(&self, points: &Vec<&MemPoint>) -> u64;

    fn calculate_mem_utilization_data(
        &self,
        points: Vec<&MemPoint>,
        owners: BTreeSet<MemID>,
    ) -> Vec<(Timestamp, f64)>;

    fn calculate_chan_utilization_data(
        &self,
        points: Vec<ChanPoint>,
        owners: BTreeSet<ChanID>,
    ) -> Vec<(Timestamp, f64)>;

    fn op_provenance(&self, op_id: OpID) -> Option<String>;

    fn chan_entry_initiation(&self, entry: ChanEntryRef) -> OpID;
    fn chan_entry_name(&self, entry: ChanEntryRef) -> String;
    fn chan_entry_color(&self, entry: ChanEntryRef) -> Color;
    fn chan_entry_provenance(&self, entry: ChanEntryRef) -> Option<String>;
}

impl StatePostprocess for State {
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

    fn group_procs(&self) -> BTreeMap<ProcGroup, Vec<ProcID>> {
        let mut groups = BTreeMap::new();
        for proc in self.procs.values() {
            // Do NOT filter empty procs here because they count towards
            // utilization totals
            let nodes = [None, Some(proc.proc_id.node_id())];
            for node in nodes {
                let group = ProcGroup(node, proc.kind);
                groups
                    .entry(group)
                    .or_insert_with(Vec::new)
                    .push(proc.proc_id);
            }
        }
        groups
    }

    fn group_mems(&self) -> BTreeMap<MemGroup, Vec<MemID>> {
        let mut groups = BTreeMap::new();
        for mem in self.mems.values() {
            if !mem.time_points.is_empty() {
                let nodes = [None, Some(mem.mem_id.node_id())];
                for node in nodes {
                    let group = MemGroup(node, mem.kind);
                    groups
                        .entry(group)
                        .or_insert_with(Vec::new)
                        .push(mem.mem_id);
                }
            }
        }
        groups
    }

    fn group_chans(&self) -> BTreeMap<Option<NodeID>, Vec<ChanID>> {
        let mut groups = BTreeMap::new();

        for (chan_id, chan) in &self.chans {
            if !chan.time_points.is_empty() && chan_id.node_id().is_some() {
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
                    groups.entry(node).or_insert_with(Vec::new).push(*chan_id)
                }
            }
        }

        groups
    }

    fn proc_group_timepoints(&self, procs: &Vec<ProcID>) -> Vec<&Vec<ProcPoint>> {
        let mut timepoints = Vec::new();
        for proc_id in procs {
            let proc = self.procs.get(proc_id).unwrap();
            timepoints.push(&proc.util_time_points);
        }
        timepoints
    }

    fn mem_group_timepoints(&self, mems: &Vec<MemID>) -> Vec<&Vec<MemPoint>> {
        let mut timepoints = Vec::new();
        for mem_id in mems {
            let mem = self.mems.get(mem_id).unwrap();
            timepoints.push(&mem.time_points);
        }
        timepoints
    }

    fn chan_group_timepoints(&self, chans: &Vec<ChanID>) -> Vec<&Vec<ChanPoint>> {
        let mut timepoints = Vec::new();
        for chan_id in chans {
            let chan = self.chans.get(chan_id).unwrap();
            timepoints.push(&chan.time_points);
        }
        timepoints
    }

    fn group_node_proc_kind_timepoints(
        &self,
    ) -> (
        BTreeMap<ProcGroup, Vec<(ProcID, &Vec<ProcPoint>)>>,
        BTreeMap<ProcGroup, u64>,
    ) {
        let mut timepoint = BTreeMap::new();
        let mut proc_count = BTreeMap::new();

        for proc in self.procs.values() {
            let nodes = [None, Some(proc.proc_id.node_id())];
            for node in nodes {
                let group = ProcGroup(node, proc.kind);
                proc_count.entry(group).and_modify(|i| *i += 1).or_insert(1);
                if !proc.is_empty() {
                    timepoint
                        .entry(group)
                        .or_insert_with(Vec::new)
                        .push((proc.proc_id, &proc.util_time_points));
                }
            }
        }

        (timepoint, proc_count)
    }

    fn group_node_mem_kind_timepoints(&self) -> BTreeMap<MemGroup, Vec<(MemID, &Vec<MemPoint>)>> {
        let mut result = BTreeMap::new();
        for mem in self.mems.values() {
            if !mem.time_points.is_empty() {
                let nodes = [None, Some(mem.mem_id.node_id())];
                for node in nodes {
                    let group = MemGroup(node, mem.kind);
                    result
                        .entry(group)
                        .or_insert_with(Vec::new)
                        .push((mem.mem_id, &mem.time_points))
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
            if !chan.time_points.is_empty() && chan_id.node_id().is_some() {
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
                        .or_insert_with(Vec::new)
                        .push((*chan_id, &chan.time_points))
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

        assert!(!owners.is_empty());

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

    fn calculate_dynamic_memory_size(&self, points: &Vec<&MemPoint>) -> u64 {
        let mut max_count = 0;
        let mut count = 0;

        for point in points {
            let inst = self.find_inst(point.entry).unwrap();
            if point.first {
                count += inst.size.unwrap();
            } else {
                count -= inst.size.unwrap();
            }
            if count > max_count {
                max_count = count;
            }
        }

        max(max_count, 1)
    }

    fn calculate_mem_utilization_data(
        &self,
        points: Vec<&MemPoint>,
        owners: BTreeSet<MemID>,
    ) -> Vec<(Timestamp, f64)> {
        assert!(!owners.is_empty());

        let mut result = Vec::new();

        let mut max_count = 0;
        for mem_id in owners {
            let mem = self.mems.get(&mem_id).unwrap();
            max_count += mem.capacity;
        }

        if max_count == 0 {
            // we are in external memory, so we need to calculate the max capacity
            max_count = self.calculate_dynamic_memory_size(&points);
        }

        let max_count = max_count as f64;
        let mut last_time = None;

        let mut count = 0;
        for point in &points {
            let inst = self.find_inst(point.entry).unwrap();
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

        assert!(!owners.is_empty());

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
            } else if count > 0.0 {
                utilization.push((point.time, 1.0));
            } else {
                utilization.push((point.time, count / max_count));
            }
            last_time = Some(point.time);
        }

        utilization
    }

    fn op_provenance(&self, op_id: OpID) -> Option<String> {
        self.find_op(op_id).and_then(|op| op.provenance.clone())
    }

    fn chan_entry_initiation(&self, entry: ChanEntryRef) -> OpID {
        match entry {
            ChanEntryRef::Copy(_, copy) => copy.op_id.unwrap(),
            ChanEntryRef::Fill(_, fill) => fill.op_id.unwrap(),
            ChanEntryRef::DepPart(_, deppart) => deppart.op_id,
        }
    }

    fn chan_entry_name(&self, entry: ChanEntryRef) -> String {
        match entry {
            ChanEntryRef::Copy(_, copy) => {
                let nreqs = copy.copy_inst_infos.len();
                if nreqs > 0 {
                    format!(
                        "{}: size={}, num reqs={}{}",
                        copy.copy_kind.unwrap(),
                        SizePretty(copy.size.unwrap()),
                        nreqs,
                        CopyInstInfoVec(&copy.copy_inst_infos, self)
                    )
                } else {
                    format!(
                        "Copy: size={}, num reqs={}",
                        SizePretty(copy.size.unwrap()),
                        nreqs
                    )
                }
            }
            ChanEntryRef::Fill(_, fill) => {
                let nreqs = fill.fill_inst_infos.len();
                if nreqs > 0 {
                    format!(
                        "Fill: num reqs={}{}",
                        nreqs,
                        FillInstInfoVec(&fill.fill_inst_infos, self)
                    )
                } else {
                    format!("Fill: num reqs={}", nreqs)
                }
            }
            ChanEntryRef::DepPart(_, deppart) => format!("{}", deppart.part_op),
        }
    }

    fn chan_entry_color(&self, entry: ChanEntryRef) -> Color {
        let initiation = self.chan_entry_initiation(entry);
        self.get_op_color(initiation)
    }

    fn chan_entry_provenance(&self, entry: ChanEntryRef) -> Option<String> {
        let initiation = self.chan_entry_initiation(entry);
        self.op_provenance(initiation)
    }
}

pub struct SizePretty(pub u64);

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
pub struct ISpacePretty<'a>(pub ISpaceID, pub &'a State);

impl fmt::Display for ISpacePretty<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ISpacePretty(ispace_id, state) = self;

        let ispace = state.index_spaces.get(ispace_id);
        if ispace.is_none() {
            write!(f, "ispace:{}", ispace_id.0)?;
            return Ok(());
        }
        let ispace = ispace.unwrap();

        if ispace.bounds == Bounds::Empty {
            write!(f, "empty index space")?;
            return Ok(());
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
            } else if dim == dim_last.unwrap().0 {
                if *dim_order == DimKind::DimF {
                    soa = true;
                }
            } else if *dim_order == DimKind::DimF {
                cmpx_order = true;
            }

            // SOA + order -> DIM_X, DIM_Y,.. DIM_F-> column_major
            // or .. DIM_Y, DIM_X, DIM_F? -> row_major
            if *dim_last.unwrap().1 == DimKind::DimF && *dim_order != DimKind::DimF {
                if *dim_order == DimKind::try_from(dim.0).unwrap() {
                    column_major += 1;
                }
                if *dim_order == DimKind::try_from(dim_last.unwrap().0 .0 - dim.0 - 1).unwrap() {
                    row_major += 1;
                }
            }

            // AOS + order -> DIM_F, DIM_X, DIM_Y -> column_major
            // or DIM_F, DIM_Y, DIM_X -> row_major?
            if *dim_first.unwrap().1 == DimKind::DimF && *dim_order != DimKind::DimF {
                if *dim_order == DimKind::try_from(dim.0 - 1).unwrap() {
                    column_major += 1;
                }
                if *dim_order == DimKind::try_from(dim_last.unwrap().0 .0 - dim.0).unwrap() {
                    row_major += 1;
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
        if inst.dim_order.len() > 0 {
            write!(f, "$Layout Order: {} ", DimOrderPretty(inst))?;
        }
        write!(
            f,
            "$Inst: 0x{:x} $Size: {}",
            inst.inst_id.unwrap().0,
            SizePretty(inst.size.unwrap())
        )?;

        Ok(())
    }
}

#[derive(Debug)]
pub struct CopyInstInfoDisplay<'a>(
    pub Option<&'a Inst>, // src_inst
    pub Option<&'a Inst>, // src_dst
    pub InstUID,          // src_inst_uid
    pub InstUID,          // dst_inst_uid
);

impl fmt::Display for CopyInstInfoDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut src_inst_id = 0;
        let mut dst_inst_id = 0;
        if let Some(src_inst) = self.0 {
            src_inst_id = src_inst.inst_id.unwrap().0;
        }
        if let Some(dst_inst) = self.1 {
            dst_inst_id = dst_inst.inst_id.unwrap().0;
        }
        match (self.2 .0, self.3 .0) {
            (0, 0) => unreachable!(),
            (0, _) => {
                write!(f, "Scatter: dst_indirect_inst=0x{:x}", dst_inst_id)
            }
            (_, 0) => {
                write!(f, "Gather: src_indirect_inst=0x{:x}", src_inst_id)
            }
            (_, _) => {
                write!(
                    f,
                    "src_inst=0x{:x}, dst_inst=0x{:x}",
                    src_inst_id, dst_inst_id
                )
            }
        }
    }
}

#[derive(Debug)]
pub struct CopyInstInfoVec<'a>(pub &'a Vec<CopyInstInfo>, pub &'a State);

impl fmt::Display for CopyInstInfoVec<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, elt) in self.0.iter().enumerate() {
            let src_inst = self.1.find_inst(elt.src_inst_uid);
            let dst_inst = self.1.find_inst(elt.dst_inst_uid);
            write!(
                f,
                "$req[{}]: {}",
                i,
                CopyInstInfoDisplay(src_inst, dst_inst, elt.src_inst_uid, elt.dst_inst_uid)
            )?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct CopyInstInfoDumpInstVec<'a>(pub &'a Vec<CopyInstInfo>, pub &'a State);

impl fmt::Display for CopyInstInfoDumpInstVec<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // remove duplications
        let mut insts_set = BTreeSet::new();
        for elt in self.0.iter() {
            if let Some(src_inst) = self.1.find_inst(elt.src_inst_uid) {
                insts_set.insert(src_inst);
            }
            if let Some(dst_inst) = self.1.find_inst(elt.dst_inst_uid) {
                insts_set.insert(dst_inst);
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

#[derive(Debug)]
pub struct FillInstInfoDisplay<'a>(pub Option<&'a Inst>, pub FieldID);

impl fmt::Display for FillInstInfoDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut inst_id = 0;
        if let Some(inst) = self.0 {
            inst_id = inst.inst_id.unwrap().0;
        }
        write!(f, "dst_inst=0x{:x}, fid={}", inst_id, self.1 .0)
    }
}

#[derive(Debug)]
pub struct FillInstInfoVec<'a>(pub &'a Vec<FillInstInfo>, pub &'a State);

impl fmt::Display for FillInstInfoVec<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, elt) in self.0.iter().enumerate() {
            let inst = self.1.find_inst(elt.dst_inst_uid);
            write!(f, "$req[{}]: {}", i, FillInstInfoDisplay(inst, elt.fid))?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct FillInstInfoDumpInstVec<'a>(pub &'a Vec<FillInstInfo>, pub &'a State);

impl fmt::Display for FillInstInfoDumpInstVec<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // remove duplications
        let mut insts_set = BTreeSet::new();
        for elt in self.0.iter() {
            let dst_inst = self.1.find_inst(elt.dst_inst_uid).unwrap();
            insts_set.insert(dst_inst);
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
