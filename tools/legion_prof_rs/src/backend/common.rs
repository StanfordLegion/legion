use std::cmp::max;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::state::{
    Align, Bounds, ChanEntry, ChanID, ChanPoint, Config, Container, CopyInstInfo, DeviceKind,
    DimKind, FSpace, FieldID, FillInstInfo, ISpaceID, Inst, MemID, MemKind, MemPoint, NodeID,
    ProcID, ProcKind, ProcPoint, ProfUID, State, TimePoint, Timestamp,
};

use crate::conditional_assert;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProcGroup(pub Option<NodeID>, pub ProcKind, pub Option<DeviceKind>);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MemGroup(pub Option<NodeID>, pub MemKind);

pub trait StatePostprocess {
    fn has_multiple_nodes(&self) -> bool;

    fn group_procs(&self) -> BTreeMap<ProcGroup, Vec<ProcID>>;
    fn group_mems(&self) -> BTreeMap<MemGroup, Vec<MemID>>;
    fn group_chans(&self) -> BTreeMap<Option<NodeID>, Vec<ChanID>>;
    fn group_depparts(&self) -> BTreeMap<Option<NodeID>, Vec<ChanID>>;

    fn proc_group_timepoints(
        &self,
        device: Option<DeviceKind>,
        procs: &Vec<ProcID>,
    ) -> Vec<&Vec<ProcPoint>>;
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
}

impl StatePostprocess for State {
    fn has_multiple_nodes(&self) -> bool {
        let mut node = None;
        for proc in self.procs.values() {
            if !proc.is_visible() {
                continue;
            }
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
            if !proc.is_visible() {
                continue;
            }
            // Do NOT filter empty procs here because they count towards
            // utilization totals
            let nodes = [None, Some(proc.proc_id.node_id())];
            let devices: &'static [_] = match proc.kind.unwrap() {
                ProcKind::GPU => &[Some(DeviceKind::Device), Some(DeviceKind::Host)],
                _ => &[None],
            };
            for node in nodes {
                for device in devices {
                    let group = ProcGroup(node, proc.kind.unwrap(), *device);
                    groups
                        .entry(group)
                        .or_insert_with(Vec::new)
                        .push(proc.proc_id);
                }
            }
        }
        groups
    }

    fn group_mems(&self) -> BTreeMap<MemGroup, Vec<MemID>> {
        let mut groups = BTreeMap::new();
        for mem in self.mems.values() {
            if !mem.is_visible() {
                continue;
            }
            if !mem.util_time_points(None).is_empty() {
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
            match *chan_id {
                ChanID::Copy { .. }
                | ChanID::Fill { .. }
                | ChanID::Gather { .. }
                | ChanID::Scatter { .. } => {} // ok
                _ => {
                    continue;
                }
            }

            if !chan.is_visible() {
                continue;
            }
            if !chan.util_time_points(None).is_empty() {
                let mut nodes = vec![None];
                match *chan_id {
                    ChanID::Copy { src, dst } => {
                        nodes.push(Some(src.node_id()));
                        nodes.push(Some(dst.node_id()));
                    }
                    ChanID::Fill { dst } | ChanID::Gather { dst } => {
                        nodes.push(Some(dst.node_id()))
                    }
                    ChanID::Scatter { src } => nodes.push(Some(src.node_id())),
                    ChanID::DepPart { .. } => unreachable!(),
                }
                nodes.dedup();
                for node in nodes {
                    groups.entry(node).or_insert_with(Vec::new).push(*chan_id)
                }
            }
        }

        groups
    }

    fn group_depparts(&self) -> BTreeMap<Option<NodeID>, Vec<ChanID>> {
        let mut groups = BTreeMap::new();

        for (chan_id, chan) in &self.chans {
            match *chan_id {
                ChanID::DepPart { .. } => {} // ok
                _ => {
                    continue;
                }
            }
            if !chan.is_visible() {
                continue;
            }
            if !chan.util_time_points(None).is_empty() {
                let mut nodes = vec![None];
                match *chan_id {
                    ChanID::DepPart { node_id } => nodes.push(Some(node_id)),
                    _ => unreachable!(),
                }
                nodes.dedup();
                for node in nodes {
                    groups.entry(node).or_insert_with(Vec::new).push(*chan_id);
                }
            }
        }

        groups
    }

    fn proc_group_timepoints(
        &self,
        device: Option<DeviceKind>,
        procs: &Vec<ProcID>,
    ) -> Vec<&Vec<ProcPoint>> {
        let mut timepoints = Vec::new();
        for proc_id in procs {
            let proc = self.procs.get(proc_id).unwrap();
            if proc.is_visible() {
                timepoints.push(proc.util_time_points(device));
            }
        }
        timepoints
    }

    fn mem_group_timepoints(&self, mems: &Vec<MemID>) -> Vec<&Vec<MemPoint>> {
        let mut timepoints = Vec::new();
        for mem_id in mems {
            let mem = self.mems.get(mem_id).unwrap();
            if mem.is_visible() {
                timepoints.push(mem.util_time_points(None));
            }
        }
        timepoints
    }

    fn chan_group_timepoints(&self, chans: &Vec<ChanID>) -> Vec<&Vec<ChanPoint>> {
        let mut timepoints = Vec::new();
        for chan_id in chans {
            let chan = self.chans.get(chan_id).unwrap();
            if chan.is_visible() {
                timepoints.push(chan.util_time_points(None));
            }
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
            if !proc.is_visible() {
                continue;
            }
            let nodes = [None, Some(proc.proc_id.node_id())];
            let devices: &'static [_] = match proc.kind.unwrap() {
                ProcKind::GPU => &[Some(DeviceKind::Device), Some(DeviceKind::Host)],
                _ => &[None],
            };
            for node in nodes {
                for device in devices {
                    let group = ProcGroup(node, proc.kind.unwrap(), *device);
                    proc_count.entry(group).and_modify(|i| *i += 1).or_insert(1);
                    if !proc.is_empty() {
                        timepoint
                            .entry(group)
                            .or_insert_with(Vec::new)
                            .push((proc.proc_id, proc.util_time_points(*device)));
                    }
                }
            }
        }

        (timepoint, proc_count)
    }

    fn group_node_mem_kind_timepoints(&self) -> BTreeMap<MemGroup, Vec<(MemID, &Vec<MemPoint>)>> {
        let mut result = BTreeMap::new();
        for mem in self.mems.values() {
            if !mem.is_visible() {
                continue;
            }
            if !mem.time_points(None).is_empty() {
                let nodes = [None, Some(mem.mem_id.node_id())];
                for node in nodes {
                    let group = MemGroup(node, mem.kind);
                    result
                        .entry(group)
                        .or_insert_with(Vec::new)
                        .push((mem.mem_id, mem.util_time_points(None)))
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
            if !chan.is_visible() {
                continue;
            }
            if !chan.time_points(None).is_empty() {
                let mut nodes = vec![None];
                match *chan_id {
                    ChanID::Copy { src, dst } => {
                        nodes.push(Some(src.node_id()));
                        nodes.push(Some(dst.node_id()));
                    }
                    ChanID::Fill { dst } | ChanID::Gather { dst } => {
                        nodes.push(Some(dst.node_id()))
                    }
                    ChanID::Scatter { src } => nodes.push(Some(src.node_id())),
                    ChanID::DepPart { node_id } => nodes.push(Some(node_id)),
                }
                nodes.dedup();
                for node in nodes {
                    if node.map_or(true, |n| State::is_on_visible_nodes(&self.visible_nodes, n)) {
                        result
                            .entry(node)
                            .or_insert_with(Vec::new)
                            .push((*chan_id, chan.util_time_points(None)))
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
pub struct FSpaceShort<'a>(pub &'a FSpace);

impl fmt::Display for FSpaceShort<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let FSpaceShort(fspace) = self;

        if let Some(name) = &fspace.name {
            write!(f, "{} <{}>", name, fspace.fspace_id.0)
        } else {
            write!(f, "<{}>", fspace.fspace_id.0)
        }
    }
}

#[derive(Debug)]
pub struct FieldPretty<'a>(pub &'a FSpace, pub FieldID, pub &'a Align);

impl fmt::Display for FieldPretty<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let FieldPretty(fspace, field_id, align) = self;

        if let Some(field) = fspace.fields.get(field_id) {
            write!(f, "{} <{}>", field.name, field_id.0)?;
        } else {
            write!(f, "<{}>", field_id.0)?;
        }

        if align.has_align {
            write!(f, " (align={})", align.align_desc)?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct FieldsPretty<'a>(pub &'a FSpace, pub &'a Inst);

impl fmt::Display for FieldsPretty<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let FieldsPretty(fspace, inst) = self;

        let field_ids = inst.fields.get(&fspace.fspace_id).unwrap();
        let align_desc = inst.align_desc.get(&fspace.fspace_id).unwrap();
        let mut i = field_ids.iter().zip(align_desc.iter()).peekable();
        while let Some((field_id, align)) = i.next() {
            write!(f, "{}", FieldPretty(&fspace, *field_id, align))?;
            if i.peek().is_some() {
                write!(f, ", ")?;
            }
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
                    write!(f, "fid:{}:{}", field.0, fld.name)?;
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
pub struct DimOrderPretty<'a>(pub &'a Inst, pub bool);

impl fmt::Display for DimOrderPretty<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let DimOrderPretty(inst, brackets) = self;

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

        let open = |f: &mut fmt::Formatter<'_>, previous: &mut bool| -> fmt::Result {
            if *brackets {
                write!(f, "[")?;
            } else if *previous {
                write!(f, ", ")?;
            }
            Ok(())
        };
        let close = |f: &mut fmt::Formatter<'_>, previous: &mut bool| -> fmt::Result {
            if *brackets {
                write!(f, "]")?;
            }
            *previous = true;
            Ok(())
        };

        let mut previous = false;

        if dim_last.map_or(false, |(d, _)| d.0 != 1) {
            if column_major == dim_last.unwrap().0 .0 && !cmpx_order {
                open(f, &mut previous)?;
                write!(f, "Column Major")?;
                close(f, &mut previous)?;
            } else if row_major == dim_last.unwrap().0 .0 && !cmpx_order {
                open(f, &mut previous)?;
                write!(f, "Row Major")?;
                close(f, &mut previous)?;
            }
        }
        if cmpx_order {
            open(f, &mut previous)?;
            for (dim, dim_order) in &inst.dim_order {
                write!(f, "{:?}", dim_order)?;
                if *brackets && (dim.0 + 1) % 4 == 0 && dim != dim_last.unwrap().0 {
                    write!(f, "$")?;
                }
            }
            close(f, &mut previous)?;
        } else if aos {
            open(f, &mut previous)?;
            write!(f, "Array-of-structs (AOS)")?;
            close(f, &mut previous)?;
        } else if soa {
            open(f, &mut previous)?;
            write!(f, "Struct-of-arrays (SOA)")?;
            close(f, &mut previous)?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct InstShort<'a>(pub &'a Inst);

impl fmt::Display for InstShort<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let InstShort(inst) = self;

        write!(f, "0x{:x}", inst.inst_id.unwrap().0)
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
            write!(f, "$Layout Order: {} ", DimOrderPretty(inst, true))?;
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
pub struct ChanEntryShort<'a>(pub &'a ChanEntry);

impl fmt::Display for ChanEntryShort<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ChanEntryShort(entry) = self;

        match entry {
            ChanEntry::Copy(copy) => write!(f, "{}", copy.copy_kind.unwrap()),
            ChanEntry::Fill(_) => write!(f, "Fill"),
            ChanEntry::DepPart(deppart) => write!(f, "{}", deppart.part_op),
        }
    }
}

#[derive(Debug)]
pub struct ChanEntryFieldsPretty<'a>(pub Option<&'a Inst>, pub &'a Vec<FieldID>, pub &'a State);

impl fmt::Display for ChanEntryFieldsPretty<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ChanEntryFieldsPretty(inst, field_ids, state) = self;

        let fspace = inst.and_then(|inst| {
            // FIXME (Elliott): not sure how we're supposed to do this if we
            // have more than one field space in an instance
            if inst.fspace_ids.len() == 1 {
                let fspace_id = inst.fspace_ids[0];
                Some(state.field_spaces.get(&fspace_id).unwrap())
            } else {
                None
            }
        });

        let mut i = field_ids.iter().peekable();
        while let Some(fid) = i.next() {
            if let Some(field) = fspace.and_then(|fs| fs.fields.get(fid)) {
                write!(f, "{} <{}>", field.name, fid.0)?;
            } else {
                write!(f, "<{}>", fid.0)?;
            }
            if i.peek().is_some() {
                write!(f, ", ")?;
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct CopyInstInfoDisplay<'a>(
    pub Option<&'a Inst>, // src_inst
    pub Option<&'a Inst>, // src_dst
    pub ProfUID,          // src_inst_uid
    pub ProfUID,          // dst_inst_uid
    pub FieldID,          // src_fid
    pub FieldID,          // dst_fid
    pub u32,              // num_hops
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
                write!(
                    f,
                    "Scatter: dst_indirect_inst=0x{:x}, fid={}",
                    dst_inst_id, self.5 .0
                )
            }
            (_, 0) => {
                write!(
                    f,
                    "Gather: src_indirect_inst=0x{:x}, fid={}",
                    src_inst_id, self.4 .0
                )
            }
            (_, _) => {
                write!(
                    f,
                    "src_inst=0x{:x}, src_fid={}, dst_inst=0x{:x}, dst_fid={}, num_hops={}",
                    src_inst_id, self.4 .0, dst_inst_id, self.5 .0, self.6
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
                CopyInstInfoDisplay(
                    src_inst,
                    dst_inst,
                    elt.src_inst_uid,
                    elt.dst_inst_uid,
                    elt.src_fid,
                    elt.dst_fid,
                    elt.num_hops
                )
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
            // src_inst_uid = 0 means scatter (indirection inst)
            if elt.src_inst_uid != ProfUID(0) {
                if let Some(src_inst) = self.1.find_inst(elt.src_inst_uid) {
                    insts_set.insert(src_inst);
                } else {
                    conditional_assert!(
                        false,
                        Config::all_logs(),
                        "Copy can not find src_inst:0x{:x}",
                        elt.src_inst_uid.0
                    );
                }
            }
            // dst_inst_uid = 0 means gather (indirection inst)
            if elt.dst_inst_uid != ProfUID(0) {
                if let Some(dst_inst) = self.1.find_inst(elt.dst_inst_uid) {
                    insts_set.insert(dst_inst);
                } else {
                    conditional_assert!(
                        false,
                        Config::all_logs(),
                        "Copy can not find dst_inst:0x{:x}",
                        elt.dst_inst_uid.0
                    );
                }
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
            if let Some(dst_inst) = self.1.find_inst(elt.dst_inst_uid) {
                insts_set.insert(dst_inst);
            } else {
                conditional_assert!(
                    false,
                    Config::all_logs(),
                    "Fill can not find dst_inst:0x{:x}",
                    elt.dst_inst_uid.0
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
