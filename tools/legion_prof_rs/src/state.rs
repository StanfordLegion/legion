use std::cmp::{max, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::convert::TryFrom;
use std::fmt;

use derive_more::{Add, From, LowerHex, Sub};
use num_enum::TryFromPrimitive;

use rayon::prelude::*;

use crate::serialize::Record;

const TASK_GRANULARITY_THRESHOLD: Timestamp = Timestamp::from_us(10);

// Make sure this is up to date with lowlevel.h
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, TryFromPrimitive)]
#[repr(i32)]
pub enum ProcKind {
    GPU = 1,
    CPU = 2,
    Utility = 3,
    IO = 4,
    ProcGroup = 5,
    ProcSet = 6,
    OpenMP = 7,
    Python = 8,
}

// Make sure this is up to date with lowlevel.h
#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, TryFromPrimitive)]
#[repr(i32)]
pub enum MemKind {
    NoMemKind = 0,
    Global = 1,
    System = 2,
    Registered = 3,
    Socket = 4,
    ZeroCopy = 5,
    Framebuffer = 6,
    Disk = 7,
    HDF5 = 8,
    File = 9,
    L3Cache = 10,
    L2Cache = 11,
    L1Cache = 12,
}

impl fmt::Display for MemKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemKind::ZeroCopy => write!(f, "Zero-Copy"),
            _ => write!(f, "{:?}", self),
        }
    }
}

// Make sure this is up to date with lowlevel.h
#[derive(Debug, Copy, Clone, Eq, PartialEq, TryFromPrimitive)]
#[repr(i32)]
pub enum DepPartKind {
    Union = 0,
    Unions = 1,
    UnionReduction = 2,
    Intersection = 3,
    Intersections = 4,
    IntersectionReduction = 5,
    Difference = 6,
    Differences = 7,
    EqualPartition = 8,
    PartitionByField = 9,
    PartitionByImage = 10,
    PartitionByImageRange = 11,
    PartitionByPreimage = 12,
    PartitionByPreimageRange = 13,
    CreateAssociation = 14,
    PartitionByWeights = 15,
}

impl fmt::Display for DepPartKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DepPartKind::UnionReduction => write!(f, "Union Reduction"),
            DepPartKind::IntersectionReduction => write!(f, "Intersection Reduction"),
            DepPartKind::EqualPartition => write!(f, "Equal Partition"),
            DepPartKind::PartitionByField => write!(f, "Partition by Field"),
            DepPartKind::PartitionByImage => write!(f, "Partition by Image"),
            DepPartKind::PartitionByImageRange => write!(f, "Partition by Image Range"),
            DepPartKind::PartitionByPreimage => write!(f, "Partition by Preimage"),
            DepPartKind::PartitionByPreimageRange => write!(f, "Partition by Preimage Range"),
            DepPartKind::CreateAssociation => write!(f, "Create Association"),
            DepPartKind::PartitionByWeights => write!(f, "Partition by Weights"),
            _ => write!(f, "{:?}", self),
        }
    }
}

// Make sure this is up to date with lowlevel.h
#[derive(Debug, Copy, Clone, Eq, PartialEq, TryFromPrimitive)]
#[repr(u32)]
pub enum DimKind {
    DimX = 0,
    DimY = 1,
    DimZ = 2,
    DimW = 3,
    DimV = 4,
    DimU = 5,
    DimT = 6,
    DimS = 7,
    DimR = 8,
    DimF = 9,
    InnerDimX = 10,
    OuterDimX = 11,
    InnerDimY = 12,
    OuterDimY = 13,
    InnerDimZ = 14,
    OuterDimZ = 15,
    InnerDimW = 16,
    OuterDimW = 17,
    InnerDimV = 18,
    OuterDimV = 19,
    InnerDimU = 20,
    OuterDimU = 21,
    InnerDimT = 22,
    OuterDimT = 23,
    InnerDimS = 24,
    OuterDimS = 25,
    InnerDimR = 26,
    OuterDimR = 27,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Default, Add, Sub, From)]
pub struct Timestamp(pub u64 /* ns */);

impl Timestamp {
    pub const fn from_us(microseconds: u64) -> Timestamp {
        Timestamp(microseconds * 1000)
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Time is stored in nanoseconds. But it is displayed in microseconds.
        let nanoseconds = self.0;
        let divisor = 1000;
        let microseconds = nanoseconds / divisor;
        let remainder = nanoseconds % divisor;
        write!(f, "{}.{:0>3}", microseconds, remainder)
    }
}

#[derive(Debug, Copy, Clone)]
pub enum ProcEntry {
    Task(OpID),
    MetaTask(OpID, VariantID, usize),
    MapperCall(usize),
    RuntimeCall(usize),
    ProfTask(usize),
}

#[derive(Debug, Copy, Clone)]
pub struct TimePoint<Entry, Secondary>
where
    Entry: std::marker::Copy,
    Secondary: std::marker::Copy,
{
    pub time: Timestamp,
    // Secondary sort_key, used for breaking ties in sorting
    // In practice, we plan for this to be a nanosecond timestamp,
    // like the time field above.
    pub secondary_sort_key: Secondary,
    pub entry: Entry,
    pub first: bool,
}

impl<Entry, Secondary> TimePoint<Entry, Secondary>
where
    Entry: std::marker::Copy,
    Secondary: std::marker::Copy,
{
    pub fn new(time: Timestamp, entry: Entry, first: bool, secondary_sort_key: Secondary) -> Self {
        TimePoint {
            time,
            entry,
            first,
            secondary_sort_key: secondary_sort_key,
        }
    }
    pub fn time_key(&self) -> (u64, u8, Secondary) {
        (
            self.time.0,
            if self.first { 0 } else { 1 },
            self.secondary_sort_key,
        )
    }
}

pub type ProcPoint = TimePoint<ProcEntry, u64>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, LowerHex)]
pub struct ProcID(pub u64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeID(pub u64);

impl ProcID {
    // Important: keep this in sync with realm/id.h
    // PROCESSOR:   tag:8 = 0x1d, owner_node:16,   (unused):28, proc_idx: 12
    // owner_node = proc_id[55:40]
    // proc_idx = proc_id[11:0]
    pub fn node_id(&self) -> NodeID {
        NodeID((self.0 >> 40) & ((1 << 16) - 1))
    }
    pub fn proc_in_node(&self) -> u64 {
        (self.0) & ((1 << 12) - 1)
    }
}

#[derive(Debug)]
pub struct Proc {
    pub proc_id: ProcID,
    pub kind: ProcKind,
    pub tasks: BTreeMap<OpID, Task>,
    pub meta_tasks: BTreeMap<(OpID, VariantID), Vec<MetaTask>>,
    pub mapper_calls: Vec<MapperCall>,
    pub runtime_calls: Vec<RuntimeCall>,
    pub prof_tasks: Vec<ProfTask>,
    pub max_levels: u32,
    pub max_levels_ready: u32,
    pub time_points: Vec<ProcPoint>,
    pub util_time_points: Vec<ProcPoint>,
}

impl Proc {
    fn new(proc_id: ProcID, kind: ProcKind) -> Self {
        Proc {
            proc_id,
            kind,
            tasks: BTreeMap::new(),
            meta_tasks: BTreeMap::new(),
            mapper_calls: Vec::new(),
            runtime_calls: Vec::new(),
            prof_tasks: Vec::new(),
            max_levels: 0,
            max_levels_ready: 0,
            time_points: Vec::new(),
            util_time_points: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
            && self.meta_tasks.is_empty()
            && self.mapper_calls.is_empty()
            && self.runtime_calls.is_empty()
            && self.prof_tasks.is_empty()
    }

    pub fn entry(&self, entry: ProcEntry) -> (&Base, &TimeRange, &Waiters) {
        match entry {
            ProcEntry::Task(op_id) => {
                let task = self.tasks.get(&op_id).unwrap();
                (&task.base, &task.time_range, &task.waiters)
            }
            ProcEntry::MetaTask(op_id, variant_id, idx) => {
                let task = &self.meta_tasks.get(&(op_id, variant_id)).unwrap()[idx];
                (&task.base, &task.time_range, &task.waiters)
            }
            ProcEntry::MapperCall(idx) => {
                let call = &self.mapper_calls[idx];
                (&call.base, &call.time_range, &call.waiters)
            }
            ProcEntry::RuntimeCall(idx) => {
                let call = &self.runtime_calls[idx];
                (&call.base, &call.time_range, &call.waiters)
            }
            ProcEntry::ProfTask(idx) => {
                let task = &self.prof_tasks.get(idx).unwrap();
                (&task.base, &task.time_range, &task.waiters)
            }
        }
    }

    pub fn entry_mut(&mut self, entry: ProcEntry) -> (&mut Base, &mut TimeRange, &mut Waiters) {
        match entry {
            ProcEntry::Task(op_id) => {
                let task = self.tasks.get_mut(&op_id).unwrap();
                (&mut task.base, &mut task.time_range, &mut task.waiters)
            }
            ProcEntry::MetaTask(op_id, variant_id, idx) => {
                let task = &mut self.meta_tasks.get_mut(&(op_id, variant_id)).unwrap()[idx];
                (&mut task.base, &mut task.time_range, &mut task.waiters)
            }
            ProcEntry::MapperCall(idx) => {
                let call = &mut self.mapper_calls[idx];
                (&mut call.base, &mut call.time_range, &mut call.waiters)
            }
            ProcEntry::RuntimeCall(idx) => {
                let call = &mut self.runtime_calls[idx];
                (&mut call.base, &mut call.time_range, &mut call.waiters)
            }
            ProcEntry::ProfTask(idx) => {
                let task = self.prof_tasks.get_mut(idx).unwrap();
                (&mut task.base, &mut task.time_range, &mut task.waiters)
            }
        }
    }

    pub fn entry_base(&self, entry: ProcEntry) -> &Base {
        self.entry(entry).0
    }

    pub fn entry_base_mut(&mut self, entry: ProcEntry) -> &mut Base {
        self.entry_mut(entry).0
    }

    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) {
        // BTreeMap::retain requires Rust 1.53
        let mut removed_tasks = Vec::new();
        for (op_id, task) in self.tasks.iter_mut() {
            task.trim_time_range(start, stop);
            if task.time_range.was_removed {
                removed_tasks.push(*op_id);
            }
        }
        for op_id in removed_tasks {
            self.tasks.remove(&op_id);
        }
        for mapper_call in &mut self.mapper_calls {
            mapper_call.trim_time_range(start, stop);
        }
        self.mapper_calls.retain(|t| !t.time_range.was_removed);
        for runtime_call in &mut self.runtime_calls {
            runtime_call.trim_time_range(start, stop);
        }
        self.runtime_calls.retain(|t| !t.time_range.was_removed);
        for prof_task in &mut self.prof_tasks {
            prof_task.trim_time_range(start, stop);
        }
        self.prof_tasks.retain(|t| !t.time_range.was_removed);
    }

    fn sort_time_range(&mut self) {
        fn add(
            time: &TimeRange,
            entry: ProcEntry,
            all_points: &mut Vec<ProcPoint>,
            points: &mut Vec<ProcPoint>,
            util_points: &mut Vec<ProcPoint>,
        ) {
            let start = time.start.unwrap();
            let stop = time.stop.unwrap();
            let ready = time.ready;
            if stop - start > TASK_GRANULARITY_THRESHOLD && !ready.is_none() {
                all_points.push(ProcPoint::new(ready.unwrap(), entry, true, start.0));
                all_points.push(ProcPoint::new(stop, entry, false, 0));
            } else {
                all_points.push(ProcPoint::new(start, entry, true, 0));
                all_points.push(ProcPoint::new(stop, entry, false, 0));
            }

            points.push(ProcPoint::new(start, entry, true, 0));
            points.push(ProcPoint::new(stop, entry, false, 0));

            util_points.push(ProcPoint::new(start, entry, true, 0));
            util_points.push(ProcPoint::new(stop, entry, false, 0));
        }
        fn add_waiters(waiters: &Waiters, entry: ProcEntry, util_points: &mut Vec<ProcPoint>) {
            for wait in &waiters.wait_intervals {
                util_points.push(ProcPoint::new(wait.start, entry, false, 0));
                util_points.push(ProcPoint::new(wait.end, entry, true, 0));
            }
        }

        let mut all_points = Vec::new();
        let mut points = Vec::new();
        let mut util_points = Vec::new();

        for task in self.tasks.values() {
            let time = &task.time_range;
            let entry = ProcEntry::Task(task.op_id);
            add(&time, entry, &mut all_points, &mut points, &mut util_points);
            add_waiters(&task.waiters, entry, &mut util_points);
        }
        for tasks in self.meta_tasks.values() {
            for (idx, task) in tasks.iter().enumerate() {
                let time = &task.time_range;
                let entry = ProcEntry::MetaTask(task.op_id, task.variant_id, idx);
                add(&time, entry, &mut all_points, &mut points, &mut util_points);
                add_waiters(&task.waiters, entry, &mut util_points);
            }
        }
        for (idx, mapper_call) in self.mapper_calls.iter().enumerate() {
            let time = &mapper_call.time_range;
            let entry = ProcEntry::MapperCall(idx);
            add(&time, entry, &mut all_points, &mut points, &mut util_points);
            add_waiters(&mapper_call.waiters, entry, &mut util_points);
        }
        for (idx, runtime_call) in self.runtime_calls.iter().enumerate() {
            let time = &runtime_call.time_range;
            let entry = ProcEntry::RuntimeCall(idx);
            add(&time, entry, &mut all_points, &mut points, &mut util_points);
            add_waiters(&runtime_call.waiters, entry, &mut util_points);
        }

        for (idx, prof_task) in self.prof_tasks.iter().enumerate() {
            let time = &prof_task.time_range;
            let entry = ProcEntry::ProfTask(idx);
            add(&time, entry, &mut all_points, &mut points, &mut util_points);
            add_waiters(&prof_task.waiters, entry, &mut util_points);
        }

        points.sort_by(|a, b| a.time_key().cmp(&b.time_key()));
        util_points.sort_by(|a, b| a.time_key().cmp(&b.time_key()));

        // Hack: This is a max heap so reverse the values as they go in.
        let mut free_levels = BinaryHeap::<Reverse<u32>>::new();
        for point in &points {
            if point.first {
                let level = if let Some(level) = free_levels.pop() {
                    level.0
                } else {
                    self.max_levels += 1;
                    self.max_levels
                };
                self.entry_base_mut(point.entry).set_level(level);
            } else {
                let level = self.entry_base(point.entry).level.unwrap();
                free_levels.push(Reverse(level));
            }
        }

        all_points.sort_by(|a, b| a.time_key().cmp(&b.time_key()));

        // Hack: This is a max heap so reverse the values as they go in.
        let mut free_levels_ready = BinaryHeap::<Reverse<u32>>::new();
        for point in &all_points {
            if point.first {
                let level = if let Some(level) = free_levels_ready.pop() {
                    level.0
                } else {
                    self.max_levels_ready += 1;
                    self.max_levels_ready
                };
                self.entry_base_mut(point.entry).set_level_ready(level);
            } else {
                let level = self.entry_base(point.entry).level_ready.unwrap();
                free_levels_ready.push(Reverse(level));
            }
        }

        self.time_points = points;
        self.util_time_points = util_points;
    }
}

pub type MemEntry = (InstID, OpID);

pub type MemPoint = TimePoint<MemEntry, ()>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, LowerHex)]
pub struct MemID(pub u64);

impl MemID {
    // Important: keep this in sync with realm/id.h
    // MEMORY:      tag:8 = 0x1e, owner_node:16,   (unused):32, mem_idx: 8
    // owner_node = mem_id[55:40]
    pub fn node_id(&self) -> NodeID {
        NodeID((self.0 >> 40) & ((1 << 16) - 1))
    }
    pub fn mem_in_node(&self) -> u64 {
        (self.0) & ((1 << 8) - 1)
    }
}

#[derive(Debug)]
pub struct Mem {
    pub mem_id: MemID,
    pub kind: MemKind,
    pub capacity: u64,
    pub insts: BTreeMap<(InstID, OpID), Inst>,
    pub time_points: Vec<MemPoint>,
    pub max_live_insts: u32,
}

impl Mem {
    fn new(mem_id: MemID, kind: MemKind, capacity: u64) -> Self {
        Mem {
            mem_id,
            kind,
            capacity,
            insts: BTreeMap::new(),
            time_points: Vec::new(),
            max_live_insts: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.insts.is_empty()
    }

    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) {
        // BTreeMap::retain requires Rust 1.53
        let mut removed_insts = Vec::new();
        for (key, inst) in self.insts.iter_mut() {
            inst.trim_time_range(start, stop);
            if inst.time_range.was_removed {
                removed_insts.push(*key);
            }
        }
        for op_id in removed_insts {
            self.insts.remove(&op_id);
        }
    }

    fn sort_time_range(&mut self) {
        for (key, inst) in &self.insts {
            self.time_points.push(MemPoint::new(
                inst.time_range.start.unwrap(),
                *key,
                true,
                (),
            ));
            self.time_points.push(MemPoint::new(
                inst.time_range.stop.unwrap(),
                *key,
                false,
                (),
            ))
        }
        self.time_points
            .sort_by(|a, b| a.time_key().cmp(&b.time_key()));

        // Hack: This is a max heap so reverse the values as they go in.
        let mut free_levels = BinaryHeap::<Reverse<u32>>::new();
        for point in &self.time_points {
            if point.first {
                let level = if let Some(level) = free_levels.pop() {
                    level.0
                } else {
                    self.max_live_insts += 1;
                    self.max_live_insts
                };
                self.insts
                    .get_mut(&point.entry)
                    .unwrap()
                    .base
                    .set_level(level);
            } else {
                let level = self.insts.get(&point.entry).unwrap().base.level.unwrap();
                free_levels.push(Reverse(level));
            }
        }
    }
}

#[derive(Debug)]
pub struct MemProcAffinity {
    mem_id: MemID,
    pub proc_ids: Vec<ProcID>,
}

impl MemProcAffinity {
    fn new(mem_id: MemID) -> Self {
        MemProcAffinity {
            mem_id,
            proc_ids: Vec::new(),
        }
    }
    fn add_proc_id(&mut self, proc_id: ProcID) {
        self.proc_ids.push(proc_id);
    }
}

#[derive(Debug, Copy, Clone)]
pub enum ChanEntry {
    Copy(usize),
    Fill(usize),
    DepPart(usize),
}

pub type ChanPoint = TimePoint<ChanEntry, ()>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChanID {
    pub src: Option<MemID>,
    pub dst: Option<MemID>,
}

impl ChanID {
    fn new_copy(src: MemID, dst: MemID) -> Self {
        ChanID {
            src: Some(src),
            dst: Some(dst),
        }
    }
    fn new_fill(dst: MemID) -> Self {
        ChanID {
            src: None,
            dst: Some(dst),
        }
    }
    fn new_deppart() -> Self {
        ChanID {
            src: None,
            dst: None,
        }
    }

    pub fn node_id(&self) -> Option<NodeID> {
        if self.src.is_some() {
            self.src.map(|src| src.node_id())
        } else {
            self.dst.map(|dst| dst.node_id())
        }
    }
}

#[derive(Debug)]
pub struct Chan {
    pub chan_id: ChanID,
    pub copies: Vec<Copy>,
    pub fills: Vec<Fill>,
    pub depparts: Vec<DepPart>,
    pub time_points: Vec<ChanPoint>,
    pub max_levels: u32,
}

impl Chan {
    fn new(chan_id: ChanID) -> Self {
        Chan {
            chan_id,
            copies: Vec::new(),
            fills: Vec::new(),
            depparts: Vec::new(),
            time_points: Vec::new(),
            max_levels: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.copies.is_empty() && self.fills.is_empty() && self.depparts.is_empty()
    }

    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) {
        for copy in &mut self.copies {
            copy.trim_time_range(start, stop);
        }
        self.copies.retain(|t| !t.time_range.was_removed);
        for fill in &mut self.fills {
            fill.trim_time_range(start, stop);
        }
        self.fills.retain(|t| !t.time_range.was_removed);
        for deppart in &mut self.depparts {
            deppart.trim_time_range(start, stop);
        }
        self.depparts.retain(|t| !t.time_range.was_removed);
    }

    fn sort_time_range(&mut self) {
        fn add(time: &TimeRange, entry: ChanEntry, points: &mut Vec<ChanPoint>) {
            let start = time.start.unwrap();
            let stop = time.stop.unwrap();
            points.push(ChanPoint::new(start, entry, true, ()));
            points.push(ChanPoint::new(stop, entry, false, ()));
        }

        let points = &mut self.time_points;

        for (idx, copy) in self.copies.iter().enumerate() {
            let time = &copy.time_range;
            let entry = ChanEntry::Copy(idx);
            add(&time, entry, points);
        }
        for (idx, fill) in self.fills.iter().enumerate() {
            let time = &fill.time_range;
            let entry = ChanEntry::Fill(idx);
            add(&time, entry, points);
        }
        for (idx, deppart) in self.depparts.iter().enumerate() {
            let time = &deppart.time_range;
            let entry = ChanEntry::DepPart(idx);
            add(&time, entry, points);
        }

        points.sort_by(|a, b| a.time_key().cmp(&b.time_key()));

        // Hack: This is a max heap so reverse the values as they go in.
        let mut free_levels = BinaryHeap::<Reverse<u32>>::new();
        for point in points {
            if point.first {
                let level = if let Some(level) = free_levels.pop() {
                    level.0
                } else {
                    self.max_levels += 1;
                    self.max_levels
                };
                match point.entry {
                    ChanEntry::Copy(idx) => self.copies[idx].base.set_level(level),
                    ChanEntry::Fill(idx) => self.fills[idx].base.set_level(level),
                    ChanEntry::DepPart(idx) => self.depparts[idx].base.set_level(level),
                };
            } else {
                let level = match point.entry {
                    ChanEntry::Copy(idx) => self.copies[idx].base.level.unwrap(),
                    ChanEntry::Fill(idx) => self.fills[idx].base.level.unwrap(),
                    ChanEntry::DepPart(idx) => self.depparts[idx].base.level.unwrap(),
                };
                free_levels.push(Reverse(level));
            }
        }
    }

    pub fn entry(&self, entry: ChanEntry) -> (&Base, &TimeRange, &InitiationDependencies) {
        match entry {
            ChanEntry::Copy(idx) => {
                let copy = &self.copies[idx];
                (&copy.base, &copy.time_range, &copy.deps)
            }
            ChanEntry::Fill(idx) => {
                let fill = &self.fills[idx];
                (&fill.base, &fill.time_range, &fill.deps)
            }
            ChanEntry::DepPart(idx) => {
                let deppart = &self.depparts[idx];
                (&deppart.base, &deppart.time_range, &deppart.deps)
            }
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Bounds {
    Point {
        point: Vec<u64>,
        dim: u32,
    },
    Rect {
        lo: Vec<u64>,
        hi: Vec<u64>,
        dim: u32,
    },
    Empty,
    Unknown,
}

#[derive(Debug)]
pub struct ISpaceSize {
    pub dense_size: u64,
    pub sparse_size: u64,
    pub is_sparse: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ISpaceID(pub u64);

#[derive(Debug)]
pub struct ISpace {
    pub ispace_id: ISpaceID,
    pub bounds: Bounds,
    pub name: Option<String>,
    pub parent: Option<IPartID>,
    pub size: Option<ISpaceSize>,
}

impl ISpace {
    fn new(ispace_id: ISpaceID) -> Self {
        ISpace {
            ispace_id,
            bounds: Bounds::Unknown,
            name: None,
            parent: None,
            size: None,
        }
    }

    // Important: these methods can get called multiple times in a
    // sparse instance. In this case the bounds will NOT be
    // accurate. But we don't use bounds in such cases anyway since we
    // refer to the dense/sparse sizes.
    fn set_point(&mut self, dim: u32, values: &Vec<u64>) -> &mut Self {
        let new_bounds = Bounds::Point {
            point: values.clone(),
            dim,
        };
        self.bounds = new_bounds;
        self
    }
    fn set_rect(&mut self, dim: u32, values: &Vec<u64>, max_dim: i32) -> &mut Self {
        let new_bounds = Bounds::Rect {
            lo: values[0..(dim as usize)].to_owned(),
            hi: values[(max_dim as usize)..(max_dim as usize) + (dim as usize)].to_owned(),
            dim,
        };
        self.bounds = new_bounds;
        self
    }
    fn set_empty(&mut self) -> &mut Self {
        let new_bounds = Bounds::Empty;
        self.bounds = new_bounds;
        self
    }

    fn set_name(&mut self, name: &String) -> &mut Self {
        assert!(self.name.is_none());
        self.name = Some(name.to_owned());
        self
    }
    fn set_parent(&mut self, parent: IPartID) -> &mut Self {
        assert!(self.parent.is_none());
        self.parent = Some(parent);
        self
    }
    fn set_size(&mut self, dense_size: u64, sparse_size: u64, is_sparse: bool) -> &mut Self {
        assert!(self.size.is_none());
        self.size = Some(ISpaceSize {
            dense_size,
            sparse_size,
            is_sparse,
        });
        self
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct IPartID(pub u64);

#[derive(Debug)]
pub struct IPart {
    ipart_id: IPartID,
    name: Option<String>,
    pub parent: Option<ISpaceID>,
    disjoint: Option<bool>,
    point0: Option<u64>,
}

impl IPart {
    fn new(ipart_id: IPartID) -> Self {
        IPart {
            ipart_id,
            name: None,
            parent: None,
            disjoint: None,
            point0: None,
        }
    }
    fn set_name(&mut self, name: &String) -> &mut Self {
        assert!(self.name.as_ref().map_or(true, |x| x == name));
        self.name = Some(name.to_owned());
        self
    }
    fn set_parent(&mut self, parent: ISpaceID) -> &mut Self {
        assert!(self.parent.is_none());
        self.parent = Some(parent);
        self
    }
    fn set_disjoint(&mut self, disjoint: bool) -> &mut Self {
        assert!(self.disjoint.is_none());
        self.disjoint = Some(disjoint);
        self
    }
    fn set_point0(&mut self, point0: u64) -> &mut Self {
        assert!(self.point0.is_none());
        self.point0 = Some(point0);
        self
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FSpaceID(pub u64);

#[derive(Debug)]
pub struct FSpace {
    pub fspace_id: FSpaceID,
    pub name: Option<String>,
    pub fields: BTreeMap<FieldID, Field>,
}

impl FSpace {
    fn new(fspace_id: FSpaceID) -> Self {
        FSpace {
            fspace_id,
            name: None,
            fields: BTreeMap::new(),
        }
    }
    fn set_name(&mut self, name: &String) -> &mut Self {
        let new_name = Some(name.to_owned());
        assert!(self.name.is_none() || self.name == new_name);
        self.name = new_name;
        self
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FieldID(pub u32);

#[derive(Debug)]
pub struct Field {
    fspace_id: FSpaceID,
    field_id: FieldID,
    size: u64,
    pub name: String,
}

impl Field {
    fn new(fspace_id: FSpaceID, field_id: FieldID, size: u64, name: &String) -> Self {
        Field {
            fspace_id,
            field_id,
            size,
            name: name.to_owned(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TreeID(pub u32);

#[derive(Debug)]
pub struct Region {
    ispace_id: ISpaceID,
    fspace_id: FSpaceID,
    tree_id: TreeID,
    name: String,
}

impl Region {
    fn new(ispace_id: ISpaceID, fspace_id: FSpaceID, tree_id: TreeID, name: &String) -> Self {
        Region {
            ispace_id,
            fspace_id,
            tree_id,
            name: name.to_owned(),
        }
    }
}

#[derive(Debug)]
pub struct PathRange {
    start: u64,
    stop: u64,
    path: Vec<()>,
}

impl PathRange {
    fn new() -> Self {
        PathRange {
            start: 0,
            stop: 0,
            path: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct Dependencies {
    deps_in: BTreeSet<()>,
    deps_out: BTreeSet<()>,
    deps_parent: BTreeSet<()>,
    deps_children: BTreeSet<()>,
    path: PathRange,
    visited: bool,
}

impl Dependencies {
    fn new() -> Self {
        Dependencies {
            deps_in: BTreeSet::new(),
            deps_out: BTreeSet::new(),
            deps_parent: BTreeSet::new(),
            deps_children: BTreeSet::new(),
            path: PathRange::new(),
            visited: false,
        }
    }
}

#[derive(Debug)]
pub struct InitiationDependencies {
    deps: Dependencies,
    pub op_id: OpID,
}

impl InitiationDependencies {
    fn new(op_id: OpID) -> Self {
        InitiationDependencies {
            deps: Dependencies::new(),
            op_id,
        }
    }
}

#[derive(Debug)]
pub struct NoDependencies {
    deps: Dependencies,
}

impl NoDependencies {
    fn new() -> Self {
        NoDependencies {
            deps: Dependencies::new(),
        }
    }
}

#[derive(Debug)]
pub struct Align {
    field_id: FieldID,
    eqk: u32,
    pub align_desc: u32,
    pub has_align: bool,
}

impl Align {
    fn new(field_id: FieldID, eqk: u32, align_desc: u32, has_align: bool) -> Self {
        Align {
            field_id,
            eqk,
            align_desc,
            has_align,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct InstID(pub u64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Dim(pub u32);

#[derive(Debug)]
pub struct Inst {
    pub base: Base,
    pub inst_id: InstID,
    mem_id: Option<MemID>,
    pub size: Option<u64>,
    pub time_range: TimeRange,
    pub deps: InitiationDependencies,
    pub ispace_ids: Vec<ISpaceID>,
    pub fspace_ids: Vec<FSpaceID>,
    tree_id: Option<TreeID>,
    pub fields: BTreeMap<FSpaceID, Vec<FieldID>>,
    pub align_desc: BTreeMap<FSpaceID, Vec<Align>>,
    pub dim_order: BTreeMap<Dim, DimKind>,
}

impl Inst {
    fn new(base: Base, inst_id: InstID, op_id: OpID) -> Self {
        Inst {
            base,
            inst_id,
            mem_id: None,
            size: None,
            time_range: TimeRange::new_empty(),
            deps: InitiationDependencies::new(op_id),
            ispace_ids: Vec::new(),
            fspace_ids: Vec::new(),
            tree_id: None,
            fields: BTreeMap::new(),
            align_desc: BTreeMap::new(),
            dim_order: BTreeMap::new(),
        }
    }
    fn set_mem(&mut self, mem_id: MemID) -> &mut Self {
        self.mem_id = Some(mem_id);
        self
    }
    fn set_size(&mut self, size: u64) -> &mut Self {
        self.size = Some(size);
        self
    }
    fn set_start_stop(&mut self, start: Timestamp, stop: Timestamp) -> &mut Self {
        self.time_range = TimeRange::new_start(start, stop);
        self
    }
    fn set_start(&mut self, start: Timestamp) -> &mut Self {
        // don't overwrite if we have already captured the (more precise) timeline info
        if self.time_range.stop.is_none() {
            self.time_range.start = Some(start);
        }
        self
    }
    fn add_ispace(&mut self, ispace_id: ISpaceID) -> &mut Self {
        self.ispace_ids.push(ispace_id);
        self
    }
    fn add_fspace(&mut self, fspace_id: FSpaceID) -> &mut Self {
        self.fspace_ids.push(fspace_id);
        self.fields.entry(fspace_id).or_insert_with(|| Vec::new());
        self.align_desc
            .entry(fspace_id)
            .or_insert_with(|| Vec::new());
        self
    }
    fn add_field(&mut self, fspace_id: FSpaceID, field_id: FieldID) -> &mut Self {
        self.fields
            .entry(fspace_id)
            .or_insert_with(|| Vec::new())
            .push(field_id);
        self
    }
    fn add_align_desc(
        &mut self,
        fspace_id: FSpaceID,
        field_id: FieldID,
        eqk: u32,
        align_desc: u32,
        has_align: bool,
    ) -> &mut Self {
        self.align_desc
            .entry(fspace_id)
            .or_insert_with(|| Vec::new())
            .push(Align::new(field_id, eqk, align_desc, has_align));
        self
    }
    fn add_dim_order(&mut self, dim: Dim, dim_kind: DimKind) -> &mut Self {
        self.dim_order.insert(dim, dim_kind);
        self
    }
    fn set_tree(&mut self, tree_id: TreeID) -> &mut Self {
        self.tree_id = Some(tree_id);
        self
    }
    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) {
        self.time_range.trim_time_range(start, stop);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, LowerHex)]
pub struct Color(pub u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MapperCallKindID(pub u32);

#[derive(Debug)]
pub struct MapperCallKind {
    kind: MapperCallKindID,
    pub name: String,
    pub color: Option<Color>,
}

impl MapperCallKind {
    fn new(kind: MapperCallKindID, name: &String) -> Self {
        MapperCallKind {
            kind,
            name: name.to_owned(),
            color: None,
        }
    }
    fn set_color(&mut self, color: Color) -> &mut Self {
        self.color = Some(color);
        self
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RuntimeCallKindID(pub u32);

#[derive(Debug)]
pub struct RuntimeCallKind {
    kind: RuntimeCallKindID,
    pub name: String,
    pub color: Option<Color>,
}

impl RuntimeCallKind {
    fn new(kind: RuntimeCallKindID, name: &String) -> Self {
        RuntimeCallKind {
            kind,
            name: name.to_owned(),
            color: None,
        }
    }
    fn set_color(&mut self, color: Color) -> &mut Self {
        self.color = Some(color);
        self
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TaskID(pub u32);

#[derive(Debug)]
pub struct TaskKind {
    task_id: TaskID,
    pub name: Option<String>,
}

impl TaskKind {
    fn new(task_id: TaskID) -> Self {
        TaskKind {
            task_id,
            name: None,
        }
    }
    fn set_name(&mut self, name: &String, overwrite: bool) {
        if self.name.is_none() || overwrite {
            self.name = Some(name.to_owned());
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct VariantID(pub u32);

#[derive(Debug)]
pub struct Variant {
    variant_id: VariantID,
    pub name: String,
    task_id: Option<TaskID>,
    pub color: Option<Color>,
}

impl Variant {
    fn new(variant_id: VariantID, name: &String) -> Self {
        Variant {
            variant_id,
            name: name.to_owned(),
            task_id: None,
            color: None,
        }
    }
    fn set_task(&mut self, task_id: TaskID) -> &mut Self {
        match self.task_id {
            Some(id) => assert_eq!(id, task_id),
            None => {} // ok
        }
        self.task_id = Some(task_id);
        self
    }
    fn set_color(&mut self, color: Color) -> &mut Self {
        self.color = Some(color);
        self
    }
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct ProfUID(pub u64);

#[derive(Debug)]
pub struct Base {
    pub prof_uid: ProfUID,
    pub level: Option<u32>,
    pub level_ready: Option<u32>,
}

impl Base {
    fn new(allocator: &mut ProfUIDAllocator) -> Self {
        Base {
            prof_uid: allocator.get_prof_uid(),
            level: None,
            level_ready: None,
        }
    }
    fn set_level(&mut self, level: u32) -> &mut Self {
        assert_eq!(self.level, None);
        self.level = Some(level);
        self
    }
    fn set_level_ready(&mut self, level_ready: u32) -> &mut Self {
        assert_eq!(self.level_ready, None);
        self.level_ready = Some(level_ready);
        self
    }
}

#[derive(Debug)]
pub struct TimeRange {
    pub create: Option<Timestamp>,
    pub ready: Option<Timestamp>,
    pub start: Option<Timestamp>,
    pub stop: Option<Timestamp>,
    trimmed: bool,
    was_removed: bool,
}

impl TimeRange {
    fn new_full(create: Timestamp, ready: Timestamp, start: Timestamp, stop: Timestamp) -> Self {
        assert!(create <= ready);
        assert!(ready <= start);
        assert!(start <= stop);
        TimeRange {
            create: Some(create),
            ready: Some(ready),
            start: Some(start),
            stop: Some(stop),
            trimmed: false,
            was_removed: false,
        }
    }
    fn new_start(start: Timestamp, stop: Timestamp) -> Self {
        assert!(start <= stop);
        TimeRange {
            create: None,
            ready: None,
            start: Some(start),
            stop: Some(stop),
            trimmed: false,
            was_removed: false,
        }
    }
    fn new_empty() -> Self {
        TimeRange {
            create: None,
            ready: None,
            start: None,
            stop: None,
            trimmed: false,
            was_removed: false,
        }
    }
    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) {
        let clip = |value| {
            let value = value - start;
            if value < 0.into() {
                0.into()
            } else if value > stop - start {
                stop - start
            } else {
                value
            }
        };

        if self.stop.map_or(false, |x| x < start) || self.start.map_or(false, |x| x > stop) {
            self.was_removed = true;
            return;
        }
        self.create = self.create.map(clip);
        self.ready = self.ready.map(clip);
        self.start = self.start.map(clip);
        self.stop = self.stop.map(clip);
    }
}

#[derive(Debug)]
pub struct WaitInterval {
    pub start: Timestamp,
    pub ready: Timestamp,
    pub end: Timestamp,
}

impl WaitInterval {
    fn new(start: Timestamp, ready: Timestamp, end: Timestamp) -> Self {
        assert!(start <= ready);
        assert!(ready <= end);
        WaitInterval { start, ready, end }
    }
}

#[derive(Debug)]
pub struct Waiters {
    pub wait_intervals: Vec<WaitInterval>,
}

impl Waiters {
    fn new() -> Self {
        Waiters {
            wait_intervals: Vec::new(),
        }
    }
    fn add_wait_interval(&mut self, interval: WaitInterval) -> &mut Self {
        self.wait_intervals.push(interval);
        self
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct OpID(pub u64);

#[derive(Debug)]
pub struct Task {
    pub base: Base,
    pub op_id: OpID,
    pub task_id: TaskID,
    pub variant_id: VariantID,
    pub time_range: TimeRange,
    pub waiters: Waiters,
}

impl Task {
    fn new(
        base: Base,
        op_id: OpID,
        task_id: TaskID,
        variant_id: VariantID,
        time_range: TimeRange,
    ) -> Self {
        Task {
            base,
            op_id,
            task_id,
            variant_id,
            time_range,
            waiters: Waiters::new(),
        }
    }
    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) {
        self.time_range.trim_time_range(start, stop);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct OpKindID(u32);

#[derive(Debug)]
pub struct OpKind {
    pub name: String,
    pub color: Option<Color>,
}

impl OpKind {
    fn new(name: String) -> Self {
        OpKind { name, color: None }
    }
    fn set_color(&mut self, color: Color) -> &mut Self {
        self.color = Some(color);
        self
    }
}

#[derive(Debug)]
pub struct Operation {
    base: Base,
    op_id: OpID,
    pub kind: Option<OpKindID>,
    // owner: Option<OpID>,
}

impl Operation {
    fn new(base: Base, op_id: OpID) -> Self {
        Operation {
            base,
            op_id,
            kind: None,
            // owner: None,
        }
    }
    fn set_kind(&mut self, kind: OpKindID) -> &mut Self {
        assert_eq!(self.kind, None);
        self.kind = Some(kind);
        self
    }
    // fn set_owner(&mut self, owner: OpID) -> &mut Self {
    //     assert_eq!(self.owner, None);
    //     self.owner = Some(owner);
    //     self
    // }
}

#[derive(Debug)]
pub struct MetaTask {
    base: Base,
    op_id: OpID,
    pub variant_id: VariantID,
    time_range: TimeRange,
    pub deps: InitiationDependencies,
    waiters: Waiters,
}

impl MetaTask {
    fn new(base: Base, op_id: OpID, variant_id: VariantID, time_range: TimeRange) -> Self {
        MetaTask {
            base,
            op_id,
            variant_id,
            time_range,
            deps: InitiationDependencies::new(op_id),
            waiters: Waiters::new(),
        }
    }
}

#[derive(Debug)]
pub struct MapperCall {
    pub base: Base,
    pub kind: MapperCallKindID,
    op_id: OpID,
    pub time_range: TimeRange,
    pub deps: InitiationDependencies,
    pub waiters: Waiters,
}

impl MapperCall {
    fn new(base: Base, kind: MapperCallKindID, op_id: OpID, time_range: TimeRange) -> Self {
        MapperCall {
            base,
            kind,
            op_id,
            time_range,
            deps: InitiationDependencies::new(op_id),
            waiters: Waiters::new(),
        }
    }
    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) {
        self.time_range.trim_time_range(start, stop);
    }
}

#[derive(Debug)]
pub struct RuntimeCall {
    pub base: Base,
    pub kind: RuntimeCallKindID,
    pub time_range: TimeRange,
    deps: NoDependencies,
    pub waiters: Waiters,
}

impl RuntimeCall {
    fn new(base: Base, kind: RuntimeCallKindID, time_range: TimeRange) -> Self {
        RuntimeCall {
            base,
            kind,
            time_range,
            deps: NoDependencies::new(),
            waiters: Waiters::new(),
        }
    }
    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) {
        self.time_range.trim_time_range(start, stop);
    }
}

#[derive(Debug)]
pub struct ProfTask {
    base: Base,
    pub op_id: OpID,
    time_range: TimeRange,
    deps: NoDependencies,
    waiters: Waiters,
}

impl ProfTask {
    fn new(base: Base, op_id: OpID, time_range: TimeRange) -> Self {
        ProfTask {
            base,
            op_id,
            time_range,
            deps: NoDependencies::new(),
            waiters: Waiters::new(),
        }
    }
    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) {
        self.time_range.trim_time_range(start, stop);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EventID(pub u64);

#[derive(Debug)]
pub struct CopyInfo {
    src_inst: InstID,
    dst_inst: InstID,
    fevent: EventID,
    num_fields: u32,
    request_type: u32,
    num_hops: u32,
}

impl fmt::Display for CopyInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "src_inst=0x{:x}, dst_inst=0x{:x}, fields={}, type={}, hops={}",
            self.src_inst.0,
            self.dst_inst.0,
            self.num_fields,
            match self.request_type {
                0 => "fill",
                1 => "reduc",
                2 => "copy",
                _ => unreachable!(),
            },
            self.num_hops
        )
    }
}

#[derive(Debug)]
pub struct Copy {
    base: Base,
    src: MemID,
    dst: MemID,
    pub size: u64,
    time_range: TimeRange,
    pub deps: InitiationDependencies,
    fevent: EventID,
    num_requests: u32,
    pub copy_info: Vec<CopyInfo>,
}

impl Copy {
    fn new(
        base: Base,
        src: MemID,
        dst: MemID,
        size: u64,
        op_id: OpID,
        time_range: TimeRange,
        fevent: EventID,
        num_requests: u32,
        copy_info: Vec<CopyInfo>,
    ) -> Self {
        Copy {
            base,
            src,
            dst,
            size,
            time_range,
            deps: InitiationDependencies::new(op_id),
            fevent,
            num_requests,
            copy_info,
        }
    }
    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) {
        self.time_range.trim_time_range(start, stop);
    }
}

#[derive(Debug)]
pub struct Fill {
    base: Base,
    dst: MemID,
    time_range: TimeRange,
    pub deps: InitiationDependencies,
}

impl Fill {
    fn new(base: Base, dst: MemID, op_id: OpID, time_range: TimeRange) -> Self {
        Fill {
            base,
            dst,
            time_range,
            deps: InitiationDependencies::new(op_id),
        }
    }
    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) {
        self.time_range.trim_time_range(start, stop);
    }
}

#[derive(Debug)]
pub struct DepPart {
    base: Base,
    pub part_op: DepPartKind,
    time_range: TimeRange,
    pub deps: InitiationDependencies,
}

impl DepPart {
    fn new(base: Base, part_op: DepPartKind, op_id: OpID, time_range: TimeRange) -> Self {
        DepPart {
            base,
            part_op,
            time_range,
            deps: InitiationDependencies::new(op_id),
        }
    }
    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) {
        self.time_range.trim_time_range(start, stop);
    }
}

fn compute_color(step: u32, num_steps: u32) -> Color {
    assert!(step <= num_steps);
    let h = (step as f64) / (num_steps as f64);
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let q = 1.0 - f;
    let rem = (i as u32) % 6;
    let r;
    let g;
    let b;
    if rem == 0 {
        r = 1.0;
        g = f;
        b = 0.0;
    } else if rem == 1 {
        r = q;
        g = 1.0;
        b = 0.0;
    } else if rem == 2 {
        r = 0.0;
        g = 1.0;
        b = f;
    } else if rem == 3 {
        r = 0.0;
        g = q;
        b = 1.0;
    } else if rem == 4 {
        r = f;
        g = 0.0;
        b = 1.0;
    } else if rem == 5 {
        r = 1.0;
        g = 0.0;
        b = q;
    } else {
        unreachable!();
    }
    let r = (r * 255.0).floor() as u32;
    let g = (g * 255.0).floor() as u32;
    let b = (b * 255.0).floor() as u32;
    Color((r << 16) | (g << 8) | b)
}

#[derive(Debug)]
struct LFSR {
    register: u32,
    bits: u32,
    max_value: u32,
    taps: Vec<u32>,
}

impl LFSR {
    fn new(size: u64) -> Self {
        let needed_bits = (size as f64).log2().floor() as u32 + 1;
        let seed_configuration = 0b1010010011110011;
        LFSR {
            register: (seed_configuration & (((1 << needed_bits) - 1) << (16 - needed_bits)))
                >> (16 - needed_bits),
            bits: needed_bits,
            max_value: 1 << needed_bits,
            taps: match needed_bits {
                2 => vec![2, 1],
                3 => vec![3, 2],
                4 => vec![4, 3],
                5 => vec![5, 3],
                6 => vec![6, 5],
                7 => vec![7, 6],
                8 => vec![8, 6, 5, 4],
                9 => vec![9, 5],
                10 => vec![10, 7],
                11 => vec![11, 9],
                12 => vec![12, 11, 10, 4],
                13 => vec![13, 12, 11, 8],
                14 => vec![14, 13, 12, 2],
                15 => vec![15, 14],
                16 => vec![16, 14, 13, 11],
                _ => unreachable!(), // if we need more than 16 bits that is a lot tasks
            },
        }
    }
    fn next(&mut self) -> u32 {
        let mut xor = 0;
        for t in &self.taps {
            xor += (self.register >> (self.bits - t)) & 1;
        }
        xor = xor & 1;
        self.register = ((self.register >> 1) | (xor << (self.bits - 1))) & ((1 << self.bits) - 1);
        self.register
    }
}

#[derive(Debug, Default)]
struct ProfUIDAllocator {
    next_prof_uid: ProfUID,
}

impl ProfUIDAllocator {
    fn get_prof_uid(&mut self) -> ProfUID {
        self.next_prof_uid.0 += 1;
        self.next_prof_uid
    }
}

#[derive(Debug, Default)]
pub struct State {
    prof_uid_allocator: ProfUIDAllocator,
    max_dim: i32,
    pub procs: BTreeMap<ProcID, Proc>,
    pub mems: BTreeMap<MemID, Mem>,
    pub mem_proc_affinity: BTreeMap<MemID, MemProcAffinity>,
    pub chans: BTreeMap<ChanID, Chan>,
    pub task_kinds: BTreeMap<TaskID, TaskKind>,
    pub variants: BTreeMap<(TaskID, VariantID), Variant>,
    pub meta_variants: BTreeMap<VariantID, Variant>,
    meta_tasks: BTreeMap<(OpID, VariantID), ProcID>,
    pub op_kinds: BTreeMap<OpKindID, OpKind>,
    pub operations: BTreeMap<OpID, Operation>,
    prof_uid_map: BTreeMap<u64, u64>,
    pub tasks: BTreeMap<OpID, ProcID>,
    pub last_time: Timestamp,
    pub mapper_call_kinds: BTreeMap<MapperCallKindID, MapperCallKind>,
    pub runtime_call_kinds: BTreeMap<RuntimeCallKindID, RuntimeCallKind>,
    pub insts: BTreeMap<(InstID, OpID), MemID>,
    pub index_spaces: BTreeMap<ISpaceID, ISpace>,
    pub index_partitions: BTreeMap<IPartID, IPart>,
    logical_regions: BTreeMap<(ISpaceID, FSpaceID, TreeID), Region>,
    pub field_spaces: BTreeMap<FSpaceID, FSpace>,
    copy_map: BTreeMap<EventID, (ChanID, usize)>,
    has_spy_data: bool,
    spy_state: (), // TODO
}

impl State {
    fn create_op(&mut self, op_id: OpID) -> &mut Operation {
        let alloc = &mut self.prof_uid_allocator;
        self.operations
            .entry(op_id)
            .or_insert_with(|| Operation::new(Base::new(alloc), op_id))
    }

    pub fn find_op(&self, op_id: OpID) -> Option<&Operation> {
        self.operations.get(&op_id)
    }

    fn create_task(
        &mut self,
        op_id: OpID,
        proc_id: ProcID,
        task_id: TaskID,
        variant_id: VariantID,
        time_range: TimeRange,
    ) -> &mut Task {
        self.create_op(op_id);
        self.tasks.insert(op_id, proc_id);
        let alloc = &mut self.prof_uid_allocator;
        self.procs
            .get_mut(&proc_id)
            .unwrap()
            .tasks
            .entry(op_id)
            .or_insert_with(|| Task::new(Base::new(alloc), op_id, task_id, variant_id, time_range))
    }

    fn find_task_mut(&mut self, op_id: OpID) -> Option<&mut Task> {
        self.create_op(op_id);
        self.procs
            .get_mut(self.tasks.get(&op_id)?)?
            .tasks
            .get_mut(&op_id)
    }

    pub fn find_task(&self, op_id: OpID) -> Option<&Task> {
        self.procs.get(self.tasks.get(&op_id)?)?.tasks.get(&op_id)
    }

    fn create_meta(
        &mut self,
        op_id: OpID,
        variant_id: VariantID,
        proc_id: ProcID,
        time_range: TimeRange,
    ) -> &mut MetaTask {
        self.create_op(op_id);
        self.meta_tasks.insert((op_id, variant_id), proc_id);
        let tasks = self
            .procs
            .get_mut(&proc_id)
            .unwrap()
            .meta_tasks
            .entry((op_id, variant_id))
            .or_insert_with(|| Vec::new());
        let alloc = &mut self.prof_uid_allocator;
        tasks.push(MetaTask::new(
            Base::new(alloc),
            op_id,
            variant_id,
            time_range,
        ));
        tasks.last_mut().unwrap()
    }

    fn find_meta_mut(&mut self, op_id: OpID, variant_id: VariantID) -> Option<&mut MetaTask> {
        self.procs
            .get_mut(self.meta_tasks.get(&(op_id, variant_id))?)?
            .meta_tasks
            .get_mut(&(op_id, variant_id))?
            .last_mut()
    }

    fn create_mapper_call(
        &mut self,
        kind: MapperCallKindID,
        proc_id: ProcID,
        op_id: OpID,
        time_range: TimeRange,
    ) {
        self.create_op(op_id);
        let alloc = &mut self.prof_uid_allocator;
        self.procs
            .get_mut(&proc_id)
            .unwrap()
            .mapper_calls
            .push(MapperCall::new(Base::new(alloc), kind, op_id, time_range));
    }

    fn create_runtime_call(
        &mut self,
        kind: RuntimeCallKindID,
        proc_id: ProcID,
        time_range: TimeRange,
    ) {
        let alloc = &mut self.prof_uid_allocator;
        self.procs
            .get_mut(&proc_id)
            .unwrap()
            .runtime_calls
            .push(RuntimeCall::new(Base::new(alloc), kind, time_range));
    }

    fn create_prof_task(&mut self, proc_id: ProcID, op_id: OpID, time_range: TimeRange) {
        let alloc = &mut self.prof_uid_allocator;
        self.procs
            .get_mut(&proc_id)
            .unwrap()
            .prof_tasks
            .push(ProfTask::new(Base::new(alloc), op_id, time_range));
    }

    fn create_copy(
        &mut self,
        op_id: OpID,
        src: MemID,
        dst: MemID,
        size: u64,
        time_range: TimeRange,
        fevent: EventID,
        num_requests: u32,
    ) {
        self.create_op(op_id);
        let base = Base::new(&mut self.prof_uid_allocator); // FIXME: construct here to avoid mutability conflict

        let chan_id = ChanID::new_copy(src, dst);
        let chan = self.find_chan_mut(chan_id);

        let copy_id = chan.copies.len();
        chan.copies.push(Copy::new(
            base,
            src,
            dst,
            size,
            op_id,
            time_range,
            fevent,
            num_requests,
            Vec::new(),
        ));

        self.copy_map
            .entry(fevent)
            .or_insert_with(|| (chan_id, copy_id));
    }

    fn find_copy_mut(&mut self, fevent: EventID) -> Option<&mut Copy> {
        let (chan_id, copy_idx) = *self.copy_map.get(&fevent)?;
        Some(&mut self.find_chan_mut(chan_id).copies[copy_idx])
    }

    fn create_fill(&mut self, op_id: OpID, dst: MemID, time_range: TimeRange) {
        self.create_op(op_id);
        let base = Base::new(&mut self.prof_uid_allocator); // FIXME: construct here to avoid mutability conflict
        let chan = self.find_fill_chan_mut(dst);
        chan.fills.push(Fill::new(base, dst, op_id, time_range));
    }

    fn create_deppart(&mut self, op_id: OpID, part_op: DepPartKind, time_range: TimeRange) {
        self.create_op(op_id);
        let base = Base::new(&mut self.prof_uid_allocator); // FIXME: construct here to avoid mutability conflict
        let chan = self.find_deppart_chan_mut();
        chan.depparts
            .push(DepPart::new(base, part_op, op_id, time_range));
    }

    fn find_chan_mut(&mut self, chan_id: ChanID) -> &mut Chan {
        self.chans
            .entry(chan_id)
            .or_insert_with(|| Chan::new(chan_id))
    }

    fn find_fill_chan_mut(&mut self, dst: MemID) -> &mut Chan {
        let chan_id = ChanID::new_fill(dst);
        self.chans
            .entry(chan_id)
            .or_insert_with(|| Chan::new(chan_id))
    }

    fn find_deppart_chan_mut(&mut self) -> &mut Chan {
        let chan_id = ChanID::new_deppart();
        self.chans
            .entry(chan_id)
            .or_insert_with(|| Chan::new(chan_id))
    }

    fn create_inst<'a>(
        &'a mut self,
        inst_id: InstID,
        op_id: OpID,
        insts: &'a mut BTreeMap<(InstID, OpID), Inst>,
    ) -> &'a mut Inst {
        self.create_op(op_id);
        let alloc = &mut self.prof_uid_allocator;
        insts
            .entry((inst_id, op_id))
            .or_insert_with(|| Inst::new(Base::new(alloc), inst_id, op_id))
    }

    fn find_index_space_mut(&mut self, ispace_id: ISpaceID) -> &mut ISpace {
        self.index_spaces
            .entry(ispace_id)
            .or_insert_with(|| ISpace::new(ispace_id))
    }

    fn find_index_partition_mut(&mut self, ipart_id: IPartID) -> &mut IPart {
        self.index_partitions
            .entry(ipart_id)
            .or_insert_with(|| IPart::new(ipart_id))
    }

    fn find_field_space_mut(&mut self, fspace_id: FSpaceID) -> &mut FSpace {
        self.field_spaces
            .entry(fspace_id)
            .or_insert_with(|| FSpace::new(fspace_id))
    }

    fn update_last_time(&mut self, value: Timestamp) {
        self.last_time = max(value, self.last_time);
    }

    pub fn process_records(&mut self, records: &Vec<Record>) {
        // We need a separate table here because instances can't be
        // immediately linked to their associated memory from the
        // logs. Therefore we defer this process until all records
        // have been processed.
        let mut insts = BTreeMap::new();
        for record in records {
            process_record(record, self, &mut insts);
        }
        for (key, inst) in insts {
            if let Some(mem_id) = inst.mem_id {
                let mem = self.mems.get_mut(&mem_id).unwrap();
                mem.insts.insert(key, inst);
            } else {
                unreachable!();
            }
        }
    }

    pub fn trim_time_range(&mut self, start: Option<Timestamp>, stop: Option<Timestamp>) {
        if start.is_none() && stop.is_none() {
            return;
        }
        let start = start.unwrap_or(0.into());
        let stop = stop.unwrap_or(self.last_time);

        assert!(start <= stop);
        assert!(start >= 0.into());
        assert!(stop <= self.last_time);

        for proc in self.procs.values_mut() {
            proc.trim_time_range(start, stop);
        }
        for mem in self.mems.values_mut() {
            mem.trim_time_range(start, stop);
        }
        for chan in self.chans.values_mut() {
            chan.trim_time_range(start, stop);
        }

        self.last_time = stop - start;
    }

    pub fn sort_time_range(&mut self) {
        self.procs
            .par_iter_mut()
            .for_each(|(_, proc)| proc.sort_time_range());
        self.mems
            .par_iter_mut()
            .for_each(|(_, mem)| mem.sort_time_range());
        self.chans
            .par_iter_mut()
            .for_each(|(_, chan)| chan.sort_time_range());
    }

    pub fn assign_colors(&mut self) {
        let num_colors = (self.variants.len()
            + self.meta_variants.len()
            + self.op_kinds.len()
            + self.mapper_call_kinds.len()
            + self.runtime_call_kinds.len()) as u64;
        let mut lfsr = LFSR::new(num_colors);
        let num_colors = lfsr.max_value;
        for variant in self.variants.values_mut() {
            variant.set_color(compute_color(lfsr.next(), num_colors));
        }
        for variant in self.meta_variants.values_mut() {
            variant.set_color(match variant.variant_id.0 {
                1 => Color(0x006600), // Remote message => Evergreen
                2 => Color(0x333399), // Post-Execution => Deep Purple
                6 => Color(0x990000), // Garbage Collection => Crimson
                7 => Color(0x0000FF), // Logical Dependence Analysis => Duke Blue
                8 => Color(0x009900), // Operation Physical Analysis => Green
                9 => Color(0x009900), // Task Physical Analysis => Green
                _ => compute_color(lfsr.next(), num_colors),
            });
        }
        for op_kind in self.op_kinds.values_mut() {
            op_kind.set_color(compute_color(lfsr.next(), num_colors));
        }
        for kind in self.mapper_call_kinds.values_mut() {
            kind.set_color(compute_color(lfsr.next(), num_colors));
        }
        for kind in self.runtime_call_kinds.values_mut() {
            kind.set_color(compute_color(lfsr.next(), num_colors));
        }
    }
}

fn process_record(record: &Record, state: &mut State, insts: &mut BTreeMap<(InstID, OpID), Inst>) {
    match record {
        Record::MapperCallDesc { kind, name } => {
            state
                .mapper_call_kinds
                .entry(*kind)
                .or_insert_with(|| MapperCallKind::new(*kind, name));
        }
        Record::RuntimeCallDesc { kind, name } => {
            state
                .runtime_call_kinds
                .entry(*kind)
                .or_insert_with(|| RuntimeCallKind::new(*kind, name));
        }
        Record::MetaDesc { kind, message, ordered_vc, name } => {
            state
                .meta_variants
                .entry(*kind)
                .or_insert_with(|| Variant::new(*kind, name));
        }
        Record::OpDesc { kind, name } => {
            let kind = OpKindID(*kind);
            state
                .op_kinds
                .entry(kind)
                .or_insert_with(|| OpKind::new(name.clone()));
        }
        Record::ProcDesc { proc_id, kind } => {
            let kind = match ProcKind::try_from(*kind) {
                Ok(x) => x,
                Err(_) => panic!("bad processor kind"),
            };
            state
                .procs
                .entry(*proc_id)
                .or_insert_with(|| Proc::new(*proc_id, kind));
        }
        Record::MaxDimDesc { max_dim } => {
            state.max_dim = *max_dim;
        }
        Record::MemDesc {
            mem_id,
            kind,
            capacity,
        } => {
            let kind = match MemKind::try_from(*kind) {
                Ok(x) => x,
                Err(_) => panic!("bad memory kind"),
            };
            state
                .mems
                .entry(*mem_id)
                .or_insert_with(|| Mem::new(*mem_id, kind, *capacity));
        }
        Record::ProcMDesc { proc_id, mem_id } => {
            state
                .mem_proc_affinity
                .entry(*mem_id)
                .or_insert_with(|| MemProcAffinity::new(*mem_id))
                .add_proc_id(*proc_id);
        }
        Record::IndexSpacePointDesc {
            ispace_id,
            dim,
            rem,
        } => {
            state
                .find_index_space_mut(*ispace_id)
                .set_point(*dim, &rem.0);
        }
        Record::IndexSpaceRectDesc {
            ispace_id,
            dim,
            rem,
        } => {
            let max_dim = state.max_dim;
            state
                .find_index_space_mut(*ispace_id)
                .set_rect(*dim, &rem.0, max_dim);
        }
        Record::IndexSpaceEmptyDesc { ispace_id } => {
            state.find_index_space_mut(*ispace_id).set_empty();
        }
        Record::FieldDesc {
            fspace_id,
            field_id,
            size,
            name,
        } => {
            state
                .find_field_space_mut(*fspace_id)
                .fields
                .entry(*field_id)
                .or_insert_with(|| Field::new(*fspace_id, *field_id, *size, name));
        }
        Record::FieldSpaceDesc { fspace_id, name } => {
            state.find_field_space_mut(*fspace_id).set_name(name);
        }
        Record::PartDesc { unique_id, name } => {
            state.find_index_partition_mut(*unique_id).set_name(name);
        }
        Record::IndexSpaceDesc { ispace_id, name } => {
            state.find_index_space_mut(*ispace_id).set_name(name);
        }
        Record::IndexSubSpaceDesc {
            parent_id,
            ispace_id,
        } => {
            state
                .find_index_space_mut(*ispace_id)
                .set_parent(*parent_id);
        }
        Record::IndexPartitionDesc {
            parent_id,
            unique_id,
            disjoint,
            point0,
        } => {
            state.find_index_space_mut(*parent_id);
            state
                .find_index_partition_mut(*unique_id)
                .set_parent(*parent_id)
                .set_disjoint(*disjoint)
                .set_point0(*point0);
        }
        Record::IndexSpaceSizeDesc {
            ispace_id,
            dense_size,
            sparse_size,
            is_sparse,
        } => {
            state
                .find_index_space_mut(*ispace_id)
                .set_size(*dense_size, *sparse_size, *is_sparse);
        }
        Record::LogicalRegionDesc {
            ispace_id,
            fspace_id,
            tree_id,
            name,
        } => {
            let fspace_id = FSpaceID(*fspace_id as u64);
            state.find_field_space_mut(fspace_id);
            state
                .logical_regions
                .entry((*ispace_id, fspace_id, *tree_id))
                .or_insert_with(|| Region::new(*ispace_id, fspace_id, *tree_id, name));
        }
        Record::PhysicalInstRegionDesc {
            op_id,
            inst_id,
            ispace_id,
            fspace_id,
            tree_id,
        } => {
            let fspace_id = FSpaceID(*fspace_id as u64);
            state.find_field_space_mut(fspace_id);
            state
                .create_inst(*inst_id, *op_id, insts)
                .add_ispace(*ispace_id)
                .add_fspace(fspace_id)
                .set_tree(*tree_id);
        }
        Record::PhysicalInstLayoutDesc {
            op_id,
            inst_id,
            field_id,
            fspace_id,
            has_align,
            eqk,
            align_desc,
        } => {
            let fspace_id = FSpaceID(*fspace_id as u64);
            state.find_field_space_mut(fspace_id);
            state
                .create_inst(*inst_id, *op_id, insts)
                .add_field(fspace_id, *field_id)
                .add_align_desc(fspace_id, *field_id, *eqk, *align_desc, *has_align);
        }
        Record::PhysicalInstDimOrderDesc {
            op_id,
            inst_id,
            dim,
            dim_kind,
        } => {
            let dim = Dim(*dim);
            let dim_kind = match DimKind::try_from(*dim_kind) {
                Ok(x) => x,
                Err(_) => unreachable!("bad dim kind"),
            };
            state
                .create_inst(*inst_id, *op_id, insts)
                .add_dim_order(dim, dim_kind);
        }
        Record::TaskKind {
            task_id,
            name,
            overwrite,
        } => {
            state
                .task_kinds
                .entry(*task_id)
                .or_insert_with(|| TaskKind::new(*task_id))
                .set_name(name, *overwrite);
        }
        Record::TaskVariant {
            task_id,
            variant_id,
            name,
        } => {
            state
                .variants
                .entry((*task_id, *variant_id))
                .or_insert_with(|| Variant::new(*variant_id, name))
                .set_task(*task_id);
        }
        Record::OperationInstance { op_id, kind } => {
            let kind = OpKindID(*kind);
            state.create_op(*op_id).set_kind(kind);
        }
        Record::MultiTask { op_id, .. } => {
            state.create_op(*op_id);
            // .set_op_impl(OpImpl::Multi(Multi::new(*task_id)));
        }
        Record::SliceOwner { parent_id, op_id } => {
            let parent_id = OpID(*parent_id);
            state.create_op(parent_id);
            state.create_op(*op_id); //.set_owner(parent_id);
        }
        Record::TaskWaitInfo {
            op_id,
            wait_start: start,
            wait_ready: ready,
            wait_end: end,
            ..
        } => {
            state
                .find_task_mut(*op_id)
                .unwrap()
                .waiters
                .add_wait_interval(WaitInterval::new(*start, *ready, *end));
        }
        Record::MetaWaitInfo {
            op_id,
            lg_id,
            wait_start: start,
            wait_ready: ready,
            wait_end: end,
        } => {
            state.create_op(*op_id);
            state
                .find_meta_mut(*op_id, *lg_id)
                .unwrap()
                .waiters
                .add_wait_interval(WaitInterval::new(*start, *ready, *end));
        }
        Record::TaskInfo {
            op_id,
            task_id,
            variant_id,
            proc_id,
            create,
            ready,
            start,
            stop,
        } => {
            let time_range = TimeRange::new_full(*create, *ready, *start, *stop);
            state.create_task(*op_id, *proc_id, *task_id, *variant_id, time_range);
            state.update_last_time(*stop);
        }
        Record::GPUTaskInfo {
            op_id,
            task_id,
            variant_id,
            proc_id,
            create,
            ready,
            gpu_start,
            gpu_stop,
            ..
        } => {
            let time_range = TimeRange::new_full(*create, *ready, *gpu_start, *gpu_stop);
            state.create_task(*op_id, *proc_id, *task_id, *variant_id, time_range);
            state.update_last_time(*gpu_stop);
        }
        Record::MetaInfo {
            op_id,
            lg_id,
            proc_id,
            create,
            ready,
            start,
            stop,
        } => {
            let time_range = TimeRange::new_full(*create, *ready, *start, *stop);
            state.create_meta(*op_id, *lg_id, *proc_id, time_range);
            state.update_last_time(*stop);
        }
        Record::CopyInfo {
            op_id,
            src,
            dst,
            size,
            create,
            ready,
            start,
            stop,
            fevent,
            num_requests,
        } => {
            let time_range = TimeRange::new_full(*create, *ready, *start, *stop);
            state.create_copy(
                *op_id,
                *src,
                *dst,
                *size,
                time_range,
                *fevent,
                *num_requests,
            );
            state.update_last_time(*stop);
        }
        Record::CopyInstInfo {
            src_inst,
            dst_inst,
            fevent,
            num_fields,
            request_type,
            num_hops,
            ..
        } => {
            let copy_info = CopyInfo {
                src_inst: *src_inst,
                dst_inst: *dst_inst,
                fevent: *fevent,
                num_fields: *num_fields,
                request_type: *request_type,
                num_hops: *num_hops,
            };
            state
                .find_copy_mut(*fevent)
                .unwrap()
                .copy_info
                .push(copy_info);
        }
        Record::FillInfo {
            op_id,
            dst,
            create,
            ready,
            start,
            stop,
        } => {
            let time_range = TimeRange::new_full(*create, *ready, *start, *stop);
            state.create_fill(*op_id, *dst, time_range);
            state.update_last_time(*stop);
        }
        Record::InstCreateInfo {
            op_id,
            inst_id,
            create,
        } => {
            state
                .create_inst(*inst_id, *op_id, insts)
                .set_start(*create);
        }
        Record::InstUsageInfo {
            op_id,
            inst_id,
            mem_id,
            size,
        } => {
            state
                .insts
                .entry((*inst_id, *op_id))
                .or_insert_with(|| *mem_id);
            state
                .create_inst(*inst_id, *op_id, insts)
                .set_mem(*mem_id)
                .set_size(*size);
        }
        Record::InstTimelineInfo {
            op_id,
            inst_id,
            create,
            destroy,
        } => {
            state
                .create_inst(*inst_id, *op_id, insts)
                .set_start_stop(*create, *destroy);
            state.update_last_time(*destroy);
        }
        Record::PartitionInfo {
            op_id,
            part_op,
            create,
            ready,
            start,
            stop,
        } => {
            let part_op = match DepPartKind::try_from(*part_op) {
                Ok(x) => x,
                Err(_) => panic!("bad deppart kind"),
            };
            let time_range = TimeRange::new_full(*create, *ready, *start, *stop);
            state.create_deppart(*op_id, part_op, time_range);
            state.update_last_time(*stop);
        }
        Record::MapperCallInfo {
            kind,
            op_id,
            start,
            stop,
            proc_id,
        } => {
            assert!(state.mapper_call_kinds.contains_key(kind));
            assert!(*start <= *stop);
            // For now we'll only add very expensive mapper calls (more than 100 us)
            if *stop - *start >= Timestamp::from_us(100) {
                let time_range = TimeRange::new_start(*start, *stop);
                state.create_mapper_call(*kind, *proc_id, *op_id, time_range);
                state.update_last_time(*stop);
            };
        }
        Record::RuntimeCallInfo {
            kind,
            start,
            stop,
            proc_id,
        } => {
            assert!(state.runtime_call_kinds.contains_key(kind));
            let time_range = TimeRange::new_start(*start, *stop);
            state.create_runtime_call(*kind, *proc_id, time_range);
            state.update_last_time(*stop);
        }
        Record::ProfTaskInfo {
            proc_id,
            op_id,
            start,
            stop,
        } => {
            let time_range = TimeRange::new_start(*start, *stop);
            state.create_prof_task(*proc_id, *op_id, time_range);
            state.update_last_time(*stop);
        }
    }
}
