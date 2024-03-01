use std::cmp::{max, Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::convert::TryFrom;
use std::fmt;
use std::sync::OnceLock;

use derive_more::{Add, AddAssign, From, LowerHex, Sub, SubAssign};
use num_enum::TryFromPrimitive;

use rayon::prelude::*;

use serde::Serialize;

use slice_group_by::GroupBy;

use crate::backend::common::{CopyInstInfoVec, FillInstInfoVec, InstPretty, SizePretty};
use crate::num_util::Postincrement;
use crate::serialize::Record;
use crate::spy;

const TASK_GRANULARITY_THRESHOLD: Timestamp = Timestamp::from_us(10);

#[derive(Debug, Clone)]
pub enum Records {
    Prof(Vec<Record>),
    Spy(Vec<spy::serialize::Record>),
}

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
    GPUManaged = 13,
    GPUDynamic = 14,
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

// the class used to save configurations
#[derive(Debug, PartialEq)]
pub struct Config {
    filter_input: bool,
    verbose: bool,
    all_logs: bool,
}

// CONFIG can be only accessed by Config::name_of_the_member()
static CONFIG: OnceLock<Config> = OnceLock::new();

impl Config {
    // this function can be only called once, and it will be called in main
    pub fn set_config(filter_input: bool, verbose: bool, all_logs: bool) {
        let config = Config {
            filter_input,
            verbose,
            all_logs,
        };
        assert_eq!(CONFIG.set(config), Ok(()));
    }
    // return the singleton of CONFIG, usually we do not need to call it unless
    // we want to retrieve multiple members from the CONFIG
    pub fn global() -> &'static Config {
        let config = CONFIG.get();
        config.expect("config was not set")
    }
    pub fn filter_input() -> bool {
        let config = Config::global();
        config.filter_input
    }
    pub fn verbose() -> bool {
        let config = Config::global();
        config.verbose
    }
    pub fn all_logs() -> bool {
        let config = Config::global();
        config.all_logs
    }
}

#[macro_export]
macro_rules! conditional_assert {
    ($cond:expr, $mode:expr, $($arg:tt)*) => (
        if !$cond {
            if $mode {
                panic!("Error: {}", format_args!($($arg)*));
            } else {
                if Config::verbose() {
                    eprintln!("Warning: {}", format_args!($($arg)*));
                }
            }
        }
    )
}

#[derive(
    Debug,
    Copy,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Default,
    Serialize,
    Add,
    Sub,
    AddAssign,
    SubAssign,
    From,
)]
pub struct Timestamp(pub u64 /* ns */);

impl Timestamp {
    pub const MAX: Timestamp = Timestamp(std::u64::MAX);
    pub const MIN: Timestamp = Timestamp(std::u64::MIN);
    pub const fn from_us(microseconds: u64) -> Timestamp {
        Timestamp(microseconds * 1000)
    }
    pub fn to_us(&self) -> f64 {
        self.0 as f64 / 1000.0
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

#[derive(
    Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Add, Sub, From,
)]
pub struct TimestampDelta(pub i64 /* ns */);

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
            secondary_sort_key,
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

// Common methods that apply to Proc, Mem, Chan
pub trait Container {
    type E: std::marker::Copy + std::fmt::Debug;
    type S: std::marker::Copy + std::fmt::Debug;
    type Entry: ContainerEntry;

    fn max_levels(&self) -> usize;
    fn time_points(&self) -> &Vec<TimePoint<Self::E, Self::S>>;
    fn entry(&self, entry: Self::E) -> &Self::Entry;
    fn entry_mut(&mut self, entry: Self::E) -> &mut Self::Entry;
}

// Common methods that apply to ProcEntry, MemEntry, ChanEntry
pub trait ContainerEntry {
    fn base(&self) -> &Base;
    fn base_mut(&mut self) -> &mut Base;
    fn time_range(&self) -> TimeRange;
    fn time_range_mut(&mut self) -> &mut TimeRange;
    fn waiters(&self) -> Option<&Waiters>;
    fn initiation(&self) -> Option<OpID>;
    fn creator(&self) -> Option<EventID>;

    // Methods that require State access
    fn name(&self, state: &State) -> String;
    fn color(&self, state: &State) -> Color;
    fn provenance<'a, 'b>(&'a self, state: &'b State) -> Option<&'b str>;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProcEntryKind {
    Task(TaskID, VariantID),
    MetaTask(VariantID),
    MapperCall(MapperCallKindID),
    RuntimeCall(RuntimeCallKindID),
    GPUKernel(TaskID, VariantID),
    ProfTask,
}

#[derive(Debug)]
pub struct ProcEntry {
    pub base: Base,
    pub op_id: Option<OpID>,
    pub initiation_op: Option<OpID>,
    pub kind: ProcEntryKind,
    pub time_range: TimeRange,
    pub creator: EventID,
    pub fevent: EventID,
    pub waiters: Waiters,
    pub subcalls: Vec<(ProfUID, Timestamp, Timestamp)>,
}

impl ProcEntry {
    fn new(
        base: Base,
        op_id: Option<OpID>,
        initiation_op: Option<OpID>,
        kind: ProcEntryKind,
        time_range: TimeRange,
        creator: EventID,
        fevent: EventID,
    ) -> Self {
        ProcEntry {
            base,
            op_id,
            initiation_op,
            kind,
            time_range,
            creator,
            fevent,
            waiters: Waiters::new(),
            subcalls: Vec::new(),
        }
    }
    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) -> bool {
        self.time_range.trim_time_range(start, stop)
    }
}

impl ContainerEntry for ProcEntry {
    fn base(&self) -> &Base {
        &self.base
    }

    fn base_mut(&mut self) -> &mut Base {
        &mut self.base
    }

    fn time_range(&self) -> TimeRange {
        self.time_range
    }

    fn time_range_mut(&mut self) -> &mut TimeRange {
        &mut self.time_range
    }

    fn waiters(&self) -> Option<&Waiters> {
        Some(&self.waiters)
    }

    fn initiation(&self) -> Option<OpID> {
        self.initiation_op
    }

    fn creator(&self) -> Option<EventID> {
        Some(self.creator)
    }

    fn name(&self, state: &State) -> String {
        let (op_id, initiation_op) = (self.op_id, self.initiation_op);

        match self.kind {
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
            ProcEntryKind::GPUKernel(task_id, variant_id) => {
                let task_name = &state.task_kinds.get(&task_id).unwrap().name;
                let variant_name = &state.variants.get(&(task_id, variant_id)).unwrap().name;
                match task_name {
                    Some(task_name) => {
                        if task_name != variant_name {
                            format!(
                                "GPU Kernel for {} [{}] <{}>",
                                task_name,
                                variant_name,
                                op_id.unwrap().0
                            )
                        } else {
                            format!("GPU Kernel for {} <{}>", task_name, op_id.unwrap().0)
                        }
                    }
                    None => format!("GPU Kernel for {}", variant_name.clone()),
                }
            }
            ProcEntryKind::ProfTask => {
                format!("ProfTask <{:?}>", initiation_op.unwrap().0)
            }
        }
    }

    fn color(&self, state: &State) -> Color {
        match self.kind {
            ProcEntryKind::Task(task_id, variant_id)
            | ProcEntryKind::GPUKernel(task_id, variant_id) => state
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
        }
    }

    fn provenance<'a, 'b>(&'a self, state: &'b State) -> Option<&'b str> {
        if let Some(op_id) = self.op_id {
            return state.find_op_provenance(op_id);
        }
        None
    }
}

pub type ProcPoint = TimePoint<ProfUID, u64>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, LowerHex)]
pub struct ProcID(pub u64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
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
    entries: BTreeMap<ProfUID, ProcEntry>,
    tasks: BTreeMap<OpID, ProfUID>,
    meta_tasks: BTreeMap<(OpID, VariantID), Vec<ProfUID>>,
    pub max_levels: u32,
    pub max_levels_ready: u32,
    pub time_points: Vec<ProcPoint>,
    pub util_time_points: Vec<ProcPoint>,
    visible: bool,
}

impl Proc {
    fn new(proc_id: ProcID, kind: ProcKind) -> Self {
        Proc {
            proc_id,
            kind,
            entries: BTreeMap::new(),
            tasks: BTreeMap::new(),
            meta_tasks: BTreeMap::new(),
            max_levels: 0,
            max_levels_ready: 0,
            time_points: Vec::new(),
            util_time_points: Vec::new(),
            visible: true,
        }
    }

    fn create_proc_entry(
        &mut self,
        base: Base,
        op: Option<OpID>,
        initiation_op: Option<OpID>,
        kind: ProcEntryKind,
        time_range: TimeRange,
        creator: EventID,
        fevent: EventID,
        op_prof_uid: &mut BTreeMap<OpID, ProfUID>,
        prof_uid_proc: &mut BTreeMap<ProfUID, ProcID>,
        fevents: &mut BTreeMap<EventID, ProfUID>,
    ) -> &mut ProcEntry {
        if let Some(op_id) = op {
            op_prof_uid.insert(op_id, base.prof_uid);
        }
        prof_uid_proc.insert(base.prof_uid, self.proc_id);
        // Insert the fevents for tasks into the data structure
        match kind {
            ProcEntryKind::Task(_, _) | ProcEntryKind::MetaTask(_) | ProcEntryKind::ProfTask => {
                // We should only see an event once
                assert!(!fevents.contains_key(&fevent));
                fevents.insert(fevent, base.prof_uid);
            }
            _ => {}
        }
        match kind {
            ProcEntryKind::Task(_, _) => {
                self.tasks.insert(op.unwrap(), base.prof_uid);
            }
            ProcEntryKind::MetaTask(variant_id) => {
                self.meta_tasks
                    .entry((initiation_op.unwrap(), variant_id))
                    .or_insert_with(Vec::new)
                    .push(base.prof_uid);
            }
            // If we don't need to look up later... don't bother building the index
            _ => {}
        }
        self.entries.entry(base.prof_uid).or_insert_with(|| {
            ProcEntry::new(base, op, initiation_op, kind, time_range, creator, fevent)
        })
    }

    pub fn find_task(&self, op_id: OpID) -> Option<&ProcEntry> {
        let prof_uid = self.tasks.get(&op_id)?;
        self.entries.get(prof_uid)
    }

    pub fn find_task_mut(&mut self, op_id: OpID) -> Option<&mut ProcEntry> {
        let prof_uid = self.tasks.get(&op_id)?;
        self.entries.get_mut(prof_uid)
    }

    pub fn find_last_meta(&self, op_id: OpID, variant_id: VariantID) -> Option<&ProcEntry> {
        let prof_uid = self.meta_tasks.get(&(op_id, variant_id))?.last()?;
        self.entries.get(prof_uid)
    }

    pub fn find_last_meta_mut(
        &mut self,
        op_id: OpID,
        variant_id: VariantID,
    ) -> Option<&mut ProcEntry> {
        let prof_uid = self.meta_tasks.get(&(op_id, variant_id))?.last()?;
        self.entries.get_mut(prof_uid)
    }

    pub fn find_entry(&self, prof_uid: ProfUID) -> Option<&ProcEntry> {
        self.entries.get(&prof_uid)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = &ProcEntry> {
        self.entries.values()
    }

    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) {
        self.entries.retain(|_, t| !t.trim_time_range(start, stop));
    }

    fn sort_calls_and_waits(&mut self, fevents: &BTreeMap<EventID, ProfUID>) {
        // Before we sort things, we need to rearrange the waiters from
        // any tasks into the appropriate runtime/mapper calls and make the
        // runtime/mapper calls appear as waiters in the original tasks
        let mut subcalls = BTreeMap::new();
        for (uid, entry) in self.entries.iter() {
            match entry.kind {
                ProcEntryKind::MapperCall(_) | ProcEntryKind::RuntimeCall(_) => {
                    let task_uid = fevents.get(&entry.fevent).unwrap();
                    let call_start = entry.time_range.start.unwrap();
                    let call_stop = entry.time_range.stop.unwrap();
                    assert!(call_start <= call_stop);
                    subcalls
                        .entry(*task_uid)
                        .or_insert_with(Vec::new)
                        .push((*uid, call_start, call_stop));
                }
                _ => {}
            }
        }
        for (task_uid, calls) in subcalls.iter_mut() {
            // Remove the old entry from the map to keep the borrow checker happy
            let mut task_entry = self.entries.remove(&task_uid).unwrap();
            // Sort subcalls by their size from smallest to largest
            calls.sort_by_key(|a| a.2 - a.1);
            // Push waits into the smallest subcall we can find
            let mut to_remove = Vec::new();
            for (idx, wait) in task_entry.waiters.wait_intervals.iter().enumerate() {
                // Find the smallest containing call
                for (call_uid, call_start, call_stop) in calls.iter() {
                    if (*call_start <= wait.start) && (wait.end <= *call_stop) {
                        let call_entry = self.entries.get_mut(call_uid).unwrap();
                        call_entry
                            .waiters
                            .wait_intervals
                            .push(WaitInterval::new(wait.start, wait.ready, wait.end));
                        to_remove.push(idx);
                        break;
                    } else {
                        // Waits should not be partially overlapping with calls
                        assert!((wait.end <= *call_start) || (*call_stop <= wait.start));
                    }
                }
            }
            // Remove any waits that we moved into a call
            for idx in to_remove.iter().rev() {
                task_entry.waiters.wait_intervals.remove(*idx);
            }
            // For each subcall find the next largest subcall that dominates
            // it and add a wait for it, if one isn't found then we add the
            // wait to the task for that subcall
            for (idx1, (call_uid, call_start, call_stop)) in calls.iter().enumerate() {
                let mut found = false;
                for idx2 in idx1 + 1..calls.len() {
                    let (next_uid, next_start, next_stop) = calls[idx2];
                    if (next_start <= *call_start) && (*call_stop <= next_stop) {
                        let next_entry = self.entries.get_mut(&next_uid).unwrap();
                        next_entry.waiters.wait_intervals.push(WaitInterval::new(
                            *call_start,
                            *call_stop,
                            *call_stop,
                        ));
                        found = true;
                        break;
                    } else {
                        // Calls should not be partially overlapping with eachother
                        assert!((*call_stop <= next_start) || (next_stop <= *call_start));
                    }
                }
                if !found {
                    task_entry.waiters.wait_intervals.push(WaitInterval::new(
                        *call_start,
                        *call_stop,
                        *call_stop,
                    ));
                }
                // Update the operation info for the calls
                let call_entry = self.entries.get_mut(&call_uid).unwrap();
                match task_entry.kind {
                    ProcEntryKind::Task(_, _) => {
                        call_entry.initiation_op = task_entry.op_id;
                    }
                    ProcEntryKind::MetaTask(_) | ProcEntryKind::ProfTask => {
                        call_entry.initiation_op = task_entry.initiation_op;
                    }
                    _ => {
                        panic!("bad processor entry kind");
                    }
                }
            }
            // Save any calls on the proc entry
            std::mem::swap(&mut task_entry.subcalls, calls);
            // Finally add the task entry back in now that we're done mutating it
            self.entries.insert(*task_uid, task_entry);
        }
    }

    fn sort_time_range(&mut self, fevents: &BTreeMap<EventID, ProfUID>) {
        fn add(
            time: &TimeRange,
            prof_uid: ProfUID,
            all_points: &mut Vec<ProcPoint>,
            points: &mut Vec<ProcPoint>,
            util_points: &mut Vec<ProcPoint>,
            record_util: bool,
        ) {
            let start = time.start.unwrap();
            let stop = time.stop.unwrap();
            let ready = time.ready;
            if stop - start > TASK_GRANULARITY_THRESHOLD && ready.is_some() {
                all_points.push(ProcPoint::new(ready.unwrap(), prof_uid, true, start.0));
                all_points.push(ProcPoint::new(stop, prof_uid, false, 0));
            } else {
                all_points.push(ProcPoint::new(
                    start,
                    prof_uid,
                    true,
                    std::u64::MAX - stop.0,
                ));
                all_points.push(ProcPoint::new(stop, prof_uid, false, 0));
            }

            points.push(ProcPoint::new(
                start,
                prof_uid,
                true,
                std::u64::MAX - stop.0,
            ));
            points.push(ProcPoint::new(stop, prof_uid, false, 0));

            if record_util {
                util_points.push(ProcPoint::new(
                    start,
                    prof_uid,
                    true,
                    std::u64::MAX - stop.0,
                ));
                util_points.push(ProcPoint::new(stop, prof_uid, false, 0));
            }
        }
        fn add_waiters(
            waiters: &Waiters,
            prof_uid: ProfUID,
            util_points: &mut Vec<ProcPoint>,
            record_util: bool,
        ) {
            for wait in &waiters.wait_intervals {
                util_points.push(ProcPoint::new(
                    wait.start,
                    prof_uid,
                    false,
                    std::u64::MAX - wait.end.0,
                ));
                if record_util {
                    util_points.push(ProcPoint::new(wait.end, prof_uid, true, 0));
                }
            }
        }

        // Before we do anything sort the runtime/mapper calls and waiters
        self.sort_calls_and_waits(fevents);

        let mut all_points = Vec::new();
        let mut points = Vec::new();
        let mut util_points = Vec::new();

        if self.kind == ProcKind::GPU {
            // GPUs are special when it comes to utilization
            // We only want to report utilization of the actual kernels running
            // on the GPUs. However, we still want to render what happens on the
            // CPU side in case it effects the outcome of the running of the kernels
            // Therefore we still render all the entries on the GPU processor but
            // we only add the GPU kernel times to the utilization points
            for (uid, entry) in &self.entries {
                let time = &entry.time_range;
                match entry.kind {
                    ProcEntryKind::GPUKernel(_, _) => {
                        add(
                            time,
                            *uid,
                            &mut all_points,
                            &mut points,
                            &mut util_points,
                            true,
                        );
                        add_waiters(&entry.waiters, *uid, &mut util_points, true);
                    }
                    _ => {
                        add(
                            time,
                            *uid,
                            &mut all_points,
                            &mut points,
                            &mut util_points,
                            false,
                        );
                        add_waiters(&entry.waiters, *uid, &mut util_points, false);
                    }
                }
            }
        } else {
            for (uid, entry) in &self.entries {
                let time = &entry.time_range;
                add(
                    time,
                    *uid,
                    &mut all_points,
                    &mut points,
                    &mut util_points,
                    true,
                );
                add_waiters(&entry.waiters, *uid, &mut util_points, true);
            }
        }

        points.sort_by_key(|a| a.time_key());
        util_points.sort_by_key(|a| a.time_key());

        // Hack: This is a max heap so reverse the values as they go in.
        let mut free_levels = BinaryHeap::<Reverse<u32>>::new();
        for point in &points {
            if point.first {
                let level = if let Some(level) = free_levels.pop() {
                    level.0
                } else {
                    self.max_levels.postincrement()
                };
                self.entry_mut(point.entry).base.set_level(level);
            } else {
                let level = self.entry(point.entry).base.level.unwrap();
                free_levels.push(Reverse(level));
            }
        }

        all_points.sort_by_key(|a| a.time_key());

        // Hack: This is a max heap so reverse the values as they go in.
        let mut free_levels_ready = BinaryHeap::<Reverse<u32>>::new();
        for point in &all_points {
            if point.first {
                let level = if let Some(level) = free_levels_ready.pop() {
                    level.0
                } else {
                    self.max_levels_ready.postincrement()
                };
                self.entry_mut(point.entry).base.set_level_ready(level);
            } else {
                let level = self.entry(point.entry).base.level_ready.unwrap();
                free_levels_ready.push(Reverse(level));
            }
        }

        // Rendering of the profile will never use non-first points, so we can
        // throw those away now.
        points.retain(|p| p.first);

        self.time_points = points;
        self.util_time_points = util_points;
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }
}

impl Container for Proc {
    type E = ProfUID;
    type S = u64;
    type Entry = ProcEntry;

    fn max_levels(&self) -> usize {
        self.max_levels as usize
    }

    fn time_points(&self) -> &Vec<TimePoint<Self::E, Self::S>> {
        &self.time_points
    }

    fn entry(&self, prof_uid: ProfUID) -> &ProcEntry {
        self.entries.get(&prof_uid).unwrap()
    }

    fn entry_mut(&mut self, prof_uid: ProfUID) -> &mut ProcEntry {
        self.entries.get_mut(&prof_uid).unwrap()
    }
}

pub type MemEntry = Inst;

pub type MemPoint = TimePoint<InstUID, u64>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, LowerHex)]
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
    pub insts: BTreeMap<InstUID, Inst>,
    pub time_points: Vec<MemPoint>,
    pub util_time_points: Vec<MemPoint>,
    pub max_live_insts: u32,
    visible: bool,
}

impl Mem {
    fn new(mem_id: MemID, kind: MemKind, capacity: u64) -> Self {
        Mem {
            mem_id,
            kind,
            capacity,
            insts: BTreeMap::new(),
            time_points: Vec::new(),
            util_time_points: Vec::new(),
            max_live_insts: 0,
            visible: true,
        }
    }

    fn add_inst(&mut self, inst: Inst) {
        self.insts.insert(inst.inst_uid, inst);
    }

    pub fn is_empty(&self) -> bool {
        self.insts.is_empty()
    }

    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) {
        self.insts.retain(|_, i| !i.trim_time_range(start, stop));
    }

    fn sort_time_range(&mut self) {
        let mut time_points = Vec::new();
        let mut time_points_level = Vec::new();

        for (key, inst) in &self.insts {
            time_points.push(MemPoint::new(
                inst.time_range.start.unwrap(),
                *key,
                true,
                std::u64::MAX - inst.time_range.stop.unwrap().0,
            ));
            time_points.push(MemPoint::new(inst.time_range.stop.unwrap(), *key, false, 0));

            time_points_level.push(MemPoint::new(
                inst.time_range.create.unwrap(),
                *key,
                true,
                std::u64::MAX - inst.time_range.stop.unwrap().0,
            ));
            time_points_level.push(MemPoint::new(inst.time_range.stop.unwrap(), *key, false, 0));
        }
        time_points.sort_by_key(|a| a.time_key());
        time_points_level.sort_by_key(|a| a.time_key());

        // Hack: This is a max heap so reverse the values as they go in.
        let mut free_levels = BinaryHeap::<Reverse<u32>>::new();
        for point in &time_points_level {
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

        // Rendering of the profile will never use non-first points, so we can
        // throw those away now.
        self.time_points = time_points.iter().filter(|p| p.first).copied().collect();
        self.util_time_points = time_points;
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }
}

impl Container for Mem {
    type E = InstUID;
    type S = u64;
    type Entry = Inst;

    fn max_levels(&self) -> usize {
        self.max_live_insts as usize
    }

    fn time_points(&self) -> &Vec<TimePoint<Self::E, Self::S>> {
        &self.time_points
    }

    fn entry(&self, inst_uid: InstUID) -> &Inst {
        self.insts.get(&inst_uid).unwrap()
    }

    fn entry_mut(&mut self, inst_uid: InstUID) -> &mut Inst {
        self.insts.get_mut(&inst_uid).unwrap()
    }
}

#[derive(Debug)]
pub struct MemProcAffinity {
    _mem_id: MemID,
    bandwidth: u32,
    latency: u32,
    pub best_aff_proc: ProcID,
}

impl MemProcAffinity {
    fn new(mem_id: MemID, bandwidth: u32, latency: u32, best_aff_proc: ProcID) -> Self {
        MemProcAffinity {
            _mem_id: mem_id,
            bandwidth,
            latency,
            best_aff_proc,
        }
    }
    fn update_best_aff(&mut self, proc_id: ProcID, b: u32, l: u32) {
        if b > self.bandwidth {
            self.best_aff_proc = proc_id;
            self.bandwidth = b;
            self.latency = l;
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum ChanEntryKind {
    Copy(EventID),
    Fill(EventID),
    DepPart(OpID, usize),
}

#[derive(Debug)]
pub enum ChanEntry {
    Copy(Copy),
    Fill(Fill),
    DepPart(DepPart),
}

impl ChanEntry {
    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) -> bool {
        self.time_range_mut().trim_time_range(start, stop)
    }
}

impl ContainerEntry for ChanEntry {
    fn base(&self) -> &Base {
        match self {
            ChanEntry::Copy(copy) => &copy.base,
            ChanEntry::Fill(fill) => &fill.base,
            ChanEntry::DepPart(deppart) => &deppart.base,
        }
    }

    fn base_mut(&mut self) -> &mut Base {
        match self {
            ChanEntry::Copy(copy) => &mut copy.base,
            ChanEntry::Fill(fill) => &mut fill.base,
            ChanEntry::DepPart(deppart) => &mut deppart.base,
        }
    }

    fn time_range(&self) -> TimeRange {
        match self {
            ChanEntry::Copy(copy) => copy.time_range,
            ChanEntry::Fill(fill) => fill.time_range,
            ChanEntry::DepPart(deppart) => deppart.time_range,
        }
    }

    fn time_range_mut(&mut self) -> &mut TimeRange {
        match self {
            ChanEntry::Copy(copy) => &mut copy.time_range,
            ChanEntry::Fill(fill) => &mut fill.time_range,
            ChanEntry::DepPart(deppart) => &mut deppart.time_range,
        }
    }

    fn waiters(&self) -> Option<&Waiters> {
        None
    }

    fn initiation(&self) -> Option<OpID> {
        match self {
            ChanEntry::Copy(copy) => Some(copy.op_id),
            ChanEntry::Fill(fill) => Some(fill.op_id),
            ChanEntry::DepPart(deppart) => Some(deppart.op_id),
        }
    }

    fn creator(&self) -> Option<EventID> {
        match self {
            ChanEntry::Copy(copy) => Some(copy.creator),
            ChanEntry::Fill(fill) => Some(fill.creator),
            ChanEntry::DepPart(deppart) => Some(deppart.creator),
        }
    }

    fn name(&self, state: &State) -> String {
        match self {
            ChanEntry::Copy(copy) => {
                let nreqs = copy.copy_inst_infos.len();
                if nreqs > 0 {
                    format!(
                        "{}: size={}, num reqs={}{}",
                        copy.copy_kind.unwrap(),
                        SizePretty(copy.size),
                        nreqs,
                        CopyInstInfoVec(&copy.copy_inst_infos, state)
                    )
                } else {
                    format!("Copy: size={}, num reqs={}", SizePretty(copy.size), nreqs,)
                }
            }
            ChanEntry::Fill(fill) => {
                let nreqs = fill.fill_inst_infos.len();
                if nreqs > 0 {
                    format!(
                        "Fill: num reqs={}{}",
                        nreqs,
                        FillInstInfoVec(&fill.fill_inst_infos, state)
                    )
                } else {
                    format!("Fill: num reqs={}", nreqs)
                }
            }
            ChanEntry::DepPart(deppart) => format!("{}", deppart.part_op),
        }
    }

    fn color(&self, state: &State) -> Color {
        let initiation = self.initiation().unwrap();
        state.get_op_color(initiation)
    }

    fn provenance<'a, 'b>(&'a self, state: &'b State) -> Option<&'b str> {
        let initiation = self.initiation().unwrap();
        state.find_op_provenance(initiation)
    }
}

pub type ChanPoint = TimePoint<ProfUID, u64>;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, TryFromPrimitive)]
#[repr(u32)]
pub enum ChanKind {
    Copy = 0,
    Fill = 1,
    Gather = 2,
    Scatter = 3,
    DepPart = 4,
}

impl fmt::Display for ChanKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChanID {
    pub src: Option<MemID>,
    pub dst: Option<MemID>,
    pub channel_kind: ChanKind,
}

impl ChanID {
    fn new_copy(src: MemID, dst: MemID) -> Self {
        ChanID {
            src: Some(src),
            dst: Some(dst),
            channel_kind: ChanKind::Copy,
        }
    }
    fn new_fill(dst: MemID) -> Self {
        ChanID {
            src: None,
            dst: Some(dst),
            channel_kind: ChanKind::Fill,
        }
    }
    fn new_gather(dst: MemID) -> Self {
        ChanID {
            src: None,
            dst: Some(dst),
            channel_kind: ChanKind::Gather,
        }
    }
    fn new_scatter(src: MemID) -> Self {
        ChanID {
            src: Some(src),
            dst: None,
            channel_kind: ChanKind::Scatter,
        }
    }
    fn new_deppart() -> Self {
        ChanID {
            src: None,
            dst: None,
            channel_kind: ChanKind::DepPart,
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
    entries: BTreeMap<ProfUID, ChanEntry>,
    pub copies: BTreeMap<EventID, ProfUID>,
    pub fills: BTreeMap<EventID, ProfUID>,
    pub depparts: BTreeMap<OpID, Vec<ProfUID>>,
    pub time_points: Vec<ChanPoint>,
    pub util_time_points: Vec<ChanPoint>,
    pub max_levels: u32,
    visible: bool,
}

impl Chan {
    fn new(chan_id: ChanID) -> Self {
        Chan {
            chan_id,
            entries: BTreeMap::new(),
            copies: BTreeMap::new(),
            fills: BTreeMap::new(),
            depparts: BTreeMap::new(),
            time_points: Vec::new(),
            util_time_points: Vec::new(),
            max_levels: 0,
            visible: true,
        }
    }

    fn add_copy(&mut self, copy: Copy) {
        self.copies.insert(copy.fevent, copy.base.prof_uid);
        self.entries
            .entry(copy.base.prof_uid)
            .or_insert(ChanEntry::Copy(copy));
    }

    fn add_fill(&mut self, fill: Fill) {
        self.fills.insert(fill.fevent, fill.base.prof_uid);
        self.entries
            .entry(fill.base.prof_uid)
            .or_insert(ChanEntry::Fill(fill));
    }

    fn add_deppart(&mut self, deppart: DepPart) {
        self.depparts
            .entry(deppart.op_id)
            .or_insert_with(Vec::new)
            .push(deppart.base.prof_uid);
        self.entries
            .entry(deppart.base.prof_uid)
            .or_insert(ChanEntry::DepPart(deppart));
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) {
        self.entries.retain(|_, e| !e.trim_time_range(start, stop));
    }

    fn sort_time_range(&mut self) {
        fn add(time: TimeRange, prof_uid: ProfUID, points: &mut Vec<ChanPoint>) {
            let start = time.start.unwrap();
            let stop = time.stop.unwrap();
            points.push(ChanPoint::new(
                start,
                prof_uid,
                true,
                std::u64::MAX - stop.0,
            ));
            points.push(ChanPoint::new(stop, prof_uid, false, 0));
        }

        let mut points = Vec::new();

        for (prof_uid, entry) in &self.entries {
            let time = entry.time_range();
            add(time, *prof_uid, &mut points);
        }

        points.sort_by_key(|a| a.time_key());

        // Hack: This is a max heap so reverse the values as they go in.
        let mut free_levels = BinaryHeap::<Reverse<u32>>::new();
        for point in &points {
            if point.first {
                let level = if let Some(level) = free_levels.pop() {
                    level.0
                } else {
                    self.max_levels.postincrement()
                };
                self.entry_mut(point.entry).base_mut().set_level(level);
            } else {
                let level = self.entry(point.entry).base().level.unwrap();
                free_levels.push(Reverse(level));
            }
        }

        self.time_points = points.iter().filter(|p| p.first).copied().collect();
        self.util_time_points = points;
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }
}

impl Container for Chan {
    type E = ProfUID;
    type S = u64;
    type Entry = ChanEntry;

    fn max_levels(&self) -> usize {
        self.max_levels as usize
    }

    fn time_points(&self) -> &Vec<TimePoint<Self::E, Self::S>> {
        &self.time_points
    }

    fn entry(&self, prof_uid: ProfUID) -> &ChanEntry {
        self.entries.get(&prof_uid).unwrap()
    }

    fn entry_mut(&mut self, prof_uid: ProfUID) -> &mut ChanEntry {
        self.entries.get_mut(&prof_uid).unwrap()
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
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
    fn set_point(&mut self, dim: u32, values: &[u64]) -> &mut Self {
        let new_bounds = Bounds::Point {
            point: values.to_owned(),
            dim,
        };
        self.bounds = new_bounds;
        self
    }
    fn set_rect(&mut self, dim: u32, values: &[u64], max_dim: i32) -> &mut Self {
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

    fn set_name(&mut self, name: &str) -> &mut Self {
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct IPartID(pub u64);

#[derive(Debug)]
pub struct IPart {
    _ipart_id: IPartID,
    name: Option<String>,
    pub parent: Option<ISpaceID>,
    disjoint: Option<bool>,
    point0: Option<u64>,
}

impl IPart {
    fn new(ipart_id: IPartID) -> Self {
        IPart {
            _ipart_id: ipart_id,
            name: None,
            parent: None,
            disjoint: None,
            point0: None,
        }
    }
    fn set_name(&mut self, name: &str) -> &mut Self {
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
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
    fn set_name(&mut self, name: &str) -> &mut Self {
        assert!(self.name.as_ref().map_or(true, |n| n == name));
        self.name = Some(name.to_owned());
        self
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct FieldID(pub u32);

#[derive(Debug)]
pub struct Field {
    _fspace_id: FSpaceID,
    _field_id: FieldID,
    _size: u64,
    pub name: String,
}

impl Field {
    fn new(fspace_id: FSpaceID, field_id: FieldID, size: u64, name: &str) -> Self {
        Field {
            _fspace_id: fspace_id,
            _field_id: field_id,
            _size: size,
            name: name.to_owned(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct TreeID(pub u32);

#[derive(Debug)]
pub struct Region {
    _ispace_id: ISpaceID,
    _fspace_id: FSpaceID,
    _tree_id: TreeID,
    _name: String,
}

impl Region {
    fn new(ispace_id: ISpaceID, fspace_id: FSpaceID, tree_id: TreeID, name: &str) -> Self {
        Region {
            _ispace_id: ispace_id,
            _fspace_id: fspace_id,
            _tree_id: tree_id,
            _name: name.to_owned(),
        }
    }
}

#[derive(Debug)]
pub struct Dependencies {
    pub in_: BTreeSet<ProfUID>,
    pub out: BTreeSet<ProfUID>,
    pub parent: BTreeSet<ProfUID>,
    pub children: BTreeSet<ProfUID>,
}

impl Dependencies {
    fn new() -> Self {
        Dependencies {
            in_: BTreeSet::new(),
            out: BTreeSet::new(),
            parent: BTreeSet::new(),
            children: BTreeSet::new(),
        }
    }
}

#[derive(Debug)]
pub struct SpyEvent {
    preconditions: BTreeSet<EventID>,
    postconditions: BTreeSet<EventID>,
}

impl SpyEvent {
    fn new() -> Self {
        SpyEvent {
            preconditions: BTreeSet::new(),
            postconditions: BTreeSet::new(),
        }
    }
}

#[derive(Debug)]
pub struct SpyOp {
    precondition: EventID,
    postcondition: EventID,
}

impl SpyOp {
    fn new(precondition: EventID, postcondition: EventID) -> Self {
        SpyOp {
            precondition,
            postcondition,
        }
    }
}

#[derive(Debug)]
pub struct Align {
    _field_id: FieldID,
    _eqk: u32,
    pub align_desc: u32,
    pub has_align: bool,
}

impl Align {
    fn new(field_id: FieldID, eqk: u32, align_desc: u32, has_align: bool) -> Self {
        Align {
            _field_id: field_id,
            _eqk: eqk,
            align_desc,
            has_align,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct InstID(pub u64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Dim(pub u32);

#[derive(Debug)]
pub struct Inst {
    pub base: Base,
    pub inst_uid: InstUID,
    pub inst_id: Option<InstID>,
    pub op_id: Option<OpID>,
    mem_id: Option<MemID>,
    pub size: Option<u64>,
    pub time_range: TimeRange,
    pub ispace_ids: Vec<ISpaceID>,
    pub fspace_ids: Vec<FSpaceID>,
    tree_id: Option<TreeID>,
    pub fields: BTreeMap<FSpaceID, Vec<FieldID>>,
    pub align_desc: BTreeMap<FSpaceID, Vec<Align>>,
    pub dim_order: BTreeMap<Dim, DimKind>,
    pub creator: Option<EventID>,
}

impl Inst {
    fn new(base: Base, inst_uid: InstUID) -> Self {
        Inst {
            base,
            inst_uid,
            inst_id: None,
            op_id: None,
            mem_id: None,
            size: None,
            time_range: TimeRange::new_empty(),
            ispace_ids: Vec::new(),
            fspace_ids: Vec::new(),
            tree_id: None,
            fields: BTreeMap::new(),
            align_desc: BTreeMap::new(),
            dim_order: BTreeMap::new(),
            creator: None,
        }
    }
    fn set_inst_id(&mut self, inst_id: InstID) -> &mut Self {
        assert!(self.inst_id.map_or(true, |i| i == inst_id));
        self.inst_id = Some(inst_id);
        self
    }
    fn set_op_id(&mut self, op_id: OpID) -> &mut Self {
        assert!(self.op_id.map_or(true, |i| i == op_id));
        self.op_id = Some(op_id);
        self
    }
    fn set_mem(&mut self, mem_id: MemID) -> &mut Self {
        assert!(self.mem_id.map_or(true, |i| i == mem_id));
        self.mem_id = Some(mem_id);
        self
    }
    fn set_size(&mut self, size: u64) -> &mut Self {
        assert!(self.size.map_or(true, |s| s == size));
        self.size = Some(size);
        self
    }
    fn set_start_stop(&mut self, start: Timestamp, ready: Timestamp, stop: Timestamp) -> &mut Self {
        self.time_range = TimeRange::new_full(start, ready, ready, stop);
        self
    }
    fn add_ispace(&mut self, ispace_id: ISpaceID) -> &mut Self {
        self.ispace_ids.push(ispace_id);
        self
    }
    fn add_fspace(&mut self, fspace_id: FSpaceID) -> &mut Self {
        self.fspace_ids.push(fspace_id);
        self.fields.entry(fspace_id).or_insert_with(Vec::new);
        self.align_desc.entry(fspace_id).or_insert_with(Vec::new);
        self
    }
    fn add_field(&mut self, fspace_id: FSpaceID, field_id: FieldID) -> &mut Self {
        self.fields
            .entry(fspace_id)
            .or_insert_with(Vec::new)
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
            .or_insert_with(Vec::new)
            .push(Align::new(field_id, eqk, align_desc, has_align));
        self
    }
    fn add_dim_order(&mut self, dim: Dim, dim_kind: DimKind) -> &mut Self {
        self.dim_order.insert(dim, dim_kind);
        self
    }
    fn set_tree(&mut self, tree_id: TreeID) -> &mut Self {
        assert!(self.tree_id.map_or(true, |t| t == tree_id));
        self.tree_id = Some(tree_id);
        self
    }
    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) -> bool {
        self.time_range.trim_time_range(start, stop)
    }
    fn set_creator(&mut self, creator: EventID) -> &mut Self {
        assert!(self.creator.map_or(true, |c| c == creator));
        self.creator = Some(creator);
        self
    }
}

impl Ord for Inst {
    fn cmp(&self, other: &Self) -> Ordering {
        self.base.prof_uid.cmp(&other.base.prof_uid)
    }
}

impl PartialOrd for Inst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Inst {
    fn eq(&self, other: &Self) -> bool {
        self.base.prof_uid == other.base.prof_uid
    }
}

impl Eq for Inst {}

impl ContainerEntry for Inst {
    fn base(&self) -> &Base {
        &self.base
    }

    fn base_mut(&mut self) -> &mut Base {
        &mut self.base
    }

    fn time_range(&self) -> TimeRange {
        self.time_range
    }

    fn time_range_mut(&mut self) -> &mut TimeRange {
        &mut self.time_range
    }

    fn waiters(&self) -> Option<&Waiters> {
        None
    }

    fn initiation(&self) -> Option<OpID> {
        self.op_id
    }

    fn creator(&self) -> Option<EventID> {
        self.creator
    }

    fn name(&self, state: &State) -> String {
        format!("{}", InstPretty(self, state))
    }

    fn color(&self, state: &State) -> Color {
        let initiation = self.op_id;
        state.get_op_color(initiation.unwrap())
    }

    fn provenance<'a, 'b>(&'a self, state: &'b State) -> Option<&'b str> {
        if let Some(initiation) = self.op_id {
            return state.find_op_provenance(initiation);
        }
        None
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, LowerHex)]
pub struct Color(pub u32);

impl Color {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        let r = r as u32;
        let g = g as u32;
        let b = b as u32;
        Color((r << 16) | (g << 8) | b)
    }

    // These are HTML colors. Values are determined empirically by fiddling
    // with CSS styles in Firefox. As best I can tell (by visual comparison)
    // they seem to be sRGB (note: this is gamma space, not linear).
    pub const BLACK: Color = Color(0x000000);
    pub const BLUE: Color = Color(0x0000FF);
    pub const CRIMSON: Color = Color(0xDC143C);
    pub const DARKGOLDENROD: Color = Color(0xB8860B);
    pub const DARKMAGENTA: Color = Color(0x8B008B);
    pub const OLIVEDRAB: Color = Color(0x6B8E23);
    pub const ORANGERED: Color = Color(0xFF4500);
    pub const STEELBLUE: Color = Color(0x4682B4);
    pub const GRAY: Color = Color(0x808080);
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct MapperCallKindID(pub u32);

#[derive(Debug)]
pub struct MapperCallKind {
    pub kind: MapperCallKindID,
    pub name: String,
    pub color: Option<Color>,
}

impl MapperCallKind {
    fn new(kind: MapperCallKindID, name: &str) -> Self {
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct RuntimeCallKindID(pub u32);

#[derive(Debug)]
pub struct RuntimeCallKind {
    pub kind: RuntimeCallKindID,
    pub name: String,
    pub color: Option<Color>,
}

impl RuntimeCallKind {
    fn new(kind: RuntimeCallKindID, name: &str) -> Self {
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct TaskID(pub u32);

#[derive(Debug)]
pub struct TaskKind {
    pub task_id: TaskID,
    pub name: Option<String>,
}

impl TaskKind {
    fn new(task_id: TaskID) -> Self {
        TaskKind {
            task_id,
            name: None,
        }
    }
    fn set_name(&mut self, name: &str, overwrite: bool) {
        if self.name.is_none() || overwrite {
            self.name = Some(name.to_owned());
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct VariantID(pub u32);

#[derive(Debug)]
pub struct Variant {
    variant_id: VariantID,
    message: bool,
    ordered_vc: bool,
    pub name: String,
    task_id: Option<TaskID>,
    pub color: Option<Color>,
}

impl Variant {
    fn new(variant_id: VariantID, message: bool, ordered_vc: bool, name: &str) -> Self {
        Variant {
            variant_id,
            message,
            ordered_vc,
            name: name.to_owned(),
            task_id: None,
            color: None,
        }
    }
    fn set_task(&mut self, task_id: TaskID) -> &mut Self {
        assert!(self.task_id.map_or(true, |t| t == task_id));
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
        assert!(self.level.is_none());
        self.level = Some(level);
        self
    }
    fn set_level_ready(&mut self, level_ready: u32) -> &mut Self {
        assert!(self.level_ready.is_none());
        self.level_ready = Some(level_ready);
        self
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimeRange {
    pub create: Option<Timestamp>,
    pub ready: Option<Timestamp>,
    pub start: Option<Timestamp>,
    pub stop: Option<Timestamp>,
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
        }
    }
    fn new_start(start: Timestamp, stop: Timestamp) -> Self {
        assert!(start <= stop);
        TimeRange {
            create: None,
            ready: None,
            start: Some(start),
            stop: Some(stop),
        }
    }
    fn new_empty() -> Self {
        TimeRange {
            create: None,
            ready: None,
            start: None,
            stop: None,
        }
    }
    fn trim_time_range(&mut self, start: Timestamp, stop: Timestamp) -> bool {
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
            return true;
        }
        self.create = self.create.map(clip);
        self.ready = self.ready.map(clip);
        self.start = self.start.map(clip);
        self.stop = self.stop.map(clip);
        false
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct OpID(pub u64);

impl From<spy::serialize::UniqueID> for OpID {
    fn from(e: spy::serialize::UniqueID) -> Self {
        OpID(e.0)
    }
}

impl From<spy::serialize::ContextID> for OpID {
    fn from(e: spy::serialize::ContextID) -> Self {
        OpID(e.0)
    }
}

#[derive(Debug)]
pub struct MultiTask {
    pub op_id: OpID,
    pub task_id: TaskID,
}

impl MultiTask {
    fn new(op_id: OpID, task_id: TaskID) -> Self {
        MultiTask { op_id, task_id }
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
pub struct OperationInstInfo {
    pub inst_uid: InstUID,
    _index: u32,
    _field_id: FieldID,
}

impl OperationInstInfo {
    fn new(inst_uid: InstUID, index: u32, field_id: FieldID) -> Self {
        OperationInstInfo {
            inst_uid,
            _index: index,
            _field_id: field_id,
        }
    }
}

#[derive(Debug)]
pub struct Operation {
    pub base: Base,
    pub parent_id: Option<OpID>,
    pub kind: Option<OpKindID>,
    pub provenance: Option<String>,
    pub operation_inst_infos: Vec<OperationInstInfo>,
}

impl Operation {
    fn new(base: Base) -> Self {
        Operation {
            base,
            parent_id: None,
            kind: None,
            provenance: None,
            operation_inst_infos: Vec::new(),
        }
    }
    fn set_parent_id(&mut self, parent_id: OpID) -> &mut Self {
        let parent = if parent_id == OpID(std::u64::MAX) {
            None
        } else {
            Some(parent_id)
        };
        assert!(self.parent_id.is_none() || self.parent_id == parent);
        self.parent_id = parent;
        self
    }
    fn set_kind(&mut self, kind: OpKindID) -> &mut Self {
        assert!(self.kind.is_none());
        self.kind = Some(kind);
        self
    }
    fn set_provenance(&mut self, provenance: &str) -> &mut Self {
        self.provenance = Some(provenance.to_owned());
        self
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct EventID(pub u64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct InstUID(pub u64);

impl From<spy::serialize::EventID> for EventID {
    fn from(e: spy::serialize::EventID) -> Self {
        EventID(e.0 .0)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, TryFromPrimitive)]
#[repr(u32)]
pub enum CopyKind {
    Copy = 0,
    Gather = 1,
    Scatter = 2,
    GatherScatter = 3,
}

impl fmt::Display for CopyKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct CopyInstInfo {
    src: Option<MemID>,
    dst: Option<MemID>,
    pub src_fid: FieldID,
    pub dst_fid: FieldID,
    pub src_inst_uid: InstUID,
    pub dst_inst_uid: InstUID,
    _fevent: EventID,
    pub num_hops: u32,
    pub indirect: bool,
}

impl CopyInstInfo {
    fn new(
        src: Option<MemID>,
        dst: Option<MemID>,
        src_fid: FieldID,
        dst_fid: FieldID,
        src_inst_uid: InstUID,
        dst_inst_uid: InstUID,
        fevent: EventID,
        num_hops: u32,
        indirect: bool,
    ) -> Self {
        CopyInstInfo {
            src,
            dst,
            src_fid,
            dst_fid,
            src_inst_uid,
            dst_inst_uid,
            _fevent: fevent,
            num_hops,
            indirect,
        }
    }
}

#[derive(Debug)]
pub struct Copy {
    base: Base,
    creator: EventID,
    fevent: EventID,
    time_range: TimeRange,
    chan_id: Option<ChanID>,
    pub op_id: OpID,
    pub size: u64,
    pub collective: u32,
    pub copy_kind: Option<CopyKind>,
    pub copy_inst_infos: Vec<CopyInstInfo>,
}

impl Copy {
    fn new(
        base: Base,
        time_range: TimeRange,
        op_id: OpID,
        size: u64,
        creator: EventID,
        fevent: EventID,
        collective: u32,
    ) -> Self {
        Copy {
            base,
            creator,
            fevent,
            time_range,
            chan_id: None,
            op_id,
            size,
            collective,
            copy_kind: None,
            copy_inst_infos: Vec::new(),
        }
    }

    fn add_copy_inst_info(&mut self, copy_inst_info: CopyInstInfo) {
        self.copy_inst_infos.push(copy_inst_info);
    }

    fn split_by_channel(mut self, allocator: &mut ProfUIDAllocator) -> Vec<Self> {
        assert!(self.chan_id.is_none());
        assert!(self.copy_kind.is_none());

        // Assumptions:
        //
        //  1. A given Copy will always be entirely direct or entirely indirect.
        //
        //  2. A direct copy can have multiple CopyInstInfos with different
        //     src/dst memories/instances.
        //
        //  3. An indirect copy will have exactly one indirect field. However
        //     it might have multiple direct fields, and those direct fields could
        //     have different src/dst memories/instances.

        // Find the indirect field (if any). There is always at most one.
        let indirect = self
            .copy_inst_infos
            .iter()
            .position(|i| i.indirect)
            .map(|idx| self.copy_inst_infos.remove(idx));
        assert!(self.copy_inst_infos.iter().all(|i| !i.indirect));

        // Figure out which side we're indirect on, if any.
        let indirect_src = indirect.map_or(false, |i| i.src.is_some());
        let indirect_dst = indirect.map_or(false, |i| i.dst.is_some());

        let mut result = Vec::new();

        let groups = self.copy_inst_infos.linear_group_by(|a, b| {
            (indirect_src || a.src == b.src) && (indirect_dst || a.dst == b.dst)
        });
        for group in groups {
            let info = group.first().unwrap();
            let copy_kind = match (indirect_src, indirect_dst) {
                (false, false) => CopyKind::Copy,
                (true, false) => CopyKind::Gather,
                (false, true) => CopyKind::Scatter,
                (true, true) => CopyKind::GatherScatter,
            };

            let chan_id = match (indirect_src, indirect_dst, info.src, info.dst) {
                (false, false, Some(src), Some(dst)) => ChanID::new_copy(src, dst),
                (true, false, _, Some(dst)) => ChanID::new_gather(dst),
                (false, true, Some(src), _) => ChanID::new_scatter(src),
                (true, true, _, _) => unimplemented!("can't assign GatherScatter channel"),
                _ => unreachable!("invalid copy kind"),
            };

            let mut group = group.to_owned();
            // Hack: currently we just always force the indirect field to go
            // first, which matches the current Legion implementation, but is
            // not guaranteed.
            indirect.map(|i| group.insert(0, i));
            result.push(Copy {
                base: Base::new(allocator),
                copy_kind: Some(copy_kind),
                chan_id: Some(chan_id),
                copy_inst_infos: group,
                ..self
            })
        }
        result
    }
}

#[derive(Debug, Copy, Clone)]
pub struct FillInstInfo {
    _dst: MemID,
    pub fid: FieldID,
    pub dst_inst_uid: InstUID,
    _fevent: EventID,
}

impl FillInstInfo {
    fn new(dst: MemID, fid: FieldID, dst_inst_uid: InstUID, fevent: EventID) -> Self {
        FillInstInfo {
            _dst: dst,
            fid,
            dst_inst_uid,
            _fevent: fevent,
        }
    }
}

#[derive(Debug)]
pub struct Fill {
    base: Base,
    creator: EventID,
    fevent: EventID,
    time_range: TimeRange,
    chan_id: Option<ChanID>,
    pub op_id: OpID,
    pub size: u64,
    pub fill_inst_infos: Vec<FillInstInfo>,
}

impl Fill {
    fn new(
        base: Base,
        time_range: TimeRange,
        op_id: OpID,
        size: u64,
        creator: EventID,
        fevent: EventID,
    ) -> Self {
        Fill {
            base,
            creator,
            fevent,
            time_range,
            chan_id: None,
            op_id,
            size,
            fill_inst_infos: Vec::new(),
        }
    }

    fn add_fill_inst_info(&mut self, fill_inst_info: FillInstInfo) {
        self.fill_inst_infos.push(fill_inst_info);
    }

    fn add_channel(&mut self) {
        // sanity check
        assert!(self.chan_id.is_none());
        assert!(!self.fill_inst_infos.is_empty());
        let chan_dst = self.fill_inst_infos[0]._dst;
        for fill_inst_info in &self.fill_inst_infos {
            assert!(fill_inst_info._dst == chan_dst);
        }
        let chan_id = ChanID::new_fill(chan_dst);
        self.chan_id = Some(chan_id);
    }
}

#[derive(Debug)]
pub struct DepPart {
    base: Base,
    creator: EventID,
    pub part_op: DepPartKind,
    time_range: TimeRange,
    pub op_id: OpID,
}

impl DepPart {
    fn new(
        base: Base,
        part_op: DepPartKind,
        time_range: TimeRange,
        op_id: OpID,
        creator: EventID,
    ) -> Self {
        DepPart {
            base,
            creator,
            part_op,
            time_range,
            op_id,
        }
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
    let r = (r * 255.0).floor() as u8;
    let g = (g * 255.0).floor() as u8;
    let b = (b * 255.0).floor() as u8;
    Color::new(r, g, b)
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
        xor &= 1;
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
    pub num_nodes: u32,
    pub zero_time: TimestampDelta,
    pub _calibration_err: i64,
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
    op_prof_uid: BTreeMap<OpID, ProfUID>,
    pub prof_uid_proc: BTreeMap<ProfUID, ProcID>,
    pub tasks: BTreeMap<OpID, ProcID>,
    pub multi_tasks: BTreeMap<OpID, MultiTask>,
    pub last_time: Timestamp,
    pub mapper_call_kinds: BTreeMap<MapperCallKindID, MapperCallKind>,
    pub runtime_call_kinds: BTreeMap<RuntimeCallKindID, RuntimeCallKind>,
    pub insts: BTreeMap<InstUID, MemID>,
    pub index_spaces: BTreeMap<ISpaceID, ISpace>,
    pub index_partitions: BTreeMap<IPartID, IPart>,
    logical_regions: BTreeMap<(ISpaceID, FSpaceID, TreeID), Region>,
    pub field_spaces: BTreeMap<FSpaceID, FSpace>,
    pub has_prof_data: bool,
    pub visible_nodes: Vec<NodeID>,
    pub source_locator: Vec<String>,
    pub fevents: BTreeMap<EventID, ProfUID>,
}

impl State {
    fn create_op(&mut self, op_id: OpID) -> &mut Operation {
        let alloc = &mut self.prof_uid_allocator;
        self.operations
            .entry(op_id)
            .or_insert_with(|| Operation::new(Base::new(alloc)))
    }

    pub fn find_op(&self, op_id: OpID) -> Option<&Operation> {
        self.operations.get(&op_id)
    }

    fn find_op_mut(&mut self, op_id: OpID) -> Option<&mut Operation> {
        self.operations.get_mut(&op_id)
    }

    fn find_op_provenance(&self, op_id: OpID) -> Option<&str> {
        self.find_op(op_id).and_then(|op| op.provenance.as_deref())
    }

    pub fn get_op_color(&self, op_id: OpID) -> Color {
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

        Color::BLACK
    }

    fn create_task(
        &mut self,
        op_id: OpID,
        proc_id: ProcID,
        task_id: TaskID,
        variant_id: VariantID,
        time_range: TimeRange,
        creator: EventID,
        fevent: EventID,
    ) -> &mut ProcEntry {
        // Hack: we have to do this in two places, because we don't know what
        // order the logger calls are going to come in. If the operation gets
        // logged first, this will come back Some(_) and we'll store it below.
        let parent_id = self.create_op(op_id).parent_id;
        self.tasks.insert(op_id, proc_id);
        let alloc = &mut self.prof_uid_allocator;
        let proc = self.procs.get_mut(&proc_id).unwrap();
        proc.create_proc_entry(
            Base::new(alloc),
            Some(op_id),
            parent_id,
            ProcEntryKind::Task(task_id, variant_id),
            time_range,
            creator,
            fevent,
            &mut self.op_prof_uid,
            &mut self.prof_uid_proc,
            &mut self.fevents,
        )
    }

    pub fn find_task(&self, op_id: OpID) -> Option<&ProcEntry> {
        let proc = self.procs.get(self.tasks.get(&op_id)?)?;
        proc.find_task(op_id)
    }

    fn find_task_mut(&mut self, op_id: OpID) -> Option<&mut ProcEntry> {
        self.create_op(op_id); // FIXME: Elliott: do we REALLY need this? (and if so, yuck)
        let proc = self.procs.get_mut(self.tasks.get(&op_id)?)?;
        proc.find_task_mut(op_id)
    }

    fn create_meta(
        &mut self,
        op_id: OpID,
        variant_id: VariantID,
        proc_id: ProcID,
        time_range: TimeRange,
        creator: EventID,
        fevent: EventID,
    ) -> &mut ProcEntry {
        self.create_op(op_id);
        self.meta_tasks.insert((op_id, variant_id), proc_id);
        let alloc = &mut self.prof_uid_allocator;
        let proc = self.procs.get_mut(&proc_id).unwrap();
        proc.create_proc_entry(
            Base::new(alloc),
            None,
            Some(op_id), // FIXME: should really make this None if op_id == 0 but backwards compatibilty with Python is hard
            ProcEntryKind::MetaTask(variant_id),
            time_range,
            creator,
            fevent,
            &mut self.op_prof_uid,
            &mut self.prof_uid_proc,
            &mut self.fevents,
        )
    }

    fn find_last_meta_mut(&mut self, op_id: OpID, variant_id: VariantID) -> Option<&mut ProcEntry> {
        let proc = self
            .procs
            .get_mut(self.meta_tasks.get(&(op_id, variant_id))?)?;
        proc.find_last_meta_mut(op_id, variant_id)
    }

    fn create_mapper_call(
        &mut self,
        kind: MapperCallKindID,
        proc_id: ProcID,
        op_id: OpID,
        time_range: TimeRange,
        fevent: EventID,
    ) -> &mut ProcEntry {
        self.create_op(op_id);
        let alloc = &mut self.prof_uid_allocator;
        let proc = self.procs.get_mut(&proc_id).unwrap();
        proc.create_proc_entry(
            Base::new(alloc),
            None,
            if op_id.0 > 0 { Some(op_id) } else { None },
            ProcEntryKind::MapperCall(kind),
            time_range,
            fevent,
            fevent,
            &mut self.op_prof_uid,
            &mut self.prof_uid_proc,
            &mut self.fevents,
        )
    }

    fn create_runtime_call(
        &mut self,
        kind: RuntimeCallKindID,
        proc_id: ProcID,
        time_range: TimeRange,
        fevent: EventID,
    ) -> &mut ProcEntry {
        let alloc = &mut self.prof_uid_allocator;
        let proc = self.procs.get_mut(&proc_id).unwrap();
        proc.create_proc_entry(
            Base::new(alloc),
            None,
            None,
            ProcEntryKind::RuntimeCall(kind),
            time_range,
            fevent,
            fevent,
            &mut self.op_prof_uid,
            &mut self.prof_uid_proc,
            &mut self.fevents,
        )
    }

    fn create_gpu_kernel(
        &mut self,
        op_id: OpID,
        proc_id: ProcID,
        task_id: TaskID,
        variant_id: VariantID,
        time_range: TimeRange,
        fevent: EventID,
    ) -> &mut ProcEntry {
        let alloc = &mut self.prof_uid_allocator;
        let proc = self.procs.get_mut(&proc_id).unwrap();
        proc.create_proc_entry(
            Base::new(alloc),
            Some(op_id),
            None,
            ProcEntryKind::GPUKernel(task_id, variant_id),
            time_range,
            fevent,
            fevent,
            &mut self.op_prof_uid,
            &mut self.prof_uid_proc,
            &mut self.fevents,
        )
    }

    fn create_prof_task(
        &mut self,
        proc_id: ProcID,
        op_id: OpID,
        time_range: TimeRange,
        creator: EventID,
        fevent: EventID,
    ) -> &mut ProcEntry {
        let alloc = &mut self.prof_uid_allocator;
        let proc = self.procs.get_mut(&proc_id).unwrap();
        proc.create_proc_entry(
            Base::new(alloc),
            None,
            Some(op_id), // FIXME: should really make this None if op_id == 0 but backwards compatibilty with Python is hard
            ProcEntryKind::ProfTask,
            time_range,
            creator,
            fevent,
            &mut self.op_prof_uid,
            &mut self.prof_uid_proc,
            &mut self.fevents,
        )
    }

    fn create_copy<'a>(
        &'a mut self,
        time_range: TimeRange,
        op_id: OpID,
        size: u64,
        creator: EventID,
        fevent: EventID,
        collective: u32,
        copies: &'a mut BTreeMap<EventID, Copy>,
    ) -> &'a mut Copy {
        let alloc = &mut self.prof_uid_allocator;
        assert!(!copies.contains_key(&fevent));
        copies.entry(fevent).or_insert_with(|| {
            Copy::new(
                Base::new(alloc),
                time_range,
                op_id,
                size,
                creator,
                fevent,
                collective,
            )
        })
    }

    fn create_fill<'a>(
        &'a mut self,
        time_range: TimeRange,
        op_id: OpID,
        size: u64,
        creator: EventID,
        fevent: EventID,
        fills: &'a mut BTreeMap<EventID, Fill>,
    ) -> &'a mut Fill {
        let alloc = &mut self.prof_uid_allocator;
        assert!(!fills.contains_key(&fevent));
        fills.entry(fevent).or_insert_with(|| {
            Fill::new(Base::new(alloc), time_range, op_id, size, creator, fevent)
        })
    }

    fn create_deppart(
        &mut self,
        op_id: OpID,
        part_op: DepPartKind,
        time_range: TimeRange,
        creator: EventID,
    ) {
        self.create_op(op_id);
        let base = Base::new(&mut self.prof_uid_allocator); // FIXME: construct here to avoid mutability conflict
        let chan = self.find_deppart_chan_mut();
        chan.add_deppart(DepPart::new(base, part_op, time_range, op_id, creator));
    }

    fn find_chan_mut(&mut self, chan_id: ChanID) -> &mut Chan {
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
        inst_uid: InstUID,
        insts: &'a mut BTreeMap<InstUID, Inst>,
    ) -> &'a mut Inst {
        let alloc = &mut self.prof_uid_allocator;
        insts
            .entry(inst_uid)
            .or_insert_with(|| Inst::new(Base::new(alloc), inst_uid))
    }

    pub fn find_inst(&self, inst_uid: InstUID) -> Option<&Inst> {
        let mem_id = self.insts.get(&inst_uid)?;
        let mem = self.mems.get(mem_id)?;
        mem.insts.get(&inst_uid)
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

    pub fn process_records(&mut self, records: &Vec<Record>, call_threshold: Timestamp) {
        // We need a separate table here because instances can't be
        // immediately linked to their associated memory from the
        // logs. Therefore we defer this process until all records
        // have been processed.
        let mut insts = BTreeMap::new();
        let mut copies = BTreeMap::new();
        let mut fills = BTreeMap::new();
        for record in records {
            process_record(
                record,
                self,
                &mut insts,
                &mut copies,
                &mut fills,
                call_threshold,
            );
        }
        // put inst into memories
        for inst in insts.into_values() {
            if let Some(mem_id) = inst.mem_id {
                let mem = self.mems.get_mut(&mem_id).unwrap();
                mem.add_inst(inst);
            } else {
                unreachable!();
            }
        }
        // put fills into channels
        for mut fill in fills.into_values() {
            if !fill.fill_inst_infos.is_empty() {
                fill.add_channel();
                if let Some(chan_id) = fill.chan_id {
                    let chan = self.find_chan_mut(chan_id);
                    chan.add_fill(fill);
                } else {
                    unreachable!();
                }
            }
        }
        // put copies into channels
        for copy in copies.into_values() {
            if !copy.copy_inst_infos.is_empty() {
                let split = copy.split_by_channel(&mut self.prof_uid_allocator);
                for elt in split {
                    if let Some(chan_id) = elt.chan_id {
                        let chan = self.find_chan_mut(chan_id);
                        chan.add_copy(elt);
                    } else {
                        unreachable!();
                    }
                }
            }
        }
        self.has_prof_data = true;
    }

    fn compute_duration(&self, prof_uid: ProfUID) -> u64 {
        if let Some(proc_id) = self.prof_uid_proc.get(&prof_uid) {
            let proc = self.procs.get(proc_id).unwrap();
            let entry = &proc.entry(prof_uid);
            let mut total = 0;
            let mut start = entry.time_range.start.unwrap().0;
            for wait in &entry.waiters.wait_intervals {
                total += wait.start.0 - start;
                start = wait.end.0;
            }
            total += entry.time_range.stop.unwrap().0 - start;
            return total;
        }
        0
    }

    pub fn trim_time_range(&mut self, start: Option<Timestamp>, stop: Option<Timestamp>) {
        if start.is_none() && stop.is_none() {
            return;
        }
        let start = start.unwrap_or_else(|| 0.into());
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

    pub fn check_message_latencies(&self, threshold: f64 /* us */, warn_percentage: f64) {
        assert!(threshold >= 0.0);
        assert!((0.0..100.0).contains(&warn_percentage));

        let mut total_messages = 0;
        let mut bad_messages = 0;
        let mut longest_latency = Timestamp::from_us(0);
        for proc in self.procs.values() {
            for ((_, variant_id), meta_tasks) in &proc.meta_tasks {
                let variant = self.meta_variants.get(variant_id).unwrap();
                if !variant.message || variant.ordered_vc {
                    continue;
                }
                total_messages += meta_tasks.len();
                for meta_uid in meta_tasks {
                    let meta_task = proc.entry(*meta_uid);
                    let latency =
                        meta_task.time_range.ready.unwrap() - meta_task.time_range.create.unwrap();
                    if threshold <= latency.to_us() {
                        bad_messages += 1;
                    }
                    longest_latency = max(longest_latency, latency);
                }
            }
        }
        if total_messages == 0 {
            return;
        }
        let percentage = 100.0 * bad_messages as f64 / total_messages as f64;
        if warn_percentage <= percentage {
            for _ in 0..5 {
                println!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            }
            println!(
                "WARNING: A significant number of long latency messages \
                    were detected during this run meaning that the network \
                    was likely congested and could be causing a significant \
                    performance degredation. We detected {} messages that took \
                    longer than {:.2}us to run, representing {:.2}% of {} total \
                    messages. The longest latency message required {:.2}us to \
                    execute. Please report this case to the Legion developers \
                    along with an accompanying Legion Prof profile so we can \
                    better understand why the network is so congested.",
                bad_messages,
                threshold,
                percentage,
                total_messages,
                longest_latency.to_us()
            );
            for _ in 0..5 {
                println!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            }
        }
    }

    pub fn sort_time_range(&mut self) {
        self.procs
            .par_iter_mut()
            .for_each(|(_, proc)| proc.sort_time_range(&self.fevents));
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

    pub fn filter_output(&mut self) {
        if self.visible_nodes.is_empty() {
            return;
        }
        for (_, proc) in self.procs.iter_mut() {
            let node_id = proc.proc_id.node_id();
            if !self.visible_nodes.contains(&node_id) {
                proc.visible = false;
            }
        }

        let mut memid_to_be_deleted: Vec<MemID> = Vec::new();
        for (mem_id, mem) in self.mems.iter_mut() {
            let node_id = mem.mem_id.node_id();
            if !self.visible_nodes.contains(&node_id) {
                mem.visible = false;
                memid_to_be_deleted.push(*mem_id);
            }
        }

        for (_, chan) in self.chans.iter_mut() {
            let mut src_node_id: Option<NodeID> = None;
            let mut dst_node_id: Option<NodeID> = None;
            if let Some(src_mem) = chan.chan_id.src {
                src_node_id = Some(src_mem.node_id());
            }
            if let Some(dst_mem) = chan.chan_id.dst {
                dst_node_id = Some(dst_mem.node_id());
            }
            // DepPart
            if src_node_id.is_none() && dst_node_id.is_none() {
                continue;
            } else {
                if !src_node_id.map_or(false, |n| self.visible_nodes.contains(&n))
                    && !dst_node_id.map_or(false, |n| self.visible_nodes.contains(&n))
                {
                    chan.visible = false;
                } else {
                    // we need to keep memory if it is chan.src/dst
                    if let Some(src_mem) = chan.chan_id.src {
                        memid_to_be_deleted.retain(|value| *value != src_mem);
                    }
                    if let Some(dst_mem) = chan.chan_id.dst {
                        memid_to_be_deleted.retain(|value| *value != dst_mem);
                    }
                }
            }
        }

        // if filter input is enabled, we remove invisible proc/mem/chan
        // otherwise, we keep a full state
        if Config::filter_input() {
            self.procs.retain(|_, proc| proc.visible);
        }
        if Config::filter_input() {
            self.mems
                .retain(|&mem_id, _| !memid_to_be_deleted.contains(&mem_id));
            self.mem_proc_affinity
                .retain(|&mem_id, _| !memid_to_be_deleted.contains(&mem_id));
        }
        if Config::filter_input() {
            self.chans.retain(|_, chan| chan.visible);
        }
    }

    pub fn is_on_visible_nodes<'a>(visible_nodes: &'a Vec<NodeID>, node_id: NodeID) -> bool {
        visible_nodes.is_empty() || visible_nodes.contains(&node_id)
    }
}

#[derive(Debug, Default)]
pub struct SpyState {
    has_spy_data: bool,
    spy_ops: BTreeMap<OpID, SpyOp>,
    spy_op_by_precondition: BTreeMap<EventID, BTreeSet<OpID>>,
    spy_op_by_postcondition: BTreeMap<EventID, BTreeSet<OpID>>,
    spy_op_parent: BTreeMap<OpID, OpID>,
    spy_op_children: BTreeMap<OpID, BTreeSet<OpID>>,
    pub spy_op_deps: BTreeMap<ProfUID, Dependencies>,
    spy_events: BTreeMap<EventID, SpyEvent>,
    pub critical_path: Vec<ProfUID>,
}

impl SpyState {
    fn create_spy_event_depencence(&mut self, pre: EventID, post: EventID) {
        assert!(pre != post);
        self.spy_events
            .entry(post)
            .or_insert_with(SpyEvent::new)
            .preconditions
            .insert(pre);
        self.spy_events
            .entry(pre)
            .or_insert_with(SpyEvent::new)
            .postconditions
            .insert(post);
    }

    fn create_spy_op(&mut self, op: OpID, pre: EventID, post: EventID) {
        let old = self.spy_ops.insert(op, SpyOp::new(pre, post));
        // Apparently we can end up with duplicate logging containing NO_EVENTs
        if let Some(SpyOp {
            precondition,
            postcondition,
        }) = old
        {
            assert!(precondition == pre || precondition.0 == 0);
            assert!(postcondition == post || postcondition.0 == 0);
        }
        self.spy_op_by_precondition
            .entry(pre)
            .or_insert_with(BTreeSet::new)
            .insert(op);
        self.spy_op_by_postcondition
            .entry(post)
            .or_insert_with(BTreeSet::new)
            .insert(op);
    }

    fn create_spy_op_parent(&mut self, parent: OpID, child: OpID) {
        if let Some(old) = self.spy_op_parent.insert(child, parent) {
            assert!(old == parent);
        }
        self.spy_op_children
            .entry(parent)
            .or_insert_with(BTreeSet::new)
            .insert(child);
    }

    pub fn process_spy_records(&mut self, records: &Vec<spy::serialize::Record>) {
        for record in records {
            process_spy_record(record, self);
        }
        assert!(self.has_spy_data, "no Legion Spy logs in logfile");
    }

    fn traverse_dag_pre<V, N, F>(root: V, neighbors: N, mut pre: F)
    where
        V: std::marker::Copy + Ord,
        N: Fn(&mut Vec<V>, V),
        F: FnMut(V),
    {
        let mut visited = BTreeSet::new();
        let mut stack = Vec::new();
        stack.push(root);
        while let Some(node) = stack.pop() {
            if visited.contains(&node) {
                continue;
            }
            visited.insert(node);

            neighbors(&mut stack, node);

            pre(node);
        }
    }

    fn traverse_dag_post<V, I, J, N, F>(roots: I, neighbors: N, mut post: F)
    where
        V: std::marker::Copy + Ord,
        I: Iterator<Item = V>,
        J: Iterator<Item = V>,
        N: Fn(V) -> J,
        F: FnMut(V),
    {
        let mut visited = BTreeSet::new();
        let mut stack = Vec::new();
        for root in roots {
            stack.push((root, true));
            while let Some((node, first_pass)) = stack.pop() {
                if first_pass {
                    if visited.contains(&node) {
                        continue;
                    }
                    visited.insert(node);
                    stack.push((node, false));
                    stack.extend(neighbors(node).map(|x| (x, true)));
                } else {
                    post(node);
                }
            }
        }
    }

    fn compute_op_preconditions(
        op: &SpyOp,
        deps: &mut Dependencies,
        op_prof_uid: &BTreeMap<OpID, ProfUID>,
        spy_op_by_postcondition: &BTreeMap<EventID, BTreeSet<OpID>>,
        spy_events: &BTreeMap<EventID, SpyEvent>,
    ) {
        let neighbors = |stack: &mut Vec<_>, event_id| {
            if let Some(event) = spy_events.get(&event_id) {
                stack.extend(event.preconditions.iter());
            }
        };
        let visit = |event_id| {
            if let Some(op_ids) = spy_op_by_postcondition.get(&event_id) {
                for op_id in op_ids {
                    if let Some(prof_uid) = op_prof_uid.get(op_id) {
                        deps.in_.insert(*prof_uid);
                    }
                }
            }
        };
        Self::traverse_dag_pre(op.precondition, neighbors, visit);
    }

    fn compute_op_postconditions(
        op: &SpyOp,
        deps: &mut Dependencies,
        op_prof_uid: &BTreeMap<OpID, ProfUID>,
        spy_op_by_precondition: &BTreeMap<EventID, BTreeSet<OpID>>,
        spy_events: &BTreeMap<EventID, SpyEvent>,
    ) {
        let neighbors = |stack: &mut Vec<_>, event_id| {
            if let Some(event) = spy_events.get(&event_id) {
                stack.extend(event.postconditions.iter());
            }
        };
        let visit = |event_id| {
            if let Some(op_ids) = spy_op_by_precondition.get(&event_id) {
                for op_id in op_ids {
                    if let Some(prof_uid) = op_prof_uid.get(op_id) {
                        deps.out.insert(*prof_uid);
                    }
                }
            }
        };
        Self::traverse_dag_pre(op.postcondition, neighbors, visit);
    }

    fn compute_op_parent(
        op_id: OpID,
        deps: &mut Dependencies,
        op_prof_uid: &BTreeMap<OpID, ProfUID>,
        spy_op_parent: &BTreeMap<OpID, OpID>,
    ) {
        let mut stack = Vec::new();
        stack.push(op_id);
        while let Some(node) = stack.pop() {
            if let Some(parent) = spy_op_parent.get(&node) {
                if let Some(parent_uid) = op_prof_uid.get(parent) {
                    deps.parent.insert(*parent_uid);
                } else {
                    stack.push(*parent);
                }
            }
        }
    }

    fn compute_op_children(
        op_id: OpID,
        deps: &mut Dependencies,
        op_prof_uid: &BTreeMap<OpID, ProfUID>,
        spy_op_children: &BTreeMap<OpID, BTreeSet<OpID>>,
    ) {
        let mut stack = Vec::new();
        stack.push(op_id);
        while let Some(node) = stack.pop() {
            if let Some(children) = spy_op_children.get(&node) {
                for child in children {
                    if let Some(child_uid) = op_prof_uid.get(child) {
                        deps.children.insert(*child_uid);
                    } else {
                        stack.push(*child);
                    }
                }
            }
        }
    }

    fn toposort_graph(&mut self) -> Vec<ProfUID> {
        let mut postorder = Vec::new();
        let neighbors = |node| {
            let deps = self.spy_op_deps.get(&node).unwrap();
            deps.in_.iter().copied()
        };
        let visit = |node| {
            postorder.push(node);
        };
        Self::traverse_dag_post(self.spy_op_deps.keys().copied(), neighbors, visit);
        postorder
    }

    fn transitive_reduce_graph(&mut self, toposort: &Vec<ProfUID>) {
        // TODO: Elliott: legion_spy.py computes this with a bit set,
        // which may be more efficient.
        let mut reachable: BTreeMap<ProfUID, BTreeSet<ProfUID>> = BTreeMap::new();
        for root in toposort {
            // Compute the reachable sets in topological order to
            // minimize the size of the graph we have to traverse
            let mut root_reachable = BTreeSet::new();
            let deps = self.spy_op_deps.get(root).unwrap();
            for node in &deps.in_ {
                // Ok to unwrap here as we're walking in toposort
                // order to make sure this has been precomputed
                let node_reachable = reachable.get(node).unwrap();
                root_reachable.extend(node_reachable.iter());
            }

            let mut to_remove = Vec::new();
            for node in &deps.in_ {
                if root_reachable.contains(node) {
                    to_remove.push(*node);
                } else {
                    root_reachable.insert(*node);
                }
            }
            reachable.insert(*root, root_reachable);

            for node in to_remove {
                let root_deps = self.spy_op_deps.get_mut(root).unwrap();
                assert!(root_deps.in_.remove(&node));
                let node_deps = self.spy_op_deps.get_mut(&node).unwrap();
                assert!(node_deps.out.remove(root));
            }
        }
    }

    fn simplify_spy_graph(&mut self) {
        let toposort = self.toposort_graph();

        self.transitive_reduce_graph(&toposort);
    }

    fn compute_critical_path(&mut self, state: &State) {
        // Postorder DFS walking both in and child edges, computing the
        // longest path at each node based on the sum of the longest input and
        // longest child
        type Path = (u64, Vec<ProfUID>);
        fn path_max<'a>(a: &'a Path, b: &'a Path) -> &'a Path {
            if a.0 > b.0 {
                a
            } else {
                b
            }
        }

        let empty_path = (0, Vec::new());

        let mut longest_paths = BTreeMap::<ProfUID, Path>::new();
        let neighbors = |node| {
            let deps = self.spy_op_deps.get(&node).unwrap();
            let children = deps.children.iter().copied();
            children.chain(deps.in_.iter().copied())
        };
        let visit = |node| {
            let deps = self.spy_op_deps.get(&node).unwrap();
            let path = |dep| longest_paths.get(dep).unwrap();
            let long_in = deps.in_.iter().map(path).fold(&empty_path, path_max);
            let long_child = deps.children.iter().map(path).fold(&empty_path, path_max);
            let duration = long_in.0 + long_child.0 + state.compute_duration(node);
            let mut path = long_in.1.to_owned();
            path.extend(long_child.1.iter());
            path.push(node);
            longest_paths.insert(node, (duration, path));
        };
        Self::traverse_dag_post(self.spy_op_deps.keys().copied(), neighbors, visit);
        self.critical_path = longest_paths
            .values()
            .fold(&empty_path, path_max)
            .1
            .to_owned();
    }

    pub fn postprocess_spy_records(&mut self, state: &State) {
        if !self.has_spy_data {
            println!("No Legion Spy data, skipping postprocess step");
            return;
        }

        // Process tasks first
        for op_id in state.tasks.keys() {
            let prof_uid = state.op_prof_uid.get(op_id).unwrap();
            let deps = self
                .spy_op_deps
                .entry(*prof_uid)
                .or_insert_with(Dependencies::new);
            let op = self.spy_ops.get(op_id).expect("missing dependecies for op");
            Self::compute_op_preconditions(
                op,
                deps,
                &state.op_prof_uid,
                &self.spy_op_by_postcondition,
                &self.spy_events,
            );
            Self::compute_op_postconditions(
                op,
                deps,
                &state.op_prof_uid,
                &self.spy_op_by_precondition,
                &self.spy_events,
            );
            Self::compute_op_parent(*op_id, deps, &state.op_prof_uid, &self.spy_op_parent);
            Self::compute_op_children(*op_id, deps, &state.op_prof_uid, &self.spy_op_children);
        }

        // Now add the implicit dependencies on meta tasks/mapper calls/etc.
        for proc in state.procs.values() {
            for (uid, entry) in &proc.entries {
                if let ProcEntryKind::ProfTask = entry.kind {
                    // FIXME: Elliott: legion_prof.py seems to think ProfTask
                    // has an op_id not an initiation_op, so we have to work
                    // around that here
                    continue;
                }
                if let (Some(initiation_op), None) = (entry.initiation_op, entry.op_id) {
                    if let Some(task) = state.find_task(initiation_op) {
                        let task_stop = task.time_range.stop;
                        let task_uid = task.base.prof_uid;
                        let before = entry.time_range.stop < task_stop;

                        let task_deps = self
                            .spy_op_deps
                            .entry(task_uid)
                            .or_insert_with(Dependencies::new);
                        if before {
                            task_deps.in_.insert(*uid);
                        } else {
                            task_deps.out.insert(*uid);
                        }

                        let entry_deps = self
                            .spy_op_deps
                            .entry(*uid)
                            .or_insert_with(Dependencies::new);
                        if before {
                            entry_deps.out.insert(task_uid);
                        } else {
                            entry_deps.in_.insert(task_uid);
                        }
                    }
                }
            }
        }

        // Reduce the graph
        self.simplify_spy_graph();

        self.compute_critical_path(state);
    }
}

fn process_record(
    record: &Record,
    state: &mut State,
    insts: &mut BTreeMap<InstUID, Inst>,
    copies: &mut BTreeMap<EventID, Copy>,
    fills: &mut BTreeMap<EventID, Fill>,
    call_threshold: Timestamp,
) {
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
        Record::MetaDesc {
            kind,
            message,
            ordered_vc,
            name,
        } => {
            state
                .meta_variants
                .entry(*kind)
                .or_insert_with(|| Variant::new(*kind, *message, *ordered_vc, name));
        }
        Record::OpDesc { kind, name } => {
            let kind = OpKindID(*kind);
            state
                .op_kinds
                .entry(kind)
                .or_insert_with(|| OpKind::new(name.clone()));
        }
        Record::MaxDimDesc { max_dim } => {
            state.max_dim = *max_dim;
        }
        Record::MachineDesc { num_nodes, .. } => {
            state.num_nodes = *num_nodes;
        }
        Record::ZeroTime { zero_time } => {
            state.zero_time = TimestampDelta(*zero_time);
        }
        Record::CalibrationErr { calibration_err } => {
            state._calibration_err = *calibration_err;
        }
        Record::ProcDesc { proc_id, kind, .. } => {
            let kind = match ProcKind::try_from(*kind) {
                Ok(x) => x,
                Err(_) => panic!("bad processor kind"),
            };
            state
                .procs
                .entry(*proc_id)
                .or_insert_with(|| Proc::new(*proc_id, kind));
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
        Record::ProcMDesc {
            proc_id,
            mem_id,
            bandwidth,
            latency,
        } => {
            state
                .mem_proc_affinity
                .entry(*mem_id)
                .or_insert_with(|| MemProcAffinity::new(*mem_id, *bandwidth, *latency, *proc_id))
                .update_best_aff(*proc_id, *bandwidth, *latency);
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
            inst_uid,
            ispace_id,
            fspace_id,
            tree_id,
        } => {
            let fspace_id = FSpaceID(*fspace_id as u64);
            state.find_field_space_mut(fspace_id);
            state
                .create_inst(*inst_uid, insts)
                .add_ispace(*ispace_id)
                .add_fspace(fspace_id)
                .set_tree(*tree_id);
        }
        Record::PhysicalInstLayoutDesc {
            inst_uid,
            field_id,
            fspace_id,
            has_align,
            eqk,
            align_desc,
        } => {
            let fspace_id = FSpaceID(*fspace_id as u64);
            state.find_field_space_mut(fspace_id);
            state
                .create_inst(*inst_uid, insts)
                .add_field(fspace_id, *field_id)
                .add_align_desc(fspace_id, *field_id, *eqk, *align_desc, *has_align);
        }
        Record::PhysicalInstDimOrderDesc {
            inst_uid,
            dim,
            dim_kind,
        } => {
            let dim = Dim(*dim);
            let dim_kind = match DimKind::try_from(*dim_kind) {
                Ok(x) => x,
                Err(_) => unreachable!("bad dim kind"),
            };
            state
                .create_inst(*inst_uid, insts)
                .add_dim_order(dim, dim_kind);
        }
        Record::PhysicalInstanceUsage {
            inst_uid,
            op_id,
            index_id,
            field_id,
        } => {
            state.create_op(*op_id);
            let operation_inst_info = OperationInstInfo::new(*inst_uid, *index_id, *field_id);
            state
                .find_op_mut(*op_id)
                .unwrap()
                .operation_inst_infos
                .push(operation_inst_info);
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
                .or_insert_with(|| Variant::new(*variant_id, false, false, name))
                .set_task(*task_id);
        }
        Record::OperationInstance {
            op_id,
            parent_id,
            kind,
            provenance,
        } => {
            let kind = OpKindID(*kind);
            state
                .create_op(*op_id)
                .set_parent_id(*parent_id)
                .set_kind(kind)
                .set_provenance(provenance);
            // Hack: we have to do this in two places, because we don't know what
            // order the logger calls are going to come in. If the task gets
            // logged first, this will come back Some(_) and we'll store it below.
            if let Some(task) = state.find_task_mut(*op_id) {
                task.initiation_op = Some(*parent_id);
            }
        }
        Record::MultiTask { op_id, task_id } => {
            state.create_op(*op_id);
            state
                .multi_tasks
                .entry(*op_id)
                .or_insert_with(|| MultiTask::new(*op_id, *task_id));
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
                .find_last_meta_mut(*op_id, *lg_id)
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
            creator,
            fevent,
        } => {
            let time_range = TimeRange::new_full(*create, *ready, *start, *stop);
            state.create_task(
                *op_id,
                *proc_id,
                *task_id,
                *variant_id,
                time_range,
                *creator,
                *fevent,
            );
            state.update_last_time(*stop);
        }
        Record::GPUTaskInfo {
            op_id,
            task_id,
            variant_id,
            proc_id,
            create,
            ready,
            start,
            stop,
            gpu_start,
            gpu_stop,
            creator,
            fevent,
        } => {
            // it is possible that gpu_start is larger than gpu_stop when cuda hijack is disabled,
            // because the cuda event completions of these two timestamp may be out of order when
            // they are not in the same stream. Usually, when it happened, it means the GPU task is tiny.
            let mut gpu_start = *gpu_start;
            if gpu_start > *gpu_stop {
                gpu_start.0 = gpu_stop.0 - 1;
            }
            let gpu_range = TimeRange::new_start(gpu_start, *gpu_stop);
            state.create_gpu_kernel(*op_id, *proc_id, *task_id, *variant_id, gpu_range, *fevent);
            let time_range = TimeRange::new_full(*create, *ready, *start, *stop);
            state.create_task(
                *op_id,
                *proc_id,
                *task_id,
                *variant_id,
                time_range,
                *creator,
                *fevent,
            );
            if *stop < *gpu_stop {
                state.update_last_time(*gpu_stop);
            } else {
                state.update_last_time(*stop);
            }
        }
        Record::MetaInfo {
            op_id,
            lg_id,
            proc_id,
            create,
            ready,
            start,
            stop,
            creator,
            fevent,
        } => {
            let time_range = TimeRange::new_full(*create, *ready, *start, *stop);
            state.create_meta(*op_id, *lg_id, *proc_id, time_range, *creator, *fevent);
            state.update_last_time(*stop);
        }
        Record::CopyInfo {
            op_id,
            size,
            create,
            ready,
            start,
            stop,
            creator,
            fevent,
            collective,
        } => {
            let time_range = TimeRange::new_full(*create, *ready, *start, *stop);
            state.create_op(*op_id);
            state.create_copy(
                time_range,
                *op_id,
                *size,
                *creator,
                *fevent,
                *collective,
                copies,
            );
            state.update_last_time(*stop);
        }
        Record::CopyInstInfo {
            src,
            dst,
            src_fid,
            dst_fid,
            src_inst,
            dst_inst,
            fevent,
            num_hops,
            indirect,
        } => {
            let copy = copies.get_mut(fevent).unwrap();
            let mut src_mem = None;
            if *src != MemID(0) {
                src_mem = Some(*src);
            }
            let mut dst_mem = None;
            if *dst != MemID(0) {
                dst_mem = Some(*dst);
            }
            let copy_inst_info = CopyInstInfo::new(
                src_mem, dst_mem, *src_fid, *dst_fid, *src_inst, *dst_inst, *fevent, *num_hops,
                *indirect,
            );
            copy.add_copy_inst_info(copy_inst_info);
        }
        Record::FillInfo {
            op_id,
            size,
            create,
            ready,
            start,
            stop,
            creator,
            fevent,
        } => {
            let time_range = TimeRange::new_full(*create, *ready, *start, *stop);
            state.create_op(*op_id);
            state.create_fill(time_range, *op_id, *size, *creator, *fevent, fills);
            state.update_last_time(*stop);
        }
        Record::FillInstInfo {
            dst,
            fid,
            dst_inst,
            fevent,
        } => {
            let fill_inst_info = FillInstInfo::new(*dst, *fid, *dst_inst, *fevent);
            let fill = fills.get_mut(fevent).unwrap();
            fill.add_fill_inst_info(fill_inst_info);
        }
        Record::InstTimelineInfo {
            inst_uid,
            inst_id,
            mem_id,
            size,
            op_id,
            create,
            ready,
            destroy,
            creator,
        } => {
            state.create_op(*op_id);
            state.insts.entry(*inst_uid).or_insert_with(|| *mem_id);
            state
                .create_inst(*inst_uid, insts)
                .set_inst_id(*inst_id)
                .set_op_id(*op_id)
                .set_start_stop(*create, *ready, *destroy)
                .set_mem(*mem_id)
                .set_size(*size)
                .set_creator(*creator);
            state.update_last_time(*destroy);
        }
        Record::PartitionInfo {
            op_id,
            part_op,
            create,
            ready,
            start,
            stop,
            creator,
        } => {
            let part_op = match DepPartKind::try_from(*part_op) {
                Ok(x) => x,
                Err(_) => panic!("bad deppart kind"),
            };
            let time_range = TimeRange::new_full(*create, *ready, *start, *stop);
            state.create_deppart(*op_id, part_op, time_range, *creator);
            state.update_last_time(*stop);
        }
        Record::MapperCallInfo {
            kind,
            op_id,
            start,
            stop,
            proc_id,
            fevent,
        } => {
            // Check to make sure it is above the call threshold
            if call_threshold <= (*stop - *start) {
                assert!(state.mapper_call_kinds.contains_key(kind));
                let time_range = TimeRange::new_start(*start, *stop);
                state.create_mapper_call(*kind, *proc_id, *op_id, time_range, *fevent);
                state.update_last_time(*stop);
            }
        }
        Record::RuntimeCallInfo {
            kind,
            start,
            stop,
            proc_id,
            fevent,
        } => {
            // Check to make sure that it is above the call threshold
            if call_threshold <= (*stop - *start) {
                assert!(state.runtime_call_kinds.contains_key(kind));
                let time_range = TimeRange::new_start(*start, *stop);
                state.create_runtime_call(*kind, *proc_id, time_range, *fevent);
                state.update_last_time(*stop);
            }
        }
        Record::ProfTaskInfo {
            proc_id,
            op_id,
            start,
            stop,
            creator,
            fevent,
        } => {
            let time_range = TimeRange::new_start(*start, *stop);
            state.create_prof_task(*proc_id, *op_id, time_range, *creator, *fevent);
            state.update_last_time(*stop);
        }
    }
}

fn process_spy_record(record: &spy::serialize::Record, state: &mut SpyState) {
    use spy::serialize::Record;

    match record {
        Record::SpyLogging => unimplemented!("legion_prof_rs requires detailed Legion Spy logging"),
        Record::SpyDetailedLogging => {
            state.has_spy_data = true;
        }
        Record::EventDependence { id1, id2 } => {
            state.create_spy_event_depencence((*id1).into(), (*id2).into());
        }

        Record::OperationEvents { uid, pre, post } => {
            state.create_spy_op((*uid).into(), (*pre).into(), (*post).into());
        }
        Record::RealmCopy { pre, post, .. } => {
            state.create_spy_event_depencence((*pre).into(), (*post).into());
        }
        Record::IndirectCopy { pre, post, .. } => {
            state.create_spy_event_depencence((*pre).into(), (*post).into());
        }
        Record::RealmFill { pre, post, .. } => {
            state.create_spy_event_depencence((*pre).into(), (*post).into());
        }
        Record::RealmDepPart { preid, postid, .. } => {
            state.create_spy_event_depencence((*preid).into(), (*postid).into());
        }

        Record::TopTask { ctx, uid, .. } => {
            state.create_spy_op_parent((*ctx).into(), (*uid).into());
        }
        Record::IndividualTask { ctx, uid, .. } => {
            state.create_spy_op_parent((*ctx).into(), (*uid).into());
        }
        Record::IndexTask { ctx, uid, .. } => {
            state.create_spy_op_parent((*ctx).into(), (*uid).into());
        }
        Record::IndexSlice { index, slice, .. } => {
            state.create_spy_op_parent((*index).into(), (*slice).into());
        }
        Record::SliceSlice { slice1, slice2, .. } => {
            state.create_spy_op_parent((*slice1).into(), (*slice2).into());
        }
        Record::SlicePoint {
            slice, point_id, ..
        } => {
            state.create_spy_op_parent((*slice).into(), (*point_id).into());
        }
        Record::PointPoint { point1, point2, .. } => {
            state.create_spy_op_parent((*point1).into(), (*point2).into());
        }
        Record::IndexPoint {
            index, point_id, ..
        } => {
            state.create_spy_op_parent((*index).into(), (*point_id).into());
        }
        _ => {} // ok, ignore everything else
    }
}
