use std::fs::File;
use std::io::{self, BufRead, Read, Seek};
use std::num::{NonZeroU32, NonZeroU64};
use std::path::Path;
use std::str;
use std::sync::Arc;

use buffer_redux::{BufReader, policy::MinBuffered};

use flate2::read::GzDecoder;

use foldhash::{HashMap, HashMapExt};

use nonmax::NonMaxU64;

use nom::{
    IResult,
    bytes::streaming::{tag, take_till, take_while1},
    character::{is_alphanumeric, is_digit, is_newline},
    combinator::{map, map_opt, map_res, opt},
    multi::{many_m_n, separated_list1},
    number::streaming::{le_i32, le_i64, le_u8, le_u32, le_u64},
};

use serde::Serialize;

use crate::geometry::ISpaceID;
use crate::state::{
    BacktraceID, EventID, FSpaceID, FieldID, IPartID, InstID, MapperCallKindID, MapperID, MemID,
    NodeID, OpID, PrivilegeMode, ProcID, ProvenanceID, ReductionID, RuntimeCallKindID, State,
    TaskID, Timestamp, TreeID, VariantID,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ValueFormat {
    Array,
    BacktraceID,
    Bool,
    DepPartOpKind,
    IDType,
    InstID,
    MapperID,
    MappingCallKind,
    MaxDim,
    MemID,
    MemKind,
    MessageKind,
    Point,
    Uuid,
    ProcID,
    ProcKind,
    ProvenanceID,
    RuntimeCallKind,
    String,
    TaskID,
    Timestamp,
    U32,
    U64,
    I64,
    UniqueID,
    VariantID,
}

#[derive(Debug, Clone)]
pub struct FieldFormat {
    pub name: String,
    pub value: ValueFormat,
    pub size: i32,
}

#[derive(Debug, Clone)]
pub struct RecordFormat {
    pub id: u32,
    pub name: String,
    pub fields: Vec<FieldFormat>,
}

// Note: we use different, more specialized types for some of the ones
// below.
type DepPartOpKind = i32;
// type IDType = u64;
type MaxDim = i32;
type MemKind = i32;
// type MessageKind = i32;
type ProcKind = i32;
type UniqueID = u64;

#[derive(Debug, Clone, Serialize)]
pub struct Array(pub Vec<i64>);

#[derive(Debug, Clone, Serialize)]
pub struct Point(pub Vec<i64>);

#[derive(Debug, Clone, Serialize)]
pub struct Uuid(pub Vec<u8>);

#[rustfmt::skip]
#[derive(Debug, Clone, Serialize)]
pub enum Record {
    MapperName { mapper_id: MapperID, mapper_proc: ProcID, name: String },
    MapperCallDesc { kind: MapperCallKindID, name: String },
    RuntimeCallDesc { kind: RuntimeCallKindID, name: String },
    MetaDesc { kind: VariantID, message: bool, ordered_vc: bool, name: String },
    OpDesc { kind: u32, name: String },
    MaxDimDesc { max_dim: MaxDim },
    RuntimeConfig { debug: bool, spy: bool, gc: bool, inorder: bool, safe_mapper: bool, safe_runtime: bool, safe_ctrlrepl: bool, part_checks: bool, bounds_checks: bool, resilient: bool },
    MachineDesc { node_id: NodeID, num_nodes: u32, hostname: String, host_id: u64, process_id: u32 },
    ZeroTime { zero_time: i64 },
    Provenance { pid: ProvenanceID, provenance: String },
    ProcDesc { proc_id: ProcID, kind: ProcKind, cuda_device_uuid: Uuid },
    MemDesc { mem_id: MemID, kind: MemKind, capacity: u64 },
    ProcMDesc { proc_id: ProcID, mem_id: MemID, bandwidth: u32, latency: u32 },
    IndexSpacePointDesc { ispace_id: ISpaceID, dim: u32, rem: Point },
    IndexSpaceRectDesc { ispace_id: ISpaceID, dim: u32, rem: Array },
    IndexSpaceEmptyDesc { ispace_id: ISpaceID },
    FieldDesc { fspace_id: FSpaceID, field_id: FieldID, size: u64, name: String },
    FieldSpaceDesc { fspace_id: FSpaceID, name: String },
    PartDesc { unique_id: IPartID, name: String },
    IndexSpaceDesc { ispace_id: ISpaceID, name: String },
    IndexSubSpaceDesc { parent_id: IPartID, ispace_id: ISpaceID },
    IndexPartitionDesc { parent_id: ISpaceID, unique_id: IPartID, disjoint: bool, point0: u64 },
    IndexSpaceSizeDesc { ispace_id: ISpaceID, dense_size: u64, sparse_size: u64, is_sparse: bool },
    LogicalRegionDesc { ispace_id: ISpaceID, fspace_id: FSpaceID, tree_id: TreeID, name: String },
    PhysicalInstRegionDesc { fevent: EventID, ispace_id: ISpaceID, fspace_id: FSpaceID, tree_id: TreeID },
    PhysicalInstLayoutDesc { fevent: EventID, field_id: FieldID, fspace_id: FSpaceID, has_align: bool, eqk: u32, align_desc: u32 },
    PhysicalInstDimOrderDesc { fevent: EventID, dim: u32, dim_kind: u32 },
    PhysicalInstanceSpaces { fevent: EventID, union_space: Option<ISpaceID>, piece_space: Option<ISpaceID> },
    PhysicalInstanceUsage { inst_event: EventID, fevent: EventID, op_id: OpID, start: Timestamp, stop: Option<Timestamp>, index_expr: Option<ISpaceID>, privilege: PrivilegeMode, field: FieldID, index: Option<u32> },
    TaskKind { task_id: TaskID, name: String, overwrite: bool },
    TaskVariant { task_id: TaskID, variant_id: VariantID, name: String },
    OperationInstance { op_id: OpID, parent_id: Option<OpID>, kind: u32, provenance: Option<ProvenanceID> },
    MultiTask { op_id: OpID, task_id: TaskID },
    SliceOwner { parent_id: UniqueID, op_id: OpID },
    TaskWaitInfo { op_id: OpID, task_id: TaskID, variant_id: VariantID, wait_start: Timestamp, wait_ready: Timestamp, wait_end: Timestamp, wait_event: EventID },
    MetaWaitInfo { op_id: OpID, lg_id: VariantID, wait_start: Timestamp, wait_ready: Timestamp, wait_end: Timestamp, wait_event: EventID },
    TaskInfo { op_id: OpID, task_id: TaskID, variant_id: VariantID, proc_id: ProcID, create: Timestamp, ready: Timestamp, start: Timestamp, stop: Timestamp, creator: Option<EventID>, critical: Option<EventID>, fevent: EventID  },
    ImplicitTaskInfo { op_id: OpID, task_id: TaskID, variant_id: VariantID, proc_id: ProcID, create: Timestamp, ready: Timestamp, start: Timestamp, stop: Timestamp, creator: Option<EventID>, critical: Option<EventID>, fevent: EventID  },
    GPUTaskInfo { op_id: OpID, task_id: TaskID, variant_id: VariantID, proc_id: ProcID, create: Timestamp, ready: Timestamp, start: Timestamp, stop: Timestamp, gpu_start: Timestamp, gpu_stop: Timestamp, creator: Option<EventID>, critical: Option<EventID>, fevent: EventID },
    MetaInfo { op_id: OpID, lg_id: VariantID, proc_id: ProcID, create: Timestamp, ready: Timestamp, start: Timestamp, stop: Timestamp, creator: Option<EventID>, critical: Option<EventID>, fevent: EventID },
    MessageInfo { op_id: OpID, lg_id: VariantID, proc_id: ProcID, spawn: Timestamp, create: Timestamp, ready: Timestamp, start: Timestamp, stop: Timestamp, creator: Option<EventID>, critical: Option<EventID>, fevent: EventID },
    CopyInfo { op_id: OpID, size: u64, create: Timestamp, ready: Timestamp, start: Timestamp, stop: Timestamp, creator: Option<EventID>, critical: Option<EventID>, fevent: EventID, collective: u32, redop: Option<ReductionID>, copy_expr: Option<ISpaceID> },
    CopyInstInfo { src: Option<MemID>, dst: Option<MemID>, src_fid: FieldID, dst_fid: FieldID, src_inst: Option<EventID>, dst_inst: Option<EventID>, src_expr: Option<ISpaceID>, dst_expr: Option<ISpaceID>, fevent: EventID, num_hops: u32, indirect: bool },
    FillInfo { op_id: OpID, size: u64, create: Timestamp, ready: Timestamp, start: Timestamp, stop: Timestamp, creator: Option<EventID>, critical: Option<EventID>, fevent: EventID, collective: u32, fill_expr: Option<ISpaceID> },
    FillInstInfo { dst: MemID, fid: FieldID, dst_inst: EventID, fevent: EventID },
    InstTimelineInfo { fevent: EventID, inst_id: InstID, mem_id: MemID, size: u64, op_id: OpID, create: Timestamp, ready: Timestamp, destroy: Timestamp, creator: EventID },
    PartitionInfo { op_id: OpID, part_op: DepPartOpKind, create: Timestamp, ready: Timestamp, start: Timestamp, stop: Timestamp, creator: Option<EventID>, critical: Option<EventID>, fevent: EventID },
    PartInstInfo { src: MemID, fid: FieldID, src_inst: EventID, src_expr: Option<ISpaceID>, fevent: EventID },
    MapperCallInfo { mapper_id: MapperID, mapper_proc: ProcID, kind: MapperCallKindID, op_id: OpID, start: Timestamp, stop: Timestamp, proc_id: ProcID, fevent: Option<EventID> },
    RuntimeCallInfo { kind: RuntimeCallKindID, start: Timestamp, stop: Timestamp, proc_id: ProcID, fevent: Option<EventID> },
    ApplicationCallInfo { provenance: ProvenanceID, start: Timestamp, stop: Timestamp, proc_id: ProcID, fevent: Option<EventID> },
    AsyncEffectInfo { provenance: ProvenanceID, start: Timestamp, stop: Timestamp, proc_id: ProcID, creator: EventID, fevent: Option<EventID> },
    ProfTaskInfo { proc_id: ProcID, op_id: OpID, start: Timestamp, stop: Timestamp, creator: EventID, fevent: EventID, completion: bool },
    CalibrationErr { calibration_err: i64 },
    BacktraceDesc { backtrace_id: BacktraceID , backtrace: String },
    EventWaitInfo { proc_id: ProcID, fevent: EventID, event: EventID, provenance: Option<ProvenanceID>, backtrace_id: BacktraceID },
    EventMergerInfo { result: EventID, fevent: EventID, performed: Timestamp, pre0: Option<EventID>, pre1: Option<EventID>, pre2: Option<EventID>, pre3: Option<EventID> },
    EventTriggerInfo { result: EventID, fevent: EventID, precondition: Option<EventID>, performed: Timestamp },
    EventPoisonInfo { result: EventID, fevent: EventID, performed: Timestamp },
    ExternalEventInfo { external: EventID, fevent: EventID, performed: Timestamp, triggered: Timestamp, provenance: ProvenanceID },
    BarrierArrivalInfo { result: EventID, fevent: EventID, precondition: Option<EventID>, performed: Timestamp },
    ReservationAcquireInfo { result: EventID, fevent: EventID, precondition: Option<EventID>, performed: Timestamp, reservation: u64 },
    CompletionQueueInfo { result: EventID, fevent: EventID, performed: Timestamp, pre0: Option<EventID>, pre1: Option<EventID>, pre2: Option<EventID>, pre3: Option<EventID> },
    InstanceReadyInfo { result: Option<EventID>, precondition: Option<EventID>, unique: EventID, performed: Timestamp },
    InstanceRedistrictInfo { result: Option<EventID>, precondition: Option<EventID>, previous: EventID, next: EventID, performed: Timestamp },
    MakeValidInfo { result: EventID, fevent: EventID, space: Option<ISpaceID>, created: Timestamp, triggered: Timestamp },
    FetchMetadataInfo { result: EventID, fevent: EventID, inst_event: EventID, created: Timestamp, triggered: Timestamp },
    SpawnInfo { fevent: EventID, spawn: Timestamp },
}

fn convert_value_format(name: String) -> Option<ValueFormat> {
    match name.as_str() {
        "array" => Some(ValueFormat::Array),
        "BacktraceID" => Some(ValueFormat::BacktraceID),
        "bool" => Some(ValueFormat::Bool),
        "DepPartOpKind" => Some(ValueFormat::DepPartOpKind),
        "IDType" => Some(ValueFormat::IDType),
        "InstID" => Some(ValueFormat::InstID),
        "MapperID" => Some(ValueFormat::MapperID),
        "MappingCallKind" => Some(ValueFormat::MappingCallKind),
        "maxdim" => Some(ValueFormat::MaxDim),
        "MemID" => Some(ValueFormat::MemID),
        "MemKind" => Some(ValueFormat::MemKind),
        "MessageKind" => Some(ValueFormat::MessageKind),
        "point" => Some(ValueFormat::Point),
        "uuid" => Some(ValueFormat::Uuid),
        "uuid_size" => Some(ValueFormat::U32),
        "ProcID" => Some(ValueFormat::ProcID),
        "ProcKind" => Some(ValueFormat::ProcKind),
        "RuntimeCallKind" => Some(ValueFormat::RuntimeCallKind),
        "string" => Some(ValueFormat::String),
        "TaskID" => Some(ValueFormat::TaskID),
        "timestamp_t" => Some(ValueFormat::Timestamp),
        "unsigned" => Some(ValueFormat::U32),
        "unsigned long long" => Some(ValueFormat::U64),
        "long long" => Some(ValueFormat::I64),
        "UniqueID" => Some(ValueFormat::UniqueID),
        "VariantID" => Some(ValueFormat::VariantID),
        _ => None,
    }
}

//
// Text parser utilities
//

fn newline(input: &[u8]) -> IResult<&[u8], ()> {
    let (input, _) = tag("\n")(input)?;
    Ok((input, ()))
}

fn parse_text_i32(input: &[u8]) -> IResult<&[u8], i32> {
    let (input, sign) = opt(tag("-"))(input)?;
    let (input, value) = take_while1(is_digit)(input)?;
    let value: i32 = str::from_utf8(value).unwrap().parse().unwrap();
    Ok((input, if sign.is_none() { value } else { -value }))
}

fn parse_text_u32(input: &[u8]) -> IResult<&[u8], u32> {
    let (input, value) = take_while1(is_digit)(input)?;
    let value = str::from_utf8(value).unwrap().parse().unwrap();
    Ok((input, value))
}

#[inline]
pub fn is_alphanumeric_underscore(chr: u8) -> bool {
    is_alphanumeric(chr) || chr == 95 // underscore
}

#[inline]
pub fn is_alphanumeric_space(chr: u8) -> bool {
    is_alphanumeric_underscore(chr) || chr == 32 // space
}

#[inline]
pub fn is_nul(chr: u8) -> bool {
    chr == 0 // nul
}

fn parse_text_name(input: &[u8]) -> IResult<&[u8], String> {
    let (input, name) = take_while1(is_alphanumeric_underscore)(input)?;
    Ok((input, str::from_utf8(name).unwrap().to_owned()))
}

fn parse_text_type(input: &[u8]) -> IResult<&[u8], String> {
    let (input, name) = take_while1(is_alphanumeric_space)(input)?;
    Ok((input, str::from_utf8(name).unwrap().to_owned()))
}

fn parse_text_version(input: &[u8]) -> IResult<&[u8], String> {
    let (input, name) = take_till(is_newline)(input)?;
    Ok((input, str::from_utf8(name).unwrap().to_owned()))
}

//
// Text parsers for the log file header
//

fn parse_filetype(input: &[u8]) -> IResult<&[u8], (u32, String)> {
    let (input, _) = tag("FileType: BinaryLegionProf v: ")(input)?;
    let (input, prof_version) = parse_text_u32(input)?;
    let (input, _) = newline(input)?;
    let (input, legion_version) = parse_text_version(input)?;
    let (input, _) = newline(input)?;
    Ok((input, (prof_version, legion_version)))
}

fn parse_value_format(input: &[u8]) -> IResult<&[u8], ValueFormat> {
    map_opt(parse_text_type, convert_value_format)(input)
}

fn parse_field_format(input: &[u8]) -> IResult<&[u8], FieldFormat> {
    let (input, name) = parse_text_name(input)?;
    let (input, _) = tag(":")(input)?;
    let (input, value) = parse_value_format(input)?;
    let (input, _) = tag(":")(input)?;
    let (input, size) = parse_text_i32(input)?;
    Ok((input, FieldFormat { name, value, size }))
}

fn parse_record_format(input: &[u8]) -> IResult<&[u8], RecordFormat> {
    let (input, name) = parse_text_name(input)?;
    let (input, _) = tag(" {id:")(input)?;
    let (input, id) = parse_text_u32(input)?;
    let (input, _) = tag(", ")(input)?;
    let (input, fields) = separated_list1(tag(", "), parse_field_format)(input)?;
    let (input, _) = tag("}")(input)?;
    let (input, _) = newline(input)?;
    Ok((input, RecordFormat { id, name, fields }))
}

//
// Binary parsers for basic types used in records
//

fn parse_array(input: &[u8], max_dim: i32) -> IResult<&[u8], Array> {
    assert!(max_dim > -1);
    let n = (max_dim * 2) as usize;
    let (input, values) = many_m_n(n, n, le_i64)(input)?;
    Ok((input, Array(values)))
}
fn parse_bool(input: &[u8]) -> IResult<&[u8], bool> {
    map(le_u8, |x| x != 0)(input)
}
fn parse_point(input: &[u8], max_dim: i32) -> IResult<&[u8], Point> {
    assert!(max_dim > -1);
    let n = max_dim as usize;
    let (input, values) = many_m_n(n, n, le_i64)(input)?;
    Ok((input, Point(values)))
}
fn parse_cuda_device_uuid(input: &[u8]) -> IResult<&[u8], Uuid> {
    let (input, uuid_size) = le_u32(input)?;
    let n = uuid_size as usize;
    let (input, values) = many_m_n(n, n, le_u8)(input)?;
    Ok((input, Uuid(values)))
}
fn parse_string(input: &[u8]) -> IResult<&[u8], String> {
    let (input, value) = map_res(take_till(is_nul), |x: &[u8]| str::from_utf8(x))(input)?;
    let (input, terminator) = le_u8(input)?;
    assert!(is_nul(terminator));
    Ok((input, value.to_owned()))
}
fn parse_privilege_mode(input: &[u8]) -> IResult<&[u8], PrivilegeMode> {
    let (input, privilege) = le_u32(input)?;
    match privilege {
        0 => Ok((input, PrivilegeMode::NoAccess)),
        1 => Ok((input, PrivilegeMode::ReadOnly)),
        2 => Ok((input, PrivilegeMode::WriteOnly)),
        4 => {
            let (input, redop) = parse_redop_id(input)?;
            Ok((input, PrivilegeMode::Reduce(redop)))
        }
        6 => Ok((input, PrivilegeMode::ReadWrite)),
        7 => Ok((input, PrivilegeMode::ReadWrite)),
        _ => {
            panic!("Detected bad privilege {:x}", privilege);
        }
    }
}

//
// Binary parsers for type aliases
//

fn parse_option_event_id(input: &[u8]) -> IResult<&[u8], Option<EventID>> {
    map(le_u64, |x| NonZeroU64::new(x).map(EventID))(input)
}
fn parse_event_id(input: &[u8]) -> IResult<&[u8], EventID> {
    map(le_u64, |x| EventID(NonZeroU64::new(x).unwrap()))(input)
}
fn parse_inst_id(input: &[u8]) -> IResult<&[u8], InstID> {
    map(le_u64, |x| InstID(NonZeroU64::new(x).unwrap()))(input)
}
fn parse_ipart_id(input: &[u8]) -> IResult<&[u8], IPartID> {
    map(le_u64, |x| IPartID(NonZeroU64::new(x).unwrap()))(input)
}
fn parse_ispace_id(input: &[u8]) -> IResult<&[u8], ISpaceID> {
    map(le_u64, |x| ISpaceID(NonZeroU64::new(x).unwrap()))(input)
}
fn parse_option_ispace_id(input: &[u8]) -> IResult<&[u8], Option<ISpaceID>> {
    map(le_u64, |x| NonZeroU64::new(x).map(ISpaceID))(input)
}
fn parse_fspace_id(input: &[u8]) -> IResult<&[u8], FSpaceID> {
    map(le_u64, |x| FSpaceID(NonZeroU64::new(x).unwrap()))(input)
}
fn parse_field_id(input: &[u8]) -> IResult<&[u8], FieldID> {
    map(le_u32, FieldID)(input)
}
fn parse_tree_id(input: &[u8]) -> IResult<&[u8], TreeID> {
    map(le_u32, |x| TreeID(NonZeroU32::new(x).unwrap()))(input)
}
fn parse_mapper_id(input: &[u8]) -> IResult<&[u8], MapperID> {
    map(le_u32, MapperID)(input)
}
fn parse_mapper_call_kind_id(input: &[u8]) -> IResult<&[u8], MapperCallKindID> {
    map(le_u32, MapperCallKindID)(input)
}
fn parse_mem_id(input: &[u8]) -> IResult<&[u8], MemID> {
    map(le_u64, |x| MemID(NonZeroU64::new(x).unwrap()))(input)
}
fn parse_option_mem_id(input: &[u8]) -> IResult<&[u8], Option<MemID>> {
    map(le_u64, |x| NonZeroU64::new(x).map(MemID))(input)
}
fn parse_redop_id(input: &[u8]) -> IResult<&[u8], ReductionID> {
    map(le_u32, |x| ReductionID(NonZeroU32::new(x).unwrap()))(input)
}
fn parse_option_redop_id(input: &[u8]) -> IResult<&[u8], Option<ReductionID>> {
    map(le_u32, |x| NonZeroU32::new(x).map(ReductionID))(input)
}
fn parse_option_op_id(input: &[u8]) -> IResult<&[u8], Option<OpID>> {
    map(le_u64, |x| NonMaxU64::new(x).map(OpID))(input)
}
fn parse_op_id(input: &[u8]) -> IResult<&[u8], OpID> {
    map(le_u64, |x| OpID(NonMaxU64::new(x).unwrap()))(input)
}
fn parse_proc_id(input: &[u8]) -> IResult<&[u8], ProcID> {
    map(le_u64, |x| ProcID(NonZeroU64::new(x).unwrap()))(input)
}
fn parse_runtime_call_kind_id(input: &[u8]) -> IResult<&[u8], RuntimeCallKindID> {
    map(le_u32, RuntimeCallKindID)(input)
}
fn parse_option_provenance_id(input: &[u8]) -> IResult<&[u8], Option<ProvenanceID>> {
    map(le_u64, |x| NonZeroU64::new(x).map(ProvenanceID))(input)
}
fn parse_provenance_id(input: &[u8]) -> IResult<&[u8], ProvenanceID> {
    map(le_u64, |x| ProvenanceID(NonZeroU64::new(x).unwrap()))(input)
}
fn parse_task_id(input: &[u8]) -> IResult<&[u8], TaskID> {
    map(le_u32, TaskID)(input)
}
fn parse_timestamp(input: &[u8]) -> IResult<&[u8], Timestamp> {
    map(le_u64, Timestamp::from_ns)(input)
}
fn parse_option_timestamp(input: &[u8]) -> IResult<&[u8], Option<Timestamp>> {
    map(le_u64, |x| NonMaxU64::new(x).map(Timestamp::from_value))(input)
}
fn parse_variant_id(input: &[u8]) -> IResult<&[u8], VariantID> {
    map(le_u32, VariantID)(input)
}
fn parse_backtrace_id(input: &[u8]) -> IResult<&[u8], BacktraceID> {
    map(le_u64, |x| BacktraceID(NonZeroU64::new(x).unwrap()))(input)
}

//
// Binary parsers for records
//

fn parse_mapper_call_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, kind) = parse_mapper_call_kind_id(input)?;
    let (input, name) = parse_string(input)?;
    Ok((input, Record::MapperCallDesc { kind, name }))
}
fn parse_runtime_call_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, kind) = parse_runtime_call_kind_id(input)?;
    let (input, name) = parse_string(input)?;
    Ok((input, Record::RuntimeCallDesc { kind, name }))
}
fn parse_meta_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, kind) = parse_variant_id(input)?;
    let (input, message) = parse_bool(input)?;
    let (input, ordered_vc) = parse_bool(input)?;
    let (input, name) = parse_string(input)?;
    Ok((
        input,
        Record::MetaDesc {
            kind,
            message,
            ordered_vc,
            name,
        },
    ))
}
fn parse_op_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, kind) = le_u32(input)?;
    let (input, name) = parse_string(input)?;
    Ok((input, Record::OpDesc { kind, name }))
}
fn parse_max_dim_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, max_dim) = le_i32(input)?;
    Ok((input, Record::MaxDimDesc { max_dim }))
}
fn parse_runtime_config(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, debug) = parse_bool(input)?;
    let (input, spy) = parse_bool(input)?;
    let (input, gc) = parse_bool(input)?;
    let (input, inorder) = parse_bool(input)?;
    let (input, safe_mapper) = parse_bool(input)?;
    let (input, safe_runtime) = parse_bool(input)?;
    let (input, safe_ctrlrepl) = parse_bool(input)?;
    let (input, part_checks) = parse_bool(input)?;
    let (input, bounds_checks) = parse_bool(input)?;
    let (input, resilient) = parse_bool(input)?;
    Ok((
        input,
        Record::RuntimeConfig {
            debug,
            spy,
            gc,
            inorder,
            safe_mapper,
            safe_runtime,
            safe_ctrlrepl,
            part_checks,
            bounds_checks,
            resilient,
        },
    ))
}
fn parse_machine_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, nodeid) = le_u32(input)?;
    let (input, num_nodes) = le_u32(input)?;
    let (input, hostname) = parse_string(input)?;
    let (input, host_id) = le_u64(input)?;
    let (input, process_id) = le_u32(input)?;
    let node_id = NodeID(u64::from(nodeid));
    Ok((
        input,
        Record::MachineDesc {
            node_id,
            num_nodes,
            hostname,
            host_id,
            process_id,
        },
    ))
}
fn parse_zero_time(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, zero_time) = le_i64(input)?;
    Ok((input, Record::ZeroTime { zero_time }))
}
fn parse_provenance(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, pid) = parse_provenance_id(input)?;
    let (input, provenance) = parse_string(input)?;
    Ok((input, Record::Provenance { pid, provenance }))
}
fn parse_proc_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, proc_id) = parse_proc_id(input)?;
    let (input, kind) = le_i32(input)?;
    let (input, cuda_device_uuid) = parse_cuda_device_uuid(input)?;
    Ok((
        input,
        Record::ProcDesc {
            proc_id,
            kind,
            cuda_device_uuid,
        },
    ))
}
fn parse_mem_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, mem_id) = parse_mem_id(input)?;
    let (input, kind) = le_i32(input)?;
    let (input, capacity) = le_u64(input)?;
    Ok((
        input,
        Record::MemDesc {
            mem_id,
            kind,
            capacity,
        },
    ))
}
fn parse_calibration_err(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, calibration_err) = le_i64(input)?;
    Ok((input, Record::CalibrationErr { calibration_err }))
}
fn parse_mem_proc_affinity_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, proc_id) = parse_proc_id(input)?;
    let (input, mem_id) = parse_mem_id(input)?;
    let (input, bandwidth) = le_u32(input)?;
    let (input, latency) = le_u32(input)?;
    Ok((
        input,
        Record::ProcMDesc {
            proc_id,
            mem_id,
            bandwidth,
            latency,
        },
    ))
}
fn parse_index_space_point_desc(input: &[u8], max_dim: i32) -> IResult<&[u8], Record> {
    let (input, ispace_id) = parse_ispace_id(input)?;
    let (input, dim) = le_u32(input)?;
    let (input, rem) = parse_point(input, max_dim)?;
    Ok((
        input,
        Record::IndexSpacePointDesc {
            ispace_id,
            dim,
            rem,
        },
    ))
}
fn parse_index_space_rect_desc(input: &[u8], max_dim: i32) -> IResult<&[u8], Record> {
    let (input, ispace_id) = parse_ispace_id(input)?;
    let (input, dim) = le_u32(input)?;
    let (input, rem) = parse_array(input, max_dim)?;
    Ok((
        input,
        Record::IndexSpaceRectDesc {
            ispace_id,
            dim,
            rem,
        },
    ))
}
fn parse_index_space_empty_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, ispace_id) = parse_ispace_id(input)?;
    Ok((input, Record::IndexSpaceEmptyDesc { ispace_id }))
}
fn parse_field_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, fspace_id) = parse_fspace_id(input)?;
    let (input, field_id) = parse_field_id(input)?;
    let (input, size) = le_u64(input)?;
    let (input, name) = parse_string(input)?;
    Ok((
        input,
        Record::FieldDesc {
            fspace_id,
            field_id,
            size,
            name,
        },
    ))
}
fn parse_field_space_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, fspace_id) = parse_fspace_id(input)?;
    let (input, name) = parse_string(input)?;
    Ok((input, Record::FieldSpaceDesc { fspace_id, name }))
}
fn parse_part_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, unique_id) = parse_ipart_id(input)?;
    let (input, name) = parse_string(input)?;
    Ok((input, Record::PartDesc { unique_id, name }))
}
fn parse_index_space_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, ispace_id) = parse_ispace_id(input)?;
    let (input, name) = parse_string(input)?;
    Ok((input, Record::IndexSpaceDesc { ispace_id, name }))
}
fn parse_index_subspace_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, parent_id) = parse_ipart_id(input)?;
    let (input, ispace_id) = parse_ispace_id(input)?;
    Ok((
        input,
        Record::IndexSubSpaceDesc {
            parent_id,
            ispace_id,
        },
    ))
}
fn parse_index_partition_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, parent_id) = parse_ispace_id(input)?;
    let (input, unique_id) = parse_ipart_id(input)?;
    let (input, disjoint) = parse_bool(input)?;
    let (input, point0) = le_u64(input)?;
    Ok((
        input,
        Record::IndexPartitionDesc {
            parent_id,
            unique_id,
            disjoint,
            point0,
        },
    ))
}
fn parse_index_space_size_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, ispace_id) = parse_ispace_id(input)?;
    let (input, dense_size) = le_u64(input)?;
    let (input, sparse_size) = le_u64(input)?;
    let (input, is_sparse) = parse_bool(input)?;
    Ok((
        input,
        Record::IndexSpaceSizeDesc {
            ispace_id,
            dense_size,
            sparse_size,
            is_sparse,
        },
    ))
}
fn parse_logical_region_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, ispace_id) = parse_ispace_id(input)?;
    let (input, fspace_id) = parse_fspace_id(input)?;
    let (input, tree_id) = parse_tree_id(input)?;
    let (input, name) = parse_string(input)?;
    Ok((
        input,
        Record::LogicalRegionDesc {
            ispace_id,
            fspace_id,
            tree_id,
            name,
        },
    ))
}
fn parse_physical_inst_region_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, fevent) = parse_event_id(input)?;
    let (input, ispace_id) = parse_ispace_id(input)?;
    let (input, fspace_id) = parse_fspace_id(input)?;
    let (input, tree_id) = parse_tree_id(input)?;
    Ok((
        input,
        Record::PhysicalInstRegionDesc {
            fevent,
            ispace_id,
            fspace_id,
            tree_id,
        },
    ))
}
fn parse_physical_inst_layout_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, fevent) = parse_event_id(input)?;
    let (input, field_id) = parse_field_id(input)?;
    let (input, fspace_id) = parse_fspace_id(input)?;
    let (input, has_align) = parse_bool(input)?;
    let (input, eqk) = le_u32(input)?;
    let (input, align_desc) = le_u32(input)?;
    Ok((
        input,
        Record::PhysicalInstLayoutDesc {
            fevent,
            field_id,
            fspace_id,
            has_align,
            eqk,
            align_desc,
        },
    ))
}
fn parse_physical_inst_layout_dim_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, fevent) = parse_event_id(input)?;
    let (input, dim) = le_u32(input)?;
    let (input, dim_kind) = le_u32(input)?;
    Ok((
        input,
        Record::PhysicalInstDimOrderDesc {
            fevent,
            dim,
            dim_kind,
        },
    ))
}
fn parse_physical_inst_spaces(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, fevent) = parse_event_id(input)?;
    let (input, union_space) = parse_option_ispace_id(input)?;
    let (input, piece_space) = parse_option_ispace_id(input)?;
    Ok((
        input,
        Record::PhysicalInstanceSpaces {
            fevent,
            union_space,
            piece_space,
        },
    ))
}
fn parse_physical_inst_usage(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, inst_event) = parse_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, op_id) = parse_op_id(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_option_timestamp(input)?;
    let (input, index_expr) = parse_option_ispace_id(input)?;
    let (input, field) = parse_field_id(input)?;
    let (input, index) = le_i32(input)?;
    let (input, privilege) = parse_privilege_mode(input)?;
    Ok((
        input,
        Record::PhysicalInstanceUsage {
            inst_event,
            fevent,
            op_id,
            start,
            stop,
            index_expr,
            field,
            privilege,
            index: index.try_into().ok(),
        },
    ))
}
fn parse_task_kind(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, task_id) = parse_task_id(input)?;
    let (input, name) = parse_string(input)?;
    let (input, overwrite) = parse_bool(input)?;
    Ok((
        input,
        Record::TaskKind {
            task_id,
            name,
            overwrite,
        },
    ))
}
fn parse_task_variant(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, task_id) = parse_task_id(input)?;
    let (input, variant_id) = parse_variant_id(input)?;
    let (input, name) = parse_string(input)?;
    Ok((
        input,
        Record::TaskVariant {
            task_id,
            variant_id,
            name,
        },
    ))
}
fn parse_operation(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, parent_id) = parse_option_op_id(input)?;
    let (input, kind) = le_u32(input)?;
    let (input, provenance) = parse_option_provenance_id(input)?;
    Ok((
        input,
        Record::OperationInstance {
            op_id,
            parent_id,
            kind,
            provenance,
        },
    ))
}
fn parse_multi_task(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, task_id) = parse_task_id(input)?;
    Ok((input, Record::MultiTask { op_id, task_id }))
}
fn parse_slice_owner(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, parent_id) = le_u64(input)?;
    let (input, op_id) = parse_op_id(input)?;
    Ok((input, Record::SliceOwner { parent_id, op_id }))
}
fn parse_task_wait_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, task_id) = parse_task_id(input)?;
    let (input, variant_id) = parse_variant_id(input)?;
    let (input, wait_start) = parse_timestamp(input)?;
    let (input, wait_ready) = parse_timestamp(input)?;
    let (input, wait_end) = parse_timestamp(input)?;
    let (input, wait_event) = parse_event_id(input)?;
    Ok((
        input,
        Record::TaskWaitInfo {
            op_id,
            task_id,
            variant_id,
            wait_start,
            wait_ready,
            wait_end,
            wait_event,
        },
    ))
}
fn parse_meta_wait_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, lg_id) = parse_variant_id(input)?;
    let (input, wait_start) = parse_timestamp(input)?;
    let (input, wait_ready) = parse_timestamp(input)?;
    let (input, wait_end) = parse_timestamp(input)?;
    let (input, wait_event) = parse_event_id(input)?;
    Ok((
        input,
        Record::MetaWaitInfo {
            op_id,
            lg_id,
            wait_start,
            wait_ready,
            wait_end,
            wait_event,
        },
    ))
}
fn parse_task_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, task_id) = parse_task_id(input)?;
    let (input, variant_id) = parse_variant_id(input)?;
    let (input, proc_id) = parse_proc_id(input)?;
    let (input, create) = parse_timestamp(input)?;
    let (input, ready) = parse_timestamp(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    let (input, creator) = parse_option_event_id(input)?;
    let (input, critical) = parse_option_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    Ok((
        input,
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
            critical,
            fevent,
        },
    ))
}
fn parse_implicit_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, task_id) = parse_task_id(input)?;
    let (input, variant_id) = parse_variant_id(input)?;
    let (input, proc_id) = parse_proc_id(input)?;
    let (input, create) = parse_timestamp(input)?;
    let (input, ready) = parse_timestamp(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    let (input, creator) = parse_option_event_id(input)?;
    let (input, critical) = parse_option_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    Ok((
        input,
        Record::ImplicitTaskInfo {
            op_id,
            task_id,
            variant_id,
            proc_id,
            create,
            ready,
            start,
            stop,
            creator,
            critical,
            fevent,
        },
    ))
}
fn parse_gpu_task_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, task_id) = parse_task_id(input)?;
    let (input, variant_id) = parse_variant_id(input)?;
    let (input, proc_id) = parse_proc_id(input)?;
    let (input, create) = parse_timestamp(input)?;
    let (input, ready) = parse_timestamp(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    let (input, gpu_start) = parse_timestamp(input)?;
    let (input, gpu_stop) = parse_timestamp(input)?;
    let (input, creator) = parse_option_event_id(input)?;
    let (input, critical) = parse_option_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    Ok((
        input,
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
            critical,
            fevent,
        },
    ))
}
fn parse_meta_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, lg_id) = parse_variant_id(input)?;
    let (input, proc_id) = parse_proc_id(input)?;
    let (input, create) = parse_timestamp(input)?;
    let (input, ready) = parse_timestamp(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    let (input, creator) = parse_option_event_id(input)?;
    let (input, critical) = parse_option_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    Ok((
        input,
        Record::MetaInfo {
            op_id,
            lg_id,
            proc_id,
            create,
            ready,
            start,
            stop,
            creator,
            critical,
            fevent,
        },
    ))
}
fn parse_message_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, lg_id) = parse_variant_id(input)?;
    let (input, proc_id) = parse_proc_id(input)?;
    let (input, spawn) = parse_timestamp(input)?;
    let (input, create) = parse_timestamp(input)?;
    let (input, ready) = parse_timestamp(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    let (input, creator) = parse_option_event_id(input)?;
    let (input, critical) = parse_option_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    Ok((
        input,
        Record::MessageInfo {
            op_id,
            lg_id,
            proc_id,
            spawn,
            create,
            ready,
            start,
            stop,
            creator,
            critical,
            fevent,
        },
    ))
}
fn parse_copy_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, size) = le_u64(input)?;
    let (input, create) = parse_timestamp(input)?;
    let (input, ready) = parse_timestamp(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    let (input, creator) = parse_option_event_id(input)?;
    let (input, critical) = parse_option_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, collective) = le_u32(input)?;
    let (input, redop) = parse_option_redop_id(input)?;
    let (input, copy_expr) = parse_option_ispace_id(input)?;
    Ok((
        input,
        Record::CopyInfo {
            op_id,
            size,
            create,
            ready,
            start,
            stop,
            creator,
            critical,
            fevent,
            collective,
            redop,
            copy_expr,
        },
    ))
}
fn parse_copy_inst_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, src) = parse_option_mem_id(input)?;
    let (input, dst) = parse_option_mem_id(input)?;
    let (input, src_fid) = parse_field_id(input)?;
    let (input, dst_fid) = parse_field_id(input)?;
    let (input, src_inst) = parse_option_event_id(input)?;
    let (input, dst_inst) = parse_option_event_id(input)?;
    let (input, src_expr) = parse_option_ispace_id(input)?;
    let (input, dst_expr) = parse_option_ispace_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, num_hops) = le_u32(input)?;
    let (input, indirect) = parse_bool(input)?;
    Ok((
        input,
        Record::CopyInstInfo {
            src,
            dst,
            src_fid,
            dst_fid,
            src_inst,
            dst_inst,
            src_expr,
            dst_expr,
            fevent,
            num_hops,
            indirect,
        },
    ))
}
fn parse_fill_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, size) = le_u64(input)?;
    let (input, create) = parse_timestamp(input)?;
    let (input, ready) = parse_timestamp(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    let (input, creator) = parse_option_event_id(input)?;
    let (input, critical) = parse_option_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, collective) = le_u32(input)?;
    let (input, fill_expr) = parse_option_ispace_id(input)?;
    Ok((
        input,
        Record::FillInfo {
            op_id,
            size,
            create,
            ready,
            start,
            stop,
            creator,
            critical,
            fevent,
            collective,
            fill_expr,
        },
    ))
}
fn parse_fill_inst_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, dst) = parse_mem_id(input)?;
    let (input, fid) = parse_field_id(input)?;
    let (input, dst_inst) = parse_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    Ok((
        input,
        Record::FillInstInfo {
            dst,
            fid,
            dst_inst,
            fevent,
        },
    ))
}
fn parse_inst_timeline(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, fevent) = parse_event_id(input)?;
    let (input, inst_id) = parse_inst_id(input)?;
    let (input, mem_id) = parse_mem_id(input)?;
    let (input, size) = le_u64(input)?;
    let (input, op_id) = parse_op_id(input)?;
    let (input, create) = parse_timestamp(input)?;
    let (input, ready) = parse_timestamp(input)?;
    let (input, destroy) = parse_timestamp(input)?;
    let (input, creator) = parse_event_id(input)?;
    Ok((
        input,
        Record::InstTimelineInfo {
            fevent,
            inst_id,
            mem_id,
            size,
            op_id,
            create,
            ready,
            destroy,
            creator,
        },
    ))
}
fn parse_partition_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, part_op) = le_i32(input)?;
    let (input, create) = parse_timestamp(input)?;
    let (input, ready) = parse_timestamp(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    let (input, creator) = parse_option_event_id(input)?;
    let (input, critical) = parse_option_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    Ok((
        input,
        Record::PartitionInfo {
            op_id,
            part_op,
            create,
            ready,
            start,
            stop,
            creator,
            critical,
            fevent,
        },
    ))
}
fn parse_part_inst_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, src) = parse_mem_id(input)?;
    let (input, fid) = parse_field_id(input)?;
    let (input, src_inst) = parse_event_id(input)?;
    let (input, src_expr) = parse_option_ispace_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    Ok((
        input,
        Record::PartInstInfo {
            src,
            fid,
            src_inst,
            src_expr,
            fevent,
        },
    ))
}
fn parse_mapper_name(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, mapper_id) = parse_mapper_id(input)?;
    let (input, mapper_proc) = parse_proc_id(input)?;
    let (input, name) = parse_string(input)?;
    Ok((
        input,
        Record::MapperName {
            mapper_id,
            mapper_proc,
            name,
        },
    ))
}
fn parse_mapper_call_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, mapper_id) = parse_mapper_id(input)?;
    let (input, mapper_proc) = parse_proc_id(input)?;
    let (input, kind) = parse_mapper_call_kind_id(input)?;
    let (input, op_id) = parse_op_id(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    let (input, proc_id) = parse_proc_id(input)?;
    let (input, fevent) = parse_option_event_id(input)?;
    Ok((
        input,
        Record::MapperCallInfo {
            mapper_id,
            mapper_proc,
            kind,
            op_id,
            start,
            stop,
            proc_id,
            fevent,
        },
    ))
}
fn parse_runtime_call_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, kind) = parse_runtime_call_kind_id(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    let (input, proc_id) = parse_proc_id(input)?;
    let (input, fevent) = parse_option_event_id(input)?;
    Ok((
        input,
        Record::RuntimeCallInfo {
            kind,
            start,
            stop,
            proc_id,
            fevent,
        },
    ))
}
fn parse_application_call_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, provenance) = parse_provenance_id(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    let (input, proc_id) = parse_proc_id(input)?;
    let (input, fevent) = parse_option_event_id(input)?;
    Ok((
        input,
        Record::ApplicationCallInfo {
            provenance,
            start,
            stop,
            proc_id,
            fevent,
        },
    ))
}
fn parse_async_effect_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, provenance) = parse_provenance_id(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    let (input, proc_id) = parse_proc_id(input)?;
    let (input, creator) = parse_event_id(input)?;
    let (input, fevent) = parse_option_event_id(input)?;
    Ok((
        input,
        Record::AsyncEffectInfo {
            provenance,
            start,
            stop,
            proc_id,
            creator,
            fevent,
        },
    ))
}
fn parse_proftask_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, proc_id) = parse_proc_id(input)?;
    let (input, op_id) = parse_op_id(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    let (input, creator) = parse_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, completion) = parse_bool(input)?;
    Ok((
        input,
        Record::ProfTaskInfo {
            proc_id,
            op_id,
            start,
            stop,
            creator,
            fevent,
            completion,
        },
    ))
}
fn parse_backtrace_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, backtrace_id) = parse_backtrace_id(input)?;
    let (input, backtrace) = parse_string(input)?;
    Ok((
        input,
        Record::BacktraceDesc {
            backtrace_id,
            backtrace,
        },
    ))
}
fn parse_event_wait_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, proc_id) = parse_proc_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, event) = parse_event_id(input)?;
    let (input, provenance) = parse_option_provenance_id(input)?;
    let (input, backtrace_id) = parse_backtrace_id(input)?;
    Ok((
        input,
        Record::EventWaitInfo {
            proc_id,
            fevent,
            event,
            provenance,
            backtrace_id,
        },
    ))
}
fn parse_event_merger_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, result) = parse_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, performed) = parse_timestamp(input)?;
    let (input, pre0) = parse_option_event_id(input)?;
    let (input, pre1) = parse_option_event_id(input)?;
    let (input, pre2) = parse_option_event_id(input)?;
    let (input, pre3) = parse_option_event_id(input)?;
    Ok((
        input,
        Record::EventMergerInfo {
            result,
            fevent,
            performed,
            pre0,
            pre1,
            pre2,
            pre3,
        },
    ))
}
fn parse_event_trigger_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, result) = parse_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, precondition) = parse_option_event_id(input)?;
    let (input, performed) = parse_timestamp(input)?;
    Ok((
        input,
        Record::EventTriggerInfo {
            result,
            fevent,
            precondition,
            performed,
        },
    ))
}
fn parse_event_poison_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, result) = parse_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, performed) = parse_timestamp(input)?;
    Ok((
        input,
        Record::EventPoisonInfo {
            result,
            fevent,
            performed,
        },
    ))
}
fn parse_external_event_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, external) = parse_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, performed) = parse_timestamp(input)?;
    let (input, triggered) = parse_timestamp(input)?;
    let (input, provenance) = parse_provenance_id(input)?;
    Ok((
        input,
        Record::ExternalEventInfo {
            external,
            fevent,
            performed,
            triggered,
            provenance,
        },
    ))
}
fn parse_barrier_arrival_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, result) = parse_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, precondition) = parse_option_event_id(input)?;
    let (input, performed) = parse_timestamp(input)?;
    Ok((
        input,
        Record::BarrierArrivalInfo {
            result,
            fevent,
            precondition,
            performed,
        },
    ))
}
fn parse_reservation_acquire_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, result) = parse_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, precondition) = parse_option_event_id(input)?;
    let (input, performed) = parse_timestamp(input)?;
    let (input, reservation) = le_u64(input)?;
    Ok((
        input,
        Record::ReservationAcquireInfo {
            result,
            fevent,
            precondition,
            performed,
            reservation,
        },
    ))
}
fn parse_instance_ready_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, result) = parse_option_event_id(input)?;
    let (input, precondition) = parse_option_event_id(input)?;
    let (input, unique) = parse_event_id(input)?;
    let (input, performed) = parse_timestamp(input)?;
    Ok((
        input,
        Record::InstanceReadyInfo {
            result,
            precondition,
            unique,
            performed,
        },
    ))
}
fn parse_instance_redistrict_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, result) = parse_option_event_id(input)?;
    let (input, precondition) = parse_option_event_id(input)?;
    let (input, previous) = parse_event_id(input)?;
    let (input, next) = parse_event_id(input)?;
    let (input, performed) = parse_timestamp(input)?;
    Ok((
        input,
        Record::InstanceRedistrictInfo {
            result,
            precondition,
            previous,
            next,
            performed,
        },
    ))
}
fn parse_make_valid_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, result) = parse_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, space) = parse_option_ispace_id(input)?;
    let (input, created) = parse_timestamp(input)?;
    let (input, triggered) = parse_timestamp(input)?;
    Ok((
        input,
        Record::MakeValidInfo {
            result,
            fevent,
            space,
            created,
            triggered,
        },
    ))
}
fn parse_fetch_metadata_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, result) = parse_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, inst_event) = parse_event_id(input)?;
    let (input, created) = parse_timestamp(input)?;
    let (input, triggered) = parse_timestamp(input)?;
    Ok((
        input,
        Record::FetchMetadataInfo {
            result,
            fevent,
            inst_event,
            created,
            triggered,
        },
    ))
}
fn parse_completion_queue_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, result) = parse_event_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, performed) = parse_timestamp(input)?;
    let (input, pre0) = parse_option_event_id(input)?;
    let (input, pre1) = parse_option_event_id(input)?;
    let (input, pre2) = parse_option_event_id(input)?;
    let (input, pre3) = parse_option_event_id(input)?;
    Ok((
        input,
        Record::CompletionQueueInfo {
            result,
            fevent,
            performed,
            pre0,
            pre1,
            pre2,
            pre3,
        },
    ))
}
fn parse_spawn_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, fevent) = parse_event_id(input)?;
    let (input, spawn) = parse_timestamp(input)?;
    Ok((input, Record::SpawnInfo { fevent, spawn }))
}

fn filter_critical_path(record: &Record, no_critical_paths: bool) -> bool {
    match record {
        Record::EventMergerInfo { .. }
        | Record::EventTriggerInfo { .. }
        | Record::EventPoisonInfo { .. }
        | Record::ExternalEventInfo { .. }
        | Record::BarrierArrivalInfo { .. }
        | Record::ReservationAcquireInfo { .. }
        | Record::CompletionQueueInfo { .. }
        | Record::InstanceReadyInfo { .. }
        | Record::InstanceRedistrictInfo { .. }
        | Record::MakeValidInfo { .. }
        | Record::FetchMetadataInfo { .. } => !no_critical_paths,
        _ => true,
    }
}

fn filter_nodes(record: &Record, visible_nodes: &[NodeID], node_id: Option<NodeID>) -> bool {
    let Some(node_id) = node_id else {
        return true;
    };
    if State::is_on_visible_nodes(visible_nodes, node_id) {
        return true;
    }

    match record {
        Record::MapperCallDesc { .. }
        | Record::RuntimeCallDesc { .. }
        | Record::MetaDesc { .. }
        | Record::OpDesc { .. }
        | Record::MaxDimDesc { .. }
        | Record::RuntimeConfig { .. }
        | Record::MachineDesc { .. }
        | Record::ZeroTime { .. }
        | Record::ProcDesc { .. }
        | Record::MemDesc { .. }
        | Record::ProcMDesc { .. }
        | Record::IndexSpaceDesc { .. }
        | Record::IndexSpaceSizeDesc { .. }
        | Record::IndexSpacePointDesc { .. }
        | Record::IndexSpaceRectDesc { .. }
        | Record::IndexSpaceEmptyDesc { .. } => true,
        Record::TaskInfo { proc_id, .. } => {
            State::is_on_visible_nodes(visible_nodes, proc_id.node_id())
        }
        Record::ImplicitTaskInfo { proc_id, .. } => {
            State::is_on_visible_nodes(visible_nodes, proc_id.node_id())
        }
        Record::GPUTaskInfo { proc_id, .. } => {
            State::is_on_visible_nodes(visible_nodes, proc_id.node_id())
        }
        Record::MetaInfo { proc_id, .. } => {
            State::is_on_visible_nodes(visible_nodes, proc_id.node_id())
        }
        Record::MessageInfo { proc_id, .. } => {
            State::is_on_visible_nodes(visible_nodes, proc_id.node_id())
        }
        Record::CopyInfo { .. } => true,
        Record::CopyInstInfo { src, dst, .. } => {
            src.is_none_or(|m| State::is_on_visible_nodes(visible_nodes, m.node_id()))
                || dst.is_none_or(|m| State::is_on_visible_nodes(visible_nodes, m.node_id()))
        }
        Record::FillInfo { .. } => true,
        Record::FillInstInfo { dst, .. } => {
            State::is_on_visible_nodes(visible_nodes, dst.node_id())
        }
        Record::InstTimelineInfo { mem_id, .. } => {
            State::is_on_visible_nodes(visible_nodes, mem_id.node_id())
        }
        Record::PartitionInfo { .. } => true,
        _ => false,
    }
}

fn check_version(prof_version: u32, legion_version: String) {
    let expected_version: u32 = include_str!("../../../runtime/legion/tools/profiler_version.h")
        .trim()
        .parse()
        .unwrap();

    assert_eq!(
        prof_version, expected_version,
        "Legion Prof was built against an incompatible Legion version. (Expected version {}, got version {}.) Please rebuild Legion Prof with the same version of Legion {} used by the application to generate the profile logs. You can also refer to this page to identify a Legion commit to use with each Legion Prof version https://gitlab.com/StanfordLegion/legion/-/commits/master/runtime/legion/tools/profiler_version.h?ref_type=heads",
        expected_version, prof_version, legion_version
    );
}

fn parse_record<'a>(
    input: &'a [u8],
    parsers: &HashMap<u32, fn(&[u8], i32) -> IResult<&[u8], Record>>,
    max_dim: i32,
) -> IResult<&'a [u8], Record> {
    let (input, id) = le_u32(input)?;
    let parser = &parsers[&id];
    parser(input, max_dim)
}

// nom parsers don't work natively with BufReader so we adapt the interface
// here. We fill the buffer, attempt to read, and then consume bytes if
// successful.
//
// There are two failure cases. If the parse fails (Error or Failure) we just
// wrap the error and return it. If the parse is incomplete, we increase the
// min buffer size (and the buffer capacity if necessary) and then try
// again. Eventually we'll have enough data in memory to parse the record. Note
// that records are variable-size (because they contain user-specified strings)
// so we need to be prepared to parse records of any size here.
fn try_parse<T>(
    reader: &mut BufReader<impl Read, MinBuffered>,
    parser: impl Fn(&[u8]) -> IResult<&[u8], T>,
) -> io::Result<Result<T, String>> {
    loop {
        let input = reader.fill_buf()?;
        let result = parser(input);
        match result {
            Ok((remaining, value)) => {
                let consumed_bytes = input.len() - remaining.len();
                reader.consume(consumed_bytes);
                return Ok(Ok(value));
            }
            Err(nom::Err::Incomplete(_)) => {
                reader.policy_mut().0 *= 2;
                if reader.policy().0 >= reader.capacity() {
                    reader.reserve(reader.capacity());
                }
            }
            Err(err @ nom::Err::Error(_) | err @ nom::Err::Failure(_)) => {
                return Ok(Err(err.to_string()));
            }
        }
    }
}

fn must_parse<T>(
    reader: &mut BufReader<impl Read, MinBuffered>,
    parser: impl Fn(&[u8]) -> IResult<&[u8], T>,
) -> io::Result<T> {
    try_parse(reader, parser)?.map_err(io::Error::other)
}

fn maybe_parse<T>(
    reader: &mut BufReader<impl Read, MinBuffered>,
    parser: impl Fn(&[u8]) -> IResult<&[u8], T>,
) -> io::Result<Option<T>> {
    Ok(try_parse(reader, parser)?.ok())
}

fn parse(
    reader: &mut BufReader<impl Read, MinBuffered>,
    path: &Path,
    visible_nodes: &[NodeID],
    filter_input: bool,
    no_critical_paths: bool,
) -> io::Result<Vec<Record>> {
    let Ok((prof_version, legion_version)) = must_parse(reader, parse_filetype) else {
        panic!(
            "Unable to parse header of {}. This could be because it is not a valid Legion Prof logfile or because it was generated by a version of Legion prior to 911657c1140373fceb7f0359b377dc66c1aafec9 (September 4, 2025). Unfortunately Legion Prof cannot tell which case is occurring or even what Legion commit you should use for trying to parse this log file.",
            path.display(),
        );
    };
    check_version(prof_version, legion_version);

    let mut ids = HashMap::new();
    loop {
        let record_format = must_parse(reader, parse_record_format)?;
        ids.insert(record_format.name, record_format.id);

        // Keep going until we hit an empty line by itself
        if maybe_parse(reader, newline)?.is_some() {
            break;
        }
    }

    let mut parsers = HashMap::<u32, fn(&[u8], i32) -> IResult<&[u8], Record>>::new();
    let mut insert = |name, parser| {
        if let Some(id) = ids.get(name) {
            parsers.insert(*id, parser);
        }
    };
    insert("MapperName", parse_mapper_name);
    insert("MapperCallDesc", parse_mapper_call_desc);
    insert("RuntimeCallDesc", parse_runtime_call_desc);
    insert("MetaDesc", parse_meta_desc);
    insert("OpDesc", parse_op_desc);
    insert("MaxDimDesc", parse_max_dim_desc);
    insert("RuntimeConfig", parse_runtime_config);
    insert("MachineDesc", parse_machine_desc);
    insert("ZeroTime", parse_zero_time);
    insert("Provenance", parse_provenance);
    insert("ProcDesc", parse_proc_desc);
    insert("MemDesc", parse_mem_desc);
    insert("ProcMDesc", parse_mem_proc_affinity_desc);
    insert("CalibrationErr", parse_calibration_err);
    insert("IndexSpacePointDesc", parse_index_space_point_desc);
    insert("IndexSpaceRectDesc", parse_index_space_rect_desc);
    insert("IndexSpaceEmptyDesc", parse_index_space_empty_desc);
    insert("FieldDesc", parse_field_desc);
    insert("FieldSpaceDesc", parse_field_space_desc);
    insert("PartDesc", parse_part_desc);
    insert("IndexSpaceDesc", parse_index_space_desc);
    insert("IndexSubSpaceDesc", parse_index_subspace_desc);
    insert("IndexPartitionDesc", parse_index_partition_desc);
    insert("IndexSpaceSizeDesc", parse_index_space_size_desc);
    insert("LogicalRegionDesc", parse_logical_region_desc);
    insert("PhysicalInstRegionDesc", parse_physical_inst_region_desc);
    insert("PhysicalInstLayoutDesc", parse_physical_inst_layout_desc);
    insert(
        "PhysicalInstDimOrderDesc",
        parse_physical_inst_layout_dim_desc,
    );
    insert("PhysicalInstanceSpaces", parse_physical_inst_spaces);
    insert("PhysicalInstanceUsage", parse_physical_inst_usage);
    insert("TaskKind", parse_task_kind);
    insert("TaskVariant", parse_task_variant);
    insert("OperationInstance", parse_operation);
    insert("MultiTask", parse_multi_task);
    insert("SliceOwner", parse_slice_owner);
    insert("TaskWaitInfo", parse_task_wait_info);
    insert("MetaWaitInfo", parse_meta_wait_info);
    insert("TaskInfo", parse_task_info);
    insert("GPUTaskInfo", parse_gpu_task_info);
    insert("ImplicitTaskInfo", parse_implicit_info);
    insert("MetaInfo", parse_meta_info);
    insert("MessageInfo", parse_message_info);
    insert("CopyInfo", parse_copy_info);
    insert("CopyInstInfo", parse_copy_inst_info);
    insert("FillInfo", parse_fill_info);
    insert("FillInstInfo", parse_fill_inst_info);
    insert("InstTimelineInfo", parse_inst_timeline);
    insert("PartitionInfo", parse_partition_info);
    insert("PartitionInstInfo", parse_part_inst_info);
    insert("MapperCallInfo", parse_mapper_call_info);
    insert("RuntimeCallInfo", parse_runtime_call_info);
    insert("ApplicationCallInfo", parse_application_call_info);
    insert("AsyncEffectInfo", parse_async_effect_info);
    insert("ProfTaskInfo", parse_proftask_info);
    insert("BacktraceDesc", parse_backtrace_desc);
    insert("EventWaitInfo", parse_event_wait_info);
    insert("EventMergerInfo", parse_event_merger_info);
    insert("EventTriggerInfo", parse_event_trigger_info);
    insert("EventPoisonInfo", parse_event_poison_info);
    insert("ExternalEventInfo", parse_external_event_info);
    insert("BarrierArrivalInfo", parse_barrier_arrival_info);
    insert("ReservationAcquireInfo", parse_reservation_acquire_info);
    insert("InstanceReadyInfo", parse_instance_ready_info);
    insert("InstanceRedistrictInfo", parse_instance_redistrict_info);
    insert("MakeValidInfo", parse_make_valid_info);
    insert("FetchMetadataInfo", parse_fetch_metadata_info);
    insert("CompletionQueueInfo", parse_completion_queue_info);
    insert("SpawnInfo", parse_spawn_info);

    let mut max_dim = -1;
    let mut node_id: Option<NodeID> = None;
    let mut records = Vec::new();
    loop {
        // If we're empty then we reached EOF and we can stop.
        if reader.fill_buf()?.is_empty() {
            break;
        }

        let record = must_parse(reader, |input| parse_record(input, &parsers, max_dim))?;

        if let Record::MaxDimDesc { max_dim: d } = record {
            max_dim = d;
        }
        if let Record::MachineDesc { node_id: d, .. } = record {
            node_id = Some(d);
        }
        if filter_critical_path(&record, no_critical_paths)
            && (!filter_input || filter_nodes(&record, visible_nodes, node_id))
        {
            records.push(record);
        }
    }
    assert!(reader.fill_buf()?.is_empty());
    Ok(records)
}

fn make_buf_reader(r: Box<dyn Read>) -> BufReader<Box<dyn Read>, MinBuffered> {
    // We use the MinBuffered policy to ensure that we have enough data to read
    // a complete record out of the buffer. Because records are variable size
    // we also adjust this value at runtime (in the parser) as needed.
    BufReader::with_capacity_ringbuf(64 * 1024, r).set_policy(MinBuffered(1024))
}

fn make_gzip_reader(file: Arc<File>) -> io::Result<BufReader<Box<dyn Read>, MinBuffered>> {
    let mut reader = make_buf_reader(Box::new(GzDecoder::new(file)));
    // Fill the buffer once to make sure we're actually reading a
    // gzip-formatted file. If not we'll give up and look for a different
    // decompression method.
    reader.fill_buf()?;
    Ok(reader)
}

fn make_uncompressed_reader(file: Arc<File>) -> io::Result<BufReader<Box<dyn Read>, MinBuffered>> {
    Ok(make_buf_reader(Box::new(file)))
}

fn make_file_reader(path: impl AsRef<Path>) -> io::Result<BufReader<Box<dyn Read>, MinBuffered>> {
    let mut f = Arc::new(File::open(path)?);
    if let Ok(reader) = make_gzip_reader(f.clone()) {
        Ok(reader)
    } else {
        f.rewind()?;
        make_uncompressed_reader(f)
    }
}

pub fn deserialize<P: AsRef<Path>>(
    path: P,
    visible_nodes: &[NodeID],
    filter_input: bool,
    no_critical_paths: bool,
) -> io::Result<Vec<Record>> {
    let path = path.as_ref();
    let mut reader = make_file_reader(path)?;
    parse(
        &mut reader,
        path,
        visible_nodes,
        filter_input,
        no_critical_paths,
    )
}
