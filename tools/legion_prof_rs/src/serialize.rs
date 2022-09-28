use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::io::Read;
use std::path::Path;

use flate2::read::GzDecoder;

use nom;
use nom::{
    bytes::complete::{tag, take_till, take_while1},
    character::{is_alphanumeric, is_digit},
    combinator::{map, map_opt, map_res, opt},
    multi::{many1, many_m_n, separated_list1},
    number::complete::{le_i32, le_u32, le_u64, le_u8},
    IResult,
};

use crate::state::{
    EventID, FSpaceID, FieldID, IPartID, ISpaceID, InstID, MapperCallKindID, MemID, OpID, ProcID,
    RuntimeCallKindID, TaskID, Timestamp, TreeID, VariantID,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ValueFormat {
    Array,
    Bool,
    DepPartOpKind,
    IDType,
    InstID,
    MappingCallKind,
    MaxDim,
    MemID,
    MemKind,
    MessageKind,
    Point,
    ProcID,
    ProcKind,
    RuntimeCallKind,
    String,
    TaskID,
    Timestamp,
    U32,
    U64,
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

#[derive(Debug, Clone)]
pub struct Array(pub Vec<u64>);

#[derive(Debug, Clone)]
pub struct Point(pub Vec<u64>);

#[rustfmt::skip]
#[derive(Debug, Clone)]
pub enum Record {
    MapperCallDesc { kind: MapperCallKindID, name: String },
    RuntimeCallDesc { kind: RuntimeCallKindID, name: String },
    MetaDesc { kind: VariantID, message: bool, ordered_vc: bool, name: String },
    OpDesc { kind: u32, name: String },
    ProcDesc { proc_id: ProcID, kind: ProcKind },
    MaxDimDesc { max_dim: MaxDim },
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
    LogicalRegionDesc { ispace_id: ISpaceID, fspace_id: u32, tree_id: TreeID, name: String },
    PhysicalInstRegionDesc { op_id: OpID, inst_id: InstID, ispace_id: ISpaceID, fspace_id: u32, tree_id: TreeID },
    PhysicalInstLayoutDesc { op_id: OpID, inst_id: InstID, field_id: FieldID, fspace_id: u32, has_align: bool, eqk: u32, align_desc: u32 },
    PhysicalInstDimOrderDesc { op_id: OpID, inst_id: InstID, dim: u32, dim_kind: u32 },
    TaskKind { task_id: TaskID, name: String, overwrite: bool },
    TaskVariant { task_id: TaskID, variant_id: VariantID, name: String },
    OperationInstance { op_id: OpID, parent_id: OpID, kind: u32, provenance: String },
    MultiTask { op_id: OpID, task_id: TaskID },
    SliceOwner { parent_id: UniqueID, op_id: OpID },
    TaskWaitInfo { op_id: OpID, task_id: TaskID, variant_id: VariantID, wait_start: Timestamp, wait_ready: Timestamp, wait_end: Timestamp },
    MetaWaitInfo { op_id: OpID, lg_id: VariantID, wait_start: Timestamp, wait_ready: Timestamp, wait_end: Timestamp },
    TaskInfo { op_id: OpID, task_id: TaskID, variant_id: VariantID, proc_id: ProcID, create: Timestamp, ready: Timestamp, start: Timestamp, stop: Timestamp },
    GPUTaskInfo { op_id: OpID, task_id: TaskID, variant_id: VariantID, proc_id: ProcID, create: Timestamp, ready: Timestamp, start: Timestamp, stop: Timestamp, gpu_start: Timestamp, gpu_stop: Timestamp },
    MetaInfo { op_id: OpID, lg_id: VariantID, proc_id: ProcID, create: Timestamp, ready: Timestamp, start: Timestamp, stop: Timestamp },
    CopyInfo { op_id: OpID, src: MemID, dst: MemID, size: u64, create: Timestamp, ready: Timestamp, start: Timestamp, stop: Timestamp, fevent: EventID, num_requests: u32 },
    CopyInstInfo { op_id: OpID, src_inst: InstID, dst_inst: InstID, fevent: EventID, num_fields: u32, request_type: u32, num_hops: u32 },
    FillInfo { op_id: OpID, dst: MemID, create: Timestamp, ready: Timestamp, start: Timestamp, stop: Timestamp },
    InstCreateInfo { op_id: OpID, inst_id: InstID, create: Timestamp },
    InstUsageInfo { op_id: OpID, inst_id: InstID, mem_id: MemID, size: u64 },
    InstTimelineInfo { op_id: OpID, inst_id: InstID, create: Timestamp, ready: Timestamp, destroy: Timestamp },
    PartitionInfo { op_id: OpID, part_op: DepPartOpKind, create: Timestamp, ready: Timestamp, start: Timestamp, stop: Timestamp },
    MapperCallInfo { kind: MapperCallKindID, op_id: OpID, start: Timestamp, stop: Timestamp, proc_id: ProcID },
    RuntimeCallInfo { kind: RuntimeCallKindID, start: Timestamp, stop: Timestamp, proc_id: ProcID },
    ProfTaskInfo { proc_id: ProcID, op_id: OpID, start: Timestamp, stop: Timestamp },
}

fn convert_value_format(name: String) -> Option<ValueFormat> {
    match name.as_str() {
        "array" => Some(ValueFormat::Array),
        "bool" => Some(ValueFormat::Bool),
        "DepPartOpKind" => Some(ValueFormat::DepPartOpKind),
        "IDType" => Some(ValueFormat::IDType),
        "InstID" => Some(ValueFormat::InstID),
        "MappingCallKind" => Some(ValueFormat::MappingCallKind),
        "maxdim" => Some(ValueFormat::MaxDim),
        "MemID" => Some(ValueFormat::MemID),
        "MemKind" => Some(ValueFormat::MemKind),
        "MessageKind" => Some(ValueFormat::MessageKind),
        "point" => Some(ValueFormat::Point),
        "ProcID" => Some(ValueFormat::ProcID),
        "ProcKind" => Some(ValueFormat::ProcKind),
        "RuntimeCallKind" => Some(ValueFormat::RuntimeCallKind),
        "string" => Some(ValueFormat::String),
        "TaskID" => Some(ValueFormat::TaskID),
        "timestamp_t" => Some(ValueFormat::Timestamp),
        "unsigned" => Some(ValueFormat::U32),
        "unsigned long long" => Some(ValueFormat::U64),
        "UniqueID" => Some(ValueFormat::UniqueID),
        "VariantID" => Some(ValueFormat::VariantID),
        _ => None,
    }
}

///
/// Text parser utilities
///

fn newline(input: &[u8]) -> IResult<&[u8], ()> {
    let (input, _) = tag("\n")(input)?;
    Ok((input, ()))
}

fn parse_text_i32(input: &[u8]) -> IResult<&[u8], i32> {
    let (input, sign) = opt(tag("-"))(input)?;
    let (input, value) = take_while1(is_digit)(input)?;
    let value: i32 = String::from_utf8(value.to_owned())
        .unwrap()
        .parse()
        .unwrap();
    Ok((input, if sign.is_none() { value } else { -value }))
}

fn parse_text_u32(input: &[u8]) -> IResult<&[u8], u32> {
    let (input, value) = take_while1(is_digit)(input)?;
    let value = String::from_utf8(value.to_owned())
        .unwrap()
        .parse()
        .unwrap();
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
    Ok((input, String::from_utf8(name.to_owned()).unwrap()))
}

fn parse_text_type(input: &[u8]) -> IResult<&[u8], String> {
    let (input, name) = take_while1(is_alphanumeric_space)(input)?;
    Ok((input, String::from_utf8(name.to_owned()).unwrap()))
}

///
/// Text parsers for the log file header
///

fn parse_filetype(input: &[u8]) -> IResult<&[u8], (u32, u32)> {
    let (input, _) = tag("FileType: BinaryLegionProf v: ")(input)?;
    let (input, version_major) = parse_text_u32(input)?;
    let (input, _) = tag(".")(input)?;
    let (input, version_minor) = parse_text_u32(input)?;
    let (input, _) = newline(input)?;
    Ok((input, (version_major, version_minor)))
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
    Ok((
        input,
        FieldFormat {
            name: name,
            value: value,
            size: size,
        },
    ))
}

fn parse_record_format(input: &[u8]) -> IResult<&[u8], RecordFormat> {
    let (input, name) = parse_text_name(input)?;
    let (input, _) = tag(" {id:")(input)?;
    let (input, id) = parse_text_u32(input)?;
    let (input, _) = tag(", ")(input)?;
    let (input, fields) = separated_list1(tag(", "), parse_field_format)(input)?;
    let (input, _) = tag("}")(input)?;
    let (input, _) = newline(input)?;
    Ok((
        input,
        RecordFormat {
            id: id,
            name: name,
            fields: fields,
        },
    ))
}

///
/// Binary parsers for basic types used in records
///

fn parse_array(input: &[u8], max_dim: i32) -> IResult<&[u8], Array> {
    assert!(max_dim > -1);
    let n = (max_dim * 2) as usize;
    let (input, values) = many_m_n(n, n, le_u64)(input)?;
    Ok((input, Array(values)))
}
fn parse_bool(input: &[u8]) -> IResult<&[u8], bool> {
    map(le_u8, |x| x != 0)(input)
}
fn parse_point(input: &[u8], max_dim: i32) -> IResult<&[u8], Point> {
    assert!(max_dim > -1);
    let n = max_dim as usize;
    let (input, values) = many_m_n(n, n, le_u64)(input)?;
    Ok((input, Point(values)))
}
fn parse_string(input: &[u8]) -> IResult<&[u8], String> {
    let (input, value) = map_res(take_till(is_nul), |x: &[u8]| {
        String::from_utf8(x.to_owned())
    })(input)?;
    let (input, terminator) = le_u8(input)?;
    assert!(is_nul(terminator));
    Ok((input, value))
}

///
/// Binary parsers for type aliases
///

fn parse_event_id(input: &[u8]) -> IResult<&[u8], EventID> {
    map(le_u64, |x| EventID(x))(input)
}

fn parse_inst_id(input: &[u8]) -> IResult<&[u8], InstID> {
    map(le_u64, |x| InstID(x))(input)
}
fn parse_ipart_id(input: &[u8]) -> IResult<&[u8], IPartID> {
    map(le_u64, |x| IPartID(x))(input)
}
fn parse_ispace_id(input: &[u8]) -> IResult<&[u8], ISpaceID> {
    map(le_u64, |x| ISpaceID(x))(input)
}
fn parse_fspace_id(input: &[u8]) -> IResult<&[u8], FSpaceID> {
    map(le_u64, |x| FSpaceID(x))(input)
}
fn parse_field_id(input: &[u8]) -> IResult<&[u8], FieldID> {
    map(le_u32, |x| FieldID(x))(input)
}
fn parse_tree_id(input: &[u8]) -> IResult<&[u8], TreeID> {
    map(le_u32, |x| TreeID(x))(input)
}
fn parse_mapper_call_kind_id(input: &[u8]) -> IResult<&[u8], MapperCallKindID> {
    map(le_u32, |x| MapperCallKindID(x))(input)
}
fn parse_mem_id(input: &[u8]) -> IResult<&[u8], MemID> {
    map(le_u64, |x| MemID(x))(input)
}
fn parse_op_id(input: &[u8]) -> IResult<&[u8], OpID> {
    map(le_u64, |x| OpID(x))(input)
}
fn parse_proc_id(input: &[u8]) -> IResult<&[u8], ProcID> {
    map(le_u64, |x| ProcID(x))(input)
}
fn parse_runtime_call_kind_id(input: &[u8]) -> IResult<&[u8], RuntimeCallKindID> {
    map(le_u32, |x| RuntimeCallKindID(x))(input)
}
fn parse_task_id(input: &[u8]) -> IResult<&[u8], TaskID> {
    map(le_u32, |x| TaskID(x))(input)
}
fn parse_timestamp(input: &[u8]) -> IResult<&[u8], Timestamp> {
    map(le_u64, |x| Timestamp(x))(input)
}
fn parse_variant_id(input: &[u8]) -> IResult<&[u8], VariantID> {
    map(le_u32, |x| VariantID(x))(input)
}

///
/// Binary parsers for records
///

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
fn parse_proc_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, proc_id) = parse_proc_id(input)?;
    let (input, kind) = le_i32(input)?;
    Ok((input, Record::ProcDesc { proc_id, kind }))
}
fn parse_max_dim_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, max_dim) = le_i32(input)?;
    Ok((input, Record::MaxDimDesc { max_dim }))
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
    let (input, fspace_id) = le_u32(input)?;
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
    let (input, op_id) = parse_op_id(input)?;
    let (input, inst_id) = parse_inst_id(input)?;
    let (input, ispace_id) = parse_ispace_id(input)?;
    let (input, fspace_id) = le_u32(input)?;
    let (input, tree_id) = parse_tree_id(input)?;
    Ok((
        input,
        Record::PhysicalInstRegionDesc {
            op_id,
            inst_id,
            ispace_id,
            fspace_id,
            tree_id,
        },
    ))
}
fn parse_physical_inst_layout_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, inst_id) = parse_inst_id(input)?;
    let (input, field_id) = parse_field_id(input)?;
    let (input, fspace_id) = le_u32(input)?;
    let (input, has_align) = parse_bool(input)?;
    let (input, eqk) = le_u32(input)?;
    let (input, align_desc) = le_u32(input)?;
    Ok((
        input,
        Record::PhysicalInstLayoutDesc {
            op_id,
            inst_id,
            field_id,
            fspace_id,
            has_align,
            eqk,
            align_desc,
        },
    ))
}
fn parse_physical_inst_layout_dim_desc(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, inst_id) = parse_inst_id(input)?;
    let (input, dim) = le_u32(input)?;
    let (input, dim_kind) = le_u32(input)?;
    Ok((
        input,
        Record::PhysicalInstDimOrderDesc {
            op_id,
            inst_id,
            dim,
            dim_kind,
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
    let (input, parent_id) = parse_op_id(input)?;
    let (input, kind) = le_u32(input)?;
    let (input, provenance) = parse_string(input)?;
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
    Ok((
        input,
        Record::TaskWaitInfo {
            op_id,
            task_id,
            variant_id,
            wait_start,
            wait_ready,
            wait_end,
        },
    ))
}
fn parse_meta_wait_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, lg_id) = parse_variant_id(input)?;
    let (input, wait_start) = parse_timestamp(input)?;
    let (input, wait_ready) = parse_timestamp(input)?;
    let (input, wait_end) = parse_timestamp(input)?;
    Ok((
        input,
        Record::MetaWaitInfo {
            op_id,
            lg_id,
            wait_start,
            wait_ready,
            wait_end,
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
        },
    ))
}
fn parse_copy_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, src) = parse_mem_id(input)?;
    let (input, dst) = parse_mem_id(input)?;
    let (input, size) = le_u64(input)?;
    let (input, create) = parse_timestamp(input)?;
    let (input, ready) = parse_timestamp(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, num_requests) = le_u32(input)?;
    Ok((
        input,
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
        },
    ))
}
fn parse_copy_inst_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, src_inst) = parse_inst_id(input)?;
    let (input, dst_inst) = parse_inst_id(input)?;
    let (input, fevent) = parse_event_id(input)?;
    let (input, num_fields) = le_u32(input)?;
    let (input, request_type) = le_u32(input)?;
    let (input, num_hops) = le_u32(input)?;
    Ok((
        input,
        Record::CopyInstInfo {
            op_id,
            src_inst,
            dst_inst,
            fevent,
            num_fields,
            request_type,
            num_hops,
        },
    ))
}
fn parse_fill_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, dst) = parse_mem_id(input)?;
    let (input, create) = parse_timestamp(input)?;
    let (input, ready) = parse_timestamp(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    Ok((
        input,
        Record::FillInfo {
            op_id,
            dst,
            create,
            ready,
            start,
            stop,
        },
    ))
}
fn parse_inst_create(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, inst_id) = parse_inst_id(input)?;
    let (input, create) = parse_timestamp(input)?;
    Ok((
        input,
        Record::InstCreateInfo {
            op_id,
            inst_id,
            create,
        },
    ))
}
fn parse_inst_usage(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, inst_id) = parse_inst_id(input)?;
    let (input, mem_id) = parse_mem_id(input)?;
    let (input, size) = le_u64(input)?;
    Ok((
        input,
        Record::InstUsageInfo {
            op_id,
            inst_id,
            mem_id,
            size,
        },
    ))
}
fn parse_inst_timeline(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, op_id) = parse_op_id(input)?;
    let (input, inst_id) = parse_inst_id(input)?;
    let (input, create) = parse_timestamp(input)?;
    let (input, ready) = parse_timestamp(input)?;
    let (input, destroy) = parse_timestamp(input)?;
    Ok((
        input,
        Record::InstTimelineInfo {
            op_id,
            inst_id,
            create,
            ready,
            destroy,
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
    Ok((
        input,
        Record::PartitionInfo {
            op_id,
            part_op,
            create,
            ready,
            start,
            stop,
        },
    ))
}
fn parse_mapper_call_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, kind) = parse_mapper_call_kind_id(input)?;
    let (input, op_id) = parse_op_id(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    let (input, proc_id) = parse_proc_id(input)?;
    Ok((
        input,
        Record::MapperCallInfo {
            kind,
            op_id,
            start,
            stop,
            proc_id,
        },
    ))
}
fn parse_runtime_call_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, kind) = parse_runtime_call_kind_id(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    let (input, proc_id) = parse_proc_id(input)?;
    Ok((
        input,
        Record::RuntimeCallInfo {
            kind,
            start,
            stop,
            proc_id,
        },
    ))
}
fn parse_proftask_info(input: &[u8], _max_dim: i32) -> IResult<&[u8], Record> {
    let (input, proc_id) = parse_proc_id(input)?;
    let (input, op_id) = parse_op_id(input)?;
    let (input, start) = parse_timestamp(input)?;
    let (input, stop) = parse_timestamp(input)?;
    Ok((
        input,
        Record::ProfTaskInfo {
            proc_id,
            op_id,
            start,
            stop,
        },
    ))
}

fn parse_record<'a>(
    input: &'a [u8],
    parsers: &BTreeMap<u32, fn(&[u8], i32) -> IResult<&[u8], Record>>,
    max_dim: i32,
) -> IResult<&'a [u8], Record> {
    let (input, id) = le_u32(input)?;
    let parser = &parsers[&id];
    parser(input, max_dim)
}

fn parse(input: &[u8]) -> IResult<&[u8], Vec<Record>> {
    let (input, version) = parse_filetype(input)?;
    assert_eq!(version, (1, 0));
    let (input, record_formats) = many1(parse_record_format)(input)?;
    let mut ids = BTreeMap::new();
    for record_format in record_formats {
        ids.insert(record_format.name, record_format.id);
    }
    let (input, _) = newline(input)?;

    let mut parsers = BTreeMap::<u32, fn(&[u8], i32) -> IResult<&[u8], Record>>::new();
    parsers.insert(ids["MapperCallDesc"], parse_mapper_call_desc);
    parsers.insert(ids["RuntimeCallDesc"], parse_runtime_call_desc);
    parsers.insert(ids["MetaDesc"], parse_meta_desc);
    parsers.insert(ids["OpDesc"], parse_op_desc);
    parsers.insert(ids["ProcDesc"], parse_proc_desc);
    parsers.insert(ids["MaxDimDesc"], parse_max_dim_desc);
    parsers.insert(ids["MemDesc"], parse_mem_desc);
    parsers.insert(ids["ProcMDesc"], parse_mem_proc_affinity_desc);
    parsers.insert(ids["IndexSpacePointDesc"], parse_index_space_point_desc);
    parsers.insert(ids["IndexSpaceRectDesc"], parse_index_space_rect_desc);
    parsers.insert(ids["IndexSpaceEmptyDesc"], parse_index_space_empty_desc);
    parsers.insert(ids["FieldDesc"], parse_field_desc);
    parsers.insert(ids["FieldSpaceDesc"], parse_field_space_desc);
    parsers.insert(ids["PartDesc"], parse_part_desc);
    parsers.insert(ids["IndexSpaceDesc"], parse_index_space_desc);
    parsers.insert(ids["IndexSubSpaceDesc"], parse_index_subspace_desc);
    parsers.insert(ids["IndexPartitionDesc"], parse_index_partition_desc);
    parsers.insert(ids["IndexSpaceSizeDesc"], parse_index_space_size_desc);
    parsers.insert(ids["LogicalRegionDesc"], parse_logical_region_desc);
    parsers.insert(
        ids["PhysicalInstRegionDesc"],
        parse_physical_inst_region_desc,
    );
    parsers.insert(
        ids["PhysicalInstLayoutDesc"],
        parse_physical_inst_layout_desc,
    );
    parsers.insert(
        ids["PhysicalInstDimOrderDesc"],
        parse_physical_inst_layout_dim_desc,
    );
    parsers.insert(ids["TaskKind"], parse_task_kind);
    parsers.insert(ids["TaskVariant"], parse_task_variant);
    parsers.insert(ids["OperationInstance"], parse_operation);
    parsers.insert(ids["MultiTask"], parse_multi_task);
    parsers.insert(ids["SliceOwner"], parse_slice_owner);
    parsers.insert(ids["TaskWaitInfo"], parse_task_wait_info);
    parsers.insert(ids["MetaWaitInfo"], parse_meta_wait_info);
    parsers.insert(ids["TaskInfo"], parse_task_info);
    parsers.insert(ids["GPUTaskInfo"], parse_gpu_task_info);
    parsers.insert(ids["MetaInfo"], parse_meta_info);
    parsers.insert(ids["CopyInfo"], parse_copy_info);
    parsers.insert(ids["CopyInstInfo"], parse_copy_inst_info);
    parsers.insert(ids["FillInfo"], parse_fill_info);
    parsers.insert(ids["InstCreateInfo"], parse_inst_create);
    parsers.insert(ids["InstUsageInfo"], parse_inst_usage);
    parsers.insert(ids["InstTimelineInfo"], parse_inst_timeline);
    parsers.insert(ids["PartitionInfo"], parse_partition_info);
    parsers.insert(ids["MapperCallInfo"], parse_mapper_call_info);
    parsers.insert(ids["RuntimeCallInfo"], parse_runtime_call_info);
    parsers.insert(ids["ProfTaskInfo"], parse_proftask_info);

    let mut input = input;
    let mut max_dim = -1;
    let mut records = Vec::new();
    loop {
        match parse_record(input, &parsers, max_dim) {
            Ok((input_, record)) => {
                match &record {
                    Record::MaxDimDesc { max_dim: d } => {
                        max_dim = *d;
                    }
                    _ => {}
                }
                records.push(record);
                input = input_;
            }
            _ => break,
        }
    }
    Ok((input, records))
}

pub fn deserialize<P: AsRef<Path>>(path: P) -> io::Result<Vec<Record>> {
    let mut gz = GzDecoder::new(File::open(path)?);
    let mut s = Vec::<u8>::new();
    gz.read_to_end(&mut s)?;
    // throw error here if parse failed
    let (rest, records) = parse(&s).unwrap();
    assert_eq!(rest.len(), 0);
    Ok(records)
}
