use std::fmt;
use std::fs::File;
use std::io;
use std::io::Read;
use std::path::Path;

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{digit1, hex_digit1, line_ending, none_of, not_line_ending, space0, u64},
    combinator::{all_consuming, map, opt, peek, value},
    multi::many1,
    sequence::preceded,
    IResult,
};

use serde::{
    de::{Error, SeqAccess, Visitor},
    Deserialize, Deserializer,
};

use crate::serde::ascii::{from_str, HexU64};

#[derive(Debug, Clone)]
pub struct Prefix {
    _node: u64,
    //thread: u64,
    spy: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct AddressSpaceID(u64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct ProcID(HexU64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct MemID(HexU64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct IspaceID(HexU64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct IpartID(HexU64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct ExprID(u64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct FspaceID(u64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct FieldID(u64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct TreeID(u64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct InstID(HexU64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct TaskID(u64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct VariantID(u64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct ProjectionID(u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct ContextID(pub u64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct UniqueID(pub u64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct FutureID(HexU64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct EventID(pub HexU64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct IndirectID(HexU64);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Point(pub Vec<i64>);

impl<'de> Deserialize<'de> for Point {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct PointVisitor {}
        impl<'de> Visitor<'de> for PointVisitor {
            type Value = Point;
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a hexidecimal number")
            }

            fn visit_seq<A>(self, mut v: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let dim = v
                    .next_element::<i64>()?
                    .ok_or_else(|| A::Error::custom("expected dim in point"))?;
                let mut result = Vec::new();
                for _ in 0..dim {
                    result.push(
                        v.next_element()?
                            .ok_or_else(|| A::Error::custom("expected point element"))?,
                    );
                }
                // Legion always logs up to MAX_DIM, but anything
                // beyond the dynamic value of dim should be zero.
                while let Some(element) = v.next_element::<i64>()? {
                    // FIXME: Elliott: https://github.com/StanfordLegion/legion/issues/1333
                    // assert!(element == 0);
                }
                Ok(Point(result))
            }
        }

        deserializer.deserialize_seq(PointVisitor {})
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rect {
    lo: Vec<i64>,
    hi: Vec<i64>,
}

impl<'de> Deserialize<'de> for Rect {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RectVisitor {}
        impl<'de> Visitor<'de> for RectVisitor {
            type Value = Rect;
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a hexidecimal number")
            }

            fn visit_seq<A>(self, mut v: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let dim = v
                    .next_element::<i64>()?
                    .ok_or_else(|| A::Error::custom("expected dim in rect"))?;
                let mut lo = Vec::new();
                let mut hi = Vec::new();
                for i in 0..2 * dim {
                    let element = v
                        .next_element()?
                        .ok_or_else(|| A::Error::custom("expected rect element"))?;
                    if i % 2 == 0 {
                        lo.push(element);
                    } else {
                        hi.push(element);
                    }
                }
                // Legion always logs up to MAX_DIM, but anything
                // beyond the dynamic value of dim should be zero.
                while let Some(element) = v.next_element::<i64>()? {
                    assert!(element == 0);
                }
                Ok(Rect { lo, hi })
            }
        }

        deserializer.deserialize_seq(RectVisitor {})
    }
}

#[rustfmt::skip]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub enum Record {
    // Since the parser checks for these in order, arrange them in (rough)
    // order from most to least frequent to improve parsing time.

    // Physical event and operation patterns
    #[serde(rename = "Event Event")]
    EventDependence { id1: EventID, id2: EventID },
    #[serde(rename = "Ap User Event Trigger")]
    ApUserEventTrigger { id: EventID },
    #[serde(rename = "Ap User Event")]
    ApUserEvent { id: EventID, provenance: u64 },
    #[serde(rename = "Rt User Event Trigger")]
    RtUserEventTrigger { id: EventID },
    #[serde(rename = "Rt User Event")]
    RtUserEvent { id: EventID, provenance: u64 },
    #[serde(rename = "Pred Event Trigger")]
    PredEventTrigger { id: EventID },
    #[serde(rename = "Pred Event")]
    PredEvent { id: EventID },
    #[serde(rename = "Operation Events")]
    OperationEvents { uid: UniqueID, pre: EventID, post: EventID },
    #[serde(rename = "Copy Events")]
    RealmCopy { uid: UniqueID, expr: ExprID, src_tid: TreeID, dst_tid: TreeID, pre: EventID, post: EventID },
    #[serde(rename = "Copy Field")]
    RealmCopyField { id: EventID, srcfid: FieldID, srcid: EventID, dstfid: FieldID, dstid: EventID, redop: u64 },
    #[serde(rename = "Indirect Events")]
    IndirectCopy { uid: UniqueID, expr: ExprID, indirect: IndirectID, pre: EventID, post: EventID },
    #[serde(rename = "Indirect Field")]
    IndirectField { id: EventID, srcfid: FieldID, srcid: EventID, dstfid: FieldID, dstid: EventID, redop: u64 },
    #[serde(rename = "Indirect Instance")]
    IndirectInstance { indirect: IndirectID, index: u64, inst: InstID, fid: FieldID },
    #[serde(rename = "Indirect Group")]
    IndirectGroup { indirect: IndirectID, index: u64, inst: InstID, ispace: IspaceID },
    #[serde(rename = "Fill Events")]
    RealmFill { uid: UniqueID, ispace: IspaceID, fspace: FspaceID, tid: TreeID, pre: EventID, post: EventID, fill_uid: UniqueID },
    #[serde(rename = "Fill Field")]
    RealmFillField { id: EventID, fid: FieldID, dstid: EventID },
    #[serde(rename = "Deppart Events")]
    RealmDepPart { uid: UniqueID, ispace: IspaceID, preid: EventID, postid: EventID },
    #[serde(rename = "Phase Barrier Arrive")]
    BarrierArrive { uid: UniqueID, iid: EventID },
    #[serde(rename = "Phase Barrier Wait")]
    BarrierWait { uid: UniqueID, iid: EventID },
    #[serde(rename = "Replay Operation")]
    ReplayOperation { uid: UniqueID },

    // Patterns for logical analysis and region requirements
    #[serde(rename = "Logical Requirement Field")]
    RequirementField { uid: UniqueID, index: u64, fid: FieldID },
    #[serde(rename = "Logical Requirement Projection")]
    RequirementProjection { uid: UniqueID, index: u64, pid: ProjectionID },
    #[serde(rename = "Logical Requirement")]
    Requirement { uid: UniqueID, index: u64, reg: bool, iid: IspaceID, fid: FspaceID, tid: TreeID, privilege: u64, coherence: u64, redop: u64, pis: IspaceID },
    #[serde(rename = "Projection Function")]
    ProjectionFunc { pid: ProjectionID, depth: u64, invertible: bool },
    #[serde(rename = "Index Launch Rect")]
    IndexLaunchDomain { uid: UniqueID, rect: Rect },
    #[serde(rename = "Mapping Dependence")]
    MappingDependence { ctx: ContextID, prev_id: UniqueID, pidx: u64, next_id: UniqueID, nidx: u64, dtype: u64 },
    #[serde(rename = "Future Creation")]
    FutureCreate { uid: UniqueID, iid: FutureID, point: Point },
    #[serde(rename = "Future Usage")]
    FutureUse { uid: UniqueID, iid: FutureID },
    #[serde(rename = "Predicate Use")]
    PredicateUse { uid: UniqueID, pred: UniqueID },

    // Physical instance and mapping decision patterns
    #[serde(rename = "Physical Instance Field")]
    InstanceField { eid: EventID, fid: FieldID },
    #[serde(rename = "Physical Instance Creator")]
    InstanceCreator { eid: EventID, uid: UniqueID, proc: ProcID },
    #[serde(rename = "Physical Instance Creation Region")]
    InstanceCreationRegion { eid: EventID, iid: IspaceID, fid: FspaceID, tid: TreeID },
    #[serde(rename = "Physical Instance")]
    Instance { eid: EventID, iid: InstID, mid: MemID, redop: u64, expr: ExprID, fid: FspaceID, tid: TreeID },
    #[serde(rename = "Instance Specialized Constraint")]
    SpecializedConstraint { eid: EventID, kind: u64, redop: u64 },
    #[serde(rename = "Instance Memory Constraint")]
    MemConstraint { eid: EventID, kind: u64 },
    #[serde(rename = "Instance Field Constraint Field")]
    FieldConstraintField { eid: EventID, fid: FieldID },
    #[serde(rename = "Instance Field Constraint")]
    FieldConstraint { eid: EventID, contig: bool, inorder: bool, fields: u64 },
    #[serde(rename = "Instance Ordering Constraint Dimension")]
    OrderingConstraintDim { eid: EventID, dim: u64 },
    #[serde(rename = "Instance Ordering Constraint")]
    OrderingConstraint { eid: EventID, contig: bool, dims: u64 },
    #[serde(rename = "Instance Splitting Constraint")]
    SplittingConstraint { eid: EventID, dim: u64, value: u64, chunks: bool },
    #[serde(rename = "Instance Dimension Constraint")]
    DimConstraint { eid: EventID, dim: u64, eqk: u64, value: u64 },
    #[serde(rename = "Instance Alignment Constraint")]
    AlignmentConstraint { eid: EventID, fid: FieldID, eqk: u64, align: u64 },
    #[serde(rename = "Instance Offset Constraint")]
    OffsetConstraint { eid: EventID, fid: FieldID, offset: u64 },
    #[serde(rename = "Variant Decision")]
    VariantDecision { uid: UniqueID, vid: VariantID },
    #[serde(rename = "Mapping Decision")]
    MappingDecision { uid: UniqueID, idx: u64, fid: FieldID, eid: EventID },
    #[serde(rename = "Post Mapping Decision")]
    PostDecision { uid: UniqueID, idx: u64, fid: FieldID, eid: EventID },
    #[serde(rename = "Task Priority")]
    TaskPriority { uid: UniqueID, priority: i64 },
    #[serde(rename = "Task Processor")]
    TaskProcessor { uid: UniqueID, proc: ProcID },
    #[serde(rename = "Task Premapping")]
    TaskPremapping { uid: UniqueID, index: u64 },
    #[serde(rename = "Task Tunable")]
    TaskTunable { uid: UniqueID, index: u64, bytes: u64, value: String },

    // Patterns for operations
    #[serde(rename = "Task ID Name")]
    TaskName { tid: TaskID, name: String },
    #[serde(rename = "Task Variant")]
    TaskVariant { tid: TaskID, vid: VariantID, inner: bool, leaf: bool, idempotent: bool, name: String },
    #[serde(rename = "Top Task")]
    TopTask { tid: TaskID, ctx: ContextID, uid: UniqueID, name: String },
    #[serde(rename = "Individual Task")]
    IndividualTask { ctx: ContextID, tid: TaskID, uid: UniqueID, index: u64, name: String },
    #[serde(rename = "Index Task")]
    IndexTask { ctx: ContextID, tid: TaskID, uid: UniqueID, index: u64, name: String },
    #[serde(rename = "Inline Task")]
    InlineTask { uid: UniqueID },
    #[serde(rename = "Mapping Operation")]
    MappingOperation { ctx: ContextID, uid: UniqueID, index: u64 },
    #[serde(rename = "Close Operation")]
    CloseOperation { ctx: ContextID, uid: UniqueID, index: u64, is_inter: bool },
    #[serde(rename = "Internal Operation Creator")]
    InternalCreator { uid: UniqueID, cuid: UniqueID, index: u64 },
    #[serde(rename = "Fence Operation")]
    FenceOperation { ctx: ContextID, uid: UniqueID, index: u64 },
    #[serde(rename = "Trace Operation")]
    TraceOperation { ctx: ContextID, uid: UniqueID },
    #[serde(rename = "Copy Operation")]
    CopyOperation { ctx: ContextID, uid: UniqueID, kind: u64, index: u64, src_indirect: bool, dst_indirect: bool },
    #[serde(rename = "Fill Operation")]
    FillOperation { ctx: ContextID, uid: UniqueID, index: u64 },
    #[serde(rename = "Acquire Operation")]
    AcquireOperation { ctx: ContextID, uid: UniqueID, index: u64 },
    #[serde(rename = "Release Operation")]
    ReleaseOperation { ctx: ContextID, uid: UniqueID, index: u64 },
    #[serde(rename = "Creation Operation")]
    CreationOperation { ctx: ContextID, uid: UniqueID, index: u64 },
    #[serde(rename = "Deletion Operation")]
    DeletionOperation { ctx: ContextID, uid: UniqueID, index: u64, unordered: bool },
    #[serde(rename = "Attach Operation")]
    AttachOperation { ctx: ContextID, uid: UniqueID, index: u64, restricted: bool },
    #[serde(rename = "Detach Operation")]
    DetachOperation { ctx: ContextID, uid: UniqueID, index: u64, unordered: bool },
    #[serde(rename = "Unordered Operation")]
    UnorderedOperation { ctx: ContextID, uid: UniqueID, index: u64 },
    #[serde(rename = "Dynamic Collective")]
    DynamicCollective { ctx: ContextID, uid: UniqueID, index: u64 },
    #[serde(rename = "Timing Operation")]
    TimingOperation { ctx: ContextID, uid: UniqueID, index: u64 },
    #[serde(rename = "Tunable Operation")]
    TunableOperation { ctx: ContextID, uid: UniqueID, index: u64 },
    #[serde(rename = "All Reduce Operation")]
    AllReduceOperation { ctx: ContextID, uid: UniqueID, index: u64 },
    #[serde(rename = "Predicate Operation")]
    PredicateOperation { ctx: ContextID, uid: UniqueID },
    #[serde(rename = "Must Epoch Operation")]
    MustEpochOperation { ctx: ContextID, uid: UniqueID },
    #[serde(rename = "Summary Operation Creator")]
    SummaryCreator { uid: UniqueID, cuid: UniqueID },
    #[serde(rename = "Summary Operation")]
    SummaryOperation { ctx: ContextID, uid: UniqueID },
    #[serde(rename = "Dependent Partition Operation")]
    DepPartOperation { ctx: ContextID, uid: UniqueID, pid: IpartID, kind: u64, index: u64 },
    #[serde(rename = "Pending Partition Operation")]
    PendingPartOperation { ctx: ContextID, uid: UniqueID, index: u64 },
    #[serde(rename = "Pending Partition Target")]
    PendingPartTarget { uid: UniqueID, pid: IpartID, kind: u64 },
    #[serde(rename = "Index Slice")]
    IndexSlice { index: UniqueID, slice: UniqueID },
    #[serde(rename = "Slice Slice")]
    SliceSlice { slice1: UniqueID, slice2: UniqueID },
    #[serde(rename = "Slice Point")]
    SlicePoint { slice: UniqueID, point_id: UniqueID, point: Point },
    #[serde(rename = "Point Point")]
    PointPoint { point1: UniqueID, point2: UniqueID },
    #[serde(rename = "Index Point")]
    IndexPoint { index: UniqueID, point_id: UniqueID, point: Point },
    #[serde(rename = "Intra Space Dependence")]
    IntraSpace { point_id: UniqueID, point: Point },
    #[serde(rename = "Operation Index")]
    OperationIndex { parent: UniqueID, index: u64, child: UniqueID },
    #[serde(rename = "Operation Provenance")]
    OperationProvenance { uid: UniqueID, provenance: String },
    #[serde(rename = "Close Index")]
    CloseIndex { parent: UniqueID, index: u64, child: UniqueID },
    #[serde(rename = "Predicate False")]
    PredicateFalse { uid: UniqueID },

    // Patterns for the shape of region trees
    #[serde(rename = "Index Space Name")]
    IspaceName { uid: IspaceID, name: String },
    #[serde(rename = "Index Space Point")]
    IspacePoint { uid: IspaceID, point: Point },
    #[serde(rename = "Index Space Rect")]
    IspaceRect { uid: IspaceID, rect: Rect },
    #[serde(rename = "Empty Index Space")]
    IspaceEmpty { uid: IspaceID },
    #[serde(rename = "Index Space Expression")]
    IspaceExpr { uid: IspaceID, expr: ExprID },
    #[serde(rename = "Index Space Union")]
    ExprUnion { expr: ExprID, count: u64, sources: Vec<ExprID> },
    #[serde(rename = "Index Space Intersection")]
    ExprIntersect { expr: ExprID, count: u64, sources: Vec<ExprID> },
    #[serde(rename = "Index Space Difference")]
    ExprDiff { result: ExprID, left: ExprID, right: ExprID },
    #[serde(rename = "Index Space")]
    Ispace { uid: IspaceID, owner: AddressSpaceID, provenance: String },
    #[serde(rename = "Index Partition Name")]
    IpartName { uid: IpartID, name: String },
    #[serde(rename = "Index Partition")]
    Ipart { pid: IspaceID, uid: IpartID, disjoint: u8, complete: u8, color: u64, owner: AddressSpaceID, provenance: String },
    #[serde(rename = "Index Subspace")]
    IspaceSubspace { pid: IpartID, uid: IspaceID, owner: AddressSpaceID, color: Point },
    #[serde(rename = "Field Space Name")]
    FspaceName { uid: FspaceID, name: String },
    #[serde(rename = "Field Space")]
    Fspace { uid: FspaceID, owner: AddressSpaceID, provenance: String },
    #[serde(rename = "Field Creation")]
    FieldCreate { uid: FspaceID, fid: FieldID, size: u64, provenance: String },
    #[serde(rename = "Field Name")]
    FieldName { uid: FspaceID, fid: FieldID, name: String },
    #[serde(rename = "Region")]
    Region { iid: IspaceID, fid: FspaceID, tid: TreeID, owner: AddressSpaceID, provenance: String },
    #[serde(rename = "Logical Region Name")]
    RegionName { iid: IspaceID, fid: FspaceID, tid: TreeID, name: String },
    #[serde(rename = "Logical Partition Name")]
    PartName { iid: IpartID, fid: FspaceID, tid: TreeID, name: String },

    // Configuration patterns
    #[serde(rename = "Legion Spy Logging")]
    SpyLogging,
    #[serde(rename = "Legion Spy Detailed Logging")]
    SpyDetailedLogging,
    #[serde(rename = "Processor Kind")]
    ProcKind { kind: u64, name: String },
    #[serde(rename = "Memory Kind")]
    MemKind { kind: u64, name: String },
    #[serde(rename = "Processor Memory")]
    ProcMem { pid: ProcID, mid: MemID, bandwidth: u64, latency: u64 },
    #[serde(rename = "Memory Memory")]
    MemMem { mid1: MemID, mid2: MemID, bandwidth: u64, latency: u64 },
    #[serde(rename = "Processor")]
    Proc { pid: ProcID, kind: u64 },
    #[serde(rename = "Memory")]
    Mem { mid: MemID, capacity: u64, kind: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spy_logging() {
        let a = "Legion Spy Logging";
        let b = Record::SpyLogging;
        assert_eq!(b, from_str(a).unwrap());
    }

    #[test]
    fn test_proc() {
        let a = "Processor 0123abcd 123";
        let b = Record::Proc {
            pid: ProcID(HexU64(0x0123abcd)),
            kind: 123,
        };
        assert_eq!(b, from_str(a).unwrap());
    }

    #[test]
    fn test_ispace() {
        let a = "Index Space 1 0 ";
        let b = Record::Ispace {
            uid: IspaceID(HexU64(0x1)),
            owner: AddressSpaceID(0),
            provenance: "".to_owned(),
        };
        assert_eq!(b, from_str(a).unwrap());
    }

    #[test]
    fn test_ispace_provenance() {
        let a = "Index Space 1 0 asdf qwer zxcv";
        let b = Record::Ispace {
            uid: IspaceID(HexU64(0x1)),
            owner: AddressSpaceID(0),
            provenance: "asdf qwer zxcv".to_owned(),
        };
        assert_eq!(b, from_str(a).unwrap());
    }

    #[test]
    fn test_ispace_point() {
        let a = "Index Space Point abcd1234 3 1 2 3";
        let b = Record::IspacePoint {
            uid: IspaceID(HexU64(0xabcd1234)),
            point: Point(vec![1, 2, 3]),
        };
        assert_eq!(b, from_str(a).unwrap());
    }

    #[test]
    fn test_ispace_rect() {
        let a = "Index Space Rect abcd1234 3 1 2 3 4 5 6";
        let b = Record::IspaceRect {
            uid: IspaceID(HexU64(0xabcd1234)),
            rect: Rect {
                lo: vec![1, 3, 5],
                hi: vec![2, 4, 6],
            },
        };
        assert_eq!(b, from_str(a).unwrap());
    }

    #[test]
    fn test_task_name() {
        let a = "Task ID Name 10000 make_private_partition";
        let b = Record::TaskName {
            tid: TaskID(10000),
            name: "make_private_partition".to_owned(),
        };
        assert_eq!(b, from_str(a).unwrap());
    }
}

fn parse_prefix(input: &str) -> IResult<&str, Prefix> {
    // node and thread
    let (input, _) = tag("[")(input)?;
    let (input, node) = u64(input)?;
    let (input, _) = tag(" - ")(input)?;
    let (input, _) = hex_digit1(input)?;
    let (input, _) = tag("]")(input)?;

    // timestamp
    let (input, _) = space0(input)?;
    let (input, _) = digit1(input)?;
    let (input, _) = tag(".")(input)?;
    let (input, _) = digit1(input)?;
    let (input, _) = space0(input)?;

    // log level
    let (input, _) = tag("{")(input)?;
    let (input, _) = digit1(input)?;
    let (input, _) = tag("}{")(input)?;
    let (input, spy) = alt((
        value(true, preceded(tag("legion_spy"), peek(tag("}")))),
        value(false, many1(none_of("}"))),
    ))(input)?;
    let (input, _) = tag("}:")(input)?;
    let (input, _) = space0(input)?;
    Ok((input, Prefix { _node: node, spy }))
}

fn discard_rest_of_line(input: &str) -> IResult<&str, Option<Record>> {
    let (input, _) = not_line_ending(input)?;
    let (input, _) = line_ending(input)?;
    Ok((input, None))
}

fn parse_record(input: &str) -> IResult<&str, Option<Record>> {
    let (input, prefix) = opt(parse_prefix)(input)?;
    if !prefix.map_or(false, |p| p.spy) {
        return discard_rest_of_line(input);
    }
    let (input, record) = map(not_line_ending, |line| from_str(line).unwrap())(input)?;
    let (input, _) = line_ending(input)?;
    Ok((input, Some(record)))
}

fn parse(input: &str) -> IResult<&str, Vec<Record>> {
    let (input, records) = all_consuming(many1(parse_record))(input)?;
    Ok((input, records.into_iter().filter_map(|x| x).collect()))
}

pub fn deserialize<P: AsRef<Path>>(path: P) -> io::Result<Vec<Record>> {
    let mut f = File::open(path)?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    // throw error here if parse failed
    let (rest, records) = parse(&s).unwrap();
    assert_eq!(rest.len(), 0);
    Ok(records)
}
