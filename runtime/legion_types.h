/* Copyright 2014 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef __LEGION_TYPES_H__
#define __LEGION_TYPES_H__

#include "lowlevel.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <stdint.h>

#include "limits.h"

#include <map>
#include <set>
#include <list>
#include <deque>
#include <vector>

// If we enable field tree acceleration
// then enable it for both the logical and
// physical tree analyses.
#ifdef FIELD_TREE
#define LOGICAL_FIELD_TREE
#define PHYSICAL_FIELD_TREE
#endif

#define AUTO_GENERATE_ID   UINT_MAX

#ifndef MAX_RETURN_SIZE
#define MAX_RETURN_SIZE    2048 // maximum return type size in bytes
#endif

#ifndef MAX_FIELDS
#define MAX_FIELDS         128 // must be divisible by 2^FIELD_SHIFT
#endif

#ifndef FIELD_LOG2
#define FIELD_LOG2         7 // log2(MAX_FIELDS)
#endif
// The folowing macros are used in the FieldMask instantiation of BitMask
// If you change one you probably have to change the others too
#define FIELD_TYPE          uint64_t 
#define FIELD_SHIFT         6
#define FIELD_MASK          0x3F
#define FIELD_ALL_ONES      0xFFFFFFFFFFFFFFFF

// Some default values 

// The maximum number of nodes to be run on
#ifndef MAX_NUM_NODES
#define MAX_NUM_NODES                   1024
#endif
// The maximum number of processors on a node
#ifndef MAX_NUM_PROCS
#define MAX_NUM_PROCS                   1024
#endif
// Default number of mapper slots
#ifndef DEFAULT_MAPPER_SLOTS
#define DEFAULT_MAPPER_SLOTS            8
#endif
// Default number of contexts made for each runtime instance
// Ideally this is a power of 2 (better for performance)
#ifndef DEFAULT_CONTEXTS
#define DEFAULT_CONTEXTS                8 
#endif
// Maximum number of allowed contexts ever in Legion runtime
#ifndef MAX_CONTEXTS
#define MAX_CONTEXTS                    1024
#endif
// Maximum number of sub-tasks per task at a time
#ifndef DEFAULT_MAX_TASK_WINDOW
#define DEFAULT_MAX_TASK_WINDOW         1024 
#endif
// How many tasks to group together for runtime operations
#ifndef DEFAULT_MIN_TASKS_TO_SCHEDULE
#define DEFAULT_MIN_TASKS_TO_SCHEDULE   4
#endif
// Scheduling granularity for how many operations to
// handle at a time at each stage of the pipeline
#ifndef DEFAULT_SUPERSCALAR_WIDTH
#define DEFAULT_SUPERSCALAR_WIDTH       4
#endif
// The maximum size of active messages sent by the runtime in bytes
// Note this value was picked based on making a tradeoff between
// latency and bandwidth numbers on both Cray and Infiniband
// interconnect networks.
#ifndef DEFAULT_MAX_MESSAGE_SIZE
#define DEFAULT_MAX_MESSAGE_SIZE        16384 
#endif
// Maximum number of tasks in logical region node before consolidation
#ifndef DEFAULT_MAX_FILTER_SIZE
#define DEFAULT_MAX_FILTER_SIZE         0
#endif
// Timeout before checking for whether a logical user
// should be pruned from the logical region tree data strucutre
// Making the value less than or equal to zero will
// result in checks always being performed
#ifndef DEFAULT_LOGICAL_USER_TIMEOUT
#define DEFAULT_LOGICAL_USER_TIMEOUT    32
#endif

namespace BindingLib { class Utility; } // BindingLib namespace


namespace LegionRuntime {
  /**
   * \struct LegionStaticAssert
   * Help with static assertions.
   */
  template<bool> struct LegionStaticAssert;
  template<> struct LegionStaticAssert<true> { };
#define LEGION_STATIC_ASSERT(condition) \
  do { LegionStaticAssert<(condition)>(); } while (0)

  /**
   * \struct LegionTypeEquality
   * Help with checking equality of types.
   */
  template<typename T, typename U>
  struct LegionTypeInequality {
  public:
    static const bool value = true;
  };
  template<typename T>
  struct LegionTypeInequality<T,T> {
  public:
    static const bool value = false;
  };
  
  namespace HighLevel {
    
    enum LegionErrorType {
      NO_ERROR = 0,
      ERROR_RESERVED_REDOP_ID = 1,
      ERROR_DUPLICATE_REDOP_ID = 2,
      ERROR_RESERVED_TYPE_HANDLE = 3,
      ERROR_DUPLICATE_TYPE_HANDLE = 4,
      ERROR_DUPLICATE_FIELD_ID = 5,
      ERROR_PARENT_TYPE_HANDLE_NONEXISTENT = 6,
      ERROR_MISSING_PARENT_FIELD_ID = 7,
      ERROR_RESERVED_PROJECTION_ID = 8,
      ERROR_DUPLICATE_PROJECTION_ID = 9,
      ERROR_UNREGISTERED_VARIANT = 10,
      ERROR_USE_REDUCTION_REGION_REQ = 11,
      ERROR_INVALID_ACCESSOR_REQUESTED = 12,
      ERROR_PHYSICAL_REGION_UNMAPPED = 13,
      ERROR_RESERVED_TASK_ID = 14,
      ERROR_INVALID_ARG_MAP_DESTRUCTION = 15,
      ERROR_RESERVED_MAPPING_ID = 16,
      ERROR_BAD_INDEX_PRIVILEGES = 17,
      ERROR_BAD_FIELD_PRIVILEGES = 18,
      ERROR_BAD_REGION_PRIVILEGES = 19,
      ERROR_BAD_PARTITION_PRIVILEGES = 20,
      ERROR_BAD_PARENT_INDEX = 21,
      ERROR_BAD_INDEX_PATH = 22,
      ERROR_BAD_PARENT_REGION = 23,
      ERROR_BAD_REGION_PATH = 24,
      ERROR_BAD_PARTITION_PATH = 25,
      ERROR_BAD_FIELD = 26,
      ERROR_BAD_REGION_TYPE = 27,
      ERROR_INVALID_TYPE_HANDLE = 28,
      ERROR_LEAF_TASK_VIOLATION = 29,
      ERROR_INVALID_REDOP_ID = 30,
      ERROR_REDUCTION_INITIAL_VALUE_MISMATCH = 31,
      ERROR_INVALID_UNMAP_OP = 32,
      ERROR_INVALID_DUPLICATE_MAPPING = 33,
      ERROR_INVALID_REGION_ARGUMENT_INDEX = 34,
      ERROR_INVALID_MAPPING_ACCESS = 35,
      ERROR_STALE_INLINE_MAPPING_ACCESS = 36,
      ERROR_INVALID_INDEX_SPACE_PARENT = 37,
      ERROR_INVALID_INDEX_PART_PARENT = 38,
      ERROR_INVALID_INDEX_SPACE_COLOR = 39,
      ERROR_INVALID_INDEX_PART_COLOR = 40,
      ERROR_INVALID_INDEX_SPACE_HANDLE = 41,
      ERROR_INVALID_INDEX_PART_HANDLE = 42,
      ERROR_FIELD_SPACE_FIELD_MISMATCH = 43,
      ERROR_INVALID_INSTANCE_FIELD = 44,
      ERROR_DUPLICATE_INSTANCE_FIELD = 45,
      ERROR_TYPE_INST_MISMATCH = 46,
      ERROR_TYPE_INST_MISSIZE = 47,
      ERROR_INVALID_INDEX_SPACE_ENTRY = 48,
      ERROR_INVALID_INDEX_PART_ENTRY = 49,
      ERROR_INVALID_FIELD_SPACE_ENTRY = 50,
      ERROR_INVALID_REGION_ENTRY = 51,
      ERROR_INVALID_PARTITION_ENTRY = 52,
      ERROR_ALIASED_INTRA_TASK_REGIONS = 53,
      ERROR_MAX_FIELD_OVERFLOW = 54,
      ERROR_MISSING_TASK_COLLECTION = 55,
      ERROR_INVALID_IDENTITY_PROJECTION_USE = 56,
      ERROR_INVALID_PROJECTION_ID = 57,
      ERROR_NON_DISJOINT_PARTITION = 58,
      ERROR_BAD_PROJECTION_USE = 59,
      ERROR_INDEPENDENT_SLICES_VIOLATION = 60,
      ERROR_INVALID_REGION_HANDLE = 61,
      ERROR_INVALID_PARTITION_HANDLE = 62,
      ERROR_VIRTUAL_MAP_IN_LEAF_TASK = 63,
      ERROR_LEAF_MISMATCH = 64,
      ERROR_INVALID_PROCESSOR_SELECTION = 65,
      ERROR_INVALID_VARIANT_SELECTION = 66,
      ERROR_INVALID_MAPPER_OUTPUT = 67,
      ERROR_UNINITIALIZED_REDUCTION = 68,
      ERROR_INVALID_INDEX_DOMAIN = 69,
      ERROR_INVALID_INDEX_PART_DOMAIN = 70,
      ERROR_DISJOINTNESS_TEST_FAILURE = 71,
      ERROR_NON_DISJOINT_TASK_REGIONS = 72,
      ERROR_INVALID_FIELD_ACCESSOR_PRIVILEGES = 73,
      ERROR_INVALID_PREMAPPED_REGION_LOCATION = 74,
      ERROR_IDEMPOTENT_MISMATCH = 75,
      ERROR_INVALID_MAPPER_ID = 76,
      ERROR_INVALID_TREE_ENTRY = 77,
      ERROR_SEPARATE_UTILITY_PROCS = 78,
      ERROR_MAXIMUM_NODES_EXCEEDED = 79,
      ERROR_MAXIMUM_PROCS_EXCEEDED = 80,
      ERROR_INVALID_TASK_ID = 81,
      ERROR_INVALID_MAPPER_DOMAIN_SLICE = 82,
      ERROR_UNFOLDABLE_REDUCTION_OP = 83,
      ERROR_INVALID_INLINE_ID = 84,
      ERROR_ILLEGAL_MUST_PARALLEL_INLINE = 85,
      ERROR_RETURN_SIZE_MISMATCH = 86,
      ERROR_ACCESSING_EMPTY_FUTURE = 87,
      ERROR_ILLEGAL_PREDICATE_FUTURE = 88,
      ERROR_COPY_REQUIREMENTS_MISMATCH = 89,
      ERROR_INVALID_COPY_FIELDS_SIZE = 90,
      ERROR_COPY_SPACE_MISMATCH = 91,
      ERROR_INVALID_COPY_PRIVILEGE = 92,
      ERROR_INVALID_PARTITION_COLOR = 93,
      ERROR_EXCEEDED_MAX_CONTEXTS = 94,
      ERROR_ACQUIRE_MISMATCH = 95,
      ERROR_RELEASE_MISMATCH = 96,
      ERROR_INNER_LEAF_MISMATCH = 97,
      ERROR_INVALID_FIELD_PRIVILEGES = 98,
      ERROR_ILLEGAL_NESTED_TRACE = 99,
      ERROR_UNMATCHED_END_TRACE = 100,
      ERROR_CONFLICTING_PARENT_MAPPING_DEADLOCK = 101,
      ERROR_CONFLICTING_SIBLING_MAPPING_DEADLOCK = 102,
      ERROR_INVALID_PARENT_REQUEST = 103,
      ERROR_INVALID_FIELD_ID = 104,
      ERROR_NESTED_MUST_EPOCH = 105,
      ERROR_UNMATCHED_MUST_EPOCH = 106,
      ERROR_MUST_EPOCH_FAILURE = 107,
      ERROR_DOMAIN_DIM_MISMATCH = 108,
      ERROR_INVALID_PROCESSOR_NAME = 109,
    };

    // enum and namepsaces don't really get along well
    enum PrivilegeMode {
      NO_ACCESS       = 0x00000000, // Deprecated: use the NO_ACCESS_FLAG
      READ_ONLY       = 0x00000001,
      READ_WRITE      = 0x00000111,
      WRITE_ONLY      = 0x00000010, // same as WRITE_DISCARD
      WRITE_DISCARD   = 0x00000010, // same as WRITE_ONLY
      REDUCE          = 0x00000100,
      PROMOTED        = 0x00001000, // Internal use only
    };

    enum AllocateMode {
      NO_MEMORY       = 0x00000000,
      ALLOCABLE       = 0x00000001,
      FREEABLE        = 0x00000010,
      MUTABLE         = 0x00000011,
      REGION_CREATION = 0x00000100,
      REGION_DELETION = 0x00001000,
      ALL_MEMORY      = 0x00001111,
    };

    enum CoherenceProperty {
      EXCLUSIVE    = 0,
      ATOMIC       = 1,
      SIMULTANEOUS = 2,
      RELAXED      = 3,
    };

    // Optional region requirement flags
    enum {
      NO_FLAG         = 0x00000000,
      VERIFIED_FLAG  = 0x00000001,
      NO_ACCESS_FLAG  = 0x00000002,
    };

    enum HandleType {
      SINGULAR, // a single logical region
      PART_PROJECTION, // projection from a partition
      REG_PROJECTION, // projection from a region
    };

    enum DependenceType {
      NO_DEPENDENCE = 0,
      TRUE_DEPENDENCE = 1,
      ANTI_DEPENDENCE = 2, // Write-After-Read or Write-After-Write with Write-Only privilege
      ATOMIC_DEPENDENCE = 3,
      SIMULTANEOUS_DEPENDENCE = 4,
      PROMOTED_DEPENDENCE = 5,
    };

    enum OpenState {
      NOT_OPEN            = 0,
      OPEN_READ_ONLY      = 1,
      OPEN_READ_WRITE     = 2, // unknown dirty information below
      OPEN_SINGLE_REDUCE  = 3, // only one open child with reductions below
      OPEN_MULTI_REDUCE   = 4, // multiple open children with same reduction
    };

    // Runtime task numbering 
    enum {
      INIT_FUNC_ID          = LowLevel::Processor::TASK_ID_PROCESSOR_INIT,
      SHUTDOWN_FUNC_ID      = LowLevel::Processor::TASK_ID_PROCESSOR_SHUTDOWN,
      SCHEDULER_ID          = LowLevel::Processor::TASK_ID_PROCESSOR_IDLE,
      MESSAGE_TASK_ID       = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+0),
      POST_END_TASK_ID      = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+1),
      DEFERRED_COMPLETE_ID  = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+2),
      RECLAIM_LOCAL_FID     = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+3),
      DEFERRED_COLLECT_ID   = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+4),
      TRIGGER_DEPENDENCE_ID = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+5),
      TRIGGER_OP_ID         = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+6),
      TRIGGER_TASK_ID       = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+7),
      DEFERRED_RECYCLE_ID   = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+8),
      TASK_ID_AVAILABLE     = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+9),
    };

    // Forward declarations for user level objects
    // legion.h
    class FieldSpace;
    class LogicalRegion;
    class LogicalPartition;
    class IndexAllocator;
    class FieldAllocator;
    class TaskArgument;
    class ArgumentMap;
    class Lock;
    struct LockRequest;
    class Grant;
    class PhaseBarrier;
    struct RegionRequirement;
    struct IndexSpaceRequirement;
    struct FieldSpaceRequirement;
    struct TaskLauncher;
    struct IndexLauncher;
    struct InlineLauncher;
    struct CopyLauncher;
    struct AcquireLauncher;
    struct ReleaseLauncher;
    class Future;
    class FutureMap;
    class Predicate;
    class PhysicalRegion;
    class IndexIterator;
    class Mappable;
    class Task;
    class Copy;
    class Inline;
    class Acquire;
    class Release;
    class TaskVariantCollection;
    class Mapper; 
    template<typename T> struct ColoredPoints; 
    struct InputArgs;
    class HighLevelRuntime;

    // Forward declarations for compiler level objects
    // legion.h
    class ColoringSerializer;
    class DomainColoringSerializer;

    // Forward declarations for wrapper tasks
    // legion.h
    class LegionTaskWrapper;
    class LegionSerialization;

    // Forward declarations for runtime level objects
    // runtime.h
    class Collectable;
    class ArgumentMapStore;
    class ProcessorManager;
    class Runtime;

    // legion_ops.h
    class Operation;
    class SpeculativeOp;
    class MapOp;
    class CopyOp;
    class FenceOp;
    class DeletionOp;
    class CloseOp;
    class AcquireOp;
    class ReleaseOp;
    class FuturePredOp;
    class NotPredOp;
    class AndPredOp;
    class OrPredOp;
    class MustEpochOp;
    class TaskOp;

    // legion_tasks.h
    class SingleTask;
    class MultiTask;
    class IndividualTask;
    class PointTask;
    class WrapperTask;
    class RemoteTask;
    class InlineTask;
    class IndexTask;
    class SliceTask;
    class RemoteTask;
    
    // legion_trace.h
    class LegionTrace;
    class TraceCaptureOp;
    class TraceCompleteOp;

    // region_tree.h
    class RegionTreeForest;
    class IndexTreeNode;
    class IndexSpaceNode;
    class IndexPartNode;
    class FieldSpaceNode;
    class RegionTreeNode;
    class RegionNode;
    class PartitionNode;

    class RegionTreeContext;
    class RegionTreePath;
    class PathTraverser;
    class NodeTraverser;
    template<bool DOM>
    class LogicalRegistrar;
    class LogicalInitializer;
    class LogicalInvalidator;
    class PremapTraverser;
    class MappingTraverser;

    struct LogicalState;
    class PhysicalVersion;

    class DistributedCollectable;
    class HierarchicalCollectable;
    class PhysicalManager; // base class for instance and reduction
    class LogicalView; // base class for instance and reduction
    class InstanceManager;
    class InstanceKey;
    class InstanceView;
    class MaterializedView;
    class CompositeView;
    class CompositeNode;
    class MappingRef;
    class InstanceRef;
    class InnerTaskView;
    class ReductionManager;
    class ListReductionManager;
    class FoldReductionManager;
    class ReductionView;

    class RegionAnalyzer;
    class RegionMapper;

    struct RegionUsage;

    struct EscapedUser;
    struct EscapedCopy;
    struct GenericUser;
    struct LogicalUser;
    struct PhysicalUser;
    class TreeCloser;
    struct LogicalCloser;
    struct PhysicalCloser;
    class ReductionCloser;
    class TreeCloseImpl;
    class TreeClose;
    struct CloseInfo;

    // legion_utilities.h
    class Serializer;
    class Deserializer;
    template<typename T> class Fraction;
    template<typename T, unsigned int MAX, 
             unsigned SHIFT, unsigned MASK> class BitMask;
    template<typename T, unsigned int MAX,
             unsigned SHIFT, unsigned MASK> class TLBitMask;
#ifdef __SSE2__
    template<unsigned int MAX> class SSEBitMask;
    template<unsigned int MAX> class SSETLBitMask;
#endif
#ifdef __AVX__
    template<unsigned int MAX> class AVXBitMask;
    template<unsigned int MAX> class AVXTLBitMask;
#endif
    template<typename T, unsigned LOG2MAX> class BitPermutation;

    // legion_logging.h
    class TreeStateLogger;

    typedef LowLevel::Machine Machine;
    typedef LowLevel::Domain Domain;
    typedef LowLevel::DomainPoint DomainPoint;
    typedef LowLevel::IndexSpace IndexSpace;
    typedef LowLevel::IndexSpaceAllocator IndexSpaceAllocator;
    typedef LowLevel::RegionInstance PhysicalInstance;
    typedef LowLevel::Memory Memory;
    typedef LowLevel::Processor Processor;
    typedef LowLevel::Event Event;
    typedef LowLevel::UserEvent UserEvent;
    typedef LowLevel::Reservation Reservation;
    typedef LowLevel::Barrier Barrier;
    typedef LowLevel::ReductionOpID ReductionOpID;
    typedef LowLevel::ReductionOpUntyped ReductionOp;
    typedef LowLevel::Machine::ProcessorMemoryAffinity ProcessorMemoryAffinity;
    typedef LowLevel::Machine::MemoryMemoryAffinity MemoryMemoryAffinity;
    typedef LowLevel::ElementMask::Enumerator Enumerator;
    typedef int TaskPriority;
    typedef unsigned int Color;
    typedef unsigned int IndexPartition;
    typedef unsigned int FieldID;
    typedef unsigned int TraceID;
    typedef unsigned int MapperID;
    typedef unsigned int ContextID;
    typedef unsigned int InstanceID;
    typedef unsigned int FieldSpaceID;
    typedef unsigned int GenerationID;
    typedef unsigned int TypeHandle;
    typedef unsigned int ProjectionID;
    typedef unsigned int RegionTreeID;
    typedef unsigned int DistributedID;
    typedef unsigned int AddressSpaceID;
    typedef unsigned int TunableID;
    typedef unsigned long MappingTagID;
    typedef unsigned long VariantID;
    typedef unsigned long long UniqueID;
    typedef unsigned long long VersionID;
    typedef Processor::TaskFuncID TaskID;
    typedef SingleTask* Context;
    typedef std::map<Color,ColoredPoints<ptr_t> > Coloring;
    typedef std::map<Color,Domain> DomainColoring;
    typedef std::map<Color,std::set<Domain> > MultiDomainColoring;
    typedef void (*RegistrationCallbackFnptr)(Machine *machine, 
        HighLevelRuntime *rt, const std::set<Processor> &local_procs);
    typedef LogicalRegion (*RegionProjectionFnptr)(LogicalRegion parent, 
        const DomainPoint&, HighLevelRuntime *rt);
    typedef LogicalRegion (*PartitionProjectionFnptr)(LogicalPartition parent, 
        const DomainPoint&, HighLevelRuntime *rt);
    typedef bool (*PredicateFnptr)(const void*, size_t, 
        const std::vector<Future> futures);
    typedef std::map<ProjectionID,RegionProjectionFnptr> 
      RegionProjectionTable;
    typedef std::map<ProjectionID,PartitionProjectionFnptr> 
      PartitionProjectionTable;
    typedef void (*LowLevelFnptr)(const void*,size_t,Processor);
    typedef void (*InlineFnptr)(const Task*,const std::vector<PhysicalRegion>&,
      Context,HighLevelRuntime*,void*&,size_t&);
    // A little bit of logic here to figure out the 
    // kind of bit mask to use for FieldMask
    // Disable the use of AVX field masks for now since GCC doesn't
    // know how to properly align them on the stack. If you want
    // to try it, you can build with -DDYNAMIC_FIELD_MASKS which
    // will cause all the AVXFieldMasks to allocate their own
    // aligned backing store on the heap.  While correct, this
    // will disable many compiler optimizations due to GCC and
    // other C compilers being awful at alias analysis.
#if defined(DYNAMIC_FIELD_MASKS) && defined(__AVX__)
#if (MAX_FIELDS > 256)
    typedef AVXTLBitMask<MAX_FIELDS> FieldMask;
#elif (MAX_FIELDS > 128)
    typedef AVXBitMask<MAX_FIELDS> FieldMask;
#elif (MAX_FIELDS > 64)
    typedef SSEBitMask<MAX_FIELDS> FieldMask;
#else
    typedef BitMask<FIELD_TYPE,MAX_FIELDS,FIELD_SHIFT,FIELD_MASK> FieldMask;
#endif
#elif defined(__SSE2__)
#if (MAX_FIELDS > 128)
    typedef SSETLBitMask<MAX_FIELDS> FieldMask;
#elif (MAX_FIELDS > 64)
    typedef SSEBitMask<MAX_FIELDS> FieldMask;
#else
    typedef BitMask<FIELD_TYPE,MAX_FIELDS,FIELD_SHIFT,FIELD_MASK> FieldMask;
#endif
#else
#if (MAX_FIELDS > 64)
    typedef TLBitMask<FIELD_TYPE,MAX_FIELDS,FIELD_SHIFT,FIELD_MASK> FieldMask;
#else
    typedef BitMask<FIELD_TYPE,MAX_FIELDS,FIELD_SHIFT,FIELD_MASK> FieldMask;
#endif
#endif
    typedef BitPermutation<FieldMask,FIELD_LOG2> FieldPermutation;
    typedef Fraction<unsigned long> InstFrac;

#define FRIEND_ALL_RUNTIME_CLASSES                \
    friend class HighLevelRuntime;                \
    friend class Runtime;                         \
    friend class FuturePredicate;                 \
    friend class NotPredicate;                    \
    friend class AndPredicate;                    \
    friend class OrPredicate;                     \
    friend class ProcessorManager;                \
    friend class Operation;                       \
    friend class SpeculativeOp;                   \
    friend class MapOp;                           \
    friend class CopyOp;                          \
    friend class FenceOp;                         \
    friend class FutureOp;                        \
    friend class FuturePredOp;                    \
    friend class DeletionOp;                      \
    friend class CloseOp;                         \
    friend class AcquireOp;                       \
    friend class ReleaseOp;                       \
    friend class NotPredOp;                       \
    friend class AndPredOp;                       \
    friend class OrPredOp;                        \
    friend class MustEpochOp;                     \
    friend class TaskOp;                          \
    friend class SingleTask;                      \
    friend class MultiTask;                       \
    friend class IndividualTask;                  \
    friend class PointTask;                       \
    friend class IndexTask;                       \
    friend class SliceTask;                       \
    friend class RegionTreeForest;                \
    friend class IndexSpaceNode;                  \
    friend class IndexPartNode;                   \
    friend class FieldSpaceNode;                  \
    friend class RegionTreeNode;                  \
    friend class RegionNode;                      \
    friend class PartitionNode;                   \
    friend class LogicalView;                     \
    friend class InstanceView;                    \
    friend class ReductionView;                   \
    friend class MaterializedView;                \
    friend class CompositeView;                   \
    friend class CompositeNode;                   \
    friend class PhysicalManager;                 \
    friend class InstanceManager;                 \
    friend class ReductionManager;                \
    friend class ListReductionManager;            \
    friend class FoldReductionManager;            \
    friend class TreeStateLogger;                 \
    friend class BindingLib::Utility;

    // Timing events
    enum {
#ifdef PRECISE_HIGH_LEVEL_TIMING
      TIME_HIGH_LEVEL_CREATE_REGION = 100,
      TIME_HIGH_LEVEL_DESTROY_REGION = 101,
      TIME_HIGH_LEVEL_SMASH_REGION = 102
      TIME_HIGH_LEVEL_JOIN_REGION = 103
      TIME_HIGH_LEVEL_CREATE_PARTITION = 104,
      TIME_HIGH_LEVEL_DESTROY_PARTITION = 105,
      TIME_HIGH_LEVEL_ENQUEUE_TASKS = 106,
      TIME_HIGH_LEVEL_STEAL_REQUEST = 107,
      TIME_HIGH_LEVEL_CHILDREN_MAPPED = 108,
      TIME_HIGH_LEVEL_FINISH_TASK = 109,
      TIME_HIGH_LEVEL_NOTIFY_START = 110,
      TIME_HIGH_LEVEL_NOTIFY_MAPPED = 111,
      TIME_HIGH_LEVEL_NOTIFY_FINISH = 112,
      TIME_HIGH_LEVEL_EXECUTE_TASK = 113,
      TIME_HIGH_LEVEL_SCHEDULER = 114,
      TIME_HIGH_LEVEL_ISSUE_STEAL = 115,
      TIME_HIGH_LEVEL_GET_SUBREGION = 116,
      TIME_HIGH_LEVEL_INLINE_MAP = 117,
      TIME_HIGH_LEVEL_CREATE_INDEX_SPACE = 118,
      TIME_HIGH_LEVEL_DESTROY_INDEX_SPACE = 119,
      TIME_HIGH_LEVEL_CREATE_INDEX_PARTITION = 120,
      TIME_HIGH_LEVEL_DESTROY_INDEX_PARTITION = 121,
      TIME_HIGH_LEVEL_GET_INDEX_PARTITION = 122,
      TIME_HIGH_LEVEL_GET_INDEX_SUBSPACE = 123,
      TIME_HIGH_LEVEL_CREATE_FIELD_SPACE = 124,
      TIME_HIGH_LEVEL_DESTROY_FIELD_SPACE = 125,
      TIME_HIGH_LEVEL_GET_LOGICAL_PARTITION = 126,
      TIME_HIGH_LEVEL_GET_LOGICAL_SUBREGION = 127,
      TIME_HIGH_LEVEL_ALLOCATE_FIELD = 128,
      TIME_HIGH_LEVEL_FREE_FIELD = 129,
#else
      TIME_HIGH_LEVEL_CREATE_REGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_DESTROY_REGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_SMASH_REGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_JOIN_REGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_CREATE_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_DESTROY_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_ENQUEUE_TASKS = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_STEAL_REQUEST = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_CHILDREN_MAPPED = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_FINISH_TASK = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_NOTIFY_START = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_NOTIFY_MAPPED = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_NOTIFY_FINISH = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_EXECUTE_TASK = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_SCHEDULER = TIME_HIGH_LEVEL,
      TIME_HIGH_LEVEL_ISSUE_STEAL = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_GET_SUBREGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_INLINE_MAP = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_CREATE_INDEX_SPACE = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_DESTROY_INDEX_SPACE = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_CREATE_INDEX_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_DESTROY_INDEX_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_GET_INDEX_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_GET_INDEX_SUBSPACE = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_CREATE_FIELD_SPACE = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_DESTROY_FIELD_SPACE = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_GET_LOGICAL_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_GET_LOGICAL_SUBREGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_ALLOCATE_FIELD = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_FREE_FIELD = TIME_HIGH_LEVEL, 
#endif
    };


  }; // HighLevel namespace
}; // LegionRuntime namespace

#endif // __LEGION_TYPES_H__
