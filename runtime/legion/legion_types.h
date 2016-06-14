/* Copyright 2016 Stanford University, NVIDIA Corporation
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

/**
 * \file legion_types.h
 */

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

#include "legion_config.h"
#include "legion_template_help.h"

// Make sure we have the appropriate defines in place for including realm
#define REALM_USE_LEGION_LAYOUT_CONSTRAINTS
#include "realm.h"

namespace BindingLib { class Utility; } // BindingLib namespace

namespace Legion {

  typedef ::legion_error_t LegionErrorType;
  typedef ::legion_privilege_mode_t PrivilegeMode;
  typedef ::legion_allocate_mode_t AllocateMode;
  typedef ::legion_coherence_property_t CoherenceProperty;
  typedef ::legion_region_flags_t RegionFlags;
  typedef ::legion_handle_type_t HandleType;
  typedef ::legion_partition_kind_t PartitionKind;
  typedef ::legion_dependence_type_t DependenceType;
  typedef ::legion_index_space_kind_t IndexSpaceKind;
  typedef ::legion_file_mode_t LegionFileMode;
  typedef ::legion_execution_constraint_t ExecutionConstraintKind;
  typedef ::legion_layout_constraint_t LayoutConstraintKind;
  typedef ::legion_equality_kind_t EqualityKind;
  typedef ::legion_dimension_kind_t DimensionKind;
  typedef ::legion_isa_kind_t ISAKind;
  typedef ::legion_resource_constraint_t ResourceKind;
  typedef ::legion_launch_constraint_t LaunchKind;
  typedef ::legion_specialized_constraint_t SpecializedKind;

  // Forward declarations for user level objects
  // legion.h
  class IndexSpace;
  class IndexPartition;
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
  struct FillLauncher;
  struct LayoutConstraintRegistrar;
  struct TaskVariantRegistrar;
  struct TaskGeneratorArguments;
  class Future;
  class FutureMap;
  class Predicate;
  class PhysicalRegion;
  class IndexIterator;
  template<typename T> struct ColoredPoints; 
  struct InputArgs;
  class ProjectionFunctor;
  class Task;
  class Copy;
  class InlineMapping;
  class Acquire;
  class Release;
  class Close;
  class Runtime;
  // For backwards compatibility
  typedef Runtime HighLevelRuntime;
  // Helper for saving instantiated template functions
  struct SerdezRedopFns;

  // Forward declarations for compiler level objects
  // legion.h
  class ColoringSerializer;
  class DomainColoringSerializer;

  // Forward declarations for wrapper tasks
  // legion.h
  class LegionTaskWrapper;
  class LegionSerialization;

  // Forward declarations for C wrapper objects
  // legion_c_util.h
  class TaskResult;
  class CObjectWrapper;

  // legion_utilities.h
  struct RegionUsage;
  class AutoLock;
  class ImmovableAutoLock;
  class ColorPoint;
  class Serializer;
  class Deserializer;
  class LgEvent; // base event type for legion
  class ApEvent; // application event
  class ApUserEvent; // application user event
  class ApBarrier; // application barrier
  class RtEvent; // runtime event
  class RtUserEvent; // runtime user event
  class RtBarrier;
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
  template<typename IT, typename DT, bool BIDIR = false> class IntegerSet;

  // legion_constraint.h
  class ISAConstraint;
  class ProcessorConstraint;
  class ResourceConstraint;
  class LaunchConstraint;
  class ColocationConstraint;
  class ExecutionConstraintSet;

  class SpecializedConstraint;
  class MemoryConstraint;
  class FieldConstraint;
  class OrderingConstraint;
  class SplittingConstraint;
  class DimensionConstraint;
  class AlignmentConstraint;
  class OffsetConstraint;
  class PointerConstraint;
  class LayoutConstraintSet;
  class TaskLayoutConstraintSet;

  namespace Mapping {
    class Mappable; 
    class PhysicalInstance;
    class MapperEvent;
    class ProfilingRequestSet;
    class Mapper;
    class MapperRuntime;
    class DefaultMapper;
    class ShimMapper;
    class TestMapper;
    class DebugMapper;
    class ReplayMapper;
  };
  
  namespace Internal {

    enum OpenState {
      NOT_OPEN            = 0,
      OPEN_READ_ONLY      = 1,
      OPEN_READ_WRITE     = 2, // unknown dirty information below
      OPEN_SINGLE_REDUCE  = 3, // only one open child with reductions below
      OPEN_MULTI_REDUCE   = 4, // multiple open children with same reduction
    }; 

    // redop IDs - none used in HLR right now, but 0 isn't allowed
    enum {
      REDOP_ID_AVAILABLE    = 1,
    };

    // Runtime task numbering 
    enum {
      INIT_TASK_ID            = Realm::Processor::TASK_ID_PROCESSOR_INIT,
      SHUTDOWN_TASK_ID        = Realm::Processor::TASK_ID_PROCESSOR_SHUTDOWN,
      HLR_TASK_ID             = Realm::Processor::TASK_ID_FIRST_AVAILABLE,
      HLR_LEGION_PROFILING_ID = Realm::Processor::TASK_ID_FIRST_AVAILABLE+1,
      HLR_MAPPER_PROFILING_ID = Realm::Processor::TASK_ID_FIRST_AVAILABLE+2,
      HLR_LAUNCH_TOP_LEVEL_ID = Realm::Processor::TASK_ID_FIRST_AVAILABLE+3,
      TASK_ID_AVAILABLE       = Realm::Processor::TASK_ID_FIRST_AVAILABLE+4,
    };

    // Enumeration of high-level runtime tasks
    enum HLRTaskID {
      HLR_SCHEDULER_ID,
      HLR_POST_END_ID,
      HLR_DEFERRED_MAPPING_TRIGGER_ID,
      HLR_DEFERRED_RESOLUTION_TRIGGER_ID,
      HLR_DEFERRED_EXECUTION_TRIGGER_ID,
      HLR_DEFERRED_COMMIT_TRIGGER_ID,
      HLR_DEFERRED_POST_MAPPED_ID,
      HLR_DEFERRED_EXECUTE_ID,
      HLR_DEFERRED_COMPLETE_ID,
      HLR_DEFERRED_COMMIT_ID,
      HLR_RECLAIM_LOCAL_FIELD_ID,
      HLR_DEFERRED_COLLECT_ID,
      HLR_TRIGGER_DEPENDENCE_ID,
      HLR_TRIGGER_COMPLETE_ID,
      HLR_TRIGGER_OP_ID,
      HLR_TRIGGER_TASK_ID,
      HLR_DEFERRED_RECYCLE_ID,
      HLR_DEFERRED_SLICE_ID,
      HLR_MUST_INDIV_ID,
      HLR_MUST_INDEX_ID,
      HLR_MUST_MAP_ID,
      HLR_MUST_DIST_ID,
      HLR_MUST_LAUNCH_ID,
      HLR_DEFERRED_FUTURE_SET_ID,
      HLR_DEFERRED_FUTURE_MAP_SET_ID,
      HLR_RESOLVE_FUTURE_PRED_ID,
      HLR_MPI_RANK_ID,
      HLR_CONTRIBUTE_COLLECTIVE_ID,
      HLR_STATE_ANALYSIS_ID,
      HLR_MAPPER_TASK_ID,
      HLR_DISJOINTNESS_TASK_ID,
      HLR_PART_INDEPENDENCE_TASK_ID,
      HLR_SPACE_INDEPENDENCE_TASK_ID,
      HLR_PENDING_CHILD_TASK_ID,
      HLR_DECREMENT_PENDING_TASK_ID,
      HLR_SEND_VERSION_STATE_TASK_ID,
      HLR_ADD_TO_DEP_QUEUE_TASK_ID,
      HLR_WINDOW_WAIT_TASK_ID,
      HLR_ISSUE_FRAME_TASK_ID,
      HLR_CONTINUATION_TASK_ID,
      HLR_MAPPER_CONTINUATION_TASK_ID,
      HLR_FINISH_MAPPER_CONTINUATION_TASK_ID,
      HLR_TASK_IMPL_SEMANTIC_INFO_REQ_TASK_ID,
      HLR_INDEX_SPACE_SEMANTIC_INFO_REQ_TASK_ID,
      HLR_INDEX_PART_SEMANTIC_INFO_REQ_TASK_ID,
      HLR_FIELD_SPACE_SEMANTIC_INFO_REQ_TASK_ID,
      HLR_FIELD_SEMANTIC_INFO_REQ_TASK_ID,
      HLR_REGION_SEMANTIC_INFO_REQ_TASK_ID,
      HLR_PARTITION_SEMANTIC_INFO_REQ_TASK_ID,
      HLR_SELECT_TUNABLE_TASK_ID,
      HLR_DEFERRED_ENQUEUE_TASK_ID,
      HLR_DEFER_MAPPER_MESSAGE_TASK_ID,
      HLR_DEFER_COMPOSITE_HANDLE_TASK_ID,
      HLR_DEFER_COMPOSITE_NODE_TASK_ID,
      HLR_DEFER_CREATE_COMPOSITE_VIEW_TASK_ID,
      HLR_UPDATE_VIEW_REFERENCES_TASK_ID,
      HLR_REMOVE_VERSION_STATE_REF_TASK_ID,
      HLR_MESSAGE_ID, // These four must be last (see issue_runtime_meta_task)
      HLR_SHUTDOWN_ATTEMPT_TASK_ID,
      HLR_SHUTDOWN_NOTIFICATION_TASK_ID,
      HLR_SHUTDOWN_RESPONSE_TASK_ID,
      HLR_LAST_TASK_ID, // This one should always be last
    };

    // Make this a macro so we can keep it close to 
    // declaration of the task IDs themselves
#define HLR_TASK_DESCRIPTIONS(name)                               \
      const char *name[HLR_LAST_TASK_ID] = {                      \
        "Scheduler",                                              \
        "Post-Task Execution",                                    \
        "Deferred Mapping Trigger",                               \
        "Deferred Resolution Trigger",                            \
        "Deferred Execution Trigger",                             \
        "Deferred Commit Trigger",                                \
        "Deferred Post Mapped",                                   \
        "Deferred Execute",                                       \
        "Deferred Complete",                                      \
        "Deferred Commit",                                        \
        "Reclaim Local Field",                                    \
        "Garbage Collection",                                     \
        "Logical Dependence Analysis",                            \
        "Trigger Complete",                                       \
        "Operation Physical Dependence Analysis",                 \
        "Task Physical Dependence Analysis",                      \
        "Deferred Recycle",                                       \
        "Deferred Slice",                                         \
        "Must Individual Task Dependence Analysis",               \
        "Must Index Task Dependence Analysis",                    \
        "Must Task Physical Dependence Analysis",                 \
        "Must Task Distribution",                                 \
        "Must Task Launch",                                       \
        "Deferred Future Set",                                    \
        "Deferred Future Map Set",                                \
        "Resolve Future Predicate",                               \
        "Update MPI Rank Info",                                   \
        "Contribute Collective",                                  \
        "State Analaysis",                                        \
        "Mapper Task",                                            \
        "Disjointness Test",                                      \
        "Partition Independence Test",                            \
        "Index Space Independence Test",                          \
        "Remove Pending Child",                                   \
        "Decrement Pending Task",                                 \
        "Send Version State",                                     \
        "Add to Dependence Queue",                                \
        "Window Wait",                                            \
        "Issue Frame",                                            \
        "Legion Continuation",                                    \
        "Mapper Continuation",                                    \
        "Finish Mapper Continuation",                             \
        "Task Impl Semantic Request",                             \
        "Index Space Semantic Request",                           \
        "Index Partition Semantic Request",                       \
        "Field Space Semantic Request",                           \
        "Field Semantic Request",                                 \
        "Region Semantic Request",                                \
        "Partition Semantic Request",                             \
        "Select Tunable",                                         \
        "Deferred Task Enqueue",                                  \
        "Deferred Composite Handle",                              \
        "Deferred Composite Node Ref",                            \
        "Deferred Composite View Creation",                       \
        "Deferred Mapper Message",                                \
        "Update View References for Version State",               \
        "Deferred Remove Version State Valid Ref",                \
        "Remote Message",                                         \
        "Shutdown Attempt",                                       \
        "Shutdown Notification",                                  \
        "Shutdown Response",                                      \
      };

    enum MappingCallKind {
      GET_MAPPER_NAME_CALL,
      GET_MAPER_SYNC_MODEL_CALL,
      SELECT_TASK_OPTIONS_CALL,
      PREMAP_TASK_CALL,
      SLICE_TASK_CALL,
      MAP_TASK_CALL,
      SELECT_VARIANT_CALL,
      POSTMAP_TASK_CALL,
      TASK_SELECT_SOURCES_CALL,
      TASK_SPECULATE_CALL,
      TASK_REPORT_PROFILING_CALL,
      MAP_INLINE_CALL,
      INLINE_SELECT_SOURCES_CALL,
      INLINE_REPORT_PROFILING_CALL,
      MAP_COPY_CALL,
      COPY_SELECT_SOURCES_CALL,
      COPY_SPECULATE_CALL,
      COPY_REPORT_PROFILING_CALL,
      MAP_CLOSE_CALL,
      CLOSE_SELECT_SOURCES_CALL,
      CLOSE_REPORT_PROFILING_CALL,
      MAP_ACQUIRE_CALL,
      ACQUIRE_SPECULATE_CALL,
      ACQUIRE_REPORT_PROFILING_CALL,
      MAP_RELEASE_CALL,
      RELEASE_SELECT_SOURCES_CALL,
      RELEASE_SPECULATE_CALL,
      RELEASE_REPORT_PROFILING_CALL,
      CONFIGURE_CONTEXT_CALL,
      SELECT_TUNABLE_VALUE_CALL,
      MAP_MUST_EPOCH_CALL,
      MAP_DATAFLOW_GRAPH_CALL,
      SELECT_TASKS_TO_MAP_CALL,
      SELECT_STEAL_TARGETS_CALL,
      PERMIT_STEAL_REQUEST_CALL,
      HANDLE_MESSAGE_CALL,
      HANDLE_TASK_RESULT_CALL,
      LAST_MAPPER_CALL,
    };

#define MAPPER_CALL_NAMES(name)                     \
    const char *name[LAST_MAPPER_CALL] = {          \
      "get_mapper_name",                            \
      "get_mapper_sync_model",                      \
      "select_task_options",                        \
      "premap_task",                                \
      "slice_task",                                 \
      "map_task",                                   \
      "select_task_variant",                        \
      "postmap_task",                               \
      "select_task_sources",                        \
      "speculate (for task)",                       \
      "report profiling (for task)",                \
      "map_inline",                                 \
      "select_inline_sources",                      \
      "report profiling (for inline)",              \
      "map_copy",                                   \
      "select_copy_sources",                        \
      "speculate (for copy)",                       \
      "report_profiling (for copy)",                \
      "map_close",                                  \
      "select_close_sources",                       \
      "report_profiling (for close)",               \
      "map_acquire",                                \
      "speculate (for acquire)",                    \
      "report_profiling (for acquire)",             \
      "map_release",                                \
      "select_release_sources",                     \
      "speculate (for release)",                    \
      "report_profiling (for release)",             \
      "configure_context",                          \
      "select_tunable_value",                       \
      "map_must_epoch",                             \
      "map_dataflow_graph",                         \
      "select_tasks_to_map",                        \
      "select_steal_targets",                       \
      "permit_steal_request",                       \
      "handle_message",                             \
      "handle_task_result",                         \
    }

    enum HLRPriority {
      HLR_THROUGHPUT_PRIORITY = 0, // don't care so much
      HLR_LATENCY_PRIORITY = 1, // care some but not too much
      HLR_RESOURCE_PRIORITY = 2, // this needs to be first
    };

    enum VirtualChannelKind {
      DEFAULT_VIRTUAL_CHANNEL = 0,
      INDEX_SPACE_VIRTUAL_CHANNEL = 1,
      FIELD_SPACE_VIRTUAL_CHANNEL = 2,
      LOGICAL_TREE_VIRTUAL_CHANNEL = 3,
      MAPPER_VIRTUAL_CHANNEL = 4,
      SEMANTIC_INFO_VIRTUAL_CHANNEL = 5,
      LAYOUT_CONSTRAINT_VIRTUAL_CHANNEL = 6,
      CONTEXT_VIRTUAL_CHANNEL = 7,
      MANAGER_VIRTUAL_CHANNEL = 8,
      VIEW_VIRTUAL_CHANNEL = 9,
      UPDATE_VIRTUAL_CHANNEL = 10,
      VARIANT_VIRTUAL_CHANNEL = 11,
      MAX_NUM_VIRTUAL_CHANNELS = 12, // this one must be last
    };

    enum MessageKind {
      TASK_MESSAGE,
      STEAL_MESSAGE,
      ADVERTISEMENT_MESSAGE,
      SEND_INDEX_SPACE_NODE,
      SEND_INDEX_SPACE_REQUEST,
      SEND_INDEX_SPACE_RETURN,
      SEND_INDEX_SPACE_CHILD_REQUEST,
      SEND_INDEX_SPACE_CHILD_RESPONSE,
      SEND_INDEX_SPACE_COLORS_REQUEST,
      SEND_INDEX_SPACE_COLORS_RESPONSE,
      SEND_INDEX_PARTITION_NOTIFICATION,
      SEND_INDEX_PARTITION_NODE,
      SEND_INDEX_PARTITION_REQUEST,
      SEND_INDEX_PARTITION_RETURN,
      SEND_INDEX_PARTITION_CHILD_REQUEST,
      SEND_INDEX_PARTITION_CHILD_RESPONSE,
      SEND_FIELD_SPACE_NODE,
      SEND_FIELD_SPACE_REQUEST,
      SEND_FIELD_SPACE_RETURN,
      SEND_FIELD_ALLOC_REQUEST,
      SEND_FIELD_ALLOC_NOTIFICATION,
      SEND_FIELD_SPACE_TOP_ALLOC,
      SEND_FIELD_FREE,
      SEND_TOP_LEVEL_REGION_REQUEST,
      SEND_TOP_LEVEL_REGION_RETURN,
      SEND_LOGICAL_REGION_NODE,
      INDEX_SPACE_DESTRUCTION_MESSAGE,
      INDEX_PARTITION_DESTRUCTION_MESSAGE,
      FIELD_SPACE_DESTRUCTION_MESSAGE,
      LOGICAL_REGION_DESTRUCTION_MESSAGE,
      LOGICAL_PARTITION_DESTRUCTION_MESSAGE,
      INDIVIDUAL_REMOTE_MAPPED,
      INDIVIDUAL_REMOTE_COMPLETE,
      INDIVIDUAL_REMOTE_COMMIT,
      SLICE_REMOTE_MAPPED,
      SLICE_REMOTE_COMPLETE,
      SLICE_REMOTE_COMMIT,
      DISTRIBUTED_REMOTE_REGISTRATION,
      DISTRIBUTED_VALID_UPDATE,
      DISTRIBUTED_GC_UPDATE,
      DISTRIBUTED_RESOURCE_UPDATE,
      DISTRIBUTED_CREATE_ADD,
      DISTRIBUTED_CREATE_REMOVE,
      SEND_ATOMIC_RESERVATION_REQUEST,
      SEND_ATOMIC_RESERVATION_RESPONSE,
      SEND_MATERIALIZED_VIEW,
      SEND_COMPOSITE_VIEW,
      SEND_FILL_VIEW,
      SEND_REDUCTION_VIEW,
      SEND_INSTANCE_MANAGER,
      SEND_REDUCTION_MANAGER,
      SEND_CREATE_TOP_VIEW_REQUEST,
      SEND_CREATE_TOP_VIEW_RESPONSE,
      SEND_SUBVIEW_DID_REQUEST,
      SEND_SUBVIEW_DID_RESPONSE,
      SEND_VIEW_REQUEST,
      SEND_VIEW_UPDATE_REQUEST,
      SEND_VIEW_UPDATE_RESPONSE,
      SEND_VIEW_REMOTE_UPDATE,
      SEND_VIEW_REMOTE_INVALIDATE,
      SEND_MANAGER_REQUEST,
      SEND_FUTURE_RESULT,
      SEND_FUTURE_SUBSCRIPTION,
      SEND_MAPPER_MESSAGE,
      SEND_MAPPER_BROADCAST,
      SEND_TASK_IMPL_SEMANTIC_REQ,
      SEND_INDEX_SPACE_SEMANTIC_REQ,
      SEND_INDEX_PARTITION_SEMANTIC_REQ,
      SEND_FIELD_SPACE_SEMANTIC_REQ,
      SEND_FIELD_SEMANTIC_REQ,
      SEND_LOGICAL_REGION_SEMANTIC_REQ,
      SEND_LOGICAL_PARTITION_SEMANTIC_REQ,
      SEND_TASK_IMPL_SEMANTIC_INFO,
      SEND_INDEX_SPACE_SEMANTIC_INFO,
      SEND_INDEX_PARTITION_SEMANTIC_INFO,
      SEND_FIELD_SPACE_SEMANTIC_INFO,
      SEND_FIELD_SEMANTIC_INFO,
      SEND_LOGICAL_REGION_SEMANTIC_INFO,
      SEND_LOGICAL_PARTITION_SEMANTIC_INFO,
      SEND_REMOTE_CONTEXT_REQUEST,
      SEND_REMOTE_CONTEXT_RESPONSE,
      SEND_REMOTE_CONTEXT_FREE,
      SEND_REMOTE_CONVERT_VIRTUAL,
      SEND_VERSION_STATE_PATH,
      SEND_VERSION_STATE_INIT,
      SEND_VERSION_STATE_REQUEST,
      SEND_VERSION_STATE_RESPONSE,
      SEND_INSTANCE_REQUEST,
      SEND_INSTANCE_RESPONSE,
      SEND_GC_PRIORITY_UPDATE,
      SEND_NEVER_GC_RESPONSE,
      SEND_ACQUIRE_REQUEST,
      SEND_ACQUIRE_RESPONSE,
      SEND_BACK_LOGICAL_STATE,
      SEND_VARIANT_REQUEST,
      SEND_VARIANT_RESPONSE,
      SEND_CONSTRAINT_REQUEST,
      SEND_CONSTRAINT_RESPONSE,
      SEND_CONSTRAINT_RELEASE,
      SEND_CONSTRAINT_REMOVAL,
      SEND_TOP_LEVEL_TASK_REQUEST,
      SEND_TOP_LEVEL_TASK_COMPLETE,
      SEND_SHUTDOWN_NOTIFICATION,
      SEND_SHUTDOWN_RESPONSE,
      LAST_SEND_KIND, // This one must be last
    };

#define HLR_MESSAGE_DESCRIPTIONS(name)                                \
      const char *name[LAST_SEND_KIND] = {                            \
        "Task Message",                                               \
        "Steal Message",                                              \
        "Advertisement Message",                                      \
        "Send Index Space Node",                                      \
        "Send Index Space Request",                                   \
        "Send Index Space Return",                                    \
        "Send Index Space Child Request",                             \
        "Send Index Space Child Response",                            \
        "Send Index Space Colors Request",                            \
        "Send Index Space Colors Response",                           \
        "Send Index Partition Notification",                          \
        "Send Index Partition Node",                                  \
        "Send Index Partition Request",                               \
        "Send Index Partition Return",                                \
        "Send Index Partition Child Request",                         \
        "Send Index Partition Child Response",                        \
        "Send Field Space Node",                                      \
        "Send Field Space Request",                                   \
        "Send Field Space Return",                                    \
        "Send Field Alloc Request",                                   \
        "Send Field Alloc Notification",                              \
        "Send Field Space Top Alloc",                                 \
        "Send Field Free",                                            \
        "Send Top Level Region Request",                              \
        "Send Top Level Region Return",                               \
        "Send Logical Region Node",                                   \
        "Index Space Destruction",                                    \
        "Index Partition Destruction",                                \
        "Field Space Destruction",                                    \
        "Logical Region Destruction",                                 \
        "Logical Partition Destruction",                              \
        "Individual Remote Mapped",                                   \
        "Individual Remote Complete",                                 \
        "Individual Remote Commit",                                   \
        "Slice Remote Mapped",                                        \
        "Slice Remote Complete",                                      \
        "Slice Remote Commit",                                        \
        "Distributed Remote Registration",                            \
        "Distributed Valid Update",                                   \
        "Distributed GC Update",                                      \
        "Distributed Resource Update",                                \
        "Distributed Create Add",                                     \
        "Distributed Create Remove",                                  \
        "Send Atomic Reservation Request",                            \
        "Send Atomic Reservation Response",                           \
        "Send Materialized View",                                     \
        "Send Composite View",                                        \
        "Send Fill View",                                             \
        "Send Reduction View",                                        \
        "Send Instance Manager",                                      \
        "Send Reduction Manager",                                     \
        "Send Create Top View Request",                               \
        "Send Create Top View Response",                              \
        "Send Subview DID Request",                                   \
        "Send Subview DID Response",                                  \
        "Send View Request",                                          \
        "Send View Update Request",                                   \
        "Send View Update Response",                                  \
        "Send View Remote Update",                                    \
        "Send View Remote Invalidate",                                \
        "Send Manager Request",                                       \
        "Send Future Result",                                         \
        "Send Future Subscription",                                   \
        "Send Mapper Message",                                        \
        "Send Mapper Broadcast",                                      \
        "Send Task Impl Semantic Req",                                \
        "Send Index Space Semantic Req",                              \
        "Send Index Partition Semantic Req",                          \
        "Send Field Space Semantic Req",                              \
        "Send Field Semantic Req",                                    \
        "Send Logical Region Semantic Req",                           \
        "Send Logical Partition Semantic Req",                        \
        "Send Task Impl Semantic Info",                               \
        "Send Index Space Semantic Info",                             \
        "Send Index Partition Semantic Info",                         \
        "Send Field Space Semantic Info",                             \
        "Send Field Semantic Info",                                   \
        "Send Logical Region Semantic Info",                          \
        "Send Logical Partition Semantic Info",                       \
        "Send Remote Context Request",                                \
        "Send Remote Context Response",                               \
        "Send Remote Context Free",                                   \
        "Send Remote Convert Virtual Instances",                      \
        "Send Version State Path",                                    \
        "Send Version State Init",                                    \
        "Send Version State Request",                                 \
        "Send Version State Response",                                \
        "Send Instance Request",                                      \
        "Send Instance Response",                                     \
        "Send GC Priority Update",                                    \
        "Send Never GC Response",                                     \
        "Send Acquire Request",                                       \
        "Send Acquire Response",                                      \
        "Send Back Logical State",                                    \
        "Send Task Variant Request",                                  \
        "Send Task Variant Response",                                 \
        "Send Constraint Request",                                    \
        "Send Constraint Response",                                   \
        "Send Constraint Release",                                    \
        "Send Constraint Removal",                                    \
        "Top Level Task Request",                                     \
        "Top Level Task Complete",                                    \
        "Send Shutdown Notification",                                 \
        "Send Shutdown Response",                                     \
      };

    enum RuntimeCallKind {
      PACK_BASE_TASK_CALL, 
      UNPACK_BASE_TASK_CALL,
      TASK_PRIVILEGE_CHECK_CALL,
      CLONE_TASK_CALL,
      COMPUTE_POINT_REQUIREMENTS_CALL,
      EARLY_MAP_REGIONS_CALL,
      RECORD_ALIASED_REQUIREMENTS_CALL,
      ACTIVATE_SINGLE_CALL,
      DEACTIVATE_SINGLE_CALL,
      SELECT_INLINE_VARIANT_CALL,
      INLINE_CHILD_TASK_CALL,
      PACK_SINGLE_TASK_CALL,
      UNPACK_SINGLE_TASK_CALL,
      PACK_REMOTE_CONTEXT_CALL,
      HAS_CONFLICTING_INTERNAL_CALL,
      FIND_CONFLICTING_CALL,
      FIND_CONFLICTING_INTERNAL_CALL,
      CHECK_REGION_DEPENDENCE_CALL,
      FIND_PARENT_REGION_REQ_CALL,
      FIND_PARENT_REGION_CALL,
      CHECK_PRIVILEGE_CALL,
      TRIGGER_SINGLE_CALL,
      INITIALIZE_MAP_TASK_CALL,
      FINALIZE_MAP_TASK_CALL,
      VALIDATE_VARIANT_SELECTION_CALL,
      MAP_ALL_REGIONS_CALL,
      INITIALIZE_REGION_TREE_CONTEXTS_CALL,
      INVALIDATE_REGION_TREE_CONTEXTS_CALL,
      CREATE_INSTANCE_TOP_VIEW_CALL,
      CONVERT_VIRTUAL_INSTANCE_TOP_VIEW_CALL,
      LAUNCH_TASK_CALL,
      ACTIVATE_MULTI_CALL,
      DEACTIVATE_MULTI_CALL,
      SLICE_INDEX_SPACE_CALL,
      CLONE_MULTI_CALL,
      MULTI_TRIGGER_EXECUTION_CALL,
      PACK_MULTI_CALL,
      UNPACK_MULTI_CALL,
      ACTIVATE_INDIVIDUAL_CALL,
      DEACTIVATE_INDIVIDUAL_CALL,
      INDIVIDUAL_REMOTE_STATE_ANALYSIS_CALL,
      INDIVIDUAL_PERFORM_MAPPING_CALL,
      INDIVIDUAL_RETURN_VIRTUAL_CALL,
      INDIVIDUAL_TRIGGER_COMPLETE_CALL,
      INDIVIDUAL_TRIGGER_COMMIT_CALL,
      INDIVIDUAL_POST_MAPPED_CALL,
      INDIVIDUAL_PACK_TASK_CALL,
      INDIVIDUAL_UNPACK_TASK_CALL,
      INDIVIDUAL_PACK_REMOTE_COMPLETE_CALL,
      INDIVIDUAL_UNPACK_REMOTE_COMPLETE_CALL,
      POINT_ACTIVATE_CALL,
      POINT_DEACTIVATE_CALL,
      POINT_TASK_COMPLETE_CALL,
      POINT_TASK_COMMIT_CALL,
      POINT_PACK_TASK_CALL,
      POINT_UNPACK_TASK_CALL,
      POINT_TASK_POST_MAPPED_CALL,
      REMOTE_TASK_ACTIVATE_CALL,
      REMOTE_TASK_DEACTIVATE_CALL,
      REMOTE_UNPACK_CONTEXT_CALL,
      INDEX_ACTIVATE_CALL,
      INDEX_DEACTIVATE_CALL,
      INDEX_REMOTE_STATE_ANALYSIS_CALL,
      INDEX_COMPUTE_FAT_PATH_CALL,
      INDEX_EARLY_MAP_TASK_CALL,
      INDEX_DISTRIBUTE_CALL,
      INDEX_PERFORM_MAPPING_CALL,
      INDEX_COMPLETE_CALL,
      INDEX_COMMIT_CALL,
      INDEX_PERFORM_INLINING_CALL,
      INDEX_CLONE_AS_SLICE_CALL,
      INDEX_HANDLE_FUTURE,
      INDEX_ENUMERATE_POINTS_CALL,
      INDEX_RETURN_SLICE_MAPPED_CALL,
      INDEX_RETURN_SLICE_COMPLETE_CALL,
      INDEX_RETURN_SLICE_COMMIT_CALL,
      SLICE_ACTIVATE_CALL,
      SLICE_DEACTIVATE_CALL,
      SLICE_REMOTE_STATE_ANALYSIS_CALL,
      SLICE_PREWALK_CALL,
      SLICE_APPLY_VERSION_INFO_CALL,
      SLICE_DISTRIBUTE_CALL,
      SLICE_PERFORM_MAPPING_CALL,
      SLICE_LAUNCH_CALL,
      SLICE_MAP_AND_LAUNCH_CALL,
      SLICE_PACK_TASK_CALL,
      SLICE_UNPACK_TASK_CALL,
      SLICE_CLONE_AS_SLICE_CALL,
      SLICE_HANDLE_FUTURE_CALL,
      SLICE_CLONE_AS_POINT_CALL,
      SLICE_ENUMERATE_POINTS_CALL,
      SLICE_RETURN_VIRTUAL_CALL,
      SLICE_MAPPED_CALL,
      SLICE_COMPLETE_CALL,
      SLICE_COMMIT_CALL,
      REALM_SPAWN_META_CALL,
      REALM_SPAWN_TASK_CALL,
      REALM_CREATE_INSTANCE_CALL,
      REALM_ISSUE_COPY_CALL,
      REALM_ISSUE_FILL_CALL,
      REGION_TREE_LOGICAL_ANALYSIS_CALL,
      REGION_TREE_LOGICAL_FENCE_CALL,
      REGION_TREE_INITIALIZE_CONTEXT_CALL,
      REGION_TREE_INVALIDATE_CONTEXT_CALL,
      REGION_TREE_PHYSICAL_TRAVERSE_CALL,
      REGION_TREE_PHYSICAL_TRAVERSE_AND_REGISTER_CALL,
      REGION_TREE_MAP_VIRTUAL_CALL,
      REGION_TREE_PHYSICAL_REGISTER_ONLY_CALL,
      REGION_TREE_PHYSICAL_REGISTER_USERS_CALL,
      REGION_TREE_PHYSICAL_PERFORM_CLOSE_CALL,
      REGION_TREE_PHYSICAL_CLOSE_CONTEXT_CALL,
      REGION_TREE_PHYSICAL_COPY_ACROSS_CALL,
      REGION_TREE_PHYSICAL_REDUCE_ACROSS_CALL,
      REGION_TREE_PHYSICAL_CONVERT_VIEWS_INTO_CALL,
      REGION_TREE_PHYSICAL_CONVERT_VIEWS_FROM_CALL,
      REGION_TREE_PHYSICAL_CONVERT_MAPPING_CALL,
      REGION_TREE_PHYSICAL_FILL_FIELDS_CALL,
      REGION_TREE_PHYSICAL_ATTACH_FILE_CALL,
      REGION_TREE_PHYSICAL_DETACH_FILE_CALL,
      REGION_NODE_REGISTER_LOGICAL_USER_CALL,
      REGION_NODE_OPEN_LOGICAL_NODE_CALL,
      REGION_NODE_REGISTER_LOGICAL_FAT_PATH_CALL,
      REGION_NODE_OPEN_LOGICAL_FAT_PATH_CALL,
      REGION_NODE_CLOSE_LOGICAL_NODE_CALL,
      REGION_NODE_SIPHON_LOGICAL_CHILDREN_CALL,
      REGION_NODE_PERFORM_LOGICAL_CLOSES_CALL,
      REGION_NODE_CLOSE_PHYSICAL_NODE_CALL,
      REGION_NODE_SIPHON_PHYSICAL_CHILDREN_CALL,
      REGION_NODE_CLOSE_COMPOSITE_NODE_CALL,
      REGION_NODE_SIPHON_COMPOSITE_CHILDREN_CALL,
      REGION_NODE_FIND_VALID_INSTANCE_VIEWS_CALL,
      REGION_NODE_FIND_VALID_REDUCTION_VIEWS_CALL,
      REGION_NODE_ISSUE_UPDATE_COPIES_CALL,
      REGION_NODE_SORT_COPY_INSTANCES_CALL,
      REGION_NODE_ISSUE_GROUPED_COPIES_CALL,
      REGION_NODE_ISSUE_UPDATE_REDUCTIONS_CALL,
      REGION_NODE_FLUSH_REDUCTIONS_CALL,
      REGION_NODE_MAP_VIRTUAL_CALL,
      REGION_NODE_REGISTER_REGION_CALL,
      REGION_NODE_CLOSE_STATE_CALL,
      CURRENT_STATE_RECORD_VERSION_NUMBERS_CALL,
      CURRENT_STATE_ADVANCE_VERSION_NUMBERS_CALL,
      LOGICAL_CLOSER_RECORD_VERSION_NUMBERS_CALL,
      LOGICAL_CLOSER_RECORD_TOP_VERSION_NUMBERS_CALL,
      PHYSICAL_STATE_CAPTURE_STATE_CALL,
      PHYSICAL_STATE_APPLY_PATH_ONLY_CALL,
      PHYSICAL_STATE_APPLY_STATE_CALL,
      PHYSICAL_STATE_FILTER_AND_APPLY_STATE_CALL,
      PHYSICAL_STATE_MAKE_LOCAL_CALL,
      VERSION_STATE_UPDATE_SPLIT_PREVIOUS_CALL,
      VERSION_STATE_UPDATE_SPLIT_ADVANCE_CALL,
      VERSION_STATE_UPDATE_PATH_ONLY_CALL,
      VERSION_STATE_MERGE_PATH_ONLY_CALL,
      VERSION_STATE_MERGE_PHYSICAL_STATE_CALL,
      VERSION_STATE_FILTER_AND_MERGE_PHYSICAL_STATE_CALL,
      VERSION_STATE_REQUEST_INITIAL_CALL,
      VERSION_STATE_REQUEST_FINAL_CALL,
      VERSION_STATE_SEND_STATE_CALL,
      VERSION_STATE_HANDLE_REQUEST_CALL,
      VERSION_STATE_HANDLE_RESPONSE_CALL,
      MATERIALIZED_VIEW_FIND_LOCAL_PRECONDITIONS_CALL,
      MATERIALIZED_VIEW_FIND_LOCAL_COPY_PRECONDITIONS_CALL,
      MATERIALIZED_VIEW_FILTER_PREVIOUS_USERS_CALL,
      MATERIALIZED_VIEW_FILTER_CURRENT_USERS_CALL,
      MATERIALIZED_VIEW_FILTER_LOCAL_USERS_CALL,
      COMPOSITE_VIEW_SIMPLIFY_CALL,
      COMPOSITE_VIEW_ISSUE_DEFERRED_COPIES_CALL,
      COMPOSITE_NODE_CAPTURE_PHYSICAL_STATE_CALL,
      COMPOSITE_NODE_SIMPLIFY_CALL,
      COMPOSITE_NODE_ISSUE_DEFERRED_COPIES_CALL,
      COMPOSITE_NODE_ISSUE_UPDATE_COPIES_CALL,
      COMPOSITE_NODE_ISSUE_UPDATE_REDUCTIONS_CALL,
      REDUCTION_VIEW_PERFORM_REDUCTION_CALL,
      REDUCTION_VIEW_PERFORM_DEFERRED_REDUCTION_CALL,
      REDUCTION_VIEW_PERFORM_DEFERRED_REDUCTION_ACROSS_CALL,
      REDUCTION_VIEW_FIND_COPY_PRECONDITIONS_CALL,
      REDUCTION_VIEW_FIND_USER_PRECONDITIONS_CALL,
      REDUCTION_VIEW_FILTER_LOCAL_USERS_CALL,
      LAST_RUNTIME_CALL_KIND, // This one must be last
    };

#define RUNTIME_CALL_DESCRIPTIONS(name)                               \
    const char *name[LAST_RUNTIME_CALL_KIND] = {                      \
      "Pack Base Task",                                               \
      "Unpack Base Task",                                             \
      "Task Privilege Check",                                         \
      "Clone Base Task",                                              \
      "Compute Point Requirements",                                   \
      "Early Map Regions",                                            \
      "Record Early Requirements",                                    \
      "Activate Single",                                              \
      "Deactivate Single",                                            \
      "Select Inline Variant",                                        \
      "Inline Child Task",                                            \
      "Pack Single Task",                                             \
      "Unpack Single Task",                                           \
      "Pack Remote Context",                                          \
      "Has Conflicting Internal",                                     \
      "Find Conflicting",                                             \
      "Find Conflicting Internal",                                    \
      "Check Region Dependence",                                      \
      "Find Parent Region Requirement",                               \
      "Find Parent Region",                                           \
      "Check Privilege",                                              \
      "Trigger Single",                                               \
      "Initialize Map Task",                                          \
      "Finalized Map Task",                                           \
      "Validate Variant Selection",                                   \
      "Map All Regions",                                              \
      "Initialize Region Tree Contexts",                              \
      "Invalidate Region Tree Contexts",                              \
      "Create Instance Top View",                                     \
      "Convert Virtual Instance Top View",                            \
      "Launch Task",                                                  \
      "Activate Multi",                                               \
      "Deactivate Multi",                                             \
      "Slice Index Space",                                            \
      "Clone Multi Call",                                             \
      "Multi Trigger Execution",                                      \
      "Pack Multi",                                                   \
      "Unpack Multi",                                                 \
      "Activate Individual",                                          \
      "Deactivate Individual",                                        \
      "Individual Remote State Analysis",                             \
      "Individual Perform Mapping",                                   \
      "Individual Return Virtual",                                    \
      "Individual Trigger Complete",                                  \
      "Individual Trigger Commit",                                    \
      "Individual Post Mapped",                                       \
      "Individual Pack Task",                                         \
      "Individual Unpack Task",                                       \
      "Individual Pack Remote Complete",                              \
      "Individual Unpack Remote Complete",                            \
      "Activate Point",                                               \
      "Deactivate Point",                                             \
      "Point Task Complete",                                          \
      "Point Task Commit",                                            \
      "Point Task Pack",                                              \
      "Point Task Unpack",                                            \
      "Point Task Post Mapped",                                       \
      "Remote Task Activate",                                         \
      "Remote Task Deactivate",                                       \
      "Remote Unpack Context",                                        \
      "Index Activate",                                               \
      "Index Deactivate",                                             \
      "Index Remote State Analysis",                                  \
      "Index Compute Fat Path",                                       \
      "Index Early Map Task",                                         \
      "Index Distribute",                                             \
      "Index Perform Mapping",                                        \
      "Index Complete",                                               \
      "Index Commit",                                                 \
      "Index Perform Inlining",                                       \
      "Index Clone As Slice",                                         \
      "Index Handle Future",                                          \
      "Index Enumerate Points",                                       \
      "Index Return Slice Mapped",                                    \
      "Index Return Slice Complete",                                  \
      "Index Return Slice Commit",                                    \
      "Slice Activate",                                               \
      "Slice Deactivate",                                             \
      "Slice Remote State Analysis",                                  \
      "Slice Prewalk",                                                \
      "Slice Apply Version Info",                                     \
      "Slice Distribute",                                             \
      "Slice Perform Mapping",                                        \
      "Slice Launch",                                                 \
      "Slice Map and Launch",                                         \
      "Slice Pack Task",                                              \
      "Slice Unpack Task",                                            \
      "Slice Clone As Slice",                                         \
      "Slice Handle Future",                                          \
      "Slice Cone as Point",                                          \
      "Slice Enumerate Points",                                       \
      "Slice Return Virtual",                                         \
      "Slice Mapped",                                                 \
      "Slice Complete",                                               \
      "Slice Commit",                                                 \
      "Realm Spawn Meta",                                             \
      "Realm Spawn Task",                                             \
      "Realm Create Instance",                                        \
      "Realm Issue Copy",                                             \
      "Realm Issue Fill",                                             \
      "Region Tree Logical Analysis",                                 \
      "Region Tree Logical Fence",                                    \
      "Region Tree Initialize Context",                               \
      "Region Tree Invalidate Context",                               \
      "Region Tree Physical Traverse",                                \
      "Region Tree Physical Traverse and Register",                   \
      "Region Tree Map Virtual",                                      \
      "Region Tree Physical Register Only",                           \
      "Region Tree Physical Register Users",                          \
      "Region Tree Physical Perform Close",                           \
      "Region Tree Physical Close Context",                           \
      "Region Tree Physical Copy Across",                             \
      "Region Tree Physical Reduce Across",                           \
      "Region Tree Physical Convert View Into Context",               \
      "Region Tree Physical Convert View From Context",               \
      "Region Tree Physical Convert Mapping",                         \
      "Region Tree Physical Fill Fields",                             \
      "Region Tree Physical Attach File",                             \
      "Region Tree Physical Detach File",                             \
      "Region Node Register Logical User",                            \
      "Region Node Open Logical Node",                                \
      "Region Node Register Logical Fat Path",                        \
      "Region Node Open Logical Fat Path",                            \
      "Region Node Close Logical Node",                               \
      "Region Node Siphon Logical Node",                              \
      "Region Node Perform Logical Closes",                           \
      "Region Node Close Physical Node",                              \
      "Region Node Siphon Physical Children",                         \
      "Region Node Close Composite Node",                             \
      "Region Node Siphon Composite Children",                        \
      "Region Node Find Valid Instance Views",                        \
      "Region Node Find Valid Reduction Views",                       \
      "Region Node Issue Update Copies",                              \
      "Region Node Sort Copy Instances",                              \
      "Region Node Issue Grouped Copies",                             \
      "Region Node Issue Update Reductions",                          \
      "Region Node Flush Reductions",                                 \
      "Region Node Map Virtual",                                      \
      "Region Node Register Region",                                  \
      "Region Node Close State",                                      \
      "Current State Record Verison Numbers",                         \
      "Current State Advance Version Numbers",                        \
      "Logical Closer Record Version Numbers",                        \
      "Logical Closer Record Top Version Numbers",                    \
      "Physical State Capture State",                                 \
      "Physical State Apply Path Only",                               \
      "Physical State Apply State",                                   \
      "Physical State Filter and Apply",                              \
      "Physical State Make Local",                                    \
      "Version State Update Split Previous",                          \
      "Version State Update Split Advance",                           \
      "Version State Update Path Only",                               \
      "Version State Merge Path Only",                                \
      "Version State Merge Physical State",                           \
      "Version State Filter and Merge Physical State",                \
      "Version State Request Initial",                                \
      "Version State Request Final",                                  \
      "Version State Send State",                                     \
      "Version State Handle Request",                                 \
      "Version State Handle Response",                                \
      "Materialized View Find Local Preconditions",                   \
      "Materialized View Find Local Copy Preconditions",              \
      "Materialized View Filter Previous Users",                      \
      "Materialized View Filter Current Users",                       \
      "Materialized View Filter Local Users",                         \
      "Composite View Simplify",                                      \
      "Composite View Issue Deferred Copies",                         \
      "Composite Node Capture Physical State",                        \
      "Composite Node Simplify",                                      \
      "Composite Node Issue Deferred Copies",                         \
      "Composite Node Issue Update Copies",                           \
      "Composite Node Issue Update Reductions",                       \
      "Reduction View Perform Reduction",                             \
      "Reduction View Perform Deferred Reduction",                    \
      "Reduction View Perform Deferred Reduction Across",             \
      "Reduction View Find Copy Preconditions",                       \
      "Reduction View Find User Preconditions",                       \
      "Reduction View Filter Local Users",                            \
    };

    enum SemanticInfoKind {
      INDEX_SPACE_SEMANTIC,
      INDEX_PARTITION_SEMANTIC,
      FIELD_SPACE_SEMANTIC,
      FIELD_SEMANTIC,
      LOGICAL_REGION_SEMANTIC,
      LOGICAL_PARTITION_SEMANTIC,
      TASK_SEMANTIC,
    };

    // Forward declarations for runtime level objects
    // runtime.h
    class Collectable;
    class ArgumentMapImpl;
    class ArgumentMapStore;
    class FutureImpl;
    class FutureMapImpl;
    class PhysicalRegionImpl;
    class GrantImpl;
    class PredicateImpl;
    class MPILegionHandshakeImpl;
    class ProcessorManager;
    class MemoryManager;
    class MessageManager;
    class GarbageCollectionEpoch;
    class TaskImpl;
    class VariantImpl;
    class LayoutConstraints;
    class GeneratorImpl;
    class Runtime;

    // legion_ops.h
    class Operation;
    class SpeculativeOp;
    class MapOp;
    class CopyOp;
    class FenceOp;
    class FrameOp;
    class DeletionOp;
    class CloseOp;
    class TraceCloseOp;
    class InterCloseOp;
    class ReadCloseOp;
    class PostCloseOp;
    class VirtualCloseOp;
    class AcquireOp;
    class ReleaseOp;
    class DynamicCollectiveOp;
    class FuturePredOp;
    class NotPredOp;
    class AndPredOp;
    class OrPredOp;
    class MustEpochOp;
    class PendingPartitionOp;
    class DependentPartitionOp;
    class FillOp;
    class AttachOp;
    class DetachOp;
    class TimingOp;
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
    class MinimalPoint;
    
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
    class FatTreePath;
    class PathTraverser;
    class NodeTraverser;
    class PhysicalTraverser;
    class PremapTraverser;
    class MappingTraverser;
    class RestrictInfo;

    class CurrentState;
    class PhysicalState;
    class VersionState;
    class VersionInfo;
    class RestrictInfo;

    class Collectable;
    class Notifiable;
    class ReferenceMutator;
    class LocalReferenceMutator;
    class NeverReferenceMutator;
    class DistributedCollectable;
    class LayoutDescription;
    class PhysicalManager; // base class for instance and reduction
    class CopyAcrossHelper;
    class LogicalView; // base class for instance and reduction
    class InstanceManager;
    class InstanceKey;
    class InstanceView;
    class DeferredView;
    class MaterializedView;
    class CompositeView;
    class CompositeVersionInfo;
    class CompositeNode;
    class FillView;
    class MappingRef;
    class InstanceRef;
    class InstanceSet;
    class InnerTaskView;
    class ReductionManager;
    class ListReductionManager;
    class FoldReductionManager;
    class VirtualManager;
    class ReductionView;
    class InstanceBuilder;

    class RegionAnalyzer;
    class RegionMapper;

    struct EscapedUser;
    struct EscapedCopy;
    struct GenericUser;
    struct LogicalUser;
    struct PhysicalUser;
    struct TraceInfo;
    class LogicalCloser;
    class PhysicalCloser;
    class CompositeCloser;
    class ReductionCloser;
    class TreeCloseImpl;
    class TreeClose;
    struct CloseInfo; 

    // legion_spy.h
    class TreeStateLogger;

    // legion_profiling.h
    class LegionProfiler;
    class LegionProfInstance;

    // mapper_manager.h
    class MappingCallInfo;
    class MapperManager;
    class SerializingManager;
    class ConcurrentManager;
    typedef Mapping::MapperEvent MapperEvent;

#define FRIEND_ALL_RUNTIME_CLASSES                          \
    friend class Legion::Runtime;                           \
    friend class Internal::Runtime;                         \
    friend class Internal::PhysicalRegionImpl;              \
    friend class Internal::TaskImpl;                        \
    friend class Internal::ProcessorManager;                \
    friend class Internal::MemoryManager;                   \
    friend class Internal::Operation;                       \
    friend class Internal::SpeculativeOp;                   \
    friend class Internal::MapOp;                           \
    friend class Internal::CopyOp;                          \
    friend class Internal::FenceOp;                         \
    friend class Internal::DynamicCollectiveOp;             \
    friend class Internal::FuturePredOp;                    \
    friend class Internal::DeletionOp;                      \
    friend class Internal::CloseOp;                         \
    friend class Internal::TraceCloseOp;                    \
    friend class Internal::InterCloseOp;                    \
    friend class Internal::ReadCloseOp;                     \
    friend class Internal::PostCloseOp;                     \
    friend class Internal::VirtualCloseOp;                  \
    friend class Internal::AcquireOp;                       \
    friend class Internal::ReleaseOp;                       \
    friend class Internal::NotPredOp;                       \
    friend class Internal::AndPredOp;                       \
    friend class Internal::OrPredOp;                        \
    friend class Internal::MustEpochOp;                     \
    friend class Internal::PendingPartitionOp;              \
    friend class Internal::DependentPartitionOp;            \
    friend class Internal::FillOp;                          \
    friend class Internal::AttachOp;                        \
    friend class Internal::DetachOp;                        \
    friend class Internal::TimingOp;                        \
    friend class Internal::TaskOp;                          \
    friend class Internal::SingleTask;                      \
    friend class Internal::MultiTask;                       \
    friend class Internal::IndividualTask;                  \
    friend class Internal::PointTask;                       \
    friend class Internal::IndexTask;                       \
    friend class Internal::SliceTask;                       \
    friend class Internal::RegionTreeForest;                \
    friend class Internal::IndexSpaceNode;                  \
    friend class Internal::IndexPartNode;                   \
    friend class Internal::FieldSpaceNode;                  \
    friend class Internal::RegionTreeNode;                  \
    friend class Internal::RegionNode;                      \
    friend class Internal::PartitionNode;                   \
    friend class Internal::LogicalView;                     \
    friend class Internal::InstanceView;                    \
    friend class Internal::DeferredView;                    \
    friend class Internal::ReductionView;                   \
    friend class Internal::MaterializedView;                \
    friend class Internal::CompositeView;                   \
    friend class Internal::CompositeNode;                   \
    friend class Internal::FillView;                        \
    friend class Internal::LayoutDescription;               \
    friend class Internal::PhysicalManager;                 \
    friend class Internal::InstanceManager;                 \
    friend class Internal::ReductionManager;                \
    friend class Internal::ListReductionManager;            \
    friend class Internal::FoldReductionManager;            \
    friend class Internal::TreeStateLogger;                 \
    friend class Internal::MapperManager;                   \
    friend class Internal::InstanceRef;                     \
    friend class BindingLib::Utility;                       \
    friend class CObjectWrapper;                  

#define LEGION_EXTERN_LOGGER_DECLARATIONS                        \
    extern LegionRuntime::Logger::Category log_run;              \
    extern LegionRuntime::Logger::Category log_task;             \
    extern LegionRuntime::Logger::Category log_index;            \
    extern LegionRuntime::Logger::Category log_field;            \
    extern LegionRuntime::Logger::Category log_region;           \
    extern LegionRuntime::Logger::Category log_inst;             \
    extern LegionRuntime::Logger::Category log_variant;          \
    extern LegionRuntime::Logger::Category log_allocation;       \
    extern LegionRuntime::Logger::Category log_prof;             \
    extern LegionRuntime::Logger::Category log_garbage;          \
    extern LegionRuntime::Logger::Category log_spy;              \
    extern LegionRuntime::Logger::Category log_shutdown;

  }; // Internal namespace

  // Typedefs that are needed everywhere
  typedef Realm::Runtime RealmRuntime;
  typedef Realm::Machine Machine;
  typedef Realm::Domain Domain;
  typedef Realm::DomainPoint DomainPoint;
  typedef Realm::IndexSpaceAllocator IndexSpaceAllocator;
  typedef Realm::Memory Memory;
  typedef Realm::Processor Processor;
  typedef Realm::CodeDescriptor CodeDescriptor;
  typedef Realm::Reservation Reservation;
  typedef ::legion_reduction_op_id_t ReductionOpID;
  typedef Realm::ReductionOpUntyped ReductionOp;
  typedef ::legion_custom_serdez_id_t CustomSerdezID;
  typedef Realm::CustomSerdezUntyped SerdezOp;
  typedef Realm::Machine::ProcessorMemoryAffinity ProcessorMemoryAffinity;
  typedef Realm::Machine::MemoryMemoryAffinity MemoryMemoryAffinity;
  typedef Realm::ElementMask::Enumerator Enumerator;
  typedef Realm::IndexSpace::FieldDataDescriptor FieldDataDescriptor;
  typedef std::map<CustomSerdezID, 
                   const Realm::CustomSerdezUntyped *> SerdezOpTable;
  typedef std::map<Realm::ReductionOpID, 
          const Realm::ReductionOpUntyped *> ReductionOpTable;
  typedef void (*SerdezInitFnptr)(const ReductionOp*, void *&, size_t&);
  typedef void (*SerdezFoldFnptr)(const ReductionOp*, void *&, 
                                  size_t&, const void*);
  typedef std::map<Realm::ReductionOpID, SerdezRedopFns> SerdezRedopTable;
  typedef ::legion_address_space_t AddressSpace;
  typedef ::legion_task_priority_t TaskPriority;
  typedef ::legion_garbage_collection_priority_t GCPriority;
  typedef ::legion_color_t Color;
  typedef ::legion_field_id_t FieldID;
  typedef ::legion_trace_id_t TraceID;
  typedef ::legion_mapper_id_t MapperID;
  typedef ::legion_context_id_t ContextID;
  typedef ::legion_instance_id_t InstanceID;
  typedef ::legion_index_space_id_t IndexSpaceID;
  typedef ::legion_index_partition_id_t IndexPartitionID;
  typedef ::legion_index_tree_id_t IndexTreeID;
  typedef ::legion_field_space_id_t FieldSpaceID;
  typedef ::legion_generation_id_t GenerationID;
  typedef ::legion_type_handle TypeHandle;
  typedef ::legion_projection_id_t ProjectionID;
  typedef ::legion_region_tree_id_t RegionTreeID;
  typedef ::legion_distributed_id_t DistributedID;
  typedef ::legion_address_space_id_t AddressSpaceID;
  typedef ::legion_tunable_id_t TunableID;
  typedef ::legion_generator_id_t GeneratorID;
  typedef ::legion_mapping_tag_id_t MappingTagID;
  typedef ::legion_semantic_tag_t SemanticTag;
  typedef ::legion_variant_id_t VariantID;
  typedef ::legion_unique_id_t UniqueID;
  typedef ::legion_version_id_t VersionID;
  typedef ::legion_task_id_t TaskID;
  typedef ::legion_layout_constraint_id_t LayoutConstraintID;
  typedef std::map<Color,ColoredPoints<ptr_t> > Coloring;
  typedef std::map<Color,Domain> DomainColoring;
  typedef std::map<Color,std::set<Domain> > MultiDomainColoring;
  typedef std::map<DomainPoint,ColoredPoints<ptr_t> > PointColoring;
  typedef std::map<DomainPoint,Domain> DomainPointColoring;
  typedef std::map<DomainPoint,std::set<Domain> > MultiDomainPointColoring;
  typedef void (*RegistrationCallbackFnptr)(Machine machine, 
                Runtime *rt, const std::set<Processor> &local_procs);
  typedef LogicalRegion (*RegionProjectionFnptr)(LogicalRegion parent, 
      const DomainPoint&, Runtime *rt);
  typedef LogicalRegion (*PartitionProjectionFnptr)(LogicalPartition parent, 
      const DomainPoint&, Runtime *rt);
  typedef bool (*PredicateFnptr)(const void*, size_t, 
      const std::vector<Future> futures);
  typedef std::map<ProjectionID,RegionProjectionFnptr> 
    RegionProjectionTable;
  typedef std::map<ProjectionID,PartitionProjectionFnptr> 
    PartitionProjectionTable;
  typedef void (*RealmFnptr)(const void*,size_t,
                             const void*,size_t,Processor);
  // The most magical of typedefs
  typedef Internal::SingleTask* Context;
  typedef Internal::GeneratorImpl* GeneratorContext;
  typedef void (*GeneratorFnptr)(GeneratorContext,
                                 const TaskGeneratorArguments&, Runtime*);
  // Anothing magical typedef
  namespace Mapping {
    typedef Internal::MappingCallInfo* MapperContext;
    typedef Internal::PhysicalManager* PhysicalInstanceImpl;
  };

  namespace Internal { 
    // This is only needed internally
    typedef Realm::RegionInstance PhysicalInstance;
    // Pull some of the mapper types into the internal space
    typedef Mapping::Mapper Mapper;
    typedef Mapping::PhysicalInstance MappingInstance;
    // A little bit of logic here to figure out the 
    // kind of bit mask to use for FieldMask

// The folowing macros are used in the FieldMask instantiation of BitMask
// If you change one you probably have to change the others too
#define LEGION_FIELD_MASK_FIELD_TYPE          uint64_t 
#define LEGION_FIELD_MASK_FIELD_SHIFT         6
#define LEGION_FIELD_MASK_FIELD_MASK          0x3F
#define LEGION_FIELD_MASK_FIELD_ALL_ONES      0xFFFFFFFFFFFFFFFF

#if defined(__AVX__)
#if (MAX_FIELDS > 256)
    typedef AVXTLBitMask<MAX_FIELDS> FieldMask;
#elif (MAX_FIELDS > 128)
    typedef AVXBitMask<MAX_FIELDS> FieldMask;
#elif (MAX_FIELDS > 64)
    typedef SSEBitMask<MAX_FIELDS> FieldMask;
#else
    typedef BitMask<LEGION_FIELD_MASK_FIELD_TYPE,MAX_FIELDS,
                    LEGION_FIELD_MASK_FIELD_SHIFT,
                    LEGION_FIELD_MASK_FIELD_MASK> FieldMask;
#endif
#elif defined(__SSE2__)
#if (MAX_FIELDS > 128)
    typedef SSETLBitMask<MAX_FIELDS> FieldMask;
#elif (MAX_FIELDS > 64)
    typedef SSEBitMask<MAX_FIELDS> FieldMask;
#else
    typedef BitMask<LEGION_FIELD_MASK_FIELD_TYPE,MAX_FIELDS,
                    LEGION_FIELD_MASK_FIELD_SHIFT,
                    LEGION_FIELD_MASK_FIELD_MASK> FieldMask;
#endif
#else
#if (MAX_FIELDS > 64)
    typedef TLBitMask<LEGION_FIELD_MASK_FIELD_TYPE,MAX_FIELDS,
                      LEGION_FIELD_MASK_FIELD_SHIFT,
                      LEGION_FIELD_MASK_FIELD_MASK> FieldMask;
#else
    typedef BitMask<LEGION_FIELD_MASK_FIELD_TYPE,MAX_FIELDS,
                    LEGION_FIELD_MASK_FIELD_SHIFT,
                    LEGION_FIELD_MASK_FIELD_MASK> FieldMask;
#endif
#endif
    typedef BitPermutation<FieldMask,LEGION_FIELD_LOG2> FieldPermutation;
    typedef Fraction<unsigned long> InstFrac;
#undef LEGION_FIELD_MASK_FIELD_SHIFT
#undef LEGION_FIELD_MASK_FIELD_MASK

    // Similar logic as field masks for node masks

// The following macros are used in the NodeMask instantiation of BitMask
// If you change one you probably have to change the others too
#define LEGION_NODE_MASK_NODE_TYPE           uint64_t
#define LEGION_NODE_MASK_NODE_SHIFT          6
#define LEGION_NODE_MASK_NODE_MASK           0x3F
#define LEGION_NODE_MASK_NODE_ALL_ONES       0xFFFFFFFFFFFFFFFF

#if defined(__AVX__)
#if (MAX_NUM_NODES > 256)
    typedef AVXTLBitMask<MAX_NUM_NODES> NodeMask;
#elif (MAX_NUM_NODES > 128)
    typedef AVXBitMask<MAX_NUM_NODES> NodeMask;
#elif (MAX_NUM_NODES > 64)
    typedef SSEBitMask<MAX_NUM_NODES> NodeMask;
#else
    typedef BitMask<LEGION_NODE_MASK_NODE_TYPE,MAX_NUM_NODES,
                    LEGION_NODE_MASK_NODE_SHIFT,
                    LEGION_NODE_MASK_NODE_MASK> NodeMask;
#endif
#elif defined(__SSE2__)
#if (MAX_NUM_NODES > 128)
    typedef SSETLBitMask<MAX_NUM_NODES> NodeMask;
#elif (MAX_NUM_NODES > 64)
    typedef SSEBitMask<MAX_NUM_NODES> NodeMask;
#else
    typedef BitMask<LEGION_NODE_MASK_NODE_TYPE,MAX_NUM_NODES,
                    LEGION_NODE_MASK_NODE_SHIFT,
                    LEGION_NODE_MASK_NODE_MASK> NodeMask;
#endif
#else
#if (MAX_NUM_NODES > 64)
    typedef TLBitMask<LEGION_NODE_MASK_NODE_TYPE,MAX_NUM_NODES,
                      LEGION_NODE_MASK_NODE_SHIFT,
                      LEGION_NODE_MASK_NODE_MASK> NodeMask;
#else
    typedef BitMask<LEGION_NODE_MASK_NODE_TYPE,MAX_NUM_NODES,
                    LEGION_NODE_MASK_NODE_SHIFT,
                    LEGION_NODE_MASK_NODE_MASK> NodeMask;
#endif
#endif
    typedef IntegerSet<AddressSpaceID,NodeMask> NodeSet;

#undef LEGION_NODE_MASK_NODE_SHIFT
#undef LEGION_NODE_MASK_NODE_MASK

// The following macros are used in the ProcessorMask instantiation of BitMask
// If you change one you probably have to change the others too
#define LEGION_PROC_MASK_PROC_TYPE           uint64_t
#define LEGION_PROC_MASK_PROC_SHIFT          6
#define LEGION_PROC_MASK_PROC_MASK           0x3F
#define LEGION_PROC_MASK_PROC_ALL_ONES       0xFFFFFFFFFFFFFFFF

#if defined(__AVX__)
#if (MAX_NUM_PROCS > 256)
    typedef AVXTLBitMask<MAX_NUM_PROCS> ProcessorMask;
#elif (MAX_NUM_PROCS > 128)
    typedef AVXBitMask<MAX_NUM_PROCS> ProcessorMask;
#elif (MAX_NUM_PROCS > 64)
    typedef SSEBitMask<MAX_NUM_PROCS> ProcessorMask;
#else
    typedef BitMask<LEGION_PROC_MASK_PROC_TYPE,MAX_NUM_PROCS,
                    LEGION_PROC_MASK_PROC_SHIFT,
                    LEGION_PROC_MASK_PROC_MASK> ProcessorMask;
#endif
#elif defined(__SSE2__)
#if (MAX_NUM_PROCS > 128)
    typedef SSETLBitMask<MAX_NUM_PROCS> ProcessorMask;
#elif (MAX_NUM_PROCS > 64)
    typedef SSEBitMask<MAX_NUM_PROCS> ProcessorMask;
#else
    typedef BitMask<LEGION_PROC_MASK_PROC_TYPE,MAX_NUM_PROCS,
                    LEGION_PROC_MASK_PROC_SHIFT,
                    LEGION_PROC_MASK_PROC_MASK> ProcessorMask;
#endif
#else
#if (MAX_NUM_PROCS > 64)
    typedef TLBitMask<LEGION_PROC_MASK_PROC_TYPE,MAX_NUM_PROCS,
                      LEGION_PROC_MASK_PROC_SHIFT,
                      LEGION_PROC_MASK_PROC_MASK> ProcessorMask;
#else
    typedef BitMask<LEGION_PROC_MASK_PROC_TYPE,MAX_NUM_PROCS,
                    LEGION_PROC_MASK_PROC_SHIFT,
                    LEGION_PROC_MASK_PROC_MASK> ProcessorMask;
#endif
#endif

#undef PROC_SHIFT
#undef PROC_MASK
  }; // namespace Internal

  // Legion derived event types
  class LgEvent : public Realm::Event {
  public:
    static const LgEvent NO_LG_EVENT;
  public:
    LgEvent(void) { id = 0; gen = 0; }
    LgEvent(const LgEvent &rhs) { id = rhs.id; gen = rhs.gen; }
    explicit LgEvent(const Realm::Event e) { id = e.id; gen = e.gen; }
  public:
    inline LgEvent& operator=(const LgEvent &rhs)
      { id = rhs.id; gen = rhs.gen; return *this; }
  };

  class ApEvent : public LgEvent {
  public:
    static const ApEvent NO_AP_EVENT;
  public:
    ApEvent(void) : LgEvent() { }
    ApEvent(const ApEvent &rhs) { id = rhs.id; gen = rhs.gen; }
    explicit ApEvent(const Realm::Event &e) : LgEvent(e) { }
  public:
    inline ApEvent& operator=(const ApEvent &rhs)
      { id = rhs.id; gen = rhs.gen; return *this; }
  };

  class ApUserEvent : public ApEvent {
  public:
    static const ApUserEvent NO_AP_USER_EVENT;
  public:
    ApUserEvent(void) : ApEvent() { }
    ApUserEvent(const ApUserEvent &rhs) : ApEvent(rhs) { }
    explicit ApUserEvent(const Realm::UserEvent &e) : ApEvent(e) { }
  public:
    inline ApUserEvent& operator=(const ApUserEvent &rhs)
      { id = rhs.id; gen = rhs.gen; return *this; }
    inline operator Realm::UserEvent() const
      { Realm::UserEvent e; e.id = id; e.gen = gen; return e; }
  };

  class ApBarrier : public ApEvent {
  public:
    static const ApBarrier NO_AP_BARRIER;
  public:
    ApBarrier(void) : ApEvent(), timestamp(0) { }
    ApBarrier(const ApBarrier &rhs) 
      : ApEvent(rhs), timestamp(rhs.timestamp) { }
    explicit ApBarrier(const Realm::Barrier &b) 
      : ApEvent(b), timestamp(b.timestamp) { }
  public:
    inline ApBarrier& operator=(const ApBarrier &rhs)
      { id = rhs.id; gen = rhs.gen; timestamp = rhs.timestamp; return *this; }
    inline operator Realm::Barrier() const
      { Realm::Barrier b; b.id = id; b.gen = gen; 
        b.timestamp = timestamp; return b; }
  public:
    Realm::Barrier::timestamp_t timestamp;
  };

  class RtEvent : public LgEvent {
  public:
    static const RtEvent NO_RT_EVENT;
  public:
    RtEvent(void) : LgEvent() { }
    RtEvent(const RtEvent &rhs) { id = rhs.id; gen = rhs.gen; }
    explicit RtEvent(const Realm::Event &e) : LgEvent(e) { }
  public:
    inline RtEvent& operator=(const RtEvent &rhs)
      { id = rhs.id; gen = rhs.gen; return *this; }
  };

  class RtUserEvent : public RtEvent {
  public:
    static const RtUserEvent NO_RT_USER_EVENT;
  public:
    RtUserEvent(void) : RtEvent() { }
    RtUserEvent(const RtUserEvent &rhs) : RtEvent(rhs) { }
    explicit RtUserEvent(const Realm::UserEvent &e) : RtEvent(e) { }
  public:
    inline RtUserEvent& operator=(const RtUserEvent &rhs)
      { id = rhs.id; gen = rhs.gen; return *this; }
    inline operator Realm::UserEvent() const
      { Realm::UserEvent e; e.id = id; e.gen = gen; return e; }
  };

  class RtBarrier : public RtEvent {
  public:
    static const RtBarrier NO_RT_BARRIER;
  public:
    RtBarrier(void) : RtEvent(), timestamp(0) { }
    RtBarrier(const RtBarrier &rhs)
      : RtEvent(rhs), timestamp(rhs.timestamp) { }
    explicit RtBarrier(const Realm::Barrier &b)
      : RtEvent(b), timestamp(b.timestamp) { }
  public:
    inline RtBarrier& operator=(const RtBarrier &rhs)
      { id = rhs.id; gen = rhs.gen; timestamp = rhs.timestamp; return *this; }
    inline operator Realm::Barrier() const
      { Realm::Barrier b; b.id = id; b.gen = gen; 
        b.timestamp = timestamp; return b; } 
  public:
    Realm::Barrier::timestamp_t timestamp;
  };

}; // Legion namespace

#endif // __LEGION_TYPES_H__
