/* Copyright 2026 Stanford University, NVIDIA Corporation
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

// Included from message.h - do not include this directly

// Useful for IDEs
#include "legion/managers/message.h"

namespace Legion {
  namespace Internal {

    //--------------------------------------------------------------------------
    /*static*/ constexpr VirtualChannelKind MessageManager::find_message_vc(
        MessageKind kind)
    //--------------------------------------------------------------------------
    {
      switch (kind)
      {
        case SEND_STARTUP_BARRIER:
          break;
        case TASK_MESSAGE:
          return DEFAULT_VIRTUAL_CHANNEL;
        case STEAL_MESSAGE:
          return MAPPER_VIRTUAL_CHANNEL;
        case ADVERTISEMENT_MESSAGE:
          return MAPPER_VIRTUAL_CHANNEL;
        case SEND_REGISTRATION_CALLBACK:
          break;
        case SEND_REMOTE_TASK_REPLAY:
          break;
        case SEND_REMOTE_TASK_PROFILING_RESPONSE:
          break;
        case SEND_SHARED_OWNERSHIP:
          break;
        case SEND_INDEX_SPACE_REQUEST:
          break;
        case SEND_INDEX_SPACE_RESPONSE:
          return INDEX_SPACE_VIRTUAL_CHANNEL;
        case SEND_INDEX_SPACE_RETURN:
          return INDEX_SPACE_VIRTUAL_CHANNEL;
        case SEND_INDEX_SPACE_SET:
          break;
        case SEND_INDEX_SPACE_CHILD_REQUEST:
          break;
        case SEND_INDEX_SPACE_CHILD_RESPONSE:
          break;
        case SEND_INDEX_SPACE_COLORS_REQUEST:
          break;
        case SEND_INDEX_SPACE_COLORS_RESPONSE:
          break;
        case SEND_INDEX_SPACE_GENERATE_COLOR_REQUEST:
          break;
        case SEND_INDEX_SPACE_GENERATE_COLOR_RESPONSE:
          break;
        case SEND_INDEX_SPACE_RELEASE_COLOR:
          break;
        case SEND_INDEX_PARTITION_NOTIFICATION:
          break;
        case SEND_INDEX_PARTITION_REQUEST:
          break;
        case SEND_INDEX_PARTITION_RESPONSE:
          return INDEX_SPACE_VIRTUAL_CHANNEL;
        case SEND_INDEX_PARTITION_RETURN:
          return INDEX_SPACE_VIRTUAL_CHANNEL;
        case SEND_INDEX_PARTITION_CHILD_REQUEST:
          break;
        case SEND_INDEX_PARTITION_CHILD_RESPONSE:
          break;
        case SEND_INDEX_PARTITION_CHILD_REPLICATION:
          break;
        case SEND_INDEX_PARTITION_DISJOINT_UPDATE:
          break;
        case SEND_INDEX_PARTITION_SHARD_RECTS_REQUEST:
          break;
        case SEND_INDEX_PARTITION_SHARD_RECTS_RESPONSE:
          break;
        case SEND_INDEX_PARTITION_REMOTE_INTERFERENCE_REQUEST:
          break;
        case SEND_INDEX_PARTITION_REMOTE_INTERFERENCE_RESPONSE:
          break;
        case SEND_FIELD_SPACE_NODE:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_REQUEST:
          break;
        case SEND_FIELD_SPACE_RETURN:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_ALLOCATOR_REQUEST:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_ALLOCATOR_RESPONSE:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_ALLOCATOR_INVALIDATION:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_ALLOCATOR_FLUSH:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_ALLOCATOR_FREE:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_INFOS_REQUEST:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_INFOS_RESPONSE:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_ALLOC_REQUEST:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SIZE_UPDATE:
          break;
        case SEND_FIELD_FREE:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_FREE_INDEXES:
          return FIELD_SPACE_VIRTUAL_CHANNEL;
        case SEND_FIELD_SPACE_LAYOUT_INVALIDATION:
          break;
        case SEND_LOCAL_FIELD_ALLOC_REQUEST:
          break;
        case SEND_LOCAL_FIELD_ALLOC_RESPONSE:
          break;
        case SEND_LOCAL_FIELD_FREE:
          break;
        case SEND_LOCAL_FIELD_UPDATE:
          break;
        case SEND_TOP_LEVEL_REGION_REQUEST:
          break;
        case SEND_TOP_LEVEL_REGION_RETURN:
          break;
        case INDEX_SPACE_DESTRUCTION_MESSAGE:
          return REFERENCE_VIRTUAL_CHANNEL;
        case INDEX_PARTITION_DESTRUCTION_MESSAGE:
          return REFERENCE_VIRTUAL_CHANNEL;
        case FIELD_SPACE_DESTRUCTION_MESSAGE:
          return REFERENCE_VIRTUAL_CHANNEL;
        case LOGICAL_REGION_DESTRUCTION_MESSAGE:
          return REFERENCE_VIRTUAL_CHANNEL;
        case INDIVIDUAL_REMOTE_FUTURE_SIZE:
          return TASK_VIRTUAL_CHANNEL;
        case INDIVIDUAL_REMOTE_OUTPUT_REGISTRATION:
          return TASK_VIRTUAL_CHANNEL;
        case INDIVIDUAL_REMOTE_MAPPED:
          break;
        case INDIVIDUAL_REMOTE_COMPLETE:
          return TASK_VIRTUAL_CHANNEL;
        case INDIVIDUAL_REMOTE_COMMIT:
          return TASK_VIRTUAL_CHANNEL;
        case INDIVIDUAL_CONCURRENT_REQUEST:
          break;
        case INDIVIDUAL_CONCURRENT_RESPONSE:
          break;
        case SLICE_REMOTE_MAPPED:
          return TASK_VIRTUAL_CHANNEL;
        case SLICE_REMOTE_COMPLETE:
          return TASK_VIRTUAL_CHANNEL;
        case SLICE_REMOTE_COMMIT:
          return TASK_VIRTUAL_CHANNEL;
        case SLICE_RENDEZVOUS_CONCURRENT_MAPPED:
          break;
        case SLICE_COLLECTIVE_ALLREDUCE_REQUEST:
          break;
        case SLICE_COLLECTIVE_ALLREDUCE_RESPONSE:
          break;
        case SLICE_CONCURRENT_ALLREDUCE_REQUEST:
          break;
        case SLICE_CONCURRENT_ALLREDUCE_RESPONSE:
          break;
        case SLICE_FIND_INTRA_DEP:
          break;
        case SLICE_REMOTE_COLLECTIVE_RENDEZVOUS:
          break;
        case SLICE_REMOTE_VERSIONING_COLLECTIVE_RENDEZVOUS:
          break;
        case SLICE_REMOTE_OUTPUT_EXTENTS:
          break;
        case SLICE_REMOTE_OUTPUT_REGISTRATION:
          return TASK_VIRTUAL_CHANNEL;
        case DISTRIBUTED_REMOTE_REGISTRATION:
          break;
        // Low priority so reference counting doesn't starve
        // out the rest of our work
        case DISTRIBUTED_DOWNGRADE_REQUEST:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case DISTRIBUTED_DOWNGRADE_RESPONSE:
          break;
        case DISTRIBUTED_DOWNGRADE_SUCCESS:
          break;
        // Put downgrade updates and acquire requests
        // on same ordered virtual channel so that
        // acquire requests cannot starve out an owner
        // update while it is in flight by circling
        // around and around
        case DISTRIBUTED_DOWNGRADE_UPDATE:
          return REFERENCE_VIRTUAL_CHANNEL;
        case DISTRIBUTED_DOWNGRADE_RESTART:
          break;
        case DISTRIBUTED_GLOBAL_ACQUIRE_REQUEST:
          return REFERENCE_VIRTUAL_CHANNEL;
        case DISTRIBUTED_GLOBAL_ACQUIRE_RESPONSE:
          break;
        case DISTRIBUTED_VALID_ACQUIRE_REQUEST:
          break;
        case DISTRIBUTED_VALID_ACQUIRE_RESPONSE:
          break;
        case SEND_ATOMIC_RESERVATION_REQUEST:
          break;
        case SEND_ATOMIC_RESERVATION_RESPONSE:
          break;
        case SEND_PADDED_RESERVATION_REQUEST:
          break;
        case SEND_PADDED_RESERVATION_RESPONSE:
          break;
        case SEND_CREATED_REGION_CONTEXTS:
          break;
        case SEND_MATERIALIZED_VIEW:
          break;
        case SEND_FILL_VIEW:
          break;
        case SEND_FILL_VIEW_VALUE:
          break;
        case SEND_PHI_VIEW:
          break;
        case SEND_REDUCTION_VIEW:
          break;
        case SEND_REPLICATED_VIEW:
          break;
        case SEND_ALLREDUCE_VIEW:
          break;
        case SEND_INSTANCE_MANAGER:
          break;
        case SEND_MANAGER_UPDATE:
          break;
        // Only collective operations apply to destinations need to be
        // on the ordered virtual channel since they need to be ordered
        // with respect to the same CopyFillAggregator, there's no need
        // to do the same thing read-only collectives since they can
        // never be read more than once by each CopyFillAggregator
        case SEND_COLLECTIVE_DISTRIBUTE_FILL:
          return COLLECTIVE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_DISTRIBUTE_POINT:
          break;  // read-only
        case SEND_COLLECTIVE_DISTRIBUTE_POINTWISE:
          return COLLECTIVE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_DISTRIBUTE_REDUCTION:
          break;  // read-only
        case SEND_COLLECTIVE_DISTRIBUTE_BROADCAST:
          return COLLECTIVE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_DISTRIBUTE_REDUCECAST:
          return COLLECTIVE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_DISTRIBUTE_HOURGLASS:
          return COLLECTIVE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_DISTRIBUTE_ALLREDUCE:
          break;  // no views involved so effectively read-only
        case SEND_COLLECTIVE_HAMMER_REDUCTION:
          break;  // read-only
        case SEND_COLLECTIVE_FUSE_GATHER:
          return COLLECTIVE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_USER_REQUEST:
          break;
        case SEND_COLLECTIVE_USER_RESPONSE:
          break;
        case SEND_COLLECTIVE_REGISTER_USER:
          break;
        case SEND_COLLECTIVE_REMOTE_INSTANCES_REQUEST:
          break;
        case SEND_COLLECTIVE_REMOTE_INSTANCES_RESPONSE:
          break;
        case SEND_COLLECTIVE_NEAREST_INSTANCES_REQUEST:
          break;
        case SEND_COLLECTIVE_NEAREST_INSTANCES_RESPONSE:
          break;
          // These messages need to be ordered with respect to
          // register_user messages so they go on the same VC
        case SEND_COLLECTIVE_REMOTE_REGISTRATION:
          return UPDATE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_FINALIZE_MAPPING:
          break;
        case SEND_COLLECTIVE_VIEW_CREATION:
          break;
        case SEND_COLLECTIVE_VIEW_DELETION:
          break;
        case SEND_COLLECTIVE_VIEW_RELEASE:
          break;
        case SEND_COLLECTIVE_VIEW_NOTIFICATION:
          break;
        // All these collective messages need to go on the same
        // virtual channel since they all need to be ordered
        // with respect to each other
        case SEND_COLLECTIVE_VIEW_MAKE_VALID:
          return REFERENCE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_VIEW_MAKE_INVALID:
          return REFERENCE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_VIEW_INVALIDATE_REQUEST:
          return REFERENCE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_VIEW_INVALIDATE_RESPONSE:
          return REFERENCE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_VIEW_ADD_REMOTE_REFERENCE:
          return REFERENCE_VIRTUAL_CHANNEL;
        case SEND_COLLECTIVE_VIEW_REMOVE_REMOTE_REFERENCE:
          return REFERENCE_VIRTUAL_CHANNEL;
        case SEND_CREATE_TOP_VIEW_REQUEST:
          break;
        case SEND_CREATE_TOP_VIEW_RESPONSE:
          break;
        case SEND_VIEW_REQUEST:
          break;
        case SEND_VIEW_REGISTER_USER:
          return UPDATE_VIRTUAL_CHANNEL;
        case SEND_VIEW_FIND_COPY_PRE_REQUEST:
          return UPDATE_VIRTUAL_CHANNEL;
        case SEND_VIEW_ADD_COPY_USER:
          return UPDATE_VIRTUAL_CHANNEL;
        case SEND_VIEW_FIND_LAST_USERS_REQUEST:
          break;
        case SEND_VIEW_FIND_LAST_USERS_RESPONSE:
          break;
        case SEND_MANAGER_REQUEST:
          break;
        case SEND_FUTURE_RESULT:
          break;
        case SEND_FUTURE_RESULT_SIZE:
          break;
        case SEND_FUTURE_SUBSCRIPTION:
          break;
        case SEND_FUTURE_CREATE_INSTANCE_REQUEST:
          break;
        case SEND_FUTURE_CREATE_INSTANCE_RESPONSE:
          break;
        case SEND_FUTURE_MAP_REQUEST:
          break;
        case SEND_FUTURE_MAP_RESPONSE:
          break;
        case SEND_FUTURE_MAP_POINTWISE:
          break;
        case SEND_REPL_COMPUTE_EQUIVALENCE_SETS:
          break;
        case SEND_REPL_OUTPUT_EQUIVALENCE_SET:
          break;
        case SEND_REPL_OUTPUT_OFFSET:
          break;
        case SEND_REPL_REFINE_EQUIVALENCE_SETS:
          break;
        case SEND_REPL_EQUIVALENCE_SET_NOTIFICATION:
          break;
        case SEND_REPL_BROADCAST_UPDATE:
          break;
        case SEND_REPL_CREATED_REGIONS:
          break;
        case SEND_REPL_TRACE_EVENT_REQUEST:
          break;
        case SEND_REPL_TRACE_EVENT_RESPONSE:
          break;
        case SEND_REPL_TRACE_EVENT_TRIGGER:
          break;
        case SEND_REPL_TRACE_FRONTIER_REQUEST:
          break;
        case SEND_REPL_TRACE_FRONTIER_RESPONSE:
          break;
        case SEND_REPL_TRACE_UPDATE:
          break;
        case SEND_REPL_FIND_TRACE_SETS:
          break;
        case SEND_REPL_IMPLICIT_RENDEZVOUS:
          break;
        case SEND_REPL_FIND_COLLECTIVE_VIEW:
          break;
        case SEND_REPL_POINTWISE_DEPENDENCE:
          break;
        case SEND_MAPPER_MESSAGE:
          return MAPPER_VIRTUAL_CHANNEL;
        case SEND_MAPPER_BROADCAST:
          return MAPPER_VIRTUAL_CHANNEL;
        case SEND_TASK_IMPL_SEMANTIC_REQ:
          break;
        case SEND_INDEX_SPACE_SEMANTIC_REQ:
          break;
        case SEND_INDEX_PARTITION_SEMANTIC_REQ:
          break;
        case SEND_FIELD_SPACE_SEMANTIC_REQ:
          break;
        case SEND_FIELD_SEMANTIC_REQ:
          break;
        case SEND_LOGICAL_REGION_SEMANTIC_REQ:
          break;
        case SEND_LOGICAL_PARTITION_SEMANTIC_REQ:
          break;
        case SEND_TASK_IMPL_SEMANTIC_INFO:
          break;
        case SEND_INDEX_SPACE_SEMANTIC_INFO:
          break;
        case SEND_INDEX_PARTITION_SEMANTIC_INFO:
          break;
        case SEND_FIELD_SPACE_SEMANTIC_INFO:
          break;
        case SEND_FIELD_SEMANTIC_INFO:
          break;
        case SEND_LOGICAL_REGION_SEMANTIC_INFO:
          break;
        case SEND_LOGICAL_PARTITION_SEMANTIC_INFO:
          break;
        case SEND_REMOTE_CONTEXT_REQUEST:
          break;
        case SEND_REMOTE_CONTEXT_RESPONSE:
          break;
        case SEND_REMOTE_CONTEXT_PHYSICAL_REQUEST:
          break;
        case SEND_REMOTE_CONTEXT_PHYSICAL_RESPONSE:
          break;
        case SEND_REMOTE_CONTEXT_FIND_COLLECTIVE_VIEW_REQUEST:
          break;
        case SEND_REMOTE_CONTEXT_FIND_COLLECTIVE_VIEW_RESPONSE:
          break;
        case SEND_REMOTE_CONTEXT_REFINE_EQUIVALENCE_SETS:
          break;
        case SEND_REMOTE_CONTEXT_POINTWISE_DEPENDENCE:
          break;
        case SEND_REMOTE_CONTEXT_FIND_TRACE_LOCAL_SETS_REQUEST:
          break;
        case SEND_REMOTE_CONTEXT_FIND_TRACE_LOCAL_SETS_RESPONSE:
          break;
        case SEND_COMPUTE_EQUIVALENCE_SETS_REQUEST:
          break;
        case SEND_COMPUTE_EQUIVALENCE_SETS_RESPONSE:
          break;
        case SEND_COMPUTE_EQUIVALENCE_SETS_PENDING:
          break;
        case SEND_OUTPUT_EQUIVALENCE_SET_REQUEST:
          break;
        case SEND_OUTPUT_EQUIVALENCE_SET_RESPONSE:
          break;
        case SEND_CANCEL_EQUIVALENCE_SETS_SUBSCRIPTION:
          break;
        case SEND_INVALIDATE_EQUIVALENCE_SETS_SUBSCRIPTION:
          break;
        case SEND_EQUIVALENCE_SET_CREATION:
          break;
        case SEND_EQUIVALENCE_SET_REUSE:
          break;
        case SEND_EQUIVALENCE_SET_REQUEST:
          break;
        case SEND_EQUIVALENCE_SET_RESPONSE:
          break;
        case SEND_EQUIVALENCE_SET_REPLICATION_REQUEST:
          break;
        case SEND_EQUIVALENCE_SET_REPLICATION_RESPONSE:
          break;
        case SEND_EQUIVALENCE_SET_MIGRATION:
          return MIGRATION_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_OWNER_UPDATE:
          return MIGRATION_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_CLONE_REQUEST:
          break;
        case SEND_EQUIVALENCE_SET_CLONE_RESPONSE:
          break;
        case SEND_EQUIVALENCE_SET_CAPTURE_REQUEST:
          break;
        case SEND_EQUIVALENCE_SET_CAPTURE_RESPONSE:
          break;
        case SEND_EQUIVALENCE_SET_REMOTE_REQUEST_INSTANCES:
          break;
        case SEND_EQUIVALENCE_SET_REMOTE_REQUEST_INVALID:
          break;
        case SEND_EQUIVALENCE_SET_REMOTE_REQUEST_ANTIVALID:
          break;
        case SEND_EQUIVALENCE_SET_REMOTE_UPDATES:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_REMOTE_ACQUIRES:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_REMOTE_RELEASES:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_REMOTE_COPIES_ACROSS:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_REMOTE_OVERWRITES:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_REMOTE_FILTERS:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_EQUIVALENCE_SET_REMOTE_INSTANCES:
          break;
        case SEND_EQUIVALENCE_SET_FILTER_INVALIDATIONS:
          break;
        case SEND_INSTANCE_REQUEST:
          break;
        case SEND_INSTANCE_RESPONSE:
          break;
        case SEND_EXTERNAL_CREATE_REQUEST:
          break;
        case SEND_EXTERNAL_CREATE_RESPONSE:
          break;
        case SEND_EXTERNAL_ATTACH:
          break;
        case SEND_EXTERNAL_DETACH:
          break;
        case SEND_GC_PRIORITY_UPDATE:
          break;
        case SEND_GC_REQUEST:
          break;
        case SEND_GC_RESPONSE:
          break;
        case SEND_GC_ACQUIRE:
          break;
        case SEND_GC_FAILED:
          break;
        case SEND_GC_MISMATCH:
          break;
        case SEND_GC_NOTIFY:
          // This one goes on the resource virtual channel because there
          // is nothing else preventing the deletion of the managers
          return REFERENCE_VIRTUAL_CHANNEL;
        case SEND_GC_DEBUG_REQUEST:
          break;
        case SEND_GC_DEBUG_RESPONSE:
          break;
        case SEND_GC_RECORD_EVENT:
          break;
        case SEND_ACQUIRE_REQUEST:
          break;
        case SEND_ACQUIRE_RESPONSE:
          break;
        case SEND_VARIANT_BROADCAST:
          break;
        case SEND_CONSTRAINT_REQUEST:
          return LAYOUT_CONSTRAINT_VIRTUAL_CHANNEL;
        case SEND_CONSTRAINT_RESPONSE:
          return LAYOUT_CONSTRAINT_VIRTUAL_CHANNEL;
        case SEND_CONSTRAINT_RELEASE:
          return LAYOUT_CONSTRAINT_VIRTUAL_CHANNEL;
        case SEND_TOP_LEVEL_TASK_COMPLETE:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_MPI_RANK_EXCHANGE:
          break;
        case SEND_REPLICATE_DISTRIBUTION:
          break;
        case SEND_REPLICATE_COLLECTIVE_VERSIONING:
          break;
        case SEND_REPLICATE_COLLECTIVE_MAPPING:
          break;
        case SEND_REPLICATE_VIRTUAL_RENDEZVOUS:
          break;
        case SEND_REPLICATE_STARTUP_COMPLETE:
          break;
        case SEND_REPLICATE_POST_MAPPED:
          break;
        case SEND_REPLICATE_TRIGGER_COMPLETE:
          break;
        case SEND_REPLICATE_TRIGGER_COMMIT:
          break;
        // All rendezvous messages need to be ordered
        case SEND_CONTROL_REPLICATE_RENDEZVOUS_MESSAGE:
          return RENDEZVOUS_VIRTUAL_CHANNEL;
        case SEND_LIBRARY_MAPPER_REQUEST:
          break;
        case SEND_LIBRARY_MAPPER_RESPONSE:
          break;
        case SEND_LIBRARY_TRACE_REQUEST:
          break;
        case SEND_LIBRARY_TRACE_RESPONSE:
          break;
        case SEND_LIBRARY_PROJECTION_REQUEST:
          break;
        case SEND_LIBRARY_PROJECTION_RESPONSE:
          break;
        case SEND_LIBRARY_SHARDING_REQUEST:
          break;
        case SEND_LIBRARY_SHARDING_RESPONSE:
          break;
        case SEND_LIBRARY_CONCURRENT_REQUEST:
          break;
        case SEND_LIBRARY_CONCURRENT_RESPONSE:
          break;
        case SEND_LIBRARY_EXCEPTION_REQUEST:
          break;
        case SEND_LIBRARY_EXCEPTION_RESPONSE:
          break;
        case SEND_LIBRARY_TASK_REQUEST:
          break;
        case SEND_LIBRARY_TASK_RESPONSE:
          break;
        case SEND_LIBRARY_REDOP_REQUEST:
          break;
        case SEND_LIBRARY_REDOP_RESPONSE:
          break;
        case SEND_LIBRARY_SERDEZ_REQUEST:
          break;
        case SEND_LIBRARY_SERDEZ_RESPONSE:
          break;
        case SEND_REMOTE_OP_REPORT_UNINIT:
          break;
        case SEND_REMOTE_OP_PROFILING_COUNT_UPDATE:
          break;
        case SEND_REMOTE_OP_COMPLETION_EFFECT:
          break;
        case SEND_REMOTE_TRACE_UPDATE:
          return TRACING_VIRTUAL_CHANNEL;
        case SEND_REMOTE_TRACE_RESPONSE:
          break;
        case SEND_FREE_EXTERNAL_ALLOCATION:
          break;
        case SEND_NOTIFY_COLLECTED_INSTANCES:
          break;
        case SEND_CREATE_MEMORY_POOL_REQUEST:
          return POOL_VIRTUAL_CHANNEL;
        case SEND_CREATE_MEMORY_POOL_RESPONSE:
          break;
        case SEND_RELEASE_MEMORY_POOL_REQUEST:
          return POOL_VIRTUAL_CHANNEL;
        case SEND_CREATE_UNBOUND_REQUEST:
          break;
        case SEND_CREATE_UNBOUND_RESPONSE:
          break;
        case SEND_CREATE_FUTURE_INSTANCE_REQUEST:
          break;
        case SEND_CREATE_FUTURE_INSTANCE_RESPONSE:
          break;
        case SEND_FREE_FUTURE_INSTANCE:
          break;
        case SEND_REMOTE_DISTRIBUTED_ID_REQUEST:
          break;
        case SEND_REMOTE_DISTRIBUTED_ID_RESPONSE:
          break;
        case SEND_CONTROL_REPLICATION_FUTURE_ALLREDUCE:
        case SEND_CONTROL_REPLICATION_FUTURE_BROADCAST:
        case SEND_CONTROL_REPLICATION_FUTURE_REDUCTION:
        case SEND_CONTROL_REPLICATION_VALUE_ALLREDUCE:
        case SEND_CONTROL_REPLICATION_VALUE_BROADCAST:
        case SEND_CONTROL_REPLICATION_VALUE_EXCHANGE:
        case SEND_CONTROL_REPLICATION_BUFFER_BROADCAST:
        case SEND_CONTROL_REPLICATION_SHARD_SYNC_TREE:
        case SEND_CONTROL_REPLICATION_SHARD_EVENT_TREE:
        case SEND_CONTROL_REPLICATION_SINGLE_TASK_TREE:
        case SEND_CONTROL_REPLICATION_CROSS_PRODUCT_PARTITION:
        case SEND_CONTROL_REPLICATION_SHARDING_GATHER_COLLECTIVE:
        case SEND_CONTROL_REPLICATION_INDIRECT_COPY_EXCHANGE:
        case SEND_CONTROL_REPLICATION_FIELD_DESCRIPTOR_EXCHANGE:
        case SEND_CONTROL_REPLICATION_FIELD_DESCRIPTOR_GATHER:
        case SEND_CONTROL_REPLICATION_DEPPART_RESULT_SCATTER:
        case SEND_CONTROL_REPLICATION_BUFFER_EXCHANGE:
        case SEND_CONTROL_REPLICATION_FUTURE_NAME_EXCHANGE:
        case SEND_CONTROL_REPLICATION_MUST_EPOCH_MAPPING_BROADCAST:
        case SEND_CONTROL_REPLICATION_MUST_EPOCH_MAPPING_EXCHANGE:
        case SEND_CONTROL_REPLICATION_MUST_EPOCH_DEPENDENCE_EXCHANGE:
        case SEND_CONTROL_REPLICATION_MUST_EPOCH_COMPLETION_EXCHANGE:
        case SEND_CONTROL_REPLICATION_CHECK_COLLECTIVE_MAPPING:
        case SEND_CONTROL_REPLICATION_CHECK_COLLECTIVE_SOURCES:
        case SEND_CONTROL_REPLICATION_TEMPLATE_INDEX_EXCHANGE:
        case SEND_CONTROL_REPLICATION_UNORDERED_EXCHANGE:
        case SEND_CONTROL_REPLICATION_CONSENSUS_MATCH:
        case SEND_CONTROL_REPLICATION_VERIFY_CONTROL_REPLICATION_EXCHANGE:
        case SEND_CONTROL_REPLICATION_OUTPUT_SIZE_EXCHANGE:
        case SEND_CONTROL_REPLICATION_INDEX_ATTACH_LAUNCH_SPACE:
        case SEND_CONTROL_REPLICATION_INDEX_ATTACH_UPPER_BOUND:
        case SEND_CONTROL_REPLICATION_INDEX_ATTACH_EXCHANGE:
        case SEND_CONTROL_REPLICATION_SHARD_PARTICIPANTS_EXCHANGE:
        case SEND_CONTROL_REPLICATION_IMPLICIT_SHARDING_FUNCTOR:
        case SEND_CONTROL_REPLICATION_CREATE_FILL_VIEW:
        case SEND_CONTROL_REPLICATION_VERSIONING_RENDEZVOUS:
        case SEND_CONTROL_REPLICATION_VIEW_RENDEZVOUS:
        case SEND_CONTROL_REPLICATION_CONCURRENT_MAPPING_RENDEZVOUS:
        case SEND_CONTROL_REPLICATION_CONCURRENT_ALLREDUCE:
        case SEND_CONTROL_REPLICATION_PROJECTION_TREE_EXCHANGE:
        case SEND_CONTROL_REPLICATION_TIMEOUT_MATCH_EXCHANGE:
        case SEND_CONTROL_REPLICATION_MASK_EXCHANGE:
        case SEND_CONTROL_REPLICATION_PREDICATE_EXCHANGE:
        case SEND_CONTROL_REPLICATION_CROSS_PRODUCT_EXCHANGE:
        case SEND_CONTROL_REPLICATION_TRACING_SET_DEDUPLICATION:
        case SEND_CONTROL_REPLICATION_POINTWISE_ALLREDUCE:
        case SEND_CONTROL_REPLICATION_INTERFERING_POINT_EXCHANGE:
        case SEND_CONTROL_REPLICATION_SLOW_BARRIER:
          break;
        case SEND_PROFILER_EVENT_TRIGGER:
        case SEND_PROFILER_EVENT_POISON:
          return PROFILING_VIRTUAL_CHANNEL;
        case SEND_SHUTDOWN_NOTIFICATION:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case SEND_SHUTDOWN_RESPONSE:
          return THROUGHPUT_VIRTUAL_CHANNEL;
        case LAST_SEND_KIND:
          std::abort();
      }
      return DEFAULT_VIRTUAL_CHANNEL;
    }

#define MAKE_ACTIVE_MESSAGES(                                       \
    kind, type, name, response, escape_ctx, escape_op)              \
  class type : public ActiveMessage<type> {                         \
  public:                                                           \
    static constexpr MessageKind KIND = kind;                       \
    static constexpr VirtualChannelKind CHANNEL =                   \
        MessageManager::find_message_vc(KIND);                      \
    static constexpr bool RESPONSE = response;                      \
  public:                                                           \
    type(void) : ActiveMessage<type>(KIND, escape_ctx, escape_op)   \
    { }                                                             \
  public:                                                           \
    static void handle(Deserializer& derez, AddressSpaceID source); \
  };
    LEGION_ACTIVE_MESSAGES(MAKE_ACTIVE_MESSAGES)
#undef MAKE_ACTIVE_MESSAGES

  }  // namespace Internal
}  // namespace Legion
