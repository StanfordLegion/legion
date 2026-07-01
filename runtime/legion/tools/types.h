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

#ifndef __LEGION_TOOL_TYPES_H__
#define __LEGION_TOOL_TYPES_H__

namespace Legion {
  namespace Internal {

    // Realm dependent partitioning kinds
    enum DepPartOpKind {
      DEP_PART_UNION = 0,                   // a single union
      DEP_PART_UNIONS = 1,                  // many parallel unions
      DEP_PART_UNION_REDUCTION = 2,         // union reduction to a single space
      DEP_PART_INTERSECTION = 3,            // a single intersection
      DEP_PART_INTERSECTIONS = 4,           // many parallel intersections
      DEP_PART_INTERSECTION_REDUCTION = 5,  // intersection reduction to a space
      DEP_PART_DIFFERENCE = 6,              // a single difference
      DEP_PART_DIFFERENCES = 7,             // many parallel differences
      DEP_PART_EQUAL = 8,                   // an equal partition operation
      DEP_PART_BY_FIELD = 9,                // create a partition from a field
      DEP_PART_BY_IMAGE = 10,               // create partition by image
      DEP_PART_BY_IMAGE_RANGE = 11,         // create partition by image range
      DEP_PART_BY_PREIMAGE = 12,            // create partition by preimage
      DEP_PART_BY_PREIMAGE_RANGE = 13,  // create partition by preimage range
      DEP_PART_ASSOCIATION = 14,        // create an association
      DEP_PART_WEIGHTS = 15,            // create partition by weights
    };

    // Collective copy kinds
    enum CollectiveKind {
      COLLECTIVE_NONE = 0,
      // Filling a collective instance (both normal and reductions)
      COLLECTIVE_FILL = 1,
      // Broadcasting one normal instance to a collective normal instance
      COLLECTIVE_BROADCAST = 2,
      // Reducing a collective reduction instance to either a
      // single normal or a single reduction instance
      COLLECTIVE_REDUCTION = 3,
      // Performing an all-reduce from a collective reduction instance
      // to a collective normal or reduction instance using a butterfly
      // network reduction (both instances using the same nodes)
      COLLECTIVE_BUTTERFLY_ALLREDUCE = 4,
      // Performing an all-reduce by doing a reduction down to a single
      // instance and then broadcasting the result out from that instance
      // (instances don't exist on the same set of nodes)
      COLLECTIVE_HOURGLASS_ALLREDUCE = 5,
      // Copy from one collective normal instanace to another collective
      // normal instance for each of the points in the destination
      COLLECTIVE_POINT_TO_POINT = 6,
      // Apply a reduction from a single reduction instance to
      // a collective normal instance
      COLLECTIVE_REDUCECAST = 7,
      // Degenerate case: apply a copy-across from a collective reduction
      // instance to any kind of other instance without doing an all-reduce
      COLLECTIVE_HAMMER_REDUCTION = 8,
    };

    enum MappingCallKind {
      GET_MAPPER_NAME_CALL,
      GET_MAPER_SYNC_MODEL_CALL,
      SELECT_TASK_OPTIONS_CALL,
      PREMAP_TASK_CALL,
      SLICE_TASK_CALL,
      MAP_TASK_CALL,
      REPLICATE_TASK_CALL,
      SELECT_VARIANT_CALL,
      POSTMAP_TASK_CALL,
      TASK_SELECT_SOURCES_CALL,
      TASK_SPECULATE_CALL,
      TASK_REPORT_PROFILING_CALL,
      TASK_SELECT_SHARDING_FUNCTOR_CALL,
      MAP_INLINE_CALL,
      INLINE_SELECT_SOURCES_CALL,
      INLINE_REPORT_PROFILING_CALL,
      MAP_COPY_CALL,
      COPY_SELECT_SOURCES_CALL,
      COPY_SPECULATE_CALL,
      COPY_REPORT_PROFILING_CALL,
      COPY_SELECT_SHARDING_FUNCTOR_CALL,
      CLOSE_SELECT_SOURCES_CALL,
      CLOSE_REPORT_PROFILING_CALL,
      CLOSE_SELECT_SHARDING_FUNCTOR_CALL,
      MAP_ACQUIRE_CALL,
      ACQUIRE_SPECULATE_CALL,
      ACQUIRE_REPORT_PROFILING_CALL,
      ACQUIRE_SELECT_SHARDING_FUNCTOR_CALL,
      MAP_RELEASE_CALL,
      RELEASE_SELECT_SOURCES_CALL,
      RELEASE_SPECULATE_CALL,
      RELEASE_REPORT_PROFILING_CALL,
      RELEASE_SELECT_SHARDING_FUNCTOR_CALL,
      SELECT_PARTITION_PROJECTION_CALL,
      MAP_PARTITION_CALL,
      PARTITION_SELECT_SOURCES_CALL,
      PARTITION_REPORT_PROFILING_CALL,
      PARTITION_SELECT_SHARDING_FUNCTOR_CALL,
      FILL_SELECT_SHARDING_FUNCTOR_CALL,
      MAP_FUTURE_MAP_REDUCTION_CALL,
      CONFIGURE_CONTEXT_CALL,
      SELECT_TUNABLE_VALUE_CALL,
      MUST_EPOCH_SELECT_SHARDING_FUNCTOR_CALL,
      MAP_MUST_EPOCH_CALL,
      MAP_DATAFLOW_GRAPH_CALL,
      MEMOIZE_OPERATION_CALL,
      SELECT_TASKS_TO_MAP_CALL,
      SELECT_STEAL_TARGETS_CALL,
      PERMIT_STEAL_REQUEST_CALL,
      HANDLE_MESSAGE_CALL,
      HANDLE_TASK_RESULT_CALL,
      HANDLE_INSTANCE_COLLECTION_CALL,
      APPLICATION_MAPPER_CALL,
      LAST_MAPPER_CALL,
    };

#define MAPPER_CALL_NAMES(name)                   \
  const char* name[LAST_MAPPER_CALL] = {          \
      "get_mapper_name",                          \
      "get_mapper_sync_model",                    \
      "select_task_options",                      \
      "premap_task",                              \
      "slice_task",                               \
      "map_task",                                 \
      "replicate_task",                           \
      "select_task_variant",                      \
      "postmap_task",                             \
      "select_task_sources",                      \
      "speculate (for task)",                     \
      "report profiling (for task)",              \
      "select sharding functor (for task)",       \
      "map_inline",                               \
      "select_inline_sources",                    \
      "report profiling (for inline)",            \
      "map_copy",                                 \
      "select_copy_sources",                      \
      "speculate (for copy)",                     \
      "report_profiling (for copy)",              \
      "select sharding functor (for copy)",       \
      "select_close_sources",                     \
      "report_profiling (for close)",             \
      "select sharding functor (for close)",      \
      "map_acquire",                              \
      "speculate (for acquire)",                  \
      "report_profiling (for acquire)",           \
      "select sharding functor (for acquire)",    \
      "map_release",                              \
      "select_release_sources",                   \
      "speculate (for release)",                  \
      "report_profiling (for release)",           \
      "select sharding functor (for release)",    \
      "select partition projection",              \
      "map_partition",                            \
      "select_partition_sources",                 \
      "report_profiling (for partition)",         \
      "select sharding functor (for partition)",  \
      "select sharding functor (for fill)",       \
      "map future map reduction",                 \
      "configure_context",                        \
      "select_tunable_value",                     \
      "select sharding functor (for must epoch)", \
      "map_must_epoch",                           \
      "map_dataflow_graph",                       \
      "memoize_operation",                        \
      "select_tasks_to_map",                      \
      "select_steal_targets",                     \
      "permit_steal_request",                     \
      "handle_message",                           \
      "handle_task_result",                       \
      "handle_instance_collection",               \
      "application mapper call",                  \
  }

    enum RuntimeCallKind {
      // Runtime call kinds
      RUNTIME_CREATE_INDEX_SPACE_CALL,
      RUNTIME_UNION_INDEX_SPACES_CALL,
      RUNTIME_INTERSECT_INDEX_SPACES_CALL,
      RUNTIME_SUBTRACT_INDEX_SPACES_CALL,
      RUNTIME_CREATE_SHARED_OWNERSHIP_CALL,
      RUNTIME_DESTROY_INDEX_SPACE_CALL,
      RUNTIME_DESTROY_INDEX_PARTITION_CALL,
      RUNTIME_CREATE_EQUAL_PARTITION_CALL,
      RUNTIME_CREATE_PARTITION_BY_WEIGHTS_CALL,
      RUNTIME_CREATE_PARTITION_BY_UNION_CALL,
      RUNTIME_CREATE_PARTITION_BY_INTERSECTION_CALL,
      RUNTIME_CREATE_PARTITION_BY_DIFFERENCE_CALL,
      RUNTIME_CREATE_CROSS_PRODUCT_PARTITIONS_CALL,
      RUNTIME_CREATE_ASSOCIATION_CALL,
      RUNTIME_CREATE_PARTITION_BY_RESTRICTION_CALL,
      RUNTIME_CREATE_PARTITION_BY_BLOCKIFY_CALL,
      RUNTIME_CREATE_PARTITION_BY_DOMAIN_CALL,
      RUNTIME_CREATE_PARTITION_BY_FIELD_CALL,
      RUNTIME_CREATE_PARTITION_BY_IMAGE_CALL,
      RUNTIME_CREATE_PARTITION_BY_IMAGE_RANGE_CALL,
      RUNTIME_CREATE_PARTITION_BY_PREIMAGE_CALL,
      RUNTIME_CREATE_PARTITION_BY_PREIMAGE_RANGE_CALL,
      RUNTIME_CREATE_PENDING_PARTITION_CALL,
      RUNTIME_CREATE_INDEX_SPACE_UNION_CALL,
      RUNTIME_CREATE_INDEX_SPACE_INTERSECTION_CALL,
      RUNTIME_CREATE_INDEX_SPACE_DIFFERENCE_CALL,
      RUNTIME_GET_INDEX_PARTITION_CALL,
      RUNTIME_HAS_INDEX_PARTITION_CALL,
      RUNTIME_GET_INDEX_SUBSPACE_CALL,
      RUNTIME_HAS_INDEX_SUBSPACE_CALL,
      RUNTIME_GET_INDEX_SPACE_DOMAIN_CALL,
      RUNTIME_GET_INDEX_PARTITION_COLOR_SPACE_CALL,
      RUNTIME_GET_INDEX_PARTITION_COLOR_SPACE_NAME_CALL,
      RUNTIME_GET_INDEX_SPACE_PARTITION_COLORS_CALL,
      RUNTIME_IS_INDEX_PARTITION_DISJOINT_CALL,
      RUNTIME_IS_INDEX_PARTITION_COMPLETE_CALL,
      RUNTIME_GET_INDEX_SPACE_COLOR_CALL,
      RUNTIME_GET_INDEX_SPACE_COLOR_POINT_CALL,
      RUNTIME_GET_INDEX_PARTITION_COLOR_CALL,
      RUNTIME_GET_INDEX_PARTITION_COLOR_POINT_CALL,
      RUNTIME_GET_PARENT_INDEX_SPACE_CALL,
      RUNTIME_HAS_PARENT_INDEX_PARTITION_CALL,
      RUNTIME_GET_PARENT_INDEX_PARTITION_CALL,
      RUNTIME_GET_INDEX_SPACE_DEPTH_CALL,
      RUNTIME_GET_INDEX_PARTITION_DEPTH_CALL,
      RUNTIME_SAFE_CAST_CALL,
      RUNTIME_CREATE_FIELD_SPACE_CALL,
      RUNTIME_DESTROY_FIELD_SPACE_CALL,
      RUNTIME_GET_FIELD_SIZE_CALL,
      RUNTIME_GET_FIELD_SPACE_FIELDS_CALL,
      RUNTIME_CREATE_LOGICAL_REGION_CALL,
      RUNTIME_DESTROY_LOGICAL_REGION_CALL,
      RUNTIME_RESET_EQUIVALENCE_SETS_CALL,
      RUNTIME_GET_LOGICAL_PARTITION_CALL,
      RUNTIME_GET_LOGICAL_PARTITION_BY_COLOR_CALL,
      RUNTIME_HAS_LOGICAL_PARTITION_BY_COLOR_CALL,
      RUNTIME_GET_LOGICAL_PARTITION_BY_TREE_CALL,
      RUNTIME_GET_LOGICAL_SUBREGION_CALL,
      RUNTIME_GET_LOGICAL_SUBREGION_BY_COLOR_CALL,
      RUNTIME_HAS_LOGICAL_SUBREGION_BY_COLOR_CALL,
      RUNTIME_GET_LOGICAL_SUBREGION_BY_TREE_CALL,
      RUNTIME_GET_LOGICAL_REGION_COLOR_CALL,
      RUNTIME_GET_LOGICAL_REGION_COLOR_POINT_CALL,
      RUNTIME_GET_LOGICAL_PARTITION_COLOR_CALL,
      RUNTIME_GET_LOGICAL_PARTITION_COLOR_POINT_CALL,
      RUNTIME_GET_PARENT_LOGICAL_REGION_CALL,
      RUNTIME_HAS_PARENT_LOGICAL_PARTITION_CALL,
      RUNTIME_GET_PARENT_LOGICAL_PARTITION_CALL,
      RUNTIME_CREATE_ARGUMENT_MAP_CALL,
      RUNTIME_EXECUTE_TASK_CALL,
      RUNTIME_EXECUTE_INDEX_SPACE_CALL,
      RUNTIME_REDUCE_FUTURE_MAP_CALL,
      RUNTIME_CONSTRUCT_FUTURE_MAP_CALL,
      RUNTIME_TRANSFORM_FUTURE_MAP_CALL,
      RUNTIME_MAP_REGION_CALL,
      RUNTIME_REMAP_REGION_CALL,
      RUNTIME_UNMAP_REGION_CALL,
      RUNTIME_UNMAP_ALL_REGIONS_CALL,
      RUNTIME_GET_OUTPUT_REGION_CALL,
      RUNTIME_GET_OUTPUT_REGIONS_CALL,
      RUNTIME_FILL_FIELDS_CALL,
      RUNTIME_DISCARD_FIELDS_CALL,
      RUNTIME_ATTACH_EXTERNAL_RESOURCE_CALL,
      RUNTIME_ATTACH_EXTERNAL_RESOURCES_CALL,
      RUNTIME_DETACH_EXTERNAL_RESOURCE_CALL,
      RUNTIME_DETACH_EXTERNAL_RESOURCES_CALL,
      RUNTIME_PROGRESS_UNORDERED_CALL,
      RUNTIME_ISSUE_COPY_OPERATION_CALL,
      RUNTIME_CREATE_PREDICATE_CALL,
      RUNTIME_PREDICATE_NOT_CALL,
      RUNTIME_GET_PREDICATE_FUTURE_CALL,
      RUNTIME_CREATE_LOCK_CALL,
      RUNTIME_DESTROY_LOCK_CALL,
      RUNTIME_ACQUIRE_GRANT_CALL,
      RUNTIME_RELEASE_GRANT_CALL,
      RUNTIME_CREATE_PHASE_BARRIER_CALL,
      RUNTIME_DESTROY_PHASE_BARRIER_CALL,
      RUNTIME_ADVANCE_PHASE_BARRIER_CALL,
      RUNTIME_CREATE_DYNAMIC_COLLECTIVE_CALL,
      RUNTIME_DESTROY_DYNAMIC_COLLECTIVE_CALL,
      RUNTIME_ARRIVE_DYNAMIC_COLLECTIVE_CALL,
      RUNTIME_DEFER_DYNAMIC_COLLECTIVE_CALL,
      RUNTIME_GET_DYNAMIC_COLLECTIVE_CALL,
      RUNTIME_ADVANCE_DYNAMIC_COLLECTIVE_CALL,
      RUNTIME_ISSUE_ACQUIRE_CALL,
      RUNTIME_ISSUE_RELEASE_CALL,
      RUNTIME_ISSUE_MAPPING_FENCE_CALL,
      RUNTIME_ISSUE_EXECUTION_FENCE_CALL,
      RUNTIME_BEGIN_TRACE_CALL,
      RUNTIME_END_TRACE_CALL,
      RUNTIME_COMPLETE_FRAME_CALL,
      RUNTIME_SELECT_TUNABLE_CALL,
      RUNTIME_MUST_EPOCH_CALL,
      RUNTIME_GET_LOCAL_TASK_CALL,
      RUNTIME_GET_LOCAL_VARIABLE_CALL,
      RUNTIME_SET_LOCAL_VARIABLE_CALL,
      RUNTIME_ALLOCATE_DEFERRED_VALUE_CALL,
      RUNTIME_DESTROY_DEFERRED_VALUE_CALL,
      RUNTIME_ALLOCATE_DEFERRED_BUFFER_CALL,
      RUNTIME_DESTROY_DEFERRED_BUFFER_CALL,
      RUNTIME_ISSUE_TIMING_MEASUREMENT_CALL,
      RUNTIME_GET_EXECUTING_PROCESSOR_CALL,
      RUNTIME_GET_CURRENT_TASK_CALL,
      RUNTIME_QUERY_AVAILABLE_MEMORY_CALL,
      RUNTIME_RAISE_REGION_EXCEPTION_CALL,
      RUNTIME_YIELD_CALL,
      RUNTIME_ASYNC_EFFECT_CALL,
      RUNTIME_CONCURRENT_TASK_BARRIER_CALL,
      RUNTIME_PRINT_ONCE_CALL,
      RUNTIME_LOG_ONCE_CALL,
      RUNTIME_CONSENSUS_MATCH_CALL,
      RUNTIME_GET_MAPPER_CALL,
      RUNTIME_PUSH_EXCEPTION_HANDLER_CALL,
      RUNTIME_POP_EXCEPTION_HANDLER_CALL,
      // Mapper runtime call kinds
      MAPPER_SEND_MESSAGE_CALL,
      MAPPER_BROADCAST_CALL,
      MAPPER_UNPACK_INSTANCE_CALL,
      MAPPER_CREATE_EVENT_CALL,
      MAPPER_HAS_TRIGGERED_CALL,
      MAPPER_TRIGGER_EVENT_CALL,
      MAPPER_WAIT_EVENT_CALL,
      MAPPER_FIND_EXECUTION_CONSTRAINTS_CALL,
      MAPPER_FIND_TASK_LAYOUT_CONSTRAINTS_CALL,
      MAPPER_FIND_LAYOUT_CONSTRAINTS_CALL,
      MAPPER_REGISTER_LAYOUT_CALL,
      MAPPER_RELEASE_LAYOUT_CALL,
      MAPPER_CONSTRAINTS_CONFLICT_CALL,
      MAPPER_CONSTRAINTS_ENTAIL_CALL,
      MAPPER_FIND_VALID_VARIANTS_CALL,
      MAPPER_FIND_TASK_VARIANT_NAME_CALL,
      MAPPER_IS_LEAF_VARIANT_CALL,
      MAPPER_IS_INNER_VARIANT_CALL,
      MAPPER_IS_IDEMPOTENT_VARIANT_CALL,
      MAPPER_IS_REPLICABLE_VARIANT_CALL,
      MAPPER_REGISTER_TASK_VARIANT_CALL,
      MAPPER_FILTER_VARIANTS_CALL,
      MAPPER_FILTER_INSTANCES_CALL,
      MAPPER_CREATE_PHYSICAL_INSTANCE_CALL,
      MAPPER_FIND_OR_CREATE_PHYSICAL_INSTANCE_CALL,
      MAPPER_REDISTRICT_INSTANCE_CALL,
      MAPPER_FIND_PHYSICAL_INSTANCE_CALL,
      MAPPER_FIND_PHYSICAL_INSTANCES_CALL,
      MAPPER_SET_GC_PRIORITY_CALL,
      MAPPER_ACQUIRE_INSTANCE_CALL,
      MAPPER_ACQUIRE_INSTANCES_CALL,
      MAPPER_ACQUIRE_AND_FILTER_INSTANCES_CALL,
      MAPPER_RELEASE_INSTANCE_CALL,
      MAPPER_RELEASE_INSTANCES_CALL,
      MAPPER_SUBSCRIBE_INSTANCE_CALL,
      MAPPER_UNSUBSCRIBE_INSTANCE_CALL,
      MAPPER_COLLECT_INSTANCE_CALL,
      MAPPER_COLLECT_INSTANCES_CALL,
      MAPPER_ACQUIRE_FUTURE_CALL,
      MAPPER_ACQUIRE_POOL_CALL,
      MAPPER_RELEASE_POOL_CALL,
      MAPPER_CREATE_INDEX_SPACE_CALL,
      MAPPER_UNION_INDEX_SPACES_CALL,
      MAPPER_INTERSECT_INDEX_SPACES_CALL,
      MAPPER_SUBTRACT_INDEX_SPACES_CALL,
      MAPPER_INDEX_SPACE_EMPTY_CALL,
      MAPPER_INDEX_SPACES_OVERLAP_CALL,
      MAPPER_INDEX_SPACE_DOMINATES_CALL,
      MAPPER_HAS_INDEX_PARTITION_CALL,
      MAPPER_GET_INDEX_PARTITION_CALL,
      MAPPER_GET_INDEX_SUBSPACE_CALL,
      MAPPER_GET_INDEX_SPACE_DOMAIN_CALL,
      MAPPER_GET_INDEX_PARTITION_CS_CALL,
      MAPPER_GET_INDEX_PARTITION_CS_NAME_CALL,
      MAPPER_GET_INDEX_SPACE_PARTITION_COLORS_CALL,
      MAPPER_IS_INDEX_PARTITION_DISJOINT_CALL,
      MAPPER_IS_INDEX_PARTITION_COMPLETE_CALL,
      MAPPER_GET_INDEX_SPACE_COLOR_CALL,
      MAPPER_GET_INDEX_SPACE_COLOR_POINT_CALL,
      MAPPER_GET_INDEX_PARTITION_COLOR_CALL,
      MAPPER_GET_PARENT_INDEX_SPACE_CALL,
      MAPPER_HAS_PARENT_INDEX_PARTITION_CALL,
      MAPPER_GET_PARENT_INDEX_PARTITION_CALL,
      MAPPER_GET_INDEX_SPACE_DEPTH_CALL,
      MAPPER_GET_INDEX_PARTITION_DEPTH_CALL,
      MAPPER_GET_FIELD_SIZE_CALL,
      MAPPER_GET_FIELD_SPACE_FIELDS_CALL,
      MAPPER_GET_LOGICAL_PARTITION_CALL,
      MAPPER_GET_LOGICAL_PARTITION_BY_COLOR_CALL,
      MAPPER_GET_LOGICAL_PARTITION_BY_TREE_CALL,
      MAPPER_GET_LOGICAL_SUBREGION_CALL,
      MAPPER_GET_LOGICAL_SUBREGION_BY_COLOR_CALL,
      MAPPER_GET_LOGICAL_SUBREGION_BY_TREE_CALL,
      MAPPER_GET_LOGICAL_REGION_COLOR_CALL,
      MAPPER_GET_LOGICAL_REGION_COLOR_POINT_CALL,
      MAPPER_GET_LOGICAL_PARTITION_COLOR_CALL,
      MAPPER_GET_PARENT_LOGICAL_REGION_CALL,
      MAPPER_HAS_PARENT_LOGICAL_PARTITION_CALL,
      MAPPER_GET_PARENT_LOGICAL_PARTITION_CALL,
      MAPPER_RETRIEVE_SEMANTIC_INFO_CALL,
      MAPPER_RETRIEVE_NAME_CALL,
      MAPPER_AUTO_LOCK_CALL,
      MAPPER_FIND_COLLECTIVE_INSTANCES_IN_MEMORY,
      MAPPER_FIND_COLLECTIVE_INSTANCES_NEAREST_MEMORY,
      // Old runtime call kinds
      PACK_BASE_TASK_CALL,
      UNPACK_BASE_TASK_CALL,
      TASK_PRIVILEGE_CHECK_CALL,
      CLONE_TASK_CALL,
      COMPUTE_POINT_REQUIREMENTS_CALL,
      INTRA_TASK_ALIASING_CALL,
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
      INDEX_COMPUTE_FAT_PATH_CALL,
      INDEX_PREMAP_TASK_CALL,
      INDEX_DISTRIBUTE_CALL,
      INDEX_PERFORM_MAPPING_CALL,
      INDEX_COMPLETE_CALL,
      INDEX_COMMIT_CALL,
      INDEX_PERFORM_INLINING_CALL,
      INDEX_CLONE_AS_SLICE_CALL,
      INDEX_HANDLE_FUTURE,
      INDEX_RETURN_SLICE_MAPPED_CALL,
      INDEX_RETURN_SLICE_COMPLETE_CALL,
      INDEX_RETURN_SLICE_COMMIT_CALL,
      SLICE_ACTIVATE_CALL,
      SLICE_DEACTIVATE_CALL,
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
      REGION_TREE_VERSIONING_ANALYSIS_CALL,
      REGION_TREE_ADVANCE_VERSION_NUMBERS_CALL,
      REGION_TREE_INITIALIZE_CONTEXT_CALL,
      REGION_TREE_INVALIDATE_CONTEXT_CALL,
      REGION_TREE_PREMAP_ONLY_CALL,
      REGION_TREE_PHYSICAL_REGISTER_ONLY_CALL,
      REGION_TREE_PHYSICAL_REGISTER_USERS_CALL,
      REGION_TREE_PHYSICAL_PERFORM_CLOSE_CALL,
      REGION_TREE_PHYSICAL_CLOSE_CONTEXT_CALL,
      REGION_TREE_PHYSICAL_COPY_ACROSS_CALL,
      REGION_TREE_PHYSICAL_REDUCE_ACROSS_CALL,
      REGION_TREE_PHYSICAL_CONVERT_MAPPING_CALL,
      REGION_TREE_PHYSICAL_FILL_FIELDS_CALL,
      REGION_TREE_PHYSICAL_ATTACH_EXTERNAL_CALL,
      REGION_TREE_PHYSICAL_DETACH_EXTERNAL_CALL,
      REGION_NODE_REGISTER_LOGICAL_USER_CALL,
      REGION_NODE_CLOSE_LOGICAL_NODE_CALL,
      REGION_NODE_SIPHON_LOGICAL_CHILDREN_CALL,
      REGION_NODE_SIPHON_LOGICAL_PROJECTION_CALL,
      REGION_NODE_PERFORM_LOGICAL_CLOSES_CALL,
      REGION_NODE_FIND_VALID_INSTANCE_VIEWS_CALL,
      REGION_NODE_FIND_VALID_REDUCTION_VIEWS_CALL,
      REGION_NODE_ISSUE_UPDATE_COPIES_CALL,
      REGION_NODE_SORT_COPY_INSTANCES_CALL,
      REGION_NODE_ISSUE_GROUPED_COPIES_CALL,
      REGION_NODE_ISSUE_UPDATE_REDUCTIONS_CALL,
      REGION_NODE_PREMAP_REGION_CALL,
      REGION_NODE_REGISTER_REGION_CALL,
      REGION_NODE_CLOSE_STATE_CALL,
      CURRENT_STATE_RECORD_VERSION_NUMBERS_CALL,
      CURRENT_STATE_ADVANCE_VERSION_NUMBERS_CALL,
      PHYSICAL_STATE_CAPTURE_STATE_CALL,
      PHYSICAL_STATE_APPLY_PATH_ONLY_CALL,
      PHYSICAL_STATE_APPLY_STATE_CALL,
      PHYSICAL_STATE_MAKE_LOCAL_CALL,
      MATERIALIZED_VIEW_FIND_LOCAL_PRECONDITIONS_CALL,
      MATERIALIZED_VIEW_FIND_LOCAL_COPY_PRECONDITIONS_CALL,
      MATERIALIZED_VIEW_FILTER_PREVIOUS_USERS_CALL,
      MATERIALIZED_VIEW_FILTER_CURRENT_USERS_CALL,
      MATERIALIZED_VIEW_FILTER_LOCAL_USERS_CALL,
      REDUCTION_VIEW_PERFORM_REDUCTION_CALL,
      REDUCTION_VIEW_PERFORM_DEFERRED_REDUCTION_CALL,
      REDUCTION_VIEW_PERFORM_DEFERRED_REDUCTION_ACROSS_CALL,
      REDUCTION_VIEW_FIND_COPY_PRECONDITIONS_CALL,
      REDUCTION_VIEW_FIND_USER_PRECONDITIONS_CALL,
      REDUCTION_VIEW_FILTER_LOCAL_USERS_CALL,
      PHYSICAL_TRACE_EXECUTE_CALL,
      PHYSICAL_TRACE_PRECONDITION_CHECK_CALL,
      PHYSICAL_TRACE_OPTIMIZE_CALL,
      LAST_RUNTIME_CALL_KIND,  // This one must be last
    };

#define RUNTIME_CALL_DESCRIPTIONS(name)                      \
  const char* name[Internal::LAST_RUNTIME_CALL_KIND] = {     \
      "Runtime::create_index_space",                         \
      "Runtime::union_index_space",                          \
      "Runtime::intersect_index_spaces",                     \
      "Runtime::subtract_index_spaces",                      \
      "Runtime::create_shared_ownership",                    \
      "Runtime::destroy_index_space",                        \
      "Runtime::destroy_index_partition",                    \
      "Runtime::create_equal_partition",                     \
      "Runtime::create_partition_by_weights",                \
      "Runtime::create_partition_by_union",                  \
      "Runtime::create_partition_by_intersection",           \
      "Runtime::create_partition_by_difference",             \
      "Runtime::create_cross_product_partitions",            \
      "Runtime::create_association",                         \
      "Runtime::create_partition_by_restriction",            \
      "Runtime::create_partition_by_blockify",               \
      "Runtime::create_partition_by_domain",                 \
      "Runtime::create_partition_by_field",                  \
      "Runtime::create_partition_by_image",                  \
      "Runtime::create_partition_by_image_range",            \
      "Runtime::create_partition_by_preimage",               \
      "Runtime::create_partition_by_preimage_range",         \
      "Runtime::create_pending_partition",                   \
      "Runtime::create_index_space_union",                   \
      "Runtime::create_index_space_intersection",            \
      "Runtime::create_index_space_difference",              \
      "Runtime::get_index_partition",                        \
      "Runtime::has_index_partition",                        \
      "Runtime::get_index_subspace",                         \
      "Runtime::has_index_subspace",                         \
      "Runtime::get_index_space_domain",                     \
      "Runtime::get_index_partition_color_space",            \
      "Runtime::get_index_partition_color_space_name",       \
      "Runtime::get_index_space_partition_colors",           \
      "Runtime::is_index_partition_disjoint",                \
      "Runtime::is_index_partition_complete",                \
      "Runtime::get_index_space_color",                      \
      "Runtime::get_index_space_color_point",                \
      "Runtime::get_index_partition_color",                  \
      "Runtime::get_index_partition_color_point",            \
      "Runtime::get_parent_index_space",                     \
      "Runtime::has_parent_index_partition",                 \
      "Runtime::get_parent_index_partition",                 \
      "Runtime::get_index_space_depth",                      \
      "Runtime::get_index_partition_depth",                  \
      "Runtime::safe_cast",                                  \
      "Runtime::create_field_space",                         \
      "Runtime::destroy_field_space",                        \
      "Runtime::get_field_size",                             \
      "Runtime::get_field_space_fields",                     \
      "Runtime::create_logical_region",                      \
      "Runtime::destroy_logical_region",                     \
      "Runtime::reset_equivalence_sets",                     \
      "Runtime::get_logical_partition",                      \
      "Runtime::get_logical_partition_by_color",             \
      "Runtime::has_logical_partition_by_color",             \
      "Runtime::get_logical_partition_by_tree",              \
      "Runtime::get_logical_subregion",                      \
      "Runtime::get_logical_subregion_by_color",             \
      "Runtime::has_logical_subregion_by_color",             \
      "Runtime::get_logical_subregion_by_tree",              \
      "Runtime::get_logical_region_color",                   \
      "Runtime::get_logical_region_color_point",             \
      "Runtime::get_logical_partition_color",                \
      "Runtime::get_logical_partition_color_point",          \
      "Runtime::get_parent_logical_region",                  \
      "Runtime::has_parent_logical_partition",               \
      "Runtime::get_parent_logical_partition",               \
      "Runtime::create_argument_map",                        \
      "Runtime::execute_task",                               \
      "Runtime::execute_index_space",                        \
      "Runtime::reduce_future_map",                          \
      "Runtime::construct_future_map",                       \
      "Runtime::transform_future_map",                       \
      "Runtime::map_region",                                 \
      "Runtime::remap_region",                               \
      "Runtime::unmap_region",                               \
      "Runtime::unmap_all_regions",                          \
      "Runtime::get_output_region",                          \
      "Runtime::get_output_regions",                         \
      "Runtime::fill_fields",                                \
      "Runtime::discard_fields",                             \
      "Runtime::attach_external_resource",                   \
      "Runtime::attach_external_resources",                  \
      "Runtime::detach_external_resource",                   \
      "Runtime::detach_external_resources",                  \
      "Runtime::progress_unordered_operations",              \
      "Runtime::issue_copy_operation",                       \
      "Runtime::create_predicate",                           \
      "Runtime::predicate_not",                              \
      "Runtime::get_predicate_future",                       \
      "Runtime::create_lock",                                \
      "Runtime::destroy_lock",                               \
      "Runtime::acquire_grant",                              \
      "Runtime::release_grant",                              \
      "Runtime::create_phase_barrier",                       \
      "Runtime::destroy_phase_barrier",                      \
      "Runtime::advance_phase_barrier",                      \
      "Runtime::create_dynamic_collective",                  \
      "Runtime::destroy_dynamic_collective",                 \
      "Runtime::arrive_dynamic_collective",                  \
      "Runtime::defer_dynamic_collective_arrival",           \
      "Runtime::get_dynamic_collective_result",              \
      "Runtime::advance_dynamic_collective",                 \
      "Runtime::issue_acquire",                              \
      "Runtime::issue_release",                              \
      "Runtime::issue_mapping_fence",                        \
      "Runtime::issue_execution_fence",                      \
      "Runtime::begin_trace",                                \
      "Runtime::end_trace",                                  \
      "Runtime::complete_frame",                             \
      "Runtime::execute_must_epoch",                         \
      "Runtime::select_tunable_value",                       \
      "Runtime::get_local_task",                             \
      "Runtime::set_local_task_variable",                    \
      "Runtime::get_local_task_variable",                    \
      "Runtime::allocate_deferred_value",                    \
      "Runtime::destroy_deferred_value",                     \
      "Runtime::allocate_deferred_buffer",                   \
      "Runtime::destroy_deferred_buffer",                    \
      "Runtime::issue_timing_measurement",                   \
      "Runtime::get_executing_processor",                    \
      "Runtime::get_current_task",                           \
      "Runtime::query_available_memory",                     \
      "Runtime::raise_region_exception",                     \
      "Runtime::yield",                                      \
      "Runtime::record_asynchronous_effect",                 \
      "Runtime::concurrent_task_barrier",                    \
      "Runtime::print_once",                                 \
      "Runtime::log_once",                                   \
      "Runtime::consensus_match",                            \
      "Runtime::get_mapper",                                 \
      "Runtime::push_exception_handler",                     \
      "Runtime::pop_exception_handler",                      \
      "MapperRuntime::send_message",                         \
      "MapperRuntime::broadcast",                            \
      "MapperRuntime::unpack_physical_instance",             \
      "MapperRuntime::create_mapper_event",                  \
      "MapperRuntime::has_mapper_event_triggered",           \
      "MapperRuntime::trigger_mapper_event",                 \
      "MapperRuntime::wait_on_mapper_event",                 \
      "MapperRuntime::find_execution_constraints",           \
      "MapperRuntime::find_task_layout_constraints",         \
      "MapperRuntime::find_layout_constraints",              \
      "MapperRuntime::register_layout",                      \
      "MapperRuntime::release_layout",                       \
      "MapperRuntime::do_constraints_conflict",              \
      "MapperRuntime::do_constraints_entail",                \
      "MapperRuntime::find_valid_variants",                  \
      "MapperRuntime::find_task_variant_name",               \
      "MapperRuntime::is_leaf_variant",                      \
      "MapperRuntime::is_inner_variant",                     \
      "MapperRuntime::is_idempotent_variant",                \
      "MapperRuntime::is_replicable_variant",                \
      "MapperRuntime::register_task_variant",                \
      "MapperRuntime::filter_variants",                      \
      "MapperRuntime::filter_instances",                     \
      "MapperRuntime::create_physical_instance",             \
      "MapperRuntime::find_or_create_physical_instance",     \
      "MapperRuntime::redistrict_instance",                  \
      "MapperRuntime::find_physical_instance",               \
      "MapperRuntime::find_physical_instances",              \
      "MapperRuntime::set_garbage_collection_priority",      \
      "MapperRuntime::acquire_instance",                     \
      "MapperRuntime::acquire_instances",                    \
      "MapperRuntime::acquire_and_filter_instances",         \
      "MapperRuntime::release_instance",                     \
      "MapperRuntime::release_instances",                    \
      "MapperRuntime::subscribe",                            \
      "MapperRuntime::unsubscribe",                          \
      "MapperRuntime::collect_instance",                     \
      "MapperRuntime::collect_instances",                    \
      "MapperRuntime::acquire_future",                       \
      "MapperRuntime::acquire_pool",                         \
      "MapperRuntime::release_pool",                         \
      "MapperRuntime::create_index_space",                   \
      "MapperRuntime::union_index_spaces",                   \
      "MapperRuntime::intersect_index_spaces",               \
      "MapperRuntime::subtract_index_spaces",                \
      "MapperRuntime::is_index_space_empty",                 \
      "MapperRuntime::index_spaces_overlap",                 \
      "MapperRuntime::index_space_dominates",                \
      "MapperRuntime::has_index_partition",                  \
      "MapperRuntime::get_index_partition",                  \
      "MapperRuntime::get_index_subspace",                   \
      "MapperRuntime::get_index_space_domain",               \
      "MapperRuntime::get_index_partition_color_space",      \
      "MapperRuntime::get_index_partition_color_space_name", \
      "MapperRuntime::get_index_space_parition_colors",      \
      "MapperRuntime::is_index_partition_disjoint",          \
      "MapperRuntime::is_index_partition_complete",          \
      "MapperRuntime::get_index_space_color",                \
      "MapperRuntime::get_index_space_color_point",          \
      "MapperRuntime::get_index_partition_color",            \
      "MapperRuntime::get_parent_index_space",               \
      "MapperRuntime::has_parent_index_partition",           \
      "MapperRuntime::get_parent_index_partition",           \
      "MapperRuntime::get_index_space_depth",                \
      "MapperRuntime::get_index_partition_depth",            \
      "MapperRuntime::get_field_size",                       \
      "MapperRuntime::get_field_space_fields",               \
      "MapperRuntime::get_logical_partition",                \
      "MapperRuntime::get_logical_partition_by_color",       \
      "MapperRuntime::get_logical_partition_by_tree",        \
      "MapperRuntime::get_logical_subregion",                \
      "MapperRuntime::get_logical_subregion_by_color",       \
      "MapperRuntime::get_logical_subregion_by_tree",        \
      "MapperRuntime::get_logical_region_color",             \
      "MapperRuntime::get_logical_region_color_point",       \
      "MapperRuntime::get_logical_partition_color",          \
      "MapperRuntime::get_parent_logical_region",            \
      "MapperRuntime::has_parent_logical_partition",         \
      "MapperRuntime::get_parent_logical_partition",         \
      "MapperRuntime::retrieve_semantic_information",        \
      "MapperRuntime::retrieve_name",                        \
      "MapperRuntime::AutoLock",                             \
      "CollectiveView::find_instances_in_memory",            \
      "CollectiveView::find_instances_nearest_memory",       \
      "Pack Base Task",                                      \
      "Unpack Base Task",                                    \
      "Task Privilege Check",                                \
      "Clone Base Task",                                     \
      "Compute Point Requirements",                          \
      "Intra-Task Aliasing",                                 \
      "Activate Single",                                     \
      "Deactivate Single",                                   \
      "Select Inline Variant",                               \
      "Inline Child Task",                                   \
      "Pack Single Task",                                    \
      "Unpack Single Task",                                  \
      "Pack Remote Context",                                 \
      "Has Conflicting Internal",                            \
      "Find Conflicting",                                    \
      "Find Conflicting Internal",                           \
      "Check Region Dependence",                             \
      "Find Parent Region Requirement",                      \
      "Find Parent Region",                                  \
      "Check Privilege",                                     \
      "Trigger Single",                                      \
      "Initialize Map Task",                                 \
      "Finalized Map Task",                                  \
      "Validate Variant Selection",                          \
      "Map All Regions",                                     \
      "Initialize Region Tree Contexts",                     \
      "Invalidate Region Tree Contexts",                     \
      "Create Instance Top View",                            \
      "Launch Task",                                         \
      "Activate Multi",                                      \
      "Deactivate Multi",                                    \
      "Slice Index Space",                                   \
      "Clone Multi Call",                                    \
      "Multi Trigger Execution",                             \
      "Pack Multi",                                          \
      "Unpack Multi",                                        \
      "Activate Individual",                                 \
      "Deactivate Individual",                               \
      "Individual Perform Mapping",                          \
      "Individual Return Virtual",                           \
      "Individual Trigger Complete",                         \
      "Individual Trigger Commit",                           \
      "Individual Post Mapped",                              \
      "Individual Pack Task",                                \
      "Individual Unpack Task",                              \
      "Individual Pack Remote Complete",                     \
      "Individual Unpack Remote Complete",                   \
      "Activate Point",                                      \
      "Deactivate Point",                                    \
      "Point Task Complete",                                 \
      "Point Task Commit",                                   \
      "Point Task Pack",                                     \
      "Point Task Unpack",                                   \
      "Point Task Post Mapped",                              \
      "Remote Task Activate",                                \
      "Remote Task Deactivate",                              \
      "Remote Unpack Context",                               \
      "Index Activate",                                      \
      "Index Deactivate",                                    \
      "Index Compute Fat Path",                              \
      "Index PreMap Task",                                   \
      "Index Distribute",                                    \
      "Index Perform Mapping",                               \
      "Index Complete",                                      \
      "Index Commit",                                        \
      "Index Perform Inlining",                              \
      "Index Clone As Slice",                                \
      "Index Handle Future",                                 \
      "Index Return Slice Mapped",                           \
      "Index Return Slice Complete",                         \
      "Index Return Slice Commit",                           \
      "Slice Activate",                                      \
      "Slice Deactivate",                                    \
      "Slice Apply Version Info",                            \
      "Slice Distribute",                                    \
      "Slice Perform Mapping",                               \
      "Slice Launch",                                        \
      "Slice Map and Launch",                                \
      "Slice Pack Task",                                     \
      "Slice Unpack Task",                                   \
      "Slice Clone As Slice",                                \
      "Slice Handle Future",                                 \
      "Slice Cone as Point",                                 \
      "Slice Enumerate Points",                              \
      "Slice Mapped",                                        \
      "Slice Complete",                                      \
      "Slice Commit",                                        \
      "Realm Spawn Meta",                                    \
      "Realm Spawn Task",                                    \
      "Realm Create Instance",                               \
      "Realm Issue Copy",                                    \
      "Realm Issue Fill",                                    \
      "Region Tree Logical Analysis",                        \
      "Region Tree Logical Fence",                           \
      "Region Tree Versioning Analysis",                     \
      "Region Tree Advance Version Numbers",                 \
      "Region Tree Initialize Context",                      \
      "Region Tree Invalidate Context",                      \
      "Region Tree Premap Only",                             \
      "Region Tree Physical Register Only",                  \
      "Region Tree Physical Register Users",                 \
      "Region Tree Physical Perform Close",                  \
      "Region Tree Physical Close Context",                  \
      "Region Tree Physical Copy Across",                    \
      "Region Tree Physical Reduce Across",                  \
      "Region Tree Physical Convert Mapping",                \
      "Region Tree Physical Fill Fields",                    \
      "Region Tree Physical Attach External",                \
      "Region Tree Physical Detach External",                \
      "Region Node Register Logical User",                   \
      "Region Node Close Logical Node",                      \
      "Region Node Siphon Logical Children",                 \
      "Region Node Siphon Logical Projection",               \
      "Region Node Perform Logical Closes",                  \
      "Region Node Find Valid Instance Views",               \
      "Region Node Find Valid Reduction Views",              \
      "Region Node Issue Update Copies",                     \
      "Region Node Sort Copy Instances",                     \
      "Region Node Issue Grouped Copies",                    \
      "Region Node Issue Update Reductions",                 \
      "Region Node Premap Region",                           \
      "Region Node Register Region",                         \
      "Region Node Close State",                             \
      "Logical State Record Verison Numbers",                \
      "Logical State Advance Version Numbers",               \
      "Physical State Capture State",                        \
      "Physical State Apply Path Only",                      \
      "Physical State Apply State",                          \
      "Physical State Make Local",                           \
      "Materialized View Find Local Preconditions",          \
      "Materialized View Find Local Copy Preconditions",     \
      "Materialized View Filter Previous Users",             \
      "Materialized View Filter Current Users",              \
      "Materialized View Filter Local Users",                \
      "Reduction View Perform Reduction",                    \
      "Reduction View Perform Deferred Reduction",           \
      "Reduction View Perform Deferred Reduction Across",    \
      "Reduction View Find Copy Preconditions",              \
      "Reduction View Find User Preconditions",              \
      "Reduction View Filter Local Users",                   \
      "Physical Trace Execute",                              \
      "Physical Trace Precondition Check",                   \
      "Physical Trace Optimize",                             \
  };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_TOOL_TYPES_H__
