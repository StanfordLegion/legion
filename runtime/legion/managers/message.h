/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_MESSAGE_MANAGER_H__
#define __LEGION_MESSAGE_MANAGER_H__

#include "legion/kernel/metatask.h"
#include "legion/utilities/serdez.h"

namespace Legion {
  namespace Internal {

    enum VirtualChannelKind {
      // The default and work virtual channels are unordered
      DEFAULT_VIRTUAL_CHANNEL = 0,     // latency priority
      THROUGHPUT_VIRTUAL_CHANNEL = 1,  // throughput priority
      LAST_UNORDERED_VIRTUAL_CHANNEL = THROUGHPUT_VIRTUAL_CHANNEL,
      // All the rest of these are ordered (latency-priority) channels
      MAPPER_VIRTUAL_CHANNEL = 1,
      TASK_VIRTUAL_CHANNEL = 2,
      INDEX_SPACE_VIRTUAL_CHANNEL = 3,
      FIELD_SPACE_VIRTUAL_CHANNEL = 4,
      REFERENCE_VIRTUAL_CHANNEL = 6,
      UPDATE_VIRTUAL_CHANNEL = 7,  // deferred-priority
      POOL_VIRTUAL_CHANNEL = 8,
      COLLECTIVE_VIRTUAL_CHANNEL = 9,
      LAYOUT_CONSTRAINT_VIRTUAL_CHANNEL = 10,
      EXPRESSION_VIRTUAL_CHANNEL = 11,
      MIGRATION_VIRTUAL_CHANNEL = 12,
      TRACING_VIRTUAL_CHANNEL = 13,
      RENDEZVOUS_VIRTUAL_CHANNEL = 14,
      PROFILING_VIRTUAL_CHANNEL = 15,
      MAX_NUM_VIRTUAL_CHANNELS = 16,  // this one must be last
    };

    // clang-format off
    // All the different kinds of active mssages 
    // enum, type, name, response, escape context, escape op
#define LEGION_ACTIVE_MESSAGES(__op__) \
  __op__(SEND_STARTUP_BARRIER, StartupBarrierMessage, "Startup Barrier Message", false, true, true) \
  __op__(TASK_MESSAGE, TaskMessage, "Distribute Task Message", false, true, true) \
  __op__(STEAL_MESSAGE, StealTaskMessage, "Steal Task Message", false, true, true) \
  __op__(ADVERTISEMENT_MESSAGE, AdvertiseTaskMessage, "Advertise Task Message", false, true, true) \
  __op__(SEND_REGISTRATION_CALLBACK, RegistrationCallbackMessage, "Registration Callback Message", false, true, true) \
  __op__(SEND_REMOTE_TASK_REPLAY, RemoteTaskReplay, "Remote Task Replay Message", false, false, false) \
  __op__(SEND_REMOTE_TASK_PROFILING_RESPONSE, RemoteTaskProfilingResponse, "Remote Task Profiling Response", false, false, false) \
  __op__(SEND_SHARED_OWNERSHIP, SharedOwnershipMessage, "Shared Ownership Message", false, false, true) \
  __op__(SEND_INDEX_SPACE_REQUEST, IndexSpaceRequest, "Index Space Request", false, false, false) \
  __op__(SEND_INDEX_SPACE_RESPONSE, IndexSpaceResponse, "Index Space Response", true, false, false) \
  __op__(SEND_INDEX_SPACE_RETURN, IndexSpaceReturn, "Index Space Return Message", true, false, false) \
  __op__(SEND_INDEX_SPACE_SET, IndexSpaceSet, "Index Space Set Message", false, true, true) \
  __op__(SEND_INDEX_SPACE_CHILD_REQUEST, IndexSpaceChildRequest, "Index Space Child Request", false, false, false) \
  __op__(SEND_INDEX_SPACE_CHILD_RESPONSE, IndexSpaceChildResponse, "Index Space Child Response", true, false, false) \
  __op__(SEND_INDEX_SPACE_COLORS_REQUEST, IndexSpaceColorsRequest, "Index Space Colors Request", false, false, false) \
  __op__(SEND_INDEX_SPACE_COLORS_RESPONSE, IndexSpaceColorsResponse, "Index Space Colors Response", true, false, false) \
  __op__(SEND_INDEX_SPACE_GENERATE_COLOR_REQUEST, IndexSpaceGenerateColorRequest, "Index Space Generate Color Request", false, false, false) \
  __op__(SEND_INDEX_SPACE_GENERATE_COLOR_RESPONSE, IndexSpaceGenerateColorResponse, "Index Space Generate Color Response", true, false, false) \
  __op__(SEND_INDEX_SPACE_RELEASE_COLOR, IndexSpaceReleaseColor, "Index Space Release Color Message", false, true, true) \
  __op__(SEND_INDEX_PARTITION_NOTIFICATION, IndexPartitionNotification, "Index Partition Noitification Message", false, true, true) \
  __op__(SEND_INDEX_PARTITION_REQUEST, IndexPartitionRequest, "Index Partition Request", false, false, false) \
  __op__(SEND_INDEX_PARTITION_RESPONSE, IndexPartitionResponse, "Index Partition Response", false, false, false) \
  __op__(SEND_INDEX_PARTITION_RETURN, IndexPartitionReturn, "Index Partition Return", false, false, false) \
  __op__(SEND_INDEX_PARTITION_CHILD_REQUEST, IndexPartitionChildRequest, "Index Partition Child Request", false, false, false) \
  __op__(SEND_INDEX_PARTITION_CHILD_RESPONSE, IndexPartitionChildResponse, "Index Partition Child Response", true, false, false) \
  __op__(SEND_INDEX_PARTITION_CHILD_REPLICATION, IndexPartitionChildReplication, "Index Partition Child Replication Message", false, true, true) \
  __op__(SEND_INDEX_PARTITION_DISJOINT_UPDATE, IndexPartitionDisjointUpdate, "Index Partition Disjoint Update Message", false, true, true) \
  __op__(SEND_INDEX_PARTITION_SHARD_RECTS_REQUEST, IndexPartitionShardRectsRequest, "Index Partition Shard Rectangles Request", false, false, false) \
  __op__(SEND_INDEX_PARTITION_SHARD_RECTS_RESPONSE, IndexPartitionShardRectsResponse, "Index Partitino Shard Rectangles Response", true, false, false) \
  __op__(SEND_INDEX_PARTITION_REMOTE_INTERFERENCE_REQUEST, IndexPartitionRemoteInterferenceRequest, "Index Partition Remote Interference Request", false, false, false) \
  __op__(SEND_INDEX_PARTITION_REMOTE_INTERFERENCE_RESPONSE, IndexPartitionRemoteInterferenceResponse, "Index Partition Remote Interference Response", true, false, false) \
  __op__(SEND_FIELD_SPACE_NODE, FieldSpaceNodeMessage, "Field Space Message", false, false, false) \
  __op__(SEND_FIELD_SPACE_REQUEST, FieldSpaceRequest, "Field Space Request", false, false, false) \
  __op__(SEND_FIELD_SPACE_RETURN, FieldSpaceReturn, "Field Space Return", true, false, false) \
  __op__(SEND_FIELD_SPACE_ALLOCATOR_REQUEST, FieldSpaceAllocatorRequest, "Field Space Allocator Request", false, false, false) \
  __op__(SEND_FIELD_SPACE_ALLOCATOR_RESPONSE, FieldSpaceAllocatorResponse, "Field Space Allocator Response", true, false, false) \
  __op__(SEND_FIELD_SPACE_ALLOCATOR_INVALIDATION, FieldSpaceAllocatorInvalidation, "Field Space Allocator Invalidation", false, true, true) \
  __op__(SEND_FIELD_SPACE_ALLOCATOR_FLUSH, FieldSpaceAllocatorFlush, "Field Space Allocator Flush", true, true, true) \
  __op__(SEND_FIELD_SPACE_ALLOCATOR_FREE, FieldSpaceAllocatorFree, "Field Space Allocator Free", true, true, true) \
  __op__(SEND_FIELD_SPACE_INFOS_REQUEST, FieldSpaceInfosRequest, "Field Space Information Request", false, false, false) \
  __op__(SEND_FIELD_SPACE_INFOS_RESPONSE, FieldSpaceInfosResponse, "Field Space Information Response", true, false, false) \
  __op__(SEND_FIELD_ALLOC_REQUEST, FieldAllocationRequest, "Field Allocation Request", false, false, false) \
  __op__(SEND_FIELD_SIZE_UPDATE, FieldSizeUpdate, "Field Size Update", true, true, true) \
  __op__(SEND_FIELD_FREE, FieldFreeMessage, "Field Free Message", true, true, true) \
  __op__(SEND_FIELD_FREE_INDEXES, FieldFreeIndexes, "Free Field Indexes Message", true, true, true) \
  __op__(SEND_FIELD_SPACE_LAYOUT_INVALIDATION, FieldSpaceLayoutInvalidation, "Invalidate Field Space Layouts Message", true, true, true) \
  __op__(SEND_LOCAL_FIELD_ALLOC_REQUEST, LocalFieldAllocRequest, "Local Field Allocation Request", false, false, false) \
  __op__(SEND_LOCAL_FIELD_ALLOC_RESPONSE, LocalFieldAllocResponse, "Local Field Allocation Response", true, false, false) \
  __op__(SEND_LOCAL_FIELD_FREE, LocalFieldFreeMessage, "Free Local Field Message", true, true, true) \
  __op__(SEND_LOCAL_FIELD_UPDATE, LocalFieldUpdateMessage, "Update Local Field Message", true, true, true) \
  __op__(SEND_TOP_LEVEL_REGION_REQUEST, TopLevelRegionRequest, "Top Level Region Request", false, false, false) \
  __op__(SEND_TOP_LEVEL_REGION_RETURN, TopLevelRegionReturn, "Top Level Region Response", true, false, false) \
  __op__(INDEX_SPACE_DESTRUCTION_MESSAGE, IndexSpaceDestruction, "Destroy Index Space Message", false, true, true) \
  __op__(INDEX_PARTITION_DESTRUCTION_MESSAGE, IndexPartitionDestruction, "Destroy Index Partition Message", false, true, true) \
  __op__(FIELD_SPACE_DESTRUCTION_MESSAGE, FieldSpaceDestruction, "Destroy Field Space Message", false, true, true) \
  __op__(LOGICAL_REGION_DESTRUCTION_MESSAGE, LogicalRegionDestruction, "Destroy Logical Region Message", false, true, true) \
  __op__(INDIVIDUAL_REMOTE_FUTURE_SIZE, IndividualRemoteFutureSize, "Individual Task Set Remote Future Size Message ", true, true, true) \
  __op__(INDIVIDUAL_REMOTE_OUTPUT_REGISTRATION, IndividualRemoteOutputRegistration, "Individual Task Remote Output Registration Message", true, false, false) \
  __op__(INDIVIDUAL_REMOTE_MAPPED, IndividualRemoteMapped, "Individual Task Remote Mapped Message", true, false, false) \
  __op__(INDIVIDUAL_REMOTE_COMPLETE, IndividualRemoteComplete, "Individual Task Remote Complete Message", true, false, false) \
  __op__(INDIVIDUAL_REMOTE_COMMIT, IndividualRemoteCommit, "Individual Task Remote Commit Message", true, false, false) \
  __op__(INDIVIDUAL_CONCURRENT_REQUEST, IndividualTaskConcurrentRequest, "Individual Task Concurrent Request", false, false, false) \
  __op__(INDIVIDUAL_CONCURRENT_RESPONSE, IndividualTaskConcurrentResponse, "Individual Task Concurrent Response", true, false, false) \
  __op__(SLICE_REMOTE_MAPPED, SliceRemoteMapped, "Slice Task Remote Mapped Message", true, false, false) \
  __op__(SLICE_REMOTE_COMPLETE, SliceRemoteComplete, "Slice Task Remote Complete Message", true, false, false) \
  __op__(SLICE_REMOTE_COMMIT, SliceRemoteCommit, "Slice Task Remote Commit Message", true, false, false) \
  __op__(SLICE_RENDEZVOUS_CONCURRENT_MAPPED, SliceConcurrentMapped, "Slice Task Rendezvous Concurrent Mapped Message", false, false, false) \
  __op__(SLICE_COLLECTIVE_ALLREDUCE_REQUEST, SliceCollectiveRequest, "Slice Task Collective Mapping All-Reduce Request", false, false, false) \
  __op__(SLICE_COLLECTIVE_ALLREDUCE_RESPONSE, SliceCollectiveResponse, "Slice Task Collective Mapping All-Reudce Response", true, false, false) \
  __op__(SLICE_CONCURRENT_ALLREDUCE_REQUEST, SliceConcurrentRequest, "Slice Task Concurrent All-Reduce Request", false, false, false) \
  __op__(SLICE_CONCURRENT_ALLREDUCE_RESPONSE, SliceConcurrentResponse, "Slice Task Concurrent All-Reduce Response", true, false, false) \
  __op__(SLICE_FIND_INTRA_DEP, SliceFindIntraDependence, "Slice Task Find Intra-Launch Dependence Message", false, false, false) \
  __op__(SLICE_REMOTE_COLLECTIVE_RENDEZVOUS, SliceRemoteCollective, "Slice Task Remote Collective Rendezvous Message", false, false, false) \
  __op__(SLICE_REMOTE_VERSIONING_COLLECTIVE_RENDEZVOUS, SliceRemoteVersioningCollective, "Slice Task Remote Versioning Collective Message", false, false, false) \
  __op__(SLICE_REMOTE_OUTPUT_EXTENTS, SliceRemoteOutputExtents, "Slice Task Set Remote Output Extents Message", true, false, false) \
  __op__(SLICE_REMOTE_OUTPUT_REGISTRATION, SliceRemoteOutputRegistration, "Slice Task Remote Output Registration Message", true, false, false) \
  __op__(DISTRIBUTED_REMOTE_REGISTRATION, DistributedRemoteRegistration, "Distributed Collectable Remote Registration Message", false, true, true) \
  __op__(DISTRIBUTED_DOWNGRADE_REQUEST, DistributedDowngradeRequest, "Distributed Collectable Downgrade Request", false, true, true) \
  __op__(DISTRIBUTED_DOWNGRADE_RESPONSE, DistributedDowngradeResponse, "Distributed Collectable Downgrade Response", true, true, true) \
  __op__(DISTRIBUTED_DOWNGRADE_SUCCESS, DistributedDowngradeSuccess, "Distributed Collectable Downgrade Success Message", false, true, true) \
  __op__(DISTRIBUTED_DOWNGRADE_UPDATE, DistributedDowngradeUpdate, "Distributed Collectable Downgrade Update Message", false, true, true) \
  __op__(DISTRIBUTED_DOWNGRADE_RESTART, DistributedDowngradeRestart, "Distributed Collectable Downgrade Restart Message", false, true, true) \
  __op__(DISTRIBUTED_GLOBAL_ACQUIRE_REQUEST, DistributedGlobalAcquireRequest, "Distributed Collectable Global Acquire Request", false, false, false) \
  __op__(DISTRIBUTED_GLOBAL_ACQUIRE_RESPONSE, DistributedGlobalAcquireResponse, "Distributed Collectable global Acquire Response", true, false, false) \
  __op__(DISTRIBUTED_VALID_ACQUIRE_REQUEST, DistributedValidAcquireRequest, "Distributed Collectable Valid Acquire Request", false, false, false) \
  __op__(DISTRIBUTED_VALID_ACQUIRE_RESPONSE, DistributedValidAcquireResponse, "Distributed Collectable Valid Acquire Response", true, false, false) \
  __op__(SEND_ATOMIC_RESERVATION_REQUEST, AtomicReservationRequest, "Atomic Reservation Request", false, false, false) \
  __op__(SEND_ATOMIC_RESERVATION_RESPONSE, AtomicReservationResponse, "Atomic Reservation Response", true, false, false) \
  __op__(SEND_PADDED_RESERVATION_REQUEST, PaddedReservationRequest, "Padded Reservation Request", false, false, false) \
  __op__(SEND_PADDED_RESERVATION_RESPONSE, PaddedReservationResponse, "Padded Reservation Response", true, false, false) \
  __op__(SEND_CREATED_REGION_CONTEXTS, CreatedRegionContextsMessage, "Created Region Contexts Message", false, true, true) \
  __op__(SEND_MATERIALIZED_VIEW, MaterializedViewMessage, "Materialized View Message", true, false, true) \
  __op__(SEND_FILL_VIEW, FillViewMessage, "Fill View Message", true, false, true) \
  __op__(SEND_FILL_VIEW_VALUE, FillViewValueMessage, "Fill View Value Message", false, false, true) \
  __op__(SEND_PHI_VIEW, PhiViewMessage, "Phi View Message", false, false, true) \
  __op__(SEND_REDUCTION_VIEW, ReductionViewMessage, "Reduction View Message", true, false, true) \
  __op__(SEND_REPLICATED_VIEW, ReplicatedViewMessage, "Replicated View Message", true, false, true) \
  __op__(SEND_ALLREDUCE_VIEW, AllreduceViewMessage, "Allreduce View Message", true, false, true) \
  __op__(SEND_INSTANCE_MANAGER, InstanceManagerMessage, "Instance Manager Message", true, false, true) \
  __op__(SEND_MANAGER_UPDATE, PhysicalManagerUpdate, "Physical Manager Update Message", true, false, true) \
  __op__(SEND_COLLECTIVE_DISTRIBUTE_FILL, CollectiveDistributeFill, "Collective View Broadcast Fill Message", false, false, false) \
  __op__(SEND_COLLECTIVE_DISTRIBUTE_POINT, CollectiveDistributePoint, "Collective View Broadcast Point Message", false, false, false) \
  __op__(SEND_COLLECTIVE_DISTRIBUTE_POINTWISE, CollectiveDistributePointwise, "Collective View Copy Point-wise Message", false, false, false) \
  __op__(SEND_COLLECTIVE_DISTRIBUTE_REDUCTION, CollectiveDistributeReduction, "Collective View Reduction Tree Message", false, false, false) \
  __op__(SEND_COLLECTIVE_DISTRIBUTE_BROADCAST, CollectiveDistributeBroadcast, "Collective View Broadcast Tree Message", false, false, false) \
  __op__(SEND_COLLECTIVE_DISTRIBUTE_REDUCECAST, CollectiveDistributeReducecast, "Collective View Reducecast Message", false, false, false) \
  __op__(SEND_COLLECTIVE_DISTRIBUTE_HOURGLASS, CollectiveDistributeHourglass, "Collective View Hour-glass Message", false, false, false) \
  __op__(SEND_COLLECTIVE_DISTRIBUTE_ALLREDUCE, CollectiveDistributeAllreduce, "Collective View All-reduce Message", false, false, false) \
  __op__(SEND_COLLECTIVE_HAMMER_REDUCTION, CollectiveHammerReduction, "Collective View Hammer Reduction Message", false, false, false) \
  __op__(SEND_COLLECTIVE_FUSE_GATHER, CollectiveFuseGather, "Collective View Fused Gather Message", false, false, false) \
  __op__(SEND_COLLECTIVE_USER_REQUEST, CollectiveRegisterUserRequest, "Collective View Register User Request", false, false, false) \
  __op__(SEND_COLLECTIVE_USER_RESPONSE, CollectiveRegisterUserResponse, "Collective View Register User Response", true, false, false) \
  __op__(SEND_COLLECTIVE_REGISTER_USER, CollectiveIndividualRegisterUser, "Collective View Individual Register User", false, false, false) \
  __op__(SEND_COLLECTIVE_REMOTE_INSTANCES_REQUEST, CollectiveRemoteInstancesRequest, "Collective View Remote Instances Request", false, false, false) \
  __op__(SEND_COLLECTIVE_REMOTE_INSTANCES_RESPONSE, CollectiveRemoteInstancesResponse, "Collective View Remote Instances Response", true, false, false) \
  __op__(SEND_COLLECTIVE_NEAREST_INSTANCES_REQUEST, CollectiveNearestInstancesRequest, "Collective View Nearest Instances Request", false, false, false) \
  __op__(SEND_COLLECTIVE_NEAREST_INSTANCES_RESPONSE, CollectiveNearestInstancesResponse, "Collective View Nearest Instances Response", true, false, false) \
  __op__(SEND_COLLECTIVE_REMOTE_REGISTRATION, CollectiveRemoteRegistration, "Collective View Remote Registration Message", false, false, false) \
  __op__(SEND_COLLECTIVE_FINALIZE_MAPPING, CollectiveFinalizeMapping, "Collective View Finalize Mapping", false, false, false) \
  __op__(SEND_COLLECTIVE_VIEW_CREATION, CollectiveViewCreation, "Collective View Creation Message", false, false, false) \
  __op__(SEND_COLLECTIVE_VIEW_DELETION, CollectiveViewDeletion, "Collective View Deletion Message", false, false, true) \
  __op__(SEND_COLLECTIVE_VIEW_RELEASE, CollectiveViewRelease, "Collective View Release Message", false, true, true) \
  __op__(SEND_COLLECTIVE_VIEW_NOTIFICATION, CollectiveViewNotification, "Collective View Notification Message", false, true, true) \
  __op__(SEND_COLLECTIVE_VIEW_MAKE_VALID, CollectiveViewMakeValid, "Collective View Make Valid Message", false, false, true) \
  __op__(SEND_COLLECTIVE_VIEW_MAKE_INVALID, CollectiveViewMakeInvalid, "Collective View Make Invalid Message", false, false, true) \
  __op__(SEND_COLLECTIVE_VIEW_INVALIDATE_REQUEST, CollectiveViewInvalidateRequest, "Collective View Invalidation Request", false, false, true) \
  __op__(SEND_COLLECTIVE_VIEW_INVALIDATE_RESPONSE, CollectiveViewInvalidateResponse, "Collective View Invalidation Response", true, false, true) \
  __op__(SEND_COLLECTIVE_VIEW_ADD_REMOTE_REFERENCE, CollectiveViewAddRemoteReference, "Collective View Add Remote Reference Message", false, false, true) \
  __op__(SEND_COLLECTIVE_VIEW_REMOVE_REMOTE_REFERENCE, CollectiveViewRemoveRemoteReference, "Collective View Remove Remote Reference Message", false, false, true) \
  __op__(SEND_CREATE_TOP_VIEW_REQUEST, CreateTopViewRequest, "Create Top Logical View Request", false, false, false) \
  __op__(SEND_CREATE_TOP_VIEW_RESPONSE, CreateTopViewResponse, "Create Top Logical View Respone", false, false, false) \
  __op__(SEND_VIEW_REQUEST, ViewRequestMessage, "Logical View Request", false, false, true) \
  __op__(SEND_VIEW_REGISTER_USER, ViewRegisterUser, "Logical View Register User Message", false, false, false) \
  __op__(SEND_VIEW_FIND_COPY_PRE_REQUEST, ViewFindCopyPreMessage, "Logical View Find Copy Preconditions Message", false, false, false) \
  __op__(SEND_VIEW_ADD_COPY_USER, ViewAddCopyUserMessage, "Logical View Add Copy User Message", true, false, false) \
  __op__(SEND_VIEW_FIND_LAST_USERS_REQUEST, ViewFindLastUsersRequest, "Logical View Find Last Users Request", false, false, true) \
  __op__(SEND_VIEW_FIND_LAST_USERS_RESPONSE, ViewFindLastUsersResponse, "Logical View Find Last Users Response", true, false, true) \
  __op__(SEND_MANAGER_REQUEST, ManagerRequestMessage, "Instance Manager Request", false, false, true) \
  __op__(SEND_FUTURE_RESULT, FutureResultMessage, "Set Future Result Message", true, true, true) \
  __op__(SEND_FUTURE_RESULT_SIZE, FutureSizeMessage, "Set Future Size Message", true, true, true) \
  __op__(SEND_FUTURE_SUBSCRIPTION, FutureSubscription, "Future Subscription Message", false, false, false) \
  __op__(SEND_FUTURE_CREATE_INSTANCE_REQUEST, FutureCreateInstanceRequest, "Future Create Instance Request", false, false, false) \
  __op__(SEND_FUTURE_CREATE_INSTANCE_RESPONSE, FutureCreateInstanceResponse, "Future Create Instance Response", true, false, false) \
  __op__(SEND_FUTURE_MAP_REQUEST, FutureMapFutureRequest, "Future Map Future Request", false, false, false) \
  __op__(SEND_FUTURE_MAP_RESPONSE, FutureMapFutureResponse, "Future Map Future Response", true, false, false) \
  __op__(SEND_FUTURE_MAP_POINTWISE, FutureMapPointwise, "Future Map Pointwise Message", false, false, false) \
  __op__(SEND_REPL_COMPUTE_EQUIVALENCE_SETS, ReplComputeEquivalenceSets, "Replicated Compute Equivalence Sets Message", false, false, false) \
  __op__(SEND_REPL_OUTPUT_EQUIVALENCE_SET, ReplOutputEquivalenceSet, "Replicated Output Equivalence Set Message", false, false, false) \
  __op__(SEND_REPL_OUTPUT_OFFSET, ReplOutputOffset, "Replicated Output Region Offset", false, false, false) \
  __op__(SEND_REPL_REFINE_EQUIVALENCE_SETS, ReplRefineEquivalenceSets, "Replicated Refine Equivalence Sets Message", false, false, false) \
  __op__(SEND_REPL_EQUIVALENCE_SET_NOTIFICATION, ReplEquivalenceSetNotification, "Replicated Equivalence Set Notification Message", false, false, false) \
  __op__(SEND_REPL_BROADCAST_UPDATE, ReplBroadcastUpdate, "Replicated Broadcast Update Message", false, false, true) \
  __op__(SEND_REPL_CREATED_REGIONS, ReplCreatedRegions, "Replicated Created Regions Message", false, false, true) \
  __op__(SEND_REPL_TRACE_EVENT_REQUEST, ReplTraceEventRequest, "Replicated Trace Event Request", false, false, false) \
  __op__(SEND_REPL_TRACE_EVENT_RESPONSE, ReplTraceEventResponse, "Replicated Trace Event Response", true, false, false) \
  __op__(SEND_REPL_TRACE_EVENT_TRIGGER, ReplTraceEventTrigger, "Replicated Trace Event Trigger", false, false, false) \
  __op__(SEND_REPL_TRACE_FRONTIER_REQUEST, ReplTraceFrontierRequest, "Replicated Trace Frontier Request", false, false, false) \
  __op__(SEND_REPL_TRACE_FRONTIER_RESPONSE, ReplTraceFrontierResponse, "Replicated Trace Frontier Response", true, false, false) \
  __op__(SEND_REPL_TRACE_UPDATE, ReplTraceUpdateMessage, "Replicated Trace Update Message", false, false, false) \
  __op__(SEND_REPL_FIND_TRACE_SETS, ReplFindTraceSets, "Replicated Find Trace Sets Message", false, false, false) \
  __op__(SEND_REPL_IMPLICIT_RENDEZVOUS, ReplImplicitRendezvous, "Replicated Implicit Task Rendezvous Message", false, false, false) \
  __op__(SEND_REPL_FIND_COLLECTIVE_VIEW, ReplFindCollectiveView, "Replicated Find Collective View Message", false, false, false) \
  __op__(SEND_REPL_POINTWISE_DEPENDENCE, ReplPointwiseDependence, "Replicated Find Pointwise Dependence Message", false, false, false) \
  __op__(SEND_MAPPER_MESSAGE, MapperMessage, "Mapper Message", false, true, true) \
  __op__(SEND_MAPPER_BROADCAST, MapperBroadcast, "Mapper Broadcast", false, true, true) \
  __op__(SEND_TASK_IMPL_SEMANTIC_REQ, TaskSemanticInfoRequest, "Task Semantic Information Request", false, false, false) \
  __op__(SEND_INDEX_SPACE_SEMANTIC_REQ, IndexSpaceSemanticInfoRequest, "Index Space Semantic Information Request", false, false, false) \
  __op__(SEND_INDEX_PARTITION_SEMANTIC_REQ, IndexPartSemanticInfoRequest, "Index Partition Semantic Information Request", false, false, false) \
  __op__(SEND_FIELD_SPACE_SEMANTIC_REQ, FieldSpaceSemanticInfoRequest, "Field Space Semantic Information Request", false, false, false) \
  __op__(SEND_FIELD_SEMANTIC_REQ, FieldSemanticInfoRequest, "Field Semantic Information Request", false, false, false) \
  __op__(SEND_LOGICAL_REGION_SEMANTIC_REQ, LogicalRegionSemanticInfoRequest, "Logical Region Semantic Information Request", false, false, false) \
  __op__(SEND_LOGICAL_PARTITION_SEMANTIC_REQ, LogicalPartitionSemanticInfoRequest, "Logical Partition Semantic Information Request", false, false, false) \
  __op__(SEND_TASK_IMPL_SEMANTIC_INFO, TaskSemanticInfoResponse, "Task Semantic Information Response", true, false, false) \
  __op__(SEND_INDEX_SPACE_SEMANTIC_INFO, IndexSpaceSemanticInfoResponse, "Index Space Semantic Information Response", true, false, false) \
  __op__(SEND_INDEX_PARTITION_SEMANTIC_INFO, IndexPartSemanticInfoResponse, "Index Space Semantic Information Response", true, false, false) \
  __op__(SEND_FIELD_SPACE_SEMANTIC_INFO, FieldSpaceSemanticInfoResponse, "Field Space Semantic Information Response", true, false, false) \
  __op__(SEND_FIELD_SEMANTIC_INFO, FieldSemanticInfoResponse, "Field Semantic Information Response", true, false, false) \
  __op__(SEND_LOGICAL_REGION_SEMANTIC_INFO, LogicalRegionSemanticInfoResponse, "Logical Region Semantic Information Response", true, false, false) \
  __op__(SEND_LOGICAL_PARTITION_SEMANTIC_INFO, LogicalPartitionSemanticInfoResponse, "Logical Partition Semantic Information Response", true, false, false) \
  __op__(SEND_REMOTE_CONTEXT_REQUEST, RemoteContextRequest, "Remote Context Request", false, false, false) \
  __op__(SEND_REMOTE_CONTEXT_RESPONSE, RemoteContextResponse, "Remote Context Response", true, false, false) \
  __op__(SEND_REMOTE_CONTEXT_PHYSICAL_REQUEST, RemoteContextPhysicalRequest, "Remote Context Physical Request", false, false, false) \
  __op__(SEND_REMOTE_CONTEXT_PHYSICAL_RESPONSE, RemoteContextPhysicalResponse, "Remote Context Physical Response", true, false, false) \
  __op__(SEND_REMOTE_CONTEXT_FIND_COLLECTIVE_VIEW_REQUEST, RemoteContextFindCollectiveViewRequest, "Remote Context Find Collective View Request", false, false, false) \
  __op__(SEND_REMOTE_CONTEXT_FIND_COLLECTIVE_VIEW_RESPONSE, RemoteContextFindCollectiveViewResponse, "Remote Context Find Collective View Response", true, false, false) \
  __op__(SEND_REMOTE_CONTEXT_REFINE_EQUIVALENCE_SETS, RemoteContextRefineEquivalenceSets, "Remote Context Refine Equivalence Sets Messages", false, false, false) \
  __op__(SEND_REMOTE_CONTEXT_POINTWISE_DEPENDENCE, RemoteContextPointwiseDependence, "Remote Context Pointwise Dependence Analysis", false, false, false) \
  __op__(SEND_REMOTE_CONTEXT_FIND_TRACE_LOCAL_SETS_REQUEST, RemoteContextFindTraceLocalRequest, "Remote Context Find Trace Local Sets Request", false, false, false) \
  __op__(SEND_REMOTE_CONTEXT_FIND_TRACE_LOCAL_SETS_RESPONSE, RemoteContextFindTraceLocalResponse, "Remote Context Find Trace Local Sets Response", true, false, false) \
  __op__(SEND_COMPUTE_EQUIVALENCE_SETS_REQUEST, ComputeEquivalenceSetsRequest, "Compute Equivalence Sets Request", false, false, false) \
  __op__(SEND_COMPUTE_EQUIVALENCE_SETS_RESPONSE, ComputeEquivalenceSetsResponse, "Compute Equivalence Sets Response", true, false, false) \
  __op__(SEND_COMPUTE_EQUIVALENCE_SETS_PENDING, ComputeEquivalenceSetsPending, "Compute Equivalence Sets Pending Message", false, false, false) \
  __op__(SEND_OUTPUT_EQUIVALENCE_SET_REQUEST, OutputEquivalenceSetRequest, "Output Equivalence Set Request", false, false, false) \
  __op__(SEND_OUTPUT_EQUIVALENCE_SET_RESPONSE, OutputEquivalenceSetResponse, "Output Equivalence Set Response", true, false, false) \
  __op__(SEND_CANCEL_EQUIVALENCE_SETS_SUBSCRIPTION, CancelEquivalenceSetSubscription, "Cance Equivalence Set Subscription Message", false, true, true) \
  __op__(SEND_INVALIDATE_EQUIVALENCE_SETS_SUBSCRIPTION, InvalidateEquivalenceSetSubscription, "Invalidate Equivalence Set Subscription", false, true, true) \
  __op__(SEND_EQUIVALENCE_SET_CREATION, EquivalenceSetCreation, "Equivalence Set Creation Message", false, false, false) \
  __op__(SEND_EQUIVALENCE_SET_REUSE, EquivalenceSetReuse, "Equivalence Set Reuse Message", false, false, true) \
  __op__(SEND_EQUIVALENCE_SET_REQUEST, EquivalenceSetRequest, "Equivalence Set Request", false, false, true) \
  __op__(SEND_EQUIVALENCE_SET_RESPONSE, EquivalenceSetResponse, "Equivalence Set Response", true, false, true) \
  __op__(SEND_EQUIVALENCE_SET_REPLICATION_REQUEST, EquivalenceSetReplicationRequest, "Equivalence Set Replication Request", false, false, true) \
  __op__(SEND_EQUIVALENCE_SET_REPLICATION_RESPONSE, EquivalenceSetReplicationResponse, "Equivalence Set Replication Response", true, false, true) \
  __op__(SEND_EQUIVALENCE_SET_MIGRATION, EquivalenceSetMigration, "Equivalence Set Migration Message", false, false, true) \
  __op__(SEND_EQUIVALENCE_SET_OWNER_UPDATE, EquivalenceSetOwnerUpdate, "Equivalence Set Owner Update Message", false, false, true) \
  __op__(SEND_EQUIVALENCE_SET_CLONE_REQUEST, EquivalenceSetCloneRequest, "Equivalence Set Clone Request", false, false, false) \
  __op__(SEND_EQUIVALENCE_SET_CLONE_RESPONSE, EquivalenceSetCloneResponse, "Equivalence Set Clone Response", true, false, false) \
  __op__(SEND_EQUIVALENCE_SET_CAPTURE_REQUEST, EquivalenceSetCaptureRequest, "Equivalence Set Capture Request", false, false, false) \
  __op__(SEND_EQUIVALENCE_SET_CAPTURE_RESPONSE, EquivalenceSetCaptureResponse, "Equivalence Set Capture Response", false, false, false) \
  __op__(SEND_EQUIVALENCE_SET_REMOTE_REQUEST_INSTANCES, EquivalenceSetRequestInstances, "Equivalence Set Remote Instances Request", false, false, false) \
  __op__(SEND_EQUIVALENCE_SET_REMOTE_REQUEST_INVALID, EquivalenceSetRequestInvalid, "Equivalence Set Remote Invalid Request", false, false, false) \
  __op__(SEND_EQUIVALENCE_SET_REMOTE_REQUEST_ANTIVALID, EquivalenceSetRequestAntivalid, "Equivalence Set Remote Antivalid Request", false, false, false) \
  __op__(SEND_EQUIVALENCE_SET_REMOTE_UPDATES, RemoteUpdateAnalysis, "Equivalence Set Remote Updates Request", false, false, false) \
  __op__(SEND_EQUIVALENCE_SET_REMOTE_ACQUIRES, RemoteAcquireAnalysis, "Equivalence Set Remote Acquire Request", false, false, false) \
  __op__(SEND_EQUIVALENCE_SET_REMOTE_RELEASES, RemoteReleaseAnalysis, "Equivalence Set Remote Release Request", false, false, false) \
  __op__(SEND_EQUIVALENCE_SET_REMOTE_COPIES_ACROSS, RemoteCopyAcrossAnalysis, "Equivalence Set Remote Copy Across Request", false, false, false) \
  __op__(SEND_EQUIVALENCE_SET_REMOTE_OVERWRITES, RemoteOverwriteAnalysis, "Equivalence Set Remote Overwrite Request", false, false, false) \
  __op__(SEND_EQUIVALENCE_SET_REMOTE_FILTERS, RemoteFilterAnalysis, "Equivalence Set Remote Filter Request", false, false, false) \
  __op__(SEND_EQUIVALENCE_SET_REMOTE_INSTANCES, EquivalenceSetRemoteInstances, "Physical Analysis Remote Instances Response", true, false, false) \
  __op__(SEND_EQUIVALENCE_SET_FILTER_INVALIDATIONS, EquivalenceSetFilterInvalidations, "Equivalence Set Filter Invalidations Message", false, false, false) \
  __op__(SEND_INSTANCE_REQUEST, InstanceRequest, "Instance Manager Request", false, false, false) \
  __op__(SEND_INSTANCE_RESPONSE, InstanceResponse, "Instance Manager Response", true, false, false) \
  __op__(SEND_EXTERNAL_CREATE_REQUEST, ExternalCreateRequest, "External Instance Request", false, false, false) \
  __op__(SEND_EXTERNAL_CREATE_RESPONSE, ExternalCreateResponse, "External Instance Response", true, false, false) \
  __op__(SEND_EXTERNAL_ATTACH, ExternalAttachRequest, "External Attach Request", false, false, false) \
  __op__(SEND_EXTERNAL_DETACH, ExternalDetachRequest, "External Detach Request", false, true, true) \
  __op__(SEND_GC_PRIORITY_UPDATE, GarbageCollectionPriorityUpdate, "Garbage Collection Priority Update", false, true, true) \
  __op__(SEND_GC_REQUEST, GarbageCollectionRequest, "Garbage Collection Request", false, true, true) \
  __op__(SEND_GC_RESPONSE, GarbageCollectionResponse, "Garbage Collection Response", true, true, true) \
  __op__(SEND_GC_ACQUIRE, GarbageCollectionAcquire, "Garbage Collection Acquire Message", false, true, true) \
  __op__(SEND_GC_FAILED, GarbageCollectionFailed, "Garbage Collection Failed Response", true, true, true) \
  __op__(SEND_GC_MISMATCH, GarbageCollectionMismatch, "Garbage Collection Mismatch Message", false, true, true) \
  __op__(SEND_GC_NOTIFY, GarbageCollectionNotification, "Garbage Collection Notification Message", false, true, true) \
  __op__(SEND_GC_DEBUG_REQUEST, GarbageCollectionDebugRequest, "Garbage Collection Debug Request", false, false, false) \
  __op__(SEND_GC_DEBUG_RESPONSE, GarbageCollectionDebugResponse, "Garbage Collection Debug Response", true, false, false) \
  __op__(SEND_GC_RECORD_EVENT, GarbageCollectionRecordEvent, "Garbage Collection Record Event", false, false, false) \
  __op__(SEND_ACQUIRE_REQUEST, GarbageCollectionAcquireRequest, "Garbage Collection Acquire Request", false, false, false) \
  __op__(SEND_ACQUIRE_RESPONSE, GarbageCollectionAcquireResponse, "Garbage Collection Acquire Response", true, false, false) \
  __op__(SEND_VARIANT_BROADCAST, VariantBroadcast, "Variant Broadcast Message", false, true, true) \
  __op__(SEND_CONSTRAINT_REQUEST, ConstraintRequest, "Layout Constraint Request", false, false, false) \
  __op__(SEND_CONSTRAINT_RESPONSE, ConstraintResponse, "Layout Constraint Response", true, false, false) \
  __op__(SEND_CONSTRAINT_RELEASE, ConstraintRelease, "Layout Constraint Release Message", false, true, true) \
  __op__(SEND_TOP_LEVEL_TASK_COMPLETE, TopLevelTaskComplete, "Top Level Task Complete Message", false, true, true) \
  __op__(SEND_MPI_RANK_EXCHANGE, MPIRankExchange, "MPI Rank Exchange Message", false, true, true) \
  __op__(SEND_REPLICATE_DISTRIBUTION, ReplicateDistribution, "Replicate Task Distribution", false, false, false) \
  __op__(SEND_REPLICATE_COLLECTIVE_VERSIONING, ReplicateVersioning, "Replicate Task Collective Versioning Message", false, false, false) \
  __op__(SEND_REPLICATE_COLLECTIVE_MAPPING, ReplicateCollectiveMapping, "Replicate Task Collective Mapping Message", false, false, false) \
  __op__(SEND_REPLICATE_VIRTUAL_RENDEZVOUS, ReplicateVirtualRendezvous, "Replicate Task Virtual Mapping Message", false, false, false) \
  __op__(SEND_REPLICATE_STARTUP_COMPLETE, ReplicateStartup, "Replicate Task Startup Complete Message", false, false, false) \
  __op__(SEND_REPLICATE_POST_MAPPED, ReplicatePostMapped, "Replicate Task Post Mapped Message", false, false, false) \
  __op__(SEND_REPLICATE_TRIGGER_COMPLETE, ReplicateTriggerComplete, "Replicate Task Trigger Complete Message", false, false, false) \
  __op__(SEND_REPLICATE_TRIGGER_COMMIT, ReplicateTriggerCommit, "Replicate Trigger Commit Message", false, false, false) \
  __op__(SEND_CONTROL_REPLICATE_RENDEZVOUS_MESSAGE, ReplicateRendezvousMessage, "Replicate Task Rendezvous Message", false, false, false) \
  __op__(SEND_LIBRARY_MAPPER_REQUEST, MapperLibraryRequest, "Mapper ID Library Request", false, false, false) \
  __op__(SEND_LIBRARY_MAPPER_RESPONSE, MapperLibraryResponse, "Mapper ID Library Response", true, false, false) \
  __op__(SEND_LIBRARY_TRACE_REQUEST, TraceLibraryRequest, "Trace ID Library Request", false, false, false) \
  __op__(SEND_LIBRARY_TRACE_RESPONSE, TraceLibraryResponse, "Trace ID Library Response", true, false, false) \
  __op__(SEND_LIBRARY_PROJECTION_REQUEST, ProjectionLibraryRequest, "Projection ID Library Request", false, false, false) \
  __op__(SEND_LIBRARY_PROJECTION_RESPONSE, ProjectionLibraryResponse, "Projection ID Library Response", true, false, false) \
  __op__(SEND_LIBRARY_SHARDING_REQUEST, ShardingLibraryRequest, "Sharding ID Library Request", false, false, false) \
  __op__(SEND_LIBRARY_SHARDING_RESPONSE, ShardingLibraryResponse, "Sharding ID Library Response", true, false, false) \
  __op__(SEND_LIBRARY_CONCURRENT_REQUEST, ConcurrentLibraryRequest, "Concurrent ID Library Request", false, false, false) \
  __op__(SEND_LIBRARY_CONCURRENT_RESPONSE, ConcurrentLibraryResponse, "Concurrent ID Library Response", true, false, false) \
  __op__(SEND_LIBRARY_EXCEPTION_REQUEST, ExceptionLibraryRequest, "Exception Handler ID Library Request", false, false, false) \
  __op__(SEND_LIBRARY_EXCEPTION_RESPONSE, ExceptionLibraryResponse, "Exception Handler ID Library Response", true, false, false) \
  __op__(SEND_LIBRARY_TASK_REQUEST, TaskLibraryRequest, "Task ID Library Request", false, false, false) \
  __op__(SEND_LIBRARY_TASK_RESPONSE, TaskLibraryResponse, "Task ID Library Response", true, false, false) \
  __op__(SEND_LIBRARY_REDOP_REQUEST, RedopLibraryRequest, "Reduction ID Library Request", false, false, false) \
  __op__(SEND_LIBRARY_REDOP_RESPONSE, RedopLibraryResponse, "Reduction ID Library Response", true, false, false) \
  __op__(SEND_LIBRARY_SERDEZ_REQUEST, SerdezLibraryRequest, "Serdez ID Library Request", false, false, false) \
  __op__(SEND_LIBRARY_SERDEZ_RESPONSE, SerdezLibraryResponse, "Serdez ID Library Response", true, false, false) \
  __op__(SEND_REMOTE_OP_REPORT_UNINIT, RemoteOpReportUninit, "Remote Operation Uninitialized Message", false, false, false) \
  __op__(SEND_REMOTE_OP_PROFILING_COUNT_UPDATE, RemoteOpProfilingUpdate, "Remote Operation Profiling Update Message", false, false, false) \
  __op__(SEND_REMOTE_OP_COMPLETION_EFFECT, RemoteOpCompletionEffect, "Remote Operation Completion Effect", false, false, false ) \
  __op__(SEND_REMOTE_TRACE_UPDATE, RemoteTraceUpdate, "Remote Trace Recording Update Message", false, false, false) \
  __op__(SEND_REMOTE_TRACE_RESPONSE, RemoteTraceResponse, "Remote Trace Recording Response", true, false, false) \
  __op__(SEND_FREE_EXTERNAL_ALLOCATION, FreeExternalAllocation, "Free External Allocation Message", false, true, true) \
  __op__(SEND_NOTIFY_COLLECTED_INSTANCES, NotifyCollectedInstances, "Notify Collected Instances Message", false, true, true) \
  __op__(SEND_CREATE_MEMORY_POOL_REQUEST, CreatePoolRequest, "Create Memory Pool Request", false, false, false) \
  __op__(SEND_CREATE_MEMORY_POOL_RESPONSE, CreatePoolResponse, "Create Memory Pool Response", true, false, false) \
  __op__(SEND_RELEASE_MEMORY_POOL_REQUEST, ReleasePoolRequest, "Release Memory Pool Request", false, true, true) \
  __op__(SEND_CREATE_UNBOUND_REQUEST, CreateUnboundRequest, "Create Unbound Instance Request", false, false, false) \
  __op__(SEND_CREATE_UNBOUND_RESPONSE, CreateUnboundResponse, "Create Unbound Instances Response", true, false, false) \
  __op__(SEND_CREATE_FUTURE_INSTANCE_REQUEST, CreateFutureInstanceRequest, "Create Future Instance Request", false, false, false) \
  __op__(SEND_CREATE_FUTURE_INSTANCE_RESPONSE, CreateFutureInstanceResponse, "Create Future Instance Response", true, false, false) \
  __op__(SEND_FREE_FUTURE_INSTANCE, FreeFutureInstance, "Free Future Instance Message", false, true, true) \
  __op__(SEND_REMOTE_DISTRIBUTED_ID_REQUEST, DistributedIDRequest, "Distributed Collectable ID Request", false, false, false) \
  __op__(SEND_REMOTE_DISTRIBUTED_ID_RESPONSE, DistributedIDResponse, "Distributed Collectable ID Response", true, false, false) \
  __op__(SEND_PROFILER_EVENT_TRIGGER, ProfilerEventTriggerMessage, "Profiler Record Event Trigger Message", false, true, true) \
  __op__(SEND_PROFILER_EVENT_POISON, ProfilerEventPoisonMessage, "Profiler Record Event Poison Message", false, true, true) \
  __op__(SEND_SHUTDOWN_NOTIFICATION, ShutdownNotification, "Shutdown Notification Message", false, true, true) \
  __op__(SEND_SHUTDOWN_RESPONSE, ShutdownResponse, "Shutdown Response", true, true, true)

#define LEGION_SHARD_COLLECTIVE_ACTIVE_MESSAGES(__op__) \
  __op__(SEND_CONTROL_REPLICATION_FUTURE_ALLREDUCE, "Control Replication Future All-Reduce Message") \
  __op__(SEND_CONTROL_REPLICATION_FUTURE_BROADCAST, "Control Replication Future Broadcast Message") \
  __op__(SEND_CONTROL_REPLICATION_FUTURE_REDUCTION, "Control Replication Future Reduction Message") \
  __op__(SEND_CONTROL_REPLICATION_VALUE_ALLREDUCE, "Control Replication Value All-reduce Message") \
  __op__(SEND_CONTROL_REPLICATION_VALUE_BROADCAST, "Control Replication Value Broadcast Message") \
  __op__(SEND_CONTROL_REPLICATION_VALUE_EXCHANGE, "Control Replication Value Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_BUFFER_BROADCAST, "Control Replication Buffer Broadcast Message") \
  __op__(SEND_CONTROL_REPLICATION_SHARD_SYNC_TREE, "Control Replication Shard Synchronization Tree Message") \
  __op__(SEND_CONTROL_REPLICATION_SHARD_EVENT_TREE, "Control Replication Shard Event Tree Message") \
  __op__(SEND_CONTROL_REPLICATION_SINGLE_TASK_TREE, "Control Replication Single Task Tree Message") \
  __op__(SEND_CONTROL_REPLICATION_CROSS_PRODUCT_PARTITION, "Control Replication Cross Product Partition Message") \
  __op__(SEND_CONTROL_REPLICATION_SHARDING_GATHER_COLLECTIVE, "Control Replication Sharding Gather Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_INDIRECT_COPY_EXCHANGE, "Control Replication Indirect Copy Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_FIELD_DESCRIPTOR_EXCHANGE, "Control Replication Field Descriptor Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_FIELD_DESCRIPTOR_GATHER, "Control Replication Field Descriptor Gather Message") \
  __op__(SEND_CONTROL_REPLICATION_DEPPART_RESULT_SCATTER, "Control Replication Dependent Partition Result Scatter Message") \
  __op__(SEND_CONTROL_REPLICATION_BUFFER_EXCHANGE, "Control Replication Buffer Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_FUTURE_NAME_EXCHANGE, "Control Replication Future Name Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_MUST_EPOCH_MAPPING_BROADCAST, "Control Replication Must Epoch Mapping Broadcast Message") \
  __op__(SEND_CONTROL_REPLICATION_MUST_EPOCH_MAPPING_EXCHANGE, "Control Replication Must Epoch Mapping Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_MUST_EPOCH_DEPENDENCE_EXCHANGE, "Control Replication Must Epoch Dependence Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_MUST_EPOCH_COMPLETION_EXCHANGE, "Control Replication Must Epoch Completion Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_CHECK_COLLECTIVE_MAPPING, "Control Replication Check Collective Mapping Message") \
  __op__(SEND_CONTROL_REPLICATION_CHECK_COLLECTIVE_SOURCES, "Control Replication Check Collective Sources Message") \
  __op__(SEND_CONTROL_REPLICATION_TEMPLATE_INDEX_EXCHANGE, "Control Replication Template Index Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_UNORDERED_EXCHANGE, "Control Replication Unordered Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_CONSENSUS_MATCH, "Control Replication Consensus Match Exchange") \
  __op__(SEND_CONTROL_REPLICATION_VERIFY_CONTROL_REPLICATION_EXCHANGE, "Control Replication Safety Verification Message") \
  __op__(SEND_CONTROL_REPLICATION_OUTPUT_SIZE_EXCHANGE, "Control Replication Output Size Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_INDEX_ATTACH_LAUNCH_SPACE, "Control Replication Index Attach Launch Message") \
  __op__(SEND_CONTROL_REPLICATION_INDEX_ATTACH_UPPER_BOUND, "Control Replication Index Attach Upper Bound Message") \
  __op__(SEND_CONTROL_REPLICATION_INDEX_ATTACH_EXCHANGE, "Control Replication Index Attach Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_SHARD_PARTICIPANTS_EXCHANGE, "Control Replication Shard Participants Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_IMPLICIT_SHARDING_FUNCTOR, "Control Replication Implicit Sharding Functor Message") \
  __op__(SEND_CONTROL_REPLICATION_CREATE_FILL_VIEW, "Control Replication Create Fill View Message") \
  __op__(SEND_CONTROL_REPLICATION_VERSIONING_RENDEZVOUS, "Control Replication Versioning Rendezvous Message") \
  __op__(SEND_CONTROL_REPLICATION_VIEW_RENDEZVOUS, "Control Replication View Rendezvous Message") \
  __op__(SEND_CONTROL_REPLICATION_CONCURRENT_MAPPING_RENDEZVOUS, "Control Replication Concurrent Mapping Rendezvous Message") \
  __op__(SEND_CONTROL_REPLICATION_CONCURRENT_ALLREDUCE, "Control Replication Concurrent Allreduce Message") \
  __op__(SEND_CONTROL_REPLICATION_PROJECTION_TREE_EXCHANGE, "Control Replication Projection Tree Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_TIMEOUT_MATCH_EXCHANGE, "Control Replication Timeout Match Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_MASK_EXCHANGE, "Control Replication Mask Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_PREDICATE_EXCHANGE, "Control Replication Predicate Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_CROSS_PRODUCT_EXCHANGE, "Control Replication Cross-Product Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_TRACING_SET_DEDUPLICATION, "Control Replication Tracing Set Deduplication Message") \
  __op__(SEND_CONTROL_REPLICATION_POINTWISE_ALLREDUCE, "Control Replication Pointwise Allreduce Message") \
  __op__(SEND_CONTROL_REPLICATION_INTERFERING_POINT_EXCHANGE, "Control Replication Interfering Point Exchange Message") \
  __op__(SEND_CONTROL_REPLICATION_SLOW_BARRIER, "Control Replication Slow Barrier Message")

#define CTRL_REPL_KINDS(kind, name) kind,
#define MESSAGE_KINDS(kind, type, name, resp, escape_ctx, escape_op) kind,
    enum MessageKind {
      LEGION_SHARD_COLLECTIVE_ACTIVE_MESSAGES(CTRL_REPL_KINDS)
      LEGION_ACTIVE_MESSAGES(MESSAGE_KINDS)
      LAST_SEND_KIND,  // This one must be last
    };
#undef CTRL_REPL_KINDS
#undef MESSAGE_KINDS
    // clang-format on

    struct MessageHeader : public LgTaskArgs<MessageHeader> {
    public:
      static constexpr LgTaskID TASK_ID = LG_MESSAGE_ID;
    public:
      MessageHeader(void) = default;
      MessageHeader(
          MessageKind k, VirtualChannelKind c, bool escape_ctx, bool escape_op);
      // We override handle directly since we're doing the handling
      static void handle(const void* data, size_t size);
    public:
      MessageKind kind;
      VirtualChannelKind channel;
      AddressSpaceID sender;
    };

    /**
     * \class VirtualChannel
     * This class provides the basic support for sending and receiving
     * messages for a single virtual channel.
     */
    class VirtualChannel {
    public:
      VirtualChannel(
          VirtualChannelKind kind, AddressSpaceID local_address_space,
          size_t max_message_size, bool profile);
      VirtualChannel(const VirtualChannel& rhs) = delete;
      ~VirtualChannel(void);
    public:
      VirtualChannel& operator=(const VirtualChannel& rhs) = delete;
    public:
      void send_message(
          MessageKind kind, const Serializer& rez, RtEvent send_precondition,
          Processor target, bool response);
      void record_received(void);
      void confirm_shutdown(ShutdownManager* shutdown_manager);
    private:
      void handle_message(
          MessageKind kind, AddressSpaceID remote_address_space,
          const void* args, size_t arglen);
    private:
      mutable LocalLock channel_lock;
      RtEvent last_message_event;
    public:
      const bool ordered_channel;
      const bool profile_outgoing_messages;
      const LgPriority request_priority;
      const LgPriority response_priority;
    private:
      // Keep track of the unordered events for shutdown
      std::deque<RtEvent> unordered_events;
      // Monotonically increasing counters for sent and received messages
      // 64-bit counters should pretty much never overflow so we never
      // need to reset them back to zero
      uint64_t sent_messages = 0;
      std::atomic<uint64_t> received_messages = 0;
    };

    /**
     * \class MessageManager
     * This class manages sending and receiving of message between
     * instances of the Internal runtime residing on different nodes.
     * The manager also abstracts some of the details of sending these
     * messages.  Messages can be accumulated together in bulk messages
     * for performance reason.  The runtime can also place an upper
     * bound on the size of the data communicated between runtimes in
     * an active message, which the message manager then uses to
     * break down larger messages into smaller active messages.
     *
     * On the receiving side, the message manager unpacks the messages
     * that have been sent and then call the appropriate runtime
     * methods for handling the messages.  In cases where larger
     * messages were broken down into smaller messages, then message
     * manager waits until it has received all the active messages
     * before handling the message.
     */
    class MessageManager {
    public:
      MessageManager(
          AddressSpaceID remote, size_t max, const Processor remote_util_group);
      MessageManager(const MessageManager& rhs) = delete;
      ~MessageManager(void);
    public:
      MessageManager& operator=(const MessageManager& rhs) = delete;
    public:
      void send_message(
          MessageKind kind, VirtualChannelKind vc, const Serializer& rez,
          bool response = false,
          RtEvent flush_precondition = RtEvent::NO_RT_EVENT);
      VirtualChannel& find_channel(VirtualChannelKind vc);
      void confirm_shutdown(ShutdownManager* shutdown_manager);
      static void register_handlers(void);
      // Maintain a static-mapping between message kinds and virtual channels
      static constexpr VirtualChannelKind find_message_vc(MessageKind kind);
      // Helper method to avoid cyclic header includes
      static MessageManager* find_manager(AddressSpaceID target);
      static inline void (*message_handler_table[LAST_SEND_KIND])(
          Deserializer&, AddressSpaceID) = {0};
    private:
      VirtualChannel* const channels;
    public:
      // State for sending messages
      const AddressSpaceID remote_address_space;
      const Processor target;
    };

    template<typename T>
    class ActiveMessage : public Serializer {
    public:
      ActiveMessage(MessageKind kind, bool escape_ctx, bool escape_op)
        : Serializer(), header(kind, T::CHANNEL, escape_ctx, escape_op)
      {
        serialize(header);
      }
      ActiveMessage(const ActiveMessage& rhs) = delete;
      ActiveMessage(ActiveMessage&&) = delete;
      ~ActiveMessage(void) { }
    public:
      ActiveMessage& operator=(const ActiveMessage& rhs) = delete;
      ActiveMessage& operator=(ActiveMessage&& rhs) = delete;
    public:
      inline void dispatch(
          AddressSpaceID target, RtEvent pre = RtEvent::NO_RT_EVENT) const
      {
        MessageManager* manager = MessageManager::find_manager(target);
        static_assert(T::CHANNEL < MAX_NUM_VIRTUAL_CHANNELS);
        manager->send_message(header.kind, T::CHANNEL, *this, T::RESPONSE, pre);
      }
      inline const void* get_payload(void) const
      {
        const uint8_t* payload = static_cast<const uint8_t*>(get_buffer());
        return (payload + sizeof(header));
      }
      inline size_t get_payload_size(void) const
      {
        return (get_used_bytes() - sizeof(header));
      }
    public:
      const MessageHeader header;
    };

    class ShardCollectiveMessage
      : public ActiveMessage<ShardCollectiveMessage> {
    public:
      static constexpr VirtualChannelKind CHANNEL = DEFAULT_VIRTUAL_CHANNEL;
      static constexpr bool RESPONSE = false;
    public:
      ShardCollectiveMessage(MessageKind k)
        : ActiveMessage<ShardCollectiveMessage>(
              k, false /*escape ctx*/, false /*escape op*/)
      { }
    public:
      static void handle(Deserializer& derez, AddressSpaceID source);
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/managers/message.inl"

#endif  // __LEGION_MESSAGE_MANAGER_H__
