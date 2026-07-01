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

#ifndef __LEGION_METATASK_H__
#define __LEGION_METATASK_H__

#include "legion/api/types.h"

namespace Legion {
  namespace Internal {

    // clang-format off
    // All the different kinds of active mssages
#define LEGION_META_TASKS(__op__) \
  __op__(LG_SCHEDULER_ID, ProcessorManager::SchedulerArgs, "Scheduler") \
  __op__(LG_TRIGGER_READY_ID, InnerContext::TriggerReadyArgs, "Mapping Pipeline Stage") \
  __op__(LG_TRIGGER_EXECUTION_ID, InnerContext::TriggerExecutionArgs, "Execution Pipeline Stage") \
  __op__(LG_TRIGGER_COMMIT_ID, InnerContext::TriggerCommitArgs, "Commit Pipeline Stage") \
  __op__(LG_DEFERRED_EXECUTION_ID, InnerContext::DeferredExecutionArgs, "Deferred Execution Pipeline Stage") \
  __op__(LG_DEFERRED_COMPLETION_ID, InnerContext::DeferredCompletionArgs, "Completion Pipeline Stage") \
  __op__(LG_DEFERRED_COMMIT_ID, InnerContext::DeferredCommitArgs, "Deferred Commit Pipeline Stage") \
  __op__(LG_PRE_PIPELINE_ID, InnerContext::PrepipelineArgs, "Prepipeline Stage") \
  __op__(LG_TRIGGER_DEPENDENCE_ID, InnerContext::DependenceArgs, "Logical Dependence Analysis Pipeline stage") \
  __op__(LG_DEFERRED_MAPPED_ID, InnerContext::DeferredMappedArgs, "Deferred Mapping Pipeline Stage") \
  __op__(LG_TRIGGER_OP_ID, Operation::TriggerOpArgs, "Deferred Operation Mapping") \
  __op__(LG_TRIGGER_TASK_ID, TaskOp::TriggerTaskArgs, "Deferred Task Mapping") \
  __op__(LG_DEFER_MAPPER_SCHEDULER_TASK_ID, ProcessorManager::DeferMapperSchedulerArgs, "Deferred Mapper Scheduler") \
  __op__(LG_CONTRIBUTE_COLLECTIVE_ID, FutureImpl::ContributeCollectiveArgs, "Deferred Future Contribute Collective") \
  __op__(LG_FUTURE_CALLBACK_TASK_ID, FutureImpl::FutureCallbackArgs, "Deferred Future Callback") \
  __op__(LG_CALLBACK_RELEASE_TASK_ID, FutureImpl::CallbackReleaseArgs, "Deferred Future Callback Release") \
  __op__(LG_FUTURE_BROADCAST_TASK_ID, FutureImpl::FutureBroadcastArgs, "Deferred Future Broadcast") \
  __op__(LG_TOP_FINISH_TASK_ID, Runtime::TopFinishArgs, "Deferred Top Finish") \
  __op__(LG_MAPPER_TASK_ID, Runtime::MapperTaskArgs, "Deferred Mapper Task") \
  __op__(LG_DISJOINTNESS_TASK_ID, IndexPartNode::DisjointnessArgs, "Deferred Disjointness Test") \
  __op__(LG_DEFER_TIMING_MEASUREMENT_TASK_ID, FenceOp::DeferTimingMeasurementArgs, "Deferred Timing Measurement") \
  __op__(LG_TASK_IMPL_SEMANTIC_INFO_REQ_TASK_ID, TaskImpl::SemanticRequestArgs, "Deferred Task Semantic Information") \
  __op__(LG_INDEX_SPACE_SEMANTIC_INFO_REQ_TASK_ID, IndexSpaceNode::SemanticRequestArgs, "Deferred Index Space Semantic Information") \
  __op__(LG_INDEX_PART_SEMANTIC_INFO_REQ_TASK_ID, IndexPartNode::SemanticRequestArgs, "Deferred Index Partition Semantic Information") \
  __op__(LG_FIELD_SPACE_SEMANTIC_INFO_REQ_TASK_ID, FieldSpaceNode::SemanticRequestArgs, "Deferred Field Space Semantic Information") \
  __op__(LG_FIELD_SEMANTIC_INFO_REQ_TASK_ID, FieldSpaceNode::SemanticFieldRequestArgs, "Deferred Field Space Field Semantic Information") \
  __op__(LG_DEFER_FIELD_INFOS_TASK_ID, FieldSpaceNode::DeferRequestFieldInfoArgs, "Deferred Field Space Infos") \
  __op__(LG_REGION_SEMANTIC_INFO_REQ_TASK_ID, RegionNode::SemanticRequestArgs, "Deferred Logical Region Semantic Information") \
  __op__(LG_PARTITION_SEMANTIC_INFO_REQ_TASK_ID, PartitionNode::SemanticRequestArgs, "Deferred Logical Partition Semantic Information") \
  __op__(LG_INDEX_SPACE_DEFER_CHILD_TASK_ID, IndexSpaceNode::DeferChildArgs, "Deferred Index Space Child Computation")\
  __op__(LG_INDEX_PART_DEFER_CHILD_TASK_ID, IndexPartNode::DeferChildArgs, "Deferred Index Partition Child Computation") \
  __op__(LG_INDEX_PART_DEFER_SHARD_RECTS_TASK_ID, IndexPartNode::DeferFindShardRects, "Deferred Index Partition Find Shard Rects") \
  __op__(LG_DEFERRED_ENQUEUE_TASK_ID, InnerContext::DeferredEnqueueTaskArgs, "Deferred Enqueue Task") \
  __op__(LG_DEFER_MAPPER_MESSAGE_TASK_ID, MapperManager::DeferMessageArgs, "Deferred Mapper Message") \
  __op__(LG_DEFER_MAPPER_COLLECTION_TASK_ID, MapperManager::DeferInstanceCollectionArgs, "Deferred Mapper Instance Collective") \
  __op__(LG_REMOTE_VIEW_CREATION_TASK_ID, PhysicalManager::RemoteCreateViewArgs, "Deferred Remote View Creation") \
  __op__(LG_DEFER_PERFORM_MAPPING_TASK_ID, TaskOp::DeferMappingArgs, "Deferred Task Perform Mapping") \
  __op__(LG_FINALIZE_OUTPUT_TREE_TASK_ID, TaskOp::FinalizeOutputEqKDTreeArgs, "Deferred Finalize Output Regions Equivalence Set KD Tree") \
  __op__(LG_MISPREDICATION_TASK_ID, SingleTask::MispredicationTaskArgs, "Deferred Handle Task Prediction False") \
  __op__(LG_DEFER_TRIGGER_CHILDREN_COMMIT_TASK_ID, TaskOp::DeferTriggerChildrenCommitArgs, "Deferred Task Trigger Children Commit") \
  __op__(LG_ORDER_CONCURRENT_LAUNCH_TASK_ID, SingleTask::OrderConcurrentLaunchArgs, "Deferred Order Concurrent Launch") \
  __op__(LG_DEFER_MATERIALIZED_VIEW_TASK_ID, MaterializedView::DeferMaterializedViewArgs, "Deferred Materialized View Registration") \
  __op__(LG_DEFER_REDUCTION_VIEW_TASK_ID, ReductionView::DeferReductionViewArgs, "Deferred Reduction View Registration") \
  __op__(LG_DEFER_PHI_VIEW_REGISTRATION_TASK_ID, PhiView::DeferPhiViewRegistrationArgs, "Deferred Phi View Registration") \
  __op__(LG_TIGHTEN_INDEX_SPACE_TASK_ID, IndexSpaceExpression::TightenIndexSpaceArgs, "Deferred Tighten Index Space") \
  __op__(LG_REPLAY_SLICE_TASK_ID, PhysicalTemplate::ReplaySliceArgs, "Deferred Replay Physical Trace") \
  __op__(LG_TRANSITIVE_REDUCTION_TASK_ID, PhysicalTemplate::TransitiveReductionArgs, "Deferred Trace Template Transitive Reduction") \
  __op__(LG_DELETE_TEMPLATE_TASK_ID, PhysicalTemplate::DeleteTemplateArgs, "Deferred Delete Trace Template") \
  __op__(LG_DEFER_MAKE_OWNER_TASK_ID, EquivalenceSet::DeferMakeOwnerArgs, "Deferred Equivalence Set Make Owner") \
  __op__(LG_DEFER_APPLY_STATE_TASK_ID, EquivalenceSet::DeferApplyStateArgs, "Deferred Equivalence Set Apply State") \
  __op__(LG_COPY_FILL_AGGREGATION_TASK_ID, CopyFillAggregator::CopyFillAggregation, "Deferred Copy-Fill Aggregation") \
  __op__(LG_COPY_FILL_DELETION_TASK_ID, CopyFillGuard::CopyFillDeletion, "Deferred Copy-Fill Aggregator Deletion") \
  __op__(LG_FINALIZE_EQ_SETS_TASK_ID, EqSetTracker::LgFinalizeEqSetsArgs, "Deferred Finalize Equivalence Sets") \
  __op__(LG_FINALIZE_OUTPUT_EQ_SET_TASK_ID, VersionManager::FinalizeOutputEquivalenceSetArgs, "Deferred Finalize Output Equivalence Set") \
  __op__(LG_FINALIZE_OUTPUT_REGION_TASK_ID, OutputRegionImpl::FinalizeOutputRegionArgs, "Deferred Finalize Output Region") \
  __op__(LG_DEFERRED_COPY_ACROSS_TASK_ID, CopyOp::DeferredCopyAcross, "Deferred Copy Across") \
  __op__(LG_DEFER_REMOTE_OP_DELETION_TASK_ID, RemoteOp::DeferRemoteOpDeletionArgs, "Deferred Remote Op Deletion") \
  __op__(LG_DEFER_PERFORM_TRAVERSAL_TASK_ID, PhysicalAnalysis::DeferPerformTraversalArgs, "Deferred Physical Analysis Traversal Stage") \
  __op__(LG_DEFER_PERFORM_ANALYSIS_TASK_ID, PhysicalAnalysis::DeferPerformAnalysisArgs, "Deferred Physical Analysis Analyze Equivalence Set Stage") \
  __op__(LG_DEFER_PERFORM_REMOTE_TASK_ID, PhysicalAnalysis::DeferPerformRemoteArgs, "Deferred Physical Analysis Remote Stage") \
  __op__(LG_DEFER_PERFORM_UPDATE_TASK_ID, PhysicalAnalysis::DeferPerformUpdateArgs, "Deferred Physical Analysis Update Stage") \
  __op__(LG_DEFER_PERFORM_REGISTRATION_TASK_ID, PhysicalAnalysis::DeferPerformRegistrationArgs, "Deferred Physical Analysis Registration Stage") \
  __op__(LG_DEFER_PERFORM_OUTPUT_TASK_ID, PhysicalAnalysis::DeferPerformOutputArgs, "Deferred Physical Analysis Output Stage") \
  __op__(LG_DEFER_PHYSICAL_MANAGER_TASK_ID, PhysicalManager::DeferPhysicalManagerArgs, "Deferred Physical Manager Args") \
  __op__(LG_DEFER_DELETE_PHYSICAL_MANAGER_TASK_ID, PhysicalManager::DeferDeletePhysicalManager, "Deferred Delete Physical Manager") \
  __op__(LG_DEFER_VERIFY_PARTITION_TASK_ID, InnerContext::VerifyPartitionArgs, "Deferred Verify Partition") \
  __op__(LG_DEFER_RELEASE_ACQUIRED_TASK_ID, Operation::DeferReleaseAcquiredArgs, "Deferred Release Acquired Instances") \
  __op__(LG_DEFER_COPY_ACROSS_TASK_ID, CopyAcrossExecutor::DeferCopyAcrossArgs, "Deferred Copy-Across Execution") \
  __op__(LG_MALLOC_INSTANCE_TASK_ID, MemoryManager::MallocInstanceArgs, "Deferred Malloc Instance") \
  __op__(LG_FREE_INSTANCE_TASK_ID, MemoryManager::FreeInstanceArgs, "Deferred Free Instance") \
  __op__(LG_DEFER_TRACE_UPDATE_TASK_ID, ShardedPhysicalTemplate::DeferTraceUpdateArgs, "Deferred Trace Update") \
  __op__(LG_DEFER_DELETE_FUTURE_INSTANCE_TASK_ID, FutureInstance::DeferDeleteFutureInstanceArgs, "Deferred Delete Future Instance") \
  __op__(LG_FREE_EXTERNAL_TASK_ID, FutureInstance::FreeExternalArgs, "Deferred Free External Future Instance") \
  __op__(LG_DEFER_COLLECTIVE_TASK_ID, ShardCollective::DeferCollectiveArgs, "Deferred Collective Async") \
  __op__(LG_DEFER_ISSUE_FILL_TASK_ID, FillView::DeferIssueFill, "Deferred Issue Fill") \
  __op__(LG_DEFER_MUST_EPOCH_RETURN_TASK_ID, ReplMustEpochOp::DeferMustEpochReturnResourcesArgs, "Deferred Must Epoch Return Resources") \
  __op__(LG_DEFER_DELETION_COMMIT_TASK_ID, ReplDeletionOp::DeferDeletionCommitArgs, "Deferred Deletion Commit") \
  __op__(LG_YIELD_TASK_ID, TaskContext::YieldArgs, "Deferred Yield") \
  __op__(LG_AUTO_TRACE_PROCESS_REPEATS_TASK_ID, TraceRecognizer::FindRepeatsTaskArgs, "Deferred Automatic Tracing Find Repeats") \
  __op__(LG_PROFILER_DUMP_TASK_ID, LegionProfiler::ProfilerDumpArgs, "Profiler Intermediate Dump") \
  /* Nothing after the shutdown meta-task except for messsages*/ \
  __op__(LG_RETRY_SHUTDOWN_TASK_ID, ShutdownManager::RetryShutdownArgs, "Deferred Retry Shutdown") \
  /* Message ID goes at the end so we can append additional */ \
  /* message IDs here for the profiler and separate meta-tasks */ \
  __op__(LG_MESSAGE_ID, MessageHeader, "Message Handler Meta-Task")
    // clang-format on

    // Enumeration of Legion runtime tasks
    enum LgTaskID {
#define META_TASK_KINDS(kind, type, name) kind,
      LEGION_META_TASKS(META_TASK_KINDS)
#undef META_TASK_KINDS
      LG_LAST_TASK_ID,  // This one should always be last
      // this marks the beginning of task IDs tracked by the shutdown algorithm
      LG_BEGIN_SHUTDOWN_TASK_IDS = LG_RETRY_SHUTDOWN_TASK_ID,
    };

    static_assert((LG_RETRY_SHUTDOWN_TASK_ID + 1) == LG_MESSAGE_ID);
    static_assert((LG_MESSAGE_ID + 1) == LG_LAST_TASK_ID);

    // Methodology for assigning priorities to meta-tasks:
    // Minimum and low priority are for things like profiling
    // that we don't want to interfere with normal execution.
    // Resource priority is reserved for tasks that have been
    // granted resources like reservations. Running priority
    // is the highest and guarantees that we drain out any
    // previously running tasks over starting new ones. The rest
    // of the priorities are classified as either 'throughput'
    // or 'latency' sensitive. Under each of these two major
    // categories there are four sub-priorities:
    //  - work: general work to be done
    //  - deferred: work that was already scheduled but
    //              for which a continuation had to be
    //              made so we don't want to wait behind
    //              work that hasn't started yet
    //  - messsage: a message from a remote node that we
    //              should handle sooner than our own
    //              work since work on the other node is
    //              blocked waiting on our response
    //  - response: a response message from a remote node
    //              that we should handle to unblock work
    //              on our own node
    enum LgPriority {
      LG_MIN_PRIORITY = std::numeric_limits<int>::min(),
      LG_LOW_PRIORITY = -1,
      // Throughput priorities
      LG_THROUGHPUT_WORK_PRIORITY = 0,
      LG_THROUGHPUT_DEFERRED_PRIORITY = 1,
      LG_THROUGHPUT_MESSAGE_PRIORITY = 2,
      LG_THROUGHPUT_RESPONSE_PRIORITY = 3,
      // Latency priorities
      LG_LATENCY_WORK_PRIORITY = 4,
      LG_LATENCY_DEFERRED_PRIORITY = 5,
      LG_LATENCY_MESSAGE_PRIORITY = 6,
      LG_LATENCY_RESPONSE_PRIORITY = 7,
      // Resource priorities
      LG_RESOURCE_PRIORITY = 8,
      // Running priorities
      LG_RUNNING_PRIORITY = 9,
    };

    /**
     * \class LgTaskArgs
     * The base class for all Legion Task arguments
     */
    template<typename T>
    struct LgTaskArgs {
    public:
      LgTaskArgs(void)
        : lg_task_id(T::TASK_ID),
#ifdef LEGION_DEBUG_CALLERS
          lg_call_id(implicit_task_kind),
#endif
          enclosing_context(0), provenance(0), unique_op_id(0)
      { }
      LgTaskArgs(bool escapes_ctx, bool escapes_op)
        : lg_task_id(T::TASK_ID),
#ifdef LEGION_DEBUG_CALLERS
          lg_call_id(implicit_task_kind),
#endif
          enclosing_context(escapes_ctx ? 0 : implicit_enclosing_context),
          provenance(escapes_op ? 0 : implicit_provenance),
          unique_op_id(escapes_op ? 0 : implicit_unique_op_id)
      {
        static_assert(std::is_trivially_copyable_v<T>);
        // Make sure this is aligned reasonably as well
        // so we can do the cast in the 'handle' method
        static_assert(alignof(T) <= alignof(std::max_align_t));
      }
    public:
      // Make sure this is first so we can read it out
      LgTaskID lg_task_id;
#ifdef LEGION_DEBUG_CALLERS
      LgTaskID lg_call_id;
#endif
      DistributedID enclosing_context;
      ProvenanceID provenance;
      ::legion_unique_id_t unique_op_id;
    public:
      static void handle(const void* data, size_t size)
      {
        legion_assert(sizeof(T) == size);
        // Technically this is not quite correct because the alignment
        // of the data in the struct might not be the same depending on
        // how Realm allocates it, but it hasn't caused any issues yet
        // and is higher performance so we do it like this for now.
        // It should be safe given the static assertion in the constructor.
        // If it ever breaks we'll need to copy onto the stack which
        // is fine since we know all arguments are trivially copyable
        const T* args = static_cast<const T*>(data);
        legion_assert(args->lg_task_id == T::TASK_ID);
        // If you ever change this here then make sure you also update
        // the message handler as well
#ifdef LEGION_DEBUG_CALLERS
        implicit_task_caller = args->lg_call_id;
#endif
        implicit_operation = nullptr;
        implicit_enclosing_context = args->enclosing_context;
        implicit_provenance = args->provenance;
        implicit_unique_op_id = args->unique_op_id;
        args->execute();
      }
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_METATASK_H__
