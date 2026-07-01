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

#include "legion/operations/factory.h"
#include "legion/operations/acquire.h"
#include "legion/operations/allreduce.h"
#include "legion/operations/attach.h"
#include "legion/operations/begin.h"
#include "legion/operations/boolean.h"
#include "legion/operations/close.h"
#include "legion/operations/collective.h"
#include "legion/operations/complete.h"
#include "legion/operations/copy.h"
#include "legion/operations/creation.h"
#include "legion/operations/deletion.h"
#include "legion/operations/dependent.h"
#include "legion/operations/detach.h"
#include "legion/operations/discard.h"
#include "legion/operations/dynamic.h"
#include "legion/operations/fence.h"
#include "legion/operations/fill.h"
#include "legion/operations/frame.h"
#include "legion/operations/mapping.h"
#include "legion/operations/mustepoch.h"
#include "legion/operations/partition.h"
#include "legion/operations/recurrent.h"
#include "legion/operations/refinement.h"
#include "legion/operations/release.h"
#include "legion/operations/reset.h"
#include "legion/operations/timing.h"
#include "legion/operations/tunable.h"
#include "legion/tasks/index.h"
#include "legion/tasks/individual.h"
#include "legion/tasks/point.h"
#include "legion/tasks/shard.h"
#include "legion/tasks/slice.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Operation Factory
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename OP, typename WRAP, bool CAN_DELETE>
    OperationFactory<OP, WRAP, CAN_DELETE>::~OperationFactory(void)
    //--------------------------------------------------------------------------
    {
      static_assert(std::is_base_of<OP, WRAP>::value, "must be derived");
      for (typename std::vector<OP*>::const_iterator it = available.begin();
           it != available.end(); it++)
      {
        // Do explicit deletion to keep valgrind happy
        (*it)->~OP();
        legion_free<OP>(*it, sizeof(WRAP));
      }
    }

    //--------------------------------------------------------------------------
    template<typename OP, typename WRAP, bool CAN_DELETE>
    void OperationFactory<OP, WRAP, CAN_DELETE>::create(OP*& op)
    //--------------------------------------------------------------------------
    {
      if (!available.empty())
      {
        op = available.back();
        available.pop_back();
      }
      else
      {
        static_assert(sizeof(OP) == sizeof(WRAP), "wrapper sizes should match");
        OP* ptr =
            legion_malloc<OP, CAN_DELETE ? SHORT_LIFETIME : RUNTIME_LIFETIME>(
                sizeof(WRAP), alignof(WRAP));
        op = new (ptr) WRAP();
      }
    }

    //--------------------------------------------------------------------------
    template<typename OP, typename WRAP, bool CAN_DELETE>
    void OperationFactory<OP, WRAP, CAN_DELETE>::recycle(OP* op)
    //--------------------------------------------------------------------------
    {
      legion_assert(dynamic_cast<WRAP*>(op) != nullptr);
      available.emplace_back(op);
    }

    //--------------------------------------------------------------------------
    template<typename OP, typename WRAP>
    OperationFactory<OP, WRAP, true>::~OperationFactory(void)
    //--------------------------------------------------------------------------
    {
      static_assert(std::is_base_of<OP, WRAP>::value, "must be derived");
      for (typename std::deque<OP*>::const_iterator it = available.begin();
           it != available.end(); it++)
      {
        // Do explicit deletion to keep valgrind happy
        (*it)->~OP();
        legion_free<OP>(*it, sizeof(WRAP));
      }
    }

    //--------------------------------------------------------------------------
    template<typename OP, typename WRAP>
    void OperationFactory<OP, WRAP, true>::create(OP*& op)
    //--------------------------------------------------------------------------
    {
      if (!available.empty())
      {
        op = available.back();
        available.pop_back();
      }
      else
      {
        static_assert(sizeof(OP) == sizeof(WRAP), "wrapper sizes should match");
        OP* ptr =
            legion_malloc<OP, SHORT_LIFETIME>(sizeof(WRAP), alignof(WRAP));
        op = new (ptr) WRAP();
      }
    }

    //--------------------------------------------------------------------------
    template<typename OP, typename WRAP>
    void OperationFactory<OP, WRAP, true>::recycle(OP* op)
    //--------------------------------------------------------------------------
    {
      legion_assert(dynamic_cast<WRAP*>(op) != nullptr);
      available.emplace_back(op);
      if (available.size() > LEGION_MAX_RECYCLABLE_OBJECTS)
      {
        op = available.front();
        available.pop_front();
        // Do explicit deletion to keep valgrind happy
        op->~OP();
        legion_free<OP>(op, sizeof(WRAP));
      }
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    OperationFactory<OP, OP, false>::~OperationFactory(void)
    //--------------------------------------------------------------------------
    {
      for (typename std::vector<OP*>::const_iterator it = available.begin();
           it != available.end(); it++)
      {
        // Do explicit deletion to keep valgrind happy
        (*it)->~OP();
        legion_free<OP>(*it, sizeof(OP));
      }
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void OperationFactory<OP, OP, false>::create(OP*& op)
    //--------------------------------------------------------------------------
    {
      if (!available.empty())
      {
        op = available.back();
        available.pop_back();
      }
      else
      {
        OP* ptr = legion_malloc<OP, RUNTIME_LIFETIME>(sizeof(OP), alignof(OP));
        op = new (ptr) OP();
      }
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void OperationFactory<OP, OP, false>::recycle(OP* op)
    //--------------------------------------------------------------------------
    {
      available.emplace_back(op);
    }

    // template overrides to help with allocations

    //--------------------------------------------------------------------------
    template<>
    ApEvent Memoizable<AllReduceOp>::compute_sync_precondition(
        const TraceInfo& trace_info) const
    //--------------------------------------------------------------------------
    {
      return this->execution_fence_event;
    }

    //--------------------------------------------------------------------------
    template<>
    ApEvent Memoizable<DynamicCollectiveOp>::compute_sync_precondition(
        const TraceInfo& trace_info) const
    //--------------------------------------------------------------------------
    {
      return this->execution_fence_event;
    }

    //--------------------------------------------------------------------------
    template<>
    ApEvent Memoizable<FenceOp>::compute_sync_precondition(
        const TraceInfo& trace_info) const
    //--------------------------------------------------------------------------
    {
      return this->execution_fence_event;
    }

    //--------------------------------------------------------------------------
    template<>
    ApEvent Memoizable<ReplAllReduceOp>::compute_sync_precondition(
        const TraceInfo& trace_info) const
    //--------------------------------------------------------------------------
    {
      return this->execution_fence_event;
    }

    //--------------------------------------------------------------------------
    template<>
    ApEvent Memoizable<ReplFenceOp>::compute_sync_precondition(
        const TraceInfo& trace_info) const
    //--------------------------------------------------------------------------
    {
      return this->execution_fence_event;
    }

    // explicit instantiations
    template class OperationFactory<
        IndividualTask, Memoizable<Predicated<IndividualTask> > >;
    template class OperationFactory<PointTask, Memoizable<PointTask>, true>;
    template class OperationFactory<
        IndexTask, Memoizable<Predicated<IndexTask> > >;
    template class OperationFactory<SliceTask, Memoizable<SliceTask>, true>;
    template class OperationFactory<MapOp>;
    template class OperationFactory<CopyOp, Memoizable<Predicated<CopyOp> > >;
    template class OperationFactory<
        IndexCopyOp, Memoizable<Predicated<IndexCopyOp> > >;
    template class OperationFactory<PointCopyOp, Memoizable<PointCopyOp> >;
    template class OperationFactory<FenceOp, Memoizable<FenceOp> >;
    template class OperationFactory<FrameOp>;
    template class OperationFactory<CreationOp>;
    template class OperationFactory<DeletionOp>;
    template class OperationFactory<MergeCloseOp>;
    template class OperationFactory<PostCloseOp>;
    template class OperationFactory<RefinementOp>;
    template class OperationFactory<ResetOp>;
    template class OperationFactory<
        DynamicCollectiveOp, Memoizable<DynamicCollectiveOp> >;
    template class OperationFactory<FuturePredOp>;
    template class OperationFactory<NotPredOp>;
    template class OperationFactory<AndPredOp>;
    template class OperationFactory<OrPredOp>;
    template class OperationFactory<
        AcquireOp, Memoizable<Predicated<AcquireOp> > >;
    template class OperationFactory<
        ReleaseOp, Memoizable<Predicated<ReleaseOp> > >;
    template class OperationFactory<TraceBeginOp>;
    template class OperationFactory<TraceRecurrentOp>;
    template class OperationFactory<TraceCompleteOp>;
    template class OperationFactory<MustEpochOp>;
    template class OperationFactory<PendingPartitionOp>;
    template class OperationFactory<DependentPartitionOp>;
    template class OperationFactory<PointDepPartOp, PointDepPartOp, true>;
    template class OperationFactory<FillOp, Memoizable<Predicated<FillOp> > >;
    template class OperationFactory<
        IndexFillOp, Memoizable<Predicated<IndexFillOp> > >;
    template class OperationFactory<PointFillOp, Memoizable<PointFillOp>, true>;
    template class OperationFactory<DiscardOp>;
    template class OperationFactory<AttachOp>;
    template class OperationFactory<IndexAttachOp>;
    template class OperationFactory<PointAttachOp, PointAttachOp, true>;
    template class OperationFactory<DetachOp>;
    template class OperationFactory<IndexDetachOp>;
    template class OperationFactory<PointDetachOp, PointDetachOp, true>;
    template class OperationFactory<TimingOp>;
    template class OperationFactory<TunableOp>;
    template class OperationFactory<AllReduceOp, Memoizable<AllReduceOp> >;
    template class OperationFactory<
        ReplIndividualTask, Memoizable<Predicated<ReplIndividualTask> > >;
    template class OperationFactory<
        ReplIndexTask, Memoizable<Predicated<ReplIndexTask> > >;
    template class OperationFactory<ReplMergeCloseOp>;
    template class OperationFactory<ReplRefinementOp>;
    template class OperationFactory<ReplResetOp>;
    template class OperationFactory<
        ReplFillOp, Memoizable<Predicated<ReplFillOp> > >;
    template class OperationFactory<
        ReplIndexFillOp, Memoizable<Predicated<ReplIndexFillOp> > >;
    template class OperationFactory<
        ReplCopyOp, Memoizable<Predicated<ReplCopyOp> > >;
    template class OperationFactory<
        ReplIndexCopyOp, Memoizable<Predicated<ReplIndexCopyOp> > >;
    template class OperationFactory<ReplDeletionOp>;
    template class OperationFactory<ReplPendingPartitionOp>;
    template class OperationFactory<ReplDependentPartitionOp>;
    template class OperationFactory<ReplMustEpochOp>;
    template class OperationFactory<ReplTimingOp>;
    template class OperationFactory<ReplTunableOp>;
    template class OperationFactory<
        ReplAllReduceOp, Memoizable<ReplAllReduceOp> >;
    template class OperationFactory<ReplFenceOp, Memoizable<ReplFenceOp> >;
    template class OperationFactory<ReplMapOp>;
    template class OperationFactory<ReplDiscardOp>;
    template class OperationFactory<ReplAttachOp>;
    template class OperationFactory<ReplIndexAttachOp>;
    template class OperationFactory<ReplDetachOp>;
    template class OperationFactory<ReplIndexDetachOp>;
    template class OperationFactory<
        ReplAcquireOp, Memoizable<Predicated<ReplAcquireOp> > >;
    template class OperationFactory<
        ReplReleaseOp, Memoizable<Predicated<ReplReleaseOp> > >;
    template class OperationFactory<ReplTraceBeginOp>;
    template class OperationFactory<ReplTraceRecurrentOp>;
    template class OperationFactory<ReplTraceCompleteOp>;

  }  // namespace Internal
}  // namespace Legion
