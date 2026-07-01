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

#ifndef __LEGION_ALLREDUCE_VIEW_H__
#define __LEGION_ALLREDUCE_VIEW_H__

#include "legion/views/collective.h"

namespace Legion {
  namespace Internal {

    /**
     * \class AllreduceView
     * This class represents a group of reduction instances that
     * all need to be reduced together to produce valid reduction data
     */
    class AllreduceView : public CollectiveView,
                          public Heapify<AllreduceView, CONTEXT_LIFETIME> {
    public:
      AllreduceView(
          DistributedID did, DistributedID ctx_did,
          const std::vector<IndividualView*>& views,
          const std::vector<DistributedID>& instances, bool register_now,
          CollectiveMapping* mapping, ReductionOpID redop_id);
      AllreduceView(const AllreduceView& rhs) = delete;
      virtual ~AllreduceView(void);
    public:
      AllreduceView& operator=(const AllreduceView& rhs) = delete;
    public:  // From InstanceView
      virtual void send_view(AddressSpaceID target) override;
      virtual ReductionOpID get_redop(void) const override { return redop; }
      virtual FillView* get_redop_fill_view(void) const override
      {
        return fill_view;
      }
    public:
      void perform_collective_reduction(
          const std::vector<CopySrcDstField>& dst_fields,
          const std::vector<Reservation>& reservations, ApEvent precondition,
          PredEvent predicate_guard, IndexSpaceExpression* copy_expresison,
          IndexSpace upper_bound, Operation* op, const unsigned index,
          const FieldMask& copy_mask, const FieldMask& dst_mask,
          const DistributedID src_inst_did, const UniqueInst& dst_inst,
          const LgEvent dst_unique_event, const PhysicalTraceInfo& trace_info,
          const CollectiveKind collective_kind,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          ApUserEvent result, AddressSpaceID origin);
      // Degenerate case
      ApEvent perform_hammer_reduction(
          const std::vector<CopySrcDstField>& dst_fields,
          const std::vector<Reservation>& reservations, ApEvent precondition,
          PredEvent predicate_guard, IndexSpaceExpression* copy_expresison,
          IndexSpace upper_bound, Operation* op, const unsigned index,
          const FieldMask& copy_mask, const FieldMask& dst_mask,
          const UniqueInst& dst_inst, const LgEvent dst_unique_event,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          AddressSpaceID origin);
      void perform_collective_allreduce(
          ApEvent precondition, PredEvent predicate_guard,
          IndexSpaceExpression* copy_expresison, IndexSpace upper_bound,
          Operation* op, const unsigned index, const FieldMask& copy_mask,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          const uint64_t allreduce_tag);
      void process_distribute_allreduce(
          const uint64_t allreduce_tag, const int src_rank, const int stage,
          std::vector<CopySrcDstField>& src_fields,
          const ApEvent src_precondition, ApUserEvent src_postcondition,
          ApBarrier src_barrier, ShardID bar_shard, const UniqueInst& src_inst,
          const LgEvent src_unique_event);
      uint64_t generate_unique_allreduce_tag(void);
    protected:
      inline void set_redop(std::vector<CopySrcDstField>& fields) const
      {
        legion_assert(redop > 0);
        for (CopySrcDstField& field : fields)
          field.set_redop(redop, true /*fold*/, true /*exclusive*/);
      }
      inline void clear_redop(std::vector<CopySrcDstField>& fields) const
      {
        for (CopySrcDstField& field : fields)
          field.set_redop(0 /*redop*/, false /*fold*/);
      }
      bool is_multi_instance(void);
      void perform_single_allreduce(
          const uint64_t allreduce_tag, Operation* op, unsigned index,
          ApEvent precondition, PredEvent predicate_guard,
          IndexSpaceExpression* copy_expression, IndexSpace upper_bound,
          const FieldMask& copy_mask, const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events,
          std::set<RtEvent>& applied_events);
      void perform_multi_allreduce(
          const uint64_t allreduce_tag, Operation* op, unsigned index,
          ApEvent precondition, PredEvent predicate_guard,
          IndexSpaceExpression* copy_expression, IndexSpace upper_bound,
          const FieldMask& copy_mask, const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events,
          std::set<RtEvent>& applied_events);
      ApEvent initialize_allreduce_with_reductions(
          ApEvent precondition, PredEvent predicate_guard, Operation* op,
          unsigned index, IndexSpaceExpression* copy_expression,
          IndexSpace upper_bound, const FieldMask& copy_mask,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& applied_events,
          std::vector<ApEvent>& instance_events,
          std::vector<std::vector<CopySrcDstField> >& local_fields,
          std::vector<std::vector<Reservation> >& reservations);
      void complete_initialize_allreduce_with_reductions(
          Operation* op, unsigned index, IndexSpaceExpression* copy_expression,
          IndexSpace upper_bound, const FieldMask& copy_mask,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          std::vector<ApEvent>& instance_events,
          std::vector<std::vector<CopySrcDstField> >& local_fields,
          std::vector<ApEvent>* reduced = nullptr);
      void initialize_allreduce_without_reductions(
          ApEvent precondition, PredEvent predicate_guard, Operation* op,
          unsigned index, IndexSpaceExpression* copy_expression,
          IndexSpace upper_bound, const FieldMask& copy_mask,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          std::vector<ApEvent>& instance_events,
          std::vector<std::vector<CopySrcDstField> >& local_fields,
          std::vector<std::vector<Reservation> >& reservations);
      ApEvent finalize_allreduce_with_broadcasts(
          PredEvent predicate_guard, Operation* op, unsigned index,
          IndexSpaceExpression* copy_expression, IndexSpace upper_bound,
          const FieldMask& copy_mask, const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          std::vector<ApEvent>& instance_events,
          const std::vector<std::vector<CopySrcDstField> >& local_fields,
          const unsigned final_index = 0);
      void complete_finalize_allreduce_with_broadcasts(
          Operation* op, unsigned index, IndexSpaceExpression* copy_expression,
          IndexSpace upper_bound, const FieldMask& copy_mask,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events,
          const std::vector<ApEvent>& instance_events,
          std::vector<ApEvent>* broadcast = nullptr,
          const unsigned final_index = 0);
      void finalize_allreduce_without_broadcasts(
          PredEvent predicate_guard, Operation* op, unsigned index,
          IndexSpaceExpression* copy_expression, IndexSpace upper_bound,
          const FieldMask& copy_mask, const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          std::vector<ApEvent>& instance_events,
          const std::vector<std::vector<CopySrcDstField> >& local_fields,
          const unsigned finalize_index = 0);
      void send_allreduce_stage(
          const uint64_t allreduce_tag, const int stage, const int local_rank,
          ApEvent src_precondition, PredEvent predicate_guard,
          IndexSpaceExpression* copy_expression,
          const PhysicalTraceInfo& trace_info,
          const std::vector<CopySrcDstField>& src_fields,
          const unsigned src_index, const AddressSpaceID* targets, size_t total,
          std::vector<ApEvent>& read_events);
      void receive_allreduce_stage(
          const unsigned dst_index, const uint64_t allreduce_tag,
          const int stage, Operation* op, ApEvent dst_precondition,
          PredEvent predicate_guard, IndexSpaceExpression* copy_expression,
          const FieldMask& copy_mask, const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& applied_events,
          const std::vector<CopySrcDstField>& dst_fields,
          const std::vector<Reservation>& reservations,
          const int* expected_ranks, size_t total_ranks,
          std::vector<ApEvent>& reduce_events);
      void reduce_local(
          const PhysicalManager* dst_manager, const unsigned dst_index,
          Operation* op, const unsigned index, IndexSpaceExpression* copy_expr,
          IndexSpace upper_bound, const FieldMask& copy_mask,
          ApEvent precondition, PredEvent predicate_guard,
          const std::vector<CopySrcDstField>& dst_fields,
          const std::vector<Reservation>& dst_reservations,
          const UniqueInst& dst_inst, const PhysicalTraceInfo& trace_info,
          const CollectiveKind collective_kind,
          std::vector<ApEvent>& reduced_events,
          std::set<RtEvent>& applied_events,
          std::set<RtEvent>* recorded_events = nullptr,
          const bool prepare_allreduce = false,
          std::vector<std::vector<CopySrcDstField> >* src_fields = nullptr);
    public:
      const ReductionOpID redop;
      const ReductionOp* const reduction_op;
      FillView* const fill_view;
    protected:
      struct CopyKey {
      public:
        CopyKey(void) : tag(0), rank(0), stage(0) { }
        CopyKey(uint64_t t, int r, int s) : tag(t), rank(r), stage(s) { }
      public:
        inline bool operator==(const CopyKey& rhs) const
        {
          return (tag == rhs.tag) && (rank == rhs.rank) && (stage == rhs.stage);
        }
        inline bool operator<(const CopyKey& rhs) const
        {
          if (tag < rhs.tag)
            return true;
          if (tag > rhs.tag)
            return false;
          if (rank < rhs.rank)
            return true;
          if (rank > rhs.rank)
            return false;
          return (stage < rhs.stage);
        }
      public:
        uint64_t tag;
        int rank, stage;
      };
      struct AllReduceCopy {
        std::vector<CopySrcDstField> src_fields;
        ApEvent src_precondition;
        ApUserEvent src_postcondition;
        ApBarrier barrier_postcondition;
        ShardID barrier_shard;
        UniqueInst src_inst;
        LgEvent src_unique_event;
      };
      std::map<CopyKey, AllReduceCopy> all_reduce_copies;
      struct AllReduceStage {
        unsigned dst_index;
        Operation* op;
        IndexSpaceExpression* copy_expression;
        FieldMask copy_mask;
        std::vector<CopySrcDstField> dst_fields;
        std::vector<Reservation> reservations;
        PhysicalTraceInfo* trace_info;
        ApEvent dst_precondition;
        PredEvent predicate_guard;
        std::vector<ApUserEvent> remaining_postconditions;
        std::set<RtEvent> applied_events;
        RtUserEvent applied_event;
      };
      op::map<std::pair<uint64_t, int>, AllReduceStage> remaining_stages;
    protected:
      std::atomic<uint64_t> unique_allreduce_tag;
      // A boolean flag that says whether this collective instance
      // has multiple instances on every node. This is primarily
      // useful for reduction instances where we want to pick an
      // algorithm for performing an in-place all-reduce
      std::atomic<bool> multi_instance;
      // Whether we've computed multi instance or not
      std::atomic<bool> evaluated_multi_instance;
    };

    //--------------------------------------------------------------------------
    inline AllreduceView* LogicalView::as_allreduce_view(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(is_allreduce_view());
      return static_cast<AllreduceView*>(const_cast<LogicalView*>(this));
    }

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_ALLREDUCE_VIEW_H__
