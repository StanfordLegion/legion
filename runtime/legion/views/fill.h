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

#ifndef __LEGION_FILL_VIEW_H__
#define __LEGION_FILL_VIEW_H__

#include "legion/kernel/metatask.h"
#include "legion/tracing/recording.h"
#include "legion/views/deferred.h"

namespace Legion {
  namespace Internal {

    /**
     * \class FillView
     * This is a deferred view that is used for filling in
     * fields with a default value.
     */
    class FillView : public DeferredView,
                     public Heapify<FillView, CONTEXT_LIFETIME> {
    public:
      struct DeferIssueFill : public LgTaskArgs<DeferIssueFill> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_ISSUE_FILL_TASK_ID;
      public:
        DeferIssueFill(void) = default;
        DeferIssueFill(
            FillView* view, Operation* op, IndexSpaceExpression* fill_expr,
            IndividualView* dst_view, const FieldMask& fill_mask,
            const PhysicalTraceInfo& trace_info,
            const std::vector<CopySrcDstField>& dst_fields,
            PhysicalManager* manager, ApEvent precondition,
            PredEvent pred_guard, CollectiveKind collective, bool fill_restrict,
            std::set<RtEvent>& applied);
        void execute(void) const;
      public:
        FillView* view;
        Operation* op;
        IndexSpaceExpression* fill_expr;
        IndividualView* dst_view;
        const HeapifyBox<FieldMask, OPERATION_LIFETIME>* fill_mask;
        PhysicalTraceInfo* trace_info;
        std::vector<CopySrcDstField>* dst_fields;
        PhysicalManager* manager;
        ApEvent precondition;
        PredEvent pred_guard;
        CollectiveKind collective;
        RtUserEvent applied;
        ApUserEvent done;
        bool fill_restricted;
      };
    public:
      // Don't know the fill value yet, will be set later
      FillView(
          DistributedID did, UniqueID fill_op_uid, bool register_now,
          CollectiveMapping* mapping = nullptr);
      // Already know the fill value
      FillView(
          DistributedID did, UniqueID fill_op_uid, const void* value,
          size_t size, bool register_now, CollectiveMapping* mapping = nullptr);
      FillView(const FillView& rhs) = delete;
      virtual ~FillView(void);
    public:
      FillView& operator=(const FillView& rhs) = delete;
    public:
      virtual void notify_local(void) override { /*nothing to do*/ }
      virtual void pack_valid_ref(
          shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
          shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks)
          override;
      virtual void unpack_valid_ref(
          shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
          shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks)
          override;
    public:
      virtual void send_view(AddressSpaceID target) override;
    public:
      virtual void flatten(
          CopyFillAggregator& aggregator, InstanceView* dst_view,
          const FieldMask& src_mask, IndexSpaceExpression* expr,
          PredEvent pred_guard, const PhysicalTraceInfo& trace_info,
          EquivalenceSet* tracing_eq, CopyAcrossHelper* helper) override;
    public:
      bool matches(FillView* other);
      bool matches(const void* value, size_t size);
      bool set_value(const void* value, size_t size);
      ApEvent issue_fill(
          Operation* op, IndexSpaceExpression* fill_expr,
          IndividualView* dst_view, const FieldMask& fill_mask,
          const PhysicalTraceInfo& trace_info,
          const std::vector<CopySrcDstField>& dst_fields,
          std::set<RtEvent>& applied_events, PhysicalManager* manager,
          ApEvent precondition, PredEvent pred_guard, CollectiveKind collective,
          bool fill_restricted);
    public:
      const UniqueID fill_op_uid;
    protected:
      std::atomic<void*> value;
      std::atomic<size_t> value_size;
      RtUserEvent value_ready;
      // To help with reference counting creation on collective fill views
      // we don't need to actually send the updates on our first active call
      // Note that this only works the fill view will eventually becomes
      // active on all the nodes of the collective mapping, which currently
      // it does, but that is a higher-level invariant maintained by the
      // fill view creation and not the fill view itself
      bool collective_first_active;
    };

    //--------------------------------------------------------------------------
    inline FillView* LogicalView::as_fill_view(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(is_fill_view());
      return static_cast<FillView*>(const_cast<LogicalView*>(this));
    }

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_FILL_VIEW_H__
