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

#ifndef __LEGION_PHI_VIEW_H__
#define __LEGION_PHI_VIEW_H__

#include "legion/views/deferred.h"
#include "legion/kernel/metatask.h"
#include "legion/utilities/fieldmask_map.h"

namespace Legion {
  namespace Internal {

    /**
     * \class PhiView
     * A phi view is exactly what it sounds like: a view to merge two
     * different views together from different control flow paths.
     * Specifically it is able to merge together different paths for
     * predication so that we can issue copies from both a true and
     * a false version of a predicate. This allows us to map past lazy
     * predicated operations such as fills and virtual mappings and
     * continue to get ahead of actual execution. It's not pretty
     * but it seems to work.
     */
    class PhiView : public DeferredView,
                    public Heapify<PhiView, CONTEXT_LIFETIME> {
    public:
      struct DeferPhiViewRegistrationArgs
        : public LgTaskArgs<DeferPhiViewRegistrationArgs> {
      public:
        static constexpr LgTaskID TASK_ID =
            LG_DEFER_PHI_VIEW_REGISTRATION_TASK_ID;
      public:
        DeferPhiViewRegistrationArgs(void) = default;
        DeferPhiViewRegistrationArgs(
            PhiView* v, shrt::map<DeferredView*, LamportClock>& clocks)
          : LgTaskArgs<DeferPhiViewRegistrationArgs>(false, true), view(v),
            lamport_clocks(new shrt::map<DeferredView*, LamportClock>())
        {
          lamport_clocks->swap(clocks);
        }
        void execute(void) const;
      public:
        PhiView* view;
        shrt::map<DeferredView*, LamportClock>* lamport_clocks;
      };
    public:
      PhiView(
          DistributedID did, PredEvent true_guard, PredEvent false_guard,
          shrt::FieldMaskMap<DeferredView>&& true_views,
          shrt::FieldMaskMap<DeferredView>&& false_views,
          bool register_now = true);
      PhiView(const PhiView& rhs) = delete;
      virtual ~PhiView(void);
    public:
      PhiView& operator=(const PhiView& rhs) = delete;
    public:
      virtual void notify_local(void) override;
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
          EquivalenceSet* tracign_eq, CopyAcrossHelper* helper) override;
    public:
      void add_initial_references(
          shrt::map<DeferredView*, LamportClock>& lamport_clocks);
    public:
      const PredEvent true_guard;
      const PredEvent false_guard;
      const shrt::FieldMaskMap<DeferredView> true_views, false_views;
    };

    //--------------------------------------------------------------------------
    inline PhiView* LogicalView::as_phi_view(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(is_phi_view());
      return static_cast<PhiView*>(const_cast<LogicalView*>(this));
    }

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_PHI_VIEW_H__
