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

#ifndef __LEGION_DEFERRED_VIEW_H__
#define __LEGION_DEFERRED_VIEW_H__

#include "legion/views/logical.h"

namespace Legion {
  namespace Internal {

    /**
     * \class DeferredView
     * A DeferredView class is an abstract class for representing
     * lazy computation in an equivalence set. At the moment, the
     * types only allow deferred views to capture other kinds of
     * lazy evaluation. In particular this is either a fill view
     * or nested levels of predicated fill views. It could also
     * support other kinds of lazy evaluation as well. Importantly,
     * since it only captures lazy computation and not materialized
     * data there are no InstanceView managers captured in its
     * representations, which is an important invariant for the
     * equivalence sets. Think long and hard about what you're
     * doing if you ever decide that you want to break that
     * invariant and capture the names of instance views inside
     * of a deferred view.
     */
    class DeferredView : public LogicalView {
    public:
      DeferredView(
          DistributedID did, bool register_now,
          CollectiveMapping* mapping = nullptr);
      virtual ~DeferredView(void);
    public:
      virtual void send_view(AddressSpaceID target) override = 0;
    public:
      virtual void flatten(
          CopyFillAggregator& aggregator, InstanceView* dst_view,
          const FieldMask& src_mask, IndexSpaceExpression* expr,
          PredEvent pred_guard, const PhysicalTraceInfo& trace_info,
          EquivalenceSet* tracing_eq, CopyAcrossHelper* helper) = 0;
    public:
      virtual void notify_valid(void) override;
      virtual bool notify_invalid(void) override;
    };

    //--------------------------------------------------------------------------
    inline DeferredView* LogicalView::as_deferred_view(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(is_deferred_view());
      return static_cast<DeferredView*>(const_cast<LogicalView*>(this));
    }

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_DEFERRED_VIEW_H__
