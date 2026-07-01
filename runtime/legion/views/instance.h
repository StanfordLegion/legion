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

#ifndef __LEGION_INSTANCE_VIEW_H__
#define __LEGION_INSTANCE_VIEW_H__

#include "legion/views/logical.h"
#include "legion/api/data.h"
#include "legion/utilities/fieldmask_map.h"

namespace Legion {
  namespace Internal {

    /**
     * \class InstanceView
     * The InstanceView class is used for performing the dependence
     * analysis for a single physical instance.
     * The InstaceView class has two sub-classes: materialized
     * views which represent a normal instance a reduction
     * view which is a specialized instance for storing reductions
     */
    class InstanceView : public LogicalView {
    public:
      // This structure acts as a key for performing rendezvous
      // between collective user registrations
      struct RendezvousKey {
      public:
        RendezvousKey(void) : op_context_index(0), match(0), index(0) { }
        RendezvousKey(size_t ctx, unsigned idx, const IndexSpace& m)
          : op_context_index(ctx), match(m.get_id()), index(idx)
        { }
      public:
        inline bool operator<(const RendezvousKey& rhs) const
        {
          if (op_context_index < rhs.op_context_index)
            return true;
          if (op_context_index > rhs.op_context_index)
            return false;
          if (match < rhs.match)
            return true;
          if (match > rhs.match)
            return false;
          return (index < rhs.index);
        }
      public:
        size_t op_context_index;  // unique name operation in context
        DistributedID match;      // index space of regions that should match
        unsigned index;  // uniquely name analysis for op by region req index
      };
    public:
      InstanceView(
          DistributedID did, bool register_now, CollectiveMapping* mapping);
      virtual ~InstanceView(void);
    public:
      virtual ApEvent fill_from(
          FillView* fill_view, ApEvent precondition, PredEvent predicate_guard,
          IndexSpaceExpression* expression, Operation* op, const unsigned index,
          const IndexSpace collective_match_space, const FieldMask& fill_mask,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          CopyAcrossHelper* across_helper, const bool manage_dst_events,
          const bool fill_restricted, const bool need_valid_return) = 0;
      virtual ApEvent copy_from(
          InstanceView* src_view, ApEvent precondition,
          PredEvent predicate_guard, ReductionOpID redop,
          IndexSpaceExpression* expression, Operation* op, const unsigned index,
          const IndexSpace collective_match_space, const FieldMask& copy_mask,
          PhysicalManager* src_point, const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
          CopyAcrossHelper* across_helper, const bool manage_dst_events,
          const bool copy_restricted, const bool need_valid_return) = 0;
      // Always want users to be full index space expressions
      virtual ApEvent register_user(
          const RegionUsage& usage, const FieldMask& user_mask,
          IndexSpaceNode* expr, const UniqueID op_id, const size_t op_ctx_index,
          const unsigned index, ApEvent term_event, PhysicalManager* target,
          CollectiveMapping* collective_mapping,
          size_t local_collective_arrivals,
          std::vector<RtEvent>& registered_events,
          std::set<RtEvent>& applied_events,
          const PhysicalTraceInfo& trace_info, const AddressSpaceID source,
          const bool symbolic = false) = 0;
    public:
      virtual void send_view(AddressSpaceID target) = 0;
      virtual ReductionOpID get_redop(void) const { return 0; }
      virtual FillView* get_redop_fill_view(void) const { std::abort(); }
      virtual AddressSpaceID get_analysis_space(PhysicalManager* man) const = 0;
      virtual bool aliases(InstanceView* other) const = 0;
    };

    //--------------------------------------------------------------------------
    inline InstanceView* LogicalView::as_instance_view(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(is_instance_view());
      return static_cast<InstanceView*>(const_cast<LogicalView*>(this));
    }

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_INSTANCE_VIEW_H__
