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

#ifndef __LEGION_TRACE_VIEW_SET_H__
#define __LEGION_TRACE_VIEW_SET_H__

#include "legion/nodes/expression.h"
#include "legion/views/logical.h"

namespace Legion {
  namespace Internal {

    /**
     * \class TraceViewSet
     * The trace view set stores a temporary collection of instance views
     * with valid expressions and fields for each instance. We maintain
     * the important invariant here in this class that each physical
     * instance has at most one view representing it, which requires
     * anti-aliasing collective views.
     */
    class TraceViewSet {
    public:
      struct FailedPrecondition {
      public:
        FailedPrecondition(void) : view(nullptr), expr(nullptr) { }
      public:
        LogicalView* view;
        IndexSpaceExpression* expr;
        FieldMask mask;

        std::string to_string(TaskContext* ctx) const;
      };
    public:
      TraceViewSet(
          InnerContext* context, DistributedID owner_did,
          IndexSpaceExpression* expr, RegionTreeID tree_id);
      ~TraceViewSet(void);
    public:
      void insert(
          LogicalView* view, IndexSpaceExpression* expr, const FieldMask& mask,
          bool antialiased = false);
      void insert(
          const MapView<
              LogicalView*, local::FieldMaskMap<IndexSpaceExpression> >& views,
          bool antialiased = false);
      void invalidate(
          LogicalView* view, IndexSpaceExpression* expr, const FieldMask& mask,
          std::map<IndexSpaceExpression*, unsigned>* expr_refs_to_remove =
              nullptr,
          std::map<LogicalView*, unsigned>* view_refs_to_remove = nullptr,
          bool antialiased = false);
      void invalidate_all_but(
          LogicalView* except, IndexSpaceExpression* expr,
          const FieldMask& mask,
          std::map<IndexSpaceExpression*, unsigned>* expr_refs_to_remove =
              nullptr,
          std::map<LogicalView*, unsigned>* view_refs_to_remove = nullptr,
          bool antialiased = false);
    public:
      bool dominates(
          LogicalView* view, IndexSpaceExpression* expr,
          FieldMask& non_dominated) const;
      void dominates(
          LogicalView* view, IndexSpaceExpression* expr, FieldMask mask,
          local::map<LogicalView*, local::FieldMaskMap<IndexSpaceExpression> >&
              non_dominated) const;
      void filter_independent_fields(
          IndexSpaceExpression* expr, FieldMask& mask) const;
      bool subsumed_by(
          TraceViewSet& set,
          const FieldMapView<IndexSpaceExpression>& unique_dirty_exprs,
          FailedPrecondition* condition = nullptr) const;
      bool independent_of(
          const TraceViewSet& set,
          FailedPrecondition* condition = nullptr) const;
      void record_first_failed(FailedPrecondition* condition = nullptr) const;
      void transpose_uniquely(
          local::map<IndexSpaceExpression*, local::FieldMaskMap<LogicalView> >&
              target) const;
      void find_overlaps(
          TraceViewSet& target, IndexSpaceExpression* expr,
          const bool expr_covers, const FieldMask& mask) const;
      bool empty(void) const;
    public:
      void merge(TraceViewSet& target) const;
      void pack(
          Serializer& rez, AddressSpaceID target,
          const bool pack_references) const;
      void unpack(
          Deserializer& derez, size_t num_views, AddressSpaceID source,
          std::set<RtEvent>& ready_events);
      void unpack_references(void) const;
    public:
      void dump(void) const;
    public:
      InstanceView* find_instance_view(const std::vector<DistributedID>& dids);
    protected:
      bool has_overlapping_expressions(
          LogicalView* view,
          const FieldMapView<IndexSpaceExpression>& left_exprs,
          const FieldMapView<IndexSpaceExpression>& right_exprs,
          FailedPrecondition* condition) const;
      void antialias_individual_view(IndividualView* view, FieldMask mask);
      void antialias_collective_view(
          CollectiveView* view, FieldMask mask,
          local::FieldMaskMap<InstanceView>& altviews);
    protected:
      typedef shrt::map<LogicalView*, shrt::FieldMaskMap<IndexSpaceExpression> >
          ViewExprs;
    public:
      InnerContext* const context;
      IndexSpaceExpression* const expression;
      const RegionTreeID tree_id;
      const DistributedID owner_did;
    protected:
      // At most one expression per field
      ViewExprs conditions;
      mutable std::map<LogicalView*, LamportClock> lamport_clocks;
      bool has_collective_views;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_TRACE_VIEW_SET_H__
