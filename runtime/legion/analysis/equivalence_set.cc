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

#include "legion/analysis/equivalence_set.h"
#include "legion/analysis/acquire.h"
#include "legion/analysis/across.h"
#include "legion/analysis/aggregator.h"
#include "legion/analysis/collective.h"
#include "legion/analysis/filter.h"
#include "legion/analysis/overwrite.h"
#include "legion/analysis/release.h"
#include "legion/analysis/update.h"
#include "legion/analysis/valid.h"
#include "legion/analysis/versioning.h"
#include "legion/contexts/inner.h"
#include "legion/nodes/across.h"
#include "legion/nodes/expression.h"
#include "legion/nodes/index.h"
#include "legion/nodes/kdtree.h"
#include "legion/instances/physical.h"
#include "legion/tracing/template.h"
#include "legion/tracing/viewset.h"
#include "legion/views/individual.h"
#include "legion/views/allreduce.h"
#include "legion/views/fill.h"
#include "legion/views/phi.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // CollectiveRefinementTree
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename T>
    CollectiveRefinementTree<T>::CollectiveRefinementTree(CollectiveView* c)
      : collective(c)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<typename T>
    CollectiveRefinementTree<T>::CollectiveRefinementTree(
        std::vector<DistributedID>&& dids)
      : collective(nullptr), inst_dids(dids)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<typename T>
    CollectiveRefinementTree<T>::~CollectiveRefinementTree(void)
    //--------------------------------------------------------------------------
    {
      for (typename local::FieldMaskMap<T>::const_iterator it =
               refinements.begin();
           it != refinements.end(); it++)
        delete it->first;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    const std::vector<DistributedID>&
        CollectiveRefinementTree<T>::get_instances(void) const
    //--------------------------------------------------------------------------
    {
      if (collective != nullptr)
        return collective->instances;
      else
        return inst_dids;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    template<typename... Args>
    void CollectiveRefinementTree<T>::traverse(
        InstanceView* view, const FieldMask& mask, Args&&... args)
    //--------------------------------------------------------------------------
    {
      // Check first for any instance overlap first
      const std::vector<DistributedID>& local_dids = get_instances();
      if (view->is_individual_view())
      {
        const DistributedID inst_did =
            view->as_individual_view()->get_manager()->did;
        // No instance over means there's nothing to do
        if (!std::binary_search(local_dids.begin(), local_dids.end(), inst_did))
          return;
        // See if we need to be refined
        if (local_dids.size() > 1)
        {
          // Make the refinements for all the fields that aren't already refined
          const FieldMask refinement_mask = mask - get_refinement_mask();
          if (!!refinement_mask)
          {
            std::vector<DistributedID> single_did(1, inst_did);
            refinements.insert(
                static_cast<T*>(this)->clone(
                    view, refinement_mask, std::move(single_did)),
                refinement_mask);
            std::vector<DistributedID> remainder_dids;
            remainder_dids.reserve(local_dids.size() - 1);
            for (const DistributedID& it : local_dids)
              if (it != inst_did)
                remainder_dids.emplace_back(it);
            refinements.insert(
                static_cast<T*>(this)->clone(
                    (InstanceView*)nullptr /*no view*/, refinement_mask,
                    std::move(remainder_dids)),
                refinement_mask);
          }
        }
      }
      else
      {
        // Compute the intersection first
        CollectiveView* collective_view = view->as_collective_view();
        std::vector<DistributedID> overlap;
        std::vector<DistributedID>::const_iterator first =
            collective_view->instances.begin();
        std::vector<DistributedID>::const_iterator second = local_dids.begin();
        while ((first != collective_view->instances.end()) &&
               (second != local_dids.end()))
        {
          if (*first < *second)
            first++;
          else if (*second < *first)
            second++;
          else
          {
            overlap.emplace_back(*first);
            first++;
            second++;
          }
        }
        // No instance overlap means we don't even need to be here
        if (overlap.empty())
          return;
        if (overlap.size() != local_dids.size())
        {
          // Make the refinements for all the fields that aren't already refined
          const FieldMask refinement_mask = mask - get_refinement_mask();
          if (!!refinement_mask)
          {
            // compute the difference and make the refinements
            std::vector<DistributedID> remainder;
            first = local_dids.begin();
            second = collective_view->instances.begin();
            while (first != local_dids.end())
            {
              if ((second == collective_view->instances.end()) ||
                  (*first < *second))
                remainder.emplace_back(*first++);
              else if (*second < *first)
                second++;
              else
              {
                first++;
                second++;
              }
            }
            refinements.insert(
                static_cast<T*>(this)->clone(
                    (overlap.size() == collective_view->instances.size()) ?
                        collective_view :
                        (InstanceView*)nullptr,
                    refinement_mask, std::move(overlap)),
                refinement_mask);
            refinements.insert(
                static_cast<T*>(this)->clone(
                    (InstanceView*)nullptr /*view*/, refinement_mask,
                    std::move(remainder)),
                refinement_mask);
          }
        }
      }
      for (typename local::FieldMaskMap<T>::const_iterator it =
               refinements.begin();
           it != refinements.end(); it++)
      {
        const FieldMask& overlap = mask & it->second;
        if (!overlap)
          continue;
        it->first->traverse(view, overlap, args...);
      }
      const FieldMask local_mask = mask - refinements.get_valid_mask();
      if (!!local_mask)
        static_cast<T*>(this)->analyze(view, local_mask, args...);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    InstanceView* CollectiveRefinementTree<T>::get_instance_view(
        InnerContext* context, RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      if (collective != nullptr)
        return collective;
      legion_assert(!inst_dids.empty());
      if (inst_dids.size() > 1)
      {
        RtEvent ready;
        InnerContext::CollectiveResult* result =
            context->find_or_create_collective_view(tid, inst_dids, ready);
        // At some point in the future we might want to stop doing
        // all this waiting here and turn these into continuations
        // Wait for the collective did to be ready
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        // Then wait for the collective view to be registered
        if (result->ready_event.exists() &&
            !result->ready_event.has_triggered())
          result->ready_event.wait();
        InstanceView* view =
            static_cast<InstanceView*>(runtime->find_or_request_logical_view(
                result->collective_did, ready));
        if (result->remove_reference())
          delete result;
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        return view;
      }
      else
      {
        RtEvent ready;
        PhysicalManager* manager =
            runtime->find_or_request_instance_manager(inst_dids.back(), ready);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        return context->create_instance_top_view(
            manager, runtime->address_space);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void CollectiveRefinementTree<T>::traverse_total(
        const FieldMask& mask, IndexSpaceExpression* set_expr,
        const FieldMapView<LogicalView>& total_valid_views)
    //--------------------------------------------------------------------------
    {
      if (mask * total_valid_views.get_valid_mask())
        return;
      for (FieldMapView<LogicalView>::const_iterator it =
               total_valid_views.begin();
           it != total_valid_views.end(); it++)
      {
        if (!it->first->is_instance_view())
          continue;
        const FieldMask overlap = mask & it->second;
        if (!overlap)
          continue;
        InstanceView* inst_view = it->first->as_instance_view();
        if (inst_view->aliases(collective))
          traverse(inst_view, overlap, set_expr);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void CollectiveRefinementTree<T>::traverse_partial(
        const FieldMask& mask,
        const MapView<LogicalView*, shrt::FieldMaskMap<IndexSpaceExpression>>&
            partial_valid_views)
    //--------------------------------------------------------------------------
    {
      for (MapView<LogicalView*, shrt::FieldMaskMap<IndexSpaceExpression>>::
               const_iterator it = partial_valid_views.begin();
           it != partial_valid_views.end(); it++)
      {
        if (!it->first->is_instance_view())
          continue;
        const FieldMask overlap = mask & it->second.get_valid_mask();
        if (!overlap)
          continue;
        InstanceView* inst_view = it->first->as_instance_view();
        if (inst_view->aliases(collective))
          traverse(inst_view, overlap, it->second);
      }
    }

    /////////////////////////////////////////////////////////////
    // MakeCollectiveValid
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MakeCollectiveValid::MakeCollectiveValid(
        CollectiveView* v, const FieldMapView<IndexSpaceExpression>& exprs)
      : CollectiveRefinementTree<MakeCollectiveValid>(v), view(v),
        needed_exprs(exprs)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    MakeCollectiveValid::MakeCollectiveValid(
        std::vector<DistributedID>&& insts,
        const FieldMapView<IndexSpaceExpression>& exprs,
        const FieldMapView<IndexSpaceExpression>& valid, const FieldMask& mask,
        InstanceView* v)
      : CollectiveRefinementTree<MakeCollectiveValid>(std::move(insts)),
        view(v), needed_exprs(exprs)
    //--------------------------------------------------------------------------
    {
      analyze(view, mask, valid);
    }

    //--------------------------------------------------------------------------
    MakeCollectiveValid* MakeCollectiveValid::clone(
        InstanceView* view, const FieldMask& mask,
        std::vector<DistributedID>&& insts) const
    //--------------------------------------------------------------------------
    {
      return new MakeCollectiveValid(
          std::move(insts), needed_exprs, valid_exprs, mask, view);
    }

    //--------------------------------------------------------------------------
    void MakeCollectiveValid::analyze(
        InstanceView* view, const FieldMask& mask, IndexSpaceExpression* expr)
    //--------------------------------------------------------------------------
    {
      valid_exprs.insert(expr, mask);
    }

    //--------------------------------------------------------------------------
    void MakeCollectiveValid::analyze(
        InstanceView* view, const FieldMask& mask,
        const FieldMapView<IndexSpaceExpression>& exprs)
    //--------------------------------------------------------------------------
    {
      if (!(exprs.get_valid_mask() - mask))
      {
        for (FieldMapView<IndexSpaceExpression>::const_iterator it =
                 exprs.begin();
             it != exprs.end(); it++)
          valid_exprs.insert(it->first, it->second);
      }
      else
      {
        for (FieldMapView<IndexSpaceExpression>::const_iterator it =
                 exprs.begin();
             it != exprs.end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          valid_exprs.insert(it->first, overlap);
        }
      }
    }

    //--------------------------------------------------------------------------
    void MakeCollectiveValid::visit_leaf(
        const FieldMask& mask, InnerContext* context, RegionTreeID tid,
        local::map<InstanceView*, local::FieldMaskMap<IndexSpaceExpression>>&
            updates)
    //--------------------------------------------------------------------------
    {
      local::list<FieldSet<IndexSpaceExpression*>> field_sets;
      valid_exprs.compute_field_sets(mask, field_sets);
      InstanceView* local_view = view;
      for (local::list<FieldSet<IndexSpaceExpression*>>::const_iterator fit =
               field_sets.begin();
           fit != field_sets.end(); fit++)
      {
        IndexSpaceExpression* valid_expr =
            fit->elements.empty() ? nullptr :
            (fit->elements.size() == 1) ?
                                    *(fit->elements.begin()) :
                                    runtime->union_index_spaces(fit->elements);
        for (FieldMapView<IndexSpaceExpression>::const_iterator it =
                 needed_exprs.begin();
             it != needed_exprs.end(); it++)
        {
          const FieldMask overlap = fit->set_mask & it->second;
          if (!overlap)
            continue;
          IndexSpaceExpression* needed_expr = it->first;
          if (valid_expr != nullptr)
            needed_expr =
                runtime->subtract_index_spaces(needed_expr, valid_expr);
          if (!needed_expr->is_empty())
          {
            if (local_view == nullptr)
              local_view = get_instance_view(context, tid);
            updates[local_view].insert(needed_expr, overlap);
          }
        }
      }
    }

    // Explicit instantiation for CollectiveRefinementTree
    template class CollectiveRefinementTree<MakeCollectiveValid>;

    /////////////////////////////////////////////////////////////
    // CollectiveAntiAlias
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CollectiveAntiAlias::CollectiveAntiAlias(CollectiveView* v)
      : CollectiveRefinementTree<CollectiveAntiAlias>(v)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    CollectiveAntiAlias::CollectiveAntiAlias(
        std::vector<DistributedID>&& insts,
        const FieldMapView<IndexSpaceExpression>& valid, const FieldMask& mask)
      : CollectiveRefinementTree<CollectiveAntiAlias>(std::move(insts))
    //--------------------------------------------------------------------------
    {
      analyze((InstanceView*)nullptr, mask, valid);
    }

    //--------------------------------------------------------------------------
    CollectiveAntiAlias* CollectiveAntiAlias::clone(
        InstanceView* view, const FieldMask& mask,
        std::vector<DistributedID>&& insts) const
    //--------------------------------------------------------------------------
    {
      return new CollectiveAntiAlias(std::move(insts), valid_exprs, mask);
    }

    //--------------------------------------------------------------------------
    void CollectiveAntiAlias::analyze(
        InstanceView* view, const FieldMask& mask, IndexSpaceExpression* expr)
    //--------------------------------------------------------------------------
    {
      valid_exprs.insert(expr, mask);
    }

    //--------------------------------------------------------------------------
    void CollectiveAntiAlias::analyze(
        InstanceView* view, const FieldMask& mask,
        const FieldMapView<IndexSpaceExpression>& exprs)
    //--------------------------------------------------------------------------
    {
      if (!(exprs.get_valid_mask() - mask))
      {
        for (FieldMapView<IndexSpaceExpression>::const_iterator it =
                 exprs.begin();
             it != exprs.end(); it++)
          valid_exprs.insert(it->first, it->second);
      }
      else
      {
        for (FieldMapView<IndexSpaceExpression>::const_iterator it =
                 exprs.begin();
             it != exprs.end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          valid_exprs.insert(it->first, overlap);
        }
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveAntiAlias::visit_leaf(
        const FieldMask& mask, FieldMask& allvalid_mask,
        IndexSpaceExpression* expr,
        const FieldMapView<IndexSpaceExpression>& partial_valid_exprs)
    //--------------------------------------------------------------------------
    {
      if (!allvalid_mask)
        return;
      if (!partial_valid_exprs.empty())
      {
        for (FieldMapView<IndexSpaceExpression>::const_iterator it =
                 partial_valid_exprs.begin();
             it != partial_valid_exprs.end(); it++)
          valid_exprs.insert(it->first, it->second);
      }
      if (!valid_exprs.empty())
      {
        // Sort into field sets, union, and then compare to the expression
        local::list<FieldSet<IndexSpaceExpression*>> field_sets;
        valid_exprs.compute_field_sets(allvalid_mask, field_sets);
        for (local::list<FieldSet<IndexSpaceExpression*>>::const_iterator it =
                 field_sets.begin();
             it != field_sets.end(); it++)
        {
          if (!it->elements.empty())
          {
            IndexSpaceExpression* valid_expr =
                (it->elements.size() == 1) ?
                    *(it->elements.begin()) :
                    runtime->union_index_spaces(it->elements);
            IndexSpaceExpression* overlap_expr =
                runtime->intersect_index_spaces(expr, valid_expr);
            if (overlap_expr->get_volume() < expr->get_volume())
              allvalid_mask -= it->set_mask;
          }
          else
            allvalid_mask -= it->set_mask;
        }
      }
      else
        allvalid_mask.clear();
    }

    //--------------------------------------------------------------------------
    void CollectiveAntiAlias::visit_leaf(
        const FieldMask& mask, FieldMask& dominated_mask, InnerContext* context,
        RegionTreeID tree_id, CollectiveView* view,
        local::map<LogicalView*, local::FieldMaskMap<IndexSpaceExpression>>&
            non_dominated,
        IndexSpaceExpression* expr)
    //--------------------------------------------------------------------------
    {
      if (!valid_exprs.empty() && !(mask * valid_exprs.get_valid_mask()))
      {
        legion_assert(collective != view);
        // We need to remove the entry for these fields since we'll be recording
        // new subfield entries along them
        dominated_mask |= mask;
        // Sort into field sets, union, and then compare to the expression
        local::list<FieldSet<IndexSpaceExpression*>> field_sets;
        valid_exprs.compute_field_sets(mask, field_sets);
        for (local::list<FieldSet<IndexSpaceExpression*>>::const_iterator it =
                 field_sets.begin();
             it != field_sets.end(); it++)
        {
          if (it->elements.empty())
          {
            // Record that all point-fields are non-dominated
            InstanceView* view = get_instance_view(context, tree_id);
            non_dominated[view].insert(expr, it->set_mask);
          }
          else
          {
            IndexSpaceExpression* valid_expr =
                (it->elements.size() == 1) ?
                    *(it->elements.begin()) :
                    runtime->union_index_spaces(it->elements);
            IndexSpaceExpression* diff_expr =
                runtime->subtract_index_spaces(expr, valid_expr);
            if (!diff_expr->is_empty())
            {
              // Dominated some of the points but not all of them
              InstanceView* view = get_instance_view(context, tree_id);
              non_dominated[view].insert(diff_expr, it->set_mask);
            }
          }
        }
      }
      // If we're still at the root (collective == view) and we don't have
      // any overlaps then we don't need to record it because it has already
      // been recorded and we don't need to update the dominated mask
      else if (collective != view)
      {
        // Record that all point-fields are non-dominated for these instances
        InstanceView* view = get_instance_view(context, tree_id);
        non_dominated[view].insert(expr, mask);
        // These fields have been dominated
        dominated_mask |= mask;
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveAntiAlias::visit_leaf(
        const FieldMask& mask, FieldMask& allvalid_mask, TraceViewSet& view_set,
        local::FieldMaskMap<InstanceView>& alt_views)
    //--------------------------------------------------------------------------
    {
      if (valid_exprs.empty())
        return;
      legion_assert(
          valid_exprs.size() == 1);  // should just have one null entry
      const FieldMask& local_mask = valid_exprs.get_valid_mask();
      allvalid_mask -= local_mask;
      InstanceView* view = view_set.find_instance_view(inst_dids);
      alt_views.insert(view, local_mask);
    }

    // Explicit instantiation for CollectiveRefinementTree
    template class CollectiveRefinementTree<CollectiveAntiAlias>;

    /////////////////////////////////////////////////////////////
    // InitializeCollectiveReduction
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InitializeCollectiveReduction::InitializeCollectiveReduction(
        AllreduceView* v, IndexSpaceExpression* expr)
      : CollectiveRefinementTree<InitializeCollectiveReduction>(v),
        needed_expr(expr), view(v)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    InitializeCollectiveReduction::InitializeCollectiveReduction(
        std::vector<DistributedID>&& insts, IndexSpaceExpression* expr,
        InstanceView* v,
        const local::FieldMaskMap<IndexSpaceExpression>& remainders,
        const FieldMask& covered)
      : CollectiveRefinementTree<InitializeCollectiveReduction>(
            std::move(insts)),
        needed_expr(expr), view(v), remainder_exprs(remainders),
        found_covered(covered)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    InitializeCollectiveReduction* InitializeCollectiveReduction::clone(
        InstanceView* new_view, const FieldMask& mask,
        std::vector<DistributedID>&& insts) const
    //--------------------------------------------------------------------------
    {
      return new InitializeCollectiveReduction(
          std::move(insts), needed_expr, new_view, remainder_exprs,
          found_covered);
    }

    //--------------------------------------------------------------------------
    void InitializeCollectiveReduction::analyze(
        InstanceView* view, const FieldMask& mask, IndexSpaceExpression* expr)
    //--------------------------------------------------------------------------
    {
      FieldMask remainder_mask = mask - found_covered;
      if (!!remainder_mask)
      {
        local::FieldMaskMap<IndexSpaceExpression> to_add;
        std::vector<IndexSpaceExpression*> to_delete;
        for (local::FieldMaskMap<IndexSpaceExpression>::iterator it =
                 remainder_exprs.begin();
             it != remainder_exprs.end(); it++)
        {
          const FieldMask overlap = remainder_mask & it->second;
          if (!overlap)
            continue;
          if (expr != it->first)
          {
            IndexSpaceExpression* overlap_expr =
                runtime->intersect_index_spaces(it->first, expr);
            const size_t overlap_volume = overlap_expr->get_volume();
            if (overlap_volume < it->first->get_volume())
            {
              if (overlap_volume > 0)
              {
                IndexSpaceExpression* diff_expr =
                    runtime->subtract_index_spaces(it->first, expr);
                to_add.insert(diff_expr, overlap);
                it.filter(overlap);
                if (!it->second)
                  to_delete.emplace_back(it->first);
              }
            }
            else
            {
              found_covered |= overlap;
              it.filter(overlap);
              if (!it->second)
                to_delete.emplace_back(it->first);
            }
          }
          else
          {
            found_covered |= overlap;
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(it->first);
          }
          remainder_mask -= overlap;
          if (!remainder_mask)
            break;
        }
        if (!to_delete.empty())
          for (IndexSpaceExpression* expr : to_delete)
            remainder_exprs.erase(expr);
        if (!to_add.empty())
        {
          if (!remainder_exprs.empty())
          {
            for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                     to_add.begin();
                 it != to_add.end(); it++)
              remainder_exprs.insert(it->first, it->second);
          }
          else
            remainder_exprs.swap(to_add);
        }
        if (!!remainder_mask)
        {
          if (expr != needed_expr)
          {
            IndexSpaceExpression* overlap_expr =
                runtime->intersect_index_spaces(needed_expr, expr);
            const size_t overlap_volume = overlap_expr->get_volume();
            if (overlap_volume < needed_expr->get_volume())
            {
              if (overlap_volume > 0)
              {
                IndexSpaceExpression* diff_expr =
                    runtime->subtract_index_spaces(needed_expr, expr);
                remainder_exprs.insert(diff_expr, remainder_mask);
              }
            }
            else
              found_covered |= mask;
          }
          else
            found_covered |= mask;
        }
      }
    }

    //--------------------------------------------------------------------------
    void InitializeCollectiveReduction::analyze(
        InstanceView* view, const FieldMask& mask,
        const FieldMapView<IndexSpaceExpression>& exprs)
    //--------------------------------------------------------------------------
    {
      // This method should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void InitializeCollectiveReduction::visit_leaf(
        const FieldMask& mask, IndexSpaceExpression* expr, bool& failure)
    //--------------------------------------------------------------------------
    {
      if (mask * found_covered)
      {
        if (!(mask * remainder_exprs.get_valid_mask()))
        {
          for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   remainder_exprs.begin();
               it != remainder_exprs.end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            if (expr != it->first)
            {
              IndexSpaceExpression* expr_overlap =
                  runtime->intersect_index_spaces(expr, it->first);
              // If it's not completely contained within the bounds of
              // un-filled remainder then that is a failure of the ABA
              if (expr_overlap->get_volume() < expr->get_volume())
                failure = true;
            }
          }
        }
      }
      else
        failure = true;
    }

    //--------------------------------------------------------------------------
    void InitializeCollectiveReduction::visit_leaf(
        const FieldMask& mask, InnerContext* context, UpdateAnalysis& analysis,
        CopyFillAggregator*& fill_aggregator, FillView* fill_view,
        RegionTreeID tid, EquivalenceSet* eq_set, DistributedID did,
        std::map<
            unsigned,
            std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>&
            reduction_instances)
    //--------------------------------------------------------------------------
    {
      FieldMask uncovered = mask - found_covered;
      if (!!uncovered)
      {
        InstanceView* local_view = view;
        if (local_view == nullptr)
          local_view = get_instance_view(context, tid);
        if (fill_aggregator == nullptr)
        {
          // Fill aggregators never need to wait for any other
          // aggregators since we know they won't depend on each other
          fill_aggregator = new CopyFillAggregator(
              &analysis, nullptr /*no previous guard*/, false /*track events*/);
          analysis.input_aggregators[RtEvent::NO_RT_EVENT] = fill_aggregator;
        }
        // Issue fills for any remainders
        if (!remainder_exprs.empty())
        {
          for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   remainder_exprs.begin();
               it != remainder_exprs.end(); it++)
          {
            fill_aggregator->record_fill(
                local_view, fill_view, it->second, it->first,
                PredEvent::NO_PRED_EVENT,
                analysis.trace_info.recording ? eq_set : nullptr);
            // Record any reduction instances here as well
            int fidx = it->second.find_first_set();
            while (fidx >= 0)
            {
              local_view->add_nested_valid_ref(did);
              it->first->add_nested_expression_reference(did);
              reduction_instances[fidx].emplace_back(
                  std::make_pair(local_view, it->first));
              fidx = it->second.find_next_set(fidx + 1);
            }
          }
          uncovered -= remainder_exprs.get_valid_mask();
        }
        if (!!uncovered)
        {
          // Issue a fill for the full needed_expr for these fields
          fill_aggregator->record_fill(
              local_view, fill_view, uncovered, needed_expr,
              PredEvent::NO_PRED_EVENT,
              analysis.trace_info.recording ? eq_set : nullptr);
          // Record any reduction instances here as well
          int fidx = uncovered.find_first_set();
          while (fidx >= 0)
          {
            local_view->add_nested_valid_ref(did);
            needed_expr->add_nested_expression_reference(did);
            reduction_instances[fidx].emplace_back(
                std::make_pair(local_view, needed_expr));
            fidx = uncovered.find_next_set(fidx + 1);
          }
        }
      }
    }

    // Explicit instantiation for CollectiveRefinementTree
    template class CollectiveRefinementTree<InitializeCollectiveReduction>;

    /////////////////////////////////////////////////////////////
    // Equivalence Set
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    EquivalenceSet::EquivalenceSet(
        DistributedID id, AddressSpaceID logical, IndexSpaceExpression* expr,
        RegionTreeID tid, InnerContext* ctx, bool reg_now,
        CollectiveMapping* mapping /*= nullptr*/,
        bool replicate_logical_owner /* = false*/)
      : DistributedCollectable(
            LEGION_DISTRIBUTED_HELP_ENCODE(id, EQUIVALENCE_SET_DC), reg_now,
            mapping),
        context(ctx), set_expr(expr), tree_id(tid),
        tracing_preconditions(nullptr), tracing_anticonditions(nullptr),
        tracing_postconditions(nullptr), tracing_dirty_fields(nullptr),
        logical_owner_space(logical), replicated_owner_state(nullptr),
        migration_index(0), sample_count(0)
    //--------------------------------------------------------------------------
    {
      context->add_nested_resource_ref(did);
      context->add_nested_gc_ref(did);
      set_expr->add_nested_expression_reference(did);
      next_deferral_precondition.store(0);
      if (replicate_logical_owner)
      {
        // If we've been told to replicate the logical owner space knowledge
        // than we do that now, note that this will preclude migration of
        // this equivalence set for its entire lifetime
        legion_assert(mapping != nullptr);
        legion_assert(mapping->contains(logical_owner_space));
        legion_assert(mapping->contains(local_space));
        replicated_owner_state = new ReplicatedOwnerState(true /*valid*/);
        mapping->get_children(
            logical_owner_space, local_space, replicated_owner_state->children);
      }
#ifdef LEGION_GC
      log_garbage.info(
          "GC Equivalence Set %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
      if (is_logical_owner())
        LegionSpy::log_equivalence_set(did, expr->expr_id, tree_id);
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::~EquivalenceSet(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(total_valid_instances.empty());
      legion_assert(partial_valid_instances.empty());
      legion_assert(!partial_valid_fields);
      legion_assert(initialized_data.empty());
      legion_assert(partial_invalidations.empty());
      legion_assert(reduction_instances.empty());
      legion_assert(!reduction_fields);
      legion_assert(restricted_instances.empty());
      legion_assert(!restricted_fields);
      legion_assert(released_instances.empty());
      legion_assert(collective_instances.empty());
      legion_assert(tracing_preconditions == nullptr);
      legion_assert(tracing_anticonditions == nullptr);
      legion_assert(tracing_postconditions == nullptr);
      legion_assert(tracing_dirty_fields == nullptr);
      if (replicated_owner_state != nullptr)
        delete replicated_owner_state;
      if (context->remove_nested_resource_ref(did))
        delete context;
      if (set_expr->remove_nested_expression_reference(did))
        delete set_expr;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::notify_local(void)
    //--------------------------------------------------------------------------
    {
      if (!total_valid_instances.empty())
      {
        for (shrt::FieldMaskMap<LogicalView>::const_iterator it =
                 total_valid_instances.begin();
             it != total_valid_instances.end(); it++)
          if (it->first->remove_nested_valid_ref(did))
            delete it->first;
        total_valid_instances.clear();
      }
      if (!partial_valid_instances.empty())
      {
        for (ViewExprMaskSets::iterator pit = partial_valid_instances.begin();
             pit != partial_valid_instances.end(); pit++)
        {
          for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   pit->second.begin();
               it != pit->second.end(); it++)
            if (it->first->remove_nested_expression_reference(did))
              delete it->first;
          if (pit->first->remove_nested_valid_ref(did))
            delete pit->first;
          pit->second.clear();
        }
        partial_valid_instances.clear();
        partial_valid_fields.clear();
      }
      if (!initialized_data.empty())
      {
        for (lng::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                 initialized_data.begin();
             it != initialized_data.end(); it++)
          if (it->first->remove_nested_expression_reference(did))
            delete it->first;
        initialized_data.clear();
      }
      if (!partial_invalidations.empty())
      {
        for (lng::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                 partial_invalidations.begin();
             it != partial_invalidations.end(); it++)
          if (it->first->remove_nested_expression_reference(did))
            delete it->first;
        partial_invalidations.clear();
      }
      if (!reduction_instances.empty())
      {
        for (std::pair<
                 const unsigned,
                 std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>&
                 it : reduction_instances)
        {
          while (!it.second.empty())
          {
            std::pair<InstanceView*, IndexSpaceExpression*>& back =
                it.second.back();
            if (back.first->remove_nested_valid_ref(did))
              delete back.first;
            if (back.second->remove_nested_expression_reference(did))
              delete back.second;
            it.second.pop_back();
          }
        }
        reduction_instances.clear();
        reduction_fields.clear();
      }
      if (!restricted_instances.empty())
      {
        for (ExprViewMaskSets::iterator rit = restricted_instances.begin();
             rit != restricted_instances.end(); rit++)
        {
          for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                   rit->second.begin();
               it != rit->second.end(); it++)
            if (it->first->remove_nested_valid_ref(did))
              delete it->first;
          if (rit->first->remove_nested_expression_reference(did))
            delete rit->first;
          rit->second.clear();
        }
        restricted_instances.clear();
        restricted_fields.clear();
      }
      if (!released_instances.empty())
      {
        for (ExprViewMaskSets::iterator rit = released_instances.begin();
             rit != released_instances.end(); rit++)
        {
          for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                   rit->second.begin();
               it != rit->second.end(); it++)
            if (it->first->remove_nested_valid_ref(did))
              delete it->first;
          if (rit->first->remove_nested_expression_reference(did))
            delete rit->first;
          rit->second.clear();
        }
        released_instances.clear();
      }
      if (!collective_instances.empty())
      {
        for (lng::FieldMaskMap<CollectiveView>::const_iterator it =
                 collective_instances.begin();
             it != collective_instances.end(); it++)
          if (it->first->remove_nested_resource_ref(did))
            delete it->first;
        collective_instances.clear();
      }
      if (tracing_preconditions != nullptr)
      {
        delete tracing_preconditions;
        tracing_preconditions = nullptr;
      }
      if (tracing_anticonditions != nullptr)
      {
        delete tracing_anticonditions;
        tracing_anticonditions = nullptr;
      }
      if (tracing_postconditions != nullptr)
      {
        delete tracing_postconditions;
        tracing_postconditions = nullptr;
      }
      if (tracing_dirty_fields != nullptr)
      {
        for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                 tracing_dirty_fields->begin();
             it != tracing_dirty_fields->end(); it++)
          if (it->first->remove_nested_expression_reference(did))
            delete it->first;
        delete tracing_dirty_fields;
        tracing_dirty_fields = nullptr;
      }
      // No need to check for deletion since we're still holding a
      // resource reference to the context as well
      context->remove_nested_gc_ref(did);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::initialize_collective_references(unsigned local_valid)
    //--------------------------------------------------------------------------
    {
      legion_assert(collective_mapping != nullptr);
      if (is_owner())
      {
        if (local_valid > 0)
          add_base_gc_ref(CONTEXT_REF, local_valid);
      }
      else
      {
        legion_assert(local_valid > 0);
        add_base_gc_ref(CONTEXT_REF, local_valid);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::initialize_set(
        const RegionUsage& usage, const FieldMask& user_mask,
        const bool restricted, const InstanceSet& sources,
        const std::vector<IndividualView*>& corresponding)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_logical_owner() || set_expr->is_empty());
      legion_assert(sources.size() == corresponding.size());
      AutoLock eq(eq_lock);
      if (IS_REDUCE(usage))
      {
        // Reduction-only should always be restricted for now
        // Could change if we started issuing reduction close
        // operations at the end of a context
        legion_assert(restricted);
        // Since these are restricted, we'll make these the actual
        // target logical instances and record them as restricted
        // instead of recording them as reduction instances
        for (unsigned idx = 0; idx < sources.size(); idx++)
        {
          const FieldMask& view_mask = sources[idx].get_valid_fields();
          IndividualView* view = corresponding[idx];
          shrt::FieldMaskMap<LogicalView>::iterator finder =
              total_valid_instances.find(view);
          if (finder == total_valid_instances.end())
          {
            total_valid_instances.insert(view, view_mask);
            view->add_nested_valid_ref(did);
          }
          else
            finder.merge(view_mask);
          // Always restrict reduction-only users since we know the data
          // is going to need to be flushed anyway
          record_restriction(set_expr, true /*covers*/, view_mask, view);
        }
      }
      else
      {
        for (unsigned idx = 0; idx < sources.size(); idx++)
        {
          const FieldMask& view_mask = sources[idx].get_valid_fields();
          IndividualView* view = corresponding[idx];
          legion_assert(!view->is_reduction_kind());
          shrt::FieldMaskMap<LogicalView>::iterator finder =
              total_valid_instances.find(view);
          if (finder == total_valid_instances.end())
          {
            total_valid_instances.insert(view, view_mask);
            view->add_nested_valid_ref(did);
          }
          else
            finder.merge(view_mask);
          // Check if this is a collective view we need to track
          if (view->is_collective_view() &&
              collective_instances.insert(
                  view->as_collective_view(), view_mask))
            view->add_nested_resource_ref(did);
          // If this is restricted then record it
          if (restricted)
            record_restriction(set_expr, true /*covers*/, view_mask, view);
        }
      }
      // Record that this data is all valid
      update_initialized_data(set_expr, true /*covers*/, user_mask);
      // Update any restricted fields
      if (restricted)
      {
        legion_assert(!restricted_instances.empty());
        restricted_fields |= user_mask;
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::analyze(
        PhysicalAnalysis& analysis, IndexSpaceExpression* expr,
        const bool expr_covers, FieldMask traversal_mask,
        std::set<RtEvent>& deferral_events, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // This block of code is the same across all traversals so we ensure
      // that it is done before traversing the equivalence set for all analyses
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_analysis(
            eq, analysis, traversal_mask, deferral_events, applied_events,
            already_deferred);
        return;
      }
      if (is_remote_analysis(
              analysis, traversal_mask, deferral_events, applied_events,
              expr_covers))
        return;
      // Should only be here if we're the owner
      legion_assert(is_logical_owner());
      bool check_migration = false;
      if (!partial_invalidations.empty())
      {
        // Check for any fields which have partial invalid expressions that we
        // also need to remove from from this analysis
        FieldMask invalid_mask =
            traversal_mask & partial_invalidations.get_valid_mask();
        if (!!invalid_mask)
        {
          traversal_mask -= invalid_mask;
          for (lng::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   partial_invalidations.begin();
               it != partial_invalidations.end(); it++)
          {
            const FieldMask overlap = invalid_mask & it->second;
            if (!overlap)
              continue;
            // Should never be trying to do a traversal on an equivalence
            // set that has been completely invalidated
            legion_assert(it->first != set_expr);
            IndexSpaceExpression* remainder =
                runtime->subtract_index_spaces(expr, it->first);
            if (!remainder->is_empty() &&
                analysis.perform_analysis(
                    this, remainder, false /*expr covers*/, overlap,
                    applied_events, already_deferred))
              check_migration = true;
            invalid_mask -= overlap;
            if (!invalid_mask)
              break;
          }
        }
      }
      if (!!traversal_mask && analysis.perform_analysis(
                                  this, expr, expr_covers, traversal_mask,
                                  applied_events, already_deferred))
        check_migration = true;
      if (check_migration)
        check_for_migration(analysis, applied_events);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::find_valid_instances(
        ValidInstAnalysis& analysis, IndexSpaceExpression* expr,
        const bool expr_covers, const FieldMask& user_mask,
        std::set<RtEvent>& applied_events, const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // Already holding the eq_lock from EquivalenceSet::analyze method
      // Lock the analysis so we can perform updates here
      AutoLock a_lock(analysis);
      if (analysis.redop != 0)
      {
        // Iterate over all the fields
        int fidx = user_mask.find_first_set();
        while (fidx >= 0)
        {
          std::map<
              unsigned,
              std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>::
              iterator current = reduction_instances.find(fidx);
          if (current != reduction_instances.end())
          {
            FieldMask local_mask;
            local_mask.set_bit(fidx);
            for (std::list<std::pair<InstanceView*, IndexSpaceExpression*>>::
                     const_reverse_iterator it = current->second.rbegin();
                 it != current->second.rend(); it++)
            {
              if (it->first->get_redop() != analysis.redop)
                break;
              if (!expr_covers)
              {
                IndexSpaceExpression* overlap =
                    runtime->intersect_index_spaces(expr, it->second);
                if (overlap->is_empty())
                  continue;
              }
              analysis.record_instance(it->first, local_mask);
            }
          }
          fidx = user_mask.find_next_set(fidx + 1);
        }
      }
      else
      {
        if (!(user_mask * total_valid_instances.get_valid_mask()))
        {
          for (shrt::FieldMaskMap<LogicalView>::const_iterator it =
                   total_valid_instances.begin();
               it != total_valid_instances.end(); it++)
          {
            if (!it->first->is_instance_view())
              continue;
            const FieldMask overlap = it->second & user_mask;
            if (!overlap)
              continue;
            analysis.record_instance(it->first->as_instance_view(), overlap);
          }
        }
        if (!(user_mask * partial_valid_fields))
        {
          for (ViewExprMaskSets::const_iterator pit =
                   partial_valid_instances.begin();
               pit != partial_valid_instances.end(); pit++)
          {
            if (!pit->first->is_instance_view())
              continue;
            if (expr_covers)
            {
              const FieldMask overlap =
                  user_mask & pit->second.get_valid_mask();
              if (!!overlap)
                analysis.record_instance(
                    pit->first->as_instance_view(), overlap);
              continue;
            }
            else if (user_mask * pit->second.get_valid_mask())
              continue;
            FieldMask total_overlap;
            for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                     pit->second.begin();
                 it != pit->second.end(); it++)
            {
              const FieldMask overlap = user_mask & it->second;
              if (!overlap)
                continue;
              IndexSpaceExpression* expr_overlap =
                  runtime->intersect_index_spaces(expr, it->first);
              if (expr_overlap->is_empty())
                continue;
              total_overlap |= overlap;
            }
            if (!!total_overlap)
              analysis.record_instance(
                  pit->first->as_instance_view(), total_overlap);
          }
        }
      }
      if (!(user_mask * restricted_fields))
      {
        if (!expr_covers)
        {
          // Check for the set expr first which we know overlaps
          ExprViewMaskSets::const_iterator finder =
              restricted_instances.find(set_expr);
          if ((finder == restricted_instances.end()) ||
              (finder->second.get_valid_mask() * user_mask))
          {
            for (ExprViewMaskSets::const_iterator it =
                     restricted_instances.begin();
                 it != restricted_instances.end(); it++)
            {
              if (it == finder)
                continue;
              if (it->second.get_valid_mask() * user_mask)
                continue;
              IndexSpaceExpression* overlap =
                  runtime->intersect_index_spaces(it->first, expr);
              if (overlap->is_empty())
                continue;
              analysis.record_restriction();
              break;
            }
          }
          else
            analysis.record_restriction();
        }
        else
          analysis.record_restriction();
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::find_invalid_instances(
        InvalidInstAnalysis& analysis, IndexSpaceExpression* expr,
        const bool expr_covers, const FieldMask& user_mask,
        std::set<RtEvent>& applied_events, const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // Already holding the eq_lock from EquivalenceSet::analyze method
      for (op::FieldMaskMap<LogicalView>::const_iterator vit =
               analysis.valid_instances.begin();
           vit != analysis.valid_instances.end(); vit++)
      {
        FieldMask invalid_mask = vit->second & user_mask;
        if (!invalid_mask)
          continue;
        if (vit->first->is_deferred_view())
        {
          // Should only have fill deferred views here
          // No need to worry about collective aliasing in this case
          legion_assert(vit->first->is_fill_view());
          FillView* fill = vit->first->as_fill_view();
          // Check the total valid instances first
          if (!total_valid_instances.empty())
          {
            // Check names for the easy case
            shrt::FieldMaskMap<LogicalView>::const_iterator finder =
                total_valid_instances.find(vit->first);
            if (finder != total_valid_instances.end())
            {
              invalid_mask -= finder->second;
              if (!invalid_mask)
                continue;
            }
            // Check to see if we have another fill view that matches
            for (shrt::FieldMaskMap<LogicalView>::const_iterator it =
                     total_valid_instances.begin();
                 it != total_valid_instances.end(); it++)
            {
              if (!it->first->is_fill_view())
                continue;
              FillView* view = it->first->as_fill_view();
              if (fill->matches(view))
              {
                invalid_mask -= it->second;
                if (!invalid_mask)
                  break;
              }
            }
            if (!invalid_mask)
              continue;
          }
          if (!partial_valid_instances.empty() &&
              !(invalid_mask * partial_valid_fields))
          {
            local::FieldMaskMap<IndexSpaceExpression> partial_valid_exprs;
            ViewExprMaskSets::const_iterator finder =
                partial_valid_instances.find(vit->first);
            if ((finder != partial_valid_instances.end()) &&
                !(finder->second.get_valid_mask() * invalid_mask))
            {
              shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator
                  expr_finder = finder->second.find(expr);
              if (expr_finder != finder->second.end())
              {
                invalid_mask -= expr_finder->second;
                if (!invalid_mask)
                  continue;
              }
              for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                       finder->second.begin();
                   it != finder->second.end(); it++)
              {
                if (it->first == expr)
                  continue;
                const FieldMask overlap = it->second & invalid_mask;
                if (!overlap)
                  continue;
                IndexSpaceExpression* expr_overlap =
                    runtime->intersect_index_spaces(expr, it->first);
                const size_t overlap_volume = expr_overlap->get_volume();
                if (overlap_volume == expr->get_volume())
                {
                  invalid_mask -= overlap;
                  if (!invalid_mask)
                    break;
                }
                // Record any partial valid expressions
                else if (overlap_volume > 0)
                  partial_valid_exprs.insert(expr_overlap, overlap);
              }
              if (!invalid_mask)
                continue;
              // Also check for matching logical views with expressions
              for (ViewExprMaskSets::const_iterator pit =
                       partial_valid_instances.begin();
                   pit != partial_valid_instances.end(); pit++)
              {
                if (!pit->first->is_fill_view())
                  continue;
                if (invalid_mask * pit->second.get_valid_mask())
                  continue;
                if (!fill->matches(pit->first->as_fill_view()))
                  continue;
                shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator
                    expr_finder = finder->second.find(expr);
                if (expr_finder != finder->second.end())
                {
                  invalid_mask -= expr_finder->second;
                  if (!invalid_mask)
                    break;
                }
                for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator
                         it = finder->second.begin();
                     it != finder->second.end(); it++)
                {
                  if (it->first == expr)
                    continue;
                  const FieldMask overlap = it->second & invalid_mask;
                  if (!overlap)
                    continue;
                  IndexSpaceExpression* expr_overlap =
                      runtime->intersect_index_spaces(expr, it->first);
                  const size_t overlap_volume = expr_overlap->get_volume();
                  if (overlap_volume == expr->get_volume())
                  {
                    invalid_mask -= overlap;
                    if (!invalid_mask)
                      break;
                  }
                  // Record any partial valid expressions
                  else if (overlap_volume > 0)
                    partial_valid_exprs.insert(expr_overlap, overlap);
                }
              }
              if (!invalid_mask)
                continue;
              if (partial_valid_exprs.size() > 1)
              {
                local::list<FieldSet<IndexSpaceExpression*>> field_sets;
                partial_valid_exprs.compute_field_sets(FieldMask(), field_sets);
                for (local::list<
                         FieldSet<IndexSpaceExpression*>>::const_iterator it =
                         field_sets.begin();
                     it != field_sets.end(); it++)
                {
                  // If we don't have at least two sets to union together
                  // then we know the expression is already not big enough
                  // to cover the needed expression
                  if (it->elements.size() < 2)
                    continue;
                  IndexSpaceExpression* union_expr =
                      runtime->intersect_index_spaces(it->elements);
                  IndexSpaceExpression* expr_overlap =
                      runtime->intersect_index_spaces(expr, union_expr);
                  const size_t overlap_volume = expr_overlap->get_volume();
                  if (overlap_volume == expr->get_volume())
                  {
                    invalid_mask -= it->set_mask;
                    if (!invalid_mask)
                      break;
                  }
                }
                if (!invalid_mask)
                  continue;
              }
            }
          }
        }
        else if (vit->first->is_reduction_kind())
        {
          // Reduction instance path
          InstanceView* reduction_view = vit->first->as_instance_view();
          if (reduction_view->is_collective_view())
          {
            // Collective reduction view, so need to check that all
            // intances in the collective reduction view are covered
            if (!(invalid_mask - reduction_fields))
            {
              CollectiveAntiAlias alias_analysis(
                  reduction_view->as_collective_view());
              int fidx = invalid_mask.find_first_set();
              while (fidx >= 0)
              {
                std::map<
                    unsigned,
                    std::list<std::pair<
                        InstanceView*, IndexSpaceExpression*>>>::const_iterator
                    finder = reduction_instances.find(fidx);
                if (finder == reduction_instances.end())
                  break;
                FieldMask reduction_mask;
                reduction_mask.set_bit(fidx);
                for (std::list<
                         std::pair<InstanceView*, IndexSpaceExpression*>>::
                         const_reverse_iterator rit = finder->second.rbegin();
                     rit != finder->second.rend(); rit++)
                {
                  // Can't go backwards through any
                  // reductions of a different type
                  if (rit->first->get_redop() != reduction_view->get_redop())
                    break;
                  if (!rit->first->aliases(reduction_view))
                    continue;
                  alias_analysis.traverse(
                      rit->first, reduction_mask, rit->second);
                }
                fidx = invalid_mask.find_next_set(fidx + 1);
              }
              FieldMask allvalid_mask = invalid_mask;
              local::FieldMaskMap<IndexSpaceExpression> no_partial_valid_exprs;
              alias_analysis.visit_leaves(
                  invalid_mask, allvalid_mask, expr, no_partial_valid_exprs);
              if (!!allvalid_mask)
              {
                invalid_mask -= allvalid_mask;
                if (!invalid_mask)
                  continue;
              }
            }
          }
          else
          {
            // Individual instance view so traverse like normal, but be
            // sure to check for aliasing with any collective views
            if (!(invalid_mask - reduction_fields))
            {
              int fidx = invalid_mask.find_first_set();
              while (fidx >= 0)
              {
                std::map<
                    unsigned,
                    std::list<std::pair<
                        InstanceView*, IndexSpaceExpression*>>>::const_iterator
                    finder = reduction_instances.find(fidx);
                if (finder == reduction_instances.end())
                  break;
                std::set<IndexSpaceExpression*> exprs;
                for (std::list<
                         std::pair<InstanceView*, IndexSpaceExpression*>>::
                         const_reverse_iterator rit = finder->second.rbegin();
                     rit != finder->second.rend(); rit++)
                {
                  // Can't go backwards through any
                  // reductions of a different type
                  if (rit->first->get_redop() != reduction_view->get_redop())
                    break;
                  if (!rit->first->aliases(reduction_view))
                    continue;
                  if ((rit->second == expr) || (rit->second == set_expr))
                  {
                    // covers everything
                    invalid_mask.unset_bit(fidx);
                    exprs.clear();
                    break;
                  }
                  else
                    exprs.insert(rit->second);
                }
                if (!exprs.empty())
                {
                  // See if they cover
                  IndexSpaceExpression* union_expr =
                      runtime->union_index_spaces(exprs);
                  IndexSpaceExpression* intersection =
                      runtime->intersect_index_spaces(expr, union_expr);
                  if (intersection->get_volume() == expr->get_volume())
                    invalid_mask.unset_bit(fidx);
                  else  // no point in checking the rest
                    break;
                }
                // No point in checking the rest if this wasn't covered
                else if (invalid_mask.is_set(fidx))
                  break;
                fidx = invalid_mask.find_next_set(fidx + 1);
              }
            }
            if (!invalid_mask)
              continue;
          }
        }
        else
        {
          // Normal instance path
          // Always perform a simple name check since that is easy
          if (!total_valid_instances.empty())
          {
            shrt::FieldMaskMap<LogicalView>::const_iterator finder =
                total_valid_instances.find(vit->first);
            if (finder != total_valid_instances.end())
            {
              invalid_mask -= finder->second;
              if (!invalid_mask)
                continue;
            }
          }
          if (vit->first->is_collective_view())
          {
            // Collective view case
            // Next check the partially valid views and see if they
            // cover expression
            local::FieldMaskMap<IndexSpaceExpression> partial_valid_exprs;
            ViewExprMaskSets::const_iterator finder =
                partial_valid_instances.find(vit->first);
            if ((finder != partial_valid_instances.end()) &&
                !(finder->second.get_valid_mask() * invalid_mask))
            {
              shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator
                  expr_finder = finder->second.find(expr);
              if (expr_finder != finder->second.end())
              {
                invalid_mask -= expr_finder->second;
                if (!invalid_mask)
                  continue;
              }
              for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                       finder->second.begin();
                   it != finder->second.end(); it++)
              {
                if (it->first == expr)
                  continue;
                const FieldMask overlap = it->second & invalid_mask;
                if (!overlap)
                  continue;
                IndexSpaceExpression* expr_overlap =
                    runtime->intersect_index_spaces(expr, it->first);
                const size_t overlap_volume = expr_overlap->get_volume();
                if (overlap_volume == expr->get_volume())
                {
                  invalid_mask -= overlap;
                  if (!invalid_mask)
                    break;
                }
                // Record any partial valid expressions
                else if (overlap_volume > 0)
                  partial_valid_exprs.insert(expr_overlap, overlap);
              }
              if (!invalid_mask)
                continue;
            }
            // If we make it here then still have invalid fields so we
            // need do an alias analysis with all the valid views on
            // their instances to see if they cover what we're looking for
            CollectiveView* view = vit->first->as_collective_view();
            CollectiveAntiAlias alias_analysis(view);
            alias_analysis.traverse_total(
                invalid_mask, set_expr, total_valid_instances);
            if (!(invalid_mask * partial_valid_fields))
              alias_analysis.traverse_partial(
                  invalid_mask, partial_valid_instances);
            FieldMask allvalid_mask = invalid_mask;
            alias_analysis.visit_leaves(
                invalid_mask, allvalid_mask, expr, partial_valid_exprs);
            if (!!allvalid_mask)
            {
              invalid_mask -= allvalid_mask;
              if (!invalid_mask)
                continue;
            }
          }
          else
          {
            // Non-collective view
            // See if there are any collective aliases
            local::FieldMaskMap<IndexSpaceExpression> partial_valid_exprs;
            if (vit->first->is_individual_view() &&
                !collective_instances.empty() &&
                !(collective_instances.get_valid_mask() * invalid_mask))
            {
              // Check for aliasing with any collective views
              IndividualView* view = vit->first->as_individual_view();
              // Scan through all the collective views and see which ones
              // we alias with and what they are valid for
              for (lng::FieldMaskMap<CollectiveView>::const_iterator cit =
                       collective_instances.begin();
                   cit != collective_instances.end(); cit++)
              {
                if (invalid_mask * cit->second)
                  continue;
                if (!view->aliases(cit->first))
                  continue;
                // Alias on fields and instances, check expressions
                shrt::FieldMaskMap<LogicalView>::const_iterator total_finder =
                    total_valid_instances.find(cit->first);
                if (total_finder != total_valid_instances.end())
                {
                  invalid_mask -= total_finder->second;
                  if (!invalid_mask)
                    break;
                }
                // Find all the covering and partial valid expressions
                ViewExprMaskSets::const_iterator finder =
                    partial_valid_instances.find(cit->first);
                if ((finder != partial_valid_instances.end()) &&
                    !(finder->second.get_valid_mask() * invalid_mask))
                {
                  shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator
                      expr_finder = finder->second.find(expr);
                  if (expr_finder != finder->second.end())
                  {
                    invalid_mask -= expr_finder->second;
                    if (!invalid_mask)
                      break;
                  }
                  for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator
                           it = finder->second.begin();
                       it != finder->second.end(); it++)
                  {
                    if (it->first == expr)
                      continue;
                    const FieldMask overlap = it->second & invalid_mask;
                    if (!overlap)
                      continue;
                    IndexSpaceExpression* expr_overlap =
                        runtime->intersect_index_spaces(expr, it->first);
                    const size_t overlap_volume = expr_overlap->get_volume();
                    if (overlap_volume == expr->get_volume())
                    {
                      invalid_mask -= overlap;
                      if (!invalid_mask)
                        break;
                    }
                    // Need to save partially valid expressions so we can
                    // union them together later to see if they all cover
                    else if (overlap_volume > 0)
                      partial_valid_exprs.insert(expr_overlap, overlap);
                  }
                  if (!invalid_mask)
                    break;
                }
              }
              if (!invalid_mask)
                continue;
            }
            if (!expr_covers || !partial_valid_exprs.empty())
            {
              // No collective instance aliasing so we can just do a name check
              ViewExprMaskSets::const_iterator finder =
                  partial_valid_instances.find(vit->first);
              if ((finder != partial_valid_instances.end()) &&
                  !(finder->second.get_valid_mask() * invalid_mask))
              {
                shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator
                    expr_finder = finder->second.find(expr);
                if (expr_finder != finder->second.end())
                {
                  invalid_mask -= expr_finder->second;
                  if (!invalid_mask)
                    continue;
                }
                for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator
                         it = finder->second.begin();
                     it != finder->second.end(); it++)
                {
                  if (it->first == expr)
                    continue;
                  const FieldMask overlap = it->second & invalid_mask;
                  if (!overlap)
                    continue;
                  IndexSpaceExpression* expr_overlap =
                      runtime->intersect_index_spaces(expr, it->first);
                  const size_t overlap_volume = expr_overlap->get_volume();
                  if (overlap_volume == expr->get_volume())
                  {
                    invalid_mask -= overlap;
                    if (!invalid_mask)
                      break;
                  }
                  // Keep recording partial expressions if we have any
                  // If we didn't already have any then we know this is
                  // all there will be since there's no other way to
                  // have instance name aliasing
                  else if ((overlap_volume > 0) && !partial_valid_exprs.empty())
                    partial_valid_exprs.insert(expr_overlap, overlap);
                }
                if (!invalid_mask)
                  continue;
              }
              if (!partial_valid_exprs.empty())
              {
                // Last chance, see if the union of all the partial valid
                // expressions are enough to cover the instance
                local::list<FieldSet<IndexSpaceExpression*>> expr_field_sets;
                partial_valid_exprs.compute_field_sets(
                    invalid_mask, expr_field_sets);
                for (local::list<
                         FieldSet<IndexSpaceExpression*>>::const_iterator it =
                         expr_field_sets.begin();
                     it != expr_field_sets.end(); it++)
                {
                  if (it->elements.empty())
                    continue;
                  IndexSpaceExpression* union_expr =
                      (it->elements.size() == 1) ?
                          *(it->elements.begin()) :
                          runtime->union_index_spaces(it->elements);
                  IndexSpaceExpression* expr_overlap =
                      runtime->intersect_index_spaces(expr, union_expr);
                  if (expr_overlap->get_volume() == expr->get_volume())
                  {
                    invalid_mask -= it->set_mask;
                    if (!invalid_mask)
                      break;
                  }
                }
                if (!invalid_mask)
                  continue;
              }
            }
          }
        }
        // If we get here it's because we're not valid for some expression
        // for these fields so record it
        legion_assert(!!invalid_mask);
        // Lock the analysis when recording this update
        AutoLock a_lock(analysis);
        analysis.record_instance(vit->first, invalid_mask);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::find_antivalid_instances(
        AntivalidInstAnalysis& analysis, IndexSpaceExpression* expr,
        const bool expr_covers, const FieldMask& user_mask,
        std::set<RtEvent>& applied_events, const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // Already holding the eq_lock from EquivalenceSet::analyze method
      for (op::FieldMaskMap<LogicalView>::const_iterator ait =
               analysis.antivalid_instances.begin();
           ait != analysis.antivalid_instances.end(); ait++)
      {
        legion_assert(!ait->first->is_deferred_view());
        const FieldMask antivalid_mask = ait->second & user_mask;
        if (!antivalid_mask)
          continue;
        if (ait->first->is_reduction_kind())
        {
          // Reduction instance case
          InstanceView* reduction_view = ait->first->as_instance_view();
          int fidx = antivalid_mask.find_first_set();
          while (fidx >= 0)
          {
            std::map<
                unsigned,
                std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>::
                const_iterator finder = reduction_instances.find(fidx);
            if (finder != reduction_instances.end())
            {
              for (const std::pair<InstanceView*, IndexSpaceExpression*>& it :
                   finder->second)
              {
                if (!reduction_view->aliases(it.first))
                  continue;
                FieldMask local_mask;
                local_mask.set_bit(fidx);
                if (!expr_covers)
                {
                  if ((it.second != set_expr) && (it.second != expr))
                  {
                    IndexSpaceExpression* intersection =
                        runtime->intersect_index_spaces(expr, it.second);
                    if (!intersection->is_empty())
                    {
                      AutoLock a_lock(analysis);
                      analysis.record_instance(reduction_view, local_mask);
                    }
                  }
                  else
                  {
                    AutoLock a_lock(analysis);
                    analysis.record_instance(reduction_view, local_mask);
                  }
                }
                else  // they intersect so record it
                {
                  AutoLock a_lock(analysis);
                  analysis.record_instance(reduction_view, local_mask);
                }
              }
            }
            fidx = antivalid_mask.find_next_set(fidx + 1);
          }
        }
        else
        {
          // Normal instance case
          // Check for it in the total valid instances first
          shrt::FieldMaskMap<LogicalView>::const_iterator total_finder =
              total_valid_instances.find(ait->first);
          if (total_finder != total_valid_instances.end())
          {
            const FieldMask overlap = antivalid_mask & total_finder->second;
            if (!!overlap)
            {
              AutoLock a_lock(analysis);
              analysis.record_instance(ait->first, overlap);
            }
          }
          // Then check for it in the partial valid instances
          ViewExprMaskSets::const_iterator finder =
              partial_valid_instances.find(ait->first);
          if (finder != partial_valid_instances.end())
          {
            for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                     finder->second.begin();
                 it != finder->second.end(); it++)
            {
              const FieldMask overlap = it->second & antivalid_mask;
              if (!overlap)
                continue;
              if (!expr_covers)
              {
                if ((it->first != set_expr) && (it->first != expr))
                {
                  IndexSpaceExpression* intersection =
                      runtime->intersect_index_spaces(expr, it->first);
                  if (!intersection->is_empty())
                  {
                    AutoLock a_lock(analysis);
                    analysis.record_instance(ait->first, overlap);
                  }
                }
                else
                {
                  AutoLock a_lock(analysis);
                  analysis.record_instance(ait->first, overlap);
                }
              }
              else
              {
                AutoLock a_lock(analysis);
                analysis.record_instance(ait->first, overlap);
              }
            }
          }
          // Lastly do the aliasing checks for collective instances
          if (ait->first->is_collective_view())
          {
            CollectiveView* collective = ait->first->as_collective_view();
            // Check for aliasing with any of the valid views
            if (!(antivalid_mask * total_valid_instances.get_valid_mask()))
            {
              for (shrt::FieldMaskMap<LogicalView>::const_iterator it =
                       total_valid_instances.begin();
                   it != total_valid_instances.end(); it++)
              {
                if (!it->first->is_instance_view())
                  continue;
                const FieldMask overlap = antivalid_mask & it->second;
                if (!overlap)
                  continue;
                if (!collective->aliases(it->first->as_instance_view()))
                  continue;
                AutoLock a_lock(analysis);
                analysis.record_instance(ait->first, overlap);
              }
            }
            if (!(antivalid_mask * partial_valid_fields))
            {
              for (ViewExprMaskSets::const_iterator pit =
                       partial_valid_instances.begin();
                   pit != partial_valid_instances.end(); pit++)
              {
                if (!pit->first->is_instance_view())
                  continue;
                if (!(antivalid_mask * pit->second.get_valid_mask()))
                  continue;
                if (!collective->aliases(pit->first->as_instance_view()))
                  continue;
                for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator
                         it = pit->second.begin();
                     it != pit->second.end(); it++)
                {
                  const FieldMask overlap = it->second & antivalid_mask;
                  if (!overlap)
                    continue;
                  if (!expr_covers)
                  {
                    if ((it->first != set_expr) && (it->first != expr))
                    {
                      IndexSpaceExpression* intersection =
                          runtime->intersect_index_spaces(expr, it->first);
                      if (!intersection->is_empty())
                      {
                        AutoLock a_lock(analysis);
                        analysis.record_instance(ait->first, overlap);
                      }
                    }
                    else
                    {
                      AutoLock a_lock(analysis);
                      analysis.record_instance(ait->first, overlap);
                    }
                  }
                  else
                  {
                    AutoLock a_lock(analysis);
                    analysis.record_instance(ait->first, overlap);
                  }
                }
              }
            }
          }
          else if (
              ait->first->is_instance_view() && !collective_instances.empty() &&
              !(antivalid_mask * collective_instances.get_valid_mask()))
          {
            IndividualView* view = ait->first->as_individual_view();
            // Check against any collective views to see if we alias
            for (lng::FieldMaskMap<CollectiveView>::const_iterator cit =
                     collective_instances.begin();
                 cit != collective_instances.end(); cit++)
            {
              if (antivalid_mask * cit->second)
                continue;
              if (!view->aliases(cit->first))
                continue;
              total_finder = total_valid_instances.find(cit->first);
              if (total_finder != total_valid_instances.end())
              {
                const FieldMask overlap = antivalid_mask & total_finder->second;
                if (!!overlap)
                {
                  AutoLock a_lock(analysis);
                  analysis.record_instance(ait->first, overlap);
                }
              }
              finder = partial_valid_instances.find(cit->first);
              if (finder != partial_valid_instances.end())
              {
                for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator
                         it = finder->second.begin();
                     it != finder->second.end(); it++)
                {
                  const FieldMask overlap = it->second & antivalid_mask;
                  if (!overlap)
                    continue;
                  if (!expr_covers)
                  {
                    if ((it->first != set_expr) && (it->first != expr))
                    {
                      IndexSpaceExpression* intersection =
                          runtime->intersect_index_spaces(expr, it->first);
                      if (!intersection->is_empty())
                      {
                        AutoLock a_lock(analysis);
                        analysis.record_instance(ait->first, overlap);
                      }
                    }
                    else
                    {
                      AutoLock a_lock(analysis);
                      analysis.record_instance(ait->first, overlap);
                    }
                  }
                  else
                  {
                    AutoLock a_lock(analysis);
                    analysis.record_instance(ait->first, overlap);
                  }
                }
              }
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_set(
        UpdateAnalysis& analysis, IndexSpaceExpression* expr,
        const bool expr_covers, const FieldMask& user_mask,
        std::set<RtEvent>& applied_events,
        const bool already_deferred /*=false*/)
    //--------------------------------------------------------------------------
    {
      // Already holding the eq_lock from EquivalenceSet::analyze method
      // Now that we're ready to perform the analysis
      // we need to lock the analysis
      AutoLock a_lock(analysis);
      // Check for any uninitialized data
      // Don't report uninitialized warnings for empty equivalence classes
      if (analysis.check_initialized && !set_expr->is_empty())
        check_for_uninitialized_data(
            analysis, expr, expr_covers, user_mask, applied_events);
      if (analysis.output_aggregator != nullptr)
        analysis.output_aggregator->clear_update_fields();
      if (IS_REDUCE(analysis.usage))
      {
        // Reduction-only
        // We only record reductions if the set expression is not empty
        // as we can't guarantee the reductions will ever be read for
        // empty equivalence sets which can lead to leaked instances
        if (!expr->is_empty())
        {
          record_reductions(analysis, expr, expr_covers, user_mask);
          // Flush any reductions for restricted fields and expressions
          if (!restricted_instances.empty())
          {
            const FieldMask flush_mask = user_mask & restricted_fields;
            if (!!flush_mask)
            {
              // Find all the restrictions that we overlap with and
              // apply reductions to them to flush any data
              for (ExprViewMaskSets::const_iterator rit =
                       restricted_instances.begin();
                   rit != restricted_instances.end(); rit++)
              {
                const FieldMask overlap =
                    flush_mask & rit->second.get_valid_mask();
                if (!overlap)
                  continue;
                if (expr_covers)
                {
                  // Expression covers the full restriction
                  if (rit->first->get_volume() == set_expr->get_volume())
                    apply_restricted_reductions(
                        rit->second, set_expr, true /*covers*/, overlap,
                        analysis.output_aggregator, nullptr /*no guard*/,
                        &analysis, true /*track events*/, analysis.trace_info,
                        nullptr /*no applied expr tracking*/);
                  else
                    apply_restricted_reductions(
                        rit->second, rit->first, false /*covers*/, overlap,
                        analysis.output_aggregator, nullptr /*no guard*/,
                        &analysis, true /*track events*/, analysis.trace_info,
                        nullptr /*no applied expr tracking*/);
                }
                else if (rit->first == set_expr)
                {
                  // Restriction covers the full expression
                  apply_restricted_reductions(
                      rit->second, expr, expr_covers, overlap,
                      analysis.output_aggregator, nullptr /*no guard*/,
                      &analysis, true /*track events*/, analysis.trace_info,
                      nullptr /*no applied expr tracking*/);
                }
                else
                {
                  // Check for partial overlap of restrction
                  IndexSpaceExpression* expr_overlap =
                      runtime->intersect_index_spaces(expr, rit->first);
                  if (expr_overlap->is_empty())
                    continue;
                  if (expr_overlap->get_volume() == expr->get_volume())
                    apply_restricted_reductions(
                        rit->second, expr, expr_covers, overlap,
                        analysis.output_aggregator, nullptr /*no guard*/,
                        &analysis, true /*track events*/, analysis.trace_info,
                        nullptr /*no applied expr tracking*/);
                  else
                    apply_restricted_reductions(
                        rit->second, expr_overlap, false /*covers*/, overlap,
                        analysis.output_aggregator, nullptr /*no guard*/,
                        &analysis, true /*track events*/, analysis.trace_info,
                        nullptr /*no applied expr tracking*/);
                }
              }
            }
          }
        }
      }
      else if (IS_WRITE_DISCARD(analysis.usage))
      {
        // Write-only
        // Update the initialized data before messing with the user mask
        update_initialized_data(expr, expr_covers, user_mask);
        const FieldMask reduce_filter = reduction_fields & user_mask;
        if (!!reduce_filter)
          filter_reduction_instances(expr, expr_covers, reduce_filter);
        local::FieldMaskMap<InstanceView> new_instances;
        for (unsigned idx = 0; idx < analysis.target_views.size(); idx++)
        {
          const FieldMask overlap =
              user_mask & analysis.target_views[idx].get_valid_mask();
          if (!overlap)
            continue;
          for (op::FieldMaskMap<InstanceView>::const_iterator it =
                   analysis.target_views[idx].begin();
               it != analysis.target_views[idx].end(); it++)
          {
            const FieldMask inst_overlap = user_mask & overlap;
            if (!inst_overlap)
              continue;
            new_instances.insert(it->first, inst_overlap);
          }
        }
        // Filter any normal instances that will be overwritten
        const FieldMask non_restricted = user_mask - restricted_fields;
        if (!!non_restricted)
        {
          filter_valid_instances(expr, expr_covers, non_restricted);
          // Record any non-restricted instances
          record_instances(
              expr, expr_covers, non_restricted, FieldMapView(new_instances));
        }
        // Update any tracing views
        if (analysis.trace_info.recording)
        {
          for (unsigned idx = 0; idx < analysis.target_views.size(); idx++)
          {
            for (op::FieldMaskMap<InstanceView>::const_iterator it =
                     analysis.target_views[idx].begin();
                 it != analysis.target_views[idx].end(); it++)
            {
              const FieldMask overlap = user_mask & it->second;
              if (!!overlap)
                update_tracing_write_discard_view(it->first, expr, overlap);
            }
          }
        }
        // Issue copy-out copies for any restricted fields
        if (!restricted_instances.empty())
        {
          const FieldMask restricted_mask = user_mask & restricted_fields;
          if (!!restricted_mask)
          {
            filter_unrestricted_instances(expr, expr_covers, restricted_mask);
            // Record any of our instances that are unrestricted
            record_unrestricted_instances(
                expr, expr_covers, restricted_mask,
                FieldMapView(new_instances));
            copy_out(
                expr, expr_covers, restricted_mask, FieldMapView(new_instances),
                &analysis, analysis.trace_info, analysis.output_aggregator);
          }
        }
      }
      else if (
          IS_READ_ONLY(analysis.usage) && !read_only_guards.empty() &&
          !(user_mask * read_only_guards.get_valid_mask()))
      {
        // If we're doing read-only mode, get the set of events that
        // we need to wait for before we can do our registration, this
        // ensures that we serialize read-only operations correctly
        // In order to avoid deadlock we have to make different copy fill
        // aggregators for each of the different fields of prior updates
        local::FieldMaskMap<CopyFillAggregator> to_add;
        for (shrt::FieldMaskMap<CopyFillGuard>::iterator it =
                 read_only_guards.begin();
             it != read_only_guards.end(); it++)
        {
          const FieldMask guard_mask = user_mask & it->second;
          if (!guard_mask)
            continue;
          // No matter what record our dependences on the prior guards
#ifdef NON_AGGRESSIVE_AGGREGATORS
          const RtEvent guard_event = it->first->effects_applied;
          analysis.guard_events.insert(guard_event);
#else
          const RtEvent guard_event = it->first->guard_postcondition;
          if (analysis.original_source == local_space)
            analysis.guard_events.insert(guard_event);
          else
            analysis.guard_events.insert(it->first->effects_applied);
#endif
          CopyFillAggregator* input_aggregator = nullptr;
          // See if we have an input aggregator that we can use now
          std::map<RtEvent, CopyFillAggregator*>::const_iterator finder =
              analysis.input_aggregators.find(guard_event);
          if (finder != analysis.input_aggregators.end())
          {
            input_aggregator = finder->second;
            if (input_aggregator != nullptr)
              input_aggregator->clear_update_fields();
          }
          // Use this to see if any new updates are recorded
          update_set_internal(
              input_aggregator, it->first, &analysis, analysis.usage, expr,
              expr_covers, guard_mask, analysis.target_instances,
              analysis.target_views, analysis.source_views, analysis.trace_info,
              analysis.record_valid);
          // If we did any updates record ourselves as the new guard here
          if ((input_aggregator != nullptr) &&
              ((finder == analysis.input_aggregators.end()) ||
               input_aggregator->has_update_fields()))
          {
            if (finder == analysis.input_aggregators.end())
              analysis.input_aggregators[guard_event] = input_aggregator;
            const FieldMask& update_mask =
                input_aggregator->get_update_fields();
            // Record this as a guard for later operations
            to_add.insert(input_aggregator, update_mask);
            if (!input_aggregator->record_guard_set(this, true /*read only*/))
              std::abort();
            // Remove the current guard since it doesn't matter anymore
            it.filter(update_mask);
          }
        }
        if (!to_add.empty())
        {
          for (local::FieldMaskMap<CopyFillAggregator>::const_iterator it =
                   to_add.begin();
               it != to_add.end(); it++)
            read_only_guards.insert(it->first, it->second);
        }
        // If we have unguarded fields we can easily do those
        if (!!user_mask)
        {
          CopyFillAggregator* input_aggregator = nullptr;
          // See if we have an input aggregator that we can use now
          std::map<RtEvent, CopyFillAggregator*>::const_iterator finder =
              analysis.input_aggregators.find(RtEvent::NO_RT_EVENT);
          if (finder != analysis.input_aggregators.end())
          {
            input_aggregator = finder->second;
            if (input_aggregator != nullptr)
              input_aggregator->clear_update_fields();
          }
          update_set_internal(
              input_aggregator, nullptr /*no previous guard*/, &analysis,
              analysis.usage, expr, expr_covers, user_mask,
              analysis.target_instances, analysis.target_views,
              analysis.source_views, analysis.trace_info,
              analysis.record_valid);
          // If we made the input aggregator then store it
          if ((input_aggregator != nullptr) &&
              ((finder == analysis.input_aggregators.end()) ||
               input_aggregator->has_update_fields()))
          {
            analysis.input_aggregators[RtEvent::NO_RT_EVENT] = input_aggregator;
            // Record this as a guard for later operations
            read_only_guards.insert(
                input_aggregator, input_aggregator->get_update_fields());
            if (!input_aggregator->record_guard_set(this, true /*read only*/))
              std::abort();
          }
        }
        // Record tracing views
        if (analysis.trace_info.recording)
        {
          for (unsigned idx = 0; idx < analysis.target_views.size(); idx++)
          {
            for (op::FieldMaskMap<InstanceView>::const_iterator it =
                     analysis.target_views[idx].begin();
                 it != analysis.target_views[idx].end(); it++)
            {
              const FieldMask overlap = it->second & user_mask;
              if (!!overlap)
                update_tracing_read_only_view(it->first, expr, overlap);
            }
          }
        }
      }
      else
      {
        // Read-write or read-only case
        // Read-only case if there are no guards
        CopyFillAggregator* input_aggregator = nullptr;
        // See if we have an input aggregator that we can use now
        std::map<RtEvent, CopyFillAggregator*>::const_iterator finder =
            analysis.input_aggregators.find(RtEvent::NO_RT_EVENT);
        if (finder != analysis.input_aggregators.end())
        {
          input_aggregator = finder->second;
          if (input_aggregator != nullptr)
            input_aggregator->clear_update_fields();
        }
        update_set_internal(
            input_aggregator, nullptr /*no previous guard*/, &analysis,
            analysis.usage, expr, expr_covers, user_mask,
            analysis.target_instances, analysis.target_views,
            analysis.source_views, analysis.trace_info, analysis.record_valid);
        if (IS_WRITE(analysis.usage))
        {
          update_initialized_data(expr, expr_covers, user_mask);
          // Issue copy-out copies for any restricted fields if we wrote stuff
          if (!restricted_instances.empty())
          {
            const FieldMask restricted_mask = user_mask & restricted_fields;
            if (!!restricted_mask)
              copy_out(
                  expr, expr_covers, restricted_mask, analysis.target_instances,
                  analysis.target_views, &analysis, analysis.trace_info,
                  analysis.output_aggregator);
          }
        }
        // If we made the input aggregator then store it
        if ((input_aggregator != nullptr) &&
            ((finder == analysis.input_aggregators.end()) ||
             input_aggregator->has_update_fields()))
        {
          analysis.input_aggregators[RtEvent::NO_RT_EVENT] = input_aggregator;
          // Record this as a guard for later read-only operations
          if (IS_READ_ONLY(analysis.usage))
          {
            read_only_guards.insert(
                input_aggregator, input_aggregator->get_update_fields());
            if (!input_aggregator->record_guard_set(this, true /*read only*/))
              std::abort();
          }
        }
        if (analysis.trace_info.recording)
        {
          for (unsigned idx = 0; idx < analysis.target_views.size(); idx++)
          {
            for (op::FieldMaskMap<InstanceView>::const_iterator it =
                     analysis.target_views[idx].begin();
                 it != analysis.target_views[idx].end(); it++)
            {
              const FieldMask overlap = user_mask & it->second;
              if (!overlap)
                continue;
              if (IS_READ_ONLY(analysis.usage))
                update_tracing_read_only_view(it->first, expr, overlap);
              else
                update_tracing_read_write_view(it->first, expr, overlap);
            }
          }
        }
      }
      // If we're doing a read-discard then we can invalidate the state
      if (IS_OUTPUT_DISCARD(analysis.usage))
        invalidate_state(expr, expr_covers, user_mask, false /*record*/);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::check_for_migration(
        PhysicalAnalysis& analysis, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
#ifndef LEGION_DISABLE_EQUIVALENCE_SET_MIGRATION
      legion_assert(is_logical_owner());
      // If we've replicated the logical owner state then it is not safe to
      // migrate this equivalence set because we can't currently establish
      // when all the collective analyses that might be relying on that
      // replicated knowledge are in a consistent state. If you ever want
      // to change this then you need to establish a consistent point in
      // time where all the collective analyses have all concluded that
      // the logical owner is the same. You cannot safely migrate this
      // equivalence set if some participants in a collective analysis
      // conclude there is one logical owner while other participants will
      // conclude that there is a different logical owner. That is very hard
      // to do right now because we don't track when analyses are done using
      // an equivalence set, so it's hard to know when we're in a
      // consistent state to be able to do the migration.
      if (replicated_owner_state != nullptr)
        return;
      const AddressSpaceID eq_source = analysis.original_source;
      // Record our user in the set of previous users
      bool found = false;
      std::vector<std::pair<AddressSpaceID, unsigned>>& current_samples =
          user_samples[migration_index];
      for (std::pair<AddressSpaceID, unsigned>& it : current_samples)
      {
        if (it.first != eq_source)
          continue;
        found = true;
        it.second++;
        break;
      }
      if (!found)
        current_samples.emplace_back(
            std::pair<AddressSpaceID, unsigned>(eq_source, 1));
      // Increase the sample count and if we haven't done enough
      // for a test then we can return and keep going
      if (++sample_count < SAMPLES_PER_MIGRATION_TEST)
      {
        // Check to see if the request bounced off a stale owner
        // and we should send the update message
        if ((eq_source != analysis.previous) && (eq_source != local_space) &&
            (eq_source != logical_owner_space))
        {
          RtUserEvent notification_event = Runtime::create_rt_user_event();
          EquivalenceSetOwnerUpdate rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(logical_owner_space);
            rez.serialize(notification_event);
          }
          rez.dispatch(eq_source);
          applied_events.insert(notification_event);
        }
        return;
      }
      // Issue a warning and don't migrate if we hit this case
      if (current_samples.size() == SAMPLES_PER_MIGRATION_TEST)
      {
        Warning warning;
        warning
            << "Internal runtime performance warning: equivalence set "
            << std::hex << did << std::dec << " has " << current_samples.size()
            << " different users which is the same as the sampling rate of "
            << SAMPLES_PER_MIGRATION_TEST << ". Region requirement "
            << analysis.index << " of " << *analysis.op
            << " triggered this warning. Please report this application use "
            << "case to the Legion developers mailing list.";
        warning.raise();
        // Reset the data structures for the next run
        current_samples.clear();
        sample_count = 0;
        return;
      }
      // Sort the current samples so that they are in order for
      // single epoch cases, for multi-epoch cases they will be
      // sorted by the summary computation below
      if ((MIGRATION_EPOCHS == 1) && (current_samples.size() > 1))
        std::sort(current_samples.begin(), current_samples.end());
      // Increment this for the next pass
      migration_index = (migration_index + 1) % MIGRATION_EPOCHS;
      std::vector<std::pair<AddressSpaceID, unsigned>>& next_samples =
          user_samples[migration_index];
      if (MIGRATION_EPOCHS > 1)
      {
        // Compute the summary from all the epochs into the epoch
        // that we are about to clear
        std::map<AddressSpaceID, unsigned> summary(
            next_samples.begin(), next_samples.end());
        for (unsigned idx = 1; idx < MIGRATION_EPOCHS; idx++)
        {
          const std::vector<std::pair<AddressSpaceID, unsigned>>&
              other_samples =
                  user_samples[(migration_index + idx) % MIGRATION_EPOCHS];
          for (const std::pair<AddressSpaceID, unsigned>& it : other_samples)
          {
            std::map<AddressSpaceID, unsigned>::iterator finder =
                summary.find(it.first);
            if (finder == summary.end())
              summary.insert(it);
            else
              finder->second += it.second;
          }
        }
        next_samples.clear();
        next_samples.insert(
            next_samples.begin(), summary.begin(), summary.end());
      }
      AddressSpaceID new_logical_owner = logical_owner_space;
      legion_assert(!next_samples.empty());
      if (next_samples.size() > 1)
      {
        int logical_owner_count = -1;
        // Figure out which node(s) has/have the most uses
        // Make sure that the current owner node is sticky
        // if it is tied for the most uses
        unsigned max_count = next_samples[0].second;
        AddressSpaceID max_user = next_samples[0].first;
        for (unsigned idx = 1; idx < next_samples.size(); idx++)
        {
          const AddressSpaceID user = next_samples[idx].first;
          const unsigned user_count = next_samples[idx].second;
          if (user == logical_owner_space)
            logical_owner_count = user_count;
          if (user_count < max_count)
            continue;
          // This is the part where we guarantee stickiness
          if ((user_count == max_count) && (user != logical_owner_space))
            continue;
          max_count = user_count;
          max_user = user;
        }
        if (logical_owner_count > 0)
        {
          if (logical_owner_space != max_user)
          {
            // If the logical owner is one of the current users then
            // we really better have a good reason to move this
            // equivalence set to a new node. For now the difference
            // between max_count and the current owner count has to
            // be greater than the number of nodes that we see participating
            // on this equivalence set. This heuristic should avoid
            // the ping-pong case even when our sampling rate does not
            // naturally align with the number of nodes participating
            if ((max_count - unsigned(logical_owner_count)) >
                next_samples.size())
              new_logical_owner = max_user;
          }
        }
        else
          // If we didn't have the current logical owner then
          // just pick the maximum one
          new_logical_owner = max_user;
      }
      else
        // If all the requests came from the same node, send it there
        new_logical_owner = next_samples[0].first;
      // This always get reset here
      sample_count = 0;
      // Reset this for the next iteration
      next_samples.clear();
      // See if we are actually going to do the migration
      if (logical_owner_space == new_logical_owner)
      {
        // No need to do the migration in this case
        // Check to see if the request bounced off a stale owner
        // and we should send the update message
        if ((eq_source != analysis.previous) && (eq_source != local_space) &&
            (eq_source != logical_owner_space))
        {
          RtUserEvent notification_event = Runtime::create_rt_user_event();
          EquivalenceSetOwnerUpdate rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(logical_owner_space);
            rez.serialize(notification_event);
          }
          rez.dispatch(eq_source);
          applied_events.insert(notification_event);
        }
        return;
      }
      // At this point we've decided to do the migration
      log_migration.info(
          "Migrating Equivalence Set %llx from %d to %d", did, local_space,
          new_logical_owner);
      const FieldMask all_ones(LEGION_FIELD_MASK_FIELD_ALL_ONES);
      // Do the migration
      EquivalenceSetMigration rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        pack_state(
            rez, new_logical_owner, did, set_expr, set_expr, true /*covers*/,
            all_ones, true /*pack guards*/, true /*pack invalids*/);
        pack_global_ref();
      }
      rez.dispatch(new_logical_owner);
      invalidate_state(set_expr, true /*covers*/, all_ones, false /*record*/);
      // Also invalidate the partial invalidations since we know we migrated
      // them all to the new owner node
      for (lng::FieldMaskMap<IndexSpaceExpression>::iterator it =
               partial_invalidations.begin();
           it != partial_invalidations.end(); it++)
        if (it->first->remove_nested_expression_reference(did))
          delete it->first;
      partial_invalidations.clear();
      // Now we can change the logical owner space
      logical_owner_space = new_logical_owner;
#endif  // LEGION_DISABLE_EQUIVALENCE_SET MIGRATION
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::defer_analysis(
        AutoTryLock& eq, PhysicalAnalysis& analysis, const FieldMask& mask,
        std::set<RtEvent>& deferral_events, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      legion_assert(!eq.has_lock());
      // See if we've already deferred this or not
      if (!already_deferred)
      {
        const RtUserEvent deferral_event = Runtime::create_rt_user_event();
        const RtEvent precondition = chain_deferral_events(deferral_event);
        analysis.defer_analysis(
            precondition, this, mask, deferral_events, applied_events,
            deferral_event);
      }
      else
        analysis.defer_analysis(
            eq.try_next(), this, mask, deferral_events, applied_events);
    }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::is_remote_analysis(
        PhysicalAnalysis& analysis, FieldMask& mask,
        std::set<RtEvent>& deferral_events, std::set<RtEvent>& applied_events,
        const bool expr_covers)
    //--------------------------------------------------------------------------
    {
      // Check to see if the analysis is replicated or not
      if (analysis.is_replicated())
      {
        const CollectiveMapping* mapping = analysis.get_replicated_mapping();
        legion_assert(mapping != nullptr);
        legion_assert(!analysis.immutable);
        // Check to see if we have a replicated owner node
        if (!replicate_logical_owner_space(
                local_space, mapping, false /*lock*/))
        {
          legion_assert(replicated_owner_state->ready.exists());
          analysis.defer_analysis(
              replicated_owner_state->ready, this, mask, deferral_events,
              applied_events);
          return true;
        }
        // Now figure out which analysis is going to perform the traversal
        // If an analysis is already local to an equivalence set then we'll
        // want it to do the traversal to minimimize communication. In the
        // case where the current logical owner space is not contained in
        // the mapping then we'll simply pick the space in our collective
        // that is closest to the owner in the set of linear space names
        // (assuming some degree of locality).
        if (!mapping->contains(logical_owner_space))
        {
          // If we're on the logical owner then that means we're the analysis
          // that already got migrated to the logical owner
          if (is_logical_owner())
          {
            legion_assert(
                analysis.original_source == mapping->find_nearest(local_space));
            legion_assert(analysis.is_collective_first_local());
            return false;
          }
          // There aren't any analyses that will be local to the
          // logical owner space so we need to pick the closest one
          if ((local_space == mapping->find_nearest(logical_owner_space)) &&
              analysis.is_collective_first_local())
            analysis.record_remote(this, mask, logical_owner_space);
        }
        else
        {
          // At least one node is local, see if we're it
          if ((logical_owner_space == local_space) &&
              analysis.is_collective_first_local())
            return false;
        }
        return true;
      }
      else
      {
        // See if we are the logical owner or not
        if (!is_logical_owner())
        {
          // Not the logical owner, so just need to send it to the owner
          analysis.record_remote(this, mask, logical_owner_space);
          return true;
        }
        else
          return false;
      }
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::ReplicatedOwnerState::ReplicatedOwnerState(bool val)
      : ready(
            val ? RtUserEvent::NO_RT_USER_EVENT :
                  Runtime::create_rt_user_event())
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::replicate_logical_owner_space(
        AddressSpaceID source, const CollectiveMapping* mapping, bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock eq(eq_lock);
        return replicate_logical_owner_space(
            source, mapping, false /*need lock*/);
      }
      if (mapping == nullptr)
      {
        legion_assert(source != local_space);
        // We're just chasing the logical owner at this point
        if (is_logical_owner())
        {
          EquivalenceSetReplicationResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(local_space);
          }
          rez.dispatch(source);
          if (replicated_owner_state == nullptr)
            replicated_owner_state = new ReplicatedOwnerState(true /*valid*/);
          replicated_owner_state->children.emplace_back(source);
        }
        else
        {
          // Keep forwarding this on searching for the owner space
          EquivalenceSetReplicationRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            CollectiveMapping::pack_null(rez);
            rez.serialize(source);
          }
          rez.dispatch(logical_owner_space);
        }
        return false;
      }
      if (replicated_owner_state == nullptr)
      {
        replicated_owner_state = new ReplicatedOwnerState(is_logical_owner());
        if (!is_logical_owner())
        {
          legion_assert(mapping->contains(local_space));
          const AddressSpaceID origin = mapping->get_origin();
          if (local_space != origin)
          {
            const AddressSpaceID parent =
                mapping->get_parent(origin, local_space);
            // Send the request on to our parent space
            EquivalenceSetReplicationRequest rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              mapping->pack(rez);
              rez.serialize(local_space);
            }
            rez.dispatch(parent);
          }
          else
          {
            // Send the request on to whomever we thought was the previous owner
            EquivalenceSetReplicationRequest rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              CollectiveMapping::pack_null(rez);
              rez.serialize(local_space);
            }
            rez.dispatch(logical_owner_space);
          }
        }
      }
      if (source != local_space)
      {
        replicated_owner_state->children.emplace_back(source);
      }
      if (replicated_owner_state->is_valid())
      {
        // If we're already replicated send back the response now
        if (source != local_space)
        {
          EquivalenceSetReplicationResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(logical_owner_space);
          }
          rez.dispatch(source);
        }
        return true;
      }
      else
        // We're just waiting on the response here to send out the result
        return false;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::process_replication_response(AddressSpaceID owner)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      {
        AutoLock eq(eq_lock);
        legion_assert(replicated_owner_state != nullptr);
        legion_assert(!replicated_owner_state->is_valid());
        legion_assert(!is_logical_owner());
        logical_owner_space = owner;
        // Send out messages to all the other nodes that requested them
        for (std::vector<AddressSpaceID>::iterator it =
                 replicated_owner_state->children.begin();
             it != replicated_owner_state->children.end();
             /*nothing*/)
        {
          if ((*it) != owner)
          {
            EquivalenceSetReplicationResponse rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(owner);
            }
            rez.dispatch(*it);
            it++;
          }
          else
            // If one of our children ended up becoming the owner then
            // this is where we break the dependence with it
            // This happens when we are making a new replicated owner
            // state after a migration has taken place to a new node
            it = replicated_owner_state->children.erase(it);
        }
        to_trigger = replicated_owner_state->ready;
        replicated_owner_state->ready = RtUserEvent::NO_RT_USER_EVENT;
      }
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void EquivalenceSet::check_for_uninitialized_data(
        T& analysis, IndexSpaceExpression* expr, const bool expr_covers,
        FieldMask uninit, std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      // Do the easy check for the full cover which will be the common case
      lng::FieldMaskMap<IndexSpaceExpression>::const_iterator finder =
          initialized_data.find(set_expr);
      if (finder != initialized_data.end())
      {
        uninit -= finder->second;
        if (!uninit)
          return;
      }
      if (!expr_covers)
      {
        finder = initialized_data.find(expr);
        if (finder != initialized_data.end())
        {
          uninit -= finder->second;
          if (!uninit)
            return;
        }
      }
      // All the rest of these are partial so only test them if
      // expr_covers is false because we know they aren't covered otherwise
      if (!expr_covers)
      {
        for (lng::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                 initialized_data.begin();
             it != initialized_data.end(); it++)
        {
          if (uninit * it->second)
            continue;
          // Don't actually need to subtract here since we don't care
          // about the difference size, just care about domination
          IndexSpaceExpression* overlap_expr =
              runtime->intersect_index_spaces(it->first, expr);
          if (overlap_expr->get_volume() != expr->get_volume())
            continue;
          uninit -= it->second;
          if (!uninit)
            return;
        }
      }
      // Record anything that we have left
      analysis.record_uninitialized(uninit, applied_events);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_initialized_data(
        IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& user_mask)
    //--------------------------------------------------------------------------
    {
      if (!expr_covers)
      {
        FieldMask subinit = user_mask;
        lng::FieldMaskMap<IndexSpaceExpression>::iterator finder =
            initialized_data.find(set_expr);
        // Check to see if we've already initialized it for the full set_expr
        if (finder != initialized_data.end())
        {
          subinit -= finder->second;
          // Already initialized for full expression so we are done
          if (!subinit)
            return;
        }
        local::FieldMaskMap<IndexSpaceExpression> to_add;
        std::vector<IndexSpaceExpression*> to_delete;
        for (lng::FieldMaskMap<IndexSpaceExpression>::iterator it =
                 initialized_data.begin();
             it != initialized_data.end(); it++)
        {
          if ((it->first == set_expr) || (it->first == expr))
            continue;
          const FieldMask overlap = subinit & it->second;
          if (!overlap)
            continue;
          // Compute the union expression
          IndexSpaceExpression* union_expr =
              runtime->union_index_spaces(it->first, expr);
          const size_t union_size = union_expr->get_volume();
          legion_assert(union_size <= set_expr->get_volume());
          if (union_size == it->first->get_volume())
          {
            // Existing expression already covers expr
            subinit -= overlap;
            if (!subinit)
              break;
          }
          else if (union_size == set_expr->get_volume())
          {
            // Union is the same as the set expression
            if (finder != initialized_data.end())
              finder.merge(overlap);
            else
              to_add.insert(set_expr, overlap);
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(it->first);
            subinit -= overlap;
            if (!subinit)
              break;
          }
          else if (union_size == expr->get_volume())
          {
            // New expression covers the old expression
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(it->first);
          }
          else
          {
            // Union is bigger than both expression but not set_expr
            to_add.insert(union_expr, overlap);
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(it->first);
            subinit -= overlap;
            if (!subinit)
              break;
          }
        }
        // Add new ones
        if (!to_add.empty())
        {
          for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   to_add.begin();
               it != to_add.end(); it++)
            if (initialized_data.insert(it->first, it->second))
              it->first->add_nested_expression_reference(did);
        }
        // Delete after adding to keep expressions valid
        if (!to_delete.empty())
        {
          for (IndexSpaceExpression* it : to_delete)
          {
            if (to_add.find(it) != to_add.end())
              continue;
            initialized_data.erase(it);
            if (it->remove_nested_expression_reference(did))
              delete it;
          }
        }
        // Add the new expression if we still have fields to add
        if (!!subinit && initialized_data.insert(expr, subinit))
          expr->add_nested_expression_reference(did);
      }
      else
      {
        // Remove all other expressions with overlapping fields
        if (!(user_mask * initialized_data.get_valid_mask()))
        {
          std::vector<IndexSpaceExpression*> to_delete;
          for (lng::FieldMaskMap<IndexSpaceExpression>::iterator it =
                   initialized_data.begin();
               it != initialized_data.end(); it++)
          {
            if (it->first == set_expr)
              continue;
            it.filter(user_mask);
            if (!it->second)
              to_delete.emplace_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (IndexSpaceExpression* it : to_delete)
            {
              initialized_data.erase(it);
              if (it->remove_nested_expression_reference(did))
                delete it;
            }
          }
        }
        if (initialized_data.insert(set_expr, user_mask))
          set_expr->add_nested_expression_reference(did);
      }
    }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::filter_partial_invalidations(
        const FieldMask& mask, RtUserEvent& filtered)
    //--------------------------------------------------------------------------
    {
      // It used to be the case that partial invalidations were monotonic
      // growing in an equivalence set, but then we added the
      // find_congruent_existing_equivalence_set function which allowed an
      // equivalence set to be reused with just the right shape even if it
      // has been previously invalidated for certain fields, in such cases
      // we need to remove the partial invalidations when we apply the new
      // state here so we filter out any partial invalidation overlap.
      // Importantly since we know that this only occurs with
      // find_congruent_existing_equivalence_set which ensures that any
      // initialization update that overlaps on a field of a previous
      // partial invalidation implies that we're going to be updating the
      // whole equivalence set back to an initialized state for that
      // particular field so we can invalidate all partial invalidations
      // for that particular field and not have to test expressions.
      AutoLock eq(eq_lock);
      if (!is_logical_owner())
      {
        if (!filtered.exists())
          filtered = Runtime::create_rt_user_event();
        EquivalenceSetFilterInvalidations rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(mask);
          rez.serialize(filtered);
        }
        rez.dispatch(logical_owner_space);
        return false;
      }
      else
      {
        if (!(mask * partial_invalidations.get_valid_mask()))
        {
          // Remove any partial invalidations with overlapping fields
          std::vector<IndexSpaceExpression*> to_delete;
          for (lng::FieldMaskMap<IndexSpaceExpression>::iterator it =
                   partial_invalidations.begin();
               it != partial_invalidations.end(); it++)
          {
            it.filter(mask);
            if (!it->second)
              to_delete.emplace_back(it->first);
          }
          for (IndexSpaceExpression* it : to_delete)
          {
            partial_invalidations.erase(it);
            if (it->remove_nested_expression_reference(did))
              delete it;
          }
          partial_invalidations.filter_valid_mask(mask);
        }
        return true;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSetFilterInvalidations::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet* set =
          runtime->find_or_request_equivalence_set(did, ready);
      FieldMask mask;
      derez.deserialize(mask);
      RtUserEvent filtered;
      derez.deserialize(filtered);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      if (set->filter_partial_invalidations(mask, filtered))
        Runtime::trigger_event(filtered);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void EquivalenceSet::record_instances(
        IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& record_mask, const FieldMapView<T>& target_insts)
    //--------------------------------------------------------------------------
    {
      bool rebuild_partial = false;
      if (expr_covers)
      {
        if (!(target_insts.get_valid_mask() - record_mask))
        {
          for (typename FieldMapView<T>::const_iterator it =
                   target_insts.begin();
               it != target_insts.end(); it++)
          {
            if (total_valid_instances.insert(it->first, it->second))
              it->first->add_nested_valid_ref(did);
            // Check if this is a collective view we need to track
            if (it->first->is_collective_view() &&
                collective_instances.insert(
                    it->first->as_collective_view(), it->second))
              it->first->add_nested_resource_ref(did);
            // Check to see if there are any copies of this to filter
            // from the partially valid instances
            ViewExprMaskSets::iterator finder =
                partial_valid_instances.find(it->first);
            if ((finder != partial_valid_instances.end()) &&
                !(finder->second.get_valid_mask() * it->second))
            {
              rebuild_partial = true;
              if (!(finder->second.get_valid_mask() - it->second))
              {
                // We're pruning everything so remove them all now
                for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator
                         eit = finder->second.begin();
                     eit != finder->second.end(); eit++)
                  if (eit->first->remove_nested_expression_reference(did))
                    delete eit->first;
                // Remove the reference, no need to check for deletion
                // since we know we added the same reference above
                finder->first->remove_nested_valid_ref(did);
                partial_valid_instances.erase(finder);
              }
              else
              {
                // Filter out the ones we now subsume
                local::vector<IndexSpaceExpression*> to_delete;
                for (shrt::FieldMaskMap<IndexSpaceExpression>::iterator eit =
                         finder->second.begin();
                     eit != finder->second.end(); eit++)
                {
                  eit.filter(it->second);
                  if (!eit->second)
                    to_delete.emplace_back(eit->first);
                }
                for (local::vector<IndexSpaceExpression*>::const_iterator eit =
                         to_delete.begin();
                     eit != to_delete.end(); eit++)
                {
                  finder->second.erase(*eit);
                  if ((*eit)->remove_nested_expression_reference(did))
                    delete (*eit);
                }
                if (finder->second.empty())
                {
                  // Remove the reference, no need to check for deletion
                  // since we know we added the same reference above
                  finder->first->remove_nested_valid_ref(did);
                  partial_valid_instances.erase(finder);
                }
                else
                  finder->second.tighten_valid_mask();
              }
            }
          }
        }
        else
        {
          for (typename FieldMapView<T>::const_iterator it =
                   target_insts.begin();
               it != target_insts.end(); it++)
          {
            const FieldMask valid_mask = it->second & record_mask;
            if (!valid_mask)
              continue;
            // Add it to the set
            if (total_valid_instances.insert(it->first, valid_mask))
              it->first->add_nested_valid_ref(did);
            // Check if this is a collective view we need to track
            if (it->first->is_collective_view() &&
                collective_instances.insert(
                    it->first->as_collective_view(), valid_mask))
              it->first->add_nested_resource_ref(did);
            // Check to see if there are any copies of this to filter
            // from the partially valid instances
            ViewExprMaskSets::iterator finder =
                partial_valid_instances.find(it->first);
            if ((finder != partial_valid_instances.end()) &&
                !(finder->second.get_valid_mask() * valid_mask))
            {
              rebuild_partial = true;
              if (!(finder->second.get_valid_mask() - valid_mask))
              {
                // We're pruning everything so remove them all now
                for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator
                         eit = finder->second.begin();
                     eit != finder->second.end(); eit++)
                  if (eit->first->remove_nested_expression_reference(did))
                    delete eit->first;
                // Remove the reference, no need to check for deletion
                // since we know we added the same reference above
                finder->first->remove_nested_valid_ref(did);
                partial_valid_instances.erase(finder);
              }
              else
              {
                // Filter out the ones we now subsume
                local::vector<IndexSpaceExpression*> to_delete;
                for (shrt::FieldMaskMap<IndexSpaceExpression>::iterator eit =
                         finder->second.begin();
                     eit != finder->second.end(); eit++)
                {
                  eit.filter(valid_mask);
                  if (!eit->second)
                    to_delete.emplace_back(eit->first);
                }
                for (local::vector<IndexSpaceExpression*>::const_iterator eit =
                         to_delete.begin();
                     eit != to_delete.end(); eit++)
                {
                  finder->second.erase(*eit);
                  if ((*eit)->remove_nested_expression_reference(did))
                    delete (*eit);
                }
                if (finder->second.empty())
                {
                  // Remove the reference, no need to check for deletion
                  // since we know we added the same reference above
                  finder->first->remove_nested_valid_ref(did);
                  partial_valid_instances.erase(finder);
                }
                else
                  finder->second.tighten_valid_mask();
              }
            }
          }
        }
      }
      else
      {
        if (!(target_insts.get_valid_mask() - record_mask))
        {
          for (typename FieldMapView<T>::const_iterator it =
                   target_insts.begin();
               it != target_insts.end(); it++)
            if (record_partial_valid_instance(it->first, expr, it->second))
              rebuild_partial = true;
        }
        else
        {
          for (typename FieldMapView<T>::const_iterator it =
                   target_insts.begin();
               it != target_insts.end(); it++)
          {
            const FieldMask valid_mask = it->second & record_mask;
            if (!valid_mask)
              continue;
            if (record_partial_valid_instance(it->first, expr, valid_mask))
              rebuild_partial = true;
          }
        }
      }
      if (rebuild_partial)
      {
        partial_valid_fields.clear();
        for (ViewExprMaskSets::const_iterator it =
                 partial_valid_instances.begin();
             it != partial_valid_instances.end(); it++)
          partial_valid_fields |= it->second.get_valid_mask();
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void EquivalenceSet::record_unrestricted_instances(
        IndexSpaceExpression* expr, const bool expr_covers,
        FieldMask record_mask, const FieldMapView<T>& target_insts)
    //--------------------------------------------------------------------------
    {
      legion_assert(!(record_mask - restricted_fields));
      // Check to see if there are any restrictions which cover the whole
      // set and therefore we know that there are no partial coverings
      ExprViewMaskSets::const_iterator finder =
          restricted_instances.find(set_expr);
      if (finder != restricted_instances.end())
      {
        if (tracing_postconditions != nullptr)
        {
          FieldMask overlap = record_mask & finder->second.get_valid_mask();
          if (!!overlap)
            invalidate_tracing_restricted_views(
                finder->second, set_expr, overlap);
        }
        record_mask -= finder->second.get_valid_mask();
        if (!record_mask)
          return;
      }
      // The only fields left here are the partial restrictions
      local::FieldMaskMap<IndexSpaceExpression> restrictions;
      for (const std::pair<
               IndexSpaceExpression* const, shrt::FieldMaskMap<InstanceView>>&
               it : restricted_instances)
      {
        if (it.first == finder->first)
          continue;
        FieldMask overlap = it.second.get_valid_mask() & record_mask;
        if (!overlap)
          continue;
        if (!expr_covers)
        {
          IndexSpaceExpression* overlap_expr =
              runtime->intersect_index_spaces(expr, it.first);
          if (!overlap_expr->is_empty())
          {
            restrictions.insert(overlap_expr, overlap);
            if (tracing_postconditions != nullptr)
              invalidate_tracing_restricted_views(
                  it.second, overlap_expr, overlap);
          }
        }
        else
        {
          restrictions.insert(it.first, overlap);
          if (tracing_postconditions != nullptr)
            invalidate_tracing_restricted_views(it.second, it.first, overlap);
        }
      }
      // Sort these into grouped field sets so we can union them before
      // doing the subtraction to figure out what we can record
      local::list<FieldSet<IndexSpaceExpression*>> restricted_sets;
      restrictions.compute_field_sets(record_mask, restricted_sets);
      bool need_partial_rebuild = false;
      for (local::list<FieldSet<IndexSpaceExpression*>>::const_iterator rit =
               restricted_sets.begin();
           rit != restricted_sets.end(); rit++)
      {
        IndexSpaceExpression* diff_expr = nullptr;
        if (!rit->elements.empty())
        {
          IndexSpaceExpression* union_expr =
              runtime->union_index_spaces(rit->elements);
          diff_expr = runtime->subtract_index_spaces(expr, union_expr);
        }
        else
          diff_expr = expr;
        if (!diff_expr->is_empty())
        {
          for (typename FieldMapView<T>::const_iterator it =
                   target_insts.begin();
               it != target_insts.end(); it++)
          {
            const FieldMask valid_mask = it->second & rit->set_mask;
            if (!valid_mask)
              continue;
            if (record_partial_valid_instance(it->first, diff_expr, valid_mask))
              need_partial_rebuild = true;
          }
        }
#ifdef LEGION_DEBUG
        record_mask -= rit->set_mask;
#endif
      }
      legion_assert(!record_mask);
      if (need_partial_rebuild)
      {
        partial_valid_fields.clear();
        for (ViewExprMaskSets::const_iterator it =
                 partial_valid_instances.begin();
             it != partial_valid_instances.end(); it++)
          partial_valid_fields |= it->second.get_valid_mask();
      }
    }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::record_partial_valid_instance(
        LogicalView* target, IndexSpaceExpression* expr, FieldMask valid_mask,
        bool check_total_valid)
    //--------------------------------------------------------------------------
    {
      bool need_rebuild = false;
      if (check_total_valid)
      {
        shrt::FieldMaskMap<LogicalView>::const_iterator finder =
            total_valid_instances.find(target);
        if (finder != total_valid_instances.end())
        {
          valid_mask -= finder->second;
          if (!valid_mask)
            return need_rebuild;
        }
      }
      partial_valid_fields |= valid_mask;
      ViewExprMaskSets::iterator finder = partial_valid_instances.find(target);
      if (finder != partial_valid_instances.end())
      {
        // See if we have any overlapping field expressions to add this to
        if (!(valid_mask * finder->second.get_valid_mask()))
        {
          std::vector<IndexSpaceExpression*> to_delete;
          local::FieldMaskMap<IndexSpaceExpression> to_add;
          bool need_tighten = false;
          for (shrt::FieldMaskMap<IndexSpaceExpression>::iterator it =
                   finder->second.begin();
               it != finder->second.end(); it++)
          {
            const FieldMask overlap = it->second & valid_mask;
            if (!overlap)
              continue;
            IndexSpaceExpression* union_expr =
                runtime->union_index_spaces(it->first, expr);
            const size_t union_size = union_expr->get_volume();
            legion_assert(union_size <= set_expr->get_volume());
            if (union_size == set_expr->get_volume())
            {
              // Hurray, we now cover the full expr so we can get
              // promoted up to the total valid instances
              it.filter(overlap);
              if (!it->second)
                to_delete.emplace_back(it->first);
              if (total_valid_instances.insert(target, overlap))
                target->add_nested_valid_ref(did);
              // No need for a collective instance check here since it
              // was already recorded in the partial valid instances
              need_tighten = true;
            }
            else if (union_size == expr->get_volume())
            {
              // We dominate the previous expression, so remove it
              // and put ourselves in
              it.filter(overlap);
              if (!it->second)
                to_delete.emplace_back(it->first);
              to_add.insert(expr, overlap);
            }
            else if (union_size > it->first->get_volume())
            {
              // Union dominates both so put it in instead
              it.filter(overlap);
              if (!it->second)
                to_delete.emplace_back(it->first);
              to_add.insert(union_expr, overlap);
            }
            // Else previous expr dominates so we can just leave it there
            valid_mask -= overlap;
            if (!valid_mask)
              break;
          }
          for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   to_add.begin();
               it != to_add.end(); it++)
            if (finder->second.insert(it->first, it->second))
              it->first->add_nested_expression_reference(did);
          for (IndexSpaceExpression* it : to_delete)
          {
            if (to_add.find(it) != to_add.end())
              continue;
            finder->second.erase(it);
            if (it->remove_nested_expression_reference(did))
              delete it;
          }
          if (!!valid_mask && finder->second.insert(expr, valid_mask))
            expr->add_nested_expression_reference(did);
          if (need_tighten)
          {
            if (finder->second.empty())
            {
              // Wow! everything got promoted up to total valid
              // instances, lucky us, remove the old partial stuff
              finder->first->remove_nested_valid_ref(did);
              partial_valid_instances.erase(finder);
            }
            else
              finder->second.tighten_valid_mask();
            if (!partial_valid_instances.empty())
              need_rebuild = true;
            else
              partial_valid_fields.clear();
          }
        }
        else if (finder->second.insert(expr, valid_mask))
          expr->add_nested_expression_reference(did);
      }
      else
      {
        partial_valid_instances[target].insert(expr, valid_mask);
        target->add_nested_valid_ref(did);
        expr->add_nested_expression_reference(did);
      }
      // Check to see if this is a collective view we need to record
      if (target->is_collective_view() &&
          collective_instances.insert(target->as_collective_view(), valid_mask))
        target->add_nested_resource_ref(did);
      return need_rebuild;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_valid_instances(
        IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& filter_mask,
        std::map<IndexSpaceExpression*, unsigned>* expr_refs_to_remove,
        std::map<LogicalView*, unsigned>* view_refs_to_remove)
    //--------------------------------------------------------------------------
    {
      legion_assert(!!filter_mask);
      // Clear out any collective instances
      if (!(filter_mask * collective_instances.get_valid_mask()))
      {
        // Remove all the overlapping collective instances
        if (!!(collective_instances.get_valid_mask() - filter_mask))
        {
          std::vector<CollectiveView*> to_delete;
          for (lng::FieldMaskMap<CollectiveView>::iterator it =
                   collective_instances.begin();
               it != collective_instances.end(); it++)
          {
            const FieldMask overlap = it->second & filter_mask;
            if (!overlap)
              continue;
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (CollectiveView* it : to_delete)
            {
              collective_instances.erase(it);
              if (it->remove_nested_resource_ref(did))
                delete it;
            }
          }
        }
        else
        {
          for (lng::FieldMaskMap<CollectiveView>::const_iterator it =
                   collective_instances.begin();
               it != collective_instances.end(); it++)
            if (it->first->remove_nested_resource_ref(did))
              delete it->first;
          collective_instances.clear();
        }
      }
      if (expr_covers)
      {
        // If the expr covers we can just filter everything
        if (!(filter_mask * total_valid_instances.get_valid_mask()))
        {
          // Clear out the total valid instances first
          std::vector<LogicalView*> to_delete;
          for (shrt::FieldMaskMap<LogicalView>::iterator it =
                   total_valid_instances.begin();
               it != total_valid_instances.end(); it++)
          {
            const FieldMask overlap = it->second & filter_mask;
            if (!overlap)
              continue;
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (LogicalView* it : to_delete)
            {
              total_valid_instances.erase(it);
              if (view_refs_to_remove != nullptr)
              {
                std::map<LogicalView*, unsigned>::iterator finder =
                    view_refs_to_remove->find(it);
                if (finder == view_refs_to_remove->end())
                  (*view_refs_to_remove)[it] = 1;
                else
                  finder->second += 1;
              }
              else if (it->remove_nested_valid_ref(did))
                delete it;
            }
          }
          total_valid_instances.filter_valid_mask(filter_mask);
        }
        if (!(filter_mask * partial_valid_fields))
        {
          // Then clear out the partial valid instances
          std::vector<LogicalView*> to_delete;
          for (ViewExprMaskSets::iterator pit = partial_valid_instances.begin();
               pit != partial_valid_instances.end(); pit++)
          {
            const FieldMask& summary_mask = pit->second.get_valid_mask();
            if (summary_mask * filter_mask)
              continue;
            else if (!(summary_mask - filter_mask))
            {
              // Invalidating all the expressions
              for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                       pit->second.begin();
                   it != pit->second.end(); it++)
              {
                if (expr_refs_to_remove != nullptr)
                {
                  std::map<IndexSpaceExpression*, unsigned>::iterator finder =
                      expr_refs_to_remove->find(it->first);
                  if (finder == expr_refs_to_remove->end())
                    (*expr_refs_to_remove)[it->first] = 1;
                  else
                    finder->second += 1;
                }
                else if (it->first->remove_nested_expression_reference(did))
                  delete it->first;
              }
              to_delete.emplace_back(pit->first);
            }
            else
            {
              // Only invalidating some of the expressions
              std::vector<IndexSpaceExpression*> to_erase;
              for (shrt::FieldMaskMap<IndexSpaceExpression>::iterator it =
                       pit->second.begin();
                   it != pit->second.end(); it++)
              {
                it.filter(filter_mask);
                if (!it->second)
                  to_erase.emplace_back(it->first);
              }
              if (!to_erase.empty())
              {
                for (IndexSpaceExpression* it : to_erase)
                {
                  pit->second.erase(it);
                  if (expr_refs_to_remove != nullptr)
                  {
                    std::map<IndexSpaceExpression*, unsigned>::iterator finder =
                        expr_refs_to_remove->find(it);
                    if (finder == expr_refs_to_remove->end())
                      (*expr_refs_to_remove)[it] = 1;
                    else
                      finder->second += 1;
                  }
                  else if (it->remove_nested_expression_reference(did))
                    delete it;
                }
              }
              pit->second.tighten_valid_mask();
            }
          }
          if (!to_delete.empty())
          {
            for (LogicalView* it : to_delete)
            {
              partial_valid_instances.erase(it);
              if (view_refs_to_remove != nullptr)
              {
                std::map<LogicalView*, unsigned>::iterator finder =
                    view_refs_to_remove->find(it);
                if (finder == view_refs_to_remove->end())
                  (*view_refs_to_remove)[it] = 1;
                else
                  finder->second += 1;
              }
              else if (it->remove_nested_valid_ref(did))
                delete it;
            }
          }
          partial_valid_fields -= filter_mask;
        }
      }
      else
      {
        // If the expr does not cover then we have to do partial filtering
        // Filter any partial data first
        if (!(filter_mask * partial_valid_fields))
        {
          std::vector<LogicalView*> to_delete;
          FieldMask still_partial_valid;
          for (ViewExprMaskSets::iterator pit = partial_valid_instances.begin();
               pit != partial_valid_instances.end(); pit++)
          {
            FieldMask view_overlap = pit->second.get_valid_mask() & filter_mask;
            if (!view_overlap)
              continue;
            std::vector<IndexSpaceExpression*> to_erase;
            local::FieldMaskMap<IndexSpaceExpression> to_add;
            for (shrt::FieldMaskMap<IndexSpaceExpression>::iterator it =
                     pit->second.begin();
                 it != pit->second.end(); it++)
            {
              const FieldMask overlap = it->second & view_overlap;
              if (!overlap)
                continue;
              IndexSpaceExpression* diff =
                  runtime->subtract_index_spaces(it->first, expr);
              if (diff->is_empty())
              {
                // filter expr covers, so remove it
                it.filter(overlap);
                if (!it->second)
                  to_erase.emplace_back(it->first);
              }
              else if (diff->get_volume() < it->first->get_volume())
              {
                // filter expr covers some, so these fields and make
                // the diff the new expression for these overlap fields
                it.filter(overlap);
                if (!it->second)
                  to_erase.emplace_back(it->first);
                to_add.insert(diff, overlap);
              }
              // else expr does not cover any so nothing to do here
              view_overlap -= overlap;
              if (!view_overlap)
                break;
            }
            for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                     to_add.begin();
                 it != to_add.end(); it++)
              if (pit->second.insert(it->first, it->second))
                it->first->add_nested_expression_reference(did);
            // Deletions after adding to make sure to keep referenes around
            for (IndexSpaceExpression* it : to_erase)
            {
              // Don't delete if we just added it
              if (to_add.find(it) != to_add.end())
                continue;
              shrt::FieldMaskMap<IndexSpaceExpression>::iterator finder =
                  pit->second.find(it);
              legion_assert(finder != pit->second.end());
              if (!!finder->second)
                continue;
              pit->second.erase(finder);
              if (expr_refs_to_remove != nullptr)
              {
                std::map<IndexSpaceExpression*, unsigned>::iterator finder2 =
                    expr_refs_to_remove->find(it);
                if (finder2 == expr_refs_to_remove->end())
                  (*expr_refs_to_remove)[it] = 1;
                else
                  finder2->second += 1;
              }
              else if (it->remove_nested_expression_reference(did))
                delete it;
            }
            if (!pit->second.empty())
            {
              pit->second.tighten_valid_mask();
              still_partial_valid |= pit->second.get_valid_mask();
              // Check if this a collective view to record
              if (pit->first->is_collective_view() &&
                  collective_instances.insert(
                      pit->first->as_collective_view(),
                      pit->second.get_valid_mask()))
                pit->first->add_nested_resource_ref(did);
            }
            else
              to_delete.emplace_back(pit->first);
          }
          for (LogicalView* const it : to_delete)
          {
            partial_valid_instances.erase(it);
            if (view_refs_to_remove != nullptr)
            {
              std::map<LogicalView*, unsigned>::iterator finder =
                  view_refs_to_remove->find(it);
              if (finder == view_refs_to_remove->end())
                (*view_refs_to_remove)[it] = 1;
              else
                finder->second += 1;
            }
            else if (it->remove_nested_valid_ref(did))
              delete it;
          }
          partial_valid_fields -= (filter_mask - still_partial_valid);
        }
        // Now we can filter the total valid instances back to the
        // partial valid instances
        if (!(filter_mask * total_valid_instances.get_valid_mask()))
        {
          std::vector<LogicalView*> to_delete;
          bool need_partial_rebuild = false;
          IndexSpaceExpression* diff_expr = nullptr;
          for (shrt::FieldMaskMap<LogicalView>::iterator it =
                   total_valid_instances.begin();
               it != total_valid_instances.end(); it++)
          {
            const FieldMask overlap = filter_mask & it->second;
            if (!overlap)
              continue;
            if (diff_expr == nullptr)
            {
              diff_expr = runtime->subtract_index_spaces(set_expr, expr);
              legion_assert(!diff_expr->is_empty());
            }
            if (record_partial_valid_instance(
                    it->first, diff_expr, overlap, false /*check total valid*/))
              need_partial_rebuild = true;
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(it->first);
            // Check if this is a collective view to record
            else if (
                it->first->is_collective_view() &&
                collective_instances.insert(
                    it->first->as_collective_view(), it->second))
              it->first->add_nested_resource_ref(did);
          }
          for (LogicalView* const it : to_delete)
          {
            if (view_refs_to_remove != nullptr)
            {
              std::map<LogicalView*, unsigned>::iterator finder =
                  view_refs_to_remove->find(it);
              if (finder == view_refs_to_remove->end())
                (*view_refs_to_remove)[it] = 1;
              else
                finder->second += 1;
            }
            else if (it->remove_nested_valid_ref(did))
              delete it;
            total_valid_instances.erase(it);
          }
          total_valid_instances.tighten_valid_mask();
          if (need_partial_rebuild)
          {
            partial_valid_fields.clear();
            for (ViewExprMaskSets::const_iterator it =
                     partial_valid_instances.begin();
                 it != partial_valid_instances.end(); it++)
              partial_valid_fields |= it->second.get_valid_mask();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_unrestricted_instances(
        IndexSpaceExpression* expr, const bool expr_covers,
        FieldMask filter_mask)
    //--------------------------------------------------------------------------
    {
      // Compute the expressions and field masks which are not restricted
      // First remove any fields which are restricted for this set
      ExprViewMaskSets::const_iterator finder = restricted_instances.find(expr);
      if (finder != restricted_instances.end())
      {
        filter_mask -= finder->second.get_valid_mask();
        if (!filter_mask)
          return;
      }
      // Next see if there any full set restrictions which dominate our expr
      if (expr != set_expr)
      {
        finder = restricted_instances.find(set_expr);
        if (finder != restricted_instances.end())
        {
          filter_mask -= finder->second.get_valid_mask();
          if (!filter_mask)
            return;
        }
      }
      // If we're still here, then we now have to do the hard part of
      // computing the intefering expression sets
      local::FieldMaskMap<IndexSpaceExpression> restricted_sets;
      for (const std::pair<
               IndexSpaceExpression* const, shrt::FieldMaskMap<InstanceView>>&
               it : restricted_instances)
      {
        const FieldMask overlap = it.second.get_valid_mask() & filter_mask;
        if (!overlap)
          continue;
        IndexSpaceExpression* expr_overlap =
            runtime->intersect_index_spaces(it.first, expr);
        if (expr_overlap->is_empty())
          continue;
        if (expr_overlap->get_volume() == expr->get_volume())
        {
          // If this expression dominates the expr we are done
          filter_mask -= overlap;
          if (!filter_mask)
            return;
        }
        restricted_sets.insert(expr_overlap, overlap);
      }
      legion_assert(!!filter_mask);
      // compute the field sets and take the field differences
      local::list<FieldSet<IndexSpaceExpression*>> field_sets;
      restricted_sets.compute_field_sets(filter_mask, field_sets);
      for (local::list<FieldSet<IndexSpaceExpression*>>::iterator it =
               field_sets.begin();
           it != field_sets.end(); it++)
      {
        if (it->elements.empty())
        {
          filter_valid_instances(expr, expr_covers, it->set_mask);
          continue;
        }
        IndexSpaceExpression* union_expr =
            runtime->union_index_spaces(it->elements);
        IndexSpaceExpression* diff_expr =
            runtime->subtract_index_spaces(expr, union_expr);
        if (!diff_expr->is_empty())
          filter_valid_instances(diff_expr, false /*covers*/, it->set_mask);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_reduction_instances(
        IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& filter_mask,
        std::map<IndexSpaceExpression*, unsigned>* expr_refs_to_remove,
        std::map<LogicalView*, unsigned>* view_refs_to_remove)
    //--------------------------------------------------------------------------
    {
      int fidx = filter_mask.find_first_set();
      while (fidx >= 0)
      {
        std::map<
            unsigned,
            std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>::
            iterator finder = reduction_instances.find(fidx);
        if (finder != reduction_instances.end())
        {
          if (expr_covers)
          {
            for (const std::pair<InstanceView*, IndexSpaceExpression*>& it :
                 finder->second)
            {
              if (view_refs_to_remove != nullptr)
              {
                std::map<LogicalView*, unsigned>::iterator finder =
                    view_refs_to_remove->find(it.first);
                if (finder == view_refs_to_remove->end())
                  (*view_refs_to_remove)[it.first] = 1;
                else
                  finder->second += 1;
              }
              else if (it.first->remove_nested_valid_ref(did))
                delete it.first;
              if (expr_refs_to_remove != nullptr)
              {
                std::map<IndexSpaceExpression*, unsigned>::iterator finder =
                    expr_refs_to_remove->find(it.second);
                if (finder == expr_refs_to_remove->end())
                  (*expr_refs_to_remove)[it.second] = 1;
                else
                  finder->second += 1;
              }
              else if (it.second->remove_nested_expression_reference(did))
                delete it.second;
            }
            reduction_instances.erase(finder);
            reduction_fields.unset_bit(fidx);
          }
          else
          {
            IndexSpaceExpression* full_diff = nullptr;
            for (std::list<std::pair<InstanceView*, IndexSpaceExpression*>>::
                     iterator it = finder->second.begin();
                 it != finder->second.end();
                 /*nothing*/)
            {
              if (it->second == set_expr)
              {
                if (full_diff == nullptr)
                {
                  full_diff = runtime->subtract_index_spaces(set_expr, expr);
                  legion_assert(!full_diff->is_empty());
                }
                full_diff->add_nested_expression_reference(did);
                if (expr_refs_to_remove != nullptr)
                {
                  std::map<IndexSpaceExpression*, unsigned>::iterator finder =
                      expr_refs_to_remove->find(it->second);
                  if (finder == expr_refs_to_remove->end())
                    (*expr_refs_to_remove)[it->second] = 1;
                  else
                    finder->second += 1;
                }
                else if (it->second->remove_nested_expression_reference(did))
                  delete it->second;
                it->second = full_diff;
                it++;
              }
              else
              {
                IndexSpaceExpression* diff_expr =
                    runtime->subtract_index_spaces(it->second, expr);
                if (!diff_expr->is_empty())
                {
                  if (diff_expr->get_volume() < it->second->get_volume())
                  {
                    diff_expr->add_nested_expression_reference(did);
                    if (expr_refs_to_remove != nullptr)
                    {
                      std::map<IndexSpaceExpression*, unsigned>::iterator
                          finder = expr_refs_to_remove->find(it->second);
                      if (finder == expr_refs_to_remove->end())
                        (*expr_refs_to_remove)[it->second] = 1;
                      else
                        finder->second += 1;
                    }
                    else if (it->second->remove_nested_expression_reference(
                                 did))
                      delete it->second;
                    it->second = diff_expr;
                  }
                  // Otherwise, no overlap so we keep going
                  it++;
                }
                else
                {
                  if (view_refs_to_remove != nullptr)
                  {
                    std::map<LogicalView*, unsigned>::iterator finder =
                        view_refs_to_remove->find(it->first);
                    if (finder == view_refs_to_remove->end())
                      (*view_refs_to_remove)[it->first] = 1;
                    else
                      finder->second += 1;
                  }
                  else if (it->first->remove_nested_valid_ref(did))
                    delete it->first;
                  if (expr_refs_to_remove != nullptr)
                  {
                    std::map<IndexSpaceExpression*, unsigned>::iterator finder =
                        expr_refs_to_remove->find(it->second);
                    if (finder == expr_refs_to_remove->end())
                      (*expr_refs_to_remove)[it->second] = 1;
                    else
                      finder->second += 1;
                  }
                  else if (it->second->remove_nested_expression_reference(did))
                    delete it->second;
                  it = finder->second.erase(it);
                }
              }
            }
            if (finder->second.empty())
            {
              reduction_instances.erase(finder);
              reduction_fields.unset_bit(fidx);
            }
          }
        }
        fidx = filter_mask.find_next_set(fidx + 1);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_set_internal(
        CopyFillAggregator*& input_aggregator, CopyFillGuard* previous_guard,
        PhysicalAnalysis* analysis, const RegionUsage& usage,
        IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& user_mask,
        const std::vector<PhysicalManager*>& target_instances,
        const VectorView<op::FieldMaskMap<InstanceView>>& target_views,
        const std::vector<IndividualView*>& source_views,
        const PhysicalTraceInfo& trace_info, const bool record_valid,
        const bool record_release)
    //--------------------------------------------------------------------------
    {
      // Read-write or read-only
      // Issue fills and or copies to bring the target instances up to date
      make_instances_valid(
          input_aggregator, previous_guard, analysis, false /*track*/, expr,
          expr_covers, user_mask, target_instances, target_views, source_views,
          trace_info);
      const bool is_write = IS_WRITE(usage);
      const FieldMask reduce_mask = reduction_fields & user_mask;
      const FieldMask restricted_mask = restricted_fields & user_mask;
      if (!!reduce_mask)
      {
        // Apply any reductions
        local::FieldMaskMap<IndexSpaceExpression> applied_reductions;
        apply_reductions(
            target_instances, target_views, expr, expr_covers, reduce_mask,
            input_aggregator, previous_guard, analysis, false /*track*/,
            trace_info, is_write ? nullptr : &applied_reductions);
        // If we're writing we're going to do an invalidation there anyway
        // so no need to bother with doing the invalidation based on the
        // reductions that have been applied
        if (!applied_reductions.empty())
        {
          legion_assert(!is_write);
          // See if covered the full expressions for invalidation
          local::FieldMaskMap<IndexSpaceExpression>::iterator finder =
              applied_reductions.find(expr);
          if (finder != applied_reductions.end())
          {
            if (!!restricted_mask)
            {
              const FieldMask overlap = finder->second & restricted_mask;
              if (!!overlap)
              {
                filter_unrestricted_instances(expr, expr_covers, overlap);
                finder.filter(overlap);
              }
            }
            if (!!finder->second)
              filter_valid_instances(expr, expr_covers, finder->second);
            // Remove the expression reference that flowed back
            if (finder->first->remove_nested_expression_reference(did))
              delete finder->first;
            applied_reductions.erase(finder);
          }
          if (!applied_reductions.empty())
          {
            // Handle the partial cases here
            local::list<FieldSet<IndexSpaceExpression*>> reduced_sets;
            applied_reductions.compute_field_sets(FieldMask(), reduced_sets);
            for (local::list<FieldSet<IndexSpaceExpression*>>::iterator it =
                     reduced_sets.begin();
                 it != reduced_sets.end(); it++)
            {
              IndexSpaceExpression* union_expr =
                  runtime->union_index_spaces(it->elements);
              const size_t union_size = union_expr->get_volume();
              const size_t set_size = set_expr->get_volume();
              legion_assert(union_size <= set_size);
              const bool union_covers = (union_size == set_size);
              if (!!restricted_mask)
              {
                const FieldMask overlap = it->set_mask & restricted_mask;
                if (!!overlap)
                {
                  filter_unrestricted_instances(
                      union_expr, union_covers, overlap);
                  it->set_mask -= overlap;
                }
              }
              if (!!it->set_mask)
                filter_valid_instances(union_expr, union_covers, it->set_mask);
            }
            // Remove expression references that flowed back
            for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                     applied_reductions.begin();
                 it != applied_reductions.end(); it++)
              if (it->first->remove_nested_expression_reference(did))
                delete it->first;
          }
        }
      }
      if (is_write)
      {
        if (record_release)
        {
          legion_assert(record_valid);
          legion_assert(restricted_mask == user_mask);
          // Releases are a bit strange, we actually want to invalidate
          // all the current valid instances since we're making them all
          // restricted so there are no partial unrestricted cases
          filter_valid_instances(expr, expr_covers, user_mask);
        }
        else if (!!restricted_mask)
        {
          const FieldMask non_restricted_mask = user_mask - restricted_mask;
          if (!!non_restricted_mask)
            filter_valid_instances(expr, expr_covers, non_restricted_mask);
          filter_unrestricted_instances(expr, expr_covers, restricted_mask);
        }
        else
          filter_valid_instances(expr, expr_covers, user_mask);
      }
      // Finally record the valid instances that have been updated
      if (record_valid)
      {
        if (!!restricted_mask)
        {
          const FieldMask non_restricted = user_mask - restricted_mask;
          if (!!non_restricted)
          {
            for (unsigned idx = 0; idx < target_views.size(); idx++)
            {
              const op::FieldMaskMap<InstanceView>& targets = target_views[idx];
              if (non_restricted * targets.get_valid_mask())
                continue;
              record_instances(
                  expr, expr_covers, non_restricted, FieldMapView(targets));
            }
          }
          for (unsigned idx = 0; idx < target_views.size(); idx++)
          {
            const op::FieldMaskMap<InstanceView>& targets = target_views[idx];
            if (restricted_mask * targets.get_valid_mask())
              continue;
            record_unrestricted_instances(
                expr, expr_covers, restricted_mask, FieldMapView(targets));
          }
        }
        else
        {
          for (unsigned idx = 0; idx < target_views.size(); idx++)
            record_instances(
                expr, expr_covers, user_mask, FieldMapView(target_views[idx]));
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::make_instances_valid(
        CopyFillAggregator*& aggregator, CopyFillGuard* previous_guard,
        PhysicalAnalysis* analysis, const bool track_events,
        IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& update_mask,
        const std::vector<PhysicalManager*>& target_instances,
        const VectorView<op::FieldMaskMap<InstanceView>>& target_views,
        const std::vector<IndividualView*>& source_views,
        const PhysicalTraceInfo& trace_info, const bool skip_valid_check,
        const ReductionOpID redop /*= 0*/,
        CopyAcrossHelper* across_helper /*=nullptr*/)
    //--------------------------------------------------------------------------
    {
      if (expr->is_empty())
        return;
      for (unsigned idx = 0; idx < target_views.size(); idx++)
      {
        if (target_views[idx].get_valid_mask() * update_mask)
          continue;
        for (op::FieldMaskMap<InstanceView>::const_iterator vit =
                 target_views[idx].begin();
             vit != target_views[idx].end(); vit++)
        {
          InstanceView* target = vit->first;
          FieldMask inst_mask = vit->second & update_mask;
          if (!inst_mask)
            continue;
          // If we can skip the check to see if the view is already valid
          if (!skip_valid_check)
          {
            // First check to see if the view is already marked as valid by name
            if (find_fully_valid_fields(target, inst_mask, expr, expr_covers))
              continue;
            // Next compute the partial valid expressions
            local::FieldMaskMap<IndexSpaceExpression> partial_valid_exprs;
            // If we're an individual view and there are potentially aliasing
            // collective views then we do an additional analysis here to see
            // if we are already valid for those other expression fields
            if (target->is_individual_view() && !collective_instances.empty() &&
                !(inst_mask * collective_instances.get_valid_mask()))
            {
              // We already know we're not valid by name here for these
              // field expressions, so just iterate the collective instances
              // and see if they contain the target and remove any
              // expressions and fields which are already valid
              const DistributedID inst_did =
                  target->as_individual_view()->get_manager()->did;
              for (lng::FieldMaskMap<CollectiveView>::const_iterator cit =
                       collective_instances.begin();
                   cit != collective_instances.end(); cit++)
              {
                // Check if it contains the instance
                if (!std::binary_search(
                        cit->first->instances.begin(),
                        cit->first->instances.end(), inst_did))
                  continue;
                // Check if it overlaps with the fields
                if (cit->second * inst_mask)
                  continue;
                // Now do the expression checks
                if (find_fully_valid_fields(
                        cit->first, inst_mask, expr, expr_covers))
                  break;
                if (find_partial_valid_fields(
                        cit->first, inst_mask, expr, expr_covers,
                        partial_valid_exprs))
                  break;
              }
              if (!inst_mask)
                continue;
            }
            // Do the check for any partial valid field expressions
            if (find_partial_valid_fields(
                    target, inst_mask, expr, expr_covers, partial_valid_exprs))
              continue;
            local::FieldMaskMap<IndexSpaceExpression> needed_exprs;
            if (!partial_valid_exprs.empty())
            {
              // Group expressions by fields since unions
              // and differences are expensive and hard to group later
              local::list<FieldSet<IndexSpaceExpression*>> expr_groups;
              partial_valid_exprs.compute_field_sets(FieldMask(), expr_groups);
              // Clear this in case we want to use it later
              partial_valid_exprs.clear();
              // Compute differences for each of the field groups
              for (local::list<FieldSet<IndexSpaceExpression*>>::const_iterator
                       it = expr_groups.begin();
                   it != expr_groups.end(); it++)
              {
                // No matter what we don't need to handle these fields
                // anymore for the full expression
                inst_mask -= it->set_mask;
                IndexSpaceExpression* valid_expr =
                    (it->elements.size() == 1) ?
                        *(it->elements.begin()) :
                        runtime->union_index_spaces(it->elements);
                IndexSpaceExpression* needed_expr =
                    runtime->subtract_index_spaces(expr, valid_expr);
                if (needed_expr->is_empty())
                  continue;
                needed_exprs.insert(needed_expr, it->set_mask);
              }
              if (needed_exprs.empty() && !inst_mask)
                continue;
            }
            // If the target is a collective view or there are collective
            // views that we might overlap with the target then we have to
            // do a much more sophisticated analysis to see if the data is
            // already valid by taking into account multiple views which
            // all name the the same instance(s)
            if (target->is_collective_view())
            {
              // Collective view target case
              CollectiveView* collective = target->as_collective_view();
              // Welcome to hell. We have three dimensions of validity we
              // need to check for here before we decide to issue copies:
              // 1. fields
              // 2. index space expressions
              // 3. instances (aliasing with individual and collective views)
              // Prepare for extreme complexity...
              if (!!inst_mask)
                needed_exprs.insert(expr_covers ? set_expr : expr, inst_mask);
              // Use an aliased instance analysis to do any dynamic
              // refinements as necessary for finding overlaps for
              // different sets of instances
              const FieldMapView needed_view(needed_exprs);
              MakeCollectiveValid alias_analysis(collective, needed_view);
              // See if we alias instances with any of the existing valid
              // views. The common case will be that we don't find any
              // and we'll be able to issue the updates freely.
              const FieldMask& needed_mask = needed_exprs.get_valid_mask();
              alias_analysis.traverse_total(
                  needed_mask, set_expr, total_valid_instances);
              if (!(needed_mask * partial_valid_fields))
                alias_analysis.traverse_partial(
                    needed_mask, partial_valid_instances);
              local::map<
                  InstanceView*, local::FieldMaskMap<IndexSpaceExpression>>
                  updates;
              alias_analysis.visit_leaves(
                  needed_mask, context, tree_id, updates);
              for (local::map<
                       InstanceView*,
                       local::FieldMaskMap<IndexSpaceExpression>>::
                       const_iterator uit = updates.begin();
                   uit != updates.end(); uit++)
                for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator
                         it = uit->second.begin();
                     it != uit->second.end(); it++)
                  issue_update_copies_and_fills(
                      uit->first, target_instances[idx], source_views,
                      aggregator, previous_guard, analysis, track_events,
                      it->first,
                      (it->first == set_expr) ? true : false /*expr covers*/,
                      it->second, trace_info, redop, across_helper);
              // Clear these since we've issued all our copies
              needed_exprs.clear();
              inst_mask.clear();
            }
            else
            {
              // At this point, any remaining needed_exprs are ones that
              // we can just issue the udpate copies and fills for from
              // the original target instance
              for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator
                       it = needed_exprs.begin();
                   it != needed_exprs.end(); it++)
                issue_update_copies_and_fills(
                    target, target_instances[idx], source_views, aggregator,
                    previous_guard, analysis, track_events, it->first,
                    false /*expr covers*/, it->second, trace_info, redop,
                    across_helper);
            }
          }
          // Whatever fields we have left here need updates for the whole expr
          if (!!inst_mask)
            issue_update_copies_and_fills(
                target, target_instances[idx], source_views, aggregator,
                previous_guard, analysis, track_events, expr, expr_covers,
                inst_mask, trace_info, redop, across_helper);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::find_fully_valid_fields(
        InstanceView* target, FieldMask& inst_mask, IndexSpaceExpression* expr,
        const bool expr_covers) const
    //--------------------------------------------------------------------------
    {
      shrt::FieldMaskMap<LogicalView>::const_iterator total_finder =
          total_valid_instances.find(target);
      if (total_finder != total_valid_instances.end())
      {
        inst_mask -= total_finder->second;
        if (!inst_mask)
          return true;
      }
      if (!expr_covers)
      {
        const FieldMask partial_overlap = partial_valid_fields & inst_mask;
        if (!!partial_overlap)
        {
          ViewExprMaskSets::const_iterator partial_finder =
              partial_valid_instances.find(target);
          if (partial_finder != partial_valid_instances.end())
          {
            shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator
                expr_finder = partial_finder->second.find(expr);
            if (expr_finder != partial_finder->second.end())
            {
              inst_mask -= expr_finder->second;
              if (!inst_mask)
                return true;
            }
          }
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::find_partial_valid_fields(
        InstanceView* target, FieldMask& inst_mask, IndexSpaceExpression* expr,
        const bool expr_covers,
        local::FieldMaskMap<IndexSpaceExpression>& partial_valid_exprs) const
    //--------------------------------------------------------------------------
    {
      ViewExprMaskSets::const_iterator partial_finder =
          partial_valid_instances.find(target);
      if (partial_finder != partial_valid_instances.end())
      {
        const FieldMask partial_valid =
            inst_mask & partial_finder->second.get_valid_mask();
        if (!!partial_valid)
        {
          for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator eit =
                   partial_finder->second.begin();
               eit != partial_finder->second.end(); eit++)
          {
            FieldMask overlap = eit->second & partial_valid;
            if (!overlap)
              continue;
            IndexSpaceExpression* expr_overlap;
            if (!expr_covers)
            {
              expr_overlap = runtime->intersect_index_spaces(expr, eit->first);
              const size_t expr_volume = expr_overlap->get_volume();
              if (expr_volume == 0)
                continue;
              if (expr_volume == expr->get_volume())
              {
                // expression dominates us so we are valid
                inst_mask -= overlap;
                if (!inst_mask)
                  return true;
              }
              else if (expr_volume == eit->first->get_volume())
                expr_overlap = eit->first;
            }
            else  // expr covers so we know it all intersects
              expr_overlap = eit->first;
            if (!(overlap * partial_valid_exprs.get_valid_mask()))
            {
              // If there are already some fields with expressions
              // (which can happen if this function is called more
              // than once for the same target), then we need to
              // merge expressions for any overlapping fields
              local::FieldMaskMap<IndexSpaceExpression> to_add;
              std::vector<IndexSpaceExpression*> to_delete;
              for (local::FieldMaskMap<IndexSpaceExpression>::iterator it =
                       partial_valid_exprs.begin();
                   it != partial_valid_exprs.end(); it++)
              {
                const FieldMask prev_overlap = overlap & it->second;
                if (!prev_overlap)
                  continue;
                IndexSpaceExpression* union_expr =
                    runtime->union_index_spaces(it->first, expr_overlap);
                to_add.insert(union_expr, prev_overlap);
                it.filter(prev_overlap);
                if (!it->second)
                  to_delete.emplace_back(it->first);
                overlap -= prev_overlap;
                if (!overlap)
                  break;
              }
              for (IndexSpaceExpression* it : to_delete)
                partial_valid_exprs.erase(it);
              for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator
                       it = to_add.begin();
                   it != to_add.end(); it++)
                partial_valid_exprs.insert(it->first, it->second);
              if (!overlap)
                return false;
            }
            partial_valid_exprs.insert(expr_overlap, overlap);
          }
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::issue_update_copies_and_fills(
        InstanceView* target, PhysicalManager* target_manager,
        const std::vector<IndividualView*>& source_views,
        CopyFillAggregator*& aggregator, CopyFillGuard* previous_guard,
        PhysicalAnalysis* analysis, const bool track_events,
        IndexSpaceExpression* expr, const bool expr_covers,
        FieldMask update_mask, const PhysicalTraceInfo& trace_info,
        const ReductionOpID redop, CopyAcrossHelper* across_helper)
    //--------------------------------------------------------------------------
    {
      // Before we do anything, if the user has provided an ordering of
      // source views, then go through and attempt to issue copies from
      // them before we do anything else
      if (!source_views.empty())
      {
        local::FieldMaskMap<IndexSpaceExpression> remainders;
        remainders.insert(expr, update_mask);
        for (IndividualView* src_it : source_views)
        {
          // Check to see if it is in the list of total valid instances
          shrt::FieldMaskMap<LogicalView>::const_iterator total_finder =
              total_valid_instances.find(src_it);
          if ((total_finder != total_valid_instances.end()) &&
              !(remainders.get_valid_mask() * total_finder->second))
          {
            std::vector<IndexSpaceExpression*> to_delete;
            for (local::FieldMaskMap<IndexSpaceExpression>::iterator it =
                     remainders.begin();
                 it != remainders.end(); it++)
            {
              const FieldMask overlap = it->second & total_finder->second;
              if (!overlap)
                continue;
              if (aggregator == nullptr)
                aggregator = new CopyFillAggregator(
                    analysis, previous_guard, track_events);
              aggregator->record_update(
                  target, target_manager, src_it, overlap, it->first,
                  trace_info, trace_info.recording ? this : nullptr, redop);
              it.filter(overlap);
              if (!it->second)
                to_delete.emplace_back(it->first);
            }
            if (!to_delete.empty())
            {
              for (IndexSpaceExpression* it : to_delete) remainders.erase(it);
              if (remainders.empty())
                return;
              remainders.tighten_valid_mask();
            }
          }
          // Next check to see if the instance has partial valid expressions
          ViewExprMaskSets::const_iterator partial_finder =
              partial_valid_instances.find(src_it);
          if ((partial_finder == partial_valid_instances.end()) ||
              (partial_finder->second.get_valid_mask() *
               remainders.get_valid_mask()))
            continue;
          // Compute the joins of the two field mask sets to get pairs of
          // index space expressions with the same fields
          local::map<
              std::pair<IndexSpaceExpression*, IndexSpaceExpression*>,
              FieldMask>
              join_expressions;
          unique_join_on_field_mask_sets(
              FieldMapView(remainders), FieldMapView(partial_finder->second),
              join_expressions);
          bool need_tighten = false;
          for (local::map<
                   std::pair<IndexSpaceExpression*, IndexSpaceExpression*>,
                   FieldMask>::const_iterator it = join_expressions.begin();
               it != join_expressions.end(); it++)
          {
            // Compute the intersection of the two index spaces
            IndexSpaceExpression* overlap = runtime->intersect_index_spaces(
                it->first.first, it->first.second);
            const size_t overlap_size = overlap->get_volume();
            if (overlap_size == 0)
              continue;
            local::FieldMaskMap<IndexSpaceExpression>::iterator finder =
                remainders.find(it->first.first);
            legion_assert(finder != remainders.end());
            finder.filter(it->second);
            if (!finder->second)
              remainders.erase(finder);
            if (aggregator == nullptr)
              aggregator = new CopyFillAggregator(
                  analysis, previous_guard, track_events);
            if (overlap_size < it->first.first->get_volume())
            {
              if (overlap_size == it->first.second->get_volume())
                aggregator->record_update(
                    target, target_manager, src_it, it->second,
                    it->first.second, trace_info,
                    trace_info.recording ? this : nullptr, redop);
              else
                aggregator->record_update(
                    target, target_manager, src_it, it->second, overlap,
                    trace_info, trace_info.recording ? this : nullptr, redop);
              // Compute the difference to add to the remainders
              IndexSpaceExpression* diff =
                  runtime->subtract_index_spaces(it->first.first, overlap);
              remainders.insert(diff, it->second);
            }
            else
            {
              // Covered the remainder expression
              aggregator->record_update(
                  target, target_manager, src_it, it->second, it->first.first,
                  trace_info, trace_info.recording ? this : nullptr, redop);
              if (remainders.empty())
                return;
              need_tighten = true;
            }
          }
          if (need_tighten)
            remainders.tighten_valid_mask();
        }
        legion_assert(!remainders.empty());
        // It's too hard to track all the pairs of partial sets for
        // both the source and destination instances at the same
        // time, so we recurse on this method for any expressions that
        // are not the same as the original expression except this
        // time we will not have any sources to consider
        for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                 remainders.begin();
             it != remainders.end(); it++)
        {
          if (it->first != expr)
          {
            const std::vector<IndividualView*> empty_sources;
            issue_update_copies_and_fills(
                target, target_manager, empty_sources, aggregator,
                previous_guard, analysis, track_events, it->first,
                false /*covers*/, it->second, trace_info, redop, across_helper);
          }
          else  // same expression so just keep the fields we need
            update_mask &= it->second;
        }
        // Fall through if we still have fields for this expression to handle
        if (!update_mask)
          return;
      }
      // We prefer bulk copies instead of lots of little copies, so do a quick
      // pass to see which fields we can find previous instances for that
      // completely cover our target without doing any intersection tests
      // If we find them then we'll just issue copies/fills from there,
      // otherwise we'll build partial sets and do the expensive thing
      const FieldMask total_fields =
          update_mask & total_valid_instances.get_valid_mask();
      if (!!total_fields)
      {
        if (aggregator == nullptr)
          aggregator =
              new CopyFillAggregator(analysis, previous_guard, track_events);
        if (total_fields != total_valid_instances.get_valid_mask())
        {
          // Compute selected instances that are valid for us
          local::FieldMaskMap<LogicalView> total_instances;
          for (shrt::FieldMaskMap<LogicalView>::const_iterator it =
                   total_valid_instances.begin();
               it != total_valid_instances.end(); it++)
          {
            const FieldMask overlap = it->second & update_mask;
            if (!overlap)
              continue;
            total_instances.insert(it->first, overlap);
          }
          aggregator->record_updates(
              target, target_manager, total_instances, total_fields, expr,
              trace_info, trace_info.recording ? this : nullptr, redop,
              across_helper);
        }
        else  // Total valid instances covers everything!
          aggregator->record_updates(
              target, target_manager, total_valid_instances, total_fields, expr,
              trace_info, trace_info.recording ? this : nullptr, redop,
              across_helper);
        update_mask -= total_fields;
        if (!update_mask)
          return;
      }
      // Now look through the partial valid instances for both instances
      // that cover us as well as partially valid instances
      local::FieldMaskMap<LogicalView> cover_instances;
      local::map<LogicalView*, local::FieldMaskMap<IndexSpaceExpression>>
          partial_instances;
      for (ViewExprMaskSets::const_iterator pit =
               partial_valid_instances.begin();
           pit != partial_valid_instances.end(); pit++)
      {
        if (pit->second.get_valid_mask() * update_mask)
          continue;
        for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                 pit->second.begin();
             it != pit->second.end(); it++)
        {
          FieldMask overlap = it->second & update_mask;
          if (!overlap)
            continue;
          if (!expr_covers)
          {
            if (it->first != expr)
            {
              IndexSpaceExpression* expr_overlap =
                  runtime->intersect_index_spaces(it->first, expr);
              const size_t overlap_volume = expr_overlap->get_volume();
              if (overlap_volume > 0)
              {
                if (overlap_volume < expr->get_volume())
                {
                  // partial overlap, only record this if we do not
                  // have any covering instances since we always prefer
                  // total coverings to partial ones
                  if (!cover_instances.empty())
                    overlap -= cover_instances.get_valid_mask();
                  if (!!overlap)
                  {
                    if (overlap_volume == it->first->get_volume())
                      partial_instances[pit->first].insert(it->first, overlap);
                    else
                      partial_instances[pit->first].insert(
                          expr_overlap, overlap);
                  }
                }
                else
                  cover_instances.insert(pit->first, overlap);
              }
            }
            else
              cover_instances.insert(pit->first, overlap);
          }
          else  // expr covers so everything is partial
            partial_instances[pit->first].insert(it->first, overlap);
        }
      }
      if (!cover_instances.empty())
      {
        if (aggregator == nullptr)
          aggregator =
              new CopyFillAggregator(analysis, previous_guard, track_events);
        aggregator->record_updates(
            target, target_manager, cover_instances,
            cover_instances.get_valid_mask(), expr, trace_info,
            trace_info.recording ? this : nullptr, redop, across_helper);
        update_mask -= cover_instances.get_valid_mask();
        if (!update_mask)
          return;
      }
      // This is a horrible place to be, partial updates everywhere
      // so now we need to ask the mapper which order to do them in
      // Ask the copy fll aggregator to help us out with this since
      // its probably queried the mapper about this all before
      if (!partial_instances.empty())
      {
        if (aggregator == nullptr)
          aggregator =
              new CopyFillAggregator(analysis, previous_guard, track_events);
        aggregator->record_partial_updates(
            target, target_manager, partial_instances, update_mask, expr,
            trace_info, trace_info.recording ? this : nullptr, redop,
            across_helper);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::record_reductions(
        UpdateAnalysis& analysis, IndexSpaceExpression* expr,
        const bool expr_covers, const FieldMask& user_mask)
    //--------------------------------------------------------------------------
    {
      CopyFillAggregator* fill_aggregator = nullptr;
      // See if we have an input aggregator that we can use now
      // for any fills that need to be done to initialize instances
      std::map<RtEvent, CopyFillAggregator*>::const_iterator finder =
          analysis.input_aggregators.find(RtEvent::NO_RT_EVENT);
      if (finder != analysis.input_aggregators.end())
        fill_aggregator = finder->second;
      FieldMask guard_fill_mask;
      for (unsigned idx = 0; idx < analysis.target_views.size(); idx++)
      {
        for (op::FieldMaskMap<InstanceView>::const_iterator rit =
                 analysis.target_views[idx].begin();
             rit != analysis.target_views[idx].end(); rit++)
        {
          const FieldMask reduction_mask = rit->second & user_mask;
          if (!reduction_mask)
            continue;
          reduction_fields |= reduction_mask;
          InstanceView* red_view = rit->first;
          const ReductionOpID view_redop = red_view->get_redop();
          FillView* fill_view = red_view->get_redop_fill_view();
          legion_assert(view_redop > 0);
          legion_assert(view_redop == analysis.usage.redop);
          legion_assert(
              partial_valid_instances.find(red_view) ==
              partial_valid_instances.end());
          if (red_view->is_collective_view())
          {
            // Collective reduction view path
            AllreduceView* allreduce_view = red_view->as_allreduce_view();
            InitializeCollectiveReduction alias_analysis(allreduce_view, expr);
            int fidx = reduction_mask.find_first_set();
            while (fidx >= 0)
            {
              FieldMask reduce_mask;
              reduce_mask.set_bit(fidx);
              std::list<std::pair<InstanceView*, IndexSpaceExpression*>>&
                  field_views = reduction_instances[fidx];
              for (std::pair<InstanceView*, IndexSpaceExpression*>& it :
                   field_views)
              {
                if (!allreduce_view->aliases(it.first))
                {
                  // Still need to check for the ABA problem
                  // (see below for a more detailed comment)
                  if (it.first->get_redop() == view_redop)
                    continue;
                  bool failure = false;
                  if (!expr_covers)
                  {
                    IndexSpaceExpression* overlap_expr =
                        runtime->intersect_index_spaces(expr, it.second);
                    if (overlap_expr->is_empty())
                      continue;
                    alias_analysis.visit_leaves(
                        reduce_mask, overlap_expr, failure);
                  }
                  else
                    alias_analysis.visit_leaves(
                        reduce_mask, it.second, failure);
                  if (failure)
                  {
                    // If we make it here, report the ABA violation
                    Fatal fatal;
                    fatal << "Unsafe re-use of reduction instance detected due "
                          << "to alternating un-flushed reduction operations "
                          << view_redop << " and " << it.first->get_redop()
                          << ". Please report this use case to the Legion "
                          << "developer's mailing list so that we can "
                          << "help you address it.";
                    fatal.raise();
                  }
                }
                else
                  alias_analysis.analyze(it.first, reduce_mask, it.second);
              }
              // Step to the next field
              fidx = reduction_mask.find_next_set(fidx + 1);
            }
            // Record any fill operations that need to be performed as a result
            // and update the reduction instances with new reductions
            alias_analysis.visit_leaves(
                reduction_mask, context, analysis, fill_aggregator, fill_view,
                tree_id, this, did, reduction_instances);
          }
          else
          {
            // Individual reduction view path
            // Track the case where this reduction view is also
            // stored in the valid instances in which case we
            // do not need to do any fills. This will only happen
            // if these fields are restricted because they are in
            // the total_valid_instances.
            bool already_valid = false;
            if (!!restricted_fields && !(reduction_mask * restricted_fields) &&
                (total_valid_instances.find(red_view) !=
                 total_valid_instances.end()))
              already_valid = true;
            int fidx = reduction_mask.find_first_set();
            // Figure out which fields require a fill operation
            // in order initialize the reduction instances
            local::FieldMaskMap<IndexSpaceExpression> fill_exprs;
            while (fidx >= 0)
            {
              std::list<std::pair<InstanceView*, IndexSpaceExpression*>>&
                  field_views = reduction_instances[fidx];
              // Scan through the reduction instances to see if we're
              // already in the list of valid reductions, if not then
              // we're going to need a fill to initialize the instance
              // Also check for the ABA problem on reduction instances
              // described in Legion issue #545 where we start out
              // with reductions of kind A, switch to reductions of
              // kind B, and then switch back to reductions of kind A
              // which will make it unsafe to re-use the instance
              const bool found_covered =
                  already_valid && total_valid_instances[red_view].is_set(fidx);
              // We only need to do this check if it's not already-covered
              // In the case where we know that it is already covered
              // at this point it is restricted, so everything is being
              // flushed to it anyway
              if (!found_covered)
              {
                // Scan backwards over the list of reduction instances for
                // this field looking for our reduction instance to see if
                // there are any parts that need to be initialized. While
                // doing this, we also have to check for the ABA problem
                // on reduction instances described in Legion issue #545
                // where we start out with reductions of kind A, switch
                // to reductions of kind B, and then switch back to
                // reductions of kind A which will make it unsafe to
                // re-use the instance
                IndexSpaceExpression* fill_expr = expr;
                // We'll try to merge the fill expr with an existing entry
                // if we can but we can only do that if it is not masked off
                // by different reduction operators
                bool fill_expr_merged = false;
                std::map<ReductionOpID, IndexSpaceExpression*> masked_exprs;
                for (std::list<
                         std::pair<InstanceView*, IndexSpaceExpression*>>::
                         reverse_iterator it = field_views.rbegin();
                     it != field_views.rend(); it++)
                {
                  if (red_view->aliases(it->first))
                  {
                    // Check for the ABA problem
                    for (const std::pair<
                             const ReductionOpID, IndexSpaceExpression*>& mit :
                         masked_exprs)
                    {
                      IndexSpaceExpression* overlap =
                          (it->second != set_expr) ?
                              runtime->intersect_index_spaces(
                                  mit.second, it->second) :
                              mit.second;
                      if (!overlap->is_empty())
                      {
                        // If we make it here, report the ABA violation
                        Fatal fatal;
                        fatal << "Unsafe re-use of reduction instance "
                              << "detected due to alternating un-flushed "
                              << "reduction operations " << mit.first << " and "
                              << view_redop << ". Please report "
                              << "this use case to the Legion developer's "
                              << "mailing list so that we can help you address "
                                 "it.";
                        fatal.raise();
                      }
                    }
                    if ((fill_expr == it->second) || (it->second == set_expr))
                    {
                      // We found ourself with an expression that covers the
                      // remainder needed to fill so we are done because we
                      // are already initialized for all of our points
                      fill_expr = nullptr;
                      break;
                    }
                    IndexSpaceExpression* overlap =
                        runtime->intersect_index_spaces(fill_expr, it->second);
                    if (!overlap->is_empty())
                    {
                      if (overlap->get_volume() == fill_expr->get_volume())
                      {
                        // We've initialized all our points so we're done
                        fill_expr = nullptr;
                        break;
                      }
                      if ((fill_expr == expr) && masked_exprs.empty())
                      {
                        // If the fill expr is still the original expression
                        // then that means we haven't recorded the reduction
                        // instance in the list yet so we need to do that now
                        // If the overlap covers the current expression then
                        // we can just use the fill expression as the new
                        // expression since it covers the current one. We know
                        // this is safe because of the check above confirming
                        // that we are disjoint with any prior masked exprs
                        IndexSpaceExpression* merged_expr =
                            (overlap->get_volume() ==
                             it->second->get_volume()) ?
                                fill_expr :
                                runtime->union_index_spaces(
                                    fill_expr, it->second);
                        if (it->second->remove_nested_expression_reference(did))
                          delete it->second;
                        merged_expr->add_nested_expression_reference(did);
                        it->second = merged_expr;
                        fill_expr_merged = true;
                      }
                      // Keep the remainder of the points to be initialized
                      // Need to keep iterating to check for the ABA problem
                      fill_expr =
                          runtime->subtract_index_spaces(fill_expr, overlap);
                      legion_assert(fill_expr != expr);
                    }
                  }
                  else if (it->first->get_redop() != view_redop)
                  {
                    // Look for masked expressions if this is different
                    // kind of reduction operator on the same field
                    IndexSpaceExpression* overlap =
                        (it->second != set_expr) ?
                            runtime->intersect_index_spaces(
                                fill_expr, it->second) :
                            fill_expr;
                    if (!overlap->is_empty())
                    {
                      ReductionOpID masked_redop = it->first->get_redop();
                      // Save this into the set of masked expressions
                      std::map<ReductionOpID, IndexSpaceExpression*>::iterator
                          finder = masked_exprs.find(masked_redop);
                      // If we already had an expression then merge it
                      if (finder != masked_exprs.end())
                        finder->second = runtime->union_index_spaces(
                            finder->second, overlap);
                      else
                        masked_exprs[masked_redop] = overlap;
                    }
                  }
                }
                // See if there are any fill expressions that we need to do
                // These are also the expressions that we need to add to the
                // fields views set since they won't be described by prior
                // reductions already on the list
                if (fill_expr != nullptr)
                {
                  FieldMask fill_mask;
                  fill_mask.set_bit(fidx);
                  fill_exprs.insert(fill_expr, fill_mask);
                  // If the fill_expr is still the expr then we didn't find any
                  // prior uses of this reduction instance so we need to ecord
                  // the reduction instance with its expression in the list.
                  if (!fill_expr_merged)
                  {
                    red_view->add_nested_valid_ref(did);
                    fill_expr->add_nested_expression_reference(did);
                    field_views.emplace_back(
                        std::make_pair(red_view, fill_expr));
                  }
                  if (fill_expr != expr)
                    // If we were previously initialized for any points then
                    // we need to record a guard mask on this field to make
                    // sure we pick up dependences on any fills
                    guard_fill_mask.set_bit(fidx);
                }
                else
                  guard_fill_mask.set_bit(fidx);
              }
              else
              {
                // This is already restricted, so just add it,
                // we'll be flushing it here shortly
                red_view->add_nested_valid_ref(did);
                expr->add_nested_expression_reference(did);
                field_views.emplace_back(std::make_pair(red_view, expr));
              }
              legion_assert(!field_views.empty());
              fidx = reduction_mask.find_next_set(fidx + 1);
            }
            if (!fill_exprs.empty())
            {
              if (fill_aggregator == nullptr)
              {
                // Fill aggregators never need to wait for any other
                // aggregators since we know they won't depend on each other
                fill_aggregator = new CopyFillAggregator(
                    &analysis, nullptr /*no previous guard*/,
                    false /*track events*/);
                analysis.input_aggregators[RtEvent::NO_RT_EVENT] =
                    fill_aggregator;
              }
              // Record the fill operation on the aggregator
              for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator
                       it = fill_exprs.begin();
                   it != fill_exprs.end(); it++)
                fill_aggregator->record_fill(
                    red_view, fill_view, it->second, it->first,
                    PredEvent::NO_PRED_EVENT,
                    analysis.trace_info.recording ? this : nullptr);
              // Record this as a guard for later operations
              reduction_fill_guards.insert(
                  fill_aggregator, fill_exprs.get_valid_mask());
              if (!fill_aggregator->record_guard_set(this, false /*read only*/))
                std::abort();
            }
          }
        }
      }
      // If we have any fills that were issued by a prior operation
      // that we need to reuse then check for them here. This is a
      // slight over-approximation for the mapping dependences because
      // we really only need to wait for fills to instances that we
      // care about it, but it should be minimal overhead and the
      // resulting event graph will still be precise
      if (!reduction_fill_guards.empty() && !!guard_fill_mask &&
          !(reduction_fill_guards.get_valid_mask() * guard_fill_mask))
      {
        for (shrt::FieldMaskMap<CopyFillGuard>::iterator it =
                 reduction_fill_guards.begin();
             it != reduction_fill_guards.end(); it++)
        {
          if (it->first == fill_aggregator)
            continue;
          const FieldMask guard_mask = guard_fill_mask & it->second;
          if (!guard_mask)
            continue;
          // No matter what record our dependences on the prior guards
#ifdef NON_AGGRESSIVE_AGGREGATORS
          analysis.guard_events.insert(it->first->effects_applied);
#else
          if (analysis.original_source == local_space)
            analysis.guard_events.insert(it->first->guard_postcondition);
          else
            analysis.guard_events.insert(it->first->effects_applied);
#endif
        }
      }
      // Record any reduction views here
      if (analysis.trace_info.recording)
      {
        for (unsigned idx = 0; idx < analysis.target_views.size(); idx++)
        {
          for (op::FieldMaskMap<InstanceView>::const_iterator it =
                   analysis.target_views[idx].begin();
               it != analysis.target_views[idx].end(); it++)
          {
            const FieldMask overlap = user_mask & it->second;
            if (!!overlap)
              update_tracing_reduced_view(it->first, expr, overlap);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::apply_reductions(
        const std::vector<PhysicalManager*>& target_instances,
        const VectorView<op::FieldMaskMap<InstanceView>>& target_views,
        IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& reduction_mask, CopyFillAggregator*& aggregator,
        CopyFillGuard* previous_guard, PhysicalAnalysis* analysis,
        const bool track_events, const PhysicalTraceInfo& trace_info,
        local::FieldMaskMap<IndexSpaceExpression>* applied_exprs,
        CopyAcrossHelper* across_helper /*= nullptr*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(!set_expr->is_empty());
      legion_assert(target_instances.size() == target_views.size());
      for (unsigned idx = 0; idx < target_views.size(); idx++)
      {
        for (op::FieldMaskMap<InstanceView>::const_iterator it =
                 target_views[idx].begin();
             it != target_views[idx].end(); it++)
        {
          const FieldMask& inst_mask = it->second & reduction_mask;
          if (!inst_mask)
            continue;
          apply_reduction(
              it->first, target_instances[idx], expr, expr_covers, inst_mask,
              aggregator, previous_guard, analysis, track_events, trace_info,
              applied_exprs, across_helper);
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::apply_restricted_reductions(
        const FieldMapView<InstanceView>& reduction_targets,
        IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& reduction_mask, CopyFillAggregator*& aggregator,
        CopyFillGuard* previous_guard, PhysicalAnalysis* analysis,
        const bool track_events, const PhysicalTraceInfo& trace_info,
        local::FieldMaskMap<IndexSpaceExpression>* applied_exprs)
    //--------------------------------------------------------------------------
    {
      legion_assert(!!reduction_targets.get_valid_mask());
      legion_assert(!set_expr->is_empty());
      for (FieldMapView<InstanceView>::const_iterator it =
               reduction_targets.begin();
           it != reduction_targets.end(); it++)
      {
        const FieldMask inst_mask = it->second & reduction_mask;
        if (!inst_mask)
          continue;
        apply_reduction(
            it->first, nullptr /*no manager since this is restricted*/, expr,
            expr_covers, inst_mask, aggregator, previous_guard, analysis,
            track_events, trace_info, applied_exprs, nullptr /*across*/);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::apply_reduction(
        InstanceView* target, PhysicalManager* target_manager,
        IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& reduction_mask, CopyFillAggregator*& aggregator,
        CopyFillGuard* previous_guard, PhysicalAnalysis* analysis,
        const bool track_events, const PhysicalTraceInfo& trace_info,
        local::FieldMaskMap<IndexSpaceExpression>* applied_exprs,
        CopyAcrossHelper* across_helper)
    //--------------------------------------------------------------------------
    {
      const bool track_exprs = (applied_exprs != nullptr);
      const bool target_is_reduction = target->is_reduction_kind();
      int fidx = reduction_mask.find_first_set();
      while (fidx >= 0)
      {
        std::map<
            unsigned,
            std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>::
            iterator finder = reduction_instances.find(fidx);
        legion_assert(finder != reduction_instances.end());
        legion_assert(!finder->second.empty());
        if (expr_covers)
        {
          // If the target is a reduction instance, check to see
          // that we at least have one reduction to apply
          if (target_is_reduction)
          {
            // Filter out all of our reductions
            for (std::list<std::pair<InstanceView*, IndexSpaceExpression*>>::
                     iterator it = finder->second.begin();
                 it != finder->second.end();
                 /*nothing*/)
            {
              if (it->first == target)
              {
                if (it->first->remove_nested_valid_ref(did))
                  delete it->first;
                if (it->second->remove_nested_expression_reference(did))
                  delete it->second;
                it = finder->second.erase(it);
              }
              else
                it++;
            }
            if (finder->second.empty())
            {
              // Quick out if there was nothing to apply
              reduction_instances.erase(finder);
              reduction_fields.unset_bit(fidx);
              fidx = reduction_mask.find_next_set(fidx + 1);
              continue;
            }
          }
          if (aggregator == nullptr)
            aggregator =
                new CopyFillAggregator(analysis, previous_guard, track_events);
          aggregator->record_reductions(
              target, target_manager, finder->second, fidx,
              (across_helper == nullptr) ?
                  fidx :
                  across_helper->convert_src_to_dst(fidx),
              trace_info.recording ? this : nullptr, across_helper);
          bool has_cover = false;
          for (const std::pair<InstanceView*, IndexSpaceExpression*>& it :
               finder->second)
          {
            if (it.second == set_expr)
              has_cover = true;
            if (it.first->remove_nested_valid_ref(did))
              delete it.first;
            // Only remove expression references here if we're not
            // tracking expressions
            if (!track_exprs &&
                it.second->remove_nested_expression_reference(did))
              delete it.second;
          }
          if (track_exprs)
          {
            // See if we find ourselves, if not just record all of them
            FieldMask expr_mask;
            expr_mask.set_bit(fidx);
            if (!has_cover)
            {
              // Expression references flow back but remove duplicates
              for (const std::pair<InstanceView*, IndexSpaceExpression*>& it :
                   finder->second)
                if (!applied_exprs->insert(it.second, expr_mask) &&
                    it.second->remove_nested_expression_reference(did))
                  std::abort();  // should never hit this
            }
            else
            {
              if (applied_exprs->insert(set_expr, expr_mask))
                set_expr->add_nested_expression_reference(did);
              // Now we can remove the remaining expression references
              for (const std::pair<InstanceView*, IndexSpaceExpression*>& it :
                   finder->second)
                if (it.second->remove_nested_expression_reference(did))
                  delete it.second;
            }
          }
          // We applied all these reductions so we're done
          reduction_instances.erase(finder);
          reduction_fields.unset_bit(fidx);
        }
        else
        {
          bool has_cover = false;
          std::vector<std::pair<InstanceView*, IndexSpaceExpression*>>
              to_delete;
          std::list<std::pair<InstanceView*, IndexSpaceExpression*>> to_record;
          // expr does not cover so we need intersection tests
          for (std::list<std::pair<InstanceView*, IndexSpaceExpression*>>::
                   iterator it = finder->second.begin();
               it != finder->second.end();
               /*nothing*/)
          {
            if (target_is_reduction && (it->first == target))
            {
              to_delete.emplace_back(*it);
              it = finder->second.erase(it);
            }
            else if (it->second == expr)
            {
              to_record.emplace_back(*it);
              to_delete.emplace_back(*it);
              if (track_exprs)
                has_cover = true;
              it = finder->second.erase(it);
            }
            else if (it->second == set_expr)
            {
              to_record.emplace_back(std::make_pair(it->first, expr));
              if (track_exprs)
                has_cover = true;
              IndexSpaceExpression* remainder =
                  runtime->subtract_index_spaces(set_expr, expr);
              remainder->add_nested_expression_reference(did);
              it->second = remainder;
              if (set_expr->remove_nested_expression_reference(did))
                std::abort();  // should never hit this
              it++;
            }
            else
            {
              IndexSpaceExpression* overlap =
                  runtime->intersect_index_spaces(expr, it->second);
              const size_t overlap_size = overlap->get_volume();
              if (overlap_size == 0)
              {
                it++;
                continue;
              }
              if (overlap_size == expr->get_volume())
              {
                to_record.emplace_back(std::make_pair(it->first, expr));
                if (track_exprs)
                  has_cover = true;
              }
              else
                to_record.emplace_back(std::make_pair(it->first, overlap));
              if (overlap_size == it->second->get_volume())
              {
                to_delete.emplace_back(*it);
                it = finder->second.erase(it);
              }
              else
              {
                IndexSpaceExpression* remainder =
                    runtime->subtract_index_spaces(it->second, expr);
                remainder->add_nested_expression_reference(did);
                if (it->second->remove_nested_expression_reference(did))
                  delete it->second;
                it->second = remainder;
                it++;
              }
            }
          }
          if (!to_record.empty())
          {
            if (aggregator == nullptr)
              aggregator = new CopyFillAggregator(
                  analysis, previous_guard, track_events);
            aggregator->record_reductions(
                target, target_manager, to_record, fidx,
                (across_helper == nullptr) ?
                    fidx :
                    across_helper->convert_src_to_dst(fidx),
                trace_info.recording ? this : nullptr, across_helper);
            if (track_exprs)
            {
              FieldMask expr_mask;
              expr_mask.set_bit(fidx);
              if (!has_cover)
              {
                for (const std::pair<InstanceView*, IndexSpaceExpression*>& it :
                     to_record)
                  if (applied_exprs->insert(it.second, expr_mask))
                    it.second->add_nested_expression_reference(did);
              }
              else if (applied_exprs->insert(expr, expr_mask))
                expr->add_nested_expression_reference(did);
            }
          }
          if (!to_delete.empty())
          {
            for (const std::pair<InstanceView*, IndexSpaceExpression*>& it :
                 to_delete)
            {
              if (it.first->remove_nested_valid_ref(did))
                delete it.first;
              if (it.second->remove_nested_expression_reference(did))
                delete it.second;
            }
          }
          if (finder->second.empty())
          {
            reduction_instances.erase(finder);
            reduction_fields.unset_bit(fidx);
          }
        }
        fidx = reduction_mask.find_next_set(fidx + 1);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::copy_out(
        IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& restricted_mask,
        const std::vector<PhysicalManager*>& src_insts,
        const VectorView<op::FieldMaskMap<InstanceView>>& src_views,
        PhysicalAnalysis* analysis, const PhysicalTraceInfo& trace_info,
        CopyFillAggregator*& aggregator)
    //--------------------------------------------------------------------------
    {
      legion_assert(src_insts.size() == src_views.size());
      for (unsigned idx = 0; idx < src_views.size(); idx++)
      {
        const op::FieldMaskMap<InstanceView>& sources = src_views[idx];
        const FieldMask overlap = sources.get_valid_mask() & restricted_mask;
        if (!overlap)
          continue;
        copy_out(
            expr, expr_covers, overlap, FieldMapView(sources), analysis,
            trace_info, aggregator);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void EquivalenceSet::copy_out(
        IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& restricted_mask, const FieldMapView<T>& src_insts,
        PhysicalAnalysis* analysis, const PhysicalTraceInfo& trace_info,
        CopyFillAggregator*& aggregator)
    //--------------------------------------------------------------------------
    {
      if (expr->is_empty())
        return;
      // Iterate through the restrictions looking for overlaps
      for (const std::pair<
               IndexSpaceExpression* const, shrt::FieldMaskMap<InstanceView>>&
               rit : restricted_instances)
      {
        const FieldMask overlap = rit.second.get_valid_mask() & restricted_mask;
        if (!overlap)
          continue;
        IndexSpaceExpression* overlap_expr = nullptr;
        if (expr_covers)
          overlap_expr = rit.first;
        else if (rit.first == set_expr)
          overlap_expr = expr;
        else
        {
          IndexSpaceExpression* over =
              runtime->intersect_index_spaces(rit.first, expr);
          if (over->is_empty())
            continue;
          const size_t over_size = over->get_volume();
          if (over_size == expr->get_volume())
            overlap_expr = expr;
          else if (over_size == rit.first->get_volume())
            overlap_expr = rit.first;
          else
            overlap_expr = over;
        }
        // Find the restricted destination instances for these fields
        local::map<std::pair<InstanceView*, T*>, FieldMask> restricted_copies;
        unique_join_on_field_mask_sets(
            FieldMapView(rit.second), FieldMapView(src_insts),
            restricted_copies);
        if (restricted_copies.empty())
          continue;
        for (typename local::map<
                 std::pair<InstanceView*, T*>, FieldMask>::const_iterator it =
                 restricted_copies.begin();
             it != restricted_copies.end(); it++)
        {
          if (it->first.first == it->first.second)
            continue;
          if (aggregator == nullptr)
            aggregator = new CopyFillAggregator(
                analysis, nullptr /*no previous guard*/, false /*track*/);
          aggregator->record_update(
              it->first.first, nullptr /*no manager*/, it->first.second,
              overlap, overlap_expr, trace_info,
              trace_info.recording ? this : nullptr);
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::acquire_restrictions(
        AcquireAnalysis& analysis, IndexSpaceExpression* expr,
        const bool expr_covers, const FieldMask& acquire_mask,
        std::set<RtEvent>& applied_events, const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // Already holding the eq_lock from EquivalenceSet::analyze method
      local::vector<IndexSpaceExpression*> to_delete;
      local::map<IndexSpaceExpression*, IndexSpaceExpression*> to_add;
      // Now we need to lock the analysis if we're going to do this traversal
      AutoLock a_lock(analysis);
      for (ExprViewMaskSets::iterator eit = restricted_instances.begin();
           eit != restricted_instances.end(); eit++)
      {
        FieldMask overlap = eit->second.get_valid_mask() & acquire_mask;
        if (!overlap)
          continue;
        IndexSpaceExpression* overlap_expr = nullptr;
        bool done_early = false;
        if (!expr_covers && (eit->first != expr))
        {
          overlap_expr = runtime->intersect_index_spaces(eit->first, expr);
          const size_t overlap_size = overlap_expr->get_volume();
          if (overlap_size == 0)
            continue;
          if (overlap_size == eit->first->get_volume())
          {
            overlap_expr = eit->first;
            if (overlap_size == expr->get_volume())
              done_early = (overlap == acquire_mask);
          }
          else if (overlap_size == expr->get_volume())
            overlap_expr = expr;
        }
        else
        {
          overlap_expr = eit->first;
          if (eit->first == expr)
            done_early = (overlap == acquire_mask);
        }
        ExprViewMaskSets::iterator release_finder =
            released_instances.find(overlap_expr);
        if (release_finder == released_instances.end())
        {
          overlap_expr->add_nested_expression_reference(did);
          released_instances[overlap_expr];
          release_finder = released_instances.find(overlap_expr);
        }
        if (overlap_expr == eit->first)
        {
          // Total covering of expressions
          // so remove instances no longer restricted
          if (overlap == eit->second.get_valid_mask())
          {
            // All instances are going to be released
            if (!release_finder->second.empty())
            {
              // Insert and remove duplicate references
              for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                       eit->second.begin();
                   it != eit->second.end(); it++)
              {
                analysis.record_instance(it->first, it->second);
                if (!release_finder->second.insert(it->first, it->second) &&
                    it->first->remove_nested_valid_ref(did))
                  std::abort();  // should never delete this
              }
              eit->second.clear();
            }
            else
            {
              for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                       eit->second.begin();
                   it != eit->second.end(); it++)
                analysis.record_instance(it->first, it->second);
              release_finder->second.swap(eit->second);
            }
            to_delete.emplace_back(eit->first);
          }
          else
          {
            // Filter instances whose fields overlap
            std::vector<InstanceView*> to_erase;
            for (shrt::FieldMaskMap<InstanceView>::iterator it =
                     eit->second.begin();
                 it != eit->second.end(); it++)
            {
              const FieldMask inst_overlap = overlap & it->second;
              if (!inst_overlap)
                continue;
              analysis.record_instance(it->first, inst_overlap);
              // Add it to the released instances
              if (release_finder->second.insert(it->first, inst_overlap))
                it->first->add_nested_valid_ref(did);
              // Remove it from here
              it.filter(inst_overlap);
              if (!it->second)
                to_erase.emplace_back(it->first);
              // Each field should only be represented by one instance
              overlap -= inst_overlap;
              if (!overlap)
                break;
            }
            for (InstanceView* it : to_erase)
            {
              eit->second.erase(it);
              if (it->remove_nested_valid_ref(did))
                delete it;
            }
            if (!eit->second.empty())
              eit->second.tighten_valid_mask();
            else
              to_delete.emplace_back(eit->first);
          }
        }
        else
        {
          // Only partial covering, so compute the difference
          // and record that we'll pull valid instances from here
          to_add[eit->first] = runtime->subtract_index_spaces(eit->first, expr);
          // The intersection gets merged back into relased sets
          for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                   eit->second.begin();
               it != eit->second.end(); it++)
          {
            const FieldMask inst_overlap = overlap & it->second;
            if (!inst_overlap)
              continue;
            analysis.record_instance(it->first, inst_overlap);
            if (release_finder->second.insert(it->first, it->second))
              it->first->add_nested_valid_ref(did);
            // Each field should only be represented by one instance
            overlap -= inst_overlap;
            if (!overlap)
              break;
          }
        }
        // If expressions matched and we handled all the fields then
        // we can be done since we know there are no other overlaps
        if (done_early)
          break;
      }
      for (const std::pair<IndexSpaceExpression* const, IndexSpaceExpression*>&
               eit : to_add)
      {
        if (restricted_instances.find(eit.second) == restricted_instances.end())
          eit.second->add_nested_expression_reference(did);
        shrt::FieldMaskMap<InstanceView>& old_insts =
            restricted_instances[eit.first];
        shrt::FieldMaskMap<InstanceView>& new_insts =
            restricted_instances[eit.second];
        if (!new_insts.empty() || !!(old_insts.get_valid_mask() & acquire_mask))
        {
          std::vector<InstanceView*> to_erase;
          for (shrt::FieldMaskMap<InstanceView>::iterator it =
                   old_insts.begin();
               it != old_insts.end(); it++)
          {
            const FieldMask overlap = it->second & acquire_mask;
            if (!overlap)
              continue;
            if (new_insts.insert(it->first, overlap))
              it->first->add_nested_valid_ref(did);
            it.filter(overlap);
            if (!it->second)
              to_erase.emplace_back(it->first);
          }
          for (InstanceView* const it : to_erase)
          {
            old_insts.erase(it);
            if (it->remove_nested_valid_ref(did))
              delete it;
          }
          if (old_insts.empty())
            to_delete.emplace_back(eit.first);
          else
            old_insts.tighten_valid_mask();
        }
        else
        {
          new_insts.swap(old_insts);
          to_delete.emplace_back(eit.first);
        }
      }
      for (local::vector<IndexSpaceExpression*>::const_iterator it =
               to_delete.begin();
           it != to_delete.end(); it++)
      {
        restricted_instances.erase(*it);
        if ((*it)->remove_nested_expression_reference(did))
          delete (*it);
      }
      restricted_fields.clear();
      for (const std::pair<
               IndexSpaceExpression* const, shrt::FieldMaskMap<InstanceView>>&
               it : restricted_instances)
        restricted_fields |= it.second.get_valid_mask();
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::release_restrictions(
        ReleaseAnalysis& analysis, IndexSpaceExpression* expr,
        const bool expr_covers, const FieldMask& release_mask,
        std::set<RtEvent>& applied_events, const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // Already holding the eq_lock from EquivalenceSet::analyze method
      // We need to lock the analysis at this point
      AutoLock a_lock(analysis);
      // If the target views are empty then we are just restoring the
      // existing released instances, if we have target views then we
      // know what the restricted instaces are going to be but we still
      // need to filter out any previously released instances
      if (analysis.target_views.empty())
      {
        local::map<IndexSpaceExpression*, local::FieldMaskMap<InstanceView>>
            to_update;
        local::vector<IndexSpaceExpression*> to_delete;
        local::map<IndexSpaceExpression*, IndexSpaceExpression*> to_add;
        for (ExprViewMaskSets::iterator eit = released_instances.begin();
             eit != released_instances.end(); eit++)
        {
          FieldMask overlap = eit->second.get_valid_mask() & release_mask;
          if (!overlap)
            continue;
          IndexSpaceExpression* overlap_expr = nullptr;
          if (!expr_covers && (eit->first != expr))
          {
            overlap_expr = runtime->intersect_index_spaces(eit->first, expr);
            const size_t overlap_size = overlap_expr->get_volume();
            if (overlap_size == 0)
              continue;
            if (overlap_size == eit->first->get_volume())
              overlap_expr = eit->first;
            else if (overlap_size == expr->get_volume())
              overlap_expr = expr;
          }
          else
            overlap_expr = eit->first;
          const bool overlap_covers =
              (overlap_expr->get_volume() == set_expr->get_volume());
          if (overlap_expr == eit->first)
          {
            // Total covering of expressions
            // so move all instances back to being restricted
            std::vector<InstanceView*> to_erase;
            local::FieldMaskMap<InstanceView>& updates = to_update[eit->first];
            for (shrt::FieldMaskMap<InstanceView>::iterator it =
                     eit->second.begin();
                 it != eit->second.end(); it++)
            {
              const FieldMask inst_overlap = overlap & it->second;
              if (!inst_overlap)
                continue;
              analysis.record_instance(it->first, inst_overlap);
              updates.insert(it->first, inst_overlap);
              // Record this as a restricted instance
              record_restriction(
                  overlap_expr, overlap_covers, inst_overlap, it->first);
              // Remove it from here
              it.filter(inst_overlap);
              if (!it->second)
                to_erase.emplace_back(it->first);
              // Each field should only be represented by one instance
              overlap -= inst_overlap;
              if (!overlap)
                break;
            }
            for (InstanceView* it : to_erase)
            {
              eit->second.erase(it);
              if (it->remove_nested_valid_ref(did))
                delete it;
            }
            if (!eit->second.empty())
              eit->second.tighten_valid_mask();
            else
              to_delete.emplace_back(eit->first);
          }
          else
          {
            // Only partial covering, so compute the difference
            // and record that we'll pull valid instances from here
            to_add[eit->first] =
                runtime->subtract_index_spaces(eit->first, expr);
            local::FieldMaskMap<InstanceView>& updates =
                to_update[overlap_expr];
            // The intersection gets merged back into relased sets
            for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                     eit->second.begin();
                 it != eit->second.end(); it++)
            {
              const FieldMask inst_overlap = overlap & it->second;
              if (!inst_overlap)
                continue;
              analysis.record_instance(it->first, inst_overlap);
              updates.insert(it->first, inst_overlap);
              // Record this as a restricted instance
              record_restriction(
                  overlap_expr, overlap_covers, inst_overlap, it->first);
              // Each field should only be represented by one instance
              overlap -= inst_overlap;
              if (!overlap)
                break;
            }
          }
        }
        // Record updates to the released sets
        for (const std::pair<
                 IndexSpaceExpression* const, IndexSpaceExpression*>& eit :
             to_add)
        {
          if (released_instances.find(eit.first) == released_instances.end())
            eit.first->add_nested_expression_reference(did);
          shrt::FieldMaskMap<InstanceView>& new_insts =
              released_instances[eit.first];
          shrt::FieldMaskMap<InstanceView>& old_insts =
              released_instances[eit.second];
          if (!new_insts.empty() ||
              !!(old_insts.get_valid_mask() & release_mask))
          {
            std::vector<InstanceView*> to_erase;
            for (shrt::FieldMaskMap<InstanceView>::iterator it =
                     old_insts.begin();
                 it != old_insts.end(); it++)
            {
              const FieldMask overlap = it->second & release_mask;
              if (!overlap)
                continue;
              if (new_insts.insert(it->first, overlap))
                it->first->add_nested_valid_ref(did);
              it.filter(overlap);
              if (!it->second)
                to_erase.emplace_back(it->first);
            }
            for (InstanceView* it : to_erase)
            {
              old_insts.erase(it);
              if (it->remove_nested_valid_ref(did))
                delete it;
            }
            if (old_insts.empty())
              to_delete.emplace_back(eit.first);
            else
              old_insts.tighten_valid_mask();
          }
          else
          {
            new_insts.swap(old_insts);
            to_delete.emplace_back(eit.first);
          }
        }
        for (local::vector<IndexSpaceExpression*>::const_iterator it =
                 to_delete.begin();
             it != to_delete.end(); it++)
        {
          released_instances.erase(*it);
          if ((*it)->remove_nested_expression_reference(did))
            delete (*it);
        }
        // Now generate the copies for any updates to the restricted instances
        if (analysis.release_aggregator != nullptr)
          analysis.release_aggregator->clear_update_fields();
        const RegionUsage release_usage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0);
        for (local::map<
                 IndexSpaceExpression*,
                 local::FieldMaskMap<InstanceView>>::const_iterator it =
                 to_update.begin();
             it != to_update.end(); it++)
        {
          // If we found all these views that means that they should all be
          // individual views since we can't have any restricted collective
          // views in non-control replicated settings and we can never be
          // here if we're in a control replicated context
          std::vector<PhysicalManager*> targets(it->second.size());
          local::vector<op::FieldMaskMap<InstanceView>> views(
              it->second.size());
          unsigned index = 0;
          for (local::FieldMaskMap<InstanceView>::const_iterator vit =
                   it->second.begin();
               vit != it->second.end(); vit++, index++)
          {
            legion_assert(vit->first->is_individual_view());
            IndividualView* view = vit->first->as_individual_view();
            targets[index] = view->get_manager();
            views[index].insert(vit->first, vit->second);
            update_set_internal(
                analysis.release_aggregator, nullptr /*no guard*/, &analysis,
                release_usage, it->first, (it->first == set_expr),
                it->second.get_valid_mask(), targets, views,
                analysis.source_views, analysis.trace_info,
                true /*record valid*/, true /*record release*/);
            // Finally update the tracing postconditions now that we've recorded
            // any copies as part of the trace
            if (tracing_postconditions != nullptr)
              tracing_postconditions->invalidate_all_but(
                  vit->first, it->first, vit->second);
          }
        }
      }
      else
      {
        // If we're not restoring the released instance then we should
        // record the actual instances that we are making restricted
        // Make sure that we don't have any overlapping restrictions
        filter_restricted_instances(expr, expr_covers, release_mask);
        // Make sure that we remove any old released instances
        filter_released_instances(expr, expr_covers, release_mask);
        for (unsigned idx = 0; idx < analysis.target_views.size(); idx++)
        {
          for (op::FieldMaskMap<InstanceView>::const_iterator it =
                   analysis.target_views[idx].begin();
               it != analysis.target_views[idx].end(); it++)
          {
            const FieldMask overlap = release_mask & it->second;
            if (!overlap)
              continue;
            // Record this as a restricted instance
            record_restriction(expr, expr_covers, overlap, it->first);
            // Update the tracing postconditions now that we've recorded
            // any copies as part of the trace
            if (tracing_postconditions != nullptr)
              tracing_postconditions->invalidate_all_but(
                  it->first, expr, overlap);
          }
        }
        // Now generate the copies for any updates to the restricted instances
        if (analysis.release_aggregator != nullptr)
          analysis.release_aggregator->clear_update_fields();
        const RegionUsage release_usage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0);
        update_set_internal(
            analysis.release_aggregator, nullptr /*no guard*/, &analysis,
            release_usage, expr, expr_covers, release_mask,
            analysis.target_instances, analysis.target_views,
            analysis.source_views, analysis.trace_info, true /*record valid*/,
            true /*record release*/);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::record_restriction(
        IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& restrict_mask, InstanceView* restrict_view)
    //--------------------------------------------------------------------------
    {
      // This function only looks for merges of restrictions. It assumes
      // that restrictions are inherently non-overlapping and does not
      // attempt to enforce that there are no such overlaps

      // First see if we need to merge with any existing restrictions
      if (expr_covers)
      {
        // No need to check for merging, we should be independent
        ExprViewMaskSets::iterator restricted_finder =
            restricted_instances.find(set_expr);
        if (restricted_finder == restricted_instances.end())
        {
          set_expr->add_nested_expression_reference(did);
          restrict_view->add_nested_valid_ref(did);
          restricted_instances[set_expr].insert(restrict_view, restrict_mask);
        }
        else if (restricted_finder->second.insert(restrict_view, restrict_mask))
          restrict_view->add_nested_valid_ref(did);
      }
      else
      {
        // Check to see if we can union this expression with any others
        local::FieldMaskMap<IndexSpaceExpression> to_union;
        local::vector<IndexSpaceExpression*> to_delete;
        for (ExprViewMaskSets::iterator eit = restricted_instances.begin();
             eit != restricted_instances.end(); eit++)
        {
          shrt::FieldMaskMap<InstanceView>::iterator finder =
              eit->second.find(restrict_view);
          if (finder == eit->second.end())
            continue;
          const FieldMask overlap = finder->second & restrict_mask;
          if (!overlap)
            continue;
          to_union.insert(eit->first, overlap);
          finder.filter(overlap);
          if (!finder->second)
          {
            if (finder->first->remove_nested_valid_ref(did))
              delete finder->first;
            eit->second.erase(finder);
            if (eit->second.empty())
              to_delete.emplace_back(eit->first);
            else
              eit->second.filter_valid_mask(overlap);
          }
          else
            eit->second.filter_valid_mask(overlap);
        }
        // Add in the new sets
        if (!to_union.empty())
        {
          local::list<FieldSet<IndexSpaceExpression*>> expr_sets;
          to_union.compute_field_sets(FieldMask(), expr_sets);
          for (local::list<FieldSet<IndexSpaceExpression*>>::iterator it =
                   expr_sets.begin();
               it != expr_sets.end(); it++)
          {
            it->elements.insert(expr);
            IndexSpaceExpression* union_expr =
                runtime->union_index_spaces(it->elements);
            if (union_expr->get_volume() < set_expr->get_volume())
            {
              ExprViewMaskSets::iterator restricted_finder =
                  restricted_instances.find(union_expr);
              if (restricted_finder == restricted_instances.end())
              {
                union_expr->add_nested_expression_reference(did);
                restrict_view->add_nested_valid_ref(did);
                restricted_instances[union_expr].insert(
                    restrict_view, it->set_mask);
              }
              else if (restricted_finder->second.insert(
                           restrict_view, it->set_mask))
                restrict_view->add_nested_valid_ref(did);
            }
            else
            {
              ExprViewMaskSets::iterator restricted_finder =
                  restricted_instances.find(set_expr);
              if (restricted_finder == restricted_instances.end())
              {
                set_expr->add_nested_expression_reference(did);
                restrict_view->add_nested_valid_ref(did);
                restricted_instances[set_expr].insert(
                    restrict_view, it->set_mask);
              }
              else if (restricted_finder->second.insert(
                           restrict_view, it->set_mask))
                restrict_view->add_nested_valid_ref(did);
            }
          }
          const FieldMask remainder = restrict_mask - to_union.get_valid_mask();
          if (!!remainder)
          {
            ExprViewMaskSets::iterator restricted_finder =
                restricted_instances.find(expr);
            if (restricted_finder == restricted_instances.end())
            {
              expr->add_nested_expression_reference(did);
              restrict_view->add_nested_valid_ref(did);
              restricted_instances[expr].insert(restrict_view, remainder);
            }
            else if (restricted_finder->second.insert(restrict_view, remainder))
              restrict_view->add_nested_valid_ref(did);
          }
        }
        else
        {
          // Just record ourselves since there was nothing to merge
          ExprViewMaskSets::iterator restricted_finder =
              restricted_instances.find(expr);
          if (restricted_finder == restricted_instances.end())
          {
            expr->add_nested_expression_reference(did);
            restrict_view->add_nested_valid_ref(did);
            restricted_instances[expr].insert(restrict_view, restrict_mask);
          }
          else if (restricted_finder->second.insert(
                       restrict_view, restrict_mask))
            restrict_view->add_nested_valid_ref(did);
        }
        for (local::vector<IndexSpaceExpression*>::const_iterator it =
                 to_delete.begin();
             it != to_delete.end(); it++)
        {
          restricted_instances.erase(*it);
          if ((*it)->remove_nested_expression_reference(did))
            delete (*it);
        }
      }
      legion_assert(!restricted_instances.empty());
      // Always update the restricted fields
      restricted_fields |= restrict_mask;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_reductions(
        const unsigned fidx,
        std::list<std::pair<InstanceView*, IndexSpaceExpression*>>& updates)
    //--------------------------------------------------------------------------
    {
      if (updates.empty())
        return;
      // Check for equivalence to the dst and then add our references
      const size_t volume = set_expr->get_volume();
      for (std::pair<InstanceView*, IndexSpaceExpression*>& it : updates)
      {
        it.first->add_nested_valid_ref(did);
        if (it.second->get_volume() == volume)
          it.second = set_expr;
        it.second->add_nested_expression_reference(did);
      }
      std::list<std::pair<InstanceView*, IndexSpaceExpression*>>& current =
          reduction_instances[fidx];
      current.splice(current.end(), updates);
      reduction_fields.set_bit(fidx);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_released(
        IndexSpaceExpression* expr, const bool expr_covers,
        shrt::FieldMaskMap<InstanceView>& updates)
    //--------------------------------------------------------------------------
    {
      if (expr->get_volume() == set_expr->get_volume())
        expr = set_expr;
      ExprViewMaskSets::iterator finder = released_instances.find(expr);
      if (finder != released_instances.end())
      {
        for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                 updates.begin();
             it != updates.end(); it++)
          if (finder->second.insert(it->first, it->second))
            it->first->add_nested_valid_ref(did);
      }
      else
      {
        expr->add_nested_expression_reference(did);
        for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                 updates.begin();
             it != updates.end(); it++)
          it->first->add_nested_valid_ref(did);
        released_instances[expr].swap(updates);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_initialized_data(
        IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& filter_mask,
        std::map<IndexSpaceExpression*, unsigned>* expr_refs_to_remove)
    //--------------------------------------------------------------------------
    {
      if (initialized_data.empty() ||
          (filter_mask * initialized_data.get_valid_mask()))
        return;
      if (expr_covers)
      {
        if (!(initialized_data.get_valid_mask() - filter_mask))
        {
          // filter everything
          for (lng::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   initialized_data.begin();
               it != initialized_data.end(); it++)
          {
            if (expr_refs_to_remove != nullptr)
            {
              std::map<IndexSpaceExpression*, unsigned>::iterator finder =
                  expr_refs_to_remove->find(it->first);
              if (finder == expr_refs_to_remove->end())
                (*expr_refs_to_remove)[it->first] = 1;
              else
                finder->second += 1;
            }
            else if (it->first->remove_nested_expression_reference(did))
              delete it->first;
          }
          initialized_data.clear();
        }
        else
        {
          // filter fields
          std::vector<IndexSpaceExpression*> to_delete;
          for (lng::FieldMaskMap<IndexSpaceExpression>::iterator it =
                   initialized_data.begin();
               it != initialized_data.end(); it++)
          {
            it.filter(filter_mask);
            if (!it->second)
              to_delete.emplace_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (IndexSpaceExpression* it : to_delete)
            {
              initialized_data.erase(it);
              if (expr_refs_to_remove != nullptr)
              {
                std::map<IndexSpaceExpression*, unsigned>::iterator finder =
                    expr_refs_to_remove->find(it);
                if (finder == expr_refs_to_remove->end())
                  (*expr_refs_to_remove)[it] = 1;
                else
                  finder->second += 1;
              }
              else if (it->remove_nested_expression_reference(did))
                delete it;
            }
          }
          if (!initialized_data.empty())
            initialized_data.tighten_valid_mask();
        }
      }
      else
      {
        // Filter on fields first and then on expressions
        std::vector<IndexSpaceExpression*> to_delete;
        local::FieldMaskMap<IndexSpaceExpression> to_add;
        for (lng::FieldMaskMap<IndexSpaceExpression>::iterator it =
                 initialized_data.begin();
             it != initialized_data.end(); it++)
        {
          const FieldMask overlap = filter_mask & it->second;
          if (!overlap)
            continue;
          if (it->first != set_expr)
          {
            IndexSpaceExpression* intersection =
                runtime->intersect_index_spaces(it->first, expr);
            const size_t volume = intersection->get_volume();
            if (volume == 0)
              continue;
            // We're removing this expression no matter what at this point
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(it->first);
            // See if there are any remaining points left
            if (volume < it->first->get_volume())
            {
              IndexSpaceExpression* diff =
                  runtime->subtract_index_spaces(it->first, intersection);
              to_add.insert(diff, overlap);
            }
          }
          else  // special case for when we know that the expr is the set expr
          {
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(it->first);
            to_add.insert(
                runtime->subtract_index_spaces(it->first, expr), overlap);
          }
        }
        if (!to_add.empty())
        {
          for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   to_add.begin();
               it != to_add.end(); it++)
            if (initialized_data.insert(it->first, it->second))
              it->first->add_nested_expression_reference(did);
        }
        if (!to_delete.empty())
        {
          for (IndexSpaceExpression* it : to_delete)
          {
            if (to_add.find(it) != to_add.end())
              continue;
            initialized_data.erase(it);
            if (expr_refs_to_remove != nullptr)
            {
              std::map<IndexSpaceExpression*, unsigned>::iterator finder =
                  expr_refs_to_remove->find(it);
              if (finder == expr_refs_to_remove->end())
                (*expr_refs_to_remove)[it] = 1;
              else
                finder->second += 1;
            }
            else if (it->remove_nested_expression_reference(did))
              delete it;
          }
        }
        if (!initialized_data.empty())
          initialized_data.tighten_valid_mask();
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_restricted_instances(
        IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& filter_mask,
        std::map<IndexSpaceExpression*, unsigned>* expr_refs_to_remove,
        std::map<LogicalView*, unsigned>* view_refs_to_remove)
    //--------------------------------------------------------------------------
    {
      if (restricted_instances.empty() || (filter_mask * restricted_fields))
        return;
      if (expr_covers)
      {
        if (!(restricted_fields - filter_mask))
        {
          // filter everything
          for (ExprViewMaskSets::const_iterator rit =
                   restricted_instances.begin();
               rit != restricted_instances.end(); rit++)
          {
            if (expr_refs_to_remove != nullptr)
            {
              std::map<IndexSpaceExpression*, unsigned>::iterator finder =
                  expr_refs_to_remove->find(rit->first);
              if (finder == expr_refs_to_remove->end())
                (*expr_refs_to_remove)[rit->first] = 1;
              else
                finder->second += 1;
            }
            else if (rit->first->remove_nested_expression_reference(did))
              delete rit->first;
            for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                     rit->second.begin();
                 it != rit->second.end(); it++)
            {
              if (view_refs_to_remove != nullptr)
              {
                std::map<LogicalView*, unsigned>::iterator finder =
                    view_refs_to_remove->find(it->first);
                if (finder == view_refs_to_remove->end())
                  (*view_refs_to_remove)[it->first] = 1;
                else
                  finder->second += 1;
              }
              else if (it->first->remove_nested_valid_ref(did))
                delete it->first;
            }
          }
          restricted_instances.clear();
          restricted_fields.clear();
        }
        else
        {
          // filter fields
          std::vector<IndexSpaceExpression*> to_delete;
          for (ExprViewMaskSets::iterator rit = restricted_instances.begin();
               rit != restricted_instances.end(); rit++)
          {
            if (!(rit->second.get_valid_mask() - filter_mask))
            {
              // delete all the views in this one
              for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                       rit->second.begin();
                   it != rit->second.end(); it++)
              {
                if (view_refs_to_remove != nullptr)
                {
                  std::map<LogicalView*, unsigned>::iterator finder =
                      view_refs_to_remove->find(it->first);
                  if (finder == view_refs_to_remove->end())
                    (*view_refs_to_remove)[it->first] = 1;
                  else
                    finder->second += 1;
                }
                else if (it->first->remove_nested_valid_ref(did))
                  delete it->first;
              }
              to_delete.emplace_back(rit->first);
            }
            else
            {
              // filter views based on fields
              std::vector<InstanceView*> to_erase;
              for (shrt::FieldMaskMap<InstanceView>::iterator it =
                       rit->second.begin();
                   it != rit->second.end(); it++)
              {
                it.filter(filter_mask);
                if (!it->second)
                  to_erase.emplace_back(it->first);
              }
              for (InstanceView* it : to_erase)
              {
                if (view_refs_to_remove != nullptr)
                {
                  std::map<LogicalView*, unsigned>::iterator finder =
                      view_refs_to_remove->find(it);
                  if (finder == view_refs_to_remove->end())
                    (*view_refs_to_remove)[it] = 1;
                  else
                    finder->second += 1;
                }
                else if (it->remove_nested_valid_ref(did))
                  delete it;
              }
              if (rit->second.empty())
                to_delete.emplace_back(rit->first);
              else
                rit->second.tighten_valid_mask();
            }
          }
          for (IndexSpaceExpression* it : to_delete)
          {
            restricted_instances.erase(it);
            if (expr_refs_to_remove != nullptr)
            {
              std::map<IndexSpaceExpression*, unsigned>::iterator finder =
                  expr_refs_to_remove->find(it);
              if (finder == expr_refs_to_remove->end())
                (*expr_refs_to_remove)[it] = 1;
              else
                finder->second += 1;
            }
            else if (it->remove_nested_expression_reference(did))
              delete it;
          }
          restricted_fields -= filter_mask;
        }
      }
      else
      {
        // Expression does not cover this equivalence set
        std::vector<IndexSpaceExpression*> to_delete;
        shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>
            to_add;
        for (ExprViewMaskSets::iterator rit = restricted_instances.begin();
             rit != restricted_instances.end(); rit++)
        {
          if (rit->second.get_valid_mask() * filter_mask)
            continue;
          IndexSpaceExpression* intersection =
              (rit->first == set_expr) ?
                  expr :
                  runtime->intersect_index_spaces(rit->first, expr);
          const size_t volume = intersection->get_volume();
          if (volume == 0)
            continue;
          if (volume == rit->first->get_volume())
          {
            // Covers the whole expression
            if (!(rit->second.get_valid_mask() - filter_mask))
            {
              // filter all of them
              for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                       rit->second.begin();
                   it != rit->second.end(); it++)
              {
                if (view_refs_to_remove != nullptr)
                {
                  std::map<LogicalView*, unsigned>::iterator finder =
                      view_refs_to_remove->find(it->first);
                  if (finder == view_refs_to_remove->end())
                    (*view_refs_to_remove)[it->first] = 1;
                  else
                    finder->second += 1;
                }
                else if (it->first->remove_nested_valid_ref(did))
                  delete it->first;
              }
              to_delete.emplace_back(rit->first);
            }
            else
            {
              // fitler by fields
              std::vector<InstanceView*> to_erase;
              for (shrt::FieldMaskMap<InstanceView>::iterator it =
                       rit->second.begin();
                   it != rit->second.end(); it++)
              {
                it.filter(filter_mask);
                if (!it->second)
                  to_erase.emplace_back(it->first);
              }
              for (InstanceView* it : to_erase)
              {
                rit->second.erase(it);
                if (view_refs_to_remove != nullptr)
                {
                  std::map<LogicalView*, unsigned>::iterator finder =
                      view_refs_to_remove->find(it);
                  if (finder == view_refs_to_remove->end())
                    (*view_refs_to_remove)[it] = 1;
                  else
                    finder->second += 1;
                }
                else if (it->remove_nested_valid_ref(did))
                  delete it;
              }
              if (rit->second.empty())
                to_delete.emplace_back(rit->first);
              else
                rit->second.tighten_valid_mask();
            }
          }
          else
          {
            // Only covers part, so compute diff and put them in the add set
            IndexSpaceExpression* diff =
                runtime->subtract_index_spaces(rit->first, intersection);
            if (!(rit->second.get_valid_mask() - filter_mask))
            {
              // All the views are flowing into to_add
              shrt::map<
                  IndexSpaceExpression*,
                  shrt::FieldMaskMap<InstanceView>>::iterator finder =
                  to_add.find(diff);
              if (finder != to_add.end())
              {
                // Deduplicate references in to add
                for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                         rit->second.begin();
                     it != rit->second.end(); it++)
                  if (!finder->second.insert(it->first, it->second) &&
                      it->first->remove_nested_valid_ref(did))
                    std::abort();  // should never hit this
              }
              else
                to_add[diff].swap(rit->second);
              to_delete.emplace_back(rit->first);
            }
            else
            {
              // Filter by fields
              shrt::FieldMaskMap<InstanceView>& add_set = to_add[diff];
              std::vector<InstanceView*> to_erase;
              for (shrt::FieldMaskMap<InstanceView>::iterator it =
                       rit->second.begin();
                   it != rit->second.end(); it++)
              {
                const FieldMask overlap = filter_mask & it->second;
                if (!overlap)
                  continue;
                if (add_set.insert(it->first, overlap))
                  it->first->add_nested_valid_ref(did);
                it.filter(overlap);
                if (!it->second)
                  to_erase.emplace_back(it->first);
              }
              for (InstanceView* it : to_erase)
              {
                rit->second.erase(it);
                if (view_refs_to_remove != nullptr)
                {
                  std::map<LogicalView*, unsigned>::iterator finder =
                      view_refs_to_remove->find(it);
                  if (finder == view_refs_to_remove->end())
                    (*view_refs_to_remove)[it] = 1;
                  else
                    finder->second += 1;
                }
                else if (it->remove_nested_valid_ref(did))
                  delete it;
              }
              if (rit->second.empty())
                to_delete.emplace_back(rit->first);
              else
                rit->second.tighten_valid_mask();
            }
          }
        }
        for (shrt::map<
                 IndexSpaceExpression*,
                 shrt::FieldMaskMap<InstanceView>>::iterator ait =
                 to_add.begin();
             ait != to_add.end(); ait++)
        {
          ExprViewMaskSets::iterator finder =
              restricted_instances.find(ait->first);
          if (finder == restricted_instances.end())
          {
            ait->first->add_nested_expression_reference(did);
            restricted_instances[ait->first].swap(ait->second);
          }
          else
          {
            for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                     ait->second.begin();
                 it != ait->second.end(); it++)
              // remove duplicate references
              if (!finder->second.insert(it->first, it->second) &&
                  it->first->remove_nested_valid_ref(did))
                std::abort();  // should never hit this
          }
        }
        for (IndexSpaceExpression* it : to_delete)
        {
          if (to_add.find(it) != to_add.end())
            continue;
          restricted_instances.erase(it);
          if (expr_refs_to_remove != nullptr)
          {
            std::map<IndexSpaceExpression*, unsigned>::iterator finder =
                expr_refs_to_remove->find(it);
            if (finder == expr_refs_to_remove->end())
              (*expr_refs_to_remove)[it] = 1;
            else
              finder->second += 1;
          }
          else if (it->remove_nested_expression_reference(did))
            delete it;
        }
        // Rebuild the restricted fields
        restricted_fields.clear();
        if (!restricted_instances.empty())
        {
          for (ExprViewMaskSets::const_iterator rit =
                   restricted_instances.begin();
               rit != restricted_instances.end(); rit++)
            restricted_fields |= rit->second.get_valid_mask();
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_released_instances(
        IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& filter_mask,
        std::map<IndexSpaceExpression*, unsigned>* expr_refs_to_remove,
        std::map<LogicalView*, unsigned>* view_refs_to_remove)
    //--------------------------------------------------------------------------
    {
      if (released_instances.empty())
        return;
      if (expr_covers)
      {
        // filter fields
        std::vector<IndexSpaceExpression*> to_delete;
        for (ExprViewMaskSets::iterator rit = released_instances.begin();
             rit != released_instances.end(); rit++)
        {
          if (!(rit->second.get_valid_mask() - filter_mask))
          {
            // delete all the views in this one
            for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                     rit->second.begin();
                 it != rit->second.end(); it++)
            {
              if (view_refs_to_remove != nullptr)
              {
                std::map<LogicalView*, unsigned>::iterator finder =
                    view_refs_to_remove->find(it->first);
                if (finder == view_refs_to_remove->end())
                  (*view_refs_to_remove)[it->first] = 1;
                else
                  finder->second += 1;
              }
              else if (it->first->remove_nested_valid_ref(did))
                delete it->first;
            }
            to_delete.emplace_back(rit->first);
          }
          else
          {
            // filter views based on fields
            std::vector<InstanceView*> to_erase;
            for (shrt::FieldMaskMap<InstanceView>::iterator it =
                     rit->second.begin();
                 it != rit->second.end(); it++)
            {
              it.filter(filter_mask);
              if (!it->second)
                to_erase.emplace_back(it->first);
            }
            for (InstanceView* it : to_erase)
            {
              if (view_refs_to_remove != nullptr)
              {
                std::map<LogicalView*, unsigned>::iterator finder =
                    view_refs_to_remove->find(it);
                if (finder == view_refs_to_remove->end())
                  (*view_refs_to_remove)[it] = 1;
                else
                  finder->second += 1;
              }
              else if (it->remove_nested_valid_ref(did))
                delete it;
            }
            if (rit->second.empty())
              to_delete.emplace_back(rit->first);
            else
              rit->second.tighten_valid_mask();
          }
        }
        for (IndexSpaceExpression* it : to_delete)
        {
          released_instances.erase(it);
          if (expr_refs_to_remove != nullptr)
          {
            std::map<IndexSpaceExpression*, unsigned>::iterator finder =
                expr_refs_to_remove->find(it);
            if (finder == expr_refs_to_remove->end())
              (*expr_refs_to_remove)[it] = 1;
            else
              finder->second += 1;
          }
          else if (it->remove_nested_expression_reference(did))
            delete it;
        }
      }
      else
      {
        // Expression does not cover this equivalence set
        std::vector<IndexSpaceExpression*> to_delete;
        shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>
            to_add;
        for (ExprViewMaskSets::iterator rit = released_instances.begin();
             rit != released_instances.end(); rit++)
        {
          if (rit->second.get_valid_mask() * filter_mask)
            continue;
          IndexSpaceExpression* intersection =
              (rit->first == set_expr) ?
                  expr :
                  runtime->intersect_index_spaces(rit->first, expr);
          const size_t volume = intersection->get_volume();
          if (volume == 0)
            continue;
          if (volume == rit->first->get_volume())
          {
            // Covers the whole expression
            if (!(rit->second.get_valid_mask() - filter_mask))
            {
              // filter all of them
              for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                       rit->second.begin();
                   it != rit->second.end(); it++)
              {
                if (view_refs_to_remove != nullptr)
                {
                  std::map<LogicalView*, unsigned>::iterator finder =
                      view_refs_to_remove->find(it->first);
                  if (finder == view_refs_to_remove->end())
                    (*view_refs_to_remove)[it->first] = 1;
                  else
                    finder->second += 1;
                }
                else if (it->first->remove_nested_valid_ref(did))
                  delete it->first;
              }
              to_delete.emplace_back(rit->first);
            }
            else
            {
              // fitler by fields
              std::vector<InstanceView*> to_erase;
              for (shrt::FieldMaskMap<InstanceView>::iterator it =
                       rit->second.begin();
                   it != rit->second.end(); it++)
              {
                it.filter(filter_mask);
                if (!it->second)
                  to_erase.emplace_back(it->first);
              }
              for (InstanceView* it : to_erase)
              {
                rit->second.erase(it);
                if (view_refs_to_remove != nullptr)
                {
                  std::map<LogicalView*, unsigned>::iterator finder =
                      view_refs_to_remove->find(it);
                  if (finder == view_refs_to_remove->end())
                    (*view_refs_to_remove)[it] = 1;
                  else
                    finder->second += 1;
                }
                else if (it->remove_nested_valid_ref(did))
                  delete it;
              }
              if (rit->second.empty())
                to_delete.emplace_back(rit->first);
              else
                rit->second.tighten_valid_mask();
            }
          }
          else
          {
            // Only covers part, so compute diff and put them in the add set
            IndexSpaceExpression* diff =
                runtime->subtract_index_spaces(rit->first, intersection);
            if (!(rit->second.get_valid_mask() - filter_mask))
            {
              // All the views are flowing into to_add
              shrt::map<
                  IndexSpaceExpression*,
                  shrt::FieldMaskMap<InstanceView>>::iterator finder =
                  to_add.find(diff);
              if (finder != to_add.end())
              {
                // Deduplicate references in to add
                for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                         rit->second.begin();
                     it != rit->second.end(); it++)
                  if (!finder->second.insert(it->first, it->second) &&
                      it->first->remove_nested_valid_ref(did))
                    std::abort();  // should never hit this
              }
              else
                to_add[diff].swap(rit->second);
              to_delete.emplace_back(rit->first);
            }
            else
            {
              // Filter by fields
              shrt::FieldMaskMap<InstanceView>& add_set = to_add[diff];
              std::vector<InstanceView*> to_erase;
              for (shrt::FieldMaskMap<InstanceView>::iterator it =
                       rit->second.begin();
                   it != rit->second.end(); it++)
              {
                const FieldMask overlap = filter_mask & it->second;
                if (!overlap)
                  continue;
                if (add_set.insert(it->first, overlap))
                  it->first->add_nested_valid_ref(did);
                it.filter(overlap);
                if (!it->second)
                  to_erase.emplace_back(it->first);
              }
              for (InstanceView* it : to_erase)
              {
                rit->second.erase(it);
                if (view_refs_to_remove != nullptr)
                {
                  std::map<LogicalView*, unsigned>::iterator finder =
                      view_refs_to_remove->find(it);
                  if (finder == view_refs_to_remove->end())
                    (*view_refs_to_remove)[it] = 1;
                  else
                    finder->second += 1;
                }
                else if (it->remove_nested_valid_ref(did))
                  delete it;
              }
              if (rit->second.empty())
                to_delete.emplace_back(rit->first);
              else
                rit->second.tighten_valid_mask();
            }
          }
        }
        for (shrt::map<
                 IndexSpaceExpression*,
                 shrt::FieldMaskMap<InstanceView>>::iterator ait =
                 to_add.begin();
             ait != to_add.end(); ait++)
        {
          ExprViewMaskSets::iterator finder =
              released_instances.find(ait->first);
          if (finder == released_instances.end())
          {
            ait->first->add_nested_expression_reference(did);
            released_instances[ait->first].swap(ait->second);
          }
          else
          {
            for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                     ait->second.begin();
                 it != ait->second.end(); it++)
              // remove duplicate references
              if (!finder->second.insert(it->first, it->second) &&
                  it->first->remove_nested_valid_ref(did))
                std::abort();  // should never hit this
          }
        }
        for (IndexSpaceExpression* it : to_delete)
        {
          if (to_add.find(it) != to_add.end())
            continue;
          released_instances.erase(it);
          if (expr_refs_to_remove != nullptr)
          {
            std::map<IndexSpaceExpression*, unsigned>::iterator finder =
                expr_refs_to_remove->find(it);
            if (finder == expr_refs_to_remove->end())
              (*expr_refs_to_remove)[it] = 1;
            else
              finder->second += 1;
          }
          else if (it->remove_nested_expression_reference(did))
            delete it;
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::issue_across_copies(
        CopyAcrossAnalysis& analysis, const FieldMask& src_mask,
        IndexSpaceExpression* expr, const bool expr_covers,
        std::set<RtEvent>& applied_events, const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // Already holding the eq_lock from EquivalenceSet::analyze method
      // While you might think this could a read-only lock since
      // we're just reading meta-data, that's not quite right because
      // we need exclusive access to data structures in check_for_migration
      // We need to lock the analysis at this point
      AutoLock a_lock(analysis);
      check_for_uninitialized_data(
          analysis, expr, expr_covers, src_mask, applied_events);
      // See if there are any other predicate guard fields that we need
      // to have as preconditions before applying our owner updates
      if (!read_only_guards.empty() &&
          !(src_mask * read_only_guards.get_valid_mask()))
      {
        for (shrt::FieldMaskMap<CopyFillGuard>::iterator it =
                 read_only_guards.begin();
             it != read_only_guards.end(); it++)
        {
          if (src_mask * it->second)
            continue;
          // No matter what record our dependences on the prior guards
#ifdef NON_AGGRESSIVE_AGGREGATORS
          analysis.guard_events.insert(it->first->effects_applied);
#else
          if (analysis.original_source == local_space)
            analysis.guard_events.insert(it->first->guard_postcondition);
          else
            analysis.guard_events.insert(it->first->effects_applied);
#endif
        }
      }
      // At this point we know we're going to need an aggregator since
      // this is an across copy and we have to be doing updates
      CopyFillAggregator* across_aggregator = analysis.get_across_aggregator();
      if (analysis.perfect)
      {
        make_instances_valid(
            across_aggregator, nullptr /*no guard*/, &analysis,
            true /*track events*/, expr, expr_covers, src_mask,
            analysis.target_instances, analysis.target_views,
            analysis.source_views, analysis.trace_info, true /*skip check*/,
            analysis.redop, analysis.across_helper);
        // Only need to check for reductions if we're not reducing since
        // the runtime prevents reductions-across with different reduction ops
        if ((analysis.redop == 0) && !!reduction_fields)
        {
          const FieldMask reduction_mask = src_mask & reduction_fields;
          if (!!reduction_mask)
            apply_reductions(
                analysis.target_instances, analysis.target_views, expr,
                expr_covers, reduction_mask, across_aggregator,
                nullptr /*no guard*/, &analysis, true /*track events*/,
                analysis.trace_info, nullptr /*no need to track applied exprs*/,
                analysis.across_helper);
        }
      }
      else
      {
        // Have to convert the targets views over to source fields
        op::vector<op::FieldMaskMap<InstanceView>> converted_target_views(
            analysis.target_views.size());
        for (unsigned idx = 0; idx < converted_target_views.size(); idx++)
        {
          for (op::FieldMaskMap<InstanceView>::const_iterator it =
                   analysis.target_views[idx].begin();
               it != analysis.target_views[idx].end(); it++)
          {
            const FieldMask converted =
                analysis.across_helper->convert_dst_to_src(it->second);
            converted_target_views[idx].insert(it->first, converted);
          }
        }
        make_instances_valid(
            across_aggregator, nullptr /*no guard*/, &analysis,
            true /*track events*/, expr, expr_covers, src_mask,
            analysis.target_instances, converted_target_views,
            analysis.source_views, analysis.trace_info, true /*skip check*/,
            analysis.redop, analysis.across_helper);
        // Only need to check for reductions if we're not reducing since
        // the runtime prevents reductions-across with different reduction ops
        if ((analysis.redop == 0) && !!reduction_fields)
        {
          const FieldMask reduction_mask = src_mask & reduction_fields;
          if (!!reduction_mask)
            apply_reductions(
                analysis.target_instances, converted_target_views, expr,
                expr_covers, reduction_mask, across_aggregator,
                nullptr /*no guard*/, &analysis, true /*track events*/,
                analysis.trace_info, nullptr /*no need to track applied exprs*/,
                analysis.across_helper);
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::overwrite_set(
        OverwriteAnalysis& analysis, IndexSpaceExpression* expr,
        const bool expr_covers, const FieldMask& overwrite_mask,
        std::set<RtEvent>& applied_events, const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // Already holding the eq_lock from EquivalenceSet::analyze method
      // Now that we're ready to perform the analysis
      // we need to lock the analysis
      AutoLock a_lock(analysis);
      if (analysis.output_aggregator != nullptr)
        analysis.output_aggregator->clear_update_fields();
      // Two different cases here depending on whether we have a precidate
      if (analysis.true_guard.exists())
      {
        // This case happens with a predicated fill operation
        // (No other overwrite analyses can be predicated)
        // We have two options of what we can do here:
        // 1. Generate a phi view so we can still use the base instances
        //    just with extra predicated fills done to them
        // 2. We can eagerly issue fills to all the valid instances for
        //    this expression masked by the predicate value
        // Currently we opt for option 2. While phi views still exist
        // in the code base they are currently not being used. They are
        // complicated and add nesting to the names of instance views
        // that would make them hard to reason about, so we prefer not
        // to use them if possible if we can avoid it. Hence we do the
        // second option and just issue predicated fill operations to
        // all the total and partially valid instances in the equivalence
        // set that overlap without needing to change the state of the
        // equivalence set.
        // There should be no restrictions added for predicated fills
        legion_assert(!analysis.add_restriction);
        legion_assert(!analysis.views.empty());
        legion_assert(analysis.reduction_views.empty());
        for (op::FieldMaskMap<LogicalView>::const_iterator it =
                 analysis.views.begin();
             it != analysis.views.end(); it++)
        {
          legion_assert(it->first->is_fill_view());
          const FieldMask overlap = it->second & overwrite_mask;
          if (!overlap)
            continue;
          FillView* fill = it->first->as_fill_view();
          predicate_fill_all(
              fill, overlap, expr, expr_covers, analysis.true_guard,
              analysis.false_guard, &analysis, analysis.trace_info,
              analysis.output_aggregator);
        }
      }
      else
      {
        // In all cases we're going to remove any reductions we've overwriting
        const FieldMask reduce_filter = reduction_fields & overwrite_mask;
        if (!!reduce_filter)
          filter_reduction_instances(expr, expr_covers, reduce_filter);
        if (analysis.add_restriction || !restricted_fields ||
            (restricted_fields * overwrite_mask))
        {
          // Easy case, just filter everything and add the new view
          filter_valid_instances(expr, expr_covers, overwrite_mask);
          if (!analysis.views.empty())
            record_instances(
                expr, expr_covers, overwrite_mask,
                FieldMapView(analysis.views));
        }
        else
        {
          // We overlap with some restricted fields so we can't filter
          // or update any restricted fields
          const FieldMask restricted_mask = overwrite_mask & restricted_fields;
          filter_unrestricted_instances(expr, expr_covers, restricted_mask);
          const FieldMask non_restricted = overwrite_mask - restricted_mask;
          if (!!non_restricted)
            filter_valid_instances(expr, expr_covers, non_restricted);
          if (!analysis.views.empty())
          {
            record_unrestricted_instances(
                expr, expr_covers, restricted_mask,
                FieldMapView(analysis.views));
            copy_out(
                expr, expr_covers, restricted_mask,
                FieldMapView(analysis.views), &analysis, analysis.trace_info,
                analysis.output_aggregator);
            if (!!non_restricted)
              record_instances(
                  expr, expr_covers, non_restricted,
                  FieldMapView(analysis.views));
          }
        }
        if (!analysis.reduction_views.empty())
        {
          for (op::FieldMaskMap<InstanceView>::const_iterator it =
                   analysis.reduction_views.begin();
               it != analysis.reduction_views.end(); it++)
          {
            int fidx = it->second.find_first_set();
            while (fidx >= 0)
            {
              reduction_instances[fidx].emplace_back(
                  std::make_pair(it->first, expr));
              it->first->add_nested_valid_ref(did);
              expr->add_nested_expression_reference(did);
              fidx = it->second.find_next_set(fidx + 1);
            }
          }
          reduction_fields |= analysis.reduction_views.get_valid_mask();
        }
        if (analysis.add_restriction)
        {
          legion_assert(analysis.views.size() == 1);
          op::FieldMaskMap<LogicalView>::const_iterator it =
              analysis.views.begin();
          LogicalView* log_view = it->first;
          legion_assert(log_view->is_instance_view());
          legion_assert(!(overwrite_mask - it->second));
          InstanceView* inst_view = log_view->as_instance_view();
          record_restriction(expr, expr_covers, overwrite_mask, inst_view);
        }
      }
      // Record that there is initialized data for this equivalence set
      update_initialized_data(expr, expr_covers, overwrite_mask);
      if (analysis.trace_info.recording)
      {
        legion_assert(analysis.reduction_views.empty());
        for (op::FieldMaskMap<LogicalView>::const_iterator it =
                 analysis.views.begin();
             it != analysis.views.end(); it++)
        {
          const FieldMask overlap = overwrite_mask & it->second;
          if (!!overlap)
            update_tracing_write_discard_view(it->first, expr, overlap);
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::predicate_fill_all(
        FillView* fill_view, const FieldMask& fill_mask,
        IndexSpaceExpression* expr, const bool expr_covers,
        const PredEvent true_guard, const PredEvent false_guard,
        PhysicalAnalysis* analysis, const PhysicalTraceInfo& trace_info,
        CopyFillAggregator*& aggregator)
    //--------------------------------------------------------------------------
    {
      // Importantly make sure we hold on to any references from deferred
      // views and expressions and until after we make the phi views and
      // have added the necessary references to them
      std::map<DeferredView*, unsigned> deferred_refs_to_remove;
      std::map<IndexSpaceExpression*, unsigned> expr_refs_to_remove;
      shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<DeferredView>>
          phi_views;
      // Do the partial valid views first
      if (!(fill_mask * partial_valid_fields))
      {
        std::vector<DeferredView*> to_delete;
        for (ViewExprMaskSets::iterator vit = partial_valid_instances.begin();
             vit != partial_valid_instances.end(); vit++)
        {
          if (fill_mask * vit->second.get_valid_mask())
            continue;
          if (vit->first->is_deferred_view())
          {
            DeferredView* deferred = vit->first->as_deferred_view();
            local::FieldMaskMap<IndexSpaceExpression> to_add;
            std::vector<IndexSpaceExpression*> to_remove;
            for (shrt::FieldMaskMap<IndexSpaceExpression>::iterator it =
                     vit->second.begin();
                 it != vit->second.end(); it++)
            {
              const FieldMask overlap = fill_mask & it->second;
              if (!overlap)
                continue;
              if (expr_covers)
              {
                phi_views[it->first].insert(deferred, overlap);
                it.filter(overlap);
                if (!it->second)
                  to_remove.emplace_back(it->first);
              }
              else
              {
                IndexSpaceExpression* fill_expr =
                    runtime->intersect_index_spaces(it->first, expr);
                if (fill_expr->is_empty())
                  continue;
                phi_views[fill_expr].insert(deferred, overlap);
                // remove the fields from this expression no matter what
                it.filter(overlap);
                if (!it->second)
                  to_remove.emplace_back(it->first);
                if (fill_expr->get_volume() < it->first->get_volume())
                {
                  // if we only had a partial covering then put the
                  // difference back into the partial expressions
                  IndexSpaceExpression* diff_expr =
                      runtime->subtract_index_spaces(it->first, fill_expr);
                  to_add.insert(diff_expr, overlap);
                }
              }
            }
            if (!to_remove.empty())
            {
              for (IndexSpaceExpression* it : to_remove)
              {
                if (!to_add.empty() && (to_add.find(it) != to_add.end()))
                  continue;
                vit->second.erase(it);
                std::map<IndexSpaceExpression*, unsigned>::iterator finder =
                    expr_refs_to_remove.find(it);
                if (finder == expr_refs_to_remove.end())
                  expr_refs_to_remove[it] = 1;
                else
                  finder->second++;
              }
            }
            if (!to_add.empty())
            {
              for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator
                       it = to_add.begin();
                   it != to_add.end(); it++)
                if (vit->second.insert(it->first, it->second))
                  it->first->add_nested_expression_reference(did);
            }
            if (vit->second.empty())
              to_delete.emplace_back(deferred);
            else
              vit->second.tighten_valid_mask();
          }
          else
          {
            InstanceView* inst_view = vit->first->as_instance_view();
            // Physical instance so we can just record the predicated fills
            for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                     vit->second.begin();
                 it != vit->second.end(); it++)
            {
              const FieldMask overlap = fill_mask & it->second;
              if (!overlap)
                continue;
              IndexSpaceExpression* fill_expr = it->first;
              if (!expr_covers)
              {
                fill_expr = runtime->intersect_index_spaces(it->first, expr);
                if (fill_expr->is_empty())
                  continue;
              }
              if (aggregator == nullptr)
                aggregator = new CopyFillAggregator(
                    analysis, nullptr /*no previous guard*/, false /*track*/,
                    true_guard);
              aggregator->record_fill(
                  inst_view, fill_view, overlap, fill_expr, true_guard, this);
            }
          }
        }
        if (!to_delete.empty())
        {
          for (DeferredView* it : to_delete)
          {
            partial_valid_instances.erase(it);
            std::map<DeferredView*, unsigned>::iterator finder =
                deferred_refs_to_remove.find(it);
            if (finder == deferred_refs_to_remove.end())
              deferred_refs_to_remove[it] = 1;
            else
              finder->second++;
          }
        }
        // No need to rebuild the partial valid fields here since we
        // didn't actually change them, we might have different sets of
        // views when we're done, but the same fields are all still represented
      }
      // Now do the total valid views
      bool need_partial_rebuild = false;
      if (!(fill_mask * total_valid_instances.get_valid_mask()))
      {
        std::vector<DeferredView*> to_delete;
        for (shrt::FieldMaskMap<LogicalView>::iterator it =
                 total_valid_instances.begin();
             it != total_valid_instances.end(); it++)
        {
          const FieldMask overlap = it->second & fill_mask;
          if (!overlap)
            continue;
          if (it->first->is_deferred_view())
          {
            // Record this in our set of phi views
            DeferredView* deferred = it->first->as_deferred_view();
            if (expr_covers)
              phi_views[set_expr].insert(deferred, overlap);
            else
              phi_views[expr].insert(deferred, overlap);
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(deferred);
            if (!expr_covers)
            {
              // If the predicated fill isn't covering then we need to
              // move this deferred view back to the partial valid views
              IndexSpaceExpression* diff =
                  runtime->subtract_index_spaces(set_expr, expr);
              if (record_partial_valid_instance(
                      it->first, diff, overlap, false /*check total_valid*/))
                need_partial_rebuild = true;
            }
          }
          else
          {
            // Physical instance so we can just record the predicated fill
            if (aggregator == nullptr)
              aggregator = new CopyFillAggregator(
                  analysis, nullptr /*no previous guard*/, false /*track*/,
                  true_guard);
            aggregator->record_fill(
                it->first->as_instance_view(), fill_view, overlap, expr,
                true_guard, this);
          }
        }
        if (!to_delete.empty())
        {
          for (DeferredView* it : to_delete)
          {
            total_valid_instances.erase(it);
            std::map<DeferredView*, unsigned>::iterator finder =
                deferred_refs_to_remove.find(it);
            if (finder == deferred_refs_to_remove.end())
              deferred_refs_to_remove[it] = 1;
            else
              finder->second++;
          }
        }
      }
      if (!phi_views.empty())
      {
        // Create the phi views and record them in the right data structure
        for (shrt::map<
                 IndexSpaceExpression*,
                 shrt::FieldMaskMap<DeferredView>>::iterator it =
                 phi_views.begin();
             it != phi_views.end(); it++)
        {
          const FieldMask phi_mask = it->second.get_valid_mask();
          shrt::FieldMaskMap<DeferredView> true_view;
          true_view.insert(fill_view, phi_mask);
          PhiView* phi_view = new PhiView(
              runtime->get_available_distributed_id(), true_guard, false_guard,
              std::move(true_view), std::move(it->second));
          if (it->first == set_expr)
          {
            total_valid_instances.insert(phi_view, phi_mask);
            phi_view->add_nested_valid_ref(did);
          }
          else if (record_partial_valid_instance(
                       phi_view, it->first, phi_mask,
                       false /*check total valid*/))
            need_partial_rebuild = true;
        }
      }
      // Now that we've made our phi views and registered them we
      // can remove the references that were being held
      if (!deferred_refs_to_remove.empty())
      {
        for (const std::pair<DeferredView* const, unsigned>& it :
             deferred_refs_to_remove)
          if (it.first->remove_nested_valid_ref(did, it.second))
            delete it.first;
      }
      if (!expr_refs_to_remove.empty())
      {
        for (const std::pair<IndexSpaceExpression* const, unsigned>& it :
             expr_refs_to_remove)
          if (it.first->remove_nested_expression_reference(did, it.second))
            delete it.first;
      }
      if (need_partial_rebuild)
      {
        partial_valid_fields.clear();
        for (ViewExprMaskSets::const_iterator it =
                 partial_valid_instances.begin();
             it != partial_valid_instances.end(); it++)
          partial_valid_fields |= it->second.get_valid_mask();
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_set(
        FilterAnalysis& analysis, IndexSpaceExpression* expr,
        const bool expr_covers, const FieldMask& filter_mask,
        std::set<RtEvent>& applied_events, const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // Already holding the eq_lock from EquivalenceSet::analyze method
      // No need to lock the analysis here since we're not going to change it
      if (analysis.filter_views.empty())
      {
        // This is the fast path for when we're just invalidating everything
        invalidate_state(expr, expr_covers, filter_mask, false /*record*/);
        return;
      }
      // We're filtering specific views from this set
      for (op::FieldMaskMap<InstanceView>::const_iterator fit =
               analysis.filter_views.begin();
           fit != analysis.filter_views.end(); fit++)
      {
        const FieldMask overlap = fit->second & filter_mask;
        if (!overlap)
          continue;
        filter_valid_instance(fit->first, expr, expr_covers, overlap);
        // Handle any aliasing with collective views
        if (fit->first->is_collective_view())
        {
          local::FieldMaskMap<InstanceView> to_filter;
          // Collective view case, need to check against all other views
          if (!(overlap * total_valid_instances.get_valid_mask()))
          {
            for (shrt::FieldMaskMap<LogicalView>::const_iterator it =
                     total_valid_instances.begin();
                 it != total_valid_instances.end(); it++)
            {
              if (!it->first->is_instance_view())
                continue;
              const FieldMask total_overlap = it->second & overlap;
              if (!total_overlap)
                continue;
              InstanceView* view = it->first->as_instance_view();
              if (!view->aliases(fit->first))
                continue;
              to_filter.insert(view, total_overlap);
            }
          }
          if (!(overlap * partial_valid_fields))
          {
            for (ViewExprMaskSets::const_iterator it =
                     partial_valid_instances.begin();
                 it != partial_valid_instances.end(); it++)
            {
              if (!it->first->is_instance_view())
                continue;
              const FieldMask partial_overlap =
                  it->second.get_valid_mask() & overlap;
              if (!partial_overlap)
                continue;
              InstanceView* view = it->first->as_instance_view();
              if (!view->aliases(fit->first))
                continue;
              to_filter.insert(view, partial_overlap);
            }
          }
          for (local::FieldMaskMap<InstanceView>::const_iterator it =
                   to_filter.begin();
               it != to_filter.end(); it++)
            filter_valid_instance(it->first, expr, expr_covers, it->second);
        }
        else if (
            !collective_instances.empty() &&
            !(collective_instances.get_valid_mask() * overlap))
        {
          // Individual view case, just check against the collective views
          for (lng::FieldMaskMap<CollectiveView>::const_iterator cit =
                   collective_instances.begin();
               cit != collective_instances.end(); cit++)
          {
            const FieldMask collective_overlap = cit->second & overlap;
            if (!collective_overlap)
              continue;
            if (!fit->first->aliases(cit->first))
              continue;
            filter_valid_instance(
                cit->first, expr, expr_covers, collective_overlap);
          }
        }
      }
      if (analysis.remove_restriction)
      {
        legion_assert(!analysis.filter_views.empty());
        // Note there is no need to check for aliasing of instance views
        // here (e.g. between collective and non-collective) since we
        // should always be releasing the same views that we acquired
        // in restricted mode at this point
        for (op::FieldMaskMap<InstanceView>::const_iterator fit =
                 analysis.filter_views.begin();
             fit != analysis.filter_views.end(); fit++)
        {
          const FieldMask inst_overlap = fit->second & filter_mask;
          if (!inst_overlap)
            continue;
          local::FieldMaskMap<IndexSpaceExpression> to_add;
          std::vector<IndexSpaceExpression*> to_delete;
          for (ExprViewMaskSets::iterator rit = restricted_instances.begin();
               rit != restricted_instances.end(); rit++)
          {
            shrt::FieldMaskMap<InstanceView>::iterator finder =
                rit->second.find(fit->first);
            if (finder == rit->second.end())
              continue;
            const FieldMask overlap = finder->second & inst_overlap;
            if (!overlap)
              continue;
            if (!expr_covers && (rit->first != expr))
            {
              IndexSpaceExpression* expr_overlap =
                  runtime->intersect_index_spaces(rit->first, expr);
              const size_t overlap_size = expr_overlap->get_volume();
              if (overlap_size == 0)
                continue;
              if (overlap_size < rit->first->get_volume())
              {
                // Did not cover all of it so we have to compute the diff
                IndexSpaceExpression* diff_expr =
                    runtime->subtract_index_spaces(rit->first, expr);
                legion_assert(diff_expr != nullptr);
                to_add.insert(diff_expr, overlap);
              }
            }
            // If we get here, then we're definitely removing this
            // restricted instances from this
            finder.filter(overlap);
            if (!finder->second)
            {
              if (finder->first->remove_nested_valid_ref(did))
                delete finder->first;
              rit->second.erase(finder);
              if (rit->second.empty())
                to_delete.emplace_back(rit->first);
              else
                rit->second.filter_valid_mask(overlap);
            }
            else
              rit->second.filter_valid_mask(overlap);
          }
          for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   to_add.begin();
               it != to_add.end(); it++)
          {
            ExprViewMaskSets::iterator finder =
                restricted_instances.find(it->first);
            if (finder == restricted_instances.end())
            {
              it->first->add_nested_expression_reference(did);
              fit->first->add_nested_valid_ref(did);
              restricted_instances[it->first].insert(fit->first, it->second);
            }
            else if (finder->second.insert(fit->first, it->second))
              fit->first->add_nested_valid_ref(did);
          }
          for (IndexSpaceExpression* it : to_delete)
          {
            ExprViewMaskSets::iterator finder = restricted_instances.find(it);
            legion_assert(finder != restricted_instances.end());
            // Check to see if it is still empty since we added things back
            if (!finder->second.empty())
              continue;
            restricted_instances.erase(finder);
            if (it->remove_nested_expression_reference(did))
              delete it;
          }
        }
        // Rebuild the restricted fields
        restricted_fields.clear();
        for (const std::pair<
                 IndexSpaceExpression* const, shrt::FieldMaskMap<InstanceView>>&
                 it : restricted_instances)
          restricted_fields |= it.second.get_valid_mask();
        // If the data was restricted then we just removed the only
        // valid copy so we need to filter the initialized data
        filter_initialized_data(expr, expr_covers, filter_mask);
      }
      else
      {
        // Check to see if we still have initialized data for what we filtered
        if (!total_valid_instances.empty() || !partial_valid_instances.empty())
        {
          FieldMask to_check =
              filter_mask - total_valid_instances.get_valid_mask();
          if (!!to_check)
          {
            const FieldMask no_partial = to_check - partial_valid_fields;
            if (!!no_partial)
            {
              filter_initialized_data(expr, expr_covers, no_partial);
              to_check -= no_partial;
            }
            if (!!to_check)
            {
              local::FieldMaskMap<IndexSpaceExpression> to_filter;
              to_filter.insert(expr, to_check);
              for (ViewExprMaskSets::const_iterator pit =
                       partial_valid_instances.begin();
                   pit != partial_valid_instances.end(); pit++)
              {
                if (to_check * pit->second.get_valid_mask())
                  continue;
                local::map<
                    std::pair<IndexSpaceExpression*, IndexSpaceExpression*>,
                    FieldMask>
                    filter_sets;
                unique_join_on_field_mask_sets(
                    FieldMapView(to_filter), FieldMapView(pit->second),
                    filter_sets);
                for (local::map<
                         std::pair<
                             IndexSpaceExpression*, IndexSpaceExpression*>,
                         FieldMask>::const_iterator it = filter_sets.begin();
                     it != filter_sets.end(); it++)
                {
                  IndexSpaceExpression* diff = runtime->subtract_index_spaces(
                      it->first.first, it->first.second);
                  if (diff->get_volume() == it->first.first->get_volume())
                    continue;
                  local::FieldMaskMap<IndexSpaceExpression>::iterator finder =
                      to_filter.find(it->first.first);
                  legion_assert(finder != to_filter.end());
                  finder.filter(it->second);
                  if (!finder->second)
                    to_filter.erase(finder);
                  if (!diff->is_empty())
                    to_filter.insert(diff, it->second);
                  else
                    to_check -= it->second;
                }
                if (to_filter.empty())
                  break;
              }
              if (!to_filter.empty())
              {
                for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator
                         it = to_filter.begin();
                     it != to_filter.end(); it++)
                {
                  const bool covers =
                      (it->first->get_volume() == set_expr->get_volume());
                  filter_initialized_data(it->first, covers, it->second);
                }
              }
            }
          }
        }
        else  // everything empty so filter the whole set
          filter_initialized_data(set_expr, true /*covers*/, filter_mask);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_valid_instance(
        InstanceView* to_filter, IndexSpaceExpression* expr,
        const bool expr_covers, const FieldMask& filter_mask)
    //--------------------------------------------------------------------------
    {
      ViewExprMaskSets::iterator part_finder =
          partial_valid_instances.find(to_filter);
      if (part_finder != partial_valid_instances.end())
      {
        FieldMask part_overlap =
            part_finder->second.get_valid_mask() & filter_mask;
        if (!!part_overlap)
        {
          local::FieldMaskMap<IndexSpaceExpression> to_add;
          std::vector<IndexSpaceExpression*> to_delete;
          for (shrt::FieldMaskMap<IndexSpaceExpression>::iterator it =
                   part_finder->second.begin();
               it != part_finder->second.end(); it++)
          {
            const FieldMask overlap = it->second & part_overlap;
            if (!overlap)
              continue;
            if (!expr_covers && (it->first != expr))
            {
              IndexSpaceExpression* expr_overlap =
                  runtime->intersect_index_spaces(it->first, expr);
              const size_t expr_size = expr_overlap->get_volume();
              if (expr_size == 0)
                continue;
              if (expr_size < it->first->get_volume())
              {
                IndexSpaceExpression* diff_expr =
                    runtime->subtract_index_spaces(it->first, expr_overlap);
                legion_assert(!diff_expr->is_empty());
                to_add.insert(diff_expr, overlap);
              }
            }
            // cover at least some if it so this expression will be removed
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(it->first);
            // Field overlaps should only occur once
            part_overlap -= overlap;
            if (!part_overlap)
              break;
          }
          for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   to_add.begin();
               it != to_add.end(); it++)
            if (part_finder->second.insert(it->first, it->second))
              it->first->add_nested_expression_reference(did);
          // Deletions after adds to keep references around
          if (!to_delete.empty())
          {
            for (IndexSpaceExpression* it : to_delete)
            {
              shrt::FieldMaskMap<IndexSpaceExpression>::iterator finder =
                  part_finder->second.find(it);
              legion_assert(finder != part_finder->second.end());
              if (!!finder->second)
                continue;
              part_finder->second.erase(it);
              if (it->remove_nested_expression_reference(did))
                delete it;
            }
          }
          if (part_finder->second.empty())
          {
            if (part_finder->first->remove_nested_valid_ref(did))
              delete part_finder->first;
            partial_valid_instances.erase(part_finder);
          }
          else
            part_finder->second.tighten_valid_mask();
          // Rebuild the partial valid fields
          partial_valid_fields.clear();
          if (!partial_valid_instances.empty())
          {
            for (ViewExprMaskSets::const_iterator it =
                     partial_valid_instances.begin();
                 it != partial_valid_instances.end(); it++)
              partial_valid_fields |= it->second.get_valid_mask();
          }
        }
      }
      shrt::FieldMaskMap<LogicalView>::iterator total_finder =
          total_valid_instances.find(to_filter);
      if (total_finder != total_valid_instances.end())
      {
        const FieldMask total_overlap = total_finder->second & filter_mask;
        if (!!total_overlap)
        {
          if (!expr_covers)
          {
            // Compute the difference and store it in the partial valid fields
            IndexSpaceExpression* diff_expr =
                runtime->subtract_index_spaces(set_expr, expr);
            legion_assert(!diff_expr->is_empty());
            if (record_partial_valid_instance(
                    to_filter, diff_expr, total_overlap,
                    false /*check total valid*/))
            {
              // Need to rebuild the partial valid fields
              partial_valid_fields.clear();
              for (ViewExprMaskSets::const_iterator it =
                       partial_valid_instances.begin();
                   it != partial_valid_instances.end(); it++)
                partial_valid_fields |= it->second.get_valid_mask();
            }
          }
          total_finder.filter(total_overlap);
          if (!total_finder->second)
          {
            if (total_finder->first->remove_nested_valid_ref(did))
              delete total_finder->first;
            total_valid_instances.erase(total_finder);
          }
          total_valid_instances.tighten_valid_mask();
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_tracing_read_only_view(
        InstanceView* view, IndexSpaceExpression* expr,
        const FieldMask& view_mask)
    //--------------------------------------------------------------------------
    {
      legion_assert(!view->is_reduction_kind());
      // Check to see if this instance has already been made by the trace
      local::map<LogicalView*, local::FieldMaskMap<IndexSpaceExpression>>
          not_dominated;
      if (tracing_postconditions != nullptr)
        tracing_postconditions->dominates(view, expr, view_mask, not_dominated);
      else
        not_dominated[view].insert(expr, view_mask);
      // Record everything not dominated as a precondition
      if (!not_dominated.empty())
      {
        if (tracing_preconditions == nullptr)
          tracing_preconditions =
              new TraceViewSet(context, did, set_expr, tree_id);
        tracing_preconditions->insert(not_dominated);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_tracing_write_discard_view(
        LogicalView* view, IndexSpaceExpression* expr,
        const FieldMask& view_mask)
    //--------------------------------------------------------------------------
    {
      if (tracing_postconditions != nullptr)
        tracing_postconditions->invalidate_all_but(view, expr, view_mask);
      else
        tracing_postconditions =
            new TraceViewSet(context, did, set_expr, tree_id);
      tracing_postconditions->insert(view, expr, view_mask);
      if (tracing_dirty_fields == nullptr)
        tracing_dirty_fields = new shrt::FieldMaskMap<IndexSpaceExpression>();
      if (tracing_dirty_fields->insert(expr, view_mask))
        expr->add_nested_expression_reference(did);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_tracing_read_write_view(
        InstanceView* view, IndexSpaceExpression* expr,
        const FieldMask& view_mask)
    //--------------------------------------------------------------------------
    {
      update_tracing_read_only_view(view, expr, view_mask);
      update_tracing_write_discard_view(view, expr, view_mask);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_tracing_reduced_view(
        InstanceView* view, IndexSpaceExpression* expr,
        const FieldMask& view_mask)
    //--------------------------------------------------------------------------
    {
      legion_assert(view->is_reduction_kind());
      // Check to see if we initialized the view in this trace
      local::map<LogicalView*, local::FieldMaskMap<IndexSpaceExpression>>
          not_dominated;
      if (tracing_anticonditions != nullptr)
        tracing_anticonditions->dominates(view, expr, view_mask, not_dominated);
      else
        not_dominated[view].insert(expr, view_mask);
      // Record everything not dominated as a precondition
      if (!not_dominated.empty())
      {
        if (tracing_preconditions == nullptr)
          tracing_preconditions =
              new TraceViewSet(context, did, set_expr, tree_id);
        tracing_preconditions->insert(not_dominated);
      }
      // Then record this in the postconditions
      // Note that you can keep all the non-reduction views in the
      // postconditions here since we still need them as postcondition in
      // case we don't end up applying the reductions before the end of
      // capturing the trace
      if (tracing_postconditions == nullptr)
        tracing_postconditions =
            new TraceViewSet(context, did, set_expr, tree_id);
      tracing_postconditions->insert(view, expr, view_mask);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_tracing_fill_views(
        FillView* src_view, InstanceView* dst_view, IndexSpaceExpression* expr,
        const FieldMask& view_mask, bool across)
    //--------------------------------------------------------------------------
    {
      if (dst_view->is_reduction_kind())
      {
        legion_assert(!across);
        // Initializing a reduction view
        // Here, we have to do the converse of the anticondition
        // check in update_tracing_valid_views. In particular, if the
        // trace has already read this particular equivalence set, then
        // we don't need to add the equivalence set to the anti-conditions,
        // as the reduction data has already been consumed, and can be read
        // out from the resulting instance.
        local::map<LogicalView*, local::FieldMaskMap<IndexSpaceExpression>>
            not_dominated;
        if (tracing_preconditions != nullptr)
          tracing_preconditions->dominates(
              dst_view, expr, view_mask, not_dominated);
        else
          not_dominated[dst_view].insert(expr, view_mask);
        if (!not_dominated.empty())
        {
          if (tracing_anticonditions == nullptr)
            tracing_anticonditions =
                new TraceViewSet(context, did, set_expr, tree_id);
          tracing_anticonditions->insert(not_dominated);
        }
      }
      else  // this is the same as the copy case
        update_tracing_copy_views(src_view, dst_view, expr, view_mask, across);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_tracing_copy_views(
        LogicalView* src_view, InstanceView* dst_view,
        IndexSpaceExpression* expr, const FieldMask& view_mask, bool across)
    //--------------------------------------------------------------------------
    {
      legion_assert(!dst_view->is_reduction_kind());
      // record src view in the preconditions if not dominated by post
      local::map<LogicalView*, local::FieldMaskMap<IndexSpaceExpression>>
          not_dominated;
      if (tracing_postconditions != nullptr)
        tracing_postconditions->dominates(
            src_view, expr, view_mask, not_dominated);
      else
        not_dominated[src_view].insert(expr, view_mask);
      if (!not_dominated.empty())
      {
        if (tracing_preconditions == nullptr)
          tracing_preconditions =
              new TraceViewSet(context, did, set_expr, tree_id);
        tracing_preconditions->insert(not_dominated);
      }
      // record the destination view unless this is an across copy
      if (across)
        return;
      if (tracing_postconditions == nullptr)
        tracing_postconditions =
            new TraceViewSet(context, did, set_expr, tree_id);
      tracing_postconditions->insert(dst_view, expr, view_mask);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_tracing_reduction_views(
        InstanceView* src_view, InstanceView* dst_view,
        IndexSpaceExpression* expr, const FieldMask& view_mask, bool across)
    //--------------------------------------------------------------------------
    {
      legion_assert(src_view->is_reduction_kind());
      // Check to see if we made the reduction instance in the trace
      local::map<LogicalView*, local::FieldMaskMap<IndexSpaceExpression>>
          not_dominated;
      if (tracing_anticonditions != nullptr)
        tracing_anticonditions->dominates(
            src_view, expr, view_mask, not_dominated);
      else
        not_dominated[src_view].insert(expr, view_mask);
      if (!not_dominated.empty())
      {
        if (tracing_preconditions == nullptr)
          tracing_preconditions =
              new TraceViewSet(context, did, set_expr, tree_id);
        tracing_preconditions->insert(not_dominated);
        not_dominated.clear();
      }
      // Also need to check to see if the destination was produced in the
      // trace or whether we need to record it as a precondition as well
      // since we're applying reductions to it
      // If this is an across copy though then we don't need to do this
      if (across)
        return;
      if (dst_view->is_reduction_kind())
      {
        if (tracing_anticonditions != nullptr)
          tracing_anticonditions->dominates(
              dst_view, expr, view_mask, not_dominated);
        else
          not_dominated[dst_view].insert(expr, view_mask);
      }
      else
      {
        if (tracing_postconditions != nullptr)
          tracing_postconditions->dominates(
              dst_view, expr, view_mask, not_dominated);
        else
          not_dominated[dst_view].insert(expr, view_mask);
      }
      if (!not_dominated.empty())
      {
        if (tracing_preconditions == nullptr)
          tracing_preconditions =
              new TraceViewSet(context, did, set_expr, tree_id);
        tracing_preconditions->insert(not_dominated);
      }
      if (tracing_postconditions != nullptr)
      {
        // This is only safe because Legion will never do partial reductions
        // of reduction views. If we're folding reduction views then we're
        // going to end up going all the way down to a normal instance
        if (dst_view->is_reduction_kind())
          tracing_postconditions->invalidate(src_view, expr, view_mask);
        else
          tracing_postconditions->invalidate_all_but(dst_view, expr, view_mask);
      }
      else
        tracing_postconditions =
            new TraceViewSet(context, did, set_expr, tree_id);
      tracing_postconditions->insert(dst_view, expr, view_mask);
      if (tracing_dirty_fields == nullptr)
        tracing_dirty_fields = new shrt::FieldMaskMap<IndexSpaceExpression>();
      if (tracing_dirty_fields->insert(expr, view_mask))
        expr->add_nested_expression_reference(did);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::invalidate_tracing_restricted_views(
        const FieldMapView<InstanceView>& restricted_views,
        IndexSpaceExpression* expr, FieldMask& restricted_mask)
    //--------------------------------------------------------------------------
    {
      legion_assert(tracing_postconditions != nullptr);
      for (FieldMapView<InstanceView>::const_iterator it =
               restricted_views.begin();
           it != restricted_views.end(); it++)
      {
        const FieldMask overlap = restricted_mask & it->second;
        if (!overlap)
          continue;
        tracing_postconditions->invalidate_all_but(it->first, expr, overlap);
        restricted_mask -= overlap;
        if (!restricted_mask)
          break;
      }
    }

    //--------------------------------------------------------------------------
    RtEvent EquivalenceSet::capture_trace_conditions(
        PhysicalTemplate* target, AddressSpaceID target_space,
        unsigned parent_req_index, std::atomic<unsigned>* result,
        RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      // This always needs to be sent to the owner to handle the case where
      // we are figuring out which shard owns each precondition expression
      // We can only deduplicate if they go to the same place
      if (!is_logical_owner())
      {
        if (!ready_event.exists())
          ready_event = Runtime::create_rt_user_event();
        EquivalenceSetCaptureRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(target);
          rez.serialize(target_space);
          rez.serialize(parent_req_index);
          rez.serialize(result);
          rez.serialize(ready_event);
        }
        rez.dispatch(logical_owner_space);
        return ready_event;
      }
      // Uniquify the tracing dirty fields so there is one expression per field
      local::FieldMaskMap<IndexSpaceExpression> unique_dirty_exprs;
      if (tracing_dirty_fields != nullptr)
      {
        local::list<FieldSet<IndexSpaceExpression*>> field_sets;
        tracing_dirty_fields->compute_field_sets(FieldMask(), field_sets);
        for (local::list<FieldSet<IndexSpaceExpression*>>::const_iterator it =
                 field_sets.begin();
             it != field_sets.end(); it++)
        {
          if (it->elements.size() > 1)
          {
            // Union the expressions together and produce a unique expression
            IndexSpaceExpression* union_expr =
                runtime->union_index_spaces(it->elements);
            unique_dirty_exprs.insert(union_expr, it->set_mask);
          }
          else
          {
            legion_assert(!it->elements.empty());
            unique_dirty_exprs.insert(*it->elements.begin(), it->set_mask);
          }
        }
      }
      // Now either pack up the state to send back to the target or do
      // the analysis here to create the trace condition sets
      if (target_space != local_space)
      {
        legion_assert(ready_event.exists());
        EquivalenceSetCaptureResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(target);
          rez.serialize(parent_req_index);
          rez.serialize(tree_id);
          rez.serialize(result);
          if (tracing_preconditions != nullptr)
            tracing_preconditions->pack(rez, target_space, true /*pack refs*/);
          else
            rez.serialize<size_t>(0);
          if (tracing_anticonditions != nullptr)
            tracing_anticonditions->pack(rez, target_space, true /*pack refs*/);
          else
            rez.serialize<size_t>(0);
          if (tracing_postconditions != nullptr)
            tracing_postconditions->pack(rez, target_space, true /*pack refs*/);
          else
            rez.serialize<size_t>(0);
          rez.serialize<size_t>(unique_dirty_exprs.size());
          for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   unique_dirty_exprs.begin();
               it != unique_dirty_exprs.end(); it++)
          {
            it->first->pack_expression(rez, target_space);
            rez.serialize(it->second);
          }
          rez.serialize(ready_event);
        }
        rez.dispatch(target_space);
      }
      else
      {
        target->receive_trace_conditions(
            tracing_preconditions, tracing_anticonditions,
            tracing_postconditions, unique_dirty_exprs, parent_req_index,
            tree_id, result);
        if (ready_event.exists())
          Runtime::trigger_event(ready_event);
      }
      if (tracing_preconditions != nullptr)
      {
        delete tracing_preconditions;
        tracing_preconditions = nullptr;
      }
      if (tracing_anticonditions != nullptr)
      {
        delete tracing_anticonditions;
        tracing_anticonditions = nullptr;
      }
      if (tracing_postconditions != nullptr)
      {
        delete tracing_postconditions;
        tracing_postconditions = nullptr;
      }
      if (tracing_dirty_fields != nullptr)
      {
        for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                 tracing_dirty_fields->begin();
             it != tracing_dirty_fields->end(); it++)
          if (it->first->remove_nested_expression_reference(did))
            delete it->first;
        delete tracing_dirty_fields;
        tracing_dirty_fields = nullptr;
      }
      return ready_event;
    }

    //--------------------------------------------------------------------------
    AddressSpaceID EquivalenceSet::select_collective_trace_capture_space(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(collective_mapping != nullptr);
      AutoLock eq(eq_lock);
      if (!replicate_logical_owner_space(
              local_space, collective_mapping, false))
      {
        legion_assert(replicated_owner_state->ready.exists());
        const RtEvent wait_on = replicated_owner_state->ready;
        eq.release();
        wait_on.wait();
        eq.reacquire();
      }
      if (collective_mapping->contains(logical_owner_space))
        return logical_owner_space;
      else
        return collective_mapping->find_nearest(logical_owner_space);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::remove_read_only_guard(CopyFillGuard* guard)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      // If we're no longer the logical owner then it's because we were
      // migrated and there should be no guards so we're done
      if (read_only_guards.empty())
        return;
      // We could get here when we're not the logical owner if we've unpacked
      // ourselves but haven't become the owner yet, in which case we still
      // need to prune ourselves out of the list
      shrt::FieldMaskMap<CopyFillGuard>::iterator finder =
          read_only_guards.find(guard);
      // It's also possible that the equivalence set is migrated away and
      // then migrated back before this guard is removed in which case we
      // won't find it in the update guards and can safely ignore it
      if (finder == read_only_guards.end())
        return;
      const bool should_tighten = !!finder->second;
      read_only_guards.erase(finder);
      if (should_tighten)
        read_only_guards.tighten_valid_mask();
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::remove_reduction_fill_guard(CopyFillGuard* guard)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      // If we're no longer the logical owner then it's because we were
      // migrated and there should be no guards so we're done
      if (reduction_fill_guards.empty())
        return;
      // We could get here when we're not the logical owner if we've unpacked
      // ourselves but haven't become the owner yet, in which case we still
      // need to prune ourselves out of the list
      shrt::FieldMaskMap<CopyFillGuard>::iterator finder =
          reduction_fill_guards.find(guard);
      // It's also possible that the equivalence set is migrated away and
      // then migrated back before this guard is removed in which case we
      // won't find it in the update guards and can safely ignore it
      if (finder == reduction_fill_guards.end())
        return;
      const bool should_tighten = !!finder->second;
      reduction_fill_guards.erase(finder);
      if (should_tighten)
        reduction_fill_guards.tighten_valid_mask();
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSetReplicationRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready_event;
      EquivalenceSet* set =
          runtime->find_or_request_equivalence_set(did, ready_event);
      size_t num_spaces;
      derez.deserialize(num_spaces);
      CollectiveMapping* mapping = nullptr;
      if (num_spaces > 0)
      {
        mapping = new CollectiveMapping(derez, num_spaces);
        mapping->add_reference();
      }
      derez.deserialize(source);
      if (ready_event.exists() && !ready_event.has_triggered())
        ready_event.wait();
      set->replicate_logical_owner_space(source, mapping, true /*need lock*/);
      if ((mapping != nullptr) && mapping->remove_reference())
        delete mapping;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSetReplicationResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready_event;
      EquivalenceSet* set =
          runtime->find_or_request_equivalence_set(did, ready_event);
      AddressSpaceID owner;
      derez.deserialize(owner);

      if (ready_event.exists() && !ready_event.has_triggered())
        ready_event.wait();
      set->process_replication_response(owner);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::send_equivalence_set(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner());
      // We should have had a request for this already
      legion_assert(!has_remote_instance(target));
      // If the target is in the collective mapping then we don't need to
      // bother sending the result since that just means that something
      // requested the equivalence set on a remote node before the equivalence
      // set creation could propagate there yet
      if ((collective_mapping != nullptr) &&
          collective_mapping->contains(target))
        return;
      update_remote_instances(target);
      EquivalenceSetResponse rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        set_expr->pack_expression(rez, target);
        rez.serialize(tree_id);
        context->pack_inner_context(rez);
        // There be dragons here!
        // In the case where we first make a new equivalence set on a
        // remote node that is about to be the owner, we can't mark it
        // as the owner until it receives all an unpack_state or
        // unpack_migration message which provides it valid meta-data
        // Therefore we'll tell it that we're the owner which will
        // create a cycle in the forwarding graph. This won't matter for
        // unpack_migration as it's going to overwrite the data in the
        // equivalence set anyway, but for upack_state, we'll need to
        // recognize when to break the cycle. Effectively whenever we
        // send an update to a remote node that we can tell has never
        // been the owner before (and therefore can't have migrated)
        // we know that we should just do the unpack there. This will
        // break the cycle and allow forward progress. Analysis messages
        // may go round and round a few times, but they have lower
        // priority and therefore shouldn't create a livelock.
        AutoLock eq(eq_lock, false /*exclusive*/);
        // is_ready tests whether this set expression has been set
        // it might not be in the case of an output region and we
        // don't want to block testing is_empty in that case if it
        // hasn't been set
        if (set_expr->is_set() && !set_expr->is_empty())
        {
          if (target == logical_owner_space)
            rez.serialize(local_space);
          else
            rez.serialize(logical_owner_space);
        }
        else
          rez.serialize(logical_owner_space);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSetRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID source;
      derez.deserialize(source);
      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      EquivalenceSet* set = legion_safe_cast<EquivalenceSet*>(dc);
      set->send_equivalence_set(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSetResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      IndexSpaceExpression* expr = IndexSpaceExpression::unpack_expression(
          derez, runtime->determine_owner(did));
      RegionTreeID tid;
      derez.deserialize(tid);
      InnerContext* context = InnerContext::unpack_inner_context(derez);
      AddressSpaceID logical_owner;
      derez.deserialize(logical_owner);
      void* location =
          runtime->find_or_create_pending_collectable_location<EquivalenceSet>(
              did);
      EquivalenceSet* set = new (location) EquivalenceSet(
          did, logical_owner, expr, tid, context, false /*register now*/);
      // Once construction is complete then we do the registration
      set->register_with_runtime();
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::clone_from(
        EquivalenceSet* src, const FieldMask& mask,
        IndexSpaceExpression* clone_expr, const bool record_invalidate,
        std::vector<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      legion_assert(this != src);
      legion_assert(src->tree_id == tree_id);
      AutoLock eq(eq_lock);
      if (is_logical_owner())
      {
        // Need to release the lock in case we're asked to clone to src for
        // other fiedls and expressions
        eq.release();
        src->clone_to_local(
            this, mask, clone_expr, applied_events, record_invalidate);
      }
      else
      {
        // Make a copy before release the lock so everything is consistent
        const AddressSpaceID logical_owner = logical_owner_space;
        eq.release();
        src->clone_to_remote(
            did, logical_owner, set_expr, clone_expr, mask, applied_events,
            record_invalidate);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_owner(const AddressSpaceID new_logical_owner)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      // We should never be told that we're the new owner this way
      legion_assert(new_logical_owner != local_space);
      // If we are the owner then we know this update is stale so ignore it
      if (!is_logical_owner())
        logical_owner_space = new_logical_owner;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::make_owner(RtEvent pre)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      if (pre.exists() && !pre.has_triggered())
      {
        const DeferMakeOwnerArgs args(this);
        runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY, pre);
      }
      else
      {
        // If we make it here then we can finally mark ourselves the owner
        AutoLock eq(eq_lock);
        logical_owner_space = local_space;
        if (replicated_owner_state != nullptr)
        {
          legion_assert(!replicated_owner_state->is_valid());
          // Send notifications to the children now that we know the owner
          EquivalenceSetReplicationResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(local_space);
          }
          for (const AddressSpaceID& child : replicated_owner_state->children)
            rez.dispatch(child);
          to_trigger = replicated_owner_state->ready;
          replicated_owner_state->ready = RtUserEvent::NO_RT_USER_EVENT;
        }
        unpack_global_ref();
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::DeferMakeOwnerArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      set->make_owner();
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSetOwnerUpdate::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet* set =
          runtime->find_or_request_equivalence_set(did, ready);
      AddressSpaceID new_owner;
      derez.deserialize(new_owner);
      RtUserEvent done;
      derez.deserialize(done);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      set->update_owner(new_owner);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSetMigration::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet* set =
          runtime->find_or_request_equivalence_set(did, ready);

      std::vector<RtEvent> ready_events;
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      set->unpack_state_and_apply(
          derez, source, ready_events, false /*forward*/);
      // Check to see if we're ready or we need to defer this
      if (!ready_events.empty())
        set->make_owner(Runtime::merge_events(ready_events));
      else
        set->make_owner();
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::pack_state(
        Serializer& rez, const AddressSpaceID target, DistributedID target_did,
        IndexSpaceExpression* target_expr, IndexSpaceExpression* expr,
        const bool expr_covers, const FieldMask& mask, const bool pack_guards,
        const bool pack_invalidates)
    //--------------------------------------------------------------------------
    {
      shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<LogicalView>>
          valid_updates;
      shrt::FieldMaskMap<IndexSpaceExpression> initialized_updates,
          invalid_updates;
      std::map<
          unsigned, std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>
          reduction_updates;
      shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>
          restricted_updates, released_updates;
      shrt::FieldMaskMap<CopyFillGuard> read_only_guards, reduction_fill_guards;
      TraceViewSet* precondition_updates = nullptr;
      TraceViewSet* anticondition_updates = nullptr;
      TraceViewSet* postcondition_updates = nullptr;
      shrt::FieldMaskMap<IndexSpaceExpression>* dirty_updates = nullptr;
      find_overlap_updates(
          expr, expr_covers, mask, pack_invalidates, valid_updates,
          initialized_updates, invalid_updates, reduction_updates,
          restricted_updates, released_updates,
          pack_guards ? &read_only_guards : nullptr,
          pack_guards ? &reduction_fill_guards : nullptr, precondition_updates,
          anticondition_updates, postcondition_updates, dirty_updates,
          target_did, target_expr);
      pack_updates(
          rez, target, valid_updates, initialized_updates, invalid_updates,
          reduction_updates, restricted_updates, released_updates,
          &read_only_guards, &reduction_fill_guards, precondition_updates,
          anticondition_updates, postcondition_updates, dirty_updates,
          true /*pack refs*/, true /*pack tracing reference*/);
      if (precondition_updates != nullptr)
        delete precondition_updates;
      if (anticondition_updates != nullptr)
        delete anticondition_updates;
      if (postcondition_updates != nullptr)
        delete postcondition_updates;
      if (dirty_updates != nullptr)
        delete dirty_updates;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::pack_updates(
        Serializer& rez, const AddressSpaceID target,
        const MapView<IndexSpaceExpression*, shrt::FieldMaskMap<LogicalView>>&
            valid_updates,
        const FieldMapView<IndexSpaceExpression>& initialized_updates,
        const FieldMapView<IndexSpaceExpression>& invalidated_updates,
        const std::map<
            unsigned,
            std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>&
            reduction_updates,
        const MapView<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>&
            restricted_updates,
        const MapView<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>&
            released_updates,
        const shrt::FieldMaskMap<CopyFillGuard>* read_only_updates,
        const shrt::FieldMaskMap<CopyFillGuard>* reduction_fill_updates,
        const TraceViewSet* precondition_updates,
        const TraceViewSet* anticondition_updates,
        const TraceViewSet* postcondition_updates,
        const shrt::FieldMaskMap<IndexSpaceExpression>* dirty_updates,
        const bool pack_references, const bool pack_tracing_references)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(valid_updates.size());
      for (MapView<IndexSpaceExpression*, shrt::FieldMaskMap<LogicalView>>::
               const_iterator vit = valid_updates.begin();
           vit != valid_updates.end(); vit++)
      {
        vit->first->pack_expression(rez, target);
        rez.serialize<size_t>(vit->second.size());
        for (shrt::FieldMaskMap<LogicalView>::const_iterator it =
                 vit->second.begin();
             it != vit->second.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
          if (pack_references)
            it->first->pack_valid_ref();
        }
      }
      rez.serialize<size_t>(initialized_updates.size());
      for (FieldMapView<IndexSpaceExpression>::const_iterator it =
               initialized_updates.begin();
           it != initialized_updates.end(); it++)
      {
        it->first->pack_expression(rez, target);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(invalidated_updates.size());
      for (FieldMapView<IndexSpaceExpression>::const_iterator it =
               invalidated_updates.begin();
           it != invalidated_updates.end(); it++)
      {
        it->first->pack_expression(rez, target);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(reduction_updates.size());
      for (const std::pair<
               const unsigned,
               std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>&
               rit : reduction_updates)
      {
        rez.serialize(rit.first);
        rez.serialize<size_t>(rit.second.size());
        for (const std::pair<InstanceView*, IndexSpaceExpression*>& it :
             rit.second)
        {
          rez.serialize(it.first->did);
          it.second->pack_expression(rez, target);
          if (pack_references)
            it.first->pack_valid_ref();
        }
      }
      rez.serialize<size_t>(restricted_updates.size());
      for (MapView<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>::
               const_iterator rit = restricted_updates.begin();
           rit != restricted_updates.end(); rit++)
      {
        rit->first->pack_expression(rez, target);
        rez.serialize<size_t>(rit->second.size());
        for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                 rit->second.begin();
             it != rit->second.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
          if (pack_references)
            it->first->pack_valid_ref();
        }
      }
      rez.serialize<size_t>(released_updates.size());
      for (MapView<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>::
               const_iterator rit = released_updates.begin();
           rit != released_updates.end(); rit++)
      {
        rit->first->pack_expression(rez, target);
        rez.serialize<size_t>(rit->second.size());
        for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                 rit->second.begin();
             it != rit->second.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
          if (pack_references)
            it->first->pack_valid_ref();
        }
      }
      if ((read_only_updates != nullptr) && !read_only_updates->empty())
      {
        rez.serialize<size_t>(read_only_updates->size());
        for (shrt::FieldMaskMap<CopyFillGuard>::const_iterator it =
                 read_only_updates->begin();
             it != read_only_updates->end(); it++)
        {
          it->first->pack_guard(rez);
          rez.serialize(it->second);
        }
      }
      else
        rez.serialize<size_t>(0);
      if ((reduction_fill_updates != nullptr) &&
          !reduction_fill_updates->empty())
      {
        rez.serialize<size_t>(reduction_fill_updates->size());
        for (shrt::FieldMaskMap<CopyFillGuard>::const_iterator it =
                 reduction_fill_updates->begin();
             it != reduction_fill_updates->end(); it++)
        {
          it->first->pack_guard(rez);
          rez.serialize(it->second);
        }
      }
      else
        rez.serialize<size_t>(0);
      if (precondition_updates != nullptr)
        precondition_updates->pack(rez, target, pack_tracing_references);
      else
        rez.serialize<size_t>(0);
      if (anticondition_updates != nullptr)
        anticondition_updates->pack(rez, target, pack_tracing_references);
      else
        rez.serialize<size_t>(0);
      if (postcondition_updates != nullptr)
        postcondition_updates->pack(rez, target, pack_tracing_references);
      else
        rez.serialize<size_t>(0);
      if (dirty_updates != nullptr)
      {
        rez.serialize<size_t>(dirty_updates->size());
        for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                 dirty_updates->begin();
             it != dirty_updates->end(); it++)
        {
          it->first->pack_expression(rez, target);
          rez.serialize(it->second);
        }
      }
      else
        rez.serialize<size_t>(0);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::unpack_state_and_apply(
        Deserializer& derez, const AddressSpaceID source,
        std::vector<RtEvent>& applied_events, const bool forward_to_owner)
    //--------------------------------------------------------------------------
    {
      shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<LogicalView>>
          valid_updates;
      shrt::FieldMaskMap<IndexSpaceExpression> initialized_updates,
          invalid_updates;
      std::map<
          unsigned, std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>
          reduction_updates;
      shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>
          restricted_updates, released_updates;
      shrt::FieldMaskMap<CopyFillGuard> read_only_updates,
          reduction_fill_updates;
      std::set<RtEvent> ready_events;
      size_t num_valid;
      derez.deserialize(num_valid);
      for (unsigned idx1 = 0; idx1 < num_valid; idx1++)
      {
        IndexSpaceExpression* expr =
            IndexSpaceExpression::unpack_expression(derez, source);
        size_t num_views;
        derez.deserialize(num_views);
        shrt::FieldMaskMap<LogicalView>& views = valid_updates[expr];
        for (unsigned idx2 = 0; idx2 < num_views; idx2++)
        {
          DistributedID did;
          derez.deserialize(did);
          RtEvent ready;
          LogicalView* view = runtime->find_or_request_logical_view(did, ready);
          if (ready.exists() && !ready.has_triggered())
            ready_events.insert(ready);
          FieldMask mask;
          derez.deserialize(mask);
          views.insert(view, mask);
        }
      }
      size_t num_initialized;
      derez.deserialize(num_initialized);
      for (unsigned idx = 0; idx < num_initialized; idx++)
      {
        IndexSpaceExpression* expr =
            IndexSpaceExpression::unpack_expression(derez, source);
        FieldMask mask;
        derez.deserialize(mask);
        initialized_updates.insert(expr, mask);
      }
      size_t num_invalidated;
      derez.deserialize(num_invalidated);
      for (unsigned idx = 0; idx < num_invalidated; idx++)
      {
        IndexSpaceExpression* expr =
            IndexSpaceExpression::unpack_expression(derez, source);
        FieldMask mask;
        derez.deserialize(mask);
        invalid_updates.insert(expr, mask);
      }
      size_t num_reductions;
      derez.deserialize(num_reductions);
      for (unsigned idx1 = 0; idx1 < num_reductions; idx1++)
      {
        unsigned fidx;
        derez.deserialize(fidx);
        size_t num_views;
        derez.deserialize(num_views);
        std::list<std::pair<InstanceView*, IndexSpaceExpression*>>& reductions =
            reduction_updates[fidx];
        for (unsigned idx2 = 0; idx2 < num_views; idx2++)
        {
          DistributedID did;
          derez.deserialize(did);
          RtEvent ready;
          LogicalView* view = runtime->find_or_request_logical_view(did, ready);
          if (ready.exists() && !ready.has_triggered())
            ready_events.insert(ready);
          IndexSpaceExpression* expr =
              IndexSpaceExpression::unpack_expression(derez, source);
          reductions.emplace_back(
              std::pair<InstanceView*, IndexSpaceExpression*>(
                  static_cast<InstanceView*>(view), expr));
        }
      }
      size_t num_restrictions;
      derez.deserialize(num_restrictions);
      for (unsigned idx1 = 0; idx1 < num_restrictions; idx1++)
      {
        IndexSpaceExpression* expr =
            IndexSpaceExpression::unpack_expression(derez, source);
        size_t num_views;
        derez.deserialize(num_views);
        shrt::FieldMaskMap<InstanceView>& restrictions =
            restricted_updates[expr];
        for (unsigned idx2 = 0; idx2 < num_views; idx2++)
        {
          DistributedID did;
          derez.deserialize(did);
          RtEvent ready;
          LogicalView* view = runtime->find_or_request_logical_view(did, ready);
          if (ready.exists() && !ready.has_triggered())
            ready_events.insert(ready);
          FieldMask mask;
          derez.deserialize(mask);
          restrictions.insert(static_cast<InstanceView*>(view), mask);
        }
      }
      size_t num_releases;
      derez.deserialize(num_releases);
      for (unsigned idx1 = 0; idx1 < num_releases; idx1++)
      {
        IndexSpaceExpression* expr =
            IndexSpaceExpression::unpack_expression(derez, source);
        size_t num_views;
        derez.deserialize(num_views);
        shrt::FieldMaskMap<InstanceView>& releases = released_updates[expr];
        for (unsigned idx2 = 0; idx2 < num_views; idx2++)
        {
          DistributedID did;
          derez.deserialize(did);
          RtEvent ready;
          LogicalView* view = runtime->find_or_request_logical_view(did, ready);
          if (ready.exists() && !ready.has_triggered())
            ready_events.insert(ready);
          FieldMask mask;
          derez.deserialize(mask);
          releases.insert(static_cast<InstanceView*>(view), mask);
        }
      }
      size_t num_read_only_guards;
      derez.deserialize(num_read_only_guards);
      if (num_read_only_guards)
      {
        // Need to hold the lock here to prevent copy fill guard
        // deletions from removing this before we've registered it
        AutoLock eq(eq_lock);
        for (unsigned idx = 0; idx < num_read_only_guards; idx++)
        {
          CopyFillGuard* guard = CopyFillGuard::unpack_guard(derez, this);
          FieldMask guard_mask;
          derez.deserialize(guard_mask);
          if (guard != nullptr)
          {
            read_only_guards.insert(guard, guard_mask);
            read_only_updates.insert(guard, guard_mask);
          }
        }
      }
      size_t num_reduction_fill_guards;
      derez.deserialize(num_reduction_fill_guards);
      if (num_reduction_fill_guards)
      {
        // Need to hold the lock here to prevent copy fill guard
        // deletions from removing this before we've registered it
        AutoLock eq(eq_lock);
        for (unsigned idx = 0; idx < num_reduction_fill_guards; idx++)
        {
          CopyFillGuard* guard = CopyFillGuard::unpack_guard(derez, this);
          FieldMask guard_mask;
          derez.deserialize(guard_mask);
          if (guard != nullptr)
          {
            reduction_fill_guards.insert(guard, guard_mask);
            reduction_fill_updates.insert(guard, guard_mask);
          }
        }
      }
      size_t num_preconditions;
      derez.deserialize(num_preconditions);
      TraceViewSet* precondition_updates = nullptr;
      if (num_preconditions > 0)
      {
        precondition_updates =
            new TraceViewSet(context, did, set_expr, tree_id);
        precondition_updates->unpack(
            derez, num_preconditions, source, ready_events);
      }
      size_t num_anticonditions;
      derez.deserialize(num_anticonditions);
      TraceViewSet* anticondition_updates = nullptr;
      if (num_anticonditions > 0)
      {
        anticondition_updates =
            new TraceViewSet(context, did, set_expr, tree_id);
        anticondition_updates->unpack(
            derez, num_anticonditions, source, ready_events);
      }
      size_t num_postconditions;
      derez.deserialize(num_postconditions);
      TraceViewSet* postcondition_updates = nullptr;
      if (num_postconditions > 0)
      {
        postcondition_updates =
            new TraceViewSet(context, did, set_expr, tree_id);
        postcondition_updates->unpack(
            derez, num_postconditions, source, ready_events);
      }
      size_t num_dirty;
      derez.deserialize(num_dirty);
      shrt::FieldMaskMap<IndexSpaceExpression>* dirty_updates = nullptr;
      if (num_dirty > 0)
      {
        dirty_updates = new shrt::FieldMaskMap<IndexSpaceExpression>();
        for (unsigned idx = 0; idx < num_dirty; idx++)
        {
          IndexSpaceExpression* expr =
              IndexSpaceExpression::unpack_expression(derez, source);
          FieldMask mask;
          derez.deserialize(mask);
          dirty_updates->insert(expr, mask);
        }
      }
      if (!ready_events.empty())
      {
        const RtEvent ready_event = Runtime::merge_events(ready_events);
        if (ready_event.exists() && !ready_event.has_triggered())
        {
          // Defer this until it is ready to be performed
          DeferApplyStateArgs args(
              this, forward_to_owner, applied_events, valid_updates,
              initialized_updates, invalid_updates, reduction_updates,
              restricted_updates, released_updates, read_only_updates,
              reduction_fill_updates, precondition_updates,
              anticondition_updates, postcondition_updates, dirty_updates);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_DEFERRED_PRIORITY, ready_event);
          return;
        }
      }
      // All the views are ready so we can add them now
      apply_state(
          valid_updates, initialized_updates, invalid_updates,
          reduction_updates, restricted_updates, released_updates,
          precondition_updates, anticondition_updates, postcondition_updates,
          dirty_updates, &read_only_updates, &reduction_fill_updates,
          applied_events, true /*need lock*/, forward_to_owner,
          true /*unpack tracing references*/);
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::DeferApplyStateArgs::DeferApplyStateArgs(
        EquivalenceSet* s, bool forward, std::vector<RtEvent>& applied_events,
        ExprLogicalViews& valid, shrt::FieldMaskMap<IndexSpaceExpression>& init,
        shrt::FieldMaskMap<IndexSpaceExpression>& invd,
        ExprReductionViews& reductions, ExprInstanceViews& restricted,
        ExprInstanceViews& released,
        shrt::FieldMaskMap<CopyFillGuard>& read_only,
        shrt::FieldMaskMap<CopyFillGuard>& reduc_fill,
        TraceViewSet* preconditions, TraceViewSet* anticonditions,
        TraceViewSet* postconditions,
        shrt::FieldMaskMap<IndexSpaceExpression>* dirt)
      : LgTaskArgs<DeferApplyStateArgs>(false, true), set(s),
        valid_updates(
            new shrt::map<
                IndexSpaceExpression*, shrt::FieldMaskMap<LogicalView>>()),
        initialized_updates(new shrt::FieldMaskMap<IndexSpaceExpression>()),
        invalidated_updates(new shrt::FieldMaskMap<IndexSpaceExpression>()),
        reduction_updates(
            new std::map<
                unsigned,
                std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>()),
        restricted_updates(
            new shrt::map<
                IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>()),
        released_updates(
            new shrt::map<
                IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>()),
        read_only_updates(new shrt::FieldMaskMap<CopyFillGuard>()),
        reduction_fill_updates(new shrt::FieldMaskMap<CopyFillGuard>()),
        precondition_updates(preconditions),
        anticondition_updates(anticonditions),
        postcondition_updates(postconditions), dirty_updates(dirt),
        expr_references(new std::set<IndexSpaceExpression*>()),
        done_event(Runtime::create_rt_user_event()), forward_to_owner(forward)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<
               IndexSpaceExpression* const, shrt::FieldMaskMap<LogicalView>>&
               it : valid)
        if (expr_references->insert(it.first).second)
          it.first->add_base_expression_reference(META_TASK_REF);
      valid_updates->swap(valid);
      for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
               init.begin();
           it != init.end(); it++)
        if (expr_references->insert(it->first).second)
          it->first->add_base_expression_reference(META_TASK_REF);
      initialized_updates->swap(init);
      for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
               invd.begin();
           it != invd.end(); it++)
        if (expr_references->insert(it->first).second)
          it->first->add_base_expression_reference(META_TASK_REF);
      invalidated_updates->swap(invd);
      for (const std::pair<
               const unsigned,
               std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>&
               rit : reductions)
        for (const std::pair<InstanceView*, IndexSpaceExpression*>& it :
             rit.second)
          if (expr_references->insert(it.second).second)
            it.second->add_base_expression_reference(META_TASK_REF);
      reduction_updates->swap(reductions);
      for (const std::pair<
               IndexSpaceExpression* const, shrt::FieldMaskMap<InstanceView>>&
               it : restricted)
        if (expr_references->insert(it.first).second)
          it.first->add_base_expression_reference(META_TASK_REF);
      restricted_updates->swap(restricted);
      for (const std::pair<
               IndexSpaceExpression* const, shrt::FieldMaskMap<InstanceView>>&
               it : released)
        if (expr_references->insert(it.first).second)
          it.first->add_base_expression_reference(META_TASK_REF);
      released_updates->swap(released);
      read_only_updates->swap(read_only);
      reduction_fill_updates->swap(reduc_fill);
      if (dirty_updates != nullptr)
      {
        for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                 dirty_updates->begin();
             it != dirty_updates->end(); it++)
          if (expr_references->insert(it->first).second)
            it->first->add_base_expression_reference(META_TASK_REF);
      }
      applied_events.emplace_back(done_event);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::DeferApplyStateArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      std::vector<RtEvent> applied_events;
      set->apply_state(
          *valid_updates, *initialized_updates, *invalidated_updates,
          *reduction_updates, *restricted_updates, *released_updates,
          precondition_updates, anticondition_updates, postcondition_updates,
          dirty_updates, read_only_updates, reduction_fill_updates,
          applied_events, true /*needs lock*/, forward_to_owner,
          true /*unpack tracing refs*/);
      if (!applied_events.empty())
        Runtime::trigger_event(
            done_event, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(done_event);
      for (IndexSpaceExpression* it : *expr_references)
        if (it->remove_base_expression_reference(META_TASK_REF))
          delete it;
      delete valid_updates;
      delete initialized_updates;
      delete invalidated_updates;
      delete reduction_updates;
      delete restricted_updates;
      delete released_updates;
      delete read_only_updates;
      delete reduction_fill_updates;
      delete expr_references;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::invalidate_state(
        IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& mask, bool record_invalidations)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_logical_owner());
      filter_valid_instances(expr, expr_covers, mask);
      filter_reduction_instances(expr, expr_covers, mask);
      filter_initialized_data(expr, expr_covers, mask);
      filter_restricted_instances(expr, expr_covers, mask);
      filter_released_instances(expr, expr_covers, mask);
      if (tracing_preconditions != nullptr)
      {
        tracing_preconditions->invalidate_all_but(nullptr, expr, mask);
        if (tracing_preconditions->empty())
        {
          delete tracing_preconditions;
          tracing_preconditions = nullptr;
        }
      }
      if (tracing_anticonditions != nullptr)
      {
        tracing_anticonditions->invalidate_all_but(nullptr, expr, mask);
        if (tracing_anticonditions->empty())
        {
          delete tracing_anticonditions;
          tracing_anticonditions = nullptr;
        }
      }
      if (tracing_postconditions != nullptr)
      {
        tracing_postconditions->invalidate_all_but(nullptr, expr, mask);
        if (tracing_postconditions->empty())
        {
          delete tracing_postconditions;
          tracing_postconditions = nullptr;
        }
      }
      if (tracing_dirty_fields != nullptr)
      {
        local::FieldMaskMap<IndexSpaceExpression> to_add;
        std::vector<IndexSpaceExpression*> to_delete;
        for (shrt::FieldMaskMap<IndexSpaceExpression>::iterator it =
                 tracing_dirty_fields->begin();
             it != tracing_dirty_fields->end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          IndexSpaceExpression* diff =
              runtime->subtract_index_spaces(it->first, expr);
          if (diff->get_volume() == it->first->get_volume())
            continue;
          if (!diff->is_empty())
            to_add.insert(diff, overlap);
          it.filter(overlap);
          if (!it->second)
            to_delete.emplace_back(it->first);
        }
        for (IndexSpaceExpression* it : to_delete)
        {
          if (to_add.find(it) != to_add.end())
            continue;
          tracing_dirty_fields->erase(it);
          if (it->remove_nested_expression_reference(did))
            delete it;
        }
        for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                 to_add.begin();
             it != to_add.end(); it++)
          if (tracing_dirty_fields->insert(it->first, it->second))
            it->first->add_nested_expression_reference(did);
        if (tracing_dirty_fields->empty())
        {
          delete tracing_dirty_fields;
          tracing_dirty_fields = nullptr;
        }
      }
      if (record_invalidations)
      {
        if (!(mask * partial_invalidations.get_valid_mask()))
        {
          // Should never invalidate things twice
          legion_assert(!expr_covers);
          FieldMask remaining = mask;
          local::FieldMaskMap<IndexSpaceExpression> to_add;
          std::vector<IndexSpaceExpression*> to_delete;
          for (lng::FieldMaskMap<IndexSpaceExpression>::iterator it =
                   partial_invalidations.begin();
               it != partial_invalidations.end(); it++)
          {
            const FieldMask overlap = remaining & it->second;
            if (!overlap)
              continue;
            IndexSpaceExpression* union_expr =
                runtime->union_index_spaces(it->first, expr);
            const size_t union_volume = union_expr->get_volume();
            // There shouldn't have been any overlap here
            legion_assert(
                union_volume == (it->first->get_volume() + expr->get_volume()));
            if (union_volume == set_expr->get_volume())
              to_add.insert(set_expr, overlap);
            else
              to_add.insert(union_expr, overlap);
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(it->first);
            remaining -= overlap;
            if (!remaining)
              break;
          }
          for (IndexSpaceExpression* it : to_delete)
          {
            partial_invalidations.erase(it);
            if (it->remove_nested_expression_reference(did))
              delete it;
          }
          for (local::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   to_add.begin();
               it != to_add.end(); it++)
            if (partial_invalidations.insert(it->first, it->second))
              it->first->add_nested_expression_reference(did);
          if (!!remaining && partial_invalidations.insert(expr, remaining))
            expr->add_nested_expression_reference(did);
        }
        else if (partial_invalidations.insert(expr, mask))
          expr->add_nested_expression_reference(did);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::clone_to_local(
        EquivalenceSet* dst, FieldMask mask, IndexSpaceExpression* overlap,
        std::vector<RtEvent>& applied_events, const bool record_invalidate,
        const bool need_dst_lock)
    //--------------------------------------------------------------------------
    {
      legion_assert(dst->tree_id == tree_id);
      shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<LogicalView>>
          valid_updates;
      shrt::FieldMaskMap<IndexSpaceExpression> initialized_updates,
          invalid_updates;
      std::map<
          unsigned, std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>
          reduction_updates;
      shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>
          restricted_updates, released_updates;
      TraceViewSet* precondition_updates = nullptr;
      TraceViewSet* anticondition_updates = nullptr;
      TraceViewSet* postcondition_updates = nullptr;
      shrt::FieldMaskMap<IndexSpaceExpression>* dirty_updates = nullptr;
      std::vector<IndexSpaceExpression*> temp_refs;
      {
        // Lock in exclusive mode since we're doing an invalidate
        AutoLock eq(eq_lock);
        if (!is_logical_owner())
        {
          const RtUserEvent done_event = Runtime::create_rt_user_event();
          EquivalenceSetCloneRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(dst->did);
            rez.serialize(local_space);
            dst->set_expr->pack_expression(rez, logical_owner_space);
            overlap->pack_expression(rez, logical_owner_space);
            rez.serialize(mask);
            rez.serialize(done_event);
            rez.serialize<bool>(record_invalidate);
          }
          rez.dispatch(logical_owner_space);
          applied_events.emplace_back(done_event);
          return;
        }
        bool overlap_covers = true;
        ;
        // If we get here, we're performing the clone locally for these fields
        if (!set_expr->is_empty())
        {
          const size_t overlap_volume = overlap->get_volume();
          legion_assert(overlap_volume > 0);
          overlap_covers = (overlap_volume == set_expr->get_volume());
          find_overlap_updates(
              overlap_covers ? set_expr : overlap, overlap_covers, mask,
              false /*invalids*/, valid_updates, initialized_updates,
              invalid_updates, reduction_updates, restricted_updates,
              released_updates, nullptr /*guards*/, nullptr /*guards*/,
              precondition_updates, anticondition_updates,
              postcondition_updates, dirty_updates, dst->did, dst->set_expr);
        }
        // Don't need to get updates if this equivalence set is empty
        // and the destination equivalence set is not empty
        else if (dst->set_expr->is_empty())
          find_overlap_updates(
              set_expr, true /*covers*/, mask, false /*invalids*/,
              valid_updates, initialized_updates, invalid_updates,
              reduction_updates, restricted_updates, released_updates,
              nullptr /*guards*/, nullptr /*guards*/, precondition_updates,
              anticondition_updates, postcondition_updates, dirty_updates,
              dst->did, dst->set_expr);
        // Save references on everything to keep it alive since we're about
        // to do an invalidation of the state, pack references to views in
        // case they ultimately need to be forwarded to a remote node when
        // we end up calling apply_state
        for (shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<LogicalView>>::
                 const_iterator vit = valid_updates.begin();
             vit != valid_updates.end(); vit++)
        {
          if (!std::binary_search(
                  temp_refs.begin(), temp_refs.end(), vit->first))
          {
            vit->first->add_nested_expression_reference(did);
            temp_refs.emplace_back(vit->first);
            std::sort(temp_refs.begin(), temp_refs.end());
          }
          for (shrt::FieldMaskMap<LogicalView>::const_iterator it =
                   vit->second.begin();
               it != vit->second.end(); it++)
            it->first->pack_valid_ref();
        }
        for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                 initialized_updates.begin();
             it != initialized_updates.end(); it++)
        {
          if (!std::binary_search(
                  temp_refs.begin(), temp_refs.end(), it->first))
          {
            it->first->add_nested_expression_reference(did);
            temp_refs.emplace_back(it->first);
            std::sort(temp_refs.begin(), temp_refs.end());
          }
        }
        for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                 invalid_updates.begin();
             it != invalid_updates.end(); it++)
        {
          if (!std::binary_search(
                  temp_refs.begin(), temp_refs.end(), it->first))
          {
            it->first->add_nested_expression_reference(did);
            temp_refs.emplace_back(it->first);
            std::sort(temp_refs.begin(), temp_refs.end());
          }
        }
        for (const std::pair<
                 const unsigned,
                 std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>&
                 rit : reduction_updates)
        {
          for (const std::pair<InstanceView*, IndexSpaceExpression*>& it :
               rit.second)
          {
            it.first->pack_valid_ref();
            if (!std::binary_search(
                    temp_refs.begin(), temp_refs.end(), it.second))
            {
              it.second->add_nested_expression_reference(did);
              temp_refs.emplace_back(it.second);
              std::sort(temp_refs.begin(), temp_refs.end());
            }
          }
        }
        for (shrt::map<
                 IndexSpaceExpression*,
                 shrt::FieldMaskMap<InstanceView>>::const_iterator rit =
                 restricted_updates.begin();
             rit != restricted_updates.end(); rit++)
        {
          if (!std::binary_search(
                  temp_refs.begin(), temp_refs.end(), rit->first))
          {
            rit->first->add_nested_expression_reference(did);
            temp_refs.emplace_back(rit->first);
            std::sort(temp_refs.begin(), temp_refs.end());
          }
          for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                   rit->second.begin();
               it != rit->second.end(); it++)
            it->first->pack_valid_ref();
        }
        for (shrt::map<
                 IndexSpaceExpression*,
                 shrt::FieldMaskMap<InstanceView>>::const_iterator rit =
                 released_updates.begin();
             rit != released_updates.end(); rit++)
        {
          if (!std::binary_search(
                  temp_refs.begin(), temp_refs.end(), rit->first))
          {
            rit->first->add_nested_expression_reference(did);
            temp_refs.emplace_back(rit->first);
            std::sort(temp_refs.begin(), temp_refs.end());
          }
          for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                   rit->second.begin();
               it != rit->second.end(); it++)
            it->first->pack_valid_ref();
        }
        // No need to do anything for the tracing data structures as they
        // are already keeping their own references to everything
        if (dirty_updates != nullptr)
        {
          for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   dirty_updates->begin();
               it != dirty_updates->end(); it++)
          {
            if (!std::binary_search(
                    temp_refs.begin(), temp_refs.end(), it->first))
            {
              it->first->add_nested_expression_reference(did);
              temp_refs.emplace_back(it->first);
              std::sort(temp_refs.begin(), temp_refs.end());
            }
          }
        }
        // Now we can do the invalidation
        if (overlap_covers)
          invalidate_state(set_expr, true /*covers*/, mask, record_invalidate);
        else
          invalidate_state(overlap, false /*covers*/, mask, record_invalidate);
      }
      // Call back to the destination to apply the state
      dst->apply_state(
          valid_updates, initialized_updates, invalid_updates,
          reduction_updates, restricted_updates, released_updates,
          precondition_updates, anticondition_updates, postcondition_updates,
          dirty_updates, nullptr /*guards*/, nullptr /*guards*/, applied_events,
          need_dst_lock, true /*forward to owner*/, false /*unpack tracing*/);
      // Remove the temporary references that we added to keep everything alive
      for (IndexSpaceExpression* it : temp_refs)
        if (it->remove_nested_expression_reference(did))
          delete it;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::clone_to_remote(
        DistributedID target, AddressSpaceID target_space,
        IndexSpaceExpression* target_expr, IndexSpaceExpression* overlap,
        FieldMask mask, std::vector<RtEvent>& applied_events,
        const bool record_invalidate)
    //--------------------------------------------------------------------------
    {
      legion_assert(!overlap->is_empty());
      const size_t overlap_volume = overlap->get_volume();
      const size_t set_volume = set_expr->get_volume();
      const bool overlap_covers = (overlap_volume == set_volume);
      if (overlap_covers)
        overlap = set_expr;
      // Lock in exclusive mode since we're doing an invalidate
      AutoLock eq(eq_lock);
      if (!is_logical_owner())
      {
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        EquivalenceSetCloneRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(target);
          rez.serialize(target_space);
          target_expr->pack_expression(rez, logical_owner_space);
          overlap->pack_expression(rez, logical_owner_space);
          rez.serialize(mask);
          rez.serialize(done_event);
          rez.serialize<bool>(record_invalidate);
        }
        rez.dispatch(logical_owner_space);
        applied_events.emplace_back(done_event);
      }
      else
      {
        // If we make it here, then we've got valid data for the all the fields
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        EquivalenceSetCloneResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(target);
          rez.serialize(local_space);
          rez.serialize(done_event);
          pack_state(
              rez, target_space, target, target_expr, overlap, overlap_covers,
              mask, false /*pack guards*/, false /*pack invalids*/);
        }
        rez.dispatch(target_space);
        applied_events.emplace_back(done_event);
        if (!set_expr->is_empty())
          invalidate_state(overlap, overlap_covers, mask, record_invalidate);
        else
          invalidate_state(set_expr, true /*cover*/, mask, record_invalidate);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::find_overlap_updates(
        IndexSpaceExpression* overlap_expr, const bool overlap_covers,
        const FieldMask& mask, const bool find_invalidates,
        shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<LogicalView>>&
            valid_updates,
        shrt::FieldMaskMap<IndexSpaceExpression>& initialized_updates,
        shrt::FieldMaskMap<IndexSpaceExpression>& invalidated_updates,
        std::map<
            unsigned,
            std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>&
            reduction_updates,
        shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>&
            restricted_updates,
        shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>&
            released_updates,
        shrt::FieldMaskMap<CopyFillGuard>* read_only_guard_updates,
        shrt::FieldMaskMap<CopyFillGuard>* reduction_fill_guard_updates,
        TraceViewSet*& precondition_updates,
        TraceViewSet*& anticondition_updates,
        TraceViewSet*& postcondition_updates,
        shrt::FieldMaskMap<IndexSpaceExpression>*& dirty_updates,
        DistributedID target_did, IndexSpaceExpression* target_expr) const
    //--------------------------------------------------------------------------
    {
      // Get updates from the total valid instances
      if (!total_valid_instances.empty() &&
          !(mask * total_valid_instances.get_valid_mask()))
      {
        if (!!(total_valid_instances.get_valid_mask() - mask))
        {
          // Need to filter on fields
          for (shrt::FieldMaskMap<LogicalView>::const_iterator it =
                   total_valid_instances.begin();
               it != total_valid_instances.end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            if (overlap_covers)
              valid_updates[set_expr].insert(it->first, overlap);
            else
              valid_updates[overlap_expr].insert(it->first, overlap);
          }
        }
        else
        {
          if (overlap_covers)
            valid_updates[set_expr] = total_valid_instances;
          else
            valid_updates[overlap_expr] = total_valid_instances;
        }
      }
      // Get updates from the partial valid instances
      if (!partial_valid_instances.empty() && !(mask * partial_valid_fields))
      {
        if (!!(partial_valid_fields - mask))
        {
          // Need to filter on fields
          for (ViewExprMaskSets::const_iterator pit =
                   partial_valid_instances.begin();
               pit != partial_valid_instances.end(); pit++)
          {
            if (pit->second.get_valid_mask() * mask)
              continue;
            for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                     pit->second.begin();
                 it != pit->second.end(); it++)
            {
              const FieldMask overlap = mask & it->second;
              if (!overlap)
                continue;
              if (!overlap_covers)
              {
                // Check for expression overlap
                IndexSpaceExpression* intersection =
                    runtime->intersect_index_spaces(overlap_expr, it->first);
                const size_t volume = intersection->get_volume();
                if (volume == 0)
                  continue;
                if (volume == overlap_expr->get_volume())
                  valid_updates[overlap_expr].insert(pit->first, overlap);
                else if (volume == it->first->get_volume())
                  valid_updates[it->first].insert(pit->first, overlap);
                else
                  valid_updates[intersection].insert(pit->first, overlap);
              }
              else
                valid_updates[it->first].insert(pit->first, overlap);
            }
          }
        }
        else
        {
          // No filtering on fields, just check expressions if necessary
          for (ViewExprMaskSets::const_iterator pit =
                   partial_valid_instances.begin();
               pit != partial_valid_instances.end(); pit++)
          {
            for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                     pit->second.begin();
                 it != pit->second.end(); it++)
            {
              if (!overlap_covers)
              {
                // Check for expression overlap
                IndexSpaceExpression* intersection =
                    runtime->intersect_index_spaces(overlap_expr, it->first);
                const size_t volume = intersection->get_volume();
                if (volume == 0)
                  continue;
                if (volume == overlap_expr->get_volume())
                  valid_updates[overlap_expr].insert(pit->first, it->second);
                else if (volume == it->first->get_volume())
                  valid_updates[it->first].insert(pit->first, it->second);
                else
                  valid_updates[intersection].insert(pit->first, it->second);
              }
              else
                valid_updates[it->first].insert(pit->first, it->second);
            }
          }
        }
      }
      // Get updates on the initialized data
      if (!initialized_data.empty() &&
          !(mask * initialized_data.get_valid_mask()))
      {
        if (!overlap_covers || !!(initialized_data.get_valid_mask() - mask))
        {
          for (lng::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   initialized_data.begin();
               it != initialized_data.end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            if (!overlap_covers)
            {
              IndexSpaceExpression* intersection =
                  runtime->intersect_index_spaces(it->first, overlap_expr);
              const size_t volume = intersection->get_volume();
              if (volume == 0)
                continue;
              if (volume == overlap_expr->get_volume())
                initialized_updates.insert(overlap_expr, overlap);
              else if (volume == it->first->get_volume())
                initialized_updates.insert(it->first, overlap);
              else
                initialized_updates.insert(intersection, overlap);
            }
            else
              initialized_updates.insert(it->first, overlap);
          }
        }
        else
        {
          for (lng::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   initialized_data.begin();
               it != initialized_data.end(); it++)
            initialized_updates.insert(it->first, it->second);
        }
      }
      if (find_invalidates && !partial_invalidations.empty())
      {
        // The only time we should be packing partial invalidations
        // should be if we're moving the entire equivalence set
        legion_assert(overlap_covers);
        legion_assert(!(partial_invalidations.get_valid_mask() - mask));
        for (lng::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                 partial_invalidations.begin();
             it != partial_invalidations.end(); it++)
          invalidated_updates.insert(it->first, it->second);
      }
      // Get updates from the reductions
      if (!reduction_instances.empty() && !(mask * reduction_fields))
      {
        for (const std::pair<
                 const unsigned,
                 std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>&
                 rit : reduction_instances)
        {
          if (!mask.is_set(rit.first))
            continue;
          std::list<std::pair<InstanceView*, IndexSpaceExpression*>>& updates =
              reduction_updates[rit.first];
          updates = rit.second;
          if (!overlap_covers)
          {
            for (std::list<std::pair<InstanceView*, IndexSpaceExpression*>>::
                     iterator it = updates.begin();
                 it != updates.end();
                 /*nothing*/)
            {
              if (it->second == set_expr)
              {
                it->second = overlap_expr;
                it++;
              }
              else
              {
                IndexSpaceExpression* intersection =
                    runtime->intersect_index_spaces(it->second, overlap_expr);
                const size_t volume = intersection->get_volume();
                if (volume > 0)
                {
                  if (volume == overlap_expr->get_volume())
                    it->second = overlap_expr;
                  else if (volume < it->second->get_volume())
                    it->second = intersection;
                  it++;
                }
                else
                  it = updates.erase(it);
              }
            }
          }
        }
      }
      // Get updates from the restricted instances
      if (!restricted_instances.empty() && !(mask * restricted_fields))
      {
        for (ExprViewMaskSets::const_iterator rit =
                 restricted_instances.begin();
             rit != restricted_instances.end(); rit++)
        {
          if (mask * rit->second.get_valid_mask())
            continue;
          IndexSpaceExpression* restricted_overlap = rit->first;
          if (!overlap_covers)
          {
            IndexSpaceExpression* intersection =
                runtime->intersect_index_spaces(rit->first, overlap_expr);
            const size_t volume = intersection->get_volume();
            if (volume == 0)
              continue;
            if (volume == overlap_expr->get_volume())
              restricted_overlap = overlap_expr;
            else if (volume < rit->first->get_volume())
              restricted_overlap = intersection;
          }
          shrt::FieldMaskMap<InstanceView>& updates =
              restricted_updates[restricted_overlap];
          for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                   rit->second.begin();
               it != rit->second.end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            updates.insert(it->first, overlap);
          }
        }
      }
      // Get updates from the released instances
      if (!released_instances.empty())
      {
        for (const std::pair<
                 IndexSpaceExpression* const, shrt::FieldMaskMap<InstanceView>>&
                 rit : released_instances)
        {
          if (mask * rit.second.get_valid_mask())
            continue;
          IndexSpaceExpression* released_overlap = rit.first;
          if (!overlap_covers)
          {
            IndexSpaceExpression* intersection =
                runtime->intersect_index_spaces(rit.first, overlap_expr);
            const size_t volume = intersection->get_volume();
            if (volume == 0)
              continue;
            if (volume == overlap_expr->get_volume())
              released_overlap = overlap_expr;
            else if (volume < rit.first->get_volume())
              released_overlap = intersection;
          }
          shrt::FieldMaskMap<InstanceView>& updates =
              released_updates[released_overlap];
          for (const std::pair<InstanceView* const, FieldMask>& it : rit.second)
          {
            const FieldMask overlap = mask & it.second;
            if (!overlap)
              continue;
            updates.insert(it.first, overlap);
          }
        }
      }
      // There is something really scary here so be very careful
      // It might look like we have read-only guards even though
      // read_only_guard_updates is nullptr. You might think that this
      // is very bad because we should be capturing those guards.
      // This should not be necessary though because the guards are
      // very conservative with these equivalence sets and they span
      // the whole equivalence set, even when the updates we care
      // about here might only be for a subset of the equivalence
      // set. Therefore it should be safe to ignore them in this case.
      if (!read_only_guards.empty() && (read_only_guard_updates != nullptr) &&
          !(mask * read_only_guards.get_valid_mask()))
      {
        for (shrt::FieldMaskMap<CopyFillGuard>::const_iterator it =
                 read_only_guards.begin();
             it != read_only_guards.end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          read_only_guard_updates->insert(it->first, overlap);
        }
      }
      // See same "scary" comment above because it applies here too
      if (!reduction_fill_guards.empty() &&
          (reduction_fill_guard_updates != nullptr) &&
          !(mask * reduction_fill_guards.get_valid_mask()))
      {
        for (shrt::FieldMaskMap<CopyFillGuard>::const_iterator it =
                 reduction_fill_guards.begin();
             it != reduction_fill_guards.end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          reduction_fill_guard_updates->insert(it->first, overlap);
        }
      }
      if (tracing_preconditions != nullptr)
      {
        if (precondition_updates == nullptr)
        {
          precondition_updates =
              new TraceViewSet(context, target_did, target_expr, tree_id);
          tracing_preconditions->find_overlaps(
              *precondition_updates, overlap_expr, overlap_covers, mask);
          if (precondition_updates->empty())
          {
            delete precondition_updates;
            precondition_updates = nullptr;
          }
        }
        else
          tracing_preconditions->find_overlaps(
              *precondition_updates, overlap_expr, overlap_covers, mask);
      }
      if (tracing_anticonditions != nullptr)
      {
        if (anticondition_updates == nullptr)
        {
          anticondition_updates =
              new TraceViewSet(context, target_did, target_expr, tree_id);
          tracing_anticonditions->find_overlaps(
              *anticondition_updates, overlap_expr, overlap_covers, mask);
          if (anticondition_updates->empty())
          {
            delete anticondition_updates;
            anticondition_updates = nullptr;
          }
        }
        else
          tracing_anticonditions->find_overlaps(
              *anticondition_updates, overlap_expr, overlap_covers, mask);
      }
      if (tracing_postconditions != nullptr)
      {
        if (postcondition_updates == nullptr)
        {
          postcondition_updates =
              new TraceViewSet(context, target_did, target_expr, tree_id);
          tracing_postconditions->find_overlaps(
              *postcondition_updates, overlap_expr, overlap_covers, mask);
          if (postcondition_updates->empty())
          {
            delete postcondition_updates;
            postcondition_updates = nullptr;
          }
        }
        else
          tracing_postconditions->find_overlaps(
              *postcondition_updates, overlap_expr, overlap_covers, mask);
      }
      if (tracing_dirty_fields != nullptr)
      {
        for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                 tracing_dirty_fields->begin();
             it != tracing_dirty_fields->end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          IndexSpaceExpression* overlap_expr =
              runtime->intersect_index_spaces(target_expr, it->first);
          if (overlap_expr->is_empty())
            continue;
          if (dirty_updates == nullptr)
            dirty_updates = new shrt::FieldMaskMap<IndexSpaceExpression>();
          dirty_updates->insert(overlap_expr, overlap);
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::apply_state(
        shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<LogicalView>>&
            valid_updates,
        shrt::FieldMaskMap<IndexSpaceExpression>& initialized_updates,
        shrt::FieldMaskMap<IndexSpaceExpression>& invalidated_updates,
        std::map<
            unsigned,
            std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>&
            reduction_updates,
        shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>&
            restricted_updates,
        shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>&
            released_updates,
        TraceViewSet* precondition_updates, TraceViewSet* anticondition_updates,
        TraceViewSet* postcondition_updates,
        shrt::FieldMaskMap<IndexSpaceExpression>* dirty_updates,
        shrt::FieldMaskMap<CopyFillGuard>* read_only_guard_updates,
        shrt::FieldMaskMap<CopyFillGuard>* reduction_fill_guard_updates,
        std::vector<RtEvent>& applied_events, const bool needs_lock,
        const bool forward_to_owner, const bool unpack_tracing_references)
    //--------------------------------------------------------------------------
    {
      if (needs_lock)
      {
        AutoLock eq(eq_lock);
        apply_state(
            valid_updates, initialized_updates, invalidated_updates,
            reduction_updates, restricted_updates, released_updates,
            precondition_updates, anticondition_updates, postcondition_updates,
            dirty_updates, read_only_guard_updates,
            reduction_fill_guard_updates, applied_events, false /*needs lock*/,
            forward_to_owner, unpack_tracing_references);
        return;
      }
      if (!is_logical_owner() && forward_to_owner)
      {
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        // Filter out any guard updates that have been pruned out
        // while we were not holding the lock. We know they've been
        // pruned because they will no longer be in the update_guards
        if (read_only_guard_updates != nullptr)
        {
          std::vector<CopyFillGuard*> to_delete;
          for (shrt::FieldMaskMap<CopyFillGuard>::const_iterator it =
                   read_only_guard_updates->begin();
               it != read_only_guard_updates->end(); it++)
            if (read_only_guards.find(it->first) == read_only_guards.end())
              to_delete.emplace_back(it->first);
          if (!to_delete.empty())
          {
            for (CopyFillGuard* it : to_delete)
              read_only_guard_updates->erase(it);
          }
        }
        if (reduction_fill_guard_updates != nullptr)
        {
          std::vector<CopyFillGuard*> to_delete;
          for (shrt::FieldMaskMap<CopyFillGuard>::const_iterator it =
                   reduction_fill_guard_updates->begin();
               it != reduction_fill_guard_updates->end(); it++)
            if (reduction_fill_guards.find(it->first) ==
                reduction_fill_guards.end())
              to_delete.emplace_back(it->first);
          if (!to_delete.empty())
          {
            for (CopyFillGuard* it : to_delete)
              reduction_fill_guard_updates->erase(it);
          }
        }
        // Forward this on to the logical owner
        EquivalenceSetCloneResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(local_space);
          rez.serialize(done_event);
          pack_updates(
              rez, logical_owner_space, valid_updates, initialized_updates,
              invalidated_updates, reduction_updates, restricted_updates,
              released_updates, read_only_guard_updates,
              reduction_fill_guard_updates, precondition_updates,
              anticondition_updates, postcondition_updates, dirty_updates,
              false /*pack references*/, !unpack_tracing_references);
        }
        rez.dispatch(logical_owner_space);
        applied_events.emplace_back(done_event);
        return;
      }
      const size_t dst_volume = set_expr->get_volume();
      for (shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<LogicalView>>::
               const_iterator it = valid_updates.begin();
           it != valid_updates.end(); it++)
      {
        if (it->first->get_volume() == dst_volume)
          record_instances(
              set_expr, true /*covers*/, it->second.get_valid_mask(),
              FieldMapView(it->second));
        else
          record_instances(
              it->first, false /*covers*/, it->second.get_valid_mask(),
              FieldMapView(it->second));
        for (shrt::FieldMaskMap<LogicalView>::const_iterator vit =
                 it->second.begin();
             vit != it->second.end(); vit++)
          vit->first->unpack_valid_ref();
      }
      for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
               initialized_updates.begin();
           it != initialized_updates.end(); it++)
      {
        if (it->first->get_volume() == dst_volume)
          update_initialized_data(set_expr, true /*covers*/, it->second);
        else
          update_initialized_data(it->first, false /*covers*/, it->second);
      }
      if (!invalidated_updates.empty())
      {
        // The only reaon we are moving partial invalidations is if we're
        // moving the entire equivalence sets so this should be empty and
        // we should just be able to swap it in
        legion_assert(partial_invalidations.empty());
        for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                 invalidated_updates.begin();
             it != invalidated_updates.end(); it++)
          if (partial_invalidations.insert(it->first, it->second))
            it->first->add_nested_expression_reference(did);
      }
      for (std::pair<
               const unsigned,
               std::list<std::pair<InstanceView*, IndexSpaceExpression*>>>&
               rit : reduction_updates)
      {
        // Need to make a local copy here since this can destroy the list
        local::vector<InstanceView*> local_reductions;
        local_reductions.reserve(rit.second.size());
        for (const std::pair<InstanceView*, IndexSpaceExpression*>& vit :
             rit.second)
          local_reductions.emplace_back(vit.first);
        // Update the reductions
        update_reductions(rit.first, rit.second);
        // Then unpack our valid references
        for (InstanceView* vit : local_reductions) vit->unpack_valid_ref();
      }
      for (shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>::
               const_iterator rit = restricted_updates.begin();
           rit != restricted_updates.end(); rit++)
      {
        const bool covers = (rit->first->get_volume() == dst_volume);
        for (shrt::FieldMaskMap<InstanceView>::const_iterator it =
                 rit->second.begin();
             it != rit->second.end(); it++)
        {
          record_restriction(
              covers ? set_expr : rit->first, covers, it->second, it->first);
          it->first->unpack_valid_ref();
        }
      }
      for (shrt::map<IndexSpaceExpression*, shrt::FieldMaskMap<InstanceView>>::
               iterator it = released_updates.begin();
           it != released_updates.end(); it++)
      {
        update_released(
            it->first, (it->first->get_volume() == dst_volume), it->second);
        for (shrt::FieldMaskMap<InstanceView>::const_iterator vit =
                 it->second.begin();
             vit != it->second.end(); vit++)
          vit->first->unpack_valid_ref();
      }
      if (precondition_updates != nullptr)
      {
        legion_assert(precondition_updates->owner_did == did);
        legion_assert(precondition_updates->expression == set_expr);
        legion_assert(precondition_updates->tree_id == tree_id);
        if (tracing_preconditions == nullptr)
        {
          tracing_preconditions = precondition_updates;
          if (unpack_tracing_references)
            tracing_preconditions->unpack_references();
        }
        else
        {
          precondition_updates->merge(*tracing_preconditions);
          if (unpack_tracing_references)
            precondition_updates->unpack_references();
          delete precondition_updates;
        }
      }
      if (anticondition_updates != nullptr)
      {
        legion_assert(anticondition_updates->owner_did == did);
        legion_assert(anticondition_updates->expression == set_expr);
        legion_assert(anticondition_updates->tree_id == tree_id);
        if (tracing_anticonditions == nullptr)
        {
          tracing_anticonditions = anticondition_updates;
          if (unpack_tracing_references)
            tracing_anticonditions->unpack_references();
        }
        else
        {
          anticondition_updates->merge(*tracing_anticonditions);
          if (unpack_tracing_references)
            anticondition_updates->unpack_references();
          delete anticondition_updates;
        }
      }
      if (postcondition_updates != nullptr)
      {
        legion_assert(postcondition_updates->owner_did == did);
        legion_assert(postcondition_updates->expression == set_expr);
        legion_assert(postcondition_updates->tree_id == tree_id);
        if (tracing_postconditions == nullptr)
        {
          tracing_postconditions = postcondition_updates;
          if (unpack_tracing_references)
            tracing_postconditions->unpack_references();
        }
        else
        {
          postcondition_updates->merge(*tracing_postconditions);
          if (unpack_tracing_references)
            postcondition_updates->unpack_references();
          delete postcondition_updates;
        }
      }
      if (dirty_updates != nullptr)
      {
        if (tracing_dirty_fields == nullptr)
        {
          tracing_dirty_fields = dirty_updates;
          for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   dirty_updates->begin();
               it != dirty_updates->end(); it++)
            it->first->add_nested_expression_reference(did);
        }
        else
        {
          for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                   dirty_updates->begin();
               it != dirty_updates->end(); it++)
            if (tracing_dirty_fields->insert(it->first, it->second))
              it->first->add_nested_expression_reference(did);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSetCloneRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet* set =
          runtime->find_or_request_equivalence_set(did, ready);
      DistributedID target;
      derez.deserialize(target);
      AddressSpaceID target_space;
      derez.deserialize(target_space);
      IndexSpaceExpression* target_expr =
          IndexSpaceExpression::unpack_expression(derez, source);
      IndexSpaceExpression* overlap =
          IndexSpaceExpression::unpack_expression(derez, source);
      FieldMask mask;
      derez.deserialize(mask);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      bool record_invalidate;
      derez.deserialize<bool>(record_invalidate);
      std::vector<RtEvent> applied_events;
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      // Add a reference to make sure we don't race with sending the response
      set->add_base_resource_ref(RUNTIME_REF);
      if (target_space == runtime->address_space)
      {
        // We've been sent back to the owner node
        EquivalenceSet* dst =
            runtime->find_or_request_equivalence_set(target, ready);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        dst->clone_from(set, mask, overlap, record_invalidate, applied_events);
      }
      else
        set->clone_to_remote(
            target, target_space, target_expr, overlap, mask, applied_events,
            record_invalidate);
      if (!applied_events.empty())
        Runtime::trigger_event(
            done_event, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(done_event);
      if (set->remove_base_resource_ref(RUNTIME_REF))
        delete set;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSetCloneResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet* set =
          runtime->find_or_request_equivalence_set(did, ready);
      AddressSpaceID source;
      derez.deserialize(source);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      std::vector<RtEvent> applied_events;
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      set->unpack_state_and_apply(
          derez, source, applied_events, true /*forward to owner*/);
      if (!applied_events.empty())
        Runtime::trigger_event(
            done_event, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSetCaptureRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet* set =
          runtime->find_or_request_equivalence_set(did, ready);
      PhysicalTemplate* target;
      derez.deserialize(target);
      AddressSpaceID target_space;
      derez.deserialize(target_space);
      unsigned parent_req_index;
      derez.deserialize(parent_req_index);
      std::atomic<unsigned>* result;
      derez.deserialize(result);
      RtUserEvent ready_event;
      derez.deserialize(ready_event);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      set->capture_trace_conditions(
          target, target_space, parent_req_index, result, ready_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSetCaptureResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet* set =
          runtime->find_or_request_equivalence_set(did, ready);
      PhysicalTemplate* target;
      derez.deserialize(target);
      unsigned parent_req_index;
      derez.deserialize(parent_req_index);
      RegionTreeID tree_id;
      derez.deserialize(tree_id);
      std::atomic<unsigned>* result;
      derez.deserialize(result);
      TraceViewSet* previews = nullptr;
      TraceViewSet* antiviews = nullptr;
      TraceViewSet* postviews = nullptr;
      std::set<RtEvent> ready_events;
      size_t num_previews;
      derez.deserialize(num_previews);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      if (num_previews > 0)
      {
        previews = new TraceViewSet(
            set->context, set->did, set->set_expr, set->tree_id);
        previews->unpack(derez, num_previews, source, ready_events);
      }
      size_t num_antiviews;
      derez.deserialize(num_antiviews);
      if (num_antiviews > 0)
      {
        antiviews = new TraceViewSet(
            set->context, set->did, set->set_expr, set->tree_id);
        antiviews->unpack(derez, num_antiviews, source, ready_events);
      }
      size_t num_postviews;
      derez.deserialize(num_postviews);
      if (num_postviews > 0)
      {
        postviews = new TraceViewSet(
            set->context, set->did, set->set_expr, set->tree_id);
        postviews->unpack(derez, num_postviews, source, ready_events);
      }
      size_t num_dirty_exprs;
      derez.deserialize(num_dirty_exprs);
      local::FieldMaskMap<IndexSpaceExpression> unique_dirty_exprs;
      for (unsigned idx = 0; idx < num_dirty_exprs; idx++)
      {
        IndexSpaceExpression* expr =
            IndexSpaceExpression::unpack_expression(derez, source);
        FieldMask mask;
        derez.deserialize(mask);
        unique_dirty_exprs.insert(expr, mask);
      }
      RtUserEvent done_event;
      derez.deserialize(done_event);
      legion_assert(done_event.exists());
      // Wait for the views to be ready before recording them
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      if (previews != nullptr)
        previews->unpack_references();
      if (antiviews != nullptr)
        antiviews->unpack_references();
      if (postviews != nullptr)
        postviews->unpack_references();
      target->receive_trace_conditions(
          previews, antiviews, postviews, unique_dirty_exprs, parent_req_index,
          tree_id, result);
      Runtime::trigger_event(done_event);
      if (previews != nullptr)
        delete previews;
      if (antiviews != nullptr)
        delete antiviews;
      if (postviews != nullptr)
        delete postviews;
    }

    /////////////////////////////////////////////////////////////
    // Equivalence Set Tracker
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    EqSetTracker::EqSetTracker(LocalLock& lock)
      : tracker_lock(lock), pending_equivalence_sets(nullptr),
        created_equivalence_sets(nullptr), equivalence_sets_ready(nullptr),
        waiting_infos(nullptr), creation_requests(nullptr),
        creation_rectangles(nullptr), creation_sources(nullptr),
        remaining_responses(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    EqSetTracker::~EqSetTracker(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(equivalence_sets.empty());
      legion_assert(pending_equivalence_sets == nullptr);
      legion_assert(created_equivalence_sets == nullptr);
      legion_assert(equivalence_sets_ready == nullptr);
      legion_assert(waiting_infos == nullptr);
      legion_assert(current_subscriptions.empty());
      legion_assert(!pending_invalidations);
      legion_assert(creation_requests == nullptr);
      legion_assert(creation_rectangles == nullptr);
      legion_assert(creation_sources == nullptr);
      legion_assert(remaining_responses == nullptr);
    }

    //--------------------------------------------------------------------------
    void EqSetTracker::record_equivalence_sets(
        InnerContext* context, const FieldMask& mask,
        op::FieldMaskMap<EquivalenceSet>& eq_sets,
        op::FieldMaskMap<EqKDTree>& to_create,
        op::map<EqKDTree*, Domain>& creation_rects,
        op::map<EquivalenceSet*, op::map<Domain, FieldMask>>& creation_srcs,
        op::FieldMaskMap<EqKDTree>& new_subscriptions, unsigned new_references,
        AddressSpaceID source, unsigned total_responses,
        std::vector<RtEvent>& ready_events,
        const CollectiveMapping& target_mapping,
        const std::vector<EqSetTracker*>& targets,
        const AddressSpaceID creation_target_space)
    //--------------------------------------------------------------------------
    {
      legion_assert(!targets.empty());
      legion_assert(total_responses > 0);
      legion_assert(target_mapping.size() == targets.size());
      legion_assert(target_mapping.contains(creation_target_space));
      legion_assert(target_mapping.contains(runtime->address_space));
      legion_assert(
          this == targets[target_mapping.find_index(runtime->address_space)]);
      legion_assert(
          (creation_target_space == runtime->address_space) ||
          (target_mapping.size() > 1));
      // If there are multiple targets then we only want to perform any
      // creations on the first target and then we'll broadcast the results
      // out to any other targets when we create the equivalence sets
      if (!to_create.empty() &&
          (runtime->address_space != creation_target_space))
      {
        to_create.clear();
        creation_rects.clear();
        creation_srcs.clear();
      }
      op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>> create_now;
      op::map<Domain, FieldMask> create_now_rectangles;
      op::map<EquivalenceSet*, op::map<Domain, FieldMask>> create_now_sources;
      // If we have just one response, we can just move things over
      if ((total_responses == 1) && !to_create.empty())
      {
        for (const std::pair<EqKDTree* const, Domain>& it : creation_rects)
        {
          legion_assert(to_create.find(it.first) != to_create.end());
          op::map<Domain, FieldMask>::iterator finder =
              create_now_rectangles.find(it.second);
          if (finder != create_now_rectangles.end())
            finder->second |= to_create[it.first];
          else
            create_now_rectangles[it.second] = to_create[it.first];
        }
        create_now[source].swap(to_create);
        create_now_sources.swap(creation_srcs);
      }
      {
        AutoLock t_lock(tracker_lock);
#ifdef LEGION_DEBUG
        // Sanity check: creations and invalidations can race due to
        // false aliasing in the EqKD tree (independent operations on
        // different points sharing a KD-tree node with overlapping
        // field masks). Any such overlap must be covered by an entry
        // in equivalence_sets_ready so that finalize_equivalence_sets
        // will detect it and recompute.
        if (!!pending_invalidations)
        {
          FieldMask creation_fields;
          if (!create_now_rectangles.empty())
          {
            for (op::map<Domain, FieldMask>::const_iterator it =
                     create_now_rectangles.begin();
                 it != create_now_rectangles.end(); it++)
              creation_fields |= it->second;
          }
          else
            creation_fields = to_create.get_valid_mask();
          const FieldMask overlap = pending_invalidations & creation_fields;
          if (!!overlap)
          {
            // The overlap is only safe if equivalence_sets_ready covers
            // it, ensuring finalize_equivalence_sets will handle cleanup
            legion_assert(equivalence_sets_ready != nullptr);
            FieldMask covered;
            for (shrt::map<RtUserEvent, FieldMask>::const_iterator it =
                     equivalence_sets_ready->begin();
                 it != equivalence_sets_ready->end(); it++)
              covered |= it->second;
            legion_assert(!(overlap - covered));
          }
        }
#endif
        // Record pending equivalence sets
        if (!eq_sets.empty())
        {
          if (pending_equivalence_sets == nullptr)
            pending_equivalence_sets = new shrt::FieldMaskMap<EquivalenceSet>();
          for (op::FieldMaskMap<EquivalenceSet>::const_iterator it =
                   eq_sets.begin();
               it != eq_sets.end(); it++)
            pending_equivalence_sets->insert(it->first, it->second);
        }
        if (new_references > 0)
          add_subscription_reference(new_references);
        // Record subscription owners
        if (!new_subscriptions.empty())
          record_subscriptions(source, new_subscriptions);
        if (!to_create.empty())
          // Even though we're going to make these equivalence sets, we'll
          // still effectively be recorded as subscribers of these nodes
          // once we're done with the creation
          record_subscriptions(source, to_create);
        if (!create_now.empty())
          // Also handle the case where we already swapped to_create into
          // the create_now data structure
          record_subscriptions(source, create_now[source]);
        // Record any creations that we need to perform
        if (total_responses > 1)
        {
          if (!to_create.empty())
            record_creation_sets(
                to_create, creation_rects, source, creation_srcs);
          // Now see which fields are done
          FieldMask remaining = mask;
          if (remaining_responses != nullptr)
          {
            for (shrt::map<unsigned, FieldMask>::iterator it =
                     remaining_responses->begin();
                 it != remaining_responses->end();
                 /*nothing*/)
            {
              const FieldMask overlap = remaining & it->second;
              if (!overlap)
              {
                it++;
                continue;
              }
              if (it->first > 1)  // Update it lower down the entries
                (*remaining_responses)[it->first - 1] |= overlap;
              else
                // We've seen the last response for these fields
                // so we can now merge things over to be handled
                extract_creation_sets(
                    overlap, create_now, create_now_rectangles,
                    create_now_sources);
              it->second -= overlap;
              if (!it->second)
              {
                shrt::map<unsigned, FieldMask>::iterator to_delete = it++;
                remaining_responses->erase(to_delete);
              }
              else
                it++;
              remaining -= overlap;
              if (!remaining)
                break;
            }
            if (remaining_responses->empty())
            {
              delete remaining_responses;
              remaining_responses = nullptr;
            }
          }
          if (!!remaining)
          {
            if (remaining_responses == nullptr)
              remaining_responses = new shrt::map<unsigned, FieldMask>();
            (*remaining_responses)[total_responses - 1] |= remaining;
          }
        }
      }
      // See if we have any equivalence sets for us to create right now
      if (!create_now.empty())
        create_new_equivalence_sets(
            context, ready_events, create_now, create_now_rectangles,
            create_now_sources, target_mapping, targets);
    }

    //--------------------------------------------------------------------------
    void EqSetTracker::record_output_subscriptions(
        AddressSpaceID source, local::FieldMaskMap<EqKDTree>& new_subscriptions)
    //--------------------------------------------------------------------------
    {
      legion_assert(!new_subscriptions.empty());
      AutoLock t_lock(tracker_lock);
      lng::FieldMaskMap<EqKDTree>& subscriptions =
          current_subscriptions[source];
      for (local::FieldMaskMap<EqKDTree>::const_iterator it =
               new_subscriptions.begin();
           it != new_subscriptions.end(); it++)
      {
        legion_assert(
            (subscriptions.find(it->first) == subscriptions.end()) ||
            (subscriptions.find(it->first)->second * it->second));
        subscriptions.insert(it->first, it->second);
      }
    }

    //--------------------------------------------------------------------------
    void EqSetTracker::record_subscriptions(
        AddressSpaceID source, const FieldMapView<EqKDTree>& new_subscriptions)
    //--------------------------------------------------------------------------
    {
      legion_assert(!new_subscriptions.empty());
      lng::FieldMaskMap<EqKDTree>& subscriptions =
          current_subscriptions[source];
      for (FieldMapView<EqKDTree>::const_iterator it =
               new_subscriptions.begin();
           it != new_subscriptions.end(); it++)
      {
        legion_assert(
            (subscriptions.find(it->first) == subscriptions.end()) ||
            (subscriptions.find(it->first)->second * it->second));
        subscriptions.insert(it->first, it->second);
      }
    }

    //--------------------------------------------------------------------------
    void EqSetTracker::record_creation_sets(
        op::FieldMaskMap<EqKDTree>& to_create,
        op::map<EqKDTree*, Domain>& creation_rects, AddressSpaceID source,
        op::map<EquivalenceSet*, op::map<Domain, FieldMask>>& creation_srcs)
    //--------------------------------------------------------------------------
    {
      // Lock held by caller
      // Save these into the creation requests
      for (op::map<EqKDTree*, Domain>::const_iterator it =
               creation_rects.begin();
           it != creation_rects.end(); it++)
      {
        legion_assert(to_create.find(it->first) != to_create.end());
        if (creation_rectangles != nullptr)
        {
          op::map<Domain, FieldMask>::iterator finder =
              creation_rectangles->find(it->second);
          if (finder != creation_rectangles->end())
            finder->second |= to_create[it->first];
          else
            (*creation_rectangles)[it->second] = to_create[it->first];
        }
        else
        {
          creation_rectangles = new op::map<Domain, FieldMask>();
          (*creation_rectangles)[it->second] = to_create[it->first];
        }
      }
      if (creation_requests == nullptr)
        creation_requests =
            new op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>();
      op::FieldMaskMap<EqKDTree>& requests = (*creation_requests)[source];
      if (!requests.empty())
      {
        for (const std::pair<EqKDTree* const, FieldMask>& it : to_create)
          requests.insert(it.first, it.second);
      }
      else
        requests.swap(to_create);
      // Save the creation sources
      if (creation_sources != nullptr)
      {
        for (op::map<EquivalenceSet*, op::map<Domain, FieldMask>>::iterator
                 sit = creation_srcs.begin();
             sit != creation_srcs.end(); sit++)
        {
          std::map<EquivalenceSet*, op::map<Domain, FieldMask>>::iterator
              finder = creation_sources->find(sit->first);
          if (finder != creation_sources->end())
          {
            for (op::map<Domain, FieldMask>::const_iterator it =
                     sit->second.begin();
                 it != sit->second.end(); it++)
              finder->second[it->first] |= it->second;
          }
          else
            (*creation_sources)[sit->first].swap(sit->second);
        }
      }
      else
      {
        creation_sources =
            new op::map<EquivalenceSet*, op::map<Domain, FieldMask>>();
        creation_sources->swap(creation_srcs);
      }
    }

    //--------------------------------------------------------------------------
    void EqSetTracker::extract_creation_sets(
        const FieldMask& mask,
        op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>& create_now,
        op::map<Domain, FieldMask>& create_now_rectangles,
        op::map<EquivalenceSet*, op::map<Domain, FieldMask>>&
            create_now_sources)
    //--------------------------------------------------------------------------
    {
      // Lock held by caller
      legion_assert(equivalence_sets_ready != nullptr);
      const size_t outstanding_requests = equivalence_sets_ready->size();
      legion_assert(outstanding_requests > 0);
      if (outstanding_requests > 1)
      {
        // Pull out the entries just for our fields
        if (creation_requests != nullptr)
        {
          for (op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>::iterator
                   cit = creation_requests->begin();
               cit != creation_requests->end();
               /*nothing*/)
          {
            if (mask * cit->second.get_valid_mask())
              cit++;
            else if (mask == cit->second.get_valid_mask())
            {
              create_now[cit->first].swap(cit->second);
              op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>::iterator
                  to_delete = cit++;
              creation_requests->erase(to_delete);
            }
            else
            {
              std::vector<EqKDTree*> to_delete;
              for (op::FieldMaskMap<EqKDTree>::iterator it =
                       cit->second.begin();
                   it != cit->second.end(); it++)
              {
                const FieldMask overlap = mask & it->second;
                if (!overlap)
                  continue;
                create_now[cit->first].insert(it->first, overlap);
                it.filter(overlap);
                if (!it->second)
                  to_delete.emplace_back(it->first);
              }
              for (EqKDTree* it : to_delete) cit->second.erase(it);
              if (cit->second.empty())
              {
                op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>::iterator
                    to_delete = cit++;
                creation_requests->erase(to_delete);
              }
              else
              {
                cit->second.tighten_valid_mask();
                cit++;
              }
            }
          }
          if (creation_requests->empty())
          {
            delete creation_requests;
            creation_requests = nullptr;
          }
        }
        if (creation_rectangles != nullptr)
        {
          for (op::map<Domain, FieldMask>::iterator it =
                   creation_rectangles->begin();
               it != creation_rectangles->end();
               /*nothing*/)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
            {
              it++;
              continue;
            }
            create_now_rectangles[it->first] = overlap;
            it->second -= overlap;
            if (!it->second)
            {
              op::map<Domain, FieldMask>::iterator to_delete = it++;
              creation_rectangles->erase(to_delete);
            }
            else
              it++;
          }
          if (creation_rectangles->empty())
          {
            delete creation_rectangles;
            creation_rectangles = nullptr;
          }
        }
        if (creation_sources != nullptr)
        {
          for (op::map<EquivalenceSet*, op::map<Domain, FieldMask>>::iterator
                   sit = creation_sources->begin();
               sit != creation_sources->end();
               /*nothing*/)
          {
            for (op::map<Domain, FieldMask>::iterator it = sit->second.begin();
                 it != sit->second.end();
                 /*nothing*/)
            {
              const FieldMask overlap = mask & it->second;
              if (!!overlap)
              {
                create_now_sources[sit->first][it->first] = overlap;
                it->second -= overlap;
                if (!it->second)
                {
                  op::map<Domain, FieldMask>::iterator to_delete = it++;
                  sit->second.erase(to_delete);
                }
                else
                  it++;
              }
              else
                it++;
            }
            if (sit->second.empty())
            {
              op::map<EquivalenceSet*, op::map<Domain, FieldMask>>::iterator
                  to_delete = sit++;
              creation_sources->erase(to_delete);
            }
            else
              sit++;
          }
          if (creation_sources->empty())
          {
            delete creation_sources;
            creation_sources = nullptr;
          }
        }
      }
      else
      {
        // If we're just doing one compute call then we
        // know we're going to do this for everything
        if (creation_requests != nullptr)
        {
          creation_requests->swap(create_now);
          delete creation_requests;
          creation_requests = nullptr;
        }
        if (creation_rectangles != nullptr)
        {
          creation_rectangles->swap(create_now_rectangles);
          delete creation_rectangles;
          creation_rectangles = nullptr;
        }
        if (creation_sources != nullptr)
        {
          creation_sources->swap(create_now_sources);
          delete creation_sources;
          creation_sources = nullptr;
        }
      }
    }

    //--------------------------------------------------------------------------
    EqSetTracker::SourceState::~SourceState(void)
    //--------------------------------------------------------------------------
    {
      if ((source_expr != nullptr) &&
          source_expr->remove_base_expression_reference(DISJOINT_COMPLETE_REF))
        delete source_expr;
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* EqSetTracker::SourceState::get_expression(void) const
    //--------------------------------------------------------------------------
    {
      return source_expr;
    }

    //--------------------------------------------------------------------------
    void EqSetTracker::SourceState::set_expression(IndexSpaceExpression* expr)
    //--------------------------------------------------------------------------
    {
      legion_assert(source_expr == nullptr);
      source_expr = expr;
      source_expr->add_base_expression_reference(DISJOINT_COMPLETE_REF);
    }

    //--------------------------------------------------------------------------
    void EqSetTracker::create_new_equivalence_sets(
        InnerContext* context, std::vector<RtEvent>& ready_events,
        op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>& create_now,
        op::map<Domain, FieldMask>& create_now_rectangles,
        op::map<EquivalenceSet*, op::map<Domain, FieldMask>>& creation_srcs,
        const CollectiveMapping& target_mapping,
        const std::vector<EqSetTracker*>& targets)
    //--------------------------------------------------------------------------
    {
      // Compute the rectangle sets for all the source equivalence sets
      // Also track which source equivalence sets have a unique set of fields
      // as we can use those to check for dominating the new set of rectangles
      // so we might be able to skip making a new equivalence set
      local::FieldMaskMap<EquivalenceSet> unique_sources;
      local::map<EquivalenceSet*, local::list<SourceState>> set_sources;
      {
        FieldMask multiple_sources;
        for (op::map<EquivalenceSet*, op::map<Domain, FieldMask>>::
                 const_iterator eit = creation_srcs.begin();
             eit != creation_srcs.end(); eit++)
        {
          local::list<SourceState>& src_rects = set_sources[eit->first];
          compute_field_sets(FieldMask(), MapView(eit->second), src_rects);
          FieldMask src_fields;
          for (const SourceState& it : src_rects) src_fields |= it.set_mask;
          if (!!multiple_sources)
          {
            src_fields -= multiple_sources;
            if (!src_fields)
              continue;
          }
          // Now deduplicate with prior fields
          if (!(src_fields * unique_sources.get_valid_mask()))
          {
            std::vector<EquivalenceSet*> to_delete;
            for (local::FieldMaskMap<EquivalenceSet>::iterator it =
                     unique_sources.begin();
                 it != unique_sources.end(); it++)
            {
              const FieldMask overlap = src_fields & it->second;
              if (!overlap)
                continue;
              multiple_sources |= overlap;
              it.filter(overlap);
              if (!it->second)
                to_delete.emplace_back(it->first);
              src_fields -= overlap;
              if (!src_fields)
                break;
            }
            for (EquivalenceSet* it : to_delete) unique_sources.erase(it);
            if (!src_fields)
              continue;
          }
          // We only allow the unique source optimization when the
          // equivalence set comes from the same context. If it
          // doesn't then we don't have any way to scope its
          // invalidation set down to just this set of points safely
          if (eit->first->context->get_depth() == context->get_depth())
            unique_sources.insert(eit->first, src_fields);
          else
            multiple_sources |= src_fields;
        }
      }
      // Sort the rectangles into field sets and make an equivlaence set
      // for each of them if their sources are not the same
      local::list<FieldSet<Domain>> rectangle_sets;
      // In release mode we just leave the unverse mask empty as that tells
      // the field sets routine to just make sets for the represented field
      // In debug mode, we populate the universe mask to make sure it aligns
      // with all the kd-nodes that we're supposed to target
      FieldMask universe_mask;
#ifdef LEGION_DEBUG
      for (op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>::const_iterator
               it = create_now.begin();
           it != create_now.end(); it++)
        universe_mask |= it->second.get_valid_mask();
#endif
      compute_field_sets(
          universe_mask, MapView(create_now_rectangles), rectangle_sets);
      shrt::FieldMaskMap<EquivalenceSet> created_sets;
      IndexSpaceExpression* tracker_expr = get_tracker_expression();
      const ReferenceSource ref_kind = get_reference_source_kind();
      for (local::list<FieldSet<Domain>>::iterator rit = rectangle_sets.begin();
           rit != rectangle_sets.end(); rit++)
      {
        legion_assert(!rit->elements.empty());
        // Check for the case where we have an existing equivalence set
        // that already has all the data that we need in which case there
        // is no point in making a new one and copying the data if we can
        // just reuse the existing one
        if (!(rit->set_mask * unique_sources.get_valid_mask()) &&
            check_for_congruent_source_equivalence_sets(
                *rit, ready_events, created_sets, unique_sources, create_now,
                set_sources, target_mapping, targets))
          continue;
        // First we need an expression for this equivalence set
        IndexSpaceExpression* expr =
            tracker_expr->create_from_rectangles(rit->elements);
        // Extract the things that we need to notify about this set
        op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>> to_notify;
        if (rectangle_sets.size() == 1)
        {
          to_notify.swap(create_now);
          // But swap our local space back out if there is one
          op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>::iterator finder =
              to_notify.find(runtime->address_space);
          if (finder != to_notify.end())
          {
            create_now[runtime->address_space].swap(finder->second);
            to_notify.erase(finder);
          }
        }
        else
          extract_remote_notifications(
              rit->set_mask, runtime->address_space, create_now, to_notify);
        // Next compute the CollectiveMapping for the equivalence set
        std::vector<AddressSpaceID> spaces;
        spaces.reserve(to_notify.size());
        for (op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>::const_iterator
                 it = to_notify.begin();
             it != to_notify.end(); it++)
          spaces.emplace_back(it->first);
        // Also add in any spaces for any targets that we need to send to also
        bool need_sort = false;
        for (unsigned idx = 0; idx < target_mapping.size(); idx++)
        {
          const AddressSpace target_space = target_mapping[idx];
          if (to_notify.find(target_space) != to_notify.end())
            continue;
          spaces.emplace_back(target_space);
          need_sort = true;
        }
        if (need_sort)
          std::sort(spaces.begin(), spaces.end());
        CollectiveMapping* mapping = nullptr;
        if (spaces.size() > 1)
        {
          mapping =
              new CollectiveMapping(spaces, runtime->legion_collective_radix);
          mapping->add_reference();
        }
        // Do a check to see if we already have an existing equivalence set
        // that is congruent with these points but aren't used for the fields
        // so we can reuse it if possible
        bool set_created = false;
        EquivalenceSet* set = find_congruent_existing_equivalence_set(
            expr, rit->set_mask, created_sets, context);
        if (set == nullptr)
        {
          const DistributedID did = runtime->get_available_distributed_id();
          set = new EquivalenceSet(
              did, runtime->address_space /*logical owner*/, expr,
              get_region_tree_id(), context, true /*register*/, mapping,
              (target_mapping.size() > 1));
          if (created_sets.insert(set, rit->set_mask))
            set->add_base_gc_ref(ref_kind);
          set_created = true;
        }
        // Clone any meta-data from the source equivalence sets to
        // bring this new equivalence set up to date
        RtEvent ready;
        if (!set_sources.empty())
        {
          ready = initialize_new_equivalence_set(
              set, rit->set_mask, !set_created, set_sources);
          if (ready.exists())
            ready_events.emplace_back(ready);
        }
        // Notify any equivalence set kd tree nodes about the new set
        if (mapping != nullptr)
        {
          // Broadcast this out to all the spaces and have them notify
          // their EqKDTree objects
          const AddressSpaceID local_space = runtime->address_space;
          std::vector<AddressSpaceID> children;
          mapping->get_children(local_space, local_space, children);
          legion_assert(!children.empty());
          if (set_created)
          {
            // See if this is an index space node for easy packing
            // otherwise we're going to pack all the rectangles so
            // we can make an expression on the remote nodes
            IndexSpaceNode* node = nullptr;
            if (expr == tracker_expr)
              node = dynamic_cast<IndexSpaceNode*>(expr);
            for (const AddressSpaceID& child : children)
            {
              const RtUserEvent notified = Runtime::create_rt_user_event();
              EquivalenceSetCreation rez;
              {
                RezCheck z(rez);
                rez.serialize(this);
                rez.serialize(set->did);
                if (node == nullptr)
                {
                  rez.serialize(IndexSpace::NO_SPACE);
                  rez.serialize(rit->elements.size());
                  for (const Domain& it : rit->elements) rez.serialize(it);
                }
                else
                  rez.serialize(node->handle);
                rez.serialize(set->tree_id);
                context->pack_inner_context(rez);
                mapping->pack(rez);
                rez.serialize(ready);
                rez.serialize<size_t>(to_notify.size());
                for (op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>::
                         const_iterator nit = to_notify.begin();
                     nit != to_notify.end(); nit++)
                {
                  rez.serialize(nit->first);
                  rez.serialize(nit->second.size());
                  for (op::FieldMaskMap<EqKDTree>::const_iterator it =
                           nit->second.begin();
                       it != nit->second.end(); it++)
                  {
                    rez.serialize(it->first);
                    rez.serialize(it->second);
                  }
                }
                // Also pack up any targets
                target_mapping.pack(rez);
                for (unsigned idx = 0; idx < targets.size(); idx++)
                  rez.serialize(targets[idx]);
                rez.serialize(rit->set_mask);
                rez.serialize(notified);
              }
              rez.dispatch(child);
              ready_events.emplace_back(notified);
            }
          }
          else
          {
            for (const AddressSpaceID& child : children)
            {
              const RtUserEvent notified = Runtime::create_rt_user_event();
              EquivalenceSetReuse rez;
              {
                RezCheck z(rez);
                rez.serialize(set->did);
                rez.serialize(this);
                rez.serialize(local_space);
                mapping->pack(rez);
                rez.serialize(ready);
                rez.serialize<size_t>(to_notify.size());
                for (op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>::
                         const_iterator nit = to_notify.begin();
                     nit != to_notify.end(); nit++)
                {
                  rez.serialize(nit->first);
                  rez.serialize(nit->second.size());
                  for (op::FieldMaskMap<EqKDTree>::const_iterator it =
                           nit->second.begin();
                       it != nit->second.end(); it++)
                  {
                    rez.serialize(it->first);
                    rez.serialize(it->second);
                  }
                }
                // Also pack up any targets
                target_mapping.pack(rez);
                for (unsigned idx = 0; idx < targets.size(); idx++)
                  rez.serialize(targets[idx]);
                rez.serialize(rit->set_mask);
                rez.serialize(notified);
              }
              rez.dispatch(child);
              ready_events.emplace_back(notified);
            }
          }
        }
        // Notify any local EqKDTree objects
        op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>::iterator finder =
            create_now.find(runtime->address_space);
        if (finder != create_now.end())
        {
          std::vector<EqKDTree*> to_delete;
          for (op::FieldMaskMap<EqKDTree>::iterator it = finder->second.begin();
               it != finder->second.end(); it++)
          {
            const FieldMask overlap = rit->set_mask & it->second;
            if (!overlap)
              continue;
            it->first->record_equivalence_set(
                set, overlap, ready, target_mapping, targets);
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(it->first);
          }
          for (EqKDTree* const it : to_delete) finder->second.erase(it);
          if (finder->second.empty())
            create_now.erase(finder);
        }
        if ((mapping != nullptr) && mapping->remove_reference())
          delete mapping;
      }
      legion_assert(!created_sets.empty());
      // Retake the lock and record these pending equivalence sets
      AutoLock t_lock(tracker_lock);
      if (created_equivalence_sets == nullptr)
      {
        created_equivalence_sets = new shrt::FieldMaskMap<EquivalenceSet>();
        created_equivalence_sets->swap(created_sets);
      }
      else
      {
        for (shrt::FieldMaskMap<EquivalenceSet>::const_iterator it =
                 created_sets.begin();
             it != created_sets.end(); it++)
        {
          // Add the new set and remove any duplicate references
          // which should never lead to the deletion of the set
          // since we should already be holding a reference
          if (!created_equivalence_sets->insert(it->first, it->second) &&
              it->first->remove_base_gc_ref(ref_kind))
            std::abort();  // should never actaully delete the object
        }
      }
    }

    //--------------------------------------------------------------------------
    bool EqSetTracker::check_for_congruent_source_equivalence_sets(
        FieldSet<Domain>& dest, std::vector<RtEvent>& ready_events,
        shrt::FieldMaskMap<EquivalenceSet>& created_sets,
        local::FieldMaskMap<EquivalenceSet>& unique_sources,
        op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>& create_now,
        local::map<EquivalenceSet*, local::list<SourceState>>& set_sources,
        const CollectiveMapping& target_mapping,
        const std::vector<EqSetTracker*>& targets)
    //--------------------------------------------------------------------------
    {
      size_t destination_volume = 0;
      for (const Domain& it : dest.elements)
        destination_volume += it.get_volume();
      std::vector<EquivalenceSet*> to_remove;
      const AddressSpaceID local_space = runtime->address_space;
      const ReferenceSource ref_kind = get_reference_source_kind();
      for (local::FieldMaskMap<EquivalenceSet>::iterator eit =
               unique_sources.begin();
           eit != unique_sources.end(); eit++)
      {
        const FieldMask src_mask = dest.set_mask & eit->second;
        if (!src_mask)
          continue;
        std::map<EquivalenceSet*, local::list<SourceState>>::iterator
            source_finder = set_sources.find(eit->first);
        legion_assert(source_finder != set_sources.end());
        for (local::list<SourceState>::iterator sit =
                 source_finder->second.begin();
             sit != source_finder->second.end();
             /*nothing*/)
        {
          const FieldMask overlap = src_mask & sit->set_mask;
          if (!overlap)
          {
            sit++;
            continue;
          }
          if (sit->source_volume == 0)
          {
            legion_assert(!sit->elements.empty());
            for (const Domain& it : sit->elements)
              sit->source_volume += it.get_volume();
          }
          legion_assert(sit->source_volume <= destination_volume);
          legion_assert(
              sit->source_volume <= eit->first->set_expr->get_volume());
          if ((sit->source_volume == destination_volume) &&
              (sit->source_volume == eit->first->set_expr->get_volume()))
          {
            // This equivalence set is already valid for exactly the
            // point of the overlapping fields
            // Get the set of EqKDTree nodes that we need to notify
            op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>> to_notify;
            extract_remote_notifications(
                overlap, runtime->address_space, create_now, to_notify);
            std::vector<AddressSpaceID> spaces;
            spaces.reserve(to_notify.size());
            for (op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>::
                     const_iterator it = to_notify.begin();
                 it != to_notify.end(); it++)
              spaces.emplace_back(it->first);
            bool need_sort = false;
            for (unsigned idx = 0; idx < target_mapping.size(); idx++)
            {
              const AddressSpaceID target_space = target_mapping[idx];
              if (to_notify.find(target_space) != to_notify.end())
                continue;
              spaces.emplace_back(target_space);
              need_sort = true;
            }
            if (need_sort)
              std::sort(spaces.begin(), spaces.end());
            if (spaces.size() > 1)
            {
              CollectiveMapping mapping(
                  spaces, runtime->legion_collective_radix);
              std::vector<AddressSpaceID> children;
              mapping.get_children(local_space, local_space, children);
              for (const AddressSpaceID& child : children)
              {
                const RtUserEvent notified = Runtime::create_rt_user_event();
                EquivalenceSetReuse rez;
                {
                  RezCheck z(rez);
                  // Reference still held by at least one EqKDTree
                  // for which this is in the previous set
                  rez.serialize(eit->first->did);
                  rez.serialize(this);
                  rez.serialize(local_space);
                  mapping.pack(rez);
                  rez.serialize(RtEvent::NO_RT_EVENT);
                  rez.serialize<size_t>(to_notify.size());
                  for (op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>::
                           const_iterator nit = to_notify.begin();
                       nit != to_notify.end(); nit++)
                  {
                    rez.serialize(nit->first);
                    rez.serialize(nit->second.size());
                    for (op::FieldMaskMap<EqKDTree>::const_iterator it =
                             nit->second.begin();
                         it != nit->second.end(); it++)
                    {
                      legion_assert(!(it->second - overlap));
                      rez.serialize(it->first);
                      rez.serialize(it->second);
                    }
                  }
                  // Also pack information about targets
                  target_mapping.pack(rez);
                  for (unsigned idx = 0; idx < targets.size(); idx++)
                    rez.serialize(targets[idx]);
                  rez.serialize(overlap);
                  rez.serialize(notified);
                }
                rez.dispatch(child);
                ready_events.emplace_back(notified);
              }
            }
            // Check to see if there are any local notifications to perform
            op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>::iterator
                local_finder = create_now.find(local_space);
            if ((local_finder != create_now.end()) &&
                !(overlap * local_finder->second.get_valid_mask()))
            {
              std::vector<EqKDTree*> to_delete;
              for (op::FieldMaskMap<EqKDTree>::iterator it =
                       local_finder->second.begin();
                   it != local_finder->second.end(); it++)
              {
                const FieldMask local_overlap = overlap & it->second;
                if (!local_overlap)
                  continue;
                it->first->record_equivalence_set(
                    eit->first, local_overlap, RtEvent::NO_RT_EVENT,
                    target_mapping, targets);
                it.filter(local_overlap);
                if (!it->second)
                  to_delete.emplace_back(it->first);
              }
              if (to_delete.size() < local_finder->second.size())
              {
                for (EqKDTree* it : to_delete) local_finder->second.erase(it);
                local_finder->second.tighten_valid_mask();
              }
              else if (to_delete.size() == local_finder->second.size())
                create_now.erase(local_finder);
            }
            if (created_sets.insert(eit->first, overlap))
              eit->first->add_base_gc_ref(ref_kind);
            dest.set_mask -= overlap;
            sit->set_mask -= overlap;
            if (!sit->set_mask)
            {
              local::list<SourceState>::iterator to_delete = sit++;
              source_finder->second.erase(to_delete);
            }
            else
              sit++;
          }
          else
            sit++;
        }
        if (source_finder->second.empty())
          set_sources.erase(source_finder);
        eit.filter(src_mask);
        if (!eit->second)
          to_remove.emplace_back(eit->first);
      }
      for (EquivalenceSet* const it : to_remove) unique_sources.erase(it);
      return !dest.set_mask;
    }

    //--------------------------------------------------------------------------
    EquivalenceSet* EqSetTracker::find_congruent_existing_equivalence_set(
        IndexSpaceExpression* expr, const FieldMask& mask,
        shrt::FieldMaskMap<EquivalenceSet>& created_sets, InnerContext* context)
    //--------------------------------------------------------------------------
    {
      const int depth = context->get_depth();
      // Find the canonical expression for this expr and then we can use that
      // to compare against the canonical expression of each of the existing
      // equivalence sets to see if we can find a match
      const size_t volume = expr->get_volume();
      expr = expr->get_canonical_expression();
      // Need the lock since the equivalence set data structure might
      // change at the same time that we're iterating over it
      // We're just reading though so we don't need exclusive access
      const ReferenceSource ref_kind = get_reference_source_kind();
      AutoLock t_lock(tracker_lock, false /*exclusive*/);
      for (lng::FieldMaskMap<EquivalenceSet>::const_iterator it =
               equivalence_sets.begin();
           it != equivalence_sets.end(); it++)
      {
        // Only allow this optimization for equivalence sets made
        // in the same context, which we check by depth because we
        // still want to allow equivalence sets from the other
        // ReplicateContexts in the case of control replication
        if (depth != it->first->context->get_depth())
          continue;
        // If the volumes are not the same then the shapes can't be the same
        if (volume != it->first->set_expr->get_volume())
          continue;
        IndexSpaceExpression* set_expr =
            it->first->set_expr->get_canonical_expression();
        if (expr == set_expr)
        {
          // Found one, add it to the pending sets and add a reference
          // if its the first time we're reusing it
          if (created_sets.insert(it->first, mask))
            it->first->add_base_gc_ref(ref_kind);
          return it->first;
        }
      }
      return nullptr;
    }

    //--------------------------------------------------------------------------
    void EqSetTracker::extract_remote_notifications(
        const FieldMask& mask, AddressSpaceID local_space,
        op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>& create_now,
        op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>& to_notify)
    //--------------------------------------------------------------------------
    {
      for (op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>::iterator cit =
               create_now.begin();
           cit != create_now.end();
           /*nothing*/)
      {
        if (cit->first == local_space)
        {
          cit++;
          continue;
        }
        if (mask * cit->second.get_valid_mask())
        {
          cit++;
          continue;
        }
        op::FieldMaskMap<EqKDTree>& notify = to_notify[cit->first];
        if (mask != cit->second.get_valid_mask())
        {
          std::vector<EqKDTree*> to_delete;
          for (op::FieldMaskMap<EqKDTree>::iterator it = cit->second.begin();
               it != cit->second.end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            notify.insert(it->first, overlap);
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(it->first);
          }
          for (EqKDTree* const it : to_delete) cit->second.erase(it);
        }
        else
          notify.swap(cit->second);
        if (cit->second.empty())
        {
          op::map<AddressSpaceID, op::FieldMaskMap<EqKDTree>>::iterator
              to_delete = cit++;
          create_now.erase(to_delete);
        }
        else
          cit++;
      }
    }

    //--------------------------------------------------------------------------
    RtEvent EqSetTracker::initialize_new_equivalence_set(
        EquivalenceSet* target, const FieldMask& mask,
        bool filter_invalidations,
        local::map<EquivalenceSet*, local::list<SourceState>>& set_sources)
    //--------------------------------------------------------------------------
    {
      std::vector<RtEvent> ready_events;
      if (filter_invalidations)
      {
        RtUserEvent filtered;
        target->filter_partial_invalidations(mask, filtered);
        if (filtered.exists())
          ready_events.emplace_back(filtered);
      }
      for (local::map<EquivalenceSet*, local::list<SourceState>>::iterator eit =
               set_sources.begin();
           eit != set_sources.end();
           /*nothing*/)
      {
        for (local::list<SourceState>::iterator sit = eit->second.begin();
             sit != eit->second.end();
             /*nothing*/)
        {
          const FieldMask overlap = mask & sit->set_mask;
          if (!overlap)
          {
            sit++;
            continue;
          }
          // Sometimes the source can become the target when we find a
          // congruent equivalence set to use as the target, so we skip
          // the cloning of the equivalence set data in that case
          if (eit->first != target)
          {
            IndexSpaceExpression* expression = sit->get_expression();
            if (expression == nullptr)
            {
              IndexSpaceExpression* intersection =
                  runtime->intersect_index_spaces(
                      eit->first->set_expr, target->set_expr);
              if (intersection->get_volume() == target->set_expr->get_volume())
                intersection = target->set_expr;
              else if (
                  intersection->get_volume() ==
                  eit->first->set_expr->get_volume())
                intersection = eit->first->set_expr;
              expression = intersection->create_from_rectangles(sit->elements);
              sit->set_expression(expression);
            }
            // We only record the partial invalidation if we're cloning between
            // two equivalence sets in the same context. Note that because of
            // control replication you can't just check that both sets are from
            // the same equivalence set, but what you can do is check to see if
            // they are at the same depth in the task tree
            const bool record_invalidate =
                (target->context->get_depth() ==
                 eit->first->context->get_depth());
            target->clone_from(
                eit->first, overlap, expression, record_invalidate,
                ready_events);
          }
          sit->set_mask -= overlap;
          if (!sit->set_mask)
          {
            local::list<SourceState>::iterator to_delete = sit++;
            eit->second.erase(to_delete);
          }
          else
            sit++;
        }
        if (eit->second.empty())
        {
          std::map<EquivalenceSet*, local::list<SourceState>>::iterator
              to_delete = eit++;
          set_sources.erase(to_delete);
        }
        else
          eit++;
      }
      if (ready_events.empty())
        return RtEvent::NO_RT_EVENT;
      else
        return Runtime::merge_events(ready_events);
    }

    //--------------------------------------------------------------------------
    void EqSetTracker::record_pending_equivalence_set(
        EquivalenceSet* set, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(tracker_lock);
      if (pending_equivalence_sets == nullptr)
        pending_equivalence_sets = new shrt::FieldMaskMap<EquivalenceSet>();
      pending_equivalence_sets->insert(set, mask);
    }

    //--------------------------------------------------------------------------
    void EqSetTracker::finalize_equivalence_sets(
        RtUserEvent done_event, InnerContext* enclosing,
        InnerContext* outermost, unsigned parent_req_index,
        IndexSpaceExpression* expr, UniqueID opid)
    //--------------------------------------------------------------------------
    {
      const ReferenceSource ref_kind = get_reference_source_kind();
      {
        AutoLock t_lock(tracker_lock);
        legion_assert(equivalence_sets_ready != nullptr);
        shrt::map<RtUserEvent, FieldMask>::iterator finder =
            equivalence_sets_ready->find(done_event);
        legion_assert(finder != equivalence_sets_ready->end());
        // Check for the pending invalidations case. This occurs due to
        // false aliasing in the Equivalence Set KD tree, were we might
        // have found some equivalence sets for a subset of points for
        // this analysis, but some other invalidation came through for
        // an independent set of points which required refining the
        // node in the equivalence set KD tree. This will invalidate
        // the subscriptions so we'll need to recompute the equivalence
        // sets. Note that we're guaranteed that this is only false
        // aliasing because of the logical dependence analysis which
        // implies if there were true aliasing then there would have been
        // a mapping dependence that prevented this race.
        FieldMask invalidated = finder->second & pending_invalidations;
        while (!!invalidated)
        {
          // We have pending invalidations that we deferred until we had
          // seen all the responses for the compute_equivalence_sets call
          // We've now seen all the responses, so we need to do the actual
          // work of the invalidations now that we're in a consistent state
          // First invalidate any subscriptions
          local::map<AddressSpaceID, local::FieldMaskMap<EqKDTree>> to_cancel;
          find_cancellations(invalidated, to_cancel);
          // Then we can prune states out of the pending equivalence sets
          if ((pending_equivalence_sets != nullptr) &&
              !(invalidated * pending_equivalence_sets->get_valid_mask()))
          {
            if (!(pending_equivalence_sets->get_valid_mask() - invalidated))
            {
              // Invalidate everything so we can just delete the sets
              delete pending_equivalence_sets;
              pending_equivalence_sets = nullptr;
            }
            else
            {
              // Partial invalidation of only some fields
              std::vector<EquivalenceSet*> to_delete;
              for (shrt::FieldMaskMap<EquivalenceSet>::iterator it =
                       pending_equivalence_sets->begin();
                   it != pending_equivalence_sets->end(); it++)
              {
                it.filter(invalidated);
                if (!it->second)
                  to_delete.emplace_back(it->first);
              }
              if (!to_delete.empty())
              {
                for (EquivalenceSet* it : to_delete)
                  pending_equivalence_sets->erase(it);
              }
              legion_assert(!pending_equivalence_sets->empty());
              pending_equivalence_sets->tighten_valid_mask();
            }
          }
          // Also need to prune states out of the created equivalence sets
          if ((created_equivalence_sets != nullptr) &&
              !(invalidated * created_equivalence_sets->get_valid_mask()))
          {
            if (!(created_equivalence_sets->get_valid_mask() - invalidated))
            {
              // Invalidate everyhting so we can just delete things
              for (shrt::FieldMaskMap<EquivalenceSet>::const_iterator it =
                       created_equivalence_sets->begin();
                   it != created_equivalence_sets->end(); it++)
                if (it->first->remove_base_gc_ref(ref_kind))
                  delete it->first;
              delete created_equivalence_sets;
              created_equivalence_sets = nullptr;
            }
            else
            {
              std::vector<EquivalenceSet*> to_delete;
              for (shrt::FieldMaskMap<EquivalenceSet>::iterator it =
                       created_equivalence_sets->begin();
                   it != created_equivalence_sets->end(); it++)
              {
                it.filter(invalidated);
                if (!it->second)
                  to_delete.emplace_back(it->first);
              }
              if (!to_delete.empty())
              {
                for (EquivalenceSet* it : to_delete)
                {
                  created_equivalence_sets->erase(it);
                  if (it->remove_base_gc_ref(ref_kind))
                    delete it;
                }
              }
              legion_assert(!created_equivalence_sets->empty());
              created_equivalence_sets->tighten_valid_mask();
            }
          }
          pending_invalidations -= invalidated;
          t_lock.release();
          // Perform any cancellations now that we've released the lock
          if (!to_cancel.empty())
          {
            std::vector<RtEvent> cancelled_events;
            for (local::map<AddressSpaceID, local::FieldMaskMap<EqKDTree>>::
                     const_iterator it = to_cancel.begin();
                 it != to_cancel.end(); it++)
              cancel_subscriptions(
                  it->first, FieldMapView(it->second), &cancelled_events);
            // Make sure all the cancellations are done before we try
            // again so we don't double count references
            if (!cancelled_events.empty())
            {
              const RtEvent wait_on = Runtime::merge_events(cancelled_events);
              wait_on.wait();
            }
          }
          // Now redo the computation
          // Note that if there was an initial collective rendezvous to
          // compute these equivalence sets, we won't try to redo it here
          // because we can't guarantee convergence between the different
          // points as to whether they all saw the same invalidations at
          // the same time and therefore the rendezvous is no longer safe
          std::vector<EqSetTracker*> targets(1, this);
          std::vector<AddressSpaceID> target_spaces(1, runtime->address_space);
          const RtEvent ready = enclosing->compute_equivalence_sets(
              parent_req_index, targets, target_spaces, runtime->address_space,
              expr, invalidated);
          if (ready.exists() && !ready.has_triggered())
          {
            // Launch task to finalize the sets once they are ready
            LgFinalizeEqSetsArgs args(
                this, done_event, enclosing, outermost, parent_req_index, expr);
            runtime->issue_runtime_meta_task(
                args, LG_LATENCY_DEFERRED_PRIORITY, ready);
            // We did the continuation for this so there's nothing
            // more to do here
            return;
          }
          t_lock.reacquire();
          invalidated = finder->second & pending_invalidations;
        }
        // If there are any pending equivalence sets, move them into
        // the actual equivalence sets
        if ((pending_equivalence_sets != nullptr) &&
            !(finder->second * pending_equivalence_sets->get_valid_mask()))
        {
          if (!(pending_equivalence_sets->get_valid_mask() - finder->second))
          {
            // They're all ready so we can move them over
            for (shrt::FieldMaskMap<EquivalenceSet>::iterator it =
                     pending_equivalence_sets->begin();
                 it != pending_equivalence_sets->end(); it++)
              if (equivalence_sets.insert(it->first, it->second))
                it->first->add_base_gc_ref(ref_kind);
            delete pending_equivalence_sets;
            pending_equivalence_sets = nullptr;
          }
          else
          {
            std::vector<EquivalenceSet*> to_delete;
            for (shrt::FieldMaskMap<EquivalenceSet>::iterator it =
                     pending_equivalence_sets->begin();
                 it != pending_equivalence_sets->end(); it++)
            {
              // Find what fields are overlapping
              const FieldMask overlap = it->second & finder->second;
              if (!overlap)
                continue;
              if (equivalence_sets.insert(it->first, overlap))
                it->first->add_base_gc_ref(ref_kind);
              it.filter(overlap);
              if (!it->second)
                to_delete.emplace_back(it->first);
            }
            if (!to_delete.empty())
            {
              for (EquivalenceSet* it : to_delete)
                pending_equivalence_sets->erase(it);
            }
            legion_assert(!pending_equivalence_sets->empty());
            pending_equivalence_sets->tighten_valid_mask();
          }
        }
        // If there are any created equivalence sets, move them over too
        if ((created_equivalence_sets != nullptr) &&
            !(finder->second * created_equivalence_sets->get_valid_mask()))
        {
          if (!(created_equivalence_sets->get_valid_mask() - finder->second))
          {
            // All fields are ready so we can move them all over
            // Make sure to remove duplicate references
            for (shrt::FieldMaskMap<EquivalenceSet>::iterator it =
                     created_equivalence_sets->begin();
                 it != created_equivalence_sets->end(); it++)
              if (!equivalence_sets.insert(it->first, it->second) &&
                  it->first->remove_base_gc_ref(ref_kind))
                std::abort();  // should never delete the object
            delete created_equivalence_sets;
            created_equivalence_sets = nullptr;
          }
          else
          {
            std::vector<EquivalenceSet*> to_delete;
            for (shrt::FieldMaskMap<EquivalenceSet>::iterator it =
                     created_equivalence_sets->begin();
                 it != created_equivalence_sets->end(); it++)
            {
              // Find what fields are overlapping
              const FieldMask overlap = it->second & finder->second;
              if (!overlap)
                continue;
              if (overlap == it->second)
              {
                // Moving over all of the fields so we can move over
                // our reference to or deduplicate the reference if
                // the set already exists in the equivalence_sets
                if (!equivalence_sets.insert(it->first, it->second) &&
                    it->first->remove_base_gc_ref(ref_kind))
                  std::abort();  // should never delete the object
                to_delete.emplace_back(it->first);
              }
              else
              {
                // Just moving over some of the fields which means
                // we need to add a reference if this is the first
                if (equivalence_sets.insert(it->first, overlap))
                  it->first->add_base_gc_ref(ref_kind);
                it.filter(overlap);
              }
            }
            if (!to_delete.empty())
            {
              for (EquivalenceSet* it : to_delete)
                created_equivalence_sets->erase(it);
            }
            legion_assert(!created_equivalence_sets->empty());
            created_equivalence_sets->tighten_valid_mask();
          }
        }
        if (waiting_infos != nullptr)
        {
          std::vector<VersionInfo*> to_delete;
          for (shrt::FieldMaskMap<VersionInfo>::iterator wit =
                   waiting_infos->begin();
               wit != waiting_infos->end(); wit++)
          {
            const FieldMask info_overlap = wit->second & finder->second;
            if (!info_overlap)
              continue;
            record_equivalence_sets(wit->first, info_overlap);
            wit.filter(info_overlap);
            if (!wit->second)
              to_delete.emplace_back(wit->first);
          }
          if (!to_delete.empty())
          {
            if (to_delete.size() < waiting_infos->size())
            {
              for (VersionInfo* it : to_delete) waiting_infos->erase(it);
              waiting_infos->tighten_valid_mask();
            }
            else
            {
              delete waiting_infos;
              waiting_infos = nullptr;
            }
          }
        }
        // We can relax the mask for the equivalence sets here so we don't
        // recompute in the case that we are empty
        equivalence_sets.relax_valid_mask(finder->second);
        equivalence_sets_ready->erase(finder);
        if (equivalence_sets_ready->empty())
        {
          delete equivalence_sets_ready;
          equivalence_sets_ready = nullptr;
        }
      }
      // At this point we're done so we can trigger the done event
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    void EqSetTracker::record_equivalence_sets(
        VersionInfo* version_info, const FieldMask& mask) const
    //--------------------------------------------------------------------------
    {
      for (lng::FieldMaskMap<EquivalenceSet>::const_iterator it =
               equivalence_sets.begin();
           it != equivalence_sets.end(); it++)
      {
        const FieldMask overlap = it->second & mask;
        if (!overlap)
          continue;
        version_info->record_equivalence_set(it->first, overlap);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ComputeEquivalenceSetsPending::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet* set =
          runtime->find_or_request_equivalence_set(did, ready);
      size_t num_trackers;
      derez.deserialize(num_trackers);
      for (unsigned idx = 0; idx < num_trackers; idx++)
      {
        EqSetTracker* tracker;
        derez.deserialize(tracker);
        FieldMask mask;
        derez.deserialize(mask);
        tracker->record_pending_equivalence_set(set, mask);
      }
      RtUserEvent recorded_event;
      derez.deserialize(recorded_event);
      // Then pending sets can't be processed until the equivalence set is
      // actually ready so chaing these events here
      Runtime::trigger_event(recorded_event, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSetCreation::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      EqSetTracker* owner;
      derez.deserialize(owner);
      DistributedID did;
      derez.deserialize(did);
      IndexSpace handle;
      derez.deserialize(handle);
      std::vector<Domain> rectangles;
      if (!handle.exists())
      {
        size_t num_rectangles;
        derez.deserialize(num_rectangles);
        rectangles.resize(num_rectangles);
        for (unsigned idx = 0; idx < num_rectangles; idx++)
          derez.deserialize(rectangles[idx]);
      }
      RegionTreeID tid;
      derez.deserialize(tid);
      InnerContext* context = InnerContext::unpack_inner_context(derez);
      size_t num_spaces;
      derez.deserialize(num_spaces);
      CollectiveMapping* mapping = new CollectiveMapping(derez, num_spaces);
      RtEvent ready_event;
      derez.deserialize(ready_event);
      // Make sure that we'll know when this is triggered
      ready_event.subscribe();
      derez.deserialize(num_spaces);
      local::map<AddressSpaceID, local::FieldMaskMap<EqKDTree>> to_notify;
      for (unsigned idx1 = 0; idx1 < num_spaces; idx1++)
      {
        AddressSpaceID space;
        derez.deserialize(space);
        local::FieldMaskMap<EqKDTree>& trees = to_notify[space];
        size_t num_trees;
        derez.deserialize(num_trees);
        for (unsigned idx2 = 0; idx2 < num_trees; idx2++)
        {
          EqKDTree* tree;
          derez.deserialize(tree);
          FieldMask mask;
          derez.deserialize(mask);
          trees.insert(tree, mask);
        }
      }
      size_t num_targets;
      derez.deserialize(num_targets);
      const CollectiveMapping target_mapping(derez, num_targets);
      std::vector<EqSetTracker*> targets(num_targets);
      for (unsigned idx = 0; idx < num_targets; idx++)
        derez.deserialize(targets[idx]);
      FieldMask recording_mask;
      derez.deserialize(recording_mask);
      RtUserEvent notified_event;
      derez.deserialize(notified_event);

      // Send it off to any children nodes
      std::vector<RtEvent> notified_events;
      std::vector<AddressSpaceID> children;
      const AddressSpaceID root = runtime->determine_owner(did);
      mapping->get_children(root, runtime->address_space, children);
      if (!children.empty())
      {
        // Make this a broadcast tree of ready events so not everyone
        // is subscribing to the owner node
        const RtUserEvent local_ready = Runtime::create_rt_user_event();
        Runtime::trigger_event(local_ready, ready_event);
        for (const AddressSpaceID cit : children)
        {
          const RtUserEvent notified = Runtime::create_rt_user_event();
          EquivalenceSetCreation rez;
          {
            RezCheck z(rez);
            rez.serialize(owner);
            rez.serialize(did);
            rez.serialize(handle);
            if (!handle.exists())
            {
              rez.serialize<size_t>(rectangles.size());
              for (unsigned idx = 0; idx < rectangles.size(); idx++)
                rez.serialize(rectangles[idx]);
            }
            rez.serialize(tid);
            context->pack_inner_context(rez);
            mapping->pack(rez);
            rez.serialize(local_ready);
            rez.serialize<size_t>(to_notify.size());
            for (const std::pair<
                     const AddressSpaceID, local::FieldMaskMap<EqKDTree>>& nit :
                 to_notify)
            {
              rez.serialize(nit.first);
              rez.serialize(nit.second.size());
              for (const std::pair<EqKDTree* const, FieldMask>& it : nit.second)
              {
                rez.serialize(it.first);
                rez.serialize(it.second);
              }
            }
            target_mapping.pack(rez);
            for (unsigned idx = 0; idx < targets.size(); idx++)
              rez.serialize(targets[idx]);
            rez.serialize(recording_mask);
            rez.serialize(notified);
          }
          rez.dispatch(cit);
          notified_events.emplace_back(notified);
        }
      }
      local::map<AddressSpaceID, local::FieldMaskMap<EqKDTree>>::iterator
          local_finder = to_notify.find(runtime->address_space);
      EqSetTracker* local_target = nullptr;
      local::set<Domain> rectangles_set;
      // Check to see if we have a local target to notify here
      if (target_mapping.contains(runtime->address_space))
      {
        unsigned index = target_mapping.find_index(runtime->address_space);
        local_target = targets[index];
        if (!handle.exists() && (local_finder == to_notify.end()))
          rectangles_set.insert(rectangles.begin(), rectangles.end());
      }
      legion_assert(
          (local_finder != to_notify.end()) || (local_target != nullptr));
      // Make the equivalence set
      IndexSpaceExpression* expr =
          handle.exists() ?
              runtime->get_node(handle) :
          (local_finder != to_notify.end()) ?
              local_finder->second.begin()->first->create_from_rectangles(
                  rectangles) :
              local_target->get_tracker_expression()->create_from_rectangles(
                  rectangles_set);
      void* location =
          runtime->find_or_create_pending_collectable_location<EquivalenceSet>(
              did);
      EquivalenceSet* set = new (location) EquivalenceSet(
          did, root, expr, tid, context, false /*register now*/, mapping,
          (target_mapping.size() > 1));
      // Once construction is complete then we do the registration
      set->register_with_runtime();
      // Register it with any local trees
      if (local_finder != to_notify.end())
      {
        for (local::FieldMaskMap<EqKDTree>::const_iterator it =
                 local_finder->second.begin();
             it != local_finder->second.end(); it++)
          it->first->record_equivalence_set(
              set, it->second, ready_event, target_mapping, targets);
      }
      if (local_target != nullptr)
        local_target->record_pending_equivalence_set(set, recording_mask);
      // Trigger the event that we're done
      if (!notified_events.empty())
        Runtime::trigger_event(
            notified_event, Runtime::merge_events(notified_events));
      else
        Runtime::trigger_event(notified_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSetReuse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet* set =
          runtime->find_or_request_equivalence_set(did, ready);
      EqSetTracker* owner;
      derez.deserialize(owner);
      AddressSpaceID owner_space;
      derez.deserialize(owner_space);
      size_t num_spaces;
      derez.deserialize(num_spaces);
      CollectiveMapping mapping(derez, num_spaces);
      RtEvent ready_event;
      derez.deserialize(ready_event);
      size_t num_notifications;
      derez.deserialize(num_notifications);
      local::map<AddressSpaceID, local::FieldMaskMap<EqKDTree>> to_notify;
      for (unsigned idx1 = 0; idx1 < num_notifications; idx1++)
      {
        AddressSpaceID space;
        derez.deserialize(space);
        local::FieldMaskMap<EqKDTree>& trees = to_notify[space];
        size_t num_trees;
        derez.deserialize(num_trees);
        for (unsigned idx2 = 0; idx2 < num_trees; idx2++)
        {
          EqKDTree* tree;
          derez.deserialize(tree);
          FieldMask mask;
          derez.deserialize(mask);
          trees.insert(tree, mask);
        }
      }
      size_t num_targets;
      derez.deserialize(num_targets);
      const CollectiveMapping target_mapping(derez, num_targets);
      std::vector<EqSetTracker*> targets(num_targets);
      for (unsigned idx = 0; idx < num_targets; idx++)
        derez.deserialize(targets[idx]);
      FieldMask recording_mask;
      derez.deserialize(recording_mask);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      std::vector<AddressSpaceID> children;
      mapping.get_children(owner_space, runtime->address_space, children);
      std::vector<RtEvent> done_events;
      for (const AddressSpaceID cit : children)
      {
        const RtUserEvent notified = Runtime::create_rt_user_event();
        EquivalenceSetReuse rez;
        {
          RezCheck z2(rez);
          // Reference still held by at least one EqKDTree
          // for which this is in the previous set
          rez.serialize(did);
          rez.serialize(owner);
          rez.serialize(owner_space);
          mapping.pack(rez);
          rez.serialize(ready_event);
          rez.serialize<size_t>(to_notify.size());
          for (const std::pair<
                   const AddressSpaceID, local::FieldMaskMap<EqKDTree>>& nit :
               to_notify)
          {
            rez.serialize(nit.first);
            rez.serialize(nit.second.size());
            for (const std::pair<EqKDTree* const, FieldMask>& it : nit.second)
            {
              rez.serialize(it.first);
              rez.serialize(it.second);
            }
          }
          // Also pack the target information
          target_mapping.pack(rez);
          for (unsigned idx = 0; idx < targets.size(); idx++)
            rez.serialize(targets[idx]);
          rez.serialize(recording_mask);
          rez.serialize(notified);
        }
        rez.dispatch(cit);
        done_events.emplace_back(notified);
      }
      local::map<AddressSpaceID, local::FieldMaskMap<EqKDTree>>::iterator
          local_finder = to_notify.find(runtime->address_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      if (local_finder != to_notify.end())
      {
        for (local::FieldMaskMap<EqKDTree>::const_iterator it =
                 local_finder->second.begin();
             it != local_finder->second.end(); it++)
          it->first->record_equivalence_set(
              set, it->second, ready_event, target_mapping, targets);
      }
      // Check to see if a we have a local target
      if (target_mapping.contains(runtime->address_space))
      {
        unsigned index = target_mapping.find_index(runtime->address_space);
        targets[index]->record_pending_equivalence_set(set, recording_mask);
      }
      if (!done_events.empty())
        Runtime::trigger_event(done_event, Runtime::merge_events(done_events));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    unsigned EqSetTracker::invalidate_equivalence_sets(
        const FieldMask& mask, EqKDTree* tree, AddressSpaceID source,
        std::vector<RtEvent>& invalidated_events)
    //--------------------------------------------------------------------------
    {
      unsigned source_references_to_remove = 0;
      local::map<AddressSpaceID, local::FieldMaskMap<EqKDTree>> to_cancel;
      {
        FieldMask remaining = mask;
        AutoLock t_lock(tracker_lock);
        // If we've already recorded a pending invalidation for any of these
        // fields then there's nothing more we need to do here
        if (!!pending_invalidations)
        {
          remaining -= pending_invalidations;
          if (!remaining)
            return source_references_to_remove;
        }
        // Invalidations are not precise so we need to handle the case where
        // we get an invalidation in the middle of computing our equivalence
        // sets and record that we need to do the invalidation later
        if (equivalence_sets_ready != nullptr)
        {
          FieldMask to_filter;
          for (shrt::map<RtUserEvent, FieldMask>::const_iterator it =
                   equivalence_sets_ready->begin();
               it != equivalence_sets_ready->end(); it++)
          {
            const FieldMask overlap = remaining & it->second;
            if (!overlap)
              continue;
            to_filter |= overlap;
            remaining -= overlap;
            if (!remaining)
              break;
          }
          if (!!to_filter)
          {
#ifdef LEGION_DEBUG
            // Sanity check: invalidations and creations can race due
            // to false aliasing in the EqKD tree (independent operations
            // on different points sharing a KD-tree node with overlapping
            // field masks). Any such overlap must be covered by
            // equivalence_sets_ready so finalize_equivalence_sets will
            // detect it and recompute.
            if (creation_rectangles != nullptr)
            {
              FieldMask creation_fields;
              for (op::map<Domain, FieldMask>::const_iterator it =
                       creation_rectangles->begin();
                   it != creation_rectangles->end(); it++)
                creation_fields |= it->second;
              const FieldMask overlap = to_filter & creation_fields;
              if (!!overlap)
              {
                FieldMask covered;
                for (shrt::map<RtUserEvent, FieldMask>::const_iterator it =
                         equivalence_sets_ready->begin();
                     it != equivalence_sets_ready->end(); it++)
                  covered |= it->second;
                legion_assert(!(overlap - covered));
              }
            }
#endif
            pending_invalidations |= to_filter;
            if (!remaining)
              return source_references_to_remove;
          }
        }
        // See if we won the race to invalidating the source
        lng::map<AddressSpaceID, lng::FieldMaskMap<EqKDTree>>::iterator finder =
            current_subscriptions.find(source);
        if (finder != current_subscriptions.end())
        {
          lng::FieldMaskMap<EqKDTree>::iterator source_finder =
              finder->second.find(tree);
          if (source_finder != finder->second.end())
          {
            const FieldMask overlap = source_finder->second & remaining;
            if (!!overlap)
            {
              source_references_to_remove += overlap.pop_count();
              source_finder.filter(overlap);
              if (!source_finder->second)
              {
                finder->second.erase(source_finder);
                if (finder->second.empty())
                  current_subscriptions.erase(finder);
                else
                  finder->second.tighten_valid_mask();
              }
              else
                finder->second.tighten_valid_mask();
            }
          }
        }
        // Now go through and invalidate all the other subscriptions to cancel
        find_cancellations(remaining, to_cancel);
        if (!(remaining * equivalence_sets.get_valid_mask()))
        {
          std::vector<EquivalenceSet*> to_delete;
          for (lng::FieldMaskMap<EquivalenceSet>::iterator it =
                   equivalence_sets.begin();
               it != equivalence_sets.end(); it++)
          {
            it.filter(remaining);
            if (!it->second)
              to_delete.emplace_back(it->first);
          }
          if (!to_delete.empty())
          {
            const ReferenceSource source_kind = get_reference_source_kind();
            for (EquivalenceSet* it : to_delete)
            {
              equivalence_sets.erase(it);
              if (it->remove_base_gc_ref(source_kind))
                delete it;
            }
          }
          equivalence_sets.tighten_valid_mask();
        }
      }
      if (!to_cancel.empty())
      {
        for (local::map<AddressSpaceID, local::FieldMaskMap<EqKDTree>>::
                 const_iterator it = to_cancel.begin();
             it != to_cancel.end(); it++)
          cancel_subscriptions(
              it->first, FieldMapView(it->second), &invalidated_events);
      }
      return source_references_to_remove;
    }

    //--------------------------------------------------------------------------
    void EqSetTracker::find_cancellations(
        const FieldMask& mask,
        local::map<AddressSpaceID, local::FieldMaskMap<EqKDTree>>& to_cancel)
    //--------------------------------------------------------------------------
    {
      // Lock held from caller
      for (lng::map<AddressSpaceID, lng::FieldMaskMap<EqKDTree>>::iterator cit =
               current_subscriptions.begin();
           cit != current_subscriptions.end();
           /*nothing*/)
      {
        if (cit->second.get_valid_mask() * mask)
        {
          cit++;
          continue;
        }
        if (!!(cit->second.get_valid_mask() - mask))
        {
          // Selectively filter matches
          std::vector<EqKDTree*> to_delete;
          local::FieldMaskMap<EqKDTree>& invalidations = to_cancel[cit->first];
          for (lng::FieldMaskMap<EqKDTree>::iterator it = cit->second.begin();
               it != cit->second.end(); it++)
          {
            const FieldMask overlap = it->second & mask;
            if (!overlap)
              continue;
            invalidations.insert(it->first, overlap);
            it.filter(overlap);
            if (!it->second)
              to_delete.emplace_back(it->first);
          }
          for (EqKDTree* const it : to_delete) cit->second.erase(it);
        }
        else  // Filtering all the fields so we can swap it out
        {
          for (lng::FieldMaskMap<EqKDTree>::iterator it = cit->second.begin();
               it != cit->second.end(); it++)
            to_cancel[cit->first].insert(it->first, it->second);
          cit->second.clear();
        }
        if (cit->second.empty())
        {
          lng::map<AddressSpaceID, lng::FieldMaskMap<EqKDTree>>::iterator
              delete_it = cit++;
          current_subscriptions.erase(delete_it);
        }
        else
        {
          cit->second.tighten_valid_mask();
          cit++;
        }
      }
    }

    //--------------------------------------------------------------------------
    void EqSetTracker::cancel_subscriptions(
        AddressSpaceID target, const FieldMapView<EqKDTree>& to_cancel,
        std::vector<RtEvent>* cancelled_events)
    //--------------------------------------------------------------------------
    {
      if (target != runtime->address_space)
      {
        CancelEquivalenceSetSubscription rez;
        {
          RezCheck z(rez);
          legion_assert(!to_cancel.empty());
          rez.serialize<size_t>(to_cancel.size());
          rez.serialize(this);
          for (const std::pair<EqKDTree* const, FieldMask>& it : to_cancel)
          {
            rez.serialize(it.first);
            rez.serialize(it.second);
          }
          if (cancelled_events != nullptr)
          {
            const RtUserEvent cancelled = Runtime::create_rt_user_event();
            rez.serialize(cancelled);
            cancelled_events->emplace_back(cancelled);
          }
          else
            rez.serialize(RtUserEvent::NO_RT_USER_EVENT);
        }
        rez.dispatch(target);
      }
      else
      {
        unsigned references_to_remove = 0;
        for (const std::pair<EqKDTree* const, FieldMask>& it : to_cancel)
        {
          references_to_remove += it.first->cancel_subscription(
              this, runtime->address_space, it.second);
          if (it.first->remove_reference(it.second.pop_count()))
            delete it.first;
        }
        if ((references_to_remove > 0) &&
            remove_subscription_reference(references_to_remove))
          std::abort();  // should never end up deleting ourselves
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CancelEquivalenceSetSubscription::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t num_subscribers;
      derez.deserialize(num_subscribers);
      if (num_subscribers > 0)
      {
        EqSetTracker* owner;
        derez.deserialize(owner);
        unsigned references_to_remove = 0;
        for (unsigned idx = 0; idx < num_subscribers; idx++)
        {
          EqKDTree* tree;
          derez.deserialize(tree);
          FieldMask mask;
          derez.deserialize(mask);
          references_to_remove +=
              tree->cancel_subscription(owner, source, mask);
          if (tree->remove_reference(mask.pop_count()))
            delete tree;
        }
        if (references_to_remove > 0)
        {
          InvalidateEquivalenceSetSubscription rez;
          {
            RezCheck z2(rez);
            rez.serialize<size_t>(0);  // nothing to filter
            rez.serialize(owner);
            rez.serialize(references_to_remove);
          }
          rez.dispatch(source);
        }
        RtUserEvent cancelled;
        derez.deserialize(cancelled);
        if (cancelled.exists())
          Runtime::trigger_event(cancelled);
      }
      else
      {
        EqKDTree* subscriber;
        derez.deserialize(subscriber);
        unsigned references_to_remove;
        derez.deserialize(references_to_remove);
        if (subscriber->remove_reference(references_to_remove))
          delete subscriber;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void EqSetTracker::invalidate_subscriptions(
        EqKDTree* owner,
        const MapView<AddressSpaceID, local::FieldMaskMap<EqSetTracker>>&
            subscribers,
        std::vector<RtEvent>& invalidated_events)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID local_space = runtime->address_space;
      for (MapView<AddressSpaceID, local::FieldMaskMap<EqSetTracker>>::
               const_iterator sit = subscribers.begin();
           sit != subscribers.end(); sit++)
      {
        if (sit->first != local_space)
        {
          const RtUserEvent invalidated = Runtime::create_rt_user_event();
          InvalidateEquivalenceSetSubscription rez;
          {
            RezCheck z(rez);
            legion_assert(!sit->second.empty());
            rez.serialize<size_t>(sit->second.size());
            rez.serialize(owner);
            for (local::FieldMaskMap<EqSetTracker>::const_iterator it =
                     sit->second.begin();
                 it != sit->second.end(); it++)
            {
              rez.serialize(it->first);
              rez.serialize(it->second);
            }
            rez.serialize(invalidated);
          }
          rez.dispatch(sit->first);
          invalidated_events.emplace_back(invalidated);
        }
        else
        {
          unsigned references_to_remove = 0;
          for (local::FieldMaskMap<EqSetTracker>::const_iterator it =
                   sit->second.begin();
               it != sit->second.end(); it++)
          {
            references_to_remove += it->first->invalidate_equivalence_sets(
                it->second, owner, runtime->address_space, invalidated_events);
            if (it->first->remove_subscription_reference(
                    it->second.pop_count()))
              delete it->first;
          }
          if ((references_to_remove > 0) &&
              owner->remove_reference(references_to_remove))
            delete owner;
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void InvalidateEquivalenceSetSubscription::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t num_subscribers;
      derez.deserialize(num_subscribers);
      if (num_subscribers > 0)
      {
        EqKDTree* owner;
        derez.deserialize(owner);
        unsigned references_to_remove = 0;
        std::vector<RtEvent> invalidated_events;
        for (unsigned idx = 0; idx < num_subscribers; idx++)
        {
          EqSetTracker* tracker;
          derez.deserialize(tracker);
          FieldMask mask;
          derez.deserialize(mask);
          references_to_remove += tracker->invalidate_equivalence_sets(
              mask, owner, source, invalidated_events);
          if (tracker->remove_subscription_reference(mask.pop_count()))
            delete tracker;
        }
        RtUserEvent invalidated;
        derez.deserialize(invalidated);
        if (!invalidated_events.empty())
          Runtime::trigger_event(
              invalidated, Runtime::merge_events(invalidated_events));
        else
          Runtime::trigger_event(invalidated);
        if (references_to_remove > 0)
        {
          CancelEquivalenceSetSubscription rez;
          {
            RezCheck z2(rez);
            rez.serialize<size_t>(0);  // num subscribers
            rez.serialize(owner);
            rez.serialize(references_to_remove);
          }
          rez.dispatch(source);
        }
      }
      else
      {
        EqSetTracker* subscriber;
        derez.deserialize(subscriber);
        unsigned references_to_remove;
        derez.deserialize(references_to_remove);
        if (subscriber->remove_subscription_reference(references_to_remove))
          delete subscriber;
      }
    }

    //--------------------------------------------------------------------------
    EqSetTracker::LgFinalizeEqSetsArgs::LgFinalizeEqSetsArgs(
        EqSetTracker* t, RtUserEvent c, InnerContext* enclose,
        InnerContext* outer, unsigned index, IndexSpaceExpression* e)
      : LgTaskArgs<LgFinalizeEqSetsArgs>(false, true), tracker(t), compute(c),
        enclosing(enclose), outermost(outer), expr(e), parent_req_index(index)
    //--------------------------------------------------------------------------
    {
      enclosing->add_base_gc_ref(META_TASK_REF);
      outermost->add_base_gc_ref(META_TASK_REF);
      expr->add_base_expression_reference(META_TASK_REF);
    }

    //--------------------------------------------------------------------------
    void EqSetTracker::LgFinalizeEqSetsArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      tracker->finalize_equivalence_sets(
          compute, enclosing, outermost, parent_req_index, expr, unique_op_id);
      if (enclosing->remove_base_gc_ref(META_TASK_REF))
        delete enclosing;
      if (outermost->remove_base_gc_ref(META_TASK_REF))
        delete outermost;
      if (expr->remove_base_expression_reference(META_TASK_REF))
        delete expr;
    }

  }  // namespace Internal
}  // namespace Legion
