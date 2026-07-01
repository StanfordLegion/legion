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

#include "legion/views/individual.h"
#include "legion/analysis/collective.h"
#include "legion/instances/physical.h"
#include "legion/nodes/across.h"
#include "legion/nodes/index.h"
#include "legion/nodes/region.h"
#include "legion/views/allreduce.h"
#include "legion/views/collective.h"
#include "legion/views/fill.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Physical User
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(
        const RegionUsage& u, IndexSpaceExpression* e, ApEvent term,
        UniqueID id, unsigned x, bool cpy, bool cov)
      : usage(u), expr(e), term_event(term), op_id(id), index(x),
        copy_user(cpy), covers(cov)
    //--------------------------------------------------------------------------
    {
      legion_assert(expr != nullptr);
      expr->add_base_expression_reference(PHYSICAL_USER_REF);
    }

    //--------------------------------------------------------------------------
    PhysicalUser::~PhysicalUser(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(expr != nullptr);
      if (expr->remove_base_expression_reference(PHYSICAL_USER_REF))
        delete expr;
    }

    /////////////////////////////////////////////////////////////
    // NodeView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    NodeView::NodeView(IndexTreeNode* n, IndividualView* v)
      : tree_node(n), view(v)
    //--------------------------------------------------------------------------
    {
      tree_node->add_nested_valid_ref(view->did);
    }

    //--------------------------------------------------------------------------
    NodeView::~NodeView(void)
    //--------------------------------------------------------------------------
    {
      if (tree_node->remove_nested_valid_ref(view->did))
        delete tree_node;
    }

    /////////////////////////////////////////////////////////////
    // SpaceView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SpaceView::SpaceView(IndexSpaceNode* n, IndividualView* v)
      : NodeView(n, v), node(n)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    SpaceView::~SpaceView(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(current_epoch_users.empty());
      legion_assert(previous_epoch_users.empty());
      legion_assert(subviews.empty());
    }

    //--------------------------------------------------------------------------
    bool SpaceView::dominated_by(IndexSpaceExpression* expr) const
    //--------------------------------------------------------------------------
    {
      IndexSpaceExpression* overlap =
          runtime->intersect_index_spaces(node, expr);
      return (overlap->get_volume() == node->get_volume());
    }

    //--------------------------------------------------------------------------
    bool SpaceView::is_empty(void) const
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock, false /*exclusive*/);
      return (
          current_epoch_users.empty() && previous_epoch_users.empty() &&
          subviews.empty());
    }

    //--------------------------------------------------------------------------
    void SpaceView::invalidate_users(void)
    //--------------------------------------------------------------------------
    {
      // Shouldn't be any races on deleteions so no need for the lock
      for (shrt::FieldMaskMap<PhysicalUser>::const_iterator it =
               current_epoch_users.begin();
           it != current_epoch_users.end(); it++)
        if (it->first->remove_reference())
          delete it->first;
      current_epoch_users.clear();
      for (shrt::FieldMaskMap<PhysicalUser>::const_iterator it =
               previous_epoch_users.begin();
           it != previous_epoch_users.end(); it++)
        if (it->first->remove_reference())
          delete it->first;
      previous_epoch_users.clear();
      for (lng::FieldMaskMap<PartitionView, ViewComparator<PartitionView> >::
               const_iterator it = subviews.begin();
           it != subviews.end(); it++)
      {
        it->first->invalidate_users();
        if (it->first->remove_reference())
          delete it->first;
      }
      subviews.clear();
    }

    //--------------------------------------------------------------------------
    void SpaceView::find_last_users(
        const RegionUsage& usage, IndexSpaceExpression* expr,
        const bool expr_dominates, const FieldMask& mask,
        std::set<ApEvent>& last_events) const
    //--------------------------------------------------------------------------
    {
      local::FieldMaskMap<PartitionView> to_traverse;
      {
        FieldMask dominated;
        AutoLock v_lock(view_lock, false /*exclusive*/);
        // We dominate in this case so we can do filtering
        if (!current_epoch_users.empty())
        {
          FieldMask observed, non_dominated;
          find_current_preconditions(
              usage, mask, expr, expr_dominates, last_events, observed,
              non_dominated);
          if (!!observed)
            dominated = observed - non_dominated;
        }
        if (!previous_epoch_users.empty())
        {
          const FieldMask previous_mask = mask - dominated;
          if (!!previous_mask)
            find_previous_preconditions(
                usage, previous_mask, expr, expr_dominates, last_events);
        }
        find_subviews_to_traverse(mask, to_traverse);
      }
      for (local::FieldMaskMap<PartitionView>::const_iterator it =
               to_traverse.begin();
           it != to_traverse.end(); it++)
      {
        it->first->find_last_users(
            usage, expr, expr_dominates, it->second, last_events);
        if (it->first->remove_reference())
          delete it->first;
      }
    }

    //--------------------------------------------------------------------------
    void SpaceView::find_subviews_to_traverse(
        const FieldMask& mask,
        local::FieldMaskMap<PartitionView>& to_traverse) const
    //--------------------------------------------------------------------------
    {
      if (!(subviews.get_valid_mask() * mask))
      {
        for (lng::FieldMaskMap<PartitionView, ViewComparator<PartitionView> >::
                 const_iterator it = subviews.begin();
             it != subviews.end(); it++)
        {
          const FieldMask overlap = it->second & mask;
          if (!overlap)
            continue;
          if (to_traverse.insert(it->first, overlap))
            it->first->add_reference();
        }
      }
    }

    //--------------------------------------------------------------------------
    bool SpaceView::find_user_preconditions(
        const RegionUsage& usage, IndexSpaceExpression* user_expr,
        const bool user_dominates, const FieldMask& user_mask,
        ApEvent term_event, UniqueID op_id, unsigned index,
        std::set<ApEvent>& preconditions, const bool trace_recording)
    //--------------------------------------------------------------------------
    {
      FieldMask dominated;
      local::set<PhysicalUser*> dead_users;
      local::FieldMaskMap<PhysicalUser> current_to_filter, previous_to_filter;
      local::FieldMaskMap<PartitionView> to_traverse;
      {
        AutoLock v_lock(view_lock, false /*exclusive*/);
        // Check to see if we dominate when doing this analysis and
        // can therefore filter or whether we are just intersecting
        // Do the local analysis
        if (user_dominates)
        {
          // We dominate in this case so we can do filtering
          if (!current_epoch_users.empty())
          {
            FieldMask observed, non_dominated;
            find_current_preconditions(
                usage, user_mask, user_expr, term_event, op_id, index,
                user_dominates, preconditions, dead_users, current_to_filter,
                observed, non_dominated, trace_recording, false /*copy*/);
            if (!!observed)
              dominated = observed - non_dominated;
          }
          if (!previous_epoch_users.empty())
          {
            if (!!dominated)
              find_previous_filter_users(dominated, previous_to_filter);
            const FieldMask previous_mask = user_mask - dominated;
            if (!!previous_mask)
              find_previous_preconditions(
                  usage, previous_mask, user_expr, term_event, op_id, index,
                  user_dominates, preconditions, dead_users, trace_recording,
                  false /*copy*/);
          }
        }
        else
        {
          if (!current_epoch_users.empty())
          {
            FieldMask observed, non_dominated;
            find_current_preconditions(
                usage, user_mask, user_expr, term_event, op_id, index,
                user_dominates, preconditions, dead_users, current_to_filter,
                observed, non_dominated, trace_recording, false /*copy*/);
            legion_assert(!observed);
            legion_assert(current_to_filter.empty());
          }
          if (!previous_epoch_users.empty())
            find_previous_preconditions(
                usage, user_mask, user_expr, term_event, op_id, index,
                user_dominates, preconditions, dead_users, trace_recording,
                false /*copy*/);
        }
        find_subviews_to_traverse(user_mask, to_traverse);
      }
      // It's possible that we recorded some users for fields which
      // are not actually fully dominated, if so we need to prune them
      // otherwise we can get into issues of soundness
      if (!current_to_filter.empty())
        verify_current_to_filter(dominated, current_to_filter);
      if (!dead_users.empty() || !previous_to_filter.empty() ||
          !current_to_filter.empty())
      {
        // Need exclusive permissions to modify data structures
        AutoLock v_lock(view_lock);
        if (!dead_users.empty())
          filter_dead_users(dead_users);
        if (!previous_to_filter.empty())
          filter_previous_users(previous_to_filter);
        if (!current_to_filter.empty())
          filter_current_users(current_to_filter);
      }
      for (local::FieldMaskMap<PartitionView>::const_iterator it =
               to_traverse.begin();
           it != to_traverse.end(); it++)
      {
        unsigned refs_to_remove = 1;
        if (it->first->find_user_preconditions(
                usage, user_expr, user_dominates, it->second, term_event, op_id,
                index, preconditions, trace_recording))
        {
          AutoLock v_lock(view_lock);
          lng::FieldMaskMap<
              PartitionView, ViewComparator<PartitionView> >::iterator finder =
              subviews.find(it->first);
          if ((finder != subviews.end()) && (it->first == finder->first) &&
              it->first->is_empty())
          {
            subviews.erase(finder);
            refs_to_remove++;
          }
        }
        if (it->first->remove_reference(refs_to_remove))
          delete it->first;
      }
      return is_empty();
    }

    //--------------------------------------------------------------------------
    bool SpaceView::find_copy_preconditions(
        const RegionUsage& usage, IndexSpaceExpression* copy_expr,
        const bool copy_dominates, const FieldMask& copy_mask, UniqueID op_id,
        unsigned index, std::set<ApEvent>& preconditions,
        const bool trace_recording)
    //--------------------------------------------------------------------------
    {
      FieldMask dominated;
      local::set<PhysicalUser*> dead_users;
      local::FieldMaskMap<PartitionView> to_traverse;
      local::FieldMaskMap<PhysicalUser> current_to_filter, previous_to_filter;
      // Do the first pass with a read-only lock on the events
      {
        AutoLock v_lock(view_lock, false /*exclusive*/);
        // Check to see if we dominate when doing this analysis and
        // can therefore filter or whether we are just intersecting
        // Do the local analysis
        if (copy_dominates)
        {
          // We dominate in this case so we can do filtering
          if (!current_epoch_users.empty())
          {
            FieldMask observed, non_dominated;
            find_current_preconditions(
                usage, copy_mask, copy_expr, ApEvent::NO_AP_EVENT, op_id, index,
                copy_dominates, preconditions, dead_users, current_to_filter,
                observed, non_dominated, trace_recording, true /*copy user*/);
            if (!!observed)
              dominated = observed - non_dominated;
          }
          if (!previous_epoch_users.empty())
          {
            if (!!dominated)
              find_previous_filter_users(dominated, previous_to_filter);
            const FieldMask previous_mask = copy_mask - dominated;
            if (!!previous_mask)
              find_previous_preconditions(
                  usage, previous_mask, copy_expr, ApEvent::NO_AP_EVENT, op_id,
                  index, copy_dominates, preconditions, dead_users,
                  trace_recording, true /*copy user*/);
          }
        }
        else
        {
          if (!current_epoch_users.empty())
          {
            FieldMask observed, non_dominated;
            find_current_preconditions(
                usage, copy_mask, copy_expr, ApEvent::NO_AP_EVENT, op_id, index,
                copy_dominates, preconditions, dead_users, current_to_filter,
                observed, non_dominated, trace_recording, true /*copy user*/);
            legion_assert(!observed);
            legion_assert(current_to_filter.empty());
          }
          if (!previous_epoch_users.empty())
            find_previous_preconditions(
                usage, copy_mask, copy_expr, ApEvent::NO_AP_EVENT, op_id, index,
                copy_dominates, preconditions, dead_users, trace_recording,
                true /*copy user*/);
        }
        find_subviews_to_traverse(copy_mask, to_traverse);
      }
      // It's possible that we recorded some users for fields which
      // are not actually fully dominated, if so we need to prune them
      // otherwise we can get into issues of soundness
      if (!current_to_filter.empty())
        verify_current_to_filter(dominated, current_to_filter);
      if (!dead_users.empty() || !previous_to_filter.empty() ||
          !current_to_filter.empty())
      {
        // Need exclusive permissions to modify data structures
        AutoLock v_lock(view_lock);
        if (!dead_users.empty())
          filter_dead_users(dead_users);
        if (!previous_to_filter.empty())
          filter_previous_users(previous_to_filter);
        if (!current_to_filter.empty())
          filter_current_users(current_to_filter);
      }
      for (local::FieldMaskMap<PartitionView>::const_iterator it =
               to_traverse.begin();
           it != to_traverse.end(); it++)
      {
        unsigned refs_to_remove = 1;
        if (it->first->find_copy_preconditions(
                usage, copy_expr, copy_dominates, it->second, op_id, index,
                preconditions, trace_recording))
        {
          AutoLock v_lock(view_lock);
          lng::FieldMaskMap<
              PartitionView, ViewComparator<PartitionView> >::iterator finder =
              subviews.find(it->first);
          if ((finder != subviews.end()) && (it->first == finder->first) &&
              it->first->is_empty())
          {
            subviews.erase(finder);
            refs_to_remove++;
          }
        }
        if (it->first->remove_reference(refs_to_remove))
          delete it->first;
      }
      return is_empty();
    }

    //--------------------------------------------------------------------------
    void SpaceView::insert_child(NodeView* child, const FieldMask& child_mask)
    //--------------------------------------------------------------------------
    {
      PartitionView* part_child = legion_safe_cast<PartitionView*>(child);
      legion_assert(part_child->node->parent == node);
      AutoLock v_lock(view_lock);
      if (subviews.insert(part_child, child_mask))
        part_child->add_reference();
    }

    //--------------------------------------------------------------------------
    void SpaceView::insert_user(
        PhysicalUser* user, const FieldMask& user_mask,
        local::vector<LegionColor>& path, AutoLock& parent_lock)
    //--------------------------------------------------------------------------
    {
      // Quick test to see if we can do this read-only
      if (!path.empty())
      {
        AutoLock v_lock(view_lock, false /*exclusive*/);
        lng::FieldMaskMap<PartitionView, ViewComparator<PartitionView> >::
            const_iterator finder = subviews.find(path.back());
        if ((finder != subviews.end()) && !(user_mask - finder->second))
        {
          parent_lock.release();
          path.pop_back();
          finder->first->insert_user(user, user_mask, path, v_lock);
          return;
        }
      }
      // Do hand-over-hand locking
      AutoLock v_lock(view_lock);
      parent_lock.release();
      if (!path.empty())
      {
        const LegionColor color = path.back();
        path.pop_back();
        // See if we already have it or whether we need to make it
        lng::FieldMaskMap<
            PartitionView, ViewComparator<PartitionView> >::iterator finder =
            subviews.find(color);
        if (finder == subviews.end())
        {
          // Need to make one since it doesn't exist yet
          PartitionView* child =
              new PartitionView(node->get_child(color), view);
          if (subviews.insert(child, user_mask))
            child->add_reference();
          child->insert_user(user, user_mask, path, v_lock);
        }
        else
        {
          finder.merge(user_mask);
          finder->first->insert_user(user, user_mask, path, v_lock);
        }
      }
      else if (current_epoch_users.insert(user, user_mask))
        user->add_reference();
    }

    //--------------------------------------------------------------------------
    /*static*/ void SpaceView::verify_current_to_filter(
        const FieldMask& dominated,
        local::FieldMaskMap<PhysicalUser>& current_to_filter)
    //--------------------------------------------------------------------------
    {
      if (!!dominated)
      {
        const FieldMask non_dominated =
            current_to_filter.get_valid_mask() - dominated;
        if (!non_dominated)
          return;
        if (non_dominated != current_to_filter.get_valid_mask())
        {
          // Selectively filter
          local::vector<PhysicalUser*> to_delete;
          for (local::FieldMaskMap<PhysicalUser>::iterator it =
                   current_to_filter.begin();
               it != current_to_filter.end(); it++)
          {
            it.filter(non_dominated);
            if (!it->second)
              to_delete.emplace_back(it->first);
          }
          for (PhysicalUser* user : to_delete)
          {
            current_to_filter.erase(user);
            if (user->remove_reference())
              delete user;
          }
          current_to_filter.tighten_valid_mask();
          return;
        }
      }
      // Otherwise we fall through here and clean out all the users
      for (local::FieldMaskMap<PhysicalUser>::const_iterator it =
               current_to_filter.begin();
           it != current_to_filter.end(); it++)
        if (it->first->remove_reference())
          delete it->first;
      current_to_filter.clear();
    }

    //--------------------------------------------------------------------------
    void SpaceView::filter_dead_users(
        const local::set<PhysicalUser*>& dead_users)
    //--------------------------------------------------------------------------
    {
      // Don't do this if we are in Legion Spy since we want to see
      // all of the dependences on an instance
      for (PhysicalUser* const physical_user : dead_users)
      {
        unsigned refs_to_remove = 1;
        if (spy_logging_level <= LIGHT_SPY_LOGGING)
        {
          shrt::FieldMaskMap<PhysicalUser>::iterator finder =
              current_epoch_users.find(physical_user);
          if (finder != current_epoch_users.end())
          {
            current_epoch_users.erase(finder);
            refs_to_remove++;
          }
          finder = previous_epoch_users.find(physical_user);
          if (finder != previous_epoch_users.end())
          {
            previous_epoch_users.erase(finder);
            refs_to_remove++;
          }
        }
        if (physical_user->remove_reference(refs_to_remove))
          delete physical_user;
      }
    }

    //--------------------------------------------------------------------------
    void SpaceView::filter_current_users(
        const FieldMapView<PhysicalUser>& to_filter)
    //--------------------------------------------------------------------------
    {
      // Lock needs to be held by caller
      for (FieldMapView<PhysicalUser>::const_iterator it = to_filter.begin();
           it != to_filter.end(); it++)
      {
        unsigned refs_to_remove = 1;
        shrt::FieldMaskMap<PhysicalUser>::iterator finder =
            current_epoch_users.find(it->first);
        if (finder != current_epoch_users.end())
        {
          const FieldMask overlap = it->second & finder->second;
          if (!!overlap)
          {
            finder.filter(overlap);
            if (!finder->second)
            {
              current_epoch_users.erase(finder);
              refs_to_remove++;
            }
            // Add this into the previous epoch users
            // and pass along a reference if necessary
            if (previous_epoch_users.insert(it->first, overlap))
              refs_to_remove--;
          }
        }
        if ((refs_to_remove > 0) && it->first->remove_reference(refs_to_remove))
          delete it->first;
      }
      current_epoch_users.tighten_valid_mask();
    }

    //--------------------------------------------------------------------------
    void SpaceView::filter_previous_users(
        const FieldMapView<PhysicalUser>& to_filter)
    //--------------------------------------------------------------------------
    {
      // Lock needs to be held by caller
      for (FieldMapView<PhysicalUser>::const_iterator it = to_filter.begin();
           it != to_filter.end(); it++)
      {
        unsigned refs_to_remove = 1;
        shrt::FieldMaskMap<PhysicalUser>::iterator finder =
            previous_epoch_users.find(it->first);
        if (finder != previous_epoch_users.end())
        {
          finder.filter(it->second);
          if (!finder->second)
          {
            previous_epoch_users.erase(finder);
            refs_to_remove++;
          }
        }
        // Remove the reference we were holding
        if (it->first->remove_reference(refs_to_remove))
          delete it->first;
      }
      previous_epoch_users.tighten_valid_mask();
    }

    //--------------------------------------------------------------------------
    void SpaceView::find_current_preconditions(
        const RegionUsage& usage, const FieldMask& user_mask,
        IndexSpaceExpression* user_expr, ApEvent term_event,
        const UniqueID op_id, const unsigned index, const bool user_covers,
        std::set<ApEvent>& preconditions, local::set<PhysicalUser*>& dead_users,
        local::FieldMaskMap<PhysicalUser>& filter_users, FieldMask& observed,
        FieldMask& non_dominated, const bool trace_recording,
        const bool copy_user)
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      if (user_mask * current_epoch_users.get_valid_mask())
        return;
      for (shrt::FieldMaskMap<PhysicalUser>::const_iterator it =
               current_epoch_users.begin();
           it != current_epoch_users.end(); it++)
      {
        if (it->first->term_event == term_event)
          continue;
        // We're about to do a bunch of expensive tests,
        // so first do something cheap to see if we can
        // skip all the tests.
        if (!trace_recording && (spy_logging_level <= LIGHT_SPY_LOGGING) &&
            it->first->term_event.has_triggered_faultignorant())
        {
          if (dead_users.insert(it->first).second)
            it->first->add_reference();
          continue;
        }
#if 0
        // You might think you can optimize things like this, but you can't
        // because we still need the correct epoch users for every ExprView
        // when we go to add our user later
        if (!trace_recording &&
            preconditions.find(it->first->term_event) != preconditions.end())
          continue;
#endif
        const FieldMask overlap = user_mask & it->second;
        if (!overlap)
          continue;
        bool dominates = true;
        if (has_local_precondition(
                it->first, usage, user_expr, op_id, index, user_covers,
                copy_user, &dominates))
        {
          preconditions.insert(it->first->term_event);
          if (dominates)
          {
            observed |= overlap;
            if (filter_users.insert(it->first, overlap))
              it->first->add_reference();
          }
          else
            non_dominated |= overlap;
        }
        else
          non_dominated |= overlap;
      }
    }

    //--------------------------------------------------------------------------
    void SpaceView::find_previous_preconditions(
        const RegionUsage& usage, const FieldMask& user_mask,
        IndexSpaceExpression* user_expr, ApEvent term_event,
        const UniqueID op_id, const unsigned index, const bool user_covers,
        std::set<ApEvent>& preconditions, local::set<PhysicalUser*>& dead_users,
        const bool trace_recording, const bool copy_user)
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      if (user_mask * previous_epoch_users.get_valid_mask())
        return;
      for (shrt::FieldMaskMap<PhysicalUser>::const_iterator it =
               previous_epoch_users.begin();
           it != previous_epoch_users.end(); it++)
      {
        if (it->first->term_event == term_event)
          continue;
        // We're about to do a bunch of expensive tests,
        // so first do something cheap to see if we can
        // skip all the tests.
        if (!trace_recording && (spy_logging_level <= LIGHT_SPY_LOGGING) &&
            it->first->term_event.has_triggered_faultignorant())
        {
          if (dead_users.insert(it->first).second)
            it->first->add_reference();
          continue;
        }
#if 0
        // You might think you can optimize things like this, but you can't
        // because we still need the correct epoch users for every ExprView
        // when we go to add our user later
        if (!trace_recording &&
            preconditions.find(it->first->term_event) != preconditions.end())
          continue;
#endif
        const FieldMask overlap = user_mask & it->second;
        if (!overlap)
          continue;
        if (has_local_precondition(
                it->first, usage, user_expr, op_id, index, user_covers,
                copy_user))
          preconditions.insert(it->first->term_event);
      }
    }

    //--------------------------------------------------------------------------
    void SpaceView::find_current_preconditions(
        const RegionUsage& usage, const FieldMask& mask,
        IndexSpaceExpression* expr, const bool expr_covers,
        std::set<ApEvent>& last_events, FieldMask& observed,
        FieldMask& non_dominated) const
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      if (mask * current_epoch_users.get_valid_mask())
        return;
      for (shrt::FieldMaskMap<PhysicalUser>::const_iterator it =
               current_epoch_users.begin();
           it != current_epoch_users.end(); it++)
      {
        const FieldMask overlap = mask & it->second;
        if (!overlap)
          continue;
        bool dominated = true;
        // We're just reading these and we want to see all prior
        // dependences so just give dummy opid and index
        if (has_local_precondition(
                it->first, usage, expr, 0 /*opid*/, 0 /*index*/, expr_covers,
                true /*copy user*/, &dominated))
        {
          last_events.insert(it->first->term_event);
          if (dominated)
            observed |= overlap;
          else
            non_dominated |= overlap;
        }
        else
          non_dominated |= overlap;
      }
    }

    //--------------------------------------------------------------------------
    void SpaceView::find_previous_preconditions(
        const RegionUsage& usage, const FieldMask& mask,
        IndexSpaceExpression* expr, const bool expr_covers,
        std::set<ApEvent>& last_users) const
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      if (mask * previous_epoch_users.get_valid_mask())
        return;
      for (shrt::FieldMaskMap<PhysicalUser>::const_iterator it =
               previous_epoch_users.begin();
           it != previous_epoch_users.end(); it++)
      {
        const FieldMask overlap = mask & it->second;
        if (!overlap)
          continue;
        // We're just reading these and we want to see all prior
        // dependences so just give dummy opid and index
        if (has_local_precondition(
                it->first, usage, expr, 0 /*opid*/, 0 /*index*/, expr_covers,
                true /*copy user*/))
          last_users.insert(it->first->term_event);
      }
    }

    //--------------------------------------------------------------------------
    void SpaceView::find_previous_filter_users(
        const FieldMask& dom_mask,
        local::FieldMaskMap<PhysicalUser>& filter_users)
    //--------------------------------------------------------------------------
    {
      // Lock better be held by caller
      if (dom_mask * previous_epoch_users.get_valid_mask())
        return;
      for (shrt::FieldMaskMap<PhysicalUser>::const_iterator it =
               previous_epoch_users.begin();
           it != previous_epoch_users.end(); it++)
      {
        const FieldMask overlap = dom_mask & it->second;
        if (!overlap)
          continue;
        if (filter_users.insert(it->first, overlap))
          it->first->add_reference();
      }
    }

    /////////////////////////////////////////////////////////////
    // PartitionView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PartitionView::PartitionView(IndexPartNode* n, IndividualView* v)
      : NodeView(n, v), node(n)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PartitionView::~PartitionView(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(subviews.empty());
    }

    //--------------------------------------------------------------------------
    bool PartitionView::is_empty(void) const
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock, false /*exclusive*/);
      return subviews.empty();
    }

    //--------------------------------------------------------------------------
    void PartitionView::invalidate_users(void)
    //--------------------------------------------------------------------------
    {
      // Shouldn't be any races on deleteions so no need for the lock
      for (lng::FieldMaskMap<
               SpaceView, ViewComparator<SpaceView> >::const_iterator it =
               subviews.begin();
           it != subviews.end(); it++)
      {
        it->first->invalidate_users();
        if (it->first->remove_reference())
          delete it->first;
      }
      subviews.clear();
    }

    //--------------------------------------------------------------------------
    void PartitionView::find_last_users(
        const RegionUsage& usage, IndexSpaceExpression* expr,
        const bool expr_dominates, const FieldMask& mask,
        std::set<ApEvent>& last_events) const
    //--------------------------------------------------------------------------
    {
      // First find the interfering children for this partition
      local::FieldMaskMap<SpaceView> to_traverse;
      find_subviews_to_traverse(expr, expr_dominates, mask, to_traverse);
      for (local::FieldMaskMap<SpaceView>::const_iterator it =
               to_traverse.begin();
           it != to_traverse.end(); it++)
      {
        if (expr_dominates || (expr == it->first->node))
          it->first->find_last_users(
              usage, expr, true /*expr dominates*/, it->second, last_events);
        else  // Test for whether the expression dominates or not
          it->first->find_last_users(
              usage, expr, it->first->dominated_by(expr), it->second,
              last_events);
        if (it->first->remove_reference())
          delete it->first;
      }
    }

    //--------------------------------------------------------------------------
    bool PartitionView::find_subviews_to_traverse(
        IndexSpaceExpression* expr, bool expr_dominates, const FieldMask& mask,
        local::FieldMaskMap<SpaceView>& to_traverse) const
    //--------------------------------------------------------------------------
    {
      if (expr_dominates)
      {
        // Interferes with everything
        AutoLock v_lock(view_lock, false /*exclusive*/);
        if (subviews.empty())
          return true;
        if (mask * subviews.get_valid_mask())
          return false;
        for (lng::FieldMaskMap<
                 SpaceView, ViewComparator<SpaceView> >::const_iterator it =
                 subviews.begin();
             it != subviews.end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          if (to_traverse.insert(it->first, overlap))
            it->first->add_reference();
        }
      }
      else
      {
        std::vector<LegionColor> interfering_colors;
        // This is where the magic happens because this query is
        // backed by an acceleration data structure
        node->find_interfering_children(expr, interfering_colors);
        if (!interfering_colors.empty())
        {
          AutoLock v_lock(view_lock, false /*exclusive*/);
          if (subviews.empty())
            return true;
          if (mask * subviews.get_valid_mask())
            return false;
          if (interfering_colors.size() < subviews.size())
          {
            for (const LegionColor& color : interfering_colors)
            {
              lng::FieldMaskMap<SpaceView, ViewComparator<SpaceView> >::
                  const_iterator finder = subviews.find(color);
              if (finder != subviews.end())
              {
                const FieldMask overlap = mask & finder->second;
                if (!overlap)
                  continue;
                if (to_traverse.insert(finder->first, overlap))
                  finder->first->add_reference();
              }
            }
          }
          else
          {
            std::sort(interfering_colors.begin(), interfering_colors.end());
            for (lng::FieldMaskMap<
                     SpaceView, ViewComparator<SpaceView> >::const_iterator it =
                     subviews.begin();
                 it != subviews.end(); it++)
            {
              const FieldMask overlap = mask & it->second;
              if (!overlap)
                continue;
              if (!std::binary_search(
                      interfering_colors.begin(), interfering_colors.end(),
                      it->first->node->color))
                continue;
              if (to_traverse.insert(it->first, overlap))
                it->first->add_reference();
            }
          }
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool PartitionView::find_user_preconditions(
        const RegionUsage& usage, IndexSpaceExpression* user_expr,
        const bool expr_dominates, const FieldMask& user_mask,
        ApEvent term_event, UniqueID op_id, unsigned index,
        std::set<ApEvent>& preconditions, const bool trace_recording)
    //--------------------------------------------------------------------------
    {
      // First find the interfering children for this partition
      local::FieldMaskMap<SpaceView> to_traverse;
      const bool empty = find_subviews_to_traverse(
          user_expr, expr_dominates, user_mask, to_traverse);
      for (local::FieldMaskMap<SpaceView>::const_iterator it =
               to_traverse.begin();
           it != to_traverse.end(); it++)
      {
        const bool child_dominated = expr_dominates ||
                                     (user_expr == it->first->node) ||
                                     it->first->dominated_by(user_expr);
        unsigned refs_to_remove = 1;
        if (it->first->find_user_preconditions(
                usage, user_expr, child_dominated, it->second, term_event,
                op_id, index, preconditions, trace_recording))
        {
          AutoLock v_lock(view_lock);
          lng::FieldMaskMap<SpaceView, ViewComparator<SpaceView> >::iterator
              finder = subviews.find(it->first);
          if ((finder != subviews.end()) && (it->first == finder->first) &&
              it->first->is_empty())
          {
            subviews.erase(finder);
            refs_to_remove++;
          }
        }
        if (it->first->remove_reference(refs_to_remove))
          delete it->first;
      }
      return empty;
    }

    //--------------------------------------------------------------------------
    bool PartitionView::find_copy_preconditions(
        const RegionUsage& usage, IndexSpaceExpression* copy_expr,
        const bool expr_dominates, const FieldMask& copy_mask, UniqueID op_id,
        unsigned index, std::set<ApEvent>& preconditions,
        const bool trace_recording)
    //--------------------------------------------------------------------------
    {
      local::FieldMaskMap<SpaceView> to_traverse;
      const bool empty = find_subviews_to_traverse(
          copy_expr, expr_dominates, copy_mask, to_traverse);
      for (local::FieldMaskMap<SpaceView>::const_iterator it =
               to_traverse.begin();
           it != to_traverse.end(); it++)
      {
        const bool child_dominated = expr_dominates ||
                                     (copy_expr == it->first->node) ||
                                     it->first->dominated_by(copy_expr);
        unsigned refs_to_remove = 1;
        if (it->first->find_copy_preconditions(
                usage, copy_expr, child_dominated, it->second, op_id, index,
                preconditions, trace_recording))
        {
          AutoLock v_lock(view_lock);
          lng::FieldMaskMap<SpaceView, ViewComparator<SpaceView> >::iterator
              finder = subviews.find(it->first);
          if ((finder != subviews.end()) && (it->first == finder->first) &&
              it->first->is_empty())
          {
            subviews.erase(finder);
            refs_to_remove++;
          }
        }
        if (it->first->remove_reference(refs_to_remove))
          delete it->first;
      }
      return empty;
    }

    //--------------------------------------------------------------------------
    void PartitionView::insert_child(
        NodeView* child, const FieldMask& child_mask)
    //--------------------------------------------------------------------------
    {
      SpaceView* space_child = legion_safe_cast<SpaceView*>(child);
      legion_assert(space_child->node->parent == node);
      AutoLock v_lock(view_lock);
      if (subviews.insert(space_child, child_mask))
        space_child->add_reference();
    }

    //--------------------------------------------------------------------------
    void PartitionView::insert_user(
        PhysicalUser* user, const FieldMask& user_mask,
        local::vector<LegionColor>& path, AutoLock& parent_lock)
    //--------------------------------------------------------------------------
    {
      legion_assert(!path.empty());
      // Quick test to see if we can do this read-only
      {
        AutoLock v_lock(view_lock, false /*exclusive*/);
        lng::FieldMaskMap<SpaceView, ViewComparator<SpaceView> >::const_iterator
            finder = subviews.find(path.back());
        if ((finder != subviews.end()) && !(user_mask - finder->second))
        {
          parent_lock.release();
          path.pop_back();
          finder->first->insert_user(user, user_mask, path, v_lock);
          return;
        }
      }
      // Do hand-over-hand locking
      AutoLock v_lock(view_lock);
      parent_lock.release();
      const LegionColor color = path.back();
      path.pop_back();
      // See if we already have it or whether we need to make it
      lng::FieldMaskMap<SpaceView, ViewComparator<SpaceView> >::iterator
          finder = subviews.find(color);
      if (finder == subviews.end())
      {
        // Need to make one since it doesn't exist yet
        SpaceView* child = new SpaceView(node->get_child(color), view);
        if (subviews.insert(child, user_mask))
          child->add_reference();
        child->insert_user(user, user_mask, path, v_lock);
      }
      else
      {
        finder.merge(user_mask);
        finder->first->insert_user(user, user_mask, path, v_lock);
      }
    }

    /////////////////////////////////////////////////////////////
    // IndividualView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndividualView::IndividualView(
        DistributedID did, PhysicalManager* man, AddressSpaceID log_owner,
        bool register_now, CollectiveMapping* mapping)
      : InstanceView(did, register_now, mapping), manager(man),
        logical_owner(log_owner)
    //--------------------------------------------------------------------------
    {
      legion_assert(manager != nullptr);
      // Keep the manager from being collected
      manager->add_nested_resource_ref(did);
      manager->add_nested_gc_ref(did);
    }

    //--------------------------------------------------------------------------
    IndividualView::~IndividualView(void)
    //--------------------------------------------------------------------------
    {
      if (is_logical_owner() && !view_reservations.empty())
      {
        RegionUsage usage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0 /*redop*/);
        std::set<ApEvent> done_events;
        for (lng::FieldMaskMap<NodeView>::const_iterator it = roots.begin();
             it != roots.end(); it++)
          it->first->find_last_users(
              usage,
              it->first->tree_node->is_index_space_node() ?
                  it->first->tree_node->as_index_space_node() :
                  it->first->tree_node->as_index_part_node()->parent,
              true /*dominates*/, it->second, done_events);
        const ApEvent all_done = Runtime::merge_events(nullptr, done_events);
        // No need for the lock here since we should be in a destructor
        // and there should be no more races
        for (std::pair<const unsigned, Reservation>& reservation :
             view_reservations)
          reservation.second.destroy_reservation(all_done);
      }
      for (lng::FieldMaskMap<NodeView>::const_iterator it = roots.begin();
           it != roots.end(); it++)
      {
        it->first->invalidate_users();
        if (it->first->remove_reference())
          delete it->first;
      }
      if (manager->remove_nested_resource_ref(did))
        delete manager;
    }

    //--------------------------------------------------------------------------
    void IndividualView::notify_valid(void)
    //--------------------------------------------------------------------------
    {
      legion_no_skip_assert(manager->acquire_instance(did));
      add_base_gc_ref(INTERNAL_VALID_REF);
    }

    //--------------------------------------------------------------------------
    bool IndividualView::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      manager->remove_nested_valid_ref(did);
      return remove_base_gc_ref(INTERNAL_VALID_REF);
    }

    //--------------------------------------------------------------------------
    void IndividualView::notify_local(void)
    //--------------------------------------------------------------------------
    {
      manager->remove_nested_gc_ref(did);
    }

    //--------------------------------------------------------------------------
    void IndividualView::pack_valid_ref(
        shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
        shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks)
    //--------------------------------------------------------------------------
    {
      if (view_lamport_clocks.find(this) == view_lamport_clocks.end())
        pack_global_ref(view_lamport_clocks[this]);
      if (inst_lamport_clocks.find(manager) == inst_lamport_clocks.end())
        manager->pack_valid_ref(inst_lamport_clocks[manager]);
    }

    //--------------------------------------------------------------------------
    void IndividualView::unpack_valid_ref(
        shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
        shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks)
    //--------------------------------------------------------------------------
    {
      shrt::map<LogicalView*, LamportClock>::iterator view_finder =
          view_lamport_clocks.find(this);
      if (view_finder != view_lamport_clocks.end())
      {
        unpack_global_ref(view_finder->second);
        view_lamport_clocks.erase(view_finder);
      }
      shrt::map<PhysicalManager*, LamportClock>::iterator inst_finder =
          inst_lamport_clocks.find(manager);
      if (inst_finder != inst_lamport_clocks.end())
      {
        manager->unpack_valid_ref(inst_finder->second);
        inst_lamport_clocks.erase(inst_finder);
      }
    }

    //--------------------------------------------------------------------------
    AddressSpaceID IndividualView::get_analysis_space(
        PhysicalManager* instance) const
    //--------------------------------------------------------------------------
    {
      legion_assert(instance == manager);
      return logical_owner;
    }

    //--------------------------------------------------------------------------
    bool IndividualView::aliases(InstanceView* other) const
    //--------------------------------------------------------------------------
    {
      if (other == this)
        return true;
      if (other->is_collective_view())
      {
        CollectiveView* collective = other->as_collective_view();
        return std::binary_search(
            collective->instances.begin(), collective->instances.end(),
            manager->did);
      }
      else
        return (this == other);
    }

    //--------------------------------------------------------------------------
    ApEvent IndividualView::fill_from(
        FillView* fill_view, ApEvent precondition, PredEvent predicate_guard,
        IndexSpaceExpression* fill_expression, Operation* op,
        const unsigned index, const IndexSpace collective_match_space,
        const FieldMask& fill_mask, const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
        CopyAcrossHelper* across_helper, const bool manage_dst_events,
        const bool fill_restricted, const bool need_valid_return)
    //--------------------------------------------------------------------------
    {
      legion_assert((across_helper == nullptr) || !manage_dst_events);
      // Compute the precondition first
      if (manage_dst_events)
      {
        ApEvent dst_precondition = find_copy_preconditions(
            false /*reading*/, 0 /*redop*/, fill_mask, fill_expression,
            op->get_unique_op_id(), index, applied_events, trace_info);
        if (dst_precondition.exists())
        {
          if (precondition.exists())
            precondition = Runtime::merge_events(
                &trace_info, precondition, dst_precondition);
          else
            precondition = dst_precondition;
        }
      }
      std::vector<CopySrcDstField> dst_fields;
      if (across_helper != nullptr)
      {
        const FieldMask src_mask = across_helper->convert_dst_to_src(fill_mask);
        across_helper->compute_across_offsets(src_mask, dst_fields);
      }
      else
        manager->compute_copy_offsets(fill_mask, dst_fields);
      const ApEvent result = fill_view->issue_fill(
          op, fill_expression, this, fill_mask, trace_info, dst_fields,
          applied_events, manager, precondition, predicate_guard,
          COLLECTIVE_NONE, fill_restricted);
      // Save the result
      if (manage_dst_events && result.exists())
        add_copy_user(
            false /*reading*/, 0 /*redop*/, result, fill_mask, fill_expression,
            collective_match_space, op->get_unique_op_id(), index,
            recorded_events, trace_info.recording, runtime->address_space);
      return result;
    }

    //--------------------------------------------------------------------------
    ApEvent IndividualView::copy_from(
        InstanceView* src_view, ApEvent precondition, PredEvent predicate_guard,
        ReductionOpID reduction_op_id, IndexSpaceExpression* copy_expression,
        Operation* op, const unsigned index,
        const IndexSpace collective_match_space, const FieldMask& copy_mask,
        PhysicalManager* src_point, const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
        CopyAcrossHelper* across_helper, const bool manage_dst_events,
        const bool copy_restricted, const bool need_valid_return)
    //--------------------------------------------------------------------------
    {
      legion_assert((across_helper == nullptr) || !manage_dst_events);
      // Compute the preconditions first
      const UniqueID op_id = op->get_unique_op_id();
      // We'll need to compute our destination precondition no matter what
      if (manage_dst_events)
      {
        const ApEvent dst_pre = find_copy_preconditions(
            false /*reading*/, reduction_op_id, copy_mask, copy_expression,
            op_id, index, applied_events, trace_info);
        if (dst_pre.exists())
        {
          if (precondition.exists())
            precondition =
                Runtime::merge_events(&trace_info, precondition, dst_pre);
          else
            precondition = dst_pre;
        }
      }
      FieldMask across_mask;
      if (across_helper != nullptr)
        across_mask = across_helper->convert_dst_to_src(copy_mask);
      const FieldMask& src_mask =
          (across_helper == nullptr) ? copy_mask : across_mask;
      // Several cases here:
      // 1. The source is another individual manager - just straight up
      //    compute the dependences and do the copy or reduction
      // 2. The source is a normal collective manager - issue a copy from
      //    an instance close to the destination instance
      // 3. The source is a reduction collective manager - build a reduction
      //    tree down to a source instance close to the destination instance
      ApEvent result;
      if (src_view->is_individual_view())
      {
        IndividualView* source_view = src_view->as_individual_view();
        // Case 1: Source manager is another instance manager
        const ApEvent src_pre = source_view->find_copy_preconditions(
            true /*reading*/, 0 /*redop*/, src_mask, copy_expression, op_id,
            index, applied_events, trace_info);
        if (src_pre.exists())
        {
          if (precondition.exists())
            precondition =
                Runtime::merge_events(&trace_info, precondition, src_pre);
          else
            precondition = src_pre;
        }
        // Compute the field offsets
        std::vector<CopySrcDstField> dst_fields, src_fields;
        if (across_helper == nullptr)
          manager->compute_copy_offsets(copy_mask, dst_fields);
        else
          across_helper->compute_across_offsets(src_mask, dst_fields);
        PhysicalManager* source_manager = source_view->get_manager();
        legion_assert(manager->instance.id != source_manager->instance.id);
        source_manager->compute_copy_offsets(src_mask, src_fields);
        std::vector<Reservation> reservations;
        // If we're doing a reduction operation then set the reduction
        // information on the source-dst fields
        if (reduction_op_id > 0)
        {
          legion_assert((get_redop() == 0) || (get_redop() == reduction_op_id));
          // Get the reservations
          find_field_reservations(copy_mask, reservations);
          // Set the redop on the destination fields
          // Note that we can mark these as exclusive copies since
          // we are protecting them with the reservations
          for (unsigned idx = 0; idx < dst_fields.size(); idx++)
            dst_fields[idx].set_redop(
                reduction_op_id, (get_redop() > 0) /*fold*/,
                true /*exclusive*/);
        }
        result = copy_expression->issue_copy(
            op, trace_info, dst_fields, src_fields, reservations,
            source_manager->tree_id, manager->tree_id, precondition,
            predicate_guard, source_manager->get_unique_event(),
            manager->get_unique_event(), COLLECTIVE_NONE, copy_restricted);
        if (result.exists())
        {
          source_view->add_copy_user(
              true /*reading*/, 0 /*redop*/, result, src_mask, copy_expression,
              collective_match_space, op_id, index, recorded_events,
              trace_info.recording, runtime->address_space);
          if (manage_dst_events)
            add_copy_user(
                false /*reading*/, reduction_op_id, result, copy_mask,
                copy_expression, collective_match_space, op_id, index,
                recorded_events, trace_info.recording, runtime->address_space);
        }
        if (trace_info.recording)
        {
          const UniqueInst src_inst(source_view);
          const UniqueInst dst_inst(this);
          trace_info.record_copy_insts(
              result, copy_expression, src_inst, dst_inst, src_mask, copy_mask,
              reduction_op_id, applied_events);
        }
      }
      else
      {
        CollectiveView* collective = src_view->as_collective_view();
        std::vector<CopySrcDstField> dst_fields;
        if (across_helper == nullptr)
          manager->compute_copy_offsets(copy_mask, dst_fields);
        else
          across_helper->compute_across_offsets(src_mask, dst_fields);
        std::vector<Reservation> reservations;
        if (reduction_op_id > 0)
        {
          legion_assert((get_redop() == 0) || (get_redop() == reduction_op_id));
          find_field_reservations(copy_mask, reservations);
          // Set the redop on the destination fields
          // Note that we can mark these as exclusive copies since
          // we are protecting them with the reservations
          for (unsigned idx = 0; idx < dst_fields.size(); idx++)
            dst_fields[idx].set_redop(
                reduction_op_id, (get_redop() > 0) /*fold*/,
                true /*exclusive*/);
        }
        if (collective->is_allreduce_view())
        {
          legion_assert(reduction_op_id == collective->get_redop());
          AllreduceView* allreduce = collective->as_allreduce_view();
          // Case 3
          // This is subtle as fuck
          // In the normal case where we're doing a reduction from a
          // collective instance to a normal instance then we can get
          // away with just building the reduction tree.
          //
          // An important note here: we only need to build a reduction tree
          // and not do an all-reduce for the collective reduction instance
          // because we know the equivalence set code above will only ever
          // issue a single copy from a reduction instance into another
          // instance before that reduction instance is refreshed, so it
          // is safe to break the invariant that all instances in the
          // collective manager have the same data.
          //
          // However, in the case where we are doing a copy-across, then we
          // might still be asked to do an intra-region reduction later so
          // it's unsafe to do the partial accumulations into our own
          // instances. Therefore for now we will hammer all the source
          // instances into the destination instance without any
          // intermediate reductions.
          const UniqueInst dst_inst(this);
          if (manage_dst_events)
          {
            // Reduction-tree case
            const AddressSpaceID origin =
                (src_point != nullptr) ?
                    src_point->owner_space :
                    collective->select_source_space(owner_space);
            // There will always be a single result for this copy
            if (origin != local_space)
            {
              const RtUserEvent recorded = Runtime::create_rt_user_event();
              const RtUserEvent applied = Runtime::create_rt_user_event();
              CollectiveDistributeReduction rez;
              {
                RezCheck z(rez);
                rez.serialize(allreduce->did);
                pack_fields(rez, dst_fields);
                rez.serialize<size_t>(reservations.size());
                for (unsigned idx = 0; idx < reservations.size(); idx++)
                  rez.serialize(reservations[idx]);
                rez.serialize(precondition);
                rez.serialize(predicate_guard);
                copy_expression->pack_expression(rez, origin);
                rez.serialize(collective_match_space);
                op->pack_remote_operation(rez, origin, applied_events);
                rez.serialize(index);
                rez.serialize(src_mask);
                rez.serialize(copy_mask);
                if (src_point != nullptr)
                  rez.serialize(src_point->did);
                else
                  rez.serialize<DistributedID>(0);
                dst_inst.serialize(rez);
                trace_info.pack_trace_info(rez);
                rez.serialize(recorded);
                rez.serialize(applied);
                if (trace_info.recording)
                {
                  ApBarrier bar;
                  ShardID sid =
                      trace_info.record_barrier_creation(bar, 1 /*arrivals*/);
                  rez.serialize(bar);
                  rez.serialize(sid);
                  result = bar;
                }
                else
                {
                  const ApUserEvent to_trigger =
                      Runtime::create_ap_user_event(&trace_info);
                  result = to_trigger;
                  rez.serialize(to_trigger);
                }
                rez.serialize(origin);
                rez.serialize(COLLECTIVE_REDUCTION);
              }
              rez.dispatch(origin);
              recorded_events.insert(recorded);
              applied_events.insert(applied);
            }
            else
            {
              const ApUserEvent to_trigger =
                  Runtime::create_ap_user_event(&trace_info);
              result = to_trigger;
              allreduce->perform_collective_reduction(
                  dst_fields, reservations, precondition, predicate_guard,
                  copy_expression, collective_match_space, op, index, src_mask,
                  copy_mask, (src_point != nullptr) ? src_point->did : 0,
                  dst_inst, manager->get_unique_event(), trace_info,
                  COLLECTIVE_REDUCTION, recorded_events, applied_events,
                  to_trigger, origin);
            }
          }
          else
          {
            // Hammer reduction case
            // Issue a performance warning if we're ever going to
            // be doing this case and the number of instance is large
            if (collective->instances.size() > LEGION_COLLECTIVE_RADIX)
            {
              Warning warning;
              warning
                  << "WARNING: Performing copy-across reduction hammer with "
                  << collective->instances.size()
                  << " instances into a single instance "
                  << "from collective manager " << std::hex << collective->did
                  << " to normal manager " << did << std::dec
                  << ". Please report this use case to the Legion developers' "
                     "mailing list.";
              warning.raise();
            }
            const AddressSpaceID origin =
                collective->select_source_space(owner_space);
            if (origin != local_space)
            {
              const RtUserEvent recorded = Runtime::create_rt_user_event();
              const RtUserEvent applied = Runtime::create_rt_user_event();
              CollectiveHammerReduction rez;
              {
                RezCheck z(rez);
                rez.serialize(allreduce->did);
                pack_fields(rez, dst_fields);
                rez.serialize<size_t>(reservations.size());
                for (unsigned idx = 0; idx < reservations.size(); idx++)
                  rez.serialize(reservations[idx]);
                rez.serialize(precondition);
                rez.serialize(predicate_guard);
                copy_expression->pack_expression(rez, origin);
                rez.serialize(collective_match_space);
                op->pack_remote_operation(rez, origin, applied_events);
                rez.serialize(index);
                rez.serialize(src_mask);
                rez.serialize(copy_mask);
                dst_inst.serialize(rez);
                trace_info.pack_trace_info(rez);
                rez.serialize(recorded);
                rez.serialize(applied);
                if (trace_info.recording)
                {
                  ApBarrier bar;
                  ShardID sid =
                      trace_info.record_barrier_creation(bar, 1 /*arrivals*/);
                  rez.serialize(bar);
                  rez.serialize(sid);
                  result = bar;
                }
                else
                {
                  const ApUserEvent to_trigger =
                      Runtime::create_ap_user_event(&trace_info);
                  rez.serialize(to_trigger);
                  result = to_trigger;
                }
                rez.serialize(origin);
              }
              rez.dispatch(origin);
              recorded_events.insert(recorded);
              applied_events.insert(applied);
            }
            else
              result = allreduce->perform_hammer_reduction(
                  dst_fields, reservations, precondition, predicate_guard,
                  copy_expression, collective_match_space, op, index, src_mask,
                  copy_mask, dst_inst, manager->get_unique_event(), trace_info,
                  recorded_events, applied_events, origin);
          }
        }
        else
        {
          // Case 2
          // We can issue the copy from an instance in the source
          const Memory location = manager->memory_manager->memory;
          const DomainPoint no_point;
          const AddressSpaceID origin =
              (src_point != nullptr) ?
                  src_point->owner_space :
                  collective->select_source_space(owner_space);
          const UniqueInst dst_inst(this);
          if (origin != local_space)
          {
            const RtUserEvent recorded = Runtime::create_rt_user_event();
            const RtUserEvent applied = Runtime::create_rt_user_event();
            ApUserEvent to_trigger = Runtime::create_ap_user_event(&trace_info);
            CollectiveDistributePoint rez;
            {
              RezCheck z(rez);
              rez.serialize(collective->did);
              pack_fields(rez, dst_fields);
              rez.serialize<size_t>(reservations.size());
              for (unsigned idx = 0; idx < reservations.size(); idx++)
                rez.serialize(reservations[idx]);
              rez.serialize(precondition);
              rez.serialize(predicate_guard);
              copy_expression->pack_expression(rez, origin);
              rez.serialize(collective_match_space);
              op->pack_remote_operation(rez, origin, applied_events);
              rez.serialize(index);
              rez.serialize(src_mask);
              rez.serialize(copy_mask);
              rez.serialize(location);
              dst_inst.serialize(rez);
              if (src_point != nullptr)
                rez.serialize(src_point->did);
              else
                rez.serialize<DistributedID>(0);
              trace_info.pack_trace_info(rez);
              rez.serialize(COLLECTIVE_NONE);
              rez.serialize(recorded);
              rez.serialize(applied);
              rez.serialize(to_trigger);
            }
            rez.dispatch(origin);
            recorded_events.insert(recorded);
            applied_events.insert(applied);
            result = to_trigger;
          }
          else
            result = collective->perform_collective_point(
                dst_fields, reservations, precondition, predicate_guard,
                copy_expression, collective_match_space, op, index, src_mask,
                copy_mask, location, dst_inst, manager->get_unique_event(),
                (src_point != nullptr) ? src_point->did : 0, trace_info,
                recorded_events, applied_events);
        }
        if (result.exists() && manage_dst_events)
          add_copy_user(
              false /*reading*/, reduction_op_id, result, copy_mask,
              copy_expression, collective_match_space, op_id, index,
              recorded_events, trace_info.recording, runtime->address_space);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void IndividualView::add_initial_user(
        ApEvent term_event, const RegionUsage& usage,
        const FieldMask& user_mask, IndexSpaceNode* user_expr,
        const UniqueID op_id, const unsigned index)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_logical_owner());
      add_internal_task_user(
          usage, user_expr, user_mask, term_event, op_id, index);
    }

    //--------------------------------------------------------------------------
    ApEvent IndividualView::register_user(
        const RegionUsage& usage, const FieldMask& user_mask,
        IndexSpaceNode* user_expr, const UniqueID op_id,
        const size_t op_ctx_index, const unsigned index, ApEvent term_event,
        PhysicalManager* target, CollectiveMapping* analysis_mapping,
        size_t local_collective_arrivals, std::vector<RtEvent>& registered,
        std::set<RtEvent>& applied_events, const PhysicalTraceInfo& trace_info,
        const AddressSpaceID source, const bool symbolic /*=false*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(target == manager);
      // Handle the collective rendezvous if necessary
      if (local_collective_arrivals > 0)
        return register_collective_user(
            usage, user_mask, user_expr, op_id, op_ctx_index, index, term_event,
            target, analysis_mapping, local_collective_arrivals, registered,
            applied_events, trace_info, symbolic);
      // Quick test for empty index space expressions
      if (!symbolic && user_expr->is_empty())
      {
        manager->record_instance_user(term_event, applied_events);
        return manager->get_use_event(term_event);
      }
      if (!is_logical_owner())
      {
        ApUserEvent ready_event;
        // Check to see if this user came from somewhere that wasn't
        // the logical owner, if so we need to send the update back
        // to the owner to be handled
        if (source != logical_owner)
        {
          // If we're not the logical owner send a message there
          // to do the analysis and provide a user event to trigger
          // with the precondition
          ready_event = Runtime::create_ap_user_event(&trace_info);
          RtUserEvent registered_event = Runtime::create_rt_user_event();
          RtUserEvent applied_event = Runtime::create_rt_user_event();
          ViewRegisterUser rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(target->did);
            rez.serialize(usage);
            rez.serialize(user_mask);
            rez.serialize(user_expr->handle);
            rez.serialize(op_id);
            rez.serialize(op_ctx_index);
            rez.serialize(index);
            rez.serialize(term_event);
            rez.serialize(local_collective_arrivals);
            rez.serialize(ready_event);
            rez.serialize(registered_event);
            rez.serialize(applied_event);
            trace_info.pack_trace_info(rez);
          }
          rez.dispatch(logical_owner);
          registered.emplace_back(registered_event);
          applied_events.insert(applied_event);
        }
        return ready_event;
      }
      else
      {
        // Now we can do our local analysis
        std::set<ApEvent> wait_on_events;
        ApEvent start_use_event = manager->get_use_event(term_event);
        if (start_use_event.exists())
          wait_on_events.insert(start_use_event);
        // Find the preconditions
        local::FieldMaskMap<NodeView> to_traverse;
        {
          AutoLock v_lock(view_lock, false /*exclusive*/);
          for (lng::FieldMaskMap<NodeView>::const_iterator it = roots.begin();
               it != roots.end(); it++)
          {
            const FieldMask overlap = user_mask & it->second;
            if (!overlap)
              continue;
            to_traverse.insert(it->first, overlap);
            it->first->add_reference();
          }
        }
        for (local::FieldMaskMap<NodeView>::const_iterator it =
                 to_traverse.begin();
             it != to_traverse.end(); it++)
        {
          // First check that they overlap
          IndexSpaceNode* parent =
              it->first->tree_node->is_index_space_node() ?
                  it->first->tree_node->as_index_space_node() :
                  it->first->tree_node->as_index_part_node()->parent;
          IndexSpaceExpression* overlap =
              (parent == user_expr) ?
                  user_expr :
                  runtime->intersect_index_spaces(parent, user_expr);
          unsigned refs_to_remove = 1;
          const size_t overlap_volume = overlap->get_volume();
          if ((overlap_volume > 0) &&
              it->first->find_user_preconditions(
                  usage, user_expr, (overlap_volume == parent->get_volume()),
                  it->second, term_event, op_id, index, wait_on_events,
                  trace_info.recording))
          {
            AutoLock v_lock(view_lock);
            lng::FieldMaskMap<NodeView>::iterator finder =
                roots.find(it->first);
            if ((finder != roots.end()) && it->first->is_empty())
            {
              roots.erase(finder);
              refs_to_remove++;
            }
          }
          if (it->first->remove_reference(refs_to_remove))
            delete it->first;
        }
        // Add our local user
        add_internal_task_user(
            usage, user_expr, user_mask, term_event, op_id, index);
        manager->record_instance_user(term_event, applied_events);
        // At this point tasks shouldn't be allowed to wait on themselves
        legion_assert(
            !term_event.exists() ||
            (wait_on_events.find(term_event) == wait_on_events.end()));
        // Return the merge of the events
        if (!wait_on_events.empty())
          return Runtime::merge_events(&trace_info, wait_on_events);
        else
          return ApEvent::NO_AP_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    ApEvent IndividualView::find_copy_preconditions(
        bool reading, ReductionOpID redop, const FieldMask& copy_mask,
        IndexSpaceExpression* copy_expr, UniqueID op_id, unsigned index,
        std::set<RtEvent>& applied_events, const PhysicalTraceInfo& trace_info)
    //--------------------------------------------------------------------------
    {
      if (!is_logical_owner())
      {
        // Check to see if there are any replicated fields here which we
        // can handle locally so we don't have to send a message to the owner
        ApEvent result_event;
        FieldMask request_mask(copy_mask);
        {
          // All the fields are not local, first send the request to
          // the owner to do the analysis since we're going to need
          // to do that anyway, then issue any request for replicated
          // fields to be moved to this node and record it as a
          // precondition for the mapping
          ApUserEvent ready_event = Runtime::create_ap_user_event(&trace_info);
          RtUserEvent applied = Runtime::create_rt_user_event();
          ViewFindCopyPreMessage rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize<bool>(reading);
            rez.serialize(redop);
            rez.serialize(copy_mask);
            copy_expr->pack_expression(rez, logical_owner);
            rez.serialize(op_id);
            rez.serialize(index);
            rez.serialize(ready_event);
            rez.serialize(applied);
            trace_info.pack_trace_info(rez);
          }
          rez.dispatch(logical_owner);
          applied_events.insert(applied);
          result_event = ready_event;
        }
        return result_event;
      }
      else
      {
        // In the case where we're the owner we can just handle
        // this without needing to do anything
        std::set<ApEvent> preconditions;
        const ApEvent start_use_event = manager->get_use_event();
        if (start_use_event.exists())
          preconditions.insert(start_use_event);
        const RegionUsage usage(
            reading     ? LEGION_READ_ONLY :
            (redop > 0) ? LEGION_REDUCE :
                          LEGION_READ_WRITE,
            LEGION_EXCLUSIVE, redop);
        // Find the preconditions
        local::FieldMaskMap<NodeView> to_traverse;
        {
          AutoLock v_lock(view_lock, false /*exclusive*/);
          for (lng::FieldMaskMap<NodeView>::const_iterator it = roots.begin();
               it != roots.end(); it++)
          {
            const FieldMask overlap = copy_mask & it->second;
            if (!overlap)
              continue;
            to_traverse.insert(it->first, overlap);
            it->first->add_reference();
          }
        }
        for (local::FieldMaskMap<NodeView>::const_iterator it =
                 to_traverse.begin();
             it != to_traverse.end(); it++)
        {
          // First check that they overlap
          IndexSpaceNode* parent =
              it->first->tree_node->is_index_space_node() ?
                  it->first->tree_node->as_index_space_node() :
                  it->first->tree_node->as_index_part_node()->parent;
          IndexSpaceExpression* overlap =
              (parent == copy_expr) ?
                  copy_expr :
                  runtime->intersect_index_spaces(parent, copy_expr);
          unsigned refs_to_remove = 1;
          const size_t overlap_volume = overlap->get_volume();
          if ((overlap_volume > 0) &&
              it->first->find_copy_preconditions(
                  usage, copy_expr, (overlap_volume == parent->get_volume()),
                  it->second, op_id, index, preconditions,
                  trace_info.recording))
          {
            AutoLock v_lock(view_lock);
            lng::FieldMaskMap<NodeView>::iterator finder =
                roots.find(it->first);
            if ((finder != roots.end()) && it->first->is_empty())
            {
              roots.erase(finder);
              refs_to_remove++;
            }
          }
          if (it->first->remove_reference(refs_to_remove))
            delete it->first;
        }
        if (preconditions.empty())
          return ApEvent::NO_AP_EVENT;
        return Runtime::merge_events(&trace_info, preconditions);
      }
    }

    //--------------------------------------------------------------------------
    void IndividualView::add_copy_user(
        bool reading, ReductionOpID redop, ApEvent term_event,
        const FieldMask& copy_mask, IndexSpaceExpression* copy_expr,
        IndexSpace upper_bound, UniqueID op_id, unsigned index,
        std::set<RtEvent>& applied_events, const bool trace_recording,
        const AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      legion_assert(upper_bound.exists());
      if (!is_logical_owner())
      {
        // Check to see if this update came from some place other than the
        // source in which case we need to send it back to the source
        if (source != logical_owner)
        {
          RtUserEvent applied_event = Runtime::create_rt_user_event();
          ViewAddCopyUserMessage rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize<bool>(reading);
            rez.serialize(redop);
            rez.serialize(term_event);
            rez.serialize(copy_mask);
            copy_expr->pack_expression(rez, logical_owner);
            rez.serialize(upper_bound);
            rez.serialize(op_id);
            rez.serialize(index);
            rez.serialize(applied_event);
            rez.serialize<bool>(trace_recording);
          }
          rez.dispatch(logical_owner);
          applied_events.insert(applied_event);
        }
      }
      else
      {
        // Now we can do our local analysis
        const RegionUsage usage(
            reading     ? LEGION_READ_ONLY :
            (redop > 0) ? LEGION_REDUCE :
                          LEGION_READ_WRITE,
            (redop > 0) ? LEGION_ATOMIC : LEGION_EXCLUSIVE, redop);
        IndexSpaceNode* target = dynamic_cast<IndexSpaceNode*>(copy_expr);
        if ((target != nullptr) && target->check_valid_and_increment(did))
        {
          PhysicalUser* user = new PhysicalUser(
              usage, copy_expr, term_event, op_id, index, true /*copy user*/,
              true /*covers*/);
          add_internal_node_user(user, copy_mask, target);
          if (target->remove_nested_valid_ref(did))
            std::abort();  // should never hit this
        }
        else
        {
          // Check to see if the canonical expressin is an index space node
          IndexSpaceExpression* canonical =
              copy_expr->get_canonical_expression();
          target = dynamic_cast<IndexSpaceNode*>(canonical);
          // Need to add a valid reference to make sure we can safely use
          // this node and everything above it in the tree. If the node
          // has a parent we need to add the valid reference to the
          // partition which will keep a valid reference all the way
          // up the tree and also on its immediate children
          if ((target != nullptr) &&
              ((target->parent == nullptr) ?
                   target->check_valid_and_increment(did) :
                   target->parent->check_valid_and_increment(did)))
          {
            PhysicalUser* user = new PhysicalUser(
                usage, copy_expr, term_event, op_id, index, true /*copy user*/,
                true /*covers*/);
            add_internal_node_user(user, copy_mask, target);
            // Remove the extra valid reference that we added
            if ((target->parent == nullptr) ?
                    target->remove_nested_valid_ref(did) :
                    target->parent->remove_nested_valid_ref(did))
              std::abort();  // should never hit this
          }
          else
          {
            // Need to use the upper bound to store the user unfortunately
            // which is sound but not precise, the analysis will still be
            // precise of course, but more later users might need to test
            // against this user unnecessarily to prove non-interference
            target = runtime->get_node(upper_bound);
            PhysicalUser* user = new PhysicalUser(
                usage, copy_expr, term_event, op_id, index, true /*copy user*/,
                false /*covers*/);
            add_internal_node_user(user, copy_mask, target);
          }
        }
        manager->record_instance_user(term_event, applied_events);
      }
    }

    //--------------------------------------------------------------------------
    void IndividualView::find_last_users(
        PhysicalManager* instance, std::set<ApEvent>& events,
        const RegionUsage& usage, const FieldMask& mask,
        IndexSpaceExpression* expr, std::vector<RtEvent>& ready_events) const
    //--------------------------------------------------------------------------
    {
      legion_assert(instance == manager);
      // Check to see if we're on the right node to perform this analysis
      if (logical_owner != local_space)
      {
        const RtUserEvent ready = Runtime::create_rt_user_event();
        ViewFindLastUsersRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(manager->did);
          rez.serialize(&events);
          rez.serialize(usage);
          rez.serialize(mask);
          expr->pack_expression(rez, logical_owner);
          rez.serialize(ready);
        }
        rez.dispatch(logical_owner);
        ready_events.emplace_back(ready);
      }
      else
      {
        local::FieldMaskMap<NodeView> to_traverse;
        {
          AutoLock v_lock(view_lock, false /*exclusive*/);
          for (lng::FieldMaskMap<NodeView>::const_iterator it = roots.begin();
               it != roots.end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            to_traverse.insert(it->first, overlap);
            it->first->add_reference();
          }
        }
        for (local::FieldMaskMap<NodeView>::const_iterator it =
                 to_traverse.begin();
             it != to_traverse.end(); it++)
        {
          // First check that they overlap
          IndexSpaceNode* parent =
              it->first->tree_node->is_index_space_node() ?
                  it->first->tree_node->as_index_space_node() :
                  it->first->tree_node->as_index_part_node()->parent;
          IndexSpaceExpression* overlap =
              (parent == expr) ? expr :
                                 runtime->intersect_index_spaces(parent, expr);
          const size_t overlap_volume = overlap->get_volume();
          if (overlap_volume > 0)
            it->first->find_last_users(
                usage, expr, (overlap_volume == parent->get_volume()), mask,
                events);
          if (it->first->remove_reference())
            delete it->first;
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndividualView::add_internal_task_user(
        const RegionUsage& usage, IndexSpaceNode* user_expr,
        const FieldMask& user_mask, ApEvent term_event, UniqueID op_id,
        const unsigned index)
    //--------------------------------------------------------------------------
    {
      PhysicalUser* user = new PhysicalUser(
          usage, user_expr, term_event, op_id, index, false /*copy user*/,
          true /*covers*/);
      add_internal_node_user(user, user_mask, user_expr);
    }

    //--------------------------------------------------------------------------
    void IndividualView::add_internal_node_user(
        PhysicalUser* user, const FieldMask& user_mask,
        IndexSpaceNode* user_expr)
    //--------------------------------------------------------------------------
    {
      // Traverse upwards to see if we can find a root to insert
      local::vector<LegionColor> path;
      IndexTreeNode* node = user_expr;
      // The common case for this should be read-only
      {
        AutoLock v_lock(view_lock, false /*exclusive*/);
        while (node != nullptr)
        {
          bool need_mutable = false;
          for (lng::FieldMaskMap<NodeView>::const_iterator it = roots.begin();
               it != roots.end(); it++)
          {
            // First check to see if the parent is a root
            if (it->first->tree_node == node)
            {
              if (!(user_mask - it->second))
              {
                // Insert going down
                it->first->insert_user(user, user_mask, path, v_lock);
                return;
              }
              else
              {
                need_mutable = true;
                break;
              }
            }
            // See if we have a shared parent along this path
            // If so we need to break out to go down the exclusive path
            if (it->first->tree_node->get_parent() == node)
            {
              need_mutable = true;
              break;
            }
          }
          if (need_mutable)
            break;
          path.push_back(node->color);
          node = node->get_parent();
        }
        // Failed so we are falling through to the mutable path
      }
      // Reset so we can try again
      path.clear();
      node = user_expr;
      // Now take the root lock and see if we need to insert the child
      // and update the roots
      AutoLock v_lock(view_lock);
      while (node != nullptr)
      {
        for (lng::FieldMaskMap<NodeView>::iterator it = roots.begin();
             it != roots.end(); it++)
        {
          // First check to see if the parent is a root
          if (it->first->tree_node == node)
          {
            it.merge(user_mask);
            // Insert going down
            it->first->insert_user(user, user_mask, path, v_lock);
            return;
          }
          // See if we have a shared parent along this path
          if (it->first->tree_node->get_parent() == node)
          {
            // Make the new root node
            NodeView* new_root = node->is_index_space_node() ?
                                     (NodeView*)new SpaceView(
                                         node->as_index_space_node(), this) :
                                     (NodeView*)new PartitionView(
                                         node->as_index_part_node(), this);
            new_root->insert_child(it->first, it->second);
            const FieldMask root_mask = user_mask | it->second;
            // No need to check for deletions, added another reference
            // with the insert_child call
            it->first->remove_reference();
            roots.erase(it);
            if (roots.insert(new_root, root_mask))
              new_root->add_reference();
            new_root->insert_user(user, user_mask, path, v_lock);
            return;
          }
        }
        path.push_back(node->color);
        node = node->get_parent();
      }
      // If we get here we couldn't find anything to merge with so the
      // node is its own root
      path.clear();
      SpaceView* new_root = new SpaceView(user_expr, this);
      if (roots.insert(new_root, user_mask))
        new_root->add_reference();
      new_root->insert_user(user, user_mask, path, v_lock);
    }

    //--------------------------------------------------------------------------
    void IndividualView::register_collective_analysis(
        const CollectiveView* source, CollectiveAnalysis* analysis)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner());
      const RendezvousKey key(
          analysis->get_context_index(), analysis->get_requirement_index(),
          analysis->get_match_space());
      RtUserEvent to_trigger;
      {
        AutoLock v_lock(view_lock);
        std::map<RendezvousKey, RegisteredAnalysis>::iterator finder =
            collective_analyses.find(key);
        if (finder != collective_analyses.end())
        {
          // Note that this will deduplicate multiple registrations
          if (finder->second.analysis == nullptr)
          {
            legion_assert(finder->second.ready.exists());
            analysis->add_analysis_reference();
            finder->second.analysis = analysis;
            to_trigger = finder->second.ready;
            finder->second.ready = RtUserEvent::NO_RT_USER_EVENT;
          }
          // Record the view so we know how many registrations to expect
          // We'll see one unregister for each source view
          finder->second.views.insert(source->did);
        }
        else
        {
          analysis->add_analysis_reference();
          RegisteredAnalysis& registration = collective_analyses[key];
          registration.analysis = analysis;
          registration.views.insert(source->did);
        }
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    CollectiveAnalysis* IndividualView::find_collective_analysis(
        size_t context_index, unsigned region_index, IndexSpace match_space)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner());
      RtEvent wait_on;
      const RendezvousKey key(context_index, region_index, match_space);
      {
        AutoLock v_lock(view_lock);
        std::map<RendezvousKey, RegisteredAnalysis>::iterator finder =
            collective_analyses.find(key);
        if (finder != collective_analyses.end())
        {
          if (finder->second.analysis == nullptr)
          {
            legion_assert(finder->second.ready.exists());
            wait_on = finder->second.ready;
          }
          else
            return finder->second.analysis;
        }
        else
        {
          RegisteredAnalysis& registration = collective_analyses[key];
          registration.analysis = nullptr;
          registration.ready = Runtime::create_rt_user_event();
          wait_on = registration.ready;
        }
      }
      if (!wait_on.has_triggered())
        wait_on.wait();
      AutoLock v_lock(view_lock);
      std::map<RendezvousKey, RegisteredAnalysis>::iterator finder =
          collective_analyses.find(key);
      legion_assert(finder != collective_analyses.end());
      legion_assert(finder->second.analysis != nullptr);
      return finder->second.analysis;
    }

    //--------------------------------------------------------------------------
    void IndividualView::unregister_collective_analysis(
        const CollectiveView* source, size_t context_index,
        unsigned region_index, IndexSpace match_space)
    //--------------------------------------------------------------------------
    {
      CollectiveAnalysis* removed = nullptr;
      const RendezvousKey key(context_index, region_index, match_space);
      {
        AutoLock v_lock(view_lock);
        std::map<RendezvousKey, RegisteredAnalysis>::iterator finder =
            collective_analyses.find(key);
        // Not all collective user registrations record an analysis with
        // the instance for performing copies. Specifically this is the
        // case with OverwriteAnalysis for attach operations, and
        // FilterAnalysis in the case where there is no flush
        if (finder == collective_analyses.end())
          return;
        legion_assert(finder->second.analysis != nullptr);
        legion_assert(!finder->second.ready.exists());
        std::set<DistributedID>::iterator view_finder =
            finder->second.views.find(source->did);
        legion_assert(view_finder != finder->second.views.end());
        finder->second.views.erase(view_finder);
        // If we're not the last view that needs to unregister this
        // analysis then we don't remove it yet
        if (!finder->second.views.empty())
          return;
        removed = finder->second.analysis;
        collective_analyses.erase(finder);
      }
      if (removed->remove_analysis_reference())
        delete removed;
    }

    //--------------------------------------------------------------------------
    ApEvent IndividualView::register_collective_user(
        const RegionUsage& usage, const FieldMask& user_mask,
        IndexSpaceNode* expr, const UniqueID op_id, const size_t op_ctx_index,
        const unsigned index, ApEvent term_event, PhysicalManager* target,
        CollectiveMapping* analysis_mapping, size_t local_collective_arrivals,
        std::vector<RtEvent>& registered_events,
        std::set<RtEvent>& applied_events, const PhysicalTraceInfo& trace_info,
        const bool symbolic)
    //--------------------------------------------------------------------------
    {
      // This case occurs when all the points mapping to the same logical
      // region also map to the same physical instance. Most commonly this
      // will occur with control replication doing attach operations on
      // file instances, but can occur outside of control replication as
      // well, especially in intra-node cases
      legion_assert(local_collective_arrivals > 0);
      legion_assert(
          (analysis_mapping != nullptr) || (local_collective_arrivals > 1));
      // First we need to decide which node is going to be the owner node
      // We'll prefer it to be the logical view owner since that is where
      // the event will be produced, otherwise, we'll just pick whichever
      // is closest to the logical view node
      const AddressSpaceID origin =
          (analysis_mapping == nullptr) ?
              local_space :
          analysis_mapping->contains(logical_owner) ?
              logical_owner :
              analysis_mapping->find_nearest(logical_owner);
      ApUserEvent result;
      RtUserEvent applied, registered;
      std::vector<ApEvent> term_events;
      PhysicalTraceInfo* result_info = nullptr;
      const RendezvousKey key(op_ctx_index, index, expr->handle);
      {
        AutoLock v_lock(view_lock);
        // Check to see if we're the first one to arrive on this node
        std::map<RendezvousKey, UserRendezvous>::iterator finder =
            rendezvous_users.find(key);
        if (finder == rendezvous_users.end())
        {
          // If we are then make the record for knowing when we've seen
          // all the expected arrivals
          finder =
              rendezvous_users.insert(std::make_pair(key, UserRendezvous()))
                  .first;
          UserRendezvous& rendezvous = finder->second;
          rendezvous.remaining_local_arrivals = local_collective_arrivals;
          rendezvous.local_initialized = true;
          rendezvous.remaining_remote_arrivals =
              (analysis_mapping == nullptr) ?
                  0 :
                  analysis_mapping->count_children(origin, local_space);
          rendezvous.ready_event = Runtime::create_ap_user_event(&trace_info);
          rendezvous.trace_info = new PhysicalTraceInfo(trace_info);
          if (analysis_mapping != nullptr)
          {
            rendezvous.analysis_mapping = analysis_mapping;
            analysis_mapping->add_reference();
          }
          rendezvous.expr = expr;
          expr->add_nested_expression_reference(did);
          rendezvous.registered = Runtime::create_rt_user_event();
          rendezvous.applied = Runtime::create_rt_user_event();
        }
        else if (!finder->second.local_initialized)
        {
          legion_assert(!finder->second.ready_event.exists());
          legion_assert(finder->second.trace_info == nullptr);
          legion_assert(finder->second.analysis_mapping != nullptr);
          // First local arrival
          finder->second.remaining_local_arrivals = local_collective_arrivals;
          finder->second.local_initialized = true;
          finder->second.ready_event =
              Runtime::create_ap_user_event(&trace_info);
          finder->second.trace_info = new PhysicalTraceInfo(trace_info);
          finder->second.expr = expr;
          expr->add_nested_expression_reference(did);
          if (!finder->second.remote_ready_events.empty())
          {
            for (const std::pair<const ApUserEvent, PhysicalTraceInfo*>& it :
                 finder->second.remote_ready_events)
            {
              Runtime::trigger_event(
                  it.first, finder->second.ready_event, *it.second,
                  applied_events);
              delete it.second;
            }
            finder->second.remote_ready_events.clear();
          }
        }
        result = finder->second.ready_event;
        result_info = finder->second.trace_info;
        analysis_mapping = finder->second.analysis_mapping;
        registered = finder->second.registered;
        registered_events.emplace_back(registered);
        applied = finder->second.applied;
        applied_events.insert(applied);
        if (term_event.exists())
          finder->second.term_events.emplace_back(term_event);
        legion_assert(finder->second.local_initialized);
        legion_assert(finder->second.remaining_local_arrivals > 0);
        // If we're still expecting arrivals then nothing to do yet
        if ((--finder->second.remaining_local_arrivals > 0) ||
            (finder->second.remaining_remote_arrivals > 0))
        {
          // We need to save the trace info no matter what
          if ((finder->second.mask == nullptr) && (local_space == origin))
          {
            // Save our state for performing the registration later
            finder->second.usage = usage;
            finder->second.mask =
                new HeapifyBox<FieldMask, OPERATION_LIFETIME>(user_mask);
            finder->second.op_id = op_id;
            finder->second.symbolic = symbolic;
          }
          return result;
        }
        term_events.swap(finder->second.term_events);
        expr = finder->second.expr;
        legion_assert(finder->second.remote_ready_events.empty());
        if (finder->second.mask != nullptr)
          delete finder->second.mask;
        // We're done with our entry after this so no need to keep it
        rendezvous_users.erase(finder);
      }
      if (!term_events.empty())
        term_event = Runtime::merge_events(&trace_info, term_events);
      if (local_space != origin)
      {
        legion_assert(analysis_mapping != nullptr);
        const AddressSpaceID parent =
            analysis_mapping->get_parent(origin, local_space);
        CollectiveIndividualRegisterUser rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(op_ctx_index);
          rez.serialize(index);
          rez.serialize(expr->handle);
          rez.serialize(origin);
          result_info->pack_trace_info(rez);
          analysis_mapping->pack(rez);
          rez.serialize(term_event);
          rez.serialize(result);
          rez.serialize(registered);
          rez.serialize(applied);
        }
        rez.dispatch(parent);
      }
      else
      {
        std::vector<RtEvent> local_registered;
        std::set<RtEvent> local_applied;
        const ApEvent ready = register_user(
            usage, user_mask, expr, op_id, op_ctx_index, index, term_event,
            target, nullptr /*analysis*/, 0 /*no collective arrivals*/,
            local_registered, local_applied, *result_info,
            runtime->address_space, symbolic);
        Runtime::trigger_event(result, ready, *result_info, local_applied);
        if (!local_registered.empty())
          Runtime::trigger_event(
              registered, Runtime::merge_events(local_registered));
        else
          Runtime::trigger_event(registered);
        if (!local_applied.empty())
          Runtime::trigger_event(applied, Runtime::merge_events(local_applied));
        else
          Runtime::trigger_event(applied);
      }
      if (expr->remove_nested_expression_reference(did))
        delete expr;
      if ((analysis_mapping != nullptr) && analysis_mapping->remove_reference())
        delete analysis_mapping;
      delete result_info;
      return result;
    }

    //--------------------------------------------------------------------------
    void IndividualView::process_collective_user_registration(
        const size_t op_ctx_index, const unsigned index,
        const IndexSpace match_space, const AddressSpaceID origin,
        const PhysicalTraceInfo& trace_info,
        CollectiveMapping* analysis_mapping, ApEvent remote_term_event,
        ApUserEvent remote_ready_event, RtUserEvent remote_registered,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      legion_assert(analysis_mapping != nullptr);
      UserRendezvous to_perform;
      const RendezvousKey key(op_ctx_index, index, match_space);
      {
        AutoLock v_lock(view_lock);
        // Check to see if we're the first one to arrive on this node
        std::map<RendezvousKey, UserRendezvous>::iterator finder =
            rendezvous_users.find(key);
        if (finder == rendezvous_users.end())
        {
          // If we are then make the record for knowing when we've seen
          // all the expected arrivals
          finder =
              rendezvous_users.insert(std::make_pair(key, UserRendezvous()))
                  .first;
          UserRendezvous& rendezvous = finder->second;
          rendezvous.local_initialized = false;
          rendezvous.analysis_mapping = analysis_mapping;
          analysis_mapping->add_reference();
          rendezvous.remaining_remote_arrivals =
              analysis_mapping->count_children(origin, local_space);
          // Don't make the ready event, that needs to be done with a
          // local trace_info
          rendezvous.registered = Runtime::create_rt_user_event();
          rendezvous.applied = Runtime::create_rt_user_event();
        }
        if (remote_term_event.exists())
          finder->second.term_events.emplace_back(remote_term_event);
        Runtime::trigger_event(remote_registered, finder->second.registered);
        if (finder->second.applied.exists())
          applied_events.insert(finder->second.applied);
        if (!finder->second.ready_event.exists())
          finder->second.remote_ready_events[remote_ready_event] =
              new PhysicalTraceInfo(trace_info);
        else
          Runtime::trigger_event(
              remote_ready_event, finder->second.ready_event, trace_info,
              applied_events);
        legion_assert(finder->second.remaining_remote_arrivals > 0);
        // Check to see if we've done all the arrivals
        if ((--finder->second.remaining_remote_arrivals > 0) ||
            !finder->second.local_initialized ||
            (finder->second.remaining_local_arrivals > 0))
          return;
        legion_assert(finder->second.remote_ready_events.empty());
        legion_assert(finder->second.trace_info != nullptr);
        // Last needed arrival, see if we're the origin or not
        to_perform = std::move(finder->second);
        rendezvous_users.erase(finder);
      }
      ApEvent term_event;
      if (!to_perform.term_events.empty())
        term_event = Runtime::merge_events(
            to_perform.trace_info, to_perform.term_events);
      if (local_space != origin)
      {
        legion_assert(to_perform.applied.exists());
        legion_assert(to_perform.analysis_mapping != nullptr);
        // Send the message to the parent
        const AddressSpaceID parent =
            to_perform.analysis_mapping->get_parent(origin, local_space);
        CollectiveIndividualRegisterUser rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(op_ctx_index);
          rez.serialize(index);
          rez.serialize(match_space);
          rez.serialize(origin);
          to_perform.trace_info->pack_trace_info(rez);
          to_perform.analysis_mapping->pack(rez);
          rez.serialize(term_event);
          rez.serialize(to_perform.ready_event);
          rez.serialize(to_perform.registered);
          rez.serialize(to_perform.applied);
        }
        rez.dispatch(parent);
        legion_assert(to_perform.mask == nullptr);
      }
      else
      {
        legion_assert(to_perform.applied.exists());
        std::vector<RtEvent> registered_events;
        std::set<RtEvent> local_applied;
        const ApEvent ready = register_user(
            to_perform.usage, *to_perform.mask, to_perform.expr,
            to_perform.op_id, op_ctx_index, index, term_event, manager,
            nullptr /*no analysis mapping*/, 0 /*no collective arrivals*/,
            registered_events, local_applied, *to_perform.trace_info,
            runtime->address_space, to_perform.symbolic);
        Runtime::trigger_event(
            to_perform.ready_event, ready, *to_perform.trace_info,
            local_applied);
        if (!registered_events.empty())
          Runtime::trigger_event(
              to_perform.registered, Runtime::merge_events(registered_events));
        else
          Runtime::trigger_event(to_perform.registered);
        if (!local_applied.empty())
          Runtime::trigger_event(
              to_perform.applied, Runtime::merge_events(local_applied));
        else
          Runtime::trigger_event(to_perform.applied);
        delete to_perform.mask;
      }
      if (to_perform.expr->remove_nested_expression_reference(did))
        delete to_perform.expr;
      if ((to_perform.analysis_mapping != nullptr) &&
          to_perform.analysis_mapping->remove_reference())
        delete to_perform.analysis_mapping;
      delete to_perform.trace_info;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveIndividualRegisterUser::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      IndividualView* view = static_cast<IndividualView*>(
          runtime->find_or_request_logical_view(did, ready));
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      unsigned index;
      derez.deserialize(index);
      IndexSpace match_space;
      derez.deserialize(match_space);
      AddressSpaceID origin;
      derez.deserialize(origin);
      std::set<RtEvent> applied_events;
      PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);
      size_t num_spaces;
      derez.deserialize(num_spaces);
      legion_assert(num_spaces > 0);
      CollectiveMapping* mapping = new CollectiveMapping(derez, num_spaces);
      mapping->add_reference();
      ApEvent term_event;
      derez.deserialize(term_event);
      ApUserEvent ready_event;
      derez.deserialize(ready_event);
      RtUserEvent registered_event, applied_event;
      derez.deserialize(registered_event);
      derez.deserialize(applied_event);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();

      view->process_collective_user_registration(
          op_ctx_index, index, match_space, origin, trace_info, mapping,
          term_event, ready_event, registered_event, applied_events);
      if (!applied_events.empty())
        Runtime::trigger_event(
            applied_event, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied_event);
      if (mapping->remove_reference())
        delete mapping;
    }

    //--------------------------------------------------------------------------
    void IndividualView::pack_fields(
        Serializer& rez, const std::vector<CopySrcDstField>& fields) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(manager->get_unique_event());
      rez.serialize<size_t>(fields.size());
      for (unsigned idx = 0; idx < fields.size(); idx++)
        rez.serialize(fields[idx]);
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        rez.serialize<size_t>(0);  // not part of the collective
        rez.serialize(did);
      }
    }

    //--------------------------------------------------------------------------
    void IndividualView::find_atomic_reservations(
        const FieldMask& mask, Operation* op, unsigned index, bool excl)
    //--------------------------------------------------------------------------
    {
      std::vector<Reservation> reservations;
      find_field_reservations(mask, reservations);
      for (unsigned idx = 0; idx < reservations.size(); idx++)
        op->update_atomic_locks(index, reservations[idx], excl);
    }

    //--------------------------------------------------------------------------
    void IndividualView::find_field_reservations(
        const FieldMask& mask, std::vector<Reservation>& reservations)
    //--------------------------------------------------------------------------
    {
      const RtEvent ready =
          find_field_reservations(mask, &reservations, runtime->address_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      // Sort them into order if necessary
      if (reservations.size() > 1)
        std::sort(reservations.begin(), reservations.end());
    }

    //--------------------------------------------------------------------------
    RtEvent IndividualView::find_field_reservations(
        const FieldMask& mask, std::vector<Reservation>* reservations,
        AddressSpaceID source, RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      std::vector<Reservation> results;
      if (is_logical_owner())
      {
        results.reserve(mask.pop_count());
        // We're the owner so we can make all the fields
        AutoLock v_lock(view_lock);
        for (int idx = mask.find_first_set(); idx >= 0;
             idx = mask.find_next_set(idx + 1))
        {
          std::map<unsigned, Reservation>::const_iterator finder =
              view_reservations.find(idx);
          if (finder == view_reservations.end())
          {
            // Make a new reservation and add it to the set
            Reservation handle = Reservation::create_reservation();
            view_reservations[idx] = handle;
            results.emplace_back(handle);
          }
          else
            results.emplace_back(finder->second);
        }
      }
      else
      {
        // See if we can find them all locally
        {
          AutoLock v_lock(view_lock, false /*exclusive*/);
          for (int idx = mask.find_first_set(); idx >= 0;
               idx = mask.find_next_set(idx + 1))
          {
            std::map<unsigned, Reservation>::const_iterator finder =
                view_reservations.find(idx);
            if (finder != view_reservations.end())
              results.emplace_back(finder->second);
            else
              break;
          }
        }
        if (results.size() < mask.pop_count())
        {
          // Couldn't find them all so send the request to the owner
          if (!to_trigger.exists())
            to_trigger = Runtime::create_rt_user_event();
          AtomicReservationRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(mask);
            rez.serialize(reservations);
            rez.serialize(source);
            rez.serialize(to_trigger);
          }
          rez.dispatch(logical_owner);
          return to_trigger;
        }
      }
      if (source != local_space)
      {
        legion_assert(to_trigger.exists());
        // Send the result back to the source
        AtomicReservationResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(mask);
          rez.serialize(reservations);
          rez.serialize<size_t>(results.size());
          for (const Reservation& reservation : results)
            rez.serialize(reservation);
          rez.serialize(to_trigger);
        }
        rez.dispatch(source);
      }
      else
      {
        reservations->swap(results);
        if (to_trigger.exists())
          Runtime::trigger_event(to_trigger);
      }
      return to_trigger;
    }

    //--------------------------------------------------------------------------
    void IndividualView::update_field_reservations(
        const FieldMask& mask, const std::vector<Reservation>& reservations)
    //--------------------------------------------------------------------------
    {
      legion_assert(!is_logical_owner());
      AutoLock v_lock(view_lock);
      unsigned offset = 0;
      for (int idx = mask.find_first_set(); idx >= 0;
           idx = mask.find_next_set(idx + 1))
        view_reservations[idx] = reservations[offset++];
    }

    //--------------------------------------------------------------------------
    /*static*/ void AtomicReservationRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      IndividualView* view = static_cast<IndividualView*>(
          runtime->find_or_request_logical_view(did, ready));
      FieldMask mask;
      derez.deserialize(mask);
      std::vector<Reservation>* target;
      derez.deserialize(target);
      AddressSpaceID source;
      derez.deserialize(source);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      view->find_field_reservations(mask, target, source, to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void AtomicReservationResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      IndividualView* view = static_cast<IndividualView*>(
          runtime->find_or_request_logical_view(did, ready));
      FieldMask mask;
      derez.deserialize(mask);
      std::vector<Reservation>* target;
      derez.deserialize(target);
      size_t num_reservations;
      derez.deserialize(num_reservations);
      target->resize(num_reservations);
      for (unsigned idx = 0; idx < num_reservations; idx++)
        derez.deserialize((*target)[idx]);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      view->update_field_reservations(mask, *target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ViewFindCopyPreMessage::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready = RtEvent::NO_RT_EVENT;
      LogicalView* view = runtime->find_or_request_logical_view(did, ready);

      bool reading;
      derez.deserialize<bool>(reading);
      ReductionOpID redop;
      derez.deserialize(redop);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      IndexSpaceExpression* copy_expr =
          IndexSpaceExpression::unpack_expression(derez, source);
      UniqueID op_id;
      derez.deserialize(op_id);
      unsigned index;
      derez.deserialize(index);
      ApUserEvent to_trigger;
      derez.deserialize(to_trigger);
      RtUserEvent applied;
      derez.deserialize(applied);
      std::set<RtEvent> applied_events;
      const PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);

      // This blocks the virtual channel, but keeps queries in-order
      // with respect to updates from the same node which is necessary
      // for preventing cycles in the realm event graph
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      IndividualView* inst_view = view->as_individual_view();
      const ApEvent pre = inst_view->find_copy_preconditions(
          reading, redop, copy_mask, copy_expr, op_id, index, applied_events,
          trace_info);
      Runtime::trigger_event(to_trigger, pre, trace_info, applied_events);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ViewAddCopyUserMessage::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready = RtEvent::NO_RT_EVENT;
      LogicalView* view = runtime->find_or_request_logical_view(did, ready);

      bool reading;
      derez.deserialize(reading);
      ReductionOpID redop;
      derez.deserialize(redop);
      ApEvent term_event;
      derez.deserialize(term_event);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      IndexSpaceExpression* copy_expr =
          IndexSpaceExpression::unpack_expression(derez, source);
      IndexSpace upper_bound;
      derez.deserialize(upper_bound);
      UniqueID op_id;
      derez.deserialize(op_id);
      unsigned index;
      derez.deserialize(index);
      RtUserEvent applied_event;
      derez.deserialize(applied_event);
      bool trace_recording;
      derez.deserialize(trace_recording);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      legion_assert(view->is_individual_view());
      IndividualView* inst_view = view->as_individual_view();

      std::set<RtEvent> applied_events;
      inst_view->add_copy_user(
          reading, redop, term_event, copy_mask, copy_expr, upper_bound, op_id,
          index, applied_events, trace_recording, source);
      if (!applied_events.empty())
        Runtime::trigger_event(
            applied_event, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ViewFindLastUsersRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      LogicalView* view = runtime->find_or_request_logical_view(did, ready);
      DistributedID manager_did;
      derez.deserialize(manager_did);
      RtEvent manager_ready;
      PhysicalManager* manager =
          runtime->find_or_request_instance_manager(manager_did, manager_ready);

      std::vector<ApEvent>* target;
      derez.deserialize(target);
      RegionUsage usage;
      derez.deserialize(usage);
      FieldMask mask;
      derez.deserialize(mask);
      IndexSpaceExpression* expr =
          IndexSpaceExpression::unpack_expression(derez, source);
      RtUserEvent done;
      derez.deserialize(done);

      std::set<ApEvent> result;
      std::vector<RtEvent> applied;
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      if (manager_ready.exists() && !manager_ready.has_triggered())
        manager_ready.wait();
      legion_assert(view->is_individual_view());
      IndividualView* inst_view = view->as_individual_view();
      inst_view->find_last_users(manager, result, usage, mask, expr, applied);
      if (!result.empty())
      {
        ViewFindLastUsersResponse rez;
        {
          RezCheck z2(rez);
          rez.serialize(target);
          rez.serialize<size_t>(result.size());
          for (const ApEvent& ap_event : result) rez.serialize(ap_event);
          rez.serialize(done);
          if (!applied.empty())
            rez.serialize(Runtime::merge_events(applied));
          else
            rez.serialize(RtEvent::NO_RT_EVENT);
        }
        rez.dispatch(source);
      }
      else
      {
        if (!applied.empty())
          Runtime::trigger_event(done, Runtime::merge_events(applied));
        else
          Runtime::trigger_event(done);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ViewFindLastUsersResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::set<ApEvent>* target;
      derez.deserialize(target);
      size_t num_events;
      derez.deserialize(num_events);
      for (unsigned idx = 0; idx < num_events; idx++)
      {
        ApEvent event;
        derez.deserialize(event);
        target->insert(event);
      }
      RtUserEvent done;
      derez.deserialize(done);
      RtEvent pre;
      derez.deserialize(pre);
      Runtime::trigger_event(done, pre);
    }

  }  // namespace Internal
}  // namespace Legion
