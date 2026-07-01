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

#include "legion/nodes/region.h"
#include "legion/analysis/equivalence_set.h"
#include "legion/analysis/projection.h"
#include "legion/contexts/inner.h"
#include "legion/operations/refinement.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Region Tree Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionTreeNode::RegionTreeNode(
        FieldSpaceNode* column_src, RtEvent init, RtEvent tree,
        Provenance* prov, DistributedID id, CollectiveMapping* map)
      : DistributedCollectable(
            LEGION_DISTRIBUTED_HELP_ENCODE(
                (id > 0) ? id : runtime->get_available_distributed_id(),
                REGION_TREE_NODE_DC),
            false /*register with runtime*/, map),
        column_source(column_src), provenance(prov), initialized(init),
        tree_initialized(tree), registered(false)
    //--------------------------------------------------------------------------
    {
      if (provenance != nullptr)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    RegionTreeNode::~RegionTreeNode(void)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID RegionTreeNode::get_owner_space(RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      return (tid % runtime->total_address_spaces);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::attach_semantic_information(
        SemanticTag tag, AddressSpaceID source, const void* buffer, size_t size,
        bool is_mutable, bool local_only)
    //--------------------------------------------------------------------------
    {
      // Make a copy
      bool added = true;
      {
        AutoLock n_lock(node_lock);
        // See if it already exists
        lng::map<SemanticTag, SemanticInfo>::iterator finder =
            semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          if (finder->second.is_valid())
          {
            if (!finder->second.is_mutable)
            {
              // Check to make sure that the bits are the same
              if (size != finder->second.buffer.get_size())
              {
                Error error(LEGION_INTERFACE_EXCEPTION);
                error << "Inconsistent Semantic Tag value "
                      << "for tag " << tag << " with different sizes of "
                      << size << " and " << finder->second.buffer.get_size()
                      << " for region tree node";
                error.raise();
              }
              // Otherwise do a bitwise comparison
              {
                const char* orig =
                    (const char*)finder->second.buffer.get_buffer();
                const char* next = (const char*)buffer;
                for (unsigned idx = 0; idx < size; idx++)
                {
                  char diff = orig[idx] ^ next[idx];
                  if (diff)
                  {
                    Error error(LEGION_INTERFACE_EXCEPTION);
                    error << "Inconsistent Semantic Tag value "
                          << "for tag " << tag << " with different values at "
                          << "byte " << idx << " for region tree node, "
                          << std::hex << (int)orig[idx]
                          << " != " << (int)next[idx] << std::dec;
                    error.raise();
                  }
                }
              }
              added = false;
            }
            else
            {
              // Mutable so we can just overwrite it
              finder->second.buffer.save_buffer(buffer, size);
              finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
              finder->second.is_mutable = is_mutable;
            }
          }
          else
          {
            finder->second.buffer.save_buffer(buffer, size);
            // Trigger will happen by caller
            finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
            finder->second.is_mutable = is_mutable;
          }
        }
        else
          semantic_info[tag] = SemanticInfo(buffer, size, is_mutable);
      }
      if (added)
      {
        AddressSpaceID owner_space = find_semantic_owner();
        // If we are not the owner and the message
        // didn't come from the owner, then send it
        if ((owner_space != runtime->address_space) &&
            (source != owner_space) && !local_only)
        {
          const RtUserEvent done = Runtime::create_rt_user_event();
          send_semantic_info(owner_space, tag, buffer, size, is_mutable, done);
          if (!done.has_triggered())
            done.wait();
        }
      }
    }

    //--------------------------------------------------------------------------
    bool RegionTreeNode::retrieve_semantic_information(
        SemanticTag tag, const void*& result, size_t& size, bool can_fail,
        bool wait_until)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on;
      RtUserEvent request;
      const AddressSpaceID owner_space = find_semantic_owner();
      const bool is_remote = (owner_space != runtime->address_space);
      {
        AutoLock n_lock(node_lock);
        lng::map<SemanticTag, SemanticInfo>::const_iterator finder =
            semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          // Already have the data so we are done
          if (finder->second.is_valid())
          {
            result = finder->second.buffer.get_buffer();
            size = finder->second.buffer.get_size();
            return true;
          }
          else if (is_remote)
          {
            if (can_fail)
            {
              // Have to make our own event
              request = Runtime::create_rt_user_event();
              wait_on = request;
            }
            else  // can use the canonical event
              wait_on = finder->second.ready_event;
          }
          else if (wait_until)  // local so use the canonical event
            wait_on = finder->second.ready_event;
        }
        else
        {
          // Otherwise we make an event to wait on
          if (!can_fail && wait_until)
          {
            // Make a canonical ready event
            request = Runtime::create_rt_user_event();
            semantic_info[tag] = SemanticInfo(request);
            wait_on = request;
          }
          else if (is_remote)
          {
            // Make an event just for us to use
            request = Runtime::create_rt_user_event();
            wait_on = request;
          }
        }
      }
      // We didn't find it yet, see if we have something to wait on
      if (!wait_on.exists())
      {
        if (can_fail)
          return false;
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "invalid semantic tag " << tag << " for "
                << "region tree node";
          error.raise();
        }
      }
      else
      {
        if (is_remote && request.exists())
          send_semantic_request(
              owner_space, tag, can_fail, wait_until, request);
        wait_on.wait();
      }
      // When we wake up, we should be able to find everything
      AutoLock n_lock(node_lock, false /*exclusive*/);
      lng::map<SemanticTag, SemanticInfo>::const_iterator finder =
          semantic_info.find(tag);
      if (finder == semantic_info.end())
      {
        if (can_fail)
          return false;
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "invalid semantic tag " << tag << " for "
                << "region tree node";
          error.raise();
        }
      }
      result = finder->second.buffer.get_buffer();
      size = finder->second.buffer.get_size();
      return true;
    }

    //--------------------------------------------------------------------------
    ProjectionSummary* RegionTreeNode::compute_projection_summary(
        Operation* op, unsigned index, const RegionRequirement& req,
        LogicalAnalysis& analysis, const ProjectionInfo& proj_info)
    //--------------------------------------------------------------------------
    {
      const ContextID ctx = analysis.context->get_logical_tree_context();
      LogicalState& state = get_logical_state(ctx);
      return state.find_or_create_projection_summary(
          op, index, req, analysis, proj_info);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::register_logical_user(
        LogicalRegion privilege_root, LogicalUser& user,
        const RegionTreePath& path, const LogicalTraceInfo& trace_info,
        const ProjectionInfo& proj_info, const FieldMask& user_mask,
        FieldMask& unopened_field_mask,
        const FieldMask& current_refinement_mask,
        LogicalAnalysis& logical_analysis,
        FieldMaskMap<
            RefinementOp, TASK_LOCAL_LIFETIME,
            LogicalAnalysis::RefinementComparator>& refinements)
    //--------------------------------------------------------------------------
    {
      const ContextID ctx =
          logical_analysis.context->get_logical_tree_context();
      LogicalState& state = get_logical_state(ctx);
#ifdef LEGION_DEBUG
      state.sanity_check();
#endif
      const unsigned depth = get_depth();
      const bool arrived = !path.has_child(depth);
      FieldMask open_below;
      RegionTreeNode* next_child = nullptr;
      if (!arrived)
        next_child = get_tree_child(path.get_child(depth));
      // Check to see if we need to traverse any interfering children
      // and record dependences on prior operations in that tree
      if (!!unopened_field_mask)
        siphon_interfering_children(
            state, logical_analysis, user_mask, user, privilege_root,
            next_child, open_below);
      else if (!arrived)
        // Everything is open-only so make a state and merge it in
        add_open_field_state(state, user, user_mask, next_child);
      // Perform our local dependence analysis at this node along the path
      FieldMask dominator_mask = perform_dependence_checks<true /*track dom*/>(
          privilege_root, user, state.curr_epoch_users, user_mask, open_below,
          arrived, proj_info, state, logical_analysis);
      FieldMask non_dominated_mask = user_mask - dominator_mask;
      // For the fields that weren't dominated, we have to check
      // those fields against the previous epoch's users
      if (!!non_dominated_mask)
        perform_dependence_checks<false /*track dom*/>(
            privilege_root, user, state.prev_epoch_users, non_dominated_mask,
            open_below, arrived, proj_info, state, logical_analysis);
      if (arrived)
      {
        // If we dominated and this is our final destination then we
        // can filter the operations since we actually do dominate them
        if (!!dominator_mask)
        {
          // Dominator mask is not empty
          // Mask off all the dominated fields from the previous set
          // of epoch users and remove any previous epoch users
          // that were totally dominated
          state.filter_previous_epoch_users(dominator_mask);
          // Mask off all dominated fields from current epoch users and move
          // them to prev epoch users.  If all fields masked off, then remove
          // them from the list of current epoch users.
          state.filter_current_epoch_users(dominator_mask);
        }
        // If we've arrived add ourselves as a user
        state.register_local_user(user, user_mask);
        // Update our refinement state if we're on the current refinement
        if (!!current_refinement_mask)
        {
          FieldMask new_refinements;
          if (proj_info.is_projecting())
          {
            // Update the refinement information here for deciding if we need
            // to change refinements later
            if (user.shard_proj == nullptr)
            {
              ProjectionSummary* summary =
                  state.find_or_create_projection_summary(
                      user.op, user.idx, trace_info.req, logical_analysis,
                      proj_info);
              state.update_refinement_projection(
                  ctx, summary, user.usage, current_refinement_mask,
                  new_refinements);
            }
            else
              state.update_refinement_projection(
                  ctx, user.shard_proj, user.usage, current_refinement_mask,
                  new_refinements);
          }
          else
            state.update_refinement_node(
                ctx, user.usage, current_refinement_mask, new_refinements);
          if (!!new_refinements)
            logical_analysis.record_pending_refinement(
                privilege_root, user.idx, user.op->find_parent_index(user.idx),
                this, new_refinements, refinements);
        }
      }
      else
      {
        // We haven't arrived so we need to traverse to the next child
        // Get our set of fields which are being opened for
        // the first time at the next level
        if (!!unopened_field_mask)
        {
          if (!!open_below)
            // Update our unopened children mask
            // to reflect any fields which are still open below
            unopened_field_mask &= open_below;
          else
            // Open all the unopened fields
            unopened_field_mask.clear();
        }
        else  // if they weren't open here, they shouldn't be open below
        {
          legion_assert(!open_below);
        }
        // Figure out which fields the child is part of the current refinement
        // When this mask goes down to the child it tells which fields are
        // in the current refinement for that node, when it comes back up
        // it represents which fields should be updated in the state above
        // regardless of whether they are in the current refinement or not
        FieldMask child_refinement_mask;
        if (!!current_refinement_mask && next_child->track_refinements())
        {
          FieldMask new_refinements;
          state.update_refinement_child(
              ctx, next_child, user.usage, current_refinement_mask,
              child_refinement_mask, new_refinements);
          if (!!new_refinements)
            logical_analysis.record_pending_refinement(
                privilege_root, user.idx, user.op->find_parent_index(user.idx),
                this, new_refinements, refinements);
        }
        next_child->register_logical_user(
            privilege_root, user, path, trace_info, proj_info, user_mask,
            unopened_field_mask, child_refinement_mask, logical_analysis,
            refinements);
      }
      // If we have any refinement operations then we need to perform their
      // dependence analysis now on the way back up the tree after having
      // done everything else
      if (!refinements.empty())
      {
        const ProjectionInfo no_projection_info(
            nullptr, nullptr, nullptr, IndexSpace::NO_SPACE);
        const RegionUsage ref_usage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0);
        for (FieldMaskMap<
                 RefinementOp, TASK_LOCAL_LIFETIME,
                 LogicalAnalysis::RefinementComparator>::const_iterator it =
                 refinements.begin();
             it != refinements.end(); it++)
        {
          const LogicalUser refinement_user(
              it->first, it->first->get_internal_index(), ref_usage);
          // Recording refinement dependences will record dependences on
          // anything in an interfering sub-tree without changing the
          // state of the region tree states
          state.record_refinement_dependences(
              ctx, refinement_user, it->second, no_projection_info,
              // Can only pass in the next_child to skip if the refinement
              // came from further down the tree otherwise we need to perform
              // dependence analysis on any open children
              (it->first->get_refinement_node() == this) ? nullptr : next_child,
              privilege_root, logical_analysis);
        }
        // A bit of a hairy case: if the user is not read-write and we have
        // refinements below then we need to promote the state of the child
        // sub-tree up to exclusive so that later operations will know that
        // they need to traverse the sub-tree and find the dependence on the
        // refinement operation
        if ((next_child != nullptr) && !IS_WRITE(user.usage))
          state.promote_next_child(next_child, refinements.get_valid_mask());
      }
      // Perform any filtering that we need to do for timeout users
      state.filter_timeout_users(logical_analysis);
#ifdef LEGION_DEBUG
      state.sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::record_refinement_dependences(
        ContextID ctx, const LogicalUser& refinement_user,
        const FieldMask& refinement_mask, const ProjectionInfo& no_proj_info,
        RegionTreeNode* previous_child, LogicalRegion privilege_root,
        LogicalAnalysis& logical_analysis)
    //--------------------------------------------------------------------------
    {
      LogicalState& state = get_logical_state(ctx);
#ifdef LEGION_DEBUG
      state.sanity_check();
#endif
      state.record_refinement_dependences(
          ctx, refinement_user, refinement_mask, no_proj_info, previous_child,
          privilege_root, logical_analysis);
      state.filter_timeout_users(logical_analysis);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::invalidate_logical_refinement(
        ContextID ctx, const FieldMask& invalidation_mask)
    //--------------------------------------------------------------------------
    {
      LogicalState& state = get_logical_state(ctx);
      state.invalidate_refinements(ctx, invalidation_mask);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::add_open_field_state(
        LogicalState& state, const LogicalUser& user,
        const FieldMask& open_mask, RegionTreeNode* next_child)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      state.sanity_check();
#endif
      FieldState new_state(user.usage, open_mask, next_child);
      merge_new_field_state(state, new_state);
#ifdef LEGION_DEBUG
      state.sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::siphon_interfering_children(
        LogicalState& state, LogicalAnalysis& analysis,
        const FieldMask& closing_mask, const LogicalUser& user,
        LogicalRegion privilege_root, RegionTreeNode* next_child,
        FieldMask& open_below)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      state.sanity_check();
#endif
      // These are fields for which the next_child is already open but it was
      // in the wrong state so we still need to add a new state for them
      FieldMask next_child_fields;
      // Now we can look at all the children
      for (shrt::list<FieldState>::iterator it = state.field_states.begin();
           it != state.field_states.end();
           /*nothing*/)
      {
        // Quick check for disjointness, in which case we can continue
        if (it->valid_fields() * closing_mask)
        {
          it++;
          continue;
        }
        // Now check the current state
        switch (it->open_state)
        {
          case OPEN_READ_ONLY:
            {
              if (IS_READ_ONLY(user.usage))
              {
                // We're read-only too so there is no need to traverse
                // See if the child that we want is already open
                if (next_child != nullptr)
                {
                  OrderedFieldMaskChildren::const_iterator finder =
                      it->open_children.find(next_child);
                  if (finder != it->open_children.end())
                  {
                    // Remove the child's open fields from the
                    // list of fields we need to open
                    open_below |= (finder->second & closing_mask);
                  }
                }
                it++;
              }
              else
              {
                // Not-read only so traverse the interfering children and
                // close up anything that is not the next child
                perform_close_operations(
                    user, closing_mask, it->open_children, privilege_root, this,
                    analysis, open_below, next_child, &next_child_fields,
                    true /*filter next*/);
                // See if there are still any valid open fields
                if (!it->valid_fields())
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              break;
            }
          case OPEN_READ_WRITE:
            {
              // Close up any interfering children that conflict
              perform_close_operations(
                  user, closing_mask, it->open_children, privilege_root, this,
                  analysis, open_below, next_child);
              if (!it->valid_fields())
                it = state.field_states.erase(it);
              else
                it++;
              break;
            }
          case OPEN_REDUCE:
            {
              // See if this reduction is a reduction of the same kind
              if (IS_REDUCE(user.usage) && (user.usage.redop == it->redop))
              {
                if (next_child != nullptr)
                {
                  OrderedFieldMaskChildren::const_iterator finder =
                      it->open_children.find(next_child);
                  if (finder != it->open_children.end())
                  {
                    // Already open, so add the open fields
                    open_below |= (finder->second & closing_mask);
                  }
                }
                it++;
              }
              else
              {
                // Need to close up the open field since we're going
                // to have to do it anyway
                perform_close_operations(
                    user, closing_mask, it->open_children, privilege_root, this,
                    analysis, open_below, next_child, &next_child_fields,
                    true /*filter next*/);
                if (!it->valid_fields())
                  it = state.field_states.erase(it);
                else
                  it++;
              }
              break;
            }
          default:
            std::abort();
        }
      }
      // If we had any fields that still need to be opened, create
      // a new field state and add it into the set of new states
      if (next_child != nullptr)
      {
        FieldMask open_mask = closing_mask;
        if (!!open_below)
          open_mask -= open_below;
        if (!!next_child_fields)
          open_mask |= next_child_fields;
        if (!!open_mask)
        {
          FieldState new_state(user.usage, open_mask, next_child);
          merge_new_field_state(state, new_state);
        }
      }
#ifdef LEGION_DEBUG
      state.sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::perform_close_operations(
        const LogicalUser& user, const FieldMask& closing_mask,
        OrderedFieldMaskChildren& children, LogicalRegion privilege_root,
        RegionTreeNode* path_node, LogicalAnalysis& analysis,
        FieldMask& open_below, RegionTreeNode* next_child,
        FieldMask* next_child_fields, const bool filter_next_child)
    //--------------------------------------------------------------------------
    {
      if (next_child != nullptr)
      {
        if (!are_all_children_disjoint())
        {
          // The children are not disjoint so we need to traverse them all
          // see if they are disjoint with the next child
          const LegionColor next_color = next_child->get_color();
          std::vector<RegionTreeNode*> to_delete;
          for (OrderedFieldMaskChildren::iterator it = children.begin();
               it != children.end(); it++)
          {
            if (next_child == it->first)  // we'll handle the next child below
              continue;
            const FieldMask close_mask = closing_mask & it->second;
            if (!close_mask)
              continue;
            if (are_children_disjoint(it->first->get_color(), next_color))
              continue;
            FieldMask still_open;
            it->first->close_logical_node(
                user, close_mask, privilege_root, path_node, analysis,
                still_open);
            if (!!still_open)
            {
              if (still_open != close_mask)
              {
                it.filter(close_mask - still_open);
                if (!it->second)
                  to_delete.emplace_back(it->first);
              }
            }
            else
            {
              it.filter(close_mask);
              if (!it->second)
                to_delete.emplace_back(it->first);
            }
          }
          if (!to_delete.empty())
          {
            for (RegionTreeNode* child : to_delete)
            {
              children.erase(child);
              if (child->remove_base_gc_ref(FIELD_STATE_REF))
                delete child;
            }
            children.tighten_valid_mask();
          }
        }
        // Now handle the next child
        OrderedFieldMaskChildren::iterator finder = children.find(next_child);
        if (finder != children.end())
        {
          FieldMask overlap = closing_mask & finder->second;
          if (!!overlap)
          {
            if (filter_next_child)
            {
              legion_assert(next_child_fields != nullptr);
              FieldMask child_fields;
              next_child->close_logical_node(
                  user, overlap, privilege_root, path_node, analysis,
                  child_fields);
              if (!!child_fields)
              {
                open_below |= child_fields;
                (*next_child_fields) |= child_fields;
              }
              finder.filter(overlap);
              if (!finder->second)
              {
                children.erase(finder);
                if (next_child->remove_base_gc_ref(FIELD_STATE_REF))
                  std::abort();  // should never delete the next child
              }
              children.tighten_valid_mask();
            }
            else
              open_below |= overlap;
          }
        }
      }
      else
      {
        // We don't have a next child we're doing to, so we just need to
        // close up all the open children
        std::vector<RegionTreeNode*> to_delete;
        for (OrderedFieldMaskChildren::iterator it = children.begin();
             it != children.end(); it++)
        {
          const FieldMask close_mask = closing_mask & it->second;
          if (!close_mask)
            continue;
          FieldMask still_open;
          it->first->close_logical_node(
              user, close_mask, privilege_root, path_node, analysis,
              still_open);
          if (!!still_open)
          {
            open_below |= still_open;
            if (still_open != close_mask)
            {
              it.filter(close_mask - still_open);
              if (!it->second)
                to_delete.emplace_back(it->first);
            }
          }
          else
          {
            it.filter(close_mask);
            if (!it->second)
              to_delete.emplace_back(it->first);
          }
        }
        if (!to_delete.empty())
        {
          for (RegionTreeNode* child : to_delete)
          {
            children.erase(child);
            if (child->remove_base_gc_ref(FIELD_STATE_REF))
              delete child;
          }
          children.tighten_valid_mask();
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::close_logical_node(
        const LogicalUser& user, const FieldMask& closing_mask,
        LogicalRegion privilege_root, RegionTreeNode* path_node,
        LogicalAnalysis& logical_analysis, FieldMask& still_open)
    //--------------------------------------------------------------------------
    {
      LogicalState& state = get_logical_state(
          logical_analysis.context->get_logical_tree_context());
      // Perform closing checks on both the current epoch users
      // as well as the previous epoch users
      perform_closing_checks(
          logical_analysis, state.curr_epoch_users, user, closing_mask,
          privilege_root, path_node, still_open);
      perform_closing_checks(
          logical_analysis, state.prev_epoch_users, user, closing_mask,
          privilege_root, path_node, still_open);
      if (!state.field_states.empty())
      {
        // Recursively traverse any open children and close them as well
        for (std::list<FieldState>::iterator it = state.field_states.begin();
             it != state.field_states.end();
             /*nothing*/)
        {
          FieldMask overlap = it->valid_fields() & closing_mask;
          if (!overlap)
          {
            it++;
            continue;
          }
          perform_close_operations(
              user, overlap, it->open_children, privilege_root, path_node,
              logical_analysis, still_open);
          // Remove the state if it is now empty
          if (!it->valid_fields())
            it = state.field_states.erase(it);
          else
            it++;
        }
      }
#ifdef LEGION_DEBUG
      state.sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::merge_new_field_state(
        LogicalState& state, FieldState& new_state)
    //--------------------------------------------------------------------------
    {
      legion_assert(!!new_state.valid_fields());
      for (FieldState& field_state : state.field_states)
      {
        if (field_state.overlaps(new_state))
        {
          field_state.merge(new_state, this);
          return;
        }
      }
      // Otherwise just push it on the back
      state.field_states.emplace_back(std::move(new_state));
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::report_uninitialized_usage(
        Operation* op, unsigned idx, const FieldMask& uninit,
        RtUserEvent reported)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_region());
      legion_assert(reported.exists());
      char* field_string = column_source->to_string(uninit, op->get_context());
      op->report_uninitialized_usage(idx, field_string, reported);
      free(field_string);
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::initialize_current_state(ContextID ctx)
    //--------------------------------------------------------------------------
    {
      if (!logical_states.has_entry(ctx))
        return;
      LogicalState& state = get_logical_state(ctx);
      state.check_init();
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::invalidate_current_state(ContextID ctx)
    //--------------------------------------------------------------------------
    {
      if (!logical_states.has_entry(ctx))
        return;
      LogicalState& state = get_logical_state(ctx);
      state.clear();
    }

    //--------------------------------------------------------------------------
    void RegionTreeNode::invalidate_deleted_state(
        ContextID ctx, const FieldMask& deleted_mask)
    //--------------------------------------------------------------------------
    {
      if (!logical_states.has_entry(ctx))
        return;
      LogicalState& state = get_logical_state(ctx);
      state.clear_deleted_state(ctx, deleted_mask);
    }

    //--------------------------------------------------------------------------
    template<bool TRACK_DOM>
    FieldMask RegionTreeNode::perform_dependence_checks(
        LogicalRegion root, const LogicalUser& user,
        OrderedFieldMaskUsers& prev_users, const FieldMask& check_mask,
        const FieldMask& open_below, const bool arrived,
        const ProjectionInfo& proj_info, LogicalState& state,
        LogicalAnalysis& logical_analysis)
    //--------------------------------------------------------------------------
    {
      FieldMask dominator_mask = check_mask;
      // It's not actually sound to assume we dominate something
      // if we don't observe any users of those fields.  Therefore
      // also keep track of the fields that we observe.  We'll use this
      // at the end when computing the final dominator mask.
      FieldMask observed_mask;
      if (!(check_mask * prev_users.get_valid_mask()))
      {
        bool tighten = false;
        std::vector<LogicalUser*> to_delete;
        for (OrderedFieldMaskUsers::iterator it = prev_users.begin();
             it != prev_users.end(); it++)
        {
          // Don't record dependences on any other users from the same op
          LogicalUser& prev = *(it->first);
          if ((prev.ctx_index == user.ctx_index) &&
              // Note this second condition only happens for must-epoch
              // operations where multiple tasks are coming through here
              // and we still need to record their mapping dependences
              // so we don't want to go into the scope. If we ever get
              // rid of must-epoch operations we can get rid of the
              // second part of this condition
              ((prev.op == user.op) ||
               (user.op->get_must_epoch_op() == nullptr)))
          {
            if ((prev.ctx_index == user.ctx_index) &&
                !(check_mask * it->second))
              user.op->report_interfering_requirements(prev.idx, user.idx);
            if (TRACK_DOM)
              dominator_mask -= it->second;
            continue;
          }
          const FieldMask overlap = check_mask & it->second;
          if (!!overlap)
          {
            if (TRACK_DOM)
              observed_mask |= overlap;
            const DependenceType dtype =
                check_dependence_type<true, true>(prev.usage, user.usage);
            switch (dtype)
            {
              case LEGION_NO_DEPENDENCE:
                {
                  // No dependence so remove bits from the dominator mask
                  dominator_mask -= it->second;
                  break;
                }
              case LEGION_ANTI_DEPENDENCE:
              case LEGION_ATOMIC_DEPENDENCE:
              case LEGION_SIMULTANEOUS_DEPENDENCE:
              case LEGION_TRUE_DEPENDENCE:
                {
                  // Check to see if we can record a point-wise dependence
                  // between these two operations. We can only do this if
                  // it they are both projections and we've arrived so
                  // they are both projecting from the same node in the
                  // region tree.
                  bool dominates = false;
                  if (arrived && state.record_pointwise_dependence(
                                     logical_analysis, prev, user, dominates))
                  {
                    user.op->register_pointwise_dependence(user.idx, prev);
                    // See if we the new user dominates or not
                    if (!dominates)
                      dominator_mask -= overlap;
                    LegionSpy::log_mapping_dependence(
                        user.op->get_context()->get_unique_id(), prev.uid,
                        prev.idx, user.uid, user.idx, dtype, true);
                  }
                  else
                  {
                    // If we can validate a region record which of our
                    // predecessors regions we are validating, otherwise
                    // just register a normal dependence
                    user.op->register_region_dependence(
                        user.idx, prev.op, prev.gen, prev.idx, dtype, overlap);
                    LegionSpy::log_mapping_dependence(
                        user.op->get_context()->get_unique_id(), prev.uid,
                        prev.idx, user.uid, user.idx, dtype);
                    if (prev.shard_proj != nullptr)
                    {
                      // Two operations from the same must epoch shouldn't
                      // be recording close dependences on each other so
                      // we can skip that part
                      if ((prev.ctx_index == user.ctx_index) &&
                          (user.op->get_must_epoch_op() != nullptr))
                        break;
                      // If this is a sharding projection operation then check
                      // to see if we need to record a fence dependence here to
                      // ensure that we get dependences between interfering
                      // points in different shards correct
                      // There are three sceanrios here:
                      // 1. We haven't arrived in which case we don't have any
                      //    good way to symbolically prove it is safe to elide
                      //    the fence so just record the close
                      // 2. We've arrived but we're not projection in which case
                      //    we'll interfere with any projections anyway so we
                      //    need a full fence for dependences anyway
                      // 3. We've arrived and are projecting in which case we
                      // can
                      //    try to elide things symbolically, if that doesn't
                      //    work we may still need to do an expensive analysis
                      //    to prove it is safe to elide the close which we'll
                      //    only do it we are tracing
                      if (arrived && proj_info.is_projecting())
                      {
                        // If we arrived and are projecting then we can test
                        // these two projection trees for intereference with
                        // each other and see if we can prove that they are
                        // disjoint in which case we don't need a close
                        legion_assert(
                            runtime->enable_pointwise_analysis ||
                            proj_info.is_sharding());
                        legion_assert(user.shard_proj != nullptr);
                        dominates = true;
                        if (!state.has_interfering_shards(
                                logical_analysis, prev.shard_proj,
                                user.shard_proj, dominates))
                        {
                          // If the two projections are non-interfering, then
                          // we can only consider the second projection as
                          // dominating the first if it uses all the same data.
                          // Otherwise you can cases like those that occur in
                          // https://github.com/StanfordLegion/legion/issues/1765
                          // where some index tasks push earlier index tasks out
                          // of the set of current/previous epoch users and then
                          // we end up missing a merge close op fence.
                          if (!dominates)
                            dominator_mask -= it->second;
                          break;
                        }
                      }
                      // We weren't able to prove that the projections were
                      // non-interfering with each other so we need a close
                      // Not able to do the symbolic elision so we need a fence
                      // across the shards to be safe
                      logical_analysis.record_close_dependence(
                          root, user.idx, this, &prev, overlap);
                      it.filter(overlap);
                      tighten = true;
                      if (!it->second)
                        to_delete.emplace_back(it->first);
                    }
                  }
                  break;
                }
              default:
                std::abort();  // should never get here
            }
          }
        }
        if (!to_delete.empty())
        {
          for (LogicalUser* user : to_delete)
          {
            prev_users.erase(user);
            if (user->remove_reference())
              delete user;
          }
        }
        if (tighten)
          prev_users.tighten_valid_mask();
      }
      // The result of this computation is the dominator mask.
      // It's only sound to say that we dominate fields that
      // we actually observed users for so intersect the dominator
      // mask with the observed mask
      if (TRACK_DOM)
      {
        // For writes, there is a special case here we actually
        // want to record that we are dominating fields which
        // are not actually open below even if we didn't see
        // any users on the way down
        if (IS_WRITE(user.usage))
        {
          FieldMask unobserved = check_mask - observed_mask;
          if (!!unobserved)
          {
            if (!open_below)
              observed_mask |= unobserved;
            else
              observed_mask |= (unobserved - open_below);
          }
        }
        return (dominator_mask & observed_mask);
      }
      else
        return dominator_mask;
    }

    // Instantiate the template for both templates because c++ is stupid
    template FieldMask RegionTreeNode::perform_dependence_checks<true>(
        LogicalRegion root, const LogicalUser& user,
        OrderedFieldMaskUsers& prev_users, const FieldMask& check_mask,
        const FieldMask& open_below, const bool arrived,
        const ProjectionInfo& proj_info, LogicalState& state,
        LogicalAnalysis& logical_analysis);
    template FieldMask RegionTreeNode::perform_dependence_checks<false>(
        LogicalRegion root, const LogicalUser& user,
        OrderedFieldMaskUsers& prev_users, const FieldMask& check_mask,
        const FieldMask& open_below, const bool arrived,
        const ProjectionInfo& proj_info, LogicalState& state,
        LogicalAnalysis& logical_analysis);

    //--------------------------------------------------------------------------
    /*static*/ void RegionTreeNode::perform_closing_checks(
        LogicalAnalysis& logical_analysis, OrderedFieldMaskUsers& users,
        const LogicalUser& user, const FieldMask& check_mask,
        LogicalRegion root, RegionTreeNode* path_node, FieldMask& still_open)
    //--------------------------------------------------------------------------
    {
      // Record dependences on all operations with the same field unless they
      // are different region requirements of the same user
      // There's nothing to do if the mask is disjoint from the users
      if (check_mask * users.get_valid_mask())
        return;
      std::vector<LogicalUser*> to_delete;
      for (OrderedFieldMaskUsers::iterator it = users.begin();
           it != users.end(); it++)
      {
        const FieldMask overlap = check_mask & it->second;
        if (!overlap)
          continue;
        const LogicalUser& prev = *it->first;
        // Skip any users from the same operation for different requiremnts
        if (prev.ctx_index == user.ctx_index)
        {
          user.op->report_interfering_requirements(prev.idx, user.idx);
          continue;
        }
        logical_analysis.record_close_dependence(
            root, user.idx, path_node, it->first, overlap);
        it.filter(overlap);
        if (!it->second)
          to_delete.emplace_back(it->first);
      }
      for (LogicalUser* user : to_delete)
      {
        users.erase(user);
        if (user->remove_reference())
          delete user;
      }
      users.tighten_valid_mask();
      if (!users.empty())
        still_open |= (check_mask & users.get_valid_mask());
    }

    /////////////////////////////////////////////////////////////
    // Region Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionNode::RegionNode(
        LogicalRegion r, PartitionNode* par, IndexSpaceNode* row_src,
        FieldSpaceNode* col_src, DistributedID id, RtEvent init, RtEvent tree,
        CollectiveMapping* mapping, Provenance* prov)
      : RegionTreeNode(col_src, init, tree, prov, id, mapping), handle(r),
        parent(par), row_source(row_src)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info(
          "GC Region %lld %d %lld %lld %lld",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space,
          handle.get_index_space().get_id(), handle.get_field_space().get_id(),
          handle.get_tree_id());
#endif
    }

    //--------------------------------------------------------------------------
    RegionNode::~RegionNode(void)
    //--------------------------------------------------------------------------
    {
      // The reason we would be here is if we were leaked
      if (!partition_trackers.empty())
      {
        for (PartitionTracker* tracker : partition_trackers)
          if (tracker->remove_partition_reference())
            delete tracker;
        partition_trackers.clear();
      }
      if (registered)
      {
        if (column_source->remove_nested_resource_ref(did))
          delete column_source;
        // Unregister oursleves with the row source
        if (row_source->remove_nested_resource_ref(did))
          delete row_source;
        const bool top_level = (parent == nullptr);
        // Unregister ourselves with the context
        runtime->remove_node(handle, top_level);
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::notify_local(void)
    //--------------------------------------------------------------------------
    {
      if (parent == nullptr)
      {
        runtime->release_tree_instances(handle.get_tree_id());
        if (row_source->parent == nullptr)
          row_source->remove_nested_valid_ref(did);
        else
          row_source->parent->remove_nested_valid_ref(did);
        column_source->remove_nested_gc_ref(did);
      }
      if (!partition_trackers.empty())
      {
        legion_assert(parent == nullptr);  // should only happen on the root
        for (PartitionTracker* tracker : partition_trackers)
          if (tracker->remove_partition_reference())
            delete tracker;
        partition_trackers.clear();
      }
      for (unsigned idx = 0; idx < current_versions.max_entries(); idx++)
        if (current_versions.has_entry(idx))
          get_current_version_manager(idx).finalize_manager();
    }

    //--------------------------------------------------------------------------
    void RegionNode::record_registered(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!registered);
      if (parent == nullptr)
      {
        if (row_source->parent == nullptr)
          row_source->add_nested_valid_ref(did);
        else
          row_source->parent->add_nested_valid_ref(did);
        column_source->add_nested_gc_ref(did);
      }
      else
        parent->add_child(this);
      column_source->add_nested_resource_ref(did);
      row_source->add_nested_resource_ref(did);
      registered = true;
      if (parent == nullptr)
        register_with_runtime();
    }

    //--------------------------------------------------------------------------
    bool RegionNode::has_color(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      // Ask the row source since it eagerly instantiates
      return row_source->has_color(c);
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionNode::get_child(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      // check to see if we have it, if not try to make it
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        std::map<LegionColor, PartitionNode*>::const_iterator finder =
            color_map.find(c);
        if (finder != color_map.end())
          return finder->second;
      }
      // If we get here we didn't immediately have it so try
      // to make it through the proper channels
      IndexPartNode* index_part = row_source->get_child(c);
      legion_assert(index_part != nullptr);
      LogicalPartition part_handle(
          handle.tree_did, index_part->handle, handle.field_space);
      return runtime->create_node(part_handle, this);
    }

    //--------------------------------------------------------------------------
    void RegionNode::add_child(PartitionNode* child)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      legion_assert(
          color_map.find(child->row_source->color) == color_map.end());
      color_map[child->row_source->color] = child;
    }

    //--------------------------------------------------------------------------
    void RegionNode::remove_child(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      std::map<LegionColor, PartitionNode*>::iterator finder =
          color_map.find(c);
      legion_assert(finder != color_map.end());
      color_map.erase(finder);
    }

    //--------------------------------------------------------------------------
    void RegionNode::add_tracker(PartitionTracker* tracker)
    //--------------------------------------------------------------------------
    {
      legion_assert(parent == nullptr);  // should only happen on the root
      std::vector<PartitionTracker*> to_prune;
      {
        AutoLock n_lock(node_lock);
        // To avoid leaks, see if there are any other trackers we can prune
        for (std::list<PartitionTracker*>::iterator it =
                 partition_trackers.begin();
             it != partition_trackers.end();
             /*nothing*/)
        {
          if ((*it)->can_prune())
          {
            to_prune.emplace_back(*it);
            it = partition_trackers.erase(it);
          }
          else
            it++;
        }
        partition_trackers.emplace_back(tracker);
      }
      for (PartitionTracker* tracker : to_prune)
        if (tracker->remove_reference())
          delete tracker;
    }

    //--------------------------------------------------------------------------
    void RegionNode::initialize_no_refine_fields(
        ContextID ctx, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      LogicalState& state = get_logical_state(ctx);
#ifdef LEGION_DEBUG
      state.sanity_check();
#endif
      state.initialize_no_refine_fields(mask);
    }

    //--------------------------------------------------------------------------
    unsigned RegionNode::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->depth;
    }

    //--------------------------------------------------------------------------
    LegionColor RegionNode::get_color(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->color;
    }

    //--------------------------------------------------------------------------
    IndexTreeNode* RegionNode::get_row_source(void) const
    //--------------------------------------------------------------------------
    {
      return row_source;
    }

    //--------------------------------------------------------------------------
    RegionTreeID RegionNode::get_tree_id(void) const
    //--------------------------------------------------------------------------
    {
      return handle.get_tree_id();
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* RegionNode::get_parent(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* RegionNode::get_tree_child(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      return get_child(c);
    }

    //--------------------------------------------------------------------------
    bool RegionNode::are_children_disjoint(
        const LegionColor c1, const LegionColor c2)
    //--------------------------------------------------------------------------
    {
      return row_source->are_disjoint(c1, c2);
    }

    //--------------------------------------------------------------------------
    bool RegionNode::are_all_children_disjoint(void)
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionNode::is_region(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

#ifdef LEGION_DEBUG
    //--------------------------------------------------------------------------
    RegionNode* RegionNode::as_region_node(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<RegionNode*>(this);
    }

    //--------------------------------------------------------------------------
    PartitionNode* RegionNode::as_partition_node(void) const
    //--------------------------------------------------------------------------
    {
      return nullptr;
    }
#endif

    //--------------------------------------------------------------------------
    bool RegionNode::visit_node(PathTraverser* traverser)
    //--------------------------------------------------------------------------
    {
      return traverser->visit_region(this);
    }

    //--------------------------------------------------------------------------
    bool RegionNode::visit_node(NodeTraverser* traverser)
    //--------------------------------------------------------------------------
    {
      bool continue_traversal = traverser->visit_region(this);
      if (continue_traversal)
      {
        const bool break_early = traverser->break_early();
        if (traverser->force_instantiation)
        {
          // If we are forcing instantiation, then grab the set of
          // colors from the row source and use them to instantiate children
          std::vector<LegionColor> children_colors;
          row_source->get_colors(children_colors);
          for (const LegionColor& child_color : children_colors)
          {
            bool result = get_child(child_color)->visit_node(traverser);
            continue_traversal = continue_traversal && result;
            if (!result && break_early)
              break;
          }
        }
        else
        {
          std::map<LegionColor, PartitionNode*> children;
          // Need to hold the lock when reading from
          // the color map or the valid map
          {
            AutoLock n_lock(node_lock, false /*exclusive*/);
            for (const std::pair<const LegionColor, PartitionNode*>& it :
                 color_map)
            {
              children.insert(it);
              it.second->add_base_resource_ref(REGION_TREE_REF);
            }
          }
          for (std::map<LegionColor, PartitionNode*>::const_iterator it =
                   children.begin();
               it != children.end(); it++)
          {
            const bool result = it->second->visit_node(traverser);
            if (it->second->remove_base_resource_ref(REGION_TREE_REF))
              delete it->second;
            continue_traversal = continue_traversal && result;
            if (!result && break_early)
            {
              it++;
              while (it != children.end())
              {
                if (it->second->remove_base_resource_ref(REGION_TREE_REF))
                  delete it->second;
                it++;
              }
              break;
            }
          }
        }
      }
      return continue_traversal;
    }

    //--------------------------------------------------------------------------
    bool RegionNode::is_complete(void)
    //--------------------------------------------------------------------------
    {
      // For now just assume that regions are never complete
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionNode::intersects_with(RegionTreeNode* other, bool compute)
    //--------------------------------------------------------------------------
    {
      if (other == this)
        return true;
      if (other->is_region())
        return row_source->intersects_with(
            other->as_region_node()->row_source, compute);
      else
        return row_source->intersects_with(
            other->as_partition_node()->row_source, compute);
    }

    //--------------------------------------------------------------------------
    bool RegionNode::track_refinements(void) const
    //--------------------------------------------------------------------------
    {
      if (parent == nullptr)
        return true;
      // One of two conditions needs to be met:
      // 1. this region is "large" enough on its own to track a refinement
      // 2. this region needs to be at least some fraction of the volume of
      //    the parent region of the partition
      // The first case makes sure we don't make equivalence sets that are
      // too small in general. The second case allows for some degree of
      // small equivalence sets as long as the degree of parallelism is
      // bounded to a certain extent.
      const size_t volume = row_source->get_volume();
      if (EquivalenceSet::MINIMUM_SIZE <= volume)
        return true;
      if (volume == 0)
        return false;
      const size_t parent_volume = parent->parent->row_source->get_volume();
      legion_assert(volume <= parent_volume);
      const size_t ratio = parent_volume / volume;
      return (ratio <= EquivalenceSet::MINIMUM_RATIO);
    }

    //--------------------------------------------------------------------------
    size_t RegionNode::get_num_children(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->get_num_children();
    }

    //--------------------------------------------------------------------------
    void RegionNode::send_node(Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have it in our creation set in which
      // case we are done otherwise keep going up
      bool continue_up = false;
      {
        AutoLock n_lock(node_lock);
        if (!has_remote_instance(target))
        {
          continue_up = true;
          update_remote_instances(target);
        }
      }
      if (continue_up)
      {
        if (parent != nullptr)
        {
          // Send the parent node first
          parent->send_node(rez, target);
          AutoLock n_lock(node_lock);
          for (lng::map<SemanticTag, SemanticInfo>::iterator it =
                   semantic_info.begin();
               it != semantic_info.end(); it++)
          {
            LogicalRegionSemanticInfoResponse rez2;
            {
              RezCheck z(rez2);
              rez2.serialize(handle);
              rez2.serialize(initialized);
              rez2.serialize<size_t>(1);
              rez2.serialize(it->first);
              rez2.serialize(it->second.buffer.get_size());
              rez2.serialize(
                  it->second.buffer.get_buffer(), it->second.buffer.get_size());
              rez2.serialize(it->second.is_mutable);
            }
            rez2.dispatch(target);
          }
        }
        else
        {
          rez.serialize(handle);
          rez.serialize(did);
          rez.serialize(initialized);
          if (provenance != nullptr)
            provenance->serialize(rez);
          else
            Provenance::serialize_null(rez);
          if (collective_mapping != nullptr)
            collective_mapping->pack(rez);
          else
            CollectiveMapping::pack_null(rez);
          rez.serialize<size_t>(semantic_info.size());
          for (lng::map<SemanticTag, SemanticInfo>::iterator it =
                   semantic_info.begin();
               it != semantic_info.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second.buffer.get_size());
            rez.serialize(
                it->second.buffer.get_buffer(), it->second.buffer.get_size());
            rez.serialize(it->second.is_mutable);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RegionNode::handle_node_creation(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      LogicalRegion handle;
      derez.deserialize(handle);
      DistributedID did;
      derez.deserialize(did);
      RtEvent initialized;
      derez.deserialize(initialized);
      AutoProvenance prov(Provenance::deserialize(derez), true /*has ref*/);
      size_t num_spaces;
      derez.deserialize(num_spaces);
      CollectiveMapping* mapping = nullptr;
      if (num_spaces > 0)
        mapping = new CollectiveMapping(derez, num_spaces);

      RegionNode* node = runtime->create_node(
          handle, nullptr /*parent*/, initialized, did, prov, mapping);
      legion_assert(node != nullptr);
      size_t num_semantic;
      derez.deserialize(num_semantic);
      if (num_semantic > 0)
      {
        NodeSet<TASK_LOCAL_LIFETIME> source_mask;
        source_mask.add(source);
        source_mask.add(runtime->address_space);
        for (unsigned idx = 0; idx < num_semantic; idx++)
        {
          SemanticTag tag;
          derez.deserialize(tag);
          size_t buffer_size;
          derez.deserialize(buffer_size);
          const void* buffer = derez.get_current_pointer();
          derez.advance_pointer(buffer_size);
          bool is_mutable;
          derez.deserialize(is_mutable);
          node->attach_semantic_information(
              tag, source, buffer, buffer_size, is_mutable,
              false /*local only*/);
        }
      }
    }

    //--------------------------------------------------------------------------
    void RegionNode::perform_versioning_analysis(
        ContextID ctx, InnerContext* parent_ctx, VersionInfo* version_info,
        const FieldMask& mask, Operation* op, unsigned index,
        unsigned parent_req_index, std::set<RtEvent>& applied,
        RtEvent* output_region_ready, bool collective_rendezvous)
    //--------------------------------------------------------------------------
    {
      VersionManager& manager = get_current_version_manager(ctx);
      manager.perform_versioning_analysis(
          parent_ctx, version_info, this, mask, op, index, parent_req_index,
          applied, output_region_ready, collective_rendezvous);
    }

    //--------------------------------------------------------------------------
    void RegionNode::find_open_complete_partitions(
        ContextID ctx, const FieldMask& mask,
        std::vector<LogicalPartition>& partitions)
    //--------------------------------------------------------------------------
    {
      LogicalState& state = get_logical_state(ctx);
      std::set<LogicalPartition> unique_partitions;
      for (shrt::list<FieldState>::const_iterator sit =
               state.field_states.begin();
           sit != state.field_states.end(); sit++)
      {
        if (sit->valid_fields() * mask)
          continue;
        for (OrderedFieldMaskChildren::const_iterator it =
                 sit->open_children.begin();
             it != sit->open_children.end(); it++)
        {
          if (it->second * mask)
            continue;
          PartitionNode* child = it->first->as_partition_node();
          if (child->is_complete())
            unique_partitions.insert(child->handle);
        }
      }
      partitions.insert(
          partitions.end(), unique_partitions.begin(), unique_partitions.end());
    }

    //--------------------------------------------------------------------------
    void RegionNode::send_semantic_request(
        AddressSpaceID target, SemanticTag tag, bool can_fail, bool wait_until,
        RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      LogicalRegionSemanticInfoRequest rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(can_fail);
        rez.serialize(wait_until);
        rez.serialize(ready);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID RegionNode::find_semantic_owner(void) const
    //--------------------------------------------------------------------------
    {
      // If we're the root, then the owner is the owner of the root
      // Otherwise the owner is the owner of the corresponding index space
      if (parent == nullptr)
        return owner_space;
      else
        return row_source->owner_space;
    }

    //--------------------------------------------------------------------------
    void RegionNode::send_semantic_info(
        AddressSpaceID target, SemanticTag tag, const void* buffer, size_t size,
        bool is_mutable, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      // Package up the message first
      LogicalRegionSemanticInfoResponse rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(size);
        rez.serialize(buffer, size);
        rez.serialize(is_mutable);
        rez.serialize(ready);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    void RegionNode::process_semantic_request(
        SemanticTag tag, AddressSpaceID source, bool can_fail, bool wait_until,
        RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(find_semantic_owner() == runtime->address_space);
      RtEvent precondition;
      const void* result = nullptr;
      size_t size = 0;
      bool is_mutable = false;
      {
        AutoLock n_lock(node_lock);
        // See if we already have the data
        lng::map<SemanticTag, SemanticInfo>::iterator finder =
            semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          if (finder->second.is_valid())
          {
            result = finder->second.buffer.get_buffer();
            size = finder->second.buffer.get_size();
            is_mutable = finder->second.is_mutable;
          }
          else if (!can_fail && wait_until)
            precondition = finder->second.ready_event;
        }
        else if (!can_fail && wait_until)
        {
          // Don't have it yet, make a condition and hope that one comes
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          precondition = ready_event;
          semantic_info[tag] = SemanticInfo(ready_event);
        }
      }
      if (result == nullptr)
      {
        if (can_fail || !wait_until)
          Runtime::trigger_event(ready);
        else
        {
          // Defer this until the semantic condition is ready
          SemanticRequestArgs args(this, tag, source);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_WORK_PRIORITY, precondition);
        }
      }
      else
        send_semantic_info(source, tag, result, size, is_mutable, ready);
    }

    //--------------------------------------------------------------------------
    void RegionNode::SemanticRequestArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      proxy_this->process_semantic_request(
          tag, source, false, false, RtUserEvent::NO_RT_USER_EVENT);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LogicalRegionSemanticInfoRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LogicalRegion handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      bool can_fail;
      derez.deserialize(can_fail);
      bool wait_until;
      derez.deserialize(wait_until);
      RtUserEvent ready;
      derez.deserialize(ready);
      RegionNode* node = runtime->get_node(handle);
      node->process_semantic_request(tag, source, can_fail, wait_until, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LogicalRegionSemanticInfoResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LogicalRegion handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      size_t size;
      derez.deserialize(size);
      const void* buffer = derez.get_current_pointer();
      derez.advance_pointer(size);
      bool is_mutable;
      derez.deserialize(is_mutable);
      RtUserEvent ready;
      derez.deserialize(ready);
      runtime->get_node(handle)->attach_semantic_information(
          tag, source, buffer, size, is_mutable, false /*local only*/);
      if (ready.exists())
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TopLevelRegionRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      RegionTreeID tid;
      derez.deserialize(tid);
      RegionNode* node = runtime->get_tree(tid, true /*can fail*/);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      AddressSpaceID source;
      derez.deserialize(source);
      if (node != nullptr)
      {
        // If there is a collective mapping, check to see if we're on the
        // right node and if not forward it on to the right node
        if (node->collective_mapping != nullptr)
        {
          legion_assert(!node->collective_mapping->contains(source));
          legion_assert(node->collective_mapping->contains(node->local_space));
          if (node->is_owner())
          {
            const AddressSpaceID nearest =
                node->collective_mapping->find_nearest(source);
            // If we're not the nearest then forward it on to the
            // proper node to handle the request
            if (nearest != node->local_space)
            {
              TopLevelRegionRequest rez;
              rez.serialize(tid);
              rez.serialize(done_event);
              rez.serialize(source);
              rez.dispatch(nearest);
              return;
            }
          }
          else
          {
            legion_assert(
                node->local_space ==
                node->collective_mapping->find_nearest(source));
          }
        }
        TopLevelRegionReturn rez;
        {
          RezCheck z(rez);
          node->send_node(rez, source);
          rez.serialize(done_event);
        }
        rez.dispatch(source);
      }
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TopLevelRegionReturn::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RegionNode::handle_node_creation(derez, source);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
    }

    /////////////////////////////////////////////////////////////
    // Partition Tracker
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PartitionTracker::PartitionTracker(PartitionNode* part)
      : Collectable(2 /*expecting two reference calls*/), partition(part)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool PartitionTracker::can_prune(void)
    //--------------------------------------------------------------------------
    {
      const unsigned remainder = references.load();
      legion_assert((remainder == 1) || (remainder == 2));
      return (remainder == 1);
    }

    //--------------------------------------------------------------------------
    bool PartitionTracker::remove_partition_reference()
    //--------------------------------------------------------------------------
    {
      // Pull a copy of this on to the stack in case we get deleted
      std::atomic<PartitionNode*> node(partition);
      const bool last = remove_reference();
      // If we weren't the last one that means we remove the reference
      if (!last && node.load()->remove_base_gc_ref(REGION_TREE_REF))
        delete node.load();
      return last;
    }

    /////////////////////////////////////////////////////////////
    // Partition Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PartitionNode::PartitionNode(
        LogicalPartition p, RegionNode* par, IndexPartNode* row_src,
        FieldSpaceNode* col_src, RtEvent init, RtEvent tree)
      : RegionTreeNode(col_src, init, tree, par->provenance), handle(p),
        parent(par), row_source(row_src)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info(
          "GC Partition %lld %d %lld %lld %lld",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space,
          handle.get_index_partition().get_id(),
          handle.get_field_space().get_id(), handle.get_tree_id());
#endif
    }

    //--------------------------------------------------------------------------
    PartitionNode::~PartitionNode(void)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const LegionColor, RegionNode*>& it : color_map)
        if (it.second->remove_nested_resource_ref(did))
          delete it.second;
      if (registered)
      {
        if (parent->remove_nested_resource_ref(did))
          delete parent;
        // Unregister ourselves with our row source
        if (row_source->remove_nested_resource_ref(did))
          delete row_source;
        // Then unregister ourselves with the context
        runtime->remove_node(handle);
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::notify_local(void)
    //--------------------------------------------------------------------------
    {
      parent->remove_child(row_source->color);
      row_source->remove_nested_gc_ref(did);
      // Remove gc references on all of our child nodes
      // We should not need a lock at this point since nobody else should
      // be modifying these data structures at this point
      // No need to check for deletion since we hold resource references
      for (const std::pair<const LegionColor, RegionNode*>& it : color_map)
        it.second->remove_nested_gc_ref(did);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::record_registered(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!registered);
      row_source->add_nested_resource_ref(did);
      row_source->add_nested_gc_ref(did);
      parent->add_nested_resource_ref(did);
      parent->add_child(this);
      // Create a partition deletion tracker for this node and add it to
      // both the index partition node and the root logical region for this
      // tree so that we can have our reference removed once either is deleted
      PartitionTracker* tracker = new PartitionTracker(this);
      row_source->add_tracker(tracker);
      RegionNode* root = parent;
      while (root->parent != nullptr) root = root->parent->parent;
      root->add_tracker(tracker);
      registered = true;
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::has_color(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      // Ask the row source because it eagerly instantiates
      return row_source->has_color(c);
    }

    //--------------------------------------------------------------------------
    RegionNode* PartitionNode::get_child(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      // check to see if we have it, if not try to make it
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        std::map<LegionColor, RegionNode*>::const_iterator finder =
            color_map.find(c);
        if (finder != color_map.end())
          return finder->second;
      }
      // If we get here we didn't immediately have it so try
      // to make it through the proper channels
      IndexSpaceNode* index_node = row_source->get_child(c, nullptr);
      legion_assert(index_node != nullptr);
      LogicalRegion reg_handle(
          handle.tree_did, index_node->handle, handle.field_space);
      return runtime->create_node(
          reg_handle, this, RtEvent::NO_RT_EVENT, 0 /*did*/);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::add_child(RegionNode* child)
    //--------------------------------------------------------------------------
    {
      child->add_nested_resource_ref(did);
      child->add_nested_gc_ref(did);
      AutoLock n_lock(node_lock);
      legion_assert(is_global());
      legion_assert(
          color_map.find(child->row_source->color) == color_map.end());
      color_map[child->row_source->color] = child;
    }

    //--------------------------------------------------------------------------
    unsigned PartitionNode::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->depth;
    }

    //--------------------------------------------------------------------------
    LegionColor PartitionNode::get_color(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->color;
    }

    //--------------------------------------------------------------------------
    IndexTreeNode* PartitionNode::get_row_source(void) const
    //--------------------------------------------------------------------------
    {
      return row_source;
    }

    //--------------------------------------------------------------------------
    RegionTreeID PartitionNode::get_tree_id(void) const
    //--------------------------------------------------------------------------
    {
      return handle.get_tree_id();
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* PartitionNode::get_parent(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* PartitionNode::get_tree_child(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      return get_child(c);
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::are_children_disjoint(
        const LegionColor c1, const LegionColor c2)
    //--------------------------------------------------------------------------
    {
      return row_source->are_disjoint(c1, c2);
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::are_all_children_disjoint(void)
    //--------------------------------------------------------------------------
    {
      return row_source->is_disjoint();
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::is_region(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

#ifdef LEGION_DEBUG
    //--------------------------------------------------------------------------
    RegionNode* PartitionNode::as_region_node(void) const
    //--------------------------------------------------------------------------
    {
      return nullptr;
    }

    //--------------------------------------------------------------------------
    PartitionNode* PartitionNode::as_partition_node(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<PartitionNode*>(this);
    }
#endif

    //--------------------------------------------------------------------------
    bool PartitionNode::visit_node(PathTraverser* traverser)
    //--------------------------------------------------------------------------
    {
      return traverser->visit_partition(this);
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::visit_node(NodeTraverser* traverser)
    //--------------------------------------------------------------------------
    {
      bool continue_traversal = traverser->visit_partition(this);
      if (continue_traversal)
      {
        const bool break_early = traverser->break_early();
        if (traverser->force_instantiation)
        {
          for (ColorSpaceIterator itr(row_source); itr; itr++)
          {
            bool result = get_child(*itr)->visit_node(traverser);
            continue_traversal = continue_traversal && result;
            if (!result && break_early)
              break;
          }
        }
        else
        {
          std::map<LegionColor, RegionNode*> children;
          // Need to hold the lock when reading from
          // the color map or the valid map
          {
            AutoLock n_lock(node_lock, false /*exclusive*/);
            children = color_map;
          }
          for (const std::pair<const LegionColor, RegionNode*>& it : children)
          {
            bool result = it.second->visit_node(traverser);
            continue_traversal = continue_traversal && result;
            if (!result && break_early)
              break;
          }
        }
      }
      return continue_traversal;
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::is_complete(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(parent != nullptr);
      return row_source->is_complete();
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::intersects_with(RegionTreeNode* other, bool compute)
    //--------------------------------------------------------------------------
    {
      if (other == this)
        return true;
      if (other->is_region())
        return row_source->intersects_with(
            other->as_region_node()->row_source, compute);
      else
        return row_source->intersects_with(
            other->as_partition_node()->row_source, compute);
    }

    //--------------------------------------------------------------------------
    bool PartitionNode::track_refinements(void) const
    //--------------------------------------------------------------------------
    {
      // Always track partitions
      // Complete: should always track these
      // Incomplete: either we're going to traverse to a child region and
      // we'll determine whether to traverse that child or we're projecting
      // from this partition and we'll refine the parent when partitions change
      return true;
    }

    //--------------------------------------------------------------------------
    size_t PartitionNode::get_num_children(void) const
    //--------------------------------------------------------------------------
    {
      return row_source->get_num_children();
    }

    //--------------------------------------------------------------------------
    void PartitionNode::send_node(Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have it in our creation set in which
      // case we are done otherwise keep going up
      bool continue_up = false;
      {
        AutoLock n_lock(node_lock);
        if (!has_remote_instance(target))
        {
          continue_up = true;
          update_remote_instances(target);
        }
      }
      if (continue_up)
      {
        legion_assert(parent != nullptr);
        // Send the parent node first
        parent->send_node(rez, target);
        AutoLock n_lock(node_lock);
        for (lng::map<SemanticTag, SemanticInfo>::iterator it =
                 semantic_info.begin();
             it != semantic_info.end(); it++)
        {
          LogicalPartitionSemanticInfoResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(it->first);
            rez.serialize(it->second.buffer.get_size());
            rez.serialize(
                it->second.buffer.get_buffer(), it->second.buffer.get_size());
            rez.serialize(it->second.is_mutable);
          }
          rez.dispatch(target);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PartitionNode::send_semantic_request(
        AddressSpaceID target, SemanticTag tag, bool can_fail, bool wait_until,
        RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      LogicalPartitionSemanticInfoRequest rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(can_fail);
        rez.serialize(wait_until);
        rez.serialize(ready);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID PartitionNode::find_semantic_owner(void) const
    //--------------------------------------------------------------------------
    {
      // The owner is the owner of our row source partition
      return row_source->owner_space;
    }

    //--------------------------------------------------------------------------
    void PartitionNode::send_semantic_info(
        AddressSpaceID target, SemanticTag tag, const void* buffer, size_t size,
        bool is_mutable, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      // Package up the message first
      LogicalPartitionSemanticInfoResponse rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(size);
        rez.serialize(buffer, size);
        rez.serialize(is_mutable);
        rez.serialize(ready);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::process_semantic_request(
        SemanticTag tag, AddressSpaceID source, bool can_fail, bool wait_until,
        RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(find_semantic_owner() == runtime->address_space);
      RtEvent precondition;
      const void* result = nullptr;
      size_t size = 0;
      bool is_mutable = false;
      {
        AutoLock n_lock(node_lock);
        // See if we already have the data
        lng::map<SemanticTag, SemanticInfo>::iterator finder =
            semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          if (finder->second.is_valid())
          {
            result = finder->second.buffer.get_buffer();
            size = finder->second.buffer.get_size();
            is_mutable = finder->second.is_mutable;
          }
          else if (!can_fail && wait_until)
            precondition = finder->second.ready_event;
        }
        else if (!can_fail && wait_until)
        {
          // Don't have it yet, make a condition and hope that one comes
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          precondition = ready_event;
          semantic_info[tag] = SemanticInfo(ready_event);
        }
      }
      if (result == nullptr)
      {
        if (can_fail || !wait_until)
          Runtime::trigger_event(ready);
        else
        {
          // Defer this until the semantic condition is ready
          SemanticRequestArgs args(this, tag, source);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_WORK_PRIORITY, precondition);
        }
      }
      else
        send_semantic_info(source, tag, result, size, is_mutable, ready);
    }

    //--------------------------------------------------------------------------
    void PartitionNode::SemanticRequestArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      proxy_this->process_semantic_request(
          tag, source, false, false, RtUserEvent::NO_RT_USER_EVENT);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LogicalPartitionSemanticInfoRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LogicalPartition handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      bool can_fail;
      derez.deserialize(can_fail);
      bool wait_until;
      derez.deserialize(wait_until);
      RtUserEvent ready;
      derez.deserialize(ready);
      PartitionNode* node = runtime->get_node(handle);
      node->process_semantic_request(tag, source, can_fail, wait_until, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LogicalPartitionSemanticInfoResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LogicalPartition handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      size_t size;
      derez.deserialize(size);
      const void* buffer = derez.get_current_pointer();
      derez.advance_pointer(size);
      bool is_mutable;
      derez.deserialize(is_mutable);
      RtUserEvent ready;
      derez.deserialize(ready);
      runtime->get_node(handle)->attach_semantic_information(
          tag, source, buffer, size, is_mutable, false /*local only*/);
      if (ready.exists())
        Runtime::trigger_event(ready);
    }

  }  // namespace Internal
}  // namespace Legion
