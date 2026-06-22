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

// Included from index.h - do not include this directly

// Useful for IDEs
#include "legion/nodes/index.h"

namespace Legion {
  namespace Internal {

#ifndef LEGION_DEBUG
    //--------------------------------------------------------------------------
    inline IndexSpaceNode* IndexTreeNode::as_index_space_node(void)
    //--------------------------------------------------------------------------
    {
      return static_cast<IndexSpaceNode*>(this);
    }

    //--------------------------------------------------------------------------
    inline IndexPartNode* IndexTreeNode::as_index_part_node(void)
    //--------------------------------------------------------------------------
    {
      return static_cast<IndexPartNode*>(this);
    }
#endif

#ifdef DEFINE_NT_TEMPLATES
    /////////////////////////////////////////////////////////////
    // PieceIteratorImplT
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    PieceIteratorImplT<DIM, T>::PieceIteratorImplT(
        const void* piece_list, size_t piece_list_size,
        IndexSpaceNodeT<DIM, T>* privilege_node)
      : PieceIteratorImpl()
    //--------------------------------------------------------------------------
    {
      legion_assert((piece_list_size % sizeof(Rect<DIM, T>)) == 0);
      const size_t num_pieces = piece_list_size / sizeof(Rect<DIM, T>);
      const Rect<DIM, T>* rects = static_cast<const Rect<DIM, T>*>(piece_list);
      if (privilege_node != nullptr)
      {
        DomainT<DIM, T> privilege_space =
            privilege_node->get_tight_index_space();
        for (unsigned idx = 0; idx < num_pieces; idx++)
        {
          const Rect<DIM, T>& rect = rects[idx];
          for (Realm::IndexSpaceIterator<DIM, T> itr(privilege_space);
               itr.valid; itr.step())
          {
            const Rect<DIM, T> overlap = rect.intersection(itr.rect);
            if (!overlap.empty())
              pieces.emplace_back(overlap);
          }
        }
      }
      else
      {
        pieces.resize(num_pieces);
        for (unsigned idx = 0; idx < num_pieces; idx++)
          pieces[idx] = rects[idx];
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    int PieceIteratorImplT<DIM, T>::get_next(int index, Domain& next_piece)
    //--------------------------------------------------------------------------
    {
      legion_assert(index >= -1);
      const unsigned next = index + 1;
      if (next < pieces.size())
      {
        next_piece = pieces[next];
        return int(next);
      }
      else
        return -1;
    }

    // This is a bit out of place but needs to be here for instantiation to work

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    PieceIteratorImpl* IndexSpaceOperationT<DIM, T>::create_piece_iterator(
        const void* piece_list, size_t piece_list_size,
        IndexSpaceNode* priv_node)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNodeT<DIM, T>* privilege_node =
          legion_safe_cast<IndexSpaceNodeT<DIM, T>*>(priv_node);
      if (piece_list == nullptr)
      {
        DomainT<DIM, T> realm_space = get_tight_index_space();
        // If there was no piece list it has to be because there
        // was just one piece which was a single dense rectangle
        legion_assert(realm_space.dense());
        return new PieceIteratorImplT<DIM, T>(
            &realm_space.bounds, sizeof(realm_space.bounds), privilege_node);
      }
      else
        return new PieceIteratorImplT<DIM, T>(
            piece_list, piece_list_size, privilege_node);
    }

    /////////////////////////////////////////////////////////////
    // Templated Index Space Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceNodeT<DIM, T>::IndexSpaceNodeT(
        IndexSpace handle, IndexPartNode* parent, LegionColor color,
        IndexSpaceExprID expr_id, RtEvent init, unsigned dep, Provenance* prov,
        CollectiveMapping* mapping, bool tree_valid)
      : IndexSpaceNode(
            handle, parent, color, expr_id, init, dep, prov, mapping,
            tree_valid),
        realm_index_space(Realm::IndexSpace<DIM, T>::make_empty()),
        linearization(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceNodeT<DIM, T>::~IndexSpaceNodeT(void)
    //--------------------------------------------------------------------------
    {
      // Do the clean-up for any sparsity map users
      if (!realm_index_space.dense())
      {
        // Check to see if we have any index space users that need to be
        // checked for deferring the destruction of the index space
        std::vector<ApEvent> preconditions;
        while (!index_space_users.empty())
        {
          if (!index_space_users.front().has_triggered_faultignorant())
            preconditions.emplace_back(index_space_users.front());
          index_space_users.pop_front();
        }
        if (!preconditions.empty())
        {
          if (!index_space_valid.has_triggered_faultignorant())
            preconditions.emplace_back(index_space_valid);
          index_space_valid = Runtime::merge_events(nullptr, preconditions);
          // Protect it if necessary to make sure the deletion happens
          // even if one of the users was poisoned
          if (index_space_valid.exists())
            index_space_valid =
                ApEvent(Runtime::protect_event(index_space_valid));
        }
        realm_index_space.destroy(index_space_valid);
      }
      ColorSpaceLinearizationT<DIM, T>* linear = linearization.load();
      if (linear != nullptr)
        delete linear;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM, T>::is_sparse(void)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> result = get_tight_index_space();
      return !result.dense();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    DomainT<DIM, T> IndexSpaceNodeT<DIM, T>::get_tight_index_space(void)
    //--------------------------------------------------------------------------
    {
      if (!index_space_set.load())
      {
        RtEvent wait_on;
        {
          AutoLock n_lock(node_lock);
          if (!index_space_set.load())
          {
            if (!index_space_ready.exists())
              index_space_ready = Runtime::create_rt_user_event();
            wait_on = index_space_ready;
          }
        }
        if (wait_on.exists())
          wait_on.wait();
      }
      if (!index_space_tight.load())
      {
        RtEvent wait_on;
        {
          AutoLock n_lock(node_lock);
          if (!index_space_tight.load())
          {
            if (!index_space_ready.exists())
              index_space_ready = Runtime::create_rt_user_event();
            wait_on = index_space_ready;
          }
        }
        if (wait_on.exists())
          wait_on.wait();
      }
      // Can read without the lock since there are no more modifications
      return realm_index_space;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::get_loose_index_space(
        DomainT<DIM, T>& result, ApUserEvent& to_trigger)
    //--------------------------------------------------------------------------
    {
      if (!index_space_set.load())
      {
        RtEvent wait_on;
        {
          AutoLock n_lock(node_lock);
          if (!index_space_set.load())
          {
            if (!index_space_ready.exists())
              index_space_ready = Runtime::create_rt_user_event();
            wait_on = index_space_ready;
          }
        }
        if (wait_on.exists())
          wait_on.wait();
      }
      AutoLock n_lock(node_lock);
      result = realm_index_space;
      // If this is not yet tight we need to record an event that is a user
      // that the caller will have to trigger later
      if (!index_space_tight.load())
      {
        if (!to_trigger.exists())
          to_trigger = Runtime::create_ap_user_event(nullptr);
        while (!index_space_users.empty() &&
               index_space_users.front().has_triggered_faultignorant())
          index_space_users.pop_front();
        index_space_users.emplace_back(to_trigger);
      }
      return index_space_valid;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    Domain IndexSpaceNodeT<DIM, T>::get_tight_domain(void)
    //--------------------------------------------------------------------------
    {
      return Domain(get_tight_index_space());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::get_loose_domain(
        Domain& domain, ApUserEvent& to_trigger)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> index_space;
      const ApEvent result = get_loose_index_space(index_space, to_trigger);
      domain = index_space;
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    RtEvent IndexSpaceNodeT<DIM, T>::add_sparsity_map_references(
        const Domain& domain, unsigned references)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> index_space = domain;
      legion_assert(references > 0);
      legion_assert(!index_space.dense());
      return RtEvent(index_space.sparsity.add_reference(references));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::record_index_space_user(ApEvent user)
    //--------------------------------------------------------------------------
    {
      if (user.exists())
      {
        AutoLock n_lock(node_lock);
        // Only need to save this if the sparsity map exists
        if (realm_index_space.dense())
          return;
        while (!index_space_users.empty() &&
               index_space_users.front().has_triggered_faultignorant())
          index_space_users.pop_front();
        index_space_users.emplace_back(user);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM, T>::set_realm_index_space(
        const Realm::IndexSpace<DIM, T>& value, ApEvent valid,
        bool initializing, bool broadcast, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      // If we're broadcasting and we're the source node then add references
      // for everything in the collective map since we know that they're all
      // going to end up having a copy of the realm index space
      if (!value.dense() && broadcast && (source == local_space) &&
          (collective_mapping != nullptr))
      {
        Realm::SparsityMap<DIM, T> copy = value.sparsity;
        const RtEvent ready(copy.add_reference(
            collective_mapping->size() -
            (collective_mapping->contains(local_space) ? 1 : 0)));
        // In general waiting is bad, but in this case we know that we're
        // on the same node where the index space was probably made so
        // this is probably going to be a NO_EVENT and we won't end up
        // having to wait.
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
      }
      // We can set this now and trigger the event but setting the
      // flag has to be done while holding the node_lock on the owner
      // node so that it is serialized with respect to queries from
      // remote nodes for copies about the remote instance
      {
        AutoLock n_lock(node_lock);
        legion_assert(!index_space_set.load());
        realm_index_space = value;
        index_space_valid = valid;
        index_space_set.store(true);
        if (index_space_ready.exists())
        {
          Runtime::trigger_event(index_space_ready);
          index_space_ready = RtUserEvent::NO_RT_USER_EVENT;
        }
        if (broadcast)
        {
          if ((collective_mapping != nullptr) &&
              collective_mapping->contains(local_space))
          {
            std::vector<AddressSpaceID> children;
            collective_mapping->get_children(
                owner_space, local_space, children);
            const AddressSpaceID parent_space =
                is_owner() ?
                    source :
                    collective_mapping->get_parent(owner_space, local_space);
            if (!children.empty() || (parent_space != source))
            {
              IndexSpaceSet rez;
              {
                RezCheck z(rez);
                if (parent != nullptr)
                {
                  rez.serialize(parent->handle);
                  rez.serialize(color);
                }
                else
                {
                  rez.serialize(IndexPartition::NO_PART);
                  rez.serialize(handle);
                }
                // References were already added on the source node
                pack_index_space(rez, 0 /*reference count*/);
              }
              for (const AddressSpaceID& child : children)
                if (child != source)
                  rez.dispatch(child);
              if (parent_space != source)
                rez.dispatch(parent_space);
            }
          }
          else if (!is_owner() && (source == local_space))
          {
            IndexSpaceSet rez;
            {
              RezCheck z(rez);
              if (parent != nullptr)
              {
                rez.serialize(parent->handle);
                rez.serialize(color);
              }
              else
              {
                rez.serialize(IndexPartition::NO_PART);
                rez.serialize(handle);
              }
              pack_index_space(rez, 1 /*reference count*/);
            }
            if (collective_mapping != nullptr)
              rez.dispatch(collective_mapping->find_nearest(local_space));
            else
              rez.dispatch(owner_space);
          }
        }
        if (has_remote_instances())
        {
          // We're the owner, send messages to everyone else that we've
          // sent this node to except the source
          IndexSpaceSet rez;
          {
            RezCheck z(rez);
            if (parent != nullptr)
            {
              rez.serialize(parent->handle);
              rez.serialize(color);
            }
            else
            {
              rez.serialize(IndexPartition::NO_PART);
              rez.serialize(handle);
            }
            pack_index_space(rez, count_remote_instances());
          }
          IndexSpaceSetFunctor functor(source, rez);
          map_over_remote_instances(functor);
        }
      }
      // Now we can tighten it
      tighten_index_space();
      if (is_owner() || ((collective_mapping != nullptr) &&
                         collective_mapping->contains(local_space)))
      {
        if (parent != nullptr)
          parent->set_child(this);
      }
      // Remove the reference we were holding until this was set
      if (initializing)
        return false;
      else if (parent == nullptr)
        return remove_base_gc_ref(REGION_TREE_REF);
      if (parent->remove_base_gc_ref(REGION_TREE_REF))
        delete parent;
      return false;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    RtEvent IndexSpaceNodeT<DIM, T>::get_realm_index_space_ready(
        bool need_tight)
    //--------------------------------------------------------------------------
    {
      if (index_space_tight.load())
        return RtEvent::NO_RT_EVENT;
      if (!need_tight && index_space_set.load())
        return RtEvent::NO_RT_EVENT;
      AutoLock n_lock(node_lock);
      // See if we lost the race
      if (index_space_tight.load())
        return RtEvent::NO_RT_EVENT;
      if (!need_tight && index_space_set.load())
        return RtEvent::NO_RT_EVENT;
      if (!index_space_ready.exists())
        index_space_ready = Runtime::create_rt_user_event();
      return index_space_ready;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM, T>::set_domain(
        const Domain& domain, ApEvent ready, bool take_ownership,
        bool broadcast, bool initializing)
    //--------------------------------------------------------------------------
    {
      legion_assert(domain.exists());
      DomainT<DIM, T> realm_space = domain;
      if (!take_ownership && !realm_space.dense())
      {
        ApEvent added(realm_space.sparsity.add_reference());
        if (added.exists())
        {
          if (ready.exists())
            ready = Runtime::merge_events(nullptr, ready, added);
          else
            ready = added;
        }
      }
      return set_realm_index_space(
          realm_space, ready, initializing, broadcast, runtime->address_space);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM, T>::set_output_union(
        const std::map<DomainPoint, DomainPoint>& output_sizes)
    //--------------------------------------------------------------------------
    {
      std::vector<Realm::Rect<DIM, T> > output_rects;
      output_rects.reserve(output_sizes.size());
      for (const std::pair<const DomainPoint, DomainPoint>& it : output_sizes)
      {
        legion_assert((it.first.get_dim() + it.second.dim) == DIM);
        int launch_ndim = DIM - it.second.dim;
        Point<DIM, T> lo, hi;
        for (int idx = 0; idx < launch_ndim; idx++)
        {
          lo[idx] = it.first[idx];
          hi[idx] = it.first[idx];
        }
        for (int idx = launch_ndim; idx < DIM; idx++)
        {
          lo[idx] = 0;
          hi[idx] = it.second[idx - launch_ndim] - 1;
        }
        output_rects.emplace_back(Realm::Rect<DIM, T>(lo, hi));
      }
      const Realm::IndexSpace<DIM, T> output_space(output_rects);
      return set_realm_index_space(
          output_space, ApEvent::NO_AP_EVENT, false /*init*/,
          false /*broadcast*/, runtime->address_space);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::tighten_index_space(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(index_space_set.load());
      legion_assert(!index_space_tight.load());
      const RtEvent valid_event(realm_index_space.make_valid());
      if (implicit_profiler != nullptr)
        implicit_profiler->record_make_valid(valid_event, did);
      if (!valid_event.has_triggered() || index_space_valid.exists())
      {
        // If this index space isn't ready yet, then we have to defer this
        if (!valid_event.has_triggered())
        {
          TightenIndexSpaceArgs args(this, this);
          if (index_space_valid.exists())
            runtime->issue_runtime_meta_task(
                args, LG_LATENCY_WORK_PRIORITY,
                Runtime::merge_events(
                    valid_event, Runtime::protect_event(index_space_valid)));
          else
            runtime->issue_runtime_meta_task(
                args, LG_LATENCY_WORK_PRIORITY, valid_event);
          return;
        }
        else
        {
          const RtEvent safe = Runtime::protect_event(index_space_valid);
          if (safe.exists() && !safe.has_triggered())
          {
            TightenIndexSpaceArgs args(this, this);
            runtime->issue_runtime_meta_task(
                args, LG_LATENCY_WORK_PRIORITY, safe);
            return;
          }
        }
      }
      legion_assert(realm_index_space.is_valid());
      const Realm::IndexSpace<DIM, T> tight_space = realm_index_space.tighten();
      legion_assert(tight_space.is_valid());
      Realm::IndexSpace<DIM, T> old_space;
      // Now take the lock and set everything
      {
        AutoLock n_lock(node_lock);
        old_space = realm_index_space;
        realm_index_space = tight_space;
        index_space_tight.store(true);
        if (index_space_ready.exists())
        {
          Runtime::trigger_event(index_space_ready);
          index_space_ready = RtUserEvent::NO_RT_USER_EVENT;
        }
      }
      // If the old space was not dense and the tightened one is dense
      // that means we no longer need to hold our sparsity map reference
      if (!old_space.dense() && tight_space.dense())
      {
        // We can access the index space users without the lock because
        // we know that no more users will be enqueued with a dense
        // realm index space stored
        std::vector<ApEvent> preconditions;
        while (!index_space_users.empty())
        {
          if (!index_space_users.front().has_triggered_faultignorant())
            preconditions.emplace_back(index_space_users.front());
          index_space_users.pop_front();
        }
        if (!preconditions.empty())
        {
          if (!index_space_valid.has_triggered_faultignorant())
            preconditions.emplace_back(index_space_valid);
          index_space_valid = Runtime::merge_events(nullptr, preconditions);
          // Protect it if necessary to make sure the deletion happens
          // even if one of the users was poisoned
          if (index_space_valid.exists())
            index_space_valid =
                ApEvent(Runtime::protect_event(index_space_valid));
        }
        old_space.destroy(index_space_valid);
      }
      if ((spy_logging_level > NO_SPY_LOGGING) ||
          (runtime->profiler != nullptr))
      {
        // Log subspaces being set on the owner
        const AddressSpaceID owner_space = get_owner_space();
        if (owner_space == runtime->address_space)
        {
          this->log_index_space_points(tight_space);
          if ((runtime->profiler != nullptr) && !profiler_logged.exchange(true))
            this->log_profiler_points(did, tight_space);
        }
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    DistributedID IndexSpaceNodeT<DIM, T>::record_profiler_expression(void)
    //--------------------------------------------------------------------------
    {
      if (!profiler_logged.exchange(true))
      {
        DomainT<DIM, T> tight_space = get_tight_index_space();
        log_profiler_points(did, tight_space);
      }
      return did;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM, T>::check_empty(void)
    //--------------------------------------------------------------------------
    {
      return (get_volume() == 0);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceNode* IndexSpaceNodeT<DIM, T>::create_node(
        IndexSpace new_handle, RtEvent initialized, Provenance* provenance,
        CollectiveMapping* collective_mapping, IndexSpaceExprID new_expr_id)
    //--------------------------------------------------------------------------
    {
      if (new_expr_id == 0)
        new_expr_id = expr_id;
      legion_assert(handle.get_type_tag() == new_handle.get_type_tag());
      ApUserEvent to_trigger;
      DomainT<DIM, T> local_space;
      const ApEvent ready = get_loose_index_space(local_space, to_trigger);
      IndexSpaceNode* result = runtime->create_node(
          new_handle, Domain(local_space), false /*take ownership*/,
          nullptr /*parent*/, 0 /*color*/, initialized, provenance, ready,
          new_expr_id, collective_mapping, true /*add root reference*/);
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger);
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceNodeT<DIM, T>::create_from_rectangles(
        const local::set<Domain>& rects)
    //--------------------------------------------------------------------------
    {
      return create_from_rectangles_internal<DIM, T>(rects);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    PieceIteratorImpl* IndexSpaceNodeT<DIM, T>::create_piece_iterator(
        const void* piece_list, size_t piece_list_size,
        IndexSpaceNode* priv_node)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNodeT<DIM, T>* privilege_node =
          legion_safe_cast<IndexSpaceNodeT<DIM, T>*>(priv_node);
      if (piece_list == nullptr)
      {
        DomainT<DIM, T> realm_space = get_tight_index_space();
        // If there was no piece list it has to be because there
        // was just one piece which was a single dense rectangle
        legion_assert(realm_space.dense());
        return new PieceIteratorImplT<DIM, T>(
            &realm_space.bounds, sizeof(realm_space.bounds), privilege_node);
      }
      else
        return new PieceIteratorImplT<DIM, T>(
            piece_list, piece_list_size, privilege_node);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::log_index_space_points(
        const Realm::IndexSpace<DIM, T>& tight_space) const
    //--------------------------------------------------------------------------
    {
      if (spy_logging_level == NO_SPY_LOGGING)
        return;
      // Be careful, Realm can lie to us here
      if (!tight_space.empty())
      {
        bool logged = false;
        // Iterate over the rectangles and print them out
        for (Realm::IndexSpaceIterator<DIM, T> itr(tight_space); itr.valid;
             itr.step())
        {
          const size_t rect_volume = itr.rect.volume();
          if (rect_volume == 0)
            continue;
          logged = true;
          if (rect_volume == 1)
            LegionSpy::log_index_space_point(
                handle.get_id(), Point<DIM, T>(itr.rect.lo));
          else
            LegionSpy::log_index_space_rect(
                handle.get_id(), Rect<DIM, T>(itr.rect));
        }
        // Handle the case where Realm lied to us about being empty
        if (!logged)
          LegionSpy::log_empty_index_space(handle.get_id());
      }
      else
        LegionSpy::log_empty_index_space(handle.get_id());
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::compute_pending_space(
        Operation* op, const std::vector<IndexSpace>& handles, bool is_union,
        bool broadcast)
    //--------------------------------------------------------------------------
    {
      ApUserEvent to_trigger;
      std::set<ApEvent> preconditions;
      std::vector<Realm::IndexSpace<DIM, T> > spaces(handles.size());
      for (unsigned idx = 0; idx < handles.size(); idx++)
      {
        if (handles[idx].get_type_tag() != handle.get_type_tag())
        {
          TaskContext* ctx = op->get_context();
          if (is_union)
          {
            Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
            error << "Dynamic type mismatch in 'create_index_space_union' "
                  << "performed in task " << *ctx << ".";
            error.raise();
          }
          else
          {
            Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
            error << "Dynamic type mismatch in "
                  << "'create_index_space_intersection' performed in task "
                  << *ctx << ".";
            error.raise();
          }
        }
        IndexSpaceNodeT<DIM, T>* space = static_cast<IndexSpaceNodeT<DIM, T>*>(
            runtime->get_node(handles[idx]));
        ApEvent ready = space->get_loose_index_space(spaces[idx], to_trigger);
        if (ready.exists())
          preconditions.insert(ready);
      }
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      // Kick this off to Realm
      ApEvent precondition = Runtime::merge_events(nullptr, preconditions);
      ApEvent result;
      Realm::IndexSpace<DIM, T> result_space;
      if (is_union)
      {
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != nullptr)
          runtime->profiler->add_partition_request(
              requests, op, DEP_PART_UNION_REDUCTION, precondition);
        result = ApEvent(Realm::IndexSpace<DIM, T>::compute_union(
            spaces, result_space, requests, precondition));
      }
      else
      {
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != nullptr)
          runtime->profiler->add_partition_request(
              requests, op, DEP_PART_INTERSECTION_REDUCTION, precondition);
        result = ApEvent(Realm::IndexSpace<DIM, T>::compute_intersection(
            spaces, result_space, requests, precondition));
      }
      if (set_realm_index_space(
              result_space, result, false /*init*/, broadcast, local_space))
        std::abort();  // should never hit this
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger, result);
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::compute_pending_space(
        Operation* op, IndexPartition part_handle, bool is_union,
        bool broadcast)
    //--------------------------------------------------------------------------
    {
      if (part_handle.get_type_tag() != handle.get_type_tag())
      {
        TaskContext* ctx = op->get_context();
        if (is_union)
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Dynamic type mismatch in 'create_index_space_union' "
                << "performed in task " << *ctx << ".";
          error.raise();
        }
        else
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Dynamic type mismatch in 'create_index_space_intersection' "
                << "performed in task " << *ctx << ".";
          error.raise();
        }
      }
      IndexPartNode* partition = runtime->get_node(part_handle);
      ApUserEvent to_trigger;
      std::set<ApEvent> preconditions;
      std::vector<DomainT<DIM, T> > spaces(partition->total_children);
      unsigned subspace_index = 0;
      for (ColorSpaceIterator itr(partition); itr; itr++)
      {
        IndexSpaceNodeT<DIM, T>* child =
            static_cast<IndexSpaceNodeT<DIM, T>*>(partition->get_child(*itr));
        ApEvent ready =
            child->get_loose_index_space(spaces[subspace_index++], to_trigger);
        if (ready.exists())
          preconditions.insert(ready);
      }
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      // Kick this off to Realm
      ApEvent precondition = Runtime::merge_events(nullptr, preconditions);
      ApEvent result;
      Realm::IndexSpace<DIM, T> result_space;
      if (is_union)
      {
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != nullptr)
          runtime->profiler->add_partition_request(
              requests, op, DEP_PART_UNION_REDUCTION, precondition);
        result = ApEvent(Realm::IndexSpace<DIM, T>::compute_union(
            spaces, result_space, requests, precondition));
      }
      else
      {
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != nullptr)
          runtime->profiler->add_partition_request(
              requests, op, DEP_PART_INTERSECTION_REDUCTION, precondition);
        result = ApEvent(Realm::IndexSpace<DIM, T>::compute_intersection(
            spaces, result_space, requests, precondition));
      }
      if (set_realm_index_space(
              result_space, result, false /*init*/, broadcast, local_space))
        std::abort();  // should never hit this
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger, result);
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::compute_pending_difference(
        Operation* op, IndexSpace init, const std::vector<IndexSpace>& handles,
        bool broadcast)
    //--------------------------------------------------------------------------
    {
      if (init.get_type_tag() != handle.get_type_tag())
      {
        TaskContext* ctx = op->get_context();
        Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
        error << "Dynamic type mismatch in 'create_index_space_difference' "
              << "performed in task " << *ctx << ".";
        error.raise();
      }
      ApUserEvent to_trigger;
      std::set<ApEvent> preconditions;
      std::vector<DomainT<DIM, T> > spaces(handles.size());
      for (unsigned idx = 0; idx < handles.size(); idx++)
      {
        if (handles[idx].get_type_tag() != handle.get_type_tag())
        {
          TaskContext* ctx = op->get_context();
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Dynamic type mismatch in 'create_index_space_difference' "
                << "performed in task " << *ctx << ".";
          error.raise();
        }
        IndexSpaceNodeT<DIM, T>* space = static_cast<IndexSpaceNodeT<DIM, T>*>(
            runtime->get_node(handles[idx]));
        ApEvent ready = space->get_loose_index_space(spaces[idx], to_trigger);
        if (ready.exists())
          preconditions.insert(ready);
      }
      if (op->has_execution_fence_event())
        preconditions.insert(op->get_execution_fence_event());
      ApEvent precondition = Runtime::merge_events(nullptr, preconditions);
      Realm::ProfilingRequestSet union_requests;
      Realm::ProfilingRequestSet diff_requests;
      if (runtime->profiler != nullptr)
        runtime->profiler->add_partition_request(
            union_requests, op, DEP_PART_UNION_REDUCTION, precondition);
      // Compute the union of the handles for the right-hand side
      Realm::IndexSpace<DIM, T> rhs_space;
      ApEvent rhs_ready(Realm::IndexSpace<DIM, T>::compute_union(
          spaces, rhs_space, union_requests, precondition));
      IndexSpaceNodeT<DIM, T>* lhs_node =
          static_cast<IndexSpaceNodeT<DIM, T>*>(runtime->get_node(init));
      Realm::IndexSpace<DIM, T> lhs_space, result_space;
      ApEvent lhs_ready =
          lhs_node->get_loose_index_space(lhs_space, to_trigger);
      ApEvent diff_pre = Runtime::merge_events(nullptr, lhs_ready, rhs_ready);
      if (runtime->profiler != nullptr)
        runtime->profiler->add_partition_request(
            diff_requests, op, DEP_PART_DIFFERENCE, diff_pre);
      ApEvent result(Realm::IndexSpace<DIM, T>::compute_difference(
          lhs_space, rhs_space, result_space, diff_requests, diff_pre));
      if (set_realm_index_space(
              result_space, result, false /*init*/, broadcast, local_space))
        std::abort();  // should never hit this
      // Destroy the tempory rhs space once the computation is done
      rhs_space.destroy(result);
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger, result);
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::get_index_space_domain(
        void* realm_is, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      if (type_tag == handle.get_type_tag())
      {
        Realm::IndexSpace<DIM, T>* target =
            static_cast<Realm::IndexSpace<DIM, T>*>(realm_is);
        // No need to wait since we're waiting for it to be tight
        // which implies that it will be ready
        *target = get_tight_index_space();
      }
      else
      {
        DomainT<DIM, T> target = get_tight_index_space();
        const Domain domain(target);
        RealmSpaceConverter<DIM, Realm::DIMTYPES>::convert_to(
            domain, realm_is, type_tag, "get_index_space_domain");
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t IndexSpaceNodeT<DIM, T>::get_volume(void)
    //--------------------------------------------------------------------------
    {
      if (has_volume.load())
        return volume;
      DomainT<DIM, T> volume_space = get_tight_index_space();
      volume = volume_space.volume();
      has_volume.store(true);
      return volume;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t IndexSpaceNodeT<DIM, T>::get_num_dims(void) const
    //--------------------------------------------------------------------------
    {
      return DIM;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM, T>::contains_point(
        const void* realm_point, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> test_space = get_tight_index_space();
      if (type_tag == handle.get_type_tag())
      {
        const Realm::Point<DIM, T>* point =
            static_cast<const Realm::Point<DIM, T>*>(realm_point);
        return test_space.contains(*point);
      }
      else
      {
        DomainPoint point;
        RealmPointConverter<DIM, Realm::DIMTYPES>::convert_from(
            realm_point, type_tag, point, "safe_cast");
        return test_space.contains(Point<DIM, T>(point));
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM, T>::contains_point(const DomainPoint& point)
    //--------------------------------------------------------------------------
    {
      const Point<DIM, T> p = point;
      return contains_point(p);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM, T>::contains_point(const Point<DIM, T>& p)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> test_space = get_tight_index_space();
      return test_space.contains(p);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor IndexSpaceNodeT<DIM, T>::get_max_linearized_color(void)
    //--------------------------------------------------------------------------
    {
      ColorSpaceLinearizationT<DIM, T>* linear = linearization.load();
      if (linear == nullptr)
        linear = compute_linearization_metadata();
      return linear->get_max_linearized_color();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM, T>::has_interfering_point(
        const std::vector<std::pair<DomainPoint, Domain> >& test_points,
        DomainPoint& interfering_point, DomainPoint to_skip)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> space = get_tight_index_space();
      for (const std::pair<DomainPoint, Domain>& it : test_points)
      {
        if (it.first == to_skip)
          continue;
        DomainT<DIM, T> other = it.second;
        // Check the bounds first to see if we can skip the test just
        // based on those alone without loading the sparsity map
        if (!space.bounds.overlaps(other.bounds))
          continue;
        if (!other.dense())
        {
          // Load its sparsity map on this node so we can test it
          RtEvent ready(other.make_valid());
          if (implicit_profiler != nullptr)
            implicit_profiler->record_make_valid(ready);
          ready.wait();
        }
        if (space.overlaps(other))
        {
          interfering_point = it.first;
          return true;
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ColorSpaceLinearizationT<DIM, T>*
        IndexSpaceNodeT<DIM, T>::compute_linearization_metadata(void)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> space = get_tight_index_space();
      ColorSpaceLinearizationT<DIM, T>* result =
          new ColorSpaceLinearizationT<DIM, T>(space);
      ColorSpaceLinearizationT<DIM, T>* expected = nullptr;
      if (!linearization.compare_exchange_strong(expected, result))
      {
        delete result;
        legion_assert(expected != nullptr);
        result = expected;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor IndexSpaceNodeT<DIM, T>::linearize_color(const DomainPoint& p)
    //--------------------------------------------------------------------------
    {
      const Point<DIM, T> point = p;
      return linearize_color(point);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor IndexSpaceNodeT<DIM, T>::linearize_color(
        const void* realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      ColorSpaceLinearizationT<DIM, T>* linear = linearization.load();
      if (linear == nullptr)
        linear = compute_linearization_metadata();
      Point<DIM, T> point;
      if (type_tag != handle.get_type_tag())
      {
        DomainPoint dp;
        RealmPointConverter<DIM, Realm::DIMTYPES>::convert_from(
            realm_color, type_tag, dp, "linearize_color");
        point = dp;
      }
      else
        point = *(static_cast<const Point<DIM, T>*>(realm_color));
      return linear->linearize(point);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor IndexSpaceNodeT<DIM, T>::linearize_color(
        const Point<DIM, T>& point)
    //--------------------------------------------------------------------------
    {
      ColorSpaceLinearizationT<DIM, T>* linear = linearization.load();
      if (linear == nullptr)
        linear = compute_linearization_metadata();
      return linear->linearize(point);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::delinearize_color(
        LegionColor color, Point<DIM, T>& point)
    //--------------------------------------------------------------------------
    {
      ColorSpaceLinearizationT<DIM, T>* linear = linearization.load();
      if (linear == nullptr)
        linear = compute_linearization_metadata();
      linear->delinearize(color, point);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::delinearize_color(
        LegionColor color, void* realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      ColorSpaceLinearizationT<DIM, T>* linear = linearization.load();
      if (linear == nullptr)
        linear = compute_linearization_metadata();
      Point<DIM, T> point;
      linear->delinearize(color, point);
      if (type_tag != handle.get_type_tag())
        RealmPointConverter<DIM, Realm::DIMTYPES>::convert_to(
            DomainPoint(point), realm_color, type_tag, "delinearize_color");
      else
        *(static_cast<Point<DIM, T>*>(realm_color)) = point;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t IndexSpaceNodeT<DIM, T>::compute_color_offset(LegionColor color)
    //--------------------------------------------------------------------------
    {
      ColorSpaceLinearizationT<DIM, T>* linear = linearization.load();
      if (linear == nullptr)
        linear = compute_linearization_metadata();
      return linear->compute_color_offset(color);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM, T>::contains_color(
        LegionColor color, bool report_error /*=false*/)
    //--------------------------------------------------------------------------
    {
      ColorSpaceLinearizationT<DIM, T>* linear = linearization.load();
      if (linear == nullptr)
        linear = compute_linearization_metadata();
      const bool result = linear->contains_color(color);
      if (!result && report_error)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid color request.";
        error.raise();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::instantiate_colors(
        std::vector<LegionColor>& colors)
    //--------------------------------------------------------------------------
    {
      colors.resize(get_volume());
      unsigned idx = 0;
      DomainT<DIM, T> space = get_tight_index_space();
      for (Realm::IndexSpaceIterator<DIM, T> rect_itr(space); rect_itr.valid;
           rect_itr.step())
      {
        for (Realm::PointInRectIterator<DIM, T> itr(rect_itr.rect); itr.valid;
             itr.step(), idx++)
          colors[idx] = linearize_color(&itr.p, handle.get_type_tag());
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    Domain IndexSpaceNodeT<DIM, T>::get_color_space_domain(void)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> space = get_tight_index_space();
      return Domain(DomainT<DIM, T>(space));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    DomainPoint IndexSpaceNodeT<DIM, T>::get_domain_point_color(void) const
    //--------------------------------------------------------------------------
    {
      if (parent == nullptr)
        return DomainPoint(color);
      return parent->color_space->delinearize_color_to_point(color);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    DomainPoint IndexSpaceNodeT<DIM, T>::delinearize_color_to_point(
        LegionColor c)
    //--------------------------------------------------------------------------
    {
      Point<DIM, T> color_point;
      delinearize_color(c, color_point);
      return DomainPoint(Point<DIM, T>(color_point));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::pack_index_space(
        Serializer& rez, unsigned references) const
    //--------------------------------------------------------------------------
    {
      legion_assert(index_space_set.load());
      // No need for the lock, held by the caller
      rez.serialize(realm_index_space);
      rez.serialize(index_space_valid);
      if (!realm_index_space.dense())
      {
        if (references > 0)
        {
          Realm::SparsityMap<DIM, T> copy = realm_index_space.sparsity;
          const ApEvent added(copy.add_reference(references));
          rez.serialize(added);
        }
        else
          rez.serialize(ApEvent::NO_AP_EVENT);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM, T>::unpack_index_space(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM, T> result_space;
      derez.deserialize(result_space);
      ApEvent valid_event;
      derez.deserialize(valid_event);
      if (!result_space.dense())
      {
        ApEvent added;
        derez.deserialize(added);
        if (added.exists())
        {
          added.subscribe();
          index_space_users.emplace_back(added);
        }
      }
      return set_realm_index_space(
          result_space, valid_event, false /*initialization*/,
          true /*broadcast*/, source);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_equal_children(
        Operation* op, IndexPartNode* partition, size_t granularity)
    //--------------------------------------------------------------------------
    {
      legion_assert(partition->parent == this);
      const size_t count = partition->total_children;
      if (partition->is_owner() && (partition->collective_mapping == nullptr))
      {
        // Common case is not control replication
        std::vector<DomainT<DIM, T> > subspaces;
        ApUserEvent to_trigger;
        DomainT<DIM, T> local_space;
        ApEvent ready = get_loose_index_space(local_space, to_trigger);
        if (op->has_execution_fence_event())
          ready = Runtime::merge_events(
              nullptr, ready, op->get_execution_fence_event());
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != nullptr)
          runtime->profiler->add_partition_request(
              requests, op, DEP_PART_EQUAL, ready);
        ApEvent result(local_space.create_equal_subspaces(
            count, granularity, subspaces, requests, ready));
        if ((spy_logging_level > LIGHT_SPY_LOGGING) &&
            (!result.exists() || (result == ready)))
          Runtime::rename_event(result);
        if (to_trigger.exists())
          Runtime::trigger_event_untraced(to_trigger, result);
        LegionSpy::log_deppart_events(
            op->get_unique_op_id(), expr_id, ready, result, DEP_PART_EQUAL);
        // Enumerate the colors and assign the spaces
        unsigned subspace_index = 0;
        for (ColorSpaceIterator itr(partition); itr; itr++)
        {
          IndexSpaceNodeT<DIM, T>* child =
              static_cast<IndexSpaceNodeT<DIM, T>*>(partition->get_child(*itr));
          legion_assert(subspace_index < subspaces.size());
          if (child->set_realm_index_space(subspaces[subspace_index++], result))
            delete child;
        }
        return result;
      }
      else
      {
        const size_t count = partition->total_children;
        std::set<ApEvent> done_events;
        ApUserEvent to_trigger;
        DomainT<DIM, T> local_space;
        const ApEvent local_ready =
            get_loose_index_space(local_space, to_trigger);
        // In the case of control replication we do things
        // one point at a time for the subspaces owned by this shard
        size_t color_offset = SIZE_MAX;
        for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
        {
          if (color_offset == SIZE_MAX)
            color_offset = partition->color_space->compute_color_offset(*itr);
          else
            color_offset++;
          Realm::ProfilingRequestSet requests;
          if (runtime->profiler != nullptr)
            runtime->profiler->add_partition_request(
                requests, op, DEP_PART_EQUAL, local_ready);
          DomainT<DIM, T> subspace;
          ApEvent result(local_space.create_equal_subspace(
              count, granularity, color_offset, subspace, requests,
              local_ready));
          IndexSpaceNodeT<DIM, T>* child =
              static_cast<IndexSpaceNodeT<DIM, T>*>(partition->get_child(*itr));
          if (child->set_realm_index_space(subspace, result))
            delete child;
          done_events.insert(result);
        }
        ApEvent result;
        if (!done_events.empty())
          result = Runtime::merge_events(nullptr, done_events);
        if (to_trigger.exists())
          Runtime::trigger_event_untraced(to_trigger, result);
        return result;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_by_union(
        Operation* op, IndexPartNode* partition, IndexPartNode* left,
        IndexPartNode* right)
    //--------------------------------------------------------------------------
    {
      legion_assert(partition->parent == this);
      ApUserEvent to_trigger;
      std::vector<DomainT<DIM, T> > lhs_spaces, rhs_spaces;
      std::vector<ApEvent> preconditions;
      // First we need to fill in all the subspaces
      for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM, T>* left_child =
            static_cast<IndexSpaceNodeT<DIM, T>*>(left->get_child(*itr));
        IndexSpaceNodeT<DIM, T>* right_child =
            static_cast<IndexSpaceNodeT<DIM, T>*>(right->get_child(*itr));
        lhs_spaces.resize(lhs_spaces.size() + 1);
        rhs_spaces.resize(rhs_spaces.size() + 1);
        ApEvent left_ready =
            left_child->get_loose_index_space(lhs_spaces.back(), to_trigger);
        ApEvent right_ready =
            right_child->get_loose_index_space(rhs_spaces.back(), to_trigger);
        if (left_ready.exists())
          preconditions.emplace_back(left_ready);
        if (right_ready.exists())
          preconditions.emplace_back(right_ready);
      }
      if (lhs_spaces.empty())
        return ApEvent::NO_AP_EVENT;
      std::vector<Realm::IndexSpace<DIM, T> > subspaces;
      if (op->has_execution_fence_event())
        preconditions.emplace_back(op->get_execution_fence_event());
      const ApEvent precondition =
          Runtime::merge_events(nullptr, preconditions);
      Realm::ProfilingRequestSet requests;
      if (runtime->profiler != nullptr)
        runtime->profiler->add_partition_request(
            requests, op, DEP_PART_UNIONS, precondition);
      ApEvent result(Realm::IndexSpace<DIM, T>::compute_unions(
          lhs_spaces, rhs_spaces, subspaces, requests, precondition));
      if ((spy_logging_level > LIGHT_SPY_LOGGING) &&
          (!result.exists() || (result == precondition)))
        Runtime::rename_event(result);
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger, result);
      LegionSpy::log_deppart_events(
          op->get_unique_op_id(), expr_id, precondition, result,
          DEP_PART_UNIONS);
      // Now set the index spaces for the results
      unsigned subspace_index = 0;
      for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM, T>* child =
            static_cast<IndexSpaceNodeT<DIM, T>*>(partition->get_child(*itr));
        legion_assert(subspace_index < subspaces.size());
        if (child->set_realm_index_space(subspaces[subspace_index++], result))
          delete child;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_by_intersection(
        Operation* op, IndexPartNode* partition, IndexPartNode* left,
        IndexPartNode* right)
    //--------------------------------------------------------------------------
    {
      legion_assert(partition->parent == this);
      ApUserEvent to_trigger;
      std::vector<DomainT<DIM, T> > lhs_spaces, rhs_spaces;
      std::vector<ApEvent> preconditions;
      for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM, T>* left_child =
            static_cast<IndexSpaceNodeT<DIM, T>*>(left->get_child(*itr));
        IndexSpaceNodeT<DIM, T>* right_child =
            static_cast<IndexSpaceNodeT<DIM, T>*>(right->get_child(*itr));
        lhs_spaces.resize(lhs_spaces.size() + 1);
        rhs_spaces.resize(rhs_spaces.size() + 1);
        ApEvent left_ready =
            left_child->get_loose_index_space(lhs_spaces.back(), to_trigger);
        ApEvent right_ready =
            right_child->get_loose_index_space(rhs_spaces.back(), to_trigger);
        if (left_ready.exists())
          preconditions.emplace_back(left_ready);
        if (right_ready.exists())
          preconditions.emplace_back(right_ready);
      }
      if (lhs_spaces.empty())
        return ApEvent::NO_AP_EVENT;
      std::vector<DomainT<DIM, T> > subspaces;
      if (op->has_execution_fence_event())
        preconditions.emplace_back(op->get_execution_fence_event());
      const ApEvent precondition =
          Runtime::merge_events(nullptr, preconditions);
      Realm::ProfilingRequestSet requests;
      if (runtime->profiler != nullptr)
        runtime->profiler->add_partition_request(
            requests, op, DEP_PART_INTERSECTIONS, precondition);
      ApEvent result(Realm::IndexSpace<DIM, T>::compute_intersections(
          lhs_spaces, rhs_spaces, subspaces, requests, precondition));
      if ((spy_logging_level > LIGHT_SPY_LOGGING) &&
          (!result.exists() || (result == precondition)))
        Runtime::rename_event(result);
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger, result);
      LegionSpy::log_deppart_events(
          op->get_unique_op_id(), expr_id, precondition, result,
          DEP_PART_INTERSECTIONS);
      // Now set the index spaces for the results
      unsigned subspace_index = 0;
      for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM, T>* child =
            static_cast<IndexSpaceNodeT<DIM, T>*>(partition->get_child(*itr));
        legion_assert(subspace_index < subspaces.size());
        if (child->set_realm_index_space(subspaces[subspace_index++], result))
          delete child;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_by_intersection(
        Operation* op, IndexPartNode* partition,
        // Left is implicit "this"
        IndexPartNode* right, const bool dominates)
    //--------------------------------------------------------------------------
    {
      legion_assert(partition->parent == this);
      ApUserEvent to_trigger;
      std::vector<DomainT<DIM, T> > rhs_spaces;
      std::vector<ApEvent> preconditions;
      // First we need to fill in all the subspaces
      for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM, T>* right_child =
            static_cast<IndexSpaceNodeT<DIM, T>*>(right->get_child(*itr));
        rhs_spaces.resize(rhs_spaces.size() + 1);
        ApEvent right_ready =
            right_child->get_loose_index_space(rhs_spaces.back(), to_trigger);
        if (right_ready.exists())
          preconditions.emplace_back(right_ready);
      }
      if (rhs_spaces.empty())
        return ApEvent::NO_AP_EVENT;
      ApEvent result, precondition;
      std::vector<Realm::IndexSpace<DIM, T> > subspaces;
      if (dominates)
      {
        // If we've been told that we dominate then there is no
        // need to event do the intersection tests at all
        subspaces.swap(rhs_spaces);
        result = Runtime::merge_events(nullptr, preconditions);
      }
      else
      {
        DomainT<DIM, T> lhs_space;
        ApEvent left_ready = get_loose_index_space(lhs_space, to_trigger);
        if (left_ready.exists())
          preconditions.emplace_back(left_ready);
        if (op->has_execution_fence_event())
          preconditions.emplace_back(op->get_execution_fence_event());
        precondition = Runtime::merge_events(nullptr, preconditions);
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != nullptr)
          runtime->profiler->add_partition_request(
              requests, op, DEP_PART_INTERSECTIONS, precondition);
        result = ApEvent(Realm::IndexSpace<DIM, T>::compute_intersections(
            lhs_space, rhs_spaces, subspaces, requests, precondition));
      }
      if ((spy_logging_level > LIGHT_SPY_LOGGING) &&
          (!result.exists() || (result == precondition)))
        Runtime::rename_event(result);
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger, result);
      LegionSpy::log_deppart_events(
          op->get_unique_op_id(), expr_id, precondition, result,
          DEP_PART_INTERSECTIONS);
      // Now set the index spaces for the results
      unsigned subspace_index = 0;
      for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM, T>* child =
            static_cast<IndexSpaceNodeT<DIM, T>*>(partition->get_child(*itr));
        legion_assert(subspace_index < subspaces.size());
        if (child->set_realm_index_space(subspaces[subspace_index++], result))
          delete child;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_by_difference(
        Operation* op, IndexPartNode* partition, IndexPartNode* left,
        IndexPartNode* right)
    //--------------------------------------------------------------------------
    {
      legion_assert(partition->parent == this);
      ApUserEvent to_trigger;
      std::vector<DomainT<DIM, T> > lhs_spaces, rhs_spaces;
      std::vector<ApEvent> preconditions;
      // First we need to fill in all the subspaces
      for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM, T>* left_child =
            static_cast<IndexSpaceNodeT<DIM, T>*>(left->get_child(*itr));
        IndexSpaceNodeT<DIM, T>* right_child =
            static_cast<IndexSpaceNodeT<DIM, T>*>(right->get_child(*itr));
        lhs_spaces.resize(lhs_spaces.size() + 1);
        rhs_spaces.resize(rhs_spaces.size() + 1);
        ApEvent left_ready =
            left_child->get_loose_index_space(lhs_spaces.back(), to_trigger);
        ApEvent right_ready =
            right_child->get_loose_index_space(rhs_spaces.back(), to_trigger);
        if (left_ready.exists())
          preconditions.emplace_back(left_ready);
        if (right_ready.exists())
          preconditions.emplace_back(right_ready);
      }
      if (lhs_spaces.empty())
        return ApEvent::NO_AP_EVENT;
      std::vector<Realm::IndexSpace<DIM, T> > subspaces;
      if (op->has_execution_fence_event())
        preconditions.emplace_back(op->get_execution_fence_event());
      const ApEvent precondition =
          Runtime::merge_events(nullptr, preconditions);
      Realm::ProfilingRequestSet requests;
      if (runtime->profiler != nullptr)
        runtime->profiler->add_partition_request(
            requests, op, DEP_PART_DIFFERENCES, precondition);
      ApEvent result(Realm::IndexSpace<DIM, T>::compute_differences(
          lhs_spaces, rhs_spaces, subspaces, requests, precondition));
      if ((spy_logging_level > LIGHT_SPY_LOGGING) &&
          (!result.exists() || (result == precondition)))
        Runtime::rename_event(result);
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger, result);
      LegionSpy::log_deppart_events(
          op->get_unique_op_id(), expr_id, precondition, result,
          DEP_PART_DIFFERENCES);
      // Now set the index spaces for the results
      unsigned subspace_index = 0;
      for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM, T>* child =
            static_cast<IndexSpaceNodeT<DIM, T>*>(partition->get_child(*itr));
        legion_assert(subspace_index < subspaces.size());
        if (child->set_realm_index_space(subspaces[subspace_index++], result))
          delete child;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_by_restriction(
        IndexPartNode* partition, const void* tran, const void* ext,
        int partition_dim)
    //--------------------------------------------------------------------------
    {
      // should be called on the color space
      legion_assert(this == partition->color_space);
      switch (partition_dim)
      {
#define DIMFUNC(D1)                                                            \
  case D1:                                                                     \
    {                                                                          \
      const Realm::Matrix<D1, DIM, T>* transform =                             \
          static_cast<const Realm::Matrix<D1, DIM, T>*>(tran);                 \
      const Realm::Rect<D1, T>* extent =                                       \
          static_cast<const Realm::Rect<D1, T>*>(ext);                         \
      return create_by_restriction_helper<D1>(partition, *transform, *extent); \
    }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          std::abort();
      }
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    template<int N, typename T>
    template<int M>
    ApEvent IndexSpaceNodeT<N, T>::create_by_restriction_helper(
        IndexPartNode* partition, const Realm::Matrix<M, N, T>& transform,
        const Realm::Rect<M, T>& extent)
    //--------------------------------------------------------------------------
    {
      // Get the parent index space in case it has a sparsity map
      IndexSpaceNodeT<M, T>* parent =
          static_cast<IndexSpaceNodeT<M, T>*>(partition->parent);
      // No need to wait since we'll just be messing with the bounds
      ApUserEvent to_trigger;
      DomainT<M, T> parent_is;
      ApEvent parent_ready =
          parent->get_loose_index_space(parent_is, to_trigger);
      // Iterate over our points (colors) and fill in the bounds
      for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
      {
        Point<N, T> color;
        delinearize_color(*itr, color);
        // Copy the index space from the parent
        DomainT<M, T> child_is = parent_is;
        // Compute the new bounds and intersect it with the parent bounds
        child_is.bounds =
            parent_is.bounds.intersection(extent + transform * color);
        // Get the appropriate child
        IndexSpaceNodeT<M, T>* child =
            static_cast<IndexSpaceNodeT<M, T>*>(partition->get_child(*itr));
        // Add a reference if the child index space is not dense
        ApEvent ready;
        if (!child_is.dense())
        {
          ready = ApEvent(child_is.sparsity.add_reference());
          if (parent_ready.exists())
          {
            if (ready.exists())
              ready = Runtime::merge_events(nullptr, ready, parent_ready);
            else
              ready = parent_ready;
          }
        }
        else
          ready = parent_ready;
        // Then set the new index space
        if (child->set_realm_index_space(child_is, ready))
          delete child;
      }
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger);
      // Our only precondition is that the parent index space is computed
      return parent_ready;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_by_domain(
        Operation* op, IndexPartNode* partition,
        const std::map<DomainPoint, FutureImpl*>& futures,
        const Domain& future_map_domain, bool perform_intersections)
    //--------------------------------------------------------------------------
    {
      legion_assert(partition->parent == this);
      // Demux the color space type to do the actual operations
      CreateByDomainHelper creator(
          this, partition, op, futures, future_map_domain,
          perform_intersections);
      NT_TemplateHelper::demux<CreateByDomainHelper>(
          partition->color_space->handle.get_type_tag(), &creator);
      return creator.result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_by_weights(
        Operation* op, IndexPartNode* partition,
        const std::map<DomainPoint, FutureImpl*>& weights, size_t granularity)
    //--------------------------------------------------------------------------
    {
      legion_assert(partition->parent == this);
      // Demux the color space type to do the actual operations
      CreateByWeightHelper creator(this, partition, op, weights, granularity);
      NT_TemplateHelper::demux<CreateByWeightHelper>(
          partition->color_space->handle.get_type_tag(), &creator);
      return creator.result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_by_field(
        Operation* op, FieldID fid, IndexPartNode* partition,
        const std::vector<FieldDataDescriptor>& instances,
        std::vector<DeppartResult>* results, ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(partition->parent == this);
      // Demux the color space type to do the actual operations
      CreateByFieldHelper creator(
          this, op, fid, partition, instances, results, instances_ready);
      NT_TemplateHelper::demux<CreateByFieldHelper>(
          partition->color_space->handle.get_type_tag(), &creator);
      return creator.result;
    }
#endif  // defined(DEFINE_NT_TEMPLATES)

#ifdef DEFINE_NTNT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    template<int COLOR_DIM, typename COLOR_T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_by_domain_helper(
        Operation* op, IndexPartNode* partition,
        const std::map<DomainPoint, FutureImpl*>& futures,
        const Domain& future_map_domain, bool perform_intersections)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> result_events;
      DomainT<DIM, T> parent_space;
      ApEvent parent_ready;
      ApUserEvent to_trigger;
      if (perform_intersections)
      {
        parent_ready = get_loose_index_space(parent_space, to_trigger);
        if (op->has_execution_fence_event())
        {
          if (parent_ready.exists())
            parent_ready = Runtime::merge_events(
                nullptr, parent_ready, op->get_execution_fence_event());
          else
            parent_ready = op->get_execution_fence_event();
        }
      }
      for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
      {
        const DomainPoint color =
            partition->color_space->delinearize_color_to_point(*itr);
        ApEvent child_ready;
        DomainT<DIM, T> child_space;
        if (future_map_domain.contains(color))
        {
          std::map<DomainPoint, FutureImpl*>::const_iterator finder =
              futures.find(color);
          legion_assert(finder != futures.end());
          FutureImpl* future = finder->second;
          size_t future_size = 0;
          const Domain* domain = static_cast<const Domain*>(
              future->find_runtime_buffer(op->get_context(), future_size));
          if (future_size != sizeof(Domain))
          {
            Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
            error << "An invalid future size was found in a partition by "
                  << "domain call. All futures must contain Domain objects.";
            error.raise();
          }
          child_space = *domain;
          // Add a reference to the child space and wait for it to be
          // added to ensure that it gets added before the future releases
          // the reference that it is holding
          if (!child_space.dense())
          {
            const RtEvent added(child_space.sparsity.add_reference());
            added.wait();
          }
          if (perform_intersections)
          {
            Realm::ProfilingRequestSet requests;
            if (runtime->profiler != nullptr)
              runtime->profiler->add_partition_request(
                  requests, op, DEP_PART_INTERSECTIONS, parent_ready);
            Realm::IndexSpace<DIM, T> result;
            child_ready =
                ApEvent(Realm::IndexSpace<DIM, T>::compute_intersection(
                    parent_space, child_space, result, requests, parent_ready));
            // Remove the reference that we added on the child space
            if (!child_space.dense())
              child_space.destroy(child_ready);
            // We take ownership of the new result index space
            child_space = result;
            if (child_ready.exists())
              result_events.insert(child_ready);
          }
        }
        else
          child_space = Realm::IndexSpace<DIM, T>::make_empty();
        IndexSpaceNodeT<DIM, T>* child =
            static_cast<IndexSpaceNodeT<DIM, T>*>(partition->get_child(*itr));
        if (child->set_realm_index_space(child_space, child_ready))
          delete child;
      }
      ApEvent result;
      if (!result_events.empty())
        result = Runtime::merge_events(nullptr, result_events);
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger, result);
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    template<int COLOR_DIM, typename COLOR_T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_by_weight_helper(
        Operation* op, IndexPartNode* partition,
        const std::map<DomainPoint, FutureImpl*>& futures, size_t granularity)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNodeT<COLOR_DIM, COLOR_T>* color_space =
          static_cast<IndexSpaceNodeT<COLOR_DIM, COLOR_T>*>(
              partition->color_space);
      // Enumerate the color space
      DomainT<COLOR_DIM, COLOR_T> realm_color_space =
          color_space->get_tight_index_space();
      const size_t count = realm_color_space.volume();
      // Unpack the futures and fill in the weights appropriately
      std::vector<int> weights;
      std::vector<size_t> long_weights;
      std::vector<LegionColor> child_colors(count);
      unsigned color_index = 0;
      // Make all the entries for the color space
      for (Realm::IndexSpaceIterator<COLOR_DIM, COLOR_T> rect_iter(
               realm_color_space);
           rect_iter.valid; rect_iter.step())
      {
        for (Realm::PointInRectIterator<COLOR_DIM, COLOR_T> itr(rect_iter.rect);
             itr.valid; itr.step())
        {
          const DomainPoint key(Point<COLOR_DIM, COLOR_T>(itr.p));
          std::map<DomainPoint, FutureImpl*>::const_iterator finder =
              futures.find(key);
          if (finder == futures.end())
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "A partition by weight call is missing an entry for a "
                  << "color in the color space. All colors must be present.";
            error.raise();
          }
          FutureImpl* future = finder->second;
          size_t future_size = 0;
          const void* data =
              future->find_runtime_buffer(op->get_context(), future_size);
          if (future_size == sizeof(int))
          {
            if (weights.empty())
            {
              if (!long_weights.empty())
              {
                Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
                error << "An invalid future size was found in a partition by "
                      << "weight call. All futures must be consistent int or "
                      << "size_t values.";
                error.raise();
              }
              weights.resize(count);
            }
            weights[color_index] = *(static_cast<const int*>(data));
          }
          else if (future_size == sizeof(size_t))
          {
            if (long_weights.empty())
            {
              if (!weights.empty())
              {
                Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
                error << "An invalid future size was found in a partition by "
                      << "weight call. All futures must be consistent int or "
                      << "size_t values.";
                error.raise();
              }
              long_weights.resize(count);
            }
            long_weights[color_index] = *(static_cast<const size_t*>(data));
          }
          else
          {
            Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
            error << "An invalid future size was found in a partition by "
                  << "weight call. All futures must contain int or "
                  << "size_t values.";
            error.raise();
          }
          child_colors[color_index++] = color_space->linearize_color(
              &itr.p, color_space->handle.get_type_tag());
        }
      }
      ApUserEvent to_trigger;
      DomainT<DIM, T> local_space;
      ApEvent ready = get_loose_index_space(local_space, to_trigger);
      if (op->has_execution_fence_event())
        ready = Runtime::merge_events(
            nullptr, ready, op->get_execution_fence_event());
      Realm::ProfilingRequestSet requests;
      if (runtime->profiler != nullptr)
        runtime->profiler->add_partition_request(
            requests, op, DEP_PART_WEIGHTS, ready);
      std::vector<Realm::IndexSpace<DIM, T> > subspaces;
      ApEvent result(
          weights.empty() ?
              local_space.create_weighted_subspaces(
                  count, granularity, long_weights, subspaces, requests,
                  ready) :
              local_space.create_weighted_subspaces(
                  count, granularity, weights, subspaces, requests, ready));
      if ((spy_logging_level > LIGHT_SPY_LOGGING) &&
          (!result.exists() || (result == ready)))
        Runtime::rename_event(result);
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger, result);
      LegionSpy::log_deppart_events(
          op->get_unique_op_id(), expr_id, ready, result, DEP_PART_WEIGHTS);
      // Iterate the local colors and destroy any that we don't use
      unsigned next = 0;
      for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
      {
        // Find the color
        std::vector<LegionColor>::iterator finder =
            std::lower_bound(child_colors.begin(), child_colors.end(), *itr);
        legion_assert(finder != child_colors.end());
        legion_assert(*finder == *itr);
        const unsigned offset = std::distance(child_colors.begin(), finder);
        legion_assert(next <= offset);
        while (next < offset) subspaces[next++].destroy();
        IndexSpaceNodeT<DIM, T>* child =
            static_cast<IndexSpaceNodeT<DIM, T>*>(partition->get_child(*itr));
        if (child->set_realm_index_space(subspaces[next++], result))
          delete child;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    template<int COLOR_DIM, typename COLOR_T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_by_field_helper(
        Operation* op, FieldID fid, IndexPartNode* partition,
        const std::vector<FieldDataDescriptor>& instances,
        std::vector<DeppartResult>* results, ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID source_space = runtime->address_space;
      // If we already have results then we can just fill them in
      if ((results != nullptr) && !results->empty())
      {
        DeppartResult key;
        for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
        {
          IndexSpaceNodeT<DIM, T>* child =
              static_cast<IndexSpaceNodeT<DIM, T>*>(partition->get_child(*itr));
          // Find the resulting color
          key.color = *itr;
          std::vector<DeppartResult>::const_iterator finder =
              std::lower_bound(results->begin(), results->end(), key);
          legion_assert(finder != results->end());
          legion_assert(finder->color == (*itr));
          Realm::IndexSpace<DIM, T> result = finder->domain;
          if (child->set_realm_index_space(
                  result, instances_ready, false /*initialization*/,
                  false /*broadcast*/, source_space))
            delete child;
        }
        return ApEvent::NO_AP_EVENT;
      }
      IndexSpaceNodeT<COLOR_DIM, COLOR_T>* color_space =
          static_cast<IndexSpaceNodeT<COLOR_DIM, COLOR_T>*>(
              partition->color_space);
      std::vector<Point<COLOR_DIM, COLOR_T> > colors;
      if (results == nullptr)
      {
        for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
        {
          Point<COLOR_DIM, COLOR_T> color;
          color_space->delinearize_color(*itr, color);
          colors.emplace_back(color);
        }
      }
      else
      {
        colors.resize(partition->total_children);
        results->resize(partition->total_children);
        unsigned index = 0;
        for (ColorSpaceIterator itr(partition); itr; itr++)
        {
          legion_assert(index < colors.size());
          results->at(index).color = *itr;
          color_space->delinearize_color(*itr, colors[index++]);
        }
      }
      // Translate the instances to realm field data descriptors
      typedef Realm::FieldDataDescriptor<
          Realm::IndexSpace<DIM, T>, Realm::Point<COLOR_DIM, COLOR_T> >
          RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      LargeNameClosure* closure = nullptr;
      if (runtime->profiler != nullptr)
        closure = new LargeNameClosure(this, fid, instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor& src = instances[idx];
        RealmDescriptor& dst = descriptors[idx];
        dst.index_space = src.domain;
        dst.inst = src.inst;
        dst.field_offset = fid;
        if (closure != nullptr)
          closure->record_instance_name(src.inst, src.unique, src.space);
      }
      // Perform the operation
      ApUserEvent to_trigger;
      DomainT<DIM, T> local_space;
      ApEvent parent_ready = get_loose_index_space(local_space, to_trigger);
      std::vector<ApEvent> preconditions;
      if (parent_ready.exists())
        preconditions.emplace_back(parent_ready);
      if (instances_ready.exists())
        preconditions.emplace_back(instances_ready);
      if (op->has_execution_fence_event())
        preconditions.emplace_back(op->get_execution_fence_event());
      ApEvent precondition = Runtime::merge_events(nullptr, preconditions);
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (runtime->profiler != nullptr)
        runtime->profiler->add_partition_request(
            requests, op, DEP_PART_BY_FIELD, precondition, closure);
      std::vector<Realm::IndexSpace<DIM, T> > subspaces;
      ApEvent result(local_space.create_subspaces_by_field(
          descriptors, colors, subspaces, requests, precondition));
      legion_assert(colors.size() == subspaces.size());
      if ((spy_logging_level > LIGHT_SPY_LOGGING) &&
          (!result.exists() || (result == precondition)))
        Runtime::rename_event(result);
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger, result);
      LegionSpy::log_deppart_events(
          op->get_unique_op_id(), expr_id, precondition, result,
          DEP_PART_BY_FIELD);
      unsigned index = (results == nullptr) ? 0 : colors.size();
      const bool broadcast =
          (results == nullptr) && partition->needs_subspace_broadcast();
      // Set our local children results here
      for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
      {
        if (index == colors.size())
          // Compute the index offset the first time through
          index = color_space->compute_color_offset(*itr);
        legion_assert(index < colors.size());
        IndexSpaceNodeT<DIM, T>* child =
            static_cast<IndexSpaceNodeT<DIM, T>*>(partition->get_child(*itr));
        if (child->set_realm_index_space(
                subspaces[index++], result, false /*initialization*/, broadcast,
                source_space))
          delete child;
      }
      if (results != nullptr)
        prepare_broadcast_results(partition, subspaces, *results, result);
      return result;
    }
#endif  // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::prepare_broadcast_results(
        IndexPartNode* partition, std::vector<DomainT<DIM, T> >& subspaces,
        std::vector<DeppartResult>& results, ApEvent& result)
    //--------------------------------------------------------------------------
    {
      legion_assert(subspaces.size() == results.size());
      // If we're broadcasting the resuls there should be a collective mapping
      legion_assert(partition->collective_mapping != nullptr);
      // Save the results to be broadcast if necessary
      const size_t total_spaces = partition->collective_mapping->size();
      if (total_spaces <= subspaces.size())
      {
        // This case is easy, there will be one owner of each subregion
        // so that subregion gets the reference added by Realm
        for (unsigned idx = 0; idx < subspaces.size(); idx++)
          results[idx].domain = subspaces[idx];
      }
      else
      {
        // This case is more tricky, there will be multiple owners of
        // at least some of the subspaces so we need to add extra
        // references to these subspaces if they have sparsity maps
        // because multiple subregions will be owners of the sparsity map
        const unsigned needed_references = total_spaces / subspaces.size();
        const unsigned remainder = total_spaces % subspaces.size();
        std::vector<ApEvent> reference_events;
        for (unsigned idx = 0; idx < subspaces.size(); idx++)
        {
          results[idx].domain = subspaces[idx];
          if (!subspaces[idx].dense())
          {
            const unsigned extra_references = needed_references +
                                              ((idx < remainder) ? 1 : 0) -
                                              1 /*already have one from Realm*/;
            if (extra_references > 0)
            {
              const ApEvent added(
                  subspaces[idx].sparsity.add_reference(extra_references));
              // These events should almost always be NO_EVENTs since we're
              // done this on the node where the sparsity maps were created
              if (added.exists())
                reference_events.push_back(added);
            }
          }
        }
        if (!reference_events.empty())
        {
          if (result.exists())
            reference_events.push_back(result);
          result = Runtime::merge_events(NULL, reference_events);
        }
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_by_image(
        Operation* op, FieldID fid, IndexPartNode* partition,
        IndexPartNode* projection, std::vector<FieldDataDescriptor>& instances,
        ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(partition->parent == this);
      // Demux the projection type to do the actual operations
      CreateByImageHelper creator(
          this, op, fid, partition, projection, instances, instances_ready);
      NT_TemplateHelper::demux<CreateByImageHelper>(
          projection->handle.get_type_tag(), &creator);
      return creator.result;
    }
#endif  // defined(DEFINE_NT_TEMPLATES)

#ifdef DEFINE_NTNT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM1, typename T1>
    template<int DIM2, typename T2>
    ApEvent IndexSpaceNodeT<DIM1, T1>::create_by_image_helper(
        Operation* op, FieldID fid, IndexPartNode* partition,
        IndexPartNode* projection, std::vector<FieldDataDescriptor>& instances,
        ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      ApEvent precondition;
      bool first_child = true;
      std::vector<ApEvent> results;
      ApUserEvent to_trigger;
      DomainT<DIM1, T1> local_space;
      const AddressSpaceID source_space = runtime->address_space;
      typedef Realm::FieldDataDescriptor<
          Realm::IndexSpace<DIM2, T2>, Realm::Point<DIM1, T1> >
          RealmDescriptor;
      for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
      {
        // Get the child of the projection partition
        IndexSpaceNodeT<DIM1, T1>* child =
            static_cast<IndexSpaceNodeT<DIM1, T1>*>(partition->get_child(*itr));
        // Partition by images are expensive to compute so we only want
        // to do it once so we only do it if we're the owner of the child
        // and then we'll broadcast it out to all the other copies
        if (!child->is_owner())
          continue;
        if (first_child)
        {
          std::vector<ApEvent> preconditions;
          ApEvent ready = get_loose_index_space(local_space, to_trigger);
          if (ready.exists())
            preconditions.emplace_back(ready);
          if (instances_ready.exists())
            preconditions.emplace_back(instances_ready);
          if (op->has_execution_fence_event())
            preconditions.emplace_back(op->get_execution_fence_event());
          precondition = Runtime::merge_events(nullptr, preconditions);
          // sort the instances so we can search for what we need
          std::sort(instances.begin(), instances.end());
          first_child = false;
        }
        std::vector<RealmDescriptor> descriptors;
        FieldDataDescriptor key;
        key.color = partition->color_space->delinearize_color_to_point(*itr);
        Realm::IndexSpace<DIM2, T2> source =
            Realm::IndexSpace<DIM2, T2>::make_empty();
        LargeNameClosure* closure = nullptr;
        if (runtime->profiler != nullptr)
          closure = new LargeNameClosure(this, fid);
        for (std::vector<FieldDataDescriptor>::const_iterator it =
                 std::lower_bound(instances.begin(), instances.end(), key);
             it != std::upper_bound(instances.begin(), instances.end(), key);
             it++)
        {
          descriptors.resize(descriptors.size() + 1);
          RealmDescriptor& dst = descriptors.back();
          dst.index_space = it->domain;
          source = dst.index_space;
          dst.inst = it->inst;
          dst.field_offset = fid;
          if (closure != nullptr)
            closure->record_instance_name(it->inst, it->unique, it->space);
        }
        // We should have exactly one of these here for each image
        legion_assert(descriptors.size() == 1);
        // Get the profiling requests
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != nullptr)
          runtime->profiler->add_partition_request(
              requests, op, DEP_PART_BY_IMAGE, precondition, closure);
        Realm::IndexSpace<DIM1, T1> subspace;
        ApEvent result(local_space.create_subspace_by_image(
            descriptors, source, subspace, requests, precondition));
        if ((spy_logging_level > LIGHT_SPY_LOGGING) &&
            (!result.exists() || (result == precondition)))
          Runtime::rename_event(result);
        LegionSpy::log_deppart_events(
            op->get_unique_op_id(), expr_id, precondition, result,
            DEP_PART_BY_IMAGE);
        // Set the result and indicate that we're broadcasting it
        if (child->set_realm_index_space(
                subspace, result, false /*initialization*/, true /*broadcast*/,
                source_space))
          delete child;
        if (result.exists())
          results.emplace_back(result);
      }
      ApEvent result;
      if (!results.empty())
        result = Runtime::merge_events(nullptr, results);
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger, result);
      return result;
    }
#endif  // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_by_image_range(
        Operation* op, FieldID fid, IndexPartNode* partition,
        IndexPartNode* projection, std::vector<FieldDataDescriptor>& instances,
        ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(partition->parent == this);
      // Demux the projection type to do the actual operations
      CreateByImageRangeHelper creator(
          this, op, fid, partition, projection, instances, instances_ready);
      NT_TemplateHelper::demux<CreateByImageRangeHelper>(
          projection->handle.get_type_tag(), &creator);
      return creator.result;
    }
#endif  // defined(DEFINE_NT_TEMPLATES)

#ifdef DEFINE_NTNT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM1, typename T1>
    template<int DIM2, typename T2>
    ApEvent IndexSpaceNodeT<DIM1, T1>::create_by_image_range_helper(
        Operation* op, FieldID fid, IndexPartNode* partition,
        IndexPartNode* projection, std::vector<FieldDataDescriptor>& instances,
        ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      ApEvent precondition;
      bool first_child = true;
      std::vector<ApEvent> results;
      ApUserEvent to_trigger;
      DomainT<DIM1, T1> local_space;
      const AddressSpaceID source_space = runtime->address_space;
      typedef Realm::FieldDataDescriptor<
          Realm::IndexSpace<DIM2, T2>, Realm::Rect<DIM1, T1> >
          RealmDescriptor;
      for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
      {
        // Get the child of the projection partition
        IndexSpaceNodeT<DIM1, T1>* child =
            static_cast<IndexSpaceNodeT<DIM1, T1>*>(partition->get_child(*itr));
        // Partition by images are expensive to compute so we only want
        // to do it once so we only do it if we're the owner of the child
        // and then we'll broadcast it out to all the other copies
        if (!child->is_owner())
          continue;
        if (first_child)
        {
          std::vector<ApEvent> preconditions;
          ApEvent ready = get_loose_index_space(local_space, to_trigger);
          if (ready.exists())
            preconditions.emplace_back(ready);
          if (instances_ready.exists())
            preconditions.emplace_back(instances_ready);
          if (op->has_execution_fence_event())
            preconditions.emplace_back(op->get_execution_fence_event());
          precondition = Runtime::merge_events(nullptr, preconditions);
          // sort the instances so we can search for what we need
          std::sort(instances.begin(), instances.end());
          first_child = false;
        }
        std::vector<RealmDescriptor> descriptors;
        FieldDataDescriptor key;
        key.color = partition->color_space->delinearize_color_to_point(*itr);
        Realm::IndexSpace<DIM2, T2> source =
            Realm::IndexSpace<DIM2, T2>::make_empty();
        LargeNameClosure* closure = nullptr;
        if (runtime->profiler != nullptr)
          closure = new LargeNameClosure(this, fid);
        for (std::vector<FieldDataDescriptor>::const_iterator it =
                 std::lower_bound(instances.begin(), instances.end(), key);
             it != std::upper_bound(instances.begin(), instances.end(), key);
             it++)
        {
          descriptors.resize(descriptors.size() + 1);
          RealmDescriptor& dst = descriptors.back();
          dst.index_space = it->domain;
          source = dst.index_space;
          dst.inst = it->inst;
          dst.field_offset = fid;
          if (closure != nullptr)
            closure->record_instance_name(it->inst, it->unique, it->space);
        }
        // We should have exactly one of these here for each image
        legion_assert(descriptors.size() == 1);
        // Get the profiling requests
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != nullptr)
          runtime->profiler->add_partition_request(
              requests, op, DEP_PART_BY_IMAGE_RANGE, precondition);
        Realm::IndexSpace<DIM1, T1> subspace;
        ApEvent result(local_space.create_subspace_by_image(
            descriptors, source, subspace, requests, precondition));
        if ((spy_logging_level > LIGHT_SPY_LOGGING) &&
            (!result.exists() || (result == precondition)))
          Runtime::rename_event(result);
        LegionSpy::log_deppart_events(
            op->get_unique_op_id(), expr_id, precondition, result,
            DEP_PART_BY_IMAGE_RANGE);
        // Set the result and indicate that we're broadcasting it
        if (child->set_realm_index_space(
                subspace, result, false /*initialization*/, true /*broadcast*/,
                source_space))
          delete child;
        if (result.exists())
          results.emplace_back(result);
      }
      ApEvent result;
      if (!results.empty())
        result = Runtime::merge_events(nullptr, results);
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger, result);
      return result;
    }
#endif  // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_by_preimage(
        Operation* op, FieldID fid, IndexPartNode* partition,
        IndexPartNode* projection,
        const std::vector<FieldDataDescriptor>& instances,
        const std::map<DomainPoint, Domain>* remote_targets,
        std::vector<DeppartResult>* results, ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(partition->parent == this);
      // Demux the projection type to do the actual operations
      CreateByPreimageHelper creator(
          this, op, fid, partition, projection, instances, remote_targets,
          results, instances_ready);
      NT_TemplateHelper::demux<CreateByPreimageHelper>(
          projection->handle.get_type_tag(), &creator);
      return creator.result;
    }
#endif  // defined(DEFINE_NT_TEMPLATES)

#ifdef DEFINE_NTNT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM1, typename T1>
    template<int DIM2, typename T2>
    ApEvent IndexSpaceNodeT<DIM1, T1>::create_by_preimage_helper(
        Operation* op, FieldID fid, IndexPartNode* partition,
        IndexPartNode* projection,
        const std::vector<FieldDataDescriptor>& instances,
        const std::map<DomainPoint, Domain>* remote_targets,
        std::vector<DeppartResult>* results, ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID source_space = runtime->address_space;
      // If we already have results then we can just fill them in
      if ((results != nullptr) && !results->empty())
      {
        DeppartResult key;
        for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
        {
          IndexSpaceNodeT<DIM1, T1>* child =
              static_cast<IndexSpaceNodeT<DIM1, T1>*>(
                  partition->get_child(*itr));
          // Find the resulting color
          key.color = *itr;
          std::vector<DeppartResult>::const_iterator finder =
              std::lower_bound(results->begin(), results->end(), key);
          legion_assert(finder != results->end());
          legion_assert(finder->color == (*itr));
          Realm::IndexSpace<DIM1, T1> result = finder->domain;
          if (child->set_realm_index_space(
                  result, instances_ready, false /*initialization*/,
                  false /*broadcast*/, source_space))
            delete child;
        }
        return ApEvent::NO_AP_EVENT;
      }
      // Get the target index spaces of the projection partition
      ApUserEvent to_trigger;
      std::vector<ApEvent> preconditions;
      std::vector<Realm::IndexSpace<DIM2, T2> > targets;
      if (results == nullptr)
      {
        legion_assert(remote_targets == nullptr);
        for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
        {
          const DomainPoint color =
              partition->color_space->delinearize_color_to_point(*itr);
          // Get the corresponding subspace for the targets
          const LegionColor target_color =
              projection->color_space->linearize_color(color);
          IndexSpaceNodeT<DIM2, T2>* target_child =
              static_cast<IndexSpaceNodeT<DIM2, T2>*>(
                  projection->get_child(target_color));
          targets.resize(targets.size() + 1);
          ApEvent ready =
              target_child->get_loose_index_space(targets.back(), to_trigger);
          if (ready.exists())
            preconditions.emplace_back(ready);
        }
      }
      else
      {
        legion_assert(remote_targets != nullptr);
        unsigned index = 0;
        targets.resize(partition->total_children);
        results->resize(partition->total_children);
        for (ColorSpaceIterator itr(partition); itr; itr++)
        {
          legion_assert(index < targets.size());
          results->at(index).color = *itr;
          const DomainPoint color =
              partition->color_space->delinearize_color_to_point(*itr);
          std::map<DomainPoint, Domain>::const_iterator finder =
              remote_targets->find(color);
          if (finder != remote_targets->end())
          {
            targets[index++] = finder->second;
            continue;
          }
          // Get the corresponding subspace for the targets
          const LegionColor target_color =
              projection->color_space->linearize_color(color);
          IndexSpaceNodeT<DIM2, T2>* target_child =
              static_cast<IndexSpaceNodeT<DIM2, T2>*>(
                  projection->get_child(target_color));
          ApEvent ready =
              target_child->get_loose_index_space(targets[index++], to_trigger);
          if (ready.exists())
            preconditions.emplace_back(ready);
        }
        legion_assert(index == targets.size());
      }
      // Translate the descriptors into realm descriptors
      typedef Realm::FieldDataDescriptor<
          Realm::IndexSpace<DIM1, T1>, Realm::Point<DIM2, T2> >
          RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      LargeNameClosure* closure = nullptr;
      if (runtime->profiler != nullptr)
        closure = new LargeNameClosure(this, fid, instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor& src = instances[idx];
        RealmDescriptor& dst = descriptors[idx];
        dst.index_space = src.domain;
        dst.inst = src.inst;
        dst.field_offset = fid;
        if (closure != nullptr)
          closure->record_instance_name(src.inst, src.unique, src.space);
      }
      // Perform the operation
      DomainT<DIM1, T1> local_space;
      ApEvent ready = get_loose_index_space(local_space, to_trigger);
      if (ready.exists())
        preconditions.emplace_back(ready);
      if (instances_ready.exists())
        preconditions.emplace_back(instances_ready);
      if (op->has_execution_fence_event())
        preconditions.emplace_back(op->get_execution_fence_event());
      std::vector<Realm::IndexSpace<DIM1, T1> > subspaces;
      ApEvent precondition = Runtime::merge_events(nullptr, preconditions);
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (runtime->profiler != nullptr)
        runtime->profiler->add_partition_request(
            requests, op, DEP_PART_BY_PREIMAGE, precondition, closure);
      ApEvent result(local_space.create_subspaces_by_preimage(
          descriptors, targets, subspaces, requests, precondition));
      if ((spy_logging_level > LIGHT_SPY_LOGGING) &&
          (!result.exists() || (result == precondition)))
        Runtime::rename_event(result);
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger, result);
      LegionSpy::log_deppart_events(
          op->get_unique_op_id(), expr_id, precondition, result,
          DEP_PART_BY_PREIMAGE);
      // Update any local children with their results
      unsigned index = (results == nullptr) ? 0 : subspaces.size();
      const bool broadcast =
          (results == nullptr) && partition->needs_subspace_broadcast();
      // Set our local children results here
      for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
      {
        if (index == subspaces.size())
          // Compute the index offset the first time through
          index = partition->color_space->compute_color_offset(*itr);
        legion_assert(index < subspaces.size());
        IndexSpaceNodeT<DIM1, T1>* child =
            static_cast<IndexSpaceNodeT<DIM1, T1>*>(partition->get_child(*itr));
        if (child->set_realm_index_space(
                subspaces[index++], result, false /*initialization*/, broadcast,
                source_space))
          delete child;
      }
      if (results != nullptr)
        prepare_broadcast_results(partition, subspaces, *results, result);
      return result;
    }
#endif  // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_by_preimage_range(
        Operation* op, FieldID fid, IndexPartNode* partition,
        IndexPartNode* projection,
        const std::vector<FieldDataDescriptor>& instances,
        const std::map<DomainPoint, Domain>* remote_targets,
        std::vector<DeppartResult>* results, ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(partition->parent == this);
      // Demux the projection type to do the actual operations
      CreateByPreimageRangeHelper creator(
          this, op, fid, partition, projection, instances, remote_targets,
          results, instances_ready);
      NT_TemplateHelper::demux<CreateByPreimageRangeHelper>(
          projection->handle.get_type_tag(), &creator);
      return creator.result;
    }
#endif  // defined(DEFINE_NT_TEMPLATES)

#ifdef DEFINE_NTNT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM1, typename T1>
    template<int DIM2, typename T2>
    ApEvent IndexSpaceNodeT<DIM1, T1>::create_by_preimage_range_helper(
        Operation* op, FieldID fid, IndexPartNode* partition,
        IndexPartNode* projection,
        const std::vector<FieldDataDescriptor>& instances,
        const std::map<DomainPoint, Domain>* remote_targets,
        std::vector<DeppartResult>* results, ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID source_space = runtime->address_space;
      // If we already have results then we can just fill them in
      if ((results != nullptr) && !results->empty())
      {
        DeppartResult key;
        for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
        {
          IndexSpaceNodeT<DIM1, T1>* child =
              static_cast<IndexSpaceNodeT<DIM1, T1>*>(
                  partition->get_child(*itr));
          // Find the resulting color
          key.color = *itr;
          std::vector<DeppartResult>::const_iterator finder =
              std::lower_bound(results->begin(), results->end(), key);
          legion_assert(finder != results->end());
          legion_assert(finder->color == (*itr));
          Realm::IndexSpace<DIM1, T1> result = finder->domain;
          if (child->set_realm_index_space(
                  result, instances_ready, false /*initialization*/,
                  false /*broadcast*/, source_space))
            delete child;
        }
        return ApEvent::NO_AP_EVENT;
      }

      // Get the target index spaces of the projection partition
      ApUserEvent to_trigger;
      std::vector<ApEvent> preconditions;
      std::vector<DomainT<DIM2, T2> > targets;
      if (results == nullptr)
      {
        legion_assert(remote_targets == nullptr);
        for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
        {
          const DomainPoint color =
              partition->color_space->delinearize_color_to_point(*itr);
          // Get the corresponding subspace for the targets
          const LegionColor target_color =
              projection->color_space->linearize_color(color);
          IndexSpaceNodeT<DIM2, T2>* target_child =
              static_cast<IndexSpaceNodeT<DIM2, T2>*>(
                  projection->get_child(target_color));
          targets.resize(targets.size() + 1);
          ApEvent ready =
              target_child->get_loose_index_space(targets.back(), to_trigger);
          if (ready.exists())
            preconditions.emplace_back(ready);
        }
      }
      else
      {
        legion_assert(remote_targets != nullptr);
        unsigned index = 0;
        targets.resize(partition->total_children);
        results->resize(partition->total_children);
        for (ColorSpaceIterator itr(partition); itr; itr++)
        {
          legion_assert(index < targets.size());
          results->at(index).color = *itr;
          const DomainPoint color =
              partition->color_space->delinearize_color_to_point(*itr);
          std::map<DomainPoint, Domain>::const_iterator finder =
              remote_targets->find(color);
          if (finder != remote_targets->end())
          {
            targets[index++] = finder->second;
            continue;
          }
          // Get the corresponding subspace for the targets
          const LegionColor target_color =
              projection->color_space->linearize_color(color);
          IndexSpaceNodeT<DIM2, T2>* target_child =
              static_cast<IndexSpaceNodeT<DIM2, T2>*>(
                  projection->get_child(target_color));
          ApEvent ready =
              target_child->get_loose_index_space(targets[index++], to_trigger);
          if (ready.exists())
            preconditions.emplace_back(ready);
        }
        legion_assert(index == targets.size());
      }
      // Translate the descriptors into realm descriptors
      typedef Realm::FieldDataDescriptor<
          Realm::IndexSpace<DIM1, T1>, Realm::Rect<DIM2, T2> >
          RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      LargeNameClosure* closure = nullptr;
      if (runtime->profiler != nullptr)
        closure = new LargeNameClosure(this, fid, instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor& src = instances[idx];
        RealmDescriptor& dst = descriptors[idx];
        dst.index_space = src.domain;
        dst.inst = src.inst;
        dst.field_offset = fid;
        if (closure != nullptr)
          closure->record_instance_name(src.inst, src.unique, src.space);
      }
      // Perform the operation
      DomainT<DIM1, T1> local_space;
      ApEvent ready = get_loose_index_space(local_space, to_trigger);
      if (ready.exists())
        preconditions.emplace_back(ready);
      if (instances_ready.exists())
        preconditions.emplace_back(instances_ready);
      if (op->has_execution_fence_event())
        preconditions.emplace_back(op->get_execution_fence_event());
      std::vector<Realm::IndexSpace<DIM1, T1> > subspaces;
      ApEvent precondition = Runtime::merge_events(nullptr, preconditions);
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (runtime->profiler != nullptr)
        runtime->profiler->add_partition_request(
            requests, op, DEP_PART_BY_PREIMAGE_RANGE, precondition, closure);
      ApEvent result(local_space.create_subspaces_by_preimage(
          descriptors, targets, subspaces, requests, precondition));
      if ((spy_logging_level > LIGHT_SPY_LOGGING) &&
          (!result.exists() || (result == precondition)))
        Runtime::rename_event(result);
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger, result);
      LegionSpy::log_deppart_events(
          op->get_unique_op_id(), expr_id, precondition, result,
          DEP_PART_BY_PREIMAGE_RANGE);
      // Update any local children with their results
      unsigned index = (results == nullptr) ? 0 : subspaces.size();
      const bool broadcast =
          (results == nullptr) && partition->needs_subspace_broadcast();
      // Set our local children results here
      for (ColorSpaceIterator itr(partition, true /*local only*/); itr; itr++)
      {
        if (index == subspaces.size())
          // Compute the index offset the first time through
          index = partition->color_space->compute_color_offset(*itr);
        legion_assert(index < subspaces.size());
        IndexSpaceNodeT<DIM1, T1>* child =
            static_cast<IndexSpaceNodeT<DIM1, T1>*>(partition->get_child(*itr));
        if (child->set_realm_index_space(
                subspaces[index++], result, false /*initialization*/, broadcast,
                source_space))
          delete child;
      }
      if (results != nullptr)
        prepare_broadcast_results(partition, subspaces, *results, result);
      return result;
    }
#endif  // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::create_association(
        Operation* op, FieldID fid, IndexSpaceNode* range,
        const std::vector<FieldDataDescriptor>& instances,
        ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      // Demux the range type to do the actual operation
      CreateAssociationHelper creator(
          this, op, fid, range, instances, instances_ready);
      NT_TemplateHelper::demux<CreateAssociationHelper>(
          range->handle.get_type_tag(), &creator);
      return creator.result;
    }
#endif  // defined(DEFINE_NT_TEMPLATES)

#ifdef DEFINE_NTNT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM1, typename T1>
    template<int DIM2, typename T2>
    ApEvent IndexSpaceNodeT<DIM1, T1>::create_association_helper(
        Operation* op, FieldID fid, IndexSpaceNode* range,
        const std::vector<FieldDataDescriptor>& instances,
        ApEvent instances_ready)
    //--------------------------------------------------------------------------
    {
      // Translate the descriptors into realm descriptors
      typedef Realm::FieldDataDescriptor<
          Realm::IndexSpace<DIM1, T1>, Realm::Point<DIM2, T2> >
          RealmDescriptor;
      std::vector<RealmDescriptor> descriptors(instances.size());
      LargeNameClosure* closure = nullptr;
      if (runtime->profiler != nullptr)
        closure = new LargeNameClosure(this, fid, instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const FieldDataDescriptor& src = instances[idx];
        RealmDescriptor& dst = descriptors[idx];
        dst.index_space = src.domain;
        dst.inst = src.inst;
        dst.field_offset = fid;
        if (closure != nullptr)
          closure->record_instance_name(src.inst, src.unique, src.space);
      }
      // Get the range index space
      IndexSpaceNodeT<DIM2, T2>* range_node =
          static_cast<IndexSpaceNodeT<DIM2, T2>*>(range);
      ApUserEvent to_trigger;
      DomainT<DIM2, T2> range_space;
      ApEvent range_ready =
          range_node->get_loose_index_space(range_space, to_trigger);
      std::vector<ApEvent> preconditions;
      if (range_ready.exists())
        preconditions.emplace_back(range_ready);
      DomainT<DIM1, T1> local_space;
      ApEvent local_ready = get_loose_index_space(local_space, to_trigger);
      if (local_ready.exists())
        preconditions.emplace_back(local_ready);
      if (instances_ready.exists())
        preconditions.emplace_back(instances_ready);
      if (op->has_execution_fence_event())
        preconditions.emplace_back(op->get_execution_fence_event());
      // Issue the operation
      ApEvent precondition = Runtime::merge_events(nullptr, preconditions);
      // Get the profiling requests
      Realm::ProfilingRequestSet requests;
      if (runtime->profiler != nullptr)
        runtime->profiler->add_partition_request(
            requests, op, DEP_PART_ASSOCIATION, precondition, closure);
      ApEvent result(local_space.create_association(
          descriptors, range_space, requests, precondition));
      if ((spy_logging_level > LIGHT_SPY_LOGGING) &&
          (!result.exists() || (result == precondition)))
        Runtime::rename_event(result);
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger, result);
      LegionSpy::log_deppart_events(
          op->get_unique_op_id(), expr_id, precondition, result,
          DEP_PART_ASSOCIATION);
      return result;
    }
#endif  // defined(DEFINE_NTNT_TEMPLATES)

#ifdef DEFINE_NT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t IndexSpaceNodeT<DIM, T>::get_coordinate_size(bool range) const
    //--------------------------------------------------------------------------
    {
      if (range)
        return sizeof(Realm::Rect<DIM, T>);
      else
        return sizeof(Realm::Point<DIM, T>);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    Realm::InstanceLayoutGeneric* IndexSpaceNodeT<DIM, T>::create_hdf5_layout(
        const std::vector<FieldID>& field_ids,
        const std::vector<size_t>& field_sizes,
        const std::vector<std::string>& field_files,
        const OrderingConstraint& dimension_order)
    //--------------------------------------------------------------------------
    {
      legion_assert(int(dimension_order.ordering.size()) == (DIM + 1));
      legion_assert(dimension_order.ordering.back() == LEGION_DIM_F);
#ifdef LEGION_USE_HDF5
      DomainT<DIM, T> local_space = get_tight_index_space();
      Realm::InstanceLayout<DIM, T>* layout = new Realm::InstanceLayout<DIM, T>;
      layout->bytes_used = 0;
      layout->alignment_reqd = 0;  // no allocation being made
      layout->space = local_space;
      layout->piece_lists.resize(field_ids.size());
      for (size_t i = 0; i < field_ids.size(); i++)
      {
        Realm::InstanceLayoutGeneric::FieldLayout& fl =
            layout->fields[field_ids[i]];
        fl.list_idx = i;
        fl.rel_offset = 0;
        fl.size_in_bytes = field_sizes[i];

        // create a single piece (for non-empty index spaces)
        if (!local_space.empty())
        {
          Realm::HDF5LayoutPiece<DIM, T>* hlp =
              new Realm::HDF5LayoutPiece<DIM, T>;
          hlp->bounds = local_space.bounds;
          layout->bytes_used += hlp->bounds.volume() * fl.size_in_bytes;
          hlp->dsetname = field_files[i];
          for (int j = 0; j < DIM; j++) hlp->offset[j] = 0;
          // Legion ordering constraints are listed from fastest to
          // slowest like fortran order, hdf5 is the opposite though
          // so we want to list dimensions in order from slowest to fastest
          for (unsigned idx = 0; idx < DIM; idx++)
            hlp->dim_order[idx] = dimension_order.ordering[DIM - 1 - idx];
          layout->piece_lists[i].pieces.emplace_back(hlp);
        }
      }
      return layout;
#else
      std::abort();  // should never get here
#endif
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceNodeT<DIM, T>::inline_union(
        IndexSpaceExpression* rhs)
    //--------------------------------------------------------------------------
    {
      return inline_union_internal<DIM, T>(rhs);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceNodeT<DIM, T>::inline_union(
        const SetView<IndexSpaceExpression*>& exprs)
    //--------------------------------------------------------------------------
    {
      return inline_union_internal<DIM, T>(exprs);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceNodeT<DIM, T>::inline_intersection(
        IndexSpaceExpression* rhs)
    //--------------------------------------------------------------------------
    {
      return inline_intersection_internal<DIM, T>(rhs);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceNodeT<DIM, T>::inline_intersection(
        const SetView<IndexSpaceExpression*>& exprs)
    //--------------------------------------------------------------------------
    {
      return inline_intersection_internal<DIM, T>(exprs);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceNodeT<DIM, T>::inline_subtraction(
        IndexSpaceExpression* rhs)
    //--------------------------------------------------------------------------
    {
      return inline_subtraction_internal<DIM, T>(rhs);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    uint64_t IndexSpaceNodeT<DIM, T>::get_canonical_hash(void)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> domain = get_tight_index_space();
      return get_canonical_hash_internal(domain);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::issue_fill(
        Operation* op, const PhysicalTraceInfo& trace_info,
        const std::vector<CopySrcDstField>& dst_fields, const void* fill_value,
        size_t fill_size, UniqueID fill_uid, FieldSpace handle,
        RegionTreeID tree_id, ApEvent precondition, PredEvent pred_guard,
        LgEvent unique_event, CollectiveKind collective, bool record_effect,
        int priority, bool replay)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> local_space = get_tight_index_space();
      return issue_fill_internal(
          op, local_space, trace_info, dst_fields, fill_value, fill_size,
          fill_uid, handle, tree_id, precondition, pred_guard, unique_event,
          collective, record_effect, priority, replay);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent IndexSpaceNodeT<DIM, T>::issue_copy(
        Operation* op, const PhysicalTraceInfo& trace_info,
        const std::vector<CopySrcDstField>& dst_fields,
        const std::vector<CopySrcDstField>& src_fields,
        const std::vector<Reservation>& reservations, RegionTreeID src_tree_id,
        RegionTreeID dst_tree_id, ApEvent precondition, PredEvent pred_guard,
        LgEvent src_unique, LgEvent dst_unique, CollectiveKind collective,
        bool record_effect, int priority, bool replay)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> local_space = get_tight_index_space();
      return issue_copy_internal(
          op, local_space, trace_info, dst_fields, src_fields, reservations,
          src_tree_id, dst_tree_id, precondition, pred_guard, src_unique,
          dst_unique, collective, record_effect, priority, replay);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    Realm::InstanceLayoutGeneric* IndexSpaceNodeT<DIM, T>::create_layout(
        const LayoutConstraintSet& constraints,
        const std::vector<FieldID>& field_ids,
        const std::vector<size_t>& field_sizes, bool compact, void** piece_list,
        size_t* piece_list_size, size_t* num_pieces, size_t base_alignment)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> local_is = get_tight_index_space();
      return create_layout_internal(
          local_is, constraints, field_ids, field_sizes, compact, piece_list,
          piece_list_size, num_pieces, base_alignment);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceNodeT<DIM, T>::create_layout_expression(
        const void* piece_list, size_t piece_list_size)
    //--------------------------------------------------------------------------
    {
      legion_assert((piece_list_size % sizeof(Rect<DIM, T>)) == 0);
      DomainT<DIM, T> local_is = get_tight_index_space();
      return create_layout_expression_internal(
          local_is, static_cast<const Rect<DIM, T>*>(piece_list),
          piece_list_size / sizeof(Rect<DIM, T>));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM, T>::meets_layout_expression(
        IndexSpaceExpression* space_expr, bool tight_bounds,
        const void* piece_list, size_t piece_list_size,
        const Domain* padding_delta)
    //--------------------------------------------------------------------------
    {
      legion_assert((piece_list_size % sizeof(Rect<DIM, T>)) == 0);
      return meets_layout_expression_internal<DIM, T>(
          space_expr, tight_bounds,
          static_cast<const Rect<DIM, T>*>(piece_list),
          piece_list_size / sizeof(Rect<DIM, T>), padding_delta);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpaceExpression* IndexSpaceNodeT<DIM, T>::find_congruent_expression(
        SmallPointerVector<IndexSpaceExpression, true>& expressions)
    //--------------------------------------------------------------------------
    {
      return find_congruent_expression_internal<DIM, T>(expressions);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    KDTree* IndexSpaceNodeT<DIM, T>::get_sparsity_map_kd_tree(void)
    //--------------------------------------------------------------------------
    {
      return get_sparsity_map_kd_tree_internal<DIM, T>();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    EqKDTree* IndexSpaceNodeT<DIM, T>::create_equivalence_set_kd_tree(
        size_t total_shards)
    //--------------------------------------------------------------------------
    {
      legion_assert(total_shards > 0);
      DomainT<DIM, T> realm_index_space = get_tight_index_space();
      if (total_shards == 1)
      {
        // Non replicated path
        // If it's dense we can just make a node and return it
        if (!realm_index_space.dense())
        {
          // If it's not dense then we need to make a sparse one to handle
          // all the rects associated with the sparsity map
          std::vector<Rect<DIM, T> > rects;
          for (Realm::IndexSpaceIterator<DIM, T> itr(realm_index_space);
               itr.valid; itr.step())
            rects.emplace_back(itr.rect);
          return new EqKDSparse<DIM, T>(realm_index_space.bounds, rects);
        }
        else
          return new EqKDNode<DIM, T>(realm_index_space.bounds);
      }
      else
      {
        // Control replicated path
        if (!realm_index_space.dense())
        {
          std::vector<Rect<DIM, T> > rects;
          for (Realm::IndexSpaceIterator<DIM, T> itr(realm_index_space);
               itr.valid; itr.step())
            rects.emplace_back(itr.rect);
          return new EqKDSparseSharded<DIM, T>(
              realm_index_space.bounds, 0 /*lower shard*/,
              total_shards - 1 /*upper shard*/, rects);
        }
        else
          return new EqKDSharded<DIM, T>(
              realm_index_space.bounds, 0 /*lower shard*/,
              total_shards - 1 /*upper shard*/);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::initialize_equivalence_set_kd_tree(
        EqKDTree* tree, EquivalenceSet* set, const FieldMask& mask,
        ShardID local_shard, bool current)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> realm_index_space = get_tight_index_space();
      EqKDTreeT<DIM, T>* typed_tree = tree->as_eq_kd_tree<DIM, T>();
      if (realm_index_space.empty())
      {
        // For backwards compatibility we handle the empty case which will
        // still store an equivalence set with names in it even though it
        // doesn't have to be updated ever
        legion_assert(realm_index_space.bounds.empty());
        legion_assert(typed_tree->bounds == realm_index_space.bounds);
        typed_tree->initialize_set(
            set, realm_index_space.bounds, mask, local_shard, current);
      }
      else
      {
        for (Realm::IndexSpaceIterator<DIM, T> itr(realm_index_space);
             itr.valid; itr.step())
        {
          const Rect<DIM, T> overlap =
              itr.rect.intersection(typed_tree->bounds);
          if (!overlap.empty())
            typed_tree->initialize_set(
                set, overlap, mask, local_shard, current);
        }
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::compute_equivalence_sets(
        EqKDTree* tree, LocalLock* tree_lock, const FieldMask& mask,
        const std::vector<EqSetTracker*>& trackers,
        const std::vector<AddressSpaceID>& tracker_spaces,
        std::vector<unsigned>& new_tracker_references,
        op::FieldMaskMap<EquivalenceSet>& eq_sets,
        std::vector<RtEvent>& pending_sets,
        op::FieldMaskMap<EqKDTree>& subscriptions,
        op::FieldMaskMap<EqKDTree>& to_create,
        op::map<EqKDTree*, Domain>& creation_rects,
        op::map<EquivalenceSet*, op::map<Domain, FieldMask> >& creation_srcs,
        op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> realm_index_space = get_tight_index_space();
      EqKDTreeT<DIM, T>* typed_tree = tree->as_eq_kd_tree<DIM, T>();
      // Need non-exclusive access to the tree for non-invalidations
      AutoLock t_lock(*tree_lock, false /*exclusive*/);
      for (Realm::IndexSpaceIterator<DIM, T> itr(realm_index_space); itr.valid;
           itr.step())
        typed_tree->compute_equivalence_sets(
            itr.rect, mask, trackers, tracker_spaces, new_tracker_references,
            eq_sets, pending_sets, subscriptions, to_create, creation_rects,
            creation_srcs, remote_shard_rects, local_shard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    unsigned IndexSpaceNodeT<DIM, T>::record_output_equivalence_set(
        EqKDTree* tree, LocalLock* tree_lock, EquivalenceSet* set,
        const FieldMask& mask, EqSetTracker* tracker,
        AddressSpaceID tracker_space,
        local::FieldMaskMap<EqKDTree>& subscriptions,
        op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      EqKDTreeT<DIM, T>* typed_tree = tree->as_eq_kd_tree<DIM, T>();
      DomainT<DIM, T> realm_index_space = get_tight_index_space();
      unsigned new_subs = 0;
      // Need non-exclusive access to the tree for non-invalidations
      AutoLock t_lock(*tree_lock, false /*exclusive*/);
      for (Realm::IndexSpaceIterator<DIM, T> itr(realm_index_space); itr.valid;
           itr.step())
        new_subs += typed_tree->record_output_equivalence_set(
            set, itr.rect, mask, tracker, tracker_space, subscriptions,
            remote_shard_rects, local_shard);
      return new_subs;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::invalidate_equivalence_set_kd_tree(
        EqKDTree* tree, LocalLock* tree_lock, const FieldMask& mask,
        std::vector<RtEvent>& invalidated, bool move_to_previous)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> realm_index_space = get_tight_index_space();
      EqKDTreeT<DIM, T>* typed_tree = tree->as_eq_kd_tree<DIM, T>();
      // Need exclusive access to the tree for invalidations
      AutoLock t_lock(*tree_lock);
      for (Realm::IndexSpaceIterator<DIM, T> itr(realm_index_space); itr.valid;
           itr.step())
        typed_tree->invalidate_tree(
            itr.rect, mask, invalidated, move_to_previous);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::invalidate_shard_equivalence_set_kd_tree(
        EqKDTree* tree, LocalLock* tree_lock, const FieldMask& mask,
        std::vector<RtEvent>& invalidated,
        op::map<ShardID, op::map<Domain, FieldMask> >& remote_shard_rects,
        ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> realm_index_space = get_tight_index_space();
      EqKDTreeT<DIM, T>* typed_tree = tree->as_eq_kd_tree<DIM, T>();
      // Need exclusive access to the tree for invalidations
      AutoLock t_lock(*tree_lock);
      for (Realm::IndexSpaceIterator<DIM, T> itr(realm_index_space); itr.valid;
           itr.step())
        typed_tree->invalidate_shard_tree_remote(
            itr.rect, mask, invalidated, remote_shard_rects, local_shard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::find_trace_local_sets_kd_tree(
        EqKDTree* tree, LocalLock* tree_lock, const FieldMask& mask,
        unsigned req_index, ShardID local_shard,
        std::map<EquivalenceSet*, unsigned>& current_sets)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> realm_index_space = get_tight_index_space();
      EqKDTreeT<DIM, T>* typed_tree = tree->as_eq_kd_tree<DIM, T>();
      // Need non-exclusive access to the tree for non-invalidations
      AutoLock t_lock(*tree_lock, false /*exclusive*/);
      for (Realm::IndexSpaceIterator<DIM, T> itr(realm_index_space); itr.valid;
           itr.step())
        typed_tree->find_trace_local_sets(
            itr.rect, mask, req_index, local_shard, current_sets);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::find_shard_trace_local_sets_kd_tree(
        EqKDTree* tree, LocalLock* tree_lock, const FieldMask& mask,
        unsigned req_index, std::map<EquivalenceSet*, unsigned>& current_sets,
        local::map<ShardID, FieldMask>& remote_shards, ShardID local_shard)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> realm_index_space = get_tight_index_space();
      EqKDTreeT<DIM, T>* typed_tree = tree->as_eq_kd_tree<DIM, T>();
      // Need non-exclusive access to the tree for non-invalidations
      AutoLock t_lock(*tree_lock, false /*exclusive*/);
      for (Realm::IndexSpaceIterator<DIM, T> itr(realm_index_space); itr.valid;
           itr.step())
        typed_tree->find_shard_trace_local_sets(
            itr.rect, mask, req_index, current_sets, remote_shards,
            local_shard);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::validate_slicing(
        const std::vector<IndexSpace>& slice_spaces, MultiTask* task,
        MapperManager* mapper)
    //--------------------------------------------------------------------------
    {
      std::vector<IndexSpaceNodeT<DIM, T>*> slice_nodes(slice_spaces.size());
      for (unsigned idx = 0; idx < slice_spaces.size(); idx++)
      {
        legion_assert(
            slice_spaces[idx].get_type_tag() == handle.get_type_tag());
        slice_nodes[idx] = static_cast<IndexSpaceNodeT<DIM, T>*>(
            runtime->get_node(slice_spaces[idx]));
      }
      // Iterate over the points and make sure that they exist in exactly
      // one slice space, no more, no less
      Realm::IndexSpace<DIM, T> local_space = get_tight_index_space();
      for (PointInDomainIterator<DIM, T> itr(local_space); itr(); itr++)
      {
        bool found = false;
        const Realm::Point<DIM, T>& point = *itr;
        for (unsigned idx = 0; idx < slice_nodes.size(); idx++)
        {
          if (!slice_nodes[idx]->contains_point(point))
            continue;
          if (found)
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error << "Invalid mapper output from invocation of 'slice_task' on "
                  << "mapper " << *mapper << ". Mapper returned multiple "
                  << "slices that contained the same point for task " << *task
                  << ".";
            error.raise();
          }
          else
            found = true;
        }
        if (!found)
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from invocation of 'slice_task' on "
                << "mapper " << *mapper << ". Mapper returned no slices that "
                << "contained some point(s) for task " << *task << ".";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::log_launch_space(UniqueID op_id)
    //--------------------------------------------------------------------------
    {
      Realm::IndexSpace<DIM, T> local_space = get_tight_index_space();
      for (Realm::IndexSpaceIterator<DIM, T> itr(local_space); itr.valid;
           itr.step())
        LegionSpy::log_launch_index_space_rect<DIM>(
            op_id, Rect<DIM, T>(itr.rect));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexSpace IndexSpaceNodeT<DIM, T>::create_shard_space(
        ShardingFunction* func, ShardID shard, IndexSpace shard_space,
        const Domain& shard_domain,
        const std::vector<DomainPoint>& shard_points, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> local_space = get_tight_index_space();
      Domain sharding_domain;
      if (shard_space != handle)
        runtime->find_domain(shard_space, sharding_domain);
      else
        sharding_domain = local_space;
      std::vector<Realm::Point<DIM, T> > index_points;
      if (!func->functor->is_invertible())
      {
        for (Realm::IndexSpaceIterator<DIM, T> rect_itr(local_space);
             rect_itr.valid; rect_itr.step())
        {
          for (Realm::PointInRectIterator<DIM, T> itr(rect_itr.rect); itr.valid;
               itr.step())
          {
            const ShardID point_shard = func->find_owner(
                DomainPoint(Point<DIM, T>(itr.p)), sharding_domain);
            if (point_shard == shard)
              index_points.emplace_back(itr.p);
          }
        }
      }
      else
      {
        std::vector<DomainPoint> domain_points;
        if (func->use_points)
          func->functor->invert_points(
              shard_points[shard], shard_points, shard_domain,
              Domain(local_space), sharding_domain, domain_points);
        else
          func->functor->invert(
              shard, sharding_domain, Domain(local_space), shard_points.size(),
              domain_points);
        index_points.resize(domain_points.size());
        for (unsigned idx = 0; idx < domain_points.size(); idx++)
          index_points[idx] = Point<DIM, coord_t>(domain_points[idx]);
      }
      if (index_points.empty())
        return IndexSpace::NO_SPACE;
      // Another useful case is if all the points are in the shard then
      // we can return ourselves as the result
      if (index_points.size() == get_volume())
        return handle;
      Realm::IndexSpace<DIM, T> realm_is(index_points);
      Domain domain((DomainT<DIM, T>(realm_is)));
      // Make sure this is set even if the realm index space is dense
      domain.is_type = handle.get_type_tag();
      return runtime->find_or_create_index_slice_space(
          domain, true /*take ownership*/, provenance);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexSpaceNodeT<DIM, T>::compute_range_shards(
        ShardingFunction* func, IndexSpace shard_space,
        const std::vector<DomainPoint>& shard_points,
        const Domain& shard_domain, std::set<ShardID>& range_shards)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> local_space = get_tight_index_space();
      Domain sharding_domain;
      if (shard_space.exists() && (shard_space != handle))
        runtime->find_domain(shard_space, sharding_domain);
      else
        sharding_domain = local_space;
      if (!func->functor->is_invertible())
      {
        const size_t max_size = get_volume();
        for (Realm::IndexSpaceIterator<DIM, T> rect_itr(local_space);
             rect_itr.valid; rect_itr.step())
        {
          for (Realm::PointInRectIterator<DIM, T> itr(rect_itr.rect); itr.valid;
               itr.step())
          {
            const ShardID point_shard = func->find_owner(
                DomainPoint(Point<DIM, T>(itr.p)), sharding_domain);
            if (range_shards.insert(point_shard).second &&
                (range_shards.size() == max_size))
              break;
          }
          if (range_shards.size() == max_size)
            break;
        }
      }
      else
      {
        for (ShardID shard = 0; shard < shard_points.size(); shard++)
        {
          std::vector<DomainPoint> domain_points;
          if (func->use_points)
            func->functor->invert_points(
                shard_points[shard], shard_points, shard_domain,
                Domain(local_space), sharding_domain, domain_points);
          else
            func->functor->invert(
                shard, Domain(local_space), sharding_domain,
                shard_points.size(), domain_points);
          if (!domain_points.empty())
            range_shards.insert(shard);
        }
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexSpaceNodeT<DIM, T>::has_shard_participants(
        ShardingFunction* func, ShardID shard, IndexSpace shard_space,
        const std::vector<DomainPoint>& shard_points,
        const Domain& shard_domain)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> local_space = get_tight_index_space();
      Domain sharding_domain;
      if (shard_space.exists() && (shard_space != handle))
        runtime->find_domain(shard_space, sharding_domain);
      else
        sharding_domain = local_space;
      if (!func->functor->is_invertible())
      {
        for (Realm::IndexSpaceIterator<DIM, T> rect_itr(local_space);
             rect_itr.valid; rect_itr.step())
        {
          for (Realm::PointInRectIterator<DIM, T> itr(rect_itr.rect); itr.valid;
               itr.step())
          {
            const ShardID point_shard = func->find_owner(
                DomainPoint(Point<DIM, T>(itr.p)), sharding_domain);
            if (point_shard == shard)
              return true;
          }
        }
        return false;
      }
      else
      {
        std::vector<DomainPoint> domain_points;
        if (func->use_points)
          func->functor->invert_points(
              shard_points[shard], shard_points, shard_domain,
              Domain(local_space), sharding_domain, domain_points);
        else
          func->functor->invert(
              shard, sharding_domain, Domain(local_space), shard_points.size(),
              domain_points);
        return !domain_points.empty();
      }
    }

    /////////////////////////////////////////////////////////////
    // Templated Linearized Color Space
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ColorSpaceLinearizationT<DIM, T>::ColorSpaceLinearizationT(
        const DomainT<DIM, T>& domain)
    //--------------------------------------------------------------------------
    {
      // Check for the common case of a dense color space that we can traverse
      // with a single Morton curve
      if (domain.dense())
      {
        unsigned interesting_count = 0;
        int interesting_dims[DIM] = {-1};
        size_t largest_extent = 0;
        for (int i = 0; i < DIM; i++)
        {
          size_t extent = (domain.bounds.hi[i] - domain.bounds.lo[i]) + 1;
          legion_assert(extent > 0);
          legion_assert(extent < SIZE_MAX);
          if (extent == 1)
            continue;
          interesting_dims[interesting_count++] = i;
          if (largest_extent < extent)
            largest_extent = extent;
        }
        if ((interesting_count == 0) || (interesting_count == 1))
        {
          // This is a rectangle that represents a single point or a
          // "pencil" in N-dimensions and therefore doesn't need a Morton curve
          morton_tiles.emplace_back(new MortonTile(
              domain.bounds, interesting_count, interesting_dims, 0 /*order*/));
          kdtree = nullptr;
          return;
        }
        // Find the least power of 2 >= extent
        unsigned power2 = largest_extent - 1;
        constexpr unsigned log2bits = STATIC_LOG2(8 * sizeof(power2));
        for (unsigned idx = 0; idx < log2bits; idx++)
          power2 = power2 | (power2 >> (1 << idx));
        power2++;
        // Take the log to get the number of bits to represent it
        unsigned order = STATIC_LOG2(power2);
        // Check to see if we can fit this in the available bits
        const size_t max_morton = 8 * sizeof(LegionColor) / interesting_count;
        if (order <= max_morton)
        {
          // It fits so we just need a single MortonTile
          morton_tiles.emplace_back(new MortonTile(
              domain.bounds, interesting_count, interesting_dims, order));
          kdtree = nullptr;
          return;
        }
      }
      // Iterate over the rectangles of the domain
      std::vector<std::pair<Rect<DIM, T>, MortonTile*> > tiles;
      for (RectInDomainIterator<DIM, T> itr(domain); itr(); itr++)
      {
        // Find the extent of the smallest dimension of the rectangle
        // that is > 1. Any dimensions that have extent one are not interesting
        // and will be ignored by the Morton curve
        unsigned interesting_count = 0;
        int interesting_dims[DIM] = {-1};
        size_t smallest_extent = SIZE_MAX;
        for (int i = 0; i < DIM; i++)
        {
          size_t extent = (itr->hi[i] - itr->lo[i]) + 1;
          legion_assert(extent > 0);
          legion_assert(extent < SIZE_MAX);
          if (extent == 1)
            continue;
          interesting_dims[interesting_count++] = i;
          if (extent < smallest_extent)
            smallest_extent = extent;
        }
        if ((interesting_count == 0) || (interesting_count == 1))
        {
          // This is a rectangle that represents a single point or a
          // "pencil" in N-dimensions and therefore doesn't need a Morton curve
          tiles.emplace_back(std::make_pair(
              *itr,
              new MortonTile(
                  *itr, interesting_count, interesting_dims, 0 /*order*/)));
          continue;
        }
        // Find the least power of 2 >= extent
        size_t power2 = smallest_extent - 1;
        constexpr unsigned log2bits = STATIC_LOG2(8 * sizeof(power2));
        for (unsigned idx = 0; idx < log2bits; idx++)
          power2 = power2 | (power2 >> (1 << idx));
        power2++;
        // Take the log to get the number of bits to represent it
        unsigned order = STATIC_LOG2(power2);
        // For small dimensions over-approximating is not too bad, but in
        // larger dimensions it can become expensive as the amount of waste
        // is proportion to 2^DIM, so we deicde that for more than four
        // dimensions we we'll look for the largest power of 2 <= extent
        if (DIM > 4)
        {
          // This is the least power of 2 >= extent, check if it is the
          // same as the extent, if not subtract by one to get the
          // largest power of 2 <= the extent
          legion_assert(smallest_extent <= (1ULL << order));
          if (smallest_extent != (1ULL << order))
            order--;
        }
        // If this is bigger than the largest order we support for the
        // given number of interesting dimensions then bound it
        const size_t max_morton = 8 * sizeof(LegionColor) / interesting_count;
        if (order > max_morton)
          order = max_morton;
        // Tile the rectangle
        // We could do this in a Morton-ordered way too, but for now we're
        // just going to let the KD-tree figure out the right way to sort
        // things in the case that we have to make lots of tiles
        // The KD-tree sorting algorithm should be good enough to give us
        // locality where we actually need it
        Point<DIM, T> strides = Point<DIM, T>::ZEROES();
        for (unsigned idx = 0; idx < interesting_count; idx++)
          strides[interesting_dims[idx]] = (1 << order);
        Point<DIM, T> lower = itr->lo;
        bool done = false;
        while (!done)
        {
          Rect<DIM, T> next(lower, lower + strides);
          if (interesting_count < DIM)
          {
            for (unsigned idx = 0; idx < interesting_count; idx++)
              next.hi[interesting_dims[idx]] -= 1;
          }
          else
            next.hi -= Point<DIM, T>::ONES();
          next = itr->intersection(next);
          legion_assert(next.volume() > 0);
          tiles.emplace_back(std::make_pair(
              next, new MortonTile(
                        next, interesting_count, interesting_dims, order)));
          done = true;
          for (unsigned idx = 0; idx < interesting_count; idx++)
          {
            int dim = interesting_dims[idx];
            lower[dim] += strides[dim];
            if (lower[dim] <= itr->hi[dim])
            {
              done = false;
              break;
            }
            // Otherwise reset this dimension and ripple-carry add
            lower[dim] = itr->lo[dim];
          }
        }
      }
      // Put the Morton Tiles in a KD-tree for fast lookups
      kdtree = new KDNode<DIM, T, MortonTile*>(domain.bounds, tiles);
      // Assign an order to the Morton Tiles based on their order in the
      // KD-tree which should give us good locality between the tiles
      kdtree->record_inorder_traversal(morton_tiles);
      // Now we can go through and compute the color offsets for each tile
      LegionColor offset = 0;
      color_offsets.resize(morton_tiles.size());
      for (unsigned idx = 0; idx < morton_tiles.size(); idx++)
      {
        color_offsets[idx] = offset;
        MortonTile* tile = morton_tiles[idx];
        tile->index = idx;
        LegionColor new_offset = offset;
        if (tile->morton_order == 0)
        {
          if (tile->interesting_count == 1)
          {
            int dim = tile->interesting_dims[0];
            new_offset += ((tile->bounds.hi[dim] - tile->bounds.lo[dim]) + 1);
          }
          else  // single element
            new_offset++;
        }
        else
          new_offset +=
              (1ULL << (tile->morton_order * tile->interesting_count));
        // Check for overflow which would be very bad
        if (new_offset <= offset)
        {
          Fatal fatal;
          fatal << "Failure during Morton tiling of color space. Please report "
                   "this issue "
                << "as a bug and provide a reproducer.";
          fatal.raise();
        }
        offset = new_offset;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ColorSpaceLinearizationT<DIM, T>::~ColorSpaceLinearizationT(void)
    //--------------------------------------------------------------------------
    {
      if (kdtree != nullptr)
        delete kdtree;
      for (unsigned idx = 0; idx < morton_tiles.size(); idx++)
        delete morton_tiles[idx];
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor ColorSpaceLinearizationT<DIM, T>::get_max_linearized_color(
        void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(!morton_tiles.empty());
      MortonTile* last = morton_tiles.back();
      LegionColor max_color = last->get_max_linearized_color();
      if (!color_offsets.empty())
        max_color += color_offsets.back();
      return max_color;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor ColorSpaceLinearizationT<DIM, T>::linearize(
        const Point<DIM, T>& point) const
    //--------------------------------------------------------------------------
    {
      if (morton_tiles.size() > 1)
      {
        // Find the Morton Tile that contains the point
        MortonTile* tile = kdtree->find(point);
        legion_assert(tile != nullptr);
        return color_offsets[tile->index] + tile->linearize(point);
      }
      else
      {
        legion_assert(!morton_tiles.empty());
        MortonTile* tile = morton_tiles.front();
        return tile->linearize(point);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void ColorSpaceLinearizationT<DIM, T>::delinearize(
        LegionColor color, Point<DIM, T>& point) const
    //--------------------------------------------------------------------------
    {
      if ((morton_tiles.size() > 1) && (color > 0))
      {
        std::vector<LegionColor>::const_iterator finder =
            std::upper_bound(color_offsets.begin(), color_offsets.end(), color);
        legion_assert(finder != color_offsets.begin());
        finder = std::prev(finder);
        unsigned index = std::distance(color_offsets.begin(), finder);
        legion_assert(index < morton_tiles.size());
        legion_assert(index < color_offsets.size());
        color -= color_offsets[index];
        morton_tiles[index]->delinearize(color, point);
      }
      else
      {
        legion_assert(!morton_tiles.empty());
        MortonTile* tile = morton_tiles.front();
        tile->delinearize(color, point);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool ColorSpaceLinearizationT<DIM, T>::contains_color(
        LegionColor color) const
    //--------------------------------------------------------------------------
    {
      if ((morton_tiles.size() > 1) && (color > 0))
      {
        std::vector<LegionColor>::const_iterator finder =
            std::upper_bound(color_offsets.begin(), color_offsets.end(), color);
        legion_assert(finder != color_offsets.begin());
        finder = std::prev(finder);
        unsigned index = std::distance(color_offsets.begin(), finder);
        legion_assert(index < morton_tiles.size());
        legion_assert(index < color_offsets.size());
        color -= color_offsets[index];
        return morton_tiles[index]->contains_color(color);
      }
      else
      {
        legion_assert(!morton_tiles.empty());
        MortonTile* tile = morton_tiles.front();
        return tile->contains_color(color);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t ColorSpaceLinearizationT<DIM, T>::compute_color_offset(
        LegionColor color) const
    //--------------------------------------------------------------------------
    {
      legion_assert(contains_color(color));
      if ((morton_tiles.size() > 1) && (color > 0))
      {
        std::vector<LegionColor>::const_iterator finder =
            std::upper_bound(color_offsets.begin(), color_offsets.end(), color);
        legion_assert(finder != color_offsets.begin());
        finder = std::prev(finder);
        unsigned index = std::distance(color_offsets.begin(), finder);
        legion_assert(index < morton_tiles.size());
        legion_assert(index < color_offsets.size());
        color -= color_offsets[index];
        size_t offset = morton_tiles[index]->compute_color_offset(color);
        // count all the points in the prior morton tiles
        for (unsigned idx = 0; idx < index; idx++)
          offset += morton_tiles[idx]->bounds.volume();
        return offset;
      }
      else
      {
        legion_assert(!morton_tiles.empty());
        MortonTile* tile = morton_tiles.front();
        return tile->compute_color_offset(color);
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor
        ColorSpaceLinearizationT<DIM, T>::MortonTile::get_max_linearized_color(
            void) const
    //--------------------------------------------------------------------------
    {
      if (interesting_count < 2)
        return bounds.volume();
      else
        return (1ULL << (morton_order * interesting_count));
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    LegionColor ColorSpaceLinearizationT<DIM, T>::MortonTile::linearize(
        const Point<DIM, T>& point) const
    //--------------------------------------------------------------------------
    {
      legion_assert(bounds.contains(point));
      if (morton_order == 0)
      {
        legion_assert((interesting_count == 0) || (interesting_count == 1));
        // No need for a Morton curve in these case of 0 or 1 interesting dims
        if (interesting_count == 0)
          return 0;
        return point[interesting_dims[0]] - bounds.lo[interesting_dims[0]];
      }
      else if (interesting_count < DIM)
      {
        // Slow path, not all dimensions are interesting
        // Pull them down to the localized dimensions
        unsigned coords[DIM] = {0};
        for (unsigned i = 0; i < interesting_count; i++)
          coords[i] =
              point[interesting_dims[i]] - bounds.lo[interesting_dims[i]];
        // Shift the bits for each of the coordinates
        // We could do this more efficiently by moving groups
        // of bits by the same offsets but that's more complicated
        // and error prone so we don't do it currently
        LegionColor codes[DIM] = {0};
        unsigned andbit = 1;
        unsigned shift = 0;
        for (unsigned idx = 0; idx < morton_order; idx++)
        {
          for (unsigned i = 0; i < interesting_count; i++)
            codes[i] |= (LegionColor)(coords[i] & andbit) << shift;
          andbit <<= 1;
          shift += (interesting_count - 1);
        }
        // Interleave the bits from each coordinate
        LegionColor result = 0;
        for (unsigned i = 0; i < interesting_count; i++)
          result |= (codes[i] << i);
        return result;
      }
      else
      {
        // Fast path, all dimensions are interesting
        unsigned coords[DIM];
        for (int i = 0; i < DIM; i++) coords[i] = point[i] - bounds.lo[i];
        // Shift the bits for each of the coordinates
        // We could do this more efficiently by moving groups
        // of bits by the same offsets but that's more complicated
        // and error prone so we don't do it currently
        LegionColor codes[DIM] = {0};
        unsigned andbit = 1, shift = 0;
        for (unsigned idx = 0; idx < morton_order; idx++)
        {
          for (int i = 0; i < DIM; i++)
            codes[i] |= (LegionColor)(coords[i] & andbit) << shift;
          andbit <<= 1;
          shift += (DIM - 1);
        }
        // Interleave the bits from each coordinate
        LegionColor result = 0;
        for (int i = 0; i < DIM; i++) result |= (codes[i] << i);
        return result;
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void ColorSpaceLinearizationT<DIM, T>::MortonTile::delinearize(
        LegionColor color, Point<DIM, T>& point) const
    //--------------------------------------------------------------------------
    {
      point = Point<DIM, T>::ZEROES();
      if (morton_order == 0)
      {
        legion_assert((interesting_count == 0) || (interesting_count == 1));
        if (interesting_count == 1)
          point[interesting_dims[0]] = color;
      }
      else if (interesting_count < DIM)
      {
        // Slow path, not all dimensions are interesting
        unsigned coords[DIM] = {0};
        unsigned selector = 0, shift = 0;
        for (unsigned idx = 0; idx < morton_order; idx++)
        {
          for (unsigned i = 0; i < interesting_count; i++)
            coords[i] |= (color & (1ULL << (selector + i))) >> (shift + i);
          selector += interesting_count;
          shift += (interesting_count - 1);
        }
        for (unsigned i = 0; i < interesting_count; i++)
          point[interesting_dims[i]] = coords[i];
      }
      else
      {
        unsigned coords[DIM] = {0};
        unsigned selector = 0, shift = 0;
        for (unsigned idx = 0; idx < morton_order; idx++)
        {
          for (int i = 0; i < DIM; i++)
            coords[i] |= (color & (1ULL << (selector + i))) >> (shift + i);
          selector += DIM;
          shift += (DIM - 1);
        }
        for (int i = 0; i < DIM; i++) point[i] = coords[i];
      }
      point += bounds.lo;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool ColorSpaceLinearizationT<DIM, T>::MortonTile::contains_color(
        LegionColor color) const
    //--------------------------------------------------------------------------
    {
      if (get_max_linearized_color() <= color)
        return false;
      Point<DIM, T> point;
      delinearize(color, point);
      return bounds.contains(point);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    size_t ColorSpaceLinearizationT<DIM, T>::MortonTile::compute_color_offset(
        LegionColor color) const
    //--------------------------------------------------------------------------
    {
      // Scan through all the points in this tile up to the color and check
      // that they are all in bounds
      size_t offset = 0;
      for (LegionColor c = 0; c < color; c++)
        if (contains_color(c))
          offset++;
      return offset;
    }

    /////////////////////////////////////////////////////////////
    // Templated Linearized Color Space (for DIM=1)
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename T>
    ColorSpaceLinearizationT<1, T>::ColorSpaceLinearizationT(
        const DomainT<1, T>& domain)
    //--------------------------------------------------------------------------
    {
      if (!domain.dense())
      {
        std::map<T, size_t> tile_sizes;
        for (RectInDomainIterator<1, T> itr(domain); itr(); itr++)
          tile_sizes[itr->lo[0]] = (itr->hi[0] - itr->lo[0]) + 1;
        LegionColor offset = 0;
        tiles.reserve(tile_sizes.size());
        extents.reserve(tile_sizes.size());
        color_offsets.reserve(tiles.size());
        for (typename std::map<T, size_t>::const_iterator it =
                 tile_sizes.begin();
             it != tile_sizes.end(); it++)
        {
          tiles.emplace_back(it->first);
          extents.emplace_back(it->second);
          color_offsets.emplace_back(offset);
          offset += it->second;
        }
      }
      else
      {
        tiles.emplace_back(domain.bounds.lo[0]);
        extents.emplace_back(domain.bounds.volume());
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    LegionColor ColorSpaceLinearizationT<1, T>::get_max_linearized_color(
        void) const
    //--------------------------------------------------------------------------
    {
      LegionColor max_color = extents.back();
      if (!color_offsets.empty())
        max_color += color_offsets.back();
      return max_color;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    LegionColor ColorSpaceLinearizationT<1, T>::linearize(
        const Point<1, T>& point) const
    //--------------------------------------------------------------------------
    {
      if (tiles.size() > 1)
      {
        typename std::vector<T>::const_iterator finder =
            std::upper_bound(tiles.begin(), tiles.end(), point[0]);
        if (finder != tiles.begin())
        {
          finder = std::prev(finder);
          unsigned index = std::distance(tiles.begin(), finder);
          return color_offsets[index] + (point[0] - tiles[index]);
        }
      }
      legion_assert(!tiles.empty());
      return (point[0] - tiles.front());
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void ColorSpaceLinearizationT<1, T>::delinearize(
        LegionColor color, Point<1, T>& point) const
    //--------------------------------------------------------------------------
    {
      if ((tiles.size() > 1) && (color > 0))
      {
        std::vector<LegionColor>::const_iterator finder =
            std::upper_bound(color_offsets.begin(), color_offsets.end(), color);
        legion_assert(finder != color_offsets.begin());
        finder = std::prev(finder);
        unsigned index = std::distance(color_offsets.begin(), finder);
        point[0] = tiles[index] + (color - *finder);
      }
      else
        point[0] = tiles.front() + color;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    bool ColorSpaceLinearizationT<1, T>::contains_color(LegionColor color) const
    //--------------------------------------------------------------------------
    {
      return (color < get_max_linearized_color());
    }

    //--------------------------------------------------------------------------
    template<typename T>
    size_t ColorSpaceLinearizationT<1, T>::compute_color_offset(
        LegionColor color) const
    //--------------------------------------------------------------------------
    {
      // Colors are dense here so colors are their own offsets
      return color;
    }

    /////////////////////////////////////////////////////////////
    // Templated Index Partition Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartNodeT<DIM, T>::IndexPartNodeT(
        IndexPartition p, IndexSpaceNode* par, IndexSpaceNode* cs,
        LegionColor c, bool disjoint, int complete, RtEvent init,
        CollectiveMapping* map, Provenance* prov)
      : IndexPartNode(p, par, cs, c, disjoint, complete, init, map, prov),
        kd_root(nullptr), kd_remote(nullptr), dense_shard_rects(nullptr),
        sparse_shard_rects(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartNodeT<DIM, T>::IndexPartNodeT(
        IndexPartition p, IndexSpaceNode* par, IndexSpaceNode* cs,
        LegionColor c, int comp, RtEvent init, CollectiveMapping* map,
        Provenance* prov)
      : IndexPartNode(p, par, cs, c, comp, init, map, prov), kd_root(nullptr),
        kd_remote(nullptr), dense_shard_rects(nullptr),
        sparse_shard_rects(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    IndexPartNodeT<DIM, T>::~IndexPartNodeT(void)
    //--------------------------------------------------------------------------
    {
      if (kd_root != nullptr)
        delete kd_root;
      if (kd_remote != nullptr)
        delete kd_remote;
      if (dense_shard_rects != nullptr)
        delete dense_shard_rects;
      if (sparse_shard_rects != nullptr)
        delete sparse_shard_rects;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexPartNodeT<DIM, T>::find_interfering_children_kd(
        IndexSpaceExpression* expr, std::vector<LegionColor>& colors,
        bool local)
    //--------------------------------------------------------------------------
    {
      if (kd_root == nullptr)
      {
        if (total_children <= LEGION_MAX_BVH_FANOUT)
          return false;
        DomainT<DIM, T> parent_space = parent->get_tight_domain();
        if (collective_mapping == nullptr)
        {
          // No shard mapping so we can build the full kd-tree here
          std::vector<std::pair<Rect<DIM, T>, LegionColor> > bounds;
          bounds.reserve(total_children);
          for (ColorSpaceIterator itr(this); itr; itr++)
          {
            IndexSpaceNode* child = get_child(*itr);
            DomainT<DIM, T> space = child->get_tight_domain();
            if (space.empty())
              continue;
            for (RectInDomainIterator<DIM, T> it(space); it(); it++)
              bounds.emplace_back(std::make_pair(*it, *itr));
          }
          KDNode<DIM, T, LegionColor>* root =
              new KDNode<DIM, T, LegionColor>(parent_space.bounds, bounds);
          AutoLock n_lock(node_lock);
          if (kd_root == nullptr)
            kd_root = root;
          else  // Someone else beat us to it
            delete root;
        }
        else
        {
          // There is a shard-mapping so we're going to build two kd-trees
          // One for storing any local or dense rectangles from remote nodes
          // Another for upper bound rectanges of spaces from remote nodes
          // First check to see if we're the first ones here
          RtEvent wait_on;
          {
            AutoLock n_lock(node_lock);
            if (kd_remote_ready.exists() || (kd_remote != nullptr))
              wait_on = kd_remote_ready;
            else
              kd_remote_ready = Runtime::create_rt_user_event();
          }
          if (!wait_on.exists() && (kd_remote == nullptr))
          {
            const RtEvent rects_ready = request_shard_rects();
            if (rects_ready.exists() && !rects_ready.has_triggered())
              rects_ready.wait();
            // Once we get the remote rectangles we can build the kd-trees
            if (!sparse_shard_rects->empty())
            {
              // Find the nearest address spaces for each color
              std::vector<std::pair<Rect<DIM, T>, AddressSpaceID> >
                  sparse_shard_spaces;
              sparse_shard_spaces.reserve(sparse_shard_rects->size());
              LegionColor previous_color = INVALID_COLOR;
              for (typename std::vector<std::pair<Rect<DIM, T>, LegionColor> >::
                       const_iterator it = sparse_shard_rects->begin();
                   it != sparse_shard_rects->end(); it++)
              {
                if (it->second != previous_color)
                {
                  CollectiveMapping* child_mapping = nullptr;
                  sparse_shard_spaces.emplace_back(std::make_pair(
                      it->first, this->find_color_creator_space(
                                     it->second, child_mapping)));
                  if (child_mapping != nullptr)
                    delete child_mapping;
                  previous_color = it->second;
                }
                else  // colors are the same so we know address space
                  sparse_shard_spaces.emplace_back(std::make_pair(
                      it->first, sparse_shard_spaces.back().second));
              }
              kd_remote = new KDNode<DIM, T, AddressSpaceID>(
                  parent_space.bounds, sparse_shard_spaces);
            }
            // Add any local sparse spaces into the dense remote rects
            // All the local dense spaces are already included
            for (ColorSpaceIterator itr(this, true /*local only*/); itr; itr++)
            {
              IndexSpaceNode* child = get_child(*itr);
              DomainT<DIM, T> space = child->get_tight_domain();
              if (space.empty() || space.dense())
                continue;
              for (RectInDomainIterator<DIM, T> it(space); it(); it++)
                dense_shard_rects->emplace_back(std::make_pair(*it, *itr));
            }
            KDNode<DIM, T, LegionColor>* root = new KDNode<DIM, T, LegionColor>(
                parent_space.bounds, *dense_shard_rects);
            AutoLock n_lock(node_lock);
            kd_root = root;
            Runtime::trigger_event(kd_remote_ready);
            kd_remote_ready = RtUserEvent::NO_RT_USER_EVENT;
          }
          else if (wait_on.exists() && !wait_on.has_triggered())
            wait_on.wait();
        }
      }
      DomainT<DIM, T> space = expr->get_tight_domain();
      // If we have a remote kd tree then we need to query that to see if
      // we have any remote colors to include
      std::set<LegionColor> color_set;
      if ((kd_remote != nullptr) && !local)
      {
        std::set<AddressSpaceID> remote_spaces;
        for (RectInDomainIterator<DIM, T> itr(space); itr(); itr++)
          kd_remote->find_interfering(*itr, remote_spaces);
        if (!remote_spaces.empty())
        {
          RemoteKDTracker tracker;
          RtEvent remote_ready =
              tracker.find_remote_interfering(remote_spaces, handle, expr);
          for (RectInDomainIterator<DIM, T> itr(space); itr(); itr++)
            kd_root->find_interfering(*itr, color_set);
          if (remote_ready.exists() && !remote_ready.has_triggered())
            remote_ready.wait();
          tracker.get_remote_interfering(color_set);
        }
        else
        {
          for (RectInDomainIterator<DIM, T> itr(space); itr(); itr++)
            kd_root->find_interfering(*itr, color_set);
        }
      }
      else
      {
        for (RectInDomainIterator<DIM, T> itr(space); itr(); itr++)
          kd_root->find_interfering(*itr, color_set);
      }
      if (!color_set.empty())
        colors.insert(colors.end(), color_set.begin(), color_set.end());
      return true;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexPartNodeT<DIM, T>::initialize_shard_rects(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(dense_shard_rects == nullptr);
      legion_assert(sparse_shard_rects == nullptr);
      dense_shard_rects =
          new std::vector<std::pair<Rect<DIM, T>, LegionColor> >();
      sparse_shard_rects =
          new std::vector<std::pair<Rect<DIM, T>, LegionColor> >();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    bool IndexPartNodeT<DIM, T>::find_local_shard_rects(void)
    //--------------------------------------------------------------------------
    {
      std::vector<RtEvent> ready_events;
      std::vector<IndexSpaceNodeT<DIM, T>*> children;
      for (ColorSpaceIterator itr(this, true /*local only*/); itr; itr++)
      {
        IndexSpaceNodeT<DIM, T>* child =
            static_cast<IndexSpaceNodeT<DIM, T>*>(get_child(*itr));
        // We're going to exchange these data structures between all the
        // copies of this partition so we only need to record our children
        // if they're actually the owner child
        if (!child->is_owner())
          continue;
        children.emplace_back(child);
        RtEvent ready = child->get_realm_index_space_ready(true /*tight*/);
        if (ready.exists())
          ready_events.emplace_back(ready);
      }
      if (!ready_events.empty())
      {
        const RtEvent ready = Runtime::merge_events(ready_events);
        if (ready.exists() && !ready.has_triggered())
        {
          // Defer this until they're all ready
          DeferFindShardRects args(this);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_DEFERRED_PRIORITY, ready);
          return false;
        }
      }
      // All the children are ready so we can get their spaces safely
      AutoLock n_lock(node_lock);
      legion_assert(dense_shard_rects != nullptr);
      legion_assert(sparse_shard_rects != nullptr);
      unsigned logn_children = 0;
      for (typename std::vector<IndexSpaceNodeT<DIM, T>*>::const_iterator it =
               children.begin();
           it != children.end(); it++)
      {
        DomainT<DIM, T> child_space = (*it)->get_tight_index_space();
        const std::pair<Rect<DIM, T>, LegionColor> next(
            child_space.bounds, (*it)->color);
        if (!child_space.dense())
        {
          // This is a sparse shard space and it's upper bound rectangle
          // can be arbitrarily big, so we want to give it some flexibility to
          // describe itself so others can prune out queries better. Therefore
          // we ask Realm to try to compute an apprxomate covering of the
          // index space. The number of rectangles we'll let it have will be
          // O(log N) in the number of subspaces in the partition, so the
          // more subspaces there are, there more rectangles we'll allow in
          // the approximation. This will be O(N log N) total cost across
          // all the sparse subspaces. We might consider making the number
          // of rectangles O(N^(1/2)) in the future but I'm scared of the
          // scalability implications of that.
          if (logn_children == 0)
          {
            // Compute ceil(log(N)) of the total children to know how many
            // rectangles we can ask for in the approximation
            LegionColor power = 1;
            while (power < total_children)
            {
              logn_children++;
              power *= 2;
            }
          }
          std::vector<Rect<DIM, T> > covering;
          // Note we don't care about the overhead, it can't be worse
          // than the upper bound rectangle that we already have
          if ((logn_children > 1) &&
              child_space.compute_covering(
                  logn_children, std::numeric_limits<int>::max() /*overhead*/,
                  covering))
          {
            for (typename std::vector<Rect<DIM, T> >::const_iterator cit =
                     covering.begin();
                 cit != covering.end(); cit++)
              sparse_shard_rects->emplace_back(
                  std::make_pair(*cit, (*it)->color));
          }
          else  // Can just add this as the covering failed
            sparse_shard_rects->emplace_back(next);
        }
        else if (!child_space.bounds.empty())
          dense_shard_rects->emplace_back(next);
      }
      return perform_shard_rects_notification();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexPartNodeT<DIM, T>::pack_shard_rects(Serializer& rez, bool clear)
    //--------------------------------------------------------------------------
    {
      legion_assert(dense_shard_rects != nullptr);
      legion_assert(sparse_shard_rects != nullptr);
      rez.serialize<size_t>(dense_shard_rects->size());
      for (typename std::vector<
               std::pair<Rect<DIM, T>, LegionColor> >::const_iterator it =
               dense_shard_rects->begin();
           it != dense_shard_rects->end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(sparse_shard_rects->size());
      for (typename std::vector<
               std::pair<Rect<DIM, T>, LegionColor> >::const_iterator it =
               sparse_shard_rects->begin();
           it != sparse_shard_rects->end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      if (clear)
      {
        dense_shard_rects->clear();
        sparse_shard_rects->clear();
      }
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void IndexPartNodeT<DIM, T>::unpack_shard_rects(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      legion_assert(dense_shard_rects != nullptr);
      legion_assert(sparse_shard_rects != nullptr);
      size_t num_dense;
      derez.deserialize(num_dense);
      if (num_dense > 0)
      {
        unsigned offset = dense_shard_rects->size();
        dense_shard_rects->resize(offset + num_dense);
        for (unsigned idx = 0; idx < num_dense; idx++)
        {
          std::pair<Rect<DIM, T>, LegionColor>& next =
              (*dense_shard_rects)[offset + idx];
          derez.deserialize(next.first);
          derez.deserialize(next.second);
        }
      }
      size_t num_sparse;
      derez.deserialize(num_sparse);
      if (num_sparse > 0)
      {
        unsigned offset = sparse_shard_rects->size();
        sparse_shard_rects->resize(offset + num_sparse);
        for (unsigned idx = 0; idx < num_sparse; idx++)
        {
          std::pair<Rect<DIM, T>, LegionColor>& next =
              (*sparse_shard_rects)[offset + idx];
          derez.deserialize(next.first);
          derez.deserialize(next.second);
        }
      }
    }
#endif  // defined(DEFINE_NT_TEMPLATES)

  }  // namespace Internal
}  // namespace Legion
