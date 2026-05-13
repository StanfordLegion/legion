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

#include "legion/analysis/versioning.h"
#include "legion/contexts/inner.h"
#include "legion/kernel/runtime.h"
#include "legion/nodes/region.h"
#include "legion/operations/operation.h"
#include "legion/utilities/serdez.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // VersionInfo
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VersionInfo::VersionInfo(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    VersionInfo::VersionInfo(const VersionInfo& rhs)
    //--------------------------------------------------------------------------
    {
      legion_assert(equivalence_sets.empty());
      legion_assert(rhs.equivalence_sets.empty());
    }

    //--------------------------------------------------------------------------
    VersionInfo::~VersionInfo(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    VersionInfo& VersionInfo::operator=(const VersionInfo& rhs)
    //--------------------------------------------------------------------------
    {
      legion_assert(equivalence_sets.empty());
      legion_assert(rhs.equivalence_sets.empty());
      return *this;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::pack_equivalence_sets(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(equivalence_sets.size());
      for (const std::pair<EquivalenceSet* const, FieldMask>& it :
           equivalence_sets)
      {
        rez.serialize(it.first->did);
        rez.serialize(it.second);
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::unpack_equivalence_sets(
        Deserializer& derez, std::set<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      size_t num_sets;
      derez.deserialize(num_sets);
      for (unsigned idx = 0; idx < num_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        FieldMask mask;
        derez.deserialize(mask);
        RtEvent ready_event;
        EquivalenceSet* set =
            runtime->find_or_request_equivalence_set(did, ready_event);
        equivalence_sets.insert(set, mask);
        if (ready_event.exists())
          ready_events.insert(ready_event);
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::record_equivalence_set(
        EquivalenceSet* set, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      equivalence_sets.insert(set, mask);
    }

    //--------------------------------------------------------------------------
    void VersionInfo::clear(void)
    //--------------------------------------------------------------------------
    {
      equivalence_sets.clear();
    }

    /////////////////////////////////////////////////////////////
    // Version Manager
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VersionManager::VersionManager(RegionTreeNode* n, ContextID c)
      : EqSetTracker(manager_lock), ctx(c), node(n)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    VersionManager::~VersionManager(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void VersionManager::perform_versioning_analysis(
        InnerContext* outermost, VersionInfo* version_info,
        RegionNode* region_node, const FieldMask& version_mask, Operation* op,
        unsigned index, unsigned parent_req_index,
        std::set<RtEvent>& ready_events, RtEvent* output_region_ready,
        bool collective_rendezvous)
    //--------------------------------------------------------------------------
    {
      legion_assert(node == region_node);
      if (output_region_ready != nullptr)
      {
        legion_assert(!collective_rendezvous);
        // This is a special case for output regions
        // Make a new equivalence set and record it in the current set
        // and then we are done, we'll register this equivalence set
        // with the EqKDTree later once we actually know the bounds
        // Note that by registering it we're allowing others to find it
        // here even before they can compute it in the EqKDTree which is
        // an important optimization for many applications
        const DistributedID did = runtime->get_available_distributed_id();
        EquivalenceSet* set = new EquivalenceSet(
            did, runtime->address_space /*logical owner*/,
            region_node->row_source, region_node->handle.get_tree_id(),
            outermost, true /*register*/);
        version_info->record_equivalence_set(set, version_mask);
        // Launch a meta-task to register this equivalence set with
        // EqKDTree once the index space domain is ready
        // Always register with the operation's immediate enclosing context
        FinalizeOutputEquivalenceSetArgs args(
            this, op->get_context(), parent_req_index, set);
        // If this an output region for an index space operation then we
        // use the parent region being ready as the precondition since we
        // won't be able to register the equivalence set until we know
        // the bounds of the parent region anyway
        const RtEvent precondition =
            (region_node->parent != nullptr) ?
                region_node->parent->parent->row_source->get_ready_event() :
                region_node->row_source->get_ready_event();
        *output_region_ready = runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY, precondition);
        AutoLock m_lock(manager_lock);
        legion_assert(version_mask * equivalence_sets.get_valid_mask());
        if (equivalence_sets.insert(set, version_mask))
          set->add_base_gc_ref(get_reference_source_kind());
        return;
      }
      // If we don't have equivalence classes for this region yet we
      // either need to compute them or request them from the owner
      FieldMask remaining_mask(version_mask);
      bool has_waiter = false;
      {
        AutoLock m_lock(manager_lock, false /*exclusive*/);
        // Check to see if any computations of equivalence sets are in progress
        // If so we'll skip out early and go down the slow path which should
        // be a fairly rare thing to do
        if (equivalence_sets_ready != nullptr)
        {
          for (const std::pair<const RtUserEvent, FieldMask>& it :
               *equivalence_sets_ready)
          {
            if (remaining_mask * it.second)
              continue;
            // Skip out earlier if we have at least one thing to wait
            // for since we're going to have to go down the slow path
            has_waiter = true;
            break;
          }
        }
        // If we have a waiter, then don't bother doing this
        if (!has_waiter)
        {
          // Get any fields that are already ready
          if ((version_info != nullptr) &&
              !(version_mask * equivalence_sets.get_valid_mask()))
            record_equivalence_sets(version_info, version_mask);
          remaining_mask -= equivalence_sets.get_valid_mask();
          // If we got all our fields then we are done
          if (!remaining_mask && !collective_rendezvous)
            return;
        }
      }
      // Retake the lock in exclusive mode and make sure we don't lose the race
      RtUserEvent compute_event;
      if (!!remaining_mask)
      {
        FieldMask waiting_mask;
        AutoLock m_lock(manager_lock);
        if (equivalence_sets_ready != nullptr)
        {
          for (const std::pair<const RtUserEvent, FieldMask>& it :
               *equivalence_sets_ready)
          {
            const FieldMask overlap = remaining_mask & it.second;
            if (!overlap)
              continue;
            ready_events.insert(it.first);
            waiting_mask |= overlap;
          }
          if (!!waiting_mask)
            remaining_mask -= waiting_mask;
        }
        // Get any fields that are already ready
        // Have to do this after looking for pending equivalence sets
        // to make sure we don't have pending outstanding requests
        if (!(remaining_mask * equivalence_sets.get_valid_mask()))
        {
          if (version_info != nullptr)
            record_equivalence_sets(version_info, remaining_mask);
          remaining_mask -= equivalence_sets.get_valid_mask();
          // If we got all our fields here and we're not waiting
          // on any other computations then we're done
          if (!remaining_mask && !waiting_mask && !collective_rendezvous)
            return;
        }
        // If we still have remaining fields then we need to
        // do this computation ourselves
        if (!!remaining_mask)
        {
          compute_event = Runtime::create_rt_user_event();
          if (equivalence_sets_ready == nullptr)
            equivalence_sets_ready = new shrt::map<RtUserEvent, FieldMask>();
          equivalence_sets_ready->insert(
              std::make_pair(compute_event, remaining_mask));
          ready_events.insert(compute_event);
          waiting_mask |= remaining_mask;
        }
        legion_assert(!!waiting_mask);
        // Record that our version info is waiting for these fields
        if (version_info != nullptr)
        {
          if (waiting_infos == nullptr)
            waiting_infos = new shrt::FieldMaskMap<VersionInfo>();
          waiting_infos->insert(version_info, waiting_mask);
        }
      }
      if (compute_event.exists())
      {
        RtEvent ready;
        if (!collective_rendezvous)
        {
          // Bounce this computation off the context so that we know
          // that we are on the right node to perform it
          std::vector<EqSetTracker*> targets(1, this);
          std::vector<AddressSpaceID> target_spaces(1, runtime->address_space);
          // Always start this computation at the operation's immediate context
          ready = op->get_context()->compute_equivalence_sets(
              parent_req_index, targets, target_spaces, runtime->address_space,
              region_node->row_source, remaining_mask);
        }
        else
          ready = op->perform_collective_versioning_analysis(
              index, region_node->handle, this, remaining_mask,
              parent_req_index);
        if (ready.exists() && !ready.has_triggered())
        {
          // Launch task to finalize the sets once they are ready
          LgFinalizeEqSetsArgs args(
              this, compute_event, op->get_context(), outermost,
              parent_req_index, region_node->row_source);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_DEFERRED_PRIORITY, ready);
        }
        else
          finalize_equivalence_sets(
              compute_event, op->get_context(), outermost, parent_req_index,
              region_node->row_source, op->get_unique_op_id());
      }
      else if (collective_rendezvous)
      {
        legion_assert(!remaining_mask);
        // Just need to rendezvous, no need to wait for any computation
        op->perform_collective_versioning_analysis(
            index, region_node->handle, this, remaining_mask, parent_req_index);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent VersionManager::finalize_output_equivalence_set(
        EquivalenceSet* set, InnerContext* enclosing, unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      FieldMask set_mask;
      {
        AutoLock m_lock(manager_lock, false /*exclusive*/);
        lng::FieldMaskMap<EquivalenceSet>::const_iterator finder =
            equivalence_sets.find(set);
        legion_assert(finder != equivalence_sets.end());
        set_mask = finder->second;
      }
      return enclosing->record_output_equivalence_set(
          this, runtime->address_space, parent_req_index, set, set_mask);
    }

    //--------------------------------------------------------------------------
    RegionTreeID VersionManager::get_region_tree_id(void) const
    //--------------------------------------------------------------------------
    {
      return node->get_tree_id();
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* VersionManager::get_tracker_expression(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(node->is_region());
      return node->as_region_node()->row_source;
    }

    //--------------------------------------------------------------------------
    void VersionManager::add_subscription_reference(unsigned count)
    //--------------------------------------------------------------------------
    {
      // This implicitly keeps us alive
      node->add_base_resource_ref(VERSION_MANAGER_REF, count);
    }

    //--------------------------------------------------------------------------
    bool VersionManager::remove_subscription_reference(unsigned count)
    //--------------------------------------------------------------------------
    {
      if (node->remove_base_resource_ref(VERSION_MANAGER_REF, count))
        delete node;
      // Never directly delete ourselves
      return false;
    }

    //--------------------------------------------------------------------------
    void VersionManager::finalize_manager(void)
    //--------------------------------------------------------------------------
    {
      // We need to remove any tracked equivalence sets that we have
      lng::FieldMaskMap<EquivalenceSet> to_remove;
      lng::map<AddressSpaceID, lng::FieldMaskMap<EqKDTree>> to_cancel;
      {
        AutoLock m_lock(manager_lock);
        // All these other resource should already be empty by the time
        // we are being finalized
        legion_assert(pending_equivalence_sets == nullptr);
        legion_assert(created_equivalence_sets == nullptr);
        legion_assert(waiting_infos == nullptr);
        legion_assert(equivalence_sets_ready == nullptr);
        if (!equivalence_sets.empty())
          to_remove.swap(equivalence_sets);
        else if (current_subscriptions.empty())
          return;
        to_cancel.swap(current_subscriptions);
      }
      legion_assert(node->is_region());
      if (!to_cancel.empty())
      {
        for (const std::pair<const AddressSpaceID, lng::FieldMaskMap<EqKDTree>>&
                 it : to_cancel)
          cancel_subscriptions(it.first, FieldMapView(it.second));
      }
      for (const std::pair<EquivalenceSet* const, FieldMask>& it : to_remove)
      {
        // This would be a valid assertion except for cases with control
        // replication where there is another node that owns the equivalence
        // set and we just happen to have a copy of it here
        // legion_assert((it.first->region_node != node) ||
        //        it.first->region_node->row_source->is_empty());
        if (it.first->remove_base_gc_ref(VERSION_MANAGER_REF))
          delete it.first;
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::FinalizeOutputEquivalenceSetArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      RtEvent done = proxy_this->finalize_output_equivalence_set(
          set, context, parent_req_index);
      if (done.exists())
        Processor::add_finish_event_precondition(done);
      if (set->remove_base_gc_ref(META_TASK_REF))
        delete set;
    }

  }  // namespace Internal
}  // namespace Legion
