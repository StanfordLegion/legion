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

#include "legion/operations/collective.h"
#include "legion/contexts/replicate.h"
#include "legion/kernel/runtime.h"
#include "legion/instances/physical.h"
#include "legion/managers/shard.h"
#include "legion/nodes/expression.h"
#include "legion/nodes/index.h"
#include "legion/operations/acquire.h"
#include "legion/operations/attach.h"
#include "legion/operations/deletion.h"
#include "legion/operations/dependent.h"
#include "legion/operations/detach.h"
#include "legion/operations/discard.h"
#include "legion/operations/fill.h"
#include "legion/operations/mapping.h"
#include "legion/operations/release.h"
#include "legion/tasks/index.h"
#include "legion/utilities/instance_set.h"
#include "legion/views/collective.h"
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // CollectiveViewCreatorBase
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CollectiveViewCreatorBase::RendezvousResult::RendezvousResult(
        CollectiveViewCreatorBase* own, const PendingRendezvousKey& k,
        const InstanceSet& insts, InnerContext* ctx)
      : owner(own), physical_ctx(ctx), key(k), instances(init_instances(insts)),
        ready(Runtime::create_rt_user_event())
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    /*static*/ op::vector<std::pair<DistributedID, FieldMask> >
        CollectiveViewCreatorBase::RendezvousResult::init_instances(
            const InstanceSet& insts)
    //--------------------------------------------------------------------------
    {
      op::vector<std::pair<DistributedID, FieldMask> > result(insts.size());
      for (unsigned idx = 0; idx < insts.size(); idx++)
      {
        const InstanceRef& ref = insts[idx];
        result[idx].first = ref.get_manager()->did;
        result[idx].second = ref.get_valid_fields();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    CollectiveViewCreatorBase::RendezvousResult::~RendezvousResult(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool CollectiveViewCreatorBase::RendezvousResult::matches(
        const InstanceSet& insts) const
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < insts.size(); idx++)
      {
        const InstanceRef& ref = insts[idx];
        if (instances[idx].first != ref.get_manager()->did)
          return false;
        if (instances[idx].second != ref.get_valid_fields())
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool CollectiveViewCreatorBase::RendezvousResult::finalize_rendezvous(
        CollectiveMapping* mapping, const FieldMapView<CollectiveResult>& views,
        const std::map<DistributedID, size_t>& counts, bool first,
        size_t local_analyses)
    //--------------------------------------------------------------------------
    {
      mapping->add_reference(target_mappings.size());
      for (unsigned idx = 0; idx < target_mappings.size(); idx++)
        *target_mappings[idx] = mapping;
      for (unsigned idx = 0; idx < target_first_locals.size(); idx++)
      {
        *target_first_locals[idx] = first;
        first = false;
      }
      std::vector<RtEvent> ready_events;
      op::vector<op::FieldMaskMap<InstanceView> > result_views(
          instances.size());
      std::map<InstanceView*, size_t> collective_arrivals;
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const DistributedID inst_did = instances[idx].first;
        const FieldMask& mask = instances[idx].second;
        for (FieldMapView<CollectiveResult>::const_iterator vit = views.begin();
             vit != views.end(); vit++)
        {
          const FieldMask overlap = mask & vit->second;
          if (!overlap)
            continue;
          if (!std::binary_search(
                  vit->first->individual_dids.begin(),
                  vit->first->individual_dids.end(), inst_did))
            continue;
          // Successfully found one of the views
          // Wait until it is safe to request it
          if (vit->first->ready_event.exists() &&
              !vit->first->ready_event.has_triggered())
            vit->first->ready_event.wait();
          if (vit->first->collective_did > 0)
          {
            RtEvent ready;
            CollectiveView* view = static_cast<CollectiveView*>(
                runtime->find_or_request_logical_view(
                    vit->first->collective_did, ready));
            if (ready.exists())
              ready_events.emplace_back(ready);
            result_views[idx].insert(view, overlap);
            // Now count how many local arrivals we have for
            // instances on the same address space
            size_t local_arrivals = 0;
            const AddressSpaceID inst_space =
                runtime->determine_owner(inst_did);
            for (const DistributedID& it : vit->first->individual_dids)
            {
              if (inst_space != runtime->determine_owner(it))
                continue;
              std::map<DistributedID, size_t>::const_iterator count_finder =
                  counts.find(it);
              if (count_finder != counts.end())
                local_arrivals += count_finder->second;
              else
                local_arrivals++;
            }
            collective_arrivals[view] = local_arrivals;
          }
          else
          {
            // No collective instance matches here so we can just
            // get the normal view for the instance
            legion_assert(vit->first->individual_dids.size() == 1);
            // Manager should still be here
            DistributedID inst_did = vit->first->individual_dids.back();
            PhysicalManager* manager = static_cast<PhysicalManager*>(
                runtime->find_distributed_collectable(inst_did));
            std::vector<PhysicalManager*> instances(1, manager);
            std::vector<IndividualView*> views;
            physical_ctx->convert_individual_views(instances, views);
            IndividualView* view = views.back();
            result_views[idx].insert(view, overlap);
            // Note we don't use the count of the instance uses here
            // but instead use our local number of local analyses since this
            // is an individual view and not a collective view
            collective_arrivals[view] = local_analyses;
          }
        }
        // Should have seen all the fields at this point
        legion_assert(result_views[idx].get_valid_mask() == mask);
      }
      for (int idx = target_views.size() - 1; idx >= 0; idx--)
      {
        if (idx == 0)
        {
          target_views[idx]->swap(result_views);
          target_arrivals[idx]->swap(collective_arrivals);
        }
        else
        {
          *target_views[idx] = result_views;
          *target_arrivals[idx] = collective_arrivals;
        }
      }
      // Remove this before we wake up the analyses
      const bool delete_now = owner->remove_pending_rendezvous(this);
      if (!ready_events.empty())
        Runtime::trigger_event(ready, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(ready);
      return delete_now;
    }

    //--------------------------------------------------------------------------
    CollectiveViewCreatorBase::CollectiveResult::CollectiveResult(
        const std::vector<DistributedID>& dids, DistributedID did,
        RtEvent ready)
      : individual_dids(dids.begin(), dids.end()), collective_did(did),
        ready_event(ready)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    CollectiveViewCreatorBase::CollectiveResult::CollectiveResult(
        std::vector<DistributedID>&& dids, DistributedID did, RtEvent ready)
      : individual_dids(dids), collective_did(did), ready_event(ready)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    CollectiveViewCreatorBase::CollectiveResult::CollectiveResult(
        DistributedID inst_did)
      : individual_dids(1, inst_did), collective_did(0)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    CollectiveViewCreatorBase::CollectiveResult::CollectiveResult(
        const std::vector<DistributedID>& dids)
      : individual_dids(dids), collective_did(0)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool CollectiveViewCreatorBase::CollectiveResult::matches(
        const std::vector<DistributedID>& dids) const
    //--------------------------------------------------------------------------
    {
      if (dids.size() != individual_dids.size())
        return false;
      for (unsigned idx = 0; idx < dids.size(); idx++)
        if (dids[idx] != individual_dids[idx])
          return false;
      return true;
    }

    //--------------------------------------------------------------------------
    CollectiveViewCreatorBase::RendezvousResult*
        CollectiveViewCreatorBase::find_or_create_rendezvous(
            unsigned index, unsigned analysis, LogicalRegion region,
            const InstanceSet& targets, InnerContext* physical_ctx,
            CollectiveMapping*& analysis_mapping, bool& first_local,
            op::vector<op::FieldMaskMap<InstanceView> >& target_views,
            std::map<InstanceView*, size_t>& collective_arrivals)
    //--------------------------------------------------------------------------
    {
      target_views.resize(targets.size());
      RendezvousResult* result = nullptr;
      // Find or create a rendezvous result and record for this context
      const PendingRendezvousKey key(index, analysis, region);
      AutoLock c_lock(collective_lock);
      std::vector<RendezvousResult*>& pending = pending_rendezvous[key];
      for (RendezvousResult* it : pending)
      {
        if (!it->matches(targets))
          continue;
        result = it;
        break;
      }
      if (result == nullptr)
      {
        result = new RendezvousResult(this, key, targets, physical_ctx);
        // Reference for pending_rendezvous
        result->add_reference();
        pending.emplace_back(result);
      }
      // Record all our targets in the result
      result->target_mappings.emplace_back(&analysis_mapping);
      result->target_first_locals.emplace_back(&first_local);
      result->target_views.emplace_back(&target_views);
      result->target_arrivals.emplace_back(&collective_arrivals);
      // Reference for ourselves
      result->add_reference();
      return result;
    }

    //--------------------------------------------------------------------------
    bool CollectiveViewCreatorBase::remove_pending_rendezvous(
        RendezvousResult* result)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(collective_lock);
      std::map<PendingRendezvousKey, std::vector<RendezvousResult*> >::iterator
          finder = pending_rendezvous.find(result->key);
      legion_assert(finder != pending_rendezvous.end());
      for (std::vector<RendezvousResult*>::iterator it = finder->second.begin();
           it != finder->second.end(); it++)
      {
        if ((*it) != result)
          continue;
        finder->second.erase(it);
        break;
      }
      if (finder->second.empty())
        pending_rendezvous.erase(finder);
      return result->remove_reference();
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveViewCreatorBase::update_groups_and_counts(
        CollectiveRendezvous& collective, DistributedID did,
        const FieldMask& mask, size_t count)
    //--------------------------------------------------------------------------
    {
      op::map<DistributedID, FieldMask>::iterator group_finder =
          collective.groups.find(did);
      if (group_finder != collective.groups.end())
      {
        if (group_finder->second == mask)
        {
          // Bump the counts
          std::map<DistributedID, size_t>::iterator count_finder =
              collective.counts.find(did);
          if (count_finder == collective.counts.end())
            collective.counts[did] = count + 1;
          else
            count_finder->second += count;
        }
        else
        {
          // If you ever hit this then heaven help you
          // The user has done something really out there and
          // is using the same instance with different sets of
          // fields for multiple point ops/tasks in the same
          // index space operation. All the tricks we do to
          // compute the collective arrivals are not going to
          // work in this case so the arrival counts will need
          // to look something like:
          //   std::map<InstanceView*,op::map<size_t,FieldMask> >
          Fatal fatal;
          fatal << "Something requested a very strange pattern for collective "
                << "instance rendezvous with different points asking to "
                << "rendezvous with different field sets on the same "
                << "physical instance. This isn't currently supported. "
                << "Please report your use case to the Legion "
                << "developer's mailing list.";
          fatal.raise();
        }
      }
      else  // No need to update counts since empty implies only one
      {
        collective.groups[did] = mask;
        if (count > 1)
          collective.counts[did] = count;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveViewCreatorBase::finalize_collective_mapping(
        CollectiveMapping* mapping, AddressSpaceID owner,
        std::vector<std::pair<AddressSpaceID, RendezvousResult*> >& results,
        const std::map<DistributedID, size_t>& counts,
        const FieldMapView<CollectiveResult>& views)
    //--------------------------------------------------------------------------
    {
      // Next figure out which targets to send the results to
      std::vector<AddressSpaceID> targets;
      if (mapping->contains(runtime->address_space))
        mapping->get_children(owner, runtime->address_space, targets);
      else
        targets.emplace_back(owner);
      // Send out the results to the next participants
      if (!targets.empty())
      {
        // These help out with building broadcasting trees so not all
        // the realm event registrations go to the same node
        std::map<CollectiveResult*, RtEvent> local_registered_events;
        std::map<CollectiveResult*, RtEvent> local_ready_events;
        for (const AddressSpaceID& target : targets)
        {
          CollectiveFinalizeMapping rez;
          {
            RezCheck z(rez);
            mapping->pack(rez);
            rez.serialize(owner);
            rez.serialize<size_t>(results.size());
            for (unsigned idx = 0; idx < results.size(); idx++)
            {
              rez.serialize(results[idx].first);
              rez.serialize(results[idx].second);
            }
            rez.serialize<size_t>(counts.size());
            for (const std::pair<const DistributedID, size_t>& count : counts)
            {
              rez.serialize(count.first);
              rez.serialize(count.second);
            }
            rez.serialize<size_t>(views.size());
            for (FieldMapView<CollectiveResult>::const_iterator vit =
                     views.begin();
                 vit != views.end(); vit++)
            {
              rez.serialize(vit->first->collective_did);
              rez.serialize<size_t>(vit->first->individual_dids.size());
              for (unsigned idx = 0; idx < vit->first->individual_dids.size();
                   idx++)
                rez.serialize(vit->first->individual_dids[idx]);
              if (!vit->first->ready_event.exists())
              {
                std::map<CollectiveResult*, RtEvent>::const_iterator finder =
                    local_ready_events.find(vit->first);
                if (finder == local_ready_events.end())
                {
                  const RtUserEvent local = Runtime::create_rt_user_event();
                  Runtime::trigger_event(local, vit->first->ready_event);
                  local_ready_events[vit->first] = local;
                  rez.serialize(local);
                }
                else
                  rez.serialize(finder->second);
              }
              else
                rez.serialize(vit->first->ready_event);
              rez.serialize(vit->second);
            }
          }
          rez.dispatch(target);
        }
      }
      // Now handle all of the local results
      if (targets.empty() || (targets.back() != owner))
      {
        std::vector<std::pair<AddressSpaceID, RendezvousResult*> >::iterator
            result_it = results.begin();
        // Skip the non-local results
        while (result_it->first != runtime->address_space) result_it++;
        legion_assert(result_it != results.end());
        // Count how many local analyses we have here in case we need it
        size_t local_analyses = 0;
        std::vector<std::pair<AddressSpaceID, RendezvousResult*> >::iterator
            local_it = result_it;
        while (local_it->first == runtime->address_space)
        {
          local_analyses += local_it->second->target_mappings.size();
          if (++local_it == results.end())
            break;
        }
        legion_assert(local_analyses > 0);
        while (result_it->first == runtime->address_space)
        {
          bool first = true;
          if (result_it->second->finalize_rendezvous(
                  mapping, views, counts, first, local_analyses))
            delete result_it->second;
          first = false;
          if (++result_it == results.end())
            break;
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/
    void CollectiveViewCreatorBase::handle_finalize_collective_mapping(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t num_spaces;
      derez.deserialize(num_spaces);
      CollectiveMapping* mapping = new CollectiveMapping(derez, num_spaces);
      mapping->add_reference();
      AddressSpaceID owner;
      derez.deserialize(owner);
      size_t num_results;
      derez.deserialize(num_results);
      std::vector<std::pair<AddressSpaceID, RendezvousResult*> > results(
          num_results);
      for (unsigned idx = 0; idx < num_results; idx++)
      {
        derez.deserialize(results[idx].first);
        derez.deserialize(results[idx].second);
      }
      size_t num_counts;
      derez.deserialize(num_counts);
      std::map<DistributedID, size_t> counts;
      for (unsigned idx = 0; idx < num_counts; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        derez.deserialize(counts[did]);
      }
      size_t num_views;
      derez.deserialize(num_views);
      local::FieldMaskMap<CollectiveResult> views;
      for (unsigned idx = 0; idx < num_views; idx++)
      {
        DistributedID collective_did;
        derez.deserialize(collective_did);
        size_t num_dids;
        derez.deserialize(num_dids);
        std::vector<DistributedID> individual_dids(num_dids);
        for (unsigned idx = 0; idx < num_dids; idx++)
          derez.deserialize(individual_dids[idx]);
        RtEvent view_ready;
        derez.deserialize(view_ready);
        CollectiveResult* view = new CollectiveResult(
            std::move(individual_dids), collective_did, view_ready);
        view->add_reference();
        FieldMask mask;
        derez.deserialize(mask);
        views.insert(view, mask);
      }
      finalize_collective_mapping(mapping, owner, results, counts, views);
      for (local::FieldMaskMap<CollectiveResult>::const_iterator it =
               views.begin();
           it != views.end(); it++)
        if (it->first->remove_reference())
          delete it->first;
      if (mapping->remove_reference())
        delete mapping;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveFinalizeMapping::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      CollectiveViewCreatorBase::handle_finalize_collective_mapping(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveViewCreatorBase::pack_collective_rendezvous(
        Serializer& rez,
        const std::map<LogicalRegion, CollectiveRendezvous>& rendezvous)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(rendezvous.size());
      for (const std::pair<const LogicalRegion, CollectiveRendezvous>& rit :
           rendezvous)
      {
        rez.serialize(rit.first);
        rez.serialize(rit.second.results.size());
        for (const std::pair<AddressSpaceID, RendezvousResult*>& it :
             rit.second.results)
        {
          rez.serialize(it.first);
          rez.serialize(it.second);
        }
        rez.serialize<size_t>(rit.second.groups.size());
        for (op::map<DistributedID, FieldMask>::const_iterator it =
                 rit.second.groups.begin();
             it != rit.second.groups.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        rez.serialize<size_t>(rit.second.counts.size());
        for (const std::pair<const DistributedID, size_t>& it :
             rit.second.counts)
        {
          rez.serialize(it.first);
          rez.serialize(it.second);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveViewCreatorBase::unpack_collective_rendezvous(
        Deserializer& derez,
        std::map<LogicalRegion, CollectiveRendezvous>& rendezvous)
    //--------------------------------------------------------------------------
    {
      size_t num_regions;
      derez.deserialize(num_regions);
      for (unsigned idx1 = 0; idx1 < num_regions; idx1++)
      {
        LogicalRegion region;
        derez.deserialize(region);
        std::map<LogicalRegion, CollectiveRendezvous>::iterator region_finder =
            rendezvous.find(region);
        if (region_finder != rendezvous.end())
        {
          // need to unpack out of place to do the merge
          size_t num_results;
          derez.deserialize(num_results);
          const unsigned offset = region_finder->second.results.size();
          region_finder->second.results.resize(offset + num_results);
          for (unsigned idx2 = 0; idx2 < num_results; idx2++)
          {
            derez.deserialize(
                region_finder->second.results[offset + idx2].first);
            derez.deserialize(
                region_finder->second.results[offset + idx2].second);
          }
          // unpack these and then do the merge
          op::map<DistributedID, FieldMask> groups;
          std::map<DistributedID, size_t> counts;
          size_t num_groups;
          derez.deserialize(num_groups);
          for (unsigned idx2 = 0; idx2 < num_groups; idx2++)
          {
            DistributedID did;
            derez.deserialize(did);
            derez.deserialize(groups[did]);
          }
          size_t num_counts;
          derez.deserialize(num_counts);
          for (unsigned idx2 = 0; idx2 < num_counts; idx2++)
          {
            DistributedID did;
            derez.deserialize(did);
            derez.deserialize(counts[did]);
          }
          // merge the groups and counts into the existing case
          for (op::map<DistributedID, FieldMask>::const_iterator it =
                   groups.begin();
               it != groups.end(); it++)
          {
            std::map<DistributedID, size_t>::iterator count_finder =
                counts.find(it->first);
            if (count_finder == counts.end())
              update_groups_and_counts(
                  region_finder->second, it->first, it->second);
            else
              update_groups_and_counts(
                  region_finder->second, it->first, it->second,
                  count_finder->second);
          }
        }
        else
        {
          // unpack in place since we know it doesn't exist yet
          CollectiveRendezvous& new_rendezvous = rendezvous[region];
          size_t num_results;
          derez.deserialize(num_results);
          new_rendezvous.results.resize(num_results);
          for (unsigned idx2 = 0; idx2 < num_results; idx2++)
          {
            derez.deserialize(new_rendezvous.results[idx2].first);
            derez.deserialize(new_rendezvous.results[idx2].second);
          }
          size_t num_groups;
          derez.deserialize(num_groups);
          for (unsigned idx2 = 0; idx2 < num_groups; idx2++)
          {
            DistributedID did;
            derez.deserialize(did);
            derez.deserialize(new_rendezvous.groups[did]);
          }
          size_t num_counts;
          derez.deserialize(num_counts);
          for (unsigned idx2 = 0; idx2 < num_counts; idx2++)
          {
            DistributedID did;
            derez.deserialize(did);
            derez.deserialize(new_rendezvous.counts[did]);
          }
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // CollectiveVersioningBase
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveVersioningBase::pack_collective_versioning(
        Serializer& rez,
        const op::map<LogicalRegion, RegionVersioning>& pending_versions)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(pending_versions.size());
      for (op::map<LogicalRegion, RegionVersioning>::const_iterator pit =
               pending_versions.begin();
           pit != pending_versions.end(); pit++)
      {
        legion_assert(pit->second.ready_event.exists());
        rez.serialize(pit->first);
        rez.serialize(pit->second.ready_event);
        rez.serialize<size_t>(pit->second.trackers.size());
        for (op::map<std::pair<AddressSpaceID, EqSetTracker*>, FieldMask>::
                 const_iterator it = pit->second.trackers.begin();
             it != pit->second.trackers.end(); it++)
        {
          rez.serialize(it->first.first);
          rez.serialize(it->first.second);
          rez.serialize(it->second);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ bool CollectiveVersioningBase::unpack_collective_versioning(
        Deserializer& derez,
        op::map<LogicalRegion, RegionVersioning>& pending_versions)
    //--------------------------------------------------------------------------
    {
      size_t num_regions;
      derez.deserialize(num_regions);
      for (unsigned idx1 = 0; idx1 < num_regions; idx1++)
      {
        LogicalRegion region;
        derez.deserialize(region);
        RtUserEvent ready_event;
        derez.deserialize(ready_event);
        op::map<LogicalRegion, RegionVersioning>::iterator finder =
            pending_versions.find(region);
        if (finder == pending_versions.end())
        {
          finder = pending_versions
                       .emplace(std::make_pair(region, RegionVersioning()))
                       .first;
          finder->second.ready_event = ready_event;
        }
        else
          Runtime::trigger_event(ready_event, finder->second.ready_event);
        size_t num_trackers;
        derez.deserialize(num_trackers);
        for (unsigned idx2 = 0; idx2 < num_trackers; idx2++)
        {
          std::pair<AddressSpaceID, EqSetTracker*> key;
          derez.deserialize(key.first);
          derez.deserialize(key.second);
          legion_assert(
              finder->second.trackers.find(key) ==
              finder->second.trackers.end());
          derez.deserialize(finder->second.trackers[key]);
        }
      }
      return (num_regions > 0);
    }

    /////////////////////////////////////////////////////////////
    // CollectiveVersioning
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename OP>
    void CollectiveVersioning<OP>::activate(void)
    //--------------------------------------------------------------------------
    {
      OP::activate();
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void CollectiveVersioning<OP>::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      legion_assert(pending_versioning.empty());
      OP::deactivate(freeop);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    RtEvent CollectiveVersioning<OP>::rendezvous_collective_versioning_analysis(
        unsigned index, LogicalRegion handle, EqSetTracker* tracker,
        AddressSpaceID space, const FieldMask& mask, unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      RtEvent result;
      bool done = false;
      op::map<LogicalRegion, RegionVersioning> to_perform;
      {
        AutoLock v_lock(versioning_lock);
        std::map<unsigned, PendingVersioning>::iterator finder =
            pending_versioning.find(index);
        if (finder == pending_versioning.end())
        {
          finder = pending_versioning
                       .insert(std::make_pair(index, PendingVersioning()))
                       .first;
          finder->second.remaining_arrivals = this->get_collective_points();
        }
        // Only need to record this target if it has actual fields
        if (!!mask)
        {
          std::map<LogicalRegion, RegionVersioning>::iterator region_finder =
              finder->second.region_versioning.find(handle);
          if (region_finder == finder->second.region_versioning.end())
          {
            region_finder =
                finder->second.region_versioning
                    .insert(std::make_pair(handle, RegionVersioning()))
                    .first;
            region_finder->second.ready_event = Runtime::create_rt_user_event();
          }
          const std::pair<AddressSpaceID, EqSetTracker*> key(space, tracker);
          legion_assert(
              (region_finder->second.trackers.find(key) ==
               region_finder->second.trackers.end()) ||
              (region_finder->second.trackers[key] == mask));
          region_finder->second.trackers.emplace(std::make_pair(key, mask));
          result = region_finder->second.ready_event;
        }
        legion_assert(finder->second.remaining_arrivals > 0);
        if ((--finder->second.remaining_arrivals) == 0)
        {
          done = true;
          to_perform.swap(finder->second.region_versioning);
          pending_versioning.erase(finder);
        }
      }
      if (done)
        finalize_collective_versioning_analysis(
            index, parent_req_index, to_perform);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void CollectiveVersioning<OP>::rendezvous_collective_versioning_analysis(
        unsigned index, unsigned parent_req_index,
        op::map<LogicalRegion, RegionVersioning>& to_perform)
    //--------------------------------------------------------------------------
    {
      bool done = false;
      {
        AutoLock v_lock(versioning_lock);
        std::map<unsigned, PendingVersioning>::iterator finder =
            pending_versioning.find(index);
        if (finder == pending_versioning.end())
        {
          finder = pending_versioning
                       .insert(std::make_pair(index, PendingVersioning()))
                       .first;
          finder->second.remaining_arrivals = this->get_collective_points();
        }
        if (!finder->second.region_versioning.empty())
        {
          for (op::map<LogicalRegion, RegionVersioning>::iterator pit =
                   to_perform.begin();
               pit != to_perform.end();
               /*nothing*/)
          {
            std::map<LogicalRegion, RegionVersioning>::iterator region_finder =
                finder->second.region_versioning.find(pit->first);
            if (region_finder == finder->second.region_versioning.end())
            {
              // Doesn't exist so copy it over
              RegionVersioning& versioning =
                  finder->second.region_versioning[pit->first];
              versioning.trackers.swap(pit->second.trackers);
              versioning.ready_event = pit->second.ready_event;
            }
            else
            {
              // Merge everything
              for (op::map<
                       std::pair<AddressSpaceID, EqSetTracker*>,
                       FieldMask>::const_iterator it =
                       pit->second.trackers.begin();
                   it != pit->second.trackers.end(); it++)
              {
                op::map<std::pair<AddressSpaceID, EqSetTracker*>, FieldMask>::
                    iterator tracker_finder =
                        region_finder->second.trackers.find(it->first);
                if (tracker_finder == region_finder->second.trackers.end())
                  region_finder->second.trackers.emplace(*it);
                else
                  tracker_finder->second |= it->second;
              }
              Runtime::trigger_event(
                  pit->second.ready_event, region_finder->second.ready_event);
            }
            op::map<LogicalRegion, RegionVersioning>::iterator delete_it =
                pit++;
            to_perform.erase(delete_it);
          }
        }
        else
          finder->second.region_versioning.swap(to_perform);
        legion_assert(to_perform.empty());
        legion_assert(finder->second.remaining_arrivals > 0);
        if ((--finder->second.remaining_arrivals) == 0)
        {
          done = true;
          to_perform.swap(finder->second.region_versioning);
          pending_versioning.erase(finder);
        }
      }
      if (done)
        finalize_collective_versioning_analysis(
            index, parent_req_index, to_perform);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void CollectiveVersioning<OP>::finalize_collective_versioning_analysis(
        unsigned index, unsigned parent_req_index,
        op::map<LogicalRegion, RegionVersioning>& to_perform)
    //--------------------------------------------------------------------------
    {
      InnerContext* context = this->get_context();
      for (op::map<LogicalRegion, RegionVersioning>::const_iterator pit =
               to_perform.begin();
           pit != to_perform.end(); pit++)
      {
        IndexSpaceNode* expr = runtime->get_node(pit->first.get_index_space());
        std::vector<RtEvent> preconditions;
        local::list<FieldSet<std::pair<AddressSpaceID, EqSetTracker*> > >
            fields;
        compute_field_sets(FieldMask(), MapView(pit->second.trackers), fields);
        // Be a bit careful, there is an important heuristic here
        // This heuristic decides which of the targets will be the one
        // responsible for creating any new equivalence sets if there
        // are any to be created. We do this by determining which of
        // the targets is closest to the owner node of the logical
        // region. This hopefully will spread out creations across
        // the targets in a resonable way without overly relying on
        // making these equivalence sets on smaller nodes
        // Use the index space to determine the owner since the region
        // version of this only gives you the owner for the region tree
        // If you change this heuristic make sure you update it in
        // SingleTask::perform_replicate_collective_vesioning too
        const AddressSpaceID region_owner_space =
            IndexSpaceNode::get_owner_space(pit->first.get_index_space());
        for (local::list<FieldSet<std::pair<AddressSpaceID, EqSetTracker*> > >::
                 const_iterator fit = fields.begin();
             fit != fields.end(); fit++)
        {
          std::vector<EqSetTracker*> targets;
          std::vector<AddressSpaceID> target_spaces;
          targets.reserve(fit->elements.size());
          target_spaces.reserve(fit->elements.size());
          for (const std::pair<AddressSpaceID, EqSetTracker*>& it :
               fit->elements)
          {
            targets.emplace_back(it.second);
            target_spaces.emplace_back(it.first);
          }
          RtEvent precondition;
          if (!std::binary_search(
                  target_spaces.begin(), target_spaces.end(),
                  region_owner_space))
          {
            const CollectiveMapping mapping(
                target_spaces, runtime->legion_collective_radix);
            const AddressSpaceID creation_origin =
                mapping.find_nearest(region_owner_space);
            precondition = context->compute_equivalence_sets(
                parent_req_index, targets, target_spaces, creation_origin, expr,
                fit->set_mask);
          }
          else
            precondition = context->compute_equivalence_sets(
                parent_req_index, targets, target_spaces, region_owner_space,
                expr, fit->set_mask);
          if (precondition.exists())
            preconditions.emplace_back(precondition);
        }
        legion_assert(pit->second.ready_event.exists());
        if (!preconditions.empty())
          Runtime::trigger_event(
              pit->second.ready_event, Runtime::merge_events(preconditions));
        else
          Runtime::trigger_event(pit->second.ready_event);
      }
    }

    // Explicit instantiations
    template class CollectiveVersioning<Operation>;
    template class CollectiveVersioning<MapOp>;
    template class CollectiveVersioning<FillOp>;
    template class CollectiveVersioning<AttachOp>;
    template class CollectiveVersioning<DetachOp>;
    template class CollectiveVersioning<AcquireOp>;
    template class CollectiveVersioning<ReleaseOp>;
    template class CollectiveVersioning<DiscardOp>;
    template class CollectiveVersioning<DependentPartitionOp>;
    template class CollectiveVersioning<DeletionOp>;
    template class CollectiveVersioning<TaskOp>;
    template class CollectiveVersioning<CollectiveHelperOp>;

    /////////////////////////////////////////////////////////////
    // CollectiveViewCreator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename OP>
    void CollectiveViewCreator<OP>::activate(void)
    //--------------------------------------------------------------------------
    {
      OP::activate();
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void CollectiveViewCreator<OP>::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      legion_assert(pending_rendezvous.empty());
      legion_assert(pending_collectives.empty());
      OP::deactivate(freeop);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    RtEvent CollectiveViewCreator<OP>::convert_collective_views(
        unsigned index, unsigned analysis, LogicalRegion region,
        const InstanceSet& targets, InnerContext* physical_ctx,
        CollectiveMapping*& analysis_mapping, bool& first_local,
        op::vector<op::FieldMaskMap<InstanceView> >& target_views,
        std::map<InstanceView*, size_t>& collective_arrivals)
    //--------------------------------------------------------------------------
    {
      target_views.resize(targets.size());
      // Find or create a rendezvous result and for this request
      RendezvousResult* result = find_or_create_rendezvous(
          index, analysis, region, targets, physical_ctx, analysis_mapping,
          first_local, target_views, collective_arrivals);
      // Now perform the rendezvous for this result
      rendezvous_collective_mapping(
          index, analysis, region, result, runtime->address_space,
          result->instances);
      const RtEvent ready = result->ready;
      if (result->remove_reference())
        delete result;
      return ready;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void CollectiveViewCreator<OP>::rendezvous_collective_mapping(
        unsigned requirement_index, unsigned analysis, LogicalRegion region,
        RendezvousResult* result, AddressSpaceID source,
        const op::vector<std::pair<DistributedID, FieldMask> >& insts)
    //--------------------------------------------------------------------------
    {
      std::map<LogicalRegion, CollectiveRendezvous> to_construct;
      const RendezvousKey key(requirement_index, analysis);
      std::pair<AddressSpaceID, RendezvousResult*> result_key(source, result);
      {
        AutoLock c_lock(collective_lock);
        std::map<RendezvousKey, PendingCollective>::iterator finder =
            pending_collectives.find(key);
        if (finder == pending_collectives.end())
          finder =
              pending_collectives
                  .insert(std::make_pair(
                      key, PendingCollective(this->get_collective_points())))
                  .first;
        CollectiveRendezvous& collective = finder->second.rendezvous[region];
        if (!std::binary_search(
                collective.results.begin(), collective.results.end(),
                result_key))
        {
          collective.results.emplace_back(result_key);
          std::sort(collective.results.begin(), collective.results.end());
        }
        // Now update the counts for all the instances
        for (op::vector<std::pair<DistributedID, FieldMask> >::const_iterator
                 it = insts.begin();
             it != insts.end(); it++)
          update_groups_and_counts(collective, it->first, it->second);
        legion_assert(finder->second.remaining_arrivals > 0);
        if (--finder->second.remaining_arrivals == 0)
        {
          to_construct.swap(finder->second.rendezvous);
          pending_collectives.erase(finder);
        }
      }
      if (!to_construct.empty())
        construct_collective_mapping(key, to_construct);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void CollectiveViewCreator<OP>::rendezvous_collective_mapping(
        const RendezvousKey& key,
        std::map<LogicalRegion, CollectiveRendezvous>& rendezvous)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock c_lock(collective_lock);
        std::map<RendezvousKey, PendingCollective>::iterator finder =
            pending_collectives.find(key);
        if (finder == pending_collectives.end())
          finder =
              pending_collectives
                  .insert(std::make_pair(
                      key, PendingCollective(this->get_collective_points())))
                  .first;
        for (std::map<LogicalRegion, CollectiveRendezvous>::iterator rit =
                 rendezvous.begin();
             rit != rendezvous.end();
             /*nothing*/)
        {
          std::map<LogicalRegion, CollectiveRendezvous>::iterator
              region_finder = finder->second.rendezvous.find(rit->first);
          if (region_finder == finder->second.rendezvous.end())
          {
            // Doesn't exist so we can just swap everything over
            CollectiveRendezvous& region_rendezvous =
                finder->second.rendezvous[rit->first];
            region_rendezvous.results.swap(rit->second.results);
            region_rendezvous.groups.swap(rit->second.groups);
            region_rendezvous.counts.swap(rit->second.counts);
          }
          else
          {
            // Need to do the merge
            for (const std::pair<AddressSpaceID, RendezvousResult*>& it :
                 rit->second.results)
            {
              if (std::binary_search(
                      region_finder->second.results.begin(),
                      region_finder->second.results.end(), it))
                continue;
              region_finder->second.results.emplace_back(it);
              std::sort(
                  region_finder->second.results.begin(),
                  region_finder->second.results.end());
            }
            for (op::map<DistributedID, FieldMask>::iterator it =
                     rit->second.groups.begin();
                 it != rit->second.groups.end(); it++)
            {
              std::map<DistributedID, size_t>::const_iterator count_finder =
                  rit->second.counts.find(it->first);
              if (count_finder == rit->second.counts.end())
                update_groups_and_counts(
                    region_finder->second, it->first, it->second);
              else
                update_groups_and_counts(
                    region_finder->second, it->first, it->second,
                    count_finder->second);
            }
          }
          std::map<LogicalRegion, CollectiveRendezvous>::iterator delete_it =
              rit++;
          rendezvous.erase(delete_it);
        }
        legion_assert(rendezvous.empty());
        legion_assert(finder->second.remaining_arrivals > 0);
        if (--finder->second.remaining_arrivals == 0)
        {
          rendezvous.swap(finder->second.rendezvous);
          pending_collectives.erase(finder);
        }
      }
      if (!rendezvous.empty())
        construct_collective_mapping(key, rendezvous);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void CollectiveViewCreator<OP>::construct_collective_mapping(
        const RendezvousKey& key,
        std::map<LogicalRegion, CollectiveRendezvous>& rendezvous)
    //--------------------------------------------------------------------------
    {
      const RegionTreeID tid = rendezvous.begin()->first.get_tree_id();
      InnerContext* physical_ctx =
          this->find_physical_context(key.region_index);
      for (std::pair<const LogicalRegion, CollectiveRendezvous>&
               rendezvous_pair : rendezvous)
      {
        // All the regions should be from the same region tree
        legion_assert(tid == rendezvous_pair.first.get_tree_id());
        local::list<FieldSet<DistributedID> > field_sets;
        compute_field_sets(
            FieldMask(), MapView(rendezvous_pair.second.groups), field_sets);
        local::FieldMaskMap<CollectiveResult> results;
        std::vector<RtEvent> ready_events;
        for (local::list<FieldSet<DistributedID> >::const_iterator it =
                 field_sets.begin();
             it != field_sets.end(); it++)
        {
          legion_assert(!it->elements.empty());
          if (it->elements.size() > 1)
          {
            const std::vector<DistributedID> instances(
                it->elements.begin(), it->elements.end());
            // Use the right physical context to find or create the collective
            // view in case there have been any virtual mappings
            RtEvent ready;
            InnerContext::CollectiveResult* result =
                physical_ctx->find_or_create_collective_view(
                    tid, instances, ready);
            if (ready.exists())
              ready_events.emplace_back(ready);
            // References already added by method so deduplicate references
            if (!results.insert(result, it->set_mask))
              result->remove_reference();
          }
          else
          {
            // If there is just one instance then we send back a
            // collective did of zero to indicate to just use the
            // normal single view
            CollectiveResult* result =
                new CollectiveResult(*it->elements.begin());
            if (results.insert(result, it->set_mask))
              result->add_reference();
          }
        }
        // Send the resulting views back out to all the RendezvousResults
        std::sort(
            rendezvous_pair.second.results.begin(),
            rendezvous_pair.second.results.end());
        // First build the collective mapping
        std::vector<AddressSpaceID> unique_spaces;

        for (const std::pair<AddressSpaceID, RendezvousResult*>& it :
             rendezvous_pair.second.results)
        {
          if (unique_spaces.empty() || (unique_spaces.back() != it.first))
            unique_spaces.emplace_back(it.first);
        }
        CollectiveMapping* mapping = new CollectiveMapping(
            unique_spaces, runtime->legion_collective_radix);
        mapping->add_reference();
        // Determine the owner for this collective mapping
        AddressSpaceID owner =
            mapping->contains(runtime->address_space) ?
                runtime->address_space :
                mapping->find_nearest(runtime->address_space);
        // Make sure all the results are ready before we send them out
        if (!ready_events.empty())
        {
          const RtEvent wait_on = Runtime::merge_events(ready_events);
          if (wait_on.exists() && !wait_on.has_triggered())
            wait_on.wait();
        }
        finalize_collective_mapping(
            mapping, owner, rendezvous_pair.second.results,
            rendezvous_pair.second.counts, results);
        if (mapping->remove_reference())
          delete mapping;
        for (local::FieldMaskMap<CollectiveResult>::const_iterator it =
                 results.begin();
             it != results.end(); it++)
          if (it->first->remove_reference())
            delete it->first;
      }
    }

    // Explicit instantiations
    template class CollectiveViewCreator<Operation>;
    template class CollectiveViewCreator<MapOp>;
    template class CollectiveViewCreator<AttachOp>;
    template class CollectiveViewCreator<DetachOp>;
    template class CollectiveViewCreator<AcquireOp>;
    template class CollectiveViewCreator<ReleaseOp>;
    template class CollectiveViewCreator<DependentPartitionOp>;
    template class CollectiveViewCreator<TaskOp>;
    template class CollectiveViewCreator<CollectiveHelperOp>;

    /////////////////////////////////////////////////////////////
    // Collective View Rendezvous
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CollectiveViewRendezvous::CollectiveViewRendezvous(
        CollectiveID id, ReplicateContext* ctx, Operation* o, Finalizer* f,
        const RendezvousKey& k, RegionTreeID tid)
      : GatherCollective(
            ctx, id, ctx->shard_manager->find_collective_owner(tid)),
        key(k), op(o), finalizer(f)
    //--------------------------------------------------------------------------
    {
      legion_assert(op != nullptr);
      legion_assert(finalizer != nullptr);
    }

    //--------------------------------------------------------------------------
    CollectiveViewRendezvous::~CollectiveViewRendezvous(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void CollectiveViewRendezvous::pack_collective(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      CollectiveViewCreatorBase::pack_collective_rendezvous(rez, rendezvous);
    }

    //--------------------------------------------------------------------------
    void CollectiveViewRendezvous::unpack_collective(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      CollectiveViewCreatorBase::unpack_collective_rendezvous(
          derez, rendezvous);
    }

    //--------------------------------------------------------------------------
    RtEvent CollectiveViewRendezvous::post_gather(void)
    //--------------------------------------------------------------------------
    {
      if (local_shard == target)
        finalizer->finalize_collective_mapping(key, rendezvous);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void CollectiveViewRendezvous::perform_rendezvous(
        std::map<LogicalRegion, CollectiveRendezvous>& to_rendezvous)
    //--------------------------------------------------------------------------
    {
      rendezvous.swap(to_rendezvous);
      perform_collective_async();
    }

    /////////////////////////////////////////////////////////////
    // Collective Versioning Rendezvous
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CollectiveVersioningRendezvous::CollectiveVersioningRendezvous(
        CollectiveID id, ReplicateContext* ctx, Operation* o, Finalizer* f,
        ShardID owner, unsigned idx)
      : GatherCollective(ctx, id, owner), op(o), finalizer(f), index(idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(op != nullptr);
      legion_assert(finalizer != nullptr);
    }

    //--------------------------------------------------------------------------
    CollectiveVersioningRendezvous::~CollectiveVersioningRendezvous(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void CollectiveVersioningRendezvous::pack_collective(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      CollectiveVersioningBase::pack_collective_versioning(
          rez, pending_versions);
      if (!pending_versions.empty())
        rez.serialize(parent_req_index);
    }

    //--------------------------------------------------------------------------
    void CollectiveVersioningRendezvous::unpack_collective(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      if (CollectiveVersioningBase::unpack_collective_versioning(
              derez, pending_versions))
        derez.deserialize(parent_req_index);
    }

    //--------------------------------------------------------------------------
    RtEvent CollectiveVersioningRendezvous::post_gather(void)
    //--------------------------------------------------------------------------
    {
      if (!pending_versions.empty() && (local_shard == target))
        finalizer->finalize_collective_versioning(
            index, parent_req_index, pending_versions);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void CollectiveVersioningRendezvous::perform_rendezvous(
        unsigned parent_index,
        op::map<LogicalRegion, RegionVersioning>& pending)
    //--------------------------------------------------------------------------
    {
      parent_req_index = parent_index;
      pending_versions.swap(pending);
      perform_collective_async();
    }

    /////////////////////////////////////////////////////////////
    // Repl Collective Versioning
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename OP>
    ReplCollectiveVersioning<OP>::ReplCollectiveVersioning(void) : OP()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplCollectiveVersioning<OP>::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      for (typename std::map<
               unsigned, CollectiveVersioningRendezvous*>::const_iterator it =
               collective_versioning_rendezvous.begin();
           it != collective_versioning_rendezvous.end(); it++)
        delete it->second;
      collective_versioning_rendezvous.clear();
      OP::deactivate(freeop);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplCollectiveVersioning<OP>::finalize_collective_versioning_analysis(
        unsigned index, unsigned parent_req_index,
        op::map<LogicalRegion, RegionVersioning>& to_perform)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, CollectiveVersioningRendezvous*>::const_iterator
          finder = collective_versioning_rendezvous.find(index);
      legion_assert(finder != collective_versioning_rendezvous.end());
      finder->second->perform_rendezvous(parent_req_index, to_perform);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplCollectiveVersioning<OP>::finalize_collective_versioning(
        unsigned index, unsigned parent_req_index,
        op::map<LogicalRegion, RegionVersioning>& to_perform)
    //--------------------------------------------------------------------------
    {
      // Invoke the base class version of the finalize method
      OP::finalize_collective_versioning_analysis(
          index, parent_req_index, to_perform);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplCollectiveVersioning<OP>::create_collective_rendezvous(
        unsigned requirement_index)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          collective_versioning_rendezvous.find(requirement_index) ==
          collective_versioning_rendezvous.end());
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(this->get_context());
      const CollectiveID id = repl_ctx->get_next_collective_index(
          COLLECTIVE_LOC_20, true /*logical*/);
      // Round-robin the collective analysis creation around the shards
      const ShardID owner = requirement_index % repl_ctx->total_shards;
      CollectiveVersioningRendezvous* analysis =
          new CollectiveVersioningRendezvous(
              id, repl_ctx, this, this, owner, requirement_index);
      collective_versioning_rendezvous[requirement_index] = analysis;
      // Make sure these are always included in the mapping analysis
      const RtEvent done = analysis->get_done_event();
      if (done.exists())
        this->map_applied_conditions.insert(done);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplCollectiveVersioning<OP>::shard_off_collective_rendezvous(
        std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      op::map<LogicalRegion, RegionVersioning> empty_versions;
      for (typename std::map<
               unsigned, CollectiveVersioningRendezvous*>::const_iterator it =
               collective_versioning_rendezvous.begin();
           it != collective_versioning_rendezvous.end(); it++)
        it->second->perform_rendezvous(0, empty_versions);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplCollectiveVersioning<OP>::elide_collective_rendezvous(void)
    //--------------------------------------------------------------------------
    {
      for (typename std::map<
               unsigned, CollectiveVersioningRendezvous*>::const_iterator it =
               collective_versioning_rendezvous.begin();
           it != collective_versioning_rendezvous.end(); it++)
        it->second->elide_collective();
    }

    template class ReplCollectiveVersioning<CollectiveVersioning<DeletionOp> >;
    template class ReplCollectiveVersioning<CollectiveVersioning<DiscardOp> >;
    template class ReplCollectiveVersioning<CollectiveVersioning<FillOp> >;
    template class ReplCollectiveVersioning<CollectiveViewCreator<AttachOp> >;
    template class ReplCollectiveVersioning<CollectiveViewCreator<DetachOp> >;

    /////////////////////////////////////////////////////////////
    // Repl Collective View Creator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename OP>
    ReplCollectiveViewCreator<OP>::ReplCollectiveViewCreator(void)
      : ReplCollectiveVersioning<OP>()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplCollectiveViewCreator<OP>::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      for (typename std::map<
               RendezvousKey, CollectiveViewRendezvous*>::const_iterator it =
               collective_view_rendezvous.begin();
           it != collective_view_rendezvous.end(); it++)
        delete it->second;
      collective_view_rendezvous.clear();
      ReplCollectiveVersioning<OP>::deactivate(freeop);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplCollectiveViewCreator<OP>::construct_collective_mapping(
        const RendezvousKey& key,
        std::map<LogicalRegion, CollectiveRendezvous>& rendezvous)
    //--------------------------------------------------------------------------
    {
      typename std::map<
          RendezvousKey, CollectiveViewRendezvous*>::const_iterator finder =
          collective_view_rendezvous.find(key);
      legion_assert(finder != collective_view_rendezvous.end());
      finder->second->perform_rendezvous(rendezvous);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplCollectiveViewCreator<OP>::finalize_collective_mapping(
        const RendezvousKey& key,
        std::map<LogicalRegion, CollectiveRendezvous>& rendezvous)
    //--------------------------------------------------------------------------
    {
      // Do the base task call here since we've done the collective rendezvous
      OP::construct_collective_mapping(key, rendezvous);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplCollectiveViewCreator<OP>::create_collective_rendezvous(
        RegionTreeID tid, unsigned requirement_index, unsigned analysis_index)
    //--------------------------------------------------------------------------
    {
      const RendezvousKey key(requirement_index, analysis_index);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(this->get_context());
      // This should always be in the dependence analysis stage of the pipeline
      // so we need to make sure we make the right kind of collective ID
      const CollectiveID id = repl_ctx->get_next_collective_index(
          COLLECTIVE_LOC_19, true /*logical*/);
      CollectiveViewRendezvous* rendezvous =
          new CollectiveViewRendezvous(id, repl_ctx, this, this, key, tid);
      collective_view_rendezvous[key] = rendezvous;
      // If this is the owner save the completeion event for the collective
      // in the  map_applied_conditions as this fixes a race where the the
      // collective finishes but when calling 'finalize_collective_mapping'
      // it first completes all the local points and they execute and we
      // then try to clean up the collective object before we finish the
      // finalize method and that can lead to data corruption
      if (rendezvous->is_target())
      {
        const RtEvent done = rendezvous->get_done_event();
        if (done.exists())
          this->map_applied_conditions.insert(done);
      }
      // If this is the first analysis then make the collective rendezvous
      // for performing the versioning analysis
      if (analysis_index == 0)
        ReplCollectiveVersioning<OP>::create_collective_rendezvous(
            requirement_index);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplCollectiveViewCreator<OP>::shard_off_collective_rendezvous(
        std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveVersioning<OP>::shard_off_collective_rendezvous(
          preconditions);
      std::map<LogicalRegion, CollectiveRendezvous> empty_rendezvous;
      for (typename std::map<
               RendezvousKey, CollectiveViewRendezvous*>::const_iterator it =
               collective_view_rendezvous.begin();
           it != collective_view_rendezvous.end(); it++)
      {
        it->second->perform_rendezvous(empty_rendezvous);
        preconditions.insert(it->second->get_done_event());
      }
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplCollectiveViewCreator<OP>::elide_collective_rendezvous(void)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveVersioning<OP>::elide_collective_rendezvous();
      for (typename std::map<
               RendezvousKey, CollectiveViewRendezvous*>::const_iterator it =
               collective_view_rendezvous.begin();
           it != collective_view_rendezvous.end(); it++)
        it->second->elide_collective();
    }

    template class ReplCollectiveViewCreator<CollectiveViewCreator<AcquireOp> >;
    template class ReplCollectiveViewCreator<CollectiveViewCreator<AttachOp> >;
    template class ReplCollectiveViewCreator<IndexAttachOp>;
    template class ReplCollectiveViewCreator<
        CollectiveViewCreator<DependentPartitionOp> >;
    template class ReplCollectiveViewCreator<CollectiveViewCreator<DetachOp> >;
    template class ReplCollectiveViewCreator<IndexDetachOp>;
    template class ReplCollectiveViewCreator<CollectiveViewCreator<MapOp> >;
    template class ReplCollectiveViewCreator<CollectiveViewCreator<ReleaseOp> >;
    template class ReplCollectiveViewCreator<IndexTask>;

  }  // namespace Internal
}  // namespace Legion
