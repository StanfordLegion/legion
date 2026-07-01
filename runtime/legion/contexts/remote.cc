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

#include "legion/contexts/remote.h"
#include "legion/analysis/equivalence_set.h"
#include "legion/managers/shard.h"
#include "legion/nodes/region.h"
#include "legion/nodes/kdtree.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Remote Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteTask::RemoteTask(RemoteContext* own) : owner(own), context_index(0)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RemoteTask::~RemoteTask(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID RemoteTask::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return owner->get_unique_id();
    }

    //--------------------------------------------------------------------------
    Domain RemoteTask::get_slice_domain(void) const
    //--------------------------------------------------------------------------
    {
      return Domain(index_point, index_point);
    }

    //--------------------------------------------------------------------------
    ShardID RemoteTask::get_shard_id(void) const
    //--------------------------------------------------------------------------
    {
      return owner->shard_id;
    }

    //--------------------------------------------------------------------------
    size_t RemoteTask::get_total_shards(void) const
    //--------------------------------------------------------------------------
    {
      return owner->total_shards;
    }

    //--------------------------------------------------------------------------
    DomainPoint RemoteTask::get_shard_point(void) const
    //--------------------------------------------------------------------------
    {
      return owner->shard_point;
    }

    //--------------------------------------------------------------------------
    Domain RemoteTask::get_shard_domain(void) const
    //--------------------------------------------------------------------------
    {
      return owner->shard_domain;
    }

    //--------------------------------------------------------------------------
    uint64_t RemoteTask::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteTask::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    bool RemoteTask::has_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      return (get_depth() > 0);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteTask::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if ((parent_task == nullptr) && has_parent_task())
        parent_task = owner->get_parent_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const std::string_view& RemoteTask::get_provenance_string(bool human) const
    //--------------------------------------------------------------------------
    {
      Provenance* provenance = owner->get_provenance();
      if (provenance != nullptr)
        return human ? provenance->human : provenance->machine;
      else
        return Provenance::no_provenance;
    }

    //--------------------------------------------------------------------------
    int RemoteTask::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return owner->get_depth();
    }

    //--------------------------------------------------------------------------
    const char* RemoteTask::get_task_name(void) const
    //--------------------------------------------------------------------------
    {
      TaskImpl* task_impl = runtime->find_task_impl(task_id);
      return task_impl->get_name();
    }

    //--------------------------------------------------------------------------
    bool RemoteTask::has_trace(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    /////////////////////////////////////////////////////////////
    // Remote Context
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteContext::RemoteContext(DistributedID id, CollectiveMapping* mapping)
      : HeapifyMixin<RemoteContext, InnerContext, CONTEXT_LIFETIME>(
            configure_remote_context(), (SingleTask*)nullptr, -1,
            false /*full inner*/, remote_task.regions,
            remote_task.output_regions, local_parent_req_indexes,
            local_virtual_mapped, 0 /*priority*/, ApEvent::NO_AP_EVENT, id,
            false, false, false, mapping),
        parent_ctx(nullptr), shard_manager(nullptr), provenance(nullptr),
        top_level_context(false), remote_task(RemoteTask(this)), remote_uid(0),
        repl_id(0)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    /*static*/ Mapper::ContextConfigOutput
        RemoteContext::configure_remote_context(void)
    //--------------------------------------------------------------------------
    {
      // Remote contexts are never going to have to act as a context in the
      // normal sense, but we do still need them to help progress the mapping.
      // Therefore we sill need to configure them to map outstanding tasks.
      // We're going to ignore frames here for now and just use the default
      // configuration for how far to map into the future.
      Mapper::ContextConfigOutput configuration;
      configuration.max_window_size = runtime->initial_task_window_size;
      configuration.hysteresis_percentage =
          runtime->initial_task_window_hysteresis;
      configuration.max_outstanding_frames = 0;
      configuration.min_tasks_to_schedule = runtime->initial_tasks_to_schedule;
      configuration.min_frames_to_schedule = 0;
      configuration.meta_task_vector_width =
          runtime->initial_meta_task_vector_width;
      configuration.max_templates_per_trace =
          LEGION_DEFAULT_MAX_TEMPLATES_PER_TRACE;
      configuration.mutable_priority = false;
      configuration.auto_tracing_enabled = false;
      configuration.auto_tracing_window_size = 0;
      configuration.auto_tracing_ruler_function = 0;
      configuration.auto_tracing_min_trace_length = 0;
      configuration.auto_tracing_max_trace_length = 0;
      configuration.auto_tracing_visit_threshold = 0;
      return configuration;
    }

    //--------------------------------------------------------------------------
    RemoteContext::~RemoteContext(void)
    //--------------------------------------------------------------------------
    {
      if (!local_field_infos.empty())
      {
        // If we have any local fields then tell field space that
        // we can remove them and then clear them
        for (const std::pair<const FieldSpace, std::vector<LocalFieldInfo> >&
                 it : local_field_infos)
        {
          const std::vector<LocalFieldInfo>& infos = it.second;
          std::vector<FieldID> to_remove;
          for (unsigned idx = 0; idx < infos.size(); idx++)
          {
            if (infos[idx].ancestor)
              continue;
            to_remove.emplace_back(infos[idx].fid);
          }
          if (!to_remove.empty())
            runtime->remove_local_fields(it.first, to_remove);
        }
        local_field_infos.clear();
      }
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    const Task* RemoteContext::get_task(void) const
    //--------------------------------------------------------------------------
    {
      return &remote_task;
    }

    //--------------------------------------------------------------------------
    UniqueID RemoteContext::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return remote_uid;
    }

    //--------------------------------------------------------------------------
    InnerContext* RemoteContext::find_top_context(InnerContext* previous)
    //--------------------------------------------------------------------------
    {
      if (!top_level_context)
        return find_parent_context()->find_top_context(this);
      legion_assert(previous != nullptr);
      return previous;
    }

    //--------------------------------------------------------------------------
    InnerContext* RemoteContext::find_parent_context(void)
    //--------------------------------------------------------------------------
    {
      if (top_level_context)
        return nullptr;
      // See if we already have it
      InnerContext* result = parent_ctx.load();
      if (result != nullptr)
        return result;
      legion_assert(parent_context_did != 0);
      // THIS IS ONLY SAFE BECAUSE THIS FUNCTION IS NEVER CALLED BY
      // A MESSAGE IN THE CONTEXT_VIRTUAL_CHANNEL
      result = runtime->find_or_request_inner_context(parent_context_did);
      legion_assert(result != nullptr);
      if (parent_ctx.exchange(result) == nullptr)
        remote_task.parent_task = result->get_task();
      return result;
    }

    //--------------------------------------------------------------------------
    RtEvent RemoteContext::compute_equivalence_sets(
        unsigned req_index, const std::vector<EqSetTracker*>& targets,
        const std::vector<AddressSpaceID>& target_spaces,
        AddressSpaceID creation_target_space, IndexSpaceExpression* expr,
        const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      legion_assert(!top_level_context);
      legion_assert(targets.size() == target_spaces.size());
      // If this is virtual mapped, then continue up to the parent
      if ((req_index < regions.size()) && virtual_mapped[req_index])
        return find_parent_context()->compute_equivalence_sets(
            parent_req_indexes[req_index], targets, target_spaces,
            creation_target_space, expr, mask);
      RtUserEvent ready_event = Runtime::create_rt_user_event();
      // Send off a request to the owner node to handle it
      ComputeEquivalenceSetsRequest rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize<size_t>(targets.size());
        for (unsigned idx = 0; idx < targets.size(); idx++)
        {
          rez.serialize(targets[idx]);
          rez.serialize(target_spaces[idx]);
        }
        rez.serialize(creation_target_space);
        expr->pack_expression(rez, owner_space);
        rez.serialize(mask);
        rez.serialize(req_index);
        rez.serialize(ready_event);
      }
      // Send it to the owner space
      rez.dispatch(owner_space);
      return ready_event;
    }

    //--------------------------------------------------------------------------
    RtEvent RemoteContext::record_output_equivalence_set(
        EqSetTracker* source, AddressSpaceID source_space, unsigned req_index,
        EquivalenceSet* set, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      legion_assert(regions.size() <= req_index);
      const RtUserEvent recorded = Runtime::create_rt_user_event();
      OutputEquivalenceSetRequest rez;
      {
        RezCheck z(rez);
        pack_inner_context(rez);
        rez.serialize(source);
        rez.serialize(source_space);
        rez.serialize(req_index);
        rez.serialize(set->did);
        rez.serialize(mask);
        rez.serialize(recorded);
        set->pack_global_ref(rez);
      }
      rez.dispatch(owner_space);
      return recorded;
    }

    //--------------------------------------------------------------------------
    InnerContext* RemoteContext::find_parent_physical_context(unsigned index)
    //--------------------------------------------------------------------------
    {
      legion_assert(regions.size() <= virtual_mapped.size());
      legion_assert(regions.size() <= parent_req_indexes.size());
      if (index < regions.size())
      {
        // See if it is virtual mapped
        if (virtual_mapped[index])
          return find_parent_context()->find_parent_physical_context(
              parent_req_indexes[index]);
        else  // We mapped a physical instance so we're it
          return this;
      }
      else  // We created it
      {
        // But we're the remote note, so we don't have updated created
        // requirements or returnable privileges so we need to see if
        // we already know the answer and if not, ask the owner context
        RtEvent wait_on;
        RtUserEvent request;
        {
          AutoLock rem_lock(remote_lock);
          std::map<unsigned, InnerContext*>::const_iterator finder =
              physical_contexts.find(index);
          if (finder != physical_contexts.end())
            return finder->second;
          std::map<unsigned, RtEvent>::const_iterator pending_finder =
              pending_physical_contexts.find(index);
          if (pending_finder == pending_physical_contexts.end())
          {
            // Make a new request
            request = Runtime::create_rt_user_event();
            pending_physical_contexts[index] = request;
            wait_on = request;
          }
          else  // Already sent it so just get the wait event
            wait_on = pending_finder->second;
        }
        if (request.exists())
        {
          // Send the request
          RemoteContextPhysicalRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(index);
            rez.serialize(this);
            rez.serialize(request);
          }
          rez.dispatch(owner_space);
        }
        // Wait for the result to come back to us
        wait_on.wait();
        // When we wake up it should be there
        AutoLock rem_lock(remote_lock, false /*exclusive*/);
        legion_assert(physical_contexts.find(index) != physical_contexts.end());
        return physical_contexts[index];
      }
    }

    //--------------------------------------------------------------------------
    void RemoteContext::pack_inner_context(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(did);                     // pack our distributed ID
      rez.serialize<DistributedID>(repl_id);  // shard manager ID
    }

    //--------------------------------------------------------------------------
    InnerContext::CollectiveResult*
        RemoteContext::find_or_create_collective_view(
            RegionTreeID tid, const std::vector<DistributedID>& instances,
            RtEvent& ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(instances.size() > 1);
      const RtUserEvent to_trigger = Runtime::create_rt_user_event();
      CollectiveResult* result = new CollectiveResult(instances);
      result->add_reference();
      RemoteContextFindCollectiveViewRequest rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(tid);
        rez.serialize<size_t>(instances.size());
        for (unsigned idx = 0; idx < instances.size(); idx++)
          rez.serialize(instances[idx]);
        rez.serialize(result);
        rez.serialize(to_trigger);
      }
      rez.dispatch(owner_space);
      ready = to_trigger;
      return result;
    }

    //--------------------------------------------------------------------------
    void RemoteContext::refine_equivalence_sets(
        unsigned req_index, IndexSpaceNode* node,
        const FieldMask& refinement_mask, std::vector<RtEvent>& applied_events,
        bool sharded, bool first, const CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      legion_assert(!sharded);
      if ((req_index < regions.size()) && virtual_mapped[req_index])
      {
        find_parent_context()->refine_equivalence_sets(
            parent_req_indexes[req_index], node, refinement_mask,
            applied_events, sharded, false /*first*/, mapping);
        return;
      }
      if ((mapping == nullptr) ||
          (!mapping->contains(owner_space) &&
           (local_space == mapping->find_nearest(owner_space))))
      {
        const RtUserEvent done = Runtime::create_rt_user_event();
        RemoteContextRefineEquivalenceSets rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(req_index);
          rez.serialize(node->handle);
          rez.serialize(refinement_mask);
          rez.serialize(done);
        }
        rez.dispatch(owner_space);
        applied_events.emplace_back(done);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent RemoteContext::find_pointwise_dependence(
        uint64_t context_index, const DomainPoint& point, ShardID shard,
        bool intra_space, RtUserEvent to_trigger,
        std::optional<unsigned> output_index)
    //--------------------------------------------------------------------------
    {
      if (!to_trigger.exists())
        to_trigger = Runtime::create_rt_user_event();
      RemoteContextPointwiseDependence rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(context_index);
        rez.serialize(point);
        rez.serialize(shard);
        rez.serialize(intra_space);
        rez.serialize(to_trigger);
        rez.serialize(output_index);
      }
      rez.dispatch(owner_space);
      return to_trigger;
    }

    //--------------------------------------------------------------------------
    void RemoteContext::find_trace_local_sets(
        unsigned req_index, const FieldMask& mask,
        std::map<EquivalenceSet*, unsigned>& current_sets, IndexSpaceNode* node,
        const CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      if ((req_index < regions.size()) && virtual_mapped[req_index])
      {
        if (node == nullptr)
          node = runtime->get_node(regions[req_index].region.get_index_space());
        find_parent_context()->find_trace_local_sets(
            req_index, mask, current_sets, node, mapping);
        return;
      }
      if ((mapping == nullptr) ||
          (!mapping->contains(owner_space) &&
           (local_space == mapping->find_nearest(owner_space))))
      {
        const RtUserEvent done = Runtime::create_rt_user_event();
        RemoteContextFindTraceLocalRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(req_index);
          rez.serialize(mask);
          if (node == nullptr)
            rez.serialize(IndexSpace::NO_SPACE);
          else
            rez.serialize(node->handle);
          rez.serialize(&current_sets);
          rez.serialize(done);
        }
        rez.dispatch(owner_space);
        done.wait();
      }
    }

    //--------------------------------------------------------------------------
    void RemoteContext::invalidate_logical_context(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void RemoteContext::invalidate_region_tree_contexts(
        const bool is_top_level_task, std::set<RtEvent>& applied,
        const ShardMapping* mapping, ShardID source_shard)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void RemoteContext::receive_created_region_contexts(
        const std::vector<RegionNode*>& created_nodes,
        const std::vector<EqKDTree*>& created_trees,
        std::set<RtEvent>& applied_events, const ShardMapping* shard_mapping,
        ShardID source_shard)
    //--------------------------------------------------------------------------
    {
      legion_assert(created_nodes.size() == created_trees.size());
      const RtUserEvent done_event = Runtime::create_rt_user_event();
      CreatedRegionContextsMessage rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        if (shard_mapping != nullptr)
        {
          shard_mapping->pack_mapping(rez);
          rez.serialize(source_shard);
        }
        else
          ShardMapping::pack_empty(rez);
        rez.serialize<size_t>(created_nodes.size());
        for (unsigned idx = 0; idx < created_nodes.size(); idx++)
        {
          RegionNode* region = created_nodes[idx];
          rez.serialize(region->handle);
          local::FieldMaskMap<EquivalenceSet> eq_sets;
          created_trees[idx]->find_local_equivalence_sets(
              eq_sets, source_shard);
          rez.serialize<size_t>(eq_sets.size());
          for (local::FieldMaskMap<EquivalenceSet>::const_iterator it =
                   eq_sets.begin();
               it != eq_sets.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
            it->first->pack_global_ref(rez);
          }
        }
        rez.serialize(done_event);
        for (unsigned idx = 0; idx < created_nodes.size(); idx++)
          created_nodes[idx]->pack_global_ref(rez);
        pack_global_ref(rez);
      }
      rez.dispatch(owner_space);
      applied_events.insert(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CreatedRegionContextsMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID context_did;
      derez.deserialize(context_did);
      ShardMapping src_mapping;
      src_mapping.unpack_mapping(derez);
      ShardID source_shard = 0;
      if (!src_mapping.empty())
        derez.deserialize(source_shard);
      size_t num_regions;
      derez.deserialize(num_regions);
      std::vector<RegionNode*> created_nodes(num_regions);
      std::vector<EqKDTree*> created_trees(num_regions);
      std::set<RtEvent> applied_events;
      for (unsigned idx1 = 0; idx1 < num_regions; idx1++)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        RegionNode* node = runtime->get_node(handle);
#ifdef LEGION_DEBUG
        // A little bit strange here, but we have to add a reference
        // to keep the debug checks on packing and unpacking references
        // happy here. In release mode we know we're safe because we
        // still have the packed reference we were sent with
        legion_no_skip_assert(node->check_global_and_increment(META_TASK_REF));
#endif
        created_nodes[idx1] = node;
        EqKDTree* tree = node->row_source->create_equivalence_set_kd_tree(
            src_mapping.empty() ? 1 : src_mapping.size());
        size_t num_sets;
        derez.deserialize(num_sets);
        for (unsigned idx2 = 0; idx2 < num_sets; idx2++)
        {
          DistributedID did;
          derez.deserialize(did);
          RtEvent ready;
          EquivalenceSet* set =
              runtime->find_or_request_equivalence_set(did, ready);
          FieldMask mask;
          derez.deserialize(mask);
          if (ready.exists() && !ready.has_triggered())
            ready.wait();
          set->set_expr->initialize_equivalence_set_kd_tree(
              tree, set, mask, source_shard, true /*current*/);
          set->unpack_global_ref(derez);
        }
        tree->add_reference();
        created_trees[idx1] = tree;
      }
      RtUserEvent done_event;
      derez.deserialize(done_event);

      InnerContext* context = static_cast<InnerContext*>(
          runtime->find_distributed_collectable(context_did));
      context->receive_created_region_contexts(
          created_nodes, created_trees, applied_events,
          src_mapping.empty() ? nullptr : &src_mapping, source_shard);
      if (!applied_events.empty())
        Runtime::trigger_event(
            done_event, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(done_event);
      for (RegionNode* node : created_nodes) node->unpack_global_ref(derez);
#ifdef LEGION_DEBUG
      for (RegionNode* node : created_nodes)
        if (node->remove_base_gc_ref(META_TASK_REF))
          delete node;
#endif
      for (EqKDTree* it : created_trees)
        if ((it != nullptr) && it->remove_reference())
          delete it;
      context->unpack_global_ref(derez);
    }

    //--------------------------------------------------------------------------
    void RemoteContext::unpack_remote_context(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(depth);
      top_level_context = (depth < 0);
      // If we're the top-level context then we're already done
      if (top_level_context)
        return;
      remote_task.unpack_external_task(derez);
      local_parent_req_indexes.resize(remote_task.regions.size());
      for (unsigned idx = 0; idx < local_parent_req_indexes.size(); idx++)
        derez.deserialize(local_parent_req_indexes[idx]);
      size_t num_virtual;
      derez.deserialize(num_virtual);
      local_virtual_mapped.resize(regions.size(), false);
      for (unsigned idx = 0; idx < num_virtual; idx++)
      {
        unsigned index;
        derez.deserialize(index);
        local_virtual_mapped[index] = true;
      }
      derez.deserialize(parent_context_did);
      context_coordinates.deserialize(derez);
      // Reference comes back from deserialize
      provenance = Provenance::deserialize(derez);
      derez.deserialize(remote_uid);
      // Unpack any local fields that we have
      unpack_local_field_update(derez);
      derez.deserialize(concurrent_context);
      bool replicate;
      derez.deserialize(replicate);
      if (replicate)
      {
        derez.deserialize(shard_id);
        derez.deserialize(total_shards);
        derez.deserialize(shard_point);
        derez.deserialize(shard_domain);
        derez.deserialize(repl_id);
        // See if we have a local shard manager
        shard_manager = runtime->find_shard_manager(repl_id, true /*can fail*/);
      }
      // See if we can find our parent task, if not don't worry about it
      // DO NOT CHANGE THIS UNLESS YOU THINK REALLY HARD ABOUT VIRTUAL
      // CHANNELS AND HOW CONTEXT META-DATA IS MOVED!
      InnerContext* parent = static_cast<InnerContext*>(
          runtime->weak_find_distributed_collectable(parent_context_did));
      if (parent != nullptr)
      {
        parent_ctx.store(parent);
        remote_task.parent_task = parent->get_task();
        if (parent->remove_base_resource_ref(RUNTIME_REF))
          delete parent;
      }
    }

    //--------------------------------------------------------------------------
    const Task* RemoteContext::get_parent_task(void)
    //--------------------------------------------------------------------------
    {
      // Note that it safe to actually perform the find_context call here
      // because we are no longer in the virtual channel for unpacking
      // remote contexts therefore we can page in the context
      InnerContext* parent = parent_ctx.load();
      if (parent == nullptr)
      {
        parent = runtime->find_or_request_inner_context(parent_context_did);
        const Task* result = parent->get_task();
        if (parent_ctx.exchange(parent) == nullptr)
          remote_task.parent_task = result;
        return result;
      }
      else
        return parent->get_task();
    }

    //--------------------------------------------------------------------------
    void RemoteContext::unpack_local_field_update(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      size_t num_field_spaces;
      derez.deserialize(num_field_spaces);
      if (num_field_spaces == 0)
        return;
      for (unsigned fidx = 0; fidx < num_field_spaces; fidx++)
      {
        FieldSpace handle;
        derez.deserialize(handle);
        // Reference comes back from deserialize
        Provenance* local_provenance = Provenance::deserialize(derez);
        size_t num_local;
        derez.deserialize(num_local);
        std::vector<FieldID> fields(num_local);
        std::vector<size_t> field_sizes(num_local);
        std::vector<CustomSerdezID> serdez_ids(num_local);
        std::vector<unsigned> indexes(num_local);
        {
          // Take the lock for updating this data structure
          AutoLock local_lock(local_field_lock);
          std::vector<LocalFieldInfo>& infos = local_field_infos[handle];
          infos.resize(num_local);
          for (unsigned idx = 0; idx < num_local; idx++)
          {
            LocalFieldInfo& info = infos[idx];
            derez.deserialize(info);
            // Update data structures for notifying the field space
            fields[idx] = info.fid;
            field_sizes[idx] = info.size;
            serdez_ids[idx] = info.serdez;
            indexes[idx] = info.index;
          }
        }
        runtime->update_local_fields(
            handle, fields, field_sizes, serdez_ids, indexes, local_provenance);
        if ((local_provenance != nullptr) &&
            local_provenance->remove_reference())
          delete local_provenance;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void LocalFieldUpdateMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RemoteContext* context = static_cast<RemoteContext*>(
          runtime->find_or_request_inner_context(did));
      context->unpack_local_field_update(derez);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
      context->unpack_global_ref(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContextPhysicalRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID context_did;
      derez.deserialize(context_did);
      unsigned index;
      derez.deserialize(index);
      RemoteContext* target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      RtEvent ctx_ready;
      InnerContext* local = runtime->find_or_request_inner_context(context_did);
      InnerContext* result = local->find_parent_physical_context(index);
      RemoteContextPhysicalResponse rez;
      {
        RezCheck z(rez);
        rez.serialize(target);
        rez.serialize(index);
        result->pack_inner_context(rez);
        rez.serialize(to_trigger);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    void RemoteContext::set_physical_context_result(
        unsigned index, InnerContext* result)
    //--------------------------------------------------------------------------
    {
      AutoLock rem_lock(remote_lock);
      legion_assert(physical_contexts.find(index) == physical_contexts.end());
      physical_contexts[index] = result;
      std::map<unsigned, RtEvent>::iterator finder =
          pending_physical_contexts.find(index);
      legion_assert(finder != pending_physical_contexts.end());
      pending_physical_contexts.erase(finder);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContextPhysicalResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RemoteContext* target;
      derez.deserialize(target);
      unsigned index;
      derez.deserialize(index);
      InnerContext* result = InnerContext::unpack_inner_context(derez);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      target->set_physical_context_result(index, result);
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContextFindCollectiveViewRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID context_did;
      derez.deserialize(context_did);
      InnerContext* local = runtime->find_or_request_inner_context(context_did);
      RegionTreeID tid;
      derez.deserialize(tid);
      size_t num_insts;
      derez.deserialize(num_insts);
      std::vector<DistributedID> instances(num_insts);
      for (unsigned idx = 0; idx < num_insts; idx++)
        derez.deserialize(instances[idx]);
      InnerContext::CollectiveResult* target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);

      RtEvent result_ready;
      InnerContext::CollectiveResult* result =
          local->find_or_create_collective_view(tid, instances, result_ready);
      if (result_ready.exists() && !result_ready.has_triggered())
        result_ready.wait();
      RemoteContextFindCollectiveViewResponse rez;
      {
        RezCheck z2(rez);
        rez.serialize(target);
        rez.serialize(result->collective_did);
        rez.serialize(result->ready_event);
        rez.serialize(to_trigger);
      }
      rez.dispatch(source);
      if (result->remove_reference())
        delete result;
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContextFindCollectiveViewResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      InnerContext::CollectiveResult* target;
      derez.deserialize(target);
      derez.deserialize(target->collective_did);
      derez.deserialize(target->ready_event);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      legion_assert(to_trigger.exists());
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContextRefineEquivalenceSets::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID context_did;
      derez.deserialize(context_did);
      InnerContext* local = runtime->find_or_request_inner_context(context_did);
      unsigned req_index;
      derez.deserialize(req_index);
      IndexSpace handle;
      derez.deserialize(handle);
      IndexSpaceNode* node = runtime->get_node(handle);
      FieldMask mask;
      derez.deserialize(mask);
      RtUserEvent done;
      derez.deserialize(done);
      std::vector<RtEvent> applied_events;
      local->refine_equivalence_sets(req_index, node, mask, applied_events);
      if (!applied_events.empty())
        Runtime::trigger_event(done, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContextPointwiseDependence::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID context_did;
      derez.deserialize(context_did);
      InnerContext* local = runtime->find_or_request_inner_context(context_did);
      uint64_t context_index;
      derez.deserialize(context_index);
      DomainPoint point;
      derez.deserialize(point);
      ShardID shard;
      derez.deserialize(shard);
      bool intra_space;
      derez.deserialize<bool>(intra_space);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      std::optional<unsigned> output_index;
      derez.deserialize(output_index);
      local->find_pointwise_dependence(
          context_index, point, shard, intra_space, to_trigger, output_index);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContextFindTraceLocalRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID context_did;
      derez.deserialize(context_did);
      InnerContext* local = runtime->find_or_request_inner_context(context_did);
      unsigned req_index;
      derez.deserialize(req_index);
      FieldMask mask;
      derez.deserialize(mask);
      IndexSpace handle;
      derez.deserialize(handle);
      IndexSpaceNode* node = nullptr;
      if (handle.exists())
        node = runtime->get_node(handle);
      std::map<EquivalenceSet*, unsigned> current_sets;
      local->find_trace_local_sets(req_index, mask, current_sets, node);
      std::map<EquivalenceSet*, unsigned>* target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);
      if (!current_sets.empty())
      {
        RemoteContextFindTraceLocalResponse rez;
        {
          RezCheck z2(rez);
          rez.serialize(target);
          rez.serialize<LocalLock*>(nullptr);
          rez.serialize(req_index);
          rez.serialize<size_t>(current_sets.size());
          for (const std::pair<EquivalenceSet* const, unsigned>& it :
               current_sets)
          {
            legion_assert(req_index == it.second);
            rez.serialize(it.first->did);
          }
          rez.serialize(done);
        }
        rez.dispatch(source);
      }
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContextFindTraceLocalResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::map<EquivalenceSet*, unsigned>* target;
      derez.deserialize(target);
      LocalLock* target_lock;
      derez.deserialize(target_lock);
      unsigned req_index;
      derez.deserialize(req_index);
      size_t num_sets;
      derez.deserialize(num_sets);
      std::vector<RtEvent> ready_events;
      if (target_lock != nullptr)
      {
        AutoLock t_lock(*target_lock);
        for (unsigned idx = 0; idx < num_sets; idx++)
        {
          DistributedID did;
          derez.deserialize(did);
          RtEvent ready;
          EquivalenceSet* set =
              runtime->find_or_request_equivalence_set(did, ready);
          target->emplace(std::make_pair(set, req_index));
          if (ready.exists())
            ready_events.emplace_back(ready);
        }
      }
      else
      {
        for (unsigned idx = 0; idx < num_sets; idx++)
        {
          DistributedID did;
          derez.deserialize(did);
          RtEvent ready;
          EquivalenceSet* set =
              runtime->find_or_request_equivalence_set(did, ready);
          target->emplace(std::make_pair(set, req_index));
          if (ready.exists())
            ready_events.emplace_back(ready);
        }
      }
      RtUserEvent done;
      derez.deserialize(done);
      if (!ready_events.empty())
        Runtime::trigger_event(done, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContextRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID source;
      derez.deserialize(source);
      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      InnerContext* context = legion_safe_cast<InnerContext*>(dc);
      context->send_context(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContextResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      void* location =
          runtime->find_or_create_pending_collectable_location<RemoteContext>(
              did);
      RemoteContext* context = new (location) RemoteContext(did);
      context->unpack_remote_context(derez);
      context->register_with_runtime();
    }

  }  // namespace Internal
}  // namespace Legion
