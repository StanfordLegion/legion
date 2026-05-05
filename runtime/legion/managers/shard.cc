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

#include "legion/managers/shard.h"
#include "legion/analysis/equivalence_set.h"
#include "legion/contexts/toplevel.h"
#include "legion/api/functors_impl.h"
#include "legion/api/future_impl.h"
#include "legion/managers/mapper.h"
#include "legion/nodes/region.h"
#include "legion/operations/fill.h"
#include "legion/tasks/individual.h"
#include "legion/tracing/shard.h"
#include "legion/views/fill.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Shard Mapping
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardMapping::ShardMapping(void) : Collectable()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ShardMapping::ShardMapping(const std::vector<AddressSpaceID>& spaces)
      : Collectable(), address_spaces(spaces)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ShardMapping::~ShardMapping(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    AddressSpaceID ShardMapping::operator[](unsigned idx) const
    //--------------------------------------------------------------------------
    {
      legion_assert(idx < address_spaces.size());
      return address_spaces[idx];
    }

    //--------------------------------------------------------------------------
    AddressSpaceID& ShardMapping::operator[](unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx < address_spaces.size());
      return address_spaces[idx];
    }

    //--------------------------------------------------------------------------
    void ShardMapping::pack_mapping(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(address_spaces.size());
      for (const AddressSpaceID& space : address_spaces) rez.serialize(space);
    }

    //--------------------------------------------------------------------------
    void ShardMapping::unpack_mapping(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      size_t num_spaces;
      derez.deserialize(num_spaces);
      address_spaces.resize(num_spaces);
      for (unsigned idx = 0; idx < num_spaces; idx++)
        derez.deserialize(address_spaces[idx]);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardMapping::pack_empty(Serializer& rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(0);
    }

    /////////////////////////////////////////////////////////////
    // Shard Manager
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardManager::ShardManager(
        DistributedID id, CollectiveMapping* mapping, unsigned local,
        const Mapper::ContextConfigOutput& config, bool top, bool iso, bool cr,
        const Domain& dom, std::vector<DomainPoint>&& shards,
        std::vector<DomainPoint>&& sorted, std::vector<ShardID>&& lookup,
        SingleTask* original /*= nullptr*/, RtBarrier call_bar)
      : CollectiveViewCreator<CollectiveHelperOp>(
            LEGION_DISTRIBUTED_HELP_ENCODE(id, SHARD_MANAGER_DC), true,
            mapping),
        shard_points(shards), sorted_points(sorted), shard_lookup(lookup),
        shard_domain(dom), total_shards(shard_points.size()),
        original_task(original), local_constituents(local),
        remote_constituents(
            (mapping == nullptr) ?
                0 :
                mapping->count_children(owner_space, local_space)),
        context_configuration(config), top_level_task(top),
        isomorphic_points(iso), control_replicated(cr), address_spaces(nullptr),
        local_startup_complete(0), remote_startup_complete(0),
        local_mapping_complete(0), remote_mapping_complete(0),
        trigger_local_complete(0), trigger_remote_complete(0),
        trigger_local_commit(0), trigger_remote_commit(0),
        semantic_attach_counter(0),
        future_size(std::numeric_limits<size_t>::max()),
        callback_barrier(call_bar), attach_deduplication(nullptr),
        virtual_mapping_rendezvous(nullptr)
    //--------------------------------------------------------------------------
    {
      legion_assert(total_shards > 0);
      legion_assert((local_constituents > 0) || (remote_constituents > 0));
      legion_assert(shard_points.size() == sorted_points.size());
      legion_assert(shard_points.size() == shard_lookup.size());
      if (is_owner())
      {
        legion_assert(!callback_barrier.exists());
        if (control_replicated)
        {
          callback_barrier = runtime->create_rt_barrier(
              (collective_mapping == nullptr) ? 1 : collective_mapping->size());
        }
      }
      else
      {
        legion_assert(callback_barrier.exists() == control_replicated);
      }
#ifdef LEGION_GC
      log_garbage.info(
          "GC Shard Manager %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    ShardManager::~ShardManager(void)
    //--------------------------------------------------------------------------
    {
      // We can delete our shard tasks
      for (ShardTask* shard_task : local_shards) delete shard_task;
      local_shards.clear();
      for (const std::pair<const ShardingID, ShardingFunction*>& entry :
           sharding_functions)
        delete entry.second;
      sharding_functions.clear();
      // Finally unregister ourselves with the runtime
      if (is_owner() && control_replicated)
        callback_barrier.destroy_barrier();
      if ((address_spaces != nullptr) && address_spaces->remove_reference())
        delete address_spaces;
      legion_assert(created_equivalence_sets.empty());
    }

    //--------------------------------------------------------------------------
    InnerContext* ShardManager::get_context(void)
    //--------------------------------------------------------------------------
    {
      return local_shards.front()->get_context();
    }

    //--------------------------------------------------------------------------
    InnerContext* ShardManager::find_physical_context(unsigned index)
    //--------------------------------------------------------------------------
    {
      return local_shards.front()->find_physical_context(index);
    }

    //--------------------------------------------------------------------------
    size_t ShardManager::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      return local_constituents + remote_constituents;
    }

    //--------------------------------------------------------------------------
    void ShardManager::notify_local(void)
    //--------------------------------------------------------------------------
    {
      for (ShardTask* shard_task : local_shards) shard_task->deactivate();
    }

    //--------------------------------------------------------------------------
    void ShardManager::distribute_explicit(
        SingleTask* task, VariantID variant,
        std::vector<Processor>& target_processors,
        std::vector<VariantID>& leaf_variants)
    //--------------------------------------------------------------------------
    {
      legion_assert(!local_shards.empty());
      legion_assert((variant == 0) == !leaf_variants.empty());
      // Initialize the address spaces data structure
      set_shard_mapping(target_processors);
      if (collective_mapping != nullptr)
      {
        std::vector<AddressSpaceID> children;
        collective_mapping->get_children(owner_space, local_space, children);
        for (const AddressSpaceID& child_space : children)
        {
          ReplicateDistribution rez;
          {
            RezCheck z(rez);
            pack_shard_manager(rez);
            rez.serialize<bool>(true /*explicit*/);
            rez.serialize(variant);
            for (unsigned idx = 0; idx < leaf_variants.size(); idx++)
              rez.serialize(leaf_variants[idx]);
            task->get_context()->pack_inner_context(rez);
            task->pack_single_task(rez, child_space);
          }
          rez.dispatch(child_space);
        }
      }
      for (ShardTask* shard_task : local_shards) shard_task->dispatch();
    }

    //--------------------------------------------------------------------------
    void ShardManager::distribute_implicit(
        TaskID task_id, MapperID mapper_id, Processor::Kind kind,
        unsigned shards_per_space, TopLevelContext* ctx)
    //--------------------------------------------------------------------------
    {
      if (collective_mapping == nullptr)
        return;
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(owner_space, local_space, children);
      if (!children.empty())
      {
        TaskTreeCoordinates coordinates;
        ctx->compute_task_tree_coordinates(coordinates);
        legion_assert(coordinates.size() == 1);
        legion_assert(coordinates.back().context_index == 0);
        legion_assert(coordinates.back().index_point[0] == 0);
        const coord_t implicit_coord = coordinates.back().index_point[1];
        for (const AddressSpaceID& child_space : children)
        {
          ReplicateDistribution rez;
          {
            RezCheck z(rez);
            pack_shard_manager(rez);
            rez.serialize<bool>(false /*explicit*/);
            rez.serialize(task_id);
            rez.serialize(mapper_id);
            rez.serialize(kind);
            rez.serialize(shards_per_space);
            rez.serialize(ctx->did);
            rez.serialize(ctx->get_executing_processor());
            rez.serialize(implicit_coord);
          }
          rez.dispatch(child_space);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ShardManager::pack_shard_manager(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(did);
      rez.serialize(shard_domain);
      rez.serialize<size_t>(total_shards);
      rez.serialize<bool>(isomorphic_points);
      if (isomorphic_points)
      {
        for (unsigned idx = 0; idx < total_shards; idx++)
          rez.serialize(shard_points[idx]);
      }
      else
      {
        for (unsigned idx = 0; idx < total_shards; idx++)
        {
          rez.serialize(sorted_points[idx]);
          rez.serialize(shard_lookup[idx]);
        }
      }
      rez.serialize<bool>(top_level_task);
      rez.serialize<bool>(control_replicated);
      rez.serialize(callback_barrier);
      collective_mapping->pack(rez);
      legion_assert(shard_mapping.size() == total_shards);
      for (unsigned idx = 0; idx < total_shards; idx++)
        rez.serialize(shard_mapping[idx]);
      if (control_replicated)
        rez.serialize(context_configuration);
    }

    //--------------------------------------------------------------------------
    void ShardManager::set_shard_mapping(std::vector<Processor>& mapping)
    //--------------------------------------------------------------------------
    {
      legion_assert(address_spaces == nullptr);
      legion_assert(mapping.size() == total_shards);
      shard_mapping.swap(mapping);
      address_spaces = new ShardMapping();
      address_spaces->add_reference();
      address_spaces->resize(shard_mapping.size());
      for (unsigned idx = 0; idx < shard_mapping.size(); idx++)
        (*address_spaces)[idx] = shard_mapping[idx].address_space();
      // Also initialize the collective parameters
      // We just need the collective radix, but use the existing routine
      int collective_radix = runtime->legion_collective_radix;
      int collective_log_radix, collective_stages;
      int participating_spaces, collective_last_radix;
      configure_collective_settings(
          shard_mapping.size(), runtime->address_space, collective_radix,
          collective_log_radix, collective_stages, participating_spaces,
          collective_last_radix);
    }

    //--------------------------------------------------------------------------
    ShardTask* ShardManager::create_shard(
        ShardID id, Processor target, VariantID variant, InnerContext* parent,
        SingleTask* source)
    //--------------------------------------------------------------------------
    {
      ShardTask* shard =
          new Memoizable<ShardTask>(source, parent, this, id, target, variant);
      local_shards.emplace_back(shard);
      return shard;
    }

    //--------------------------------------------------------------------------
    ShardTask* ShardManager::create_shard(
        ShardID id, Processor target, VariantID variant,
        InnerContext* parent_ctx, Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      ShardTask* shard = new Memoizable<ShardTask>(
          parent_ctx, derez, this, id, target, variant);
      local_shards.emplace_back(shard);
      return shard;
    }

    //--------------------------------------------------------------------------
    void ShardManager::finalize_collective_versioning_analysis(
        unsigned index, unsigned parent_req_index,
        op::map<LogicalRegion, RegionVersioning>& to_perform)
    //--------------------------------------------------------------------------
    {
      // All shards should be using the same region
      legion_assert(to_perform.size() <= 1);
      if (!is_owner())
      {
        const AddressSpaceID target =
            collective_mapping->get_parent(owner_space, local_space);
        ReplicateVersioning rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(index);
          rez.serialize(parent_req_index);
          pack_collective_versioning(rez, to_perform);
        }
        rez.dispatch(target);
      }
      else
        original_task->perform_replicate_collective_versioning(
            index, parent_req_index, to_perform);
    }

    //--------------------------------------------------------------------------
    void ShardManager::construct_collective_mapping(
        const RendezvousKey& key,
        std::map<LogicalRegion, CollectiveRendezvous>& rendezvous)
    //--------------------------------------------------------------------------
    {
      // All shards should be using the same region
      legion_assert(rendezvous.size() == 1);
      if (!is_owner())
      {
        const AddressSpaceID target =
            collective_mapping->get_parent(owner_space, local_space);
        ReplicateCollectiveMapping rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(key.region_index);
          rez.serialize(key.analysis);
          pack_collective_rendezvous(rez, rendezvous);
        }
        rez.dispatch(target);
      }
      else
        original_task->convert_replicate_collective_views(key, rendezvous);
    }

    //--------------------------------------------------------------------------
    void ShardManager::finalize_replicate_collective_versioning(
        unsigned index, unsigned parent_req_index,
        op::map<LogicalRegion, RegionVersioning>& to_perform)
    //--------------------------------------------------------------------------
    {
      // Dispatch back to the base class
      CollectiveViewCreator<CollectiveHelperOp>::
          finalize_collective_versioning_analysis(
              index, parent_req_index, to_perform);
    }

    //--------------------------------------------------------------------------
    void ShardManager::finalize_replicate_collective_views(
        const RendezvousKey& key,
        std::map<LogicalRegion, CollectiveRendezvous>& rendezvous)
    //--------------------------------------------------------------------------
    {
      // Dispatch back to the base class
      CollectiveViewCreator<CollectiveHelperOp>::construct_collective_mapping(
          key, rendezvous);
    }

    //--------------------------------------------------------------------------
    void ShardManager::rendezvous_check_virtual_mappings(
        ShardID shard, MapperManager* mapper,
        const std::vector<bool>& virtual_mappings)
    //--------------------------------------------------------------------------
    {
      legion_assert(!virtual_mappings.empty());
      AutoLock m_lock(manager_lock);
      if (virtual_mapping_rendezvous == nullptr)
      {
        virtual_mapping_rendezvous = new VirtualMappingRendezvous();
        virtual_mapping_rendezvous->virtual_mappings = virtual_mappings;
        virtual_mapping_rendezvous->mapper = mapper;
        virtual_mapping_rendezvous->shard = shard;
        virtual_mapping_rendezvous->remaining_arrivals =
            local_constituents + (is_owner() ? 0 : 1);
        // Send it off to any children we might have
        if (collective_mapping != nullptr)
        {
          std::vector<AddressSpaceID> children;
          collective_mapping->get_children(owner_space, local_space, children);
          if (!children.empty())
          {
            pack_global_ref(children.size());
            for (const AddressSpaceID& child : children)
            {
              ReplicateVirtualRendezvous rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
                rez.serialize(shard);
                rez.serialize<size_t>(virtual_mappings.size());
                for (unsigned idx = 0; idx < virtual_mappings.size(); idx++)
                  rez.serialize<bool>(virtual_mappings[idx]);
              }
              rez.dispatch(child);
            }
          }
        }
      }
      else
      {
        const std::vector<bool>& previous_mappings =
            virtual_mapping_rendezvous->virtual_mappings;
        legion_assert(previous_mappings.size() == virtual_mappings.size());
        // Check to see if they are the same
        int bad_index = -1;
        for (unsigned idx = 0; idx < virtual_mappings.size(); idx++)
        {
          if (previous_mappings[idx] == virtual_mappings[idx])
            continue;
          bad_index = idx;
          break;
        }
        if (bad_index >= 0)
        {
          if (mapper == nullptr)
            mapper = virtual_mapping_rendezvous->mapper;
          legion_assert(mapper != nullptr);
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error
                << "Mapper " << mapper->get_mapper_name()
                << " provided different virtual mapping outputs for "
                << "region requirement " << bad_index << " of shards "
                << ((shard < virtual_mapping_rendezvous->shard) ?
                        shard :
                        virtual_mapping_rendezvous->shard)
                << " and "
                << ((shard < virtual_mapping_rendezvous->shard) ?
                        virtual_mapping_rendezvous->shard :
                        shard)
                << " of replicated task "
                << local_shards.back()->get_task_name() << ". "
                << "All shards of a replicated task must either provide "
                << "concrete instances for a particular region requirement or "
                   "all "
                << "shards must decide to virtual map the region requirement. "
                << "Mixed virtual and concrete instances are not allowed.";
            error.raise();
          }
        }
      }
      legion_assert(virtual_mapping_rendezvous->remaining_arrivals > 0);
      if (--virtual_mapping_rendezvous->remaining_arrivals == 0)
      {
        delete virtual_mapping_rendezvous;
        virtual_mapping_rendezvous = nullptr;
      }
    }

    //--------------------------------------------------------------------------
    RtEvent ShardManager::find_pointwise_dependence(
        uint64_t context_index, const DomainPoint& point, ShardID shard,
        bool intra_space, RtUserEvent to_trigger,
        std::optional<unsigned> output_index)
    //--------------------------------------------------------------------------
    {
      // See if it's local or not
      for (ShardTask* shard_task : local_shards)
        if (shard_task->shard_id == shard)
          return shard_task->get_replicate_context()->find_pointwise_dependence(
              context_index, point, shard, intra_space, to_trigger,
              output_index);
      const AddressSpaceID target_space = (*address_spaces)[shard];
      legion_assert(target_space != runtime->address_space);
      if (!to_trigger.exists())
        to_trigger = Runtime::create_rt_user_event();
      ReplPointwiseDependence rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(context_index);
        rez.serialize(point);
        rez.serialize(shard);
        rez.serialize<bool>(intra_space);
        rez.serialize(to_trigger);
        rez.serialize(output_index);
      }
      rez.dispatch(target_space);
      return to_trigger;
    }

    //--------------------------------------------------------------------------
    EquivalenceSet* ShardManager::get_initial_equivalence_set(
        unsigned idx, LogicalRegion handle, InnerContext* context,
        bool first_shard)
    //--------------------------------------------------------------------------
    {
      RegionNode* region = runtime->get_node(handle);
      // Technically this is not correct to use the 'idx' as the op_ctx_index
      // but we know all the shards need to be initialized before any of them
      // can start running so there's no interference with the actual operation
      // indexes when the do start
      return deduplicate_equivalence_set_creation(
          region, SIZE_MAX, idx, context, first_shard);
    }

    //--------------------------------------------------------------------------
    EquivalenceSet* ShardManager::deduplicate_equivalence_set_creation(
        RegionNode* region_node, size_t op_ctx_index,
        unsigned refinement_number, InnerContext* context, bool first_shard,
        const std::vector<ShardID>* creating_shards)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on;
      const EquivalenceSetKey key(
          op_ctx_index, refinement_number, region_node->handle);
      if (first_shard)
      {
        // We're going to make this equivalence set no matter what so
        // go ahead and do that now and then send out the updates
        size_t local_users;
        CollectiveMapping* eq_mapping;
        if (creating_shards != nullptr)
        {
          // Count how many total address spaces are going to need this
          local_users = 0;
          const ShardMapping& local_mapping = get_mapping();
          std::vector<AddressSpaceID> spaces;
          for (const ShardID& shard : *creating_shards)
          {
            AddressSpaceID space = local_mapping[shard];
            if (space == runtime->address_space)
              local_users++;
            if (std::binary_search(spaces.begin(), spaces.end(), space))
              continue;
            spaces.emplace_back(space);
            std::sort(spaces.begin(), spaces.end());
          }
          eq_mapping =
              new CollectiveMapping(spaces, runtime->legion_collective_radix);
        }
        else
        {
          legion_assert(collective_mapping != nullptr);
          eq_mapping = collective_mapping;
          local_users = local_shards.size();
        }
        // Make the distributed ID and broadcast it out to all the participants
        const AddressSpaceID owner_space = runtime->address_space;
        const DistributedID eq_did = runtime->get_available_distributed_id();
        if (eq_mapping->size() > 1)
        {
          ReplEquivalenceSetNotification rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(op_ctx_index);
            rez.serialize(refinement_number);
            rez.serialize(region_node->handle);
            rez.serialize(eq_did);
            if (creating_shards != nullptr)
              eq_mapping->pack(rez);
            else
              rez.serialize<size_t>(0);  // use this to indicate all shards
          }
          std::vector<AddressSpaceID> children;
          eq_mapping->get_children(owner_space, owner_space, children);
          for (const AddressSpaceID& child : children) rez.dispatch(child);
        }
        // Now make the equivalence set locally and register it
        EquivalenceSet* result = new EquivalenceSet(
            eq_did, owner_space, region_node->row_source,
            region_node->handle.get_tree_id(), context, true /*register now*/,
            eq_mapping);
        result->initialize_collective_references(local_users);
        if (local_users > 1)
        {
          AutoLock m_lock(manager_lock);
          std::map<EquivalenceSetKey, NewEquivalenceSet>::iterator finder =
              created_equivalence_sets.find(key);
          if (finder == created_equivalence_sets.end())
          {
            NewEquivalenceSet& new_eq = created_equivalence_sets[key];
            new_eq.new_set = result;
            new_eq.remaining = local_users - 1;
          }
          else
          {
            legion_assert(finder->second.new_set == nullptr);
            legion_assert(finder->second.ready_event.exists());
            legion_assert(finder->second.remaining > 1);
            finder->second.new_set = result;
            finder->second.remaining--;
            Runtime::trigger_event(finder->second.ready_event);
          }
        }
        return result;
      }
      else
      {
        // First check to see if we've already made the entry
        AutoLock m_lock(manager_lock);
        std::map<EquivalenceSetKey, NewEquivalenceSet>::iterator finder =
            created_equivalence_sets.find(key);
        if (finder == created_equivalence_sets.end())
        {
          // Equivalence set not ready, so set up the entry for it
          NewEquivalenceSet& new_eq = created_equivalence_sets[key];
          new_eq.new_set = nullptr;
          new_eq.did = 0;
          new_eq.mapping = nullptr;
          new_eq.ready_event = Runtime::create_rt_user_event();
          wait_on = new_eq.ready_event;
          // Count how many local arrivals we will have
          if (creating_shards != nullptr)
          {
            new_eq.remaining = 0;
            const ShardMapping& local_mapping = get_mapping();
            for (const ShardID& shard : *creating_shards)
              if (local_mapping[shard] == runtime->address_space)
                new_eq.remaining++;
          }
          else
            new_eq.remaining = local_shards.size();
        }
        else
        {
          // See if the equvialence set is ready or not
          if (finder->second.new_set != nullptr)
          {
            EquivalenceSet* result = finder->second.new_set;
            legion_assert(finder->second.remaining > 0);
            if (--finder->second.remaining == 0)
              created_equivalence_sets.erase(finder);
            return result;
          }
          // Count the number of expected arrivals if they haven't
          // already been counted
          if (finder->second.remaining == 0)
          {
            if (creating_shards != nullptr)
            {
              const ShardMapping& local_mapping = get_mapping();
              for (const ShardID& shard : *creating_shards)
                if (local_mapping[shard] == runtime->address_space)
                  finder->second.remaining++;
            }
            else
              finder->second.remaining = local_shards.size();
          }
          // If we have a did from a remote notification we can use that now
          // to create the equivalence set
          if (finder->second.did > 0)
          {
            // We're the first ones to get here after a notification
            // count the number of expected arrivals here if necessary
            EquivalenceSet* result = new EquivalenceSet(
                finder->second.did,
                runtime->determine_owner(finder->second.did),
                region_node->row_source, region_node->handle.get_tree_id(),
                context, true /*register now*/, finder->second.mapping);
            legion_assert(finder->second.remaining > 0);
            result->initialize_collective_references(finder->second.remaining);
            if (--finder->second.remaining == 0)
              created_equivalence_sets.erase(finder);
            else
              finder->second.new_set = result;
            return result;
          }
          legion_assert(finder->second.ready_event.exists());
          wait_on = finder->second.ready_event;
        }
      }
      legion_assert(wait_on.exists());
      wait_on.wait();
      AutoLock m_lock(manager_lock);
      std::map<EquivalenceSetKey, NewEquivalenceSet>::iterator finder =
          created_equivalence_sets.find(key);
      legion_assert(finder != created_equivalence_sets.end());
      legion_assert(finder->second.remaining > 0);
      if (finder->second.new_set == nullptr)
      {
        legion_assert(finder->second.did > 0);
        finder->second.new_set = new EquivalenceSet(
            finder->second.did, runtime->determine_owner(finder->second.did),
            region_node->row_source, region_node->handle.get_tree_id(), context,
            true /*register now*/, finder->second.mapping);
        legion_assert(finder->second.remaining > 0);
        finder->second.new_set->initialize_collective_references(
            finder->second.remaining);
      }
      EquivalenceSet* result = finder->second.new_set;
      if (--finder->second.remaining == 0)
        created_equivalence_sets.erase(finder);
      return result;
    }

    //--------------------------------------------------------------------------
    void* ShardManager::deduplicate_fill_view_allocation(
        size_t key, bool* first)
    //--------------------------------------------------------------------------
    {
      legion_assert((first == nullptr) || !(*first));
      if (local_shards.size() > 1)
      {
        AutoLock m_lock(manager_lock);
        std::map<size_t, std::pair<void*, size_t> >::iterator finder =
            fill_view_allocations.find(key);
        if (finder != fill_view_allocations.end())
        {
          void* result = finder->second.first;
          legion_assert(finder->second.second > 0);
          if (--finder->second.second == 0)
            fill_view_allocations.erase(finder);
          return result;
        }
        else
        {
          void* result = legion_malloc<FillView, LONG_LIFETIME>(
              sizeof(FillView), alignof(FillView));
          if (first != nullptr)
            *first = true;
          fill_view_allocations.emplace(
              key, std::make_pair(result, local_shards.size() - 1));
          return result;
        }
      }
      else
      {
        void* result = legion_malloc<FillView, LONG_LIFETIME>(
            sizeof(FillView), alignof(FillView));
        if (first != nullptr)
          *first = true;
        return result;
      }
    }

    //--------------------------------------------------------------------------
    FillView* ShardManager::deduplicate_fill_view_creation(
        void* ptr, DistributedID fresh_did, UniqueID op_uid, bool* first)
    //--------------------------------------------------------------------------
    {
      legion_assert((first == nullptr) || !(*first));
      if (local_shards.size() > 1)
      {
        AutoLock m_lock(manager_lock);
        std::map<void*, size_t>::iterator finder =
            fill_view_creations.find(ptr);
        if (finder != fill_view_creations.end())
        {
          FillView* result = static_cast<FillView*>(ptr);
          legion_assert(finder->second > 0);
          if (--finder->second == 0)
            fill_view_creations.erase(finder);
          return result;
        }
        else
        {
          if (first != nullptr)
            *first = true;
          FillView* result = new (ptr) FillView(
              fresh_did, op_uid, false /*register now*/, collective_mapping);
          result->add_base_valid_ref(MAPPING_ACQUIRE_REF, local_shards.size());
          result->register_with_runtime();
          fill_view_creations.emplace(ptr, local_shards.size());
          return result;
        }
      }
      else
      {
        if (first != nullptr)
          *first = true;
        FillView* result = new (ptr) FillView(
            fresh_did, op_uid, false /*register now*/, collective_mapping);
        result->add_base_valid_ref(MAPPING_ACQUIRE_REF);
        result->register_with_runtime();
        return result;
      }
    }

    //--------------------------------------------------------------------------
    void ShardManager::deduplicate_attaches(
        const IndexAttachLauncher& launcher, std::vector<unsigned>& indexes)
    //--------------------------------------------------------------------------
    {
      // If we only have one shard then there is no need to deduplicate
      if (local_shards.size() == 1)
      {
        indexes.resize(launcher.handles.size());
        for (unsigned idx = 0; idx < indexes.size(); idx++) indexes[idx] = idx;
        return;
      }
      // If we have multiple local shards then try to deduplicate across them
      RtEvent wait_on;
      RtUserEvent to_trigger;
      {
        AutoLock m_lock(manager_lock);
        if (attach_deduplication == nullptr)
          attach_deduplication = new AttachDeduplication();
        if (attach_deduplication->launchers.empty())
        {
          legion_assert(!attach_deduplication->pending.exists());
          attach_deduplication->pending = Runtime::create_rt_user_event();
        }
        attach_deduplication->launchers.emplace_back(&launcher);
        if (attach_deduplication->launchers.size() == local_shards.size())
        {
          legion_assert(attach_deduplication->pending.exists());
          to_trigger = attach_deduplication->pending;
          // Make a new event for signaling when we are done
          attach_deduplication->pending = Runtime::create_rt_user_event();
        }
        else
          wait_on = attach_deduplication->pending;
      }
      if (to_trigger.exists())
      {
        // Before triggering, do the compuation to figure out which shard
        // is going to own any duplicates, do this by cutting across using
        // snake order of the shards to try and balance them
        bool done = false;
        unsigned index = 0;
        while (!done)
        {
          done = true;
          if ((index % 2) == 0)
          {
            for (unsigned idx = 0; idx < attach_deduplication->launchers.size();
                 idx++)
            {
              const IndexAttachLauncher* next =
                  (attach_deduplication->launchers[idx]);
              if (index >= next->handles.size())
                continue;
              done = false;
              const LogicalRegion handle = next->handles[index];
              if (attach_deduplication->owners.find(handle) ==
                  attach_deduplication->owners.end())
                attach_deduplication->owners.insert(
                    std::make_pair(handle, next));
            }
          }
          else
          {
            for (int idx = (attach_deduplication->launchers.size() - 1);
                 idx >= 0; idx--)
            {
              const IndexAttachLauncher* next =
                  (attach_deduplication->launchers[idx]);
              if (index >= next->handles.size())
                continue;
              done = false;
              const LogicalRegion handle = next->handles[index];
              if (attach_deduplication->owners.find(handle) ==
                  attach_deduplication->owners.end())
                attach_deduplication->owners.insert(
                    std::make_pair(handle, next));
            }
          }
          index++;
        }
        Runtime::trigger_event(to_trigger);
        to_trigger = RtUserEvent::NO_RT_USER_EVENT;
      }
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      // Once we're here, all the launchers can be accessed read-only
      // Figure out which of our handles we still own
      for (unsigned idx = 0; idx < launcher.handles.size(); idx++)
      {
        const LogicalRegion handle = launcher.handles[idx];
        std::map<LogicalRegion, const IndexAttachLauncher*>::const_iterator
            finder = attach_deduplication->owners.find(handle);
        legion_assert(finder != attach_deduplication->owners.end());
        // Only add it if we own it
        if (finder->second == &launcher)
          indexes.emplace_back(idx);
      }
      // When we're done we need to sync on the way out too to make sure
      // everyone is done accessing our launcher before we leave
      {
        AutoLock m_lock(manager_lock);
        legion_assert(attach_deduplication->done_count < local_shards.size());
        attach_deduplication->done_count++;
        if (attach_deduplication->done_count == local_shards.size())
          to_trigger = attach_deduplication->pending;
        else
          wait_on = attach_deduplication->pending;
      }
      if (to_trigger.exists())
      {
        // Need to clean up first
        delete attach_deduplication;
        attach_deduplication = nullptr;
        __sync_synchronize();
        Runtime::trigger_event(to_trigger);
      }
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
    }

    //--------------------------------------------------------------------------
    Future ShardManager::deduplicate_future_creation(
        ReplicateContext* ctx, DistributedID did, Operation* op,
        const DomainPoint& index_point)
    //--------------------------------------------------------------------------
    {
      if (local_shards.size() > 1)
      {
        AutoLock m_lock(manager_lock);
        // See if we already have the future or not
        std::map<DistributedID, std::pair<FutureImpl*, size_t> >::iterator
            finder = created_futures.find(did);
        if (finder != created_futures.end())
        {
          // Make sure to bump the future coordinate for this context as well
#ifdef LEGION_DEBUG
          const uint64_t index =
#endif
              ctx->get_next_blocking_index();
          legion_assert(
              index == finder->second.first->coordinate.context_index);
          Future result(finder->second.first);
          legion_assert(finder->second.second > 0);
          if (--finder->second.second == 0)
          {
            if (finder->second.first->remove_base_gc_ref(RUNTIME_REF))
              std::abort();  // should never be deleted
            created_futures.erase(finder);
          }
          return result;
        }
        // Didn't find it so make it
        FutureImpl* result = new FutureImpl(
            ctx, false /*register*/, did, op, op->get_generation(),
            ContextCoordinate(ctx->get_next_blocking_index(), index_point),
            op->get_unique_op_id(), ctx->get_depth(), op->get_provenance(),
            collective_mapping);
        LegionSpy::log_future_creation(
            op->get_unique_op_id(), result->did, index_point);
        // Add a reference to it to keep it from being deleted and then
        // register it with the runtime
        result->add_base_gc_ref(RUNTIME_REF);
        result->register_with_runtime();
        // Record it for the shards that come later
        std::pair<FutureImpl*, size_t>& pending = created_futures[did];
        pending.first = result;
        pending.second = local_shards.size() - 1;
        return Future(result);
      }
      else
      {
        FutureImpl* impl = new FutureImpl(
            ctx, false /*register*/, did, op, op->get_generation(),
            ContextCoordinate(ctx->get_next_blocking_index(), index_point),
            op->get_unique_op_id(), ctx->get_depth(), op->get_provenance(),
            collective_mapping);
        LegionSpy::log_future_creation(
            op->get_unique_op_id(), impl->did, index_point);
        // Get a reference on it before we register it
        Future result(impl);
        impl->register_with_runtime();
        return result;
      }
    }

    //--------------------------------------------------------------------------
    FutureMap ShardManager::deduplicate_future_map_creation(
        ReplicateContext* ctx, Operation* op, IndexSpaceNode* domain,
        IndexSpaceNode* shard_domain, DistributedID map_did,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (local_shards.size() > 1)
      {
        AutoLock m_lock(manager_lock);
        // See if we already have this here or not
        std::map<DistributedID, std::pair<ReplFutureMapImpl*, size_t> >::
            iterator finder = created_future_maps.find(map_did);
        if (finder != created_future_maps.end())
        {
          // Make sure to bump the future coordinate for this context as well
#ifdef LEGION_DEBUG
          const uint64_t index =
#endif
              ctx->get_next_blocking_index();
          legion_assert(index == finder->second.first->blocking_index);
          FutureMap result(finder->second.first);
          legion_assert(finder->second.second > 0);
          if (--finder->second.second == 0)
          {
            if (finder->second.first->remove_nested_gc_ref(did))
              std::abort();  // should never be deleted
            created_future_maps.erase(finder);
          }
          return result;
        }
        // Didn't find it so make it
        ReplFutureMapImpl* result = new ReplFutureMapImpl(
            ctx, this, op, domain, shard_domain, map_did, provenance,
            collective_mapping);
        // Add a reference to it to keep it from being deleted and then
        // register it with the runtime
        result->add_nested_gc_ref(did);
        result->register_with_runtime();
        // Record it for the shards that come later
        std::pair<ReplFutureMapImpl*, size_t>& pending =
            created_future_maps[map_did];
        pending.first = result;
        pending.second = local_shards.size() - 1;
        return FutureMap(result);
      }
      else
      {
        ReplFutureMapImpl* impl = new ReplFutureMapImpl(
            ctx, this, op, domain, shard_domain, map_did, provenance,
            collective_mapping);
        // Get a reference on it before we register it
        FutureMap result(impl);
        impl->register_with_runtime();
        return result;
      }
    }

    //--------------------------------------------------------------------------
    FutureMap ShardManager::deduplicate_future_map_creation(
        ReplicateContext* ctx, IndexSpaceNode* domain,
        IndexSpaceNode* shard_domain, DistributedID map_did,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      // This future map isn't associated with an oeration so no coordinate
      constexpr uint64_t coordinate = std::numeric_limits<uint64_t>::max();
      if (local_shards.size() > 1)
      {
        AutoLock m_lock(manager_lock);
        // See if we already have this here or not
        std::map<DistributedID, std::pair<ReplFutureMapImpl*, size_t> >::
            iterator finder = created_future_maps.find(map_did);
        if (finder != created_future_maps.end())
        {
          FutureMap result(finder->second.first);
          legion_assert(finder->second.second > 0);
          if (--finder->second.second == 0)
          {
            if (finder->second.first->remove_nested_gc_ref(did))
              std::abort();  // should never be deleted
            created_future_maps.erase(finder);
          }
          return result;
        }
        // Didn't find it so make it
        ReplFutureMapImpl* result = new ReplFutureMapImpl(
            ctx, this, domain, shard_domain, map_did, coordinate,
            std::optional<uint64_t>(), provenance, collective_mapping);
        // Add a reference to it to keep it from being deleted and then
        // register it with the runtime
        result->add_nested_gc_ref(did);
        result->register_with_runtime();
        // Record it for the shards that come later
        std::pair<ReplFutureMapImpl*, size_t>& pending =
            created_future_maps[map_did];
        pending.first = result;
        pending.second = local_shards.size() - 1;
        return FutureMap(result);
      }
      else
      {
        ReplFutureMapImpl* impl = new ReplFutureMapImpl(
            ctx, this, domain, shard_domain, map_did, coordinate,
            std::optional<uint64_t>(), provenance, collective_mapping);
        // Get a reference on it before we register it
        FutureMap result(impl);
        impl->register_with_runtime();
        return result;
      }
    }

    //--------------------------------------------------------------------------
    bool ShardManager::is_total_sharding(void)
    //--------------------------------------------------------------------------
    {
      if (collective_mapping == nullptr)
        return (runtime->total_address_spaces == 1);
      else
        return (runtime->total_address_spaces == collective_mapping->size());
    }

    //--------------------------------------------------------------------------
    void ShardManager::exchange_shard_local_op_data(
        uint64_t context_index, size_t exchange_index, const void* data,
        size_t size)
    //--------------------------------------------------------------------------
    {
      legion_assert(!local_shards.empty());
      if (local_shards.size() == 1)
        return;
      RtUserEvent to_trigger;
      const std::pair<size_t, size_t> key(context_index, exchange_index);
      {
        AutoLock m_lock(manager_lock);
        ShardLocalData& result = shard_local_data[key];
        result.buffer = malloc(size);
        memcpy(result.buffer, data, size);
        result.size = size;
        result.remaining = local_shards.size() - 1;
        to_trigger = result.pending;
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void ShardManager::find_shard_local_op_data(
        uint64_t context_index, size_t exchange_index, void* result,
        size_t size)
    //--------------------------------------------------------------------------
    {
      legion_assert(local_shards.size() > 1);
      RtEvent wait_on;
      const std::pair<size_t, size_t> key(context_index, exchange_index);
      {
        AutoLock m_lock(manager_lock);
        ShardLocalData& data = shard_local_data[key];
        if (data.remaining == 0)
        {
          // here before the sender
          if (!data.pending.exists())
            data.pending = Runtime::create_rt_user_event();
          wait_on = data.pending;
        }
        else
        {
          legion_assert(size == data.size);
          memcpy(result, data.buffer, data.size);
          if ((--data.remaining) == 0)
          {
            free(data.buffer);
            shard_local_data.erase(key);
          }
          return;
        }
      }
      if (!wait_on.has_triggered())
        wait_on.wait();
      AutoLock m_lock(manager_lock);
      std::map<std::pair<size_t, size_t>, ShardLocalData>::iterator finder =
          shard_local_data.find(key);
      legion_assert(finder != shard_local_data.end());
      legion_assert(finder->second.remaining > 0);
      legion_assert(size == finder->second.size);
      memcpy(result, finder->second.buffer, finder->second.size);
      if ((--finder->second.remaining) == 0)
      {
        free(finder->second.buffer);
        shard_local_data.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    void ShardManager::barrier_shard_local(
        uint64_t context_index, size_t exchange_index)
    //--------------------------------------------------------------------------
    {
      legion_assert(!local_shards.empty());
      if (local_shards.size() == 1)
        return;
      RtEvent wait_on;
      const std::pair<size_t, size_t> key(context_index, exchange_index);
      {
        AutoLock m_lock(manager_lock);
        std::map<std::pair<size_t, size_t>, ShardLocalData>::iterator finder =
            shard_local_data.find(key);
        if (finder != shard_local_data.end())
        {
          legion_assert(finder->second.remaining > 0);
          legion_assert(finder->second.pending.exists());
          finder->second.remaining--;
          if (finder->second.remaining == 0)
          {
            Runtime::trigger_event(finder->second.pending);
            shard_local_data.erase(finder);
            return;
          }
          else
            wait_on = finder->second.pending;
        }
        else
        {
          ShardLocalData& data = shard_local_data[key];
          data.pending = Runtime::create_rt_user_event();
          data.remaining = local_shards.size() - 1;
          wait_on = data.pending;
        }
      }
      if (!wait_on.has_triggered())
        wait_on.wait();
    }

    //--------------------------------------------------------------------------
    RtEvent ShardManager::complete_startup_initialization(bool local)
    //--------------------------------------------------------------------------
    {
      RtEvent result;
      bool notify = false;
      {
        AutoLock m_lock(manager_lock);
        if (local)
        {
          local_startup_complete++;
          legion_assert(local_startup_complete <= local_constituents);
        }
        else
        {
          remote_startup_complete++;
          legion_assert(remote_startup_complete <= remote_constituents);
        }
        if (!startup_complete.exists())
          startup_complete = Runtime::create_rt_user_event();
        result = startup_complete;
        notify = (local_startup_complete == local_constituents) &&
                 (remote_startup_complete == remote_constituents);
      }
      if (notify)
      {
        legion_assert(startup_complete.exists());
        if (!is_owner())
        {
          ReplicateStartup rez;
          rez.serialize(did);
          rez.serialize(startup_complete);
          rez.dispatch(
              collective_mapping->get_parent(owner_space, local_space));
        }
        else
          Runtime::trigger_event(startup_complete);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_post_mapped(bool local, RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      bool notify = false;
      {
        AutoLock m_lock(manager_lock);
        if (precondition.exists())
          mapping_preconditions.insert(precondition);
        if (local)
        {
          local_mapping_complete++;
          legion_assert(local_mapping_complete <= local_constituents);
        }
        else
        {
          remote_mapping_complete++;
          legion_assert(remote_mapping_complete <= remote_constituents);
        }
        notify = (local_mapping_complete == local_constituents) &&
                 (remote_mapping_complete == remote_constituents);
      }
      if (notify)
      {
        RtEvent mapped_precondition;
        if (!mapping_preconditions.empty())
          mapped_precondition = Runtime::merge_events(mapping_preconditions);
        if (original_task == nullptr)
        {
          ReplicatePostMapped rez;
          rez.serialize(did);
          rez.serialize(mapped_precondition);
          rez.dispatch(
              collective_mapping->get_parent(owner_space, local_space));
        }
        else
          original_task->handle_post_mapped(mapped_precondition);
      }
    }

    //--------------------------------------------------------------------------
    bool ShardManager::handle_future(
        ApEvent effects, FutureInstance* inst, const void* metadata,
        size_t metasize)
    //--------------------------------------------------------------------------
    {
      bool return_future = (original_task != nullptr);
      {
        AutoLock m_lock(manager_lock);
        // See if we're the first ones to set the future size
        if (future_size < std::numeric_limits<size_t>::max())
        {
          size_t inst_size = (inst == nullptr) ? 0 : inst->size;
          if (inst_size != future_size)
          {
            Warning warning;
            warning << "WARNING: futures returned from control "
                    << "replicated task " << local_shards[0]->get_task_name()
                    << " have different sizes "
                    << "of " << inst_size << " and " << future_size
                    << " bytes!";
            warning.raise();
          }
          return_future = false;
        }
        else if (inst != nullptr)
          future_size = inst->size;
        else
          future_size = 0;
      }
      if (return_future)
      {
        original_task->handle_future(
            effects, inst, metadata, metasize, nullptr /*functor*/,
            Processor::NO_PROC, false /*own functor*/);
        return false;
      }
      else
        return true;
    }

    //--------------------------------------------------------------------------
    ApEvent ShardManager::trigger_task_complete(bool local, ApEvent effects)
    //--------------------------------------------------------------------------
    {
      bool notify = false;
      {
        AutoLock m_lock(manager_lock);
        if (local)
        {
          trigger_local_complete++;
          legion_assert(trigger_local_complete <= local_constituents);
        }
        else
        {
          trigger_remote_complete++;
          legion_assert(trigger_remote_complete <= remote_constituents);
        }
        if (effects.exists())
          shard_effects.insert(effects);
        if (control_replicated)
        {
          // If we're control replicated we'll entangle all the effects so
          // no shard is considered done until they are all done
          if (!all_shards_complete.exists())
            all_shards_complete = Runtime::create_ap_user_event(nullptr);
          effects = all_shards_complete;
        }
        notify = (trigger_local_complete == local_constituents) &&
                 (trigger_remote_complete == remote_constituents);
      }
      if (notify)
      {
        ApEvent all_shard_effects;
        if (!shard_effects.empty())
          all_shard_effects = Runtime::merge_events(nullptr, shard_effects);
        if (original_task == nullptr)
        {
          ReplicateTriggerComplete rez;
          rez.serialize(did);
          rez.serialize(all_shard_effects);
          rez.serialize(all_shards_complete);
          rez.dispatch(
              collective_mapping->get_parent(owner_space, local_space));
        }
        else
        {
          legion_assert(!local_shards.empty());
          if (all_shard_effects.exists())
            original_task->record_completion_effect(all_shard_effects);
          if (all_shards_complete.exists())
            Runtime::trigger_event_untraced(
                all_shards_complete, all_shard_effects);
          original_task->complete_execution();
        }
      }
      return effects;
    }

    //--------------------------------------------------------------------------
    void ShardManager::trigger_task_commit(bool local, RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      bool notify = false;
      {
        AutoLock m_lock(manager_lock);
        if (local)
        {
          trigger_local_commit++;
          legion_assert(trigger_local_commit <= local_constituents);
        }
        else
        {
          trigger_remote_commit++;
          legion_assert(trigger_remote_commit <= remote_constituents);
        }
        if (precondition.exists())
          commit_preconditions.insert(precondition);
        notify = (trigger_local_commit == local_constituents) &&
                 (trigger_remote_commit == remote_constituents);
      }
      if (notify)
      {
        if (original_task == nullptr)
        {
          const RtEvent commit_precondition =
              Runtime::merge_events(commit_preconditions);
          ReplicateTriggerCommit rez;
          rez.serialize(did);
          rez.serialize(commit_precondition);
          rez.dispatch(
              collective_mapping->get_parent(owner_space, local_space));
        }
        else
        {
          if (original_task->is_top_level_task())
            local_shards[0]->report_leaks_and_duplicates(commit_preconditions);
          else
            local_shards[0]->return_resources(
                original_task->get_context(), commit_preconditions);
          RtEvent commit_precondition;
          if (!commit_preconditions.empty())
            commit_precondition = Runtime::merge_events(commit_preconditions);
          original_task->trigger_children_committed(commit_precondition);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_collective_message(
        ShardID target, const ShardCollectiveMessage& rez)
    //--------------------------------------------------------------------------
    {
      legion_assert(target < address_spaces->size());
      AddressSpaceID target_space = (*address_spaces)[target];
      // Check to see if this is a local shard
      if (target_space == runtime->address_space)
      {
        Deserializer derez(rez.get_payload(), rez.get_payload_size());
        // Have to unpack the preample we already know
        DistributedID local_repl;
        derez.deserialize(local_repl);
        handle_collective_message(derez);
      }
      else
        rez.dispatch(target_space);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_collective_message(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      // Figure out which shard we are going to
      ShardID target;
      derez.deserialize(target);
      for (ShardTask* shard : local_shards)
      {
        if (shard->shard_id == target)
        {
          shard->get_replicate_context()->handle_collective_message(derez);
          return;
        }
      }
      // Should never get here
      std::abort();
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_rendezvous_message(
        ShardID target, const ReplicateRendezvousMessage& rez)
    //--------------------------------------------------------------------------
    {
      legion_assert(target < address_spaces->size());
      AddressSpaceID target_space = (*address_spaces)[target];
      // Check to see if this is a local shard
      if (target_space == runtime->address_space)
      {
        Deserializer derez(rez.get_payload(), rez.get_payload_size());
        // Have to unpack the preample we already know
        DistributedID local_repl;
        derez.deserialize(local_repl);
        handle_rendezvous_message(derez);
      }
      else
        rez.dispatch(target_space);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_rendezvous_message(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      // Figure out which shard we are going to
      ShardID target;
      derez.deserialize(target);
      for (ShardTask* shard : local_shards)
      {
        if (shard->shard_id == target)
        {
          shard->get_replicate_context()->handle_rendezvous_message(derez);
          return;
        }
      }
      // Should never get here
      std::abort();
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_compute_equivalence_sets(
        ShardID target, const ReplComputeEquivalenceSets& rez)
    //--------------------------------------------------------------------------
    {
      legion_assert(target < address_spaces->size());
      AddressSpaceID target_space = (*address_spaces)[target];
      // Check to see if this is a local shard
      if (target_space == runtime->address_space)
      {
        Deserializer derez(rez.get_payload(), rez.get_payload_size());
        // Have to unpack the preample we already know
        DistributedID local_repl;
        derez.deserialize(local_repl);
        handle_compute_equivalence_sets(derez);
      }
      else
        rez.dispatch(target_space);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_compute_equivalence_sets(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      // Figure out which shard we are going to
      ShardID target;
      derez.deserialize(target);
      for (ShardTask* shard : local_shards)
      {
        if (shard->shard_id == target)
        {
          shard->get_replicate_context()->handle_compute_equivalence_sets(
              derez);
          return;
        }
      }
      // Should never get here
      std::abort();
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_equivalence_set_notification(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      EquivalenceSetKey key;
      derez.deserialize(key.op_ctx_index);
      derez.deserialize(key.refinement_number);
      derez.deserialize(key.handle);
      DistributedID eq_did;
      derez.deserialize(eq_did);
      size_t num_spaces;
      derez.deserialize(num_spaces);
      legion_assert(collective_mapping != nullptr);
      CollectiveMapping* eq_mapping =
          (num_spaces == 0) ? collective_mapping :
                              new CollectiveMapping(derez, num_spaces);
      const AddressSpaceID owner_space = runtime->determine_owner(eq_did);
      // Send this off to any children in the collective mapping
      std::vector<AddressSpaceID> children;
      eq_mapping->get_children(owner_space, runtime->address_space, children);
      if (!children.empty())
      {
        ReplEquivalenceSetNotification rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(key.op_ctx_index);
          rez.serialize(key.refinement_number);
          rez.serialize(key.handle);
          rez.serialize(eq_did);
          if (eq_mapping != collective_mapping)
            eq_mapping->pack(rez);
          else
            rez.serialize<size_t>(0);  // use this to indicate all the shards
        }
        for (const AddressSpaceID& child : children) rez.dispatch(child);
      }
      // Now we can save this locally and wake up any waiters
      AutoLock m_lock(manager_lock);
      std::map<EquivalenceSetKey, NewEquivalenceSet>::iterator finder =
          created_equivalence_sets.find(key);
      if (finder == created_equivalence_sets.end())
      {
        NewEquivalenceSet& new_eq = created_equivalence_sets[key];
        new_eq.new_set = nullptr;
        new_eq.did = eq_did;
        new_eq.mapping = eq_mapping;
        new_eq.remaining = 0;  // don't know what this is yet
      }
      else
      {
        legion_assert(finder->second.new_set == nullptr);
        legion_assert(finder->second.did == 0);
        legion_assert(finder->second.mapping == nullptr);
        legion_assert(finder->second.ready_event.exists());
        legion_assert(finder->second.remaining > 0);
        finder->second.did = eq_did;
        finder->second.mapping = eq_mapping;
        Runtime::trigger_event(finder->second.ready_event);
      }
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_output_equivalence_set(
        ShardID target, const ReplOutputEquivalenceSet& rez)
    //--------------------------------------------------------------------------
    {
      legion_assert(target < address_spaces->size());
      AddressSpaceID target_space = (*address_spaces)[target];
      // Check to see if this is a local shard
      if (target_space == runtime->address_space)
      {
        Deserializer derez(rez.get_payload(), rez.get_payload_size());
        // Have to unpack the preample we already know
        DistributedID local_repl;
        derez.deserialize(local_repl);
        handle_output_equivalence_set(derez);
      }
      else
        rez.dispatch(target_space);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_output_equivalence_set(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      // Figure out which shard we are going to
      ShardID target;
      derez.deserialize(target);
      for (ShardTask* shard : local_shards)
      {
        if (shard->shard_id == target)
        {
          shard->get_replicate_context()->handle_output_equivalence_set(derez);
          return;
        }
      }
      // Should never get here
      std::abort();
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_output_offset(
        ShardID target, const ReplOutputOffset& rez)
    //--------------------------------------------------------------------------
    {
      legion_assert(target < address_spaces->size());
      AddressSpaceID target_space = (*address_spaces)[target];
      // Check to see if this is a local shard
      if (target_space == runtime->address_space)
      {
        Deserializer derez(rez.get_payload(), rez.get_payload_size());
        // Have to unpack the preample we already know
        DistributedID local_repl;
        derez.deserialize(local_repl);
        handle_output_offset(derez);
      }
      else
        rez.dispatch(target_space);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_output_offset(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      ShardID target;
      derez.deserialize(target);
      for (std::vector<ShardTask*>::const_iterator it = local_shards.begin();
           it != local_shards.end(); it++)
      {
        if ((*it)->shard_id == target)
        {
          (*it)->get_replicate_context()->handle_output_offset(derez);
          return;
        }
      }
      // Should never get here
      std::abort();
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_refine_equivalence_sets(
        ShardID target, const ReplRefineEquivalenceSets& rez)
    //--------------------------------------------------------------------------
    {
      legion_assert(target < address_spaces->size());
      AddressSpaceID target_space = (*address_spaces)[target];
      // Check to see if this is a local shard
      if (target_space == runtime->address_space)
      {
        Deserializer derez(rez.get_payload(), rez.get_payload_size());
        // Have to unpack the preample we already know
        DistributedID local_repl;
        derez.deserialize(local_repl);
        handle_refine_equivalence_sets(derez);
      }
      else
        rez.dispatch(target_space);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_refine_equivalence_sets(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      // Figure out which shard we are going to
      ShardID target;
      derez.deserialize(target);
      for (ShardTask* shard : local_shards)
      {
        if (shard->shard_id == target)
        {
          shard->get_replicate_context()->handle_refine_equivalence_sets(derez);
          return;
        }
      }
      // Should never get here
      std::abort();
    }

    //--------------------------------------------------------------------------
    void ShardManager::broadcast_resource_update(
        ShardTask* source, const Serializer& rez,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      broadcast_message(source, rez, RESOURCE_UPDATE_KIND, applied_events);
    }

    //--------------------------------------------------------------------------
    void ShardManager::broadcast_created_region_contexts(
        ShardTask* source, const Serializer& rez,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      broadcast_message(
          source, rez, CREATED_REGION_UPDATE_KIND, applied_events);
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_created_region_contexts(
        ShardID target, ReplCreatedRegions& rez,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      legion_assert(target < address_spaces->size());
      AddressSpaceID target_space = (*address_spaces)[target];
      if (target_space == runtime->address_space)
      {
        Deserializer derez(rez.get_payload(), rez.get_payload_size());
        // Have to unpack the preample we already know
        DistributedID local_repl;
        derez.deserialize(local_repl);
        handle_created_region_contexts(derez, applied_events);
      }
      else
      {
        const RtUserEvent applied = Runtime::create_rt_user_event();
        rez.serialize(applied);
        rez.dispatch(target_space);
        applied_events.insert(applied);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplCreatedRegions::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID repl_id;
      derez.deserialize(repl_id);
      ShardManager* manager = runtime->find_shard_manager(repl_id);
      std::set<RtEvent> applied_events;
      manager->handle_created_region_contexts(derez, applied_events);
      RtUserEvent applied;
      derez.deserialize(applied);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_created_region_contexts(
        Deserializer& derez, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      ShardID target;
      derez.deserialize(target);
      for (ShardTask* shard : local_shards)
      {
        if (shard->shard_id == target)
        {
          shard->get_replicate_context()->handle_created_region_contexts(
              derez, applied_events);
          return;
        }
      }
      std::abort();  // should never get here
    }

    //--------------------------------------------------------------------------
    bool ShardManager::has_empty_shard_subtree(
        AddressSpaceID space, ShardingFunction* sharding,
        IndexSpaceNode* full_space, IndexSpace sharding_space)
    //--------------------------------------------------------------------------
    {
      // First check to see if any shards with the space are non-empty
      const ShardMapping& mapping = *address_spaces;
      for (ShardID shard = 0; shard < total_shards; shard++)
      {
        if (mapping[shard] != space)
          continue;
        if (sharding->has_participants(shard, full_space, sharding_space))
          return false;
      }
      // Check any children of this space
      if (collective_mapping != nullptr)
      {
        std::vector<AddressSpaceID> children;
        collective_mapping->get_children(owner_space, space, children);
        for (const AddressSpaceID& child : children)
          if (!has_empty_shard_subtree(
                  child, sharding, full_space, sharding_space))
            return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void ShardManager::broadcast_message(
        ShardTask* source, const Serializer& rez, BroadcastMessageKind kind,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      // First pack it out and send it out to any remote nodes
      if (collective_mapping != nullptr)
      {
        legion_assert(collective_mapping->contains(local_space));
        std::vector<AddressSpaceID> targets;
        collective_mapping->get_children(local_space, local_space, targets);
        for (const AddressSpaceID& target : targets)
        {
          RtEvent next_done = Runtime::create_rt_user_event();
          ReplBroadcastUpdate rez2;
          rez2.serialize(did);
          rez2.serialize(local_space);
          rez2.serialize(kind);
          rez2.serialize<size_t>(rez.get_used_bytes());
          rez2.serialize(rez.get_buffer(), rez.get_used_bytes());
          rez2.serialize(next_done);
          rez2.dispatch(target);
          applied_events.insert(next_done);
        }
      }
      // Then send it to any other local shards
      for (ShardTask* shard : local_shards)
      {
        // Skip the source since that's where it came from
        if (shard == source)
          continue;
        Deserializer derez(rez.get_buffer(), rez.get_used_bytes());
        switch (kind)
        {
          case RESOURCE_UPDATE_KIND:
            {
              shard->get_replicate_context()->handle_resource_update(
                  derez, applied_events);
              break;
            }
          case CREATED_REGION_UPDATE_KIND:
            {
              shard->get_replicate_context()->handle_created_region_contexts(
                  derez, applied_events);
              break;
            }
          default:
            std::abort();
        }
      }
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_broadcast(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      legion_assert(collective_mapping != nullptr);
      AddressSpaceID origin;
      derez.deserialize(origin);
      BroadcastMessageKind kind;
      derez.deserialize(kind);
      size_t message_size;
      derez.deserialize(message_size);
      const void* message = derez.get_current_pointer();
      derez.advance_pointer(message_size);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      // First pack it out and send it out to any remote nodes
      std::vector<AddressSpaceID> targets;
      collective_mapping->get_children(origin, local_space, targets);
      std::set<RtEvent> remote_handled;
      if (!targets.empty())
      {
        for (const AddressSpaceID& target : targets)
        {
          RtEvent next_done = Runtime::create_rt_user_event();
          ReplBroadcastUpdate rez;
          rez.serialize(did);
          rez.serialize(origin);
          rez.serialize(kind);
          rez.serialize<size_t>(message_size);
          rez.serialize(message, message_size);
          rez.serialize(next_done);
          rez.dispatch(target);
          remote_handled.insert(next_done);
        }
      }
      // Handle it on all our local shards
      for (ShardTask* shard : local_shards)
      {
        Deserializer derez2(message, message_size);
        switch (kind)
        {
          case RESOURCE_UPDATE_KIND:
            {
              shard->get_replicate_context()->handle_resource_update(
                  derez2, remote_handled);
              break;
            }
          case CREATED_REGION_UPDATE_KIND:
            {
              shard->get_replicate_context()->handle_created_region_contexts(
                  derez2, remote_handled);
              break;
            }
          default:
            std::abort();
        }
      }
      if (!remote_handled.empty())
        Runtime::trigger_event(
            done_event, Runtime::merge_events(remote_handled));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_trace_event_request(
        ShardedPhysicalTemplate* physical_template, ShardID shard_source,
        AddressSpaceID template_source, size_t template_index, ApEvent event,
        AddressSpaceID event_space, RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      // See whether we are on the right node to handle this request, if not
      // then forward the request onto the proper node
      if (event_space != runtime->address_space)
      {
        legion_assert(template_source == runtime->address_space);
        // Check to see if we have a shard on that address space, if not
        // then we know that this event can't have come from there
        bool found = false;
        for (unsigned idx = 0; idx < address_spaces->size(); idx++)
        {
          if ((*address_spaces)[idx] != event_space)
            continue;
          found = true;
          break;
        }
        if (found)
        {
          ReplTraceEventRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(physical_template);
            rez.serialize(template_index);
            rez.serialize(shard_source);
            rez.serialize(event);
            rez.serialize(done_event);
          }
          rez.dispatch(event_space);
        }
        else
          send_trace_event_response(
              physical_template, template_source, event,
              ApBarrier::NO_AP_BARRIER, done_event);
      }
      else
      {
        // Ask each of our local shards to check for the event in the template
        for (ShardTask* shard : local_shards)
        {
          const ApBarrier result =
              shard->get_replicate_context()->handle_find_trace_shard_event(
                  template_index, event, shard_source);
          // If we found it then we are done
          if (result.exists())
          {
            send_trace_event_response(
                physical_template, template_source, event, result, done_event);
            return;
          }
        }
        // If we make it here then we didn't find it so return the result
        send_trace_event_response(
            physical_template, template_source, event, ApBarrier::NO_AP_BARRIER,
            done_event);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplTraceEventRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID repl_id;
      derez.deserialize(repl_id);
      ShardedPhysicalTemplate* physical_template;
      derez.deserialize(physical_template);
      size_t template_index;
      derez.deserialize(template_index);
      ShardID shard_source;
      derez.deserialize(shard_source);
      ApEvent event;
      derez.deserialize(event);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      ShardManager* manager = runtime->find_shard_manager(repl_id);
      manager->send_trace_event_request(
          physical_template, shard_source, source, template_index, event,
          runtime->address_space, done_event);
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_trace_event_response(
        ShardedPhysicalTemplate* physical_template, AddressSpaceID temp_source,
        ApEvent event, ApBarrier result, RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      if (temp_source != runtime->address_space)
      {
        // Not local so send the response message
        ReplTraceEventResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(physical_template);
          rez.serialize(event);
          rez.serialize(result);
          rez.serialize(done_event);
        }
        rez.dispatch(temp_source);
      }
      else  // This is local so handle it here
      {
        physical_template->record_trace_shard_event(event, result);
        Runtime::trigger_event(done_event);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplTraceEventResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      ShardedPhysicalTemplate* physical_template;
      derez.deserialize(physical_template);
      ApEvent event;
      derez.deserialize(event);
      ApBarrier result;
      derez.deserialize(result);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      physical_template->record_trace_shard_event(event, result);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    RtEvent ShardManager::send_trace_event_trigger(
        TraceID tid, AddressSpaceID target, ApUserEvent lhs, ApEvent rhs,
        const TraceLocalID& tlid)
    //--------------------------------------------------------------------------
    {
      if (target != local_space)
      {
        legion_assert(collective_mapping != nullptr);
        legion_assert(collective_mapping->contains(target));
        const RtUserEvent done = Runtime::create_rt_user_event();
        ReplTraceEventTrigger rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(tid);
          rez.serialize(lhs);
          rez.serialize(rhs);
          tlid.serialize(rez);
          rez.serialize(done);
        }
        rez.dispatch(target);
        return done;
      }
      else
      {
        for (ShardTask* shard : local_shards)
        {
          ReplicateContext* ctx = shard->get_replicate_context();
          ShardedPhysicalTemplate* tpl = ctx->find_current_shard_template(tid);
          if (tpl->record_shard_event_trigger(lhs, rhs, tlid))
            return RtEvent::NO_RT_EVENT;
        }
        // Should never get here, we shold always find it
        std::abort();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplTraceEventTrigger::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID repl_id;
      derez.deserialize(repl_id);
      TraceID tid;
      derez.deserialize(tid);
      ApUserEvent lhs;
      derez.deserialize(lhs);
      ApEvent rhs;
      derez.deserialize(rhs);
      TraceLocalID tlid;
      tlid.deserialize(derez);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      ShardManager* manager = runtime->find_shard_manager(repl_id);
      RtEvent done = manager->send_trace_event_trigger(
          tid, runtime->address_space, lhs, rhs, tlid);
      Runtime::trigger_event(done_event, done);
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_trace_frontier_request(
        ShardedPhysicalTemplate* physical_template, ShardID shard_source,
        AddressSpaceID template_source, size_t template_index, ApEvent event,
        AddressSpaceID event_space, unsigned frontier, RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      // See whether we are on the right node to handle this request, if not
      // then forward the request onto the proper node
      if (event_space != runtime->address_space)
      {
        legion_assert(template_source == runtime->address_space);
        // Check to see if we have a shard on that address space, if not
        // then we know that this event can't have come from there
        bool found = false;
        for (unsigned idx = 0; idx < address_spaces->size(); idx++)
        {
          if ((*address_spaces)[idx] != event_space)
            continue;
          found = true;
          break;
        }
        if (found)
        {
          ReplTraceFrontierRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(physical_template);
            rez.serialize(template_index);
            rez.serialize(shard_source);
            rez.serialize(event);
            rez.serialize(frontier);
            rez.serialize(done_event);
          }
          rez.dispatch(event_space);
        }
        else
          send_trace_frontier_response(
              physical_template, template_source, frontier,
              ApBarrier::NO_AP_BARRIER, done_event);
      }
      else
      {
        // Ask each of our local shards to check for the event in the template
        for (ShardTask* shard : local_shards)
        {
          const ApBarrier result =
              shard->get_replicate_context()->handle_find_trace_shard_frontier(
                  template_index, event, shard_source);
          // If we found it then we are done
          if (result.exists())
          {
            send_trace_frontier_response(
                physical_template, template_source, frontier, result,
                done_event);
            return;
          }
        }
        // If we couldn't find it then send back a NO_BARRIER
        send_trace_frontier_response(
            physical_template, template_source, frontier,
            ApBarrier::NO_AP_BARRIER, done_event);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplTraceFrontierRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID repl_id;
      derez.deserialize(repl_id);
      ShardedPhysicalTemplate* physical_template;
      derez.deserialize(physical_template);
      size_t template_index;
      derez.deserialize(template_index);
      ShardID shard_source;
      derez.deserialize(shard_source);
      ApEvent event;
      derez.deserialize(event);
      unsigned frontier;
      derez.deserialize(frontier);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      ShardManager* manager = runtime->find_shard_manager(repl_id);
      manager->send_trace_frontier_request(
          physical_template, shard_source, source, template_index, event,
          runtime->address_space, frontier, done_event);
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_trace_frontier_response(
        ShardedPhysicalTemplate* physical_template, AddressSpaceID temp_source,
        unsigned frontier, ApBarrier result, RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      if (temp_source != runtime->address_space)
      {
        // Not local so send the response message
        ReplTraceFrontierResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(physical_template);
          rez.serialize(frontier);
          rez.serialize(result);
          rez.serialize(done_event);
        }
        rez.dispatch(temp_source);
      }
      else  // This is local so handle it here
      {
        physical_template->record_trace_shard_frontier(frontier, result);
        Runtime::trigger_event(done_event);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplTraceFrontierResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      ShardedPhysicalTemplate* physical_template;
      derez.deserialize(physical_template);
      unsigned frontier;
      derez.deserialize(frontier);
      ApBarrier result;
      derez.deserialize(result);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      physical_template->record_trace_shard_frontier(frontier, result);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_trace_update(
        ShardID target, const ReplTraceUpdateMessage& rez)
    //--------------------------------------------------------------------------
    {
      legion_assert(target < address_spaces->size());
      AddressSpaceID target_space = (*address_spaces)[target];
      // Check to see if this is a local shard
      if (target_space == runtime->address_space)
      {
        Deserializer derez(rez.get_payload(), rez.get_payload_size());
        // Have to unpack the preample we already know
        DistributedID local_repl;
        derez.deserialize(local_repl);
        handle_trace_update(derez, target_space);
      }
      else
        rez.dispatch(target_space);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_trace_update(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      // Figure out which shard we are going to
      ShardID target;
      derez.deserialize(target);
      for (ShardTask* shard : local_shards)
      {
        if (shard->shard_id == target)
        {
          shard->get_replicate_context()->handle_trace_update(derez, source);
          return;
        }
      }
      // Should never get here
      std::abort();
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_find_trace_local_sets(
        ShardID target, const ReplFindTraceSets& rez)
    //--------------------------------------------------------------------------
    {
      legion_assert(target < address_spaces->size());
      AddressSpaceID target_space = (*address_spaces)[target];
      // Check to see if this is a local shard
      if (target_space == runtime->address_space)
      {
        Deserializer derez(rez.get_payload(), rez.get_payload_size());
        // Have to unpack the preample we already know
        DistributedID local_repl;
        derez.deserialize(local_repl);
        handle_find_trace_local_sets(derez, target_space);
      }
      else
        rez.dispatch(target_space);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_find_trace_local_sets(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      // Figure out which shard we are going to
      ShardID target;
      derez.deserialize(target);
      for (ShardTask* shard : local_shards)
      {
        if (shard->shard_id == target)
        {
          shard->get_replicate_context()->handle_find_trace_local_sets(
              derez, source);
          return;
        }
      }
      // Should never get here
      std::abort();
    }

    //--------------------------------------------------------------------------
    ShardID ShardManager::find_collective_owner(RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      // This is the node that made the logical region tree
      const AddressSpaceID tree_owner = RegionTreeNode::get_owner_space(tid);
      // This is the node in the replicate context that will handle
      // all the collective instance creations for this region tree
      const AddressSpaceID target_owner =
          collective_mapping->contains(tree_owner) ?
              tree_owner :
              collective_mapping->find_nearest(tree_owner);
      // We'll just assign all view creation to the first shard on this node
      const ShardMapping& mapping = get_mapping();
      for (ShardID shard = 0; shard < mapping.size(); shard++)
        if (mapping[shard] == target_owner)
          return shard;
      std::abort();
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_find_or_create_collective_view(
        ShardID target, const ReplFindCollectiveView& rez)
    //--------------------------------------------------------------------------
    {
      legion_assert(target < address_spaces->size());
      AddressSpaceID target_space = (*address_spaces)[target];
      // Check to see if this is a local shard
      if (target_space == runtime->address_space)
      {
        Deserializer derez(rez.get_payload(), rez.get_payload_size());
        // Have to unpack the preample we already know
        DistributedID local_repl;
        derez.deserialize(local_repl);
        handle_find_or_create_collective_view(derez);
      }
      else
        rez.dispatch(target_space);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_find_or_create_collective_view(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      ShardID target;
      derez.deserialize(target);
      for (ShardTask* local_shard : local_shards)
      {
        if (local_shard->shard_id == target)
        {
          InnerContext* context = local_shard->get_replicate_context();
          RegionTreeID tid;
          derez.deserialize(tid);
          size_t num_insts;
          derez.deserialize(num_insts);
          std::vector<DistributedID> instances(num_insts);
          for (unsigned idx = 0; idx < num_insts; idx++)
            derez.deserialize(instances[idx]);
          InnerContext::CollectiveResult* target;
          derez.deserialize(target);
          AddressSpaceID source;
          derez.deserialize(source);
          RtUserEvent to_trigger;
          derez.deserialize(to_trigger);
          RtEvent ready;
          InnerContext::CollectiveResult* result =
              context->find_or_create_collective_view(tid, instances, ready);
          if (ready.exists() && !ready.has_triggered())
            ready.wait();
          RemoteContextFindCollectiveViewResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(target);
            rez.serialize(result->collective_did);
            rez.serialize(result->ready_event);
            rez.serialize(to_trigger);
          }
          rez.dispatch(source);
          if (result->remove_reference())
            delete result;
          return;
        }
      }
      // Should never get here
      std::abort();
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplFindCollectiveView::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID repl_id;
      derez.deserialize(repl_id);
      ShardManager* manager = runtime->find_shard_manager(repl_id);
      manager->handle_find_or_create_collective_view(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplPointwiseDependence::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      uint64_t context_index;
      derez.deserialize(context_index);
      DomainPoint point;
      derez.deserialize(point);
      ShardID shard;
      derez.deserialize(shard);
      bool intra_space;
      derez.deserialize(intra_space);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      std::optional<unsigned> output_index;
      derez.deserialize(output_index);
      ShardManager* manager = runtime->find_shard_manager(did);
      manager->find_pointwise_dependence(
          context_index, point, shard, intra_space, to_trigger, output_index);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateDistribution::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID repl_id;
      derez.deserialize(repl_id);
      Domain shard_domain;
      derez.deserialize(shard_domain);
      size_t total_shards;
      derez.deserialize(total_shards);
      std::vector<DomainPoint> shard_points(total_shards);
      std::vector<DomainPoint> sorted_points(total_shards);
      std::vector<ShardID> shard_lookup(total_shards);
      bool isomorphic_points;
      derez.deserialize(isomorphic_points);
      if (isomorphic_points)
      {
        for (unsigned idx = 0; idx < total_shards; idx++)
        {
          derez.deserialize(shard_points[idx]);
          sorted_points[idx] = shard_points[idx];
          shard_lookup[idx] = idx;
        }
      }
      else
      {
        for (unsigned idx = 0; idx < total_shards; idx++)
        {
          derez.deserialize(sorted_points[idx]);
          derez.deserialize(shard_lookup[idx]);
          shard_points[shard_lookup[idx]] = sorted_points[idx];
        }
      }
      bool top_level_task, control_replicated;
      derez.deserialize(top_level_task);
      derez.deserialize(control_replicated);
      RtBarrier callback_barrier;
      derez.deserialize(callback_barrier);
      size_t num_spaces;
      derez.deserialize(num_spaces);
      legion_assert(num_spaces > 0);
      CollectiveMapping* mapping = new CollectiveMapping(derez, num_spaces);
      unsigned local_shards = 0;
      std::vector<Processor> target_processors(total_shards);
      for (unsigned idx = 0; idx < total_shards; idx++)
      {
        derez.deserialize(target_processors[idx]);
        if (target_processors[idx].address_space() == runtime->address_space)
          local_shards++;
      }
      Mapper::ContextConfigOutput context_configuration;
      if (control_replicated)
        derez.deserialize(context_configuration);
      ShardManager* manager = new ShardManager(
          repl_id, mapping, local_shards, context_configuration, top_level_task,
          isomorphic_points, control_replicated, shard_domain,
          std::move(shard_points), std::move(sorted_points),
          std::move(shard_lookup), nullptr /*original*/, callback_barrier);
      bool explicit_distribution;
      derez.deserialize<bool>(explicit_distribution);
      if (explicit_distribution)
      {
        VariantID variant;
        derez.deserialize(variant);
        std::vector<VariantID> leaf_variants((variant == 0) ? total_shards : 0);
        for (unsigned idx = 0; idx < leaf_variants.size(); idx++)
          derez.deserialize(leaf_variants[idx]);
        InnerContext* parent_ctx = InnerContext::unpack_inner_context(derez);
        // Create the local shards
        ShardTask* first_shard = nullptr;
        for (unsigned idx = 0; idx < total_shards; idx++)
        {
          if (target_processors[idx].address_space() != runtime->address_space)
            continue;
          if (first_shard == nullptr)
            first_shard = manager->create_shard(
                idx, target_processors[idx],
                leaf_variants.empty() ? variant : leaf_variants[idx],
                parent_ctx, derez);
          else
            manager->create_shard(
                idx, target_processors[idx],
                leaf_variants.empty() ? variant : leaf_variants[idx],
                parent_ctx, first_shard);
        }
        manager->distribute_explicit(
            first_shard, variant, target_processors, leaf_variants);
      }
      else
      {
        TaskID task_id;
        derez.deserialize(task_id);
        MapperID mapper_id;
        derez.deserialize(mapper_id);
        Processor::Kind kind;
        derez.deserialize(kind);
        unsigned shards_per_space;
        derez.deserialize(shards_per_space);
        DistributedID ctx_did;
        derez.deserialize(ctx_did);
        Processor exec_proc;
        derez.deserialize(exec_proc);
        coord_t implicit_coord;
        derez.deserialize(implicit_coord);
        // This is a top-level implicit context so we know we can make
        // a new TopLevelContext here directly
        TopLevelContext* top_context = new TopLevelContext(
            exec_proc, 0 /*normal id*/, implicit_coord, ctx_did, mapping);
        top_context->register_with_runtime();
        manager->set_shard_mapping(target_processors);
        // Continue the distribution on to the other nodes
        manager->distribute_implicit(
            task_id, mapper_id, kind, shards_per_space, top_context);
        ImplicitShardManager* implicit_manager =
            runtime->find_implicit_shard_manager(
                task_id, mapper_id, kind, shards_per_space);
        RtUserEvent to_trigger =
            implicit_manager->set_shard_manager(manager, top_context);
        if (to_trigger.exists())
          Runtime::trigger_event(to_trigger);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateVersioning::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      unsigned index, parent_req_index;
      derez.deserialize(index);
      derez.deserialize(parent_req_index);
      op::map<LogicalRegion, ShardManager::RegionVersioning> to_perform;
      ShardManager::unpack_collective_versioning(derez, to_perform);

      ShardManager* manager = runtime->find_shard_manager(did);
      manager->rendezvous_collective_versioning_analysis(
          index, parent_req_index, to_perform);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateCollectiveMapping::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      ShardManager::RendezvousKey key;
      derez.deserialize(key.region_index);
      derez.deserialize(key.analysis);
      std::map<LogicalRegion, ShardManager::CollectiveRendezvous> rendezvous;
      ShardManager::unpack_collective_rendezvous(derez, rendezvous);

      ShardManager* manager = runtime->find_shard_manager(did);
      manager->rendezvous_collective_mapping(key, rendezvous);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateVirtualRendezvous::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      ShardID shard;
      derez.deserialize(shard);
      size_t num_mappings;
      derez.deserialize(num_mappings);
      std::vector<bool> virtual_mappings(num_mappings);
      for (unsigned idx = 0; idx < num_mappings; idx++)
      {
        bool virtual_mapping;
        derez.deserialize(virtual_mapping);
        virtual_mappings[idx] = virtual_mapping;
      }

      ShardManager* manager = runtime->find_shard_manager(did);
      manager->rendezvous_check_virtual_mappings(
          shard, nullptr, virtual_mappings);
      manager->unpack_global_ref();
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateStartup::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID repl_id;
      derez.deserialize(repl_id);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      ShardManager* manager = runtime->find_shard_manager(repl_id);
      Runtime::trigger_event(
          to_trigger,
          manager->complete_startup_initialization(false /*local*/));
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicatePostMapped::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID repl_id;
      derez.deserialize(repl_id);
      RtEvent precondition;
      derez.deserialize(precondition);
      ShardManager* manager = runtime->find_shard_manager(repl_id);
      manager->handle_post_mapped(false /*local*/, precondition);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateTriggerComplete::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID repl_id;
      derez.deserialize(repl_id);
      ApEvent all_shards_done;
      derez.deserialize(all_shards_done);
      ApUserEvent all_shards_complete;
      derez.deserialize(all_shards_complete);
      ShardManager* manager = runtime->find_shard_manager(repl_id);
      ApEvent complete =
          manager->trigger_task_complete(false /*local*/, all_shards_done);
      if (all_shards_complete.exists())
        Runtime::trigger_event_untraced(all_shards_complete, complete);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateTriggerCommit::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID repl_id;
      derez.deserialize(repl_id);
      ShardManager* manager = runtime->find_shard_manager(repl_id);
      RtEvent commit_precondition;
      derez.deserialize(commit_precondition);
      manager->trigger_task_commit(false /*local*/, commit_precondition);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardCollectiveMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID repl_id;
      derez.deserialize(repl_id);
      ShardManager* manager = runtime->find_shard_manager(repl_id);
      manager->handle_collective_message(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateRendezvousMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID repl_id;
      derez.deserialize(repl_id);
      ShardManager* manager = runtime->find_shard_manager(repl_id);
      manager->handle_rendezvous_message(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplTraceUpdateMessage::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DistributedID repl_id;
      derez.deserialize(repl_id);
      ShardManager* manager = runtime->find_shard_manager(repl_id);
      manager->handle_trace_update(derez, source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplFindTraceSets::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DistributedID repl_id;
      derez.deserialize(repl_id);
      ShardManager* manager = runtime->find_shard_manager(repl_id);
      manager->handle_find_trace_local_sets(derez, source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplComputeEquivalenceSets::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID repl_id;
      derez.deserialize(repl_id);
      ShardManager* manager = runtime->find_shard_manager(repl_id);
      manager->handle_compute_equivalence_sets(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplOutputEquivalenceSet::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID repl_id;
      derez.deserialize(repl_id);
      ShardManager* manager = runtime->find_shard_manager(repl_id);
      manager->handle_output_equivalence_set(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplOutputOffset::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID repl_id;
      derez.deserialize(repl_id);
      ShardManager* manager = runtime->find_shard_manager(repl_id);
      manager->handle_output_offset(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplRefineEquivalenceSets::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID repl_id;
      derez.deserialize(repl_id);
      ShardManager* manager = runtime->find_shard_manager(repl_id);
      manager->handle_refine_equivalence_sets(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplEquivalenceSetNotification::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID repl_id;
      derez.deserialize(repl_id);
      ShardManager* manager = runtime->find_shard_manager(repl_id);
      manager->handle_equivalence_set_notification(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplBroadcastUpdate::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID repl_id;
      derez.deserialize(repl_id);
      ShardManager* manager = runtime->find_shard_manager(repl_id);
      manager->handle_broadcast(derez);
    }

    //--------------------------------------------------------------------------
    ShardingFunction* ShardManager::find_sharding_function(
        ShardingID sid, bool skip_checks)
    //--------------------------------------------------------------------------
    {
      // Check to see if it is in the cache
      {
        AutoLock m_lock(manager_lock, false /*exclusive*/);
        std::map<ShardingID, ShardingFunction*>::const_iterator finder =
            sharding_functions.find(sid);
        if (finder != sharding_functions.end())
          return finder->second;
      }
      // Get the functor from the runtime
      ShardingFunctor* functor = runtime->find_sharding_functor(sid);
      // Retake the lock
      AutoLock m_lock(manager_lock);
      // See if we lost the race
      std::map<ShardingID, ShardingFunction*>::const_iterator finder =
          sharding_functions.find(sid);
      if (finder != sharding_functions.end())
        return finder->second;
      ShardingFunction* result =
          new ShardingFunction(functor, this, sid, skip_checks);
      // Save the result for the future
      sharding_functions[sid] = result;
      return result;
    }

#ifdef LEGION_USE_LIBDL
    //--------------------------------------------------------------------------
    void ShardManager::perform_global_registration_callbacks(
        Realm::DSOReferenceImplementation* dso, const void* buffer,
        size_t buffer_size, bool withargs, size_t dedup_tag, RtEvent local_done,
        RtEvent global_done, std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      legion_assert(control_replicated);
      // See if we're the first one to handle this DSO
      const Runtime::RegistrationKey key(
          dedup_tag, dso->dso_name, dso->symbol_name);
      {
        AutoLock m_lock(manager_lock);
        // Check to see if we've already handled this
        std::set<Runtime::RegistrationKey>::const_iterator finder =
            unique_registration_callbacks.find(key);
        if (finder != unique_registration_callbacks.end())
          return;
        unique_registration_callbacks.insert(key);
      }
      // We're the first one so handle it
      if (!is_total_sharding())
      {
        std::set<RtEvent> local_preconditions;
        if (collective_mapping != nullptr)
        {
          legion_assert(collective_mapping->contains(local_space));
          const unsigned index = collective_mapping->find_index(local_space);
          for (AddressSpaceID space = 0; space < runtime->total_address_spaces;
               space++)
          {
            if ((collective_mapping != nullptr) &&
                collective_mapping->contains(space))
              continue;
            if ((space % collective_mapping->size()) == index)
              runtime->send_registration_callback(
                  space, dso, global_done, local_preconditions, buffer,
                  buffer_size, withargs, true /*deduplicate*/, dedup_tag);
          }
        }
        else
        {
          // Just send it to everyone
          for (AddressSpaceID space = 0; space < runtime->total_address_spaces;
               space++)
          {
            if (space == local_space)
              continue;
            runtime->send_registration_callback(
                space, dso, global_done, local_preconditions, buffer,
                buffer_size, withargs, true /*deduplicate*/, dedup_tag);
          }
        }
        if (!local_preconditions.empty())
        {
          local_preconditions.insert(local_done);
          runtime->phase_barrier_arrive(
              callback_barrier, 1 /*count*/,
              Runtime::merge_events(local_preconditions));
        }
        else
          runtime->phase_barrier_arrive(
              callback_barrier, 1 /*count*/, local_done);
      }
      else  // there will be a callback on every node anyway
        runtime->phase_barrier_arrive(
            callback_barrier, 1 /*count*/, local_done);
      preconditions.insert(callback_barrier);
      Runtime::advance_barrier(callback_barrier);
      if (!callback_barrier.exists())
      {
        Fatal fatal;
        fatal << "Need support for refreshing exhausted callback phase "
              << "barrier generations.";
        fatal.raise();
      }
    }
#endif  // LEGION_USE_LIBDL

    //--------------------------------------------------------------------------
    bool ShardManager::perform_semantic_attach(void)
    //--------------------------------------------------------------------------
    {
      if (local_shards.size() == 1)
        return true;
      AutoLock m_lock(manager_lock);
      legion_assert(semantic_attach_counter < local_shards.size());
      if (++semantic_attach_counter == local_shards.size())
      {
        semantic_attach_counter = 0;
        return true;
      }
      else
        return false;
    }

    /////////////////////////////////////////////////////////////
    // Implicit Shard Manager
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ImplicitShardManager::ImplicitShardManager(
        TaskID tid, MapperID mid, Processor::Kind k, unsigned shards_per_space)
      : Collectable(), task_id(tid), mapper_id(mid), kind(k),
        shards_per_address_space(shards_per_space),
        remaining_local_arrivals(shards_per_space), local_shard_id(0),
        top_context(nullptr), shard_manager(nullptr),
        collective_mapping(nullptr), local_task_name(nullptr)
    //--------------------------------------------------------------------------
    {
      legion_assert(runtime->total_address_spaces > 0);
      std::vector<AddressSpaceID> spaces(runtime->total_address_spaces);
      for (unsigned idx = 0; idx < spaces.size(); idx++) spaces[idx] = idx;
      collective_mapping =
          new CollectiveMapping(spaces, runtime->legion_collective_radix);
      collective_mapping->add_reference();
      remaining_remote_arrivals =
          collective_mapping->count_children(0, runtime->address_space);
    }

    //--------------------------------------------------------------------------
    ImplicitShardManager::~ImplicitShardManager(void)
    //--------------------------------------------------------------------------
    {
      runtime->unregister_implicit_shard_manager(task_id);
      if (collective_mapping->remove_reference())
        delete collective_mapping;
    }

    //--------------------------------------------------------------------------
    ShardTask* ImplicitShardManager::create_shard(
        int shard_id, const DomainPoint& point, Processor proxy,
        const char* task_name)
    //--------------------------------------------------------------------------
    {
      // Do our registrations and then wait for the shard manager to be ready
      ShardTask* task = nullptr;
      {
        AutoLock m_lock(manager_lock);
        legion_assert(local_shard_id < shards_per_address_space);
        local_proxy = proxy;
        local_task_name = task_name;
        const ShardID shard =
            (shard_id < 0) ?
                (runtime->address_space * shards_per_address_space +
                 local_shard_id++) :
                shard_id;
        const size_t total_shards =
            shards_per_address_space * runtime->total_address_spaces;
        if (total_shards <= shard)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "All shard IDs must be contained within [0," << total_shards
                << ") for implicit "
                << "control replicated task " << task_name;
          error.raise();
        }
        const DomainPoint shard_point =
            (point.get_dim() > 0) ? point : DomainPoint(shard);
        const bool result = shard_points
                                .emplace(std::make_pair(
                                    shard_point, std::make_pair(shard, proxy)))
                                .second;
        if (!result)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error
              << "Discovered multiple ranks with the same implicit shard point "
              << "for implicit control replicated task " << task_name;
          error.raise();
        }
        if (remaining_local_arrivals == 0)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Too many arrivals for implicit control replicated task "
                << task_name << ". "
                << "Only " << shards_per_address_space << " are permitted.";
          error.raise();
        }
        if ((--remaining_local_arrivals == 0) &&
            (remaining_remote_arrivals == 0))
        {
          if (runtime->address_space > 0)
            request_shard_manager();
          else
            create_shard_manager();
        }
        if (shard_manager == nullptr)
        {
          if (!manager_ready.exists())
            manager_ready = Runtime::create_rt_user_event();
          const RtEvent wait_on = manager_ready;
          m_lock.release();
          wait_on.wait();
          m_lock.reacquire();
        }
        legion_assert(top_context != nullptr);
        legion_assert(shard_manager != nullptr);
        task = shard_manager->create_shard(
            shard, proxy, 0 /*variant id*/, top_context, nullptr /*source*/);
      }
      top_context->increment_pending();
      implicit_context = top_context;
      task->initialize_implicit_task(task_id, mapper_id, proxy);
      return task;
    }

    //--------------------------------------------------------------------------
    void ImplicitShardManager::create_shard_manager(void)
    //--------------------------------------------------------------------------
    {
      const size_t total_shards =
          runtime->total_address_spaces * shards_per_address_space;
      legion_assert(runtime->address_space == 0);
      legion_assert(top_context == nullptr);
      legion_assert(shard_manager == nullptr);
      legion_assert(shard_points.size() == total_shards);
      IndividualTask* implicit_top = runtime->create_implicit_top_level(
          task_id, mapper_id, local_proxy, local_task_name, collective_mapping);
      top_context =
          legion_safe_cast<TopLevelContext*>(implicit_top->get_context());
      // Now we need to make the shard manager
      const DistributedID repl_context =
          runtime->get_available_distributed_id();
      // Fill in the shard points
      std::vector<DomainPoint> points(total_shards);
      std::vector<DomainPoint> sorted_points;
      sorted_points.reserve(total_shards);
      std::vector<ShardID> shard_lookup;
      shard_lookup.reserve(total_shards);
      bool isomorphic_points = true;
      // Should not be any duplicate shard domains
      if (shard_points.size() != total_shards)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Discovered multiple ranks with the same implicit shard point "
              << "for implicit control replicated task " << local_task_name;
        error.raise();
      }
      std::vector<Processor> shard_mapping(total_shards);
      for (const std::pair<const DomainPoint, std::pair<ShardID, Processor> >&
               it : shard_points)
      {
        if (isomorphic_points &&
            ((it.first.get_dim() != 1) || (it.first[0] != it.second.first)))
          isomorphic_points = false;
        sorted_points.emplace_back(it.first);
        shard_lookup.emplace_back(it.second.first);
        legion_assert(it.second.first < points.size());
        // Should not be any duplicate shard IDs
        if (points[it.second.first].get_dim() > 0)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Discovered multiple ranks with the same implicit shard ID "
                << "for implicit control replicated task " << local_task_name;
          error.raise();
        }
        points[it.second.first] = it.first;
        shard_mapping[it.second.first] = it.second.second;
      }
      Domain shard_domain;
      if (isomorphic_points)
        shard_domain = Domain(DomainPoint(0), DomainPoint(total_shards - 1));
      Mapper::ContextConfigOutput configuration;
      implicit_top->configure_execution_context(configuration);
      // The shard manager will take ownership of this
      ShardManager* manager = new ShardManager(
          repl_context, collective_mapping, shards_per_address_space,
          configuration, true /*top level*/, isomorphic_points,
          true /*control replicated*/, shard_domain, std::move(points),
          std::move(sorted_points), std::move(shard_lookup), implicit_top);
      shard_manager = manager;
      implicit_top->set_shard_manager(manager);
      manager->set_shard_mapping(shard_mapping);
      LegionSpy::log_replication(
          implicit_top->get_unique_id(), repl_context,
          true /*control replication*/);
      manager->distribute_implicit(
          task_id, mapper_id, kind, shards_per_address_space, top_context);
      if (manager_ready.exists())
        Runtime::trigger_event(manager_ready);
    }

    //--------------------------------------------------------------------------
    void ImplicitShardManager::request_shard_manager(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(shard_manager == nullptr);
      legion_assert(runtime->address_space > 0);
      ReplImplicitRendezvous rez;
      {
        RezCheck z(rez);
        rez.serialize(task_id);
        rez.serialize(mapper_id);
        rez.serialize(kind);
        rez.serialize(shards_per_address_space);
        rez.serialize<size_t>(shard_points.size());
        for (const std::pair<const DomainPoint, std::pair<ShardID, Processor> >&
                 it : shard_points)
        {
          rez.serialize(it.first);
          rez.serialize(it.second.first);
          rez.serialize(it.second.second);
        }
      }
      rez.dispatch(collective_mapping->get_parent(0, runtime->address_space));
    }

    //--------------------------------------------------------------------------
    void ImplicitShardManager::process_implicit_rendezvous(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      size_t num_points = 0;
      derez.deserialize(num_points);
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        DomainPoint point;
        derez.deserialize(point);
        legion_assert(shard_points.find(point) == shard_points.end());
        std::pair<ShardID, Processor>& pair = shard_points[point];
        derez.deserialize(pair.first);
        derez.deserialize(pair.second);
      }
      legion_assert(remaining_remote_arrivals > 0);
      if ((--remaining_remote_arrivals == 0) && (remaining_local_arrivals == 0))
      {
        if (runtime->address_space > 0)
          request_shard_manager();
        else
          create_shard_manager();
      }
    }

    //--------------------------------------------------------------------------
    RtUserEvent ImplicitShardManager::set_shard_manager(
        ShardManager* m, TopLevelContext* c)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      legion_assert(top_context == nullptr);
      legion_assert(shard_manager == nullptr);
      legion_assert(manager_ready.exists());
      top_context = c;
      shard_manager = m;
      RtUserEvent to_trigger = manager_ready;
      manager_ready = RtUserEvent::NO_RT_USER_EVENT;
      return to_trigger;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplImplicitRendezvous::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      TaskID task_id;
      derez.deserialize(task_id);
      MapperID mapper_id;
      derez.deserialize(mapper_id);
      Processor::Kind kind;
      derez.deserialize(kind);
      unsigned shards_per_address_space;
      derez.deserialize(shards_per_address_space);
      ImplicitShardManager* manager = runtime->find_implicit_shard_manager(
          task_id, mapper_id, kind, shards_per_address_space);
      manager->process_implicit_rendezvous(derez);
    }

  }  // namespace Internal
}  // namespace Legion
