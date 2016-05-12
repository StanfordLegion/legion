/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include "legion.h"
#include "runtime.h"
#include "legion_ops.h"
#include "legion_tasks.h"
#include "region_tree.h"
#include "legion_spy.h"
#include "legion_profiling.h"
#include "legion_instances.h"
#include "legion_views.h"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    // This is the shit right here: super-cool helper function

    //--------------------------------------------------------------------------
    template<unsigned LOG2MAX>
    static inline void compress_mask(FieldMask &x, FieldMask m)
    //--------------------------------------------------------------------------
    {
      FieldMask mk, mp, mv, t;
      // See hacker's delight 7-4
      x = x & m;
      mk = ~m << 1;
      for (unsigned i = 0; i < LOG2MAX; i++)
      {
        mp = mk ^ (mk << 1);
        for (unsigned idx = 1; idx < LOG2MAX; idx++)
          mp = mp ^ (mp << (1 << idx));
        mv = mp & m;
        m = (m ^ mv) | (mv >> (1 << i));
        t = x & mv;
        x = (x ^ t) | (t >> (1 << i));
        mk = mk & ~mp;
      }
    }

    /////////////////////////////////////////////////////////////
    // Copy Across Helper 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void CopyAcrossHelper::compute_across_offsets(const FieldMask &src_mask,
                               std::vector<Domain::CopySrcDstField> &dst_fields)
    //--------------------------------------------------------------------------
    {
      FieldMask compressed; 
      bool found_in_cache = false;
      for (LegionDeque<std::pair<FieldMask,FieldMask> >::aligned::const_iterator
            it = compressed_cache.begin(); it != compressed_cache.end(); it++)
      {
        if (it->first == src_mask)
        {
          compressed = it->second;
          found_in_cache = true;
          break;
        }
      }
      if (!found_in_cache)
      {
        compressed = src_mask;
        compress_mask<STATIC_LOG2(MAX_FIELDS)>(compressed, full_mask);
        compressed_cache.push_back(
            std::pair<FieldMask,FieldMask>(src_mask, compressed));
      }
      int pop_count = FieldMask::pop_count(compressed);
#ifdef DEBUG_LEGION
      assert(pop_count == FieldMask::pop_count(src_mask));
#endif
      unsigned offset = dst_fields.size();
      dst_fields.resize(offset + pop_count);
      int next_start = 0;
      for (int idx = 0; idx < pop_count; idx++)
      {
        int index = compressed.find_next_set(next_start);
        Domain::CopySrcDstField &field = dst_fields[offset+idx];
        field = offsets[index];
        // We'll start looking again at the next index after this one
        next_start = index + 1;
      }
    }

    /////////////////////////////////////////////////////////////
    // Layout Description 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LayoutDescription::LayoutDescription(FieldSpaceNode *own,
                                         const FieldMask &mask,
                                         LayoutConstraints *con,
                                   const std::vector<unsigned> &mask_index_map,
                                     const std::vector<CustomSerdezID> &serdez,
                    const std::vector<std::pair<FieldID,size_t> > &field_sizes)
      : allocated_fields(mask), constraints(con), owner(own) 
    //--------------------------------------------------------------------------
    {
      constraints->add_reference();
      layout_lock = Reservation::create_reservation();
      field_infos.resize(field_sizes.size());
      // Switch data structures from layout by field order to order
      // of field locations in the bit mask
#ifdef DEBUG_LEGION
      assert(mask_index_map.size() == 
                size_t(FieldMask::pop_count(allocated_fields)));
#endif
#ifndef NEW_INSTANCE_CREATION
      std::vector<size_t> offsets(field_sizes.size(),0);
      for (unsigned idx = 1; idx < field_sizes.size(); idx++)
        offsets[idx] = offsets[idx-1] + field_sizes[idx-1].second;
#endif
      for (unsigned idx = 0; idx < mask_index_map.size(); idx++)
      {
        // This gives us the index in the field ordered data structures
        unsigned index = mask_index_map[idx];
        FieldID fid = field_sizes[index].first;
        field_indexes[fid] = idx;
        Domain::CopySrcDstField &info = field_infos[idx];
        info.offset = offsets[index];
        info.size = field_sizes[index].second;
        info.field_id = fid;
        info.serdez_id = serdez[index];
      }
    }

    //--------------------------------------------------------------------------
    LayoutDescription::LayoutDescription(const FieldMask &mask,
                                         LayoutConstraints *con)
      : allocated_fields(mask), constraints(con), owner(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LayoutDescription::LayoutDescription(const LayoutDescription &rhs)
      : allocated_fields(rhs.allocated_fields), 
        constraints(rhs.constraints), owner(rhs.owner)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LayoutDescription::~LayoutDescription(void)
    //--------------------------------------------------------------------------
    {
      comp_cache.clear();
      layout_lock.destroy_reservation();
      layout_lock = Reservation::NO_RESERVATION;
      if (constraints->remove_reference())
        legion_delete(constraints);
    }

    //--------------------------------------------------------------------------
    LayoutDescription& LayoutDescription::operator=(
                                                   const LayoutDescription &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void* LayoutDescription::operator new(size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<LayoutDescription,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::operator delete(void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::log_instance_layout(PhysicalInstance inst) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(Runtime::legion_spy_enabled);
#endif
      std::vector<FieldID> fields;  
      owner->get_field_ids(allocated_fields, fields);
      for (std::vector<FieldID>::const_iterator it = fields.begin();
            it != fields.end(); it++)
        LegionSpy::log_physical_instance_field(inst.id, *it);
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::compute_copy_offsets(const FieldMask &copy_mask,
                                                 PhysicalInstance instance,
                                   std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
      uint64_t hash_key = copy_mask.get_hash_key();
      bool found_in_cache = false;
      FieldMask compressed;
      // First check to see if we've memoized this result 
      {
        AutoLock o_lock(layout_lock,1,false/*exclusive*/);
        std::map<LEGION_FIELD_MASK_FIELD_TYPE,
                 LegionList<std::pair<FieldMask,FieldMask> >::aligned>::
                   const_iterator finder = comp_cache.find(hash_key);
        if (finder != comp_cache.end())
        {
          for (LegionList<std::pair<FieldMask,FieldMask> >::aligned::
                const_iterator it = finder->second.begin(); 
                it != finder->second.end(); it++)
          {
            if (it->first == copy_mask)
            {
              found_in_cache = true;
              compressed = it->second;
              break;
            }
          }
        }
      }
      if (!found_in_cache)
      {
        compressed = copy_mask;
        compress_mask<STATIC_LOG2(MAX_FIELDS)>(compressed, allocated_fields);
        // Save the result in the cache, duplicates from races here are benign
        AutoLock o_lock(layout_lock);
        comp_cache[hash_key].push_back(
            std::pair<FieldMask,FieldMask>(copy_mask,compressed));
      }
      // It is absolutely imperative that these infos be added in
      // the order in which they appear in the field mask so that 
      // they line up in the same order with the source/destination infos
      // (depending on the calling context of this function
      int pop_count = FieldMask::pop_count(compressed);
#ifdef DEBUG_LEGION
      assert(pop_count == FieldMask::pop_count(copy_mask));
#endif
      unsigned offset = fields.size();
      fields.resize(offset + pop_count);
      int next_start = 0;
      for (int idx = 0; idx < pop_count; idx++)
      {
        int index = compressed.find_next_set(next_start);
        Domain::CopySrcDstField &field = fields[offset+idx];
        field = field_infos[index];
        // Our field infos are annonymous so specify the instance now
        field.inst = instance;
        // We'll start looking again at the next index after this one
        next_start = index + 1;
      }
    } 

    //--------------------------------------------------------------------------
    void LayoutDescription::compute_copy_offsets(FieldID fid, 
        PhysicalInstance instance, std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
      std::map<FieldID,unsigned>::const_iterator finder = 
        field_indexes.find(fid);
#ifdef DEBUG_LEGION
      assert(finder != field_indexes.end());
#endif
      fields.push_back(field_infos[finder->second]);
      // Since instances are annonymous in layout descriptions we
      // have to fill them in when we add the field info
      fields.back().inst = instance;
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::compute_copy_offsets(
                                   const std::vector<FieldID> &copy_fields, 
                                   PhysicalInstance instance,
                                   std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
      unsigned offset = fields.size();
      fields.resize(offset + copy_fields.size());
      for (unsigned idx = 0; idx < copy_fields.size(); idx++)
      {
        std::map<FieldID,unsigned>::const_iterator
          finder = field_indexes.find(copy_fields[idx]);
#ifdef DEBUG_LEGION
        assert(finder != field_indexes.end());
#endif
        Domain::CopySrcDstField &info = fields[offset+idx];
        info = field_infos[finder->second];
        // Since instances are annonymous in layout descriptions we
        // have to fill them in when we add the field info
        info.inst = instance;
      }
    }

    //--------------------------------------------------------------------------
    bool LayoutDescription::has_field(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      return (field_indexes.find(fid) != field_indexes.end());
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::has_fields(std::map<FieldID,bool> &to_test) const
    //--------------------------------------------------------------------------
    {
      for (std::map<FieldID,bool>::iterator it = to_test.begin();
            it != to_test.end(); it++)
      {
        if (field_indexes.find(it->first) != field_indexes.end())
          it->second = true;
        else
          it->second = false;
      }
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::remove_space_fields(std::set<FieldID> &filter) const
    //--------------------------------------------------------------------------
    {
      std::vector<FieldID> to_remove;
      for (std::set<FieldID>::const_iterator it = filter.begin();
            it != filter.end(); it++)
      {
        if (field_indexes.find(*it) != field_indexes.end())
          to_remove.push_back(*it);
      }
      if (!to_remove.empty())
      {
        for (std::vector<FieldID>::const_iterator it = to_remove.begin();
              it != to_remove.end(); it++)
          filter.erase(*it);
      }
    }

    //--------------------------------------------------------------------------
    const Domain::CopySrcDstField& LayoutDescription::find_field_info(
                                                              FieldID fid) const
    //--------------------------------------------------------------------------
    {
      std::map<FieldID,unsigned>::const_iterator finder = 
        field_indexes.find(fid);
#ifdef DEBUG_LEGION
      assert(finder != field_indexes.end());
#endif
      return field_infos[finder->second];
    }

    //--------------------------------------------------------------------------
    size_t LayoutDescription::get_total_field_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      // Add up all the field sizes
      for (std::vector<Domain::CopySrcDstField>::const_iterator it = 
            field_infos.begin(); it != field_infos.end(); it++)
      {
        result += it->size;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::get_fields(std::vector<FieldID>& fields) const
    //--------------------------------------------------------------------------
    {
      // order field ids by their offsets by inserting them to std::map
      std::map<unsigned, FieldID> offsets;
      for (std::map<FieldID,unsigned>::const_iterator it = 
            field_indexes.begin(); it != field_indexes.end(); it++)
      {
        const Domain::CopySrcDstField &info = field_infos[it->second];
        offsets[info.offset] = it->first;
      }
      for (std::map<unsigned, FieldID>::const_iterator it = offsets.begin();
           it != offsets.end(); ++it)
        fields.push_back(it->second);
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::compute_destroyed_fields(
             std::vector<PhysicalInstance::DestroyedField> &serdez_fields) const
    //--------------------------------------------------------------------------
    {
      // See if we have any special fields which need serdez deletion
      for (std::vector<Domain::CopySrcDstField>::const_iterator it = 
            field_infos.begin(); it != field_infos.end(); it++)
      {
        if (it->serdez_id > 0)
          serdez_fields.push_back(PhysicalInstance::DestroyedField(it->offset, 
                                                    it->size, it->serdez_id));
      }
    }

    //--------------------------------------------------------------------------
    bool LayoutDescription::match_layout(
                         const LayoutConstraintSet &candidate_constraints) const
    //--------------------------------------------------------------------------
    {
      // Layout descriptions are always complete, so just check for conflicts
      if (constraints->conflicts(candidate_constraints))
        return false;
      // If they don't conflict they have to be the same
      return true;
    }

    //--------------------------------------------------------------------------
    bool LayoutDescription::match_layout(const LayoutDescription *layout) const
    //--------------------------------------------------------------------------
    {
      if (layout->allocated_fields != allocated_fields)
        return false;
      // Layout descriptions are always complete so just check for conflicts
      if (constraints->conflicts(layout->constraints))
        return false;
      // If they don't conflict they have to be the same
      return true;
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::set_descriptor(FieldDataDescriptor &desc,
                                           FieldID fid) const
    //--------------------------------------------------------------------------
    {
      std::map<FieldID,unsigned>::const_iterator finder = 
        field_indexes.find(fid);
#ifdef DEBUG_LEGION
      assert(finder != field_indexes.end());
#endif
      const Domain::CopySrcDstField &info = field_infos[finder->second];
      desc.field_offset = info.offset;
      desc.field_size = info.size;
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::pack_layout_description(Serializer &rez,
                                                    AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      // Do a quick check to see if the target already has the layout
      // We don't need to hold a lock here since if we lose the race
      // we will just send the layout twice and everything will be
      // resolved on the far side
      if (known_nodes.contains(target))
        rez.serialize<bool>(true);
      else
        rez.serialize<bool>(false);
      rez.serialize(constraints->layout_id);
      rez.serialize(allocated_fields);
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::update_known_nodes(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Hold the lock to get serial access to this data structure
      AutoLock l_lock(layout_lock);
      known_nodes.add(target);
    }

    //--------------------------------------------------------------------------
    /*static*/ LayoutDescription* LayoutDescription::
      handle_unpack_layout_description(Deserializer &derez,
                                 AddressSpaceID source, RegionNode *region_node)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      bool has_local;
      derez.deserialize(has_local);
      FieldSpaceNode *field_space_node = region_node->column_source;
      LayoutDescription *result = NULL;
      LayoutConstraintID layout_id;
      derez.deserialize(layout_id);
      LayoutConstraints *constraints = 
        region_node->context->runtime->find_layout_constraints(layout_id);
      FieldMask mask;
      derez.deserialize(mask);
      if (has_local)
      {
        // If we have a local layout, then we should be able to find it
        result = field_space_node->find_layout_description(mask, constraints);
      }
      else
      {
        const std::vector<FieldID> &field_set = 
          constraints->field_constraint.get_field_set();
        std::vector<std::pair<FieldID,size_t> > field_sizes(field_set.size());
        std::vector<unsigned> mask_index_map(field_set.size());
        std::vector<CustomSerdezID> serdez(field_set.size());
        mask.clear();
        field_space_node->compute_create_offsets(field_set, field_sizes,
                                        mask_index_map, serdez, mask);
        result = field_space_node->create_layout_description(mask, constraints,
                                       mask_index_map, serdez, field_sizes);
      }
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      // Record that the sender already has this layout
      // Only do this after we've registered the instance
      result->update_known_nodes(source);
      return result;
    }

    /////////////////////////////////////////////////////////////
    // PhysicalManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalManager::PhysicalManager(RegionTreeForest *ctx,
                                     MemoryManager *memory, 
                                     LayoutDescription *desc,
                                     const PointerConstraint &constraint,
                                     DistributedID did,
                                     AddressSpaceID owner_space,
                                     AddressSpaceID local_space,
                                     RegionNode *node,
                                     PhysicalInstance inst, const Domain &d, 
                                     bool own, bool register_now)
      : DistributedCollectable(ctx->runtime, did, 
                               owner_space, local_space, register_now), 
        context(ctx), memory_manager(memory), region_node(node), layout(desc),
        instance(inst), instance_domain(d), 
        own_domain(own), pointer_constraint(constraint)
    //--------------------------------------------------------------------------
    {
      if (region_node != NULL)
        region_node->register_physical_manager(this);
      // If we are not the owner, add a resource reference
      if (!is_owner())
        add_base_resource_ref(REMOTE_DID_REF);
    }

    //--------------------------------------------------------------------------
    PhysicalManager::~PhysicalManager(void)
    //--------------------------------------------------------------------------
    {
      if (region_node != NULL)
        region_node->unregister_physical_manager(this);
      // If we're the owner remove our resource references
      if (is_owner())
      {
        UpdateReferenceFunctor<RESOURCE_REF_KIND,false/*add*/> functor(this);
        map_over_remote_instances(functor);
      }
      else
        memory_manager->unregister_remote_instance(this);
      if (is_owner() && instance.exists())
      {
        log_leak.warning("Leaking physical instance " IDFMT " in memory"
                               IDFMT "", instance.id, get_memory().id);
      }
      // If we own our domain, then we need to delete it now
      if (own_domain)
      {
        Realm::IndexSpace is = instance_domain.get_index_space();
        is.destroy();
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_active(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (is_owner())
        assert(instance.exists());
#endif
      // Will be null for virtual managers
      if (memory_manager != NULL)
        memory_manager->activate_instance(this);
      // If we are not the owner, send a reference
      if (!is_owner())
        send_remote_gc_update(owner_space, 1/*count*/, true/*add*/);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_inactive(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (is_owner())
        assert(instance.exists());
#endif
      // Will be null for virtual managers
      if (memory_manager != NULL)
        memory_manager->deactivate_instance(this);
      if (!is_owner())
        send_remote_gc_update(owner_space, 1/*count*/, false/*add*/);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_valid(void)
    //--------------------------------------------------------------------------
    {
      // No need to do anything
#ifdef DEBUG_LEGION
      if (is_owner())
        assert(instance.exists());
#endif
      // Will be null for virtual managers
      if (memory_manager != NULL)
        memory_manager->validate_instance(this);
      // If we are not the owner, send a reference
      if (!is_owner())
        send_remote_valid_update(owner_space, 1/*count*/, true/*add*/);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (is_owner())
        assert(instance.exists());
#endif
      // Will be null for virtual managers
      if (memory_manager != NULL)
        memory_manager->invalidate_instance(this);
      if (!is_owner())
        send_remote_valid_update(owner_space, 1/*count*/, false/*add*/);
    }

    //--------------------------------------------------------------------------
    /*static*/void PhysicalManager::handle_manager_request(Deserializer &derez,
                                        Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      PhysicalManager *manager = dynamic_cast<PhysicalManager*>(dc);
      assert(manager != NULL);
#else
      PhysicalManager *manager = dynamic_cast<PhysicalManager*>(dc);
#endif
      manager->send_manager(source);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::register_active_context(SingleTask *context)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner()); // should always be on the owner node
#endif
      // Save the context to be notified when we deleted
      GenerationID gen = context->get_generation();
      AutoLock gc(gc_lock);
      // Perfectly alright to overwrite earlier generations
      active_contexts[context] = gen;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::unregister_active_context(SingleTask *context)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(active_contexts.find(context) != active_contexts.end());
#endif
      active_contexts.erase(context);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::meets_region_tree(
                                const std::vector<LogicalRegion> &regions) const
    //--------------------------------------------------------------------------
    {
      for (std::vector<LogicalRegion>::const_iterator it = 
            regions.begin(); it != regions.end(); it++)
      {
        // Check to see if the region tree IDs are the same
        if (it->get_tree_id() != region_node->handle.get_tree_id())
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::meets_regions(
      const std::vector<LogicalRegion> &regions, bool tight_region_bounds) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(region_node != NULL); // only happens with VirtualManager
#endif
      RegionTreeID tree_id = region_node->handle.get_tree_id();
      // Special case for tight bounds 
      if (tight_region_bounds && (regions.size() > 1))
      {
        // If ever are at the local depth and not the same, we fail
        const unsigned local_depth = region_node->get_depth();
        // Tight region bounds for multiple regions is defined
        // as being exactly the common ancestor of the set of regions
        RegionNode *common_ancestor = NULL;
        for (std::vector<LogicalRegion>::const_iterator it = 
              regions.begin(); it != regions.end(); it++)
        {
          // If they are not the same tree ID that is really bad
          if (tree_id != it->get_tree_id())
            return false;
          RegionNode *handle_node = context->get_node(*it);
          if (common_ancestor == NULL)
          {
            common_ancestor = handle_node;
            continue;
          }
          if (common_ancestor == handle_node)
            continue;
          // Get both nodes at the same depth
          unsigned ancestor_depth = common_ancestor->get_depth();
          unsigned handle_depth = handle_node->get_depth();
          while (ancestor_depth > handle_depth)
          {
            common_ancestor = common_ancestor->parent->parent;
            ancestor_depth -= 2;
            if ((ancestor_depth <= local_depth) && 
                (common_ancestor != region_node))
              return false;
          }
          while (handle_depth > ancestor_depth)
          {
            handle_node = handle_node->parent->parent;
            handle_depth -= 2;
          }
          // Walk up until they are the same 
          while (common_ancestor != handle_node)
          {
            common_ancestor = common_ancestor->parent->parent;
            handle_node = handle_node->parent->parent;
            ancestor_depth -= 2;
            if ((ancestor_depth <= local_depth) &&
                (common_ancestor != region_node))
              return false;
          }
        }
#ifdef DEBUG_LEGION
        assert(common_ancestor != NULL);
#endif
        return (common_ancestor == region_node);
      }
      for (std::vector<LogicalRegion>::const_iterator it = 
            regions.begin(); it != regions.end(); it++)
      {
        // If they are not the same tree ID that is really bad
        if (tree_id != it->get_tree_id())
          return false;
        RegionNode *handle_node = context->get_node(*it);
        // If we want tight bounds and there is only one region, if
        // this instance is not of the right size then we fail
        // Note we already handled tight_region_bounds for multiple
        // regions above so this check is only if there is a single region
        if (tight_region_bounds && (handle_node != region_node))
          return false;
        // For now this instance must be a sub-region of the 
        // ancestor logical region.
        if (handle_node != region_node)
        {
          RegionNode *up_node = handle_node;
          while ((up_node != region_node) && (up_node->parent != NULL))
            up_node = up_node->parent->parent;
          if (up_node != region_node)
            return false;
        }
        else // we can just continue in this case since we know we are good
          continue;
        // Now check to see if our instance domain dominates the region
        IndexSpaceNode *index_node = handle_node->row_source; 
        std::vector<Domain> to_check;
        index_node->get_domains_blocking(to_check);
        switch (instance_domain.get_dim())
        {
          case 0:
            {
              // For now just check to make sure we have space
              const size_t space_size = instance_domain.get_index_space().
                                          get_valid_mask().get_num_elmts();
              for (unsigned idx = 0; idx < to_check.size(); idx++)
              {
                const size_t other_size = to_check[idx].get_index_space().
                                          get_valid_mask().get_num_elmts();
                if (space_size < other_size)
                  return false;
              }
              break;
            }
          case 1:
            {
              LegionRuntime::Arrays::Rect<1> our_rect = 
                instance_domain.get_rect<1>();
              for (unsigned idx = 0; idx < to_check.size(); idx++)
              {
                LegionRuntime::Arrays::Rect<1> other_rect = 
                  to_check[idx].get_rect<1>();
                if (!our_rect.dominates(other_rect))
                  return false;
              }
              break;
            }
          case 2:
            {
              LegionRuntime::Arrays::Rect<2> our_rect = 
                instance_domain.get_rect<2>();
              for (unsigned idx = 0; idx < to_check.size(); idx++)
              {
                LegionRuntime::Arrays::Rect<2> other_rect = 
                  to_check[idx].get_rect<2>();
                if (!our_rect.dominates(other_rect))
                  return false;
              }
              break;
            }
          case 3:
            {
              LegionRuntime::Arrays::Rect<3> our_rect = 
                instance_domain.get_rect<3>();
              for (unsigned idx = 0; idx < to_check.size(); idx++)
              {
                LegionRuntime::Arrays::Rect<3> other_rect = 
                  to_check[idx].get_rect<3>();
                if (!our_rect.dominates(other_rect))
                  return false;
              }
              break;
            }
          default:
            assert(false); // unhandled number of dimensions
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::entails(LayoutConstraints *constraints) const
    //--------------------------------------------------------------------------
    {
      // Always test the pointer constraint locally
      if (!pointer_constraint.entails(constraints->pointer_constraint))
        return false;
      return layout->constraints->entails_without_pointer(constraints);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::entails(const LayoutConstraintSet &constraints) const
    //--------------------------------------------------------------------------
    {
      // Always test the pointer constraint locally
      if (!pointer_constraint.entails(constraints.pointer_constraint))
        return false;
      return layout->constraints->entails_without_pointer(constraints);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::conflicts(LayoutConstraints *constraints) const
    //--------------------------------------------------------------------------
    {
      // Always test the pointer constraint locally
      if (pointer_constraint.conflicts(constraints->pointer_constraint))
        return true;
      // We know our layouts don't have a pointer constraint so nothing special
      return layout->constraints->conflicts(constraints);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::conflicts(const LayoutConstraintSet &constraints)const
    //--------------------------------------------------------------------------
    {
      // Always test the pointer constraint locally
      if (pointer_constraint.conflicts(constraints.pointer_constraint))
        return true;
      // We know our layouts don't have a pointer constraint so nothing special
      return layout->constraints->conflicts(constraints);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::perform_deletion(Event deferred_event) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      log_garbage.info("Deleting physical instance " IDFMT " in memory " 
                       IDFMT "", instance.id, memory_manager->memory.id);
#ifndef DISABLE_GC
      std::vector<PhysicalInstance::DestroyedField> serdez_fields;
      layout->compute_destroyed_fields(serdez_fields); 
      if (!serdez_fields.empty())
        instance.destroy(serdez_fields, deferred_event);
      else
        instance.destroy(deferred_event);
#endif
      // Notify any contexts of our deletion
      // No need to hold the lock, there should be no more contexts
      // being added since we are being deleted
      for (std::map<SingleTask*,GenerationID>::const_iterator it = 
            active_contexts.begin(); it != active_contexts.end(); it++)
      {
        it->first->notify_instance_deletion(
            const_cast<PhysicalManager*>(this), it->second);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::set_garbage_collection_priority(MapperID mapper_id,
                                            Processor proc, GCPriority priority)
    //--------------------------------------------------------------------------
    {
      memory_manager->set_garbage_collection_priority(this, mapper_id,
                                                      proc, priority);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::delete_physical_manager(
                                                       PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      if (manager->is_reduction_manager())
      {
        ReductionManager *reduc_manager = manager->as_reduction_manager();
        if (reduc_manager->is_list_manager())
          legion_delete(reduc_manager->as_list_manager());
        else
          legion_delete(reduc_manager->as_fold_manager());
      }
      else
        legion_delete(manager->as_instance_manager());
    }

    /////////////////////////////////////////////////////////////
    // InstanceManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceManager::InstanceManager(RegionTreeForest *ctx, DistributedID did,
                                     AddressSpaceID owner_space, 
                                     AddressSpaceID local_space,
                                     MemoryManager *mem, PhysicalInstance inst,
                                     const Domain &instance_domain, bool own,
                                     RegionNode *node, LayoutDescription *desc, 
                                     const PointerConstraint &constraint,
                                     bool register_now, Event u_event, 
                                     InstanceFlag flags/*=NO_INSTANCE_FLAGS*/)
      : PhysicalManager(ctx, mem, desc, constraint, encode_instance_did(did), 
                        owner_space, local_space, node, inst, instance_domain, 
                        own, register_now),
        use_event(u_event), instance_flags(flags)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
      {
        // Register it with the memory manager, the memory manager
        // on the owner node will handle this
        memory_manager->register_remote_instance(this);
      }
      // Add a reference to the layout
      layout->add_reference();
#ifdef LEGION_GC
      log_garbage.info("GC Instance Manager %ld %d " IDFMT " " IDFMT " ",
       LEGION_DISTRIBUTED_ID_FILTER(did), local_space, inst.id, mem->memory.id);
#endif
      if (is_owner() && Runtime::legion_spy_enabled)
      {
        LegionSpy::log_physical_instance(inst.id, mem->memory.id, 0);
        LegionSpy::log_physical_instance_region(inst.id, region_node->handle);
        layout->log_instance_layout(inst);
      }
    }

    //--------------------------------------------------------------------------
    InstanceManager::InstanceManager(const InstanceManager &rhs)
      : PhysicalManager(NULL, NULL, NULL, rhs.pointer_constraint, 0, 0, 0, NULL,
                    PhysicalInstance::NO_INST, Domain::NO_DOMAIN, false, false),
        use_event(Event::NO_EVENT)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    InstanceManager::~InstanceManager(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info("GC Deletion %ld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
      if (layout->remove_reference())
        delete layout;
    }

    //--------------------------------------------------------------------------
    InstanceManager& InstanceManager::operator=(const InstanceManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        InstanceManager::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instance.exists());
#endif
      return instance.get_accessor();
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        InstanceManager::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instance.exists());
      assert(layout != NULL);
#endif
      const Domain::CopySrcDstField &info = layout->find_field_info(fid);
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic> temp = 
                                                    instance.get_accessor();
      return temp.get_untyped_field_accessor(info.offset, info.size);
    }

    //--------------------------------------------------------------------------
    size_t InstanceManager::get_instance_size(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(layout != NULL);
#endif
      size_t field_sizes = layout->get_total_field_size();
      size_t volume = 
        region_node->row_source->get_domain_blocking().get_volume();
      return (field_sizes * volume);
    }

    //--------------------------------------------------------------------------
    InstanceView* InstanceManager::create_instance_top_view(SingleTask *own_ctx,
                                                   AddressSpaceID logical_owner)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      DistributedID view_did = 
        context->runtime->get_available_distributed_id(false);
      InstanceView* result = 
              legion_new<MaterializedView>(context, view_did, 
                                           owner_space, owner_space, 
                                           logical_owner, region_node,
                                           const_cast<InstanceManager*>(this),
                                           (MaterializedView*)NULL/*parent*/, 
                                           own_ctx, true/*register now*/);
      register_active_context(own_ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceManager::compute_copy_offsets(const FieldMask &copy_mask,
                                  std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(layout != NULL);
#endif
      // Pass in our physical instance so the layout knows how to specialize
      layout->compute_copy_offsets(copy_mask, instance, fields);
    }

    //--------------------------------------------------------------------------
    void InstanceManager::compute_copy_offsets(FieldID fid,
                                  std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(layout != NULL);
#endif
      // Pass in our physical instance so the layout knows how to specialize
      layout->compute_copy_offsets(fid, instance, fields);
    }

    //--------------------------------------------------------------------------
    void InstanceManager::compute_copy_offsets(
                                  const std::vector<FieldID> &copy_fields,
                                  std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(layout != NULL);
#endif
      // Pass in our physical instance so the layout knows how to specialize
      layout->compute_copy_offsets(copy_fields, instance, fields);
    }

    //--------------------------------------------------------------------------
    void InstanceManager::initialize_across_helper(CopyAcrossHelper *helper,
                                                   const FieldMask &dst_mask,
                                     const std::vector<unsigned> &src_indexes,
                                     const std::vector<unsigned> &dst_indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(src_indexes.size() == dst_indexes.size());
#endif
      std::vector<Domain::CopySrcDstField> dst_fields;
      layout->compute_copy_offsets(dst_mask, instance, dst_fields);
#ifdef DEBUG_LEGION
      assert(dst_fields.size() == dst_indexes.size());
#endif
      helper->offsets.resize(dst_fields.size());
      // We've got the offsets compressed based on their destination mask
      // order, now we need to translate them to their source mask order
      // Figure out the permutation from destination mask ordering to 
      // source mask ordering. 
      // First let's figure out the order of the source indexes
      std::vector<unsigned> src_order(src_indexes.size());
      std::map<unsigned,unsigned> translate_map;
      for (unsigned idx = 0; idx < src_indexes.size(); idx++)
        translate_map[src_indexes[idx]] = idx;
      unsigned index = 0;
      for (std::map<unsigned,unsigned>::const_iterator it = 
            translate_map.begin(); it != translate_map.end(); it++, index++)
        src_order[it->second] = index; 
      // Now we can translate the destination indexes
      translate_map.clear();
      for (unsigned idx = 0; idx < dst_indexes.size(); idx++)
        translate_map[dst_indexes[idx]] = idx;
      index = 0; 
      for (std::map<unsigned,unsigned>::const_iterator it = 
            translate_map.begin(); it != translate_map.end(); it++, index++)
      {
        unsigned src_index = src_order[it->second];
        helper->offsets[src_index] = dst_fields[index];
      }
    }

    //--------------------------------------------------------------------------
    void InstanceManager::set_descriptor(FieldDataDescriptor &desc,
                                         unsigned fid_idx) const
    //--------------------------------------------------------------------------
    {
      // Fill in the information about our instance
      desc.inst = instance;
      // Ask the layout to fill in the information about field offset and size
      layout->set_descriptor(desc, fid_idx);
    }

    //--------------------------------------------------------------------------
    void InstanceManager::send_manager(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(owner_space);
        rez.serialize(memory_manager->memory);
        rez.serialize(instance);
        rez.serialize(instance_domain);
        rez.serialize(region_node->handle);
        rez.serialize(use_event);
        rez.serialize(instance_flags);
        layout->pack_layout_description(rez, target);
        pointer_constraint.serialize(rez);
      }
      context->runtime->send_instance_manager(target, rez);
      update_remote_instances(target);
      // Finally we can update our known nodes
      // It's only safe to do this after the message
      // has been sent
      layout->update_known_nodes(target);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceManager::handle_send_manager(Runtime *runtime, 
                                     AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID owner_space;
      derez.deserialize(owner_space);
      Memory mem;
      derez.deserialize(mem);
      PhysicalInstance inst;
      derez.deserialize(inst);
      Domain inst_domain;
      derez.deserialize(inst_domain);
      LogicalRegion handle;
      derez.deserialize(handle);
      Event use_event;
      derez.deserialize(use_event);
      InstanceFlag flags;
      derez.deserialize(flags);
      RegionNode *target_node = runtime->forest->get_node(handle);
      LayoutDescription *layout = 
        LayoutDescription::handle_unpack_layout_description(derez, source, 
                                                            target_node);
      PointerConstraint pointer_constraint;
      pointer_constraint.deserialize(derez);
      MemoryManager *memory = runtime->find_memory_manager(mem);
      void *location;
      InstanceManager *man = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        man = legion_new_in_place<InstanceManager>(location,runtime->forest,did,
                                             owner_space,runtime->address_space,
                                             memory, inst, inst_domain, 
                                             false/*owns*/, target_node, layout,
                                             pointer_constraint, 
                                             false/*reg now*/, use_event,flags);
      else
        man = legion_new<InstanceManager>(runtime->forest, did, owner_space,
                                    runtime->address_space, memory, inst,
                                    inst_domain, false/*owns*/,
                                    target_node, layout, pointer_constraint, 
                                    false/*reg now*/, use_event, flags);
      // Hold-off doing the registration until construction is complete
      man->register_with_runtime();
    }

    //--------------------------------------------------------------------------
    bool InstanceManager::is_attached_file(void) const
    //--------------------------------------------------------------------------
    {
      return (instance_flags & ATTACH_FILE_FLAG);
    }

    /////////////////////////////////////////////////////////////
    // ReductionManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReductionManager::ReductionManager(RegionTreeForest *ctx, DistributedID did,
                                       FieldID f, AddressSpaceID owner_space, 
                                       AddressSpaceID local_space,
                                       MemoryManager *mem,PhysicalInstance inst,
                                       LayoutDescription *desc, 
                                       const PointerConstraint &constraint,
                                       const Domain &inst_domain, bool own_dom,
                                       RegionNode *node, ReductionOpID red, 
                                       const ReductionOp *o, bool register_now)
      : PhysicalManager(ctx, mem, desc, constraint, did, owner_space, 
                  local_space, node, inst, inst_domain, own_dom, register_now),
        op(o), redop(red), logical_field(f)
    //--------------------------------------------------------------------------
    {  
      if (is_owner() && Runtime::legion_spy_enabled)
      {
        LegionSpy::log_physical_instance(inst.id, mem->memory.id, redop);
        LegionSpy::log_physical_instance_region(inst.id, region_node->handle);
        layout->log_instance_layout(inst);
      }
    }

    //--------------------------------------------------------------------------
    ReductionManager::~ReductionManager(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool ReductionManager::has_field(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      return (logical_field == fid); 
    }

    //--------------------------------------------------------------------------
    void ReductionManager::has_fields(std::map<FieldID,bool> &fields) const
    //--------------------------------------------------------------------------
    {
      for (std::map<FieldID,bool>::iterator it = fields.begin();
            it != fields.end(); it++)
      {
        if (it->first == logical_field)
          it->second = true;
        else
          it->second = false;
      }
    }

    //--------------------------------------------------------------------------
    void ReductionManager::remove_space_fields(std::set<FieldID> &fields) const
    //--------------------------------------------------------------------------
    {
      fields.erase(logical_field);
    }

    //--------------------------------------------------------------------------
    void ReductionManager::send_manager(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(owner_space);
        rez.serialize(memory_manager->memory);
        rez.serialize(instance);
        rez.serialize(instance_domain);
        rez.serialize(redop);
        rez.serialize(logical_field);
        rez.serialize(region_node->handle);
        rez.serialize<bool>(is_foldable());
        rez.serialize(get_pointer_space());
        rez.serialize(get_use_event());
        layout->pack_layout_description(rez, target);
        pointer_constraint.serialize(rez);
      }
      // Now send the message
      context->runtime->send_reduction_manager(target, rez);
      update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReductionManager::handle_send_manager(Runtime *runtime, 
                                     AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID owner_space;
      derez.deserialize(owner_space);
      Memory mem;
      derez.deserialize(mem);
      PhysicalInstance inst;
      derez.deserialize(inst);
      Domain inst_dom;
      derez.deserialize(inst_dom);
      ReductionOpID redop;
      derez.deserialize(redop);
      FieldID logical_field;
      derez.deserialize(logical_field);
      LogicalRegion handle;
      derez.deserialize(handle);
      bool foldable;
      derez.deserialize(foldable);
      Domain ptr_space;
      derez.deserialize(ptr_space);
      Event use_event;
      derez.deserialize(use_event);
      RegionNode *target_node = runtime->forest->get_node(handle);
      LayoutDescription *layout = 
        LayoutDescription::handle_unpack_layout_description(derez, source, 
                                                            target_node);
      PointerConstraint pointer_constraint;
      pointer_constraint.deserialize(derez);
      MemoryManager *memory = runtime->find_memory_manager(mem);
      const ReductionOp *op = Runtime::get_reduction_op(redop);
      ReductionManager *man = NULL;
      if (foldable)
      {
        void *location;
        if (runtime->find_pending_collectable_location(did, location))
          man = legion_new_in_place<FoldReductionManager>(location, 
                                                    runtime->forest,
                                                    did, logical_field,
                                                    owner_space,
                                                    runtime->address_space,
                                                    memory, inst, layout,
                                                    pointer_constraint, 
                                                    inst_dom, false/*owner*/,
                                                    target_node, redop, op,
                                                    use_event,false/*reg now*/);
        else
          man = legion_new<FoldReductionManager>(runtime->forest, 
                                           did, logical_field, owner_space, 
                                           runtime->address_space, memory, inst,
                                           layout, pointer_constraint, inst_dom,
                                           false/*own*/, target_node, redop, op,
                                           use_event, false/*reg now*/);
      }
      else
      {
        void *location;
        if (runtime->find_pending_collectable_location(did, location))
          man = legion_new_in_place<ListReductionManager>(location, 
                                                    runtime->forest,
                                                    did, logical_field,
                                                    owner_space, 
                                                    runtime->address_space,
                                                    memory, inst, layout,
                                                    pointer_constraint, 
                                                    inst_dom, false/*owner*/,
                                                    target_node, redop, op,
                                                    ptr_space,false/*reg now*/);
        else
          man = legion_new<ListReductionManager>(runtime->forest, did, 
                                           logical_field, owner_space, 
                                           runtime->address_space, memory, inst,
                                           layout, pointer_constraint, inst_dom, 
                                           false/*own*/, target_node, redop,op,
                                           ptr_space, false/*reg now*/);
      }
      man->register_with_runtime();
    }

    //--------------------------------------------------------------------------
    InstanceView* ReductionManager::create_instance_top_view(
                              SingleTask *own_ctx, AddressSpaceID logical_owner)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      DistributedID view_did = 
        context->runtime->get_available_distributed_id(false);
      InstanceView *result = 
             legion_new<ReductionView>(context, view_did, 
                                       owner_space, owner_space, 
                                       logical_owner, region_node, 
                                       const_cast<ReductionManager*>(this),
                                       own_ctx, true/*register now*/);
      register_active_context(own_ctx);
      return result;
    }

    /////////////////////////////////////////////////////////////
    // ListReductionManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ListReductionManager::ListReductionManager(RegionTreeForest *ctx, 
                                               DistributedID did,
                                               FieldID f,
                                               AddressSpaceID owner_space, 
                                               AddressSpaceID local_space,
                                               MemoryManager *mem,
                                               PhysicalInstance inst, 
                                               LayoutDescription *desc,
                                               const PointerConstraint &cons,
                                               const Domain &d, bool own_dom,
                                               RegionNode *node,
                                               ReductionOpID red,
                                               const ReductionOp *o, 
                                               Domain dom, bool register_now)
      : ReductionManager(ctx, encode_reduction_list_did(did), f, owner_space, 
                         local_space, mem, inst, desc, cons, d, own_dom, node, 
                         red, o, register_now), ptr_space(dom)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
      {
        // Register it with the memory manager, the memory manager
        // on the owner node will handle this
        memory_manager->register_remote_instance(this);
      }
#ifdef LEGION_GC
      log_garbage.info("GC List Reduction Manager %ld %d " IDFMT " " IDFMT " ",
       LEGION_DISTRIBUTED_ID_FILTER(did), local_space, inst.id, mem->memory.id);
#endif
    }

    //--------------------------------------------------------------------------
    ListReductionManager::ListReductionManager(const ListReductionManager &rhs)
      : ReductionManager(NULL, 0, 0, 0, 0, NULL,
                         PhysicalInstance::NO_INST, NULL,rhs.pointer_constraint,
                         Domain::NO_DOMAIN, false, NULL, 0, NULL, false),
        ptr_space(Domain::NO_DOMAIN)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ListReductionManager::~ListReductionManager(void)
    //--------------------------------------------------------------------------
    {
      // Free up our pointer space
      ptr_space.get_index_space().destroy();
#ifdef LEGION_GC
      log_garbage.info("GC Deletion %ld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    ListReductionManager& ListReductionManager::operator=(
                                                const ListReductionManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        ListReductionManager::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
      // TODO: Implement this 
      assert(false);
      return instance.get_accessor();
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        ListReductionManager::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return instance.get_accessor();
    }

    //--------------------------------------------------------------------------
    size_t ListReductionManager::get_instance_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = op->sizeof_rhs;
      if (ptr_space.get_dim() == 0)
      {
        const Realm::ElementMask &mask = 
          ptr_space.get_index_space().get_valid_mask();
        result *= mask.get_num_elmts();
      }
      else
        result *= ptr_space.get_volume();
      return result;
    }
    
    //--------------------------------------------------------------------------
    bool ListReductionManager::is_foldable(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    void ListReductionManager::find_field_offsets(const FieldMask &reduce_mask,
                                  std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instance.exists());
#endif
      // Assume that it's all the fields for right now
      // but offset by the pointer size
      fields.push_back(
          Domain::CopySrcDstField(instance, sizeof(ptr_t), 
                                  op->sizeof_rhs, logical_field));
    }

    //--------------------------------------------------------------------------
    Event ListReductionManager::issue_reduction(Operation *op,
        const std::vector<Domain::CopySrcDstField> &src_fields,
        const std::vector<Domain::CopySrcDstField> &dst_fields,
        RegionTreeNode *dst, Event precondition, bool reduction_fold, 
        bool precise, RegionTreeNode *intersect)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instance.exists());
#endif
      // TODO: use the "new" Realm interface for list instances
      assert(false);
      return Event::NO_EVENT;
    }

    //--------------------------------------------------------------------------
    Domain ListReductionManager::get_pointer_space(void) const
    //--------------------------------------------------------------------------
    {
      return ptr_space;
    }

    //--------------------------------------------------------------------------
    Event ListReductionManager::get_use_event(void) const
    //--------------------------------------------------------------------------
    {
      return Event::NO_EVENT;
    }

    /////////////////////////////////////////////////////////////
    // FoldReductionManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FoldReductionManager::FoldReductionManager(RegionTreeForest *ctx, 
                                               DistributedID did,
                                               FieldID f,
                                               AddressSpaceID owner_space, 
                                               AddressSpaceID local_space,
                                               MemoryManager *mem,
                                               PhysicalInstance inst, 
                                               LayoutDescription *desc,
                                               const PointerConstraint &cons,
                                               const Domain &d, bool own_dom,
                                               RegionNode *node,
                                               ReductionOpID red,
                                               const ReductionOp *o,
                                               Event u_event,
                                               bool register_now)
      : ReductionManager(ctx, encode_reduction_fold_did(did), f, owner_space, 
                         local_space, mem, inst, desc, cons, d, own_dom, node, 
                         red, o, register_now), use_event(u_event)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
      {
        // Register it with the memory manager, the memory manager
        // on the owner node will handle this
        memory_manager->register_remote_instance(this);
      }
#ifdef LEGION_GC
      log_garbage.info("GC Fold Reduction Manager %ld %d " IDFMT " " IDFMT " ",
       LEGION_DISTRIBUTED_ID_FILTER(did), local_space, inst.id, mem->memory.id);
#endif
    }

    //--------------------------------------------------------------------------
    FoldReductionManager::FoldReductionManager(const FoldReductionManager &rhs)
      : ReductionManager(NULL, 0, 0, 0, 0, NULL,
                         PhysicalInstance::NO_INST, NULL,rhs.pointer_constraint,
                         Domain::NO_DOMAIN, false, NULL, 0, NULL, false),
        use_event(Event::NO_EVENT)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FoldReductionManager::~FoldReductionManager(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info("GC Deletion %ld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    FoldReductionManager& FoldReductionManager::operator=(
                                                const FoldReductionManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        FoldReductionManager::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
      return instance.get_accessor();
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        FoldReductionManager::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return instance.get_accessor();
    }

    //--------------------------------------------------------------------------
    size_t FoldReductionManager::get_instance_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = op->sizeof_rhs;
      const Domain &d = region_node->row_source->get_domain_blocking();
      if (d.get_dim() == 0)
      {
        const Realm::ElementMask &mask = 
          d.get_index_space().get_valid_mask();
        result *= mask.get_num_elmts();
      }
      else
        result *= d.get_volume();
      return result;
    }
    
    //--------------------------------------------------------------------------
    bool FoldReductionManager::is_foldable(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    void FoldReductionManager::find_field_offsets(const FieldMask &reduce_mask,
                                  std::vector<Domain::CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instance.exists());
#endif
      // Assume that its all the fields for now
      // until we find a different way to do reductions on a subset of fields
      fields.push_back(
          Domain::CopySrcDstField(instance, 0/*offset*/, 
                                  op->sizeof_rhs, logical_field));
    }

    //--------------------------------------------------------------------------
    Event FoldReductionManager::issue_reduction(Operation *op,
        const std::vector<Domain::CopySrcDstField> &src_fields,
        const std::vector<Domain::CopySrcDstField> &dst_fields,
        RegionTreeNode *dst, Event precondition, bool reduction_fold, 
        bool precise, RegionTreeNode *intersect)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instance.exists());
#endif
      // Doesn't matter if this one is precise or not
      return dst->issue_copy(op, src_fields, dst_fields, precondition, 
                             intersect, redop, reduction_fold);
    }

    //--------------------------------------------------------------------------
    Domain FoldReductionManager::get_pointer_space(void) const
    //--------------------------------------------------------------------------
    {
      return Domain::NO_DOMAIN;
    }

    //--------------------------------------------------------------------------
    Event FoldReductionManager::get_use_event(void) const
    //--------------------------------------------------------------------------
    {
      return use_event;
    }

    /////////////////////////////////////////////////////////////
    // Virtual Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VirtualManager::VirtualManager(RegionTreeForest *ctx, 
                                   LayoutDescription *desc,
                                   const PointerConstraint &constraint,
                                   DistributedID did,AddressSpaceID local_space)
      : PhysicalManager(ctx, NULL/*memory*/, desc, constraint, did, local_space,
                        local_space, NULL/*region*/, PhysicalInstance::NO_INST,
                        Domain::NO_DOMAIN, false/*own domain*/, true/*reg now*/)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VirtualManager::VirtualManager(const VirtualManager &rhs)
      : PhysicalManager(NULL, NULL, NULL, rhs.pointer_constraint, 0, 0, 0,
               NULL, PhysicalInstance::NO_INST, Domain::NO_DOMAIN, false, false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    VirtualManager::~VirtualManager(void)
    //--------------------------------------------------------------------------
    {
      // this should never be deleted since it is a singleton
      assert(false);
    }

    //--------------------------------------------------------------------------
    VirtualManager& VirtualManager::operator=(const VirtualManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          VirtualManager::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return PhysicalInstance::NO_INST.get_accessor();
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          VirtualManager::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return PhysicalInstance::NO_INST.get_accessor();
    }

    //--------------------------------------------------------------------------
    size_t VirtualManager::get_instance_size(void) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    void VirtualManager::send_manager(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool VirtualManager::has_field(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      // has all fields
      return true;
    }

    //--------------------------------------------------------------------------
    void VirtualManager::has_fields(std::map<FieldID,bool> &fields) const
    //--------------------------------------------------------------------------
    {
      for (std::map<FieldID,bool>::iterator it = fields.begin();
            it != fields.end(); it++)
        it->second = true;
    }

    //--------------------------------------------------------------------------
    void VirtualManager::remove_space_fields(std::set<FieldID> &fields) const
    //--------------------------------------------------------------------------
    {
      fields.clear();
    }

    //--------------------------------------------------------------------------
    InstanceView* VirtualManager::create_instance_top_view(SingleTask *context,
                                                   AddressSpaceID logical_owner)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    /*static*/ void VirtualManager::initialize_virtual_instance(Runtime *rt,
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      VirtualManager *&singleton = get_singleton();
      // make a layout constraints
      LayoutConstraintSet constraint_set;
      constraint_set.add_constraint(
          SpecializedConstraint(SpecializedConstraint::VIRTUAL_SPECIALIZE));
      LayoutConstraints *constraints = 
        rt->register_layout(FieldSpace::NO_SPACE, constraint_set);
      FieldMask all_ones(LEGION_FIELD_MASK_FIELD_ALL_ONES);
      std::vector<unsigned> mask_index_map;
      std::vector<CustomSerdezID> serdez;
      std::vector<std::pair<FieldID,size_t> > field_sizes;
      LayoutDescription *layout = new LayoutDescription(all_ones, constraints);
      PointerConstraint pointer_constraint(Memory::NO_MEMORY, 0);
      singleton = new VirtualManager(rt->forest, layout, pointer_constraint,
                                     did, rt->address_space);
      // put a permenant resource reference on this so it is never deleted
      singleton->add_base_resource_ref(NEVER_GC_REF);
    }

    /////////////////////////////////////////////////////////////
    // Instance Builder
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    size_t InstanceBuilder::compute_needed_size(RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      if (!valid)
        initialize(forest);
      size_t total_field_bytes = 0;
      for (unsigned idx = 0; idx < field_sizes.size(); idx++)
        total_field_bytes += field_sizes[idx].second;
      return (total_field_bytes * instance_domain.get_volume());
    }

    //--------------------------------------------------------------------------
    PhysicalManager* InstanceBuilder::create_physical_instance(
                                                       RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      if (!valid)
        initialize(forest);
      if (field_sizes.empty())
      {
        log_run.warning("WARNING: Ignoring request to create instance in "
                        "memory " IDFMT " with no fields.", 
                        memory_manager->memory.id);
        return NULL;
      }
      // If there are no fields then we are done
#ifdef NEW_INSTANCE_CREATION
      PhysicalInstance instance = PhysicalInstance::NO_INST;
      Event ready = forest->create_instance(instance_domain, 
                  memory_manager->memory, field_sizes, instance, constraints);
#else
      PhysicalInstance instance = forest->create_instance(instance_domain,
                                       memory_manager->memory, sizes_only, 
                                       block_size, redop_id, creator_id);
      Event ready = Event::NO_EVENT;
#endif
      // If we couldn't make it then we are done
      if (!instance.exists())
        return NULL;
      // Figure out what kind of instance we just made
      PhysicalManager *result = NULL;
      DistributedID did = forest->runtime->get_available_distributed_id(false);
      AddressSpaceID local_space = forest->runtime->address_space;
      FieldSpaceNode *field_node = ancestor->column_source;
      // Important implementation detail here: we pull the pointer constraint
      // out of the set of constraints here and don't include it in the layout
      // constraints so we can abstract over lots of different layouts. We'll
      // store the pointer constraint separately in the physical instance
      PointerConstraint pointer_constraint = constraints.pointer_constraint;
      constraints.pointer_constraint = PointerConstraint();
      // Now let's find the layout constraints to use for this instance
      LayoutDescription *layout = 
        field_node->find_layout_description(instance_mask, constraints);
      // If we couldn't find one then we make one
      if (layout == NULL)
      {
        // First make a new layout constraint
        LayoutConstraints *layout_constraints = 
         forest->runtime->register_layout(field_node->handle,constraints);
        // Then make our description
        layout = field_node->create_layout_description(instance_mask,
                 layout_constraints, mask_index_map, serdez, field_sizes);
      }
      switch (constraints.specialized_constraint.get_kind())
      {
        case SpecializedConstraint::NORMAL_SPECIALIZE:
          {
            
            // Now we can make the manager
            result = legion_new<InstanceManager>(forest, did, local_space,
                                                 local_space, memory_manager,
                                                 instance, instance_domain, 
                                                 own_domain, ancestor, layout, 
                                                 pointer_constraint, 
                                                 true/*register now*/, ready);
            break;
          }
        case SpecializedConstraint::REDUCTION_FOLD_SPECIALIZE:
          {
            if (field_sizes.size() > 1)
            {
              log_run.error("ERROR: Illegal request for a reduction instance "
                            "containing multiple fields. Only a single field "
                            "is currently permitted for reduction instances.");
#ifdef DEBUG_LEGION
              assert(false);
#endif
              exit(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME);
            }
            // Before we can actually use this instance, we have to 
            // initialize it with a fill operation of the proper value
            // Don't record this fill operation because it is just part
            // of the semantics of reduction instances and not something
            // that we want Legion Spy to see
            void *fill_buffer = malloc(reduction_op->sizeof_rhs);
            reduction_op->init(fill_buffer, 1);
            std::vector<Domain::CopySrcDstField> dsts(1,
                Domain::CopySrcDstField(instance, 0/*offset*/, 
                  field_sizes[0].second, field_sizes[0].first, 0));
            Realm::ProfilingRequestSet requests;
            if (forest->runtime->profiler != NULL)
              forest->runtime->profiler->add_fill_request(requests, creator_id);
            Event filled_and_ready = instance_domain.fill(dsts, requests,
                                  fill_buffer, reduction_op->sizeof_rhs, ready);
            // We can free the buffer after we've issued the fill
            free(fill_buffer);
            result = legion_new<FoldReductionManager>(forest, did, 
                                              field_sizes[0].first, local_space,
                                              local_space, memory_manager, 
                                              instance, layout, 
                                              pointer_constraint, 
                                              instance_domain, own_domain,
                                              ancestor, redop_id,
                                              reduction_op, filled_and_ready,
                                              true/*register now*/); 
            break;
          }
        case SpecializedConstraint::REDUCTION_LIST_SPECIALIZE:
          {
            // TODO: implement this
            assert(false);
            break;
          }
        default:
          assert(false); // illegal specialized case
      }
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceBuilder::initialize(RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      compute_ancestor_and_domain(forest); 
#ifdef NEW_INSTANCE_CREATION
      compute_new_parameters();
#else
      compute_old_parameters();
#endif
    }

    //--------------------------------------------------------------------------
    void InstanceBuilder::compute_ancestor_and_domain(RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      // Check to see if they are all empty, in which case we will make
      // an empty instance with its ancestor being the root of the region
      // tree so it can satisfy all empty regions in this region tree safely
      std::vector<RegionNode*> non_empty_regions;
      std::vector<const Domain*> non_empty_domains;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        RegionNode *next = forest->get_node(regions[idx]);
        const Domain &next_domain = next->get_domain_blocking();
        // Check for empty
        size_t volume = next_domain.get_volume();
        if (volume == 0)
        {
          // Do something special if we know we aren't going to have
          // any non-empty regions
          if ((idx == (regions.size()-1)) && non_empty_regions.empty())
          {
            // We're going to make an empty instance which is all fine, but
            // in order to be sound for other parts of the analysis, we need
            // the ancestor to be the root of the region tree so that this
            // instance can be safely used for any empty region in this tree.
            instance_domain = next_domain;
            while (next->parent != NULL)
              next = next->parent->parent;
            ancestor = next;
            return;
          }
          continue;
        }
        non_empty_regions.push_back(next);
        non_empty_domains.push_back(&next_domain);
      }
      // At this point we have at least one non-empty region
#ifdef DEBUG_LEGION
      assert(!non_empty_regions.empty());
#endif
      ancestor = non_empty_regions[0];
      if (non_empty_regions.size() > 1)
      {
        // Compute an union of the all the index spaces for the basis
        // and the common ancestor of all regions
        const Domain *first = non_empty_domains[0];
        switch (first->get_dim())
        {
          case 0:
            {
              Realm::ElementMask result = 
                first->get_index_space().get_valid_mask();
              for (unsigned idx = 1; idx < non_empty_regions.size(); idx++)
              {
                RegionNode *next = non_empty_regions[idx];
                const Domain *next_domain = non_empty_domains[idx];
                result |= next_domain->get_index_space().get_valid_mask();
                // Find the common ancestor
                ancestor = find_common_ancestor(ancestor, next);
              }
              instance_domain = Domain(
                  Realm::IndexSpace::create_index_space(result));
              own_domain = true;
              break;
            }
          case 1:
            {
              LegionRuntime::Arrays::Rect<1> result = first->get_rect<1>();
              for (unsigned idx = 1; idx < non_empty_regions.size(); idx++)
              {
                RegionNode *next = non_empty_regions[idx];
                const Domain *next_domain = non_empty_domains[idx];
                LegionRuntime::Arrays::Rect<1> next_rect = 
                  next_domain->get_rect<1>();
                result = result.convex_hull(next_rect);
                // Find the common ancesstor
                ancestor = find_common_ancestor(ancestor, next); 
              }
              instance_domain = Domain::from_rect<1>(result);
              break;
            }
          case 2:
            {
              LegionRuntime::Arrays::Rect<2> result = first->get_rect<2>();
              for (unsigned idx = 1; idx < non_empty_regions.size(); idx++)
              {
                RegionNode *next = non_empty_regions[idx];
                const Domain *next_domain = non_empty_domains[idx];
                LegionRuntime::Arrays::Rect<2> next_rect = 
                  next_domain->get_rect<2>();
                result = result.convex_hull(next_rect);
                // Find the common ancesstor
                ancestor = find_common_ancestor(ancestor, next); 
              }
              instance_domain = Domain::from_rect<2>(result);
              break;
            }
          case 3:
            {
              LegionRuntime::Arrays::Rect<3> result = first->get_rect<3>();
              for (unsigned idx = 1; idx < non_empty_regions.size(); idx++)
              {
                RegionNode *next = non_empty_regions[idx];
                const Domain *next_domain = non_empty_domains[idx]; 
                LegionRuntime::Arrays::Rect<3> next_rect = 
                  next_domain->get_rect<3>();
                result = result.convex_hull(next_rect);
                // Find the common ancesstor
                ancestor = find_common_ancestor(ancestor, next); 
              }
              instance_domain = Domain::from_rect<3>(result);
              break;
            }
          default:
            assert(false); // unsupported number of dimensions
        }
      }
      else
        instance_domain = *(non_empty_domains[0]);
    }

    //--------------------------------------------------------------------------
    RegionNode* InstanceBuilder::find_common_ancestor(RegionNode *one,
                                                      RegionNode *two) const
    //--------------------------------------------------------------------------
    {
      // Make them the same level
      while (one->row_source->depth > two->row_source->depth)
      {
#ifdef DEBUG_LEGION
        assert(one->parent != NULL);
#endif
        one = one->parent->parent;
      }
      while (one->row_source->depth < two->row_source->depth)
      {
#ifdef DEBUG_LEGION
        assert(two->parent != NULL);
#endif
        two = two->parent->parent;
      }
      // While they are not the same, make them both go up
      while (one != two)
      {
#ifdef DEBUG_LEGION
        assert(one->parent != NULL);
        assert(two->parent != NULL);
#endif
        one = one->parent->parent;
        two = two->parent->parent;
      }
      return one;
    }

    //--------------------------------------------------------------------------
    void InstanceBuilder::compute_new_parameters(void)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *field_node = ancestor->column_source;      
      const std::vector<FieldID> &field_set = 
        constraints.field_constraint.get_field_set(); 
      field_sizes.resize(field_set.size());
      mask_index_map.resize(field_set.size());
      serdez.resize(field_set.size());
      field_node->compute_create_offsets(field_set, field_sizes,
                                         mask_index_map, serdez, instance_mask);
    }

    //--------------------------------------------------------------------------
    void InstanceBuilder::compute_old_parameters(void)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *field_node = ancestor->column_source;      
      const std::vector<FieldID> &field_set = 
        constraints.field_constraint.get_field_set(); 
      field_sizes.resize(field_set.size());
      mask_index_map.resize(field_set.size());
      serdez.resize(field_set.size());
      field_node->compute_create_offsets(field_set, field_sizes,
                                         mask_index_map, serdez, instance_mask);
      sizes_only.resize(field_sizes.size());
      for (unsigned idx = 0; idx < field_sizes.size(); idx++)
        sizes_only[idx] = field_sizes[idx].second;
      // Now figure out what kind of instance we're going to make, look at
      // the constraints and see if we recognize any of them
      switch (constraints.specialized_constraint.get_kind())
      {
        case SpecializedConstraint::NORMAL_SPECIALIZE:
          {
            const std::vector<DimensionKind> &ordering = 
                                      constraints.ordering_constraint.ordering;
            size_t max_block_size = instance_domain.get_volume();
            // I hate unstructured index spaces
            if (instance_domain.get_dim() == 0)
              max_block_size = instance_domain.get_index_space().
                                              get_valid_mask().get_num_elmts();
            // See if we are making an AOS or SOA instance
            if (!ordering.empty())
            {
              // If fields are first, it is AOS if the fields
              // are last it is SOA, otherwise, see if we can find
              // fields in which case we can't support it yet
              if (ordering.front() == DIM_F)
                block_size = 1;
              else if (ordering.back() == DIM_F)
                block_size = max_block_size;
              else
              {
                for (unsigned idx = 0; idx < ordering.size(); idx++)
                {
                  if (ordering[idx] == DIM_F)
                    assert(false); // need to handle this case
                }
                block_size = max_block_size;
              }
            }
            else
              block_size = max_block_size;
            // redop id is already zero
            break;
          }
        case SpecializedConstraint::REDUCTION_FOLD_SPECIALIZE:
          {
            block_size = 1;
            redop_id = constraints.specialized_constraint.get_reduction_op();
            reduction_op = Runtime::get_reduction_op(redop_id);
            break;
          }
        case SpecializedConstraint::REDUCTION_LIST_SPECIALIZE:
          {
            // TODO: implement list reduction instances
            assert(false);
            redop_id = constraints.specialized_constraint.get_reduction_op();
            reduction_op = Runtime::get_reduction_op(redop_id);
            break;
          }
        case SpecializedConstraint::VIRTUAL_SPECIALIZE:
          {
            log_run.error("Illegal request to create a virtual instance");
            assert(false);
          }
        default:
          assert(false); // unknown kind
      }
    }

  }; // namespace Internal 
}; // namespace Legion

