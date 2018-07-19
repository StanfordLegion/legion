/* Copyright 2018 Stanford University, NVIDIA Corporation
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
#include "legion/runtime.h"
#include "legion/legion_ops.h"
#include "legion/legion_tasks.h"
#include "legion/region_tree.h"
#include "legion/legion_spy.h"
#include "legion/legion_context.h"
#include "legion/legion_profiling.h"
#include "legion/legion_instances.h"
#include "legion/legion_views.h"

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
                                       std::vector<CopySrcDstField> &dst_fields)
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
        CopySrcDstField &field = dst_fields[offset+idx];
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
                                         const unsigned dims,
                                         LayoutConstraints *con,
                                   const std::vector<unsigned> &mask_index_map,
                                   const std::vector<FieldID> &field_ids,
                                   const std::vector<size_t> &field_sizes,
                                   const std::vector<CustomSerdezID> &serdez)
      : allocated_fields(mask), constraints(con), owner(own), total_dims(dims)
    //--------------------------------------------------------------------------
    {
      constraints->add_base_gc_ref(LAYOUT_DESC_REF);
      field_infos.resize(field_sizes.size());
      // Switch data structures from layout by field order to order
      // of field locations in the bit mask
#ifdef DEBUG_LEGION
      assert(mask_index_map.size() == 
                size_t(FieldMask::pop_count(allocated_fields)));
#endif
      for (unsigned idx = 0; idx < mask_index_map.size(); idx++)
      {
        // This gives us the index in the field ordered data structures
        unsigned index = mask_index_map[idx];
        FieldID fid = field_ids[index];
        field_indexes[fid] = idx;
        CopySrcDstField &info = field_infos[idx];
        info.size = field_sizes[index];
        info.field_id = fid;
        info.serdez_id = serdez[index];
      }
    }

    //--------------------------------------------------------------------------
    LayoutDescription::LayoutDescription(const FieldMask &mask,
                                         LayoutConstraints *con)
      : allocated_fields(mask), constraints(con), owner(NULL), total_dims(0)
    //--------------------------------------------------------------------------
    {
      constraints->add_base_gc_ref(LAYOUT_DESC_REF);
    }

    //--------------------------------------------------------------------------
    LayoutDescription::LayoutDescription(const LayoutDescription &rhs)
      : allocated_fields(rhs.allocated_fields), constraints(rhs.constraints), 
        owner(rhs.owner), total_dims(rhs.total_dims)
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
      if (constraints->remove_base_gc_ref(LAYOUT_DESC_REF))
        delete (constraints);
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
    void LayoutDescription::log_instance_layout(ApEvent inst_event) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(implicit_runtime->legion_spy_enabled);
#endif
      std::vector<FieldID> fields;  
      owner->get_field_ids(allocated_fields, fields);
      for (std::vector<FieldID>::const_iterator it = fields.begin();
            it != fields.end(); it++)
        LegionSpy::log_physical_instance_field(inst_event, *it);
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::compute_copy_offsets(const FieldMask &copy_mask,
                                                 PhysicalManager *manager,
                                           std::vector<CopySrcDstField> &fields)
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
      const PhysicalInstance instance = manager->instance;
#ifdef LEGION_SPY
      const ApEvent inst_event = manager->get_use_event();
#endif
      for (int idx = 0; idx < pop_count; idx++)
      {
        int index = compressed.find_next_set(next_start);
        CopySrcDstField &field = fields[offset+idx];
        field = field_infos[index];
        // Our field infos are annonymous so specify the instance now
        field.inst = instance;
        // We'll start looking again at the next index after this one
        next_start = index + 1;
#ifdef LEGION_SPY
        field.inst_event = inst_event;
#endif
      }
    } 

    //--------------------------------------------------------------------------
    void LayoutDescription::compute_copy_offsets(FieldID fid, 
                 PhysicalManager *manager, std::vector<CopySrcDstField> &fields)
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
      fields.back().inst = manager->instance;
#ifdef LEGION_SPY
      fields.back().inst_event = manager->get_use_event();
#endif
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::compute_copy_offsets(
                                   const std::vector<FieldID> &copy_fields, 
                                   PhysicalManager *manager,
                                   std::vector<CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
      unsigned offset = fields.size();
      fields.resize(offset + copy_fields.size());
      const PhysicalInstance instance = manager->instance;
#ifdef LEGION_SPY
      const ApEvent inst_event = manager->get_use_event();
#endif
      for (unsigned idx = 0; idx < copy_fields.size(); idx++)
      {
        std::map<FieldID,unsigned>::const_iterator
          finder = field_indexes.find(copy_fields[idx]);
#ifdef DEBUG_LEGION
        assert(finder != field_indexes.end());
#endif
        CopySrcDstField &info = fields[offset+idx];
        info = field_infos[finder->second];
        // Since instances are annonymous in layout descriptions we
        // have to fill them in when we add the field info
        info.inst = instance;
#ifdef LEGION_SPY
        info.inst_event = inst_event;
#endif
      }
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::get_fields(std::set<FieldID> &fields) const
    //--------------------------------------------------------------------------
    {
      for (std::map<FieldID,unsigned>::const_iterator 
	     it = field_indexes.begin(); it != field_indexes.end(); ++it)
	fields.insert(it->first);
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
    const CopySrcDstField& LayoutDescription::find_field_info(FieldID fid) const
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
      for (std::vector<CopySrcDstField>::const_iterator it = 
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
      fields = constraints->field_constraint.get_field_set();
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::compute_destroyed_fields(
             std::vector<PhysicalInstance::DestroyedField> &serdez_fields) const
    //--------------------------------------------------------------------------
    {
      // See if we have any special fields which need serdez deletion
      for (std::vector<CopySrcDstField>::const_iterator it = 
            field_infos.begin(); it != field_infos.end(); it++)
      {
        if (it->serdez_id > 0)
          serdez_fields.push_back(PhysicalInstance::DestroyedField(it->field_id, 
                                                    it->size, it->serdez_id));
      }
    }

    //--------------------------------------------------------------------------
    bool LayoutDescription::match_layout(
      const LayoutConstraintSet &candidate_constraints, unsigned num_dims) const
    //--------------------------------------------------------------------------
    {
      if (num_dims != total_dims)
        return false;
      // Layout descriptions are always complete, so just check for conflicts
      if (constraints->conflicts(candidate_constraints, total_dims))
        return false;
      // If they don't conflict they have to be the same
      return true;
    }

    //--------------------------------------------------------------------------
    bool LayoutDescription::match_layout(const LayoutDescription *layout,
                                         unsigned num_dims) const
    //--------------------------------------------------------------------------
    {
      if (num_dims != total_dims)
        return false;
      if (layout->allocated_fields != allocated_fields)
        return false;
      // Layout descriptions are always complete so just check for conflicts
      if (constraints->conflicts(layout->constraints, total_dims))
        return false;
      // If they don't conflict they have to be the same
      return true;
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::pack_layout_description(Serializer &rez,
                                                    AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      rez.serialize(constraints->layout_id);
    }

    //--------------------------------------------------------------------------
    /*static*/ LayoutDescription* LayoutDescription::
      handle_unpack_layout_description(Deserializer &derez,
                                 AddressSpaceID source, RegionNode *region_node)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *field_space_node = region_node->column_source;
      LayoutConstraintID layout_id;
      derez.deserialize(layout_id);

      LayoutConstraints *constraints = 
        region_node->context->runtime->find_layout_constraints(layout_id);

      FieldMask instance_mask;
      const std::vector<FieldID> &field_set = 
        constraints->field_constraint.get_field_set(); 
      std::vector<size_t> field_sizes(field_set.size());
      std::vector<unsigned> mask_index_map(field_set.size());
      std::vector<CustomSerdezID> serdez(field_set.size());
      field_space_node->compute_field_layout(field_set, field_sizes,
                               mask_index_map, serdez, instance_mask);
      const unsigned total_dims = region_node->row_source->get_num_dims();
      LayoutDescription *result = 
        field_space_node->create_layout_description(instance_mask, total_dims,
                  constraints, mask_index_map, field_set, field_sizes, serdez);
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
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
                                     RegionNode *node,
                                     PhysicalInstance inst, IndexSpaceNode *d, 
                                     bool own, bool register_now)
      : DistributedCollectable(ctx->runtime, did, owner_space, register_now), 
        context(ctx), memory_manager(memory), region_node(node), layout(desc),
        instance(inst), instance_domain(d), 
        own_domain(own), pointer_constraint(constraint)
    //--------------------------------------------------------------------------
    {
      if (region_node != NULL)
      {
        region_node->add_base_gc_ref(PHYSICAL_MANAGER_REF);
        region_node->register_physical_manager(this);
      }
      if (instance_domain != NULL)
        instance_domain->add_base_resource_ref(PHYSICAL_MANAGER_REF);
      // Add a reference to the layout
      if (layout != NULL)
        layout->add_reference();
    }

    //--------------------------------------------------------------------------
    PhysicalManager::~PhysicalManager(void)
    //--------------------------------------------------------------------------
    {
      if (region_node != NULL)
      {
        region_node->unregister_physical_manager(this);
        if (region_node->remove_base_gc_ref(PHYSICAL_MANAGER_REF))
          delete region_node;
      }
      if ((instance_domain != NULL) && 
          instance_domain->remove_base_resource_ref(PHYSICAL_MANAGER_REF))
        delete instance_domain;
      // Remote references removed by DistributedCollectable destructor
      if (!is_owner())
        memory_manager->unregister_remote_instance(this);
      // If we own our domain, then we need to delete it now
      if (own_domain && is_owner())
        region_node->context->destroy_index_space(instance_domain->handle,
                                                  runtime->address_space);
      if ((layout != NULL) && layout->remove_reference())
        delete layout;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::log_instance_creation(UniqueID creator_id,
                Processor proc, const std::vector<LogicalRegion> &regions) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(runtime->legion_spy_enabled);
#endif
      const ApEvent inst_event = get_use_event();
      LegionSpy::log_physical_instance_creator(inst_event, creator_id, proc.id);
      for (unsigned idx = 0; idx < regions.size(); idx++)
        LegionSpy::log_physical_instance_creation_region(inst_event, 
                                                         regions[idx]);
      const LayoutConstraints *constraints = layout->constraints;
      LegionSpy::log_instance_specialized_constraint(inst_event,
          constraints->specialized_constraint.kind, 
          constraints->specialized_constraint.redop);
#ifdef DEBUG_HIGH_LEVEL
      assert(constraints->memory_constraint.has_kind);
#endif
      if (constraints->memory_constraint.is_valid())
        LegionSpy::log_instance_memory_constraint(inst_event,
            constraints->memory_constraint.kind);
      LegionSpy::log_instance_field_constraint(inst_event,
          constraints->field_constraint.contiguous, 
          constraints->field_constraint.inorder,
          constraints->field_constraint.field_set.size());
      for (std::vector<FieldID>::const_iterator it = 
            constraints->field_constraint.field_set.begin(); it !=
            constraints->field_constraint.field_set.end(); it++)
        LegionSpy::log_instance_field_constraint_field(inst_event, *it);
      LegionSpy::log_instance_ordering_constraint(inst_event,
          constraints->ordering_constraint.contiguous,
          constraints->ordering_constraint.ordering.size());
      for (std::vector<DimensionKind>::const_iterator it = 
            constraints->ordering_constraint.ordering.begin(); it !=
            constraints->ordering_constraint.ordering.end(); it++)
        LegionSpy::log_instance_ordering_constraint_dimension(inst_event, *it);
      for (std::vector<SplittingConstraint>::const_iterator it = 
            constraints->splitting_constraints.begin(); it !=
            constraints->splitting_constraints.end(); it++)
        LegionSpy::log_instance_splitting_constraint(inst_event,
                                it->kind, it->value, it->chunks);
      for (std::vector<DimensionConstraint>::const_iterator it = 
            constraints->dimension_constraints.begin(); it !=
            constraints->dimension_constraints.end(); it++)
        LegionSpy::log_instance_dimension_constraint(inst_event,
                                    it->kind, it->eqk, it->value);
      for (std::vector<AlignmentConstraint>::const_iterator it = 
            constraints->alignment_constraints.begin(); it !=
            constraints->alignment_constraints.end(); it++)
        LegionSpy::log_instance_alignment_constraint(inst_event,
                                it->fid, it->eqk, it->alignment);
      for (std::vector<OffsetConstraint>::const_iterator it = 
            constraints->offset_constraints.begin(); it != 
            constraints->offset_constraints.end(); it++)
        LegionSpy::log_instance_offset_constraint(inst_event,
                                          it->fid, it->offset);
    } 

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_active(ReferenceMutator *mutator)
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
        send_remote_gc_update(owner_space, mutator, 1/*count*/, true/*add*/);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_inactive(ReferenceMutator *mutator)
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
        send_remote_gc_update(owner_space, mutator, 1/*count*/, false/*add*/);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_valid(ReferenceMutator *mutator)
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
        send_remote_valid_update(owner_space, mutator, 1/*count*/, true/*add*/);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_invalid(ReferenceMutator *mutator)
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
        send_remote_valid_update(owner_space, mutator, 1/*count*/,false/*add*/);
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
    void PhysicalManager::register_active_context(InnerContext *context)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner()); // should always be on the owner node
#endif
      context->add_reference();
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(active_contexts.find(context) == active_contexts.end());
#endif
      active_contexts.insert(context);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::unregister_active_context(InnerContext *context)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner()); // should always be on the owner node
#endif
      {
        AutoLock gc(gc_lock);
        std::set<InnerContext*>::iterator finder = 
          active_contexts.find(context);
        // We could already have removed this context if this
        // physical instance was deleted
        if (finder == active_contexts.end())
          return;
        active_contexts.erase(finder);
      }
      if (context->remove_reference())
        delete context;
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
        else
        {
          // We have the same region name, all we have to do is 
          // a check for empty which is not actually allowed
          if (instance_domain->get_volume() == 0)
          {
            // Check to see if the region really is empty or not
            if (region_node->row_source->get_volume() == 0)
              continue;
            else
              return false;
          }
          else // Not empty so this is the proper region
            continue;
        }
        // Now check to see if our instance domain dominates the region
        IndexSpaceNode *index_node = handle_node->row_source; 
        if (!instance_domain->dominates(index_node))
          return false;
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
      return layout->constraints->entails_without_pointer(constraints,
              (instance_domain != NULL) ? instance_domain->get_num_dims() : 0);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::entails(const LayoutConstraintSet &constraints) const
    //--------------------------------------------------------------------------
    {
      // Always test the pointer constraint locally
      if (!pointer_constraint.entails(constraints.pointer_constraint))
        return false;
      return layout->constraints->entails_without_pointer(constraints,
              (instance_domain != NULL) ? instance_domain->get_num_dims() : 0);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::conflicts(LayoutConstraints *constraints) const
    //--------------------------------------------------------------------------
    {
      // Always test the pointer constraint locally
      if (pointer_constraint.conflicts(constraints->pointer_constraint))
        return true;
      // We know our layouts don't have a pointer constraint so nothing special
      return layout->constraints->conflicts(constraints,
              (instance_domain != NULL) ? instance_domain->get_num_dims() : 0);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::conflicts(const LayoutConstraintSet &constraints)const
    //--------------------------------------------------------------------------
    {
      // Always test the pointer constraint locally
      if (pointer_constraint.conflicts(constraints.pointer_constraint))
        return true;
      // We know our layouts don't have a pointer constraint so nothing special
      return layout->constraints->conflicts(constraints,
              (instance_domain != NULL) ? instance_domain->get_num_dims() : 0);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::acquire_instance(ReferenceSource source,
                                           ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // Do an atomic operation to check to see if we are already valid
      // and increment our count if we are, in this case the acquire 
      // has succeeded and we are done, this should be the common case
      // since we are likely already holding valid references elsewhere
      // Note that we cannot do this for external instances as they might
      // have been detached while still holding valid references so they
      // have to go through the full path every time
      if (!is_external_instance() && check_valid_and_increment(source))
        return true;
      // If we're not the owner, we're not going to succeed past this
      // since we aren't on the same node as where the instance lives
      // which is where the point of serialization is for garbage collection
      if (!is_owner())
        return false;
      // Tell our manager, we're attempting an acquire, if it tells
      // us false then we are not allowed to proceed
      if (!memory_manager->attempt_acquire(this))
        return false;
      // At this point we're in the clear to add our valid reference
      add_base_valid_ref(source, mutator);
      // Complete the handshake with the memory manager
      memory_manager->complete_acquire(this);
      return true;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::perform_deletion(RtEvent deferred_event)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      log_garbage.spew("Deleting physical instance " IDFMT " in memory " 
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
      // Grab a copy of this in case we get any removal calls
      // while we are doing the deletion. We know that there
      // will be no more additions because we are being deleted
      std::set<InnerContext*> copy_active_contexts;
      {
        AutoLock gc(gc_lock);
        if (active_contexts.empty())
          return;
        copy_active_contexts = active_contexts;
        active_contexts.clear();
      }
      for (std::set<InnerContext*>::const_iterator it = 
           copy_active_contexts.begin(); it != copy_active_contexts.end(); it++)
      {
        (*it)->notify_instance_deletion(const_cast<PhysicalManager*>(this));
        if ((*it)->remove_reference())
          delete (*it);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::force_deletion(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      log_garbage.spew("Force deleting physical instance " IDFMT " in memory "
                       IDFMT "", instance.id, memory_manager->memory.id);
#ifndef DISABLE_GC
      std::vector<PhysicalInstance::DestroyedField> serdez_fields;
      layout->compute_destroyed_fields(serdez_fields); 
      if (!serdez_fields.empty())
        instance.destroy(serdez_fields);
      else
        instance.destroy();
#endif
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
    RtEvent PhysicalManager::detach_external_instance(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_external_instance());
#endif
      return memory_manager->detach_external_instance(this);
    }

    /////////////////////////////////////////////////////////////
    // InstanceManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceManager::InstanceManager(RegionTreeForest *ctx, DistributedID did,
                                     AddressSpaceID owner_space, 
                                     MemoryManager *mem, PhysicalInstance inst,
                                     IndexSpaceNode *instance_domain, bool own,
                                     RegionNode *node, LayoutDescription *desc, 
                                     const PointerConstraint &constraint,
                                     bool register_now, ApEvent u_event,
                                     bool external_instance,
                                     Reservation read_only_reservation) 
      : PhysicalManager(ctx, mem, desc, constraint, 
                        encode_instance_did(did, external_instance), 
                        owner_space, node, inst, instance_domain, 
                        own, register_now), use_event(u_event),
                        read_only_mapping_reservation(read_only_reservation)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
      {
        // Register it with the memory manager, the memory manager
        // on the owner node will handle this
        memory_manager->register_remote_instance(this);
      } 
#ifdef LEGION_GC
      log_garbage.info("GC Instance Manager %lld %d " IDFMT " " IDFMT " ",
       LEGION_DISTRIBUTED_ID_FILTER(did), local_space, inst.id, mem->memory.id);
#endif
      if (runtime->legion_spy_enabled)
      {
#ifdef DEBUG_LEGION
        assert(use_event.exists());
#endif
        LegionSpy::log_physical_instance(use_event, inst.id, mem->memory.id, 0);
        LegionSpy::log_physical_instance_region(use_event, region_node->handle);
        layout->log_instance_layout(use_event);
      }
    }

    //--------------------------------------------------------------------------
    InstanceManager::InstanceManager(const InstanceManager &rhs)
      : PhysicalManager(NULL, NULL, NULL, rhs.pointer_constraint, 0, 0, NULL,
                    PhysicalInstance::NO_INST, NULL, false, false),
        use_event(ApEvent::NO_AP_EVENT)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    InstanceManager::~InstanceManager(void)
    //--------------------------------------------------------------------------
    {
      // If we are the owner, we get to destroy the read only reservation
      if (is_owner())
        read_only_mapping_reservation.destroy_reservation();
      read_only_mapping_reservation = Reservation::NO_RESERVATION;
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
      const CopySrcDstField &info = layout->find_field_info(fid);
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic> temp = 
                                                    instance.get_accessor();
      return temp.get_untyped_field_accessor(info.field_id, info.size);
    }

    //--------------------------------------------------------------------------
    size_t InstanceManager::get_instance_size(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(layout != NULL);
#endif
      size_t field_sizes = layout->get_total_field_size();
      size_t volume = instance_domain->get_volume(); 
      return (field_sizes * volume);
    }

    //--------------------------------------------------------------------------
    InstanceView* InstanceManager::create_instance_top_view(
                            InnerContext *own_ctx, AddressSpaceID logical_owner)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      DistributedID view_did = 
        context->runtime->get_available_distributed_id();
      UniqueID context_uid = own_ctx->get_context_uid();
      InstanceView* result = 
              new MaterializedView(context, view_did, owner_space, 
                                   logical_owner, region_node,
                                   const_cast<InstanceManager*>(this),
                                   (MaterializedView*)NULL/*parent*/, 
                                   context_uid, true/*register now*/);
      register_active_context(own_ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceManager::compute_copy_offsets(const FieldMask &copy_mask,
                                           std::vector<CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(layout != NULL);
#endif
      // Pass in our physical instance so the layout knows how to specialize
      layout->compute_copy_offsets(copy_mask, this, fields);
    }

    //--------------------------------------------------------------------------
    void InstanceManager::compute_copy_offsets(FieldID fid,
                                           std::vector<CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(layout != NULL);
#endif
      // Pass in our physical instance so the layout knows how to specialize
      layout->compute_copy_offsets(fid, this, fields);
    }

    //--------------------------------------------------------------------------
    void InstanceManager::compute_copy_offsets(
                                  const std::vector<FieldID> &copy_fields,
                                  std::vector<CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(layout != NULL);
#endif
      // Pass in our physical instance so the layout knows how to specialize
      layout->compute_copy_offsets(copy_fields, this, fields);
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
      std::vector<CopySrcDstField> dst_fields;
      layout->compute_copy_offsets(dst_mask, this, dst_fields);
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
        rez.serialize(instance_domain->handle);
        rez.serialize(region_node->handle);
        rez.serialize(use_event);
        layout->pack_layout_description(rez, target);
        pointer_constraint.serialize(rez);
        rez.serialize(read_only_mapping_reservation);
      }
      context->runtime->send_instance_manager(target, rez);
      update_remote_instances(target);
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
      IndexSpace inst_handle;
      derez.deserialize(inst_handle);
      LogicalRegion handle;
      derez.deserialize(handle);
      ApEvent use_event;
      derez.deserialize(use_event);
      IndexSpaceNode *inst_domain = runtime->forest->get_node(inst_handle);
      RegionNode *target_node = runtime->forest->get_node(handle);
      LayoutDescription *layout = 
        LayoutDescription::handle_unpack_layout_description(derez, source, 
                                                            target_node);
      PointerConstraint pointer_constraint;
      pointer_constraint.deserialize(derez);
      Reservation read_only_reservation;
      derez.deserialize(read_only_reservation);
      MemoryManager *memory = runtime->find_memory_manager(mem);
      void *location;
      InstanceManager *man = NULL;
      const bool external_instance = PhysicalManager::is_external_did(did);
      if (runtime->find_pending_collectable_location(did, location))
        man = new(location) InstanceManager(runtime->forest,did,
                                            owner_space, memory, inst, 
                                            inst_domain, false/*owns*/, 
                                            target_node, layout,
                                            pointer_constraint,
                                            false/*reg now*/, use_event,
                                            external_instance, 
                                            read_only_reservation);
      else
        man = new InstanceManager(runtime->forest, did, owner_space,
                                  memory, inst, inst_domain, false/*owns*/,
                                  target_node, layout, pointer_constraint, 
                                  false/*reg now*/, use_event,
                                  external_instance,
                                  read_only_reservation);
      // Hold-off doing the registration until construction is complete
      man->register_with_runtime(NULL/*no remote registration needed*/);
    }

    /////////////////////////////////////////////////////////////
    // ReductionManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReductionManager::ReductionManager(RegionTreeForest *ctx, DistributedID did,
                                       AddressSpaceID owner_space, 
                                       MemoryManager *mem,PhysicalInstance inst,
                                       LayoutDescription *desc, 
                                       const PointerConstraint &constraint,
                                       IndexSpaceNode *inst_domain,bool own_dom,
                                       RegionNode *node, ReductionOpID red, 
                                       const ReductionOp *o, ApEvent u_event,
                                       bool register_now)
      : PhysicalManager(ctx, mem, desc, constraint, did, owner_space, 
                        node, inst, inst_domain, own_dom, register_now),
        op(o), redop(red), use_event(u_event)
    //--------------------------------------------------------------------------
    {  
      if (runtime->legion_spy_enabled)
      {
#ifdef DEBUG_LEGION
        assert(use_event.exists());
#endif
        LegionSpy::log_physical_instance(use_event, inst.id, 
                                         mem->memory.id, redop);
        LegionSpy::log_physical_instance_region(use_event, region_node->handle);
        layout->log_instance_layout(use_event);
      }
    }

    //--------------------------------------------------------------------------
    ReductionManager::~ReductionManager(void)
    //--------------------------------------------------------------------------
    {
#if 0
      if (!created_index_spaces.empty())
      {
        for (std::vector<Realm::IndexSpace>::const_iterator it = 
              created_index_spaces.begin(); it != 
              created_index_spaces.end(); it++)
          it->destroy();
      }
#endif
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
        rez.serialize(instance_domain->handle);
        rez.serialize(redop);
        rez.serialize(region_node->handle);
        rez.serialize<bool>(is_foldable());
        rez.serialize(get_pointer_space());
        rez.serialize(use_event);
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
      IndexSpace inst_handle;
      derez.deserialize(inst_handle);
      ReductionOpID redop;
      derez.deserialize(redop);
      LogicalRegion handle;
      derez.deserialize(handle);
      bool foldable;
      derez.deserialize(foldable);
      Domain ptr_space;
      derez.deserialize(ptr_space);
      ApEvent use_event;
      derez.deserialize(use_event);
      IndexSpaceNode *inst_dom = runtime->forest->get_node(inst_handle);
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
          man = new(location) FoldReductionManager(runtime->forest,
                                                   did, owner_space,
                                                   memory, inst, layout,
                                                   pointer_constraint, 
                                                   inst_dom, false/*owner*/,
                                                   target_node, redop, op,
                                                   use_event,
                                                   false/*reg now*/);
        else
          man = new FoldReductionManager(runtime->forest, 
                                         did, owner_space, memory, inst,
                                         layout, pointer_constraint, inst_dom,
                                         false/*own*/, target_node, redop, op,
                                         use_event, false/*reg now*/);
      }
      else
      {
        void *location;
        if (runtime->find_pending_collectable_location(did, location))
          man = new(location) ListReductionManager(runtime->forest,
                                                   did, owner_space, 
                                                   memory, inst, layout,
                                                   pointer_constraint, 
                                                   inst_dom, false/*owner*/,
                                                   target_node, redop, op,
                                                   ptr_space, use_event,
                                                   false/*reg now*/);
        else
          man = new ListReductionManager(runtime->forest, did, 
                                         owner_space, memory, inst,
                                         layout, pointer_constraint, inst_dom,
                                         false/*own*/, target_node, redop,op,
                                         ptr_space, use_event,false/*reg now*/);
      }
      man->register_with_runtime(NULL/*no remote registration needed*/);
    }

    //--------------------------------------------------------------------------
    InstanceView* ReductionManager::create_instance_top_view(
                            InnerContext *own_ctx, AddressSpaceID logical_owner)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      DistributedID view_did = 
        context->runtime->get_available_distributed_id();
      UniqueID context_uid = own_ctx->get_context_uid();
      InstanceView *result = 
             new ReductionView(context, view_did, owner_space, 
                               logical_owner, region_node, 
                               const_cast<ReductionManager*>(this),
                               context_uid, true/*register now*/);
      register_active_context(own_ctx);
      return result;
    }

#if 0
    //--------------------------------------------------------------------------
    Domain ReductionManager::compute_reduction_domain(PhysicalInstance target,
                             const Domain &copy_domain, ApEvent copy_domain_pre)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
      {
        // If we're not the owner we have to send a message
        assert(false);
      }
      // For now, if we need to wait, then do that now
      // Later we can chain these dependences when we compute intersections
      if (copy_domain_pre.exists() && !copy_domain_pre.has_triggered())
      {
        RtEvent wait_on = Runtime::protect_event(copy_domain_pre);
        wait_on.wait();
      }
      Domain result = Domain::NO_DOMAIN;
      AutoLock m_lock(manager_lock);
      // See if we've handled this destination before
      std::map<PhysicalInstance,std::vector<Domain> >::const_iterator finder =
        reduction_domains.find(target);
      if (finder != reduction_domains.end())
      {
        const std::vector<Domain> &prev = finder->second;
        switch (copy_domain.get_dim())
        {
          case 0:
            {
              Realm::ElementMask copy_mask = 
                copy_domain.get_index_space().get_valid_mask();
              for (std::vector<Domain>::const_iterator it = prev.begin();
                    it != prev.end(); it++)
              {
#ifdef DEBUG_LEGION
                assert(it->get_dim() == 0);
#endif
                copy_mask &= it->get_index_space().get_valid_mask();
                if (!copy_mask)
                  break;
              }
              if (!!copy_mask)
              {
                result = 
                  Domain(Realm::IndexSpace::create_index_space(copy_mask));
                created_index_spaces.push_back(result.get_index_space());
              }
              break;
            }
          case 1:
            {
              Rect<1> copy_rect = copy_domain.get_rect<1>();
              for (std::vector<Domain>::const_iterator it = prev.begin();
                    it != prev.end(); it++)
              {
#ifdef DEBUG_LEGION
                assert(it->get_dim() == 1);
#endif
                copy_rect = copy_rect.intersection(it->get_rect<1>());
                if (copy_rect.volume() == 0)
                  break;
              }
              if (copy_rect.volume() > 0)
                result = Domain::from_rect<1>(copy_rect);
              break;
            }
          case 2:
            {
              Rect<2> copy_rect = copy_domain.get_rect<2>();
              for (std::vector<Domain>::const_iterator it = prev.begin();
                    it != prev.end(); it++)
              {
#ifdef DEBUG_LEGION
                assert(it->get_dim() == 2);
#endif
                copy_rect = copy_rect.intersection(it->get_rect<2>());
                if (copy_rect.volume() == 0)
                  break;
              }
              if (copy_rect.volume() > 0)
                result = Domain::from_rect<2>(copy_rect);
              break;
            }
          case 3:
            {
              Rect<3> copy_rect = copy_domain.get_rect<3>();
              for (std::vector<Domain>::const_iterator it = prev.begin();
                    it != prev.end(); it++)
              {
#ifdef DEBUG_LEGION
                assert(it->get_dim() == 3);
#endif
                copy_rect = copy_rect.intersection(it->get_rect<3>());
                if (copy_rect.volume() == 0)
                  break;
              }
              if (copy_rect.volume() > 0)
                result = Domain::from_rect<3>(copy_rect);
              break;
            }
          default:
            assert(false);
        }
      }
      // Add the result domain to the set and return
      if (result.exists())
        reduction_domains[target].push_back(result);
      return result;
    }
#endif

    /////////////////////////////////////////////////////////////
    // ListReductionManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ListReductionManager::ListReductionManager(RegionTreeForest *ctx, 
                                               DistributedID did,
                                               AddressSpaceID owner_space, 
                                               MemoryManager *mem,
                                               PhysicalInstance inst, 
                                               LayoutDescription *desc,
                                               const PointerConstraint &cons,
                                               IndexSpaceNode *d, bool own_dom,
                                               RegionNode *node,
                                               ReductionOpID red,
                                               const ReductionOp *o, 
                                               Domain dom, ApEvent use_event, 
                                               bool register_now)
      : ReductionManager(ctx, encode_reduction_list_did(did), owner_space, 
                         mem, inst, desc, cons, d, own_dom, node, 
                         red, o, use_event, register_now), ptr_space(dom)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dom.is_id == 0); // shouldn't have a sparsity map for dom
#endif
      if (!is_owner())
      {
        // Register it with the memory manager, the memory manager
        // on the owner node will handle this
        memory_manager->register_remote_instance(this);
      }
#ifdef LEGION_GC
      log_garbage.info("GC List Reduction Manager %lld %d " IDFMT " " IDFMT " ",
       LEGION_DISTRIBUTED_ID_FILTER(did), local_space, inst.id, mem->memory.id);
#endif
    }

    //--------------------------------------------------------------------------
    ListReductionManager::ListReductionManager(const ListReductionManager &rhs)
      : ReductionManager(NULL, 0, 0, NULL,
                         PhysicalInstance::NO_INST, NULL,rhs.pointer_constraint,
                         NULL, false, NULL, 0, NULL,ApEvent::NO_AP_EVENT,false),
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
      return ptr_space.get_volume();
    }
    
    //--------------------------------------------------------------------------
    bool ListReductionManager::is_foldable(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    void ListReductionManager::find_field_offsets(const FieldMask &reduce_mask,
                                           std::vector<CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instance.exists());
#endif
      // TODO: implement this for list reduction instances
      assert(false);
    }

    //--------------------------------------------------------------------------
    ApEvent ListReductionManager::issue_reduction(Operation *op, 
        const std::vector<CopySrcDstField> &src_fields,
        const std::vector<CopySrcDstField> &dst_fields,
        RegionTreeNode *dst, ApEvent precondition, PredEvent guard,
        bool reduction_fold, bool precise, PhysicalTraceInfo &trace_info,
        RegionTreeNode *intersect)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instance.exists());
#endif
      // TODO: use the "new" Realm interface for list instances
      assert(false);
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    Domain ListReductionManager::get_pointer_space(void) const
    //--------------------------------------------------------------------------
    {
      return ptr_space;
    }

    /////////////////////////////////////////////////////////////
    // FoldReductionManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FoldReductionManager::FoldReductionManager(RegionTreeForest *ctx, 
                                               DistributedID did,
                                               AddressSpaceID owner_space, 
                                               MemoryManager *mem,
                                               PhysicalInstance inst, 
                                               LayoutDescription *desc,
                                               const PointerConstraint &cons,
                                               IndexSpaceNode *d, bool own_dom,
                                               RegionNode *node,
                                               ReductionOpID red,
                                               const ReductionOp *o,
                                               ApEvent u_event,
                                               bool register_now)
      : ReductionManager(ctx, encode_reduction_fold_did(did), owner_space, 
                         mem, inst, desc, cons, d, own_dom, node, 
                         red, o, u_event, register_now)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
      {
        // Register it with the memory manager, the memory manager
        // on the owner node will handle this
        memory_manager->register_remote_instance(this);
      }
#ifdef LEGION_GC
      log_garbage.info("GC Fold Reduction Manager %lld %d " IDFMT " " IDFMT " ",
       LEGION_DISTRIBUTED_ID_FILTER(did), local_space, inst.id, mem->memory.id);
#endif
    }

    //--------------------------------------------------------------------------
    FoldReductionManager::FoldReductionManager(const FoldReductionManager &rhs)
      : ReductionManager(NULL, 0, 0, NULL,
                         PhysicalInstance::NO_INST, NULL,rhs.pointer_constraint,
                         NULL, false, NULL, 0, NULL, ApEvent::NO_AP_EVENT,false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FoldReductionManager::~FoldReductionManager(void)
    //--------------------------------------------------------------------------
    {
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
#ifdef DEBUG_LEGION
      assert(instance.exists());
#endif
      return instance.get_accessor();
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        FoldReductionManager::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instance.exists());
      assert(layout != NULL);
#endif
      const CopySrcDstField &info = layout->find_field_info(fid);
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic> temp = 
                                                    instance.get_accessor();
      return temp.get_untyped_field_accessor(info.field_id, info.size);
    }

    //--------------------------------------------------------------------------
    size_t FoldReductionManager::get_instance_size(void) const
    //--------------------------------------------------------------------------
    {
      unsigned field_count = FieldMask::pop_count(layout->allocated_fields);
      size_t field_size = op->sizeof_rhs;
      size_t volume = instance_domain->get_volume();
      return (field_count * field_size * volume);
    }
    
    //--------------------------------------------------------------------------
    bool FoldReductionManager::is_foldable(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    void FoldReductionManager::find_field_offsets(const FieldMask &reduce_mask,
                                           std::vector<CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instance.exists());
      assert(layout != NULL);
#endif
      layout->compute_copy_offsets(reduce_mask, this, fields);
    }

    //--------------------------------------------------------------------------
    ApEvent FoldReductionManager::issue_reduction(Operation *op,
        const std::vector<CopySrcDstField> &src_fields,
        const std::vector<CopySrcDstField> &dst_fields,
        RegionTreeNode *dst, ApEvent precondition, PredEvent guard,
        bool reduction_fold, bool precise,
        PhysicalTraceInfo &trace_info, RegionTreeNode *intersect)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instance.exists());
#endif
      // Doesn't matter if this one is precise or not
      return dst->issue_copy(op, src_fields, dst_fields, precondition,
                             guard, trace_info, intersect, redop,
                             reduction_fold);
    }

    //--------------------------------------------------------------------------
    Domain FoldReductionManager::get_pointer_space(void) const
    //--------------------------------------------------------------------------
    {
      return Domain::NO_DOMAIN;
    }

    /////////////////////////////////////////////////////////////
    // Virtual Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VirtualManager::VirtualManager(RegionTreeForest *ctx, 
                                   LayoutDescription *desc,
                                   const PointerConstraint &constraint,
                                   DistributedID did)
      : PhysicalManager(ctx, NULL/*memory*/, desc, constraint, did, 
                        ctx->runtime->address_space,
                        NULL/*region*/, PhysicalInstance::NO_INST,
                        NULL, false/*own domain*/, true/*reg now*/)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VirtualManager::VirtualManager(const VirtualManager &rhs)
      : PhysicalManager(NULL, NULL, NULL, rhs.pointer_constraint, 0, 0,
                        NULL, PhysicalInstance::NO_INST, NULL, false, false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    VirtualManager::~VirtualManager(void)
    //--------------------------------------------------------------------------
    {
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
    ApEvent VirtualManager::get_use_event(void) const
    //--------------------------------------------------------------------------
    {
      return ApEvent::NO_AP_EVENT;
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
    InstanceView* VirtualManager::create_instance_top_view(
                            InnerContext *context, AddressSpaceID logical_owner)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return NULL;
    }

    /////////////////////////////////////////////////////////////
    // Instance Builder
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceBuilder::~InstanceBuilder(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    size_t InstanceBuilder::compute_needed_size(RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      if (!valid)
        initialize(forest);
      size_t total_field_bytes = 0;
      for (unsigned idx = 0; idx < field_sizes.size(); idx++)
        total_field_bytes += field_sizes[idx];
      return (total_field_bytes * instance_domain->get_volume());
    }

    //--------------------------------------------------------------------------
    PhysicalManager* InstanceBuilder::create_physical_instance(
                                                       RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      if (!valid)
        initialize(forest);
      // If there are no fields then we are done
      if (field_sizes.empty())
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORE_MEMORY_REQUEST,
                        "Ignoring request to create instance in "
                        "memory " IDFMT " with no fields.",
                        memory_manager->memory.id);
        return NULL;
      }
      // Construct the realm layout each time since (realm will take ownership 
      // after every instance call, so we need a new one each time)
      Realm::InstanceLayoutGeneric *realm_layout = 
        instance_domain->create_layout(realm_constraints, 
                                       constraints.ordering_constraint);
#ifdef DEBUG_LEGION
      assert(realm_layout != NULL);
#endif
      Realm::ProfilingRequestSet requests;
      // Add a profiling request to see if the instance is actually allocated
      // Make it very high priority so we get the response quickly
      ProfilingResponseBase base(this);
      Realm::ProfilingRequest &req = requests.add_request(
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID,
          &base, sizeof(base), LG_RESOURCE_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::InstanceAllocResult>();
      // Create a user event to wait on for the result of the profiling response
      profiling_ready = Runtime::create_rt_user_event();
#ifdef DEBUG_LEGION
      assert(!instance.exists()); // shouldn't exist before this
#endif
      ApEvent ready;
      if (runtime->profiler != NULL)
      {
        runtime->profiler->add_inst_request(requests, creator_id);
        ready = ApEvent(PhysicalInstance::create_instance(instance,
                  memory_manager->memory, realm_layout, requests));
        if (instance.exists())
        {
          unsigned long long creation_time = 
            Realm::Clock::current_time_in_nanoseconds();
          runtime->profiler->record_instance_creation(instance, 
              memory_manager->memory, creator_id, creation_time);
        }
      }
      else
        ready = ApEvent(PhysicalInstance::create_instance(instance,
                  memory_manager->memory, realm_layout, requests));
      // Wait for the profiling response
      if (!profiling_ready.has_triggered())
        profiling_ready.wait();
      // If we couldn't make it then we are done
      if (!instance.exists())
        return NULL;
      // For Legion Spy we need a unique ready event if it doesn't already
      // exist so we can uniquely identify the instance
      if (!ready.exists() && runtime->legion_spy_enabled)
      {
        ApUserEvent rename_ready = Runtime::create_ap_user_event();
        Runtime::trigger_event(rename_ready);
        ready = rename_ready;
      }
      // If we successfully made the instance then Realm 
      // took over ownership of the layout
      PhysicalManager *result = NULL;
      DistributedID did = forest->runtime->get_available_distributed_id();
      AddressSpaceID local_space = forest->runtime->address_space;
      FieldSpaceNode *field_node = ancestor->column_source;
      // Important implementation detail here: we pull the pointer constraint
      // out of the set of constraints here and don't include it in the layout
      // constraints so we can abstract over lots of different layouts. We'll
      // store the pointer constraint separately in the physical instance
      PointerConstraint pointer_constraint = constraints.pointer_constraint;
      constraints.pointer_constraint = PointerConstraint();
      // If we successfully made it then we can 
      // switch over the polarity of our constraints, this
      // shouldn't be necessary once Realm gets its act together
      // and actually tells us what the resulting constraints are
      constraints.field_constraint.contiguous = true;
      constraints.field_constraint.inorder = true;
      constraints.ordering_constraint.contiguous = true;
      constraints.memory_constraint = MemoryConstraint(
                                        memory_manager->memory.kind());
      const unsigned num_dims = instance_domain->get_num_dims();
      // Now let's find the layout constraints to use for this instance
      LayoutDescription *layout = 
        field_node->find_layout_description(instance_mask,num_dims,constraints);
      // If we couldn't find one then we make one
      if (layout == NULL)
      {
        // First make a new layout constraint
        LayoutConstraints *layout_constraints = 
          forest->runtime->register_layout(field_node->handle,
                                           constraints, true/*internal*/);
        // Then make our description
        layout = field_node->create_layout_description(instance_mask, num_dims,
                                  layout_constraints, mask_index_map,
                                  constraints.field_constraint.get_field_set(),
                                  field_sizes, serdez);
      }
      // Figure out what kind of instance we just made
      switch (constraints.specialized_constraint.get_kind())
      {
        case NO_SPECIALIZE:
        case NORMAL_SPECIALIZE:
          {
            // Now we can make the manager
            Reservation read_only_reservation = 
              Reservation::create_reservation();
            result = new InstanceManager(forest, did, local_space,
                                         memory_manager,
                                         instance, instance_domain, 
                                         own_domain, ancestor, layout, 
                                         pointer_constraint, 
                                         true/*register now*/, ready,
                                         false/*external instance*/,
                                         read_only_reservation);
            break;
          }
        case REDUCTION_FOLD_SPECIALIZE:
          {
            // TODO: this can go away once realm understands reduction
            // instances that contain multiple fields, Legion is ready
            // though so all you should have to do is delete this check
            if (field_sizes.size() > 1)
              REPORT_LEGION_ERROR(ERROR_ILLEGAL_REDUCTION_REQUEST,
                            "Illegal request for a reduction instance "
                            "containing multiple fields. Only a single field "
                            "is currently permitted for reduction instances.")
            ApUserEvent filled_and_ready = Runtime::create_ap_user_event();
            result = new FoldReductionManager(forest, did, local_space,
                                              memory_manager, 
                                              instance, layout, 
                                              pointer_constraint, 
                                              instance_domain, own_domain,
                                              ancestor, redop_id,
                                              reduction_op, filled_and_ready,
                                              true/*register now*/);
            // Before we can actually use this instance, we have to 
            // initialize it with a fill operation of the proper value
            // Don't record this fill operation because it is just part
            // of the semantics of reduction instances and not something
            // that we want Legion Spy to see
            void *fill_buffer = malloc(reduction_op->sizeof_rhs);
            reduction_op->init(fill_buffer, 1);
            std::vector<CopySrcDstField> dsts;
            {
              const std::vector<FieldID> &fill_fields = 
                constraints.field_constraint.get_field_set();
              layout->compute_copy_offsets(fill_fields, result, dsts);
            }
            PhysicalTraceInfo info;
#ifdef LEGION_SPY
            std::vector<Realm::CopySrcDstField> realm_dsts(dsts.size());
            for (unsigned idx = 0; idx < dsts.size(); idx++)
              realm_dsts[idx] = dsts[idx];
            ApEvent filled =
              instance_domain->issue_fill(NULL/*op*/, realm_dsts, fill_buffer,
                                          reduction_op->sizeof_rhs, ready,
                                          PredEvent::NO_PRED_EVENT, info);
#else
            ApEvent filled =
              instance_domain->issue_fill(NULL/*op*/, dsts, fill_buffer,
                                          reduction_op->sizeof_rhs, ready,
                                          PredEvent::NO_PRED_EVENT, info);
#endif
            // We can free the buffer after we've issued the fill
            free(fill_buffer);
            // Trigger our filled_and_ready event
            Runtime::trigger_event(filled_and_ready, filled);
            break;
          }
        case REDUCTION_LIST_SPECIALIZE:
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
    void InstanceBuilder::handle_profiling_response(
                                       const Realm::ProfilingResponse &response)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(response.has_measurement<
          Realm::ProfilingMeasurements::InstanceAllocResult>());
#endif
      Realm::ProfilingMeasurements::InstanceAllocResult result;
      result.success = false; // Need this to avoid compiler warnings
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      const bool measured =  
#endif
#endif
        response.get_measurement<
              Realm::ProfilingMeasurements::InstanceAllocResult>(result);
#ifdef DEBUG_LEGION
      assert(measured);
#endif
      // If we failed then clear the instance name since it is not valid
      if (!result.success)
      {
        // Destroy the instance first so that Realm can reclaim the ID
        instance.destroy();
        instance = PhysicalInstance::NO_INST;
      }
      // No matter what trigger the event
      Runtime::trigger_event(profiling_ready);
    }

    //--------------------------------------------------------------------------
    void InstanceBuilder::initialize(RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      compute_ancestor_and_domain(forest); 
      compute_layout_parameters();
      valid = true;
    }

    //--------------------------------------------------------------------------
    void InstanceBuilder::compute_ancestor_and_domain(RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      // Check to see if they are all empty, in which case we will make
      // an empty instance with its ancestor being the root of the region
      // tree so it can satisfy all empty regions in this region tree safely
      std::vector<RegionNode*> non_empty_regions;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        RegionNode *next = forest->get_node(regions[idx]);
        // Check for empty
        size_t volume = next->row_source->get_volume();
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
            instance_domain = next->row_source;
            while (next->parent != NULL)
              next = next->parent->parent;
            ancestor = next;
            return;
          }
          continue;
        }
        non_empty_regions.push_back(next);
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
        unsigned index = 0;
        std::vector<IndexSpace> union_spaces(non_empty_regions.size());
        for (std::vector<RegionNode*>::const_iterator it = 
              non_empty_regions.begin(); it != 
              non_empty_regions.end(); it++, index++)
        {
          union_spaces[index] = (*it)->row_source->handle;
          // Also find the common ancestor
          if (index > 0)
            ancestor = find_common_ancestor(ancestor, *it);
        }
        IndexSpace union_space(forest->runtime->get_unique_index_space_id(),
                               ancestor->handle.index_space.get_tree_id(),
                               ancestor->handle.get_type_tag());
        DistributedID did = 
          forest->runtime->get_available_distributed_id();
        forest->create_union_space(union_space, NULL/*task op*/,
                                   union_spaces, did);
        instance_domain = forest->get_node(union_space);
        own_domain = true;
      }
      else
        instance_domain = non_empty_regions[0]->row_source;
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
    void InstanceBuilder::compute_layout_parameters(void)
    //--------------------------------------------------------------------------
    {
      // First look at the OrderingConstraint to Figure out what kind
      // of instance we are building here, SOA, AOS, or hybrid
      // Make sure to check for splitting constraints if see sub-dimensions
      if (!constraints.splitting_constraints.empty())
        REPORT_LEGION_FATAL(ERROR_UNSUPPORTED_LAYOUT_CONSTRAINT,
            "Splitting layout constraints are not currently supported")
      const size_t num_dims = instance_domain->get_num_dims();
      OrderingConstraint &ord = constraints.ordering_constraint;
      if (!ord.ordering.empty())
      {
        // Find the index of the fields, if it is specified
        int field_idx = -1;
        std::set<DimensionKind> spatial_dims, to_remove;
        for (unsigned idx = 0; idx < ord.ordering.size(); idx++)
        {
          if (ord.ordering[idx] == DIM_F)
          {
            // Should never be duplicated 
            if (field_idx != -1)
              REPORT_LEGION_ERROR(ERROR_ILLEGAL_LAYOUT_CONSTRAINT,
                  "Illegal ordering constraint used during instance "
                  "creation contained multiple instances of DIM_F")
            else
            {
              // Check for AOS or SOA for now
              if ((idx > 0) && (idx != (ord.ordering.size()-1)))
                REPORT_LEGION_FATAL(ERROR_UNSUPPORTED_LAYOUT_CONSTRAINT,
                    "Ordering constraints must currently place DIM_F "
                    "in the first or last position as only AOS and SOA "
                    "layout constraints are currently supported")
              field_idx = idx;
            }
          }
          else if (ord.ordering[idx] > DIM_F)
            REPORT_LEGION_FATAL(ERROR_UNSUPPORTED_LAYOUT_CONSTRAINT,
              "Splitting layout constraints are not currently supported")
          else
          {
            // Should never be duplicated
            if (spatial_dims.find(ord.ordering[idx]) != spatial_dims.end())
              REPORT_LEGION_ERROR(ERROR_ILLEGAL_LAYOUT_CONSTRAINT,
                  "Illegal ordering constraint used during instance "
                  "creation contained multiple instances of dimension %d",
                  ord.ordering[idx])
            else
            {
              // Check to make sure that it is one of our dims
              // if not we can just filter it out of the ordering
              if (ord.ordering[idx] >= num_dims)
                to_remove.insert(ord.ordering[idx]);
              else
                spatial_dims.insert(ord.ordering[idx]);
            }
          }
        }
        // Remove any dimensions which don't matter
        if (!to_remove.empty())
        {
          for (std::vector<DimensionKind>::iterator it = ord.ordering.begin();
                it != ord.ordering.end(); /*nothing*/)
          {
            if (to_remove.find(*it) != to_remove.end())
              it = ord.ordering.erase(it);
            else
              it++;
          }
        }
#ifdef DEBUG_LEGION
        assert(spatial_dims.size() <= num_dims);
#endif
        // Fill in any spatial dimensions that we didn't see if necessary
        if (spatial_dims.size() < num_dims)
        {
          // See if we should push these dims front or back
          if (field_idx > -1)
          {
            // See if we should add these at the front or the back
            if (field_idx == 0)
            {
              // Add them to the back
              for (unsigned idx = 0; idx < num_dims; idx++)
              {
                DimensionKind dim = (DimensionKind)(DIM_X + idx);
                if (spatial_dims.find(dim) == spatial_dims.end())
                  ord.ordering.push_back(dim);
              }
            }
            else if (field_idx == int(ord.ordering.size()-1))
            {
              // Add them to the front
              for (int idx = (num_dims-1); idx >= 0; idx--)
              {
                DimensionKind dim = (DimensionKind)(DIM_X + idx);
                if (spatial_dims.find(dim) == spatial_dims.end())
                  ord.ordering.insert(ord.ordering.begin(), dim);
              }
            }
            else // Should either be AOS or SOA for now
              assert(false);
          }
          else
          {
            // No field dimension so just add the spatial ones on the back
            for (unsigned idx = 0; idx < num_dims; idx++)
            {
              DimensionKind dim = (DimensionKind)(DIM_X + idx);
              if (spatial_dims.find(dim) == spatial_dims.end())
                ord.ordering.push_back(dim);
            }
          }
        }
        // If we didn't see the field dimension either then add that
        // at the end to give us SOA layouts in general
        if (field_idx == -1)
          ord.ordering.push_back(DIM_F);
        // We've now got all our dimensions so we can set the
        // contiguous flag to true
        ord.contiguous = true;
      }
      else
      {
        // We had no ordering constraints so populate it with 
        // SOA constraints for now
        for (unsigned idx = 0; idx < num_dims; idx++)
          ord.ordering.push_back((DimensionKind)(DIM_X + idx));
        ord.ordering.push_back(DIM_F);
        ord.contiguous = true;
      }
#ifdef DEBUG_LEGION
      assert(ord.contiguous);
      assert(ord.ordering.size() == (num_dims + 1));
#endif
      // From this we should be able to compute the field groups 
      // Use the FieldConstraint to put any fields in the proper order
      FieldSpaceNode *field_node = ancestor->column_source;      
      const std::vector<FieldID> &field_set = 
        constraints.field_constraint.get_field_set(); 
      field_sizes.resize(field_set.size());
      mask_index_map.resize(field_set.size());
      serdez.resize(field_set.size());
      field_node->compute_field_layout(field_set, field_sizes,
                                       mask_index_map, serdez, instance_mask);
      // See if we have any specialization here that will 
      // require us to update the field sizes
      switch (constraints.specialized_constraint.get_kind())
      {
        case NO_SPECIALIZE:
        case NORMAL_SPECIALIZE:
          break;
        case REDUCTION_FOLD_SPECIALIZE:
          {
            // Reduction folds are a special case of normal specialize
            redop_id = constraints.specialized_constraint.get_reduction_op();
            reduction_op = Runtime::get_reduction_op(redop_id);
            for (unsigned idx = 0; idx < field_sizes.size(); idx++)
            {
              if (field_sizes[idx] != reduction_op->sizeof_lhs)
                REPORT_LEGION_ERROR(ERROR_UNSUPPORTED_LAYOUT_CONSTRAINT,
                    "Illegal reduction instance request with field %d "
                    "which has size %d but the LHS type of reduction "
                    "operator %d is %d", field_set[idx], int(field_sizes[idx]),
                    redop_id, int(reduction_op->sizeof_lhs))
              // Update the field sizes to the rhs of the reduction op
              field_sizes[idx] = reduction_op->sizeof_rhs;
            }
            break;
          }
        case REDUCTION_LIST_SPECIALIZE:
          {
            // TODO: implement list reduction instances
            assert(false);
            redop_id = constraints.specialized_constraint.get_reduction_op();
            reduction_op = Runtime::get_reduction_op(redop_id);
            break;
          }
        case VIRTUAL_SPECIALIZE:
          {
            REPORT_LEGION_ERROR(ERROR_ILLEGAL_REQUEST_VIRTUAL_INSTANCE,
                          "Illegal request to create a virtual instance");
            assert(false);
          }
        default:
          assert(false); // unknown kind
      }
      // Compute the field groups for realm 
      convert_layout_constraints(constraints, field_set, 
                                 field_sizes, realm_constraints); 
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceBuilder::convert_layout_constraints(
                    const LayoutConstraintSet &constraints,
                    const std::vector<FieldID> &field_set,
                    const std::vector<size_t> &field_sizes,
                            Realm::InstanceLayoutConstraints &realm_constraints)
    //--------------------------------------------------------------------------
    {
      const OrderingConstraint &ord = constraints.ordering_constraint;

      std::map<FieldID, size_t> field_alignments;
      if (ord.ordering.front() == DIM_F)
      {
        // AOS - all field in same group
        // Use a GCD of field sizes by default to make fields tighly packed
#ifdef DEBUG_LEGION
        assert(field_set.size() > 0);
        assert(field_sizes.size() > 0);
#endif
        size_t gcd = field_sizes[0];
        for (unsigned idx = 0; idx < field_set.size(); idx++)
        {
          while (field_sizes[idx] % gcd != 0)
            gcd >>= 1;
        }
#ifdef DEBUG_LEGION
        assert(gcd != 0);
#endif
        for (unsigned idx = 0; idx < field_set.size(); idx++)
          field_alignments[field_set[idx]] = gcd;
      }
      else if (ord.ordering.back() == DIM_F)
      {
        // SOA - each field is its own group
        // Use natural alignment by default
        for (unsigned idx = 0; idx < field_set.size(); idx++)
          field_alignments[field_set[idx]] = field_sizes[idx];
      }
      else // Have to be AOS or SOA for now
        assert(false);

      const std::vector<AlignmentConstraint> &alignments =
        constraints.alignment_constraints;
      for (std::vector<AlignmentConstraint>::const_iterator it =
           alignments.begin(); it != alignments.end(); ++it)
      {
        // TODO: We support only equality constraints for now
        assert(it->eqk != EQ_EK);
        field_alignments[it->fid] = it->alignment;
      }

      if (ord.ordering.front() == DIM_F)
      {
        // AOS - all field in same group
        realm_constraints.field_groups.resize(1);
        realm_constraints.field_groups[0].resize(field_set.size());
        for (unsigned idx = 0; idx < field_set.size(); idx++)
        {
#ifdef DEBUG_LEGION
          assert(field_alignments.find(field_set[idx]) !=
                 field_alignments.end());
#endif
          realm_constraints.field_groups[0][idx].field_id = field_set[idx];
          realm_constraints.field_groups[0][idx].offset = -1;
          realm_constraints.field_groups[0][idx].size = field_sizes[idx];
          realm_constraints.field_groups[0][idx].alignment =
            field_alignments[field_set[idx]];
        }
      }
      else if (ord.ordering.back() == DIM_F)
      {
        // SOA - each field is its own group
        realm_constraints.field_groups.resize(field_set.size());
        for (unsigned idx = 0; idx < field_set.size(); idx++)
        {
#ifdef DEBUG_LEGION
          assert(field_alignments.find(field_set[idx]) !=
                 field_alignments.end());
#endif
          realm_constraints.field_groups[idx].resize(1);
          realm_constraints.field_groups[idx][0].field_id = field_set[idx];
          realm_constraints.field_groups[idx][0].offset = -1;
          realm_constraints.field_groups[idx][0].size = field_sizes[idx];
          realm_constraints.field_groups[idx][0].alignment =
            field_alignments[field_set[idx]];
        }
      }
      else // Have to be AOS or SOA for now
        assert(false);
      // TODO: Next go through and check for any offset constraints for fields
    }
    
  }; // namespace Internal
}; // namespace Legion

