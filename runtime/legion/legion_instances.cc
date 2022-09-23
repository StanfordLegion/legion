/* Copyright 2022 Stanford University, NVIDIA Corporation
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

namespace LegionRuntime {
  namespace Accessor {
    namespace DebugHooks {
      // these are calls that can be implemented by a higher level (e.g. Legion)
      // to perform privilege/bounds checks on accessor reference and produce 
      // more useful information for debug

      /*extern*/ void (*check_bounds_ptr)(void *region, ptr_t ptr) = 0;
      /*extern*/ void (*check_bounds_dpoint)(void *region, 
                                    const Legion::DomainPoint &dp) = 0;

      /*extern*/ const char *(*find_privilege_task_name)(void *region) = 0;
    };
  };
};

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
      for (LegionDeque<std::pair<FieldMask,FieldMask> >::const_iterator
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
        compress_mask<STATIC_LOG2(LEGION_MAX_FIELDS)>(compressed, full_mask);
        compressed_cache.push_back(
            std::pair<FieldMask,FieldMask>(src_mask, compressed));
      }
      const unsigned pop_count = FieldMask::pop_count(compressed);
#ifdef DEBUG_LEGION
      assert(pop_count == FieldMask::pop_count(src_mask));
#endif
      unsigned offset = dst_fields.size();
      dst_fields.resize(offset + pop_count);
      int next_start = 0;
      for (unsigned idx = 0; idx < pop_count; idx++)
      {
        int index = compressed.find_next_set(next_start);
        CopySrcDstField &field = dst_fields[offset+idx];
        field = offsets[index];
        // We'll start looking again at the next index after this one
        next_start = index + 1;
      }
    }

    //--------------------------------------------------------------------------
    FieldMask CopyAcrossHelper::convert_src_to_dst(const FieldMask &src_mask)
    //--------------------------------------------------------------------------
    {
      FieldMask dst_mask;
      if (!src_mask)
        return dst_mask;
      if (forward_map.empty())
      {
#ifdef DEBUG_LEGION
        assert(src_indexes.size() == dst_indexes.size());
#endif
        for (unsigned idx = 0; idx < src_indexes.size(); idx++)
        {
#ifdef DEBUG_LEGION
          assert(forward_map.find(src_indexes[idx]) == forward_map.end());
#endif
          forward_map[src_indexes[idx]] = dst_indexes[idx];
        }
      }
      int index = src_mask.find_first_set();
      while (index >= 0)
      {
#ifdef DEBUG_LEGION
        assert(forward_map.find(index) != forward_map.end());
#endif
        dst_mask.set_bit(forward_map[index]);
        index = src_mask.find_next_set(index+1);
      }
      return dst_mask;
    }

    //--------------------------------------------------------------------------
    FieldMask CopyAcrossHelper::convert_dst_to_src(const FieldMask &dst_mask)
    //--------------------------------------------------------------------------
    {
      FieldMask src_mask;
      if (!dst_mask)
        return src_mask;
      if (backward_map.empty())
      {
#ifdef DEBUG_LEGION
        assert(src_indexes.size() == dst_indexes.size());
#endif
        for (unsigned idx = 0; idx < dst_indexes.size(); idx++)
        {
#ifdef DEBUG_LEGION
          assert(backward_map.find(dst_indexes[idx]) == backward_map.end());
#endif
          backward_map[dst_indexes[idx]] = src_indexes[idx];
        }
      }
      int index = dst_mask.find_first_set();
      while (index >= 0)
      {
#ifdef DEBUG_LEGION
        assert(backward_map.find(index) != backward_map.end());
#endif
        src_mask.set_bit(backward_map[index]);
        index = dst_mask.find_next_set(index+1);
      }
      return src_mask;
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
      // Greater than or equal because local fields can alias onto the
      // same index for the allocated instances, note that the fields
      // themselves still get allocated their own space in the instance
      assert(mask_index_map.size() >= 
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
      for (std::map<FieldID,unsigned>::const_iterator it = 
            field_indexes.begin(); it != field_indexes.end(); it++)
        LegionSpy::log_physical_instance_field(inst_event, it->first);
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::compute_copy_offsets(const FieldMask &copy_mask,
                                           const PhysicalInstance instance,
#ifdef LEGION_SPY
                                           const ApEvent inst_event,
#endif
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
                 LegionList<std::pair<FieldMask,FieldMask> > >::const_iterator
                   finder = comp_cache.find(hash_key);
        if (finder != comp_cache.end())
        {
          for (LegionList<std::pair<FieldMask,FieldMask> >::const_iterator it =
                finder->second.begin(); it != finder->second.end(); it++)
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
        compress_mask<STATIC_LOG2(LEGION_MAX_FIELDS)>(compressed, 
                                                      allocated_fields);
        // Save the result in the cache, duplicates from races here are benign
        AutoLock o_lock(layout_lock);
        comp_cache[hash_key].push_back(
            std::pair<FieldMask,FieldMask>(copy_mask,compressed));
      }
      // It is absolutely imperative that these infos be added in
      // the order in which they appear in the field mask so that 
      // they line up in the same order with the source/destination infos
      // (depending on the calling context of this function
      const unsigned pop_count = FieldMask::pop_count(compressed);
#ifdef DEBUG_LEGION
      assert(pop_count == FieldMask::pop_count(copy_mask));
#endif
      unsigned offset = fields.size();
      fields.resize(offset + pop_count);
      int next_start = 0;
      for (unsigned idx = 0; idx < pop_count; idx++)
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
    void LayoutDescription::compute_copy_offsets(
                                   const std::vector<FieldID> &copy_fields, 
                                   const PhysicalInstance instance,
#ifdef LEGION_SPY
                                   const ApEvent inst_event,
#endif
                                   std::vector<CopySrcDstField> &fields)
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
      // We need to check equality on the entire constraint sets
      return *constraints == candidate_constraints;
    }

    //--------------------------------------------------------------------------
    bool LayoutDescription::match_layout(const LayoutDescription *layout,
                                         unsigned num_dims) const
    //--------------------------------------------------------------------------
    {
      if (num_dims != total_dims)
        return false;
      // This is a sound test, but it doesn't guarantee that the field sets
      // match since fields can be allocated and freed between instance
      // creations, so while this is a necessary precondition, it is not
      // sufficient that the two sets of fields are the same, to guarantee
      // that we actually need to check the FieldIDs which happens next
      if (layout->allocated_fields != allocated_fields)
        return false;

      // Check equality on the entire constraint sets
      return *layout->constraints == *constraints;
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
      handle_unpack_layout_description(LayoutConstraints *constraints,
                            FieldSpaceNode *field_space_node, size_t total_dims)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(constraints != NULL);
#endif
      FieldMask instance_mask;
      const std::vector<FieldID> &field_set = 
        constraints->field_constraint.get_field_set(); 
      std::vector<size_t> field_sizes(field_set.size());
      std::vector<unsigned> mask_index_map(field_set.size());
      std::vector<CustomSerdezID> serdez(field_set.size());
      field_space_node->compute_field_layout(field_set, field_sizes,
                               mask_index_map, serdez, instance_mask);
      LayoutDescription *result = 
        field_space_node->create_layout_description(instance_mask, total_dims,
                  constraints, mask_index_map, field_set, field_sizes, serdez);
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      return result;
    }

    /////////////////////////////////////////////////////////////
    // InstanceManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceManager::InstanceManager(RegionTreeForest *ctx,
                                     AddressSpaceID owner_space,
                                     DistributedID did, LayoutDescription *desc,
                                     FieldSpaceNode *node, 
                                     IndexSpaceExpression *domain,
                                     RegionTreeID tid, bool register_now)
      : DistributedCollectable(ctx->runtime, did, owner_space, register_now), 
        context(ctx), layout(desc), field_space_node(node),
        instance_domain(domain), tree_id(tid)
    //--------------------------------------------------------------------------
    {
      // Add a reference to the layout
      if (layout != NULL)
        layout->add_reference();
      if (field_space_node != NULL)
        field_space_node->add_nested_gc_ref(did);
      if (instance_domain != NULL)
        instance_domain->add_nested_expression_reference(did);
    }

    //--------------------------------------------------------------------------
    InstanceManager::~InstanceManager(void)
    //--------------------------------------------------------------------------
    {
      if ((layout != NULL) && layout->remove_reference())
        delete layout;
      if ((field_space_node != NULL) &&
          field_space_node->remove_nested_gc_ref(did))
        delete field_space_node;
      if ((instance_domain != NULL) && 
          instance_domain->remove_nested_expression_reference(did))
        delete instance_domain;
    } 

    //--------------------------------------------------------------------------
    bool InstanceManager::meets_region_tree(
                                const std::vector<LogicalRegion> &regions) const
    //--------------------------------------------------------------------------
    {
      for (std::vector<LogicalRegion>::const_iterator it = 
            regions.begin(); it != regions.end(); it++)
      {
        // Check to see if the region tree IDs are the same
        if (it->get_field_space() != tree_id)
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool InstanceManager::entails(LayoutConstraints *constraints,
                                  const DomainPoint &key,
                               const LayoutConstraint **failed_constraint) const
    //--------------------------------------------------------------------------
    {
      const PointerConstraint &pointer = constraints->pointer_constraint;
      if (pointer.is_valid)
      {
        PointerConstraint pointer_constraint = get_pointer_constraint(key);
        // Always test the pointer constraint locally
        if (!pointer_constraint.entails(constraints->pointer_constraint))
        {
          if (failed_constraint != NULL)
            *failed_constraint = &pointer;
          return false;
        }
      }
      return layout->constraints->entails_without_pointer(constraints,
              (instance_domain != NULL) ? instance_domain->get_num_dims() : 0,
              failed_constraint);
    }

    //--------------------------------------------------------------------------
    bool InstanceManager::entails(const LayoutConstraintSet &constraints,
                                  const DomainPoint &key,
                               const LayoutConstraint **failed_constraint) const
    //--------------------------------------------------------------------------
    {
      const PointerConstraint &pointer = constraints.pointer_constraint;
      if (pointer.is_valid)
      {
        PointerConstraint pointer_constraint = get_pointer_constraint(key);
        // Always test the pointer constraint locally
        if (!pointer_constraint.entails(constraints.pointer_constraint))
        {
          if (failed_constraint != NULL)
            *failed_constraint = &pointer;
          return false;
        }
      }
      return layout->constraints->entails_without_pointer(constraints,
              (instance_domain != NULL) ? instance_domain->get_num_dims() : 0,
              failed_constraint);
    }

    //--------------------------------------------------------------------------
    bool InstanceManager::conflicts(LayoutConstraints *constraints,
                                    const DomainPoint &key,
                             const LayoutConstraint **conflict_constraint) const
    //--------------------------------------------------------------------------
    {
      const PointerConstraint &pointer = constraints->pointer_constraint;
      if (pointer.is_valid)
      {
        PointerConstraint pointer_constraint = get_pointer_constraint(key);
        // Always test the pointer constraint locally
        if (pointer_constraint.conflicts(constraints->pointer_constraint))
        {
          if (conflict_constraint != NULL)
            *conflict_constraint = &pointer;
          return true;
        }
      }
      // We know our layouts don't have a pointer constraint so nothing special
      return layout->constraints->conflicts(constraints,
              (instance_domain != NULL) ? instance_domain->get_num_dims() : 0,
              conflict_constraint);
    }

    //--------------------------------------------------------------------------
    bool InstanceManager::conflicts(const LayoutConstraintSet &constraints,
                                    const DomainPoint &key,
                             const LayoutConstraint **conflict_constraint) const
    //--------------------------------------------------------------------------
    {
      const PointerConstraint &pointer = constraints.pointer_constraint;
      if (pointer.is_valid)
      {
        PointerConstraint pointer_constraint = get_pointer_constraint(key);
        // Always test the pointer constraint locally
        if (pointer_constraint.conflicts(constraints.pointer_constraint))
        {
          if (conflict_constraint != NULL)
            *conflict_constraint = &pointer;
          return true;
        }
      }
      // We know our layouts don't have a pointer constraint so nothing special
      return layout->constraints->conflicts(constraints,
              (instance_domain != NULL) ? instance_domain->get_num_dims() : 0,
              conflict_constraint);
    }

    /////////////////////////////////////////////////////////////
    // PhysicalManager
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalManager::PhysicalManager(RegionTreeForest *ctx, 
                                     LayoutDescription *layout, 
                                     DistributedID did, 
                                     AddressSpaceID owner_space, 
                                     const size_t footprint, 
                                     ReductionOpID redop_id, 
                                     const ReductionOp *rop,
                                     FieldSpaceNode *node, 
                                     IndexSpaceExpression *index_domain, 
                                     const void *pl, size_t pl_size,
                                     RegionTreeID tree_id, ApEvent u_event,
                                     bool register_now)
      : InstanceManager(ctx, owner_space, did, layout, node,
          // If we're on the owner node we need to produce the expression
          // that actually describes this points in this space
          // On remote nodes we'll already have it from the owner
          (owner_space == ctx->runtime->address_space) ?
             index_domain->create_layout_expression(pl, pl_size) : index_domain,
          tree_id, register_now), 
        instance_footprint(footprint), reduction_op(rop), redop(redop_id),
        unique_event(u_event), piece_list(pl), piece_list_size(pl_size),
        gc_state(COLLECTABLE_GC_STATE), pending_changes(0),
        remaining_collection_guards(0), currently_active(false),
        min_gc_priority(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalManager::~PhysicalManager(void)
    //--------------------------------------------------------------------------
    {
      if (!gc_events.empty())
      {
        // There's no need to launch a task to do this, if we're being
        // deleted it's because the instance was deleted and therefore
        // all the users are done using it
        for (std::map<CollectableView*,CollectableInfo>::iterator it = 
              gc_events.begin(); it != gc_events.end(); it++)
          CollectableView::handle_deferred_collect(it->first,
                                                   it->second.view_events);
        gc_events.clear();
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::log_instance_creation(UniqueID creator_id,
                Processor proc, const std::vector<LogicalRegion> &regions) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(runtime->legion_spy_enabled);
#endif
      const ApEvent inst_event = get_unique_event();
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
    InstanceView* PhysicalManager::create_instance_top_view(
                            InnerContext *own_ctx, AddressSpaceID logical_owner)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      const DistributedID view_did = 
        context->runtime->get_available_distributed_id();
      const UniqueID context_uid = own_ctx->get_context_uid();
      register_active_context(own_ctx);
      if (redop > 0)
        return new ReductionView(context, view_did, owner_space,
            logical_owner, this, context_uid, true/*register now*/);
      else
        return new MaterializedView(context, view_did, owner_space, 
              logical_owner, this, context_uid, true/*register now*/);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
      assert(!currently_active);
      assert((gc_state == COLLECTABLE_GC_STATE) ||
              (gc_state == ACQUIRED_GC_STATE));
      assert(!deferred_deletion.exists());
#endif
      currently_active = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
      assert(currently_active);
#endif
      currently_active = false;
      if (deferred_deletion.exists())
      {
#ifdef DEBUG_LEGION
        assert((gc_state == COLLECTED_GC_STATE) || is_external_instance());
#endif
        Runtime::trigger_event(deferred_deletion);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifndef DEBUG_LEGION
      // In non-debug mode we can just add the valid reference to the
      // owner without needing to check. While this isn't strictly necessary
      // for correctness, it is an important performance optimization to help
      // the garbage collector quickly detect when instances should not be
      // eligible for collection early in the process before trying to do
      // the whole distributed protocol.
      if (!is_owner())
        send_remote_valid_increment(owner_space, mutator);
#endif
      AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
      assert(!deferred_deletion.exists());
      assert(gc_state != COLLECTED_GC_STATE);
      // In debug mode we eagerly add valid references such that the owner
      // is valid as long as a copy of the manager on one node is valid
      // This way we can easily check that acquires are being done safely
      // if instance isn't already valid somewhere
      if ((gc_state != ACQUIRED_GC_STATE) && !is_external_instance())
      {
        // Should never be here if we're the owner as it indicates that
        // we tried to add a valid reference without first doing an acquire
        assert(!is_owner());
        // Send a message to check that we can safely do the acquire
        const RtUserEvent done = Runtime::create_rt_user_event();
        std::atomic<bool> result(true);
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(&result);
          rez.serialize(done);
          rez.serialize<bool>(false/*acquired*/);
        }
        runtime->send_gc_debug_request(owner_space, rez);
        if (!done.has_triggered())
          done.wait();
        assert(result.load());
      }
      else if (!is_owner())
      {
        // Send the message to do the acquire on the owner node
        const RtUserEvent done = Runtime::create_rt_user_event();
        std::atomic<bool> result(true);
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(&result);
          rez.serialize(done);
          rez.serialize<bool>(true/*acquired*/);
        }
        runtime->send_gc_debug_request(owner_space, rez);
        if (!done.has_triggered())
          done.wait();
        assert(result.load());
      }
#endif
      gc_state = VALID_GC_STATE;
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_garbage_collection_debug_request(
                   Runtime *runtime, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      std::atomic<bool> *target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);
      bool acquired;
      derez.deserialize(acquired);

      PhysicalManager *manager = static_cast<PhysicalManager*>(
          runtime->weak_find_distributed_collectable(did));
      if (manager != NULL)
      {
        if (acquired)
        {
          LocalReferenceMutator mutator;
          // Should be guaranteed to be able to acquire thi
          if (manager->acquire_instance(REMOTE_DID_REF, &mutator))
          {
            Runtime::trigger_event(done, mutator.get_done_event());
            return;
          }
        }
        else
        {
          if (manager->check_valid_and_increment(REMOTE_DID_REF))
          {
            // This must already be valid for us to succeed
            Runtime::trigger_event(done);
            return;
          }
        }
      }
      // If we get here, we failed so send the response
      Serializer rez;
      {
        RezCheck z2(rez);
        rez.serialize(target);
        rez.serialize(done);
      }
      runtime->send_gc_debug_response(source, rez);
#else
      assert(false); // should never get this in release mode
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_garbage_collection_debug_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      DerezCheck z(derez);
      std::atomic<bool> *target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);

      target->store(false);
      Runtime::trigger_event(done);
#else
      assert(false); // should never get this in release mode
#endif
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        send_remote_valid_decrement(owner_space, mutator);
      AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
      assert(gc_state == VALID_GC_STATE);
#endif
      if (pending_changes == 0)
      {
        gc_state = COLLECTABLE_GC_STATE;
        prune_gc_events();
      }
      else
        gc_state = ACQUIRED_GC_STATE;
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
      bool success = false;
      {
        AutoLock i_lock(inst_lock);
        switch (gc_state)
        {
          case VALID_GC_STATE:
          case ACQUIRED_GC_STATE:
            {
              pending_changes++;
              success = true;
              break;
            }
          case COLLECTABLE_GC_STATE:
            {
              gc_state = ACQUIRED_GC_STATE;
              pending_changes++;
              success = true;
              break;
            }
          case PENDING_COLLECTED_GC_STATE:
            {
              // Hurry the garbage collector is trying to eat it!
              if (is_owner())
              {
                // We're the owner so we can save this
                gc_state = ACQUIRED_GC_STATE;
                // We add the collection counts to the pending
                // changes to ensure we won't try to collect again
                // until all the participants of the previous collection
                // have checked the result of the collection
                pending_changes++;
                success = true;
              }
              // Not the owner so we need to send a message to the
              // owner to have it try to do the acquire
              break;
            }
          case COLLECTED_GC_STATE:
            return false;
          default:
            assert(false);
        }
      }
      if (success)
      {
        add_base_valid_ref(source, mutator);
        AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
        assert(pending_changes > 0);
        assert(gc_state == VALID_GC_STATE);
#endif
        pending_changes--;
        return true;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!is_owner()); 
#endif
        std::atomic<bool> result(false);
        const RtUserEvent ready = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(this);
          rez.serialize(source);
          rez.serialize(mutator);
          rez.serialize(&result);
          rez.serialize(ready);
        }
        runtime->send_acquire_request(owner_space, rez);
        ready.wait();
        if (result.load())
          return true;
        // The only reason we fail is if the instance has been collected
        // so we can update the gc_state
        AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
        assert((gc_state == PENDING_COLLECTED_GC_STATE) || 
                (gc_state == COLLECTED_GC_STATE)); 
#endif
        gc_state = COLLECTED_GC_STATE;
        return false;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_acquire_request(Runtime *runtime,
                                     Deserializer &derez, AddressSpaceID source) 
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      PhysicalManager *remote;
      derez.deserialize(remote);
      ReferenceSource ref;
      derez.deserialize(ref);
      ReferenceMutator *remote_mutator;
      derez.deserialize(remote_mutator);
      std::atomic<bool> *result;
      derez.deserialize(result);
      RtUserEvent ready;
      derez.deserialize(ready);

      PhysicalManager *manager = static_cast<PhysicalManager*>(
          runtime->weak_find_distributed_collectable(did));
      if (manager != NULL)
      {
        LocalReferenceMutator mutator;
        if (manager->acquire_instance(REMOTE_DID_REF, &mutator))
        {
          // We succeeded so send the response back with the reference
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(remote);
            rez.serialize(ref);
            rez.serialize(remote_mutator);
            rez.serialize(mutator.get_done_event());
            rez.serialize(result);
            rez.serialize(ready);
          }
          runtime->send_acquire_response(source, rez);
        }
      }
      // We failed, so the flag is already set, just trigger the event
      Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_acquire_response(
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      PhysicalManager *manager;
      derez.deserialize(manager);
      ReferenceSource ref;
      derez.deserialize(ref);
      ReferenceMutator *local_mutator;
      derez.deserialize(local_mutator);
      RtEvent applied;
      derez.deserialize(applied);
      std::atomic<bool> *result;
      derez.deserialize(result);
      RtUserEvent ready;
      derez.deserialize(ready);

      // Add the local reference with the right kind and then remove the
      // remote distributed reference we added on the owner node
      LocalReferenceMutator mutator;
      manager->add_base_valid_ref(ref, &mutator);
      if (applied.exists())
        mutator.record_reference_mutation_effect(applied);
      manager->send_remote_valid_decrement(source, local_mutator,
          mutator.get_done_event());
      result->store(true);
      Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::can_collect(AddressSpaceID source,
                                      bool &already_collected)
    //--------------------------------------------------------------------------
    {
      // This is a lightweight test that shouldn't involve any communication
      // or commitment to performing a collection. It's just for finding
      // instances that we know are locally collectable
      already_collected = false;
      AutoLock i_lock(inst_lock);
      // Do a quick to check to see if we can do a collection on the local node
      if ((gc_state == ACQUIRED_GC_STATE) || (gc_state == VALID_GC_STATE))
        return false;
      // If it's already collected then we're done
      if (gc_state == COLLECTED_GC_STATE)
      {
#ifdef DEBUG_LEGION
        assert(is_owner());
#endif
        already_collected = true;
        return false;
      }
      if (!is_owner() && (source == owner_space))
        gc_state = PENDING_COLLECTED_GC_STATE;
      return true;
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_garbage_collection_request(
                   Runtime *runtime, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      std::atomic<bool> *result;
      derez.deserialize(result);
      RtEvent *target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);

      PhysicalManager *manager = static_cast<PhysicalManager*>(
          runtime->weak_find_distributed_collectable(did));
      if (manager == NULL)
      {
        // This was already collected, so indicate that
        Serializer rez;
        {
          RezCheck z2(rez);
          rez.serialize(result);
          rez.serialize(target);
          rez.serialize(RtEvent::NO_RT_EVENT);
          rez.serialize(done);
        }
        runtime->send_gc_response(source, rez);
        return;
      }
      RtEvent ready;
      if (manager->collect(ready))
      {
        Serializer rez;
        {
          RezCheck z2(rez);
          rez.serialize(result);
          rez.serialize(target);
          rez.serialize(ready);
          rez.serialize(done);
        }
        runtime->send_gc_response(source, rez);
      }
      else // Couldn't collect so we are done
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_garbage_collection_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::atomic<bool> *result;
      derez.deserialize(result);
      RtEvent *target;
      derez.deserialize(target);
      derez.deserialize(*target);
      RtUserEvent done;
      derez.deserialize(done);

      result->store(true);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_garbage_collection_acquire(
                   Runtime *runtime, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      std::atomic<unsigned> *target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);

      RtEvent ready;
      PhysicalManager *manager = 
        runtime->find_or_request_instance_manager(did, ready);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();

      bool dummy_collected = false;
      if (manager->can_collect(source, dummy_collected))
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(target);
          rez.serialize(done);
        }
        runtime->send_gc_acquired(source, rez);
      }
      else
        Runtime::trigger_event(done);
#ifdef DEBUG_LEGION
      assert(!dummy_collected); // should never be set here
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_garbage_collection_acquired(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::atomic<unsigned> *target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);

#ifdef DEBUG_LEGION
#ifndef NDEBUG
      const unsigned prev =
#endif
#endif
      target->fetch_sub(1);
#ifdef DEBUG_LEGION
      assert(prev > 0);
#endif
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::pack_garbage_collection_state(Serializer &rez,
                                                        AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // We have to atomically get the current collection state and 
      // update the set of remote instances, note that it can be read-only
      // since we're just reading the state and the `update-remote_instaces'
      // call will take its own exclusive lock
      AutoLock i_lock(inst_lock,1,false/*exclusive*/);
      switch (gc_state)
      {
        case VALID_GC_STATE:
        case ACQUIRED_GC_STATE:
        case COLLECTABLE_GC_STATE:
          {
            rez.serialize(COLLECTABLE_GC_STATE);
            break;
          }
        case PENDING_COLLECTED_GC_STATE:
        case COLLECTED_GC_STATE:
          {
            rez.serialize(gc_state);
            break;
          }
        default:
          assert(false);
      }
      update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::initialize_remote_gc_state(
                                                   GarbageCollectionState state)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
      assert(!is_owner());
      assert(gc_state == COLLECTABLE_GC_STATE);
#endif
      gc_state = state;
      // If we're in a pending collectable state, then add a reference
      if (state == PENDING_COLLECTED_GC_STATE)
        add_base_resource_ref(PENDING_COLLECTIVE_REF);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::collect(RtEvent &ready)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
      // Do a quick to check to see if we can do a collection on the local node
      if ((gc_state == ACQUIRED_GC_STATE) || (gc_state == VALID_GC_STATE))
        return false;
      // If it's already collected then we're done
      if (gc_state == COLLECTED_GC_STATE)
        return true;
      if (is_owner())
      {
        // Check to see if anyone is already performing a deletion
        // on this manager, if so then deduplicate
        if (gc_state == COLLECTABLE_GC_STATE)
        {
#ifdef DEBUG_LEGION
          assert(pending_changes == 0);
#endif
          gc_state = PENDING_COLLECTED_GC_STATE;
          const size_t needed_guards = count_remote_instances();
          if (needed_guards > 0)
          {
            remaining_collection_guards.store(needed_guards);
            struct AcquireFunctor {
              AcquireFunctor(DistributedID d, Runtime *rt, 
                             std::atomic<unsigned> *c)
                : did(d), runtime(rt), count(c) { }
              inline void apply(AddressSpaceID target)
              {
                if (target == runtime->address_space)
                  return;
                const RtUserEvent ready_event = Runtime::create_rt_user_event();
                Serializer rez;
                {
                  RezCheck z(rez);
                  rez.serialize(did);
                  rez.serialize(count);
                  rez.serialize(ready_event);
                }
                runtime->send_gc_acquire(target, rez);
                ready_events.push_back(ready_event);
              }
              const DistributedID did;
              Runtime *const runtime;
              std::atomic<unsigned> *const count;
              std::vector<RtEvent> ready_events;
            };
            AcquireFunctor functor(did, runtime, &remaining_collection_guards);
            map_over_remote_instances(functor);
            collection_ready = Runtime::merge_events(functor.ready_events);
          }
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(gc_state == PENDING_COLLECTED_GC_STATE); 
          // Should alaready have outstanding changes for this deletion
          assert(pending_changes > 0);
#endif
        }
        pending_changes++;
        const RtEvent wait_on = collection_ready;
        if (!wait_on.has_triggered())
        {
          i_lock.release();
          wait_on.wait();
          i_lock.reacquire();
        }
#ifdef DEBUG_LEGION
        assert(pending_changes > 0);
#endif
        switch (gc_state)
        {
          // Anything in these states means the collection attempt failed
          case VALID_GC_STATE:
          case ACQUIRED_GC_STATE:
            {
              // Something else has acquired the instance before we
              // could finish the collection and already sent the
              // release notifications, remove our pending reference
              if (--pending_changes == 0)
              {
                // If we're not in the valid state then go back
                // to the collectable state so we can try again
                if (gc_state == ACQUIRED_GC_STATE)
                {
                  gc_state = COLLECTABLE_GC_STATE;
                  prune_gc_events();
                }
              }
              break;
            }
          case PENDING_COLLECTED_GC_STATE:
            {
#ifdef DEBUG_LEGION
              // Precondition should have triggered if we're here
              assert(collection_ready.has_triggered());
#endif
              // Check to see if there were any collection guards we
              // were unable to acquire on remote nodes
              if (remaining_collection_guards.load() > 0)
              {
                // See if we're the last release, if not then we
                // keep it in this state
                if (--pending_changes == 0)
                {
                  gc_state = COLLECTABLE_GC_STATE;
                  prune_gc_events();
                }
              }
              else
              {
                // Deletion success and we're the first ones to discover it
                // Move to the deletion state and send the deletion messages
                // to mark that we successfully performed the deletion
                gc_state = COLLECTED_GC_STATE;
                // Now we can perform the deletion which will release the lock
                ready = perform_deletion(runtime->address_space, &i_lock);
                return true;
              }
              break;
            }
          case COLLECTED_GC_STATE:
            {
              // Save the event for when the collection is done
              ready = collection_ready;
              return true;
            }
          default:
            assert(false); // should not be in any other state
        }
        return false;
      }
      else
      {
        // No longer need the lock here since we're just sending a message
        i_lock.release();
        // Send it to the owner to check
        std::atomic<bool> result(false);
        const RtUserEvent done = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(&result);
          rez.serialize(&ready);
          rez.serialize(done);
        }
        runtime->send_gc_request(owner_space, rez);
        done.wait();
        return result.load();
      }
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalManager::set_garbage_collection_priority(MapperID mapper_id,
                                               Processor p, GCPriority priority)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_external_instance());
#endif
      RtEvent wait_on;
      RtUserEvent done_event;
      bool add_never_reference = false;
      bool remove_never_reference = false;
      { 
        const std::pair<MapperID,Processor> key(mapper_id, p);
        AutoLock i_lock(inst_lock);
        // If this thing is already deleted then there is nothing to do
        if (gc_state == COLLECTED_GC_STATE)
          return RtEvent::NO_RT_EVENT;
        std::map<std::pair<MapperID,Processor>,GCPriority>::iterator finder =
          mapper_gc_priorities.find(key);
        if (finder == mapper_gc_priorities.end())
        {
          mapper_gc_priorities[key] = priority;
          if (min_gc_priority <= priority)
            return RtEvent::NO_RT_EVENT;
        }
        else
        {
          // See if we're the minimum priority
          if (min_gc_priority < finder->second)
          {
            // We weren't one of the minimum priorities before
            finder->second = priority;
            if (min_gc_priority <= priority)
              return RtEvent::NO_RT_EVENT;
            // Otherwise fall through and update the min priority
          }
          else
          {
            // We were one of the minimum priorities before
#ifdef DEBUG_LEGION
            assert(finder->second == min_gc_priority);
#endif
            // If things don't change then there is nothing to do
            if (finder->second == priority)
              return RtEvent::NO_RT_EVENT;
            finder->second = priority;
            if (min_gc_priority < priority)
            {
              // Raising one of the old minimum priorities
              // See what the new min priority is
              for (std::map<std::pair<MapperID,Processor>,GCPriority>::
                    const_iterator it = mapper_gc_priorities.begin(); it !=
                    mapper_gc_priorities.end(); it++)
              {
                // If the new minimum priority is still the same we're done
                if (it->second == min_gc_priority)
                  return RtEvent::NO_RT_EVENT;
                if (it->second < priority)
                  priority = it->second;
              }
#ifdef DEBUG_LEGION
              // If we get here then we're increasing the minimum priority
              assert(min_gc_priority < priority);
#endif
            }
            // Else lowering the minimum priority
          }
        }
        // If we get here then we're changing the minimum priority
#ifdef DEBUG_LEGION
        assert(priority != min_gc_priority);
#endif
        // Only deal with never collection refs on the owner node where
        // the ultimate garbage collection decisions are to be made
        if (is_owner())
        {
          if (priority < min_gc_priority)
          {
#ifdef DEBUG_LEGION
            assert(LEGION_GC_NEVER_PRIORITY < min_gc_priority);
#endif
            if (priority == LEGION_GC_NEVER_PRIORITY)
            {
              add_never_reference = true;
              // Check the garbage collection state because this is going 
              // to be like an acquire operation
              switch (gc_state)
              {
                case VALID_GC_STATE:
                case ACQUIRED_GC_STATE:
                  {
                    pending_changes++;
                    break;
                  }
                case COLLECTABLE_GC_STATE:
                  {
                    gc_state = ACQUIRED_GC_STATE;
                    pending_changes++;
                    break;
                  }
                case PENDING_COLLECTED_GC_STATE:
                  {
                    // Garbage collector is trying to eat it, save it!
                    gc_state = ACQUIRED_GC_STATE;
                    pending_changes++;
                    break;
                  }
                default:
                  assert(false);
              }
            }
          }
          else
          {
            if (min_gc_priority == LEGION_GC_NEVER_PRIORITY)
              remove_never_reference = true;
          }
        }
        min_gc_priority = priority;
        // Make an event for when the priority updates are done
        wait_on = priority_update_done;
        done_event = Runtime::create_rt_user_event();
        priority_update_done = done_event;
      }
      // If we make it here then we need to do the update
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      // Record the priority update
      const RtEvent updated = update_garbage_collection_priority(priority);
      LocalReferenceMutator mutator;
      if (updated.exists())
        mutator.record_reference_mutation_effect(updated);
      if (add_never_reference)
      {
        add_base_valid_ref(NEVER_GC_REF, &mutator);
        // Remove our pending change
        AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
        assert(pending_changes > 0);
        assert(gc_state == VALID_GC_STATE);
#endif
        pending_changes--;
      }
      if (remove_never_reference && 
          remove_base_valid_ref(NEVER_GC_REF, &mutator))
        assert(false); // should never end up deleting ourselves
      Runtime::trigger_event(done_event, mutator.get_done_event());
      return done_event;
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_garbage_collection_priority_update(
                   Runtime *runtime, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      GCPriority priority;
      derez.deserialize(priority);
      RtUserEvent done;
      derez.deserialize(done);

      PhysicalManager *manager = static_cast<PhysicalManager*>(
          runtime->weak_find_distributed_collectable(did));

      if (manager != NULL)
      {
        // To avoid collisiions with existing local mappers which could lead
        // to aliasing of priority updates, we use "invalid" processor IDs
        // here that will never conflict with existing processor IDs
        // Note that the NO_PROC is a valid processor ID for mappers in the
        // case where the mapper handles all the processors in a node. We
        // therefore always add the owner address space to the source to 
        // produce a non-zero processor ID. Note that this formulation also
        // avoid conflicts from different remote sources.
        const Processor fake_proc = { source + manager->owner_space };
#ifdef DEBUG_LEGION
        assert(fake_proc.id != 0);
#endif
        Runtime::trigger_event(done, manager->set_garbage_collection_priority(
                                  0/*default mapper ID*/, fake_proc, priority));
      }
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::register_active_context(InnerContext *context)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner()); // should always be on the owner node
#endif
      context->add_reference();
      AutoLock inst(inst_lock);
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
        AutoLock inst(inst_lock);
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
    PieceIteratorImpl* PhysicalManager::create_piece_iterator(
                                                 IndexSpaceNode *privilege_node)
    //--------------------------------------------------------------------------
    {
      return instance_domain->create_piece_iterator(piece_list, 
                              piece_list_size, privilege_node);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::defer_collect_user(CollectableView *view,
                                             ApEvent term_event,RtEvent collect,
                                             std::set<ApEvent> &to_collect,
                                             bool &add_ref, bool &remove_ref) 
    //--------------------------------------------------------------------------
    {
      AutoLock inst(inst_lock);
      CollectableInfo &info = gc_events[view]; 
      if (info.view_events.empty())
        add_ref = true;
      info.view_events.insert(term_event);
      info.events_added++;
      if (collect.exists())
        info.collect_event = collect;
      // Skip collections if there is a collection event guarding 
      // collection in the case of tracing
      if (info.collect_event.exists())
      {
        if (!info.collect_event.has_triggered())
          return;
        else
          info.collect_event = RtEvent::NO_RT_EVENT;
      }
      // Only do the pruning for every so many adds
      if (info.events_added >= runtime->gc_epoch_size)
      {
        for (std::set<ApEvent>::iterator it = info.view_events.begin();
              it != info.view_events.end(); /*nothing*/)
        {
          if (it->has_triggered_faultignorant())
          {
            to_collect.insert(*it);
            std::set<ApEvent>::iterator to_delete = it++;
            info.view_events.erase(to_delete);
          }
          else
            it++;
        }
        if (info.view_events.empty())
        {
          gc_events.erase(view);
          if (add_ref)
            add_ref = false;
          else
            remove_ref = true;
        }
        else // Reset the counter for the next time
          info.events_added = 0;
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::find_shutdown_preconditions(
                                               std::set<ApEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      AutoLock inst(inst_lock,1,false/*exclusive*/);
      for (std::map<CollectableView*,CollectableInfo>::const_iterator git =
            gc_events.begin(); git != gc_events.end(); git++)
      {
        // Make sure to test these for having triggered or risk a shutdown hang
        for (std::set<ApEvent>::const_iterator it = 
              git->second.view_events.begin(); it != 
              git->second.view_events.end(); it++)
          if (!it->has_triggered_faultignorant())
            preconditions.insert(*it);
      }
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::meets_regions(
      const std::vector<LogicalRegion> &regions, bool tight_region_bounds) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tree_id > 0); // only happens with VirtualManager
      assert(!regions.empty());
#endif
      std::set<IndexSpaceExpression*> region_exprs;
      for (std::vector<LogicalRegion>::const_iterator it = 
            regions.begin(); it != regions.end(); it++)
      {
        // If the region tree IDs don't match that is bad
        if (it->get_tree_id() != tree_id)
          return false;
        RegionNode *node = context->get_node(*it);
        region_exprs.insert(node->row_source);
      }
      IndexSpaceExpression *space_expr = (region_exprs.size() == 1) ?
        *(region_exprs.begin()) : context->union_index_spaces(region_exprs);
      return meets_expression(space_expr, tight_region_bounds);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::meets_expression(IndexSpaceExpression *space_expr,
                                           bool tight_bounds) const
    //--------------------------------------------------------------------------
    {
      return instance_domain->meets_layout_expression(space_expr, tight_bounds,
                                                  piece_list, piece_list_size);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::prune_gc_events(void)
    //--------------------------------------------------------------------------
    {
      // Must be holding the lock from caller
      // If we have any gc events then launch tasks to actually prune
      // off their references when they are done since we are now eligible
      // for collection by the garbage collector
      if (gc_events.empty())
        return;
      for (std::map<CollectableView*,CollectableInfo>::iterator it =
            gc_events.begin(); it != gc_events.end(); it++)
      {
        GarbageCollectionArgs args(it->first, new std::set<ApEvent>());
        RtEvent precondition = 
          Runtime::protect_merge_events(it->second.view_events);
        args.to_collect->swap(it->second.view_events);
        if (it->second.collect_event.exists() &&
            !it->second.collect_event.has_triggered())
          precondition = Runtime::merge_events(precondition, 
                                  it->second.collect_event);
        runtime->issue_runtime_meta_task(args, 
            LG_THROUGHPUT_WORK_PRIORITY, precondition);
      }
      gc_events.clear();
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
    /*static*/ ApEvent PhysicalManager::fetch_metadata(PhysicalInstance inst, 
                                                       ApEvent use_event)
    //--------------------------------------------------------------------------
    {
      ApEvent ready(inst.fetch_metadata(Processor::get_executing_processor()));
      if (!use_event.exists())
        return ready;
      if (!ready.exists())
        return use_event;
      return Runtime::merge_events(NULL, ready, use_event);
    }

    /////////////////////////////////////////////////////////////
    // IndividualManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndividualManager::IndividualManager(RegionTreeForest *ctx, 
                        DistributedID did, AddressSpaceID owner_space,
                        MemoryManager *memory, PhysicalInstance inst, 
                        IndexSpaceExpression *instance_domain,
                        const void *pl, size_t pl_size,
                        FieldSpaceNode *node, RegionTreeID tree_id,
                        LayoutDescription *desc, ReductionOpID redop_id, 
                        bool register_now, size_t footprint,
                        ApEvent u_event, bool external_instance,
                        const ReductionOp *op /*= NULL*/)
      : PhysicalManager(ctx, desc, encode_instance_did(did, external_instance,
            (redop_id != 0), false/*collective*/),
          owner_space, footprint, redop_id, (op != NULL) ? op : 
           (redop_id == 0) ? NULL : ctx->runtime->get_reduction(redop_id), node,
          instance_domain, pl, pl_size, tree_id, u_event, register_now),
        memory_manager(memory), instance(inst),
        use_event(fetch_metadata(inst, u_event))
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
                        LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, 
                        inst.id, memory->memory.id);
#endif
      if (runtime->legion_spy_enabled)
      {
#ifdef DEBUG_LEGION
        assert(unique_event.exists());
#endif
        LegionSpy::log_physical_instance(unique_event,inst.id,memory->memory.id,
         instance_domain->expr_id, field_space_node->handle, tree_id, redop);
        layout->log_instance_layout(unique_event);
      }
    }

    //--------------------------------------------------------------------------
    IndividualManager::~IndividualManager(void)
    //--------------------------------------------------------------------------
    {
      // Remote references removed by DistributedCollectable destructor
      if (!is_owner())
        memory_manager->unregister_remote_instance(this);
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        IndividualManager::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instance.exists());
#endif
      return LegionRuntime::Accessor::RegionAccessor<
	LegionRuntime::Accessor::AccessorType::Generic>(instance);
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        IndividualManager::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(instance.exists());
      assert(layout != NULL);
#endif
      const CopySrcDstField &info = layout->find_field_info(fid);
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic> temp(instance);
      return temp.get_untyped_field_accessor(info.field_id, info.size);
    }

    //--------------------------------------------------------------------------
    PointerConstraint IndividualManager::get_pointer_constraint(
                                                   const DomainPoint &key) const
    //--------------------------------------------------------------------------
    {
      if (use_event.exists() && !use_event.has_triggered_faultignorant())
        use_event.wait_faultignorant();
      void *inst_ptr = instance.pointer_untyped(0/*offset*/, 0/*elem size*/);
      return PointerConstraint(memory_manager->memory, uintptr_t(inst_ptr));
    }

    //--------------------------------------------------------------------------
    ApEvent IndividualManager::fill_from(FillView *fill_view,
                                         ApEvent precondition,
                                         PredEvent predicate_guard,
                                         IndexSpaceExpression *fill_expression,
                                         const FieldMask &fill_mask,
                                         const PhysicalTraceInfo &trace_info,
                                const FieldMaskSet<FillView> *tracing_srcs,
                                const FieldMaskSet<InstanceView> *tracing_dsts,
                                         std::set<RtEvent> &effects_applied,
                                         CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
      std::vector<CopySrcDstField> dst_fields;
      if (across_helper == NULL)
        compute_copy_offsets(fill_mask, dst_fields); 
      else
        across_helper->compute_across_offsets(fill_mask, dst_fields);
      const ApEvent result = fill_expression->issue_fill(trace_info, dst_fields, 
                                                 fill_view->value->value,
                                                 fill_view->value->value_size,
#ifdef LEGION_SPY
                                                 fill_view->fill_op_uid,
                                                 field_space_node->handle,
                                                 tree_id,
#endif
                                                 precondition, predicate_guard);
      if (trace_info.recording)
        trace_info.record_fill_views(result, fill_expression,
              *tracing_srcs, *tracing_dsts, effects_applied, (redop > 0));
      return result;
    }

    //--------------------------------------------------------------------------
    ApEvent IndividualManager::copy_from(PhysicalManager *source_manager,
                                         ApEvent precondition,
                                         PredEvent predicate_guard, 
                                         ReductionOpID reduction_op_id,
                                         IndexSpaceExpression *copy_expression,
                                         const FieldMask &copy_mask,
                                         const PhysicalTraceInfo &trace_info,
                                 const FieldMaskSet<InstanceView> *tracing_srcs,
                                 const FieldMaskSet<InstanceView> *tracing_dsts,
                                         std::set<RtEvent> &effects_applied,
                                         CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
      std::vector<CopySrcDstField> dst_fields, src_fields;
      if (across_helper == NULL)
        compute_copy_offsets(copy_mask, dst_fields);
      else
        across_helper->compute_across_offsets(copy_mask, dst_fields);
      source_manager->compute_copy_offsets(copy_mask, src_fields);
      if (reduction_op_id > 0) {
        for (unsigned idx = 0; idx < dst_fields.size(); idx++)
          dst_fields[idx].set_redop(reduction_op_id, false/*fold*/);
      }
      const std::vector<Reservation> no_reservations{};
      const ApEvent result = copy_expression->issue_copy(trace_info, dst_fields,
                                         src_fields, no_reservations,
#ifdef LEGION_SPY
                                         source_manager->tree_id, tree_id,
#endif
                                         precondition, predicate_guard); 
      if (trace_info.recording)
        trace_info.record_copy_views(result, copy_expression,
              *tracing_srcs, *tracing_dsts, effects_applied);
      return result;
    }

    //--------------------------------------------------------------------------
    void IndividualManager::compute_copy_offsets(const FieldMask &copy_mask,
                                           std::vector<CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(layout != NULL);
      assert(instance.exists());
#endif
      // Pass in our physical instance so the layout knows how to specialize
      layout->compute_copy_offsets(copy_mask, instance, 
#ifdef LEGION_SPY
                                   unique_event,
#endif
                                   fields);
    }

    //--------------------------------------------------------------------------
    void IndividualManager::initialize_across_helper(CopyAcrossHelper *helper,
                                                    const FieldMask &dst_mask,
                                     const std::vector<unsigned> &src_indexes,
                                     const std::vector<unsigned> &dst_indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(src_indexes.size() == dst_indexes.size());
#endif
      std::vector<CopySrcDstField> dst_fields;
      layout->compute_copy_offsets(dst_mask, instance, 
#ifdef LEGION_SPY
                                   unique_event,
#endif
                                   dst_fields);
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
    void IndividualManager::send_manager(AddressSpaceID target)
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
        rez.serialize(instance_footprint);
        // No need for a reference here since we know we'll continue holding it
        instance_domain->pack_expression(rez, target);
        rez.serialize(piece_list_size);
        if (piece_list_size > 0)
          rez.serialize(piece_list, piece_list_size);
        rez.serialize(field_space_node->handle);
        rez.serialize(tree_id);
        rez.serialize(unique_event);
        layout->pack_layout_description(rez, target);
        rez.serialize(redop);
        pack_garbage_collection_state(rez, target);
      }
      context->runtime->send_instance_manager(target, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualManager::handle_send_manager(Runtime *runtime, 
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
      size_t inst_footprint;
      derez.deserialize(inst_footprint);
      PendingRemoteExpression pending;
      RtEvent domain_ready;
      IndexSpaceExpression *inst_domain = 
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source,
                                                pending, domain_ready);
      size_t piece_list_size;
      derez.deserialize(piece_list_size);
      void *piece_list = NULL;
      if (piece_list_size > 0)
      {
        piece_list = malloc(piece_list_size);
        derez.deserialize(piece_list, piece_list_size);
      }
      FieldSpace handle;
      derez.deserialize(handle);
      RtEvent fs_ready;
      FieldSpaceNode *space_node = runtime->forest->get_node(handle, &fs_ready);
      RegionTreeID tree_id;
      derez.deserialize(tree_id);
      ApEvent unique_event;
      derez.deserialize(unique_event);
      LayoutConstraintID layout_id;
      derez.deserialize(layout_id);
      RtEvent layout_ready;
      LayoutConstraints *constraints = 
        runtime->find_layout_constraints(layout_id, 
                    false/*can fail*/, &layout_ready);
      ReductionOpID redop;
      derez.deserialize(redop);
      GarbageCollectionState gc_state;
      derez.deserialize(gc_state);

      if (domain_ready.exists() || fs_ready.exists() || layout_ready.exists())
      {
        const RtEvent precondition = 
          Runtime::merge_events(domain_ready, fs_ready, layout_ready);
        if (precondition.exists() && !precondition.has_triggered())
        {
          // We need to defer this instance creation
          DeferIndividualManagerArgs args(did, owner_space, mem, inst,
              inst_footprint, inst_domain, pending, 
              handle, tree_id, layout_id, unique_event, redop, 
              piece_list, piece_list_size, source, gc_state);
          runtime->issue_runtime_meta_task(args,
              LG_LATENCY_RESPONSE_PRIORITY, precondition);
          return;
        }
        // If we fall through we need to refetch things that we didn't get
        if (domain_ready.exists())
          inst_domain = runtime->forest->find_remote_expression(pending);
        if (fs_ready.exists())
          space_node = runtime->forest->get_node(handle);
        if (layout_ready.exists())
          constraints = 
            runtime->find_layout_constraints(layout_id, false/*can fail*/);
      }
      // If we fall through here we can create the manager now
      create_remote_manager(runtime, did, owner_space, mem, inst,inst_footprint,
                            inst_domain, piece_list, piece_list_size, 
                            space_node, tree_id, constraints, unique_event,
                            redop, gc_state);
    }

    //--------------------------------------------------------------------------
    IndividualManager::DeferIndividualManagerArgs::DeferIndividualManagerArgs(
            DistributedID d, AddressSpaceID own, Memory m, PhysicalInstance i, 
            size_t f, IndexSpaceExpression *lx,
            const PendingRemoteExpression &p, FieldSpace h, RegionTreeID tid,
            LayoutConstraintID l, ApEvent u, ReductionOpID r, const void *pl, 
            size_t pl_size, AddressSpaceID src, GarbageCollectionState gc)
      : LgTaskArgs<DeferIndividualManagerArgs>(implicit_provenance),
            did(d), owner(own), mem(m), inst(i), footprint(f), pending(p),
            local_expr(lx), handle(h), tree_id(tid),
            layout_id(l), use_event(u), redop(r), piece_list(pl),
            piece_list_size(pl_size), source(src), state(gc)
    //--------------------------------------------------------------------------
    {
      if (local_expr != NULL)
        local_expr->add_base_expression_reference(META_TASK_REF);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualManager::handle_defer_manager(const void *args,
                                                            Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const DeferIndividualManagerArgs *dargs = 
        (const DeferIndividualManagerArgs*)args; 
      IndexSpaceExpression *inst_domain = dargs->local_expr;
      if (inst_domain == NULL)
        inst_domain = runtime->forest->find_remote_expression(dargs->pending);
      FieldSpaceNode *space_node = runtime->forest->get_node(dargs->handle);
      LayoutConstraints *constraints = 
        runtime->find_layout_constraints(dargs->layout_id);
      create_remote_manager(runtime, dargs->did, dargs->owner, dargs->mem,
          dargs->inst, dargs->footprint, inst_domain, dargs->piece_list,
          dargs->piece_list_size, space_node, dargs->tree_id, constraints, 
          dargs->use_event, dargs->redop, dargs->state);
      // Remove the local expression reference if necessary
      if ((dargs->local_expr != NULL) &&
          dargs->local_expr->remove_base_expression_reference(META_TASK_REF))
        delete dargs->local_expr;
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualManager::create_remote_manager(Runtime *runtime, 
          DistributedID did, AddressSpaceID owner_space, Memory mem, 
          PhysicalInstance inst, size_t inst_footprint, 
          IndexSpaceExpression *inst_domain, const void *piece_list,
          size_t piece_list_size, FieldSpaceNode *space_node, 
          RegionTreeID tree_id, LayoutConstraints *constraints, 
          ApEvent use_event, ReductionOpID redop, GarbageCollectionState state)
    //--------------------------------------------------------------------------
    {
      LayoutDescription *layout = 
        LayoutDescription::handle_unpack_layout_description(constraints,
                                space_node, inst_domain->get_num_dims());
      MemoryManager *memory = runtime->find_memory_manager(mem);
      const ReductionOp *op = 
        (redop == 0) ? NULL : runtime->get_reduction(redop);
      void *location;
      IndividualManager *man = NULL;
      const bool external_instance = InstanceManager::is_external_did(did);
      if (runtime->find_pending_collectable_location(did, location))
        man = new(location) IndividualManager(runtime->forest, did, owner_space,
                                              memory, inst, inst_domain, 
                                              piece_list, piece_list_size, 
                                              space_node, tree_id, layout, 
                                              redop, false/*reg now*/, 
                                              inst_footprint, use_event, 
                                              external_instance, op);
      else
        man = new IndividualManager(runtime->forest, did, owner_space, memory, 
                              inst, inst_domain, piece_list, piece_list_size,
                              space_node, tree_id, layout, redop, 
                              false/*reg now*/, inst_footprint, use_event, 
                              external_instance, op);
      man->initialize_remote_gc_state(state);
      // Hold-off doing the registration until construction is complete
      man->register_with_runtime(NULL/*no remote registration needed*/);
    }

    //--------------------------------------------------------------------------
    void IndividualManager::get_instance_pointers(Memory memory,
                                         std::vector<uintptr_t> &pointers) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(memory == instance.get_location());
#endif
      void *inst_ptr = instance.pointer_untyped(0/*offset*/, 0/*elem size*/);
      pointers.push_back(uintptr_t(inst_ptr));
    }

    //--------------------------------------------------------------------------
    RtEvent IndividualManager::perform_deletion(AddressSpaceID source,
                                                AutoLock *i_lock /* = NULL*/)
    //--------------------------------------------------------------------------
    {
      if (i_lock == NULL)
      {
        AutoLock instance_lock(inst_lock);
        return perform_deletion(source, &instance_lock);
      }
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(source == local_space);
      assert(!deferred_deletion.exists());
#endif
      log_garbage.spew("Deleting physical instance " IDFMT " in memory " 
                       IDFMT "", instance.id, memory_manager->memory.id);
      prune_gc_events();
      // Grab the set of active contexts to notify
      std::set<InnerContext*> to_notify;
      to_notify.swap(active_contexts);
#ifndef DISABLE_GC
      // If we're still active that means there are still outstanding
      // users so make an event for when we are done, not we're holding
      // the instance lock when this is called
      if (currently_active)
        deferred_deletion = Runtime::create_rt_user_event();
      // Now we can release the lock since we're done with the atomic updates
      i_lock->release();
      std::vector<PhysicalInstance::DestroyedField> serdez_fields;
      layout->compute_destroyed_fields(serdez_fields); 
      if (!serdez_fields.empty())
        instance.destroy(serdez_fields, deferred_deletion);
      else
        instance.destroy(deferred_deletion);
#ifdef LEGION_MALLOC_INSTANCES
      if (!is_external_instance())
        memory_manager->free_legion_instance(this, deferred_deletion);
#endif
#else
      // Release the i_lock since we're done with the atomic updates
      i_lock->release();
#endif
      // Notify any contexts of our deletion
      if (!to_notify.empty())
      {
        for (std::set<InnerContext*>::const_iterator it =
              to_notify.begin(); it != to_notify.end(); it++)
        {
          (*it)->notify_instance_deletion(this);
          if ((*it)->remove_reference())
            delete (*it);
        }
      }
      // We issued the deletion to Realm so all our effects are done
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void IndividualManager::force_deletion(void)
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
#ifdef LEGION_MALLOC_INSTANCES
      if (!is_external_instance())
        memory_manager->free_legion_instance(this, RtEvent::NO_RT_EVENT);
#endif
#endif
    }

    //--------------------------------------------------------------------------
    RtEvent IndividualManager::update_garbage_collection_priority(
                                                            GCPriority priority)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
      {
        const RtUserEvent done = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(priority);
          rez.serialize(done);
        }
        runtime->send_gc_priority_update(owner_space, rez);
        return done;
      }
      else
      {
        memory_manager->set_garbage_collection_priority(this, priority);
        return RtEvent::NO_RT_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    RtEvent IndividualManager::attach_external_instance(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_external_instance());
#endif
      return memory_manager->attach_external_instance(this);
    }

    //--------------------------------------------------------------------------
    RtEvent IndividualManager::detach_external_instance(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_external_instance());
#endif
      return memory_manager->detach_external_instance(this);
    }

    //--------------------------------------------------------------------------
    bool IndividualManager::has_visible_from(const std::set<Memory> &mems) const
    //--------------------------------------------------------------------------
    {
      return (mems.find(memory_manager->memory) != mems.end());
    }

    //--------------------------------------------------------------------------
    Memory IndividualManager::get_memory(void) const
    //--------------------------------------------------------------------------
    {
      return memory_manager->memory;
    }

    /////////////////////////////////////////////////////////////
    // Collective Manager
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CollectiveManager::CollectiveManager(RegionTreeForest *ctx, 
                                DistributedID did, AddressSpaceID owner_space,
                                IndexSpaceNode *points,
                                IndexSpaceExpression *instance_domain,
                                const void *pl, size_t pl_size,
                                FieldSpaceNode *node, RegionTreeID tree_id,
                                LayoutDescription *desc, ReductionOpID redop_id,
                                bool register_now, size_t footprint,
                                ApEvent u_event, bool external_instance)
      : PhysicalManager(ctx, desc, encode_instance_did(did, external_instance,
            (redop_id != 0), true/*collective*/),
          owner_space, footprint, redop_id, (redop_id == 0) ? NULL : 
            ctx->runtime->get_reduction(redop_id),
          node, instance_domain, pl, pl_size, tree_id, u_event, register_now), 
        point_space(points), finalize_messages(0), deleted_or_detached(false)
    //--------------------------------------------------------------------------
    {
      point_space->add_nested_valid_ref(did);
#if 0
#ifdef DEBUG_LEGION
      if (is_owner())
        assert(left == local_space);
      else
        assert(left != local_space);
#endif
      if (!right_spaces.empty())
        add_nested_resource_ref(did);
#endif
#ifdef LEGION_GC
      log_garbage.info("GC Collective Manager %lld %d",
                        LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space); 
#endif
    }

    //--------------------------------------------------------------------------
    CollectiveManager::~CollectiveManager(void)
    //--------------------------------------------------------------------------
    {
      if (point_space->remove_nested_valid_ref(did))
        delete point_space;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::finalize_collective_instance(ApUserEvent inst_event)
    //--------------------------------------------------------------------------
    {
      // TODO: implement this
      assert(false);
    }

    //--------------------------------------------------------------------------
    ApEvent CollectiveManager::get_use_event(void) const
    //--------------------------------------------------------------------------
    {
      // TODO: implement this
      assert(false);
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    PhysicalInstance CollectiveManager::get_instance(const DomainPoint &k) const
    //--------------------------------------------------------------------------
    {
      // TODO: implement this
      assert(false);
      return PhysicalInstance::NO_INST;
    }

    //--------------------------------------------------------------------------
    PointerConstraint CollectiveManager::get_pointer_constraint(
                                                   const DomainPoint &key) const
    //--------------------------------------------------------------------------
    {
      // TODO: implement this
      assert(false);
      return PointerConstraint();
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        CollectiveManager::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
      // not supported
      assert(false);
      return LegionRuntime::Accessor::RegionAccessor<
	LegionRuntime::Accessor::AccessorType::Generic>(instances[0]);
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        CollectiveManager::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      // not supported
      assert(false);
      const CopySrcDstField &info = layout->find_field_info(fid);
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic> temp(instances[0]);
      return temp.get_untyped_field_accessor(info.field_id, info.size);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::get_instance_pointers(Memory memory,
                                         std::vector<uintptr_t> &pointers) const
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < memories.size(); idx++)
      {
        if (memories[idx]->memory != memory)
          continue;
        void *ptr = instances[idx].pointer_untyped(0/*offset*/, 0/*elem size*/);
        pointers.push_back(uintptr_t(ptr));
      }
#ifdef DEBUG_LEGION
      assert(!pointers.empty());
#endif
    }

    //--------------------------------------------------------------------------
    RtEvent CollectiveManager::perform_deletion(AddressSpaceID source, 
                                                AutoLock *i_lock /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      // TODO: implement this
      assert(false);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::force_deletion(void)
    //--------------------------------------------------------------------------
    {
      force_delete(true/*left*/);
    }

    //--------------------------------------------------------------------------
    RtEvent CollectiveManager::update_garbage_collection_priority(
                                                            GCPriority priority)
    //--------------------------------------------------------------------------
    {
      // TODO: implement this
      assert(false);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent CollectiveManager::attach_external_instance(void)
    //--------------------------------------------------------------------------
    {
      // TODO: implement this
      assert(false);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent CollectiveManager::detach_external_instance(void)
    //--------------------------------------------------------------------------
    {
      const RtUserEvent result = Runtime::create_rt_user_event(); 
      detach_external(result, true/*left*/);
      return result;
    }

    //--------------------------------------------------------------------------
    bool CollectiveManager::has_visible_from(const std::set<Memory> &mems) const
    //--------------------------------------------------------------------------
    {
      for (std::vector<MemoryManager*>::const_iterator it = 
            memories.begin(); it != memories.end(); it++)
        if (mems.find((*it)->memory) != mems.end())
          return true;
      return false;
    }

    //--------------------------------------------------------------------------
    Memory CollectiveManager::get_memory(void) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return Memory::NO_MEMORY;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::perform_delete(RtEvent deferred_event, bool left)
    //--------------------------------------------------------------------------
    {
      if (left)
      {
        if (local_space == left_space)
        {
          // See if we can do the deletion
          {
            AutoLock i_lock(inst_lock);
            if (deleted_or_detached)
              return;
            deleted_or_detached = true;
          }
          // If we make it here we are the first ones so so the deletion
          collective_deletion(deferred_event);
          for (std::vector<AddressSpaceID>::const_iterator it = 
                right_spaces.begin(); it != right_spaces.end(); it++)
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(PERFORM_DELETE_MESSAGE);
              rez.serialize(deferred_event);
              rez.serialize<bool>(false/*left*/);
            }
            runtime->send_collective_instance_message(*it, rez);
          }
        }
        else
        {
          // Forward this on to the left space
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(PERFORM_DELETE_MESSAGE);
            rez.serialize(deferred_event);
            rez.serialize<bool>(true/*left*/);
          }
          AutoLock i_lock(inst_lock);
          if (!deleted_or_detached)
            runtime->send_collective_instance_message(left_space, rez);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(local_space != left_space);
#endif
        // Update our local event
        {
          AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
          assert(!deleted_or_detached);
#endif
          deleted_or_detached = true;
        }
        // Do the deletion
        collective_deletion(deferred_event); 
        // If we have no right users send back messages
        if (right_spaces.empty())
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(FINALIZE_MESSAGE);
          }
          runtime->send_collective_instance_message(left_space, rez);
        }
        else
        {
          for (std::vector<AddressSpaceID>::const_iterator it = 
                right_spaces.begin(); it != right_spaces.end(); it++)
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(PERFORM_DELETE_MESSAGE);
              rez.serialize(deferred_event);
              rez.serialize(false/*left*/);
            }
            runtime->send_collective_instance_message(*it, rez);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::force_delete(bool left)
    //--------------------------------------------------------------------------
    {
      if (left)
      {
        if (local_space == left_space)
        {
          // See if we can do the deletion
          {
            AutoLock i_lock(inst_lock);
            if (deleted_or_detached)
              return;
            deleted_or_detached = true;
          }
          // If we make it here we are the first ones so so the deletion
          collective_force();
          for (std::vector<AddressSpaceID>::const_iterator it = 
                right_spaces.begin(); it != right_spaces.end(); it++)
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(FORCE_DELETE_MESSAGE);
              rez.serialize<bool>(false/*left*/);
            }
            runtime->send_collective_instance_message(*it, rez);
          }
        }
        else
        {
          // Forward this on to the left space
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(FORCE_DELETE_MESSAGE);
            rez.serialize<bool>(true/*left*/);
          }
          AutoLock i_lock(inst_lock);
          if (!deleted_or_detached)
            runtime->send_collective_instance_message(left_space, rez);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(local_space != left_space);
#endif
        // Update our local event
        {
          AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
          assert(!deleted_or_detached);
#endif
          deleted_or_detached = true;
        }
        // Do the deletion
        collective_force(); 
        // If we have no right users send back messages
        if (right_spaces.empty())
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(FINALIZE_MESSAGE);
          }
          runtime->send_collective_instance_message(left_space, rez);
        }
        else
        {
          for (std::vector<AddressSpaceID>::const_iterator it = 
                right_spaces.begin(); it != right_spaces.end(); it++)
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(FORCE_DELETE_MESSAGE);
              rez.serialize(false/*left*/);
            }
            runtime->send_collective_instance_message(*it, rez);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::detach_external(RtUserEvent to_trigger, 
                                            bool left, RtEvent full_detach)
    //--------------------------------------------------------------------------
    {
      if (left)
      {
        if (local_space == left_space)
        {
          // See if we can do the deletion
          {
            AutoLock i_lock(inst_lock);
            if (deleted_or_detached)
            {
#ifdef DEBUG_LEGION
              assert(detached.exists());
#endif
              Runtime::trigger_event(to_trigger, detached);
              return;
            }
            deleted_or_detached = true;
            detached = to_trigger;
          }
          // If we make it here we are the first ones so so the deletion
          std::set<RtEvent> preconditions;
          collective_detach(preconditions);
          for (std::vector<AddressSpaceID>::const_iterator it = 
                right_spaces.begin(); it != right_spaces.end(); it++)
          {
            const RtUserEvent right_event = Runtime::create_rt_user_event();
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(DETACH_EXTERNAL_MESSAGE);
              rez.serialize(right_event);
              rez.serialize<bool>(false/*left*/);
              rez.serialize(to_trigger);
            }
            runtime->send_collective_instance_message(*it, rez);
            preconditions.insert(right_event);
          }
          if (!preconditions.empty())
            Runtime::trigger_event(to_trigger, 
                Runtime::merge_events(preconditions));
          else
            Runtime::trigger_event(to_trigger);
        }
        else
        {
          // Forward this on to the left space
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(DETACH_EXTERNAL_MESSAGE);
            rez.serialize(to_trigger);
            rez.serialize<bool>(true/*left*/);
          }
          AutoLock i_lock(inst_lock);
          if (deleted_or_detached)
          {
#ifdef DEBUG_LEGION
            assert(detached.exists());
#endif
            Runtime::trigger_event(to_trigger, detached);
          }
          else
            runtime->send_collective_instance_message(left_space, rez);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(local_space != left_space);
#endif
        // Update our local event
        {
          AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
          assert(!deleted_or_detached);
          assert(!detached.exists());
#endif
          deleted_or_detached = true;
          detached = full_detach;
        }
        // Do the deletion
        std::set<RtEvent> preconditions;
        collective_detach(preconditions);
        // If we have no right users send back messages
        if (right_spaces.empty())
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(FINALIZE_MESSAGE);
          }
          runtime->send_collective_instance_message(left_space, rez);
        }
        else
        {
          for (std::vector<AddressSpaceID>::const_iterator it = 
                right_spaces.begin(); it != right_spaces.end(); it++)
          {
            const RtUserEvent right_event = Runtime::create_rt_user_event();
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(DETACH_EXTERNAL_MESSAGE);
              rez.serialize(right_event);
              rez.serialize(false/*left*/);
              rez.serialize(full_detach);
            }
            runtime->send_collective_instance_message(*it, rez);
            preconditions.insert(right_event);
          }
        }
        if (!preconditions.empty())
            Runtime::trigger_event(to_trigger, 
                Runtime::merge_events(preconditions));
          else
            Runtime::trigger_event(to_trigger);
      }
    }

    //--------------------------------------------------------------------------
    bool CollectiveManager::finalize_message(void)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
      assert(finalize_messages < right_spaces.size());
#endif
      if (++finalize_messages == right_spaces.size())
      {
        if (left_space != local_space)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(FINALIZE_MESSAGE);
          }
          runtime->send_collective_instance_message(left_space, rez);
        }
        return true;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::collective_deletion(RtEvent deferred_event)
    //--------------------------------------------------------------------------
    {
#ifndef DISABLE_GC
      std::vector<PhysicalInstance::DestroyedField> serdez_fields;
      layout->compute_destroyed_fields(serdez_fields); 
      if (!serdez_fields.empty())
      {
        for (std::vector<PhysicalInstance>::const_iterator it = 
              instances.begin(); it != instances.end(); it++)
          it->destroy(serdez_fields, deferred_event);
      }
      else
      {
        for (std::vector<PhysicalInstance>::const_iterator it = 
              instances.begin(); it != instances.end(); it++)
          it->destroy(deferred_event);
      }
#endif
      // Notify any contexts of our deletion
      // Grab a copy of this in case we get any removal calls
      // while we are doing the deletion. We know that there
      // will be no more additions because we are being deleted
      std::set<InnerContext*> copy_active_contexts;
      {
        AutoLock inst(inst_lock);
        if (active_contexts.empty())
          return;
        copy_active_contexts = active_contexts;
        active_contexts.clear();
      }
      for (std::set<InnerContext*>::iterator it = copy_active_contexts.begin(); 
            it != copy_active_contexts.end(); it++)
      {
        (*it)->notify_instance_deletion(this);
        if ((*it)->remove_reference())
          delete (*it);
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::collective_force(void)
    //--------------------------------------------------------------------------
    {
#ifndef DISABLE_GC
      std::vector<PhysicalInstance::DestroyedField> serdez_fields;
      layout->compute_destroyed_fields(serdez_fields); 
      if (!serdez_fields.empty())
      {
        for (std::vector<PhysicalInstance>::const_iterator it = 
              instances.begin(); it != instances.end(); it++)
          it->destroy(serdez_fields);
      }
      else
      {
        for (std::vector<PhysicalInstance>::const_iterator it = 
              instances.begin(); it != instances.end(); it++)
          it->destroy();
      }
#endif
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::collective_detach(std::set<RtEvent> &detach_events)
    //--------------------------------------------------------------------------
    {
      for (std::vector<MemoryManager*>::const_iterator it = 
            memories.begin(); it != memories.end(); it++)
      {
        const RtEvent detach = (*it)->detach_external_instance(this);
        if (detach.exists())
          detach_events.insert(detach);
      }
    }

    //--------------------------------------------------------------------------
    ApEvent CollectiveManager::fill_from(FillView *fill_view,
                                ApEvent precondition, PredEvent predicate_guard,
                                IndexSpaceExpression *expression,
                                const FieldMask &fill_mask,
                                const PhysicalTraceInfo &trace_info,
                                const FieldMaskSet<FillView> *tracing_srcs,
                                const FieldMaskSet<InstanceView> *tracing_dsts,
                                std::set<RtEvent> &effects_applied,
                                CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
      // TODO: implement this
      assert(false);
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent CollectiveManager::copy_from(PhysicalManager *manager, 
                                ApEvent precondition,
                                PredEvent predicate_guard, ReductionOpID redop,
                                IndexSpaceExpression *expression,
                                const FieldMask &copy_mask,
                                const PhysicalTraceInfo &trace_info,
                                const FieldMaskSet<InstanceView> *tracing_srcs,
                                const FieldMaskSet<InstanceView> *tracing_dsts,
                                std::set<RtEvent> &effects_applied,
                                CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
      // TODO: implement this
      assert(false);
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::compute_copy_offsets(const FieldMask &mask,
                                           std::vector<CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::send_manager(AddressSpaceID target)
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
        rez.serialize(point_space->handle);
        rez.serialize(instance_footprint);
        // No need for a reference here since we know we'll continue holding it
        instance_domain->pack_expression(rez, target);
        rez.serialize(field_space_node->handle);
        rez.serialize(tree_id);
        rez.serialize(redop);
        rez.serialize(unique_event);
        layout->pack_layout_description(rez, target);
        pack_garbage_collection_state(rez, target);
      }
      context->runtime->send_collective_instance_manager(target, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_send_manager(Runtime *runtime, 
                                     AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID owner_space;
      derez.deserialize(owner_space);
      IndexSpace points_handle;
      derez.deserialize(points_handle);
      RtEvent points_ready;
      IndexSpaceNode *point_space = 
        runtime->forest->get_node(points_handle, &points_ready); 
      size_t inst_footprint;
      derez.deserialize(inst_footprint);
      PendingRemoteExpression pending;
      RtEvent domain_ready;
      IndexSpaceExpression *inst_domain = 
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source,
                                                pending, domain_ready);
      size_t piece_list_size;
      derez.deserialize(piece_list_size);
      void *piece_list = NULL;
      if (piece_list_size > 0)
      {
        piece_list = malloc(piece_list_size);
        derez.deserialize(piece_list, piece_list_size);
      }
      FieldSpace handle;
      derez.deserialize(handle);
      RtEvent fs_ready;
      FieldSpaceNode *space_node = runtime->forest->get_node(handle, &fs_ready);
      RegionTreeID tree_id;
      derez.deserialize(tree_id);
      ReductionOpID redop;
      derez.deserialize(redop);
      ApEvent unique_event;
      derez.deserialize(unique_event);
      LayoutConstraintID layout_id;
      derez.deserialize(layout_id);
      RtEvent layout_ready;
      LayoutConstraints *constraints = 
        runtime->find_layout_constraints(layout_id, 
                    false/*can fail*/, &layout_ready);
      GarbageCollectionState state;
      derez.deserialize(state);
      if (points_ready.exists() || domain_ready.exists() || 
          fs_ready.exists() || layout_ready.exists())
      {
        std::set<RtEvent> preconditions;
        if (points_ready.exists())
          preconditions.insert(points_ready);
        if (domain_ready.exists())
          preconditions.insert(domain_ready);
        if (fs_ready.exists())
          preconditions.insert(fs_ready);
        if (layout_ready.exists())
          preconditions.insert(layout_ready);
        const RtEvent precondition = Runtime::merge_events(preconditions);
        if (precondition.exists() && !precondition.has_triggered())
        {
          // We need to defer this instance creation
          DeferCollectiveManagerArgs args(did, owner_space, points_handle, 
              inst_footprint, inst_domain, pending, handle, tree_id, layout_id,
              unique_event, redop, piece_list, piece_list_size, source, state);
          runtime->issue_runtime_meta_task(args,
              LG_LATENCY_RESPONSE_PRIORITY, precondition);
          return;
        }
        // If we fall through we need to refetch things that we didn't get
        if (points_ready.exists())
          point_space = runtime->forest->get_node(points_handle);
        if (domain_ready.exists())
          inst_domain = runtime->forest->find_remote_expression(pending);
        if (fs_ready.exists())
          space_node = runtime->forest->get_node(handle);
        if (layout_ready.exists())
          constraints = 
            runtime->find_layout_constraints(layout_id, false/*can fail*/);
      }
      // If we fall through here we can create the manager now
      create_collective_manager(runtime, did, owner_space, point_space,
          inst_footprint, inst_domain, piece_list, piece_list_size, space_node,
          tree_id, constraints, unique_event, redop, state);
    }

    //--------------------------------------------------------------------------
    CollectiveManager::DeferCollectiveManagerArgs::DeferCollectiveManagerArgs(
            DistributedID d, AddressSpaceID own, IndexSpace points, 
            size_t f, IndexSpaceExpression *lx,
            const PendingRemoteExpression &p, FieldSpace h, RegionTreeID tid,
            LayoutConstraintID l, ApEvent use, ReductionOpID r,
            const void *pl, size_t pl_size, AddressSpace src,
            GarbageCollectionState gc)
      : LgTaskArgs<DeferCollectiveManagerArgs>(implicit_provenance),
        did(d), owner(own), point_space(points), footprint(f), local_expr(lx),
        pending(p), handle(h), tree_id(tid), layout_id(l), use_event(use),
        redop(r), piece_list(pl), piece_list_size(pl_size),source(src),state(gc)
    //--------------------------------------------------------------------------
    {
      if (local_expr != NULL)
        local_expr->add_base_expression_reference(META_TASK_REF);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_defer_manager(const void *args,
                                                            Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const DeferCollectiveManagerArgs *dargs = 
        (const DeferCollectiveManagerArgs*)args; 
      IndexSpaceNode *point_space = 
        runtime->forest->get_node(dargs->point_space);
      IndexSpaceExpression *inst_domain = dargs->local_expr;
      if (inst_domain == NULL)
        inst_domain = runtime->forest->find_remote_expression(dargs->pending);
      FieldSpaceNode *space_node = runtime->forest->get_node(dargs->handle);
      LayoutConstraints *constraints = 
        runtime->find_layout_constraints(dargs->layout_id);
      create_collective_manager(runtime, dargs->did, dargs->owner, point_space,
          dargs->footprint, inst_domain, dargs->piece_list,
          dargs->piece_list_size, space_node, dargs->tree_id, constraints, 
          dargs->use_event, dargs->redop, dargs->state);
      // Remove the local expression reference if necessary
      if ((dargs->local_expr != NULL) &&
          dargs->local_expr->remove_base_expression_reference(META_TASK_REF))
        delete dargs->local_expr;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::create_collective_manager(
          Runtime *runtime, DistributedID did, AddressSpaceID owner_space, 
          IndexSpaceNode *point_space, size_t inst_footprint, 
          IndexSpaceExpression *inst_domain, const void *piece_list,
          size_t piece_list_size, FieldSpaceNode *space_node, 
          RegionTreeID tree_id,LayoutConstraints *constraints,
          ApEvent use_event, ReductionOpID redop, GarbageCollectionState state)
    //--------------------------------------------------------------------------
    {
      LayoutDescription *layout = 
        LayoutDescription::handle_unpack_layout_description(constraints,
                                space_node, inst_domain->get_num_dims());
      void *location;
      CollectiveManager *man = NULL;
      const bool external_instance = PhysicalManager::is_external_did(did);
      if (runtime->find_pending_collectable_location(did, location))
        man = new(location) CollectiveManager(runtime->forest, did,
                                            owner_space, point_space, 
                                            inst_domain, piece_list, 
                                            piece_list_size, space_node,tree_id,
                                            layout, redop, false/*reg now*/, 
                                            inst_footprint, use_event, 
                                            external_instance); 
      else
        man = new CollectiveManager(runtime->forest, did, owner_space, 
                                  point_space, inst_domain, piece_list, 
                                  piece_list_size, space_node, tree_id, layout,
                                  redop, false/*reg now*/, inst_footprint, 
                                  use_event, external_instance);
      man->initialize_remote_gc_state(state);
      // Hold-off doing the registration until construction is complete
      man->register_with_runtime(NULL/*no remote registration needed*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_collective_message(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      CollectiveManager *manager = static_cast<CollectiveManager*>(
                          runtime->find_distributed_collectable(did));
      MessageKind kind;
      derez.deserialize(kind);
      switch (kind)
      {
        case PERFORM_DELETE_MESSAGE:
          {
            RtEvent deferred_event;
            derez.deserialize(deferred_event);
            bool left;
            derez.deserialize(left);
            manager->perform_delete(deferred_event, left);
            break;
          }
        case FORCE_DELETE_MESSAGE:
          {
            bool left;
            derez.deserialize(left);
            manager->force_delete(left);
            break;
          }
        case DETACH_EXTERNAL_MESSAGE:
          {
            RtUserEvent to_trigger;
            derez.deserialize(to_trigger);
            bool left;
            derez.deserialize(left);
            manager->detach_external(to_trigger, left);
            break;
          }
        case FINALIZE_MESSAGE:
          {
            if (manager->finalize_message() &&
                manager->remove_nested_resource_ref(did))
              delete manager;
            break;
          }
        default:
          assert(false);
      }
    }

    /////////////////////////////////////////////////////////////
    // Virtual Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VirtualManager::VirtualManager(Runtime *runtime, DistributedID did,
                                   LayoutDescription *desc)
      : InstanceManager(runtime->forest, runtime->address_space, did, desc,
                        NULL/*field space node*/,NULL/*index space expression*/,
                        0/*tree id*/, true/*register now*/)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info("GC Virtual Manager %lld %d",
                        LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space); 
#endif
    }

    //--------------------------------------------------------------------------
    VirtualManager::VirtualManager(const VirtualManager &rhs)
      : InstanceManager(NULL, 0, 0, NULL, NULL, NULL, 0, false)
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
      return LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
	(PhysicalInstance::NO_INST);
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          VirtualManager::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
	(PhysicalInstance::NO_INST);
    }

    //--------------------------------------------------------------------------
    void VirtualManager::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void VirtualManager::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void VirtualManager::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void VirtualManager::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ApEvent VirtualManager::get_use_event(void) const
    //--------------------------------------------------------------------------
    {
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent VirtualManager::get_unique_event(void) const
    //--------------------------------------------------------------------------
    {
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    PhysicalInstance VirtualManager::get_instance(const DomainPoint &p) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return PhysicalInstance::NO_INST;
    }

    //--------------------------------------------------------------------------
    PointerConstraint VirtualManager::get_pointer_constraint(
                                                   const DomainPoint &key) const
    //--------------------------------------------------------------------------
    {
      return PointerConstraint(Memory::NO_MEMORY, 0);
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
    InstanceBuilder::InstanceBuilder(const std::vector<LogicalRegion> &regs,
                      IndexSpaceExpression *expr, FieldSpaceNode *node, 
                      RegionTreeID tid, const LayoutConstraintSet &cons, 
                      Runtime *rt, MemoryManager *memory, UniqueID cid,
                      const void *pl, size_t pl_size)
      : regions(regs), constraints(cons), runtime(rt), memory_manager(memory),
        creator_id(cid), instance(PhysicalInstance::NO_INST), 
        field_space_node(node), instance_domain(expr), tree_id(tid), 
        redop_id(0), reduction_op(NULL), realm_layout(NULL), piece_list(NULL),
        piece_list_size(0), valid(true)
    //--------------------------------------------------------------------------
    {
      if (pl != NULL)
      {
        piece_list_size = pl_size;
        piece_list = malloc(piece_list_size);
        memcpy(piece_list, pl, piece_list_size);
      }
      compute_layout_parameters();
    }

    //--------------------------------------------------------------------------
    InstanceBuilder::~InstanceBuilder(void)
    //--------------------------------------------------------------------------
    {
      if (realm_layout != NULL)
        delete realm_layout;
      if (piece_list != NULL)
        free(piece_list);
    }

    //--------------------------------------------------------------------------
    PhysicalManager* InstanceBuilder::create_physical_instance(
                RegionTreeForest *forest, CollectiveManager *collective_inst,
                DomainPoint *collective_point, LayoutConstraintKind *unsat_kind,
                unsigned *unsat_index, size_t *footprint, RtEvent precondition)
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
        if (footprint != NULL)
          *footprint = 0;
        if (unsat_kind != NULL)
          *unsat_kind = LEGION_FIELD_CONSTRAINT;
        if (unsat_index != NULL)
          *unsat_index = 0;
        return NULL;
      }
      if (realm_layout == NULL)
      {
        const std::vector<FieldID> &field_set = 
          constraints.field_constraint.get_field_set();
        bool compact = false;
        switch (constraints.specialized_constraint.get_kind())
        {
          case LEGION_COMPACT_SPECIALIZE:
          case LEGION_COMPACT_REDUCTION_SPECIALIZE:
            {
              compact = true;
              break;
            }
          default:
            break;
        }
        realm_layout =
          instance_domain->create_layout(constraints, field_set, 
             field_sizes, compact, unsat_kind, unsat_index, 
             &piece_list, &piece_list_size);
        // If constraints were unsatisfied then return now
        if (realm_layout == NULL)
          return NULL;
      }
      // Clone the realm layout each time since (realm will take ownership 
      // after every instance call, so we need a new one each time)
      Realm::InstanceLayoutGeneric *inst_layout = realm_layout->clone();
#ifdef DEBUG_LEGION
      assert(inst_layout != NULL);
#endif
      // Have to grab this now since realm is going to take ownership of
      // the instance layout generic object once we do the creation call
      const size_t instance_footprint = inst_layout->bytes_used;
      // Save the footprint size if we need to
      if (footprint != NULL)
        *footprint = instance_footprint;
      Realm::ProfilingRequestSet requests;
      // Add a profiling request to see if the instance is actually allocated
      // Make it very high priority so we get the response quickly
      ProfilingResponseBase base(this);
#ifndef LEGION_MALLOC_INSTANCES
      Realm::ProfilingRequest &req = requests.add_request(
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID,
          &base, sizeof(base), LG_RESOURCE_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::InstanceAllocResult>();
      // Create a user event to wait on for the result of the profiling response
      profiling_ready = Runtime::create_rt_user_event();
#endif
#ifdef DEBUG_LEGION
      assert(!instance.exists()); // shouldn't exist before this
#endif
      ApEvent ready;
#ifndef LEGION_MALLOC_INSTANCES
      if (runtime->profiler != NULL)
        runtime->profiler->add_inst_request(requests, creator_id);
      ready = ApEvent(PhysicalInstance::create_instance(instance,
            memory_manager->memory, inst_layout, requests, precondition));
      // Wait for the profiling response
      if (!profiling_ready.has_triggered())
        profiling_ready.wait();
#else
      if (precondition.exists() && !precondition.has_triggered())
        precondition.wait();
      uintptr_t base_ptr = 0;
      if (instance_footprint > 0)
      {
        base_ptr = 
          memory_manager->allocate_legion_instance(instance_footprint);
        if (base_ptr == 0)
        {
          if (unsat_kind != NULL)
            *unsat_kind = LEGION_MEMORY_CONSTRAINT;
          if (unsat_index != NULL)
            *unsat_index = 0;
          return NULL;
        }
      }
      ready = ApEvent(PhysicalInstance::create_external(instance,
            memory_manager->memory, base_ptr, inst_layout, requests));
#endif
      // If we couldn't make it then we are done
      if (!instance.exists())
      {
        if (unsat_kind != NULL)
          *unsat_kind = LEGION_MEMORY_CONSTRAINT;
        if (unsat_index != NULL)
          *unsat_index = 0;
        return NULL;
      }
      // For Legion Spy we need a unique ready event if it doesn't already
      // exist so we can uniquely identify the instance
      if (!ready.exists() && runtime->legion_spy_enabled)
      {
        ApUserEvent rename_ready = Runtime::create_ap_user_event(NULL);
        Runtime::trigger_event(NULL, rename_ready);
        ready = rename_ready;
      }
      // If we successfully made the instance then Realm 
      // took over ownership of the layout
      PhysicalManager *result = NULL;
      DistributedID did = forest->runtime->get_available_distributed_id();
      AddressSpaceID local_space = forest->runtime->address_space;
#ifdef LEGION_DEBUG
      assert(!constraints.pointer_constraint.is_valid);
#endif
      // If we successfully made it then we can 
      // switch over the polarity of our constraints, this
      // shouldn't be necessary once Realm gets its act together
      // and actually tells us what the resulting constraints are
      constraints.field_constraint.contiguous = true;
      constraints.field_constraint.inorder = true;
      constraints.ordering_constraint.contiguous = true;
      constraints.memory_constraint = MemoryConstraint(
                                        memory_manager->memory.kind());
      constraints.specialized_constraint.collective = Domain();
      const unsigned num_dims = instance_domain->get_num_dims();
      // Now let's find the layout constraints to use for this instance
      LayoutDescription *layout = field_space_node->find_layout_description(
                                        instance_mask, num_dims, constraints);
      // If we couldn't find one then we make one
      if (layout == NULL)
      {
        // First make a new layout constraint
        LayoutConstraints *layout_constraints = 
          forest->runtime->register_layout(field_space_node->handle,
                                           constraints, true/*internal*/);
        // Then make our description
        layout = field_space_node->create_layout_description(instance_mask, 
                                  num_dims, layout_constraints, mask_index_map,
                                  constraints.field_constraint.get_field_set(),
                                  field_sizes, serdez);
      }
      // Figure out what kind of instance we just made
      switch (constraints.specialized_constraint.get_kind())
      {
        case LEGION_NO_SPECIALIZE:
        case LEGION_AFFINE_SPECIALIZE:
        case LEGION_COMPACT_SPECIALIZE:
          {
            // Now we can make the manager
            result = new IndividualManager(forest, did, local_space,
                                           memory_manager,
                                           instance, instance_domain, 
                                           piece_list, piece_list_size,
                                           field_space_node, tree_id,
                                           layout, 0/*redop id*/,
                                           true/*register now*/, 
                                           instance_footprint, ready,
                                           false/*external instance*/);
            // manager takes ownership of the piece list
            piece_list = NULL;
            break;
          }
        case LEGION_AFFINE_REDUCTION_SPECIALIZE:
        case LEGION_COMPACT_REDUCTION_SPECIALIZE:
          {
            result = new IndividualManager(forest, did, local_space,
                                           memory_manager, instance, 
                                           instance_domain, piece_list,
                                           piece_list_size, field_space_node,
                                           tree_id, layout, redop_id,
                                           true/*register now*/,
                                           instance_footprint, ready,
                                           false/*external instance*/,
                                           reduction_op);
            // manager takes ownership of the piece list
            piece_list = NULL;
            break;
          }
        default:
          assert(false); // illegal specialized case
      }
#ifdef LEGION_MALLOC_INSTANCES
      memory_manager->record_legion_instance(result, base_ptr); 
#endif
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      if (runtime->profiler != NULL)
      {
        // Log the logical regions and fields that make up this instance
        for (std::vector<LogicalRegion>::const_iterator it =
              regions.begin(); it != regions.end(); it++)
          runtime->profiler->record_physical_instance_region(creator_id, 
                                                      instance.id, *it);
        runtime->profiler->record_physical_instance_layout(
                                                     creator_id,
                                                     instance.id,
                                                     layout->owner->handle,
                                                     layout->constraints);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    CollectiveManager* InstanceBuilder::create_collective_instance(
        RegionTreeForest *forest, Memory::Kind mem_kind, 
        IndexSpaceNode *point_space, LayoutConstraintKind *unsat_kind, 
        unsigned *unsat_index, ApEvent ready_event, size_t *footprint)
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
        if (footprint != NULL)
          *footprint = 0;
        if (unsat_kind != NULL)
          *unsat_kind = LEGION_FIELD_CONSTRAINT;
        if (unsat_index != NULL)
          *unsat_index = 0;
        return NULL;
      }
      if (realm_layout == NULL)
      {
        const std::vector<FieldID> &field_set = 
          constraints.field_constraint.get_field_set();
        bool compact = false;
        switch (constraints.specialized_constraint.get_kind())
        {
          case LEGION_COMPACT_SPECIALIZE:
          case LEGION_COMPACT_REDUCTION_SPECIALIZE:
            {
              compact = true;
              break;
            }
          default:
            break;
        }
        realm_layout =
          instance_domain->create_layout(constraints, field_set, 
             field_sizes, compact, unsat_kind, unsat_index, 
             &piece_list, &piece_list_size);
        // If constraints were unsatisfied then return now
        if (realm_layout == NULL)
          return NULL;
      }
      const size_t instance_footprint = realm_layout->bytes_used;
      // Save the footprint size if we need to
      if (footprint != NULL)
        *footprint = instance_footprint;
      // If we successfully made the layout then tighten the constraints
      constraints.field_constraint.contiguous = true;
      constraints.field_constraint.inorder = true;
      constraints.ordering_constraint.contiguous = true;
      constraints.memory_constraint = MemoryConstraint(mem_kind);
      constraints.specialized_constraint.collective = Domain();
      const unsigned num_dims = instance_domain->get_num_dims();
      // Now let's find the layout constraints to use for this instance
      LayoutDescription *layout = field_space_node->find_layout_description(
                                        instance_mask, num_dims, constraints);
      // If we couldn't find one then we make one
      if (layout == NULL)
      {
        // First make a new layout constraint
        LayoutConstraints *layout_constraints = 
          forest->runtime->register_layout(field_space_node->handle,
                                           constraints, true/*internal*/);
        // Then make our description
        layout = field_space_node->create_layout_description(instance_mask, 
                                  num_dims, layout_constraints, mask_index_map,
                                  constraints.field_constraint.get_field_set(),
                                  field_sizes, serdez);
      }
      CollectiveManager *result = NULL;
      DistributedID did = forest->runtime->get_available_distributed_id();
      AddressSpaceID local_space = forest->runtime->address_space;
      switch (constraints.specialized_constraint.get_kind())
      {
        case LEGION_NO_SPECIALIZE:
        case LEGION_AFFINE_SPECIALIZE:
        case LEGION_COMPACT_SPECIALIZE:
          {
            result = new CollectiveManager(forest, did, local_space,
                point_space, instance_domain, piece_list, piece_list_size,
                field_space_node, tree_id, layout, 0/*redop*/, true/*register*/,
                instance_footprint, ready_event, false/*external*/); 
            break;
          }
        case LEGION_AFFINE_REDUCTION_SPECIALIZE:
        case LEGION_COMPACT_REDUCTION_SPECIALIZE:
          {
            result = new CollectiveManager(forest, did, local_space,
                point_space, instance_domain, piece_list, piece_list_size,
                field_space_node, tree_id, layout, redop_id, true/*register*/,
                instance_footprint, ready_event, false/*external*/); 
            break;
          }
        default:
          assert(false);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void InstanceBuilder::handle_profiling_response(
                                       const ProfilingResponseBase *base,
                                       const Realm::ProfilingResponse &response,
                                       const void *orig, size_t orig_length)
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
        if (runtime->profiler != NULL)
          runtime->profiler->handle_failed_instance_allocation();
      }
      else if (runtime->profiler != NULL)
      {
        unsigned long long creation_time = 
          Realm::Clock::current_time_in_nanoseconds();
        runtime->profiler->record_instance_creation(instance,
            memory_manager->memory, creator_id, creation_time);
      }
      // No matter what trigger the event
      Runtime::trigger_event(profiling_ready);
    }

    //--------------------------------------------------------------------------
    void InstanceBuilder::initialize(RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      compute_space_and_domain(forest); 
      compute_layout_parameters();
      valid = true;
    }

    //--------------------------------------------------------------------------
    void InstanceBuilder::compute_space_and_domain(RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!regions.empty());
      assert(field_space_node == NULL);
      assert(instance_domain == NULL);
      assert(tree_id == 0);
#endif
      std::set<IndexSpaceExpression*> region_exprs;
      for (std::vector<LogicalRegion>::const_iterator it = 
            regions.begin(); it != regions.end(); it++)
      {
        if (field_space_node == NULL)
          field_space_node = forest->get_node(it->get_field_space());
        if (tree_id == 0)
          tree_id = it->get_tree_id();
#ifdef DEBUG_LEGION
        // Check to make sure that all the field spaces have the same handle
        assert(field_space_node->handle == it->get_field_space());
        assert(tree_id == it->get_tree_id());
#endif
        region_exprs.insert(forest->get_node(it->get_index_space()));
      }
      instance_domain = (region_exprs.size() == 1) ? 
        *(region_exprs.begin()) : forest->union_index_spaces(region_exprs);
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
          if (ord.ordering[idx] == LEGION_DIM_F)
          {
            // Should never be duplicated 
            if (field_idx != -1)
              REPORT_LEGION_ERROR(ERROR_ILLEGAL_LAYOUT_CONSTRAINT,
                  "Illegal ordering constraint used during instance "
                  "creation contained multiple instances of DIM_F")
            else
              field_idx = idx;
          }
          else if (ord.ordering[idx] > LEGION_DIM_F)
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
                DimensionKind dim = (DimensionKind)(LEGION_DIM_X + idx);
                if (spatial_dims.find(dim) == spatial_dims.end())
                  ord.ordering.push_back(dim);
              }
            }
            else if (field_idx == int(ord.ordering.size()-1))
            {
              // Add them to the front
              for (int idx = (num_dims-1); idx >= 0; idx--)
              {
                DimensionKind dim = (DimensionKind)(LEGION_DIM_X + idx);
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
              DimensionKind dim = (DimensionKind)(LEGION_DIM_X + idx);
              if (spatial_dims.find(dim) == spatial_dims.end())
                ord.ordering.push_back(dim);
            }
          }
        }
        // If we didn't see the field dimension either then add that
        // at the end to give us SOA layouts in general
        if (field_idx == -1)
          ord.ordering.push_back(LEGION_DIM_F);
        // We've now got all our dimensions so we can set the
        // contiguous flag to true
        ord.contiguous = true;
      }
      else
      {
        // We had no ordering constraints so populate it with 
        // SOA constraints for now
        for (unsigned idx = 0; idx < num_dims; idx++)
          ord.ordering.push_back((DimensionKind)(LEGION_DIM_X + idx));
        ord.ordering.push_back(LEGION_DIM_F);
        ord.contiguous = true;
      }
#ifdef DEBUG_LEGION
      assert(ord.contiguous);
      assert(ord.ordering.size() == (num_dims + 1));
#endif
      // From this we should be able to compute the field groups 
      // Use the FieldConstraint to put any fields in the proper order
      const std::vector<FieldID> &field_set = 
        constraints.field_constraint.get_field_set(); 
      field_sizes.resize(field_set.size());
      mask_index_map.resize(field_set.size());
      serdez.resize(field_set.size());
      field_space_node->compute_field_layout(field_set, field_sizes,
                                       mask_index_map, serdez, instance_mask);
      // See if we have any specialization here that will 
      // require us to update the field sizes
      switch (constraints.specialized_constraint.get_kind())
      {
        case LEGION_NO_SPECIALIZE:
        case LEGION_AFFINE_SPECIALIZE:
        case LEGION_COMPACT_SPECIALIZE:
          break;
        case LEGION_AFFINE_REDUCTION_SPECIALIZE:
        case LEGION_COMPACT_REDUCTION_SPECIALIZE:
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
        case LEGION_VIRTUAL_SPECIALIZE:
          {
            REPORT_LEGION_ERROR(ERROR_ILLEGAL_REQUEST_VIRTUAL_INSTANCE,
                          "Illegal request to create a virtual instance");
            assert(false);
          }
        default:
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_REQUEST_VIRTUAL_INSTANCE,
                        "Illegal request to create instance of type %d", 
                        constraints.specialized_constraint.get_kind())
      }
    }

  }; // namespace Internal
}; // namespace Legion

