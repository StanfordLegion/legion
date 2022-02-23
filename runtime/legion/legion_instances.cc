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
#include "legion/legion_replication.h"

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

    //--------------------------------------------------------------------------
    unsigned CopyAcrossHelper::convert_src_to_dst(unsigned index)
    //--------------------------------------------------------------------------
    {
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
#ifdef DEBUG_LEGION
      assert(forward_map.find(index) != forward_map.end());
#endif
      return forward_map[index];
    }

    //--------------------------------------------------------------------------
    unsigned CopyAcrossHelper::convert_dst_to_src(unsigned index)
    //--------------------------------------------------------------------------
    {
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
#ifdef DEBUG_LEGION
      assert(backward_map.find(index) != backward_map.end());
#endif
      return backward_map[index];
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
    // Collective Mapping
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CollectiveMapping::CollectiveMapping(
                            const std::vector<AddressSpaceID> &spaces, size_t r)
      : total_spaces(spaces.size()), radix(r)
    //--------------------------------------------------------------------------
    {
      for (std::vector<AddressSpaceID>::const_iterator it =
            spaces.begin(); it != spaces.end(); it++)
        unique_sorted_spaces.add(*it);
    }

    //--------------------------------------------------------------------------
    CollectiveMapping::CollectiveMapping(const ShardMapping &mapping, size_t r)
      : total_spaces(mapping.size()), radix(r)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < total_spaces; idx++)
        unique_sorted_spaces.add(mapping[idx]);
    }

    //--------------------------------------------------------------------------
    CollectiveMapping::CollectiveMapping(Deserializer &derez, size_t total)
      : total_spaces(total)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(total_spaces > 0);
#endif
      derez.deserialize(unique_sorted_spaces);
#ifdef DEBUG_LEGION
      assert(total_spaces == unique_sorted_spaces.size());
#endif
      derez.deserialize(radix);
    }

    //--------------------------------------------------------------------------
    bool CollectiveMapping::operator==(const CollectiveMapping &rhs) const
    //--------------------------------------------------------------------------
    {
      if (radix != rhs.radix)
        return false;
      return unique_sorted_spaces == rhs.unique_sorted_spaces;
    }

    //--------------------------------------------------------------------------
    bool CollectiveMapping::operator!=(const CollectiveMapping &rhs) const
    //--------------------------------------------------------------------------
    {
      return !((*this) == rhs);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID CollectiveMapping::get_parent(const AddressSpaceID origin,
                                               const AddressSpaceID local) const
    //--------------------------------------------------------------------------
    {
      const unsigned local_index = find_index(local);
      const unsigned origin_index = find_index(origin);
#ifdef DEBUG_LEGION
      assert(local_index < total_spaces);
      assert(origin_index < total_spaces);
#endif
      const unsigned offset = convert_to_offset(local_index, origin_index);
      const unsigned index = convert_to_index((offset-1) / radix, origin_index);
      return unique_sorted_spaces.get_index(index);
    }
    
    //--------------------------------------------------------------------------
    size_t CollectiveMapping::count_children(const AddressSpaceID origin,
                                             const AddressSpaceID local) const
    //--------------------------------------------------------------------------
    {
      const unsigned local_index = find_index(local);
      const unsigned origin_index = find_index(origin);
#ifdef DEBUG_LEGION
      assert(local_index < total_spaces);
      assert(origin_index < total_spaces);
#endif
      const unsigned offset = radix *
        convert_to_offset(local_index, origin_index);
      size_t result = 0;
      for (unsigned idx = 1; idx <= radix; idx++)
      {
        const unsigned child_offset = offset + idx;
        if (child_offset < total_spaces)
          result++;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void CollectiveMapping::get_children(const AddressSpaceID origin,
        const AddressSpaceID local, std::vector<AddressSpaceID> &children) const
    //--------------------------------------------------------------------------
    {
      const unsigned local_index = find_index(local);
      const unsigned origin_index = find_index(origin);
#ifdef DEBUG_LEGION
      assert(local_index < total_spaces);
      assert(origin_index < total_spaces);
#endif
      const unsigned offset = radix *
        convert_to_offset(local_index, origin_index);
      for (unsigned idx = 1; idx <= radix; idx++)
      {
        const unsigned child_offset = offset + idx;
        if (child_offset < total_spaces)
        {
          const unsigned index = convert_to_index(child_offset, origin_index);
          children.push_back(unique_sorted_spaces.get_index(index));
        }
      }
    }

    //--------------------------------------------------------------------------
    AddressSpaceID CollectiveMapping::find_nearest(AddressSpaceID search) const
    //--------------------------------------------------------------------------
    {
      unsigned first = 0;
      unsigned last = size() - 1;
      if (search < (*this)[first])
        return (*this)[first];
      if (search > (*this)[last])
        return (*this)[last];
      // Contained somewhere in the middle so binary
      // search for the two nearest options
      unsigned mid = 0;
      while (first <= last)
      {
        mid = (first + last) / 2;
        const AddressSpaceID midval = (*this)[mid];
#ifdef DEBUG_LEGION
        // Should never actually find it
        assert(search != midval);
#endif
        if (search < midval)
          last = mid - 1;
        else if (midval < search)
          first = mid + 1;
        else
          break;
      }
#ifdef DEBUG_LEGION
      assert(first != last);
#endif
      const unsigned diff_low = search - (*this)[first];
      const unsigned diff_high = (*this)[last] - search;
      if (diff_low < diff_high)
        return (*this)[first];
      else
        return (*this)[last];
    }

    //--------------------------------------------------------------------------
    bool CollectiveMapping::contains(const CollectiveMapping &rhs) const
    //--------------------------------------------------------------------------
    {
      return !(rhs.unique_sorted_spaces - unique_sorted_spaces);
    }

    //--------------------------------------------------------------------------
    CollectiveMapping* CollectiveMapping::clone_with(AddressSpaceID space) const
    //--------------------------------------------------------------------------
    {
      CollectiveMapping *result = new CollectiveMapping(*this);
      result->unique_sorted_spaces.insert(space);
      result->total_spaces = result->unique_sorted_spaces.size();
      return result;
    }

    //--------------------------------------------------------------------------
    void CollectiveMapping::pack(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(total_spaces > 0);
#endif
      rez.serialize(total_spaces);
      rez.serialize(unique_sorted_spaces);
      rez.serialize(radix);
    }

    //--------------------------------------------------------------------------
    unsigned CollectiveMapping::convert_to_offset(unsigned index,
                                                  unsigned origin_index) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index < total_spaces);
      assert(origin_index < total_spaces);
#endif
      if (index < origin_index)
      {
        // Modulus arithmetic here
        return ((index + total_spaces) - origin_index);
      }
      else
        return (index - origin_index);
    }

    //--------------------------------------------------------------------------
    unsigned CollectiveMapping::convert_to_index(unsigned offset,
                                                 unsigned origin_index) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(offset < total_spaces);
      assert(origin_index < total_spaces);
#endif
      unsigned result = origin_index + offset;
      if (result >= total_spaces)
        result -= total_spaces;
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
                                     RegionTreeID tid, bool register_now,
                                     CollectiveMapping *mapping)
      : DistributedCollectable(ctx->runtime, did, owner_space, 
                               register_now, mapping),
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
                                     bool register_now, bool shadow,
                                     bool output, CollectiveMapping *mapping)
      : InstanceManager(ctx, owner_space, did, layout, node,
          // If we're on the owner node we need to produce the expression
          // that actually describes this points in this space
          // On remote nodes we'll already have it from the owner
          (owner_space == ctx->runtime->address_space) && !output ?
             index_domain->create_layout_expression(pl, pl_size) : index_domain,
          tree_id, register_now, mapping), 
        instance_footprint(footprint), reduction_op(rop), redop(redop_id),
        unique_event(u_event), piece_list(pl), piece_list_size(pl_size), 
        shadow_instance(shadow)
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
#ifdef LEGION_GPU_REDUCTIONS
      if (!shadow_reduction_instances.empty())
      {
        for (std::map<std::pair<unsigned,ReductionOpID>,ReductionView*>::
              const_iterator it = shadow_reduction_instances.begin();
              it != shadow_reduction_instances.end(); it++)
          if ((it->second != NULL) &&
              it->second->remove_nested_resource_ref(did))
            delete it->second;
        shadow_reduction_instances.clear();
      }
#endif
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
    InstanceView* PhysicalManager::construct_top_view(
                                           AddressSpaceID logical_owner,
                                           DistributedID view_did, UniqueID uid,
                                           CollectiveMapping *mapping)
    //--------------------------------------------------------------------------
    {
      if (redop > 0)
        return new ReductionView(context, view_did, owner_space,
            logical_owner, this, uid, true/*register now*/, mapping);
      else
        return new MaterializedView(context, view_did, owner_space, 
              logical_owner, this, uid, true/*register now*/, mapping);
    }

    //--------------------------------------------------------------------------
    InstanceView* PhysicalManager::find_or_create_instance_top_view(
                                                   InnerContext *own_ctx,
                                                   AddressSpaceID logical_owner,
                                                   CollectiveMapping *mapping)
    //--------------------------------------------------------------------------
    {
      ContextKey key(own_ctx->get_replication_id(), own_ctx->get_context_uid());
      // If we're a replicate context then we want to ignore the specific
      // context UID since there might be several shards on this node
      if (key.first > 0)
        key.second = 0;
      // No matter what we're going to store the context so grab a reference
      own_ctx->add_reference();
      RtEvent wait_for;
      {
        AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
        // All contexts should always be new since they should be deduplicating
        // on their side before calling this method
        assert(active_contexts.find(own_ctx) == active_contexts.end());
#endif
        std::map<ContextKey,ViewEntry>::iterator finder =
          context_views.find(key);
        if (finder != context_views.end())
        {
#ifdef DEBUG_LEGION
          // This should only happen with control replication because normal
          // contexts should be deduplicating on their side
          assert(key.first > 0);
#endif
          // This better be a new context so bump the reference count
          active_contexts.insert(own_ctx);
          finder->second.second++;
          return finder->second.first;
        }
        // Check to see if someone else from this context is making the view 
        if (key.first > 0)
        {
          // Only need to do this for control replication, otherwise the
          // context will have deduplicated for us
          std::map<ReplicationID,RtUserEvent>::iterator pending_finder =
            pending_views.find(key.first);
          if (pending_finder != pending_views.end())
          {
            if (!pending_finder->second.exists())
              pending_finder->second = Runtime::create_rt_user_event();
            wait_for = pending_finder->second;
          }
          else
            pending_views[key.first] = RtUserEvent::NO_RT_USER_EVENT;
        }
      }
      if (wait_for.exists())
      {
        if (!wait_for.has_triggered())
          wait_for.wait();
        AutoLock i_lock(inst_lock);
        std::map<ContextKey,ViewEntry>::iterator finder =
          context_views.find(key);
#ifdef DEBUG_LEGION
        assert(finder != context_views.end());
        assert(key.first > 0);
#endif
        // This better be a new context so bump the reference count
        active_contexts.insert(own_ctx);
        finder->second.second++;
        return finder->second.first;
      }
      // At this point we're repsonsibile for doing the work to make the view 
      InstanceView *result = NULL;
      // Check to see if we're the owner
      if (is_owner())
      {
        // We're going to construct the view no matter what, see which 
        // node is going to be the logical owner
        DistributedID view_did = runtime->get_available_distributed_id(); 
        result = construct_top_view((mapping == NULL) ? logical_owner :
            owner_space, view_did, own_ctx->get_context_uid(), mapping);
      }
      else if (mapping != NULL)
      {
        // If we're collectively making this view then we're just going to
        // do that and use the owner node as the logical owner for the view
        // We still need to get the distributed ID from the next node down
        // in the collective mapping though
        std::atomic<DistributedID> view_did(0);
        RtUserEvent ready = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(key.first);
          rez.serialize(key.second);
          rez.serialize(owner_space);
          mapping->pack(rez);
          rez.serialize(&view_did);
          rez.serialize(ready);
        }
        AddressSpaceID target = mapping->get_parent(owner_space, local_space);
        runtime->send_create_top_view_request(target, rez); 
        ready.wait();
        result = construct_top_view(owner_space, view_did.load(), 
                            own_ctx->get_context_uid(), mapping);
      }
      else
      {
        // We're not collective and not the owner so send the request
        // to the owner to make the logical view and send back the result
        std::atomic<DistributedID> view_did(0);
        RtUserEvent ready = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(key.first);
          rez.serialize(key.second);
          rez.serialize(logical_owner);
          rez.serialize<size_t>(0); // no mapping
          rez.serialize(&view_did);
          rez.serialize(ready);
        }
        runtime->send_create_top_view_request(owner_space, rez); 
        ready.wait();
        RtEvent view_ready;
        result = static_cast<InstanceView*>(
            runtime->find_or_request_logical_view(view_did.load(), view_ready));
        if (view_ready.exists() && !view_ready.has_triggered())
          view_ready.wait();
      }
      // Retake the lock, save the view, and signal any other waiters
      AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
      assert(context_views.find(key) == context_views.end());
#endif
      ViewEntry &entry = context_views[key];
      entry.first = result;
      entry.second = 1/*only a single initial reference*/;
      active_contexts.insert(own_ctx);
      if (key.first > 0)
      {
        std::map<ReplicationID,RtUserEvent>::iterator finder =
          pending_views.find(key.first);
#ifdef DEBUG_LEGION
        assert(finder != pending_views.end());
#endif
        if (finder->second.exists())
          Runtime::trigger_event(finder->second);
        pending_views.erase(finder);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::unregister_active_context(InnerContext *own_ctx)
    //--------------------------------------------------------------------------
    {
      ContextKey key(own_ctx->get_replication_id(), own_ctx->get_context_uid());
      // If we're a replicate context then we want to ignore the specific
      // context UID since there might be several shards on this node
      if (key.first > 0)
        key.second = 0;
      {
        AutoLock inst(inst_lock);
        std::set<InnerContext*>::iterator finder = 
          active_contexts.find(own_ctx);
        // We could already have removed this context if this
        // physical instance was deleted
        if (finder == active_contexts.end())
          return;
        active_contexts.erase(finder);
        // Remove the reference on the view entry and remove it from our
        // manager if it no longer has anymore active contexts
        std::map<ContextKey,ViewEntry>::iterator view_finder =
          context_views.find(key);
#ifdef DEBUG_LEGION
        assert(view_finder != context_views.end());
        assert(view_finder->second.second > 0);
#endif
        if (--view_finder->second.second == 0)
          context_views.erase(view_finder);
      }
      if (own_ctx->remove_reference())
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
      // If we have any gc events then launch tasks to actually prune
      // off their references when they are done since we are now eligible
      // for collection by the garbage collector
      // We can test this without the lock because the race here is with
      // the shutdown detection code (see find_shutdown_preconditions)
      // which is also only reading the data structure
      if (gc_events.empty())
        return;
      // We do need the lock if we're going to be modifying this
      AutoLock inst(inst_lock);
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
    size_t PhysicalManager::get_instance_size(void) const
    //--------------------------------------------------------------------------
    {
      AutoLock lock(inst_lock,1,false/*exlcusive*/);
      return instance_footprint;
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

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_top_view_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez); 
      DistributedID did;
      derez.deserialize(did);
      RtEvent man_ready;
      PhysicalManager *manager =
        runtime->find_or_request_instance_manager(did, man_ready);
      ReplicationID repl_id;
      derez.deserialize(repl_id);
      UniqueID ctx_uid;
      derez.deserialize(ctx_uid);
      RtEvent ctx_ready;
      InnerContext *context = NULL;
      if (repl_id > 0)
      {
        // See if we're on a node where there is a shard manager for
        // this replicated context
        ShardManager *shard_manager = 
          runtime->find_shard_manager(repl_id, true/*can fail*/);
        if (shard_manager != NULL)
          context = shard_manager->find_local_context();
      }
      if (context == NULL)
        context = runtime->find_context(ctx_uid,false/*can't fail*/,&ctx_ready);
      AddressSpaceID logical_owner;
      derez.deserialize(logical_owner);
      CollectiveMapping *mapping = NULL;
      size_t total_spaces;
      derez.deserialize(total_spaces);
      if (total_spaces > 0)
      {
        mapping = new CollectiveMapping(derez, total_spaces);
        mapping->add_reference();
      }
      std::atomic<DistributedID> *target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);
      // See if we're ready or we need to defer this until later
      if ((man_ready.exists() && !man_ready.has_triggered()) ||
          (ctx_ready.exists() && !ctx_ready.has_triggered()))
      {
        RemoteCreateViewArgs args(manager, context, logical_owner,
                                  mapping, target, source, done);
        if (!man_ready.exists())
          runtime->issue_runtime_meta_task(args,
              LG_LATENCY_DEFERRED_PRIORITY, ctx_ready);
        else if (!ctx_ready.exists())
          runtime->issue_runtime_meta_task(args,
              LG_LATENCY_DEFERRED_PRIORITY, man_ready);
        else
          runtime->issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY,
              Runtime::merge_events(man_ready, ctx_ready));
        return;
      }
      process_top_view_request(manager, context, logical_owner, mapping,
                               target, source, done, runtime);
      if ((mapping != NULL) && mapping->remove_reference())
        delete mapping;
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::process_top_view_request(
        PhysicalManager *manager, InnerContext *context, AddressSpaceID logical,
        CollectiveMapping *mapping, std::atomic<DistributedID> *target,
        AddressSpaceID source, RtUserEvent done_event, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      // Get the view from the context
      InstanceView *view =
        context->create_instance_top_view(manager, logical, mapping); 
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(target);
        rez.serialize(view->did);
        rez.serialize(done_event);
      }
      runtime->send_create_top_view_response(source, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_top_view_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::atomic<DistributedID> *target;
      derez.deserialize(target);
      DistributedID did;
      derez.deserialize(did);
      target->store(did);
      RtUserEvent done;
      derez.deserialize(done);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_top_view_creation(const void *args,
                                                              Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const RemoteCreateViewArgs *rargs = (const RemoteCreateViewArgs*)args; 
      process_top_view_request(rargs->manager, rargs->context,
          rargs->logical_owner, rargs->mapping, rargs->target,
          rargs->source, rargs->done_event, runtime);
      if ((rargs->mapping != NULL) && rargs->mapping->remove_reference())
        delete rargs->mapping;
    }

#ifdef LEGION_GPU_REDUCTIONS
    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_create_shadow_request(
                   Runtime *runtime, AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      unsigned fidx;
      derez.deserialize(fidx);
      ReductionOpID redop;
      derez.deserialize(redop);
      AddressSpaceID request;
      derez.deserialize(request);
      UniqueID opid;
      derez.deserialize(opid);
      PhysicalManager *target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);

      RtEvent ready;
      PhysicalManager *manager = 
        runtime->find_or_request_instance_manager(did, ready); 
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      ReductionView *result = 
        manager->find_or_create_shadow_reduction(fidx, redop, request, opid);
      Serializer rez;
      if (result != NULL)
      {
        RezCheck z2(rez);
        rez.serialize(fidx);
        rez.serialize(redop);
        rez.serialize(target);
        rez.serialize(result->did);
        rez.serialize(to_trigger);
      }
      else
      {
        RezCheck z2(rez);
        rez.serialize(fidx);
        rez.serialize(redop);
        rez.serialize(target);
        rez.serialize<DistributedID>(0);
        rez.serialize(to_trigger);
      }
      runtime->send_create_shadow_reduction_response(source, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_create_shadow_response(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      unsigned fidx;
      derez.deserialize(fidx);
      ReductionOpID redop;
      derez.deserialize(redop);
      PhysicalManager *manager;
      derez.deserialize(manager);
      DistributedID view_did;
      derez.deserialize(view_did);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);

      if (view_did > 0)
      {
        RtEvent ready;
        LogicalView *view = 
          runtime->find_or_request_logical_view(view_did, ready);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
#ifdef DEBUG_LEGION
        assert(view->is_reduction_view());
#endif
        manager->record_remote_shadow_reduction(fidx, redop, 
                                                view->as_reduction_view());
      }
      else
        manager->record_remote_shadow_reduction(fidx, redop, NULL);

      Runtime::trigger_event(to_trigger);
    }
#endif // LEGION_GPU_REDUCTIONS

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
                        ApEvent u_event, InstanceKind k,
                        const ReductionOp *op /*= NULL*/, bool shadow/*=false*/,
                        CollectiveMapping *mapping)
      : PhysicalManager(ctx, desc, encode_instance_did(did, 
           (k != INTERNAL_INSTANCE_KIND), (redop_id != 0), false/*collective*/),
          owner_space, footprint, redop_id, (op != NULL) ? op : 
           (redop_id == 0) ? NULL : ctx->runtime->get_reduction(redop_id), node,
          instance_domain, pl, pl_size, tree_id, u_event, register_now, shadow,
          (k == UNBOUND_INSTANCE_KIND), mapping), memory_manager(memory),
        instance(inst), use_event(Runtime::create_ap_user_event(NULL)),
        instance_ready((k == UNBOUND_INSTANCE_KIND) ? 
            Runtime::create_rt_user_event() : RtUserEvent::NO_RT_USER_EVENT),
        kind(k), external_pointer(-1UL),
        producer_event(
            (k == UNBOUND_INSTANCE_KIND) ? u_event : ApEvent::NO_AP_EVENT)
    //--------------------------------------------------------------------------
    {
      // If the manager was initialized with a valid Realm instance,
      // trigger the use event with the ready event of the instance metadata
      if (kind != UNBOUND_INSTANCE_KIND)
      {
#ifdef DEBUG_LEGION
        assert(instance.exists());
#endif
        Runtime::trigger_event(NULL,use_event,fetch_metadata(instance,u_event));
      }
      else // add a resource reference to remove once this manager is set
        add_base_resource_ref(PENDING_UNBOUND_REF);

      if (!is_owner() && !shadow_instance)
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
      if (runtime->legion_spy_enabled && (kind != UNBOUND_INSTANCE_KIND))
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
    IndividualManager::IndividualManager(const IndividualManager &rhs)
      : PhysicalManager(NULL, NULL, 0, 0, 0, 0, NULL, NULL, NULL, NULL, 0, 0, 
                        ApEvent::NO_AP_EVENT, false, false),
        memory_manager(NULL), instance(PhysicalInstance::NO_INST)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndividualManager::~IndividualManager(void)
    //--------------------------------------------------------------------------
    {
      // Remote references removed by DistributedCollectable destructor
      if (!is_owner() && !shadow_instance)
        memory_manager->unregister_remote_instance(this);
    }

    //--------------------------------------------------------------------------
    IndividualManager& IndividualManager::operator=(const IndividualManager &rh)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void IndividualManager::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (is_owner())
        assert((kind == UNBOUND_INSTANCE_KIND) || instance.exists());
#endif
      // Shadow instances do not participate here
      if (shadow_instance)
        return;
      // Will be null for virtual managers
      if (memory_manager != NULL)
        memory_manager->activate_instance(this);
      // If we are not the owner, send a reference
      if (!is_owner())
        send_remote_gc_increment(owner_space, mutator);
    }

    //--------------------------------------------------------------------------
    void IndividualManager::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (is_owner())
        assert((kind == UNBOUND_INSTANCE_KIND) || instance.exists());
#endif
      // Shadow instances do not participate here
      if (shadow_instance)
        return;
      // Will be null for virtual managers
      if (memory_manager != NULL)
        memory_manager->deactivate_instance(this);
      if (!is_owner())
        send_remote_gc_decrement(owner_space, mutator);
    }

    //--------------------------------------------------------------------------
    void IndividualManager::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // No need to do anything
#ifdef DEBUG_LEGION
      if (is_owner())
        assert((kind == UNBOUND_INSTANCE_KIND) || instance.exists());
#endif
      // Shadow instances do not participate here
      if (shadow_instance)
        return;
      // Will be null for virtual managers
      if (memory_manager != NULL)
        memory_manager->validate_instance(this);
      // If we are not the owner, send a reference
      if (!is_owner())
        send_remote_valid_increment(owner_space, mutator);
    }

    //--------------------------------------------------------------------------
    void IndividualManager::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (is_owner())
        assert((kind == UNBOUND_INSTANCE_KIND) || instance.exists());
#endif 
      prune_gc_events(); 
      // Shadow instances do not participate here
      if (shadow_instance)
        return;
      // Will be null for virtual managers
      if (memory_manager != NULL)
        memory_manager->invalidate_instance(this);
      if (!is_owner())
        send_remote_valid_decrement(owner_space, mutator);
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
    ApEvent IndividualManager::get_use_event(ApEvent user) const
    //--------------------------------------------------------------------------
    {
      if (kind != UNBOUND_INSTANCE_KIND)
        return use_event;
      else
        // If the user is the one that is going to bind an instance
        // to this manager, return a no event
        return (user == producer_event) ? ApEvent::NO_AP_EVENT : use_event;
    }

    //--------------------------------------------------------------------------
    RtEvent IndividualManager::get_instance_ready_event(void) const
    //--------------------------------------------------------------------------
    {
      return instance_ready;
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
        trace_info.record_fill_views(result, fill_expression, *tracing_srcs, 
                                     *tracing_dsts,effects_applied,(redop > 0));
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
#ifdef LEGION_GPU_REDUCTIONS
#ifndef LEGION_SPY
      // Realm is really bad at applying reductions to GPU instances right
      // now so let's help it out by running tasks to apply reductions for it
      // See github issues #372 and #821
      if ((reduction_op_id > 0) &&
          (memory_manager->memory.kind() == Memory::GPU_FB_MEM) &&
          is_gpu_visible(source_manager))
      {
        const GPUReductionTable &gpu_reductions = 
          Runtime::get_gpu_reduction_table();
        std::map<ReductionOpID,TaskID>::const_iterator finder = 
          gpu_reductions.find(reduction_op_id);
        if (finder != gpu_reductions.end())
        {
          // If we can directly perform memory accesses between the
          // two memories then we can launch a kernel that just runs
          // normal CUDA kernels without having any problems
          const ApEvent result = copy_expression->gpu_reduction(trace_info,
              dst_fields, src_fields, memory_manager->get_local_gpu(), 
              finder->second, this, source_manager, precondition, 
              predicate_guard, reduction_op_id, false/*fold*/);
          if (trace_info.recording)
            trace_info.record_copy_views(result, copy_expression, *tracing_srcs,
                                         *tracing_dsts, effects_applied);
          return result;
        }
      }
#endif
#endif 
      const ApEvent result = copy_expression->issue_copy(trace_info, 
                                         dst_fields, src_fields,
#ifdef LEGION_SPY
                                         source_manager->tree_id, tree_id,
#endif
                                         precondition, predicate_guard,
                                         reduction_op_id, false/*fold*/); 
      if (trace_info.recording)
        trace_info.record_copy_views(result, copy_expression, *tracing_srcs, 
                                     *tracing_dsts, effects_applied);
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
    ApEvent IndividualManager::register_collective_user(InstanceView *view, 
                                            const RegionUsage &usage,
                                            const FieldMask &user_mask,
                                            IndexSpaceNode *expr,
                                            const UniqueID op_id,
                                            const size_t op_ctx_index,
                                            const unsigned index,
                                            ApEvent term_event,
                                            RtEvent collect_event,
                                            std::set<RtEvent> &applied_events,
                                            const CollectiveMapping *mapping,
                                            const PhysicalTraceInfo &trace_info,
                                            const AddressSpaceID source,
                                            bool symbolic)
    //--------------------------------------------------------------------------
    {
      // This should only ever be called on collective instances
      assert(false);
      return ApEvent::NO_AP_EVENT;
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
        AutoLock lock(inst_lock,1,false/*exlcusive*/);
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
        if (kind != UNBOUND_INSTANCE_KIND)
          rez.serialize(unique_event);
        else
          rez.serialize(producer_event);
        layout->pack_layout_description(rez, target);
        rez.serialize(redop);
        rez.serialize<bool>(shadow_instance);
        rez.serialize(kind);
      }
      context->runtime->send_instance_manager(target, rez);
      update_remote_instances(target);
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
      bool shadow_inst;
      derez.deserialize<bool>(shadow_inst);
      InstanceKind kind;
      derez.deserialize(kind);
      if (domain_ready.exists() || fs_ready.exists() || layout_ready.exists())
      {
        const RtEvent precondition = 
          Runtime::merge_events(domain_ready, fs_ready, layout_ready);
        if (precondition.exists() && !precondition.has_triggered())
        {
          // We need to defer this instance creation
          DeferIndividualManagerArgs args(did, owner_space, mem, inst,
              inst_footprint, inst_domain, pending, 
              handle, tree_id, layout_id, unique_event, kind,
              redop, piece_list, piece_list_size, shadow_inst);
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
                            kind, redop, shadow_inst);
    }

    //--------------------------------------------------------------------------
    IndividualManager::DeferIndividualManagerArgs::DeferIndividualManagerArgs(
            DistributedID d, AddressSpaceID own, Memory m, PhysicalInstance i, 
            size_t f, IndexSpaceExpression *lx, 
            const PendingRemoteExpression &p, FieldSpace h, RegionTreeID tid,
            LayoutConstraintID l, ApEvent u, InstanceKind k, ReductionOpID r,
            const void *pl, size_t pl_size, bool shadow)
      : LgTaskArgs<DeferIndividualManagerArgs>(implicit_provenance),
            did(d), owner(own), mem(m), inst(i), footprint(f), pending(p),
            local_expr(lx), handle(h), tree_id(tid), layout_id(l), 
            use_event(u), kind(k), redop(r), piece_list(pl),
            piece_list_size(pl_size), shadow_instance(shadow)
    //--------------------------------------------------------------------------
    {
      if (local_expr != NULL)
        local_expr->add_base_expression_reference(META_TASK_REF);
    }

    //--------------------------------------------------------------------------
    IndividualManager::DeferDeleteIndividualManager
                     ::DeferDeleteIndividualManager(IndividualManager *manager_)
      : LgTaskArgs<DeferDeleteIndividualManager>(implicit_provenance),
        manager(manager_)
    //--------------------------------------------------------------------------
    {
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
          dargs->use_event, dargs->kind, dargs->redop, dargs->shadow_instance);
      // Remove the local expression reference if necessary
      if ((dargs->local_expr != NULL) &&
          dargs->local_expr->remove_base_expression_reference(META_TASK_REF))
        delete dargs->local_expr;
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualManager::handle_defer_perform_deletion(
                                             const void *args, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const DeferDeleteIndividualManager *dargs =
        (const DeferDeleteIndividualManager*)args;
      dargs->manager->perform_deletion(RtEvent::NO_RT_EVENT);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualManager::create_remote_manager(Runtime *runtime, 
          DistributedID did, AddressSpaceID owner_space, Memory mem, 
          PhysicalInstance inst, size_t inst_footprint, 
          IndexSpaceExpression *inst_domain, const void *piece_list,
          size_t piece_list_size, FieldSpaceNode *space_node, 
          RegionTreeID tree_id, LayoutConstraints *constraints, 
          ApEvent use_event, InstanceKind kind, ReductionOpID redop,
          bool shadow_instance)
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
      if (runtime->find_pending_collectable_location(did, location))
        man = new(location) IndividualManager(runtime->forest, did, owner_space,
                                              memory, inst, inst_domain, 
                                              piece_list, piece_list_size, 
                                              space_node, tree_id, layout, 
                                              redop, false/*reg now*/, 
                                              inst_footprint, use_event, 
                                              kind, op,
                                              shadow_instance);
      else
        man = new IndividualManager(runtime->forest, did, owner_space, memory, 
                              inst, inst_domain, piece_list, piece_list_size,
                              space_node, tree_id, layout, redop, 
                              false/*reg now*/, inst_footprint, use_event, 
                              kind, op, shadow_instance);
      // Hold-off doing the registration until construction is complete
      man->register_with_runtime(NULL/*no remote registration needed*/);
    }

    //--------------------------------------------------------------------------
    bool IndividualManager::acquire_instance(ReferenceSource source,
                                             ReferenceMutator *mutator,
                                             const DomainPoint &point,
                                             AddressSpaceID *remote_target)
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
      {
        if (remote_target != NULL)
          *remote_target = owner_space;
        return false;
      }
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
    void IndividualManager::perform_deletion(RtEvent deferred_event)
    //--------------------------------------------------------------------------
    {
      if (!instance_ready.has_triggered())
      {
        DeferDeleteIndividualManager args(this);
        runtime->issue_runtime_meta_task(
            args, LG_LOW_PRIORITY,
            Runtime::merge_events(deferred_event, instance_ready));
        return;
      }

#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(kind != UNBOUND_INSTANCE_KIND);
#endif
      log_garbage.spew("Deleting physical instance " IDFMT " in memory " 
                       IDFMT "", instance.id, memory_manager->memory.id);
#ifndef DISABLE_GC
      std::vector<PhysicalInstance::DestroyedField> serdez_fields;
      layout->compute_destroyed_fields(serdez_fields);

#ifndef LEGION_MALLOC_INSTANCES
      // If this is an owned external instance, deallocate it manually
      if (kind == EXTERNAL_OWNED_INSTANCE_KIND)
      {
        memory_manager->free_external_allocation(
            external_pointer, instance_footprint);
        if (!serdez_fields.empty())
          instance.destroy(serdez_fields, deferred_event);
        else
          instance.destroy(deferred_event);
      }
      // If this is an eager allocation, return it back to the eager pool
      else if (kind == EAGER_INSTANCE_KIND)
        memory_manager->free_eager_instance(instance, deferred_event);
      else
#endif
      {
        if (!serdez_fields.empty())
          instance.destroy(serdez_fields, deferred_event);
        else
          instance.destroy(deferred_event);
      }
#ifdef LEGION_MALLOC_INSTANCES
      if (!is_external_instance())
        memory_manager->free_legion_instance(instance, deferred_event);
#endif
#ifdef LEGION_GPU_REDUCTIONS
      for (std::map<std::pair<unsigned/*fidx*/,ReductionOpID>,ReductionView*>::
            const_iterator it = shadow_reduction_instances.begin();
            it != shadow_reduction_instances.end(); it++)
      {
        if (it->second == NULL)
          continue;
        PhysicalManager *manager = it->second->get_manager();
        manager->perform_deletion(deferred_event);
      }
#endif
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
#ifdef DEBUG_LEGION
        assert(pending_views.empty());
#endif
        context_views.clear();
      }
      for (std::set<InnerContext*>::const_iterator it = 
           copy_active_contexts.begin(); it != copy_active_contexts.end(); it++)
      {
        (*it)->notify_instance_deletion(const_cast<IndividualManager*>(this));
        if ((*it)->remove_reference())
          delete (*it);
      }
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
#ifndef LEGION_MALLOC_INSTANCES
      // If this is an owned external instance, deallocate it manually
      if (kind == EXTERNAL_OWNED_INSTANCE_KIND)
      {
        memory_manager->free_external_allocation(
            external_pointer, instance_footprint);
        if (!serdez_fields.empty())
          instance.destroy(serdez_fields);
        else
          instance.destroy();
      }
      // If this is an eager allocation, return it back to the eager pool
      else if (kind == EAGER_INSTANCE_KIND)
        memory_manager->free_eager_instance(instance, RtEvent::NO_RT_EVENT);
      else
#endif
      {
        if (!serdez_fields.empty())
          instance.destroy(serdez_fields);
        else
          instance.destroy();
      }
#ifdef LEGION_MALLOC_INSTANCES
      if (!is_external_instance())
        memory_manager->free_legion_instance(instance, RtEvent::NO_RT_EVENT);
#endif
#ifdef LEGION_GPU_REDUCTIONS
      for (std::map<std::pair<unsigned/*fidx*/,ReductionOpID>,ReductionView*>::
            const_iterator it = shadow_reduction_instances.begin();
            it != shadow_reduction_instances.end(); it++)
      {
        if (it->second == NULL)
          continue;
        PhysicalManager *manager = it->second->get_manager();
        manager->force_deletion();
      }
#endif
#endif
    }

    //--------------------------------------------------------------------------
    void IndividualManager::set_garbage_collection_priority(MapperID mapper_id,
                  Processor proc, GCPriority priority, const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      memory_manager->set_garbage_collection_priority(this, mapper_id,
                                                      proc, priority);
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

#ifdef LEGION_GPU_REDUCTIONS
    //--------------------------------------------------------------------------
    bool IndividualManager::is_gpu_visible(PhysicalManager *other) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memory_manager->memory.kind() == Memory::GPU_FB_MEM);
#endif
      // TODO: support collective managers
      if (other->is_collective_manager())
        return false;
      const Processor gpu = memory_manager->get_local_gpu();
      IndividualManager *manager = other->as_individual_manager();
      return runtime->is_visible_memory(gpu, manager->memory_manager->memory);
    }
    
    //--------------------------------------------------------------------------
    ReductionView* IndividualManager::find_or_create_shadow_reduction(
                                    unsigned fidx, ReductionOpID red, 
                                    AddressSpaceID request_space, UniqueID opid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(redop == 0); // this should not be a reduction instance
#endif
      const std::pair<unsigned,ReductionOpID> key(fidx,red);
      // First check to see if we have it
      RtEvent wait_on;
      RtUserEvent to_trigger;
      {
        AutoLock inst(inst_lock);
        std::map<std::pair<unsigned,ReductionOpID>,ReductionView*>::
          const_iterator finder = shadow_reduction_instances.find(key);
        if (finder != shadow_reduction_instances.end())
          return finder->second;
        // If we didn't find it, see if we should wait for it or make it
        std::map<std::pair<unsigned,ReductionOpID>,RtEvent>::const_iterator
          pending_finder = pending_reduction_shadows.find(key);
        if (pending_finder == pending_reduction_shadows.end())
        {
          to_trigger = Runtime::create_rt_user_event();
          pending_reduction_shadows[key] = to_trigger;
        }
        else
          wait_on = pending_finder->second;
      }
      // If we're not the owner, send a message there to do this
      if (!is_owner() && to_trigger.exists())
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(fidx);
          rez.serialize(red);
          rez.serialize(request_space);
          rez.serialize(opid);
          rez.serialize<PhysicalManager*>(this);
          rez.serialize(to_trigger);
        }
        runtime->send_create_shadow_reduction_request(owner_space, rez);
        wait_on = to_trigger;
      }
      if (wait_on.exists())
      {
        if (!wait_on.has_triggered())
          wait_on.wait();
        AutoLock inst(inst_lock,1,false/*exlcusive*/);
        std::map<std::pair<unsigned,ReductionOpID>,ReductionView*>::
          const_iterator finder = shadow_reduction_instances.find(key);
#ifdef DEBUG_LEGION
        assert(finder != shadow_reduction_instances.end());
#endif
        return finder->second; 
      }
#ifdef DEBUG_LEGION
      assert(to_trigger.exists());
#endif
      // Try to make the shadow instance
      // First create the layout constraints
      LayoutConstraintSet shadow_constraints = *(layout->constraints);
      SpecializedConstraint &specialized = 
        shadow_constraints.specialized_constraint;
      switch (specialized.get_kind())
      {
        case LEGION_NO_SPECIALIZE:
        case LEGION_AFFINE_SPECIALIZE:
          {
            specialized = 
              SpecializedConstraint(LEGION_AFFINE_REDUCTION_SPECIALIZE, red); 
            break;
          }
        case LEGION_COMPACT_SPECIALIZE:
          {
            specialized = 
              SpecializedConstraint(LEGION_COMPACT_REDUCTION_SPECIALIZE, red);
            break;
          }
        default:
          assert(false);
      }
      // Only need on field here
      FieldConstraint &fields = shadow_constraints.field_constraint;
      FieldMask mask;
      mask.set_bit(fidx);
      std::set<FieldID> find_fids;
      std::set<FieldID> basis_fids(
          fields.field_set.begin(), fields.field_set.end());
      field_space_node->get_field_set(mask, basis_fids, find_fids);
#ifdef DEBUG_LEGION
      assert(find_fids.size() == 1);
#endif
      const FieldID fid = *(find_fids.begin());
      fields.field_set.clear();
      fields.field_set.push_back(fid);
      // Construct the instance builder from the constraints
      std::vector<LogicalRegion> dummy_regions;
      InstanceBuilder builder(dummy_regions, instance_domain, field_space_node,
          tree_id, shadow_constraints, runtime, memory_manager, opid,
          piece_list, piece_list_size, true/*shadow instance*/);
      // Then ask the memory manager to try to create it
      PhysicalManager *manager = 
        memory_manager->create_shadow_instance(builder);
      ReductionView *result = NULL;
      // No matter what record this for the future
      if (manager != NULL)
      {
        const DistributedID view_did = 
          context->runtime->get_available_distributed_id();
        result = new ReductionView(context, view_did, 
            local_space, request_space, manager, 0/*uid*/, true/*register*/);
        result->add_nested_resource_ref(did);
      }
      AutoLock inst(inst_lock);
#ifdef DEBUG_LEGION
      assert(shadow_reduction_instances.find(key) == 
              shadow_reduction_instances.end());
#endif
      shadow_reduction_instances[key] = result;
      std::map<std::pair<unsigned,ReductionOpID>,RtEvent>::iterator
        pending_finder = pending_reduction_shadows.find(key); 
#ifdef DEBUG_LEGION
      assert(pending_finder != pending_reduction_shadows.end());
#endif
      pending_reduction_shadows.erase(pending_finder);
      Runtime::trigger_event(to_trigger);
      return result;
    } 

    //--------------------------------------------------------------------------
    void IndividualManager::record_remote_shadow_reduction(unsigned fidx,
                                         ReductionOpID red, ReductionView *view)
    //--------------------------------------------------------------------------
    {
      const std::pair<unsigned,ReductionOpID> key(fidx,red);
      if (view != NULL)
        view->add_nested_resource_ref(did);
      AutoLock inst(inst_lock);
#ifdef DEBUG_LEGION
      assert(shadow_reduction_instances.find(key) == 
              shadow_reduction_instances.end());
#endif
      shadow_reduction_instances[key] = view;
      std::map<std::pair<unsigned,ReductionOpID>,RtEvent>::iterator
        pending_finder = pending_reduction_shadows.find(key); 
#ifdef DEBUG_LEGION
      assert(pending_finder != pending_reduction_shadows.end());
#endif
      pending_reduction_shadows.erase(pending_finder);
    }
#endif // LEGION_GPU_REDUCTIONS

    //--------------------------------------------------------------------------
    bool IndividualManager::update_physical_instance(
                                                  PhysicalInstance new_instance,
                                                  InstanceKind new_kind,
                                                  size_t new_footprint,
                                                  uintptr_t new_pointer)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock lock(inst_lock);
#ifdef DEBUG_LEGION
        assert(kind == UNBOUND_INSTANCE_KIND);
        assert(instance_footprint == -1U);
#endif
        instance = new_instance;
        kind = new_kind;
        external_pointer = new_pointer;
#ifdef DEBUG_LEGION
        assert((kind != EXTERNAL_OWNED_INSTANCE_KIND) || 
                (external_pointer != -1UL));
#endif

        update_instance_footprint(new_footprint);

        Runtime::trigger_event(instance_ready);

        if (runtime->legion_spy_enabled)
        {
          LegionSpy::log_physical_instance(unique_event, instance.id,
            memory_manager->memory.id, instance_domain->expr_id,
            field_space_node->handle, tree_id, redop);
          layout->log_instance_layout(unique_event);
        }

        if (is_owner() && has_remote_instances())
          broadcast_manager_update();

        Runtime::trigger_event(
            NULL, use_event, fetch_metadata(instance, producer_event));
      }
      return remove_base_resource_ref(PENDING_UNBOUND_REF);
    }

    //--------------------------------------------------------------------------
    void IndividualManager::broadcast_manager_update(void)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(instance);
        rez.serialize(instance_footprint);
        rez.serialize(kind);
      }
      BroadcastFunctor functor(context->runtime, rez);
      map_over_remote_instances(functor);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualManager::handle_send_manager_update(
                   Runtime *runtime, AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      PhysicalInstance instance;
      derez.deserialize(instance);
      size_t footprint;
      derez.deserialize(footprint);
      InstanceKind kind;
      derez.deserialize(kind);

      RtEvent manager_ready;
      PhysicalManager *manager =
        runtime->find_or_request_instance_manager(did, manager_ready);
      if (manager_ready.exists() && !manager_ready.has_triggered())
        manager_ready.wait();

      if (manager->as_individual_manager()->update_physical_instance(
                                              instance, kind, footprint))
        delete manager;
    }

    /////////////////////////////////////////////////////////////
    // Collective Manager
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CollectiveManager::CollectiveManager(RegionTreeForest *ctx, 
                                DistributedID did, AddressSpaceID owner_space,
                                IndexSpaceNode *points, size_t total,
                                CollectiveMapping *mapping,
                                IndexSpaceExpression *instance_domain,
                                const void *pl, size_t pl_size,
                                FieldSpaceNode *node, RegionTreeID tree_id,
                                LayoutDescription *desc, ReductionOpID redop_id,
                                bool register_now, size_t footprint,
                                ApBarrier u_barrier, bool external_instance)
      : PhysicalManager(ctx, desc, encode_instance_did(did, external_instance,
            (redop_id != 0), true/*collective*/),
          owner_space, footprint, redop_id, (redop_id == 0) ? NULL : 
            ctx->runtime->get_reduction(redop_id),
          node, instance_domain, pl, pl_size, tree_id, u_barrier, register_now,
          false/*shadow*/, false/*output*/, mapping),  total_points(total),
        point_space(points), collective_barrier(u_barrier),
        finalize_messages(0), deleted_or_detached(false)
    //--------------------------------------------------------------------------
    {
      if (point_space != NULL)
        point_space->add_nested_valid_ref(did);
      if (collective_mapping->count_children(owner_space, local_space) > 0)
        add_nested_resource_ref(did);
#ifdef LEGION_GC
      log_garbage.info("GC Collective Manager %lld %d",
                        LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space); 
#endif
    }

    //--------------------------------------------------------------------------
    CollectiveManager::CollectiveManager(const CollectiveManager &rhs)
      : PhysicalManager(rhs), total_points(rhs.total_points),
        point_space(rhs.point_space)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CollectiveManager::~CollectiveManager(void)
    //--------------------------------------------------------------------------
    {
      if ((point_space != NULL) && point_space->remove_nested_valid_ref(did))
        delete point_space;
      if (is_owner())
        collective_barrier.destroy_barrier();
    }

    //--------------------------------------------------------------------------
    CollectiveManager& CollectiveManager::operator=(
                                                   const CollectiveManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool CollectiveManager::contains_isomorphic_points(
                                                   IndexSpaceNode *points) const
    //--------------------------------------------------------------------------
    {
      if (points->get_volume() != total_points)
        return false;
      if (point_space != NULL)
      {
#ifdef DEBUG_LEGION
        assert(point_space->get_volume() == points->get_volume());
#endif
        IndexSpaceExpression *intersection =
          runtime->forest->intersect_index_spaces(point_space, points);
        return (intersection->get_volume() == total_points);
      }
      // Have to do this the hard way by looking up all the points
      ApEvent ready;
      const Domain point_domain = points->get_domain(ready, true/*need tight*/);
      if (ready.exists() && !ready.has_triggered_faultignorant())
        ready.wait_faultignorant();
      std::set<DomainPoint> known_points;
      {
        AutoLock i_lock(inst_lock,1,false/*exclusive*/);
        for (unsigned idx = 0; idx < instance_points.size(); idx++)
          known_points.insert(instance_points[idx]);
        for (std::map<DomainPoint,
                      std::pair<PhysicalInstance,unsigned> >::const_iterator
              it = remote_instances.begin(); it != remote_instances.end(); it++)
          known_points.insert(it->first);
      }
      std::vector<DomainPoint> unknown_points;
      for (Domain::DomainPointIterator itr(point_domain); itr; itr++)
      {
        if (known_points.find(*itr) != known_points.end())
          continue;
        unknown_points.push_back(*itr);
      }
      if (unknown_points.empty())
        return true;
      // Broadcast out a request for the remote instance
      std::vector<AddressSpaceID> child_spaces;
      collective_mapping->get_children(local_space, local_space, child_spaces);
      if (child_spaces.empty())
        return false;
      std::vector<RtEvent> ready_events;
      ready_events.reserve(child_spaces.size());
      for (std::vector<AddressSpaceID>::const_iterator it =
            child_spaces.begin(); it != child_spaces.end(); it++)
      {
        const RtUserEvent ready_event = Runtime::create_rt_user_event();
        Serializer rez; 
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(COLLECTIVE_REMOTE_INSTANCE_REQUEST);
          rez.serialize<size_t>(unknown_points.size());
          for (unsigned idx = 0; idx < unknown_points.size(); idx++)
            rez.serialize(unknown_points[idx]);
          rez.serialize(local_space);
          rez.serialize(ready_event);
        }
        runtime->send_collective_instance_message(*it, rez);
        ready_events.push_back(ready_event);
      }
      const RtEvent wait_on = Runtime::merge_events(ready_events);
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      {
        AutoLock i_lock(inst_lock,1,false/*exclusive*/);
        for (std::map<DomainPoint,
                      std::pair<PhysicalInstance,unsigned> >::const_iterator
              it = remote_instances.begin(); it != remote_instances.end(); it++)
          known_points.insert(it->first);
      }
      for (std::vector<DomainPoint>::const_iterator it =
            unknown_points.begin(); it != unknown_points.end(); it++)
        if (known_points.find(*it) == known_points.end())
          return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool CollectiveManager::contains_point(const DomainPoint &point) const
    //--------------------------------------------------------------------------
    {
      if (point_space != NULL)
        return point_space->contains_point(point);
      // Check the local points first since they are read-only at this point
      for (std::vector<DomainPoint>::const_iterator it =
            instance_points.begin(); it != instance_points.end(); it++)
        if ((*it) == point)
          return true;
      {
        AutoLock i_lock(inst_lock,1,false/*exclusive*/);
        std::map<DomainPoint,
                 std::pair<PhysicalInstance,unsigned> >::const_iterator finder =
          remote_instances.find(point);
        if (finder != remote_instances.end())
          return true;
      }
      // Broadcast out a request for this remote instance
      std::vector<AddressSpaceID> child_spaces;
      collective_mapping->get_children(local_space, local_space, child_spaces);
      if (child_spaces.empty())
        return false;
      std::vector<RtEvent> ready_events;
      ready_events.reserve(child_spaces.size());
      for (std::vector<AddressSpaceID>::const_iterator it =
            child_spaces.begin(); it != child_spaces.end(); it++)
      {
        const RtUserEvent ready_event = Runtime::create_rt_user_event();
        Serializer rez; 
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(COLLECTIVE_REMOTE_INSTANCE_REQUEST);
          rez.serialize<size_t>(1); // total number of points
          rez.serialize(point);
          rez.serialize(local_space);
          rez.serialize(ready_event);
        }
        runtime->send_collective_instance_message(*it, rez);
        ready_events.push_back(ready_event);
      }
      const RtEvent wait_on = Runtime::merge_events(ready_events);
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      AutoLock i_lock(inst_lock,1,false/*exclusive*/);
      return (remote_instances.find(point) != remote_instances.end());
    }

    //--------------------------------------------------------------------------
    bool CollectiveManager::is_first_local_point(const DomainPoint &point) const
    //--------------------------------------------------------------------------
    {
      // Check the local points first since they are read-only at this point
      for (unsigned idx = 0; idx < instance_points.size(); idx++)
        if (instance_points[idx] == point)
          return (idx == 0);
      {
        AutoLock i_lock(inst_lock,1,false/*exclusive*/);
        std::map<DomainPoint,
                 std::pair<PhysicalInstance,unsigned> >::const_iterator finder =
          remote_instances.find(point);
        if (finder != remote_instances.end())
          return (finder->second.second == 0);
      }
      // Broadcast out a request for this remote instance
      std::vector<AddressSpaceID> child_spaces;
      collective_mapping->get_children(local_space, local_space, child_spaces);
#ifdef DEBUG_LEGION
      assert(!child_spaces.empty());
#endif
      std::vector<RtEvent> ready_events;
      ready_events.reserve(child_spaces.size());
      for (std::vector<AddressSpaceID>::const_iterator it =
            child_spaces.begin(); it != child_spaces.end(); it++)
      {
        const RtUserEvent ready_event = Runtime::create_rt_user_event();
        Serializer rez; 
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(COLLECTIVE_REMOTE_INSTANCE_REQUEST);
          rez.serialize<size_t>(1); // total number of points
          rez.serialize(point);
          rez.serialize(local_space);
          rez.serialize(ready_event);
        }
        runtime->send_collective_instance_message(*it, rez);
        ready_events.push_back(ready_event);
      }
      const RtEvent wait_on = Runtime::merge_events(ready_events);
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      AutoLock i_lock(inst_lock,1,false/*exclusive*/);
      std::map<DomainPoint,
                 std::pair<PhysicalInstance,unsigned> >::const_iterator finder =
          remote_instances.find(point);
#ifdef DEBUG_LEGION
      assert(finder != remote_instances.end());
#endif
      return (finder->second.second == 0);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::find_or_forward_physical_instance(
     AddressSpaceID origin,std::set<DomainPoint> &points,RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(origin != local_space);
#endif
      std::map<DomainPoint,std::pair<PhysicalInstance,unsigned> > found_insts;
      for (unsigned idx = 0; idx < instance_points.size(); idx++)
      {
        std::set<DomainPoint>::iterator finder = 
          points.find(instance_points[idx]);
        if (finder == points.end())
          continue;
        found_insts[instance_points[idx]] = std::make_pair(instances[idx], idx);
        points.erase(finder);
        if (points.empty())
          break;
      }
      if (!points.empty())
      {
        AutoLock i_lock(inst_lock,1,false/*exclusive*/);
        for (std::set<DomainPoint>::iterator it =
              points.begin(); it != points.end(); /*nothing*/)
        {
          std::map<DomainPoint,
                   std::pair<PhysicalInstance,unsigned> >::const_iterator 
            finder = remote_instances.find(*it);
          if (finder != remote_instances.end())
          {
            found_insts.insert(*finder);
            std::set<DomainPoint>::iterator to_delete = it++;
            points.erase(to_delete);
          }
          else
            it++;
        }
      }
      std::vector<RtEvent> ready_events;
      // Send back anything that we found
      if (!found_insts.empty())
      {
        const RtUserEvent ready_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(COLLECTIVE_REMOTE_INSTANCE_RESPONSE);
          rez.serialize<size_t>(found_insts.size());
          for (std::map<DomainPoint,
                        std::pair<PhysicalInstance,unsigned>>::const_iterator
                it = found_insts.begin(); it != found_insts.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second.first);
            rez.serialize(it->second.second);
          }
          rez.serialize(ready_event);
        }
        runtime->send_collective_instance_message(origin, rez);
        ready_events.push_back(ready_event);
      }
      std::vector<AddressSpaceID> child_spaces;
      if (!points.empty())
        collective_mapping->get_children(origin, local_space, child_spaces);
      if (child_spaces.empty())
      {
#ifdef DEBUG_LEGION
        assert(ready_events.empty() || (ready_events.size() == 1));
#endif
        if (!ready_events.empty())
          Runtime::trigger_event(to_trigger, ready_events.back());
        else
          Runtime::trigger_event(to_trigger);
        return;
      }
      ready_events.reserve(ready_events.size() + child_spaces.size());
      for (unsigned idx = 0; idx < child_spaces.size(); idx++)
      {
        const RtUserEvent ready_event = Runtime::create_rt_user_event();
        Serializer rez; 
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(COLLECTIVE_REMOTE_INSTANCE_REQUEST);
          rez.serialize<size_t>(points.size());
          for (std::set<DomainPoint>::const_iterator it =
                points.begin(); it != points.end(); it++)
            rez.serialize(*it);
          rez.serialize(origin);
          rez.serialize(ready_event);
        }
        runtime->send_collective_instance_message(child_spaces[idx], rez);
        ready_events.push_back(ready_event);
      }
      Runtime::trigger_event(to_trigger, Runtime::merge_events(ready_events));
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::record_remote_physical_instances(
           const std::map<DomainPoint,
                          std::pair<PhysicalInstance,unsigned> > &new_instances)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
      for (std::map<DomainPoint,
                    std::pair<PhysicalInstance,unsigned> >::const_iterator it =
            new_instances.begin(); it != new_instances.end(); it++)
      {
        std::map<DomainPoint,
                 std::pair<PhysicalInstance,unsigned>>::const_iterator finder =
          remote_instances.find(it->first);
        if (finder == remote_instances.end())
          remote_instances.insert(*it);
#ifndef NDEBUG
        else
          assert(finder->second == it->second);
#endif
      }
#else
      remote_instances.insert(new_instances.begin(), new_instances.end()); 
#endif
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::record_point_instance(const DomainPoint &point,
                                                  PhysicalInstance instance)
    //--------------------------------------------------------------------------
    {
      const Memory mem = instance.get_location();
      MemoryManager *memory = runtime->find_memory_manager(mem);
#ifdef DEBUG_LEGION
      assert(memory->is_owner);
      assert((point_space == NULL) || point_space->contains_point(point));
#endif
      AutoLock i_lock(inst_lock);
      memories.push_back(memory);
      instances.push_back(instance);
      instance_points.push_back(point);
    }

    //--------------------------------------------------------------------------
    ApEvent CollectiveManager::get_use_event(ApEvent user) const
    //--------------------------------------------------------------------------
    {
      return unique_event;
    }

    //--------------------------------------------------------------------------
    RtEvent CollectiveManager::get_instance_ready_event(void) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    PhysicalInstance CollectiveManager::get_instance(const DomainPoint &p) const
    //--------------------------------------------------------------------------
    {
      // Check the local points first since they are read-only at this point
      for (unsigned idx = 0; idx < instance_points.size(); idx++)
        if (instance_points[idx] == p)
          return instances[idx];
      {
        AutoLock i_lock(inst_lock,1,false/*exclusive*/);
        std::map<DomainPoint,
                 std::pair<PhysicalInstance,unsigned> >::const_iterator finder =
          remote_instances.find(p);
        if (finder != remote_instances.end())
          return finder->second.first;
      }
      // Broadcast out a request for this remote instance
      std::vector<AddressSpaceID> child_spaces;
      collective_mapping->get_children(local_space, local_space, child_spaces);
#ifdef DEBUG_LEGION
      assert(!child_spaces.empty());
#endif
      std::vector<RtEvent> ready_events;
      ready_events.reserve(child_spaces.size());
      for (std::vector<AddressSpaceID>::const_iterator it =
            child_spaces.begin(); it != child_spaces.end(); it++)
      {
        const RtUserEvent ready_event = Runtime::create_rt_user_event();
        Serializer rez; 
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(COLLECTIVE_REMOTE_INSTANCE_REQUEST);
          rez.serialize<size_t>(1); // total number of points
          rez.serialize(p);
          rez.serialize(local_space);
          rez.serialize(ready_event);
        }
        runtime->send_collective_instance_message(*it, rez);
        ready_events.push_back(ready_event);
      }
      const RtEvent wait_on = Runtime::merge_events(ready_events);
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      AutoLock i_lock(inst_lock,1,false/*exclusive*/);
      std::map<DomainPoint,
               std::pair<PhysicalInstance,unsigned> >::const_iterator finder =
        remote_instances.find(p);
#ifdef DEBUG_LEGION
      assert(finder != remote_instances.end());
#endif
      return finder->second.first;
    }

    //--------------------------------------------------------------------------
    PointerConstraint CollectiveManager::get_pointer_constraint(
                                                   const DomainPoint &key) const
    //--------------------------------------------------------------------------
    {
      const PhysicalInstance instance = get_instance(key);
      void *inst_ptr = instance.pointer_untyped(0/*offset*/, 0/*elem size*/);
      return PointerConstraint(instance.get_location(), uintptr_t(inst_ptr));
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
        activate_collective(mutator);
      else
        send_remote_gc_increment(
            collective_mapping->get_parent(owner_space, local_space), mutator);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
        deactivate_collective(mutator);
      else
        send_remote_gc_decrement(
            collective_mapping->get_parent(owner_space, local_space), mutator);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
        validate_collective(mutator);
      else
        send_remote_valid_increment(
            collective_mapping->get_parent(owner_space, local_space), mutator);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
        invalidate_collective(mutator);
      else
        send_remote_valid_decrement(
            collective_mapping->get_parent(owner_space, local_space), mutator);
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
    void CollectiveManager::activate_collective(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    { 
      std::vector<AddressSpaceID> right_spaces;
      collective_mapping->get_children(owner_space, local_space, right_spaces);
      for (std::vector<AddressSpaceID>::const_iterator it = 
            right_spaces.begin(); it != right_spaces.end(); it++)
      {
        RtUserEvent done_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(COLLECTIVE_ACTIVATE_MESSAGE);
          rez.serialize(done_event);
        }
        runtime->send_collective_instance_message(*it, rez);
        mutator->record_reference_mutation_effect(done_event);
      }
      for (std::vector<MemoryManager*>::const_iterator it = 
            memories.begin(); it != memories.end(); it++)
        (*it)->activate_instance(this);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::deactivate_collective(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    { 
      std::vector<AddressSpaceID> right_spaces;
      collective_mapping->get_children(owner_space, local_space, right_spaces);
      for (std::vector<AddressSpaceID>::const_iterator it = 
            right_spaces.begin(); it != right_spaces.end(); it++)
      {
        RtUserEvent done_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(COLLECTIVE_DEACTIVATE_MESSAGE);
          rez.serialize(done_event);
        }
        runtime->send_collective_instance_message(*it, rez);
        mutator->record_reference_mutation_effect(done_event);
      }
      for (std::vector<MemoryManager*>::const_iterator it = 
            memories.begin(); it != memories.end(); it++)
        (*it)->deactivate_instance(this);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::validate_collective(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    { 
      std::vector<AddressSpaceID> right_spaces;
      collective_mapping->get_children(owner_space, local_space, right_spaces);
      for (std::vector<AddressSpaceID>::const_iterator it = 
            right_spaces.begin(); it != right_spaces.end(); it++)
      {
        RtUserEvent done_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(COLLECTIVE_VALIDATE_MESSAGE);
          rez.serialize(done_event);
        }
        runtime->send_collective_instance_message(*it, rez);
        mutator->record_reference_mutation_effect(done_event);
      }
      for (std::vector<MemoryManager*>::const_iterator it = 
            memories.begin(); it != memories.end(); it++)
        (*it)->validate_instance(this);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::invalidate_collective(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    { 
      std::vector<AddressSpaceID> right_spaces;
      collective_mapping->get_children(owner_space, local_space, right_spaces);
      for (std::vector<AddressSpaceID>::const_iterator it = 
            right_spaces.begin(); it != right_spaces.end(); it++)
      {
        RtUserEvent done_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(COLLECTIVE_INVALIDATE_MESSAGE);
          rez.serialize(done_event);
        }
        runtime->send_collective_instance_message(*it, rez);
        mutator->record_reference_mutation_effect(done_event);
      }
      prune_gc_events();
      for (std::vector<MemoryManager*>::const_iterator it = 
            memories.begin(); it != memories.end(); it++)
        (*it)->validate_instance(this);
    }

    //--------------------------------------------------------------------------
    bool CollectiveManager::acquire_instance(ReferenceSource source, 
                                            ReferenceMutator *mutator,
                                            const DomainPoint &collective_point,
                                            AddressSpaceID *remote_space)
    //--------------------------------------------------------------------------
    {
      // TODO: implement this
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::perform_deletion(RtEvent deferred_event)
    //--------------------------------------------------------------------------
    {
      perform_delete(deferred_event, true/*left*/);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::force_deletion(void)
    //--------------------------------------------------------------------------
    {
      force_delete(true/*left*/);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::set_garbage_collection_priority(MapperID mapper_id, 
                  Processor proc, GCPriority priority, const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      const PhysicalInstance inst = get_instance(point);
      MemoryManager *memory_manager =
        runtime->find_memory_manager(inst.get_location());
      memory_manager->set_garbage_collection_priority(this, mapper_id,
                                                      proc, priority);
    }

    //--------------------------------------------------------------------------
    RtEvent CollectiveManager::attach_external_instance(void)
    //--------------------------------------------------------------------------
    {
      // At this point the points should all be filled in so these
      // data structures are all read-only
#ifdef DEBUG_LEGION
      assert(!memories.empty());
#endif
      if (memories.size() > 1)
      {
        // Need to make sure we don't duplicate memory attaches
        // in the case where we have multiple instances in the
        // same memory
        std::set<MemoryManager*> unique_memories;
        std::vector<RtEvent> ready_events;
        for (unsigned idx = 0; idx < memories.size(); idx++)
        {
          MemoryManager *manager = memories[idx];
          if (unique_memories.find(manager) != unique_memories.end())
            continue;
          unique_memories.insert(manager);
          const RtEvent ready = manager->attach_external_instance(this);
          if (ready.exists())
            ready_events.push_back(ready);
        }
        if (!ready_events.empty())
          return Runtime::merge_events(ready_events);
        else
          return RtEvent::NO_RT_EVENT;
      }
      else
        return memories.back()->attach_external_instance(this);
    }

    //--------------------------------------------------------------------------
    RtEvent CollectiveManager::detach_external_instance(void)
    //--------------------------------------------------------------------------
    {
 #ifdef DEBUG_LEGION
      assert(!memories.empty());
#endif
      if (memories.size() > 1)
      {
        // Need to make sure we don't duplicate memory attaches
        // in the case where we have multiple instances in the
        // same memory
        std::set<MemoryManager*> unique_memories;
        std::vector<RtEvent> ready_events;
        for (unsigned idx = 0; idx < memories.size(); idx++)
        {
          MemoryManager *manager = memories[idx];
          if (unique_memories.find(manager) != unique_memories.end())
            continue;
          unique_memories.insert(manager);
          const RtEvent ready = manager->detach_external_instance(this);
          if (ready.exists())
            ready_events.push_back(ready);
        }
        if (!ready_events.empty())
          return Runtime::merge_events(ready_events);
        else
          return RtEvent::NO_RT_EVENT;
      }
      else
        return memories.back()->detach_external_instance(this);      
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

#ifdef LEGION_GPU_REDUCTIONS
    //--------------------------------------------------------------------------
    bool CollectiveManager::is_gpu_visible(PhysicalManager *other) const
    //--------------------------------------------------------------------------
    {
      // TODO: implement this
      return false;
    }
    
    //--------------------------------------------------------------------------
    ReductionView* CollectiveManager::find_or_create_shadow_reduction(
                                    unsigned fidx, ReductionOpID redop, 
                                    AddressSpaceID request_space, UniqueID opid)
    //--------------------------------------------------------------------------
    {
      // TODO: implement this
      return NULL;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::record_remote_shadow_reduction(unsigned fidx,
                                       ReductionOpID redop, ReductionView *view)
    //--------------------------------------------------------------------------
    {
      // TODO: implement this
    }
#endif // LEGION_GPU_REDUCTIONS

    //--------------------------------------------------------------------------
    void CollectiveManager::perform_delete(RtEvent deferred_event, bool left)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID left_space =
        collective_mapping->get_parent(owner_space, local_space);
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
          // If we make it here we are the first ones so do the deletion
          collective_deletion(deferred_event);
          std::vector<AddressSpaceID> right_spaces;
          collective_mapping->get_children(owner_space, local_space,
                                           right_spaces);
          for (std::vector<AddressSpaceID>::const_iterator it = 
                right_spaces.begin(); it != right_spaces.end(); it++)
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(COLLECTIVE_PERFORM_DELETE_MESSAGE);
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
            rez.serialize(COLLECTIVE_PERFORM_DELETE_MESSAGE);
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
        std::vector<AddressSpaceID> right_spaces;
        collective_mapping->get_children(owner_space, local_space,right_spaces);
        // If we have no right users send back messages
        if (right_spaces.empty())
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(COLLECTIVE_FINALIZE_MESSAGE);
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
              rez.serialize(COLLECTIVE_PERFORM_DELETE_MESSAGE);
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
      const AddressSpaceID left_space =
        collective_mapping->get_parent(owner_space, local_space);
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
          std::vector<AddressSpaceID> right_spaces;
          collective_mapping->get_children(owner_space, local_space,
                                           right_spaces);
          for (std::vector<AddressSpaceID>::const_iterator it = 
                right_spaces.begin(); it != right_spaces.end(); it++)
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(COLLECTIVE_FORCE_DELETE_MESSAGE);
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
            rez.serialize(COLLECTIVE_FORCE_DELETE_MESSAGE);
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
        std::vector<AddressSpaceID> right_spaces;
        collective_mapping->get_children(owner_space, local_space,right_spaces);
        // If we have no right users send back messages
        if (right_spaces.empty())
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(COLLECTIVE_FINALIZE_MESSAGE);
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
              rez.serialize(COLLECTIVE_FORCE_DELETE_MESSAGE);
              rez.serialize(false/*left*/);
            }
            runtime->send_collective_instance_message(*it, rez);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    bool CollectiveManager::finalize_message(void)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
      assert(finalize_messages <
              collective_mapping->count_children(owner_space, local_space));
#endif
      if (++finalize_messages ==
          collective_mapping->count_children(owner_space, local_space))
      {
        const AddressSpaceID left_space =
          collective_mapping->get_parent(owner_space, local_space);
        if (left_space != local_space)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(COLLECTIVE_FINALIZE_MESSAGE);
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
        for (unsigned idx = 0; idx < instances.size(); idx++)
        {
          log_garbage.spew("Deleting collective instance " IDFMT " in memory "
                       IDFMT "", instances[idx].id, memories[idx]->memory.id);
          instances[idx].destroy(serdez_fields, deferred_event);
        }
      }
      else
      {
        for (unsigned idx = 0; idx < instances.size(); idx++)
        {
          log_garbage.spew("Deleting collective instance " IDFMT " in memory "
                       IDFMT "", instances[idx].id, memories[idx]->memory.id);
          instances[idx].destroy(deferred_event);
        }
      }
#ifdef LEGION_MALLOC_INSTANCES
      if (is_external_instance())
      {
        for (unsigned idx = 0; idx < instances.size(); idx++)
          memories[idx]->free_legion_instance(instances[idx], deferred_event);
      }
#endif
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
#ifdef DEBUG_LEGION
        assert(pending_views.empty());
#endif
        context_views.clear();
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
        for (unsigned idx = 0; idx < instances.size(); idx++)
        {
          log_garbage.spew("Force deleting collective instance " IDFMT 
           " in memory " IDFMT "", instances[idx].id, memories[idx]->memory.id);
          instances[idx].destroy(serdez_fields);
        }
      }
      else
      {
        for (unsigned idx = 0; idx < instances.size(); idx++)
        {
          log_garbage.spew("Force deleting collective instance " IDFMT
           " in memory " IDFMT "", instances[idx].id, memories[idx]->memory.id);
          instances[idx].destroy();
        }
      }
#ifdef LEGION_MALLOC_INSTANCES
      if (is_external_instance())
      {
        for (unsigned idx = 0; idx < instances.size(); idx++)
          memories[idx]->free_legion_instance(instances[idx], 
                                              RtEvent::NO_RT_EVENT);
      }
#endif
#endif
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
      // TODO: implement this
      assert(false);
    }

    //--------------------------------------------------------------------------
    ApEvent CollectiveManager::register_collective_user(InstanceView *view, 
                                            const RegionUsage &usage,
                                            const FieldMask &user_mask,
                                            IndexSpaceNode *expr,
                                            const UniqueID op_id,
                                            const size_t op_ctx_index,
                                            const unsigned index,
                                            ApEvent term_event,
                                            RtEvent collect_event,
                                            std::set<RtEvent> &applied_events,
                                            const CollectiveMapping *mapping,
                                            const PhysicalTraceInfo &trace_info,
                                            const AddressSpaceID source,
                                            bool symbolic)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapping != NULL);
      assert(collective_mapping != NULL);
      assert(collective_mapping->contains(local_space));
      // CollectiveMapping for the analyses should all align with 
      // the CollectiveMapping for the collective manager
      assert((mapping == collective_mapping) ||
          ((*mapping) == (*collective_mapping)));
      assert(term_event.exists());
#endif
      // This function does a parallel rendezvous across all the analyses
      // trying to use this collective instance for an operation so that only
      // one of them will actually do the view analysis, but it will seem to
      // all of them like they actually did it. To that end, this function 
      // must do five important things:
      // 1. Make sure that the precondition return event is broadcast out
      //    to all the participants in the rendezvous
      // 2. Make sure that the recorded event in the view is a merge of the 
      //    term_event from all of the participants
      // 3. SUBTLE!!! Don't start the view analysis until all the participants
      //    have arrived because that is the only way to know that all the 
      //    copy effects from analyses by any participant have been recorded
      //    by the view.
      // 4. Do NOT block in this function call or you can risk deadlock because
      //    we might be doing several of these calls for a region requirement
      //    on different instances and the orders might vary on each node.
      // 5. In the case of tracing we don't want to build any explicit event
      //    trees, so instead we will return the actual event precondition
      //    back out to all the particpants. Furthermore, for the term events,
      //    we will have all of them arrive on a barrier that is shared across
      //    all the participants and that the trace can reuse.
      
      // First let's figure out which node is ultimately going to do analysis
      // If the logical owner of the view is contained in the mapping then
      // that node will be the one to do the analysis for locality reasons
      // If the logical owner of the view is not in the mapping then we'll
      // pick whichever node is closest in the collective mapping to the 
      // logical owner (assuming some degree of locality)
      // Note this code assumes that logical_owner on the view is const
      // which it is declared to be today
      const AddressSpaceID origin = mapping->contains(view->logical_owner) ?
        view->logical_owner : mapping->find_nearest(view->logical_owner);
      // The unique tag for the rendezvous is our context ID which will be
      // the same across all points and the index of our region requirement
      ApEvent result;
      RtUserEvent local_applied;
      std::map<ApEvent,PhysicalTraceInfo*> term_events;
      const std::pair<size_t,unsigned> tag(op_ctx_index, index);
      {
        AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
        assert(!instances.empty());
#endif
        // Check to see if we're the first one to arrive on this node
        std::map<std::pair<size_t,unsigned>,UserRendezvous>::iterator
          finder = rendezvous_users.find(tag);
        if (finder == rendezvous_users.end())
        {
          // If we are then make the record for knowing when we've seen
          // all the expected arrivals
          finder = rendezvous_users.insert(
              std::make_pair(tag,UserRendezvous())).first; 
          UserRendezvous &rendezvous = finder->second;
          // Count how many expected arrivals we have
          rendezvous.remaining_local_arrivals = instances.size();
          rendezvous.remaining_remote_arrivals =
            mapping->count_children(origin, local_space);
          rendezvous.applied = Runtime::create_rt_user_event();
          rendezvous.deferred = RtUserEvent::NO_RT_USER_EVENT;
        }
#ifdef DEBUG_LEGION
        assert(finder->second.term_events.find(term_event) ==
                finder->second.term_events.end());
#endif
        finder->second.term_events[term_event] = trace_info.recording ?
          new PhysicalTraceInfo(trace_info) : NULL;
        // Record the applied events
        applied_events.insert(finder->second.applied);
        // See if we need to make our own result event or not
        if (trace_info.recording)
        {
          // Always make a new ready event in the case of tracing
          const ApUserEvent local_ready =
            Runtime::create_ap_user_event(&trace_info);
          // Can re-use the trace-info from the term events, the
          // term events will always own the trace info
          finder->second.ready_events[local_ready] =
            finder->second.term_events[term_event];
          result = local_ready;
        }
        else
        {
          if (finder->second.ready_events.empty())
          {
            // We're first, so make the ready event
            const ApUserEvent local_ready = 
              Runtime::create_ap_user_event(&trace_info);
            // no trace info required in this case
            finder->second.ready_events[local_ready] = NULL; 
          }
          result = finder->second.ready_events.begin()->first;
        }
#ifdef DEBUG_LEGION
        assert(finder->second.remaining_local_arrivals > 0);
#endif
        // If we're not done or we're not the owner then we can simply
        // return the ready event since there is nothing for us to do
        if (--finder->second.remaining_local_arrivals == 0)
        {
          // Check to see if we're the origin or not
          if (local_space == origin)
          {
            // We're the origin node so we're the node that has to
            // actually call back into the view to do the analysis
            // See if we're waiting for anymore remote arrivals
            if (finder->second.remaining_remote_arrivals > 0)
            {
              // Launch a meta-task to capture the arguments for
              // when we need to actually perform the analysis
#ifdef DEBUG_LEGION
              assert(!finder->second.deferred.exists()); 
#endif
              finder->second.deferred = Runtime::create_rt_user_event();
              PhysicalTraceInfo *info = trace_info.recording ? 
                finder->second.term_events[term_event] : 
                new PhysicalTraceInfo(trace_info);
              DeferCollectiveRendezvousArgs args(this, view, usage, user_mask,
                 expr, op_id, op_ctx_index, index, finder->second.term_events,
                 collect_event, finder->second.applied, info, origin, source,
                 symbolic, applied_events);
              runtime->issue_runtime_meta_task(args,
                  LG_LATENCY_DEFERRED_PRIORITY, finder->second.deferred);
              return result;
            }
            // Otherwise we can fall through and perform the analysis now
            // Grab the term events so we can record them for later
            term_events.swap(finder->second.term_events);
#ifdef DEBUG_LEGION
            // If we're tracing we should have exactly as many term events as
            // we do instances on the local node
            assert(!trace_info.recording ||
                (term_events.size() == instances.size()));
            assert(finder->second.applied.exists());
#endif
            local_applied = finder->second.applied;
          }
          else
          {
            // Not the origin, so see if our arrivals are all done
            // If they are then we can send the message
            if (finder->second.remaining_remote_arrivals == 0)
            {
              // Send the message to the parent in the rendezvous tree
              ApEvent merged_term;
              ApUserEvent to_trigger;
              RtUserEvent applied;
              // Only need the merged event in the case we're not tracing
              // In the tracing case we'll get a response message with a barrier
              if (!trace_info.recording)
              {
                std::vector<ApEvent> terms;
                terms.resize(finder->second.term_events.size());
                for (std::map<ApEvent,PhysicalTraceInfo*>::const_iterator it =
                      finder->second.term_events.begin(); it !=
                      finder->second.term_events.end(); it++)
                {
#ifdef DEBUG_LEGION
                  assert(it->second == NULL);
#endif
                  terms.push_back(it->first);
                }
                merged_term = Runtime::merge_events(&trace_info, terms);
#ifdef DEBUG_LEGION
                assert(finder->second.ready_events.size() == 1);
#endif
                to_trigger = finder->second.ready_events.begin()->first;
                applied = finder->second.applied;
              }
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
                rez.serialize(COLLECTIVE_RENDEZVOUS_REQUEST);
                rez.serialize(origin);
                rez.serialize(op_ctx_index);
                rez.serialize(index);
                rez.serialize(merged_term);
                rez.serialize(to_trigger); 
                rez.serialize(applied);
              }
              const AddressSpaceID parent = 
                mapping->get_parent(origin, local_space);
              runtime->send_collective_instance_message(parent, rez);
              if (!trace_info.recording)
                // Not expecting a response for tracing so we can
                // remove the entry from the rendezvous_users
                rendezvous_users.erase(finder);
              // Otheriwse we're expecting a callback to give us the name
              // of the barrier to use for tracing so we can just leave
              // the rendezvous in-place for when it comes back
            }
            return result;
          }
        }
        else
          return result;
      }
      // If we get here then we're the origin so do the analysis immediately
#ifdef DEBUG_LEGION
      assert(local_space == origin);
#endif
      finalize_collective_user(view, usage, user_mask, expr, op_id,
          op_ctx_index, index, term_events, collect_event, local_applied,
          trace_info, origin, source, symbolic);
      // Clean-up the physical trace infos we made if we're tracing
      if (trace_info.recording)
      {
        for (std::map<ApEvent,PhysicalTraceInfo*>::iterator it =
              term_events.begin(); it != term_events.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(it->second != NULL);
#endif
          delete it->second; 
        }
      }
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      else
      {
        for (std::map<ApEvent,PhysicalTraceInfo*>::iterator it =
              term_events.begin(); it != term_events.end(); it++)
          assert(it->second == NULL);
      }
#endif
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::finalize_collective_user(
                              InstanceView *view, 
                              const RegionUsage &usage,
                              const FieldMask &user_mask,
                              IndexSpaceNode *expr,
                              const UniqueID op_id,
                              const size_t op_ctx_index,
                              const unsigned index,
                              std::map<ApEvent,PhysicalTraceInfo*> &term_events,
                              RtEvent collect_event,
                              RtUserEvent applied,
                              const PhysicalTraceInfo &trace_info,
                              const AddressSpaceID origin,
                              const AddressSpaceID source,
                              const bool symbolic)
    //--------------------------------------------------------------------------
    {
      // Call back into the view but without a collective mapping this time
      std::set<RtEvent> local_applied;
      ApEvent result;
      ApBarrier common_term;
      ShardID owner_shard = 0;
      if (trace_info.recording)
      {
        // Create a barrier for use in the trace as the common term event
        common_term = ApBarrier(Realm::Barrier::create_barrier(total_points));
        owner_shard = 
          trace_info.record_managed_barrier(common_term, total_points);
        result = view->register_user(usage, user_mask, expr, op_id,
            op_ctx_index, index, common_term, collect_event, local_applied,
            NULL/*collective mapping*/, trace_info, source, symbolic);
        for (std::map<ApEvent,PhysicalTraceInfo*>::const_iterator it =
              term_events.begin(); it != term_events.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(it->second != NULL);
#endif
          Runtime::phase_barrier_arrive(common_term, 1/*count*/, it->first);
          // Record the barrier with the trace
          it->second->record_barrier_arrival(common_term, it->first, 1/*count*/,
                                             local_applied, owner_shard);
        }
      }
      else
      {
        std::vector<ApEvent> terms;
        terms.reserve(term_events.size());
        for (std::map<ApEvent,PhysicalTraceInfo*>::const_iterator it =
              term_events.begin(); it != term_events.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(it->second == NULL);
#endif
          terms.push_back(it->first);
        }
        const ApEvent merged_term = Runtime::merge_events(&trace_info, terms);
        result = view->register_user(usage, user_mask, expr, op_id,
            op_ctx_index, index, merged_term, collect_event, local_applied,
            NULL/*collective mapping*/, trace_info, source, symbolic);
      }
      const std::pair<size_t,unsigned> tag(op_ctx_index, index);
      // Retake the lock and do what we need to do to send the results
      // back to the other participants
      AutoLock i_lock(inst_lock);
      std::map<std::pair<size_t,unsigned>,UserRendezvous>::iterator finder =
        rendezvous_users.find(tag);
#ifdef DEBUG_LEGION
      assert(finder != rendezvous_users.end());
      assert(!finder->second.deferred.exists());
      assert(finder->second.term_events.empty());
#endif
      if (!local_applied.empty())
        Runtime::trigger_event(finder->second.applied, 
            Runtime::merge_events(local_applied));
      else
        Runtime::trigger_event(finder->second.applied);
      if (trace_info.recording)
      {
#ifdef DEBUG_LEGION
        assert(common_term.exists());
#endif
        // Send the result message back with the barrier and the ready event
        std::vector<AddressSpaceID> children;
        collective_mapping->get_children(origin, local_space, children);
        if (!children.empty())
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(COLLECTIVE_RENDEZVOUS_RESPONSE);
            rez.serialize(origin);
            rez.serialize(op_ctx_index);
            rez.serialize(index);
            rez.serialize(result);
            rez.serialize(common_term);
            rez.serialize(owner_shard);
            rez.serialize(finder->second.applied);
          }
          for (std::vector<AddressSpaceID>::const_iterator it =
                children.begin(); it != children.end(); it++)
            runtime->send_collective_instance_message(*it, rez);
        }
        // Trigger any other events with the precondition
        for (std::map<ApUserEvent,PhysicalTraceInfo*>::iterator it =
              finder->second.ready_events.begin(); it !=
              finder->second.ready_events.end(); it++)
          Runtime::trigger_event(it->second, it->first, result);
      }
      else // trigger the broadcast tree for the ready event
      {
#ifdef DEBUG_LEGION
        assert(!common_term.exists());
        assert(finder->second.ready_events.size() == 1);
#endif
        std::map<ApUserEvent,PhysicalTraceInfo*>::iterator first =
          finder->second.ready_events.begin();
        Runtime::trigger_event(first->second, first->first, result);
        result = first->first;
      }
      rendezvous_users.erase(finder);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::process_rendezvous_request(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      AddressSpaceID origin;
      derez.deserialize(origin);
      std::pair<size_t,unsigned> tag;
      derez.deserialize(tag.first);
      derez.deserialize(tag.second);
      ApEvent remote_term;
      derez.deserialize(remote_term);
      ApUserEvent remote_ready;
      derez.deserialize(remote_ready);
      RtUserEvent remote_applied;
      derez.deserialize(remote_applied);
      const bool recording = !remote_ready.exists();

      AutoLock i_lock(inst_lock);
      // Check to see if we have an entry yet or not
      std::map<std::pair<size_t,unsigned>,UserRendezvous>::iterator finder =
        rendezvous_users.find(tag);
      if (finder == rendezvous_users.end())
      {
        finder = rendezvous_users.insert(
            std::make_pair(tag,UserRendezvous())).first; 
        UserRendezvous &rendezvous = finder->second;
        // Count how many expected arrivals we have
        rendezvous.remaining_local_arrivals = instances.size();
        rendezvous.remaining_remote_arrivals =
          collective_mapping->count_children(origin, local_space);
        rendezvous.applied = Runtime::create_rt_user_event();
        rendezvous.deferred = RtUserEvent::NO_RT_USER_EVENT;
      }
      if (!recording)
      {
        finder->second.term_events[remote_term] = NULL;
        // If we don't have a ready event yet then make one
        if (finder->second.ready_events.empty())
        {
          const ApUserEvent local_ready = Runtime::create_ap_user_event(NULL);
          finder->second.ready_events[local_ready] = NULL;
          Runtime::trigger_event(NULL, remote_ready, local_ready);
        }
        else
          Runtime::trigger_event(NULL, remote_ready,
              finder->second.ready_events.begin()->first);
        Runtime::trigger_event(remote_applied, finder->second.applied);
      }
#ifdef DEBUG_LEGION
      assert(finder->second.remaining_remote_arrivals > 0);
#endif
      if ((--finder->second.remaining_remote_arrivals == 0) &&
          (finder->second.remaining_local_arrivals == 0))
      {
        // See if we're the origin or not
        if (local_space != origin)
        {
          if (!recording)
          {
            std::vector<ApEvent> terms;
            terms.reserve(finder->second.term_events.size());
            for (std::map<ApEvent,PhysicalTraceInfo*>::const_iterator it =
                  finder->second.term_events.begin(); it !=
                  finder->second.term_events.end(); it++)
            {
#ifdef DEBUG_LEGION
              assert(it->second == NULL); 
#endif
              terms.push_back(it->first);
            }
            remote_term = Runtime::merge_events(NULL, terms);
#ifdef DEBUG_LEGION
            assert(finder->second.ready_events.size() == 1);
#endif
            remote_ready = finder->second.ready_events.begin()->first;
            remote_applied = finder->second.applied;
            // We can prune this entry out now since we know we're
            // not going to get a response
            rendezvous_users.erase(finder);
          }
          // Not the origin so continue sending to the next parent
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(COLLECTIVE_RENDEZVOUS_REQUEST);
            rez.serialize(origin);
            rez.serialize(tag.first);
            rez.serialize(tag.second);
            rez.serialize(remote_term);
            rez.serialize(remote_ready);
            rez.serialize(remote_applied);
          }
          const AddressSpaceID parent = 
            collective_mapping->get_parent(origin, local_space);
          runtime->send_collective_instance_message(parent, rez);
        }
        else
        {
#ifdef DEBUG_LEGION
          // Better have a deferred event in this case for us to trigger
          assert(finder->second.deferred.exists());
#endif
          Runtime::trigger_event(finder->second.deferred);
        }
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::process_rendezvous_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      AddressSpaceID origin;
      derez.deserialize(origin);
      std::pair<size_t,unsigned> tag;
      derez.deserialize(tag.first);
      derez.deserialize(tag.second);
      ApEvent ready;
      derez.deserialize(ready);
      ApBarrier common_term;
      derez.deserialize(common_term);
      ShardID owner_shard;
      derez.deserialize(owner_shard);
      RtEvent applied;
      derez.deserialize(applied);

      // Forward this on to any children first
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(origin, local_space, children);
      if (!children.empty())
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(COLLECTIVE_RENDEZVOUS_RESPONSE);
          rez.serialize(origin);
          rez.serialize(tag.first);
          rez.serialize(tag.second);
          rez.serialize(ready);
          rez.serialize(common_term);
          rez.serialize(owner_shard);
          // Get our local term event here for scalability
          // rather than passing along the applied event
          AutoLock i_lock(inst_lock,1,false/*exclusive*/);
          std::map<std::pair<size_t,unsigned>,UserRendezvous>::const_iterator
            finder = rendezvous_users.find(tag);
#ifdef DEBUG_LEGION
          assert(finder != rendezvous_users.end());
#endif
          rez.serialize(finder->second.applied);
        }
        for (std::vector<AddressSpaceID>::const_iterator it =
              children.begin(); it != children.end(); it++)
          runtime->send_collective_instance_message(*it, rez);
      }
      // Now take the lock and do our local updates
      AutoLock i_lock(inst_lock);
      std::map<std::pair<size_t,unsigned>,UserRendezvous>::iterator finder =
        rendezvous_users.find(tag);
#ifdef DEBUG_LEGION
      assert(finder != rendezvous_users.end());
      assert(finder->second.term_events.size() == instances.size());
      assert(finder->second.ready_events.size() == instances.size());
#endif
      // Do the ready events first so we can delete the trace infos
      // when we do the term_events
      for (std::map<ApUserEvent,PhysicalTraceInfo*>::const_iterator it =
            finder->second.ready_events.begin(); it !=
            finder->second.ready_events.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->second != NULL);
#endif
        Runtime::trigger_event(it->second, it->first, ready);
      }
      std::set<RtEvent> local_applied;
      for (std::map<ApEvent,PhysicalTraceInfo*>::iterator it =
            finder->second.term_events.begin(); it !=
            finder->second.term_events.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->second != NULL);
#endif
        Runtime::phase_barrier_arrive(common_term, 1/*count*/, it->first);
        it->second->record_barrier_arrival(common_term, it->first, 1/*count*/,
                                           local_applied, owner_shard);
        // Now we can delete the trace info
        delete it->second;
      }
      if (!local_applied.empty())
      {
        local_applied.insert(applied);
        Runtime::trigger_event(finder->second.applied,
            Runtime::merge_events(local_applied));
      }
      else
        Runtime::trigger_event(finder->second.applied, applied);
      rendezvous_users.erase(finder);
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
        rez.serialize(total_points);
        if (point_space == NULL)
          rez.serialize(IndexSpace::NO_SPACE);
        else
          rez.serialize(point_space->handle);
        collective_mapping->pack(rez);
        rez.serialize(instance_footprint);
        // No need for a reference here since we know we'll continue holding it
        instance_domain->pack_expression(rez, target);
        rez.serialize(field_space_node->handle);
        rez.serialize(tree_id);
        rez.serialize(redop);
        rez.serialize(unique_event.id);
        layout->pack_layout_description(rez, target);
      }
      context->runtime->send_collective_instance_manager(target, rez);
      update_remote_instances(target);
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
      size_t total_points;
      derez.deserialize(total_points);
      IndexSpace points_handle;
      derez.deserialize(points_handle);
      RtEvent points_ready;
      IndexSpaceNode *point_space = points_handle.exists() ?
        runtime->forest->get_node(points_handle, &points_ready) : NULL; 
      size_t total_spaces;
      derez.deserialize(total_spaces);
      CollectiveMapping *mapping = new CollectiveMapping(derez, total_spaces);
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
      // Little cheat here on the barrier
      ApBarrier unique_barrier;
      derez.deserialize(unique_barrier.id);
      LayoutConstraintID layout_id;
      derez.deserialize(layout_id);
      RtEvent layout_ready;
      LayoutConstraints *constraints = 
        runtime->find_layout_constraints(layout_id, 
                    false/*can fail*/, &layout_ready);
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
              total_points, mapping, inst_footprint, inst_domain, pending,
              handle, tree_id, layout_id, unique_barrier, redop, piece_list,
              piece_list_size, source);
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
          total_points, mapping, inst_footprint, inst_domain, piece_list,
          piece_list_size,space_node,tree_id,constraints,unique_barrier,redop);
    }

    //--------------------------------------------------------------------------
    CollectiveManager::DeferCollectiveManagerArgs::DeferCollectiveManagerArgs(
            DistributedID d, AddressSpaceID own, IndexSpace points, size_t tot,
            CollectiveMapping *map, size_t f, IndexSpaceExpression *lx, 
            const PendingRemoteExpression &p, FieldSpace h, RegionTreeID tid,
            LayoutConstraintID l, ApBarrier use, ReductionOpID r,
            const void *pl, size_t pl_size, const AddressSpaceID src)
      : LgTaskArgs<DeferCollectiveManagerArgs>(implicit_provenance),
        did(d), owner(own), point_space(points), total_points(tot),
        mapping(map), footprint(f), local_expr(lx), pending(p), handle(h),
        tree_id(tid), layout_id(l), use_barrier(use), redop(r), piece_list(pl),
        piece_list_size(pl_size), source(src)
    //--------------------------------------------------------------------------
    {
      mapping->add_reference();
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
          dargs->total_points, dargs->mapping, dargs->footprint, inst_domain,
          dargs->piece_list, dargs->piece_list_size, space_node, dargs->tree_id,
          constraints, dargs->use_barrier, dargs->redop);
      // Remove the local expression reference if necessary
      if ((dargs->local_expr != NULL) &&
          dargs->local_expr->remove_base_expression_reference(META_TASK_REF))
        delete dargs->local_expr;
      if (dargs->mapping->remove_reference())
        delete dargs->mapping;
    }

    //--------------------------------------------------------------------------
    CollectiveManager::DeferCollectiveRendezvousArgs::
      DeferCollectiveRendezvousArgs(CollectiveManager *man, InstanceView *v,
          const RegionUsage &use, const FieldMask &m, IndexSpaceNode *n,
          UniqueID id, size_t ctx, unsigned idx, 
          std::map<ApEvent,PhysicalTraceInfo*> &terms, RtEvent collect,
          RtUserEvent ap, const PhysicalTraceInfo *info, AddressSpaceID orig,
          AddressSpaceID src, bool sym, std::set<RtEvent> &applied_events)
      : LgTaskArgs<DeferCollectiveRendezvousArgs>(id), manager(man), view(v),
        usage(use), mask(new FieldMask(m)), expr(n), op_id(id),
        op_ctx_index(ctx), index(idx), 
        term_events(new std::map<ApEvent,PhysicalTraceInfo*>()),
        collect_event(collect), applied(ap), trace_info(info), origin(orig),
        source(src), symbolic(sym)
    //--------------------------------------------------------------------------
    {
      term_events->swap(terms);
      WrapperReferenceMutator mutator(applied_events);
      expr->add_base_expression_reference(META_TASK_REF, &mutator);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_defer_rendezvous(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferCollectiveRendezvousArgs *dargs =
        (const DeferCollectiveRendezvousArgs*)args;
      dargs->manager->finalize_collective_user(dargs->view, dargs->usage,
          *(dargs->mask), dargs->expr, dargs->op_id, dargs->op_ctx_index,
          dargs->index, *(dargs->term_events), dargs->collect_event,
          dargs->applied, *(dargs->trace_info), dargs->origin, dargs->source,
          dargs->symbolic);
      // Clean up our allocated data
      if (dargs->trace_info->recording)
      {
        for (std::map<ApEvent,PhysicalTraceInfo*>::const_iterator it =
             dargs->term_events->begin(); it != dargs->term_events->end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(it->second != NULL);
#endif
          delete it->second;
        }
        // dargs->trace_info is one of the trace infos in term_events
      }
      else
        delete dargs->trace_info;
      delete dargs->term_events;
      delete dargs->mask;
      if (dargs->expr->remove_base_expression_reference(META_TASK_REF))
        delete dargs->expr;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::create_collective_manager(
          Runtime *runtime, DistributedID did, AddressSpaceID owner_space, 
          IndexSpaceNode *point_space, size_t points,
          CollectiveMapping *mapping, size_t inst_footprint, 
          IndexSpaceExpression *inst_domain, const void *piece_list,
          size_t piece_list_size, FieldSpaceNode *space_node, 
          RegionTreeID tree_id,LayoutConstraints *constraints,
          ApBarrier use_barrier, ReductionOpID redop)
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
                                            owner_space, point_space, points,
                                            mapping, inst_domain, piece_list, 
                                            piece_list_size, space_node,tree_id,
                                            layout, redop, false/*reg now*/, 
                                            inst_footprint, use_barrier, 
                                            external_instance); 
      else
        man = new CollectiveManager(runtime->forest, did, owner_space, 
                                  point_space, points, mapping, inst_domain, 
                                  piece_list, piece_list_size, space_node, 
                                  tree_id, layout, redop, false/*reg now*/,
                                  inst_footprint,use_barrier,external_instance);
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
        case COLLECTIVE_ACTIVATE_MESSAGE:
          {
            RtUserEvent to_trigger;
            derez.deserialize(to_trigger);
            LocalReferenceMutator mutator;
            manager->activate_collective(&mutator);
            Runtime::trigger_event(to_trigger, mutator.get_done_event()); 
            break;
          }
        case COLLECTIVE_DEACTIVATE_MESSAGE:
          {
            RtUserEvent to_trigger;
            derez.deserialize(to_trigger);
            LocalReferenceMutator mutator;
            manager->deactivate_collective(&mutator);
            Runtime::trigger_event(to_trigger, mutator.get_done_event());
            break;
          }
        case COLLECTIVE_VALIDATE_MESSAGE:
          {
            RtUserEvent to_trigger;
            derez.deserialize(to_trigger);
            LocalReferenceMutator mutator;
            manager->validate_collective(&mutator);
            Runtime::trigger_event(to_trigger, mutator.get_done_event());
            break;
          }
        case COLLECTIVE_INVALIDATE_MESSAGE:
          {
            RtUserEvent to_trigger;
            derez.deserialize(to_trigger);
            LocalReferenceMutator mutator;
            manager->invalidate_collective(&mutator);
            Runtime::trigger_event(to_trigger, mutator.get_done_event());
            break;
          }
        case COLLECTIVE_PERFORM_DELETE_MESSAGE:
          {
            RtEvent deferred_event;
            derez.deserialize(deferred_event);
            bool left;
            derez.deserialize(left);
            manager->perform_delete(deferred_event, left);
            break;
          }
        case COLLECTIVE_FORCE_DELETE_MESSAGE:
          {
            bool left;
            derez.deserialize(left);
            manager->force_delete(left);
            break;
          }
        case COLLECTIVE_FINALIZE_MESSAGE:
          {
            if (manager->finalize_message() &&
                manager->remove_nested_resource_ref(did))
              delete manager;
            break;
          }
        case COLLECTIVE_REMOTE_INSTANCE_REQUEST:
          {
            size_t num_points;
            derez.deserialize(num_points);
            std::set<DomainPoint> points;
            for (unsigned idx = 0; idx < num_points; idx++)
            {
              DomainPoint point;
              derez.deserialize(point);
              points.insert(point);
            }
            AddressSpaceID origin;
            derez.deserialize(origin);
            RtUserEvent to_trigger;
            derez.deserialize(to_trigger);
            manager->find_or_forward_physical_instance(origin, points,
                                                       to_trigger);
            break;
          }
        case COLLECTIVE_REMOTE_INSTANCE_RESPONSE:
          {
            size_t num_points;
            derez.deserialize(num_points);
            std::map<DomainPoint,std::pair<PhysicalInstance,unsigned> > insts;
            for (unsigned idx = 0; idx < num_points; idx++)
            {
              DomainPoint point;
              derez.deserialize(point);
              std::pair<PhysicalInstance,unsigned> &inst = insts[point];
              derez.deserialize(inst.first);
              derez.deserialize(inst.second);
            }
            RtUserEvent to_trigger;
            derez.deserialize(to_trigger);
            manager->record_remote_physical_instances(insts);
            Runtime::trigger_event(to_trigger);
            break;
          }
        case COLLECTIVE_RENDEZVOUS_REQUEST:
          {
            manager->process_rendezvous_request(derez);
            break;
          }
        case COLLECTIVE_RENDEZVOUS_RESPONSE:
          {
            manager->process_rendezvous_response(derez);
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
    ApEvent VirtualManager::get_use_event(ApEvent user) const
    //--------------------------------------------------------------------------
    {
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent VirtualManager::get_instance_ready_event(void) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return RtEvent::NO_RT_EVENT;
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

    /////////////////////////////////////////////////////////////
    // Pending Collective Instance 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PendingCollectiveManager::PendingCollectiveManager(DistributedID id,
                                    size_t total, IndexSpace points,
                                    ApBarrier ready, CollectiveMapping *mapping)
      : did(id), total_points(total), point_space(points), ready_barrier(ready),
        collective_mapping(mapping)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(did > 0);
      assert(collective_mapping != NULL);
#endif
      collective_mapping->add_reference();
    }

    //--------------------------------------------------------------------------
    PendingCollectiveManager::PendingCollectiveManager(
                                            const PendingCollectiveManager &rhs)
      : did(rhs.did), total_points(rhs.total_points),
        point_space(rhs.point_space), ready_barrier(rhs.ready_barrier),
        collective_mapping(rhs.collective_mapping)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PendingCollectiveManager::~PendingCollectiveManager(void)
    //--------------------------------------------------------------------------
    {
      if (collective_mapping->remove_reference())
        delete collective_mapping;
    }

    //--------------------------------------------------------------------------
    PendingCollectiveManager& PendingCollectiveManager::operator=(
                                            const PendingCollectiveManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PendingCollectiveManager::pack(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(did);
      rez.serialize(total_points);
      rez.serialize(point_space);
      rez.serialize(ready_barrier);
      collective_mapping->pack(rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ PendingCollectiveManager* PendingCollectiveManager::unpack(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      if (did == 0)
        return NULL;
      size_t total_points;
      derez.deserialize(total_points);
      IndexSpace point_space;
      derez.deserialize(point_space);
      ApBarrier ready_barrier;
      derez.deserialize(ready_barrier);
      size_t total_spaces;
      derez.deserialize(total_spaces);
      CollectiveMapping *mapping = new CollectiveMapping(derez, total_spaces);
      return new PendingCollectiveManager(did, total_points, point_space,
                                          ready_barrier, mapping);
    }

    /////////////////////////////////////////////////////////////
    // Instance Builder
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceBuilder::InstanceBuilder(const std::vector<LogicalRegion> &regs,
                      IndexSpaceExpression *expr, FieldSpaceNode *node, 
                      RegionTreeID tid, const LayoutConstraintSet &cons, 
                      Runtime *rt, MemoryManager *memory, UniqueID cid,
                      const void *pl, size_t pl_size, bool shadow)
      : regions(regs), constraints(cons), runtime(rt), memory_manager(memory),
        creator_id(cid), instance(PhysicalInstance::NO_INST), 
        field_space_node(node), instance_domain(expr), tree_id(tid), 
        redop_id(0), reduction_op(NULL), realm_layout(NULL), piece_list(NULL),
        piece_list_size(0), shadow_instance(shadow), valid(true)
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
        RegionTreeForest *forest, PendingCollectiveManager *pending_collective,
        const DomainPoint *collective_point, LayoutConstraintKind *unsat_kind,
        unsigned *unsat_index, size_t *footprint)
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
        // If we didn't make a collective instance we still need to arrive
        // on its ready barrier so the barrier completes
        if (pending_collective != NULL)
          Runtime::phase_barrier_arrive(pending_collective->ready_barrier, 1);
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
        {
          // If we didn't make a collective instance we still need to arrive
          // on its ready barrier so the barrier completes
          if (pending_collective != NULL)
            Runtime::phase_barrier_arrive(pending_collective->ready_barrier, 1);
          return NULL;
        }
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
      {
        runtime->profiler->add_inst_request(requests, creator_id);
        ready = ApEvent(PhysicalInstance::create_instance(instance,
                  memory_manager->memory, inst_layout, requests));
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
                  memory_manager->memory, inst_layout, requests));
      // Wait for the profiling response
      if (!profiling_ready.has_triggered())
        profiling_ready.wait();
#else
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
          // If we didn't make a collective instance we still need to arrive
          // on its ready barrier so the barrier completes
          if (pending_collective != NULL)
            Runtime::phase_barrier_arrive(pending_collective->ready_barrier, 1);
          return NULL;
        }
      }
      Realm::ExternalMemoryResource resource(base_ptr,
          inst_layout->bytes_used, false/*read only*/);
      ready = ApEvent(PhysicalInstance::create_external_instance(instance,
                memory_manager->memory, inst_layout, resource, requests));
#endif
      // If we couldn't make it then we are done
      if (!instance.exists())
      {
#ifndef LEGION_MALLOC_INSTANCES
        if (runtime->profiler != NULL)
          runtime->profiler->handle_failed_instance_allocation();
#endif
        if (unsat_kind != NULL)
          *unsat_kind = LEGION_MEMORY_CONSTRAINT;
        if (unsat_index != NULL)
          *unsat_index = 0;
        // If we didn't make a collective instance we still need to arrive
        // on its ready barrier so the barrier completes
        if (pending_collective != NULL)
          Runtime::phase_barrier_arrive(pending_collective->ready_barrier, 1);
        return NULL;
      }
#ifdef LEGION_DEBUG
      assert(!constraints.pointer_constraint.is_valid);
#endif
      // If we successfully made the instance then Realm 
      // took over ownership of the layout
      PhysicalManager *result = NULL;
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
      const AddressSpaceID local_space = forest->runtime->address_space;
      if (pending_collective != NULL)
      {
        // Creating a collective manager
#ifdef DEBUG_LEGION
        assert(collective_point != NULL);
#endif
        // See if we're the first point on this node to try to make it
        RtEvent manager_ready;
        // Preallocate a buffer to use assuming we get to use it
        // If we don't use it then we'll easily free it
        void *buffer = 
          legion_alloc_aligned<CollectiveManager,false/*bytes*/>(1/*count*/);
        DistributedCollectable *collectable = NULL;
        CollectiveManager *manager = NULL;
        if (forest->runtime->find_or_create_distributed_collectable(
              pending_collective->did, collectable, manager_ready, buffer))
        {
          const AddressSpaceID owner_space =
              forest->runtime->determine_owner(pending_collective->did);
          IndexSpaceNode *point_space = 
            pending_collective->point_space.exists() ?
              forest->get_node(pending_collective->point_space) : NULL;
          // First point so create it
          manager = new(buffer) CollectiveManager(forest,
              pending_collective->did, owner_space, point_space,
              pending_collective->total_points,
              pending_collective->collective_mapping,
              instance_domain, piece_list, piece_list_size,
              field_space_node, tree_id, layout, redop_id,
              true/*register now*/, instance_footprint,
              pending_collective->ready_barrier, false/*external*/);
#ifdef DEBUG_LEGION
          assert(manager == collectable);
#endif
          // Then register it for anyone coming later
          forest->runtime->register_distributed_collectable(
              pending_collective->did, manager);
        }
        else
        {
          // Free the buffer since we didn't use it
          legion_free(CollectiveManager::alloc_type, buffer, 
                      sizeof(CollectiveManager));
          // Not the first point, so wait for it and then can safely cast
          if (manager_ready.exists() && !manager_ready.has_triggered())
            manager_ready.wait();
#ifdef DEBUG_LEGION
          manager = dynamic_cast<CollectiveManager*>(collectable);
          assert(manager != NULL);
#else
          manager = static_cast<CollectiveManager*>(collectable);
#endif
        }
        manager->record_point_instance(*collective_point, instance);
        // Signal that the point instance has been updated and that
        // the manager is ready once it is triggered
        Runtime::phase_barrier_arrive(pending_collective->ready_barrier,
            1/*count*/, ready);
        result = manager;
      }
      else
      {
        // Creating an individual manager
#ifdef DEBUG_LEGION
        assert(collective_point == NULL);
#endif
        // For Legion Spy we need a unique ready event if it doesn't already
        // exist so we can uniquely identify the instance
        if (!ready.exists() && runtime->legion_spy_enabled)
        {
          ApUserEvent rename_ready = Runtime::create_ap_user_event(NULL);
          Runtime::trigger_event(NULL, rename_ready);
          ready = rename_ready;
        }
        DistributedID did = forest->runtime->get_available_distributed_id();
        // Figure out what kind of instance we just made
        switch (constraints.specialized_constraint.get_kind())
        {
          case LEGION_NO_SPECIALIZE:
          case LEGION_AFFINE_SPECIALIZE:
          case LEGION_COMPACT_SPECIALIZE:
            {
#ifdef DEBUG_LEGION
              assert(!shadow_instance);
#endif
              // Now we can make the manager
              result = new IndividualManager(forest, did, local_space,
                                             memory_manager,
                                             instance, instance_domain, 
                                             piece_list, piece_list_size,
                                             field_space_node, tree_id,
                                             layout, 0/*redop id*/,
                                             true/*register now*/, 
                                             instance_footprint, ready,
                                       PhysicalManager::INTERNAL_INSTANCE_KIND); 
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
                                        PhysicalManager::INTERNAL_INSTANCE_KIND,
                                             reduction_op, shadow_instance);
              break;
            }
          default:
            assert(false); // illegal specialized case
        }
      }
      // manager takes ownership of the piece list
      piece_list = NULL;
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
#ifdef LEGION_MALLOC_INSTANCES
      memory_manager->record_legion_instance(instance, base_ptr); 
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

