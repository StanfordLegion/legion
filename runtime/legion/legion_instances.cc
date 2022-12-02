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
    void LayoutDescription::log_instance_layout(LgEvent inst_event) const
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
      }
    } 

    //--------------------------------------------------------------------------
    void LayoutDescription::compute_copy_offsets(
                                   const std::vector<FieldID> &copy_fields, 
                                   const PhysicalInstance instance,
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
      : radix(r)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < mapping.size(); idx++)
        unique_sorted_spaces.add(mapping[idx]);
      total_spaces = unique_sorted_spaces.size();
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
    CollectiveMapping::CollectiveMapping(const CollectiveMapping &rhs)
      : Collectable(), unique_sorted_spaces(rhs.unique_sorted_spaces),
        total_spaces(rhs.total_spaces), radix(rhs.radix)
    //--------------------------------------------------------------------------
    {
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
      const int result = unique_sorted_spaces.get_index(index);
#ifdef DEBUG_LEGION
      assert(result >= 0);
#endif
      return result;
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
          const int child = unique_sorted_spaces.get_index(index);
#ifdef DEBUG_LEGION
          assert(child >= 0);
#endif
          children.push_back(child);
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
                                     DistributedID did, LayoutDescription *desc,
                                     FieldSpaceNode *node, 
                                     IndexSpaceExpression *domain,
                                     RegionTreeID tid, bool register_now,
                                     CollectiveMapping *mapping)
      : DistributedCollectable(ctx->runtime, did, register_now, mapping),
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
    bool InstanceManager::entails(LayoutConstraints *constraints,
                               const LayoutConstraint **failed_constraint) const
    //--------------------------------------------------------------------------
    {
      const PointerConstraint &pointer = constraints->pointer_constraint;
      if (pointer.is_valid)
      {
        PointerConstraint pointer_constraint = get_pointer_constraint();
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
                               const LayoutConstraint **failed_constraint) const
    //--------------------------------------------------------------------------
    {
      const PointerConstraint &pointer = constraints.pointer_constraint;
      if (pointer.is_valid)
      {
        PointerConstraint pointer_constraint = get_pointer_constraint();
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
                             const LayoutConstraint **conflict_constraint) const
    //--------------------------------------------------------------------------
    {
      const PointerConstraint &pointer = constraints->pointer_constraint;
      if (pointer.is_valid)
      {
        PointerConstraint pointer_constraint = get_pointer_constraint();
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
                             const LayoutConstraint **conflict_constraint) const
    //--------------------------------------------------------------------------
    {
      const PointerConstraint &pointer = constraints.pointer_constraint;
      if (pointer.is_valid)
      {
        PointerConstraint pointer_constraint = get_pointer_constraint();
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
    PhysicalManager::PhysicalManager(RegionTreeForest *ctx, DistributedID did,
                        MemoryManager *memory, PhysicalInstance inst, 
                        IndexSpaceExpression *instance_domain,
                        const void *pl, size_t pl_size,
                        FieldSpaceNode *node, RegionTreeID tree_id,
                        LayoutDescription *layout, ReductionOpID redop_id,
                        bool register_now, size_t footprint,
                        ApEvent u_event, InstanceKind k,
                        const ReductionOp *op /*= NULL*/,
                        CollectiveMapping *mapping /*=NULL*/,
                        ApEvent p_event /*= ApEvent::NO_AP_EVENT*/)
      : InstanceManager(ctx, encode_instance_did(did, 
          (k == EXTERNAL_ATTACHED_INSTANCE_KIND), (redop_id > 0)), layout, node,
          // If we're on the owner node we need to produce the expression
          // that actually describes this points in this space
          // On remote nodes we'll already have it from the owner
          (ctx->runtime->determine_owner(did) == ctx->runtime->address_space) &&
            (k != UNBOUND_INSTANCE_KIND) ?
            instance_domain->create_layout_expression(pl, pl_size) : 
            instance_domain, tree_id, register_now, mapping), 
        memory_manager(memory), unique_event(u_event), 
        instance_footprint(footprint), reduction_op((redop_id == 0) ? NULL : 
            ctx->runtime->get_reduction(redop_id)), redop(redop_id),
        piece_list(pl), piece_list_size(pl_size), instance(inst),
        use_event(Runtime::create_ap_user_event(NULL)),
        instance_ready((k == UNBOUND_INSTANCE_KIND) ? 
            Runtime::create_rt_user_event() : RtUserEvent::NO_RT_USER_EVENT),
        kind(k), external_pointer(-1UL), producer_event(p_event),
        gc_state(COLLECTABLE_GC_STATE), pending_changes(0),
        failed_collection_count(0), min_gc_priority(0), added_gc_events(0),
        valid_references(0), sent_valid_references(0),
        received_valid_references(0)
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

      if (!is_owner() && !is_external_instance())
        memory_manager->register_remote_instance(this);
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
    PhysicalManager::~PhysicalManager(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(subscribers.empty());
      assert(valid_references == 0);
#endif
      // Remote references removed by DistributedCollectable destructor
      if (!is_owner() && !is_external_instance())
        memory_manager->unregister_remote_instance(this);
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        PhysicalManager::get_accessor(void) const
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
        PhysicalManager::get_field_accessor(FieldID fid) const
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
    ApEvent PhysicalManager::get_use_event(ApEvent user) const
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
    PointerConstraint PhysicalManager::get_pointer_constraint(void) const
    //--------------------------------------------------------------------------
    {
      if (use_event.exists() && !use_event.has_triggered_faultignorant())
        use_event.wait_faultignorant();
      void *inst_ptr = instance.pointer_untyped(0/*offset*/, 0/*elem size*/);
      return PointerConstraint(memory_manager->memory, uintptr_t(inst_ptr));
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::log_instance_creation(UniqueID creator_id,
                Processor proc, const std::vector<LogicalRegion> &regions) const
    //--------------------------------------------------------------------------
    {
      const LgEvent inst_event = get_unique_event();
      const LayoutConstraints *constraints = layout->constraints;
      LegionSpy::log_physical_instance_creator(inst_event, creator_id, proc.id);
      for (unsigned idx = 0; idx < regions.size(); idx++)
        LegionSpy::log_physical_instance_creation_region(inst_event, 
                                                         regions[idx]);
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
    void PhysicalManager::compute_copy_offsets(const FieldMask &copy_mask,
                                           std::vector<CopySrcDstField> &fields)
    //--------------------------------------------------------------------------
    {
      // Make sure the instance is ready before we compute the offsets
      if (instance_ready.exists() && !instance_ready.has_triggered())
        instance_ready.wait();
#ifdef DEBUG_LEGION
      assert(layout != NULL);
      assert(instance.exists());
#endif
      // Pass in our physical instance so the layout knows how to specialize
      layout->compute_copy_offsets(copy_mask, instance, fields);
    }

    //--------------------------------------------------------------------------
    IndividualView* PhysicalManager::construct_top_view(
                                           AddressSpaceID logical_owner,
                                           DistributedID view_did,
                                           InnerContext *own_ctx,
                                           CollectiveMapping *mapping)
    //--------------------------------------------------------------------------
    {
      if (redop > 0)
      {
        if (mapping != NULL)
        {
          // Handle the case where we already requested this view on this
          // node from an unrelated meta-task execution
          void *location = runtime->find_or_create_pending_collectable_location(
              view_did, sizeof(ReductionView));
          return new (location) ReductionView(runtime, view_did,
              logical_owner, this, true/*register now*/, mapping);
        }
        else
          return new ReductionView(runtime, view_did,
              logical_owner, this, true/*register now*/, mapping);
      }
      else
      {
        if (mapping != NULL)
        {
          // Handle the case where we already requested this view on this
          // node from an unrelated meta-task execution
          void *location = runtime->find_or_create_pending_collectable_location(
              view_did, sizeof(MaterializedView));
          return new (location) MaterializedView(runtime, view_did,
                logical_owner, this, true/*register now*/, mapping);
        }
        else
          return new MaterializedView(runtime, view_did,
                logical_owner, this, true/*register now*/, mapping);
      }
    }

    //--------------------------------------------------------------------------
    IndividualView* PhysicalManager::find_or_create_instance_top_view(
                                                   InnerContext *own_ctx,
                                                   AddressSpaceID logical_owner,
                                                   CollectiveMapping *mapping)
    //--------------------------------------------------------------------------
    {
      // If we're a replicate context then we want to ignore the specific
      // context DID since there might be several shards on this node
      bool replicated = false;
      DistributedID key = own_ctx->get_replication_id();
      if (key == 0)
        key = own_ctx->did;
      else
        replicated = true;
      RtEvent wait_for;
      {
        AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
        // All contexts should always be new since they should be deduplicating
        // on their side before calling this method
        assert(subscribers.find(own_ctx) == subscribers.end());
#endif
        std::map<DistributedID,ViewEntry>::iterator finder =
          context_views.find(key);
        if (finder != context_views.end())
        {
#ifdef DEBUG_LEGION
          // This should only happen with control replication because normal
          // contexts should be deduplicating on their side
          assert(replicated);
#endif
          // This better be a new context so bump the reference count
          if (subscribers.insert(own_ctx).second)
            own_ctx->add_subscriber_reference(this);
          finder->second.second++;
          return finder->second.first;
        }
        // Check to see if someone else from this context is making the view 
        if (replicated)
        {
          // Only need to do this for control replication, otherwise the
          // context will have deduplicated for us
          std::map<DistributedID,RtUserEvent>::iterator pending_finder =
            pending_views.find(key);
          if (pending_finder != pending_views.end())
          {
            if (!pending_finder->second.exists())
              pending_finder->second = Runtime::create_rt_user_event();
            wait_for = pending_finder->second;
          }
          else
            pending_views[key] = RtUserEvent::NO_RT_USER_EVENT;
        }
      }
      if (wait_for.exists())
      {
        if (!wait_for.has_triggered())
          wait_for.wait();
        AutoLock i_lock(inst_lock);
        std::map<DistributedID,ViewEntry>::iterator finder =
          context_views.find(key);
#ifdef DEBUG_LEGION
        assert(replicated);
        assert(finder != context_views.end());
#endif
        // This better be a new context so bump the reference count
        if (subscribers.insert(own_ctx).second)
          own_ctx->add_subscriber_reference(this);
        finder->second.second++;
        return finder->second.first;
      }
      // At this point we're repsonsibile for doing the work to make the view 
      IndividualView *result = NULL;
      // Check to see if we're the owner
      if (is_owner())
      {
        // We're going to construct the view no matter what, see which 
        // node is going to be the logical owner
        DistributedID view_did = runtime->get_available_distributed_id(); 
        result = construct_top_view((mapping == NULL) ? logical_owner :
            owner_space, view_did, own_ctx, mapping);
      }
      else if ((mapping != NULL) && mapping->contains(local_space))
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
          rez.serialize(key);
          rez.serialize(own_ctx->did);
          rez.serialize(owner_space);
          mapping->pack(rez);
          rez.serialize(&view_did);
          rez.serialize(ready);
        }
        AddressSpaceID target = mapping->get_parent(owner_space, local_space);
        runtime->send_create_top_view_request(target, rez); 
        ready.wait();
        // For collective instances each node of the instance serves as its
        // own logical owner view
        result = construct_top_view(runtime->address_space, view_did.load(),
                                    own_ctx, mapping);
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
          rez.serialize(key);
          rez.serialize(own_ctx->did);
          rez.serialize(logical_owner);
          rez.serialize<size_t>(0); // no mapping
          rez.serialize(&view_did);
          rez.serialize(ready);
        }
        runtime->send_create_top_view_request(owner_space, rez); 
        ready.wait();
        RtEvent view_ready;
        result = static_cast<IndividualView*>(
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
      if (subscribers.insert(own_ctx).second)
        own_ctx->add_subscriber_reference(this);
      if (replicated)
      {
        std::map<DistributedID,RtUserEvent>::iterator finder =
          pending_views.find(key);
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
    void PhysicalManager::register_deletion_subscriber(
                                         InstanceDeletionSubscriber *subscriber)
    //--------------------------------------------------------------------------
    {
      subscriber->add_subscriber_reference(this);
      AutoLock inst(inst_lock);
#ifdef DEBUG_LEGION
      assert(subscribers.find(subscriber) == subscribers.end());
#endif
      subscribers.insert(subscriber);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::unregister_deletion_subscriber(
                                         InstanceDeletionSubscriber *subscriber)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock inst(inst_lock);
        std::set<InstanceDeletionSubscriber*>::iterator finder =
          subscribers.find(subscriber);
        if (finder == subscribers.end())
          return;
        subscribers.erase(finder);
      }
      if (subscriber->remove_subscriber_reference(this))
        delete subscriber;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::unregister_active_context(InnerContext *own_ctx)
    //--------------------------------------------------------------------------
    {
      // If we're a replicate context then we want to ignore the specific
      // context UID since there might be several shards on this node
      DistributedID key = own_ctx->get_replication_id();
      if (key == 0)
        key = own_ctx->did;
      {
        AutoLock inst(inst_lock);
        std::set<InstanceDeletionSubscriber*>::iterator finder = 
          subscribers.find(own_ctx);
        // We could already have removed this context if this
        // physical instance was deleted
        if (finder == subscribers.end())
          return;
        subscribers.erase(finder);
        // Remove the reference on the view entry and remove it from our
        // manager if it no longer has anymore active contexts
        std::map<DistributedID,ViewEntry>::iterator view_finder =
          context_views.find(key);
#ifdef DEBUG_LEGION
        assert(view_finder != context_views.end());
        assert(view_finder->second.second > 0);
#endif
        if (--view_finder->second.second == 0)
          context_views.erase(view_finder);
      }
      if (own_ctx->remove_subscriber_reference(this))
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
    void PhysicalManager::record_instance_user(ApEvent user_event,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      AutoLock inst(inst_lock);
#ifdef DEBUG_LEGION
      assert(gc_state != COLLECTED_GC_STATE);
      assert(added_gc_events < runtime->gc_epoch_size);
#endif
      if (is_owner() || (gc_state != PENDING_COLLECTED_GC_STATE))
      {
        if (gc_events.insert(user_event).second && 
            (++added_gc_events == runtime->gc_epoch_size))
        {
          // Go through and prune out any events that have triggered
          for (std::set<ApEvent>::iterator it = gc_events.begin();
                it != gc_events.end(); /*nothing*/)
          {
            if (it->has_triggered_faultignorant())
            {
              std::set<ApEvent>::iterator to_delete = it++;
              gc_events.erase(to_delete);
            }
            else
              it++;
          }
          added_gc_events = 0;
        }
      }
      else
      {
        const RtEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(user_event);
          rez.serialize(applied);
        }
        pack_global_ref();
        runtime->send_gc_record_event(owner_space, rez);
        applied_events.insert(applied);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_record_event(Runtime *runtime,
                                                         Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      ApEvent user_event;
      derez.deserialize(user_event);
      RtUserEvent done;
      derez.deserialize(done);

      PhysicalManager *manager = static_cast<PhysicalManager*>(
          runtime->find_distributed_collectable(did));
      std::set<RtEvent> applied;
      manager->record_instance_user(user_event, applied);
      manager->unpack_global_ref();
      if (!applied.empty())
        Runtime::trigger_event(done, Runtime::merge_events(applied));
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::find_shutdown_preconditions(
                                               std::set<ApEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      AutoLock inst(inst_lock,1,false/*exclusive*/);
      for (std::set<ApEvent>::const_iterator it =
            gc_events.begin(); it != gc_events.end(); it++)
        if (!it->has_triggered_faultignorant())
            preconditions.insert(*it);
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
    void PhysicalManager::notify_local(void)
    //--------------------------------------------------------------------------
    {
      // Nothing to do here 
    } 

    //--------------------------------------------------------------------------
    void PhysicalManager::pack_valid_ref(void)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
      // We should always be holding a valid reference when we
      // pack a valid reference so the state should always be valid
      assert(gc_state == VALID_GC_STATE);
#endif
      sent_valid_references++;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::unpack_valid_ref(void)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
      received_valid_references++;
    }

#ifdef DEBUG_LEGION_GC
    //--------------------------------------------------------------------------
    void PhysicalManager::add_base_valid_ref_internal(
                                                ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
      valid_references += cnt;
      std::map<ReferenceSource,int>::iterator finder = 
        detailed_base_valid_references.find(source);
      if (finder == detailed_base_valid_references.end())
        detailed_base_valid_references[source] = cnt;
      else
        finder->second += cnt;
      if (valid_references == cnt)
        notify_valid(true/*need check*/);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::add_nested_valid_ref_internal(
                                                  DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
      valid_references += cnt;
      std::map<DistributedID,int>::iterator finder = 
        detailed_nested_valid_references.find(source);
      if (finder == detailed_nested_valid_references.end())
        detailed_nested_valid_references[source] = cnt;
      else
        finder->second += cnt;
      if (valid_references == cnt)
        notify_valid(true/*need check*/);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::remove_base_valid_ref_internal(
                                                ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
      assert(valid_references >= cnt);
#endif
      valid_references -= cnt;
      std::map<ReferenceSource,int>::iterator finder = 
        detailed_base_valid_references.find(source);
      assert(finder != detailed_base_valid_references.end());
      assert(finder->second >= cnt);
      finder->second -= cnt;
      if (finder->second == 0)
        detailed_base_valid_references.erase(finder);
      if (valid_references == 0)
        return notify_invalid();
      else
        return false;
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::remove_nested_valid_ref_internal(
                                                  DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
      assert(valid_references >= cnt);
#endif
      valid_references -= cnt;
      std::map<DistributedID,int>::iterator finder = 
        detailed_nested_valid_references.find(source);
      assert(finder != detailed_nested_valid_references.end());
      assert(finder->second >= cnt);
      finder->second -= cnt;
      if (finder->second == 0)
        detailed_nested_valid_references.erase(finder);
      if (valid_references == 0)
        return notify_invalid();
      else
        return false;
    } 

    //--------------------------------------------------------------------------
    void PhysicalManager::add_valid_reference(int cnt, bool need_check)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
      if (valid_references == 0)
        notify_valid(need_check);
      valid_references += cnt;
    }
#else // DEBUG_LEGION_GC 
    //--------------------------------------------------------------------------
    void PhysicalManager::add_valid_reference(int cnt, bool need_check)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
      if (valid_references.fetch_add(cnt) == 0)
        notify_valid(need_check);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::remove_valid_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
      assert(valid_references.load() >= cnt);
#endif
      if (valid_references.fetch_sub(cnt) == cnt)
        return notify_invalid();
      else
        return false;
    }
#endif // !defined DEBUG_LEGION_GC

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_valid(bool need_check)
    //--------------------------------------------------------------------------
    {
      // No need for the lock, it is held by the caller
#ifdef DEBUG_LEGION
      assert(gc_state != VALID_GC_STATE);
      assert(gc_state != COLLECTED_GC_STATE);
      // In debug mode we eagerly add valid references such that the owner
      // is valid as long as a copy of the manager on one node is valid
      // This way we can easily check that acquires are being done safely
      // if instance isn't already valid somewhere
      if (need_check && (!is_external_instance() || !is_owner()))
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
        }
        pack_global_ref();
        runtime->send_gc_debug_request(owner_space, rez);
        if (!done.has_triggered())
          done.wait();
        if (!result.load())
          REPORT_LEGION_FATAL(LEGION_FATAL_GARBAGE_COLLECTION_RACE,
                "Found an internal garbage collection race. Please "
                "run with -lg:safe_mapper and see if it reports any "
                "errors. If not, then please report this as a bug.")
      }
#else
      if (gc_state == COLLECTED_GC_STATE)
        REPORT_LEGION_FATAL(LEGION_FATAL_GARBAGE_COLLECTION_RACE,
                "Found an internal garbage collection race. Please "
                "run with -lg:safe_mapper and see if it reports any "
                "errors. If not, then please report this as a bug.")
#endif
      gc_state = VALID_GC_STATE;
      add_base_gc_ref(INTERNAL_VALID_REF);
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

      PhysicalManager *manager = static_cast<PhysicalManager*>(
          runtime->find_distributed_collectable(did));
      // Should be guaranteed to be able to acquire this
      if (manager->acquire_instance(REMOTE_DID_REF))
      {
        Runtime::trigger_event(done);
        // Remove the reference that we just got
        manager->remove_base_valid_ref(REMOTE_DID_REF);
      }
      else
      {
        // If we get here, we failed so send the response
        Serializer rez;
        {
          RezCheck z2(rez);
          rez.serialize(target);
          rez.serialize(done);
        }
        runtime->send_gc_debug_response(source, rez);
      }
      manager->unpack_global_ref();
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
    bool PhysicalManager::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      // No need for the lock it is held by the caller
#ifdef DEBUG_LEGION
      assert(gc_state == VALID_GC_STATE);
#endif
      gc_state = COLLECTABLE_GC_STATE;
      return remove_base_gc_ref(INTERNAL_VALID_REF);
    }

    //--------------------------------------------------------------------------
#ifdef DEBUG_LEGION_GC
    template<typename T>
    bool PhysicalManager::acquire_internal(T source, 
                                     std::map<T,int> &detailed_valid_references)
#else
    bool PhysicalManager::acquire_internal(void) 
#endif
    //--------------------------------------------------------------------------
    {
      {
        bool success = false;
        AutoLock i_lock(inst_lock);
        // Check our current state
        switch (gc_state)
        {
          case VALID_GC_STATE:
            {
#ifdef DEBUG_LEGION
              assert(valid_references > 0);
#endif
              success = true;
              break;
            }
          case COLLECTABLE_GC_STATE:
            {
              notify_valid(false/*need check*/);
              success = true;
              break;
            }
          case PENDING_COLLECTED_GC_STATE:
            {
              // Hurry the garbage collector is trying to eat it!
              if (is_owner())
              {
                // We're the owner so we can save this
                notify_valid(false/*need check*/);
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
        if (success)
        {
#ifdef DEBUG_LEGION_GC
          valid_references++;
          typename std::map<T,int>::iterator finder =
            detailed_valid_references.find(source);
          if (finder == detailed_valid_references.end())
            detailed_valid_references[source] = 1;
          else
            finder->second++;
#else
          valid_references.fetch_add(1);
#endif
          return true;
        }
      }
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
        rez.serialize(&result);
        rez.serialize(ready);
      }
      runtime->send_acquire_request(owner_space, rez);
      ready.wait();
      if (result.load())
      {
#ifdef DEBUG_LEGION_GC
        AutoLock i_lock(inst_lock);
        typename std::map<T,int>::iterator finder =
          detailed_valid_references.find(source);
        if (finder == detailed_valid_references.end())
          detailed_valid_references[source] = 1;
        else
          finder->second++;
#endif
        return true;
      }
      else
      {
        std::set<InstanceDeletionSubscriber*> to_notify;
        {
          AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
          assert((gc_state == PENDING_COLLECTED_GC_STATE) ||
                  (gc_state == COLLECTED_GC_STATE));
#endif
          gc_state = COLLECTED_GC_STATE;
          to_notify.swap(subscribers);
        }
        for (std::set<InstanceDeletionSubscriber*>::const_iterator it =
              to_notify.begin(); it != to_notify.end(); it++)
        {
          (*it)->notify_instance_deletion(this);
          if ((*it)->remove_subscriber_reference(this))
            delete (*it);
        }
        return false;
      }
    }

#ifdef DEBUG_LEGION_GC
    // Explicit template instantiations
    template bool PhysicalManager::acquire_internal<ReferenceSource>(
                            ReferenceSource, std::map<ReferenceSource,int>&);
    template bool PhysicalManager::acquire_internal<DistributedID>(
                            DistributedID, std::map<DistributedID,int>&);
#endif

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
      std::atomic<bool> *result;
      derez.deserialize(result);
      RtUserEvent ready;
      derez.deserialize(ready);

      PhysicalManager *manager = static_cast<PhysicalManager*>(
          runtime->find_distributed_collectable(did));
      if (manager->acquire_instance(REMOTE_DID_REF))
      {
        // We succeeded so send the response back with the reference
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote);
          rez.serialize(result);
          rez.serialize(ready);
        }
        runtime->send_acquire_response(source, rez);
        // Wait for the result to be applied and then remove
        // the reference that we acquired on this node
        ready.wait();
        manager->remove_base_valid_ref(REMOTE_DID_REF);
      }
      else
      {
        // We failed, so the flag is already set, just trigger the event
        Runtime::trigger_event(ready);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_acquire_response(
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      PhysicalManager *manager;
      derez.deserialize(manager);
      std::atomic<bool> *result;
      derez.deserialize(result);
      RtUserEvent ready;
      derez.deserialize(ready);

      // Just add the reference for now
      manager->add_valid_reference(1/*count*/, false/*need check*/);
      result->store(true);
      // Triggering the event removes the reference we added on the remote node
      Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::can_collect(bool &already_collected) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      // This is a lightweight test that shouldn't involve any communication
      // or commitment to performing a collection. It's just for finding
      // instances that we know are locally collectable
      already_collected = false;
      AutoLock i_lock(inst_lock,1,false/*exclusive*/);
      // Do a quick to check to see if we can do a collection on the local node
      if (gc_state == VALID_GC_STATE)
        return false;
      // If it's already collected then we're done
      if (gc_state == COLLECTED_GC_STATE)
      {
        already_collected = true;
        return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::acquire_collect(std::set<ApEvent> &remote_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner());
#endif
      AutoLock i_lock(inst_lock);
      // Do a quick to check to see if we can do a collection on the local node
      if (gc_state == VALID_GC_STATE)
        return false;
#ifdef DEBUG_LEGION
      assert(gc_state != COLLECTED_GC_STATE);
#endif
      gc_state = PENDING_COLLECTED_GC_STATE;
      remote_events.swap(gc_events);
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
          runtime->find_distributed_collectable(did));
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
      manager->unpack_global_ref();
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
                                          Runtime *runtime, Deserializer &derez)
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
      std::set<ApEvent> gc_events;
      const AddressSpaceID owner = manager->owner_space;
      if (!manager->acquire_collect(gc_events))
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(target);
          rez.serialize(done);
        }
        runtime->send_gc_failed(owner, rez);
      }
      else
      {
        std::set<RtEvent> ready_events;
        // Send the gc events back to the owner if we have any, merge
        // them all back together first so there is just one remote
        // event on this node
        if (!gc_events.empty())
        {
          const ApEvent remote = Runtime::merge_events(NULL, gc_events);
          if (remote.exists())
            manager->record_instance_user(remote, ready_events);
        }
        const AddressSpaceID local = manager->local_space;
        // Check to see if we need to broadcast this out to more places
        if ((manager->collective_mapping != NULL) &&
            manager->collective_mapping->contains(local))
        {
          // Broadcast this out to all our children
          std::vector<AddressSpaceID> children;
          manager->collective_mapping->get_children(owner, local, children);
          if (!children.empty())
          {
            for (std::vector<AddressSpaceID>::const_iterator it =
                  children.begin(); it != children.end(); it++)
            {
              const RtUserEvent child_done = Runtime::create_rt_user_event();
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
                rez.serialize(target);
                rez.serialize(child_done);
              }
              runtime->send_gc_acquire(*it, rez);
              ready_events.insert(child_done);
            }
          }
        }
        if (!ready_events.empty())
          Runtime::trigger_event(done, Runtime::merge_events(ready_events));
        else
          Runtime::trigger_event(done);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_garbage_collection_failed(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::atomic<unsigned> *target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);

      target->fetch_add(1);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_garbage_collection_notify(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);

      // Should still be able to find this manager here
      PhysicalManager *manager = static_cast<PhysicalManager*>(
          runtime->find_distributed_collectable(did));
      manager->notify_remote_deletion();
      manager->unpack_global_ref();
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_remote_deletion(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner());
#endif
      // Forward on the deletion notification to any children
      if ((collective_mapping != NULL) && 
          collective_mapping->contains(local_space))
      {
        std::vector<AddressSpaceID> children;
        collective_mapping->get_children(owner_space, local_space, children);
        if (!children.empty())
        {
<<<<<<< HEAD
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
          }
          for (std::vector<AddressSpaceID>::const_iterator it = 
                children.begin(); it != children.end(); it++)
          {
            pack_global_ref();
            runtime->send_gc_notify(*it, rez);
          }
=======
          rez.serialize<size_t>(gc_events.size());
          for (std::set<ApEvent>::const_iterator it =
                gc_events.begin(); it != gc_events.end(); it++)
            rez.serialize(*it);
>>>>>>> control_replication
        }
      }
      std::set<InstanceDeletionSubscriber*> to_notify;
      {
        AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
        assert((gc_state == COLLECTED_GC_STATE) ||
                (gc_state == PENDING_COLLECTED_GC_STATE));
#endif
        gc_state = COLLECTED_GC_STATE;
        to_notify.swap(subscribers);
      }
      if (!to_notify.empty())
      {
        for (std::set<InstanceDeletionSubscriber*>::const_iterator it =
              to_notify.begin(); it != to_notify.end(); it++)
        {
          (*it)->notify_instance_deletion(this);
          if ((*it)->remove_subscriber_reference(this))
            delete (*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::pack_garbage_collection_state(Serializer &rez,
                                          AddressSpaceID target, bool need_lock)
    //--------------------------------------------------------------------------
    {
      // We have to atomically get the current collection state and 
      // update the set of remote instances, note that it can be read-only
      // since we're just reading the state and the `update-remote_instaces'
      // call will take its own exclusive lock
      if (need_lock)
      {
        AutoLock i_lock(inst_lock,1,false/*exclusive*/);
        pack_garbage_collection_state(rez, target, false/*need lock*/);
      }
      else
      {
        switch (gc_state)
        {
          case VALID_GC_STATE:
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
      if (gc_state == VALID_GC_STATE)
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
          gc_state = PENDING_COLLECTED_GC_STATE;
          failed_collection_count.store(0);
          std::vector<RtEvent> ready_events;
          if (collective_mapping != NULL)
          {
#ifdef DEBUG_LEGION
            // We're the owner so it should contain ourselves
            assert(collective_mapping->contains(local_space));
#endif
            std::vector<AddressSpaceID> children;
            collective_mapping->get_children(owner_space, local_space,children);
            for (std::vector<AddressSpaceID>::const_iterator it =
                  children.begin(); it != children.end(); it++)
            {
              const RtUserEvent ready_event = Runtime::create_rt_user_event();
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
                rez.serialize(&failed_collection_count);
                rez.serialize(ready_event);
              }
              runtime->send_gc_acquire(*it, rez);
              ready_events.push_back(ready_event);
            }
          }
          const size_t needed_guards = count_remote_instances();
          if (needed_guards > 0)
          {
            struct AcquireFunctor {
              AcquireFunctor(DistributedID d, Runtime *rt, 
                             std::vector<RtEvent> &r,
                             std::atomic<unsigned> *c)
                : did(d), runtime(rt), ready_events(r), count(c) { }
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
              std::vector<RtEvent> &ready_events;
              std::atomic<unsigned> *const count;
            };
            AcquireFunctor functor(did, runtime, ready_events,
                                   &failed_collection_count);
            map_over_remote_instances(functor);
          }
          if (!ready_events.empty())
            collection_ready = Runtime::merge_events(ready_events);
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
          // because something else acquired a valid reference while
          // the collection was in progress
          case VALID_GC_STATE:
          case COLLECTABLE_GC_STATE:
            break;
          case PENDING_COLLECTED_GC_STATE:
            {
#ifdef DEBUG_LEGION
              // Precondition should have triggered if we're here
              assert(collection_ready.has_triggered());
#endif
              // Check to see if there were any collection guards we
              // were unable to acquire on remote nodes or whether there
              // are still packed valid reference outstanding
              if ((failed_collection_count.load() > 0) ||
                  (total_sent_references != total_received_references))
              {
                // See if we're the last release, if not then we
                // keep it in this state
                if (--pending_changes == 0)
                  gc_state = COLLECTABLE_GC_STATE;
              }
              else
              {
                // Deletion success and we're the first ones to discover it
                // Move to the deletion state and send the deletion messages
                // to mark that we successfully performed the deletion
                gc_state = COLLECTED_GC_STATE;
                // Grab the set of active contexts to notify
                std::set<InstanceDeletionSubscriber*> to_notify;
                // Notify the subscribers if we've been collected
                to_notify.swap(subscribers);
                // Now we can perform the deletion which will release the lock
                ready = perform_deletion(runtime->address_space, &i_lock);
                // Send notification messages to the remote nodes to tell
                // them that this instance has been deleted, this is needed
                // so that we can invalidate any subscribers on those nodes
                if (collective_mapping != NULL)
                {
#ifdef DEBUG_LEGION
                  // We're the owner so it should contain ourselves
                  assert(collective_mapping->contains(local_space));
#endif
                  std::vector<AddressSpaceID> children;
                  collective_mapping->get_children(owner_space, local_space,
                                                   children);
                  if (!children.empty())
                  {
                    pack_global_ref(children.size());
                    for (std::vector<AddressSpaceID>::const_iterator it =
                          children.begin(); it != children.end(); it++)
                    {
                      Serializer rez;
                      {
                        RezCheck z(rez);
                        rez.serialize(did);
                      }
                      runtime->send_gc_notify(*it, rez);
                    }
                  }
                }
                const size_t needed_guards = count_remote_instances();
                if (needed_guards > 0)
                {
                  struct NotifyFunctor {
                    NotifyFunctor(DistributedID d, Runtime *rt) 
                      : did(d), runtime(rt), count(0) { }
                    inline void apply(AddressSpaceID target)
                    {
                      if (target == runtime->address_space)
                        return;
                      Serializer rez;
                      {
                        RezCheck z(rez);
                        rez.serialize(did);
                      }
                      runtime->send_gc_notify(target, rez);
                      count++;
                    }
                    const DistributedID did;
                    Runtime *const runtime;
                    unsigned count;
                  };
                  NotifyFunctor functor(did, runtime);
                  map_over_remote_instances(functor);
                  if (functor.count > 0)
                    pack_global_ref(functor.count);
                }
                // Now that the lock is released we can notify the subscribers
                if (!to_notify.empty())
                {
                  for (std::set<InstanceDeletionSubscriber*>::const_iterator
                        it = to_notify.begin(); it != to_notify.end(); it++)
                  {
                    (*it)->notify_instance_deletion(this);
                    if ((*it)->remove_subscriber_reference(this))
                      delete (*it);
                  }
                }
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
        pack_global_ref();
        runtime->send_gc_request(owner_space, rez);
        done.wait();
        return result.load();
      }
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalManager::set_garbage_collection_priority(MapperID mapper_id,
                        Processor p, AddressSpaceID source, GCPriority priority)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_external_instance());
#endif
      RtEvent wait_on;
      RtUserEvent done_event;
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
              // Check the garbage collection state because this is going 
              // to be like an acquire operation
              switch (gc_state)
              {
                case VALID_GC_STATE:
                  break;
                case COLLECTABLE_GC_STATE:
                // Garbage collector is trying to eat it, save it!
                case PENDING_COLLECTED_GC_STATE:
                  {
                    gc_state = VALID_GC_STATE;
                    break;
                  }
                default:
                  assert(false);
              }
              // Update the references
#ifdef LEGION_GC
              log_base_ref<true>(VALID_REF_KIND, did, 
                                 local_space, NEVER_GC_REF, 1);
#endif
#ifdef DEBUG_LEGION_GC
              valid_references++;
              std::map<ReferenceSource,int>::iterator finder = 
                detailed_base_valid_references.find(NEVER_GC_REF);
              if (finder == detailed_base_valid_references.end())
                detailed_base_valid_references[NEVER_GC_REF] = 1;
              else
                finder->second++;
#else
              valid_references.fetch_add(1);
#endif
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
      const RtEvent updated = 
        update_garbage_collection_priority(source, priority);
      if (remove_never_reference && remove_base_valid_ref(NEVER_GC_REF))
        assert(false); // should never end up deleting ourselves
      Runtime::trigger_event(done_event, updated);
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
          runtime->find_distributed_collectable(did));

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
                        0/*default mapper ID*/, fake_proc, source, priority));
      manager->unpack_global_ref();
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
      DistributedID repl_id, ctx_did;
      derez.deserialize(repl_id);
      derez.deserialize(ctx_did);
      RtEvent ctx_ready;
      InnerContext *context = NULL;
      if (repl_id != ctx_did)
      {
        // See if we're on a node where there is a shard manager for
        // this replicated context
        ShardManager *shard_manager = 
          runtime->find_shard_manager(repl_id, true/*can fail*/);
        if (shard_manager != NULL)
          context = shard_manager->find_local_context();
      }
      if (context == NULL)
        context = runtime->find_or_request_inner_context(ctx_did, ctx_ready);
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

    //--------------------------------------------------------------------------
    void PhysicalManager::initialize_across_helper(CopyAcrossHelper *helper,
                                                   const FieldMask &dst_mask,
                                     const std::vector<unsigned> &src_indexes,
                                     const std::vector<unsigned> &dst_indexes)
    //--------------------------------------------------------------------------
    {
      // Make sure the instance is ready before we compute the offsets
      if (instance_ready.exists() && !instance_ready.has_triggered())
        instance_ready.wait();
#ifdef DEBUG_LEGION
      assert(src_indexes.size() == dst_indexes.size());
#endif
      std::vector<CopySrcDstField> dst_fields;
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
    void PhysicalManager::send_manager(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert((collective_mapping == NULL) ||
          !collective_mapping->contains(target));
#endif
      Serializer rez;
      {
        AutoLock lock(inst_lock,1,false/*exlcusive*/);
        RezCheck z(rez);
        rez.serialize(did);
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
        rez.serialize(kind);
        pack_garbage_collection_state(rez, target, false/*need lock*/);
      }
      context->runtime->send_instance_manager(target, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_send_manager(Runtime *runtime, 
                                     AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
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
      InstanceKind kind;
      derez.deserialize(kind);
      GarbageCollectionState gc_state;
      derez.deserialize(gc_state);

      if (domain_ready.exists() || fs_ready.exists() || layout_ready.exists())
      {
        const RtEvent precondition = 
          Runtime::merge_events(domain_ready, fs_ready, layout_ready);
        if (precondition.exists() && !precondition.has_triggered())
        {
          // We need to defer this instance creation
          DeferPhysicalManagerArgs args(did, mem, inst,
              inst_footprint, inst_domain, pending, 
              handle, tree_id, layout_id, unique_event, kind,
              redop, piece_list, piece_list_size, gc_state);
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
      create_remote_manager(runtime, did, mem, inst, inst_footprint,
                            inst_domain, piece_list, piece_list_size, 
                            space_node, tree_id, constraints, unique_event, 
                            kind, redop, gc_state);
    }

    //--------------------------------------------------------------------------
    PhysicalManager::DeferPhysicalManagerArgs::DeferPhysicalManagerArgs(
            DistributedID d, Memory m, PhysicalInstance i, 
            size_t f, IndexSpaceExpression *lx, 
            const PendingRemoteExpression &p, FieldSpace h, RegionTreeID tid,
            LayoutConstraintID l, ApEvent u, InstanceKind k, ReductionOpID r,
            const void *pl, size_t pl_size, GarbageCollectionState gc)
      : LgTaskArgs<DeferPhysicalManagerArgs>(implicit_provenance),
            did(d), mem(m), inst(i), footprint(f), pending(p),
            local_expr(lx), handle(h), tree_id(tid), layout_id(l), 
            use_event(u), kind(k), redop(r), piece_list(pl),
            piece_list_size(pl_size), state(gc)
    //--------------------------------------------------------------------------
    {
      if (local_expr != NULL)
        local_expr->add_base_expression_reference(META_TASK_REF);
    }

    //--------------------------------------------------------------------------
    PhysicalManager::DeferDeletePhysicalManager
                     ::DeferDeletePhysicalManager(PhysicalManager *manager_)
      : LgTaskArgs<DeferDeletePhysicalManager>(implicit_provenance),
        manager(manager_), done(Runtime::create_rt_user_event())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_defer_manager(const void *args,
                                                          Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const DeferPhysicalManagerArgs *dargs = 
        (const DeferPhysicalManagerArgs*)args; 
      IndexSpaceExpression *inst_domain = dargs->local_expr;
      if (inst_domain == NULL)
        inst_domain = runtime->forest->find_remote_expression(dargs->pending);
      FieldSpaceNode *space_node = runtime->forest->get_node(dargs->handle);
      LayoutConstraints *constraints = 
        runtime->find_layout_constraints(dargs->layout_id);
      create_remote_manager(runtime, dargs->did, dargs->mem,
          dargs->inst, dargs->footprint, inst_domain, dargs->piece_list,
          dargs->piece_list_size, space_node, dargs->tree_id, constraints, 
          dargs->use_event, dargs->kind, dargs->redop, dargs->state);
      // Remove the local expression reference if necessary
      if ((dargs->local_expr != NULL) &&
          dargs->local_expr->remove_base_expression_reference(META_TASK_REF))
        delete dargs->local_expr;
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_defer_perform_deletion(
                                             const void *args, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const DeferDeletePhysicalManager *dargs =
        (const DeferDeletePhysicalManager*)args;
      Runtime::trigger_event(dargs->done,
          dargs->manager->perform_deletion(runtime->address_space));
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::create_remote_manager(Runtime *runtime, 
          DistributedID did, Memory mem, 
          PhysicalInstance inst, size_t inst_footprint, 
          IndexSpaceExpression *inst_domain, const void *piece_list,
          size_t piece_list_size, FieldSpaceNode *space_node, 
          RegionTreeID tree_id, LayoutConstraints *constraints, 
          ApEvent use_event, InstanceKind kind, ReductionOpID redop,
          GarbageCollectionState state)
    //--------------------------------------------------------------------------
    {
      LayoutDescription *layout = 
        LayoutDescription::handle_unpack_layout_description(constraints,
                                space_node, inst_domain->get_num_dims());
      MemoryManager *memory = runtime->find_memory_manager(mem);
      const ReductionOp *op = 
        (redop == 0) ? NULL : runtime->get_reduction(redop);
      void *location;
      PhysicalManager *man = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        man = new(location) PhysicalManager(runtime->forest, did,
                                            memory, inst, inst_domain, 
                                            piece_list, piece_list_size, 
                                            space_node, tree_id, layout, 
                                            redop, false/*reg now*/, 
                                            inst_footprint, use_event, 
                                            kind, op);
      else
        man = new PhysicalManager(runtime->forest, did, memory, 
                              inst, inst_domain, piece_list, piece_list_size,
                              space_node, tree_id, layout, redop, 
                              false/*reg now*/, inst_footprint, use_event, 
                              kind, op);
      man->initialize_remote_gc_state(state);
      // Hold-off doing the registration until construction is complete
      man->register_with_runtime();
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalManager::perform_deletion(AddressSpaceID source,
                                              AutoLock *i_lock /* = NULL*/)
    //--------------------------------------------------------------------------
    {
      if (i_lock == NULL)
      {
        AutoLock instance_lock(inst_lock);
        return perform_deletion(source, &instance_lock);
      }
      if (instance_ready.exists() && !instance_ready.has_triggered())
      {
        DeferDeletePhysicalManager args(this);
        runtime->issue_runtime_meta_task(
            args, LG_LOW_PRIORITY, instance_ready);
        return args.done;
      }
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(source == local_space);
      assert(pending_views.empty());
#endif
      log_garbage.spew("Deleting physical instance " IDFMT " in memory " 
                       IDFMT "", instance.id, memory_manager->memory.id);
#ifndef LEGION_DISABLE_GC
      RtEvent deferred_deletion;
      // Get the deferred deletion event from the gc events
      if (!gc_events.empty())
        deferred_deletion = Runtime::protect_merge_events(gc_events);
      // Now we can release the lock since we're done with the atomic updates
      i_lock->release();
      std::vector<PhysicalInstance::DestroyedField> serdez_fields;
      layout->compute_destroyed_fields(serdez_fields);
#ifndef LEGION_MALLOC_INSTANCES
      if (kind == EAGER_INSTANCE_KIND)
        memory_manager->free_eager_instance(instance, deferred_deletion);
      else
#endif
      {
        if (!serdez_fields.empty())
          instance.destroy(serdez_fields, deferred_deletion);
        else
          instance.destroy(deferred_deletion);
      }
#ifdef LEGION_MALLOC_INSTANCES
      if (!is_external_instance())
        memory_manager->free_legion_instance(instance, deferred_deletion);
#endif
#else
      // Release the i_lock since we're done with the atomic updates
      i_lock->release();
#endif
      // We issued the deletion to Realm so all our effects are done
      return RtEvent::NO_RT_EVENT;
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
#ifndef LEGION_DISABLE_GC
      std::vector<PhysicalInstance::DestroyedField> serdez_fields;
      layout->compute_destroyed_fields(serdez_fields);
#ifndef LEGION_MALLOC_INSTANCES
      // If this is an eager allocation, return it back to the eager pool
      if (kind == EAGER_INSTANCE_KIND)
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
#endif
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalManager::update_garbage_collection_priority(
                                     AddressSpaceID source, GCPriority priority)
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
        pack_global_ref();
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
    RtEvent PhysicalManager::attach_external_instance(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_external_instance());
#endif
      return memory_manager->attach_external_instance(this);
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
    
    //--------------------------------------------------------------------------
    uintptr_t PhysicalManager::get_instance_pointer(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      void *inst_ptr = instance.pointer_untyped(0/*offset*/, 0/*elem size*/);
      return uintptr_t(inst_ptr);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::has_visible_from(const std::set<Memory> &mems) const
    //--------------------------------------------------------------------------
    {
      return (mems.find(memory_manager->memory) != mems.end());
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::update_physical_instance(
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
        assert(external_pointer != -1UL);
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
    void PhysicalManager::broadcast_manager_update(void)
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
    /*static*/ void PhysicalManager::handle_send_manager_update(
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

      if (manager->update_physical_instance(instance, kind, footprint))
        delete manager;
    }

#ifdef NO_EXPLICIT_COLLECTIVES
    /////////////////////////////////////////////////////////////
    // IndividualManager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ApEvent IndividualManager::fill_from(FillView *fill_view, 
                                         InstanceView *dst_view,
                                         ApEvent precondition,
                                         PredEvent predicate_guard,
                                         IndexSpaceExpression *fill_expression,
                                         Operation *op, const unsigned index,
                                         const FieldMask &fill_mask,
                                         const PhysicalTraceInfo &trace_info,
                                         std::set<RtEvent> &recorded_events,
                                         std::set<RtEvent> &applied_events,
                                         CopyAcrossHelper *across_helper,
                                         const bool manage_dst_events,
                                         const bool fill_restricted,
                                         const bool need_valid_return)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dst_view->manager == this);
      assert((across_helper == NULL) || !manage_dst_events);
#endif
      // Compute the precondition first
      if (manage_dst_events)
      {
        ApEvent dst_precondition = dst_view->find_copy_preconditions(
            false/*reading*/, 0/*redop*/, fill_mask, fill_expression,
            op->get_unique_op_id(), index, applied_events, trace_info);
        if (dst_precondition.exists())
        {
          if (dst_precondition.exists())
            precondition =
              Runtime::merge_events(&trace_info,precondition,dst_precondition);
          else
            precondition = dst_precondition;
        }
      }
      std::vector<CopySrcDstField> dst_fields;
      if (across_helper != NULL)
      {
        const FieldMask src_mask = across_helper->convert_dst_to_src(fill_mask);
        across_helper->compute_across_offsets(src_mask, dst_fields);
      }
      else
        compute_copy_offsets(fill_mask, dst_fields); 
      const ApEvent result = fill_expression->issue_fill(op, trace_info,
                                                 dst_fields,
                                                 fill_view->value->value,
                                                 fill_view->value->value_size,
#ifdef LEGION_SPY
                                                 fill_view->fill_op_uid,
                                                 field_space_node->handle,
                                                 tree_id,
#endif
                                                 precondition, predicate_guard);
      // Save the result
      if (manage_dst_events && result.exists())
      {
        const RtEvent collect_event = trace_info.get_collect_event();
        dst_view->add_copy_user(false/*reading*/, 0/*redop*/, result, 
          collect_event, fill_mask, fill_expression, op->get_unique_op_id(),
          index, recorded_events, trace_info.recording, runtime->address_space);
      }
      if (trace_info.recording)
      {
        const DomainPoint no_point;
        const UniqueInst dst_inst(dst_view, no_point);
        trace_info.record_fill_inst(result, fill_expression, dst_inst,
                                    fill_mask, applied_events, (redop > 0));
      }
      return result;
    }

    //--------------------------------------------------------------------------
    ApEvent IndividualManager::copy_from(InstanceView *src_view,
                                         InstanceView *dst_view,
                                         PhysicalManager *source_manager,
                                         ApEvent precondition,
                                         PredEvent predicate_guard, 
                                         ReductionOpID reduction_op_id,
                                         IndexSpaceExpression *copy_expression,
                                         Operation *op, const unsigned index,
                                         const FieldMask &copy_mask,
                                         const DomainPoint &src_point,
                                         const PhysicalTraceInfo &trace_info,
                                         std::set<RtEvent> &recorded_events,
                                         std::set<RtEvent> &applied_events,
                                         CopyAcrossHelper *across_helper,
                                         const bool manage_dst_events,
                                         const bool copy_restricted,
                                         const bool need_valid_return)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dst_view->manager == this);
      assert(src_view->manager == source_manager);
      assert((across_helper == NULL) || !manage_dst_events);
#endif
      // Compute the preconditions first
      const UniqueID op_id = op->get_unique_op_id();
      // We'll need to compute our destination precondition no matter what
      if (manage_dst_events)
      {
        const ApEvent dst_pre = dst_view->find_copy_preconditions(
          false/*reading*/, reduction_op_id, copy_mask, copy_expression,
          op_id, index, applied_events, trace_info);  
        if (dst_pre.exists())
        {
          if (precondition.exists())
            precondition =
              Runtime::merge_events(&trace_info, precondition, dst_pre);
          else
            precondition = dst_pre;
        }
      }
      const FieldMask *src_mask = (across_helper == NULL) ? &copy_mask :
          new FieldMask(across_helper->convert_dst_to_src(copy_mask));
      // Several cases here:
      // 1. The source is another individual manager - just straight up 
      //    compute the dependences and do the copy or reduction
      // 2. The source is a normal collective manager - issue a copy from
      //    an instance close to the destination instance
      // 3. The source is a reduction collective manager - build a reduction
      //    tree down to a source instance close to the destination instance
      ApEvent result;
      if (!source_manager->is_collective_manager())
      {
        // Case 1: Source manager is another instance manager
        const ApEvent src_pre = src_view->find_copy_preconditions(
            true/*reading*/, 0/*redop*/, *src_mask, copy_expression,
            op_id, index, applied_events, trace_info);
        if (src_pre.exists())
        {
          if (precondition.exists())
            precondition =
              Runtime::merge_events(&trace_info, precondition, src_pre);
          else
            precondition = src_pre;
        }
        // Compute the field offsets
        std::vector<CopySrcDstField> dst_fields, src_fields;
        if (across_helper == NULL)
          compute_copy_offsets(copy_mask, dst_fields);
        else
          across_helper->compute_across_offsets(*src_mask, dst_fields);
        source_manager->compute_copy_offsets(*src_mask, src_fields);
        std::vector<Reservation> reservations;
        // If we're doing a reduction operation then set the reduction
        // information on the source-dst fields
        if (reduction_op_id > 0)
        {
#ifdef DEBUG_LEGION
          assert((redop == 0) || (redop == reduction_op_id));
#endif
          // Get the reservations
          const DomainPoint nopoint;
          dst_view->find_field_reservations(copy_mask, nopoint, reservations);
          // Set the redop on the destination fields
          // Note that we can mark these as exclusive copies since
          // we are protecting them with the reservations
          for (unsigned idx = 0; idx < dst_fields.size(); idx++)
            dst_fields[idx].set_redop(reduction_op_id, (redop > 0)/*fold*/,
                                      true/*exclusive*/);
        }
        result = copy_expression->issue_copy(op, trace_info, dst_fields,
                                             src_fields, reservations,
#ifdef LEGION_SPY
                                             source_manager->tree_id, tree_id,
#endif
                                             precondition, predicate_guard);
        if (result.exists())
        {
          const RtEvent collect_event = trace_info.get_collect_event();
          src_view->add_copy_user(true/*reading*/, 0/*redop*/, result,
              collect_event, *src_mask, copy_expression, op_id, index,
              recorded_events, trace_info.recording, runtime->address_space);
          if (manage_dst_events)
            dst_view->add_copy_user(false/*reading*/, reduction_op_id, result,
                collect_event, copy_mask, copy_expression, op_id, index,
              recorded_events, trace_info.recording, runtime->address_space);
        }
        if (trace_info.recording)
        {
          const DomainPoint no_point;
          const UniqueInst src_inst(src_view, src_point);
          const UniqueInst dst_inst(dst_view, no_point);
          trace_info.record_copy_insts(result, copy_expression, src_inst,
              dst_inst, *src_mask, copy_mask, reduction_op_id, applied_events);
        }
      }
      else
      {
        CollectiveManager *collective = source_manager->as_collective_manager();
        std::vector<CopySrcDstField> dst_fields;
        if (across_helper == NULL)
          compute_copy_offsets(copy_mask, dst_fields);
        else
          across_helper->compute_across_offsets(*src_mask, dst_fields);
        std::vector<Reservation> reservations;
        if (reduction_op_id > 0)
        {
#ifdef DEBUG_LEGION
          assert((redop == 0) || (redop == reduction_op_id));
#endif
          const DomainPoint nopoint;
          dst_view->find_field_reservations(copy_mask, nopoint, reservations);
          // Set the redop on the destination fields
          // Note that we can mark these as exclusive copies since
          // we are protecting them with the reservations
          for (unsigned idx = 0; idx < dst_fields.size(); idx++)
            dst_fields[idx].set_redop(reduction_op_id, (redop > 0)/*fold*/,
                                      true/*exclusive*/);
        }
        if (collective->is_reduction_manager())
        {
#ifdef DEBUG_LEGION
          assert(reduction_op_id == collective->redop);
#endif
          // Case 3
          // This is subtle as fuck
          // In the normal case where we're doing a reduction from a
          // collective instance to a normal instance then we can get
          // away with just building the reduction tree.
          //
          // An important note here: we only need to build a reduction tree
          // and not do an all-reduce for the collective reduction instance
          // because we know the equivalence set code above will only ever
          // issue a single copy from a reduction instance into another 
          // instance before that reduction instance is refreshed, so it
          // is safe to break the invariant that all instances in the 
          // collective manager have the same data.
          //
          // However, in the case where we are doing a copy-across, then we
          // might still be asked to do an intra-region reduction later so 
          // it's unsafe to do the partial accumulations into our own
          // instances. Therefore for now we will hammer all the source
          // instances into the destination instance without any
          // intermediate reductions.
          const DomainPoint no_point;
          const UniqueInst dst_inst(dst_view, no_point);
          if (manage_dst_events)
          {
            // Reduction-tree case
            const AddressSpaceID origin = src_point.exists() ?
              collective->get_instance(src_point).address_space() :
              collective->select_source_space(owner_space);
            // There will always be a single result for this copy
            if (origin != local_space)
            {
              const RtUserEvent recorded = Runtime::create_rt_user_event();
              const RtUserEvent applied = Runtime::create_rt_user_event();
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(collective->did);
                rez.serialize(src_view->did);
                pack_fields(rez, dst_fields);
                rez.serialize<size_t>(reservations.size());
                for (unsigned idx = 0; idx < reservations.size(); idx++)
                  rez.serialize(reservations[idx]);
                rez.serialize(precondition);
                rez.serialize(predicate_guard);
                copy_expression->pack_expression(rez, origin);
                op->pack_remote_operation(rez, origin, applied_events);
                rez.serialize(index);
                rez.serialize(*src_mask);
                rez.serialize(copy_mask);
                rez.serialize(src_point);
                dst_inst.serialize(rez);
                trace_info.pack_trace_info(rez, applied_events);
                rez.serialize(recorded);
                rez.serialize(applied);
                if (trace_info.recording)
                {
                  ApBarrier bar(Realm::Barrier::create_barrier(1/*arrivals*/));
                  const ShardID sid = trace_info.record_managed_barrier(bar, 1);
                  rez.serialize(bar);
                  if (bar.exists())
                    rez.serialize(sid);
                  result = bar;
                }
                else
                {
                  const ApUserEvent to_trigger =
                    Runtime::create_ap_user_event(&trace_info);
                  result = to_trigger;
                  rez.serialize(to_trigger);
                }
                rez.serialize(origin);
              }
              runtime->send_collective_distribute_reduction(origin, rez);
              recorded_events.insert(recorded);
              applied_events.insert(applied);
            }
            else
            {
              const ApUserEvent to_trigger =
                Runtime::create_ap_user_event(&trace_info);
              result = to_trigger;
              collective->perform_collective_reduction(src_view, dst_fields,
                  reservations, precondition, predicate_guard, copy_expression,
                  op, index, *src_mask, copy_mask, src_point, dst_inst,
                  trace_info, recorded_events, applied_events, 
                  to_trigger, origin);
            }
          }
          else
          {
            // Hammer reduction case
            // Issue a performance warning if we're ever going to 
            // be doing this case and the number of instance is large
            if (collective->total_points > LEGION_COLLECTIVE_RADIX)
              REPORT_LEGION_WARNING(LEGION_WARNING_COLLECTIVE_HAMMER_REDUCTION,
                  "WARNING: Performing copy-across reduction hammer with %zd "
                  "instances into a single instance from collective manager "
                  "%llx to normal manager %llx. Please report this use case "
                  "to the Legion developers' mailing list.",
                  collective->total_points, collective->did, did)
            const AddressSpaceID origin =
              collective->select_source_space(owner_space);
            if (origin != local_space)
            {
              const RtUserEvent recorded = Runtime::create_rt_user_event();
              const RtUserEvent applied = Runtime::create_rt_user_event();
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(collective->did);
                rez.serialize(src_view->did);
                pack_fields(rez, dst_fields);
                rez.serialize<size_t>(reservations.size());
                for (unsigned idx = 0; idx < reservations.size(); idx++)
                  rez.serialize(reservations[idx]);
                rez.serialize(precondition);
                rez.serialize(predicate_guard);
                copy_expression->pack_expression(rez, origin);
                op->pack_remote_operation(rez, origin, applied_events);
                rez.serialize(index);
                rez.serialize(*src_mask);
                rez.serialize(copy_mask);
                dst_inst.serialize(rez);
                trace_info.pack_trace_info(rez, applied_events);
                rez.serialize(recorded);
                rez.serialize(applied);
                if (trace_info.recording)
                {
                  ApBarrier bar(Realm::Barrier::create_barrier(1/*arrivals*/));
                  ShardID sid = trace_info.record_managed_barrier(bar, 1);
                  rez.serialize(bar);
                  rez.serialize(sid);
                  result = bar;
                }
                else
                {
                  const ApUserEvent to_trigger =
                    Runtime::create_ap_user_event(&trace_info);
                  rez.serialize(to_trigger);             
                  result = to_trigger; 
                }
                rez.serialize(origin);
              }
              runtime->send_collective_hammer_reduction(origin, rez);
              recorded_events.insert(recorded);
              applied_events.insert(applied);
            }
            else
              result = collective->perform_hammer_reduction(src_view,
                  dst_fields, reservations, precondition, predicate_guard,
                  copy_expression, op, index, *src_mask, copy_mask, dst_inst,
                  trace_info, recorded_events, applied_events, origin);
          }
        }
        else
        {
          // Case 2
          // We can issue the copy from an instance in the source
          const Memory location = instance.get_location();
          const DomainPoint no_point;
          const AddressSpaceID origin = src_point.exists() ?
              collective->get_instance(src_point).address_space() :
              collective->select_source_space(owner_space);
          if (origin != local_space)
          {
            const RtUserEvent recorded = Runtime::create_rt_user_event();
            const RtUserEvent applied = Runtime::create_rt_user_event();
            ApUserEvent to_trigger = Runtime::create_ap_user_event(&trace_info);
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(collective->did);
              rez.serialize(src_view->did);
              pack_fields(rez, dst_fields);
              rez.serialize<size_t>(reservations.size());
              for (unsigned idx = 0; idx < reservations.size(); idx++)
                rez.serialize(reservations[idx]);
              rez.serialize(precondition);
              rez.serialize(predicate_guard);
              copy_expression->pack_expression(rez, origin);
              op->pack_remote_operation(rez, origin, applied_events);
              rez.serialize(index);
              rez.serialize(*src_mask);
              rez.serialize(copy_mask);
              rez.serialize(location);
              rez.serialize(dst_view->did);
              rez.serialize(no_point);
              rez.serialize(src_point);
              trace_info.pack_trace_info(rez, applied_events);
              rez.serialize(recorded);
              rez.serialize(applied);
              rez.serialize(to_trigger);             
            }
            runtime->send_collective_distribute_point(origin, rez);
            recorded_events.insert(recorded);
            applied_events.insert(applied);
            result = to_trigger;
          }
          else
            result = collective->perform_collective_point(src_view,
                dst_fields, reservations, precondition, predicate_guard,
                copy_expression, op, index, *src_mask, copy_mask, location, 
                dst_view->did, no_point, src_point, trace_info,
                recorded_events, applied_events);
        }
        if (result.exists() && manage_dst_events)
        {
          const RtEvent collect_event = trace_info.get_collect_event();
          dst_view->add_copy_user(false/*reading*/, reduction_op_id, result,
              collect_event, copy_mask, copy_expression, op_id, index,
            recorded_events, trace_info.recording, runtime->address_space);
        }
      } 
      if (across_helper != NULL)
        delete src_mask;
      return result;
    }

    //--------------------------------------------------------------------------
    void IndividualManager::pack_fields(Serializer &rez,
                               const std::vector<CopySrcDstField> &fields) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(fields.size());
      for (unsigned idx = 0; idx < fields.size(); idx++)
        rez.serialize(fields[idx]);
      if (runtime->legion_spy_enabled)
      {
        rez.serialize(ApEvent::NO_AP_EVENT);
        rez.serialize(did);
      }
    } 

    //--------------------------------------------------------------------------
    void IndividualManager::update_field_reservations(const FieldMask &mask,
           DistributedID view_did, const std::vector<Reservation> &reservations)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner());
#endif
      AutoLock i_lock(inst_lock);
      std::map<unsigned,Reservation> &atomic_reservations =
          view_reservations[view_did];
      unsigned offset = 0;
      for (int idx = mask.find_first_set(); idx >= 0;
            idx = mask.find_next_set(idx+1))
        atomic_reservations[idx] = reservations[offset++];
    }

    //--------------------------------------------------------------------------
    void IndividualManager::reclaim_field_reservations(DistributedID view_did,
                                            std::vector<Reservation> &to_delete)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        return;
      std::map<DistributedID,std::map<unsigned,Reservation> >::iterator
        finder = view_reservations.find(view_did);
      if (finder == view_reservations.end())
        return;
      for (std::map<unsigned,Reservation>::const_iterator it =
            finder->second.begin(); it != finder->second.end(); it++)
        to_delete.push_back(it->second);
      view_reservations.erase(finder);
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
                                         Operation *local_collective_op,
                                         const PhysicalTraceInfo &trace_info,
                                         const bool symbolic)
    //--------------------------------------------------------------------------
    {
      // Somewhat strangely we can still get calls to this method in cases
      // with control replication for things like acquire/release on individual
      // managers that represent file instances. In this case we'll just have
      // a single node perform the view analysis and then we broadcast out the
      // resulting event out to all the participants. 
#ifdef DEBUG_LEGION
      assert(mapping != NULL);
      assert(mapping->contains(local_space));
      assert(local_collective_op != NULL);
#endif
      const size_t local_collective_arrivals =
        local_collective_op->get_collective_local_arrivals();
      // First we need to decide which node is going to be the owner node
      // We'll prefer it to be the logical view owner since that is where
      // the event will be produced, otherwise, we'll just pick whichever
      // is closest to the logical view node
      const AddressSpaceID origin = mapping->contains(view->logical_owner) ?
        view->logical_owner : mapping->find_nearest(view->logical_owner);
      ApUserEvent result;
      RtUserEvent registered;
      std::vector<ApEvent> term_events;
      PhysicalTraceInfo *result_info = NULL;
      const RendezvousKey key(view->did, op_ctx_index, index);
      {
        AutoLock i_lock(inst_lock);
        // Check to see if we're the first one to arrive on this node
        std::map<RendezvousKey,UserRendezvous>::iterator finder =
          rendezvous_users.find(key);
        if (finder == rendezvous_users.end())
        {
          // If we are then make the record for knowing when we've seen
          // all the expected arrivals
          finder = rendezvous_users.insert(
              std::make_pair(key,UserRendezvous())).first; 
          UserRendezvous &rendezvous = finder->second;
          rendezvous.remaining_local_arrivals = local_collective_arrivals;
          rendezvous.local_initialized = true;
          rendezvous.remaining_remote_arrivals =
            mapping->count_children(origin, local_space);
          rendezvous.ready_event = Runtime::create_ap_user_event(&trace_info);
          rendezvous.trace_info = new PhysicalTraceInfo(trace_info);
          rendezvous.registered = Runtime::create_rt_user_event();
        }
        else if (!finder->second.local_initialized)
        {
#ifdef DEBUG_LEGION
          assert(!finder->second.ready_event.exists());
          assert(finder->second.trace_info == NULL);
#endif
          // First local arrival
          finder->second.remaining_local_arrivals = local_collective_arrivals;
          finder->second.ready_event =
            Runtime::create_ap_user_event(&trace_info);
          finder->second.trace_info = new PhysicalTraceInfo(trace_info);
          if (!finder->second.remote_ready_events.empty())
          {
            for (std::map<ApUserEvent,PhysicalTraceInfo*>::const_iterator it =
                  finder->second.remote_ready_events.begin(); it !=
                  finder->second.remote_ready_events.end(); it++)
            {
              Runtime::trigger_event(it->second, it->first, 
                                finder->second.ready_event);
              delete it->second;
            }
            finder->second.remote_ready_events.clear();
          }
        }
        result = finder->second.ready_event;
        result_info = finder->second.trace_info;
        registered = finder->second.registered;
        applied_events.insert(registered);
        if (term_event.exists())
          finder->second.term_events.push_back(term_event);
#ifdef DEBUG_LEGION
        assert(finder->second.local_initialized);
        assert(finder->second.remaining_local_arrivals > 0);
#endif
        // If we're still expecting arrivals then nothing to do yet
        if ((--finder->second.remaining_local_arrivals > 0) ||
            (finder->second.remaining_remote_arrivals > 0))
        {
          // We need to save the trace info no matter what
          if (finder->second.view == NULL)
          {
            if (local_space == origin)
            {
              // Save our state for performing the registration later
              finder->second.view = view;
              finder->second.usage = usage;
              finder->second.mask = new FieldMask(user_mask);
              finder->second.expr = expr;
              WrapperReferenceMutator mutator(applied_events);
              expr->add_nested_expression_reference(did, &mutator);
              finder->second.op_id = op_id;
              finder->second.collect_event = collect_event;
              finder->second.symbolic = symbolic;
            }
            else
            {
              finder->second.applied = Runtime::create_rt_user_event();
              applied_events.insert(finder->second.applied);
            }
          }
          else if (local_space != origin)
          {
#ifdef DEBUG_LEGION
            assert(finder->second.applied.exists());
#endif
            applied_events.insert(finder->second.applied);
          }
          return result;
        }
        term_events.swap(finder->second.term_events);
#ifdef DEBUG_LEGION
        assert(finder->second.remote_ready_events.empty());
#endif
        // We're done with our entry after this so no need to keep it
        rendezvous_users.erase(finder);
      }
      if (!term_events.empty())
        term_event = Runtime::merge_events(&trace_info, term_events);
      if (local_space != origin)
      {
        const AddressSpaceID parent = 
          collective_mapping->get_parent(origin, local_space);
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(view->did);
          rez.serialize(op_ctx_index);
          rez.serialize(index);
          rez.serialize(origin);
          mapping->pack(rez);
          result_info->pack_trace_info(rez, applied_events);
          rez.serialize(term_event);
          rez.serialize(result);
          rez.serialize(registered);
        }
        runtime->send_collective_individual_register_user(parent, rez);
      }
      else
      {
        std::set<RtEvent> registered_events; 
        const ApEvent ready = view->register_user(usage, user_mask, expr, op_id,
            op_ctx_index, index, term_event, collect_event, registered_events,
            NULL/*collective mapping*/, NULL/*no collective op*/, *result_info,
            runtime->address_space, symbolic);
        Runtime::trigger_event(result_info, result, ready);
        if (!registered_events.empty())
          Runtime::trigger_event(registered,
              Runtime::merge_events(registered_events));
        else
          Runtime::trigger_event(registered);
      }
      delete result_info;
      return result;
    }

    //--------------------------------------------------------------------------
    void IndividualManager::process_collective_user_registration(
                                            const DistributedID view_did,
                                            const size_t op_ctx_index,
                                            const unsigned index,
                                            const AddressSpaceID origin,
                                            const CollectiveMapping *mapping,
                                            const PhysicalTraceInfo &trace_info,
                                            ApEvent remote_term_event,
                                            ApUserEvent remote_ready_event,
                                            RtUserEvent remote_registered)
    //--------------------------------------------------------------------------
    {
      UserRendezvous to_perform;
      const RendezvousKey key(view_did, op_ctx_index, index);
      {
        AutoLock i_lock(inst_lock);
        // Check to see if we're the first one to arrive on this node
        std::map<RendezvousKey,UserRendezvous>::iterator finder =
          rendezvous_users.find(key);
        if (finder == rendezvous_users.end())
        {
          // If we are then make the record for knowing when we've seen
          // all the expected arrivals
          finder = rendezvous_users.insert(
              std::make_pair(key,UserRendezvous())).first; 
          UserRendezvous &rendezvous = finder->second;
          rendezvous.local_initialized = false;
          rendezvous.remaining_remote_arrivals =
            mapping->count_children(origin, local_space);
          // Don't make the ready event, that needs to be done with a
          // local trace_info
          rendezvous.registered = Runtime::create_rt_user_event();
        }
        if (remote_term_event.exists())
          finder->second.term_events.push_back(remote_term_event);
        Runtime::trigger_event(remote_registered, finder->second.registered);
        if (!finder->second.ready_event.exists())
          finder->second.remote_ready_events[remote_ready_event] =
            new PhysicalTraceInfo(trace_info);
        else
          Runtime::trigger_event(&trace_info, remote_ready_event, 
                                 finder->second.ready_event);
#ifdef DEBUG_LEGION
        assert(finder->second.remaining_remote_arrivals > 0);
#endif
        // Check to see if we've done all the arrivals
        if ((--finder->second.remaining_remote_arrivals > 0) ||
            !finder->second.local_initialized ||
            (finder->second.remaining_local_arrivals > 0))
          return;
#ifdef DEBUG_LEGION
        assert(finder->second.remote_ready_events.empty());
        assert(finder->second.trace_info != NULL);
#endif
        // Last needed arrival, see if we're the origin or not
        to_perform = std::move(finder->second);
        rendezvous_users.erase(finder);
      }
      ApEvent term_event;
      if (!to_perform.term_events.empty())
        term_event =
          Runtime::merge_events(to_perform.trace_info, to_perform.term_events);
      if (local_space != origin)
      {
#ifdef DEBUG_LEGION
        assert(to_perform.applied.exists());
        assert(external_pointer != -1UL);
#endif
        // Send the message to the parent
        const AddressSpaceID parent = 
            collective_mapping->get_parent(origin, local_space);
        std::set<RtEvent> applied_events;
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(view_did);
          rez.serialize(op_ctx_index);
          rez.serialize(index);
          rez.serialize(origin);
          mapping->pack(rez);
          to_perform.trace_info->pack_trace_info(rez, applied_events);
          rez.serialize(term_event);
          rez.serialize(to_perform.ready_event);
          rez.serialize(to_perform.registered);
        }
        runtime->send_collective_individual_register_user(parent, rez);
        if (!applied_events.empty())
          Runtime::trigger_event(to_perform.applied,
              Runtime::merge_events(applied_events));
        else
          Runtime::trigger_event(to_perform.applied);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!to_perform.applied.exists());
#endif
        std::set<RtEvent> registered_events;
        const ApEvent ready = to_perform.view->register_user(to_perform.usage,
            *to_perform.mask, to_perform.expr, to_perform.op_id, op_ctx_index,
            index, term_event, to_perform.collect_event, registered_events,
            NULL/*mapping*/, NULL/*no collective op*/, *to_perform.trace_info,
            runtime->address_space, to_perform.symbolic);
        Runtime::trigger_event(to_perform.trace_info, 
                      to_perform.ready_event, ready);
        if (!registered_events.empty())
          Runtime::trigger_event(to_perform.registered,
              Runtime::merge_events(registered_events));
        else
          Runtime::trigger_event(to_perform.registered);
        if (to_perform.expr->remove_nested_expression_reference(did))
          delete to_perform.expr;
        delete to_perform.mask;
      }
      delete to_perform.trace_info;
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualManager::handle_collective_user_registration(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      IndividualManager *manager = static_cast<IndividualManager*>(
              runtime->find_or_request_instance_manager(did, ready));
      DistributedID view_did;
      derez.deserialize(view_did);
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      unsigned index;
      derez.deserialize(index);
      AddressSpaceID origin;
      derez.deserialize(origin);
      size_t mapping_size;
      derez.deserialize(mapping_size);
#ifdef DEBUG_LEGION
      assert(mapping_size > 0);
#endif
      CollectiveMapping *mapping = new CollectiveMapping(derez, mapping_size);
      mapping->add_reference();
      PhysicalTraceInfo trace_info = 
        PhysicalTraceInfo::unpack_trace_info(derez, runtime); 
      ApEvent term_event;
      derez.deserialize(term_event);
      ApUserEvent ready_event;
      derez.deserialize(ready_event);
      RtUserEvent registered_event;
      derez.deserialize(registered_event);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();

      manager->process_collective_user_registration(view_did,op_ctx_index,index,
        origin, mapping, trace_info, term_event, ready_event, registered_event);
      if (mapping->remove_reference())
        delete mapping;
    }

    /////////////////////////////////////////////////////////////
    // Collective Manager
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CollectiveManager::CollectiveManager(RegionTreeForest *ctx, 
                                DistributedID did, AddressSpaceID owner_space,
                                const Domain &dense, size_t total,
                                CollectiveMapping *mapping,
                                IndexSpaceExpression *instance_domain,
                                const void *pl, size_t pl_size,
                                FieldSpaceNode *node, RegionTreeID tree_id,
                                LayoutDescription *desc, ReductionOpID redop_id,
                                bool register_now, size_t footprint,
                                bool external_instance, bool multi)
      : PhysicalManager(ctx, desc, encode_instance_did(did, external_instance,
            (redop_id != 0), true/*collective*/),
          owner_space, footprint, redop_id, (redop_id == 0) ? NULL : 
            ctx->runtime->get_reduction(redop_id),
          node, instance_domain, pl, pl_size, tree_id, register_now,
          false/*output*/, mapping),  total_points(total), dense_points(dense),
        unique_allreduce_tag(mapping->find_index(local_space)), 
        multi_instance(multi)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((dense_points.get_dim() == 0) || dense_points.dense());
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
      for (std::map<std::pair<DistributedID,DomainPoint>,
                    std::map<unsigned,Reservation> >::iterator it1 =
            view_reservations.begin(); it1 != view_reservations.end(); it1++)
      {
        // Skip any non-local points
        if (std::find(instance_points.begin(), instance_points.end(), 
                      it1->first.second) == instance_points.end())
          continue;
        for (std::map<unsigned,Reservation>::iterator it2 =
              it1->second.begin(); it2 != it1->second.end(); it2++)
          it2->second.destroy_reservation();
      }
    }

    //--------------------------------------------------------------------------
    bool CollectiveManager::contains_isomorphic_points(
                                                   IndexSpaceNode *points) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(collective_mapping != NULL);
#endif
      if (points->get_volume() != total_points)
        return false;
      ApEvent ready;
      const Domain point_domain = points->get_domain(ready, true/*need tight*/);
      if (ready.exists() && !ready.has_triggered_faultignorant())
        ready.wait_faultignorant();
      if (dense_points.get_dim() > 0)
      {
#ifdef DEBUG_LEGION
        assert(dense_points.get_volume() == points->get_volume());
#endif
        return (dense_points == point_domain);
      }
      // Have to do this the hard way by looking up all the points
      std::set<DomainPoint> known_points;
      {
        AutoLock i_lock(inst_lock,1,false/*exclusive*/);
        for (unsigned idx = 0; idx < instance_points.size(); idx++)
          known_points.insert(instance_points[idx]);
        for (std::map<DomainPoint,RemoteInstInfo>::const_iterator it =
              remote_points.begin(); it != remote_points.end(); it++)
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
      RtEvent wait_on;
      if (collective_mapping->contains(local_space))
      {
        std::vector<AddressSpaceID> child_spaces;
        collective_mapping->get_children(local_space, local_space,child_spaces);
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
            rez.serialize<size_t>(unknown_points.size());
            for (unsigned idx = 0; idx < unknown_points.size(); idx++)
              rez.serialize(unknown_points[idx]);
            rez.serialize(local_space);
            rez.serialize(local_space);
            rez.serialize(ready_event);
          }
          runtime->send_collective_point_request(*it, rez);
          ready_events.push_back(ready_event);
        }
        wait_on = Runtime::merge_events(ready_events);
      }
      else
      {
        const RtUserEvent ready_event = Runtime::create_rt_user_event();
        const AddressSpaceID origin = 
          collective_mapping->find_nearest(local_space);
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize<size_t>(unknown_points.size());
          for (unsigned idx = 0; idx < unknown_points.size(); idx++)
            rez.serialize(unknown_points[idx]);
          rez.serialize(local_space);
          rez.serialize(origin);
          rez.serialize(ready_event);
        }
        runtime->send_collective_point_request(origin, rez);
        wait_on = ready_event;
      }
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      {
        AutoLock i_lock(inst_lock,1,false/*exclusive*/);
        for (std::map<DomainPoint,RemoteInstInfo>::const_iterator it =
              remote_points.begin(); it != remote_points.end(); it++)
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
#ifdef DEBUG_LEGION
      assert(point.get_dim() > 0);
      assert(collective_mapping != NULL);
#endif
      if (dense_points.get_dim() > 0)
      {
        // If the dimensionalities are different then this will never succeed
        if (dense_points.get_dim() != point.get_dim())
          return false;
        return dense_points.contains(point);
      }
      // If the dimensionalities are different then this will never succeed
      if (!instance_points.empty() &&
          (instance_points.front().get_dim() != point.get_dim()))
        return false;
      // Check the local points first since they are read-only at this point
      for (std::vector<DomainPoint>::const_iterator it =
            instance_points.begin(); it != instance_points.end(); it++)
        if ((*it) == point)
          return true;
      {
        AutoLock i_lock(inst_lock,1,false/*exclusive*/);
        std::map<DomainPoint,RemoteInstInfo>::const_iterator finder =
          remote_points.find(point);
        if (finder != remote_points.end())
          return true;
      }
      // Broadcast out a request for this remote instance
      const RtEvent wait_on = broadcast_point_request(point); 
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      AutoLock i_lock(inst_lock,1,false/*exclusive*/);
      return (remote_points.find(point) != remote_points.end());
    }

    //--------------------------------------------------------------------------
    bool CollectiveManager::is_first_local_point(const DomainPoint &point) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(point.get_dim() > 0);
      assert(collective_mapping != NULL);
#endif
      // Check the local points first since they are read-only at this point
      for (unsigned idx = 0; idx < instance_points.size(); idx++)
        if (instance_points[idx] == point)
          return (idx == 0);
      {
        AutoLock i_lock(inst_lock,1,false/*exclusive*/);
        std::map<DomainPoint,RemoteInstInfo>::const_iterator finder =
          remote_points.find(point);
        if (finder != remote_points.end())
          return (finder->second.index == 0);
      }
      // Broadcast out a request for this remote instance
      const RtEvent wait_on = broadcast_point_request(point); 
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      AutoLock i_lock(inst_lock,1,false/*exclusive*/);
      std::map<DomainPoint,RemoteInstInfo>::const_iterator finder =
          remote_points.find(point);
#ifdef DEBUG_LEGION
      assert(finder != remote_points.end());
#endif
      return (finder->second.index == 0);
    }

    //--------------------------------------------------------------------------
    RtEvent CollectiveManager::broadcast_point_request(
                                                 const DomainPoint &point) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(collective_mapping != NULL);
#endif
      if (collective_mapping->contains(local_space))
      {
        std::vector<AddressSpaceID> child_spaces;
        collective_mapping->get_children(local_space, local_space,child_spaces);
        if (child_spaces.empty())
          return RtEvent::NO_RT_EVENT;
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
            rez.serialize<size_t>(1);
            rez.serialize(point);
            rez.serialize(local_space);
            rez.serialize(local_space);
            rez.serialize(ready_event);
          }
          runtime->send_collective_point_request(*it, rez);
          ready_events.push_back(ready_event);
        }
        return Runtime::merge_events(ready_events);
      }
      else
      {
        const RtUserEvent ready_event = Runtime::create_rt_user_event();
        const AddressSpaceID origin = 
          collective_mapping->find_nearest(local_space);
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize<size_t>(1);
          rez.serialize(point);
          rez.serialize(local_space);
          rez.serialize(origin);
          rez.serialize(ready_event);
        }
        runtime->send_collective_point_request(origin, rez);
        return ready_event;
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::find_or_forward_physical_instance(
                          AddressSpaceID source, AddressSpaceID origin,
                          std::set<DomainPoint> &points, RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(collective_mapping != NULL);
      assert(collective_mapping->contains(local_space));
#endif
      std::map<DomainPoint,RemoteInstInfo> found_insts;
      for (unsigned idx = 0; idx < instance_points.size(); idx++)
      {
        std::set<DomainPoint>::iterator finder = 
          points.find(instance_points[idx]);
        if (finder == points.end())
          continue;
        found_insts[instance_points[idx]] =
          RemoteInstInfo{instances[idx], instance_events[idx], idx}; 
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
          std::map<DomainPoint,RemoteInstInfo>::const_iterator finder =
            remote_points.find(*it);
          if (finder != remote_points.end())
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
          rez.serialize<size_t>(found_insts.size());
          for (std::map<DomainPoint,RemoteInstInfo>::const_iterator it =
                found_insts.begin(); it != found_insts.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second.instance);
            rez.serialize(it->second.unique_event);
            rez.serialize(it->second.index);
          }
          rez.serialize(ready_event);
        }
        runtime->send_collective_point_response(source, rez);
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
          rez.serialize<size_t>(points.size());
          for (std::set<DomainPoint>::const_iterator it =
                points.begin(); it != points.end(); it++)
            rez.serialize(*it);
          rez.serialize(source);
          rez.serialize(origin);
          rez.serialize(ready_event);
        }
        runtime->send_collective_point_request(child_spaces[idx], rez);
        ready_events.push_back(ready_event);
      }
      Runtime::trigger_event(to_trigger, Runtime::merge_events(ready_events));
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::record_remote_physical_instances(
                      const std::map<DomainPoint,RemoteInstInfo> &new_instances)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
      for (std::map<DomainPoint,RemoteInstInfo>::const_iterator it =
            new_instances.begin(); it != new_instances.end(); it++)
      {
        std::map<DomainPoint,RemoteInstInfo>::const_iterator finder =
          remote_points.find(it->first);
        if (finder == remote_points.end())
          remote_points.insert(*it);
#ifndef NDEBUG
        else
          assert(finder->second == it->second);
#endif
      }
#else
      remote_points.insert(new_instances.begin(), new_instances.end()); 
#endif
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::record_point_instance(const DomainPoint &point,
                                       PhysicalInstance instance, ApEvent ready)
    //--------------------------------------------------------------------------
    {
      const Memory mem = instance.get_location();
      MemoryManager *memory = runtime->find_memory_manager(mem);
#ifdef DEBUG_LEGION
      assert(memory->is_owner);
      assert((dense_points.get_dim() == 0) || dense_points.contains(point));
      assert(!runtime->legion_spy_enabled || ready.exists());
#endif
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_physical_instance(ready, instance.id, mem.id,
            instance_domain->expr_id, field_space_node->handle, tree_id, redop);
        layout->log_instance_layout(ready);
      }
      AutoLock i_lock(inst_lock);
      memories.push_back(memory);
      instances.push_back(instance);
      instance_points.push_back(point);
      instance_events.push_back(ready);
    }

    //--------------------------------------------------------------------------
    bool CollectiveManager::finalize_point_instance(const DomainPoint &point,
                                        bool success, bool acquire, bool remote)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_external_instance());
#endif
      for (unsigned idx = 0; idx < instance_points.size(); idx++)
      {
        if (instance_points[idx] != point)
          continue;
        if (!success)
        {
          // Destroy the instance since we didn't succeed in making all
          // the instances for the collective instance
          std::vector<PhysicalInstance::DestroyedField> serdez_fields;
          layout->compute_destroyed_fields(serdez_fields);
          if (!serdez_fields.empty())
            instances[idx].destroy(serdez_fields);
          else
            instances[idx].destroy();
#ifdef LEGION_MALLOC_INSTANCES
          memories[idx]->free_legion_instance(instances[idx],
                                              RtEvent::NO_RT_EVENT);
#endif
          return memories[idx]->remove_pending_collective_instance(this);
        }
        else
        {
          memories[idx]->finalize_pending_collective_instance(this,
                                                  acquire, remote);
          return false;
        }
      }
      // Should never get here
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_instance_creation(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // This comes from the MapperManager that sends it at the end
      // of making a collective instance when the point instance it
      // made was remote from the local node
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      CollectiveManager *manager = static_cast<CollectiveManager*>(
                          runtime->find_distributed_collectable(did));
      DomainPoint point;
      derez.deserialize(point);
      bool success, acquire;
      derez.deserialize<bool>(success);
      derez.deserialize<bool>(acquire);
      RtUserEvent done;
      derez.deserialize(done);

      if (manager->finalize_point_instance(point, success, acquire,
                                           true/*remote*/))
        delete manager;
      if (done.exists())
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    ApEvent CollectiveManager::get_use_event(ApEvent user) const
    //--------------------------------------------------------------------------
    {
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent CollectiveManager::get_unique_event(const DomainPoint &point) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(point.get_dim() > 0);
      assert(collective_mapping != NULL);
#endif
      // Check the local points first since they are read-only at this point
      for (unsigned idx = 0; idx < instance_points.size(); idx++)
        if (instance_points[idx] == point)
          return instance_events[idx];
      {
        AutoLock i_lock(inst_lock,1,false/*exclusive*/);
        std::map<DomainPoint,RemoteInstInfo>::const_iterator finder =
          remote_points.find(point);
        if (finder != remote_points.end())
          return finder->second.unique_event;
      }
      // Broadcast out a request for this remote instance
      const RtEvent wait_on = broadcast_point_request(point); 
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      AutoLock i_lock(inst_lock,1,false/*exclusive*/);
      std::map<DomainPoint,RemoteInstInfo>::const_iterator finder =
        remote_points.find(point);
#ifdef DEBUG_LEGION
      assert(finder != remote_points.end());
#endif
      return finder->second.unique_event;
    }

    //--------------------------------------------------------------------------
    PhysicalInstance CollectiveManager::get_instance(const DomainPoint &p,
                                                     bool from_mapper) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(collective_mapping != NULL);
#endif
      if (from_mapper && (p.get_dim() == 0))
        REPORT_LEGION_ERROR(ERROR_MAPPER_COLLECTIVE_INSTANCE_NOPOINT,
            "Mapper did not pass in a domain point when calling "
            "PhysicalInstance::get_instance or "
            "PhysicalInstance::get_location for a collective instance. "
            "Mappers must always specify a point when calling these "
            "methods on a collective instance.")
      // Check the local points first since they are read-only at this point
      for (unsigned idx = 0; idx < instance_points.size(); idx++)
        if (instance_points[idx] == p)
          return instances[idx];
      {
        AutoLock i_lock(inst_lock,1,false/*exclusive*/);
        std::map<DomainPoint,RemoteInstInfo>::const_iterator finder =
          remote_points.find(p);
        if (finder != remote_points.end())
          return finder->second.instance;
      }
      // Broadcast out a request for this remote instance
      const RtEvent wait_on = broadcast_point_request(p); 
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      AutoLock i_lock(inst_lock,1,false/*exclusive*/);
      std::map<DomainPoint,RemoteInstInfo>::const_iterator finder =
        remote_points.find(p);
      if (from_mapper && (finder == remote_points.end()))
        REPORT_LEGION_ERROR(ERROR_MAPPER_COLLECTIVE_INSTANCE_NOPOINT,
            "Unable to find point in collective instance from "
            "invocation of PhysicalInstance::get_instance or "
            "PhysicalInstance::get_location inside a mapper call.")
#ifdef DEBUG_LEGION
      assert(finder != remote_points.end());
#endif
      return finder->second.instance;
    }

    //--------------------------------------------------------------------------
    PointerConstraint CollectiveManager::get_pointer_constraint(
                                                 const DomainPoint &point) const
    //--------------------------------------------------------------------------
    {
      const PhysicalInstance instance = get_instance(point);
      void *inst_ptr = instance.pointer_untyped(0/*offset*/, 0/*elem size*/);
      return PointerConstraint(instance.get_location(), uintptr_t(inst_ptr));
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
      if (i_lock == NULL)
      {
        AutoLock instance_lock(inst_lock);
        return perform_deletion(source, &instance_lock);
      }
#ifdef DEBUG_LEGION
      assert(pending_views.empty());
      assert(!deferred_deletion.exists());
      assert(collective_mapping != NULL);
      // Should always be sending this along the tree
      assert((is_owner() && (source == local_space)) ||
        (collective_mapping->contains(local_space) && 
         (source == collective_mapping->get_parent(owner_space, local_space))));
#endif
      std::vector<RtEvent> done_events;
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(owner_space, local_space, children);
      for (std::vector<AddressSpaceID>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        const RtUserEvent done = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(local_space);
          rez.serialize(done);
        }
        runtime->send_collective_deletion(*it, rez);
        done_events.push_back(done);
      }
      prune_gc_events();
      // Grab the set of active contexts to notify
      std::set<InstanceDeletionSubscriber*> to_notify;
      to_notify.swap(subscribers);
      std::map<std::pair<DistributedID,DomainPoint>,
                std::map<unsigned,Reservation> > to_destroy;
      to_destroy.swap(view_reservations);
#ifndef LEGION_DISABLE_GC
      // If we're still active that means there are still outstanding
      // users so make an event for when we are done, not we're holding
      // the instance lock when this is called
      if (currently_active)
        deferred_deletion = Runtime::create_rt_user_event();
      // Now we can release the lock since we're done with the atomic updates
      i_lock->release();
      std::vector<PhysicalInstance::DestroyedField> serdez_fields;
      layout->compute_destroyed_fields(serdez_fields);
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        log_garbage.spew("Deleting physical instance " IDFMT " in memory " 
                         IDFMT "", instances[idx].id, memories[idx]->memory.id);
        if (!serdez_fields.empty())
          instances[idx].destroy(serdez_fields, deferred_deletion);
        else
          instances[idx].destroy(deferred_deletion);
#ifdef LEGION_MALLOC_INSTANCES
        if (!is_external_instance())
          memories[idx]->free_legion_instance(instances[idx],deferred_deletion);
#endif
      }
#else
      // Release the i_lock since we're done with the atomic updates
      i_lock->release();
#endif
      // Notify any contexts of our deletion
      if (!to_notify.empty())
      {
        for (std::set<InstanceDeletionSubscriber*>::const_iterator it =
              to_notify.begin(); it != to_notify.end(); it++)
        {
          (*it)->notify_instance_deletion(this);
          if ((*it)->remove_subscriber_reference(this))
            delete (*it);
        }
      }
      if (!to_destroy.empty())
      {
        for (std::map<std::pair<DistributedID,DomainPoint>,
                      std::map<unsigned,Reservation> >::iterator it1 =
              to_destroy.begin(); it1 != to_destroy.end(); it1++)
        {
          // Skip any non-local points
          if (std::find(instance_points.begin(), instance_points.end(), 
                        it1->first.second) == instance_points.end())
            continue;
          for (std::map<unsigned,Reservation>::iterator it2 =
                it1->second.begin(); it2 != it1->second.end(); it2++)
            it2->second.destroy_reservation();
        }
      }
      if (!done_events.empty())
        return Runtime::merge_events(done_events);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::force_deletion(void)
    //--------------------------------------------------------------------------
    {
#ifndef LEGION_DISABLE_GC
      // Have to deduplicate across force-delete calls here to handle the
      // case where we have multiple instances in the same memory
      std::vector<PhysicalInstance> to_destroy;
      {
        AutoLock i_lock(inst_lock);
        if (instances.empty())
          return;
        to_destroy.swap(instances);
      }
      std::vector<PhysicalInstance::DestroyedField> serdez_fields;
      layout->compute_destroyed_fields(serdez_fields);
      for (unsigned idx = 0; idx < to_destroy.size(); idx++)
      {
        if (!serdez_fields.empty())
          to_destroy[idx].destroy(serdez_fields);
        else
          to_destroy[idx].destroy();
#ifdef LEGION_MALLOC_INSTANCES
        if (!is_external_instance())
          memories[idx]->free_legion_instance(to_destroy[idx],
                                              RtEvent::NO_RT_EVENT);
#endif
      }
#endif
    }

    //--------------------------------------------------------------------------
    RtEvent CollectiveManager::update_garbage_collection_priority(
                                     AddressSpaceID source, GCPriority priority)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(collective_mapping != NULL);
#endif
      // Check to see if the source is in the collective mapping
      if (collective_mapping->contains(source))
      {
        std::vector<RtEvent> done_events;
        // Always send up the tree
        std::vector<AddressSpaceID> children;
        collective_mapping->get_children(owner_space, local_space, children); 
        for (std::vector<AddressSpaceID>::const_iterator it =
              children.begin(); it != children.end(); it++)
        {
          // Don't need to send back to any children that sent this to us
          if ((*it) == source)
            continue;
          const RtUserEvent done = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(priority);
            rez.serialize(done);
          }
          runtime->send_gc_priority_update(*it, rez);
          done_events.push_back(done);
        }
        if (local_space != owner_space)
        {
          const AddressSpaceID parent =
            collective_mapping->get_parent(owner_space, local_space);
          if (source != parent)
          {
            // Send down the tree too if we're not the owner this didn't come
            // from the parent space in the tree
            const RtUserEvent done = Runtime::create_rt_user_event();
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(priority);
              rez.serialize(done);
            }
            runtime->send_gc_priority_update(parent, rez);
            done_events.push_back(done);
          }
        }
        // Perform our local updates on the managers
#ifdef DEBUG_LEGION
        assert(!memories.empty());
#endif
        for (std::vector<MemoryManager*>::const_iterator it =
              memories.begin(); it != memories.end(); it++)
          (*it)->set_garbage_collection_priority(this, priority);
        if (!done_events.empty())
          return Runtime::merge_events(done_events);
        return RtEvent::NO_RT_EVENT;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(memories.empty());
#endif
        // Just send the update to the owner node
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
    void CollectiveManager::collective_deletion(RtEvent deferred_event)
    //--------------------------------------------------------------------------
    {
#ifndef LEGION_DISABLE_GC
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
      std::set<InstanceDeletionSubscribers*> copy_subscribers;
      std::map<std::pair<DistributedID,DomainPoint>,
                std::map<unsigned,Reservation> > copy_view_atomics;
      {
        AutoLock inst(inst_lock);
        if (subscribers.empty())
          return;
        copy_subscribers.swap(subscribers);
        copy_view_atomics.swap(view_reservations);
#ifdef DEBUG_LEGION
        assert(pending_views.empty());
#endif
        context_views.clear();
      }
      for (std::set<InstanceDeletionSubscribers*>::iterator it =
            copy_subscribers.begin(); it != copy_subscribers.end(); it++)
      {
        (*it)->notify_instance_deletion(this);
        if ((*it)->remove_subscriber_reference())
          delete (*it);
      }
      // Clean up any reservations that we own associated with this instance
      for (std::map<std::pair<DistributedID,DomainPoint>,
                    std::map<unsigned,Reservation> >::iterator it1 =
            copy_view_atomics.begin(); it1 != copy_view_atomics.end(); it1++)
      {
        // Skip any non-local points
        if (std::find(instance_points.begin(), instance_points.end(), 
                      it1->first.second) == instance_points.end())
          continue;
        for (std::map<unsigned,Reservation>::iterator it2 =
              it1->second.begin(); it2 != it1->second.end(); it2++)
          it2->second.destroy_reservation();
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::collective_force(void)
    //--------------------------------------------------------------------------
    {
#ifndef LEGION_DISABLE_GC
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
                                         InstanceView *dst_view,
                                         ApEvent precondition,
                                         PredEvent predicate_guard,
                                         IndexSpaceExpression *fill_expression,
                                         Operation *op, const unsigned index,
                                         const FieldMask &fill_mask,
                                         const PhysicalTraceInfo &trace_info,
                                         std::set<RtEvent> &recorded_events,
                                         std::set<RtEvent> &applied_events,
                                         CopyAcrossHelper *across_helper,
                                         const bool manage_dst_events,
                                         const bool fill_restricted,
                                         const bool need_valid_return)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Should never have a copy-across with a collective manager as the target
      assert(manage_dst_events);
      assert(across_helper == NULL);
      assert(collective_mapping != NULL);
#endif
      // This one is easy, just tree broadcast out to all the nodes and 
      // perform the fill operation on each one of them
      ApEvent result;
      if (need_valid_return)
        result = Runtime::create_ap_user_event(&trace_info);
      if (!collective_mapping->contains(local_space))
      {
        // This node doesn't have any instances, so start at one that
        // is contained within the collective mapping
        AddressSpaceID origin = collective_mapping->find_nearest(local_space);
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(fill_view->did);
          rez.serialize(dst_view->did);
          rez.serialize(precondition);
          rez.serialize(predicate_guard);
          fill_expression->pack_expression(rez, origin);
          rez.serialize<bool>(fill_restricted);
          if (fill_restricted)
            op->pack_remote_operation(rez, origin, applied_events);
          rez.serialize(index);
          rez.serialize(op->get_ctx_index());
          rez.serialize(fill_mask);
          trace_info.pack_trace_info(rez, applied_events);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (trace_info.recording)
          {
            ApBarrier bar;
            ShardID sid = 0;
            if (need_valid_return)
            {
              bar = ApBarrier(Realm::Barrier::create_barrier(1/*arrivals*/));
              sid = trace_info.record_managed_barrier(bar, 1/*arrivals*/);
              result = bar;
            }
            rez.serialize(bar);
            if (bar.exists())
              rez.serialize(sid);
          }
          else
          {
            ApUserEvent to_trigger;
            if (need_valid_return)
            {
              to_trigger = Runtime::create_ap_user_event(&trace_info);
              result = to_trigger;
            }
            rez.serialize(to_trigger);
          }
          rez.serialize(origin);
        }
        runtime->send_collective_distribute_fill(origin, rez);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      else
      {
        ApUserEvent to_trigger;
        if (need_valid_return)
        {
          to_trigger = Runtime::create_ap_user_event(&trace_info);
          result = to_trigger;
        }
        perform_collective_fill(fill_view, dst_view, precondition,
            predicate_guard, fill_expression, op, index, op->get_ctx_index(),
            fill_mask, trace_info, recorded_events, applied_events,
            to_trigger, local_space, fill_restricted);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::perform_collective_fill(FillView *fill_view, 
                                         InstanceView *dst_view,
                                         ApEvent precondition,
                                         PredEvent predicate_guard,
                                         IndexSpaceExpression *fill_expression,
                                         Operation *op, const unsigned index,
                                         const size_t op_context_index,
                                         const FieldMask &fill_mask,
                                         const PhysicalTraceInfo &trace_info,
                                         std::set<RtEvent> &recorded_events,
                                         std::set<RtEvent> &applied_events,
                                         ApUserEvent ready_event,
                                         AddressSpaceID origin,
                                         const bool fill_restricted)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(collective_mapping != NULL);
      assert(collective_mapping->contains(local_space));
      assert((op != NULL) || !fill_restricted);
#endif
      RtEvent analyses_ready;
      const std::vector<CollectiveCopyFillAnalysis*> *local_analyses = NULL;
      if (!fill_restricted)
      {
        // If this is not a fill-out to a restricted collective instance 
        // then we should be able to find our local analyses to use for 
        // performing operations
        analyses_ready = find_collective_analyses(dst_view->did,
                          op_context_index, index, local_analyses);
#ifdef DEBUG_LEGION
        assert(local_analyses != NULL);
#endif
        // If we're recording then we need to wait now to get a valid
        // trace info for capturing the trace for events we send to 
        // remote nodes, otherwise we just need to wait before doing
        // the fill calls
        if ((trace_info.recording || (op == NULL)) && 
            analyses_ready.exists() && !analyses_ready.has_triggered())
          analyses_ready.wait();
#ifdef DEBUG_LEGION
        assert(local_analyses != NULL);
#endif
        if (op == NULL)
          op = local_analyses->front()->op;
      }
#ifdef DEBUG_LEGION
      assert(op != NULL);
#endif
      const PhysicalTraceInfo &local_info = 
        ((local_analyses == NULL) || !trace_info.recording) ? trace_info : 
        local_analyses->front()->trace_info;
#ifdef DEBUG_LEGION
      assert(local_info.recording == trace_info.recording);
#endif
      // Send it on to any children in the broadcast tree first
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(origin, local_space, children);
      std::vector<ApEvent> ready_events;
      ApBarrier trace_barrier;
      ShardID trace_shard = 0;
      for (std::vector<AddressSpaceID>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(fill_view->did);
          rez.serialize(dst_view->did);
          rez.serialize(precondition);
          rez.serialize(predicate_guard);
          fill_expression->pack_expression(rez, *it);
          rez.serialize<bool>(fill_restricted);
          if (fill_restricted)
            op->pack_remote_operation(rez, *it, applied_events);
          rez.serialize(index);
          rez.serialize(op_context_index);
          rez.serialize(fill_mask);
          local_info.pack_trace_info(rez, applied_events);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (local_info.recording)
          {
            if (ready_event.exists() && !trace_barrier.exists())
            {
              trace_barrier =
                ApBarrier(Realm::Barrier::create_barrier(children.size()));
              trace_shard = local_info.record_managed_barrier(trace_barrier,
                                                            children.size());
              ready_events.push_back(trace_barrier);
            }
            rez.serialize(trace_barrier);
            if (trace_barrier.exists())
              rez.serialize(trace_shard);
          }
          else
          {
            ApUserEvent child_ready;
            if (ready_event.exists())
            {
              child_ready = Runtime::create_ap_user_event(&local_info);
              ready_events.push_back(child_ready);
            }
            rez.serialize(child_ready);
          }
          rez.serialize(origin);
        }
        runtime->send_collective_distribute_fill(*it, rez);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      // Now we can perform the fills for our instances
      // The precondition will be the same across all our local instances
      const UniqueID op_id = op->get_unique_op_id();
      ApEvent dst_precondition = dst_view->find_copy_preconditions(
          false/*reading*/, 0/*redop*/, fill_mask, fill_expression,
          op_id, index, applied_events, local_info);
      if (dst_precondition.exists())
      {
        if (dst_precondition.exists())
          precondition =
            Runtime::merge_events(&local_info, precondition, dst_precondition);
        else
          precondition = dst_precondition;
      }
      std::vector<ApEvent> local_events;
      // Do the last wait before we need our analyses for recording 
      // and profiling requests from the mappers
      if (analyses_ready.exists() && !analyses_ready.has_triggered())
        analyses_ready.wait();
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        std::vector<CopySrcDstField> dst_fields;
        layout->compute_copy_offsets(fill_mask, instances[idx],
#ifdef LEGION_SPY
                                     instance_events[idx],
#endif
                                     dst_fields);
        const PhysicalTraceInfo &inst_info = (local_analyses == NULL) ?
          trace_info : local_analyses->at(idx)->trace_info;
        const ApEvent result = fill_expression->issue_fill(op,
                                                 inst_info, dst_fields,
                                                 fill_view->value->value,
                                                 fill_view->value->value_size,
#ifdef LEGION_SPY
                                                 fill_view->fill_op_uid,
                                                 field_space_node->handle,
                                                 tree_id,
#endif
                                                 precondition, predicate_guard);
        if (result.exists())
          local_events.push_back(result);
        if (inst_info.recording)
        {
          const UniqueInst dst_inst(dst_view, instance_points[idx]);
          inst_info.record_fill_inst(result, fill_expression, dst_inst, 
                                     fill_mask, applied_events, (redop > 0));
        }
      }
      if (!local_events.empty())
      {
        ApEvent local_ready = Runtime::merge_events(&local_info, local_events);
        if (local_ready.exists())
        {
          const RtEvent collect_event = local_info.get_collect_event();
          dst_view->add_copy_user(false/*reading*/, 0/*redop*/, local_ready,
              collect_event, fill_mask, fill_expression, op_id, index,
              recorded_events, local_info.recording, runtime->address_space);
          if (ready_event.exists())
            ready_events.push_back(local_ready);
        }
      }
      // Use the trace info for doing the trigger if necessary
      if (!ready_events.empty())
      {
#ifdef DEBUG_LEGION
        assert(ready_event.exists());
#endif
        Runtime::trigger_event(&trace_info, ready_event,
            Runtime::merge_events(&local_info, ready_events));
      }
      else if (ready_event.exists())
        Runtime::trigger_event(&trace_info, ready_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_distribute_fill(Runtime *runtime,
                                     AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID man_did, fill_did, dst_did;
      derez.deserialize(man_did);
      RtEvent man_ready, fill_ready, dst_ready;
      CollectiveManager *manager = static_cast<CollectiveManager*>(
          runtime->find_or_request_instance_manager(man_did, man_ready));
      derez.deserialize(fill_did);
      FillView *fill_view = static_cast<FillView*>(
          runtime->find_or_request_logical_view(fill_did, fill_ready));
      derez.deserialize(dst_did);
      InstanceView *dst_view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(dst_did, dst_ready));
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      IndexSpaceExpression *fill_expression =
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      bool fill_restricted;
      derez.deserialize<bool>(fill_restricted);
      Operation *op = NULL;
      std::set<RtEvent> ready_events;
      if (fill_restricted)
        op = RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      FieldMask fill_mask;
      derez.deserialize(fill_mask);
      std::set<RtEvent> recorded_events, applied_events;
      PhysicalTraceInfo trace_info =
        PhysicalTraceInfo::unpack_trace_info(derez, runtime);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApUserEvent ready;
      if (trace_info.recording)
      {
        ApBarrier bar;
        derez.deserialize(bar);
        if (bar.exists())
        {
          ShardID sid;
          derez.deserialize(sid);
          // Copy-elmination will take care of this for us
          // when the trace is optimized
          ready = Runtime::create_ap_user_event(&trace_info);
          Runtime::phase_barrier_arrive(bar, 1/*count*/, ready);
          trace_info.record_barrier_arrival(bar, ready, 1/*count*/, 
                                            applied_events, sid);
        }
      }
      else
        derez.deserialize(ready);
      AddressSpaceID origin;
      derez.deserialize(origin);
      

      // Make sure all the distributed collectables are ready
      if (man_ready.exists() && !man_ready.has_triggered())
        ready_events.insert(man_ready);
      if (fill_ready.exists() && !fill_ready.has_triggered())
        ready_events.insert(fill_ready);
      if (dst_ready.exists() && !dst_ready.has_triggered())
        ready_events.insert(dst_ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      manager->perform_collective_fill(fill_view, dst_view, precondition,
          predicate_guard, fill_expression, op, index, op_ctx_index,
          fill_mask, trace_info, recorded_events, applied_events, ready,
          origin, fill_restricted);

      if (!recorded_events.empty())
        Runtime::trigger_event(recorded,Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (op != NULL)
        delete op;
    }

    //--------------------------------------------------------------------------
    ApEvent CollectiveManager::perform_collective_point(InstanceView *src_view,
                                const std::vector<CopySrcDstField> &dst_fields,
                                const std::vector<Reservation> &reservations,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expression,
                                Operation *op, const unsigned index,
                                const FieldMask &copy_mask,
                                const FieldMask &dst_mask,
                                const Memory location,
                                const DistributedID dst_view_did,
                                const DomainPoint &dst_point,
                                const DomainPoint &src_point,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!instances.empty());
      assert(src_view->manager == this);
      assert(collective_mapping != NULL);
      assert(collective_mapping->contains(local_space));
#endif
      const UniqueID op_id = op->get_unique_op_id();
      // Compute the source precondition to get that in flight
      const ApEvent src_pre = src_view->find_copy_preconditions(
          true/*reading*/, 0/*redop*/, copy_mask, copy_expression,
          op_id, index, applied_events, trace_info);
      if (src_pre.exists())
      {
        if (precondition.exists())
          precondition =
            Runtime::merge_events(&trace_info, precondition, src_pre);
        else
          precondition = src_pre;
      }
      // Figure out which instance we're going to use for the copy
      unsigned instance_index = 0;
      if (src_point.exists())
      {
#ifdef DEBUG_LEGION
        instance_index = UINT_MAX;
#endif
        for (unsigned idx = 0; idx < instance_points.size(); idx++)
        {
          if (instance_points[idx] != src_point)
            continue;
          instance_index = idx;
          break;
        }
#ifdef DEBUG_LEGION
        assert(instance_index != UINT_MAX);
#endif
      }
      else if (instances.size() > 1)
      {
        // Handle the special, but common case where the source has
        // the same point as the destination, which is usually semantically
        // meaningful to the application
        bool found = false;
        if (dst_point.get_dim() > 0)
        {
          for (unsigned idx = 0; idx < instance_points.size(); idx++)
          {
            if (dst_point != instance_points[idx])
              continue;
            instance_index = idx;
            found = true;
            break;
          }
        }
        if (!found)
        {
          int best_bandwidth = -1;
          const Machine &machine = runtime->machine;
          Machine::AffinityDetails details;
          if (machine.has_affinity(location,
                instances[0].get_location(), &details))
            best_bandwidth = details.bandwidth;
          for (unsigned idx = 1; idx < instances.size(); idx++)
          {
            if (machine.has_affinity(location,
                  instances[idx].get_location(), &details))
            {
              if ((best_bandwidth < 0) || 
                  (int(details.bandwidth) > best_bandwidth))
              {
                best_bandwidth = details.bandwidth;
                instance_index = idx;
              }
            }
          }
        }
      }
      // Compute the src_fields
      std::vector<CopySrcDstField> src_fields;
      layout->compute_copy_offsets(copy_mask, instances[instance_index],
#ifdef LEGION_SPY
                                   instance_events[instance_index],
#endif
                                   src_fields);
      // Issue the copy
      const ApEvent copy_post = copy_expression->issue_copy(op,
            trace_info, dst_fields, src_fields, reservations,
#ifdef LEGION_SPY
            tree_id, tree_id,
#endif
            precondition, predicate_guard);
      // Record the user
      if (copy_post.exists())
      {
        const RtEvent collect_event = trace_info.get_collect_event();
        src_view->add_copy_user(true/*reading*/, 0/*redop*/, copy_post,
            collect_event, copy_mask, copy_expression, op_id, index,
            recorded_events, trace_info.recording, runtime->address_space);
      }
      if (trace_info.recording)
      {
        const UniqueInst src_inst(src_view, instance_points[instance_index]);
        const UniqueInst dst_inst(dst_view_did, dst_point);
        trace_info.record_copy_insts(copy_post, copy_expression,
            src_inst, dst_inst, copy_mask, dst_mask, redop, applied_events);
      }
      return copy_post;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_distribute_point(Runtime *runtime,
                                     AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID man_did, src_did;
      derez.deserialize(man_did);
      RtEvent man_ready, src_ready;
      CollectiveManager *manager = static_cast<CollectiveManager*>(
          runtime->find_or_request_instance_manager(man_did, man_ready));
      derez.deserialize(src_did);
      InstanceView *src_view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(src_did, src_ready));
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<CopySrcDstField> dst_fields(num_fields);
      std::set<RtEvent> recorded_events, ready_events, applied_events;
      unpack_fields(dst_fields, derez, ready_events, manager,man_ready,runtime);
      size_t num_reservations;
      derez.deserialize(num_reservations);
      std::vector<Reservation> reservations(num_reservations);
      for (unsigned idx = 0; idx < num_reservations; idx++)
        derez.deserialize(reservations[idx]);
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      IndexSpaceExpression *copy_expression =
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      Operation *op =
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      FieldMask copy_mask, dst_mask;
      derez.deserialize(copy_mask);
      derez.deserialize(dst_mask);
      Memory location;
      derez.deserialize(location);
      DistributedID dst_view_did;
      derez.deserialize(dst_view_did);
      DomainPoint dst_point, src_point;
      derez.deserialize(dst_point);
      derez.deserialize(src_point);
      PhysicalTraceInfo trace_info =
        PhysicalTraceInfo::unpack_trace_info(derez, runtime);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApUserEvent ready;
      derez.deserialize(ready);

      if (man_ready.exists() && !man_ready.has_triggered())
        ready_events.insert(man_ready);
      if (src_ready.exists() && !src_ready.has_triggered())
        ready_events.insert(src_ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      const ApEvent result = manager->perform_collective_point(src_view,
          dst_fields, reservations, precondition, predicate_guard,
          copy_expression, op, index, copy_mask, dst_mask, location, 
          dst_view_did, dst_point, src_point, trace_info,
          recorded_events, applied_events);

      Runtime::trigger_event(&trace_info, ready, result);
      if (!recorded_events.empty())
        Runtime::trigger_event(recorded,Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      delete op;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::perform_collective_reduction(InstanceView *src_view,
                                const std::vector<CopySrcDstField> &dst_fields,
                                const std::vector<Reservation> &reservations,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expression,
                                Operation *op, const unsigned index,
                                const FieldMask &copy_mask,
                                const FieldMask &dst_mask,
                                const DomainPoint &src_point,
                                const UniqueInst &dst_inst,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                ApUserEvent result, AddressSpaceID origin)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(redop > 0);
      assert(op != NULL);
      assert(result.exists());
      assert(!instances.empty());
      assert(src_view->manager == this);
      assert(collective_mapping != NULL);
      assert(collective_mapping->contains(local_space));
#endif
      unsigned target_index = 0;
      if (src_point.exists())
      {
#ifdef DEBUG_LEGION
        target_index = UINT_MAX;
#endif
        for (unsigned idx = 0; idx < instance_points.size(); idx++)
        {
          if (instance_points[idx] != src_point)
            continue;
          target_index = idx;
          break;
        }
#ifdef DEBUG_LEGION
        assert(target_index != UINT_MAX);
#endif
      }
      // Get the dst_fields and reservations for performing the local reductions
      std::vector<CopySrcDstField> local_fields;
      layout->compute_copy_offsets(copy_mask, instances[target_index],
#ifdef LEGION_SPY
                                   instance_events[target_index],
#endif
                                   local_fields);

      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(origin, local_space, children);
      // Get the precondition for performing reductions to one of our instances
      ApEvent reduce_pre; 
      std::vector<Reservation> local_reservations;
      const UniqueID op_id = op->get_unique_op_id();
      if (!children.empty() || (instances.size() > 1))
      {
        // Compute the precondition for performing any reductions
        reduce_pre = src_view->find_copy_preconditions(false/*reading*/, redop,
          copy_mask, copy_expression, op_id, index, applied_events, trace_info);
        // If we're going to be doing reductions we need the reservations
        src_view->find_field_reservations(copy_mask, 
            instance_points[target_index], local_reservations);
        for (unsigned idx = 0; idx < local_fields.size(); idx++)
          local_fields[idx].set_redop(redop, true/*fold*/, true/*exclusive*/);
      }
      std::vector<ApEvent> reduce_events;
      // If we have any children, send them messages to reduce to our instance
      ApBarrier trace_barrier;
      ShardID trace_shard = 0;
      const UniqueInst local_inst(src_view, instance_points[target_index]);
      for (std::vector<AddressSpaceID>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(src_view->did);
          pack_fields(rez, local_fields);
          rez.serialize<size_t>(local_reservations.size());
          for (unsigned idx = 0; idx < local_reservations.size(); idx++)
            rez.serialize(local_reservations[idx]);
          rez.serialize(reduce_pre);
          rez.serialize(predicate_guard);
          copy_expression->pack_expression(rez, *it);
          op->pack_remote_operation(rez, *it, applied_events);
          rez.serialize(index);
          rez.serialize(copy_mask);
          rez.serialize(dst_mask);
          const DomainPoint nopoint;
          rez.serialize(nopoint);
          local_inst.serialize(rez);
          trace_info.pack_trace_info(rez, applied_events);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (trace_info.recording)
          {
            if (!trace_barrier.exists())
            {
              trace_barrier = 
                ApBarrier(Realm::Barrier::create_barrier(children.size()));
              trace_shard = trace_info.record_managed_barrier(trace_barrier,
                                                              children.size());
              reduce_events.push_back(trace_barrier);
            }
            rez.serialize(trace_barrier);
            if (trace_barrier.exists())
              rez.serialize(trace_shard);
          }
          else
          {
            const ApUserEvent reduced =
              Runtime::create_ap_user_event(&trace_info);
            rez.serialize(reduced);
            reduce_events.push_back(reduced);
          }
          rez.serialize(origin);
        }
        runtime->send_collective_distribute_reduction(*it, rez);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      // Compute the reading precondition for our instances
      ApEvent read_pre = src_view->find_copy_preconditions(
          true/*reading*/, 0/*redop*/, copy_mask, copy_expression,
          op_id, index, applied_events, trace_info);
      // Make sure we don't apply any reductions to instance[0]
      // unless it is safe to to do so
      if (reduce_pre.exists())
      {
        if (read_pre.exists())
          read_pre = Runtime::merge_events(&trace_info, read_pre, reduce_pre);
        else
          read_pre = reduce_pre;
      }
      // Perform our local reductions
      // TODO: We could build a tree reduction here inside the
      // local node as well, but that seems unnecessary for most
      // cases so we're just going to reduce everything to the 
      // target for now
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        if (idx == target_index)
          continue;
        std::vector<CopySrcDstField> src_fields;
        layout->compute_copy_offsets(copy_mask, instances[idx],
#ifdef LEGION_SPY
                                     instance_events[idx],
#endif
                                     src_fields);
        ApEvent local_reduce = copy_expression->issue_copy(
            op, trace_info, local_fields, src_fields, local_reservations,
#ifdef LEGION_SPY
            tree_id, tree_id,
#endif
            read_pre, predicate_guard); 
        if (local_reduce.exists())
          reduce_events.push_back(local_reduce);
        if (trace_info.recording)
        {
          const UniqueInst src_inst(src_view, instance_points[idx]);
          trace_info.record_copy_insts(local_reduce, copy_expression,
             src_inst, local_inst, copy_mask, copy_mask, redop, applied_events);
        }
      }
      // Peform the reduction back to the destination
      // No need to swap the local fields back to being non-reduction
      // since they're going in the src location here so realm should
      // ignore the reduction parts of them
      // Incorporate any reductions along with the read precondition
      // and the precondition from the destination to compute any 
      // preconditions for the reduction to our caller
      if (read_pre.exists())
        reduce_events.push_back(read_pre);
      if (!reduce_events.empty())
      {
        if (precondition.exists())
          reduce_events.push_back(precondition);
        precondition = Runtime::merge_events(&trace_info, reduce_events);
      }
      // Set the redops back to 0
      for (unsigned idx = 0; idx < local_fields.size(); idx++)
        local_fields[idx].set_redop(0, false/*fold*/);
      // Perform the reduction to the destination
      const ApEvent reduce_post = copy_expression->issue_copy(
          op, trace_info, dst_fields, local_fields, reservations,
#ifdef LEGION_SPY
          tree_id, tree_id,
#endif
          precondition, predicate_guard);
      // Trigger the output
      Runtime::trigger_event(&trace_info, result, reduce_post);
      // Save the result, note that this reading of this final reduction
      // always dominates any incoming reductions so we don't need to 
      // record them separately
      if (reduce_post.exists())
      {
        const RtEvent collect_event = trace_info.get_collect_event();
        src_view->add_copy_user(true/*reading*/, 0/*redop*/, reduce_post,
            collect_event, copy_mask, copy_expression, op_id, index,
            recorded_events, trace_info.recording, runtime->address_space);
      }
      if (trace_info.recording)
        trace_info.record_copy_insts(reduce_post, copy_expression,
            local_inst, dst_inst, copy_mask, dst_mask, redop, applied_events);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_distribute_reduction(
                   Runtime *runtime, AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID man_did, src_did;
      derez.deserialize(man_did);
      RtEvent man_ready, src_ready;
      CollectiveManager *manager = static_cast<CollectiveManager*>(
          runtime->find_or_request_instance_manager(man_did, man_ready));
      derez.deserialize(src_did);
      InstanceView *src_view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(src_did, src_ready));
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<CopySrcDstField> dst_fields(num_fields);
      std::set<RtEvent> recorded_events, ready_events, applied_events;
      unpack_fields(dst_fields, derez, ready_events, manager,man_ready,runtime);
      size_t num_reservations;
      derez.deserialize(num_reservations);
      std::vector<Reservation> reservations(num_reservations);
      for (unsigned idx = 0; idx < num_reservations; idx++)
        derez.deserialize(reservations[idx]);
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      IndexSpaceExpression *copy_expression =
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      Operation *op =
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      FieldMask copy_mask, dst_mask;
      derez.deserialize(copy_mask);
      derez.deserialize(dst_mask);
      DomainPoint src_point;
      derez.deserialize(src_point);
      UniqueInst dst_inst;
      dst_inst.deserialize(derez);
      PhysicalTraceInfo trace_info =
        PhysicalTraceInfo::unpack_trace_info(derez, runtime);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApUserEvent ready;
      if (trace_info.recording)
      {
        ApBarrier bar;
        derez.deserialize(bar);
        ShardID sid;
        derez.deserialize(sid);
        // Copy-elmination will take care of this for us
        // when the trace is optimized
        ready = Runtime::create_ap_user_event(&trace_info);
        Runtime::phase_barrier_arrive(bar, 1/*count*/, ready);
        trace_info.record_barrier_arrival(bar, ready, 1/*count*/, 
                                          applied_events, sid);
      }
      else
        derez.deserialize(ready);
      AddressSpaceID origin;
      derez.deserialize(origin);

      if (man_ready.exists() && !man_ready.has_triggered())
        ready_events.insert(man_ready);
      if (src_ready.exists() && !src_ready.has_triggered())
        ready_events.insert(src_ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      manager->perform_collective_reduction(src_view, dst_fields, reservations,
          precondition, predicate_guard, copy_expression, op, index, copy_mask,
          dst_mask, src_point, dst_inst, trace_info, recorded_events, 
          applied_events, ready, origin);

      if (!recorded_events.empty())
        Runtime::trigger_event(recorded,Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      delete op;
    }

    //--------------------------------------------------------------------------
    ApEvent CollectiveManager::copy_from(InstanceView *src_view,
                                         InstanceView *dst_view,
                                         PhysicalManager *source_manager,
                                         ApEvent precondition,
                                         PredEvent predicate_guard, 
                                         ReductionOpID reduction_op_id,
                                         IndexSpaceExpression *copy_expression,
                                         Operation *op, const unsigned index,
                                         const FieldMask &copy_mask,
                                         const DomainPoint &src_point,
                                         const PhysicalTraceInfo &trace_info,
                                         std::set<RtEvent> &recorded_events,
                                         std::set<RtEvent> &applied_events,
                                         CopyAcrossHelper *across_helper,
                                         const bool manage_dst_events,
                                         const bool copy_restricted,
                                         const bool need_valid_return)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Should never have a copy-across with a collective manager as the target
      assert(manage_dst_events);
      assert(across_helper == NULL);
      assert(dst_view->manager == this);
      assert(collective_mapping != NULL);
      assert(reduction_op_id == source_manager->redop);
#endif
      // Several cases here:
      // 1. The source is a normal individual manager - in this case we'll issue
      //    the copy/reduction from the source to an instance on the closest
      //    node and then build the broadcast tree from there 
      // 2. The source is another normal collective manager - we'll do a 
      //    broadcast out to all the nodes and have each of them pick a 
      //    source instance to copy from and then do the copy
      // 3. The source is a reduction collective instance with the same 
      //      collective mapping as the destination - broadcast control
      //    out to all the nodes and then perform the all-reduce between the
      //    instances of the source, then do the reduction the same as the 
      //    case for copies with a normal collective manager
      // 4. The source is a reduction manager that is either an individual
      //      instance or a collective instance with a different mapping
      //      than the destination - Build a reduction tree down to a
      //    single instance if necessary and then broadcast out the
      //    reduction data to all the other instances
      ApUserEvent all_done;
      if (need_valid_return)
        all_done = Runtime::create_ap_user_event(&trace_info);
      if (!source_manager->is_collective_manager())
      {
        // Case 1: the source is an individual manager
        // Copy to one of our instances and then broadcast it
        IndividualManager *source = source_manager->as_individual_manager();
        const UniqueID op_id = op->get_unique_op_id();
        // Get the precondition as well
        const ApEvent src_pre = src_view->find_copy_preconditions(
            true/*reading*/, 0/*redop*/, copy_mask, copy_expression,
            op_id, index, applied_events, trace_info);
        if (src_pre.exists())
        {
          if (precondition.exists())
            precondition =
              Runtime::merge_events(&trace_info, precondition, src_pre);
          else
            precondition = src_pre;
        }
        std::vector<CopySrcDstField> src_fields;
        source->compute_copy_offsets(copy_mask, src_fields);
        // We have to follow the tree for other kinds of operations here
        const AddressSpaceID origin = select_origin_space(); 
        ApUserEvent copy_done = Runtime::create_ap_user_event(&trace_info);
        // Record the copy done event on the source view
        src_view->add_copy_user(true/*reading*/, 0/*redop*/, copy_done,
            trace_info.get_collect_event(), copy_mask, copy_expression,
            op_id, index, recorded_events, trace_info.recording,
            runtime->address_space);
        ApBarrier all_bar;
        ShardID owner_shard = 0;
        if (trace_info.recording && (all_done.exists() || (source->redop > 0)))
        {
          const size_t arrivals = collective_mapping->size();
          all_bar = ApBarrier(Realm::Barrier::create_barrier(arrivals));
          owner_shard = trace_info.record_managed_barrier(all_bar, arrivals);
          // Tracing copy-optimization will eliminate this when
          // the trace gets optimized
          if (all_done.exists())
            Runtime::trigger_event(&trace_info, all_done, all_bar);
          if (source->redop > 0)
          {
            Runtime::trigger_event(&trace_info, copy_done, all_bar);
#ifdef DEBUG_LEGION
            copy_done = ApUserEvent::NO_AP_USER_EVENT;
#endif
          }
        }
        const DomainPoint no_point;
        const UniqueInst src_inst(src_view, no_point);
        if (origin != local_space)
        {
          const RtUserEvent recorded = Runtime::create_rt_user_event();
          const RtUserEvent applied = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(this->did);
            if (reduction_op_id > 0)
              rez.serialize(source->did);
            rez.serialize(dst_view->did);
            source->pack_fields(rez, src_fields);
            src_inst.serialize(rez);
            rez.serialize(precondition);
            rez.serialize(predicate_guard);
            copy_expression->pack_expression(rez, origin);
            rez.serialize<bool>(copy_restricted);
            if (copy_restricted)
              op->pack_remote_operation(rez, origin, applied_events);
            rez.serialize(index);
            rez.serialize(op->get_ctx_index());
            rez.serialize(copy_mask);
            trace_info.pack_trace_info(rez, applied_events);
            rez.serialize(recorded);
            rez.serialize(applied);
            if (trace_info.recording)
            {
              // If this is a reducecast case, then the barrier is for
              // all of the different reductions
              if (source->redop == 0)
              {
                ApBarrier copy_bar(Realm::Barrier::create_barrier(1/*count*/));
                ShardID sid = trace_info.record_managed_barrier(copy_bar, 1);
                Runtime::trigger_event(&trace_info, copy_done, copy_bar);
                rez.serialize(copy_bar);
                rez.serialize(sid);
              }
              rez.serialize(all_bar);
              if (all_bar.exists())
                rez.serialize(owner_shard);
            }
            else
            {
              rez.serialize(copy_done);
              if (source->redop == 0)
                rez.serialize(all_done);
            }
            rez.serialize(origin);
          }
          if (reduction_op_id > 0)
            runtime->send_collective_distribute_reducecast(origin, rez);
          else
            runtime->send_collective_distribute_broadcast(origin, rez);
          recorded_events.insert(recorded);
          applied_events.insert(applied);
        }
        else
        {
          if (reduction_op_id > 0)
            perform_collective_reducecast(source, dst_view, src_fields,
                precondition, predicate_guard, copy_expression, op, index, 
                op->get_ctx_index(), copy_mask, src_inst, trace_info,
                recorded_events, applied_events, copy_done, all_bar,
                owner_shard, origin, copy_restricted);
          else
            perform_collective_broadcast(dst_view, src_fields, precondition,
                predicate_guard, copy_expression, op, index, 
                op->get_ctx_index(), copy_mask, src_inst, trace_info,
                recorded_events, applied_events, copy_done, all_done, all_bar,
                owner_shard, origin, copy_restricted); 
        }
      }
      else
      {
        CollectiveManager *collective = source_manager->as_collective_manager();
        const AddressSpaceID origin = select_origin_space();
        // If the source is a reduction collective instance then we need
        // to see if we can go down the point-wise route based on performing
        // an all-reduce, or whether we have to do a tree reduction followed
        // by a tree broadcast. To do the all-reduce path we need all the
        // collective mappings for both collective instances to be the same
        uint64_t allreduce_tag = 0;
        if (collective->is_reduction_manager())
        {
          // Case 3: this is conceptually an all-reduce
          // We'll handle two separate cases here depending on whether
          // the two collective instances have matching collective mappings
          if ((collective_mapping != collective->collective_mapping) &&
              (*collective_mapping != *(collective->collective_mapping)))
          {
            // The two collective mappings do not align, which should
            // be fairly uncommon, but we'll handle it anyway
            // In this case we'll do a reduction down to a single
            // instance in the source collective manager and then 
            // broadcast back out to all the destination instances
            // For correctness, the reduce cast must start whereever
            // a comparable broadcast or fill would have started
            // on the destination collective instance
            perform_collective_hourglass(collective, src_view, dst_view,
                precondition, predicate_guard, copy_expression, op, index, 
                copy_mask, src_point, trace_info, recorded_events, 
                applied_events, all_done, origin, copy_restricted);
            return all_done;
          }
          // Otherwise we can fall through and do the allreduce as part
          // of the pointwise copy, get a tag through for unique identification
          allreduce_tag = 
            unique_allreduce_tag.fetch_add(collective_mapping->size());
        }
        ApBarrier all_bar;
        ShardID owner_shard;
        if (all_done.exists() && trace_info.recording)
        {
          const size_t arrivals = collective_mapping->size();
          all_bar = ApBarrier(Realm::Barrier::create_barrier(arrivals));
          owner_shard = trace_info.record_managed_barrier(all_bar, arrivals);
          // Tracing copy-optimization will eliminate this when
          // the trace gets optimized
          Runtime::trigger_event(&trace_info, all_done, all_bar);
        }
        const DomainPoint origin_point = op->get_collective_instance_point();
        // Case 2 and 3 (all-reduce): Broadcast out the point-wise command
        if (origin != local_space)
        {
          const RtUserEvent recorded = Runtime::create_rt_user_event();
          const RtUserEvent applied = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(this->did);
            rez.serialize(collective->did);
            rez.serialize(src_view->did);
            rez.serialize(dst_view->did);
            rez.serialize(precondition);
            rez.serialize(predicate_guard);
            copy_expression->pack_expression(rez, origin);
            rez.serialize<bool>(copy_restricted);
            if (copy_restricted)
              op->pack_remote_operation(rez, origin, applied_events);
            rez.serialize(index);
            rez.serialize(op->get_ctx_index());
            rez.serialize(copy_mask);
            rez.serialize(origin_point);
            rez.serialize(src_point);
            trace_info.pack_trace_info(rez, applied_events);
            rez.serialize(recorded);
            rez.serialize(applied);
            if (trace_info.recording)
            {
              rez.serialize(all_bar);
              if (all_bar.exists())
                rez.serialize(owner_shard);
            }
            else
              rez.serialize(all_done);
            rez.serialize(origin);
            rez.serialize(allreduce_tag);
          }
          runtime->send_collective_distribute_pointwise(origin, rez);
          recorded_events.insert(recorded);
          applied_events.insert(applied);
        }
        else
          perform_collective_pointwise(collective, src_view, dst_view,
              precondition, predicate_guard, copy_expression, op, index,
              op->get_ctx_index(), copy_mask, origin_point, src_point,
              trace_info, recorded_events, applied_events, all_done, all_bar,
              owner_shard, origin, allreduce_tag, copy_restricted);
      }
      return all_done;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::perform_collective_pointwise(
                                          CollectiveManager *source,
                                          InstanceView *src_view,
                                          InstanceView *dst_view,
                                          ApEvent precondition,
                                          PredEvent predicate_guard,
                                          IndexSpaceExpression *copy_expression,
                                          Operation *op, const unsigned index,
                                          const size_t op_ctx_index,
                                          const FieldMask &copy_mask,
                                          const DomainPoint &origin_point,
                                          const DomainPoint &origin_src_point,
                                          const PhysicalTraceInfo &trace_info,
                                          std::set<RtEvent> &recorded_events,
                                          std::set<RtEvent> &applied_events,
                                          ApUserEvent all_done,
                                          ApBarrier all_bar,
                                          ShardID owner_shard,
                                          AddressSpaceID origin,
                                          const uint64_t allreduce_tag,
                                          const bool copy_restricted)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!instances.empty());
      assert(dst_view->manager == this);
      assert(src_view->manager == source);
      assert(collective_mapping->contains(local_space));
      assert((op != NULL) || !copy_restricted);
#endif
      RtEvent analyses_ready;
      const std::vector<CollectiveCopyFillAnalysis*> *local_analyses = NULL;
      if (!copy_restricted)
      {
        // If this is not a copy-out to a restricted collective instance 
        // then we should be able to find our local analyses to use for 
        // performing operations
        analyses_ready = find_collective_analyses(dst_view->did,
                            op_ctx_index, index, local_analyses);
#ifdef DEBUG_LEGION
        assert(local_analyses != NULL);
#endif
        // If we're recording then we need to wait now to get a valid
        // trace info for capturing the trace for events we send to 
        // remote nodes, otherwise we just need to wait before doing
        // the fill calls
        if ((trace_info.recording || (op == NULL)) && 
            analyses_ready.exists() && !analyses_ready.has_triggered())
          analyses_ready.wait();
        if (op == NULL)
          op = local_analyses->front()->op;
      }
#ifdef DEBUG_LEGION
      assert(op != NULL);
#endif
      const PhysicalTraceInfo &local_info = 
        ((local_analyses == NULL) || !trace_info.recording) ? trace_info : 
        local_analyses->front()->trace_info;
      // First distribute this off to all the child nodes
      std::vector<ApEvent> done_events;
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(origin, local_space, children);
      for (std::vector<AddressSpaceID>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(this->did);
          rez.serialize(source->did);
          rez.serialize(src_view->did);
          rez.serialize(dst_view->did);
          rez.serialize(precondition);
          rez.serialize(predicate_guard);
          copy_expression->pack_expression(rez, *it);
          rez.serialize<bool>(copy_restricted);
          if (copy_restricted)
            op->pack_remote_operation(rez, *it, applied_events);
          rez.serialize(index);
          rez.serialize(op_ctx_index);
          rez.serialize(copy_mask);
          rez.serialize(origin_point);
          rez.serialize(origin_src_point);
          trace_info.pack_trace_info(rez, applied_events);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (local_info.recording)
          {
            rez.serialize(all_bar);
            if (all_bar.exists())
              rez.serialize(owner_shard);
          }
          else
          {
            ApUserEvent done; 
            if (all_done.exists())
            {
              done = Runtime::create_ap_user_event(&local_info);
              done_events.push_back(done);
            }
            rez.serialize(done);
          }
          rez.serialize(origin);
          rez.serialize(allreduce_tag);
        }
        runtime->send_collective_distribute_pointwise(*it, rez);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      const UniqueID op_id = op->get_unique_op_id();
      // If the source is a reduction manager, this is where we need
      // to perform the all-reduce before issuing the pointwise copies
      if (source->is_reduction_manager())
      {
#ifdef DEBUG_LEGION
        // Better have the same collective mappings if we're doing all-reduce
        assert((collective_mapping == source->collective_mapping) ||
            ((*collective_mapping) == (*(source->collective_mapping))));
        assert(src_view->is_reduction_view());
#endif
        ReductionView *red_view = src_view->as_reduction_view();
        // Wait for the analyses to be available if they aren't already
        if (analyses_ready.exists() && !analyses_ready.has_triggered())
          analyses_ready.wait();
        source->perform_collective_allreduce(red_view, precondition,
            predicate_guard, copy_expression, op, index, copy_mask, local_info,
            local_analyses, recorded_events, applied_events, allreduce_tag);
      }
      // Find the precondition for all our local copies
      const ApEvent dst_pre = dst_view->find_copy_preconditions(
          false/*reading*/, source->redop, copy_mask, copy_expression,
          op_id, index, applied_events, local_info);
      if (dst_pre.exists())
      {
        if (precondition.exists())
          precondition =
            Runtime::merge_events(&local_info, precondition, dst_pre);
        else
          precondition = dst_pre;
      }
      std::vector<ApEvent> local_events;
      // Wait for the analyses to be available if they aren't already
      if (analyses_ready.exists() && !analyses_ready.has_triggered())
        analyses_ready.wait();
      // Now we can do our local copies
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        // Get our dst_fields
        std::vector<CopySrcDstField> dst_fields;
        layout->compute_copy_offsets(copy_mask, instances[idx],
#ifdef LEGION_SPY
                                     instance_events[idx],
#endif
                                     dst_fields); 
        std::vector<Reservation> reservations;
        if (source->redop > 0)
        {
          dst_view->find_field_reservations(copy_mask, instance_points[idx],
                                            reservations);
          for (unsigned idx = 0; idx < dst_fields.size(); idx++)
            dst_fields[idx].set_redop(source->redop, false/*fold*/,
                                      true/*exclusive*/);
        }
        const Memory location = instances[idx].get_location();
        const PhysicalTraceInfo &inst_info = (local_analyses == NULL) ?
          trace_info : local_analyses->at(idx)->trace_info;
        // Now we need to pick the source point for this copy if it hasn't
        // already been picked by the mapper
        DomainPoint src_point;
        if (!copy_restricted)
        {
#ifdef DEBUG_LEGION
          assert(local_analyses != NULL);
#endif
          if (instance_points[idx] != origin_point)
          {
            // invoke the mapper to pick the source point in this case
            CollectiveCopyFillAnalysis *analysis = local_analyses->at(idx);
            std::vector<InstanceView*> src_views(1, src_view);
            std::vector<unsigned> ranking;
            std::map<unsigned,DomainPoint> collective_keys;
            analysis->op->select_sources(analysis->index, dst_view,
                                      src_views, ranking, collective_keys);
            std::map<unsigned,DomainPoint>::const_iterator finder = 
              collective_keys.find(0);
            if (finder != collective_keys.end())
              src_point = finder->second;
          }
          else // mapper already had a chance to pick the source point
            src_point = origin_src_point;
        }
        // TODO: how to let the mapper pick in copy-out cases
        // If the mapper didn't pick a source point then we can
        const AddressSpaceID src = src_point.exists() ?
          source->get_instance(src_point).address_space() :
          source->select_source_space(local_space);
        if (src != local_space)
        {
          const RtUserEvent recorded = Runtime::create_rt_user_event();
          const RtUserEvent applied = Runtime::create_rt_user_event();
          ApUserEvent done = Runtime::create_ap_user_event(&inst_info);
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(source->did);
            rez.serialize(src_view->did);
            pack_fields(rez, dst_fields);
            rez.serialize<size_t>(reservations.size());
            for (unsigned idx2 = 0; idx2 < reservations.size(); idx2++)
              rez.serialize(reservations[idx2]);
            rez.serialize(precondition);
            rez.serialize(predicate_guard);
            copy_expression->pack_expression(rez, src);
            op->pack_remote_operation(rez, src, applied_events);
            rez.serialize(index);
            rez.serialize(copy_mask);
            rez.serialize(copy_mask); // again for dst mask
            rez.serialize(location);
            rez.serialize(dst_view->did);
            rez.serialize(instance_points[idx]);
            rez.serialize(src_point);
            inst_info.pack_trace_info(rez, applied_events);
            rez.serialize(recorded);
            rez.serialize(applied);
            rez.serialize(done);
          }
          runtime->send_collective_distribute_point(src, rez);
          recorded_events.insert(recorded);
          applied_events.insert(applied);
          local_events.push_back(done);
        }
        else
        {
          const ApEvent done = source->perform_collective_point(src_view,
              dst_fields, reservations, precondition, predicate_guard,
              copy_expression, op, index, copy_mask, copy_mask, location,
              dst_view->did, instance_points[idx], src_point, inst_info, 
              recorded_events, applied_events);
          if (done.exists())
            local_events.push_back(done);
        }
      }
      // Record our destination event
      if (!local_events.empty())
      {
        ApEvent local_done = Runtime::merge_events(&local_info, local_events);
        if (local_done.exists())
        {
          const RtEvent collect_event = local_info.get_collect_event();
          dst_view->add_copy_user(false/*reading*/, source->redop,
              local_done, collect_event, copy_mask, copy_expression,
              op_id, index, recorded_events, local_info.recording,
              runtime->address_space);
          done_events.push_back(local_done);
        }
      }
      if (all_bar.exists())
      {
        ApEvent arrival;
        if (!done_events.empty())
          arrival = Runtime::merge_events(&local_info, done_events);
        Runtime::phase_barrier_arrive(all_bar, 1/*count*/, arrival);
        local_info.record_barrier_arrival(all_bar, arrival, 1/*count*/,
                                          applied_events, owner_shard);
      }
      else if (all_done.exists())
      {
        if (!done_events.empty())
          Runtime::trigger_event(&local_info, all_done,
              Runtime::merge_events(&local_info, done_events));
        else
          Runtime::trigger_event(&local_info, all_done);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_distribute_pointwise(
                   Runtime *runtime, AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez); 
      DistributedID did;
      derez.deserialize(did);
      RtEvent dst_man_ready, src_man_ready, dst_view_ready, src_view_ready;
      CollectiveManager *target = static_cast<CollectiveManager*>(
          runtime->find_or_request_instance_manager(did, dst_man_ready));
      derez.deserialize(did);
      CollectiveManager *manager = static_cast<CollectiveManager*>(
          runtime->find_or_request_instance_manager(did, src_man_ready));
      derez.deserialize(did);
      InstanceView *src_view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(did, src_view_ready));
      derez.deserialize(did);
      InstanceView *dst_view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(did, dst_view_ready));
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      IndexSpaceExpression *copy_expression =
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      bool copy_restricted;
      derez.deserialize(copy_restricted);
      Operation *op = NULL;
      std::set<RtEvent> ready_events;
      if (copy_restricted)
        op = RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      DomainPoint origin_point, origin_src_point;
      derez.deserialize(origin_point);
      derez.deserialize(origin_src_point);
      std::set<RtEvent> recorded_events, applied_events;
      PhysicalTraceInfo trace_info =
        PhysicalTraceInfo::unpack_trace_info(derez, runtime);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApBarrier all_bar;
      ShardID owner_shard = 0;
      ApUserEvent all_done;
      if (trace_info.recording)
      {
        derez.deserialize(all_bar);
        if (all_bar.exists())
          derez.deserialize(owner_shard);
      }
      else
        derez.deserialize(all_done);
      AddressSpaceID origin;
      derez.deserialize(origin);
      uint64_t allreduce_tag;
      derez.deserialize(allreduce_tag); 

      if (dst_man_ready.exists() && !dst_man_ready.has_triggered())
        ready_events.insert(dst_man_ready);
      if (src_man_ready.exists() && !src_man_ready.has_triggered())
        ready_events.insert(src_man_ready);
      if (src_view_ready.exists() && !src_view_ready.has_triggered())
        ready_events.insert(src_view_ready);
      if (dst_view_ready.exists() && !dst_view_ready.has_triggered())
        ready_events.insert(dst_view_ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      target->perform_collective_pointwise(manager, src_view, dst_view,
          precondition, predicate_guard, copy_expression, op, index,
          op_ctx_index, copy_mask, origin_point, origin_src_point, 
          trace_info, recorded_events, applied_events, all_done, all_bar,
          owner_shard, origin, allreduce_tag, copy_restricted);

      if (!recorded_events.empty())
        Runtime::trigger_event(recorded,Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (op != NULL)
        delete op;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::perform_collective_allreduce(ReductionView *view,
                                          ApEvent precondition,
                                          PredEvent predicate_guard,
                                          IndexSpaceExpression *copy_expression,
                                          Operation *op, const unsigned index,
                                          const FieldMask &copy_mask,
                                          const PhysicalTraceInfo &trace_info,
                 const std::vector<CollectiveCopyFillAnalysis*> *local_analyses,
                                          std::set<RtEvent> &recorded_events,
                                          std::set<RtEvent> &applied_events,
                                          const uint64_t allreduce_tag)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(redop > 0);
      assert(op != NULL);
      assert(collective_mapping != NULL);
      assert(collective_mapping->contains(local_space));
#endif
      // We're guaranteed to get one call to this function for each space
      // in the collective mapping from perform_collective_pointwise, so
      // we've already distributed control
      // Our job in this function is to build a butterfly all-reduce network
      // for exchanging the reduction data so each reduction instance in this
      // collective instance contains all the same data
      // There is a major complicating factor here because we can't do a 
      // natural in-place all-reduce across our instances since the finish
      // event for Realm copies only says when the whole copy is done and not
      // when the copy has finished reading out from the source instance.
      // Furthermore, we can't control when the reductions into the destination
      // instances start happening as they precondition just governs the start
      // of the whole copy. Therefore, we need to fake an in-place all-reduce.
      // We fake things in one of two ways:
      // Case 1: If we know that each node has at least two instances, then 
      //         we can use one instance as the source for outgoing reduction
      //         copies and the other as the destination for incoming
      //         reduction copies and ping pong between them.
      // Case 2: If we don't have at least two instances on each node then
      //         we will pair up nodes and have them do the same trick as in
      //         case 1 but using the two instances on adjacent nodes as the
      //         sources and destinations.
      // We handle unusual numbers of nodes that are not a power of the 
      // collective radix in the normal way by picking a number of participants
      // that is the largest power of the radix still less than or equal to
      // the number of nodes and using an extra stage to fold-in the 
      // non-participants values before doing the butterfly.

      // First reduce all our local instances down to the first local instance
      const UniqueID op_id = op->get_unique_op_id(); 
      const ApEvent pre = view->find_copy_preconditions(false/*reading*/,
          0/*redop*/, copy_mask, copy_expression, op_id, index, 
          applied_events, trace_info); 
      if (pre.exists())
      {
        if (precondition.exists())
          precondition = Runtime::merge_events(&trace_info, precondition, pre);
        else
          precondition = pre;
      }
      std::vector<std::vector<CopySrcDstField> > local_fields(instances.size());
      layout->compute_copy_offsets(copy_mask, instances.front(),
#ifdef LEGION_SPY
                                   instance_events.front(),
#endif
                                   local_fields.front());
      std::vector<std::vector<Reservation> > reservations(instances.size());
      view->find_field_reservations(copy_mask, instance_points.front(),
                                    reservations.front());
      std::vector<ApEvent> instance_preconditions(instances.size(),
                                                  precondition);
      std::vector<ApEvent> local_init_events;
      if (instances.size() > 1)
      {
        set_redop(local_fields[0]);
        for (unsigned idx = 1; idx < instances.size(); idx++)
        {
          // Find the reservations for the other instances for later
          view->find_field_reservations(copy_mask, instance_points[idx],
                                        reservations[idx]);
          layout->compute_copy_offsets(copy_mask, instances[idx],
#ifdef LEGION_SPY
                                       instance_events[idx],
#endif
                                       local_fields[idx]);
          const PhysicalTraceInfo &inst_info = (local_analyses == NULL) ?
            trace_info : local_analyses->at(idx)->trace_info;
          const ApEvent reduced = copy_expression->issue_copy(op, inst_info,
              local_fields.front(), local_fields[idx], reservations.front(),
#ifdef LEGION_SPY
              tree_id, tree_id,
#endif
              precondition, predicate_guard);
          if (reduced.exists())
          {
            instance_preconditions[idx] = reduced;
            local_init_events.push_back(reduced);
          }
          if (inst_info.recording)
          {
            const UniqueInst src_inst(view, instance_points.front());
            const UniqueInst dst_inst(view, instance_points[idx]);
            inst_info.record_copy_insts(reduced, copy_expression,
               src_inst, dst_inst, copy_mask, copy_mask, redop, applied_events);
          }
        }
        clear_redop(local_fields[0]);
      }
      unsigned final_inst_index = 0;
      std::vector<ApEvent> local_final_events;
      // See if we've got to do the multi-node all-reduce
      if (collective_mapping->size() > 1)
      {
#ifdef DEBUG_LEGION
        // Better have an identity for initializing data
        assert(reduction_op->identity != NULL);
#endif
        if (multi_instance)
          // Case 1: each node has multiple instances
          final_inst_index = perform_multi_allreduce(view->fill_view, view->did,
              allreduce_tag, op, predicate_guard, copy_expression, copy_mask,
              trace_info, local_analyses, applied_events,instance_preconditions,
              local_fields, reservations, local_init_events,local_final_events);
        else
          // Case 2: there are some nodes that only have one instance
          // Pair up nodes to have them cooperate to have two buffers
          // that we can ping-pong between to do the all-reduce "inplace"
          perform_single_allreduce(view->fill_view, view->did, allreduce_tag,
              op, predicate_guard, copy_expression, copy_mask, trace_info,
              applied_events, instance_preconditions, local_fields,
              reservations, local_init_events, local_final_events);
      }
      else if (!local_init_events.empty())
      {
#ifdef DEBUG_LEGION
        assert(final_inst_index == 0);
#endif
        // All the instances were local so just record 
        // that the first instance is ready when all the 
        // reductions are done
        instance_preconditions[final_inst_index] =
          Runtime::merge_events(&trace_info, local_init_events);
      }
      // Finally broadcast out the result from the first instance to all
      // our local instances so that they all have the same data
      // Reset the redop for the final inst fields
      const std::vector<Reservation> no_reservations;
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        // Skip the one that has the final result that we're copying from
        if (idx == final_inst_index)
        {
          local_final_events.push_back(instance_preconditions[idx]);
          continue;
        }
        std::vector<CopySrcDstField> &dst_fields = local_fields[idx];
        std::vector<CopySrcDstField> &src_fields = 
          local_fields[final_inst_index];
        // Issue the copy
        const PhysicalTraceInfo &inst_info = (local_analyses == NULL) ?
          trace_info : local_analyses->at(idx)->trace_info;
        const ApEvent local_pre = Runtime::merge_events(&inst_info,
         instance_preconditions[idx], instance_preconditions[final_inst_index]);
        const ApEvent local_post = copy_expression->issue_copy(op, inst_info,
            dst_fields, src_fields, no_reservations,
#ifdef LEGION_SPY
            tree_id, tree_id,
#endif
            local_pre, predicate_guard);
        if (local_post.exists())
          local_final_events.push_back(local_post); 
        if (inst_info.recording)
        {
          const UniqueInst src_inst(view, instance_points[final_inst_index]);
          const UniqueInst dst_inst(view, instance_points[idx]);
          inst_info.record_copy_insts(local_post, copy_expression, src_inst,
                dst_inst, copy_mask, copy_mask, 0/*redop*/, applied_events);
        }
      }
      // Now compute the event for when all the reductions are done
      ApEvent done = Runtime::merge_events(&trace_info, local_final_events);
      if (done.exists())
      {
        const RtEvent collect_event = trace_info.get_collect_event();
        view->add_copy_user(false/*reading*/, 0/*redop*/, done, collect_event,
            copy_mask, copy_expression, op_id, index, recorded_events,
            trace_info.recording, runtime->address_space);
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::perform_single_allreduce(FillView *fill_view,
                                  const DistributedID reduce_view_did,
                                  const uint64_t allreduce_tag,
                                  Operation *op, PredEvent predicate_guard,
                                  IndexSpaceExpression *copy_expression,
                                  const FieldMask &copy_mask,
                                  const PhysicalTraceInfo &trace_info,
                                  std::set<RtEvent> &applied_events,
                                  std::vector<ApEvent> &instance_preconditions,
                      std::vector<std::vector<CopySrcDstField> > &local_fields,
                const std::vector<std::vector<Reservation> > &reservations,
                                  std::vector<ApEvent> &local_init_events,
                                  std::vector<ApEvent> &local_final_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!multi_instance);
#endif
      // Case 2: there are some nodes that only have one instance
      // Pair up nodes to have them cooperate to have two buffers
      // that we can ping-pong between to do the all-reduce "inplace"
      const int participants = collective_mapping->size() / 2; // truncate
      const int local_index = collective_mapping->find_index(local_space);
      const int local_rank = local_index / 2;
      const int local_offset = local_index % 2;
      int collective_radix = runtime->legion_collective_radix;
      int collective_log_radix, collective_stages;
      int participating_ranks, collective_last_radix;
      const bool participating = configure_collective_settings(
          participants, local_rank, collective_radix, collective_log_radix,
          collective_stages, participating_ranks, collective_last_radix);
      if (participating)
      {
        // Check to see if we need to handle stage -1 from non-participants
        // As well as from offset=1 down to offset=0
        if (local_offset == 0)
        {
          // We definitely will be expecting our partner
          std::vector<int> expected_ranks(1, local_rank);
          // We could be expecting up to two non-participants
          // User their index instead of rank to avoid key collision
          const int nonpart_index = local_index + 2*participating_ranks;
          for (int offset = 0; offset < 2; offset++)
          {
            const int rank = nonpart_index + offset;
            if (rank >= int(collective_mapping->size()))
              break;
            expected_ranks.push_back(rank);
          }
          set_redop(local_fields[0]);
          const UniqueInst dst_inst(reduce_view_did, instance_points.front());
          receive_allreduce_stage(dst_inst, allreduce_tag, -1/*stage*/, op,
            instance_preconditions[0], predicate_guard, copy_expression,
            copy_mask, trace_info, applied_events, local_fields[0], 
            reservations[0], &expected_ranks.front(),
            expected_ranks.size(), local_init_events);
          clear_redop(local_fields[0]);
          if (!local_init_events.empty())
            instance_preconditions[0] =
              Runtime::merge_events(&trace_info, local_init_events);
        }
        else
        {
          // local_offset == 1
          if (!local_init_events.empty())
            instance_preconditions[0] = 
              Runtime::merge_events(&trace_info, local_init_events);
          // Just need to send the reduction down to our partner
          const AddressSpaceID target = (*collective_mapping)[local_index-1];
          std::vector<ApEvent> src_events;
          send_allreduce_stage(allreduce_tag, -1/*stage*/, local_rank,
              instance_preconditions[0], predicate_guard, copy_expression,
              trace_info, local_fields[0], instance_points.front(),
              &target, 1/*target count*/, src_events);
          if (!src_events.empty())
          {
#ifdef DEBUG_LEGION
            assert(src_events.size() == 1);
#endif
            instance_preconditions[0] = src_events[0];
          }
        }
        // Do the stages
        for (int stage = 0; stage < collective_stages; stage++)
        {
          // Figure out the participating ranks
          std::vector<int> stage_ranks;
          if (stage < (collective_stages-1))
          {
            // Normal radix
            stage_ranks.reserve(collective_radix);
            for (int r = 1; r < collective_radix; r++)
            {
              int target = local_rank ^
                (r << (stage * collective_log_radix));
              stage_ranks.push_back(target);
            }
          }
          else
          {
            // Last stage so special radix
            stage_ranks.reserve(collective_last_radix);
            for (int r = 1; r < collective_last_radix; r++)
            {
              int target = local_rank ^
                (r << (stage * collective_log_radix));
              stage_ranks.push_back(target);
            }
          }
#ifdef DEBUG_LEGION
          assert(!stage_ranks.empty());
#endif
          // Always include ourselves in the ranks as well
          stage_ranks.push_back(local_rank);
          // Check to see if we're sending or receiving this stage
          if ((stage % 2) == local_offset)
          {
            // We're doing a sending stage
            std::vector<AddressSpaceID> targets(stage_ranks.size());
            for (unsigned idx = 0; idx < stage_ranks.size(); idx++)
            {
              // If we're even, send to the odd
              // If we're odd, send to the even
              const unsigned index =
                2 * stage_ranks[idx] + ((local_offset == 0) ? 1 : 0);
#ifdef DEBUG_LEGION
              assert(index < collective_mapping->size());
#endif
              targets[idx] = (*collective_mapping)[index];
            }
            std::vector<ApEvent> src_events;
            send_allreduce_stage(allreduce_tag, stage, local_rank,
                instance_preconditions[0], predicate_guard, copy_expression,
                trace_info, local_fields[0], instance_points.front(),
                &targets.front(), targets.size(), src_events);
            if (!src_events.empty())
              instance_preconditions[0] =
                Runtime::merge_events(&trace_info, src_events);
          }
          else
          {
            // We're doing a receiving stage
            // First issue a fill to initialize the instance
            // Realm should ignore the redop data on these fields
            instance_preconditions[0] = copy_expression->issue_fill(
                op, trace_info, local_fields[0], reduction_op->identity,
                reduction_op->sizeof_rhs,
#ifdef LEGION_SPY
                fill_view->fill_op_uid, field_space_node->handle, tree_id,
#endif
                instance_preconditions[0], predicate_guard);
            if (trace_info.recording)
            {
              const UniqueInst dst_inst(reduce_view_did, instance_points[0]);
              trace_info.record_fill_inst(instance_preconditions[0],
                  copy_expression, dst_inst, copy_mask,
                  applied_events, (redop > 0));
            }
            // Then check to see if we've received any reductions
            std::vector<ApEvent> dst_events;
            set_redop(local_fields[0]);
            const UniqueInst dst_inst(reduce_view_did, instance_points.front());
            receive_allreduce_stage(dst_inst, allreduce_tag, stage, op,
                instance_preconditions[0], predicate_guard, copy_expression,
                copy_mask, trace_info, applied_events, local_fields[0],
                reservations[0], &stage_ranks.front(),
                stage_ranks.size(), dst_events);
            clear_redop(local_fields[0]);
            if (!dst_events.empty())
              instance_preconditions[0] =
                Runtime::merge_events(&trace_info, dst_events);
          }
        }
        // If we have to do stage -1 then we can do that now
        // Check to see if we have the valid data or not
        if ((collective_stages % 2) == local_offset)
        {
          // We have the valid data, send it to up to two 
          // non-participants as well as our partner
          // If we're odd then make us even and vice-versa
          int partner_index = local_index + ((local_offset == 0) ? 1 : -1);
          const AddressSpaceID partner = (*collective_mapping)[partner_index];
          std::vector<AddressSpaceID> targets(1, partner);
          // Check for the two non-participants
          const unsigned offset = 2*participating_ranks;
          const unsigned one = offset + local_index;
          if (one < collective_mapping->size())
            targets.push_back((*collective_mapping)[one]);
          const unsigned two = offset + partner_index;
          if (two < collective_mapping->size())
            targets.push_back((*collective_mapping)[two]);
          send_allreduce_stage(allreduce_tag, -2/*stage*/, local_rank,
              instance_preconditions[0], predicate_guard, copy_expression,
              trace_info, local_fields[0], instance_points.front(),
              &targets.front(), targets.size(), local_final_events);
        }
        else
        {
          // Not reducing here, just standard copy
          // See if we received the copy from our partner
          std::vector<ApEvent> dst_events;
          // No reservations since this is a straight copy
          const std::vector<Reservation> no_reservations;
          const UniqueInst dst_inst(reduce_view_did, instance_points.front());
          receive_allreduce_stage(dst_inst, allreduce_tag, -2/*stage*/, op,
              instance_preconditions[0], predicate_guard, copy_expression,
              copy_mask, trace_info, applied_events, local_fields[0],
              no_reservations, &local_rank, 1/*total ranks*/, dst_events);
          if (!dst_events.empty())
          {
#ifdef DEBUG_LEGION
            assert(dst_events.size() == 1);
#endif
            instance_preconditions[0] = dst_events[0];
          }
        }
      }
      else
      {
        // Not a participant in the stages, so just need to do
        // the stage -1 send and receive
        if (!local_init_events.empty())
          instance_preconditions.front() = 
            Runtime::merge_events(&trace_info, local_init_events);
        // Truncate down
        const int target_rank = (local_index - 2*participating_ranks) / 2;
#ifdef DEBUG_LEGION
        assert(target_rank >= 0);
#endif
        // Then convert back to the appropriate index
        const int target_index = 2 * target_rank;
#ifdef DEBUG_LEGION
        assert(target_index < int(collective_mapping->size()));
#endif
        const AddressSpaceID target = (*collective_mapping)[target_index];
        std::vector<ApEvent> src_events;
        // Intentionally use the local_index here to avoid key collisions
        send_allreduce_stage(allreduce_tag, -1/*stage*/, local_index,
            instance_preconditions[0], predicate_guard, copy_expression,
            trace_info, local_fields[0], instance_points.front(),
            &target, 1/*total targets*/, src_events);
        if (!src_events.empty())
        {
#ifdef DEBUG_LEGION
          assert(src_events.size() == 1);
#endif
          instance_preconditions[0] = src_events[0];
        }
        // Check to see if we received the copy back yet
        // Keep the redop data zeroed out since we're doing normal copies
        // No reservations since this is a straight copy
        const std::vector<Reservation> no_reservations;
        std::vector<ApEvent> dst_events;
        const UniqueInst dst_inst(reduce_view_did, instance_points.front());
        receive_allreduce_stage(dst_inst, allreduce_tag, -2/*stage*/, op,
            instance_preconditions[0], predicate_guard, copy_expression,
            copy_mask, trace_info, applied_events, local_fields[0],
            no_reservations, &target_rank, 1/*total ranks*/, dst_events);
        if (!dst_events.empty())
        {
#ifdef DEBUG_LEGION
          assert(dst_events.size() == 1);
#endif
          instance_preconditions[0] = dst_events[0];
        }
      }
    }

    //--------------------------------------------------------------------------
    unsigned CollectiveManager::perform_multi_allreduce(FillView *fill_view,
                                  const DistributedID reduce_view_did,
                                  const uint64_t allreduce_tag,
                                  Operation *op, PredEvent predicate_guard,
                                  IndexSpaceExpression *copy_expression,
                                  const FieldMask &copy_mask,
                                  const PhysicalTraceInfo &trace_info,
                const std::vector<CollectiveCopyFillAnalysis*> *local_analyses,
                                  std::set<RtEvent> &applied_events,
                                  std::vector<ApEvent> &instance_preconditions,
                      std::vector<std::vector<CopySrcDstField> > &local_fields,
                const std::vector<std::vector<Reservation> > &reservations,
                                  std::vector<ApEvent> &local_init_events,
                                  std::vector<ApEvent> &local_final_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Case 1: each node has multiple instances
      assert(redop > 0);
      assert(multi_instance);
      assert(instances.size() > 1);
#endif
      const int participants = collective_mapping->size();
      const int local_rank = collective_mapping->find_index(local_space);
      int collective_radix = runtime->legion_collective_radix;
      int collective_log_radix, collective_stages;
      int participating_ranks, collective_last_radix;
      const bool participating = configure_collective_settings(
          participants, local_rank, collective_radix, collective_log_radix,
          collective_stages, participating_ranks, collective_last_radix);
      if (participating)
      {
        // Check to see if we need to wait for a remainder copy
        // for any non-participating ranks
        int remainder_rank = local_rank + participating_ranks;
        if (collective_mapping->size() <= size_t(remainder_rank))
          remainder_rank = -1;
        if (remainder_rank >= 0)
        {
          set_redop(local_fields[0]);
          const UniqueInst dst_inst(reduce_view_did, instance_points.front());
          receive_allreduce_stage(dst_inst, allreduce_tag, -1/*stage*/, op,
              instance_preconditions[0], predicate_guard, copy_expression,
              copy_mask, trace_info, applied_events, local_fields[0],
              reservations[0], &remainder_rank, 1/*total ranks*/, 
              local_init_events);
          clear_redop(local_fields[0]);
        }
        // We've now recorded any local reductions so update
        // the precondition event for the first local instance
        if (!local_init_events.empty())
          instance_preconditions.front() = 
            Runtime::merge_events(&trace_info, local_init_events);
        unsigned src_inst_index = 0;
        unsigned dst_inst_index = 1;
        // Issue the stages
        for (int stage = 0; stage < collective_stages; stage++)
        { 
          // Figure out where to send out messages first
          std::vector<int> stage_ranks;
          if (stage < (collective_stages-1))
          {
            // Normal radix
            stage_ranks.reserve(collective_radix-1);
            for (int r = 1; r < collective_radix; r++)
            {
              int target = local_rank ^
                (r << (stage * collective_log_radix));
              stage_ranks.push_back(target);
            }
          }
          else
          {
            // Last stage so special radix
            stage_ranks.reserve(collective_last_radix-1);
            for (int r = 1; r < collective_last_radix; r++)
            {
              int target = local_rank ^
                (r << (stage * collective_log_radix));
              stage_ranks.push_back(target);
            }
          }
#ifdef DEBUG_LEGION
          assert(!stage_ranks.empty());
#endif
          // Send out the messages to the dst ranks to perform copies
          std::vector<AddressSpaceID> targets(stage_ranks.size());
          for (unsigned idx = 0; idx < stage_ranks.size(); idx++)
            targets[idx] = (*collective_mapping)[stage_ranks[idx]];
          std::vector<ApEvent> src_events;
          const PhysicalTraceInfo &src_info = (local_analyses == NULL) ?
            trace_info : local_analyses->at(src_inst_index)->trace_info;
          send_allreduce_stage(allreduce_tag, stage, local_rank,
              instance_preconditions[src_inst_index], predicate_guard,
              copy_expression, src_info, local_fields[src_inst_index],
              instance_points[src_inst_index], &targets.front(), 
              targets.size(), src_events);
          // Issuse the fill for the destination instance
          // Realm should ignore the redop data on these fields
          const PhysicalTraceInfo &dst_info = (local_analyses == NULL) ?
            trace_info : local_analyses->at(dst_inst_index)->trace_info;
          instance_preconditions[dst_inst_index] =
            copy_expression->issue_fill(op, dst_info,
                local_fields[dst_inst_index],
                reduction_op->identity, reduction_op->sizeof_rhs,
#ifdef LEGION_SPY
                fill_view->fill_op_uid, field_space_node->handle, tree_id,
#endif
                instance_preconditions[dst_inst_index],
                predicate_guard);
          if (dst_info.recording)
          {
            const UniqueInst dst_inst(reduce_view_did, 
                      instance_points[dst_inst_index]);
            dst_info.record_fill_inst(instance_preconditions[dst_inst_index],
                copy_expression, dst_inst, copy_mask, 
                applied_events, true/*reduction*/);
          }
          set_redop(local_fields[dst_inst_index]);
          // Issue the reduction from the source to the destination
          ApEvent local_precondition = Runtime::merge_events(&dst_info,
              instance_preconditions[src_inst_index],
              instance_preconditions[dst_inst_index]);
          const ApEvent local_post = copy_expression->issue_copy(op, dst_info,
              local_fields[dst_inst_index], local_fields[src_inst_index],
              reservations[dst_inst_index],
#ifdef LEGION_SPY
              tree_id, tree_id,
#endif
              local_precondition, predicate_guard);
          std::vector<ApEvent> dst_events;
          if (local_post.exists())
          {
            src_events.push_back(local_post);
            dst_events.push_back(local_post);
          }
          if (dst_info.recording)
          {
            const UniqueInst src_inst(reduce_view_did,
                instance_points[src_inst_index]);
            const UniqueInst dst_inst(reduce_view_did,
                instance_points[dst_inst_index]);
            dst_info.record_copy_insts(local_post, copy_expression,
               src_inst, dst_inst, copy_mask, copy_mask, redop, applied_events);
          }
          // Update the source instance precondition
          // to reflect all the reduction copies read from it
          if (!src_events.empty())
            instance_preconditions[src_inst_index] =
              Runtime::merge_events(&src_info, src_events);
          // Now check to see if we're received any messages
          // for this stage, and if not make place holders for them
          const UniqueInst dst_inst(reduce_view_did, 
                                    instance_points[dst_inst_index]);
          receive_allreduce_stage(dst_inst, allreduce_tag, stage, op,
              instance_preconditions[dst_inst_index], predicate_guard,
              copy_expression, copy_mask, dst_info, applied_events, 
              local_fields[dst_inst_index], reservations[dst_inst_index],
              &stage_ranks.front(), stage_ranks.size(), dst_events);
          clear_redop(local_fields[dst_inst_index]);
          if (!dst_events.empty())
            instance_preconditions[dst_inst_index] =
              Runtime::merge_events(&dst_info, dst_events);
          // Update the src and dst instances for the next stage
          if (++src_inst_index == instances.size())
            src_inst_index = 0;
          if (++dst_inst_index == instances.size())
            dst_inst_index = 0;
        }
        // Send out the result to any non-participating ranks
        if (remainder_rank >= 0)
        {
          const AddressSpaceID target = (*collective_mapping)[remainder_rank];
          send_allreduce_stage(allreduce_tag, -1/*stage*/, local_rank,
              instance_preconditions[src_inst_index], predicate_guard,
              copy_expression, trace_info, local_fields[src_inst_index],
              instance_points[src_inst_index], &target, 1/*total targets*/,
              local_final_events);
        }
        return src_inst_index;
      }
      else
      {
        // Not a participant in the stages so just need to 
        // do the stage -1 send and receive
#ifdef DEBUG_LEGION
        assert(local_rank >= participating_ranks);
#endif
        if (!local_init_events.empty())
          instance_preconditions[0] = 
            Runtime::merge_events(&trace_info, local_init_events);
        const int mirror_rank = local_rank - participating_ranks;
        const AddressSpaceID target = (*collective_mapping)[mirror_rank];
        std::vector<ApEvent> src_events;
        send_allreduce_stage(allreduce_tag, -1/*stage*/, local_rank,
            instance_preconditions[0], predicate_guard, copy_expression,
            trace_info, local_fields[0], instance_points.front(),
            &target, 1/*total targets*/, src_events);
        if (!src_events.empty())
        {
#ifdef DEBUG_LEGION
          assert(src_events.size() == 1);
#endif
          instance_preconditions[0] = src_events[0];
        }
        // We can put this back in the first buffer without any
        // anti-dependences because we know the computation of the
        // result coming back had to already depend on the copy we
        // sent out to the target
        // Keep the local fields redop cleared since we're going to 
        // doing direct copies here into these instance and not reductions
        std::vector<ApEvent> dst_events;
        const std::vector<Reservation> no_reservations;
        const UniqueInst dst_inst(reduce_view_did, instance_points.front());
        receive_allreduce_stage(dst_inst, allreduce_tag, -1/*stage*/, op,
            instance_preconditions[0], predicate_guard, copy_expression,
            copy_mask, trace_info, applied_events, local_fields[0],
            no_reservations, &mirror_rank, 1/*total ranks*/, dst_events);
        if (!dst_events.empty())
        {
#ifdef DEBUG_LEGION
          assert(dst_events.size() == 1);
#endif
          instance_preconditions[0] = dst_events[0];
        }
        return 0;
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::send_allreduce_stage(const uint64_t allreduce_tag,
                                 const int stage, const int local_rank,
                                 ApEvent precondition,PredEvent predicate_guard,
                                 IndexSpaceExpression *copy_expression, 
                                 const PhysicalTraceInfo &trace_info,
                                 const std::vector<CopySrcDstField> &src_fields,
                                 const DomainPoint &src_point,
                                 const AddressSpaceID *targets, size_t total,
                                 std::vector<ApEvent> &src_events)
    //--------------------------------------------------------------------------
    {
      ApBarrier src_bar;
      ShardID src_bar_shard = 0;
      for (unsigned t = 0; t < total; t++)
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(allreduce_tag);
          rez.serialize(local_rank);
          rez.serialize(stage);
          pack_fields(rez, src_fields);
          rez.serialize(src_point);
          rez.serialize(precondition);
          rez.serialize<bool>(trace_info.recording);
          if (trace_info.recording)
          {
            if (!src_bar.exists())
            {
              src_bar = 
                ApBarrier(Realm::Barrier::create_barrier(total));
              src_bar_shard =
                trace_info.record_managed_barrier(src_bar, total);
              src_events.push_back(src_bar);
            }
            rez.serialize(src_bar);
            rez.serialize(src_bar_shard);
          }
          else
          {
            const ApUserEvent src_done =
              Runtime::create_ap_user_event(&trace_info);
            rez.serialize(src_done);
            src_events.push_back(src_done);
          }
        }
        runtime->send_collective_distribute_allreduce(targets[t], rez);
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::receive_allreduce_stage(const UniqueInst dst_inst,
                            const uint64_t allreduce_tag, 
                            const int stage, Operation *op,
                            ApEvent dst_precondition, PredEvent predicate_guard,
                            IndexSpaceExpression *copy_expression,
                            const FieldMask &copy_mask,
                            const PhysicalTraceInfo &trace_info,
                            std::set<RtEvent> &applied_events,
                            const std::vector<CopySrcDstField> &dst_fields,
                            const std::vector<Reservation> &reservations,
                            const int *expected_ranks, size_t total_ranks,
                            std::vector<ApEvent> &dst_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((stage != -2) || (total_ranks == 1));
#endif
      std::vector<AllReduceCopy> to_perform;
      {
        unsigned remaining = 0;
        AutoLock i_lock(inst_lock);
        for (unsigned r = 0; r < total_ranks; r++)
        {
          const CopyKey key(allreduce_tag, expected_ranks[r], stage);
          std::map<CopyKey,AllReduceCopy>::iterator finder =
            all_reduce_copies.find(key);
          if (finder != all_reduce_copies.end())
          {
            to_perform.emplace_back(std::move(finder->second));
            all_reduce_copies.erase(finder);
          }
          else
            remaining++;
        }
        if (remaining > 0)
        {
          // If we still have outstanding copies, save a data
          // structure for them for when they arrive
          const std::pair<uint64_t,int> key(allreduce_tag, stage);
#ifdef DEBUG_LEGION
          assert(remaining_stages.find(key) == remaining_stages.end());
#endif
          AllReduceStage &pending = remaining_stages[key];
          pending.dst_inst = dst_inst;
          pending.op = op;
          pending.copy_expression = copy_expression;
          copy_expression->add_nested_expression_reference(
              this->did, applied_events);
          pending.copy_mask = copy_mask;
          pending.dst_fields = dst_fields;
          pending.reservations = reservations;
          pending.trace_info = new PhysicalTraceInfo(trace_info);
          pending.dst_precondition = dst_precondition;
          pending.predicate_guard = predicate_guard;
          pending.remaining_postconditions.reserve(remaining);
          for (unsigned idx = 0; idx < remaining; idx++)
          {
            const ApUserEvent post = Runtime::create_ap_user_event(&trace_info);
            pending.remaining_postconditions.push_back(post);
            dst_events.push_back(post);
          }
          if (trace_info.recording)
          {
            pending.applied_event = Runtime::create_rt_user_event();
            applied_events.insert(pending.applied_event);
          }
        }
      }
      // Now we can perform any copies that we received
      for (std::vector<AllReduceCopy>::const_iterator it =
            to_perform.begin(); it != to_perform.end(); it++)
      {
        const ApEvent pre = Runtime::merge_events(&trace_info,
          it->src_precondition, dst_precondition);
        const ApEvent post = copy_expression->issue_copy(
            op, trace_info, dst_fields, it->src_fields, reservations,
#ifdef LEGION_SPY
            tree_id, tree_id,
#endif
            pre, predicate_guard);
        if (trace_info.recording)
        {
          const UniqueInst src_inst(dst_inst.view_did, it->src_point);
          trace_info.record_copy_insts(post, copy_expression,
              src_inst, dst_inst, copy_mask, copy_mask, redop, applied_events);
        }
        if (it->barrier_postcondition.exists())
        {
          Runtime::phase_barrier_arrive(
              it->barrier_postcondition, 1/*count*/, post);
          if (trace_info.recording)
            trace_info.record_barrier_arrival(it->barrier_postcondition,
                post, 1/*count*/, applied_events, it->barrier_shard);
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(it->src_postcondition.exists());
#endif
          Runtime::trigger_event(&trace_info, it->src_postcondition, post);
        }
        if (post.exists())
          dst_events.push_back(post);
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::process_distribute_allreduce(
              const uint64_t allreduce_tag, const int src_rank, const int stage,
              std::vector<CopySrcDstField> &src_fields,
              const ApEvent src_precondition, ApUserEvent src_postcondition,
              ApBarrier src_barrier, ShardID barrier_shard, 
              const DomainPoint &src_point)
    //--------------------------------------------------------------------------
    {
      LegionMap<std::pair<uint64_t,int>,AllReduceStage>::iterator finder;
      {
        AutoLock i_lock(inst_lock);
        const std::pair<uint64_t,int> stage_key(allreduce_tag, stage);
        finder = remaining_stages.find(stage_key);
        if (finder == remaining_stages.end())
        {
          const CopyKey key(allreduce_tag, src_rank, stage);
#ifdef DEBUG_LEGION
          assert(all_reduce_copies.find(key) == all_reduce_copies.end());
#endif
          AllReduceCopy &copy = all_reduce_copies[key];
          copy.src_fields.swap(src_fields);
          copy.src_precondition = src_precondition;
          copy.src_postcondition = src_postcondition;
          copy.barrier_postcondition = src_barrier;
          copy.barrier_shard = barrier_shard;
          copy.src_point = src_point;
          return;
        }
#ifdef DEBUG_LEGION
        assert(!finder->second.remaining_postconditions.empty());
#endif
        // We can release the lock because we know map iterators are 
        // not invalidated by insertion/deletion and any other copies
        // for this same stage are also just going to be reading except
        // for when we need to grab our event at the end to trigger
        // which we can re-take the lock to do
      }
      const ApEvent precondition = Runtime::merge_events(
          finder->second.trace_info, src_precondition,
          finder->second.dst_precondition);
      const ApEvent copy_post = finder->second.copy_expression->issue_copy(
          finder->second.op, *(finder->second.trace_info),
          finder->second.dst_fields, src_fields, finder->second.reservations,
#ifdef LEGION_SPY
          tree_id, tree_id,
#endif
          precondition, finder->second.predicate_guard);
      std::set<RtEvent> applied_events;
      if (finder->second.trace_info->recording)
      {
        const UniqueInst src_inst(finder->second.dst_inst.view_did, src_point);
        finder->second.trace_info->record_copy_insts(copy_post, 
            finder->second.copy_expression, src_inst, finder->second.dst_inst,
            finder->second.copy_mask, finder->second.copy_mask,
            redop, applied_events);
      }
      if (src_barrier.exists())
      {
        Runtime::phase_barrier_arrive(src_barrier, 1/*count*/, copy_post);
        finder->second.trace_info->record_barrier_arrival(src_barrier,
            copy_post, 1/*count*/, applied_events, barrier_shard);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(src_postcondition.exists());
#endif
        Runtime::trigger_event(finder->second.trace_info, 
                               src_postcondition, copy_post);
      }
      RtUserEvent applied;
      ApUserEvent to_trigger;
      PhysicalTraceInfo *trace_info = NULL;
      IndexSpaceExpression *copy_expression = NULL;
      {
        // Retake the lock and see if we're the last arrival
        AutoLock i_lock(inst_lock);
        // Save any applied events that we have
        if (!applied_events.empty())
        {
          finder->second.applied_events.insert(
              applied_events.begin(), applied_events.end());
#ifdef DEBUG_LEGION
          applied_events.clear();
#endif
        }
#ifdef DEBUG_LEGION
        assert(!finder->second.remaining_postconditions.empty());
#endif
        to_trigger = finder->second.remaining_postconditions.back();
        finder->second.remaining_postconditions.pop_back();
        if (finder->second.remaining_postconditions.empty())
        {
          // Last pass through, grab data and remove from the stages
          trace_info = finder->second.trace_info;
          copy_expression = finder->second.copy_expression;
          applied = finder->second.applied_event;
          applied_events.swap(finder->second.applied_events);
          remaining_stages.erase(finder);
        }
        else // Need a copy of this
          trace_info = new PhysicalTraceInfo(*(finder->second.trace_info));
      }
      Runtime::trigger_event(trace_info, to_trigger, copy_post); 
      if (applied.exists())
      {
        if (!applied_events.empty())
          Runtime::trigger_event(applied,Runtime::merge_events(applied_events));
        else
          Runtime::trigger_event(applied);
#ifdef DEBUG_LEGION
        applied_events.clear();
#endif
      }
#ifdef DEBUG_LEGION
      assert(applied_events.empty());
#endif
      delete trace_info;
      if ((copy_expression != NULL) &&
          copy_expression->remove_nested_expression_reference(this->did))
        delete copy_expression;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_distribute_allreduce(
                   Runtime *runtime, AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      CollectiveManager *manager = static_cast<CollectiveManager*>(
          runtime->find_or_request_instance_manager(did, ready));
      uint64_t allreduce_tag;
      derez.deserialize(allreduce_tag);
      int src_rank;
      derez.deserialize(src_rank);
      int stage;
      derez.deserialize(stage);
      size_t num_src_fields;
      derez.deserialize(num_src_fields);
      std::vector<CopySrcDstField> src_fields(num_src_fields);
      std::set<RtEvent> ready_events;
      unpack_fields(src_fields, derez, ready_events, manager, ready, runtime);
      DomainPoint src_point;
      derez.deserialize(src_point);
      ApEvent src_precondition;
      derez.deserialize(src_precondition);
      bool recording;
      derez.deserialize<bool>(recording);
      ApBarrier src_barrier;
      ShardID barrier_shard = 0;
      ApUserEvent src_postcondition;
      if (recording)
      {
        derez.deserialize(src_barrier);
        derez.deserialize(barrier_shard);
      }
      else
        derez.deserialize(src_postcondition);

      if (ready.exists() && !ready.has_triggered())
        ready_events.insert(ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      manager->process_distribute_allreduce(allreduce_tag, src_rank, stage,
                            src_fields, src_precondition, src_postcondition,
                            src_barrier, barrier_shard, src_point);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::perform_collective_broadcast(InstanceView *dst_view,
                                const std::vector<CopySrcDstField> &src_fields,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expression,
                                Operation *op, const unsigned index,
                                const size_t op_ctx_index,
                                const FieldMask &copy_mask,
                                const UniqueInst &src_inst,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                ApUserEvent copy_done, ApUserEvent all_done,
                                ApBarrier all_bar, ShardID owner_shard,
                                AddressSpaceID origin, 
                                const bool copy_restricted)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(copy_done.exists());
      assert(!instances.empty());
      assert(dst_view->manager == this);
      assert(collective_mapping != NULL);
      assert(collective_mapping->contains(local_space));
      assert((op != NULL) || !copy_restricted);
#endif
      RtEvent analyses_ready;
      const std::vector<CollectiveCopyFillAnalysis*> *local_analyses = NULL;
      if (!copy_restricted)
      {
        // If this is not a copy-out to a restricted collective instance 
        // then we should be able to find our local analyses to use for 
        // performing operations
        analyses_ready = find_collective_analyses(dst_view->did,
                            op_ctx_index, index, local_analyses);
#ifdef DEBUG_LEGION
        assert(local_analyses != NULL);
#endif
        // If we're recording then we need to wait now to get a valid
        // trace info for capturing the trace for events we send to 
        // remote nodes, otherwise we just need to wait before doing
        // the fill calls
        if ((trace_info.recording || (op == NULL)) && 
            analyses_ready.exists() && !analyses_ready.has_triggered())
          analyses_ready.wait();
        if (op == NULL)
          op = local_analyses->front()->op;
      }
#ifdef DEBUG_LEGION
      assert(op != NULL);
#endif
      const PhysicalTraceInfo &local_info = 
        ((local_analyses == NULL) || !trace_info.recording) ? trace_info : 
        local_analyses->front()->trace_info;
      const UniqueID op_id = op->get_unique_op_id();
      // Do the copy to our local instance first
      const ApEvent dst_pre = dst_view->find_copy_preconditions(
          false/*reading*/, 0/*redop*/, copy_mask, copy_expression,
          op_id, index, applied_events, local_info);
      // Get the precondition for the local copy
      if (dst_pre.exists())
      {
        if (precondition.exists())
          precondition =
            Runtime::merge_events(&local_info, precondition, dst_pre);
        else
          precondition = dst_pre;
      }
      // Get the dst_fields and reservations for performing the local reductions
      std::vector<CopySrcDstField> local_fields;
      layout->compute_copy_offsets(copy_mask, instances.front(),
#ifdef LEGION_SPY
                                   instance_events.front(),
#endif
                                   local_fields);
      const std::vector<Reservation> no_reservations;
      const ApEvent copy_post = copy_expression->issue_copy(
          op, local_info, local_fields, src_fields, no_reservations,
#ifdef LEGION_SPY
          tree_id, tree_id,
#endif
          precondition, predicate_guard);
      if (local_info.recording)
      {
        const UniqueInst dst_inst(dst_view, instance_points.front());
        local_info.record_copy_insts(copy_post, copy_expression,
          src_inst, dst_inst, copy_mask, copy_mask, 0/*redop*/, applied_events);
      }
      Runtime::trigger_event(&trace_info, copy_done, copy_post);
      // Always record the writer to ensure later reads catch it
      dst_view->add_copy_user(false/*reading*/, 0/*redop*/, copy_post,
          local_info.get_collect_event(), copy_mask, copy_expression,
          op_id, index, recorded_events, local_info.recording,
          runtime->address_space);
      // Broadcast out the copy events to any children
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(origin, local_space, children);
      // See if we're done
      if (children.empty() && (instances.size() == 1))
      {
        if (all_done.exists())
          Runtime::trigger_event(&trace_info, all_done, copy_post);
        return;
      }
      ApBarrier broadcast_bar;
      ShardID broadcast_shard = 0;
      std::vector<ApEvent> read_events, done_events;
      const UniqueInst local_inst(dst_view, instance_points.front());
      for (std::vector<AddressSpaceID>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(dst_view->did);
          pack_fields(rez, local_fields);
          local_inst.serialize(rez);
          rez.serialize(copy_post);
          rez.serialize(predicate_guard);
          copy_expression->pack_expression(rez, *it);
          rez.serialize<bool>(copy_restricted);
          if (copy_restricted)
            op->pack_remote_operation(rez, *it, applied_events);
          rez.serialize(index);
          rez.serialize(op_ctx_index);
          rez.serialize(copy_mask);
          local_info.pack_trace_info(rez, applied_events);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (local_info.recording)
          {
            if (!broadcast_bar.exists())
            {
              broadcast_bar =
                ApBarrier(Realm::Barrier::create_barrier(children.size()));
              broadcast_shard = local_info.record_managed_barrier(broadcast_bar,
                                                               children.size());
              read_events.push_back(broadcast_bar);
            }
            rez.serialize(broadcast_bar);
            rez.serialize(broadcast_shard);
            rez.serialize(all_bar);
            if (all_bar.exists())
              rez.serialize(owner_shard);
          }
          else
          {
            const ApUserEvent broadcast = 
              Runtime::create_ap_user_event(&local_info);
            rez.serialize(broadcast);
            read_events.push_back(broadcast);
            ApUserEvent done;
            if (all_done.exists())
            {
              done = Runtime::create_ap_user_event(&local_info);
              done_events.push_back(done);
            }
            rez.serialize(done);
          }
          rez.serialize(origin);
        }
        runtime->send_collective_distribute_broadcast(*it, rez);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      // Now broadcast out to the rest of our local instances
      // TODO: for now we just blast this out but we could at
      // some point build a local broadcast tree here for the
      // instances within this node
      // Do the last wait before we need our analyses for recording 
      // and profiling requests from the mappers
      if (analyses_ready.exists() && !analyses_ready.has_triggered())
        analyses_ready.wait();
      for (unsigned idx = 1; idx < instances.size(); idx++)
      {
        std::vector<CopySrcDstField> dst_fields;
        layout->compute_copy_offsets(copy_mask, instances[idx],
#ifdef LEGION_SPY
                                     instance_events[idx],
#endif
                                     dst_fields);
        const PhysicalTraceInfo &inst_info = (local_analyses == NULL) ?
          trace_info : local_analyses->at(idx)->trace_info;
        const ApEvent local_copy = copy_expression->issue_copy(
            op, inst_info, dst_fields, local_fields, no_reservations,
#ifdef LEGION_SPY
            tree_id, tree_id,
#endif
            copy_post, predicate_guard);
        if (local_copy.exists())
          read_events.push_back(local_copy);
        if (inst_info.recording)
        {
          const UniqueInst dst_inst(dst_view, instance_points[idx]);
          inst_info.record_copy_insts(local_copy, copy_expression, local_inst,
              dst_inst, copy_mask, copy_mask, 0/*redop*/, applied_events);
        }
      }
      // We've done all the copies, if we have any readers, then merge
      // them to together to record them as a copy user, they'll dominate
      // the write to the local instance so there's no need to record that
      // explicitly. If we don't have any readers though, then we'll record
      // the writer explicitly
      if (!read_events.empty())
      {
        ApEvent read_done = Runtime::merge_events(&local_info, read_events);
        if (read_done.exists())
        {
          dst_view->add_copy_user(false/*reading*/, 0/*redop*/, read_done,
              local_info.get_collect_event(), copy_mask, copy_expression,
              op_id, index, recorded_events, local_info.recording,
              runtime->address_space);
          if (all_bar.exists() || all_done.exists())
            done_events.push_back(all_done);
        }
      }
      if (all_bar.exists())
      {
        ApEvent arrival;
        if (!done_events.empty())
          arrival = Runtime::merge_events(&local_info, done_events);
        Runtime::phase_barrier_arrive(all_bar, 1/*count*/, arrival);
        local_info.record_barrier_arrival(all_bar, arrival, 1/*count*/,
                                          applied_events, owner_shard);
      }
      else if (all_done.exists())
      {
        if (!done_events.empty())
          Runtime::trigger_event(&trace_info, all_done,
              Runtime::merge_events(&local_info, done_events));
        else
          Runtime::trigger_event(&local_info, all_done);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_distribute_broadcast(
                   Runtime *runtime, AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID man_did, dst_did;
      derez.deserialize(man_did);
      RtEvent man_ready, dst_ready;
      CollectiveManager *manager = static_cast<CollectiveManager*>(
          runtime->find_or_request_instance_manager(man_did, man_ready));
      derez.deserialize(dst_did);
      InstanceView *dst_view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(dst_did, dst_ready));
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<CopySrcDstField> src_fields(num_fields);
      std::set<RtEvent> recorded_events, ready_events, applied_events;
      unpack_fields(src_fields, derez, ready_events, manager,man_ready,runtime);
      UniqueInst src_inst;
      src_inst.deserialize(derez);
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      IndexSpaceExpression *copy_expression =
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      bool copy_restricted;
      derez.deserialize(copy_restricted);
      Operation *op = NULL;
      if (copy_restricted)
        op = RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      PhysicalTraceInfo trace_info =
        PhysicalTraceInfo::unpack_trace_info(derez, runtime);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApUserEvent ready, all_done;
      ApBarrier all_bar;
      ShardID owner_shard = 0;
      if (trace_info.recording)
      {
        ApBarrier broadcast_bar;
        derez.deserialize(broadcast_bar);
        ShardID broadcast_shard;
        derez.deserialize(broadcast_shard);
        // Copy-elmination will take care of this for us
        // when the trace is optimized
        ready = Runtime::create_ap_user_event(&trace_info);
        Runtime::phase_barrier_arrive(broadcast_bar, 1/*count*/, ready);
        trace_info.record_barrier_arrival(broadcast_bar, ready, 1/*count*/, 
                                          applied_events, broadcast_shard);
        derez.deserialize(all_bar);
        if (all_bar.exists())
          derez.deserialize(owner_shard);
      }
      else
      {
        derez.deserialize(ready);
        derez.deserialize(all_done);
      }
      AddressSpaceID origin;
      derez.deserialize(origin); 

      if (man_ready.exists() && !man_ready.has_triggered())
        ready_events.insert(man_ready);
      if (dst_ready.exists() && !dst_ready.has_triggered())
        ready_events.insert(dst_ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      manager->perform_collective_broadcast(dst_view, src_fields, precondition,
          predicate_guard, copy_expression, op, index, op_ctx_index,
          copy_mask, src_inst, trace_info, recorded_events, applied_events,
          ready, all_done, all_bar, owner_shard, origin, copy_restricted);

      if (!recorded_events.empty())
        Runtime::trigger_event(recorded,Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (op != NULL)
        delete op;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::perform_collective_reducecast(
                                IndividualManager *source,
                                InstanceView *dst_view,
                                const std::vector<CopySrcDstField> &src_fields,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expression,
                                Operation *op, const unsigned index,
                                const size_t op_ctx_index,
                                const FieldMask &copy_mask,
                                const UniqueInst &src_inst,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                ApUserEvent reduce_done, ApBarrier all_bar,
                                ShardID owner_shard, AddressSpaceID origin, 
                                const bool copy_restricted)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(source->redop > 0);
      assert(!instances.empty());
      assert(dst_view->manager == this);
      assert(collective_mapping != NULL);
      assert(collective_mapping->contains(local_space));
      assert((op != NULL) || !copy_restricted);
      // Only one of these should be valid
      assert(reduce_done.exists() != all_bar.exists());
#endif
      // If we have any children, broadcast this out to the first in parallel
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(origin, local_space, children);
      std::vector<ApEvent> reduce_events;
      if (!children.empty() && !trace_info.recording)
      {
        // Help out with broadcasting the precondition event
        // In the tracing case the precondition is a barrier 
        // so there's no need for us to do this
        const ApUserEvent local_precondition =
          Runtime::create_ap_user_event(&trace_info);
        Runtime::trigger_event(&trace_info, local_precondition, precondition);
        precondition = local_precondition;
      }
      for (std::vector<AddressSpaceID>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(source->did);
          rez.serialize(dst_view->did);
          source->pack_fields(rez, src_fields);
          src_inst.serialize(rez);
          rez.serialize(precondition);
          rez.serialize(predicate_guard);
          copy_expression->pack_expression(rez, *it);
          rez.serialize<bool>(copy_restricted);
          if (copy_restricted)
            op->pack_remote_operation(rez, *it, applied_events);
          rez.serialize(index);
          rez.serialize(op_ctx_index);
          rez.serialize(copy_mask);
          trace_info.pack_trace_info(rez, applied_events);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (trace_info.recording)
          {
            rez.serialize(all_bar);
            rez.serialize(owner_shard);
          }
          else
          {
            const ApUserEvent reduced =
              Runtime::create_ap_user_event(&trace_info);
            rez.serialize(reduced);
            reduce_events.push_back(reduced);
          }
          rez.serialize(origin);
        }
        runtime->send_collective_distribute_reducecast(*it, rez);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      RtEvent analyses_ready;
      const std::vector<CollectiveCopyFillAnalysis*> *local_analyses = NULL;
      if (!copy_restricted)
      {
        // If this is not a copy-out to a restricted collective instance 
        // then we should be able to find our local analyses to use for 
        // performing operations
        analyses_ready = find_collective_analyses(dst_view->did,
                            op_ctx_index, index, local_analyses);
#ifdef DEBUG_LEGION
        assert(local_analyses != NULL);
#endif
        // If we're recording then we need to wait now to get a valid
        // trace info for capturing the trace for events we send to 
        // remote nodes, otherwise we just need to wait before doing
        // the fill calls
        if ((trace_info.recording || (op == NULL)) && 
            analyses_ready.exists() && !analyses_ready.has_triggered())
          analyses_ready.wait();
        if (op == NULL)
          op = local_analyses->front()->op;
      }
#ifdef DEBUG_LEGION
      assert(op != NULL);
#endif
      const PhysicalTraceInfo &local_info = 
        ((local_analyses == NULL) || !trace_info.recording) ? trace_info : 
        local_analyses->front()->trace_info;
      const UniqueID op_id = op->get_unique_op_id();
      // Compute the reducing precondition for our local instances
      ApEvent reduce_pre = dst_view->find_copy_preconditions(
          false/*reading*/, source->redop, copy_mask, copy_expression,
          op_id, index, applied_events, local_info);
      if (precondition.exists())
      {
        if (reduce_pre.exists())
          reduce_pre =
            Runtime::merge_events(&local_info, precondition, reduce_pre);
        else
          reduce_pre = precondition;
      }
      std::vector<CopySrcDstField> local_fields;
      std::vector<Reservation> local_reservations;
      std::vector<ApEvent> local_done_events;
      // Issue the reductions to our local instances
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        layout->compute_copy_offsets(copy_mask, instances[idx],
#ifdef LEGION_SPY
                                     instance_events[idx],
#endif
                                     local_fields);
        for (std::vector<CopySrcDstField>::iterator it =
              local_fields.begin(); it != local_fields.end(); it++)
          it->set_redop(source->redop, (redop > 0), true/*exclusive*/);
        dst_view->find_field_reservations(copy_mask, instance_points[idx],
                                          local_reservations);
        const PhysicalTraceInfo &inst_info = (local_analyses == NULL) ?
          trace_info : local_analyses->at(idx)->trace_info;
        const ApEvent reduce_done = copy_expression->issue_copy(
            op, inst_info, local_fields, src_fields, local_reservations,
#ifdef LEGION_SPY
            tree_id, tree_id,
#endif
            reduce_pre, predicate_guard);
        if (reduce_done.exists())
          local_done_events.push_back(reduce_done);
        if (inst_info.recording)
        {
          const UniqueInst dst_inst(dst_view, instance_points[idx]);
          inst_info.record_copy_insts(reduce_done, copy_expression, src_inst,
              dst_inst, copy_mask, copy_mask, source->redop, applied_events);
        }
        local_fields.clear();
        local_reservations.clear();
      }
      ApEvent local_done;
      if (!local_done_events.empty())
      {
        local_done = Runtime::merge_events(&local_info, local_done_events);
        if (local_done.exists())
        {
          dst_view->add_copy_user(false/*reading*/, source->redop,
              local_done, local_info.get_collect_event(), copy_mask,
              copy_expression, op_id, index, recorded_events,
              local_info.recording, runtime->address_space);
          if (reduce_done.exists())
            reduce_events.push_back(local_done);
        }
      }
      if (all_bar.exists())
      {
        Runtime::phase_barrier_arrive(all_bar, 1/*count*/, local_done);
        local_info.record_barrier_arrival(all_bar, local_done, 1/*count*/,
                                          applied_events, owner_shard);
      }
      else
      {
        if (!reduce_events.empty())
          Runtime::trigger_event(&local_info, reduce_done, 
              Runtime::merge_events(&local_info, reduce_events));
        else
          Runtime::trigger_event(&local_info, reduce_done);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_distribute_reducecast(
                   Runtime *runtime, AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID man_did, src_did, dst_did;
      derez.deserialize(man_did);
      RtEvent man_ready, src_ready, dst_ready;
      CollectiveManager *manager = static_cast<CollectiveManager*>(
          runtime->find_or_request_instance_manager(man_did, man_ready));
      derez.deserialize(src_did);
      IndividualManager *source_manager = static_cast<IndividualManager*>(
          runtime->find_or_request_instance_manager(src_did, src_ready));
      derez.deserialize(dst_did);
      InstanceView *dst_view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(dst_did, dst_ready));
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<CopySrcDstField> src_fields(num_fields);
      std::set<RtEvent> recorded_events, ready_events, applied_events;
      unpack_fields(src_fields, derez, ready_events, manager,man_ready,runtime);
      UniqueInst src_inst;
      src_inst.deserialize(derez);
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      IndexSpaceExpression *copy_expression =
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      bool copy_restricted;
      derez.deserialize(copy_restricted);
      Operation *op = NULL;
      if (copy_restricted)
        op = RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      PhysicalTraceInfo trace_info =
        PhysicalTraceInfo::unpack_trace_info(derez, runtime);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApUserEvent ready;
      ApBarrier all_bar;
      ShardID owner_shard = 0;
      if (trace_info.recording)
      {
        derez.deserialize(all_bar);
        if (all_bar.exists())
          derez.deserialize(owner_shard);
      }
      else
        derez.deserialize(ready);
      AddressSpaceID origin;
      derez.deserialize(origin); 

      if (man_ready.exists() && !man_ready.has_triggered())
        ready_events.insert(man_ready);
      if (src_ready.exists() && !src_ready.has_triggered())
        ready_events.insert(src_ready);
      if (dst_ready.exists() && !dst_ready.has_triggered())
        ready_events.insert(dst_ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      manager->perform_collective_reducecast(source_manager, dst_view,
          src_fields, precondition, predicate_guard, copy_expression, op, 
          index, op_ctx_index, copy_mask, src_inst, trace_info, recorded_events,
          applied_events, ready, all_bar, owner_shard, origin, copy_restricted);

      if (!recorded_events.empty())
        Runtime::trigger_event(recorded,Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (op != NULL)
        delete op;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::perform_collective_hourglass(
                                          CollectiveManager *source,
                                          InstanceView *src_view,
                                          InstanceView *dst_view,
                                          ApEvent precondition,
                                          PredEvent predicate_guard,
                                          IndexSpaceExpression *copy_expression,
                                          Operation *op, const unsigned index,
                                          const FieldMask &copy_mask,
                                          const DomainPoint &src_point,
                                          const PhysicalTraceInfo &trace_info,
                                          std::set<RtEvent> &recorded_events,
                                          std::set<RtEvent> &applied_events,
                                          ApUserEvent all_done,
                                          AddressSpaceID target,
                                          const bool copy_restricted)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(op != NULL);
      assert(dst_view->manager == this);
      assert(src_view->manager == source);
      assert(collective_mapping != NULL);
      assert(collective_mapping->contains(local_space));
      assert(source->is_reduction_manager());
#endif
      if (target != local_space)
      {
        // Send this to where the target address space is
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(this->did);
          rez.serialize(source->did);
          rez.serialize(src_view->did);
          rez.serialize(dst_view->did);
          rez.serialize(precondition);
          rez.serialize(predicate_guard);
          copy_expression->pack_expression(rez, target);
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize(copy_mask);
          rez.serialize(src_point);
          trace_info.pack_trace_info(rez, applied_events);
          rez.serialize(recorded);
          rez.serialize(applied);
          rez.serialize(all_done);
          rez.serialize(copy_restricted);
        }
        runtime->send_collective_distribute_hourglass(target, rez);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
        return;
      }
#ifdef DEBUG_LEGION
      assert(!instances.empty());
#endif
      const UniqueID op_id = op->get_unique_op_id();
      // Perform the collective reduction first on the source
      const ApEvent reduce_pre = dst_view->find_copy_preconditions(
          false/*reding*/, 0/*redop*/, copy_mask, copy_expression,
          op_id, index, applied_events, trace_info);
      if (reduce_pre.exists())
      {
        if (precondition.exists())
          precondition =
            Runtime::merge_events(&trace_info, precondition, reduce_pre);
        else
          precondition = reduce_pre;
      }
      // We'll just use the first instance for the target
      std::vector<CopySrcDstField> local_fields;
      layout->compute_copy_offsets(copy_mask, instances.front(),
#ifdef LEGION_SPY
                                   instance_events.front(),
#endif
                                   local_fields);
      std::vector<Reservation> reservations;
      dst_view->find_field_reservations(copy_mask, instance_points.front(),
                                        reservations);
      for (unsigned idx = 0; idx < local_fields.size(); idx++)
        local_fields[idx].set_redop(source->redop, false/*fold*/,
                                    true/*exclusive*/);
      // Build the reduction tree down to our first instance
      const AddressSpaceID origin = src_point.exists() ? 
        source->get_instance(src_point).address_space() :
        source->select_source_space(local_space);
      ApEvent reduced;
      const UniqueInst local_inst(dst_view, instance_points.front());
      if (origin != local_space)
      {
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(source->did);
          rez.serialize(src_view->did);
          pack_fields(rez, local_fields);
          rez.serialize<size_t>(reservations.size());
          for (unsigned idx = 0; idx < reservations.size(); idx++)
            rez.serialize(reservations[idx]);
          rez.serialize(precondition);
          rez.serialize(predicate_guard);
          copy_expression->pack_expression(rez, origin);
          op->pack_remote_operation(rez, origin, applied_events); 
          rez.serialize(index);
          rez.serialize(copy_mask);
          rez.serialize(copy_mask);
          rez.serialize(src_point);
          local_inst.serialize(rez);
          trace_info.pack_trace_info(rez, applied_events);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (trace_info.recording)
          {
            ApBarrier bar(Realm::Barrier::create_barrier(1/*arrivals*/));
            const ShardID sid = trace_info.record_managed_barrier(bar, 1);
            rez.serialize(bar);
            rez.serialize(sid);
            reduced = bar;
          }
          else
          {
            const ApUserEvent to_trigger = 
              Runtime::create_ap_user_event(&trace_info);
            rez.serialize(to_trigger);
            reduced = to_trigger;
          }
          rez.serialize(origin);
        }
        runtime->send_collective_distribute_reduction(origin, rez);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      else
      {
        const ApUserEvent to_trigger = 
          Runtime::create_ap_user_event(&trace_info);
        source->perform_collective_reduction(src_view, local_fields,
            reservations, precondition, predicate_guard, copy_expression,
            op, index, copy_mask, copy_mask, src_point, local_inst, trace_info,
            recorded_events, applied_events, to_trigger, origin);
        reduced = to_trigger;
      }
      // Do the broadcast out, start with any children
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(local_space, local_space, children);
      ApBarrier broadcast_bar, all_bar;
      ShardID broadcast_shard = 0, owner_shard = 0;
      std::vector<ApEvent> broadcast_events;
      std::vector<ApEvent> all_done_events;
      if (all_done.exists() && trace_info.recording)
      {
        const size_t arrivals = collective_mapping->size();
        all_bar = ApBarrier(Realm::Barrier::create_barrier(arrivals));
        owner_shard = trace_info.record_managed_barrier(all_bar, arrivals);
      }
      for (std::vector<AddressSpaceID>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(this->did);
          rez.serialize(dst_view->did);
          pack_fields(rez, local_fields);
          local_inst.serialize(rez);
          rez.serialize(reduced);
          rez.serialize(predicate_guard);
          copy_expression->pack_expression(rez, *it);
          rez.serialize<bool>(copy_restricted);
          if (copy_restricted)
            op->pack_remote_operation(rez, origin, applied_events); 
          rez.serialize(index);
          rez.serialize(op->get_ctx_index());
          rez.serialize(copy_mask);
          trace_info.pack_trace_info(rez, applied_events);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (trace_info.recording)
          {
            if (!broadcast_bar.exists())
            {
              broadcast_bar =
                ApBarrier(Realm::Barrier::create_barrier(children.size()));
              broadcast_shard = trace_info.record_managed_barrier(broadcast_bar,
                                                               children.size());
              broadcast_events.push_back(broadcast_bar);
            }
            rez.serialize(broadcast_bar);
            rez.serialize(broadcast_shard);
            rez.serialize(all_bar);
            if (all_bar.exists())
              rez.serialize(owner_shard);
          }
          else
          {
            const ApUserEvent done = Runtime::create_ap_user_event(&trace_info);
            rez.serialize(done);
            broadcast_events.push_back(done);
            ApUserEvent all;
            if (all_done.exists())
            {
              all = Runtime::create_ap_user_event(&trace_info);
              all_done_events.push_back(all);
            }
            rez.serialize(all);
          }
          rez.serialize(origin);
        }
        runtime->send_collective_distribute_broadcast(origin, rez);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      // Then do our local broadcast
      // TODO: if the number of local instances is large then we could
      // turn this into a tree broadcast, but for now we're just going
      // to copy everything out of the first instance
      for (unsigned idx = 1; idx < instances.size(); idx++)
      {
        std::vector<CopySrcDstField> dst_fields;
        layout->compute_copy_offsets(copy_mask, instances[idx],
#ifdef LEGION_SPY
                                     instance_events[idx],
#endif
                                     local_fields);
        const std::vector<Reservation> no_reservations;
        const ApEvent local_copy = copy_expression->issue_copy(
            op, trace_info, dst_fields, local_fields, no_reservations,
#ifdef LEGION_SPY
            tree_id, tree_id,
#endif
            reduced, predicate_guard);
        if (local_copy.exists())
          broadcast_events.push_back(local_copy);
        if (trace_info.recording)
        {
          const UniqueInst dst_inst(dst_view, instance_points[idx]);
          trace_info.record_copy_insts(local_copy, copy_expression, local_inst,
                 local_inst, copy_mask, copy_mask, 0/*redop*/, applied_events);
        }
      }
      if (!broadcast_events.empty())
      {
        // Broadcast events will dominated the reduced event so there
        // is no need to include it specifically
        const ApEvent broadcast_done =
          Runtime::merge_events(&trace_info, broadcast_events);
        if (broadcast_done.exists())
        {
          dst_view->add_copy_user(false/*reading*/, 0/*redop*/, broadcast_done,
              trace_info.get_collect_event(), copy_mask, copy_expression,
              op_id, index, recorded_events, trace_info.recording,
              runtime->address_space);
          if (all_done.exists())
            all_done_events.push_back(broadcast_done);
        }
      }
      else 
      {
        dst_view->add_copy_user(false/*reading*/, 0/*redop*/, reduced,
            trace_info.get_collect_event(), copy_mask, copy_expression,
            op_id, index, recorded_events, trace_info.recording,
            runtime->address_space);
        if (all_done.exists())
          all_done_events.push_back(reduced);
      }
      if (all_done.exists())
      {
        if (all_bar.exists())
        {
          ApEvent arrival;
          if (!all_done_events.empty())
            arrival = Runtime::merge_events(&trace_info, all_done_events);
          Runtime::phase_barrier_arrive(all_bar, 1/*count*/, arrival);
          trace_info.record_barrier_arrival(all_bar, arrival, 1/*count*/,
                                            applied_events, owner_shard);
          Runtime::trigger_event(&trace_info, all_done, all_bar);
        }
        else
        {
          if (!all_done_events.empty())
            Runtime::trigger_event(&trace_info, all_done,
                Runtime::merge_events(&trace_info, all_done_events));
          else
            Runtime::trigger_event(&trace_info, all_done);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_distribute_hourglass(
                   Runtime *runtime, AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent dst_man_ready, src_man_ready, dst_view_ready, src_view_ready;
      CollectiveManager *target = static_cast<CollectiveManager*>(
          runtime->find_or_request_instance_manager(did, dst_man_ready));
      derez.deserialize(did);
      CollectiveManager *manager = static_cast<CollectiveManager*>(
          runtime->find_or_request_instance_manager(did, src_man_ready));
      derez.deserialize(did);
      InstanceView *src_view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(did, src_view_ready));
      derez.deserialize(did);
      InstanceView *dst_view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(did, dst_view_ready));
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      IndexSpaceExpression *copy_expression =
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      std::set<RtEvent> ready_events;
      Operation *op =
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events); 
      unsigned index;
      derez.deserialize(index);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      DomainPoint src_point;
      derez.deserialize(src_point);
      std::set<RtEvent> recorded_events, applied_events;
      PhysicalTraceInfo trace_info =
        PhysicalTraceInfo::unpack_trace_info(derez, runtime);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApUserEvent all_done;
      derez.deserialize(all_done);
      bool copy_restricted;
      derez.deserialize<bool>(copy_restricted);

      if (dst_man_ready.exists() && !dst_man_ready.has_triggered())
        ready_events.insert(dst_man_ready);
      if (src_man_ready.exists() && !src_man_ready.has_triggered())
        ready_events.insert(src_man_ready);
      if (src_view_ready.exists() && !src_view_ready.has_triggered())
        ready_events.insert(src_view_ready);
      if (dst_view_ready.exists() && !dst_view_ready.has_triggered())
        ready_events.insert(dst_view_ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      target->perform_collective_hourglass(manager, src_view, dst_view,
          precondition, predicate_guard, copy_expression, op, index,
          copy_mask, src_point, trace_info, recorded_events, applied_events,
          all_done, runtime->address_space, copy_restricted);

      if (!recorded_events.empty())
        Runtime::trigger_event(recorded,Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      delete op;
    }

    //--------------------------------------------------------------------------
    ApEvent CollectiveManager::perform_hammer_reduction(InstanceView *src_view,
                                const std::vector<CopySrcDstField> &dst_fields,
                                const std::vector<Reservation> &reservations,
                                ApEvent precondition,
                                PredEvent predicate_guard,
                                IndexSpaceExpression *copy_expression,
                                Operation *op, const unsigned index,
                                const FieldMask &copy_mask,
                                const FieldMask &dst_mask,
                                const UniqueInst &dst_inst,
                                const PhysicalTraceInfo &trace_info,
                                std::set<RtEvent> &recorded_events,
                                std::set<RtEvent> &applied_events,
                                AddressSpaceID origin)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(redop > 0);
      assert(op != NULL);
      assert(!instances.empty());
      assert(src_view->manager == this);
      assert(collective_mapping != NULL);
      assert(collective_mapping->contains(local_space));
#endif
      // Distribute out to the other nodes first
      std::vector<ApEvent> done_events;
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(origin, local_space, children);
      ApBarrier trace_barrier;
      ShardID trace_shard = 0;
      for (std::vector<AddressSpaceID>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(this->did);
          rez.serialize(src_view->did);
          pack_fields(rez, dst_fields);
          rez.serialize<size_t>(reservations.size());
          for (unsigned idx = 0; idx < reservations.size(); idx++)
            rez.serialize(reservations[idx]);
          rez.serialize(precondition);
          rez.serialize(predicate_guard);
          copy_expression->pack_expression(rez, *it);
          op->pack_remote_operation(rez, *it, applied_events); 
          rez.serialize(index);
          rez.serialize(copy_mask);
          rez.serialize(dst_mask);
          dst_inst.serialize(rez);
          trace_info.pack_trace_info(rez, applied_events);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (trace_info.recording)
          {
            if (!trace_barrier.exists())
            {
              trace_barrier =
                ApBarrier(Realm::Barrier::create_barrier(children.size()));
              trace_shard = trace_info.record_managed_barrier(trace_barrier,
                                                              children.size());
              done_events.push_back(trace_barrier);
            }
            rez.serialize(trace_barrier);
            rez.serialize(trace_shard);
          }
          else
          {
            const ApUserEvent done = Runtime::create_ap_user_event(&trace_info);
            rez.serialize(done);
            done_events.push_back(done);
          }
          rez.serialize(origin);
        }
        runtime->send_collective_hammer_reduction(*it, rez);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      const UniqueID op_id = op->get_unique_op_id();
      // Now we can perform our reduction copies to the destination
      // Get the source precondition for the copy
      const ApEvent src_pre = src_view->find_copy_preconditions(
          true/*reading*/, 0/*redop*/, copy_mask, copy_expression,
          op_id, index, applied_events, trace_info);
      if (src_pre.exists())
      {
        if (precondition.exists())
          precondition =
            Runtime::merge_events(&trace_info, precondition, src_pre);
        else
          precondition = src_pre;
      }
      // Issue the copies
      std::vector<ApEvent> local_events;
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        std::vector<CopySrcDstField> src_fields;
        layout->compute_copy_offsets(copy_mask, instances[idx],
#ifdef LEGION_SPY
                                     instance_events[idx],
#endif
                                     src_fields);
        const ApEvent copy_post = copy_expression->issue_copy(
            op, trace_info, dst_fields, src_fields, reservations,
#ifdef LEGION_SPY
            tree_id, tree_id,
#endif
            precondition, predicate_guard);
        if (copy_post.exists())
          local_events.push_back(copy_post);
        if (trace_info.recording)
        {
          const UniqueInst src_inst(src_view, instance_points[idx]);
          trace_info.record_copy_insts(copy_post, copy_expression, src_inst,
                      dst_inst, copy_mask, dst_mask, redop, applied_events);
        }
      }
      // Record the copy completion event
      if (!local_events.empty())
      {
        ApEvent local_done = Runtime::merge_events(&trace_info, local_events);
        if (local_done.exists())
        {
          const RtEvent collect_event = trace_info.get_collect_event();
          src_view->add_copy_user(true/*reading*/, 0/*redop*/, local_done,
              collect_event, copy_mask, copy_expression, op_id, index,
              recorded_events, trace_info.recording, runtime->address_space);
          done_events.push_back(local_done);
        }
      }
      // Merge the done events together
      if (done_events.empty())
        return ApEvent::NO_AP_EVENT;
      return Runtime::merge_events(&trace_info, done_events);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_hammer_reduction(
                   Runtime *runtime, AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID man_did, src_did;
      derez.deserialize(man_did);
      RtEvent man_ready, src_ready;
      CollectiveManager *manager = static_cast<CollectiveManager*>(
          runtime->find_or_request_instance_manager(man_did, man_ready));
      derez.deserialize(src_did);
      InstanceView *src_view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(src_did, src_ready));
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<CopySrcDstField> dst_fields(num_fields);
      std::set<RtEvent> recorded_events, ready_events, applied_events;
      unpack_fields(dst_fields, derez, ready_events, manager,man_ready,runtime);
      size_t num_reservations;
      derez.deserialize(num_reservations);
      std::vector<Reservation> reservations(num_reservations);
      for (unsigned idx = 0; idx < num_reservations; idx++)
        derez.deserialize(reservations[idx]);
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      IndexSpaceExpression *copy_expression =
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      Operation *op =
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      FieldMask copy_mask, dst_mask;
      derez.deserialize(copy_mask);
      derez.deserialize(dst_mask);
      UniqueInst dst_inst;
      dst_inst.deserialize(derez);
      PhysicalTraceInfo trace_info =
        PhysicalTraceInfo::unpack_trace_info(derez, runtime);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApUserEvent ready;
      if (trace_info.recording)
      {
        ApBarrier bar;
        derez.deserialize(bar);
        ShardID sid;
        derez.deserialize(sid);
        ready = Runtime::create_ap_user_event(&trace_info);
        Runtime::phase_barrier_arrive(bar, 1/*count*/, ready);
        trace_info.record_barrier_arrival(bar, ready, 1/*count*/,
                                          applied_events, sid);
      }
      else
        derez.deserialize(ready);
      AddressSpaceID origin;
      derez.deserialize(origin);

      if (man_ready.exists() && !man_ready.has_triggered())
        ready_events.insert(man_ready);
      if (src_ready.exists() && !src_ready.has_triggered())
        ready_events.insert(src_ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      const ApEvent result = manager->perform_hammer_reduction(src_view,
          dst_fields, reservations, precondition, predicate_guard,
          copy_expression, op, index, copy_mask, dst_mask, dst_inst,
          trace_info, recorded_events, applied_events, origin);

      Runtime::trigger_event(&trace_info, ready, result);
      if (!recorded_events.empty())
        Runtime::trigger_event(recorded,Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      delete op;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::pack_fields(Serializer &rez,
                               const std::vector<CopySrcDstField> &fields) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(fields.size());
      for (unsigned idx = 0; idx < fields.size(); idx++)
        rez.serialize(fields[idx]);
      if (runtime->legion_spy_enabled)
      {
        // Pack the instance points for these instances so we can check to 
        // see if we already fetched them on the remote node
        for (std::vector<CopySrcDstField>::const_iterator it =
              fields.begin(); it != fields.end(); it++)
        {
          bool found = false;
          for (unsigned idx = 0; idx < instances.size(); idx++)
          {
            if (instances[idx] != it->inst)
              continue;
            rez.serialize(instance_events[idx]);
            rez.serialize(idx);
            rez.serialize(instance_points[idx]);
            found = true;
            break;
          }
          if (!found)
          {
            AutoLock i_lock(inst_lock,1,false/*exclusive*/);
            for (std::map<DomainPoint,RemoteInstInfo>::const_iterator rit =
                  remote_points.begin(); rit != remote_points.end(); rit++)
            {
              if (it->inst != rit->second.instance)
                continue;
              rez.serialize(rit->second.unique_event);
              rez.serialize(rit->second.index);
              rez.serialize(rit->first);
              found = true;
              break;
            }
#ifdef DEBUG_LEGION
            assert(found);
#endif
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::unpack_fields(
                std::vector<CopySrcDstField> &fields,
                Deserializer &derez, std::set<RtEvent> &ready_events,
                CollectiveManager *manager, RtEvent man_ready, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!fields.empty());
#endif
      const Processor local_proc = Processor::get_executing_processor();
      for (unsigned idx = 0; idx < fields.size(); idx++)
      {
        CopySrcDstField &field = fields[idx];
        derez.deserialize(field);
        // Check to see if we fetched the metadata for this instance
        RtEvent ready(field.inst.fetch_metadata(local_proc));
        if (ready.exists() && !ready.has_triggered())
          ready_events.insert(ready);
      }
      if (runtime->legion_spy_enabled)
      {
        // Legion Spy is a bit dumb currently and needs to have logged every
        // instance on every node where it might be used currently, so check
        // to make sure we've logged it
        std::vector<unsigned> indexes(fields.size());
        std::vector<DomainPoint> points(fields.size());
        std::vector<ApEvent> events(fields.size());
        for (unsigned idx = 0; idx < fields.size(); idx++)
        {
          derez.deserialize(events[idx]);
          if (!events[idx].exists())
          {
#ifdef DEBUG_LEGION
            assert(idx == 0); // should only happen on the first iteration
#endif
            // These fields are from an individual manager so just
            // load a copy of it here
            DistributedID did;
            derez.deserialize(did);
            RtEvent ready;
            runtime->find_or_request_instance_manager(did, ready);
            if (ready.exists())
              ready_events.insert(ready);
            return;
          }
          derez.deserialize(indexes[idx]);
          derez.deserialize(points[idx]);
        }
        if (man_ready.exists() && !man_ready.has_triggered())
          man_ready.wait();
        manager->log_remote_point_instances(fields, indexes, points, events);
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::log_remote_point_instances(
                                    const std::vector<CopySrcDstField> &fields,
                                    const std::vector<unsigned> &indexes,
                                    const std::vector<DomainPoint> &points,
                                    const std::vector<ApEvent> &events)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
      for (unsigned idx = 0; idx < points.size(); idx++)
      {
        // Skip anything that we already logged
        if (remote_points.find(points[idx]) != remote_points.end())
          continue;
        RemoteInstInfo &info = remote_points[points[idx]];
        info.instance = fields[idx].inst;
        info.unique_event = events[idx];
        info.index = indexes[idx];
        LegionSpy::log_physical_instance(info.unique_event, info.instance.id,
            info.instance.get_location().id, instance_domain->expr_id,
            field_space_node->handle, tree_id, redop);
        layout->log_instance_layout(info.unique_event);
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::compute_copy_offsets(const FieldMask &copy_mask,
      std::vector<CopySrcDstField> &fields, const DomainPoint *collective_point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(layout != NULL);
      assert(collective_point != NULL);
#endif
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        if (instance_points[idx] != *collective_point)
          continue;
        layout->compute_copy_offsets(copy_mask, instances[idx],
#ifdef LEGION_SPY
                                     instance_events[idx],
#endif
                                     fields);
        return;
      }
      // We should never get here because the instance should always be local
      assert(false);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_remote_registration(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      CollectiveManager *manager = static_cast<CollectiveManager*>(
                          runtime->find_distributed_collectable(did));
      DistributedID view_did;
      derez.deserialize(view_did);
      RtEvent ready;
      InstanceView *view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(view_did, ready));
      RegionUsage usage;
      derez.deserialize(usage);
      FieldMask user_mask;
      derez.deserialize(user_mask);
      IndexSpace handle;
      derez.deserialize(handle);
      IndexSpaceNode *expr = runtime->forest->get_node(handle);
      UniqueID op_id;
      derez.deserialize(op_id);
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      unsigned index;
      derez.deserialize(index);
      ApEvent term_event;
      derez.deserialize(term_event);
      RtEvent collect_event;
      derez.deserialize(collect_event);
      RtUserEvent applied;
      derez.deserialize(applied);
      const PhysicalTraceInfo trace_info =
        PhysicalTraceInfo::unpack_trace_info(derez, runtime);
      bool symbolic;
      derez.deserialize(symbolic);
      ApUserEvent result;
      derez.deserialize(result);

      std::set<RtEvent> applied_events;
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      Runtime::trigger_event(&trace_info, result,
          manager->register_collective_user(view, usage, user_mask, expr,
            op_id, op_ctx_index, index, term_event, collect_event,
            applied_events, manager->collective_mapping, NULL/*no op*/,
            trace_info, symbolic));
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
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
                                         Operation *local_collective_op,
                                         const PhysicalTraceInfo &trace_info,
                                         const bool symbolic)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapping != NULL);
      assert(collective_mapping != NULL);
      // CollectiveMapping for the analyses should all align with 
      // the CollectiveMapping for the collective manager
      assert((mapping == collective_mapping) ||
          ((*mapping) == (*collective_mapping)));
      assert(term_event.exists());
#endif
      // Check to make sure we're on the right node for this point
      if (local_collective_op != NULL)
      {
        bool local = false;
        const DomainPoint point =
          local_collective_op->get_collective_instance_point();
        for (unsigned idx = 0; instance_points.size(); idx++)
        {
          if (instance_points[idx] != point)
            continue;
          local = true;
          break;
        }
        if (!local)
        {
          // Figure out which node is local
          const PhysicalInstance inst = get_instance(point);
          const AddressSpaceID target = inst.address_space();
#ifdef DEBUG_LEGION
          assert(collective_mapping->contains(target));
#endif
          const ApUserEvent result = Runtime::create_ap_user_event(&trace_info);
          const RtUserEvent applied = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(view->did);
            rez.serialize(usage);
            rez.serialize(user_mask);
            rez.serialize(expr->handle);
            rez.serialize(op_id);
            rez.serialize(op_ctx_index);
            rez.serialize(index);
            rez.serialize(term_event);
            rez.serialize(collect_event);
            rez.serialize(applied);
            trace_info.pack_trace_info(rez, applied_events);
            rez.serialize<bool>(symbolic);
            rez.serialize(result);
          }
          runtime->send_collective_remote_registration(target, rez);
          return result;
        }
      }
#ifdef DEBUG_LEGION
      assert(collective_mapping->contains(local_space));
#endif
      // We performing a collective analysis, this function performs a 
      // parallel rendezvous to ensure several important invariants.
      // 1. SUBTLE!!! Make sure that all the participants have arrived
      //    at this function before performing any view analysis. This
      //    is required to ensure that any copies that need to be issued
      //    have had a chance to record their view users first before we
      //    attempt to look for preconditions for this user.
      // 2. Similarly make sure that the applied events reflects the case
      //    where all the users have been recorded across the views on 
      //    each node to ensure that any downstream copies or users will
      //    observe all the most recent users.
      // 3. Deduplicate across all the participants on the same node since
      //    there is always just a single view on each node. This function
      //    call will always return the local user precondition for the
      //    local instances. Make sure to merge together all the partcipant
      //    postconditions for the local instances to reflect in the view
      //    that the local instances are ready when they are all ready.
      // 4. Do NOT block in this function call or you can risk deadlock because
      //    we might be doing several of these calls for a region requirement
      //    on different instances and the orders might vary on each node.
      
      // The unique tag for the rendezvous is our context ID which will be
      // the same across all points and the index of our region requirement
      ApUserEvent result;
      PhysicalTraceInfo *result_info;
      RtUserEvent local_registered, global_registered;
      std::vector<RtEvent> remote_registered;
      std::vector<ApEvent> local_term_events;
      std::vector<CollectiveCopyFillAnalysis*> analyses;
      const RendezvousKey key(view->did, op_ctx_index, index);
      {
        AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
        assert(!instances.empty());
#endif
        // Check to see if we're the first one to arrive on this node
        std::map<RendezvousKey,UserRendezvous>::iterator finder =
          rendezvous_users.find(key);
        if (finder == rendezvous_users.end())
        {
          // If we are then make the record for knowing when we've seen
          // all the expected arrivals
          finder = rendezvous_users.insert(
              std::make_pair(key,UserRendezvous())).first; 
          UserRendezvous &rendezvous = finder->second;
          // Count how many expected arrivals we have
          // If we're doing collective per space 
          rendezvous.remaining_local_arrivals = instances.size();
          rendezvous.local_initialized = true;
          rendezvous.remaining_remote_arrivals =
            mapping->count_children(owner_space, local_space);
          rendezvous.ready_event = Runtime::create_ap_user_event(&trace_info);
          rendezvous.trace_info = new PhysicalTraceInfo(trace_info);
          rendezvous.local_registered = Runtime::create_rt_user_event();
          rendezvous.global_registered = Runtime::create_rt_user_event();
        }
        else if (!finder->second.local_initialized)
        {
          // First local arrival, but rendezvous was made by a remote
          // arrival so we need to make the ready event
#ifdef DEBUG_LEGION
          assert(!finder->second.ready_event.exists());
          assert(finder->second.trace_info == NULL);
#endif
          finder->second.ready_event =
            Runtime::create_ap_user_event(&trace_info);
          finder->second.trace_info = new PhysicalTraceInfo(trace_info);
          finder->second.remaining_local_arrivals = instances.size();
          finder->second.local_initialized = true;
        }
        if (term_event.exists())
          finder->second.local_term_events.push_back(term_event);
        // Record the applied events
        applied_events.insert(finder->second.global_registered);
        // The result will be the ready event
        result = finder->second.ready_event;
        result_info = finder->second.trace_info;
#ifdef DEBUG_LEGION
        assert(finder->second.local_initialized);
        assert(finder->second.remaining_local_arrivals > 0);
#endif
        // See if we've seen all the arrivals
        if (--finder->second.remaining_local_arrivals == 0)
        {
          // If we're going to need to defer this then save
          // all of our local state needed to perform registration
          // for when it is safe to do so
          if (!is_owner() || 
              (finder->second.remaining_remote_arrivals > 0))
          {
            // Save the state that we need for finalization later
            finder->second.view = view;
            finder->second.usage = usage;
            finder->second.mask = new FieldMask(user_mask);
            finder->second.expr = expr;
            WrapperReferenceMutator mutator(applied_events);
            expr->add_nested_expression_reference(did, &mutator);
            finder->second.op_id = op_id;
            finder->second.collect_event = collect_event;
            finder->second.symbolic = symbolic;
          }
          if (finder->second.remaining_remote_arrivals == 0)
          {
            if (!is_owner())
            {
              // Not the owner so send the message to the parent
              RtEvent registered = finder->second.local_registered;
              if (!finder->second.remote_registered.empty())
              {
                finder->second.remote_registered.push_back(registered);
                registered =
                  Runtime::merge_events(finder->second.remote_registered);
              }
              const AddressSpaceID parent = 
                collective_mapping->get_parent(owner_space, local_space);
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
                rez.serialize(view->did);
                rez.serialize(op_ctx_index);
                rez.serialize(index);
                rez.serialize(registered);
              }
              runtime->send_collective_register_user_request(parent, rez);
              return result;
            }
            else
            {
              // We're going to fall through so grab the state
              // that we need to do the finalization now
              remote_registered.swap(finder->second.remote_registered);
              local_registered = finder->second.local_registered;
              global_registered = finder->second.global_registered;
              local_term_events.swap(finder->second.local_term_events);
              analyses.swap(finder->second.analyses);
              // We can erase this from the data structure now
              rendezvous_users.erase(finder);
            }
          }
          else // Still waiting for remote arrivals
            return result;
        }
        else // Not the last local arrival so we can just return the result
          return result;
      }
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      ApEvent local_term; 
      if (!local_term_events.empty())
        local_term = Runtime::merge_events(&trace_info, local_term_events);
      finalize_collective_user(view, usage, user_mask, expr, op_id,
         op_ctx_index, index, collect_event, local_registered,
         global_registered, result, local_term, result_info, analyses,symbolic);
      RtEvent all_registered = local_registered;
      if (!remote_registered.empty())
      {
        remote_registered.push_back(all_registered);
        all_registered = Runtime::merge_events(remote_registered);
      }
      Runtime::trigger_event(global_registered, all_registered); 
      return result;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::process_register_user_request(
                        const DistributedID view_did, const size_t op_ctx_index,
                        const unsigned index, RtEvent registered)
    //--------------------------------------------------------------------------
    {
      UserRendezvous to_perform;
      const RendezvousKey key(view_did, op_ctx_index, index);
      {
        AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
        assert(!instances.empty());
#endif
        // Check to see if we're the first one to arrive on this node
        std::map<RendezvousKey,UserRendezvous>::iterator
          finder = rendezvous_users.find(key);
        if (finder == rendezvous_users.end())
        {
          // If we are then make the record for knowing when we've seen
          // all the expected arrivals
          finder = rendezvous_users.insert(
              std::make_pair(key,UserRendezvous())).first; 
          UserRendezvous &rendezvous = finder->second;
          rendezvous.local_initialized = false;
          rendezvous.remaining_remote_arrivals =
            collective_mapping->count_children(owner_space, local_space);
          rendezvous.local_registered = Runtime::create_rt_user_event();
          rendezvous.global_registered = Runtime::create_rt_user_event();
        }
        finder->second.remote_registered.push_back(registered);
#ifdef DEBUG_LEGION
        assert(finder->second.remaining_remote_arrivals > 0);
#endif
        // If we're not the last arrival then we're done
        if ((--finder->second.remaining_remote_arrivals > 0) ||
            !finder->second.local_initialized ||
            (finder->second.remaining_local_arrivals > 0))
          return;
        if (!is_owner())
        {
          // Continue sending the message up the tree to the parent
          registered = finder->second.local_registered;
          if (!finder->second.remote_registered.empty())
          {
            finder->second.remote_registered.push_back(registered);
            registered =
              Runtime::merge_events(finder->second.remote_registered);
          }
          const AddressSpaceID parent = 
            collective_mapping->get_parent(owner_space, local_space);
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(view_did);
            rez.serialize(op_ctx_index);
            rez.serialize(index);
            rez.serialize(registered);
          }
          runtime->send_collective_register_user_request(parent, rez);
          return;
        }
        // We're the owner so we can start doing the user registration
        // Grab everything we need to call finalize_collective_user
        to_perform = std::move(finder->second);
        // Then we can erase the entry
        rendezvous_users.erase(finder);
      }
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      ApEvent local_term; 
      if (!to_perform.local_term_events.empty())
        local_term = Runtime::merge_events(to_perform.trace_info,
                                    to_perform.local_term_events);
      finalize_collective_user(to_perform.view, to_perform.usage,
          *(to_perform.mask), to_perform.expr, to_perform.op_id,
          op_ctx_index, index, to_perform.collect_event,
          to_perform.local_registered, to_perform.global_registered,
          to_perform.ready_event, local_term, to_perform.trace_info,
          to_perform.analyses, to_perform.symbolic);
      RtEvent all_registered = to_perform.local_registered;
      if (!to_perform.remote_registered.empty())
      {
        to_perform.remote_registered.push_back(all_registered);
        all_registered = Runtime::merge_events(to_perform.remote_registered);
      }
      Runtime::trigger_event(to_perform.global_registered, all_registered);
      if (to_perform.expr->remove_nested_expression_reference(did))
        delete to_perform.expr;
      delete to_perform.mask;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_register_user_request(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      CollectiveManager *manager = static_cast<CollectiveManager*>(
              runtime->find_or_request_instance_manager(did, ready));
      DistributedID view_did;
      derez.deserialize(view_did);
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      unsigned index;
      derez.deserialize(index);
      RtEvent registered;
      derez.deserialize(registered);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      manager->process_register_user_request(view_did, op_ctx_index, 
                                             index, registered);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::process_register_user_response(
                        const DistributedID view_did, const size_t op_ctx_index,
                        const unsigned index, RtEvent registered)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner());
#endif
      UserRendezvous to_perform;
      const RendezvousKey key(view_did, op_ctx_index, index);
      {
        AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
        assert(!instances.empty());
#endif
        // Check to see if we're the first one to arrive on this node
        std::map<RendezvousKey,UserRendezvous>::iterator finder =
          rendezvous_users.find(key);
#ifdef DEBUG_LEGION
        assert(finder != rendezvous_users.end());
#endif
        to_perform = std::move(finder->second);
        // Can now remove this from the data structure
        rendezvous_users.erase(finder);
      }
      // Now we can perform the user registration
      ApEvent local_term; 
      if (!to_perform.local_term_events.empty())
        local_term = Runtime::merge_events(to_perform.trace_info,
                                    to_perform.local_term_events);
      finalize_collective_user(to_perform.view, to_perform.usage,
          *(to_perform.mask), to_perform.expr, to_perform.op_id,
          op_ctx_index, index, to_perform.collect_event,
          to_perform.local_registered, to_perform.global_registered,
          to_perform.ready_event, local_term, to_perform.trace_info,
          to_perform.analyses, to_perform.symbolic);
      Runtime::trigger_event(to_perform.global_registered, registered);
      if (to_perform.expr->remove_nested_expression_reference(did))
        delete to_perform.expr;
      delete to_perform.mask;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_register_user_response(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      CollectiveManager *manager = static_cast<CollectiveManager*>(
              runtime->find_or_request_instance_manager(did, ready));
      DistributedID view_did;
      derez.deserialize(view_did);
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      unsigned index;
      derez.deserialize(index);
      RtEvent registered;
      derez.deserialize(registered);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      manager->process_register_user_response(view_did, op_ctx_index,
                                              index, registered);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::find_points_in_memory(Memory memory,
                                         std::vector<DomainPoint> &points) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(collective_mapping != NULL);
#endif
      const AddressSpaceID space = memory.address_space();
      if (space != local_space)
      {
        if (!collective_mapping->contains(space))
          return;
        const RtUserEvent ready_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(memory);
          rez.serialize(&points);
          rez.serialize(ready_event);
        }
        runtime->send_collective_find_points_request(space, rez);
        if (!ready_event.has_triggered())
          ready_event.wait();
      }
      else
      {
        for (unsigned idx = 0; idx < memories.size(); idx++)
          if (memories[idx]->memory == memory)
            points.push_back(instance_points[idx]);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_find_points_request(
                   Runtime *runtime, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      Memory memory;
      derez.deserialize(memory);
      std::vector<DomainPoint> *target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);

      CollectiveManager *manager = static_cast<CollectiveManager*>(
          runtime->weak_find_distributed_collectable(did));
      if (manager != NULL)
      {
        std::vector<DomainPoint> results;
        manager->find_points_in_memory(memory, results);
        if (!results.empty())
        {
          Serializer rez;
          {
            RezCheck z2(rez);
            rez.serialize(target);
            rez.serialize<size_t>(results.size());
            for (unsigned idx = 0; idx < results.size(); idx++)
              rez.serialize(results[idx]);
            rez.serialize(done);
          }
          runtime->send_collective_find_points_response(source, rez);
          if (manager->remove_base_resource_ref(RUNTIME_REF))
            delete manager;
          return;
        }
        else if (manager->remove_base_resource_ref(RUNTIME_REF))
          delete manager;
      }
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_find_points_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::vector<DomainPoint> *target;
      derez.deserialize(target);
      size_t num_points;
      derez.deserialize(num_points);
      target->resize(num_points);
      for (unsigned idx = 0; idx < num_points; idx++)
        derez.deserialize((*target)[idx]);
      RtUserEvent done;
      derez.deserialize(done);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::find_points_nearest_memory(Memory memory,
                     std::map<DomainPoint,Memory> &points, bool bandwidth) const
    //--------------------------------------------------------------------------
    {
      std::atomic<size_t> best(bandwidth ? 0 : GUARD_SIZE);
      const AddressSpaceID origin = select_origin_space();
      const RtEvent ready = find_points_nearest_memory(memory, local_space,
          &points, &best, origin, bandwidth ? 0 : GUARD_SIZE, bandwidth);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
    }

    //--------------------------------------------------------------------------
    RtEvent CollectiveManager::find_points_nearest_memory(Memory memory,
                    AddressSpaceID source, std::map<DomainPoint,Memory> *points,
                    std::atomic<size_t> *target, AddressSpaceID origin, 
                    size_t best, bool bandwidth) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(collective_mapping != NULL);
#endif
      const AddressSpaceID space = memory.address_space();
      if (space != local_space)
      {
        if (collective_mapping->contains(space))
        {
#ifdef DEBUG_LEGION
          assert(source == local_space);
#endif
          // Assume that all memmories in the same space are always inherently
          // closer to the target memory than any others, so we can send the
          // request straight to that node and do the lookup
          const RtUserEvent done = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(memory);
            rez.serialize(source);
            rez.serialize(points);
            rez.serialize(target);
            rez.serialize(origin);
            rez.serialize(best);
            rez.serialize<bool>(bandwidth);
            rez.serialize(done);
          }
          runtime->send_collective_nearest_points_request(space, rez);
          return done;
        }
        else
        {
          if (collective_mapping->contains(local_space))
          {
            // Do our local check and update the best
            std::map<DomainPoint,Memory> local_results;
            find_nearest_local_points(memory, best, local_results, bandwidth);
            std::vector<RtEvent> done_events;
            std::vector<AddressSpaceID> children;
            collective_mapping->get_children(origin, local_space, children);
            for (std::vector<AddressSpaceID>::const_iterator it = 
                  children.begin(); it != children.end(); it++)
            {
              const RtUserEvent done = Runtime::create_rt_user_event();
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
                rez.serialize(memory);
                rez.serialize(source);
                rez.serialize(points);
                rez.serialize(target);
                rez.serialize(origin);
                rez.serialize(best);
                rez.serialize<bool>(bandwidth);
                rez.serialize(done);
              }
              runtime->send_collective_nearest_points_request(*it, rez);
              done_events.push_back(done);
            }
            if (!local_results.empty())
            {
              const RtUserEvent done = Runtime::create_rt_user_event();
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(points);
                rez.serialize(target);
                rez.serialize(best);
                rez.serialize<size_t>(local_results.size());
                for (std::map<DomainPoint,Memory>::const_iterator it =
                      local_results.begin(); it != local_results.end(); it++)
                {
                  rez.serialize(it->first);
                  rez.serialize(it->second);
                }
                rez.serialize<bool>(bandwidth);
                rez.serialize(done);
              }
              runtime->send_collective_nearest_points_response(source, rez);
              done_events.push_back(done);
            }
            if (!done_events.empty())
              return Runtime::merge_events(done_events);
          }
          else
          {
#ifdef DEBUG_LEGION
            assert(source == local_space);
#endif
            // Send to the origin to start
            const RtUserEvent done = Runtime::create_rt_user_event();
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(memory);
              rez.serialize(source);
              rez.serialize(points);
              rez.serialize(target);
              rez.serialize(origin);
              rez.serialize(best);
              rez.serialize<bool>(bandwidth);
              rez.serialize(done);
            }
            runtime->send_collective_nearest_points_request(origin, rez);
            return done;
          }
        }
      }
      else
      {
        // Assume that all memories in the same space are always inherently
        // closer to the target memory than any others
        // See if we find the memory itself
        std::map<DomainPoint,Memory> results;
        find_nearest_local_points(memory, best, results, bandwidth);
        if (source != local_space)
        {
          if (!results.empty())
          {
            const RtUserEvent done = Runtime::create_rt_user_event();
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(points);
              rez.serialize(target);
              rez.serialize(best);
              rez.serialize<size_t>(results.size());
              for (std::map<DomainPoint,Memory>::const_iterator it =
                    results.begin(); it != results.end(); it++)
              {
                rez.serialize(it->first);
                rez.serialize(it->second);
              }
              rez.serialize<bool>(bandwidth);
              rez.serialize(done);
            }
            runtime->send_collective_nearest_points_response(source, rez);
            return done;
          }
        }
        else
        {
          // This is the local case, so there's no atomicity required
          points->swap(results);
          target->store(best);
        }
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::find_nearest_local_points(Memory memory,
      size_t &best, std::map<DomainPoint,Memory> &results, bool bandwidth) const
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < memories.size(); idx++)
      {
        if (memories[idx]->memory == memory)
          results[instance_points[idx]] = memory;
      }
      if (results.empty())
      {
        // Nothing in the memory itself, so see which of our memories
        // are closer to anything else
        std::map<Memory,size_t> searches;
        for (unsigned idx = 0; idx < memories.size(); idx++)
        {
          const Memory local = memories[idx]->memory;
          std::map<Memory,size_t>::const_iterator finder =
            searches.find(local);
          if (finder == searches.end())
          {
            Realm::Machine::AffinityDetails affinity;
            if (runtime->machine.has_affinity(memory, local, &affinity))
            {
#ifdef DEBUG_LEGION
              assert(0 < affinity.bandwidth);
#ifndef __clang__ // Apparently all clangs are stupid about this
              assert(affinity.bandwidth < GUARD_SIZE);
#endif
#endif
              if (bandwidth)
              {
                searches[local] = affinity.bandwidth;
                if (affinity.bandwidth >= best)
                {
                  if (affinity.bandwidth > best)
                  {
                    results.clear();
                    best = affinity.bandwidth;
                  }
                  results[instance_points[idx]] = local;
                }
              }
              else
              {
#ifdef DEBUG_LEGION
                assert(0 < affinity.latency);
#ifndef __clang__ // Apparently all clangs are stupid about this
                assert(affinity.latency < GUARD_SIZE);
#endif
#endif
                searches[local] = affinity.latency;
                if (affinity.latency <= best)
                {
                  if (affinity.latency < best)
                  {
                    results.clear();
                    best = affinity.latency;
                  }
                  results[instance_points[idx]] = local;
                }
              }
            }
            else
              searches[local] = bandwidth ? 0 : GUARD_SIZE;
          }
          else if (finder->second == best)
            results[instance_points[idx]] = local;
        }
      }
      else
        best = bandwidth ? GUARD_SIZE-1 : 1;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_nearest_points_request(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      Memory memory;
      derez.deserialize(memory);
      AddressSpaceID source;
      derez.deserialize(source);
      std::map<DomainPoint,Memory> *points;
      derez.deserialize(points);
      std::atomic<size_t> *target;
      derez.deserialize(target);
      AddressSpaceID origin;
      derez.deserialize(origin);
      size_t best;
      derez.deserialize(best);
      bool bandwidth;
      derez.deserialize(bandwidth);
      RtUserEvent done;
      derez.deserialize(done);

      CollectiveManager *manager = static_cast<CollectiveManager*>(
          runtime->weak_find_distributed_collectable(did));
      if (manager != NULL)     
      {
        Runtime::trigger_event(done, manager->find_points_nearest_memory(
              memory, source, points, target, origin, best, bandwidth));
        if (manager->remove_base_resource_ref(RUNTIME_REF))
          delete manager;
      }
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_nearest_points_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::map<DomainPoint,Memory> *points;
      derez.deserialize(points);
      std::atomic<size_t> *target;
      derez.deserialize(target);
      size_t best;
      derez.deserialize(best);
      size_t num_points;
      derez.deserialize(num_points);
      std::vector<std::pair<DomainPoint,Memory> > results(num_points);
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        derez.deserialize(results[idx].first);
        derez.deserialize(results[idx].second);
      }
      bool bandwidth;
      derez.deserialize(bandwidth);
      // spin until we can get safely set the guard to add our entries
      const size_t guard = bandwidth ? GUARD_SIZE : 0;
      size_t current = target->load();
      while ((current == guard) ||
             (bandwidth && (current <= best)) ||
             (!bandwidth && (best <= current)))
      {
        if (!target->compare_exchange_weak(current, guard))
          continue;
        // If someone else still holds the guard then keep trying
        if (current == guard)
          continue;
        if (bandwidth)
        {
          if (current < best)
            points->clear();
          for (unsigned idx = 0; idx < results.size(); idx++)
            points->insert(results[idx]);
        }
        else
        {
          if (best < current)
            points->clear();
          for (unsigned idx = 0; idx < results.size(); idx++)
            points->insert(results[idx]);
        }
        target->store(best);
        break;
      }
      RtUserEvent done;
      derez.deserialize(done);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID CollectiveManager::select_source_space(
                                               AddressSpaceID destination) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(collective_mapping != NULL);
#endif
      // 1. If the collective manager has instances on the same node
      //    as the destination then we'll use one of them
      if (collective_mapping->contains(destination))
        return destination;
      // 2. If the collective manager has instances on the local node
      //    then we'll use one of them
      if (collective_mapping->contains(local_space))
        return local_space;
      // 3. Pick the node closest to the destination in the collective
      //    manager and use that to issue copies
      return collective_mapping->find_nearest(destination);
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::register_collective_analysis(DistributedID view_did,
                                           CollectiveCopyFillAnalysis *analysis)
    //--------------------------------------------------------------------------
    {
      int index = -1;
      // Figure out which index we are, if we are not local we can ignore it
      for (unsigned idx = 0; idx < instance_points.size(); idx++)
      {
        if (instance_points[idx] != analysis->collective_point)
          continue;
        index = idx;
        break;
      }
      if (index < 0)
        return;
      const RendezvousKey key(view_did,analysis->context_index,analysis->index);
      AutoLock i_lock(inst_lock);
      std::map<RendezvousKey,UserRendezvous>::iterator finder =
        rendezvous_users.find(key);
      if (finder == rendezvous_users.end())
      {
        finder = rendezvous_users.insert(
            std::make_pair(key,UserRendezvous())).first; 
        UserRendezvous &rendezvous = finder->second;
        // Count how many expected arrivals we have
        rendezvous.local_initialized = false;
        rendezvous.remaining_remote_arrivals =
          collective_mapping->count_children(owner_space, local_space);
        rendezvous.local_registered = Runtime::create_rt_user_event();
        rendezvous.global_registered = Runtime::create_rt_user_event();
      }
      // Perform the registration
      if (finder->second.analyses.empty())
        finder->second.analyses.resize(instances.size(), NULL);
#ifdef DEBUG_LEGION
      assert(unsigned(index) < finder->second.analyses.size());
      assert(finder->second.analyses[index] == NULL);
      assert(finder->second.valid_analyses < instances.size());
#endif
      finder->second.analyses[index] = analysis;
      analysis->add_reference();
      if ((++finder->second.valid_analyses == instances.size()) &&
          finder->second.analyses_ready.exists())
        Runtime::trigger_event(finder->second.analyses_ready);
    }

    //--------------------------------------------------------------------------
    RtEvent CollectiveManager::find_collective_analyses(DistributedID view_did,
                      size_t context_index, unsigned index, 
                      const std::vector<CollectiveCopyFillAnalysis*> *&analyses)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!instances.empty());
      assert(collective_mapping != NULL);
#endif
      const RendezvousKey key(view_did, context_index, index);
      AutoLock i_lock(inst_lock);
      std::map<RendezvousKey,UserRendezvous>::iterator finder =
        rendezvous_users.find(key);
      if (finder == rendezvous_users.end())
      {
        finder = rendezvous_users.insert(
            std::make_pair(key,UserRendezvous())).first; 
        UserRendezvous &rendezvous = finder->second;
        rendezvous.local_initialized = false;
        rendezvous.remaining_remote_arrivals =
          collective_mapping->count_children(owner_space, local_space);
        rendezvous.local_registered = Runtime::create_rt_user_event();
        rendezvous.global_registered = Runtime::create_rt_user_event();
      }
      analyses = &finder->second.analyses;
#ifdef DEBUG_LEGION
      assert(finder->second.valid_analyses <= instances.size());
#endif
      if ((finder->second.valid_analyses < instances.size()) &&
          !finder->second.analyses_ready.exists())
        finder->second.analyses_ready = Runtime::create_rt_user_event();
      return finder->second.analyses_ready;
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
                            RtEvent collect_event,
                            RtUserEvent local_registered,
                            RtEvent global_registered,
                            ApUserEvent ready_event,
                            ApEvent term_event,
                            const PhysicalTraceInfo *trace_info,
                            std::vector<CollectiveCopyFillAnalysis*> &analyses,
                            const bool symbolic) const
    //--------------------------------------------------------------------------
    {
      // First send out any messages to the children so they can start
      // their own registrations
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(owner_space, local_space, children);
      if (!children.empty())
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(view->did);
          rez.serialize(op_ctx_index);
          rez.serialize(index);
          rez.serialize(global_registered);
        }
        for (std::vector<AddressSpaceID>::const_iterator it =
              children.begin(); it != children.end(); it++)
          runtime->send_collective_register_user_response(*it, rez);
      }
      // Perform the local registration on the view
      std::set<RtEvent> registered_events;
      const ApEvent ready = view->register_user(usage, user_mask, expr, op_id,
          op_ctx_index, index, term_event, collect_event, registered_events,
          NULL/*collective mapping*/, NULL/*no collective op*/, *trace_info,
          runtime->address_space, symbolic);
      Runtime::trigger_event(trace_info, ready_event, ready);
      if (!registered_events.empty())
        Runtime::trigger_event(local_registered,
            Runtime::merge_events(registered_events));
      else
        Runtime::trigger_event(local_registered);
      // Remove any references on the analyses
      for (std::vector<CollectiveCopyFillAnalysis*>::const_iterator it =
            analyses.begin(); it != analyses.end(); it++)
        if ((*it)->remove_reference())
          delete (*it);
      delete trace_info;
    }

    //--------------------------------------------------------------------------
    RtEvent CollectiveManager::find_field_reservations(const FieldMask &mask,
                                DistributedID view_did,const DomainPoint &point,
                                std::vector<Reservation> *reservations,
                                AddressSpaceID source, RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(point.get_dim() > 0);
      assert((dense_points.get_dim() == 0) || dense_points.contains(point));
#endif
      std::vector<Reservation> results;
      // Check to see if it is a local point or not
      const bool is_local = 
        (std::find(instance_points.begin(), instance_points.end(), point) !=
         instance_points.end());
      const std::pair<DistributedID,DomainPoint> key(view_did, point);
      if (is_local)
      {
        results.reserve(mask.pop_count());
        // We're the owner so we can make all the fields
        AutoLock i_lock(inst_lock);
        std::map<unsigned,Reservation> &atomic_reservations =
          view_reservations[key];
        for (int idx = mask.find_first_set(); idx >= 0;
              idx = mask.find_next_set(idx+1))
        {
          std::map<unsigned,Reservation>::const_iterator finder =
            atomic_reservations.find(idx);
          if (finder == atomic_reservations.end())
          {
            // Make a new reservation and add it to the set
            Reservation handle = Reservation::create_reservation();
            atomic_reservations[idx] = handle;
            results.push_back(handle);
          }
          else
            results.push_back(finder->second);
        }
      }
      else
      {
        // See if we can find them all locally
        {
          AutoLock i_lock(inst_lock, 1, false/*exclusive*/);
          const std::map<unsigned,Reservation> &atomic_reservations =
            view_reservations[key];
          for (int idx = mask.find_first_set(); idx >= 0;
                idx = mask.find_next_set(idx+1))
          {
            std::map<unsigned,Reservation>::const_iterator finder =
              atomic_reservations.find(idx);
            if (finder != atomic_reservations.end())
              results.push_back(finder->second);
            else
              break;
          }
        }
        if (results.size() < mask.pop_count())
        {
          // Couldn't find them all so send the request to the node
          // that should own the instance
          if (!to_trigger.exists())
            to_trigger = Runtime::create_rt_user_event();
          const PhysicalInstance inst = get_instance(point);
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(mask);
            rez.serialize(view_did);
            rez.serialize(point);
            rez.serialize(reservations);
            rez.serialize(source);
            rez.serialize(to_trigger);
          }
          runtime->send_atomic_reservation_request(inst.address_space(), rez);
          return to_trigger;
        }
      }
      if (source != local_space)
      {
#ifdef DEBUG_LEGION
        assert(to_trigger.exists());
#endif
        // Send the result back to the source
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(mask);
          rez.serialize(view_did);
          rez.serialize(point);
          rez.serialize(reservations);
          rez.serialize<size_t>(results.size());
          for (std::vector<Reservation>::const_iterator it =
                results.begin(); it != results.end(); it++)
            rez.serialize(*it);
          rez.serialize(to_trigger);
        }
        runtime->send_atomic_reservation_response(source, rez);
      }
      else
      {
        reservations->swap(results);
        if (to_trigger.exists())
          Runtime::trigger_event(to_trigger);
      }
      return to_trigger;
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::update_field_reservations(const FieldMask &mask,
                               DistributedID view_did, const DomainPoint &point,
                               const std::vector<Reservation> &reservations)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(point.get_dim() > 0);
      assert((dense_points.get_dim() == 0) || dense_points.contains(point));
#endif
      const std::pair<DistributedID,DomainPoint> key(view_did, point);
      AutoLock i_lock(inst_lock);
      std::map<unsigned,Reservation> &atomic_reservations =
          view_reservations[key];
      unsigned offset = 0;
      for (int idx = mask.find_first_set(); idx >= 0;
            idx = mask.find_next_set(idx+1))
        atomic_reservations[idx] = reservations[offset++];
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::reclaim_field_reservations(DistributedID view_did,
                                            std::vector<Reservation> &to_delete)
    //--------------------------------------------------------------------------
    {
      for (std::vector<DomainPoint>::const_iterator it =
            instance_points.begin(); it != instance_points.end(); it++)
      {
        const std::pair<DistributedID,DomainPoint> key(view_did, *it);
        std::map<std::pair<DistributedID,DomainPoint>,
                  std::map<unsigned,Reservation> >::iterator finder =
                    view_reservations.find(key);
        if (finder == view_reservations.end())
          continue;
        for (std::map<unsigned,Reservation>::const_iterator it =
              finder->second.begin(); it != finder->second.end(); it++)
          to_delete.push_back(it->second);
        view_reservations.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveManager::send_manager(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert((collective_mapping == NULL) ||
          !collective_mapping->contains(target));
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(owner_space);
        rez.serialize(total_points);
        rez.serialize(dense_points);
        collective_mapping->pack(rez);
        rez.serialize(instance_footprint);
        // No need for a reference here since we know we'll continue holding it
        instance_domain->pack_expression(rez, target);
        rez.serialize(piece_list_size);
        if (piece_list_size > 0)
          rez.serialize(piece_list, piece_list_size);
        rez.serialize(field_space_node->handle);
        rez.serialize(tree_id);
        rez.serialize(redop);
        rez.serialize<bool>(multi_instance);
        layout->pack_layout_description(rez, target);
        pack_garbage_collection_state(rez, target, true/*need lock*/);
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
      size_t total_points;
      derez.deserialize(total_points);
      IndexSpace points_handle;
      derez.deserialize(points_handle);
      Domain dense_points;
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
      bool multi_instance;
      derez.deserialize(multi_instance);

      LayoutConstraintID layout_id;
      derez.deserialize(layout_id);
      RtEvent layout_ready;
      LayoutConstraints *constraints = 
        runtime->find_layout_constraints(layout_id, 
                    false/*can fail*/, &layout_ready);
      GarbageCollectionState state;
      derez.deserialize(state);
      if (domain_ready.exists() || fs_ready.exists() || layout_ready.exists())
      {
        std::set<RtEvent> preconditions;
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
          DeferCollectiveManagerArgs args(did, owner_space, dense_points, 
              total_points, mapping, inst_footprint, inst_domain, pending,
              handle, tree_id, layout_id, redop, piece_list,
              piece_list_size, source, state, multi_instance);
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
      create_collective_manager(runtime, did, owner_space, dense_points,
          total_points, mapping, inst_footprint, inst_domain, piece_list,
          piece_list_size, space_node, tree_id, constraints,
          redop, state, multi_instance);
    }

    //--------------------------------------------------------------------------
    CollectiveManager::DeferCollectiveManagerArgs::DeferCollectiveManagerArgs(
            DistributedID d, AddressSpaceID own, const Domain &pts, size_t tot,
            CollectiveMapping *map, size_t f, IndexSpaceExpression *lx, 
            const PendingRemoteExpression &p, FieldSpace h, RegionTreeID tid,
            LayoutConstraintID l, ReductionOpID r, const void *pl,
            size_t pl_size, const AddressSpaceID src,
            GarbageCollectionState gc, bool m)
      : LgTaskArgs<DeferCollectiveManagerArgs>(implicit_provenance),
        did(d), owner(own), dense_points(pts), total_points(tot),
        mapping(map), footprint(f), local_expr(lx), pending(p), handle(h),
        tree_id(tid), layout_id(l), redop(r), piece_list(pl),
        piece_list_size(pl_size), source(src), state(gc), multi_instance(m)
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
      IndexSpaceExpression *inst_domain = dargs->local_expr;
      if (inst_domain == NULL)
        inst_domain = runtime->forest->find_remote_expression(dargs->pending);
      FieldSpaceNode *space_node = runtime->forest->get_node(dargs->handle);
      LayoutConstraints *constraints = 
        runtime->find_layout_constraints(dargs->layout_id);
      create_collective_manager(runtime, dargs->did, dargs->owner, 
          dargs->dense_points, dargs->total_points, dargs->mapping,
          dargs->footprint, inst_domain, dargs->piece_list, 
          dargs->piece_list_size, space_node, dargs->tree_id, constraints,
          dargs->redop, dargs->state, dargs->multi_instance);
      // Remove the local expression reference if necessary
      if ((dargs->local_expr != NULL) &&
          dargs->local_expr->remove_base_expression_reference(META_TASK_REF))
        delete dargs->local_expr;
      if (dargs->mapping->remove_reference())
        delete dargs->mapping;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::create_collective_manager(
          Runtime *runtime, DistributedID did, AddressSpaceID owner_space, 
          const Domain &dense_points, size_t points,
          CollectiveMapping *mapping, size_t inst_footprint, 
          IndexSpaceExpression *inst_domain, const void *piece_list,
          size_t piece_list_size, FieldSpaceNode *space_node, 
          RegionTreeID tree_id,LayoutConstraints *constraints,
          ReductionOpID redop,GarbageCollectionState state,bool multi_instance)
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
                                            owner_space, dense_points, points,
                                            mapping, inst_domain, piece_list,
                                            piece_list_size, space_node,tree_id,
                                            layout, redop, false/*reg now*/,
                                            inst_footprint, external_instance,
                                            multi_instance);
      else
        man = new CollectiveManager(runtime->forest, did, owner_space,
                                    dense_points, points, mapping, inst_domain,
                                    piece_list, piece_list_size, space_node,
                                    tree_id, layout, redop, false/*reg now*/,
                                    inst_footprint, external_instance,
                                    multi_instance);
      man->initialize_remote_gc_state(state);
      // Hold-off doing the registration until construction is complete
      man->register_with_runtime();
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_point_request(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      CollectiveManager *manager = static_cast<CollectiveManager*>(
                          runtime->weak_find_distributed_collectable(did));
      size_t num_points;
      derez.deserialize(num_points);
      std::set<DomainPoint> points;
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        DomainPoint point;
        derez.deserialize(point);
        points.insert(point);
      }
      AddressSpaceID source;
      derez.deserialize(source);
      AddressSpaceID origin;
      derez.deserialize(origin);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);

      if (manager != NULL)
      {
        manager->find_or_forward_physical_instance(source, origin, 
                                                   points, to_trigger);
        if (manager->remove_base_resource_ref(RUNTIME_REF))
          delete manager;
      }
      else
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_point_response(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      CollectiveManager *manager = static_cast<CollectiveManager*>(
                          runtime->find_distributed_collectable(did));
      size_t num_points;
      derez.deserialize(num_points);
      std::map<DomainPoint,RemoteInstInfo> insts;
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        DomainPoint point;
        derez.deserialize(point);
        RemoteInstInfo &inst = insts[point];
        derez.deserialize(inst.instance);
        derez.deserialize(inst.unique_event);
        derez.deserialize(inst.index);
      }
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      manager->record_remote_physical_instances(insts);
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveManager::handle_deletion(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      CollectiveManager *manager = static_cast<CollectiveManager*>(
                          runtime->find_distributed_collectable(did));
      AddressSpaceID source;
      derez.deserialize(source);
      RtUserEvent done;
      derez.deserialize(done);
      Runtime::trigger_event(done, manager->perform_deletion(source));
    }
#endif // NO_EXPLICIT_COLLECTIVES

    /////////////////////////////////////////////////////////////
    // Virtual Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VirtualManager::VirtualManager(Runtime *runtime, DistributedID did,
                            LayoutDescription *desc, CollectiveMapping *mapping)
      : InstanceManager(runtime->forest, did, desc,
                        NULL/*field space node*/,NULL/*index space expression*/,
                        0/*tree id*/, true/*register now*/, mapping)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info("GC Virtual Manager %lld %d",
                        LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space); 
#endif
    }

    //--------------------------------------------------------------------------
    VirtualManager::VirtualManager(const VirtualManager &rhs)
      : InstanceManager(NULL, 0, NULL, NULL, NULL, 0, false)
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
    PointerConstraint VirtualManager::get_pointer_constraint(void) const
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

#ifdef NO_EXPLICIT_COLLECTIVES
    /////////////////////////////////////////////////////////////
    // Pending Collective Instance 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PendingCollectiveManager::PendingCollectiveManager(DistributedID id,
                                    size_t total, const Domain &dense,
                                    CollectiveMapping *mapping, bool multi_inst)
      : did(id), total_points(total), dense_points(dense),
        collective_mapping(mapping), multi_instance(multi_inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(did > 0);
      assert(collective_mapping != NULL);
#endif
      collective_mapping->add_reference();
    }

    //--------------------------------------------------------------------------
    PendingCollectiveManager::~PendingCollectiveManager(void)
    //--------------------------------------------------------------------------
    {
      if (collective_mapping->remove_reference())
        delete collective_mapping;
    }

    //--------------------------------------------------------------------------
    void PendingCollectiveManager::pack(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(did);
      rez.serialize(total_points);
      rez.serialize(dense_points);
      collective_mapping->pack(rez);
      rez.serialize<bool>(multi_instance);
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
      Domain dense_points;
      derez.deserialize(dense_points);
      size_t total_spaces;
      derez.deserialize(total_spaces);
      CollectiveMapping *mapping = new CollectiveMapping(derez, total_spaces);
      bool multi_instance;
      derez.deserialize(multi_instance);
      return new PendingCollectiveManager(did, total_points, dense_points,
                                          mapping, multi_instance);
    }
#endif // NO_EXPLICIT_COLLECTIVES

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
        RegionTreeForest *forest, LayoutConstraintKind *unsat_kind,
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
      Realm::ExternalMemoryResource resource(base_ptr,
          inst_layout->bytes_used, false/*read only*/);
      ready = ApEvent(PhysicalInstance::create_external_instance(instance,
                memory_manager->memory, inst_layout, resource, requests));
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
#ifdef NO_EXPLICIT_COLLECTIVES
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
          // First point so create it
          manager = new(buffer) CollectiveManager(forest,
              pending_collective->did, owner_space,
              pending_collective->dense_points,
              pending_collective->total_points,
              pending_collective->collective_mapping,
              instance_domain, piece_list, piece_list_size,
              field_space_node, tree_id, layout, redop_id,
              true/*register now*/, instance_footprint,
              false/*external*/, pending_collective->multi_instance);
#ifdef DEBUG_LEGION
          assert((manager == collectable) || (collectable == NULL));
#endif
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
        manager->record_point_instance(*collective_point, instance, ready);
        result = manager;
      }
      else
#endif // NO_EXPLICIT_COLLECTIVES
      {
        // Creating an individual manager
        DistributedID did = forest->runtime->get_available_distributed_id();
        // Figure out what kind of instance we just made
        switch (constraints.specialized_constraint.get_kind())
        {
          case LEGION_NO_SPECIALIZE:
          case LEGION_AFFINE_SPECIALIZE:
          case LEGION_COMPACT_SPECIALIZE:
            {
              // Now we can make the manager
              result = new PhysicalManager(forest, did, memory_manager,
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
              result = new PhysicalManager(forest, did,
                                           memory_manager, instance, 
                                           instance_domain, piece_list,
                                           piece_list_size, field_space_node,
                                           tree_id, layout, redop_id,
                                           true/*register now*/,
                                           instance_footprint, ready,
                                        PhysicalManager::INTERNAL_INSTANCE_KIND,
                                           reduction_op);
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

