/* Copyright 2024 Stanford University, NVIDIA Corporation
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
    /*static*/ void CollectiveMapping::pack_null(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(0);
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
      return layout->constraints->entails(constraints,
              (instance_domain != NULL) ? instance_domain->get_num_dims() : 0,
              failed_constraint, false/*test pointer*/);
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
      return layout->constraints->entails(constraints,
              (instance_domain != NULL) ? instance_domain->get_num_dims() : 0,
              failed_constraint, false/*test pointer*/);
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
                        ApEvent u_event, LgEvent unique, InstanceKind k,
                        const ReductionOp *op /*= NULL*/,
                        CollectiveMapping *mapping /*=NULL*/,
                        ApEvent p_event /*= ApEvent::NO_AP_EVENT*/,
                        GarbageCollectionState init /*COLLECTABLE_GC_STATE*/)
      : InstanceManager(ctx, encode_instance_did(did, 
          (k == EXTERNAL_ATTACHED_INSTANCE_KIND), (redop_id > 0)), layout, node,
          // If we're on the owner node we need to produce the expression
          // that actually describes this points in this space
          // On remote nodes we'll already have it from the owner
          (ctx->runtime->determine_owner(did) == ctx->runtime->address_space) &&
            (k != UNBOUND_INSTANCE_KIND) ?
            instance_domain->create_layout_expression(pl, pl_size) : 
            instance_domain, tree_id, register_now, mapping), 
        memory_manager(memory), unique_event(unique), 
        instance_footprint(footprint), reduction_op((redop_id == 0) ? NULL : 
            ctx->runtime->get_reduction(redop_id)), redop(redop_id),
        piece_list(pl), piece_list_size(pl_size), instance(inst),
        use_event(Runtime::create_ap_user_event(NULL)),
        instance_ready((k == UNBOUND_INSTANCE_KIND) ? 
            Runtime::create_rt_user_event() : RtUserEvent::NO_RT_USER_EVENT),
        kind(k), external_pointer(-1UL), producer_event(p_event),
        gc_state(init), pending_changes(0),
        failed_collection_count(0), min_gc_priority(0), added_gc_events(0),
        valid_references(0), sent_valid_references(0),
        received_valid_references(0), padded_reservations(NULL)
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
        add_base_valid_ref(PENDING_UNBOUND_REF);
      // If we're in a pending collectable state, then add a reference
      if (gc_state == PENDING_COLLECTED_GC_STATE)
      {
#ifdef DEBUG_LEGION
        assert(!is_owner());
#endif
        add_base_resource_ref(PENDING_COLLECTIVE_REF);
      }
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
      if (padded_reservations != NULL)
      {
        // If this is the owner view, delete any atomic reservations
        if (is_owner())
        {
          for (std::map<unsigned,Reservation>::iterator it = 
                padded_reservations->begin(); it != 
                padded_reservations->end(); it++)
            it->second.destroy_reservation();
        }
        delete padded_reservations;
      }
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
      for (std::vector<TilingConstraint>::const_iterator it = 
            constraints->tiling_constraints.begin(); it !=
            constraints->tiling_constraints.end(); it++)
        LegionSpy::log_instance_tiling_constraint(inst_event,
                                it->dim, it->value, it->tiles);
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
    PhysicalInstance PhysicalManager::get_instance(void) const
    //--------------------------------------------------------------------------
    {
      if (instance_ready.exists() && !instance_ready.has_triggered())
        instance_ready.wait();
      return instance;
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
          void *location = 
            runtime->find_or_create_pending_collectable_location<ReductionView>(
              view_did);
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
          void *location = runtime->find_or_create_pending_collectable_location<
            MaterializedView>(view_did);
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
          own_ctx->pack_inner_context(rez);
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
          own_ctx->pack_inner_context(rez);
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
    bool PhysicalManager::register_deletion_subscriber(
                                         InstanceDeletionSubscriber *subscriber)
    //--------------------------------------------------------------------------
    {
      subscriber->add_subscriber_reference(this);
      {
        AutoLock inst(inst_lock);
        if (gc_state != COLLECTED_GC_STATE)
        {
#ifdef DEBUG_LEGION
          assert(subscribers.find(subscriber) == subscribers.end());
#endif
          subscribers.insert(subscriber);
          return true;
        }
      }
      if (subscriber->remove_subscriber_reference(this))
        delete subscriber;
      return false;
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
      // Only need to get these if we didn't already delete the manager
      // If we already deleted the manager there is already a meta-task in
      // flight that summarizes these events and makes sure we can't shutdown
      // without it running, see perform_deletion
      if (gc_state != COLLECTED_GC_STATE)
      {
        for (std::set<ApEvent>::const_iterator it =
              gc_events.begin(); it != gc_events.end(); it++)
          if (!it->has_triggered_faultignorant())
            preconditions.insert(*it);
      }
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::meets_regions(
      const std::vector<LogicalRegion> &regions, bool tight_region_bounds,
      const Domain *padding_delta) const
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
      return meets_expression(space_expr, tight_region_bounds, padding_delta);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::meets_expression(IndexSpaceExpression *space_expr,
                           bool tight_bounds, const Domain *padding_delta) const
    //--------------------------------------------------------------------------
    {
      return instance_domain->meets_layout_expression(space_expr, tight_bounds,
                                    piece_list, piece_list_size, padding_delta);
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
        return notify_invalid(i_lock);
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
        return notify_invalid(i_lock);
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
        return notify_invalid(i_lock);
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
      if (need_check && (kind != UNBOUND_INSTANCE_KIND) &&
          (!is_external_instance() || !is_owner()))
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
    bool PhysicalManager::notify_invalid(AutoLock &i_lock)
    //--------------------------------------------------------------------------
    {
      // No need for the lock it is held by the caller
#ifdef DEBUG_LEGION
      assert(kind != UNBOUND_INSTANCE_KIND);
      assert((gc_state == VALID_GC_STATE) || is_external_instance());
#endif
      // If we're an external instance that has already been detached and
      // therfore deleted then we don't ever want to go back to collectable
      if (!is_external_instance() || (gc_state != COLLECTED_GC_STATE))
      {
        gc_state = COLLECTABLE_GC_STATE;
        // If we're an eagerly allocated instance start the collection
        // immediately since there's no point in re-use
        if (kind == EAGER_INSTANCE_KIND)
        {
          RtEvent dummy_ready;
          collect(dummy_ready, &i_lock);
        }
      }
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
        switch (gc_state.load())
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
    bool PhysicalManager::acquire_collect(std::set<ApEvent> &remote_events,
                                 uint64_t &sent_valid, uint64_t &received_valid)
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
      sent_valid = sent_valid_references;
      received_valid = received_valid_references;
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
      uint64_t sent_valid = 0, received_valid = 0;
      if (!manager->acquire_collect(gc_events, sent_valid, received_valid))
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
        // If we have different numbers of sent and received valid
        // references then we need to tell the owner that too
        if (sent_valid != received_valid)
        {
          const RtUserEvent notified = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(sent_valid);
            rez.serialize(received_valid);
            rez.serialize(notified);
          }
          runtime->send_gc_mismatch(owner, rez);
          ready_events.insert(notified);
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
    void PhysicalManager::process_remote_reference_mismatch(
                                               uint64_t sent, uint64_t received)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(gc_state != COLLECTED_GC_STATE);
#endif
      sent_valid_references += sent;
      received_valid_references += received;
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_garbage_collection_mismatch(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      uint64_t remote_sent, remote_received;
      derez.deserialize(remote_sent);
      derez.deserialize(remote_received);
      RtUserEvent done;
      derez.deserialize(done);
      // Should still be able to find this manager here
      PhysicalManager *manager = static_cast<PhysicalManager*>(
          runtime->find_distributed_collectable(did));
      manager->process_remote_reference_mismatch(remote_sent, remote_received);
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
        switch (gc_state.load())
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
    bool PhysicalManager::is_collected(void) const
    //--------------------------------------------------------------------------
    {
      return (gc_state == COLLECTED_GC_STATE);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::collect(RtEvent &ready, AutoLock *i_lock)
    //--------------------------------------------------------------------------
    {
      if (i_lock == NULL)
      {
        AutoLock i2_lock(inst_lock);
        return collect(ready, &i2_lock);
      }
      // Do a quick to check to see if we can do a collection on the local node
      if (gc_state == VALID_GC_STATE)
        return false;
      // If it's already collected then we're done
      if (gc_state == COLLECTED_GC_STATE)
        return true;
      bool has_local_references = false;
      uint64_t local_valid_sent = 0, local_valid_received = 0;
      if (is_owner())
      {
        // Check to see if anyone is already performing a deletion
        // on this manager, if so then deduplicate
        if (gc_state == COLLECTABLE_GC_STATE)
        {
          gc_state = PENDING_COLLECTED_GC_STATE;
          failed_collection_count.store(0);
          // Pull a copy of these onto the stack in case we fail to 
          // collect and we need to restore them
          local_valid_sent = sent_valid_references;
          local_valid_received = received_valid_references;
          has_local_references = true;
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
          i_lock->release();
          wait_on.wait();
          i_lock->reacquire();
        }
#ifdef DEBUG_LEGION
        assert(pending_changes > 0);
#endif
        switch (gc_state.load())
        {
          // Anything in these states means the collection attempt failed
          // because something else acquired a valid reference while
          // the collection was in progress
          case VALID_GC_STATE:
          case COLLECTABLE_GC_STATE:
            {
              // Restore our local sent/received counts
              if (has_local_references)
              {
                sent_valid_references = local_valid_sent;
                received_valid_references = local_valid_received;
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
              // were unable to acquire on remote nodes or whether there
              // are still packed valid reference outstanding
              if ((failed_collection_count.load() > 0) ||
                  (sent_valid_references != received_valid_references))
              {
                // Restore our local sent/received counts
                if (has_local_references)
                {
                  sent_valid_references = local_valid_sent;
                  received_valid_references = local_valid_received;
                }
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
                // Grab the set of active contexts to notify
                std::set<InstanceDeletionSubscriber*> to_notify;
                // Notify the subscribers if we've been collected
                to_notify.swap(subscribers);
                // Now we can perform the deletion which will release the lock
                perform_deletion(runtime->address_space, i_lock);
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
        i_lock->release();
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
        i_lock->reacquire();
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
              switch (gc_state.load())
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
                                                           Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID source;
      derez.deserialize(source);
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
      InnerContext *context = InnerContext::unpack_inner_context(derez,runtime);
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
      if (man_ready.exists() && !man_ready.has_triggered())
      {
        RemoteCreateViewArgs args(manager, context, logical_owner,
                                  mapping, target, source, done);
        runtime->issue_runtime_meta_task(args,
            LG_LATENCY_DEFERRED_PRIORITY, man_ready);
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
        rez.serialize(unique_event);
        if (kind != UNBOUND_INSTANCE_KIND)
          rez.serialize<ApEvent>(use_event);
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
      LgEvent unique_event;
      derez.deserialize(unique_event);
      ApEvent use_event;
      derez.deserialize(use_event);
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
              handle, tree_id, layout_id, use_event, unique_event, kind,
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
                            space_node, tree_id, constraints, use_event,
                            unique_event, kind, redop, gc_state);
    }

    //--------------------------------------------------------------------------
    PhysicalManager::DeferPhysicalManagerArgs::DeferPhysicalManagerArgs(
            DistributedID d, Memory m, PhysicalInstance i, 
            size_t f, IndexSpaceExpression *lx, 
            const PendingRemoteExpression &p, FieldSpace h, RegionTreeID tid,
            LayoutConstraintID l, ApEvent use, LgEvent unique, InstanceKind k, 
            ReductionOpID r, const void *pl, size_t pl_size,
            GarbageCollectionState gc)
      : LgTaskArgs<DeferPhysicalManagerArgs>(implicit_provenance),
            did(d), mem(m), inst(i), footprint(f), pending(p),
            local_expr(lx), handle(h), tree_id(tid), layout_id(l), 
            use_event(use), unique_event(unique), kind(k), redop(r),
            piece_list(pl), piece_list_size(pl_size), state(gc)
    //--------------------------------------------------------------------------
    {
      if (local_expr != NULL)
        local_expr->add_base_expression_reference(META_TASK_REF);
    }

    //--------------------------------------------------------------------------
    PhysicalManager::DeferDeletePhysicalManager
                     ::DeferDeletePhysicalManager(PhysicalManager *manager_)
      : LgTaskArgs<DeferDeletePhysicalManager>(implicit_provenance),
        manager(manager_)
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
          dargs->use_event, dargs->unique_event, dargs->kind,
          dargs->redop, dargs->state);
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
      PhysicalManager *manager = dargs->manager;
      manager->memory_manager->unregister_deleted_instance(manager);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::create_remote_manager(Runtime *runtime, 
          DistributedID did, Memory mem, 
          PhysicalInstance inst, size_t inst_footprint, 
          IndexSpaceExpression *inst_domain, const void *piece_list,
          size_t piece_list_size, FieldSpaceNode *space_node, 
          RegionTreeID tree_id, LayoutConstraints *constraints, 
          ApEvent use_event, LgEvent unique_event, InstanceKind kind,
          ReductionOpID redop, GarbageCollectionState state)
    //--------------------------------------------------------------------------
    {
      LayoutDescription *layout = 
        LayoutDescription::handle_unpack_layout_description(constraints,
                                space_node, inst_domain->get_num_dims());
      MemoryManager *memory = runtime->find_memory_manager(mem);
      const ReductionOp *op = 
        (redop == 0) ? NULL : runtime->get_reduction(redop);
      void *location = runtime->find_or_create_pending_collectable_location<
                                                        PhysicalManager>(did);
      PhysicalManager *man = new(location) PhysicalManager(runtime->forest,
                                            did, memory, inst, inst_domain, 
                                            piece_list, piece_list_size, 
                                            space_node, tree_id, layout, 
                                            redop, false/*reg now*/, 
                                            inst_footprint, use_event, 
                                            unique_event, kind, op,
                                            NULL, ApEvent::NO_AP_EVENT, state);
      // Hold-off doing the registration until construction is complete
      man->register_with_runtime();
      // Remove the reference we got back on the layout description
      if (layout->remove_reference())
        delete layout;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::perform_deletion(AddressSpaceID source,
                                           AutoLock *i_lock /* = NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(source == local_space);
#endif
      if (i_lock == NULL)
      {
        AutoLock instance_lock(inst_lock);
        perform_deletion(source, &instance_lock);
        return;
      }
#ifdef DEBUG_LEGION
      assert(pending_views.empty());
      assert(gc_state != COLLECTED_GC_STATE);
#endif
      gc_state = COLLECTED_GC_STATE;
      log_garbage.spew("Deleting physical instance " IDFMT " in memory " 
                       IDFMT "", instance.id, memory_manager->memory.id);
      RtEvent deferred_deletion;
#ifndef LEGION_DISABLE_GC
      // Get the deferred deletion event from the gc events
      if (!gc_events.empty())
        deferred_deletion = Runtime::protect_merge_events(gc_events);
      // Now we can release the lock since we're done with the atomic updates
      i_lock->release();
      std::vector<PhysicalInstance::DestroyedField> serdez_fields;
      layout->compute_destroyed_fields(serdez_fields);
      // Handle a small race where the instance name is not ready yet
      if (instance_ready.exists() && !instance_ready.has_triggered())
        instance_ready.wait();
#ifdef LEGION_MALLOC_INSTANCES
      if (kind == INTERNAL_INSTANCE_KIND)
        memory_manager->free_legion_instance(this, deferred_deletion);
#else
      if (kind == EAGER_INSTANCE_KIND)
        memory_manager->free_eager_instance(instance, deferred_deletion);
      else
      {
        if (!serdez_fields.empty())
          instance.destroy(serdez_fields, deferred_deletion);
        else
          instance.destroy(deferred_deletion);
      }
#endif
#else
      // Release the i_lock since we're done with the atomic updates
      i_lock->release();
#endif
      // Once the deletion is actually done then we can tell the memory
      // manager that the deletion is finished and it is safe to remove
      // this manager for its list of current instances
      if (deferred_deletion.exists() && !deferred_deletion.has_triggered())
      {
        DeferDeletePhysicalManager args(this);
        runtime->issue_runtime_meta_task(
            args, LG_LOW_PRIORITY, deferred_deletion);
      }
      else
        memory_manager->unregister_deleted_instance(this);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::force_deletion(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      // If we already deleted this then it's deferred deletion task should
      // have run and pruned it out of the memory manager to prevent calling
      // force_deletion on this physical manager
      assert(gc_state != COLLECTED_GC_STATE);
#endif
      // There are no races by the time we get here so we don't need the lock
      log_garbage.spew("Force deleting physical instance " IDFMT " in memory "
                       IDFMT "", instance.id, memory_manager->memory.id);
#ifndef LEGION_DISABLE_GC
      std::vector<PhysicalInstance::DestroyedField> serdez_fields;
      layout->compute_destroyed_fields(serdez_fields);
#ifdef LEGION_MALLOC_INSTANCES
      if (kind == INTERNAL_INSTANCE_KIND)
        memory_manager->free_legion_instance(this, RtEvent::NO_RT_EVENT);
#else
      // If this is an eager allocation, return it back to the eager pool
      if (kind == EAGER_INSTANCE_KIND)
        memory_manager->free_eager_instance(instance, RtEvent::NO_RT_EVENT);
      else
      {
        if (!serdez_fields.empty())
          instance.destroy(serdez_fields);
        else
          instance.destroy();
      }
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
    void PhysicalManager::detach_external_instance(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_external_instance());
#endif
      memory_manager->detach_external_instance(this);
    }
    
    //--------------------------------------------------------------------------
    uintptr_t PhysicalManager::get_instance_pointer(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      if (use_event.exists() && !use_event.has_triggered_faultignorant())
        use_event.wait_faultignorant();
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
        kind = EAGER_INSTANCE_KIND;
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
      return remove_base_valid_ref(PENDING_UNBOUND_REF);
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

      RtEvent manager_ready;
      PhysicalManager *manager =
        runtime->find_or_request_instance_manager(did, manager_ready);
      if (manager_ready.exists() && !manager_ready.has_triggered())
        manager_ready.wait();

      if (manager->update_physical_instance(instance, footprint))
        delete manager;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::find_padded_reservations(const FieldMask &mask,
                                                  Operation *op, unsigned index)
    //--------------------------------------------------------------------------
    {
      std::vector<Reservation> reservations(mask.pop_count());
      find_field_reservations(mask, reservations);
      for (unsigned idx = 0; idx < reservations.size(); idx++)
        op->update_atomic_locks(index, reservations[idx], true/*exclusive*/);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::find_field_reservations(const FieldMask &mask,
                                         std::vector<Reservation> &reservations)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mask.pop_count() == reservations.size());
#endif
      unsigned offset = 0;
      if (is_owner())
      {
        AutoLock i_lock(inst_lock);
        if (padded_reservations == NULL)
          padded_reservations = new std::map<unsigned,Reservation>();
        for (int idx = mask.find_first_set(); idx >= 0;
              idx = mask.find_next_set(idx+1))
        {
          std::map<unsigned,Reservation>::const_iterator finder = 
            padded_reservations->find(idx);
          if (finder == padded_reservations->end())
          {
            // Make a new reservation and add it to the set
            Reservation handle = Reservation::create_reservation();
            padded_reservations->insert(std::make_pair(idx, handle));
            reservations[offset++] = handle;
          }
          else
            reservations[offset++] = finder->second;
        }
      }
      else
      {
        // Figure out which fields we need requests for and send them
        FieldMask needed_fields;
        {
          AutoLock i_lock(inst_lock, 1, false);
          if (padded_reservations == NULL)
          {
            for (int idx = mask.find_first_set(); idx >= 0;
                  idx = mask.find_next_set(idx+1))
              needed_fields.set_bit(idx);
          }
          else
          {
            for (int idx = mask.find_first_set(); idx >= 0;
                  idx = mask.find_next_set(idx+1))
            {
              std::map<unsigned,Reservation>::const_iterator finder = 
                padded_reservations->find(idx);
              if (finder == padded_reservations->end())
                needed_fields.set_bit(idx);
              else
                reservations[offset++] = finder->second;
            }
          }
        }
        if (!!needed_fields)
        {
          RtUserEvent wait_on = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(needed_fields);
            rez.serialize(wait_on);
          }
          runtime->send_padded_reservation_request(owner_space, rez);
          wait_on.wait();
          // Now retake the lock and get the remaining reservations
          AutoLock i_lock(inst_lock, 1, false);
#ifdef DEBUG_LEGION
          assert(padded_reservations != NULL);
#endif
          for (int idx = needed_fields.find_first_set(); idx >= 0;
                idx = needed_fields.find_next_set(idx+1))
          {
            std::map<unsigned,Reservation>::const_iterator finder =
              padded_reservations->find(idx);
#ifdef DEBUG_LEGION
            assert(finder != padded_reservations->end());
#endif
            reservations[offset++] = finder->second;
          }
        }
      }
#ifdef DEBUG_LEGION
      assert(offset == reservations.size());
#endif
      // Sort them before returning
      if (reservations.size() > 1)
        std::sort(reservations.begin(), reservations.end());
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_padded_reservation_request(
                   Runtime *runtime, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      FieldMask needed_fields;
      derez.deserialize(needed_fields);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      PhysicalManager *target = dynamic_cast<PhysicalManager*>(dc);
      assert(target != NULL);
#else
      PhysicalManager *target = static_cast<PhysicalManager*>(dc);
#endif
      std::vector<Reservation> reservations(needed_fields.pop_count());
      target->find_field_reservations(needed_fields, reservations);
      Serializer rez;
      {
        RezCheck z2(rez);
        rez.serialize(did);
        rez.serialize(needed_fields);
        for (unsigned idx = 0; idx < reservations.size(); idx++)
          rez.serialize(reservations[idx]);
        rez.serialize(to_trigger);
      }
      runtime->send_padded_reservation_response(source, rez);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::update_field_reservations(const FieldMask &mask,
                                   const std::vector<Reservation> &reservations)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner());
      assert(mask.pop_count() == reservations.size());
#endif
      unsigned offset = 0;
      AutoLock i_lock(inst_lock);
      if (padded_reservations == NULL)
        padded_reservations = new std::map<unsigned,Reservation>();
      for (int idx = mask.find_first_set(); idx >= 0;
            idx = mask.find_next_set(idx+1))
        padded_reservations->insert(std::make_pair(idx,reservations[offset++]));
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::handle_padded_reservation_response(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      FieldMask mask;
      derez.deserialize(mask);
      std::vector<Reservation> reservations(mask.pop_count());
      for (unsigned idx = 0; idx < reservations.size(); idx++)
        derez.deserialize(reservations[idx]);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      PhysicalManager *target = dynamic_cast<PhysicalManager*>(dc);
      assert(target != NULL);
#else
      PhysicalManager *target = static_cast<PhysicalManager*>(dc);
#endif
      target->update_field_reservations(mask, reservations);
      Runtime::trigger_event(to_trigger);
    }

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
        const SpecializedConstraint &spec = constraints.specialized_constraint;
        switch (spec.get_kind())
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
        size_t num_pieces = 0;
        realm_layout =
          instance_domain->create_layout(constraints, field_set,
             field_sizes, compact, &piece_list, &piece_list_size, &num_pieces);
#ifdef DEBUG_LEGION
        assert(realm_layout != NULL);
#endif
        // If we were doing a compact layout then Check that we met 
        // the constraints for efficiency and number of pieces
        if (compact && (spec.max_pieces < num_pieces))
        {
          if (unsat_kind != NULL)
            *unsat_kind = LEGION_SPECIALIZED_CONSTRAINT;
          if (unsat_index != NULL)
            *unsat_index = 0;
          if (footprint != NULL)
            *footprint = realm_layout->bytes_used;
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
      LgEvent unique_event;
      if (runtime->legion_spy_enabled || (runtime->profiler != NULL))
      {
        RtUserEvent unique = Runtime::create_rt_user_event();
        Runtime::trigger_event(unique);
        unique_event = unique;
      }
      ApEvent ready;
      if (runtime->profiler != NULL)
        runtime->profiler->add_inst_request(requests, creator_id, unique_event);
#ifndef LEGION_MALLOC_INSTANCES
      ready = ApEvent(PhysicalInstance::create_instance(instance,
            memory_manager->memory, inst_layout, requests, precondition));
      // Wait for the profiling response
      if (!profiling_ready.has_triggered())
        profiling_ready.wait();
#else
      if (precondition.exists() && !precondition.has_triggered())
        precondition.wait();
      ready = ApEvent(memory_manager->allocate_legion_instance(inst_layout, 
                                                      requests, instance));
      if (!instance.exists())
      {
        if (unsat_kind != NULL)
          *unsat_kind = LEGION_MEMORY_CONSTRAINT;
        if (unsat_index != NULL)
          *unsat_index = 0;
        return NULL;
      }
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
                                         instance_footprint,
                                         ready, unique_event,
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
                                         instance_footprint,
                                         ready, unique_event,
                                      PhysicalManager::INTERNAL_INSTANCE_KIND,
                                         reduction_op);
            break;
          }
        default:
          assert(false); // illegal specialized case
      }
      // manager takes ownership of the piece list
      piece_list = NULL;
      // Remove the reference we got back from finding or creating the layout
      if (layout->remove_reference())
        delete layout;
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
#ifdef LEGION_MALLOC_INSTANCES
      memory_manager->record_legion_instance(result, instance);
#endif
      if (runtime->profiler != NULL)
      {
        // Log the logical regions and fields that make up this instance
        for (std::vector<LogicalRegion>::const_iterator it =
              regions.begin(); it != regions.end(); it++)
          if (it->exists())
            runtime->profiler->record_physical_instance_region(unique_event, 
                                                               *it);
        runtime->profiler->record_physical_instance_layout(unique_event,
                                                     layout->owner->handle,
                                                     *layout->constraints);
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
        if (!it->exists())
          continue;
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
      // Check the tiling constraints
      if (!constraints.tiling_constraints.empty())
      {
        // Check to make sure we're not asking for a compact-sparse instance
        switch (constraints.specialized_constraint.get_kind())
        {
          case LEGION_COMPACT_SPECIALIZE:
          case LEGION_COMPACT_REDUCTION_SPECIALIZE:
            REPORT_LEGION_ERROR(ERROR_ILLEGAL_LAYOUT_CONSTRAINT,
                "Illegal tiling constraints specified for compact-sparse "
                "instance creation. Tiling constraints can only be specified "
                "on affine instances currently. If you have a compelling use "
                "case for tiling the pieces of an compact-sparse instance "
                "please report it to the Legion developer's mailing list.")
          default:
            break;
        }
        // Make sure that each of the dimensions are valid and aren't duplicated
        std::vector<bool> observed(num_dims, false);
        for (std::vector<TilingConstraint>::iterator it =
              constraints.tiling_constraints.begin(); it !=
              constraints.tiling_constraints.end(); /*nothing*/)
        {
          if ((it->dim < num_dims) && !observed[it->dim])
          {
            observed[it->dim] = true;
            it++;
          }
          else
            it = constraints.tiling_constraints.erase(it);
        }
      }
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
#ifdef DEBUG_LEGION
      assert((constraints.padding_constraint.delta.get_dim() == 0) ||
             (constraints.padding_constraint.delta.get_dim() == (int)num_dims));
#endif
      // If we don't have a padding constraint then record that we 
      // don't have any padding on this instance
      if (constraints.padding_constraint.delta.get_dim() == 0)
      {
        DomainPoint empty;
        empty.dim = num_dims;
        for (unsigned dim = 0; dim < num_dims; dim++)
          empty[dim] = 0; // no padding
        constraints.padding_constraint.delta = Domain(empty, empty);
      }
      else
      {
        DomainPoint lo = constraints.padding_constraint.delta.lo();
        DomainPoint hi = constraints.padding_constraint.delta.hi();
        for (unsigned dim = 0; dim < num_dims; dim++)
        {
          if (lo[dim] < 0)
            lo[dim] = 0;
          if (hi[dim] < 0)
            hi[dim] = 0;
        }
        constraints.padding_constraint.delta = Domain(lo, hi);
      }
    }

  }; // namespace Internal
}; // namespace Legion

