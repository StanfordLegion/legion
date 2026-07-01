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

#include "legion/instances/layout.h"
#include "legion/kernel/runtime.h"
#include "legion/api/registrars.h"
#include "legion/nodes/field.h"
#include "legion/tools/spy.h"
#include "legion/utilities/bitmask.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Layout Description
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LayoutDescription::LayoutDescription(
        FieldSpaceNode* own, const FieldMask& mask, const unsigned dims,
        LayoutConstraints* con, const std::vector<unsigned>& mask_index_map,
        const std::vector<FieldID>& field_ids,
        const std::vector<size_t>& field_sizes,
        const std::vector<CustomSerdezID>& serdez)
      : allocated_fields(mask), constraints(con), owner(own), total_dims(dims)
    //--------------------------------------------------------------------------
    {
      constraints->add_base_gc_ref(LAYOUT_DESC_REF);
      field_infos.resize(field_sizes.size());
      // Switch data structures from layout by field order to order
      // of field locations in the bit mask
      // Greater than or equal because local fields can alias onto the
      // same index for the allocated instances, note that the fields
      // themselves still get allocated their own space in the instance
      legion_assert(
          mask_index_map.size() >=
          size_t(FieldMask::pop_count(allocated_fields)));
      for (unsigned idx = 0; idx < mask_index_map.size(); idx++)
      {
        // This gives us the index in the field ordered data structures
        unsigned index = mask_index_map[idx];
        FieldID fid = field_ids[index];
        field_indexes[fid] = idx;
        CopySrcDstField& info = field_infos[idx];
        info.size = field_sizes[index];
        info.field_id = fid;
        info.serdez_id = serdez[index];
      }
    }

    //--------------------------------------------------------------------------
    LayoutDescription::LayoutDescription(
        const FieldMask& mask, LayoutConstraints* con)
      : allocated_fields(mask), constraints(con), owner(nullptr), total_dims(0)
    //--------------------------------------------------------------------------
    {
      constraints->add_base_gc_ref(LAYOUT_DESC_REF);
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
    void LayoutDescription::log_instance_layout(LgEvent inst_event) const
    //--------------------------------------------------------------------------
    {
      if (spy_logging_level == NO_SPY_LOGGING)
        return;
      for (const std::pair<const FieldID, unsigned>& it : field_indexes)
        LegionSpy::log_physical_instance_field(inst_event, it.first);
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::compute_copy_offsets(
        const FieldMask& copy_mask, const PhysicalInstance instance,
        std::vector<CopySrcDstField>& fields)
    //--------------------------------------------------------------------------
    {
      uint64_t hash_key = copy_mask.get_hash_key();
      bool found_in_cache = false;
      FieldMask compressed;
      // First check to see if we've memoized this result
      {
        AutoLock o_lock(layout_lock, false /*exclusive*/);
        std::map<
            LEGION_FIELD_MASK_FIELD_TYPE,
            lng::list<std::pair<FieldMask, FieldMask> > >::const_iterator
            finder = comp_cache.find(hash_key);
        if (finder != comp_cache.end())
        {
          for (lng::list<std::pair<FieldMask, FieldMask> >::const_iterator it =
                   finder->second.begin();
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
        compress_mask<STATIC_LOG2(LEGION_MAX_FIELDS)>(
            compressed, allocated_fields);
        // Save the result in the cache, duplicates from races here are benign
        AutoLock o_lock(layout_lock);
        comp_cache[hash_key].emplace_back(
            std::pair<FieldMask, FieldMask>(copy_mask, compressed));
      }
      // It is absolutely imperative that these infos be added in
      // the order in which they appear in the field mask so that
      // they line up in the same order with the source/destination infos
      // (depending on the calling context of this function
      const unsigned pop_count = FieldMask::pop_count(compressed);
      legion_assert(pop_count == FieldMask::pop_count(copy_mask));
      unsigned offset = fields.size();
      fields.resize(offset + pop_count);
      int next_start = 0;
      for (unsigned idx = 0; idx < pop_count; idx++)
      {
        int index = compressed.find_next_set(next_start);
        CopySrcDstField& field = fields[offset + idx];
        field = field_infos[index];
        // Our field infos are annonymous so specify the instance now
        field.inst = instance;
        // We'll start looking again at the next index after this one
        next_start = index + 1;
      }
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::compute_copy_offsets(
        const std::vector<FieldID>& copy_fields,
        const PhysicalInstance instance, std::vector<CopySrcDstField>& fields)
    //--------------------------------------------------------------------------
    {
      unsigned offset = fields.size();
      fields.resize(offset + copy_fields.size());
      for (unsigned idx = 0; idx < copy_fields.size(); idx++)
      {
        std::map<FieldID, unsigned>::const_iterator finder =
            field_indexes.find(copy_fields[idx]);
        legion_assert(finder != field_indexes.end());
        CopySrcDstField& info = fields[offset + idx];
        info = field_infos[finder->second];
        // Since instances are annonymous in layout descriptions we
        // have to fill them in when we add the field info
        info.inst = instance;
      }
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::get_fields(std::set<FieldID>& fields) const
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const FieldID, unsigned>& it : field_indexes)
        fields.insert(it.first);
    }

    //--------------------------------------------------------------------------
    bool LayoutDescription::has_field(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      return (field_indexes.find(fid) != field_indexes.end());
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::has_fields(std::map<FieldID, bool>& to_test) const
    //--------------------------------------------------------------------------
    {
      for (std::pair<const FieldID, bool>& test_pair : to_test)
      {
        if (field_indexes.find(test_pair.first) != field_indexes.end())
          test_pair.second = true;
        else
          test_pair.second = false;
      }
    }

    //--------------------------------------------------------------------------
    void LayoutDescription::remove_space_fields(std::set<FieldID>& filter) const
    //--------------------------------------------------------------------------
    {
      std::vector<FieldID> to_remove;
      for (const FieldID& field_id : filter)
      {
        if (field_indexes.find(field_id) != field_indexes.end())
          to_remove.emplace_back(field_id);
      }
      if (!to_remove.empty())
      {
        for (const FieldID& field_id : to_remove) filter.erase(field_id);
      }
    }

    //--------------------------------------------------------------------------
    const CopySrcDstField& LayoutDescription::find_field_info(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      std::map<FieldID, unsigned>::const_iterator finder =
          field_indexes.find(fid);
      legion_assert(finder != field_indexes.end());
      return field_infos[finder->second];
    }

    //--------------------------------------------------------------------------
    size_t LayoutDescription::get_total_field_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      // Add up all the field sizes
      for (const CopySrcDstField& it : field_infos) result += it.size;
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
        std::vector<PhysicalInstance::DestroyedField>& serdez_fields) const
    //--------------------------------------------------------------------------
    {
      // See if we have any special fields which need serdez deletion
      for (const CopySrcDstField& it : field_infos)
      {
        if (it.serdez_id > 0)
          serdez_fields.emplace_back(PhysicalInstance::DestroyedField(
              it.field_id, it.size, it.serdez_id));
      }
    }

    //--------------------------------------------------------------------------
    bool LayoutDescription::match_layout(
        const LayoutConstraintSet& candidate_constraints,
        unsigned num_dims) const
    //--------------------------------------------------------------------------
    {
      if (num_dims != total_dims)
        return false;
      // We need to check equality on the entire constraint sets
      return *constraints == candidate_constraints;
    }

    //--------------------------------------------------------------------------
    bool LayoutDescription::match_layout(
        const LayoutDescription* layout, unsigned num_dims) const
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
    void LayoutDescription::pack_layout_description(
        Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      rez.serialize(constraints->layout_id);
    }

    //--------------------------------------------------------------------------
    /*static*/ LayoutDescription*
        LayoutDescription::handle_unpack_layout_description(
            LayoutConstraints* constraints, FieldSpaceNode* field_space_node,
            size_t total_dims)
    //--------------------------------------------------------------------------
    {
      legion_assert(constraints != nullptr);
      FieldMask instance_mask;
      const std::vector<FieldID>& field_set =
          constraints->field_constraint.get_field_set();
      std::vector<size_t> field_sizes(field_set.size());
      std::vector<unsigned> mask_index_map(field_set.size());
      std::vector<CustomSerdezID> serdez(field_set.size());
      field_space_node->compute_field_layout(
          field_set, field_sizes, mask_index_map, serdez, instance_mask);
      LayoutDescription* result = field_space_node->create_layout_description(
          instance_mask, total_dims, constraints, mask_index_map, field_set,
          field_sizes, serdez);
      legion_assert(result != nullptr);
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Layout Constraints
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LayoutConstraints::LayoutConstraints(
        LayoutConstraintID lay_id, FieldSpace h, bool inter, DistributedID did)
      : LayoutConstraintSet(),
        DistributedCollectable(
            LEGION_DISTRIBUTED_HELP_ENCODE(
                (did > 0) ? did : runtime->get_available_distributed_id(),
                CONSTRAINT_SET_DC),
            false /*register*/),
        layout_id(lay_id), handle(h), internal(inter), constraints_name(nullptr)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info(
          "GC Constraints %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    LayoutConstraints::LayoutConstraints(
        LayoutConstraintID lay_id, const LayoutConstraintRegistrar& registrar,
        bool inter, DistributedID did, CollectiveMapping* collective_mapping)
      : LayoutConstraintSet(registrar.layout_constraints),
        DistributedCollectable(
            LEGION_DISTRIBUTED_HELP_ENCODE(
                (did > 0) ? did : runtime->get_available_distributed_id(),
                CONSTRAINT_SET_DC),
            false /*register with runtime*/, collective_mapping),
        layout_id(lay_id), handle(registrar.handle), internal(inter)
    //--------------------------------------------------------------------------
    {
      if (registrar.layout_name == nullptr)
      {
        constraints_name = (char*)malloc(64 * sizeof(char));
        snprintf(constraints_name, 64, "layout constraints %ld", layout_id);
      }
      else
        constraints_name = strdup(registrar.layout_name);
#ifdef LEGION_GC
      log_garbage.info(
          "GC Constraints %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    LayoutConstraints::LayoutConstraints(
        LayoutConstraintID lay_id, const LayoutConstraintSet& cons,
        FieldSpace h, bool inter)
      : LayoutConstraintSet(cons),
        DistributedCollectable(
            LEGION_DISTRIBUTED_HELP_ENCODE(
                runtime->get_available_distributed_id(), CONSTRAINT_SET_DC),
            false /*register with runtime*/),
        layout_id(lay_id), handle(h), internal(inter)
    //--------------------------------------------------------------------------
    {
      constraints_name = (char*)malloc(64 * sizeof(char));
      snprintf(constraints_name, 64, "layout constraints %ld", layout_id);
#ifdef LEGION_GC
      log_garbage.info(
          "GC Constraints %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    LayoutConstraints::~LayoutConstraints(void)
    //--------------------------------------------------------------------------
    {
      if (constraints_name != nullptr)
        free(constraints_name);
    }

    //--------------------------------------------------------------------------
    bool LayoutConstraints::operator==(const LayoutConstraints& rhs) const
    //--------------------------------------------------------------------------
    {
      // We check equalities only on the members of LayoutConstraintSet
      return equals(rhs);
    }

    //--------------------------------------------------------------------------
    bool LayoutConstraints::operator==(const LayoutConstraintSet& rhs) const
    //--------------------------------------------------------------------------
    {
      // We check equalities only on the members of LayoutConstraintSet
      return equals(rhs);
    }

    //--------------------------------------------------------------------------
    void LayoutConstraints::notify_local(void)
    //--------------------------------------------------------------------------
    {
      runtime->unregister_layout(layout_id);
    }

    //--------------------------------------------------------------------------
    void LayoutConstraints::send_constraint_response(
        AddressSpaceID target, RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      ConstraintResponse rez;
      {
        RezCheck z(rez);
        rez.serialize(layout_id);
        rez.serialize(did);
        rez.serialize(handle);
        rez.serialize<bool>(internal);
        size_t name_len = strlen(constraints_name) + 1;
        rez.serialize(name_len);
        rez.serialize(constraints_name, name_len);
        // pack the constraints
        serialize(rez);
        // pack the done events
        rez.serialize(done_event);
      }
      rez.dispatch(target);
      update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    void LayoutConstraints::update_constraints(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      legion_assert(constraints_name == nullptr);
      size_t name_len;
      derez.deserialize(name_len);
      constraints_name = (char*)malloc(name_len);
      derez.deserialize(constraints_name, name_len);
      // unpack the constraints
      deserialize(derez);
    }

    //--------------------------------------------------------------------------
    bool LayoutConstraints::entails(
        LayoutConstraints* constraints, unsigned total_dims,
        const LayoutConstraint** failed_constraint, bool test_pointer)
    //--------------------------------------------------------------------------
    {
      const std::pair<LayoutConstraintID, unsigned> key(
          constraints->layout_id, total_dims);
      // Check to see if the result is in the cache
      if (test_pointer)
      {
        AutoLock lay(layout_lock, false /*exclusive*/);
        std::map<
            std::pair<LayoutConstraintID, unsigned>,
            const LayoutConstraint*>::const_iterator finder =
            entailment_cache.find(key);
        if (finder != entailment_cache.end())
        {
          if (finder->second != nullptr)
          {
            if (failed_constraint != nullptr)
              *failed_constraint = finder->second;
            return false;
          }
          else
            return true;
        }
      }
      else
      {
        AutoLock lay(layout_lock, false /*exclusive*/);
        std::map<
            std::pair<LayoutConstraintID, unsigned>,
            const LayoutConstraint*>::const_iterator finder =
            no_pointer_entailment_cache.find(key);
        if (finder != no_pointer_entailment_cache.end())
        {
          if (finder->second != nullptr)
          {
            if (failed_constraint != nullptr)
              *failed_constraint = finder->second;
            return false;
          }
          else
            return true;
        }
      }
      // Didn't find it, so do the test for real
      const LayoutConstraint* result = nullptr;
      const bool entailment =
          entails(*constraints, total_dims, &result, test_pointer);
      legion_assert(
          entailment ^ (result != nullptr));  // only one should be true
      if (!entailment && (failed_constraint != nullptr))
        *failed_constraint = result;
      // Save the result in the cache
      AutoLock lay(layout_lock);
      if (test_pointer)
        entailment_cache[key] = result;
      else
        no_pointer_entailment_cache[key] = result;
      return entailment;
    }

    //--------------------------------------------------------------------------
    bool LayoutConstraints::entails(
        const LayoutConstraintSet& other, unsigned total_dims,
        const LayoutConstraint** failed_constraint, bool test_pointer) const
    //--------------------------------------------------------------------------
    {
      return LayoutConstraintSet::entails(
          other, total_dims, failed_constraint, test_pointer);
    }

    //--------------------------------------------------------------------------
    bool LayoutConstraints::conflicts(
        LayoutConstraints* constraints, unsigned total_dims,
        const LayoutConstraint** conflict_constraint)
    //--------------------------------------------------------------------------
    {
      const std::pair<LayoutConstraintID, unsigned> key(
          constraints->layout_id, total_dims);
      // Check to see if the result is in the cache
      {
        AutoLock lay(layout_lock, false /*exclusive*/);
        std::map<
            std::pair<LayoutConstraintID, unsigned>,
            const LayoutConstraint*>::const_iterator finder =
            conflict_cache.find(key);
        if (finder != conflict_cache.end())
        {
          if (finder->second != nullptr)
          {
            if (conflict_constraint != nullptr)
              *conflict_constraint = finder->second;
            return true;
          }
          else
            return false;
        }
      }
      // Didn't find it, so do the test for real
      const LayoutConstraint* result = nullptr;
      const bool conflicted = conflicts(*constraints, total_dims, &result);
      legion_assert(
          conflicted ^ (result == nullptr));  // only one should be true
      // Save the result in the cache
      AutoLock lay(layout_lock);
      conflict_cache[key] = result;
      if (conflicted && (conflict_constraint != nullptr))
        *conflict_constraint = result;
      return conflicted;
    }

    //--------------------------------------------------------------------------
    bool LayoutConstraints::conflicts(
        const LayoutConstraintSet& other, unsigned total_dims,
        const LayoutConstraint** conflict_constraint) const
    //--------------------------------------------------------------------------
    {
      return LayoutConstraintSet::conflicts(
          other, total_dims, conflict_constraint);
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID LayoutConstraints::get_owner_space(
        LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      return (layout_id % runtime->total_address_spaces);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ConstraintRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LayoutConstraintID lay_id;
      derez.deserialize(lay_id);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      bool can_fail;
      derez.deserialize(can_fail);
      LayoutConstraints* constraints =
          runtime->find_layout_constraints(lay_id, can_fail);
      if (can_fail && (constraints == nullptr))
        Runtime::trigger_event(done_event);
      else
        constraints->send_constraint_response(source, done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ConstraintResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LayoutConstraintID lay_id;
      derez.deserialize(lay_id);
      DistributedID did;
      derez.deserialize(did);
      FieldSpace handle;
      derez.deserialize(handle);
      bool internal;
      derez.deserialize(internal);
      // Make it an unpack it, then try to register it
      LayoutConstraints* new_constraints =
          new LayoutConstraints(lay_id, handle, internal, did);
      new_constraints->update_constraints(derez);
      // Now try to register this with the runtime
      if (!runtime->register_layout(new_constraints))
        delete new_constraints;
      // Trigger our done event and then return it
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
    }

  }  // namespace Internal
}  // namespace Legion
