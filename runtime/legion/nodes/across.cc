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

#include "legion/nodes/across.h"
#include "legion/kernel/runtime.h"
#include "legion/instances/physical.h"
#include "legion/nodes/index.h"
#include "legion/nodes/field.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Indirect Record
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndirectRecord::IndirectRecord(
        const RegionRequirement& req, const InstanceSet& insts,
        size_t total_points)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* is = runtime->get_node(req.region.get_index_space());
      ApUserEvent to_trigger;
      domain_ready = is->get_loose_domain(domain, to_trigger);
      // This call adds 'total_points' references to the sparsity map of
      // the domain (if there is one). Each point will then make a
      // CopyAcrossUnstructured object that will own a reference and then
      // remove the reference when the CopyAcrossUnstructured object is
      // deleted. Note this is necessary for handling tracing cases where
      // the CopyAcrossUnstructure object can outlive the operation that
      // created it and we need to keep the sparsity maps alive.
      RtEvent added;
      if (!domain.dense())
        added = is->add_sparsity_map_references(domain, total_points);
      index_space = req.region.get_index_space();
      FieldSpaceNode* fs = runtime->get_node(req.region.get_field_space());
      std::vector<unsigned> field_indexes(req.instance_fields.size());
      fs->get_field_indexes(req.instance_fields, field_indexes);
      instances.resize(field_indexes.size());
      if ((runtime->profiler != nullptr) ||
          (spy_logging_level > NO_SPY_LOGGING))
        instance_events.resize(field_indexes.size());
      // For each of the fields in the region requirement
      // (importantly in the order they will be copied)
      // find the corresponding instance and store them
      // in the indirect record
      for (unsigned fidx = 0; fidx < field_indexes.size(); fidx++)
      {
        [[maybe_unused]] bool found = false;
        for (unsigned idx = 0; idx < insts.size(); idx++)
        {
          const InstanceRef& ref = insts[idx];
          const FieldMask& mask = ref.get_valid_fields();
          if (!mask.is_set(field_indexes[fidx]))
            continue;
          PhysicalManager* manager = ref.get_physical_manager();
          instances[fidx] = manager->get_instance();
          if (!instance_events.empty())
            instance_events[fidx] = manager->get_unique_event();
          found = true;
          break;
        }
        legion_assert(found);
      }
      // Wait for the sparsity map references to be added if necessary
      if (added.exists() && !added.has_triggered())
        added.wait();
      // If we had a to_trigger event we can trigger it now since we added
      // our own references to the sparsity map at this point to keep it alive
      if (to_trigger.exists())
        Runtime::trigger_event_untraced(to_trigger);
    }

    //--------------------------------------------------------------------------
    void IndirectRecord::serialize(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(domain);
      rez.serialize(domain_ready);
      rez.serialize<size_t>(instances.size());
      for (const PhysicalInstance& instance : instances)
        rez.serialize(instance);
      rez.serialize<size_t>(instance_events.size());
      for (const LgEvent& event : instance_events) rez.serialize(event);
      rez.serialize(index_space);
    }

    //--------------------------------------------------------------------------
    void IndirectRecord::deserialize(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(domain);
      derez.deserialize(domain_ready);
      size_t num_instances;
      derez.deserialize(num_instances);
      instances.resize(num_instances);
      for (unsigned idx = 0; idx < num_instances; idx++)
        derez.deserialize(instances[idx]);
      size_t num_events;
      derez.deserialize(num_events);
      instance_events.resize(num_events);
      for (unsigned idx = 0; idx < num_events; idx++)
        derez.deserialize(instance_events[idx]);
      derez.deserialize(index_space);
    }

    /////////////////////////////////////////////////////////////
    // Copy Across Helper
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void CopyAcrossHelper::compute_across_offsets(
        const FieldMask& src_mask, std::vector<CopySrcDstField>& dst_fields)
    //--------------------------------------------------------------------------
    {
      FieldMask compressed;
      bool found_in_cache = false;
      for (const std::pair<FieldMask, FieldMask>& cache_entry :
           compressed_cache)
      {
        if (cache_entry.first == src_mask)
        {
          compressed = cache_entry.second;
          found_in_cache = true;
          break;
        }
      }
      if (!found_in_cache)
      {
        compressed = src_mask;
        compress_mask<STATIC_LOG2(LEGION_MAX_FIELDS)>(compressed, full_mask);
        compressed_cache.emplace_back(
            std::pair<FieldMask, FieldMask>(src_mask, compressed));
      }
      const unsigned pop_count = FieldMask::pop_count(compressed);
      legion_assert(pop_count == FieldMask::pop_count(src_mask));
      unsigned offset = dst_fields.size();
      dst_fields.resize(offset + pop_count);
      int next_start = 0;
      for (unsigned idx = 0; idx < pop_count; idx++)
      {
        int index = compressed.find_next_set(next_start);
        legion_assert(index >= 0);
        dst_fields[offset + idx] = offsets[index];
        // We'll start looking again at the next index after this one
        next_start = index + 1;
      }
    }

    //--------------------------------------------------------------------------
    FieldMask CopyAcrossHelper::convert_src_to_dst(const FieldMask& src_mask)
    //--------------------------------------------------------------------------
    {
      FieldMask dst_mask;
      if (!src_mask)
        return dst_mask;
      if (forward_map.empty())
      {
        legion_assert(src_indexes.size() == dst_indexes.size());
        for (unsigned idx = 0; idx < src_indexes.size(); idx++)
        {
          legion_assert(
              forward_map.find(src_indexes[idx]) == forward_map.end());
          forward_map[src_indexes[idx]] = dst_indexes[idx];
        }
      }
      int index = src_mask.find_first_set();
      while (index >= 0)
      {
        legion_assert(forward_map.find(index) != forward_map.end());
        dst_mask.set_bit(forward_map[index]);
        index = src_mask.find_next_set(index + 1);
      }
      return dst_mask;
    }

    //--------------------------------------------------------------------------
    FieldMask CopyAcrossHelper::convert_dst_to_src(const FieldMask& dst_mask)
    //--------------------------------------------------------------------------
    {
      FieldMask src_mask;
      if (!dst_mask)
        return src_mask;
      if (backward_map.empty())
      {
        legion_assert(src_indexes.size() == dst_indexes.size());
        for (unsigned idx = 0; idx < dst_indexes.size(); idx++)
        {
          legion_assert(
              backward_map.find(dst_indexes[idx]) == backward_map.end());
          backward_map[dst_indexes[idx]] = src_indexes[idx];
        }
      }
      int index = dst_mask.find_first_set();
      while (index >= 0)
      {
        legion_assert(backward_map.find(index) != backward_map.end());
        src_mask.set_bit(backward_map[index]);
        index = dst_mask.find_next_set(index + 1);
      }
      return src_mask;
    }

    //--------------------------------------------------------------------------
    unsigned CopyAcrossHelper::convert_src_to_dst(unsigned index)
    //--------------------------------------------------------------------------
    {
      if (forward_map.empty())
      {
        legion_assert(src_indexes.size() == dst_indexes.size());
        for (unsigned idx = 0; idx < src_indexes.size(); idx++)
        {
          legion_assert(
              forward_map.find(src_indexes[idx]) == forward_map.end());
          forward_map[src_indexes[idx]] = dst_indexes[idx];
        }
      }
      legion_assert(forward_map.find(index) != forward_map.end());
      return forward_map[index];
    }

    //--------------------------------------------------------------------------
    unsigned CopyAcrossHelper::convert_dst_to_src(unsigned index)
    //--------------------------------------------------------------------------
    {
      if (backward_map.empty())
      {
        legion_assert(src_indexes.size() == dst_indexes.size());
        for (unsigned idx = 0; idx < dst_indexes.size(); idx++)
        {
          legion_assert(
              backward_map.find(dst_indexes[idx]) == backward_map.end());
          backward_map[dst_indexes[idx]] = src_indexes[idx];
        }
      }
      legion_assert(backward_map.find(index) != backward_map.end());
      return backward_map[index];
    }

    /////////////////////////////////////////////////////////////
    // Copy Across Executor
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CopyAcrossExecutor::DeferCopyAcrossArgs::DeferCopyAcrossArgs(
        CopyAcrossExecutor* e, Operation* o, PredEvent g, ApEvent copy_pre,
        ApEvent src_pre, ApEvent dst_pre, bool repl, bool recurrent, unsigned s)
      : LgTaskArgs<DeferCopyAcrossArgs>(false, false), executor(e), op(o),
        guard(g), copy_precondition(copy_pre),
        src_indirect_precondition(src_pre), dst_indirect_precondition(dst_pre),
        done_event(Runtime::create_ap_user_event(nullptr)), stage(s + 1),
        replay(repl), recurrent_replay(recurrent)
    //--------------------------------------------------------------------------
    {
      executor->add_reference();
    }

    //--------------------------------------------------------------------------
    void CopyAcrossExecutor::DeferCopyAcrossArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      // Dummy trace info since we can't be tracing if we're here
      const PhysicalTraceInfo trace_info(op, -1U);
      Runtime::trigger_event_untraced(
          done_event, executor->execute(
                          op, guard, copy_precondition,
                          src_indirect_precondition, dst_indirect_precondition,
                          trace_info, replay, recurrent_replay, stage));
      if (executor->remove_reference())
        delete executor;
    }

    /////////////////////////////////////////////////////////////
    // Copy Across Unstructured
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CopyAcrossUnstructured::~CopyAcrossUnstructured(void)
    //--------------------------------------------------------------------------
    {
      // Need to release the sparsity map references being held by the
      // indirect records
      for (IndirectRecord& record : src_indirections)
        record.domain.destroy(last_copy);
      for (IndirectRecord& record : dst_indirections)
        record.domain.destroy(last_copy);
    }

    //--------------------------------------------------------------------------
    void CopyAcrossUnstructured::initialize_source_fields(
        const RegionRequirement& req, const InstanceSet& insts,
        const PhysicalTraceInfo& trace_info)
    //--------------------------------------------------------------------------
    {
      legion_assert(src_fields.empty());
      FieldSpaceNode* fs = runtime->get_node(req.region.get_field_space());
      std::vector<unsigned> indexes(req.instance_fields.size());
      fs->get_field_indexes(req.instance_fields, indexes);
      src_fields.reserve(indexes.size());
      src_unique_events.reserve(indexes.size());
      for (const unsigned& field_index : indexes)
      {
        [[maybe_unused]] bool found = false;
        for (unsigned idx = 0; idx < insts.size(); idx++)
        {
          const InstanceRef& ref = insts[idx];
          const FieldMask& mask = ref.get_valid_fields();
          if (!mask.is_set(field_index))
            continue;
          FieldMask copy_mask;
          copy_mask.set_bit(field_index);
          PhysicalManager* manager = ref.get_physical_manager();
          manager->compute_copy_offsets(copy_mask, src_fields);
          src_unique_events.emplace_back(manager->get_unique_event());
          found = true;
          break;
        }
        legion_assert(found);
      }
    }

    //--------------------------------------------------------------------------
    void CopyAcrossUnstructured::initialize_destination_fields(
        const RegionRequirement& req, const InstanceSet& insts,
        const PhysicalTraceInfo& trace_info, const bool exclusive_redop)
    //--------------------------------------------------------------------------
    {
      legion_assert(dst_fields.empty());
      FieldSpaceNode* fs = runtime->get_node(req.region.get_field_space());
      std::vector<unsigned> indexes(req.instance_fields.size());
      fs->get_field_indexes(req.instance_fields, indexes);
      dst_fields.reserve(indexes.size());
      dst_unique_events.reserve(indexes.size());
      for (const unsigned& field_index : indexes)
      {
        [[maybe_unused]] bool found = false;
        for (unsigned idx = 0; idx < insts.size(); idx++)
        {
          const InstanceRef& ref = insts[idx];
          const FieldMask& mask = ref.get_valid_fields();
          if (!mask.is_set(field_index))
            continue;
          FieldMask copy_mask;
          copy_mask.set_bit(field_index);
          PhysicalManager* manager = ref.get_physical_manager();
          manager->compute_copy_offsets(copy_mask, dst_fields);
          dst_unique_events.emplace_back(manager->get_unique_event());
          found = true;
          break;
        }
        legion_assert(found);
      }
      if (req.redop != 0)
      {
        for (CopySrcDstField& dst_field : dst_fields)
          dst_field.set_redop(req.redop, false /*fold*/, exclusive_redop);
      }
    }

    //--------------------------------------------------------------------------
    void CopyAcrossUnstructured::initialize_source_indirections(
        std::vector<IndirectRecord>& records, const RegionRequirement& src_req,
        const RegionRequirement& idx_req, const InstanceRef& indirect_instance,
        const bool are_range, const bool possible_out_of_range)
    //--------------------------------------------------------------------------
    {
      legion_assert(src_fields.empty());
      legion_assert(idx_req.privilege_fields.size() == 1);
      src_indirections.swap(records);
      src_indirect_field = *(idx_req.privilege_fields.begin());
      PhysicalManager* manager = indirect_instance.get_physical_manager();
      src_indirect_instance = manager->get_instance();
      src_indirect_instance_event = manager->get_unique_event();
      src_indirect_type = src_req.region.get_index_space().get_type_tag();
      is_range_indirection = are_range;
      possible_src_out_of_range = possible_out_of_range;
      src_fields.resize(src_req.instance_fields.size());
      FieldSpaceNode* fs = runtime->get_node(src_req.region.get_field_space());
      for (unsigned idx = 0; idx < src_fields.size(); idx++)
      {
        const FieldID fid = src_req.instance_fields[idx];
        src_fields[idx].set_indirect(
            0 /*dummy indirection for now*/, fid, fs->get_field_size(fid));
      }
    }

    //--------------------------------------------------------------------------
    void CopyAcrossUnstructured::initialize_destination_indirections(
        std::vector<IndirectRecord>& records, const RegionRequirement& dst_req,
        const RegionRequirement& idx_req, const InstanceRef& indirect_instance,
        const bool are_range, const bool possible_out_of_range,
        const bool possible_aliasing, const bool exclusive_redop)
    //--------------------------------------------------------------------------
    {
      legion_assert(dst_fields.empty());
      legion_assert(idx_req.privilege_fields.size() == 1);
      dst_indirections.swap(records);
      dst_indirect_field = *(idx_req.privilege_fields.begin());
      PhysicalManager* manager = indirect_instance.get_physical_manager();
      dst_indirect_instance = manager->get_instance();
      dst_indirect_instance_event = manager->get_unique_event();
      dst_indirect_type = dst_req.region.get_index_space().get_type_tag();
      is_range_indirection = are_range;
      possible_dst_out_of_range = possible_out_of_range;
      possible_dst_aliasing = possible_aliasing;
      dst_fields.resize(dst_req.instance_fields.size());
      FieldSpaceNode* fs = runtime->get_node(dst_req.region.get_field_space());
      for (unsigned idx = 0; idx < dst_fields.size(); idx++)
      {
        const FieldID fid = dst_req.instance_fields[idx];
        dst_fields[idx].set_indirect(
            0 /*dummy indirection for now*/, fid, fs->get_field_size(fid));
        if (dst_req.redop != 0)
          dst_fields[idx].set_redop(
              dst_req.redop, false /*fold*/, exclusive_redop);
      }
    }

    //--------------------------------------------------------------------------
    LgEvent CopyAcrossUnstructured::find_instance_name(
        PhysicalInstance instance) const
    //--------------------------------------------------------------------------
    {
      if (instance == src_indirect_instance)
        return src_indirect_instance_event;
      if (instance == dst_indirect_instance)
        return dst_indirect_instance_event;
      for (unsigned idx = 0; idx < src_fields.size(); idx++)
        if (src_fields[idx].inst == instance)
          return src_unique_events[idx];
      for (unsigned idx = 0; idx < dst_fields.size(); idx++)
        if (dst_fields[idx].inst == instance)
          return dst_unique_events[idx];
      for (const IndirectRecord& record : src_indirections)
        for (unsigned idx = 0; idx < record.instances.size(); idx++)
          if (record.instances[idx] == instance)
            return record.instance_events[idx];
      for (const IndirectRecord& record : dst_indirections)
        for (unsigned idx = 0; idx < record.instances.size(); idx++)
          if (record.instances[idx] == instance)
            return record.instance_events[idx];
      AutoLock p_lock(preimage_lock, false /*exclusive*/);
      std::map<PhysicalInstance, LgEvent>::const_iterator finder =
          profiling_shadow_instances.find(instance);
      if (finder != profiling_shadow_instances.end())
        return finder->second;
      // Should always have found it before this
      std::abort();
    }

    //--------------------------------------------------------------------------
    DistributedID CopyAcrossUnstructured::find_instance_subspace(
        PhysicalInstance instance) const
    //--------------------------------------------------------------------------
    {
      if (instance == src_indirect_instance)
        return expr->get_distributed_id();
      if (instance == dst_indirect_instance)
        return expr->get_distributed_id();
      for (unsigned idx = 0; idx < src_fields.size(); idx++)
        if (src_fields[idx].inst == instance)
          return expr->get_distributed_id();
      for (unsigned idx = 0; idx < dst_fields.size(); idx++)
        if (dst_fields[idx].inst == instance)
          return expr->get_distributed_id();
      for (const IndirectRecord& record : src_indirections)
        for (unsigned idx = 0; idx < record.instances.size(); idx++)
          if (record.instances[idx] == instance)
            return record.index_space.get_id(false /*ftiler*/);
      for (const IndirectRecord& record : dst_indirections)
        for (unsigned idx = 0; idx < record.instances.size(); idx++)
          if (record.instances[idx] == instance)
            return record.index_space.get_id(false /*filter*/);
      AutoLock p_lock(preimage_lock, false /*exclusive*/);
      std::map<PhysicalInstance, LgEvent>::const_iterator finder =
          profiling_shadow_instances.find(instance);
      if (finder != profiling_shadow_instances.end())
        return expr->get_distributed_id();
      // Should always have found it before this
      std::abort();
    }

    //--------------------------------------------------------------------------
    DistributedID CopyAcrossUnstructured::find_copy_expression(void) const
    //--------------------------------------------------------------------------
    {
      return expr->record_profiler_expression();
    }

    //--------------------------------------------------------------------------
    ReductionOpID CopyAcrossUnstructured::find_redop(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(!dst_fields.empty());
      return dst_fields.back().redop_id;
    }

  }  // namespace Internal
}  // namespace Legion
