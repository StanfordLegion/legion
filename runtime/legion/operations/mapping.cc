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

#include "legion/operations/mapping.h"
#include "legion/contexts/replicate.h"
#include "legion/api/physical_region_impl.h"
#include "legion/managers/mapper.h"
#include "legion/nodes/field.h"
#include "legion/utilities/collectives.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // External Mapping
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalMapping::ExternalMapping(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ExternalMapping::pack_external_mapping(
        Serializer& rez, AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_region_requirement(requirement, rez);
      rez.serialize(grants.size());
      for (unsigned idx = 0; idx < grants.size(); idx++)
        pack_grant(grants[idx], rez);
      rez.serialize(wait_barriers.size());
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        pack_phase_barrier(wait_barriers[idx], rez);
      rez.serialize(arrive_barriers.size());
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        pack_phase_barrier(arrive_barriers[idx], rez);
      rez.serialize(layout_constraint_id);
      pack_mappable(*this, rez);
      rez.serialize(get_context_index());
    }

    //--------------------------------------------------------------------------
    void ExternalMapping::unpack_external_mapping(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      unpack_region_requirement(requirement, derez);
      size_t num_grants;
      derez.deserialize(num_grants);
      grants.resize(num_grants);
      for (unsigned idx = 0; idx < grants.size(); idx++)
        unpack_grant(grants[idx], derez);
      size_t num_wait_barriers;
      derez.deserialize(num_wait_barriers);
      wait_barriers.resize(num_wait_barriers);
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        unpack_phase_barrier(wait_barriers[idx], derez);
      size_t num_arrive_barriers;
      derez.deserialize(num_arrive_barriers);
      arrive_barriers.resize(num_arrive_barriers);
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        unpack_phase_barrier(arrive_barriers[idx], derez);
      derez.deserialize(layout_constraint_id);
      unpack_mappable(*this, derez);
      uint64_t index;
      derez.deserialize(index);
      set_context_index(index);
    }

    /////////////////////////////////////////////////////////////
    // Map Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MapOp::MapOp(void) : ExternalMapping(), Operation()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    MapOp::~MapOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PhysicalRegion MapOp::initialize(
        InnerContext* ctx, const InlineLauncher& launcher,
        Provenance* provenance, unsigned requirement_index)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_operation(ctx, provenance);
      requirement = launcher.requirement;
      if (runtime->safe_model)
        verify_requirement(requirement);
      parent_req_index = ctx->find_parent_region_index(this, requirement);
      const ApUserEvent term_event = Runtime::create_ap_user_event(nullptr);
      region = PhysicalRegion(new PhysicalRegionImpl(
          requirement, get_mapped_event(), ready_event, term_event,
          true /*mapped*/, ctx, map_id, tag, false /*leaf*/,
          false /*virtual mapped*/, true /*collective for replication*/,
          ctx->get_next_blocking_index(), requirement_index));
      termination_event = term_event;
      grants = launcher.grants;
      // Register ourselves with all the grants
      for (unsigned idx = 0; idx < grants.size(); idx++)
        grants[idx].impl->register_operation(termination_event);
      wait_barriers = launcher.wait_barriers;
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        for (const PhaseBarrier& bar : launcher.arrive_barriers)
        {
          arrive_barriers.emplace_back(bar);
          LegionSpy::log_event_dependence(
              bar.phase_barrier, arrive_barriers.back().phase_barrier);
        }
      }
      else
        arrive_barriers = launcher.arrive_barriers;
      map_id = launcher.map_id;
      tag = launcher.tag;
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
        legion_assert(mapper_data == nullptr);
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      layout_constraint_id = launcher.layout_constraint_id;

      LegionSpy::log_mapping_operation(
          parent_ctx->get_unique_id(), unique_op_id);
      return region;
    }

    //--------------------------------------------------------------------------
    void MapOp::initialize(
        InnerContext* ctx, const PhysicalRegion& reg, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      parent_task = ctx->get_task();
      requirement = reg.impl->get_requirement();
      // Remove any discard masks we might have had
      requirement.privilege = FILTER_DISCARD(requirement);
      // No need to do verification, it's already been verified
      parent_req_index = ctx->find_parent_region_index(this, requirement);
      map_id = reg.impl->map_id;
      tag = reg.impl->tag;
      region = reg;
      termination_event = region.impl->remap_region(
          ready_event, ctx->get_next_blocking_index());
      remap_region = true;
      // No need to check the privileges here since we know that we have
      // them from the first time that we made this physical region
      LegionSpy::log_mapping_operation(
          parent_ctx->get_unique_id(), unique_op_id);
    }

    //--------------------------------------------------------------------------
    void MapOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
      parent_ctx = nullptr;
      remap_region = false;
      mapper = nullptr;
      layout_constraint_id = 0;
      ready_event = Runtime::create_ap_user_event(nullptr);
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      copy_fill_priority = 0;
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
    }

    //--------------------------------------------------------------------------
    void MapOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      // Remove our reference to the region
      region = PhysicalRegion();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      version_info.clear();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      atomic_locks.clear();
      map_applied_conditions.clear();
      profiling_requests.clear();
      if (mapper_data != nullptr)
      {
        free(mapper_data);
        mapper_data = nullptr;
        mapper_data_size = 0;
      }
      Operation::deactivate(false /*free*/);
      // Now return this operation to the queue
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* MapOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[MAP_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind MapOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return MAP_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t MapOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    Mappable* MapOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    void MapOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        LegionSpy::log_logical_requirement(
            unique_op_id, 0 /*index*/, true /*region*/,
            requirement.region.index_space.get_id(),
            requirement.region.field_space.get_id(),
            requirement.region.get_tree_id(), requirement.privilege,
            requirement.prop, requirement.redop,
            requirement.parent.index_space.get_id());
        LegionSpy::log_requirement_fields(
            unique_op_id, 0 /*index*/, requirement.privilege_fields);
      }
    }

    //--------------------------------------------------------------------------
    void MapOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(
            this, wait_barriers, arrive_barriers);
      analyze_region_requirements();
    }

    //--------------------------------------------------------------------------
    void MapOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Compute the version numbers for this mapping operation
      std::set<RtEvent> preconditions;
      perform_versioning_analysis(
          0 /*idx*/, requirement, version_info, preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void MapOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      const PhysicalTraceInfo trace_info(this, 0 /*index*/);
      // If we have any wait preconditions from phase barriers or
      // grants then we use them to compute a precondition for doing
      // any copies or anything else for this operation
      ApEvent init_precondition = execution_fence_event;
      if (!wait_barriers.empty() || !grants.empty())
      {
        ApEvent sync_precondition =
            merge_sync_preconditions(trace_info, grants, wait_barriers);
        if (sync_precondition.exists())
        {
          if (init_precondition.exists())
            init_precondition = Runtime::merge_events(
                &trace_info, init_precondition, sync_precondition);
          else
            init_precondition = sync_precondition;
        }
      }
      bool record_valid = true;
      InstanceSet mapped_instances;
      std::vector<PhysicalManager*> source_instances;
      // If we are remapping then we know the answer
      // so we don't need to do any premapping
      if (!remap_region)
      {
        // Now we've got the valid instances so invoke the mapper
        record_valid = invoke_mapper(mapped_instances, source_instances);
        // First mapping so set the references now
        region.impl->set_references(mapped_instances);
      }
      else
        region.impl->get_references(mapped_instances);
      // Then we can register our mapped instances
      ApEvent map_complete_event = physical_perform_updates_and_registration(
          requirement, version_info, 0 /*idx*/, init_precondition,
          termination_event, mapped_instances, source_instances, trace_info,
          map_applied_conditions, false /*no dynamic rendezvous*/,
          record_valid);
      legion_assert(
          IS_NO_ACCESS(requirement) || requirement.privilege_fields.empty() ||
          !mapped_instances.empty());
      log_mapping_decision(0 /*idx*/, requirement, mapped_instances);

      if (!atomic_locks.empty() || !arrive_barriers.empty())
      {
        // They've already been sorted in order
        for (const std::pair<const Reservation, bool>& it : atomic_locks)
        {
          map_complete_event = Runtime::acquire_ap_reservation(
              it.first, it.second, map_complete_event);
          // We can also issue the release condition on our termination
          Runtime::release_reservation(it.first, termination_event);
        }
        for (PhaseBarrier& barrier : arrive_barriers)
        {
          LegionSpy::log_phase_barrier_arrival(
              unique_op_id, barrier.phase_barrier);
          runtime->phase_barrier_arrive(
              barrier.phase_barrier, 1 /*count*/, termination_event);
        }
      }
      LegionSpy::log_operation_events(
          unique_op_id, map_complete_event, ready_event);
      // Map operations do not wait for the unmapping to be considered complete
      record_completion_effect(map_complete_event);
      // We can trigger the ready event now that we know its precondition
      Runtime::trigger_event_untraced(ready_event, map_complete_event);
      // Now we can trigger the mapping event and indicate
      // to all our mapping dependences that we are mapped.
      RtEvent mapping_applied;
      if (!map_applied_conditions.empty())
        mapping_applied = Runtime::merge_events(map_applied_conditions);
      if (!acquired_instances.empty())
        mapping_applied = release_nonempty_acquired_instances(
            mapping_applied, acquired_instances);
      complete_mapping(finalize_complete_mapping(mapping_applied));
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void MapOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if we need to do a profiling response
      if (profiling_reported.exists() && (outstanding_profiling_requests == 0))
      {
        // We're not expecting any profiling callbacks so we need to
        // do one ourself to inform the mapper that there won't be any
        Mapping::Mapper::InlineProfilingInfo info;
        info.total_reports = 0;
        info.fill_response = false;  // make valgrind happy
        mapper->invoke_inline_report_profiling(this, info);
        Runtime::trigger_event(profiling_reported);
      }
      // Don't commit this operation until we've reported our profiling
      commit_operation(true /*deactivate*/, profiling_reported);
    }

    //--------------------------------------------------------------------------
    unsigned MapOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx == 0);
      legion_assert(parent_req_index != TRACED_PARENT_INDEX);
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void MapOp::select_sources(
        const unsigned index, PhysicalManager* target,
        const std::vector<InstanceView*>& sources,
        std::vector<unsigned>& ranking,
        std::map<unsigned, PhysicalManager*>& points)
    //--------------------------------------------------------------------------
    {
      legion_assert(index == 0);
      Mapper::SelectInlineSrcInput input;
      Mapper::SelectInlineSrcOutput output;
      prepare_for_mapping(
          sources, input.source_instances, input.collective_views);
      prepare_for_mapping(target, input.target);
      if (mapper == nullptr)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_select_inline_sources(this, input, output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking, points);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*, unsigned>* MapOp::get_acquired_instances_ref(
        void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    void MapOp::update_atomic_locks(
        const unsigned index, Reservation lock, bool exclusive)
    //--------------------------------------------------------------------------
    {
      legion_assert(index == 0);
      AutoLock o_lock(op_lock);
      std::map<Reservation, bool>::iterator finder = atomic_locks.find(lock);
      if (finder != atomic_locks.end())
      {
        if (!finder->second && exclusive)
          finder->second = true;
      }
      else
        atomic_locks[lock] = exclusive;
    }

    //--------------------------------------------------------------------------
    UniqueID MapOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t MapOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void MapOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int MapOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* MapOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == nullptr)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const std::string_view& MapOp::get_provenance_string(bool human) const
    //--------------------------------------------------------------------------
    {
      Provenance* provenance = get_provenance();
      if (provenance != nullptr)
        return human ? provenance->human : provenance->machine;
      else
        return Provenance::no_provenance;
    }

    //--------------------------------------------------------------------------
    bool MapOp::invoke_mapper(
        InstanceSet& chosen_instances,
        std::vector<PhysicalManager*>& source_instances)
    //--------------------------------------------------------------------------
    {
      Mapper::MapInlineInput input;
      Mapper::MapInlineOutput output;
      // Invoke the mapper
      if (mapper == nullptr)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      if (mapper->request_valid_instances)
      {
        InstanceSet valid_instances;
        local::FieldMaskMap<ReplicatedView> collectives;
        physical_premap_region(
            0 /*idx*/, requirement, version_info, valid_instances, collectives,
            map_applied_conditions);
        if (!requirement.is_no_access())
        {
          std::set<Memory> visible_memories;
          runtime->find_visible_memories(
              parent_ctx->get_executing_processor(), visible_memories);
          prepare_for_mapping(
              valid_instances, collectives, visible_memories,
              input.valid_instances, input.valid_collectives);
        }
        else
          prepare_for_mapping(
              valid_instances, collectives, input.valid_instances,
              input.valid_collectives);
      }
      mapper->invoke_map_inline(this, input, output);
      copy_fill_priority = output.copy_fill_priority;
      if (!output.source_instances.empty())
        physical_convert_sources(
            requirement, output.source_instances, source_instances,
            runtime->safe_mapper ? &acquired_instances : nullptr);
      if (!output.profiling_requests.empty())
      {
        filter_copy_request_kinds(
            mapper, output.profiling_requests.requested_measurements,
            profiling_requests, true /*warn*/);
        profiling_priority = output.profiling_priority;
        legion_assert(!profiling_reported.exists());
        profiling_reported = Runtime::create_rt_user_event();
      }
      // Now we have to validate the output
      // Go through the instances and make sure we got one for every field
      // Also check to make sure that none of them are composite instances
      RegionTreeID bad_tree = 0;
      std::vector<FieldID> missing_fields;
      std::vector<PhysicalManager*> unacquired;
      int virtual_index = physical_convert_mapping(
          requirement, output.chosen_instances, chosen_instances, bad_tree,
          missing_fields, &acquired_instances, unacquired,
          runtime->safe_mapper);
      if (bad_tree > 0)
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from invocation of 'map_inline' on "
                 "mapper "
              << *mapper << ". Mapper selected instance from region tree "
              << bad_tree << " to satisfy a region requirement for " << *this
              << " whose region tree is " << requirement.region.get_tree_id()
              << ".";
        error.raise();
      }
      if (!missing_fields.empty())
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from invocation of 'map_inline' on "
                 "mapper "
              << *mapper
              << ". Mapper failed to specify a physical instance for "
              << missing_fields.size() << " fields of the region "
              << "requirement for " << *this << ". The missing fields are: ";
        bool first = true;
        for (const FieldID& fid : missing_fields)
        {
          const void* name;
          size_t name_size;
          if (!runtime->retrieve_semantic_information(
                  requirement.region.get_field_space(), fid,
                  LEGION_NAME_SEMANTIC_TAG, name, name_size, true, false))
            name = "(no name)";
          if (first)
            first = false;
          else
            error << ", ";
          error << static_cast<const char*>(name) << "(FieldID: " << fid << ")";
        }
        error.raise();
      }
      if (!unacquired.empty())
      {
        for (PhysicalManager* manager : unacquired)
        {
          if (acquired_instances.find(manager) == acquired_instances.end())
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error
                << "Invalid mapper output from 'map_inline' invocation on "
                   "mapper "
                << *mapper << ". Mapper selected physical instance for "
                << *this << " which has already been collected. If the mapper "
                << "had properly acquired this instance as part of the mapper "
                   "call "
                << "it would have detected this. Please update the mapper to "
                   "abide "
                << "by proper mapping conventions.";
            error.raise();
          }
        }
        // If we did successfully acquire them, still issue the warning
        Warning warning;
        warning << "Mapper " << *mapper << "failed to acquire instance for "
                << *this
                << "in 'map_inline' call. You may experience undefined "
                << "behavior as a consequence.";
        warning.raise();
      }
      if (virtual_index >= 0)
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from invocation of 'map_inline' on "
                 "mapper "
              << *mapper << ". Mapper requested creation of a "
              << "virtual mapping for " << *this
              << ". Inline mapping operations "
              << "are not permitted to do perform virtual mappings.";
        error.raise();
      }
      if (!output.track_valid_region && !IS_READ_ONLY(requirement))
      {
        Warning warning;
        warning << "Ignoring request by mapper " << *mapper
                << " to not track valid instances for " << *this << " because "
                << "the region requirement does not have read-only privileges.";
        warning.raise();
        output.track_valid_region = true;
      }
      // If we are doing unsafe mapping, then we can return
      if (!runtime->safe_mapper)
        return output.track_valid_region;
      // If this requirement doesn't have a no access flag then we
      // need to check to make sure that the instances are visible
      if (!requirement.is_no_access())
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        std::set<Memory> visible_memories;
        runtime->find_visible_memories(exec_proc, visible_memories);
        for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
        {
          const Memory mem = chosen_instances[idx].get_memory();
          if (visible_memories.find(mem) == visible_memories.end())
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error << "Invalid mapper output from invocation of 'map_inline' on "
                     "mapper "
                  << *mapper << ". Mapper selected a physical "
                  << "instance in memory " << mem
                  << " which is not visible from processor " << exec_proc
                  << ".";
            error.raise();
          }
        }
      }
      // Iterate over the instances and make sure they are all valid
      // for the given logical region which we are mapping
      std::vector<LogicalRegion> regions_to_check(1, requirement.region);
      for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
      {
        PhysicalManager* manager = chosen_instances[idx].get_physical_manager();
        if (!manager->meets_regions(regions_to_check))
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from invocation of 'map_inline' on "
                   "mapper "
                << *mapper << ". Mapper specified an instance that "
                << "does not meet the logical region requirement.";
          error.raise();
        }
      }
      // If this is a reduction region requirement, make sure all the
      // chosen instances are specialized reduction instances
      if (IS_REDUCE(requirement))
      {
        for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
          if (!chosen_instances[idx].get_manager()->is_reduction_manager())
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error << "Invalid mapper output from invocation of 'map_inline' on "
                     "mapper "
                  << *mapper << ". Mapper failed to select "
                  << "specialized reduction instances for region requirement "
                     "with "
                  << "reduction-only privileges for " << *this << ".";
            error.raise();
          }
      }
      else
      {
        for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
        {
          if (chosen_instances[idx].get_manager()->is_reduction_manager())
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error << "Invalid mapper output from invocation of 'map_inline' on "
                     "mapper "
                  << *mapper << ". Mapper selected an illegal "
                  << "specialized reduction instance for region requirement "
                     "without "
                  << "reduction privileges for " << *this << ".";
            error.raise();
          }
        }
      }
      if (layout_constraint_id > 0)
      {
        // Check the layout constraints are valid
        LayoutConstraints* constraints =
            runtime->find_layout_constraints(layout_constraint_id);
        for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
        {
          PhysicalManager* manager =
              chosen_instances[idx].get_physical_manager();
          const LayoutConstraint* conflict_constraint = nullptr;
          if (manager->conflicts(constraints, &conflict_constraint))
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error << "Invalid mapper output. Mapper " << *mapper
                  << " selected instance for " << *this
                  << " which failed to satisfy the corresponding layout "
                     "constraints.";
            error.raise();
          }
        }
        // See if there is a padding constraint to get reservations for
        if (constraints->padding_constraint.delta.get_dim() > 0)
        {
          FieldMask padding_mask;
          FieldSpaceNode* fs =
              runtime->get_node(requirement.region.get_field_space());
          if (!constraints->field_constraint.field_set.empty())
          {
            std::set<FieldID> field_set;
            for (const FieldID& fid : constraints->field_constraint.field_set)
            {
              field_set.insert(fid);
              region.impl->add_padded_field(fid);
            }
            padding_mask = fs->get_field_mask(field_set);
          }
          else
          {
            padding_mask = fs->get_field_mask(requirement.privilege_fields);
            for (const FieldID& fid : requirement.privilege_fields)
              region.impl->add_padded_field(fid);
          }
          for (unsigned idx = 0; idx < chosen_instances.size(); idx++)
          {
            const InstanceRef& ref = chosen_instances[idx];
            const FieldMask overlap = padding_mask & ref.get_valid_fields();
            if (!overlap)
              continue;
            PhysicalManager* manager = ref.get_physical_manager();
            manager->find_padded_reservations(overlap, this, 0 /*index*/);
            padding_mask -= overlap;
            if (!padding_mask)
              break;
          }
          legion_assert(!padding_mask);
        }
      }
      return output.track_valid_region;
    }

    //--------------------------------------------------------------------------
    int MapOp::add_copy_profiling_request(
        const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& requests,
        bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return copy_fill_priority;
      OpProfilingResponse response(this, info.index, info.dst_index, fill);
      Realm::ProfilingRequest& request = requests.add_request(
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, &response,
          sizeof(response), profiling_priority);
      bool has_finish = false;
      for (const ProfilingMeasurementID& it : profiling_requests)
      {
        const Realm::ProfilingMeasurementID measurement =
            (Realm::ProfilingMeasurementID)it;
        request.add_measurement(measurement);
        if (measurement == Realm::PMID_OP_FINISH_EVENT)
          has_finish = true;
      }
      // Need thetimeline for the operation to know how to profile this
      // profiling response
      if (!has_finish && (runtime->profiler != nullptr))
        request.add_measurement(Realm::PMID_OP_FINISH_EVENT);
      handle_profiling_update(count);
      return copy_fill_priority;
    }

    //--------------------------------------------------------------------------
    bool MapOp::handle_profiling_response(
        const Realm::ProfilingResponse& response, const void* orig,
        size_t orig_length, LgEvent& fevent, bool& failed_alloc)
    //--------------------------------------------------------------------------
    {
      legion_assert(mapper != nullptr);
      const OpProfilingResponse* op_info =
          static_cast<const OpProfilingResponse*>(response.user_data());
      Realm::ProfilingMeasurements::OperationFinishEvent finish_event;
      if (response.get_measurement(finish_event))
        fevent = LgEvent(finish_event.finish_event);
      // Check to see if we are done mapping, if not then we need to defer
      // this until we are done mapping so we know how many reports to expect
      const RtEvent mapped = get_mapped_event();
      if (!mapped.has_triggered())
        mapped.wait();
      // If we get here then we can handle the response now
      Mapping::Mapper::InlineProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      info.total_reports = outstanding_profiling_requests;
      info.fill_response = op_info->fill;
      mapper->invoke_inline_report_profiling(this, info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
      legion_assert(count <= outstanding_profiling_requests);
      if (count == outstanding_profiling_requests)
        Runtime::trigger_event(profiling_reported);
      // Always record these as part of profiling
      return true;
    }

    //--------------------------------------------------------------------------
    void MapOp::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
      legion_assert(count > 0);
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void MapOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_mapping(rez, target);
      rez.serialize(copy_fill_priority);
      rez.serialize<size_t>(profiling_requests.size());
      if (!profiling_requests.empty())
      {
        for (unsigned idx = 0; idx < profiling_requests.size(); idx++)
          rez.serialize(profiling_requests[idx]);
        rez.serialize(profiling_priority);
        rez.serialize(runtime->find_utility_group());
        // Create a user event for this response
        const RtUserEvent response = Runtime::create_rt_user_event();
        rez.serialize(response);
        applied_events.insert(response);
      }
    }

    /////////////////////////////////////////////////////////////
    // Repl Map Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplMapOp::ReplMapOp(void)
      : ReplCollectiveViewCreator<CollectiveViewCreator<MapOp> >()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplMapOp::~ReplMapOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplMapOp::initialize_replication(ReplicateContext* ctx)
    //--------------------------------------------------------------------------
    {
      // Mark that this is collective
      requirement.prop |= LEGION_COLLECTIVE_MASK;
      if (!remap_region && runtime->safe_mapper)
      {
        mapping_check = ctx->get_next_collective_index(COLLECTIVE_LOC_74);
        sources_check = ctx->get_next_collective_index(COLLECTIVE_LOC_104);
      }
      if (!grants.empty())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal use of grants with an inline mapping in control "
              << "replicated parent task " << parent_ctx->get_task_name()
              << " (UID " << parent_ctx->get_unique_id()
              << "). Use of non-canonical "
              << "Legion features such as grants are not permitted with "
              << "control replication.";
        error.raise();
      }
      if (!wait_barriers.empty())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error
            << "Illegal use of wait phase barriers with an inline mapping in "
            << "control replicated parent task " << parent_ctx->get_task_name()
            << " (UID " << parent_ctx->get_unique_id() << "). Use of "
            << "non-canonical Legion features such as wait phase barriers are "
            << "not permitted with control replication.";
        error.raise();
      }
      if (!arrive_barriers.empty())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error
            << "Illegal use of arrive phase barriers with an inline mapping in "
            << "control replicated parent task " << parent_ctx->get_task_name()
            << " (UID " << parent_ctx->get_unique_id() << "). Use of "
            << "non-canonical Legion features such as arrive phase barriers "
               "are "
            << "not permitted with control replication.";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    void ReplMapOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(requirement.handle_type == LEGION_SINGULAR_PROJECTION);
      analyze_region_requirements();
      // If this a write requirement then we need to perform syncs on the
      // way in and the way out of the physical analysis across the shards
      // to ensure we don't do any exclusive updates out of order
      if (IS_WRITE(requirement))
      {
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(parent_ctx);
        legion_assert(!collective_map_barrier.exists());
        collective_map_barrier = repl_ctx->get_next_collective_map_barriers();
      }
      // We're always going to do collective rendezvous for this requirement
      create_collective_rendezvous(requirement.parent.get_tree_id(), 0);
    }

    //--------------------------------------------------------------------------
    void ReplMapOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Signal that all our mapping dependences have been met
      if (collective_map_barrier.exists())
        runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/);
      std::set<RtEvent> preconditions;
      // Compute the version numbers for this mapping operation
      perform_versioning_analysis(
          0 /*idx*/, requirement, version_info, preconditions,
          nullptr /*output region*/, true /*rendezvous*/);
      if (collective_map_barrier.exists())
      {
        if (!collective_map_barrier.has_triggered())
          preconditions.insert(collective_map_barrier);
        Runtime::advance_barrier(collective_map_barrier);
      }
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    bool ReplMapOp::invoke_mapper(
        InstanceSet& mapped_instances,
        std::vector<PhysicalManager*>& source_instances)
    //--------------------------------------------------------------------------
    {
      const bool result =
          MapOp::invoke_mapper(mapped_instances, source_instances);
      if (runtime->safe_mapper)
      {
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(parent_ctx);
        // For read-write or write-discard cases make sure that all the
        // shards mapped to independent physical instances
        if (IS_WRITE(requirement))
        {
          CheckCollectiveMapping mapping_collective(repl_ctx, mapping_check);
          mapping_collective.verify(mapped_instances, mapper);
        }
        // For anything that is not a reduce inline mapping we check that
        // the names of the sources are the same across all the shards
        if (!IS_REDUCE(requirement))
        {
          CheckCollectiveSources sources_collective(repl_ctx, sources_check);
          if (!sources_collective.verify(source_instances))
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error << "Invalid mapper output from invocation of 'map_inline' "
                  << "by mapper " << mapper->get_mapper_name()
                  << ". Mapper selected different 'source_instances' "
                  << "on shard 0 and shard " << repl_ctx->owner_shard->shard_id
                  << " when mapping an inline mapping in "
                  << "control-replicated parent task "
                  << parent_ctx->get_task_name() << " (UID "
                  << parent_ctx->get_unique_id() << "). Each inline "
                  << "mapping in a control-replicated parent task must provide "
                  << "same 'source_instances' across all shards.";
            error.raise();
          }
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    RtEvent ReplMapOp::finalize_complete_mapping(RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      if (collective_map_barrier.exists())
      {
        runtime->phase_barrier_arrive(
            collective_map_barrier, 1 /*count*/, precondition);
        const RtEvent result = collective_map_barrier;
        collective_map_barrier = RtBarrier::NO_RT_BARRIER;
        return result;
      }
      else
        return precondition;
    }

    //--------------------------------------------------------------------------
    bool ReplMapOp::perform_collective_analysis(
        CollectiveMapping*& mapping, bool& first_local)
    //--------------------------------------------------------------------------
    {
      // Yes, we want to do a collective analysis, but we'll need to
      // construct a collective view here for all the instances
      return true;
    }

    //--------------------------------------------------------------------------
    bool ReplMapOp::find_shard_participants(std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      // All the shards are participating
      return true;
    }

    //--------------------------------------------------------------------------
    void ReplMapOp::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveViewCreator<CollectiveViewCreator<MapOp> >::activate();
      mapping_check = 0;
      sources_check = 0;
      collective_map_barrier = RtBarrier::NO_RT_BARRIER;
    }

    //--------------------------------------------------------------------------
    void ReplMapOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      // Make sure that we consumed this if we had one
      legion_assert(!collective_map_barrier.exists());
      ReplCollectiveViewCreator<CollectiveViewCreator<MapOp> >::deactivate(
          false);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    RtEvent ReplMapOp::perform_collective_versioning_analysis(
        unsigned index, LogicalRegion handle, EqSetTracker* tracker,
        const FieldMask& mask, unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      return rendezvous_collective_versioning_analysis(
          index, handle, tracker, runtime->address_space, mask,
          parent_req_index);
    }

    /////////////////////////////////////////////////////////////
    // Check Collective Mapping
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CheckCollectiveMapping::CheckCollectiveMapping(
        ReplicateContext* ctx, CollectiveID id)
      : AllGatherCollective<true>(ctx, id)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    CheckCollectiveMapping::~CheckCollectiveMapping(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void CheckCollectiveMapping::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(mapped_instances.size());
      for (const std::pair<const PhysicalInstance, ShardFields>& mit :
           mapped_instances)
      {
        rez.serialize(mit.first);
        rez.serialize<size_t>(mit.second.size());
        for (ShardFields::const_iterator it = mit.second.begin();
             it != mit.second.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
      }
    }

    //--------------------------------------------------------------------------
    void CheckCollectiveMapping::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_instances;
      derez.deserialize(num_instances);
      for (unsigned idx1 = 0; idx1 < num_instances; idx1++)
      {
        PhysicalInstance inst;
        derez.deserialize(inst);
        ShardFields& shard_fields = mapped_instances[inst];
        size_t offset = shard_fields.size();
        size_t num_copies;
        derez.deserialize(num_copies);
        shard_fields.resize(offset + num_copies);
        for (unsigned idx2 = 0; idx2 < num_copies; idx2++)
        {
          derez.deserialize(shard_fields[offset + idx2].first);
          derez.deserialize(shard_fields[offset + idx2].second);
        }
      }
    }

    //--------------------------------------------------------------------------
    void CheckCollectiveMapping::verify(
        const InstanceSet& instances, MapperManager* mapper)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const InstanceRef& ref = instances[idx];
        PhysicalManager* manager = ref.get_physical_manager();
        PhysicalInstance inst = manager->get_instance();
        mapped_instances[inst].emplace_back(
            std::make_pair(local_shard, ref.get_valid_fields()));
      }
      perform_collective_sync();
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const InstanceRef& ref = instances[idx];
        PhysicalManager* manager = ref.get_physical_manager();
        PhysicalInstance inst = manager->get_instance();
        ShardFields& shard_fields = mapped_instances[inst];
        legion_assert(!shard_fields.empty());
        for (ShardFields::const_iterator it = shard_fields.begin();
             it != shard_fields.end(); it++)
        {
          if (it->first == local_shard)
            continue;
          if (it->second * ref.get_valid_fields())
            continue;
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from invocation of 'map_inline' "
                << "by mapper " << mapper->get_mapper_name()
                << ". Mapper selected the same physical instance " << inst.id
                << " on both shards " << local_shard << " and " << it->first
                << " with write privileges for "
                << "inline mapping in control-replicated parent task "
                << context->get_task_name() << " (UID "
                << context->get_unique_id()
                << "). Each inline mapping with write privileges in a "
                << "control-replicated parent task must map to a different "
                << "physical instance to avoid races.";
          error.raise();
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Check Collective Sources
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CheckCollectiveSources::CheckCollectiveSources(
        ReplicateContext* ctx, CollectiveID id)
      : BroadcastCollective(ctx, id, 0 /*origin shard*/)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    CheckCollectiveSources::~CheckCollectiveSources(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void CheckCollectiveSources::pack_collective(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(source_instances.size());
      for (const DistributedID& it : source_instances) rez.serialize(it);
    }

    //--------------------------------------------------------------------------
    void CheckCollectiveSources::unpack_collective(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      size_t num_instances;
      derez.deserialize(num_instances);
      source_instances.resize(num_instances);
      for (unsigned idx = 0; idx < num_instances; idx++)
        derez.deserialize(source_instances[idx]);
    }

    //--------------------------------------------------------------------------
    bool CheckCollectiveSources::verify(
        const std::vector<PhysicalManager*>& instances)
    //--------------------------------------------------------------------------
    {
      if (local_shard == 0)
      {
        source_instances.resize(instances.size());
        for (unsigned idx = 0; idx < instances.size(); idx++)
          source_instances[idx] = instances[idx]->did;
        perform_collective_async();
      }
      else
      {
        perform_collective_wait();
        if (instances.size() != source_instances.size())
          return false;
        for (unsigned idx = 0; idx < instances.size(); idx++)
          if (source_instances[idx] != instances[idx]->did)
            return false;
      }
      return true;
    }

    /////////////////////////////////////////////////////////////
    // Remote Map Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteMapOp::RemoteMapOp(Operation* ptr, AddressSpaceID src)
      : ExternalMapping(), RemoteOp(ptr, src)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RemoteMapOp::~RemoteMapOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID RemoteMapOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t RemoteMapOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteMapOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteMapOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteMapOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == nullptr)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const std::string_view& RemoteMapOp::get_provenance_string(bool human) const
    //--------------------------------------------------------------------------
    {
      Provenance* provenance = get_provenance();
      if (provenance != nullptr)
        return human ? provenance->human : provenance->machine;
      else
        return Provenance::no_provenance;
    }

    //--------------------------------------------------------------------------
    const char* RemoteMapOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[MAP_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind RemoteMapOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return MAP_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteMapOp::select_sources(
        const unsigned index, PhysicalManager* target,
        const std::vector<InstanceView*>& sources,
        std::vector<unsigned>& ranking,
        std::map<unsigned, PhysicalManager*>& points)
    //--------------------------------------------------------------------------
    {
      if (source == runtime->address_space)
      {
        // If we're on the owner node we can just do this
        remote_ptr->select_sources(index, target, sources, ranking, points);
        return;
      }
      legion_assert(index == 0);
      Mapper::SelectInlineSrcInput input;
      Mapper::SelectInlineSrcOutput output;
      prepare_for_mapping(
          sources, input.source_instances, input.collective_views);
      prepare_for_mapping(target, input.target);
      if (mapper == nullptr)
        mapper = runtime->find_mapper(map_id);
      mapper->invoke_select_inline_sources(this, input, output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking, points);
    }

    //--------------------------------------------------------------------------
    void RemoteMapOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_mapping(rez, target);
      pack_profiling_requests(rez, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteMapOp::unpack(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      unpack_external_mapping(derez);
      unpack_profiling_requests(derez);
    }

  }  // namespace Internal
}  // namespace Legion
