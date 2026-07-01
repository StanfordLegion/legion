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

#include "legion/operations/acquire.h"
#include "legion/analysis/acquire.h"
#include "legion/contexts/replicate.h"
#include "legion/api/physical_region_impl.h"
#include "legion/managers/mapper.h"
#include "legion/managers/shard.h"
#include "legion/tracing/recognizer.h"
#include "legion/utilities/provenance.h"
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // External Acquire
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalAcquire::ExternalAcquire(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ExternalAcquire::pack_external_acquire(
        Serializer& rez, AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(logical_region);
      rez.serialize(parent_region);
      rez.serialize<size_t>(fields.size());
      for (const FieldID& fid : fields) rez.serialize(fid);
      rez.serialize(grants.size());
      for (unsigned idx = 0; idx < grants.size(); idx++)
        pack_grant(grants[idx], rez);
      rez.serialize(wait_barriers.size());
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        pack_phase_barrier(wait_barriers[idx], rez);
      rez.serialize(arrive_barriers.size());
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        pack_phase_barrier(arrive_barriers[idx], rez);
      pack_mappable(*this, rez);
      rez.serialize(get_context_index());
    }

    //--------------------------------------------------------------------------
    void ExternalAcquire::unpack_external_acquire(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(logical_region);
      derez.deserialize(parent_region);
      size_t num_fields;
      derez.deserialize(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        fields.insert(fid);
      }
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
      unpack_mappable(*this, derez);
      uint64_t index;
      derez.deserialize(index);
      set_context_index(index);
    }

    /////////////////////////////////////////////////////////////
    // Acquire Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AcquireOp::AcquireOp(void) : ExternalAcquire(), PredicatedOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    AcquireOp::~AcquireOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void AcquireOp::initialize(
        InnerContext* ctx, const AcquireLauncher& launcher,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_predication(ctx, launcher.predicate, provenance);
      // Note we give it READ WRITE EXCLUSIVE to make sure that nobody
      // can be re-ordered around this operation for mapping or
      // normal dependences.  We won't actually read or write anything.
      requirement = RegionRequirement(
          launcher.logical_region, LEGION_READ_WRITE, LEGION_EXCLUSIVE,
          launcher.parent_region);
      requirement.privilege_fields = launcher.fields;
      if (runtime->safe_model)
        verify_requirement(requirement);
      parent_req_index = ctx->find_parent_region_index(
          this, requirement, 0 /*index*/, true /*skip privileges*/);
      logical_region = launcher.logical_region;
      restricted_region = launcher.physical_region;
      if (restricted_region.impl != nullptr)
      {
        const RegionRequirement& region_req =
            restricted_region.impl->get_requirement();
        if (region_req.privilege_fields != launcher.fields)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error
              << "The privilege fields for " << *this
              << "do not match the fields for the PhysicalRegion object being "
              << "used for establishing restricted coherence. "
              << "The field sets must match exactly.";
          error.raise();
        }
      }
      parent_region = launcher.parent_region;
      fields = launcher.fields;
      // Mark the requirement restricted
      grants = launcher.grants;
      // Register ourselves with all the grants
      for (unsigned idx = 0; idx < grants.size(); idx++)
        grants[idx].impl->register_operation(get_completion_event());
      wait_barriers = launcher.wait_barriers;
      if (spy_logging_level > LIGHT_SPY_LOGGING)
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
      LegionSpy::log_acquire_operation(
          parent_ctx->get_unique_id(), unique_op_id);
    }

    //--------------------------------------------------------------------------
    void AcquireOp::activate(void)
    //--------------------------------------------------------------------------
    {
      PredicatedOp::activate();
      mapper = nullptr;
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      copy_fill_priority = 0;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      restricted_region = PhysicalRegion();
      version_info.clear();
      fields.clear();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      map_applied_conditions.clear();
      profiling_requests.clear();
      if (mapper_data != nullptr)
      {
        free(mapper_data);
        mapper_data = nullptr;
        mapper_data_size = 0;
      }
      PredicatedOp::deactivate(false /*free*/);
      // Return this operation to the runtime
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* AcquireOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[ACQUIRE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind AcquireOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return ACQUIRE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t AcquireOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    Mappable* AcquireOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      log_acquire_requirement();
    }

    //--------------------------------------------------------------------------
    void AcquireOp::log_acquire_requirement(void)
    //--------------------------------------------------------------------------
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

    //--------------------------------------------------------------------------
    void AcquireOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // First register any mapping dependences that we have
      analyze_region_requirements();
    }

    //--------------------------------------------------------------------------
    void AcquireOp::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      // Otherwise do the things needed to clean up this operation
      complete_execution();
      if (!map_applied_conditions.empty())
        complete_mapping(finalize_complete_mapping(
            Runtime::merge_events(map_applied_conditions)));
      else
        complete_mapping();
    }

    //--------------------------------------------------------------------------
    void AcquireOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      if (parent_req_index == TRACED_PARENT_INDEX)
        parent_req_index = parent_ctx->find_parent_region_index(
            this, requirement, 0 /*idx*/, true /*skip privileges*/,
            true /*force*/);
      std::set<RtEvent> preconditions;
      perform_versioning_analysis(
          0 /*idx*/, requirement, version_info, preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void AcquireOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      const PhysicalTraceInfo trace_info(this, 0 /*index*/);
      // Invoke the mapper before doing anything else
      invoke_mapper();
      InstanceSet restricted_instances;
      if (restricted_region.impl != nullptr)
        restricted_region.impl->get_references(restricted_instances);
      const ApEvent init_precondition = compute_sync_precondition(trace_info);
      ApUserEvent acquire_post = Runtime::create_ap_user_event(&trace_info);
      ApEvent acquire_complete = acquire_restrictions(
          requirement, version_info, 0 /*idx*/, init_precondition, acquire_post,
          restricted_instances, trace_info, map_applied_conditions);
      Runtime::trigger_event(
          acquire_post, acquire_complete, trace_info, map_applied_conditions);
      record_completion_effect(acquire_post);
      log_mapping_decision(0 /*idx*/, requirement, restricted_instances);
      LegionSpy::log_operation_events(
          unique_op_id, acquire_complete, acquire_post);
      // Remove profiling our guard and trigger the profiling event if necessary
      if ((outstanding_profiling_requests.fetch_sub(1) == 1) &&
          profiling_reported.exists())
        Runtime::trigger_event(profiling_reported);
      if (is_recording())
        trace_info.record_complete_replay(map_applied_conditions);
      // Mark that we completed mapping
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
    void AcquireOp::trigger_complete(ApEvent complete)
    //--------------------------------------------------------------------------
    {
      // Chain any arrival barriers
      if (!arrive_barriers.empty())
      {
        for (const PhaseBarrier& bar : arrive_barriers)
        {
          LegionSpy::log_phase_barrier_arrival(unique_op_id, bar.phase_barrier);
          runtime->phase_barrier_arrive(
              bar.phase_barrier, 1 /*count*/, complete);
        }
      }
      complete_operation(complete);
    }

    //--------------------------------------------------------------------------
    void AcquireOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if we need to do a profiling response
      if (profiling_reported.exists() && (outstanding_profiling_requests == 0))
      {
        // We're not expecting any profiling callbacks so we need to
        // do one ourself to inform the mapper that there won't be any
        Mapping::Mapper::AcquireProfilingInfo info;
        info.total_reports = 0;
        info.fill_response = false;  // make valgrind happy
        mapper->invoke_acquire_report_profiling(this, info);
        Runtime::trigger_event(profiling_reported);
      }
      // Don't commit thisoperation until we've reported profiling information
      commit_operation(true /*deactivate*/, profiling_reported);
    }

    //--------------------------------------------------------------------------
    bool AcquireOp::record_trace_hash(
        TraceHashRecorder& recorder, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      Murmur3Hasher hasher;
      hasher.hash(get_operation_kind());
      hasher.hash(logical_region);
      hasher.hash(parent_region);
      for (const FieldID& fid : fields) hasher.hash(fid);
      return recorder.record_operation_hash(this, hasher, opidx);
    }

    //--------------------------------------------------------------------------
    unsigned AcquireOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx == 0);
      legion_assert(parent_req_index != TRACED_PARENT_INDEX);
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*, unsigned>* AcquireOp::get_acquired_instances_ref(
        void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    UniqueID AcquireOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t AcquireOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int AcquireOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* AcquireOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == nullptr)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const std::string_view& AcquireOp::get_provenance_string(bool human) const
    //--------------------------------------------------------------------------
    {
      Provenance* provenance = get_provenance();
      if (provenance != nullptr)
        return human ? provenance->human : provenance->machine;
      else
        return Provenance::no_provenance;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_replay_operation(unique_op_id);
      complete_mapping(finalize_complete_mapping(RtEvent::NO_RT_EVENT));
    }

    //--------------------------------------------------------------------------
    void AcquireOp::complete_replay(ApEvent acquire_complete_event)
    //--------------------------------------------------------------------------
    {
      // Chain all the unlock and barrier arrivals off of the
      // copy complete event
      if (!arrive_barriers.empty())
      {
        for (PhaseBarrier& barrier : arrive_barriers)
        {
          LegionSpy::log_phase_barrier_arrival(
              unique_op_id, barrier.phase_barrier);
          runtime->phase_barrier_arrive(
              barrier.phase_barrier, 1 /*count*/, acquire_complete_event);
        }
      }
      // Handle the case for marking when the copy completes
      record_completion_effect(acquire_complete_event);
      complete_execution();
    }

    //--------------------------------------------------------------------------
    const VersionInfo& AcquireOp::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      return version_info;
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& AcquireOp::get_requirement(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      return requirement;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::invoke_mapper(void)
    //--------------------------------------------------------------------------
    {
      Mapper::MapAcquireInput input;
      Mapper::MapAcquireOutput output;
      if (mapper == nullptr)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_map_acquire(this, input, output);
      copy_fill_priority = output.copy_fill_priority;
      if (!output.profiling_requests.empty())
      {
        filter_copy_request_kinds(
            mapper, output.profiling_requests.requested_measurements,
            profiling_requests, true /*warn*/);
        profiling_priority = output.profiling_priority;
        legion_assert(!profiling_reported.exists());
        profiling_reported = Runtime::create_rt_user_event();
      }
    }

    //--------------------------------------------------------------------------
    int AcquireOp::add_copy_profiling_request(
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
    bool AcquireOp::handle_profiling_response(
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
      Mapping::Mapper::AcquireProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      info.total_reports = outstanding_profiling_requests;
      info.fill_response = op_info->fill;
      mapper->invoke_acquire_report_profiling(this, info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
      legion_assert(count <= outstanding_profiling_requests);
      if (count == outstanding_profiling_requests)
        Runtime::trigger_event(profiling_reported);
      // Always record these as part of profiling
      return true;
    }

    //--------------------------------------------------------------------------
    void AcquireOp::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
      legion_assert(count > 0);
      legion_assert(!mapped_event.has_triggered());
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void AcquireOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_acquire(rez, target);
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

    //--------------------------------------------------------------------------
    ApEvent AcquireOp::acquire_restrictions(
        const RegionRequirement& req, const VersionInfo& version_info,
        unsigned index, ApEvent precondition, ApEvent term_event,
        InstanceSet& restricted_instances, const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& map_applied_events)
    //--------------------------------------------------------------------------
    {
      legion_assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
      // should be exclusive
      legion_assert(IS_EXCLUSIVE(req));
      const bool known_targets = !restricted_instances.empty();
      RegionNode* region = runtime->get_node(req.region);
      AcquireAnalysis* analysis =
          new AcquireAnalysis(this, index, region, trace_info);
      analysis->add_reference();
      RtEvent views_ready;
      if (known_targets)
        views_ready = analysis->convert_views(req.region, restricted_instances);
      // Iterate through the equivalence classes and find all the restrictions
      const RtEvent traversal_done = analysis->perform_traversal(
          views_ready, version_info, map_applied_events);
      RtEvent remote_ready;
      if (traversal_done.exists() || analysis->has_remote_sets())
        remote_ready =
            analysis->perform_remote(traversal_done, map_applied_events);
      if (!known_targets)
      {
        if (remote_ready.exists() && !remote_ready.has_triggered())
          remote_ready.wait();
        op::FieldMaskMap<LogicalView> instances;
        analysis->report_instances(instances);
        restricted_instances.resize(instances.size());
        analysis->target_instances.resize(instances.size());
        analysis->target_views.resize(instances.size());
        unsigned inst_index = 0;
        // Note that all of these should be individual views.
        // The only way to get collective restricted view is by
        // doing attaches in control replicated contexts and we insist
        // that all acquire operations in control replicated context
        // explicitly provide a PhysicalRegion argument so we should
        // always go through the known_targets path, therefore there
        // should be no collective views here.
        for (op::FieldMaskMap<LogicalView>::const_iterator it =
                 instances.begin();
             it != instances.end(); it++, inst_index++)
        {
          legion_assert(it->first->is_individual_view());
          IndividualView* inst_view = it->first->as_individual_view();
          PhysicalManager* manager = inst_view->get_manager();
          restricted_instances[inst_index] = InstanceRef(manager, it->second);
          analysis->target_instances[inst_index] = manager;
          analysis->target_views[inst_index].insert(inst_view, it->second);
        }
      }
      // Now add users for all the instances
      ApEvent instances_ready;
      const RegionUsage usage(req);
      analysis->perform_registration(
          remote_ready, usage, map_applied_events, precondition, term_event,
          instances_ready);
      if (analysis->remove_reference())
        delete analysis;
      return instances_ready;
    }

    /////////////////////////////////////////////////////////////
    // Repl Acquire Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplAcquireOp::ReplAcquireOp(void)
      : ReplCollectiveViewCreator<CollectiveViewCreator<AcquireOp> >()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplAcquireOp::~ReplAcquireOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplAcquireOp::initialize_replication(
        ReplicateContext* context, bool first_local_shard)
    //--------------------------------------------------------------------------
    {
      is_first_local_shard = first_local_shard;
      if (restricted_region.impl == nullptr)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error
            << "Acquire operation in control replicated parent task "
            << parent_ctx->get_task_name() << " (UID "
            << parent_ctx->get_unique_id() << ") did not specify a "
            << "`physical_region' argument. All acquire operations in control "
            << "replicated contexts must specify an explicit PhysicalRegion.";
        error.raise();
      }
      if (!grants.empty())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal use of grants with an acquire operation in control "
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
            << "Illegal use of wait phase barriers with an acquire operation "
               "in "
            << "control replicated parent task " << parent_ctx->get_task_name()
            << " (UID " << parent_ctx->get_unique_id() << "). Use of "
            << "non-canonical Legion features such as wait phase barriers are "
            << "not permitted with control replication.";
        error.raise();
      }
      if (!arrive_barriers.empty())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal use of arrive phase barriers with an acquire "
                 "operation in "
              << "control replicated parent task "
              << parent_ctx->get_task_name() << " (UID "
              << parent_ctx->get_unique_id() << "). Use of "
              << "non-canonical Legion features such as arrive phase barriers "
                 "are "
              << "not permitted with control replication.";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    void ReplAcquireOp::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveViewCreator<CollectiveViewCreator<AcquireOp> >::activate();
      collective_map_barrier = RtBarrier::NO_RT_BARRIER;
      is_first_local_shard = false;
    }

    //--------------------------------------------------------------------------
    void ReplAcquireOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      // Make sure we didn't leak our barrier
      legion_assert(!collective_map_barrier.exists());
      ReplCollectiveViewCreator<CollectiveViewCreator<AcquireOp> >::deactivate(
          false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplAcquireOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      collective_map_barrier = repl_ctx->get_next_collective_map_barriers();
      // See if we need to make a collective view rendezvous
      if (restricted_region.impl->collective)
        create_collective_rendezvous(requirement.parent.get_tree_id(), 0);
      // Then do the base class analysis
      AcquireOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void ReplAcquireOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(collective_map_barrier.exists());
      // Signal that all of our mapping dependences are satisfied
      runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/);
      if (parent_req_index == TRACED_PARENT_INDEX)
        parent_req_index = parent_ctx->find_parent_region_index(
            this, requirement, 0 /*idx*/, true /*skip privileges*/,
            true /*force*/);
      std::set<RtEvent> preconditions;
      perform_versioning_analysis(
          0 /*idx*/, requirement, version_info, preconditions,
          nullptr /*output region*/, true /*rendezvous*/);
      if (!collective_map_barrier.has_triggered())
        preconditions.insert(collective_map_barrier);
      Runtime::advance_barrier(collective_map_barrier);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    RtEvent ReplAcquireOp::finalize_complete_mapping(RtEvent pre)
    //--------------------------------------------------------------------------
    {
      if (collective_map_barrier.exists())
      {
        // Normal analysis path
        runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/, pre);
        return collective_map_barrier;
      }
      else  // Tracing path
        return pre;
    }

    //--------------------------------------------------------------------------
    bool ReplAcquireOp::perform_collective_analysis(
        CollectiveMapping*& mapping, bool& first_local)
    //--------------------------------------------------------------------------
    {
      if (!restricted_region.impl->collective)
      {
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(parent_ctx);
        legion_assert(!collective_map_barrier.exists());
        mapping = &repl_ctx->shard_manager->get_collective_mapping();
        mapping->add_reference();
        first_local = is_first_local_shard;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void ReplAcquireOp::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(collective_map_barrier.exists());
      runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/);
      Runtime::advance_barrier(collective_map_barrier);
      elide_collective_rendezvous();
      AcquireOp::predicate_false();
    }

    //--------------------------------------------------------------------------
    void ReplAcquireOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(collective_map_barrier.exists());
      // Elide both generations of the mapping fence barrier
      runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/);
      Runtime::advance_barrier(collective_map_barrier);
      runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/);
      collective_map_barrier = RtBarrier::NO_RT_BARRIER;
      elide_collective_rendezvous();
      AcquireOp::trigger_replay();
    }

    //--------------------------------------------------------------------------
    RtEvent ReplAcquireOp::perform_collective_versioning_analysis(
        unsigned index, LogicalRegion handle, EqSetTracker* tracker,
        const FieldMask& mask, unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      return rendezvous_collective_versioning_analysis(
          index, handle, tracker, runtime->address_space, mask,
          parent_req_index);
    }

    /////////////////////////////////////////////////////////////
    // Remote Acquire Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteAcquireOp::RemoteAcquireOp(Operation* ptr, AddressSpaceID src)
      : ExternalAcquire(), RemoteOp(ptr, src)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RemoteAcquireOp::~RemoteAcquireOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID RemoteAcquireOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t RemoteAcquireOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteAcquireOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteAcquireOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteAcquireOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == nullptr)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const std::string_view& RemoteAcquireOp::get_provenance_string(
        bool human) const
    //--------------------------------------------------------------------------
    {
      Provenance* provenance = get_provenance();
      if (provenance != nullptr)
        return human ? provenance->human : provenance->machine;
      else
        return Provenance::no_provenance;
    }

    //--------------------------------------------------------------------------
    const char* RemoteAcquireOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[ACQUIRE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind RemoteAcquireOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return ACQUIRE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteAcquireOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_acquire(rez, target);
      pack_profiling_requests(rez, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteAcquireOp::unpack(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      unpack_external_acquire(derez);
      unpack_profiling_requests(derez);
    }

  }  // namespace Internal
}  // namespace Legion
