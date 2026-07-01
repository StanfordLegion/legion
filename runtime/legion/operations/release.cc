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

#include "legion/operations/release.h"
#include "legion/analysis/release.h"
#include "legion/contexts/replicate.h"
#include "legion/api/physical_region_impl.h"
#include "legion/managers/mapper.h"
#include "legion/managers/shard.h"
#include "legion/operations/mapping.h"
#include "legion/tracing/recognizer.h"
#include "legion/utilities/provenance.h"
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // External Release
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalRelease::ExternalRelease(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ExternalRelease::pack_external_release(
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
    void ExternalRelease::unpack_external_release(Deserializer& derez)
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
    // Release Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReleaseOp::ReleaseOp(void) : ExternalRelease(), PredicatedOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReleaseOp::~ReleaseOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReleaseOp::initialize(
        InnerContext* ctx, const ReleaseLauncher& launcher,
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
              << " do not match the fields for the PhysicalRegion object being "
              << "used for establishing restricted coherence. The field sets "
              << "must match exactly.";
          error.raise();
        }
      }
      parent_region = launcher.parent_region;
      fields = launcher.fields;
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
      LegionSpy::log_release_operation(
          parent_ctx->get_unique_id(), unique_op_id);
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::activate(void)
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
    void ReleaseOp::deactivate(bool freeop)
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
    const char* ReleaseOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[RELEASE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind ReleaseOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return RELEASE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t ReleaseOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    Mappable* ReleaseOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent index
      log_release_requirement();
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::log_release_requirement(void)
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
    void ReleaseOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(
            this, wait_barriers, arrive_barriers);
      // First register any mapping dependences that we have
      analyze_region_requirements();
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      // Do the things needed to clean up this operation
      complete_execution();
      if (!map_applied_conditions.empty())
        complete_mapping(finalize_complete_mapping(
            Runtime::merge_events(map_applied_conditions)));
      else
        complete_mapping();
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::trigger_ready(void)
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
    void ReleaseOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      const PhysicalTraceInfo trace_info(this, 0 /*index*/);
      // Invoke the mapper before doing anything else
      std::vector<PhysicalManager*> source_instances;
      invoke_mapper(source_instances);
      InstanceSet restricted_instances;
      if (restricted_region.impl != nullptr)
        restricted_region.impl->get_references(restricted_instances);
      const ApEvent init_precondition = compute_sync_precondition(trace_info);
      ApUserEvent release_post = Runtime::create_ap_user_event(&trace_info);
      ApEvent release_complete = release_restrictions(
          requirement, version_info, 0 /*idx*/, init_precondition, release_post,
          restricted_instances, source_instances, trace_info,
          map_applied_conditions);
      Runtime::trigger_event(
          release_post, release_complete, trace_info, map_applied_conditions);
      record_completion_effect(release_post);
      log_mapping_decision(0 /*idx*/, requirement, restricted_instances);
      LegionSpy::log_operation_events(
          unique_op_id, release_complete, release_post);
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
    void ReleaseOp::trigger_complete(ApEvent complete)
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
    void ReleaseOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if we need to do a profiling response
      if (profiling_reported.exists() && (outstanding_profiling_requests == 0))
      {
        // We're not expecting any profiling callbacks so we need to
        // do one ourself to inform the mapper that there won't be any
        Mapping::Mapper::ReleaseProfilingInfo info;
        info.total_reports = 0;
        info.fill_response = false;  // make valgrind happy
        mapper->invoke_release_report_profiling(this, info);
        Runtime::trigger_event(profiling_reported);
      }
      // Don't commit this operation until the profiling is done
      commit_operation(true /*deactivate*/, profiling_reported);
    }

    //--------------------------------------------------------------------------
    bool ReleaseOp::record_trace_hash(
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
    unsigned ReleaseOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx == 0);
      legion_assert(parent_req_index != TRACED_PARENT_INDEX);
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::select_sources(
        const unsigned index, PhysicalManager* target,
        const std::vector<InstanceView*>& sources,
        std::vector<unsigned>& ranking,
        std::map<unsigned, PhysicalManager*>& points)
    //--------------------------------------------------------------------------
    {
      legion_assert(index == 0);
      Mapper::SelectReleaseSrcInput input;
      Mapper::SelectReleaseSrcOutput output;
      prepare_for_mapping(target, input.target);
      prepare_for_mapping(
          sources, input.source_instances, input.collective_views);
      if (mapper == nullptr)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_select_release_sources(this, input, output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking, points);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*, unsigned>* ReleaseOp::get_acquired_instances_ref(
        void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    UniqueID ReleaseOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t ReleaseOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int ReleaseOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* ReleaseOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == nullptr)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const std::string_view& ReleaseOp::get_provenance_string(bool human) const
    //--------------------------------------------------------------------------
    {
      Provenance* provenance = get_provenance();
      if (provenance != nullptr)
        return human ? provenance->human : provenance->machine;
      else
        return Provenance::no_provenance;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_replay_operation(unique_op_id);
      complete_mapping(finalize_complete_mapping(RtEvent::NO_RT_EVENT));
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::complete_replay(ApEvent release_complete_event)
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
              barrier.phase_barrier, 1 /*count*/, release_complete_event);
        }
      }
      // Handle the case for marking when the copy completes
      record_completion_effect(release_complete_event);
      complete_execution();
    }

    //--------------------------------------------------------------------------
    const VersionInfo& ReleaseOp::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      return version_info;
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& ReleaseOp::get_requirement(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      return requirement;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::invoke_mapper(std::vector<PhysicalManager*>& src_instances)
    //--------------------------------------------------------------------------
    {
      Mapper::MapReleaseInput input;
      Mapper::MapReleaseOutput output;
      if (mapper == nullptr)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_map_release(this, input, output);
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
      if (!output.source_instances.empty())
        physical_convert_sources(
            requirement, output.source_instances, src_instances,
            runtime->safe_mapper ? &acquired_instances : nullptr);
    }

    //--------------------------------------------------------------------------
    int ReleaseOp::add_copy_profiling_request(
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
    bool ReleaseOp::handle_profiling_response(
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
      Mapping::Mapper::ReleaseProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      info.total_reports = outstanding_profiling_requests;
      info.fill_response = op_info->fill;
      mapper->invoke_release_report_profiling(this, info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
      legion_assert(count <= outstanding_profiling_requests);
      if (count == outstanding_profiling_requests)
        Runtime::trigger_event(profiling_reported);
      // Always record these as part of profiling
      return true;
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
      legion_assert(count > 0);
      legion_assert(!mapped_event.has_triggered());
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void ReleaseOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_release(rez, target);
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
    ApEvent ReleaseOp::release_restrictions(
        const RegionRequirement& req, const VersionInfo& version_info,
        unsigned index, ApEvent precondition, ApEvent term_event,
        InstanceSet& restricted_instances,
        const std::vector<PhysicalManager*>& sources,
        const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& map_applied_events)
    //--------------------------------------------------------------------------
    {
      legion_assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
      legion_assert(IS_EXCLUSIVE(req));
      const bool known_targets = !restricted_instances.empty();
      RegionNode* region = runtime->get_node(req.region);
      ReleaseAnalysis* analysis =
          new ReleaseAnalysis(this, index, precondition, region, trace_info);
      analysis->add_reference();
      RtEvent views_ready;
      if (known_targets)
        views_ready =
            analysis->convert_views(req.region, restricted_instances, &sources);
      // Iterate through the equivalence classes and find all the restrictions
      const RtEvent traversal_done = analysis->perform_traversal(
          views_ready, version_info, map_applied_events);
      RtEvent remote_ready;
      if (traversal_done.exists() || analysis->has_remote_sets())
        remote_ready =
            analysis->perform_remote(traversal_done, map_applied_events);
      // Issue any release copies/fills that need to be done
      RtEvent updates_done =
          analysis->perform_updates(traversal_done, map_applied_events);
      // There are two cases here: one where we have the target intances
      // already from the operation and we know where to put the users
      // and the second case where we need to wait for the analysis to
      // tell us the names of the instances which are restricted
      const RegionUsage usage(req);
      std::vector<ApEvent> released_events;
      if (!known_targets)
      {
        if (remote_ready.exists() && !remote_ready.has_triggered())
          remote_ready.wait();
        op::FieldMaskMap<LogicalView> instances;
        analysis->report_instances(instances);
        analysis->target_instances.resize(instances.size());
        analysis->target_views.resize(instances.size());
        restricted_instances.resize(instances.size());
        unsigned inst_index = 0;
        // Note that all of these should be individual views.
        // The only way to get collective restricted view is by
        // doing attaches in control replicated contexts and we insist
        // that all release operations in control replicated context
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
      else
      {
        if (remote_ready.exists())
        {
          if (updates_done.exists())
            updates_done = Runtime::merge_events(updates_done, remote_ready);
          else
            updates_done = remote_ready;
        }
      }
      ApEvent instances_ready;
      analysis->perform_registration(
          updates_done, usage, map_applied_events, precondition, term_event,
          instances_ready);
      if (analysis->remove_reference())
        delete analysis;
      return instances_ready;
    }

    /////////////////////////////////////////////////////////////
    // Repl Release Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplReleaseOp::ReplReleaseOp(void)
      : ReplCollectiveViewCreator<CollectiveViewCreator<ReleaseOp> >()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplReleaseOp::~ReplReleaseOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplReleaseOp::initialize_replication(
        ReplicateContext* context, bool first_local_shard)
    //--------------------------------------------------------------------------
    {
      if (runtime->safe_mapper)
        sources_check = context->get_next_collective_index(COLLECTIVE_LOC_23);
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
        error << "Illegal use of grants with a release operation in control "
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
            << "Illegal use of wait phase barriers with a release operation in "
            << "control replicated parent task " << parent_ctx->get_task_name()
            << " (UID " << parent_ctx->get_unique_id() << "). Use of "
            << "non-canonical Legion features such as wait phase barriers are "
            << "not permitted with control replication.";
        error.raise();
      }
      if (!arrive_barriers.empty())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal use of arrive phase barriers with a release "
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
    void ReplReleaseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveViewCreator<CollectiveViewCreator<ReleaseOp> >::activate();
      sources_check = 0;
      collective_map_barrier = RtBarrier::NO_RT_BARRIER;
      is_first_local_shard = false;
    }

    //--------------------------------------------------------------------------
    void ReplReleaseOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      // Make sure we didn't leak our barrier
      legion_assert(!collective_map_barrier.exists());
      ReplCollectiveViewCreator<CollectiveViewCreator<ReleaseOp> >::deactivate(
          false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplReleaseOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      collective_map_barrier = repl_ctx->get_next_collective_map_barriers();
      // See if we need to make a collective view rendezvous
      if (restricted_region.impl->collective)
        create_collective_rendezvous(requirement.parent.get_tree_id(), 0);
      // Then do the base class analysis
      ReleaseOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void ReplReleaseOp::trigger_ready(void)
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
    RtEvent ReplReleaseOp::finalize_complete_mapping(RtEvent pre)
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
    bool ReplReleaseOp::perform_collective_analysis(
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
    void ReplReleaseOp::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(collective_map_barrier.exists());
      runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/);
      Runtime::advance_barrier(collective_map_barrier);
      elide_collective_rendezvous();
      ReleaseOp::predicate_false();
    }

    //--------------------------------------------------------------------------
    void ReplReleaseOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(collective_map_barrier.exists());
      // Elide both generations of the mapping fence barrier
      runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/);
      Runtime::advance_barrier(collective_map_barrier);
      runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/);
      collective_map_barrier = RtBarrier::NO_RT_BARRIER;
      elide_collective_rendezvous();
      ReleaseOp::trigger_replay();
    }

    //--------------------------------------------------------------------------
    void ReplReleaseOp::invoke_mapper(
        std::vector<PhysicalManager*>& source_instances)
    //--------------------------------------------------------------------------
    {
      // Do the base call
      ReleaseOp::invoke_mapper(source_instances);
      // If we're checking the mapping then do that now to make sure
      // all the shards have the same source instances
      if (runtime->safe_mapper)
      {
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(parent_ctx);
        CheckCollectiveSources sources_collective(repl_ctx, sources_check);
        if (!sources_collective.verify(source_instances))
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from the invocation of 'map_release' "
                << "by mapper " << mapper->get_mapper_name()
                << ". Mapper selected difference 'source_instances' "
                << "on shard 0 and shard " << repl_ctx->owner_shard->shard_id
                << " when mapping a release operation in "
                << "control-replicated parent task "
                << parent_ctx->get_task_name() << " (UID "
                << parent_ctx->get_unique_id()
                << "). Each release mapping in a control-replicated parent "
                   "task must provide the "
                << "same 'source_instances' across all the shards.";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    RtEvent ReplReleaseOp::perform_collective_versioning_analysis(
        unsigned index, LogicalRegion handle, EqSetTracker* tracker,
        const FieldMask& mask, unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      return rendezvous_collective_versioning_analysis(
          index, handle, tracker, runtime->address_space, mask,
          parent_req_index);
    }

    /////////////////////////////////////////////////////////////
    // Remote Release Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteReleaseOp::RemoteReleaseOp(Operation* ptr, AddressSpaceID src)
      : ExternalRelease(), RemoteOp(ptr, src)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RemoteReleaseOp::~RemoteReleaseOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID RemoteReleaseOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t RemoteReleaseOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteReleaseOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteReleaseOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteReleaseOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == nullptr)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const std::string_view& RemoteReleaseOp::get_provenance_string(
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
    const char* RemoteReleaseOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[RELEASE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind RemoteReleaseOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return RELEASE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteReleaseOp::select_sources(
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
      Mapper::SelectReleaseSrcInput input;
      Mapper::SelectReleaseSrcOutput output;
      prepare_for_mapping(
          sources, input.source_instances, input.collective_views);
      prepare_for_mapping(target, input.target);
      if (mapper == nullptr)
        mapper = runtime->find_mapper(map_id);
      mapper->invoke_select_release_sources(this, input, output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking, points);
    }

    //--------------------------------------------------------------------------
    void RemoteReleaseOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_release(rez, target);
      pack_profiling_requests(rez, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteReleaseOp::unpack(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      unpack_external_release(derez);
      unpack_profiling_requests(derez);
    }

  }  // namespace Internal
}  // namespace Legion
