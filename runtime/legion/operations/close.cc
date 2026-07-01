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

#include "legion/operations/close.h"
#include "legion/contexts/inner.h"
#include "legion/managers/mapper.h"
#include "legion/nodes/region.h"
#include "legion/tools/spy.h"
#include "legion/tracing/physical.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // External Close
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalClose::ExternalClose(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ExternalClose::pack_external_close(
        Serializer& rez, AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      pack_region_requirement(requirement, rez);
      rez.serialize(get_context_index());
    }

    //--------------------------------------------------------------------------
    void ExternalClose::unpack_external_close(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      unpack_region_requirement(requirement, derez);
      uint64_t index;
      derez.deserialize(index);
      set_context_index(index);
    }

    /////////////////////////////////////////////////////////////
    // Close Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CloseOp::CloseOp(void) : InternalOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    CloseOp::~CloseOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID CloseOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t CloseOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void CloseOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int CloseOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* CloseOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == nullptr)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const std::string_view& CloseOp::get_provenance_string(bool human) const
    //--------------------------------------------------------------------------
    {
      Provenance* provenance = get_provenance();
      if (provenance != nullptr)
        return human ? provenance->human : provenance->machine;
      else
        return Provenance::no_provenance;
    }

    //--------------------------------------------------------------------------
    Mappable* CloseOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    size_t CloseOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    const FieldMask& CloseOp::get_internal_mask(void) const
    //--------------------------------------------------------------------------
    {
      // should only be called by inherited classes
      std::abort();
    }

    //--------------------------------------------------------------------------
    void CloseOp::initialize_close(
        InnerContext* ctx, const RegionRequirement& req)
    //--------------------------------------------------------------------------
    {
      // Only initialize the operation here, this is not a trace-able op
      initialize_operation(ctx);
      // Never track this so don't get the close index
      parent_task = ctx->get_task();
      requirement = req;
    }

    //--------------------------------------------------------------------------
    void CloseOp::initialize_close(
        Operation* creator, unsigned idx, const RegionRequirement& req)
    //--------------------------------------------------------------------------
    {
      initialize_internal(creator, idx);
      // We always track this so get the close index
      parent_task = parent_ctx->get_task();
      requirement = req;
    }

    //--------------------------------------------------------------------------
    void CloseOp::perform_logging(
        Operation* creator, unsigned index, bool merge)
    //--------------------------------------------------------------------------
    {
      if (spy_logging_level == NO_SPY_LOGGING)
        return;
      LegionSpy::log_close_operation(
          parent_ctx->get_unique_id(), unique_op_id, merge);
      LegionSpy::log_internal_op_creator(
          unique_op_id, creator->get_unique_op_id(), index);
      if (requirement.handle_type == LEGION_PARTITION_PROJECTION)
        LegionSpy::log_logical_requirement(
            unique_op_id, 0 /*idx*/, false /*region*/,
            requirement.partition.index_partition.get_id(),
            requirement.partition.field_space.get_id(),
            requirement.partition.get_tree_id(), requirement.privilege,
            requirement.prop, requirement.redop,
            requirement.parent.index_space.get_id());
      else
        LegionSpy::log_logical_requirement(
            unique_op_id, 0 /*idx*/, true /*region*/,
            requirement.region.index_space.get_id(),
            requirement.region.field_space.get_id(),
            requirement.region.get_tree_id(), requirement.privilege,
            requirement.prop, requirement.redop,
            requirement.parent.index_space.get_id());
      LegionSpy::log_requirement_fields(
          unique_op_id, 0 /*idx*/, requirement.privilege_fields);
    }

    //--------------------------------------------------------------------------
    void CloseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      InternalOp::activate();
    }

    //--------------------------------------------------------------------------
    void CloseOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (mapper_data != nullptr)
      {
        free(mapper_data);
        mapper_data = nullptr;
        mapper_data_size = 0;
      }
      InternalOp::deactivate(freeop);
    }

    //--------------------------------------------------------------------------
    void CloseOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      commit_operation(true /*deactivate*/);
    }

    /////////////////////////////////////////////////////////////
    // Inter Close Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MergeCloseOp::MergeCloseOp(void) : CloseOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    MergeCloseOp::~MergeCloseOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void MergeCloseOp::initialize(
        InnerContext* ctx, const RegionRequirement& req, int close_idx,
        Operation* creator)
    //--------------------------------------------------------------------------
    {
      initialize_close(creator, close_idx, req);
      if (tracing)
      {
        parent_req_index = creator->find_parent_index(close_idx);
#ifdef LEGION_DEBUG_COLLECTIVES
        trace->register_close(
            this, creator_req_idx,
            (req.handle_type == LEGION_SINGULAR_PROJECTION) ?
                (RegionTreeNode*)runtime->get_node(req.region) :
                (RegionTreeNode*)runtime->get_node(req.partition),
            req);
#else
        trace->register_close(this, creator_req_idx, req);
#endif
      }
      else
      {
        if (trace == nullptr)
          parent_req_index = creator->find_parent_index(close_idx);
        trace = nullptr;
      }
    }

    //--------------------------------------------------------------------------
    void MergeCloseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      CloseOp::activate();
      parent_req_index = TRACED_PARENT_INDEX;
    }

    //--------------------------------------------------------------------------
    void MergeCloseOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      close_mask.clear();
      CloseOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* MergeCloseOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[MERGE_CLOSE_OP_KIND];
    }

    //-------------------------------------------------------------------------
    OpKind MergeCloseOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return MERGE_CLOSE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    const FieldMask& MergeCloseOp::get_internal_mask(void) const
    //--------------------------------------------------------------------------
    {
      return close_mask;
    }

    //--------------------------------------------------------------------------
    unsigned MergeCloseOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx == 0);
      legion_assert(parent_req_index != TRACED_PARENT_INDEX);
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void MergeCloseOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Populate our privilege fields
      RegionTreeNode* node =
          (requirement.handle_type == LEGION_SINGULAR_PROJECTION) ?
              (RegionTreeNode*)runtime->get_node(requirement.region) :
              (RegionTreeNode*)runtime->get_node(requirement.partition);
      node->column_source->get_field_set(
          close_mask, parent_ctx, requirement.privilege_fields);
      // Do our logging
      perform_logging(create_op, creator_req_idx, true /*merge close*/);
    }

    //--------------------------------------------------------------------------
    void MergeCloseOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      complete_mapping();
      complete_execution();
    }

    /////////////////////////////////////////////////////////////
    // Post Close Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PostCloseOp::PostCloseOp(void) : CloseOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PostCloseOp::~PostCloseOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void PostCloseOp::initialize(
        InnerContext* ctx, unsigned idx, const InstanceSet& targets)
    //--------------------------------------------------------------------------
    {
      initialize_close(ctx, ctx->regions[idx]);
      parent_idx = idx;
      target_instances = targets;
      localize_region_requirement(requirement);
      perform_logging(ctx->owner_task, idx, false /*merge close*/);
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      CloseOp::activate();
      mapper = nullptr;
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      map_applied_conditions.clear();
      profiling_requests.clear();
      target_instances.clear();
      version_info.clear();
      CloseOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* PostCloseOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[POST_CLOSE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind PostCloseOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return POST_CLOSE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // This stage is only done for close operations issued
      // at the end of the task as dependence analysis for other
      // close operations is done inline in the region tree traversal
      // for other kinds of operations
      // see RegionTreeNode::register_logical_node
      analyze_region_requirements();
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      perform_versioning_analysis(
          0 /*idx*/, requirement, version_info, preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      const PhysicalTraceInfo trace_info(this, 0 /*index*/);
      ApUserEvent close_event = Runtime::create_ap_user_event(&trace_info);
      std::vector<PhysicalManager*> dummy_sources;
      ApEvent instances_ready = physical_perform_updates_and_registration(
          requirement, version_info, 0 /*idx*/, ApEvent::NO_AP_EVENT,
          close_event, target_instances, dummy_sources, trace_info,
          map_applied_conditions, false /*check collective*/,
          true /*record valid*/, false /*check initialized*/);
      Runtime::trigger_event(
          close_event, instances_ready, trace_info, map_applied_conditions);
      record_completion_effect(close_event);
      log_mapping_decision(0 /*idx*/, requirement, target_instances);
      // No need to apply our mapping because we are done!
      RtEvent mapping_applied;
      if (!map_applied_conditions.empty())
        mapping_applied = Runtime::merge_events(map_applied_conditions);
      if (!acquired_instances.empty())
        mapping_applied = release_nonempty_acquired_instances(
            mapping_applied, acquired_instances);
      complete_mapping(mapping_applied);
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if we need to do a profiling response
      if (profiling_reported.exists() && (outstanding_profiling_requests == 0))
      {
        // We're not expecting any profiling callbacks so we need to
        // do one ourself to inform the mapper that there won't be any
        Mapping::Mapper::CloseProfilingInfo info;
        info.total_reports = 0;
        info.fill_response = false;  // make valgrind happy
        mapper->invoke_close_report_profiling(this, info);
        Runtime::trigger_event(profiling_reported);
      }
      // Only commit this operation if we are done profiling
      commit_operation(true /*deactivate*/, profiling_reported);
    }

    //--------------------------------------------------------------------------
    unsigned PostCloseOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx == 0);
      legion_assert(parent_idx != TRACED_PARENT_INDEX);
      return parent_idx;
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::select_sources(
        const unsigned index, PhysicalManager* target,
        const std::vector<InstanceView*>& sources,
        std::vector<unsigned>& ranking,
        std::map<unsigned, PhysicalManager*>& points)
    //--------------------------------------------------------------------------
    {
      legion_assert(index == 0);
      Mapper::SelectCloseSrcInput input;
      Mapper::SelectCloseSrcOutput output;
      prepare_for_mapping(target, input.target);
      prepare_for_mapping(
          sources, input.source_instances, input.collective_views);
      if (mapper == nullptr)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_select_close_sources(this, input, output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking, points);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*, unsigned>*
        PostCloseOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    int PostCloseOp::add_copy_profiling_request(
        const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& requests,
        bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any profiling requests
      if (profiling_requests.empty())
        return 0;
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
      return 0;
    }

    //--------------------------------------------------------------------------
    bool PostCloseOp::handle_profiling_response(
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
      Mapping::Mapper::CloseProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      info.total_reports = outstanding_profiling_requests;
      info.fill_response = op_info->fill;
      mapper->invoke_close_report_profiling(this, info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
      legion_assert(count <= outstanding_profiling_requests);
      if (count == outstanding_profiling_requests)
        Runtime::trigger_event(profiling_reported);
      // Always record these as part of profiling
      return true;
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
      legion_assert(count > 0);
      legion_assert(!mapped_event.has_triggered());
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void PostCloseOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_close(rez, target);
      rez.serialize<int>(0);
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
    // Repl Merge Close Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplMergeCloseOp::ReplMergeCloseOp(void) : MergeCloseOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplMergeCloseOp::~ReplMergeCloseOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplMergeCloseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      MergeCloseOp::activate();
      mapped_barrier = RtBarrier::NO_RT_BARRIER;
    }

    //--------------------------------------------------------------------------
    void ReplMergeCloseOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      MergeCloseOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplMergeCloseOp::set_repl_close_info(RtBarrier mapped)
    //--------------------------------------------------------------------------
    {
      legion_assert(!mapped_barrier.exists());
      mapped_barrier = mapped;
    }

    //--------------------------------------------------------------------------
    void ReplMergeCloseOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(mapped_barrier.exists());
      runtime->phase_barrier_arrive(mapped_barrier, 1 /*count*/);
      // Then complete the mapping once the barrier has triggered
      // A small performance optimization here: if we have a physical trace
      // and we're replaying it then we don't need to actually do the
      // synchronization across the shards since we know all the shards
      // can replay independently
      if ((trace != nullptr) && trace->has_physical_trace())
      {
        PhysicalTrace* physical = trace->get_physical_trace();
        if (physical->is_replaying())
          complete_mapping();
        else
          complete_mapping(mapped_barrier);
      }
      else
        complete_mapping(mapped_barrier);
      complete_execution();
    }

    /////////////////////////////////////////////////////////////
    // Remote Close Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteCloseOp::RemoteCloseOp(Operation* ptr, AddressSpaceID src)
      : ExternalClose(), RemoteOp(ptr, src)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RemoteCloseOp::~RemoteCloseOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID RemoteCloseOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t RemoteCloseOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteCloseOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteCloseOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteCloseOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == nullptr)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const std::string_view& RemoteCloseOp::get_provenance_string(
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
    const char* RemoteCloseOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[POST_CLOSE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind RemoteCloseOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return POST_CLOSE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteCloseOp::select_sources(
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
      Mapper::SelectCloseSrcInput input;
      Mapper::SelectCloseSrcOutput output;
      prepare_for_mapping(
          sources, input.source_instances, input.collective_views);
      prepare_for_mapping(target, input.target);
      if (mapper == nullptr)
        mapper = runtime->find_mapper(map_id);
      mapper->invoke_select_close_sources(this, input, output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking, points);
    }

    //--------------------------------------------------------------------------
    void RemoteCloseOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_close(rez, target);
      pack_profiling_requests(rez, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteCloseOp::unpack(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      unpack_external_close(derez);
      unpack_profiling_requests(derez);
    }

  }  // namespace Internal
}  // namespace Legion
