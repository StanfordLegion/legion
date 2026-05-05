/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#include "legion/operations/fill.h"
#include "legion/analysis/overwrite.h"
#include "legion/contexts/replicate.h"
#include "legion/api/future_impl.h"
#include "legion/managers/mapper.h"
#include "legion/managers/shard.h"
#include "legion/nodes/region.h"
#include "legion/tracing/recognizer.h"
#include "legion/utilities/provenance.h"
#include "legion/views/fill.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // External Fill
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalFill::ExternalFill(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ExternalFill::pack_external_fill(
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
      rez.serialize<bool>(is_index_space);
      rez.serialize(index_domain);
      rez.serialize(index_point);
      pack_mappable(*this, rez);
      rez.serialize(get_context_index());
    }

    //--------------------------------------------------------------------------
    void ExternalFill::unpack_external_fill(Deserializer& derez)
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
      derez.deserialize<bool>(is_index_space);
      derez.deserialize(index_domain);
      derez.deserialize(index_point);
      unpack_mappable(*this, derez);
      uint64_t index;
      derez.deserialize(index);
      set_context_index(index);
    }

    /////////////////////////////////////////////////////////////
    // Fill Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FillOp::FillOp(void) : PredicatedOp(), ExternalFill()
    //--------------------------------------------------------------------------
    {
      this->is_index_space = false;
    }

    //--------------------------------------------------------------------------
    FillOp::~FillOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void FillOp::initialize(
        InnerContext* ctx, const FillLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      parent_task = ctx->get_task();
      initialize_predication(ctx, launcher.predicate, provenance);
      requirement = RegionRequirement(
          launcher.handle, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE,
          launcher.parent);
      requirement.privilege_fields = launcher.fields;
      if (runtime->safe_model)
        verify_requirement(requirement);
      parent_req_index = ctx->find_parent_region_index(this, requirement);
      if (launcher.future.impl != nullptr)
        future = launcher.future;
      else if (launcher.argument.get_size() > 0)
      {
        value_size = launcher.argument.get_size();
        value = malloc(value_size);
        memcpy(value, launcher.argument.get_ptr(), value_size);
      }
      else
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "No fill value found for " << *this << ". All fill operations "
              << "must be given a non-empty argument or a future to use as a "
              << "fill value.";
        error.raise();
      }
      grants = launcher.grants;
      wait_barriers = launcher.wait_barriers;
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
      index_point = launcher.point;
      index_domain = Domain(index_point, index_point);
      sharding_space = launcher.sharding_space;
      LegionSpy::log_fill_operation(parent_ctx->get_unique_id(), unique_op_id);
      if (future.impl != nullptr)
        LegionSpy::log_future_use(unique_op_id, future.impl->did);
    }

    //--------------------------------------------------------------------------
    void FillOp::activate(void)
    //--------------------------------------------------------------------------
    {
      PredicatedOp::activate();
      fill_view_ready = RtEvent::NO_RT_EVENT;
      fill_view = nullptr;
      value = nullptr;
      value_size = 0;
      set_view = false;
    }

    //--------------------------------------------------------------------------
    void FillOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      version_info.clear();
      map_applied_conditions.clear();
      future = Future();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      if (value != nullptr)
        free(value);
      if (mapper_data != nullptr)
      {
        free(mapper_data);
        mapper_data = nullptr;
        mapper_data_size = 0;
      }
      PredicatedOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* FillOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[FILL_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind FillOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return FILL_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t FillOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    Mappable* FillOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    UniqueID FillOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t FillOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void FillOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int FillOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* FillOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == nullptr)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const std::string_view& FillOp::get_provenance_string(bool human) const
    //--------------------------------------------------------------------------
    {
      Provenance* provenance = get_provenance();
      if (provenance != nullptr)
        return human ? provenance->human : provenance->machine;
      else
        return Provenance::no_provenance;
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*, unsigned>* FillOp::get_acquired_instances_ref(
        void)
    //--------------------------------------------------------------------------
    {
      // Fill Ops should never actually need this, but this method might
      // be called in the process of doing a mapper call
      return nullptr;
    }

    //--------------------------------------------------------------------------
    int FillOp::add_copy_profiling_request(
        const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& reqeusts,
        bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do for the moment
      return 0;
    }

    //--------------------------------------------------------------------------
    FillView* FillOp::get_fill_view(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(fill_view != nullptr);
      return fill_view;
    }

    //--------------------------------------------------------------------------
    void FillOp::log_fill_requirement(void) const
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
    void FillOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent index
      log_fill_requirement();
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_base_dependence_analysis();
      analyze_region_requirements();
    }

    //--------------------------------------------------------------------------
    void FillOp::perform_base_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(
            this, wait_barriers, arrive_barriers);
      // If we are waiting on a future register a dependence
      if (future.impl != nullptr)
        future.impl->register_dependence(this);
      legion_assert(fill_view == nullptr);
      // Note that this returns a fill view with a reference added to it
      // We remove the reference in trigger_complete
      if (future.impl != nullptr)
        fill_view = parent_ctx->find_or_create_fill_view(
            this, future, set_view, fill_view_ready);
      else
        fill_view = parent_ctx->find_or_create_fill_view(
            this, value, value_size, fill_view_ready);
    }

    //--------------------------------------------------------------------------
    void FillOp::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      // Otherwise do the work to clean up this operation
      // Mark that this operation has completed both
      // execution and mapping indicating that we are done
      // Do it in this order to avoid calling 'execute_trigger'
      complete_execution();
      if (!map_applied_conditions.empty())
        complete_mapping(finalize_complete_mapping(
            Runtime::merge_events(map_applied_conditions)));
      else
        complete_mapping(finalize_complete_mapping(RtEvent::NO_RT_EVENT));
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      if (parent_req_index == TRACED_PARENT_INDEX)
        parent_req_index = parent_ctx->find_parent_region_index(
            this, requirement, 0 /*idx*/, false /*skip privileges*/,
            true /*force*/);
      std::set<RtEvent> preconditions;
      if (fill_view_ready.exists())
        preconditions.insert(fill_view_ready);
      perform_versioning_analysis(
          0 /*idx*/, requirement, version_info, preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      const PhysicalTraceInfo trace_info(this, 0 /*index*/);
      // This is nullptr for now until we implement tracing for fills
      const ApEvent init_precondition = compute_sync_precondition(trace_info);
      fill_fields(get_fill_view(), init_precondition, trace_info);
      if (set_view)
      {
        legion_assert(future.impl != nullptr);
        // This will make sure we have a mapping locally
        future.impl->request_runtime_instance(this);
      }
      if (is_recording())
        trace_info.record_complete_replay(map_applied_conditions);
      if (!map_applied_conditions.empty())
        complete_mapping(finalize_complete_mapping(
            Runtime::merge_events(map_applied_conditions)));
      else
        complete_mapping(finalize_complete_mapping(RtEvent::NO_RT_EVENT));
      if (set_view)
      {
        const RtEvent future_ready_event =
            future.impl->find_runtime_instance_ready();
        complete_execution(future_ready_event);
      }
      else
        complete_execution();
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_complete(ApEvent complete)
    //--------------------------------------------------------------------------
    {
      // Now that we've mapped we can remove the reference on our fill_view
      if (fill_view != nullptr)
      {
        if (set_view)
        {
          size_t value_size = 0;
          const void* value =
              future.impl->find_runtime_buffer(parent_ctx, value_size);
          if (fill_view->set_value(value, value_size))
            delete fill_view;
        }
        if (fill_view->remove_base_valid_ref(MAPPING_ACQUIRE_REF))
          delete fill_view;
      }
      // See if we have any arrivals to trigger
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
    bool FillOp::record_trace_hash(TraceHashRecorder& recorder, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      Murmur3Hasher hasher;
      hasher.hash(get_operation_kind());
      hash_requirement(hasher, requirement);
      hasher.hash<bool>(is_index_space);
      if (is_index_space)
        hasher.hash(index_domain);
      if (future.exists())
        hasher.hash(future.impl->did);
      else
        hasher.hash(value, value_size);
      return recorder.record_operation_hash(this, hasher, opidx);
    }

    //--------------------------------------------------------------------------
    unsigned FillOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx == 0);
      legion_assert(parent_req_index != TRACED_PARENT_INDEX);
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      commit_operation(true /*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void FillOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_replay_operation(unique_op_id);
      complete_mapping(finalize_complete_mapping(RtEvent::NO_RT_EVENT));
    }

    //--------------------------------------------------------------------------
    void FillOp::complete_replay(ApEvent fill_complete_event)
    //--------------------------------------------------------------------------
    {
      record_completion_effect(fill_complete_event);
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void FillOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_fill(rez, target);
    }

    //--------------------------------------------------------------------------
    void FillOp::fill_fields(
        FillView* fill_view, ApEvent precondition,
        const PhysicalTraceInfo& trace_info)
    //--------------------------------------------------------------------------
    {
      legion_assert(requirement.handle_type == LEGION_SINGULAR_PROJECTION);
      RegionNode* region_node = runtime->get_node(requirement.region);
      bool first_local = true;
      CollectiveMapping* collective_mapping = nullptr;
      perform_collective_analysis(collective_mapping, first_local);
      OverwriteAnalysis* analysis = new OverwriteAnalysis(
          this, 0 /*index*/, RegionUsage(requirement), region_node->row_source,
          fill_view, version_info.get_valid_mask(), trace_info,
          collective_mapping, precondition, true_guard, false_guard,
          false /*add restriction*/, first_local);
      analysis->add_reference();
      const RtEvent traversal_done = analysis->perform_traversal(
          RtEvent::NO_RT_EVENT, version_info, map_applied_conditions);
      if (traversal_done.exists() || analysis->has_remote_sets())
        analysis->perform_remote(traversal_done, map_applied_conditions);
      if (traversal_done.exists() || analysis->has_output_updates())
        analysis->perform_output(traversal_done, map_applied_conditions);
      if (analysis->remove_reference())
        delete analysis;
    }

    /////////////////////////////////////////////////////////////
    // Index Fill Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexFillOp::IndexFillOp(void) : PointwiseAnalyzable<FillOp>()
    //--------------------------------------------------------------------------
    {
      this->is_index_space = true;
    }

    //--------------------------------------------------------------------------
    IndexFillOp::~IndexFillOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void IndexFillOp::initialize(
        InnerContext* ctx, const IndexFillLauncher& launcher,
        IndexSpace launch_sp, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      parent_task = ctx->get_task();
      initialize_predication(ctx, launcher.predicate, provenance);
      legion_assert(launch_sp.exists());
      launch_space = runtime->get_node(launch_sp);
      add_launch_space_reference(launch_space);
      if (!launcher.launch_domain.exists())
        index_domain = launch_space->get_tight_domain();
      else
        index_domain = launcher.launch_domain;
      sharding_space = launcher.sharding_space;
      if (launcher.region.exists())
      {
        legion_assert(!launcher.partition.exists());
        requirement = RegionRequirement(
            launcher.region, launcher.projection, LEGION_WRITE_DISCARD,
            LEGION_EXCLUSIVE, launcher.parent);
      }
      else
      {
        legion_assert(launcher.partition.exists());
        requirement = RegionRequirement(
            launcher.partition, launcher.projection, LEGION_WRITE_DISCARD,
            LEGION_EXCLUSIVE, launcher.parent);
      }
      requirement.privilege_fields = launcher.fields;
      if (runtime->safe_model)
        verify_requirement(requirement, 0 /*index*/, true /*allow projection*/);
      parent_req_index = ctx->find_parent_region_index(this, requirement);
      // Note that this returns a fill view with a reference added to it
      if (launcher.future.impl != nullptr)
        future = launcher.future;
      else if (launcher.argument.get_size() > 0)
      {
        value_size = launcher.argument.get_size();
        value = malloc(value_size);
        memcpy(value, launcher.argument.get_ptr(), value_size);
      }
      else
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "No fill value found for " << *this << ". All fill operations "
              << "must be given a non-empty argument or a future to use as a "
              << "fill value.";
        error.raise();
      }
      grants = launcher.grants;
      wait_barriers = launcher.wait_barriers;
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
      LegionSpy::log_fill_operation(parent_ctx->get_unique_id(), unique_op_id);
      if (future.impl != nullptr)
        LegionSpy::log_future_use(unique_op_id, future.impl->did);
      log_launch_space(launch_space->handle);
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::activate(void)
    //--------------------------------------------------------------------------
    {
      PointwiseAnalyzable<FillOp>::activate();
      index_domain = Domain::NO_DOMAIN;
      sharding_space = IndexSpace::NO_SPACE;
      launch_space = nullptr;
      points_completed.store(0);
      points_committed = 0;
      commit_request = false;
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      // We can deactivate our point operations
      for (PointFillOp* point : points) point->deactivate();
      points.clear();
      pending_pointwise_dependences.clear();
      if (remove_launch_space_reference(launch_space))
        delete launch_space;
      PointwiseAnalyzable<FillOp>::deactivate(false /*free*/);
      // Return the operation to the runtime
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // Promote a singular region requirement up to a projection
      if (requirement.handle_type == LEGION_SINGULAR_PROJECTION)
      {
        requirement.handle_type = LEGION_REGION_PROJECTION;
        requirement.projection = 0;
      }
      log_index_fill_requirement();
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::log_index_fill_requirement(void)
    //--------------------------------------------------------------------------
    {
      if (spy_logging_level == NO_SPY_LOGGING)
        return;
      const bool reg =
          (requirement.handle_type == LEGION_SINGULAR_PROJECTION) ||
          (requirement.handle_type == LEGION_REGION_PROJECTION);
      const bool proj =
          (requirement.handle_type == LEGION_REGION_PROJECTION) ||
          (requirement.handle_type == LEGION_PARTITION_PROJECTION);

      LegionSpy::log_logical_requirement(
          unique_op_id, 0 /*idx*/, reg,
          reg ? requirement.region.index_space.get_id() :
                requirement.partition.index_partition.get_id(),
          reg ? requirement.region.field_space.get_id() :
                requirement.partition.field_space.get_id(),
          reg ? requirement.region.get_tree_id() :
                requirement.partition.get_tree_id(),
          requirement.privilege, requirement.prop, requirement.redop,
          requirement.parent.index_space.get_id());
      LegionSpy::log_requirement_fields(
          unique_op_id, 0 /*idx*/, requirement.privilege_fields);
      if (proj)
        LegionSpy::log_requirement_projection(
            unique_op_id, 0 /*idx*/, requirement.projection);
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_base_dependence_analysis();
      analyze_region_requirements(launch_space);
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      if (parent_req_index == TRACED_PARENT_INDEX)
        parent_req_index = parent_ctx->find_parent_region_index(
            this, requirement, 0 /*idx*/, false /*skip privileges*/,
            true /*force*/);
      // Enumerate the points
      enumerate_points();
      if (future.impl != nullptr)
      {
        legion_assert(future.impl != nullptr);
        // This will make sure we have a mapping locally
        future.impl->request_runtime_instance(this);
      }
      // Launch the points
      for (unsigned idx = 0; idx < points.size(); idx++)
      {
        map_applied_conditions.insert(points[idx]->get_mapped_event());
        points[idx]->launch(fill_view_ready);
      }
      // Record that we are mapped when all our points are mapped
      // and we are executed when all our points are executed
      complete_mapping(Runtime::merge_events(map_applied_conditions));
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::handle_point_complete(ApEvent effect)
    //--------------------------------------------------------------------------
    {
      if (effect.exists())
        record_completion_effect(effect);
      const unsigned received = ++points_completed;
      legion_assert(received <= points.size());
      if (received == points.size())
      {
        if (set_view)
        {
          const RtEvent future_ready_event =
              future.impl->find_runtime_instance_ready();
          complete_execution(future_ready_event);
        }
        else
          complete_execution();
      }
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      bool commit_now = false;
      {
        AutoLock o_lock(op_lock);
        legion_assert(!commit_request);
        commit_request = true;
        commit_now = (points.size() == points_committed);
      }
      if (commit_now)
        commit_operation(true /*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_replaying());
      LegionSpy::log_replay_operation(unique_op_id);
      // Enumerate the points
      enumerate_points();
      // Then call replay analysis on all of them
      std::vector<RtEvent> mapped_preconditions(points.size());
      for (unsigned idx = 0; idx < points.size(); idx++)
      {
        mapped_preconditions[idx] = points[idx]->get_mapped_event();
        points[idx]->trigger_replay();
      }
      complete_mapping(Runtime::merge_events(mapped_preconditions));
    }

    //--------------------------------------------------------------------------
    size_t IndexFillOp::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      return get_shard_points()->get_volume();
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::enumerate_points(void)
    //--------------------------------------------------------------------------
    {
      // Enumerate the points
      // Need to get the launch domain in case it is different than
      // the original index domain due to control replication
      IndexSpaceNode* local_points = get_shard_points();
      Domain launch_domain = local_points->get_tight_domain();
      // Now enumerate the points
      size_t num_points = local_points->get_volume();
      legion_assert(num_points > 0);
      std::vector<PointFillOp*> temp_points;
      temp_points.reserve(num_points);
      for (Domain::DomainPointIterator itr(launch_domain); itr; itr++)
      {
        PointFillOp* point = runtime->get_operation<PointFillOp>();
        point->initialize(this, itr.p);
        temp_points.emplace_back(point);
      }
      // Now we have to do the projection
      ProjectionFunction* function =
          runtime->find_projection_function(requirement.projection);
      std::vector<ProjectionPoint*> projection_points(
          temp_points.begin(), temp_points.end());
      function->project_points(
          this, 0 /*idx*/, requirement, index_domain, projection_points,
          pointwise_dependences.empty() ?
              nullptr :
              &pointwise_dependences.begin()->second,
          parent_ctx->get_total_shards(), is_replaying());
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        for (PointFillOp* point : temp_points) point->log_fill_requirement();
      }
      // Need the lock to avoid races with the pointwise dependence analysis
      AutoLock o_lock(op_lock);
      legion_assert(points.empty());
      points.swap(temp_points);
      // See if we have any pending pointwise dependences to trigger
      for (const std::pair<const DomainPoint, RtUserEvent>& pit :
           pending_pointwise_dependences)
      {
        PointFillOp* point = nullptr;
        for (PointFillOp* it : points)
        {
          if (pit.first != it->index_point)
            continue;
          point = it;
          break;
        }
        legion_assert(point != nullptr);
        Runtime::trigger_event(pit.second, point->get_mapped_event());
      }
    }

    //--------------------------------------------------------------------------
    RtEvent IndexFillOp::find_pointwise_dependence(
        const DomainPoint& point, GenerationID needed_gen, bool intra_space,
        RtUserEvent to_trigger, std::optional<unsigned> output_index)
    //--------------------------------------------------------------------------
    {
      legion_assert(!intra_space);
      legion_assert(!output_index);
      AutoLock o_lock(op_lock);
      legion_assert(needed_gen <= gen);
      if ((needed_gen < gen) || mapped ||
          (predication_state == PREDICATED_FALSE_STATE))
      {
        if (to_trigger.exists())
          Runtime::trigger_event(to_trigger);
        return RtEvent::NO_RT_EVENT;
      }
      if (points.empty())
      {
        std::map<DomainPoint, RtUserEvent>::const_iterator finder =
            pending_pointwise_dependences.find(point);
        if (finder != pending_pointwise_dependences.end())
        {
          if (to_trigger.exists())
          {
            Runtime::trigger_event(to_trigger, finder->second);
            return to_trigger;
          }
          else
            return finder->second;
        }
        if (!to_trigger.exists())
          to_trigger = Runtime::create_rt_user_event();
        pending_pointwise_dependences.emplace(
            std::make_pair(point, to_trigger));
        return to_trigger;
      }
      for (PointFillOp* it : points)
      {
        if (point != it->index_point)
          continue;
        if (to_trigger.exists())
        {
          Runtime::trigger_event(to_trigger, it->get_mapped_event());
          return to_trigger;
        }
        else
          return it->get_mapped_event();
      }
      // Should never get here, if we do that means we couldn't find the point
      std::abort();
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      // Trigger any pending pointwise dependences since they will not
      // be run, safe to do without the lock because we are protected
      // by the predication_state having been set before this
      if (!pending_pointwise_dependences.empty())
      {
        for (const std::pair<const DomainPoint, RtUserEvent>& it :
             pending_pointwise_dependences)
          Runtime::trigger_event(it.second);
        pending_pointwise_dependences.clear();
      }
      FillOp::predicate_false();
    }

    //--------------------------------------------------------------------------
    void IndexFillOp::handle_point_commit(void)
    //--------------------------------------------------------------------------
    {
      bool commit_now = false;
      {
        AutoLock o_lock(op_lock);
        points_committed++;
        commit_now = commit_request && (points.size() == points_committed);
      }
      if (commit_now)
        commit_operation(true /*deactivate*/);
    }

    /////////////////////////////////////////////////////////////
    // Point Fill Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointFillOp::PointFillOp(void) : FillOp()
    //--------------------------------------------------------------------------
    {
      this->is_index_space = true;
    }

    //--------------------------------------------------------------------------
    PointFillOp::~PointFillOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void PointFillOp::initialize(IndexFillOp* own, const DomainPoint& p)
    //--------------------------------------------------------------------------
    {
      // Initialize the operation
      initialize_operation(own->get_context(), own->get_provenance());
      index_point = p;
      index_domain = own->index_domain;
      sharding_space = own->sharding_space;
      owner = own;
      context_index = own->get_context_index();
      exception_handler = own->get_exception_handler();
      execution_fence_event = own->get_execution_fence_event();
      // From Memoizable
      trace_local_id = owner->get_trace_local_id().context_index;
      tpl = owner->get_template();
      if (tpl != nullptr)
        memo_state = owner->get_memoizable_state();
      // From Fill
      requirement = owner->get_requirement();
      grants = owner->grants;
      wait_barriers = owner->wait_barriers;
      parent_task = owner->parent_task;
      map_id = owner->map_id;
      tag = owner->tag;
      mapper_data_size = owner->mapper_data_size;
      if (mapper_data_size > 0)
      {
        legion_assert(mapper_data == nullptr);
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, owner->mapper_data, mapper_data_size);
      }
      // From FillOp
      true_guard = owner->true_guard;
      false_guard = owner->false_guard;
      version_info = owner->version_info;
      LegionSpy::log_index_point(owner->get_unique_op_id(), unique_op_id, p);
    }

    //--------------------------------------------------------------------------
    void PointFillOp::activate(void)
    //--------------------------------------------------------------------------
    {
      FillOp::activate();
      owner = nullptr;
    }

    //--------------------------------------------------------------------------
    void PointFillOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      pointwise_mapping_dependences.clear();
      FillOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void PointFillOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PointFillOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PointFillOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PointFillOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      memo_state = MEMO_REPLAY;
      tpl->register_operation(this);
      FillOp::trigger_replay();
    }

    //--------------------------------------------------------------------------
    void PointFillOp::launch(RtEvent view_ready)
    //--------------------------------------------------------------------------
    {
      // Perform the version info
      std::set<RtEvent> preconditions;
      if (!pointwise_mapping_dependences.empty())
        preconditions.insert(
            pointwise_mapping_dependences.begin(),
            pointwise_mapping_dependences.end());
      if (view_ready.exists())
        preconditions.insert(view_ready);
      perform_versioning_analysis(
          0 /*idx*/, requirement, version_info, preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void PointFillOp::trigger_complete(ApEvent effects)
    //--------------------------------------------------------------------------
    {
      owner->handle_point_complete(effects);
      complete_operation();
    }

    //--------------------------------------------------------------------------
    void PointFillOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      // Don't commit this operation until we've reported our profiling
      // Out index owner will deactivate the operation
      commit_operation(false /*deactivate*/);
      // Tell our owner that we are done, they will do the deactivate
      owner->handle_point_commit();
    }

    //--------------------------------------------------------------------------
    FillView* PointFillOp::get_fill_view(void) const
    //--------------------------------------------------------------------------
    {
      return owner->get_fill_view();
    }

    //--------------------------------------------------------------------------
    size_t PointFillOp::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      return owner->get_collective_points();
    }

    //--------------------------------------------------------------------------
    bool PointFillOp::find_shard_participants(std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      return owner->find_shard_participants(shards);
    }

    //--------------------------------------------------------------------------
    const DomainPoint& PointFillOp::get_domain_point(void) const
    //--------------------------------------------------------------------------
    {
      return index_point;
    }

    //--------------------------------------------------------------------------
    void PointFillOp::set_projection_result(unsigned idx, LogicalRegion result)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx == 0);
      requirement.region = result;
      requirement.handle_type = LEGION_SINGULAR_PROJECTION;
    }

    //--------------------------------------------------------------------------
    void PointFillOp::record_intra_space_dependences(
        unsigned index, const std::vector<DomainPoint>& dependences)
    //--------------------------------------------------------------------------
    {
      // Ignore any intra-space requirements on fills, we know that they
      // are all filling the same value so they can be done in any order
    }

    //--------------------------------------------------------------------------
    void PointFillOp::record_pointwise_dependence(
        uint64_t previous_context_index, const DomainPoint& previous_point,
        ShardID shard)
    //--------------------------------------------------------------------------
    {
      legion_assert(previous_context_index < context_index);
      const RtEvent pre = parent_ctx->find_pointwise_dependence(
          previous_context_index, previous_point, shard, false /*intra space*/);
      if (pre.exists())
        pointwise_mapping_dependences.emplace_back(pre);
    }

    //--------------------------------------------------------------------------
    TraceLocalID PointFillOp::get_trace_local_id(void) const
    //--------------------------------------------------------------------------
    {
      return TraceLocalID(trace_local_id, index_point);
    }

    /////////////////////////////////////////////////////////////
    // Repl Fill Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplFillOp::ReplFillOp(void)
      : ReplCollectiveVersioning<CollectiveVersioning<FillOp> >()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplFillOp::~ReplFillOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplFillOp::initialize_replication(
        ReplicateContext* ctx, bool is_first)
    //--------------------------------------------------------------------------
    {
      is_first_local_shard = is_first;
    }

    //--------------------------------------------------------------------------
    void ReplFillOp::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveVersioning<CollectiveVersioning<FillOp> >::activate();
      collective_map_barrier = RtBarrier::NO_RT_BARRIER;
      is_first_local_shard = false;
    }

    //--------------------------------------------------------------------------
    void ReplFillOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      // Make sure we didn't leak our barrier
      legion_assert(!collective_map_barrier.exists());
      ReplCollectiveVersioning<CollectiveVersioning<FillOp> >::deactivate(
          false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplFillOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      collective_map_barrier = repl_ctx->get_next_collective_map_barriers();
      create_collective_rendezvous(0 /*requirement index*/);
      // Then do the base class analysis
      FillOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void ReplFillOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(collective_map_barrier.exists());
      // Signal that all of our mapping dependences are satisfied
      runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/);
      if (parent_req_index == TRACED_PARENT_INDEX)
        parent_req_index = parent_ctx->find_parent_region_index(
            this, requirement, 0 /*idx*/, false /*skip privileges*/,
            true /*force*/);
      std::set<RtEvent> preconditions;
      if (fill_view_ready.exists())
        preconditions.insert(fill_view_ready);
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
    RtEvent ReplFillOp::perform_collective_versioning_analysis(
        unsigned index, LogicalRegion handle, EqSetTracker* tracker,
        const FieldMask& mask, unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      return rendezvous_collective_versioning_analysis(
          index, handle, tracker, runtime->address_space, mask,
          parent_req_index);
    }

    //--------------------------------------------------------------------------
    bool ReplFillOp::perform_collective_analysis(
        CollectiveMapping*& mapping, bool& first_local)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      mapping = &(repl_ctx->shard_manager->get_collective_mapping());
      mapping->add_reference();
      first_local = is_first_local_shard;
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent ReplFillOp::finalize_complete_mapping(RtEvent pre)
    //--------------------------------------------------------------------------
    {
      if (collective_map_barrier.exists())
      {
        // Normal analysis path
        runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/, pre);
        const RtEvent result = collective_map_barrier;
        collective_map_barrier = RtBarrier::NO_RT_BARRIER;
        return result;
      }
      else  // Tracing path
        return pre;
    }

    //--------------------------------------------------------------------------
    void ReplFillOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(collective_map_barrier.exists());
      // Trigger both generations of the barrier and move on
      runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/);
      // Advance the first generation of the barrier for trigger_ready
      Runtime::advance_barrier(collective_map_barrier);
      // Trigger the second generation
      runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/);
      collective_map_barrier = RtBarrier::NO_RT_BARRIER;
      elide_collective_rendezvous();
      // Second generation triggered by callback to finalize_complete_mapping
      FillOp::trigger_replay();
    }

    //--------------------------------------------------------------------------
    void ReplFillOp::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      // Trigger the first generation of the collective_map_barrier
      runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/);
      // Advance the first generation of the barrier for trigger_ready
      Runtime::advance_barrier(collective_map_barrier);
      elide_collective_rendezvous();
      // Second generation triggered by callback to finalize_complete_mapping
      FillOp::predicate_false();
    }

    /////////////////////////////////////////////////////////////
    // Repl Index Fill Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplIndexFillOp::ReplIndexFillOp(void) : IndexFillOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplIndexFillOp::~ReplIndexFillOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplIndexFillOp::activate(void)
    //--------------------------------------------------------------------------
    {
      IndexFillOp::activate();
      sharding_functor = std::numeric_limits<ShardingID>::max();
      sharding_function = nullptr;
      shard_points = nullptr;
      mapper = nullptr;
      sharding_collective = nullptr;
    }

    //--------------------------------------------------------------------------
    void ReplIndexFillOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (sharding_collective != nullptr)
        delete sharding_collective;
      remove_launch_space_reference(shard_points);
      IndexFillOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplIndexFillOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      // Do the mapper call to get the sharding function to use
      if (mapper == nullptr)
        mapper =
            runtime->find_mapper(parent_ctx->get_executing_processor(), map_id);
      Mapper::SelectShardingFunctorInput* input = repl_ctx->shard_manager;
      Mapper::SelectShardingFunctorOutput output;
      mapper->invoke_fill_select_sharding_functor(this, *input, output);
      if (output.chosen_functor == std::numeric_limits<ShardingID>::max())
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Mapper " << mapper->get_mapper_name()
              << " failed to pick a valid sharding functor for "
              << "index fill in task " << parent_ctx->get_task_name()
              << " (UID " << parent_ctx->get_unique_id() << ").";
        error.raise();
      }
      this->sharding_functor = output.chosen_functor;
      sharding_function =
          repl_ctx->shard_manager->find_sharding_function(sharding_functor);
      if (runtime->safe_mapper)
      {
        legion_assert(sharding_collective != nullptr);
        sharding_collective->contribute(this->sharding_functor);
        if (sharding_collective->is_target() &&
            !sharding_collective->validate(this->sharding_functor))
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Mapper " << mapper->get_mapper_name()
                << " chose different sharding functions "
                << "for index fill in task " << parent_ctx->get_task_name()
                << " (UID " << parent_ctx->get_unique_id() << ").";
          error.raise();
        }
      }
      // Now we can do the normal prepipeline stage
      IndexFillOp::trigger_prepipeline_stage();
    }

    //--------------------------------------------------------------------------
    void ReplIndexFillOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_base_dependence_analysis();
      analyze_region_requirements(
          launch_space, sharding_function, sharding_space);
    }

    //--------------------------------------------------------------------------
    void ReplIndexFillOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      legion_assert(launch_space != nullptr);
      // Compute the local index space of points for this shard
      IndexSpace local_space;
      if (sharding_space.exists())
        local_space = sharding_function->find_shard_space(
            repl_ctx->owner_shard->shard_id, launch_space, sharding_space,
            get_provenance());
      else
        local_space = sharding_function->find_shard_space(
            repl_ctx->owner_shard->shard_id, launch_space, launch_space->handle,
            get_provenance());
      // If we're recording then record the local_space
      if (is_recording())
      {
        legion_assert((tpl != nullptr) && tpl->is_recording());
        tpl->record_local_space(trace_local_id, local_space);
      }
      // If it's empty we're done, otherwise we go back on the queue
      if (!local_space.exists())
      {
        // Still have to do this for legion spy
        LegionSpy::log_operation_events(
            unique_op_id, ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
        // We have no local points, so we can just trigger
        if (!map_applied_conditions.empty())
          complete_mapping(Runtime::merge_events(map_applied_conditions));
        else
          complete_mapping();
        if (future.impl != nullptr)
        {
          RtEvent future_ready = future.impl->subscribe();
          // Make sure both the future and the view are ready
          if (fill_view_ready.exists() && !fill_view_ready.has_triggered())
          {
            if (!future_ready.has_triggered())
              future_ready =
                  Runtime::merge_events(fill_view_ready, future_ready);
            else
              future_ready = fill_view_ready;
          }
          if (!future_ready.has_triggered())
            parent_ctx->add_to_trigger_execution_queue(this, future_ready);
          else
            trigger_execution();  // can do the completion now
        }
        else if (fill_view_ready.exists() && !fill_view_ready.has_triggered())
          parent_ctx->add_to_trigger_execution_queue(this, fill_view_ready);
        else
          trigger_execution();
      }
      else  // We have valid points, so it goes on the ready queue
      {
        shard_points = runtime->get_node(local_space);
        add_launch_space_reference(shard_points);
        IndexFillOp::trigger_ready();
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexFillOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(tpl != nullptr);
      const IndexSpace local_space = tpl->find_local_space(trace_local_id);
      // If it's empty we're done, otherwise we do the replay
      if (!local_space.exists())
      {
        LegionSpy::log_replay_operation(unique_op_id);
        LegionSpy::log_operation_events(
            unique_op_id, ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
        // We have no local points, so we can just trigger
        complete_mapping();
        complete_execution();
      }
      else
      {
        shard_points = runtime->get_node(local_space);
        add_launch_space_reference(shard_points);
        IndexFillOp::trigger_replay();
      }
    }

    //--------------------------------------------------------------------------
    bool ReplIndexFillOp::find_shard_participants(std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      legion_assert(sharding_function != nullptr);
      if (sharding_space.exists())
        return sharding_function->find_shard_participants(
            launch_space, sharding_space, shards);
      else
        return sharding_function->find_shard_participants(
            launch_space, launch_space->handle, shards);
    }

    /////////////////////////////////////////////////////////////
    // Remote Fill Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteFillOp::RemoteFillOp(Operation* ptr, AddressSpaceID src)
      : ExternalFill(), RemoteOp(ptr, src)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RemoteFillOp::~RemoteFillOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID RemoteFillOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t RemoteFillOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteFillOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteFillOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteFillOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == nullptr)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const std::string_view& RemoteFillOp::get_provenance_string(
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
    const char* RemoteFillOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[FILL_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind RemoteFillOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return FILL_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteFillOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_fill(rez, target);
    }

    //--------------------------------------------------------------------------
    void RemoteFillOp::unpack(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      unpack_external_fill(derez);
    }

  }  // namespace Internal
}  // namespace Legion
