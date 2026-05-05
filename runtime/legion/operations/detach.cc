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

#include "legion/operations/detach.h"
#include "legion/analysis/filter.h"
#include "legion/analysis/update.h"
#include "legion/contexts/replicate.h"
#include "legion/api/future_impl.h"
#include "legion/api/physical_region_impl.h"
#include "legion/managers/shard.h"
#include "legion/nodes/region.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Detach Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DetachOp::DetachOp(void) : Operation()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    DetachOp::~DetachOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    Future DetachOp::initialize_detach(
        InnerContext* ctx, PhysicalRegion region, const bool flsh,
        const bool unordered, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      flush = flsh;
      // Get a reference to the region to keep it alive
      this->region = region;
      requirement = region.impl->get_requirement();
      // Make sure that the privileges are read-write so that we wait for
      // all prior users of this particular region unless we're not flushing
      // in which case we can make the privileges write-discard
      requirement.privilege = flush ? LEGION_READ_WRITE : LEGION_WRITE_DISCARD;
      requirement.prop = LEGION_EXCLUSIVE;
      if (runtime->safe_model)
        verify_requirement(requirement);
      parent_req_index = ctx->find_parent_region_index(this, requirement);
      // Create the future result that we will complete when we're done
      result = Future(new FutureImpl(
          parent_ctx, true /*register*/,
          runtime->get_available_distributed_id(), get_provenance(), this));
      LegionSpy::log_detach_operation(
          parent_ctx->get_unique_id(), unique_op_id, unordered);
      return result;
    }

    //--------------------------------------------------------------------------
    void DetachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
      detach_event = ApEvent::NO_AP_EVENT;
      flush = true;
    }

    //--------------------------------------------------------------------------
    void DetachOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      region = PhysicalRegion();
      version_info.clear();
      map_applied_conditions.clear();
      result = Future();  // clear any references on the future
      Operation::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* DetachOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DETACH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind DetachOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return DETACH_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t DetachOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    void DetachOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      log_requirement();
    }

    //--------------------------------------------------------------------------
    void DetachOp::log_requirement(void)
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
    void DetachOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      analyze_region_requirements();
    }

    //--------------------------------------------------------------------------
    void DetachOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      perform_versioning_analysis(
          0 /*idx*/, requirement, version_info, preconditions,
          nullptr /*output region*/, is_point_detach());
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void DetachOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Logical dependence analysis should guarantee that we are valid
      // by the time we get here because the inline mapping/attach op
      // that made the physical region should have mapped before us
      legion_assert(region.impl->get_mapped_event().has_triggered());
      // Now we can get the reference we need for the detach operation
      InstanceSet references;
      region.impl->get_references(references);
      legion_assert(references.size() == 1);
      const InstanceRef& reference = references[0];
      // Add a valid reference to the instances to act as an acquire to keep
      // them valid through the end of mapping them, we'll release the valid
      // references when we are done mapping
      PhysicalManager* manager = reference.get_physical_manager();
      if (!manager->is_external_instance())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal " << *this << ". Detach was performed on an region "
              << "that had not previously been attached.";
        error.raise();
      }
      legion_assert(!manager->is_reduction_manager());
      manager->add_base_valid_ref(MAPPING_ACQUIRE_REF);
      const PhysicalTraceInfo trace_info(this, 0 /*idx*/);
      // If we need to flush then register this operation to bring the
      // data that it has up to date, use READ-ONLY privileges since we're
      // not going to invalidate the existing data. Don't register ourselves
      // either since we'll get all the preconditions for detaching it
      // as part of the detach_external call
      ApUserEvent detach_post = Runtime::create_ap_user_event(&trace_info);
      RtEvent filter_precondition;
      if (flush)
      {
        requirement.privilege = LEGION_READ_ONLY;
        std::vector<PhysicalManager*> dummy_sources;
        UpdateAnalysis* analysis = nullptr;
        filter_precondition = physical_perform_updates(
            requirement, version_info, 0 /*idx*/, ApEvent::NO_AP_EVENT,
            detach_post, references, dummy_sources, trace_info,
            map_applied_conditions, analysis, false /*check collective*/,
            false /*record valid*/, false /*check initialized*/);
        if (analysis->remove_reference())
          delete analysis;
        requirement.privilege = LEGION_READ_WRITE;
      }

      detach_event = detach_external(
          references, detach_post, trace_info, filter_precondition, flush);
      Runtime::trigger_event(
          detach_post, detach_event, trace_info, map_applied_conditions);
      record_completion_effect(detach_post);
      log_mapping_decision(0 /*idx*/, requirement, references);
      LegionSpy::log_operation_events(unique_op_id, detach_event, detach_post);
      if (!map_applied_conditions.empty())
        complete_mapping(finalize_complete_mapping(
            Runtime::merge_events(map_applied_conditions)));
      else
        complete_mapping(finalize_complete_mapping(RtEvent::NO_RT_EVENT));
      complete_execution();
    }

    //--------------------------------------------------------------------------
    unsigned DetachOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx == 0);
      legion_assert(parent_req_index != TRACED_PARENT_INDEX);
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void DetachOp::detach_external_instance(PhysicalManager* manager)
    //--------------------------------------------------------------------------
    {
      // It's only safe to actually perform the detach after the mapping
      // is performed to know that all the updates to the instance have
      // been mapped
      manager->detach_external_instance();
    }

    //--------------------------------------------------------------------------
    void DetachOp::trigger_complete(ApEvent effects)
    //--------------------------------------------------------------------------
    {
      // Can be nullptr if this is a PointDetachOp
      if (result.impl != nullptr)
        result.impl->set_result(effects);
      InstanceSet references;
      region.impl->get_references(references);
      legion_assert(references.size() == 1);
      const InstanceRef& reference = references[0];
      PhysicalManager* manager = reference.get_physical_manager();
      detach_external_instance(manager);
      // We can remove the acquire reference that we added after we're mapped
      if (manager->remove_base_valid_ref(MAPPING_ACQUIRE_REF))
        delete manager;
      complete_operation(effects);
    }

    //--------------------------------------------------------------------------
    void DetachOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      commit_operation(true /*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void DetachOp::select_sources(
        const unsigned index, PhysicalManager* target,
        const std::vector<InstanceView*>& sources,
        std::vector<unsigned>& ranking,
        std::map<unsigned, PhysicalManager*>& points)
    //--------------------------------------------------------------------------
    {
      legion_assert(index == 0);
      // TODO: invoke the mapper
      std::deque<MappingInstance> dummy_out;
      compute_ranking(nullptr, dummy_out, sources, ranking, points);
    }

    //--------------------------------------------------------------------------
    int DetachOp::add_copy_profiling_request(
        const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& reqeusts,
        bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
      return 0;
    }

    //--------------------------------------------------------------------------
    void DetachOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      const ContextCoordinate coordinate = get_task_tree_coordinate();
      rez.serialize(coordinate.index_point);
    }

    //--------------------------------------------------------------------------
    ApEvent DetachOp::detach_external(
        const InstanceSet& instances, const ApEvent termination_event,
        const PhysicalTraceInfo& trace_info, RtEvent filter_precondition,
        const bool second_analysis)
    //--------------------------------------------------------------------------
    {
      legion_assert(requirement.handle_type == LEGION_SINGULAR_PROJECTION);
      legion_assert(instances.size() == 1);
      RegionNode* region = runtime->get_node(requirement.region);
      FilterAnalysis* analysis = new FilterAnalysis(
          this, 0 /*index*/, region, trace_info, true /*remove restriction*/);
      analysis->add_reference();
      // If we have a filter precondition, then we know this is not the first
      // potential collective analysis to be used here
      const RtEvent views_ready = analysis->convert_views(
          requirement.region, instances, nullptr /*sources*/, nullptr /*usage*/,
          false /*rendezvous*/, second_analysis ? 1 : 0);
      // Don't start the analysis until the views are ready and the filter
      // precondition has been met
      const RtEvent traversal_precondition =
          Runtime::merge_events(views_ready, filter_precondition);
      const RtEvent traversal_done = analysis->perform_traversal(
          traversal_precondition, version_info, map_applied_conditions);
      // Send out any remote updates
      if (traversal_done.exists() || analysis->has_remote_sets())
        analysis->perform_remote(traversal_done, map_applied_conditions);
      ApEvent instances_ready;
      const RegionUsage usage(requirement);
      analysis->perform_registration(
          traversal_precondition, usage, map_applied_conditions,
          ApEvent::NO_AP_EVENT /*no precondition*/, termination_event,
          instances_ready);
      if (analysis->remove_reference())
        delete analysis;
      return instances_ready;
    }

    /////////////////////////////////////////////////////////////
    // Index Detach Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexDetachOp::IndexDetachOp(void)
      : PointwiseAnalyzable<CollectiveViewCreator<Operation> >()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    IndexDetachOp::~IndexDetachOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void IndexDetachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      PointwiseAnalyzable<CollectiveViewCreator<Operation> >::activate();
      launch_space = nullptr;
      points_completed.store(0);
      points_committed = 0;
      commit_request = false;
      flush = false;
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      resources = ExternalResources();
      // We can deactivate all of our point operations
      for (PointDetachOp* point : points) point->deactivate();
      points.clear();
      map_applied_conditions.clear();
      result = Future();
      PointwiseAnalyzable<CollectiveViewCreator<Operation> >::deactivate(
          false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* IndexDetachOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DETACH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind IndexDetachOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return DETACH_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t IndexDetachOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    Future IndexDetachOp::initialize_detach(
        InnerContext* ctx, LogicalRegion parent, RegionTreeNode* upper_bound,
        IndexSpaceNode* launch_bounds, ExternalResourcesImpl* external,
        const std::vector<FieldID>& privilege_fields,
        const std::vector<PhysicalRegion>& regions, bool flsh, bool unordered,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      // Construct the region requirement
      // We'll get the projection later after we know its been written
      // in the dependence analysis stage of the pipeline
      if (upper_bound->is_region())
        requirement = RegionRequirement(
            upper_bound->as_region_node()->handle, 0 /*fake*/,
            LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, parent);
      else
        requirement = RegionRequirement(
            upper_bound->as_partition_node()->handle, 0 /*fake*/,
            LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, parent);
      for (const FieldID& fid : privilege_fields) requirement.add_field(fid);
      if (runtime->safe_model)
        verify_requirement(requirement, 0 /*index*/, true /*allow projection*/);
      parent_req_index = ctx->find_parent_region_index(this, requirement);
      resources = ExternalResources(external);
      launch_space = launch_bounds;
      points.reserve(regions.size());
      flush = flsh;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        PointDetachOp* point = runtime->get_operation<PointDetachOp>();
        const DomainPoint index_point = Point<1>(idx);
        point->initialize_detach(this, ctx, regions[idx], index_point, flush);
        points.emplace_back(point);
      }
      // Create the future result that we will complete when we're done
      result = Future(new FutureImpl(
          parent_ctx, true /*register*/,
          runtime->get_available_distributed_id(), get_provenance(), this));
      LegionSpy::log_detach_operation(
          parent_ctx->get_unique_id(), unique_op_id, unordered);
      log_launch_space(launch_space->handle);
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // Promote a singular region requirement up to a projection
      if (requirement.handle_type == LEGION_SINGULAR_PROJECTION)
      {
        requirement.handle_type = LEGION_REGION_PROJECTION;
        requirement.projection = 0;
      }
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Get the projection ID which we know is valid on the external resources
      requirement.projection = resources.impl->get_projection();
      log_requirement();
      analyze_region_requirements(launch_space);
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      if (!pointwise_dependences.empty())
      {
        std::vector<LogicalRegion> regions(points.size());
        for (unsigned idx = 0; idx < points.size(); idx++)
          regions[idx] = points[idx]->requirement.region;
        legion_assert(pointwise_dependences.size() == 1);
        legion_assert(pointwise_dependences.begin()->first == 0);
        std::vector<std::vector<RtEvent> > preconditions(points.size());
        for (const PointwiseDependence& pit :
             pointwise_dependences.begin()->second)
        {
          std::map<LogicalRegion, std::vector<DomainPoint> > dependences;
          pit.find_dependences(requirement, regions, dependences);
          if (pit.sharding != nullptr)
          {
            const Domain launch_domain =
                pit.sharding_domain->get_tight_domain();
            for (unsigned idx = 0; idx < points.size(); idx++)
            {
              std::map<LogicalRegion, std::vector<DomainPoint> >::const_iterator
                  finder = dependences.find(regions[idx]);
              legion_assert(finder != dependences.end());
              for (const DomainPoint& point : finder->second)
              {
                ShardID shard = pit.sharding->shard(
                    point, launch_domain, parent_ctx->get_total_shards());
                RtEvent precondition = parent_ctx->find_pointwise_dependence(
                    pit.context_index, point, shard, false /*intra space*/);
                if (precondition.exists())
                  preconditions[idx].emplace_back(precondition);
              }
            }
          }
          else
          {
            for (unsigned idx = 0; idx < points.size(); idx++)
            {
              std::map<LogicalRegion, std::vector<DomainPoint> >::const_iterator
                  finder = dependences.find(regions[idx]);
              legion_assert(finder != dependences.end());
              for (const DomainPoint& point : finder->second)
              {
                RtEvent precondition = parent_ctx->find_pointwise_dependence(
                    pit.context_index, point, 0 /*shard*/,
                    false /*intra space*/);
                if (precondition.exists())
                  preconditions[idx].emplace_back(precondition);
              }
            }
          }
        }
        for (unsigned idx = 0; idx < points.size(); idx++)
        {
          map_applied_conditions.insert(points[idx]->get_mapped_event());
          points[idx]->enqueue_ready_operation(
              Runtime::merge_events(preconditions[idx]));
        }
      }
      else
      {
        for (unsigned idx = 0; idx < points.size(); idx++)
        {
          map_applied_conditions.insert(points[idx]->get_mapped_event());
          points[idx]->trigger_ready();
        }
      }
      // Record that we are mapped when all our points are mapped
      // and we are executed when all our points are executed
      complete_mapping(Runtime::merge_events(map_applied_conditions));
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::handle_point_complete(ApEvent effect)
    //--------------------------------------------------------------------------
    {
      if (effect.exists())
        record_completion_effect(effect);
      const unsigned received = ++points_completed;
      legion_assert(received <= points.size());
      if (received == points.size())
        complete_execution();
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::trigger_complete(ApEvent effects)
    //--------------------------------------------------------------------------
    {
      result.impl->set_result(effects);
      LegionSpy::log_operation_events(
          unique_op_id, effects, ApEvent::NO_AP_EVENT);
      complete_operation(effects);
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::trigger_commit(void)
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
    void IndexDetachOp::handle_point_commit(void)
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

    //--------------------------------------------------------------------------
    unsigned IndexDetachOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx == 0);
      legion_assert(parent_req_index != TRACED_PARENT_INDEX);
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void IndexDetachOp::log_requirement(void)
    //--------------------------------------------------------------------------
    {
      if (requirement.handle_type == LEGION_PARTITION_PROJECTION)
        LegionSpy::log_logical_requirement(
            unique_op_id, 0 /*index*/, false /*region*/,
            requirement.partition.index_partition.get_id(),
            requirement.partition.field_space.get_id(),
            requirement.partition.get_tree_id(), requirement.privilege,
            requirement.prop, requirement.redop,
            requirement.parent.index_space.get_id());
      else
        LegionSpy::log_logical_requirement(
            unique_op_id, 0 /*index*/, true /*region*/,
            requirement.region.index_space.get_id(),
            requirement.region.field_space.get_id(),
            requirement.region.get_tree_id(), requirement.privilege,
            requirement.prop, requirement.redop,
            requirement.parent.index_space.get_id());
      LegionSpy::log_requirement_projection(
          unique_op_id, 0 /*index*/, requirement.projection);
      LegionSpy::log_requirement_fields(
          unique_op_id, 0 /*index*/, requirement.privilege_fields);
    }

    //--------------------------------------------------------------------------
    size_t IndexDetachOp::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      return points.size();
    }

    //--------------------------------------------------------------------------
    RtEvent IndexDetachOp::find_pointwise_dependence(
        const DomainPoint& point, GenerationID needed_gen, bool intra_space,
        RtUserEvent to_trigger, std::optional<unsigned> output_index)
    //--------------------------------------------------------------------------
    {
      legion_assert(!intra_space);
      legion_assert(!output_index);
      AutoLock o_lock(op_lock, false /*exclusive*/);
      legion_assert(needed_gen <= gen);
      if ((needed_gen < gen) || mapped)
      {
        if (to_trigger.exists())
          Runtime::trigger_event(to_trigger);
        return RtEvent::NO_RT_EVENT;
      }
      legion_assert(!points.empty());
      for (PointDetachOp* it : points)
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

    /////////////////////////////////////////////////////////////
    // Point Detach Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointDetachOp::PointDetachOp(void) : DetachOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PointDetachOp::~PointDetachOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void PointDetachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      DetachOp::activate();
      owner = nullptr;
    }

    //--------------------------------------------------------------------------
    void PointDetachOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      DetachOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void PointDetachOp::initialize_detach(
        IndexDetachOp* own, InnerContext* ctx, const PhysicalRegion& region,
        const DomainPoint& point, bool flsh)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, own->get_provenance());
      index_point = point;
      owner = own;
      flush = flsh;
      context_index = own->get_context_index();
      exception_handler = own->get_exception_handler();
      // Get a reference to the region to keep it alive
      this->region = region;
      requirement = region.impl->get_requirement();
      // Make sure that the privileges are read-write so that we wait for
      // all prior users of this particular region
      requirement.privilege = LEGION_READ_WRITE;
      requirement.prop = LEGION_EXCLUSIVE;
      // No need for a future here
      LegionSpy::log_index_point(
          owner->get_unique_op_id(), unique_op_id, point);
      log_requirement();
    }

    //--------------------------------------------------------------------------
    void PointDetachOp::trigger_complete(ApEvent effect)
    //--------------------------------------------------------------------------
    {
      owner->handle_point_complete(effect);
      DetachOp::trigger_complete(ApEvent::NO_AP_EVENT);
    }

    //--------------------------------------------------------------------------
    void PointDetachOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      commit_operation(false /*deactivate*/);
      // Tell our owner that we are done, they will do the deactivate
      owner->handle_point_commit();
    }

    //--------------------------------------------------------------------------
    size_t PointDetachOp::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      return owner->get_collective_points();
    }

    //--------------------------------------------------------------------------
    bool PointDetachOp::find_shard_participants(std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      return owner->find_shard_participants(shards);
    }

    //--------------------------------------------------------------------------
    RtEvent PointDetachOp::convert_collective_views(
        unsigned requirement_index, unsigned analysis_index,
        LogicalRegion region, const InstanceSet& targets,
        InnerContext* physical_ctx, CollectiveMapping*& analysis_mapping,
        bool& first_local,
        op::vector<op::FieldMaskMap<InstanceView> >& target_views,
        std::map<InstanceView*, size_t>& collective_arrivals)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_collective_rendezvous(
          unique_op_id, requirement_index, analysis_index);
      return owner->convert_collective_views(
          requirement_index, analysis_index, region, targets, physical_ctx,
          analysis_mapping, first_local, target_views, collective_arrivals);
    }

    //--------------------------------------------------------------------------
    bool PointDetachOp::perform_collective_analysis(
        CollectiveMapping*& mapping, bool& first_local)
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent PointDetachOp::perform_collective_versioning_analysis(
        unsigned index, LogicalRegion handle, EqSetTracker* tracker,
        const FieldMask& mask, unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      return owner->rendezvous_collective_versioning_analysis(
          index, handle, tracker, runtime->address_space, mask,
          parent_req_index);
    }

    /////////////////////////////////////////////////////////////
    // Repl Detach Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplDetachOp::ReplDetachOp(void)
      : ReplCollectiveViewCreator<CollectiveViewCreator<DetachOp> >()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplDetachOp::~ReplDetachOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplDetachOp::initialize_replication(
        ReplicateContext* ctx, bool collective, bool first_local_shard)
    //--------------------------------------------------------------------------
    {
      collective_instances = collective;
      is_first_local_shard = first_local_shard;
    }

    //--------------------------------------------------------------------------
    void ReplDetachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveViewCreator<CollectiveViewCreator<DetachOp> >::activate();
      collective_map_barrier = RtBarrier::NO_RT_BARRIER;
      effects_barrier = ApBarrier::NO_AP_BARRIER;
      exchange_index = 0;
      collective_instances = false;
      is_first_local_shard = false;
    }

    //--------------------------------------------------------------------------
    void ReplDetachOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveViewCreator<CollectiveViewCreator<DetachOp> >::deactivate(
          false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplDetachOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      analyze_region_requirements();
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      legion_assert(!collective_map_barrier.exists());
      collective_map_barrier = repl_ctx->get_next_collective_map_barriers();
      effects_barrier = repl_ctx->get_next_detach_effects_barrier();
      if (collective_instances)
      {
        create_collective_rendezvous(requirement.parent.get_tree_id(), 0);
        // If we're flushing we need a second analysis rendezvous
        if (flush)
          create_collective_rendezvous(
              requirement.parent.get_tree_id(), 0 /*requirement index*/,
              1 /*analysis index*/);
      }
      else  // Only need the versioning rendezvous in this case
        ReplCollectiveVersioning<
            CollectiveViewCreator<DetachOp> >::create_collective_rendezvous(0);
    }

    //--------------------------------------------------------------------------
    void ReplDetachOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      legion_assert(collective_map_barrier.exists());
      // Signal that all our mapping dependences are met
      runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/);
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
    RtEvent ReplDetachOp::finalize_complete_mapping(RtEvent pre)
    //--------------------------------------------------------------------------
    {
      legion_assert(effects_barrier.exists());
      legion_assert(collective_map_barrier.exists());
      // Always arrive on the effects barrier with the detach event
      runtime->phase_barrier_arrive(effects_barrier, 1 /*count*/, detach_event);
      // Then update the detach event with the effects barrier
      detach_event = effects_barrier;
      runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/, pre);
      return collective_map_barrier;
    }

    //--------------------------------------------------------------------------
    bool ReplDetachOp::perform_collective_analysis(
        CollectiveMapping*& mapping, bool& first_local)
    //--------------------------------------------------------------------------
    {
      if (!collective_instances)
      {
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(parent_ctx);
        mapping = &repl_ctx->shard_manager->get_collective_mapping();
        mapping->add_reference();
        first_local = is_first_local_shard;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool ReplDetachOp::find_shard_participants(std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      // All shards are participating
      return true;
    }

    //--------------------------------------------------------------------------
    void ReplDetachOp::record_unordered_kind(
        std::map<std::pair<LogicalRegion, FieldID>, Operation*>& detachments)
    //--------------------------------------------------------------------------
    {
      legion_assert(!requirement.privilege_fields.empty());
      const std::pair<LogicalRegion, FieldID> key(
          requirement.region, *(requirement.privilege_fields.begin()));
      legion_assert(detachments.find(key) == detachments.end());
      detachments[key] = this;
    }

    //--------------------------------------------------------------------------
    void ReplDetachOp::detach_external_instance(PhysicalManager* manager)
    //--------------------------------------------------------------------------
    {
      if (collective_instances)
      {
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(parent_ctx);
        ShardManager* shard_manager = repl_ctx->shard_manager;
        // See if all local shards have the same manager or not
        if (is_first_local_shard)
        {
          shard_manager->exchange_shard_local_op_data(
              context_index, exchange_index++, manager);
          manager->detach_external_instance();
        }
        else
        {
          PhysicalManager* first_manager =
              shard_manager->find_shard_local_op_data<PhysicalManager*>(
                  context_index, exchange_index++);
          // If the managers are different then we do the detach as well
          if (manager != first_manager)
            manager->detach_external_instance();
        }
      }
      else if (manager->is_owner())
        manager->detach_external_instance();
    }

    //--------------------------------------------------------------------------
    RtEvent ReplDetachOp::perform_collective_versioning_analysis(
        unsigned index, LogicalRegion handle, EqSetTracker* tracker,
        const FieldMask& mask, unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      return rendezvous_collective_versioning_analysis(
          index, handle, tracker, runtime->address_space, mask,
          parent_req_index);
    }

    /////////////////////////////////////////////////////////////
    // Repl Index Detach Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplIndexDetachOp::ReplIndexDetachOp(void)
      : ReplCollectiveViewCreator<IndexDetachOp>()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplIndexDetachOp::~ReplIndexDetachOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplIndexDetachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveViewCreator<IndexDetachOp>::activate();
      sharding_function = nullptr;
      effects_barrier = ApBarrier::NO_AP_BARRIER;
      participants = nullptr;
    }

    //--------------------------------------------------------------------------
    void ReplIndexDetachOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (participants != nullptr)
        delete participants;
      ReplCollectiveViewCreator<IndexDetachOp>::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplIndexDetachOp::initialize_replication(ReplicateContext* ctx)
    //--------------------------------------------------------------------------
    {
      participants = new ShardParticipantsExchange(ctx, COLLECTIVE_LOC_103);
      participants->exchange(points.size() > 0);
    }

    //--------------------------------------------------------------------------
    void ReplIndexDetachOp::record_unordered_kind(
        std::map<std::pair<LogicalRegion, FieldID>, Operation*>&
            region_detachments,
        std::map<std::pair<LogicalPartition, FieldID>, Operation*>&
            part_detachments)
    //--------------------------------------------------------------------------
    {
      legion_assert(!requirement.privilege_fields.empty());
      if (requirement.handle_type == LEGION_PARTITION_PROJECTION)
      {
        const std::pair<LogicalPartition, FieldID> key(
            requirement.partition, *(requirement.privilege_fields.begin()));
        legion_assert(part_detachments.find(key) == part_detachments.end());
        part_detachments[key] = this;
      }
      else
      {
        const std::pair<LogicalRegion, FieldID> key(
            requirement.region, *(requirement.privilege_fields.begin()));
        legion_assert(region_detachments.find(key) == region_detachments.end());
        region_detachments[key] = this;
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexDetachOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(sharding_function == nullptr);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      sharding_function = repl_ctx->get_attach_detach_sharding_function();
      IndexDetachOp::trigger_prepipeline_stage();
    }

    //--------------------------------------------------------------------------
    void ReplIndexDetachOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(sharding_function != nullptr);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      // Get the projection ID which we know is valid on the external resources
      requirement.projection = resources.impl->get_projection();
      log_requirement();
      analyze_region_requirements(launch_space, sharding_function);
      create_collective_rendezvous(requirement.parent.get_tree_id(), 0);
      // If we're flushing we need a second analysis rendezvous
      if (flush)
        create_collective_rendezvous(requirement.parent.get_tree_id(), 0, 1);
      effects_barrier = repl_ctx->get_next_detach_effects_barrier();
    }

    //--------------------------------------------------------------------------
    void ReplIndexDetachOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      if (points.empty())
      {
        // Still need to make sure our collective is done
        if (!map_applied_conditions.empty())
          complete_mapping(Runtime::merge_events(map_applied_conditions));
        else
          complete_mapping();
        const RtEvent collective_done =
            participants->perform_collective_wait(false /*block*/);
        std::set<RtEvent> done_events;
        shard_off_collective_rendezvous(done_events);
        if (!done_events.empty())
        {
          if (collective_done.exists() && !collective_done.has_triggered())
            done_events.insert(collective_done);
          complete_execution(Runtime::merge_events(done_events));
        }
        else
          complete_execution(collective_done);
      }
      else
        IndexDetachOp::trigger_ready();
    }

    //--------------------------------------------------------------------------
    bool ReplIndexDetachOp::find_shard_participants(
        std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      legion_assert(participants != nullptr);
      return participants->find_shard_participants(shards);
    }

    /////////////////////////////////////////////////////////////
    // Remote Detach Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteDetachOp::RemoteDetachOp(Operation* ptr, AddressSpaceID src)
      : RemoteOp(ptr, src)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RemoteDetachOp::~RemoteDetachOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID RemoteDetachOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t RemoteDetachOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteDetachOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteDetachOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const char* RemoteDetachOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DETACH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind RemoteDetachOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return DETACH_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteDetachOp::select_sources(
        const unsigned index, PhysicalManager* target,
        const std::vector<InstanceView*>& sources,
        std::vector<unsigned>& ranking,
        std::map<unsigned, PhysicalManager*>& points)
    //--------------------------------------------------------------------------
    {
      legion_assert(index == 0);
      // TODO: invoke the mapper
      std::deque<MappingInstance> dummy_out;
      compute_ranking(nullptr, dummy_out, sources, ranking, points);
    }

    //--------------------------------------------------------------------------
    void RemoteDetachOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      rez.serialize(index_point);
    }

    //--------------------------------------------------------------------------
    void RemoteDetachOp::unpack(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(index_point);
    }

  }  // namespace Internal
}  // namespace Legion
