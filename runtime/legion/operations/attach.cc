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

#include "legion/operations/attach.h"
#include "legion/analysis/overwrite.h"
#include "legion/contexts/replicate.h"
#include "legion/api/physical_region_impl.h"
#include "legion/managers/shard.h"
#include "legion/nodes/region.h"
#include "legion/tools/spy.h"
#ifdef LEGION_USE_HDF5
#include "realm/hdf5/hdf5_access.h"
#endif

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Attach Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AttachOp::AttachOp(void) : Operation()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    AttachOp::~AttachOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PhysicalRegion AttachOp::initialize(
        InnerContext* ctx, const AttachLauncher& launcher,
        Provenance* provenance, unsigned requirement_index)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      resource = launcher.resource;
      layout_constraint_set = launcher.constraints;
      restricted = launcher.restricted;
      requirement = RegionRequirement(
          launcher.handle, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE,
          launcher.parent);
      requirement.privilege_fields = launcher.privilege_fields;
      if (runtime->safe_model)
        verify_requirement(requirement);
      parent_req_index = ctx->find_parent_region_index(this, requirement);
      if (launcher.resource == LEGION_EXTERNAL_HDF5_FILE)
      {
#ifdef LEGION_USE_HDF5
        const FieldConstraint& field_constraint =
            layout_constraint_set.field_constraint;
        hdf5_field_files.reserve(field_constraint.field_set.size());
        for (const FieldID& fid : field_constraint.field_set)
        {
          std::map<FieldID, const char*>::const_iterator finder =
              launcher.field_files.find(fid);
          if (finder == launcher.field_files.end())
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "Unable to find field file name for field " << fid
                  << " of HDF5 file " << *this
                  << ". Every field in an HDF5 attach must have a "
                  << "corresponding field file specified field_files.";
            error.raise();
          }
          hdf5_field_files.emplace_back(std::string(finder->second));
        }
#else
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid attach HDF5 file " << *this
              << ". Legion must be built with HDF5 support to attach regions "
              << "to HDF5 files";
        error.raise();
#endif
      }
      if (launcher.external_resource != nullptr)
      {
        external_resource = launcher.external_resource->clone();
      }
      else
      {
        // These are all the deprecated pathways, turn off deprecated warnings
        LEGION_DISABLE_DEPRECATED_WARNINGS
        switch (launcher.resource)
        {
          case LEGION_EXTERNAL_POSIX_FILE:
            {
              external_resource = new Realm::ExternalFileResource(
                  std::string(launcher.file_name), launcher.mode);
              break;
            }
          case LEGION_EXTERNAL_HDF5_FILE:
            {
#ifdef LEGION_USE_HDF5
              external_resource = new Realm::ExternalHDF5Resource(
                  launcher.file_name, launcher.mode);
#endif
              break;
            }
          case LEGION_EXTERNAL_INSTANCE:
            {
              external_resource = new Realm::ExternalMemoryResource(
                  layout_constraint_set.pointer_constraint.ptr,
                  launcher.footprint, false /*read only*/);
              const Memory memory = external_resource->suggested_memory();
              const PointerConstraint& pointer =
                  layout_constraint_set.pointer_constraint;
              if ((memory != pointer.memory) && pointer.memory.exists())
              {
                Warning warning;
                warning
                    << pointer.memory.kind() << " memory " << pointer.memory
                    << " in pointer constraint for " << *this
                    << " differs from the Realm-suggested " << memory.kind()
                    << " memory " << memory << " for the "
                    << "external instance. Legion is going to use the more "
                       "precise "
                    << "Realm-specified memory. Please make sure that you do "
                       "not "
                    << "have any code in your application or your mapper that "
                    << "relies on the instance being in the originally "
                       "specified "
                    << "memory. To silence this warning you can pass in a "
                       "NO_MEMORY "
                    << "to the pointer constraint.";
                warning.raise();
              }
              if (!layout_constraint_set.pointer_constraint.is_valid)
              {
                Error error(LEGION_INTERFACE_EXCEPTION);
                error
                    << "External array " << *this
                    << " issued with no pointer constraint. All external array "
                    << "attach operations must have a pointer constraint "
                    << "specified in the launcher.";
                error.raise();
              }
              break;
            }
          default:
            std::abort();  // should never get here
        }
        LEGION_REENABLE_DEPRECATED_WARNINGS
      }
      layout_constraint_set.specialized_constraint =
          SpecializedConstraint(LEGION_AFFINE_SPECIALIZE);
      layout_constraint_set.memory_constraint =
          MemoryConstraint(external_resource->suggested_memory().kind());
      // Pretend like the privileges for the region requirement are read-write
      // for cases where uses actually want to map it
      requirement.privilege = LEGION_READ_WRITE;
      region = PhysicalRegion(new PhysicalRegionImpl(
          requirement, get_mapped_event(), get_completion_event(),
          ApUserEvent::NO_AP_USER_EVENT, false /*mapped*/, ctx, 0 /*map id*/,
          0 /*tag*/, false /*leaf*/, false /*virtual mapped*/,
          launcher.collective, ctx->get_next_blocking_index(),
          requirement_index));
      // Restore privileges back to write-discard
      requirement.privilege = LEGION_WRITE_DISCARD;
      LegionSpy::log_attach_operation(
          parent_ctx->get_unique_id(), unique_op_id, restricted);
      return region;
    }

    //--------------------------------------------------------------------------
    void AttachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
      termination_event = ApEvent::NO_AP_EVENT;
      external_resource = nullptr;
      restricted = true;
    }

    //--------------------------------------------------------------------------
    void AttachOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      region = PhysicalRegion();
      version_info.clear();
      map_applied_conditions.clear();
      external_instances.clear();
      hdf5_field_files.clear();
      layout_constraint_set = LayoutConstraintSet();
      if (external_resource != nullptr)
        delete external_resource;
      Operation::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* AttachOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[ATTACH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind AttachOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return ATTACH_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t AttachOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    void AttachOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      create_external_instance();
      log_requirement();
    }

    //--------------------------------------------------------------------------
    void AttachOp::create_external_instance(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(requirement.handle_type == LEGION_SINGULAR_PROJECTION);
      RegionNode* attach_node = runtime->get_node(requirement.region);
      external_instances.resize(1);
      external_instances[0] =
          attach_node->column_source->create_external_instance(
              requirement.privilege_fields,
              layout_constraint_set.field_constraint.field_set, attach_node,
              this);
    }

    //--------------------------------------------------------------------------
    void AttachOp::log_requirement(void)
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
    void AttachOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      analyze_region_requirements();
    }

    //--------------------------------------------------------------------------
    void AttachOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      perform_versioning_analysis(
          0 /*idx*/, requirement, version_info, preconditions,
          nullptr /*output region*/, is_point_attach());
      // Register the instance with the memory manager and make sure it is
      // done before we perform our mapping
      legion_assert(!external_instances.empty());
      legion_assert(external_instances[0].has_ref());
      PhysicalManager* manager = external_instances[0].get_physical_manager();
      const RtEvent attached = manager->attach_external_instance();
      if (attached.exists())
        preconditions.insert(attached);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void AttachOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      const PhysicalTraceInfo trace_info(this, 0 /*idx*/);
      ApUserEvent attach_post = Runtime::create_ap_user_event(&trace_info);
      ApEvent attach_event = attach_external(attach_post, trace_info);
      log_mapping_decision(0 /*idx*/, requirement, external_instances);
      Runtime::trigger_event(
          attach_post, attach_event, trace_info, map_applied_conditions);
      record_completion_effect(attach_post);
      LegionSpy::log_operation_events(unique_op_id, attach_event, attach_post);
      // This operation is ready once the instance is attached
      region.impl->set_reference(external_instances[0]);
      // Once we have created the instance, then we are done
      if (!map_applied_conditions.empty())
        complete_mapping(finalize_complete_mapping(
            Runtime::merge_events(map_applied_conditions)));
      else
        complete_mapping(finalize_complete_mapping(RtEvent::NO_RT_EVENT));
      complete_execution();
    }

    //--------------------------------------------------------------------------
    unsigned AttachOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx == 0);
      legion_assert(parent_req_index != TRACED_PARENT_INDEX);
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void AttachOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      commit_operation(true /*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void AttachOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      const ContextCoordinate coordinate = get_task_tree_coordinate();
      rez.serialize(coordinate.index_point);
    }

    //--------------------------------------------------------------------------
    PhysicalManager* AttachOp::create_manager(
        RegionNode* node, const std::vector<FieldID>& field_set,
        const std::vector<size_t>& sizes,
        const std::vector<unsigned>& mask_index_map,
        const std::vector<CustomSerdezID>& serdez,
        const FieldMask& external_mask)
    //--------------------------------------------------------------------------
    {
      size_t footprint = 0;
      LgEvent unique_event;
      PhysicalInstance result = PhysicalInstance::NO_INST;
      const ApEvent ready_event = create_external(
          node, field_set, sizes, result, unique_event, footprint);
      // Check to see if this instance is local or whether we need
      // to send this request to a remote node to make it
      // Only external instances can be non-local, file instances
      // are always "local" to the node that they are on
      if ((resource == LEGION_EXTERNAL_INSTANCE) &&
          (result.address_space() != runtime->address_space))
      {
        ExternalCreateRequest rez;
        std::atomic<DistributedID> remote_did(0);
        const RtUserEvent wait_for = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(node->column_source->handle);
          rez.serialize(result);
          rez.serialize(ready_event);
          rez.serialize(unique_event);
          rez.serialize(footprint);
          layout_constraint_set.serialize(rez);
          rez.serialize(external_mask);
          rez.serialize<size_t>(field_set.size());
          for (unsigned idx = 0; idx < field_set.size(); idx++)
          {
            rez.serialize(field_set[idx]);
            rez.serialize(sizes[idx]);
            rez.serialize(mask_index_map[idx]);
            rez.serialize(serdez[idx]);
          }
          rez.serialize(node->row_source->handle);
          rez.serialize<size_t>(0);  // no collective mapping
          rez.serialize(&remote_did);
          rez.serialize(wait_for);
        }
        rez.dispatch(result.address_space());
        // Wait for the response to come back
        wait_for.wait();
        // Now we can request the physical manager
        RtEvent wait_on;
        PhysicalManager* result = runtime->find_or_request_instance_manager(
            remote_did.load(), wait_on);
        if (wait_on.exists())
          wait_on.wait();
        return result;
      }
      else  // Local so we can just do this call here
        return node->column_source->create_external_manager(
            result, ready_event, footprint, layout_constraint_set, field_set,
            sizes, external_mask, mask_index_map, unique_event, node, serdez,
            runtime->get_available_distributed_id());
    }

    //--------------------------------------------------------------------------
    ApEvent AttachOp::create_external(
        RegionNode* node, const std::vector<FieldID>& field_set,
        const std::vector<size_t>& sizes, PhysicalInstance& result,
        LgEvent& unique_event, size_t& footprint)
    //--------------------------------------------------------------------------
    {
      Realm::ProfilingRequestSet requests;
      if ((runtime->profiler != NULL) || (spy_logging_level > NO_SPY_LOGGING))
      {
        Runtime::rename_event(unique_event);
        if (runtime->profiler != NULL)
          runtime->profiler->add_inst_request(requests, this, unique_event);
      }
      // If we're doing an HDF5 instance creation we have to make a special
      // instance layout using HDF5 pieces. It's a bit unforuntate that we
      // have to have this special path but it is what it is
      Realm::InstanceLayoutGeneric* ilg =
          hdf5_field_files.empty() ?
              // Normal path
              // Make the base alignment be one since we want to establish the
              // minimal alignment that we require for this layout to see if it
              // can be supported by the particular external resource
              node->row_source->create_layout(
                  layout_constraint_set, field_set, sizes, false /*compact*/,
                  nullptr, nullptr, nullptr, 1 /*base alignment*/) :
              // Special path for HDF5
              node->row_source->create_hdf5_layout(
                  field_set, sizes, hdf5_field_files,
                  layout_constraint_set.ordering_constraint);
      if (!external_resource->satisfies(*ilg))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error
            << "External resources provided to attach operation " << *this
            << " are insufficient to support the specified layout constraint.";
        error.raise();
      }
      footprint = ilg->bytes_used;
      const ApEvent ready_event(PhysicalInstance::create_external_instance(
          result, external_resource->suggested_memory(), *ilg,
          *external_resource, requests));
      delete ilg;
      if (implicit_profiler != NULL)
      {
        implicit_profiler->register_physical_instance_region(
            unique_event, requirement.region);
        implicit_profiler->register_physical_instance_layout(
            unique_event, requirement.region.field_space,
            layout_constraint_set);
        if (ready_event.exists())
          implicit_profiler->record_instance_ready(ready_event, unique_event);
      }
      return ready_event;
    }

    //--------------------------------------------------------------------------
    ApEvent AttachOp::attach_external(
        const ApEvent termination_event, const PhysicalTraceInfo& trace_info)
    //--------------------------------------------------------------------------
    {
      legion_assert(requirement.handle_type == LEGION_SINGULAR_PROJECTION);
      IndexSpaceNode* expr_node =
          runtime->get_node(requirement.region.get_index_space());
      OverwriteAnalysis* analysis = new OverwriteAnalysis(
          this, 0 /*index*/, RegionUsage(requirement), expr_node, trace_info,
          ApEvent::NO_AP_EVENT, restricted);
      analysis->add_reference();
      const RtEvent views_ready =
          analysis->convert_views(requirement.region, external_instances);
      const RtEvent traversal_done = analysis->perform_traversal(
          views_ready, version_info, map_applied_conditions);
      // Send out any remote updates
      if (traversal_done.exists() || analysis->has_remote_sets())
        analysis->perform_remote(traversal_done, map_applied_conditions);
      // We can perform the registration in parallel with everything else
      ApEvent instances_ready;
      const RegionUsage usage(requirement);
      RtEvent registration_done = analysis->perform_registration(
          views_ready, usage, map_applied_conditions, ApEvent::NO_AP_EVENT,
          termination_event, instances_ready);
      if (registration_done.exists())
        map_applied_conditions.insert(registration_done);
      if (analysis->remove_reference())
        delete analysis;
      return instances_ready;
    }

    /////////////////////////////////////////////////////////////
    // Index Attach Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexAttachOp::IndexAttachOp(void)
      : PointwiseAnalyzable<CollectiveViewCreator<Operation> >()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    IndexAttachOp::~IndexAttachOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void IndexAttachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      PointwiseAnalyzable<CollectiveViewCreator<Operation> >::activate();
      launch_space = nullptr;
      points_completed.store(0);
      points_committed = 0;
      commit_request = false;
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      resources = ExternalResources();
      // We can deactivate all of our point operations
      for (PointAttachOp* point : points) point->deactivate();
      points.clear();
      map_applied_conditions.clear();
      commit_preconditions.clear();
      PointwiseAnalyzable<CollectiveViewCreator<Operation> >::deactivate(
          false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* IndexAttachOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[ATTACH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind IndexAttachOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return ATTACH_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t IndexAttachOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    ExternalResources IndexAttachOp::initialize(
        InnerContext* ctx, RegionTreeNode* upper_bound,
        IndexSpaceNode* launch_bounds, const IndexAttachLauncher& launcher,
        const std::vector<unsigned>& indexes, Provenance* provenance,
        const bool replicated, unsigned requirement_offset)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      // Construct the region requirement
      // Use a fake projection ID for now, we'll fill it in later during the
      // prepipeline stage before the logical dependence analysis
      // so that our computation of it is off the critical path
      if (upper_bound->is_region())
        requirement = RegionRequirement(
            upper_bound->as_region_node()->handle, 0 /*fake*/,
            LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, launcher.parent);
      else
        requirement = RegionRequirement(
            upper_bound->as_partition_node()->handle, 0 /*fake*/,
            LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, launcher.parent);
      requirement.privilege_fields = launcher.privilege_fields;
      if (runtime->safe_model)
        verify_requirement(requirement, 0 /*index*/, true /*allow projectino*/);
      parent_req_index = ctx->find_parent_region_index(this, requirement);
      launch_space = launch_bounds;
      // Create the result and the point attach operations
      ExternalResourcesImpl* result = new ExternalResourcesImpl(
          ctx, indexes.size(), upper_bound, launch_bounds, launcher.parent,
          requirement.privilege_fields);
      points.resize(indexes.size());
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        points[idx] = runtime->get_operation<PointAttachOp>();
        const DomainPoint index_point = Point<1>(indexes[idx]);
        PhysicalRegionImpl* region = points[idx]->initialize(
            this, ctx, launcher, index_point, indexes[idx], requirement_offset);
        result->set_region(idx, region);
      }
      LegionSpy::log_attach_operation(
          parent_ctx->get_unique_id(), unique_op_id, false /*restricted*/);
      if (launch_space != nullptr)
        log_launch_space(launch_space->handle);
      resources = ExternalResources(result);
      return resources;
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // Promote a singular region requirement up to a projection
      if (requirement.handle_type == LEGION_SINGULAR_PROJECTION)
      {
        requirement.handle_type = LEGION_REGION_PROJECTION;
        requirement.projection = 0;
      }
      // Have each of the point tasks create their external instances
      for (unsigned idx = 0; idx < points.size(); idx++)
        points[idx]->create_external_instance();
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Compute the projection function id for this requirement
      std::vector<IndexSpace> spaces(points.size());
      for (unsigned idx = 0; idx < points.size(); idx++)
        spaces[idx] = points[idx]->get_requirement().region.get_index_space();
      if (requirement.handle_type == LEGION_PARTITION_PROJECTION)
        requirement.projection = parent_ctx->compute_index_attach_projection(
            runtime->get_node(requirement.partition.index_partition), this,
            0 /*start*/, spaces.size(), spaces);
      else
        requirement.projection = parent_ctx->compute_index_attach_projection(
            runtime->get_node(requirement.region.index_space), this,
            0 /*start*/, spaces.size(), spaces);
      // Save this for later when we go to detach it
      resources.impl->set_projection(requirement.projection);
      log_requirement();
      analyze_region_requirements(launch_space);
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->safe_model)
        start_check_point_requirements();
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
    void IndexAttachOp::handle_point_complete(ApEvent effect)
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
    void IndexAttachOp::trigger_complete(ApEvent effects)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_operation_events(
          unique_op_id, ApEvent::NO_AP_EVENT, effects);
      complete_operation(effects);
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::trigger_commit(void)
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
      {
        if (!commit_preconditions.empty())
          commit_operation(
              true /*deactivate*/, Runtime::merge_events(commit_preconditions));
        else
          commit_operation(true /*deactivate*/);
      }
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::handle_point_commit(void)
    //--------------------------------------------------------------------------
    {
      bool commit_now = false;
      {
        AutoLock o_lock(op_lock);
        points_committed++;
        commit_now = commit_request && (points.size() == points_committed);
      }
      if (commit_now)
      {
        if (!commit_preconditions.empty())
          commit_operation(
              true /*deactivate*/, Runtime::merge_events(commit_preconditions));
        else
          commit_operation(true /*deactivate*/);
      }
    }

    //--------------------------------------------------------------------------
    unsigned IndexAttachOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx == 0);
      legion_assert(parent_req_index != TRACED_PARENT_INDEX);
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::start_check_point_requirements(void)
    //--------------------------------------------------------------------------
    {
      // Use a full vector here even though we have just one region requirement
      // because we need the type of `finish_check_point_requirements' to be
      // the same across all kinds of operations
      std::map<unsigned, std::vector<std::pair<DomainPoint, Domain> > >
          point_domains;
      std::vector<std::pair<DomainPoint, Domain> >& domains = point_domains[0];
      domains.reserve(points.size());
      for (PointAttachOp* point : points)
      {
        const RegionRequirement& req = point->get_requirement(0 /*index*/);
        Domain point_domain;
        runtime->find_domain(req.region.get_index_space(), point_domain);
        domains.emplace_back(
            std::make_pair(point->get_index_point(), point_domain));
      }
      finish_check_point_requirements(point_domains);
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::finish_check_point_requirements(
        std::map<unsigned, std::vector<std::pair<DomainPoint, Domain> > >&
            point_domains)
    //--------------------------------------------------------------------------
    {
      legion_assert(point_domains.size() == 1);
      std::vector<std::pair<DomainPoint, Domain> >& domains = point_domains[0];
      for (PointAttachOp* point : points)
      {
        const RegionRequirement& req = point->get_requirement(0 /*index*/);
        IndexSpaceNode* node = runtime->get_node(req.region.get_index_space());
        DomainPoint interfering;
        if (node->has_interfering_point(
                domains, interfering, point->get_index_point()))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Index " << *this << " has interfering region requirements "
                << "between point " << point->get_index_point() << " and point "
                << interfering << ". All regions specified for an index attach "
                << "operation must have non-interfering regions.";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexAttachOp::log_requirement(void)
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
    size_t IndexAttachOp::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      return points.size();
    }

    //--------------------------------------------------------------------------
    RtEvent IndexAttachOp::find_pointwise_dependence(
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
      for (PointAttachOp* it : points)
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
    // Point Attach Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointAttachOp::PointAttachOp(void) : AttachOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PointAttachOp::~PointAttachOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void PointAttachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      AttachOp::activate();
      owner = nullptr;
    }

    //--------------------------------------------------------------------------
    void PointAttachOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      AttachOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    PhysicalRegionImpl* PointAttachOp::initialize(
        IndexAttachOp* own, InnerContext* ctx,
        const IndexAttachLauncher& launcher, const DomainPoint& point,
        unsigned index, unsigned requirement_offset)
    //--------------------------------------------------------------------------
    {
      legion_assert(index < launcher.handles.size());
      initialize_operation(ctx, own->get_provenance());
      owner = own;
      index_point = point;
      context_index = own->get_context_index();
      exception_handler = own->get_exception_handler();
      layout_constraint_set = launcher.constraints;
      restricted = launcher.restricted;
      requirement = RegionRequirement(
          launcher.handles[index], LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE,
          launcher.parent);
      requirement.privilege_fields = launcher.privilege_fields;
      resource = launcher.resource;

#ifdef LEGION_USE_HDF5
      if (launcher.resource == LEGION_EXTERNAL_HDF5_FILE)
      {
        const FieldConstraint& field_constraint =
            layout_constraint_set.field_constraint;
        hdf5_field_files.reserve(field_constraint.field_set.size());
        for (const FieldID& fid : field_constraint.field_set)
        {
          std::map<FieldID, std::vector<const char*> >::const_iterator finder =
              launcher.field_files.find(fid);
          if ((finder == launcher.field_files.end()) ||
              (index >= finder->second.size()))
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error
                << "Unable to find field file name for field " << fid << " of "
                << "HDF5 file " << *this
                << ". Every field in an HDF5 attach must have a corresponding "
                << "field file specified field_files.";
            error.raise();
          }
          hdf5_field_files.emplace_back(std::string(finder->second[index]));
        }
      }
#endif
      if (!launcher.external_resources.empty())
      {
        if (index >= launcher.external_resources.size())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Insufficient 'external_resource' provided by index "
                << *this << " launch. Launcher has " << launcher.handles.size()
                << " logical regions but only "
                << launcher.external_resources.size() << " external resources.";
          error.raise();
        }
        external_resource = launcher.external_resources[index]->clone();
      }
      else
      {
        // These are all the deprecated pathways, turn off deprecated warnings
        LEGION_DISABLE_DEPRECATED_WARNINGS
        switch (launcher.resource)
        {
          case LEGION_EXTERNAL_POSIX_FILE:
            {
              if (index >= launcher.file_names.size())
              {
                Error error(LEGION_INTERFACE_EXCEPTION);
                error << "Insufficient 'file_names' provided by index " << *this
                      << ". Launcher has " << launcher.handles.size()
                      << " logical regions but only "
                      << launcher.file_names.size() << " POSIX file names.";
                error.raise();
              }
              external_resource = new Realm::ExternalFileResource(
                  std::string(launcher.file_names[index]), launcher.mode);
              break;
            }
          case LEGION_EXTERNAL_HDF5_FILE:
            {
              if (index >= launcher.file_names.size())
              {
                Error error(LEGION_INTERFACE_EXCEPTION);
                error << "Insufficient 'file_names' provided by index " << *this
                      << ". Launcher has " << launcher.handles.size()
                      << " logical regions but only "
                      << launcher.file_names.size() << " HDF5 file names.";
                error.raise();
              }
#ifndef LEGION_USE_HDF5
              Error error(LEGION_INTERFACE_EXCEPTION);
              error << "Invalid HDF5 file " << *this
                    << ". Legion must be built with HDF5 support to attach "
                       "regions "
                    << "to HDF5 files";
              error.raise();
#else
              external_resource = new Realm::ExternalHDF5Resource(
                  launcher.file_names[index], launcher.mode);
#endif
              break;
            }
          case LEGION_EXTERNAL_INSTANCE:
            {
              if (index >= launcher.pointers.size())
              {
                Error error(LEGION_INTERFACE_EXCEPTION);
                error << "Insufficient 'pointers' provided by index " << *this
                      << ". Launcher has " << launcher.handles.size()
                      << " logical regions but only "
                      << launcher.pointers.size() << " pointers names.";
                error.raise();
              }
              const PointerConstraint& pointer = launcher.pointers[index];
              external_resource = new Realm::ExternalMemoryResource(
                  pointer.ptr, launcher.footprint[index], false /*read only*/);
              const Memory memory = external_resource->suggested_memory();
              if ((memory != pointer.memory) && pointer.memory.exists())
              {
                Warning warning;
                warning
                    << "WARNING: " << pointer.memory.kind() << " memory "
                    << pointer.memory << " in pointer constraint for " << *this
                    << "differs from the Realm-suggested " << memory.kind()
                    << " memory " << memory << " for the external instance. "
                    << "Legion is going to use the more precise "
                       "Realm-specified "
                    << "memory. Please make sure that you do not have any code "
                    << "in your application or your mapper that relies on the "
                    << "instance being in the originally specified memory. To "
                    << "silence this warning you can pass in a NO_MEMORY to "
                    << "the pointer constraint.";
                warning.raise();
              }
              break;
            }
          default:
            std::abort();  // should never get here
        }
        LEGION_REENABLE_DEPRECATED_WARNINGS
      }
      layout_constraint_set.specialized_constraint =
          SpecializedConstraint(LEGION_AFFINE_SPECIALIZE);
      layout_constraint_set.memory_constraint =
          MemoryConstraint(external_resource->suggested_memory().kind());
      // Pretend like the privileges for the region requirement are read-write
      // for cases where uses actually want to map it
      requirement.privilege = LEGION_READ_WRITE;
      region = PhysicalRegion(new PhysicalRegionImpl(
          requirement, get_mapped_event(), get_completion_event(),
          ApUserEvent::NO_AP_USER_EVENT, false /*mapped*/, ctx, 0 /*map id*/,
          0 /*tag*/, false /*leaf*/, false /*virtual mapped*/,
          false /*collective*/, ctx->get_next_blocking_index(),
          requirement_offset + index));
      // Restore privileges back to write-discard
      requirement.privilege = LEGION_WRITE_DISCARD;
      LegionSpy::log_index_point(
          owner->get_unique_op_id(), unique_op_id, point);
      log_requirement();
      return region.impl;
    }

    //--------------------------------------------------------------------------
    void PointAttachOp::trigger_complete(ApEvent effects)
    //--------------------------------------------------------------------------
    {
      owner->handle_point_complete(effects);
      complete_operation();
    }

    //--------------------------------------------------------------------------
    void PointAttachOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      commit_operation(false /*deactivate*/);
      // Tell our owner that we are done, they will do the deactivate
      owner->handle_point_commit();
    }

    //--------------------------------------------------------------------------
    size_t PointAttachOp::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      return owner->get_collective_points();
    }

    //--------------------------------------------------------------------------
    bool PointAttachOp::find_shard_participants(std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      return owner->find_shard_participants(shards);
    }

    //--------------------------------------------------------------------------
    RtEvent PointAttachOp::convert_collective_views(
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
    bool PointAttachOp::perform_collective_analysis(
        CollectiveMapping*& mapping, bool& first_local)
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent PointAttachOp::perform_collective_versioning_analysis(
        unsigned index, LogicalRegion handle, EqSetTracker* tracker,
        const FieldMask& mask, unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      return owner->rendezvous_collective_versioning_analysis(
          index, handle, tracker, runtime->address_space, mask,
          parent_req_index);
    }

    /////////////////////////////////////////////////////////////
    // Index Attach Launch Space
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexAttachLaunchSpace::IndexAttachLaunchSpace(
        ReplicateContext* ctx, CollectiveIndexLocation loc)
      : AllGatherCollective<false>(loc, ctx), nonzeros(0)
    //--------------------------------------------------------------------------
    {
      sizes.resize(manager->total_shards, 0);
    }

    //--------------------------------------------------------------------------
    IndexAttachLaunchSpace::~IndexAttachLaunchSpace(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void IndexAttachLaunchSpace::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize(nonzeros);
      for (unsigned idx = 0; idx < sizes.size(); idx++)
      {
        size_t size = sizes[idx];
        if (size == 0)
          continue;
        rez.serialize(idx);
        rez.serialize(size);
      }
    }

    //--------------------------------------------------------------------------
    void IndexAttachLaunchSpace::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      unsigned num_nonzeros;
      derez.deserialize(num_nonzeros);
      for (unsigned idx = 0; idx < num_nonzeros; idx++)
      {
        unsigned index;
        derez.deserialize(index);
        if (sizes[index] == 0)
          nonzeros++;
        derez.deserialize(sizes[index]);
      }
    }

    //--------------------------------------------------------------------------
    void IndexAttachLaunchSpace::exchange_counts(size_t count)
    //--------------------------------------------------------------------------
    {
      if (count > 0)
      {
        legion_assert(local_shard < sizes.size());
        sizes[local_shard] = count;
        nonzeros++;
      }
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* IndexAttachLaunchSpace::get_launch_space(
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      perform_collective_wait();
      ReplicateContext* repl_ctx = legion_safe_cast<ReplicateContext*>(context);
      return repl_ctx->compute_index_attach_launch_spaces(sizes, provenance);
    }

    /////////////////////////////////////////////////////////////
    // Index Attach Upper Bound
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexAttachUpperBound::IndexAttachUpperBound(
        ReplicateContext* ctx, CollectiveIndexLocation loc)
      : AllGatherCollective<false>(loc, ctx), node(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    IndexAttachUpperBound::~IndexAttachUpperBound(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void IndexAttachUpperBound::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      if (node != nullptr)
      {
        if (node->is_region())
        {
          rez.serialize<bool>(true);  // is region
          rez.serialize(node->as_region_node()->handle);
        }
        else
        {
          rez.serialize<bool>(false);  // is_region
          rez.serialize(node->as_partition_node()->handle);
        }
      }
      else
      {
        rez.serialize<bool>(true);  // is region
        rez.serialize(LogicalRegion::NO_REGION);
      }
    }

    //--------------------------------------------------------------------------
    void IndexAttachUpperBound::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      bool is_region;
      derez.deserialize(is_region);
      RegionTreeNode* next = nullptr;
      if (is_region)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        if (!handle.exists())
          return;
        next = runtime->get_node(handle);
      }
      else
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        next = runtime->get_node(handle);
      }
      if (node == nullptr)
      {
        node = next;
        return;
      }
      if (next == node)
        return;
      // Bring them to the same depth
      unsigned next_depth = next->get_depth();
      unsigned node_depth = node->get_depth();
      while (next_depth < node_depth)
      {
        legion_assert(node_depth > 0);
        node = node->get_parent();
        node_depth--;
      }
      while (node_depth < next_depth)
      {
        legion_assert(next_depth > 0);
        next = next->get_parent();
        next_depth--;
      }
      while (node != next)
      {
        node = node->get_parent();
        next = next->get_parent();
      }
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* IndexAttachUpperBound::find_upper_bound(RegionTreeNode* n)
    //--------------------------------------------------------------------------
    {
      legion_assert(node == nullptr);
      node = n;
      perform_collective_sync();
      return node;
    }

    /////////////////////////////////////////////////////////////
    // Index Attach Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexAttachExchange::IndexAttachExchange(
        ReplicateContext* ctx, CollectiveIndexLocation loc)
      : AllGatherCollective<false>(loc, ctx)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    IndexAttachExchange::~IndexAttachExchange(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void IndexAttachExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(shard_spaces.size());
      for (const std::pair<const ShardID, std::vector<IndexSpace> >& sit :
           shard_spaces)
      {
        rez.serialize(sit.first);
        rez.serialize<size_t>(sit.second.size());
        for (const IndexSpace& it : sit.second) rez.serialize(it);
      }
    }

    //--------------------------------------------------------------------------
    void IndexAttachExchange::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_shards;
      derez.deserialize(num_shards);
      for (unsigned idx1 = 0; idx1 < num_shards; idx1++)
      {
        ShardID sid;
        derez.deserialize(sid);
        size_t num_spaces;
        derez.deserialize(num_spaces);
        std::vector<IndexSpace>& spaces = shard_spaces[sid];
        spaces.resize(num_spaces);
        for (unsigned idx2 = 0; idx2 < num_spaces; idx2++)
          derez.deserialize(spaces[idx2]);
      }
    }

    //--------------------------------------------------------------------------
    void IndexAttachExchange::exchange_spaces(std::vector<IndexSpace>& spaces)
    //--------------------------------------------------------------------------
    {
      shard_spaces[local_shard].swap(spaces);
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    size_t IndexAttachExchange::get_spaces(
        std::vector<IndexSpace>& spaces, unsigned& local_start)
    //--------------------------------------------------------------------------
    {
      perform_collective_wait();
      size_t total_spaces = 0;
      for (const std::pair<const ShardID, std::vector<IndexSpace> >& it :
           shard_spaces)
        total_spaces += it.second.size();
      spaces.reserve(total_spaces);
      size_t local_size = 0;
      for (const std::pair<const ShardID, std::vector<IndexSpace> >& it :
           shard_spaces)
      {
        if (it.first == local_shard)
        {
          local_start = spaces.size();
          local_size = it.second.size();
        }
        spaces.insert(spaces.end(), it.second.begin(), it.second.end());
      }
      return local_size;
    }

    /////////////////////////////////////////////////////////////
    // Repl Attach Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplAttachOp::ReplAttachOp(void)
      : ReplCollectiveViewCreator<CollectiveViewCreator<AttachOp> >()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplAttachOp::~ReplAttachOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplAttachOp::initialize_replication(
        ReplicateContext* ctx, bool collective_inst, bool dedup_across_shards,
        bool first_local_shard)
    //--------------------------------------------------------------------------
    {
      legion_assert(did_broadcast == nullptr);
      legion_assert(single_broadcast == nullptr);
      resource_barrier = ctx->get_next_attach_resource_barrier();
      collective_instances = collective_inst;
      deduplicate_across_shards = dedup_across_shards;
      is_first_local_shard = first_local_shard;
      if (!collective_instances)
      {
        // Figure out which shard should be the one to make the
        // owner manager and therefore the distributed ID
        ShardID owner_shard = 0;
        switch (resource)
        {
          case LEGION_EXTERNAL_POSIX_FILE:
          case LEGION_EXTERNAL_HDF5_FILE:
            // always use shard 0 for files
            break;
          case LEGION_EXTERNAL_INSTANCE:
            {
              const Memory memory = external_resource->suggested_memory();
              const AddressSpaceID owner_space = memory.address_space();
              const ShardMapping& mapping = ctx->shard_manager->get_mapping();
              for (ShardID sid = 0; sid < mapping.size(); sid++)
              {
                if (mapping[sid] != owner_space)
                  continue;
                owner_shard = sid;
                contains_individual = true;
                break;
              }
              // If we didn't find it we default to 0
              break;
            }
          default:
            break;
        }
        // Setup the distributed ID broadcast and send out the value
        did_broadcast = new ValueBroadcast<DistributedID>(
            ctx, owner_shard, COLLECTIVE_LOC_78);
        // Can only do the broadcast if we know we can make the ID safely
        // For external instances, if they are remote from all shards then
        // we'll need to create a remote manager with a remote distributed ID
        if (did_broadcast->is_origin() &&
            ((resource != LEGION_EXTERNAL_INSTANCE) || contains_individual))
          did_broadcast->broadcast(runtime->get_available_distributed_id());
        single_broadcast = new ValueBroadcast<InstanceEvents>(
            ctx, owner_shard, COLLECTIVE_LOC_75);
      }
    }

    //--------------------------------------------------------------------------
    void ReplAttachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveViewCreator<CollectiveViewCreator<AttachOp> >::activate();
      collective_map_barrier = RtBarrier::NO_RT_BARRIER;
      exchange_index = 0;
      collective_instances = false;
      deduplicate_across_shards = false;
      is_first_local_shard = false;
      contains_individual = false;
      resource_barrier = RtBarrier::NO_RT_BARRIER;
      did_broadcast = nullptr;
      single_broadcast = nullptr;
    }

    //--------------------------------------------------------------------------
    void ReplAttachOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (did_broadcast != nullptr)
        delete did_broadcast;
      if (single_broadcast != nullptr)
        delete single_broadcast;
      ReplCollectiveViewCreator<CollectiveViewCreator<AttachOp> >::deactivate(
          false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplAttachOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      analyze_region_requirements();
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      legion_assert(!collective_map_barrier.exists());
      // We need collective attach barriers for synchronizing the collective
      // updates to the equivalence sets across the shards
      collective_map_barrier = repl_ctx->get_next_collective_map_barriers();
      if (collective_instances)
        create_collective_rendezvous(requirement.parent.get_tree_id(), 0);
      else  // Only need the versioning rendezvous in this case
        ReplCollectiveVersioning<
            CollectiveViewCreator<AttachOp> >::create_collective_rendezvous(0);
    }

    //--------------------------------------------------------------------------
    void ReplAttachOp::trigger_ready(void)
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
    void ReplAttachOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Register this instance with the memory manager
      PhysicalManager* external_manager =
          external_instances[0].get_physical_manager();
      if (collective_instances)
      {
        // Everybody does the attach in the case of collective construction
        if (!deduplicate_across_shards || is_first_local_shard)
        {
          const RtEvent attached = external_manager->attach_external_instance();
          runtime->phase_barrier_arrive(
              resource_barrier, 1 /*count*/, attached);
        }
        else
          runtime->phase_barrier_arrive(resource_barrier, 1 /*count*/);
      }
      else if (external_manager->is_owner())
      {
        const RtEvent attached = external_manager->attach_external_instance();
        runtime->phase_barrier_arrive(resource_barrier, 1 /*count*/, attached);
      }
      else
        runtime->phase_barrier_arrive(resource_barrier, 1 /*count*/);
      // Make sure the attaches are done across all shards before continuing
      if (!resource_barrier.has_triggered())
        resource_barrier.wait();
      // Now perform the base call
      AttachOp::trigger_mapping();
    }

    //--------------------------------------------------------------------------
    RtEvent ReplAttachOp::finalize_complete_mapping(RtEvent pre)
    //--------------------------------------------------------------------------
    {
      legion_assert(collective_map_barrier.exists());
      runtime->phase_barrier_arrive(collective_map_barrier, 1 /*count*/, pre);
      return collective_map_barrier;
    }

    //--------------------------------------------------------------------------
    bool ReplAttachOp::perform_collective_analysis(
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
    bool ReplAttachOp::find_shard_participants(std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      // All shards are participating
      return true;
    }

    //--------------------------------------------------------------------------
    PhysicalManager* ReplAttachOp::create_manager(
        RegionNode* node, const std::vector<FieldID>& field_set,
        const std::vector<size_t>& field_sizes,
        const std::vector<unsigned>& mask_index_map,
        const std::vector<CustomSerdezID>& serdez,
        const FieldMask& external_mask)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      // Only some of the shards are going to actually be creating the
      // instances in this case, this flag will say whether the local shard
      // will be one of the ones performing an instance creation
      const bool making_instance =
          (collective_instances &&
           (is_first_local_shard || !deduplicate_across_shards)) ||
          ((single_broadcast != nullptr) && single_broadcast->is_origin());
      ApEvent ready_event;
      LgEvent unique_event;
      size_t footprint = 0;
      PhysicalInstance instance = PhysicalInstance::NO_INST;
      if (making_instance)
      {
        ready_event = create_external(
            node, field_set, field_sizes, instance, unique_event, footprint);
        if (single_broadcast != nullptr)
          single_broadcast->broadcast({instance, ready_event, unique_event});
      }
      // Do the arrival on the attach barrier for any collective instances
      else if ((single_broadcast != nullptr) && !single_broadcast->is_origin())
      {
        // If we're making a single instance get the name
        const InstanceEvents result = single_broadcast->get_value();
        instance = result.instance;
        ready_event = result.ready_event;
        unique_event = result.unique_event;
      }
      ShardManager* shard_manager = repl_ctx->shard_manager;
      // Now we need to make the instance to span the shards
      if (collective_instances)
      {
        if (deduplicate_across_shards)
        {
          if (is_first_local_shard)
          {
            PhysicalManager* manager =
                node->column_source->create_external_manager(
                    instance, ready_event, footprint, layout_constraint_set,
                    field_set, field_sizes, external_mask, mask_index_map,
                    unique_event, node, serdez,
                    runtime->get_available_distributed_id());
            shard_manager->exchange_shard_local_op_data(
                context_index, exchange_index++, manager);
            return manager;
          }
          else
            return shard_manager->find_shard_local_op_data<PhysicalManager*>(
                context_index, exchange_index++);
        }
        else
        {
          // Each shard is just going to make its own physical manager
          return node->column_source->create_external_manager(
              instance, ready_event, footprint, layout_constraint_set,
              field_set, field_sizes, external_mask, mask_index_map,
              unique_event, node, serdez,
              runtime->get_available_distributed_id());
        }
      }
      else
      {
        // Figure out what the collective mapping is for this instance
        CollectiveMapping* mapping = &shard_manager->get_collective_mapping();
        std::atomic<DistributedID> manager_did(0);
        if ((resource == LEGION_EXTERNAL_INSTANCE) && !contains_individual)
        {
          // We need to send a message to the remote node where no shard
          // lives in order to make this particular instance, we'll give
          // it a collective mapping containing all our address spaces
          // plus the remote address space where the instance lives
          if (did_broadcast->is_origin())
          {
            // Create new collective mapping with the remote address
            // space contained in all of our spaces
            const AddressSpaceID owner_space = instance.address_space();
            legion_assert(!mapping->contains(owner_space));
            mapping = mapping->clone_with(owner_space);
            // We're the ones to send the message to the owner
            const RtUserEvent wait_for = Runtime::create_rt_user_event();
            ExternalCreateRequest rez;
            {
              RezCheck z(rez);
              rez.serialize(node->column_source->handle);
              rez.serialize(instance);
              rez.serialize(ready_event);
              rez.serialize(unique_event);
              rez.serialize(footprint);
              layout_constraint_set.serialize(rez);
              rez.serialize(external_mask);
              rez.serialize<size_t>(field_set.size());
              for (unsigned idx = 0; idx < field_set.size(); idx++)
              {
                rez.serialize(field_set[idx]);
                rez.serialize(field_sizes[idx]);
                rez.serialize(mask_index_map[idx]);
                rez.serialize(serdez[idx]);
              }
              rez.serialize(node->row_source->handle);
              mapping->pack(rez);
              rez.serialize(&manager_did);
              rez.serialize(wait_for);
            }
            rez.dispatch(owner_space);
            // Wait for the response to come back
            wait_for.wait();
            legion_assert(manager_did.load() > 0);
            did_broadcast->broadcast(manager_did.load());
          }
          else
            manager_did.store(did_broadcast->get_value(false /*not origin*/));
        }
        else
          manager_did.store(
              did_broadcast->get_value(!did_broadcast->is_origin()));
        // Making an individual instance across all shards
        // Have the first shard be the one to make it
        if (is_first_local_shard)
        {
          mapping->add_reference();
          PhysicalManager* manager =
              node->column_source->create_external_manager(
                  instance, ready_event, footprint, layout_constraint_set,
                  field_set, field_sizes, external_mask, mask_index_map,
                  unique_event, node, serdez, manager_did.load(), mapping);
          if (mapping->remove_reference())
            delete mapping;
          shard_manager->exchange_shard_local_op_data(
              context_index, exchange_index++, manager);
          return manager;
        }
        else
        {
          PhysicalManager* manager =
              shard_manager->find_shard_local_op_data<PhysicalManager*>(
                  context_index, exchange_index++);
          return manager;
        }
      }
    }

    //--------------------------------------------------------------------------
    RtEvent ReplAttachOp::perform_collective_versioning_analysis(
        unsigned index, LogicalRegion handle, EqSetTracker* tracker,
        const FieldMask& mask, unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      return rendezvous_collective_versioning_analysis(
          index, handle, tracker, runtime->address_space, mask,
          parent_req_index);
    }

    /////////////////////////////////////////////////////////////
    // Repl Index Attach Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplIndexAttachOp::ReplIndexAttachOp(void)
      : ReplCollectiveViewCreator<IndexAttachOp>()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplIndexAttachOp::~ReplIndexAttachOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplIndexAttachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveViewCreator<IndexAttachOp>::activate();
      collective = nullptr;
      participants = nullptr;
      sharding_function = nullptr;
      interfering_check_id = 0;
      interfering_exchange = nullptr;
    }

    //--------------------------------------------------------------------------
    void ReplIndexAttachOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (collective != nullptr)
        delete collective;
      if (participants != nullptr)
        delete participants;
      if (interfering_exchange != nullptr)
        delete interfering_exchange;
      ReplCollectiveViewCreator<IndexAttachOp>::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplIndexAttachOp::initialize_replication(ReplicateContext* ctx)
    //--------------------------------------------------------------------------
    {
      legion_assert(collective == nullptr);
      collective = new IndexAttachExchange(ctx, COLLECTIVE_LOC_25);
      std::vector<IndexSpace> spaces(points.size());
      for (unsigned idx = 0; idx < points.size(); idx++)
        spaces[idx] = points[idx]->get_requirement().region.get_index_space();
      collective->exchange_spaces(spaces);
      participants = new ShardParticipantsExchange(ctx, COLLECTIVE_LOC_103);
      participants->exchange(points.size() > 0);
      if (runtime->safe_model)
        interfering_check_id =
            ctx->get_next_collective_index(COLLECTIVE_LOC_108);
    }

    //--------------------------------------------------------------------------
    void ReplIndexAttachOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(sharding_function == nullptr);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      sharding_function = repl_ctx->get_attach_detach_sharding_function();
      IndexAttachOp::trigger_prepipeline_stage();
    }

    //--------------------------------------------------------------------------
    void ReplIndexAttachOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(sharding_function != nullptr);
      std::vector<IndexSpace> spaces;
      unsigned local_start = 0;
      size_t local_size = collective->get_spaces(spaces, local_start);
      if (requirement.handle_type == LEGION_PARTITION_PROJECTION)
        requirement.projection = parent_ctx->compute_index_attach_projection(
            runtime->get_node(requirement.partition.index_partition), this,
            local_start, local_size, spaces, false /*can use identity*/);
      else
        requirement.projection = parent_ctx->compute_index_attach_projection(
            runtime->get_node(requirement.region.index_space), this,
            local_start, local_size, spaces, false /*can use identity*/);
      // Save this for later when we go to detach it
      resources.impl->set_projection(requirement.projection);
      log_requirement();
      analyze_region_requirements(launch_space, sharding_function);
      // Always perform a collective rendezvous for these points
      create_collective_rendezvous(requirement.parent.get_tree_id(), 0);
    }

    //--------------------------------------------------------------------------
    void ReplIndexAttachOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      if (points.empty())
      {
        if (runtime->safe_model)
          start_check_point_requirements();
        // Still need to wait for our collectives to be done
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
        IndexAttachOp::trigger_ready();
    }

    //--------------------------------------------------------------------------
    void ReplIndexAttachOp::finish_check_point_requirements(
        std::map<unsigned, std::vector<std::pair<DomainPoint, Domain> > >&
            point_domains)
    //--------------------------------------------------------------------------
    {
      // See if this is the first time through or not
      if (interfering_exchange == nullptr)
      {
        // First time through, make the exchange and kick it off
        legion_assert(interfering_check_id > 0);
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(parent_ctx);
        interfering_exchange = new InterferingPointExchange<ReplIndexAttachOp>(
            repl_ctx, interfering_check_id, this);
        // Record a dependence on this to make sure it is done before we
        // clean up the operation
        commit_preconditions.insert(interfering_exchange->get_done_event());
        interfering_exchange->exchange_domain_points(point_domains);
      }
      else  // Second time through call the base class since we have the
            // results
        IndexAttachOp::finish_check_point_requirements(point_domains);
    }

    //--------------------------------------------------------------------------
    bool ReplIndexAttachOp::are_all_direct_children(bool local)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      AllReduceCollective<ProdReduction<bool>, false> all_direct_children(
          repl_ctx, repl_ctx->get_next_collective_index(
                        COLLECTIVE_LOC_27, true /*logical*/));
      return all_direct_children.sync_all_reduce(local);
    }

    //--------------------------------------------------------------------------
    bool ReplIndexAttachOp::find_shard_participants(
        std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      legion_assert(participants != nullptr);
      return participants->find_shard_participants(shards);
    }

    /////////////////////////////////////////////////////////////
    // Remote Attach Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteAttachOp::RemoteAttachOp(Operation* ptr, AddressSpaceID src)
      : RemoteOp(ptr, src)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RemoteAttachOp::~RemoteAttachOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID RemoteAttachOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t RemoteAttachOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteAttachOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteAttachOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const char* RemoteAttachOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[ATTACH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind RemoteAttachOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return ATTACH_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteAttachOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      rez.serialize(index_point);
    }

    //--------------------------------------------------------------------------
    void RemoteAttachOp::unpack(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(index_point);
    }

  }  // namespace Internal
}  // namespace Legion
