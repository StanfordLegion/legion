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

#include "legion/operations/dependent.h"
#include "legion/contexts/replicate.h"
#include "legion/managers/mapper.h"
#include "legion/managers/shard.h"
#include "legion/nodes/region.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // External Partition
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalPartition::ExternalPartition(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ExternalPartition::pack_external_partition(
        Serializer& rez, AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_region_requirement(requirement, rez);
      rez.serialize<bool>(is_index_space);
      rez.serialize(index_domain);
      rez.serialize(index_point);
      pack_mappable(*this, rez);
      rez.serialize(get_context_index());
    }

    //--------------------------------------------------------------------------
    void ExternalPartition::unpack_external_partition(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      unpack_region_requirement(requirement, derez);
      derez.deserialize<bool>(is_index_space);
      derez.deserialize(index_domain);
      derez.deserialize(index_point);
      unpack_mappable(*this, derez);
      uint64_t index;
      derez.deserialize(index);
      set_context_index(index);
    }

    /////////////////////////////////////////////////////////////
    // Dependent Partition Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DependentPartitionOp::DependentPartitionOp(void)
      : ExternalPartition(), Operation(), thunk(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    DependentPartitionOp::~DependentPartitionOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_field(
        InnerContext* ctx, IndexPartition pid, LogicalRegion handle,
        LogicalRegion parent, IndexSpace color_space, FieldID fid, MapperID id,
        MappingTagID t, const UntypedBuffer& marg, Provenance* prov)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_operation(ctx, prov);
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      requirement =
          RegionRequirement(handle, LEGION_READ_ONLY, LEGION_EXCLUSIVE, parent);
      requirement.add_field(fid);
      if (runtime->safe_model)
      {
        verify_requirement(requirement);
        const size_t field_size =
            runtime->get_field_size(handle.get_field_space(), fid);
        const size_t coord_size =
            runtime->get_coordinate_size(color_space, false /*range*/);
        if (field_size != coord_size)
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "The field size for partition-by-field operation does not "
                << "match the size of the coordinate type of the color space "
                << "of the resulting partition. Field " << fid << " has size "
                << field_size << " bytes but the coordinates "
                << "of color space " << color_space << " of partition " << pid
                << " are " << coord_size << " bytes for " << *this << ".";
          error.raise();
        }
        const CustomSerdezID serdez =
            runtime->get_field_serdez(handle.get_field_space(), fid);
        if (serdez != 0)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Serdez fields are not permitted to be used for any "
                << "dependent partitioning calls. Field " << fid
                << " has serdez function " << serdez << " and was passed "
                << "to partition-by-field " << *this << ".";
          error.raise();
        }
      }
      parent_req_index = ctx->find_parent_region_index(this, requirement);
      map_id = id;
      tag = t;
      mapper_data_size = marg.get_size();
      if (mapper_data_size > 0)
      {
        legion_assert(mapper_data == nullptr);
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, marg.get_ptr(), mapper_data_size);
      }
      legion_assert(thunk == nullptr);
      thunk = new ByFieldThunk(pid);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_image(
        InnerContext* ctx, IndexPartition pid, IndexSpace handle,
        LogicalPartition projection, LogicalRegion parent, FieldID fid,
        MapperID id, MappingTagID t, const UntypedBuffer& marg,
        Provenance* prov)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_operation(ctx, prov);
      requirement = RegionRequirement(
          projection, 0 /*identity*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
          parent);
      requirement.add_field(fid);
      if (runtime->safe_model)
      {
        verify_requirement(requirement, 0, true /*allow projection*/);
        const size_t field_size =
            runtime->get_field_size(projection.get_field_space(), fid);
        const size_t coord_size =
            runtime->get_coordinate_size(handle, false /*range*/);
        if (field_size != coord_size)
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "The field size for partition-by-image operation does not "
                << "match the size of the coordinate types of the projection "
                << "partition. Field " << fid << "  has size " << field_size
                << " bytes but the coordinates of the projection partition "
                << pid << " are " << coord_size << " bytes for " << *this
                << ".";
          error.raise();
        }
        const CustomSerdezID serdez =
            runtime->get_field_serdez(projection.get_field_space(), fid);
        if (serdez != 0)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Serdez fields are not permitted to be used for any "
                << "dependent partitioning calls. Field " << fid
                << " has serdez function " << serdez
                << " and was passed to partition-by-image " << *this << ".";
          error.raise();
        }
      }
      parent_req_index = ctx->find_parent_region_index(this, requirement);
      map_id = id;
      tag = t;
      mapper_data_size = marg.get_size();
      if (mapper_data_size > 0)
      {
        legion_assert(mapper_data == nullptr);
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, marg.get_ptr(), mapper_data_size);
      }
      legion_assert(thunk == nullptr);
      thunk = new ByImageThunk(pid, projection.get_index_partition());
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_image_range(
        InnerContext* ctx, IndexPartition pid, IndexSpace handle,
        LogicalPartition projection, LogicalRegion parent, FieldID fid,
        MapperID id, MappingTagID t, const UntypedBuffer& marg,
        Provenance* prov)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_operation(ctx, prov);
      requirement = RegionRequirement(
          projection, 0 /*identity*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE,
          parent);
      requirement.add_field(fid);
      if (runtime->safe_model)
      {
        verify_requirement(requirement, 0, true /*allow projection*/);
        const size_t field_size =
            runtime->get_field_size(projection.get_field_space(), fid);
        const size_t coord_size =
            runtime->get_coordinate_size(handle, true /*range*/);
        if (field_size != coord_size)
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error
              << "The field size for partition-by-image-range operation does "
              << "not match the size of the coordinate types of the projection "
              << "partition. Field " << fid << " has size " << field_size
              << " bytes but the coordinates of the projection partition "
              << pid << "  are " << coord_size << " bytes for dependent "
              << "partition " << *this << ".";
          error.raise();
        }
        const CustomSerdezID serdez =
            runtime->get_field_serdez(projection.get_field_space(), fid);
        if (serdez != 0)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Serdez fields are not permitted to be used for any "
                << "dependent partitioning calls. Field " << fid
                << " has serdez function " << serdez
                << " and was passed to partition-by-image-range " << *this
                << ".";
          error.raise();
        }
      }
      parent_req_index = ctx->find_parent_region_index(this, requirement);
      map_id = id;
      tag = t;
      mapper_data_size = marg.get_size();
      if (mapper_data_size > 0)
      {
        legion_assert(mapper_data == nullptr);
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, marg.get_ptr(), mapper_data_size);
      }
      legion_assert(thunk == nullptr);
      thunk = new ByImageRangeThunk(pid, projection.get_index_partition());
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_preimage(
        InnerContext* ctx, IndexPartition pid, IndexPartition proj,
        LogicalRegion handle, LogicalRegion parent, FieldID fid, MapperID id,
        MappingTagID t, const UntypedBuffer& marg, Provenance* prov)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_operation(ctx, prov);
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      requirement =
          RegionRequirement(handle, LEGION_READ_ONLY, LEGION_EXCLUSIVE, parent);
      requirement.add_field(fid);
      if (runtime->safe_model)
      {
        verify_requirement(requirement);
        const size_t field_size =
            runtime->get_field_size(handle.get_field_space(), fid);
        IndexSpace proj_parent = runtime->get_parent_index_space(proj);
        const size_t coord_size =
            runtime->get_coordinate_size(proj_parent, false /*range*/);
        if (field_size != coord_size)
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error
              << "The field size for partition-by-preimage operation does not "
              << "match the size of the coordinate types of the projection "
              << "partition. Field " << fid << " has size " << field_size
              << " bytes but the coordinates of the projection partition "
              << proj << " are " << coord_size << " bytes for dependent "
              << "partition " << *this << ".";
          error.raise();
        }
        const CustomSerdezID serdez =
            runtime->get_field_serdez(handle.get_field_space(), fid);
        if (serdez != 0)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Serdez fields are not permitted to be used for any "
                << "dependent partitioning calls. Field " << fid
                << " has serdez function " << serdez
                << " and was passed to partition-by-preimage " << *this << ".";
          error.raise();
        }
      }
      parent_req_index = ctx->find_parent_region_index(this, requirement);
      map_id = id;
      tag = t;
      mapper_data_size = marg.get_size();
      if (mapper_data_size > 0)
      {
        legion_assert(mapper_data == nullptr);
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, marg.get_ptr(), mapper_data_size);
      }
      legion_assert(thunk == nullptr);
      thunk = new ByPreimageThunk(pid, proj);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_preimage_range(
        InnerContext* ctx, IndexPartition pid, IndexPartition proj,
        LogicalRegion handle, LogicalRegion parent, FieldID fid, MapperID id,
        MappingTagID t, const UntypedBuffer& marg, Provenance* prov)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_operation(ctx, prov);
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      requirement =
          RegionRequirement(handle, LEGION_READ_ONLY, LEGION_EXCLUSIVE, parent);
      requirement.add_field(fid);
      if (runtime->safe_model)
      {
        verify_requirement(requirement);
        const size_t field_size =
            runtime->get_field_size(handle.get_field_space(), fid);
        IndexSpace proj_parent = runtime->get_parent_index_space(proj);
        const size_t coord_size =
            runtime->get_coordinate_size(proj_parent, true /*range*/);
        if (field_size != coord_size)
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error
              << "The field size for partition-by-preimage-range operation "
                 "does "
              << "not match the size of the coordinate types of the projection "
              << "partition. Field " << fid << " has size " << field_size
              << " bytes but the coordinates of the projection partition "
              << proj << " are " << coord_size << " bytes for dependent "
              << "partition " << *this << ".";
          error.raise();
        }
        const CustomSerdezID serdez =
            runtime->get_field_serdez(handle.get_field_space(), fid);
        if (serdez != 0)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Serdez fields are not permitted to be used for any "
                << "dependent partitioning calls. Field " << fid
                << " has serdez function " << serdez
                << " and was passed to partition-by-preimage-range " << *this
                << ".";
          error.raise();
        }
      }
      parent_req_index = ctx->find_parent_region_index(this, requirement);
      map_id = id;
      tag = t;
      mapper_data_size = marg.get_size();
      if (mapper_data_size > 0)
      {
        legion_assert(mapper_data == nullptr);
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, marg.get_ptr(), mapper_data_size);
      }
      legion_assert(thunk == nullptr);
      thunk = new ByPreimageRangeThunk(pid, proj);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::initialize_by_association(
        InnerContext* ctx, LogicalRegion domain, LogicalRegion domain_parent,
        FieldID fid, IndexSpace range, MapperID id, MappingTagID t,
        const UntypedBuffer& marg, Provenance* prov)
    //--------------------------------------------------------------------------
    {
      parent_task = ctx->get_task();
      initialize_operation(ctx, prov);
      // start-off with non-projection requirement
      requirement = RegionRequirement(
          domain, LEGION_READ_WRITE, LEGION_EXCLUSIVE, domain_parent);
      requirement.add_field(fid);
      if (runtime->safe_model)
      {
        verify_requirement(requirement);
        const size_t field_size =
            runtime->get_field_size(domain.get_field_space(), fid);
        const size_t coord_size =
            runtime->get_coordinate_size(range, false /*range*/);
        if (field_size != coord_size)
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error
              << "The field size for create-by-association operation does not "
              << "match the size of the range index space. Field " << fid
              << " has size " << field_size << " bytes but the coordinates of "
              << "the range index space " << range << " are " << coord_size
              << " bytes for create-by-association " << *this << ".";
          error.raise();
        }
        const CustomSerdezID serdez =
            runtime->get_field_serdez(domain.get_field_space(), fid);
        if (serdez != 0)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Serdez fields are not permitted to be used for any "
                << "dependent partitioning calls. Field " << fid
                << " has serdez function " << serdez
                << " and was passed to create-by-association " << *this << ".";
          error.raise();
        }
      }
      parent_req_index = ctx->find_parent_region_index(this, requirement);
      map_id = id;
      tag = t;
      mapper_data_size = marg.get_size();
      if (mapper_data_size > 0)
      {
        legion_assert(mapper_data == nullptr);
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, marg.get_ptr(), mapper_data_size);
      }
      legion_assert(thunk == nullptr);
      thunk = new AssociationThunk(domain.get_index_space(), range);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::perform_logging(void) const
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_dependent_partition_operation(
          parent_ctx->get_unique_id(), unique_op_id,
          thunk->get_partition().get_id(), thunk->get_kind());
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::log_requirement(void) const
    //--------------------------------------------------------------------------
    {
      if (requirement.handle_type == LEGION_PARTITION_PROJECTION)
      {
        LegionSpy::log_logical_requirement(
            unique_op_id, 0 /*idx*/, false /*region*/,
            requirement.partition.index_partition.get_id(),
            requirement.partition.field_space.get_id(),
            requirement.partition.get_tree_id(), requirement.privilege,
            requirement.prop, requirement.redop,
            requirement.parent.index_space.get_id());
        LegionSpy::log_requirement_projection(
            unique_op_id, 0 /*idx*/, requirement.projection);
        log_launch_space(launch_space->handle);
      }
      else
        LegionSpy::log_logical_requirement(
            unique_op_id, 0 /*idx*/, true /*region*/,
            requirement.region.index_space.get_id(),
            requirement.region.field_space.get_id(),
            requirement.region.get_tree_id(), requirement.privilege,
            requirement.prop, requirement.redop,
            requirement.parent.index_space.get_id());
      LegionSpy::log_requirement_fields(
          unique_op_id, 0 /*index*/, requirement.privilege_fields);
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& DependentPartitionOp::get_requirement(
        unsigned idx) const
    //--------------------------------------------------------------------------
    {
      return requirement;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // If we're an index space op, promote a singular region requirement
      // up to a projection region requirement for accuracy
      if (is_index_space &&
          (requirement.handle_type == LEGION_SINGULAR_PROJECTION))
      {
        requirement.handle_type = LEGION_REGION_PROJECTION;
        requirement.projection = 0;
      }
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Before doing the dependence analysis we have to ask the
      // mapper whether it would like to make this an index space
      // operation or a single operation
      select_partition_projection();
      // Do thise now that we've picked our region requirement
      log_requirement();
      analyze_region_requirements(is_index_space ? launch_space : nullptr);
      // Record this dependent partition op with the context so that it
      // can track implicit dependences on it for later operations
      parent_ctx->update_current_implicit_creation(this);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::select_partition_projection(void)
    //--------------------------------------------------------------------------
    {
      // If this is an image then we already made this a projection
      // region requirement to reflect that
      IndexPartNode* partition_node = nullptr;
      if (thunk->is_image())
      {
        legion_assert(requirement.handle_type == LEGION_PARTITION_PROJECTION);
        partition_node =
            runtime->get_node(requirement.partition.get_index_partition());
      }
      else
      {
        legion_assert(requirement.handle_type == LEGION_SINGULAR_PROJECTION);
        // Not an image so ask the mapper if it wants to make this into
        // and index space operation or not
        Mapper::SelectPartitionProjectionInput input;
        Mapper::SelectPartitionProjectionOutput output;
        // Find the open complete projections, and then invoke the mapper call
        find_open_complete_partitions(input.open_complete_partitions);
        // Invoke the mapper
        if (mapper == nullptr)
        {
          Processor exec_proc = parent_ctx->get_executing_processor();
          mapper = runtime->find_mapper(exec_proc, map_id);
        }
        mapper->invoke_select_partition_projection(this, input, output);
        // Check the output
        if (output.chosen_partition == LogicalPartition::NO_PART)
          return;
        partition_node =
            runtime->get_node(output.chosen_partition.get_index_partition());
        // Make sure that it is complete, and then update our information
        // We also allow the mapper to pick the same projection partition
        // if the partition operation is an image or image-range
        if (runtime->safe_mapper && !partition_node->is_complete(false) &&
            !thunk->safe_projection(partition_node->handle))
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from invocation of "
                << "'select_partition_projection' on mapper " << *mapper
                << ".Mapper selected a logical "
                << "partition that is not complete for " << *this << ".";
          error.raise();
        }
        // Update the region requirement and other information
        requirement.partition = output.chosen_partition;
        requirement.handle_type = LEGION_PARTITION_PROJECTION;
        requirement.projection = 0;  // always default
      }
      launch_space = partition_node->color_space;
      add_launch_space_reference(launch_space);
      index_domain = partition_node->color_space->get_color_space_domain();
      is_index_space = true;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // See if this is an index space operation
      if (is_index_space)
      {
        legion_assert(requirement.handle_type == LEGION_PARTITION_PROJECTION);
        // Need to get the launch domain in case it is different than
        // the original index domain due to control replication
        IndexSpaceNode* local_points = get_shard_points();
        Domain launch_domain = local_points->get_tight_domain();
        // Now enumerate the points and kick them off
        size_t num_points = local_points->get_volume();
        legion_assert(num_points > 0);
        points.reserve(num_points);
        for (Domain::DomainPointIterator itr(launch_domain); itr; itr++)
        {
          PointDepPartOp* point = runtime->get_operation<PointDepPartOp>();
          point->initialize(this, itr.p);
          points.emplace_back(point);
        }
        // Perform the projections
        ProjectionFunction* function =
            runtime->find_projection_function(requirement.projection);
        std::vector<ProjectionPoint*> projection_points(
            points.begin(), points.end());
        function->project_points(
            this, 0 /*idx*/, requirement, index_domain, projection_points,
            nullptr /*no pointwise*/, parent_ctx->get_total_shards(),
            false /*is replaying*/);
        // Launch the points
        for (PointDepPartOp* point : points)
        {
          point->log_requirement();
          map_applied_conditions.insert(point->get_mapped_event());
          point->launch();
        }
        LegionSpy::log_operation_events(
            unique_op_id, ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
        // We are mapped when all our points are mapped
        finalize_mapping();
      }
      else
      {
        std::set<RtEvent> preconditions;
        // Path for a non-index space implementation
        perform_versioning_analysis(
            0 /*idx*/, requirement, version_info, preconditions);
        // Give these operations slightly higher priority since
        // they are likely needed for other operations
        if (!preconditions.empty())
          enqueue_ready_operation(
              Runtime::merge_events(preconditions),
              LG_THROUGHPUT_DEFERRED_PRIORITY);
        else
          enqueue_ready_operation(
              RtEvent::NO_RT_EVENT, LG_THROUGHPUT_DEFERRED_PRIORITY);
      }
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(requirement.handle_type == LEGION_SINGULAR_PROJECTION);
      const PhysicalTraceInfo trace_info(this, 0 /*index*/);
      // Perform the mapping call to get the physical isntances
      InstanceSet mapped_instances;
      std::vector<PhysicalManager*> source_instances;
      const bool record_valid =
          invoke_mapper(mapped_instances, source_instances);
      log_mapping_decision(0 /*idx*/, requirement, mapped_instances);
      legion_assert(!mapped_instances.empty());
      // Then we can register our mapped_instances
      ApUserEvent part_done = Runtime::create_ap_user_event(&trace_info);
      ApEvent instances_ready = physical_perform_updates_and_registration(
          requirement, version_info, 0 /*idx*/, ApEvent::NO_AP_EVENT, part_done,
          mapped_instances, source_instances, trace_info,
          map_applied_conditions, false /*check collective*/, record_valid);
      ApEvent done_event = trigger_thunk(
          requirement.region.get_index_space(), instances_ready,
          mapped_instances, trace_info, index_point);
      Runtime::trigger_event(
          part_done, done_event, trace_info, map_applied_conditions);
      record_completion_effect(part_done);
      LegionSpy::log_operation_events(unique_op_id, done_event, part_done);
      // Once we are done running these routines, we can mark
      // that the handles have all been completed
      finalize_mapping();
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::finalize_mapping(void)
    //--------------------------------------------------------------------------
    {
      RtEvent mapping_applied;
      if (!map_applied_conditions.empty())
        mapping_applied = Runtime::merge_events(map_applied_conditions);
      if (!acquired_instances.empty())
        mapping_applied = release_nonempty_acquired_instances(
            mapping_applied, acquired_instances);
      complete_mapping(mapping_applied);
    }

    //--------------------------------------------------------------------------
    ApEvent DependentPartitionOp::trigger_thunk(
        IndexSpace handle, ApEvent instances_ready,
        const InstanceSet& mapped_insts, const PhysicalTraceInfo& info,
        const DomainPoint& color)
    //--------------------------------------------------------------------------
    {
      legion_assert(requirement.privilege_fields.size() == 1);
      legion_assert(mapped_insts.size() == 1);
      IndexSpaceNode* node = runtime->get_node(handle);
      Domain domain;
      ApUserEvent to_trigger;
      ApEvent domain_ready = node->get_loose_domain(domain, to_trigger);
      if (is_index_space)
      {
        // Update our data structure and see if we are the ones
        // to perform the operation
        bool ready = false;
        {
          AutoLock o_lock(op_lock);
          instances.resize(instances.size() + 1);
          FieldDataDescriptor& desc = instances.back();
          const InstanceRef& ref = mapped_insts[0];
          PhysicalManager* manager = ref.get_physical_manager();
          desc.inst = manager->get_instance();
          desc.domain = domain;
          desc.color = color;
          desc.unique = manager->get_unique_event();
          desc.space = handle.get_id(false /*filter*/);
          if (instances_ready.exists())
            index_preconditions.emplace_back(instances_ready);
          if (domain_ready.exists())
            index_preconditions.emplace_back(domain_ready);
          legion_assert(!points.empty());
          ready = (instances.size() == points.size());
          if (!intermediate_index_event.exists())
            intermediate_index_event = Runtime::create_ap_user_event(&info);
        }
        if (ready)
        {
          const FieldID fid = *(requirement.privilege_fields.begin());
          ApEvent done_event = thunk->perform(
              this, fid, Runtime::merge_events(&info, index_preconditions),
              instances);
          Runtime::trigger_event(
              intermediate_index_event, done_event, info,
              map_applied_conditions);
        }
        if (to_trigger.exists())
          Runtime::trigger_event_untraced(to_trigger, intermediate_index_event);
        return intermediate_index_event;
      }
      else
      {
        legion_assert(instances.empty());
        instances.resize(1);
        FieldDataDescriptor& desc = instances[0];
        const InstanceRef& ref = mapped_insts[0];
        PhysicalManager* manager = ref.get_physical_manager();
        desc.inst = manager->get_instance();
        desc.domain = domain;
        desc.color = color;
        desc.unique = manager->get_unique_event();
        desc.space = handle.get_id(false /*filter*/);
        const FieldID fid = *(requirement.privilege_fields.begin());
        if (domain_ready.exists())
        {
          if (instances_ready.exists())
            instances_ready =
                Runtime::merge_events(&info, domain_ready, instances_ready);
          else
            instances_ready = domain_ready;
        }
        ApEvent result = thunk->perform(this, fid, instances_ready, instances);
        if (to_trigger.exists())
          Runtime::trigger_event_untraced(to_trigger, result);
        return result;
      }
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::handle_point_complete(ApEvent effect)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_index_space);
      legion_assert(-1 <= points_completed.load());
      if (effect.exists())
        record_completion_effect(effect);
      const unsigned received = ++points_completed;
      legion_assert(received <= points.size());
      if (received == points.size())
        complete_execution();
    }

    //--------------------------------------------------------------------------
    bool DependentPartitionOp::invoke_mapper(
        InstanceSet& mapped_instances,
        std::vector<PhysicalManager*>& source_instances)
    //--------------------------------------------------------------------------
    {
      Mapper::MapPartitionInput input;
      Mapper::MapPartitionOutput output;
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
        prepare_for_mapping(
            valid_instances, collectives, input.valid_instances,
            input.valid_collectives);
      }
      output.copy_fill_priority = 0;
      mapper->invoke_map_partition(this, input, output);
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
          requirement, output.chosen_instances, mapped_instances, bad_tree,
          missing_fields, &acquired_instances, unacquired,
          runtime->safe_mapper);
      if (bad_tree > 0)
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from invocation of 'map_partition' on "
                 "mapper "
              << *mapper << ". Mapper selected instance from region tree "
              << bad_tree << " to satisfy a region requirement for " << *this
              << " whose logical region is from region tree "
              << requirement.region.get_tree_id() << ".";
        error.raise();
      }
      if (!missing_fields.empty())
      {
        for (const FieldID& fid : missing_fields)
        {
          const void* name;
          size_t name_size;
          if (!runtime->retrieve_semantic_information(
                  requirement.region.get_field_space(), fid,
                  LEGION_NAME_SEMANTIC_TAG, name, name_size, true, false))
            name = "(no name)";
          log_legion.error(
              "Missing instance for field %s (FieldID: %d)",
              static_cast<const char*>(name), fid);
        }
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from invocation of 'map_partition' on "
                 "mapper "
              << *mapper << ". Mapper failed to specify a physical "
              << "instance for " << missing_fields.size()
              << " fields of the region requirement for " << *this
              << ". This missing fields are listed above.";
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
                << "Invalid mapper output from 'map_partition' invocation on "
                   "mapper "
                << *mapper << ". Mapper selected physical instance for "
                << *this << " which has already been collected. "
                << "If the mapper had properly acquired this instance as part "
                   "of the "
                << "mapper call it would have detected this. Please update the "
                << "mapper to abide by proper mapping conventions.";
            error.raise();
          }
        }
        // If we did successfully acquire them, still issue the warning
        Warning warning;
        warning << "Mapper " << *mapper << " faield to acquire instance for "
                << *this
                << "in 'map_partition' call. You may experience undefined "
                << "behavior as a consequence.";
        warning.raise();
      }
      if (virtual_index >= 0)
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from invocation of 'map_partition' on "
                 "mapper "
              << *mapper << ". Mapper requested creation of a virtual "
              << "mapping for " << *this << ".";
        error.raise();
      }
      // If we are doing unsafe mapping, then we can return
      if (!runtime->safe_mapper)
        return output.track_valid_region;
      // Iterate over the instances and make sure they are all valid
      // for the given logical region which we are mapping
      std::vector<LogicalRegion> regions_to_check(1, requirement.region);
      for (unsigned idx = 0; idx < mapped_instances.size(); idx++)
      {
        PhysicalManager* manager = mapped_instances[idx].get_physical_manager();
        if (!manager->meets_regions(regions_to_check))
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error
              << "Invalid mapper output from invocation of 'map_partition' on "
                 "mapper "
              << *mapper << ". Mapper specified an instance that "
              << "does not meet the logical region requirement for " << *this
              << ". ";
          error.raise();
        }
        if (manager->is_reduction_manager())
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error
              << "Invalid mapper output from invocation of 'map_partition' on "
                 "mapper "
              << *mapper << ". Mapper selected an illegal specialized "
              << "reduction instance for " << *this << ".";
          error.raise();
        }
        // This is a temporary check to guarantee that instances for
        // dependent partitioning operations are in memories that
        // Realm supports for now. In the future this should be fixed
        // so that realm supports all kinds of memories for dependent
        // partitioning operations (see issue #516)
        const Memory::Kind mem_kind =
            manager->layout->constraints->memory_constraint.get_kind();
        if ((mem_kind != Memory::GLOBAL_MEM) &&
            (mem_kind != Memory::SYSTEM_MEM) &&
            (mem_kind != Memory::REGDMA_MEM) &&
            (mem_kind != Memory::SOCKET_MEM) &&
            (mem_kind != Memory::Z_COPY_MEM))
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error
              << "Invalid mapper output from invocation of 'map_partition' on "
                 "mapper "
              << *mapper << " for " << *this << ". Mapper specified"
              << " an instance in memory(memories) with kind " << mem_kind
              << " which is not supported for dependent partition operations "
              << "currently (see Legion issue #516). Please pick an "
              << "instance in a CPU-visible memory for now.";
          error.raise();
        }
      }
      return output.track_valid_region;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      if (profiling_reported.exists())
        finalize_partition_profiling();
      bool commit_now = false;
      if (is_index_space)
      {
        AutoLock o_lock(op_lock);
        legion_assert(!commit_request);
        commit_request = true;
        commit_now = (points.size() == points_committed);
      }
      else
        commit_now = true;
      if (commit_now)
        commit_operation(true /*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::finalize_partition_profiling(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(profiling_reported.exists());
      if (outstanding_profiling_requests == 0)
      {
        // We're not expecting any profiling callbacks so we need to
        // do one ourself to inform the mapper that there won't be any
        Mapping::Mapper::PartitionProfilingInfo info;
        info.total_reports = 0;
        info.fill_response = false;  // make valgrind happy
        mapper->invoke_partition_report_profiling(this, info);
        Runtime::trigger_event(profiling_reported);
      }
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::handle_point_commit(RtEvent point_committed)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_index_space);
      bool commit_now = false;
      {
        AutoLock o_lock(op_lock);
        points_committed++;
        if (point_committed.exists())
          commit_preconditions.insert(point_committed);
        commit_now = commit_request && (points.size() == points_committed);
      }
      if (commit_now)
        commit_operation(
            true /*deactivate*/, Runtime::merge_events(commit_preconditions));
    }

    //--------------------------------------------------------------------------
    ApEvent DependentPartitionOp::ByFieldThunk::perform(
        DependentPartitionOp* op, FieldID fid, ApEvent instances_ready,
        std::vector<FieldDataDescriptor>& instances,
        const std::map<DomainPoint, Domain>* remote_targets,
        std::vector<DeppartResult>* results)
    //--------------------------------------------------------------------------
    {
      legion_assert((remote_targets == nullptr) || remote_targets->empty());
      IndexPartNode* partition = runtime->get_node(pid);
      return partition->parent->create_by_field(
          op, fid, partition, instances, results, instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent DependentPartitionOp::ByImageThunk::perform(
        DependentPartitionOp* op, FieldID fid, ApEvent instances_ready,
        std::vector<FieldDataDescriptor>& instances,
        const std::map<DomainPoint, Domain>* remote_targets,
        std::vector<DeppartResult>* results)
    //--------------------------------------------------------------------------
    {
      // Should never see these here
      legion_assert(remote_targets == nullptr);
      legion_assert(results == nullptr);
      IndexPartNode* part = runtime->get_node(pid);
      IndexPartNode* proj = runtime->get_node(projection);
      return part->parent->create_by_image(
          op, fid, part, proj, instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent DependentPartitionOp::ByImageRangeThunk::perform(
        DependentPartitionOp* op, FieldID fid, ApEvent instances_ready,
        std::vector<FieldDataDescriptor>& instances,
        const std::map<DomainPoint, Domain>* remote_targets,
        std::vector<DeppartResult>* results)
    //--------------------------------------------------------------------------
    {
      // Should never see these here
      legion_assert(remote_targets == nullptr);
      legion_assert(results == nullptr);
      IndexPartNode* part = runtime->get_node(pid);
      IndexPartNode* proj = runtime->get_node(projection);
      return part->parent->create_by_image_range(
          op, fid, part, proj, instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent DependentPartitionOp::ByPreimageThunk::perform(
        DependentPartitionOp* op, FieldID fid, ApEvent instances_ready,
        std::vector<FieldDataDescriptor>& instances,
        const std::map<DomainPoint, Domain>* remote_targets,
        std::vector<DeppartResult>* results)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* part = runtime->get_node(pid);
      IndexPartNode* proj = runtime->get_node(projection);
      return part->parent->create_by_preimage(
          op, fid, part, proj, instances, remote_targets, results,
          instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent DependentPartitionOp::ByPreimageRangeThunk::perform(
        DependentPartitionOp* op, FieldID fid, ApEvent instances_ready,
        std::vector<FieldDataDescriptor>& instances,
        const std::map<DomainPoint, Domain>* remote_targets,
        std::vector<DeppartResult>* results)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* part = runtime->get_node(pid);
      IndexPartNode* proj = runtime->get_node(projection);
      return part->parent->create_by_preimage_range(
          op, fid, part, proj, instances, remote_targets, results,
          instances_ready);
    }

    //--------------------------------------------------------------------------
    ApEvent DependentPartitionOp::AssociationThunk::perform(
        DependentPartitionOp* op, FieldID fid, ApEvent instances_ready,
        std::vector<FieldDataDescriptor>& instances,
        const std::map<DomainPoint, Domain>* remote_targets,
        std::vector<DeppartResult>* results)
    //--------------------------------------------------------------------------
    {
      // Should never see these here
      legion_assert(remote_targets == nullptr);
      legion_assert(results == nullptr);
      IndexSpaceNode* dom = runtime->get_node(domain);
      IndexSpaceNode* rng = runtime->get_node(range);
      return dom->create_association(op, fid, rng, instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    unsigned DependentPartitionOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx == 0);
      legion_assert(parent_req_index != TRACED_PARENT_INDEX);
      return parent_req_index;
    }

    //--------------------------------------------------------------------------
    Partition::PartitionKind DependentPartitionOp::get_partition_kind(
        void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(thunk != nullptr);
      return thunk->get_kind();
    }

    //--------------------------------------------------------------------------
    UniqueID DependentPartitionOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t DependentPartitionOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int DependentPartitionOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* DependentPartitionOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == nullptr)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const std::string_view& DependentPartitionOp::get_provenance_string(
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
    Mappable* DependentPartitionOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
      is_index_space = false;
      launch_space = nullptr;
      index_domain = Domain::NO_DOMAIN;
      parent_req_index = 0;
      thunk = nullptr;
      // can be changed for control rep
      mapper = nullptr;
      points_completed.store(0);
      points_committed = 0;
      commit_request = false;
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      copy_fill_priority = 0;
      intermediate_index_event = ApUserEvent::NO_AP_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (thunk != nullptr)
      {
        delete thunk;
        thunk = nullptr;
      }
      version_info.clear();
      map_applied_conditions.clear();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      // We deactivate all of our point operations
      for (PointDepPartOp* point : points) point->deactivate();
      points.clear();
      instances.clear();
      index_preconditions.clear();
      commit_preconditions.clear();
      profiling_requests.clear();
      if (mapper_data != nullptr)
      {
        free(mapper_data);
        mapper_data = nullptr;
        mapper_data_size = 0;
      }
      if (remove_launch_space_reference(launch_space))
        delete launch_space;
      Operation::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* DependentPartitionOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DEPENDENT_PARTITION_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind DependentPartitionOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return DEPENDENT_PARTITION_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t DependentPartitionOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::select_sources(
        const unsigned index, PhysicalManager* target,
        const std::vector<InstanceView*>& sources,
        std::vector<unsigned>& ranking,
        std::map<unsigned, PhysicalManager*>& points)
    //--------------------------------------------------------------------------
    {
      legion_assert(index == 0);
      Mapper::SelectPartitionSrcInput input;
      Mapper::SelectPartitionSrcOutput output;
      prepare_for_mapping(
          sources, input.source_instances, input.collective_views);
      prepare_for_mapping(target, input.target);
      if (mapper == nullptr)
      {
        Processor exec_proc = parent_ctx->get_executing_processor();
        mapper = runtime->find_mapper(exec_proc, map_id);
      }
      mapper->invoke_select_partition_sources(this, input, output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking, points);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*, unsigned>*
        DependentPartitionOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    int DependentPartitionOp::add_copy_profiling_request(
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
    bool DependentPartitionOp::handle_profiling_response(
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
      Mapping::Mapper::PartitionProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      info.total_reports = outstanding_profiling_requests;
      info.fill_response = op_info->fill;
      mapper->invoke_partition_report_profiling(this, info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
      legion_assert(count <= outstanding_profiling_requests);
      if (count == outstanding_profiling_requests)
        Runtime::trigger_event(profiling_reported);
      // Always record these as part of profiling
      return true;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
      legion_assert(count > 0);
      legion_assert(!mapped_event.has_triggered());
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_partition(rez, target);
      rez.serialize<PartitionKind>(get_partition_kind());
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
    size_t DependentPartitionOp::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      if (is_index_space)
        return get_shard_points()->get_volume();
      else
        return 1;
    }

    //--------------------------------------------------------------------------
    void DependentPartitionOp::find_open_complete_partitions(
        std::vector<LogicalPartition>& partitions) const
    //--------------------------------------------------------------------------
    {
      ContextID ctx = parent_ctx->get_logical_tree_context();
      legion_assert(requirement.handle_type == LEGION_SINGULAR_PROJECTION);
      RegionNode* region_node = runtime->get_node(requirement.region);
      FieldMask user_mask = region_node->column_source->get_field_mask(
          requirement.privilege_fields);
      region_node->find_open_complete_partitions(ctx, user_mask, partitions);
    }

    /////////////////////////////////////////////////////////////
    // Point Dependent Partition Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointDepPartOp::PointDepPartOp(void)
      : DependentPartitionOp(), owner(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PointDepPartOp::~PointDepPartOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void PointDepPartOp::initialize(
        DependentPartitionOp* own, const DomainPoint& p)
    //--------------------------------------------------------------------------
    {
      initialize_operation(own->get_context(), own->get_provenance());
      index_point = p;
      owner = own;
      context_index = owner->get_context_index();
      exception_handler = owner->get_exception_handler();
      index_domain = owner->index_domain;
      requirement = owner->requirement;
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
      version_info = owner->version_info;
      parent_req_index = owner->parent_req_index;
      LegionSpy::log_index_point(own->get_unique_op_id(), unique_op_id, p);
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::launch(void)
    //--------------------------------------------------------------------------
    {
      // Perform the version analysis for our point
      std::set<RtEvent> preconditions;
      perform_versioning_analysis(
          0 /*idx*/, requirement, version_info, preconditions);
      // Then put ourselves in the queue of operations ready to map
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::activate(void)
    //--------------------------------------------------------------------------
    {
      DependentPartitionOp::activate();
      // Reset this to true after it was cleared by the base call
      is_index_space = true;
      owner = nullptr;
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      DependentPartitionOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    ApEvent PointDepPartOp::trigger_thunk(
        IndexSpace handle, ApEvent inst_ready,
        const InstanceSet& mapped_instances,
        const PhysicalTraceInfo& trace_info, const DomainPoint& color)
    //--------------------------------------------------------------------------
    {
      return owner->trigger_thunk(
          handle, inst_ready, mapped_instances, trace_info, color);
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::trigger_complete(ApEvent effects)
    //--------------------------------------------------------------------------
    {
      owner->handle_point_complete(effects);
      complete_operation();
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      if (profiling_reported.exists())
        finalize_partition_profiling();
      // Don't commit this operation until we've reported our profiling
      // Out index owner will deactivate the operation
      commit_operation(false /*deactivate*/, profiling_reported);
      // Tell our owner that we are done, they will do the deactivate
      owner->handle_point_commit(profiling_reported);
    }

    //--------------------------------------------------------------------------
    size_t PointDepPartOp::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      return owner->get_collective_points();
    }

    //--------------------------------------------------------------------------
    bool PointDepPartOp::find_shard_participants(std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      return owner->find_shard_participants(shards);
    }

    //--------------------------------------------------------------------------
    Partition::PartitionKind PointDepPartOp::get_partition_kind(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(owner != nullptr);
      return owner->get_partition_kind();
    }

    //--------------------------------------------------------------------------
    const DomainPoint& PointDepPartOp::get_domain_point(void) const
    //--------------------------------------------------------------------------
    {
      return index_point;
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::set_projection_result(
        unsigned idx, LogicalRegion result)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx == 0);
      legion_assert(requirement.handle_type == LEGION_PARTITION_PROJECTION);
      requirement.region = result;
      requirement.handle_type = LEGION_SINGULAR_PROJECTION;
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::record_intra_space_dependences(
        unsigned index, const std::vector<DomainPoint>& dependences)
    //--------------------------------------------------------------------------
    {
      // Should never get here because our requirements are always read-only
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PointDepPartOp::record_pointwise_dependence(
        uint64_t previous_context_index, const DomainPoint& point,
        ShardID shard)
    //--------------------------------------------------------------------------
    {
      // Should never get here because we don't support pointwise analysis
      std::abort();
    }

    /////////////////////////////////////////////////////////////
    // Deppart Result Scatter
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeppartResultScatter::DeppartResultScatter(
        ReplicateContext* ctx, CollectiveID id, std::vector<DeppartResult>& res)
      : BroadcastCollective(ctx, id, 0 /*origin shard*/), results(res),
        done_event(Runtime::create_ap_user_event(nullptr))
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    DeppartResultScatter::~DeppartResultScatter(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void DeppartResultScatter::pack_collective(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(results.size());
      for (const DeppartResult& result : results)
      {
        rez.serialize(result.domain);
        rez.serialize(result.color);
      }
      rez.serialize<ApEvent>(done_event);
    }

    //--------------------------------------------------------------------------
    void DeppartResultScatter::unpack_collective(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      size_t num_results;
      derez.deserialize(num_results);
      results.resize(num_results);
      for (DeppartResult& result : results)
      {
        derez.deserialize(result.domain);
        derez.deserialize(result.color);
      }
      ApEvent done;
      derez.deserialize(done);
      Runtime::trigger_event_untraced(done_event, done);
    }

    //--------------------------------------------------------------------------
    void DeppartResultScatter::broadcast_results(ApEvent done)
    //--------------------------------------------------------------------------
    {
      Runtime::trigger_event_untraced(done_event, done);
      perform_collective_async();
    }

    /////////////////////////////////////////////////////////////
    // Field Descriptor Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldDescriptorExchange::FieldDescriptorExchange(
        ReplicateContext* ctx, CollectiveID id,
        std::vector<FieldDataDescriptor>& descs)
      : AllGatherCollective<true>(ctx, id), descriptors(descs)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    FieldDescriptorExchange::~FieldDescriptorExchange(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void FieldDescriptorExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(descriptors.size());
      for (const FieldDataDescriptor& descriptor : descriptors)
        descriptor.serialize(rez);
    }

    //--------------------------------------------------------------------------
    void FieldDescriptorExchange::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      // If this is stack -1 and we're not participating then we're
      // unpacking the full results back onto this node so overwrite
      // our current results, otherwise we can safely append
      const unsigned offset =
          ((stage < 0) && !participating) ? 0 : descriptors.size();
      size_t num_descriptors;
      derez.deserialize(num_descriptors);
      descriptors.resize(offset + num_descriptors);
      for (unsigned idx = 0; idx < num_descriptors; idx++)
        descriptors[offset + idx].deserialize(derez);
    }

    /////////////////////////////////////////////////////////////
    // Field Descriptor Gather
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldDescriptorGather::FieldDescriptorGather(
        ReplicateContext* ctx, CollectiveID id,
        std::vector<FieldDataDescriptor>& descs,
        std::map<DomainPoint, Domain>& targets)
      : GatherCollective(ctx, id, 0 /*origin shard*/), descriptors(descs),
        remote_targets(targets)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    FieldDescriptorGather::~FieldDescriptorGather(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void FieldDescriptorGather::pack_collective(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(descriptors.size());
      for (const FieldDataDescriptor& descriptor : descriptors)
        descriptor.serialize(rez);
      rez.serialize<size_t>(remote_targets.size());
      for (const std::pair<const DomainPoint, Domain>& it : remote_targets)
      {
        rez.serialize(it.first);
        rez.serialize(it.second);
      }
      if (!ready_events.empty())
        rez.serialize(Runtime::merge_events(nullptr, ready_events));
      else
        rez.serialize(ApEvent::NO_AP_EVENT);
    }

    //--------------------------------------------------------------------------
    void FieldDescriptorGather::unpack_collective(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      const unsigned offset = descriptors.size();
      size_t num_descriptors;
      derez.deserialize(num_descriptors);
      descriptors.resize(offset + num_descriptors);
      for (unsigned idx = 0; idx < num_descriptors; idx++)
        descriptors[offset + idx].deserialize(derez);
      size_t num_targets;
      derez.deserialize(num_targets);
      for (unsigned idx = 0; idx < num_targets; idx++)
      {
        DomainPoint point;
        derez.deserialize(point);
        derez.deserialize(remote_targets[point]);
      }
      ApEvent ready;
      derez.deserialize(ready);
      if (ready.exists())
        ready_events.emplace_back(ready);
    }

    //--------------------------------------------------------------------------
    void FieldDescriptorGather::contribute_instances(ApEvent ready)
    //--------------------------------------------------------------------------
    {
      if (ready.exists())
        ready_events.emplace_back(ready);
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    ApEvent FieldDescriptorGather::get_ready_event(void)
    //--------------------------------------------------------------------------
    {
      if (ready_events.empty())
        return ApEvent::NO_AP_EVENT;
      else
        return Runtime::merge_events(nullptr, ready_events);
    }

    /////////////////////////////////////////////////////////////
    // Repl Dependent Partition Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplDependentPartitionOp::ReplDependentPartitionOp(void)
      : ReplCollectiveViewCreator<
            CollectiveViewCreator<DependentPartitionOp> >()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplDependentPartitionOp::~ReplDependentPartitionOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::initialize_replication(ReplicateContext* ctx)
    //--------------------------------------------------------------------------
    {
      mapping_barrier = ctx->get_next_dependent_partition_mapping_barrier();
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveViewCreator<
          CollectiveViewCreator<DependentPartitionOp> >::activate();
      sharding_function = nullptr;
      shard_points = nullptr;
      gather = nullptr;
      scatter = nullptr;
      exchange = nullptr;
      collective_ready = ApBarrier::NO_AP_BARRIER;
      collective_done = ApBarrier::NO_AP_BARRIER;
      sharding_collective = nullptr;
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (gather != nullptr)
        delete gather;
      if (scatter != nullptr)
        delete scatter;
      if (exchange != nullptr)
        delete exchange;
      remote_targets.clear();
      deppart_results.clear();
      if (sharding_collective != nullptr)
        delete sharding_collective;
      remove_launch_space_reference(shard_points);
      ReplCollectiveViewCreator<CollectiveViewCreator<DependentPartitionOp> >::
          deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::select_sharding_function(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      legion_assert(repl_ctx != nullptr);
      legion_assert(sharding_function == nullptr);
      // Do the mapper call to get the sharding function to use
      if (mapper == nullptr)
        mapper =
            runtime->find_mapper(parent_ctx->get_executing_processor(), map_id);
      Mapper::SelectShardingFunctorInput* input = repl_ctx->shard_manager;
      Mapper::SelectShardingFunctorOutput output;
      mapper->invoke_partition_select_sharding_functor(this, *input, output);
      if (output.chosen_functor == std::numeric_limits<ShardingID>::max())
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Mapper " << mapper->get_mapper_name()
              << " failed to pick a valid sharding functor for "
              << "dependent partition in task " << parent_ctx->get_task_name()
              << " (UID " << parent_ctx->get_unique_id() << ").";
        error.raise();
      }
      sharding_function = repl_ctx->shard_manager->find_sharding_function(
          output.chosen_functor);
      if (runtime->safe_mapper)
      {
        legion_assert(sharding_collective != nullptr);
        sharding_collective->contribute(output.chosen_functor);
        if (sharding_collective->is_target() &&
            !sharding_collective->validate(output.chosen_functor))
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Mapper " << mapper->get_mapper_name()
                << " chose different sharding functions "
                << "for dependent partition op in task "
                << parent_ctx->get_task_name() << " (UID "
                << parent_ctx->get_unique_id() << ").";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::select_partition_projection(void)
    //--------------------------------------------------------------------------
    {
      if (thunk->is_image() || !runtime->safe_mapper)
        DependentPartitionOp::select_partition_projection();
      else
      {
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(parent_ctx);
        legion_assert(sharding_function == nullptr);
        // Check here that all the shards pick the same partition
        requirement.partition = LogicalPartition::NO_PART;
        DependentPartitionOp::select_partition_projection();
        ValueBroadcast<LogicalPartition> part_check(
            repl_ctx->get_next_collective_index(
                COLLECTIVE_LOC_22, true /*logical*/),
            repl_ctx, 0 /*origin shard*/);
        if (repl_ctx->owner_shard->shard_id > 0)
        {
          const LogicalPartition chosen_part = part_check.get_value();
          if (chosen_part != requirement.partition)
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error << "Invalid mapper output from invocation of "
                     "'select_partition_projection' on mapper "
                  << mapper->get_mapper_name()
                  << " for depedent partitioning operation launched in "
                  << parent_ctx->get_task_name() << " (UID "
                  << parent_ctx->get_unique_id()
                  << "). Mapper selected a logical partition "
                     "on shard "
                  << repl_ctx->owner_shard->shard_id
                  << " that is different than the logical "
                     "partition selected by shard 0. All shards must "
                     "select the same logical partition.";
            error.raise();
          }
        }
        else
          part_check.broadcast(requirement.partition);
      }
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Before doing the dependence analysis we have to ask the
      // mapper whether it would like to make this an index space
      // operation or a single operation
      select_partition_projection();
      // Do thise now that we've picked our region requirement
      log_requirement();
      if (is_index_space)
      {
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(parent_ctx);
        // Now that we know that we have the right region requirement we
        // can ask the mapper to also pick the sharding function
        select_sharding_function();
        // We can also initialize the barriers and exchange we will need
        if (thunk->is_image())
        {
          exchange = new FieldDescriptorExchange(
              repl_ctx,
              repl_ctx->get_next_collective_index(
                  COLLECTIVE_LOC_30, true /*logical*/),
              instances);
          collective_ready =
              repl_ctx->get_next_dependent_partition_execution_barrier();
          collective_done =
              repl_ctx->get_next_dependent_partition_execution_barrier();
        }
        else
        {
          gather = new FieldDescriptorGather(
              repl_ctx,
              repl_ctx->get_next_collective_index(
                  COLLECTIVE_LOC_61, true /*logical*/),
              instances, remote_targets);
          scatter = new DeppartResultScatter(
              repl_ctx,
              repl_ctx->get_next_collective_index(
                  COLLECTIVE_LOC_62, true /*logical*/),
              deppart_results);
        }
      }
      else
      {
        create_collective_rendezvous(requirement.parent.get_tree_id(), 0);
        if (sharding_collective != nullptr)
          sharding_collective->elide_collective();
      }
      analyze_region_requirements(
          is_index_space ? launch_space : nullptr,
          is_index_space ? sharding_function : nullptr);

      // Record this dependent partition op with the context so that it
      // can track implicit dependences on it for later operations
      parent_ctx->update_current_implicit_creation(this);
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      // Do different things if this is an index space point or a single point
      if (is_index_space)
      {
        legion_assert(sharding_function != nullptr);
        legion_assert(points_completed.load() == 0);
        // This is a bit tricky, but set the points_completed to be -1 as
        // a guard so that we don't call complete_execution until we've see
        // all our completed points and had a chance to run trigger_execution
        // for ourselves as they both need to be done before finishing execution
        points_completed.store(-1);
        // Compute the local index space of points for this shard
        IndexSpace local_space = sharding_function->find_shard_space(
            repl_ctx->owner_shard->shard_id, launch_space, launch_space->handle,
            get_provenance());
        // If it's empty we're done, otherwise we go back on the queue
        if (!local_space.exists())
        {
          // Still have to do this for legion spy
          LegionSpy::log_operation_events(
              unique_op_id, ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
          // We have no local points, so we're done mapping
          finalize_mapping();
          RtEvent ready;
          if (thunk->is_image())
          {
            legion_assert(exchange != nullptr);
            // We won't have any preconditions on the collective ready event
            runtime->phase_barrier_arrive(collective_ready, 1 /*count*/);
            // Perform the exchange of the instance data and then
            // trigger execution when it is ready
            exchange->perform_collective_async();
            ready = exchange->get_done_event();
          }
          else
          {
            legion_assert(gather != nullptr);
            legion_assert(scatter != nullptr);
            std::vector<ApEvent> preconditions;
            if (thunk->is_preimage())
            {
              ApUserEvent to_trigger;
              find_remote_targets(preconditions, to_trigger);
              if (to_trigger.exists())
                Runtime::trigger_event_untraced(
                    to_trigger, scatter->get_done_event());
            }
            if (preconditions.empty())
              gather->contribute_instances(ApEvent::NO_AP_EVENT);
            else
              gather->contribute_instances(
                  Runtime::merge_events(nullptr, preconditions));
            if (gather->target == repl_ctx->owner_shard->shard_id)
              ready = gather->perform_collective_wait(false /*block*/);
            else
              ready = scatter->perform_collective_wait(false /*block*/);
          }
          legion_assert(ready.exists());
          parent_ctx->add_to_trigger_execution_queue(this, ready);
        }
        else  // If we have valid points then we do the base call
        {
          shard_points = runtime->get_node(local_space);
          add_launch_space_reference(shard_points);
          DependentPartitionOp::trigger_ready();
        }
      }
      else
      {
        // In this case we're all going to map the source instance
        // and then perform the partition creation collective
        std::set<RtEvent> preconditions;
        // Path for a non-index space implementation
        perform_versioning_analysis(
            0 /*idx*/, requirement, version_info, preconditions,
            nullptr /*output region*/, true /*rendezvous*/);
        // Give these operations slightly higher priority since
        // they are likely needed for other operations
        if (!preconditions.empty())
          enqueue_ready_operation(
              Runtime::merge_events(preconditions),
              LG_THROUGHPUT_DEFERRED_PRIORITY);
        else
          enqueue_ready_operation(
              RtEvent::NO_RT_EVENT, LG_THROUGHPUT_DEFERRED_PRIORITY);
      }
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::finalize_mapping(void)
    //--------------------------------------------------------------------------
    {
      RtEvent precondition;
      if (!map_applied_conditions.empty())
        precondition = Runtime::merge_events(map_applied_conditions);
      runtime->phase_barrier_arrive(mapping_barrier, 1 /*count*/, precondition);
      if (!acquired_instances.empty())
        precondition = release_nonempty_acquired_instances(
            mapping_barrier, acquired_instances);
      else
        precondition = mapping_barrier;
      complete_mapping(precondition);
    }

    //--------------------------------------------------------------------------
    ApEvent ReplDependentPartitionOp::trigger_thunk(
        IndexSpace handle, ApEvent instances_ready,
        const InstanceSet& mapped_insts, const PhysicalTraceInfo& info,
        const DomainPoint& color)
    //--------------------------------------------------------------------------
    {
      legion_assert(mapped_insts.size() == 1);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      if (is_index_space)
      {
        IndexSpaceNode* node = runtime->get_node(handle);
        Domain domain;
        ApUserEvent to_trigger;
        ApEvent domain_ready = node->get_loose_domain(domain, to_trigger);
        bool ready = false;
        {
          AutoLock o_lock(op_lock);
          instances.resize(instances.size() + 1);
          FieldDataDescriptor& desc = instances.back();
          const InstanceRef& ref = mapped_insts[0];
          PhysicalManager* manager = ref.get_physical_manager();
          desc.inst = manager->get_instance();
          desc.domain = domain;
          desc.color = color;
          desc.unique = manager->get_unique_event();
          desc.space = handle.get_id(false /*filter*/);
          if (instances_ready.exists())
            index_preconditions.emplace_back(instances_ready);
          if (domain_ready.exists())
            index_preconditions.emplace_back(domain_ready);
          legion_assert(!points.empty());
          ready = (instances.size() == points.size());
        }
        if (ready)
        {
          if (thunk->is_image())
          {
            // Images can be sharded so we do them in parallel and
            // exchange the field descriptors with all the shards
            // Get the exchange in flight
            exchange->perform_collective_async();
            // Arrive on the ready barrier
            if (index_preconditions.empty())
              runtime->phase_barrier_arrive(collective_ready, 1 /*count*/);
            else
              runtime->phase_barrier_arrive(
                  collective_ready, 1 /*count*/,
                  Runtime::merge_events(&info, index_preconditions));
            const RtEvent exchanged =
                exchange->perform_collective_wait(false /*block*/);
            if (exchanged.exists() && !exchanged.has_triggered())
              parent_ctx->add_to_trigger_execution_queue(this, exchanged);
            else
              trigger_execution();
          }
          else
          {
            // For all other dependent partition operations we gather all
            // the field descriptors to one node to perform the computation
            // and then we scatter them all back out to the targets after
            // we've computed them on one node. We do this because Realm can
            // perform non-trivial optimizations for partition-by-field and
            // partition-by-preimage for those cases when it sees a single call
            if (thunk->is_preimage())
              find_remote_targets(index_preconditions, to_trigger);
            if (index_preconditions.empty())
              gather->contribute_instances(ApEvent::NO_AP_EVENT);
            else
              gather->contribute_instances(
                  Runtime::merge_events(&info, index_preconditions));
            if (gather->target == repl_ctx->owner_shard->shard_id)
            {
              const RtEvent gathered =
                  gather->perform_collective_wait(false /*block*/);
              if (gathered.exists() && !gathered.has_triggered())
                parent_ctx->add_to_trigger_execution_queue(this, gathered);
              else
                trigger_execution();
            }
            else
            {
              const RtEvent scattered =
                  scatter->perform_collective_wait(false /*block*/);
              if (scattered.exists() && !scattered.has_triggered())
                parent_ctx->add_to_trigger_execution_queue(this, scattered);
              else
                trigger_execution();
            }
          }
        }
        if (thunk->is_image())
        {
          if (to_trigger.exists())
            Runtime::trigger_event_untraced(to_trigger, collective_done);
          return collective_done;
        }
        else
        {
          const ApEvent result = scatter->get_done_event();
          if (to_trigger.exists())
            Runtime::trigger_event_untraced(to_trigger, result);
          return result;
        }
      }
      else
      {
        // Only need to perform this if we're the first local shard
        if (repl_ctx->shard_manager->is_first_local_shard(
                repl_ctx->owner_shard))
          return DependentPartitionOp::trigger_thunk(
              handle, instances_ready, mapped_insts, info, color);
        else
          return ApEvent::NO_AP_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_index_space);
      legion_assert(requirement.privilege_fields.size() == 1);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      legion_assert(-1 <= points_completed.load());
      // Check to see if we're the first shard in this address space
      const bool first_local_shard =
          repl_ctx->shard_manager->is_first_local_shard(repl_ctx->owner_shard);
      if (thunk->is_image())
      {
        ApEvent done_event;
        if (first_local_shard)
        {
          const FieldID fid = *(requirement.privilege_fields.begin());
          done_event = thunk->perform(this, fid, collective_ready, instances);
        }
        runtime->phase_barrier_arrive(collective_done, 1 /*count*/, done_event);
      }
      else
      {
        if (gather->target == repl_ctx->owner_shard->shard_id)
        {
          legion_assert(first_local_shard);
          legion_assert(scatter->origin == gather->target);
          const FieldID fid = *(requirement.privilege_fields.begin());
          ApEvent done_event = thunk->perform(
              this, fid, gather->get_ready_event(), instances, &remote_targets,
              &deppart_results);
          scatter->broadcast_results(done_event);
        }
        else if (first_local_shard)
        {
          const FieldID fid = *(requirement.privilege_fields.begin());
          thunk->perform(
              this, fid, scatter->get_done_event(), instances, &remote_targets,
              &deppart_results);
        }
      }
      // Remove our guard that we added in trigger_ready and see if we're done
      const unsigned received = ++points_completed;
      legion_assert(received <= points.size());
      if (received == points.size())
        complete_execution();
    }

    //--------------------------------------------------------------------------
    bool ReplDependentPartitionOp::find_shard_participants(
        std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_index_space);
      legion_assert(sharding_function != nullptr);
      return sharding_function->find_shard_participants(
          launch_space, launch_space->handle, shards);
    }

    //--------------------------------------------------------------------------
    bool ReplDependentPartitionOp::perform_collective_analysis(
        CollectiveMapping*& mapping, bool& first_local)
    //--------------------------------------------------------------------------
    {
      // If we're not an index space launch then we know all the shards
      // are going to be using the same region so we can do a collective
      // rendezvous to create a collective view
      return !is_index_space;
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::find_remote_targets(
        std::vector<ApEvent>& preconditions, ApUserEvent& to_trigger)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* node = runtime->get_node(thunk->get_projection());
      if (node->is_owner() ||
          ((node->collective_mapping != nullptr) &&
           node->collective_mapping->contains(node->local_space)))
      {
        for (ColorSpaceIterator itr(node, true /*local only*/); itr; itr++)
        {
          DomainPoint color =
              node->color_space->delinearize_color_to_point(*itr);
          IndexSpaceNode* child = node->get_child(*itr);
          ApEvent ready =
              child->get_loose_domain(remote_targets[color], to_trigger);
          if (ready.exists())
            preconditions.emplace_back(ready);
        }
      }
    }

    //--------------------------------------------------------------------------
    RtEvent ReplDependentPartitionOp::perform_collective_versioning_analysis(
        unsigned index, LogicalRegion handle, EqSetTracker* tracker,
        const FieldMask& mask, unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      return rendezvous_collective_versioning_analysis(
          index, handle, tracker, runtime->address_space, mask,
          parent_req_index);
    }

    /////////////////////////////////////////////////////////////
    // Remote Partition Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemotePartitionOp::RemotePartitionOp(Operation* ptr, AddressSpaceID src)
      : ExternalPartition(), RemoteOp(ptr, src)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RemotePartitionOp::~RemotePartitionOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID RemotePartitionOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t RemotePartitionOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemotePartitionOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemotePartitionOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* RemotePartitionOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == nullptr)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const std::string_view& RemotePartitionOp::get_provenance_string(
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
    Partition::PartitionKind RemotePartitionOp::get_partition_kind(void) const
    //--------------------------------------------------------------------------
    {
      return part_kind;
    }

    //--------------------------------------------------------------------------
    const char* RemotePartitionOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DEPENDENT_PARTITION_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind RemotePartitionOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return DEPENDENT_PARTITION_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemotePartitionOp::select_sources(
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
      Mapper::SelectPartitionSrcInput input;
      Mapper::SelectPartitionSrcOutput output;
      prepare_for_mapping(
          sources, input.source_instances, input.collective_views);
      prepare_for_mapping(target, input.target);
      if (mapper == nullptr)
        mapper = runtime->find_mapper(map_id);
      mapper->invoke_select_partition_sources(this, input, output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking, points);
    }

    //--------------------------------------------------------------------------
    void RemotePartitionOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_partition(rez, target);
      rez.serialize(part_kind);
      pack_profiling_requests(rez, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemotePartitionOp::unpack(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      unpack_external_partition(derez);
      derez.deserialize(part_kind);
      unpack_profiling_requests(derez);
    }

  }  // namespace Internal
}  // namespace Legion
