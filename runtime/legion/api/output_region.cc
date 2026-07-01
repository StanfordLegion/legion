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

#include "legion/api/output_region_impl.h"
#include "legion/api/data.h"
#include "legion/contexts/context.h"
#include "legion/kernel/runtime.h"
#include "legion/instances/physical.h"
#include "legion/nodes/region.h"
#include "legion/tasks/single.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // Output Region
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  OutputRegion::OutputRegion(void) : impl(nullptr)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  OutputRegion::OutputRegion(const OutputRegion& rhs) : impl(rhs.impl)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  OutputRegion::OutputRegion(Internal::OutputRegionImpl* i) : impl(i)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  OutputRegion::~OutputRegion(void)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
    {
      if (impl->remove_reference())
        delete impl;
      impl = nullptr;
    }
  }

  //--------------------------------------------------------------------------
  OutputRegion& OutputRegion::operator=(const OutputRegion& rhs)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
    {
      if (impl->remove_reference())
        delete impl;
    }
    impl = rhs.impl;
    if (impl != nullptr)
      impl->add_reference();
    return *this;
  }

  //--------------------------------------------------------------------------
  Memory OutputRegion::target_memory(void) const
  //--------------------------------------------------------------------------
  {
    return impl->target_memory();
  }

  //--------------------------------------------------------------------------
  LogicalRegion OutputRegion::get_logical_region(void) const
  //--------------------------------------------------------------------------
  {
    return impl->get_logical_region();
  }

  //--------------------------------------------------------------------------
  bool OutputRegion::is_bounded_output_region(void) const
  //--------------------------------------------------------------------------
  {
    return impl->is_bounded_output_region();
  }

  //--------------------------------------------------------------------------
  void OutputRegion::check_type_tag(TypeTag type_tag) const
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    impl->check_type_tag(type_tag);
  }

  //--------------------------------------------------------------------------
  void OutputRegion::check_field_size(FieldID field_id, size_t field_size) const
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    impl->check_field_size(field_id, field_size);
  }

  //--------------------------------------------------------------------------
  void OutputRegion::get_layout(
      FieldID field_id, std::vector<DimensionKind>& ordering,
      size_t& alignment) const
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    impl->get_layout(field_id, ordering, alignment);
  }

  //--------------------------------------------------------------------------
  void OutputRegion::return_data(
      const DomainPoint& extents, FieldID field_id,
      Realm::RegionInstance instance, bool check_constraints /*= true */)
  //--------------------------------------------------------------------------
  {
    return_data(extents, field_id, instance, nullptr, check_constraints);
  }

  //--------------------------------------------------------------------------
  void OutputRegion::return_data(
      const DomainPoint& extents, FieldID field_id,
      Realm::RegionInstance instance, const LayoutConstraintSet* constraints,
      bool check_constraints)
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    impl->return_data(
        extents, field_id, instance, constraints, check_constraints);
  }

  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Output Region Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OutputRegionImpl::OutputRegionImpl(
        unsigned idx, const OutputRequirement& r, const InstanceSet& instances,
        TaskContext* ctx, const bool global, const bool bounded,
        const bool grouped)
      : Collectable(), context(ctx), req(r),
        region(runtime->get_node(req.region)), index(idx),
        created_region(
            (req.flags & LEGION_CREATED_OUTPUT_REQUIREMENT_FLAG) && !bounded),
        global_indexing(global), grouped_fields(grouped)
    //--------------------------------------------------------------------------
    {
      region->add_base_gc_ref(OUTPUT_REGION_REF);
      context->add_base_gc_ref(OUTPUT_REGION_REF);

      managers.resize(req.instance_fields.size(), nullptr);
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const InstanceRef& ref = instances[idx];
        std::vector<FieldID> fields;
        region->column_source->get_field_set(
            ref.get_valid_fields(), context, fields);
        PhysicalManager* manager = ref.get_physical_manager();
        for (const FieldID& field : fields)
        {
          std::vector<FieldID>::const_iterator finder = std::find(
              req.instance_fields.begin(), req.instance_fields.end(), field);
          legion_assert(finder != req.instance_fields.end());
          const unsigned offset =
              std::distance(req.instance_fields.begin(), finder);
          legion_assert(offset < managers.size());
          legion_assert(managers[offset] == nullptr);
          managers[offset] = manager;
          manager->add_base_gc_ref(OUTPUT_REGION_REF);
        }
      }
    }

    //--------------------------------------------------------------------------
    OutputRegionImpl::~OutputRegionImpl(void)
    //--------------------------------------------------------------------------
    {
      if (region->remove_base_gc_ref(OUTPUT_REGION_REF))
        delete region;
      if (context->remove_base_gc_ref(OUTPUT_REGION_REF))
        delete context;
      for (PhysicalManager* const & manager : managers)
        if (manager->remove_base_gc_ref(OUTPUT_REGION_REF))
          delete manager;
    }

    //--------------------------------------------------------------------------
    Memory OutputRegionImpl::target_memory(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(!managers.empty());
      return managers.front()->get_memory();
    }

    //--------------------------------------------------------------------------
    LogicalRegion OutputRegionImpl::get_logical_region(void) const
    //--------------------------------------------------------------------------
    {
      if (!is_bounded_output_region())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Logical region cannot be retrieved from output region "
              << index << " of " << *context << " whose index space "
              << "is yet to be determined.";
        error.raise();
      }
      return req.region;
    }

    //--------------------------------------------------------------------------
    bool OutputRegionImpl::is_bounded_output_region(void) const
    //--------------------------------------------------------------------------
    {
      return !created_region;
    }

    //--------------------------------------------------------------------------
    void OutputRegionImpl::check_type_tag(TypeTag type_tag) const
    //--------------------------------------------------------------------------
    {
      if (type_tag == req.type_tag)
        return;

      Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
      error << "The deferred buffer passed to output region " << index << " of "
            << *context << " is incompatible with the "
            << "output region. Make sure the deferred buffer "
            << "has the right dimension and coordinate type.";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void OutputRegionImpl::check_field_size(
        FieldID field_id, size_t field_size) const
    //--------------------------------------------------------------------------
    {
      size_t impl_field_size = get_field_size(field_id);
      if (field_size == impl_field_size)
        return;

      Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
      error << "The deferred buffer passed to field " << field_id
            << " of output region " << index << " of " << *context
            << " has elements of " << field_size
            << " bytes each, but the field "
            << "size is " << impl_field_size << " bytes. Make sure you pass a "
            << "buffer of the right element type.";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void OutputRegionImpl::get_layout(
        FieldID field_id, std::vector<DimensionKind>& ordering,
        size_t& alignment) const
    //--------------------------------------------------------------------------
    {
      PhysicalManager* manager = get_manager(field_id);
      LayoutConstraints* cons = manager->layout->constraints;

      legion_assert(cons->ordering_constraint.ordering.size() > 1);
      legion_assert(cons->ordering_constraint.ordering.back() == LEGION_DIM_F);
      int32_t ndim = NT_TemplateHelper::get_dim(req.type_tag);
      DimensionKind max_dim =
          static_cast<DimensionKind>(static_cast<int32_t>(LEGION_DIM_X) + ndim);
      ordering.resize(ndim);
      uint32_t idx = 0;
      for (const DimensionKind& kind : cons->ordering_constraint.ordering)
      {
        if (kind < max_dim)
          ordering[idx++] = kind;
      }

      for (const AlignmentConstraint& constraint : cons->alignment_constraints)
      {
        if ((constraint.fid == field_id) && (constraint.eqk == LEGION_EQ_EK))
        {
          alignment = constraint.alignment;
          return;
        }
      }

      // If no alignment constraint was given, use the field size
      // for alignment
      alignment = get_field_size(field_id);
    }

    //--------------------------------------------------------------------------
    size_t OutputRegionImpl::get_field_size(FieldID field_id) const
    //--------------------------------------------------------------------------
    {
      std::vector<FieldID>::const_iterator finder = std::find(
          req.instance_fields.begin(), req.instance_fields.end(), field_id);
      if (finder == req.instance_fields.end())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Field " << field_id << " does not exist in output region "
              << index << " of " << *context << ".";
        error.raise();
      }
      return region->column_source->get_field_size(field_id);
    }

    //--------------------------------------------------------------------------
    void OutputRegionImpl::return_data(
        const DomainPoint& new_extents, FieldID field_id,
        PhysicalInstance instance, const LayoutConstraintSet* constraints,
        bool check_constraints)
    //--------------------------------------------------------------------------
    {
      if (req.privilege_fields.find(field_id) == req.privilege_fields.end())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Output region " << index << " of " << *context
              << " does not have privilege on field " << field_id << ".";
        error.raise();
      }
      if (returned_instances.find(field_id) != returned_instances.end())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Data has already been set to field " << field_id
              << " of output region " << index << " of " << *context
              << ". You can only return data for each field of an output "
              << "region once.";
        error.raise();
      }
      PhysicalManager* manager = get_manager(field_id);
      const LayoutConstraints* manager_cons = manager->layout->constraints;
      if (check_constraints && (constraints != nullptr))
      {
        bool has_conflict = false;
        if (!req.global_indexing && context->owner_task->is_index_space)
        {
          // Unfortunately, for local indexing, the ordering constraint
          // prescribes the ordering of dimensions that the returned
          // buffer does not contain. (those dimensions are added
          // by the runtime and invisible to the point task.) So, here
          // we filter out the dimensions that would otherwise fail
          // the constraint check innocuously.
          LayoutConstraintSet copied;
          copied.alignment_constraints = manager_cons->alignment_constraints;
          std::vector<DimensionKind> ordering;
          int32_t ndim = NT_TemplateHelper::get_dim(req.type_tag);
          for (const DimensionKind& kind :
               manager_cons->ordering_constraint.ordering)
          {
            int32_t dim = kind;
            if (((dim - LEGION_DIM_X) < ndim) || (dim == LEGION_DIM_F))
              ordering.emplace_back(static_cast<DimensionKind>(dim));
          }
          copied.ordering_constraint = OrderingConstraint(
              ordering, manager_cons->ordering_constraint.contiguous);

          has_conflict = constraints->conflicts(copied);
        }
        else
          has_conflict = constraints->conflicts(*manager_cons);

        if (has_conflict)
        {
          Fatal fatal;
          fatal
              << "The returned instance for field " << field_id
              << " of output region " << index << " of " << *context
              << " does not satisfy the layout constraints "
              << "chosen by the mapper. This is an illegal usage right now. "
              << "In the future, the runtime will copy this returned instance "
              << "into a fresh one with the correct layout.";
          fatal.raise();
        }
      }
      else if (check_constraints)
      {
        Fatal fatal;
        fatal
            << "Currently the constraint checks need to be turned off to "
            << "pass naked instances to output regions. In the future, layout "
            << "constraints will be inferred from the instances and used for "
            << "the checks.";
        fatal.raise();
      }
      if (extents.dim == 0)
      {
        extents = new_extents;
        if (created_region)
        {
          // We now know the extent so we can report it back
          // Set the result if we're not doing global indexing with a parent
          if (region->parent == nullptr)
          {
            DomainPoint lo;
            lo.dim = extents.dim;
            Domain domain(lo, extents - 1);
            if (region->row_source->set_domain(
                    domain, ApEvent::NO_AP_EVENT, true /*take ownership*/,
                    true /*broadcast*/))
              std::abort();  // should never end up deleting this
          }
          else
          {
            DomainPoint color = region->row_source->get_domain_point_color();
            if (!global_indexing)
            {
              Domain domain;
              domain.dim = color.dim + extents.dim;
              legion_assert(domain.dim <= LEGION_MAX_DIM);
              for (int idx = 0; idx < color.dim; ++idx)
              {
                domain.rect_data[idx] = color[idx];
                domain.rect_data[idx + domain.dim] = color[idx];
              }
              for (int idx = 0; idx < extents.dim; ++idx)
              {
                int off = color.dim + idx;
                domain.rect_data[off] = 0;
                domain.rect_data[domain.dim + off] = extents[idx] - 1;
              }
              if (region->row_source->set_domain(
                      domain, ApEvent::NO_AP_EVENT, true /*take ownership*/,
                      true /*broadcast*/))
                std::abort();  // should never end up deleting this
            }
            context->owner_task->record_output_extent(index, color, extents);
          }
        }
      }
      else if (new_extents != extents)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Output region " << index << " of " << *context
              << " has already been initialized to extents " << extents
              << ", but the new output has extents " << new_extents
              << ". You must return data having the same extents to all the "
              << "fields in the same output region.";
        error.raise();
      }
      if (instance.exists() &&
          (instance.get_location() != manager->get_memory()))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Field " << field_id << " of output region " << index << " of "
              << *context << " is requested "
              << "to have an instance in " << manager->get_memory().kind()
              << " memory " << manager->get_memory()
              << ", but the returned instance is allocated in "
              << instance.get_location().kind() << " memory "
              << instance.get_location() << ".";
        error.raise();
      }
      if (grouped_fields && !returned_instances.empty())
      {
        // Make sure that all the fields have the same instance
        if (instance != returned_instances.begin()->second)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Instance passed to field " << field_id
                << " of output region " << index << " of " << *context
                << " is not the same as previous instance returned "
                << "for this output region requirement. The layout constraints "
                << "for this output region suggested that the fields need to "
                << "be grouped into one instance and therefore they must all "
                << "be in the same instance.";
          error.raise();
        }
      }
      returned_instances.emplace(std::make_pair(field_id, instance));
    }

    //--------------------------------------------------------------------------
    RtEvent OutputRegionImpl::finalize(RtEvent safe_effects, bool escaped)
    //--------------------------------------------------------------------------
    {
      // If this is the first invocation, check to see if the index space
      // is ready for the region. If not then we escape the instances
      // and then defer until the bounds for the region are ready at
      // which point we can actually finalize the output regions
      if (!escaped)
      {
        RtEvent precondition = region->row_source->get_ready_event();
        if (precondition.exists() && !precondition.has_triggered())
        {
          // Escape each of the returned instances before we return
          // since the calling context can release the pool these
          // instances are associated with as soon as we return
          // and we need to keep these instances alive beyond then
          std::vector<RtEvent> preconditions;
          for (std::pair<const FieldID, PhysicalInstance>& instance :
               returned_instances)
          {
            if (!instance.second.exists())
              continue;
            // We don't care what the unique event is here because this
            // instance only needs to live until we redistrict it again
            // to the actual bounds
            LgEvent unique_event;
            const RtEvent ready = context->escape_task_local_instance(
                instance.second, safe_effects, 1 /*count*/, &instance.second,
                &unique_event);
            if (ready.exists())
              preconditions.push_back(ready);
          }
          // Now we can defer this finalize
          if (!preconditions.empty())
          {
            preconditions.push_back(precondition);
            precondition = Runtime::merge_events(preconditions);
          }
          if (precondition.exists() && !precondition.has_triggered())
          {
            // Launch a continuation to actually finalize this output region
            FinalizeOutputRegionArgs args(this, safe_effects);
            return runtime->issue_runtime_meta_task(
                args, LG_LATENCY_WORK_PRIORITY, precondition);
          }
          else
            escaped = true;
        }
      }

      // Transpose the returned instances
      std::map<PhysicalInstance, std::vector<FieldID> > instance_fields;
      for (const std::pair<const FieldID, PhysicalInstance>& inst :
           returned_instances)
        instance_fields[inst.second].emplace_back(inst.first);

      for (const std::pair<const PhysicalInstance, std::vector<FieldID> >& pit :
           instance_fields)
      {
        const Realm::InstanceLayoutGeneric* current =
            pit.first.exists() ? pit.first.get_layout() : nullptr;
        if (grouped_fields)
        {
          // Redistricting to just one layout for all the fields
          std::vector<size_t> sizes(pit.second.size());
          for (unsigned idx = 0; idx < pit.second.size(); idx++)
            sizes[idx] = get_field_size(pit.second[idx]);

          PhysicalManager* manager = get_manager(pit.second.back());
          const LayoutConstraints* constraints = manager->layout->constraints;

          Realm::InstanceLayoutGeneric* layout =
              region->row_source->create_layout(
                  *constraints, pit.second, sizes, false /*compact*/, nullptr,
                  nullptr, nullptr,
                  (current != nullptr) ? current->alignment_reqd : 1);
          legion_assert(
              (current == nullptr) ||
              (layout->bytes_used <= current->bytes_used));
          legion_assert(
              (current == nullptr) ||
              (layout->alignment_reqd == current->alignment_reqd));
          PhysicalInstance instance = pit.first;
          const size_t footprint = layout->bytes_used;
          if (instance.exists())
          {
            LgEvent unique_event = manager->get_unique_event();
            const RtEvent ready = context->escape_task_local_instance(
                instance, safe_effects, 1 /*count*/, &instance, &unique_event,
                (const Realm::InstanceLayoutGeneric**)&layout, escaped);
            if (manager->update_physical_instance(instance, ready, footprint))
              delete manager;
          }
          else
          {
            // We don't have an existing instance so we need to make one
            Realm::ProfilingRequestSet requests;
            if (runtime->profiler != nullptr)
            {
              const LgEvent unique_event = manager->get_unique_event();
              legion_assert(unique_event.exists());
              runtime->profiler->add_inst_request(
                  requests, context->get_unique_id(), unique_event);
            }
            const RtEvent ready(Realm::RegionInstance::create_instance(
                instance, manager->memory_manager->memory, *layout, requests));
            if (ready.exists() && (implicit_profiler != nullptr))
              implicit_profiler->record_instance_ready(
                  ready, manager->get_unique_event());
            if (manager->update_physical_instance(instance, ready, footprint))
              delete manager;
          }
          delete layout;
        }
        else if (pit.first.exists())
        {
          // Use redistricting to make N instances for each manager
          std::vector<Realm::InstanceLayoutGeneric*> layouts(pit.second.size());
          std::vector<LgEvent> unique_events(pit.second.size());
          std::vector<PhysicalManager*> managers(pit.second.size());
          for (unsigned idx = 0; idx < layouts.size(); idx++)
          {
            const FieldID field_id = pit.second[idx];
            PhysicalManager* manager = get_manager(field_id);
            managers[idx] = manager;
            // Create a layout to use for redistricting
            const LayoutConstraints* constraints = manager->layout->constraints;

            const std::vector<FieldID> fields(1, field_id);
            const std::vector<size_t> sizes(1, get_field_size(field_id));

            // Base alignment of 1 if not specified
            size_t alignment = 1;
            if (!constraints->alignment_constraints.empty())
            {
              // Should only be one alignment constraint at most since there
              // should just be one field for this manager
              legion_assert(constraints->alignment_constraints.size() == 1);
              const AlignmentConstraint& constraint =
                  constraints->alignment_constraints.front();
              legion_assert(constraint.fid == field_id);
              alignment = constraint.alignment;
            }
            layouts[idx] = region->row_source->create_layout(
                *constraints, fields, sizes, false /*compact*/, nullptr,
                nullptr, nullptr, alignment);
            unique_events[idx] = manager->get_unique_event();
          }
          std::vector<PhysicalInstance> instances(layouts.size());
          const RtEvent ready = context->escape_task_local_instance(
              pit.first, safe_effects, instances.size(), &instances.front(),
              &unique_events.front(),
              (const Realm::InstanceLayoutGeneric**)&layouts.front(), escaped);
          for (unsigned idx = 0; idx < instances.size(); idx++)
          {
            if (managers[idx]->update_physical_instance(
                    instances[idx], ready, layouts[idx]->bytes_used))
              delete managers[idx];
            delete layouts[idx];
          }
        }
        else
        {
          // Make an empty instance for each manager
          for (unsigned idx = 0; idx < pit.second.size(); idx++)
          {
            const FieldID field_id = pit.second[idx];
            PhysicalManager* manager = get_manager(field_id);
            // Create a layout to use for redistricting
            const LayoutConstraints* constraints = manager->layout->constraints;

            const std::vector<FieldID> fields(1, field_id);
            const std::vector<size_t> sizes(1, get_field_size(field_id));
            Realm::InstanceLayoutGeneric* layout =
                region->row_source->create_layout(
                    *constraints, fields, sizes, false /*compact*/);
            Realm::ProfilingRequestSet requests;
            if (runtime->profiler != nullptr)
            {
              const LgEvent unique_event = manager->get_unique_event();
              legion_assert(unique_event.exists());
              runtime->profiler->add_inst_request(
                  requests, context->owner_task->get_unique_id(), unique_event);
            }
            PhysicalInstance instance;
            const RtEvent ready(Realm::RegionInstance::create_instance(
                instance, manager->memory_manager->memory, *layout, requests));
            if (ready.exists() && (implicit_profiler != nullptr))
              implicit_profiler->record_instance_ready(
                  ready, manager->get_unique_event());
            if (manager->update_physical_instance(
                    instance, ready, 0 /*footprint*/))
              delete manager;
            delete layout;
          }
        }
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void OutputRegionImpl::FinalizeOutputRegionArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      RtEvent finalized = region->finalize(safe_effects, true /*escaped*/);
      if (finalized.exists())
        Processor::add_finish_event_precondition(finalized);
      if (region->remove_reference())
        delete region;
    }

    //--------------------------------------------------------------------------
    bool OutputRegionImpl::is_complete(FieldID& unbound_field) const
    //--------------------------------------------------------------------------
    {
      for (const FieldID& fid : req.instance_fields)
      {
        if (returned_instances.find(fid) == returned_instances.end())
        {
          unbound_field = fid;
          return false;
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    PhysicalManager* OutputRegionImpl::get_manager(FieldID field_id) const
    //--------------------------------------------------------------------------
    {
      std::vector<FieldID>::const_iterator finder = std::find(
          req.instance_fields.begin(), req.instance_fields.end(), field_id);
      if (finder == req.instance_fields.end())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Field " << field_id << " does not exist in output region "
              << index << " of " << *context << ".";
        error.raise();
      }
      return managers[std::distance(req.instance_fields.begin(), finder)];
    }

  }  // namespace Internal
}  // namespace Legion
