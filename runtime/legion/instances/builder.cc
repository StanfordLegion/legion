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

#include "legion/instances/builder.h"
#include "legion/kernel/runtime.h"
#include "legion/instances/physical.h"
#include "legion/managers/memory.h"
#include "legion/nodes/expression.h"
#include "legion/nodes/field.h"
#include "legion/nodes/index.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Instance Builder
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceBuilder::~InstanceBuilder(void)
    //--------------------------------------------------------------------------
    {
      if (realm_layout != nullptr)
        delete realm_layout;
      if (piece_list != nullptr)
        free(piece_list);
    }

    //--------------------------------------------------------------------------
    PhysicalManager* InstanceBuilder::create_physical_instance(
        LayoutConstraintKind* unsat_kind, unsigned* unsat_index,
        size_t* footprint, RtEvent precondition, PhysicalInstance hole,
        LgEvent hole_unique_event)
    //--------------------------------------------------------------------------
    {
      legion_assert(valid);
      // If there are no fields then we are done
      if (field_sizes.empty())
      {
        {
          Warning warn;
          warn << "Ignoring request to create instance in memory "
               << memory_manager->memory.id << " with no fields.";
          warn.raise();
        }
        if (footprint != nullptr)
          *footprint = 0;
        if (unsat_kind != nullptr)
          *unsat_kind = LEGION_FIELD_CONSTRAINT;
        if (unsat_index != nullptr)
          *unsat_index = 0;
        return nullptr;
      }
      if (realm_layout == nullptr)
      {
        const std::vector<FieldID>& field_set =
            constraints.field_constraint.get_field_set();
        bool compact = false;
        const SpecializedConstraint& spec = constraints.specialized_constraint;
        switch (spec.get_kind())
        {
          case LEGION_COMPACT_SPECIALIZE:
          case LEGION_COMPACT_REDUCTION_SPECIALIZE:
            {
              compact = true;
              break;
            }
          default:
            break;
        }
        size_t num_pieces = 0;
        realm_layout = instance_domain->create_layout(
            constraints, field_set, field_sizes, compact, &piece_list,
            &piece_list_size, &num_pieces);
        legion_assert(realm_layout != nullptr);
        // If we were doing a compact layout then Check that we met
        // the constraints for efficiency and number of pieces
        if (compact && (spec.max_pieces < num_pieces))
        {
          if (unsat_kind != nullptr)
            *unsat_kind = LEGION_SPECIALIZED_CONSTRAINT;
          if (unsat_index != nullptr)
            *unsat_index = 0;
          if (footprint != nullptr)
            *footprint = realm_layout->bytes_used;
          return nullptr;
        }
      }
      legion_assert(realm_layout != nullptr);
      // Have to grab this now since realm is going to take ownership of
      // the instance layout generic object once we do the creation call
      const size_t instance_footprint = realm_layout->bytes_used;
      // Save the footprint size if we need to
      if (footprint != nullptr)
        *footprint = instance_footprint;
      Realm::ProfilingRequestSet requests;
      // Add a profiling request to see if the instance is actually allocated
      // Make it very high priority so we get the response quickly
      ProfilingResponseBase base(this, creator_id, false /*completion*/);
#ifndef LEGION_MALLOC_INSTANCES
      Realm::ProfilingRequest& req = requests.add_request(
          runtime->find_local_group(), LG_LEGION_PROFILING_ID, &base,
          sizeof(base), LG_RESOURCE_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::InstanceAllocResult>();
      // Create a user event to wait on for the result of the profiling response
      profiling_ready = Runtime::create_rt_user_event();
#endif
      legion_assert(!allocated);
      legion_assert(!instance.exists());  // shouldn't exist before this
      LgEvent unique_event;
      if ((spy_logging_level > NO_SPY_LOGGING) ||
          (runtime->profiler != nullptr))
        Runtime::rename_event(unique_event);
      ApEvent ready;
      if (runtime->profiler != nullptr)
      {
        runtime->profiler->add_inst_request(requests, creator_id, unique_event);
        caller_fevent = implicit_fevent;
        current_unique_event = unique_event;
      }
#ifndef LEGION_MALLOC_INSTANCES
      if (hole.exists())
        ready = ApEvent(
            hole.redistrict(instance, realm_layout, requests, precondition));
      else
        ready = ApEvent(PhysicalInstance::create_instance(
            instance, memory_manager->memory, *realm_layout, requests,
            precondition));
      // Wait for the profiling response
      if (!profiling_ready.has_triggered())
        profiling_ready.wait();
#else
      if (precondition.exists() && !precondition.has_triggered())
        precondition.wait();
      ready = ApEvent(memory_manager->allocate_legion_instance(
          realm_layout, requests, instance, unique_event));
      allocated = instance.exists();
#endif
      // If we couldn't make it then we are done
      if (!allocated)
      {
        if (instance.exists())
        {
          // Destroy the instance ID so Realm can reclaim the ID
          instance.destroy();
          instance = PhysicalInstance::NO_INST;
        }
        if (unsat_kind != nullptr)
          *unsat_kind = LEGION_MEMORY_CONSTRAINT;
        if (unsat_index != nullptr)
          *unsat_index = 0;
        return nullptr;
      }
      legion_assert(!constraints.pointer_constraint.is_valid);
      // If we successfully made the instance then Realm
      // took over ownership of the layout
      PhysicalManager* result = nullptr;
      // If we successfully made it then we can
      // switch over the polarity of our constraints, this
      // shouldn't be necessary once Realm gets its act together
      // and actually tells us what the resulting constraints are
      constraints.field_constraint.contiguous = true;
      constraints.field_constraint.inorder = true;
      constraints.ordering_constraint.contiguous = true;
      constraints.memory_constraint =
          MemoryConstraint(memory_manager->memory.kind());
      const unsigned num_dims = instance_domain->get_num_dims();
      // Now let's find the layout constraints to use for this instance
      LayoutDescription* layout = field_space_node->find_layout_description(
          instance_mask, num_dims, constraints);
      // If we couldn't find one then we make one
      if (layout == nullptr)
      {
        // First make a new layout constraint
        LayoutConstraints* layout_constraints = runtime->register_layout(
            field_space_node->handle, constraints, true /*internal*/);
        // Then make our description
        layout = field_space_node->create_layout_description(
            instance_mask, num_dims, layout_constraints, mask_index_map,
            constraints.field_constraint.get_field_set(), field_sizes, serdez);
      }
      // Creating an individual manager
      DistributedID did = runtime->get_available_distributed_id();
      // Figure out what kind of instance we just made
      switch (constraints.specialized_constraint.get_kind())
      {
        case LEGION_NO_SPECIALIZE:
        case LEGION_AFFINE_SPECIALIZE:
        case LEGION_COMPACT_SPECIALIZE:
          {
            // Now we can make the manager
            result = new PhysicalManager(
                did, memory_manager, instance, instance_domain, piece_list,
                piece_list_size, field_space_node, tree_id, layout,
                0 /*redop id*/, true /*register now*/, instance_footprint,
                ready, unique_event, PhysicalManager::INTERNAL_INSTANCE_KIND);
            break;
          }
        case LEGION_AFFINE_REDUCTION_SPECIALIZE:
        case LEGION_COMPACT_REDUCTION_SPECIALIZE:
          {
            result = new PhysicalManager(
                did, memory_manager, instance, instance_domain, piece_list,
                piece_list_size, field_space_node, tree_id, layout, redop_id,
                true /*register now*/, instance_footprint, ready, unique_event,
                PhysicalManager::INTERNAL_INSTANCE_KIND, reduction_op);
            break;
          }
        default:
          std::abort();  // illegal specialized case
      }
      // manager takes ownership of the piece list
      piece_list = nullptr;
      // Remove the reference we got back from finding or creating the layout
      if (layout->remove_reference())
        delete layout;
      legion_assert(result != nullptr);
#ifdef LEGION_MALLOC_INSTANCES
      memory_manager->record_legion_instance(result, instance);
#else
      if (ready.exists() && (implicit_profiler != nullptr))
      {
        if (hole.exists())
          implicit_profiler->record_instance_redistrict(
              ready, hole_unique_event, unique_event, precondition);
        else
          implicit_profiler->record_instance_ready(ready, unique_event);
      }
#endif
      if (implicit_profiler != nullptr)
      {
        // Log the logical regions and fields that make up this instance
        for (const LogicalRegion& region : regions)
          if (region.exists())
            implicit_profiler->register_physical_instance_region(
                unique_event, region);
        implicit_profiler->register_physical_instance_layout(
            unique_event, layout->owner->handle, *layout->constraints);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    bool InstanceBuilder::handle_profiling_response(
        const Realm::ProfilingResponse& response, const void* orig,
        size_t orig_length, LgEvent& fevent, bool& failed_alloc)
    //--------------------------------------------------------------------------
    {
      legion_assert(response.has_measurement<
                    Realm::ProfilingMeasurements::InstanceAllocResult>());
      Realm::ProfilingMeasurements::InstanceAllocResult result;
      result.success = false;  // Need this to avoid compiler warnings
      legion_no_skip_assert(
          response.get_measurement<
              Realm::ProfilingMeasurements::InstanceAllocResult>(result));
      allocated = result.success;
      failed_alloc = !allocated;
      // Set the fevent in case we are profiling
      if (failed_alloc)
        fevent = caller_fevent;
      else
        fevent = current_unique_event;
      // No matter what trigger the event
      // Can't read anything after trigger the event as the object
      // might be deleted after we do that
      Runtime::trigger_event(profiling_ready);
      return true;
    }

    //--------------------------------------------------------------------------
    bool InstanceBuilder::initialize(void)
    //--------------------------------------------------------------------------
    {
      if (!compute_space_and_domain())
        return false;
      compute_layout_parameters();
      valid = true;
      return true;
    }

    //--------------------------------------------------------------------------
    bool InstanceBuilder::compute_space_and_domain(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!regions.empty());
      legion_assert(field_space_node == nullptr);
      legion_assert(instance_domain == nullptr);
      legion_assert(tree_id == 0);
      std::set<IndexSpaceExpression*> region_exprs;
      for (const LogicalRegion& region : regions)
      {
        if (!region.exists())
          continue;
        if (field_space_node == nullptr)
          field_space_node = runtime->get_node(region.get_field_space());
        if (tree_id == 0)
          tree_id = region.get_tree_id();
        // Check to make sure that all the field spaces have the same handle
        legion_assert(field_space_node->handle == region.get_field_space());
        legion_assert(tree_id == region.get_tree_id());
        IndexSpaceNode* node = runtime->get_node(
            region.get_index_space(), nullptr, true /*can fail*/);
        if ((node == nullptr) || !node->try_add_live_reference())
        {
          Warning warning;
          warning << "Unable to create instance for region " << region
                  << " because the corresponding index space has been deleted.";
          warning.raise();
          return false;
        }
        region_exprs.insert(runtime->get_node(region.get_index_space()));
      }
      instance_domain = (region_exprs.size() == 1) ?
                            *(region_exprs.begin()) :
                            runtime->union_index_spaces(region_exprs);
      return true;
    }

    //--------------------------------------------------------------------------
    void InstanceBuilder::compute_layout_parameters(void)
    //--------------------------------------------------------------------------
    {
      // First look at the OrderingConstraint to Figure out what kind
      // of instance we are building here, SOA, AOS, or hybrid
      const size_t num_dims = instance_domain->get_num_dims();
      OrderingConstraint& ord = constraints.ordering_constraint;
      if (!ord.ordering.empty())
      {
        // Find the index of the fields, if it is specified
        int field_idx = -1;
        std::set<DimensionKind> spatial_dims, to_remove;
        for (unsigned idx = 0; idx < ord.ordering.size(); idx++)
        {
          if (ord.ordering[idx] == LEGION_DIM_F)
          {
            // Should never be duplicated
            if (field_idx != -1)
            {
              Error err(LEGION_PROGRAMMING_MODEL_EXCEPTION);
              err << "Illegal ordering constraint used during instance "
                  << "creation contained multiple instances of DIM_F";
              err.raise();
            }
            else
              field_idx = idx;
          }
          else
          {
            // Should never be duplicated
            if (spatial_dims.find(ord.ordering[idx]) != spatial_dims.end())
            {
              Error err(LEGION_PROGRAMMING_MODEL_EXCEPTION);
              err << "Illegal ordering constraint used during instance "
                  << "creation contained multiple instances of dimension "
                  << ord.ordering[idx];
              err.raise();
            }
            else
            {
              // Check to make sure that it is one of our dims
              // if not we can just filter it out of the ordering
              if (ord.ordering[idx] >= num_dims)
                to_remove.insert(ord.ordering[idx]);
              else
                spatial_dims.insert(ord.ordering[idx]);
            }
          }
        }
        // Remove any dimensions which don't matter
        if (!to_remove.empty())
        {
          for (std::vector<DimensionKind>::iterator it = ord.ordering.begin();
               it != ord.ordering.end();
               /*nothing*/)
          {
            if (to_remove.find(*it) != to_remove.end())
              it = ord.ordering.erase(it);
            else
              it++;
          }
        }
        legion_assert(spatial_dims.size() <= num_dims);
        // Fill in any spatial dimensions that we didn't see if necessary
        if (spatial_dims.size() < num_dims)
        {
          // See if we should push these dims front or back
          if (field_idx > -1)
          {
            // See if we should add these at the front or the back
            if (field_idx == 0)
            {
              // Add them to the back
              for (unsigned idx = 0; idx < num_dims; idx++)
              {
                DimensionKind dim = (DimensionKind)(LEGION_DIM_X + idx);
                if (spatial_dims.find(dim) == spatial_dims.end())
                  ord.ordering.emplace_back(dim);
              }
            }
            else if (field_idx == int(ord.ordering.size() - 1))
            {
              // Add them to the front
              for (int idx = (num_dims - 1); idx >= 0; idx--)
              {
                DimensionKind dim = (DimensionKind)(LEGION_DIM_X + idx);
                if (spatial_dims.find(dim) == spatial_dims.end())
                  ord.ordering.insert(ord.ordering.begin(), dim);
              }
            }
            else  // Should either be AOS or SOA for now
              std::abort();
          }
          else
          {
            // No field dimension so just add the spatial ones on the back
            for (unsigned idx = 0; idx < num_dims; idx++)
            {
              DimensionKind dim = (DimensionKind)(LEGION_DIM_X + idx);
              if (spatial_dims.find(dim) == spatial_dims.end())
                ord.ordering.emplace_back(dim);
            }
          }
        }
        // If we didn't see the field dimension either then add that
        // at the end to give us SOA layouts in general
        if (field_idx == -1)
          ord.ordering.emplace_back(LEGION_DIM_F);
        // We've now got all our dimensions so we can set the
        // contiguous flag to true
        ord.contiguous = true;
      }
      else
      {
        // We had no ordering constraints so populate it with
        // SOA constraints for now
        for (unsigned idx = 0; idx < num_dims; idx++)
          ord.ordering.emplace_back((DimensionKind)(LEGION_DIM_X + idx));
        ord.ordering.emplace_back(LEGION_DIM_F);
        ord.contiguous = true;
      }
      legion_assert(ord.contiguous);
      legion_assert(ord.ordering.size() == (num_dims + 1));
      // Check the tiling constraints
      if (!constraints.tiling_constraints.empty())
      {
        // Check to make sure we're not asking for a compact-sparse instance
        switch (constraints.specialized_constraint.get_kind())
        {
          case LEGION_COMPACT_SPECIALIZE:
          case LEGION_COMPACT_REDUCTION_SPECIALIZE:
            {
              Error err(LEGION_PROGRAMMING_MODEL_EXCEPTION);
              err << "Illegal tiling constraints specified for compact-sparse "
                  << "instance creation. Tiling constraints can only be "
                     "specified "
                  << "on affine instances currently. If you have a compelling "
                     "use "
                  << "case for tiling the pieces of an compact-sparse instance "
                  << "please report it to the Legion developer's mailing list.";
              err.raise();
            }
          default:
            break;
        }
        // Make sure that each of the dimensions are valid and aren't duplicated
        std::vector<bool> observed(num_dims, false);
        for (std::vector<TilingConstraint>::iterator it =
                 constraints.tiling_constraints.begin();
             it != constraints.tiling_constraints.end();
             /*nothing*/)
        {
          if ((it->dim < num_dims) && !observed[it->dim])
          {
            observed[it->dim] = true;
            it++;
          }
          else
            it = constraints.tiling_constraints.erase(it);
        }
      }
      // From this we should be able to compute the field groups
      // Use the FieldConstraint to put any fields in the proper order
      const std::vector<FieldID>& field_set =
          constraints.field_constraint.get_field_set();
      field_sizes.resize(field_set.size());
      mask_index_map.resize(field_set.size());
      serdez.resize(field_set.size());
      field_space_node->compute_field_layout(
          field_set, field_sizes, mask_index_map, serdez, instance_mask);
      // See if we have any specialization here that will
      // require us to update the field sizes
      switch (constraints.specialized_constraint.get_kind())
      {
        case LEGION_NO_SPECIALIZE:
        case LEGION_AFFINE_SPECIALIZE:
        case LEGION_COMPACT_SPECIALIZE:
          break;
        case LEGION_AFFINE_REDUCTION_SPECIALIZE:
        case LEGION_COMPACT_REDUCTION_SPECIALIZE:
          {
            // Reduction folds are a special case of normal specialize
            redop_id = constraints.specialized_constraint.get_reduction_op();
            reduction_op = Runtime::get_reduction_op(redop_id);
            for (unsigned idx = 0; idx < field_sizes.size(); idx++)
            {
              if (field_sizes[idx] != reduction_op->sizeof_lhs)
              {
                Error err(LEGION_PROGRAMMING_MODEL_EXCEPTION);
                err << "Illegal reduction instance request with field "
                    << field_set[idx] << " which has size " << field_sizes[idx]
                    << " but the LHS type of reduction operator " << redop_id
                    << " is " << reduction_op->sizeof_lhs;
                err.raise();
              }
              // Update the field sizes to the rhs of the reduction op
              field_sizes[idx] = reduction_op->sizeof_rhs;
            }
            break;
          }
        case LEGION_VIRTUAL_SPECIALIZE:
          {
            Error err(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            err << "Illegal request to create a virtual instance";
            err.raise();
          }
        default:
          {
            Error err(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            err << "Illegal request to create instance of type "
                << constraints.specialized_constraint.get_kind();
            err.raise();
          }
      }
      legion_assert(
          (constraints.padding_constraint.delta.get_dim() == 0) ||
          (constraints.padding_constraint.delta.get_dim() == (int)num_dims));
      // If we don't have a padding constraint then record that we
      // don't have any padding on this instance
      if (constraints.padding_constraint.delta.get_dim() == 0)
      {
        DomainPoint empty;
        empty.dim = num_dims;
        for (unsigned dim = 0; dim < num_dims; dim++)
          empty[dim] = 0;  // no padding
        constraints.padding_constraint.delta = Domain(empty, empty);
      }
      else
      {
        DomainPoint lo = constraints.padding_constraint.delta.lo();
        DomainPoint hi = constraints.padding_constraint.delta.hi();
        for (unsigned dim = 0; dim < num_dims; dim++)
        {
          if (lo[dim] < 0)
            lo[dim] = 0;
          if (hi[dim] < 0)
            hi[dim] = 0;
        }
        constraints.padding_constraint.delta = Domain(lo, hi);
      }
    }

  }  // namespace Internal
}  // namespace Legion
