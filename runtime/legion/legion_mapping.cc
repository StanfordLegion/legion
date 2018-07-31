/* Copyright 2018 Stanford University, NVIDIA Corporation
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

#include "legion.h"
#include "legion/region_tree.h"
#include "legion/legion_mapping.h"
#include "legion/mapper_manager.h"
#include "legion/legion_instances.h"

namespace Legion {
  namespace Mapping {

    /////////////////////////////////////////////////////////////
    // PhysicalInstance 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalInstance::PhysicalInstance(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalInstance::PhysicalInstance(PhysicalInstanceImpl i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      // By holding resource references, we prevent the data
      // structure from being collected, it doesn't change if 
      // the actual instance itself can be collected or not
      if (impl != NULL)
        impl->add_base_resource_ref(Internal::INSTANCE_MAPPER_REF);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance::PhysicalInstance(const PhysicalInstance &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_base_resource_ref(Internal::INSTANCE_MAPPER_REF);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance::~PhysicalInstance(void)
    //--------------------------------------------------------------------------
    {
      if ((impl != NULL) && 
          impl->remove_base_resource_ref(Internal::INSTANCE_MAPPER_REF))
        delete (impl);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance& PhysicalInstance::operator=(const PhysicalInstance &rhs)
    //--------------------------------------------------------------------------
    {
      if ((impl != NULL) && 
          impl->remove_base_resource_ref(Internal::INSTANCE_MAPPER_REF))
        delete (impl);
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_base_resource_ref(Internal::INSTANCE_MAPPER_REF);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::operator<(const PhysicalInstance &rhs) const
    //--------------------------------------------------------------------------
    {
      return (impl < rhs.impl);
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::operator==(const PhysicalInstance &rhs) const
    //--------------------------------------------------------------------------
    {
      return (impl == rhs.impl);
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::operator!=(const PhysicalInstance &rhs) const
    //--------------------------------------------------------------------------
    {
      return (impl != rhs.impl);
    }

    //--------------------------------------------------------------------------
    Memory PhysicalInstance::get_location(void) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return Memory::NO_MEMORY;
      return impl->get_memory();
    }

    //--------------------------------------------------------------------------
    unsigned long PhysicalInstance::get_instance_id(void) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return 0;
      return impl->get_instance().id;
    }

    //--------------------------------------------------------------------------
    LogicalRegion PhysicalInstance::get_logical_region(void) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return LogicalRegion::NO_REGION;
      return impl->region_node->handle;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintID PhysicalInstance::get_layout_id(void) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return 0;
      return impl->layout->constraints->layout_id;
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::exists(bool strong_test /*= false*/) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return false;
      // Check to see if it still exists for now, maybe in the future
      // we could do a full check to see if it still exists on its owner node
      if (strong_test)
        assert(false); // implement this
      return impl->get_instance().exists();
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::is_normal_instance(void) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return false;
      return impl->is_normal_instance();
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::is_virtual_instance(void) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return false;
      return impl->is_virtual_instance();
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::is_reduction_instance(void) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return false;
      return impl->is_reduction_instance();
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::is_external_instance(void) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return false;
      return impl->is_external_instance();
    }

    //--------------------------------------------------------------------------
    /*static*/ PhysicalInstance PhysicalInstance::get_virtual_instance(void)
    //--------------------------------------------------------------------------
    {
      return PhysicalInstance(Internal::implicit_runtime->virtual_manager);
    }

    //--------------------------------------------------------------------------
    void PhysicalInstance::get_fields(std::set<FieldID> &fields) const
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->get_fields(fields);
    }

    //--------------------------------------------------------------------------
    bool PhysicalInstance::has_field(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return false;
      return impl->has_field(fid);
    }

    //--------------------------------------------------------------------------
    void PhysicalInstance::has_fields(std::map<FieldID,bool> &fields) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
      {
        for (std::map<FieldID,bool>::iterator it = fields.begin();
              it != fields.end(); it++)
          it->second = false;
        return;
      }
      return impl->has_fields(fields);
    }

    //--------------------------------------------------------------------------
    void PhysicalInstance::remove_space_fields(std::set<FieldID> &fields) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return;
      impl->remove_space_fields(fields);
    }

    //--------------------------------------------------------------------------
    /*friend*/ std::ostream& operator<<(std::ostream& os,
					const PhysicalInstance& p)
    //--------------------------------------------------------------------------
    {
      return os << p.impl->get_instance();
    }

    /////////////////////////////////////////////////////////////
    // ProfilingRequest
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void ProfilingRequest::populate_realm_profiling_request(
					      Realm::ProfilingRequest& req)
    //--------------------------------------------------------------------------
    {
      for (std::set<ProfilingMeasurementID>::const_iterator it =
	     requested_measurements.begin();
	   it != requested_measurements.end();
	   ++it)
      {
	if((int)(*it) <= (int)(Realm::PMID_REALM_LAST))
	  req.add_measurement((Realm::ProfilingMeasurementID)(*it));
      }
    }

    /////////////////////////////////////////////////////////////
    // ProfilingResponse
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void ProfilingResponse::attach_realm_profiling_response(
					const Realm::ProfilingResponse& resp)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(realm_resp == NULL);
#endif
      realm_resp = &resp;
    }
    
    //--------------------------------------------------------------------------
    void ProfilingResponse::attach_overhead(
                                   ProfilingMeasurements::RuntimeOverhead *over)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(overhead == NULL);
#endif
      overhead = over; 
    }

    /////////////////////////////////////////////////////////////
    // Mapper 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Mapper::Mapper(MapperRuntime *rt)
      : runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Mapper::~Mapper(void)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // MapperRuntime
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MapperRuntime::MapperRuntime(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MapperRuntime::~MapperRuntime(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::is_locked(MapperContext ctx) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->is_locked(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::lock_mapper(MapperContext ctx, bool read_only) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->lock_mapper(ctx, read_only);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::unlock_mapper(MapperContext ctx) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->unlock_mapper(ctx);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::is_reentrant(MapperContext ctx) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->is_reentrant(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::enable_reentrant(MapperContext ctx) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->enable_reentrant(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::disable_reentrant(MapperContext ctx) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->disable_reentrant(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::update_mappable_tag(MapperContext ctx,
                               const Mappable &mappable, MappingTagID tag) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->update_mappable_tag(ctx, mappable, tag);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::update_mappable_data(MapperContext ctx,
      const Mappable &mappable, const void *mapper_data, size_t data_size) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->update_mappable_data(ctx, mappable, mapper_data, data_size);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::send_message(MapperContext ctx, Processor target,
          const void *message, size_t message_size, unsigned message_kind) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->send_message(ctx, target, 
                                 message, message_size, message_kind);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::broadcast(MapperContext ctx, const void *message,
                    size_t message_size, unsigned message_kind, int radix) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->broadcast(ctx, message, message_size, message_kind, radix);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::pack_physical_instance(MapperContext ctx, 
                               Serializer &rez, PhysicalInstance instance) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->pack_physical_instance(ctx, rez, instance);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::unpack_physical_instance(MapperContext ctx,
                          Deserializer &derez, PhysicalInstance &instance) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->unpack_physical_instance(ctx, derez, instance);
    }

    //--------------------------------------------------------------------------
    MapperEvent MapperRuntime::create_mapper_event(MapperContext ctx) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->create_mapper_event(ctx);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::has_mapper_event_triggered(MapperContext ctx,
                                                      MapperEvent event) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->has_mapper_event_triggered(ctx, event);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::trigger_mapper_event(MapperContext ctx, 
                                                MapperEvent event) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->trigger_mapper_event(ctx, event);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::wait_on_mapper_event(MapperContext ctx,
                                                MapperEvent event) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->wait_on_mapper_event(ctx, event);
    }

    //--------------------------------------------------------------------------
    const ExecutionConstraintSet& MapperRuntime::find_execution_constraints(
                         MapperContext ctx, TaskID task_id, VariantID vid) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->find_execution_constraints(ctx, task_id, vid);
    }

    //--------------------------------------------------------------------------
    const TaskLayoutConstraintSet& 
      MapperRuntime::find_task_layout_constraints(MapperContext ctx, 
                                            TaskID task_id, VariantID vid) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->find_task_layout_constraints(ctx, task_id, vid);
    }

    //--------------------------------------------------------------------------
    const LayoutConstraintSet&
      MapperRuntime::find_layout_constraints(MapperContext ctx, 
                                                LayoutConstraintID id) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->find_layout_constraints(ctx, id); 
    }

    //--------------------------------------------------------------------------
    LayoutConstraintID MapperRuntime::register_layout(MapperContext ctx,
                                   const LayoutConstraintSet &constraints,
                                   FieldSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->register_layout(ctx, constraints, handle);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::release_layout(MapperContext ctx,
                                          LayoutConstraintID layout_id) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->release_layout(ctx, layout_id);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::do_constraints_conflict(MapperContext ctx,
        LayoutConstraintID set1, LayoutConstraintID set2) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->do_constraints_conflict(ctx, set1, set2);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::do_constraints_entail(MapperContext ctx,
        LayoutConstraintID source, LayoutConstraintID target) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->do_constraints_entail(ctx, source, target);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::find_valid_variants(MapperContext ctx,TaskID task_id,
                                         std::vector<VariantID> &valid_variants,
                                         Processor::Kind kind) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->find_valid_variants(ctx,task_id,valid_variants,kind);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::is_leaf_variant(MapperContext ctx, TaskID task_id,
                                           VariantID variant_id) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->is_leaf_variant(ctx, task_id, variant_id);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::is_inner_variant(MapperContext ctx, TaskID task_id,
                                            VariantID variant_id) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->is_inner_variant(ctx, task_id, variant_id);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::is_idempotent_variant(MapperContext ctx, 
                                     TaskID task_id, VariantID variant_id) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->is_idempotent_variant(ctx, task_id, variant_id);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::filter_variants(MapperContext ctx, const Task &task,
            const std::vector<std::vector<PhysicalInstance> > &chosen_instances,
                                           std::vector<VariantID> &variants)
    //--------------------------------------------------------------------------
    {
      ctx->manager->filter_variants(ctx, task, chosen_instances, variants);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::filter_instances(MapperContext ctx, const Task &task,
                                      VariantID chosen_variant, 
                        std::vector<std::vector<PhysicalInstance> > &instances,
                               std::vector<std::set<FieldID> > &missing_fields)
    //--------------------------------------------------------------------------
    {
      ctx->manager->filter_instances(ctx, task, chosen_variant, 
                                     instances, missing_fields);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::filter_instances(MapperContext ctx, const Task &task,
                                      unsigned index, VariantID chosen_variant,
                                      std::vector<PhysicalInstance> &instances,
                                      std::set<FieldID> &missing_fields)
    //--------------------------------------------------------------------------
    {
      ctx->manager->filter_instances(ctx, task, index, chosen_variant,
                                     instances, missing_fields);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::create_physical_instance(
                                    MapperContext ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints, 
                                    const std::vector<LogicalRegion> &regions,
                                    PhysicalInstance &result, 
                                    bool acquire, GCPriority priority) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->create_physical_instance(ctx, target_memory, 
                      constraints, regions, result, acquire, priority);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::create_physical_instance(
                                    MapperContext ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    PhysicalInstance &result,
                                    bool acquire, GCPriority priority) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->create_physical_instance(ctx, target_memory, 
                        layout_id, regions, result, acquire, priority);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::find_or_create_physical_instance(
                                    MapperContext ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints, 
                                    const std::vector<LogicalRegion> &regions, 
                                    PhysicalInstance &result, bool &created, 
                                    bool acquire, GCPriority priority,
                                    bool tight_bounds) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->find_or_create_physical_instance(ctx, target_memory, 
       constraints, regions, result, created, acquire, priority, tight_bounds);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::find_or_create_physical_instance(
                                    MapperContext ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    PhysicalInstance &result, bool &created, 
                                    bool acquire, GCPriority priority,
                                    bool tight_bounds) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->find_or_create_physical_instance(ctx, target_memory,
         layout_id, regions, result, created, acquire, priority, tight_bounds);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::find_physical_instance(
                                    MapperContext ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    PhysicalInstance &result,
                                    bool acquire, bool tight_bounds) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->find_physical_instance(ctx, target_memory, 
                  constraints, regions, result, acquire, tight_bounds);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::find_physical_instance(
                                    MapperContext ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions, 
                                    PhysicalInstance &result,
                                    bool acquire, bool tight_bounds) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->find_physical_instance(ctx, target_memory,
                  layout_id, regions, result, acquire, tight_bounds);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::set_garbage_collection_priority(MapperContext ctx,
                    const PhysicalInstance &instance, GCPriority priority) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->set_garbage_collection_priority(ctx, instance, priority);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::acquire_instance(MapperContext ctx, 
                                         const PhysicalInstance &instance) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->acquire_instance(ctx, instance);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::acquire_instances(MapperContext ctx,
                          const std::vector<PhysicalInstance> &instances) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->acquire_instances(ctx, instances);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::acquire_and_filter_instances(MapperContext ctx,
                                std::vector<PhysicalInstance> &instances) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->acquire_and_filter_instances(ctx, instances);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::acquire_instances(MapperContext ctx,
            const std::vector<std::vector<PhysicalInstance> > &instances) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->acquire_instances(ctx, instances);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::acquire_and_filter_instances(MapperContext ctx,
                  std::vector<std::vector<PhysicalInstance> > &instances) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->acquire_and_filter_instances(ctx, instances);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::release_instance(MapperContext ctx, 
                                         const PhysicalInstance &instance) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->release_instance(ctx, instance);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::release_instances(MapperContext ctx,
                          const std::vector<PhysicalInstance> &instances) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->release_instances(ctx, instances);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::release_instances(MapperContext ctx,
            const std::vector<std::vector<PhysicalInstance> > &instances) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->release_instances(ctx, instances);
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::create_index_space(MapperContext ctx, 
                                                 Domain bounds) const
    //--------------------------------------------------------------------------
    {
      switch (bounds.get_dim())
      {
        case 1:
          {
            DomainT<1,coord_t> realm_is = bounds;
            return ctx->manager->create_index_space(ctx, bounds, &realm_is,
                Legion::Internal::NT_TemplateHelper::encode_tag<1,coord_t>());
          }
        case 2:
          {
            DomainT<2,coord_t> realm_is = bounds;
            return ctx->manager->create_index_space(ctx, bounds, &realm_is,
                Legion::Internal::NT_TemplateHelper::encode_tag<2,coord_t>());
          }
        case 3:
          {
            DomainT<3,coord_t> realm_is = bounds;
            return ctx->manager->create_index_space(ctx, bounds, &realm_is,
                Legion::Internal::NT_TemplateHelper::encode_tag<3,coord_t>());
          }
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::create_index_space(MapperContext ctx,
                                   const std::vector<DomainPoint> &points) const
    //--------------------------------------------------------------------------
    {
      switch (points[0].get_dim())
      {
        case 1:
          {
            std::vector<Realm::Point<1,coord_t> > realm_points(points.size());
            for (unsigned idx = 0; idx < points.size(); idx++)
              realm_points[idx] = Point<1,coord_t>(points[idx]);
            DomainT<1,coord_t> realm_is(
                (Realm::IndexSpace<1,coord_t>(realm_points)));
            const Domain domain(realm_is);
            return ctx->manager->create_index_space(ctx, domain, &realm_is,
                      Internal::NT_TemplateHelper::encode_tag<1,coord_t>());
          }
        case 2:
          {
            std::vector<Realm::Point<2,coord_t> > realm_points(points.size());
            for (unsigned idx = 0; idx < points.size(); idx++)
              realm_points[idx] = Point<2,coord_t>(points[idx]);
            DomainT<2,coord_t> realm_is(
                (Realm::IndexSpace<2,coord_t>(realm_points)));
            const Domain domain(realm_is);
            return ctx->manager->create_index_space(ctx, domain, &realm_is,
                      Internal::NT_TemplateHelper::encode_tag<2,coord_t>());
          }
        case 3:
          {
            std::vector<Realm::Point<3,coord_t> > realm_points(points.size());
            for (unsigned idx = 0; idx < points.size(); idx++)
              realm_points[idx] = Point<3,coord_t>(points[idx]);
            DomainT<3,coord_t> realm_is(
                (Realm::IndexSpace<3,coord_t>(realm_points)));
            const Domain domain(realm_is);
            return ctx->manager->create_index_space(ctx, domain, &realm_is,
                      Internal::NT_TemplateHelper::encode_tag<3,coord_t>());
          }
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::create_index_space(MapperContext ctx,
                                         const std::vector<Domain> &rects) const
    //--------------------------------------------------------------------------
    {
      switch (rects[0].get_dim())
      {
        case 1:
          {
            std::vector<Realm::Rect<1,coord_t> > realm_rects(rects.size());
            for (unsigned idx = 0; idx < rects.size(); idx++)
              realm_rects[idx] = Rect<1,coord_t>(rects[idx]);
            DomainT<1,coord_t> realm_is(
                (Realm::IndexSpace<1,coord_t>(realm_rects)));
            const Domain domain(realm_is);
            return ctx->manager->create_index_space(ctx, domain, &realm_is,
                      Internal::NT_TemplateHelper::encode_tag<1,coord_t>());
          }
        case 2:
          {
            std::vector<Realm::Rect<2,coord_t> > realm_rects(rects.size());
            for (unsigned idx = 0; idx < rects.size(); idx++)
              realm_rects[idx] = Rect<2,coord_t>(rects[idx]);
            DomainT<2,coord_t> realm_is(
                (Realm::IndexSpace<2,coord_t>(realm_rects)));
            const Domain domain(realm_is);
            return ctx->manager->create_index_space(ctx, domain, &realm_is,
                      Internal::NT_TemplateHelper::encode_tag<2,coord_t>());
          }
        case 3:
          {
            std::vector<Realm::Rect<3,coord_t> > realm_rects(rects.size());
            for (unsigned idx = 0; idx < rects.size(); idx++)
              realm_rects[idx] = Rect<3,coord_t>(rects[idx]);
            DomainT<3,coord_t> realm_is(
                (Realm::IndexSpace<3,coord_t>(realm_rects)));
            const Domain domain(realm_is);
            return ctx->manager->create_index_space(ctx, domain, &realm_is,
                      Internal::NT_TemplateHelper::encode_tag<3,coord_t>());
          }
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::create_index_space_internal(MapperContext ctx,
                  const Domain &d, const void *realm_is, TypeTag type_tag) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->create_index_space(ctx, d, realm_is, type_tag);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::has_index_partition(MapperContext ctx,
                                           IndexSpace parent, Color color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->has_index_partition(ctx, parent, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition MapperRuntime::get_index_partition(MapperContext ctx,
                                           IndexSpace parent, Color color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_partition(ctx, parent, color);
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::get_index_subspace(MapperContext ctx, 
                                          IndexPartition p, Color c) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_subspace(ctx, p, c);
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::get_index_subspace(MapperContext ctx, 
                               IndexPartition p, const DomainPoint &color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_subspace(ctx, p, color);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::has_multiple_domains(MapperContext ctx,
                                                IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->has_multiple_domains(ctx, handle);
    }

    //--------------------------------------------------------------------------
    Domain MapperRuntime::get_index_space_domain(MapperContext ctx, 
                                                    IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_space_domain(ctx, handle);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::get_index_space_domains(MapperContext ctx, 
                          IndexSpace handle, std::vector<Domain> &domains) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_space_domains(ctx, handle, domains);
    }

    //--------------------------------------------------------------------------
    Domain MapperRuntime::get_index_partition_color_space(MapperContext ctx,
                                                         IndexPartition p) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_partition_color_space(ctx, p);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::get_index_space_partition_colors(MapperContext ctx,
                              IndexSpace handle, std::set<Color> &colors) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->get_index_space_partition_colors(ctx, handle, colors);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::is_index_partition_disjoint(MapperContext ctx,
                                                       IndexPartition p) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->is_index_partition_disjoint(ctx, p);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::is_index_partition_complete(MapperContext ctx,
                                                    IndexPartition p) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->is_index_partition_complete(ctx, p);
    }

    //--------------------------------------------------------------------------
    Color MapperRuntime::get_index_space_color(MapperContext ctx, 
                                                  IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_space_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    DomainPoint MapperRuntime::get_index_space_color_point(MapperContext ctx, 
                                                        IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_space_color_point(ctx, handle);
    }

    //--------------------------------------------------------------------------
    Color MapperRuntime::get_index_partition_color(MapperContext ctx,
                                                    IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_partition_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperRuntime::get_parent_index_space(MapperContext ctx,
                                                    IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_parent_index_space(ctx, handle);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::has_parent_index_partition(MapperContext ctx,
                                                      IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->has_parent_index_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition MapperRuntime::get_parent_index_partition(
                                     MapperContext ctx, IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_parent_index_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    unsigned MapperRuntime::get_index_space_depth(MapperContext ctx,
                                                  IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_space_depth(ctx, handle);
    }

    //--------------------------------------------------------------------------
    unsigned MapperRuntime::get_index_partition_depth(MapperContext ctx,
                                                    IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_partition_depth(ctx, handle);
    }

    //--------------------------------------------------------------------------
    size_t MapperRuntime::get_field_size(MapperContext ctx,
                                           FieldSpace handle, FieldID fid) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_field_size(ctx, handle, fid);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::get_field_space_fields(MapperContext ctx, 
                          FieldSpace handle, std::vector<FieldID> &fields) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->get_field_space_fields(ctx, handle, fields);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::get_field_space_fields(MapperContext ctx, 
                          FieldSpace handle, std::set<FieldID> &fields) const
    //--------------------------------------------------------------------------
    {
      std::vector<FieldID> local;
      ctx->manager->get_field_space_fields(ctx, handle, local);
      fields.insert(local.begin(), local.end());
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperRuntime::get_logical_partition(MapperContext ctx,
                              LogicalRegion parent, IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition(ctx, parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperRuntime::get_logical_partition_by_color(
                        MapperContext ctx, LogicalRegion par, Color color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition_by_color(ctx, par, color);
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperRuntime::get_logical_partition_by_color(
           MapperContext ctx, LogicalRegion par, const DomainPoint &color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition_by_color(ctx, par, color);
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperRuntime::get_logical_partition_by_tree(
                                      MapperContext ctx, IndexPartition part,
                                      FieldSpace fspace, RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition_by_tree(ctx, part, fspace,tid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperRuntime::get_logical_subregion(MapperContext ctx,
                               LogicalPartition parent, IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_subregion(ctx, parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperRuntime::get_logical_subregion_by_color(
                     MapperContext ctx, LogicalPartition par, Color color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_subregion_by_color(ctx, par, color);
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperRuntime::get_logical_subregion_by_color(
        MapperContext ctx, LogicalPartition par, const DomainPoint &color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_subregion_by_color(ctx, par, color);
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperRuntime::get_logical_subregion_by_tree(
                                      MapperContext ctx, IndexSpace handle, 
                                      FieldSpace fspace, RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_subregion_by_tree(ctx,handle,fspace,tid);
    }

    //--------------------------------------------------------------------------
    Color MapperRuntime::get_logical_region_color(MapperContext ctx,
                                                     LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_region_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    DomainPoint MapperRuntime::get_logical_region_color_point(MapperContext ctx,
                                                     LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_region_color_point(ctx, handle);
    }

    //--------------------------------------------------------------------------
    Color MapperRuntime::get_logical_partition_color(MapperContext ctx,
                                                  LogicalPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperRuntime::get_parent_logical_region(MapperContext ctx,
                                                    LogicalPartition part) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_parent_logical_region(ctx, part);
    }
    
    //--------------------------------------------------------------------------
    bool MapperRuntime::has_parent_logical_partition(MapperContext ctx,
                                                     LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->has_parent_logical_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperRuntime::get_parent_logical_partition(
                                       MapperContext ctx, LogicalRegion r) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_parent_logical_partition(ctx, r);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::retrieve_semantic_information(MapperContext ctx,
        TaskID task_id, SemanticTag tag, const void *&result, size_t &size,
        bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      return ctx->manager->retrieve_semantic_information(ctx, task_id,
					  tag, result,
                                          size, can_fail, wait_until_ready);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::retrieve_semantic_information(MapperContext ctx, 
          IndexSpace handle, SemanticTag tag, const void *&result, size_t &size,
          bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      return ctx->manager->retrieve_semantic_information(ctx, handle,
					  tag, result,
                                          size, can_fail, wait_until_ready);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::retrieve_semantic_information(MapperContext ctx,
          IndexPartition handle, SemanticTag tag, const void *&result, 
          size_t &size, bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      return ctx->manager->retrieve_semantic_information(ctx, handle,
					  tag, result,
                                          size, can_fail, wait_until_ready);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::retrieve_semantic_information(MapperContext ctx,
          FieldSpace handle, SemanticTag tag, const void *&result, size_t &size,
          bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      return ctx->manager->retrieve_semantic_information(ctx, handle,
					  tag, result,
                                          size, can_fail, wait_until_ready);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::retrieve_semantic_information(MapperContext ctx, 
          FieldSpace handle, FieldID fid, SemanticTag tag, const void *&result, 
          size_t &size, bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      return ctx->manager->retrieve_semantic_information(ctx, handle, fid, tag,
					  result, size,
                                          can_fail, wait_until_ready);
    }

    //--------------------------------------------------------------------------
    bool MapperRuntime::retrieve_semantic_information(MapperContext ctx,
          LogicalRegion handle, SemanticTag tag, const void *&result, 
          size_t &size, bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      return ctx->manager->retrieve_semantic_information(ctx, handle, tag,
					  result, size,
                                          can_fail, wait_until_ready);
    }

    //--------------------------------------------------------------------------
   bool MapperRuntime::retrieve_semantic_information(MapperContext ctx,
          LogicalPartition handle, SemanticTag tag, const void *&result, 
          size_t &size, bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      return ctx->manager->retrieve_semantic_information(ctx, handle, tag,
					  result, size,
                                          can_fail, wait_until_ready);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::retrieve_name(MapperContext ctx, TaskID task_id,
                                         const char *&result)
    //--------------------------------------------------------------------------
    {
      ctx->manager->retrieve_name(ctx, task_id, result);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::retrieve_name(MapperContext ctx, IndexSpace handle,
                                         const char *&result)
    //--------------------------------------------------------------------------
    {
      ctx->manager->retrieve_name(ctx, handle, result);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::retrieve_name(MapperContext ctx, 
                                     IndexPartition handle, const char *&result)
    //--------------------------------------------------------------------------
    {
      ctx->manager->retrieve_name(ctx, handle, result);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::retrieve_name(MapperContext ctx,
                                         FieldSpace handle, const char *&result)
    //--------------------------------------------------------------------------
    {
      ctx->manager->retrieve_name(ctx, handle, result);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::retrieve_name(MapperContext ctx, FieldSpace handle,
                                         FieldID fid, const char *&result)
    //--------------------------------------------------------------------------
    {
      ctx->manager->retrieve_name(ctx, handle, fid, result);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::retrieve_name(MapperContext ctx, 
                                      LogicalRegion handle, const char *&result)
    //--------------------------------------------------------------------------
    {
      ctx->manager->retrieve_name(ctx, handle, result);
    }

    //--------------------------------------------------------------------------
    void MapperRuntime::retrieve_name(MapperContext ctx,
                                   LogicalPartition handle, const char *&result)
    //--------------------------------------------------------------------------
    {
      ctx->manager->retrieve_name(ctx, handle, result);
    }

  }; // namespace Mapping
}; // namespace Legion

