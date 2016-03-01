/* Copyright 2016 Stanford University, NVIDIA Corporation
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
#include "region_tree.h"
#include "legion_mapping.h"
#include "mapper_manager.h"
#include "legion_instances.h"

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
        legion_delete(impl);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance& PhysicalInstance::operator=(const PhysicalInstance &rhs)
    //--------------------------------------------------------------------------
    {
      if ((impl != NULL) && 
          impl->remove_base_resource_ref(Internal::INSTANCE_MAPPER_REF))
        legion_delete(impl);
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
      return (impl == NULL);
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
    /*static*/ PhysicalInstance PhysicalInstance::get_virtual_instance(void)
    //--------------------------------------------------------------------------
    {
      return PhysicalInstance();
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

    /////////////////////////////////////////////////////////////
    // Mapper 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Mapper::Mapper(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Mapper::~Mapper(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void Mapper::mapper_rt_find_valid_variants(MapperContext ctx,TaskID task_id,
                                         std::vector<VariantID> &valid_variants,
                                         Processor::Kind kind) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->find_valid_variants(ctx,task_id,valid_variants,kind);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_is_leaf_variant(MapperContext ctx, TaskID task_id,
                                           VariantID variant_id) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->is_leaf_variant(ctx, task_id, variant_id);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_is_inner_variant(MapperContext ctx, TaskID task_id,
                                            VariantID variant_id) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->is_inner_variant(ctx, task_id, variant_id);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_is_idempotent_variant(MapperContext ctx, 
                                     TaskID task_id, VariantID variant_id) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->is_idempotent_variant(ctx, task_id, variant_id);
    }

    //--------------------------------------------------------------------------
    void Mapper::mapper_rt_filter_variants(MapperContext ctx, const Task &task,
             const std::vector<std::vector<PhysicalInstance> > &chosen_intances,
                                           std::vector<VariantID> &variants)
    //--------------------------------------------------------------------------
    {
      ctx->manager->filter_variants(ctx, task, chosen_instances, variants);
    }

    //--------------------------------------------------------------------------
    void Mapper::mapper_rt_filter_instances(MapperContext ctx, const Task &task,
                                      VariantID chosen_variant, 
                        std::vector<std::vector<PhysicalInstance> > &instances,
                               std::vector<std::set<FieldID> > &missing_fields)
    //--------------------------------------------------------------------------
    {
      ctx->manager->filter_instances(ctx, task, chosen_variant, 
                                     instances, missing_fields);
    }

    //--------------------------------------------------------------------------
    void Mapper::mapper_rt_filter_instances(MapperContext ctx, const Task &task,
                                      unsigned index, VariantID chosen_variant,
                                      std::vector<PhysicalInstance> &instances,
                                      std::set<FieldID> &missing_fields)
    //--------------------------------------------------------------------------
    {
      ctx->manager->filter_instances(ctx, task, index, chosen_variant,
                                     instances, missing_fields);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_create_physical_instance(
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
    bool Mapper::mapper_rt_create_physical_instance(
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
    bool Mapper::mapper_rt_find_or_create_physical_instance(
                                    MapperContext ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints, 
                                    const std::vector<LogicalRegion> &regions, 
                                    PhysicalInstance &result, bool &created, 
                                    bool acquire, GCPriority priority) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->find_or_create_physical_instance(ctx, target_memory, 
                     constraints, regions, result, created, acquire, priority);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_find_or_create_physical_instance(
                                    MapperContext ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    PhysicalInstance &result, bool &created, 
                                    bool acquire, GCPriority priority) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->find_or_create_physical_instance(ctx, target_memory,
                       layout_id, regions, result, created, acquire, priority);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_find_physical_instance(
                                    MapperContext ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    PhysicalInstance &result,
                                    bool acquire) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->find_physical_instance(ctx, target_memory, 
                              constraints, regions, result, acquire);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_find_physical_instance(
                                    MapperContext ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions, 
                                    PhysicalInstance &result,
                                    bool acquire) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->find_physical_instance(ctx, target_memory,
                                layout_id, regions, result, acquire);
    }

    //--------------------------------------------------------------------------
    void Mapper::mapper_rt_set_garbage_collection_priority(MapperContext ctx,
                    const PhysicalInstance &instance, GCPriority priority) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->set_garbage_collection_priority(ctx, instance, priority);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_acquire_instance(MapperContext ctx, 
                                         const PhysicalInstance &instance) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->acquire_instance(ctx, instance);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_acquire_instances(MapperContext ctx,
                          const std::vector<PhysicalInstance> &instances) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->acquire_instances(ctx, instances);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_acquire_and_filter_instances(MapperContext ctx,
                                std::vector<PhysicalInstance> &instances) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->acquire_and_filter_instances(ctx, instances);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_acquire_instances(MapperContext ctx,
            const std::vector<std::vector<PhysicalInstance> > &instances) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->acquire_instances(ctx, instances);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_acquire_and_filter_instances(MapperContext ctx,
                  std::vector<std::vector<PhysicalInstance> > &instances) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->acquire_and_filter_instances(ctx, instances);
    }

    //--------------------------------------------------------------------------
    void Mapper::mapper_rt_release_instance(MapperContext ctx, 
                                         const PhysicalInstance &instance) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->release_instance(ctx, instance);
    }

    //--------------------------------------------------------------------------
    void Mapper::mapper_rt_release_instances(MapperContext ctx,
                          const std::vector<PhysicalInstance> &instances) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->release_instances(ctx, instances);
    }

    //--------------------------------------------------------------------------
    void Mapper::mapper_rt_release_instances(MapperContext ctx,
            const std::vector<std::vector<PhysicalInstance> > &instances) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->release_instances(ctx, instances);
    }

    //--------------------------------------------------------------------------
    IndexPartition Mapper::mapper_rt_get_index_partition(MapperContext ctx,
                                           IndexSpace parent, Color color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_partition(ctx, parent, color);
    }

    //--------------------------------------------------------------------------
    IndexSpace Mapper::mapper_rt_get_index_subspace(MapperContext ctx, 
                                          IndexPartition p, Color c) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_subspace(ctx, p, c);
    }

    //--------------------------------------------------------------------------
    IndexSpace Mapper::mapper_rt_get_index_subspace(MapperContext ctx, 
                               IndexPartition p, const DomainPoint &color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_subspace(ctx, p, color);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_has_multiple_domains(MapperContext ctx,
                                                IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->has_multiple_domains(ctx, handle);
    }

    //--------------------------------------------------------------------------
    Domain Mapper::mapper_rt_get_index_space_domain(MapperContext ctx, 
                                                    IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_space_domain(ctx, handle);
    }

    //--------------------------------------------------------------------------
    void Mapper::mapper_rt_get_index_space_domains(MapperContext ctx, 
                          IndexSpace handle, std::vector<Domain> &domains) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_space_domains(ctx, handle, domains);
    }

    //--------------------------------------------------------------------------
    Domain Mapper::mapper_rt_get_index_partition_color_space(MapperContext ctx,
                                                         IndexPartition p) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_partition_color_space(ctx, p);
    }

    //--------------------------------------------------------------------------
    void Mapper::mapper_rt_get_index_space_partition_colors(MapperContext ctx,
                              IndexSpace handle, std::set<Color> &colors) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->get_index_space_partition_colors(ctx, handle, colors);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_is_index_partition_disjoint(MapperContext ctx,
                                                       IndexPartition p) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->is_index_partition_disjoint(ctx, p);
    }

    //--------------------------------------------------------------------------
    Color Mapper::mapper_rt_get_index_space_color(MapperContext ctx, 
                                                  IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_space_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    Color Mapper::mapper_rt_get_index_partition_color(MapperContext ctx,
                                                    IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_partition_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace Mapper::mapper_rt_get_parent_index_space(MapperContext ctx,
                                                    IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_parent_index_space(ctx, handle);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_has_parent_index_partition(MapperContext ctx,
                                                      IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->has_parent_index_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition Mapper::mapper_rt_get_parent_index_partition(
                                     MapperContext ctx, IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_parent_index_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    size_t Mapper::mapper_rt_get_field_size(MapperContext ctx,
                                           FieldSpace handle, FieldID fid) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_field_size(ctx, handle, fid);
    }

    //--------------------------------------------------------------------------
    void Mapper::mapper_rt_get_field_space_fields(MapperContext ctx, 
                          FieldSpace handle, std::vector<FieldID> &fields) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->get_field_space_fields(ctx, handle, fields);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Mapper::mapper_rt_get_logical_partition(MapperContext ctx,
                              LogicalRegion parent, IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition(ctx, parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Mapper::mapper_rt_get_logical_partition_by_color(
                        MapperContext ctx, LogicalRegion par, Color color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition_by_color(ctx, par, color);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Mapper::mapper_rt_get_logical_partition_by_tree(
                                      MapperContext ctx, IndexPartition part,
                                      FieldSpace fspace, RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition_by_tree(ctx, part, fspace,tid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Mapper::mapper_rt_get_logical_subregion(MapperContext ctx,
                               LogicalPartition parent, IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_subregion(ctx, parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Mapper::mapper_rt_get_logical_subregion_by_color(
                     MapperContext ctx, LogicalPartition par, Color color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_subregion_by_color(ctx, par, color);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Mapper::mapper_rt_get_logical_subregion_by_tree(
                                      MapperContext ctx, IndexSpace handle, 
                                      FieldSpace fspace, RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_subregion_by_tree(ctx,handle,fspace,tid);
    }

    //--------------------------------------------------------------------------
    Color Mapper::mapper_rt_get_logical_region_color(MapperContext ctx,
                                                     LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_region_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    Color Mapper::mapper_rt_get_logical_partition_color(MapperContext ctx,
                                                  LogicalPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Mapper::mapper_rt_get_parent_logical_region(MapperContext ctx,
                                                    LogicalPartition part) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_parent_logical_region(ctx, part);
    }
    
    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_has_parent_logical_partition(MapperContext ctx,
                                                     LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->has_parent_logical_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Mapper::mapper_rt_get_parent_logical_partition(
                                       MapperContext ctx, LogicalRegion r) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_parent_logical_partition(ctx, r);
    }

  }; // namespace Mapping
}; // namespace Legion

