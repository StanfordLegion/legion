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
#include "legion_mapping.h"
#include "mapper_manager.h"

namespace Legion {
  namespace Mapping {

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
    IndexPartition Mapper::mapper_rt_get_index_partition(MapperContext ctx,
                                           IndexSpace parent, Color color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_partition(parent, color);
    }

    //--------------------------------------------------------------------------
    IndexSpace Mapper::mapper_rt_get_index_subspace(MapperContext ctx, 
                                          IndexPartition p, Color c) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_subspace(p, c);
    }

    //--------------------------------------------------------------------------
    IndexSpace Mapper::mapper_rt_get_index_subspace(MapperContext ctx, 
                               IndexPartition p, const DomainPoint &color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_subspace(p, color);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_has_multiple_domains(MapperContext ctx,
                                                IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->has_multiple_domains(handle);
    }

    //--------------------------------------------------------------------------
    Domain Mapper::mapper_rt_get_index_space_domain(MapperContext ctx, 
                                                    IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_space_domain(handle);
    }

    //--------------------------------------------------------------------------
    void Mapper::mapper_rt_get_index_space_domains(MapperContext ctx, 
                          IndexSpace handle, std::vector<Domain> &domains) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_space_domains(handle, domains);
    }

    //--------------------------------------------------------------------------
    Domain Mapper::mapper_rt_get_index_partition_color_space(MapperContext ctx,
                                                         IndexPartition p) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_partition_color_space(p);
    }

    //--------------------------------------------------------------------------
    void Mapper::mapper_rt_get_index_space_partition_colors(MapperContext ctx,
                              IndexSpace handle, std::set<Color> &colors) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->get_index_space_partition_colors(handle, colors);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_is_index_partition_disjoint(MapperContext ctx,
                                                       IndexPartition p) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->is_index_partition_disjoint(p);
    }

    //--------------------------------------------------------------------------
    Color Mapper::mapper_rt_get_index_space_color(MapperContext ctx, 
                                                  IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_space_color(handle);
    }

    //--------------------------------------------------------------------------
    Color Mapper::mapper_rt_get_index_partition_color(MapperContext ctx,
                                                    IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_partition_color(handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace Mapper::mapper_rt_get_parent_index_space(MapperContext ctx,
                                                    IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_parent_index_space(handle);
    }

    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_has_parent_index_partition(MapperContext ctx,
                                                      IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->has_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition Mapper::mapper_rt_get_parent_index_partition(
                                     MapperContext ctx, IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    size_t Mapper::mapper_rt_get_field_size(MapperContext ctx,
                                           FieldSpace handle, FieldID fid) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_field_size(handle, fid);
    }

    //--------------------------------------------------------------------------
    void Mapper::mapper_rt_get_field_space_fields(MapperContext ctx, 
                          FieldSpace handle, std::vector<FieldID> &fields) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->get_field_space_fields(handle, fields);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Mapper::mapper_rt_get_logical_partition(MapperContext ctx,
                              LogicalRegion parent, IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Mapper::mapper_rt_get_logical_partition_by_color(
                        MapperContext ctx, LogicalRegion par, Color color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition_by_color(par, color);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Mapper::mapper_rt_get_logical_partition_by_tree(
                                      MapperContext ctx, IndexPartition part,
                                      FieldSpace fspace, RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition_by_tree(part, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Mapper::mapper_rt_get_logical_subregion(MapperContext ctx,
                               LogicalPartition parent, IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_subregion(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Mapper::mapper_rt_get_logical_subregion_by_color(
                     MapperContext ctx, LogicalPartition par, Color color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_subregion_by_color(par, color);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Mapper::mapper_rt_get_logical_subregion_by_tree(
                                      MapperContext ctx, IndexSpace handle, 
                                      FieldSpace fspace, RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_subregion_by_tree(handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    Color Mapper::mapper_rt_get_logical_region_color(MapperContext ctx,
                                                     LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_region_color(handle);
    }

    //--------------------------------------------------------------------------
    Color Mapper::mapper_rt_get_logical_partition_color(MapperContext ctx,
                                                  LogicalPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition_color(handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Mapper::mapper_rt_get_parent_logical_region(MapperContext ctx,
                                                    LogicalPartition part) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_parent_logical_region(part);
    }
    
    //--------------------------------------------------------------------------
    bool Mapper::mapper_rt_has_parent_logical_partition(MapperContext ctx,
                                                     LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->has_parent_logical_partition(handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Mapper::mapper_rt_get_parent_logical_partition(
                                       MapperContext ctx, LogicalRegion r) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_parent_logical_partition(r);
    }

  }; // namespace Mapping
}; // namespace Legion

