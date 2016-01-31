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

#include "legion_mapping.h"
#include "mapper_manager.h"

namespace Legion {
  namespace Mapping {

    /////////////////////////////////////////////////////////////
    // Mappable 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Mappable::Mappable(void)
      : map_id(0), tag(0)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Task::Task(void)
      : Mappable(), args(NULL), arglen(0), local_args(NULL), local_arglen(0),
        parent_task(NULL)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Copy 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Copy::Copy(void)
      : Mappable(), parent_task(NULL)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Inline Mapping 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InlineMapping::InlineMapping(void)
      : Mappable(), parent_task(NULL)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Acquire 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Acquire::Acquire(void)
      : Mappable(), parent_task(NULL)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Release 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Release::Release(void)
      : Mappable(), parent_task(NULL)
    //--------------------------------------------------------------------------
    {
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
    IndexPartition Mapper::get_index_partition(MapperManager ctx,
                                           IndexSpace parent, Color color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_partition(parent, color);
    }

    //--------------------------------------------------------------------------
    IndexSpace Mapper::get_index_subspace(MapperManager ctx, 
                                          IndexPartition p, Color c) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_subspace(p, c);
    }

    //--------------------------------------------------------------------------
    IndexSpace Mapper::get_index_subspace(MapperManager ctx, IndexPartition p, 
                                          const DomainPoint &color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_subspace(p, color);
    }

    //--------------------------------------------------------------------------
    bool Mapper::has_multiple_domains(MapperManager ctx,IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->has_multiple_domains(handle);
    }

    //--------------------------------------------------------------------------
    Domain Mapper::get_index_space_domain(MapperManager ctx, 
                                          IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_space_domain(handle);
    }

    //--------------------------------------------------------------------------
    void Mapper::get_index_space_domains(MapperManager ctx, IndexSpace handle,
                                         std::vector<Domain> &domains) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_space_domains(handle, domains);
    }

    //--------------------------------------------------------------------------
    Domain Mapper::get_index_partition_color_space(MapperManager ctx,
                                                   IndexPartition p) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_partition_color_space(p);
    }

    //--------------------------------------------------------------------------
    void Mapper::get_index_space_partition_colors(MapperManager ctx,
                              IndexSpace handle, std::set<Color> &colors) const
    //--------------------------------------------------------------------------
    {
      ctx->manager->get_index_space_partition_colors(handle, colors);
    }

    //--------------------------------------------------------------------------
    bool Mapper::is_index_partition_disjoint(MapperManager ctx,
                                             IndexPartition p) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->is_index_partition_disjoint(p);
    }

    //--------------------------------------------------------------------------
    Color Mapper::get_index_space_color(MapperManager ctx, 
                                        IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_space_color(handle);
    }

    //--------------------------------------------------------------------------
    Color Mapper::get_index_partition_color(MapperManager ctx,
                                            IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_index_partition_color(handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace Mapper::get_parent_index_space(MapperManager ctx,
                                              IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_parent_index_space(handle);
    }

    //--------------------------------------------------------------------------
    bool Mapper::has_parent_index_partition(MapperManager ctx,
                                            IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->has_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition Mapper::get_parent_index_partition(MapperManager ctx,
                                                      IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    size_t Mapper::get_field_size(MapperManager ctx,
                                  FieldSpace handle, FieldID fid) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_field_size(handle, fid);
    }

    //--------------------------------------------------------------------------
    void Mapper::get_field_space_fields(MapperContext ctx, FieldSpace handle, 
                                        std::set<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      ctx->manager->get_field_space_fields(handle, fields);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Mapper::get_logical_partition(MapperContext ctx,
                              LogicalRegion parent, IndexPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Mapper::get_logical_partition_by_color(MapperContext ctx,
                                           LogicalRegion par, Color color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition_by_color(par, color);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Mapper::get_logical_partition_by_tree(MapperContext ctx,
                                                        IndexPartition part,
                                                        FieldSpace fspace, 
                                                        RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition_by_tree(part, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Mapper::get_logical_subregion(MapperContext ctx,
                               LogicalPartition parent, IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_subregion(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Mapper::get_logical_subregion_by_color(MapperContext ctx,
                                        LogicalPartition par, Color color) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_subregion_by_color(par, color);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Mapper::get_logical_subregion_by_tree(MapperContext ctx,
                   IndexSpace handle, FieldSpace fspace, RegionTreeID tid) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_subregion_by_tree(handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    Color Mapper::get_logical_region_color(MapperContext ctx,
                                           LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_region_color(handle);
    }

    //--------------------------------------------------------------------------
    Color Mapper::get_logical_partition_color(MapperContext ctx,
                                              LogicalPartition handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_logical_partition_color(handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Mapper::get_parent_logical_region(MapperContext ctx,
                                                    LogicalPartition part) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_parent_logical_region(part);
    }
    
    //--------------------------------------------------------------------------
    bool Mapper::has_parent_logical_partition(MapperContext ctx,
                                              LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->has_parent_logical_partition(handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Mapper::get_parent_logical_partition(MapperContext ctx,
                                                          LogicalRegion r) const
    //--------------------------------------------------------------------------
    {
      return ctx->manager->get_parent_logical_partition(r);
    }

  }; // namespace Mapping
}; // namespace Legion

