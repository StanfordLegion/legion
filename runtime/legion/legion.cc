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
#include "legion/runtime.h"
#include "legion/legion_ops.h"
#include "legion/legion_tasks.h"
#include "legion/legion_context.h"
#include "legion/legion_profiling.h"
#include "legion/legion_allocation.h"

namespace Legion {
#ifndef DISABLE_PARTITION_SHIM
  namespace PartitionShim {

    template<int COLOR_DIM>
    /*static*/ TaskID ColorPoints<COLOR_DIM>::TASK_ID;
    template<int COLOR_DIM, int RANGE_DIM>
    /*static*/ TaskID ColorRects<COLOR_DIM,RANGE_DIM>::TASK_ID;
    
    //--------------------------------------------------------------------------
    template<int CDIM>
    ColorPoints<CDIM>::ColorPoints(const Coloring &coloring, 
        LogicalRegion region, FieldID color_field, FieldID pointer_field)
      : TaskLauncher(TASK_ID, TaskArgument(), Predicate::TRUE_PRED,
                     PARTITION_SHIM_MAPPER_ID)
    //--------------------------------------------------------------------------
    {
      add_region_requirement(
          RegionRequirement(region, WRITE_DISCARD, EXCLUSIVE, region));
      add_field(0/*index*/, color_field);
      add_field(0/*index*/, pointer_field);
      // Serialize the coloring into the argument buffer 
      assert(CDIM == 1);
      rez.serialize<size_t>(coloring.size());
      for (Coloring::const_iterator cit = coloring.begin(); 
            cit != coloring.end(); cit++)
      {
        const Point<CDIM,coord_t> color = DomainPoint(cit->first); 
        rez.serialize(color);
        rez.serialize<size_t>(cit->second.points.size());
        for (std::set<ptr_t>::const_iterator it = cit->second.points.begin();
              it != cit->second.points.end(); it++)
        {
          const Point<1,coord_t> point = it->value;
          rez.serialize(point);
        }
        // No need to do ranges since we know that they are all empty
      }
      argument = TaskArgument(rez.get_buffer(), rez.get_used_bytes()); 
    }

    //--------------------------------------------------------------------------
    template<int CDIM>
    ColorPoints<CDIM>::ColorPoints(const PointColoring &coloring, 
        LogicalRegion region, FieldID color_field, FieldID pointer_field)
      : TaskLauncher(TASK_ID, TaskArgument(), Predicate::TRUE_PRED,
                     PARTITION_SHIM_MAPPER_ID)
    //--------------------------------------------------------------------------
    {
      add_region_requirement(
          RegionRequirement(region, WRITE_DISCARD, EXCLUSIVE, region));
      add_field(0/*index*/, color_field);
      add_field(0/*index*/, pointer_field);
      // Serialize the coloring into the argument buffer 
      rez.serialize<size_t>(coloring.size());
      for (PointColoring::const_iterator cit = coloring.begin();
            cit != coloring.end(); cit++)
      {
        const Point<CDIM,coord_t> color = cit->first; 
        rez.serialize(color);
        rez.serialize<size_t>(cit->second.points.size());
        for (std::set<ptr_t>::const_iterator it = cit->second.points.begin();
              it != cit->second.points.end(); it++)
        {
          const Point<1,coord_t> point = it->value;
          rez.serialize(point);
        }
        // No need to do any ranges since we know they are all empty
      }
      argument = TaskArgument(rez.get_buffer(), rez.get_used_bytes());
    }
 
    //--------------------------------------------------------------------------
    template<int CDIM>
    /*static*/ void ColorPoints<CDIM>::register_task(void)
    //--------------------------------------------------------------------------
    {
      TASK_ID = Legion::Runtime::generate_static_task_id();
      char variant_name[128];
      snprintf(variant_name,128,"Color Points <%d>", CDIM);
      TaskVariantRegistrar registrar(TASK_ID, variant_name);
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Legion::Runtime::preregister_task_variant<
        ColorPoints<CDIM>::cpu_variant>(registrar, variant_name);
    }

    //--------------------------------------------------------------------------
    template<int CDIM>
    /*static*/ void ColorPoints<CDIM>::cpu_variant(const Task *task, 
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const FieldAccessor<WRITE_DISCARD,Point<CDIM,coord_t>,1,coord_t>
        fa_color(regions[0], task->regions[0].instance_fields[0]);
      const FieldAccessor<WRITE_DISCARD,Point<1,coord_t>,1,coord_t>
        fa_point(regions[0], task->regions[0].instance_fields[1]);
      Deserializer derez(task->args, task->arglen);
      size_t num_colors;
      derez.deserialize(num_colors);
      coord_t next_entry = 0;
      for (unsigned cidx = 0; cidx < num_colors; cidx++)
      {
        Point<CDIM,coord_t> color;
        derez.deserialize(color);
        size_t num_points;
        derez.deserialize(num_points);
        for (unsigned idx = 0; idx < num_points; idx++, next_entry++)
        {
          Point<1,coord_t> point;
          derez.deserialize(point);
          fa_color.write(next_entry, color);
          fa_point.write(next_entry, point);
        }
      }
    }

    //--------------------------------------------------------------------------
    template<int CDIM, int RDIM>
    ColorRects<CDIM,RDIM>::ColorRects(const DomainColoring &coloring, 
        LogicalRegion region, FieldID color_field, FieldID range_field)
      : TaskLauncher(TASK_ID, TaskArgument(), Predicate::TRUE_PRED,
                     PARTITION_SHIM_MAPPER_ID)
    //--------------------------------------------------------------------------
    {
      add_region_requirement(
          RegionRequirement(region, WRITE_DISCARD, EXCLUSIVE, region));
      add_field(0/*index*/, color_field);
      add_field(0/*index*/, range_field);
      // Serialize the coloring into the argument buffer 
      assert(CDIM == 1);
      rez.serialize<size_t>(coloring.size());
      for (DomainColoring::const_iterator cit = coloring.begin();
            cit != coloring.end(); cit++)
      {
        const Point<CDIM,coord_t> color = DomainPoint(cit->first);
        rez.serialize(color);
        rez.serialize<size_t>(1); // number of rects
        const Rect<RDIM,coord_t> rect = cit->second;
        rez.serialize(rect);
      }
      argument = TaskArgument(rez.get_buffer(), rez.get_used_bytes());
    }

    //--------------------------------------------------------------------------
    template<int CDIM, int RDIM>
    ColorRects<CDIM,RDIM>::ColorRects(const MultiDomainColoring &coloring, 
        LogicalRegion region, FieldID color_field, FieldID range_field)
      : TaskLauncher(TASK_ID, TaskArgument(), Predicate::TRUE_PRED,
                     PARTITION_SHIM_MAPPER_ID)
    //--------------------------------------------------------------------------
    {
      add_region_requirement(
          RegionRequirement(region, WRITE_DISCARD, EXCLUSIVE, region));
      add_field(0/*index*/, color_field);
      add_field(0/*index*/, range_field);
      // Serialize the coloring into the argument buffer 
      assert(CDIM == 1);
      rez.serialize<size_t>(coloring.size());
      for (MultiDomainColoring::const_iterator cit = coloring.begin();
            cit != coloring.end(); cit++)
      {
        const Point<CDIM,coord_t> color = DomainPoint(cit->first);
        rez.serialize(color);
        rez.serialize<size_t>(cit->second.size());
        for (std::set<Domain>::const_iterator it = cit->second.begin();
              it != cit->second.end(); it++)
        {
          const Rect<RDIM,coord_t> rect = *it;
          rez.serialize(rect);
        }
      }
      argument = TaskArgument(rez.get_buffer(), rez.get_used_bytes());
    }

    //--------------------------------------------------------------------------
    template<int CDIM, int RDIM>
    ColorRects<CDIM,RDIM>::ColorRects(const DomainPointColoring &coloring, 
        LogicalRegion region, FieldID color_field, FieldID range_field)
      : TaskLauncher(TASK_ID, TaskArgument(), Predicate::TRUE_PRED,
                     PARTITION_SHIM_MAPPER_ID)
    //--------------------------------------------------------------------------
    {
      add_region_requirement(
          RegionRequirement(region, WRITE_DISCARD, EXCLUSIVE, region));
      add_field(0/*index*/, color_field);
      add_field(0/*index*/, range_field);
      // Serialize the coloring into the argument buffer 
      rez.serialize<size_t>(coloring.size());
      for (DomainPointColoring::const_iterator cit = coloring.begin();
            cit != coloring.end(); cit++)
      {
        const Point<CDIM,coord_t> color = cit->first;
        rez.serialize(color);
        rez.serialize<size_t>(1); // number of rects
        const Rect<RDIM,coord_t> rect = cit->second;
        rez.serialize(rect);
      }
      argument = TaskArgument(rez.get_buffer(), rez.get_used_bytes());
    }

    //--------------------------------------------------------------------------
    template<int CDIM, int RDIM>
    ColorRects<CDIM,RDIM>::ColorRects(const MultiDomainPointColoring &coloring,
        LogicalRegion region, FieldID color_field, FieldID range_field)
      : TaskLauncher(TASK_ID, TaskArgument(), Predicate::TRUE_PRED,
                     PARTITION_SHIM_MAPPER_ID)
    //--------------------------------------------------------------------------
    {
      add_region_requirement(
          RegionRequirement(region, WRITE_DISCARD, EXCLUSIVE, region));
      add_field(0/*index*/, color_field);
      add_field(0/*index*/, range_field);
      // Serialize the coloring into the argument buffer 
      rez.serialize<size_t>(coloring.size());
      for (MultiDomainPointColoring::const_iterator cit = coloring.begin();
            cit != coloring.end(); cit++)
      {
        const Point<CDIM,coord_t> color = cit->first;
        rez.serialize(color);
        rez.serialize<size_t>(cit->second.size());
        for (std::set<Domain>::const_iterator it = cit->second.begin();
              it != cit->second.end(); it++)
        {
          const Rect<RDIM,coord_t> rect = *it;
          rez.serialize(rect);
        }
      }
      argument = TaskArgument(rez.get_buffer(), rez.get_used_bytes());
    }

    //--------------------------------------------------------------------------
    template<int CDIM, int RDIM>
    ColorRects<CDIM,RDIM>::ColorRects(const Coloring &coloring, 
                 LogicalRegion region, FieldID color_field, FieldID range_field)
      : TaskLauncher(TASK_ID, TaskArgument(), Predicate::TRUE_PRED,
                     PARTITION_SHIM_MAPPER_ID)
    //--------------------------------------------------------------------------
    {
      add_region_requirement(
          RegionRequirement(region, WRITE_DISCARD, EXCLUSIVE, region));
      add_field(0/*index*/, color_field);
      add_field(0/*index*/, range_field);
      rez.serialize<size_t>(coloring.size());
      for (Coloring::const_iterator cit = coloring.begin(); 
            cit != coloring.end(); cit++)
      {
        const Point<CDIM,coord_t> color = DomainPoint(cit->first); 
        rez.serialize(color);
        // Count how many rectangles there will be
        size_t total_rects = cit->second.points.size();
        for (std::set<std::pair<ptr_t,ptr_t> >::const_iterator it = 
              cit->second.ranges.begin(); it != cit->second.ranges.end(); it++)
        {
          // Skip empty ranges
          if (it->first.value > it->second.value)
            continue;
          total_rects++;
        }
        rez.serialize(total_rects);
        for (std::set<ptr_t>::const_iterator it = 
              cit->second.points.begin(); it != cit->second.points.end(); it++)
        {
          const Point<1,coord_t> point(*it);
          const Rect<1,coord_t> rect(point, point);
          rez.serialize(rect);
        }
        for (std::set<std::pair<ptr_t,ptr_t> >::const_iterator it = 
              cit->second.ranges.begin(); it != cit->second.ranges.end(); it++)
        {
          // Skip empty ranges
          if (it->first.value > it->second.value)
            continue;
          const Point<1,coord_t> lo(it->first.value);
          const Point<1,coord_t> hi(it->second.value);
          const Rect<1,coord_t> rect(lo, hi);
          rez.serialize(rect);
        }
      }
      argument = TaskArgument(rez.get_buffer(), rez.get_used_bytes());
    }

    //--------------------------------------------------------------------------
    template<int CDIM, int RDIM>
    ColorRects<CDIM,RDIM>::ColorRects(const PointColoring &coloring, 
                 LogicalRegion region, FieldID color_field, FieldID range_field)
      : TaskLauncher(TASK_ID, TaskArgument(), Predicate::TRUE_PRED,
                     PARTITION_SHIM_MAPPER_ID)
    //--------------------------------------------------------------------------
    {
      add_region_requirement(
          RegionRequirement(region, WRITE_DISCARD, EXCLUSIVE, region));
      add_field(0/*index*/, color_field);
      add_field(0/*index*/, range_field);
      rez.serialize<size_t>(coloring.size());
      for (PointColoring::const_iterator cit = coloring.begin();
            cit != coloring.end(); cit++)
      {
        const Point<CDIM,coord_t> color = cit->first; 
        rez.serialize(color);
        // Count how many rectangles there will be
        size_t total_rects = cit->second.points.size();
        for (std::set<std::pair<ptr_t,ptr_t> >::const_iterator it = 
              cit->second.ranges.begin(); it != cit->second.ranges.end(); it++)
        {
          // Skip empty ranges
          if (it->first.value > it->second.value)
            continue;
          total_rects++;
        }
        rez.serialize(total_rects);
        for (std::set<ptr_t>::const_iterator it = 
              cit->second.points.begin(); it != cit->second.points.end(); it++)
        {
          const Point<1,coord_t> point(*it);
          const Rect<1,coord_t> rect(point, point);
          rez.serialize(rect);
        }
        for (std::set<std::pair<ptr_t,ptr_t> >::const_iterator it = 
              cit->second.ranges.begin(); it != cit->second.ranges.end(); it++)
        {
          // Skip empty ranges
          if (it->first.value > it->second.value)
            continue;
          const Point<1,coord_t> lo(it->first.value);
          const Point<1,coord_t> hi(it->second.value);
          const Rect<1,coord_t> rect(lo, hi);
          rez.serialize(rect);
        }
      }
      argument = TaskArgument(rez.get_buffer(), rez.get_used_bytes());
    }

    //--------------------------------------------------------------------------
    template<int CDIM, int RDIM>
    /*static*/ void ColorRects<CDIM,RDIM>::register_task(void)
    //--------------------------------------------------------------------------
    {
      TASK_ID = Legion::Runtime::generate_static_task_id();
      char variant_name[128];
      snprintf(variant_name,128,"Color Rects <%d,%d>", CDIM, RDIM);
      TaskVariantRegistrar registrar(TASK_ID, variant_name);
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Legion::Runtime::preregister_task_variant<
        ColorRects<CDIM,RDIM>::cpu_variant>(registrar, variant_name);
    }

    //--------------------------------------------------------------------------
    template<int CDIM, int RDIM>
    /*static*/ void ColorRects<CDIM,RDIM>::cpu_variant(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const FieldAccessor<WRITE_DISCARD,Point<CDIM,coord_t>,1,coord_t>
        fa_color(regions[0], task->regions[0].instance_fields[0]);
      const FieldAccessor<WRITE_DISCARD,Rect<RDIM,coord_t>,1,coord_t>
        fa_range(regions[0], task->regions[0].instance_fields[1]);
      Deserializer derez(task->args, task->arglen);
      size_t num_colors;
      derez.deserialize(num_colors);
      coord_t next_entry = 0;
      for (unsigned cidx = 0; cidx < num_colors; cidx++)
      {
        Point<CDIM,coord_t> color;
        derez.deserialize(color);
        size_t num_ranges;
        derez.deserialize(num_ranges);
        for (unsigned idx = 0; idx < num_ranges; idx++, next_entry++)
        {
          Rect<RDIM,coord_t> range;
          derez.deserialize(range);
          fa_color.write(next_entry, color);
          fa_range.write(next_entry, range);
        }
      }
    }

    // Do the explicit instantiation
    template class ColorPoints<1>;
    template class ColorPoints<2>;
    template class ColorPoints<3>;
    template class ColorRects<1,1>;
    template class ColorRects<1,2>;
    template class ColorRects<1,3>;
    template class ColorRects<2,1>;
    template class ColorRects<2,2>;
    template class ColorRects<2,3>;
    template class ColorRects<3,1>;
    template class ColorRects<3,2>;
    template class ColorRects<3,3>;
  };
#endif // DISABLE_PARTITION_SHIM

    namespace Internal {
      LEGION_EXTERN_LOGGER_DECLARATIONS
    };

    const LogicalRegion LogicalRegion::NO_REGION = LogicalRegion();
    const LogicalPartition LogicalPartition::NO_PART = LogicalPartition();  
    const Domain Domain::NO_DOMAIN = Domain();

    // Cache static type tags so we don't need to recompute them all the time
    static const TypeTag TYPE_TAG_1D =
      Internal::NT_TemplateHelper::encode_tag<1,coord_t>();
    static const TypeTag TYPE_TAG_2D =
      Internal::NT_TemplateHelper::encode_tag<2,coord_t>();
    static const TypeTag TYPE_TAG_3D =
      Internal::NT_TemplateHelper::encode_tag<3,coord_t>();

    /////////////////////////////////////////////////////////////
    // Mappable 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Mappable::Mappable(void)
      : map_id(0), tag(0), mapper_data(NULL), mapper_data_size(0)
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
    // Close 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Close::Close(void)
      : Mappable(), parent_task(NULL)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Fill 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Fill::Fill(void)
      : Mappable(), parent_task(NULL)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Partition
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Partition::Partition(void)
      : Mappable(), parent_task(NULL)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // IndexSpace 
    /////////////////////////////////////////////////////////////

    /*static*/ const IndexSpace IndexSpace::NO_SPACE = IndexSpace();

    //--------------------------------------------------------------------------
    IndexSpace::IndexSpace(IndexSpaceID _id, IndexTreeID _tid, TypeTag _tag)
      : id(_id), tid(_tid), type_tag(_tag)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpace::IndexSpace(void)
      : id(0), tid(0), type_tag(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpace::IndexSpace(const IndexSpace &rhs)
      : id(rhs.id), tid(rhs.tid), type_tag(rhs.type_tag)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // IndexPartition 
    /////////////////////////////////////////////////////////////

    /*static*/ const IndexPartition IndexPartition::NO_PART = IndexPartition();

    //--------------------------------------------------------------------------
    IndexPartition::IndexPartition(IndexPartitionID _id, 
                                   IndexTreeID _tid, TypeTag _tag)
      : id(_id), tid(_tid), type_tag(_tag)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexPartition::IndexPartition(void)
      : id(0), tid(0), type_tag(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexPartition::IndexPartition(const IndexPartition &rhs)
      : id(rhs.id), tid(rhs.tid), type_tag(rhs.type_tag)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // FieldSpace 
    /////////////////////////////////////////////////////////////

    /*static*/ const FieldSpace FieldSpace::NO_SPACE = FieldSpace(0);

    //--------------------------------------------------------------------------
    FieldSpace::FieldSpace(unsigned _id)
      : id(_id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldSpace::FieldSpace(void)
      : id(0)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    FieldSpace::FieldSpace(const FieldSpace &rhs)
      : id(rhs.id)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Logical Region  
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalRegion::LogicalRegion(RegionTreeID tid, IndexSpace index, 
                                 FieldSpace field)
      : tree_id(tid), index_space(index), field_space(field)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalRegion::LogicalRegion(void)
      : tree_id(0), index_space(IndexSpace::NO_SPACE), 
        field_space(FieldSpace::NO_SPACE)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalRegion::LogicalRegion(const LogicalRegion &rhs)
      : tree_id(rhs.tree_id), index_space(rhs.index_space), 
        field_space(rhs.field_space)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Logical Partition 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalPartition::LogicalPartition(RegionTreeID tid, IndexPartition pid, 
                                       FieldSpace field)
      : tree_id(tid), index_partition(pid), field_space(field)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalPartition::LogicalPartition(void)
      : tree_id(0), index_partition(IndexPartition::NO_PART), 
        field_space(FieldSpace::NO_SPACE)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalPartition::LogicalPartition(const LogicalPartition &rhs)
      : tree_id(rhs.tree_id), index_partition(rhs.index_partition), 
        field_space(rhs.field_space)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Field Allocator 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldAllocator::FieldAllocator(void)
      : field_space(FieldSpace::NO_SPACE), parent(NULL), runtime(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldAllocator::FieldAllocator(const FieldAllocator &allocator)
      : field_space(allocator.field_space), parent(allocator.parent), 
        runtime(allocator.runtime)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldAllocator::FieldAllocator(FieldSpace f, Context p, 
                                   Runtime *rt)
      : field_space(f), parent(p), runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldAllocator::~FieldAllocator(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldAllocator& FieldAllocator::operator=(const FieldAllocator &rhs)
    //--------------------------------------------------------------------------
    {
      field_space = rhs.field_space;
      parent = rhs.parent;
      runtime = rhs.runtime;
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Argument Map 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ArgumentMap::ArgumentMap(void)
    //--------------------------------------------------------------------------
    {
      impl = new Internal::ArgumentMapImpl();
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      impl->add_reference();
    }

    //--------------------------------------------------------------------------
    ArgumentMap::ArgumentMap(const FutureMap &rhs)
    //--------------------------------------------------------------------------
    {
      impl = new Internal::ArgumentMapImpl(rhs);
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      impl->add_reference();
    }

    //--------------------------------------------------------------------------
    ArgumentMap::ArgumentMap(const ArgumentMap &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    ArgumentMap::ArgumentMap(Internal::ArgumentMapImpl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    ArgumentMap::~ArgumentMap(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        // Remove our reference and if we were the
        // last reference holder, then delete it
        if (impl->remove_reference())
        {
          delete impl;
        }
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    ArgumentMap& ArgumentMap::operator=(const FutureMap &rhs)
    //--------------------------------------------------------------------------
    {
      // Check to see if our current impl is not NULL,
      // if so remove our reference
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          delete impl;
        }
      }
      impl = new Internal::ArgumentMapImpl(rhs);
      impl->add_reference();
      return *this;
    }
    
    //--------------------------------------------------------------------------
    ArgumentMap& ArgumentMap::operator=(const ArgumentMap &rhs)
    //--------------------------------------------------------------------------
    {
      // Check to see if our current impl is not NULL,
      // if so remove our reference
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          delete impl;
        }
      }
      impl = rhs.impl;
      // Add our reference to the new impl
      if (impl != NULL)
      {
        impl->add_reference();
      }
      return *this;
    }

    //--------------------------------------------------------------------------
    bool ArgumentMap::has_point(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      return impl->has_point(point);
    }

    //--------------------------------------------------------------------------
    void ArgumentMap::set_point(const DomainPoint &point, 
                                const TaskArgument &arg, bool replace/*= true*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      impl->set_point(point, arg, replace);
    }

    //--------------------------------------------------------------------------
    bool ArgumentMap::remove_point(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      return impl->remove_point(point);
    }

    //--------------------------------------------------------------------------
    TaskArgument ArgumentMap::get_point(const DomainPoint &point) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      return impl->get_point(point);
    }

    /////////////////////////////////////////////////////////////
    // Predicate 
    /////////////////////////////////////////////////////////////

    const Predicate Predicate::TRUE_PRED = Predicate(true);
    const Predicate Predicate::FALSE_PRED = Predicate(false);

    //--------------------------------------------------------------------------
    Predicate::Predicate(void)
      : impl(NULL), const_value(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Predicate::Predicate(const Predicate &p)
    //--------------------------------------------------------------------------
    {
      const_value = p.const_value;
      impl = p.impl;
      if (impl != NULL)
        impl->add_predicate_reference();
    }

    //--------------------------------------------------------------------------
    Predicate::Predicate(bool value)
      : impl(NULL), const_value(value)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Predicate::Predicate(Internal::PredicateImpl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_predicate_reference();
    }

    //--------------------------------------------------------------------------
    Predicate::~Predicate(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        impl->remove_predicate_reference();
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    Predicate& Predicate::operator=(const Predicate &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->remove_predicate_reference();
      const_value = rhs.const_value;
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_predicate_reference();
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Lock 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Lock::Lock(void)
      : reservation_lock(Reservation::NO_RESERVATION)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Lock::Lock(Reservation r)
      : reservation_lock(r)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool Lock::operator<(const Lock &rhs) const
    //--------------------------------------------------------------------------
    {
      return (reservation_lock < rhs.reservation_lock);
    }

    //--------------------------------------------------------------------------
    bool Lock::operator==(const Lock &rhs) const
    //--------------------------------------------------------------------------
    {
      return (reservation_lock == rhs.reservation_lock);
    }

    //--------------------------------------------------------------------------
    void Lock::acquire(unsigned mode /*=0*/, bool exclusive /*=true*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(reservation_lock.exists());
#endif
      Internal::ApEvent lock_event(reservation_lock.acquire(mode,exclusive));
      lock_event.wait();
    }

    //--------------------------------------------------------------------------
    void Lock::release(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(reservation_lock.exists());
#endif
      reservation_lock.release();
    }

    /////////////////////////////////////////////////////////////
    // Grant 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Grant::Grant(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Grant::Grant(Internal::GrantImpl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    Grant::Grant(const Grant &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    Grant::~Grant(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          delete impl;
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    Grant& Grant::operator=(const Grant &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          delete impl;
      }
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_reference();
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Phase Barrier 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhaseBarrier::PhaseBarrier(void)
      : phase_barrier(Internal::ApBarrier::NO_AP_BARRIER)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhaseBarrier::PhaseBarrier(Internal::ApBarrier b)
      : phase_barrier(b)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool PhaseBarrier::operator<(const PhaseBarrier &rhs) const
    //--------------------------------------------------------------------------
    {
      return (phase_barrier < rhs.phase_barrier);
    }

    //--------------------------------------------------------------------------
    bool PhaseBarrier::operator==(const PhaseBarrier &rhs) const
    //--------------------------------------------------------------------------
    {
      return (phase_barrier == rhs.phase_barrier);
    }

    //--------------------------------------------------------------------------
    bool PhaseBarrier::operator!=(const PhaseBarrier &rhs) const
    //--------------------------------------------------------------------------
    {
      return (phase_barrier != rhs.phase_barrier);
    }

    //--------------------------------------------------------------------------
    void PhaseBarrier::arrive(unsigned count /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(phase_barrier.exists());
#endif
      Internal::Runtime::phase_barrier_arrive(*this, count);
    }

    //--------------------------------------------------------------------------
    void PhaseBarrier::wait(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(phase_barrier.exists());
#endif
      Internal::ApEvent e = Internal::Runtime::get_previous_phase(*this);
      e.wait();
    }

    //--------------------------------------------------------------------------
    void PhaseBarrier::alter_arrival_count(int delta)
    //--------------------------------------------------------------------------
    {
      Internal::Runtime::alter_arrival_count(*this, delta);
    }

    //--------------------------------------------------------------------------
    bool PhaseBarrier::exists(void) const
    //--------------------------------------------------------------------------
    {
      return phase_barrier.exists();
    }

    /////////////////////////////////////////////////////////////
    // Dynamic Collective 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DynamicCollective::DynamicCollective(void)
      : PhaseBarrier(), redop(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DynamicCollective::DynamicCollective(Internal::ApBarrier b, ReductionOpID r)
      : PhaseBarrier(b), redop(r)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void DynamicCollective::arrive(const void *value, size_t size, 
                                   unsigned count /*=1*/)
    //--------------------------------------------------------------------------
    {
      Internal::Runtime::phase_barrier_arrive(*this, count, 
                                  Internal::ApEvent::NO_AP_EVENT, value, size);
    }

    /////////////////////////////////////////////////////////////
    // Region Requirement 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                                        const std::set<FieldID> &priv_fields,
                                        const std::vector<FieldID> &inst_fields,
                                        PrivilegeMode _priv, 
                                        CoherenceProperty _prop, 
                                        LogicalRegion _parent,
				        MappingTagID _tag, bool _verified)
      : region(_handle), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(SINGULAR), projection(0)
    //--------------------------------------------------------------------------
    { 
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
      // For backwards compatibility with the old encoding
      if (privilege == WRITE_PRIV)
        privilege = WRITE_DISCARD;
#ifdef DEBUG_LEGION
      if (IS_REDUCE(*this)) // Shouldn't use this constructor for reductions
        REPORT_LEGION_ERROR(ERROR_USE_REDUCTION_REGION_REQ, 
                                   "Use different RegionRequirement "
                            "constructor for reductions");
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalPartition pid, 
                ProjectionID _proj, 
                const std::set<FieldID> &priv_fields,
                const std::vector<FieldID> &inst_fields,
                PrivilegeMode _priv, CoherenceProperty _prop,
                LogicalRegion _parent, MappingTagID _tag, bool _verified)
      : partition(pid), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG),
        handle_type(PART_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    { 
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
      // For backwards compatibility with the old encoding
      if (privilege == WRITE_PRIV)
        privilege = WRITE_DISCARD;
#ifdef DEBUG_LEGION
      if (IS_REDUCE(*this))
        REPORT_LEGION_ERROR(ERROR_USE_REDUCTION_REGION_REQ, 
                                   "Use different RegionRequirement "
                            "constructor for reductions");
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                ProjectionID _proj,
                const std::set<FieldID> &priv_fields,
                const std::vector<FieldID> &inst_fields,
                PrivilegeMode _priv, CoherenceProperty _prop,
                LogicalRegion _parent, MappingTagID _tag, bool _verified)
      : region(_handle), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG),
        handle_type(REG_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
      // For backwards compatibility with the old encoding
      if (privilege == WRITE_PRIV)
        privilege = WRITE_DISCARD;
#ifdef DEBUG_LEGION
      if (IS_REDUCE(*this))
        REPORT_LEGION_ERROR(ERROR_USE_REDUCTION_REGION_REQ, 
                                   "Use different RegionRequirement "
                                   "constructor for reductions")
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle,  
                                    const std::set<FieldID> &priv_fields,
                                    const std::vector<FieldID> &inst_fields,
                                    ReductionOpID op, CoherenceProperty _prop, 
                                    LogicalRegion _parent, MappingTagID _tag, 
                                    bool _verified)
      : region(_handle), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(SINGULAR)
    //--------------------------------------------------------------------------
    {
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_LEGION
      if (redop == 0)
        REPORT_LEGION_ERROR(ERROR_RESERVED_REDOP_ID, 
                                   "Zero is not a valid ReductionOpID")
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalPartition pid, 
                        ProjectionID _proj,  
                        const std::set<FieldID> &priv_fields,
                        const std::vector<FieldID> &inst_fields,
                        ReductionOpID op, CoherenceProperty _prop,
                        LogicalRegion _parent, MappingTagID _tag, 
                        bool _verified)
      : partition(pid), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG),
        handle_type(PART_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_LEGION
      if (redop == 0)
        REPORT_LEGION_ERROR(ERROR_RESERVED_REDOP_ID, 
                                   "Zero is not a valid ReductionOpID")        
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                        ProjectionID _proj,
                        const std::set<FieldID> &priv_fields,
                        const std::vector<FieldID> &inst_fields,
                        ReductionOpID op, CoherenceProperty _prop,
                        LogicalRegion _parent, MappingTagID _tag, 
                        bool _verified)
      : region(_handle), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG),
        handle_type(REG_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_LEGION
      if (redop == 0)
        REPORT_LEGION_ERROR(ERROR_RESERVED_REDOP_ID, 
                                   "Zero is not a valid ReductionOpID")
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                                         PrivilegeMode _priv, 
                                         CoherenceProperty _prop, 
                                         LogicalRegion _parent,
					 MappingTagID _tag, 
                                         bool _verified)
      : region(_handle), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(SINGULAR)
    //--------------------------------------------------------------------------
    { 
      // For backwards compatibility with the old encoding
      if (privilege == WRITE_PRIV)
        privilege = WRITE_DISCARD;
#ifdef DEBUG_LEGION
      if (IS_REDUCE(*this)) // Shouldn't use this constructor for reductions
        REPORT_LEGION_ERROR(ERROR_USE_REDUCTION_REGION_REQ, 
                                   "Use different RegionRequirement "
                                   "constructor for reductions")
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalPartition pid, 
                                         ProjectionID _proj, 
                                         PrivilegeMode _priv, 
                                         CoherenceProperty _prop,
                                         LogicalRegion _parent, 
                                         MappingTagID _tag, 
                                         bool _verified)
      : partition(pid), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(PART_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    { 
      // For backwards compatibility with the old encoding
      if (privilege == WRITE_PRIV)
        privilege = WRITE_DISCARD;
#ifdef DEBUG_LEGION
      if (IS_REDUCE(*this))
        REPORT_LEGION_ERROR(ERROR_USE_REDUCTION_REGION_REQ, 
                                   "Use different RegionRequirement "
                                   "constructor for reductions")
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                                         ProjectionID _proj,
                                         PrivilegeMode _priv, 
                                         CoherenceProperty _prop,
                                         LogicalRegion _parent, 
                                         MappingTagID _tag, 
                                         bool _verified)
      : region(_handle), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(REG_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
      // For backwards compatibility with the old encoding
      if (privilege == WRITE_PRIV)
        privilege = WRITE_DISCARD;
#ifdef DEBUG_LEGION
      if (IS_REDUCE(*this))
        REPORT_LEGION_ERROR(ERROR_USE_REDUCTION_REGION_REQ, 
                                   "Use different RegionRequirement "
                                   "constructor for reductions")
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle,  
                                         ReductionOpID op, 
                                         CoherenceProperty _prop, 
                                         LogicalRegion _parent, 
                                         MappingTagID _tag, 
                                         bool _verified)
      : region(_handle), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(SINGULAR)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (redop == 0)
        REPORT_LEGION_ERROR(ERROR_RESERVED_REDOP_ID, 
                                   "Zero is not a valid ReductionOpID")        
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalPartition pid, 
                                         ProjectionID _proj,  
                                         ReductionOpID op, 
                                         CoherenceProperty _prop,
                                         LogicalRegion _parent, 
                                         MappingTagID _tag, 
                                         bool _verified)
      : partition(pid), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(PART_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (redop == 0)
        REPORT_LEGION_ERROR(ERROR_RESERVED_REDOP_ID, 
                                   "Zero is not a valid ReductionOpID")
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                                         ProjectionID _proj,
                                         ReductionOpID op, 
                                         CoherenceProperty _prop,
                                         LogicalRegion _parent, 
                                         MappingTagID _tag, 
                                         bool _verified)
      : region(_handle), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(REG_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (redop == 0)
        REPORT_LEGION_ERROR(ERROR_RESERVED_REDOP_ID, 
                                   "Zero is not a valid ReductionOpID")
#endif
    }

    //--------------------------------------------------------------------------
    bool RegionRequirement::operator==(const RegionRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      if ((handle_type == rhs.handle_type) && (privilege == rhs.privilege) &&
          (prop == rhs.prop) && (parent == rhs.parent) && (redop == rhs.redop)
          && (tag == rhs.tag) && (flags == rhs.flags))
      {
        if (((handle_type == SINGULAR) && (region == rhs.region)) ||
            ((handle_type == PART_PROJECTION) && (partition == rhs.partition) &&
             (projection == rhs.projection)) ||
            ((handle_type == REG_PROJECTION) && (region == rhs.region)))
        {
          if ((privilege_fields.size() == rhs.privilege_fields.size()) &&
              (instance_fields.size() == rhs.instance_fields.size()))
          {
            return ((privilege_fields == rhs.privilege_fields) 
                && (instance_fields == rhs.instance_fields));
          }
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionRequirement::operator<(const RegionRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      if (handle_type < rhs.handle_type)
        return true;
      else if (handle_type > rhs.handle_type)
        return false;
      else
      {
        if (privilege < rhs.privilege)
          return true;
        else if (privilege > rhs.privilege)
          return false;
        else
        {
          if (prop < rhs.prop)
            return true;
          else if (prop > rhs.prop)
            return false;
          else
          {
            if (parent < rhs.parent)
              return true;
            else if (!(parent == rhs.parent)) // therefore greater than
              return false;
            else
            {
              if (redop < rhs.redop)
                return true;
              else if (redop > rhs.redop)
                return false;
              else
              {
                if (tag < rhs.tag)
                  return true;
                else if (tag > rhs.tag)
                  return false;
                else
                {
                  if (flags < rhs.flags)
                    return true;
                  else if (flags > rhs.flags)
                    return false;
                  else
                  {
                    if (privilege_fields < rhs.privilege_fields)
                      return true;
                    else if (privilege_fields > rhs.privilege_fields)
                      return false;
                    else
                    {
                      if (instance_fields < rhs.instance_fields)
                        return true;
                      else if (instance_fields > rhs.instance_fields)
                        return false;
                      else
                      {
                        if (handle_type == SINGULAR)
                          return (region < rhs.region);
                        else if (handle_type == PART_PROJECTION)
                        {
                          if (partition < rhs.partition)
                            return true;
                          // therefore greater than
                          else if (partition != rhs.partition) 
                            return false;
                          else
                            return (projection < rhs.projection);
                        }
                        else
                        {
                          if (region < rhs.region)
                            return true;
                          else if (region != rhs.region)
                            return false;
                          else
                            return (projection < rhs.projection);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

#ifdef PRIVILEGE_CHECKS
    //--------------------------------------------------------------------------
    unsigned RegionRequirement::get_accessor_privilege(void) const
    //--------------------------------------------------------------------------
    {
      switch (privilege)
      {
        case NO_ACCESS:
          return LegionRuntime::ACCESSOR_NONE;
        case READ_ONLY:
          return LegionRuntime::ACCESSOR_READ;
        case READ_WRITE:
        case WRITE_DISCARD:
          return LegionRuntime::ACCESSOR_ALL;
        case REDUCE:
          return LegionRuntime::ACCESSOR_REDUCE;
        default:
          assert(false);
      }
      return LegionRuntime::ACCESSOR_NONE;
    }
#endif

    //--------------------------------------------------------------------------
    bool RegionRequirement::has_field_privilege(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      return (privilege_fields.find(fid) != privilege_fields.end());
    }

    /////////////////////////////////////////////////////////////
    // Index Space Requirement 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexSpaceRequirement::IndexSpaceRequirement(void)
      : handle(IndexSpace::NO_SPACE), privilege(NO_MEMORY), 
        parent(IndexSpace::NO_SPACE), verified(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpaceRequirement::IndexSpaceRequirement(IndexSpace _handle, 
                                                 AllocateMode _priv,
                                                 IndexSpace _parent, 
                                                 bool _verified /*=false*/)
      : handle(_handle), privilege(_priv), parent(_parent), verified(_verified)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceRequirement::operator<(
                                        const IndexSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      if (handle < rhs.handle)
        return true;
      else if (handle != rhs.handle) // therefore greater than
        return false;
      else
      {
        if (privilege < rhs.privilege)
          return true;
        else if (privilege > rhs.privilege)
          return false;
        else
        {
          if (parent < rhs.parent)
            return true;
          else if (parent != rhs.parent) // therefore greater than
            return false;
          else
            return verified < rhs.verified;
        }
      }
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceRequirement::operator==(
                                        const IndexSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      return (handle == rhs.handle) && (privilege == rhs.privilege) &&
             (parent == rhs.parent) && (verified == rhs.verified);
    }

    /////////////////////////////////////////////////////////////
    // Field Space Requirement 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldSpaceRequirement::FieldSpaceRequirement(void)
      : handle(FieldSpace::NO_SPACE), privilege(NO_MEMORY), verified(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldSpaceRequirement::FieldSpaceRequirement(FieldSpace _handle, 
                                                 AllocateMode _priv,
                                                 bool _verified /*=false*/)
      : handle(_handle), privilege(_priv), verified(_verified)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceRequirement::operator<(
                                        const FieldSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      if (handle < rhs.handle)
        return true;
      else if (!(handle == rhs.handle)) // therefore greater than
        return false;
      else
      {
        if (privilege < rhs.privilege)
          return true;
        else if (privilege > rhs.privilege)
          return false;
        else
          return verified < rhs.verified;
      }
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceRequirement::operator==(
                                        const FieldSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      return (handle == rhs.handle) && 
              (privilege == rhs.privilege) && (verified == rhs.verified);
    }

    /////////////////////////////////////////////////////////////
    // StaticDependence 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    StaticDependence::StaticDependence(void)
      : previous_offset(0), previous_req_index(0), current_req_index(0),
        dependence_type(NO_DEPENDENCE), validates(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    StaticDependence::StaticDependence(unsigned prev, unsigned prev_req,
                           unsigned current_req, DependenceType dtype, bool val)
      : previous_offset(prev), previous_req_index(prev_req),
        current_req_index(current_req), dependence_type(dtype), validates(val)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // TaskLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskLauncher::TaskLauncher(void)
      : task_id(0), argument(TaskArgument()), predicate(Predicate::TRUE_PRED),
        map_id(0), tag(0), point(DomainPoint()), static_dependences(NULL),
        enable_inlining(false), independent_requirements(false), 
        silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TaskLauncher::TaskLauncher(Processor::TaskFuncID tid, TaskArgument arg,
                               Predicate pred /*= Predicate::TRUE_PRED*/,
                               MapperID mid /*=0*/, MappingTagID t /*=0*/)
      : task_id(tid), argument(arg), predicate(pred), map_id(mid), tag(t), 
        point(DomainPoint()), static_dependences(NULL), enable_inlining(false),
        independent_requirements(false), silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // IndexTaskLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexTaskLauncher::IndexTaskLauncher(void)
      : task_id(0), launch_domain(Domain::NO_DOMAIN), 
        launch_space(IndexSpace::NO_SPACE), global_arg(TaskArgument()), 
        argument_map(ArgumentMap()), predicate(Predicate::TRUE_PRED), 
        must_parallelism(false), map_id(0), tag(0), static_dependences(NULL), 
        enable_inlining(false), independent_requirements(false), 
        silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexTaskLauncher::IndexTaskLauncher(Processor::TaskFuncID tid, Domain dom,
                                     TaskArgument global,
                                     ArgumentMap map,
                                     Predicate pred /*= Predicate::TRUE_PRED*/,
                                     bool must /*=false*/, MapperID mid /*=0*/,
                                     MappingTagID t /*=0*/)
      : task_id(tid), launch_domain(dom), launch_space(IndexSpace::NO_SPACE),
        global_arg(global), argument_map(map), predicate(pred), 
        must_parallelism(must), map_id(mid), tag(t), static_dependences(NULL),
        enable_inlining(false), independent_requirements(false), 
        silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexTaskLauncher::IndexTaskLauncher(Processor::TaskFuncID tid, 
                                     IndexSpace space,
                                     TaskArgument global,
                                     ArgumentMap map,
                                     Predicate pred /*= Predicate::TRUE_PRED*/,
                                     bool must /*=false*/, MapperID mid /*=0*/,
                                     MappingTagID t /*=0*/)
      : task_id(tid), launch_domain(Domain::NO_DOMAIN), launch_space(space),
        global_arg(global), argument_map(map), predicate(pred), 
        must_parallelism(must), map_id(mid), tag(t), static_dependences(NULL),
        enable_inlining(false), independent_requirements(false), 
        silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // InlineLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InlineLauncher::InlineLauncher(void)
      : map_id(0), tag(0), layout_constraint_id(0), static_dependences(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InlineLauncher::InlineLauncher(const RegionRequirement &req,
                                   MapperID mid /*=0*/, MappingTagID t /*=0*/,
                                   LayoutConstraintID lay_id /*=0*/)
      : requirement(req), map_id(mid), tag(t), layout_constraint_id(lay_id),
        static_dependences(NULL)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // CopyLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CopyLauncher::CopyLauncher(Predicate pred /*= Predicate::TRUE_PRED*/,
                               MapperID mid /*=0*/, MappingTagID t /*=0*/)
      : predicate(pred), map_id(mid), tag(t), static_dependences(NULL), 
        silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // IndexCopyLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexCopyLauncher::IndexCopyLauncher(void) 
      : launch_domain(Domain::NO_DOMAIN), launch_space(IndexSpace::NO_SPACE),
        predicate(Predicate::TRUE_PRED), map_id(0), tag(0),
        static_dependences(NULL), silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexCopyLauncher::IndexCopyLauncher(Domain dom, 
                                    Predicate pred /*= Predicate::TRUE_PRED*/,
                                    MapperID mid /*=0*/, MappingTagID t /*=0*/) 
      : launch_domain(dom), launch_space(IndexSpace::NO_SPACE), predicate(pred),
        map_id(mid),tag(t), static_dependences(NULL), silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexCopyLauncher::IndexCopyLauncher(IndexSpace space, 
                                    Predicate pred /*= Predicate::TRUE_PRED*/,
                                    MapperID mid /*=0*/, MappingTagID t /*=0*/) 
      : launch_domain(Domain::NO_DOMAIN), launch_space(space), predicate(pred),
        map_id(mid), tag(t), static_dependences(NULL), silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // AcquireLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AcquireLauncher::AcquireLauncher(LogicalRegion reg, LogicalRegion par,
                                     PhysicalRegion phy,
                                     Predicate pred /*= Predicate::TRUE_PRED*/,
                                     MapperID id /*=0*/, MappingTagID t /*=0*/)
      : logical_region(reg), parent_region(par), physical_region(phy), 
        predicate(pred), map_id(id), tag(t), static_dependences(NULL),
        silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // ReleaseLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReleaseLauncher::ReleaseLauncher(LogicalRegion reg, LogicalRegion par,
                                     PhysicalRegion phy,
                                     Predicate pred /*= Predicate::TRUE_PRED*/,
                                     MapperID id /*=0*/, MappingTagID t /*=0*/)
      : logical_region(reg), parent_region(par), physical_region(phy), 
        predicate(pred), map_id(id), tag(t), static_dependences(NULL),
        silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // FillLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FillLauncher::FillLauncher(void)
      : handle(LogicalRegion::NO_REGION), parent(LogicalRegion::NO_REGION),
        map_id(0), tag(0), static_dependences(NULL), silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FillLauncher::FillLauncher(LogicalRegion h, LogicalRegion p,
                               TaskArgument arg, 
                               Predicate pred /*= Predicate::TRUE_PRED*/,
                               MapperID id /*=0*/, MappingTagID t /*=0*/)
      : handle(h), parent(p), argument(arg), predicate(pred), map_id(id), 
        tag(t), static_dependences(NULL), silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FillLauncher::FillLauncher(LogicalRegion h, LogicalRegion p, Future f,
                               Predicate pred /*= Predicate::TRUE_PRED*/,
                               MapperID id /*=0*/, MappingTagID t /*=0*/)
      : handle(h), parent(p), future(f), predicate(pred), map_id(id), tag(t), 
        static_dependences(NULL), silence_warnings(false) 
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // IndexFillLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexFillLauncher::IndexFillLauncher(void)
      : launch_domain(Domain::NO_DOMAIN), launch_space(IndexSpace::NO_SPACE),
        region(LogicalRegion::NO_REGION), partition(LogicalPartition::NO_PART), 
        projection(0), map_id(0), tag(0), static_dependences(NULL), 
        silence_warnings(false) 
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexFillLauncher::IndexFillLauncher(Domain dom, LogicalRegion h, 
                               LogicalRegion p, TaskArgument arg, 
                               ProjectionID proj, Predicate pred,
                               MapperID id /*=0*/, MappingTagID t /*=0*/)
      : launch_domain(dom), launch_space(IndexSpace::NO_SPACE), region(h), 
        partition(LogicalPartition::NO_PART), parent(p), projection(proj), 
        argument(arg), predicate(pred), map_id(id), tag(t), 
        static_dependences(NULL), silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexFillLauncher::IndexFillLauncher(Domain dom, LogicalRegion h,
                                LogicalRegion p, Future f,
                                ProjectionID proj, Predicate pred,
                                MapperID id /*=0*/, MappingTagID t /*=0*/)
      : launch_domain(dom), launch_space(IndexSpace::NO_SPACE), region(h), 
        partition(LogicalPartition::NO_PART), parent(p), projection(proj), 
        future(f), predicate(pred), map_id(id), tag(t), 
        static_dependences(NULL), silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexFillLauncher::IndexFillLauncher(IndexSpace space, LogicalRegion h, 
                               LogicalRegion p, TaskArgument arg, 
                               ProjectionID proj, Predicate pred,
                               MapperID id /*=0*/, MappingTagID t /*=0*/)
      : launch_domain(Domain::NO_DOMAIN), launch_space(space), region(h), 
        partition(LogicalPartition::NO_PART), parent(p), projection(proj), 
        argument(arg), predicate(pred), map_id(id), tag(t), 
        static_dependences(NULL), silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexFillLauncher::IndexFillLauncher(IndexSpace space, LogicalRegion h,
                                LogicalRegion p, Future f,
                                ProjectionID proj, Predicate pred,
                                MapperID id /*=0*/, MappingTagID t /*=0*/)
      : launch_domain(Domain::NO_DOMAIN), launch_space(space), region(h), 
        partition(LogicalPartition::NO_PART), parent(p), projection(proj), 
        future(f), predicate(pred), map_id(id), tag(t), 
        static_dependences(NULL), silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexFillLauncher::IndexFillLauncher(Domain dom, LogicalPartition h,
                                         LogicalRegion p, TaskArgument arg,
                                         ProjectionID proj, Predicate pred,
                                         MapperID id /*=0*/, 
                                         MappingTagID t /*=0*/)
      : launch_domain(dom), launch_space(IndexSpace::NO_SPACE), 
        region(LogicalRegion::NO_REGION), partition(h),
        parent(p), projection(proj), argument(arg), predicate(pred),
        map_id(id), tag(t), static_dependences(NULL), silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexFillLauncher::IndexFillLauncher(Domain dom, LogicalPartition h,
                                         LogicalRegion p, Future f,
                                         ProjectionID proj, Predicate pred,
                                         MapperID id /*=0*/, 
                                         MappingTagID t /*=0*/)
      : launch_domain(dom), launch_space(IndexSpace::NO_SPACE), 
        region(LogicalRegion::NO_REGION), partition(h),
        parent(p), projection(proj), future(f), predicate(pred),
        map_id(id), tag(t), static_dependences(NULL), silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexFillLauncher::IndexFillLauncher(IndexSpace space, LogicalPartition h,
                                         LogicalRegion p, TaskArgument arg,
                                         ProjectionID proj, Predicate pred,
                                         MapperID id /*=0*/, 
                                         MappingTagID t /*=0*/)
      : launch_domain(Domain::NO_DOMAIN), launch_space(space), 
        region(LogicalRegion::NO_REGION), partition(h),
        parent(p), projection(proj), argument(arg), predicate(pred),
        map_id(id), tag(t), static_dependences(NULL), silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexFillLauncher::IndexFillLauncher(IndexSpace space, LogicalPartition h,
                                         LogicalRegion p, Future f,
                                         ProjectionID proj, Predicate pred,
                                         MapperID id /*=0*/, 
                                         MappingTagID t /*=0*/)
      : launch_domain(Domain::NO_DOMAIN), launch_space(space), 
        region(LogicalRegion::NO_REGION), partition(h),
        parent(p), projection(proj), future(f), predicate(pred),
        map_id(id), tag(t), static_dependences(NULL), silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // AttachLauncher
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AttachLauncher::AttachLauncher(ExternalResource r, 
                                   LogicalRegion h, LogicalRegion p)
      : resource(r), handle(h), parent(p), 
        file_name(NULL), mode(LEGION_FILE_READ_ONLY), static_dependences(NULL)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // PredicateLauncher
    /////////////////////////////////////////////////////////////


    //--------------------------------------------------------------------------
    PredicateLauncher::PredicateLauncher(bool and_)
      : and_op(and_)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // TimingLauncher
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TimingLauncher::TimingLauncher(TimingMeasurement m)
      : measurement(m)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // MustEpochLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochLauncher::MustEpochLauncher(MapperID id /*= 0*/,   
                                         MappingTagID tag/*= 0*/)
      : map_id(id), mapping_tag(tag), silence_warnings(false)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // LayoutConstraintRegistrar
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LayoutConstraintRegistrar::LayoutConstraintRegistrar(void)
      : handle(FieldSpace::NO_SPACE), layout_name(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LayoutConstraintRegistrar::LayoutConstraintRegistrar(FieldSpace h,
                                                  const char *layout/*= NULL*/)
      : handle(h), layout_name(layout)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // TaskVariantRegistrar 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskVariantRegistrar::TaskVariantRegistrar(void)
      : task_id(0), global_registration(true), 
        task_variant_name(NULL), leaf_variant(false), 
        inner_variant(false), idempotent_variant(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TaskVariantRegistrar::TaskVariantRegistrar(TaskID task_id, bool global,
                                               const char *variant_name)
      : task_id(task_id), global_registration(global), 
        task_variant_name(variant_name), leaf_variant(false), 
        inner_variant(false), idempotent_variant(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TaskVariantRegistrar::TaskVariantRegistrar(TaskID task_id,
					       const char *variant_name,
					       bool global/*=true*/)
      : task_id(task_id), global_registration(global), 
        task_variant_name(variant_name), leaf_variant(false), 
        inner_variant(false), idempotent_variant(false)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // MPILegionHandshake 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MPILegionHandshake::MPILegionHandshake(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MPILegionHandshake::MPILegionHandshake(const MPILegionHandshake &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    MPILegionHandshake::~MPILegionHandshake(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          delete impl;
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    MPILegionHandshake::MPILegionHandshake(Internal::MPILegionHandshakeImpl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    MPILegionHandshake& MPILegionHandshake::operator=(
                                                  const MPILegionHandshake &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          delete impl;
      }
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshake::mpi_handoff_to_legion(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      impl->mpi_handoff_to_legion();
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshake::mpi_wait_on_legion(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      impl->mpi_wait_on_legion();
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshake::legion_handoff_to_mpi(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      impl->legion_handoff_to_mpi();
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshake::legion_wait_on_mpi(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      impl->legion_wait_on_mpi();
    }

    //--------------------------------------------------------------------------
    PhaseBarrier MPILegionHandshake::get_legion_wait_phase_barrier(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      return impl->get_legion_wait_phase_barrier();
    }

    //--------------------------------------------------------------------------
    PhaseBarrier MPILegionHandshake::get_legion_arrive_phase_barrier(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      return impl->get_legion_arrive_phase_barrier();
    }
    
    //--------------------------------------------------------------------------
    void MPILegionHandshake::advance_legion_handshake(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      impl->advance_legion_handshake();
    }

    /////////////////////////////////////////////////////////////
    // Future 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Future::Future(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Future::Future(const Future &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_base_gc_ref(Internal::FUTURE_HANDLE_REF);
    }

    //--------------------------------------------------------------------------
    Future::~Future(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_base_gc_ref(Internal::FUTURE_HANDLE_REF))
          delete impl;
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    Future::Future(Internal::FutureImpl *i, bool need_reference)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if ((impl != NULL) && need_reference)
        impl->add_base_gc_ref(Internal::FUTURE_HANDLE_REF);
    }

    //--------------------------------------------------------------------------
    Future& Future::operator=(const Future &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_base_gc_ref(Internal::FUTURE_HANDLE_REF))
          delete impl;
      }
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_base_gc_ref(Internal::FUTURE_HANDLE_REF);
      return *this;
    }

    //--------------------------------------------------------------------------
    void Future::get_void_result(bool silence_warnings) const
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->get_void_result(silence_warnings);
    }

    //--------------------------------------------------------------------------
    bool Future::is_empty(bool block /*= true*/, 
                          bool silence_warnings/*=false*/) const
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        return impl->is_empty(block, silence_warnings);
      return true;
    }

    //--------------------------------------------------------------------------
    bool Future::is_ready(void) const
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        return impl->is_ready();
      return true; // Empty futures are always ready
    }

    //--------------------------------------------------------------------------
    void* Future::get_untyped_result(bool silence_warnings) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        REPORT_LEGION_ERROR(ERROR_REQUEST_FOR_EMPTY_FUTURE, 
                          "Illegal request for future value from empty future")
      return impl->get_untyped_result(silence_warnings);
    }

    //--------------------------------------------------------------------------
    size_t Future::get_untyped_size(void)
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        REPORT_LEGION_ERROR(ERROR_REQUEST_FOR_EMPTY_FUTURE, 
                          "Illegal request for future size from empty future");
      return impl->get_untyped_size();
    }

    /////////////////////////////////////////////////////////////
    // Future Map 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureMap::FutureMap(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureMap::FutureMap(const FutureMap &map)
      : impl(map.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_base_gc_ref(Internal::FUTURE_HANDLE_REF);
    }

    //--------------------------------------------------------------------------
    FutureMap::FutureMap(Internal::FutureMapImpl *i, bool need_reference)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if ((impl != NULL) && need_reference)
        impl->add_base_gc_ref(Internal::FUTURE_HANDLE_REF);
    }

    //--------------------------------------------------------------------------
    FutureMap::~FutureMap(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_base_gc_ref(Internal::FUTURE_HANDLE_REF))
          delete impl;
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    FutureMap& FutureMap::operator=(const FutureMap &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_base_gc_ref(Internal::FUTURE_HANDLE_REF))
          delete impl;
      }
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_base_gc_ref(Internal::FUTURE_HANDLE_REF);
      return *this;
    }

    //--------------------------------------------------------------------------
    Future FutureMap::get_future(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      return impl->get_future(point);
    }

    //--------------------------------------------------------------------------
    void FutureMap::get_void_result(const DomainPoint &point, 
                                    bool silence_warnings)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->get_void_result(point, silence_warnings);
    }

    //--------------------------------------------------------------------------
    void FutureMap::wait_all_results(bool silence_warnings)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->wait_all_results(silence_warnings);
    }

    /////////////////////////////////////////////////////////////
    // Physical Region 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalRegion::PhysicalRegion(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::PhysicalRegion(const PhysicalRegion &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::PhysicalRegion(Internal::PhysicalRegionImpl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::~PhysicalRegion(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          delete impl;
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    PhysicalRegion& PhysicalRegion::operator=(const PhysicalRegion &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          delete impl;
      }
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegion::is_mapped(void) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
        return false;
      return impl->is_mapped();
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::wait_until_valid(bool silence_warnings)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      impl->wait_until_valid(silence_warnings);
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegion::is_valid(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      return impl->is_valid();
    }

    //--------------------------------------------------------------------------
    LogicalRegion PhysicalRegion::get_logical_region(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      return impl->get_logical_region();
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        PhysicalRegion::get_accessor(bool silence_warnings) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      return impl->get_accessor(silence_warnings);
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        PhysicalRegion::get_field_accessor(FieldID fid, 
                                           bool silence_warnings) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      return impl->get_field_accessor(fid, silence_warnings);
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::get_memories(std::set<Memory>& memories) const
    //--------------------------------------------------------------------------
    {
      impl->get_memories(memories);
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::get_fields(std::vector<FieldID>& fields) const
    //--------------------------------------------------------------------------
    {
      impl->get_fields(fields);
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::get_bounds(void *realm_is, TypeTag type_tag) const 
    //--------------------------------------------------------------------------
    {
      impl->get_bounds(realm_is, type_tag);
    }

    //--------------------------------------------------------------------------
    Realm::RegionInstance PhysicalRegion::get_instance_info(PrivilegeMode mode,
                              FieldID fid, size_t field_size, void *realm_is, 
                              TypeTag type_tag, bool silence_warnings, 
                              bool generic_accessor, bool check_field_size,
                              ReductionOpID redop) const
    //--------------------------------------------------------------------------
    {
      return impl->get_instance_info(mode, fid, field_size, realm_is, type_tag, 
                   silence_warnings, generic_accessor, check_field_size, redop);
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::fail_bounds_check(DomainPoint p, FieldID fid,
                                           PrivilegeMode mode) const
    //--------------------------------------------------------------------------
    {
      impl->fail_bounds_check(p, fid, mode);
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::fail_bounds_check(Domain d, FieldID fid,
                                           PrivilegeMode mode) const
    //--------------------------------------------------------------------------
    {
      impl->fail_bounds_check(d, fid, mode);
    }

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    /////////////////////////////////////////////////////////////
    // Index Iterator  
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexIterator::IndexIterator(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexIterator::IndexIterator(const Domain &dom, ptr_t start)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dom.get_dim() == 1);
#endif
      const DomainT<1,coord_t> is = dom;
      is_iterator = Realm::IndexSpaceIterator<1,coord_t>(is);
    }

    //--------------------------------------------------------------------------
    IndexIterator::IndexIterator(Runtime *rt, Context ctx,
                                 IndexSpace space, ptr_t start)
    //--------------------------------------------------------------------------
    {
      Domain dom = rt->get_index_space_domain(ctx, space);
#ifdef DEBUG_LEGION
      assert(dom.get_dim() == 1);
#endif
      const DomainT<1,coord_t> is = dom;
      is_iterator = Realm::IndexSpaceIterator<1,coord_t>(is);
    }

    //--------------------------------------------------------------------------
    IndexIterator::IndexIterator(Runtime *rt, Context ctx,
                                 LogicalRegion handle, ptr_t start)
    //--------------------------------------------------------------------------
    {
      Domain dom = rt->get_index_space_domain(ctx, handle.get_index_space());
#ifdef DEBUG_LEGION
      assert(dom.get_dim() == 1);
#endif
      const DomainT<1,coord_t> is = dom;
      is_iterator = Realm::IndexSpaceIterator<1,coord_t>(is);
    }

    //--------------------------------------------------------------------------
    IndexIterator::IndexIterator(Runtime *rt, IndexSpace space, ptr_t start)
    //--------------------------------------------------------------------------
    {
      Domain dom = rt->get_index_space_domain(space);
#ifdef DEBUG_LEGION
      assert(dom.get_dim() == 1);
#endif
      const DomainT<1,coord_t> is = dom;
      is_iterator = Realm::IndexSpaceIterator<1,coord_t>(is);
    }

    //--------------------------------------------------------------------------
    IndexIterator::IndexIterator(const IndexIterator &rhs)
      : is_iterator(rhs.is_iterator), rect_iterator(rhs.rect_iterator)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexIterator::~IndexIterator(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexIterator& IndexIterator::operator=(const IndexIterator &rhs)
    //--------------------------------------------------------------------------
    {
      is_iterator = rhs.is_iterator;
      rect_iterator = rhs.rect_iterator;
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // IndexAllocator 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexAllocator::IndexAllocator(void)
      : index_space(IndexSpace::NO_SPACE)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAllocator::IndexAllocator(const IndexAllocator &rhs)
      : index_space(rhs.index_space), iterator(rhs.iterator)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAllocator::IndexAllocator(IndexSpace is, IndexIterator itr)
      : index_space(is), iterator(itr)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAllocator::~IndexAllocator(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAllocator& IndexAllocator::operator=(const IndexAllocator &rhs)
    //--------------------------------------------------------------------------
    {
      index_space = rhs.index_space;
      iterator = rhs.iterator;
      return *this;
    }

    //--------------------------------------------------------------------------
    ptr_t IndexAllocator::alloc(unsigned num_elements)
    //--------------------------------------------------------------------------
    {
      size_t allocated = 0;
      ptr_t result = iterator.next_span(allocated, num_elements);
      if (allocated == num_elements)
        return result;
      else
        return ptr_t::nil();
    }

    //--------------------------------------------------------------------------
    void IndexAllocator::free(ptr_t ptr, unsigned num_elements)
    //--------------------------------------------------------------------------
    {
      Internal::log_run.error("Dynamic free of index space points is "
                              "no longer supported");
      assert(false);
    }
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif

    /////////////////////////////////////////////////////////////
    // Task Config Options 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskConfigOptions::TaskConfigOptions(bool l /*=false*/,
                                         bool in /*=false*/,
                                         bool idem /*=false*/)
      : leaf(l), inner(in), idempotent(idem)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // ProjectionFunctor 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProjectionFunctor::ProjectionFunctor(void)
      : runtime(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ProjectionFunctor::ProjectionFunctor(Runtime *rt)
      : runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ProjectionFunctor::~ProjectionFunctor(void)
    //--------------------------------------------------------------------------
    {
    }

// FIXME: This exists for backwards compatibility but it is tripping
// over our own deprecation warnings. Turn those off inside this method.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    //--------------------------------------------------------------------------
    LogicalRegion ProjectionFunctor::project(const Mappable *mappable, 
            unsigned index, LogicalRegion upper_bound, const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      REPORT_LEGION_WARNING(LEGION_WARNING_NEW_PROJECTION_FUNCTORS, 
                                "THERE ARE NEW METHODS FOR PROJECTION FUNCTORS "
                                "THAT MUST BE OVERRIDEN! CALLING DEPRECATED "
                            "METHODS FOR NOW!");
#endif
      switch (mappable->get_mappable_type())
      {
        case Mappable::TASK_MAPPABLE:
          return project(0/*dummy ctx*/, const_cast<Task*>(mappable->as_task()),
                         index, upper_bound, point);
        default:
          REPORT_LEGION_ERROR(ERROR_UNKNOWN_MAPPABLE, 
                              "Unknown mappable type passed to projection "
                              "functor! You must override the default "
                              "implementations of the non-deprecated "
                              "'project' methods!");
      }
      return LogicalRegion::NO_REGION;
    }
#pragma GCC diagnostic pop

// FIXME: This exists for backwards compatibility but it is tripping
// over our own deprecation warnings. Turn those off inside this method.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    //--------------------------------------------------------------------------
    LogicalRegion ProjectionFunctor::project(const Mappable *mappable,
         unsigned index, LogicalPartition upper_bound, const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      REPORT_LEGION_WARNING(LEGION_WARNING_NEW_PROJECTION_FUNCTORS, 
                                "THERE ARE NEW METHODS FOR PROJECTION FUNCTORS "
                                "THAT MUST BE OVERRIDEN! CALLING DEPRECATED "
                                "METHODS FOR NOW!");
#endif
      switch (mappable->get_mappable_type())
      {
        case Mappable::TASK_MAPPABLE:
          return project(0/*dummy ctx*/, const_cast<Task*>(mappable->as_task()),
                         index, upper_bound, point);
        default:
          REPORT_LEGION_ERROR(ERROR_UNKNOWN_MAPPABLE, 
                                  "Unknown mappable type passed to projection "
                                  "functor! You must override the default "
                                  "implementations of the non-deprecated "
                                  "'project' methods!");
              assert(false);
      }
      return LogicalRegion::NO_REGION;
    }
#pragma GCC diagnostic pop

    //--------------------------------------------------------------------------
    LogicalRegion ProjectionFunctor::project(Context ctx, Task *task,
            unsigned index, LogicalRegion upper_bound, const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_DEPRECATED_PROJECTION, 
                              "INVOCATION OF DEPRECATED PROJECTION "
                              "FUNCTOR METHOD WITHOUT AN OVERRIDE!");
      return LogicalRegion::NO_REGION;
    }

    //--------------------------------------------------------------------------
    LogicalRegion ProjectionFunctor::project(Context ctx, Task *task,
         unsigned index, LogicalPartition upper_bound, const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_DEPRECATED_PROJECTION, 
                              "INVOCATION OF DEPRECATED PROJECTION "
                              "FUNCTOR METHOD WITHOUT AN OVERRIDE!");
      return LogicalRegion::NO_REGION;
    }
    
    /////////////////////////////////////////////////////////////
    // Coloring Serializer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ColoringSerializer::ColoringSerializer(const Coloring &c)
      : coloring(c)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    size_t ColoringSerializer::legion_buffer_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = sizeof(size_t); // number of elements
      for (Coloring::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
      {
        result += sizeof(Color);
        result += 2*sizeof(size_t); // number of each kind of pointer
        result += (it->second.points.size() * sizeof(ptr_t));
        result += (it->second.ranges.size() * 2 * sizeof(ptr_t));
      }
      return result;
    }

    //--------------------------------------------------------------------------
    size_t ColoringSerializer::legion_serialize(void *buffer) const
    //--------------------------------------------------------------------------
    {
      char *target = (char*)buffer; 
      *((size_t*)target) = coloring.size();
      target += sizeof(size_t);
      for (Coloring::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
      {
        *((Color*)target) = it->first;
        target += sizeof(it->first);
        *((size_t*)target) = it->second.points.size();
        target += sizeof(size_t);
        for (std::set<ptr_t>::const_iterator ptr_it = it->second.points.begin();
              ptr_it != it->second.points.end(); ptr_it++)
        {
          *((ptr_t*)target) = *ptr_it;
          target += sizeof(ptr_t);
        }
        *((size_t*)target) = it->second.ranges.size();
        target += sizeof(size_t);
        for (std::set<std::pair<ptr_t,ptr_t> >::const_iterator range_it = 
              it->second.ranges.begin(); range_it != it->second.ranges.end();
              range_it++)
        {
          *((ptr_t*)target) = range_it->first;
          target += sizeof(range_it->first);
          *((ptr_t*)target) = range_it->second;
          target += sizeof(range_it->second);
        }
      }
      return (size_t(target) - size_t(buffer));
    }

    //--------------------------------------------------------------------------
    size_t ColoringSerializer::legion_deserialize(const void *buffer)
    //--------------------------------------------------------------------------
    {
      const char *source = (const char*)buffer;
      size_t num_colors = *((const size_t*)source);
      source += sizeof(num_colors);
      for (unsigned idx = 0; idx < num_colors; idx++)
      {
        Color c = *((const Color*)source);
        source += sizeof(c);
        coloring[c]; // Force coloring to exist even if empty.
        size_t num_points = *((const size_t*)source);
        source += sizeof(num_points);
        for (unsigned p = 0; p < num_points; p++)
        {
          ptr_t ptr = *((const ptr_t*)source);
          source += sizeof(ptr);
          coloring[c].points.insert(ptr);
        }
        size_t num_ranges = *((const size_t*)source);
        source += sizeof(num_ranges);
        for (unsigned r = 0; r < num_ranges; r++)
        {
          ptr_t start = *((const ptr_t*)source);
          source += sizeof(start);
          ptr_t stop = *((const ptr_t*)source);
          source += sizeof(stop);
          coloring[c].ranges.insert(std::pair<ptr_t,ptr_t>(start,stop));
        }
      }
      // Return the number of bytes consumed
      return (size_t(source) - size_t(buffer));
    }

    /////////////////////////////////////////////////////////////
    // Domain Coloring Serializer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DomainColoringSerializer::DomainColoringSerializer(const DomainColoring &d)
      : coloring(d)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    size_t DomainColoringSerializer::legion_buffer_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = sizeof(size_t); // number of elements
      result += (coloring.size() * (sizeof(Color) + sizeof(Domain)));
      return result;
    }

    //--------------------------------------------------------------------------
    size_t DomainColoringSerializer::legion_serialize(void *buffer) const
    //--------------------------------------------------------------------------
    {
      char *target = (char*)buffer;
      *((size_t*)target) = coloring.size();
      target += sizeof(size_t);
      for (DomainColoring::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
      {
        *((Color*)target) = it->first; 
        target += sizeof(it->first);
        *((Domain*)target) = it->second;
        target += sizeof(it->second);
      }
      return (size_t(target) - size_t(buffer));
    }

    //--------------------------------------------------------------------------
    size_t DomainColoringSerializer::legion_deserialize(const void *buffer)
    //--------------------------------------------------------------------------
    {
      const char *source = (const char*)buffer;
      size_t num_elements = *((const size_t*)source);
      source += sizeof(size_t);
      for (unsigned idx = 0; idx < num_elements; idx++)
      {
        Color c = *((const Color*)source);
        source += sizeof(c);
        Domain d = *((const Domain*)source);
        source += sizeof(d);
        coloring[c] = d;
      }
      // Return the number of bytes consumed
      return (size_t(source) - size_t(buffer));
    }

    /////////////////////////////////////////////////////////////
    // Legion Runtime 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Runtime::Runtime(Internal::Runtime *rt)
      : runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space(Context ctx,
                                                    size_t max_num_elmts)
    //--------------------------------------------------------------------------
    {
      Rect<1,coord_t> bounds((Point<1,coord_t>(0)),
                             (Point<1,coord_t>(max_num_elmts-1)));
      DomainT<1,coord_t> realm_index_space(bounds);
      return create_index_space_internal(ctx, &realm_index_space, TYPE_TAG_1D);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space(Context ctx, Domain domain)
    //--------------------------------------------------------------------------
    {
      switch (domain.get_dim())
      {
        case 1:
          {
            Rect<1,coord_t> bounds = domain;
            DomainT<1,coord_t> realm_is(bounds);
            return create_index_space_internal(ctx, &realm_is, TYPE_TAG_1D);
          }
        case 2:
          {
            Rect<2,coord_t> bounds = domain;
            DomainT<2,coord_t> realm_is(bounds);
            return create_index_space_internal(ctx, &realm_is, TYPE_TAG_2D);
          }
        case 3:
          {
            Rect<3,coord_t> bounds = domain;
            DomainT<3,coord_t> realm_is(bounds);
            return create_index_space_internal(ctx, &realm_is, TYPE_TAG_3D);
          }
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space(Context ctx, 
                                           const std::set<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      std::vector<Domain> rects(domains.begin(), domains.end());
      return create_index_space(ctx, rects); 
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space(Context ctx,
                                         const std::vector<DomainPoint> &points)
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
            return runtime->create_index_space(ctx, &realm_is, TYPE_TAG_1D);
          }
        case 2:
          {
            std::vector<Realm::Point<2,coord_t> > realm_points(points.size());
            for (unsigned idx = 0; idx < points.size(); idx++)
              realm_points[idx] = Point<2,coord_t>(points[idx]);
            DomainT<2,coord_t> realm_is(
                (Realm::IndexSpace<2,coord_t>(realm_points)));
            return runtime->create_index_space(ctx, &realm_is, TYPE_TAG_2D);
          }
        case 3:
          {
            std::vector<Realm::Point<3,coord_t> > realm_points(points.size());
            for (unsigned idx = 0; idx < points.size(); idx++)
              realm_points[idx] = Point<3,coord_t>(points[idx]);
            DomainT<3,coord_t> realm_is(
                (Realm::IndexSpace<3,coord_t>(realm_points)));
            return runtime->create_index_space(ctx, &realm_is, TYPE_TAG_3D);
          }
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space(Context ctx,
                                           const std::vector<Domain> &rects)
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
            return runtime->create_index_space(ctx, &realm_is, TYPE_TAG_1D);
          }
        case 2:
          {
            std::vector<Realm::Rect<2,coord_t> > realm_rects(rects.size());
            for (unsigned idx = 0; idx < rects.size(); idx++)
              realm_rects[idx] = Rect<2,coord_t>(rects[idx]);
            DomainT<2,coord_t> realm_is(
                (Realm::IndexSpace<2,coord_t>(realm_rects)));
            return runtime->create_index_space(ctx, &realm_is, TYPE_TAG_2D);
          }
        case 3:
          {
            std::vector<Realm::Rect<3,coord_t> > realm_rects(rects.size());
            for (unsigned idx = 0; idx < rects.size(); idx++)
              realm_rects[idx] = Rect<3,coord_t>(rects[idx]);
            DomainT<3,coord_t> realm_is(
                (Realm::IndexSpace<3,coord_t>(realm_rects)));
            return runtime->create_index_space(ctx, &realm_is, TYPE_TAG_3D);
          }
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_internal(Context ctx, 
                                         const void *realm_is, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_space(ctx, realm_is, type_tag);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::union_index_spaces(Context ctx,
                                          const std::vector<IndexSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      return runtime->union_index_spaces(ctx, spaces);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::intersect_index_spaces(Context ctx,
                                          const std::vector<IndexSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      return runtime->intersect_index_spaces(ctx, spaces);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::subtract_index_spaces(Context ctx,
                                              IndexSpace left, IndexSpace right)
    //--------------------------------------------------------------------------
    {
      return runtime->subtract_index_spaces(ctx, left, right);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_index_space(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_index_space(ctx, handle);
    } 

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(Context ctx,
                                          IndexSpace parent,
                                          const Domain &color_space,
                                          const PointColoring &coloring,
                                          PartitionKind part_kind,
                                          Color color, bool allocable)
    //--------------------------------------------------------------------------
    {
#ifndef DISABLE_PARTITION_SHIM
      if (allocable)
        Internal::log_run.warning("WARNING: allocable index partitions are "
                                  "no longer supported");
      // Count how many entries there are in the coloring
      coord_t num_entries = 0;
      bool do_ranges = false;
      for (PointColoring::const_iterator cit = coloring.begin(); 
            cit != coloring.end(); cit++)
      {
#ifdef DEBUG_LEGION
        assert(cit->first.get_dim() == color_space.get_dim());
#endif
        num_entries += cit->second.points.size();
        for (std::set<std::pair<ptr_t,ptr_t> >::const_iterator it = 
              cit->second.ranges.begin(); it != cit->second.ranges.end(); it++)
        {
          // Skip empty ranges
          if (it->first.value > it->second.value)
            continue;
          num_entries++;
          do_ranges = true;
        }
      }
      // Now make a temporary logical region with two fields to handle
      // the colors and points
      Rect<1,coord_t> bounds(Point<1,coord_t>(0),
                                     Point<1,coord_t>(num_entries-1));
      IndexSpaceT<1,coord_t> temp_is = create_index_space(ctx, bounds);
      FieldSpace temp_fs = create_field_space(ctx);
      const FieldID color_fid = 1;
      const FieldID pointer_fid = 2;
      {
        FieldAllocator allocator = create_field_allocator(ctx,temp_fs);
        switch (color_space.get_dim())
        {
          case 1:
            {
              allocator.allocate_field(
                  sizeof(Point<1,coord_t>), color_fid);
              break;
            }
          case 2:
            {
              allocator.allocate_field(
                  sizeof(Point<2,coord_t>), color_fid);
              break;
            }
          case 3:
            {
              allocator.allocate_field(
                  sizeof(Point<3,coord_t>), color_fid);
              break;
            }
          default:
            assert(false);
        }
        if (do_ranges)
          allocator.allocate_field(sizeof(Rect<1,coord_t>), 
                                   pointer_fid);
        else
          allocator.allocate_field(sizeof(Point<1,coord_t>), 
                                   pointer_fid);
      }
      LogicalRegionT<1,coord_t> temp_lr = create_logical_region(ctx,
                                            temp_is, temp_fs, true);
      // Fill in the logical region with the data
      // Do this with a task launch to maintain deferred execution
      switch (color_space.get_dim())
      {
        case 1:
          {
            if (do_ranges)
            {
              PartitionShim::ColorRects<1,1> launcher(coloring, temp_lr,
                                                color_fid, pointer_fid);
              runtime->execute_task(ctx, launcher);
            }
            else
            {
              PartitionShim::ColorPoints<1> launcher(coloring, temp_lr,
                                                color_fid, pointer_fid);
              runtime->execute_task(ctx, launcher);
            }
            break;
          }
        case 2:
          {
            if (do_ranges)
            {
              PartitionShim::ColorRects<2,1> launcher(coloring, temp_lr,
                                                color_fid, pointer_fid);
              runtime->execute_task(ctx, launcher);
            }
            else
            {
              PartitionShim::ColorPoints<2> launcher(coloring, temp_lr,
                                                color_fid, pointer_fid);
              runtime->execute_task(ctx, launcher);
            }
            break;
          }
        case 3:
          {
            if (do_ranges)
            {
              PartitionShim::ColorRects<3,1> launcher(coloring, temp_lr,
                                                color_fid, pointer_fid);
              runtime->execute_task(ctx, launcher);
            }
            else
            {
              PartitionShim::ColorPoints<3> launcher(coloring, temp_lr,
                                                color_fid, pointer_fid);
              runtime->execute_task(ctx, launcher);
            }
            break;
          }
        default:
          assert(false);
      }
      // Make an index space for the color space, just leak it for now
      IndexSpace index_color_space = create_index_space(ctx, color_space);
      // Partition the logical region by the color field
      IndexPartition temp_ip = create_partition_by_field(ctx, temp_lr, 
                                    temp_lr, color_fid, index_color_space,
                                    AUTO_GENERATE_ID, PARTITION_SHIM_MAPPER_ID);
      // Then project the partition image through the pointer field
      LogicalPartition temp_lp = get_logical_partition(temp_lr, temp_ip);
      IndexPartition result;
      if (do_ranges)
        result = create_partition_by_image_range(ctx, parent, temp_lp,
                    temp_lr, pointer_fid, index_color_space, part_kind, color,
                    PARTITION_SHIM_MAPPER_ID);
      else
        result = create_partition_by_image(ctx, parent, temp_lp, 
                    temp_lr, pointer_fid, index_color_space, part_kind, color,
                    PARTITION_SHIM_MAPPER_ID);
      // Clean everything up
      destroy_logical_region(ctx, temp_lr);
      destroy_field_space(ctx, temp_fs);
      destroy_index_space(ctx, temp_is);
      return result;
#else // DISABLE_PARTITION_SHIM
      log_run.error("THE PARTITION SHIM HAS BEEN DISABLED!");
      assert(false);
      return IndexPartition::NO_PART;
#endif
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(
                                          Context ctx, IndexSpace parent,
                                          const Coloring &coloring,
                                          bool disjoint,
                                          Color part_color)
    //--------------------------------------------------------------------------
    {
#ifndef DISABLE_PARTITION_SHIM
      // Count how many entries there are in the coloring
      coord_t num_entries = 0;
      bool do_ranges = false;
      Color lower_bound = UINT_MAX, upper_bound = 0;
      for (Coloring::const_iterator cit = coloring.begin(); 
            cit != coloring.end(); cit++)
      {
        if (cit->first < lower_bound)
          lower_bound = cit->first;
        if (cit->first > upper_bound)
          upper_bound = cit->first;
        num_entries += cit->second.points.size();
        for (std::set<std::pair<ptr_t,ptr_t> >::const_iterator it = 
              cit->second.ranges.begin(); it != cit->second.ranges.end(); it++)
        {
          // Skip empty ranges
          if (it->first.value > it->second.value)
            continue;
          num_entries++;
          do_ranges = true;
        }
      }
#ifdef DEBUG_LEGION
      assert(lower_bound <= upper_bound);
#endif
      // Make the color space
      Rect<1,coord_t> 
        color_space((Point<1,coord_t>(lower_bound)),
                    (Point<1,coord_t>(upper_bound)));
      // Now make a temporary logical region with two fields to handle
      // the colors and points
      Rect<1,coord_t> bounds(Point<1,coord_t>(0),
                                     Point<1,coord_t>(num_entries-1));
      IndexSpaceT<1,coord_t> temp_is = create_index_space(ctx, bounds);
      FieldSpace temp_fs = create_field_space(ctx);
      const FieldID color_fid = 1;
      const FieldID pointer_fid = 2;
      {
        FieldAllocator allocator = create_field_allocator(ctx,temp_fs);
        allocator.allocate_field(sizeof(Point<1,coord_t>), color_fid);
        if (do_ranges)
          allocator.allocate_field(sizeof(Rect<1,coord_t>),
                                   pointer_fid);
        else
          allocator.allocate_field(sizeof(Point<1,coord_t>), 
                                   pointer_fid);
      }
      LogicalRegionT<1,coord_t> temp_lr = create_logical_region(ctx,
                                            temp_is, temp_fs, true);
      // Fill in the logical region with the data
      // Do this with a task launch to maintain deferred execution
      if (do_ranges)
      {
        PartitionShim::ColorRects<1,1> launcher(coloring, temp_lr, 
                                              color_fid, pointer_fid);
        runtime->execute_task(ctx, launcher);
      }
      else
      {
        PartitionShim::ColorPoints<1> launcher(coloring, temp_lr, 
                                               color_fid, pointer_fid);
        runtime->execute_task(ctx, launcher);
      }
      
      // Make an index space for the color space, we'll leak it for now
      IndexSpaceT<1,coord_t> index_color_space = 
                                  create_index_space(ctx, color_space);
      // Partition the logical region by the color field
      IndexPartitionT<1,coord_t> temp_ip = create_partition_by_field(ctx,
                            temp_lr, temp_lr, color_fid, index_color_space,
                            AUTO_GENERATE_ID, PARTITION_SHIM_MAPPER_ID);
      // Then project the partition image through the pointer field
      LogicalPartitionT<1,coord_t> temp_lp = 
                                     get_logical_partition(temp_lr, temp_ip);
      IndexPartitionT<1,coord_t> result;
      if (do_ranges)
        result = create_partition_by_image_range(ctx, 
            IndexSpaceT<1,coord_t>(parent), temp_lp, temp_lr, pointer_fid,
            index_color_space, (disjoint ? DISJOINT_KIND : ALIASED_KIND),
            part_color, PARTITION_SHIM_MAPPER_ID);
      else
        result = create_partition_by_image(ctx,
            IndexSpaceT<1,coord_t>(parent), temp_lp, temp_lr, pointer_fid, 
            index_color_space, (disjoint ? DISJOINT_KIND : ALIASED_KIND), 
            part_color, PARTITION_SHIM_MAPPER_ID);
      // Clean everything up
      destroy_logical_region(ctx, temp_lr);
      destroy_field_space(ctx, temp_fs);
      destroy_index_space(ctx, temp_is);
      return result;
#else // DISABLE_PARTITION_SHIM
      log_run.error("THE PARTITION SHIM HAS BEEN DISABLED!");
      assert(false);
      return IndexPartition::NO_PART;
#endif
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(Context ctx,
                                          IndexSpace parent, 
                                          const Domain &color_space,
                                          const DomainPointColoring &coloring,
                                          PartitionKind part_kind, Color color)
    //--------------------------------------------------------------------------
    {
#ifndef DISABLE_PARTITION_SHIM
      // Count how many entries there are in the coloring
      const coord_t num_entries = coloring.size();
      // Now make a temporary logical region with two fields to handle
      // the colors and points
      Rect<1,coord_t> bounds(Point<1,coord_t>(0),
                                     Point<1,coord_t>(num_entries-1));
      IndexSpaceT<1,coord_t> temp_is = create_index_space(ctx, bounds);
      FieldSpace temp_fs = create_field_space(ctx);
      const FieldID color_fid = 1;
      const FieldID range_fid = 2;
      const int color_dim = color_space.get_dim();
      const int range_dim = coloring.begin()->second.get_dim();
      {
        FieldAllocator allocator = create_field_allocator(ctx,temp_fs);
        switch (color_dim)
        {
          case 1:
            {
              allocator.allocate_field(
                  sizeof(Point<1,coord_t>), color_fid);
              break;
            }
          case 2:
            {
              allocator.allocate_field(
                  sizeof(Point<2,coord_t>), color_fid);
              break;
            }
          case 3:
            {
              allocator.allocate_field(
                  sizeof(Point<3,coord_t>), color_fid);
              break;
            }
          default:
            assert(false);
        }
        switch (range_dim)
        {
          case 1:
            {
              allocator.allocate_field(
                  sizeof(Rect<1,coord_t>), range_fid);
              break;
            }
          case 2:
            {
              allocator.allocate_field(
                  sizeof(Rect<2,coord_t>), range_fid);
              break;
            }
          case 3:
            {
              allocator.allocate_field(
                  sizeof(Rect<3,coord_t>), range_fid);
              break;
            }
          default:
            assert(false);
        }
      }
      LogicalRegionT<1,coord_t> temp_lr = create_logical_region(ctx,
                                            temp_is, temp_fs, true);
      // Fill in the logical region with the data
      // Do this with a task launch to maintain deferred execution
      switch (color_dim)
      {
        case 1:
          {
            switch (range_dim)
            {
              case 1:
                {
                  PartitionShim::ColorRects<1,1> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              case 2:
                {
                  PartitionShim::ColorRects<1,2> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              case 3:
                {
                  PartitionShim::ColorRects<1,3> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              default:
                assert(false);
            }
            break;
          }
        case 2:
          {
            switch (range_dim)
            {
              case 1:
                {
                  PartitionShim::ColorRects<2,1> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              case 2:
                {
                  PartitionShim::ColorRects<2,2> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              case 3:
                {
                  PartitionShim::ColorRects<2,3> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              default:
                assert(false);
            }
            break;
          }
        case 3:
          {
            switch (range_dim)
            {
              case 1:
                {
                  PartitionShim::ColorRects<3,1> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              case 2:
                {
                  PartitionShim::ColorRects<3,2> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              case 3:
                {
                  PartitionShim::ColorRects<3,3> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              default:
                assert(false);
            }
            break;
          }
        default:
          assert(false);
      }
      // Make an index space for the color space, just leak it for now
      IndexSpace index_color_space = create_index_space(ctx, color_space);
      // Partition the logical region by the color field
      IndexPartition temp_ip = create_partition_by_field(ctx, temp_lr, 
                                temp_lr, color_fid, index_color_space,
                                AUTO_GENERATE_ID, PARTITION_SHIM_MAPPER_ID);
      // Then project the partition image through the range field
      LogicalPartition temp_lp = get_logical_partition(temp_lr, temp_ip);
      IndexPartition result = create_partition_by_image_range(ctx, parent, 
                           temp_lp, temp_lr, range_fid, index_color_space, 
                           part_kind, color, PARTITION_SHIM_MAPPER_ID);
      // Clean everything up
      destroy_logical_region(ctx, temp_lr);
      destroy_field_space(ctx, temp_fs);
      destroy_index_space(ctx, temp_is);
      return result;
#else // DISABLE_PARTITION_SHIM
      log_run.error("THE PARTITION SHIM HAS BEEN DISABLED!");
      assert(false);
      return IndexPartition::NO_PART;
#endif
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(
                                          Context ctx, IndexSpace parent,
                                          Domain color_space,
                                          const DomainColoring &coloring,
                                          bool disjoint, Color part_color)
    //--------------------------------------------------------------------------
    {
#ifndef DISABLE_PARTITION_SHIM
      // Count how many entries there are in the coloring
      const coord_t num_entries = coloring.size();
      // Now make a temporary logical region with two fields to handle
      // the colors and points
      Rect<1,coord_t> bounds(Point<1,coord_t>(0),
                                     Point<1,coord_t>(num_entries-1));
      IndexSpaceT<1,coord_t> temp_is = create_index_space(ctx, bounds);
      FieldSpace temp_fs = create_field_space(ctx);
      const FieldID color_fid = 1;
      const FieldID range_fid = 2;
      const int range_dim = coloring.begin()->second.get_dim();
      {
        FieldAllocator allocator = create_field_allocator(ctx,temp_fs);
        allocator.allocate_field(sizeof(Point<1,coord_t>), color_fid);
        switch (range_dim)
        {
          case 1:
            {
              allocator.allocate_field(
                  sizeof(Rect<1,coord_t>), range_fid);
              break;
            }
          case 2:
            {
              allocator.allocate_field(
                  sizeof(Rect<2,coord_t>), range_fid);
              break;
            }
          case 3:
            {
              allocator.allocate_field(
                  sizeof(Rect<3,coord_t>), range_fid);
              break;
            }
          default:
            assert(false);
        }
      }
      LogicalRegionT<1,coord_t> temp_lr = create_logical_region(ctx,
                                            temp_is, temp_fs, true);
      // Fill in the logical region with the data
      // Do this with a task launch to maintain deferred execution
      switch (range_dim)
      {
        case 1:
          {
            PartitionShim::ColorRects<1,1> launcher(coloring,
                temp_lr, color_fid, range_fid);
            runtime->execute_task(ctx, launcher);
            break;
          }
        case 2:
          {
            PartitionShim::ColorRects<1,2> launcher(coloring,
                temp_lr, color_fid, range_fid);
            runtime->execute_task(ctx, launcher);
            break;
          }
        case 3:
          {
            PartitionShim::ColorRects<1,3> launcher(coloring,
                temp_lr, color_fid, range_fid);
            runtime->execute_task(ctx, launcher);
            break;
          }
        default:
          assert(false);
      }

      IndexSpaceT<1,coord_t> index_color_space = 
                            create_index_space<1,coord_t>(ctx, color_space);
      // Partition the logical region by the color field
      IndexPartition temp_ip = create_partition_by_field(ctx, temp_lr,
                                    temp_lr, color_fid, index_color_space,
                                    AUTO_GENERATE_ID, PARTITION_SHIM_MAPPER_ID);
      // Then project the partition image through the pointer field
      LogicalPartition temp_lp = get_logical_partition(temp_lr, temp_ip);
      IndexPartition result = create_partition_by_image_range(ctx,
          parent, temp_lp, temp_lr, range_fid, index_color_space, 
          (disjoint ? DISJOINT_KIND : ALIASED_KIND), part_color,
          PARTITION_SHIM_MAPPER_ID);
      // Clean everything up
      destroy_logical_region(ctx, temp_lr);
      destroy_field_space(ctx, temp_fs);
      destroy_index_space(ctx, temp_is);
      return result;
#else // DISABLE_PARTITION_SHIM
      log_run.error("THE PARTITION SHIM HAS BEEN DISABLED!");
      assert(false);
      return IndexPartition::NO_PART;
#endif
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(Context ctx,
                                       IndexSpace parent,
                                       const Domain &color_space,
                                       const MultiDomainPointColoring &coloring,
                                       PartitionKind part_kind, Color color)
    //--------------------------------------------------------------------------
    {
#ifndef DISABLE_PARTITION_SHIM
      // Count how many entries there are in the coloring
      coord_t num_entries = 0;
      for (MultiDomainPointColoring::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
        num_entries += it->second.size(); 
      // Now make a temporary logical region with two fields to handle
      // the colors and points
      Rect<1,coord_t> bounds(Point<1,coord_t>(0),
                                     Point<1,coord_t>(num_entries-1));
      IndexSpaceT<1,coord_t> temp_is = create_index_space(ctx, bounds);
      FieldSpace temp_fs = create_field_space(ctx);
      const FieldID color_fid = 1;
      const FieldID range_fid = 2;
      const int color_dim = color_space.get_dim();
      const int range_dim = coloring.begin()->second.begin()->get_dim();
      {
        FieldAllocator allocator = create_field_allocator(ctx,temp_fs);
        switch (color_dim)
        {
          case 1:
            {
              allocator.allocate_field(
                  sizeof(Point<1,coord_t>), color_fid);
              break;
            }
          case 2:
            {
              allocator.allocate_field(
                  sizeof(Point<2,coord_t>), color_fid);
              break;
            }
          case 3:
            {
              allocator.allocate_field(
                  sizeof(Point<3,coord_t>), color_fid);
              break;
            }
          default:
            assert(false);
        }
        switch (range_dim)
        {
          case 1:
            {
              allocator.allocate_field(
                  sizeof(Rect<1,coord_t>), range_fid);
              break;
            }
          case 2:
            {
              allocator.allocate_field(
                  sizeof(Rect<2,coord_t>), range_fid);
              break;
            }
          case 3:
            {
              allocator.allocate_field(
                  sizeof(Rect<3,coord_t>), range_fid);
              break;
            }
          default:
            assert(false);
        }
      }
      LogicalRegionT<1,coord_t> temp_lr = create_logical_region(ctx,
                                            temp_is, temp_fs, true);
      // Fill in the logical region with the data
      // Do this with a task launch to maintain deferred execution
      switch (color_dim)
      {
        case 1:
          {
            switch (range_dim)
            {
              case 1:
                {
                  PartitionShim::ColorRects<1,1> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              case 2:
                {
                  PartitionShim::ColorRects<1,2> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              case 3:
                {
                  PartitionShim::ColorRects<1,3> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              default:
                assert(false);
            }
            break;
          }
        case 2:
          {
            switch (range_dim)
            {
              case 1:
                {
                  PartitionShim::ColorRects<2,1> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              case 2:
                {
                  PartitionShim::ColorRects<2,2> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              case 3:
                {
                  PartitionShim::ColorRects<2,3> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              default:
                assert(false);
            }
            break;
          }
        case 3:
          {
            switch (range_dim)
            {
              case 1:
                {
                  PartitionShim::ColorRects<3,1> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              case 2:
                {
                  PartitionShim::ColorRects<3,2> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              case 3:
                {
                  PartitionShim::ColorRects<3,3> launcher(coloring,
                      temp_lr, color_fid, range_fid);
                  runtime->execute_task(ctx, launcher);
                  break;
                }
              default:
                assert(false);
            }
            break;
          }
        default:
          assert(false);
      }
      // Make an index space for the color space, just leak it for now
      IndexSpace index_color_space = create_index_space(ctx, color_space);
      // Partition the logical region by the color field
      IndexPartition temp_ip = create_partition_by_field(ctx, temp_lr, 
                                    temp_lr, color_fid, index_color_space,
                                    AUTO_GENERATE_ID, PARTITION_SHIM_MAPPER_ID);
      // Then project the partition image through the range field
      LogicalPartition temp_lp = get_logical_partition(temp_lr, temp_ip);
      IndexPartition result = create_partition_by_image_range(ctx, parent, 
            temp_lp, temp_lr, range_fid, index_color_space, part_kind, color,
            PARTITION_SHIM_MAPPER_ID);
      // Clean everything up
      destroy_logical_region(ctx, temp_lr);
      destroy_field_space(ctx, temp_fs);
      destroy_index_space(ctx, temp_is);
      return result;
#else // DISABLE_PARTITION_SHIM
      log_run.error("THE PARTITION SHIM HAS BEEN DISABLED!");
      assert(false);
      return IndexPartition::NO_PART;
#endif
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(
                                          Context ctx, IndexSpace parent,
                                          Domain color_space,
                                          const MultiDomainColoring &coloring,
                                          bool disjoint, Color part_color)
    //--------------------------------------------------------------------------
    {
#ifndef DISABLE_PARTITION_SHIM
      // Count how many entries there are in the coloring
      coord_t num_entries = 0;
      for (MultiDomainColoring::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
        num_entries += it->second.size();
      // Now make a temporary logical region with two fields to handle
      // the colors and points
      Rect<1,coord_t> bounds(Point<1,coord_t>(0),
                                     Point<1,coord_t>(num_entries-1));
      IndexSpaceT<1,coord_t> temp_is = create_index_space(ctx, bounds);
      FieldSpace temp_fs = create_field_space(ctx);
      const FieldID color_fid = 1;
      const FieldID range_fid = 2;
      const int range_dim = coloring.begin()->second.begin()->get_dim();
      {
        FieldAllocator allocator = create_field_allocator(ctx,temp_fs);
        allocator.allocate_field(sizeof(Point<1,coord_t>), color_fid);
        switch (range_dim)
        {
          case 1:
            {
              allocator.allocate_field(
                  sizeof(Rect<1,coord_t>), range_fid);
              break;
            }
          case 2:
            {
              allocator.allocate_field(
                  sizeof(Rect<2,coord_t>), range_fid);
              break;
            }
          case 3:
            {
              allocator.allocate_field(
                  sizeof(Rect<3,coord_t>), range_fid);
              break;
            }
          default:
            assert(false);
        }
      }
      LogicalRegionT<1,coord_t> temp_lr = create_logical_region(ctx,
                                            temp_is, temp_fs, true);
      // Fill in the logical region with the data
      // Do this with a task launch to maintain deferred execution
      switch (range_dim)
      {
        case 1:
          {
            PartitionShim::ColorRects<1,1> launcher(coloring,
                temp_lr, color_fid, range_fid);
            runtime->execute_task(ctx, launcher);
            break;
          }
        case 2:
          {
            PartitionShim::ColorRects<1,2> launcher(coloring,
                temp_lr, color_fid, range_fid);
            runtime->execute_task(ctx, launcher);
            break;
          }
        case 3:
          {
            PartitionShim::ColorRects<1,3> launcher(coloring,
                temp_lr, color_fid, range_fid);
            runtime->execute_task(ctx, launcher);
            break;
          }
        default:
          assert(false);
      }
      IndexSpaceT<1,coord_t> index_color_space = 
                            create_index_space<1,coord_t>(ctx, color_space);
      // Partition the logical region by the color field
      IndexPartition temp_ip = create_partition_by_field(ctx, temp_lr,
                                    temp_lr, color_fid, index_color_space,
                                    AUTO_GENERATE_ID, PARTITION_SHIM_MAPPER_ID);
      // Then project the partition image through the pointer field
      LogicalPartition temp_lp = get_logical_partition(temp_lr, temp_ip);
      IndexPartition result = create_partition_by_image_range(ctx,
          parent, temp_lp, temp_lr, range_fid, index_color_space, 
          (disjoint ? DISJOINT_KIND : ALIASED_KIND), part_color,
          PARTITION_SHIM_MAPPER_ID);
      // Clean everything up
      destroy_logical_region(ctx, temp_lr);
      destroy_field_space(ctx, temp_fs);
      destroy_index_space(ctx, temp_is);
      return result;
#else // DISABLE_PARTITION_SHIM
      log_run.error("THE PARTITION SHIM HAS BEEN DISABLED!");
      assert(false);
      return IndexPartition::NO_PART; 
#endif
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(
                                          Context ctx, IndexSpace parent,
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic> field_accessor,
                                                      Color part_color)
    //--------------------------------------------------------------------------
    {
      Internal::log_run.error("Call to deprecated 'create_index_partition' "
                    "method with an accessor in task %s (UID %lld) should be "
                    "replaced with a call to create_partition_by_field.",
                    ctx->get_task_name(), ctx->get_unique_id());
      assert(false);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_index_partition(Context ctx, 
                                                   IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_index_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_equal_partition(Context ctx, 
                                                      IndexSpace parent,
                                                      IndexSpace color_space,
                                                      size_t granularity,
                                                      Color color)
    //--------------------------------------------------------------------------
    {
      return runtime->create_equal_partition(ctx, parent, color_space,
                                             granularity, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_union(Context ctx,
                                    IndexSpace parent, IndexPartition handle1,
                                    IndexPartition handle2, 
                                    IndexSpace color_space, PartitionKind kind,
                                    Color color)
    //--------------------------------------------------------------------------
    {
      return runtime->create_partition_by_union(ctx, parent, handle1, handle2, 
                                                color_space, kind, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_intersection(
                                                Context ctx, IndexSpace parent,
                                                IndexPartition handle1, 
                                                IndexPartition handle2,
                                                IndexSpace color_space,
                                                PartitionKind kind, Color color) 
    //--------------------------------------------------------------------------
    {
      return runtime->create_partition_by_intersection(ctx, parent, handle1,
                                                       handle2, color_space,
                                                       kind, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_difference(
                                                Context ctx, IndexSpace parent,
                                                IndexPartition handle1,
                                                IndexPartition handle2,
                                                IndexSpace color_space,
                                                PartitionKind kind, Color color)
    //--------------------------------------------------------------------------
    {
      return runtime->create_partition_by_difference(ctx, parent, handle1,
                                                     handle2, color_space,
                                                     kind, color);
    }

    //--------------------------------------------------------------------------
    Color Runtime::create_cross_product_partitions(Context ctx,
                                IndexPartition handle1, IndexPartition handle2,
                                std::map<IndexSpace,IndexPartition> &handles,
                                PartitionKind kind, Color color)
    //--------------------------------------------------------------------------
    {
      return runtime->create_cross_product_partitions(ctx, handle1, handle2, 
                                                      handles, kind, color);
    }

    //--------------------------------------------------------------------------
    void Runtime::create_association(Context ctx,
                                     LogicalRegion domain,
                                     LogicalRegion domain_parent,
                                     FieldID domain_fid,
                                     IndexSpace range,
                                     MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      runtime->create_association(ctx, domain, domain_parent, domain_fid,
                                  range, id, tag);
    }

    //--------------------------------------------------------------------------
    void Runtime::create_bidirectional_association(Context ctx,
                                      LogicalRegion domain,
                                      LogicalRegion domain_parent,
                                      FieldID domain_fid,
                                      LogicalRegion range,
                                      LogicalRegion range_parent,
                                      FieldID range_fid,
                                      MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      // Realm guarantees that creating association in either direction
      // will produce the same result, so we can do these separately
      create_association(ctx, domain, domain_parent, domain_fid, 
                         range.get_index_space(), id, tag);
      create_association(ctx, range, range_parent, range_fid, 
                         domain.get_index_space(), id, tag);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_restriction(Context ctx,
                                                        IndexSpace par,
                                                        IndexSpace cs,
                                                        DomainTransform tran,
                                                        Domain ext,
                                                        PartitionKind part_kind,
                                                        Color color)
    //--------------------------------------------------------------------------
    {
      switch (ext.get_dim())
      {
        case 1:
          {
            const IndexSpaceT<1,coord_t> parent(par);
            const Rect<1,coord_t> extent(ext);
            switch (tran.n)
            {
              case 1:
                {
                  const Transform<1,1> transform(tran);
                  const IndexSpaceT<1,coord_t> color_space(cs);
                  return create_partition_by_restriction<1,1,coord_t>(ctx, 
                      parent, color_space, transform, extent, part_kind, color);
                }
              case 2:
                {
                  const Transform<1,2> transform(tran);
                  const IndexSpaceT<2,coord_t> color_space(cs);
                  return create_partition_by_restriction<1,2,coord_t>(ctx,
                      parent, color_space, transform, extent, part_kind, color);
                }
              case 3:
                {
                  const Transform<1,3> transform(tran);
                  const IndexSpaceT<3,coord_t> color_space(cs);
                  return create_partition_by_restriction<1,3,coord_t>(ctx,
                      parent, color_space, transform, extent, part_kind, color);
                }
              default:
                assert(false);
            }
          }
        case 2:
          {
            const IndexSpaceT<2,coord_t> parent(par);
            const Rect<2,coord_t> extent(ext);
            switch (tran.n)
            {
              case 1:
                {
                  const Transform<2,1> transform(tran);
                  const IndexSpaceT<1,coord_t> color_space(cs);
                  return create_partition_by_restriction<2,1,coord_t>(ctx, 
                      parent, color_space, transform, extent, part_kind, color);
                }
              case 2:
                {
                  const Transform<2,2> transform(tran);
                  const IndexSpaceT<2,coord_t> color_space(cs);
                  return create_partition_by_restriction<2,2,coord_t>(ctx,
                      parent, color_space, transform, extent, part_kind, color);
                }
              case 3:
                {
                  const Transform<2,3> transform(tran);
                  const IndexSpaceT<3,coord_t> color_space(cs);
                  return create_partition_by_restriction<2,3,coord_t>(ctx,
                      parent, color_space, transform, extent, part_kind, color);
                }
              default:
                assert(false);
            }
          }
        case 3:
          {
            const IndexSpaceT<3,coord_t> parent(par);
            const Rect<3,coord_t> extent(ext);
            switch (tran.n)
            {
              case 1:
                {
                  const Transform<3,1> transform(tran);
                  const IndexSpaceT<1,coord_t> color_space(cs);
                  return create_partition_by_restriction<3,1,coord_t>(ctx, 
                      parent, color_space, transform, extent, part_kind, color);
                }
              case 2:
                {
                  const Transform<3,2> transform(tran);
                  const IndexSpaceT<2,coord_t> color_space(cs);
                  return create_partition_by_restriction<3,2,coord_t>(ctx,
                      parent, color_space, transform, extent, part_kind, color);
                }
              case 3:
                {
                  const Transform<3,3> transform(tran);
                  const IndexSpaceT<3,coord_t> color_space(cs);
                  return create_partition_by_restriction<3,3,coord_t>(ctx,
                      parent, color_space, transform, extent, part_kind, color);
                }
              default:
                assert(false);
            }
          }
        default:
          assert(false);
      }
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_blockify(Context ctx,
                                                         IndexSpace par,
                                                         DomainPoint bf,
                                                         Color color)
    //--------------------------------------------------------------------------
    {
      switch (bf.get_dim())
      {
        case 1:
          {
            const IndexSpaceT<1,coord_t> parent(par);
            const Point<1,coord_t> blocking_factor(bf);
            return create_partition_by_blockify<1,coord_t>(ctx, parent, 
                                                blocking_factor, color);
          }
        case 2:
          {
            const IndexSpaceT<2,coord_t> parent(par);
            const Point<2,coord_t> blocking_factor(bf);
            return create_partition_by_blockify<2,coord_t>(ctx, parent, 
                                                blocking_factor, color);
          }
        case 3:
          {
            const IndexSpaceT<3,coord_t> parent(par);
            const Point<3,coord_t> blocking_factor(bf);
            return create_partition_by_blockify<3,coord_t>(ctx, parent, 
                                                blocking_factor, color);
          }
        default:
          assert(false);
      }
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_blockify(Context ctx,
                                                         IndexSpace par,
                                                         DomainPoint bf,
                                                         DomainPoint orig,
                                                         Color color)
    //--------------------------------------------------------------------------
    {
      switch (bf.get_dim())
      {
        case 1:
          {
            const IndexSpaceT<1,coord_t> parent(par);
            const Point<1,coord_t> blocking_factor(bf);
            const Point<1,coord_t> origin(orig);
            return create_partition_by_blockify<1,coord_t>(ctx, parent, 
                                        blocking_factor, origin, color);
          }
        case 2:
          {
            const IndexSpaceT<2,coord_t> parent(par);
            const Point<2,coord_t> blocking_factor(bf);
            const Point<2,coord_t> origin(orig);
            return create_partition_by_blockify<2,coord_t>(ctx, parent, 
                                        blocking_factor, origin, color);
          }
        case 3:
          {
            const IndexSpaceT<3,coord_t> parent(par);
            const Point<3,coord_t> blocking_factor(bf);
            const Point<3,coord_t> origin(orig);
            return create_partition_by_blockify<3,coord_t>(ctx, parent, 
                                        blocking_factor, origin, color);
          }
        default:
          assert(false);
      }
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_restricted_partition(Context ctx,
                                                        IndexSpace parent, 
                                                        IndexSpace color_space,
                                                        const void *transform,
                                                        size_t transform_size,
                                                        const void *extent, 
                                                        size_t extent_size,
                                                        PartitionKind part_kind,
                                                        Color color)
    //--------------------------------------------------------------------------
    {
      return runtime->create_restricted_partition(ctx, parent, color_space,
                            transform, transform_size, extent, extent_size,
                            part_kind, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_field(Context ctx,
                   LogicalRegion handle, LogicalRegion parent, FieldID fid, 
                   IndexSpace color_space, Color color, 
                   MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->create_partition_by_field(ctx, handle, parent, fid,
                                                color_space, color, id, tag);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_image(Context ctx,
                  IndexSpace handle, LogicalPartition projection,
                  LogicalRegion parent, FieldID fid, IndexSpace color_space,
                  PartitionKind part_kind, Color color,
                  MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->create_partition_by_image(ctx, handle, projection,
                                                parent, fid, color_space,
                                                part_kind, color, id, tag);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_image_range(Context ctx,
                  IndexSpace handle, LogicalPartition projection,
                  LogicalRegion parent, FieldID fid, IndexSpace color_space,
                  PartitionKind part_kind, Color color, 
                  MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->create_partition_by_image_range(ctx, handle, projection,
                                                      parent, fid, color_space,
                                                      part_kind, color, id,tag);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_preimage(Context ctx,
                  IndexPartition projection, LogicalRegion handle,
                  LogicalRegion parent, FieldID fid, IndexSpace color_space,
                  PartitionKind part_kind, Color color,
                  MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->create_partition_by_preimage(ctx, projection, handle,
                                                   parent, fid, color_space,
                                                   part_kind, color, id, tag);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_preimage_range(Context ctx,
                  IndexPartition projection, LogicalRegion handle,
                  LogicalRegion parent, FieldID fid, IndexSpace color_space,
                  PartitionKind part_kind, Color color,
                  MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->create_partition_by_preimage_range(ctx, projection,handle,
                                                       parent, fid, color_space,
                                                       part_kind, color,id,tag);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_pending_partition(Context ctx,
                             IndexSpace parent, IndexSpace color_space, 
                             PartitionKind part_kind, Color color)
    //--------------------------------------------------------------------------
    {
      return runtime->create_pending_partition(ctx, parent, color_space, 
                                               part_kind, color);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_union(Context ctx,
                      IndexPartition parent, const DomainPoint &color,
                      const std::vector<IndexSpace> &handles) 
    //--------------------------------------------------------------------------
    {
      switch (color.get_dim())
      {
        case 1:
          {
            Point<1,coord_t> point = color;
            return runtime->create_index_space_union(ctx, parent, &point,
                                                     TYPE_TAG_1D, handles);
          }
        case 2:
          {
            Point<2,coord_t> point = color;
            return runtime->create_index_space_union(ctx, parent, &point,
                                                     TYPE_TAG_2D, handles);
          }
        case 3:
          {
            Point<3,coord_t> point = color;
            return runtime->create_index_space_union(ctx, parent, &point,
                                                     TYPE_TAG_3D, handles);
          }
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_union_internal(Context ctx,
                    IndexPartition parent, const void *color, TypeTag type_tag,
                    const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_space_union(ctx, parent, 
                                               color, type_tag, handles);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_union(Context ctx,
                      IndexPartition parent, const DomainPoint &color,
                      IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      switch (color.get_dim())
      {
        case 1:
          {
            Point<1,coord_t> point = color;
            return runtime->create_index_space_union(ctx, parent, &point,
                                                     TYPE_TAG_1D, handle);
          }
        case 2:
          {
            Point<2,coord_t> point = color;
            return runtime->create_index_space_union(ctx, parent, &point,
                                                     TYPE_TAG_2D, handle);
          }
        case 3:
          {
            Point<3,coord_t> point = color;
            return runtime->create_index_space_union(ctx, parent, &point,
                                                     TYPE_TAG_3D, handle);
          }
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_union_internal(Context ctx,
                        IndexPartition parent, const void *realm_color, 
                        TypeTag type_tag, IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_space_union(ctx, parent, realm_color,
                                               type_tag, handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_intersection(Context ctx,
                      IndexPartition parent, const DomainPoint &color,
                      const std::vector<IndexSpace> &handles) 
    //--------------------------------------------------------------------------
    {
      switch (color.get_dim())
      {
        case 1:
          {
            Point<1,coord_t> point = color;
            return runtime->create_index_space_intersection(ctx, parent, &point,
                                                          TYPE_TAG_1D, handles);
          }
        case 2:
          {
            Point<2,coord_t> point = color;
            return runtime->create_index_space_intersection(ctx, parent, &point,
                                                          TYPE_TAG_2D, handles);
          }
        case 3:
          {
            Point<3,coord_t> point = color;
            return runtime->create_index_space_intersection(ctx, parent, &point,
                                                          TYPE_TAG_3D, handles);
          }
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_intersection_internal(Context ctx,
                    IndexPartition parent, const void *color, TypeTag type_tag,
                    const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_space_intersection(ctx, parent, 
                                                      color, type_tag, handles);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_intersection(Context ctx,
                      IndexPartition parent, const DomainPoint &color,
                      IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      switch (color.get_dim())
      {
        case 1:
          {
            Point<1,coord_t> point = color;
            return runtime->create_index_space_intersection(ctx, parent, &point,
                                                           TYPE_TAG_1D, handle);
          }
        case 2:
          {
            Point<2,coord_t> point = color;
            return runtime->create_index_space_intersection(ctx, parent, &point,
                                                           TYPE_TAG_2D, handle);
          }
        case 3:
          {
            Point<3,coord_t> point = color;
            return runtime->create_index_space_intersection(ctx, parent, &point,
                                                           TYPE_TAG_3D, handle);
          }
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_intersection_internal(Context ctx,
                        IndexPartition parent, const void *realm_color, 
                        TypeTag type_tag, IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_space_intersection(ctx, parent, realm_color,
                                                      type_tag, handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_difference(Context ctx,
          IndexPartition parent, const DomainPoint &color, IndexSpace initial, 
          const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      switch (color.get_dim())
      {
        case 1:
          {
            Point<1,coord_t> point = color;
            return runtime->create_index_space_difference(ctx, parent, &point,
                                                TYPE_TAG_1D, initial, handles);
          }
        case 2:
          {
            Point<2,coord_t> point = color;
            return runtime->create_index_space_difference(ctx, parent, &point,
                                                TYPE_TAG_2D, initial, handles);
          }
        case 3:
          {
            Point<3,coord_t> point = color;
            return runtime->create_index_space_difference(ctx, parent, &point,
                                                TYPE_TAG_3D, initial, handles);
          }
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_difference_internal(Context ctx,
        IndexPartition parent, const void *realm_color, TypeTag type_tag,
        IndexSpace initial, const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_space_difference(ctx, parent,
                        realm_color, type_tag, initial, handles);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_index_partition(Context ctx, 
                                                IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition(ctx, parent, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_index_partition(Context ctx,
                                    IndexSpace parent, const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      return get_index_partition(ctx, parent, color.get_color());
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_index_partition(IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition(parent, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_index_partition(IndexSpace parent,
                                                const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      return get_index_partition(parent, color.get_color());
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_index_partition(Context ctx, IndexSpace parent, Color c)
    //--------------------------------------------------------------------------
    {
      return runtime->has_index_partition(ctx, parent, c);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_index_partition(Context ctx, IndexSpace parent,
                                               const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      return runtime->has_index_partition(ctx, parent, color.get_color());
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_index_partition(IndexSpace parent, Color c)
    //--------------------------------------------------------------------------
    {
      return runtime->has_index_partition(parent, c);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_index_partition(IndexSpace parent,
                                      const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      return runtime->has_index_partition(parent, color.get_color());
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_subspace(Context ctx, 
                                                  IndexPartition p, Color color)
    //--------------------------------------------------------------------------
    {
      Point<1,coord_t> point = color;
      return runtime->get_index_subspace(ctx, p, &point, TYPE_TAG_1D);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_subspace(Context ctx,
                                     IndexPartition p, const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      switch (color.get_dim())
      {
        case 1:
          {
            Point<1,coord_t> point = color;
            return runtime->get_index_subspace(ctx, p, &point, TYPE_TAG_1D);
          }
        case 2:
          {
            Point<2,coord_t> point = color;
            return runtime->get_index_subspace(ctx, p, &point, TYPE_TAG_2D);
          }
        case 3:
          {
            Point<3,coord_t> point = color;
            return runtime->get_index_subspace(ctx, p, &point, TYPE_TAG_3D);
          }
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_subspace(IndexPartition p, Color color)
    //--------------------------------------------------------------------------
    {
      Point<1,coord_t> point = color;
      return runtime->get_index_subspace(p, &point, TYPE_TAG_1D);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_subspace(IndexPartition p, 
                                           const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      switch (color.get_dim())
      {
        case 1:
          {
            Point<1,coord_t> point = color;
            return runtime->get_index_subspace(p, &point, TYPE_TAG_1D);
          }
        case 2:
          {
            Point<2,coord_t> point = color;
            return runtime->get_index_subspace(p, &point, TYPE_TAG_2D);
          }
        case 3:
          {
            Point<3,coord_t> point = color;
            return runtime->get_index_subspace(p, &point, TYPE_TAG_3D);
          }
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_subspace_internal(IndexPartition p,
                                      const void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_subspace(p, realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_index_subspace(Context ctx, 
                                     IndexPartition p, const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      switch (color.get_dim())
      {
        case 1:
          {
            Point<1,coord_t> point = color;
            return runtime->has_index_subspace(ctx, p, &point, TYPE_TAG_1D);
          }
        case 2:
          {
            Point<2,coord_t> point = color;
            return runtime->has_index_subspace(ctx, p, &point, TYPE_TAG_2D);
          }
        case 3:
          {
            Point<3,coord_t> point = color;
            return runtime->has_index_subspace(ctx, p, &point, TYPE_TAG_3D);
          }
        default:
          assert(false);
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_index_subspace(IndexPartition p, const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      switch (color.get_dim())
      {
        case 1:
          {
            Point<1,coord_t> point = color;
            return runtime->has_index_subspace(p, &point, TYPE_TAG_1D);
          }
        case 2:
          {
            Point<2,coord_t> point = color;
            return runtime->has_index_subspace(p, &point, TYPE_TAG_2D);
          }
        case 3:
          {
            Point<3,coord_t> point = color;
            return runtime->has_index_subspace(p, &point, TYPE_TAG_3D);
          }
        default:
          assert(false);
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_index_subspace_internal(IndexPartition p,
                                      const void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      return runtime->has_index_subspace(p, realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_multiple_domains(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      // Multiple domains supported implicitly
      return false;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_multiple_domains(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      // Multiple domains supported implicitly
      return false;
    }

    //--------------------------------------------------------------------------
    Domain Runtime::get_index_space_domain(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      const TypeTag type_tag = handle.get_type_tag();
      switch (Internal::NT_TemplateHelper::get_dim(type_tag))
      {
        case 1:
          {
            DomainT<1,coord_t> realm_is;
            runtime->get_index_space_domain(ctx, handle, &realm_is, type_tag);
            return Domain(realm_is);
          }
        case 2:
          {
            DomainT<2,coord_t> realm_is;
            runtime->get_index_space_domain(ctx, handle, &realm_is, type_tag);
            return Domain(realm_is);
          }
        case 3:
          {
            DomainT<3,coord_t> realm_is;
            runtime->get_index_space_domain(ctx, handle, &realm_is, type_tag);
            return Domain(realm_is);
          }
        default:
          assert(false);
      }
      return Domain::NO_DOMAIN;
    }

    //--------------------------------------------------------------------------
    Domain Runtime::get_index_space_domain(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      const TypeTag type_tag = handle.get_type_tag();
      switch (Internal::NT_TemplateHelper::get_dim(type_tag))
      {
        case 1:
          {
            DomainT<1,coord_t> realm_is;
            runtime->get_index_space_domain(handle, &realm_is, type_tag);
            return Domain(realm_is);
          }
        case 2:
          {
            DomainT<2,coord_t> realm_is;
            runtime->get_index_space_domain(handle, &realm_is, type_tag);
            return Domain(realm_is);
          }
        case 3:
          {
            DomainT<3,coord_t> realm_is;
            runtime->get_index_space_domain(handle, &realm_is, type_tag);
            return Domain(realm_is);
          }
        default:
          assert(false);
      }
      return Domain::NO_DOMAIN;
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_domain_internal(IndexSpace handle,
                                         void *realm_is, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      runtime->get_index_space_domain(handle, realm_is, type_tag);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_domains(Context ctx, 
                                IndexSpace handle, std::vector<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      domains.push_back(get_index_space_domain(ctx, handle));
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_domains(IndexSpace handle,
                                          std::vector<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      domains.push_back(get_index_space_domain(handle));
    }

    //--------------------------------------------------------------------------
    Domain Runtime::get_index_partition_color_space(Context ctx, 
                                                    IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition_color_space(ctx, p);
    }

    //--------------------------------------------------------------------------
    Domain Runtime::get_index_partition_color_space(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition_color_space(p);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_partition_color_space_internal(IndexPartition p,
                                               void *realm_is, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      runtime->get_index_partition_color_space(p, realm_is, type_tag);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_partition_color_space_name(Context ctx,
                                                             IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition_color_space_name(ctx, p);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_partition_color_space_name(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition_color_space_name(p);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_partition_colors(Context ctx, IndexSpace sp,
                                                        std::set<Color> &colors)
    //--------------------------------------------------------------------------
    {
      runtime->get_index_space_partition_colors(ctx, sp, colors);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_partition_colors(Context ctx, IndexSpace sp,
                                                  std::set<DomainPoint> &colors)
    //--------------------------------------------------------------------------
    {
      std::set<Color> temp_colors;
      runtime->get_index_space_partition_colors(ctx, sp, temp_colors);
      for (std::set<Color>::const_iterator it = temp_colors.begin();
            it != temp_colors.end(); it++)
        colors.insert(DomainPoint(*it));
    }
    
    //--------------------------------------------------------------------------
    void Runtime::get_index_space_partition_colors(IndexSpace sp,
                                                   std::set<Color> &colors)
    //--------------------------------------------------------------------------
    {
      runtime->get_index_space_partition_colors(sp, colors);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_partition_colors(IndexSpace sp,
                                                  std::set<DomainPoint> &colors)
    //--------------------------------------------------------------------------
    {
      std::set<Color> temp_colors;
      runtime->get_index_space_partition_colors(sp, temp_colors);
      for (std::set<Color>::const_iterator it = temp_colors.begin();
            it != temp_colors.end(); it++)
        colors.insert(DomainPoint(*it));
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_index_partition_disjoint(Context ctx, IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return runtime->is_index_partition_disjoint(ctx, p);
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_index_partition_disjoint(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return runtime->is_index_partition_disjoint(p);
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_index_partition_complete(Context ctx, IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return runtime->is_index_partition_complete(ctx, p);
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_index_partition_complete(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return runtime->is_index_partition_complete(p);
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_index_space_color(Context ctx, 
                                                  IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      Point<1,coord_t> point;
      runtime->get_index_space_color_point(ctx, handle, &point, TYPE_TAG_1D);
      return point[0];
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_index_space_color(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      Point<1,coord_t> point;
      runtime->get_index_space_color_point(handle, &point, TYPE_TAG_1D);
      return point[0];
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::get_index_space_color_point(Context ctx,
                                                              IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_space_color_point(ctx, handle); 
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::get_index_space_color_point(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_space_color_point(handle);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_color_internal(IndexSpace handle,
                                            void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      runtime->get_index_space_color_point(handle, realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_index_partition_color(Context ctx,
                                                      IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_index_partition_color(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition_color(handle);
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::get_index_partition_color_point(Context ctx,
                                                          IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return DomainPoint(runtime->get_index_partition_color(ctx, handle));
    }
    
    //--------------------------------------------------------------------------
    DomainPoint Runtime::get_index_partition_color_point(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return DomainPoint(runtime->get_index_partition_color(handle));
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_parent_index_space(Context ctx,
                                                        IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_index_space(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_parent_index_space(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_index_space(handle);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_parent_index_partition(Context ctx,
                                                      IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->has_parent_index_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_parent_index_partition(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->has_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_parent_index_partition(Context ctx,
                                                              IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_index_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_parent_index_partition(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::get_index_space_depth(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_space_depth(ctx, handle);
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::get_index_space_depth(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_space_depth(handle);
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::get_index_partition_depth(Context ctx,  
                                                IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition_depth(ctx, handle);
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::get_index_partition_depth(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition_depth(handle);
    }

    //--------------------------------------------------------------------------
    ptr_t Runtime::safe_cast(Context ctx, ptr_t pointer, 
                                      LogicalRegion region)
    //--------------------------------------------------------------------------
    {
      if (pointer.is_null())
        return pointer;
      Point<1,coord_t> p(pointer.value);
      if (runtime->safe_cast(ctx, region, &p, TYPE_TAG_1D))
        return pointer;
      return ptr_t::nil();
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::safe_cast(Context ctx, DomainPoint point, 
                                            LogicalRegion region)
    //--------------------------------------------------------------------------
    {
      switch (point.get_dim())
      {
        case 1:
          {
            Point<1,coord_t> p(point);
            if (runtime->safe_cast(ctx, region, &p, TYPE_TAG_1D))
              return point;
            break;
          }
        case 2:
          {
            Point<2,coord_t> p(point);
            if (runtime->safe_cast(ctx, region, &p, TYPE_TAG_2D)) 
              return point;
            break;
          }
        case 3:
          {
            Point<3,coord_t> p(point);
            if (runtime->safe_cast(ctx, region, &p, TYPE_TAG_3D)) 
              return point;
            break;
          }
        default:
          assert(false);
      }
      return DomainPoint::nil();
    }

    //--------------------------------------------------------------------------
    bool Runtime::safe_cast_internal(Context ctx, LogicalRegion region,
                                     const void *realm_point, TypeTag type_tag) 
    //--------------------------------------------------------------------------
    {
      return runtime->safe_cast(ctx, region, realm_point, type_tag);
    }

    //--------------------------------------------------------------------------
    FieldSpace Runtime::create_field_space(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->create_field_space(ctx);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_field_space(Context ctx, FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_field_space(ctx, handle);
    }

    //--------------------------------------------------------------------------
    size_t Runtime::get_field_size(Context ctx, FieldSpace handle,
                                            FieldID fid)
    //--------------------------------------------------------------------------
    {
      return runtime->get_field_size(ctx, handle, fid);
    }

    //--------------------------------------------------------------------------
    size_t Runtime::get_field_size(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      return runtime->get_field_size(handle, fid);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_field_space_fields(Context ctx, FieldSpace handle,
                                         std::vector<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      runtime->get_field_space_fields(ctx, handle, fields);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_field_space_fields(FieldSpace handle,
                                         std::vector<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      runtime->get_field_space_fields(handle, fields);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_field_space_fields(Context ctx, FieldSpace handle,
                                         std::set<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      std::vector<FieldID> local;
      runtime->get_field_space_fields(ctx, handle, local);
      fields.insert(local.begin(), local.end());
    }

    //--------------------------------------------------------------------------
    void Runtime::get_field_space_fields(FieldSpace handle,
                                         std::set<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      std::vector<FieldID> local;
      runtime->get_field_space_fields(handle, local);
      fields.insert(local.begin(), local.end());
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::create_logical_region(Context ctx, 
                           IndexSpace index, FieldSpace fields, bool task_local)
    //--------------------------------------------------------------------------
    {
      return runtime->create_logical_region(ctx, index, fields, task_local);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_logical_region(Context ctx, 
                                                  LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_logical_region(ctx, handle);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_logical_partition(Context ctx, 
                                                     LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_logical_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition(Context ctx, 
                                    LogicalRegion parent, IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition(ctx, parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition(LogicalRegion parent, 
                                                    IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_color(
                                    Context ctx, LogicalRegion parent, Color c)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_by_color(ctx, parent, c);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_color(
                        Context ctx, LogicalRegion parent, const DomainPoint &c)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_by_color(ctx, parent,c.get_color());
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_color(
                                                  LogicalRegion parent, Color c)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_by_color(parent, c);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_color(
                                     LogicalRegion parent, const DomainPoint &c)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_by_color(parent, c.get_color());
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_logical_partition_by_color(Context ctx,
                                     LogicalRegion parent, const DomainPoint &c)
    //--------------------------------------------------------------------------
    {
      return runtime->has_logical_partition_by_color(ctx, parent,c.get_color());
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_logical_partition_by_color(LogicalRegion parent, 
                                                 const DomainPoint &c)
    //--------------------------------------------------------------------------
    {
      return runtime->has_logical_partition_by_color(parent, c.get_color());
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_tree(
                                            Context ctx, IndexPartition handle, 
                                            FieldSpace fspace, RegionTreeID tid) 
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_by_tree(ctx, handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_tree(
                                            IndexPartition handle, 
                                            FieldSpace fspace, RegionTreeID tid) 
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_by_tree(handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion(Context ctx, 
                                    LogicalPartition parent, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion(ctx, parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion(LogicalPartition parent, 
                                                 IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_color(Context ctx, 
                                             LogicalPartition parent, Color c)
    //--------------------------------------------------------------------------
    {
      Point<1,coord_t> point(c);
      return runtime->get_logical_subregion_by_color(ctx, parent, 
                                                     &point, TYPE_TAG_1D);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_color(Context ctx,
                                  LogicalPartition parent, const DomainPoint &c)
    //--------------------------------------------------------------------------
    {
      switch (c.get_dim())
      {
        case 1:
          {
            Point<1,coord_t> point(c);
            return runtime->get_logical_subregion_by_color(ctx, parent, 
                                                           &point, TYPE_TAG_1D);
          }
        case 2:
          {
            Point<2,coord_t> point(c);
            return runtime->get_logical_subregion_by_color(ctx, parent, 
                                                           &point, TYPE_TAG_2D);
          }
        case 3:
          {
            Point<3,coord_t> point(c);
            return runtime->get_logical_subregion_by_color(ctx, parent, 
                                                           &point, TYPE_TAG_3D);
          }
        default:
          assert(false);
      }
      return LogicalRegion::NO_REGION;
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_color(
                                               LogicalPartition parent, Color c)
    //--------------------------------------------------------------------------
    {
      Point<1,coord_t> point(c);
      return runtime->get_logical_subregion_by_color(parent,&point,TYPE_TAG_1D);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_color(
                                  LogicalPartition parent, const DomainPoint &c)
    //--------------------------------------------------------------------------
    {
      switch (c.get_dim())
      {
        case 1:
          {
            Point<1,coord_t> point(c);
            return runtime->get_logical_subregion_by_color(parent, &point,
                                                           TYPE_TAG_1D);
          }
        case 2:
          {
            Point<2,coord_t> point(c);
            return runtime->get_logical_subregion_by_color(parent, &point,
                                                           TYPE_TAG_2D);
          }
        case 3:
          {
            Point<3,coord_t> point(c);
            return runtime->get_logical_subregion_by_color(parent, &point,
                                                           TYPE_TAG_3D);
          }
        default:
          assert(false);
      }
      return LogicalRegion::NO_REGION;
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_color_internal(
             LogicalPartition parent, const void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion_by_color(parent, 
                                                     realm_color, type_tag);
    }
    
    //--------------------------------------------------------------------------
    bool Runtime::has_logical_subregion_by_color(Context ctx,
                                  LogicalPartition parent, const DomainPoint &c)
    //--------------------------------------------------------------------------
    {
      switch (c.get_dim())
      {
        case 1:
          {
            Point<1,coord_t> point(c);
            return runtime->has_logical_subregion_by_color(ctx, parent, &point,
                                                           TYPE_TAG_1D);
          }
        case 2:
          {
            Point<2,coord_t> point(c);
            return runtime->has_logical_subregion_by_color(ctx, parent, &point,
                                                           TYPE_TAG_2D);
          }
        case 3:
          {
            Point<3,coord_t> point(c);
            return runtime->has_logical_subregion_by_color(ctx, parent, &point,
                                                           TYPE_TAG_3D);
          }
        default:
          assert(false);
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_logical_subregion_by_color(LogicalPartition parent, 
                                                 const DomainPoint &c)
    //--------------------------------------------------------------------------
    {
      switch (c.get_dim())
      {
        case 1:
          {
            Point<1,coord_t> point(c);
            return runtime->has_logical_subregion_by_color(parent, &point,
                                                           TYPE_TAG_1D);
          }
        case 2:
          {
            Point<2,coord_t> point(c);
            return runtime->has_logical_subregion_by_color(parent, &point,
                                                           TYPE_TAG_2D);
          }
        case 3:
          {
            Point<3,coord_t> point(c);
            return runtime->has_logical_subregion_by_color(parent, &point,
                                                           TYPE_TAG_3D);
          }
        default:
          assert(false);
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_logical_subregion_by_color_internal(
             LogicalPartition parent, const void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      return runtime->has_logical_subregion_by_color(parent, 
                                                     realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_tree(Context ctx, 
                        IndexSpace handle, FieldSpace fspace, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion_by_tree(ctx, handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_tree(IndexSpace handle, 
                                            FieldSpace fspace, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion_by_tree(handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_logical_region_color(Context ctx, LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      Point<1,coord_t> point;
      runtime->get_logical_region_color(ctx, handle, &point, TYPE_TAG_1D);
      return point[0];
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::get_logical_region_color_point(Context ctx,
                                                        LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_region_color_point(ctx, handle); 
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_logical_region_color(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      Point<1,coord_t> point;
      runtime->get_logical_region_color(handle, &point, TYPE_TAG_1D);
      return point[0];
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::get_logical_region_color_point(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_region_color_point(handle); 
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_logical_partition_color(Context ctx,
                                                        LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::get_logical_partition_color_point(Context ctx,
                                                        LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return DomainPoint(runtime->get_logical_partition_color(ctx, handle));
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_logical_partition_color(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_color(handle);
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::get_logical_partition_color_point(
                                                        LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return DomainPoint(runtime->get_logical_partition_color(handle));
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_parent_logical_region(Context ctx,
                                                        LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_logical_region(ctx, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_parent_logical_region(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_logical_region(handle);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_parent_logical_partition(Context ctx,
                                                        LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return runtime->has_parent_logical_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_parent_logical_partition(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return runtime->has_parent_logical_partition(handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_parent_logical_partition(Context ctx,
                                                           LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_logical_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_parent_logical_partition(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_logical_partition(handle);
    }

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    //--------------------------------------------------------------------------
    IndexAllocator Runtime::create_index_allocator(Context ctx, IndexSpace is)
    //--------------------------------------------------------------------------
    {
      Internal::log_run.warning("Dynamic index space allocation is no longer "
                                "supported. You can only make one allocator "
                                "per index space and it must always be in the "
                                "same task that created the index space.");
      return IndexAllocator(is, IndexIterator(this, ctx, is));
    }
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif

    //--------------------------------------------------------------------------
    FieldAllocator Runtime::create_field_allocator(Context ctx, 
                                                            FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->create_field_allocator(ctx, handle);
    }

    //--------------------------------------------------------------------------
    ArgumentMap Runtime::create_argument_map(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->create_argument_map();
    }

    //--------------------------------------------------------------------------
    Future Runtime::execute_task(Context ctx, 
                                          const TaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return runtime->execute_task(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    FutureMap Runtime::execute_index_space(Context ctx, 
                                              const IndexTaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return runtime->execute_index_space(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    Future Runtime::execute_index_space(Context ctx, 
                         const IndexTaskLauncher &launcher, ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
      return runtime->execute_index_space(ctx, launcher, redop);
    }

    //--------------------------------------------------------------------------
    Future Runtime::execute_task(Context ctx, 
                        Processor::TaskFuncID task_id,
                        const std::vector<IndexSpaceRequirement> &indexes,
                        const std::vector<FieldSpaceRequirement> &fields,
                        const std::vector<RegionRequirement> &regions,
                        const TaskArgument &arg, 
                        const Predicate &predicate,
                        MapperID id, 
                        MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      TaskLauncher launcher(task_id, arg, predicate, id, tag);
      launcher.index_requirements = indexes;
      launcher.region_requirements = regions;
      return runtime->execute_task(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    FutureMap Runtime::execute_index_space(Context ctx, 
                        Processor::TaskFuncID task_id,
                        const Domain domain,
                        const std::vector<IndexSpaceRequirement> &indexes,
                        const std::vector<FieldSpaceRequirement> &fields,
                        const std::vector<RegionRequirement> &regions,
                        const TaskArgument &global_arg, 
                        const ArgumentMap &arg_map,
                        const Predicate &predicate,
                        bool must_parallelism, 
                        MapperID id, 
                        MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      IndexTaskLauncher launcher(task_id, domain, global_arg, arg_map,
                                 predicate, must_parallelism, id, tag);
      launcher.index_requirements = indexes;
      launcher.region_requirements = regions;
      return runtime->execute_index_space(ctx, launcher);
    }


    //--------------------------------------------------------------------------
    Future Runtime::execute_index_space(Context ctx, 
                        Processor::TaskFuncID task_id,
                        const Domain domain,
                        const std::vector<IndexSpaceRequirement> &indexes,
                        const std::vector<FieldSpaceRequirement> &fields,
                        const std::vector<RegionRequirement> &regions,
                        const TaskArgument &global_arg, 
                        const ArgumentMap &arg_map,
                        ReductionOpID reduction, 
                        const TaskArgument &initial_value,
                        const Predicate &predicate,
                        bool must_parallelism, 
                        MapperID id, 
                        MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      IndexTaskLauncher launcher(task_id, domain, global_arg, arg_map,
                                 predicate, must_parallelism, id, tag);
      launcher.index_requirements = indexes;
      launcher.region_requirements = regions;
      return runtime->execute_index_space(ctx, launcher, reduction);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Runtime::map_region(Context ctx, 
                                                const InlineLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return runtime->map_region(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Runtime::map_region(Context ctx, 
                    const RegionRequirement &req, MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      InlineLauncher launcher(req, id, tag);
      return runtime->map_region(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Runtime::map_region(Context ctx, unsigned idx, 
                                                  MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->map_region(ctx, idx, id, tag);
    }

    //--------------------------------------------------------------------------
    void Runtime::remap_region(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      runtime->remap_region(ctx, region);
    }

    //--------------------------------------------------------------------------
    void Runtime::unmap_region(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      runtime->unmap_region(ctx, region);
    }

    //--------------------------------------------------------------------------
    void Runtime::unmap_all_regions(Context ctx)
    //--------------------------------------------------------------------------
    {
      runtime->unmap_all_regions(ctx);
    }

    //--------------------------------------------------------------------------
    void Runtime::fill_field(Context ctx, LogicalRegion handle,
                                      LogicalRegion parent, FieldID fid,
                                      const void *value, size_t size,
                                      Predicate pred)
    //--------------------------------------------------------------------------
    {
      FillLauncher launcher(handle, parent, TaskArgument(value, size), pred);
      launcher.add_field(fid);
      runtime->fill_fields(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    void Runtime::fill_field(Context ctx, LogicalRegion handle,
                                      LogicalRegion parent, FieldID fid,
                                      Future f, Predicate pred)
    //--------------------------------------------------------------------------
    {
      FillLauncher launcher(handle, parent, TaskArgument(), pred);
      launcher.set_future(f);
      launcher.add_field(fid);
      runtime->fill_fields(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    void Runtime::fill_fields(Context ctx, LogicalRegion handle,
                                       LogicalRegion parent,
                                       const std::set<FieldID> &fields,
                                       const void *value, size_t size,
                                       Predicate pred)
    //--------------------------------------------------------------------------
    {
      FillLauncher launcher(handle, parent, TaskArgument(value, size), pred);
      launcher.fields = fields;
      runtime->fill_fields(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    void Runtime::fill_fields(Context ctx, LogicalRegion handle,
                                       LogicalRegion parent, 
                                       const std::set<FieldID> &fields,
                                       Future f, Predicate pred)
    //--------------------------------------------------------------------------
    {
      FillLauncher launcher(handle, parent, TaskArgument(), pred);
      launcher.set_future(f);
      launcher.fields = fields;
      runtime->fill_fields(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    void Runtime::fill_fields(Context ctx, const FillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      runtime->fill_fields(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    void Runtime::fill_fields(Context ctx, const IndexFillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      runtime->fill_fields(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Runtime::attach_external_resource(Context ctx, 
                                                 const AttachLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return runtime->attach_external_resource(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    Future Runtime::detach_external_resource(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      return runtime->detach_external_resource(ctx, region);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Runtime::attach_hdf5(Context ctx, 
                                                 const char *file_name,
                                                 LogicalRegion handle,
                                                 LogicalRegion parent,
                                 const std::map<FieldID,const char*> &field_map,
                                                 LegionFileMode mode)
    //--------------------------------------------------------------------------
    {
      AttachLauncher launcher(EXTERNAL_HDF5_FILE, handle, parent);
      launcher.attach_hdf5(file_name, field_map, mode);
      return runtime->attach_external_resource(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    void Runtime::detach_hdf5(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      runtime->detach_external_resource(ctx, region);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Runtime::attach_file(Context ctx,
                                                 const char *file_name,
                                                 LogicalRegion handle,
                                                 LogicalRegion parent,
                                 const std::vector<FieldID> &field_vec,
                                                 LegionFileMode mode)
    //--------------------------------------------------------------------------
    {
      AttachLauncher launcher(EXTERNAL_POSIX_FILE, handle, parent);
      launcher.attach_file(file_name, field_vec, mode);
      return runtime->attach_external_resource(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    void Runtime::detach_file(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      runtime->detach_external_resource(ctx, region);
    }
    
    //--------------------------------------------------------------------------
    void Runtime::issue_copy_operation(Context ctx,const CopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      runtime->issue_copy_operation(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_copy_operation(Context ctx,
                                       const IndexCopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      runtime->issue_copy_operation(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    Predicate Runtime::create_predicate(Context ctx, const Future &f)
    //--------------------------------------------------------------------------
    {
      return runtime->create_predicate(ctx, f);
    }

    //--------------------------------------------------------------------------
    Predicate Runtime::predicate_not(Context ctx, const Predicate &p) 
    //--------------------------------------------------------------------------
    {
      return runtime->predicate_not(ctx, p);
    }

    //--------------------------------------------------------------------------
    Predicate Runtime::predicate_and(Context ctx, 
                                       const Predicate &p1, const Predicate &p2)
    //--------------------------------------------------------------------------
    {
      PredicateLauncher launcher(true/*and*/);
      launcher.add_predicate(p1);
      launcher.add_predicate(p2);
      return runtime->create_predicate(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    Predicate Runtime::predicate_or(Context ctx,
                                       const Predicate &p1, const Predicate &p2)  
    //--------------------------------------------------------------------------
    {
      PredicateLauncher launcher(false/*and*/);
      launcher.add_predicate(p1);
      launcher.add_predicate(p2);
      return runtime->create_predicate(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    Predicate Runtime::create_predicate(Context ctx, 
                                        const PredicateLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return runtime->create_predicate(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    Future Runtime::get_predicate_future(Context ctx, const Predicate &p)
    //--------------------------------------------------------------------------
    {
      return runtime->get_predicate_future(ctx, p);
    }

    //--------------------------------------------------------------------------
    Lock Runtime::create_lock(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->create_lock(ctx);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_lock(Context ctx, Lock l)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_lock(ctx, l);
    }

    //--------------------------------------------------------------------------
    Grant Runtime::acquire_grant(Context ctx,
                                      const std::vector<LockRequest> &requests)
    //--------------------------------------------------------------------------
    {
      return runtime->acquire_grant(ctx, requests);
    }

    //--------------------------------------------------------------------------
    void Runtime::release_grant(Context ctx, Grant grant)
    //--------------------------------------------------------------------------
    {
      runtime->release_grant(ctx, grant);
    }

    //--------------------------------------------------------------------------
    PhaseBarrier Runtime::create_phase_barrier(Context ctx, 
                                                        unsigned arrivals)
    //--------------------------------------------------------------------------
    {
      return runtime->create_phase_barrier(ctx, arrivals);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_phase_barrier(Context ctx, PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_phase_barrier(ctx, pb);
    }

    //--------------------------------------------------------------------------
    PhaseBarrier Runtime::advance_phase_barrier(Context ctx, 
                                                         PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      return runtime->advance_phase_barrier(ctx, pb);
    }

    //--------------------------------------------------------------------------
    DynamicCollective Runtime::create_dynamic_collective(Context ctx,
                                                        unsigned arrivals,
                                                        ReductionOpID redop,
                                                        const void *init_value,
                                                        size_t init_size)
    //--------------------------------------------------------------------------
    {
      return runtime->create_dynamic_collective(ctx, arrivals, redop,
                                                init_value, init_size);
    }
    
    //--------------------------------------------------------------------------
    void Runtime::destroy_dynamic_collective(Context ctx, 
                                                      DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_dynamic_collective(ctx, dc);
    }

    //--------------------------------------------------------------------------
    void Runtime::arrive_dynamic_collective(Context ctx,
                                                     DynamicCollective dc,
                                                     const void *buffer,
                                                     size_t size, 
                                                     unsigned count)
    //--------------------------------------------------------------------------
    {
      runtime->arrive_dynamic_collective(ctx, dc, buffer, size, count);
    }

    //--------------------------------------------------------------------------
    void Runtime::defer_dynamic_collective_arrival(Context ctx,
                                                   DynamicCollective dc,
                                                   const Future &f, 
                                                   unsigned count)
    //--------------------------------------------------------------------------
    {
      runtime->defer_dynamic_collective_arrival(ctx, dc, f, count);
    }

    //--------------------------------------------------------------------------
    Future Runtime::get_dynamic_collective_result(Context ctx,
                                                           DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      return runtime->get_dynamic_collective_result(ctx, dc);
    }

    //--------------------------------------------------------------------------
    DynamicCollective Runtime::advance_dynamic_collective(Context ctx,
                                                           DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      return runtime->advance_dynamic_collective(ctx, dc);
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_acquire(Context ctx,
                                         const AcquireLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      runtime->issue_acquire(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_release(Context ctx,
                                         const ReleaseLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      runtime->issue_release(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_mapping_fence(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->issue_mapping_fence(ctx);
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_execution_fence(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->issue_execution_fence(ctx);
    }

    //--------------------------------------------------------------------------
    void Runtime::begin_trace(
                        Context ctx, TraceID tid, bool logical_only /*= false*/)
    //--------------------------------------------------------------------------
    {
      runtime->begin_trace(ctx, tid, logical_only);
    }

    //--------------------------------------------------------------------------
    void Runtime::end_trace(Context ctx, TraceID tid)
    //--------------------------------------------------------------------------
    {
      runtime->end_trace(ctx, tid);
    }

    //--------------------------------------------------------------------------
    void Runtime::begin_static_trace(Context ctx,
                                     const std::set<RegionTreeID> *managed)
    //--------------------------------------------------------------------------
    {
      runtime->begin_static_trace(ctx, managed);
    }

    //--------------------------------------------------------------------------
    void Runtime::end_static_trace(Context ctx)
    //--------------------------------------------------------------------------
    {
      runtime->end_static_trace(ctx);
    }

    //--------------------------------------------------------------------------
    void Runtime::complete_frame(Context ctx)
    //--------------------------------------------------------------------------
    {
      runtime->complete_frame(ctx);
    }

    //--------------------------------------------------------------------------
    FutureMap Runtime::execute_must_epoch(Context ctx,
                                              const MustEpochLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return runtime->execute_must_epoch(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    Future Runtime::select_tunable_value(Context ctx, TunableID tid,
                                         MapperID mid, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->select_tunable_value(ctx, tid, mid, tag);
    }

    //--------------------------------------------------------------------------
    int Runtime::get_tunable_value(Context ctx, TunableID tid,
                                            MapperID mid, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->get_tunable_value(ctx, tid, mid, tag);
    }

    //--------------------------------------------------------------------------
    const Task* Runtime::get_local_task(Context ctx)
    //--------------------------------------------------------------------------
    {
      return ctx->get_task();
    }

    //--------------------------------------------------------------------------
    void* Runtime::get_local_task_variable_untyped(Context ctx,
                                                   LocalVariableID id)
    //--------------------------------------------------------------------------
    {
      return runtime->get_local_task_variable(ctx, id);
    }

    //--------------------------------------------------------------------------
    void Runtime::set_local_task_variable_untyped(Context ctx,
               LocalVariableID id, const void* value, void (*destructor)(void*))
    //--------------------------------------------------------------------------
    {
      runtime->set_local_task_variable(ctx, id, value, destructor);
    }

    //--------------------------------------------------------------------------
    Future Runtime::get_current_time(Context ctx, Future precondition)
    //--------------------------------------------------------------------------
    {
      TimingLauncher launcher(MEASURE_SECONDS);
      launcher.add_precondition(precondition);
      return runtime->issue_timing_measurement(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    Future Runtime::get_current_time_in_microseconds(Context ctx, Future pre)
    //--------------------------------------------------------------------------
    {
      TimingLauncher launcher(MEASURE_MICRO_SECONDS);
      launcher.add_precondition(pre);
      return runtime->issue_timing_measurement(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    Future Runtime::get_current_time_in_nanoseconds(Context ctx, Future pre)
    //--------------------------------------------------------------------------
    {
      TimingLauncher launcher(MEASURE_NANO_SECONDS);
      launcher.add_precondition(pre);
      return runtime->issue_timing_measurement(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    Future Runtime::issue_timing_measurement(Context ctx, 
                                             const TimingLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return runtime->issue_timing_measurement(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    /*static*/ long long Runtime::get_zero_time(void)
    //--------------------------------------------------------------------------
    {
      return Realm::Clock::get_zero_time();
    }

    //--------------------------------------------------------------------------
    Mapping::Mapper* Runtime::get_mapper(Context ctx, MapperID id,
                                         Processor target)
    //--------------------------------------------------------------------------
    {
      return runtime->get_mapper(ctx, id, target);
    }

    //--------------------------------------------------------------------------
    Processor Runtime::get_executing_processor(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->get_executing_processor(ctx);
    }

    //--------------------------------------------------------------------------
    void Runtime::raise_region_exception(Context ctx, 
                                                  PhysicalRegion region,
                                                  bool nuclear)
    //--------------------------------------------------------------------------
    {
      runtime->raise_region_exception(ctx, region, nuclear);
    }

    //--------------------------------------------------------------------------
    const std::map<int,AddressSpace>& 
                                Runtime::find_forward_MPI_mapping(void)
    //--------------------------------------------------------------------------
    {
      return runtime->find_forward_MPI_mapping();
    }

    //--------------------------------------------------------------------------
    const std::map<AddressSpace,int>&
                                Runtime::find_reverse_MPI_mapping(void)
    //--------------------------------------------------------------------------
    {
      return runtime->find_reverse_MPI_mapping();
    }

    //--------------------------------------------------------------------------
    int Runtime::find_local_MPI_rank(void)
    //--------------------------------------------------------------------------
    {
      return runtime->find_local_MPI_rank();
    }

    //--------------------------------------------------------------------------
    Mapping::MapperRuntime* Runtime::get_mapper_runtime(void)
    //--------------------------------------------------------------------------
    {
      return runtime->get_mapper_runtime();
    }

    //--------------------------------------------------------------------------
    MapperID Runtime::generate_dynamic_mapper_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_mapper_id();
    }

    //--------------------------------------------------------------------------
    MapperID Runtime::generate_library_mapper_ids(const char *name, size_t cnt)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_library_mapper_ids(name, cnt);
    }

    //--------------------------------------------------------------------------
    /*static*/ MapperID Runtime::generate_static_mapper_id(void)
    //--------------------------------------------------------------------------
    {
      return Internal::Runtime::generate_static_mapper_id();
    }

    //--------------------------------------------------------------------------
    void Runtime::add_mapper(MapperID map_id, Mapping::Mapper *mapper, 
                             Processor proc)
    //--------------------------------------------------------------------------
    {
      runtime->add_mapper(map_id, mapper, proc);
    }

    //--------------------------------------------------------------------------
    void Runtime::replace_default_mapper(Mapping::Mapper *mapper,Processor proc)
    //--------------------------------------------------------------------------
    {
      runtime->replace_default_mapper(mapper, proc);
    }

    //--------------------------------------------------------------------------
    ProjectionID Runtime::generate_dynamic_projection_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_projection_id();
    }

    //--------------------------------------------------------------------------
    ProjectionID Runtime::generate_library_projection_ids(const char *name,
                                                          size_t count)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_library_projection_ids(name, count);
    }

    //--------------------------------------------------------------------------
    /*static*/ ProjectionID Runtime::generate_static_projection_id(void)
    //--------------------------------------------------------------------------
    {
      return Internal::Runtime::generate_static_projection_id();
    }

    //--------------------------------------------------------------------------
    void Runtime::register_projection_functor(ProjectionID pid,
                                              ProjectionFunctor *func,
                                              bool silence_warnings)
    //--------------------------------------------------------------------------
    {
      runtime->register_projection_functor(pid, func, true/*need zero check*/,
                                           silence_warnings);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::preregister_projection_functor(ProjectionID pid,
                                                        ProjectionFunctor *func)
    //--------------------------------------------------------------------------
    {
      Internal::Runtime::preregister_projection_functor(pid, func);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(TaskID task_id, SemanticTag tag,
                                   const void *buffer, size_t size, bool is_mut)
    //--------------------------------------------------------------------------
    {
      runtime->attach_semantic_information(task_id, tag, buffer, size, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(IndexSpace handle,
                                                       SemanticTag tag,
                                                       const void *buffer,
                                                       size_t size, bool is_mut)
    //--------------------------------------------------------------------------
    {
      runtime->attach_semantic_information(handle, tag, buffer, size, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(IndexPartition handle,
                                                       SemanticTag tag,
                                                       const void *buffer,
                                                       size_t size, bool is_mut)
    //--------------------------------------------------------------------------
    {
      runtime->attach_semantic_information(handle, tag, buffer, size, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(FieldSpace handle,
                                                       SemanticTag tag,
                                                       const void *buffer,
                                                       size_t size, bool is_mut)
    //--------------------------------------------------------------------------
    {
      runtime->attach_semantic_information(handle, tag, buffer, size, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(FieldSpace handle,
                                                       FieldID fid,
                                                       SemanticTag tag,
                                                       const void *buffer,
                                                       size_t size, bool is_mut)
    //--------------------------------------------------------------------------
    {
      runtime->attach_semantic_information(handle, fid, tag, buffer, 
                                           size, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(LogicalRegion handle,
                                                       SemanticTag tag,
                                                       const void *buffer,
                                                       size_t size, bool is_mut)
    //--------------------------------------------------------------------------
    {
      runtime->attach_semantic_information(handle, tag, buffer, size, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(LogicalPartition handle,
                                                       SemanticTag tag,
                                                       const void *buffer,
                                                       size_t size, bool is_mut)
    //--------------------------------------------------------------------------
    {
      runtime->attach_semantic_information(handle, tag, buffer, size, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_name(TaskID task_id, const char *name, bool is_mutable)
    //--------------------------------------------------------------------------
    {
      Runtime::attach_semantic_information(task_id,
          NAME_SEMANTIC_TAG, name, strlen(name) + 1, is_mutable);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_name(IndexSpace handle, const char *name, bool is_mut)
    //--------------------------------------------------------------------------
    {
      Runtime::attach_semantic_information(handle,
          NAME_SEMANTIC_TAG, name, strlen(name) + 1, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_name(IndexPartition handle, const char *name, bool ism)
    //--------------------------------------------------------------------------
    {
      Runtime::attach_semantic_information(handle,
          NAME_SEMANTIC_TAG, name, strlen(name) + 1, ism);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_name(FieldSpace handle, const char *name, bool is_mut)
    //--------------------------------------------------------------------------
    {
      Runtime::attach_semantic_information(handle,
          NAME_SEMANTIC_TAG, name, strlen(name) + 1, is_mut);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_name(FieldSpace handle,
                                       FieldID fid,
                                       const char *name, bool is_mutable)
    //--------------------------------------------------------------------------
    {
      Runtime::attach_semantic_information(handle, fid,
          NAME_SEMANTIC_TAG, name, strlen(name) + 1, is_mutable);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_name(LogicalRegion handle, const char *name, bool ism)
    //--------------------------------------------------------------------------
    {
      Runtime::attach_semantic_information(handle,
          NAME_SEMANTIC_TAG, name, strlen(name) + 1, ism);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_name(LogicalPartition handle, const char *name, bool m)
    //--------------------------------------------------------------------------
    {
      Runtime::attach_semantic_information(handle,
          NAME_SEMANTIC_TAG, name, strlen(name) + 1, m);
    }

    //--------------------------------------------------------------------------
    bool Runtime::retrieve_semantic_information(TaskID task_id, SemanticTag tag,
                                              const void *&result, size_t &size,
                                                bool can_fail, bool wait_until)
    //--------------------------------------------------------------------------
    {
      return runtime->retrieve_semantic_information(task_id, tag, result, size,
                                                    can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Runtime::retrieve_semantic_information(IndexSpace handle,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size,
                                                         bool can_fail,
                                                         bool wait_until)
    //--------------------------------------------------------------------------
    {
      return runtime->retrieve_semantic_information(handle, tag, result, size,
                                                    can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Runtime::retrieve_semantic_information(IndexPartition handle,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size,
                                                         bool can_fail,
                                                         bool wait_until)
    //--------------------------------------------------------------------------
    {
      return runtime->retrieve_semantic_information(handle, tag, result, size,
                                                    can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Runtime::retrieve_semantic_information(FieldSpace handle,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size,
                                                         bool can_fail,
                                                         bool wait_until)
    //--------------------------------------------------------------------------
    {
      return runtime->retrieve_semantic_information(handle, tag, result, size,
                                                    can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Runtime::retrieve_semantic_information(FieldSpace handle,
                                                         FieldID fid,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size,
                                                         bool can_fail,
                                                         bool wait_until)
    //--------------------------------------------------------------------------
    {
      return runtime->retrieve_semantic_information(handle, fid, tag, result, 
                                                    size, can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Runtime::retrieve_semantic_information(LogicalRegion handle,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size,
                                                         bool can_fail,
                                                         bool wait_until)
    //--------------------------------------------------------------------------
    {
      return runtime->retrieve_semantic_information(handle, tag, result, size,
                                                    can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Runtime::retrieve_semantic_information(LogicalPartition part,
                                                         SemanticTag tag,
                                                         const void *&result,
                                                         size_t &size,
                                                         bool can_fail,
                                                         bool wait_until)
    //--------------------------------------------------------------------------
    {
      return runtime->retrieve_semantic_information(part, tag, result, size,
                                                    can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_name(TaskID task_id, const char *&result)
    //--------------------------------------------------------------------------
    {
      const void* dummy_ptr; size_t dummy_size;
      Runtime::retrieve_semantic_information(task_id, NAME_SEMANTIC_TAG,
                                         dummy_ptr, dummy_size, false, false);
      result = reinterpret_cast<const char*>(dummy_ptr);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_name(IndexSpace handle, const char *&result)
    //--------------------------------------------------------------------------
    {
      const void* dummy_ptr; size_t dummy_size;
      Runtime::retrieve_semantic_information(handle,
          NAME_SEMANTIC_TAG, dummy_ptr, dummy_size, false, false);
      result = reinterpret_cast<const char*>(dummy_ptr);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_name(IndexPartition handle,
                                         const char *&result)
    //--------------------------------------------------------------------------
    {
      const void* dummy_ptr; size_t dummy_size;
      Runtime::retrieve_semantic_information(handle,
          NAME_SEMANTIC_TAG, dummy_ptr, dummy_size, false, false);
      result = reinterpret_cast<const char*>(dummy_ptr);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_name(FieldSpace handle, const char *&result)
    //--------------------------------------------------------------------------
    {
      const void* dummy_ptr; size_t dummy_size;
      Runtime::retrieve_semantic_information(handle,
          NAME_SEMANTIC_TAG, dummy_ptr, dummy_size, false, false);
      result = reinterpret_cast<const char*>(dummy_ptr);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_name(FieldSpace handle,
                                         FieldID fid,
                                         const char *&result)
    //--------------------------------------------------------------------------
    {
      const void* dummy_ptr; size_t dummy_size;
      Runtime::retrieve_semantic_information(handle, fid,
          NAME_SEMANTIC_TAG, dummy_ptr, dummy_size, false, false);
      result = reinterpret_cast<const char*>(dummy_ptr);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_name(LogicalRegion handle,
                                         const char *&result)
    //--------------------------------------------------------------------------
    {
      const void* dummy_ptr; size_t dummy_size;
      Runtime::retrieve_semantic_information(handle,
          NAME_SEMANTIC_TAG, dummy_ptr, dummy_size, false, false);
      result = reinterpret_cast<const char*>(dummy_ptr);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_name(LogicalPartition part,
                                         const char *&result)
    //--------------------------------------------------------------------------
    {
      const void* dummy_ptr; size_t dummy_size;
      Runtime::retrieve_semantic_information(part,
          NAME_SEMANTIC_TAG, dummy_ptr, dummy_size, false, false);
      result = reinterpret_cast<const char*>(dummy_ptr);
    }

    //--------------------------------------------------------------------------
    FieldID Runtime::allocate_field(Context ctx, FieldSpace space,
                                             size_t field_size, FieldID fid,
                                             bool local, CustomSerdezID sd_id)
    //--------------------------------------------------------------------------
    {
      return runtime->allocate_field(ctx, space, field_size, fid, local, sd_id);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_field(Context ctx, FieldSpace sp, FieldID fid)
    //--------------------------------------------------------------------------
    {
      runtime->free_field(ctx, sp, fid);
    }

    //--------------------------------------------------------------------------
    void Runtime::allocate_fields(Context ctx, FieldSpace space,
                                           const std::vector<size_t> &sizes,
                                         std::vector<FieldID> &resulting_fields,
                                         bool local, CustomSerdezID _id)
    //--------------------------------------------------------------------------
    {
      runtime->allocate_fields(ctx, space, sizes, resulting_fields, local, _id);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_fields(Context ctx, FieldSpace space,
                                       const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      runtime->free_fields(ctx, space, to_free);
    }

    //--------------------------------------------------------------------------
    Future Runtime::from_value(const void *value, 
                                        size_t value_size, bool owned)
    //--------------------------------------------------------------------------
    {
      Future result = runtime->help_create_future();
      // Set the future result
      result.impl->set_result(value, value_size, owned);
      // Complete the future right away so that it is always complete
      result.impl->complete_future();
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ int Runtime::start(int argc, char **argv, bool background)
    //--------------------------------------------------------------------------
    {
      return Internal::Runtime::start(argc, argv, background);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::initialize(int *argc, char ***argv)
    //--------------------------------------------------------------------------
    {
      Internal::Runtime::initialize(argc, argv);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::wait_for_shutdown(void)
    //--------------------------------------------------------------------------
    {
      Internal::Runtime::wait_for_shutdown();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::set_top_level_task_id(Processor::TaskFuncID top_id)
    //--------------------------------------------------------------------------
    {
      Internal::Runtime::set_top_level_task_id(top_id);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::configure_MPI_interoperability(int rank)
    //--------------------------------------------------------------------------
    {
      Internal::Runtime::configure_MPI_interoperability(rank);
    }

    //--------------------------------------------------------------------------
    /*static*/ MPILegionHandshake Runtime::create_handshake(bool init_in_MPI,
                                                        int mpi_participants,
                                                        int legion_participants)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mpi_participants > 0);
      assert(legion_participants > 0);
#endif
      MPILegionHandshake result(
          new Internal::MPILegionHandshakeImpl(init_in_MPI,
                                       mpi_participants, legion_participants));
      Internal::Runtime::register_handshake(result);
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ const ReductionOp* Runtime::get_reduction_op(
                                                        ReductionOpID redop_id)
    //--------------------------------------------------------------------------
    {
      return Internal::Runtime::get_reduction_op(redop_id);
    }

    //--------------------------------------------------------------------------
    /*static*/ const SerdezOp* Runtime::get_serdez_op(CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      return Internal::Runtime::get_serdez_op(serdez_id);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::add_registration_callback(
                                            RegistrationCallbackFnptr callback)
    //--------------------------------------------------------------------------
    {
      Internal::Runtime::add_registration_callback(callback);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::set_registration_callback(
                                            RegistrationCallbackFnptr callback)
    //--------------------------------------------------------------------------
    {
      Internal::Runtime::add_registration_callback(callback);
    }

    //--------------------------------------------------------------------------
    /*static*/ const InputArgs& Runtime::get_input_args(void)
    //--------------------------------------------------------------------------
    {
      // If we have an implicit runtime we use that
      if (Internal::implicit_runtime != NULL)
        return Internal::implicit_runtime->input_args;
      // Otherwise this is not from a Legion task, so fallback to the_runtime
      return Internal::Runtime::the_runtime->input_args;
    }

    //--------------------------------------------------------------------------
    /*static*/ Runtime* Runtime::get_runtime(Processor p)
    //--------------------------------------------------------------------------
    {
      // If we have an implicit runtime we use that
      if (Internal::implicit_runtime != NULL)
        return Internal::implicit_runtime->external;
      // Otherwise this is not from a Legion task, so fallback to the_runtime
      return Internal::Runtime::the_runtime->external;
    }

    //--------------------------------------------------------------------------
    /*static*/ Context Runtime::get_context(void)
    //--------------------------------------------------------------------------
    {
      return Internal::implicit_context;
    }

    //--------------------------------------------------------------------------
    /*static*/ ReductionOpTable& Runtime::get_reduction_table(void)
    //--------------------------------------------------------------------------
    {
      return Internal::Runtime::get_reduction_table();
    }

    //--------------------------------------------------------------------------
    /*static*/ SerdezOpTable& Runtime::get_serdez_table(void)
    //--------------------------------------------------------------------------
    {
      return Internal::Runtime::get_serdez_table();
    }

    /*static*/ SerdezRedopTable& Runtime::get_serdez_redop_table(void)
    //--------------------------------------------------------------------------
    {
      return Internal::Runtime::get_serdez_redop_table();
    }

    //--------------------------------------------------------------------------
    TaskID Runtime::generate_dynamic_task_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_task_id();
    }

    //--------------------------------------------------------------------------
    TaskID Runtime::generate_library_task_ids(const char *name, size_t count)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_library_task_ids(name, count);
    }

    //--------------------------------------------------------------------------
    /*static*/ TaskID Runtime::generate_static_task_id(void)
    //--------------------------------------------------------------------------
    {
      return Internal::Runtime::generate_static_task_id();
    }

    //--------------------------------------------------------------------------
    VariantID Runtime::register_task_variant(
                  const TaskVariantRegistrar &registrar,
		  const CodeDescriptor &codedesc,
		  const void *user_data /*= NULL*/,
		  size_t user_len /*= 0*/)
    //--------------------------------------------------------------------------
    {
      // if this needs to be correct, we need two versions...
      bool has_return = false;
      CodeDescriptor *realm_desc = new CodeDescriptor(codedesc);
      return register_variant(registrar, has_return, user_data, user_len,
                              realm_desc);
    }

    //--------------------------------------------------------------------------
    /*static*/ VariantID Runtime::preregister_task_variant(
              const TaskVariantRegistrar &registrar,
	      const CodeDescriptor &codedesc,
	      const void *user_data /*= NULL*/,
	      size_t user_len /*= 0*/,
	      const char *task_name /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      // if this needs to be correct, we need two versions...
      bool has_return = false;
      CodeDescriptor *realm_desc = new CodeDescriptor(codedesc);
      return preregister_variant(registrar, user_data, user_len,
				 realm_desc, has_return, task_name,
                                 AUTO_GENERATE_ID);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::legion_task_preamble(
                                       const void *data, size_t datalen,
                                       Processor p, const Task *& task,
                                       const std::vector<PhysicalRegion> *& reg,
                                       Context& ctx, Runtime *& runtime)
    //--------------------------------------------------------------------------
    {
      // Read the context out of the buffer
#ifdef DEBUG_LEGION
      assert(datalen == sizeof(Context));
#endif
      ctx = *((const Context*)data);
      task = ctx->get_task();

      reg = &ctx->begin_task(runtime);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::legion_task_postamble(Runtime *runtime,Context ctx,
                                                   const void *retvalptr,
                                                   size_t retvalsize)
    //--------------------------------------------------------------------------
    {
      ctx->end_task(retvalptr, retvalsize, false/*owned*/);
    }

    //--------------------------------------------------------------------------
    VariantID Runtime::register_variant(const TaskVariantRegistrar &registrar,
                  bool has_return, const void *user_data, size_t user_data_size,
                  CodeDescriptor *realm)
    //--------------------------------------------------------------------------
    {
      return runtime->register_variant(registrar, user_data, user_data_size,
                                       realm, has_return);
    }
    
    //--------------------------------------------------------------------------
    /*static*/ VariantID Runtime::preregister_variant(
                                  const TaskVariantRegistrar &registrar,
                                  const void *user_data, size_t user_data_size,
                                  CodeDescriptor *realm,
                                  bool has_return, const char *task_name, 
                                  VariantID vid, bool check_task_id)
    //--------------------------------------------------------------------------
    {
      return Internal::Runtime::preregister_variant(registrar, user_data, 
          user_data_size, realm, has_return, task_name, vid, check_task_id);
    } 

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::enable_profiling(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::disable_profiling(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::dump_profiling(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LayoutConstraintID Runtime::register_layout(
                                     const LayoutConstraintRegistrar &registrar)
    //--------------------------------------------------------------------------
    {
      return runtime->register_layout(registrar, AUTO_GENERATE_ID);
    }

    //--------------------------------------------------------------------------
    void Runtime::release_layout(LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      runtime->release_layout(layout_id);
    }

    //--------------------------------------------------------------------------
    /*static*/ LayoutConstraintID Runtime::preregister_layout(
                                     const LayoutConstraintRegistrar &registrar,
                                     LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      return Internal::Runtime::preregister_layout(registrar, layout_id);
    }

    //--------------------------------------------------------------------------
    FieldSpace Runtime::get_layout_constraint_field_space(
                                                   LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      return runtime->get_layout_constraint_field_space(layout_id);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_layout_constraints(LayoutConstraintID layout_id,
                                        LayoutConstraintSet &layout_constraints)
    //--------------------------------------------------------------------------
    {
      runtime->get_layout_constraints(layout_id, layout_constraints);
    }

    //--------------------------------------------------------------------------
    const char* Runtime::get_layout_constraints_name(LayoutConstraintID id)
    //--------------------------------------------------------------------------
    {
      return runtime->get_layout_constraints_name(id);
    }

}; // namespace Legion

// EOF

