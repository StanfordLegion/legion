/* Copyright 2024 Stanford University
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


#include <cmath>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"
#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  WRITER_TASK_ID,
  READER_TASK_ID,
};

enum FieldID {
  FID_X,
};

enum ProjID {
  DIV_PID = 1,
  MOD_PID = 2,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  unsigned num_points = 4;
  unsigned num_elements = 1024;
  // See how many points to run
  const InputArgs &command_args = Runtime::get_input_args();
  for (int i = 1; i < command_args.argc; i++) {
    if (!strcmp(command_args.argv[i],"-p"))
      num_points = atoi(command_args.argv[++i]);
    else if (!strcmp(command_args.argv[i],"-n"))
      num_elements = atoi(command_args.argv[++i]);
  }
  assert(num_points > 0);
  assert(num_elements > 0);
  printf("Running collective_writes for %d points...\n", num_points);
  Rect<1> launch_rect(0, num_points-1);
  IndexSpace launch_space = runtime->create_index_space(ctx, launch_rect);

  Rect<2> elem_rect(Point<2>(0,0),Point<2>(num_elements-1,num_elements-1));
  IndexSpaceT<2> is = runtime->create_index_space(ctx, elem_rect);
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(unsigned), FID_X);
  }
  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  // Make two partitions: one of the rows and one of the columns
  unsigned subregions = std::floor(std::sqrt(num_points));
  Rect<1> color_rect(0, subregions-1);
  IndexSpace cs = runtime->create_index_space(ctx, color_rect);
  unsigned chunk = (num_elements + subregions - 1) / subregions;
  LogicalPartition rows_lp, columns_lp;
  {
    Transform<2,1> transform;
    transform[0][0] = chunk;
    transform[1][0] = 0;
    Rect<2> extent(Point<2>(0,0), Point<2>(chunk-1,num_elements-1));
    IndexPartition rows_ip =
      runtime->create_partition_by_restriction(ctx, is, cs, transform, extent);
    rows_lp = runtime->get_logical_partition(ctx, lr, rows_ip);
  }
  {
    Transform<2,1> transform;
    transform[0][0] = 0;
    transform[1][0] = chunk;
    Rect<2> extent(Point<2>(0,0), Point<2>(num_elements-1,chunk-1));
    IndexPartition columns_ip =
      runtime->create_partition_by_restriction(ctx, is, cs, transform, extent);
    columns_lp = runtime->get_logical_partition(ctx, lr, columns_ip);
  }

  // Launch a collective writing task
  ArgumentMap arg_map;
  {
    IndexTaskLauncher write_launcher(WRITER_TASK_ID, launch_space,
                              TaskArgument(NULL, 0), arg_map);
    write_launcher.add_region_requirement(
        RegionRequirement(rows_lp, DIV_PID, LEGION_WRITE_DISCARD,
          LEGION_COLLECTIVE_EXCLUSIVE, lr));
    write_launcher.add_field(0, FID_X);
    runtime->execute_index_space(ctx, write_launcher);
  }

  // Followed by a collective reading task
  {
    IndexTaskLauncher read_launcher(READER_TASK_ID, launch_space,
        TaskArgument(&subregions, sizeof(subregions)), arg_map);
    read_launcher.add_region_requirement(
        RegionRequirement(columns_lp, MOD_PID, LEGION_READ_ONLY,
          LEGION_COLLECTIVE_EXCLUSIVE, lr));
    read_launcher.add_field(0, FID_X);
    runtime->execute_index_space(ctx, read_launcher);
  }

  runtime->destroy_index_space(ctx, is);
  runtime->destroy_index_space(ctx, cs);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_logical_region(ctx, lr);
}

void writer_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  const FieldAccessor<LEGION_WRITE_ONLY,unsigned,2> acc(regions[0], FID_X);
  Rect<2> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  const unsigned color = runtime->get_index_space_color(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<2> pir(rect); pir(); pir++)
    acc[*pir] = color;
}

void reader_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->arglen == sizeof(unsigned));
  const unsigned expected_colors = *(const unsigned*)task->args;

  const FieldAccessor<LEGION_READ_ONLY,unsigned,2> acc(regions[0], FID_X);
  Rect<2> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  std::vector<unsigned> colors;
  for (PointInRectIterator<2> pir(rect); pir(); pir++)
  {
    const unsigned color = acc[*pir];
    if (std::binary_search(colors.begin(), colors.end(), color))
      continue;
    colors.push_back(color);
    std::sort(colors.begin(), colors.end());
  }
  // Since this cuts across all the rows we should have seen all the colors
  assert(expected_colors == colors.size());
}

class DivFunctor : public ProjectionFunctor {
public:
  using ProjectionFunctor::project;
  virtual LogicalRegion project(LogicalPartition upper_bound,
                                const DomainPoint &p,
                                const Domain &launch_domain) override
  {
    Rect<1> bounds = runtime->get_index_partition_color_space(
        upper_bound.get_index_partition());
    unsigned subregions = (bounds.hi[0] - bounds.lo[0] + 1);
    Rect<1> launch = launch_domain;
    unsigned chunk = ((launch.hi[0] - launch.lo[0] + 1) + subregions - 1) / subregions;
    Point<1> point = p;
    DomainPoint color = point[0] / chunk;
    return runtime->get_logical_subregion_by_color(upper_bound, color);
  }
  virtual bool is_exclusive(void) const override { return true; }
  virtual bool is_functional(void) const override { return true; }
  virtual unsigned get_depth(void) const override { return 0; }
};

class ModFunctor : public ProjectionFunctor {
public:
  using ProjectionFunctor::project;
  virtual LogicalRegion project(LogicalPartition upper_bound,
                                const DomainPoint &p,
                                const Domain &launch_domain) override
  {
    Rect<1> bounds = runtime->get_index_partition_color_space(
        upper_bound.get_index_partition());
    unsigned subregions = (bounds.hi[0] - bounds.lo[0] + 1);
    Point<1> point = p;
    DomainPoint color = point[0] % subregions;
    return runtime->get_logical_subregion_by_color(upper_bound, color);
  }
  virtual bool is_exclusive(void) const override { return true; }
  virtual bool is_functional(void) const override { return true; }
  virtual unsigned get_depth(void) const override { return 0; }
};

// Need a custom mapper to make sure all the instances for the 
// collective writes are different from each other
class CollectiveMapper : public DefaultMapper {
public:
  CollectiveMapper(MapperRuntime *rt, Machine machine, Processor local)
    : DefaultMapper(rt, machine, local, "collective mapper") { }
public:
  virtual LayoutConstraintID default_policy_select_layout_constraints(
                                    MapperContext ctx, Memory target_memory,
                                    const RegionRequirement &req,
                                    MappingKind mapping_kind,
                                    bool needs_field_constraint_check,
                                    bool &force_new_instances) override
  {
    const LayoutConstraintID result = 
      DefaultMapper::default_policy_select_layout_constraints(ctx, target_memory,
          req, mapping_kind, needs_field_constraint_check, force_new_instances);
    force_new_instances = true;
    return result;
  }
  virtual CachedMappingPolicy default_policy_select_task_cache_policy(
                                  MapperContext ctx, const Task &task) override
  {
    return DEFAULT_CACHE_POLICY_DISABLE;
  }
  virtual LogicalRegion default_policy_select_instance_region(
                                    MapperContext ctx, Memory target_memory,
                                    const RegionRequirement &req,
                                    const LayoutConstraintSet &constraints,
                                    bool force_new_instances,
                                    bool meets_constraints) override
  {
    return req.region;
  }
};

void registration_callback(Machine machine, Runtime *runtime,
                           const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
    runtime->replace_default_mapper(new CollectiveMapper(
          runtime->get_mapper_runtime(), machine, *it));
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  Runtime::preregister_projection_functor(DIV_PID, new DivFunctor());
  Runtime::preregister_projection_functor(MOD_PID, new ModFunctor());
  Runtime::add_registration_callback(registration_callback);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(WRITER_TASK_ID, "writer_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<writer_task>(registrar, "writer_task");
  }

  {
    TaskVariantRegistrar registrar(READER_TASK_ID, "reader_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<reader_task>(registrar, "reader_task");
  }

  return Runtime::start(argc, argv);
}
