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


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"
#include "mappers/default_mapper.h"

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  READ_FIELD_TASK_ID,
  REDUCE_FIELD_TASK_ID,
};

enum FieldIDs {
  FID_DATA,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_elements = 1024;
  int num_points = 4;
  int num_iterations = 4;
  {
      const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-i"))
        num_iterations = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-p"))
        num_points = atoi(command_args.argv[++i]);
    }
  }
  printf("Running with %d elements and %d points and %d iterations...\n",
      num_elements, num_points, num_iterations);

  Rect<1> elem_rect(0,num_elements-1);
  IndexSpaceT<1> is = runtime->create_index_space(ctx, elem_rect);
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(uint64_t), FID_DATA);
  }
  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);
  
  Rect<1> launch_bounds(0, num_points-1);
  IndexSpaceT<1> launch_is = runtime->create_index_space(ctx, launch_bounds);

  const uint64_t zero = 0;
  FillLauncher fill_launcher(lr, lr, TaskArgument(&zero, sizeof(zero)));
  fill_launcher.add_field(FID_DATA);
  runtime->fill_fields(ctx, fill_launcher);

  ArgumentMap arg_map;

  IndexLauncher reduce_launcher(REDUCE_FIELD_TASK_ID, launch_is,
                                TaskArgument(NULL, 0), arg_map);
  reduce_launcher.add_region_requirement(
      RegionRequirement(lr, 0/*projection ID*/,
                        LEGION_REDOP_SUM_UINT64, LEGION_EXCLUSIVE, lr));
  reduce_launcher.add_field(0, FID_DATA);

  IndexLauncher read_launcher(READ_FIELD_TASK_ID, launch_is,
                              TaskArgument(NULL, 0), arg_map);
  read_launcher.add_region_requirement(
      RegionRequirement(lr, 0/*projection ID*/,
                        LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr));
  read_launcher.add_field(0, FID_DATA);

  for (int idx = 1; idx <= num_iterations; idx++)
  {
    runtime->execute_index_space(ctx, reduce_launcher);
    read_launcher.global_arg = TaskArgument(&idx, sizeof(idx));
    runtime->execute_index_space(ctx, read_launcher);
  }

  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
  runtime->destroy_index_space(ctx, launch_is);
}

void reduce_field_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  const Point<1> point = task->index_point;

  const ReductionAccessor<SumReduction<uint64_t>,false/*exclusive*/,1,
                  coord_t, Realm::AffineAccessor<uint64_t,1,coord_t> >
    accessor(regions[0], FID_DATA, LEGION_REDOP_SUM_UINT64);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    accessor.reduce(*pir, point[0]);
}

void read_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->arglen == sizeof(int));
  const int iteration = *((const int*)task->args);

  const FieldAccessor<LEGION_READ_ONLY,uint64_t,1,coord_t,
                 Realm::AffineAccessor<uint64_t,1,coord_t> >
    accessor(regions[0], FID_DATA);

  Rect<1> bounds = task->index_domain;
  uint64_t expected = 0;
  for (PointInRectIterator<1> pir(bounds); pir(); pir++)
    expected += (*pir)[0];
  expected *= iteration;

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    assert(accessor[*pir] == expected);
}

class CollectiveInstanceMapper : public Mapping::DefaultMapper {
public:
  CollectiveInstanceMapper(Mapping::MapperRuntime *rt, Machine machine, 
                           Processor local, const char *mapper_name)
    : DefaultMapper(rt, machine, local, mapper_name) { }

  virtual void select_task_options(const Mapping::MapperContext ctx,
                                   const Task& task,
                                   Mapping::Mapper::TaskOptions& options)
  {
    // Do the base mapper call and look for collective instances
    DefaultMapper::select_task_options(ctx, task, options);
    if ((task.task_id == REDUCE_FIELD_TASK_ID) ||
        (task.task_id == READ_FIELD_TASK_ID))
      options.check_collective_regions.insert(0);
  }
};

void update_mappers(Machine machine, Runtime *runtime,
                    const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it =
        local_procs.begin(); it != local_procs.end(); it++)
  {
    CollectiveInstanceMapper *mapper =
      new CollectiveInstanceMapper(runtime->get_mapper_runtime(),
                                   machine, *it, "collective_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(REDUCE_FIELD_TASK_ID, "reduce_field");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<reduce_field_task>(registrar, "reduce_field");
  }

  {
    TaskVariantRegistrar registrar(READ_FIELD_TASK_ID, "read_field");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<read_field_task>(registrar, "read_field");
  }

  Runtime::add_registration_callback(update_mappers);

  return Runtime::start(argc, argv);
}
