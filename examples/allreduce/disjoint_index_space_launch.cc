/* Copyright 2021 Stanford University
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
#include <unistd.h>
#include "legion.h"
#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

#define ITERATIONS 2

enum {
  POINT_WISE_LOGICAL_ANALYSIS_MAPPER_ID = 1,
};

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INTRA_IS_ORDERING_TASK_ID,
};

enum FieldIDs {
  FID_DATA,
};

// Select Task to map
class PointWiseLogicalAnalysisMapper: public DefaultMapper {
  private:
    int current_point;
    int current_point_count;
    int total_point;
    MapperEvent select_tasks_to_map_event;

  public:
    PointWiseLogicalAnalysisMapper(Machine m,
        Runtime *rt, Processor p)
      : DefaultMapper(rt->get_mapper_runtime(), m, p)
    {
      int num_iterations=ITERATIONS;
      {
        int argc = HighLevelRuntime::get_input_args().argc;
        char **argv = HighLevelRuntime::get_input_args().argv;
        for (int i = 1; i < argc; i++)
        {
          if (!strcmp(argv[i],"-i"))
            num_iterations = atoi(argv[++i]);
        }
      }
      current_point = 0;
      current_point_count = 0;
      total_point = num_iterations;
    }
  public:
    void select_tasks_to_map(const MapperContext          ctx,
                             const SelectMappingInput&    input,
                                   SelectMappingOutput&   output)
    {
      unsigned count = 0;
      for (std::list<const Task*>::const_iterator it =
            input.ready_tasks.begin(); (count < max_schedule_count) &&
            (it != input.ready_tasks.end()); it++)
      {
        if (!(*it)->is_index_space)
        {
          output.map_tasks.insert(*it);
          count ++;
        }
        else
        {
          Domain slice_domain = (*it)->get_slice_domain();
          if (slice_domain.rect_data[0] == current_point)
          {
            output.map_tasks.insert(*it);
            count ++;
            if (++current_point_count == total_point)
            {
              current_point_count = 0;
              current_point++;
            }
          }
        }
      }
      if (count == 0)
      {
        select_tasks_to_map_event = this->runtime->create_mapper_event(ctx);
        output.deferral_event = select_tasks_to_map_event;
      }
    }

    virtual void slice_task(const MapperContext ctx,
                            const Task& task,
                            const SliceTaskInput& input,
                                  SliceTaskOutput& output)
    {
      output.slices.resize(input.domain.get_volume());
      unsigned idx = 0;
      Rect<1> rect = input.domain;
      for (PointInRectIterator<1> pir(rect); pir(); pir++, idx++)
      {
        Rect<1> slice(*pir, *pir);
        output.slices[idx] = TaskSlice(slice,
            task.target_proc,
            false/*recurse*/, true/*stealable*/);
      }
    }

    static void register_my_mapper(Machine m,
                                   Runtime *rt,
                                   const std::set<Processor> &local_procs)
    {
      for (auto proc: local_procs) {
        rt->replace_default_mapper(new PointWiseLogicalAnalysisMapper(m, rt, proc), proc);
      }
    }
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_points = 4;
  int num_iterations = ITERATIONS;
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-i"))
        num_iterations = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-p"))
        num_points = atoi(command_args.argv[++i]);
    }
  }
  printf("Running with %d points and %d iterations...\n",
      num_points, num_iterations);

  Rect<1> launch_bounds(0, num_points-1);
  IndexSpaceT<1> launch_is = runtime->create_index_space(ctx, launch_bounds);

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(uint64_t), FID_DATA);
  }
  LogicalRegion lr = runtime->create_logical_region(ctx, launch_is, fs);

  const uint64_t zero = 0;
  FillLauncher fill_launcher(lr, lr, TaskArgument(&zero, sizeof(zero)));
  fill_launcher.add_field(FID_DATA);
  runtime->fill_fields(ctx, fill_launcher);

  IndexPartition ip = runtime->create_equal_partition(ctx, launch_is, launch_is);

  LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);

  ArgumentMap arg_map;

  IndexLauncher intra_is_ordering_task_launcher(INTRA_IS_ORDERING_TASK_ID, launch_is,
                                TaskArgument(NULL, 0), arg_map);
  intra_is_ordering_task_launcher.add_region_requirement(
      RegionRequirement(lp, 0/*projection ID*/,
                        LEGION_READ_WRITE, LEGION_EXCLUSIVE, lr));
  intra_is_ordering_task_launcher.add_field(0, FID_DATA);

  for (int idx = 0; idx < num_iterations; idx++)
  {
    runtime->execute_index_space(ctx, intra_is_ordering_task_launcher);
  }

  runtime->destroy_index_space(ctx, launch_is);
}

void intra_is_ordering_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  const Point<1> point = task->index_point;
  if(point.x() == 0) usleep(1000);
  printf("Executing task %lld\n", point.x());
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
    TaskVariantRegistrar registrar(INTRA_IS_ORDERING_TASK_ID, "intra_is_ordering_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<intra_is_ordering_task>(registrar, "intra_is_ordering_task");
  }

	Runtime::add_registration_callback(PointWiseLogicalAnalysisMapper::register_my_mapper);

  return Runtime::start(argc, argv);
}
