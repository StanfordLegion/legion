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
#include <unistd.h>
#include "legion.h"
#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  POINTWISE_ANALYSABLE_FILL_ID,
  POINTWISE_ANALYSABLE_INC_ID,
  POINTWISE_ANALYSABLE_SUM_ID,
};

enum FieldIDs {
  FID_DATA,
};

#define TOTAL_POINTS 4
#define DATA_MULTIPLIER 6553600

class StreamingMapper: public DefaultMapper {
  private:
    int current_point;
    int points_executed;
    int point_types;
    bool enable_point_wise_analysis = false;
    struct InFlightTask {
      // An event that we will trigger when the task completes.
      MapperEvent event;
    };
    std::deque<InFlightTask> queue;

  public:
    StreamingMapper(Machine m,
        Runtime *rt, Processor p)
      : DefaultMapper(rt->get_mapper_runtime(), m, p)
    {
      current_point = 0;
      points_executed = 0;
      point_types = 3; // type_of_task

      int argc = Legion::HighLevelRuntime::get_input_args().argc;
      char **argv = Legion::HighLevelRuntime::get_input_args().argv;
      // Parse some command line parameters.
      for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-lg:enable_pointwise_analysis") == 0) {
          this->enable_point_wise_analysis = true;
          continue;
        }
      }
    }
  public:
    void select_tasks_to_map(const MapperContext ctx,
        const SelectMappingInput& input,
        SelectMappingOutput& output)
    {
      if (!this->enable_point_wise_analysis)
        DefaultMapper::select_tasks_to_map(ctx, input, output);
      else
      {
        MapperEvent return_event = this->runtime->create_mapper_event(ctx);

        for (std::list<const Task*>::const_iterator it =
            input.ready_tasks.begin();
            (it != input.ready_tasks.end()); it++)
        {
          output.map_tasks.insert(*it);
        }
        // If we don't schedule any tasks for mapping, the runtime needs to know
        // when to ask us again to schedule more things. Return the MapperEvent we
        // selected earlier.
        if (output.map_tasks.size() == 0)
        {
          printf("Did not get any task to select\n");
          assert(return_event.exists());
          output.deferral_event = return_event;
        }
      }
    }

    void map_task(const MapperContext ctx,
                  const Task& task,
                  const MapTaskInput& input,
                  MapTaskOutput& output)
    {
      if (this->enable_point_wise_analysis)
      {
        if (task.task_id == POINTWISE_ANALYSABLE_FILL_ID ||
            task.task_id == POINTWISE_ANALYSABLE_INC_ID  ||
            task.task_id == POINTWISE_ANALYSABLE_SUM_ID)
        {

          Processor::Kind target_kind = task.target_proc.kind();
          VariantInfo chosen;
          if (input.shard_processor.exists())
          {
            const std::pair<TaskID,Processor::Kind> key(
                task.task_id, input.shard_processor.kind());
            std::map<std::pair<TaskID,Processor::Kind>,VariantInfo>::const_iterator
              finder = preferred_variants.find(key);
            if (finder == preferred_variants.end())
            {
              chosen.variant = input.shard_variant;
              chosen.proc_kind = input.shard_processor.kind();
              chosen.tight_bound = true;
              chosen.is_inner =
                runtime->is_inner_variant(ctx, task.task_id, input.shard_variant);
              chosen.is_leaf =
                runtime->is_leaf_variant(ctx, task.task_id, input.shard_variant);
              chosen.is_replicable = true;
              preferred_variants.emplace(std::make_pair(key, chosen));
            }
            else
              chosen = finder->second;
          }
          else
            chosen = default_find_preferred_variant(task, ctx,
                            true/*needs tight bound*/, true/*cache*/, target_kind);
          output.chosen_variant = chosen.variant;
          output.task_priority = default_policy_select_task_priority(ctx, task);
          output.postmap_task = false;
          // Figure out our target processors
          if (input.shard_processor.exists())
            output.target_procs.resize(1, input.shard_processor);
          else
            default_policy_select_target_processors(ctx, task, output.target_procs);
          Processor target_proc = output.target_procs[0];

          for(size_t i = 0; i < task.regions.size(); i++) {
            Mapping::PhysicalInstance inst;
            MemoryConstraint mem_constraint =
              find_memory_constraint(ctx, task,
                  output.chosen_variant, i);
            Memory target_memory =
              default_policy_select_target_memory(ctx,
                  target_proc, task.regions[i],
                  mem_constraint);
            LayoutConstraintSet constraints;
            constraints.add_constraint(FieldConstraint(
                  task.regions[i].privilege_fields,
                  false /*!contiguous*/));
            std::vector<LogicalRegion> regions(1,
                task.regions[i].region);
            bool created;
            bool ok = runtime->find_or_create_physical_instance(ctx,
                      target_memory,
                      constraints,
                      regions,
                      inst,
                      created,
                      true/*acquire*/,
                      0/*priority*/,
                      true/*tight_region_bounds*/
                      );
            if (ok)
              output.chosen_instances[i].push_back(inst);
            else {
              output.abort_mapping = true;
              return;
            }
          }
        }
        else
        {
          DefaultMapper::map_task(ctx, task, input, output);
        }
      }
      else
      {
        DefaultMapper::map_task(ctx, task, input, output);
      }
    }

    static void register_my_mapper(Machine m,
        Runtime *rt,
        const std::set<Processor> &local_procs)
    {
      for (auto proc: local_procs) {
        rt->replace_default_mapper(new StreamingMapper(m, rt, proc), proc);
      }
    }
};

void top_level_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  int num_points = TOTAL_POINTS;
  printf("Running with ...\n");
  printf("Number of Point tasks for each IndexSpace Launch: %d\n", num_points);
  printf("Number of data points for each point task: %d\n", DATA_MULTIPLIER);
  double data_size = (num_points * DATA_MULTIPLIER * sizeof(uint64_t)) / (1024 * 1024);
  printf("Size of allocated data (Number of points * data points for each point task * sizeof(uint64_t): %lf MB\n", data_size);

  Rect<1> launch_bounds(0, num_points - 1);
  IndexSpaceT<1> launch_is = runtime->create_index_space(ctx, launch_bounds);

  Rect<1> data_bounds(0, (num_points * DATA_MULTIPLIER) - 1);
  IndexSpaceT<1> data_is = runtime->create_index_space(ctx, data_bounds);

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(uint64_t), FID_DATA);
  }
  LogicalRegion lr = runtime->create_logical_region(ctx, data_is, fs);

  const uint64_t zero = 0;

  IndexPartition ip = runtime->create_equal_partition(ctx, data_is, launch_is);

  LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);

  ArgumentMap arg_map;

  IndexLauncher point_wise_analysable_fill_launcher(POINTWISE_ANALYSABLE_FILL_ID,
      launch_is, TaskArgument(NULL, 0), arg_map);
  point_wise_analysable_fill_launcher.add_region_requirement(
      RegionRequirement(lp, 0/*projection ID*/,
        LEGION_WRITE_ONLY, LEGION_EXCLUSIVE, lr));
  point_wise_analysable_fill_launcher.add_field(0, FID_DATA);
  point_wise_analysable_fill_launcher.global_arg = TaskArgument(&zero, sizeof(zero));

  IndexLauncher point_wise_analysable_inc_launcher(POINTWISE_ANALYSABLE_INC_ID,
      launch_is, TaskArgument(NULL, 0), arg_map);
  point_wise_analysable_inc_launcher.add_region_requirement(
      RegionRequirement(lp, 0/*projection ID*/,
        LEGION_READ_WRITE, LEGION_EXCLUSIVE, lr));
  point_wise_analysable_inc_launcher.add_field(0, FID_DATA);

  IndexLauncher point_wise_analysable_sum_launcher(POINTWISE_ANALYSABLE_SUM_ID,
      launch_is, TaskArgument(NULL, 0), arg_map);
  point_wise_analysable_sum_launcher.add_region_requirement(
      RegionRequirement(lp, 0/*projection ID*/,
        LEGION_READ_ONLY | LEGION_DISCARD_OUTPUT_MASK, LEGION_EXCLUSIVE, lr));
  point_wise_analysable_sum_launcher.add_field(0, FID_DATA);

  {
    runtime->execute_index_space(ctx, point_wise_analysable_fill_launcher);
    runtime->execute_index_space(ctx, point_wise_analysable_inc_launcher);
    Future f = runtime->execute_index_space(ctx, point_wise_analysable_sum_launcher, LEGION_REDOP_SUM_UINT64);
    uint64_t result = f.get_result<uint64_t>();
    uint64_t expected = num_points * DATA_MULTIPLIER;
    assert (result == expected);
  }

  runtime->destroy_index_space(ctx, launch_is);
}

void point_wise_analysable_fill(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  const Point<1> point = task->index_point;
  printf("Fill Task %d\n", int(point.x()));
  const FieldAccessor<LEGION_WRITE_ONLY,uint64_t,1,coord_t,
        Realm::AffineAccessor<uint64_t,1,coord_t> >
          accessor(regions[0], FID_DATA);

  const uint64_t fill = *((const uint64_t*)task->args);
  Rect<1> rect = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
  {
    accessor[*pir] = fill;
  }
}

void point_wise_analysable_inc(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  const Point<1> point = task->index_point;
  printf("Inc Task %d\n", int(point.x()));
  const FieldAccessor<LEGION_READ_WRITE,uint64_t,1,coord_t,
        Realm::AffineAccessor<uint64_t,1,coord_t> >
          accessor(regions[0], FID_DATA);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
  {
    accessor[*pir] += 1;
  }
}

uint64_t point_wise_analysable_sum(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  const Point<1> point = task->index_point;
  printf("Sum Task %d\n", int(point.x()));
  const FieldAccessor<LEGION_READ_ONLY, uint64_t,1,coord_t,
        Realm::AffineAccessor<uint64_t,1,coord_t> >
          accessor(regions[0], FID_DATA);

  uint64_t sum = 0;
  Rect<1> rect = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    sum += accessor[*pir];

  return sum;
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(POINTWISE_ANALYSABLE_FILL_ID, "fill_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<point_wise_analysable_fill>(registrar, "point_wise_analysable_fill");
  }
  {
    TaskVariantRegistrar registrar(POINTWISE_ANALYSABLE_INC_ID, "inc_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<point_wise_analysable_inc>(registrar, "point_wise_analysable_inc");
  }
  {
    TaskVariantRegistrar registrar(POINTWISE_ANALYSABLE_SUM_ID, "sum_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<uint64_t, point_wise_analysable_sum>(registrar, "point_wise_analysable_sum");
  }
  Runtime::add_registration_callback(StreamingMapper::register_my_mapper);
  return Runtime::start(argc, argv);
}
