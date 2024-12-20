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
    bool disable_point_wise_analysis = false;
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
        if (strcmp(argv[i], "-lg:disable_point_wise_analysis") == 0) {
          this->disable_point_wise_analysis = true;
          continue;
        }
      }
    }
  public:
    void select_tasks_to_map(const MapperContext          ctx,
        const SelectMappingInput&    input,
        SelectMappingOutput&   output)
    {
      if (this->disable_point_wise_analysis)
        DefaultMapper::select_tasks_to_map(ctx, input, output);
      else
      {
        MapperEvent return_event;

        for (std::list<const Task*>::const_iterator it =
            input.ready_tasks.begin();
            (it != input.ready_tasks.end()); it++)
        {
          if ((*it)->task_id == POINTWISE_ANALYSABLE_FILL_ID ||
              (*it)->task_id == POINTWISE_ANALYSABLE_INC_ID  ||
              (*it)->task_id == POINTWISE_ANALYSABLE_SUM_ID)
          {
            if (this->queue.size() > 0)
              return_event = this->queue.front().event;

            Domain slice_domain = (*it)->get_slice_domain();
            if (slice_domain.rect_data[0] == current_point)
            {
              output.map_tasks.insert(*it);
              // Otherwise, we can schedule the task. Create a new event
              // and queue it up on the processor.
              this->queue.push_back({
                .event = this->runtime->create_mapper_event(ctx),
              });

              printf("Seleted a task to map: %lld ctx_idx: %lu current_point %d \n", (*it)->get_unique_id(), (*it)->get_context_index(), current_point);
            }
          }
          else
          {
            output.map_tasks.insert(*it);
          }
        }
        // If we don't schedule any tasks for mapping, the runtime needs to know
        // when to ask us again to schedule more things. Return the MapperEvent we
        // selected earlier.
        if (output.map_tasks.size() == 0)
        {
          assert(return_event.exists());
          output.deferral_event = return_event;
        }
      }
    }

    void map_task(const MapperContext ctx,
                  const Task& task,
                  const MapTaskInput& input,
                  MapTaskOutput& output) override {
      DefaultMapper::map_task(ctx, task, input, output);
      if (!this->disable_point_wise_analysis)
      {
        printf("Mapping Task: %lld ctx_idx: %lu\n", task.get_unique_id(), task.get_context_index());
        if (task.task_id == POINTWISE_ANALYSABLE_FILL_ID ||
            task.task_id == POINTWISE_ANALYSABLE_INC_ID  ||
            task.task_id == POINTWISE_ANALYSABLE_SUM_ID)
        {
          output.task_prof_requests.add_measurement<ProfilingMeasurements::OperationStatus>();
        }
      }
    }

    void report_profiling(const MapperContext ctx,
                          const Task& task,
                          const TaskProfilingInfo& input) override {
      // Only specific tasks should have profiling information.
      assert (task.task_id == POINTWISE_ANALYSABLE_FILL_ID ||
          task.task_id == POINTWISE_ANALYSABLE_INC_ID  ||
          task.task_id == POINTWISE_ANALYSABLE_SUM_ID);

      // We expect all of our tasks to complete successfully.
      auto prof = input.profiling_responses.get_measurement<ProfilingMeasurements::OperationStatus>();
      assert(prof->result == Realm::ProfilingMeasurements::OperationStatus::COMPLETED_SUCCESSFULLY);
      // Clean up after ourselves.
      delete prof;
      printf("Completed task: %lld ctx_idx: %lu\n", task.get_unique_id(), task.get_context_index());

      MapperEvent event;
      this->points_executed++;
      if (this->points_executed == point_types)
      {
        event = this->queue.front().event;
        this->queue.clear();
        points_executed = 0;
        current_point++;
      }

      // Trigger the event so that the runtime knows it's time to schedule
      // some more tasks to map.
      if (event.exists())
        this->runtime->trigger_mapper_event(ctx, event);
    }

    virtual void slice_task(const MapperContext ctx,
        const Task& task,
        const SliceTaskInput& input,
        SliceTaskOutput& output)
    {
      output.slices.resize(input.domain.get_volume());
      unsigned idx = 0;
      for (RectInDomainIterator<1> itr(input.domain); itr(); itr++)
      {
        for (PointInRectIterator<1> pir(*itr); pir(); pir++, idx++)
        {
          Rect<1> slice(*pir, *pir);
          output.slices[idx] = TaskSlice(slice,
              task.target_proc,
              false/*recurse*/, true/*stealable*/);
        }
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
  // -ll:fsize 512 - in MB
  // -ll:csize 512 - in MB
}

void point_wise_analysable_fill(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

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
