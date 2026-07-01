/* Copyright 2026 Stanford University
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
  EXTRACT_TASK_ID,
  TRANSFORM_TASK_ID,
  LOAD_TASK_ID,
};

enum FieldIDs {
  FID_DATA,
};

struct ConfigArgs {
  int tasks_per_processor = 4;
  int points_per_task = 6553600;
};

class StreamingMapper: public DefaultMapper {
  private:
    MapperEvent deferral_event;
    std::optional<DomainPoint> current_point;
    std::optional<uint64_t> current_index;
    bool enable_point_wise_analysis = false;

  public:
    StreamingMapper(Machine m,
        Runtime *rt, Processor p)
      : DefaultMapper(rt->get_mapper_runtime(), m, p)
    {
      int argc = Legion::Runtime::get_input_args().argc;
      char **argv = Legion::Runtime::get_input_args().argv;
      // Parse some command line parameters.
      for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-lg:enable_pointwise_analysis") == 0) {
          this->enable_point_wise_analysis = true;
          continue;
        }
      }
    }
  public:
    virtual void select_tasks_to_map(MapperContext ctx,
        const SelectMappingInput& input,
        SelectMappingOutput& output) override
    {
      assert(!input.ready_tasks.empty());
      if (this->enable_point_wise_analysis)
      {
        if (!current_point)
        {
          const Task* task = input.ready_tasks.front();
          if (task->task_id == TOP_LEVEL_TASK_ID)
          {
            output.map_tasks.insert(task);
            return;
          }
          current_point = task->index_point;
          current_index = task->get_context_index();
          // Initialize the current point for this processor
          for (std::list<const Task*>::const_iterator it =
                input.ready_tasks.begin(); it !=
                input.ready_tasks.end(); it++)
          {
            if ((*it)->index_point < *current_point)
              current_point = (*it)->index_point;
            if ((*it)->get_context_index() < *current_index)
              current_index = (*it)->get_context_index();
          }
        }
        // See if we can find the next task to map
        const Task* next = nullptr;
        for (std::list<const Task*>::const_iterator it =
              input.ready_tasks.begin(); it !=
              input.ready_tasks.end(); it++)
        {
          if ((*it)->index_point != *current_point)
            continue;
          if ((*it)->get_context_index() != *current_index)
            continue;
          next = *it;
          break;
        }
        if (next == nullptr)
        {
          if (!deferral_event.exists())
            deferral_event = runtime->create_mapper_event(ctx);
          output.deferral_event = deferral_event;
        }
        else
        {
          output.map_tasks.insert(next);
          if (++(*current_index) == 4)
          {
            // Reset for the next chain
            current_index.reset();
            current_point.reset();
          }
          if (deferral_event.exists())
          {
            MapperEvent to_trigger;
            std::swap(to_trigger, deferral_event);
            runtime->trigger_mapper_event(ctx, to_trigger);
          }
        }
      }
      else
        DefaultMapper::select_tasks_to_map(ctx, input, output);
    }

    virtual void slice_task(MapperContext ctx,
                            const Task& task,
                            const SliceTaskInput& input,
                                  SliceTaskOutput& output) override
    {
      // Just one slice on the local processor
      if (input.domain_is.exists())
        output.slices.emplace_back(TaskSlice(input.domain_is,
              local_proc, false, false));
      else
        output.slices.emplace_back(TaskSlice(input.domain,
              local_proc, false, false));
    }

    virtual void map_task(MapperContext ctx,
                  const Task& task,
                  const MapTaskInput& input,
                  MapTaskOutput& output) override
    {
      if (this->enable_point_wise_analysis)
      {
        if (task.task_id == TRANSFORM_TASK_ID  ||
            task.task_id == LOAD_TASK_ID)
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
                  false /*!contiguous*/, false/*inorder*/));
            std::vector<LogicalRegion> regions(1,
                task.regions[i].region);
            bool ok = runtime->find_physical_instance(ctx,
                      target_memory,
                      constraints,
                      regions,
                      inst,
                      true/*acquire*/,
                      true/*tight_region_bounds*/
                      );
            if (ok)
              output.chosen_instances[i].push_back(inst);
            else
              std::abort();
          }
        }
        else
          DefaultMapper::map_task(ctx, task, input, output);
      }
      else
      {
        DefaultMapper::map_task(ctx, task, input, output);
      }
      if (task.task_id == EXTRACT_TASK_ID)
      {
        const Memory target =
          default_policy_select_output_target(ctx, task.target_proc);
        assert(target.exists());
        output.leaf_pool_bounds[target] = PoolBounds(LEGION_STRICT_UNBOUNDED_POOL);
      }
    }

    static void register_my_mapper(Machine m,
        Runtime *rt,
        const std::set<Processor> &local_procs)
    {
      for (auto proc: local_procs)
        rt->replace_default_mapper(new StreamingMapper(m, rt, proc), proc);
    }
};

void top_level_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  Future total_cpus_f = runtime->select_tunable_value(
      ctx, DefaultMapper::DEFAULT_TUNABLE_GLOBAL_CPUS);

  const InputArgs &command_args = Runtime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;

  ConfigArgs args;
  for (int i = 0; i < argc; i++)
    if (strcmp(argv[i], "-tasks") == 0)
      args.tasks_per_processor = std::atoi(argv[++i]);
    else if (strcmp(argv[i], "-points") == 0)
      args.points_per_task = std::atoi(argv[++i]);
  assert(args.tasks_per_processor > 0);
  assert(args.points_per_task > 0);

  const size_t total_procs = total_cpus_f.get<size_t>(); 
  const size_t total_point_tasks = total_procs * args.tasks_per_processor;

  LEGION_PRINT_ONCE(runtime, ctx, stdout, "Tasks per processor: %d\n", args.tasks_per_processor);
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "Points per task: %d\n", args.points_per_task);
  const uint64_t total_points = total_point_tasks * args.points_per_task;
  const double data_size = (total_points * sizeof(uint64_t)) / (1024 * 1024);
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "Total data size: %.2lf MB\n", data_size);

  const Rect<1> launch_bounds(0, total_point_tasks - 1);
  IndexSpaceT<1> launch_is = runtime->create_index_space(ctx, launch_bounds);

  const FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(uint64_t), FID_DATA);
  }

  const TaskArgument task_args(&args, sizeof(args));
  std::vector<OutputRequirement> output_requirements;
  OutputRequirement& out_req = output_requirements.emplace_back(
      OutputRequirement(fs, {FID_DATA}, 1/*dimension*/, true/*global indexing*/));

  // Launch the extract task
  IndexTaskLauncher extract_launcher(EXTRACT_TASK_ID, launch_is, task_args);
  runtime->execute_index_space(ctx, extract_launcher, &output_requirements);

  // We can get back the name of the logical partition created here
  const LogicalRegion parent = out_req.parent;
  const LogicalPartition lp = out_req.partition; 

  IndexTaskLauncher transform_launcher(TRANSFORM_TASK_ID, launch_is);
  RegionRequirement& transform_req =
    transform_launcher.add_region_requirement(
        RegionRequirement(lp, 0/*identity projection*/, LEGION_READ_WRITE, LEGION_EXCLUSIVE, parent));
  transform_req.add_field(FID_DATA);
  runtime->execute_index_space(ctx, transform_launcher);

  IndexTaskLauncher load_launcher(LOAD_TASK_ID, launch_is);
  RegionRequirement& load_req =
    load_launcher.add_region_requirement(
        RegionRequirement(lp, 0/*identity projection*/, 
          LEGION_READ_ONLY | LEGION_DISCARD_OUTPUT_MASK, LEGION_EXCLUSIVE, parent));
  load_req.add_field(FID_DATA);
  Future f = runtime->execute_index_space(ctx, load_launcher, LEGION_REDOP_SUM_UINT64);

  runtime->destroy_logical_region(ctx, parent);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, parent.get_index_space());

  const uint64_t result = f.get<uint64_t>();
  assert(result == total_points);
}

void extract_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  assert(task->arglen == sizeof(ConfigArgs));
  const ConfigArgs* args = static_cast<const ConfigArgs*>(task->args);

  const Point<1> point = task->index_point;
  printf("Extract Task %d\n", int(point.x()));

  std::vector<OutputRegion> outputs;
  runtime->get_output_regions(ctx, outputs);
  assert(outputs.size() == 1);
  OutputRegion &output = outputs.back();

  // Extract the data from the filesystem. We're not actually
  // going to load data from the file system here but we're
  // using an output region to showcase that you can handle
  // variable sized data coming from an external source.
  // We return the same size output for each task, but that
  // is not required for output regions.

  uint64_t initial_value = 0;
  output.create_buffer<uint64_t, 1>(
      Point<1>(args->points_per_task), FID_DATA, &initial_value, true/*return buffer*/);
}

void transform_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  // Transform the data. We're just going to increment it. 

  const Point<1> point = task->index_point;
  printf("Transform Task %d\n", int(point.x()));
  const FieldAccessor<LEGION_READ_WRITE,uint64_t,1,coord_t,
        Realm::AffineAccessor<uint64_t,1,coord_t> >
          accessor(regions[0], FID_DATA);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    accessor[*pir] += 1;
}

uint64_t load_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  // Load the data into whatever form you want. In this
  // case we're just going to reduce it down to a value

  const Point<1> point = task->index_point;
  printf("Load Task %d\n", int(point.x()));
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
    TaskVariantRegistrar registrar(EXTRACT_TASK_ID, "extract_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<extract_task>(registrar, "extract_task");
  }
  {
    TaskVariantRegistrar registrar(TRANSFORM_TASK_ID, "transform_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<transform_task>(registrar, "transform_task");
  }
  {
    TaskVariantRegistrar registrar(LOAD_TASK_ID, "load_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<uint64_t, load_task>(registrar, "load_task");
  }
  Runtime::add_registration_callback(StreamingMapper::register_my_mapper);
  return Runtime::start(argc, argv);
}
