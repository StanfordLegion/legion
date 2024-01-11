/* Copyright 2024 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "legion.h"
#include "default_mapper.h"
#include "logging_wrapper.h"

using namespace Legion;
using namespace Legion::Mapping;

enum TaskIds
{
  TOP_TASK_ID,
  READER_TASK_ID,
};

class ReplicationMapper : public DefaultMapper
{
public:
  ReplicationMapper(Machine, Runtime *, Processor);

  virtual void select_task_options(const MapperContext ctx,
                                   const Task &task,
                                   TaskOptions &output) override;

  virtual void replicate_task(MapperContext ctx,
                              const Task& task,
                              const ReplicateTaskInput& input,
                                    ReplicateTaskOutput& output) override;

  virtual LayoutConstraintID default_policy_select_layout_constraints(
                                    MapperContext ctx, Memory target_memory,
                                    const RegionRequirement &req,
                                    MappingKind mapping_kind,
                                    bool needs_field_constraint_check,
                                    bool &force_new_instances) override;
protected:
  std::vector<Processor> all_cpus;
};

ReplicationMapper::ReplicationMapper(Machine m, Runtime *rt, Processor p)
    : DefaultMapper(rt->get_mapper_runtime(), m, p)
{
  Machine::ProcessorQuery proc_it(machine);
  proc_it.only_kind(Processor::LOC_PROC);
  for (auto it = proc_it.begin(); it != proc_it.end(); it++)
    all_cpus.push_back(*it);
  if (all_cpus.size() <= 1)
    fprintf(stderr,"Must have multiple CPUs for this example, run with -ll:cpu 2 or more\n");
  assert(all_cpus.size() > 1);
}

void ReplicationMapper::select_task_options(const MapperContext ctx, const Task &task,
                                      TaskOptions &output)
{
  DefaultMapper::select_task_options(ctx, task, output);
  output.replicate = true;
}

void ReplicationMapper::replicate_task(MapperContext ctx,
                                 const Task& task,
                                 const ReplicateTaskInput& input,
                                       ReplicateTaskOutput& output)
{
  const Processor::Kind target_kind = task.target_proc.kind();
  // Get the variant that we are going to use to map this task
  const VariantInfo chosen = default_find_preferred_variant(
      task, ctx, true /*needs tight bound*/, true /*cache*/, target_kind);

  assert(chosen.is_replicable);

  output.chosen_variant = chosen.variant;

  // Replicate every task across all the processors
  output.target_processors = all_cpus; 
}

LayoutConstraintID ReplicationMapper::default_policy_select_layout_constraints(
                                    MapperContext ctx, Memory target_memory,
                                    const RegionRequirement &req,
                                    MappingKind mapping_kind,
                                    bool needs_field_constraint_check,
                                    bool &force_new_instances)
{
  LayoutConstraintID result = 
    DefaultMapper::default_policy_select_layout_constraints(
        ctx, target_memory, req, mapping_kind, 
        needs_field_constraint_check, force_new_instances);
  force_new_instances = true;
  return result;
}

void mapper_registration(Machine machine, Runtime *rt,
                         const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
       it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(
        new LoggingWrapper(new ReplicationMapper(machine, rt, *it)), *it);
  }
}

void reader(const Task *task, const std::vector<PhysicalRegion> &regions,
            Context ctx, Runtime *rt)
{
  printf("Running reader for shard %d\n", task->get_shard_id());
  return;
}

void top(const Task *task, const std::vector<PhysicalRegion> &regions,
         Context ctx, Runtime *rt)
{
  printf("Running top for shard %d\n", task->get_shard_id());
  Rect<1> domain(0, 1);
  auto ispace = rt->create_index_space(ctx, domain);
  auto fspace = rt->create_field_space(ctx);
  {
    auto fal = rt->create_field_allocator(ctx, fspace);
    fal.allocate_field(sizeof(int), 0);
  }
  auto lr = rt->create_logical_region(ctx, ispace, fspace);

  rt->fill_field<int>(ctx, lr, lr, 0, 0);

  TaskLauncher rl(READER_TASK_ID, TaskArgument());
  rl.add_region_requirement(RegionRequirement(lr, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr));
  rl.add_field(0, 0);
  rt->execute_task(ctx, rl);
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_TASK_ID, "top");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable(true);
    Runtime::preregister_task_variant<top>(registrar, "top");
  }
  {
    TaskVariantRegistrar registrar(READER_TASK_ID, "reader");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable(true);
    Runtime::preregister_task_variant<reader>(registrar, "reader");
  }

  Runtime::add_registration_callback(mapper_registration);
  return Runtime::start(argc, argv);
}
