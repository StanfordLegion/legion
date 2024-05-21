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

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  CONCURRENT_TASK_ID,
  DIV_CONCURRENT_TASK_ID,
  MOD_CONCURRENT_TASK_ID,
};

enum {
  DIV_CONCURRENT_FUNCTOR = 1,
  MOD_CONCURRENT_FUNCTOR = 2,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  unsigned num_points = 4;
  // See how many points to run
  const InputArgs &command_args = Runtime::get_input_args();
  for (int i = 1; i < command_args.argc; i++) {
    if (!strcmp(command_args.argv[i],"-p"))
      num_points = atoi(command_args.argv[++i]);
  }
  assert(num_points > 0);
  // Need even number of points for partial concurrent index tasks
  assert((num_points % 2) == 0);
  printf("Running concurrent for %d points...\n", num_points);

  // Create some phase barriers for each number of points
  std::vector<PhaseBarrier> barriers(num_points);
  for (unsigned idx = 0; idx < num_points; idx++)
    barriers[idx] = runtime->create_phase_barrier(ctx, idx+1);

  IndexTaskLauncher concurrent_launcher;
  concurrent_launcher.task_id = CONCURRENT_TASK_ID;
  concurrent_launcher.concurrent = true;

  // Do three iterations so we can replay the trace twice
  for (unsigned iter = 0; iter < 3; iter++)
  {
    runtime->begin_trace(ctx, 0);
    // Launch an index space for each number of points from 1 up to num_points 
    for (unsigned idx = 0; idx < num_points; idx++)
    {
      // Update the argument and the launch bounds
      concurrent_launcher.launch_domain = Rect<1>(0,idx);
      concurrent_launcher.global_arg = 
        UntypedBuffer(&barriers[idx], sizeof(barriers[idx]));
      runtime->execute_index_space(ctx, concurrent_launcher);
      // Advance the phase barrier at this level too
      barriers[idx] = runtime->advance_phase_barrier(ctx, barriers[idx]);
    }
    runtime->end_trace(ctx, 0);
  }

  // You can also create concurrent domains within subsets of points
  // in a index space task launch which means that not all points need
  // to execute concurrently with each other. You do this by providing
  // a concurrent coloring function which will color points in the 
  // index space task launch. All point tasks assigned the same color
  // will be guaranteed to execute concurrently. Point tasks assigned
  // different colors will have no such guarantee.
  std::vector<PhaseBarrier> divmod_barriers(num_points);
  // Each barrier will synchronize two points in the launch
  for (unsigned idx = 0; idx < divmod_barriers.size(); idx++)
    divmod_barriers[idx] = runtime->create_phase_barrier(ctx, 2);

  IndexTaskLauncher div_concurrent_launcher;
  div_concurrent_launcher.task_id = DIV_CONCURRENT_TASK_ID;
  div_concurrent_launcher.launch_domain = Rect<1>(0,num_points-1);
  div_concurrent_launcher.concurrent = true;
  div_concurrent_launcher.concurrent_functor = DIV_CONCURRENT_FUNCTOR;
  div_concurrent_launcher.global_arg =
    UntypedBuffer(&divmod_barriers.front(),
        (divmod_barriers.size()/2)*sizeof(PhaseBarrier));

  IndexTaskLauncher mod_concurrent_launcher;
  mod_concurrent_launcher.task_id = MOD_CONCURRENT_TASK_ID;
  mod_concurrent_launcher.launch_domain = Rect<1>(0,num_points-1);
  mod_concurrent_launcher.concurrent = true;
  mod_concurrent_launcher.concurrent_functor = MOD_CONCURRENT_FUNCTOR;
  mod_concurrent_launcher.global_arg =
    UntypedBuffer(&divmod_barriers[divmod_barriers.size()/2],
        (divmod_barriers.size()/2)*sizeof(PhaseBarrier));

  for (unsigned iter = 0; iter < 3; iter++)
  {
    runtime->begin_trace(ctx, 1);
    runtime->execute_index_space(ctx, div_concurrent_launcher);
    runtime->execute_index_space(ctx, mod_concurrent_launcher);
    runtime->end_trace(ctx, 1);
    // Advance the barriers
    for (unsigned idx = 0; idx < divmod_barriers.size(); idx++)
      divmod_barriers[idx] = runtime->advance_phase_barrier(ctx, divmod_barriers[idx]);
  }

  // Execution fence to make sure we're done
  runtime->issue_execution_fence(ctx).wait();

  // Now we can delete our phase barriers
  for (unsigned idx = 0; idx < num_points; idx++)
    runtime->destroy_phase_barrier(ctx, barriers[idx]);
  for (unsigned idx = 0; idx < divmod_barriers.size(); idx++)
    runtime->destroy_phase_barrier(ctx, divmod_barriers[idx]);
  printf("Success because we didn't hang!\n");
}

void concurrent_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  PhaseBarrier pb = *(const PhaseBarrier*)task->args; 
  // We arrive on the phase barrier
  pb.arrive();
  // We advance it
  pb = runtime->advance_phase_barrier(ctx, pb);
  // Then we wait for the others to arrive.
  // If we don't have a concurrent launch where all the point
  // tasks are running together then this can hang
  pb.wait();
}

void div_concurrent_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  const Point<1> point = task->index_point;
  PhaseBarrier pb = ((const PhaseBarrier*)task->args)[point[0]/2];
  // We arrive on the phase barrier
  pb.arrive();
  // We advance it
  pb = runtime->advance_phase_barrier(ctx, pb);
  // Then we wait for the others to arrive.
  // If we don't have a concurrent launch where all the point
  // tasks are running together then this can hang
  pb.wait();
}

void mod_concurrent_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  const Point<1> point = task->index_point;
  const Rect<1> bounds = task->index_domain;
  const unsigned mod = (bounds.hi[0] - bounds.lo[0] + 1) / 2;
  PhaseBarrier pb = ((const PhaseBarrier*)task->args)[point[0] % mod];
  // We arrive on the phase barrier
  pb.arrive();
  // We advance it
  pb = runtime->advance_phase_barrier(ctx, pb);
  // Then we wait for the others to arrive.
  // If we don't have a concurrent launch where all the point
  // tasks are running together then this can hang
  pb.wait();
}

class DivColoringFunctor : public ConcurrentColoringFunctor {
public:
  virtual Color color(const DomainPoint &point, const Domain &domain) override
  {
    const Point<1> p = point;
    return (p[0]/2);
  }
  virtual bool supports_max_color(void) override { return true; }
  virtual Color max_color(const Domain &domain) override
  {
    const Rect<1> bounds = domain;
    return (bounds.hi[0] - bounds.lo[0] + 1) / 2 - 1;
  }
};

class ModColoringFunctor : public ConcurrentColoringFunctor {
public:
  virtual Color color(const DomainPoint &point, const Domain &domain) override
  {
    const Point<1> p = point;
    const Rect<1> bounds = domain;
    const unsigned mod = (bounds.hi[0] - bounds.lo[0] + 1) / 2;
    return (p[0] % mod);
  }
};

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
    TaskVariantRegistrar registrar(CONCURRENT_TASK_ID, "concurrent_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    registrar.set_concurrent();
    Runtime::preregister_task_variant<concurrent_task>(registrar, "concurrent_task");
  }

  {
    TaskVariantRegistrar registrar(DIV_CONCURRENT_TASK_ID, "div_concurrent_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    registrar.set_concurrent();
    Runtime::preregister_task_variant<div_concurrent_task>(registrar, "div_concurrent_task");
  }

  {
    TaskVariantRegistrar registrar(MOD_CONCURRENT_TASK_ID, "mod_concurrent_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    registrar.set_concurrent();
    Runtime::preregister_task_variant<mod_concurrent_task>(registrar, "mod_concurrent_task");
  }

  Runtime::preregister_concurrent_coloring_functor(DIV_CONCURRENT_FUNCTOR,
      new DivColoringFunctor());
  Runtime::preregister_concurrent_coloring_functor(MOD_CONCURRENT_FUNCTOR,
      new ModColoringFunctor());

  return Runtime::start(argc, argv);
}
