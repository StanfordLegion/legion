/* Copyright 2022 Stanford University
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
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_points = 4;
  // See how many points to run
  const InputArgs &command_args = Runtime::get_input_args();
  for (int i = 1; i < command_args.argc; i++) {
    if (!strcmp(command_args.argv[i],"-p"))
      num_points = atoi(command_args.argv[++i]);
  }
  assert(num_points > 0);
  printf("Running concurrent for %d points...\n", num_points);

  // Create some phase barriers for each number of points
  std::vector<PhaseBarrier> barriers(num_points);
  for (unsigned idx = 0; idx < num_points; idx++)
    barriers[idx] = runtime->create_phase_barrier(ctx, idx+1);

  IndexLauncher concurrent_launcher;
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
  // Execution fence to make sure we're done
  runtime->issue_execution_fence(ctx).wait();

  // Now we can delete our phase barriers
  for (unsigned idx = 0; idx < num_points; idx++)
    runtime->destroy_phase_barrier(ctx, barriers[idx]);
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

  return Runtime::start(argc, argv);
}
