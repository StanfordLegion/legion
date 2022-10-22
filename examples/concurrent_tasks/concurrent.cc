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
    if (command_args.argv[i][0] == '-') {
      i++;
      continue;
    }

    num_points = atoi(command_args.argv[i]);
    assert(num_points > 0);
    break;
  }
  printf("Running concurrent for %d points...\n", num_points);

  Rect<1> launch_bounds(0,num_points-1);

  // Create a phase barrier that we'll use to synchronize 
  // between all the point tasks. If we don't do this with a 
  // concurrent index space launch then we could hang
  PhaseBarrier pb = runtime->create_phase_barrier(ctx, num_points);

  ArgumentMap arg_map;
  IndexLauncher concurrent_launcher(CONCURRENT_TASK_ID,
                                    launch_bounds,
                                    TaskArgument(&pb, sizeof(pb)),
                                    arg_map);
  // Indicate that this index space task launch must be executed concurrently
  concurrent_launcher.concurrent = true;
  FutureMap fm = runtime->execute_index_space(ctx, concurrent_launcher);
  // Here we wait for all the futures to be ready
  fm.wait_all_results();

  // Now we can delete our phase barrier
  runtime->destroy_phase_barrier(ctx, pb);
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
