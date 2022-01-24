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
  GENERATE_RANDOM_TASK_ID,
  CONDITION_TEST_TASK_ID,
  CONDITION_TRUE_TASK_ID,
  CONDITION_FALSE_TASK_ID,
};

#define TEST_VALUE    0.5

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_rounds = 10;
  // The command line arguments to a Legion application are
  // available through the runtime 'get_input_args' call.  We'll 
  // use this to get the number of Fibonacci numbers to compute.
  const InputArgs &command_args = Runtime::get_input_args();
  for (int i = 1; i < command_args.argc; i++) {
    // Skip any legion runtime configuration parameters
    if (command_args.argv[i][0] == '-') {
      i++;
      continue;
    }

    num_rounds = atoi(command_args.argv[i]);
    assert(num_rounds >= 0);
    break;
  }
  printf("Running %d rounds of tests...\n", num_rounds);

  for (int i = 0; i < num_rounds; i++)
  {
    TaskLauncher generate(GENERATE_RANDOM_TASK_ID, TaskArgument());
    Future value = runtime->execute_task(ctx, generate);

    TaskLauncher test(CONDITION_TEST_TASK_ID, TaskArgument());
    test.add_future(value);
    Future condition = runtime->execute_task(ctx, test);

    Predicate true_pred = runtime->create_predicate(ctx, condition);
    Predicate false_pred = runtime->predicate_not(ctx, true_pred); 

    TaskLauncher true_task(CONDITION_TRUE_TASK_ID, 
                  TaskArgument(&i, sizeof(i)), true_pred);
    runtime->execute_task(ctx, true_task);

    TaskLauncher false_task(CONDITION_FALSE_TASK_ID,
                  TaskArgument(&i, sizeof(i)), false_pred);
    runtime->execute_task(ctx, false_task);
  }
}

double generate_random_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
  return drand48(); 
}

bool test_condition_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  assert(task->futures.size() == 1);
  double value = task->futures[0].get_result<double>();
  return (value < TEST_VALUE);
}

void true_condition_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  printf("Round %d had value less than %.8g\n", *(const int*)task->args, TEST_VALUE); 
}

void false_condition_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  printf("Round %d had value greather than or equal %.8g\n", *(const int*)task->args, TEST_VALUE);
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
    TaskVariantRegistrar registrar(GENERATE_RANDOM_TASK_ID, "generate_random");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<double, generate_random_task>(registrar, "generate_random");
  }

  {
    TaskVariantRegistrar registrar(CONDITION_TEST_TASK_ID, "test_condition");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<bool, test_condition_task>(registrar, "test_condition");
  }

  {
    TaskVariantRegistrar registrar(CONDITION_TRUE_TASK_ID, "true_condition");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<true_condition_task>(registrar, "true_condition");
  }

  {
    TaskVariantRegistrar registrar(CONDITION_FALSE_TASK_ID, "false_condition");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<false_condition_task>(registrar, "false_condition");
  }

  return Runtime::start(argc, argv);
}
