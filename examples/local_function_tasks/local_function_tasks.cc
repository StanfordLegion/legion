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
#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  FIBONACCI_TASK_ID,
  SUM_TASK_ID,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_fibonacci = 7;
  const InputArgs &command_args = Runtime::get_input_args();
  for (int i = 1; i < command_args.argc; i++) {
    // Skip any legion runtime configuration parameters
    if (command_args.argv[i][0] == '-') {
      i++;
      continue;
    }

    num_fibonacci = atoi(command_args.argv[i]);
    assert(num_fibonacci >= 0);
    break;
  }
  printf("Computing the first %d Fibonacci numbers...\n", num_fibonacci);

  std::vector<Future> fib_results;

  Future fib_start_time = runtime->get_current_time(ctx);
  std::vector<Future> fib_finish_times;
  
  for (int i = 0; i < num_fibonacci; i++)
  {
    TaskLauncher launcher(FIBONACCI_TASK_ID, TaskArgument(&i,sizeof(i)));
    // Launch these with local function tasks so that they never go anywhere
    launcher.local_function_task = true;
    fib_results.push_back(runtime->execute_task(ctx, launcher));
    fib_finish_times.push_back(runtime->get_current_time(ctx, fib_results.back()));
  }
  
  for (int i = 0; i < num_fibonacci; i++)
  {
    int result = fib_results[i].get_result<int>(); 
    double elapsed = (fib_finish_times[i].get_result<double>() -
                      fib_start_time.get_result<double>());
    printf("Fibonacci(%d) = %d (elapsed = %.2f s)\n", i, result, elapsed);
  }
  
  fib_results.clear();
}

int fibonacci_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime)
{
  assert(task->arglen == sizeof(int));
  int fib_num = *(const int*)task->args; 
  if (fib_num == 0)
    return 0;
  if (fib_num == 1)
    return 1;

  const int fib1 = fib_num-1;
  TaskLauncher t1(FIBONACCI_TASK_ID, TaskArgument(&fib1,sizeof(fib1)));
  // Launch this is a local function task
  t1.local_function_task = true;
  Future f1 = runtime->execute_task(ctx, t1);

  const int fib2 = fib_num-2;
  TaskLauncher t2(FIBONACCI_TASK_ID, TaskArgument(&fib2,sizeof(fib2)));
  // Launch this is a local function task
  t2.local_function_task = true;
  Future f2 = runtime->execute_task(ctx, t2);

  TaskLauncher sum(SUM_TASK_ID, TaskArgument(NULL, 0));
  sum.add_future(f1);
  sum.add_future(f2);
  // Launch this is a local function task
  sum.local_function_task = true;
  Future result = runtime->execute_task(ctx, sum);

  return result.get_result<int>();
}

int sum_task(const Task *task,
             const std::vector<PhysicalRegion> &regions,
             Context ctx, Runtime *runtime)
{
  assert(task->futures.size() == 2);
  Future f1 = task->futures[0];
  int r1 = f1.get_result<int>();
  Future f2 = task->futures[1];
  int r2 = f2.get_result<int>();

  return (r1 + r2);
}

// We need a custom mapper to ensure that we keep these tasks local
class LocalFunctionMapper : public DefaultMapper {
public:
  LocalFunctionMapper(Machine machine, Runtime *rt, Processor local)
    : DefaultMapper(rt->get_mapper_runtime(), machine, local) { }
public:
  virtual void select_task_options(const MapperContext ctx,
                                   const Task &task,
                                         TaskOptions &output)
  {
    if (task.task_id != TOP_LEVEL_TASK_ID)
    {
      // all local-function tasks need to run on a local processor
      assert(task.local_function);
      output.initial_proc = default_get_next_local_cpu();
    }
    else
    {
      assert(!task.local_function);
      DefaultMapper::select_task_options(ctx, task, output);
    }
  }
};

// we need a callback to register our inline mapper
void mapper_registration(Machine machine, Runtime *rt,
                         const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it =
        local_procs.begin(); it != local_procs.end(); it++)
    rt->replace_default_mapper(new LocalFunctionMapper(machine, rt, *it), *it);
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
    TaskVariantRegistrar registrar(FIBONACCI_TASK_ID, "fibonacci");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<int, fibonacci_task>(registrar, "fibonacci");
  }

  {
    TaskVariantRegistrar registrar(SUM_TASK_ID, "sum");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<int, sum_task>(registrar, "sum");
  }

  // Callback for registering the inline mapper
  Runtime::add_registration_callback(mapper_registration);

  return Runtime::start(argc, argv);
}
