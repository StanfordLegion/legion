/* Copyright 2017 Stanford University
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

#include "legion.h"

// All of the important user-level objects live 
// in the Legion namespace.
using namespace Legion;

// We use an enum to declare the IDs for user-level tasks
enum TaskID {
  TOP_LEVEL_TASK_ID,  
  HELLO_WORLD_TASK_ID,
};

#ifdef __cplusplus
extern "C" {
void hello_world_task_f(int* rank, int* n, const char* cstring[], int* rt_val);
}
#endif

void top_level_task(const Task *task, 
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    int nb_tasks = 10;
    
    {
      const InputArgs &command_args = Runtime::get_input_args();
      for (int i = 1; i < command_args.argc; i++)
      {
        if (!strcmp(command_args.argv[i],"-n"))
          nb_tasks = atoi(command_args.argv[++i]);
      }
    }
    
    for (int i = 0; i < nb_tasks; i++) {
        TaskLauncher launcher(HELLO_WORLD_TASK_ID, TaskArgument(&i,sizeof(i)));
        runtime->execute_task(ctx, launcher);
    }
}

// All single-launch tasks in Legion must have this signature with
// the extension that they can have different return values.
void hello_world_task(const Task *task, 
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
    assert(task->arglen == sizeof(int));
    int rank = *(const int*)task->args; 
      
    const char* cstring[] = { "Hello", "World"};
    int n = 2;
    int rt_val = -1;
    hello_world_task_f(&rank, &n, cstring, &rt_val);
    assert(rt_val == rank);
}

// We have a main function just like a standard C++ program.
// Once we start the runtime, it will begin running the top-level task.
int main(int argc, char **argv)
{
  // Before starting the Legion runtime, you first have to tell it
  // what the ID is for the top-level task.
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  // Before starting the Legion runtime, all possible tasks that the
  // runtime can potentially run must be registered with the runtime.
  // A task may have multiple variants (versions of the same code for
  // different processors, data layouts, etc.) Each variant is
  // registered described by a TaskVariantRegistrar object.  The
  // registrar takes a number of constraints which determine where it
  // is valid to run the task variant.  The ProcessorConstraint
  // specifies the kind of processor on which the task can be run:
  // latency optimized cores (LOC) aka CPUs or throughput optimized
  // cores (TOC) aka GPUs.  The function pointer is passed as a
  // template argument to the preregister_task_variant call.

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }
    {
        TaskVariantRegistrar registrar(HELLO_WORLD_TASK_ID, "hello_world");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<hello_world_task>(registrar, "hello_world");
    }

  // Now we're ready to start the runtime, so tell it to begin the
  // execution.  We'll never return from this call, but its return 
  // signature will return an int to satisfy the type checker.
    return Runtime::start(argc, argv);
}
