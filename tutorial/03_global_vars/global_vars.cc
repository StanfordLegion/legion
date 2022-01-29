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

#include "legion.h"

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
};

// GLOBAL VARIABLES IN LEGION ARE ILLEGAL!
// There is no way for the runtime to know how
// that this variable exists and how to keep it
// up to date across multiple processes running
// on different nodes across the system.
int global_var = 0;

// The one exception to this is constant values.
// This is ok since the value doesn't change
// and is therefore guaranteed to be the same
// across all processes on all machines in the system.
const int global_constant = 4;

// Thread local variables are still global variables
// and are therfore also illegal. It's important to
// realize that tasks are not threads and the same
// Legion task may execute on multiple different 
// hardware threads throughout the course of its
// lifetime, so the same task may not even end up
// accessing the same instance of a thread local 
// variable during the course of its execution. 
// We provide explicit Legion runtime calls if users 
// would like to create task-local global variables 
// that have the lifetime of a single task.
__thread int thread_local_global_var = 0;

// Another thing to be careful of is that function
// pointers are another kind of global variable that
// may differ between multiple processes in the machine.
// Therefore passing function pointers around Legion
// programs is strongly discouraged.
void foo(void)
{

}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  printf("The value of global_var %d is undefined\n", global_var);

  printf("The value of global_constant %d will always be the same\n", global_constant);

  printf("The value of thread_local_global_var %d is also undefined\n", thread_local_global_var);

  printf("The function pointer to foo %p may be different on different processors\n", foo);

  // In general, Legion tasks should not allocate
  // memory directly but should instead create
  // logical regions and use them for storing data.
  // The one exception to this is that tasks can
  // use standard C and C++ memory allocation routines
  // as long as the lifetime of the allocation does
  // not exceed the lifetime of the task. Furthermore
  // any pointers referencing the allocation should
  // not be passed to sub-tasks or escape the task's
  // context. Violating either of these conditions
  // will result in a Legion application with 
  // undefined behavior.
  void *some_memory = malloc(16*sizeof(int));
  free(some_memory);
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  return Runtime::start(argc, argv);
}
