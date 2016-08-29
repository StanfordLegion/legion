/* Copyright 2016 Stanford University
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

#include <cstdlib>
#include <cstdio>

#include "legion.h"

#include "default_mapper.h"

// All of the important user-level objects live 
// in the Legion namespace.
using namespace Legion;
using namespace Legion::Mapping;
// We use an enum to declare the IDs for user-level tasks
enum TaskID {
  HELLO_WORLD_ID,
};

class BroadcastTest: public DefaultMapper{
public:
	BroadcastTest(Machine machine, 
      Runtime *rt, Processor local);
public:
  virtual void select_task_options(const MapperContext    ctx,
                                       const Task&            task,
                                             TaskOptions&     output);
   virtual void handle_message(const MapperContext           ctx,
                                  const MapperMessage&          message);
 std::set<Processor> all_procs;
};

void mapper_registration(Machine machine, Runtime *rt,
                          const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(
        new BroadcastTest(machine, rt, *it), *it);
  }
}

BroadcastTest::BroadcastTest(Machine m, Runtime *rt, Processor p):DefaultMapper(rt->get_mapper_runtime(), m, p)
{
  machine.get_all_processors(all_procs);
}

void BroadcastTest::select_task_options(const MapperContext ctx,
                                            const Task& task,
                                                  TaskOptions& output)
{
int m=123;
void *message = &m;
if (all_procs.begin()->id + 1 == local_proc.id) mapper_runtime->broadcast(ctx, message, sizeof(int));

//std::set<Processor>::iterator it = all_procs.begin(); std::advance(it, 7);
//if (all_procs.begin()->id+1 == local_proc.id) mapper_runtime->send_message(ctx,*it, message, sizeof(int));
DefaultMapper::select_task_options(ctx, task, output);
}

void BroadcastTest::handle_message(const MapperContext           ctx,
                const MapperMessage&          message){
					std::cout<<"handle message\t Node: "<<node_id<<"\tProcessor: "<<local_proc.id<<"\tSender: "<<message.sender.id<<"\tMessage "<<*(int *)message.message<<"\n";
}


// All single-launch tasks in Legion must have this signature with
// the extension that they can have different return values.
void hello_world_task(const Task *task, 
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
  // A task runs just like a normal C++ function.
  printf("Hello World!\n");
}

// We have a main function just like a standard C++ program.
// Once we start the runtime, it will begin running the top-level task.
int main(int argc, char **argv)
{
  // Before starting the Legion runtime, you first have to tell it
  // what the ID is for the top-level task.
  Runtime::set_top_level_task_id(HELLO_WORLD_ID);
  // Before starting the Legion runtime, all possible tasks that the
  // runtime can potentially run must be registered with the runtime.
  // The function pointer is passed as a template argument.  The second
  // argument specifies the kind of processor on which the task can be
  // run: latency optimized cores (LOC) aka CPUs or throughput optimized
  // cores (TOC) aka GPUs.  The last two arguments specify whether the
  // task can be run as a single task or an index space task (covered
  // in more detail in later examples).  The top-level task must always
  // be able to be run as a single task.
  Runtime::register_legion_task<hello_world_task>(HELLO_WORLD_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);

  // Now we're ready to start the runtime, so tell it to begin the
  // execution.  We'll never return from this call, but its return 
  // signature will return an int to satisfy the type checker.
	  Runtime::set_registration_callback(mapper_registration);

  return Runtime::start(argc, argv);
}
