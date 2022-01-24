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

/*
 * To illustrate task launches and futures in Legion
 * we implement a program to compute the first N
 * Fibonacci numbers.  While we note that this is not
 * the fastest way to compute Fibonacci numbers, it
 * is designed to showcase the functional nature of
 * Legion tasks and futures.
 */

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

    num_fibonacci = atoi(command_args.argv[i]);
    assert(num_fibonacci >= 0);
    break;
  }
  printf("Computing the first %d Fibonacci numbers...\n", num_fibonacci);

  // This is a vector which we'll use to store the future
  // results of all the tasks that we launch.  The goal here
  // is to launch all of our tasks up front to get them in
  // flight before waiting on a future value.  This exposes
  // as many tasks as possible to the Legion runtime to
  // maximize performance.
  std::vector<Future> fib_results;

  // We'll also time how long these tasks take to run.  Since
  // tasks in Legion execute in a deferred fashion, we ask the
  // runtime for the "current_time" for our context, which doesn't
  // actually record it until some other Future becomes ready.
  // We are given a Future that will hold the timer value once it
  // has been recorded so that we can keep issuing more tasks.
  Future fib_start_time = runtime->get_current_time(ctx);
  std::vector<Future> fib_finish_times;
  
  // Compute the first num_fibonacci numbers
  for (int i = 0; i < num_fibonacci; i++)
  {
    // All Legion tasks are spawned from a launcher object.  A
    // 'TaskLauncher' is a struct used for specifying the arguments
    // necessary for launching a task.  Launchers contain many
    // fields which we will explore throughout the examples.  Here
    // we look at the first two arguments: the ID of the kind of
    // task to launch and a 'TaskArgument'.  The ID of the task
    // must correspond to one of the IDs registered with the Legion
    // runtime before the application began.  A 'TaskArgument' points
    // to a buffer and specifies the size in bytes to copy by value 
    // from the buffer.  It is important to note that this buffer is 
    // not actually copied until 'execute_task' is called.  The buffer 
    // should remain live until the launcher goes out of scope.
    TaskLauncher launcher(FIBONACCI_TASK_ID, TaskArgument(&i,sizeof(i)));
    // To launch a task, a TaskLauncher object is passed to the runtime
    // along with the context.  Legion tasks are asynchronous which means
    // that this call returns immediately and returns a future value which
    // we store in our vector of future results.  Note that launchers can 
    // be reused to launch as many tasks as desired, and can be modified 
    // immediately after the 'execute_task' call returns.
    fib_results.push_back(runtime->execute_task(ctx, launcher));
    // We can use the future for the task's result to make sure we record
    // the execution time only once that task has finished
    fib_finish_times.push_back(runtime->get_current_time(ctx, fib_results.back()));
  }
  
  // Print out our results
  for (int i = 0; i < num_fibonacci; i++)
  {
    // One way to use a future is to explicitly ask for its value using
    // the 'get_result' method.  This is a blocking call which will cause
    // this task (the top-level task) to pause until the sub-task which
    // is generating the future returns.  Note that waiting on a future
    // that is not ready blocks this task, but does not block the processor
    // on which the task is running.  If additional tasks have been mapped 
    // onto this processor and they are ready to execute, then they will 
    // begin running as soon as the call to 'get_result' is made.
    //
    // The 'get_result' method is templated on the type of the return 
    // value which tells the Legion runtime how to interpret the bits 
    // being returned.  In most cases the bits are cast, however, if 
    // the type passed in the template has the methods 'legion_buffer_size', 
    // 'legion_serialize', and 'legion_deserialize' defined, then Legion 
    // automatically supports deep copies of more complex types (see the 
    // ColoringSerializer class in legion.h for an example).  While this 
    // way of using features requires blocking this task, we examine a 
    // non-blocking way of using future below.
    int result = fib_results[i].get_result<int>(); 
    // We used 'get_current_time', which returns a double containing the
    // number of seconds since the runtime started up.
    double elapsed = (fib_finish_times[i].get_result<double>() -
                      fib_start_time.get_result<double>());
    printf("Fibonacci(%d) = %d (elapsed = %.2f s)\n", i, result, elapsed);
  }
  
  // Implementation detail for those who are interested: since futures
  // are shared between the runtime and the application, we reference
  // count them and automatically delete their resources when there
  // are no longer any references to them.  The 'Future' type is
  // actually a light-weight handle which simply contains a pointer
  // to the actual future implementation, so copying future values
  // around is inexpensive.  Here we explicitly clear the vector
  // which invokes the Future destructor and removes the references.
  // This would have happened anyway when the vector went out of
  // scope, but we have the statement so we could put this comment here.
  fib_results.clear();
}

int fibonacci_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime)
{
  // The 'TaskArgument' value passed to a task and its size
  // in bytes is available in the 'args' and 'arglen' fields
  // on the 'Task' object.
  //
  // Since there is no type checking when writing to
  // the runtime API (a benefit provided by our Legion compiler)
  // we encourage programmers to check that they are getting
  // what they expect in their values.
  assert(task->arglen == sizeof(int));
  int fib_num = *(const int*)task->args; 
  // Fibonacci base cases
  // Note that tasks return values the same as C functions.
  // If a task is running remotely from its parent task then
  // Legion automatically packages up the result and returns
  // it to the origin location.
  if (fib_num == 0)
    return 0;
  if (fib_num == 1)
    return 1;

  // Launch fib-1
  const int fib1 = fib_num-1;
  TaskLauncher t1(FIBONACCI_TASK_ID, TaskArgument(&fib1,sizeof(fib1)));
  Future f1 = runtime->execute_task(ctx, t1);

  // Launch fib-2
  const int fib2 = fib_num-2;
  TaskLauncher t2(FIBONACCI_TASK_ID, TaskArgument(&fib2,sizeof(fib2)));
  Future f2 = runtime->execute_task(ctx, t2);

  // Here will illustrate a non-blocking way of using a future. 
  // Rather than waiting for the values and passing the results
  // directly to the summation task, we instead pass the futures
  // through the TaskLauncher object.  Legion then will 
  // ensure that the sum task does not begin until both futures
  // are ready and that the future values are available wherever
  // the sum task is run (even if it is run remotely).  Futures
  // should NEVER be passed through a TaskArgument.
  TaskLauncher sum(SUM_TASK_ID, TaskArgument(NULL, 0));
  sum.add_future(f1);
  sum.add_future(f2);
  Future result = runtime->execute_task(ctx, sum);

  // Our API does not permit returning Futures as the result of 
  // a task.  Any attempt to do so will result in a failed static 
  // assertion at compile-time.  In general, waiting for one or 
  // more futures at the end of a task is inexpensive since we 
  // have already exposed the available sub-tasks for execution 
  // to the Legion runtime so we can extract as much task-level
  // parallelism as possible from the application.
  return result.get_result<int>();
}

int sum_task(const Task *task,
             const std::vector<PhysicalRegion> &regions,
             Context ctx, Runtime *runtime)
{
  assert(task->futures.size() == 2);
  // Note that even though it looks like we are performing
  // blocking calls to get these future results, the
  // Legion runtime is smart enough to not run this task
  // until all the future values passed through the
  // task launcher have completed.
  Future f1 = task->futures[0];
  int r1 = f1.get_result<int>();
  Future f2 = task->futures[1];
  int r2 = f2.get_result<int>();

  return (r1 + r2);
}
              
int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  // Note that tasks which return values must pass the type of
  // the return argument as the first template parameter.

  {
    TaskVariantRegistrar registrar(FIBONACCI_TASK_ID, "fibonacci");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<int, fibonacci_task>(registrar, "fibonacci");
  }

  // The sum-task has a very special property which is that it is
  // guaranteed never to make any runtime calls.  We call these
  // kinds of tasks "leaf" tasks and tell the runtime system
  // about them using the 'TaskConfigOptions' struct.  Being
  // a leaf task allows the runtime to perform significant
  // optimizations that minimize the overhead of leaf task
  // execution.

  {
    TaskVariantRegistrar registrar(SUM_TASK_ID, "sum");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<int, sum_task>(registrar, "sum");
  }

  return Runtime::start(argc, argv);
}
