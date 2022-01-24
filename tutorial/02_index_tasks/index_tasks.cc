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
 * This example is a redux version of hello world 
 * which shows how launch a large array of tasks
 * using a single runtime call.  We also describe
 * the basic Legion types for arrays, domains,
 * and points and give examples of how they work.
 */

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INDEX_SPACE_TASK_ID,
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
  printf("Running hello world redux for %d points...\n", num_points);

  // To aid in describing structured data, Legion supports
  // a Rect type which is used to describe an array of
  // points.  Rects are templated on the number of 
  // dimensions that they describe.  To specify a Rect
  // a user gives two Points which specify the lower and
  // upper bounds on each dimension respectively.  Similar
  // to the Rect type, a Point type is templated on the
  // dimensions accessed.  Here we create a 1-D Rect which
  // we'll use to launch an array of tasks.  Note that the
  // bounds on Rects are inclusive.
  Rect<1> launch_bounds(0,num_points-1);

  // When we go to launch a large group of tasks in a single
  // call, we may want to pass different arguments to each
  // task.  ArgumentMaps allow the user to pass different
  // arguments to different points.  Note that ArgumentMaps
  // do not need to specify arguments for all points.  Legion
  // is intelligent about only passing arguments to the tasks
  // that have them.  Here we pass some values that we'll
  // use to illustrate how values get returned from an index
  // task launch.
  ArgumentMap arg_map;
  for (int i = 0; i < num_points; i++)
  {
    int input = i + 10;
    arg_map.set_point(i,TaskArgument(&input,sizeof(input)));
  }
  // Legion supports launching an array of tasks with a 
  // single call.  We call these index tasks as we are launching
  // an array of tasks with one task for each point in the
  // array.  Index tasks are launched similar to single
  // tasks by using an index task launcher.  IndexLauncher
  // objects take the additional arguments of an ArgumentMap,
  // a TaskArgument which is a global argument that will
  // be passed to all tasks launched, and a domain describing
  // the points to be launched.
  IndexLauncher index_launcher(INDEX_SPACE_TASK_ID,
                               launch_bounds,
                               TaskArgument(NULL, 0),
                               arg_map);
  // Index tasks are launched the same as single tasks, but
  // return a future map which will store a future for all
  // points in the index space task launch.  Application
  // tasks can either wait on the future map for all tasks
  // in the index space to finish, or it can pull out 
  // individual futures for specific points on which to wait.
  FutureMap fm = runtime->execute_index_space(ctx, index_launcher);
  // Here we wait for all the futures to be ready
  fm.wait_all_results();
  // Now we can check that the future results that came back
  // from all the points in the index task are double 
  // their input.
  bool all_passed = true;
  for (int i = 0; i < num_points; i++)
  {
    int expected = 2*(i+10);
    int received = fm.get_result<int>(i);
    if (expected != received)
    {
      printf("Check failed for point %d: %d != %d\n", i, expected, received);
      all_passed = false;
    }
  }
  if (all_passed)
    printf("All checks passed!\n");
}

int index_space_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  // The point for this task is available in the task
  // structure under the 'index_point' field.
  assert(task->index_point.get_dim() == 1); 
  printf("Hello world from task %lld!\n", task->index_point.point_data[0]);
  // Values passed through an argument map are available 
  // through the local_args and local_arglen fields.
  assert(task->local_arglen == sizeof(int));
  int input = *((const int*)task->local_args);
  return (2*input);
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
    TaskVariantRegistrar registrar(INDEX_SPACE_TASK_ID, "index_space_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<int, index_space_task>(registrar, "index_space_task");
  }

  return Runtime::start(argc, argv);
}
