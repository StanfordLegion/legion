/* Copyright 2026 Stanford University
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

/*
 * This test verifies that futures returned by void-returning tasks
 * ("empty" futures) correctly enforce control dependences when passed
 * to subsequent tasks via TaskLauncher::add_future().
 *
 * An "empty" future carries no return value but still represents
 * the completion of a task.  When passed to another task through
 * add_future(), it should create a control dependence: the receiving
 * task must not begin until the producing task completes.
 *
 * To test this we use a global std::atomic<int> counter that Legion
 * does NOT track -- there are no logical regions or other Legion data
 * structures involved.  Every task in the chain is a void leaf task
 * with no region requirements.  The ONLY mechanism ordering them is
 * the chain of empty futures.
 *
 * Each task atomically reads-and-increments the counter and asserts
 * the value it read matches its expected sequence number.  If the
 * empty futures correctly enforce ordering the counter will advance
 * 0, 1, 2, ...  If they do not, tasks may overlap and a task will
 * see the wrong counter value, triggering an assertion failure.
 *
 * The test should be run with multiple CPUs (e.g. -ll:cpu 2) so that
 * the runtime has the opportunity to execute tasks concurrently --
 * this is what makes the empty-future ordering meaningful. It does
 * not work correctly in multi-node executions obviously.
 */

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <unistd.h>
#include <atomic>
#include "legion.h"
using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  ORDERED_TASK_ID,
  SENTINEL_TASK_ID,
};

// ---------------------------------------------------------------
// Shared state that Legion does NOT track.
// The only thing preventing concurrent access is the empty-future
// chain that serialises the tasks.
// ---------------------------------------------------------------
static std::atomic<int> g_counter(0);

// Guard value for counter
static constexpr int GARBAGE = 0xdeadbeef;

/*
 * A void leaf task with no region requirements.
 * It atomically reads-and-increments the global counter and asserts
 * that the value matches its expected sequence number (passed as a
 * task argument).  If the empty-future chain is working, this task
 * will always see exactly the right counter value.
 */
void ordered_task(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime)
{
  printf("Running on processor %llx\n",
      runtime->get_executing_processor(ctx).id);
  assert(task->arglen == sizeof(int));
  int expected = *(const int*)task->args;

  int observed = g_counter.exchange(GARBAGE);
  if (observed != expected)
  {
    fprintf(stderr,
            "FAIL: task %d saw counter = %d (expected %d)\n",
            expected, observed, expected);
    assert(false);
  }
  // Sleep for 100 ms to really detect races
  usleep(100000);
  if (g_counter.exchange(expected + 1) != GARBAGE)
  {
    fprintf(stderr,
            "FAIL: task %d saw counter = %d (expected %d)\n",
            expected, observed, expected);
    assert(false);
  }
}

/*
 * A leaf task that returns a sentinel value after waiting on the
 * empty future from the last task in the chain.  Returning a value
 * lets the top-level task confirm the entire chain completed.
 */
int sentinel_task(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime)
{
  assert(task->futures.size() == 1);
  task->futures[0].get_void_result(true/*silence_warnings*/);

  // Return the final counter value so the parent can verify
  return g_counter.load();
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_tasks = 20;
  const InputArgs &command_args = Runtime::get_input_args();
  for (int i = 1; i < command_args.argc; i++)
    if (!strcmp(command_args.argv[i], "-n"))
      num_tasks = atoi(command_args.argv[++i]);
  assert(num_tasks > 0);

  printf("Testing empty future dependences with %d tasks...\n", num_tasks);

  // Reset the global counter
  g_counter.store(0);

  // Launch a chain of void leaf tasks with NO region requirements.
  // The ONLY thing that orders them is the empty-future chain.
  //
  //   ordered_task(0) --f0--> ordered_task(1) --f1--> ... --f(N-2)--> ordered_task(N-1)
  //
  Future prev_future;
  for (int i = 0; i < num_tasks; i++)
  {
    TaskLauncher launcher(ORDERED_TASK_ID, TaskArgument(&i, sizeof(i)));
    if (prev_future.exists())
      launcher.add_future(prev_future);

    // execute_task returns a Future even for void tasks.
    // This "empty" future carries no data but should still
    // represent the completion of ordered_task(i).
    prev_future = runtime->execute_task(ctx, launcher);
  }

  // Launch a sentinel task at the end of the chain that returns
  // the final counter value.
  TaskLauncher sentinel(SENTINEL_TASK_ID, TaskArgument(NULL, 0));
  sentinel.add_future(prev_future);
  Future result = runtime->execute_task(ctx, sentinel);

  int final_value = result.get_result<int>();
  printf("Final counter value: %d (expected %d)\n",
         final_value, num_tasks);
  assert(final_value == num_tasks);

  // Also verify that get_void_result works on an empty future
  // from the parent task context.
  prev_future.get_void_result(true/*silence_warnings*/);

  printf("All tests passed!\n");
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  // Void leaf task -- no return type template parameter
  {
    TaskVariantRegistrar registrar(ORDERED_TASK_ID, "ordered");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<ordered_task>(registrar, "ordered");
  }

  // Int-returning leaf task
  {
    TaskVariantRegistrar registrar(SENTINEL_TASK_ID, "sentinel");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<int, sentinel_task>(registrar, "sentinel");
  }

  return Runtime::start(argc, argv);
}
