/* Copyright 2018 Stanford University
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

#if __cplusplus < 201103L
#error This test requires C++11 or better.
#endif

#include "legion.h"
#include "legion/legion_stl.h"

using namespace Legion;
using namespace Legion::STL;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  TEST_TASK_ID,
};

void test_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
  int x;
  long y;
  long long z;
  float w;
  assert((task->arglen == get_serialized_size<int, long, long long, float>()));
  std::tie(x, y, z, w) = deserialize<int, long, long long, float>(task->args);
  printf("test_task got: x %d y %ld z %lld w %.2f\n", x, y, z, w);
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  // Test launching a task with typed arguments
  {
    TypedArgument<int, long, long long, float> args(1, 2, 3, 4);
    TaskLauncher launcher(TEST_TASK_ID, args);
    runtime->execute_task(ctx, launcher);
  }

  // Test for packing and unpacking typed arguments
  {
    int x = 123;
    TypedArgument<int> args(x);
    int x1;
    std::tie(x1) = deserialize<int>(args.get_ptr());
    printf("x1 %d\n", x1);
  }

  {
    int x = 123;
    long y = 456;
    TypedArgument<int, long> args(x, y);
    int x1;
    long y1;
    std::tie(x1, y1) = deserialize<int, long>(args.get_ptr());
    printf("x1 %d y1 %ld\n", x1, y1);
  }

  {
    int x = 123;
    long y = 456;
    long long z = 789;
    TypedArgument<int, long, long long> args(x, y, z);
    int x1;
    long y1;
    long long z1;
    std::tie(x1, y1, z1) = deserialize<int, long, long long>(args.get_ptr());
    printf("x1 %d y1 %ld z1 %lld\n", x1, y1, z1);
  }

  {
    int x = 123;
    long y = 456;
    long long z = 789;
    float w = 3.14;
    TypedArgument<int, long, long long, float> args(x, y, z, w);
    int x1;
    long y1;
    long long z1;
    float w1;
    std::tie(x1, y1, z1, w1) = deserialize<int, long, long long, float>(args.get_ptr());
    printf("x1 %d y1 %ld z1 %lld w1 %.2f\n", x1, y1, z1, w1);
  }
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
    TaskVariantRegistrar registrar(TEST_TASK_ID, "test");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<test_task>(registrar, "test");
  }

  return Runtime::start(argc, argv);
}
