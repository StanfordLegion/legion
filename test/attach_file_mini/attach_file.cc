/* Copyright 2015 Stanford University
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
#include <math.h>
#include "legion.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  COMPUTE_TASK_ID,
  COMPUTE_FROM_FILE_TASK_ID,
};

enum FieldIDs {
  FID_VAL,
  FID_DERIV,
  FID_CP
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
   char input_file[64];
  //sprintf(input_file, "/scratch/sdb1_ext4/input.dat");
  sprintf(input_file, "input.dat");

  Rect<1> rect_A(Point<1>(0), Point<1>(1023));
  IndexSpace is_A = runtime->create_index_space(ctx,
                          Domain::from_rect<1>(rect_A));
  FieldSpace fs_A = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs_A);
    allocator.allocate_field(sizeof(double),FID_VAL);
  }
  LogicalRegion lr_A = runtime->create_logical_region(ctx, is_A, fs_A);
  PhysicalRegion pr_A;
  std::vector<FieldID> field_vec;
  field_vec.push_back(FID_VAL);
  pr_A = runtime->attach_file(ctx, input_file, lr_A, lr_A, field_vec, LEGION_FILE_CREATE);
  runtime->detach_file(ctx, pr_A);
  runtime->destroy_logical_region(ctx, lr_A);
  runtime->destroy_field_space(ctx, fs_A);
  runtime->destroy_index_space(ctx, is_A);
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);
  return HighLevelRuntime::start(argc, argv);
}
