/* Copyright 2023 Stanford University
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

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  COMPUTE_TASK_ID,
  COMPUTE_FROM_FILE_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
   char input_file[64];
  snprintf(input_file, sizeof input_file, "input.dat");

  Rect<1> rect_A(0,1023);
  IndexSpace is_A = runtime->create_index_space(ctx, rect_A);
  FieldSpace fs_A = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs_A);
    allocator.allocate_field(sizeof(double),FID_X);
    allocator.allocate_field(sizeof(double),FID_Y);
  }
  LogicalRegion lr_A = runtime->create_logical_region(ctx, is_A, fs_A);

  // create an instance of Y
  PhysicalRegion pr_Y = runtime->map_region(ctx,
					    RegionRequirement(lr_A, WRITE_DISCARD, EXCLUSIVE, lr_A)
					    .add_field(FID_Y));
  pr_Y.wait_until_valid();
  runtime->unmap_region(ctx, pr_Y);

  PhysicalRegion pr_A;
  std::vector<FieldID> field_vec;
  field_vec.push_back(FID_X);
  for(int reps = 0; reps < 2; reps++) {
    AttachLauncher alr(EXTERNAL_POSIX_FILE, lr_A, lr_A);
    alr.initialize_constraints(false/*column major*/, true/*soa*/, field_vec);
    alr.privilege_fields.insert(FID_X);
    Realm::ExternalFileResource resource(std::string(input_file), LEGION_FILE_CREATE);
    alr.external_resource = &resource;
    pr_A = runtime->attach_external_resource(ctx, alr);
    
    CopyLauncher clr;
    clr.add_copy_requirements(RegionRequirement(lr_A, READ_ONLY, EXCLUSIVE, lr_A).add_field(FID_Y),
			      RegionRequirement(lr_A, READ_WRITE, EXCLUSIVE, lr_A).add_field(FID_X));
    runtime->issue_copy_operation(ctx, clr);

    runtime->detach_external_resource(ctx, pr_A);
  }
  runtime->destroy_logical_region(ctx, lr_A);
  runtime->destroy_field_space(ctx, fs_A);
  runtime->destroy_index_space(ctx, is_A);
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
