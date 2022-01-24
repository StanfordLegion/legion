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

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_MATRIX_TASK_ID,
  TRANSPOSE_MATRIX_TASK_ID,
  CHECK_MATRIX_TASK_ID,
};

enum FieldIDs {
  FID_MATRIX,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int matrix_dim = 32; 
  // See if we have any command line arguments to parse
  // Note we now have a new command line parameter which specifies
  // how many subregions we should make.
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        matrix_dim = atoi(command_args.argv[++i]);
    }
  }
  printf("Running transpose for %dx%d matrix...\n", matrix_dim, matrix_dim);

  // Create our logical regions using the same schemas as earlier examples
  Rect<2> matrix_rect(Point<2>(0,0),Point<2>(matrix_dim-1,matrix_dim-1));
  IndexSpace matrix_is = runtime->create_index_space(ctx, matrix_rect); 
  runtime->attach_name(matrix_is, "matrix_is");
  FieldSpace matrix_fs = runtime->create_field_space(ctx);
  runtime->attach_name(matrix_fs, "matrix_fs");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, matrix_fs);
    allocator.allocate_field(sizeof(coord_t),FID_MATRIX);
    runtime->attach_name(matrix_fs, FID_MATRIX, "X");
  }
  LogicalRegion input_lr = runtime->create_logical_region(ctx, matrix_is, matrix_fs);
  runtime->attach_name(input_lr, "input_lr");
  LogicalRegion output_lr = runtime->create_logical_region(ctx, matrix_is, matrix_fs);
  runtime->attach_name(output_lr, "output_lr");

  TaskLauncher init_launcher(INIT_MATRIX_TASK_ID, TaskArgument()); 
  init_launcher.add_region_requirement(
      RegionRequirement(input_lr, WRITE_DISCARD, EXCLUSIVE, input_lr));
  init_launcher.region_requirements[0].add_field(FID_MATRIX);
  runtime->execute_task(ctx, init_launcher);

  TaskLauncher transpose_launcher(TRANSPOSE_MATRIX_TASK_ID, TaskArgument());
  transpose_launcher.add_region_requirement(
      RegionRequirement(output_lr, WRITE_DISCARD, EXCLUSIVE, output_lr));
  transpose_launcher.region_requirements[0].add_field(FID_MATRIX);
  transpose_launcher.add_region_requirement(
      RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  transpose_launcher.region_requirements[1].add_field(FID_MATRIX);
  runtime->execute_task(ctx, transpose_launcher);

  TaskLauncher check_launcher(CHECK_MATRIX_TASK_ID, TaskArgument()); 
  check_launcher.add_region_requirement(
      RegionRequirement(output_lr, READ_ONLY, EXCLUSIVE, output_lr));
  check_launcher.region_requirements[0].add_field(FID_MATRIX);
  runtime->execute_task(ctx, check_launcher);

  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, matrix_fs);
  runtime->destroy_index_space(ctx, matrix_is);
}

void init_matrix_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  printf("Initializing matrix...\n");

  const FieldAccessor<WRITE_DISCARD,coord_t,2,coord_t,
        Realm::AffineAccessor<coord_t,2,coord_t> > acc(regions[0], fid);
  Rect<2> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  const coord_t pitch = (rect.hi[0] - rect.lo[0]) + 1;
  for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
    for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
      acc[x][y] = y * pitch + x;
}

void transpose_matrix_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const FieldAccessor<WRITE_DISCARD,coord_t,2,coord_t,
        Realm::AffineAccessor<coord_t,2,coord_t> > acc_out(regions[0], FID_MATRIX);
  const FieldAccessor<READ_ONLY,coord_t,2,coord_t,
        Realm::AffineAccessor<coord_t,2,coord_t> > acc_in(regions[1], FID_MATRIX);
  printf("Transposing matrix...\n");
  Rect<2> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
#ifdef SAFE_TRANSPOSE
  // This is always a safe way to do the transpose explicitly
  for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
    for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
      acc_out[y][x] = acc_in[x][y];
#else
  // Note that because we've explicitly specified layout constraints 
  // with opposing dimension orders when we registered this task variant
  // then we can just do a straight memcpy here as the transpose was 
  // done by the DMA engine for the input to row-major layout
  const coord_t *in_ptr = acc_in.ptr(rect);
  coord_t *out_ptr = acc_out.ptr(rect);
  memcpy(out_ptr, in_ptr, rect.volume() * sizeof(coord_t));
#endif
}

void check_matrix_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  const FieldAccessor<READ_ONLY,coord_t,2,coord_t,
        Realm::AffineAccessor<coord_t,2,coord_t> > acc(regions[0], FID_MATRIX);

  printf("Checking results...\n");
  Rect<2> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  bool all_passed = true;
  const coord_t pitch = (rect.hi[1] - rect.lo[1]) + 1;
  for (coord_t y = rect.lo[1]; y <= rect.hi[1]; y++)
    for (coord_t x = rect.lo[0]; x <= rect.hi[0]; x++)
      if (acc[x][y] != (x * pitch + y))
        all_passed = false;
  if (all_passed)
    printf("SUCCESS!\n");
  else
    printf("FAILURE!\n");
  assert(all_passed);
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  LayoutConstraintID column_major;
  {
    OrderingConstraint order(true/*contiguous*/);
    order.ordering.push_back(DIM_X);
    order.ordering.push_back(DIM_Y);
    order.ordering.push_back(DIM_F);
    LayoutConstraintRegistrar registrar;
    registrar.add_constraint(order);
    column_major = Runtime::preregister_layout(registrar);
  }

  LayoutConstraintID row_major;
  {
    OrderingConstraint order(true/*contiguous*/);
    order.ordering.push_back(DIM_Y);
    order.ordering.push_back(DIM_X);
    order.ordering.push_back(DIM_F);
    LayoutConstraintRegistrar registrar;
    registrar.add_constraint(order);
    row_major = Runtime::preregister_layout(registrar);
  }

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INIT_MATRIX_TASK_ID, "init_matrix");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(0, column_major); 
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_matrix_task>(registrar, "init_matrix");
  }

  {
    TaskVariantRegistrar registrar(TRANSPOSE_MATRIX_TASK_ID, "transpose");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(0, column_major);
    registrar.add_layout_constraint_set(1, row_major);
    registrar.set_leaf();
    Runtime::preregister_task_variant<transpose_matrix_task>(registrar, "transpose");
  }

  {
    TaskVariantRegistrar registrar(CHECK_MATRIX_TASK_ID, "check_matrix");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(0, column_major);
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_matrix_task>(registrar, "check_matrix");
  }

  return Runtime::start(argc, argv);
}
