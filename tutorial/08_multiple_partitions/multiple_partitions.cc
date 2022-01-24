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
 * In this example we illustrate how the Legion
 * programming model supports multiple partitions
 * of the same logical region and the benefits it
 * provides by allowing multiple views onto the
 * same logical region.  We compute a simple 5-point
 * 1D stencil using the standard formula:
 * f'(x) = (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h))/12h
 * For simplicity we'll assume h=1.
 */

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_ID,
  STENCIL_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_VAL,
  FID_DERIV,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_elements = 1024;
  int num_subregions = 4;
  // Check for any command line arguments
  {
      const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-b"))
        num_subregions = atoi(command_args.argv[++i]);
    }
  }
  printf("Running stencil computation for %d elements...\n", num_elements);
  printf("Partitioning data into %d sub-regions...\n", num_subregions);

  // For this example we'll create a single logical region with two
  // fields.  We'll initialize the field identified by 'FID_VAL' with
  // our input data and then compute the derivatives stencil values 
  // and write them into the field identified by 'FID_DERIV'.
  Rect<1> elem_rect(0,num_elements-1);
  IndexSpaceT<1> is = runtime->create_index_space(ctx, elem_rect);
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double),FID_VAL);
    allocator.allocate_field(sizeof(double),FID_DERIV);
  }
  LogicalRegion stencil_lr = runtime->create_logical_region(ctx, is, fs);
  
  // Make our color_domain based on the number of subregions
  // that we want to create.
  Rect<1> color_bounds(0,num_subregions-1);
  IndexSpaceT<1> color_is = runtime->create_index_space(ctx, color_bounds);

  // In this example we need to create two partitions: one disjoint
  // partition for describing the output values that are going to
  // be computed by each sub-task that we launch and a second
  // aliased partition which will describe the input values needed
  // for performing each task.  Note that for the second partition
  // each subregion will be a superset of its corresponding region
  // in the first partition, but will also require two 'ghost' cells
  // on each side.  The need for these ghost cells means that the
  // subregions in the second partition will be aliased.
  IndexPartition disjoint_ip = 
    runtime->create_equal_partition(ctx, is, color_is);
  const int block_size = (num_elements + num_subregions - 1) / num_subregions;
  Transform<1,1> transform;
  transform[0][0] = block_size;
  Rect<1> extent(-2, block_size + 1);
  IndexPartition ghost_ip = 
    runtime->create_partition_by_restriction(ctx, is, color_is, transform, extent);

  // Once we've created our index partitions, we can get the
  // corresponding logical partitions for the stencil_lr
  // logical region.
  LogicalPartition disjoint_lp = 
    runtime->get_logical_partition(ctx, stencil_lr, disjoint_ip);
  LogicalPartition ghost_lp = 
    runtime->get_logical_partition(ctx, stencil_lr, ghost_ip);

  // Our launch domain will again be isomorphic to our coloring domain.
  ArgumentMap arg_map;

  // First initialize the 'FID_VAL' field with some data
  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is,
                              TaskArgument(NULL, 0), arg_map);
  init_launcher.add_region_requirement(
      RegionRequirement(disjoint_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, stencil_lr));
  init_launcher.add_field(0, FID_VAL);
  runtime->execute_index_space(ctx, init_launcher);

  // Now we're going to launch our stencil computation.  We
  // specify two region requirements for the stencil task.
  // Each region requirement is upper bounded by one of our
  // two partitions.  The first region requirement requests
  // read-only privileges on the ghost partition.  Note that
  // because we are only requesting read-only privileges, all
  // of our sub-tasks in the index space launch will be 
  // non-interfering.  The second region requirement asks for
  // read-write privileges on the disjoint partition for
  // the 'FID_DERIV' field.  Again this meets with the 
  // mandate that all points in our index space task
  // launch be non-interfering.
  IndexLauncher stencil_launcher(STENCIL_TASK_ID, color_is,
       TaskArgument(&num_elements, sizeof(num_elements)), arg_map);
  stencil_launcher.add_region_requirement(
      RegionRequirement(ghost_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, stencil_lr));
  stencil_launcher.add_field(0, FID_VAL);
  stencil_launcher.add_region_requirement(
      RegionRequirement(disjoint_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, stencil_lr));
  stencil_launcher.add_field(1, FID_DERIV);
  runtime->execute_index_space(ctx, stencil_launcher);

  // Finally, we launch a single task to check the results.
  TaskLauncher check_launcher(CHECK_TASK_ID, 
      TaskArgument(&num_elements, sizeof(num_elements)));
  check_launcher.add_region_requirement(
      RegionRequirement(stencil_lr, READ_ONLY, EXCLUSIVE, stencil_lr));
  check_launcher.add_field(0, FID_VAL);
  check_launcher.add_region_requirement(
      RegionRequirement(stencil_lr, READ_ONLY, EXCLUSIVE, stencil_lr));
  check_launcher.add_field(1, FID_DERIV);
  runtime->execute_task(ctx, check_launcher);

  // Clean up our region, index space, and field space
  runtime->destroy_logical_region(ctx, stencil_lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
  runtime->destroy_index_space(ctx, color_is);
}

// The standard initialize field task from earlier examples
void init_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const int point = task->index_point.point_data[0];
  printf("Initializing field %d for block %d...\n", fid, point);

  const FieldAccessor<WRITE_DISCARD,double,1> acc(regions[0], fid);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    acc[*pir] = drand48();
}

// Our stencil tasks is interesting because it
// has both slow and fast versions depending
// on whether or not its bounds have been clamped.
void stencil_task(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);
  assert(task->arglen == sizeof(int));
  const int max_elements = *((const int*)task->args);
  const int point = task->index_point.point_data[0];
  
  FieldID read_fid = *(task->regions[0].privilege_fields.begin());
  FieldID write_fid = *(task->regions[1].privilege_fields.begin());

  const FieldAccessor<READ_ONLY,double,1> read_acc(regions[0], read_fid);
  const FieldAccessor<WRITE_DISCARD,double,1> write_acc(regions[1], write_fid);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[1].region.get_index_space());
  // If we are on the edges of the entire space we are 
  // operating over, then we're going to do the slow
  // path which checks for clamping when necessary.
  // If not, then we can do the fast path without
  // any checks.
  if ((rect.lo[0] < 2) || (rect.hi[0] > (max_elements-3)))
  {
    printf("Running slow stencil path for point %d...\n", point);
    // Note in the slow path that there are checks which
    // perform clamps when necessary before reading values.
    for (PointInRectIterator<1> pir(rect); pir(); pir++)
    {
      double l2, l1, r1, r2;
      if (pir[0] < 2)
        l2 = read_acc[0];
      else
        l2 = read_acc[*pir - 2];
      if (pir[0] < 1)
        l1 = read_acc[0];
      else
        l1 = read_acc[*pir - 1];
      if (pir[0] > (max_elements-2))
        r1 = read_acc[max_elements-1];
      else
        r1 = read_acc[*pir + 1];
      if (pir[0] > (max_elements-3))
        r2 = read_acc[max_elements-1];
      else
        r2 = read_acc[*pir + 2];
      
      double result = (-l2 + 8.0*l1 - 8.0*r1 + r2) / 12.0;
      write_acc[*pir] = result;
    }
  }
  else
  {
    printf("Running fast stencil path for point %d...\n", point);
    // In the fast path, we don't need any checks
    for (PointInRectIterator<1> pir(rect); pir(); pir++)
    {
      double l2 = read_acc[*pir - 2];
      double l1 = read_acc[*pir - 1];
      double r1 = read_acc[*pir + 1];
      double r2 = read_acc[*pir + 2];

      double result = (-l2 + 8.0*l1 - 8.0*r1 + r2) / 12.0;
      write_acc[*pir] = result;
    }
  }
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);
  assert(task->arglen == sizeof(int));
  const int max_elements = *((const int*)task->args);

  FieldID src_fid = *(task->regions[0].privilege_fields.begin());
  FieldID dst_fid = *(task->regions[1].privilege_fields.begin());

  const FieldAccessor<READ_ONLY,double,1> src_acc(regions[0], src_fid);
  const FieldAccessor<READ_ONLY,double,1> dst_acc(regions[1], dst_fid);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[1].region.get_index_space());

  // This is the checking task so we can just do the slow path
  bool all_passed = true;
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
  {
    double l2, l1, r1, r2;
    if (pir[0] < 2)
      l2 = src_acc[0];
    else
      l2 = src_acc[*pir - 2];
    if (pir[0] < 1)
      l1 = src_acc[0];
    else
      l1 = src_acc[*pir - 1];
    if (pir[0] > (max_elements-2))
      r1 = src_acc[max_elements-1];
    else
      r1 = src_acc[*pir + 1];
    if (pir[0] > (max_elements-3))
      r2 = src_acc[max_elements-1];
    else
      r2 = src_acc[*pir + 2];
    
    double expected = (-l2 + 8.0*l1 - 8.0*r1 + r2) / 12.0;
    double received = dst_acc[*pir];
    // Probably shouldn't bitwise compare floating point
    // numbers but the order of operations are the same so they
    // should be bitwise equal.
    if (expected != received)
      all_passed = false;
  }
  if (all_passed)
    printf("SUCCESS!\n");
  else
    printf("FAILURE!\n");
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
    TaskVariantRegistrar registrar(INIT_FIELD_TASK_ID, "init_field");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_field_task>(registrar, "init_field");
  }

  {
    TaskVariantRegistrar registrar(STENCIL_TASK_ID, "stencil");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<stencil_task>(registrar, "stencil");
  }

  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  return Runtime::start(argc, argv);
}
