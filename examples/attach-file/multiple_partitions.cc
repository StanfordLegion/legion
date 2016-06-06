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


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"
using namespace Legion;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

/*
 * In this example we illustrate how the Legion
 * programming model supports multiple partitions
 * of the same logical region and the benefits it
 * provides by allowing multiple views onto the
 * same logical region.  We compute a simple 5-point
 * 1D stencil using the standard forumala:
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
  FID_CP
};

void generate_hdf_file(const char* file_name, int num_elemnts)
{
  double *arr;
  arr = (double*) calloc(num_elemnts, sizeof(double));
  hid_t file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hsize_t dims[1];
  dims[0] = num_elemnts;
  hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
  hid_t dataset = H5Dcreate2(file_id, "FID_CP", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_IEEE_F64BE, H5S_ALL, H5S_ALL, H5P_DEFAULT, arr);
  H5Dclose(dataset);
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
  free(arr);
}

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
  Rect<1> elem_rect(Point<1>(0),Point<1>(num_elements-1));
  IndexSpace is = runtime->create_index_space(ctx, 
                          Domain::from_rect<1>(elem_rect));
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double),FID_VAL);
    allocator.allocate_field(sizeof(double),FID_DERIV);
  }
  LogicalRegion stencil_lr = runtime->create_logical_region(ctx, is, fs);

  FieldSpace cp_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, cp_fs);
    allocator.allocate_field(sizeof(double),FID_CP);
  }
  LogicalRegion cp_lr = runtime->create_logical_region(ctx, is, cp_fs);
  
  // Make our color_domain based on the number of subregions
  // that we want to create.
  Rect<1> color_bounds(Point<1>(0),Point<1>(num_subregions-1));
  Domain color_domain = Domain::from_rect<1>(color_bounds);

  // In this example we need to create two partitions: one disjoint
  // partition for describing the output values that are going to
  // be computed by each sub-task that we launch and a second
  // aliased partition which will describe the input values needed
  // for performing each task.  Note that for the second partition
  // each subregion will be a superset of its corresponding region
  // in the first partition, but will also require two 'ghost' cells
  // on each side.  The need for these ghost cells means that the
  // subregions in the second partition will be aliased.
  IndexPartition disjoint_ip, ghost_ip;
  {
    const int lower_bound = num_elements/num_subregions;
    const int upper_bound = lower_bound+1;
    const int number_small = num_subregions - (num_elements % num_subregions);
    DomainColoring disjoint_coloring, ghost_coloring;
    int index = 0;
    // Iterate over all the colors and compute the entry
    // for both partitions for each color.
    for (int color = 0; color < num_subregions; color++)
    {
      int num_elmts = color < number_small ? lower_bound : upper_bound;
      assert((index+num_elmts) <= num_elements);
      Rect<1> subrect(Point<1>(index),Point<1>(index+num_elmts-1));
      disjoint_coloring[color] = Domain::from_rect<1>(subrect);
      // Now compute the points assigned to this color for
      // the second partition.  Here we need a superset of the
      // points that we just computed including the two additional
      // points on each side.  We handle the edge cases by clamping
      // values to their minimum and maximum values.  This creates
      // four cases of clamping both above and below, clamping below,
      // clamping above, and no clamping.
      if (index < 2)
      {
        if ((index+num_elmts+2) > num_elements)
        {
          // Clamp both
          Rect<1> ghost_rect(Point<1>(0),Point<1>(num_elements-1));
          ghost_coloring[color] = Domain::from_rect<1>(ghost_rect);
        }
        else
        {
          // Clamp below
          Rect<1> ghost_rect(Point<1>(0),Point<1>(index+num_elmts+1));
          ghost_coloring[color] = Domain::from_rect<1>(ghost_rect);
        }
      }
      else
      {
        if ((index+num_elmts+2) > num_elements)
        {
          // Clamp above
          Rect<1> ghost_rect(Point<1>(index-2),Point<1>(num_elements-1));
          ghost_coloring[color] = Domain::from_rect<1>(ghost_rect);
        }
        else
        {
          // Normal case
          Rect<1> ghost_rect(Point<1>(index-2),Point<1>(index+num_elmts+1));
          ghost_coloring[color] = Domain::from_rect<1>(ghost_rect);
        }
      }
      index += num_elmts;
    }
    // Once we've computed both of our colorings then we can
    // create our partitions.  Note that we tell the runtime
    // that one is disjoint will the second one is not.
    disjoint_ip = runtime->create_index_partition(ctx, is, color_domain,
                                    disjoint_coloring, true/*disjoint*/);
    ghost_ip = runtime->create_index_partition(ctx, is, color_domain,
                                    ghost_coloring, false/*disjoint*/);
  }

  // Once we've created our index partitions, we can get the
  // corresponding logical partitions for the stencil_lr
  // logical region.
  LogicalPartition disjoint_lp = 
    runtime->get_logical_partition(ctx, stencil_lr, disjoint_ip);
  LogicalPartition ghost_lp = 
    runtime->get_logical_partition(ctx, stencil_lr, ghost_ip);

  // Our launch domain will again be isomorphic to our coloring domain.
  Domain launch_domain = color_domain;
  ArgumentMap arg_map;

  // First initialize the 'FID_VAL' field with some data
  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, launch_domain,
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
  IndexLauncher stencil_launcher(STENCIL_TASK_ID, launch_domain,
       TaskArgument(&num_elements, sizeof(num_elements)), arg_map);
  stencil_launcher.add_region_requirement(
      RegionRequirement(ghost_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, stencil_lr));
  stencil_launcher.add_field(0, FID_VAL);
  stencil_launcher.add_region_requirement(
      RegionRequirement(disjoint_lp, 0/*projection ID*/,
                        READ_WRITE, EXCLUSIVE, stencil_lr));
  stencil_launcher.add_field(1, FID_DERIV);
  runtime->execute_index_space(ctx, stencil_launcher);

  // Launcher a copy operation that performs checkpoint
  std::vector<FieldID> field_vec;
  field_vec.push_back(FID_CP);
  struct timespec ts_start, ts_mid, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);
  PhysicalRegion cp_pr = runtime->attach_file(ctx, "checkpoint.dat", cp_lr, cp_lr, field_vec, LEGION_FILE_CREATE);
  //cp_pr.wait_until_valid();
  CopyLauncher copy_launcher;
  copy_launcher.add_copy_requirements(
      RegionRequirement(stencil_lr, READ_ONLY, EXCLUSIVE, stencil_lr),
      RegionRequirement(cp_lr, WRITE_DISCARD, EXCLUSIVE, cp_lr));
  copy_launcher.add_src_field(0, FID_DERIV);
  copy_launcher.add_dst_field(0, FID_CP);
  runtime->issue_copy_operation(ctx, copy_launcher);
  
  clock_gettime(CLOCK_MONOTONIC, &ts_mid);
  runtime->detach_file(ctx, cp_pr);
  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  double attach_time = ((1.0 * (ts_mid.tv_sec - ts_start.tv_sec)) +
                     (1e-9 * (ts_mid.tv_nsec - ts_start.tv_nsec)));
  double detach_time = ((1.0 * (ts_end.tv_sec - ts_mid.tv_sec)) +
                     (1e-9 * (ts_end.tv_nsec - ts_mid.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", attach_time);
  printf("ELAPSED TIME = %7.3f s\n", detach_time);
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
  runtime->destroy_logical_region(ctx, cp_lr);
  runtime->destroy_field_space(ctx, cp_fs);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
  printf("End of TOP_LEVEL_TASK\n");
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

  RegionAccessor<AccessorType::Generic, double> acc = 
    regions[0].get_field_accessor(fid).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();
  for (GenericPointInRectIterator<1> pir(rect); pir; pir++)
  {
    acc.write(DomainPoint::from_point<1>(pir.p), drand48());
  }
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

  RegionAccessor<AccessorType::Generic, double> read_acc = 
    regions[0].get_field_accessor(read_fid).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> write_acc = 
    regions[1].get_field_accessor(write_fid).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[1].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();
  const DomainPoint zero = DomainPoint::from_point<1>(Point<1>(0));
  const DomainPoint max = DomainPoint::from_point<1>(Point<1>(max_elements-1));
  const Point<1> one(1);
  const Point<1> two(2);
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
    for (GenericPointInRectIterator<1> pir(rect); pir; pir++)
    {
      double l2, l1, r1, r2;
      if (pir.p[0] < 2)
        l2 = read_acc.read(zero);
      else
        l2 = read_acc.read(DomainPoint::from_point<1>(pir.p-two));
      if (pir.p[0] < 1)
        l1 = read_acc.read(zero);
      else
        l1 = read_acc.read(DomainPoint::from_point<1>(pir.p-one));
      if (pir.p[0] > (max_elements-2))
        r1 = read_acc.read(max);
      else
        r1 = read_acc.read(DomainPoint::from_point<1>(pir.p+one));
      if (pir.p[0] > (max_elements-3))
        r2 = read_acc.read(max);
      else
        r2 = read_acc.read(DomainPoint::from_point<1>(pir.p+two));
      
      double result = (-l2 + 8.0*l1 - 8.0*r1 + r2) / 12.0;
      write_acc.write(DomainPoint::from_point<1>(pir.p), result);
    }
  }
  else
  {
    printf("Running fast stencil path for point %d...\n", point);
    // In the fast path, we don't need any checks
    for (GenericPointInRectIterator<1> pir(rect); pir; pir++)
    {
      double l2 = read_acc.read(DomainPoint::from_point<1>(pir.p-two));
      double l1 = read_acc.read(DomainPoint::from_point<1>(pir.p-one));
      double r1 = read_acc.read(DomainPoint::from_point<1>(pir.p+one));
      double r2 = read_acc.read(DomainPoint::from_point<1>(pir.p+two));

      double result = (-l2 + 8.0*l1 - 8.0*r1 + r2) / 12.0;
      write_acc.write(DomainPoint::from_point<1>(pir.p), result);
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

  RegionAccessor<AccessorType::Generic, double> src_acc = 
    regions[0].get_field_accessor(src_fid).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> dst_acc = 
    regions[1].get_field_accessor(dst_fid).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[1].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();
  const DomainPoint zero = DomainPoint::from_point<1>(Point<1>(0));
  const DomainPoint max = DomainPoint::from_point<1>(Point<1>(max_elements-1));
  const Point<1> one(1);
  const Point<1> two(2);

  // This is the checking task so we can just do the slow path
  bool all_passed = true;
  for (GenericPointInRectIterator<1> pir(rect); pir; pir++)
  {
    double l2, l1, r1, r2;
    if (pir.p[0] < 2)
      l2 = src_acc.read(zero);
    else
      l2 = src_acc.read(DomainPoint::from_point<1>(pir.p-two));
    if (pir.p[0] < 1)
      l1 = src_acc.read(zero);
    else
      l1 = src_acc.read(DomainPoint::from_point<1>(pir.p-one));
    if (pir.p[0] > (max_elements-2))
      r1 = src_acc.read(max);
    else
      r1 = src_acc.read(DomainPoint::from_point<1>(pir.p+one));
    if (pir.p[0] > (max_elements-3))
      r2 = src_acc.read(max);
    else
      r2 = src_acc.read(DomainPoint::from_point<1>(pir.p+two));
    
    double expected = (-l2 + 8.0*l1 - 8.0*r1 + r2) / 12.0;
    double received = dst_acc.read(DomainPoint::from_point<1>(pir.p));
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
  Runtime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);
  Runtime::register_legion_task<init_field_task>(INIT_FIELD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/);
  Runtime::register_legion_task<stencil_task>(STENCIL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/);
  Runtime::register_legion_task<check_task>(CHECK_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/);

  return Runtime::start(argc, argv);
}
