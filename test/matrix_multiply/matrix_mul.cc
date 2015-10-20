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
#include "legion.h"
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  COMPUTE_TASK_ID,
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
                    Context ctx, HighLevelRuntime *runtime)
{
  int nsize = 1024;
  int nregions = 4;
  int need_init_file = 0;
  char input_file[128];
  sprintf(input_file, "input_file.dat");

  // Check for any command line arguments
  {
      const InputArgs &command_args = HighLevelRuntime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        nsize = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-s"))
        nregions = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-f"))
        need_init_file = atoi(command_args.argv[++i]);
    }
  }

  printf("Running stencil computation for %d elements...\n", nsize);
  printf("Partitioning data into %d sub-regions...\n", nregions);

  Rect<2> rect_A(make_point(0, 0),make_point(nsize * nregions - 1, nsize - 1));
  IndexSpace is_A = runtime->create_index_space(ctx,
                          Domain::from_rect<2>(rect_A));
  FieldSpace fs_A = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs_A);
    allocator.allocate_field(sizeof(double),FID_VAL);
  }
  LogicalRegion lr_A = runtime->create_logical_region(ctx, is_A, fs_A);

  if (need_init_file) {
    RegionRequirement req(lr_A, WRITE_DISCARD, EXCLUSIVE, lr_A);
    req.add_field(FID_VAL);
    InlineLauncher input_launcher(req);
    PhysicalRegion pr_A = runtime->map_region(ctx, input_launcher);
    pr_A.wait_until_valid();
    RegionAccessor<AccessorType::Generic, double> acc =
      pr_A.get_field_accessor(FID_VAL).typeify<double>();
    for (GenericPointInRectIterator<2> pir(rect_A); pir; pir++) {
      acc.write(DomainPoint::from_point<2>(pir.p), drand48());
    }

    LogicalRegion lr_C = runtime->create_logical_region(ctx, is_A, fs_A);
    std::vector<FieldID> field_vec;
    field_vec.push_back(FID_VAL);
    PhysicalRegion pr_C = runtime->attach_file(ctx, input_file, lr_C, lr_C, field_vec, LEGION_FILE_CREATE);
    runtime->remap_region(ctx, pr_C);
    CopyLauncher copy_launcher;
    copy_launcher.add_copy_requirements(
        RegionRequirement(lr_A, READ_ONLY, EXCLUSIVE, lr_A),
        RegionRequirement(lr_C, WRITE_DISCARD, EXCLUSIVE, lr_C));
    copy_launcher.add_src_field(0, FID_VAL);
    copy_launcher.add_dst_field(0, FID_VAL);
    runtime->issue_copy_operation(ctx, copy_launcher);

    runtime->unmap_region(ctx, pr_A);
    runtime->unmap_region(ctx, pr_C);
    runtime->detach_file(ctx, pr_C);
    runtime->destroy_logical_region(ctx, lr_A);
    runtime->destroy_logical_region(ctx, lr_C);
    runtime->destroy_field_space(ctx, fs_A);
    runtime->destroy_index_space(ctx, is_A);
    return;
  }

  Rect<2> rect_B(make_point(0, 0),make_point(nsize - 1, nsize - 1));
  IndexSpace is_B = runtime->create_index_space(ctx,
                          Domain::from_rect<2>(rect_B));
  FieldSpace fs_B = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, fs_B);
    allocator.allocate_field(sizeof(double),FID_VAL);
  }
  LogicalRegion lr_B = runtime->create_logical_region(ctx, is_B, fs_B);

  std::vector<FieldID> field_vec;
  field_vec.push_back(FID_VAL);
  PhysicalRegion pr_A = runtime->attach_file(ctx, input_file, lr_A, lr_A, field_vec, LEGION_FILE_READ_ONLY);
  runtime->remap_region(ctx, pr_A);
  pr_A.wait_until_valid();

  // Acquire the logical reagion so that we can launch sub-operations that make copies
  AcquireLauncher acquire_launcher(lr_A, lr_A, pr_A);
  acquire_launcher.add_field(FID_VAL);
  runtime->issue_acquire(ctx, acquire_launcher);

  // Make our color_domain based on the number of subregions
  // that we want to create.
  Rect<1> color_bounds(Point<1>(0),Point<1>(nregions-1));
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
  IndexPartition ip_A, ip_B;
  {
    DomainColoring coloring_A, coloring_B;
    // Iterate over all the colors and compute the entry
    // for both partitions for each color.
    for (int color = 0; color < nregions; color++)
    {
      Rect<2> subrect_A(make_point(color * nsize, 0),make_point(color * nsize + nsize - 1, nsize - 1));
      Rect<2> subrect_B(make_point(0, 0), make_point(nsize - 1, nsize - 1));
      coloring_A[color] = Domain::from_rect<2>(subrect_A);
      coloring_B[color] = Domain::from_rect<2>(subrect_B);
    }
    // Once we've computed both of our colorings then we can
    // create our partitions.  Note that we tell the runtime
    // that one is disjoint will the second one is not.
    ip_A = runtime->create_index_partition(ctx, is_A, color_domain,
                                           coloring_A, true/*disjoint*/);
    ip_B = runtime->create_index_partition(ctx, is_B, color_domain,
                                           coloring_B, false);
  }

  // Once we've created our index partitions, we can get the
  // corresponding logical partitions for the stencil_lr
  // logical region.
  LogicalPartition lp_A =
    runtime->get_logical_partition(ctx, lr_A, ip_A);
  LogicalPartition lp_B =
    runtime->get_logical_partition(ctx, lr_B, ip_B);

  // Our launch domain will again be isomorphic to our coloring domain.
  Domain launch_domain = color_domain;
  ArgumentMap arg_map;

  // First initialize the 'FID_VAL' field with some data
  IndexLauncher compute_launcher(COMPUTE_TASK_ID, launch_domain,
                              TaskArgument(NULL, 0), arg_map);
  compute_launcher.add_region_requirement(
      RegionRequirement(lp_A, 0, READ_ONLY, EXCLUSIVE, lr_A));
  compute_launcher.add_field(0, FID_VAL);
  compute_launcher.add_region_requirement(
      RegionRequirement(lp_B, 0, READ_ONLY, EXCLUSIVE, lr_B));
  compute_launcher.add_field(1, FID_VAL);
  runtime->execute_index_space(ctx, compute_launcher);

  //Release and unmap the attached physicalregion
  ReleaseLauncher release_launcher(lr_A, lr_A, pr_A);
  release_launcher.add_field(FID_VAL);
  runtime->issue_release(ctx, release_launcher);
  runtime->unmap_region(ctx, pr_A);
  runtime->detach_file(ctx, pr_A);

  // Clean up our region, index space, and field space
  runtime->destroy_logical_region(ctx, lr_A);
  runtime->destroy_logical_region(ctx, lr_B);
  runtime->destroy_field_space(ctx, fs_A);
  runtime->destroy_field_space(ctx, fs_B);
  runtime->destroy_index_space(ctx, is_A);
  runtime->destroy_index_space(ctx, is_B);
}

// Our stencil tasks is interesting because it
// has both slow and fast versions depending
// on whether or not its bounds have been clamped.
void compute_task(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);
  FieldID fid_A = *(task->regions[0].privilege_fields.begin());
  FieldID fid_B = *(task->regions[1].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, double> acc_A =
    regions[0].get_field_accessor(fid_A).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_B =
    regions[1].get_field_accessor(fid_B).typeify<double>();

  Domain dom_A = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  int nsize = dom_A.get_rect<2>().dim_size(0);
  assert(dom_A.get_rect<2>().dim_size(1) == nsize);
  Point<2> lo_A = dom_A.get_rect<2>().lo;
  double sum = 0;
  for (int k = 0; k < nsize; k++)
    for (int i = 0; i < nsize; i++)
      for (int j = 0; j < nsize; j++) {
        double x = acc_A.read(DomainPoint::from_point<2>(lo_A + make_point(i, k)));
        double y = acc_B.read(DomainPoint::from_point<2>(make_point(j, k)));
        sum += x * y;
      }
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);
  HighLevelRuntime::register_legion_task<compute_task>(COMPUTE_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/);

  return HighLevelRuntime::start(argc, argv);
}
