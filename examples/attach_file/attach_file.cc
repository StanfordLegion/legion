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
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include "legion.h"
#ifdef LEGION_USE_HDF5
#include <hdf5.h>
#endif
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
  FID_CP
};

bool generate_disk_file(const char *file_name, int num_elements)
{
  // strip off any filename prefix starting with a colon
  {
    const char *pos = strchr(file_name, ':');
    if(pos) file_name = pos + 1;
  }

  // create the file if needed
  int fd = open(file_name, O_CREAT | O_WRONLY, 0666);
  if(fd < 0) {
    perror("open");
    return false;
  }

  // make it large enough to hold 'num_elements' doubles
  int res = ftruncate(fd, num_elements * sizeof(double));
  if(res < 0) {
    perror("ftruncate");
    close(fd);
    return false;
  }

  // now close the file - the Legion runtime will reopen it on the attach
  close(fd);
  return true;
}

#ifdef LEGION_USE_HDF5
bool generate_hdf_file(const char *file_name, const char *dataset_name, int num_elements)
{
  // strip off any filename prefix starting with a colon
  {
    const char *pos = strchr(file_name, ':');
    if(pos) file_name = pos + 1;
  }

  hid_t file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if(file_id < 0) {
    printf("H5Fcreate failed: %lld\n", (long long)file_id);
    return false;
  }

  hsize_t dims[1];
  dims[0] = num_elements;
  hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
  if(dataspace_id < 0) {
    printf("H5Screate_simple failed: %lld\n", (long long)dataspace_id);
    H5Fclose(file_id);
    return false;
  }

  hid_t loc_id = file_id;
  std::vector<hid_t> group_ids;
  // leading slash in dataset path is optional - ignore if present
  if(*dataset_name == '/') dataset_name++;
  while(true) {
    const char *pos = strchr(dataset_name, '/');
    if(!pos) break;
    char *group_name = strndup(dataset_name, pos - dataset_name);
    hid_t id = H5Gcreate2(loc_id, group_name,
			  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if(id < 0) {
      printf("H5Gcreate2 failed: %lld\n", (long long)id);
      for(std::vector<hid_t>::const_iterator it = group_ids.begin();
	  it != group_ids.end();
	  ++it)
	H5Gclose(*it);
      H5Sclose(dataspace_id);
      H5Fclose(file_id);
      return false;
    }
    group_ids.push_back(id);
    loc_id = id;
    dataset_name = pos + 1;
  }
  
  hid_t dataset = H5Dcreate2(loc_id, dataset_name,
			     H5T_IEEE_F64LE, dataspace_id,
			     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if(dataset < 0) {
    printf("H5Dcreate2 failed: %lld\n", (long long)dataset);
    for(std::vector<hid_t>::const_iterator it = group_ids.begin();
	it != group_ids.end();
	++it)
      H5Gclose(*it);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    return false;
  }

  // close things up - attach will reopen later
  H5Dclose(dataset);
  for(std::vector<hid_t>::const_iterator it = group_ids.begin();
      it != group_ids.end();
      ++it)
    H5Gclose(*it);
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
  return true;
}
#endif

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_elements = 1024;
  int num_subregions = 4;
  char disk_file_name[256];
  strcpy(disk_file_name, "checkpoint.dat");
#ifdef LEGION_USE_HDF5
  char hdf5_file_name[256];
  char hdf5_dataset_name[256];
  hdf5_file_name[0] = 0;
  strcpy(hdf5_dataset_name, "FID_CP");
#endif
  // Check for any command line arguments
  {
      const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-b"))
        num_subregions = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-f"))
	strcpy(disk_file_name, command_args.argv[++i]);
#ifdef LEGION_USE_HDF5
      if (!strcmp(command_args.argv[i],"-h"))
	strcpy(hdf5_file_name, command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-d"))
	strcpy(hdf5_dataset_name, command_args.argv[++i]);
#endif
    }
  }
  printf("Running stencil computation for %d elements...\n", num_elements);
  printf("Partitioning data into %d sub-regions...\n", num_subregions);

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

  FieldSpace cp_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, cp_fs);
    allocator.allocate_field(sizeof(double), FID_CP);
  }
  LogicalRegion cp_lr = runtime->create_logical_region(ctx, is, cp_fs);
  
  Rect<1> color_bounds(0,num_subregions-1);
  IndexSpaceT<1> color_is = runtime->create_index_space(ctx, color_bounds);

  IndexPartition disjoint_ip = 
    runtime->create_equal_partition(ctx, is, color_is);
  const int block_size = (num_elements + num_subregions - 1) / num_subregions;
  Transform<1,1> transform;
  transform[0][0] = block_size;
  Rect<1> extent(-2, block_size + 1);
  IndexPartition ghost_ip = 
    runtime->create_partition_by_restriction(ctx, is, color_is, transform, extent);

  LogicalPartition disjoint_lp = 
    runtime->get_logical_partition(ctx, stencil_lr, disjoint_ip);
  LogicalPartition ghost_lp = 
    runtime->get_logical_partition(ctx, stencil_lr, ghost_ip);

  ArgumentMap arg_map;

  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is,
                              TaskArgument(NULL, 0), arg_map);
  init_launcher.add_region_requirement(
      RegionRequirement(disjoint_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, stencil_lr));
  init_launcher.add_field(0, FID_VAL);
  runtime->execute_index_space(ctx, init_launcher);

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

  // Launcher a copy operation that performs checkpoint
  //struct timespec ts_start, ts_mid, ts_end;
  //clock_gettime(CLOCK_MONOTONIC, &ts_start);
  double ts_start, ts_mid, ts_end;
  ts_start = Realm::Clock::current_time_in_microseconds();
  PhysicalRegion cp_pr;
#ifdef LEGION_USE_HDF5
  if(*hdf5_file_name) {
    // create the HDF5 file first - attach wants it to already exist
    bool ok = generate_hdf_file(hdf5_file_name, hdf5_dataset_name, num_elements);
    assert(ok);
    std::map<FieldID,const char*> field_map;
    field_map[FID_CP] = hdf5_dataset_name;
    printf("Checkpointing data to HDF5 file '%s' (dataset='%s')\n",
	   hdf5_file_name, hdf5_dataset_name);
    AttachLauncher al(LEGION_EXTERNAL_HDF5_FILE, cp_lr, cp_lr);
    al.attach_hdf5(hdf5_file_name, field_map, LEGION_FILE_READ_WRITE);
    cp_pr = runtime->attach_external_resource(ctx, al);
  } else
#endif
  {
    // create the disk file first - attach wants it to already exist
    bool ok = generate_disk_file(disk_file_name, num_elements);
    assert(ok);
    std::vector<FieldID> field_vec;
    field_vec.push_back(FID_CP);
    printf("Checkpointing data to disk file '%s'\n",
	   disk_file_name);
    AttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, cp_lr, cp_lr);
    al.attach_file(disk_file_name, field_vec, LEGION_FILE_READ_WRITE);
    cp_pr = runtime->attach_external_resource(ctx, al);
  }
  //cp_pr.wait_until_valid();
  CopyLauncher copy_launcher;
  copy_launcher.add_copy_requirements(
      RegionRequirement(stencil_lr, READ_ONLY, EXCLUSIVE, stencil_lr),
      RegionRequirement(cp_lr, WRITE_DISCARD, EXCLUSIVE, cp_lr));
  copy_launcher.add_src_field(0, FID_DERIV);
  copy_launcher.add_dst_field(0, FID_CP);
  runtime->issue_copy_operation(ctx, copy_launcher);
  
  //clock_gettime(CLOCK_MONOTONIC, &ts_mid);
  ts_mid = Realm::Clock::current_time_in_microseconds();
  {
    Future f = runtime->detach_external_resource(ctx, cp_pr);
    f.get_void_result(true /*silence warnings*/);
  }
  //clock_gettime(CLOCK_MONOTONIC, &ts_end);
  ts_end = Realm::Clock::current_time_in_microseconds();
  //double attach_time = ((1.0 * (ts_mid.tv_sec - ts_start.tv_sec)) +
  //                   (1e-9 * (ts_mid.tv_nsec - ts_start.tv_nsec)));
  //double detach_time = ((1.0 * (ts_end.tv_sec - ts_mid.tv_sec)) +
  //                   (1e-9 * (ts_end.tv_nsec - ts_mid.tv_nsec)));
  double attach_time = 1e-6 * (ts_mid - ts_start);
  double detach_time = 1e-6 * (ts_end - ts_mid);
  printf("ELAPSED TIME (ATTACH) = %7.3f s\n", attach_time);
  printf("ELAPSED TIME (DETACH) = %7.3f s\n", detach_time);

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
    Runtime::preregister_task_variant<init_field_task>(registrar, "init_field");
  }

  {
    TaskVariantRegistrar registrar(STENCIL_TASK_ID, "stencil");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<stencil_task>(registrar, "stencil");
  }

  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  return Runtime::start(argc, argv);
}
