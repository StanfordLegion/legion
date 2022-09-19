/* Copyright 2022 Stanford University, Los Alamos National Laboratory
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
#include <sys/time.h>
#include <math.h>
#include <limits>
#include <unistd.h>
#include "legion.h"
using namespace Legion;

typedef FieldAccessor<READ_ONLY,double,1,coord_t,
                      Realm::AffineAccessor<double,1,coord_t> > AccessorRO;
typedef FieldAccessor<WRITE_DISCARD,double,1,coord_t,
                      Realm::AffineAccessor<double,1,coord_t> > AccessorWD;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  DAXPY_TASK_ID,
  CHECK_TASK_ID,
  GPU_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y,
  FID_Z,
};

typedef struct{
    double x;
    double y;
    double z;
}daxpy_t;

double get_cur_time() {
  struct timeval   tv;
  struct timezone  tz;
  double cur_time;

  gettimeofday(&tv, &tz);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;

  return cur_time;
}

bool compare_double(double a, double b)
{
  return fabs(a - b) < std::numeric_limits<double>::epsilon();
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_elements = 1024; 
  int num_subregions = 4;
  // See if we have any command line arguments to parse
  // Note we now have a new command line parameter which specifies
  // how many subregions we should make.
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
  printf("Running daxpy for %d elements...\n", num_elements);
  printf("Partitioning data into %d sub-regions...\n", num_subregions);

  // Create our logical regions using the same schemas as earlier examples
  Rect<1> elem_rect(0,num_elements-1);
  // IndexSpace is = runtime->create_index_space(ctx, elem_rect, 0, "Element IndexSpace");
  Domain elem_domain = Domain(elem_rect);
  Future bound_future = Future::from_value<Domain>(elem_domain);
  std::string is_prov = "Element IndexSpace:" + std::to_string(__LINE__);
  IndexSpace is = runtime->create_index_space(ctx, 1, bound_future, 0, is_prov.c_str()); 
  runtime->attach_name(is, "is");
  Future field_size_future = Future::from_value<size_t>(sizeof(double));
  std::vector<Future> field_sizes{field_size_future, field_size_future};
  std::vector<FieldID> field_ids{FID_X, FID_Y};
  std::string field_xy_prov = "Element FieldSpace XY:" + std::to_string(__LINE__);
  FieldSpace fs = runtime->create_field_space(ctx, field_sizes, field_ids, 0, field_xy_prov.c_str());
  runtime->attach_name(fs, "fs");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    std::string field_z_prov = "Element FieldSpace Z:" + std::to_string(__LINE__);
    allocator.allocate_field(sizeof(double),FID_Z, 0, false, field_z_prov.c_str());
    runtime->attach_name(fs, FID_Z, "Z");
  }
  std::string input_lr_prov = "Input LR:" + std::to_string(__LINE__);
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, fs, false, input_lr_prov.c_str());
  runtime->attach_name(input_lr, "input_lr");
  std::string output_lr_prov = "Output LR:" + std::to_string(__LINE__);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, is, fs, false, output_lr_prov.c_str());
  runtime->attach_name(output_lr, "output_lr");
  
  PhysicalRegion xy_pr, z_pr;
  double *z_ptr = NULL;
  double *xy_ptr = NULL;
  double *xyz_ptr = NULL;

  // AttachLauncher
  xy_ptr = (double*)malloc(2*sizeof(double)*(num_elements));
  z_ptr = (double*)malloc(sizeof(double)*(num_elements));
  for (int j = 0; j < num_elements; j++ ) {
      xy_ptr[j]               = drand48();
      xy_ptr[num_elements+j]  = drand48();
      z_ptr[j]                = drand48();
  }
  {
    printf("Attach SOA array fid %d, fid %d, ptr %p\n", 
          FID_X, FID_Y, xy_ptr);
    AttachLauncher launcher(EXTERNAL_INSTANCE, input_lr, input_lr);
    std::vector<FieldID> attach_fields(2);
    attach_fields[0] = FID_X;
    attach_fields[1] = FID_Y;
    launcher.attach_array_soa(xy_ptr, false/*column major*/, attach_fields);
    std::string launcher_prov = "Attach XY SOA:" + std::to_string(__LINE__);
    launcher.provenance = launcher_prov;
    xy_pr = runtime->attach_external_resource(ctx, launcher);
  }
  { 
    printf("Attach SOA array fid %d, ptr %p\n", FID_Z, z_ptr);
    AttachLauncher launcher(EXTERNAL_INSTANCE, output_lr, output_lr);
    std::vector<FieldID> attach_fields(1);
    attach_fields[0] = FID_Z;
    launcher.attach_array_soa(z_ptr, false/*column major*/, attach_fields);
    std::string launcher_prov = "Attach Z SOA:" + std::to_string(__LINE__);
    launcher.provenance = launcher_prov;
    z_pr = runtime->attach_external_resource(ctx, launcher);
  }
  
  Rect<1> color_bounds(0,num_subregions-1);
  Domain color_domain = Domain(color_bounds);
  Future color_future = Future::from_value<Domain>(color_domain); 
  std::string color_is_prov = "Color IndexSpace:" + std::to_string(__LINE__);
  IndexSpace color_is = runtime->create_index_space(ctx, 1, color_future, 0, color_is_prov.c_str());

  std::string ip_prov = "Equal Partition:" + std::to_string(__LINE__);
  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is, 
                                                      1, LEGION_AUTO_GENERATE_ID,
                                                      ip_prov.c_str());
  runtime->attach_name(ip, "ip");

  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
  runtime->attach_name(input_lp, "input_lp");
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, ip);
  runtime->attach_name(output_lp, "output_lp");

  ArgumentMap arg_map;

  // init x and y
  // test IndexFillLauncher
  double x_value = 0.5;
  UntypedBuffer x_buff(&x_value, sizeof(x_value));
  IndexFillLauncher fillx_launcher(color_is, input_lp, input_lr, x_buff);
  std::string fillx_launcher_prov = "Fill_X IndexFillLauncher:" + std::to_string(__LINE__);
  fillx_launcher.provenance = fillx_launcher_prov;
  fillx_launcher.add_field(FID_X);
  runtime->fill_fields(ctx, fillx_launcher);

  // test IndexCopyLauncher
  IndexCopyLauncher init_copylauncher(color_is);
  std::string init_copylauncher_prov = "Copy_Y IndexCopyLauncher:" + std::to_string(__LINE__);
  init_copylauncher.provenance = init_copylauncher_prov;
  init_copylauncher.add_copy_requirements(
    RegionRequirement(input_lp, 0/*projection ID*/, READ_ONLY, EXCLUSIVE, input_lr),
    RegionRequirement(input_lp, 0/*projection ID*/, WRITE_DISCARD, EXCLUSIVE, input_lr)
  );
  init_copylauncher.add_src_field(0, FID_X);
  init_copylauncher.add_dst_field(0, FID_Y);
  runtime->issue_copy_operation(ctx, init_copylauncher);

  // daxpy
  // test IndexLauncher
  const double alpha = 0.1;
  IndexLauncher daxpy_launcher(DAXPY_TASK_ID, color_is,
                TaskArgument(&alpha, sizeof(alpha)), arg_map);
  std::string daxpy_launcher_prov = "Daxpy IndexLauncher:" + std::to_string(__LINE__);
  daxpy_launcher.provenance = daxpy_launcher_prov;
  daxpy_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, input_lr));
  daxpy_launcher.region_requirements[0].add_field(FID_X);
  daxpy_launcher.region_requirements[0].add_field(FID_Y);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, output_lr));
  daxpy_launcher.region_requirements[1].add_field(FID_Z);
  FutureMap fm = runtime->execute_index_space(ctx, daxpy_launcher);
  //fm.wait_all_results();

  // check
  // test TaskLauncher
  TaskLauncher check_launcher(CHECK_TASK_ID, TaskArgument(&alpha, sizeof(alpha)));
  std::string check_launcher_prov = "CheckLauncher:" + std::to_string(__LINE__);
  check_launcher.provenance = check_launcher_prov;
  check_launcher.add_region_requirement(
      RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  check_launcher.region_requirements[0].add_field(FID_X);
  check_launcher.region_requirements[0].add_field(FID_Y);
  check_launcher.add_region_requirement(
      RegionRequirement(output_lr, READ_ONLY, EXCLUSIVE, output_lr));
  check_launcher.region_requirements[1].add_field(FID_Z);
  Future fu = runtime->execute_task(ctx, check_launcher);
  fu.wait();

  std::string detach_xy_prov = "Detach XY:" + std::to_string(__LINE__);
  runtime->detach_external_resource(ctx, xy_pr, true, false, detach_xy_prov.c_str());
  std::string detach_z_prov = "Detach Z:" + std::to_string(__LINE__);
  runtime->detach_external_resource(ctx, z_pr, true, false, detach_z_prov.c_str());
  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
  runtime->destroy_index_space(ctx, color_is);
  if (xyz_ptr == NULL) free(xyz_ptr);
  if (xy_ptr == NULL) free(xy_ptr);
  if (z_ptr == NULL) free(z_ptr);
}

void daxpy_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(double));
  const double alpha = *((const double*)task->args);
  const int point = task->index_point.point_data[0];

  const AccessorRO acc_y(regions[0], FID_Y);
  const AccessorRO acc_x(regions[0], FID_X);
  const AccessorWD acc_z(regions[1], FID_Z);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  printf("Running daxpy computation with alpha %.8g for point %d, xptr %p, y_ptr %p, z_ptr %p...\n", 
          alpha, point, 
          acc_x.ptr(rect.lo), acc_y.ptr(rect.lo), acc_z.ptr(rect.lo));
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    acc_z[*pir] = alpha * acc_x[*pir] + acc_y[*pir];
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(double));

  const AccessorRO acc_x(regions[0], FID_X);
  const AccessorRO acc_y(regions[0], FID_Y);
  const AccessorRO acc_z(regions[1], FID_Z);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  const void *ptr = acc_z.ptr(rect.lo);
  printf("Checking results... xptr %p, y_ptr %p, z_ptr %p...\n", 
          acc_x.ptr(rect.lo), acc_y.ptr(rect.lo), ptr);
  bool all_passed = true;
  double expected = 0.55;
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
  {
    double received = acc_z[*pir];
    // Probably shouldn't check for floating point equivalence but
    // the order of operations are the same should they should
    // be bitwise equal.
    if (!compare_double(expected, received)) {
      all_passed = false;
      printf("expected %f, received %f\n", expected, received);
    }
  }
  if (all_passed)
    printf("SUCCESS!\n");
  else {
    printf("FAILURE!\n");
    abort();
  }

  Rect<1> color_bounds(0,3);
  Domain color_domain = Domain(color_bounds);
  Future color_future = Future::from_value<Domain>(color_domain); 
  IndexSpace color_is = runtime->create_index_space(ctx, 1, color_future, 0, "GPU Color IndexSpace");
  ArgumentMap arg_map;
  IndexLauncher gpu_launcher(GPU_TASK_ID, color_is, TaskArgument(NULL, 0), arg_map);
  std::string gpu_launcher_prov = "GPU IndexLauncher:" + std::to_string(__LINE__);
  gpu_launcher.provenance = gpu_launcher_prov;
  runtime->execute_index_space(ctx, gpu_launcher);
  runtime->destroy_index_space(ctx, color_is);
}

void gpu_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  usleep(10*1000);
  printf("GPU task done\n");
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
    TaskVariantRegistrar registrar(DAXPY_TASK_ID, "daxpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<daxpy_task>(registrar, "daxpy");
  }

  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  {
    TaskVariantRegistrar registrar(GPU_TASK_ID, "gpu");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<gpu_task>(registrar, "gpu");
  }

  return Runtime::start(argc, argv);
}
