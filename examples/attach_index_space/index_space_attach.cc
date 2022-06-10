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
#include "legion.h"
using namespace Legion;

typedef FieldAccessor<READ_ONLY,int,1,coord_t,
                      Realm::AffineAccessor<int,1,coord_t> > AccessorRO;
typedef FieldAccessor<WRITE_DISCARD,int,1,coord_t,
                      Realm::AffineAccessor<int,1,coord_t> > AccessorWD;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  AXPY_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y,
  FID_Z,
};

double get_cur_time() {
  struct timeval   tv;
  struct timezone  tz;
  double cur_time;

  gettimeofday(&tv, &tz);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;

  return cur_time;
}


void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_elements = 1024; 
  int num_subregions = 4;
  int soa_flag = 0;
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
      if (!strcmp(command_args.argv[i],"-s"))
        soa_flag = atoi(command_args.argv[++i]);
    }
  }
  printf("Running axpy for %d elements...\n", num_elements);
  printf("Partitioning data into %d sub-regions...\n", num_subregions);

  // Create our logical regions using the same schemas as earlier examples
  Rect<1> elem_rect(0,num_elements-1);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect); 
  runtime->attach_name(is, "is");
  FieldSpace fs = runtime->create_field_space(ctx);
  runtime->attach_name(fs, "fs");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(int),FID_X);
    runtime->attach_name(fs, FID_X, "X");
    allocator.allocate_field(sizeof(int),FID_Y);
    runtime->attach_name(fs, FID_Y, "Y");
    allocator.allocate_field(sizeof(int),FID_Z);
    runtime->attach_name(fs, FID_Z, "Z");
  }
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, fs);
  runtime->attach_name(input_lr, "input_lr");
  LogicalRegion output_lr = runtime->create_logical_region(ctx, is, fs);
  runtime->attach_name(output_lr, "output_lr");

  Rect<1> color_bounds(0,num_subregions-1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);

  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
  runtime->attach_name(ip, "ip");

  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
  runtime->attach_name(input_lp, "input_lp");
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, ip);
  runtime->attach_name(output_lp, "output_lp");
  
  std::vector<int*>   z_ptrs;
  std::vector<int*>  xy_ptrs;
  std::vector<int*> xyz_ptrs;

  IndexAttachLauncher xy_launcher(LEGION_EXTERNAL_INSTANCE, input_lr, false/*restricted*/);
  IndexAttachLauncher z_launcher(LEGION_EXTERNAL_INSTANCE, output_lr, false/*restricted*/); 
  int offset = 0;
  const ShardID local_shard = task->get_shard_id();
  const size_t total_shards = task->get_total_shards();
  for (int i = 0; i < num_subregions; ++i) {
    const DomainPoint point = Point<1>(i);
    IndexSpace child_space = runtime->get_index_subspace(ctx, ip, point);
    const Rect<1> bounds = runtime->get_index_space_domain(ctx, child_space);
    const size_t child_elements = bounds.volume();
    // Handle control replication here, index space attach operations are collective
    // meaning that each shard should pass in a subset of the pointers for different
    // subregions in a way that all shards cover all the subregions
    // We'll do this with the simple load balancing technique of round-robin mapping 
    if ((i % total_shards) != local_shard) {
      offset += child_elements;
      // still need to add the privilege fields
      xy_launcher.privilege_fields.insert(FID_X);
      xy_launcher.privilege_fields.insert(FID_Y);
      z_launcher.privilege_fields.insert(FID_Z);
      continue;
    }
    LogicalRegion input_handle = 
          runtime->get_logical_subregion_by_tree(ctx, child_space, fs, input_lr.get_tree_id());
    LogicalRegion output_handle = 
          runtime->get_logical_subregion_by_tree(ctx, child_space, fs, output_lr.get_tree_id());
    if (soa_flag) 
    { // SOA
      int *xy_ptr = (int*)malloc(2*sizeof(int)*(child_elements));
      int *z_ptr = (int*)malloc(sizeof(int)*(child_elements));
      for (unsigned j = 0; j < child_elements; j++ ) {
          xy_ptr[j]                 = offset+j;     // x
          xy_ptr[child_elements+j]  = 3*(offset+j); // y
          z_ptr[j]                  = 0;            // z
      }
      {
        std::vector<FieldID> attach_fields(2);
        attach_fields[0] = FID_X;
        attach_fields[1] = FID_Y;
        xy_launcher.attach_array_soa(input_handle, xy_ptr, false/*column major*/,
                                     attach_fields);
      }
      xy_ptrs.push_back(xy_ptr);
      { 
        std::vector<FieldID> attach_fields(1);
        attach_fields[0] = FID_Z;
        z_launcher.attach_array_soa(output_handle, z_ptr, false/*column major*/,
                                    attach_fields);
      }
      z_ptrs.push_back(z_ptr);
    } 
    else 
    { // AOS
      int *xyz_ptr = (int*)malloc(3*sizeof(int)*child_elements);
      for (unsigned j = 0; j < child_elements; j++) {
        xyz_ptr[3*j]   = offset+j;      // x
        xyz_ptr[3*j+1] = 3*(offset+j);  // y
        xyz_ptr[3*j+2] = 0;             // z
      }
      std::vector<FieldID> layout_constraint_fields(3);
      layout_constraint_fields[0] = FID_X;
      layout_constraint_fields[1] = FID_Y;
      layout_constraint_fields[2] = FID_Z;
      {
        xy_launcher.attach_array_aos(input_handle, xyz_ptr, false/*column major*/,
                                     layout_constraint_fields);
      }
      {
        z_launcher.attach_array_aos(output_handle, xyz_ptr, false/*column major*/,
                                    layout_constraint_fields);
      }
      xyz_ptrs.push_back(xyz_ptr);
    }
    offset += child_elements;
  }
  // remove unnecessary privilege fields from the launchers for the aos case
  if (!soa_flag) {
    xy_launcher.privilege_fields.erase(FID_Z);
    z_launcher.privilege_fields.erase(FID_X);
    z_launcher.privilege_fields.erase(FID_Y);
  }
  // Now we can do the attach operations
  ExternalResources xy_resources = runtime->attach_external_resources(ctx, xy_launcher);
  ExternalResources z_resources = runtime->attach_external_resources(ctx, z_launcher);

  ArgumentMap arg_map;

  const int alpha = 5;
  // We launch the subtasks for performing the axpy computation
  // in a similar way to the initialize field tasks.  Note we
  // again make use of two RegionRequirements which use a
  // partition as the upper bound for the privileges for the task.
  IndexLauncher axpy_launcher(AXPY_TASK_ID, color_is,
                TaskArgument(&alpha, sizeof(alpha)), arg_map);
  axpy_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, input_lr));
  axpy_launcher.region_requirements[0].add_field(FID_X);
  axpy_launcher.region_requirements[0].add_field(FID_Y);
  axpy_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, output_lr));
  axpy_launcher.region_requirements[1].add_field(FID_Z);
  runtime->execute_index_space(ctx, axpy_launcher);
                    
  // While we could also issue parallel subtasks for the checking
  // task, we only issue a single task launch to illustrate an
  // important Legion concept.  Note the checking task operates
  // on the entire 'input_lr' and 'output_lr' regions and not
  // on the subregions.  Even though the previous tasks were
  // all operating on subregions, Legion will correctly compute
  // data dependences on all the subtasks that generated the
  // data in these two regions.  
  TaskLauncher check_launcher(CHECK_TASK_ID, TaskArgument(&alpha, sizeof(alpha)));
  check_launcher.add_region_requirement(
      RegionRequirement(output_lr, READ_ONLY, EXCLUSIVE, output_lr));
  check_launcher.region_requirements[0].add_field(FID_Z);
  runtime->execute_task(ctx, check_launcher);

  Future f1 = runtime->detach_external_resources(ctx, xy_resources);
  Future f2 = runtime->detach_external_resources(ctx, z_resources);
  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
  f1.wait();
  f2.wait();
  for (unsigned idx = 0; idx < xyz_ptrs.size(); idx++)
    free(xyz_ptrs[idx]);
  for (unsigned idx = 0; idx < xy_ptrs.size(); idx++)
    free(xy_ptrs[idx]);
  for (unsigned idx = 0; idx < z_ptrs.size(); idx++)
    free(z_ptrs[idx]);
}

void axpy_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(int));
  const int alpha = *((const int*)task->args);
  const int point = task->index_point.point_data[0];

  const AccessorRO acc_y(regions[0], FID_Y);
  const AccessorRO acc_x(regions[0], FID_X);
  const AccessorWD acc_z(regions[1], FID_Z);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  printf("Running axpy computation with alpha %d for point %d, xptr %p, y_ptr %p, z_ptr %p...\n", 
          alpha, point, 
          acc_x.ptr(rect.lo), acc_y.ptr(rect.lo), acc_z.ptr(rect.lo));
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    acc_z[*pir] = alpha * acc_x[*pir] + acc_y[*pir];
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(task->arglen == sizeof(int));
  const int alpha = *((const int*)task->args);

  const AccessorRO acc_z(regions[0], FID_Z);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  const void *ptr = acc_z.ptr(rect.lo);
  printf("Checking results... z_ptr %p...\n", ptr);
  bool all_passed = true;
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
  {
    int expected = (alpha + 3) * pir[0];
    int received = acc_z[*pir];
    // Probably shouldn't check for floating point equivalence but
    // the order of operations are the same should they should
    // be bitwise equal.
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
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(AXPY_TASK_ID, "axpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<axpy_task>(registrar, "axpy");
  }

  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  return Runtime::start(argc, argv);
}
