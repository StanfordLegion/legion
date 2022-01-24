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
#define LEGION_ENABLE_C_BINDINGS
#include "legion.h"
#undef LEGION_ENABLE_C_BINDINGS
using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_ID,
  DAXPY_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y,
  FID_Z,
};

#ifdef __cplusplus
extern "C" {
void daxpy_task_f(legion_task_t task, const legion_physical_region_t *regionptr, int num_regions, legion_context_t ctx, legion_runtime_t runtime);
void daxpy_task_f2(const void *data, size_t datalen, const void *userdata, size_t userlen, realm_id_t p);
}
#endif

//#define CTASK

void
legion_task_cxx_to_c(
  const Task *task,
  const std::vector<PhysicalRegion> *regions,
  Context ctx, 
  Runtime *runtime,
  legion_task_t *taskptr,
  const legion_physical_region_t **regionptr,
  unsigned * num_regions_ptr,
  legion_context_t * ctxptr,
  legion_runtime_t * runtimeptr);

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
  IndexSpace is = runtime->create_index_space(ctx, elem_rect); 
  runtime->attach_name(is, "is");
  FieldSpace input_fs = runtime->create_field_space(ctx);
  runtime->attach_name(input_fs, "input_fs");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, input_fs);
    allocator.allocate_field(sizeof(double),FID_X);
    runtime->attach_name(input_fs, FID_X, "X");
    allocator.allocate_field(sizeof(double),FID_Y);
    runtime->attach_name(input_fs, FID_Y, "Y");
  }
  FieldSpace output_fs = runtime->create_field_space(ctx);
  runtime->attach_name(output_fs, "output_fs");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, output_fs);
    allocator.allocate_field(sizeof(double),FID_Z);
    runtime->attach_name(output_fs, FID_Z, "Z");
  }
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);
  runtime->attach_name(input_lr, "input_lr");
  LogicalRegion output_lr = runtime->create_logical_region(ctx, is, output_fs);
  runtime->attach_name(output_lr, "output_lr");

  // In addition to using rectangles and domains for launching index spaces
  // of tasks (see example 02), Legion also uses them for performing 
  // operations on logical regions.  Here we create a rectangle and a
  // corresponding domain for describing the space of subregions that we
  // want to create.  Each subregion is assigned a 'color' which is why
  // we name the variables 'color_bounds' and 'color_domain'.  We'll use
  // these below when we partition the region.
  Rect<1> color_bounds(0,num_subregions-1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);

  // Parallelism in Legion is implicit.  This means that rather than
  // explicitly saying what should run in parallel, Legion applications
  // partition up data and tasks specify which regions they access.
  // The Legion runtime computes non-interference as a function of 
  // regions, fields, and privileges and then determines which tasks 
  // are safe to run in parallel.
  //
  // Data partitioning is performed on index spaces.  The partitioning 
  // operation is used to break an index space of points into subsets 
  // of points each of which will become a sub index space.  Partitions 
  // created on an index space are then transitively applied to all the 
  // logical regions created using the index space.  We will show how
  // to get names to the subregions later in this example.
  //
  // Here we want to create the IndexPartition 'ip'.  We'll illustrate
  // two ways of creating an index partition depending on whether the
  // array being partitioned can be evenly partitioned into subsets
  // or not.  There are other methods to partitioning index spaces
  // which are not covered here.  We'll cover the case of coloring
  // individual points in an index space in our capstone circuit example.
  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
  runtime->attach_name(ip, "ip");

  // The index space 'is' was used in creating two logical regions: 'input_lr'
  // and 'output_lr'.  By creating an IndexPartitiong of 'is' we implicitly
  // created a LogicalPartition for each of the logical regions created using
  // 'is'.  The Legion runtime provides several ways of getting the names for
  // these LogicalPartitions.  We'll look at one of them here.  The
  // 'get_logical_partition' method takes a LogicalRegion and an IndexPartition
  // and returns the LogicalPartition of the given LogicalRegion that corresponds
  // to the given IndexPartition.  
  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
  runtime->attach_name(input_lp, "input_lp");
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, ip);
  runtime->attach_name(output_lp, "output_lp");

  // Create our launch domain.  Note that is the same as color domain
  // as we are going to launch one task for each subregion we created.
  ArgumentMap arg_map;

  // As in previous examples, we now want to launch tasks for initializing 
  // both the fields.  However, to increase the amount of parallelism
  // exposed to the runtime we will launch separate sub-tasks for each of
  // the logical subregions created by our partitioning.  To express this
  // we create an IndexLauncher for launching an index space of tasks
  // the same as example 02.
  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is, 
                              TaskArgument(NULL, 0), arg_map);
  // For index space task launches we don't want to have to explicitly
  // enumerate separate region requirements for all points in our launch
  // domain.  Instead Legion allows applications to place an upper bound
  // on privileges required by subtasks and then specify which privileges
  // each subtask receives using a projection function.  In the case of
  // the field initialization task, we say that all the subtasks will be
  // using some subregion of the LogicalPartition 'input_lp'.  Applications
  // may also specify upper bounds using logical regions and not partitions.
  //
  // The Legion implementation assumes that all all points in an index
  // space task launch request non-interfering privileges and for performance
  // reasons this is unchecked.  This means if two tasks in the same index
  // space are accessing aliased data, then they must either both be
  // with read-only or reduce privileges.
  //
  // When the runtime enumerates the launch_domain, it will invoke the
  // projection function for each point in the space and use the resulting
  // LogicalRegion computed for each point in the index space of tasks.
  // The projection ID '0' is reserved and corresponds to the identity 
  // function which simply zips the space of tasks with the space of
  // subregions in the partition.  Applications can register their own
  // projections functions via the 'register_region_projection' and
  // 'register_partition_projection' functions before starting 
  // the runtime similar to how tasks are registered.
  init_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection ID*/, 
                        WRITE_DISCARD, EXCLUSIVE, input_lr));
  init_launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_index_space(ctx, init_launcher);

  // Modify our region requirement to initialize the other field
  // in the same way.  Note that after we do this we have exposed
  // 2*num_subregions task-level parallelism to the runtime because
  // we have launched tasks that are both data-parallel on
  // sub-regions and task-parallel on accessing different fields.
  // The power of Legion is that it allows programmers to express
  // these data usage patterns and automatically extracts both
  // kinds of parallelism in a unified programming framework.
  init_launcher.region_requirements[0].privilege_fields.clear();
  init_launcher.region_requirements[0].instance_fields.clear();
  init_launcher.region_requirements[0].add_field(FID_Y);
  runtime->execute_index_space(ctx, init_launcher);

  const double alpha = 1.0;
  // We launch the subtasks for performing the daxpy computation
  // in a similar way to the initialize field tasks.  Note we
  // again make use of two RegionRequirements which use a
  // partition as the upper bound for the privileges for the task.
  IndexLauncher daxpy_launcher(DAXPY_TASK_ID, color_is,
                TaskArgument(&alpha, sizeof(alpha)), arg_map);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, input_lr));
  daxpy_launcher.region_requirements[0].add_field(FID_X);
  daxpy_launcher.region_requirements[0].add_field(FID_Y);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, output_lr));
  daxpy_launcher.region_requirements[1].add_field(FID_Z);
  runtime->execute_index_space(ctx, daxpy_launcher);
                    
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
      RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  check_launcher.region_requirements[0].add_field(FID_X);
  check_launcher.region_requirements[0].add_field(FID_Y);
  check_launcher.add_region_requirement(
      RegionRequirement(output_lr, READ_ONLY, EXCLUSIVE, output_lr));
  check_launcher.region_requirements[1].add_field(FID_Z);
  runtime->execute_task(ctx, check_launcher);

  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, input_fs);
  runtime->destroy_field_space(ctx, output_fs);
  runtime->destroy_index_space(ctx, is);
}

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
  // Note here that we get the domain for the subregion for
  // this task from the runtime which makes it safe for running
  // both as a single task and as part of an index space of tasks.
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    acc[*pir] = drand48();
}


#ifdef CTASK
void daxpy_task(const void *data, size_t datalen, const void *userdata, size_t userlen, realm_id_t p)
{
    printf("daxpy task C\n");
    daxpy_task_f2(data, datalen, userdata, userlen, p);
}
#else
void daxpy_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  legion_task_t taskptr;
  const legion_physical_region_t *regionptr;
  unsigned num_regions;
  legion_context_t ctxptr;
  legion_runtime_t runtimeptr;
  legion_task_cxx_to_c(task, &regions, ctx, runtime,
    &taskptr, &regionptr, &num_regions, &ctxptr, &runtimeptr);
  
  printf("daxpy task C++ %d\n", num_regions);
  daxpy_task_f(taskptr, regionptr, num_regions, ctxptr, runtimeptr);
}
#endif

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(double));
  const double alpha = *((const double*)task->args);

  const FieldAccessor<READ_ONLY,double,1> acc_x(regions[0], FID_X);
  const FieldAccessor<READ_ONLY,double,1> acc_y(regions[0], FID_Y);
  const FieldAccessor<READ_ONLY,double,1> acc_z(regions[1], FID_Z);

  printf("Checking results...");
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  bool all_passed = true;
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
  {
    double expected = alpha * acc_x[*pir] + acc_y[*pir];
    double received = acc_z[*pir];
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
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INIT_FIELD_TASK_ID, "init_field");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_field_task>(registrar, "init_field");
  }
#ifdef CTASK  
  legion_execution_constraint_set_t execution_constraints = legion_execution_constraint_set_create();
  legion_execution_constraint_set_add_processor_constraint(execution_constraints, LOC_PROC);
  legion_task_layout_constraint_set_t layout_constraints = legion_task_layout_constraint_set_create();
  legion_task_config_options_t config_options = {.leaf = false, .inner = false, .idempotent = false};
  legion_runtime_preregister_task_variant_fnptr(DAXPY_TASK_ID, AUTO_GENERATE_ID,
                                                "daxpy",
                                                "cpu_variant",
                                                execution_constraints,
                                                layout_constraints,
                                                config_options,
                                                daxpy_task,
                                                NULL,
                                                0);
#else
  {
    TaskVariantRegistrar registrar(DAXPY_TASK_ID, "daxpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<daxpy_task>(registrar, "daxpy");
  }
#endif
  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  return Runtime::start(argc, argv);
}
