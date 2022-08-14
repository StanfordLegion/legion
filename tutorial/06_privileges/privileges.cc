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
  INIT_FIELD_TASK_ID,
  DAXPY_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y,
  FID_Z,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_elements = 1024; 
  // See if we have any command line arguments to parse
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
    }
  }
  printf("Running daxpy for %d elements...\n", num_elements);

  // Create our logical regions using the same schema that
  // we used in the previous example.
  const Rect<1> elem_rect(0,num_elements-1);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect); 
  FieldSpace input_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, input_fs);
    allocator.allocate_field(sizeof(double),FID_X);
    allocator.allocate_field(sizeof(double),FID_Y);
  }
  FieldSpace output_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, output_fs);
    allocator.allocate_field(sizeof(double),FID_Z);
  }
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, is, output_fs);

  // Instead of using an inline mapping to initialize the fields for
  // daxpy, in this case we will launch two separate tasks for initializing
  // each of the fields in parallel.  To launch the sub-tasks for performing
  // the initialization we again use the launcher objects that were
  // introduced earlier.  The only difference now is that instead of passing
  // arguments by value, we now want to specify the logical regions
  // that the tasks may access as their arguments.  We again make use of
  // the RegionRequirement struct to name the logical regions and fields
  // for which the task should have privileges.  In this case we launch
  // a task that asks for WRITE_DISCARD privileges on the 'X' field.
  //
  // An important property of the Legion programming model is that sub-tasks
  // are only allowed to request privileges which are a subset of a 
  // parent task's privileges.  When a task creates a logical region it
  // is granted full read-write privileges for that logical region.  It
  // can then pass them down to sub-tasks.  In this example the top-level
  // task has full privileges on all the fields of input_lr and output_lr.
  // In this call it passing read-write privileges down to the sub-task
  // on input_lr on field 'X'.  Legion will enforce the property that the 
  // sub-task only accesses the 'X' field of input_lr.  This property of
  // Legion is crucial for the implementation of Legion's hierarchical
  // scheduling algorithm which is described in detail in our two papers.
  TaskLauncher init_launcher(INIT_FIELD_TASK_ID, TaskArgument(NULL, 0));
  init_launcher.add_region_requirement(
      RegionRequirement(input_lr, WRITE_DISCARD, EXCLUSIVE, input_lr));
  init_launcher.add_field(0/*idx*/, FID_X);
  // Note that when we launch this task we don't record the future.
  // This is because we're going to let Legion be responsible for 
  // computing the data dependences between how different tasks access
  // logical regions.
  runtime->execute_task(ctx, init_launcher);

  // Re-use the same launcher but with a slightly different RegionRequirement
  // that requests privileges on field 'Y' instead of 'X'.  Since these
  // two instances of the init_field_task are accessing different fields
  // of the input_lr region, they can be run in parallel (whether or not
  // they do is dependent on the mapping discussed in a later example).
  // Legion automatically will discover this parallelism since the runtime
  // understands the fields present on the logical region.
  //
  // We now call attention to a unique property of the init_field_task.
  // In this example we've actually called the task with two different
  // region requirements containing different fields.  The init_field_task
  // is an example of a field-polymorphic task which is capable of 
  // performing the same operation on different fields of a logical region.
  // In practice this is very useful property for a task to maintain as
  // it allows one implementation of a task to be written which is capable
  // of being used in many places.
  init_launcher.region_requirements[0].privilege_fields.clear();
  init_launcher.region_requirements[0].instance_fields.clear();
  init_launcher.add_field(0/*idx*/, FID_Y);

  runtime->execute_task(ctx, init_launcher);

  // Now we launch the task to perform the daxpy computation.  We pass
  // in the alpha value as an argument.  All the rest of the arguments
  // are RegionRequirements specifying that we are reading the two
  // fields on the input_lr region and writing results to the output_lr
  // region.  Legion will automatically compute data dependences
  // from the two init_field_tasks and will ensure that the program
  // order execution is obeyed.
  const double alpha = drand48();
  TaskLauncher daxpy_launcher(DAXPY_TASK_ID, TaskArgument(&alpha, sizeof(alpha)));
  daxpy_launcher.add_region_requirement(
      RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  daxpy_launcher.add_field(0/*idx*/, FID_X);
  daxpy_launcher.add_field(0/*idx*/, FID_Y);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(output_lr, WRITE_DISCARD, EXCLUSIVE, output_lr));
  daxpy_launcher.add_field(1/*idx*/, FID_Z);

  runtime->execute_task(ctx, daxpy_launcher);

  // Finally we launch a task to perform the check on the output.  Note
  // that Legion will compute a data dependence on the first RegionRequirement
  // with the two init_field_tasks, but not on daxpy task since they 
  // both request read-only privileges on the 'X' and 'Y' fields.  However,
  // Legion will compute a data dependence on the second region requirement
  // as the daxpy task was writing the 'Z' field on output_lr and this task
  // is reading the 'Z' field of the output_lr region.
  TaskLauncher check_launcher(CHECK_TASK_ID, TaskArgument(&alpha, sizeof(alpha)));
  check_launcher.add_region_requirement(
      RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  check_launcher.add_field(0/*idx*/, FID_X);
  check_launcher.add_field(0/*idx*/, FID_Y);
  check_launcher.add_region_requirement(
      RegionRequirement(output_lr, READ_ONLY, EXCLUSIVE, output_lr));
  check_launcher.add_field(1/*idx*/, FID_Z);

  runtime->execute_task(ctx, check_launcher);

  // Notice that we never once blocked waiting on the result of any sub-task
  // in the execution of the top-level task.  We don't even block before
  // destroying any of our resources.  This works because Legion understands
  // the data being accessed by all of these operations and defers all of
  // their executions until they are safe to perform.  Legion is still smart
  // enough to know that the top-level task is not finished until all of
  // the sub operations that have been performed are completed.  However,
  // from the programmer's perspective, all of these operations can be
  // done without ever blocking and thereby exposing as much task-level
  // parallelism to the Legion runtime as possible.  We'll discuss the
  // implications of Legion's deferred execution model in a later example.
  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, input_fs);
  runtime->destroy_field_space(ctx, output_fs);
  runtime->destroy_index_space(ctx, is);
}

// Note that tasks get a physical region for every region requirement
// that they requested when they were launched in the vector of 'regions'.
// In some cases the mapper may have chosen not to map the logical region
// which means that the task has the necessary privileges to access the
// region but not a physical instance to access.
void init_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  // Check that the inputs look right since we have no
  // static checking to help us out.
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);
  // This is a field polymorphic function so figure out
  // which field we are responsible for initializing.
  FieldID fid = *(task->regions[0].privilege_fields.begin());
  printf("Initializing field %d...\n", fid);
  // Note that Legion's default mapper always map regions
  // and the Legion runtime is smart enough not to start
  // the task until all the regions contain valid data.  
  // Therefore in this case we don't need to call 'wait_until_valid'
  // on our physical regions and we know that getting this
  // accessor will never block the task's execution.  If
  // however we chose to unmap this physical region and then
  // remap it then we would need to call 'wait_until_valid'
  // again to ensure that we were accessing valid data.
  const FieldAccessor<WRITE_DISCARD,double,1> acc(regions[0], fid);

  Rect<1> rect = runtime->get_index_space_domain(ctx, 
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    acc[*pir] = drand48();
}

void daxpy_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(double));
  const double alpha = *((const double*)task->args);

  const FieldAccessor<READ_ONLY,double,1> acc_x(regions[0], FID_X);
  const FieldAccessor<READ_ONLY,double,1> acc_y(regions[0], FID_Y);
  const FieldAccessor<WRITE_DISCARD,double,1> acc_z(regions[1], FID_Z);

  printf("Running daxpy computation with alpha %.8g...\n", alpha);
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
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
    if (expected != received) {
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
    TaskVariantRegistrar registrar(DAXPY_TASK_ID, "daxpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<daxpy_task>(registrar, "daxpy");
  }

  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  return Runtime::start(argc, argv);
}
