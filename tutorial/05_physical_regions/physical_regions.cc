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
 * In this section we use a sequential
 * implementation of daxpy to show how
 * to create physical instances of logical
 * regions.  In later sections we will
 * show how to extend this daxpy example
 * so that it will run with sub-tasks
 * and also run in parallel.
 */

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
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

  // We'll create two logical regions with a common index space
  // for storing our inputs and outputs.  The input region will
  // have two fields for storing the 'x' and 'y' fields of the
  // daxpy computation, and the output region will have a single
  // field 'z' for storing the result.
  Rect<1> elem_rect(0,num_elements-1);
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

  // Use fill_field to set an initial value for the fields in the input
  // logical region.
  runtime->fill_field<double>(ctx, input_lr, input_lr, FID_X, 0.0);
  runtime->fill_field<double>(ctx, input_lr, input_lr, FID_Y, 0.0);

  // Now that we have our logical regions we want to instantiate physical
  // instances of these regions which we can use for storing data.  One way
  // of creating a physical instance of a logical region is via an inline
  // mapping operation.  (We'll discuss the other way of creating physical
  // instances in the next example.)  Inline mappings map a physical instance
  // of logical region inside of this task's context.  This will give the
  // task an up-to-date copy of the data stored in these logical regions.
  // In this particular daxpy example, the data has yet to be initialized so
  // really this just creates un-initialized physical regions for the 
  // application to use.
  //
  // To perform an inline mapping we use an InlineLauncher object which is
  // similar to other launcher objects shown earlier.  The argument passed
  // to this launcher is a 'RegionRequirement' which is used to specify which
  // logical region we should be mapping as well as with what privileges
  // and coherence.  In this example we are mapping the input_lr logical
  // region with READ-WRITE privilege and EXCLUSIVE coherence.  We'll see
  // examples of other privileges in later examples.  If you're interested
  // in learning about relaxed coherence modes we refer you to our OOPSLA paper.
  // The last argument in the RegionRequirement is the logical region for 
  // which the enclosing task has privileges which in this case is the 
  // same input_lr logical region.  We'll discuss restrictions on privileges
  // more in the next example.
  RegionRequirement req(input_lr, READ_WRITE, EXCLUSIVE, input_lr);
  // We also need to specify which fields we plan to access in our
  // RegionRequirement.  To do this we invoke the 'add_field' method
  // on the RegionRequirement.
  req.add_field(FID_X);
  req.add_field(FID_Y);
  InlineLauncher input_launcher(req);

  // Once we have set up our launcher, we as the runtime to map a physical
  // region instance of our requested logical region with the given 
  // privileges and coherence.  This returns a PhysicalRegion object
  // which is handle to the physical instance which contains data
  // for the logical region.  In keeping with Legion's deferred execution
  // model the 'map_region' call is asynchronous.  This allows the
  // application to issue many of these operations in flight and to
  // perform other useful work while waiting for the region to be ready.
  //
  // One common criticism about Legion applications is that there exists
  // a dichotomy between logical and physical regions.  Programmers
  // are explicitly required to keep track of both kinds of regions and
  // know when and how to use them.  If you feel this way as well, we
  // encourage you to try out our Legion compiler in which this
  // dichotomy does not exist.  There are simply regions and the compiler
  // automatically manages the logical and physical nature of them
  // in a way that is analogous to how compilers manage the mapping 
  // between variables and architectural registers.  This runtime API
  // is designed to be expressive for all Legion programs and is not
  // necessarily designed for programmer productivity.
  PhysicalRegion input_region = runtime->map_region(ctx, input_launcher);
  // The application can either poll a physical region to see when it
  // contains valid data for the logical region using the 'is_valid'
  // method or it can explicitly wait using the 'wait_until_valid' 
  // method.  Just like waiting on a future, if the region is not ready
  // this task is pre-empted and other tasks may run while waiting
  // for the region to be ready.  Note that an application does not
  // need to explicitly wait for the physical region to be ready before
  // using it, but any call to get an accessor (described next) on
  // physical region that does not yet have valid data will implicitly
  // call 'wait_until_valid' to guarantee correct execution by ensuring
  // the application only can access the physical instance once the
  // data is valid.
  input_region.wait_until_valid();

  // To actually access data within a physical region, an application
  // must create a RegionAccessor.  RegionAccessors provide a level
  // of indirection between a physical instance and the application 
  // which is necessary for supporting general task code that is
  // independent of data layout.  RegionAccessors are templated on
  // the kind of accessor that they are and the type of element they
  // are accessing.  Here we illustrate only Generic accessors.  
  // Generic accessors are the simplest accessors to use and have
  // the ability to dynamically verify that all accesses they perform
  // are correct.  However, they are also very slow.  Therefore
  // they should NEVER be used in production code.  In general, we
  // encourage application developers to write two versions of any
  // function: one using Generic accessors that can be used to check
  // correctness, and then a faster version using higher performance
  // accessors (discussed in a later example).
  //
  // Note that each accessor must specify which field it is 
  // accessing and then check that the types match with the
  // field that is being accessed.  This provides a combination
  // of dynamic and static checking which ensures that the 
  // correct data is being accessed.
  const FieldAccessor<READ_WRITE,double,1> acc_x(input_region, FID_X);
  const FieldAccessor<READ_WRITE,double,1> acc_y(input_region, FID_Y);

  // We initialize our regions with some random data.  To iterate
  // over all the points in each of the regions we use an iterator
  // which can be used to enumerate all the points within an array.
  // The points in the array (the 'p' field in the iterator) are
  // used to access different locations in each of the physical
  // instances.
  for (PointInRectIterator<1> pir(elem_rect); pir(); pir++)
  {
    acc_x[*pir] = drand48();
    acc_y[*pir] = drand48();
  }

  // Now we map our output region so we can do the actual computation.
  // We use another inline launcher with a different RegionRequirement
  // that specifies our privilege to be WRITE-DISCARD.  WRITE-DISCARD
  // says that we can discard any data presently residing the region
  // because we are going to overwrite it.
  InlineLauncher output_launcher(RegionRequirement(output_lr, WRITE_DISCARD,
                                                   EXCLUSIVE, output_lr));
  output_launcher.requirement.add_field(FID_Z);

  // Map the region
  PhysicalRegion output_region = runtime->map_region(ctx, output_launcher);

  // Note that this accessor invokes the implicit 'wait_until_valid'
  // call described earlier.
  const double alpha = drand48();
  {
    const FieldAccessor<WRITE_DISCARD,double,1> acc_z(output_region, FID_Z);

    printf("Running daxpy computation with alpha %.8g...", alpha);
    // Iterate over our points and perform the daxpy computation.  Note
    // we can use the same iterator because both the input and output
    // regions were created using the same index space.
    for (PointInRectIterator<1> pir(elem_rect); pir(); pir++)
      acc_z[*pir] = alpha * acc_x[*pir] + acc_y[*pir];
  }
  printf("Done!\n");

  // In some cases it may be necessary to unmap regions and then 
  // remap them.  We'll give a compelling example of this in the
  // next example.   In this case we'll remap the output region
  // with READ-ONLY privileges to check the output result.
  // We really could have done this directly since WRITE-DISCARD
  // privileges are equivalent to READ-WRITE privileges in terms
  // of allowing reads and writes, but we'll explicitly unmap
  // and then remap.  Unmapping is done with the unmap call.
  // After this call the physical region no longer contains valid
  // data and all accessors from the physical region are invalidated.
  runtime->unmap_region(ctx, output_region);

  // We can then remap the region.  Note if we wanted to remap
  // with the same privileges we could have used the 'remap_region'
  // call.  However, we want different privileges so we update
  // the launcher and then remap the region.  The 'remap_region' 
  // call also guarantees that we would get the same physical 
  // instance.  By calling 'map_region' again, we have no such 
  // guarantee.  We may get the same physical instance or a new 
  // one.  The orthogonality of correctness from mapping decisions
  // ensures that we will access the same data regardless.
  output_launcher.requirement.privilege = READ_ONLY;
  output_region = runtime->map_region(ctx, output_launcher);

  // Since we may have received a new physical instance we need
  // to update our accessor as well.  Again this implicitly calls
  // 'wait_until_valid' to ensure we have valid data.
  const FieldAccessor<READ_ONLY,double,1> acc_z(output_region, FID_Z);

  printf("Checking results...");
  bool all_passed = true;
  // Check our results are the same
  for (PointInRectIterator<1> pir(elem_rect); pir(); pir++)
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

  // Clean up all our data structures.
  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, input_fs);
  runtime->destroy_field_space(ctx, output_fs);
  runtime->destroy_index_space(ctx, is);
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
