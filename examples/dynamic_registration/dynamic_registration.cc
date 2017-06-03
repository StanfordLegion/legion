/* Copyright 2017 Stanford University
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
#include "legion_c.h"

using namespace Legion;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

#ifdef REALM_USE_LLVM
#include "realm/llvmjit/llvmjit.h"
#endif

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
  WRAPPED_CPP_TASK_ID,
  WRAPPED_C_TASK_ID,
#ifdef REALM_USE_LLVM
  WRAPPED_LLVM_TASK_ID,
#endif
};

enum FieldIDs {
  FID_VAL,
  FID_DERIV,
};

// Forward declarations

void init_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime);

void stencil_task(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime);

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime);

void wrapped_cpp_task(const void *data, size_t datalen,
		      const void *userdata, size_t userlen, Processor p)
{
  const Task *task;
  const std::vector<PhysicalRegion> *regions;
  Context ctx;
  Runtime *runtime;
  LegionTaskWrapper::legion_task_preamble(data, datalen, p,
					  task,
					  regions,
					  ctx,
					  runtime);
  printf("hello from wrapped_cpp_task (msg='%.*s')\n",
	 (int)userlen, (const char *)userdata);
  LegionTaskWrapper::legion_task_postamble(runtime, ctx);
}

void wrapped_c_task(const void *data, size_t datalen,
		    const void *userdata, size_t userlen, Processor p)
{
  legion_task_t task;
  const legion_physical_region_t *regions;
  unsigned num_regions;
  legion_context_t ctx;
  legion_runtime_t runtime;
  legion_task_preamble(data, datalen, p.id,
		       &task,
		       &regions,
		       &num_regions,
		       &ctx,
		       &runtime);
  printf("hello from wrapped_c_task (msg='%.*s')\n",
	 (int)userlen, (const char *)userdata);
  legion_task_postamble(runtime, ctx, 0, 0);
}

#ifdef REALM_USE_LLVM
const char llvm_ir[] = 
  "%struct.legion_physical_region_t = type { i8* }\n"
  "%struct.legion_task_t = type { i8* }\n"
  "%struct.legion_context_t = type { i8* }\n"
  "%struct.legion_runtime_t = type { i8* }\n"
  "declare i32 @printf(i8*, ...)\n"
  "declare void @legion_task_preamble(i8*, i64, i64, %struct.legion_task_t*, %struct.legion_physical_region_t**, i32*, %struct.legion_context_t*, %struct.legion_runtime_t*)\n"
  "declare void @legion_task_postamble(%struct.legion_runtime_t, %struct.legion_context_t, i8*, i64)\n"
  "@.str = private unnamed_addr constant [31 x i8] c\"hello from llvm wrapped task!\\0A\\00\", align 1\n"
  "define void @body(%struct.legion_task_t %task, %struct.legion_physical_region_t* %regions, i32 %num_regions, %struct.legion_context_t %ctx, %struct.legion_runtime_t %runtime) {\n"
  "  %1 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([31 x i8]* @.str, i32 0, i32 0))\n"
  "  ret void\n"
  "}\n"
  "define void @llvm_wrapper(i8* %data, i64 %datalen, i8* %userdata, i64 %userlen, i64 %proc_id) {\n"
  "  %task_ptr = alloca %struct.legion_task_t, align 8\n"
  "  %regions_ptr = alloca %struct.legion_physical_region_t*, align 8\n"
  "  %num_regions_ptr = alloca i32, align 4\n"
  "  %ctx_ptr = alloca %struct.legion_context_t, align 8\n"
  "  %runtime_ptr = alloca %struct.legion_runtime_t, align 8\n"
  "  call void @legion_task_preamble(i8* %data, i64 %datalen, i64 %proc_id, %struct.legion_task_t* %task_ptr, %struct.legion_physical_region_t** %regions_ptr, i32* %num_regions_ptr, %struct.legion_context_t* %ctx_ptr, %struct.legion_runtime_t* %runtime_ptr)\n"
  "  %task = load %struct.legion_task_t* %task_ptr\n"
  "  %regions = load %struct.legion_physical_region_t** %regions_ptr\n"
  "  %num_regions = load i32* %num_regions_ptr\n"
  "  %ctx = load %struct.legion_context_t* %ctx_ptr\n"
  "  %runtime = load %struct.legion_runtime_t* %runtime_ptr\n"
  "  call void @body(%struct.legion_task_t %task, %struct.legion_physical_region_t* %regions, i32 %num_regions, %struct.legion_context_t %ctx, %struct.legion_runtime_t %runtime)\n"
  "  call void @legion_task_postamble(%struct.legion_runtime_t %runtime, %struct.legion_context_t %ctx, i8* null, i64 0)\n"
  "  ret void\n"
  "}\n"
  ;
#endif

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double),FID_VAL);
    allocator.allocate_field(sizeof(double),FID_DERIV);
  }
  // Make an SOA constraint and use it as the layout constraint for
  // all the different task variants that we are registering
  LayoutConstraintRegistrar layout_registrar(fs, "SOA layout");
  std::vector<DimensionKind> dim_order(2);
  dim_order[0] = DIM_X;
  dim_order[1] = DIM_F; // fields go last for SOA
  layout_registrar.add_constraint(OrderingConstraint(dim_order, false/*contig*/));

  LayoutConstraintID soa_layout_id = runtime->register_layout(layout_registrar);

  // Dynamically register some more tasks
  TaskVariantRegistrar init_registrar(INIT_FIELD_TASK_ID,
                                      "cpu_init_variant");
  // Add our constraints
  init_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
      .add_layout_constraint_set(0/*index*/, soa_layout_id);
  runtime->register_task_variant<init_field_task>(init_registrar);

  TaskVariantRegistrar stencil_registrar(STENCIL_TASK_ID,
                                         "cpu_stencil_variant");
  stencil_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
      .add_layout_constraint_set(0/*index*/, soa_layout_id)
      .add_layout_constraint_set(1/*index*/, soa_layout_id);
  runtime->register_task_variant<stencil_task>(stencil_registrar);

  TaskVariantRegistrar check_registrar(CHECK_TASK_ID,
                                       "cpu_check_variant");
  check_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
      .add_layout_constraint_set(0/*index*/, soa_layout_id)
      .add_layout_constraint_set(1/*index*/, soa_layout_id);
  runtime->register_task_variant<check_task>(check_registrar);

  TaskVariantRegistrar wrapped_cpp_registrar(WRAPPED_CPP_TASK_ID,
					     "wrapped_cpp_variant",
					      true /*global*/);
  wrapped_cpp_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  const char cpp_msg[] = "user data for cpp task";
  runtime->register_task_variant(wrapped_cpp_registrar,
				 CodeDescriptor(wrapped_cpp_task),
				 cpp_msg, sizeof(cpp_msg));

  TaskVariantRegistrar wrapped_c_registrar(WRAPPED_C_TASK_ID,
					   "wrapped_c_variant",
					   true /*global*/);
  wrapped_c_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  const char c_msg[] = "user data for c task";
  runtime->register_task_variant(wrapped_c_registrar,
				 CodeDescriptor(wrapped_c_task),
				 c_msg, sizeof(c_msg));

#ifdef REALM_USE_LLVM
  TaskVariantRegistrar wrapped_llvm_registrar(WRAPPED_LLVM_TASK_ID, true/*global*/,
					      "wrapped_llvm_variant");
  wrapped_llvm_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  const char llvm_msg[] = "user data for llvm task";
  CodeDescriptor llvm_cd(Realm::Type::from_cpp_type<Processor::TaskFuncPtr>());
  llvm_cd.add_implementation(new Realm::LLVMIRImplementation(llvm_ir, sizeof(llvm_ir),
							     "llvm_wrapper"));
  runtime->register_task_variant(wrapped_llvm_registrar,
				 llvm_cd,
				 llvm_msg, sizeof(llvm_msg));
#endif

  // Attach semantic infos to the task names
  runtime->attach_name(INIT_FIELD_TASK_ID, "init task");
  runtime->attach_name(STENCIL_TASK_ID, "stencil task");
  runtime->attach_name(CHECK_TASK_ID, "check task");
  runtime->attach_name(WRAPPED_CPP_TASK_ID, "wrapped cpp task");
  runtime->attach_name(WRAPPED_C_TASK_ID, "wrapped c task");
#ifdef REALM_USE_LLVM
  runtime->attach_name(WRAPPED_LLVM_TASK_ID, "wrapped llvm task");
#endif

  {
    int val = 55;
    TaskLauncher l(WRAPPED_CPP_TASK_ID, TaskArgument(&val, sizeof(val)));
    Future f = runtime->execute_task(ctx, l);
    f.get_void_result();
  }

  {
    int val = 66;
    TaskLauncher l(WRAPPED_C_TASK_ID, TaskArgument(&val, sizeof(val)));
    Future f = runtime->execute_task(ctx, l);
    f.get_void_result();
  }

#ifdef REALM_USE_LLVM
  {
    int val = 77;
    TaskLauncher l(WRAPPED_LLVM_TASK_ID, TaskArgument(&val, sizeof(val)));
    Future f = runtime->execute_task(ctx, l);
    f.get_void_result();
  }
#endif

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
  
  LogicalRegion stencil_lr = runtime->create_logical_region(ctx, is, fs);
  
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
  // We'll only register our top-level task here
  TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID,
                                 "top_level_variant");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  Runtime::preregister_task_variant<top_level_task>(registrar,"top_level_task");

  return Runtime::start(argc, argv);
}
