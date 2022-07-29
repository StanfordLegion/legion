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
#include "mappers/default_mapper.h"

using namespace Legion;

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
  CLASS_METHOD_TASK_ID,
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
  LegionTaskWrapper::legion_task_postamble(ctx);
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

class ClassWithTaskMethods {
public:
  ClassWithTaskMethods(int _x) : x(_x) {}

  void method_task(const Task *task,
		   const std::vector<PhysicalRegion> &regions,
		   Context ctx, Runtime *runtime)
  {
    printf("hello from class method: this=%p x=%d\n", this, x);
  }

  static void static_entry_method(const Task *task,
				  const std::vector<PhysicalRegion> &regions,
				  Context ctx, Runtime *runtime,
				  ClassWithTaskMethods * const & _this)
  {
    // just call through to actual class method
    _this->method_task(task, regions, ctx, runtime);
  }

protected:
  int x;
};

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
  "  %1 = bitcast [31 x i8]* @.str to i8*\n"
  "  %2 = call i32 (i8*, ...) @printf(i8* %1)\n"
  "  ret void\n"
  "}\n"
  "define void @llvm_wrapper(i8* %data, i64 %datalen, i8* %userdata, i64 %userlen, i64 %proc_id) {\n"
  "  %task_ptr = alloca %struct.legion_task_t, align 8\n"
  "  %regions_ptr = alloca %struct.legion_physical_region_t*, align 8\n"
  "  %num_regions_ptr = alloca i32, align 4\n"
  "  %ctx_ptr = alloca %struct.legion_context_t, align 8\n"
  "  %runtime_ptr = alloca %struct.legion_runtime_t, align 8\n"
  "  call void @legion_task_preamble(i8* %data, i64 %datalen, i64 %proc_id, %struct.legion_task_t* %task_ptr, %struct.legion_physical_region_t** %regions_ptr, i32* %num_regions_ptr, %struct.legion_context_t* %ctx_ptr, %struct.legion_runtime_t* %runtime_ptr)\n"
  "  %task = load %struct.legion_task_t, %struct.legion_task_t* %task_ptr\n"
  "  %regions = load %struct.legion_physical_region_t*, %struct.legion_physical_region_t** %regions_ptr\n"
  "  %num_regions = load i32, i32* %num_regions_ptr\n"
  "  %ctx = load %struct.legion_context_t, %struct.legion_context_t* %ctx_ptr\n"
  "  %runtime = load %struct.legion_runtime_t, %struct.legion_runtime_t* %runtime_ptr\n"
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

#ifdef REALM_USE_LIBDL
  // rely on dladdr/dlsym to make function pointers portable for global
  //  task registration
  bool global_taskreg = true;
#else
  // function pointers will not be portable, so limit tasks to local node
  const bool global_taskreg = false;
#endif

  // Dynamically register some more tasks
  TaskVariantRegistrar init_registrar(INIT_FIELD_TASK_ID,
                                      "cpu_init_variant",
				      global_taskreg);
  // Add our constraints
  init_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
      .add_layout_constraint_set(0/*index*/, soa_layout_id);
  runtime->register_task_variant<init_field_task>(init_registrar);

  TaskVariantRegistrar stencil_registrar(STENCIL_TASK_ID,
                                         "cpu_stencil_variant",
					 global_taskreg);
  stencil_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
      .add_layout_constraint_set(0/*index*/, soa_layout_id)
      .add_layout_constraint_set(1/*index*/, soa_layout_id);
  runtime->register_task_variant<stencil_task>(stencil_registrar);

  TaskVariantRegistrar check_registrar(CHECK_TASK_ID,
                                       "cpu_check_variant",
				       global_taskreg);
  check_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
      .add_layout_constraint_set(0/*index*/, soa_layout_id)
      .add_layout_constraint_set(1/*index*/, soa_layout_id);
  runtime->register_task_variant<check_task>(check_registrar);

  TaskVariantRegistrar wrapped_cpp_registrar(WRAPPED_CPP_TASK_ID,
					     "wrapped_cpp_variant",
					     global_taskreg);
  wrapped_cpp_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  const char cpp_msg[] = "user data for cpp task";
  runtime->register_task_variant(wrapped_cpp_registrar,
				 CodeDescriptor(wrapped_cpp_task),
				 cpp_msg, sizeof(cpp_msg));

  TaskVariantRegistrar wrapped_c_registrar(WRAPPED_C_TASK_ID,
					   "wrapped_c_variant",
					   global_taskreg);
  wrapped_c_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  const char c_msg[] = "user data for c task";
  runtime->register_task_variant(wrapped_c_registrar,
				 CodeDescriptor(wrapped_c_task),
				 c_msg, sizeof(c_msg));

  ClassWithTaskMethods object_with_task_methods(22);
  TaskVariantRegistrar class_method_registrar(CLASS_METHOD_TASK_ID,
					      "class_method_variant",
					      false /*can't be global*/);
  class_method_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  runtime->register_task_variant<ClassWithTaskMethods *,
				 ClassWithTaskMethods::static_entry_method>
    (class_method_registrar,
     &object_with_task_methods /*pointer to object passed as 'user_data'*/);

#ifdef REALM_USE_LLVM
  // LLVM IR is portable, so we can do global registration even without libdl
  TaskVariantRegistrar wrapped_llvm_registrar(WRAPPED_LLVM_TASK_ID,
					      "wrapped_llvm_variant",
					      true /*global*/);
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
    if (!global_taskreg)
      l.tag |= Legion::Mapping::DefaultMapper::SAME_ADDRESS_SPACE;
    Future f = runtime->execute_task(ctx, l);
    f.get_void_result();
  }

  {
    int val = 66;
    TaskLauncher l(WRAPPED_C_TASK_ID, TaskArgument(&val, sizeof(val)));
    if (!global_taskreg)
      l.tag |= Legion::Mapping::DefaultMapper::SAME_ADDRESS_SPACE;
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

  {
    int val = 88;
    TaskLauncher l(CLASS_METHOD_TASK_ID, TaskArgument(&val, sizeof(val)));
    // task uses locally allocated object, so must stay local
    l.tag |= Legion::Mapping::DefaultMapper::SAME_ADDRESS_SPACE;
    Future f = runtime->execute_task(ctx, l);
    f.get_void_result();
  }

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

  Rect<1> elem_rect(0,num_elements-1);
  IndexSpaceT<1> is = runtime->create_index_space(ctx, elem_rect);
  LogicalRegion stencil_lr = runtime->create_logical_region(ctx, is, fs);
  
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
  if (!global_taskreg)
    init_launcher.tag |= Legion::Mapping::DefaultMapper::SAME_ADDRESS_SPACE;
  init_launcher.add_region_requirement(
      RegionRequirement(disjoint_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, stencil_lr));
  init_launcher.add_field(0, FID_VAL);
  runtime->execute_index_space(ctx, init_launcher);

  IndexLauncher stencil_launcher(STENCIL_TASK_ID, color_is,
       TaskArgument(&num_elements, sizeof(num_elements)), arg_map);
  if (!global_taskreg)
    stencil_launcher.tag |= Legion::Mapping::DefaultMapper::SAME_ADDRESS_SPACE;
  stencil_launcher.add_region_requirement(
      RegionRequirement(ghost_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, stencil_lr));
  stencil_launcher.add_field(0, FID_VAL);
  stencil_launcher.add_region_requirement(
      RegionRequirement(disjoint_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, stencil_lr));
  stencil_launcher.add_field(1, FID_DERIV);
  runtime->execute_index_space(ctx, stencil_launcher);

  TaskLauncher check_launcher(CHECK_TASK_ID, 
      TaskArgument(&num_elements, sizeof(num_elements)));
  if (!global_taskreg)
    check_launcher.tag |= Legion::Mapping::DefaultMapper::SAME_ADDRESS_SPACE;
  check_launcher.add_region_requirement(
      RegionRequirement(stencil_lr, READ_ONLY, EXCLUSIVE, stencil_lr));
  check_launcher.add_field(0, FID_VAL);
  check_launcher.add_region_requirement(
      RegionRequirement(stencil_lr, READ_ONLY, EXCLUSIVE, stencil_lr));
  check_launcher.add_field(1, FID_DERIV);
  runtime->execute_task(ctx, check_launcher);

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
  // We'll only register our top-level task here
  TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID,
                                 "top_level_variant");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  Runtime::preregister_task_variant<top_level_task>(registrar,"top_level_task");

  return Runtime::start(argc, argv);
}
