/* Copyright 2018 Stanford University
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
#include <sys/time.h>
#include "legion.h"
using namespace Legion;

#define DEBUG_RESILIENCE

#define PRINT_KSMURTHY


#define SLEEP_LOOP(num) volatile unsigned int sl1 = 0; while(sl1++ < num); 

#define SINGLE_ERROR
//#define DUAL_ERRORS
//#define CONCURRENT_ERRORS


enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_ID,
  DAXPY_TASK_ID,
  CHECK_TASK_ID,
  FINALIZE_TASK_ID,
};

enum FieldIDs {
  FID_X,
};

struct task_ar {
  int cur_itr;
  int error_itr;
  unsigned int init_sleep;
  unsigned int daxpy_sleep;
  unsigned int check_sleep;
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_elements = 1; 
  int num_itr = 2; 
  int error_itr = 1;
  unsigned int init_sleep = 0;
  unsigned int daxpy_sleep = 0;
  unsigned int check_sleep = 0;

  // See if we have any command line arguments to parse
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-i"))
        num_itr = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-e"))
        error_itr = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-x"))
        init_sleep = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-y"))
        daxpy_sleep = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-z"))
        check_sleep = atoi(command_args.argv[++i]);
    }
  }
  printf("Running daxpy for %d elements... %d iterations, error:%d itr processor:%llx\n", 
						num_elements, num_itr, error_itr, runtime->get_executing_processor(ctx).id);

  // Create our logical regions using the same schema that
  // we used in the previous example.
  const Rect<1> elem_rect(0,num_elements-1);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect); 
  FieldSpace input_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, input_fs);
    allocator.allocate_field(sizeof(double),FID_X);
  }
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);

#if 1
    // Better timing for Legion
    TimingLauncher timing_launcher(MEASURE_MICRO_SECONDS);
    //runtime->issue_execution_fence(ctx);
    Future f_start = runtime->issue_timing_measurement(ctx, timing_launcher);
#endif

  struct task_ar *arg = (struct task_ar *)malloc(sizeof(struct task_ar));;
  arg->error_itr = error_itr;
  arg->cur_itr = 0;
  arg->init_sleep = init_sleep;
  arg->daxpy_sleep = daxpy_sleep;
  arg->check_sleep = check_sleep;

  TaskLauncher init_launcher(INIT_FIELD_TASK_ID, 
                      TaskArgument(arg, sizeof(struct task_ar *)));
  init_launcher.add_region_requirement(
      RegionRequirement(input_lr, WRITE_ONLY, EXCLUSIVE, input_lr));
  init_launcher.add_field(0/*idx*/, FID_X);
  runtime->execute_task(ctx, init_launcher);

  int cur_itr = 1;
  while(cur_itr <= num_itr) {

    struct task_ar *arg = (struct task_ar *)malloc(sizeof(struct task_ar));;
    arg->error_itr = error_itr;
    arg->cur_itr = cur_itr;
    arg->init_sleep = init_sleep;
    arg->daxpy_sleep = daxpy_sleep;
    arg->check_sleep = check_sleep;

    TaskLauncher init_launcher(INIT_FIELD_TASK_ID, 
                      TaskArgument(arg, sizeof(struct task_ar *)));
    init_launcher.add_region_requirement(
        RegionRequirement(input_lr, READ_WRITE, EXCLUSIVE, input_lr));
    init_launcher.add_field(0/*idx*/, FID_X);
    runtime->execute_task(ctx, init_launcher);
  
    TaskLauncher daxpy_launcher(DAXPY_TASK_ID, 
                      TaskArgument(arg, sizeof(struct task_ar *)));
    daxpy_launcher.add_region_requirement(
        RegionRequirement(input_lr, READ_WRITE, EXCLUSIVE, input_lr));
    daxpy_launcher.add_field(0/*idx*/, FID_X);
    runtime->execute_task(ctx, daxpy_launcher);
  
    TaskLauncher check_launcher(CHECK_TASK_ID, 
                      TaskArgument(arg, sizeof(struct task_ar *)));
    check_launcher.add_region_requirement(
        RegionRequirement(input_lr, READ_WRITE, EXCLUSIVE, input_lr));
    check_launcher.add_field(0/*idx*/, FID_X);
    runtime->execute_task(ctx, check_launcher);
 
    cur_itr++; 
  } //end of iterations

    arg->error_itr = error_itr;
    arg->cur_itr = cur_itr;
    TaskLauncher finalize_launcher(FINALIZE_TASK_ID, 
                      TaskArgument(arg, sizeof(struct task_ar *)));
    finalize_launcher.add_region_requirement(
        RegionRequirement(input_lr, READ_WRITE, EXCLUSIVE, input_lr));
    finalize_launcher.add_field(0/*idx*/, FID_X);
    Future f_cdt = runtime->execute_task(ctx, finalize_launcher);

#if 1
    // get stopping timestamp
    timing_launcher.preconditions.clear();
    // Measure after f_cdt is ready which is when the cycle is complete
    timing_launcher.add_precondition(f_cdt);
    Future f_stop = runtime->issue_timing_measurement(ctx, timing_launcher);
    const double tbegin = f_start.get_result<long long>(true/*silence warnings*/);
    const double tend = f_stop.get_result<long long>(true/*silence warnings*/);
    const double walltime = tend - tbegin;
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "time = %14.6g\n\n", walltime);
#endif

 
//  runtime->destroy_logical_region(ctx, input_lr);
//  runtime->destroy_field_space(ctx, input_fs);
//  runtime->destroy_index_space(ctx, is);

}

void init_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->arglen == sizeof(struct task_ar *));
  const int cur_itr= ((struct task_ar *)task->args)->cur_itr;
#ifdef DEBUG_RESILIENCE
  static int hello_tracker1 = 10;
  //SLEEP_LOOP((rand()%2)*1000000);
#endif

  SLEEP_LOOP(((struct task_ar *)task->args)->init_sleep);
  const FieldAccessor<READ_WRITE,double,1> acc(regions[0], FID_X);
  Rect<1> rect = runtime->get_index_space_domain(ctx, 
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++) {
    acc[*pir] = cur_itr * hello_tracker1;
#ifdef PRINT_KSMURTHY
    std::cout<<"OUTPUT_PRINTS: [initfield] in itr:" << cur_itr 
             << " value wrote in accx:" << acc[*pir] << 
            " running on processor:" << runtime->get_executing_processor(ctx) << std::endl;
#endif
  }
}

void daxpy_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->arglen == sizeof(struct task_ar *));
  const int cur_itr= ((struct task_ar *)task->args)->cur_itr;
  const int error_itr = ((struct task_ar *)task->args)->error_itr;
  const FieldAccessor<READ_WRITE,double,1> acc_x(regions[0], FID_X);
  const double alpha = 0.08; 
  static int hello_tracker2 = 0;
#ifdef DEBUG_RESILIENCE
  hello_tracker2++;
#endif
#ifdef PRINT_KSMURTHY
  printf("OUTPUT_PRINTS: Running DAXPY iteration:%d tracker:%d processor:%llx\n",cur_itr, hello_tracker2,runtime->get_executing_processor(ctx).id);
#endif
  Rect<1> rect_init = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect_init); pir(); pir++) {
#ifdef PRINT_KSMURTHY
    std::cout<<"OUTPUT_PRINTS: [daxpy] value found in accx:" << acc_x[*pir] <<  
            " running on processor:" << runtime->get_executing_processor(ctx) << std::endl;
#endif
    acc_x[*pir] = hello_tracker2; 
  }
#ifdef DEBUG_RESILIENCE
  if((hello_tracker2 == error_itr) && (runtime->get_executing_processor(ctx).id == 0x1d00010000000001)) {
#ifdef PRINT_KSMURTHY
    printf("\n ABOUT TO FAIL in DAXPY iteration:%d tracker:%d processor:%llx\n",cur_itr,hello_tracker2,runtime->get_executing_processor(ctx).id); 
#endif
    throw std::exception();
  }
#endif
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  SLEEP_LOOP(((struct task_ar *)task->args)->daxpy_sleep);
  for (PointInRectIterator<1> pir(rect); pir(); pir++) {
    acc_x[*pir] = alpha * acc_x[*pir] + acc_x[*pir];
#ifdef PRINT_KSMURTHY
    std::cout<<"OUTPUT_PRINTS: [daxpy] value WRITTEN in accx:" << acc_x[*pir] <<  
            " running on processor:" << runtime->get_executing_processor(ctx) << std::endl;
#endif
  }
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  //sleep(5);
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->arglen == sizeof(struct task_ar *));
  const FieldAccessor<READ_WRITE,double,1> acc_x(regions[0], FID_X);
  const int cur_itr= ((struct task_ar *)task->args)->cur_itr;
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  SLEEP_LOOP(((struct task_ar *)task->args)->check_sleep);
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
  {
#ifdef PRINT_KSMURTHY
    std::cout<<"OUTPUT_PRINTS: [check] in itr:" << cur_itr 
             << " accx:" << acc_x[*pir] <<  
            " running on processor:" << runtime->get_executing_processor(ctx) << std::endl;
#endif
  }
}

void finalize_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  printf("\n\nFINALIZE processor:%llx\n\n\n",runtime->get_executing_processor(ctx).id);
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

  {
    TaskVariantRegistrar registrar(FINALIZE_TASK_ID, "finalize");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<finalize_task>(registrar, "finalize");
  }

  return Runtime::start(argc, argv);
}
