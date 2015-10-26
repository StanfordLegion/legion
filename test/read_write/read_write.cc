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
#include <math.h>
#include "legion.h"
#include "default_mapper.h"

#include <errno.h>
#include <aio.h>
// included for file memory data transfer
#include <unistd.h>
#include <linux/aio_abi.h>
#include <sys/syscall.h>

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  READ_ONLY_TASK_ID,
  READ_WRITE_TASK_ID,
};

enum FieldIDs {
  FID_VAL,
};

enum TestMode {
  NONE,
  ATTACH,
  READFILE,
  INIT
};

class ReadWriteMapper : public DefaultMapper {
public:
  ReadWriteMapper(Machine machine, HighLevelRuntime *runtime, Processor local)
    : DefaultMapper(machine, runtime, local)
  {
    std::set<Memory> all_mem;
    machine.get_all_memories(all_mem);
    for (std::set<Memory>::iterator it = all_mem.begin(); it != all_mem.end(); it++) {
      if (it->kind() == Memory::SYSTEM_MEM)
        sys_mem.push_back(*it);
    }
    assert(sys_mem.size() == 1);
    std::set<Processor> all_procs;
    machine.get_all_processors(all_procs);
    first_proc = Processor::NO_PROC, last_proc = Processor::NO_PROC;
    for (std::set<Processor>::iterator it = all_procs.begin(); it != all_procs.end(); it++) {
      if (it->kind() == Processor::LOC_PROC) {
        if (first_proc == Processor::NO_PROC) {
          first_proc = *it;
        }
        last_proc = *it;
      }
    }
  }

  virtual void select_task_options(Task *task)
  {
    task->inline_task = false;
    task->spawn_task = false;
    task->map_locally = true;
    task->profile_task = false;
    if (task->task_id == READ_ONLY_TASK_ID)
      task->target_proc = first_proc;
    else if (task->task_id == READ_WRITE_TASK_ID)
      task->target_proc = last_proc;
  }

  virtual bool map_task(Task *task)
  {
    for (unsigned idx = 0; idx < task->regions.size(); idx++)
    {
      task->regions[idx].target_ranking.push_back(sys_mem[0]);
      task->regions[idx].virtual_map = false;
      task->regions[idx].enable_WAR_optimization = true;
      task->regions[idx].reduction_list = false;
      // Make everything SOA
      task->regions[idx].blocking_factor =
        task->regions[idx].max_blocking_factor;
    }
    return true;
  }
public:
  std::vector<Memory> sys_mem;
  Processor first_proc, last_proc;
};

static void update_mappers(Machine machine, HighLevelRuntime *rt,
                           const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(new ReadWriteMapper(machine, rt, *it), *it);
  }
}


void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  int nsize = 1024;
  int ntask = 4;
  TestMode mode = NONE;
  char input_file[128];
  //sprintf(input_file, "/scratch/sdb1_ext4/input.dat");
  sprintf(input_file, "input.dat");

  // Check for any command line arguments
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        nsize = atoi(command_args.argv[++i]);
      else if (!strcmp(command_args.argv[i],"-t"))
        ntask = atoi(command_args.argv[++i]);
      else if (!strcmp(command_args.argv[i],"-init"))
        mode = INIT;
    }
  }
  printf("Running read/write tasks with nsize = %d\n", nsize);
  printf("Generating #tasks = %d\n", ntask);

  Rect<1> rect_A(Point<1>(0), Point<1>(nsize - 1));
  IndexSpace is_A = runtime->create_index_space(ctx,
                          Domain::from_rect<1>(rect_A));
  FieldSpace fs_A = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs_A);
    allocator.allocate_field(sizeof(int),FID_VAL);
  }
  LogicalRegion lr_A = runtime->create_logical_region(ctx, is_A, fs_A);

  if (mode == INIT) {
    printf("Creating input file: %s\n", input_file);
    RegionRequirement req(lr_A, WRITE_DISCARD, EXCLUSIVE, lr_A);
    req.add_field(FID_VAL);
    InlineLauncher input_launcher(req);
    PhysicalRegion pr_A = runtime->map_region(ctx, input_launcher);
    pr_A.wait_until_valid();
    RegionAccessor<AccessorType::Generic, int> acc =
      pr_A.get_field_accessor(FID_VAL).typeify<int>();
    for (GenericPointInRectIterator<1> pir(rect_A); pir; pir++) {
      acc.write(DomainPoint::from_point<1>(pir.p), 1);
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

  PhysicalRegion pr_A;

  std::vector<FieldID> field_vec;
  field_vec.push_back(FID_VAL);
  pr_A = runtime->attach_file(ctx, input_file, lr_A, lr_A, field_vec, LEGION_FILE_READ_WRITE);
  runtime->remap_region(ctx, pr_A);
  pr_A.wait_until_valid();

  // Start Computation
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);

  runtime->unmap_region(ctx, pr_A);
  for (int iter = 0; iter < ntask; iter++) {
    // Acquire the logical reagion so that we can launch sub-operations that make copies
    AcquireLauncher acquire_launcher(lr_A, lr_A, pr_A);
    acquire_launcher.add_field(FID_VAL);
    runtime->issue_acquire(ctx, acquire_launcher);

    Future future;
    // Perform task
    if (iter % 2 == 0) {
      // read_only task
      TaskLauncher task_launcher(READ_ONLY_TASK_ID,
                                 TaskArgument(NULL, 0));
      task_launcher.add_region_requirement(
          RegionRequirement(lr_A, READ_ONLY, EXCLUSIVE, lr_A));
      task_launcher.add_field(0, FID_VAL);
      future = runtime->execute_task(ctx, task_launcher);
    } else {
      // read_write task
      TaskLauncher task_launcher(READ_WRITE_TASK_ID,
                                 TaskArgument(NULL, 0));
      task_launcher.add_region_requirement(
          RegionRequirement(lr_A, READ_WRITE, EXCLUSIVE, lr_A));
      task_launcher.add_field(0, FID_VAL);
      future = runtime->execute_task(ctx, task_launcher);
    }

    if(iter == ntask - 1) {
      future.get_void_result();
    }

    //Release the attached physicalregion
    ReleaseLauncher release_launcher(lr_A, lr_A, pr_A);
    release_launcher.add_field(FID_VAL);
    runtime->issue_release(ctx, release_launcher);
  }

  clock_gettime(CLOCK_MONOTONIC, &ts_end);

  runtime->remap_region(ctx, pr_A);
  runtime->detach_file(ctx, pr_A);

  double exec_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
                     (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", exec_time);

  // Clean up our region, index space, and field space
  runtime->destroy_logical_region(ctx, lr_A);
  runtime->destroy_field_space(ctx, fs_A);
  runtime->destroy_index_space(ctx, is_A);
}

void read_only_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid_A = *(task->regions[0].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, int> acc_A =
    regions[0].get_field_accessor(fid_A).typeify<int>();

  Domain dom_A = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  int nsize = dom_A.get_rect<1>().dim_size(0);

  int sum;
  for (int k = 0; k < nsize; k++)
    for (int i = 0; i < nsize; i++) {
      int x = acc_A.read(ptr_t(k));
      int y = acc_A.read(ptr_t(i));
      if (i % 2 == 0) sum += x * y; else sum -= x * y;
    }
  printf("OK!\n");
}

void read_write_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid_A = *(task->regions[0].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, int> acc_A =
    regions[0].get_field_accessor(fid_A).typeify<int>();

  Domain dom_A = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  int nsize = dom_A.get_rect<1>().dim_size(0);

  int sum;
  for (int k = 0; k < nsize; k++)
    for (int i = 0; i < nsize; i++) {
      int x = acc_A.read(ptr_t(k));
      int y = acc_A.read(ptr_t(i));
      if (i % 2 == 0) sum += x * y; else sum -= x * y;
    }
  for (int k = 0; k < nsize; k++) {
    acc_A.write(ptr_t(k), sum);
  }
  printf("OK!\n");
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);
  HighLevelRuntime::register_legion_task<read_only_task>(READ_ONLY_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);
  HighLevelRuntime::register_legion_task<read_write_task>(READ_WRITE_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);

  // Register custom mappers
  HighLevelRuntime::set_registration_callback(update_mappers);
  return HighLevelRuntime::start(argc, argv);
}
