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
#include <queue>
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

typedef int ElementType;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  COMPUTE_TASK_ID,
  COMPUTE_FROM_FILE_TASK_ID,
};

enum FieldIDs {
  FID_VAL,
  FID_DERIV,
  FID_CP
};

inline int io_setup(unsigned nr, aio_context_t *ctxp)
{
  return syscall(__NR_io_setup, nr, ctxp);
}

inline int io_destroy(aio_context_t ctx)
{
  return syscall(__NR_io_destroy, ctx);
}

inline int io_submit(aio_context_t ctx, long nr, struct iocb **iocbpp)
{
  return syscall(__NR_io_submit, ctx, nr, iocbpp);
}

inline int io_getevents(aio_context_t ctx, long min_nr, long max_nr,
                        struct io_event *events, struct timespec *timeout)
{
  return syscall(__NR_io_getevents, ctx, min_nr, max_nr, events, timeout);
}

enum TestMode {
  NONE,
  ATTACH,
  READFILE,
  INIT
};

class PipelineMapper : public DefaultMapper {
public:
  PipelineMapper(Machine machine, HighLevelRuntime *runtime, Processor local)
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
    for (std::set<Processor>::iterator it = all_procs.begin(); it != all_procs.end(); it++) {
      if (it->kind() == Processor::LOC_PROC) {
        cpu_procs.push_back(*it);
      }
    }
  }

  virtual void select_task_options(Task *task)
  {
    task->inline_task = false;
    task->spawn_task = false;
    task->map_locally = true;
    task->profile_task = false;
    std::set<Processor> all_procs;
    machine.get_all_processors(all_procs);
    if (task->task_id == COMPUTE_TASK_ID) {
      int idx = *((int*)task->args);
      task->target_proc = cpu_procs[idx % cpu_procs.size()];
    }
  }

  virtual bool map_task(Task *task)
  {
    for (unsigned idx = 0; idx < task->regions.size(); idx++)
    {
      task->regions[idx].target_ranking.push_back(sys_mem[0]);
      task->regions[idx].virtual_map = false;
      task->regions[idx].enable_WAR_optimization = true;
      task->regions[idx].reduction_list = false;
      task->regions[idx].blocking_factor =
        task->regions[idx].max_blocking_factor;
    }
    return true;
  }

  virtual bool map_copy(Copy *copy)
  {
    bool ret = DefaultMapper::map_copy(copy);
    copy->dst_requirements[0].target_ranking.clear();
    copy->dst_requirements[0].target_ranking.push_back(sys_mem[0]);
    return ret;
  }
public:
  std::vector<Memory> sys_mem;
  std::vector<Processor> cpu_procs;
};

static void update_mappers(Machine machine, HighLevelRuntime *rt,
                           const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(new PipelineMapper(machine, rt, *it), *it);
  }
}

void main_task(Context ctx, HighLevelRuntime *runtime, int nsize, int nregions, TestMode mode)
{
  char input_file[64];
  //sprintf(input_file, "/scratch/sdb1_ext4/input.dat");
  sprintf(input_file, "input.dat");

  printf ("nsize = %d, npar = %d, TestMode = %d, ", nsize, nregions, mode);
  Rect<1> rect_A(Point<1>(0), Point<1>(nsize * nregions * nsize - 1));
  IndexSpace is_A = runtime->create_index_space(ctx,
                          Domain::from_rect<1>(rect_A));
  FieldSpace fs_A = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs_A);
    allocator.allocate_field(sizeof(ElementType),FID_VAL);
  }
  LogicalRegion lr_A = runtime->create_logical_region(ctx, is_A, fs_A);

  if (mode == INIT) {
    printf("Creating input file: %s\n", input_file);
    RegionRequirement req(lr_A, WRITE_DISCARD, EXCLUSIVE, lr_A);
    req.add_field(FID_VAL);
    InlineLauncher input_launcher(req);
    PhysicalRegion pr_A = runtime->map_region(ctx, input_launcher);
    pr_A.wait_until_valid();
    RegionAccessor<AccessorType::Generic, ElementType> acc =
      pr_A.get_field_accessor(FID_VAL).typeify<ElementType>();
    for (GenericPointInRectIterator<1> pir(rect_A); pir; pir++) {
      acc.write(DomainPoint::from_point<1>(pir.p), 0.01);
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

  Rect<1> rect_B(Point<1>(0), Point<1>(nsize * nsize - 1));
  IndexSpace is_B = runtime->create_index_space(ctx,
                          Domain::from_rect<1>(rect_B));
  FieldSpace fs_B = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, fs_B);
    allocator.allocate_field(sizeof(ElementType),FID_VAL);
  }
  LogicalRegion lr_B = runtime->create_logical_region(ctx, is_B, fs_B);

  PhysicalRegion pr_A;
  int global_fd = 0;
  if (mode == ATTACH) {
    std::vector<FieldID> field_vec;
    field_vec.push_back(FID_VAL);
    pr_A = runtime->attach_file(ctx, input_file, lr_A, lr_A, field_vec, LEGION_FILE_READ_ONLY);
    pr_A.wait_until_valid();
    runtime->unmap_region(ctx, pr_A);

    // Acquire the logical reagion so that we can launch sub-operations that make copies
    /*AcquireLauncher acquire_launcher(lr_A, lr_A, pr_A);
    acquire_launcher.add_field(FID_VAL);
    runtime->issue_acquire(ctx, acquire_launcher);*/
  } else if (mode == READFILE) {
    global_fd = open(input_file, O_RDONLY | O_DIRECT, 00777);
    assert(global_fd != 0);
  } else {
    assert(mode == NONE);
    // Initialize lr_A
    RegionRequirement req(lr_A, WRITE_DISCARD, EXCLUSIVE, lr_A);
    req.add_field(FID_VAL);
    InlineLauncher input_launcher(req);
    PhysicalRegion pr_A = runtime->map_region(ctx, input_launcher);
    pr_A.wait_until_valid();
    RegionAccessor<AccessorType::Generic, ElementType> acc =
      pr_A.get_field_accessor(FID_VAL).typeify<ElementType>();
    for (GenericPointInRectIterator<1> pir(rect_A); pir; pir++) {
      acc.write(DomainPoint::from_point<1>(pir.p), 1);
    }
  }

  // Initialize lr_B
  {
    RegionRequirement req(lr_B, WRITE_DISCARD, EXCLUSIVE, lr_B);
    req.add_field(FID_VAL);
    InlineLauncher input_launcher(req);
    PhysicalRegion pr_B = runtime->map_region(ctx, input_launcher);
    pr_B.wait_until_valid();
    RegionAccessor<AccessorType::Generic, ElementType> acc =
      pr_B.get_field_accessor(FID_VAL).typeify<ElementType>();
    for (GenericPointInRectIterator<1> pir(rect_B); pir; pir++) {
      acc.write(DomainPoint::from_point<1>(pir.p), 1);
    }
  }

  // Start Computation
  struct timespec ts_start, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);
  // Make our color_domain based on the number of subregions
  // that we want to create.
  Rect<1> color_bounds(Point<1>(0),Point<1>(nregions-1));
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
  IndexPartition ip_A, ip_B;
  {
    DomainColoring coloring_A, coloring_B;
    // Iterate over all the colors and compute the entry
    // for both partitions for each color.
    for (int color = 0; color < nregions; color++)
    {
      Rect<1> subrect_A(Point<1>(color * nsize * nsize), Point<1>((color + 1) * nsize * nsize - 1));
      Rect<1> subrect_B(Point<1>(0), Point<1>(nsize * nsize - 1));
      coloring_A[color] = Domain::from_rect<1>(subrect_A);
      coloring_B[color] = Domain::from_rect<1>(subrect_B);
    }
    // Once we've computed both of our colorings then we can
    // create our partitions.  Note that we tell the runtime
    // that one is disjoint will the second one is not.
    ip_A = runtime->create_index_partition(ctx, is_A, color_domain,
                                           coloring_A, true/*disjoint*/);
    ip_B = runtime->create_index_partition(ctx, is_B, color_domain,
                                           coloring_B, false);
  }

  // Once we've created our index partitions, we can get the
  // corresponding logical partitions for the stencil_lr
  // logical region.
  LogicalPartition lp_A =
    runtime->get_logical_partition(ctx, lr_A, ip_A);
  LogicalPartition lp_B =
    runtime->get_logical_partition(ctx, lr_B, ip_B);

  // Our launch domain will again be isomorphic to our coloring domain.
  Domain launch_domain = color_domain;
  ArgumentMap arg_map;

  if (mode == NONE) {
    // First initialize the 'FID_VAL' field with some data
    IndexLauncher compute_launcher(COMPUTE_TASK_ID, launch_domain,
                                TaskArgument(&nsize, sizeof(nsize)), arg_map);
    compute_launcher.add_region_requirement(
        RegionRequirement(lp_A, 0, READ_ONLY, EXCLUSIVE, lr_A));
    compute_launcher.add_field(0, FID_VAL);
    //compute_launcher.add_region_requirement(
      //  RegionRequirement(lp_B, 0, READ_ONLY, EXCLUSIVE, lr_B));
    //compute_launcher.add_field(1, FID_VAL);
    FutureMap exec_f = runtime->execute_index_space(ctx, compute_launcher);
    exec_f.wait_all_results();
  } else if (mode == ATTACH) {
    unsigned nslots = 8;
    std::queue<Future> future_vec;
    Rect<1> rect_I(Point<1>(0), Point<1>(nsize * nsize - 1));
    IndexSpace is_I = runtime->create_index_space(ctx,
	                          Domain::from_rect<1>(rect_I));
    FieldSpace fs_I = runtime->create_field_space(ctx);
    {
      FieldAllocator allocator =
        runtime->create_field_allocator(ctx, fs_I);
      allocator.allocate_field(sizeof(ElementType),FID_VAL);
    }

    for (int i = 0; i < nregions; i++) {
      while (future_vec.size() >= nslots) {
        future_vec.front().get_void_result();
        future_vec.pop();
      }
      CopyLauncher copy_launcher;
      LogicalRegion lr_sub = runtime->get_logical_subregion_by_color(ctx, lp_A, i);
      IndexSpace is_mid = runtime->get_index_subspace(ctx, ip_A, i);
      LogicalRegion lr_mid = runtime->create_logical_region(ctx, is_mid, fs_A);
      copy_launcher.add_copy_requirements(
          RegionRequirement(lr_sub, READ_ONLY, EXCLUSIVE, lr_A),
          RegionRequirement(lr_mid, WRITE_DISCARD, EXCLUSIVE, lr_mid));
      copy_launcher.add_src_field(0, FID_VAL);
      copy_launcher.add_dst_field(0, FID_VAL);
      runtime->issue_copy_operation(ctx, copy_launcher);
      TaskLauncher task_launcher(COMPUTE_TASK_ID, TaskArgument(&i, sizeof(i)));
      task_launcher.add_region_requirement(
          RegionRequirement(lr_mid, READ_ONLY, EXCLUSIVE, lr_mid));
      task_launcher.add_field(0, FID_VAL);
      //task_launcher.add_region_requirement(
        //  RegionRequirement(lr_B, READ_ONLY, EXCLUSIVE, lr_B));
      //task_launcher.add_field(1, FID_VAL);
      future_vec.push(runtime->execute_task(ctx, task_launcher));
      runtime->destroy_logical_region(ctx, lr_mid);
    }
    while (!future_vec.empty()) {
      future_vec.front().get_void_result();
      future_vec.pop();
    }

    //Release and unmap the attached physicalregion
    /*ReleaseLauncher release_launcher(lr_A, lr_A, pr_A);
    release_launcher.add_field(FID_VAL);
    runtime->issue_release(ctx, release_launcher);
    runtime->remap_region(ctx, pr_A);
    runtime->detach_file(ctx, pr_A);*/
    runtime->destroy_field_space(ctx, fs_I);
    runtime->destroy_index_space(ctx, is_I);
  } else {
    assert(mode == READFILE);
    IndexLauncher compute_launcher(COMPUTE_FROM_FILE_TASK_ID, launch_domain,
                                   TaskArgument(&global_fd, sizeof(int)), arg_map);
    compute_launcher.add_region_requirement(
        RegionRequirement(lp_A, 0, READ_WRITE, EXCLUSIVE, lr_A));
    compute_launcher.add_field(0, FID_VAL);
    compute_launcher.add_region_requirement(
        RegionRequirement(lp_B, 0, READ_ONLY, EXCLUSIVE, lr_B));
    compute_launcher.add_field(1, FID_VAL);
    FutureMap exec_f = runtime->execute_index_space(ctx, compute_launcher);
    close(global_fd);
    exec_f.wait_all_results();
  }
  clock_gettime(CLOCK_MONOTONIC, &ts_end);

  double exec_time = ((1.0 * (ts_end.tv_sec - ts_start.tv_sec)) +
                     (1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec)));
  printf("ELAPSED TIME = %7.3f s\n", exec_time);

  // Clean up our region, index space, and field space
  runtime->destroy_logical_region(ctx, lr_A);
  runtime->destroy_logical_region(ctx, lr_B);
  runtime->destroy_field_space(ctx, fs_A);
  runtime->destroy_field_space(ctx, fs_B);
  runtime->destroy_index_space(ctx, is_A);
  runtime->destroy_index_space(ctx, is_B);
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  int nsize = 1024;
  int nregions = 4;
  TestMode mode = NONE;

  // Check for any command line arguments
  {
      const InputArgs &command_args = HighLevelRuntime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        nsize = atoi(command_args.argv[++i]);
      else if (!strcmp(command_args.argv[i],"-s"))
        nregions = atoi(command_args.argv[++i]);
      else if (!strcmp(command_args.argv[i],"-init"))
        mode = INIT;
      else if (!strcmp(command_args.argv[i], "-attach"))
        mode = ATTACH;
      else if (!strcmp(command_args.argv[i], "-readfile"))
        mode = READFILE;
    }
  }

  printf("Running matrix multiplication with nsize = %d...\n", nsize);
  printf("Partitioning data into %d sub-regions...\n", nregions);
  printf("TestMode = %d\n", mode);
  for (int i = 0; i < 3; i++)
    for(nregions = 16; nregions < 1024; nregions *= 2)
      if (i == 1 && nregions == 128)
      main_task(ctx, runtime, nsize, nregions, (TestMode)i);
}

void compute_task(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);
  //assert(task->regions[1].privilege_fields.size() == 1);

  FieldID fid_A = *(task->regions[0].privilege_fields.begin());
  //FieldID fid_B = *(task->regions[1].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, ElementType> acc_A =
    regions[0].get_field_accessor(fid_A).typeify<ElementType>();
  //RegionAccessor<AccessorType::Generic, ElementType> acc_B =
    //regions[1].get_field_accessor(fid_B).typeify<ElementType>();

  Domain dom_A = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Point<1> lo_A = dom_A.get_rect<1>().lo;
  int offset = lo_A.x[0];
  int nnsize = dom_A.get_rect<1>().dim_size(0);
  int nsize = (int) round(sqrt(nnsize));
  ElementType sum = 0;
  int times = 0;
  for (times = 0; times * times < nsize; times++)
  for (int k = 0; k < nsize; k++)
    for (int i = 0; i < nsize; i++) {
      ElementType x = acc_A.read(ptr_t(offset + k * nsize + i));
      ElementType y = acc_A.read(ptr_t(offset + i));
      sum += x * y;
    }
  // int idx = *((int*)task->args);
  // printf("idx = %d\n", idx);
  // assert(fabs(sum/times - 0.01 * 0.01 * nsize * nsize) < 1e-6);
}

void compute_from_file_task(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context context, HighLevelRuntime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->arglen == sizeof(int));
  int fd = *((int*)task->args);
  int index = task->index_point.point_data[0];

  FieldID fid_A = *(task->regions[0].privilege_fields.begin());
  FieldID fid_B = *(task->regions[1].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, ElementType> acc_A =
    regions[0].get_field_accessor(fid_A).typeify<ElementType>();
  RegionAccessor<AccessorType::Generic, ElementType> acc_B =
    regions[1].get_field_accessor(fid_B).typeify<ElementType>();

  Domain dom_A = runtime->get_index_space_domain(context,
      task->regions[0].region.get_index_space());
  Point<1> lo_A = dom_A.get_rect<1>().lo;
  int offset = lo_A.x[0];
  int nnsize = dom_A.get_rect<1>().dim_size(0);
  int nsize = (int)round(sqrt(nnsize));

  ElementType* arr_A = (ElementType*) calloc(nsize * nsize, sizeof(ElementType));
  aio_context_t ctx;
  struct iocb cb;
  struct iocb *cbs[1];
  struct io_event events[1];
  ctx = 0;
  int ret = io_setup(1, &ctx);
  assert(ret >= 0);
  memset(&cb, 0, sizeof(cb));
  cb.aio_lio_opcode = IOCB_CMD_PREAD;
  cb.aio_fildes = fd;
  cb.aio_buf = (uint64_t)(arr_A);
  cb.aio_nbytes = nsize * nsize * sizeof(ElementType);
  cb.aio_offset = cb.aio_nbytes * index;
  // printf("aio_offset = %lld, aio_nbytes = %llu\n", cb.aio_offset, cb.aio_nbytes);
  cbs[0] = &cb;
  ret = io_submit(ctx, 1, cbs);
  if (ret < 0) {
    perror("io_submit error");
  }
  int nr = io_getevents(ctx, 1, 1, events, NULL);
  if (nr < 0)
    perror("io_getevents error");
  assert(nr == 1);
  assert(events[0].res == (int64_t) cb.aio_nbytes);
  io_destroy(ctx);

  ElementType sum = 0;
  int times = 0;
  for (times = 0; times * times < nsize; times++)
  for (int k = 0; k < nsize; k++)
    for (int i = 0; i < nsize; i++) {
      ElementType x = arr_A[k * nsize + i];
      acc_A.read(ptr_t(offset + k * nsize + i));
      ElementType y = acc_B.read(ptr_t(i));
      sum += x * y;
    }
  //assert(fabs(sum/times - 0.01 * 0.88 * nsize * nsize) < 1e-4);
  free(arr_A);
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);
  HighLevelRuntime::register_legion_task<compute_task>(COMPUTE_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);
  HighLevelRuntime::register_legion_task<compute_from_file_task>(COMPUTE_FROM_FILE_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/);

  // Register custom mappers
  HighLevelRuntime::set_registration_callback(update_mappers);
  return HighLevelRuntime::start(argc, argv);
}
