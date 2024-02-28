/* Copyright 2024 Stanford University
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

// this test makes use of lots of deprecated Legion API calls - ignore for now
#define LEGION_DEPRECATED(x)

#include "legion.h"
#include "shim_mapper.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_TASK_ID,
  MAIN_TASK_ID,
  WORKER_TASK_ID,
};

enum FieldIDs {
  FID_VAL,
};

enum SerdezIDs {
  SERDEZ_ID = 123
};

enum ColorConstant {
  RED = 111,
  GREEN = 222,
  BLUE = 333
};

int num_cpus = 0;

class RenderingMapper : public ShimMapper {
public:
  RenderingMapper(Machine machine, HighLevelRuntime *runtime, Processor local)
    : ShimMapper(machine, runtime, runtime->get_mapper_runtime(), local)
  {
    std::set<Memory> all_mem;
    std::vector<Memory> sys_mem;
    machine.get_all_memories(all_mem);
    for (std::set<Memory>::iterator it = all_mem.begin(); it != all_mem.end(); it++) {
      if ((it->kind() == Memory::SYSTEM_MEM) && (it->capacity() > 0))
        sys_mem.push_back(*it);
    }
    //printf("num sys_mem = %lu\n", sys_mem.size());
    std::set<Processor> all_procs;
    machine.get_all_processors(all_procs);
    for (std::set<Processor>::iterator it = all_procs.begin(); it != all_procs.end(); it++) {
      if (it->kind() == Processor::LOC_PROC) {
        worker_cpu_procs.push_back(*it);
      }
    }
    num_cpus = worker_cpu_procs.size();
    //printf("cpu size = %lu\n", worker_cpu_procs.size());
  }

  using ShimMapper::select_task_options;
  virtual void select_task_options(Task *task)
  {
    ShimMapper::select_task_options(task);
    task->inline_task = false;
    task->spawn_task = false;
    task->map_locally = true;
    task->profile_task = false;
    //if (task->task_id == TOP_LEVEL_TASK_ID
     //|| task->task_id == MAIN_TASK_ID
     //|| task->task_id == INIT_TASK_ID);
    if (task->task_id == WORKER_TASK_ID) {
      //int idx = task->index_point.point_data[0];
      //task->target_proc = worker_cpu_procs[idx % worker_cpu_procs.size()];
      //printf("index = %d, proc ID = %d/%u\n", idx, gasnet_mynode(), task->target_proc.id);
    }
  }

  using ShimMapper::map_task;
  virtual bool map_task(Task *task)
  {
    bool ret = ShimMapper::map_task(task);
    std::set<Memory> vis_mems;
    std::vector<Memory> sys_mem;
    //printf("task_idx = %d, target_proc = %d\n", task->index_point.point_data[0], task->target_proc.id);
    machine.get_visible_memories(task->target_proc, vis_mems);
    for (std::set<Memory>::iterator it = vis_mems.begin(); it != vis_mems.end(); it++) {
      if ((it->kind() == Memory::SYSTEM_MEM) && (it->capacity() > 0))
        sys_mem.push_back(*it);
    }
    assert(sys_mem.size() == 1);
    for (unsigned idx = 0; idx < task->regions.size(); idx++)
    {
      if (task->task_id == WORKER_TASK_ID) {
        for (std::map<Memory,bool>::iterator it = task->regions[idx].current_instances.begin(); it != task->regions[idx].current_instances.end(); it++) {
          //printf("idx = %d, mem = %x, has = %d\n", task->index_point.point_data[0], it->first.id, it->second);
        }
        //printf("final dec = %x\n", sys_mem[0].id);
      }
      task->regions[idx].target_ranking.clear();
      task->regions[idx].target_ranking.push_back(sys_mem[0]);
      assert(task->regions[idx].virtual_map == false);
      task->regions[idx].virtual_map = false;
      // this is not required for correctness, but it forces extra
      //  copies which helps test the serialization stuff
      task->regions[idx].enable_WAR_optimization = true;
      //task->regions[idx].reduction_list = false;
      // Make everything SOA
      //task->regions[idx].blocking_factor =
        //task->regions[idx].max_blocking_factor;
    }
    return ret;
  }
public:
  std::vector<Processor> worker_cpu_procs;
};

static void update_mappers(Machine machine, HighLevelRuntime *rt,
                           const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(new RenderingMapper(machine, rt, *it), *it);
  }
}

struct RGB {
  int r, g, b;
};

class Object {
public:
  enum {dim_size = 32};
  Object(void) {
    for (int i = 0; i < dim_size * dim_size;i++) {
      texture[i].r = texture[i].g = texture[i].b = 0;
    }
  }
public:
  RGB texture[dim_size * dim_size];
};

/*template<typename T>
class SerdezObject {
public:
  typedef T* FIELD_TYPE;

  static size_t serialized_size(const FIELD_TYPE& val) {
    return sizeof(Object);
  }

  static size_t serialize(const FIELD_TYPE& val, void *buffer) {
    memcpy(buffer, val, sizeof(Object));
    return sizeof(Object);
  }

  static size_t deserialize(FIELD_TYPE& val, const void *buffer) {
    val = new Object;
    memcpy(val, buffer, sizeof(Object));
    return sizeof(Object);
  }

  static void destroy(FIELD_TYPE& val) {
    delete val;
  }
};*/

struct Config {
  int nsize, niter, npar, iters_per_check;
  bool cache_optimization;
};

class AverageCounter {
public:
  AverageCounter() {
    count = 0;
    total = 0;
  }
  void add(double new_val) {
    count++;
    total += new_val;
  }
  double result() {
    if (count == 0)
      return 0;
    else
      return total / count;
  }
private:
  int count;
  double total;
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  srand(123456);
  int nsize = 64;
  int niter = 10;
  int npar = num_cpus;
  int iters_per_check = 4;
  bool cache_optimization = false;
  // Check for any command line arguments
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        nsize = atoi(command_args.argv[++i]);
      else if (!strcmp(command_args.argv[i],"-i"))
        niter = atoi(command_args.argv[++i]);
      //else if (!strcmp(command_args.argv[i],"-p"))
      //  npar = atoi(command_args.argv[++i]);
      else if (!strcmp(command_args.argv[i],"-c"))
        cache_optimization = true;
    }
  }
  //make sure number of partitions is equal to number of cpus
  assert(npar == num_cpus);
  
  Rect<1> rect_A(Point<1>(0), Point<1>(nsize - 1));
  IndexSpace is_A = runtime->create_index_space(ctx,
                          Domain::from_rect<1>(rect_A));
  FieldSpace fs_A = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs_A);
    allocator.allocate_field(sizeof(Object*), FID_VAL, SERDEZ_ID);
    //allocator.allocate_field(sizeof(Object*), FID_VAL);
  }
  LogicalRegion lr_A = runtime->create_logical_region(ctx, is_A, fs_A);

  TaskLauncher init_launcher(INIT_TASK_ID,
                             TaskArgument(NULL, 0));
  init_launcher.add_region_requirement(
      RegionRequirement(lr_A, WRITE_DISCARD,
                        EXCLUSIVE, lr_A));
  init_launcher.add_field(0, FID_VAL);
  runtime->execute_task(ctx, init_launcher);

  Config config;
  config.nsize = nsize;
  config.niter = niter;
  config.npar = npar;
  config.iters_per_check = iters_per_check;
  config.cache_optimization = cache_optimization;
  TaskLauncher main_launcher(MAIN_TASK_ID,
                             TaskArgument(&config, sizeof(config)));
  main_launcher.add_region_requirement(
      RegionRequirement(lr_A, READ_ONLY,
                        SIMULTANEOUS, lr_A));
  main_launcher.add_field(0, FID_VAL);
  runtime->execute_task(ctx, main_launcher);
  // Clean up our region, index space, and field space
  runtime->destroy_logical_region(ctx, lr_A);
  runtime->destroy_field_space(ctx, fs_A);
  runtime->destroy_index_space(ctx, is_A);
}

void init_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid_A = *(task->regions[0].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, Object*> acc_A =
    regions[0].get_field_accessor(fid_A).typeify<Object*>();
  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<1> rect_A = dom.get_rect<1>();
  for (GenericPointInRectIterator<1> pir(rect_A); pir; pir++) {
    Object* new_rect = new Object();
    for (int x = 0; x < Object::dim_size; x++)
      for (int y = 0; y < Object::dim_size; y++) {
        new_rect->texture[x * Object::dim_size + y].r = RED;
        new_rect->texture[x * Object::dim_size + y].g = GREEN;
        new_rect->texture[x * Object::dim_size + y].b = BLUE;
      }
    acc_A.write(DomainPoint::from_point<1>(pir.p), new_rect);
  }
}

void main_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->arglen == sizeof(Config));
  LogicalRegion lr_A = task->regions[0].region;
  PhysicalRegion pr_A = regions[0];
  Config config = *((Config*)task->args);
  int niter = config.niter;
  int npar = config.npar;
  int iters_per_check = config.iters_per_check;
  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<1> rect_A = dom.get_rect<1>();
  printf("Running graph rendering with nsize = %zd\n", rect_A.volume());
  printf("Generating iterations = %d\n", niter);
  printf("Num of partitions = %d\n", npar);

  Rect<1> color_bounds(Point<1>(0),Point<1>(npar-1));
  Domain color_domain = Domain::from_rect<1>(color_bounds);
  DomainColoring coloring;
  for (int color = 0; color < npar; color ++) {
    int left = color * (rect_A.volume() / npar);
    int right = left + rect_A.volume() / npar - 1;
    Rect<1> sub_rect(make_point(left), make_point(right));
    coloring[color] = Domain::from_rect<1>(sub_rect);
  }
  IndexPartition ip = runtime->create_index_partition(ctx, lr_A.get_index_space(), color_domain, coloring, true);
  LogicalPartition lp =
    runtime->get_logical_partition(ctx, lr_A, ip);

  // Start Computation
  AverageCounter total_timer, core_timer;
  //double ts_start = Realm::Clock::current_time();
  // Acquire the logical reagion so that we can launch sub-operations that make copies
  if (config.cache_optimization) {
    AcquireLauncher acquire_launcher(lr_A, lr_A, pr_A);
    acquire_launcher.add_field(FID_VAL);
    runtime->issue_acquire(ctx, acquire_launcher);
  }
  for (int iter = 0; iter < niter; iter++) {
    double sub_ts_start = Realm::Clock::current_time();
    if (!config.cache_optimization) {
      AcquireLauncher acquire_launcher(lr_A, lr_A, pr_A);
      acquire_launcher.add_field(FID_VAL);
      runtime->issue_acquire(ctx, acquire_launcher);
    } else if (iter % iters_per_check == 0) {
      // release and reacquire
      ReleaseLauncher release_launcher(lr_A, lr_A, pr_A);
      release_launcher.add_field(FID_VAL);
      runtime->issue_release(ctx, release_launcher);
      AcquireLauncher acquire_launcher(lr_A, lr_A, pr_A);
      acquire_launcher.add_field(FID_VAL);
      runtime->issue_acquire(ctx, acquire_launcher);
    }
    ArgumentMap arg_map;
    IndexLauncher index_launcher(WORKER_TASK_ID, color_domain, TaskArgument(NULL, 0), arg_map);
    index_launcher.add_region_requirement(
        RegionRequirement(lp, 0, READ_ONLY, EXCLUSIVE, lr_A));
    index_launcher.add_field(0, FID_VAL); 
    FutureMap future_map = runtime->execute_index_space(ctx, index_launcher);
    future_map.wait_all_results();
    if (!config.cache_optimization) {
      ReleaseLauncher release_launcher(lr_A, lr_A, pr_A);
      release_launcher.add_field(FID_VAL);
      runtime->issue_release(ctx, release_launcher);
    }
    double sub_ts_end = Realm::Clock::current_time();
    double exec_time = sub_ts_end - sub_ts_start;
    if (config.cache_optimization && iter % iters_per_check != 0)
      core_timer.add(exec_time);
    total_timer.add(exec_time);
    printf("time(%d) = %7.3f s\n", iter, exec_time);
  }
  //Release the attached physicalregion
  if (config.cache_optimization) {
    ReleaseLauncher release_launcher(lr_A, lr_A, pr_A);
    release_launcher.add_field(FID_VAL);
    runtime->issue_release(ctx, release_launcher);
  }
  //double ts_end = Realm::Clock::current_time();

  //double exec_time = ts_end - ts_start;
  //printf("ELAPSED TIME = %7.3f s\n", exec_time);
  if (config.cache_optimization) {
    printf("Legion+ER Time per Iter = %7.3f s\n", total_timer.result());
    printf("Computation Time per Iter = %7.3f s\n", core_timer.result());
  }
  else {
    printf("Baseline Time per Iter = %7.3f s\n", total_timer.result());
  }
}

void worker_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);
  //printf("worker_task: node = %d, idx = %d\n", gasnet_mynode(), task->index_point.point_data[0]);
  FieldID fid_A = *(task->regions[0].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, Object*> acc_A =
    regions[0].get_field_accessor(fid_A).typeify<Object*>();
  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<1> rect_A = dom.get_rect<1>();
  RGB* bg = (RGB*) calloc(Object::dim_size * Object::dim_size, sizeof(RGB));
  for (GenericPointInRectIterator<1> pir(rect_A); pir; pir++) {
    Object* obj = acc_A.read(DomainPoint::from_point<1>(pir.p));
    for (int x = 0; x < Object::dim_size; x++)
      for (int y = 0; y < Object::dim_size; y++) {
        bg[x * Object::dim_size + y].r += obj->texture[x * Object::dim_size + y].r;
        assert(obj->texture[x * Object::dim_size + y].r == RED);
        bg[x * Object::dim_size + y].g += obj->texture[x * Object::dim_size + y].g;
        assert(obj->texture[x * Object::dim_size + y].g == GREEN);
        bg[x * Object::dim_size + y].b += obj->texture[x * Object::dim_size + y].b;
        assert(obj->texture[x * Object::dim_size + y].b == BLUE);
      }
  } 
  free(bg);
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/,
      AUTO_GENERATE_ID, TaskConfigOptions(false/*leaf task*/), "top_level_task");
  HighLevelRuntime::register_legion_task<init_task>(INIT_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf task*/), "init_task");
  HighLevelRuntime::register_legion_task<main_task>(MAIN_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/,
      AUTO_GENERATE_ID, TaskConfigOptions(false/*leaf task*/), "main_task");
  HighLevelRuntime::register_legion_task<worker_task>(WORKER_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf task*/), "worker_task");

  // Register custom mappers
  HighLevelRuntime::set_registration_callback(update_mappers);
  HighLevelRuntime::register_custom_serdez_op<Realm::SerdezObject<Object> >(SERDEZ_ID);
  return HighLevelRuntime::start(argc, argv);
}

