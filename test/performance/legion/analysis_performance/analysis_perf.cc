/* Copyright 2017 NVIDIA Corporation
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

#include "legion.h"
#include "default_mapper.h"

#include <cstdlib>
#include <cstring>
#include <unistd.h>

using namespace std;
using namespace Legion;
using namespace Legion::Mapping;
using namespace LegionRuntime::Arrays;

enum
{
  TOP_LEVEL_TASK_ID,
  DO_NOTHING_TASK_ID,
  BLOCK_TASK_ID,
};

//------------------------------------------------------------------------------
// Mapper
//------------------------------------------------------------------------------

static unsigned num_slices = 1;

class PerfMapper : public DefaultMapper
{
  public:
    PerfMapper(MapperRuntime *rt, Machine machine, Processor local,
               const char *mapper_name,
               vector<Processor>* procs_list,
               //vector<Memory>* sysmems_list,
               //map<Memory, vector<Processor> >* sysmem_local_procs,
               map<Processor, Memory>* proc_sysmems);

    virtual ~PerfMapper();

    virtual Mapper::MapperSyncModel get_mapper_sync_model(void) const;

    virtual void select_task_options(const MapperContext ctx,
                                     const Task&         task,
                                           TaskOptions&  output);

    virtual void default_policy_select_target_processors(MapperContext ctx,
                                                         const Task &task,
                                               vector<Processor> &target_procs);

    virtual void slice_task(const MapperContext      ctx,
                            const Task&              task,
                            const SliceTaskInput&    input,
                                  SliceTaskOutput&   output);

    virtual void map_task(const MapperContext  ctx,
                          const Task&          task,
                          const MapTaskInput&  input,
                                MapTaskOutput& output);

  private:
    typedef vector<vector<PhysicalInstance> > CachedMapping;

  private:
    vector<Processor>& procs_list;
    //vector<Memory>& sysmems_list;
    //map<Memory, vector<Processor> >& sysmem_local_procs;
    map<Processor, Memory>& proc_sysmems;
    // [partition color][task point] --> cached mapping
    vector<vector<CachedMapping> > mapping_cache;
    vector<VariantID> variant_id;
    vector<TaskSlice> slice_cache;
};

PerfMapper::PerfMapper(MapperRuntime *rt, Machine machine, Processor local,
                       const char *mapper_name,
                       vector<Processor>* _procs_list,
                       //vector<Memory>* _sysmems_list,
                       //map<Memory, vector<Processor> >* _sysmem_local_procs,
                       map<Processor, Memory>* _proc_sysmems)
  : DefaultMapper(rt, machine, local, mapper_name),
    procs_list(*_procs_list),
    //sysmems_list(*_sysmems_list),
    //sysmem_local_procs(*_sysmem_local_procs),
    proc_sysmems(*_proc_sysmems),
    variant_id(0)
{
}

PerfMapper::~PerfMapper()
{
}

Mapper::MapperSyncModel PerfMapper::get_mapper_sync_model(void) const
{
  return CONCURRENT_MAPPER_MODEL;
}

void PerfMapper::select_task_options(const MapperContext ctx,
                                     const Task&         task,
                                           TaskOptions&  output)
{
  output.initial_proc = procs_list[0];
  if (task.task_id == TOP_LEVEL_TASK_ID && procs_list.size() > 1)
    output.initial_proc = procs_list[1];
  output.inline_task = false;
  output.stealable = false;
  output.map_locally = false;
}

void PerfMapper::default_policy_select_target_processors(MapperContext ctx,
                                                         const Task &task,
                                                vector<Processor> &target_procs)
{
  target_procs.push_back(task.target_proc);
}

void PerfMapper::slice_task(const MapperContext      ctx,
                            const Task&              task,
                            const SliceTaskInput&    input,
                                  SliceTaskOutput&   output)
{
  if (task.task_id == DO_NOTHING_TASK_ID)
  {
    if (slice_cache.size() == 0)
    {
      Rect<1> dom = input.domain.get_rect<1>();
      assert(dom.volume() >= num_slices);
      size_t block_size = dom.volume() / num_slices;
      assert(block_size > 0);
      Point<1> lo = dom.lo;
      for (unsigned i = 0; i < num_slices; ++i)
      {
        Point<1> hi = lo + make_point(block_size) - Point<1>(1);
        if (i == num_slices - 1) hi = dom.hi;
        slice_cache.push_back(TaskSlice(
              Domain::from_rect<1>(Rect<1>(lo, hi)),
              procs_list[0], false, false));
        lo = hi + Point<1>(1);
      }
    }
    output.slices = slice_cache;
    output.verify_correctness = false;
  }
  else
    DefaultMapper::slice_task(ctx, task, input, output);
}

void PerfMapper::map_task(const MapperContext  ctx,
                          const Task&          task,
                          const MapTaskInput&  input,
                                MapTaskOutput& output)
{
  if (task.task_id == DO_NOTHING_TASK_ID)
  {
    if (variant_id.size() == 0)
    {
      runtime->find_valid_variants(ctx, DO_NOTHING_TASK_ID,
          variant_id, Processor::LOC_PROC);
      assert(variant_id.size() == 1);
    }
    output.task_priority = 0;
    output.chosen_variant = variant_id[0];
    output.postmap_task = false;
    output.target_procs.push_back(procs_list[0]);
    if (task.regions.size() == 0) return;

    const RegionRequirement& req = task.regions[0];
    size_t point = runtime->get_logical_region_color_point(ctx, req.region)[0];
    size_t part_id = runtime->get_logical_partition_color(ctx,
        runtime->get_parent_logical_partition(ctx, req.region));
    if (mapping_cache.size() <= part_id) mapping_cache.resize(part_id + 1);
    vector<CachedMapping>& mappings = mapping_cache[part_id];
    if (mappings.size() <= point) mappings.resize(point + 1);
    CachedMapping& cached_mapping = mappings[point];
    if (cached_mapping.size() == 0)
    {
      cached_mapping.resize(task.regions.size());
      Memory target_memory = proc_sysmems[procs_list[0]];
      for (size_t idx = 0; idx < task.regions.size(); ++idx)
      {
        PhysicalInstance inst;
        vector<LogicalRegion> target_region;
        target_region.push_back(task.regions[idx].region);
        LayoutConstraintSet constraints;
        std::vector<DimensionKind> dimension_ordering(4);
        dimension_ordering[0] = DIM_X;
        dimension_ordering[1] = DIM_Y;
        dimension_ordering[2] = DIM_Z;
        dimension_ordering[3] = DIM_F;
        constraints.add_constraint(MemoryConstraint(target_memory.kind()))
          .add_constraint(FieldConstraint(req.instance_fields, false, false))
          .add_constraint(OrderingConstraint(dimension_ordering, false));
        runtime->create_physical_instance(ctx, target_memory,
              constraints, target_region, inst);
        runtime->set_garbage_collection_priority(ctx, inst, GC_NEVER_PRIORITY);
        cached_mapping[idx].push_back(inst);
      }
    }
    else
    {
#ifdef DEBUG_LEGION
      bool ok =
#endif
        runtime->acquire_and_filter_instances(ctx, cached_mapping);
#ifdef DEBUG_LEGION
      assert(ok);
#endif
    }

    output.chosen_instances = cached_mapping;
  }
  else
    DefaultMapper::map_task(ctx, task, input, output);
}

static void register_mappers(Machine machine, Runtime *runtime,
                             const set<Processor> &local_procs)
{
  vector<Processor>* procs_list = new vector<Processor>();
  //vector<Memory>* sysmems_list = new vector<Memory>();
  //map<Memory, vector<Processor> >* sysmem_local_procs =
  //  new map<Memory, vector<Processor> >();
  ::map<Processor, Memory>* proc_sysmems = new map<Processor, Memory>();

  vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
  machine.get_proc_mem_affinity(proc_mem_affinities);

  for (unsigned idx = 0; idx < proc_mem_affinities.size(); ++idx) {
    Machine::ProcessorMemoryAffinity& affinity = proc_mem_affinities[idx];
    if (affinity.p.kind() == Processor::LOC_PROC) {
      if (affinity.m.kind() == Memory::SYSTEM_MEM) {
        (*proc_sysmems)[affinity.p] = affinity.m;
      }
    }
  }

  for (map<Processor, Memory>::iterator it = proc_sysmems->begin();
       it != proc_sysmems->end(); ++it) {
    procs_list->push_back(it->first);
    //(*sysmem_local_procs)[it->second].push_back(it->first);
  }

  //for (map<Memory, vector<Processor> >::iterator it =
  //      sysmem_local_procs->begin(); it != sysmem_local_procs->end(); ++it)
  //  sysmems_list->push_back(it->first);

  for (set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    PerfMapper* mapper = new PerfMapper(runtime->get_mapper_runtime(),
                                        machine, *it, "perf_test_mapper",
                                        procs_list,
                                        //sysmems_list,
                                        //sysmem_local_procs,
                                        proc_sysmems);
    runtime->replace_default_mapper(mapper, *it);
  }
}

//------------------------------------------------------------------------------
// Projection Functor
//------------------------------------------------------------------------------
class TreeTraversingFunctor : public ProjectionFunctor
{
  public:
    TreeTraversingFunctor(int depth);
    TreeTraversingFunctor(Runtime* rt, int depth);
    virtual ~TreeTraversingFunctor();

    virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                  LogicalPartition upper_bound,
                                  const DomainPoint &point);
    virtual unsigned get_depth(void) const;

  private:
    int depth;
};

TreeTraversingFunctor::TreeTraversingFunctor(int _depth)
  : ProjectionFunctor(), depth(_depth)
{
}

TreeTraversingFunctor::TreeTraversingFunctor(Runtime* rt, int _depth)
  : ProjectionFunctor(rt), depth(_depth)
{
}

TreeTraversingFunctor::~TreeTraversingFunctor()
{
}

LogicalRegion TreeTraversingFunctor::project(const Mappable *mappable,
                                             unsigned index,
                                             LogicalPartition upper_bound,
                                             const DomainPoint &point)
{
  LogicalRegion lr;
  LogicalPartition lp = upper_bound;
  Color part_color = runtime->get_logical_partition_color(lp);
  for (int d = 0; d < depth; ++d)
  {
    lr = runtime->get_logical_subregion_by_color(lp, point);
    if (d + 1 < depth)
      lp = runtime->get_logical_partition_by_color(lr, part_color);
  }
  return lr;
}

unsigned TreeTraversingFunctor::get_depth(void) const
{
  return depth - 1;
}

//------------------------------------------------------------------------------
// Tasks
//------------------------------------------------------------------------------

void block(const Task *task,
           const vector<PhysicalRegion> &regions,
           Context ctx, Runtime *runtime)
{
  fprintf(stderr, "Sleeping for 10 seconds...\n");
  sleep(10);
}

void do_nothing(const Task *task,
                const vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
}

static void parse_arguments(char** argv, int argc, unsigned& num_tasks,
                            unsigned& num_loops, unsigned& num_regions,
                            unsigned& num_partitions, unsigned& tree_depth,
                            unsigned& num_fields, bool& alternate,
                            bool& single_launch, bool& block)
{
  int i = 1;
  while (i < argc)
  {
    if (strcmp(argv[i], "-n") == 0) num_tasks = atoi(argv[++i]);
    else if (strcmp(argv[i], "-l") == 0) num_loops = atoi(argv[++i]);
    else if (strcmp(argv[i], "-r") == 0) num_regions = atoi(argv[++i]);
    else if (strcmp(argv[i], "-p") == 0) num_partitions = atoi(argv[++i]);
    else if (strcmp(argv[i], "-d") == 0) tree_depth = atoi(argv[++i]);
    else if (strcmp(argv[i], "-f") == 0) num_fields = atoi(argv[++i]);
    else if (strcmp(argv[i], "-a") == 0) alternate = true;
    else if (strcmp(argv[i], "-s") == 0) single_launch = true;
    else if (strcmp(argv[i], "-b") == 0) block = true;
    ++i;
  }
}

void create_index_partitions(Context ctx, Runtime *runtime, IndexSpace is,
                             int fanout, int part_color, bool alternate,
                             int depth, int max_depth)
{
  if (depth == max_depth) return;
  IndexPartition ip;
  Rect<1> rect = runtime->get_index_space_domain(ctx, is).get_rect<1>();
  size_t num_elmts = rect.volume();
  assert(num_elmts > 0);
  size_t block_size = num_elmts / fanout;
  assert(block_size > 0);
  if (alternate && part_color % 2 == 1)
  {
    Domain color_space =
      Domain::from_rect<1>(Rect<1>(make_point(0), make_point(fanout - 1)));
    DomainPointColoring coloring;
    Point<1> start = rect.lo;
    for (int i = 0; i < fanout; ++i)
    {
      Point<1> end = start + make_point(block_size);
      coloring[DomainPoint::from_point<1>(make_point(i))] =
        Domain::from_rect<1>(Rect<1>(
              Point<1>::max(start, rect.lo),
              Point<1>::min(rect.hi, end)));
      start = end - Point<1>(1);
    }
    ip = runtime->create_index_partition(ctx, is, color_space, coloring,
      ALIASED_KIND, part_color);
  }
  else
  {
    Blockify<1> blockify(num_elmts / fanout, rect.lo);
    ip = runtime->create_index_partition(ctx, is, blockify, part_color);
  }

  for (int i = 0; i < fanout; ++i)
  {
    IndexSpace sis = runtime->get_index_subspace(ctx, ip,
        DomainPoint::from_point<1>(make_point(i)));
    create_index_partitions(ctx, runtime, sis, fanout, part_color, alternate,
        depth + 1, max_depth);
  }
}

void top_level_task(const Task *task,
                    const vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  unsigned num_tasks = 1;
  unsigned num_loops = 10;
  unsigned num_regions = 1;
  unsigned num_partitions = 1;
  unsigned tree_depth = 1;
  unsigned num_fields = 1;
  bool alternate = false;
  bool single_launch = false;
  bool block = false;

  {
    const InputArgs &command_args = Runtime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_arguments(argv, argc, num_tasks, num_loops, num_regions, num_partitions,
        tree_depth, num_fields, alternate, single_launch, block);
    if (num_regions == 0) num_partitions = 1;
    if (num_regions > 0 && num_partitions > 0 && tree_depth == 0)
    {
      fprintf(stderr,
          "The depth of tree should be greater than 0 "
          "if partitions are being used.\n");
      exit(-1);
    }
    if (!single_launch && num_partitions == 0)
    {
      fprintf(stderr,
          "Index launch cannot be used if there is no partition.\n");
      exit(-1);
    }
    if (num_tasks < num_slices)
    {
      fprintf(stderr,
          "Number of slices cannot be greater than number of tasks.\n");
      exit(-1);
    }
  }

  // TODO: Single task launches with root regions should be supported
  if (num_partitions == 0)
    fprintf(stderr,
        "Single task launches with root regions are not supported yet.\n");
  assert(num_partitions > 0);

  printf("***************************************\n");
  printf("* Runtime Analysis Performance Test   *\n");
  printf("*                                     *\n");
  printf("* Number of Tasks       :       %5d *\n", num_tasks);
  printf("* Number of Iterations  :       %5d *\n", num_loops);
  printf("* Number of Regions     :       %5d *\n", num_regions);
  printf("* Number of Partitions per Region : %1d *\n", num_partitions);
  printf("* Depth of Region Trees :       %5d *\n", tree_depth);
  printf("* Number of Fields      :       %5d *\n", num_fields);
  printf("* Alternate disjoint/aliased :    %s *\n", alternate ? "yes" : " no");
  printf("* Use Single Launch     :         %s *\n", single_launch ? "yes" : " no");
  printf("* Number of Slices      :       %5d *\n", num_slices);
  printf("***************************************\n");

  Domain launch_domain =
    Domain::from_rect<1>(Rect<1>(make_point(0), make_point(num_tasks - 1)));
  vector<LogicalRegion> lrs;
  vector<vector<LogicalPartition> > lps;
  ProjectionID pid = 0;

  if (num_regions > 0)
  {
    coord_t num_elmts = 1;
    for (unsigned d = 0; d < tree_depth; ++d) num_elmts *= num_tasks;

    FieldSpace fs = runtime->create_field_space(ctx);
    {
      FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
      for (unsigned i = 0; i < num_fields; ++i)
        allocator.allocate_field(sizeof(int), 100 + i);
    }

    Domain region_domain =
      Domain::from_rect<1>(Rect<1>(make_point(1), make_point(num_elmts)));
    IndexSpace is = runtime->create_index_space(ctx, region_domain);

    for (unsigned p = 0; p < num_partitions; ++p)
      create_index_partitions(ctx, runtime, is, num_tasks, p, alternate, 0,
          tree_depth);

    for (unsigned i = 0; i < num_regions; ++i)
      lrs.push_back(runtime->create_logical_region(ctx, is, fs));

    for (unsigned i = 0; i < num_regions; ++i)
    {
      lps.push_back(vector<LogicalPartition>());
      for (unsigned p = 0; p < num_partitions; ++p)
        lps[i].push_back(
            runtime->get_logical_partition_by_color(ctx, lrs[i], p));
    }

    pid = 1;
    runtime->register_projection_functor(pid,
        new TreeTraversingFunctor(runtime, tree_depth));
  }

  PhaseBarrier barrier = runtime->create_phase_barrier(ctx, 1);
  PhaseBarrier next_barrier = runtime->advance_phase_barrier(ctx, barrier);

  PhaseBarrier barrier_for_block = runtime->create_phase_barrier(ctx, 1);
  PhaseBarrier next_barrier_for_block = runtime->advance_phase_barrier(ctx, barrier_for_block);

  if (block)
  {
    TaskLauncher launcher(BLOCK_TASK_ID, TaskArgument());
    launcher.add_wait_barrier(next_barrier_for_block);
    launcher.add_arrival_barrier(barrier);
    runtime->execute_task(ctx, launcher);
  }

  if (single_launch)
  {
    for (unsigned l = 0; l < num_loops; ++l)
    {
      for (unsigned p = 0; p < num_partitions; ++p)
      {
        PrivilegeMode priv = alternate && p % 2 == 1 ? READ_ONLY : READ_WRITE;
        for (unsigned i = 0; i < num_tasks; ++i)
        {
          TaskLauncher launcher(DO_NOTHING_TASK_ID, TaskArgument());
          if (block && l == 0 && p == 0) launcher.add_wait_barrier(next_barrier);
          DomainPoint dp = DomainPoint::from_point<1>(make_point(i));
          for (unsigned r = 0; r < num_regions; ++r)
          {
            LogicalPartition lp = lps[r][p];
            LogicalRegion lr;
            for (unsigned d = 0; d < tree_depth; ++d)
            {
              lr = runtime->get_logical_subregion_by_color(ctx, lp, dp);
              if (d + 1 < tree_depth)
                lp = runtime->get_logical_partition_by_color(ctx, lr, p);
            }

            RegionRequirement req(lr, priv, EXCLUSIVE, lrs[r]);
            for (unsigned k = 0; k < num_fields; ++k) req.add_field(100 + k);
            launcher.add_region_requirement(req);
          }

          runtime->execute_task(ctx, launcher);
        }
      }
    }
  }
  else
  {
    for (unsigned l = 0; l < num_loops; ++l)
    {
      for (unsigned p = 0; p < num_partitions; ++p)
      {
        PrivilegeMode priv = alternate && p % 2 == 1 ? READ_ONLY : READ_WRITE;
        IndexTaskLauncher launcher(DO_NOTHING_TASK_ID, launch_domain,
                                   TaskArgument(), ArgumentMap());
        if (block && l == 0 && p == 0) launcher.add_wait_barrier(next_barrier);
        for (unsigned r = 0; r < num_regions; ++r)
        {
          RegionRequirement req(lps[r][p], pid, priv, EXCLUSIVE, lrs[r]);
          for (unsigned k = 0; k < num_fields; ++k) req.add_field(100 + k);
          launcher.add_region_requirement(req);
        }

        runtime->execute_index_space(ctx, launcher);
      }
    }
  }
  barrier_for_block.arrive(1);
}

static void parse_num_slices(char** argv, int argc)
{
  int i = 1;
  while (i < argc)
  {
    if (strcmp(argv[i], "-S") == 0) num_slices = atoi(argv[++i]);
    ++i;
  }
}

int main(int argc, char** argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(DO_NOTHING_TASK_ID, "do_nothing");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<do_nothing>(registrar, "do_nothing");
  }
  {
    TaskVariantRegistrar registrar(BLOCK_TASK_ID, "block");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<block>(registrar, "block");
  }

  parse_num_slices(argv, argc);
  Runtime::add_registration_callback(register_mappers);

  return Runtime::start(argc, argv);
}
