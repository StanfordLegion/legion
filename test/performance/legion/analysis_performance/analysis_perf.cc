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

enum
{
  RO = 1,
  WO = 2,
  RW = 3,
  RD = 4,
};

//------------------------------------------------------------------------------
// Command-line Parser
//------------------------------------------------------------------------------
#define expect(q, c) \
{ \
  const char* __p = (q); \
  if (*__p != (c)) { \
    fprintf(stderr, "Ill-formed pattern\n"); \
    exit(-1); \
  } \
} \


static void parse_pattern(const char* pattern_string, vector<int>& pattern)
{
  const char* p = pattern_string;
  while (*p != '\0')
  {
    if (*p == 'w')
    {
      expect(++p, 'o');
      pattern.push_back(WO);
    }
    else if(*p++ == 'r')
    {
      if (*p == 'd') pattern.push_back(RD);
      else if (*p == 'w') pattern.push_back(RW);
      else if (*p == 'o') pattern.push_back(RO);
      else
      {
        fprintf(stderr, "Ill-formed pattern\n");
        exit(-1);
      }
    }

    if (*++p != '\0') expect(p++, '-');
  }
  if (pattern.empty())
  {
    fprintf(stderr, "ERROR: Empty alternation pattern.\n");
    exit(-1);
  }
}

static void parse_arguments(char** argv, int argc, unsigned &num_tasks,
                            unsigned &num_loops, unsigned &num_regions,
                            unsigned &num_partitions, unsigned &num_slices,
                            unsigned &tree_depth, unsigned &num_fields,
                            unsigned &dims, unsigned &blast, unsigned &slide,
                            bool &alternate, bool &alternate_loop,
                            bool &single_launch, bool &block,
                            bool &cache_mapping, bool &tracing,
                            vector<int> &pattern)
{
  int i = 1;
  while (i < argc)
  {
    if (strcmp(argv[i], "-n") == 0) num_tasks = atoi(argv[++i]);
    else if (strcmp(argv[i], "-l") == 0) num_loops = atoi(argv[++i]);
    else if (strcmp(argv[i], "-r") == 0) num_regions = atoi(argv[++i]);
    else if (strcmp(argv[i], "-p") == 0) num_partitions = atoi(argv[++i]);
    else if (strcmp(argv[i], "-S") == 0) num_slices = atoi(argv[++i]);
    else if (strcmp(argv[i], "-d") == 0) tree_depth = atoi(argv[++i]);
    else if (strcmp(argv[i], "-f") == 0) num_fields = atoi(argv[++i]);
    else if (strcmp(argv[i], "-D") == 0) dims = atoi(argv[++i]);
    else if (strcmp(argv[i], "-B") == 0) blast = atoi(argv[++i]);
    else if (strcmp(argv[i], "-L") == 0) slide = atoi(argv[++i]);
    else if (strcmp(argv[i], "-a") == 0) alternate = true;
    else if (strcmp(argv[i], "-A") == 0) alternate_loop = true;
    else if (strcmp(argv[i], "-s") == 0) single_launch = true;
    else if (strcmp(argv[i], "-b") == 0) block = true;
    else if (strcmp(argv[i], "-F") == 0) cache_mapping = false;
    else if (strcmp(argv[i], "-T") == 0) tracing = true;
    else if (strcmp(argv[i], "-P") == 0) parse_pattern(argv[++i], pattern);
    ++i;
  }
  if (pattern.size() == 0)
    parse_pattern("rw-rd-rd-ro-rd-ro-rw-rd-ro", pattern);
}

//------------------------------------------------------------------------------
// Reduction Operator
//------------------------------------------------------------------------------
enum
{
  REDUCE_ID = 1,
};

class ReduceNothing {
public:
  typedef float LHS;
  typedef float RHS;
  static const float identity;

  template <bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
  template <bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

const float ReduceNothing::identity = 0.0f;

template <>
void ReduceNothing::apply<true>(LHS &lhs, RHS rhs)
{
}

template<>
void ReduceNothing::apply<false>(LHS &lhs, RHS rhs)
{
}

template <>
void ReduceNothing::fold<true>(RHS &rhs1, RHS rhs2)
{
}

template<>
void ReduceNothing::fold<false>(RHS &rhs1, RHS rhs2)
{
}

//------------------------------------------------------------------------------
// Mapper
//------------------------------------------------------------------------------

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

    virtual bool default_policy_select_close_virtual(const MapperContext ctx,
                                                     const Close &close);

  private:
    typedef vector<vector<PhysicalInstance> > CachedMapping;
    typedef vector<vector<LayoutConstraintSet> > CachedConstraints;

  private:
    unsigned num_slices;
    bool cache_mapping;
    bool tracing;
    unsigned skip_count;
    vector<Processor>& procs_list;
    //vector<Memory>& sysmems_list;
    //map<Memory, vector<Processor> >& sysmem_local_procs;
    map<Processor, Memory>& proc_sysmems;
    // [partition color][task point] --> cached mapping
    vector<vector<CachedMapping> > mapping_cache;
    vector<vector<CachedConstraints> > constraint_cache;
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
    num_slices(1),
    cache_mapping(true),
    tracing(false),
    skip_count(1),
    procs_list(*_procs_list),
    //sysmems_list(*_sysmems_list),
    //sysmem_local_procs(*_sysmem_local_procs),
    proc_sysmems(*_proc_sysmems),
    variant_id(0)
{
  unsigned num_tasks = 1;
  unsigned num_loops = 10;
  unsigned num_regions = 1;
  unsigned num_partitions = 1;
  unsigned tree_depth = 1;
  unsigned num_fields = 1;
  unsigned dims = 1;
  unsigned blast = 1;
  unsigned slide = 0;
  bool alternate = false;
  bool alternate_loop = false;
  bool single_launch = false;
  bool block = false;
  vector<int> pattern;

  const InputArgs &command_args = Runtime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;

  parse_arguments(argv, argc, num_tasks, num_loops, num_regions,
      num_partitions, num_slices, tree_depth, num_fields, dims, blast, slide,
      alternate, alternate_loop, single_launch, block, cache_mapping,
      tracing, pattern);

  if (tracing && !cache_mapping)
  {
    fprintf(stderr,
        "WARNING: Only the first mapping decision will be effective "
        "because of the physical tracing\n");
  }
}

PerfMapper::~PerfMapper()
{
}

Mapper::MapperSyncModel PerfMapper::get_mapper_sync_model(void) const
{
  return SERIALIZED_REENTRANT_MAPPER_MODEL;
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
  if (task.task_id == DO_NOTHING_TASK_ID)
    output.memoize = tracing && task.regions[0].tag >= skip_count;
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
      Domain dom = input.domain;
      assert(dom.get_volume() >= num_slices);
      size_t block_size = dom.get_volume() / num_slices;
      assert(block_size > 0);
      int dim = dom.dim;
      coord_t lo = dom.rect_data[0];
      for (unsigned i = 0; i < num_slices; ++i)
      {
        coord_t hi = lo + block_size - 1;
        if (i == num_slices - 1) hi = dom.rect_data[dim];
        Domain slice;
        slice.dim = dim;
        for (int k = 0; k < dim * 2; ++k) slice.rect_data[k] = 0;
        slice.rect_data[0] = lo;
        slice.rect_data[dim] = hi;
        slice_cache.push_back(TaskSlice(slice, procs_list[0], false, false));
        lo = hi + 1;
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
      std::vector<VariantID> local_variant_id;
      runtime->find_valid_variants(ctx, DO_NOTHING_TASK_ID,
          local_variant_id, Processor::LOC_PROC);
      if (variant_id.size() == 0)
        variant_id = local_variant_id;
#ifdef DEBUG_LEGION
      else
        assert(variant_id == local_variant_id);
#endif
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
    bool has_reduction = req.privilege == REDUCE;
    if (has_reduction)
    {
      if (constraint_cache.size() <= part_id)
        constraint_cache.resize(part_id + 1);
      vector<CachedConstraints>& cons_for_partition =
        constraint_cache[part_id];
      if (cons_for_partition.size() <= point)
        cons_for_partition.resize(point + 1);
      CachedConstraints cons_for_task = cons_for_partition[point];
      Memory target_memory = proc_sysmems[procs_list[0]];
      if (cons_for_task.size() == 0)
      {
        for (size_t idx = 0; idx < task.regions.size(); ++idx)
        {
          cons_for_task.push_back(vector<LayoutConstraintSet>());
          size_t num_fields = task.regions[idx].instance_fields.size();
          for (size_t fidx = 0; fidx < num_fields; ++fidx)
          {
            LayoutConstraintSet constraints;
            std::vector<FieldID> fields(1,
                task.regions[idx].instance_fields[fidx]);
            constraints.add_constraint(SpecializedConstraint(
                  REDUCTION_FOLD_SPECIALIZE, task.regions[idx].redop))
              .add_constraint(FieldConstraint(fields, false, false))
              .add_constraint(MemoryConstraint(target_memory.kind()));
            cons_for_task.back().push_back(constraints);
          }
        }
      }
      for (size_t idx = 0; idx < task.regions.size(); ++idx)
      {
        size_t num_constraints = cons_for_task[idx].size();
        for (size_t cidx = 0; cidx < num_constraints; ++cidx)
        {
          PhysicalInstance inst;
          vector<LogicalRegion> target_region;
          target_region.push_back(task.regions[idx].region);
          runtime->create_physical_instance(ctx, target_memory,
              cons_for_task[idx][cidx], target_region, inst);
          runtime->set_garbage_collection_priority(ctx, inst, GC_FIRST_PRIORITY);
          output.chosen_instances[idx].push_back(inst);
        }
      }
    }
    else
    {
      if (mapping_cache.size() <= part_id) mapping_cache.resize(part_id + 1);
      vector<CachedMapping>& mappings = mapping_cache[part_id];
      if (mappings.size() <= point) mappings.resize(point + 1);
      CachedMapping cached_mapping = mappings[point];
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
            .add_constraint(FieldConstraint(
                  task.regions[idx].instance_fields, false, false))
            .add_constraint(OrderingConstraint(dimension_ordering, false));
          runtime->create_physical_instance(ctx, target_memory,
                constraints, target_region, inst);
          runtime->set_garbage_collection_priority(ctx, inst,
              cache_mapping ? GC_NEVER_PRIORITY : GC_FIRST_PRIORITY);
          cached_mapping[idx].push_back(inst);
        }
        if (cache_mapping) mapping_cache[part_id][point] = cached_mapping;
      }
      else
      {
#ifdef DEBUG_LEGION
        bool ok =
#endif
          runtime->acquire_instances(ctx, cached_mapping);
#ifdef DEBUG_LEGION
        assert(ok);
#endif
      }
      output.chosen_instances = cached_mapping;
    }
  }
  else
    DefaultMapper::map_task(ctx, task, input, output);
}

bool PerfMapper::default_policy_select_close_virtual(const MapperContext ctx,
                                                     const Close &close)
{
  return true;
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

template<int DIM>
void create_index_partitions(Context ctx, Runtime *runtime, IndexSpace is,
                             int fanout, int part_color, bool alternate,
                             int depth, int max_depth,
                             const vector<int>& pattern)
{
  if (depth == max_depth) return;
  IndexPartition ip;
  Rect<DIM> rect = runtime->get_index_space_domain(ctx, is).get_rect<DIM>();
  size_t num_elmts = rect.volume();
  assert(num_elmts > 0);
  size_t block_size = num_elmts / fanout;
  assert(block_size > 0);
  if (alternate && (pattern[part_color % pattern.size()] & WO) == 0)
  {
    Point<DIM> colors;
    colors.x[0] = fanout - 1;
    for (unsigned idx = 1; idx < DIM; ++idx) colors.x[idx] = 0;

    Domain color_space =
      Domain::from_rect<DIM>(Rect<DIM>(Point<DIM>::ZEROES(), colors));
    DomainPointColoring coloring;
    Point<DIM> start = rect.lo;
    Point<DIM> block;
    block.x[0] = block_size;
    for (unsigned idx = 1; idx < DIM; ++idx) block.x[idx] = 0;
    Point<DIM> one;
    one.x[0] = 1;
    for (unsigned idx = 1; idx < DIM; ++idx) one.x[idx] = 0;
    for (int i = 0; i < fanout; ++i)
    {
      Point<DIM> end = start + block;
      Point<DIM> color;
      color.x[0] = i;
      for (unsigned idx = 1; idx < DIM; ++idx) color.x[idx] = 0;
      coloring[DomainPoint::from_point<DIM>(color)] =
        Domain::from_rect<DIM>(Rect<DIM>(
              Point<DIM>::max(start, rect.lo),
              Point<DIM>::min(rect.hi, end)));
      start = end - one;
    }
    ip = runtime->create_index_partition(ctx, is, color_space, coloring,
      ALIASED_KIND, part_color);
  }
  else
  {
    Point<DIM> block;
    block.x[0] = num_elmts / fanout;
    for (unsigned idx = 1; idx < DIM; ++idx) block.x[idx] = 1;
    Blockify<DIM> blockify(block, rect.lo);
    ip = runtime->create_index_partition(ctx, is, blockify, part_color);
  }

  for (int i = 0; i < fanout; ++i)
  {
    Point<DIM> color;
    color.x[0] = i;
    for (unsigned idx = 1; idx < DIM; ++idx) color.x[idx] = 0;
    IndexSpace sis = runtime->get_index_subspace(ctx, ip,
        DomainPoint::from_point<DIM>(color));
    create_index_partitions<DIM>(ctx, runtime, sis, fanout, part_color,
        alternate, depth + 1, max_depth, pattern);
  }
}

static void print_pattern(const vector<int>& pattern)
{
  string str;
  for (vector<int>::const_iterator it = pattern.begin(); it != pattern.end();
       ++it)
  {
    if (*it == RO) str += "ro";
    else if (*it == WO) str += "wo";
    else if (*it == RW) str += "rw";
    else if (*it == RD) str += "rd";
    if (it + 1 != pattern.end()) str += "-";
  }
  string pad(max(25 - ((int64_t)pattern.size() * 3 - 1), (int64_t)0), ' ');
  printf("* Pattern : %s%s *\n", pad.c_str(), str.c_str());
}

void top_level_task(const Task *task,
                    const vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  unsigned num_tasks = 1;
  unsigned num_loops = 10;
  unsigned num_regions = 1;
  unsigned num_partitions = 1;
  unsigned num_slices = 1;
  unsigned tree_depth = 1;
  unsigned num_fields = 1;
  unsigned dims = 1;
  unsigned blast = 1;
  unsigned slide = 0;
  bool alternate = false;
  bool alternate_loop = false;
  bool single_launch = false;
  bool block = false;
  bool cache_mapping = true;
  bool tracing = false;
  vector<int> pattern;

  {
    const InputArgs &command_args = Runtime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_arguments(argv, argc, num_tasks, num_loops, num_regions,
        num_partitions, num_slices, tree_depth, num_fields, dims, blast, slide,
        alternate, alternate_loop, single_launch, block, cache_mapping,
        tracing, pattern);
    if (num_regions == 0) num_partitions = 1;
    if (num_regions > 0 && num_partitions > 0 && tree_depth == 0)
    {
      fprintf(stderr,
          "ERROR: The depth of tree should be greater than 0 "
          "if partitions are being used.\n");
      exit(-1);
    }
    if (!single_launch && num_partitions == 0)
    {
      fprintf(stderr,
          "ERROR: Index launch cannot be used if there is no partition.\n");
      exit(-1);
    }
    if (num_tasks < num_slices)
    {
      fprintf(stderr,
          "ERROR: Number of slices cannot be greater than number of tasks.\n");
      exit(-1);
    }
    if (dims > 3)
    {
      fprintf(stderr, "ERROR: Dimensionality should be 1D, 2D, or 3D.\n");
      exit(-1);
    }
    if (num_fields % blast != 0)
    {
      fprintf(stderr,
          "ERROR: Number of fields should be divisible by blast factor.\n");
      exit(-1);
    }
    if (alternate && alternate_loop)
    {
      fprintf(stderr, "ERROR: Two alternate modes cannot coexist.\n");
      exit(-1);
    }
    if (block && tracing)
    {
      fprintf(stderr, "ERROR: Tracing cannot be used in the blocking mode.\n");
      exit(-1);
    }
  }

  if (tracing && alternate_loop && num_loops % pattern.size() != 0)
  {
    fprintf(stderr,
        "WARNING: Rounding up the number of loops "
        "to the closest multiple of the pattern size\n");
    num_loops += pattern.size() - num_loops % pattern.size();
    assert(num_loops % pattern.size() == 0);
  }

  // TODO: Single task launches with root regions should be supported
  if (num_partitions == 0)
    fprintf(stderr,
        "Single task launches with root regions are not supported yet.\n");
  assert(num_partitions > 0);

  printf("***************************************\n");
  printf("* Runtime Analysis Performance Test   *\n");
  printf("*                                     *\n");
  printf("* Number of Tasks       :       %5u *\n", num_tasks);
  printf("* Number of Iterations  :       %5u *\n", num_loops);
  printf("* Number of Regions     :       %5u *\n", num_regions);
  printf("* Number of Partitions per Region : %1d *\n", num_partitions);
  printf("* Depth of Region Trees :       %5u *\n", tree_depth);
  printf("* Number of Fields      :       %5u *\n", num_fields);
  printf("* Alternate Partitions  :         %s *\n", alternate ? "yes" : " no");
  printf("* Alternate Iterations  :         %s *\n",
      alternate_loop ? "yes" : " no");
  if (alternate || alternate_loop) print_pattern(pattern);
  printf("* Use Single Launch     :         %s *\n",
      single_launch ? "yes" : " no");
  printf("* Cache Mapping         :         %s *\n",
      cache_mapping ? "yes" : " no");
  printf("* Block until Analyze   :         %s *\n", block ? "yes" : " no");
  printf("* Tracing               :         %s *\n", tracing ? "yes" : " no");
  printf("* Number of Slices      :       %5u *\n", num_slices);
  printf("* Dimensionality        :       %5u *\n", dims);
  printf("* Blast Factor          :       %5u *\n", blast);
  printf("* Sliding Factor        :       %5u *\n", slide);
  printf("***************************************\n");

  Domain launch_domain;
  switch (dims)
  {
    case 1 :
      {
        launch_domain =
          Domain::from_rect<1>(Rect<1>(make_point(0),
                                       make_point(num_tasks - 1)));
        break;
      }
    case 2 :
      {
        launch_domain =
          Domain::from_rect<2>(Rect<2>(make_point(0, 0),
                                       make_point(num_tasks - 1, 0)));
        break;
      }
    case 3 :
      {
        launch_domain =
          Domain::from_rect<3>(Rect<3>(make_point(0, 0, 0),
                                       make_point(num_tasks - 1, 0, 0)));
        break;
      }
    default:
      assert(false);
  }
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
      unsigned fields_to_allocate = num_fields + (num_partitions - 1) * slide;
      for (unsigned i = 0; i < fields_to_allocate; ++i)
        allocator.allocate_field(sizeof(int), 100 + i);
    }

    Domain region_domain;
    switch (dims)
    {
      case 1 :
        {
          region_domain =
            Domain::from_rect<1>(Rect<1>(make_point(1),
                                         make_point(num_elmts)));
          break;
        }
      case 2 :
        {
          region_domain =
            Domain::from_rect<2>(Rect<2>(make_point(1, 1),
                                         make_point(num_elmts, 1)));
          break;
        }
      case 3 :
        {
          region_domain =
            Domain::from_rect<3>(Rect<3>(make_point(1, 1, 1),
                                         make_point(num_elmts, 1, 1)));
          break;
        }
      default:
        assert(false);
    }


    IndexSpace is = runtime->create_index_space(ctx, region_domain);

    for (unsigned p = 0; p < num_partitions; ++p)
    {
      switch (dims)
      {
        case 1 :
          {
            create_index_partitions<1>(ctx, runtime, is, num_tasks, p,
                alternate, 0, tree_depth, pattern);
            break;
          }
        case 2 :
          {
            create_index_partitions<2>(ctx, runtime, is, num_tasks, p,
                alternate, 0, tree_depth, pattern);
            break;
          }
        case 3 :
          {
            create_index_partitions<3>(ctx, runtime, is, num_tasks, p,
                alternate, 0, tree_depth, pattern);
            break;
          }
        default:
          assert(false);
      }
    }

    for (unsigned i = 0; i < num_regions; ++i)
      lrs.push_back(runtime->create_logical_region(ctx, is, fs));

    for (unsigned i = 0; i < num_regions; ++i)
    {
      lps.push_back(vector<LogicalPartition>());
      for (unsigned p = 0; p < num_partitions; ++p)
        lps[i].push_back(
            runtime->get_logical_partition_by_color(ctx, lrs[i], p));
    }

    if (tree_depth > 1)
    {
      pid = 1;
      runtime->register_projection_functor(pid,
          new TreeTraversingFunctor(runtime, tree_depth));
    }
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
      if (tracing && (!alternate_loop || l % pattern.size() == 0))
        runtime->begin_trace(ctx, 0);

      unsigned bounds = alternate ? pattern.size() : num_partitions;
      for (unsigned j = 0; j < bounds; ++j)
      {
        unsigned p = j % num_partitions;
        for (unsigned i = 0; i < num_tasks; ++i)
        {
          TaskLauncher launcher(DO_NOTHING_TASK_ID, TaskArgument());
          if (block && l == 0 && p == 0) launcher.add_wait_barrier(next_barrier);
          DomainPoint dp;
          dp.dim = dims;
          dp.point_data[0] = i;
          for (unsigned k = 1; k < dims; ++k) dp.point_data[k] = 0;
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

            if ((alternate && pattern[j] == RD) ||
                (alternate_loop && pattern[l % pattern.size()] == RD))
            {
              unsigned offset = slide * p;
              RegionRequirement req(lr, REDUCE_ID, SIMULTANEOUS, lrs[r]);
              for (unsigned k = 0; k < num_fields; ++k)
              {
                req.tag = l / (alternate_loop ? pattern.size() : 1);
                req.add_field(100 + k + offset);
              }
              launcher.add_region_requirement(req);
            }
            else
            {
              FieldID fid = 100;
              unsigned fid_block = num_fields / blast;
              unsigned offset = slide * p;
              for (unsigned b = 0; b < blast; ++b)
              {
                PrivilegeMode priv = READ_WRITE;
                if (alternate)
                {
                  if (pattern[j] == RO) priv = READ_ONLY;
                  else if (pattern[j] == WO) priv = WRITE_ONLY;
                }
                else if (alternate_loop)
                {
                  if (pattern[l % pattern.size()] == RO) priv = READ_ONLY;
                  else if (pattern[l % pattern.size()] == WO) priv = WRITE_ONLY;
                }
                RegionRequirement req(lr, priv, EXCLUSIVE, lrs[r]);
                req.tag = l / (alternate_loop ? pattern.size() : 1);
                for (unsigned k = 0; k < fid_block; ++k)
                {
                  req.add_field(fid + k + offset);
                }
                launcher.add_region_requirement(req);
                fid += fid_block;
              }
            }
          }
          runtime->execute_task(ctx, launcher);
        }
      }
      if (tracing && (!alternate_loop || (l + 1) % pattern.size() == 0))
        runtime->end_trace(ctx, 0);
    }
  }
  else
  {
    for (unsigned l = 0; l < num_loops; ++l)
    {
      if (tracing && (!alternate_loop || l % pattern.size() == 0))
        runtime->begin_trace(ctx, 0);
      unsigned bounds = alternate ? pattern.size() : num_partitions;
      for (unsigned j = 0; j < bounds; ++j)
      {
        unsigned p = j % num_partitions;
        IndexTaskLauncher launcher(DO_NOTHING_TASK_ID, launch_domain,
                                   TaskArgument(), ArgumentMap());
        if (block && l == 0 && p == 0) launcher.add_wait_barrier(next_barrier);
        for (unsigned r = 0; r < num_regions; ++r)
        {
          if ((alternate && pattern[j] == RD) ||
              (alternate_loop && pattern[l % pattern.size()] == RD))
          {
            unsigned offset = slide * p;
              RegionRequirement req(lps[r][p], pid, REDUCE_ID, SIMULTANEOUS,
                  lrs[r]);
            for (unsigned k = 0; k < num_fields; ++k)
            {
              req.tag = l / (alternate_loop ? pattern.size() : 1);
              req.add_field(100 + k + offset);
            }
              launcher.add_region_requirement(req);
          }
          else
          {
            FieldID fid = 100;
            unsigned fid_block = num_fields / blast;
            unsigned offset = slide * p;
            for (unsigned b = 0; b < blast; ++b)
            {
              PrivilegeMode priv = READ_WRITE;
              if (alternate)
              {
                if (pattern[j] == RO) priv = READ_ONLY;
                else if (pattern[j] == WO) priv = WRITE_ONLY;
              }
              else if (alternate_loop)
              {
                if (pattern[l % pattern.size()] == RO) priv = READ_ONLY;
                else if (pattern[l % pattern.size()] == WO) priv = WRITE_ONLY;
              }
              RegionRequirement req(lps[r][p], pid, priv, EXCLUSIVE,
                  lrs[r]);
              req.tag = l / (alternate_loop ? pattern.size() : 1);
              for (unsigned k = 0; k < fid_block; ++k)
              {
                req.add_field(fid + k + offset);
              }
              launcher.add_region_requirement(req);
              fid += fid_block;
            }
          }
        }
        runtime->execute_index_space(ctx, launcher);
      }
      if (tracing && (!alternate_loop || (l + 1) % pattern.size() == 0))
        runtime->end_trace(ctx, 0);
    }
  }
  barrier_for_block.arrive(1);
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

  Runtime::add_registration_callback(register_mappers);
  Runtime::register_reduction_op<ReduceNothing>(REDUCE_ID);

  return Runtime::start(argc, argv);
}
