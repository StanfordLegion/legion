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

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "legion.h"

#include "mappers/default_mapper.h"

using namespace std;
using namespace Legion;
using namespace Legion::Mapping;

Logger log_mapper("all_to_all");

enum FIDs
{
  FID_X = 100,
};

enum TaskIDs
{
  TID_MAIN = 100,
  TID_INIT_RANGES = 101,
  TID_DENSE = 102,
  TID_SPARSE = 103,
};

struct TilingFunctor : public ShardingFunctor
{
  TilingFunctor() : ShardingFunctor() {}
  virtual ShardID shard(const DomainPoint& point,
                        const Domain& launch_space,
                        const size_t total_shards)
  {
    const size_t num_tasks = launch_space.get_volume();
    const size_t tasks_per_shard = (num_tasks + total_shards - 1) / total_shards;
    ShardID shard_id = static_cast<ShardingID>(point[0]) / tasks_per_shard;
    return shard_id;
  }

  static const ShardingID functor_id = 12345;
  static void register_functor(Runtime* runtime)
  {
    runtime->register_sharding_functor(functor_id, new TilingFunctor());
  }
};

class AllToAllMapper : public DefaultMapper
{
 public:
  AllToAllMapper(MapperRuntime *rt, Machine machine, Processor local,
                 const char *mapper_name);
 public:
  virtual void select_sharding_functor(const Mapping::MapperContext ctx,
                                       const Task& task,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output);

 public:
  virtual LogicalRegion default_policy_select_instance_region(MapperContext,
                                                              Memory,
                                                              const RegionRequirement &req,
                                                              const LayoutConstraintSet&,
                                                              bool,
                                                              bool)
  {
    return req.region;
  }

 public:
  virtual void slice_task(const MapperContext ctx,
                          const Task& task,
                          const SliceTaskInput& input,
                          SliceTaskOutput& output);
  virtual void map_task(const MapperContext ctx,
                        const Task& task,
                        const MapTaskInput& input,
                        MapTaskOutput& output);
 private:
  Memory local_sysmem;
};

AllToAllMapper::AllToAllMapper(MapperRuntime *rt,
                               Machine machine,
                               Processor local,
                               const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
{
  Machine::MemoryQuery visible_memories(machine);
  visible_memories.has_affinity_to(local);
  visible_memories.only_kind(Memory::SYSTEM_MEM);
  local_sysmem = visible_memories.first();
}

void AllToAllMapper::select_sharding_functor(const Mapping::MapperContext ctx,
                                             const Task& task,
                                             const SelectShardingFunctorInput& input,
                                             SelectShardingFunctorOutput& output)
{
  output.chosen_functor = TilingFunctor::functor_id;
}

void AllToAllMapper::slice_task(const MapperContext ctx,
                                const Task& task,
                                const SliceTaskInput& input,
                                SliceTaskOutput& output)
{
  const Rect<1> rect(input.domain);
  size_t idx = 0;
  for (PointInRectIterator<1> pir(rect); pir(); ++pir, ++idx) {
    Rect<1> slice(*pir, *pir);
    output.slices.push_back(TaskSlice(slice, local_cpus[idx % local_cpus.size()], false, false));
  }
}

void AllToAllMapper::map_task(const MapperContext ctx,
                              const Task& task,
                              const MapTaskInput& input,
                              MapTaskOutput& output)
{
  if (task.task_id != TID_SPARSE)
  {
    DefaultMapper::map_task(ctx, task, input, output);
    return;
  }

  output.task_priority = 0;
  output.postmap_task  = false;
  output.target_procs.push_back(task.target_proc);
  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, task.task_id, variants, Processor::LOC_PROC);
  assert(!variants.empty());
  output.chosen_variant = *variants.begin();

  assert(task.regions.size() == 1);

  std::vector<DimensionKind> ordering;
  ordering.push_back(DIM_X);
  ordering.push_back(DIM_F);

  LayoutConstraintSet constraints;
  constraints.add_constraint(MemoryConstraint(local_sysmem.kind()))
    .add_constraint(OrderingConstraint(ordering, false))
    .add_constraint(FieldConstraint(task.regions[0].instance_fields, false, false))
    .add_constraint(SpecializedConstraint(LEGION_COMPACT_SPECIALIZE, 0, false, true));

  std::vector<LogicalRegion> regions(1, task.regions[0].region);

  PhysicalInstance instance;
  size_t footprint = 0;
  bool created     = false;
  if (!runtime->find_or_create_physical_instance(ctx,
                                                 local_sysmem,
                                                 constraints,
                                                 regions,
                                                 instance,
                                                 created,
                                                 true,
                                                 GC_DEFAULT_PRIORITY,
                                                 true,
                                                 &footprint))
    log_mapper.error(
      "Failed allocation of size %zd bytes for "
      "region requirement %u of task %s (UID %lld) in memory " IDFMT " for processor " IDFMT
      ". This means the working "
      "set of your application is too big for the allotted "
      "capacity of the given memory under the default "
      "mapper's mapping scheme. You have three choices: "
      "ask Realm to allocate more memory, write a custom "
      "mapper to better manage working sets, or find a bigger "
      "machine.",
      footprint,
      0,
      task.get_task_name(),
      task.get_unique_id(),
      local_sysmem.id,
      task.target_proc.id);


  output.chosen_instances[0].push_back(instance);
}

void leaf_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx,
               Runtime *runtime)
{
}

struct InitRangesArg
{
  int64_t size;
  int32_t num_colors;
  int32_t num_neighbors;
};

typedef FieldAccessor<WRITE_DISCARD, Rect<1>, 1, coord_t, Realm::AffineAccessor<Rect<1>, 1, coord_t> > RangeAccessor;

void init_ranges_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx,
                      Runtime *runtime)
{
  InitRangesArg* args = static_cast<InitRangesArg*>(task->args);
  int64_t size = args->size;
  int64_t num_colors = args->num_colors;
  int64_t num_neighbors = args->num_neighbors;

  int64_t num_chunks = num_colors * num_neighbors;
  int64_t subregion_offset = size / num_colors;
  int64_t chunk_offset = subregion_offset / num_colors;

  RangeAccessor acc(regions[0], FID_X);

  if (num_colors == num_neighbors)
    for (int64_t i = 0; i < num_chunks; ++i)
    {
      int64_t subregion_idx = i % num_colors;
      int64_t chunk_idx = i / num_colors;
      acc[i] = Rect<1>(subregion_idx * subregion_offset + chunk_idx * chunk_offset,
                       subregion_idx * subregion_offset + (chunk_idx + 1) * chunk_offset - 1);
    }
  else
    for (int64_t i = 0; i < num_chunks; ++i)
    {
      int64_t my_idx = i / num_neighbors;
      int64_t subregion_idx = (i % num_neighbors + my_idx) % num_colors;
      int64_t chunk_idx = i / num_neighbors;
      acc[i] = Rect<1>(subregion_idx * subregion_offset + chunk_idx * chunk_offset,
                       subregion_idx * subregion_offset + (chunk_idx + 1) * chunk_offset - 1);
    }
}

void main_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx,
               Runtime *runtime)
{
  // Parse command line arguments
  int64_t size_per_subregion = 100000;
  int32_t num_colors = 2;
  uint32_t num_loops = 2;
  int32_t num_neighbors = -1;
  {
    const InputArgs &args = Runtime::get_input_args();
    for (int32_t i = 0; i < args.argc; ++i)
    {
      if (strcmp(args.argv[i], "-l") == 0)
      {
        ++i;
        num_loops = atoi(args.argv[i]);
      }
      else if (strcmp(args.argv[i], "-p") == 0)
      {
        ++i;
        num_colors = atoi(args.argv[i]);
      }
      else if (strcmp(args.argv[i], "-s") == 0)
      {
        ++i;
        size_per_subregion = atoi(args.argv[i]);
      }
      else if (strcmp(args.argv[i], "-n") == 0)
      {
        ++i;
        num_neighbors = atoi(args.argv[i]);
      }
    }
  }

  if (num_neighbors == -1 || num_neighbors > num_colors)
    num_neighbors = num_colors;

  // Create a data region
  int64_t size = size_per_subregion * num_colors;

  IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, size - 1));
  IndexSpace cs = runtime->create_index_space(ctx, Rect<1>(0, num_colors - 1));

  LogicalRegion r;
  {
    //std::vector<size_t> sizes(1, 8);
    //std::vector<FieldID> fids(1, FID_X);
    //FieldSpace fs = runtime->create_field_space(ctx, sizes, fids);
    FieldSpace fs = runtime->create_field_space(ctx);
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(int64_t), FID_X);
    r = runtime->create_logical_region(ctx, is, fs);
  }

  // Create a range region
  IndexSpace ranges_is = runtime->create_index_space(ctx, Rect<1>(0, num_colors * num_neighbors - 1));
  LogicalRegion ranges;
  {
    //std::vector<size_t> sizes(1, sizeof(Rect<1>));
    //std::vector<FieldID> fids(1, FID_X);
    //FieldSpace fs = runtime->create_field_space(ctx, sizes, fids);
    FieldSpace fs = runtime->create_field_space(ctx);
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(Rect<1>), FID_X);
    ranges = runtime->create_logical_region(ctx, ranges_is, fs);
  }

  // Initialize ranges
  InitRangesArg arg;
  arg.size = size;
  arg.num_colors = num_colors;
  arg.num_neighbors = num_neighbors;

  TaskLauncher init_ranges(TID_INIT_RANGES, TaskArgument(&arg, sizeof(InitRangesArg)));
  {
    RegionRequirement req(ranges, WRITE_DISCARD, EXCLUSIVE, ranges);
    req.add_field(FID_X);
    init_ranges.add_region_requirement(req);
  }
  runtime->execute_task(ctx, init_ranges);

  // Partition ranges equally
  LogicalPartition p_ranges;
  {
    IndexPartition ip = runtime->create_equal_partition(ctx, ranges_is, cs);
    p_ranges = runtime->get_logical_partition(ctx, ranges, ip);
  }

  LogicalPartition p;
  {
    IndexPartition ip = runtime->create_equal_partition(ctx, is, cs);
    p = runtime->get_logical_partition(ctx, r, ip);
  }

  LogicalPartition q;
  {
    IndexPartition ip = runtime->create_partition_by_image_range(
      ctx, is, p_ranges, ranges, FID_X, cs,
      num_colors == num_neighbors ? LEGION_DISJOINT_COMPLETE_KIND : LEGION_DISJOINT_KIND);
    q = runtime->get_logical_partition(ctx, r, ip);
  }

  IndexTaskLauncher dense_launcher(
      TID_DENSE, Domain(Rect<1>(0, num_colors - 1)), TaskArgument(), ArgumentMap());
  {
    RegionRequirement req(p, 0, WRITE_DISCARD, EXCLUSIVE, r);
    req.add_field(FID_X);
    dense_launcher.add_region_requirement(req);
  }

  IndexTaskLauncher sparse_launcher(
      TID_SPARSE, Domain(Rect<1>(0, num_colors - 1)), TaskArgument(), ArgumentMap());
  {
    RegionRequirement req(q, 0, READ_WRITE, EXCLUSIVE, r);
    req.add_field(FID_X);
    sparse_launcher.add_region_requirement(req);
  }

  ShardID shard_id = runtime->get_shard_id(ctx, true);
  for (uint32_t l = 0; l < num_loops; ++l)
  {
    runtime->execute_index_space(ctx, dense_launcher);

    Future f_start = runtime->get_current_time_in_microseconds(ctx, runtime->issue_execution_fence(ctx));
    runtime->execute_index_space(ctx, sparse_launcher);
    Future f_end = runtime->get_current_time_in_microseconds(ctx, runtime->issue_execution_fence(ctx));
    int64_t elapsed_time = f_end.get_result<int64_t>() - f_start.get_result<int64_t>();
    if (shard_id == 0)
      printf("loop %u elapsed time %ld us\n", l, elapsed_time);
  }
}

static void create_mappers(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs)
{
  TilingFunctor::register_functor(runtime);
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    AllToAllMapper* mapper = new AllToAllMapper(
      runtime->get_mapper_runtime(), machine, *it, "all_to_all_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }
}

int main(int argc, char **argv)
{
  {
    TaskVariantRegistrar registrar(TID_MAIN, "main");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    registrar.set_leaf(false);
    registrar.set_inner(true);
    registrar.set_replicable(true);
    Runtime::preregister_task_variant<main_task>(registrar, "main");
  }
  {
    TaskVariantRegistrar registrar(TID_INIT_RANGES, "init_ranges");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<init_ranges_task>(registrar, "init_ranges");
  }
  {
    TaskVariantRegistrar registrar(TID_DENSE, "dense");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<leaf_task>(registrar, "dense");
  }
  {
    TaskVariantRegistrar registrar(TID_SPARSE, "sparse");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<leaf_task>(registrar, "sparse");
  }
  Runtime::add_registration_callback(create_mappers);

  Runtime::set_top_level_task_id(TID_MAIN);

  Runtime::start(argc, argv);

  return 0;
}
