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
#include <cstring>
#include <cstdlib>
#include <unistd.h>

#include "legion.h"

#include "mappers/default_mapper.h"

using namespace std;
using namespace Legion;
using namespace Legion::Mapping;

#define SIZE 10

enum FIDs
{
  FID_X = 100,
  FID_Y = 101,
  FID_Z = 102,
  FID_W = 103,
};

enum TaskIDs
{
  TID_MAIN = 100,
  TID_PRODUCER = 101,
  TID_CONSUMER = 102,
  TID_CONDITION = 103,
};

enum MappingTags
{
  TAG_REUSE = 100,
  TAG_CREATE_NEW = 101,
  TAG_LOCAL_PROCESSOR = 102,
};

class OutReqTestMapper : public DefaultMapper
{
 public:
  OutReqTestMapper(MapperRuntime *rt, Machine machine, Processor local,
                      const char *mapper_name);
 public:
  virtual LogicalRegion default_policy_select_instance_region(
      MapperContext, Memory, const RegionRequirement &req,
      const LayoutConstraintSet&, bool, bool)
  {
    return req.region;
  }
  virtual void default_policy_select_instance_fields(
                                MapperContext ctx,
                                const RegionRequirement &req,
                                const std::set<FieldID> &needed_fields,
                                std::vector<FieldID> &fields)
  {
    fields.insert(fields.end(), needed_fields.begin(), needed_fields.end());
  }

 public:
  virtual int default_policy_select_garbage_collection_priority(
                                    MapperContext ctx,
                                    MappingKind kind, Memory memory,
                                    const PhysicalInstance &instance,
                                    bool meets_fill_constraints,
                                    bool reduction)
  {
    return LEGION_GC_FIRST_PRIORITY;
  }

 public:
  using DefaultMapper::speculate;
  virtual void select_task_options(const MapperContext    ctx,
                                   const Task&            task,
                                         TaskOptions&     output);
  virtual void speculate(const MapperContext      ctx,
                         const Task&              task,
                               SpeculativeOutput& output);
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
  bool request_speculate;
};

OutReqTestMapper::OutReqTestMapper(MapperRuntime *rt,
                                         Machine machine,
                                         Processor local,
                                         const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name),
    request_speculate(false)
{
  Machine::MemoryQuery visible_memories(machine);
  visible_memories.has_affinity_to(local);
  visible_memories.only_kind(Memory::SYSTEM_MEM);
  local_sysmem = visible_memories.first();

  const InputArgs &command_args = Runtime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;

  for (int i = 0; i < argc; ++i)
    if (strcmp(argv[i], "-speculate") == 0)
      request_speculate = true;
}

void OutReqTestMapper::select_task_options(const MapperContext    ctx,
                                           const Task&            task,
                                                 TaskOptions&     output)
{
  DefaultMapper::select_task_options(ctx, task, output);
  if (task.tag == TAG_LOCAL_PROCESSOR)
    output.initial_proc = task.parent_task->current_proc;
}

void OutReqTestMapper::speculate(const MapperContext      ctx,
                                 const Task&              task,
                                       SpeculativeOutput& output)
{
  if (task.task_id == TID_PRODUCER)
    output.speculate = request_speculate;
  else
    output.speculate = false;
}

void OutReqTestMapper::slice_task(const MapperContext ctx,
                                     const Task& task,
                                     const SliceTaskInput& input,
                                     SliceTaskOutput& output)
{
  size_t idx = 0;
  for (Domain::DomainPointIterator itr(input.domain); itr; ++itr, ++idx) {
    Domain slice(*itr, *itr);
    output.slices.push_back(
      TaskSlice(slice, local_cpus[idx % local_cpus.size()], false, false)
    );
  }
}

void OutReqTestMapper::map_task(const MapperContext ctx,
                                const Task& task,
                                const MapTaskInput& input,
                                MapTaskOutput& output)
{
  if (task.task_id != TID_CONSUMER)
  {
    DefaultMapper::map_task(ctx, task, input, output);
    return;
  }

  output.task_priority = 0;
  output.postmap_task  = false;
  output.target_procs.push_back(task.target_proc);
  std::vector<VariantID> variants;
  runtime->find_valid_variants(
      ctx, task.task_id, variants, Processor::LOC_PROC);
  assert(!variants.empty());
  output.chosen_variant = *variants.begin();

  for (unsigned ridx = 0; ridx < task.regions.size(); ++ridx)
  {
    const RegionRequirement &req = task.regions[ridx];
    Domain domain =
      runtime->get_index_space_domain(ctx, req.region.get_index_space());

    std::vector<DimensionKind> ordering;
    for (int i = 0; i < domain.dim; ++i)
      ordering.push_back(static_cast<DimensionKind>(DIM_X + i));
    ordering.push_back(DIM_F);

    std::vector<LogicalRegion> regions(1, req.region);

    if (req.tag == TAG_REUSE)
    {
      for (unsigned idx = 0; idx < req.instance_fields.size(); ++idx)
      {
        std::vector<FieldID> fields(1, req.instance_fields[idx]);
        LayoutConstraintSet constraints;
        constraints.add_constraint(MemoryConstraint(local_sysmem.kind()))
          .add_constraint(OrderingConstraint(ordering, false))
          .add_constraint(FieldConstraint(fields, false, false))
          .add_constraint(
            SpecializedConstraint(LEGION_AFFINE_SPECIALIZE, 0, false, true));

        PhysicalInstance instance;
        assert(runtime->find_physical_instance(ctx,
                                               local_sysmem,
                                               constraints,
                                               regions,
                                               instance,
                                               true,
                                               true));
        output.chosen_instances[ridx].push_back(instance);
      }
    }
    else
    {
      assert(req.tag == TAG_CREATE_NEW);
      LayoutConstraintSet constraints;
      constraints.add_constraint(MemoryConstraint(local_sysmem.kind()))
        .add_constraint(OrderingConstraint(ordering, false))
        .add_constraint(FieldConstraint(req.instance_fields, false, false))
        .add_constraint(
          SpecializedConstraint(LEGION_AFFINE_SPECIALIZE, 0, false, true));

      PhysicalInstance instance;
      size_t footprint;
      assert(runtime->create_physical_instance(ctx,
                                               local_sysmem,
                                               constraints,
                                               regions,
                                               instance,
                                               true,
                                               0,
                                               true,
                                               &footprint));
      output.chosen_instances[ridx].push_back(instance);
    }
  }
}

struct TestArgs {
  TestArgs(void)
    : index_launch(false), predicate(false), empty(false), replicate(false)
  { }
  bool index_launch;
  bool predicate;
  bool empty;
  bool replicate;
};

bool condition_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx,
                     Runtime *runtime)
{
  usleep(2000);
  return false;
}

void producer_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx,
                   Runtime *runtime)
{
  TestArgs *args = reinterpret_cast<TestArgs*>(task->args);

  std::vector<OutputRegion> outputs;
  runtime->get_output_regions(ctx, outputs);

  if (args->empty)
  {
    outputs[0].return_data(0, FID_X, NULL);
    outputs[0].return_data(0, FID_Y, NULL);
    outputs[1].return_data(0, FID_Z, NULL);
    outputs[1].return_data(0, FID_W, NULL);
    return;
  }

  void *ptr_x = malloc(SIZE * sizeof(int64_t));
  DeferredBuffer<int32_t,1> buf_y(Rect<1>(1, SIZE), Memory::SYSTEM_MEM);
  int32_t *ptr_y = buf_y.ptr(1);
  void *ptr_z = malloc(SIZE * sizeof(int64_t));

  for (unsigned idx = 0; idx < SIZE; ++idx)
    static_cast<int64_t*>(ptr_x)[idx] = 111 + task->index_point[0];

  for (unsigned idx = 0; idx < SIZE; ++idx)
    ptr_y[idx] = 222 + task->index_point[0];

  for (unsigned idx = 0; idx < SIZE; ++idx)
    static_cast<int64_t*>(ptr_z)[idx] = 333 + task->index_point[0];

  outputs[0].return_data(SIZE, FID_X, ptr_x);
  outputs[0].return_data(FID_Y, buf_y);
  outputs[1].return_data(SIZE, FID_Z, ptr_z);
  outputs[1].return_data(SIZE, FID_W, NULL);
}

typedef FieldAccessor<READ_ONLY, int64_t, 1, coord_t,
                      Realm::AffineAccessor<int64_t, 1, coord_t> >
        Int64Accessor1D;

typedef FieldAccessor<READ_ONLY, int64_t, 2, coord_t,
                      Realm::AffineAccessor<int64_t, 2, coord_t> >
        Int64Accessor2D;

typedef FieldAccessor<READ_ONLY, int32_t, 1, coord_t,
                      Realm::AffineAccessor<int32_t, 1, coord_t> >
        Int32Accessor1D;

void consumer_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx,
                   Runtime *runtime)
{
  TestArgs *args = reinterpret_cast<TestArgs*>(task->args);

  if (args->empty || args->predicate)
  {
    Rect<1> r1(regions[0]);
    assert(r1.empty());
    if (args->index_launch)
    {
      Rect<2> r2(regions[1]);
      assert(r2.empty());
      fprintf(stderr,
          "[Consumer %lld] region 1: %lld -- %lld, "
          "region 2: (%lld, %lld) -- (%lld, %lld)\n",
          task->index_point[0], r1.lo[0], r1.hi[0],
          r2.lo[0], r2.lo[1], r2.hi[0], r2.hi[1]);
    }
    else
    {
      Rect<1> r2(regions[1]);
      assert(r2.empty());
      fprintf(stderr,
          "[Consumer %lld] region 1: %lld -- %lld, "
          "region 2: %lld -- %lld\n",
          task->index_point[0], r1.lo[0], r1.hi[0], r1.lo[0], r1.hi[0]);
    }
    return;
  }

  if (args->index_launch)
  {
    Rect<1> r1(regions[0]);
    Rect<2> r2(regions[1]);
    assert(r1 ==
           Rect<1>(Point<1>(task->index_point[0] * SIZE),
                   Point<1>((task->index_point[0] + 1) * SIZE - 1)));
    assert(r2 ==
           Rect<2>(Point<2>(task->index_point[0], 0),
                   Point<2>(task->index_point[0], SIZE - 1)));
    fprintf(stderr,
        "[Consumer %lld] region 1: %lld -- %lld, "
        "region 2: (%lld, %lld) -- (%lld, %lld)\n",
        task->index_point[0], r1.lo[0], r1.hi[0],
        r2.lo[0], r2.lo[1], r2.hi[0], r2.hi[1]);
  }
  else
  {
    Rect<1> r1(regions[0]);
    Rect<1> r2(regions[1]);
    assert(r1 == Rect<1>(Point<1>(0), Point<1>(SIZE - 1)));
    assert(r2 == Rect<1>(Point<1>(0), Point<1>(SIZE - 1)));
    fprintf(stderr,
        "[Consumer %lld] region 1: %lld -- %lld, "
        "region 2: %lld -- %lld\n",
        task->index_point[0], r1.lo[0], r1.hi[0], r1.lo[0], r1.hi[0]);
  }

  const int64_t *ptr_x = NULL;
  const int32_t *ptr_y = NULL;
  const int64_t *ptr_z = NULL;

  Int64Accessor1D acc_x(regions[0], FID_X);
  Int32Accessor1D acc_y(regions[0], FID_Y);
  Rect<1> r1(regions[0]);

  ptr_x = acc_x.ptr(r1.lo);
  ptr_y = acc_y.ptr(r1.lo);

  if (args->index_launch)
  {
    Int64Accessor2D acc_z(regions[1], FID_Z);
    Rect<2> r2(regions[1]);
    ptr_z = acc_z.ptr(r2.lo);
  }
  else
  {
    Int64Accessor1D acc_z(regions[1], FID_Z);
    Rect<1> r2(regions[1]);
    ptr_z = acc_z.ptr(r2.lo);
  }

  for (unsigned idx = 0; idx < SIZE; ++idx)
    assert(ptr_x[idx] == 111 + task->index_point[0]);

  for (unsigned idx = 0; idx < SIZE; ++idx)
    assert(ptr_y[idx] == 222 + task->index_point[0]);

  for (unsigned idx = 0; idx < SIZE; ++idx)
    assert(ptr_z[idx] == 333 + task->index_point[0]);
}

void main_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx,
               Runtime *runtime)
{
  const InputArgs &command_args = Runtime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;

  TestArgs args;

  for (int i = 0; i < argc; ++i)
    if (strcmp(argv[i], "-index") == 0)
      args.index_launch = true;
    else if (strcmp(argv[i], "-predicate") == 0)
      args.predicate = true;
    else if (strcmp(argv[i], "-empty") == 0)
      args.empty = true;
    else if (strcmp(argv[i], "-replicate") == 0)
      args.replicate = true;

  Predicate pred = Predicate::TRUE_PRED;
  if (args.predicate)
  {
    TaskLauncher condition_launcher(TID_CONDITION, TaskArgument());
    Future f = runtime->execute_task(ctx, condition_launcher);
    pred = runtime->create_predicate(ctx, f);
  }

  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  allocator.allocate_field(sizeof(int64_t), FID_X);
  allocator.allocate_field(sizeof(int32_t), FID_Y);
  allocator.allocate_field(sizeof(int64_t), FID_Z);
  allocator.allocate_field(0, FID_W);

  std::set<FieldID> field_set1;
  field_set1.insert(FID_X);
  field_set1.insert(FID_Y);

  std::set<FieldID> field_set2;
  field_set2.insert(FID_Z);
  field_set2.insert(FID_W);

  std::vector<OutputRequirement> out_reqs;
  out_reqs.push_back(OutputRequirement(fs, field_set1, true));
  out_reqs.push_back(OutputRequirement(fs, field_set2, false));

  Domain launch_domain(Rect<1>(Point<1>(0), Point<1>(6)));
  TaskArgument task_args(&args, sizeof(args));

  if (args.index_launch)
  {
    IndexTaskLauncher producer_launcher(
      TID_PRODUCER, launch_domain, task_args, ArgumentMap(), pred);
    runtime->execute_index_space(ctx, producer_launcher, &out_reqs);
  }
  else
  {
    TaskLauncher producer_launcher(TID_PRODUCER, task_args, pred);
    producer_launcher.point = Point<1>(0);
    runtime->execute_task(ctx, producer_launcher, &out_reqs);
  }

  OutputRequirement &out1 = out_reqs[0];
  OutputRequirement &out2 = out_reqs[1];

  MappingTags tags[] = { TAG_REUSE, TAG_CREATE_NEW };

  for (unsigned i = 0; i < 2; ++i)
  {
    if (i == 0 && args.predicate) continue;

    if (args.index_launch)
    {
      IndexTaskLauncher consumer_launcher(
          TID_CONSUMER, launch_domain, task_args, ArgumentMap());
      RegionRequirement req1(out1.partition, 0, READ_ONLY, EXCLUSIVE, out1.parent);
      req1.add_field(FID_X);
      req1.add_field(FID_Y);
      req1.tag = tags[i];
      consumer_launcher.add_region_requirement(req1);
      RegionRequirement req2(out2.partition, 0, READ_ONLY, EXCLUSIVE, out2.parent);
      req2.add_field(FID_Z);
      req2.tag = tags[i];
      consumer_launcher.add_region_requirement(req2);
      runtime->execute_index_space(ctx, consumer_launcher);
    }
    else
    {
      MappingTagID tag = 0;
      if (i == 0)
        tag = TAG_LOCAL_PROCESSOR;
      TaskLauncher consumer_launcher(
          TID_CONSUMER, task_args, Predicate::TRUE_PRED, 0, tag);
      consumer_launcher.point = Point<1>(0);
      RegionRequirement req1(out1.region, READ_ONLY, EXCLUSIVE, out1.region);
      req1.add_field(FID_X);
      req1.add_field(FID_Y);
      req1.tag = tags[i];
      consumer_launcher.add_region_requirement(req1);
      RegionRequirement req2(out2.region, READ_ONLY, EXCLUSIVE, out2.region);
      req2.add_field(FID_Z);
      req2.tag = tags[i];
      consumer_launcher.add_region_requirement(req2);
      runtime->execute_task(ctx, consumer_launcher);
    }
  }
}

static void create_mappers(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    OutReqTestMapper* mapper = new OutReqTestMapper(
      runtime->get_mapper_runtime(), machine, *it, "output_requirement_test_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }
}

int main(int argc, char **argv)
{
  bool replicate = false;
  for (int i = 0; i < argc; ++i)
    if (strcmp(argv[i], "-replicate") == 0)
      replicate = true;
  {
    TaskVariantRegistrar registrar(TID_MAIN, "main");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    registrar.set_leaf(false);
    registrar.set_inner(true);
    registrar.set_replicable(replicate);
    Runtime::preregister_task_variant<main_task>(registrar, "main");
  }
  {
    TaskVariantRegistrar registrar(TID_PRODUCER, "producer");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(false);
    registrar.set_inner(false);
    Runtime::preregister_task_variant<producer_task>(registrar, "producer");
  }
  {
    TaskVariantRegistrar registrar(TID_CONSUMER, "consumer");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<consumer_task>(registrar, "consumer");
  }
  {
    TaskVariantRegistrar registrar(TID_CONDITION, "condition");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<bool, condition_task>(registrar, "condition");
  }
  Runtime::add_registration_callback(create_mappers);

  Runtime::set_top_level_task_id(TID_MAIN);

  Runtime::start(argc, argv);

  return 0;
}
