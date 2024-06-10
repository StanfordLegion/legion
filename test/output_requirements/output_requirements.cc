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
#include <cstring>
#include <cstdlib>
#include <unistd.h>

#include "legion.h"

#include "mappers/default_mapper.h"

using namespace std;
using namespace Legion;
using namespace Legion::Mapping;

#define SIZE 10

static Logger log_test("out_req_test");

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
  TID_PRODUCER_GLOBAL = 101,
  TID_PRODUCER_LOCAL = 102,
  TID_CONSUMER_GLOBAL = 103,
  TID_CONSUMER_LOCAL = 104,
  TID_CONDITION = 105,
  TID_PRODUCER_PROJ_GLOBAL = 106,
  TID_PRODUCER_PROJ_LOCAL = 107,
  TID_CONSUMER_PROJ_GLOBAL = 108,
  TID_CONSUMER_PROJ_LOCAL = 109,
};

enum MappingTags
{
  TAG_REUSE = 100,
  TAG_CREATE_NEW = 101,
  TAG_LOCAL_PROCESSOR = 102,
};

enum ProjectionIDs
{
  PID_ROW_MAJOR = 100,
  PID_COL_MAJOR = 101,
};

static const int32_t DIM_SIZE = 3;

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
  virtual void select_task_options(const MapperContext    ctx,
                                   const Task&            task,
                                         TaskOptions&     output);
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
  std::map<TaskIDs, Processor> producer_mappings;
};

OutReqTestMapper::OutReqTestMapper(MapperRuntime *rt,
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

void OutReqTestMapper::select_task_options(const MapperContext    ctx,
                                           const Task&            task,
                                                 TaskOptions&     output)
{
  DefaultMapper::select_task_options(ctx, task, output);
  if (task.task_id == TID_PRODUCER_GLOBAL || task.task_id == TID_PRODUCER_LOCAL)
    producer_mappings[static_cast<TaskIDs>(task.task_id)] = output.initial_proc;
  else if (task.tag == TAG_LOCAL_PROCESSOR)
  {
    if (task.task_id == TID_CONSUMER_GLOBAL)
      output.initial_proc = producer_mappings[TID_PRODUCER_GLOBAL];
    else
    {
      assert(task.task_id == TID_CONSUMER_LOCAL);
      output.initial_proc = producer_mappings[TID_PRODUCER_LOCAL];
    }
  }
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
  if (!(task.task_id == TID_PRODUCER_GLOBAL
        || task.task_id == TID_PRODUCER_LOCAL
        || task.task_id == TID_CONSUMER_GLOBAL
        || task.task_id == TID_CONSUMER_LOCAL))
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

  if (task.task_id == TID_PRODUCER_GLOBAL)
  {
    output.output_targets[0] = local_sysmem;

    LayoutConstraintSet &constraints = output.output_constraints[0];

    std::vector<DimensionKind> ordering;
    ordering.push_back(DIM_X);
    ordering.push_back(DIM_Y);
    ordering.push_back(DIM_F);
    constraints.ordering_constraint = OrderingConstraint(ordering, false);

    constraints.alignment_constraints.push_back(
      AlignmentConstraint(FID_X, LEGION_EQ_EK, 32));

    return;
  }
  else if (task.task_id == TID_PRODUCER_LOCAL)
  {
    output.output_targets[0] = local_sysmem;

    LayoutConstraintSet &constraints = output.output_constraints[0];

    std::vector<DimensionKind> ordering;
    if (task.is_index_space)
    {
      ordering.push_back(DIM_Z);
      ordering.push_back(DIM_Y);
      ordering.push_back(DIM_X);
      ordering.push_back(DIM_F);
    }
    else
    {
      ordering.push_back(DIM_Y);
      ordering.push_back(DIM_X);
      ordering.push_back(DIM_F);
    }
    constraints.ordering_constraint = OrderingConstraint(ordering, false);

    return;
  }

  const RegionRequirement &req = task.regions[0];
  std::vector<LogicalRegion> regions(1, req.region);
  std::vector<DimensionKind> ordering;
  if (task.task_id == TID_CONSUMER_GLOBAL)
  {
    ordering.push_back(DIM_X);
    ordering.push_back(DIM_Y);
    ordering.push_back(DIM_F);
  }
  else
  {
    assert(task.task_id == TID_CONSUMER_LOCAL);
    if (task.is_index_space)
    {
      ordering.push_back(DIM_Z);
      ordering.push_back(DIM_Y);
      ordering.push_back(DIM_X);
      ordering.push_back(DIM_F);
    }
    else
    {
      ordering.push_back(DIM_Y);
      ordering.push_back(DIM_X);
      ordering.push_back(DIM_F);
    }
  }

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
      output.chosen_instances[0].push_back(instance);
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
    output.chosen_instances[0].push_back(instance);
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

struct RowMajorTransform : public ProjectionFunctor {
  RowMajorTransform(Runtime* rt) : ProjectionFunctor(rt) { }
  virtual bool is_functional(void) const { return true; }
  virtual bool is_exclusive(void) const { return true; }
  virtual unsigned get_depth(void) const { return 0; }

  using ProjectionFunctor::project;
  virtual LogicalRegion project(LogicalPartition upper_bound,
                                const DomainPoint& point,
                                const Domain& launch_domain)
  {
    assert(point.dim == 1);
    DomainPoint result;
    result.dim = 2;
    result[0] = point[0] % DIM_SIZE;
    result[1] = point[0] / DIM_SIZE;
    assert(runtime->has_logical_subregion_by_color(upper_bound, result));
    return runtime->get_logical_subregion_by_color(upper_bound, result);
  }
};

struct ColMajorTransform : public ProjectionFunctor {
  ColMajorTransform(Runtime* rt) : ProjectionFunctor(rt) { }
  virtual bool is_functional(void) const { return true; }
  virtual bool is_exclusive(void) const { return true; }
  virtual unsigned get_depth(void) const { return 0; }

  using ProjectionFunctor::project;
  virtual LogicalRegion project(LogicalPartition upper_bound,
                                const DomainPoint& point,
                                const Domain& launch_domain)
  {
    assert(point.dim == 1);
    DomainPoint result;
    result.dim = 2;
    result[0] = point[0] / DIM_SIZE;
    result[1] = point[0] % DIM_SIZE;
    assert(runtime->has_logical_subregion_by_color(upper_bound, result));
    return runtime->get_logical_subregion_by_color(upper_bound, result);
  }
};

bool condition_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx,
                     Runtime *runtime)
{
  usleep(2000);
  return false;
}

void producer_global_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx,
                          Runtime *runtime)
{
  static constexpr int DIM = 2;

  TestArgs *args = reinterpret_cast<TestArgs*>(task->args);

  std::vector<OutputRegion> outputs;
  runtime->get_output_regions(ctx, outputs);
  OutputRegion& output = outputs.front();

  if (args->empty)
  {
    Rect<DIM, int32_t> bounds(Point<DIM, int32_t>::ONES(), Point<DIM, int32_t>::ZEROES());
    DeferredBuffer<int64_t, 2, int32_t> buf_x(bounds, Memory::Kind::SYSTEM_MEM, NULL, 32, true);
    output.return_data(bounds.hi, FID_X, buf_x);
    outputs[0].create_buffer<int32_t, 2, int32_t>(bounds.hi, FID_Y, NULL, true);
    return;
  }

  Point<DIM, int32_t> extents;
  if (task->is_index_space)
    for (int32_t dim = 0; dim < DIM; ++dim)
      extents[dim] = SIZE - task->index_point[dim];
  else
    for (int32_t dim = 0; dim < DIM; ++dim)
      extents[dim] = SIZE;

  size_t volume = 1;
  for (int32_t dim = 0; dim < DIM; ++dim) volume *= extents[dim];

  Point<DIM, int32_t> hi(extents);
  hi -= Point<DIM, int32_t>::ONES();
  Rect<DIM, int32_t> bounds(Point<DIM, int32_t>::ZEROES(), hi);

  DeferredBuffer<int64_t, 2, int32_t> buf_x(
    bounds, Memory::Kind::SYSTEM_MEM, NULL, 32, true);
  DeferredBuffer<int32_t, 2, int32_t> buf_y =
    outputs[0].create_buffer<int32_t, 2, int32_t>(extents, FID_Y, NULL, true);

  int64_t *ptr_x = buf_x.ptr(Point<2, int32_t>::ZEROES());
  int32_t *ptr_y = buf_y.ptr(Point<2, int32_t>::ZEROES());

  for (size_t idx = 0; idx < volume; ++idx)
  {
    ptr_x[idx] = 111 + static_cast<int64_t>(idx);
    ptr_y[idx] = 222 + static_cast<int32_t>(idx);
  }

  output.return_data(extents, FID_X, buf_x);
}

void producer_local_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime *runtime)
{
  static constexpr int DIM = 2;

  TestArgs *args = reinterpret_cast<TestArgs*>(task->args);

  std::vector<OutputRegion> outputs;
  runtime->get_output_regions(ctx, outputs);
  OutputRegion& output = outputs.front();

  if (args->empty)
  {
    Rect<DIM, int32_t> bounds(
        Point<DIM, int32_t>::ONES(), Point<DIM, int32_t>::ZEROES());
    DeferredBuffer<int64_t, DIM, int32_t> buf_z(
        bounds, Memory::Kind::SYSTEM_MEM);
    output.return_data(bounds.hi, FID_Z, buf_z);
    // TODO: Currently we can't return a buffer to FID_W, as deferred buffers
    //       of a zero-size field cannot be created. Put back this test case
    //       once we add APIs to return untyped buffers to output regions.
    // DeferredBuffer<int8_t, DIM, int32_t> buf_w(
    //   bounds, Memory::Kind::SYSTEM_MEM);
    // output.return_data(bounds.hi, FID_W, buf_w);
    return;
  }

  Point<DIM, int32_t> extents;
  if (task->is_index_space)
    for (int32_t dim = 0; dim < DIM; ++dim)
      extents[dim] = SIZE - task->index_point[0];
  else
    for (int32_t dim = 0; dim < DIM; ++dim)
      extents[dim] = SIZE;

  size_t volume = 1;
  for (int32_t dim = 0; dim < DIM; ++dim) volume *= extents[dim];

  Point<DIM, int32_t> hi(extents);
  hi -= Point<DIM>::ONES();
  Rect<DIM, int32_t> bounds(Point<DIM>::ZEROES(), hi);

  DeferredBuffer<int64_t, DIM, int32_t> buf_z =
    output.create_buffer<int64_t, 2, int32_t>(extents, FID_Z, NULL, false);
  int64_t *ptr_z = buf_z.ptr(Point<2, int32_t>::ZEROES());

  for (size_t idx = 0; idx < volume; ++idx)
    ptr_z[idx] = 333 + static_cast<int64_t>(idx);

  output.return_data(extents, FID_Z, buf_z);

  // TODO: Currently we can't return a buffer to FID_W, as deferred buffers
  //       of a zero-size field cannot be created. Put back this test case
  //       once we add APIs to return untyped buffers to output regions.
  //DeferredBuffer<int8_t, DIM, int32_t> buf_w(
  //  bounds, Memory::Kind::SYSTEM_MEM, NULL, 16, false);
  //output.return_data(extents, FID_W, buf_w);
}

typedef FieldAccessor<READ_ONLY, int64_t, 2, int32_t,
                      Realm::AffineAccessor<int64_t, 2, int32_t> >
        Int64Accessor2D;

typedef FieldAccessor<READ_ONLY, int32_t, 2, int32_t,
                      Realm::AffineAccessor<int32_t, 2, int32_t> >
        Int32Accessor2D;

typedef FieldAccessor<READ_ONLY, int64_t, 3, int32_t,
                      Realm::AffineAccessor<int64_t, 3, int32_t> >
        Int64Accessor3D;

void consumer_global_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx,
                          Runtime *runtime)
{
  static constexpr int DIM = 2;

  TestArgs *args = reinterpret_cast<TestArgs*>(task->args);

  Rect<DIM, int32_t> r(regions[0]);
  log_test.print() << "[Consumer " << task->index_point
                   << ", global indexing] region: " << r;

  if (args->empty || args->predicate)
  {
    assert(r.empty());
    return;
  }

  if (args->index_launch)
  {
    Rect<DIM, int32_t> r(regions[0]);
    static int32_t offsets[] = {0, SIZE, 2 * SIZE - 1, 3 * SIZE - 3};

    for (int32_t dim = 0; dim < DIM; ++dim)
    {
      assert(r.lo[dim] == offsets[task->index_point[dim]]);
      assert(r.hi[dim] == offsets[task->index_point[dim] + 1] - 1);
    }
  }
  else
  {
    Rect<DIM, int32_t> r(regions[0]);
    for (int32_t dim = 0; dim < DIM; ++dim)
    {
      assert(r.lo[dim] == 0);
      assert(r.hi[dim] == SIZE - 1);
    }
  }

  Int64Accessor2D acc_x(regions[0], FID_X);
  Int32Accessor2D acc_y(regions[0], FID_Y);

  Point<DIM, int32_t> extents = r.hi;
  extents -= r.lo;
  extents += Point<DIM, int32_t>::ONES();

  int32_t volume = r.volume();
  for (int32_t idx = 0; idx < volume; ++idx)
  {
    int32_t x0 = idx % extents[0];
    int32_t x1 = idx / extents[0];
    Point<2, int32_t> p(x0, x1);
    assert(acc_x[p + r.lo] == 111 + idx);
    assert(acc_y[p + r.lo] == 222 + idx);
  }
}

void consumer_local_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime *runtime)
{
  TestArgs *args = reinterpret_cast<TestArgs*>(task->args);

  if (args->index_launch)
  {
    static constexpr int DIM = 3;

    Rect<DIM, int32_t> r(regions[0]);
    log_test.print() << "[Consumer " << task->index_point
                     << ", local indexing] region: " << r;
    if (args->empty || args->predicate)
    {
      assert(r.empty());
      return;
    }

    assert(r.lo[0] == task->index_point[0]);
    assert(r.hi[0] == task->index_point[0]);
    for (int32_t dim = 0; dim < DIM - 1; ++dim)
    {
      assert(r.lo[dim + 1] == 0);
      assert(r.hi[dim + 1] == SIZE - task->index_point[0] - 1);
    }

    Int64Accessor3D acc_z(regions[0], FID_Z);

    Point<DIM, int32_t> extents = r.hi;
    extents -= r.lo;
    extents += Point<DIM, int32_t>::ONES();

    int32_t volume = r.volume();
    for (int32_t idx = 0; idx < volume; ++idx)
    {
      int32_t x0 = idx / extents[2];
      int32_t x1 = idx % extents[2];
      Point<3, int32_t> p(task->index_point[0], x0 + r.lo[1], x1 + r.lo[2]);
      assert(acc_z[p] == 333 + idx);
    }
  }
  else
  {
    static constexpr int DIM = 2;

    Rect<DIM, int32_t> r(regions[0]);
    log_test.print() << "[Consumer " << task->index_point
                     << ", local indexing] region: " << r;
    if (args->empty || args->predicate)
    {
      assert(r.empty());
      return;
    }

    for (int32_t dim = 0; dim < DIM; ++dim)
    {
      assert(r.lo[dim] == 0);
      assert(r.hi[dim] == SIZE - 1);
    }

    Int64Accessor2D acc_z(regions[0], FID_Z);

    Point<DIM, int32_t> extents = r.hi;
    extents -= r.lo;
    extents += Point<DIM>::ONES();

    int32_t volume = r.volume();
    for (int32_t idx = 0; idx < volume; ++idx)
    {
      int32_t x0 = idx / extents[1];
      int32_t x1 = idx % extents[1];
      Point<2, int32_t> p(x0, x1);
      assert(acc_z[p] == 333 + idx);
    }
  }
}

void producer_proj_global_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx,
                               Runtime *runtime)
{
  int32_t value = task->index_point[0];

  Point<2, int32_t> extents(1, 1);

  std::vector<OutputRegion> outputs;
  runtime->get_output_regions(ctx, outputs);

  DeferredBuffer<int64_t, 2, int32_t> buf0 =
    outputs[0].create_buffer<int64_t, 2, int32_t>(extents, FID_X, NULL, true);
  DeferredBuffer<int64_t, 2, int32_t> buf1 =
    outputs[1].create_buffer<int64_t, 2, int32_t>(extents, FID_X, NULL, true);

  buf0[Point<2, int32_t>(0, 0)] = value;
  buf1[Point<2, int32_t>(0, 0)] = value;
}

void producer_proj_local_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx,
                              Runtime *runtime)
{
  int32_t value = task->index_point[0];

  Point<1, int32_t> extents(1);

  std::vector<OutputRegion> outputs;
  runtime->get_output_regions(ctx, outputs);

  DeferredBuffer<int64_t, 1, int32_t> buf0 =
    outputs[0].create_buffer<int64_t, 1, int32_t>(extents, FID_X, NULL, true);
  DeferredBuffer<int64_t, 1, int32_t> buf1 =
    outputs[1].create_buffer<int64_t, 1, int32_t>(extents, FID_X, NULL, true);

  buf0[Point<1, int32_t>(0)] = value;
  buf1[Point<1, int32_t>(0)] = value;
}

void consumer_proj_global_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx,
                               Runtime *runtime)
{
  const int32_t dim_size = DIM_SIZE;

  Int64Accessor2D acc0(regions[0], FID_X);
  Int64Accessor2D acc1(regions[1], FID_X);

  {
    std::stringstream ss;
    int32_t counter = 0;
    for (int32_t col = 0; col < dim_size; ++col)
      for (int32_t row = 0; row < dim_size; ++row)
      {
        Point<2, int32_t> p(row, col);
        assert(acc0[p] == counter++);
        ss << p << " ";
      }
    log_test.print() << "[Consumer, proj-global] row major order: " << ss.str();
  }

  {
    std::stringstream ss;
    int32_t counter = 0;
    for (int32_t row = 0; row < dim_size; ++row)
      for (int32_t col = 0; col < dim_size; ++col)
      {
        Point<2, int32_t> p(row, col);
        assert(acc1[p] == counter++);
        ss << p << " ";
      }
    log_test.print() << "[Consumer, proj-global] column major order: " << ss.str();
  }
}

void consumer_proj_local_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx,
                              Runtime *runtime)
{
  const int32_t dim_size = DIM_SIZE;

  Int64Accessor3D acc0(regions[0], FID_X);
  Int64Accessor3D acc1(regions[1], FID_X);

  Point<3> p(task->index_point[0], task->index_point[1], 0);

  int64_t val0 = acc0[p];
  int64_t val1 = acc1[p];

  assert(val0 == task->index_point[0] + task->index_point[1] * dim_size);
  assert(val1 == task->index_point[0] * dim_size + task->index_point[1]);

  log_test.print() << "[Consumer " << task->index_point << ", proj-local] row major order: " << val0;
  log_test.print() << "[Consumer " << task->index_point << ", proj-local] column major order: " << val1;
}
void basic_test(const TestArgs& args, Context ctx, Runtime *runtime)
{
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

  std::set<FieldID> field_set1{FID_X, FID_Y};
  // TODO: Currently we can't return a buffer to FID_W, as deferred buffers
  //       of a zero-size field cannot be created. Put back this test case
  //       once we add APIs to return untyped buffers to output regions.
  // std::set<FieldID> field_set2{FID_Z, FID_W};
  std::set<FieldID> field_set2{FID_Z};

  std::vector<OutputRequirement> out_reqs_global;
  std::vector<OutputRequirement> out_reqs_local;
  out_reqs_global.push_back(OutputRequirement(fs, field_set1, 2, true));
  out_reqs_global.back().set_type_tag<2, int32_t>();
  out_reqs_local.push_back(OutputRequirement(fs, field_set2, 2, false));
  out_reqs_local.back().set_type_tag<2, int32_t>();

  TaskArgument task_args(&args, sizeof(args));

  if (args.index_launch)
  {
    {
      Domain launch_domain(Rect<2>(Point<2>(0, 0), Point<2>(2, 2)));
      IndexTaskLauncher launcher(
        TID_PRODUCER_GLOBAL, launch_domain, task_args, ArgumentMap(), pred);
      runtime->execute_index_space(ctx, launcher, &out_reqs_global);
    }

    {
      Domain launch_domain(Rect<1>(Point<1>(0), Point<1>(2)));
      IndexTaskLauncher launcher(
        TID_PRODUCER_LOCAL, launch_domain, task_args, ArgumentMap(), pred);
      runtime->execute_index_space(ctx, launcher, &out_reqs_local);
    }
  }
  else
  {
    {
      TaskLauncher launcher(TID_PRODUCER_GLOBAL, task_args, pred);
      launcher.point = Point<2>::ZEROES();
      runtime->execute_task(ctx, launcher, &out_reqs_global);
    }

    {
      TaskLauncher launcher(TID_PRODUCER_LOCAL, task_args, pred);
      launcher.point = Point<1>::ZEROES();
      runtime->execute_task(ctx, launcher, &out_reqs_local);
    }
  }

  OutputRequirement &out_global = out_reqs_global.front();
  OutputRequirement &out_local = out_reqs_local.front();

  MappingTags tags[] = { TAG_REUSE, TAG_CREATE_NEW };

  for (unsigned i = 0; i < 2; ++i)
  {
    if (i == 0 && args.predicate) continue;

    if (args.index_launch)
    {
      {
        Domain launch_domain(Rect<2>(Point<2>(0, 0), Point<2>(2, 2)));
        IndexTaskLauncher launcher(
            TID_CONSUMER_GLOBAL, launch_domain, task_args, ArgumentMap());
        RegionRequirement req(
            out_global.partition, 0, READ_ONLY, EXCLUSIVE, out_global.parent);
        req.add_field(FID_X);
        req.add_field(FID_Y);
        req.tag = tags[i];
        launcher.add_region_requirement(req);
        runtime->execute_index_space(ctx, launcher);
      }

      {
        Domain launch_domain(Rect<1>(Point<1>(0), Point<1>(2)));
        IndexTaskLauncher launcher(
            TID_CONSUMER_LOCAL, launch_domain, task_args, ArgumentMap());
        RegionRequirement req(
            out_local.partition, 0, READ_ONLY, EXCLUSIVE, out_local.parent);
        req.add_field(FID_Z);
        req.tag = tags[i];
        launcher.add_region_requirement(req);
        runtime->execute_index_space(ctx, launcher);
      }
    }
    else
    {
      MappingTagID tag = 0;
      if (i == 0)
        tag = TAG_LOCAL_PROCESSOR;
      {
        TaskLauncher launcher(
            TID_CONSUMER_GLOBAL, task_args, Predicate::TRUE_PRED, 0, tag);
        launcher.point = Point<2>::ZEROES();
        RegionRequirement req(
            out_global.region, READ_ONLY, EXCLUSIVE, out_global.region);
        req.add_field(FID_X);
        req.add_field(FID_Y);
        req.tag = tags[i];
        launcher.add_region_requirement(req);
        runtime->execute_task(ctx, launcher);
      }
      {
        TaskLauncher launcher(
            TID_CONSUMER_LOCAL, task_args, Predicate::TRUE_PRED, 0, tag);
        launcher.point = Point<1>::ZEROES();
        RegionRequirement req(
            out_local.region, READ_ONLY, EXCLUSIVE, out_local.region);
        req.add_field(FID_Z);
        req.tag = tags[i];
        launcher.add_region_requirement(req);
        runtime->execute_task(ctx, launcher);
      }
    }
  }
}

void projection_test(const TestArgs& args, Context ctx, Runtime *runtime, bool global)
{
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  allocator.allocate_field(sizeof(int64_t), FID_X);

  Rect<2> colors(Point<2>(0, 0), Point<2>(DIM_SIZE - 1, DIM_SIZE - 1));
  IndexSpace color_space = runtime->create_index_space(ctx, colors);

  int32_t out_dim = global ? 2 : 1;

  std::set<FieldID> field_set{FID_X};

  std::vector<OutputRequirement> out_reqs;
  out_reqs.push_back(OutputRequirement(fs, {FID_X}, out_dim, global));
  if (global) out_reqs.back().set_type_tag<2, int32_t>();
  else out_reqs.back().set_type_tag<1, int32_t>();
  out_reqs.back().set_projection(PID_ROW_MAJOR, color_space);
  out_reqs.push_back(OutputRequirement(fs, {FID_X}, out_dim, global));
  if (global) out_reqs.back().set_type_tag<2, int32_t>();
  else out_reqs.back().set_type_tag<1, int32_t>();
  out_reqs.back().set_projection(PID_COL_MAJOR, color_space);

  TaskArgument task_args(&args, sizeof(args));

  {
    Domain launch_domain(Rect<1>(Point<1>(0), Point<1>(DIM_SIZE * DIM_SIZE - 1)));
    TaskID task_id = global ? TID_PRODUCER_PROJ_GLOBAL : TID_PRODUCER_PROJ_LOCAL;
    IndexTaskLauncher launcher(task_id, launch_domain, task_args, ArgumentMap());
    runtime->execute_index_space(ctx, launcher, &out_reqs);
  }

  if (global) {
    TaskLauncher launcher(TID_CONSUMER_PROJ_GLOBAL, task_args);
    RegionRequirement req0(
        out_reqs[0].parent, READ_ONLY, EXCLUSIVE, out_reqs[0].parent);
    req0.add_field(FID_X);
    launcher.add_region_requirement(req0);
    RegionRequirement req1(
        out_reqs[1].parent, READ_ONLY, EXCLUSIVE, out_reqs[1].parent);
    req1.add_field(FID_X);
    launcher.add_region_requirement(req1);
    runtime->execute_task(ctx, launcher);
  }
  else {
    Domain launch_domain(colors);
    IndexTaskLauncher launcher(
      TID_CONSUMER_PROJ_LOCAL, launch_domain, task_args, ArgumentMap());
    RegionRequirement req0(
        out_reqs[0].partition, 0, READ_ONLY, EXCLUSIVE, out_reqs[0].parent);
    req0.add_field(FID_X);
    launcher.add_region_requirement(req0);
    RegionRequirement req1(
        out_reqs[1].partition, 0, READ_ONLY, EXCLUSIVE, out_reqs[1].parent);
    req1.add_field(FID_X);
    launcher.add_region_requirement(req1);
    runtime->execute_index_space(ctx, launcher);
  }
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

  basic_test(args, ctx, runtime);
  runtime->issue_execution_fence(ctx);
  projection_test(args, ctx, runtime, true);
  runtime->issue_execution_fence(ctx);
  projection_test(args, ctx, runtime, false);
}

static void create_mappers(Machine machine,
                           Runtime *runtime,
                           const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    OutReqTestMapper* mapper = new OutReqTestMapper(
      runtime->get_mapper_runtime(), machine, *it, "out_req_test_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }

  runtime->register_projection_functor(
      PID_ROW_MAJOR, new RowMajorTransform(runtime), true /*silence warnings*/);
  runtime->register_projection_functor(
      PID_COL_MAJOR, new ColMajorTransform(runtime), true /*silence warnings*/);
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
    TaskVariantRegistrar registrar(TID_PRODUCER_GLOBAL, "producer_global");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(false);
    registrar.set_inner(false);
    Runtime::preregister_task_variant<producer_global_task>(registrar, "producer_global");
  }
  {
    TaskVariantRegistrar registrar(TID_PRODUCER_LOCAL, "producer_local");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    registrar.set_inner(false);
    registrar.leaf_pool_bounds.emplace(Memory::Kind::SYSTEM_MEM,
        PoolBounds(SIZE*SIZE*sizeof(int64_t)));
    Runtime::preregister_task_variant<producer_local_task>(registrar, "producer_local");
  }
  {
    TaskVariantRegistrar registrar(TID_CONSUMER_GLOBAL, "consumer_global");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<consumer_global_task>(registrar, "consumer_global");
  }
  {
    TaskVariantRegistrar registrar(TID_CONSUMER_LOCAL, "consumer_local");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<consumer_local_task>(registrar, "consumer_local");
  }
  {
    TaskVariantRegistrar registrar(TID_CONDITION, "condition");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<bool, condition_task>(registrar, "condition");
  }
  {
    TaskVariantRegistrar registrar(TID_PRODUCER_PROJ_GLOBAL, "producer_proj_global");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    registrar.set_inner(false);
    Runtime::preregister_task_variant<producer_proj_global_task>(registrar, "producer_proj_global");
  }
  {
    TaskVariantRegistrar registrar(TID_PRODUCER_PROJ_LOCAL, "producer_proj_local");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    registrar.set_inner(false);
    Runtime::preregister_task_variant<producer_proj_local_task>(registrar, "producer_proj_local");
  }
  {
    TaskVariantRegistrar registrar(TID_CONSUMER_PROJ_GLOBAL, "consumer_proj_global");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    registrar.set_inner(false);
    Runtime::preregister_task_variant<consumer_proj_global_task>(registrar, "consumer_proj_global");
  }
  {
    TaskVariantRegistrar registrar(TID_CONSUMER_PROJ_LOCAL, "consumer_proj_local");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    registrar.set_inner(false);
    Runtime::preregister_task_variant<consumer_proj_local_task>(registrar, "consumer_proj_local");
  }
  Runtime::add_registration_callback(create_mappers);

  Runtime::set_top_level_task_id(TID_MAIN);

  Runtime::start(argc, argv);

  return 0;
}
