/* Copyright 2015 Stanford University, NVIDIA Corporation
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

#include <cmath>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>

#include "kmeans.h"
#include "legion.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

LegionRuntime::Logger::Category log_kmeans("kmeans");

void parse_input_args(int &num_points, int &num_centers);

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  int num_points = 1500;
  int num_centers = 10;
  parse_input_args(num_points, num_centers);
  log_kmeans.print("Running kmeans with %d points and %d centers", 
                    num_points, num_centers);

  // Make our two logical regions
  PointSet points(ctx, runtime, num_points); 
  CenterSet centers(ctx, runtime, num_centers);

  // Figure out how many chunks to partition our points into
  const int num_chunks = runtime->get_tunable_value(ctx, CHUNK_TUNABLE);
  points.partition_set(num_points, num_chunks);

  // Set up the queue for pending futures
  std::deque<Future> pending_energies;
  const size_t max_energy_depth = runtime->get_tunable_value(ctx, PREDICATION_DEPTH_TUNABLE);

  // Initialize our input with some random points
  InitializeTask init_task(points, centers, &num_centers);
  init_task.dispatch(ctx, runtime);

  Predicate loop_pred = Predicate::TRUE_PRED;

  // Do the first energy computation
  KmeansEnergyTask init_energy_task(points, centers, loop_pred);
  Future energy = init_energy_task.dispatch(ctx, runtime);
  pending_energies.push_back(energy);

  // Keep iterating until we converge, note that all tasks
  // issued in an iteration are predicated on the previous
  // iteration not having converged
  int iteration = 0;
  while (true) {
    // Update the locations of the centers
    UpdateCentersTask update_center_task(centers, loop_pred);
    update_center_task.dispatch(ctx, runtime);

    // Recompute the kmeans energy
    KmeansEnergyTask energy_task(points, centers, loop_pred);
    Future next_energy = energy_task.dispatch(ctx, runtime);

    // Issue our test for convergence so we can get the predicate for the next iteration
    ConvergenceTask convergence_test(pending_energies.back(), next_energy, loop_pred);
    loop_pred = convergence_test.dispatch(ctx, runtime);
    pending_energies.push_back(next_energy);

    // If we are far enough ahead then test to see if we actually converged
    // so we can stop issuing predicated tasks
    if (pending_energies.size() >= max_energy_depth) {
      // Pop the first future off the queue
      double old_energy = pending_energies.front().get_result<double>();
      log_kmeans.print("Energy is %.8g for iteration %d", old_energy, iteration++); 
      pending_energies.pop_front();
      // Check it against the next future
      double new_energy = pending_energies.front().get_result<double>();
      if (old_energy <= new_energy) {
        log_kmeans.print("Converged at energy %.8g on iteration %d", old_energy, iteration);
        break;
      }
    }
  }
}

void parse_input_args(int &num_points, int &num_centers)
{
  const InputArgs &input_args = HighLevelRuntime::get_input_args();
  int argc = input_args.argc;
  char **argv = input_args.argv;
  for (int i = 1; i < argc; i++) {
    if (!strcmp("-p", argv[i])) {
      num_points = atoi(argv[++i]);
      continue;
    }
    if (!strcmp("-k", argv[i])) {
      num_centers = atoi(argv[++i]);
      continue;
    }
    log_kmeans.warning("WARNING: Unknown command line option %s", argv[i]);
  }
}

void mapper_registration_callback(Machine machine, HighLevelRuntime *rt, 
                                  const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(new KmeansMapper(machine, rt, *it), *it);
  }
}

int main(int argc, char **argv)
{
  // Register a callback for performing mapper registration
  HighLevelRuntime::set_registration_callback(mapper_registration_callback);
  // Register our task variants
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/,
      AUTO_GENERATE_ID, TaskConfigOptions(), "top_level");
  InitializeTask::register_variants();
  UpdateCentersTask::register_variants();
  KmeansEnergyTask::register_variants();
  ConvergenceTask::register_variants();
  // Register our reduciton operations
  HighLevelRuntime::register_reduction_op<IntegerSum>(INT_SUM_REDUCTION);
  HighLevelRuntime::register_reduction_op<DoubleSum>(DOUBLE_SUM_REDUCTION);
  // Start the runtime
  return HighLevelRuntime::start(argc, argv);
}

PointSet::PointSet(Context c, HighLevelRuntime *rt, int num_points)
  : ctx(c), runtime(rt), partition(LogicalPartition::NO_PART)
{
  Rect<1> rect(Point<1>(0), Point<1>(num_points-1));
  IndexSpace is = runtime->create_index_space(ctx, Domain::from_rect<1>(rect));
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  allocator.allocate_field(sizeof(double), FID_LOCATION);
  handle = runtime->create_logical_region(ctx, is, fs);
}

PointSet::PointSet(const PointSet &rhs)
  : ctx(rhs.ctx), runtime(rhs.runtime)
{
  // should never be called
  assert(false);
}

PointSet::~PointSet(void)
{
  runtime->destroy_logical_region(ctx, handle);
  runtime->destroy_field_space(ctx, handle.get_field_space());
  runtime->destroy_index_space(ctx, handle.get_index_space());
}

PointSet& PointSet::operator=(const PointSet &rhs)
{
  // should never be called
  assert(false);
  return *this;
}

void PointSet::partition_set(int num_points, int num_chunks)
{
  assert(partition == LogicalPartition::NO_PART);
  Point<1> blocking_factor(num_points/num_chunks);
  Blockify<1> blocking_map(blocking_factor);

  IndexPartition ip = runtime->create_index_partition(ctx, handle.get_index_space(),
                                                      blocking_map);
  partition = runtime->get_logical_partition(ctx, handle, ip);
  color_domain = runtime->get_index_partition_color_space(ctx, ip);
}

CenterSet::CenterSet(Context c, HighLevelRuntime *rt, int num_centers)
  : ctx(c), runtime(rt)
{
  Rect<1> rect(Point<1>(0), Point<1>(num_centers-1));
  IndexSpace is = runtime->create_index_space(ctx, Domain::from_rect<1>(rect));
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  allocator.allocate_field(sizeof(double), FID_LOCATION);
  allocator.allocate_field(sizeof(double), FID_PENDING_SUM);
  allocator.allocate_field(sizeof(int), FID_PENDING_COUNT);
  handle = runtime->create_logical_region(ctx, is, fs);
  // Fill in the two fields that won't be set by initialize
  runtime->fill_field<double>(ctx, handle, handle, FID_PENDING_SUM, 0.0);
  runtime->fill_field<int>(ctx, handle, handle, FID_PENDING_COUNT, 0);
}

CenterSet::CenterSet(const CenterSet &rhs)
  : ctx(rhs.ctx), runtime(rhs.runtime)
{
  // should never be called
  assert(false);
}

CenterSet::~CenterSet(void)
{
  runtime->destroy_logical_region(ctx, handle);
  runtime->destroy_field_space(ctx, handle.get_field_space());
  runtime->destroy_index_space(ctx, handle.get_index_space());
}

CenterSet& CenterSet::operator=(const CenterSet &rhs)
{
  // should never be called
  assert(false);
  return *this;
}

InitializeTask::InitializeTask(const PointSet &points, const CenterSet &centers,
                               int *num_centers)
  : TaskLauncher(INITIALIZE_TASK_ID, TaskArgument(num_centers, sizeof(int)))
{  
  RegionRequirement rr_points(points.get_region(), WRITE_DISCARD,
                              EXCLUSIVE, points.get_region());
  rr_points.add_field(PointSet::FID_LOCATION);
  add_region_requirement(rr_points);

  RegionRequirement rr_centers(centers.get_region(), WRITE_DISCARD,
                               EXCLUSIVE, centers.get_region());
  rr_centers.add_field(CenterSet::FID_LOCATION);
  add_region_requirement(rr_centers);
}

void InitializeTask::dispatch(Context ctx, HighLevelRuntime *runtime)
{
  runtime->execute_task(ctx, *this);
}

/*static*/
void InitializeTask::cpu_variant(const Task *task,
                                 const std::vector<PhysicalRegion> &regions,
                                 Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 2);  
  RegionAccessor<AccessorType::Generic, double> acc_points = 
    regions[0].get_field_accessor(PointSet::FID_LOCATION).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_centers = 
    regions[1].get_field_accessor(CenterSet::FID_LOCATION).typeify<double>();

  Domain point_dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Domain center_dom = runtime->get_index_space_domain(ctx,
      task->regions[1].region.get_index_space());

  for (Domain::DomainPointIterator itr(point_dom); itr; itr++) 
  {
    double point = drand48() * 256.0;
    acc_points.write(itr.p, point);
  }
  const size_t total_points = point_dom.get_volume();
  for (Domain::DomainPointIterator itr(center_dom); itr; itr++)
  {
    Point<1> location(lrand48() % total_points);
    double point = acc_points.read(DomainPoint::from_point<1>(location));
    acc_centers.write(itr.p, point);
  }
}

/*static*/
void InitializeTask::register_variants(void)
{
  HighLevelRuntime::register_legion_task<cpu_variant>(INITIALIZE_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "initialize_cpu");
}

KmeansEnergyTask::KmeansEnergyTask(const PointSet &points,
                                   const CenterSet &centers,
                                   const Predicate &pred)
  : IndexLauncher(KMEANS_ENERGY_TASK_ID, points.get_domain(), 
                  TaskArgument(), ArgumentMap(), pred)
{
  RegionRequirement rr_points(points.get_partition(), 0, READ_ONLY,
                              EXCLUSIVE, points.get_region());
  rr_points.add_field(PointSet::FID_LOCATION);
  add_region_requirement(rr_points);

  RegionRequirement rr_centers(centers.get_region(), READ_ONLY,
                               EXCLUSIVE, centers.get_region());
  rr_centers.add_field(CenterSet::FID_LOCATION);
  add_region_requirement(rr_centers);

  RegionRequirement rr_sum(centers.get_region(), DOUBLE_SUM_REDUCTION,
                           EXCLUSIVE, centers.get_region());
  rr_sum.add_field(CenterSet::FID_PENDING_SUM);
  add_region_requirement(rr_sum);

  RegionRequirement rr_count(centers.get_region(), INT_SUM_REDUCTION,
                             EXCLUSIVE, centers.get_region());
  rr_count.add_field(CenterSet::FID_PENDING_COUNT);
  add_region_requirement(rr_count);
}

Future KmeansEnergyTask::dispatch(Context ctx, HighLevelRuntime *runtime)
{
  // Have to set a false result in case predication fails
  double false_result = 0.0;
  set_predicate_false_result(TaskArgument(&false_result, sizeof(false_result)));
  return runtime->execute_index_space(ctx, *this, DOUBLE_SUM_REDUCTION);
}

static inline double norm2(double x, double y)
{
  return (x-y)*(x-y);
}

/*static*/
double KmeansEnergyTask::cpu_variant(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 4);

  RegionAccessor<AccessorType::Generic, double> acc_points = 
    regions[0].get_field_accessor(PointSet::FID_LOCATION).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_centers = 
    regions[1].get_field_accessor(CenterSet::FID_LOCATION).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_sum = 
    regions[2].get_accessor().typeify<double>();
  RegionAccessor<AccessorType::Generic, int> acc_count = 
    regions[3].get_accessor().typeify<int>();

  double total_energy = 0.0;

  Domain point_dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Domain center_dom = runtime->get_index_space_domain(ctx,
      task->regions[1].region.get_index_space());

  size_t total_centers = center_dom.get_volume();
  double *centers = (double*)malloc(total_centers * sizeof(double));
  unsigned center_idx = 0;
  for (Domain::DomainPointIterator itr(center_dom); itr; itr++, center_idx++)
    centers[center_idx] = acc_centers.read(itr.p);

  for (Domain::DomainPointIterator itr(point_dom); itr; itr++)
  {
    double point = acc_points.read(itr.p);
    // Figure out the closest center
    int nearest_idx = -1;
    double nearest_distance = INFINITY; // start at infinity
    for (unsigned idx = 0; idx < total_centers; idx++) {
      double distance = norm2(point, centers[idx]);
      if (distance < nearest_distance) {
        nearest_idx = idx;
        nearest_distance = distance;
      } 
    }
    assert(nearest_idx >= 0);
    acc_sum.reduce<DoubleSum>(nearest_idx, point);
    acc_count.reduce<IntegerSum>(nearest_idx, 1);
    total_energy += nearest_distance;
  }

  free(centers);

  return total_energy;
}

/*static*/
void KmeansEnergyTask::register_variants(void)
{
  HighLevelRuntime::register_legion_task<double,cpu_variant>(KMEANS_ENERGY_TASK_ID,
      Processor::LOC_PROC, false/*single*/, true/*index*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "kmeans_energy_cpu");
}

UpdateCentersTask::UpdateCentersTask(const CenterSet &centers,
                                     const Predicate &pred)
 : TaskLauncher(UPDATE_CENTERS_TASK_ID, TaskArgument(), pred)
{
  RegionRequirement rr_input(centers.get_region(), READ_WRITE,
                             EXCLUSIVE, centers.get_region());
  rr_input.add_field(CenterSet::FID_PENDING_SUM);
  rr_input.add_field(CenterSet::FID_PENDING_COUNT);
  add_region_requirement(rr_input);
  
  RegionRequirement rr_output(centers.get_region(), WRITE_DISCARD,
                              EXCLUSIVE, centers.get_region());
  rr_output.add_field(CenterSet::FID_LOCATION);
  add_region_requirement(rr_output);
}

void UpdateCentersTask::dispatch(Context ctx, HighLevelRuntime *runtime)
{
  runtime->execute_task(ctx, *this);
}

/*static*/
void UpdateCentersTask::cpu_variant(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 2);

  RegionAccessor<AccessorType::Generic, double> acc_sum = 
    regions[0].get_field_accessor(CenterSet::FID_PENDING_SUM).typeify<double>();
  RegionAccessor<AccessorType::Generic, int> acc_count = 
    regions[0].get_field_accessor(CenterSet::FID_PENDING_COUNT).typeify<int>();

  RegionAccessor<AccessorType::Generic, double> acc_loc = 
    regions[1].get_field_accessor(CenterSet::FID_LOCATION).typeify<double>();

  Domain center_dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
  for (Domain::DomainPointIterator itr(center_dom); itr; itr++)
  {
    double sum = acc_sum.read(itr.p);
    int count = acc_count.read(itr.p);
    double new_location = sum / double(count);
    acc_loc.write(itr.p, new_location);
    // Reset the sums and counts too
    acc_sum.write(itr.p, 0.0);
    acc_count.write(itr.p, 0);
  }
}

/*static*/
void UpdateCentersTask::register_variants(void)
{
  HighLevelRuntime::register_legion_task<cpu_variant>(UPDATE_CENTERS_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "update_centers_cpu");
}

ConvergenceTask::ConvergenceTask(const Future &prev, const Future &next,
                                 const Predicate &pred)
  : TaskLauncher(CONVERGENCE_TASK_ID, TaskArgument(), pred)
{
  add_future(prev);
  add_future(next);
}

Predicate ConvergenceTask::dispatch(Context ctx, HighLevelRuntime *runtime)
{
  // If we end up predicated false, then we have converged
  bool false_result = true;
  set_predicate_false_result(TaskArgument(&false_result, sizeof(false_result)));
  Future converged = runtime->execute_task(ctx, *this);
  Predicate conv_pred = runtime->create_predicate(ctx, converged);
  // Return the negated predicate since we only 
  // want to continue executing if we haven't converged
  // And it together with previous predicate to handle the
  // case where we've already converged
  return runtime->predicate_and(ctx, predicate, 
          runtime->predicate_not(ctx, conv_pred));
}

/*static*/
bool ConvergenceTask::cpu_variant(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, HighLevelRuntime *runtime)
{
  assert(task->futures.size() == 2);
  Future prev_future = task->futures[0];
  Future next_future = task->futures[1];
  double prev = prev_future.get_result<double>();
  double next = next_future.get_result<double>();
  return (prev <= next);
}

/*static*/
void ConvergenceTask::register_variants(void)
{
  HighLevelRuntime::register_legion_task<bool,cpu_variant>(CONVERGENCE_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "convergence_cpu");
}

const int IntegerSum::identity = 0;

template<>
void IntegerSum::apply<true>(LHS &lhs, RHS rhs)
{
  lhs += rhs;
}

template<>
void IntegerSum::apply<false>(LHS &lhs, RHS rhs)
{
  __sync_fetch_and_add(&lhs, rhs);
}

template<>
void IntegerSum::fold<true>(RHS &rhs1, RHS rhs2)
{
  rhs1 += rhs2;
}

template<>
void IntegerSum::fold<false>(RHS &rhs1, RHS rhs2)
{
  __sync_fetch_and_add(&rhs1, rhs2);
}

const double DoubleSum::identity = 0.0;

template<>
void DoubleSum::apply<true>(LHS &lhs, RHS rhs)
{
  lhs += rhs;
}

template<>
void DoubleSum::apply<false>(LHS &lhs, RHS rhs)
{
  uint64_t *target = (uint64_t*)&lhs;
  union { uint64_t as_int; double as_float; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs;
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

template<>
void DoubleSum::fold<true>(RHS &rhs1, RHS rhs2)
{
  rhs1 += rhs2;
}

template<>
void DoubleSum::fold<false>(RHS &rhs1, RHS rhs2)
{
  uint64_t *target = (uint64_t*)&rhs1;
  union { uint64_t as_int; double as_float; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs2;
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

KmeansMapper::KmeansMapper(Machine m, HighLevelRuntime *rt, Processor p)
  : DefaultMapper(m, rt, p) 
{
}

int KmeansMapper::get_tunable_value(const Task *task, TunableID tid, MappingTagID tag)
{
  int result = -1;
  switch (tid)
  {
    case CHUNK_TUNABLE:
      {
        // Make as many chunks as we have CPUs
        std::set<Processor> all_cpus;
        machine_interface.filter_processors(machine, Processor::LOC_PROC, all_cpus);
        result = all_cpus.size();
        break;
      }
    case PREDICATION_DEPTH_TUNABLE:
      {
        // A depth of four should be good enough to hide 
        // the latency for most machines
        result = 4;
        break;
      }
    default:
      assert(false);
  }
  return result;
}

