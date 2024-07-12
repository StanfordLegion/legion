/* Copyright 2021 Stanford University
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
#include "legion.h"
#include "mappers/default_mapper.h"

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INTRA_IS_ORDERING_TASK_ID,
};

enum FieldIDs {
  FID_DATA,
};

enum ProjectionIDs
{
  PID_INTRA_IS_TASK_ORDERING = 100
};

typedef Legion::ProjectionFunctor ProjectionFunctor;

class IntraIsTaskOrderingProjectionFunctor : public ProjectionFunctor {
public:
  using ProjectionFunctor::project;
  virtual Legion::LogicalRegion project(Legion::LogicalRegion upper_bound,
                                const Legion::DomainPoint &point,
                                const Domain &launch_domain) override
  {
    return upper_bound;
  }
  virtual void invert(Legion::LogicalRegion region, Legion::LogicalRegion upper,
                      const Legion::Domain &launch_domain,
                      std::vector<Legion::DomainPoint> &ordered_points) override
  {
   for (Domain::DomainPointIterator itr(launch_domain); itr; itr++)
   {
     ordered_points.push_back(itr.p);
    }
  }
  virtual unsigned get_depth(void) const override { return 0; }
  virtual bool is_functional(void) const override { return true; }
  virtual bool is_invertible(void) const override { return true; }
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_points = 4;
  int num_iterations = 1;
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-i"))
        num_iterations = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-p"))
        num_points = atoi(command_args.argv[++i]);
    }
  }
  printf("Running with %d points and %d iterations...\n",
      num_points, num_iterations);

  Rect<1> launch_bounds(0, num_points-1);
  IndexSpaceT<1> launch_is = runtime->create_index_space(ctx, launch_bounds);

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(uint64_t), FID_DATA);
  }
  LogicalRegion lr = runtime->create_logical_region(ctx, launch_is, fs);

  const uint64_t zero = 0;
  FillLauncher fill_launcher(lr, lr, TaskArgument(&zero, sizeof(zero)));
  fill_launcher.add_field(FID_DATA);
  runtime->fill_fields(ctx, fill_launcher);

  ArgumentMap arg_map;

  IndexLauncher intra_is_ordering_task_launcher(INTRA_IS_ORDERING_TASK_ID, launch_is,
                                TaskArgument(NULL, 0), arg_map);
  intra_is_ordering_task_launcher.add_region_requirement(
      RegionRequirement(lr, PID_INTRA_IS_TASK_ORDERING/*projection ID*/,
                        LEGION_READ_WRITE, LEGION_EXCLUSIVE, lr));
  intra_is_ordering_task_launcher.add_field(0, FID_DATA);

  for (int idx = 1; idx <= num_iterations; idx++)
  {
    runtime->execute_index_space(ctx, intra_is_ordering_task_launcher);
  }

  runtime->destroy_index_space(ctx, launch_is);
}

void intra_is_ordering_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  const Point<1> point = task->index_point;
  printf("Executing task %lld\n", point.x());
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INTRA_IS_ORDERING_TASK_ID, "intra_is_ordering_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<intra_is_ordering_task>(registrar, "intra_is_ordering_task");
  }

  Runtime::preregister_projection_functor(PID_INTRA_IS_TASK_ORDERING,
            new IntraIsTaskOrderingProjectionFunctor());

  return Runtime::start(argc, argv);
}
