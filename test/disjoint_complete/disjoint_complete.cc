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


#include <cmath>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  unsigned num_points = 16;
  unsigned num_elements = 1024;
  // See how many points to run
  const InputArgs &command_args = Runtime::get_input_args();
  for (int i = 1; i < command_args.argc; i++) {
    if (!strcmp(command_args.argv[i],"-p"))
      num_points = atoi(command_args.argv[++i]);
    else if (!strcmp(command_args.argv[i],"-n"))
      num_elements = atoi(command_args.argv[++i]);
  }
  assert(num_points > 0);
  assert(num_elements > 0);
  assert((4*num_points) < num_elements);
  printf("Running disjoint and complete tests for %d points...\n", num_points);

  Rect<2> elem_rect(Point<2>(0,0),Point<2>(num_elements-1,num_elements-1));
  IndexSpaceT<2> is = runtime->create_index_space(ctx, elem_rect);
  Rect<2> color_rect(Point<2>(0,0),Point<2>(num_points-1,num_points-1));
  IndexSpaceT<2> cs = runtime->create_index_space(ctx, color_rect);
  // Create a disjoint and complete partition (no information)
  {
    const coord_t chunk = (num_elements + num_points - 1) / num_points; 
    Transform<2,2> transform;
    transform[0][0] = chunk;
    transform[0][1] = 0;
    transform[1][0] = 0;
    transform[1][1] = chunk;
    Rect<2> extent(Point<2>(0,0),Point<2>(chunk-1,chunk-1));
    IndexPartition ip =
      runtime->create_partition_by_restriction(ctx, is, cs, transform, extent);
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    assert(runtime->is_index_partition_complete(ctx, ip));
  }
  // Create a disjoint and complete partition (known disjoint but not complete)
  {
    const coord_t chunk = (num_elements + num_points - 1) / num_points; 
    Transform<2,2> transform;
    transform[0][0] = chunk;
    transform[0][1] = 0;
    transform[1][0] = 0;
    transform[1][1] = chunk;
    Rect<2> extent(Point<2>(0,0),Point<2>(chunk-1,chunk-1));
    IndexPartition ip =
      runtime->create_partition_by_restriction(ctx, is, cs, transform, extent, LEGION_DISJOINT_KIND);
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    assert(runtime->is_index_partition_complete(ctx, ip));
  }
  // Create a disjoint and complete partition (known complete but not disjoint)
  {
    const coord_t chunk = (num_elements + num_points - 1) / num_points; 
    Transform<2,2> transform;
    transform[0][0] = chunk;
    transform[0][1] = 0;
    transform[1][0] = 0;
    transform[1][1] = chunk;
    Rect<2> extent(Point<2>(0,0),Point<2>(chunk-1,chunk-1));
    IndexPartition ip =
      runtime->create_partition_by_restriction(ctx, is, cs, transform, extent, LEGION_COMPUTE_COMPLETE_KIND);
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    assert(runtime->is_index_partition_complete(ctx, ip));
  }
  // Create a disjoint and incomplate partition (no information)
  {
    const coord_t chunk = (num_elements + num_points - 1) / num_points;
    const coord_t tile = (num_elements + 2*num_points - 1) / (2*num_points);
    Transform<2,2> transform;
    transform[0][0] = chunk;
    transform[0][1] = 0;
    transform[1][0] = 0;
    transform[1][1] = chunk;
    Rect<2> extent(Point<2>(0,0),Point<2>(tile-1,tile-1));
    IndexPartition ip =
      runtime->create_partition_by_restriction(ctx, is, cs, transform, extent);
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    assert(!runtime->is_index_partition_complete(ctx, ip));
  }
  // Create a disjoint and incomplate partition (known disjoint but not incomplete
  {
    const coord_t chunk = (num_elements + num_points - 1) / num_points;
    const coord_t tile = (num_elements + 2*num_points - 1) / (2*num_points);
    Transform<2,2> transform;
    transform[0][0] = chunk;
    transform[0][1] = 0;
    transform[1][0] = 0;
    transform[1][1] = chunk;
    Rect<2> extent(Point<2>(0,0),Point<2>(tile-1,tile-1));
    IndexPartition ip =
      runtime->create_partition_by_restriction(ctx, is, cs, transform, extent, LEGION_DISJOINT_KIND);
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    assert(!runtime->is_index_partition_complete(ctx, ip));
  }
  // Create a disjoint and incomplate partition (known incomplete but not disjoint)
  {
    const coord_t chunk = (num_elements + num_points - 1) / num_points;
    const coord_t tile = (num_elements + 2*num_points - 1) / (2*num_points);
    Transform<2,2> transform;
    transform[0][0] = chunk;
    transform[0][1] = 0;
    transform[1][0] = 0;
    transform[1][1] = chunk;
    Rect<2> extent(Point<2>(0,0),Point<2>(tile-1,tile-1));
    IndexPartition ip =
      runtime->create_partition_by_restriction(ctx, is, cs, transform, extent, LEGION_COMPUTE_INCOMPLETE_KIND);
    assert(runtime->is_index_partition_disjoint(ctx, ip));
    assert(!runtime->is_index_partition_complete(ctx, ip));
  }
  // Create an aliased and complete partition (no information)
  {
    const coord_t chunk = (num_elements + num_points - 1) / num_points;
    Transform<2,2> transform;
    transform[0][0] = chunk;
    transform[0][1] = 0;
    transform[1][0] = 0;
    transform[1][1] = chunk;
    Rect<2> extent(Point<2>(-1,-1),Point<2>(chunk,chunk));
    IndexPartition ip =
      runtime->create_partition_by_restriction(ctx, is, cs, transform, extent);
    assert(!runtime->is_index_partition_disjoint(ctx, ip));
    assert(runtime->is_index_partition_complete(ctx, ip));
  }
  // Create an aliased and complete partition (known aliased but not complete)
  {
    const coord_t chunk = (num_elements + num_points - 1) / num_points;
    Transform<2,2> transform;
    transform[0][0] = chunk;
    transform[0][1] = 0;
    transform[1][0] = 0;
    transform[1][1] = chunk;
    Rect<2> extent(Point<2>(-1,-1),Point<2>(chunk,chunk));
    IndexPartition ip =
      runtime->create_partition_by_restriction(ctx, is, cs, transform, extent, LEGION_ALIASED_KIND);
    assert(!runtime->is_index_partition_disjoint(ctx, ip));
    assert(runtime->is_index_partition_complete(ctx, ip));
  }
  // Create an aliased and complete partition (known aliased but not complete)
  {
    const coord_t chunk = (num_elements + num_points - 1) / num_points;
    Transform<2,2> transform;
    transform[0][0] = chunk;
    transform[0][1] = 0;
    transform[1][0] = 0;
    transform[1][1] = chunk;
    Rect<2> extent(Point<2>(-1,-1),Point<2>(chunk,chunk));
    IndexPartition ip =
      runtime->create_partition_by_restriction(ctx, is, cs, transform, extent, LEGION_COMPUTE_COMPLETE_KIND);
    assert(!runtime->is_index_partition_disjoint(ctx, ip));
    assert(runtime->is_index_partition_complete(ctx, ip));
  }
  // Create an aliased and incomplete partition (no information)
  {
    const coord_t chunk = (num_elements + (2*num_points) - 1) / (2*num_points);  
    Transform<2,2> transform;
    transform[0][0] = chunk;
    transform[0][1] = 0;
    transform[1][0] = 0;
    transform[1][1] = chunk;
    Rect<2> extent(Point<2>(-1,-1),Point<2>(chunk,chunk));
    IndexPartition ip =
      runtime->create_partition_by_restriction(ctx, is, cs, transform, extent);
    assert(!runtime->is_index_partition_disjoint(ctx, ip));
    assert(!runtime->is_index_partition_complete(ctx, ip));
  }
  // Create an aliased and incomplete partition (known aliased but not incomplete)
  {
    const coord_t chunk = (num_elements + (2*num_points) - 1) / (2*num_points);  
    Transform<2,2> transform;
    transform[0][0] = chunk;
    transform[0][1] = 0;
    transform[1][0] = 0;
    transform[1][1] = chunk;
    Rect<2> extent(Point<2>(-1,-1),Point<2>(chunk,chunk));
    IndexPartition ip =
      runtime->create_partition_by_restriction(ctx, is, cs, transform, extent, LEGION_ALIASED_KIND);
    assert(!runtime->is_index_partition_disjoint(ctx, ip));
    assert(!runtime->is_index_partition_complete(ctx, ip));
  }
  // Create an aliased and incomplete partition (known incomplete but not aliased)
  {
    const coord_t chunk = (num_elements + (2*num_points) - 1) / (2*num_points);  
    Transform<2,2> transform;
    transform[0][0] = chunk;
    transform[0][1] = 0;
    transform[1][0] = 0;
    transform[1][1] = chunk;
    Rect<2> extent(Point<2>(-1,-1),Point<2>(chunk,chunk));
    IndexPartition ip =
      runtime->create_partition_by_restriction(ctx, is, cs, transform, extent, LEGION_COMPUTE_INCOMPLETE_KIND);
    assert(!runtime->is_index_partition_disjoint(ctx, ip));
    assert(!runtime->is_index_partition_complete(ctx, ip));
  }
  runtime->destroy_index_space(ctx, is);
  runtime->destroy_index_space(ctx, cs);
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

  return Runtime::start(argc, argv);
}
