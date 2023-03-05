/* Copyright 2023 Stanford University, NVIDIA Corporation
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
using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_TASK_ID,
  TILING_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_VALUE,
};

// tile sizes do not need to evenly divide dimensions
#define EXTENT 100 
#define TILE_SIZE 32

using value_t = uint32_t;

void
top_level_task(
  const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx,
  Runtime *runtime) {

  Rect<3> rect(Point<3>::ZEROES(), Point<3>(EXTENT, EXTENT, EXTENT));
  IndexSpace is = runtime->create_index_space(ctx, rect);
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(value_t), FID_VALUE);
  }

  LogicalRegion lr_in = runtime->create_logical_region(ctx, is, fs);
  LogicalRegion lr_out = runtime->create_logical_region(ctx, is, fs);

  {
    TaskLauncher init_launcher(INIT_TASK_ID, TaskArgument());
    init_launcher.add_region_requirement(
      RegionRequirement(lr_in, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_in));
    init_launcher.region_requirements[0].add_field(FID_VALUE);
    runtime->execute_task(ctx, init_launcher);
  }
  {
    TaskLauncher tiling_launcher(TILING_TASK_ID, TaskArgument());
    tiling_launcher.add_region_requirement(
      RegionRequirement(lr_in, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_in));
    tiling_launcher.region_requirements[0].add_field(FID_VALUE);
    tiling_launcher.add_region_requirement(
      RegionRequirement(lr_out, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_out));
    tiling_launcher.region_requirements[1].add_field(FID_VALUE);
    runtime->execute_task(ctx, tiling_launcher);
  }
  {
    TaskLauncher check_launcher(CHECK_TASK_ID, TaskArgument());
    check_launcher.add_region_requirement(
      RegionRequirement(lr_in, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_in));
    check_launcher.region_requirements[0].add_field(FID_VALUE);
    check_launcher.add_region_requirement(
      RegionRequirement(lr_out, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_out));
    check_launcher.region_requirements[1].add_field(FID_VALUE);
    runtime->execute_task(ctx, check_launcher);
  }

  runtime->destroy_logical_region(ctx, lr_in);
  runtime->destroy_logical_region(ctx, lr_out);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
}

void
init_task(
  const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  const FieldAccessor<
    LEGION_WRITE_DISCARD,value_t,3,coord_t,
    Realm::AffineAccessor<value_t,3,coord_t>>
    acc(regions[0], FID_VALUE);
  Rect<3> rect =
    runtime->get_index_space_domain(
      ctx,
      task->regions[0].region.get_index_space());
  for (PointInRectIterator<3> pir(rect); pir(); pir++)
    acc[*pir] = (pir[0] + 1) * (pir[1] + 1) * (pir[2] + 1);
}

void
tiling_task(
  const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);

  const FieldAccessor<
    LEGION_READ_ONLY,value_t,3,coord_t,
    Realm::AffineAccessor<value_t,3,coord_t>>
    acc_in(regions[0], FID_VALUE);
  
  // Iterate over each of the tiles
  for (PieceIterator pit(regions[1], FID_VALUE); pit; pit++) {
    // Get the rectangle for this tile
    const Rect<3> rect = *pit;
    // Make an accessor for the tile
    const FieldAccessor<
      LEGION_WRITE_DISCARD,value_t,3,coord_t,
      Realm::AffineAccessor<value_t,3,coord_t>>
      acc_out(regions[1], FID_VALUE, rect);
    // Fill in the points for the tile
    for (PointInRectIterator<3> pir(rect, true/*col major*/); pir(); pir++)
      acc_out[*pir] = acc_in[*pir];
  }
}

void
check_task(
  const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);

  const FieldAccessor<
    LEGION_READ_ONLY,value_t,3,coord_t,
    Realm::AffineAccessor<value_t,3,coord_t>>
    acc_in(regions[0], FID_VALUE);
  const FieldAccessor<
    LEGION_READ_ONLY,value_t,3,coord_t,
    Realm::AffineAccessor<value_t,3,coord_t>>
    acc_out(regions[1], FID_VALUE);
  Rect<3> rect =
    runtime->get_index_space_domain(
      ctx,
      task->regions[0].region.get_index_space());
  for (PointInRectIterator<3> pir(rect); pir(); pir++)
    assert(acc_out[*pir] == acc_in[*pir]);
}

int
main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  // Layout the initial data un-tiled in row-major (C) order
  LayoutConstraintID initial_layout;
  {
    OrderingConstraint order(true/*contiguous*/);
    order.ordering.push_back(LEGION_DIM_Z);
    order.ordering.push_back(LEGION_DIM_Y);
    order.ordering.push_back(LEGION_DIM_X);
    order.ordering.push_back(LEGION_DIM_F);
    LayoutConstraintRegistrar registrar;
    registrar.add_constraint(order);
    initial_layout = Runtime::preregister_layout(registrar);
  }

  // Create a tiled layout over two of the three dimensions
  // You can tile over any or all dimensions
  LayoutConstraintID tiled_layout;
  {
    OrderingConstraint order(true/*contiguous*/);
    // Layout tile dimensions in column major (Fortran) order
    order.ordering.push_back(LEGION_DIM_X);
    order.ordering.push_back(LEGION_DIM_Y);
    order.ordering.push_back(LEGION_DIM_Z);
    order.ordering.push_back(LEGION_DIM_F);
    // Only tile the Y and Z dimensions
    TilingConstraint tile_y(LEGION_DIM_Y, TILE_SIZE);
    TilingConstraint tile_z(LEGION_DIM_Z, TILE_SIZE);
    LayoutConstraintRegistrar registrar;
    // Ordering of tiling constraints determines order tiles are laid out
    // Layout tiles along Z dimension first then followed by Y dimension
    registrar.add_constraint(order).add_constraint(tile_z).add_constraint(tile_y);
    tiled_layout = Runtime::preregister_layout(registrar);
  }

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INIT_TASK_ID, "initialize");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(0, initial_layout);
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_task>(registrar, "initialize");
  }

  {
    TaskVariantRegistrar registrar(TILING_TASK_ID, "tiling");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(0, initial_layout);
    registrar.add_layout_constraint_set(1, tiled_layout);
    registrar.set_leaf();
    Runtime::preregister_task_variant<tiling_task>(registrar, "tiling");
  }

  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(0, initial_layout);
    registrar.add_layout_constraint_set(1, initial_layout);
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  return Runtime::start(argc, argv);
}
