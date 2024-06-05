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
#include "legion.h"
using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_TASK_ID,
  MULTIPLE_CONSTRAINT_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_VALUE,
  FID_Y,
};

#define EXTENT 4
#define NUM_DIMS 3
#define TILE_SIZE 2
#define IS_VOLUME 125
#define NUM_TILES (EXTENT+1)/TILE_SIZE + (EXTENT+1)%TILE_SIZE
using value_t = uint32_t;

template<typename T, unsigned DIM, typename COORD_T = coord_t>
        using AccessorRO = Legion::FieldAccessor< LEGION_READ_ONLY, T, DIM, COORD_T,
	Realm::AffineAccessor<T, DIM, COORD_T>>;

template<typename T, unsigned DIM, typename COORD_T = coord_t>
        using AccessorWD = Legion::FieldAccessor< LEGION_WRITE_DISCARD, T, DIM, COORD_T,
	Realm::AffineAccessor<T, DIM, COORD_T>>;

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
    allocator.allocate_field(sizeof(value_t), FID_Y);

  }

  LogicalRegion lr_in = runtime->create_logical_region(ctx, is, fs);
  LogicalRegion lr_out = runtime->create_logical_region(ctx, is, fs);

  {
    TaskLauncher init_launcher(INIT_TASK_ID, TaskArgument());
    init_launcher.add_region_requirement(
      RegionRequirement(lr_in, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_in));
    init_launcher.region_requirements[0].add_field(FID_VALUE);
    init_launcher.region_requirements[0].add_field(FID_Y);
    runtime->execute_task(ctx, init_launcher);
  }
  {
    TaskLauncher multiple_constraint_launcher(MULTIPLE_CONSTRAINT_TASK_ID, TaskArgument());
    multiple_constraint_launcher.add_region_requirement(
      RegionRequirement(lr_in, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_in));
    multiple_constraint_launcher.region_requirements[0].add_field(FID_VALUE);
    multiple_constraint_launcher.region_requirements[0].add_field(FID_Y);
    multiple_constraint_launcher.add_region_requirement(
      RegionRequirement(lr_out, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_out));
    multiple_constraint_launcher.region_requirements[1].add_field(FID_VALUE);
    multiple_constraint_launcher.region_requirements[1].add_field(FID_Y);
    runtime->execute_task(ctx, multiple_constraint_launcher);
  }
  {
    TaskLauncher check_launcher(CHECK_TASK_ID, TaskArgument());
    check_launcher.add_region_requirement(
      RegionRequirement(lr_in, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_in));
    check_launcher.region_requirements[0].add_field(FID_VALUE);
    check_launcher.region_requirements[0].add_field(FID_Y);
    check_launcher.add_region_requirement(
      RegionRequirement(lr_out, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_out));
    check_launcher.region_requirements[1].add_field(FID_VALUE);
    check_launcher.region_requirements[1].add_field(FID_Y);
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
  assert(task->regions[0].privilege_fields.size() == 2);

  const AccessorWD<value_t, 3> acc(regions[0], FID_VALUE);
  const AccessorWD<value_t, 3> acc_y(regions[0], FID_Y);
  Rect<3> rect =
    runtime->get_index_space_domain(
      ctx,
      task->regions[0].region.get_index_space());

  for (PointInRectIterator<3> pir(rect); pir(); pir++) {
    acc[*pir] = (pir[0] + 1) * (pir[1] + 1) * (pir[2] + 1);
    acc_y[*pir] = (pir[0] + 1) * (pir[1] + 1) * (pir[2] + 1);
  }
}


void
multiple_constraints_task(
  const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  unsigned num_rects=0;
  {
    const AccessorRO<value_t, 3> acc_in(regions[0], FID_VALUE);
    // Iterate over each of the tiles
    for (PieceIterator pit(regions[1], FID_VALUE); pit; pit++) {
      ++num_rects;
      const Rect<3> rect = *pit;
      const FieldAccessor<
	LEGION_WRITE_DISCARD,value_t,3,coord_t,
	Realm::AffineAccessor<value_t,3,coord_t>>
	acc_out(regions[1], FID_VALUE, rect);

      // Fill in the points for the tile
      for (PointInRectIterator<3> pir(rect); pir(); pir++)
	acc_out[*pir] = acc_in[*pir];
    }
  // if tiling constraint is satisfied for region requirement 1, field FID_VALUE, num_rects = NUM_TILES
  assert(num_rects==NUM_TILES);
  }

  {
    num_rects=0;
    for (PieceIterator pit(regions[1], FID_Y); pit; pit++) {
      ++num_rects;
      // Get the rectangle
      const Rect<3> rect = *pit;
      const FieldAccessor<
	LEGION_READ_ONLY,value_t,3,coord_t,
	Realm::AffineAccessor<value_t,3,coord_t>>
	acc_in(regions[0], FID_Y, rect);
      const FieldAccessor<
	LEGION_WRITE_DISCARD,value_t,3,coord_t,
	Realm::AffineAccessor<value_t,3,coord_t>>
	acc_out(regions[1], FID_Y, rect);
      for (PointInRectIterator<3> pir(rect); pir(); pir++)
	acc_out[*pir] = acc_in[*pir];
    }
    // if constraint is satified for region requirement 1, field FID_Y, num_rects = 1
    assert(num_rects==1);
  }
}

void
check_task(
  const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Rect<3> rect =
    runtime->get_index_space_domain(
				    ctx,
				    task->regions[0].region.get_index_space());
    const AccessorRO<value_t, 3> acc_in(regions[0], FID_VALUE);
    const AccessorRO<value_t, 3> acc_out(regions[1], FID_VALUE);
    for (PointInRectIterator<3> pir(rect, true /* column major */); pir(); pir++) {
      assert(acc_out[*pir] == acc_in[*pir]);
    }
    const AccessorRO<value_t, 3> acc_in_y(regions[0], FID_Y);
    const AccessorRO<value_t, 3> acc_out_y(regions[1], FID_Y);

    for (PointInRectIterator<3> pir(rect, true /* column major */); pir(); pir++) {
      assert(acc_out_y[*pir] == acc_in_y[*pir]);
    }
}

int
main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  // Layout the initial data in row-major (C) order, SOA
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

  // Layout column-major (F) order, SOA
  LayoutConstraintID column_layout;
  {
    OrderingConstraint order(true/*contiguous*/);
    order.ordering.push_back(LEGION_DIM_X);
    order.ordering.push_back(LEGION_DIM_Y);
    order.ordering.push_back(LEGION_DIM_Z);
    order.ordering.push_back(LEGION_DIM_F);
    LayoutConstraintRegistrar registrar;
    registrar.add_constraint(order);
    column_layout = Runtime::preregister_layout(registrar);
  }

  // Layout 1 dimension (x) tiled layout
  LayoutConstraintID tiled_layout;
  {
    OrderingConstraint order(true/*contiguous*/);
    // Layout tile dimensions in row major (C) order, SOA
    order.ordering.push_back(LEGION_DIM_Z);
    order.ordering.push_back(LEGION_DIM_Y);
    order.ordering.push_back(LEGION_DIM_X);
    order.ordering.push_back(LEGION_DIM_F);
    // tile x dimension
    TilingConstraint tile_x(LEGION_DIM_X, TILE_SIZE);
    LayoutConstraintRegistrar registrar;

    // Layout tiles along X dimension
    registrar.add_constraint(order).add_constraint(tile_x);
    tiled_layout = Runtime::preregister_layout(registrar);
  }

  // create a field constraint for FID_Y
   LayoutConstraintID field_layout;
   {
     LayoutConstraintRegistrar registrar;
     registrar.add_constraint(FieldConstraint(std::vector<FieldID>({FID_Y}), true, true));
     field_layout = Runtime::preregister_layout(registrar);
   }

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    // initial layout should apply to all fields in region requirement 0 and 1
    TaskVariantRegistrar registrar(INIT_TASK_ID, "initialize");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(0, initial_layout);
    registrar.add_layout_constraint_set(1, initial_layout);
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_task>(registrar, "initialize");
  }

  {
    TaskVariantRegistrar registrar(MULTIPLE_CONSTRAINT_TASK_ID, "multiple_constraints");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(0, initial_layout);
    // tiling should only apply to FID_VALUE
    registrar.add_layout_constraint_set(1, tiled_layout);
    // field constraint should apply to FID_Y
    registrar.add_layout_constraint_set(1, field_layout);
    registrar.set_leaf();
    Runtime::preregister_task_variant<multiple_constraints_task>(registrar, "multiple_constraints");
  }

  {
    // column layout should apply to all fields in region requirement 0 and 1
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(0, column_layout);
    registrar.add_layout_constraint_set(1, column_layout);
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  return Runtime::start(argc, argv);
}
