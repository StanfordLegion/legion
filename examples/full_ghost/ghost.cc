/* Copyright 2014 Stanford University
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
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

#define ORDER 2

enum {
  TOP_LEVEL_TASK_ID,
  SPMD_TASK_ID,
  INIT_FIELD_TASK_ID,
  STENCIL_TASK_ID,
};

enum {
  FID_VAL,
  FID_DERIV,
  FID_GHOST,
};

enum {
  GHOST_LEFT,
  GHOST_RIGHT,
};

struct SPMDArgs {
public:
  PhaseBarrier notify_ready[2];
  PhaseBarrier notify_empty[2];
  PhaseBarrier wait_ready[2];
  PhaseBarrier wait_empty[2];
  int num_steps;
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  int num_elements = 1024;
  int num_subregions = 4;
  int num_steps = 10;
  // Check for any command line arguments
  {
      const InputArgs &command_args = HighLevelRuntime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-b"))
        num_subregions = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-s"))
        num_steps = atoi(command_args.argv[++i]);
    }
  }
  // This algorithm needs at least two sub-regions to work
  assert(num_subregions > 1);
  printf("Running stencil computation for %d elements for %d steps...\n", 
          num_elements, num_steps);
  printf("Partitioning data into %d sub-regions...\n", num_subregions);

  // For this example we'll create a single index space tree, but we
  // will make different logical regions from this index space.  The
  // index space will have two levels of partitioning.  One level for
  // describing the partioning into pieces, and then a second level 
  // for capturing partitioning to describe ghost regions.
  Rect<1> elem_rect(Point<1>(0),Point<1>(num_elements-1));
  IndexSpace is = runtime->create_index_space(ctx,
                          Domain::from_rect<1>(elem_rect));
  
  FieldSpace ghost_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, ghost_fs);
    allocator.allocate_field(sizeof(double),FID_GHOST);
  }

  Rect<1> color_bounds(Point<1>(0),Point<1>(num_subregions-1));
  Domain color_domain = Domain::from_rect<1>(color_bounds);

  // Create the partition for pieces
  IndexPartition disjoint_ip;
  {
    const int lower_bound = num_elements/num_subregions;
    const int upper_bound = lower_bound+1;
    const int number_small = num_subregions - (num_elements % num_subregions);
    DomainColoring disjoint_coloring;
    int index = 0;
    for (int color = 0; color < num_subregions; color++)
    {
      int num_elmts = color < number_small ? lower_bound : upper_bound;
      assert((index+num_elmts) <= num_elements);
      Rect<1> subrect(Point<1>(index),Point<1>(index+num_elmts-1));
      disjoint_coloring[color] = Domain::from_rect<1>(subrect);
      index += num_elmts;
    }
    disjoint_ip = runtime->create_index_partition(ctx, is, color_domain,
                                    disjoint_coloring, true/*disjoint*/);
  }
  // Now iterate over each of the sub-regions and make the ghost partitions
  Rect<1> ghost_bounds(Point<1>((int)GHOST_LEFT),Point<1>((int)GHOST_RIGHT));
  Domain ghost_domain = Domain::from_rect<1>(ghost_bounds);
  std::vector<LogicalRegion> ghost_left;
  std::vector<LogicalRegion> ghost_right;
  for (int color = 0; color < num_subregions; color++)
  {
    // Get each of the subspaces
    IndexSpace subspace = runtime->get_index_subspace(ctx, disjoint_ip, color);
    Domain dom = runtime->get_index_space_domain(ctx, subspace);
    Rect<1> rect = dom.get_rect<1>(); 
    // Make two sub-regions, one on the left, and one on the right
    DomainColoring ghost_coloring;
    Rect<1> left(rect.lo, rect.lo[0]+(ORDER-1));
    ghost_coloring[GHOST_LEFT] = Domain::from_rect<1>(left);
    Rect<1> right(rect.hi[0]-(ORDER-1),rect.hi);
    ghost_coloring[GHOST_RIGHT] = Domain::from_rect<1>(right);
    IndexPartition ghost_ip =
      runtime->create_index_partition(ctx, subspace, ghost_domain,
                                      ghost_coloring, true/*disjoint*/);
    // Make explicit logical regions for each of the ghost spaces
    for (int idx = GHOST_LEFT; idx <= GHOST_RIGHT; idx++)
    {
      IndexSpace ghost_space = runtime->get_index_subspace(ctx, ghost_ip, idx);
      LogicalRegion ghost_lr = 
        runtime->create_logical_region(ctx, ghost_space, ghost_fs);  
      if (idx == GHOST_LEFT)
        ghost_left.push_back(ghost_lr);
      else
        ghost_right.push_back(ghost_lr);
    }
  }

  // Create all of the phase barriers for this computation
  std::vector<PhaseBarrier> left_ready_barriers;
  std::vector<PhaseBarrier> left_empty_barriers;
  std::vector<PhaseBarrier> right_ready_barriers;
  std::vector<PhaseBarrier> right_empty_barriers;
  for (int color = 0; color < num_subregions; color++)
  {
    left_ready_barriers.push_back(runtime->create_phase_barrier(ctx, 1));
    left_empty_barriers.push_back(runtime->create_phase_barrier(ctx, 1));
    right_ready_barriers.push_back(runtime->create_phase_barrier(ctx, 1));
    right_empty_barriers.push_back(runtime->create_phase_barrier(ctx, 1));
  }

  // In order to guarantee that all of our spmd_tasks execute in parallel
  // we have to use a must epoch launcher.  This instructs the runtime
  // to check that all of the operations in the must epoch are capable of
  // executing in parallel making it possible for them to synchronize using
  // named barriers with potential deadlock.  If for some reason they
  // cannot run in parallel, the runtime will report an error and indicate
  // the cause of it.
  {
    MustEpochLauncher must_epoch_launcher;
    // Need a separate array for storing these until we call the runtime
    std::vector<SPMDArgs> args(num_subregions);
    // For each of our parallel tasks launch off a task with the ghost regions
    // for its neighbors as well as our ghost regions and the  necessary phase 
    // barriers.  Assume periodic boundary conditions.
    for (int color = 0; color < num_subregions; color++)
    {
      args[color].notify_ready[GHOST_LEFT] = left_ready_barriers[color];
      args[color].notify_ready[GHOST_RIGHT] = right_ready_barriers[color];
      args[color].wait_empty[GHOST_LEFT] = left_empty_barriers[color];
      args[color].wait_empty[GHOST_RIGHT] = right_empty_barriers[color];
      if (color == 0)
      {
        args[color].wait_ready[GHOST_LEFT] = right_ready_barriers[num_subregions-1];
        args[color].notify_empty[GHOST_LEFT] = right_empty_barriers[num_subregions-1];
      }
      else
      {
        args[color].wait_ready[GHOST_LEFT] = right_ready_barriers[color-1];
        args[color].notify_empty[GHOST_LEFT] = right_empty_barriers[color-1];
      }
      if (color == (num_subregions-1))
      {
        args[color].wait_ready[GHOST_RIGHT] = left_ready_barriers[0];
        args[color].notify_empty[GHOST_RIGHT] = left_empty_barriers[0];
      }
      else
      {
        args[color].wait_ready[GHOST_RIGHT] = left_ready_barriers[color+1];
        args[color].notify_empty[GHOST_RIGHT] = left_empty_barriers[color+1];
      }
      args[color].num_steps = num_steps;

      TaskLauncher spmd_launcher(SPMD_TASK_ID,
                        TaskArgument(&args[color], sizeof(SPMDArgs)));
      // Our Left 
      spmd_launcher.add_region_requirement(
          RegionRequirement(ghost_left[color], READ_WRITE, 
                            SIMULTANEOUS, ghost_left[color]));
      spmd_launcher.region_requirements[0].flags |= NO_ACCESS_FLAG;
      // Our Right
      spmd_launcher.add_region_requirement(
          RegionRequirement(ghost_right[color], READ_WRITE,
                            SIMULTANEOUS, ghost_right[color]));
      spmd_launcher.region_requirements[1].flags |= NO_ACCESS_FLAG;
      // Left Ghost
      if (color == 0)
        spmd_launcher.add_region_requirement(
            RegionRequirement(ghost_left[num_subregions-1], READ_ONLY,
                              SIMULTANEOUS, ghost_left[num_subregions-1]));
      else
        spmd_launcher.add_region_requirement(
            RegionRequirement(ghost_left[color-1], READ_ONLY,
                              SIMULTANEOUS, ghost_left[color-1]));
      spmd_launcher.region_requirements[2].flags |= NO_ACCESS_FLAG;
      // Right Ghost
      if (color == (num_subregions-1))
        spmd_launcher.add_region_requirement(
            RegionRequirement(ghost_right[0], READ_ONLY,
                              SIMULTANEOUS, ghost_right[0]));
      else
        spmd_launcher.add_region_requirement(
            RegionRequirement(ghost_right[color+1], READ_ONLY,
                              SIMULTANEOUS, ghost_right[color+1]));
      spmd_launcher.region_requirements[3].flags |= NO_ACCESS_FLAG;
      for (unsigned idx = 0; idx < 4; idx++)
        spmd_launcher.add_field(idx, FID_GHOST);

      DomainPoint point(color);
      must_epoch_launcher.add_single_task(point, spmd_launcher);
    }
    runtime->execute_must_epoch(ctx, must_epoch_launcher);
  }


  // Clean up our mess when we are done
  for (unsigned idx = 0; idx < ghost_left.size(); idx++)
    runtime->destroy_logical_region(ctx, ghost_left[idx]);
  for (unsigned idx = 0; idx < ghost_right.size(); idx++)
    runtime->destroy_logical_region(ctx, ghost_right[idx]);
  for (unsigned idx = 0; idx < left_ready_barriers.size(); idx++)
    runtime->destroy_phase_barrier(ctx, left_ready_barriers[idx]);
  for (unsigned idx = 0; idx < left_empty_barriers.size(); idx++)
    runtime->destroy_phase_barrier(ctx, left_empty_barriers[idx]);
  for (unsigned idx = 0; idx < right_ready_barriers.size(); idx++)
    runtime->destroy_phase_barrier(ctx, right_ready_barriers[idx]);
  for (unsigned idx = 0; idx < right_empty_barriers.size(); idx++)
    runtime->destroy_phase_barrier(ctx, right_empty_barriers[idx]);
  ghost_left.clear();
  ghost_right.clear();
  left_ready_barriers.clear();
  left_empty_barriers.clear();
  right_ready_barriers.clear();
  right_empty_barriers.clear();
  runtime->destroy_index_space(ctx, is);
  runtime->destroy_field_space(ctx, ghost_fs);
}

void spmd_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, HighLevelRuntime *runtime)
{
  // Unmap all the regions we were given since we won't actually use them
  runtime->unmap_all_regions(ctx);

  SPMDArgs *args = (SPMDArgs*)task->args; 
  LogicalRegion neighbor_lr[2];
  neighbor_lr[GHOST_LEFT] = task->regions[0].region;
  neighbor_lr[GHOST_RIGHT] = task->regions[1].region;
  LogicalRegion ghosts_lr[2];
  ghosts_lr[GHOST_LEFT] = task->regions[2].region;
  ghosts_lr[GHOST_RIGHT] = task->regions[3].region;
  // Create the logical region that we'll use for our data
  FieldSpace local_fs;
  LogicalRegion local_lr;
  {
    IndexSpace ghost_is = task->regions[0].region.get_index_space(); 
    IndexPartition ghost_ip = runtime->get_parent_index_partition(ctx, ghost_is);
    IndexSpace local_is = runtime->get_parent_index_space(ctx, ghost_ip);
    local_fs = runtime->create_field_space(ctx);
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, local_fs);
    allocator.allocate_field(sizeof(double),FID_VAL);
    allocator.allocate_field(sizeof(double),FID_DERIV);
    local_lr = runtime->create_logical_region(ctx, local_is, local_fs);
  }
  // Run a bunch of steps
  for (int s = 0; s < args->num_steps; s++)
  {
    // Launch a task to initialize our field with some data
    TaskLauncher init_launcher(INIT_FIELD_TASK_ID,
                                TaskArgument(NULL, 0));
    init_launcher.add_region_requirement(
        RegionRequirement(local_lr, WRITE_DISCARD,
                          EXCLUSIVE, local_lr));
    init_launcher.add_field(0, FID_VAL);
    runtime->execute_task(ctx, init_launcher);

    // Issue explicit region-to-region copies
    for (unsigned idx = GHOST_LEFT; idx <= GHOST_RIGHT; idx++)
    {
      CopyLauncher copy_launcher;
      copy_launcher.add_copy_requirements(
          RegionRequirement(local_lr, READ_ONLY,
                            EXCLUSIVE, local_lr),
          RegionRequirement(neighbor_lr[idx], WRITE_DISCARD,
                            EXCLUSIVE, neighbor_lr[idx]));
      copy_launcher.add_src_field(0, FID_VAL);
      copy_launcher.add_dst_field(0, FID_GHOST);
      // It's not safe to issue the copy until we know
      // that the destination instance is empty. Only
      // need to do this after the first iteration.
      if (s > 0)
        copy_launcher.add_wait_barrier(args->wait_empty[idx]);
      // When we are done with the copy, signal that the
      // destination instnace is now ready
      copy_launcher.add_arrival_barrier(args->notify_ready[idx]);
      runtime->issue_copy_operation(ctx, copy_launcher);
      // Once we've issued our copy operation, advance both of
      // the barriers to the next generation.
      if (s > 0)
        args->wait_empty[idx] = 
          runtime->advance_phase_barrier(ctx, args->wait_empty[idx]);
      args->notify_ready[idx] = 
        runtime->advance_phase_barrier(ctx, args->notify_ready[idx]);
    }

    // Acquire coherence on our left and right ghost regions
    for (unsigned idx = GHOST_LEFT; idx <= GHOST_RIGHT; idx++)
    {
      AcquireLauncher acquire_launcher(ghosts_lr[idx],
                                       ghosts_lr[idx],
                                       regions[2+idx]);
      // The acquire operation needs to wait for the data to
      // be ready to consume, so wait on the ready barrier.
      acquire_launcher.add_wait_barrier(args->wait_ready[idx]);
      runtime->issue_acquire(ctx, acquire_launcher);
      // Now we can advance the wait ready barrier
      args->wait_ready[idx] = 
        runtime->advance_phase_barrier(ctx, args->wait_ready[idx]);
    }

    // Run the stencil computation
    TaskLauncher stencil_launcher(STENCIL_TASK_ID,
                                  TaskArgument(NULL, 0));
    stencil_launcher.add_region_requirement(
        RegionRequirement(local_lr, WRITE_DISCARD, EXCLUSIVE, local_lr));
    stencil_launcher.add_field(0, FID_DERIV);
    stencil_launcher.add_region_requirement(
        RegionRequirement(local_lr, READ_ONLY, EXCLUSIVE, local_lr));
    stencil_launcher.add_field(1, FID_VAL);
    for (unsigned idx = GHOST_LEFT; idx <= GHOST_RIGHT; idx++)
    {
      stencil_launcher.add_region_requirement(
          RegionRequirement(ghosts_lr[idx], READ_ONLY, EXCLUSIVE, ghosts_lr[idx]));
      stencil_launcher.add_field(idx+2, FID_GHOST);
    }
    runtime->execute_task(ctx, stencil_launcher);

    // Release coherence on our left and right ghost regions
    for (unsigned idx = GHOST_LEFT; idx <= GHOST_RIGHT; idx++)
    {
      ReleaseLauncher release_launcher(ghosts_lr[idx],
                                       ghosts_lr[idx],
                                       regions[2+idx]);
      // On all but the last iteration we need to signal that
      // we have now consumed the ghost instances and it is
      // safe to issue the next copy.
      if (s < (args->num_steps-1))
        release_launcher.add_arrival_barrier(args->notify_empty[idx]);
      runtime->issue_release(ctx, release_launcher);
      if (s < (args->num_steps-1))
        args->notify_empty[idx] = 
          runtime->advance_phase_barrier(ctx, args->notify_empty[idx]);
    }
  }
  runtime->destroy_logical_region(ctx, local_lr);
  runtime->destroy_field_space(ctx, local_fs);
}

void init_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, double> acc = 
    regions[0].get_field_accessor(fid).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();
  for (GenericPointInRectIterator<1> pir(rect); pir; pir++)
  {
    double value = drand48();
    acc.write(DomainPoint::from_point<1>(pir.p), value);
  }
}

void stencil_field_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  for (int idx = 0; idx < 4; idx++)
    assert(task->regions[idx].privilege_fields.size() == 1);

  FieldID write_fid = *(task->regions[0].privilege_fields.begin());
  FieldID read_fid = *(task->regions[1].privilege_fields.begin());
  FieldID ghost_fid = *(task->regions[2].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, double> write_acc = 
    regions[0].get_field_accessor(write_fid).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> read_acc = 
    regions[1].get_field_accessor(read_fid).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> left_ghost_acc = 
    regions[2].get_field_accessor(ghost_fid).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> right_ghost_acc = 
    regions[3].get_field_accessor(ghost_fid).typeify<double>();

  Domain main_dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Domain left_dom = runtime->get_index_space_domain(ctx,
      task->regions[2].region.get_index_space());
  Domain right_dom = runtime->get_index_space_domain(ctx,
      task->regions[3].region.get_index_space());

  Rect<1> left_rect = left_dom.get_rect<1>();
  Rect<1> right_rect = right_dom.get_rect<1>();
  Rect<1> main_rect = main_dom.get_rect<1>();
  // Break the main rect into left, main, and right components
  Rect<1> left_edge = main_rect;
  left_edge.hi = left_edge.lo[0] + (ORDER-1);
  Rect<1> right_edge = main_rect;
  right_edge.lo = right_edge.hi[0] - (ORDER-1);
  // Doctor the main bounds
  main_rect.lo = left_edge.hi;
  main_rect.lo = main_rect.lo[0] + 1;
  main_rect.hi = right_edge.lo;
  main_rect.hi = main_rect.hi[0] - 1;

  double window[2*ORDER+1];
  // Prime the window with the left data
  unsigned idx = 0;
  for (GenericPointInRectIterator<1> pir(left_rect); pir; pir++, idx++)
  {
    window[idx] = left_ghost_acc.read(DomainPoint::from_point<1>(pir.p));
  }
  for (GenericPointInRectIterator<1> pir(left_edge); pir; pir++, idx++)
  {
    window[idx] = read_acc.read(DomainPoint::from_point<1>(pir.p));
  }
  // This code assumes the order is 2
  assert(ORDER == 2);
  for (GenericPointInRectIterator<1> pir(main_rect); pir; pir++)
  {
    DomainPoint point = DomainPoint::from_point<1>(pir.p);
    window[2*ORDER] = read_acc.read(point);
    // Do the compuation
    double deriv = -window[0] + 8.0 * window[1] -
                   8.0 * window[3] + window[4];
    // Write the derivative two spots to the left
    point.point_data[0] -= ORDER;
    write_acc.write(point, deriv);
    // Shift down all the values in the window
    for (int j = 0; j < (2*ORDER); j++)
      window[j] = window[j+1];
  }
  // Finally handle the last few points on the edge
  GenericPointInRectIterator<1> oir(right_edge);
  for (GenericPointInRectIterator<1> pir(right_rect); pir; pir++, oir++)
  {
    window[2*ORDER] = right_ghost_acc.read(DomainPoint::from_point<1>(pir.p)); 
    // Do the computation
    double deriv = -window[0] + 8.0* window[1] -
                   8.0 * window[3] + window[4];
    write_acc.write(DomainPoint::from_point<1>(oir.p), deriv);
    for (int j = 0; j < (2*ORDER); j++)
      window[j] = window[j+1];
  }
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);
  HighLevelRuntime::register_legion_task<spmd_task>(SPMD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*single*/);
  HighLevelRuntime::register_legion_task<init_field_task>(INIT_FIELD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*single*/);
  HighLevelRuntime::register_legion_task<stencil_field_task>(STENCIL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*single*/);

  return HighLevelRuntime::start(argc, argv);
}

