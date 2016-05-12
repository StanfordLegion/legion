/* Copyright 2016 Stanford University
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
#include <unistd.h>
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
  CHECK_FIELD_TASK_ID,
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
  int num_elements;
  int num_subregions;
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

  // we're going to use a must epoch launcher, so we need at least as many
  //  processors in our system as we have subregions - check that now
  std::set<Processor> all_procs;
  Realm::Machine::get_machine().get_all_processors(all_procs);
  int num_loc_procs = 0;
  for(std::set<Processor>::const_iterator it = all_procs.begin();
      it != all_procs.end();
      it++)
    if((*it).kind() == Processor::LOC_PROC)
      num_loc_procs++;

  if(num_loc_procs < num_subregions) {
    printf("FATAL ERROR: This test uses a must epoch launcher, which requires\n");
    printf("  a separate Realm processor for each subregion.  %d of the necessary\n",
	   num_loc_procs);
    printf("  %d are available.  Please rerun with '-ll:cpu %d'.\n",
	   num_subregions, num_subregions);
    exit(1);
  }

  // For this example we'll create a single index space tree, but we
  // will make different logical regions from this index space.  The
  // index space will have two levels of partitioning.  One level for
  // describing the partioning into pieces, and then a second level 
  // for capturing partitioning to describe ghost regions.
  Rect<1> elem_rect(Point<1>(0),Point<1>(num_elements-1));
  IndexSpace is = runtime->create_index_space(ctx,
                          Domain::from_rect<1>(elem_rect));
  runtime->attach_name(is, "is");
  
  FieldSpace ghost_fs = runtime->create_field_space(ctx);
  runtime->attach_name(ghost_fs, "ghost_fs");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, ghost_fs);
    allocator.allocate_field(sizeof(double),FID_GHOST);
    runtime->attach_name(ghost_fs, FID_GHOST, "GHOST");
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
  runtime->attach_name(disjoint_ip, "disjoint_ip");
  // Now iterate over each of the sub-regions and make the ghost partitions
  Rect<1> ghost_bounds(Point<1>((int)GHOST_LEFT),Point<1>((int)GHOST_RIGHT));
  Domain ghost_domain = Domain::from_rect<1>(ghost_bounds);
  std::vector<LogicalRegion> ghost_left;
  std::vector<LogicalRegion> ghost_right;
  char buf[32]; const char* parts[2] = {"L", "R"};
  for (int color = 0; color < num_subregions; color++)
  {
    // Get each of the subspaces
    IndexSpace subspace = runtime->get_index_subspace(ctx, disjoint_ip, color);
    sprintf(buf, "disjoint_subspace_%d", color);
    runtime->attach_name(subspace, buf);
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
    sprintf(buf, "ghost_ip_%d", color);
    runtime->attach_name(ghost_ip, buf);
    // Make explicit logical regions for each of the ghost spaces
    for (int idx = GHOST_LEFT; idx <= GHOST_RIGHT; idx++)
    {
      IndexSpace ghost_space = runtime->get_index_subspace(ctx, ghost_ip, idx);
      LogicalRegion ghost_lr = 
        runtime->create_logical_region(ctx, ghost_space, ghost_fs);  
      sprintf(buf, "ghost_lr_%d_%s", color, parts[idx]);
      runtime->attach_name(ghost_lr, buf);
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
      args[color].num_elements = num_elements;
      args[color].num_subregions = num_subregions;
      args[color].num_steps = num_steps;

      TaskLauncher spmd_launcher(SPMD_TASK_ID,
                        TaskArgument(&args[color], sizeof(SPMDArgs)));
      // Our Left 
      spmd_launcher.add_region_requirement(
          RegionRequirement(ghost_left[color], READ_WRITE, 
                            SIMULTANEOUS, ghost_left[color]));
      spmd_launcher.region_requirements[0].flags = NO_ACCESS_FLAG;
      // Our Right
      spmd_launcher.add_region_requirement(
          RegionRequirement(ghost_right[color], READ_WRITE,
                            SIMULTANEOUS, ghost_right[color]));
      spmd_launcher.region_requirements[1].flags = NO_ACCESS_FLAG;
      // Left Ghost
      if (color == 0)
        spmd_launcher.add_region_requirement(
            RegionRequirement(ghost_right[num_subregions-1], READ_ONLY,
                              SIMULTANEOUS, ghost_right[num_subregions-1]));
      else
        spmd_launcher.add_region_requirement(
            RegionRequirement(ghost_right[color-1], READ_ONLY,
                              SIMULTANEOUS, ghost_right[color-1]));
      spmd_launcher.region_requirements[2].flags = NO_ACCESS_FLAG;
      // Right Ghost
      if (color == (num_subregions-1))
        spmd_launcher.add_region_requirement(
            RegionRequirement(ghost_left[0], READ_ONLY,
                              SIMULTANEOUS, ghost_left[0]));
      else
        spmd_launcher.add_region_requirement(
            RegionRequirement(ghost_left[color+1], READ_ONLY,
                              SIMULTANEOUS, ghost_left[color+1]));
      spmd_launcher.region_requirements[3].flags = NO_ACCESS_FLAG;
      for (unsigned idx = 0; idx < 4; idx++)
        spmd_launcher.add_field(idx, FID_GHOST);

      spmd_launcher.add_index_requirement(IndexSpaceRequirement(is,
								NO_MEMORY,
								is));

      DomainPoint point(color);
      must_epoch_launcher.add_single_task(point, spmd_launcher);
    }
    FutureMap fm = runtime->execute_must_epoch(ctx, must_epoch_launcher);
    // wait for completion at least
    fm.wait_all_results();

    printf("Test completed.\n");
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
    char buf[16];
    IndexSpace ghost_is = task->regions[0].region.get_index_space(); 
    IndexPartition ghost_ip = runtime->get_parent_index_partition(ctx, ghost_is);
    IndexSpace local_is = runtime->get_parent_index_space(ctx, ghost_ip);
    local_fs = runtime->create_field_space(ctx);
    sprintf(buf, "local_fs_%lld", task->index_point.get_index());
    runtime->attach_name(local_fs, buf);
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, local_fs);
    allocator.allocate_field(sizeof(double),FID_VAL);
    runtime->attach_name(local_fs, FID_VAL, "VAL");
    allocator.allocate_field(sizeof(double),FID_DERIV);
    runtime->attach_name(local_fs, FID_DERIV, "DERIV");
    local_lr = runtime->create_logical_region(ctx, local_is, local_fs);
    sprintf(buf, "local_lr_%lld", task->index_point.get_index());
    runtime->attach_name(local_lr, buf);
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
      if (s > 0) {
	// advance the barrier first - we're waiting for the next phase
	//  to start
        args->wait_empty[idx] = 
          runtime->advance_phase_barrier(ctx, args->wait_empty[idx]);
        copy_launcher.add_wait_barrier(args->wait_empty[idx]);
      }
      // When we are done with the copy, signal that the
      // destination instnace is now ready
      copy_launcher.add_arrival_barrier(args->notify_ready[idx]);
      runtime->issue_copy_operation(ctx, copy_launcher);
      // Once we've issued our copy operation, advance both of
      // the barriers to the next generation.
      args->notify_ready[idx] = 
        runtime->advance_phase_barrier(ctx, args->notify_ready[idx]);
    }

    // Acquire coherence on our left and right ghost regions
    for (unsigned idx = GHOST_LEFT; idx <= GHOST_RIGHT; idx++)
    {
      AcquireLauncher acquire_launcher(ghosts_lr[idx],
                                       ghosts_lr[idx],
                                       regions[2+idx]);
      acquire_launcher.add_field(FID_GHOST);
      // The acquire operation needs to wait for the data to
      // be ready to consume, so wait on the next phase of the
      // ready barrier.
      args->wait_ready[idx] = 
        runtime->advance_phase_barrier(ctx, args->wait_ready[idx]);
      acquire_launcher.add_wait_barrier(args->wait_ready[idx]);
      runtime->issue_acquire(ctx, acquire_launcher);
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
      release_launcher.add_field(FID_GHOST);
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

  // now check our results
  {
    TaskLauncher check_launcher(CHECK_FIELD_TASK_ID,
				TaskArgument(args, sizeof(SPMDArgs)));
    check_launcher.add_region_requirement(
			  RegionRequirement(local_lr, READ_ONLY,
                          EXCLUSIVE, local_lr));
    check_launcher.add_field(0, FID_DERIV);
    Future f = runtime->execute_task(ctx, check_launcher);
    int errors = f.get_result<int>();
    if(errors > 0) {
      printf("Errors detected in check task!\n");
      sleep(1); // let other tasks also report errors if they wish
      exit(1);
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
    // use a ramp with little ripples
    const int ripple_period = 4;
    const double ripple[ripple_period] = { 0, 0.25, 0, -0.25 };

    double value = (double)(pir.p[0]) + ripple[pir.p[0] % ripple_period];
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

  double window[2*ORDER+1];

  // we're going to perform the stencil computation with 4 iterators: read iterators
  //  for the left, main, and right rectangles, and a write iterator for the main
  //  rectangle (the read and write iterators for the main rectangle will effectively
  //  be offset by ORDER
  GenericPointInRectIterator<1> pir_left(left_rect);
  GenericPointInRectIterator<1> pir_main_read(main_rect);
  GenericPointInRectIterator<1> pir_main_write(main_rect);
  GenericPointInRectIterator<1> pir_right(right_rect);

  // Prime the window with the left data and the first ORDER elements of main
  for (int i = 0; i < ORDER; i++)
  {
    window[i] = left_ghost_acc.read(DomainPoint::from_point<1>((pir_left++).p));
    window[i + ORDER] = read_acc.read(DomainPoint::from_point<1>((pir_main_read++).p));
  }

  // now iterate over the main rectangle's write value, pulling from the right ghost
  //  data once the main read iterator is exhausted
  while (pir_main_write)
  {
    if (pir_main_read)
      window[2 * ORDER] = read_acc.read(DomainPoint::from_point<1>((pir_main_read++).p));
    else
      window[2 * ORDER] = right_ghost_acc.read(DomainPoint::from_point<1>((pir_right++).p));

    // only have calculation for ORDER == 2
    double deriv;
    switch(ORDER)
    {
      case 2: {
	deriv = (window[0] - 8.0 * window[1] +
		 8.0 * window[3] - window[4]);
#ifdef DEBUG_STENCIL_CALC
	printf("A: [%d] %g %g %g %g %g -> %g\n",
	       pir_main_write.p[0], 
	       window[0],
	       window[1],
	       window[2],
	       window[3],
	       window[4],
	       deriv);
#endif
	break;
      }

      default: assert(0);
    }
    
    write_acc.write(DomainPoint::from_point<1>((pir_main_write++).p),
		    deriv);

    // slide the window for the next point
    for (int j = 0; j < (2*ORDER); j++)
      window[j] = window[j+1];
  }

  // check that we exhausted all the iterators
  assert(!pir_left);
  assert(!pir_main_read);
  assert(!pir_main_write);
  assert(!pir_right);
}

int check_field_task(const Task *task,
		     const std::vector<PhysicalRegion> &regions,
		     Context ctx, HighLevelRuntime *runtime)
{
  SPMDArgs *args = (SPMDArgs*)task->args; 

  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, double> acc = 
    regions[0].get_field_accessor(fid).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();
  int errors = 0;
  for (GenericPointInRectIterator<1> pir(rect); pir; pir++)
  {
    // the derivative of a ramp with ripples is a constant function with ripples
    const int ripple_period = 4;
    const double deriv_ripple[ripple_period] = { 4.0, 0, -4.0, 0 };
    double exp_value = 12.0 + deriv_ripple[pir.p[0] % ripple_period];

    // correct for the wraparound cases
    if(pir.p[0] < ORDER) {
      // again only actually supporting ORDER == 2
      assert(ORDER == 2);
      if(pir.p[0] == 0) exp_value += -7.0 * args->num_elements;
      if(pir.p[1] == 1) exp_value += 1.0 * args->num_elements;
    }
    if(pir.p[0] >= (args->num_elements - ORDER)) {
      // again only actually supporting ORDER == 2
      assert(ORDER == 2);
      if(pir.p[0] == (args->num_elements - 1)) exp_value += -7.0 * args->num_elements;
      if(pir.p[1] == (args->num_elements - 2)) exp_value += 1.0 * args->num_elements;
    }

    double act_value = acc.read(DomainPoint::from_point<1>(pir.p));

    // polarity is important here - comparisons with NaN always return false
    bool ok = ((exp_value < 0) ? ((-act_value >= 0.99 * -exp_value) && 
				  (-act_value <= 1.01 * -exp_value)) :
	       (exp_value > 0) ? ((act_value >= 0.99 * exp_value) &&
				  (act_value <= 1.01 * exp_value)) :
 	                         (act_value == 0));

    if(!ok) {
      printf("ERROR: check for location %lld failed: expected=%g, actual=%g\n",
	     pir.p[0], exp_value, act_value);
      errors++;
    }
  }

  return errors;
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/,
      AUTO_GENERATE_ID, TaskConfigOptions(), "top_level");
  HighLevelRuntime::register_legion_task<spmd_task>(SPMD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(), "spmd");
  HighLevelRuntime::register_legion_task<init_field_task>(INIT_FIELD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true), "init");
  HighLevelRuntime::register_legion_task<stencil_field_task>(STENCIL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true), "stencil");
  HighLevelRuntime::register_legion_task<int, check_field_task>(CHECK_FIELD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*single*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true), "check");

  return HighLevelRuntime::start(argc, argv);
}

