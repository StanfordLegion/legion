/* Copyright 2023 NVIDIA Corporation
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

// tests various forms of gather/scatter copies

#include <cstdio>
#include <cassert>
#include <cstdlib>

// uses deprecated Legion partitioning calls until we have compute_interference
#define LEGION_DEPRECATED(x)

#include "legion.h"
#include "mappers/null_mapper.h"
#include "realm/cmdline.h"
#include "philox.h"

using namespace Legion;

Logger log_app("app");

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_AFFINITY_TASK_ID,
  INIT_OWNED_TASK_ID,
  INIT_GHOST_TASK_ID,
  CHECK_GHOST_TASK_ID,
  INIT_FIELD_TASK_ID,
  DAXPY_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_AFFINITY,
  FID_GHOST_VAL,
  FID_GHOST_PTR,
  FID_OWNED_VAL,
  FID_X,
  FID_Y,
  FID_Z,
};

enum {
  PID_OWNED_IMAGE_PER_RANK = 55,
  PID_OWNED_IMAGE_BLOATED_PER_RANK,
  PID_GHOST_PREIMAGE_PER_RANK,
};

template <PrivilegeMode MODE>
struct PointerAccessor {
  typedef FieldAccessor<MODE, coord_t, 1, coord_t, Realm::AffineAccessor<coord_t,1,coord_t> > type;
};

class ProjectionOneLevel : public ProjectionFunctor {
public:
  ProjectionOneLevel(int _index1) : index1(_index1) {}
  virtual bool is_functional(void) const { return true; }
  virtual bool is_invertible(void) const { return true; }
  virtual unsigned get_depth(void) const { return 0; }
  using ProjectionFunctor::project;
  using ProjectionFunctor::invert;
  virtual LogicalRegion project(LogicalPartition upper_bound,
				const DomainPoint &point,
				const Domain &launch_domain)
  {
    return runtime->get_logical_subregion_by_color(upper_bound,
						   point[index1]);
  }
  virtual void invert(LogicalRegion region, LogicalPartition upper_bound,
                      const Domain &launch_domain,
                      std::vector<DomainPoint> &ordered_points)
  {
    // Invert this the dumb way for now since these are likely to be
    // small launch domains
    for (Domain::DomainPointIterator itr(launch_domain); itr; itr++)
    {
      LogicalRegion lr = project(upper_bound, itr.p, launch_domain);
      if (lr != region)
        continue;
      ordered_points.push_back(itr.p);
    }
  }

protected:
  int index1;
};

class ProjectionTwoLevel : public ProjectionFunctor {
public:
  ProjectionTwoLevel(int _index1, Color _color, int _index2) :
    index1(_index1), color(_color), index2(_index2) {}
  virtual bool is_functional(void) const { return true; }
  virtual unsigned get_depth(void) const { return 1; }
  using ProjectionFunctor::project;
  virtual LogicalRegion project(LogicalPartition upper_bound,
				const DomainPoint &point,
				const Domain &launch_domain)
  {
    LogicalRegion lr = runtime->get_logical_subregion_by_color(upper_bound,
							       point[index1]);
    LogicalPartition lp = runtime->get_logical_partition_by_color(lr, color);
    return runtime->get_logical_subregion_by_color(lp, point[index2]);
  }

protected:
  int index1;
  Color color;
  int index2;
};

// projection functor IDs
enum {
  PFID_START = 1000,
  PFID_IJ_TO_I,
  PFID_IJ_TO_J,
  PFID_IJ_TO_I_PREIMAGE_BY_RANK_J,
};

// shards N things onto M shard in coarse subgroups
class CoarseShardingFunctor : public ShardingFunctor {
public:
  virtual ShardID shard(const DomainPoint &point,
			const Domain &full_space,
			const size_t total_shards)
  {
    //log_app.print() << "shard: " << point << " " << full_space << " " << total_shards;
    int cur_point = point[0];
    int num_points = full_space.get_volume();
    int points_per_shard = (num_points + total_shards - 1) / total_shards;
    return (cur_point / points_per_shard);
  }
};

enum {
  SHARD_ID_COARSE = 2000,
};

enum {
  TRACE_ID_COPY = 3000,
};

namespace TestConfig {
  int num_pieces = 4;
  int num_owned_per_piece = 8;
  int num_ghost_per_piece = 16;
  int gather_mode = 1;
  int random_seed = 12345;
  int replicate = 1;
  int num_iterations = 1;
  // code to enable tracing exists, but tracing doesn't like use of fences
  int use_tracing = 0;
  char affinity_pattern[256] = "1d";
};

class InitOwnedTask {
public:
  static void init_owned(Runtime *runtime, Context ctx,
			 IndexSpace is_pieces,
			 LogicalRegion lr_owned, LogicalPartition lp_owned,
			 int offset)
  {
    IndexTaskLauncher itl(INIT_OWNED_TASK_ID,
			  is_pieces,
			  TaskArgument(&offset, sizeof(offset)),
			  ArgumentMap());

    itl.add_region_requirement(RegionRequirement(lp_owned, 0 /*identity*/,
						 WRITE_DISCARD, EXCLUSIVE,
						 lr_owned)
			       .add_field(FID_OWNED_VAL));

    runtime->execute_index_space(ctx, itl);    
  }

  static void register_tasks()
  {
    TaskVariantRegistrar registrar(INIT_OWNED_TASK_ID, "init_owned");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<InitOwnedTask::task_body>(registrar, "init_owned");
  }

  //protected:
  static void task_body(const Task *task,
			const std::vector<PhysicalRegion> &regions,
			Context ctx, Runtime *runtime)
  {
    int offset = *static_cast<const int *>(task->args);

    Rect<1> bounds = regions[0].get_bounds<1,coord_t>().bounds;

    log_app.info() << "init owned: bounds=" << bounds << " offset=" << offset;

    typedef FieldAccessor<WRITE_DISCARD, int, 1, coord_t, Realm::AffineAccessor<int,1,coord_t> > ValueAccessor;

    ValueAccessor acc_val(regions[0], FID_OWNED_VAL);

    for(int i = bounds.lo; i <= bounds.hi; i++)
      acc_val[i] = i + offset;
  }
};

class InitGhostPointersTask {
public:
  struct InitArgs {
    int random_seed;
    LogicalPartition lp_owned;
    bool with_replacement;
    bool sort_indices;
    // TODO: disallow duplicates?
  };
  
  static void init_pointers(Runtime *runtime, Context ctx,
			    IndexSpace is_pieces,
			    LogicalRegion lr_affinity,
			    LogicalRegion lr_ghost,
			    LogicalPartition lp_ghost,
			    LogicalPartition lp_owned,
			    int random_seed)
  {
    InitArgs args;
    args.random_seed = random_seed;
    args.lp_owned = lp_owned;
    args.with_replacement = true;
    args.sort_indices = false;

    IndexTaskLauncher itl(INIT_GHOST_TASK_ID,
			  is_pieces,
			  TaskArgument(&args, sizeof(args)),
			  ArgumentMap());

    itl.add_region_requirement(RegionRequirement(lr_affinity,
						 READ_ONLY, EXCLUSIVE,
						 lr_affinity)
			       .add_field(FID_AFFINITY));

    itl.add_region_requirement(RegionRequirement(lp_ghost, 0 /*identity*/,
						 WRITE_DISCARD, EXCLUSIVE,
						 lr_ghost)
			       .add_field(FID_GHOST_PTR));

    runtime->execute_index_space(ctx, itl);    
  }

  static void register_tasks()
  {
    TaskVariantRegistrar registrar(INIT_GHOST_TASK_ID, "init_ghost");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<InitGhostPointersTask::task_body>(registrar, "init_ghost");
  }

  //protected:
  static void task_body(const Task *task,
			const std::vector<PhysicalRegion> &regions,
			Context ctx, Runtime *runtime)
  {
    const InitArgs& args = *static_cast<const InitArgs *>(task->args);

    int index = task->index_point[0];
    Rect<1> bounds = regions[1].get_bounds<1,coord_t>().bounds;

    log_app.info() << "init ghost: index=" << index << " bounds=" << bounds;

    typedef FieldAccessor<READ_ONLY, int, 2, coord_t, Realm::AffineAccessor<int,2,coord_t> > AffinityAccessor;

    AffinityAccessor acc_aff(regions[0], FID_AFFINITY);

    // first read the affinity table to see which ranges we're supposed to
    //  point into and with which weights
    std::vector<int> weights;
    std::vector<Rect<1> > ranges;
    int total_weight = 0;

    Rect<2> aff_bounds = regions[0].get_bounds<2,coord_t>().bounds;

    for(int dest = aff_bounds.lo[1]; dest <= aff_bounds.hi[1]; dest++) {
      int w = acc_aff[index][dest];
      if(w > 0) {
	LogicalRegion lr = runtime->get_logical_subregion_by_color(ctx,
								   args.lp_owned,
								   dest);
	Rect<1> rng = runtime->get_index_space_domain(lr.get_index_space()).bounds<1,coord_t>();
	log_app.info() << "aff: src=" << index << " dest=" << dest << " weight=" << w << " range=" << rng;
	weights.push_back(w);
	ranges.push_back(rng);
	total_weight += w;
      }
    }

    int samples_needed = bounds.volume();
    
    if(!args.with_replacement) {
      // scale up weights to sum to at least the number of points we need
      if(total_weight < samples_needed) {
	int scale = 1 + ((samples_needed - 1) / total_weight);
	for(std::vector<int>::iterator it = weights.begin(); it != weights.end(); ++it)
	  (*it) *= scale;
	total_weight *= scale;
      }
    }

    PointerAccessor<WRITE_DISCARD>::type acc_ptr(regions[1], FID_GHOST_PTR);

    // keep a count of how many random numbers we've requested
    int ctr = 0;
    for(int i = bounds.lo; i <= bounds.hi; i++) {
      int d_index = 0;
      if(weights.size() > 1) {
	int rv = Philox_2x32<>::rand_int(args.random_seed, index, ctr++, total_weight);
	for(std::vector<int>::iterator it = weights.begin(); it != weights.end(); ++it) {
	  rv -= *it;
	  if(rv < 0) break;
	  d_index++;
	}
	assert(rv < 0);
      }
      if(!args.with_replacement) {
	weights[d_index]--;
	total_weight--;
      }

      // now pick a random point in the range
      coord_t c = (ranges[d_index].lo +
		   Philox_2x32<>::rand_int(args.random_seed, index, ctr++,
					   (ranges[d_index].hi - ranges[d_index].lo + 1)));

      log_app.debug() << "assign [" << i << "] = " << c;
      acc_ptr[i] = c;
    }
  }
};

class CheckGhostDataTask {
public:
  static int check_data(Runtime *runtime, Context ctx,
			IndexSpace is_pieces,
			LogicalRegion lr_ghost, LogicalPartition lp_ghost,
			int offset)
  {
    IndexTaskLauncher itl(CHECK_GHOST_TASK_ID,
			  is_pieces,
			  TaskArgument(&offset, sizeof(offset)),
			  ArgumentMap());

    itl.add_region_requirement(RegionRequirement(lp_ghost, 0 /*identity*/,
						 READ_ONLY, EXCLUSIVE,
						 lr_ghost)
			       .add_field(FID_GHOST_PTR)
			       .add_field(FID_GHOST_VAL));

    Future f = runtime->execute_index_space(ctx, itl,
					    LEGION_REDOP_SUM_INT32);

    return f.get_result<int>(false /*silence*/);
  }

  static void register_tasks()
  {
    TaskVariantRegistrar registrar(CHECK_GHOST_TASK_ID, "check_ghost");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<int, CheckGhostDataTask::task_body>(registrar, "check_ghost");
  }

  //protected:
  static int task_body(const Task *task,
			const std::vector<PhysicalRegion> &regions,
			Context ctx, Runtime *runtime)
  {
    int offset = *static_cast<const int *>(task->args);

    Rect<1> bounds = regions[0].get_bounds<1,coord_t>().bounds;

    log_app.info() << "check ghost: bounds=" << bounds << " offset=" << offset;

    typedef FieldAccessor<READ_ONLY, int, 1, coord_t, Realm::AffineAccessor<int,1,coord_t> > ValueAccessor;

    PointerAccessor<READ_ONLY>::type acc_ptr(regions[0], FID_GHOST_PTR);
    ValueAccessor acc_val(regions[0], FID_GHOST_VAL);

    int errors = 0;
    for(int i = bounds.lo; i <= bounds.hi; i++) {
      int expected = acc_ptr[i] + offset;
      int actual = acc_val[i];
      if(expected != actual)
	if(++errors <= 10)
	  log_app.error() << " [" << i << "]: exp=" << expected << " act=" << actual;
    }

    return errors;
  }
};

class InitAffinityTask {
public:
  struct InitArgs {
    int num_pieces;
  };

  static void init_matrix(Runtime *runtime, Context ctx,
			  LogicalRegion lr_affinity,
			  int num_pieces)
  {
    // affinity matrix starts out as all zeros
    runtime->fill_field<int>(ctx, lr_affinity, lr_affinity,
			     FID_AFFINITY, 0);

    InitArgs args;
    args.num_pieces = num_pieces;

    TaskLauncher tl(INIT_AFFINITY_TASK_ID,
		    TaskArgument(&args, sizeof(args)));

    tl.add_region_requirement(RegionRequirement(lr_affinity,
						READ_WRITE, EXCLUSIVE,
						lr_affinity)
			      .add_field(FID_AFFINITY));

    runtime->execute_task(ctx, tl);
  }

  static void register_tasks()
  {
    TaskVariantRegistrar registrar(INIT_AFFINITY_TASK_ID, "init_affinity");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<InitAffinityTask::task_body>(registrar, "init_affinity");
  }

  //protected:
  static void task_body(const Task *task,
			const std::vector<PhysicalRegion> &regions,
			Context ctx, Runtime *runtime)
  {
    const InitArgs& args = *static_cast<const InitArgs *>(task->args);

    FieldAccessor<READ_WRITE, int, 2, coord_t, Realm::AffineAccessor<int,2,coord_t> > acc(regions[0], FID_AFFINITY);

    if(!strcmp(TestConfig::affinity_pattern, "all")) {
      // all-pairs connectivity
      for(int i = 0; i < args.num_pieces; i++)
	for(int j = 0; j < args.num_pieces; j++)
	  if(i != j)
	    acc[i][j] = 1;
      return;
    }

    if(!strncmp(TestConfig::affinity_pattern, "1d", 2)) {
      // simple 1-D ring affinity
      int order = 1;
      if(TestConfig::affinity_pattern[2] != 0) {
	int count = sscanf(TestConfig::affinity_pattern+2, ":%d", &order);
	if(count != 1) {
	  log_app.fatal() << "affinity syntax: expected '1d[:n]', got '"
			  << TestConfig::affinity_pattern << "'";
	  abort();
	}
      }
      for(int i = 0; i < args.num_pieces; i++)
	for(int j = 1; j <= order; j++) {
	  acc[i][(i + j) % args.num_pieces] = 1;
	  acc[i][(i + args.num_pieces - j) % args.num_pieces] = 1;
	}
      return;
    }

    if(!strncmp(TestConfig::affinity_pattern, "2d", 2)) {
      int x, y, pts;
      int count = sscanf(TestConfig::affinity_pattern+2, ":%dx%d:%d",
			 &x, &y, &pts);
      if(count != 3) {
	log_app.fatal() << "affinity syntax: expected '2d:{x}x{y}:{pts}', got '"
			<< TestConfig::affinity_pattern << "'";
	abort();
      }
      if((x * y) != args.num_pieces) {
	log_app.fatal() << "affinity error: 2d grid size mismatch: "
			<< x << " x " << y << " != " << args.num_pieces;
	abort();
      }
      if((pts != 4) && (pts != 8)) {
	log_app.fatal() << "affinity error: 2d points must be 4 or 8, got: "
			<< pts;
	abort();
      }
      const int stencil_2d[][3] = { { -1, 0, 10 }, { 1, 0, 10 },
				    { 0, -1, 10 }, { 0, 1, 10 },
				    { -1, -1, 1 }, { -1, 1, 1 },
				    { 1, -1, 1 }, { 1, 1, 1 } };
      for(int j = 0; j < y; j++)
	for(int i = 0; i < x; i++) {
	  int src = i + (j * x);
	  for(int k = 0; k < pts; k++) {
	    int i2 = (i + x + stencil_2d[k][0]) % x;
	    int j2 = (j + y + stencil_2d[k][1]) % y;
	    int dst = i2 + (j2 * x);
	    acc[src][dst] = stencil_2d[k][2];
	  }
	}
      return;
    }

    if(!strncmp(TestConfig::affinity_pattern, "file:", 5)) {
      FILE *f = fopen(TestConfig::affinity_pattern+5, "r");
      if(!f) {
	log_app.fatal() << "affinity error: cannot read '" << (TestConfig::affinity_pattern+5) << "'";
	abort();
      }
      char line[80];
      while(fgets(line, 80, f)) {
	int src, dst, amt;
	int count = sscanf(line, "%d %d %d", &src, &dst, &amt);
	if(count != 3) continue;
	if((src < 0) || (src >= args.num_pieces) ||
	   (dst < 0) || (dst >= args.num_pieces)) {
	  log_app.fatal() << "affinity error: indices out of bounds in '" << line << "'";
	  abort();
	}
	if(amt > 0)
	  acc[src][dst] = amt;
      }
      fclose(f);

      return;
    }

    log_app.fatal() << "affinity syntax: unrecognized pattern: '"
		    << TestConfig::affinity_pattern << "'";
    abort();
  }
};

void do_gather_copy(Runtime *runtime, Context ctx,
		    int mode,
		    IndexSpace is_pieces,
		    IndexSpace is_affinity,
		    IndexSpace is_interference,
		    LogicalRegion lr_owned, LogicalPartition lp_owned,
		    LogicalPartition lp_owned_image,
		    LogicalPartition lp_owned_image_bloated,
		    LogicalRegion lr_ghost, LogicalPartition lp_ghost)
{
  switch(mode) {
    case 0: {
      // for i: ghost[i] = ptr[i]->owned
      IndexCopyLauncher icl(is_pieces);

      icl.add_copy_requirements(RegionRequirement(lr_owned,
						  READ_ONLY, EXCLUSIVE,
						  lr_owned)
				.add_field(FID_OWNED_VAL),
				RegionRequirement(lp_ghost, 0 /*identity*/,
						  WRITE_DISCARD, EXCLUSIVE,
						  lr_ghost)
				.add_field(FID_GHOST_VAL));

      icl.add_src_indirect_field(FID_GHOST_PTR,
				 RegionRequirement(lp_ghost, 0 /*identity*/,
						   READ_ONLY, EXCLUSIVE,
						   lr_ghost),
				 false /*!range*/);
      icl.possible_src_indirect_out_of_range = false;
      icl.collective_src_indirect_points = false;

      runtime->issue_copy_operation(ctx, icl);
      break;
    }

    case 1: {
      // for i: ghost[i] = ptr[i]->owned[*]
      IndexCopyLauncher icl(is_pieces);

      icl.add_copy_requirements(RegionRequirement(lp_owned, 0 /*identity*/,
						  READ_ONLY, EXCLUSIVE,
						  lr_owned)
				.add_field(FID_OWNED_VAL),
				RegionRequirement(lp_ghost, 0 /*identity*/,
						  WRITE_DISCARD, EXCLUSIVE,
						  lr_ghost)
				.add_field(FID_GHOST_VAL));

      icl.add_src_indirect_field(FID_GHOST_PTR,
				 RegionRequirement(lp_ghost, 0 /*identity*/,
						   READ_ONLY, EXCLUSIVE,
						   lr_ghost),
				 false /*!range*/);
      icl.possible_src_indirect_out_of_range = false;
      icl.collective_src_indirect_points = true;

      runtime->issue_copy_operation(ctx, icl);
      break;
    }

    case 2: {
      // for i: ghost[i] = ptr[i]->owned_image[i]
      IndexCopyLauncher icl(is_pieces);

      icl.add_copy_requirements(RegionRequirement(lp_owned_image,
						  0 /*identity*/,
						  READ_ONLY, EXCLUSIVE,
						  lr_owned)
				.add_field(FID_OWNED_VAL),
				RegionRequirement(lp_ghost, 0 /*identity*/,
						  WRITE_DISCARD, EXCLUSIVE,
						  lr_ghost)
				.add_field(FID_GHOST_VAL));

      icl.add_src_indirect_field(FID_GHOST_PTR,
				 RegionRequirement(lp_ghost, 0 /*identity*/,
						   READ_ONLY, EXCLUSIVE,
						   lr_ghost),
				 false /*!range*/);
      icl.possible_src_indirect_out_of_range = false;
      icl.collective_src_indirect_points = false;

      runtime->issue_copy_operation(ctx, icl);
      break;
    }

    case 3: {
      // for i: ghost[i] = ptr[i]->owned_image_bloated[i]
      IndexCopyLauncher icl(is_pieces);

      icl.add_copy_requirements(RegionRequirement(lp_owned_image_bloated,
						  0 /*identity*/,
						  READ_ONLY, EXCLUSIVE,
						  lr_owned)
				.add_field(FID_OWNED_VAL),
				RegionRequirement(lp_ghost, 0 /*identity*/,
						  WRITE_DISCARD, EXCLUSIVE,
						  lr_ghost)
				.add_field(FID_GHOST_VAL));

      icl.add_src_indirect_field(FID_GHOST_PTR,
				 RegionRequirement(lp_ghost, 0 /*identity*/,
						   READ_ONLY, EXCLUSIVE,
						   lr_ghost),
				 false /*!range*/);
      icl.possible_src_indirect_out_of_range = false;
      icl.collective_src_indirect_points = false;

      runtime->issue_copy_operation(ctx, icl);
      break;
    }

    case 4: {
      // for i,j: ghost[i].preimage[j] = ptr[i]->owned[j]
      IndexCopyLauncher icl(is_affinity);

      icl.add_copy_requirements(RegionRequirement(lp_owned,
						  PFID_IJ_TO_J,
						  READ_ONLY, EXCLUSIVE,
						  lr_owned)
				.add_field(FID_OWNED_VAL),
				RegionRequirement(lp_ghost,
						  PFID_IJ_TO_I_PREIMAGE_BY_RANK_J,
						  WRITE_DISCARD, EXCLUSIVE,
						  lr_ghost)
				.add_field(FID_GHOST_VAL));

      icl.add_src_indirect_field(FID_GHOST_PTR,
				 RegionRequirement(lp_ghost,
						   PFID_IJ_TO_I,
						   READ_ONLY, EXCLUSIVE,
						   lr_ghost),
				 false /*!range*/);
      icl.possible_src_indirect_out_of_range = false;
      icl.collective_src_indirect_points = false;

      runtime->issue_copy_operation(ctx, icl);
      break;
    }

    case 5: {
      // for i,j: ghost[i]{preimage[j]} = ptr[i]->owned[j]
      IndexCopyLauncher icl(is_affinity);

      icl.add_copy_requirements(RegionRequirement(lp_owned,
						  PFID_IJ_TO_J,
						  READ_ONLY, EXCLUSIVE,
						  lr_owned)
				.add_field(FID_OWNED_VAL),
				RegionRequirement(lp_ghost,
						  PFID_IJ_TO_I,
						  READ_WRITE, SIMULTANEOUS,
						  lr_ghost)
				.add_field(FID_GHOST_VAL));

      icl.add_src_indirect_field(FID_GHOST_PTR,
				 RegionRequirement(lp_ghost,
						   PFID_IJ_TO_I,
						   READ_ONLY, EXCLUSIVE,
						   lr_ghost),
				 false /*!range*/);
      icl.possible_src_indirect_out_of_range = true;
      icl.collective_src_indirect_points = false;

      runtime->issue_copy_operation(ctx, icl);
      break;
    }

    case 6: {
      // for i,j in is_interference:
      //   ghost[i].preimage[j] = ptr[i]->owned[j]
      IndexCopyLauncher icl(is_interference);

      icl.add_copy_requirements(RegionRequirement(lp_owned,
						  PFID_IJ_TO_J,
						  READ_ONLY, EXCLUSIVE,
						  lr_owned)
				.add_field(FID_OWNED_VAL),
				RegionRequirement(lp_ghost,
						  PFID_IJ_TO_I_PREIMAGE_BY_RANK_J,
						  WRITE_DISCARD, EXCLUSIVE,
						  lr_ghost)
				.add_field(FID_GHOST_VAL));

      icl.add_src_indirect_field(FID_GHOST_PTR,
				 RegionRequirement(lp_ghost,
						   PFID_IJ_TO_I,
						   READ_ONLY, EXCLUSIVE,
						   lr_ghost),
				 false /*!range*/);
      icl.possible_src_indirect_out_of_range = false;
      icl.collective_src_indirect_points = false;

      runtime->issue_copy_operation(ctx, icl);
      break;
    }

    case 7: {
      // for i,j in is_interference:
      //   ghost[i]{preimage[j]} = ptr[i]->owned[j]
      IndexCopyLauncher icl(is_interference);

      icl.add_copy_requirements(RegionRequirement(lp_owned,
						  PFID_IJ_TO_J,
						  READ_ONLY, EXCLUSIVE,
						  lr_owned)
				.add_field(FID_OWNED_VAL),
				RegionRequirement(lp_ghost,
						  PFID_IJ_TO_I,
						  READ_WRITE, SIMULTANEOUS,
						  lr_ghost)
				.add_field(FID_GHOST_VAL));

      icl.add_src_indirect_field(FID_GHOST_PTR,
				 RegionRequirement(lp_ghost,
						   PFID_IJ_TO_I,
						   READ_ONLY, EXCLUSIVE,
						   lr_ghost),
				 false /*!range*/);
      icl.possible_src_indirect_out_of_range = true;
      icl.collective_src_indirect_points = false;

      runtime->issue_copy_operation(ctx, icl);
      break;
    }

    default: assert(0);
  }
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  // construct the affinity matrix
  IndexSpace is_affinity;
  FieldSpace fs_affinity;
  LogicalRegion lr_affinity;
  {
    Rect<2> domain(Point<2>(0, 0), Point<2>(TestConfig::num_pieces - 1,
					    TestConfig::num_pieces - 1));
    is_affinity = runtime->create_index_space(ctx, domain);
    runtime->attach_name(is_affinity, "is_affinity");

    std::vector<size_t> sizes(1, sizeof(int));
    std::vector<FieldID> ids(1, FID_AFFINITY);
    fs_affinity = runtime->create_field_space(ctx, sizes, ids);
    runtime->attach_name(fs_affinity, "fs_affinity");
    runtime->attach_name(fs_affinity, FID_AFFINITY, "affinity");

    lr_affinity = runtime->create_logical_region(ctx,
						 is_affinity, fs_affinity);
    runtime->attach_name(lr_affinity, "lr_affinity");
  }

  // initialize affinity matrix inline
  InitAffinityTask::init_matrix(runtime, ctx, lr_affinity, TestConfig::num_pieces);

  // create the index space we'll to name pieces
  IndexSpace is_pieces;
  {
    Rect<1> domain(0, TestConfig::num_pieces - 1);
    is_pieces = runtime->create_index_space(ctx, domain);
  }

  // create and partition owned region
  IndexSpace is_owned;
  FieldSpace fs_owned;
  LogicalRegion lr_owned;
  LogicalPartition lp_owned;
  {
    Rect<1> domain(0,
		   TestConfig::num_pieces * TestConfig::num_owned_per_piece - 1);
    is_owned = runtime->create_index_space(ctx, domain);
    runtime->attach_name(is_owned, "is_owned");

    std::vector<size_t> sizes(1, sizeof(int));
    std::vector<FieldID> ids(1, FID_OWNED_VAL);
    fs_owned = runtime->create_field_space(ctx, sizes, ids);
    runtime->attach_name(fs_owned, "fs_owned");
    runtime->attach_name(fs_owned, FID_OWNED_VAL, "owned_val");

    lr_owned = runtime->create_logical_region(ctx, is_owned, fs_owned);
    runtime->attach_name(lr_owned, "lr_owned");

    IndexPartition ip_owned = runtime->create_partition_by_blockify(ctx,
								    is_owned,
								    TestConfig::num_owned_per_piece);
    runtime->attach_name(ip_owned, "ip_owned");
    lp_owned = runtime->get_logical_partition(ctx, lr_owned, ip_owned);
    runtime->attach_name(lp_owned, "lp_owned");
  }

  // create and partition ghost region
  IndexSpace is_ghost;
  FieldSpace fs_ghost;
  LogicalRegion lr_ghost;
  LogicalPartition lp_ghost;
  {
    Rect<1> domain(0,
		   TestConfig::num_pieces * TestConfig::num_ghost_per_piece - 1);
    is_ghost = runtime->create_index_space(ctx, domain);
    runtime->attach_name(is_ghost, "is_ghost");

    std::vector<size_t> sizes;
    std::vector<FieldID> ids;
    ids.push_back(FID_GHOST_PTR);
    sizes.push_back(sizeof(coord_t));
    ids.push_back(FID_GHOST_VAL);
    sizes.push_back(sizeof(int));
    fs_ghost = runtime->create_field_space(ctx, sizes, ids);
    runtime->attach_name(fs_ghost, "fs_ghost");
    runtime->attach_name(fs_ghost, FID_GHOST_PTR, "ghost_ptr");
    runtime->attach_name(fs_ghost, FID_GHOST_VAL, "ghost_val");

    lr_ghost = runtime->create_logical_region(ctx, is_ghost, fs_ghost);
    runtime->attach_name(lr_ghost, "lr_ghost");

    IndexPartition ip_ghost = runtime->create_partition_by_blockify(ctx,
								    is_ghost,
								    TestConfig::num_ghost_per_piece);
    runtime->attach_name(ip_ghost, "ip_ghost");
    lp_ghost = runtime->get_logical_partition(ctx, lr_ghost, ip_ghost);
    runtime->attach_name(lp_ghost, "lp_ghost");
  }

  InitGhostPointersTask::init_pointers(runtime, ctx,
				       is_pieces, lr_affinity,
				       lr_ghost, lp_ghost, lp_owned,
				       TestConfig::random_seed);

  // compute some dependent partitions that can hopefully be used to make
  //  gather copies more efficient

  Future f_dpstart = runtime->get_current_time(ctx,
					       runtime->issue_execution_fence(ctx));

  // we have a bunch of different gather modes, and they rely on different
  //  dependent partitions - since we're timing this and all, only compute
  //  the ones we actually need
  // names match the slides, '+' means a depedent partition is used directly
  //  in the gather, while '-' means it is a necessary intermediate
  //
  // gather mode       preimage  pmg  image  img  bloated  blt  is_interference
  //    0
  //    1
  //    2                               +
  //    3                               -     -      +                -
  //    4                  -      +
  //    5
  //    6                  -      +     -     -                       +
  //    7                               -     -                       +
  LogicalPartition lp_owned_image = LogicalPartition::NO_PART;

  if(((1 << TestConfig::gather_mode) & 0x00cc) != 0) {  // 2, 3, 6, 7
    // compute the image of lp_ghost[i] in lr_owned
    IndexPartition ip = runtime->create_partition_by_image(ctx,
							   is_owned,
							   lp_ghost,
							   lr_ghost,
							   FID_GHOST_PTR,
							   is_pieces,
							   // probably aliased and probably
							   //  incomplete, but Legion doesn't
							   //  let you say that
							   //LEGION_ALIASED_INCOMPLETE_KIND);
							   LEGION_COMPUTE_KIND);
    runtime->attach_name(ip, "ip_image");
    lp_owned_image = runtime->get_logical_partition(ctx, lr_owned, ip);
    runtime->attach_name(lp_owned_image, "lp_image");
  }

  if(((1 << TestConfig::gather_mode) & 0x00c8) != 0) {  // 3, 6, 7
    // now the pairwise intersection of lp_owned[j] with lp_owned_image[i],
    //  as a subspace of lp_owned[j]
    std::map<IndexSpace,IndexPartition> dummy; // don't want names now
    runtime->create_cross_product_partitions(ctx,
					     lp_owned.get_index_partition(),
					     lp_owned_image.get_index_partition(),
					     dummy,
					     // probably aliased and probably
					     //  incomplete, but Legion doesn't
					     //  let you say that
					     //LEGION_ALIASED_INCOMPLETE_KIND,
					     LEGION_COMPUTE_KIND,
					     PID_OWNED_IMAGE_PER_RANK);
  }

  IndexSpace is_interference = IndexSpace::NO_SPACE;

  if(((1 << TestConfig::gather_mode) & 0x00c8) != 0) {  // 3, 6, 7
    // TODO: once implemented, use actual interference test
    // a sparse 2d space lights up the (src,dst) pairs that move data in
    //  the gather (i.e. the intersection of lp_owned_image with lp_owned)

    std::vector<Point<2,coord_t> > points;

    // iterate over destinations first
    for(int j = 0; j < TestConfig::num_pieces; j++) {
      IndexSpace is_dst = runtime->get_index_subspace(ctx,
						      lp_owned.get_index_partition(),
						      j);
      IndexPartition ip_per_rank = runtime->get_index_partition(ctx,
								is_dst,
								PID_OWNED_IMAGE_PER_RANK);

      // now iterate over each source and test for non-emptiness
      for(int i = 0; i < TestConfig::num_pieces; i++) {
	IndexSpace is_src_in_dst = runtime->get_index_subspace(ctx,
							       ip_per_rank,
							       i);
	// construct an interator and see if it has any points at all
	Domain d = runtime->get_index_space_domain(ctx,
						   is_src_in_dst);
	Domain::DomainPointIterator dpi(d);
	if(!!dpi) {
	  log_app.debug() << "interference: " << i << " -> " << j;
	  points.push_back(Point<2,coord_t>(i,j));
	}
      }
    }

    is_interference = runtime->create_index_space(ctx, points);
    runtime->attach_name(is_interference, "is_interference");
  }

  LogicalPartition lp_owned_image_bloated = LogicalPartition::NO_PART;

  if(((1 << TestConfig::gather_mode) & 0x0008) != 0) {  // 3
    // use the interference matrix to define an over-approximation of the
    //  image that includes every index in a destination rank if any index
    //  is included in the image
    MultiDomainColoring mdc;
    
    for(Domain::DomainPointIterator dpi(runtime->get_index_space_domain(ctx,
									is_interference));
	dpi;
	dpi.step()) {
      int src = (*dpi)[0];
      int dst = (*dpi)[1];
      IndexSpace is_dst = runtime->get_index_subspace(ctx,
						      lp_owned.get_index_partition(),
						      dst);
      mdc[src].insert(runtime->get_index_space_domain(ctx, is_dst));
    }

    IndexPartition ip = runtime->create_index_partition(ctx,
							is_owned,
							runtime->get_index_space_domain(ctx, is_pieces),
							mdc,
							false /*!disjoint*/);
    runtime->attach_name(ip, "ip_bloated");
    lp_owned_image_bloated = runtime->get_logical_partition(ctx, lr_owned, ip);
    runtime->attach_name(lp_owned_image_bloated, "lp_bloated");
  }
    
  if(((1 << TestConfig::gather_mode) & 0x0000) != 0) {  // none yet!
    // now the pairwise intersection of lp_owned[j] with lp_owned_image_bloated[i],
    //  as a subspace of lp_owned[j]
    std::map<IndexSpace,IndexPartition> dummy; // don't want names now
    runtime->create_cross_product_partitions(ctx,
					     lp_owned.get_index_partition(),
					     lp_owned_image_bloated.get_index_partition(),
					     dummy,
					     LEGION_ALIASED_COMPLETE_KIND,
					     PID_OWNED_IMAGE_BLOATED_PER_RANK);
  }
  
  LogicalPartition lp_ghost_preimage = LogicalPartition::NO_PART;

  if(((1 << TestConfig::gather_mode) & 0x0050) != 0) {  // 4, 6
    // compute the preimage of lp_owned[j] in lr_ghost
    //  we know it'll be disjoint but not complete
    IndexPartition ip = runtime->create_partition_by_preimage(ctx,
							      lp_owned.get_index_partition(),
							      lr_ghost,
							      lr_ghost,
							      FID_GHOST_PTR,
							      is_pieces,
							      LEGION_DISJOINT_COMPLETE_KIND);
    runtime->attach_name(ip, "ip_preimage");
    lp_ghost_preimage = runtime->get_logical_partition(ctx, lr_ghost, ip);
    runtime->attach_name(lp_ghost_preimage, "lp_preimage");
  }

  if(((1 << TestConfig::gather_mode) & 0x0050) != 0) {  // 4, 6
    // now the pairwise intersection of lp_ghost[i] with lp_ghost_preimage[j],
    //  as a subspace of lp_ghost[i]
    std::map<IndexSpace,IndexPartition> dummy; // don't want names now
    runtime->create_cross_product_partitions(ctx,
					     lp_ghost.get_index_partition(),
					     lp_ghost_preimage.get_index_partition(),
					     dummy,
					     LEGION_DISJOINT_COMPLETE_KIND,
					     PID_GHOST_PREIMAGE_PER_RANK);
  }

  Future f_dpend = runtime->get_current_time(ctx,
					     runtime->issue_execution_fence(ctx));

  // now that we've issued all the necessary deppart ops, time how long they
  //  took
  double t_dpstart = f_dpstart.get_result<double>(true /*silence_warnings*/);
  double t_dpend = f_dpend.get_result<double>(true /*silence_warnings*/);

  log_app.print() << "partitioning: start=" << t_dpstart << " end=" << t_dpend << " elapsed=" << (t_dpend - t_dpstart);

  for(int i = 0; i < TestConfig::num_iterations; i++) {
    if(TestConfig::use_tracing)
      runtime->begin_trace(ctx, TRACE_ID_COPY);

    InitOwnedTask::init_owned(runtime, ctx, is_pieces,
			      lr_owned, lp_owned, 5 + i);

    Future f_cpstart = runtime->get_current_time(ctx,
						 runtime->issue_execution_fence(ctx));

    do_gather_copy(runtime, ctx, TestConfig::gather_mode,
		   is_pieces, is_affinity, is_interference,
		   lr_owned, lp_owned, lp_owned_image, lp_owned_image_bloated,
		   lr_ghost, lp_ghost);

    Future f_cpend = runtime->get_current_time(ctx,
					       runtime->issue_execution_fence(ctx));

    int errors = CheckGhostDataTask::check_data(runtime, ctx, is_pieces,
						lr_ghost, lp_ghost, 5 + i);
    if(errors > 0) {
      log_app.error() << "copy iter " << i << ": " << errors << " errors detected!";
      runtime->set_return_code(1);
      break;
    }

    if(TestConfig::use_tracing)
      runtime->end_trace(ctx, TRACE_ID_COPY);

    double t_cpstart = f_cpstart.get_result<double>(true /*silence_warnings*/);
    double t_cpend = f_cpend.get_result<double>(true /*silence_warnings*/);

    // compute the aggregate bandwidth of the gather
    double agg_bw = (TestConfig::num_pieces *
		     TestConfig::num_ghost_per_piece *
		     sizeof(int) * 1e-9 /
		     (t_cpend - t_cpstart));
    log_app.print() << "copy iter " << i << ": start=" << t_cpstart << " end=" << t_cpend << " elapsed=" << (t_cpend - t_cpstart) << " agg_bw=" << agg_bw << " GB/s";
  }

  runtime->destroy_logical_region(ctx, lr_affinity);
  runtime->destroy_index_space(ctx, is_affinity);
  runtime->destroy_field_space(ctx, fs_affinity);
}

class GatherMapper : public Mapping::NullMapper {
public:
  GatherMapper(Mapping::MapperRuntime *_rt, Machine _machine, bool _replicate)
    : NullMapper(_rt, _machine)
    , replicate(_replicate)
  {
    Machine::MemoryQuery mq(machine);
    mq.only_kind(Memory::SYSTEM_MEM).has_capacity(1);
    for(Machine::MemoryQuery::iterator it = mq.begin();
	it != mq.end();
	++it) {
      //log_app.print() << "memory: " << *it;
      memories.push_back(*it);

      // get a processor with affinity to that memory
      Processor p = Machine::ProcessorQuery(machine).only_kind(Processor::LOC_PROC).has_affinity_to(*it).first();
      assert(p.exists());
      //log_app.print() << " proc: " << p;
      procs.push_back(p);
    }
  }

  const char *get_mapper_name(void) const
  {
    return "gathermapper";
  }

  MapperSyncModel get_mapper_sync_model(void) const
  {
    return CONCURRENT_MAPPER_MODEL;
  }

  bool request_valid_instances(void) const
  {
    return false;
  }

  void select_steal_targets(const Mapping::MapperContext ctx,
			    const SelectStealingInput& input,
			    SelectStealingOutput& output)
  {
    // no stealing
  }

  void select_task_options(const Mapping::MapperContext ctx,
			   const Task& task, TaskOptions& output)
  {
    // top level task should be replicated, if requested
    if(replicate && (task.task_id == TOP_LEVEL_TASK_ID))
      output.replicate = true;

    // we're going to do all mapping from node 0
    output.map_locally = true;
  }

  void select_tasks_to_map(const Mapping::MapperContext ctx,
			   const SelectMappingInput& input,
			   SelectMappingOutput& output)
  {
    // map 'em all
    output.map_tasks.insert(input.ready_tasks.begin(),
			    input.ready_tasks.end());
  }

  void map_task(const Mapping::MapperContext ctx,
		const Task& task,
		const MapTaskInput& input, MapTaskOutput& output)
  {
    //log_app.print() << "map task id=" << task.task_id;

    Processor p = Processor::NO_PROC;

    switch(task.task_id) {
    case TOP_LEVEL_TASK_ID:
      {
	p = procs[0];
	break;
      }

    case INIT_AFFINITY_TASK_ID:
    case INIT_OWNED_TASK_ID:
    case INIT_GHOST_TASK_ID:
    case CHECK_GHOST_TASK_ID:
      {
	int index_point, num_points;
	// WAR: in master branch, a point task launch does not have a valid
	//  task.index_domain?
	if(task.is_index_space) {
	  index_point = task.index_point[0];
	  num_points = task.index_domain.get_volume();
	} else {
	  index_point = 0;
	  num_points = 1;
	}
	//log_app.print() << "map task: id=" << task.task_id << " pt=" << index_point << " num=" << num_points;
	int points_per_mem = 1 + (num_points - 1) / memories.size();
	int mem_idx = index_point / points_per_mem;
	p = procs[mem_idx];

	for(size_t i = 0; i < task.regions.size(); i++) {
	  Mapping::PhysicalInstance inst;
	  inst = choose_instance(ctx, index_point, num_points,
				 task.regions[i]);
	  output.chosen_instances[i].push_back(inst);
	}
	break;
      }

    default:
      assert(0);
    }

    output.target_procs.push_back(p);

    std::vector<VariantID> valid_variants;
    runtime->find_valid_variants(ctx, task.task_id, valid_variants, p.kind());
    assert(!valid_variants.empty());
    output.chosen_variant = valid_variants[0];
  }

  void map_replicate_task(const Mapping::MapperContext ctx,
			  const Task& task, const MapTaskInput& input,
			  const MapTaskOutput& default_output,
			  MapReplicateTaskOutput& output)
  {
    // only the top-level task should end up here
    assert(task.task_id == TOP_LEVEL_TASK_ID);

    // TODO: maybe need to keep a separate 'control_procs' list?
    output.task_mappings.resize(procs.size(), default_output);
    output.control_replication_map = procs;

    std::vector<VariantID> valid_variants;
    runtime->find_valid_variants(ctx, task.task_id, valid_variants, procs[0].kind());
    assert(!valid_variants.empty());

    for(size_t i = 0; i < procs.size(); i++) {
      output.task_mappings[i].target_procs.push_back(procs[i]);
      output.task_mappings[i].chosen_variant = valid_variants[0];
    }
  }

  void configure_context(const Mapping::MapperContext ctx,
			 const Task& task, ContextConfigOutput& output)
  {
    // defaults are fine
  }

  void map_inline(const Mapping::MapperContext ctx,
		  const InlineMapping& inline_op,
		  const MapInlineInput& input, MapInlineOutput& output)
  {
    Mapping::PhysicalInstance inst;
    inst = choose_instance(ctx, 0, 1, inline_op.requirement);
    output.chosen_instances.push_back(inst);
  }

  void slice_task(const Mapping::MapperContext ctx,
		  const Task& task, const SliceTaskInput& input,
		  SliceTaskOutput& output)
  {
    // even though we're going to map everything from the first processor,
    //  we need to slice apart all the points so that they can be mapped
    //  to different places
    for(Domain::DomainPointIterator dpi(input.domain); dpi; dpi.step())
      output.slices.push_back(TaskSlice(Domain(dpi.p, dpi.p), procs[0],
					false /*!recurse*/,
					false /*!stealable*/));
  }

  void select_partition_projection(const Mapping::MapperContext  ctx,
				   const Partition& partition,
				   const SelectPartitionProjectionInput& input,
				   SelectPartitionProjectionOutput& output)
  {
    // expect to always have an open complete partition to use
    assert(!input.open_complete_partitions.empty());
    output.chosen_partition = input.open_complete_partitions[0];
  }

  void map_partition(const Mapping::MapperContext ctx,
		     const Partition& partition,
		     const MapPartitionInput& input,
		     MapPartitionOutput& output)
  {
    // we didn't ask for valid instances
    assert(input.valid_instances.empty());

    int index_point = partition.index_point[0];
    int num_points = partition.index_domain.get_volume();

    Mapping::PhysicalInstance inst;
    inst = choose_instance(ctx, index_point, num_points,
			   partition.requirement);
    output.chosen_instances.push_back(inst);
  }

  void map_copy(const Mapping::MapperContext ctx,
		const Copy& copy, const MapCopyInput& input,
		MapCopyOutput& output)
  {
    int src_index, dst_index, num_points;
    if(copy.index_domain.get_dim() == 1) {
      src_index = copy.index_point[0];
      dst_index = copy.index_point[0];
      num_points = copy.index_domain.get_volume();
    } else {
      // the 2D index copies we launch are indexed by:
      //             (gathering_piece, data_source_piece)
      src_index = copy.index_point[1];
      dst_index = copy.index_point[0];
      Rect<2,coord_t> bounds = copy.index_domain.bounds<2,coord_t>();
      num_points = bounds.hi[0] - bounds.lo[0] + 1;
    }

    for(size_t i = 0; i < copy.src_requirements.size(); i++) {
      Mapping::PhysicalInstance inst;
      inst = choose_instance(ctx, src_index, num_points,
			     copy.src_requirements[i]);
      output.src_instances[i].push_back(inst);
    }

    for(size_t i = 0; i < copy.dst_requirements.size(); i++) {
      Mapping::PhysicalInstance inst;
      inst = choose_instance(ctx, dst_index, num_points,
			     copy.dst_requirements[i]);
      output.dst_instances[i].push_back(inst);
    }

    if(!copy.src_indirect_requirements.empty()) {
      assert(copy.src_indirect_requirements.size() == 1);

      Mapping::PhysicalInstance inst;
      inst = choose_instance(ctx, dst_index, num_points,
			     copy.src_indirect_requirements[0]);
      output.src_indirect_instances[0] = inst;
    }
  }

  void select_task_sources(const Mapping::MapperContext ctx,
			   const Task& task,
			   const SelectTaskSrcInput& input,
			   SelectTaskSrcOutput& output)
  {
    // let the runtime decide (this just occurs when we broadcast the
    //  affinity matrix at startup)
  }

  void select_copy_sources(const Mapping::MapperContext ctx,
			   const Copy& copy,
			   const SelectCopySrcInput& input,
			   SelectCopySrcOutput& output)
  {
    // let the runtime decide (this is just used for constructing large
    //  images for the O(1) and O(N) copies)
  }

  void select_sharding_functor(const Mapping::MapperContext ctx,
			       const Task& task,
			       const SelectShardingFunctorInput& input,
			       SelectShardingFunctorOutput& output)
  {
    // same sharding function for everything
    output.chosen_functor = SHARD_ID_COARSE;
  }

  void select_sharding_functor(const Mapping::MapperContext ctx,
			       const Copy& copy,
			       const SelectShardingFunctorInput& input,
			       SelectShardingFunctorOutput& output)
  {
    // same sharding function for everything
    output.chosen_functor = SHARD_ID_COARSE;
  }

  void select_sharding_functor(const Mapping::MapperContext ctx,
			       const Fill& fill,
			       const SelectShardingFunctorInput& input,
			       SelectShardingFunctorOutput& output)
  {
    // same sharding function for everything
    output.chosen_functor = SHARD_ID_COARSE;
  }

  void select_sharding_functor(const Mapping::MapperContext ctx,
			       const Partition& partition,
			       const SelectShardingFunctorInput& input,
			       SelectShardingFunctorOutput& output)
  {
    // same sharding function for everything
    output.chosen_functor = SHARD_ID_COARSE;
  }

  void memoize_operation(const Mapping::MapperContext ctx,
			 const Mappable& mappable, const MemoizeInput& input,
			 MemoizeOutput& output)
  {
    // memoize all the things
    output.memoize = true;
  }

protected:
  Mapping::PhysicalInstance choose_instance(const Mapping::MapperContext ctx,
					    int piece_index, int num_pieces,
					    const RegionRequirement& req,
					    bool all_fields = true)
  {
    int pieces_per_mem = 1 + (num_pieces - 1) / memories.size();
    int mem_idx = piece_index / pieces_per_mem;
    Memory m = memories[mem_idx];

    LayoutConstraintSet constraints;
    if(all_fields) {
      FieldConstraint fc(false /*!contiguous*/);
      runtime->get_field_space_fields(ctx, req.region.get_field_space(),
				      fc.field_set);
      constraints.add_constraint(fc);
    } else {
      constraints.add_constraint(FieldConstraint(req.privilege_fields,
						 false /*!contiguous*/));
    }
    std::vector<LogicalRegion> regions(1, req.region);
    Mapping::PhysicalInstance result;
    bool created;
    bool ok = runtime->find_or_create_physical_instance(ctx,
							m,
							constraints,
							regions,
							result,
							created);
    assert(ok);
    return result;
  }

  // track a bunch of things for each piece
  std::vector<Memory> memories;
  std::vector<Processor> procs;
  bool replicate;
};

void mapper_registration(Machine machine, Runtime *rt,
                          const std::set<Processor> &local_procs)
{
  for(std::set<Processor>::const_iterator it = local_procs.begin();
      it != local_procs.end();
      ++it)
    rt->replace_default_mapper(new GatherMapper(rt->get_mapper_runtime(),
						machine,
						TestConfig::replicate), *it);
}

int main(int argc, char **argv)
{
  Runtime::initialize(&argc, &argv);

  {
    Realm::CommandLineParser clp;
    clp.add_option_int("-p", TestConfig::num_pieces);
    clp.add_option_int("-o", TestConfig::num_owned_per_piece);
    clp.add_option_int("-g", TestConfig::num_ghost_per_piece);
    clp.add_option_int("-m", TestConfig::gather_mode);
    clp.add_option_int("-s", TestConfig::random_seed);
    clp.add_option_int("-r", TestConfig::replicate);
    clp.add_option_int("-i", TestConfig::num_iterations);
    clp.add_option_int("-t", TestConfig::use_tracing);
    clp.add_option_string("-a", TestConfig::affinity_pattern, 256);

    bool ok = clp.parse_command_line(argc,
				     const_cast<const char **>(argv));
    assert(ok);
  }

  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable(true);
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  InitAffinityTask::register_tasks();
  InitOwnedTask::register_tasks();
  InitGhostPointersTask::register_tasks();
  CheckGhostDataTask::register_tasks();

  Runtime::preregister_projection_functor(PFID_IJ_TO_I,
					  new ProjectionOneLevel(0));
  Runtime::preregister_projection_functor(PFID_IJ_TO_J,
					  new ProjectionOneLevel(1));
  Runtime::preregister_projection_functor(PFID_IJ_TO_I_PREIMAGE_BY_RANK_J,
					  new ProjectionTwoLevel(0,
								 PID_GHOST_PREIMAGE_PER_RANK,
								 1));

  Runtime::preregister_sharding_functor(SHARD_ID_COARSE,
					new CoarseShardingFunctor);

  Runtime::add_registration_callback(mapper_registration);

  return Runtime::start(argc, argv);
}
