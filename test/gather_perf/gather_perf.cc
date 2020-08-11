/* Copyright 2020 NVIDIA Corporation
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
#include "realm/cmdline.h"
#include "philox.h"

using namespace Legion;

Logger log_app("app");

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
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
  virtual unsigned get_depth(void) const { return 0; }
  using ProjectionFunctor::project;
  virtual LogicalRegion project(LogicalPartition upper_bound,
				const DomainPoint &point,
				const Domain &launch_domain)
  {
    return runtime->get_logical_subregion_by_color(upper_bound,
						   point[index1]);
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

void initialize_affinity_matrix(Runtime *runtime, Context ctx,
				LogicalRegion lr_affinity,
				int num_pieces)
{
  // affinity matrix starts out as all zeros
  runtime->fill_field<int>(ctx, lr_affinity, lr_affinity,
			   FID_AFFINITY, 0);

  // inline mapping
  InlineLauncher il(RegionRequirement(lr_affinity, READ_WRITE, EXCLUSIVE,
				      lr_affinity)
		    .add_field(FID_AFFINITY));
  PhysicalRegion pr = runtime->map_region(ctx, il);
  pr.wait_until_valid(true /*silence*/);

  FieldAccessor<READ_WRITE, int, 2, coord_t, Realm::AffineAccessor<int,2,coord_t> > acc(pr, FID_AFFINITY);

  // simple 1-D ring affinity for now
  for(int i = 0; i < num_pieces; i++) {
    acc[i][(i + 1) % num_pieces] = 1;
    acc[i][(i + num_pieces - 1) % num_pieces] = 1;
  }

  runtime->unmap_region(ctx, pr);
}

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

    default: assert(0);
  }
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_pieces = 4;
  int num_owned_per_piece = 8;
  int num_ghost_per_piece = 16;
  int gather_mode = 1;
  int random_seed = 12345;

  {
    const InputArgs &command_args = Runtime::get_input_args();

    Realm::CommandLineParser clp;
    clp.add_option_int("-p", num_pieces);
    clp.add_option_int("-o", num_owned_per_piece);
    clp.add_option_int("-g", num_ghost_per_piece);
    clp.add_option_int("-m", gather_mode);
    clp.add_option_int("-s", random_seed);

    bool ok = clp.parse_command_line(command_args.argc,
				     const_cast<const char **>(command_args.argv));
    assert(ok);
  }

  // construct the affinity matrix
  IndexSpace is_affinity;
  FieldSpace fs_affinity;
  LogicalRegion lr_affinity;
  {
    Rect<2> domain(Point<2>(0, 0), Point<2>(num_pieces - 1, num_pieces - 1));
    is_affinity = runtime->create_index_space(ctx, domain);

    std::vector<size_t> sizes(1, sizeof(int));
    std::vector<FieldID> ids(1, FID_AFFINITY);
    fs_affinity = runtime->create_field_space(ctx, sizes, ids);

    lr_affinity = runtime->create_logical_region(ctx,
						 is_affinity, fs_affinity);
  }

  // initialize affinity matrix inline
  initialize_affinity_matrix(runtime, ctx, lr_affinity, num_pieces);

  // create the index space we'll to name pieces
  IndexSpace is_pieces;
  {
    Rect<1> domain(0, num_pieces - 1);
    is_pieces = runtime->create_index_space(ctx, domain);
  }

  // create and partition owned region
  IndexSpace is_owned;
  FieldSpace fs_owned;
  LogicalRegion lr_owned;
  LogicalPartition lp_owned;
  {
    Rect<1> domain(0, num_pieces * num_owned_per_piece - 1);
    is_owned = runtime->create_index_space(ctx, domain);

    std::vector<size_t> sizes(1, sizeof(int));
    std::vector<FieldID> ids(1, FID_OWNED_VAL);
    fs_owned = runtime->create_field_space(ctx, sizes, ids);

    lr_owned = runtime->create_logical_region(ctx, is_owned, fs_owned);

    IndexPartition ip_owned = runtime->create_partition_by_blockify(ctx,
								    is_owned,
								    num_owned_per_piece);
    lp_owned = runtime->get_logical_partition(ctx, lr_owned, ip_owned);
  }

  // create and partition ghost region
  IndexSpace is_ghost;
  FieldSpace fs_ghost;
  LogicalRegion lr_ghost;
  LogicalPartition lp_ghost;
  {
    Rect<1> domain(0, num_pieces * num_ghost_per_piece - 1);
    is_ghost = runtime->create_index_space(ctx, domain);

    std::vector<size_t> sizes;
    std::vector<FieldID> ids;
    ids.push_back(FID_GHOST_PTR);
    sizes.push_back(sizeof(coord_t));
    ids.push_back(FID_GHOST_VAL);
    sizes.push_back(sizeof(int));
    fs_ghost = runtime->create_field_space(ctx, sizes, ids);

    lr_ghost = runtime->create_logical_region(ctx, is_ghost, fs_ghost);

    IndexPartition ip_ghost = runtime->create_partition_by_blockify(ctx,
								    is_ghost,
								    num_ghost_per_piece);
    lp_ghost = runtime->get_logical_partition(ctx, lr_ghost, ip_ghost);
  }

  InitGhostPointersTask::init_pointers(runtime, ctx,
				       is_pieces, lr_affinity,
				       lr_ghost, lp_ghost, lp_owned,
				       random_seed);

  // compute some dependent partitions that can hopefully be used to make
  //  gather copies more efficient
  LogicalPartition lp_owned_image;
  {
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
    lp_owned_image = runtime->get_logical_partition(ctx, lr_owned, ip);
  }

  // now the pairwise intersection of lp_owned[j] with lp_owned_image[i],
  //  as a subspace of lp_owned[j]
  {
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

  IndexSpace is_interference;
  {
    // TODO: once implemented, use actual interference test
    // a sparse 2d space lights up the (src,dst) pairs that move data in
    //  the gather (i.e. the intersection of lp_owned_image with lp_owned)

    std::vector<Point<2,coord_t> > points;

    // iterate over destinations first
    for(int j = 0; j < num_pieces; j++) {
      IndexSpace is_dst = runtime->get_index_subspace(ctx,
						      lp_owned.get_index_partition(),
						      j);
      IndexPartition ip_per_rank = runtime->get_index_partition(ctx,
								is_dst,
								PID_OWNED_IMAGE_PER_RANK);

      // now iterate over each source and test for non-emptiness
      for(int i = 0; i < num_pieces; i++) {
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
  }

  LogicalPartition lp_owned_image_bloated;
  {
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
    lp_owned_image_bloated = runtime->get_logical_partition(ctx, lr_owned, ip);
  }
    
  // now the pairwise intersection of lp_owned[j] with lp_owned_image_bloated[i],
  //  as a subspace of lp_owned[j]
  {
    std::map<IndexSpace,IndexPartition> dummy; // don't want names now
    runtime->create_cross_product_partitions(ctx,
					     lp_owned.get_index_partition(),
					     lp_owned_image_bloated.get_index_partition(),
					     dummy,
					     LEGION_ALIASED_COMPLETE_KIND,
					     PID_OWNED_IMAGE_BLOATED_PER_RANK);
  }
  
  LogicalPartition lp_ghost_preimage;
  {
    // compute the preimage of lp_owned[j] in lr_ghost
    //  we know it'll be disjoint but not complete
    IndexPartition ip = runtime->create_partition_by_preimage(ctx,
							      lp_owned.get_index_partition(),
							      lr_ghost,
							      lr_ghost,
							      FID_GHOST_PTR,
							      is_pieces,
							      LEGION_DISJOINT_COMPLETE_KIND);
    lp_ghost_preimage = runtime->get_logical_partition(ctx, lr_ghost, ip);
  }

  // now the pairwise intersection of lp_ghost[i] with lp_ghost_preimage[j],
  //  as a subspace of lp_ghost[i]
  {
    std::map<IndexSpace,IndexPartition> dummy; // don't want names now
    runtime->create_cross_product_partitions(ctx,
					     lp_ghost.get_index_partition(),
					     lp_ghost_preimage.get_index_partition(),
					     dummy,
					     LEGION_DISJOINT_COMPLETE_KIND,
					     PID_GHOST_PREIMAGE_PER_RANK);
  }

  InitOwnedTask::init_owned(runtime, ctx, is_pieces,
			    lr_owned, lp_owned, 5);

  do_gather_copy(runtime, ctx, gather_mode,
		 is_pieces, is_affinity, is_interference,
		 lr_owned, lp_owned, lp_owned_image, lp_owned_image_bloated,
		 lr_ghost, lp_ghost);

  int errors = CheckGhostDataTask::check_data(runtime, ctx, is_pieces,
				 lr_ghost, lp_ghost, 5);
  if(errors > 0) {
    log_app.error() << errors << " errors detected!";
    runtime->set_return_code(1);
  }

  runtime->destroy_logical_region(ctx, lr_affinity);
  runtime->destroy_index_space(ctx, is_affinity);
  runtime->destroy_field_space(ctx, fs_affinity);
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

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

  return Runtime::start(argc, argv);
}
