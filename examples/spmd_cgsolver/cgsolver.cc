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
#include <queue>
#include "legion.h"

#include "cgsolver.h"
#include "cgtasks.h"
#include "cgmapper.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

LegionRuntime::Logger::Category log_app("app");

#define LOG_ASSERT(cond, logger, ...) do {	\
  if(!(cond)) {					\
    (logger).fatal(__VA_ARGS__);		\
    assert(false);				\
  }						\
  } while(0)

enum {
  TOP_LEVEL_TASK_ID,
  SPMD_INIT_TASK_ID,
  SPMD_MAIN_TASK_ID,
  INIT_FIELD_TASK_ID,
  SPMV_FIELD_TASK_ID,
  DOTP_FIELD_TASK_ID,
  ADD_FIELD_TASK_ID,
};

enum {
  TRACE_ID_CG_ITER = 1,
};

enum {
  DIR_X = 0,
  DIR_Y = 1,
  DIR_Z = 2
};

enum {
  SIDE_MINUS = 0,
  SIDE_PLUS = 1,
};

template <unsigned DIM>
static int volume(const Point<DIM>& p)
{
  int v = 1;
  for(int i = 0; i < DIM; i++)
    v *= p.x[i];
  return v;
}

template <unsigned DIM>
inline Point<DIM> LEFT(const Point<DIM>& p)
{
  Point<DIM> p2(p);
  p2.x[0]--;
  return p2;
}

template <unsigned DIM>
inline Point<DIM> RIGHT(const Point<DIM>& p)
{
  Point<DIM> p2(p);
  p2.x[0]++;
  return p2;
}

template <unsigned DIM>
inline Point<DIM> UP(const Point<DIM>& p)
{
  Point<DIM> p2(p);
  p2.x[1]--;
  return p2;
}

template <unsigned DIM>
inline Point<DIM> DOWN(const Point<DIM>& p)
{
  Point<DIM> p2(p);
  p2.x[1]++;
  return p2;
}

template <unsigned DIM>
inline Point<DIM> FRONT(const Point<DIM>& p)
{
  Point<DIM> p2(p);
  p2.x[2]--;
  return p2;
}

template <unsigned DIM>
inline Point<DIM> BACK(const Point<DIM>& p)
{
  Point<DIM> p2(p);
  p2.x[2]++;
  return p2;
}

// a "reduction" op that "merges" together PhaseBarrier's created by each shard
struct BarrierCombineReductionOp {
  static const ReductionOpID redop_id = 77;

  typedef PhaseBarrier LHS;
  typedef PhaseBarrier RHS;
  static const PhaseBarrier identity;

  typedef RegionAccessor<AccessorType::ReductionFold<BarrierCombineReductionOp>, RHS> AccType;

  static void combine(LHS& lhs, RHS rhs)
  {
    // no exists() method for PhaseBarrier...
    if(!(rhs == PhaseBarrier())) {
      assert(lhs == PhaseBarrier());
      lhs = rhs;
    }
  }

  template <bool EXCL>
  static void apply(LHS& lhs, RHS rhs) { combine(lhs, rhs); }

  template <bool EXCL>
  static void fold(RHS& rhs1, RHS rhs2) { combine(rhs1, rhs2); }
};

/*static*/ const PhaseBarrier BarrierCombineReductionOp::identity;

struct DoubleAddReductionOp {
  static const ReductionOpID redop_id = 78;

  typedef double LHS;
  typedef double RHS;
  static const double identity;

  template <bool EXCL>
  static void apply(LHS& lhs, RHS rhs) { lhs += rhs; }

  template <bool EXCL>
  static void fold(RHS& rhs1, RHS rhs2) { rhs1 += rhs2; }
};

/*static*/ const double DoubleAddReductionOp::identity = 0.0;

template <typename T1, typename T2>
struct FutureLessThan {
  static TaskID taskid;

  static void preregister_task(TaskID new_taskid = AUTO_GENERATE_ID);

  static Future compute(Runtime *runtime, Context ctx, Future lhs, Future rhs);

  static bool cpu_task(const Task *task,
		       const std::vector<PhysicalRegion> &regions,
		       Context ctx, HighLevelRuntime *runtime);
};

template <typename T1, typename T2>
/*static*/ void FutureLessThan<T1,T2>::preregister_task(TaskID new_taskid /*= AUTO_GENERATE_ID*/)
{
  taskid = ((new_taskid == AUTO_GENERATE_ID) ?
  	      Runtime::generate_static_task_id() :
	      new_taskid);
  const char *name = "future_less_than";
  TaskVariantRegistrar tvr(taskid, name);
  tvr.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  tvr.set_leaf(true);
  Runtime::preregister_task_variant<bool, &FutureLessThan<T1,T2>::cpu_task>(tvr, name);
}

template <typename T1, typename T2>
/*static*/ Future FutureLessThan<T1,T2>::compute(Runtime *runtime, Context ctx, Future lhs, Future rhs)
{
  TaskLauncher launcher(taskid, TaskArgument());
  launcher.add_future(lhs);
  launcher.add_future(rhs);
  return runtime->execute_task(ctx, launcher);
}

template <typename T1, typename T2>
/*static*/ bool FutureLessThan<T1,T2>::cpu_task(const Task *task,
						const std::vector<PhysicalRegion> &regions,
						Context ctx, HighLevelRuntime *runtime)
{
  assert(task->futures.size() == 2);
  T1 lhs = Future(task->futures[0]).get_result<T1>();
  T2 rhs = Future(task->futures[1]).get_result<T2>();
  //std::cout << "FLT: " << lhs << " < " << rhs << "?\n";
  return lhs < rhs;
}

typedef FutureLessThan<double,double> FLT_double;

template <>
/*static*/ TaskID FutureLessThan<double,double>::taskid = 0;

template <typename TR, typename T1, typename T2>
struct FutureDivide {
  static TaskID taskid;

  static void preregister_task(TaskID new_taskid = AUTO_GENERATE_ID);

  static Future compute(Runtime *runtime, Context ctx, Future lhs, Future rhs);

  static TR cpu_task(const Task *task,
		     const std::vector<PhysicalRegion> &regions,
		     Context ctx, HighLevelRuntime *runtime);
};

template <typename TR, typename T1, typename T2>
/*static*/ void FutureDivide<TR,T1,T2>::preregister_task(TaskID new_taskid /*= AUTO_GENERATE_ID*/)
{
  taskid = ((new_taskid == AUTO_GENERATE_ID) ?
  	      Runtime::generate_static_task_id() :
	      new_taskid);
  const char *name = "future_divide";
  TaskVariantRegistrar tvr(taskid, name);
  tvr.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  tvr.set_leaf(true);
  Runtime::preregister_task_variant<TR, &FutureDivide<TR,T1,T2>::cpu_task>(tvr, name);
}

template <typename TR, typename T1, typename T2>
/*static*/ Future FutureDivide<TR,T1,T2>::compute(Runtime *runtime, Context ctx, Future lhs, Future rhs)
{
  TaskLauncher launcher(taskid, TaskArgument());
  launcher.add_future(lhs);
  launcher.add_future(rhs);
  return runtime->execute_task(ctx, launcher);
}

template <typename TR, typename T1, typename T2>
/*static*/ TR FutureDivide<TR,T1,T2>::cpu_task(const Task *task,
					       const std::vector<PhysicalRegion> &regions,
					       Context ctx, HighLevelRuntime *runtime)
{
  assert(task->futures.size() == 2);
  T1 lhs = Future(task->futures[0]).get_result<T1>();
  T2 rhs = Future(task->futures[1]).get_result<T2>();
  TR res = lhs / rhs;
  //std::cout << "FDV: " << lhs << " / " << rhs << " = " << res << "\n";
  return res;
}

typedef FutureDivide<double,double,double> FDV_double;

template <>
/*static*/ TaskID FutureDivide<double,double,double>::taskid = 0;

class FieldHelperBase {
protected:
  // this is shared by all FieldHelper<T>'s
  static FieldID next_static_id;
};

/*static*/ FieldID FieldHelperBase::next_static_id = 10000;


template <typename T>
class FieldHelper : protected FieldHelperBase {
public:
  static const FieldID ASSIGN_STATIC_ID = AUTO_GENERATE_ID - 1;

  FieldHelper(const char *_name, FieldID _fid = ASSIGN_STATIC_ID)
    : name(_name), fid(_fid)
  {
    if(fid == ASSIGN_STATIC_ID)
      fid = next_static_id++;
  }

  ~FieldHelper(void) {}

  operator FieldID(void) const
  {
    assert(fid != AUTO_GENERATE_ID);
    return fid;
  }

  void allocate(Runtime *runtime, Context ctx, FieldSpace fs)
  {
    FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
    fid = fa.allocate_field(sizeof(T), fid);
    runtime->attach_name(fs, fid, name);
  }

  template <typename AT>
  RegionAccessor<AT, T> accessor(const PhysicalRegion& pr)
  {
    assert(fid != AUTO_GENERATE_ID);
    return pr.get_field_accessor(fid).template typeify<T>().template convert<AT>();
  }

  template <typename AT>
  RegionAccessor<AT, T> fold_accessor(const PhysicalRegion& pr)
  {
    assert(fid != AUTO_GENERATE_ID);
    std::vector<FieldID> fields;
    pr.get_fields(fields);
    assert((fields.size() == 1) && (fields[0] == fid));
    return pr.get_accessor().template typeify<T>().template convert<AT>();
  }

protected:
  const char *name;
  FieldID fid;
};

FieldHelper<int> fid_owner_shard("owner_shard");
FieldHelper<int> fid_neighbor_count("neighbor_count");
FieldHelper<PhaseBarrier> fid_ready_barrier("ready_barrier");
FieldHelper<PhaseBarrier> fid_done_barrier("done_barrier");
FieldHelper<double> fid_sol_b("sol_b");
FieldHelper<double> fid_sol_x("sol_x");
FieldHelper<double> fid_sol_p("sol_p");
FieldHelper<double> fid_sol_r("sol_r");
FieldHelper<double> fid_sol_Ap("sol_Ap");

struct SpmdInitArgs {
  Point<3> grid_dim, block_dim, blocks;
  int shard;
};

struct SpmdMainArgs {
  Point<3> grid_dim, block_dim, blocks;
  DynamicCollective dc_reduction;
  int max_iters;
  int future_lag;
  int show_residuals;
  int use_tracing;
  bool verbose;
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  Point<3> grid_dim, block_dim;
  int max_iters = 0;
  int future_lag = 0;
  int show_residuals = 1;
  int use_tracing = 1;
  bool verbose = false;

  for(int i = 0; i < 3; i++) {
    grid_dim.x[i] = 1;
    block_dim.x[i] = 1;
  }
  grid_dim.x[0] = 8;
  block_dim.x[0] = 8;

  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    for(int i = 1; i < command_args.argc; i++) {
      if(!strcmp(command_args.argv[i], "-g")) {
	int g = atoi(command_args.argv[++i]);
	grid_dim.x[0] = grid_dim.x[1] = grid_dim.x[2] = g;
	continue;
      }
      if(!strcmp(command_args.argv[i], "-gx")) {
	int g = atoi(command_args.argv[++i]);
	grid_dim.x[0] = g;
	continue;
      }
      if(!strcmp(command_args.argv[i], "-gy")) {
	int g = atoi(command_args.argv[++i]);
	grid_dim.x[1] = g;
	continue;
      }
      if(!strcmp(command_args.argv[i], "-gz")) {
	int g = atoi(command_args.argv[++i]);
	grid_dim.x[2] = g;
	continue;
      }
      if(!strcmp(command_args.argv[i], "-b")) {
	int b = atoi(command_args.argv[++i]);
	block_dim.x[0] = block_dim.x[1] = block_dim.x[2] = b;
	continue;
      }
      if(!strcmp(command_args.argv[i], "-bx")) {
	int b = atoi(command_args.argv[++i]);
	block_dim.x[0] = b;
	continue;
      }
      if(!strcmp(command_args.argv[i], "-by")) {
	int b = atoi(command_args.argv[++i]);
	block_dim.x[1] = b;
	continue;
      }
      if(!strcmp(command_args.argv[i], "-bz")) {
	int b = atoi(command_args.argv[++i]);
	block_dim.x[2] = b;
	continue;
      }
      if(!strcmp(command_args.argv[i], "-m")) {
	max_iters = atoi(command_args.argv[++i]);
	continue;
      }
      if(!strcmp(command_args.argv[i], "-f")) {
	future_lag = atoi(command_args.argv[++i]);
	continue;
      }
      if(!strcmp(command_args.argv[i], "-r")) {
	show_residuals = atoi(command_args.argv[++i]);
	continue;
      }
      if(!strcmp(command_args.argv[i], "-t")) {
	use_tracing = atoi(command_args.argv[++i]);
	continue;
      }
      if(!strcmp(command_args.argv[i], "-v")) {
	verbose = true;
	continue;
      }
    }
  }

  // if the user hasn't set a max, a reasonable upper bound is the manhattan distance across the grid
  if(max_iters == 0)
    max_iters = grid_dim.x[0] + grid_dim.x[1] + grid_dim.x[2];

  // compute the number of blocks in each direction
  Point<3> blocks;
  for(int i = 0; i < 3; i++) {
    int g = grid_dim.x[i];
    int b = block_dim.x[i];
    LOG_ASSERT((b != 0), log_app, "block size cannot be zero (dim=%d)", i);
    LOG_ASSERT(((g % b) == 0), log_app, "block size must evenly divide grid size (dim=%d, grid=%d, block=%d)", i, g, b);
    blocks.x[i] = g / b;
  }
  log_app.print() << "grid pts = " << volume(grid_dim) << ", block pts = " << volume(block_dim) << ", block count = " << volume(blocks);

  // define the main grid index space and its partition into blocks
  IndexSpace is_grid = runtime->create_index_space(ctx,
						   Domain::from_rect<3>(Rect<3>(Point<3>::ZEROES(),
										grid_dim - Point<3>::ONES())));
  Blockify<3> grid2blk_map(block_dim);
  IndexPartition ip_grid = runtime->create_index_partition(ctx,
							   is_grid,
							   grid2blk_map,
							   0 /*color*/);

  // we also use an index space with an element per block to record things like which shard owns each block
  IndexSpace is_blks = runtime->create_index_space(ctx,
						   Domain::from_rect<3>(Rect<3>(Point<3>::ZEROES(),
										blocks - Point<3>::ONES())));
  FieldSpace fs_blks = runtime->create_field_space(ctx);
  fid_owner_shard.allocate(runtime, ctx, fs_blks);
  fid_neighbor_count.allocate(runtime, ctx, fs_blks);
  fid_ready_barrier.allocate(runtime, ctx, fs_blks);
  fid_done_barrier.allocate(runtime, ctx, fs_blks);

  LogicalRegion lr_blks = runtime->create_logical_region(ctx, is_blks, fs_blks);

  // assign blocks to shards

  // step 1: ask the mapper how many shards we should even have
  int num_shards = runtime->get_tunable_value(ctx, CGMapper::TID_NUM_SHARDS);
  // force for now
  log_app.print() << "shards = " << num_shards;

  // step 2: fill in the owner IDs and neighbor counts
  {
    InlineLauncher launcher(RegionRequirement(lr_blks, WRITE_DISCARD, EXCLUSIVE, lr_blks)
			    .add_field(fid_owner_shard)
			    .add_field(fid_neighbor_count));
    PhysicalRegion pr_blks = runtime->map_region(ctx, launcher);
    pr_blks.wait_until_valid();

    
    RegionAccessor<AccessorType::Affine<3>, int> fa_owner = fid_owner_shard.accessor<AccessorType::Affine<3> >(pr_blks);
    RegionAccessor<AccessorType::Affine<3>, int> fa_neighbors = fid_neighbor_count.accessor<AccessorType::Affine<3> >(pr_blks);

    int num_blks = volume(blocks);

    {
      int i = 0;
      for(GenericPointInRectIterator<3> pir(Rect<3>(Point<3>::ZEROES(),
						    blocks - Point<3>::ONES()));
	  pir; ++pir)
      fa_owner[pir.p] = (i++ * num_shards) / num_blks;
    }

    for(GenericPointInRectIterator<3> pir(Rect<3>(Point<3>::ZEROES(),
						  blocks - Point<3>::ONES()));
	pir; ++pir) {
      int owner = fa_owner[pir.p];
      int neighbors = 0;
      if((pir.p.x[0] > 0) && (owner != fa_owner[LEFT(pir.p)])) neighbors++;
      if((pir.p.x[0] < blocks.x[0] - 1) && (owner != fa_owner[RIGHT(pir.p)])) neighbors++;
      if((pir.p.x[1] > 0) && (owner != fa_owner[UP(pir.p)])) neighbors++;
      if((pir.p.x[1] < blocks.x[1] - 1) && (owner != fa_owner[DOWN(pir.p)])) neighbors++;
      if((pir.p.x[2] > 0) && (owner != fa_owner[FRONT(pir.p)])) neighbors++;
      if((pir.p.x[2] < blocks.x[2] - 1) && (owner != fa_owner[BACK(pir.p)])) neighbors++;
      fa_neighbors[pir.p] = neighbors;
    }

    runtime->unmap_region(ctx, pr_blks);
  }

  // create a logical region that holds the residual "vector" - this will be shared between all shards
  FieldSpace fs_sol = runtime->create_field_space(ctx);
  fid_sol_p.allocate(runtime, ctx, fs_sol);

  LogicalRegion lr_sol = runtime->create_logical_region(ctx, is_grid, fs_sol);
  LogicalPartition lp_sol = runtime->get_logical_partition(ctx, lr_sol, ip_grid);

  // remap the blocks region as read-only to be compatible with subtask launches
  InlineLauncher launcher(RegionRequirement(lr_blks, READ_ONLY, EXCLUSIVE, lr_blks)
			  .add_field(fid_owner_shard));
  PhysicalRegion pr_blks = runtime->map_region(ctx, launcher);
  pr_blks.wait_until_valid();

  RegionAccessor<AccessorType::Affine<3>, int> fa_owner = fid_owner_shard.accessor<AccessorType::Affine<3> >(pr_blks);

  runtime->fill_field(ctx, lr_blks, lr_blks, fid_ready_barrier, PhaseBarrier());
  runtime->fill_field(ctx, lr_blks, lr_blks, fid_done_barrier, PhaseBarrier());
    
  {
    log_app.info() << "launching init tasks";

    // launch init tasks on each shard that create the needed instances
    std::vector<Future> futures;
    for(int shard = 0; shard < num_shards; shard++) {
      SpmdInitArgs args;
      args.grid_dim = grid_dim;
      args.block_dim = block_dim;
      args.blocks = blocks;
      args.shard = shard;

      TaskLauncher launcher(SPMD_INIT_TASK_ID,
			    TaskArgument(&args, sizeof(args)),
			    Predicate::TRUE_PRED,
			    0 /*default mapper*/,
			    CGMapper::SHARD_TAG(shard));
      launcher.add_region_requirement(RegionRequirement(lr_blks, READ_ONLY, EXCLUSIVE, lr_blks)
				      .add_field(fid_owner_shard)
				      .add_field(fid_neighbor_count));

      // we'll use "reductions" to allow each shard to fill in barriers for blocks it owns
      launcher.add_region_requirement(RegionRequirement(lr_blks,
							BarrierCombineReductionOp::redop_id,
							EXCLUSIVE, lr_blks)
				      .add_field(fid_ready_barrier));
      launcher.add_region_requirement(RegionRequirement(lr_blks,
							BarrierCombineReductionOp::redop_id,
							EXCLUSIVE, lr_blks)
				      .add_field(fid_done_barrier));

      for(GenericPointInRectIterator<3> pir(Rect<3>(Point<3>::ZEROES(),
						    blocks - Point<3>::ONES()));
	  pir; ++pir) {
	if(fa_owner[pir.p] != shard)
	  continue;

	LogicalRegion lr_solblk = runtime->get_logical_subregion_by_color(ctx,
									  lp_sol,
									  DomainPoint::from_point<3>(pir.p));
	launcher.add_region_requirement(RegionRequirement(lr_solblk, WRITE_DISCARD, EXCLUSIVE, lr_sol)
					.add_field(fid_sol_p));
      }

      Future f = runtime->execute_task(ctx, launcher);
      futures.push_back(f);
    }

    log_app.info() << "waiting for init tasks";

    // now wait on all the futures
    for(std::vector<Future>::iterator it = futures.begin();
	it != futures.end();
	it++)
      it->get_void_result();

    log_app.info() << "init tasks complete";
  }

  // now the main simulation, launched as a must epoch
  {
    SpmdMainArgs args;

    args.grid_dim = grid_dim;
    args.block_dim = block_dim;
    args.blocks = blocks;
    args.max_iters = max_iters;
    args.future_lag = future_lag;
    args.show_residuals = show_residuals;
    args.use_tracing = use_tracing;
    args.verbose = verbose;
    {
      double zero = 0.0;
      args.dc_reduction = runtime->create_dynamic_collective(ctx,
							     volume(blocks),
							     DoubleAddReductionOp::redop_id,
							     &zero,
							     sizeof(zero));
    }

    Rect<3> blk_space(Point<3>::ZEROES(), blocks - Point<3>::ONES());

    MustEpochLauncher must;

#define USE_SINGLE_TASK_LAUNCHES_IN_MUST_EPOCH
#ifdef USE_SINGLE_TASK_LAUNCHES_IN_MUST_EPOCH
    for(int shard = 0; shard < num_shards; shard++)
#endif
    {
#ifdef USE_SINGLE_TASK_LAUNCHES_IN_MUST_EPOCH
      TaskLauncher launcher(SPMD_MAIN_TASK_ID,
			    TaskArgument(&args, sizeof(args)),
			    Predicate::TRUE_PRED,
			    0 /*default mapper*/,
			    CGMapper::SHARD_TAG(shard));
#else
      IndexLauncher launcher(SPMD_MAIN_TASK_ID,
	                     Domain::from_rect<1>(Rect<1>(0, num_shards - 1)),
	                     TaskArgument(&args, sizeof(args)),
			     ArgumentMap());
#endif

      launcher.add_region_requirement(RegionRequirement(lr_blks, READ_ONLY, EXCLUSIVE, lr_blks)
	                              .add_field(fid_owner_shard)
				      .add_field(fid_neighbor_count)
				      .add_field(fid_ready_barrier)
				      .add_field(fid_done_barrier));

      for(GenericPointInRectIterator<3> pir(blk_space); pir; ++pir) {
	LogicalRegion lr_solblk = runtime->get_logical_subregion_by_color(ctx,
									  lp_sol,
									  DomainPoint::from_point<3>(pir.p));
	launcher.add_region_requirement(RegionRequirement(lr_solblk, READ_WRITE, SIMULTANEOUS, lr_sol,
							  CGMapper::SHARD_TAG(fa_owner[pir.p]))
					.add_field(fid_sol_p)
					.add_flags(NO_ACCESS_FLAG));
      }
#ifdef USE_SINGLE_TASK_LAUNCHES_IN_MUST_EPOCH
      must.add_single_task(DomainPoint::from_point<1>(shard), launcher);
#else
      must.add_index_task(launcher);
#endif
    }

    log_app.info() << "launching spmd tasks";

    FutureMap fm = runtime->execute_must_epoch(ctx, must);

    log_app.info() << "waiting for spmd tasks";

    fm.wait_all_results();

    log_app.info() << "spmd tasks complete";
  }
}

void spmd_init_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  const SpmdInitArgs& args = *(const SpmdInitArgs *)(task->args);

  log_app.print() << "in spmd_init_task, shard=" << args.shard << ", proc=" << runtime->get_executing_processor(ctx) << ", regions=" << regions.size();

  RegionAccessor<AccessorType::Affine<3>, int> fa_owner = fid_owner_shard.accessor<AccessorType::Affine<3> >(regions[0]);
  RegionAccessor<AccessorType::Affine<3>, int> fa_neighbors = fid_neighbor_count.accessor<AccessorType::Affine<3> >(regions[0]);

  RegionAccessor<AccessorType::Affine<3>, PhaseBarrier> fa_ready = fid_ready_barrier.fold_accessor<AccessorType::Affine<3> >(regions[1]);
  RegionAccessor<AccessorType::Affine<3>, PhaseBarrier> fa_done = fid_done_barrier.fold_accessor<AccessorType::Affine<3> >(regions[2]);

  for(GenericPointInRectIterator<3> pir(Rect<3>(Point<3>::ZEROES(),
						args.blocks - Point<3>::ONES()));
      pir; ++pir) {
    int owner = fa_owner[pir.p];
    if(owner != args.shard)
      continue;

    int neighbors = fa_neighbors[pir.p];
    if(neighbors > 0) {
      PhaseBarrier pb_ready = runtime->create_phase_barrier(ctx, 1);
      PhaseBarrier pb_done = runtime->create_phase_barrier(ctx, neighbors);

      log_app.print() << "pbs: shard=" << args.shard << " blk=" << pir.p << " neighbors=" << neighbors << " ready=" << pb_ready << " done=" << pb_done;

      fa_ready[pir.p] = pb_ready;
      fa_done[pir.p] = pb_done;
    }
  }
}

std::ostream& operator<<(std::ostream& os, const BlockMetadata& bm)
{
  os << "bounds=" << bm.bounds;
  os << " ispace=" << bm.ispace;
  os << " pvt=" << bm.lr_private << " shr=" << bm.lr_shared << "\n";
  os << "    " << bm.neighbors << " neighbors: ready=" << bm.pb_shared_ready << " done=" << bm.pb_shared_done;
  for(int dir = DIR_X; dir <= DIR_Z; dir++)
    for(int side = SIDE_MINUS; side <= SIDE_PLUS; side++) {
      const GhostInfo& g = bm.ghosts[dir][side];
      os << "\n    " << (char)('X' + dir) << ("-+"[side]) << ": gtype=" << g.gtype;
      os << " ready=" << g.pb_shared_ready << " done=" << g.pb_shared_done;
    }
  return os;
}

struct InitFieldArgs {
  Point<3> grid_dim;
  Rect<3> bounds;
};

struct SpmvFieldArgs {
  Rect<3> bounds;
  GhostInfo::GhostType gtypes[3][2];
};

struct AddFieldArgs {
  Rect<3> bounds;
};



void spmd_main_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  const SpmdMainArgs& args = *(const SpmdMainArgs *)(task->args);

  int shard = task->index_point.get_point<1>();

  log_app.print() << "in spmd_main_task, shard=" << shard << ", proc=" << runtime->get_executing_processor(ctx) << ", regions=" << regions.size();

  // iterate over the distribution info, capturing the info for the subblocks we own
  std::map<Point<3>, BlockMetadata, Point<3>::STLComparator> myblocks;

  FieldSpace fs_private = runtime->create_field_space(ctx);
  fid_sol_b.allocate(runtime, ctx, fs_private);
  fid_sol_x.allocate(runtime, ctx, fs_private);
  fid_sol_r.allocate(runtime, ctx, fs_private);
  fid_sol_Ap.allocate(runtime, ctx, fs_private);

  RegionAccessor<AccessorType::Affine<3>, int> fa_owner = fid_owner_shard.accessor<AccessorType::Affine<3> >(regions[0]);
  RegionAccessor<AccessorType::Affine<3>, int> fa_neighbors = fid_neighbor_count.accessor<AccessorType::Affine<3> >(regions[0]);
  RegionAccessor<AccessorType::Affine<3>, PhaseBarrier> fa_ready = fid_ready_barrier.accessor<AccessorType::Affine<3> >(regions[0]);
  RegionAccessor<AccessorType::Affine<3>, PhaseBarrier> fa_done = fid_done_barrier.accessor<AccessorType::Affine<3> >(regions[0]);

  Rect<3> blk_space(Point<3>::ZEROES(), args.blocks - Point<3>::ONES());

  int pr_idx = 1;
  FortranArrayLinearization<3> fal(blk_space, pr_idx);

  for(GenericPointInRectIterator<3> pir(blk_space); pir; ++pir, ++pr_idx) {
    int owner = fa_owner[pir.p];
    if(owner != shard)
      continue;

    assert((int)fal.image(pir.p) == pr_idx);

    BlockMetadata& mdata = myblocks[pir.p];

    mdata.lr_shared = regions[pr_idx].get_logical_region();
    mdata.ispace = mdata.lr_shared.get_index_space();
    mdata.bounds = runtime->get_index_space_domain(ctx, mdata.ispace).get_rect<3>();

    mdata.lr_private = runtime->create_logical_region(ctx, mdata.ispace, fs_private);

    mdata.neighbors = fa_neighbors[pir.p];
    if(mdata.neighbors > 0) {
      mdata.pb_shared_ready = fa_ready[pir.p];
      mdata.pb_shared_done = fa_done[pir.p];
    } else {
      mdata.pb_shared_ready = PhaseBarrier();
      mdata.pb_shared_done = PhaseBarrier();
    }

    for(int dir = DIR_X; dir <= DIR_Z; dir++) {
      for(int side = SIDE_MINUS; side <= SIDE_PLUS; side++) {
	GhostInfo& g = mdata.ghosts[dir][side];
	if(pir.p.x[dir] == ((side == SIDE_MINUS) ? 0 : args.blocks.x[dir] - 1)) {
	  // boundary
	  g.gtype = GhostInfo::GHOST_BOUNDARY;
	} else {
	  Point<3> p2(pir.p);
	  if(side == SIDE_MINUS)
	    p2.x[dir]--;
	  else
	    p2.x[dir]++;

	  if(fa_owner[p2] == owner) {
	    g.gtype = GhostInfo::GHOST_LOCAL;
	    g.lr_parent = regions[fal.image(p2)].get_logical_region();
	  } else {
	    g.gtype = GhostInfo::GHOST_REMOTE;
	    g.lr_parent = regions[fal.image(p2)].get_logical_region();
	    g.pb_shared_ready = fa_ready[p2];
	    g.pb_shared_done = fa_done[p2];
	  }

	  IndexSpace is_parent = g.lr_parent.get_index_space();
	  Rect<3> r_parent = runtime->get_index_space_domain(ctx, is_parent).get_rect<3>();
	  Rect<3> r_subset(r_parent);
	  if(side == SIDE_MINUS)
	    r_subset.lo.x[dir] = r_subset.hi.x[dir];  // just 1 plane
	  else
	    r_subset.hi.x[dir] = r_subset.lo.x[dir];  // just 1 plane
	  std::cout << "dir=" << dir << " side=" << side << " -> " << r_parent << " -> " << r_subset << "\n";
	  DomainColoring dc;
	  dc[0] = Domain::from_rect<3>(r_subset);
	  IndexPartition ip = runtime->create_index_partition(ctx,
							      is_parent,
							      Domain::from_rect<1>(Rect<1>(0, 0)),
							      dc,
							      dir*2+side);
	  IndexSpace is_subset = runtime->get_index_subspace(ctx, ip, 0);
	  assert(runtime->get_index_space_domain(ctx, is_subset).get_rect<3>() == r_subset);
	  g.ispace = is_subset;
	  g.lr_shared = runtime->get_logical_subregion_by_tree(ctx,
							       is_subset,
							       g.lr_parent.get_field_space(),
							       g.lr_parent.get_tree_id());

	  if(fa_owner[p2] == owner) {
	    g.lr_ghost = g.lr_shared;
	  } else {
	    g.lr_ghost = runtime->create_logical_region(ctx,
							is_subset,
							g.lr_parent.get_field_space());
	  }
	}
      }
    }
  }

  // now that we're done with that, unmap all of our regions so that they
  //  don't interfere with child task launches
  runtime->unmap_all_regions(ctx);

  std::cout << "my block count = " << myblocks.size() << "\n";
  for(std::map<Point<3>, BlockMetadata>::const_iterator it = myblocks.begin();
      it != myblocks.end();
      it++)
    std::cout << "  " << it->first << ": " << it->second << "\n";

  // this shouldn't be necessary, but it seems to avoid some sort of memory corruption when
  //  a shard has no work
  if(myblocks.empty())
    return;

  // initialize x, b
  for(std::map<Point<3>, BlockMetadata>::iterator it = myblocks.begin();
      it != myblocks.end();
      it++) {
    BlockMetadata& bm = it->second;

    {
      InitFieldArgs cargs;
      cargs.grid_dim = args.grid_dim;
      cargs.bounds = bm.bounds;

      TaskLauncher launcher(INIT_FIELD_TASK_ID,
			    TaskArgument(&cargs, sizeof(cargs)),
			    Predicate::TRUE_PRED,
			    0 /*default mapper*/,
			    CGMapper::TAG_LOCAL_SHARD);
      launcher.add_region_requirement(RegionRequirement(bm.lr_private,
							READ_WRITE,
							EXCLUSIVE,
							bm.lr_private)
				      .add_field(fid_sol_b));
      runtime->execute_task(ctx, launcher);
    }

    // initial solution is zero
    runtime->fill_field<double>(ctx, bm.lr_private, bm.lr_private, fid_sol_x, 0.0);

    // initial residual is equal to b
    {
      CopyLauncher copy;
      copy.add_copy_requirements(RegionRequirement(bm.lr_private, READ_ONLY, EXCLUSIVE, bm.lr_private)
				 .add_field(fid_sol_b),
				 RegionRequirement(bm.lr_private, WRITE_DISCARD, EXCLUSIVE, bm.lr_private)
				 .add_field(fid_sol_r));
      runtime->issue_copy_operation(ctx, copy);
    }

    // initial direction vector p is also equal to b
    {
      CopyLauncher copy;
      copy.add_copy_requirements(RegionRequirement(bm.lr_private, READ_ONLY, EXCLUSIVE, bm.lr_private)
				 .add_field(fid_sol_b),
				 RegionRequirement(bm.lr_shared, WRITE_DISCARD, EXCLUSIVE, bm.lr_shared)
				 .add_field(fid_sol_p));
      if(bm.neighbors > 0) {
	copy.add_arrival_barrier(bm.pb_shared_ready);
	bm.pb_shared_ready = runtime->advance_phase_barrier(ctx, bm.pb_shared_ready);
      }
      runtime->issue_copy_operation(ctx, copy);
    }
  }

  DynamicCollective dc_reduction = args.dc_reduction;

  // compute initial residual magnitude
  Future f_resold = DotProduct::compute(myblocks, dc_reduction, runtime, ctx,
					fid_sol_b, true /*private*/,
					fid_sol_b, true /*private*/);
  std::queue<Future> f_residuals;
  int res_iter = 0;  // this lags the normal iteration count
  if(args.show_residuals > 0) {
    double resold = f_resold.get_result<double>();
    if(shard == 0)
      std::cout << "resold = " << resold << "\n";
  }

  Future f_restarget = Future::from_value<double>(runtime, 1e-10);

  Predicate p_notdone = Predicate::TRUE_PRED;

  int iter = 0;
  while(true) {
    iter++;

    if(args.use_tracing)
      runtime->begin_trace(ctx, TRACE_ID_CG_ITER);

    // compute Ap = A * p
    for(std::map<Point<3>, BlockMetadata>::iterator it = myblocks.begin();
	it != myblocks.end();
	it++) {
      BlockMetadata& bm = it->second;
      
      SpmvFieldArgs cargs;
      cargs.bounds = bm.bounds;
      for(int dir = DIR_X; dir <= DIR_Z; dir++)
	for(int side = SIDE_MINUS; side <= SIDE_PLUS; side++)
	  cargs.gtypes[dir][side] = bm.ghosts[dir][side].gtype;

      TaskLauncher launcher(SPMV_FIELD_TASK_ID,
			    TaskArgument(&cargs, sizeof(cargs)),
			    p_notdone,
			    0 /*default mapper*/,
			    CGMapper::TAG_LOCAL_SHARD);
      launcher.add_region_requirement(RegionRequirement(bm.lr_private,
							READ_WRITE,
							EXCLUSIVE,
							bm.lr_private)
				      .add_field(fid_sol_Ap));
      launcher.add_region_requirement(RegionRequirement(bm.lr_shared,
							READ_ONLY,
							EXCLUSIVE,
							bm.lr_shared)
				      .add_field(fid_sol_p));
      for(int dir = DIR_X; dir <= DIR_Z; dir++)
	for(int side = SIDE_MINUS; side <= SIDE_PLUS; side++) {
	  GhostInfo& g = bm.ghosts[dir][side];
	  switch(g.gtype) {
	  case GhostInfo::GHOST_BOUNDARY: 
	    {
	      // can't do empty region requirements right now
#if 0
	      launcher.add_region_requirement(RegionRequirement(LogicalRegion::NO_REGION,
								NO_ACCESS,
								EXCLUSIVE,
								LogicalRegion::NO_REGION));
#endif
	      break;
	    }

	  case GhostInfo::GHOST_LOCAL:
	    {
	      launcher.add_region_requirement(RegionRequirement(g.lr_shared,
								READ_ONLY,
								EXCLUSIVE,
								g.lr_parent)
					      .add_field(fid_sol_p));
	      break;
	    }

	  case GhostInfo::GHOST_REMOTE:
	    {
	      CopyLauncher copy(p_notdone);
	      copy.add_copy_requirements(RegionRequirement(g.lr_shared, READ_ONLY, EXCLUSIVE, g.lr_parent)
					 .add_field(fid_sol_p),
					 RegionRequirement(g.lr_ghost, WRITE_DISCARD, EXCLUSIVE, g.lr_ghost)
					 .add_field(fid_sol_p));
	      g.pb_shared_ready = runtime->advance_phase_barrier(ctx, g.pb_shared_ready);
	      copy.add_wait_barrier(g.pb_shared_ready);
	      copy.add_arrival_barrier(g.pb_shared_done);
	      g.pb_shared_done = runtime->advance_phase_barrier(ctx, g.pb_shared_done);
	      runtime->issue_copy_operation(ctx, copy);

	      launcher.add_region_requirement(RegionRequirement(g.lr_ghost,
								READ_ONLY,
								EXCLUSIVE,
								g.lr_ghost)
					      .add_field(fid_sol_p));
	      break;
	    }

	  default:
	    assert(0);
	  }
	}
      
      runtime->execute_task(ctx, launcher);
    }

    // alpha = p' * Ap
    Future ff2 = DotProduct::compute(myblocks, dc_reduction, runtime, ctx,
				     fid_sol_p, false /*!private*/,
				     fid_sol_Ap, true /*private*/,
				     p_notdone);
    Future f_alpha = FDV_double::compute(runtime, ctx, f_resold, ff2);
    //double alpha = resold / ff2.get_result<double>();
    double alpha = f_alpha.get_result<double>();
    //std::cout << "alpha = " << alpha << "\n";

    // x += alpha * p
    VectorAcc::compute(myblocks, runtime, ctx,
		       alpha, fid_sol_p, false /*!private*/,
		       1.0, fid_sol_x, true /*private*/,
		       p_notdone);

    // r -= alpha * Ap
    VectorAcc::compute(myblocks, runtime, ctx,
		       -alpha, fid_sol_Ap, true /*private*/,
		       1.0, fid_sol_r, true /*private*/,
		       p_notdone);

    //PrintField::compute(myblocks, runtime, ctx, "r", fid_sol_r, true /*private*/);

    Future f_resnew = DotProduct::compute(myblocks, dc_reduction, runtime, ctx,
					  fid_sol_r, true /*private*/,
					  fid_sol_r, true /*private*/,
					  p_notdone);
    f_residuals.push(f_resnew);
    if(f_residuals.size() > (size_t)args.future_lag) {
      Future f = f_residuals.front();
      f_residuals.pop();
      double resnew = f.get_result<double>();
      ++res_iter;
      if((shard == 0) && (args.show_residuals > 0) && ((res_iter % args.show_residuals) == 0))
	std::cout << "iter " << res_iter << ": resnew = " << resnew << "\n";

      if(resnew < 1e-10) {
	if(shard == 0)
	  std::cout << "converged after " << res_iter << " iterations\n";
	break;
      }

      if(res_iter >= args.max_iters) {
	if(shard == 0)
	  std::cout << "failed to converge after " << res_iter << " iterations\n";
	break;
      }
    }

    // never speculate past max_iters
    if(iter >= args.max_iters) {
      if(shard == 0)
	std::cout << "not speculating past " << iter << " iterations\n";
      break;
    }

    if(args.future_lag > 0) {
      Future f_notdone = FLT_double::compute(runtime, ctx, f_restarget, f_resnew);
      p_notdone = runtime->predicate_and(ctx, p_notdone,
					 runtime->create_predicate(ctx, f_notdone));
    }

    // p = r + (resnew/resold)*p
    Future f_beta = FDV_double::compute(runtime, ctx, f_resnew, f_resold);
    double beta = f_beta.get_result<double>();
    VectorAcc::compute(myblocks, runtime, ctx,
		       1.0, fid_sol_r, true /*private*/,
		       beta, fid_sol_p, false /*!private*/,
		       p_notdone);

    f_resold = f_resnew;

    if(args.use_tracing)
      runtime->end_trace(ctx, TRACE_ID_CG_ITER);
  }

  // last iteration of the loop above always breaks out early, so make sure we end the trace
  if(args.use_tracing)
    runtime->end_trace(ctx, TRACE_ID_CG_ITER);

  if(0)
    PrintField::compute(myblocks, runtime, ctx,
			"x", fid_sol_x, true /*private*/,
			0);//1e-5);

  // check result
  // copy p = x
  for(std::map<Point<3>, BlockMetadata>::iterator it = myblocks.begin();
      it != myblocks.end();
      it++) {
    BlockMetadata& bm = it->second;

    CopyLauncher copy;
    copy.add_copy_requirements(RegionRequirement(bm.lr_private, READ_ONLY, EXCLUSIVE, bm.lr_private)
			       .add_field(fid_sol_x),
			       RegionRequirement(bm.lr_shared, WRITE_DISCARD, EXCLUSIVE, bm.lr_shared)
			       .add_field(fid_sol_p));
    if(bm.neighbors > 0) {
      bm.pb_shared_done = runtime->advance_phase_barrier(ctx, bm.pb_shared_done);
      copy.add_wait_barrier(bm.pb_shared_done);
      copy.add_arrival_barrier(bm.pb_shared_ready);
      bm.pb_shared_ready = runtime->advance_phase_barrier(ctx, bm.pb_shared_ready);
    }
    runtime->issue_copy_operation(ctx, copy);
  }

  // compute Ap = A * x
  for(std::map<Point<3>, BlockMetadata>::iterator it = myblocks.begin();
      it != myblocks.end();
      it++) {
    BlockMetadata& bm = it->second;
      
    SpmvFieldArgs cargs;
    cargs.bounds = bm.bounds;
    for(int dir = DIR_X; dir <= DIR_Z; dir++)
      for(int side = SIDE_MINUS; side <= SIDE_PLUS; side++)
	cargs.gtypes[dir][side] = bm.ghosts[dir][side].gtype;

    TaskLauncher launcher(SPMV_FIELD_TASK_ID,
			  TaskArgument(&cargs, sizeof(cargs)),
			  Predicate::TRUE_PRED,
			  0 /*default mapper*/,
			  CGMapper::TAG_LOCAL_SHARD);
    launcher.add_region_requirement(RegionRequirement(bm.lr_private,
						      READ_WRITE,
						      EXCLUSIVE,
						      bm.lr_private)
				    .add_field(fid_sol_Ap));
    launcher.add_region_requirement(RegionRequirement(bm.lr_shared,
						      READ_ONLY,
						      EXCLUSIVE,
						      bm.lr_shared)
				    .add_field(fid_sol_p));
    for(int dir = DIR_X; dir <= DIR_Z; dir++)
      for(int side = SIDE_MINUS; side <= SIDE_PLUS; side++) {
	GhostInfo& g = bm.ghosts[dir][side];
	switch(g.gtype) {
	case GhostInfo::GHOST_BOUNDARY: 
	  {
	    // can't do empty region requirements right now
#if 0
	    launcher.add_region_requirement(RegionRequirement(LogicalRegion::NO_REGION,
							      NO_ACCESS,
							      EXCLUSIVE,
							      LogicalRegion::NO_REGION));
#endif
	    break;
	  }
	  
	case GhostInfo::GHOST_LOCAL:
	  {
	    launcher.add_region_requirement(RegionRequirement(g.lr_shared,
							      READ_ONLY,
							      EXCLUSIVE,
							      g.lr_parent)
					    .add_field(fid_sol_p));
	    break;
	  }

	case GhostInfo::GHOST_REMOTE:
	  {
	    CopyLauncher copy;
	    copy.add_copy_requirements(RegionRequirement(g.lr_shared, READ_ONLY, EXCLUSIVE, g.lr_parent)
				       .add_field(fid_sol_p),
				       RegionRequirement(g.lr_ghost, WRITE_DISCARD, EXCLUSIVE, g.lr_ghost)
				       .add_field(fid_sol_p));
	    g.pb_shared_ready = runtime->advance_phase_barrier(ctx, g.pb_shared_ready);
	    copy.add_wait_barrier(g.pb_shared_ready);
	    copy.add_arrival_barrier(g.pb_shared_done);
	    g.pb_shared_done = runtime->advance_phase_barrier(ctx, g.pb_shared_done);
	    runtime->issue_copy_operation(ctx, copy);
	    
	    launcher.add_region_requirement(RegionRequirement(g.lr_ghost,
							      READ_ONLY,
							      EXCLUSIVE,
							      g.lr_ghost)
					    .add_field(fid_sol_p));
	    break;
	  }

	default:
	  assert(0);
	}
      }
    
    runtime->execute_task(ctx, launcher);
  }

  // Ap -= b
  VectorAcc::compute(myblocks, runtime, ctx,
		     -1.0, fid_sol_b, true /*private*/,
		     1.0, fid_sol_Ap, true /*private*/);
  if(args.verbose)
    PrintField::compute(myblocks, runtime, ctx,
			"check", fid_sol_Ap, true /*private*/,
			1e-5);
}

void init_field_task(const Task *task,
		     const std::vector<PhysicalRegion> &regions,
		     Context ctx, HighLevelRuntime *runtime)
{
  const InitFieldArgs& args = *(const InitFieldArgs *)(task->args);

  log_app.info() << "init_field task - bounds=" << args.bounds << ", fid=" << task->regions[0].instance_fields[0] << ", proc=" << runtime->get_executing_processor(ctx);

  RegionAccessor<AccessorType::Affine<3>, double> fa = regions[0].get_field_accessor(task->regions[0].instance_fields[0]).typeify<double>().convert<AccessorType::Affine<3> >();
  
  for(GenericPointInRectIterator<3> pir(args.bounds); pir; ++pir) {
#if 0
    double v = 0.0;
    if((pir.p.x[0] == (args.grid_dim.x[0] / 2)) &&
       (pir.p.x[1] == (args.grid_dim.x[1] / 2)) &&
       (pir.p.x[2] == (args.grid_dim.x[2] / 2)))
      v = 1.0;
#else
    srand48(pir.p.x[0] * 100000 + pir.p.x[1] * 1000 + pir.p.x[2]);
    double v = drand48();
#endif
    //double v = pir.p.x[0] * 10000 + pir.p.x[1] * 100 + pir.p.x[2];
    //printf("%p <- %g\n", &fa[pir.p], v);
    fa[pir.p] = v;
  }
}

void spmv_field_task(const Task *task,
		     const std::vector<PhysicalRegion> &regions,
		     Context ctx, HighLevelRuntime *runtime)
{
  const SpmvFieldArgs& args = *(const SpmvFieldArgs *)(task->args);

  log_app.info() << "init_field task - bounds=" << args.bounds << ", fid=" << task->regions[0].instance_fields[0] << ", proc=" << runtime->get_executing_processor(ctx);

  RegionAccessor<AccessorType::Affine<3>, double> fa_out = regions[0].get_field_accessor(task->regions[0].instance_fields[0]).typeify<double>().convert<AccessorType::Affine<3> >();
  RegionAccessor<AccessorType::Affine<3>, double> fa_in = regions[1].get_field_accessor(task->regions[1].instance_fields[0]).typeify<double>().convert<AccessorType::Affine<3> >();
  size_t pr_idx = 2;
  RegionAccessor<AccessorType::Affine<3>, double> fa_ghost[3][2];
  for(int dir = DIR_X; dir <= DIR_Z; dir++)
    for(int side = SIDE_MINUS; side <= SIDE_PLUS; side++)
      if(args.gtypes[dir][side] != GhostInfo::GHOST_BOUNDARY) {
	fa_ghost[dir][side] = regions[pr_idx].get_field_accessor(task->regions[pr_idx].instance_fields[0]).typeify<double>().convert<AccessorType::Affine<3> >();
	pr_idx++;
      }
  assert(pr_idx == regions.size());
  
  for(GenericPointInRectIterator<3> pir(args.bounds); pir; ++pir) {
    double v_neighbors[3][2];
    for(int dir = DIR_X; dir <= DIR_Z; dir++)
      for(int side = SIDE_MINUS; side <= SIDE_PLUS; side++) {
	Point<3> p2(pir.p);
	bool interior;
	if(side == SIDE_MINUS) {
	  p2.x[dir]--;
	  interior = (p2.x[dir] >= args.bounds.lo[dir]);
	} else {
	  p2.x[dir]++;
	  interior = (p2.x[dir] <= args.bounds.hi[dir]);
	}
	if(interior) {
	  v_neighbors[dir][side] = fa_in[p2];
	} else {
	  if(args.gtypes[dir][side] == GhostInfo::GHOST_BOUNDARY)
	    v_neighbors[dir][side] = 0.0;
	  else
	    v_neighbors[dir][side] = fa_ghost[dir][side][p2];
	}
      }
	
    double v_in = fa_in[pir.p];
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;
    double v_out = (((-v_neighbors[DIR_X][SIDE_MINUS] + 2 * v_in - v_neighbors[DIR_X][SIDE_PLUS]) / (dx * dx)) +
		    ((-v_neighbors[DIR_Y][SIDE_MINUS] + 2 * v_in - v_neighbors[DIR_Y][SIDE_PLUS]) / (dy * dy)) +
		    ((-v_neighbors[DIR_Z][SIDE_MINUS] + 2 * v_in - v_neighbors[DIR_Z][SIDE_PLUS]) / (dz * dz)));
#if 0
    std::cout << pir.p << " " << v_out << " <- " << v_in
	      << " x:(" << v_neighbors[DIR_X][SIDE_MINUS] << "," << v_neighbors[DIR_X][SIDE_PLUS] << ")"
	      << " y:(" << v_neighbors[DIR_Y][SIDE_MINUS] << "," << v_neighbors[DIR_Y][SIDE_PLUS] << ")"
	      << " z:(" << v_neighbors[DIR_Z][SIDE_MINUS] << "," << v_neighbors[DIR_Z][SIDE_PLUS] << ")\n";
#endif
    fa_out[pir.p] = v_out;
  }
}

static void update_mappers(Machine machine, HighLevelRuntime *runtime,
                           const std::set<Processor> &local_procs)
{
  for(std::set<Processor>::const_iterator it = local_procs.begin();
      it != local_procs.end();
      it++) 
    runtime->replace_default_mapper(new CGMapper(machine, runtime, *it), *it);
}

int main(int argc, char **argv)
{

  {
    TaskVariantRegistrar tvr(TOP_LEVEL_TASK_ID, "top_level_task");
    tvr.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(tvr, "top_level_task");
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  }

  {
    TaskVariantRegistrar tvr(SPMD_INIT_TASK_ID, "spmd_init_task");
    tvr.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    tvr.set_leaf(true);
    Runtime::preregister_task_variant<spmd_init_task>(tvr, "spmd_init_task");
  }

  {
    TaskVariantRegistrar tvr(SPMD_MAIN_TASK_ID, "spmd_main_task");
    tvr.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<spmd_main_task>(tvr, "spmd_main_task");
  }

  {
    TaskVariantRegistrar tvr(INIT_FIELD_TASK_ID, "init_field_task");
    tvr.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    tvr.set_leaf(true);
    Runtime::preregister_task_variant<init_field_task>(tvr, "init_field_task");
  }

  {
    TaskVariantRegistrar tvr(SPMV_FIELD_TASK_ID, "spmv_field_task");
    tvr.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    tvr.set_leaf(true);
    Runtime::preregister_task_variant<spmv_field_task>(tvr, "spmv_field_task");
  }

  PrintField::preregister_tasks();
  DotProduct::preregister_tasks();
  VectorAdd::preregister_tasks();
  VectorAcc::preregister_tasks();

  FLT_double::preregister_task();
  FDV_double::preregister_task();

  Runtime::set_registration_callback(update_mappers);

  Runtime::register_reduction_op<BarrierCombineReductionOp>(BarrierCombineReductionOp::redop_id);
  Runtime::register_reduction_op<DoubleAddReductionOp>(DoubleAddReductionOp::redop_id);

  return Runtime::start(argc, argv);
}

