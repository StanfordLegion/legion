#include "cgtasks.h"
#include "cgmapper.h"

#include <math.h>

extern Logger log_app;

////////////////////////////////////////////////////////////////////////
//
// class PrintField
//

/*static*/ TaskID PrintField::taskid = 65;//AUTO_GENERATE_ID;

/*static*/ void PrintField::compute(const MyBlockMap& myblocks,
				    Runtime *runtime, Context ctx,
				    const char *prefix,
				    FieldID fid1, bool private1,
				    double minval /*= 0*/)
{
  std::vector<Future> futures;
  for(std::map<Point<3>, BlockMetadata>::const_iterator it = myblocks.begin();
      it != myblocks.end();
      it++) {
    const BlockMetadata& bm = it->second;
    
    PrintFieldArgs cargs;
    cargs.bounds = bm.bounds;
    strncpy(cargs.prefix, prefix, 32);
    cargs.prefix[31] = 0;
    cargs.minval = minval;

    TaskLauncher launcher(taskid,
			  TaskArgument(&cargs, sizeof(cargs)),
			  Predicate::TRUE_PRED,
			  0 /*default mapper*/,
			  CGMapper::TAG_LOCAL_SHARD);
    LogicalRegion lr1 = private1 ? bm.lr_private : bm.lr_shared;
    launcher.add_region_requirement(RegionRequirement(lr1,
						      READ_ONLY,
						      EXCLUSIVE,
						      lr1)
				    .add_field(fid1));

    Future f = runtime->execute_task(ctx, launcher);
    futures.push_back(f);
  }
  log_app.debug() << "launched " << futures.size() << " print_field tasks";
  for(std::vector<Future>::iterator it = futures.begin();
      it != futures.end();
      it++)
    it->get_void_result();
  log_app.debug() << "all tasks finished";
}

/*static*/ void PrintField::preregister_tasks(void)
{
  //taskid = Runtime::generate_static_task_id();
  TaskVariantRegistrar tvr(taskid, "print_field_task");
  tvr.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  tvr.set_leaf(true);
  Runtime::preregister_task_variant<print_field_task>(tvr, "print_field_task");
}

/*static*/ void PrintField::print_field_task(const Task *task,
					     const std::vector<PhysicalRegion> &regions,
					     Context ctx, Runtime *runtime)
{
  const PrintFieldArgs& args = *(const PrintFieldArgs *)(task->args);

  const char *fname1 = "(unknown)";
  runtime->retrieve_name(task->regions[0].region.get_field_space(), task->regions[0].instance_fields[0], fname1);
  log_app.debug() << "print_field task - bounds=" << args.bounds 
		 << ", fid1=" << task->regions[0].instance_fields[0] << "(" << fname1 << ")"
		 << ", proc=" << runtime->get_executing_processor(ctx);

  const AccessorROdouble fa1(regions[0], task->regions[0].instance_fields[0]);

  log_app.debug() << "&fid1[" << args.bounds.lo << "] = " << (void *)(&fa1[args.bounds.lo]) << "\n";

  std::ostringstream oss;
  for(int z = args.bounds.lo[2]; z <= args.bounds.hi[2]; z++)
    for(int y = args.bounds.lo[1]; y <= args.bounds.hi[1]; y++) {
      int x_lo = args.bounds.lo[0];
      int x_hi = args.bounds.hi[0];
      if(args.minval > 0) {
	bool show = false;
	for(int x = x_lo; x <= x_hi; x++) {
	  double v = fa1[Point<3>(x, y, z)];
	  if(fabs(v) >= args.minval) {
	    show = true;
	    break;
	  }
	}
	if(!show) continue;
      }
      oss << args.prefix << ": z=" << z << " y=" << y << " x=" << x_lo << ".." << x_hi << ":";
      for(int x = x_lo; x <= x_hi; x++) {
	double v = fa1[Point<3>(x, y, z)];
	if(fabs(v) < args.minval)
	  oss << " -";
	else
	  oss << ' ' << v;
      }
      oss << '\n';
    }
  std::cout << oss.str();
}


////////////////////////////////////////////////////////////////////////
//
// class DotProduct
//

/*static*/ TaskID DotProduct::taskid = 66;//AUTO_GENERATE_ID;

/*static*/ Future DotProduct::compute(const MyBlockMap& myblocks,
				      DynamicCollective& dc_reduction,
				      Runtime *runtime, Context ctx,
				      FieldID fid1, bool private1,
				      FieldID fid2, bool private2,
				      Predicate pred /*= Predicate::TRUE_PRED*/)
{
  for(std::map<Point<3>, BlockMetadata>::const_iterator it = myblocks.begin();
      it != myblocks.end();
      it++) {
    const BlockMetadata& bm = it->second;
    
    DotpFieldArgs cargs;
    cargs.bounds = bm.bounds;

    TaskLauncher launcher(taskid,
			  TaskArgument(&cargs, sizeof(cargs)),
			  pred,
			  0 /*default mapper*/,
			  CGMapper::TAG_LOCAL_SHARD);
    double zero = 0.0;
    launcher.set_predicate_false_result(TaskArgument(&zero, sizeof(zero)));

    LogicalRegion lr1 = private1 ? bm.lr_private : bm.lr_shared;
    launcher.add_region_requirement(RegionRequirement(lr1,
						      READ_ONLY,
						      EXCLUSIVE,
						      lr1)
				    .add_field(fid1));
    LogicalRegion lr2 = private2 ? bm.lr_private : bm.lr_shared;
    launcher.add_region_requirement(RegionRequirement(lr2,
						      READ_ONLY,
						      EXCLUSIVE,
						      lr2)
				    .add_field(fid2));
    Future f = runtime->execute_task(ctx, launcher);
    runtime->defer_dynamic_collective_arrival(ctx, 
					      dc_reduction,
					      f);
    //double d = f.get_result<double>();
    //std::cout << "d = " << d << "\n";
  }
  dc_reduction = runtime->advance_dynamic_collective(ctx, dc_reduction);
  Future ff2 = runtime->get_dynamic_collective_result(ctx, dc_reduction);
  return ff2;
}

/*static*/ void DotProduct::preregister_tasks(void)
{
  //taskid = Runtime::generate_static_task_id();
  TaskVariantRegistrar tvr(taskid, "dotp_field_task");
  tvr.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  tvr.set_leaf(true);
  Runtime::preregister_task_variant<double, dotp_field_task>(tvr, "dotp_field_task");
}

/*static*/ double DotProduct::dotp_field_task(const Task *task,
					      const std::vector<PhysicalRegion> &regions,
					      Context ctx, Runtime *runtime)
{
  const DotpFieldArgs& args = *(const DotpFieldArgs *)(task->args);

  log_app.debug() << "dotp_field task - bounds=" << args.bounds 
		 << ", fid1=" << task->regions[0].instance_fields[0]
		 << ", fid2=" << task->regions[1].instance_fields[0]
		 << ", proc=" << runtime->get_executing_processor(ctx);

  const AccessorROdouble fa1(regions[0], task->regions[0].instance_fields[0]);
  const AccessorROdouble fa2(regions[1], task->regions[1].instance_fields[0]);
  double sum = 0.0;
  
  for(PointInRectIterator<3> pir(args.bounds); pir(); ++pir)
    sum += fa1[*pir] * fa2[*pir];

  return sum;
}


////////////////////////////////////////////////////////////////////////
//
// class VectorAdd
//

/*static*/ TaskID VectorAdd::taskid = 67;//AUTO_GENERATE_ID;

/*static*/ void VectorAdd::compute(const MyBlockMap& myblocks,
				   Runtime *runtime, Context ctx,
				   double alpha1, FieldID fid1, bool private1,
				   double alpha2, FieldID fid2, bool private2,
				   FieldID fid_sum, bool private_sum,
				   Predicate pred /*= Predicate::TRUE_PRED*/)
{
  for(std::map<Point<3>, BlockMetadata>::const_iterator it = myblocks.begin();
      it != myblocks.end();
      it++) {
    const BlockMetadata& bm = it->second;
    
    AddFieldArgs cargs;
    cargs.bounds = bm.bounds;
    cargs.alpha1 = alpha1;
    cargs.alpha2 = alpha2;

    TaskLauncher launcher(taskid,
			  TaskArgument(&cargs, sizeof(cargs)),
			  pred,
			  0 /*default mapper*/,
			  CGMapper::TAG_LOCAL_SHARD);
    LogicalRegion lr_sum = private_sum ? bm.lr_private : bm.lr_shared;
    launcher.add_region_requirement(RegionRequirement(lr_sum,
						      WRITE_DISCARD,
						      EXCLUSIVE,
						      lr_sum)
				    .add_field(fid_sum));
    LogicalRegion lr1 = private1 ? bm.lr_private : bm.lr_shared;
    launcher.add_region_requirement(RegionRequirement(lr1,
						      READ_ONLY,
						      EXCLUSIVE,
						      lr1)
				    .add_field(fid1));
    LogicalRegion lr2 = private2 ? bm.lr_private : bm.lr_shared;
    launcher.add_region_requirement(RegionRequirement(lr2,
						      READ_ONLY,
						      EXCLUSIVE,
						      lr2)
				    .add_field(fid2));
    runtime->execute_task(ctx, launcher);
  }
}

/*static*/ void VectorAdd::preregister_tasks(void)
{
  //taskid = Runtime::generate_static_task_id();
  TaskVariantRegistrar tvr(taskid, "add_field_task");
  tvr.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  tvr.set_leaf(true);
  Runtime::preregister_task_variant<add_field_task>(tvr, "add_field_task");
}

/*static*/ void VectorAdd::add_field_task(const Task *task,
					  const std::vector<PhysicalRegion> &regions,
					  Context ctx, Runtime *runtime)
{
  const AddFieldArgs& args = *(const AddFieldArgs *)(task->args);

  log_app.debug() << "add_field task - bounds=" << args.bounds 
		 << ", sum=" << task->regions[0].instance_fields[0]
		 << ", fid1=" << task->regions[1].instance_fields[0]
		 << ", fid2=" << task->regions[2].instance_fields[0]
		 << ", proc=" << runtime->get_executing_processor(ctx);

  const AccessorWDdouble fa_sum(regions[0], task->regions[0].instance_fields[0]);
  const AccessorROdouble fa1(regions[1], task->regions[1].instance_fields[0]);
  const AccessorROdouble fa2(regions[2], task->regions[2].instance_fields[0]);

  for(PointInRectIterator<3> pir(args.bounds); pir(); ++pir)
    fa_sum[*pir] = args.alpha1 * fa1[*pir] + args.alpha2 * fa2[*pir];
}


////////////////////////////////////////////////////////////////////////
//
// class VectorAcc
//

/*static*/ TaskID VectorAcc::taskid = 68;//AUTO_GENERATE_ID;

/*static*/ void VectorAcc::compute(MyBlockMap& myblocks,
				   Runtime *runtime, Context ctx,
				   double alpha_in, FieldID fid_in, bool private_in,
				   double alpha_acc, FieldID fid_acc, bool private_acc,
				   Predicate pred /*= Predicate::TRUE_PRED*/)
{
  for(std::map<Point<3>, BlockMetadata>::iterator it = myblocks.begin();
      it != myblocks.end();
      it++) {
    BlockMetadata& bm = it->second;
    
    AccFieldArgs cargs;
    cargs.bounds = bm.bounds;
    cargs.alpha_in = alpha_in;
    cargs.alpha_acc = alpha_acc;

    TaskLauncher launcher(taskid,
			  TaskArgument(&cargs, sizeof(cargs)),
			  pred,
			  0 /*default mapper*/,
			  CGMapper::TAG_LOCAL_SHARD);
    LogicalRegion lr_acc = private_acc ? bm.lr_private : bm.lr_shared;
    launcher.add_region_requirement(RegionRequirement(lr_acc,
						      READ_WRITE,
						      EXCLUSIVE,
						      lr_acc)
				    .add_field(fid_acc));
    LogicalRegion lr_in = private_in ? bm.lr_private : bm.lr_shared;
    launcher.add_region_requirement(RegionRequirement(lr_in,
						      READ_ONLY,
						      EXCLUSIVE,
						      lr_in)
				    .add_field(fid_in));

    if(!private_acc && (bm.neighbors > 0)) {
      bm.pb_shared_done = runtime->advance_phase_barrier(ctx, bm.pb_shared_done);
      launcher.add_wait_barrier(bm.pb_shared_done);
      launcher.add_arrival_barrier(bm.pb_shared_ready);
      bm.pb_shared_ready = runtime->advance_phase_barrier(ctx, bm.pb_shared_ready);
    }
    runtime->execute_task(ctx, launcher);
  }
}

/*static*/ void VectorAcc::preregister_tasks(void)
{
  //taskid = Runtime::generate_static_task_id();
  TaskVariantRegistrar tvr(taskid, "acc_field_task");
  tvr.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  tvr.set_leaf(true);
  Runtime::preregister_task_variant<acc_field_task>(tvr, "acc_field_task");
}

/*static*/ void VectorAcc::acc_field_task(const Task *task,
					  const std::vector<PhysicalRegion> &regions,
					  Context ctx, Runtime *runtime)
{
  const AccFieldArgs& args = *(const AccFieldArgs *)(task->args);

  const char *fname1 = "(unknown)";
  runtime->retrieve_name(task->regions[0].region.get_field_space(), task->regions[0].instance_fields[0], fname1);
  const char *fname2 = "(unknown)";
  runtime->retrieve_name(task->regions[1].region.get_field_space(), task->regions[1].instance_fields[0], fname2);
  log_app.debug() << "acc_field task - bounds=" << args.bounds 
		 << ", acc=" << task->regions[0].instance_fields[0] << "(" << fname1 << ")"
		 << ", in=" << task->regions[1].instance_fields[0] << "(" << fname2 << ")"
		 << ", proc=" << runtime->get_executing_processor(ctx);

  const AccessorRWdouble fa_acc(regions[0], task->regions[0].instance_fields[0]);
  const AccessorROdouble fa_in(regions[1], task->regions[1].instance_fields[0]);

  log_app.debug() << "&acc[" << args.bounds.lo << "] = " << (void *)(&fa_acc[args.bounds.lo]) << "\n";

  for(PointInRectIterator<3> pir(args.bounds); pir(); ++pir)
    fa_acc[*pir] = args.alpha_in * fa_in[*pir] + args.alpha_acc * fa_acc[*pir];
}

