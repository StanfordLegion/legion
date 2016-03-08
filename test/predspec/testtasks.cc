#include "testtasks.h"

SetEntry::Launcher::Launcher(LogicalRegion region, int idx, FieldID fid, int newval,
			     Predicate pred /*= Predicate::TRUE_PRED*/,
			     MappingTagID tag /*= 0*/)
{
  this->task_id = taskid;
  Args *args = new Args; // this is a memory leak for now
  args->idx = idx;
  args->newval = newval;
  this->argument = TaskArgument(args, sizeof(Args));
  this->predicate = pred;
  this->tag = tag;

  this->add_region_requirement(RegionRequirement(region, READ_WRITE, EXCLUSIVE, region)
			       .add_field(fid));
}

Future SetEntry::run(Runtime *runtime, Context ctx,
		     LogicalRegion region, int idx, FieldID fid, int newval,
		     Predicate pred /*= Predicate::TRUE_PRED*/,
		     MappingTagID tag /*= 0*/)
{
  Launcher l(region, idx, fid, newval, pred, tag);
  return runtime->execute_task(ctx, l);
}

/*static*/ TaskID SetEntry::taskid = -1;
void SetEntry::preregister_tasks(void)
{
  taskid = Runtime::generate_static_task_id();
  TaskVariantRegistrar tvr(taskid, "set_entry");
  tvr.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  tvr.set_leaf(true);
  Runtime::preregister_task_variant<cpu_task>(tvr, "set_entry");
}

void SetEntry::cpu_task(const Task *task,
			const std::vector<PhysicalRegion> &regions,
			Context ctx, HighLevelRuntime *runtime)
{
  const Args& args = *(const Args *)(task->args);
  RegionAccessor<AccessorType::Affine<1>, int> ra = regions[0].get_field_accessor(task->regions[0].instance_fields[0]).typeify<int>().convert<AccessorType::Affine<1> >();

  log_app.debug() << task->regions[0].region << "." << task->regions[0].instance_fields[0] << "[" << args.idx << "] = " << args.newval;
  ra[args.idx] = args.newval;
}
  
CheckEntry::Launcher::Launcher(LogicalRegion region, int idx, FieldID fid, int checkval,
			       Predicate pred /*= Predicate::TRUE_PRED*/,
			       MappingTagID tag /*= 0*/)
{
  this->task_id = taskid;
  Args *args = new Args; // this is a memory leak for now
  args->idx = idx;
  args->checkval = checkval;
  this->argument = TaskArgument(args, sizeof(Args));
  this->predicate = pred;
  this->tag = tag;

  this->add_region_requirement(RegionRequirement(region, READ_WRITE, EXCLUSIVE, region)
			       .add_field(fid));
}

Future CheckEntry::run(Runtime *runtime, Context ctx,
		       LogicalRegion region, int idx, FieldID fid, int checkval,
		       Predicate pred /*= Predicate::TRUE_PRED*/,
		       MappingTagID tag /*= 0*/)
{
  Launcher l(region, idx, fid, checkval, pred, tag);
  return runtime->execute_task(ctx, l);
}

/*static*/ TaskID CheckEntry::taskid = -1;
void CheckEntry::preregister_tasks(void)
{
  taskid = Runtime::generate_static_task_id();
  TaskVariantRegistrar tvr(taskid, "check_entry");
  tvr.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  tvr.set_leaf(true);
  Runtime::preregister_task_variant<bool, cpu_task>(tvr, "check_entry");
}

bool CheckEntry::cpu_task(const Task *task,
			  const std::vector<PhysicalRegion> &regions,
			  Context ctx, HighLevelRuntime *runtime)
{
  const Args& args = *(const Args *)(task->args);
  RegionAccessor<AccessorType::Affine<1>, int> ra = regions[0].get_field_accessor(task->regions[0].instance_fields[0]).typeify<int>().convert<AccessorType::Affine<1> >();

  int actval = ra[args.idx];
  if(actval == args.checkval) {
    log_app.debug() << task->regions[0].region << "." << task->regions[0].instance_fields[0] << "[" << args.idx << "] == " << args.checkval;
    return true;
  } else {
    log_app.error() << task->regions[0].region << "." << task->regions[0].instance_fields[0] << "[" << args.idx << "] == " << actval << " (exp: " << args.checkval << ")";
    return false;
  }
}
  
