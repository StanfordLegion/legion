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
#include <math.h>
#include <queue>
#include "legion.h"
#include "default_mapper.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

LegionRuntime::Logger::Category log_app("app");

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  RUN_TEST_TASK_ID,
};

enum FieldIDs {
  FID_X = 10000,
  FID_Y,
  FID_Z,
};

#define FID(n) (FID_X + (n))

enum {
  REDOPID_BOOL_OR = 77,
};

template <typename T>
inline T logical_or(T lhs, T rhs) { return lhs || rhs; }

template <typename T, T (*BINOP)(T,T), int REDOPID>
struct BinaryReductionOp {
  static const ReductionOpID redop_id = REDOPID;

  typedef T LHS;
  typedef T RHS;
  static const T identity = T();

  template<bool EXCL>
  static void apply(LHS &lhs, const RHS& rhs)
  {
    lhs = (*BINOP)(lhs, rhs);
  }

  template<bool EXCL>
  static void fold(RHS& rhs1, const RHS& rhs2)
  {
    rhs1 = (*BINOP)(rhs1, rhs2);
  }

  static void register_redop(void)
  {
    Runtime::register_reduction_op<BinaryReductionOp<T,BINOP,REDOPID> >(REDOPID);
  }
};

typedef BinaryReductionOp<bool, logical_or<bool>, REDOPID_BOOL_OR> BooleanOrRedop;

class SetEntry {
public:
  struct Args {
    int idx;
    int newval;
  };

  class Launcher : public TaskLauncher {
  public:
    Launcher(LogicalRegion region, int idx, FieldID fid, int newval,
	     Predicate pred = Predicate::TRUE_PRED);
  };

  static TaskID taskid;

  static Future run(Runtime *runtime, Context ctx,
		    LogicalRegion region, int idx, FieldID fid, int newval,
		    Predicate pred = Predicate::TRUE_PRED);

  static void preregister_tasks(void);

  static void cpu_task(const Task *task,
		       const std::vector<PhysicalRegion> &regions,
		       Context ctx, HighLevelRuntime *runtime);
};

SetEntry::Launcher::Launcher(LogicalRegion region, int idx, FieldID fid, int newval,
			     Predicate pred /*= Predicate::TRUE_PRED*/)
{
  this->task_id = taskid;
  Args *args = new Args; // this is a memory leak for now
  args->idx = idx;
  args->newval = newval;
  this->argument = TaskArgument(args, sizeof(Args));
  this->predicate = pred;

  this->add_region_requirement(RegionRequirement(region, READ_WRITE, EXCLUSIVE, region)
			       .add_field(fid));
}

Future SetEntry::run(Runtime *runtime, Context ctx,
		     LogicalRegion region, int idx, FieldID fid, int newval,
		     Predicate pred /*= Predicate::TRUE_PRED*/)
{
  Launcher l(region, idx, fid, newval, pred);
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
  
class CheckEntry {
public:
  struct Args {
    int idx;
    int checkval;
  };

  class Launcher : public TaskLauncher {
  public:
    Launcher(LogicalRegion region, int idx, FieldID fid, int checkval,
	     Predicate pred = Predicate::TRUE_PRED);
  };

  static TaskID taskid;

  static Future run(Runtime *runtime, Context ctx,
		    LogicalRegion region, int idx, FieldID fid, int checkval,
		    Predicate pred = Predicate::TRUE_PRED);

  static void preregister_tasks(void);

  static bool cpu_task(const Task *task,
		       const std::vector<PhysicalRegion> &regions,
		       Context ctx, HighLevelRuntime *runtime);
};

CheckEntry::Launcher::Launcher(LogicalRegion region, int idx, FieldID fid, int checkval,
			     Predicate pred /*= Predicate::TRUE_PRED*/)
{
  this->task_id = taskid;
  Args *args = new Args; // this is a memory leak for now
  args->idx = idx;
  args->checkval = checkval;
  this->argument = TaskArgument(args, sizeof(Args));
  this->predicate = pred;

  this->add_region_requirement(RegionRequirement(region, READ_WRITE, EXCLUSIVE, region)
			       .add_field(fid));
}

Future CheckEntry::run(Runtime *runtime, Context ctx,
		     LogicalRegion region, int idx, FieldID fid, int checkval,
		     Predicate pred /*= Predicate::TRUE_PRED*/)
{
  Launcher l(region, idx, fid, checkval, pred);
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
  

struct Globals {
  IndexSpace ispace;
  FieldSpace fspace;
  std::vector<LogicalRegion> regions;
};

enum TestResult {
  RESULT_UNKNOWN,
  RESULT_PASS,
  RESULT_FAIL,
  RESULT_SKIP,
};

typedef TestResult (*TestEntryFn)(Runtime *runtime, Context ctx, const Globals& g);  

class DelayedPredicate {
public:
  DelayedPredicate(Runtime *_runtime, Context _ctx);
  operator Predicate(void) const;
  DelayedPredicate& operator=(bool newval);
  DelayedPredicate& operator=(Future f);
protected:
  Runtime *runtime;
  Context ctx;
  DynamicCollective dc;
  Predicate p;
};

DelayedPredicate::DelayedPredicate(Runtime *_runtime, Context _ctx)
  : runtime(_runtime), ctx(_ctx)
{
  bool b_false = false;
  dc = runtime->create_dynamic_collective(ctx, 1, REDOPID_BOOL_OR, &b_false, sizeof(b_false));
  DynamicCollective dcnext = runtime->advance_dynamic_collective(ctx, dc);
  Future f = runtime->get_dynamic_collective_result(ctx, dcnext);
  p = runtime->create_predicate(ctx, f);
}

DelayedPredicate::operator Predicate(void) const
{
  return p;
}

DelayedPredicate& DelayedPredicate::operator=(bool newval)
{
  runtime->arrive_dynamic_collective(ctx, dc, &newval, sizeof(newval));
  return *this;
}

DelayedPredicate& DelayedPredicate::operator=(Future f)
{
  runtime->defer_dynamic_collective_arrival(ctx, dc, f);
  return *this;
}

template <bool PREDVAL>
TestResult test_pred_const(Runtime *runtime, Context ctx, const Globals& g)
{
  SetEntry::run(runtime, ctx, g.regions[0], 0, FID(0), 45);
  SetEntry::run(runtime, ctx, g.regions[0], 0, FID(0), 46,
		(PREDVAL ? Predicate::TRUE_PRED : Predicate::FALSE_PRED));
  Future f = CheckEntry::run(runtime, ctx, g.regions[0], 0, FID(0), 
			     (PREDVAL ? 46 : 45));
  bool b = f.get_result<bool>();
  return b ? RESULT_PASS : RESULT_FAIL;
}

template <bool PREDVAL, bool EARLY>
TestResult test_pred_simple(Runtime *runtime, Context ctx, const Globals& g)
{
  SetEntry::run(runtime, ctx, g.regions[0], 0, FID(0), 45);
  DelayedPredicate dp(runtime, ctx);
  if(EARLY)
    dp = PREDVAL;
  SetEntry::run(runtime, ctx, g.regions[0], 0, FID(0), 46, dp);
  Future f = CheckEntry::run(runtime, ctx, g.regions[0], 0, FID(0), 
			     (PREDVAL ? 46 : 45));
  if(!EARLY)
    dp = PREDVAL;
  bool b = f.get_result<bool>();
  return b ? RESULT_PASS : RESULT_FAIL;
}

struct RunSingleTestArgs {
  char name[40];
  TestEntryFn entry_fn;
  int num_regions;
  int num_elements;
  int num_fields;
};

void run_test_task(const Task *task,
		   const std::vector<PhysicalRegion> &regions,
		   Context ctx, HighLevelRuntime *runtime)
{
  const RunSingleTestArgs& args = *(const RunSingleTestArgs *)(task->args);
  Globals g;

  g.ispace = runtime->create_index_space(ctx, Domain::from_rect<1>(Rect<1>(0, args.num_elements-1)));
  g.fspace = runtime->create_field_space(ctx);
  {
    FieldAllocator fa = runtime->create_field_allocator(ctx, g.fspace);
    for(int i = 0; i < args.num_fields; i++)
      fa.allocate_field(sizeof(int), FID(i));
  }
  for(int i = 0; i < args.num_regions; i++)
    g.regions.push_back(runtime->create_logical_region(ctx, g.ispace, g.fspace));

  TestResult res = (args.entry_fn)(runtime, ctx, g);

  switch(res) {
  case RESULT_PASS: 
    {
      log_app.print() << "PASS: " << args.name;
      break;
    }
  case RESULT_FAIL:
    {
      log_app.error() << "FAIL: " << args.name;
      break;
    }
  case RESULT_SKIP:
    {
      log_app.info() << "SKIP: " << args.name;
      break;
    }
  default:
    {
      log_app.error() << "UNKNOWN(" << res << "): " << args.name;
      break;
    }
  }

  for(int i = 0; i < args.num_regions; i++)
    runtime->destroy_logical_region(ctx, g.regions[i]);
  runtime->destroy_field_space(ctx, g.fspace);
  runtime->destroy_index_space(ctx, g.ispace);
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  RunSingleTestArgs cargs;
  //strcpy(cargs.name, "foo");
  //cargs.entry_fn = test_foo;
  cargs.num_elements = 1;
  cargs.num_fields = 2;
  cargs.num_regions = 1;
  TaskLauncher launcher(RUN_TEST_TASK_ID, TaskArgument(&cargs, sizeof(cargs)));
  {
    strcpy(cargs.name, "pred_const_false");
    cargs.entry_fn = test_pred_const<false>;
    Future f = runtime->execute_task(ctx, launcher);
    f.get_void_result();
  }
  {
    strcpy(cargs.name, "pred_const_true");
    cargs.entry_fn = test_pred_const<true>;
    Future f = runtime->execute_task(ctx, launcher);
    f.get_void_result();
  }
  {
    strcpy(cargs.name, "pred_simple_false");
    cargs.entry_fn = test_pred_simple<false, false>;
    Future f = runtime->execute_task(ctx, launcher);
    f.get_void_result();
  }
  {
    strcpy(cargs.name, "pred_simple_true");
    cargs.entry_fn = test_pred_simple<true, false>;
    Future f = runtime->execute_task(ctx, launcher);
    f.get_void_result();
  }
  {
    strcpy(cargs.name, "pred_simple_false_early");
    cargs.entry_fn = test_pred_simple<false, true>;
    Future f = runtime->execute_task(ctx, launcher);
    f.get_void_result();
  }
  {
    strcpy(cargs.name, "pred_simple_true_early");
    cargs.entry_fn = test_pred_simple<true, true>;
    Future f = runtime->execute_task(ctx, launcher);
    f.get_void_result();
  }
}

static void update_mappers(Machine machine, HighLevelRuntime *rt,
                           const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    //rt->replace_default_mapper(new GCMapper(machine, rt, *it), *it);
  }
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
    TaskVariantRegistrar tvr(RUN_TEST_TASK_ID, "run_test_task");
    tvr.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    tvr.set_leaf(false);
    Runtime::preregister_task_variant<run_test_task>(tvr, "run_test_task");
  }

  SetEntry::preregister_tasks();
  CheckEntry::preregister_tasks();

  BooleanOrRedop::register_redop();

  Runtime::set_registration_callback(update_mappers);

  return Runtime::start(argc, argv);
}
