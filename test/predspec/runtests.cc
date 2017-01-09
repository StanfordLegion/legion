/* Copyright 2017 Stanford University
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

#include "runtests.h"
#include "testtasks.h"
#include "testmapper.h"

LegionRuntime::Logger::Category log_app("app");

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  RUN_TEST_TASK_ID,
};

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


////////////////////////////////////////////////////////////////////////
//
// class TestInformation
//

namespace {
  TestInformation *test_list_head = 0;
};

TestInformation::TestInformation(const char *_name, TestEntryFn _entry_fn,
				 int _num_regions, int _num_elements, int _num_fields)
  : entry_fn(_entry_fn)
  , num_regions(_num_regions)
  , num_elements(_num_elements)
  , num_fields(_num_fields)
{
  strcpy(name, _name);
  do {
    next = test_list_head;
  } while(!__sync_bool_compare_and_swap(&test_list_head, next, this));
}


////////////////////////////////////////////////////////////////////////
//
// class DelayedPredicate
//

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


////////////////////////////////////////////////////////////////////////
//
// task definitions
//

void run_test_task(const Task *task,
		   const std::vector<PhysicalRegion> &regions,
		   Context ctx, HighLevelRuntime *runtime)
{
  const TestInformation& args = *(const TestInformation *)(task->args);
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
  int argc = HighLevelRuntime::get_input_args().argc;
  const char **argv = (const char **)HighLevelRuntime::get_input_args().argv;

  const TestInformation *test_info = test_list_head;
  while(test_info) {
    bool run_this_test = true;
    for(int i = 1; i < argc; i++)
      if(argv[i][0] == '-') {
	if(!strcmp(argv[i], "-only")) {
	  run_this_test = false; // default is now false
	  continue;
	}

	if(!strcmp(argv[i]+1, test_info->name)) {
	  run_this_test = false;
	  break;
	}
      } else {
	if(!strcmp(argv[i], test_info->name)) {
	  run_this_test = true;
	  break;
	}
      }
  
    if(run_this_test) {
      log_app.info() << "starting test: " << test_info->name;
      TaskLauncher launcher(RUN_TEST_TASK_ID, TaskArgument(test_info, sizeof(TestInformation)));
      Future f = runtime->execute_task(ctx, launcher);
      f.get_void_result();
    }

    test_info = test_info->next;
  }
}

static void update_mappers(Machine machine, HighLevelRuntime *rt,
                           const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(new TestMapper(machine, rt, *it), *it);
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
