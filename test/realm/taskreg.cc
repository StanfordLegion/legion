#include "realm/realm.h"
#ifdef REALM_USE_LLVM
#include "realm/llvmjit/llvmjit.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <csignal>
#include <cmath>

#include <time.h>
#include <unistd.h>

using namespace Realm;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  CHILD_TASK_ID_START,
#ifdef REALM_USE_LLVM
  LLVM_TASK_ID = 100,
#endif
};

#ifdef REALM_USE_LLVM
const char llvmir[] = 
"@.str = private unnamed_addr constant [30 x i8] c\"hello from LLVM JIT! %d %lld\\0A\\00\", align 1\n"
"declare i32 @printf(i8*, ...)\n"
"define void @foo(i32* %a, i64 %b, i32* %c, i64 %d, i64 %e) {\n"
"  %1 = load i32* %a, align 4\n"
"  %2 = add i32 %1, 57\n"
"  %3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([30 x i8]* @.str, i32 0, i32 0), i32 %2, i64 %b)\n"
"  ret void\n"
"}\n";
#endif

void child_task(const void *args, size_t arglen, 
		const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "child task on " << p << ": arglen=" << arglen << ", userlen=" << userlen;
}

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "top task running on " << p;

  Machine machine = Machine::get_machine();
  Processor::TaskFuncID func_id = CHILD_TASK_ID_START;
 
  // first test - register a task individually on each processor and run it
  {
    std::set<Event> finish_events;

    CodeDescriptor child_task_desc(child_task);
    int count = 0;

    std::set<Processor> all_processors;
    machine.get_all_processors(all_processors);
    for(std::set<Processor>::const_iterator it = all_processors.begin();
	it != all_processors.end();
	it++) {
      Processor pp = (*it);

      Event e = pp.register_task(func_id, child_task_desc,
				 ProfilingRequestSet(),
				 &pp, sizeof(pp));

      Event e2 = pp.spawn(func_id, &count, sizeof(count), e);

      finish_events.insert(e2);

      func_id++;
    }

    Event merged = Event::merge_events(finish_events);

    merged.wait();
  }

  // second test - register a task on all LOC_PROCs
  {
    std::set<Event> finish_events;

    CodeDescriptor child_task_desc(child_task);

    Event e = Processor::register_task_by_kind(Processor::LOC_PROC, true /*global*/,
					       func_id,
					       child_task_desc,
					       ProfilingRequestSet());

    int count = 0;

    std::set<Processor> all_processors;
    machine.get_all_processors(all_processors);
    for(std::set<Processor>::const_iterator it = all_processors.begin();
	it != all_processors.end();
	it++) {
      Processor pp = (*it);

      // only LOC_PROCs
      if(pp.kind() != Processor::LOC_PROC)
	continue;

      Event e2 = pp.spawn(func_id, &count, sizeof(count), e);

      finish_events.insert(e2);
    }

    func_id++;

    Event merged = Event::merge_events(finish_events);

    merged.wait();
  }

#ifdef REALM_USE_LLVM
  // third test - LLVM (if available)
  {
    CodeDescriptor llvm_task_desc(TypeConv::from_cpp_type<Processor::TaskFuncPtr>());
    llvm_task_desc.add_implementation(new LLVMIRImplementation(llvmir, sizeof(llvmir),
							       "foo"));

    Event e = Processor::register_task_by_kind(Processor::LOC_PROC, true /*global*/,
					       LLVM_TASK_ID,
					       llvm_task_desc,
					       ProfilingRequestSet());

    int count = 0;
    std::set<Event> finish_events;
    std::set<Processor> all_processors;
    machine.get_all_processors(all_processors);
    for(std::set<Processor>::const_iterator it = all_processors.begin();
	it != all_processors.end();
	it++) {
      Processor pp = (*it);

      // only LOC_PROCs
      if(pp.kind() != Processor::LOC_PROC)
	continue;

      Event e2 = pp.spawn(LLVM_TASK_ID, &count, sizeof(count), e);

      count++;

      finish_events.insert(e2);
    }

    Event merged = Event::merge_events(finish_events);

    merged.wait();
  }
#endif

  log_app.print() << "all done!";
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  // select a processor to run the top level task on
  Processor p = Processor::NO_PROC;
  {
    std::set<Processor> all_procs;
    Machine::get_machine().get_all_processors(all_procs);
    for(std::set<Processor>::const_iterator it = all_procs.begin();
	it != all_procs.end();
	it++)
      if(it->kind() == Processor::LOC_PROC) {
	p = *it;
	break;
      }
  }
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // request shutdown once that task is complete
  rt.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();
  
  return 0;
}
