#include "realm.h"
#ifdef REALM_USE_LLVM
#include "realm/llvmjit/llvmjit.h"
#endif
#ifdef REALM_USE_PYTHON
#include "realm/python/python_source.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <cmath>
#ifdef REALM_USE_LIBDL
#include <dlfcn.h>
#endif

#include <time.h>

#include "osdep.h"

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
#if REALM_LLVM_VERSION >= 37
"  %1 = load i32, i32* %a, align 4\n"
"  %2 = add i32 %1, 57\n"
"  %3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str, i32 0, i32 0), i32 %2, i64 %b)\n"
#else
"  %1 = load i32* %a, align 4\n"
"  %2 = add i32 %1, 57\n"
"  %3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([30 x i8]* @.str, i32 0, i32 0), i32 %2, i64 %b)\n"
#endif
"  ret void\n"
"}\n";
#endif

void child_task(const void *args, size_t arglen, 
		const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "child task on " << p << ": arglen=" << arglen << ", userlen=" << userlen;
}

#ifdef REALM_USE_LIBDL
// helper for looking up symbols
template <typename T>
void lookup_symbol(const char *name, T& addr)
{
  void *res = dlsym(RTLD_DEFAULT, name);
  if(!res) {
    log_app.fatal() << "symbol lookup error: cannot find '" << name << "'";
    abort();
  }
  addr = reinterpret_cast<T>(res);
}
#endif

#ifdef REALM_USE_PYTHON
void python_cpp_task(const void *args, size_t arglen,
		     const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "python cpp task on " << p << ": arglen=" << arglen << ", userlen=" << userlen;

#ifdef REALM_USE_LIBDL
  // try to talk to the python interpreter - use dlsym to find symbols so we
  //  don't have a compile-time dependency on python
  typedef void *PyGILState_STATE;
  PyGILState_STATE (*PyGILState_Ensure)();
  void (*PyGILState_Release)(PyGILState_STATE);
  int (*PyRun_SimpleString)(const char *);

  lookup_symbol("PyGILState_Ensure", PyGILState_Ensure);
  lookup_symbol("PyGILState_Release", PyGILState_Release);
  lookup_symbol("PyRun_SimpleString", PyRun_SimpleString);

  PyGILState_STATE state = PyGILState_Ensure();

  int ret = PyRun_SimpleString("print(\"inside python interpreter from cpp task!\")");
  if(ret != 0) {
    log_app.fatal() << "PyRun_SimpleString returned " << ret;
    abort();
  }

  PyGILState_Release(state);
#endif
}
#endif

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "top task running on " << p;

  Machine machine = Machine::get_machine();
  Processor::TaskFuncID func_id = CHILD_TASK_ID_START;
 
  // first test - register a task individually on each processor and run it
  {
    std::set<Event> finish_events;

    int count = 0;

    std::set<Processor> all_processors;
    machine.get_all_processors(all_processors);
    for(std::set<Processor>::const_iterator it = all_processors.begin();
	it != all_processors.end();
	it++) {
      Processor pp = (*it);

      CodeDescriptor *task_desc = 0;
      switch(pp.kind()) {
      case Processor::LOC_PROC:
      case Processor::UTIL_PROC:
      case Processor::IO_PROC:
	{
	  task_desc = new CodeDescriptor(child_task);
	  break;
	}

#ifdef REALM_USE_PYTHON
      case Processor::PY_PROC:
	{
	  task_desc = new CodeDescriptor(Realm::Type::from_cpp_type<Processor::TaskFuncPtr>());
	  task_desc->add_implementation(new PythonSourceImplementation("taskreg_helper",
								       "task1"));
	  break;
	}
#endif

      default:
	/* do nothing */
	break;
      }
      if(!task_desc) {
	log_app.warning() << "no task variant available for processor " << p << " (kind " << pp.kind() << ")";
	continue;
      }

      Event e = pp.register_task(func_id, *task_desc,
				 ProfilingRequestSet(),
				 &pp, sizeof(pp));

      delete task_desc;

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

    Event e = Processor::register_task_by_kind(Processor::LOC_PROC,
#ifdef _MSC_VER
    // no portable task registration for windows yet
                                               false /*!global*/,
#else
                                               true /*global*/,
#endif
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

#ifdef REALM_USE_PYTHON
  // c++ task on python processor
  {
    Machine::ProcessorQuery pq(machine);
    pq.only_kind(Processor::PY_PROC);
    for(Machine::ProcessorQuery::iterator it = pq.begin(); it != pq.end(); ++it) {
      Processor pp = *it;

      Event e = pp.register_task(func_id,
				 CodeDescriptor(python_cpp_task),
				 ProfilingRequestSet(),
				 &pp, sizeof(pp));

      Event e2 = pp.spawn(func_id, 0, 0, e);

      e2.wait();
    }
  }
#endif

  log_app.print() << "all done!";
}

int main(int argc, char **argv)
{
#ifdef REALM_USE_PYTHON
  // do this before any threads are spawned
  setenv("PYTHONPATH", ".", true /*overwrite*/);
#endif

  Runtime rt;

  rt.init(&argc, &argv);

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC)
    .first();
  assert(p.exists());

  Event e1 = Processor::register_task_by_kind(p.kind(),
					      false /*!global*/,
					      TOP_LEVEL_TASK,
					      CodeDescriptor(top_level_task),
					      ProfilingRequestSet());

  // collective launch of a single task - everybody gets the same finish event
  Event e2 = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0, e1);

  // request shutdown once that task is complete
  rt.shutdown(e2);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();
  
  return 0;
}
