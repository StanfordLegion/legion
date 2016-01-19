#include "realm/realm.h"

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
};

void child_task(const void *args, size_t arglen, 
		const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "child task on " << p << ": arglen=" << arglen << ", userlen=" << userlen;
}

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
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

  printf("all done!\n");

  Runtime::get_runtime().shutdown();
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  // Start the machine running
  // Control never returns from this call
  // Note we only run the top level task on one processor
  // You can also run the top level task on all processors or one processor per node
  rt.run(TOP_LEVEL_TASK, Runtime::ONE_TASK_ONLY);

  //rt.shutdown();
  return 0;
}
