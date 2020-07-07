#include "realm.h"

using namespace Realm;

// execute a task on Processor::ACCEL_PROC processor

Logger log_app("app");

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  CHILD_TASK_ID_START
};

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
  CodeDescriptor child_task_desc(child_task);

  std::set<Event> finish_events;
  Event e = Processor::register_task_by_kind(Processor::ACCEL_PROC, true /*global*/,
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

    // only ACCEL_PROCs
    if(pp.kind() != Processor::ACCEL_PROC)
      continue;

    Event e2 = pp.spawn(func_id, &count, sizeof(count), e);

    finish_events.insert(e2);
  }

  func_id++;

  Event merged = Event::merge_events(finish_events);

  merged.wait();

  log_app.print() << "all done!";
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::ACCEL_PROC)
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
