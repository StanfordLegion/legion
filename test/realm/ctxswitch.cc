#include "realm/realm.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <csignal>

#include <time.h>
#include <unistd.h>

using namespace Realm;

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  CHILD_TASK     = Processor::TASK_ID_FIRST_AVAILABLE+1,
};

// each child needs to know how high to count, which reservation to wait
//  on and which to use to signal the next child
struct ChildArgs {
  int iterations;
  bool release_first;
  Reservation acquire_me;
  Reservation release_me;
};

// we're going to use alarm() as a watchdog to detect deadlocks
void sigalrm_handler(int sig)
{
  fprintf(stderr, "HELP!  Alarm triggered - likely deadlock!\n");
  exit(1);
}

void child_task(const void *args, size_t arglen, Processor p)
{
  assert(arglen == sizeof(ChildArgs));
  const ChildArgs& c_args = *(const ChildArgs *)args;

#ifdef DEBUG_CHILDREN
  printf("starting child task on processor " IDFMT "\n", p.id);
#endif

  for(int count = 0; count < c_args.iterations; count++) {
    if(c_args.release_first)
      c_args.release_me.release();

    Event e = c_args.acquire_me.acquire();
#ifdef DEBUG_CHILDREN
    printf("acquire %d of " IDFMT " -> " IDFMT "/%d\n",
	   count, c_args.acquire_me.id, e.id, e.gen);
#endif
    e.wait();

    if(!c_args.release_first)
      c_args.release_me.release();
  }

#ifdef DEBUG_CHILDREN
  printf("ending task on processor " IDFMT "\n", p.id);
#endif
}

static int num_children = 4;
static int num_iterations = 100000;
static int timeout_seconds = 10;

void top_level_task(const void *args, size_t arglen, Processor p)
{
  printf("Realm context switching test - %d children, %d iterations, %ds timeout\n",
	 num_children, num_iterations, timeout_seconds);

  // iterate over all the CPU kinds we can find - test each kind once
  Machine machine = Machine::get_machine();
  std::set<Processor::Kind> seen;
  {
    std::set<Processor> all_processors;
    machine.get_all_processors(all_processors);
    for(std::set<Processor>::const_iterator it = all_processors.begin();
	it != all_processors.end();
	it++) {
      Processor pp = (*it);
      Processor::Kind k = pp.kind();
      if(seen.count(k) > 0) continue;

      printf("testing processor " IDFMT " (kind=%d)\n", pp.id, k);
      seen.insert(k);

      // set the watchdog timeout before we do anything that could get stuck
      alarm(timeout_seconds);

      // we're going to need a reservation per child - all start out acquired
      std::vector<Reservation> rsrvs(num_children);
      for(int i = 0; i < num_children; i++) {
	rsrvs[i] = Reservation::create_reservation();
	Event e = rsrvs[i].acquire();
	// uncontended, so this should be immediate
	assert(!e.exists());
      }

      // create the child tasks
      std::set<Event> finish_events;
      for(int i = 0; i < num_children; i++) {
	ChildArgs c_args;
	c_args.iterations = num_iterations;
	c_args.release_first = (i == (num_children - 1));
	c_args.acquire_me = rsrvs[i];
	c_args.release_me = rsrvs[(i + 1) % num_children];

	finish_events.insert(pp.spawn(CHILD_TASK, &c_args, sizeof(c_args)));
      }

      // now merge them and see how long it takes them to complete
      double t_start = Clock::current_time();
      Event e = Event::merge_events(finish_events);
      e.wait();
      double t_end = Clock::current_time();

      // turn off the watchdog timer
      alarm(0);

      double elapsed = t_end - t_start;
      double ns_per_switch = 1e9 * elapsed / num_iterations / num_children;
      printf("proc " IDFMT " (kind=%d) finished: elapsed=%5.2fs time/switch=%6.0fns\n",
	     pp.id, k, elapsed, ns_per_switch);
    }
  }

  printf("all done!\n");

  sleep(1);
  // shutdown the runtime - this should implicitly wait for the report task to run
  {
    std::set<Processor> all_procs;
    machine.get_all_processors(all_procs);
    for (std::set<Processor>::const_iterator it = all_procs.begin();
          it != all_procs.end(); it++)
    {
      // Damn you C++ and your broken const qualifiers
      Processor handle_copy = *it;
      // Send the kill pill
      handle_copy.spawn(0,NULL,0);
    }
  }
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-c")) {
      num_children = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-i")) {
      num_iterations = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-t")) {
      timeout_seconds = atoi(argv[++i]);
      continue;
    }
  }

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(CHILD_TASK, child_task);

  signal(SIGALRM, sigalrm_handler);

  // Start the machine running
  // Control never returns from this call
  // Note we only run the top level task on one processor
  // You can also run the top level task on all processors or one processor per node
  rt.run(TOP_LEVEL_TASK, Runtime::ONE_TASK_ONLY);

  rt.shutdown();
  return 0;
}
