#include "realm.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <cmath>

#include <time.h>

#include "osdep.h"

using namespace Realm;

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  SWITCH_TEST_TASK,
  SLEEP_TEST_TASK,
  PRIORITY_MASTER_TASK,
  PRIORITY_CHILD_TASK,
};

// we're going to use alarm() as a watchdog to detect deadlocks
void sigalrm_handler(int sig)
{
  fprintf(stderr, "HELP!  Alarm triggered - likely deadlock!\n");
  exit(1);
}

// each child needs to know how high to count, which reservation to wait
//  on and which to use to signal the next child
struct SwitchTestArgs {
  int iterations;
  bool release_first;
  Reservation acquire_me;
  Reservation release_me;
};

void switch_task(const void *args, size_t arglen, 
		 const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(SwitchTestArgs));
  const SwitchTestArgs& c_args = *(const SwitchTestArgs *)args;

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
  printf("ending switch task on processor " IDFMT "\n", p.id);
#endif
}

struct SleepTestArgs {
  int sleep_useconds;
};

void sleep_task(const void *args, size_t arglen, 
		const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(SleepTestArgs));
  const SleepTestArgs& c_args = *(const SleepTestArgs *)args;

#ifdef DEBUG_CHILDREN
  printf("starting sleep task on processor " IDFMT "\n", p.id);
#endif
  
  usleep(c_args.sleep_useconds);

#ifdef DEBUG_CHILDREN
  printf("ending sleep task on processor " IDFMT "\n", p.id);
#endif
}

struct PriorityTestArgs {
  int *counter;
  int exp_val1, exp_val2;
  int new_priority;
  Event wait_on;
  Barrier barrier;
};

void priority_child_task(const void *args, size_t arglen, 
			 const void *userdata, size_t userlen, Processor p)
{
  const PriorityTestArgs& targs = *static_cast<const PriorityTestArgs *>(args);

  int act_val1 = __sync_fetch_and_add(targs.counter, 1);
  assert(act_val1 == targs.exp_val1);

  if(targs.barrier.exists())
    targs.barrier.arrive();

  if(targs.new_priority >= 0)
    Processor::set_current_task_priority(targs.new_priority);

  targs.wait_on.wait();

  int act_val2 = __sync_fetch_and_add(targs.counter, 1);
  assert(act_val2 == targs.exp_val2);
}

void priority_master_task(const void *args, size_t arglen, 
			  const void *userdata, size_t userlen, Processor p)
{
  int counter = 55;

  // We will create 5 workers that each try to increment the count twice.
  //  The first three are created before the master sleeps, and arrive at
  //  the barrier before going to sleep.  The master then wakes up, creates
  //  the last two, makes the first 3 ready, and sleeps until everybody's
  //  done.  Here is the expected order of increments:
  //
  // M:                 58                      66
  // W1: 1->0     56                         65
  // W2: 0->2        57    59
  // W3: 2->1  55                   62
  // W4: 1                             63 64
  // W5: 2                    60 61

  UserEvent event1 = UserEvent::create_user_event();
  UserEvent event2 = UserEvent::create_user_event();
  UserEvent event3 = UserEvent::create_user_event();
  Barrier barrier = Barrier::create_barrier(3);

  std::vector<Event> finish_events;

  PriorityTestArgs targs;
  targs.counter = &counter;
  targs.barrier = barrier;
  {
    targs.exp_val1 = 56;
    targs.exp_val2 = 65;
    targs.wait_on = event1;
    targs.new_priority = 0;
    Event e = p.spawn(PRIORITY_CHILD_TASK, &targs, sizeof(targs),
		      Event::NO_EVENT, 1);
    finish_events.push_back(e);
  }

  {
    targs.exp_val1 = 57;
    targs.exp_val2 = 59;
    targs.wait_on = event2;
    targs.new_priority = -1;
    Event e = p.spawn(PRIORITY_CHILD_TASK, &targs, sizeof(targs),
		      Event::NO_EVENT, 0);
    finish_events.push_back(e);
  }

  {
    targs.exp_val1 = 55;
    targs.exp_val2 = 62;
    targs.wait_on = event3;
    targs.new_priority = 1;
    Event e = p.spawn(PRIORITY_CHILD_TASK, &targs, sizeof(targs),
		      Event::NO_EVENT, 2);
    finish_events.push_back(e);
  }

  barrier.wait();

  int act_val1 = __sync_fetch_and_add(&counter, 1);
  assert(act_val1 == 58);

  finish_events[1].set_operation_priority(2);

  event1.trigger();
  event2.trigger();
  event3.trigger();

  targs.barrier = Barrier::NO_BARRIER;
  targs.wait_on = Event::NO_EVENT;
  targs.new_priority = -1;

  {
    targs.exp_val1 = 63;
    targs.exp_val2 = 64;
    Event e = p.spawn(PRIORITY_CHILD_TASK, &targs, sizeof(targs),
		      Event::NO_EVENT, 1);
    finish_events.push_back(e);
  }

  {
    targs.exp_val1 = 60;
    targs.exp_val2 = 61;
    Event e = p.spawn(PRIORITY_CHILD_TASK, &targs, sizeof(targs),
		      Event::NO_EVENT, 2);
    finish_events.push_back(e);
  }

  Event::merge_events(finish_events).wait();

  int act_val2 = __sync_fetch_and_add(&counter, 1);
  assert(act_val2 == 66);
} 

static int num_children = 4;
static int num_iterations = 100000;
static int timeout_seconds = 10;
static int sleep_useconds = 500000;
static int concurrent_io = 1;

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  int errors = 0;

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

      // first, the switching test
      if(num_iterations > 0) {
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
	  SwitchTestArgs c_args;
	  c_args.iterations = num_iterations;
	  c_args.release_first = (i == (num_children - 1));
	  c_args.acquire_me = rsrvs[i];
	  c_args.release_me = rsrvs[(i + 1) % num_children];

	  finish_events.insert(pp.spawn(SWITCH_TEST_TASK, &c_args, sizeof(c_args)));
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
	printf("switch: proc " IDFMT " (kind=%d) finished: elapsed=%5.2fs time/switch=%6.0fns\n",
               pp.id, k, elapsed, ns_per_switch);
      }

      // now the sleep (i.e. kernel-level switching, if possible) test
      if(sleep_useconds > 0) {
        double exp_time = 1e-6 * sleep_useconds;
	if(k != Processor::IO_PROC)
	  exp_time *= num_children;  // no overlapping of tasks
	else
	  exp_time *= (num_children + concurrent_io - 1) / concurrent_io;

        // set the watchdog timeout before we do anything that could get stuck
        alarm((int)ceil(1e-6 * sleep_useconds * num_children) * 2);

        // create the child tasks
        std::set<Event> finish_events;
	for(int i = 0; i < num_children; i++) {
	  SleepTestArgs c_args;
	  c_args.sleep_useconds = sleep_useconds;

	  finish_events.insert(pp.spawn(SLEEP_TEST_TASK, &c_args, sizeof(c_args)));
        }

        // now merge them and see how long it takes them to complete
	double t_start = Clock::current_time();
	Event e = Event::merge_events(finish_events);
	e.wait();
	double t_end = Clock::current_time();

	// turn off the watchdog timer
	alarm(0);

	double elapsed = t_end - t_start;
	printf("sleep: proc " IDFMT " (kind=%d) finished: elapsed=%5.2fs expected=%5.2fs\n",
               pp.id, k, elapsed, exp_time);
	if(elapsed < (0.75 * exp_time)) {
	  printf("TOO FAST!\n");
	  errors++;
	}
#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
        // macOS ARM seems slower for some reason...
	if(elapsed > (1.75 * exp_time)) {
#else
	if(elapsed > (1.25 * exp_time)) {
#endif
	  printf("TOO SLOW!\n");
	  errors++;
	}
      }

      // test task prioritization on this processor kind
      {
	Event e = pp.spawn(PRIORITY_MASTER_TASK, 0, 0);
	e.wait();
      }
    }
  }

  if(errors > 0) {
    printf("Exiting with errors\n");
    exit(1);
  }

  printf("all done!\n");
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

    if(!strcmp(argv[i], "-s")) {
      sleep_useconds = atoi(argv[++i]);
      continue;
    }

    // peek at Realm configuration here...
    if(!strcmp(argv[i], "-ll:concurrent_io")) {
      concurrent_io = atoi(argv[++i]);
      continue;
    }
  }

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(SWITCH_TEST_TASK, switch_task);
  rt.register_task(SLEEP_TEST_TASK, sleep_task);
  rt.register_task(PRIORITY_MASTER_TASK, priority_master_task);
  rt.register_task(PRIORITY_CHILD_TASK, priority_child_task);

  signal(SIGALRM, sigalrm_handler);

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
