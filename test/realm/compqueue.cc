/* Copyright 2023 Stanford University, NVIDIA Corporation
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

// Realm test for completion queues

#include <realm.h>
#include <realm/cmdline.h>

#include "philox.h"

#include "osdep.h"

using namespace Realm;

Logger log_app("app");

typedef Philox_2x32<> PRNG;

enum {
  // PRNG keys
  PKEY_PROCS,
  PKEY_DELAY,
};

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  SPAWNER_TASK,
  WORK_TASK,
};

struct TestConfig {
  size_t tasks_per_proc;
  size_t max_in_flight;
  size_t max_to_pop;
  long long min_exec_time, max_exec_time;
  int watchdog_timeout;
};

struct SpawnerArgs {
  TestConfig config;
  CompletionQueue cq;
  int index;
};

struct WorkTaskArgs {
  long long exec_time;
};

void worker_task(const void *args, size_t arglen,
		 const void *userdata, size_t userlen, Processor p)
{
  const WorkTaskArgs& w_args = *reinterpret_cast<const WorkTaskArgs *>(args);

  // we model doing work by just sleeping for the requested amount of time
  usleep(w_args.exec_time);
}

void reap_events(CompletionQueue cq, size_t& in_flight,
		 size_t max_in_flight, size_t max_to_pop)
{
  while(in_flight > max_in_flight) {
    // try to pop an event
    size_t to_pop = std::min(max_to_pop, in_flight);
    std::vector<Event> popped(to_pop, Event::NO_EVENT);
    size_t count = cq.pop_events(&popped[0], to_pop);
    assert(count <= to_pop);
    if(count > 0) {
      popped.resize(count);
      log_app.info() << "popped " << count << " events: " << PrettyVector<Event>(popped);
      in_flight -= count;
    } else {
	// instead of hammering cq, ask for a nonempty event and wait on it
	//  before trying again
      Event nonempty = cq.get_nonempty_event();
      nonempty.wait();
    }
  }
}

void spawner_task(const void *args, size_t arglen,
		  const void *userdata, size_t userlen, Processor p)
{
  const SpawnerArgs& s_args = *reinterpret_cast<const SpawnerArgs *>(args);

  CompletionQueue cq = s_args.cq;

  // get the list of processors we'll spawn work tasks on
  Machine::ProcessorQuery pq(Machine::get_machine());
  pq.only_kind(p.kind());
  std::vector<Processor> procs(pq.begin(), pq.end());

  size_t in_flight = 0;
  for(size_t i = 0; i < s_args.config.tasks_per_proc; i++) {
    // choose a random processor
    Processor target = procs[PRNG::rand_int(PKEY_PROCS, s_args.index, i, procs.size())];
    // and choose a random delay
    WorkTaskArgs w_args;
    w_args.exec_time = s_args.config.min_exec_time;
    if(s_args.config.max_exec_time > s_args.config.min_exec_time)
      w_args.exec_time += PRNG::rand_int(PKEY_DELAY, s_args.index, i,
					 (s_args.config.max_exec_time -
					  s_args.config.min_exec_time));
    // work tasks get a higher priority to get ahead of the reaping task
    Event e = target.spawn(WORK_TASK, &w_args, sizeof(w_args),
			   ProfilingRequestSet(), Event::NO_EVENT, 1);
    log_app.info() << "added event: " << e;
    cq.add_event(e);
    in_flight++;

    // make sure we limit the number in flight
    reap_events(s_args.cq, in_flight,
		s_args.config.max_in_flight - 1,
		s_args.config.max_to_pop);
  }

  // reap events until we have none in flight
  reap_events(s_args.cq, in_flight,
	      0 /*max_in_flight*/,
	      s_args.config.max_to_pop);
}
  

void test_cq(const TestConfig& config,
	     const std::vector<Processor>& procs, size_t cq_size)
{
  CompletionQueue cq = CompletionQueue::create_completion_queue(cq_size);

  // set a timeout to catch hangs
  if(config.watchdog_timeout > 0)
    alarm(config.watchdog_timeout);

  // launch a spawner on each processor
  std::vector<Event> events;
  for(size_t i = 0; i < procs.size(); i++) {
    SpawnerArgs s_args;
    s_args.config = config;
    s_args.cq = cq;
    s_args.index = i;
    Event e = procs[i].spawn(SPAWNER_TASK, &s_args, sizeof(s_args));
    events.push_back(e);
  }
  // wait on all spawners to finish
  Event::merge_events(events).wait();
  
  // all tasks done - turn off alarm
  if(config.watchdog_timeout > 0)
    alarm(0);
  
  cq.destroy();
}

void top_level_task(const void *args, size_t arglen,
		    const void *userdata, size_t userlen, Processor p)
{
  const TestConfig& config = *reinterpret_cast<const TestConfig *>(args);

  Machine::ProcessorQuery pq(Machine::get_machine());
  pq.only_kind(p.kind());
  std::vector<Processor> procs(pq.begin(), pq.end());

  log_app.print() << "compqueue test: " << procs.size() << " procs, "
		  << config.tasks_per_proc << " tasks/proc, "
		  << config.max_in_flight << " in flight, "
		  << config.max_to_pop << " max pop";

  // create a completion queue - it should not ever have to hold more
  //  than #procs * max_in_flight events
  size_t cq_size = procs.size() * config.max_in_flight;
  test_cq(config, procs, cq_size);

  // also test the dynamic completion queue case
  test_cq(config, procs, 0);

  log_app.info() << "completed successfully";
  
  Runtime::get_runtime().shutdown(Event::NO_EVENT, 0 /*success*/);
}

// we're going to use alarm() as a watchdog to detect deadlocks
void sigalrm_handler(int sig)
{
  log_app.fatal() << "HELP!  Alarm triggered - likely deadlock!";
  abort();
}

int main(int argc, const char **argv)
{
  Runtime rt;

  rt.init(&argc, (char ***)&argv);

  TestConfig config;
  config.tasks_per_proc = 1000;
  config.max_in_flight = 20;
  config.max_to_pop = 10;
  config.min_exec_time = 10;  // 10us
  config.max_exec_time = 10000; // 10ms
  // above parameters results in ~1000 * 10ms = 10s of work per processor
  // if we take more than ~5x that, something's wrong
  config.watchdog_timeout = 60; // 60 seconds

  CommandLineParser clp;
  clp.add_option_int("-t", config.tasks_per_proc);
  clp.add_option_int("-m", config.max_in_flight);
  clp.add_option_int("-p", config.max_to_pop);
  clp.add_option_int("-min", config.min_exec_time);
  clp.add_option_int("-max", config.max_exec_time);
  clp.add_option_int("-timeout", config.watchdog_timeout);

  bool ok = clp.parse_command_line(argc, argv);
  assert(ok);

  // try to use a cpu proc, but if that doesn't exist, take whatever we can get
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC)
    .first();
  if(!p.exists())
    p = Machine::ProcessorQuery(Machine::get_machine()).first();
  assert(p.exists());

  Processor::register_task_by_kind(p.kind(), false /*!global*/,
				   TOP_LEVEL_TASK,
				   CodeDescriptor(top_level_task),
				   ProfilingRequestSet()).external_wait();
  Processor::register_task_by_kind(p.kind(), false /*!global*/,
				   SPAWNER_TASK,
				   CodeDescriptor(spawner_task),
				   ProfilingRequestSet()).external_wait();
  Processor::register_task_by_kind(p.kind(), false /*!global*/,
				   WORK_TASK,
				   CodeDescriptor(worker_task),
				   ProfilingRequestSet()).external_wait();

  signal(SIGALRM, sigalrm_handler);

  // collective launch of a single top level task
  rt.collective_spawn(p, TOP_LEVEL_TASK, &config, sizeof(config));

  // now sleep this thread until that shutdown actually happens
  int ret = rt.wait_for_shutdown();
  
  return ret;
}
