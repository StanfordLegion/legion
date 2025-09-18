/* Copyright 2024 Stanford University
 * Copyright 2024 Los Alamos National Laboratory
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
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <set>
#include <time.h>

#include <realm.h>

using namespace Realm;

// TASK IDs
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  MAKE_LOCKS_TASK = Processor::TASK_ID_FIRST_AVAILABLE+1,
  RETURN_LOCKS_TASK = Processor::TASK_ID_FIRST_AVAILABLE+2,
  LAUNCH_FAIR_LOCK_TASK = Processor::TASK_ID_FIRST_AVAILABLE+3,
  LAUNCH_UNFAIR_LOCK_TASK = Processor::TASK_ID_FIRST_AVAILABLE+4,
  ADD_FINAL_EVENT_TASK = Processor::TASK_ID_FIRST_AVAILABLE+5,
  DUMMY_TASK = Processor::TASK_ID_FIRST_AVAILABLE+6,
};

struct InputArgs {
  int argc;
  char **argv;
};

struct FairStruct {
  Processor orig;
  Reservation lock;
  Event precondition;
  size_t depth;
};

// forward declaration
void fair_locks_task(const void *args, size_t arglen, 
                     const void *userdata, size_t userlen, Processor p);

InputArgs& get_input_args(void)
{
  static InputArgs args;
  return args;
}

std::set<Event>& get_final_events(void)
{
  static std::set<Event> final_events;
  return final_events;
}

std::set<Reservation>& get_lock_set(void)
{
  static std::set<Reservation> lock_set;
  return lock_set;
}

Processor get_next_processor(Processor cur)
{
  Machine machine = Machine::get_machine();
  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);
  for (std::set<Processor>::const_iterator it = all_procs.begin();
        it != all_procs.end(); it++)
  {
    if (*it == cur)
    {
      // Advance the iterator once to get the next, handle
      // the wrap around case too
      it++;
      if (it == all_procs.end())
      {
        return *(all_procs.begin());
      }
      else
      {
        return *it;
      }
    }
  }
  // Should always find one
  assert(false);
  return Processor::NO_PROC;
}

struct MakeLocksParams {
  realm_id_t proc;
  size_t locks_per_processor;
};

struct ReturnLocksParams {
  size_t num_locks;
  realm_id_t reservations[1];
};
struct LaunchUnfairParams {
  realm_id_t proc;
  realm_id_t start_event;
  size_t tasks_per_processor_per_lock;
  size_t lock_set_size;
  realm_id_t reservations[1];
};

void top_level_task(const void *args, size_t arglen, 
                    const void *userdata, size_t userlen, Processor p)
{
  bool fair = false;
  size_t locks_per_processor = 16;
  size_t tasks_per_processor_per_lock = 8;
  // Parse the input arguments
#define INT_ARG(argname, varname)                                                        \
  do {                                                                                   \
    if(!strcmp((argv)[i], argname)) {                                                    \
      varname = static_cast<decltype(varname)>(atoi((argv)[++i]));                       \
      continue;                                                                          \
    }                                                                                    \
  } while(0)

#define BOOL_ARG(argname, varname) do { \
        if(!strcmp((argv)[i], argname)) {		\
          varname = true;				\
          continue;					\
        } } while(0)
  {
    InputArgs &inputs = get_input_args();
    char **argv = inputs.argv;
    for (int i = 1; i < inputs.argc; i++)
    {
      INT_ARG("-lpp", locks_per_processor);
      INT_ARG("-tpppl",tasks_per_processor_per_lock);
      BOOL_ARG("-fair",fair);
    }
    assert(locks_per_processor > 0);
    assert(tasks_per_processor_per_lock > 0);
  }
#undef INT_ARG
#undef BOOL_ARG

  UserEvent start_event = UserEvent::create_user_event();

  std::set<Processor> all_procs;
  Machine::get_machine().get_all_processors(all_procs);
  // Send a request to each processor to make the given number of locks
  // Send a request to each processor to make the given number of locks
  {
    MakeLocksParams params;
    params.proc = p.id;
    params.locks_per_processor = locks_per_processor;
    for(Processor copy : all_procs) {
      Event wait_event = copy.spawn(MAKE_LOCKS_TASK, &params, sizeof(params));
      wait_event.wait();
    }
  }
  if (fair)
  {
    fprintf(stdout,
            "Running FAIR lock contention experiment with %zu locks per processor and "
            "%zu tasks per lock per processor\n",
            locks_per_processor, tasks_per_processor_per_lock);
    // For each lock in the lock set, stripe it through all the processors with dependences
    size_t lock_depth = tasks_per_processor_per_lock * all_procs.size();
    std::set<Reservation> &lock_set = get_lock_set();
    for (std::set<Reservation>::const_iterator it = lock_set.begin();
          it != lock_set.end(); it++)
    {
      FairStruct fair = { p, *it, start_event, lock_depth };
      // We can just call it locally here to start on our processor
      fair_locks_task(&fair,sizeof(FairStruct),0,0,p);
    }
  }
  else
  {
    fprintf(stdout,
            "Running UNFAIR lock contention experiment with %zu locks per processor and "
            "%zu tasks per lock per processor\n",
            locks_per_processor, tasks_per_processor_per_lock);
    std::set<Reservation> &lock_set = get_lock_set();
    // Package up all the locks and tell the processor how many tasks to register for each
    size_t buffer_size =
        sizeof(LaunchUnfairParams) + ((lock_set.size() - 1) * sizeof(Reservation));
    LaunchUnfairParams *params = (LaunchUnfairParams *)malloc(buffer_size);
    params->proc = p.id;
    params->start_event = start_event.id;
    params->tasks_per_processor_per_lock = tasks_per_processor_per_lock;
    params->lock_set_size = 0;
    for(Reservation reservation : lock_set) {
      params->reservations[params->lock_set_size++] = reservation.id;
    }
    // Send the message to all the processors
    for(Processor target : all_procs) {
      Event wait_event = target.spawn(LAUNCH_UNFAIR_LOCK_TASK, params, buffer_size);
      wait_event.wait();
    }
    free(params);
  }

  Event final_event = Event::merge_events(get_final_events());
  assert(final_event.exists());

  // Now we're ready to start our simulation
  fprintf(stdout,"Running experiment...\n");
  {
    double start, stop; 
    start = Realm::Clock::current_time_in_microseconds();
    // Trigger the start event
    start_event.trigger();
    // Wait for the final event
    final_event.wait();
    stop = Realm::Clock::current_time_in_microseconds();

    double latency = stop - start;
    fprintf(stdout,"Total time: %7.3f us\n", latency);
    double grants_per_sec = locks_per_processor * tasks_per_processor_per_lock * all_procs.size() / latency;
    fprintf(stdout,"Reservation Grants/s (in Thousands): %7.3f\n", grants_per_sec);
  }
  
  fprintf(stdout,"Cleaning up...\n");
}

void make_locks_task(const void *args, size_t arglen, 
                     const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(MakeLocksParams));
  const MakeLocksParams *param = reinterpret_cast<const MakeLocksParams *>(args);
  Processor orig;
  orig.id = param->proc;
  size_t num_locks = param->locks_per_processor;

  size_t buffer_size = sizeof(ReturnLocksParams) + (num_locks - 1) * sizeof(Reservation);
  ReturnLocksParams *rparams = (ReturnLocksParams *)malloc(buffer_size);
  rparams->num_locks = num_locks;
  for(size_t idx = 0; idx < num_locks; idx++) {
    Reservation res = Reservation::create_reservation();
    rparams->reservations[idx] = res.id;
  }
  Event wait_event = orig.spawn(RETURN_LOCKS_TASK, rparams, buffer_size);
  wait_event.wait();
  free(rparams);
}

void return_locks_task(const void *args, size_t arglen, 
                       const void *userdata, size_t userlen, Processor p)
{
  assert(arglen >= sizeof(ReturnLocksParams));
  const ReturnLocksParams *params = reinterpret_cast<const ReturnLocksParams *>(args);
  size_t num_locks = params->num_locks;
  std::set<Reservation> &lockset = get_lock_set();
  for(size_t idx = 0; idx < num_locks; idx++) {
    Reservation remote;
    remote.id = params->reservations[idx];
    lockset.insert(remote);
  }
}

void fair_locks_task(const void *args, size_t arglen, 
                     const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(FairStruct));
  FairStruct fair = *((FairStruct*)args);
  if (fair.depth == 0)
  {
    // Sent the precondition back to the original processor
    Event wait_event = fair.orig.spawn(ADD_FINAL_EVENT_TASK,&(fair.precondition),sizeof(Event));
    wait_event.wait();
  }
  else
  {
    // Chain the lock acquistion, task call, lock release
    Event lock_event = fair.lock.acquire(0,true,fair.precondition);
    Event task_event = p.spawn(DUMMY_TASK,NULL,0,lock_event);
    fair.lock.release(task_event);
    FairStruct next_struct = { fair.orig, fair.lock, task_event, fair.depth-1 };
    Processor next_proc = get_next_processor(p);
    Event wait_event = next_proc.spawn(LAUNCH_FAIR_LOCK_TASK,&next_struct,sizeof(FairStruct));
    wait_event.wait();
  }
}

void unfair_locks_task(const void *args, size_t arglen, 
                       const void *userdata, size_t userlen, Processor p)
{
  assert(arglen >= sizeof(LaunchUnfairParams));
  const LaunchUnfairParams *params = reinterpret_cast<const LaunchUnfairParams *>(args);
  Processor orig;
  orig.id = params->proc;
  Event precondition;
  precondition.id = params->start_event;
  const size_t tasks_per_processor_per_lock = params->tasks_per_processor_per_lock;
  const size_t num_locks = params->lock_set_size;
  std::set<Reservation> lock_set;
  for (unsigned idx = 0; idx < num_locks; idx++)
  {
    Reservation lock;
    lock.id = params->reservations[idx];
    lock_set.insert(lock);
  }
  std::set<Event> wait_for_events;
  for(Reservation lock : lock_set) {
    for(size_t idx = 0; idx < tasks_per_processor_per_lock; idx++) {
      Event lock_event = lock.acquire(0,true,precondition);
      Event task_event = p.spawn(DUMMY_TASK,NULL,0,lock_event);
      lock.release(task_event);
      wait_for_events.insert(task_event);
    }
  }
  // Merge all the wait for events together and send back the result
  Event final_event = Event::merge_events(wait_for_events);
  Event wait_event = orig.spawn(ADD_FINAL_EVENT_TASK,&final_event,sizeof(Event));
  wait_event.wait();
}

void add_final_event(const void *args, size_t arglen, 
                     const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(Event));
  Event result = *((Event*)args);
  get_final_events().insert(result);
}

void dummy_task(const void *args, size_t arglen, 
                const void *userdata, size_t userlen, Processor p)
{
  // Do nothing
}

int main(int argc, char **argv)
{
  Runtime r;

  bool ok = r.init(&argc, &argv);
  assert(ok);

  r.register_task(TOP_LEVEL_TASK, top_level_task);
  r.register_task(MAKE_LOCKS_TASK, make_locks_task);
  r.register_task(RETURN_LOCKS_TASK, return_locks_task);
  r.register_task(LAUNCH_FAIR_LOCK_TASK, fair_locks_task);
  r.register_task(LAUNCH_UNFAIR_LOCK_TASK, unfair_locks_task);
  r.register_task(ADD_FINAL_EVENT_TASK, add_final_event);
  r.register_task(DUMMY_TASK, dummy_task);

  // Set the input args
  get_input_args().argv = argv;
  get_input_args().argc = argc;

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
  Event e = r.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // request shutdown once that task is complete
  r.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  r.wait_for_shutdown();
  
  return 0;
}

