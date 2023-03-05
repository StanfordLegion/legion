/* Copyright 2023 Stanford University
 * Copyright 2023 Los Alamos National Laboratory 
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
  int depth;
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

void top_level_task(const void *args, size_t arglen, 
                    const void *userdata, size_t userlen, Processor p)
{
  bool fair = false;
  int locks_per_processor = 16;
  int tasks_per_processor_per_lock = 8;
  // Parse the input arguments
#define INT_ARG(argname, varname) do { \
        if(!strcmp((argv)[i], argname)) {		\
          varname = atoi((argv)[++i]);		\
          continue;					\
        } } while(0)

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
  {
    size_t buffer_size = sizeof(Processor) + sizeof(int);
    void *buffer = malloc(buffer_size);
    char *ptr = (char*)buffer;
    *((Processor*)ptr) = p;
    ptr += sizeof(Processor);
    *((int*)ptr) = locks_per_processor;
    for (std::set<Processor>::const_iterator it = all_procs.begin();
          it != all_procs.end(); it++)
    {
      Processor copy = *it;
      Event wait_event = copy.spawn(MAKE_LOCKS_TASK,buffer,buffer_size);
      wait_event.wait();
    }
    free(buffer);
  }
  if (fair)
  {
    fprintf(stdout,"Running FAIR lock contention experiment with %d locks per processor and %d tasks per lock per processor\n",
            locks_per_processor, tasks_per_processor_per_lock);
    // For each lock in the lock set, stripe it through all the processors with dependences
    int lock_depth = tasks_per_processor_per_lock * all_procs.size();
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
    fprintf(stdout,"Running UNFAIR lock contention experiment with %d locks per processor and %d tasks per lock per processor\n",
            locks_per_processor, tasks_per_processor_per_lock);
    std::set<Reservation> &lock_set = get_lock_set();
    // Package up all the locks and tell the processor how many tasks to register for each
    size_t buffer_size = sizeof(Processor) + sizeof(Event) + sizeof(int) + sizeof(size_t) + (lock_set.size() * sizeof(Reservation));
    void *buffer = malloc(buffer_size);
    char *ptr = (char*)buffer;
    *((Processor*)ptr) = p;
    ptr += sizeof(Processor);
    *((Event*)ptr) = start_event;
    ptr += sizeof(Event);
    *((int*)ptr) = tasks_per_processor_per_lock;
    ptr += sizeof(int);
    *((size_t*)ptr) = lock_set.size();
    ptr += sizeof(size_t);
    for (std::set<Reservation>::const_iterator it = lock_set.begin();
          it != lock_set.end(); it++)
    {
      Reservation lock = *it;
      *((Reservation*)ptr) = lock;
      ptr += sizeof(Reservation);
    }
    // Send the message to all the processors
    for (std::set<Processor>::const_iterator it = all_procs.begin();
          it != all_procs.end(); it++)
    {
      Processor target = *it;
      Event wait_event = target.spawn(LAUNCH_UNFAIR_LOCK_TASK,buffer,buffer_size);
      wait_event.wait();
    }
    free(buffer);
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
  assert(arglen == (sizeof(Processor) + sizeof(int)));
  char *ptr = (char*)args;
  Processor orig = *((Processor*)ptr);
  ptr += sizeof(Processor);
  int num_locks = *((int*)ptr);

  size_t buffer_size = sizeof(int) + num_locks*sizeof(Reservation);
  void * buffer = malloc(buffer_size);
  ptr = (char*)buffer;
  *((int*)ptr) = num_locks;
  ptr += sizeof(int);

  for (int idx = 0; idx < num_locks; idx++)
  {
    Realm::Reservation r = Realm::Reservation::create_reservation();
    memcpy(ptr, &r, sizeof(Realm::Reservation));
    ptr += sizeof(Realm::Reservation);
  }
  Event wait_event = orig.spawn(RETURN_LOCKS_TASK,buffer,buffer_size);
  free(buffer);
  wait_event.wait();
}

void return_locks_task(const void *args, size_t arglen, 
                       const void *userdata, size_t userlen, Processor p)
{
  char *ptr = (char*)args;
  int num_locks = *((int*)ptr);
  ptr += sizeof(int);
  std::set<Reservation> &lockset = get_lock_set();
  for (int idx = 0; idx < num_locks; idx++)
  {
    Reservation remote = *((Reservation*)ptr);
    ptr += sizeof(Reservation);
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
  char *ptr = (char*)args;
  Processor orig = *((Processor*)ptr);
  ptr += sizeof(Processor);
  Event precondition = *((Event*)ptr);
  ptr += sizeof(Event);
  int tasks_per_processor_per_lock = *((int*)ptr);
  ptr += sizeof(int);
  size_t num_locks = *((size_t*)ptr);
  ptr += sizeof(size_t);
  std::set<Reservation> lock_set;
  for (unsigned idx = 0; idx < num_locks; idx++)
  {
    Reservation lock = *((Reservation*)ptr);
    ptr += sizeof(Reservation);
    lock_set.insert(lock);
  }
  std::set<Event> wait_for_events;
  for (std::set<Reservation>::const_iterator it = lock_set.begin();
        it != lock_set.end(); it++)
  {
    Reservation lock = *it;
    for (int idx = 0; idx < tasks_per_processor_per_lock; idx++)
    {
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

