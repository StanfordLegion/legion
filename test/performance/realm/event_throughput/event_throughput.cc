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

#include <time.h>

#include <realm.h>

using namespace Realm;

#define DEFAULT_LEVELS 32 
#define DEFAULT_TRACKS 32 
#define DEFAULT_FANOUT 16 

// TASK IDs
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0, 
  LEVEL_BUILDER  = Processor::TASK_ID_FIRST_AVAILABLE+1,
  SET_REMOTE_EVENT = Processor::TASK_ID_FIRST_AVAILABLE+2,
  DUMMY_TASK = Processor::TASK_ID_FIRST_AVAILABLE+3,
};

struct InputArgs {
  int argc;
  char **argv;
};

InputArgs& get_input_args(void)
{
  static InputArgs args;
  return args;
}

typedef std::set<Event> EventSet;

EventSet& get_event_set(void)
{
  static EventSet event_set;
  return event_set;
}

void send_level_commands(int fanout, Processor local, const EventSet &send_events, 
			 const std::set<Processor> &all_procs, bool first)
{
  assert(!send_events.empty());
  size_t buffer_size = sizeof(Processor) + sizeof(bool) + sizeof(size_t) + (send_events.size() * sizeof(Event));
  void * buffer = malloc(buffer_size);
  char *ptr = (char*)buffer;
  *((Processor*)ptr) = local;
  ptr += sizeof(Processor);
  *((bool*)ptr) = first;
  ptr += sizeof(bool);
  size_t num_events = send_events.size();
  *((size_t*)ptr) = num_events;
  ptr += sizeof(size_t);
  for (EventSet::const_iterator it = send_events.begin(); 
       it != send_events.end(); it++)
    {
      *((Event*)ptr) = *it;
      ptr += sizeof(Event);
    }
  std::set<Processor>::const_iterator it = all_procs.begin();
  for (int i = 0; i < fanout; i++)
    {
      Processor copy = *it;
      // This was: Event wait_for = copy.spawn.builder(buffer,buffer_size);
      // not sure how that ever worked -dpx
      Event wait_for = copy.spawn(LEVEL_BUILDER, buffer, buffer_size);
      // Update the iterator while we're waiting
      it++;
      if (it == all_procs.end()) // if we reach the end, reset
	it = all_procs.begin();
      // Wait for it to finish so we know when we're done
      wait_for.wait();
    }
  free(buffer);
  assert(int(get_event_set().size()) == fanout);
}

void construct_track(int levels, int fanout, Processor local, Event precondition, EventSet &wait_for, const std::set<Processor> &all_procs)
{
  EventSet send_events;   
  EventSet &receive_events = get_event_set();
  receive_events.clear();
  // For the first level there is only one event that has to be sent
  send_events.insert(precondition);
#ifdef DEBUG_PRINT
  fprintf(stdout,"Building first level\n");
  fflush(stdout);
#endif
  send_level_commands(fanout, local, send_events, all_procs, true/*first*/);
  for (int i = 1; i < levels; i++)
  {
#ifdef DEBUG_PRINT
    usleep(1000);
    fprintf(stdout,"Building level %d\n",i);
    fflush(stdout);
#endif
    // Copy the send events from the receive events
    send_events = receive_events;
    receive_events.clear();
    send_level_commands(fanout, local, send_events, all_procs, false/*first*/);
  }
  // Put all the receive events from the last level into the wait for set
  wait_for.insert(receive_events.begin(),receive_events.end());
  receive_events.clear();
}

void top_level_task(const void *args, size_t arglen, 
                    const void *userdata, size_t userlen, Processor p)
{
  int levels = DEFAULT_LEVELS;
  int tracks = DEFAULT_TRACKS;
  int fanout = DEFAULT_FANOUT;
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
      INT_ARG("-l", levels);
      INT_ARG("-t", tracks);
      INT_ARG("-f", fanout);
    }
    assert(levels > 0);
    assert(tracks > 0);
    assert(fanout > 0);
  }
#undef INT_ARG
#undef BOOL_ARG
  
  // Make a user event that will be the trigger
  UserEvent start_event = UserEvent::create_user_event();
  std::set<Event> wait_for_finish;
 
  long total_events;
  long total_triggers;
  // Initialize a bunch of experiments, each track does an all-to-all event communication for each level
  fprintf(stdout,"Initializing event throughput experiment with %d tracks and %d levels per track with fanout %d...\n",tracks,levels,fanout);
  fflush(stdout);
  {
    Realm::Machine machine = Realm::Machine::get_machine();
    std::set<Processor> all_procs;
    machine.get_all_processors(all_procs);
    for (int t = 0; t < tracks; t++)
    {
      construct_track(levels, fanout, p, start_event, wait_for_finish, all_procs);
    }
    assert(int(wait_for_finish.size()) == (fanout * tracks));
    // Compute the total number of events to be triggered
    total_events = fanout * levels * tracks;
    total_triggers = total_events * fanout; // each event sends a trigger to every processor
  }
  // Merge all the finish events together into one finish event
#ifdef DEBUG_PRINT
  fprintf(stdout,"Merging finish events: ");
  for (std::set<Event>::const_iterator it = wait_for_finish.begin();
        it != wait_for_finish.end(); it++)
  {
    fprintf(stdout,"%x ",(*it).id);
  }
  fprintf(stdout,"\n");
#endif
  Event finish_event = Event::merge_events(wait_for_finish);

  // Now we're ready to start our simulation
  fprintf(stdout,"Running experiment...\n");
  {
    double start, stop; 
    start = Realm::Clock::current_time_in_microseconds();
    // Trigger the start event
    start_event.trigger();
    // Wait for the final event
    finish_event.wait();
    stop = Realm::Clock::current_time_in_microseconds();

    double latency = (stop - start) * 0.001;
    fprintf(stdout,"Total time: %7.3f ms\n", latency);
    fprintf(stdout,"Events triggered: %ld\n", total_events);
    fprintf(stdout,"Events throughput: %7.3f Thousands/s\n",(double(total_events)/latency));
    fprintf(stdout,"Triggers performed: %ld\n", total_triggers);
    fprintf(stdout,"Triggers throughput: %7.3f Thousands/s\n",(double(total_triggers)/latency));
  }

  fprintf(stdout,"Cleaning up...\n");
}

void level_builder(const void *args, size_t arglen, 
                   const void *userdata, size_t userlen, Processor p)
{
  // Unpack everything
  std::set<Event> wait_for_events;
  const char* ptr = (const char*)args;
  Processor orig = *((Processor*)ptr);
  ptr += sizeof(Processor);
  bool first = *((bool*)ptr);
  ptr += sizeof(bool);
  size_t total_events = *((size_t*)ptr);
  ptr += sizeof(size_t);
#ifdef DEBUG_PRINT
  fprintf(stdout,"Merging events ");
#endif
  for (unsigned i = 0; i < total_events; i++)
  {
    Event wait_event = *((Event*)ptr);
    ptr += sizeof(Event);
    wait_for_events.insert(wait_event);
#ifdef DEBUG_PRINT
    fprintf(stdout,"%x ",wait_event.id);
#endif
  }
#ifdef DEBUG_PRINT
  fprintf(stdout,"\n");
#endif
  // Merge all the wait for events together
  Event finish_event = Event::NO_EVENT;
  if (first)
  {
    Event launch_event = Event::merge_events(wait_for_events);
    // Launch the task on this processor
    finish_event = p.spawn(DUMMY_TASK,NULL,0,launch_event);
  }
  else
  {
    finish_event = Event::merge_events(wait_for_events);
  }
  assert(finish_event.exists());
  // Send back the event for this processor
#ifdef DEBUG_PRINT
  fprintf(stdout,"Processor %x reporting event %x\n",p.id,finish_event.id);
  fflush(stdout);
#endif
  {
    size_t buffer_size = sizeof(Event);
    void * buffer = malloc(buffer_size);
    char * ptr = (char*)buffer;
    *((Event*)ptr) = finish_event;
    // Send it back, wait for it to finish
    Event report_event = orig.spawn(SET_REMOTE_EVENT,buffer,buffer_size);
    free(buffer);
    report_event.wait();
  }
}

void set_remote_event(const void *args, size_t arglen, 
                      const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == (sizeof(Event))); 
  const char* ptr = (const char*)args;
  Event result = *((Event*)ptr);
  EventSet &event_set = get_event_set();
  event_set.insert(result);
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
  r.register_task(LEVEL_BUILDER, level_builder);
  r.register_task(SET_REMOTE_EVENT, set_remote_event);
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
