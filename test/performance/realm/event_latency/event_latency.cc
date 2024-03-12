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

#include <time.h>

#include <realm.h>
#include <realm/timers.h>
#include <realm/cmdline.h>

using namespace Realm;

Logger log_app("app");

#define DEFAULT_DEPTH 1024 

// TASK IDs
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  THUNK_BUILDER  = Processor::TASK_ID_FIRST_AVAILABLE+1,
};

struct TopLevelArgs {
  int chain_depth;
};

struct ThunkBuilderArgs {
  int depth_remaining;
  UserEvent next_event;
  Barrier chain_done;
};

template <typename T>
class AssignRedop {
public:
  typedef T LHS;
  typedef T RHS;

  static void apply(T& lhs, T rhs) { lhs = rhs; }
};

class UserEventAssignRedop {
public:
  typedef UserEvent LHS;
  typedef UserEvent RHS;

  template <bool EXCL>
  static void apply(UserEvent& lhs, UserEvent rhs) { lhs = rhs; }

  static const UserEvent identity; // = UserEvent::NO_USER_EVENT;

  template <bool EXCL>
  static void fold(UserEvent& rhs1, UserEvent rhs2) { rhs1 = rhs2; }
};

/*static*/ const UserEvent UserEventAssignRedop::identity = UserEvent::NO_USER_EVENT;

enum {
  REDOP_ASSIGN = 66,
};

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
  const TopLevelArgs *targs = static_cast<const TopLevelArgs *>(args);

  // create the final event in the chain
  UserEvent final_event = UserEvent::create_user_event();

  // create a barrier that will be used to communicate the first event in
  //  the chain back to us
  Barrier chain_done = Barrier::create_barrier(1,
					       REDOP_ASSIGN,
					       &UserEvent::NO_EVENT,
					       sizeof(UserEvent));

  // construct chain
  log_app.print() << "initializing event latency experiment with a depth of "
		  << targs->chain_depth << " events...";
  double t1 = Clock::current_time();
  if(targs->chain_depth > 1) {
    ThunkBuilderArgs bargs;
    bargs.depth_remaining = targs->chain_depth - 1;
    bargs.next_event = final_event;
    bargs.chain_done = chain_done;
    Processor nextp = get_next_processor(p);
    nextp.spawn(THUNK_BUILDER, &bargs, sizeof(bargs));
  } else {
    // single-event chain
    chain_done.arrive(1, Event::NO_EVENT, &final_event, sizeof(UserEvent));
  }
  chain_done.wait();
  UserEvent first_event;
  bool ok = chain_done.get_result(&first_event, sizeof(UserEvent));
  assert(ok);
  double t2 = Clock::current_time();
  {
    double elapsed = t2 - t1;
    double per_task = elapsed / (targs->chain_depth + 1);
    log_app.print() << "chain construction: " << (1e6 * per_task) << " us/event";
  }

  // trigger first event and see how long it takes to get to the end
  double t3 = Clock::current_time();
  first_event.trigger();
  final_event.wait();
  double t4 = Clock::current_time();
  {
    double elapsed = t4 - t3;
    double per_task = elapsed / (targs->chain_depth + 1);
    log_app.print() << "chain trigger: " << (1e6 * per_task) << " us/event, " << elapsed << " s total";
  }
}

void thunk_builder(const void *args, size_t arglen, 
                   const void *userdata, size_t userlen, Processor p)
{
  const ThunkBuilderArgs *bargs = static_cast<const ThunkBuilderArgs *>(args);

  // create a user event and stick it on the front of the chain
  UserEvent my_event = UserEvent::create_user_event();
  bargs->next_event.trigger(my_event);

  if(bargs->depth_remaining > 1) {
    // continue chain on next processor
    ThunkBuilderArgs next_args;
    next_args.depth_remaining = bargs->depth_remaining - 1;
    next_args.next_event = my_event;
    next_args.chain_done = bargs->chain_done;
    Processor nextp = get_next_processor(p);
    nextp.spawn(THUNK_BUILDER, &next_args, sizeof(next_args));
  } else {
    // chain is done - signal main task
    bargs->chain_done.arrive(1, Event::NO_EVENT, &my_event, sizeof(UserEvent));
  }
}

int main(int argc, char **argv)
{
  Runtime r;

  bool ok = r.init(&argc, &argv);
  assert(ok);

  TopLevelArgs top_args;
  top_args.chain_depth = DEFAULT_DEPTH;

  CommandLineParser cp;
  cp.add_option_int("-d", top_args.chain_depth);
  ok = cp.parse_command_line(argc, (const char **)argv);

  r.register_task(TOP_LEVEL_TASK, top_level_task);
  r.register_task(THUNK_BUILDER, thunk_builder);

  r.register_reduction<UserEventAssignRedop>(REDOP_ASSIGN);

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
  Event e = r.collective_spawn(p, TOP_LEVEL_TASK, &top_args, sizeof(top_args));

  // request shutdown once that task is complete
  r.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  r.wait_for_shutdown();
  
  return 0;
}
