#include "realm.h"
#include "realm/cmdline.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>

using namespace Realm;

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
};

// The goal of this test is to confirm that several different Realm
// synchronization promitives are sufficient to act as synchronization
// primitives under the C++ memory model. The most valuable version of
// this test will be run with thread-sanitizer to confirm these relationships

int x, y;

struct SyncPrimitives {
  UserEvent test_wait;
  UserEvent test_triggered;
  Barrier test_bar;
};

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  const SyncPrimitives *primitives = (const SyncPrimitives*)args;
  // Make sure that task launching synchronizes
  assert(x == 11);
  // Make sure that user-event triggers synchronzie with waits
  x = 42;
  primitives->test_wait.trigger();
  // Make sure that user-event triggers synchronize with has_triggered
  primitives->test_triggered.subscribe();
  while (!primitives->test_triggered.has_triggered()) { }
  assert(x == 1729);
  // Make sure that barrier arrive and wait synchronize
  x = 6174;
  primitives->test_bar.arrive();
  primitives->test_bar.wait();
  assert(y == 23);
  // Make sure that task completion events synchronize
  y = 2520;
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  SyncPrimitives primitives;
  primitives.test_wait = UserEvent::create_user_event();
  primitives.test_triggered = UserEvent::create_user_event();
  primitives.test_bar = Barrier::create_barrier(2);

  // Make sure that task launching synchronizes
  x = 11;
  // Collective launch a single task on each CPU processor of each node, that
  // just means this test will be done once on each node
  Event e = rt.collective_spawn_by_kind(Processor::LOC_PROC, TOP_LEVEL_TASK,
                      &primitives, sizeof(primitives), true/*one per node*/);
  // request shutdown once that task is complete
  rt.shutdown(e);

  // Make sure that user-event triggers synchronize with waits
  primitives.test_wait.external_wait();
  assert(x == 42);
  // Make sure that user-events synchronize with has_triggered
  x = 1729;
  primitives.test_triggered.trigger();
  // Make sure that barrier arrive and wait synchronize
  y = 23;
  primitives.test_bar.arrive();
  primitives.test_bar.wait();
  assert(x == 6174);

  // Make sure that task completion events synchronize
  e.external_wait();
  assert(y == 2520);
  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();
  
  return 0;
}

