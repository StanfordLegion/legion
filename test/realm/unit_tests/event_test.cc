#include "realm/event_impl.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

struct EventTestCase {};

class EventTest : public ::testing::TestWithParam<EventTestCase> {};

class DeferredOperation : public EventWaiter {
public:
  void defer(Event wait_on) {}
  virtual void event_triggered(bool poisoned, TimeLimit work_until) { triggered = true; }
  virtual void print(std::ostream &os) const {}
  virtual Event get_finish_event(void) const { return Event::NO_EVENT; }
  bool triggered = false;
};

TEST_P(EventTest, GetCurrentEvent)
{
  GenEventImpl event;
  event.init(ID::make_event(0, 0, 0), 0);
  EXPECT_EQ(ID(event.current_event()).event_generation(),
            ID::make_event(0, 0, 1).event_generation());
}

TEST_P(EventTest, AddRemoveWaiterSameGen)
{
  DeferredOperation waiter;
  GenEventImpl event;
  event.init(ID::make_event(0, 0, 0), 0);

  event.add_waiter(1, &waiter);
  event.remove_waiter(1, &waiter);

  EXPECT_TRUE(event.current_local_waiters.empty());
}

TEST_P(EventTest, ProcessUpdateNonOwner)
{
  DeferredOperation waiter_one;
  DeferredOperation waiter_two;

  GenEventImpl event;
  event.init(ID::make_event(0, 0, 0), 1);

  event.process_update(1, 0, 0, TimeLimit::responsive());

  event.add_waiter(1, &waiter_one);
  event.add_waiter(1, &waiter_two);

  EXPECT_TRUE(waiter_one.triggered);
  EXPECT_TRUE(waiter_two.triggered);
}

TEST_P(EventTest, AddRemoveWaiterDifferentGens)
{
  DeferredOperation waiter;
  GenEventImpl event;
  event.init(ID::make_event(0, 0, 0), 0);

  EXPECT_TRUE(event.add_waiter(1, &waiter));
  EXPECT_FALSE(event.remove_waiter(2, &waiter));

  EXPECT_FALSE(event.current_local_waiters.empty());
}

TEST_P(EventTest, Subscribe)
{
  GenEventImpl event;
  event.init(ID::make_event(0, 0, 0), 1);

  // TODO: needs active messages
  // event->subscribe(2);
}

TEST_P(EventTest, BasicPoisonedTest)
{
  GenEventImpl event;
  event.init(ID::make_event(0, 0, 0), 0);

  EXPECT_FALSE(event.is_generation_poisoned(1));
}

TEST_P(EventTest, BasicHasTriggerdTest)
{
  GenEventImpl event;
  event.init(ID::make_event(0, 0, 0), 0);

  bool poisoned = false;
  EXPECT_FALSE(event.has_triggered(1, poisoned));
  EXPECT_FALSE(poisoned);
}

TEST_P(EventTest, BasicTriggerTest)
{
  GenEventImpl event;
  event.init(ID::make_event(0, 0, 0), 0);

  // TODO: free_event hitting get_runtime
  // event->trigger(1, 0, 0, TimeLimit::responsive());
}

const static EventTestCase kEventTestCases[] = {
    EventTestCase{},
};

INSTANTIATE_TEST_SUITE_P(Foo, EventTest, testing::ValuesIn(kEventTestCases));
