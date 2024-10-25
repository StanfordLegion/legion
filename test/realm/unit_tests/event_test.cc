#include "realm/event_impl.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

struct EventTestCase {};

class EventTest : public ::testing::TestWithParam<EventTestCase> {};

class DeferredOperation : public EventWaiter {
public:
  void defer(Event wait_on) {}
  virtual void event_triggered(bool poisoned, TimeLimit work_until) {}
  virtual void print(std::ostream &os) const {}
  virtual Event get_finish_event(void) const { return Event::NO_EVENT; }
};

TEST_P(EventTest, AddRemoveWaiterSameGen)
{
  DeferredOperation waiter;
  GenEventImpl *event = new GenEventImpl();
  event->init(ID::make_event(0, 0, 0), 0);

  event->add_waiter(1, &waiter);
  event->remove_waiter(1, &waiter);

  EXPECT_TRUE(event->current_local_waiters.empty());
}

TEST_P(EventTest, AddRemoveWaiterDifferentGens)
{
  DeferredOperation waiter;
  GenEventImpl *event = new GenEventImpl();
  event->init(ID::make_event(0, 0, 0), 0);

  EXPECT_TRUE(event->add_waiter(1, &waiter));
  EXPECT_FALSE(event->remove_waiter(2, &waiter));

  EXPECT_FALSE(event->current_local_waiters.empty());
}

TEST_P(EventTest, Subscribe)
{
  GenEventImpl *event = new GenEventImpl();
  event->init(ID::make_event(0, 0, 0), 1);

  // TODO: needs active messages
  // event->subscribe(2);
}

TEST_P(EventTest, BasicPoisonedTest)
{
  GenEventImpl *event = new GenEventImpl();
  event->init(ID::make_event(0, 0, 0), 0);

  EXPECT_FALSE(event->is_generation_poisoned(1));
}

TEST_P(EventTest, BasicHasTriggerdTest)
{
  GenEventImpl *event = new GenEventImpl();
  event->init(ID::make_event(0, 0, 0), 0);

  bool poisoned = false;
  EXPECT_FALSE(event->has_triggered(1, poisoned));
  EXPECT_FALSE(poisoned);
}

TEST_P(EventTest, BasicTriggerTest)
{
  GenEventImpl *event = new GenEventImpl();
  event->init(ID::make_event(0, 0, 0), 0);

  // TODO: free_event hitting get_runtime
  // event->trigger(1, 0, 0, TimeLimit::responsive());
}

const static EventTestCase kEventTestCases[] = {
    EventTestCase{},
};

INSTANTIATE_TEST_SUITE_P(Foo, EventTest, testing::ValuesIn(kEventTestCases));
