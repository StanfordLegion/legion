#include "realm/event_impl.h"
#include "realm/comp_queue_impl.h"
#include "realm/activemsg.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

class DeferredOperation : public EventWaiter {
public:
  void defer(Event wait_on) {}
  virtual void event_triggered(bool poisoned, TimeLimit work_until) { triggered = true; }
  virtual void print(std::ostream &os) const {}
  virtual Event get_finish_event(void) const { return Event::NO_EVENT; }
  bool triggered = false;
};

class MockCompQueueCommunicator : public CompQueueCommunicator {
  int sent_trigger_count = 0;
  int sent_subscription_count = 0;
  int sent_notification_count = 0;
};

class MockEventCommunicator : public EventCommunicator {
public:
  virtual void trigger(Event event, NodeID owner, bool poisoned) { sent_trigger_count++; }

  virtual void update(Event event, NodeID to_update,
                      span<EventImpl::gen_t> poisoned_generationse)
  {
    sent_notification_count++;
  }

  virtual void subscribe(Event event, NodeID owner,
                         EventImpl::gen_t previous_subscribe_gen)
  {
    sent_subscription_count++;
  }

  int sent_trigger_count = 0;
  int sent_subscription_count = 0;
  int sent_notification_count = 0;
};

class CompQueueTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    comp_queue_comm = new MockCompQueueCommunicator();
    event_comm = new MockEventCommunicator();
  }

  void TearDown() override {}

  MockCompQueueCommunicator *comp_queue_comm;
  MockEventCommunicator *event_comm;
};

TEST_F(CompQueueTest, SetCapacity)
{
  const NodeID owner = 0;
  const int index = 0;
  const size_t max_size = 16;
  CompQueueImpl compqueue(comp_queue_comm);

  compqueue.init(ID::make_compqueue(owner, index).convert<CompletionQueue>(), 0);
  compqueue.set_capacity(max_size, /*!resizable=*/false);

  EXPECT_EQ(compqueue.get_capacity(), max_size);
}

TEST_F(CompQueueTest, AddEvent)
{
  const NodeID owner = 0;
  const int index = 0;
  const size_t max_size = 16;
  const GenEventImpl::gen_t trigger_gen = 1;
  CompQueueImpl compqueue(comp_queue_comm);
  GenEventImpl event_a(nullptr, new MockEventCommunicator());
  GenEventImpl event_b(nullptr, new MockEventCommunicator());

  event_a.init(ID::make_event(owner, index, 0), 0);
  event_b.init(ID::make_event(owner, index, 0), 0);
  compqueue.init(ID::make_compqueue(owner, index).convert<CompletionQueue>(), 0);
  compqueue.set_capacity(max_size, /*!resizable=*/false);
  compqueue.add_event(&event_a, /*faultaware=*/false);
  compqueue.add_event(&event_b, /*faultaware=*/false);
  size_t num_pending_events = compqueue.get_pending_events();

  EXPECT_EQ(num_pending_events, 2);
}

TEST_F(CompQueueTest, AddAndTriggerEvent)
{
  const NodeID owner = 0;
  const int index = 0;
  const size_t max_size = 16;
  const GenEventImpl::gen_t trigger_gen = 1;
  CompQueueImpl compqueue(comp_queue_comm);
  GenEventImpl event_a(nullptr, new MockEventCommunicator());
  GenEventImpl event_b(nullptr, new MockEventCommunicator());

  event_a.init(ID::make_event(owner, index, 0), 0);
  compqueue.init(ID::make_compqueue(owner, index).convert<CompletionQueue>(), 0);
  compqueue.set_capacity(max_size, /*!resizable=*/false);
  compqueue.add_event(&event_a, /*faultaware=*/false);
  compqueue.add_event(&event_b, /*faultaware=*/false);
  bool free_event =
      event_a.trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive());
  size_t num_pending_events = compqueue.get_pending_events();

  EXPECT_TRUE(free_event);
  EXPECT_EQ(num_pending_events, 1);
}
