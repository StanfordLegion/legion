#include "realm/event_impl.h"
#include "realm/comp_queue_impl.h"
#include <gtest/gtest.h>

using namespace Realm;

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
    event_comm = new MockEventCommunicator();
    event_triggerer = new EventTriggerNotifier();
  }

  void TearDown() override {}

  EventTriggerNotifier *event_triggerer;
  MockEventCommunicator *event_comm;
};

TEST_F(CompQueueTest, SetCapacity)
{
  const NodeID owner = 0;
  const int index = 0;
  const size_t max_size = 16;
  CompQueueImpl compqueue;

  compqueue.init(ID::make_compqueue(owner, index).convert<CompletionQueue>(), 0);
  compqueue.set_capacity(max_size, /*!resizable=*/false);

  EXPECT_EQ(compqueue.get_capacity(), max_size);
}

TEST_F(CompQueueTest, Destroy)
{
  const NodeID owner = 0;
  const int index = 0;
  const size_t max_size = 16;
  CompQueueImpl compqueue;

  compqueue.init(ID::make_compqueue(owner, index).convert<CompletionQueue>(), 0);
  compqueue.set_capacity(max_size, /*!resizable=*/false);
  compqueue.destroy();

  EXPECT_EQ(compqueue.get_capacity(), 0);
}

TEST_F(CompQueueTest, AddEvent)
{
  const NodeID owner = 0;
  const int index = 0;
  const size_t max_size = 16;
  CompQueueImpl compqueue;
  GenEventImpl event_a(event_triggerer, new MockEventCommunicator());
  GenEventImpl event_b(event_triggerer, new MockEventCommunicator());

  event_a.init(ID::make_event(owner, index, 0), 0);
  event_b.init(ID::make_event(owner, index, 0), 0);
  compqueue.init(ID::make_compqueue(owner, index).convert<CompletionQueue>(), 0);
  compqueue.set_capacity(max_size, /*!resizable=*/false);
  compqueue.add_event(event_a.current_event(), &event_a, /*faultaware=*/false);
  compqueue.add_event(event_b.current_event(), &event_b, /*faultaware=*/false);
  size_t num_pending_events = compqueue.get_pending_events();

  EXPECT_EQ(num_pending_events, 2);
  EXPECT_TRUE(event_a.trigger(1, 0, /*poisoned=*/false, TimeLimit::responsive()));
  EXPECT_TRUE(event_b.trigger(1, 0, /*poisoned=*/false, TimeLimit::responsive()));
}

TEST_F(CompQueueTest, AddAndCompleteEventRemote)
{
  const NodeID owner = 1;
  const int index = 0;
  const size_t max_size = 16;
  const GenEventImpl::gen_t trigger_gen = 1;
  bool poisoned_a = false;
  bool poisoned_b = true;
  CompQueueImpl compqueue;
  GenEventImpl event_a(event_triggerer, new MockEventCommunicator());
  GenEventImpl event_b(event_triggerer, event_comm);

  event_a.init(ID::make_event(owner, index, 0), 0);
  compqueue.init(ID::make_compqueue(owner, index).convert<CompletionQueue>(), 0);
  compqueue.set_capacity(max_size, /*!resizable=*/false);
  compqueue.add_event(event_a.current_event(), &event_a, /*faultaware=*/false);
  compqueue.add_event(event_b.current_event(), &event_b, /*faultaware=*/false);
  bool free_event =
      event_b.trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive());
  size_t num_pending_events = compqueue.get_pending_events();
  bool ok_a = event_a.has_triggered(trigger_gen, poisoned_a);
  bool ok_b = event_b.has_triggered(trigger_gen, poisoned_b);

  EXPECT_FALSE(free_event);
  EXPECT_EQ(num_pending_events, 1);
  EXPECT_FALSE(ok_a);
  EXPECT_TRUE(ok_b);
  EXPECT_FALSE(poisoned_a);
  EXPECT_FALSE(poisoned_b);
  EXPECT_TRUE(
      event_a.trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive()));
}

TEST_F(CompQueueTest, PopEventsEmpty)
{
  const NodeID owner = 0;
  const int index = 0;
  const size_t max_size = 12;
  std::vector<Event> events;
  CompQueueImpl compqueue;
  GenEventImpl event_a(nullptr, event_comm);

  event_a.init(ID::make_event(owner, index, 0), 0);
  compqueue.init(ID::make_compqueue(owner, index).convert<CompletionQueue>(), 0);
  compqueue.set_capacity(max_size, /*!resizable=*/false);

  size_t size = compqueue.pop_events(events.data(), max_size);

  EXPECT_EQ(size, 0);
  EXPECT_TRUE(events.empty());
}

TEST_F(CompQueueTest, PopLessEvents)
{
  const NodeID owner = 0;
  int index = 0;
  const size_t max_size = 4;
  const size_t untriggered = 2;
  const GenEventImpl::gen_t trigger_gen = 1;
  std::vector<Event> pop_events(max_size);
  CompQueueImpl compqueue;
  std::vector<GenEventImpl *> events;
  std::vector<Event> completed_events;

  for(size_t i = 0; i < max_size; i++) {
    events.push_back(new GenEventImpl(event_triggerer, new MockEventCommunicator));
    events[i]->init(ID::make_event(owner, index++, 0), 0);
    completed_events.push_back(events[i]->current_event());
  }
  compqueue.init(ID::make_compqueue(owner, index).convert<CompletionQueue>(), 0);
  compqueue.set_capacity(max_size, /*!resizable=*/true);
  for(size_t i = 0; i < max_size; i++) {
    compqueue.add_event(events[i]->current_event(), events[i], /*faultaware=*/false);
  }
  for(size_t i = 0; i < max_size - untriggered; i++) {
    events[i]->trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive());
  }
  size_t num_pending_events = compqueue.get_pending_events();
  size_t pop_size = compqueue.pop_events(pop_events.data(), max_size);

  EXPECT_EQ(num_pending_events, untriggered);
  EXPECT_EQ(pop_size, max_size - untriggered);
  for(size_t i = 0; i < max_size - untriggered; i++) {
    bool poisoned = false;
    EXPECT_TRUE(events[i]->has_triggered(trigger_gen, poisoned));
    EXPECT_FALSE(poisoned);
  }
  for(size_t i = 0; i < pop_size; i++) {
    EXPECT_EQ(pop_events[i].id, completed_events[i].id);
  }
  for(size_t i = max_size - untriggered; i < max_size; i++) {
    events[i]->trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive());
  }
  for(size_t i = 0; i < max_size; i++) {
    delete events[i];
  }
}

TEST_F(CompQueueTest, DISABLED_GetLocalProgressEvent)
{
  const NodeID owner = 0;
  const int index = 0;
  const size_t max_size = 1;
  CompQueueImpl compqueue;
  GenEventImpl event_a(nullptr, event_comm);

  event_a.init(ID::make_event(owner, index, 0), 0);
  compqueue.init(ID::make_compqueue(owner, index).convert<CompletionQueue>(), 0);
  compqueue.set_capacity(max_size, /*!resizable=*/false);
  Event e = compqueue.get_local_progress_event();

  EXPECT_EQ(e, Event::NO_EVENT);
}

TEST_F(CompQueueTest, DISABLED_AddRemoteProgressEvent)
{
  const NodeID owner = 0;
  const int index = 0;
  const size_t max_size = 1;
  CompQueueImpl compqueue;
  GenEventImpl event_a(event_triggerer, event_comm);
  GenEventImpl event_b(event_triggerer, new MockEventCommunicator());

  event_a.init(ID::make_event(owner, index, 0), 0);
  compqueue.init(ID::make_compqueue(owner, index).convert<CompletionQueue>(), 0);
  compqueue.set_capacity(max_size, /*!resizable=*/false);
  compqueue.add_event(event_b.current_event(), &event_b, /*faultaware=*/false);
  compqueue.add_remote_progress_event(event_a.current_event());

  EXPECT_EQ(compqueue.get_pending_events(), 1);
  event_b.trigger(1, 0, /*poisoned=*/false, TimeLimit::responsive());
}
