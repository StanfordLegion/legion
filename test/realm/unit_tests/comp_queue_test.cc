#include "realm/event_impl.h"
#include "realm/comp_queue_impl.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>

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

struct PopEventsTestCase {
  size_t max_queue_size;
  size_t num_events;
  size_t num_triggered_events;
  size_t num_pop_events;
  bool resizable = true;
};

class PopEventsTest : public ::testing::TestWithParam<PopEventsTestCase> {
public:
  void SetUp() override
  {
    event_comm = new MockEventCommunicator();
    event_triggerer = new EventTriggerNotifier();
  }

  void TearDown() override {}

  EventTriggerNotifier *event_triggerer;
  MockEventCommunicator *event_comm;
};

TEST_P(PopEventsTest, Base)
{
  auto test_case = GetParam();
  std::vector<Event> pop_events(test_case.max_queue_size);
  CompQueueImpl compqueue;
  std::vector<GenEventImpl *> events;
  std::vector<Event> completed_events;
  NodeID owner = 0;
  constexpr GenEventImpl::gen_t trigger_gen = 1;

  int index = 0;
  for(size_t i = 0; i < test_case.num_events; i++) {
    events.push_back(new GenEventImpl(event_triggerer, new MockEventCommunicator));
    events[i]->init(ID::make_event(owner, index++, 0), 0);
    completed_events.push_back(events[i]->current_event());
  }

  compqueue.init(ID::make_compqueue(owner, index).convert<CompletionQueue>(), 0);
  compqueue.set_capacity(test_case.max_queue_size, test_case.resizable);

  for(size_t i = 0; i < test_case.num_events; i++) {
    compqueue.add_event(events[i]->current_event(), events[i], /*faultaware=*/false);
  }

  for(size_t i = 0; i < test_case.num_triggered_events; i++) {
    events[i]->trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive());
  }

  size_t num_pending_events = compqueue.get_pending_events();
  size_t pop_size = compqueue.pop_events(pop_events.data(), test_case.num_pop_events);

  EXPECT_EQ(num_pending_events, test_case.num_events - test_case.num_triggered_events);
  EXPECT_EQ(pop_size, test_case.num_triggered_events);

  for(size_t i = 0; i < test_case.num_triggered_events; i++) {
    bool poisoned = false;
    EXPECT_TRUE(events[i]->has_triggered(trigger_gen, poisoned));
    EXPECT_FALSE(poisoned);
  }

  for(size_t i = 0; i < pop_size; i++) {
    EXPECT_EQ(pop_events[i].id, completed_events[i].id);
  }

  for(size_t i = test_case.num_triggered_events; i < test_case.num_events; i++) {
    events[i]->trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive());
  }

  for(size_t i = 0; i < test_case.num_events; i++) {
    delete events[i];
  }
}

const static PopEventsTestCase kPopEventsTestCases[] = {
    PopEventsTestCase{/*max_queue_size=*/0, /*num_events=*/0, /*num_triggered_events=*/0,
                      /*num_pop_events=*/1},

    PopEventsTestCase{/*max_queue_size=*/4, /*num_events=*/4, /*num_triggered_events=*/4,
                      /*num_pop_events=*/4},

    PopEventsTestCase{/*max_queue_size=*/4, /*num_events=*/3, /*num_triggered_events=*/3,
                      /*num_pop_events=*/4},

    PopEventsTestCase{/*max_queue_size=*/4, /*num_events=*/5, /*num_triggered_events=*/4,
                      /*num_pop_events=*/5},

    PopEventsTestCase{/*max_queue_size=*/4, /*num_events=*/5, /*num_triggered_events=*/2,
                      /*num_pop_events=*/5},

    // PopEventsTestCase{/*max_queue_size=*/4, /*num_events=*/5,
    // /*num_triggered_events=*/5,
    ///*num_pop_events=*/5, /*resizable=*/false},
};

INSTANTIATE_TEST_SUITE_P(Foo, PopEventsTest, testing::ValuesIn(kPopEventsTestCases));

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
