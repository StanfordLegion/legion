#include "realm/event_impl.h"
#include "realm/activemsg.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

class EventTest : public ::testing::Test {};

class DeferredOperation : public EventWaiter {
public:
  void defer(Event wait_on) {}
  virtual void event_triggered(bool poisoned, TimeLimit work_until) { triggered = true; }
  virtual void print(std::ostream &os) const {}
  virtual Event get_finish_event(void) const { return Event::NO_EVENT; }
  bool triggered = false;
};

class MockEventCommunicator : public EventCommunicator {
public:
  virtual void trigger(Event event, NodeID owner, bool poisoned) { sent_trigger_count++; }

  virtual void update(Event event, NodeID to_update,
                      EventImpl::gen_t *poisoned_generations, size_t size)
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

TEST_F(EventTest, GetCurrentEvent)
{
  GenEventImpl event(nullptr, nullptr);
  event.init(ID::make_event(0, 0, 0), 0);
  EXPECT_EQ(ID(event.current_event()).event_generation(),
            ID::make_event(0, 0, 1).event_generation());
}

TEST_F(EventTest, AddRemoveWaiterSameGen)
{
  DeferredOperation waiter;
  GenEventImpl event(nullptr, nullptr);
  event.init(ID::make_event(0, 0, 0), 0);

  event.add_waiter(1, &waiter);
  event.remove_waiter(1, &waiter);

  EXPECT_TRUE(event.current_local_waiters.empty());
}

TEST_F(EventTest, ProcessUpdateNonOwner)
{
  DeferredOperation waiter_one;
  DeferredOperation waiter_two;

  GenEventImpl event(nullptr, nullptr);
  event.init(ID::make_event(0, 0, 0), 1);

  event.process_update(1, 0, 0, TimeLimit::responsive());

  event.add_waiter(1, &waiter_one);
  event.add_waiter(1, &waiter_two);

  EXPECT_TRUE(waiter_one.triggered);
  EXPECT_TRUE(waiter_two.triggered);
}

TEST_F(EventTest, AddRemoveWaiterDifferentGens)
{
  DeferredOperation waiter;
  GenEventImpl event(nullptr, nullptr);
  event.init(ID::make_event(0, 0, 0), 0);

  EXPECT_TRUE(event.add_waiter(1, &waiter));
  EXPECT_FALSE(event.remove_waiter(2, &waiter));

  EXPECT_FALSE(event.current_local_waiters.empty());
}

TEST_F(EventTest, Subscribe)
{
  MockEventCommunicator *event_comm = new MockEventCommunicator();
  GenEventImpl event(nullptr, nullptr, event_comm);
  event.init(ID::make_event(0, 0, 0), 1);

  event.subscribe(2);

  EXPECT_EQ(event_comm->sent_subscription_count, 1);
}

TEST_F(EventTest, HandleRemoteSubscriptionUntriggered)
{
  const NodeID sender = 1;
  const GenEventImpl::gen_t subscribe_gen = 1;
  MockEventCommunicator *event_comm = new MockEventCommunicator();
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), 0);
  event.handle_remote_subscription(sender, subscribe_gen, 0);

  EXPECT_EQ(event_comm->sent_notification_count, 0);
  EXPECT_TRUE(event.remote_waiters.contains(sender));
}

TEST_F(EventTest, HandleRemoteSubscriptionTriggered)
{
  const NodeID sender = 1;
  const GenEventImpl::gen_t subscribe_gen = 1;
  MockEventCommunicator *event_comm = new MockEventCommunicator();
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), 0);
  event.trigger(subscribe_gen, 0, 0, TimeLimit::responsive());
  event.handle_remote_subscription(sender, subscribe_gen, 0);

  EXPECT_EQ(event_comm->sent_notification_count, 1);
}

TEST_F(EventTest, BasicPoisonedTest)
{
  GenEventImpl event(nullptr, nullptr);
  event.init(ID::make_event(0, 0, 0), 0);

  EXPECT_FALSE(event.is_generation_poisoned(1));
}

TEST_F(EventTest, BasicHasTriggerdTest)
{
  GenEventImpl event(nullptr, nullptr);
  event.init(ID::make_event(0, 0, 0), 0);

  bool poisoned = false;
  EXPECT_FALSE(event.has_triggered(1, poisoned));
  EXPECT_FALSE(poisoned);
}

TEST_F(EventTest, LocalTrigger)
{
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  GenEventImpl event(nullptr, local_event_free_list);
  event.init(ID::make_event(0, 0, 0), 0);

  event.trigger(1, 0, 0, TimeLimit::responsive());

  bool poisoned = false;
  EXPECT_TRUE(event.has_triggered(1, poisoned));
  EXPECT_FALSE(poisoned);
}

TEST_F(EventTest, LocalTriggerNotify)
{
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  DeferredOperation waiter;
  GenEventImpl event(nullptr, local_event_free_list);

  event.init(ID::make_event(0, 0, 0), 0);
  EXPECT_TRUE(event.add_waiter(1, &waiter));
  event.trigger(1, 0, 0, TimeLimit::responsive());
}

TEST_F(EventTest, LocalTriggerPoisoned)
{
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  GenEventImpl event(nullptr, local_event_free_list);
  event.init(ID::make_event(0, 0, 0), 0);

  event.trigger(1, 0, /*poisoned=*/true, TimeLimit::responsive());

  bool poisoned = false;
  EXPECT_TRUE(event.has_triggered(1, poisoned));
  EXPECT_TRUE(poisoned);
}

TEST_F(EventTest, TriggerWithRemoteSubscription)
{
  const NodeID sender = 1;
  const GenEventImpl::gen_t subscribe_gen = 1;
  MockEventCommunicator *event_comm = new MockEventCommunicator();
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), 0);
  event.handle_remote_subscription(sender, subscribe_gen, 0);
  event.trigger(subscribe_gen, 0, 0, TimeLimit::responsive());

  EXPECT_EQ(event_comm->sent_notification_count, 1);
}

TEST_F(EventTest, RemoteTriggerCurrentGen)
{
  const NodeID sender = 1;
  const GenEventImpl::gen_t subscribe_gen = 1;
  MockEventCommunicator *event_comm = new MockEventCommunicator();
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), /*owner=*/1);
  event.trigger(1, 0, 0, TimeLimit::responsive());

  EXPECT_EQ(event_comm->sent_trigger_count, 1);
}

TEST_F(EventTest, RemoteTriggerFutureGen)
{
  const NodeID sender = 1;
  const GenEventImpl::gen_t subscribe_gen = 1;
  const GenEventImpl::gen_t trigger_gen = 2;
  MockEventCommunicator *event_comm = new MockEventCommunicator();
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), /*owner=*/1);
  event.trigger(trigger_gen, 0, 0, TimeLimit::responsive());

  EXPECT_EQ(event_comm->sent_trigger_count, 1);
  EXPECT_EQ(event_comm->sent_subscription_count, 1);
}

TEST_F(EventTest, EventMergerIsActive)
{
  GenEventImpl event(nullptr, nullptr);
  event.init(ID::make_event(0, 0, 0), 0);
  EventMerger merger(&event);
  EXPECT_FALSE(merger.is_active());
}
