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

TEST_F(EventTest, LocalAddRemoveWaiterSameGen)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t needed_gen = 1;
  DeferredOperation waiter;
  GenEventImpl event(nullptr, nullptr);

  event.init(ID::make_event(0, 0, 0), owner);
  event.add_waiter(needed_gen, &waiter);
  event.remove_waiter(needed_gen, &waiter);

  EXPECT_TRUE(event.current_local_waiters.empty());
}

TEST_F(EventTest, RemoteAddRemoveWaiterDifferentGens)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t needed_gen_add = 1;
  const GenEventImpl::gen_t needed_gen_rem = 2;
  MockEventCommunicator *event_comm = new MockEventCommunicator();
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  GenEventImpl event(nullptr, local_event_free_list, event_comm);
  DeferredOperation waiter;

  event.init(ID::make_event(0, 0, 0), owner);
  EXPECT_TRUE(event.add_waiter(needed_gen_add, &waiter));
  EXPECT_FALSE(event.remove_waiter(needed_gen_rem, &waiter));

  EXPECT_FALSE(event.current_local_waiters.empty());
  EXPECT_EQ(event_comm->sent_subscription_count, 1);
}

TEST_F(EventTest, ProcessUpdateNonOwner)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t needed_gen = 1;
  const GenEventImpl::gen_t current_gen = 1;
  DeferredOperation waiter_one;
  DeferredOperation waiter_two;
  GenEventImpl event(nullptr, nullptr);

  event.init(ID::make_event(0, 0, 0), owner);
  event.process_update(current_gen, 0, 0, TimeLimit::responsive());
  event.add_waiter(needed_gen, &waiter_one);
  event.add_waiter(needed_gen, &waiter_two);

  EXPECT_TRUE(waiter_one.triggered);
  EXPECT_TRUE(waiter_two.triggered);
}

TEST_F(EventTest, Subscribe)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t subscribe_gen = 2;
  MockEventCommunicator *event_comm = new MockEventCommunicator();
  GenEventImpl event(nullptr, nullptr, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.subscribe(subscribe_gen);

  EXPECT_EQ(event_comm->sent_subscription_count, 1);
}

TEST_F(EventTest, HandleRemoteSubscriptionUntriggered)
{
  const NodeID owner = 0;
  const NodeID sender = 1;
  const GenEventImpl::gen_t subscribe_gen = 1;
  MockEventCommunicator *event_comm = new MockEventCommunicator();
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.handle_remote_subscription(sender, subscribe_gen, 0);

  EXPECT_EQ(event_comm->sent_notification_count, 0);
  EXPECT_TRUE(event.remote_waiters.contains(sender));
}

TEST_F(EventTest, HandleRemoteSubscriptionTriggered)
{
  const NodeID owner = 0;
  const NodeID sender = 1;
  const GenEventImpl::gen_t subscribe_gen = 1;
  MockEventCommunicator *event_comm = new MockEventCommunicator();
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.trigger(subscribe_gen, 0, 0, TimeLimit::responsive());
  event.handle_remote_subscription(sender, subscribe_gen, 0);

  EXPECT_EQ(event_comm->sent_notification_count, 1);
}

TEST_F(EventTest, BasicPoisonedTest)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t gen = 1;
  GenEventImpl event(nullptr, nullptr);

  event.init(ID::make_event(0, 0, 0), owner);

  EXPECT_FALSE(event.is_generation_poisoned(gen));
}

TEST_F(EventTest, BasicHasTriggerdTest)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t gen = 1;
  GenEventImpl event(nullptr, nullptr);

  event.init(ID::make_event(0, 0, 0), owner);
  bool poisoned = false;
  EXPECT_FALSE(event.has_triggered(gen, poisoned));

  EXPECT_FALSE(poisoned);
}

TEST_F(EventTest, LocalTrigger)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t gen = 1;
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  GenEventImpl event(nullptr, local_event_free_list);

  event.init(ID::make_event(0, 0, 0), owner);
  event.trigger(gen, 0, 0, TimeLimit::responsive());

  bool poisoned = false;
  EXPECT_TRUE(event.has_triggered(1, poisoned));
  EXPECT_FALSE(poisoned);
}

TEST_F(EventTest, LocalTriggerNotify)
{
  const NodeID owner = 0;
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  DeferredOperation waiter;
  GenEventImpl event(nullptr, local_event_free_list);

  event.init(ID::make_event(0, 0, 0), owner);
  EXPECT_TRUE(event.add_waiter(1, &waiter));
  event.trigger(1, 0, 0, TimeLimit::responsive());
}

TEST_F(EventTest, LocalTriggerPoisoned)
{
  const NodeID owner = 0;
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  GenEventImpl event(nullptr, local_event_free_list);
  event.init(ID::make_event(0, 0, 0), owner);

  event.trigger(1, 0, /*poisoned=*/true, TimeLimit::responsive());

  bool poisoned = false;
  EXPECT_TRUE(event.has_triggered(1, poisoned));
  EXPECT_TRUE(poisoned);
}

TEST_F(EventTest, TriggerWithRemoteSubscription)
{
  const NodeID owner = 0;
  const NodeID sender = 1;
  const GenEventImpl::gen_t subscribe_gen = 1;
  MockEventCommunicator *event_comm = new MockEventCommunicator();
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.handle_remote_subscription(sender, subscribe_gen, 0);
  event.trigger(subscribe_gen, 0, 0, TimeLimit::responsive());

  EXPECT_EQ(event_comm->sent_notification_count, 1);
}

TEST_F(EventTest, RemoteTriggerCurrentGen)
{
  const NodeID owner = 1;
  const NodeID sender = 1;
  const GenEventImpl::gen_t subscribe_gen = 1;
  MockEventCommunicator *event_comm = new MockEventCommunicator();
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.trigger(1, 0, 0, TimeLimit::responsive());

  EXPECT_EQ(event_comm->sent_trigger_count, 1);
}

TEST_F(EventTest, RemoteTriggerFutureGen)
{
  const NodeID owner = 1;
  const NodeID sender = 1;
  const GenEventImpl::gen_t subscribe_gen = 1;
  const GenEventImpl::gen_t trigger_gen = 2;
  MockEventCommunicator *event_comm = new MockEventCommunicator();
  DynamicTable<LocalEventTableAllocator> local_events;
  LocalEventTableAllocator::FreeList *local_event_free_list =
      new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.trigger(trigger_gen, 0, 0, TimeLimit::responsive());

  EXPECT_EQ(event_comm->sent_trigger_count, 1);
  EXPECT_EQ(event_comm->sent_subscription_count, 1);
}

TEST_F(EventTest, EventMergerIsActive)
{
  const NodeID owner = 0;
  GenEventImpl event(nullptr, nullptr);
  event.init(ID::make_event(0, 0, 0), owner);
  EventMerger merger(&event);
  EXPECT_FALSE(merger.is_active());
}
