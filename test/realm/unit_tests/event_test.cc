#include "realm/event_impl.h"
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

class GenEventTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    event_comm = new MockEventCommunicator();
    local_event_free_list =
        new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  }

  void TearDown() override { delete local_event_free_list; }

  LocalEventTableAllocator::FreeList *local_event_free_list;
  MockEventCommunicator *event_comm;
  DynamicTable<LocalEventTableAllocator> local_events;
};

TEST_F(GenEventTest, GetCurrentEvent)
{
  GenEventImpl event(nullptr, nullptr);
  event.init(ID::make_event(0, 0, 0), 0);
  EXPECT_EQ(ID(event.current_event()).event_generation(),
            ID::make_event(0, 0, 1).event_generation());
}

/*TEST_F(GenEventTest, BasicPoisonedTest)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t gen = 1;
  GenEventImpl event(nullptr, nullptr);

  event.init(ID::make_event(0, 0, 0), owner);

  EXPECT_FALSE(event.is_generation_poisoned(gen));
}*/

TEST_F(GenEventTest, LocalAddWaiter)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t needed_gen = 1;
  DeferredOperation waiter;
  GenEventImpl event(nullptr, nullptr);

  event.init(ID::make_event(0, 0, 0), owner);
  event.add_waiter(needed_gen, &waiter);

  EXPECT_FALSE(event.current_local_waiters.empty());
}

TEST_F(GenEventTest, RemoteAddWaiter)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t needed_gen = 1;
  DeferredOperation waiter;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.add_waiter(needed_gen, &waiter);

  EXPECT_FALSE(event.current_local_waiters.empty());
  EXPECT_EQ(event_comm->sent_subscription_count, 1);
}

TEST_F(GenEventTest, LocalRemoveWaiterSameGen)
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

TEST_F(GenEventTest, LocalRemoveWaiterDifferentGens)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t needed_gen_add = 1;
  const GenEventImpl::gen_t needed_gen_rem = 2;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);
  DeferredOperation waiter;

  event.init(ID::make_event(0, 0, 0), owner);
  EXPECT_TRUE(event.add_waiter(needed_gen_add, &waiter));
  EXPECT_FALSE(event.remove_waiter(needed_gen_rem, &waiter));

  EXPECT_FALSE(event.current_local_waiters.empty());
}

TEST_F(GenEventTest, ProcessUpdateNonOwner)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t needed_gen = 1;
  const GenEventImpl::gen_t current_gen = 1;
  DeferredOperation waiter_one;
  DeferredOperation waiter_two;
  std::vector<GenEventImpl::gen_t> poisoned_gens{1, 2};
  GenEventImpl event(nullptr, nullptr);

  event.init(ID::make_event(0, 0, 0), owner);
  event.process_update(current_gen, poisoned_gens.data(), poisoned_gens.size(),
                       TimeLimit::responsive());
  event.add_waiter(needed_gen, &waiter_one);
  event.add_waiter(needed_gen, &waiter_two);

  EXPECT_TRUE(waiter_one.triggered);
  EXPECT_TRUE(waiter_two.triggered);
  EXPECT_EQ(event.generation.load(), current_gen);
  EXPECT_EQ(event.num_poisoned_generations.load(), 2);
}

TEST_F(GenEventTest, RemoteSubscribeNextGen)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t subscribe_gen = 2;
  MockEventCommunicator *event_comm = new MockEventCommunicator();
  GenEventImpl event(nullptr, nullptr, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.subscribe(subscribe_gen);

  EXPECT_EQ(event_comm->sent_subscription_count, 1);
}

TEST_F(GenEventTest, RemoteSubscribeCurrGen)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t subscribe_gen = 1;
  MockEventCommunicator *event_comm = new MockEventCommunicator();
  GenEventImpl event(nullptr, nullptr, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.process_update(subscribe_gen, 0, 0, TimeLimit::responsive());
  event.subscribe(subscribe_gen);

  EXPECT_EQ(event_comm->sent_subscription_count, 0);
}

TEST_F(GenEventTest, HasTriggeredOnUntriggered)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t gen = 1;
  GenEventImpl event(nullptr, nullptr);

  event.init(ID::make_event(0, 0, 0), owner);
  bool poisoned = false;
  EXPECT_FALSE(event.has_triggered(gen, poisoned));

  EXPECT_FALSE(poisoned);
}

TEST_F(GenEventTest, LocalTrigger)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t trigger_gen = 1;
  GenEventImpl event(nullptr, local_event_free_list);

  event.init(ID::make_event(0, 0, 0), owner);
  event.trigger(trigger_gen, 0, 0, TimeLimit::responsive());

  bool poisoned = false;
  EXPECT_TRUE(event.has_triggered(trigger_gen, poisoned));
  EXPECT_FALSE(poisoned);
}

TEST_F(GenEventTest, LocalTriggerWithPoison)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t trigger_gen = 1;
  GenEventImpl event(nullptr, local_event_free_list);
  event.init(ID::make_event(0, 0, 0), owner);

  event.trigger(trigger_gen, 0, /*poisoned=*/true, TimeLimit::responsive());

  bool poisoned = false;
  EXPECT_TRUE(event.has_triggered(trigger_gen, poisoned));
  EXPECT_TRUE(poisoned);
}

TEST_F(GenEventTest, LocalTriggerWithWaiter)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t trigger_gen = 1;
  DeferredOperation waiter_one;
  GenEventImpl event(nullptr, local_event_free_list);

  event.init(ID::make_event(0, 0, 0), owner);
  EXPECT_TRUE(event.add_waiter(trigger_gen, &waiter_one));
  event.trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive());

  bool poisoned = false;
  EXPECT_TRUE(event.has_triggered(trigger_gen, poisoned));
  EXPECT_TRUE(waiter_one.triggered);
}

TEST_F(GenEventTest, LocalTriggerWithRemoteSubscription)
{
  const NodeID owner = 0;
  const NodeID sender_a = 1;
  const NodeID sender_b = 2;
  const GenEventImpl::gen_t subscribe_gen = 1;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.trigger(subscribe_gen, 0, 0, TimeLimit::responsive());
  event.handle_remote_subscription(sender_a, subscribe_gen, 0);
  event.handle_remote_subscription(sender_b, subscribe_gen, 0);

  EXPECT_EQ(event_comm->sent_notification_count, 2);
}

TEST_F(GenEventTest, HandleRemoteSubscriptionUntriggered)
{
  const NodeID owner = 0;
  const NodeID sender = 1;
  const GenEventImpl::gen_t subscribe_gen = 1;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.handle_remote_subscription(sender, subscribe_gen, 0);

  EXPECT_EQ(event_comm->sent_notification_count, 0);
  EXPECT_TRUE(event.remote_waiters.contains(sender));
}

TEST_F(GenEventTest, RemoteTrigger)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t trigger_gen = 1;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.trigger(trigger_gen, 0, 0, TimeLimit::responsive());

  EXPECT_EQ(event_comm->sent_trigger_count, 1);
}

TEST_F(GenEventTest, RemoteTriggerWithWaiter)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t trigger_gen = 1;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);
  DeferredOperation waiter_one;
  DeferredOperation waiter_two;

  event.init(ID::make_event(0, 0, 0), owner);
  EXPECT_TRUE(event.add_waiter(trigger_gen, &waiter_one));
  EXPECT_TRUE(event.add_waiter(trigger_gen + 1, &waiter_two));
  event.trigger(trigger_gen, 0, 0, TimeLimit::responsive());

  EXPECT_TRUE(waiter_one.triggered);
  EXPECT_FALSE(waiter_two.triggered);
  EXPECT_EQ(event_comm->sent_trigger_count, 1);
}

TEST_F(GenEventTest, RemoteTriggerCurrentGen)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t trigger_gen = 1;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.trigger(trigger_gen, 0, 0, TimeLimit::responsive());

  EXPECT_EQ(event_comm->sent_trigger_count, 1);
  EXPECT_EQ(event_comm->sent_subscription_count, 0);
}

TEST_F(GenEventTest, RemoteTriggerFutureGen)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t trigger_gen = 2;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.trigger(trigger_gen, 0, 0, TimeLimit::responsive());

  EXPECT_EQ(event_comm->sent_trigger_count, 1);
  EXPECT_EQ(event_comm->sent_subscription_count, 1);
}

TEST_F(GenEventTest, EventMergerIsActive)
{
  const NodeID owner = 0;
  GenEventImpl event(nullptr, nullptr);
  event.init(ID::make_event(0, 0, 0), owner);
  EventMerger merger(&event);
  EXPECT_FALSE(merger.is_active());
}
