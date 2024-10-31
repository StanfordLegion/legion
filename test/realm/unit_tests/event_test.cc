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
  virtual ~MockEventCommunicator();

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

TEST_F(GenEventTest, LocalAddWaiter)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t needed_gen = 1;
  DeferredOperation waiter;
  GenEventImpl event(nullptr, nullptr);

  event.init(ID::make_event(0, 0, 0), owner);
  bool ok = event.add_waiter(needed_gen, &waiter);

  EXPECT_TRUE(ok);
  EXPECT_FALSE(event.current_local_waiters.empty());
}

TEST_F(GenEventTest, RemoteAddWaiter)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t needed_gen = 1;
  DeferredOperation waiter;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  bool ok = event.add_waiter(needed_gen, &waiter);

  EXPECT_TRUE(ok);
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
  bool add_ok = event.add_waiter(needed_gen, &waiter);
  bool rem_ok = event.remove_waiter(needed_gen, &waiter);

  EXPECT_TRUE(add_ok);
  EXPECT_TRUE(rem_ok);
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
  bool add_ok = event.add_waiter(needed_gen_add, &waiter);
  bool rem_ok = event.remove_waiter(needed_gen_rem, &waiter);

  EXPECT_TRUE(add_ok);
  EXPECT_FALSE(rem_ok);
  EXPECT_FALSE(event.current_local_waiters.empty());
}

TEST_F(GenEventTest, ProcessFutureGenerationTest)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t current_gen = 5;
  DeferredOperation waiter;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.add_waiter(current_gen, &waiter);
  event.process_update(current_gen, nullptr, 0, TimeLimit::responsive());

  EXPECT_TRUE(waiter.triggered);
  EXPECT_EQ(event.num_poisoned_generations.load(), 0);
  EXPECT_FALSE(event.is_generation_poisoned(current_gen));
}

TEST_F(GenEventTest, ProcessPoisonedGenerationsTest)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t current_gen = 3;
  const GenEventImpl::gen_t poisoned_gens[] = {2, 3};
  DeferredOperation waiter_one, waiter_two;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.add_waiter(2, &waiter_one);
  event.add_waiter(3, &waiter_two);
  event.process_update(current_gen, poisoned_gens, 2, TimeLimit::responsive());

  EXPECT_TRUE(waiter_one.triggered);
  EXPECT_TRUE(waiter_two.triggered);
  EXPECT_TRUE(event.is_generation_poisoned(2));
  EXPECT_TRUE(event.is_generation_poisoned(3));
}

TEST_F(GenEventTest, ProcessOutdatedGenerationUpdateTest)
{
  const NodeID owner = 1;
  DeferredOperation waiter;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.add_waiter(5, &waiter);
  // Processing an older generation (current event generation is higher)
  event.process_update(3, nullptr, 0, TimeLimit::responsive());

  EXPECT_FALSE(waiter.triggered);
}

TEST_F(GenEventTest, ProcessOrderedFutureGenerationsTriggeringTest)
{
  const NodeID owner = 1;
  DeferredOperation waiter_one, waiter_two;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.add_waiter(1, &waiter_one);
  event.add_waiter(2, &waiter_two);
  event.process_update(2, nullptr, 0, TimeLimit::responsive());

  EXPECT_TRUE(waiter_one.triggered);
  EXPECT_TRUE(waiter_two.triggered);
}

// TODO(apryakhin@): Fix handling of this test case
TEST_F(GenEventTest, DISABLED_ExceedPoisonedGenerationLimitTest)
{
  const NodeID owner = 1;
  const int poisoned_generation_limit = GenEventImpl::POISONED_GENERATION_LIMIT;
  GenEventImpl::gen_t poisoned_gens[poisoned_generation_limit + 1];
  for(int i = 0; i < poisoned_generation_limit + 1; ++i) {
    poisoned_gens[i] = i + 1;
  }
  GenEventImpl event(nullptr, local_event_free_list, nullptr);

  event.init(ID::make_event(0, 0, 0), owner);
  // Process update with excess poisoned generations
  event.process_update(poisoned_generation_limit + 1, poisoned_gens,
                       poisoned_generation_limit + 1, TimeLimit::responsive());

  // Ensure only POISONED_GENERATION_LIMIT generations were recorded
  EXPECT_EQ(event.num_poisoned_generations.load(), poisoned_generation_limit);
}

TEST_F(GenEventTest, ProcessUpdateWithGenerationalGapsTest)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t current_gen = 5;
  DeferredOperation waiter_one, waiter_two;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.add_waiter(3, &waiter_one);
  event.add_waiter(5, &waiter_two);
  event.process_update(current_gen, nullptr, 0, TimeLimit::responsive());

  EXPECT_TRUE(waiter_one.triggered);
  EXPECT_TRUE(waiter_two.triggered);
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
  bool poisoned = false;
  GenEventImpl event(nullptr, nullptr);

  event.init(ID::make_event(0, 0, 0), owner);
  bool ok = event.has_triggered(gen, poisoned);

  EXPECT_FALSE(ok);
  EXPECT_FALSE(poisoned);
}

TEST_F(GenEventTest, LocalTrigger)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t trigger_gen = 1;
  bool poisoned = false;
  GenEventImpl event(nullptr, local_event_free_list);

  event.init(ID::make_event(0, 0, 0), owner);
  event.trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive());

  EXPECT_TRUE(event.has_triggered(trigger_gen, poisoned));
  EXPECT_FALSE(poisoned);
}

TEST_F(GenEventTest, LocalTriggerWithPoison)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t trigger_gen = 1;
  bool poisoned = false;
  GenEventImpl event(nullptr, local_event_free_list);
  event.init(ID::make_event(0, 0, 0), owner);

  event.trigger(trigger_gen, 0, /*poisoned=*/true, TimeLimit::responsive());

  EXPECT_TRUE(event.has_triggered(trigger_gen, poisoned));
  EXPECT_TRUE(poisoned);
}

TEST_F(GenEventTest, LocalTriggerWithWaiter)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t trigger_gen = 1;
  bool poisoned = false;
  DeferredOperation waiter_one;
  GenEventImpl event(nullptr, local_event_free_list);

  event.init(ID::make_event(0, 0, 0), owner);
  bool ok = event.add_waiter(trigger_gen, &waiter_one);
  event.trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive());

  EXPECT_TRUE(ok);
  EXPECT_TRUE(event.has_triggered(trigger_gen, poisoned));
  EXPECT_TRUE(waiter_one.triggered);
}

TEST_F(GenEventTest, LocalTriggerWithMultipleWaitersPoisoned)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t trigger_gen = 1;
  bool poisoned = false;
  DeferredOperation waiter_one;
  DeferredOperation waiter_two;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  bool ok1 = event.add_waiter(trigger_gen, &waiter_one);
  bool ok2 = event.add_waiter(trigger_gen, &waiter_two);
  event.trigger(trigger_gen, 0, /*poisoned=*/true, TimeLimit::responsive());

  EXPECT_TRUE(ok1);
  EXPECT_TRUE(ok2);
  EXPECT_TRUE(waiter_one.triggered);
  EXPECT_TRUE(waiter_two.triggered);
  EXPECT_TRUE(event.is_generation_poisoned(trigger_gen));
  EXPECT_TRUE(event.has_triggered(trigger_gen, poisoned));
}

TEST_F(GenEventTest, LocalTriggerWithMultipleRemoteSubscriptions)
{
  const NodeID owner = 0;
  const NodeID sender_a = 1;
  const NodeID sender_b = 2;
  const GenEventImpl::gen_t subscribe_gen = 1;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.handle_remote_subscription(sender_a, subscribe_gen, 0);
  event.handle_remote_subscription(sender_b, subscribe_gen, 0);
  event.trigger(subscribe_gen, 0, /*poisoned=*/false, TimeLimit::responsive());

  EXPECT_EQ(event_comm->sent_notification_count, 2);
  EXPECT_FALSE(event.remote_waiters.contains(sender_a));
  EXPECT_FALSE(event.remote_waiters.contains(sender_b));
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
  event.trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive());

  EXPECT_EQ(event_comm->sent_trigger_count, 1);
}

TEST_F(GenEventTest, RemoteTriggerWithWaiters)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t trigger_gen = 1;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);
  DeferredOperation waiter_one;
  DeferredOperation waiter_two;

  event.init(ID::make_event(0, 0, 0), owner);
  bool ok1 = event.add_waiter(trigger_gen, &waiter_one);
  bool ok2 = event.add_waiter(trigger_gen + 1, &waiter_two);
  event.trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive());

  EXPECT_TRUE(ok1);
  EXPECT_TRUE(ok2);
  EXPECT_TRUE(waiter_one.triggered);
  EXPECT_FALSE(waiter_two.triggered);
  EXPECT_EQ(event_comm->sent_trigger_count, 1);
}

TEST_F(GenEventTest, RemoteTriggerFutureGen)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t trigger_gen = 2;
  GenEventImpl event(nullptr, local_event_free_list, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive());

  EXPECT_EQ(event_comm->sent_trigger_count, 1);
  EXPECT_EQ(event_comm->sent_subscription_count, 1);
}

TEST_F(GenEventTest, EventMergerIsActive)
{
  const NodeID owner = 0;
  GenEventImpl event(nullptr, nullptr);
  event.init(ID::make_event(0, 0, 0), owner);
  EventMerger merger(&event);

  bool ok = merger.is_active();

  EXPECT_FALSE(ok);
}
