#include "realm/event_impl.h"
#include "realm/barrier_impl.h"
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

class MockBarrierCommunicator : public BarrierCommunicator {
public:
  virtual void adjust(NodeID target, Barrier barrier, int delta, Event wait_on,
                      NodeID sender, bool forwarded, const void *data, size_t datalen)
  {
    sent_adjust_arrivals++;
  }

  virtual void trigger(NodeID target, ID::IDType barrier_id, EventImpl::gen_t trigger_gen,
                       EventImpl::gen_t previous_gen, EventImpl::gen_t first_gen,
                       ReductionOpID redop_id, NodeID migration_target,
                       int base_arrival_count, const void *data, size_t datalen)
  {
    sent_trigger_count++;
  }

  virtual void subscribe(NodeID target, ID::IDType barrier_id,
                         EventImpl::gen_t subscribe_gen, NodeID subscriber,
                         bool forwarded)
  {
    sent_subscription_count++;
  }

  int sent_adjust_arrivals = 0;
  int sent_trigger_count = 0;
  int sent_subscription_count = 0;
  int sent_notification_count = 0;
};

class BarrierTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    barrier_comm = new MockBarrierCommunicator();
    // local_event_free_list =
    //  new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  }

  void TearDown() override {}

  // LocalEventTableAllocator::FreeList *local_event_free_list;
  MockBarrierCommunicator *barrier_comm;
  // DynamicTable<LocalEventTableAllocator> local_events;
};

TEST_F(BarrierTest, RemoteSubscribe)
{
  const NodeID owner = 1;
  const EventImpl::gen_t subscribe_gen = 1;
  BarrierImpl barrier(barrier_comm);

  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.subscribe(subscribe_gen);

  EXPECT_EQ(barrier_comm->sent_subscription_count, 1);
}

TEST_F(BarrierTest, LocalArrive)
{
  const NodeID owner = 0;
  const EventImpl::gen_t arrival_gen = 1;
  bool poisoned = false;
  BarrierImpl barrier;

  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.base_arrival_count = 2;
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/1, 0, 0, 0,
                         TimeLimit::responsive());
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/2, 0, 0, 0,
                         TimeLimit::responsive());
  bool ok = barrier.has_triggered(arrival_gen, poisoned);

  EXPECT_EQ(barrier_comm->sent_trigger_count, 0);
  EXPECT_EQ(barrier.generation.load(), arrival_gen);
  EXPECT_TRUE(ok);
  EXPECT_FALSE(poisoned);
}

// TODO(apryakhin@): This crases on assert and should be handled more
// gracefully.
TEST_F(BarrierTest, DISABLED_LocalArriveOnTriggered)
{
  const NodeID owner = 0;
  const EventImpl::gen_t arrival_gen = 1;
  BarrierImpl barrier;

  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.base_arrival_count = 2;
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/1, 0, 0, 0,
                         TimeLimit::responsive());
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/2, 0, 0, 0,
                         TimeLimit::responsive());
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/3, 0, 0, 0,
                         TimeLimit::responsive());

  EXPECT_EQ(barrier_comm->sent_trigger_count, 0);
  EXPECT_EQ(barrier.generation.load(), arrival_gen);
}

TEST_F(BarrierTest, LocalArriveWithWaiter)
{
  const NodeID owner = 0;
  const EventImpl::gen_t arrival_gen = 1;
  DeferredOperation waiter_one;
  DeferredOperation waiter_two;
  BarrierImpl barrier;

  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.base_arrival_count = 2;
  barrier.add_waiter(arrival_gen, &waiter_one);
  barrier.add_waiter(arrival_gen + 1, &waiter_two);
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/1, 0, 0, 0,
                         TimeLimit::responsive());
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/2, 0, 0, 0,
                         TimeLimit::responsive());

  EXPECT_EQ(barrier_comm->sent_trigger_count, 0);
  EXPECT_EQ(barrier.generation.load(), arrival_gen);
  EXPECT_TRUE(waiter_one.triggered);
  EXPECT_FALSE(waiter_two.triggered);
}

TEST_F(BarrierTest, LocalArriveFutureGen)
{
  const NodeID owner = 0;
  const EventImpl::gen_t arrival_gen = 2;
  BarrierImpl barrier;

  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.base_arrival_count = 2;
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/1, 0, 0, 0,
                         TimeLimit::responsive());
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/2, 0, 0, 0,
                         TimeLimit::responsive());

  EXPECT_EQ(barrier_comm->sent_trigger_count, 0);
  EXPECT_EQ(barrier.generation.load(), 0);
}

TEST_F(BarrierTest, LocalOutOfOrderArrive)
{
  const NodeID owner = 0;
  const EventImpl::gen_t arrival_gen = 2;
  const int delta = -1;
  BarrierImpl barrier;

  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.base_arrival_count = 2;
  barrier.adjust_arrival(arrival_gen, delta, 0, Event::NO_EVENT, /*sender=*/1, 0, 0, 0,
                         TimeLimit::responsive());
  barrier.adjust_arrival(arrival_gen, delta, 0, Event::NO_EVENT, /*sender=*/2, 0, 0, 0,
                         TimeLimit::responsive());
  barrier.adjust_arrival(arrival_gen - 1, delta, 0, Event::NO_EVENT, /*sender=*/3, 0, 0,
                         0, TimeLimit::responsive());
  barrier.adjust_arrival(arrival_gen - 1, delta, 0, Event::NO_EVENT, /*sender=*/4, 0, 0,
                         0, TimeLimit::responsive());

  EXPECT_EQ(barrier_comm->sent_trigger_count, 0);
  EXPECT_EQ(barrier.generation.load(), arrival_gen);
}

TEST_F(BarrierTest, LocalArriveWithRemoteSubscribers)
{
  const NodeID owner = 0;
  const EventImpl::gen_t subscribe_gen = 1;
  const EventImpl::gen_t arrival_gen = 1;
  BarrierImpl barrier(barrier_comm);

  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.handle_remote_subscription(/*subscriber=*/1, subscribe_gen, 0, 0, 0);
  barrier.handle_remote_subscription(/*subscriber=*/2, subscribe_gen, 0, 0, 0);
  barrier.handle_remote_subscription(/*subscriber=*/3, subscribe_gen, 0, 0, 0);

  barrier.base_arrival_count = 2;
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/1, 0, 0, 0,
                         TimeLimit::responsive());
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/2, 0, 0, 0,
                         TimeLimit::responsive());

  EXPECT_EQ(barrier_comm->sent_trigger_count, 3);
  EXPECT_EQ(barrier.generation.load(), arrival_gen);
}

TEST_F(BarrierTest, LocalArriveRemoteFutureSubscription)
{
  const NodeID owner = 0;
  const EventImpl::gen_t subscribe_gen = 1;
  const EventImpl::gen_t arrival_gen = 1;
  BarrierImpl barrier(barrier_comm);

  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.handle_remote_subscription(/*subscriber=*/1, subscribe_gen, 0, 0, 0);
  barrier.handle_remote_subscription(/*subscriber=*/2, subscribe_gen, 0, 0, 0);
  barrier.handle_remote_subscription(/*subscriber=*/3, (subscribe_gen + 1), 0, 0, 0);

  barrier.base_arrival_count = 2;
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/1, 0, 0, 0,
                         TimeLimit::responsive());
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/2, 0, 0, 0,
                         TimeLimit::responsive());

  EXPECT_EQ(barrier_comm->sent_trigger_count, 3);
}

TEST_F(BarrierTest, RemoteArrive)
{
  const NodeID owner = 1;
  BarrierImpl barrier(barrier_comm);
  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.base_arrival_count = 2;
  barrier.adjust_arrival(/*arrival_gen=*/1, -1, 0, Event::NO_EVENT, /*sender=*/0, 0, 0, 0,
                         TimeLimit::responsive());

  EXPECT_EQ(barrier_comm->sent_adjust_arrivals, 1);
}
