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
  BarrierImpl barrier;
  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.base_arrival_count = 2;
  barrier.adjust_arrival(1, -1, 0, Event::NO_EVENT, 1, 0, 0, 0, TimeLimit::responsive());
  barrier.adjust_arrival(1, -1, 0, Event::NO_EVENT, 1, 0, 0, 0, TimeLimit::responsive());
}

TEST_F(BarrierTest, LocalArriveWithRemoteSubscriber)
{
  const NodeID owner = 0;
  const EventImpl::gen_t subscribe_gen = 1;
  BarrierImpl barrier(barrier_comm);

  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.handle_remote_subscription(1, subscribe_gen, 0, 0, 0);
  // barrier.subscribe(subscribe_gen);
  barrier.base_arrival_count = 2;
  barrier.adjust_arrival(1, -1, 0, Event::NO_EVENT, 1, 0, 0, 0, TimeLimit::responsive());
  barrier.adjust_arrival(1, -1, 0, Event::NO_EVENT, 1, 0, 0, 0, TimeLimit::responsive());

  EXPECT_EQ(barrier_comm->sent_trigger_count, 1);
}

TEST_F(BarrierTest, RemoteArrive)
{
  const NodeID owner = 1;
  BarrierImpl barrier(barrier_comm);
  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.base_arrival_count = 2;
  barrier.adjust_arrival(1, -1, 0, Event::NO_EVENT, /*sender*/ 0, 0, 0, 0,
                         TimeLimit::responsive());

  EXPECT_EQ(barrier_comm->sent_adjust_arrivals, 1);
}
