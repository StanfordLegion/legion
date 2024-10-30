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
  virtual void trigger(Event event, NodeID owner, bool poisoned) { sent_trigger_count++; }

  virtual void subscribe(NodeID target, ID::IDType barrier_id,
                         EventImpl::gen_t subscribe_gen, NodeID subscriber,
                         bool forwarded)
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
    barrier_comm = new MockBarrierCommunicator();
    // local_event_free_list =
    //  new LocalEventTableAllocator::FreeList(local_events, Network::my_node_id);
  }

  void TearDown() override {}

  // LocalEventTableAllocator::FreeList *local_event_free_list;
  MockBarrierCommunicator *barrier_comm;
  // DynamicTable<LocalEventTableAllocator> local_events;
};

TEST_F(GenEventTest, RemoteSubscribe)
{
  const NodeID owner = 1;
  const EventImpl::gen_t subscribe_gen = 1;
  BarrierImpl barrier(barrier_comm);
  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.subscribe(subscribe_gen);

  EXPECT_EQ(barrier_comm->sent_subscription_count, 1);
  // barrier.base_arrival_count = 2;
  // barrier.adjust_arrival(1, 1, 0, Event::NO_EVENT, 1, 0, 0, 0,
  // TimeLimit::responsive()); barrier.adjust_arrival(1, 1, 0, Event::NO_EVENT, 1, 0, 0,
  // 0, TimeLimit::responsive());
}

TEST_F(GenEventTest, LocalArrive)
{
  const NodeID owner = 0;
  BarrierImpl barrier;
  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.base_arrival_count = 2;
  barrier.adjust_arrival(1, -1, 0, Event::NO_EVENT, 1, 0, 0, 0, TimeLimit::responsive());
  barrier.adjust_arrival(1, -1, 0, Event::NO_EVENT, 1, 0, 0, 0, TimeLimit::responsive());
  // GenEventImpl event(nullptr, nullptr);
  // event.init(ID::make_event(0, 0, 0), 0);
  // EXPECT_EQ(ID(event.current_event()).event_generation(),
  //        ID::make_event(0, 0, 1).event_generation());
}

TEST_F(GenEventTest, LocalArriveWithRemoteSubscriber)
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
  // TODO need to overload trigger
}
