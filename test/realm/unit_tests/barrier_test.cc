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
  MockBarrierCommunicator(size_t _max_rec_size = 1024)
    : max_rec_size(_max_rec_size)
  {}

  virtual void adjust(NodeID target, Barrier barrier, int delta, Event wait_on,
                      NodeID sender, bool forwarded, const void *data, size_t datalen)
  {
    sent_adjust_arrivals++;
  }

  virtual void trigger(NodeID target, ID::IDType barrier_id, const void *data,
                       size_t datalen, size_t max_payload_size = 0)
  {
    payload_size = datalen;
    buf = data;
    sent_trigger_count++;
  }

  virtual void subscribe(NodeID target, ID::IDType barrier_id,
                         EventImpl::gen_t subscribe_gen, NodeID subscriber,
                         bool forwarded)
  {
    sent_subscription_count++;
  }

  virtual size_t recommend_max_payload(NodeID node, size_t size, bool with_congestion)
  {
    return max_rec_size;
  }

  int sent_adjust_arrivals = 0;
  int sent_trigger_count = 0;
  int sent_subscription_count = 0;
  int sent_notification_count = 0;
  size_t max_rec_size;
  size_t payload_size = 0;
  const void *buf = nullptr;
};

class BarrierTest : public ::testing::Test {
protected:
  void SetUp() override { barrier_comm = new MockBarrierCommunicator(); }

  void TearDown() override {}

  MockBarrierCommunicator *barrier_comm;
};

// TODO(apryakhin@):
// 1. Test redop
// 2. Test migration

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
  BarrierImpl barrier(barrier_comm);

  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.base_arrival_count = 2;
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/1, 0, 0, 0,
                         TimeLimit::responsive());
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/2, 0, 0, 0,
                         TimeLimit::responsive());
  bool ok = barrier.has_triggered(arrival_gen, poisoned);

  EXPECT_EQ(barrier_comm->sent_adjust_arrivals, 0);
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
  BarrierImpl barrier(barrier_comm);

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

TEST_F(BarrierTest, LocalArriveTriggerOneWaiter)
{
  const NodeID owner = 0;
  const EventImpl::gen_t arrival_gen = 1;
  DeferredOperation waiter_one;
  DeferredOperation waiter_two;
  BarrierImpl barrier(barrier_comm);

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

TEST_F(BarrierTest, LocalArriveWithoutWaiterTrigger)
{
  const NodeID owner = 0;
  const EventImpl::gen_t arrival_gen = 1;
  DeferredOperation waiter_one;
  DeferredOperation waiter_two;
  BarrierImpl barrier(barrier_comm);

  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.base_arrival_count = 2;
  barrier.add_waiter(arrival_gen, &waiter_one);
  barrier.add_waiter(arrival_gen + 1, &waiter_two);
  barrier.remove_waiter(arrival_gen, &waiter_one);
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/1, 0, 0, 0,
                         TimeLimit::responsive());
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/2, 0, 0, 0,
                         TimeLimit::responsive());

  EXPECT_EQ(barrier_comm->sent_trigger_count, 0);
  EXPECT_EQ(barrier.generation.load(), arrival_gen);
  EXPECT_FALSE(waiter_one.triggered);
  EXPECT_FALSE(waiter_two.triggered);
}

TEST_F(BarrierTest, LocalArriveTriggerBothWaiters)
{
  const NodeID owner = 0;
  const EventImpl::gen_t arrival_gen = 1;
  DeferredOperation waiter_one;
  DeferredOperation waiter_two;
  BarrierImpl barrier(barrier_comm);

  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.base_arrival_count = 2;
  barrier.add_waiter(arrival_gen, &waiter_one);
  barrier.add_waiter(arrival_gen + 1, &waiter_two);
  barrier.adjust_arrival(arrival_gen, -2, 0, Event::NO_EVENT, /*sender=*/1, 0, 0, 0,
                         TimeLimit::responsive());
  barrier.adjust_arrival(arrival_gen + 1, -2, 0, Event::NO_EVENT, /*sender=*/2, 0, 0, 0,
                         TimeLimit::responsive());

  EXPECT_EQ(barrier_comm->sent_trigger_count, 0);
  EXPECT_EQ(barrier.generation.load(), arrival_gen + 1);
  EXPECT_TRUE(waiter_one.triggered);
  EXPECT_TRUE(waiter_two.triggered);
}

TEST_F(BarrierTest, LocalArriveFutureGen)
{
  const NodeID owner = 0;
  const EventImpl::gen_t arrival_gen = 2;
  BarrierImpl barrier(barrier_comm);

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
  BarrierImpl barrier(barrier_comm);

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

TEST_F(BarrierTest, LocalArriveWithRemoteSubscribersRadix4SplitPayload)
{
  const NodeID owner = 0;
  const EventImpl::gen_t subscribe_gen = 1;
  const EventImpl::gen_t arrival_gen = 1;
  const int radix = 4;
  const int num_subscribers = 3;
  const int message_split = 1;
  int num_active_messages = 0;

  MockBarrierCommunicator *barrier_communicator = new MockBarrierCommunicator(
      sizeof(BarrierTriggerMessageArgsInternal) + sizeof(size_t) +
      sizeof(RemoteNotification) * message_split);

  BarrierImpl barrier(barrier_communicator, radix);

  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  for(int i = 1; i <= num_subscribers; i++) {
    barrier.handle_remote_subscription(/*subscriber=*/i, subscribe_gen, 0, 0, 0);
    num_active_messages += message_split;
  }

  barrier.base_arrival_count = 2;
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/1, 0, 0, 0,
                         TimeLimit::responsive());
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/2, 0, 0, 0,
                         TimeLimit::responsive());

  EXPECT_EQ(barrier_communicator->sent_trigger_count, num_active_messages);
  EXPECT_EQ(barrier.generation.load(), arrival_gen);
}

TEST_F(BarrierTest, LocalArriveWithRemoteSubscribersRadix4)
{
  const NodeID owner = 0;
  const EventImpl::gen_t subscribe_gen = 1;
  const EventImpl::gen_t arrival_gen = 1;
  const int radix = 4;
  BarrierImpl barrier(barrier_comm, radix);

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

TEST_F(BarrierTest, LocalArriveWithRemoteSubscribersRadix2)
{
  const NodeID owner = 0;
  const EventImpl::gen_t subscribe_gen = 1;
  const EventImpl::gen_t arrival_gen = 1;
  const int radix = 2;
  BarrierImpl barrier(barrier_comm, radix);

  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.handle_remote_subscription(/*subscriber=*/1, subscribe_gen, 0, 0, 0);
  barrier.handle_remote_subscription(/*subscriber=*/2, subscribe_gen, 0, 0, 0);
  barrier.handle_remote_subscription(/*subscriber=*/3, subscribe_gen, 0, 0, 0);

  barrier.base_arrival_count = 2;
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/1, 0, 0, 0,
                         TimeLimit::responsive());
  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/2, 0, 0, 0,
                         TimeLimit::responsive());

  EXPECT_EQ(barrier_comm->sent_trigger_count, 2);
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

TEST_F(BarrierTest, HandleRemoteSubscription)
{
  const NodeID owner = 1;
  const EventImpl::gen_t subscribe_gen = 1;
  BarrierImpl barrier(barrier_comm);

  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.handle_remote_subscription(/*subscriber=*/1, subscribe_gen, 0, 0, 0);

  EXPECT_EQ(barrier_comm->sent_subscription_count, 1);
  EXPECT_EQ(barrier.generation.load(), 0);
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

TEST_F(BarrierTest, RemoteAddWaiter)
{
  NodeID owner = 1;
  BarrierImpl barrier(barrier_comm);
  EventImpl::gen_t trigger_gen = 1;
  Realm::ID barrier_id = ID::make_barrier(owner, 0, 0);
  DeferredOperation waiter_one;
  DeferredOperation waiter_two;

  barrier.init(barrier_id, owner);
  barrier.add_waiter(trigger_gen, &waiter_one);
  barrier.add_waiter(trigger_gen + 1, &waiter_two);

  EXPECT_EQ(barrier_comm->sent_subscription_count, 2);
  EXPECT_FALSE(waiter_one.triggered);
  EXPECT_FALSE(waiter_two.triggered);
}

TEST_F(BarrierTest, HandleRemoteTriggerNextGen)
{
  NodeID owner = 1;
  BarrierImpl barrier(barrier_comm);
  EventImpl::gen_t trigger_gen = 1;
  EventImpl::gen_t previous_gen = 0;
  EventImpl::gen_t first_gen = 0;
  NodeID migration_target = owner;

  int broadcast_index = 0;

  unsigned base_count = 2;
  auto barrier_id = ID::make_barrier(owner, 0, 0);
  DeferredOperation waiter_one;
  DeferredOperation waiter_two;

  barrier.init(barrier_id, owner);
  barrier.add_waiter(trigger_gen, &waiter_one);
  barrier.add_waiter(trigger_gen + 1, &waiter_two);

  barrier.handle_remote_trigger(0, 0, trigger_gen, previous_gen, first_gen, 0,
                                migration_target, broadcast_index, base_count, nullptr, 0,
                                TimeLimit::responsive());

  EXPECT_EQ(barrier.generation.load(), trigger_gen);
  EXPECT_TRUE(waiter_one.triggered);
  EXPECT_FALSE(waiter_two.triggered);
}

TEST_F(BarrierTest, HandleRemoteTriggerCurrGen)
{
  NodeID owner = 1;
  BarrierImpl barrier(barrier_comm);
  EventImpl::gen_t trigger_gen = 2;
  EventImpl::gen_t previous_gen = 2;
  EventImpl::gen_t first_gen = 0;
  NodeID migration_target = owner;

  int broadcast_index = 0;

  unsigned base_count = 2;
  auto barrier_id = ID::make_barrier(owner, 0, 0);

  barrier.init(barrier_id, owner);
  barrier.handle_remote_trigger(0, 0, trigger_gen, previous_gen, first_gen, 0,
                                migration_target, broadcast_index, base_count, nullptr, 0,
                                TimeLimit::responsive());

  EXPECT_EQ(barrier.generation.load(), 0);
}

TEST_F(BarrierTest, HandleRemoteTriggerHigherPrevGen)
{
  NodeID owner = 1;
  BarrierImpl barrier(barrier_comm);
  EventImpl::gen_t trigger_gen = 1;
  EventImpl::gen_t previous_gen = 2;
  EventImpl::gen_t first_gen = 0;
  NodeID migration_target = owner;

  int broadcast_index = 0;

  unsigned base_count = 2;
  auto barrier_id = ID::make_barrier(owner, 0, 0);

  barrier.init(barrier_id, owner);
  barrier.handle_remote_trigger(0, 0, trigger_gen, previous_gen, first_gen, 0,
                                migration_target, broadcast_index, base_count, nullptr, 0,
                                TimeLimit::responsive());

  EXPECT_EQ(barrier.generation.load(), 0);
}

TEST_F(BarrierTest, HandleRemoteTriggerCheckPayload)
{
  NodeID owner = 1;
  auto barrier_id = ID::make_barrier(owner, 0, 0);
  EventImpl::gen_t trigger_gen = 1;
  EventImpl::gen_t previous_gen = 0;
  EventImpl::gen_t first_gen = 0;
  NodeID migration_target = owner;
  unsigned base_count = 2;

  BarrierImpl barrier(barrier_comm);

  int broadcast_index = 0;

  barrier.init(barrier_id, owner);
  barrier.base_arrival_count = 2;

  BarrierTriggerPayload payload;
  RemoteNotification rn;
  rn.trigger_gen = trigger_gen;
  rn.node = 4;
  rn.previous_gen = 0;
  payload.remotes.emplace_back(rn);

  Serialization::DynamicBufferSerializer dbs(payload.remotes.size() *
                                             sizeof(RemoteNotification));
  bool ok = dbs & payload;
  assert(ok);

  barrier.handle_remote_trigger(
      0, 0, trigger_gen, previous_gen, first_gen, 0, migration_target, broadcast_index,
      base_count, dbs.get_buffer(), dbs.bytes_used(), TimeLimit::responsive());

  EXPECT_TRUE(ok);

  EXPECT_EQ(barrier.generation.load(), trigger_gen);
  ASSERT_GE(barrier_comm->payload_size, sizeof(BarrierTriggerMessageArgs));
  ASSERT_EQ(dbs.bytes_used(),
            barrier_comm->payload_size - sizeof(BarrierTriggerMessageArgs));
}

class ReductionOpIntAdd {
public:
  typedef int LHS;
  typedef int RHS;

  template <bool EXCL>
  static void apply(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  // both of these are optional
  static const RHS identity;

  template <bool EXCL>
  static void fold(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }
};

const ReductionOpIntAdd::RHS ReductionOpIntAdd::identity = 0;

class BarrierRedopTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    barrier_comm = new MockBarrierCommunicator();
    redop = new ReductionOp<ReductionOpIntAdd>();
  }

  void TearDown() override { delete redop; }

  MockBarrierCommunicator *barrier_comm;
  ReductionOpUntyped *redop;
  ReductionOpID redop_id = 1;
};

template <typename T, size_t BYTES>
struct Pad {
  static_assert(BYTES >= sizeof(T), "Padding size must be at least the size of T");

  T val;
  char padding[BYTES - sizeof(T)];

  Pad() {}
  Pad(T _val)
    : val(_val)
  {}

  operator T() const { return val; }

  Pad &operator+=(const T &rhs)
  {
    val += rhs;
    return *this;
  }

  Pad &operator*=(const T &rhs)
  {
    val *= rhs;
    return *this;
  }
};

template <typename T, size_t BYTES>
std::ostream &operator<<(std::ostream &, const Pad<T, BYTES> &);

template <typename T, size_t BYTES>
std::ostream &operator<<(std::ostream &os, const Pad<T, BYTES> &pad)
{
  os << pad.val;
  return os;
}

class LargeReductionOpIntAdd {
public:
  static constexpr int bytes = 64;
  typedef Pad<float, bytes> LHS;
  typedef Pad<float, bytes> RHS;

  template <bool EXCL>
  static void apply(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  static const RHS identity;

  template <bool EXCL>
  static void fold(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }
};

const LargeReductionOpIntAdd::RHS LargeReductionOpIntAdd::identity = 0;

TEST_F(BarrierRedopTest, LocalArriveWithRemoteSubscribersRadix4SplitPayload)
{
  const NodeID owner = 0;
  const EventImpl::gen_t subscribe_gen = 1;
  const EventImpl::gen_t arrival_gen = 1;
  const int radix = 4;
  const int num_subscribers = 3;

  std::unique_ptr<ReductionOpUntyped> large_redop =
      std::make_unique<ReductionOp<LargeReductionOpIntAdd>>();

  MockBarrierCommunicator *barrier_communicator = new MockBarrierCommunicator(
      sizeof(BarrierTriggerMessageArgsInternal) + sizeof(size_t) + 16);

  BarrierImpl barrier(barrier_communicator, radix);

  std::vector<LargeReductionOpIntAdd::LHS> reduce_value(1, 4);
  LargeReductionOpIntAdd::LHS result(0);
  barrier.init(ID::make_barrier(owner, 0, 0), owner);
  barrier.initial_value = std::make_unique<char[]>(sizeof(LargeReductionOpIntAdd::LHS));
  barrier.redop = large_redop.get();
  barrier.redop_id = redop_id;
  barrier.base_arrival_count = 2;

  for(int i = 1; i <= num_subscribers; i++) {
    barrier.handle_remote_subscription(/*subscriber=*/i, subscribe_gen, 0, 0, 0);
  }

  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/1, 0,
                         reinterpret_cast<void *>(reduce_value.data()),
                         (sizeof(reduce_value[0])), TimeLimit::responsive());

  barrier.adjust_arrival(arrival_gen, -1, 0, Event::NO_EVENT, /*sender=*/2, 0,
                         reinterpret_cast<void *>(reduce_value.data()),
                         sizeof(reduce_value[0]), TimeLimit::responsive());

  for(unsigned i = 0; i < barrier.base_arrival_count; i++) {
    LargeReductionOpIntAdd::apply<true>(result, reduce_value[0]);
  }

  ASSERT_GE(barrier_communicator->sent_trigger_count, radix - 1);

  /*for(int i = 1; i <= num_subscribers; i++) {
    // ASSERT_EQ(barrier_communicator->data_bytes[i], sizeof(reduce_value[0]));
    LargeReductionOpIntAdd::LHS pad_value;
    // std::memcpy(&pad_value, barrier_communicator->payloads[i].reduction.data(),
    // sizeof(LargeReductionOpIntAdd::LHS));

    ASSERT_EQ(pad_value, result);
  }*/
  ASSERT_EQ(barrier.generation.load(), arrival_gen);
}

TEST_F(BarrierRedopTest, GetEmptyResultForUntriggeredGen)
{
  NodeID owner = 0;
  auto barrier_id = ID::make_barrier(owner, 0, 0);
  EventImpl::gen_t result_gen = 1;
  void *value = nullptr;
  size_t value_size = 0;
  BarrierImpl barrier(barrier_comm);

  barrier.init(barrier_id, owner);
  bool ok = barrier.get_result(result_gen, value, value_size);

  EXPECT_FALSE(ok);
}

TEST_F(BarrierRedopTest, GetResultForTriggeredGen)
{
  NodeID owner = 0;
  auto barrier_id = ID::make_barrier(owner, 0, 0);
  EventImpl::gen_t result_gen = 1;
  std::vector<int> reduce_value(1, 4);
  std::vector<int> result(1, 0);
  BarrierImpl barrier(barrier_comm);

  barrier.init(barrier_id, owner);
  barrier.initial_value = std::make_unique<char[]>(sizeof(int));
  barrier.redop = redop;
  barrier.redop_id = redop_id;
  barrier.base_arrival_count = 2;
  barrier.adjust_arrival(result_gen, -1, 0, Event::NO_EVENT, /*sender=*/1, 0,
                         reinterpret_cast<void *>(reduce_value.data()),
                         sizeof(reduce_value[0]), TimeLimit::responsive());
  barrier.adjust_arrival(result_gen, -1, 0, Event::NO_EVENT, /*sender=*/2, 0,
                         reinterpret_cast<void *>(reduce_value.data()),
                         sizeof(reduce_value[0]), TimeLimit::responsive());
  bool ok = barrier.get_result(result_gen, reinterpret_cast<void *>(result.data()),
                               sizeof(result[0]));

  EXPECT_TRUE(ok);
  EXPECT_EQ(barrier.generation.load(), result_gen);
  EXPECT_EQ(reduce_value[0] + reduce_value[0], result[0]);
}

TEST_F(BarrierRedopTest, GetResultForRemoteTriggeredGen)
{
  NodeID owner = 1;
  auto barrier_id = ID::make_barrier(owner, 0, 0);
  EventImpl::gen_t trigger_gen = 1;
  EventImpl::gen_t previous_gen = 0;
  EventImpl::gen_t first_gen = 0;
  NodeID migration_target = owner;
  unsigned base_count = 2;
  std::vector<int> reduce_value(1, 4);
  std::vector<int> result(1, 0);
  BarrierImpl barrier(barrier_comm);

  int broadcast_index = 0;

  barrier.init(barrier_id, owner);
  barrier.initial_value = std::make_unique<char[]>(sizeof(int));
  barrier.redop = redop;
  barrier.redop_id = redop_id;
  barrier.base_arrival_count = 2;

  const char *reduce_data_ptr = reinterpret_cast<const char *>(reduce_value.data());
  BarrierTriggerPayload payload;
  payload.reduction.insert(payload.reduction.end(), reduce_data_ptr,
                           reduce_data_ptr + sizeof(int) * reduce_value.size());
  size_t payload_size = payload.reduction.size();

  Serialization::DynamicBufferSerializer dbs(payload_size);
  bool ok = dbs & payload;
  assert(ok);

  barrier.handle_remote_trigger(0, 0, trigger_gen, previous_gen, first_gen, redop_id,
                                migration_target, broadcast_index, base_count,
                                dbs.get_buffer(), dbs.bytes_used(),
                                TimeLimit::responsive());

  ok = barrier.get_result(trigger_gen, reinterpret_cast<void *>(result.data()),
                          sizeof(result[0]));

  EXPECT_TRUE(ok);
  EXPECT_EQ(barrier.generation.load(), trigger_gen);
  EXPECT_EQ(reduce_value[0], result[0]);
}

TEST_F(BarrierRedopTest, GetResultRemoteTriggeredGens)
{
  NodeID owner = 1;
  auto barrier_id = ID::make_barrier(owner, 0, 0);
  EventImpl::gen_t trigger_gen = 1;
  EventImpl::gen_t previous_gen = 0;
  EventImpl::gen_t first_gen = 0;
  NodeID migration_target = owner;
  unsigned base_count = 2;
  std::vector<int> reduce_value(1, 4);
  std::vector<int> result_1(1, 0);
  std::vector<int> result_2(1, 0);
  BarrierImpl barrier(barrier_comm);

  int broadcast_index = 0;

  barrier.init(barrier_id, owner);
  barrier.initial_value = std::make_unique<char[]>(sizeof(int));
  barrier.redop = redop;
  barrier.redop_id = redop_id;
  barrier.base_arrival_count = 2;

  const char *reduce_data_ptr = reinterpret_cast<const char *>(reduce_value.data());
  BarrierTriggerPayload payload;
  payload.reduction.insert(payload.reduction.end(), reduce_data_ptr,
                           reduce_data_ptr + sizeof(int) * reduce_value.size());
  size_t payload_size = payload.reduction.size();

  Serialization::DynamicBufferSerializer dbs(payload_size);
  bool ok = dbs & payload;
  assert(ok);

  barrier.handle_remote_trigger(/*sender=*/0, 0, trigger_gen, previous_gen, first_gen,
                                redop_id, migration_target, broadcast_index, base_count,
                                dbs.get_buffer(), dbs.bytes_used(),
                                TimeLimit::responsive());

  barrier.handle_remote_trigger(/*sender=*/2, 0, trigger_gen + 1, previous_gen + 1,
                                first_gen, redop_id, migration_target, broadcast_index,
                                base_count, dbs.get_buffer(), dbs.bytes_used(),
                                TimeLimit::responsive());

  bool ok_gen1 = barrier.get_result(
      trigger_gen, reinterpret_cast<void *>(result_1.data()), sizeof(result_1[0]));
  bool ok_gen2 = barrier.get_result(
      trigger_gen + 1, reinterpret_cast<void *>(result_2.data()), sizeof(result_2[0]));

  EXPECT_TRUE(ok_gen1);
  EXPECT_TRUE(ok_gen2);
  EXPECT_EQ(barrier.generation.load(), trigger_gen + 1);
  EXPECT_EQ(reduce_value[0], result_1[0]);
  EXPECT_EQ(reduce_value[0], result_2[0]);
}
