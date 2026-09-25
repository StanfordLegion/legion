/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "realm/event_impl.h"
#include "realm/activemsg.h"
#include "realm/operation.h"
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

class EventTestCommunicator : public EventCommunicator {
public:
  virtual void trigger(Event event, NodeID owner, bool poisoned)
  {
    sent_trigger_count++;
    last_trigger_poisoned = poisoned;
  }

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
  bool last_trigger_poisoned = false;
};

// An Operation whose only job is to report when it is destroyed.  Constructed against
//  an event owned by another node it mimics a Task that the owner spawned on this node
//  via SpawnTaskMessage: the operation hands its reference to the finish event and
//  relies on the event to release it when the generation triggers.
class TestOperation : public Operation {
public:
  TestOperation(GenEventImpl *finish_event, EventImpl::gen_t finish_gen, bool *destroyed)
    : Operation(finish_event, finish_gen, ProfilingRequestSet())
    , destroyed(destroyed)
  {}
  virtual ~TestOperation(void) { *destroyed = true; }
  virtual void print(std::ostream &os) const { os << "TestOperation"; }

  // drive the operation through its normal lifecycle to completion
  void run(void)
  {
    mark_ready();
    mark_started();
    mark_finished(true /*successful*/);
  }

  bool *destroyed;
};

// A communicator that plays the part of a remote owner which, on receiving our trigger
//  for one generation, immediately recycles the event and spawns the next generation's
//  operation back on this node.  GenEventImpl::trigger sends the trigger message before
//  it updates local state, so doing the spawn inside trigger() lands the new operation
//  in exactly that window.
class RespawningEventCommunicator : public EventTestCommunicator {
public:
  virtual void trigger(Event event, NodeID owner, bool poisoned)
  {
    EventTestCommunicator::trigger(event, owner, poisoned);
    if((spawned == nullptr) && (event_impl != nullptr)) {
      spawned = new TestOperation(event_impl, respawn_gen, destroyed);
    }
  }

  GenEventImpl *event_impl = nullptr;
  EventImpl::gen_t respawn_gen = 0;
  bool *destroyed = nullptr;
  TestOperation *spawned = nullptr;
};

class GenEventTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    event_comm = new EventTestCommunicator();
    event_notifier = new EventTriggerNotifier();
  }

  void TearDown() override
  {

#ifdef DEBUG_REALM
    // TODO(apryakhin@): consider mocking event notifier.
    event_notifier->shutdown_work_item();
#endif

    delete event_notifier;
  }

  EventTestCommunicator *event_comm;
  EventTriggerNotifier *event_notifier;
};

TEST_F(GenEventTest, GetCurrentEvent)
{
  const NodeID creator_node = 2;
  const int event_index = 7;
  GenEventImpl event(event_notifier, event_comm);

  event.init(ID::make_event(creator_node, event_index, 0), 0);

  EXPECT_EQ(ID(event.current_event()).event_generation(),
            ID::make_event(creator_node, event_index, 1).event_generation());
  EXPECT_EQ(ID(event.current_event()).event_creator_node(), creator_node);
  EXPECT_EQ(ID(event.current_event()).event_gen_event_idx(), event_index);
}

TEST_F(GenEventTest, LocalAddWaiter)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t needed_gen = 1;
  DeferredOperation waiter;
  GenEventImpl event(event_notifier, event_comm);

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
  GenEventImpl event(event_notifier, event_comm);

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
  GenEventImpl event(event_notifier, event_comm);

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
  GenEventImpl event(event_notifier, event_comm);
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
  GenEventImpl event(event_notifier, event_comm);

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
  GenEventImpl event(event_notifier, event_comm);

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
  GenEventImpl event(event_notifier, event_comm);

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
  GenEventImpl event(event_notifier, event_comm);

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
  GenEventImpl event(event_notifier, event_comm);

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
  GenEventImpl event(event_notifier, event_comm);

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
  EventTestCommunicator *event_comm = new EventTestCommunicator();
  GenEventImpl event(event_notifier, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.subscribe(subscribe_gen);

  EXPECT_EQ(event_comm->sent_subscription_count, 1);
}

TEST_F(GenEventTest, RemoteSubscribeCurrGen)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t subscribe_gen = 1;
  EventTestCommunicator *event_comm = new EventTestCommunicator();
  GenEventImpl event(event_notifier, event_comm);

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
  GenEventImpl event(event_notifier, event_comm);

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
  GenEventImpl event(event_notifier, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  bool free_event =
      event.trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive());

  EXPECT_TRUE(event.has_triggered(trigger_gen, poisoned));
  EXPECT_FALSE(poisoned);
  EXPECT_TRUE(free_event);
}

TEST_F(GenEventTest, LocalTriggerWithPoison)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t trigger_gen = 1;
  bool poisoned = false;
  GenEventImpl event(event_notifier, event_comm);
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
  GenEventImpl event(event_notifier, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  bool ok = event.add_waiter(trigger_gen, &waiter_one);
  event.trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive());

  EXPECT_TRUE(ok);
  EXPECT_TRUE(event.has_triggered(trigger_gen, poisoned));
  EXPECT_TRUE(waiter_one.triggered);
}

TEST_F(GenEventTest, LocalTriggerWithZeroTimeLimitPoisoned)
{
  const NodeID owner = 0;
  const GenEventImpl::gen_t trigger_gen = 1;
  bool poisoned = false;
  DeferredOperation waiter_one;
  DeferredOperation waiter_two;
  GenEventImpl event(event_notifier, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  bool ok1 = event.add_waiter(trigger_gen, &waiter_one);
  bool ok2 = event.add_waiter(trigger_gen, &waiter_two);
  event.trigger(trigger_gen, 0, /*poisoned=*/true, TimeLimit::relative(0));

  EXPECT_TRUE(ok1);
  EXPECT_TRUE(ok2);
  EXPECT_TRUE(waiter_one.triggered);
  EXPECT_FALSE(waiter_two.triggered);
  EXPECT_TRUE(event.is_generation_poisoned(trigger_gen));
  EXPECT_TRUE(event.has_triggered(trigger_gen, poisoned));
}

TEST_F(GenEventTest, LocalTriggerWithMultipleRemoteSubscriptions)
{
  const NodeID owner = 0;
  const NodeID sender_a = 1;
  const NodeID sender_b = 2;
  const GenEventImpl::gen_t subscribe_gen = 1;
  GenEventImpl event(event_notifier, event_comm);

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
  GenEventImpl event(event_notifier, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.handle_remote_subscription(sender, subscribe_gen, 0);

  EXPECT_EQ(event_comm->sent_notification_count, 0);
  EXPECT_TRUE(event.remote_waiters.contains(sender));
}

// A task spawned here by the remote owner of its finish event hands its reference to
//  that event; completing the task must release it.  (Regression: the reference was
//  only released in the owner branch of GenEventImpl::trigger, leaking every such Task
//  and its argument buffer until the same event slot happened to be reused.)
TEST_F(GenEventTest, RemoteTriggerReleasesTriggerOp)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t trigger_gen = 1;
  bool poisoned = false;
  bool destroyed = false;
  GenEventImpl event(event_notifier, event_comm);
  event.init(ID::make_event(0, 0, 0), owner);

  TestOperation *op = new TestOperation(&event, trigger_gen, &destroyed);
  ASSERT_EQ(event.current_trigger_op, op);
  EXPECT_EQ(event.current_trigger_op_gen, trigger_gen);
  EXPECT_FALSE(destroyed);

  op->run();

  EXPECT_EQ(event_comm->sent_trigger_count, 1);
  EXPECT_TRUE(event.has_triggered(trigger_gen, poisoned));
  EXPECT_FALSE(poisoned);
  EXPECT_EQ(event.current_trigger_op, nullptr);
  EXPECT_TRUE(destroyed);
}

// Our copy of a remote event only learns about generations through local triggers and
//  owner updates, so the owner may hand us an operation for a generation well ahead of
//  our 'generation'.  It must still be tracked (so cancellation can find it) and
//  released when that generation triggers.
TEST_F(GenEventTest, RemoteTriggerOpWithLaggingGeneration)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t trigger_gen = 3;
  bool poisoned = false;
  bool destroyed = false;
  GenEventImpl event(event_notifier, event_comm);
  event.init(ID::make_event(0, 0, 0), owner);

  TestOperation *op = new TestOperation(&event, trigger_gen, &destroyed);
  ASSERT_EQ(event.current_trigger_op, op);
  EXPECT_EQ(event.current_trigger_op_gen, trigger_gen);

  Operation *found = event.get_trigger_op(trigger_gen);
  EXPECT_EQ(found, op);
  if(found != nullptr)
    found->remove_reference();
  EXPECT_EQ(event.get_trigger_op(trigger_gen - 1), nullptr);
  EXPECT_FALSE(destroyed);

  op->run();

  EXPECT_EQ(event_comm->sent_trigger_count, 1);
  EXPECT_TRUE(event.has_triggered(trigger_gen, poisoned));
  EXPECT_EQ(event.current_trigger_op, nullptr);
  EXPECT_TRUE(destroyed);
}

// trigger() tells the owner about a generation before updating our local state, so the
//  owner can recycle the event and spawn the next generation's task here before our
//  'generation' advances.  Installing the newer operation must release the older,
//  already-triggered one, and the delayed local half of the older generation's trigger
//  must leave the newer operation alone.
TEST_F(GenEventTest, RemoteTriggerOpReplacedBeforeLocalUpdate)
{
  const NodeID owner = 1;
  bool poisoned = false;
  bool destroyed_one = false;
  bool destroyed_two = false;
  RespawningEventCommunicator *comm = new RespawningEventCommunicator();
  GenEventImpl event(event_notifier, comm);
  event.init(ID::make_event(0, 0, 0), owner);
  comm->event_impl = &event;
  comm->respawn_gen = 2;
  comm->destroyed = &destroyed_two;

  TestOperation *op_one = new TestOperation(&event, 1, &destroyed_one);
  ASSERT_EQ(event.current_trigger_op, op_one);

  // completing op_one sends the trigger for generation 1; the "owner" reacts inside
  //  that send by spawning generation 2's operation here, before generation 1's local
  //  state update has run
  op_one->run();

  ASSERT_NE(comm->spawned, nullptr);
  EXPECT_EQ(comm->sent_trigger_count, 1);
  EXPECT_TRUE(destroyed_one);
  EXPECT_FALSE(destroyed_two);
  EXPECT_EQ(event.current_trigger_op, comm->spawned);
  EXPECT_EQ(event.current_trigger_op_gen, GenEventImpl::gen_t(2));
  EXPECT_TRUE(event.has_triggered(1, poisoned));
  EXPECT_FALSE(event.has_triggered(2, poisoned));

  comm->spawned->run();

  EXPECT_EQ(comm->sent_trigger_count, 2);
  EXPECT_TRUE(event.has_triggered(2, poisoned));
  EXPECT_EQ(event.current_trigger_op, nullptr);
  EXPECT_TRUE(destroyed_two);
}

// Poison in a dynamically-added finish precondition must not destroy an operation or
// report its finish event until the operation's main work item has also finished.
TEST_F(GenEventTest, RemoteFinishPreconditionPoisonWaitsForOperation)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t trigger_gen = 1;
  bool poisoned = false;
  bool destroyed = false;
  GenEventImpl event(event_notifier, event_comm);
  event.init(ID::make_event(0, 0, 0), owner);

  TestOperation *op = new TestOperation(&event, trigger_gen, &destroyed);
  ASSERT_TRUE(op->mark_ready());
  ASSERT_TRUE(op->mark_started());

  // Model a finish precondition that was registered by the running operation and then
  // poisoned.  The merger's initial count represents the still-running main work item.
  event.merger.get_next_precondition()->event_triggered(true /*poisoned*/,
                                                        TimeLimit::responsive());

  EXPECT_FALSE(event.has_triggered(trigger_gen, poisoned));
  EXPECT_FALSE(destroyed);
  EXPECT_EQ(event.current_trigger_op, op);
  EXPECT_EQ(event_comm->sent_trigger_count, 0);

  op->mark_finished(true /*successful*/);

  EXPECT_TRUE(event.has_triggered(trigger_gen, poisoned));
  EXPECT_TRUE(event_comm->last_trigger_poisoned);
  EXPECT_EQ(event.current_trigger_op, nullptr);
  EXPECT_TRUE(destroyed);
}

TEST_F(GenEventTest, LocalEventMergerDefersPoisonUntilAllPreconditions)
{
  const NodeID owner = 0;
  // Prevent this stack-allocated event from being returned to the runtime's free list.
  const GenEventImpl::gen_t trigger_gen = (1U << ID::EVENT_GENERATION_WIDTH) - 1;
  bool poisoned = false;
  GenEventImpl event(event_notifier, event_comm);
  event.init(ID::make_event(0, 0, 0), owner);
  event.generation.store(trigger_gen - 1);
  event.merger.prepare_merger(event.make_event(trigger_gen), false /*ignore faults*/,
                              std::optional<size_t>(),
                              EventMerger::FaultPropagation::AFTER_PRECONDITIONS);

  event.merger.get_next_precondition()->event_triggered(true /*poisoned*/,
                                                        TimeLimit::responsive());

  EXPECT_FALSE(event.has_triggered(trigger_gen, poisoned));

  // The arm models the operation's main work item reaching completion.
  event.merger.arm_merger();

  EXPECT_TRUE(event.has_triggered(trigger_gen, poisoned));
  EXPECT_TRUE(poisoned);
}

// The delayed policy above is operation-specific.  Ordinary event mergers retain their
// existing eager poison propagation behavior.
TEST_F(GenEventTest, EventMergerStillPropagatesPoisonEarlyByDefault)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t trigger_gen = 1;
  bool poisoned = false;
  GenEventImpl event(event_notifier, event_comm);
  event.init(ID::make_event(0, 0, 0), owner);
  event.merger.prepare_merger(event.make_event(trigger_gen), false /*ignore faults*/);

  event.merger.get_next_precondition()->event_triggered(true /*poisoned*/,
                                                        TimeLimit::responsive());

  EXPECT_TRUE(event.has_triggered(trigger_gen, poisoned));
  EXPECT_TRUE(event_comm->last_trigger_poisoned);

  // Supply the merger's arm count so it can retire its precondition storage.
  event.merger.arm_merger();
}

TEST_F(GenEventTest, RemoteTrigger)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t trigger_gen = 1;
  GenEventImpl event(event_notifier, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive());

  EXPECT_EQ(event_comm->sent_trigger_count, 1);
}

TEST_F(GenEventTest, RemoteTriggerWithWaiters)
{
  const NodeID owner = 1;
  const GenEventImpl::gen_t trigger_gen = 1;
  GenEventImpl event(event_notifier, event_comm);
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
  GenEventImpl event(event_notifier, event_comm);

  event.init(ID::make_event(0, 0, 0), owner);
  event.trigger(trigger_gen, 0, /*poisoned=*/false, TimeLimit::responsive());

  EXPECT_EQ(event_comm->sent_trigger_count, 1);
  EXPECT_EQ(event_comm->sent_subscription_count, 1);
}

TEST_F(GenEventTest, EventMergerIsActive)
{
  const NodeID owner = 0;
  GenEventImpl event(event_notifier, event_comm);
  event.init(ID::make_event(0, 0, 0), owner);
  EventMerger merger(&event);

  bool ok = merger.is_active();

  EXPECT_FALSE(ok);
}
