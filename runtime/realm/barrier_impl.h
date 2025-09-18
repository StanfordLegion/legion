/* Copyright 2024 Stanford University, NVIDIA Corporation
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

// Barrier implementations for Realm

#ifndef REALM_BARRIER_IMPL_H
#define REALM_BARRIER_IMPL_H

#include "realm/event.h"
#include "realm/event_impl.h"
#include "realm/id.h"
#include "realm/nodeset.h"
#include "realm/redop.h"

#include <vector>
#include <map>
#include <memory>

namespace Realm {

  struct BarrierTriggerMessageArgsInternal {
    EventImpl::gen_t trigger_gen = 0;
    EventImpl::gen_t previous_gen = 0;
    EventImpl::gen_t first_generation = 0;
    ReductionOpID redop_id = 0;
    NodeID migration_target = 0;
    unsigned base_arrival_count = 0;
    int broadcast_index = 0;
  };

  struct RemoteNotification;

  struct BarrierTriggerPayload {
    std::vector<RemoteNotification> remotes;
    std::vector<char> reduction;
  };

  struct BarrierTriggerMessageArgs {
    BarrierTriggerMessageArgsInternal internal;
  };

  class BarrierCommunicator {
  public:
    virtual ~BarrierCommunicator() = default;

    virtual void adjust(NodeID target, Barrier barrier, int delta, Event wait_on,
                        NodeID sender, bool forwarded, const void *data, size_t datalen);

    virtual void trigger(NodeID target, ID::IDType barrier_id, const void *data,
                         size_t datalen, size_t max_payload_size = 0);

    virtual void subscribe(NodeID target, ID::IDType barrier_id,
                           EventImpl::gen_t subscribe_gen, NodeID subscriber,
                           bool forwarded);

    virtual size_t recommend_max_payload(NodeID node, size_t size,
                                         bool with_congestion = true);
  };

  struct RemoteNotification {
    NodeID node;
    EventImpl::gen_t trigger_gen, previous_gen;
  };

#ifndef BARRIER_ENABLE_BROADCAST
  struct BarrierTriggerMessage {
    ID::IDType barrier_id;
    EventImpl::gen_t trigger_gen;
    EventImpl::gen_t previous_gen;
    EventImpl::gen_t first_generation;
    ReductionOpID redop_id;
    NodeID migration_target;
    unsigned base_arrival_count;

    static void handle_message(NodeID sender, const BarrierTriggerMessage &msg,
                               const void *data, size_t datalen, TimeLimit work_until);

    static void send_request(NodeID target, ID::IDType barrier_id,
                             EventImpl::gen_t trigger_gen, EventImpl::gen_t previous_gen,
                             EventImpl::gen_t first_generation, ReductionOpID redop_id,
                             NodeID migration_target, unsigned base_arrival_count,
                             const void *data, size_t datalen);
  };
#endif

  class BarrierImpl : public EventImpl {
  public:
    static const ID::ID_Types ID_TYPE = ID::ID_BARRIER;

    static const int BARRIER_TIMESTAMP_NODEID_SHIFT = 48;
    static atomic<Barrier::timestamp_t> barrier_adjustment_timestamp;

    BarrierImpl(void);
    BarrierImpl(BarrierCommunicator *_barrier_comm, int _broadcast_radix = 4);
    ~BarrierImpl(void);

    void init(ID _me, unsigned _init_owner);

    static ID make_id(const BarrierImpl &dummy, int owner, ID::IDType index)
    {
      return ID::make_barrier(owner, index, 0);
    }

    // get the Barrier (id+generation) for the current (i.e. untriggered) generation
    Barrier current_barrier(Barrier::timestamp_t timestamp = 0) const;

    // helper to create the Barrier for an arbitrary generation
    Barrier make_barrier(gen_t gen, Barrier::timestamp_t timestamp = 0) const;

    static BarrierImpl *create_barrier(unsigned expected_arrivals, ReductionOpID redopid,
                                       const void *initial_value = 0,
                                       size_t initial_value_size = 0);

    // test whether an event has triggered without waiting
    virtual bool has_triggered(gen_t needed_gen, bool &poisoned);

    virtual void subscribe(gen_t subscribe_gen);

    virtual void external_wait(gen_t needed_gen, bool &poisoned);
    virtual bool external_timedwait(gen_t needed_gen, bool &poisoned, long long max_ns);

    virtual bool add_waiter(gen_t needed_gen,
                            EventWaiter *waiter /*, bool pre_subscribed = false*/);

    // use this sparingly - it has to hunt through waiter lists while
    //  holding locks
    virtual bool remove_waiter(gen_t needed_gen, EventWaiter *waiter);

    // used to adjust a barrier's arrival count either up or down
    // if delta > 0, timestamp is current time (on requesting node)
    // if delta < 0, timestamp says which positive adjustment this arrival must wait for
    void adjust_arrival(gen_t barrier_gen, int delta, Barrier::timestamp_t timestamp,
                        Event wait_on, NodeID sender, bool forwarded,
                        const void *reduce_value, size_t reduce_value_size,
                        TimeLimit work_until);

    void handle_remote_subscription(NodeID subscriber, EventImpl::gen_t subscribe_gen,
                                    bool forwarded, const void *data, size_t datalen);

    void handle_remote_trigger(NodeID sender, ID::IDType barrier_id,
                               EventImpl::gen_t trigger_gen,
                               EventImpl::gen_t previous_gen, EventImpl::gen_t first_gen,
                               ReductionOpID redop_id, NodeID migration_target,
                               int broadcast_index, unsigned base_count, const void *data,
                               size_t datalen, TimeLimit work_until);

    bool get_result(gen_t result_gen, void *value, size_t value_size);

  protected:
    void broadcast_trigger(const std::vector<RemoteNotification> &ordered_notifications,
                           const std::vector<NodeID> &broadcast_targets,
                           EventImpl::gen_t oldest_previous,
                           EventImpl::gen_t broadcast_previous,
                           EventImpl::gen_t first_generation, NodeID migration_target,
                           unsigned base_arrival_count, ReductionOpID redop_id,
                           const void *data, size_t datalen,
                           bool include_notifications = true);

  public:
    atomic<gen_t> generation = atomic<gen_t>(0);
    atomic<gen_t> gen_subscribed = atomic<gen_t>(0);
    gen_t first_generation = 0;
    BarrierImpl *next_free = nullptr;

    std::unique_ptr<BarrierCommunicator> barrier_comm;

    Mutex mutex; // controls which local thread has access to internal data (not
                 // runtime-visible event)

    // class to track per-generation status
    class Generation {
    public:
      struct PerNodeUpdates {
        Barrier::timestamp_t last_ts;
        std::map<Barrier::timestamp_t, int> pending;
      };

      int unguarded_delta;
      EventWaiter::EventWaiterList local_waiters;
      std::map<int, PerNodeUpdates *> pernode;

      Generation(void);
      ~Generation(void);

      void handle_adjustment(Barrier::timestamp_t ts, int delta);
    };

    std::map<gen_t, Generation *> generations;

    // external waiters on this node are notifies via a condition variable
    bool has_external_waiters = false;
    // use kernel mutex for timedwait functionality
    KernelMutex external_waiter_mutex;
    KernelMutex::CondVar external_waiter_condvar;

    // a list of remote waiters and the latest generation they're interested in
    // also the latest generation that each node (that has ever subscribed) has been told
    // about
    std::map<unsigned, gen_t> remote_subscribe_gens, remote_trigger_gens;
    std::map<gen_t, gen_t> held_triggers;

    unsigned base_arrival_count = 0;
    ReductionOpID redop_id = 0;
    const ReductionOpUntyped *redop = nullptr;
    std::unique_ptr<char[]> initial_value{};
    unsigned value_capacity = 0;
    std::vector<char> final_values;
    bool needs_ordering;
    std::vector<std::pair<int, std::vector<RemoteNotification>>> ordered_buffer;
    int broadcast_radix;
  };
}; // namespace Realm

#include "realm/barrier_impl.inl"

#endif // ifndef REALM_BARRIER_IMPL_H
