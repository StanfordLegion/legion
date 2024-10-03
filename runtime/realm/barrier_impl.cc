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

#include "realm/event_impl.h"
#include "realm/barrier_impl.h"

#include "realm/proc_impl.h"
#include "realm/runtime_impl.h"
#include "realm/logging.h"
#include "realm/threads.h"
#include "realm/profiling.h"

#define DISABLE_BARRIER_MIGRATION

namespace BarrierConfig {
  int reduction_radix = 2;
  int broadcast_radix = 2;
  int max_arrivals_payload = 256;
  int max_notifies_payload = 256;
}; // namespace BarrierConfig

TYPE_IS_SERIALIZABLE(Realm::RemoteNotification);

namespace Realm {

  Logger log_barrier("barrier");
  Logger log_barrier_scale("barrier_scale");

  // used in places that don't currently propagate poison but should
  static const bool POISON_FIXME = false;

  ////////////////////////////////////////////////////////////////////////
  //
  // class Barrier
  //

  /*static*/ const Barrier Barrier::NO_BARRIER = {/* zero-initialization */};

  /*static*/ const ::realm_event_gen_t Barrier::MAX_PHASES =
      (::realm_event_gen_t(1) << REALM_EVENT_GENERATION_BITS) - 1;

  /*static*/ Barrier Barrier::create_barrier(unsigned expected_arrivals,
                                             ReductionOpID redop_id /*= 0*/,
                                             const void *initial_value /*= 0*/,
                                             size_t initial_value_size /*= 0*/)
  {
    BarrierImpl *impl = BarrierImpl::create_barrier(expected_arrivals, redop_id,
                                                    initial_value, initial_value_size);
    Barrier b = impl->current_barrier();

    return b;
  }

  void Barrier::destroy_barrier(void)
  {
    log_barrier.info() << "barrier destruction request: " << *this;
  }

  Barrier Barrier::advance_barrier(void) const
  {
    ID nextid(id);
    EventImpl::gen_t gen = ID(id).barrier_generation() + 1;
#ifdef DEBUG_REALM
    assert(MAX_PHASES <= nextid.barrier_generation().MAXVAL);
#endif
    // return NO_BARRIER if the count overflows
    if(gen > MAX_PHASES)
      return Barrier::NO_BARRIER;
    nextid.barrier_generation() = ID(id).barrier_generation() + 1;

    Barrier nextgen = nextid.convert<Barrier>();
    nextgen.timestamp = 0;

    return nextgen;
  }

  /*static*/ atomic<Barrier::timestamp_t> BarrierImpl::barrier_adjustment_timestamp(0);

  Barrier Barrier::alter_arrival_count(int delta) const
  {
    timestamp_t timestamp = BarrierImpl::barrier_adjustment_timestamp.fetch_add(1);
    BarrierImpl *impl = get_runtime()->get_barrier_impl(*this);
    impl->adjust_arrival(ID(id).barrier_generation(), delta, timestamp, Event::NO_EVENT,
                         Network::my_node_id, false /*!forwarded*/, 0, 0,
                         TimeLimit::responsive());

    Barrier with_ts;
    with_ts.id = id;
    with_ts.timestamp = timestamp;

    return with_ts;
  }

  Barrier Barrier::get_previous_phase(void) const
  {
    ID previd(id);
    EventImpl::gen_t gen = ID(id).barrier_generation();
    // can't back up before generation 0
    previd.barrier_generation() = ((gen > 0) ? (gen - 1) : gen);

    Barrier prevgen = previd.convert<Barrier>();
    prevgen.timestamp = 0;

    return prevgen;
  }

  void Barrier::arrive(unsigned count /*= 1*/, Event wait_on /*= Event::NO_EVENT*/,
                       const void *reduce_value /*= 0*/,
                       size_t reduce_value_size /*= 0*/) const
  {
    // arrival uses the timestamp stored in this barrier object
    BarrierImpl *impl = get_runtime()->get_barrier_impl(*this);
    impl->adjust_arrival(ID(id).barrier_generation(), -int(count), timestamp, wait_on,
                         Network::my_node_id, false /*!forwarded*/, reduce_value,
                         reduce_value_size, TimeLimit::responsive());
  }

  bool Barrier::get_result(void *value, size_t value_size) const
  {
    BarrierImpl *impl = get_runtime()->get_barrier_impl(*this);
    return impl->get_result(ID(id).barrier_generation(), value, value_size);
  }

  /*static*/ BarrierImpl *BarrierImpl::create_barrier(unsigned expected_arrivals,
                                                      ReductionOpID redopid,
                                                      const void *initial_value /*= 0*/,
                                                      size_t initial_value_size /*= 0*/)
  {
    BarrierImpl *impl = get_runtime()->local_barrier_free_list->alloc_entry();
    assert(impl);
    assert(ID(impl->me).is_barrier());

    // set the arrival count
    impl->base_arrival_count = expected_arrivals;

    if(redopid == 0) {
      assert(initial_value_size == 0);
      impl->redop_id = 0;
      impl->redop = 0;
      impl->initial_value = 0;
      impl->value_capacity = 0;
      impl->final_values = 0;
    } else {
      impl->redop_id = redopid; // keep the ID too so we can share it
      impl->redop = get_runtime()->reduce_op_table.get(redopid, 0);
      if(impl->redop == 0) {
        log_barrier.fatal() << "no reduction op registered for ID " << redopid;
        abort();
      }

      assert(initial_value != 0);
      assert(initial_value_size == impl->redop->sizeof_lhs);

      impl->initial_value = (char *)malloc(initial_value_size);
      memcpy(impl->initial_value, initial_value, initial_value_size);

      impl->value_capacity = 0;
      impl->final_values = 0;
    }

    // and let the barrier rearm as many times as necessary without being released
    // impl->free_generation = (unsigned)-1;

    log_barrier.info() << "barrier created: " << impl->me << "/"
                       << impl->generation.load()
                       << " base_count=" << impl->base_arrival_count
                       << " redop=" << redopid;
#ifdef EVENT_TRACING
    {
      EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
      item.event_id = impl->me.id();
      item.event_gen = impl->me.gen;
      item.action = EventTraceItem::ACT_CREATE;
    }
#endif
    return impl;
  }

  BarrierImpl::BarrierImpl(void)
    : generation(0)
    , gen_subscribed(0)
    , has_external_waiters(false)
    , external_waiter_condvar(external_waiter_mutex)
  {
    first_generation = /*free_generation =*/0;
    next_free = 0;
    remote_subscribe_gens.clear();
    remote_trigger_gens.clear();
    base_arrival_count = 0;
    redop = 0;
    initial_value = 0;
    value_capacity = 0;
    final_values = 0;
  }

  BarrierImpl::~BarrierImpl(void)
  {
    if(initial_value)
      free(initial_value);
    if(final_values)
      free(final_values);
  }

  void BarrierImpl::init(ID _me, unsigned _init_owner)
  {
    me = _me;
    owner = _init_owner;
    gen_subscribed.store(0);
    first_generation = /*free_generation =*/0;
    next_free = 0;
    remote_subscribe_gens.clear();
    remote_trigger_gens.clear();
    base_arrival_count = 0;
    redop = 0;
    initial_value = 0;
    value_capacity = 0;
    final_values = 0;
    generation.store_release(0);
  }

  /*static*/ void BarrierAdjustMessage::handle_message(NodeID sender,
                                                       const BarrierAdjustMessage &args,
                                                       const void *data, size_t datalen,
                                                       TimeLimit work_until)
  {
    log_barrier.info() << "received barrier arrival: delta=" << args.delta
                       << " in=" << args.wait_on << " out=" << args.barrier << " ("
                       << args.barrier.timestamp << ")";
    BarrierImpl *impl = get_runtime()->get_barrier_impl(args.barrier);
    EventImpl::gen_t gen = ID(args.barrier).barrier_generation();
    impl->adjust_arrival(gen, args.delta, args.barrier.timestamp, args.wait_on,
                         args.sender, args.forwarded, datalen ? data : 0, datalen,
                         work_until);
  }

  /*static*/ void BarrierAdjustMessage::send_request(NodeID target, Barrier barrier,
                                                     int delta, Event wait_on,
                                                     NodeID sender, bool forwarded,
                                                     const void *data, size_t datalen)
  {
    ActiveMessage<BarrierAdjustMessage> amsg(target, datalen);
    amsg->barrier = barrier;
    amsg->delta = delta;
    amsg->wait_on = wait_on;
    amsg->sender = sender;
    amsg->forwarded = forwarded;
    amsg.add_payload(data, datalen);
    amsg.commit();
  }

  /*static*/ void BarrierSubscribeMessage::send_request(NodeID target,
                                                        ID::IDType barrier_id,
                                                        EventImpl::gen_t subscribe_gen,
                                                        NodeID subscriber, bool forwarded)
  {
    ActiveMessage<BarrierSubscribeMessage> amsg(target);
    amsg->subscriber = subscriber;
    amsg->forwarded = forwarded;
    amsg->barrier_id = barrier_id;
    amsg->subscribe_gen = subscribe_gen;
    amsg.commit();
  }

  /*static*/ void BarrierTriggerMessage::send_request(
      NodeID target, ID::IDType barrier_id, EventImpl::gen_t trigger_gen,
      EventImpl::gen_t previous_gen, EventImpl::gen_t first_generation,
      ReductionOpID redop_id, NodeID migration_target, unsigned base_arrival_count,
      int broadcast_index, const void *data, size_t datalen,
      EventImpl::gen_t invalid_reduction_gen)
  {
    BarrierTriggerMessageArgs trigger_args;
    trigger_args.internal.trigger_gen = trigger_gen;
    trigger_args.internal.previous_gen = previous_gen;
    trigger_args.internal.first_generation = first_generation;
    trigger_args.internal.redop_id = redop_id;
    trigger_args.internal.migration_target = migration_target;
    trigger_args.internal.base_arrival_count = base_arrival_count;
    trigger_args.internal.broadcast_index = broadcast_index;
    trigger_args.internal.invalid_reduction_gen = invalid_reduction_gen;
    Serialization::DynamicBufferSerializer dbs(datalen);
    bool ok = dbs & trigger_args;
    assert(ok);
    if(datalen > 0) {
      ok = dbs.append_bytes(data, datalen);
    }

    size_t total_payload_size = dbs.bytes_used();

    ActiveMessage<BarrierTriggerMessage> amsg(target, total_payload_size);
    amsg->barrier_id = barrier_id;
    amsg.add_payload(dbs.get_buffer(), total_payload_size);

    ID id(barrier_id);
    id.barrier_generation() = trigger_gen;
    Barrier b = id.convert<Barrier>();
    BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

    amsg.commit();
  }

  /*static*/ void
  BarrierTriggerMessage::send_request(NodeID target, ID::IDType barrier_id,
                                      BarrierTriggerMessageArgs &trigger_args,
                                      const void *data, size_t datalen)
  {
    Serialization::DynamicBufferSerializer dbs(datalen);
    bool ok = dbs & trigger_args;
    assert(ok);
    if(datalen > 0) {
      ok = dbs.append_bytes(data, datalen);
    }
    size_t total_payload_size = dbs.bytes_used();
    ActiveMessage<BarrierTriggerMessage> amsg(target, total_payload_size);
    amsg->barrier_id = barrier_id;
    amsg.add_payload(dbs.get_buffer(), total_payload_size);

    ID id(barrier_id);
    id.barrier_generation() = trigger_args.internal.trigger_gen;
    Barrier b = id.convert<Barrier>();
    BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

    amsg.commit();
  }

  // like strdup, but works on arbitrary byte arrays
  static void *bytedup(const void *data, size_t datalen)
  {
    if(datalen == 0)
      return 0;
    void *dst = malloc(datalen);
    assert(dst != 0);
    memcpy(dst, data, datalen);
    return dst;
  }

  class DeferredBarrierArrival : public EventWaiter {
  public:
    DeferredBarrierArrival(Barrier _barrier, int _delta, NodeID _sender, bool _forwarded,
                           const void *_data, size_t _datalen)
      : barrier(_barrier)
      , delta(_delta)
      , sender(_sender)
      , forwarded(_forwarded)
      , data(bytedup(_data, _datalen))
      , datalen(_datalen)
    {}

    virtual ~DeferredBarrierArrival(void)
    {
      if(data)
        free(data);
    }

    virtual void event_triggered(bool poisoned, TimeLimit work_until)
    {
      // TODO: handle poison
      assert(poisoned == POISON_FIXME);
      log_barrier.info() << "deferred barrier arrival: " << barrier << " ("
                         << barrier.timestamp << "), delta=" << delta;
      BarrierImpl *impl = get_runtime()->get_barrier_impl(barrier);
      impl->adjust_arrival(ID(barrier).barrier_generation(), delta, barrier.timestamp,
                           Event::NO_EVENT, sender, forwarded, data, datalen, work_until);
      // not attached to anything, so delete ourselves when we're done
      delete this;
    }

    virtual void print(std::ostream &os) const
    {
      os << "deferred arrival: barrier=" << barrier << " (" << barrier.timestamp << ")"
         << ", delta=" << delta << " datalen=" << datalen;
    }

    virtual Event get_finish_event(void) const { return barrier; }

  protected:
    Barrier barrier;
    int delta;
    NodeID sender;
    bool forwarded;
    void *data;
    size_t datalen;
  };

  BarrierImpl::Generation::Generation(void)
    : unguarded_delta(0)
  {}
  BarrierImpl::Generation::~Generation(void)
  {
    for(std::map<int, PerNodeUpdates *>::iterator it = pernode.begin();
        it != pernode.end(); it++)
      delete(it->second);
  }

  void BarrierImpl::Generation::handle_adjustment(Barrier::timestamp_t ts, int delta)
  {
    if(ts == 0) {
      // simple case - apply delta directly
      unguarded_delta += delta;
      return;
    }

    int node = ts >> BARRIER_TIMESTAMP_NODEID_SHIFT;
    PerNodeUpdates *pn;
    std::map<int, PerNodeUpdates *>::iterator it = pernode.find(node);
    if(it != pernode.end()) {
      pn = it->second;
    } else {
      pn = new PerNodeUpdates;
      pernode[node] = pn;
    }
    if(delta > 0) {
      // TODO: really need two timestamps to properly order increments
      unguarded_delta += delta;
      pn->last_ts = ts;
      std::map<Barrier::timestamp_t, int>::iterator it2 = pn->pending.begin();
      while((it2 != pn->pending.end()) && (it2->first <= pn->last_ts)) {
        log_barrier.info("applying pending delta: %llx/%d", it2->first, it2->second);
        unguarded_delta += it2->second;
        pn->pending.erase(it2);
        it2 = pn->pending.begin();
      }
    } else {
      // if the timestamp is late enough, we can apply this directly
      if(ts <= pn->last_ts) {
        log_barrier.info("adjustment can be applied immediately: %llx/%d (%llx)", ts,
                         delta, pn->last_ts);
        unguarded_delta += delta;
      } else {
        log_barrier.info("adjustment must be deferred: %llx/%d (%llx)", ts, delta,
                         pn->last_ts);
        pn->pending[ts] += delta;
      }
    }
  }

  static inline void get_broadcast_targets(NodeID local, int num_peers, int radix,
                                           std::vector<NodeID> &broadcast_targets)
  {
    assert(broadcast_targets.empty());
    for(int i = 1; i <= radix; i++) {
      NodeID target = local * radix + i;
      if(num_peers <= (target - 1))
        break;
      broadcast_targets.push_back(target - 1);
    }
  }

  static void broadcast_trigger(Barrier barrier, bool include_notification_payload,
                                const std::vector<RemoteNotification> &notifications,
                                const std::vector<NodeID> &broadcast_targets,
                                EventImpl::gen_t oldest_previous,
                                EventImpl::gen_t broadcast_previous,
                                EventImpl::gen_t first_generation,
                                NodeID migration_target, unsigned base_arrival_count,
                                ReductionOpID redop_id, const void *data, size_t datalen)
  {
    BarrierImpl *impl = get_barrier_impl(barrier);
    for(const NodeID target : broadcast_targets) {
      EventImpl::gen_t trigger_gen = notifications[target].trigger_gen;

      const void *reduce_data = nullptr;
      size_t reduce_data_size = 0;

      if(datalen == 0 && redop_id != 0) {
        reduce_data = (char *)data +
                      ((broadcast_previous - oldest_previous) * impl->redop->sizeof_lhs);
        reduce_data_size = (trigger_gen - broadcast_previous) * impl->redop->sizeof_lhs;
      } else {
        reduce_data = data;
        reduce_data_size = datalen;
      }

      BarrierTriggerMessageArgs trigger_args;
      trigger_args.internal.trigger_gen = trigger_gen;
      trigger_args.internal.previous_gen = notifications[target].previous_gen;
      trigger_args.internal.first_generation = first_generation;
      trigger_args.internal.redop_id = redop_id;
      trigger_args.internal.migration_target = migration_target;
      trigger_args.internal.base_arrival_count = base_arrival_count;
      trigger_args.internal.broadcast_index = target + 1;

      const size_t max_recommended_payload = Network::recommended_max_payload(
          notifications[target].node, /*with_congestion=*/true,
          sizeof(BarrierTriggerMessage));

      assert(((long long)max_recommended_payload - (long long)reduce_data_size -
              (long long)sizeof(BarrierTriggerMessageArgsInternal) -
              (long long)sizeof(size_t)) > 0);

      const size_t max_notifications =
          std::min(size_t(BarrierConfig::max_notifies_payload),
                   (max_recommended_payload - reduce_data_size -
                    sizeof(BarrierTriggerMessageArgsInternal) - sizeof(size_t)) /
                       sizeof(RemoteNotification));

      for(size_t start = 0; start < notifications.size(); start += max_notifications) {
        size_t end = std::min(start + max_notifications, notifications.size());

        if(include_notification_payload) {
          std::vector<RemoteNotification> chunk_remote_notifications(
              notifications.begin() + start, notifications.begin() + end);
          trigger_args.remote_notifications = chunk_remote_notifications;
        }

        trigger_args.internal.is_complete_list = (end == notifications.size());

        log_barrier_scale.info()
            << "trigger broadcast N:" << Network::my_node_id << " index:" << target
            << " target:" << notifications[target].node
            << " trigger_gen:" << notifications[target].trigger_gen
            << " datalen:" << datalen
            << " prev_gen:" << notifications[target].previous_gen
            << " bcast_prev_gen:" << broadcast_previous
            << " max_node:" << Network::max_node_id;

        BarrierTriggerMessage::send_request(notifications[target].node, barrier.id,
                                            trigger_args, reduce_data, reduce_data_size);
      }
    }
  }

  // used to adjust a barrier's arrival count either up or down
  // if delta > 0, timestamp is current time (on requesting node)
  // if delta < 0, timestamp says which positive adjustment this arrival must wait for
  void BarrierImpl::adjust_arrival(gen_t barrier_gen, int delta,
                                   Barrier::timestamp_t timestamp, Event wait_on,
                                   NodeID sender, bool forwarded,
                                   const void *reduce_value, size_t reduce_value_size,
                                   TimeLimit work_until)
  {
    Barrier b = make_barrier(barrier_gen, timestamp);
    if(!wait_on.has_triggered()) {
      // deferred arrival

      // only forward deferred arrivals if the precondition is not one that looks like
      // it'll
      //  trigger here first
      if(owner != Network::my_node_id) {
        ID wait_id(wait_on);
        int wait_node;
        if(wait_id.is_event())
          wait_node = wait_id.event_creator_node();
        else
          wait_node = wait_id.barrier_creator_node();
        if(wait_node != (int)Network::my_node_id) {
          // let deferral happen on owner node (saves latency if wait_on event
          //   gets triggered there)
          // printf("sending deferred arrival to %d for " IDFMT "/%d (" IDFMT "/%d)\n",
          //       owner, e.id, e.gen, wait_on.id, wait_on.gen);
          log_barrier.info() << "forwarding deferred barrier arrival: delta=" << delta
                             << " in=" << wait_on << " out=" << b << " (" << timestamp
                             << ")";
          BarrierAdjustMessage::send_request(owner, b, delta, wait_on, sender,
                                             (sender != Network::my_node_id),
                                             reduce_value, reduce_value_size);
          return;
        }
      }

      log_barrier.info() << "deferring barrier arrival: delta=" << delta
                         << " in=" << wait_on << " out=" << b << " (" << timestamp << ")";
      EventImpl::add_waiter(wait_on,
                            new DeferredBarrierArrival(b, delta, sender, forwarded,
                                                       reduce_value, reduce_value_size));
      return;
    }

    log_barrier.info() << "barrier adjustment: event=" << b << " delta=" << delta
                       << " ts=" << timestamp;

#ifdef DEBUG_BARRIER_REDUCTIONS
    if(reduce_value_size) {
      char buffer[129];
      for(size_t i = 0; (i < reduce_value_size) && (i < 64); i++)
        snprintf(buffer + 2 * i, sizeof buffer - 2 * i, "%02x",
                 ((const unsigned char *)reduce_value)[i]);
      log_barrier.info("barrier reduction: event=" IDFMT "/%d size=%zd data=%s", me.id(),
                       barrier_gen, reduce_value_size, buffer);
    }
#endif

    // can't actually trigger while holding the lock, so remember which generation(s),
    //  if any, to trigger and do it at the end
    gen_t trigger_gen = 0;
    EventWaiter::EventWaiterList local_notifications;
    std::vector<RemoteNotification> remote_notifications;
    std::vector<NodeID> remote_broadcast_targets;
    gen_t broadcast_previous = 0;
    gen_t oldest_previous = 0;
    void *final_values_copy = 0;
    NodeID migration_target = (NodeID)-1;
    NodeID forward_to_node = (NodeID)-1;
    NodeID inform_migration = (NodeID)-1;

    do { // so we can use 'break' from the middle
      AutoLock<> a(mutex);

      bool generation_updated = false;

      // ownership can change, so check it inside the lock
      if(owner != Network::my_node_id) {
        forward_to_node = owner;
        break;
      } else {
        // if this message had to be forwarded to get here, tell the original sender we
        // are the
        //  new owner
        if(forwarded && (sender != Network::my_node_id))
          inform_migration = sender;
      }

      // sanity checks - is this a valid barrier?
      // assert(generation < free_generation);
      assert(base_arrival_count > 0);

      // update whatever generation we're told to
      {
        assert(barrier_gen > generation.load());
        Generation *g;
        std::map<gen_t, Generation *>::iterator it = generations.find(barrier_gen);
        if(it != generations.end()) {
          g = it->second;
        } else {
          g = new Generation;
          generations[barrier_gen] = g;
          log_barrier.info() << "added tracker for barrier " << me << ", generation "
                             << barrier_gen;
        }

        g->handle_adjustment(timestamp, delta);
      }

      // if the update was to the next generation, it may cause one or more generations
      //  to trigger
      if(barrier_gen == (generation.load() + 1)) {
        std::map<gen_t, Generation *>::iterator it = generations.begin();
        while((it != generations.end()) && (it->first == (generation.load() + 1)) &&
              ((base_arrival_count + it->second->unguarded_delta) == 0)) {
          // keep the list of local waiters to wake up once we release the lock
          local_notifications.absorb_append(it->second->local_waiters);
          trigger_gen = it->first;
          generation.store_release(it->first);
          generation_updated = true;

          delete it->second;
          generations.erase(it);
          it = generations.begin();
        }

        // if any triggers occurred, figure out
        //  which remote nodes need notifications (i.e. any who subscribed)
        if(trigger_gen >= barrier_gen) {
          std::map<unsigned, gen_t>::iterator it = remote_subscribe_gens.begin();
          while(it != remote_subscribe_gens.end()) {
            RemoteNotification rn;
            rn.node = it->first;
            if(it->second <= trigger_gen) {
              // we have fulfilled the entire subscription
              rn.trigger_gen = it->second;
              std::map<unsigned, gen_t>::iterator to_nuke = it++;
              remote_subscribe_gens.erase(to_nuke);
            } else {
              // subscription remains valid
              rn.trigger_gen = trigger_gen;
              ++it;
            }
            // also figure out what the previous generation this node knew about was
            {
              std::map<unsigned, gen_t>::iterator it2 = remote_trigger_gens.find(rn.node);
              if(it2 != remote_trigger_gens.end()) {
                rn.previous_gen = it2->second;
                it2->second = rn.trigger_gen;
              } else {
                rn.previous_gen = first_generation;
                remote_trigger_gens[rn.node] = rn.trigger_gen;
              }
            }
            if(remote_notifications.empty() || (rn.previous_gen < oldest_previous))
              oldest_previous = rn.previous_gen;
            remote_notifications.push_back(rn);

            if(remote_notifications.size() == 1) {
              broadcast_previous = rn.previous_gen;
            } else {
              broadcast_previous = std::min(broadcast_previous, rn.previous_gen);
            }
          }
        }

#ifndef DISABLE_BARRIER_MIGRATION
        // if there were zero local waiters and a single remote waiter, this barrier is
        // an obvious
        //  candidate for migration
        // don't migrate a barrier more than once though (i.e. only if it's on the
        // creator node still) also, do not migrate a barrier if we have any local
        // involvement in future generations
        //  (either arrivals or waiters or a subscription that will become a waiter)
        // finally (hah!), do not migrate barriers using reduction ops
        if(local_notifications.empty() && (remote_notifications.size() == 1) &&
           generations.empty() && (gen_subscribed.load() <= generation.load()) &&
           (redop == 0) &&
           (NodeID(ID(me).barrier_creator_node()) == Network::my_node_id)) {
          log_barrier.info() << "barrier migration: " << me << " -> "
                             << remote_notifications[0].node;
          migration_target = remote_notifications[0].node;
          owner = migration_target;
          // remember that we had up to date information up to this generation so that
          // we don't try to
          //   subscribe to things we already know about
          gen_subscribed.store(generation.load());
        }
#endif
      }

      // do we have reduction data to apply?  we can do this even if the actual
      // adjustment is
      //  being held - no need to have lots of reduce values lying around
      if(reduce_value_size > 0) {
        assert(redop != 0);
        assert(redop->sizeof_rhs == reduce_value_size);

        // do we have space for this reduction result yet?
        int rel_gen = barrier_gen - first_generation;
        assert(rel_gen > 0);

        if((size_t)rel_gen > value_capacity) {
          size_t new_capacity = rel_gen;
          final_values = (char *)realloc(final_values, new_capacity * redop->sizeof_lhs);
          while(value_capacity < new_capacity) {
            memcpy(final_values + (value_capacity * redop->sizeof_lhs), initial_value,
                   redop->sizeof_lhs);
            value_capacity += 1;
          }
        }

        (redop->cpu_apply_excl_fn)(final_values + ((rel_gen - 1) * redop->sizeof_lhs), 0,
                                   reduce_value, 0, 1, redop->userdata);
      }

      // do this AFTER we actually update the reduction value above :)
      // if any remote notifications are going to occur and we have reduction values,
      // make a copy so
      //  we have something stable after we let go of the lock
      if(trigger_gen && redop) {
        int rel_gen = oldest_previous + 1 - first_generation;
        assert(rel_gen > 0);
        int count = trigger_gen - oldest_previous;
        final_values_copy = bytedup(final_values + ((rel_gen - 1) * redop->sizeof_lhs),
                                    count * redop->sizeof_lhs);
      }

      // external waiters need to be signalled inside the lock
      if(generation_updated && has_external_waiters) {
        has_external_waiters = false;
        // also need external waiter mutex
        AutoLock<KernelMutex> al2(external_waiter_mutex);
        external_waiter_condvar.broadcast();
      }

      NodeID node = (Network::my_node_id - owner + Network::max_node_id + 1) %
                    (Network::max_node_id + 1);
      get_broadcast_targets(node, remote_notifications.size(),
                            BarrierConfig::broadcast_radix, remote_broadcast_targets);
    } while(0);

    if(forward_to_node != (NodeID)-1) {
      Barrier b = make_barrier(barrier_gen, timestamp);
      BarrierAdjustMessage::send_request(forward_to_node, b, delta, Event::NO_EVENT,
                                         sender, (sender != Network::my_node_id),
                                         reduce_value, reduce_value_size);
      return;
    }

    if(inform_migration != (NodeID)-1) {
      Barrier b = make_barrier(barrier_gen, timestamp);
      BarrierMigrationMessage::send_request(inform_migration, b, Network::my_node_id);
    }

    if(trigger_gen != 0) {
      log_barrier.info() << "barrier trigger: event=" << me << "/" << trigger_gen;

      // notify local waiters first
      if(!local_notifications.empty()) {
        get_runtime()->event_triggerer.trigger_event_waiters(local_notifications,
                                                             POISON_FIXME, work_until);
      }

      // now do remote notifications

      {
        AutoLock<> al(mutex);
        broadcast_trigger(b, true, remote_notifications, remote_broadcast_targets,
                          oldest_previous, broadcast_previous, first_generation,
                          migration_target, base_arrival_count, redop_id,
                          final_values_copy, 0);
      }
    }

    // free our copy of the final values, if we had one
    if(final_values_copy)
      free(final_values_copy);
  }

  bool BarrierImpl::has_triggered(gen_t needed_gen, bool &poisoned)
  {
    poisoned = POISON_FIXME;

    // no need to take lock to check current generation
    if(needed_gen <= generation.load_acquire())
      return true;

#ifdef BARRIER_HAS_TRIGGERED_DOES_SUBSCRIBE
    // update the subscription (even on the local node), but do a
    //  quick test first to avoid taking a lock if the subscription is
    //  clearly already done
    if(needed_gen > gen_subscribed.load()) {
      // looks like it needs an update - take lock to avoid duplicate
      //  subscriptions
      gen_t previous_subscription;
      bool send_subscription_request = false;
      NodeID cur_owner = (NodeID)-1;
      {
        AutoLock<> a(mutex);
        previous_subscription = gen_subscribed.load();
        if(needed_gen > previous_subscription) {
          gen_subscribed.store(needed_gen);
          // test ownership while holding the mutex
          if(owner != Network::my_node_id) {
            send_subscription_request = true;
            cur_owner = owner;
          }
        }
      }

      // if we're not the owner, send subscription if we haven't already
      if(send_subscription_request) {
        log_barrier.info() << "subscribing to barrier " << make_barrier(needed_gen)
                           << " (prev=" << previous_subscription << ")";
        BarrierSubscribeMessage::send_request(cur_owner, me.id, needed_gen,
                                              Network::my_node_id, false /*!forwarded*/);
      }
    }
#endif

    // whether or not we subscribed, the answer for now is "no"
    return false;
  }

  void BarrierImpl::subscribe(gen_t subscribe_gen)
  {
    // update the subscription (even on the local node), but do a
    //  quick test first to avoid taking a lock if the subscription is
    //  clearly already done
    if(subscribe_gen > gen_subscribed.load()) {
      // looks like it needs an update - take lock to avoid duplicate
      //  subscriptions
      gen_t previous_subscription;
      bool send_subscription_request = false;
      NodeID cur_owner = (NodeID)-1;
      {
        AutoLock<> a(mutex);
        previous_subscription = gen_subscribed.load();
        if(previous_subscription < subscribe_gen) {
          gen_subscribed.store(subscribe_gen);
          // test ownership while holding the mutex
          if(owner != Network::my_node_id) {
            send_subscription_request = true;
            cur_owner = owner;
          }
        }
      }

      // if we're not the owner, send subscription if we haven't already
      if(send_subscription_request) {
        log_barrier.info() << "subscribing to barrier " << make_barrier(subscribe_gen)
                           << " (prev=" << previous_subscription << ")";
        BarrierSubscribeMessage::send_request(cur_owner, me.id, subscribe_gen,
                                              Network::my_node_id, false /*!forwarded*/);
      }
    }
  }

  void BarrierImpl::external_wait(gen_t gen_needed, bool &poisoned)
  {
    poisoned = POISON_FIXME;

    // early out for now without taking lock (TODO: fix for poisoning)
    if(gen_needed <= generation.load_acquire())
      return;

    // make sure we're subscribed to a (potentially-remote) trigger
    this->subscribe(gen_needed);

    {
      AutoLock<> a(mutex);

      // wait until the generation has advanced far enough
      while(gen_needed > generation.load()) {
        has_external_waiters = true;
        // must wait on external_waiter_condvar with external_waiter_mutex
        //  but NOT with base mutex - hand-over-hand lock on the way in,
        //  and then release external_waiter mutex before retaking main
        //  mutex
        external_waiter_mutex.lock();
        mutex.unlock();
        external_waiter_condvar.wait();
        external_waiter_mutex.unlock();
        mutex.lock();
      }
    }
  }

  bool BarrierImpl::external_timedwait(gen_t gen_needed, bool &poisoned, long long max_ns)
  {
    poisoned = POISON_FIXME;

    // early out for now without taking lock (TODO: fix for poisoning)
    if(gen_needed <= generation.load_acquire())
      return true;

    // make sure we're subscribed to a (potentially-remote) trigger
    this->subscribe(gen_needed);

    long long deadline = Clock::current_time_in_nanoseconds() + max_ns;
    {
      AutoLock<> a(mutex);

      // wait until the generation has advanced far enough
      while(gen_needed > generation.load()) {
        long long now = Clock::current_time_in_nanoseconds();
        if(now >= deadline)
          return false; // trigger has not occurred
        has_external_waiters = true;
        // we don't actually care what timedwait returns - we'll recheck
        //  the generation ourselves
        // must wait on external_waiter_condvar with external_waiter_mutex
        //  but NOT with base mutex - hand-over-hand lock on the way in,
        //  and then release external_waiter mutex before retaking main
        //  mutex
        external_waiter_mutex.lock();
        mutex.unlock();
        external_waiter_condvar.timedwait(deadline - now);
        external_waiter_mutex.unlock();
        mutex.lock();
      }
    }
    return true;
  }

  bool BarrierImpl::add_waiter(gen_t needed_gen,
                               EventWaiter *waiter /*, bool pre_subscribed = false*/)
  {
    bool trigger_now = false;
    gen_t previous_subscription;
    bool send_subscription_request = false;
    NodeID cur_owner = (NodeID)-1;
    {
      AutoLock<> a(mutex);

      if(needed_gen > generation.load()) {
        Generation *g;
        std::map<gen_t, Generation *>::iterator it = generations.find(needed_gen);
        if(it != generations.end()) {
          g = it->second;
        } else {
          g = new Generation;
          generations[needed_gen] = g;
          log_barrier.info() << "added tracker for barrier " << make_barrier(needed_gen);
        }
        g->local_waiters.push_back(waiter);

        // check to see if we need to subscribe
        if((owner != Network::my_node_id) && (gen_subscribed.load() < needed_gen)) {
          previous_subscription = gen_subscribed.load();
          gen_subscribed.store(needed_gen);
          send_subscription_request = true;
          cur_owner = owner;
        }
      } else {
        // needed generation has already occurred - trigger this waiter once we let go of
        // lock
        trigger_now = true;
      }
    }

    if(send_subscription_request) {
      log_barrier.info() << "subscribing to barrier " << make_barrier(needed_gen)
                         << " (prev=" << previous_subscription << ")";
      BarrierSubscribeMessage::send_request(cur_owner, me.id, needed_gen,
                                            Network::my_node_id, false /*!forwarded*/);
    }

    if(trigger_now) {
      waiter->event_triggered(POISON_FIXME, TimeLimit::responsive());
    }

    return true;
  }

  bool BarrierImpl::remove_waiter(gen_t needed_gen, EventWaiter *waiter)
  {
    AutoLock<> a(mutex);

    if(needed_gen <= generation.load()) {
      // already triggered, so nothing to remove
      return false;
    }

    // find the right generation - this should not fail
    std::map<gen_t, Generation *>::iterator it = generations.find(needed_gen);
    assert(it != generations.end());
    bool ok = it->second->local_waiters.erase(waiter);
    assert(ok);
    return true;
  }

  /*static*/ void
  BarrierSubscribeMessage::handle_message(NodeID sender,
                                          const BarrierSubscribeMessage &args,
                                          const void *data, size_t datalen)
  {
    ID id(args.barrier_id);
    id.barrier_generation() = args.subscribe_gen;
    Barrier b = id.convert<Barrier>();
    BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

    // take the lock and add the subscribing node - notice if they need to be notified for
    //  any generations that have already triggered
    EventImpl::gen_t trigger_gen = 0;
    EventImpl::gen_t previous_gen = 0;
    void *final_values_copy = 0;
    size_t final_values_size = 0;
    NodeID forward_to_node = (NodeID)-1;
    NodeID inform_migration = (NodeID)-1;

    do {
      AutoLock<> a(impl->mutex);

      // first check - are we even the current owner?
      if(impl->owner != Network::my_node_id) {
        forward_to_node = impl->owner;
        break;
      } else {
        if(args.forwarded) {
          // our own request wrapped back around can be ignored - we've already added the
          // local waiter
          if(args.subscriber == Network::my_node_id) {
            break;
          } else {
            inform_migration = args.subscriber;
          }
        }
      }

      // make sure the subscription is for this "lifetime" of the barrier
      assert(args.subscribe_gen > impl->first_generation);

      bool already_subscribed = false;
      {
        std::map<unsigned, EventImpl::gen_t>::iterator it =
            impl->remote_subscribe_gens.find(args.subscriber);
        if(it != impl->remote_subscribe_gens.end()) {
          // a valid subscription should always be for a generation that hasn't
          //  triggered yet
          assert(it->second > impl->generation.load());
          if(it->second >= args.subscribe_gen)
            already_subscribed = true;
          else
            it->second = args.subscribe_gen;
        } else {
          // new subscription - don't reset remote_trigger_gens because the node may have
          //  been subscribed in the past
          // NOTE: remote_subscribe_gens should only hold subscriptions for
          //  generations that haven't triggered, so if we're subscribing to
          //  an old generation, don't add it
          if(args.subscribe_gen > impl->generation.load())
            impl->remote_subscribe_gens[args.subscriber] = args.subscribe_gen;
        }
      }

      // as long as we're not already subscribed to this generation, check to see if
      //  any trigger notifications are needed
      if(!already_subscribed && (impl->generation.load() > impl->first_generation)) {
        std::map<unsigned, EventImpl::gen_t>::iterator it =
            impl->remote_trigger_gens.find(args.subscriber);
        if((it == impl->remote_trigger_gens.end()) ||
           (it->second < impl->generation.load())) {
          previous_gen = ((it == impl->remote_trigger_gens.end()) ? impl->first_generation
                                                                  : it->second);
          trigger_gen = impl->generation.load();
          impl->remote_trigger_gens[args.subscriber] = impl->generation.load();

          if(impl->redop) {
            int rel_gen = previous_gen + 1 - impl->first_generation;
            assert(rel_gen > 0);
            final_values_size = (trigger_gen - previous_gen) * impl->redop->sizeof_lhs;
            final_values_copy =
                bytedup(impl->final_values + ((rel_gen - 1) * impl->redop->sizeof_lhs),
                        final_values_size);
          }
        }
      }
    } while(0);

    if(forward_to_node != (NodeID)-1) {
      BarrierSubscribeMessage::send_request(forward_to_node, args.barrier_id,
                                            args.subscribe_gen, args.subscriber,
                                            (args.subscriber != Network::my_node_id));
    }

    if(inform_migration != (NodeID)-1) {
      BarrierMigrationMessage::send_request(inform_migration, b, Network::my_node_id);
    }

    // send trigger message outside of lock, if needed
    if(trigger_gen > 0) {
      log_barrier.info("sending immediate barrier trigger: " IDFMT "/%d -> %d",
                       args.barrier_id, previous_gen, trigger_gen);
      BarrierTriggerMessage::send_request(
          args.subscriber, args.barrier_id, trigger_gen, previous_gen,
          impl->first_generation, impl->redop_id, (NodeID)-1 /*no migration*/,
          0 /*dummy arrival count*/, 0, final_values_copy, final_values_size);
    }

    if(final_values_copy)
      free(final_values_copy);
  }

  /*static*/ void BarrierTriggerMessage::handle_message(NodeID sender,
                                                        const BarrierTriggerMessage &args,
                                                        const void *data, size_t datalen,
                                                        TimeLimit work_until)
  {
    log_barrier.info("received remote barrier trigger: " IDFMT "/%d -> %d",
                     args.barrier_id, args.previous_gen, args.trigger_gen);

    Serialization::FixedBufferDeserializer fbd(data, datalen);
    BarrierTriggerMessageArgs trigger_args;
    bool ok = fbd & trigger_args;
    assert(ok);

    data = (char *)data + (datalen - fbd.bytes_left());
    datalen = fbd.bytes_left();

    EventImpl::gen_t trigger_gen = trigger_args.internal.trigger_gen;

    ID id(args.barrier_id);
    id.barrier_generation() = trigger_gen;
    Barrier b = id.convert<Barrier>();
    BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

    if(trigger_args.remote_notifications.size() > 0) {
      AutoLock<> a(impl->mutex);

      impl->buffered_notifications.insert(impl->buffered_notifications.end(),
                                          trigger_args.remote_notifications.begin(),
                                          trigger_args.remote_notifications.end());

      if(trigger_args.internal.is_complete_list == false) {
        return;
      }

      size_t num_peers = impl->buffered_notifications.size();

      // TODO(apryakhin:): Make it so that we don't need
      // broadcast_index.
      NodeID node = trigger_args.internal.broadcast_index;
      std::vector<NodeID> broadcast_targets;
      get_broadcast_targets(node, num_peers, BarrierConfig::broadcast_radix,
                            broadcast_targets);

      broadcast_trigger(b, true, impl->buffered_notifications, broadcast_targets, 0, 0,
                        trigger_args.internal.first_generation,
                        trigger_args.internal.migration_target,
                        trigger_args.internal.base_arrival_count,
                        trigger_args.internal.redop_id, data, datalen);

      impl->buffered_notifications.clear();
    }

    // we'll probably end up with a list of local waiters to notify
    EventWaiter::EventWaiterList local_notifications;
    {
      AutoLock<> a(impl->mutex);

      bool generation_updated = false;

      // handle migration of the barrier ownership (possibly to us)
      if(trigger_args.internal.migration_target != (NodeID)-1) {
        log_barrier.info() << "barrier " << b << " has migrated to "
                           << trigger_args.internal.migration_target;
        impl->owner = trigger_args.internal.migration_target;
        impl->base_arrival_count = trigger_args.internal.base_arrival_count;
      }

      // it's theoretically possible for multiple trigger messages to arrive out
      //  of order, so check if this message triggers the oldest possible range
      // NOTE: it's ok for previous_gen to be earlier than our current generation - this
      //  occurs with barrier migration because the new owner may not know which
      //  notifications have already been performed
      if(trigger_args.internal.previous_gen <= impl->generation.load()) {
        // see if we can pick up any of the held triggers too
        while(!impl->held_triggers.empty()) {
          std::map<EventImpl::gen_t, EventImpl::gen_t>::iterator it =
              impl->held_triggers.begin();
          // if it's not contiguous, we're done
          if(it->first != trigger_gen)
            break;
          // it is contiguous, so absorb it into this message and remove the held trigger
          log_barrier.info("collapsing future trigger: " IDFMT "/%d -> %d -> %d",
                           args.barrier_id, trigger_args.internal.previous_gen,
                           trigger_gen, it->second);
          trigger_gen = it->second;
          impl->held_triggers.erase(it);
        }

        if(trigger_gen > impl->generation.load()) {
          impl->generation.store_release(trigger_gen);
          generation_updated = true;
        }

        // now iterate through any generations up to and including the latest triggered
        //  generation, and accumulate local waiters to notify
        while(!impl->generations.empty()) {
          std::map<EventImpl::gen_t, BarrierImpl::Generation *>::iterator it =
              impl->generations.begin();
          if(it->first > trigger_gen)
            break;

          local_notifications.absorb_append(it->second->local_waiters);
          delete it->second;
          impl->generations.erase(it);
        }
      } else {
        // hold this trigger until we get messages for the earlier generation(s)
        log_barrier.info("holding future trigger: " IDFMT "/%d (%d -> %d)",
                         args.barrier_id, impl->generation.load(),
                         trigger_args.internal.previous_gen, trigger_gen);
        impl->held_triggers[trigger_args.internal.previous_gen] = trigger_gen;
      }

      // is there any data we need to store?
      if(datalen) {
        assert(trigger_args.internal.redop_id != 0);

        // TODO: deal with invalidation of previous instance of a barrier
        impl->redop_id = trigger_args.internal.redop_id;
        impl->redop =
            get_runtime()->reduce_op_table.get(trigger_args.internal.redop_id, 0);
        if(impl->redop == 0) {
          log_barrier.fatal() << "no reduction op registered for ID "
                              << trigger_args.internal.redop_id;
          abort();
        }
        impl->first_generation = trigger_args.internal.first_generation;

        int rel_gen = trigger_gen - impl->first_generation;
        assert(rel_gen > 0);
        if(impl->value_capacity < (size_t)rel_gen) {
          size_t new_capacity = rel_gen;
          impl->final_values =
              (char *)realloc(impl->final_values, new_capacity * impl->redop->sizeof_lhs);
          // no need to initialize new entries - we'll overwrite them now or when data
          // does show up
          impl->value_capacity = new_capacity;
        }
        assert(trigger_args.internal.trigger_gen <= trigger_gen);
        // trigger_gen might have changed so make sure you use args.trigger_gen here
        assert(datalen ==
               (impl->redop->sizeof_lhs * (trigger_args.internal.trigger_gen -
                                           trigger_args.internal.previous_gen)));
        assert(trigger_args.internal.previous_gen >= impl->first_generation);
        memcpy(impl->final_values +
                   ((trigger_args.internal.previous_gen - impl->first_generation) *
                    impl->redop->sizeof_lhs),
               data, datalen);
      }

      // external waiters need to be signalled inside the lock
      if(generation_updated && impl->has_external_waiters) {
        impl->has_external_waiters = false;
        // also need external waiter mutex
        AutoLock<KernelMutex> al2(impl->external_waiter_mutex);
        impl->external_waiter_condvar.broadcast();
      }
    }

    // with lock released, perform any local notifications
    if(!local_notifications.empty())
      get_runtime()->event_triggerer.trigger_event_waiters(local_notifications,
                                                           POISON_FIXME, work_until);
  }

  bool BarrierImpl::get_result(gen_t result_gen, void *value, size_t value_size)
  {
    // generation hasn't triggered yet?
    if(result_gen > generation.load_acquire())
      return false;

    // take the lock so we can safely see how many results (if any) are on hand
    AutoLock<> al(mutex);

    // if it has triggered, we should have the data
    int rel_gen = result_gen - first_generation;
    assert(rel_gen > 0);
    assert((size_t)rel_gen <= value_capacity);

    assert(redop != 0);
    assert(value_size == redop->sizeof_lhs);
    assert(value != 0);
    memcpy(value, final_values + ((rel_gen - 1) * redop->sizeof_lhs), redop->sizeof_lhs);
    return true;
  }

  /*static*/ void
  BarrierMigrationMessage::handle_message(NodeID sender,
                                          const BarrierMigrationMessage &args,
                                          const void *data, size_t datalen)
  {
    log_barrier.info() << "received barrier migration: barrier=" << args.barrier
                       << " owner=" << args.current_owner;
    BarrierImpl *impl = get_runtime()->get_barrier_impl(args.barrier);
    {
      AutoLock<> a(impl->mutex);
      impl->owner = args.current_owner;
    }
  }

  /*static*/ void BarrierMigrationMessage::send_request(NodeID target, Barrier barrier,
                                                        NodeID owner)
  {
    ActiveMessage<BarrierMigrationMessage> amsg(target);
    amsg->barrier = barrier;
    amsg->current_owner = owner;
    amsg.commit();
  }

  ActiveMessageHandlerReg<BarrierAdjustMessage> barrier_adjust_message_handler;
  ActiveMessageHandlerReg<BarrierSubscribeMessage> barrier_subscribe_message_handler;
  ActiveMessageHandlerReg<BarrierTriggerMessage> barrier_trigger_message_handler;
  ActiveMessageHandlerReg<BarrierMigrationMessage> barrier_migration_message_handler;

}; // namespace Realm
