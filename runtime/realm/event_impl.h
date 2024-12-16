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

// Event/UserEvent implementations for Realm

#ifndef REALM_EVENT_IMPL_H
#define REALM_EVENT_IMPL_H

#include "realm/event.h"
#include "realm/id.h"
#include "realm/nodeset.h"
#include "realm/faults.h"

#include "realm/network.h"
#include <realm/activemsg.h>

#include "realm/lists.h"
#include "realm/threads.h"
#include "realm/logging.h"
#include "realm/redop.h"
#include "realm/bgwork.h"
#include "realm/dynamic_table.h"

#include <vector>
#include <map>
#include <memory>

namespace Realm {

  class GenEventImpl;

  extern Logger log_poison; // defined in event_impl.cc
  class ProcessorImpl;      // defined in proc_impl.h

  class EventWaiter {
  public:
    virtual ~EventWaiter(void) {}
    virtual void event_triggered(bool poisoned, TimeLimit work_until) = 0;
    virtual void print(std::ostream &os) const = 0;
    virtual Event get_finish_event(void) const = 0;

    IntrusiveListLink<EventWaiter> ew_list_link;
    REALM_PMTA_DEFN(EventWaiter, IntrusiveListLink<EventWaiter>, ew_list_link);
    typedef IntrusiveList<EventWaiter, REALM_PMTA_USE(EventWaiter, ew_list_link),
                          DummyLock>
        EventWaiterList;
  };

  // triggering events can often result in recursive expansion of work -
  //  this widget flattens the call stack and defers excessive triggers
  //  to avoid stalling the initial triggerer longer than they want
  class EventTriggerNotifier : public BackgroundWorkItem {
  public:
    EventTriggerNotifier();

    void trigger_event_waiters(EventWaiter::EventWaiterList &to_trigger, bool poisoned,
                               TimeLimit trigger_until);

    virtual bool do_work(TimeLimit work_until);

  protected:
    Mutex mutex;
    EventWaiter::EventWaiterList delayed_normal;
    EventWaiter::EventWaiterList delayed_poisoned;

    static REALM_THREAD_LOCAL EventWaiter::EventWaiterList *nested_normal;
    static REALM_THREAD_LOCAL EventWaiter::EventWaiterList *nested_poisoned;
  };

  // parent class of GenEventImpl and BarrierImpl
  class EventImpl {
  public:
    typedef unsigned gen_t;

    EventImpl(void);
    virtual ~EventImpl(void);

    // test whether an event has triggered without waiting
    virtual bool has_triggered(gen_t needed_gen, bool &poisoned) = 0;

    virtual void subscribe(gen_t subscribe_gen) = 0;

    // causes calling thread to block until event has occurred
    // void wait(Event::gen_t needed_gen);

    virtual void external_wait(gen_t needed_gen, bool &poisoned) = 0;
    virtual bool external_timedwait(gen_t needed_gen, bool &poisoned,
                                    long long max_ns) = 0;

    // helper to create the Event for an arbitrary generation
    Event make_event(gen_t gen) const;

    virtual bool add_waiter(gen_t needed_gen,
                            EventWaiter *waiter /*, bool pre_subscribed = false*/) = 0;

    static bool add_waiter(Event needed, EventWaiter *waiter);

    // use this sparingly - it has to hunt through waiter lists while
    //  holding locks
    virtual bool remove_waiter(gen_t needed_gen, EventWaiter *waiter) = 0;

    static bool detect_event_chain(Event search_from, Event target, int max_depth,
                                   bool print_chain);

  public:
    ID me;
    ProcessorImpl *owning_processor;
    NodeID owner;
  };

  class GenEventImpl;

  class EventMerger {
  public:
    EventMerger(GenEventImpl *_event_impl);
    ~EventMerger(void);

    bool is_active(void) const;

    void prepare_merger(Event _finish_event, bool _ignore_faults,
                        unsigned _max_preconditions);

    void add_precondition(Event wait_for);

    void arm_merger(void);

    class MergeEventPrecondition : public EventWaiter {
    public:
      EventMerger *merger;

      virtual void event_triggered(bool poisoned, TimeLimit work_until);
      virtual void print(std::ostream &os) const;
      virtual Event get_finish_event(void) const;
    };

    // as an alternative to add_precondition, get_next_precondition can
    //  be used to get a precondition that can manually be added to a waiter
    //  list
    MergeEventPrecondition *get_next_precondition(void);

  protected:
    void precondition_triggered(bool poisoned, TimeLimit work_until);

    friend class MergeEventPrecondition;

    GenEventImpl *event_impl;
    EventImpl::gen_t finish_gen;
    bool ignore_faults;
    atomic<int> count_needed;
    atomic<int> faults_observed;

    static const size_t MAX_INLINE_PRECONDITIONS = 6;
    MergeEventPrecondition inline_preconditions[MAX_INLINE_PRECONDITIONS];
    MergeEventPrecondition *preconditions;
    unsigned num_preconditions, max_preconditions;
  };

  class EventCommunicator {
  public:
    virtual ~EventCommunicator() = default;

    virtual void trigger(Event event, NodeID owner, bool poisoned);

    virtual void update(Event event, NodeSet to_update,
                        span<EventImpl::gen_t> poisoned_generations);

    virtual void update(Event event, NodeID to_update,
                        span<EventImpl::gen_t> poisoned_generations);

    virtual void subscribe(Event event, NodeID owner,
                           EventImpl::gen_t previous_subscribe_gen);
  };

  class GenEventImpl : public EventImpl {
  public:
    static const ID::ID_Types ID_TYPE = ID::ID_EVENT;

    GenEventImpl(void);
    GenEventImpl(EventTriggerNotifier *_event_triggerer, EventCommunicator *_event_comm);
    ~GenEventImpl(void);

    void init(ID _me, unsigned _init_owner);

    static GenEventImpl *create_genevent(void);

    static ID make_id(const GenEventImpl &dummy, int owner, ID::IDType index)
    {
      return ID::make_event(owner, index, 0);
    }

    // get the Event (id+generation) for the current (i.e. untriggered) generation
    Event current_event(void) const;

    // test whether an event has triggered without waiting
    virtual bool has_triggered(gen_t needed_gen, bool &poisoned);

    virtual void subscribe(gen_t subscribe_gen);
    void handle_remote_subscription(NodeID sender, gen_t subscribe_gen,
                                    gen_t previous_subscribe_gen);

    virtual void external_wait(gen_t needed_gen, bool &poisoned);
    virtual bool external_timedwait(gen_t needed_gen, bool &poisoned, long long max_ns);

    virtual bool add_waiter(gen_t needed_gen, EventWaiter *waiter);

    // use this sparingly - it has to hunt through waiter lists while
    //  holding locks
    virtual bool remove_waiter(gen_t needed_gen, EventWaiter *waiter);

    // creates an event that won't trigger until all input events have
    static Event merge_events(span<const Event> wait_for, bool ignore_faults);
    static Event merge_events(Event ev1, Event ev2, Event ev3 = Event::NO_EVENT,
                              Event ev4 = Event::NO_EVENT, Event ev5 = Event::NO_EVENT,
                              Event ev6 = Event::NO_EVENT);
    static Event ignorefaults(Event wait_for);

    // record that the event has triggered and notify anybody who cares
    bool trigger(gen_t gen_triggered, int trigger_node, bool poisoned,
                 TimeLimit work_until);

    // helper for triggering with an Event (which must be backed by a GenEventImpl)
    static void trigger(Event e, bool poisoned);
    static void trigger(Event e, bool poisoned, TimeLimit work_until);

    // process an update message from the owner
    void process_update(gen_t current_gen, const gen_t *new_poisoned_generations,
                        int new_poisoned_count, TimeLimit work_until);

    // Set the operation that will trigger this event's generation.
    void set_trigger_op(gen_t gen, Operation *op);
    // Get the operation that will trigger this event's generation.
    // The returned operation's reference is incremented and must be removed by the
    // caller.
    Operation *get_trigger_op(gen_t gen);

  public: // protected:
    // these state variables are monotonic, so can be checked without a lock for
    //  early-out conditions
    atomic<gen_t> generation = atomic<gen_t>(0);
    atomic<gen_t> gen_subscribed = atomic<gen_t>(0);
    atomic<int> num_poisoned_generations = atomic<int>(0);
    bool has_local_triggers = false;

    bool is_generation_poisoned(gen_t gen) const; // helper function - linear search

    // this is only manipulated when the event is "idle"
    GenEventImpl *next_free = 0;

    // used for merge_events and delayed UserEvent triggers
    EventMerger merger;

    EventTriggerNotifier *event_triggerer;
    std::unique_ptr<EventCommunicator> event_comm;

    // everything below here protected by this mutex
    Mutex mutex;

    // The operation that will trigger this generation
    Operation *current_trigger_op = nullptr;

    // local waiters are tracked by generation - an easily-accessed list is used
    //  for the "current" generation, whereas a map-by-generation-id is used for
    //  "future" generations (i.e. ones ahead of what we've heard about if we're
    //  not the owner)
    EventWaiter::EventWaiterList current_local_waiters;
    std::map<gen_t, EventWaiter::EventWaiterList> future_local_waiters;

    // external waiters on this node are notifies via a condition variable
    bool has_external_waiters = false;
    // use kernel mutex for timedwait functionality
    KernelMutex external_waiter_mutex;
    KernelMutex::CondVar external_waiter_condvar;

    // remote waiters are kept in a bitmask for the current generation - this is
    //  only maintained on the owner, who never has to worry about more than one
    //  generation
    NodeSet remote_waiters;

    // we'll set an upper bound on how many times any given event can be poisoned - this
    // keeps update messages from growing without bound
    static const int POISONED_GENERATION_LIMIT = 16;

    // note - we don't bother sorting the list below - the overhead of a binary search
    //  dominates for short lists
    // we also can't use an STL vector because reallocation prevents us from reading the
    //  list without the lock - instead we'll allocate the max size if/when we need
    //  any space
    gen_t *poisoned_generations = 0;

    // local triggerings - if we're not the owner, but we've triggered/poisoned events,
    //  we need to give consistent answers for those generations, so remember what we've
    //  done until our view of the distributed event catches up
    // value stored in map is whether generation was poisoned
    std::map<gen_t, bool> local_triggers;

    // these resolve a race condition between the early trigger of a
    //  poisoned merge and the last precondition
    bool free_list_insertion_delayed = false;
    friend class EventMerger;
  };
}; // namespace Realm

#include "realm/event_impl.inl"

#endif // ifndef REALM_EVENT_IMPL_H

