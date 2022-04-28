/* Copyright 2022 Stanford University, NVIDIA Corporation
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

// Event/UserEvent/Barrier implementations for Realm

#ifndef REALM_EVENT_IMPL_H
#define REALM_EVENT_IMPL_H

#include "realm/event.h"
#include "realm/id.h"
#include "realm/nodeset.h"
#include "realm/faults.h"

#include "realm/network.h"

#include "realm/lists.h"
#include "realm/threads.h"
#include "realm/logging.h"
#include "realm/redop.h"
#include "realm/bgwork.h"

#include <vector>
#include <map>

namespace Realm {

#ifdef EVENT_TRACING
    // For event tracing
    struct EventTraceItem {
    public:
      enum Action {
        ACT_CREATE = 0,
        ACT_QUERY = 1,
        ACT_TRIGGER = 2,
        ACT_WAIT = 3,
      };
    public:
      unsigned time_units, event_id, event_gen, action;
    };
#endif

    extern Logger log_poison; // defined in event_impl.cc

    class EventWaiter {
    public:
      virtual ~EventWaiter(void) {}
      virtual void event_triggered(bool poisoned, TimeLimit work_until) = 0;
      virtual void print(std::ostream& os) const = 0;
      virtual Event get_finish_event(void) const = 0;

      IntrusiveListLink<EventWaiter> ew_list_link;
      REALM_PMTA_DEFN(EventWaiter,IntrusiveListLink<EventWaiter>,ew_list_link);
      typedef IntrusiveList<EventWaiter, REALM_PMTA_USE(EventWaiter,ew_list_link), DummyLock> EventWaiterList;
    };

    // triggering events can often result in recursive expansion of work -
    //  this widget flattens the call stack and defers excessive triggers
    //  to avoid stalling the initial triggerer longer than they want
    class EventTriggerNotifier : public BackgroundWorkItem {
    public:
      EventTriggerNotifier();

      void trigger_event_waiters(EventWaiter::EventWaiterList& to_trigger,
				 bool poisoned,
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
      virtual bool has_triggered(gen_t needed_gen, bool& poisoned) = 0;

      virtual void subscribe(gen_t subscribe_gen) = 0;

      // causes calling thread to block until event has occurred
      //void wait(Event::gen_t needed_gen);

      virtual void external_wait(gen_t needed_gen, bool& poisoned) = 0;
      virtual bool external_timedwait(gen_t needed_gen, bool& poisoned,
				      long long max_ns) = 0;

      // helper to create the Event for an arbitrary generation
      Event make_event(gen_t gen) const;

      virtual bool add_waiter(gen_t needed_gen, EventWaiter *waiter/*, bool pre_subscribed = false*/) = 0;

      static bool add_waiter(Event needed, EventWaiter *waiter);

      // use this sparingly - it has to hunt through waiter lists while
      //  holding locks
      virtual bool remove_waiter(gen_t needed_gen, EventWaiter *waiter) = 0;

      static bool detect_event_chain(Event search_from, Event target, int max_depth, bool print_chain);

    public:
      ID me;
      NodeID owner;
    };

    class GenEventImpl;

    class EventMerger {
    public:
      EventMerger(GenEventImpl *_event_impl);
      ~EventMerger(void);

      bool is_active(void) const;

      void prepare_merger(Event _finish_event, bool _ignore_faults, unsigned _max_preconditions);

      void add_precondition(Event wait_for);

      void arm_merger(void);

      class MergeEventPrecondition : public EventWaiter {
      public:
	EventMerger *merger;

	virtual void event_triggered(bool poisoned, TimeLimit work_until);
	virtual void print(std::ostream& os) const;
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

    class GenEventImpl : public EventImpl {
    public:
      static const ID::ID_Types ID_TYPE = ID::ID_EVENT;

      GenEventImpl(void);
      ~GenEventImpl(void);

      void init(ID _me, unsigned _init_owner);

      static GenEventImpl *create_genevent(void);

      // get the Event (id+generation) for the current (i.e. untriggered) generation
      Event current_event(void) const;

      // test whether an event has triggered without waiting
      virtual bool has_triggered(gen_t needed_gen, bool& poisoned);

      virtual void subscribe(gen_t subscribe_gen);

      virtual void external_wait(gen_t needed_gen, bool& poisoned);
      virtual bool external_timedwait(gen_t needed_gen, bool& poisoned,
				      long long max_ns);

      virtual bool add_waiter(gen_t needed_gen, EventWaiter *waiter);

      // use this sparingly - it has to hunt through waiter lists while
      //  holding locks
      virtual bool remove_waiter(gen_t needed_gen, EventWaiter *waiter);

      // creates an event that won't trigger until all input events have
      static Event merge_events(const std::set<Event>& wait_for,
				bool ignore_faults);
      static Event merge_events(span<const Event> wait_for,
				bool ignore_faults);
      static Event merge_events(Event ev1, Event ev2,
				Event ev3 = Event::NO_EVENT, Event ev4 = Event::NO_EVENT,
				Event ev5 = Event::NO_EVENT, Event ev6 = Event::NO_EVENT);
      static Event ignorefaults(Event wait_for);

      // record that the event has triggered and notify anybody who cares
      void trigger(gen_t gen_triggered, int trigger_node, bool poisoned,
		   TimeLimit work_until);

      // helper for triggering with an Event (which must be backed by a GenEventImpl)
      static void trigger(Event e, bool poisoned);
      static void trigger(Event e, bool poisoned, TimeLimit work_until);

      // process an update message from the owner
      void process_update(gen_t current_gen,
			  const gen_t *new_poisoned_generations,
			  int new_poisoned_count,
			  TimeLimit work_until);

    public: //protected:
      // these state variables are monotonic, so can be checked without a lock for
      //  early-out conditions
      atomic<gen_t> generation;
      atomic<gen_t> gen_subscribed;
      atomic<int> num_poisoned_generations;
      bool has_local_triggers;

      bool is_generation_poisoned(gen_t gen) const; // helper function - linear search

      // this is only manipulated when the event is "idle"
      GenEventImpl *next_free;

      // used for merge_events and delayed UserEvent triggers
      EventMerger merger;

      // everything below here protected by this mutex
      Mutex mutex;

      // local waiters are tracked by generation - an easily-accessed list is used
      //  for the "current" generation, whereas a map-by-generation-id is used for
      //  "future" generations (i.e. ones ahead of what we've heard about if we're
      //  not the owner)
      EventWaiter::EventWaiterList current_local_waiters;
      std::map<gen_t, EventWaiter::EventWaiterList> future_local_waiters;

      // external waiters on this node are notifies via a condition variable
      bool has_external_waiters;
      // use kernel mutex for timedwait functionality
      KernelMutex external_waiter_mutex;
      KernelMutex::CondVar external_waiter_condvar;

      // remote waiters are kept in a bitmask for the current generation - this is
      //  only maintained on the owner, who never has to worry about more than one
      //  generation
      NodeSet remote_waiters;

      // we'll set an upper bound on how many times any given event can be poisoned - this keeps
      // update messages from growing without bound
      static const int POISONED_GENERATION_LIMIT = 16;

      // note - we don't bother sorting the list below - the overhead of a binary search
      //  dominates for short lists
      // we also can't use an STL vector because reallocation prevents us from reading the
      //  list without the lock - instead we'll allocate the max size if/when we need
      //  any space
      gen_t *poisoned_generations;

      // local triggerings - if we're not the owner, but we've triggered/poisoned events,
      //  we need to give consistent answers for those generations, so remember what we've
      //  done until our view of the distributed event catches up
      // value stored in map is whether generation was poisoned
      std::map<gen_t, bool> local_triggers;

      // these resolve a race condition between the early trigger of a
      //  poisoned merge and the last precondition
      bool free_list_insertion_delayed;
      friend class EventMerger;
      void perform_delayed_free_list_insertion(void);
    };

    class BarrierImpl : public EventImpl {
    public:
      static const ID::ID_Types ID_TYPE = ID::ID_BARRIER;

      static const int BARRIER_TIMESTAMP_NODEID_SHIFT = 48;
      static atomic<Barrier::timestamp_t> barrier_adjustment_timestamp;

      BarrierImpl(void);
      ~BarrierImpl(void);

      void init(ID _me, unsigned _init_owner);

      // get the Barrier (id+generation) for the current (i.e. untriggered) generation
      Barrier current_barrier(Barrier::timestamp_t timestamp = 0) const;

      // helper to create the Barrier for an arbitrary generation
      Barrier make_barrier(gen_t gen, Barrier::timestamp_t timestamp = 0) const;

      static BarrierImpl *create_barrier(unsigned expected_arrivals, ReductionOpID redopid,
					 const void *initial_value = 0, size_t initial_value_size = 0);

      // test whether an event has triggered without waiting
      virtual bool has_triggered(gen_t needed_gen, bool& poisoned);

      virtual void subscribe(gen_t subscribe_gen);

      virtual void external_wait(gen_t needed_gen, bool& poisoned);
      virtual bool external_timedwait(gen_t needed_gen, bool& poisoned,
				      long long max_ns);

      virtual bool add_waiter(gen_t needed_gen, EventWaiter *waiter/*, bool pre_subscribed = false*/);

      // use this sparingly - it has to hunt through waiter lists while
      //  holding locks
      virtual bool remove_waiter(gen_t needed_gen, EventWaiter *waiter);

      // used to adjust a barrier's arrival count either up or down
      // if delta > 0, timestamp is current time (on requesting node)
      // if delta < 0, timestamp says which positive adjustment this arrival must wait for
      void adjust_arrival(gen_t barrier_gen, int delta, 
			  Barrier::timestamp_t timestamp, Event wait_on,
			  NodeID sender, bool forwarded,
			  const void *reduce_value, size_t reduce_value_size,
			  TimeLimit work_until);

      bool get_result(gen_t result_gen, void *value, size_t value_size);

    public: //protected:
      atomic<gen_t> generation; // can be read without holding mutex
      atomic<gen_t> gen_subscribed;
      gen_t first_generation;
      BarrierImpl *next_free;

      Mutex mutex; // controls which local thread has access to internal data (not runtime-visible event)

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
      bool has_external_waiters;
      // use kernel mutex for timedwait functionality
      KernelMutex external_waiter_mutex;
      KernelMutex::CondVar external_waiter_condvar;

      // a list of remote waiters and the latest generation they're interested in
      // also the latest generation that each node (that has ever subscribed) has been told about
      std::map<unsigned, gen_t> remote_subscribe_gens, remote_trigger_gens;
      std::map<gen_t, gen_t> held_triggers;

      unsigned base_arrival_count;
      ReductionOpID redop_id;
      const ReductionOpUntyped *redop;
      char *initial_value;  // for reduction barriers

      unsigned value_capacity; // how many values the two allocations below can hold
      char *final_values;   // results of completed reductions
    };

    class CompQueueImpl {
    public:
      CompQueueImpl(void);
      ~CompQueueImpl(void);

      void init(CompletionQueue _me, int _owner);

      void set_capacity(size_t _max_size, bool _resizable);

      void destroy(void);

      void add_event(Event event, bool faultaware);

      Event get_local_progress_event(void);
      void add_remote_progress_event(Event event);

      size_t pop_events(Event *events, size_t max_to_pop);

      CompletionQueue me;
      int owner;
      CompQueueImpl *next_free;

      class DeferredDestroy : public EventWaiter {
      public:
	void defer(CompQueueImpl *_cq, Event wait_on);
	virtual void event_triggered(bool poisoned, TimeLimit work_until);
	virtual void print(std::ostream& os) const;
	virtual Event get_finish_event(void) const;

      protected:
	CompQueueImpl *cq;
      };
      DeferredDestroy deferred_destroy;

      // used to track pending remote pop requests
      class RemotePopRequest {
      public:
	RemotePopRequest(Event *_events, size_t _capacity);

	Mutex mutex;
        Mutex::CondVar condvar;
	bool completed;
	size_t count, capacity;
	Event *events;
      };

    protected:
      class CompQueueWaiter : public EventWaiter {
      public:
        virtual void event_triggered(bool poisoned, TimeLimit work_until);
	virtual void print(std::ostream& os) const;
	virtual Event get_finish_event(void) const;

	CompQueueImpl *cq;
	Event wait_on;
	bool faultaware;
	CompQueueWaiter *next_free;
      };

      void add_completed_event(Event event, CompQueueWaiter *waiter,
			       TimeLimit work_until);

      static const size_t CQWAITER_BATCH_SIZE = 16;
      class CompQueueWaiterBatch {
      public:
	CompQueueWaiterBatch(CompQueueImpl *cq, CompQueueWaiterBatch *_next);
	~CompQueueWaiterBatch(void);

	CompQueueWaiter waiters[CQWAITER_BATCH_SIZE];
	CompQueueWaiterBatch *next_batch;
      };

      Mutex mutex; // protects everything below here

      bool resizable;
      size_t max_events;
      atomic<size_t> wr_ptr, rd_ptr, pending_events;
      // used if resizable==false
      atomic<size_t> commit_ptr, consume_ptr;
      // used if resizable==true
      size_t cur_events;

      Event *completed_events;

      atomic<bool> has_progress_events;
      GenEventImpl *local_progress_event;
      EventImpl::gen_t local_progress_event_gen;
      // TODO: small vector
      std::vector<Event> remote_progress_events;
      atomic<CompQueueWaiter *> first_free_waiter;
      CompQueueWaiterBatch *batches;
    };

  // active messages

  struct EventSubscribeMessage {
    Event event;
    EventImpl::gen_t previous_subscribe_gen;

    static void handle_message(NodeID sender, const EventSubscribeMessage &msg,
			       const void *data, size_t datalen);

  };

  struct EventTriggerMessage {
    Event event;
    bool poisoned;

    static void handle_message(NodeID sender, const EventTriggerMessage &msg,
			       const void *data, size_t datalen,
			       TimeLimit work_until);

  };

  struct EventUpdateMessage {
    Event event;

    static void handle_message(NodeID sender, const EventUpdateMessage &msg,
			       const void *data, size_t datalen,
			       TimeLimit work_until);

  };

  struct BarrierAdjustMessage {
    NodeID sender;
    int forwarded;
    int delta;
    Barrier barrier;
    Event wait_on;

    static void handle_message(NodeID sender, const BarrierAdjustMessage &msg,
			       const void *data, size_t datalen,
			       TimeLimit work_until);
    static void send_request(NodeID target, Barrier barrier, int delta, Event wait_on,
			     NodeID sender, bool forwarded,
			     const void *data, size_t datalen);
    };

  struct BarrierSubscribeMessage {
    NodeID subscriber;
    ID::IDType barrier_id;
    EventImpl::gen_t subscribe_gen;
    bool forwarded;

    static void handle_message(NodeID sender, const BarrierSubscribeMessage &msg,
			       const void *data, size_t datalen);

    static void send_request(NodeID target, ID::IDType barrier_id,
			     EventImpl::gen_t subscribe_gen,
			     NodeID subscriber, bool forwarded);
  };

  struct BarrierTriggerMessage {
    ID::IDType barrier_id;
    EventImpl::gen_t trigger_gen;
    EventImpl::gen_t previous_gen;
    EventImpl::gen_t first_generation;
    ReductionOpID redop_id;
    NodeID migration_target;
    unsigned base_arrival_count;

    static void handle_message(NodeID sender, const BarrierTriggerMessage &msg,
			       const void *data, size_t datalen,
			       TimeLimit work_until);

    static void send_request(NodeID target, ID::IDType barrier_id,
			     EventImpl::gen_t trigger_gen, EventImpl::gen_t previous_gen,
			     EventImpl::gen_t first_generation, ReductionOpID redop_id,
			     NodeID migration_target, unsigned base_arrival_count,
			     const void *data, size_t datalen);
  };

  struct BarrierMigrationMessage {
    Barrier barrier;
    NodeID current_owner;

    static void handle_message(NodeID sender, const BarrierMigrationMessage &msg,
			       const void *data, size_t datalen);
    static void send_request(NodeID target, Barrier barrier, NodeID owner);
  };

  struct CompQueueDestroyMessage {
    CompletionQueue comp_queue;
    Event wait_on;

    static void handle_message(NodeID sender, const CompQueueDestroyMessage &msg,
			       const void *data, size_t datalen);
  };

  struct CompQueueAddEventMessage {
    CompletionQueue comp_queue;
    Event event;
    bool faultaware;

    static void handle_message(NodeID sender, const CompQueueAddEventMessage &msg,
			       const void *data, size_t datalen);
  };

  struct CompQueueRemoteProgressMessage {
    CompletionQueue comp_queue;
    Event progress;

    static void handle_message(NodeID sender, const CompQueueRemoteProgressMessage &msg,
			       const void *data, size_t datalen);
  };

  struct CompQueuePopRequestMessage {
    CompletionQueue comp_queue;
    size_t max_to_pop;
    bool discard_events;
    intptr_t request;

    static void handle_message(NodeID sender,
			       const CompQueuePopRequestMessage &msg,
			       const void *data, size_t datalen);
  };

  struct CompQueuePopResponseMessage {
    size_t count;
    intptr_t request;

    static void handle_message(NodeID sender,
			       const CompQueuePopResponseMessage &msg,
			       const void *data, size_t datalen);
  };

}; // namespace Realm

#include "realm/event_impl.inl"

#endif // ifndef REALM_EVENT_IMPL_H

