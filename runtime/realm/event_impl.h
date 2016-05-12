/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include "event.h"
#include "id.h"
#include "nodeset.h"
#include "faults.h"

#include "activemsg.h"

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
      virtual bool event_triggered(Event e, bool poisoned) = 0;
      virtual void print(std::ostream& os) const = 0;
      virtual Event get_finish_event(void) const = 0;
    };

    // parent class of GenEventImpl and BarrierImpl
    class EventImpl {
    public:
      // test whether an event has triggered without waiting
      virtual bool has_triggered(Event::gen_t needed_gen, bool& poisoned) = 0;

      // causes calling thread to block until event has occurred
      //void wait(Event::gen_t needed_gen);

      virtual void external_wait(Event::gen_t needed_gen, bool& poisoned) = 0;

      virtual bool add_waiter(Event::gen_t needed_gen, EventWaiter *waiter/*, bool pre_subscribed = false*/) = 0;

      static bool add_waiter(Event needed, EventWaiter *waiter);

      static bool detect_event_chain(Event search_from, Event target, int max_depth, bool print_chain);
    };

    class GenEventImpl : public EventImpl {
    public:
      static const ID::ID_Types ID_TYPE = ID::ID_EVENT;

      GenEventImpl(void);

      void init(ID _me, unsigned _init_owner);

      static GenEventImpl *create_genevent(void);

      // get the Event (id+generation) for the current (i.e. untriggered) generation
      Event current_event(void) const;

      // helper to create the Event for an arbitrary generation
      Event make_event(Event::gen_t gen) const;

      // test whether an event has triggered without waiting
      virtual bool has_triggered(Event::gen_t needed_gen, bool& poisoned);

      virtual void external_wait(Event::gen_t needed_gen, bool& poisoned);

      virtual bool add_waiter(Event::gen_t needed_gen, EventWaiter *waiter);

      // creates an event that won't trigger until all input events have
      static Event merge_events(const std::set<Event>& wait_for,
				bool ignore_faults);
      static Event merge_events(Event ev1, Event ev2,
				Event ev3 = Event::NO_EVENT, Event ev4 = Event::NO_EVENT,
				Event ev5 = Event::NO_EVENT, Event ev6 = Event::NO_EVENT);

      // record that the event has triggered and notify anybody who cares
      void trigger(Event::gen_t gen_triggered, int trigger_node, bool poisoned);

      // helper for triggering with an Event (which must be backed by a GenEventImpl)
      static void trigger(Event e, bool poisoned);

      // process an update message from the owner
      void process_update(Event::gen_t current_gen,
			  const Event::gen_t *new_poisoned_generations,
			  int new_poisoned_count);

    public: //protected:
      ID me;
      unsigned owner;
      
      // these state variables are monotonic, so can be checked without a lock for
      //  early-out conditions
      Event::gen_t generation, gen_subscribed;
      int num_poisoned_generations;
      bool has_local_triggers;

      bool is_generation_poisoned(Event::gen_t gen) const; // helper function - linear search

      // this is only manipulated when the event is "idle"
      GenEventImpl *next_free;

      // everything below here protected by this mutex
      GASNetHSL mutex;

      // local waiters are tracked by generation - an easily-accessed list is used
      //  for the "current" generation, whereas a map-by-generation-id is used for
      //  "future" generations (i.e. ones ahead of what we've heard about if we're
      //  not the owner)
      std::vector<EventWaiter *> current_local_waiters;
      std::map<Event::gen_t, std::vector<EventWaiter *> > future_local_waiters;

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
      Event::gen_t *poisoned_generations;

      // local triggerings - if we're not the owner, but we've triggered/poisoned events,
      //  we need to give consistent answers for those generations, so remember what we've
      //  done until our view of the distributed event catches up
      // value stored in map is whether generation was poisoned
      std::map<Event::gen_t, bool> local_triggers;
    };

    class BarrierImpl : public EventImpl {
    public:
      static const ID::ID_Types ID_TYPE = ID::ID_BARRIER;

      static const int BARRIER_TIMESTAMP_NODEID_SHIFT = 48;
      static Barrier::timestamp_t barrier_adjustment_timestamp;

      BarrierImpl(void);

      void init(ID _me, unsigned _init_owner);

      // get the Barrier (id+generation) for the current (i.e. untriggered) generation
      Barrier current_barrier(Barrier::timestamp_t timestamp = 0) const;

      // helper to create the Barrier for an arbitrary generation
      Barrier make_barrier(Event::gen_t gen, Barrier::timestamp_t timestamp = 0) const;

      static BarrierImpl *create_barrier(unsigned expected_arrivals, ReductionOpID redopid,
					 const void *initial_value = 0, size_t initial_value_size = 0);

      // test whether an event has triggered without waiting
      virtual bool has_triggered(Event::gen_t needed_gen, bool& poisoned);
      virtual void external_wait(Event::gen_t needed_gen, bool& poisoned);

      virtual bool add_waiter(Event::gen_t needed_gen, EventWaiter *waiter/*, bool pre_subscribed = false*/);

      // used to adjust a barrier's arrival count either up or down
      // if delta > 0, timestamp is current time (on requesting node)
      // if delta < 0, timestamp says which positive adjustment this arrival must wait for
      void adjust_arrival(Event::gen_t barrier_gen, int delta, 
			  Barrier::timestamp_t timestamp, Event wait_on,
			  gasnet_node_t sender, bool forwarded,
			  const void *reduce_value, size_t reduce_value_size);

      bool get_result(Event::gen_t result_gen, void *value, size_t value_size);

    public: //protected:
      ID me;
      unsigned owner;
      Event::gen_t generation, gen_subscribed;
      Event::gen_t first_generation;
      BarrierImpl *next_free;

      GASNetHSL mutex; // controls which local thread has access to internal data (not runtime-visible event)

      // class to track per-generation status
      class Generation {
      public:
	struct PerNodeUpdates {
	  Barrier::timestamp_t last_ts;
	  std::map<Barrier::timestamp_t, int> pending;
	};

	int unguarded_delta;
	std::vector<EventWaiter *> local_waiters;
	std::map<int, PerNodeUpdates *> pernode;
      
	
	Generation(void);
	~Generation(void);

	void handle_adjustment(Barrier::timestamp_t ts, int delta);
      };

      std::map<Event::gen_t, Generation *> generations;

      // a list of remote waiters and the latest generation they're interested in
      // also the latest generation that each node (that has ever subscribed) has been told about
      std::map<unsigned, Event::gen_t> remote_subscribe_gens, remote_trigger_gens;
      std::map<Event::gen_t, Event::gen_t> held_triggers;

      unsigned base_arrival_count;
      ReductionOpID redop_id;
      const ReductionOpUntyped *redop;
      char *initial_value;  // for reduction barriers

      unsigned value_capacity; // how many values the two allocations below can hold
      char *final_values;   // results of completed reductions
    };

  // active messages

  struct EventSubscribeMessage {
    struct RequestArgs {
      gasnet_node_t node;
      Event event;
      Event::gen_t previous_subscribe_gen;
    };

    static void handle_request(RequestArgs args);

    typedef ActiveMessageShortNoReply<EVENT_SUBSCRIBE_MSGID,
				      RequestArgs,
				      handle_request> Message;

    static void send_request(gasnet_node_t target, Event event, Event::gen_t previous_gen);
  };

  // EventTriggerMessage is used by non-owner nodes to trigger an event
  // EventUpdateMessage is used by the owner node to tell non-owner nodes about one or
  //   more triggerings of an event

  struct EventTriggerMessage {
    struct RequestArgs {
      gasnet_node_t node;
      Event event;
      bool poisoned;
    };

    static void handle_request(RequestArgs args);

    typedef ActiveMessageShortNoReply<EVENT_TRIGGER_MSGID,
				       RequestArgs,
				       handle_request> Message;

    static void send_request(gasnet_node_t target, Event event, bool poisoned);
  };

  struct EventUpdateMessage {
    struct RequestArgs : public BaseMedium {
      Event event;

      void apply(gasnet_node_t target);
    };

    static void handle_request(RequestArgs args, const void *data, size_t datalen);

    typedef ActiveMessageMediumNoReply<EVENT_UPDATE_MSGID,
				       RequestArgs,
				       handle_request> Message;

    static void send_request(gasnet_node_t target, Event event,
			     int num_poisoned, const Event::gen_t *poisoned_generations);
    static void broadcast_request(const NodeSet& targets, Event event,
				  int num_poisoned, const Event::gen_t *poisoned_generations);
  };

    struct BarrierAdjustMessage {
      struct RequestArgs : public BaseMedium {
	int sender;
	//bool forwarded;  no room to store this, so encoded as: sender < 0
	int delta;
	Barrier barrier;
        Event wait_on;
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<BARRIER_ADJUST_MSGID,
					 RequestArgs,
					 handle_request> Message;

      static void send_request(gasnet_node_t target, Barrier barrier, int delta, Event wait_on,
			       gasnet_node_t sender, bool forwarded,
			       const void *data, size_t datalen);
    };

    struct BarrierSubscribeMessage {
      struct RequestArgs {
	gasnet_node_t subscriber;
	ID::IDType barrier_id;
	Event::gen_t subscribe_gen;
	bool forwarded;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<BARRIER_SUBSCRIBE_MSGID,
					RequestArgs,
					handle_request> Message;

      static void send_request(gasnet_node_t target, ID::IDType barrier_id,
			       Event::gen_t subscribe_gen,
			       gasnet_node_t subscriber, bool forwarded);
    };

    struct BarrierTriggerMessage {
      struct RequestArgs : public BaseMedium {
	gasnet_node_t node;
	ID::IDType barrier_id;
	Event::gen_t trigger_gen;
	Event::gen_t previous_gen;
	Event::gen_t first_generation;
	ReductionOpID redop_id;
	gasnet_node_t migration_target;
	unsigned base_arrival_count;
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<BARRIER_TRIGGER_MSGID,
					 RequestArgs,
					 handle_request> Message;

      static void send_request(gasnet_node_t target, ID::IDType barrier_id,
			       Event::gen_t trigger_gen, Event::gen_t previous_gen,
			       Event::gen_t first_generation, ReductionOpID redop_id,
			       gasnet_node_t migration_target, unsigned base_arrival_count,
			       const void *data, size_t datalen);
    };

    struct BarrierMigrationMessage {
      struct RequestArgs {
	Barrier barrier;
	gasnet_node_t current_owner;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<BARRIER_MIGRATE_MSGID,
					RequestArgs,
					handle_request> Message;

      static void send_request(gasnet_node_t target, Barrier barrier, gasnet_node_t owner);
    };
	
}; // namespace Realm

#include "event_impl.inl"

#endif // ifndef REALM_EVENT_IMPL_H

