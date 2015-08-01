/* Copyright 2015 Stanford University, NVIDIA Corporation
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

    class EventWaiter {
    public:
      virtual ~EventWaiter(void) {}
      virtual bool event_triggered(void) = 0;
      virtual void print_info(FILE *f) = 0;
    };

    // parent class of GenEventImpl and BarrierImpl
    class EventImpl {
    public:
      // test whether an event has triggered without waiting
      virtual bool has_triggered(Event::gen_t needed_gen) = 0;

      // causes calling thread to block until event has occurred
      //void wait(Event::gen_t needed_gen);

      virtual void external_wait(Event::gen_t needed_gen) = 0;

      virtual bool add_waiter(Event::gen_t needed_gen, EventWaiter *waiter/*, bool pre_subscribed = false*/) = 0;

      static bool add_waiter(Event needed, EventWaiter *waiter);
    };

    class GenEventImpl : public EventImpl {
    public:
      static const ID::ID_Types ID_TYPE = ID::ID_EVENT;

      GenEventImpl(void);

      void init(ID _me, unsigned _init_owner);

      static GenEventImpl *create_genevent(void);

      // get the Event (id+generation) for the current (i.e. untriggered) generation
      Event current_event(void) const { Event e = me.convert<Event>(); e.gen = generation+1; return e; }

      // test whether an event has triggered without waiting
      virtual bool has_triggered(Event::gen_t needed_gen);

      virtual void external_wait(Event::gen_t needed_gen);

      virtual bool add_waiter(Event::gen_t needed_gen, EventWaiter *waiter/*, bool pre_subscribed = false*/);

      // creates an event that won't trigger until all input events have
      static Event merge_events(const std::set<Event>& wait_for);
      static Event merge_events(Event ev1, Event ev2,
				Event ev3 = Event::NO_EVENT, Event ev4 = Event::NO_EVENT,
				Event ev5 = Event::NO_EVENT, Event ev6 = Event::NO_EVENT);

      // record that the event has triggered and notify anybody who cares
      void trigger(Event::gen_t gen_triggered, int trigger_node, Event wait_on = Event::NO_EVENT);

      // if you KNOW you want to trigger the current event (which by definition cannot
      //   have already been triggered) - this is quicker:
      void trigger_current(void);

      void check_for_catchup(Event::gen_t implied_trigger_gen);

    public: //protected:
      ID me;
      unsigned owner;
      Event::gen_t generation, gen_subscribed;
      GenEventImpl *next_free;

      GASNetHSL mutex; // controls which local thread has access to internal data (not runtime-visible event)

      NodeSet remote_waiters;
      std::vector<EventWaiter *> local_waiters; // set of local threads that are waiting on event
    };

    class BarrierImpl : public EventImpl {
    public:
      static const ID::ID_Types ID_TYPE = ID::ID_BARRIER;

      static const int BARRIER_TIMESTAMP_NODEID_SHIFT = 48;
      static Barrier::timestamp_t barrier_adjustment_timestamp;

      BarrierImpl(void);

      void init(ID _me, unsigned _init_owner);

      static BarrierImpl *create_barrier(unsigned expected_arrivals, ReductionOpID redopid,
					 const void *initial_value = 0, size_t initial_value_size = 0);

      // test whether an event has triggered without waiting
      virtual bool has_triggered(Event::gen_t needed_gen);

      virtual void external_wait(Event::gen_t needed_gen);

      virtual bool add_waiter(Event::gen_t needed_gen, EventWaiter *waiter/*, bool pre_subscribed = false*/);

      // used to adjust a barrier's arrival count either up or down
      // if delta > 0, timestamp is current time (on requesting node)
      // if delta < 0, timestamp says which positive adjustment this arrival must wait for
      void adjust_arrival(Event::gen_t barrier_gen, int delta, 
			  Barrier::timestamp_t timestamp, Event wait_on,
			  const void *reduce_value, size_t reduce_value_size);

      bool get_result(Event::gen_t result_gen, void *value, size_t value_size);

    public: //protected:
      ID me;
      unsigned owner;
      Event::gen_t generation, gen_subscribed;
      Event::gen_t first_generation, free_generation;
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

  struct EventTriggerMessage {
    struct RequestArgs {
      gasnet_node_t node;
      Event event;

      void apply(gasnet_node_t target);
    };

    static void handle_request(RequestArgs args);

    typedef ActiveMessageShortNoReply<EVENT_TRIGGER_MSGID,
				      RequestArgs,
				      handle_request> Message;

    static void send_request(gasnet_node_t target, Event event);
    static void broadcast_request(const NodeSet& targets, Event event);
  };

    struct BarrierAdjustMessage {
      struct RequestArgs : public BaseMedium {
	Barrier barrier;
	int delta;
        Event wait_on;
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<BARRIER_ADJUST_MSGID,
					 RequestArgs,
					 handle_request> Message;

      static void send_request(gasnet_node_t target, Barrier barrier, int delta, Event wait_on,
			       const void *data, size_t datalen);
    };

    struct BarrierSubscribeMessage {
      struct RequestArgs {
	gasnet_node_t node;
	ID::IDType barrier_id;
	Event::gen_t subscribe_gen;
      };

      static void handle_request(RequestArgs args);

      typedef ActiveMessageShortNoReply<BARRIER_SUBSCRIBE_MSGID,
					RequestArgs,
					handle_request> Message;

      static void send_request(gasnet_node_t target, ID::IDType barrier_id, Event::gen_t subscribe_gen);
    };

    struct BarrierTriggerMessage {
      struct RequestArgs : public BaseMedium {
	gasnet_node_t node;
	ID::IDType barrier_id;
	Event::gen_t trigger_gen;
	Event::gen_t previous_gen;
	Event::gen_t first_generation;
	ReductionOpID redop_id;
      };

      static void handle_request(RequestArgs args, const void *data, size_t datalen);

      typedef ActiveMessageMediumNoReply<BARRIER_TRIGGER_MSGID,
					 RequestArgs,
					 handle_request> Message;

      static void send_request(gasnet_node_t target, ID::IDType barrier_id,
			       Event::gen_t trigger_gen, Event::gen_t previous_gen,
			       Event::gen_t first_generation, ReductionOpID redop_id,
			       const void *data, size_t datalen);
    };

	
}; // namespace Realm

//include "event_impl.inl"

#endif // ifndef REALM_EVENT_IMPL_H

