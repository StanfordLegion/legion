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

// events/barriers for Realm

#ifndef REALM_EVENT_H
#define REALM_EVENT_H

#include "realm/realm_c.h"

#include <vector>
#include <set>
#include <iostream>

namespace Realm {

  typedef ::realm_reduction_op_id_t ReductionOpID;

    class REALM_PUBLIC_API Event {
    public:
      typedef ::realm_id_t id_t;

      id_t id;
      bool operator<(const Event& rhs) const;
      bool operator==(const Event& rhs) const;
      bool operator!=(const Event& rhs) const;

      static const Event NO_EVENT;

      bool exists(void) const;

      // test whether an event has triggered without waiting
      bool has_triggered(void) const;

      // causes calling thread to block until event has occurred
      void wait(void) const;

      // used by non-legion threads to wait on an event - always blocking
      void external_wait(void) const;

      // external wait with a timeout - returns true if event triggers, false
      //  if the maximum delay occurs first
      bool external_timedwait(long long max_ns) const;

      // fault-aware versions of the above (the above versions will cause the
      //  caller to fault as well if a poisoned event is queried)
      bool has_triggered_faultaware(bool& poisoned) const;
      void wait_faultaware(bool& poisoned) const;
      void external_wait_faultaware(bool& poisoned) const;
      bool external_timedwait_faultaware(bool& poisoned, long long max_ns) const;

      // subscribes to an event, ensuring the triggeredness of it will be
      //  available as soon as possible (and without having to call wait)
      void subscribe(void) const;

      // attempts to cancel the operation associated with this event
      // "reason_data" will be provided to any profilers of the operation
      void cancel_operation(const void *reason_data, size_t reason_size) const;

      // attempts to change the priority of the operation associated with
      //  this event
      void set_operation_priority(int new_priority) const;
 
      // creates an event that won't trigger until all input events have
      static Event merge_events(const std::set<Event>& wait_for);
      static Event merge_events(const std::vector<Event>& wait_for);
      static Event merge_events(Event ev1, Event ev2,
				Event ev3 = NO_EVENT, Event ev4 = NO_EVENT,
				Event ev5 = NO_EVENT, Event ev6 = NO_EVENT);

      // normal merged events propagate poison - this version ignores poison on
      //  inputs - use carefully!
      static Event merge_events_ignorefaults(const std::set<Event>& wait_for);
      static Event merge_events_ignorefaults(const std::vector<Event>& wait_for);
      static Event ignorefaults(Event wait_for);

      // the following calls are used to give Realm bounds on when the UserEvent
      //  will be triggered - in addition to being useful for diagnostic purposes
      //  (e.g. detecting event cycles), having a "late bound" (i.e. an event that
      //  is guaranteed to occur after the UserEvent is triggered) allows Realm to
      //  judge that the UserEvent trigger is "in flight"
      static void advise_event_ordering(Event happens_before, Event happens_after);
      static void advise_event_ordering(const std::set<Event>& happens_before,
					Event happens_after, bool all_must_trigger = true);
    };

    // A user level event has all the properties of event, except
    // it can be triggered by the user.  This prevents users from
    // triggering arbitrary events without doing something like
    // an unsafe cast.
    class REALM_PUBLIC_API UserEvent : public Event {
    public:
      static UserEvent create_user_event(void);

      void trigger(Event wait_on = Event::NO_EVENT,
		   bool ignore_faults = false) const;

      // cancels (poisons) the event
      void cancel(void) const;

      static const UserEvent NO_USER_EVENT;
    };

    // a Barrier is similar to a UserEvent, except that it has a count of how
    //  many threads (or whatever) need to "trigger" before the actual trigger
    //  occurs
    class REALM_PUBLIC_API Barrier : public Event {
    public:
      typedef ::realm_barrier_timestamp_t timestamp_t; // used to avoid race conditions with arrival adjustments

      timestamp_t timestamp;

      static const Barrier NO_BARRIER;

      static Barrier create_barrier(unsigned expected_arrivals, ReductionOpID redop_id = 0,
				    const void *initial_value = 0, size_t initial_value_size = 0);
      void destroy_barrier(void);

      static const ::realm_event_gen_t MAX_PHASES;

      // barriers can be reused up to MAX_PHASES times by using "advance_barrier"
      //  to advance a Barrier handle to the next phase - attempts to advance
      //  beyond the last phase return NO_BARRIER instead
      Barrier advance_barrier(void) const;
      Barrier alter_arrival_count(int delta) const;
      Barrier get_previous_phase(void) const;

      void arrive(unsigned count = 1, Event wait_on = Event::NO_EVENT,
		  const void *reduce_value = 0, size_t reduce_value_size = 0) const;

      bool get_result(void *value, size_t value_size) const;
    };

    // a CompletionQueue funnels the completion of unordered events into a
    //  single stream that can be queried (and waited on) by a single servicer
    class REALM_PUBLIC_API CompletionQueue {
    public:
      typedef ::realm_id_t id_t;

      id_t id;
      bool operator<(const CompletionQueue& rhs) const;
      bool operator==(const CompletionQueue& rhs) const;
      bool operator!=(const CompletionQueue& rhs) const;

      static const CompletionQueue NO_QUEUE;

      bool exists(void) const;

      // creates a completion queue that can hold at least 'max_size'
      //  triggered events (at the moment, overflow is a fatal error)
      // a 'max_size' of 0 allows for arbitrary queue growth, at the cost
      //  of additional overhead
      static CompletionQueue create_completion_queue(size_t max_size);

      // destroy a completion queue
      void destroy(Event wait_on = Event::NO_EVENT);

      // adds an event to the completion queue (once it triggers)
      // non-faultaware version raises a fatal error if the specified 'event'
      //  is poisoned
      void add_event(Event event);
      void add_event_faultaware(Event event);

      // requests up to 'max_events' triggered events to be popped from the
      //  queue and stored in the provided 'events' array (if null, the
      //  identities of the triggered events are discarded)
      // this call returns the actual number of events popped, which may be
      //  zero (this call is nonblocking)
      // when 'add_event_faultaware' is used, any poisoning of the returned
      //  events is not signalled explicitly - the caller is expected to
      //  check via 'has_triggered_faultaware' itself
      size_t pop_events(Event *events, size_t max_events);

      // get an event that, once triggered, guarantees that (at least) one
      //  call to pop_events made since the non-empty event was requested
      //  will return a non-zero number of triggered events
      // once a call to pop_events has been made (by the caller of
      //  get_nonempty_event or anybody else), the guarantee is lost and
      //  a new non-empty event must be requested
      // note that 'get_nonempty_event().has_triggered()' is unlikely to
      //  ever return 'true' if called from a node other than the one that
      //  created the completion queue (i.e. the query at least has the
      //  round-trip network communication latency to deal with) - if polling
      //  on the completion queue is unavoidable, the loop should poll on
      //  pop_events directly
      Event get_nonempty_event(void);
    };
    

}; // namespace Realm

#include "realm/event.inl"

#endif // ifndef REALM_EVENT_H

