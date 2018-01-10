/* Copyright 2018 Stanford University, NVIDIA Corporation
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

    class Event {
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

      // fault-aware versions of the above (the above versions will cause the
      //  caller to fault as well if a poisoned event is queried)
      bool has_triggered_faultaware(bool& poisoned) const;
      void wait_faultaware(bool& poisoned) const;
      void external_wait_faultaware(bool& poisoned) const;

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
    class UserEvent : public Event {
    public:
      static UserEvent create_user_event(void);
      void trigger(Event wait_on = Event::NO_EVENT) const;

      // cancels (poisons) the event
      void cancel(void) const;

      static const UserEvent NO_USER_EVENT;
    };

    // a Barrier is similar to a UserEvent, except that it has a count of how
    //  many threads (or whatever) need to "trigger" before the actual trigger
    //  occurs
    class Barrier : public Event {
    public:
      typedef ::realm_barrier_timestamp_t timestamp_t; // used to avoid race conditions with arrival adjustments

      timestamp_t timestamp;

      static const Barrier NO_BARRIER;

      static Barrier create_barrier(unsigned expected_arrivals, ReductionOpID redop_id = 0,
				    const void *initial_value = 0, size_t initial_value_size = 0);
      void destroy_barrier(void);

      Barrier advance_barrier(void) const;
      Barrier alter_arrival_count(int delta) const;
      Barrier get_previous_phase(void) const;

      void arrive(unsigned count = 1, Event wait_on = Event::NO_EVENT,
		  const void *reduce_value = 0, size_t reduce_value_size = 0) const;

      bool get_result(void *value, size_t value_size) const;
    };


}; // namespace Realm

#include "realm/event.inl"

#endif // ifndef REALM_EVENT_H

