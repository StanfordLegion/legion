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

// events/barriers for Realm

#ifndef REALM_EVENT_H
#define REALM_EVENT_H

#include "lowlevel_config.h"

#include "redop.h"

namespace Realm {

    class Event {
    public:
      typedef ::legion_lowlevel_id_t id_t;
      typedef unsigned gen_t;

      id_t id;
      gen_t gen;
      bool operator<(const Event& rhs) const 
      { 
        if (id < rhs.id)
          return true;
        else if (id > rhs.id)
          return false;
        else
          return (gen < rhs.gen);
      }
      bool operator==(const Event& rhs) const { return (id == rhs.id) && (gen == rhs.gen); }
      bool operator!=(const Event& rhs) const { return (id != rhs.id) || (gen != rhs.gen); }

      static const Event NO_EVENT;

      bool exists(void) const { return id != 0; }

      // test whether an event has triggered without waiting
      bool has_triggered(void) const;

      // causes calling thread to block until event has occurred
      void wait(void) const;

      // used by non-legion threads to wait on an event - always blocking
      void external_wait(void) const;

      // creates an event that won't trigger until all input events have
      static Event merge_events(const std::set<Event>& wait_for);
      static Event merge_events(Event ev1, Event ev2,
				Event ev3 = NO_EVENT, Event ev4 = NO_EVENT,
				Event ev5 = NO_EVENT, Event ev6 = NO_EVENT);

      // the following calls are used to give Realm bounds on when the UserEvent
      //  will be triggered - in addition to being useful for diagnostic purposes
      //  (e.g. detecting event cycles), having a "late bound" (i.e. an event that
      //  is guaranteed to occur after the UserEvent is triggered) allows Realm to
      //  judge that the UserEvent trigger is "in flight"
      static void advise_event_ordering(Event happens_before, Event happens_after);
      static void advise_event_ordering(const std::set<Event>& happens_before,
					Event happens_after, bool all_must_trigger = true);
    };

    inline std::ostream& operator<<(std::ostream& os, Event e) { return os << std::hex << e.id << std::dec << '/' << e.gen; }

    // A user level event has all the properties of event, except
    // it can be triggered by the user.  This prevents users from
    // triggering arbitrary events without doing something like
    // an unsafe cast.
    class UserEvent : public Event {
    public:
      static UserEvent create_user_event(void);
      void trigger(Event wait_on = Event::NO_EVENT) const;

      static const UserEvent NO_USER_EVENT;
    };

    // a Barrier is similar to a UserEvent, except that it has a count of how
    //  many threads (or whatever) need to "trigger" before the actual trigger
    //  occurs
    class Barrier : public Event {
    public:
      typedef unsigned long long timestamp_t; // used to avoid race conditions with arrival adjustments

      timestamp_t timestamp;

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

//include "event.inl"

#endif // ifndef REALM_EVENT_H

