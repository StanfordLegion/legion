/* Copyright 2023 Stanford University, NVIDIA Corporation
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

/**
 * \file event.h
 * This file provides a C++ interface to the Realm events.
 */

namespace Realm {

  typedef ::realm_reduction_op_id_t ReductionOpID;

  /**
   * \class Event
   * Event is created by the runtime and is used to synchronize
   * operations.  An event is triggered when the operation it
   * represents is complete and can be used as pre and post conditions
   * for other operations. This class represents a handle to the event
   * itself and can be passed-by-value as well as
   * serialized/deserialized anywhere in the program. Note that events
   * do not need to be explicitly garbage collected.
   */
    class REALM_PUBLIC_API Event {
    public:
      typedef ::realm_id_t id_t;

      id_t id;
      bool operator<(const Event& rhs) const;
      bool operator==(const Event& rhs) const;
      bool operator!=(const Event& rhs) const;


      /**
       * \brief The value should be usued to initialize an event
       * handle. NO_EVENT is always in has triggered state .
       */
      static const Event NO_EVENT;

      /**
       * Check whether an event has a valid ID.
       * @return true if the event has a valid ID, false otherwise
       */
      bool exists(void) const;

      /**
       * Test whether an event has triggered without waiting.
       * @return true if the event has triggered, false otherwise
       */
      bool has_triggered(void) const;


      /**
       * Wait for an event to trigger.
       */
      void wait(void) const;

      /**
       * Used by non-legion threads to wait on an event - always blocking
       */
      void external_wait(void) const;

      /**
       * External wait with a timeout - returns true if event triggers, false
       * if the maximum delay occurs first
       * @param max_ns the maximum number of nanoseconds to wait
       * @return true if the event has triggered, false if the timeout occurred
       */
      bool external_timedwait(long long max_ns) const;

      /**
       * Fault-aware versions of the above (the above versions will cause the
       * caller to fault as well if a poisoned event is queried).
       * @param poisoned set to true if the event is poisoned
       * @return true if the event has triggered, false otherwise
       */
      bool has_triggered_faultaware(bool& poisoned) const;


      /**
       * Fault-aware versions of the wait function.
       * @param poisoned set to true if the event is poisoned
       * @return true if the event has triggered, false otherwise
       */
      void wait_faultaware(bool& poisoned) const;

      /**
       * Fault-aware versions of the external wait function.
       * @param poisoned set to true if the event is poisoned
       * @return true if the event has triggered, false otherwise
       */
      void external_wait_faultaware(bool& poisoned) const;

      /**
       * Fault-aware versions of the external timed wait function.
       * @param poisoned set to true if the event is poisoned
       * @param max_ns the maximum number of nanoseconds to wait
       * @return true if the event has triggered, false if the timeout occurred
       */
      bool external_timedwait_faultaware(bool& poisoned, long long max_ns) const;

      /**
       * Subscribe to an event, ensuring that the triggeredness of the
       * event will be available as soon as possible (and without having to call
       * wait).
       */
      void subscribe(void) const;

      /**
       * Attempt to cancel the operation associated with this event.
       * @param reason_data will be provided to any profilers of the operation
       * @param reason_size the size of the reason data
       */
      void cancel_operation(const void *reason_data, size_t reason_size) const;

      /**
       * Attempt to change the priority of the operation associated with this
       * event.
       * @param new_priority the new priority.
       */
      void set_operation_priority(int new_priority) const;

      ///@{
      /**
       * Create an event that won't trigger until all input events
       * have.
       * @param wait_for the events to wait for
       * @return the event that will trigger when all input events
       * have.
       */
      static Event merge_events(const std::set<Event>& wait_for);
      static Event merge_events(const std::vector<Event>& wait_for);
      static Event merge_events(Event ev1, Event ev2,
				Event ev3 = NO_EVENT, Event ev4 = NO_EVENT,
				Event ev5 = NO_EVENT, Event ev6 = NO_EVENT);
      ///@}

      /**
       * Create an event that won't trigger until all input events
       * have, ignoring any poison on the input events.
       * @param wait_for the events to wait for
       * @return the event that will trigger when all input events
       * have.
       */
      static Event merge_events_ignorefaults(const std::set<Event>& wait_for);
      static Event merge_events_ignorefaults(const std::vector<Event>& wait_for);
      static Event ignorefaults(Event wait_for);

      /**
       * The following call is used to give Realm a bound on when the UserEvent
       * will be triggered.  In addition to being useful for diagnostic purposes
       * (e.g. detecting event cycles), having a "happens_before" allows Realm
       * to judge that the UserEvent trigger is "in flight".
       * @param happens_before the event that must occur before the UserEvent
       * @param happens_after the event that must occur after the UserEvent
       */
      static void advise_event_ordering(Event happens_before, Event happens_after);
      static void advise_event_ordering(const std::set<Event>& happens_before,
					Event happens_after, bool all_must_trigger = true);
    };

    /**
     * \class UserEvent
     * UserEvents are events that can be scheduled to trigger at a
     * future point in an application and can be waited upon
     * dynamically. This is in contrast to a Realm::Event, which must
     * be a direct consequence of the completion of a specific
     * operation.
     */
    class REALM_PUBLIC_API UserEvent : public Event {
    public:

      /**
       * Create a new user event.
       * @return the new user event
       */
      static UserEvent create_user_event(void);

      /**
       * Trigger the user event.
       * @param wait_on an event that must trigger before this user event
       * can.
       * @param ignore_faults if true, the user event will be triggered even if
       * the event it is waiting on is poisoned.
       */
      void trigger(Event wait_on = Event::NO_EVENT,
                   bool ignore_faults = false) const;

      /*
       * Attempt to cancell all the operations waiting on this user
       * event.
       */
      void cancel(void) const;

      static const UserEvent NO_USER_EVENT;
    };

    /**
     * \class Barrier
     * A barrier is similar to a user event, except that it has a count
     * of how many threads (or whatever) need to "trigger" before the
     * actual trigger occurs.
     */
    class REALM_PUBLIC_API Barrier : public Event {
    public:
      typedef ::realm_barrier_timestamp_t timestamp_t; // used to avoid race conditions with arrival adjustments

      timestamp_t timestamp;

      static const Barrier NO_BARRIER;

      static Barrier create_barrier(unsigned expected_arrivals, ReductionOpID redop_id = 0,
				    const void *initial_value = 0, size_t initial_value_size = 0);
      void destroy_barrier(void);

      static const ::realm_event_gen_t MAX_PHASES;

      /*
       * Advance a barrier to the next phase, returning a new barrier
       * handle. Attemps to advance beyond the last phase return NO_BARRIER
       * instead.
       * @return the new barrier handle.
       */
      Barrier advance_barrier(void) const;

      /*
       * Alter the arrival count of a barrier.
       * @param delta the amount to adjust the arrival count by
       * @return the new barrier handle.
       */
      Barrier alter_arrival_count(int delta) const;

      /*
       * Get the previous phase of a barrier.
       * @return the previous phase of the barrier
       */
      Barrier get_previous_phase(void) const;

      /*
       * Adjust the arrival count of a barrier.
       * @param count the amount to adjust the arrival count by
       * @param wait_on an event that must trigger before the arrival count
       * can be adjusted.
       * @param ignore_faults if true, the arrival count will be adjusted even
       * if the event it is waiting on is poisoned.
       * @param reduce_value if non-null, the value will be used to update the
       * reduction value associated with the barrier.
       * @param reduce_value_size the size of the reduction value
       */
      void arrive(unsigned count = 1, Event wait_on = Event::NO_EVENT,
                  const void* reduce_value = 0,
                  size_t reduce_value_size = 0) const;


      /*
       * Get the resulting barrier value.
       * @param value the resulting value
       * @param value_size the size of the resulting value
       * @return true if the value was successfully retrieved,
       * generation hasn't triggered yet.
       */
      bool get_result(void *value, size_t value_size) const;
    };


    /**
     * \class CompletionQueue
     * A completion queue funnels the completion of unordered events into a
     * single queue that can be queried (and waited on) by a single servicer
     * task.
     */
    class REALM_PUBLIC_API CompletionQueue {
    public:
      typedef ::realm_id_t id_t;

      id_t id;
      bool operator<(const CompletionQueue& rhs) const;
      bool operator==(const CompletionQueue& rhs) const;
      bool operator!=(const CompletionQueue& rhs) const;

      static const CompletionQueue NO_QUEUE;

      bool exists(void) const;

      /**
       * Create a completion queue that can hold at least 'max_size'
       * triggered events (at the moment, overflow is a fatal error).
       * A 'max_size' of 0 allows for arbitrary queue growth, at the cost
       * of additional overhead.
       * @param max_size the maximum size of the queue
       * @return the completion queue
       */
      static CompletionQueue create_completion_queue(size_t max_size);

      /**
       * Destroy a completion queue.
       * @param wait_on an event to wait on before destroying the
       * queue.
       */
      void destroy(Event wait_on = Event::NO_EVENT);

      ///@{
      /**
       * Add an event to the completion queue (once it triggers).
       * non-faultaware version raises a fatal error if the specified 'event'
       * is poisoned
       * @param event the event to add
       */
      void add_event(Event event);
      void add_event_faultaware(Event event);
      ///@}

      /**
       * Requests up to 'max_events' triggered events to be popped from the
       * queue and stored in the provided 'events' array (if null, the
       * identities of the triggered events are discarded).
       * This call returns the actual number of events popped, which may be
       * zero (this call is nonblocking).
       * When 'add_event_faultaware' is used, any poisoning of the returned
       * events is not signalled explicitly - the caller is expected to
       * check via 'has_triggered_faultaware' itself.
       * @param events the array to store the events in
       * @param max_events the maximum number of events to pop
       * @return the number of events popped
       */
      size_t pop_events(Event *events, size_t max_events);

      /**
       * Get an event that, once triggered, guarantees that (at least) one
       * call to pop_events made since the non-empty event was requested
       * will return a non-zero number of triggered events.
       * Once a call to pop_events has been made (by the caller of
       * get_nonempty_event or anybody else), the guarantee is lost and
       * a new non-empty event must be requested.
       * Note that 'get_nonempty_event().has_triggered()' is unlikely to
       * ever return 'true' if called from a node other than the one that
       * created the completion queue (i.e. the query at least has the
       * round-trip network communication latency to deal with) - if polling
       * on the completion queue is unavoidable, the loop should poll on
       * pop_events directly.
       * @return the non-empty event
       */
      Event get_nonempty_event(void);
    };
    

}; // namespace Realm

#include "realm/event.inl"

#endif // ifndef REALM_EVENT_H

