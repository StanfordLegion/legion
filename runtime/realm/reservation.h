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

// reservations for Realm

#ifndef REALM_RESERVATION_H
#define REALM_RESERVATION_H

#include "realm/realm_c.h"

#include "realm/event.h"

namespace Realm {

    class Reservation {
    public:
      typedef ::realm_id_t id_t;
      id_t id;
      bool operator<(const Reservation& rhs) const { return id < rhs.id; }
      bool operator==(const Reservation& rhs) const { return id == rhs.id; }
      bool operator!=(const Reservation& rhs) const { return id != rhs.id; }

      static const Reservation NO_RESERVATION;

      bool exists(void) const { return id != 0; }

      // requests ownership (either exclusive or shared) of the reservation with a 
      //   specified mode - returns an event that will trigger when the reservation 
      //   is granted
      Event acquire(unsigned mode = 0, bool exclusive = true, Event wait_on = Event::NO_EVENT) const;

      // tries to acquire ownership of the reservation with the given 'mode' and 'exclusive'ity
      // if immediately successful, returns Event::NO_EVENT - check with exists(), not has_triggered()!
      // if not, the reservation is NOT acquired (ever), and it returns an Event that should be
      //  allowed to trigger before the caller tries again - also, the caller MUST retry until successful,
      //  setting 'retry' to true on subsequent attempts
      Event try_acquire(bool retry, unsigned mode = 0, bool exclusive = true,
			Event wait_on = Event::NO_EVENT) const;

      // releases a held reservation - release can be deferred until an event triggers
      void release(Event wait_on = Event::NO_EVENT) const;

      // Create a new reservation, destroy an existing reservation 
      static Reservation create_reservation(size_t _data_size = 0);
      void destroy_reservation();

      size_t data_size(void) const;
      void *data_ptr(void) const;
    };

    inline std::ostream& operator<<(std::ostream& os, Reservation r) { return os << std::hex << r.id << std::dec; }

  // a FastReservation is a wrapper around a Reservation that signficantly
  //  reduces overhead for repeated use of the same Reservation within a
  //  (coherent) address space.  Unlike Reservation's, a FastReservation is
  //  an opaque object that is not copyable or transferrable.  However, a
  //  FastReservation _does_ play nice with other users of the underlying
  //  Reservation, on the same node or on others

  class FastReservation {
  public:
    FastReservation(Reservation _rsrv = Reservation::NO_RESERVATION);
    ~FastReservation(void);

  private:
    // NOT copyable
    FastReservation(const FastReservation&);
    FastReservation& operator=(const FastReservation&);

  public:
    enum WaitMode {
      SPIN,           // keep trying until lock is available
      ALWAYS_SPIN,    // spin, even if holder is suspended (DANGEROUS)
      WAIT,           // wait on Realm::Event, allowing other tasks to run
      EXTERNAL_WAIT,  // wait on kernel mutex, for non-Realm threads
    };

    // these are inlined and cover the fast (i.e. uncontended) cases - they
    //  automatically fall back to the ..._slow versions below when needed
    // if the lock cannot be obtained without waiting, these calls return an
    //  Event that the caller should wait on before trying again
    // Note #1: use exists() (not has_triggered()!) to check if the acquisition
    //            failed and needs to be retried
    // Note #2: unlike try_acquire above, the caller is NOT required to try again
    Event lock(WaitMode mode = SPIN); // synonym for wrlock()
    Event wrlock(WaitMode mode = SPIN);
    Event rdlock(WaitMode mode = SPIN);
    void unlock(void);

    // in general, a thread holding a lock should not go to sleep on other
    //  events, as another thread waiting on the lock may spin and prevent the
    //  holding thread from running on that core
    // to avoid this, a thread holding the lock that intends to sleep should
    //  create a UserEvent and advise on entry/exit from the region in which
    //  sleeping might occur (do not provide the event the holder is waiting
    //  for - the user event must be triggered after the advise_sleep_exit call)
    void advise_sleep_entry(UserEvent guard_event);
    void advise_sleep_exit(void);

    // the fast path stores several things in a 32-bit word that can be
    //  atomically updated
    typedef uint32_t State;
    static const State STATE_READER_COUNT_MASK = 0x03ffffff;
    static const State STATE_SLEEPER           = 0x04000000;
    static const State STATE_WRITER            = 0x08000000;
    static const State STATE_WRITER_WAITING    = 0x10000000;
    static const State STATE_BASE_RSRV         = 0x20000000;
    static const State STATE_BASE_RSRV_WAITING = 0x40000000;
    static const State STATE_SLOW_FALLBACK     = 0x80000000;

  protected:
    Event wrlock_slow(WaitMode mode);
    Event rdlock_slow(WaitMode mode);
    void unlock_slow(void);

    // we will make use of atomics for the fast path, so make sure we take
    //  a full cache line for our data to avoid false sharing
    static const size_t CACHE_LINE_SIZE = 64;
    union {
      State state;
      // this is slightly fragile, but we want to have enough room to store
      //  the implementation-specific stuff without another layer of
      //  indirection
      char opaque[CACHE_LINE_SIZE * 4];
    } __attribute__((aligned(16)));
  };
	
}; // namespace Realm

#include "reservation.inl"

#endif // ifndef REALM_RESERVATION_H

