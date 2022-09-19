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

// implementation of Reservations for Realm

#ifndef REALM_RSRV_IMPL_H
#define REALM_RSRV_IMPL_H

#include "realm/reservation.h"

#include "realm/id.h"
#include "realm/network.h"
#include "realm/nodeset.h"
#include "realm/mutex.h"
#include "realm/bgwork.h"
#include "realm/event_impl.h"

namespace Realm {

#ifdef LOCK_TRACING
    // For lock tracing
    struct LockTraceItem {
    public:
      enum Action {
        ACT_LOCAL_REQUEST = 0, // request for a lock where the owner is local
        ACT_REMOTE_REQUEST = 1, // request for a lock where the owner is not local
        ACT_FORWARD_REQUEST = 2, // for forwarding of requests
        ACT_LOCAL_GRANT = 3, // local grant of the lock
        ACT_REMOTE_GRANT = 4, // remote grant of the lock (change owners)
        ACT_REMOTE_RELEASE = 5, // remote release of a shared lock
      };
    public:
      unsigned time_units, lock_id, owner, action;
    };
#endif

    namespace Config {
      extern bool use_fast_reservation_fallback;
    };

    class ReservationImpl {
    public:
      ReservationImpl(void);

      static const ID::ID_Types ID_TYPE = ID::ID_LOCK;

      void init(Reservation _me, unsigned _init_owner);

      //protected:
      Reservation me;
      NodeID owner; // which node owns the lock
      unsigned count; // number of locks held by local threads
      unsigned mode;  // lock mode
      bool in_use;

      enum { MODE_EXCL = 0, ZERO_COUNT = 0x11223344 };

      Mutex mutex; // controls which local thread has access to internal data (not runtime-visible lock)

      // bitmasks of which remote nodes are waiting on a lock (or sharing it)
      NodeSet remote_waiter_mask, remote_sharer_mask;
      //std::list<LockWaiter *> local_waiters; // set of local threads that are waiting on lock

      typedef EventWaiter::EventWaiterList WaiterList;

      WaiterList local_excl_waiters;

      struct LocalSharedInfo {
	unsigned count;
	WaiterList waiters;
      };
      std::map<unsigned, LocalSharedInfo> local_shared;

      struct RetryInfo {
	unsigned count;
	Event event;
      };
      std::map<unsigned, RetryInfo> retries;
      bool requested; // do we have a request for the lock in flight?

      static Mutex freelist_mutex;
      static ReservationImpl *first_free;
      ReservationImpl *next_free;

      enum AcquireType {
	// normal Reservation::acquire() - returns an event when the reservation is granted
	ACQUIRE_BLOCKING,

	// Reservation::try_acquire() - grants immediately or returns an event for when a retry should be performed
	ACQUIRE_NONBLOCKING,

	// a retried version of try_acquire()
	ACQUIRE_NONBLOCKING_RETRY,

	// used when the try_acquire is preconditioned on something else first, so we never grant, but record a retry'er
	ACQUIRE_NONBLOCKING_PLACEHOLDER,
      };

      // creates an Event if needed to describe when reservation is granted
      Event acquire(unsigned new_mode, bool exclusive,
		    AcquireType acquire_type,
		    Event after_lock = Event::NO_EVENT);

      bool select_local_waiters(WaiterList& to_wake, Event& retry);

      void release(TimeLimit work_until);

      bool is_locked(unsigned check_mode, bool excl_ok);

      void release_reservation(void);
    };

  // active messages
  struct LockRequestMessage {
    NodeID node;
    Reservation lock;
    unsigned mode;

    static void handle_message(NodeID sender,const LockRequestMessage &msg,
			       const void *data, size_t datalen);
  };

  struct LockReleaseMessage {
    NodeID node;
    Reservation lock;
    
    static void handle_message(NodeID sender,const LockReleaseMessage &msg,
			       const void *data, size_t datalen);
  };

  struct LockGrantMessage {
    Reservation lock;
    unsigned mode;

    static void handle_message(NodeID sender,const LockGrantMessage &msg,
			       const void *data, size_t datalen,
			       TimeLimit work_until);
  };

  struct DestroyLockMessage {
    Reservation actual;
    Reservation dummy;
    Event wait_on;

    static void handle_message(NodeID sender,const DestroyLockMessage &msg,
			       const void *data, size_t datalen);
  };

}; // namespace Realm

#endif // ifndef REALM_RSRV_IMPL_H

