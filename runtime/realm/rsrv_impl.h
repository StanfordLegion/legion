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

// implementation of Reservations for Realm

#ifndef REALM_RSRV_IMPL_H
#define REALM_RSRV_IMPL_H

#include "realm/reservation.h"

#include "realm/id.h"
#include "realm/activemsg.h"
#include "realm/nodeset.h"

#define REALM_RSRV_USE_CIRCQUEUE
#ifdef REALM_RSRV_USE_CIRCQUEUE
#include "realm/circ_queue.h"
#endif

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

      void init(Reservation _me, unsigned _init_owner, size_t _data_size = 0);

      template <class T>
      void set_local_data(T *data)
      {
	local_data = data;
	local_data_size = sizeof(T);
        own_local = false;
      }

      //protected:
      Reservation me;
      NodeID owner; // which node owns the lock
      unsigned count; // number of locks held by local threads
      unsigned mode;  // lock mode
      bool in_use;

      enum { MODE_EXCL = 0, ZERO_COUNT = 0x11223344 };

      GASNetHSL mutex; // controls which local thread has access to internal data (not runtime-visible lock)

      // bitmasks of which remote nodes are waiting on a lock (or sharing it)
      NodeSet remote_waiter_mask, remote_sharer_mask;
      //std::list<LockWaiter *> local_waiters; // set of local threads that are waiting on lock

#ifdef REALM_RSRV_USE_CIRCQUEUE
      typedef CircularQueue<Event> WaiterList;
#else
      typedef std::deque<Event> WaiterList;
#endif

      std::map<unsigned, WaiterList> local_waiters;
      std::map<unsigned, unsigned> retry_count;
      std::map<unsigned, Event> retry_events;
      bool requested; // do we have a request for the lock in flight?

      // local data protected by lock
      void *local_data;
      size_t local_data_size;
      bool own_local;

      static GASNetHSL freelist_mutex;
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

      bool select_local_waiters(WaiterList& to_wake);

      void release(void);

      bool is_locked(unsigned check_mode, bool excl_ok);

      void release_reservation(void);

      struct PackFunctor {
      public:
        PackFunctor(int *p) : pos(p) { }
      public:
        inline void apply(int target) { *pos++ = target; }
      public:
        int *pos;
      };
    };

    template <typename T>
    class StaticAccess {
    public:
      typedef typename T::StaticData StaticData;

      StaticAccess(T* thing_with_data, bool already_valid = false);

      ~StaticAccess(void) {}

      const StaticData *operator->(void) { return data; }

    protected:
      StaticData *data;
    };

    template <typename T>
    class SharedAccess {
    public:
      typedef typename T::CoherentData CoherentData;

      SharedAccess(T* thing_with_data, bool already_held = false);

      ~SharedAccess(void)
      {
	lock->release();
      }

      const CoherentData *operator->(void) { return data; }

    protected:
      CoherentData *data;
      ReservationImpl *lock;
    };

    template <class T>
    class ExclusiveAccess {
    public:
      typedef typename T::CoherentData CoherentData;

      ExclusiveAccess(T* thing_with_data, bool already_held = false);

      ~ExclusiveAccess(void)
      {
	lock->release();
      }

      CoherentData *operator->(void) { return data; }

    protected:
      CoherentData *data;
      ReservationImpl *lock;
    };

  // active messages

  struct LockRequestMessage {
    struct RequestArgs {
      NodeID node;
      Reservation lock;
      unsigned mode;
    };

    static void handle_request(RequestArgs args);

    typedef ActiveMessageShortNoReply<LOCK_REQUEST_MSGID, 
				      RequestArgs, 
				      handle_request> Message;

    static void send_request(NodeID target, NodeID req_node,
			     Reservation lock, unsigned mode);
  };

  struct LockReleaseMessage {
    struct RequestArgs {
      NodeID node;
      Reservation lock;
    };
    
    static void handle_request(RequestArgs args);

    typedef ActiveMessageShortNoReply<LOCK_RELEASE_MSGID,
				      RequestArgs,
				      handle_request> Message;

    static void send_request(NodeID target, Reservation lock);
  };

  struct LockGrantMessage {
    struct RequestArgs : public BaseMedium {
      Reservation lock;
      unsigned mode;
    };

    static void handle_request(RequestArgs args, const void *data, size_t datalen);

    typedef ActiveMessageMediumNoReply<LOCK_GRANT_MSGID,
				       RequestArgs,
				       handle_request> Message;

    static void send_request(NodeID target, Reservation lock,
			     unsigned mode, const void *data, size_t datalen,
			     int payload_mode);
  };

  struct DestroyLockMessage {
    struct RequestArgs {
      Reservation actual;
      Reservation dummy;
    };

    static void handle_request(RequestArgs args);

    typedef ActiveMessageShortNoReply<DESTROY_LOCK_MSGID,
				      RequestArgs,
				      handle_request> Message;

    static void send_request(NodeID target, Reservation lock);
  };

}; // namespace Realm

#include "realm/rsrv_impl.inl"

#endif // ifndef REALM_RSRV_IMPL_H

