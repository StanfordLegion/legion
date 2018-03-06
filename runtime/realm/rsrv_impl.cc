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

#include "realm/rsrv_impl.h"

#include "realm/logging.h"
#include "realm/event_impl.h"
#include "realm/runtime_impl.h"

#if defined(__SSE__)
// technically pause is an "SSE2" instruction, but it's defined in xmmintrin
#include <xmmintrin.h>
static void mm_pause(void) { _mm_pause(); }
#else
static void mm_pause(void) { /* do nothing */ }
#endif

namespace Realm {

  Logger log_reservation("reservation");

  ////////////////////////////////////////////////////////////////////////
  //
  // class DeferredLockRequest
  //

    class DeferredLockRequest : public EventWaiter {
    public:
      DeferredLockRequest(Reservation _lock, unsigned _mode, bool _exclusive,
			  Event _after_lock)
	: lock(_lock), mode(_mode), exclusive(_exclusive), after_lock(_after_lock) {}

      virtual ~DeferredLockRequest(void) { }

      virtual bool event_triggered(Event e, bool poisoned)
      {
	// if input event is poisoned, do not attempt to take the lock - simply poison
	//  the output event too
	if(poisoned) {
	  log_poison.info() << "poisoned deferred lock skipped - lock=" << lock << " after=" << after_lock;
	  GenEventImpl::trigger(after_lock, true /*poisoned*/);
	} else {
	  get_runtime()->get_lock_impl(lock)->acquire(mode, exclusive,
						      ReservationImpl::ACQUIRE_BLOCKING,
						      after_lock);
	}
        return true;
      }

      virtual void print(std::ostream& os) const
      {
	os << "deferred lock: lock=" << lock << " after=" << after_lock;
      }

      virtual Event get_finish_event(void) const
      {
	return Event::NO_EVENT;
      }

    protected:
      Reservation lock;
      unsigned mode;
      bool exclusive;
      Event after_lock;
    };

  ////////////////////////////////////////////////////////////////////////
  //
  // class DeferredUnlockRequest
  //

    class DeferredUnlockRequest : public EventWaiter {
    public:
      DeferredUnlockRequest(Reservation _lock)
	: lock(_lock) {}

      virtual ~DeferredUnlockRequest(void) { }

      virtual bool event_triggered(Event e, bool poisoned)
      {
	// if input event is poisoned, do not attempt to release the lock
	// we don't have an output event here, so this may result in a hang if nobody is
	//  paying attention
	if(poisoned) {
	  log_poison.warning() << "poisoned deferred unlock skipped - POSSIBLE HANG - lock=" << lock;
	} else {
	  get_runtime()->get_lock_impl(lock)->release();
	}
        return true;
      }

      virtual void print(std::ostream& os) const
      {
	os << "deferred unlock: lock=" << lock;
      }

      virtual Event get_finish_event(void) const
      {
	return Event::NO_EVENT; // WRONG!  Should be ALL the waiting events
      }

    protected:
      Reservation lock;
    };


  ////////////////////////////////////////////////////////////////////////
  //
  // class DeferredLockDestruction
  //

    class DeferredLockDestruction : public EventWaiter {
    public:
      DeferredLockDestruction(Reservation _lock) : lock(_lock) {}

      virtual ~DeferredLockDestruction(void) { }

      virtual bool event_triggered(Event e, bool poisoned)
      {
	// if input event is poisoned, do not attempt to destroy the lock
	// we don't have an output event here, so this may result in a leak if nobody is
	//  paying attention
	if(poisoned) {
	  log_poison.info() << "poisoned deferred lock destruction skipped - POSSIBLE LEAK - lock=" << lock;
	} else {
	  get_runtime()->get_lock_impl(lock)->release_reservation();
	}
        return true;
      }

      virtual void print(std::ostream& os) const
      {
	os << "deferred lock destruction: lock=" << lock;
      }

      virtual Event get_finish_event(void) const
      {
	return Event::NO_EVENT;
      }

    protected:
      Reservation lock;
    };


  ////////////////////////////////////////////////////////////////////////
  //
  // class Reservation
  //

    /*static*/ const Reservation Reservation::NO_RESERVATION = { 0 };

    Event Reservation::acquire(unsigned mode /* = 0 */, bool exclusive /* = true */,
		     Event wait_on /* = Event::NO_EVENT */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      //printf("LOCK(" IDFMT ", %d, %d, " IDFMT ") -> ", id, mode, exclusive, wait_on.id);
      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(wait_on.has_triggered()) {
	Event e = get_runtime()->get_lock_impl(*this)->acquire(mode, exclusive,
							       ReservationImpl::ACQUIRE_BLOCKING);
	log_reservation.info() << "reservation acquire: rsrv=" << *this << " finish=" << e;
	//printf("(" IDFMT "/%d)\n", e.id, e.gen);
	return e;
      } else {
	Event after_lock = GenEventImpl::create_genevent()->current_event();
	log_reservation.info() << "reservation acquire: rsrv=" << *this << " finish=" << after_lock << " wait_on=" << wait_on;
	EventImpl::add_waiter(wait_on, new DeferredLockRequest(*this, mode, exclusive, after_lock));
	//printf("*(" IDFMT "/%d)\n", after_lock.id, after_lock.gen);
	return after_lock;
      }
    }

    Event Reservation::try_acquire(bool retry, unsigned mode /* = 0 */, bool exclusive /* = true */,
				   Event wait_on /* = Event::NO_EVENT */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

      ReservationImpl *impl = get_runtime()->get_lock_impl(*this);

      // if we have an unsatisfied precondition, we need to wait for that before actually trying
      //  to get the reservation, but those will be marked as retries, so get a placeholder request
      //  in here
      if(!wait_on.has_triggered()) {
	impl->acquire(mode, exclusive,
		      ReservationImpl::ACQUIRE_NONBLOCKING_PLACEHOLDER);
	log_reservation.info() << "reservation try_acquire: rsrv=" << *this << " wait_on=" << wait_on << " finish=" << wait_on;
	return wait_on;
      }

      // attempt the nonblocking acquire
      Event e = impl->acquire(mode, exclusive,
			      (retry ?
 			         ReservationImpl::ACQUIRE_NONBLOCKING_RETRY :
			         ReservationImpl::ACQUIRE_NONBLOCKING));
      log_reservation.info() << "reservation try_acquire: rsrv=" << *this << " wait_on=" << wait_on << " finish=" << e;
      return e;
    }

    // releases a held lock - release can be deferred until an event triggers
    void Reservation::release(Event wait_on /* = Event::NO_EVENT */) const
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(wait_on.has_triggered()) {
	log_reservation.info() << "reservation release: rsrv=" << *this;
	get_runtime()->get_lock_impl(*this)->release();
      } else {
	log_reservation.info() << "reservation release: rsrv=" << *this << " wait_on=" << wait_on;
	EventImpl::add_waiter(wait_on, new DeferredUnlockRequest(*this));
      }
    }

    // Create a new lock, destroy an existing lock
    /*static*/ Reservation Reservation::create_reservation(size_t _data_size /*= 0*/)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      //DetailedTimer::ScopedPush sp(18);

      // see if the freelist has an event we can reuse
      ReservationImpl *impl = get_runtime()->local_reservation_free_list->alloc_entry();
      assert(impl);
      assert(ID(impl->me).is_reservation());
      if(impl) {
	AutoHSLLock al(impl->mutex);

	assert(impl->owner == my_node_id);
	assert(impl->count == ReservationImpl::ZERO_COUNT);
	assert(impl->mode == ReservationImpl::MODE_EXCL);
	assert(impl->local_waiters.size() == 0);
        assert(impl->remote_waiter_mask.empty());
	assert(!impl->in_use);

	impl->in_use = true;

	log_reservation.info() << "reservation created: rsrv=" << impl->me;
	return impl->me;
      }
      assert(false);
      return Reservation::NO_RESERVATION;
#if 0
      // TODO: figure out if it's safe to iterate over a vector that is
      //  being resized?
      AutoHSLLock a(get_runtime()->nodes[my_node_id].mutex);

      std::vector<ReservationImpl>& locks = 
        get_runtime()->nodes[my_node_id].locks;

#ifdef REUSE_LOCKS
      // try to find an lock we can reuse
      for(std::vector<ReservationImpl>::iterator it = locks.begin();
	  it != locks.end();
	  it++) {
	// check the owner and in_use without taking the lock - conservative check
	if((*it).in_use || ((*it).owner != my_node_id)) continue;

	// now take the lock and make sure it really isn't in use
	AutoHSLLock a((*it).mutex);
	if(!(*it).in_use && ((*it).owner == my_node_id)) {
	  // now we really have the lock
	  (*it).in_use = true;
	  Reservation r = (*it).me;
	  return r;
	}
      }
#endif

      // couldn't reuse an lock - make a new one
      // TODO: take a lock here!?
      unsigned index = locks.size();
      assert((index+1) < MAX_LOCAL_LOCKS);
      locks.resize(index + 1);
      Reservation r = ID(ID::ID_LOCK, my_node_id, index).convert<Reservation>();
      locks[index].init(r, my_node_id);
      locks[index].in_use = true;
      get_runtime()->nodes[my_node_id].num_locks = index + 1;
      log_reservation.info() << "created new reservation: reservation=" << r;
      return r;
#endif
    }

    void Reservation::destroy_reservation()
    {
      log_reservation.info() << "reservation destroyed: rsrv=" << *this;

      // a lock has to be destroyed on the node that created it
      if(ID(*this).rsrv.creator_node != my_node_id) {
	DestroyLockMessage::send_request(ID(*this).rsrv.creator_node, *this);
	return;
      }

      // to destroy a local lock, we first must lock it (exclusively)
      ReservationImpl *lock_impl = get_runtime()->get_lock_impl(*this);
      Event e = lock_impl->acquire(0, true, ReservationImpl::ACQUIRE_BLOCKING);
      if(!e.has_triggered()) {
	EventImpl::add_waiter(e, new DeferredLockDestruction(*this));
      } else {
	// got grant immediately - can release reservation now
	lock_impl->release_reservation();
      }
    }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ReservationImpl
  //

    ReservationImpl::ReservationImpl(void)
    {
      init(Reservation::NO_RESERVATION, -1);
    }

    void ReservationImpl::init(Reservation _me, unsigned _init_owner,
			  size_t _data_size /*= 0*/)
    {
      me = _me;
      owner = _init_owner;
      count = ZERO_COUNT;
      log_reservation.spew("count init " IDFMT "=[%p]=%d", me.id, &count, count);
      mode = 0;
      in_use = false;
      remote_waiter_mask = NodeSet(); 
      remote_sharer_mask = NodeSet();
      requested = false;
      if(_data_size) {
	local_data = malloc(_data_size);
	local_data_size = _data_size;
        own_local = true;
      } else {
        local_data = 0;
	local_data_size = 0;
        own_local = false;
      }
    }

    /*static*/ void LockRequestMessage::send_request(NodeID target,
						     NodeID req_node,
						     Reservation lock,
						     unsigned mode)
    {
      RequestArgs args;

      args.node = req_node; // NOT my_node_id - may be forwarding a request
      args.lock = lock;
      args.mode = mode;
      Message::request(target, args);
    }

    /*static*/ void LockRequestMessage::handle_request(LockRequestMessage::RequestArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      ReservationImpl *impl = get_runtime()->get_lock_impl(args.lock);

      log_reservation.debug("reservation request: reservation=" IDFMT ", node=%d, mode=%d",
	       args.lock.id, args.node, args.mode);

      // can't send messages while holding mutex, so remember args and who
      //  (if anyone) to send to
      int req_forward_target = -1;
      int grant_target = -1;
      NodeSet copy_waiters;

      do {
	AutoHSLLock a(impl->mutex);

	// case 1: we don't even own the lock any more - pass the request on
	//  to whoever we think the owner is
	if(impl->owner != my_node_id) {
	  // can reuse the args we were given
	  log_reservation.debug(              "forwarding reservation request: reservation=" IDFMT ", from=%d, to=%d, mode=%d",
		   args.lock.id, args.node, impl->owner, args.mode);
	  req_forward_target = impl->owner;
	  break;
	}

	// it'd be bad if somebody tried to take a lock that had been 
	//   deleted...  (info is only valid on a lock's home node)
	assert((ID(impl->me).rsrv.creator_node != my_node_id) ||
	       impl->in_use);

	// case 2: we're the owner, and nobody is holding the lock, so grant
	//  it to the (original) requestor
	if((impl->count == ReservationImpl::ZERO_COUNT) && 
           (impl->remote_sharer_mask.empty())) {
          assert(impl->remote_waiter_mask.empty());

	  log_reservation.debug("granting reservation request: reservation=" IDFMT ", node=%d, mode=%d",
				args.lock.id, args.node, args.mode);
	  grant_target = args.node;
          copy_waiters = impl->remote_waiter_mask;

	  impl->owner = args.node;
	  break;
	}

	// case 3: we're the owner, but we can't grant the lock right now -
	//  just set a bit saying that the node is waiting and get back to
	//  work
	log_reservation.debug("deferring reservation request: reservation=" IDFMT ", node=%d, mode=%d (count=%d cmode=%d)",
			      args.lock.id, args.node, args.mode, impl->count, impl->mode);
        impl->remote_waiter_mask.add(args.node);
      } while(0);

      if(req_forward_target != -1)
      {
	LockRequestMessage::send_request(req_forward_target, args.node,
					 args.lock, args.mode);
#ifdef LOCK_TRACING
        {
          LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
          item.lock_id = impl->me.id;
          item.owner = req_forward_target;
          item.action = LockTraceItem::ACT_FORWARD_REQUEST;
        }
#endif
      }

      if(grant_target != -1)
      {
        // Make a buffer for storing our waiter mask and the the local data
	size_t waiter_count = copy_waiters.size();
        size_t payload_size = ((waiter_count+1) * sizeof(int)) + impl->local_data_size;
        int *payload = (int*)malloc(payload_size);
	int *pos = payload;
	*pos++ = waiter_count;
	// TODO: switch to iterator
        ReservationImpl::PackFunctor functor(pos);
        copy_waiters.map(functor);
        pos = functor.pos;
	//for(int i = 0; i < MAX_NUM_NODES; i++)
	//  if(copy_waiters.contains(i))
	//    *pos++ = i;
        memcpy(pos, impl->local_data, impl->local_data_size);
	LockGrantMessage::send_request(grant_target, args.lock,
				       0, // always grant exclusive for now
				       payload, payload_size, PAYLOAD_FREE);
#ifdef LOCK_TRACING
        {
          LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
          item.lock_id = impl->me.id;
          item.owner = grant_target;
          item.action = LockTraceItem::ACT_REMOTE_GRANT;
        }
#endif
      }
    }

    /*static*/ void LockReleaseMessage::send_request(NodeID target,
						     Reservation lock)
    {
      RequestArgs args;

      args.node = my_node_id;
      args.lock = lock;
      Message::request(target, args);
    }

    /*static*/ void LockReleaseMessage::handle_request(RequestArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      assert(0);
    }

    /*static*/ void LockGrantMessage::send_request(NodeID target,
						   Reservation lock, unsigned mode,
						   const void *data, size_t datalen,
						   int payload_mode)
    {
      RequestArgs args;

      args.lock = lock;
      args.mode = mode;
      Message::request(target, args, data, datalen, payload_mode);
    }

    /*static*/ void LockGrantMessage::handle_request(RequestArgs args,
						     const void *data, size_t datalen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      log_reservation.debug(          "reservation request granted: reservation=" IDFMT " mode=%d", // mask=%lx",
	       args.lock.id, args.mode); //, args.remote_waiter_mask);

      ReservationImpl::WaiterList to_wake;

      ReservationImpl *impl = get_runtime()->get_lock_impl(args.lock);
      {
	AutoHSLLock a(impl->mutex);

	// make sure we were really waiting for this lock
	assert(impl->owner != my_node_id);
	assert(impl->requested);

	// first, update our copy of the protected data (if any)
	const int *pos = (const int *)data;

	size_t waiter_count = *pos++;
	assert(datalen == (((waiter_count+1) * sizeof(int)) + impl->local_data_size));
	impl->remote_waiter_mask.clear();
	for(size_t i = 0; i < waiter_count; i++)
	  impl->remote_waiter_mask.add(*pos++);

	// is there local data to grab?
	if(impl->local_data_size > 0)
          memcpy(impl->local_data, pos, impl->local_data_size);

	if(args.mode == 0) // take ownership if given exclusive access
	  impl->owner = my_node_id;
	impl->mode = args.mode;
	impl->requested = false;

#ifndef NDEBUG
	bool any_local =
#endif
	  impl->select_local_waiters(to_wake);
	assert(any_local);
      }

      for(ReservationImpl::WaiterList::iterator it = to_wake.begin();
	  it != to_wake.end();
	  it++) {
	log_reservation.debug() << "release trigger: reservation=" << args.lock << " event=" << (*it);
	GenEventImpl::trigger(*it, false /*!poisoned*/);
      }
    }

    Event ReservationImpl::acquire(unsigned new_mode, bool exclusive,
				   AcquireType acquire_type,
				   Event after_lock /*= Event:NO_EVENT*/)
    {
      log_reservation.debug() << "local reservation request: reservation=" << me
			      << " mode=" << new_mode << " excl=" << exclusive
			      << " acq=" << acquire_type
			      << " event=" << after_lock
			      << " count=" << count << " impl=" << this;

      // collapse exclusivity into mode
      if(exclusive) new_mode = MODE_EXCL;

      bool got_lock = false;
      int lock_request_target = -1;
      WaiterList bonus_grants;

      {
	AutoHSLLock a(mutex); // hold mutex on lock while we check things

	// it'd be bad if somebody tried to take a lock that had been 
	//   deleted...  (info is only valid on a lock's home node)
	assert((ID(me).rsrv.creator_node != my_node_id) ||
	       in_use);

	// if this is just a placeholder nonblocking acquire, update the retry_count and
	//  return immediately
	if(acquire_type == ACQUIRE_NONBLOCKING_PLACEHOLDER) {
	  retry_count[new_mode]++;
	  return Event::NO_EVENT;
	}

	if(owner == my_node_id) {
#ifdef LOCK_TRACING
          {
            LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
            item.lock_id = me.id;
            item.owner = my_node_id;
            item.action = LockTraceItem::ACT_LOCAL_REQUEST;
          }
#endif
	  // case 1: we own the lock
	  // can we grant it?  (don't if there is a higher priority waiter)
	  if((count == ZERO_COUNT) ||
	     ((mode == new_mode) &&
	      (mode != MODE_EXCL) &&
	      (local_waiters.empty() || (local_waiters.begin()->first > mode)))) {
	    mode = new_mode;
	    count++;
	    log_reservation.spew("count ++(1) [%p]=%d", &count, count);
	    got_lock = true;
	    // fun special case here - if we grant a shared mode and there were local waiters and/or
	    //  a retry event for that mode, we can trigger them to see if they want to come along
	    //  for the ride
	    if(new_mode != MODE_EXCL) {
	      std::map<unsigned, WaiterList>::iterator it = local_waiters.find(new_mode);
	      if(it != local_waiters.end()) {
		bonus_grants.swap(it->second);
		local_waiters.erase(it);
	      }
	      std::map<unsigned, Event>::iterator it2 = retry_events.find(new_mode);
	      if(it2 != retry_events.end()) {
		bonus_grants.push_back(it2->second);
		retry_events.erase(it2);
	      }
	    }
	    //
#ifdef LOCK_TRACING
            {
              LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
              item.lock_id = me.id;
              item.owner = my_node_id;
              item.action = LockTraceItem::ACT_LOCAL_GRANT;
            }
#endif
	  }
	} else {
	  // somebody else owns it
	
	  // are we sharing?
	  if((count > ZERO_COUNT) && (mode == new_mode)) {
	    // we're allowed to grant additional sharers with the same mode
	    assert(mode != MODE_EXCL);
	    if(mode == new_mode) {
	      count++;
	      log_reservation.spew("count ++(2) [%p]=%d", &count, count);
	      got_lock = true;
	    }
	  }
	
	  // if we didn't get the lock, we'll have to ask for it from the
	  //  other node (even if we're currently sharing with the wrong mode)
	  if(!got_lock && !requested) {
	    log_reservation.debug(                "requesting reservation: reservation=" IDFMT " node=%d mode=%d",
		     me.id, owner, new_mode);
	    lock_request_target = owner;
	    // don't actually send message here because we're holding the
	    //  lock's mutex, which'll be bad if we get a message related to
	    //  this lock inside gasnet calls
	  
	    requested = true;
	  }
	}
  
	log_reservation.debug(            "local reservation result: reservation=" IDFMT " got=%d req=%d count=%d",
		 me.id, got_lock ? 1 : 0, requested ? 1 : 0, count);

	// if this was a successful retry of a nonblocking request, decrement the retry_count
	if(got_lock && (acquire_type == ACQUIRE_NONBLOCKING_RETRY)) {
	  std::map<unsigned, unsigned>::iterator it = retry_count.find(new_mode);
	  assert(it != retry_count.end());
	  if(it->second > 1) {
	    it->second--;
	  } else {
	    retry_count.erase(it);
	  }
	}

	// if we didn't get the lock, put our event on the queue of local
	//  waiters - create an event if we weren't given one to use
	if(!got_lock) {
	  switch(acquire_type) {
	  case ACQUIRE_BLOCKING:
	    {
	      if(!after_lock.exists())
		after_lock = GenEventImpl::create_genevent()->current_event();
	      local_waiters[new_mode].push_back(after_lock);
	      break;
	    }

	  case ACQUIRE_NONBLOCKING:
	    {
	      // first, record that we'll eventually see a retry of this
	      retry_count[new_mode]++;

	      // can't handle an existing after_event
	      assert(!after_lock.exists());

	      // now, make a retry event if we don't have one, or reuse an existing one
	      std::map<unsigned, Event>::iterator it = retry_events.find(new_mode);
	      if(it != retry_events.end()) {
		after_lock = it->second;
	      } else {
		after_lock = GenEventImpl::create_genevent()->current_event();
		retry_events[new_mode] = after_lock;
	      }
	      break;
	    }

	  case ACQUIRE_NONBLOCKING_RETRY:
	    {
	      // same as ACQUIRE_NONBLOCKING, but no increment of the retry count, since we
	      //  already did that on the first request

	      // can't handle an existing after_event
	      assert(!after_lock.exists());

	      // now, make a retry event if we don't have one, or reuse an existing one
	      std::map<unsigned, Event>::iterator it = retry_events.find(new_mode);
	      if(it != retry_events.end()) {
		after_lock = it->second;
	      } else {
		after_lock = GenEventImpl::create_genevent()->current_event();
		retry_events[new_mode] = after_lock;
	      }
	      break;
	    }

	  default:
	    assert(0);
	  }
	}
      }

      if(lock_request_target != -1)
      {
	LockRequestMessage::send_request(lock_request_target, my_node_id,
					 me, new_mode);
#ifdef LOCK_TRACING
        {
          LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
          item.lock_id = me.id;
          item.owner = lock_request_target;
          item.action = LockTraceItem::ACT_REMOTE_REQUEST;
        }
#endif
      }

      // if we got the lock, trigger an event if we were given one
      if(got_lock && after_lock.exists()) 
	GenEventImpl::trigger(after_lock, false /*!poisoned*/);

      // trigger any bonus grants too
      if(!bonus_grants.empty()) {
	for(WaiterList::iterator it = bonus_grants.begin();
	    it != bonus_grants.end();
	    it++) {
	  log_reservation.debug() << "acquire bonus grant: reservation=" << me << " event=" << (*it);
	  GenEventImpl::trigger(*it, false /*!poisoned*/);
	}
      }

      return after_lock;
    }

    // factored-out code to select one or more local waiters on a lock
    //  fills events to trigger into 'to_wake' and returns true if any were
    //  found - NOTE: ASSUMES LOCK IS ALREADY HELD!
    // also looks at retry_events and triggers one of those if it's higher
    //  priority than any blocking waiter
    bool ReservationImpl::select_local_waiters(WaiterList& to_wake)
    {
      if(local_waiters.empty() && retry_events.empty())
	return false;

      // further favor exclusive waiters
      if(local_waiters.find(MODE_EXCL) != local_waiters.end()) {
	WaiterList& excl_waiters = local_waiters[MODE_EXCL];
	to_wake.push_back(excl_waiters.front());
	excl_waiters.pop_front();
	  
	// if the set of exclusive waiters is empty, delete it
	if(excl_waiters.size() == 0)
	  local_waiters.erase(MODE_EXCL);
	  
	mode = MODE_EXCL;
	count = ZERO_COUNT + 1;
	log_reservation.spew("count <-1 [%p]=%d", &count, count);
      } else {
	// find the highest priority retry event and also the highest priority shared blocking waiters
	std::map<unsigned, WaiterList>::iterator it = local_waiters.begin();
	std::map<unsigned, Event>::iterator it2 = retry_events.begin();

	if((it != local_waiters.end()) &&
	   ((it2 == retry_events.end()) || (it->first <= it2->first))) {
	  mode = it->first;
	  count = ZERO_COUNT + it->second.size();
	  log_reservation.spew("count <-waiters [%p]=%d", &count, count);
	  assert(count > ZERO_COUNT);
	  // grab the list of events wanting to share the lock
	  to_wake.swap(it->second);
	  local_waiters.erase(it);  // actually pull list off map!
	  // TODO: can we share with any other nodes?
	} else {
	  // wake up one or more folks that will retry their try_acquires
	  to_wake.push_back(it2->second);
	  retry_events.erase(it2);
	}
      }
#ifdef LOCK_TRACING
      {
        LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
        item.lock_id = me.id;
        item.owner = my_node_id;
        item.action = LockTraceItem::ACT_LOCAL_GRANT;
      }
#endif

      return true;
    }

    void ReservationImpl::release(void)
    {
      // make a list of events that we be woken - can't do it while holding the
      //  lock's mutex (because the event we trigger might try to take the lock)
      WaiterList to_wake;

      int release_target = -1;
      int grant_target = -1;
      NodeSet copy_waiters;

      do {
#ifdef RSRV_DEBUG_MSGS
	log_reservation.debug(            "release: reservation=" IDFMT " count=%d mode=%d owner=%d", // share=%lx wait=%lx",
			me.id, count, mode, owner); //, remote_sharer_mask, remote_waiter_mask);
#endif
	AutoHSLLock a(mutex); // hold mutex on lock for entire function

	assert(count > ZERO_COUNT);

	// if this isn't the last holder of the lock, just decrement count
	//  and return
	count--;
#ifdef RSRV_DEBUG_MSGS
	log_reservation.spew("count -- [%p]=%d", &count, count);
	log_reservation.debug(            "post-release: reservation=" IDFMT " count=%d mode=%d", // share=%lx wait=%lx",
		 me.id, count, mode); //, remote_sharer_mask, remote_waiter_mask);
#endif
	if(count > ZERO_COUNT) break;

	// case 1: if we were sharing somebody else's lock, tell them we're
	//  done
	if(owner != my_node_id) {
	  assert(mode != MODE_EXCL);
	  mode = 0;

	  release_target = owner;
	  break;
	}

	// case 2: we own the lock, so we can give it to a local waiter (or a retry list)
	bool any_local = select_local_waiters(to_wake);
	if(any_local) {
	  // we'll wake the blocking waiter(s) below
	  assert(!to_wake.empty());
	  break;
	}

	// case 3: we can grant to a remote waiter (if any) if we don't expect any local retries
	if(!remote_waiter_mask.empty() && retry_count.empty()) {
	  // nobody local wants it, but another node does
	  //HACK int new_owner = remote_waiter_mask.find_first_set();
	  // TODO: use iterator - all we need is *begin()
          int new_owner = 0;  while(!remote_waiter_mask.contains(new_owner)) new_owner++;
          remote_waiter_mask.remove(new_owner);

#ifdef RSRV_DEBUG_MSGS
	  log_reservation.debug(              "reservation going to remote waiter: new=%d", // mask=%lx",
		   new_owner); //, remote_waiter_mask);
#endif

	  grant_target = new_owner;
          copy_waiters = remote_waiter_mask;

	  owner = new_owner;
          remote_waiter_mask = NodeSet();
	}

	// nobody wants it?  just sits in available state
	assert(local_waiters.empty());
	assert(retry_events.empty());
	assert(remote_waiter_mask.empty());
      } while(0);

      if(release_target != -1)
      {
	log_reservation.debug("releasing reservation " IDFMT " back to owner %d",
			      me.id, release_target);
	LockReleaseMessage::send_request(release_target, me);
#ifdef LOCK_TRACING
        {
          LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
          item.lock_id = me.id;
          item.owner = release_target;
          item.action = LockTraceItem::ACT_REMOTE_RELEASE;
        }
#endif
      }

      if(grant_target != -1)
      {
        // Make a buffer for storing our waiter mask and the the local data
	size_t waiter_count = copy_waiters.size();
        size_t payload_size = ((waiter_count+1) * sizeof(int)) + local_data_size;
        int *payload = (int*)malloc(payload_size);
	int *pos = payload;
	*pos++ = waiter_count;
	// TODO: switch to iterator
        PackFunctor functor(pos);
        copy_waiters.map(functor);
        pos = functor.pos;
	//for(int i = 0; i < MAX_NUM_NODES; i++)
	//  if(copy_waiters.contains(i))
	//    *pos++ = i;
        memcpy(pos, local_data, local_data_size);
	LockGrantMessage::send_request(grant_target, me,
				       0, // TODO: figure out shared cases
				       payload, payload_size, PAYLOAD_FREE);
#ifdef LOCK_TRACING
        {
          LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
          item.lock_id = me.id;
          item.owner = grant_target;
          item.action = LockTraceItem::ACT_REMOTE_GRANT;
        }
#endif
      }

      if(!to_wake.empty()) {
	for(WaiterList::iterator it = to_wake.begin();
	    it != to_wake.end();
	    it++) {
#ifdef RSRV_DEBUG_MSGS
	  log_reservation.debug() << "release trigger: reservation=" << me << " event=" << (*it);
#endif
	  GenEventImpl::trigger(*it, false /*!poisoned*/);
	}
      }
    }

    bool ReservationImpl::is_locked(unsigned check_mode, bool excl_ok)
    {
      // checking the owner can be done atomically, so doesn't need mutex
      if(owner != my_node_id) return false;

      // conservative check on lock count also doesn't need mutex
      if(count == ZERO_COUNT) return false;

      // a careful check of the lock mode and count does require the mutex
      bool held;
      {
	AutoHSLLock a(mutex);

	held = ((count > ZERO_COUNT) &&
		((mode == check_mode) || ((mode == 0) && excl_ok)));
      }

      return held;
    }

    void ReservationImpl::release_reservation(void)
    {
      // take the lock's mutex to sanity check it and clear the in_use field
      {
	AutoHSLLock al(mutex);

	// should only get here if the current node holds an exclusive lock
	assert(owner == my_node_id);
	assert(count == 1 + ZERO_COUNT);
	assert(mode == MODE_EXCL);
	assert(local_waiters.size() == 0);
        assert(remote_waiter_mask.empty());
	assert(in_use);
        // Mark that we no longer own our data
        if (own_local)
          free(local_data);
        local_data = NULL;
        local_data_size = 0;
        own_local = false;
      	in_use = false;
	count = ZERO_COUNT;
      }
      log_reservation.info() << "releasing reservation: reservation=" << me;

      get_runtime()->local_reservation_free_list->free_entry(this);
    }

    /*static*/ void DestroyLockMessage::send_request(NodeID target,
						     Reservation lock)
    {
      RequestArgs args;

      args.actual = lock;
      args.dummy = lock;
      Message::request(target, args);
    }

    /*static*/ void DestroyLockMessage::handle_request(RequestArgs args)
    {
      args.actual.destroy_reservation();
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class FastReservation

  // first define the actual internal state of a fast reservation - this
  //  is overlaid on the public state union, so must start with the state
  //  word
  struct FastRsrvState {
    FastReservation::State state;
    ReservationImpl *rsrv_impl;   // underlying reservation
    pthread_mutex_t mutex;        // protects rest of internal state
    Event rsrv_ready;             // ready event for a pending rsrv request
    unsigned sleeper_count;
    Event sleeper_event;
    pthread_cond_t condvar;    // for external waiters

    // must be called while holding mutex
    Event request_base_rsrv(void);
  };

  Event FastRsrvState::request_base_rsrv(void)
  {
    // make a request of the base reservation if we haven't already
    if(!rsrv_ready.exists())
      rsrv_ready = rsrv_impl->acquire(0, true /*excl*/,
				      ReservationImpl::ACQUIRE_BLOCKING);

    // now check if event has triggered (could be satisfaction of an earlier
    //  request that we're noticing now, or immediate grant in this call)
    if(rsrv_ready.has_triggered()) {
      rsrv_ready = Event::NO_EVENT;
      FastReservation::State prev = __sync_fetch_and_sub(&state, 
							 FastReservation::STATE_BASE_RSRV);
      assert((prev & FastReservation::STATE_BASE_RSRV) != 0);
    }

    return rsrv_ready;
  }

#ifdef REALM_DEBUG_FRSRV_HOLDERS
  GASNetHSL frsv_debug_mutex;
  std::map<Thread *, FastReservationDebugInfo *> frsv_debug_map;

  /*static*/ FastReservationDebugInfo *FastReservationDebugInfo::lookup_debuginfo(void)
  {
    FastReservationDebugInfo **infoptr;
    {
      AutoHSLLock al(frsv_debug_mutex);
      infoptr = &frsv_debug_map[Thread::self()];
    }
    if(!*infoptr) {
      FastReservationDebugInfo *info = new FastReservationDebugInfo;
      info->owner = Thread::self();
      *infoptr = info;
    }
    return *infoptr;
  } 

  namespace ThreadLocal {
    __thread FastReservationDebugInfo *frsv_debug = 0;
  };
#endif

  namespace Config {
    bool use_fast_reservation_fallback = false;
  };

  FastReservation::FastReservation(Reservation _rsrv /*= Reservation::NO_RESERVATION*/)
  {
    if(sizeof(FastRsrvState) > sizeof(opaque)) {
      log_reservation.fatal() << "FastReservation opaque state too small! (" << sizeof(opaque) << " < " << sizeof(FastRsrvState);
      assert(0);
    }
    FastRsrvState& frs = *(FastRsrvState *)opaque;

    if(_rsrv.exists()) {
      // if there's an underlying reservation, it initially owns the lock
      frs.state = STATE_BASE_RSRV;
      frs.rsrv_impl = get_runtime()->get_lock_impl(_rsrv);
    } else {
      // it's allowed to have no underlying reservation
      frs.state = 0;
      frs.rsrv_impl = 0;
    }
    pthread_mutex_init(&frs.mutex, 0);
    frs.rsrv_ready = Event::NO_EVENT;
    frs.sleeper_count = 0;
    frs.sleeper_event = Event::NO_EVENT;
    pthread_cond_init(&frs.condvar, 0);
    if(Config::use_fast_reservation_fallback) {
      frs.state |= STATE_SLOW_FALLBACK;
      if(!frs.rsrv_impl)
	frs.rsrv_impl = get_runtime()->get_lock_impl(Reservation::create_reservation());
    }
  }

  FastReservation::~FastReservation(void)
  {
    FastRsrvState& frs = *(FastRsrvState *)opaque;

    // if we have an underlying reservation and it does not hold the lock
    //  at the moment, give it back
    if(frs.rsrv_impl != 0) {
      if((frs.state & STATE_BASE_RSRV) == 0) {
	// if the SLOW_FALLBACK is set, we delete the reservation rather than
	//  just releasing it
	if((frs.state & STATE_SLOW_FALLBACK) != 0)
	  frs.rsrv_impl->me.destroy_reservation();
	else
	  frs.rsrv_impl->release();
      }
    }
    // clean up our pthread resources too
    pthread_mutex_destroy(&frs.mutex);
    pthread_cond_destroy(&frs.condvar);
  }

  // NOT copyable
  FastReservation::FastReservation(const FastReservation&)
  {
    assert(0);
  }

  FastReservation& FastReservation::operator=(const FastReservation&)
  {
    assert(0);
    return *this;
  }

  // the use of nonblocking acquires for the fallback path requires
  //  tracking the number of outstanding unsuccessful acquisition attempts
  //  so that we can have as many successful NONBLOCKING_RETRY attempts as
  //  we had unsuccessful NONBLOCKING attempts (without this, a reservation
  //  will have a permanent non-zero expected retry account and cannot be
  //  transferred to another node) - this counter does NOT need to be
  //  thread-local
  static volatile int fallback_retry_count = 0;

  Event FastReservation::wrlock_slow(WaitMode mode)
  {
    FastRsrvState& frs = *(FastRsrvState *)opaque;

    if((frs.state & STATE_SLOW_FALLBACK) != 0) {
      assert(frs.rsrv_impl != 0);
      ReservationImpl::AcquireType acqtype;
      int current_count;
      do {
	current_count = fallback_retry_count;
	if(current_count == 0) {
	  acqtype = ReservationImpl::ACQUIRE_NONBLOCKING;
	  break;
	} else
	  acqtype = ReservationImpl::ACQUIRE_NONBLOCKING_RETRY;
      } while(!__sync_bool_compare_and_swap(&fallback_retry_count,
					    current_count,
					    current_count - 1));
      Event e = frs.rsrv_impl->acquire(0, true /*excl*/, acqtype);
      if(e.exists()) {
	// attempt failed, so we'll retry later - increment count
	__sync_fetch_and_and(&fallback_retry_count, 1);
      }
      //log_reservation.print() << "wrlock " << (void *)this << " = " << e;
      return e;
    }

    // repeat until we succeed
    while(1) {
      // read the current state to see if any exceptional conditions exist
      State cur_state = __sync_fetch_and_or(&frs.state, 0);

      // if there are no exceptional conditions (sleepers, base_rsrv stuff),
      //  try to clear the WRITER_WAITING (if set), set WRITER, on the
      //  assumption that the READER_COUNT is 0 (i.e. we want the CAS to fail
      //  if there are still readers)
      if((cur_state & (STATE_SLOW_FALLBACK |
		       STATE_BASE_RSRV | STATE_BASE_RSRV_WAITING |
		       STATE_SLEEPER)) == 0) {
	State prev_state = (cur_state & STATE_WRITER_WAITING);
	State new_state = STATE_WRITER;
	if(__sync_bool_compare_and_swap(&frs.state, prev_state, new_state))
	  return Event::NO_EVENT;

	// if it failed and we've been asked to spin, assume this is regular
	//  contention and try again shortly
	if((mode == SPIN) || (mode == ALWAYS_SPIN)) {
	  // if we're going to spin as a writer, set a flag that prevents
	  //  new readers from taking the lock until we (or some other writer)
	  //  get our turn
          // unfortunately, this update is not atomic with the test above,
          //  so only set the flag if the state has not been changed to avoid
          //  setting the WAITING flag and then possibly going to sleep if
          //  some exceptional condition comes along - if the CAS fails
          //  because the read count changed, that's not the end of the world
          __sync_bool_compare_and_swap(&frs.state,
                                       cur_state,
                                       cur_state | STATE_WRITER_WAITING);

	  mm_pause();
	  continue;
	}

	// waiting is more complicated
	assert(0);
      }

      // any other transition requires holding the fast reservation's mutex
      {
	pthread_mutex_lock(&frs.mutex);

	// resample the state - since we hold the lock, exceptional bits
	//  cannot change out from under us
	cur_state = __sync_fetch_and_add(&frs.state, 0);

	// goal is to find (or possibly create) a condition we can wait
	//  on before trying again
	Event wait_for = Event::NO_EVENT;

	while(true) {
	  // case 1: the base reservation still owns the lock
	  if((cur_state & STATE_BASE_RSRV) != 0) {
	    wait_for = frs.request_base_rsrv();
	    break;
	  }

	  // case 2: a current lock holder is sleeping
	  if((cur_state & STATE_SLEEPER) != 0) {
	    wait_for = frs.sleeper_event;
	    break;
	  }

          // case 3: if we're back to normal readers/writers, don't sleep
          //   after all
          if((cur_state & ~(STATE_READER_COUNT_MASK | STATE_WRITER | STATE_WRITER_WAITING)) == 0) {
            wait_for = Event::NO_EVENT;
            break;
          }

	  // other cases?
	  log_reservation.fatal() << "wrlock_slow: unexpected state = "
				  << std::hex << cur_state << std::dec;
	  assert(0);
	}

	// now that we have our event, we're done messing with internal state
	pthread_mutex_unlock(&frs.mutex);

	if(wait_for.exists()) {
	  switch(mode) {
	  case ALWAYS_SPIN:
	    // what to do?
	    assert(0);

	  case SPIN:
	  case WAIT:
	    // return event to caller to wait
	    return wait_for;

	  case EXTERNAL_WAIT:
	    // wait on event, then try again (continue is outside switch)
	    wait_for.external_wait();
	    break;
	  }
	}
      }
      // now retry acquisition
      continue;
    }
    // not reachable
    assert(0);
    return Event::NO_EVENT;
  }

  Event FastReservation::rdlock_slow(WaitMode mode)
  {
    FastRsrvState& frs = *(FastRsrvState *)opaque;

    if((frs.state & STATE_SLOW_FALLBACK) != 0) {
      assert(frs.rsrv_impl != 0);
      ReservationImpl::AcquireType acqtype;
      int current_count;
      do {
	current_count = fallback_retry_count;
	if(current_count == 0) {
	  acqtype = ReservationImpl::ACQUIRE_NONBLOCKING;
	  break;
	} else
	  acqtype = ReservationImpl::ACQUIRE_NONBLOCKING_RETRY;
      } while(!__sync_bool_compare_and_swap(&fallback_retry_count,
					    current_count,
					    current_count - 1));
      Event e = frs.rsrv_impl->acquire(1, false /*!excl*/, acqtype);
      if(e.exists()) {
	// attempt failed, so we'll retry later - increment count
	__sync_fetch_and_and(&fallback_retry_count, 1);
      }
      //log_reservation.print() << "rdlock " << (void *)this << " = " << e;
      return e;
    }

    // repeat until we succeed
    while(1) {
      // check the current state for things that might involve waiting
      //  before trying to increment the count
      State cur_state = __sync_fetch_and_add(&frs.state, 0);

      // if there are no exceptional conditions (sleeping writer (sleeping
      //  reader is ok), base_rsrv stuff), increment the
      //  reader count and then make sure we didn't race with some other
      //  change to the state
      // if we observe a non-sleeping writer, or a waiting writer, we skip the
      //  count increment (to avoid cache-fighting with the writer) and
      //  follow the contention path
      bool sleeping_writer = ((cur_state & (STATE_WRITER | STATE_SLEEPER)) ==
			      (STATE_WRITER | STATE_SLEEPER));
      if(((cur_state & (STATE_SLOW_FALLBACK |
			STATE_BASE_RSRV | STATE_BASE_RSRV_WAITING)) == 0) &&
	 !sleeping_writer) {
	if((cur_state & (STATE_WRITER | STATE_WRITER_WAITING)) == 0) {
	  State prev_state = __sync_fetch_and_add(&frs.state, 1);
	  if((prev_state & ~(STATE_SLEEPER | STATE_READER_COUNT_MASK)) == 0) {
	    // no conflicts - we have the lock
	    return Event::NO_EVENT;
	  }
	  // decrement the count again if we failed
	  __sync_fetch_and_sub(&frs.state, 1);
	}

	// if it failed and we've been asked to spin, assume this is regular
	//  contention and try again shortly
	if((mode == SPIN) || (mode == ALWAYS_SPIN)) {
	  mm_pause();
	  continue;
	}

	// waiting is more complicated
	assert(0);
      }
	  
      // any other transition requires holding the fast reservation's mutex
      {
	pthread_mutex_lock(&frs.mutex);

	// resample the state - since we hold the lock, exceptional bits
	//  cannot change out from under us
	cur_state = __sync_fetch_and_add(&frs.state, 0);

	// goal is to find (or possibly create) a condition we can wait
	//  on before trying again
	Event wait_for = Event::NO_EVENT;

	while(true) {
	  // case 1: the base reservation still owns the lock
	  if((cur_state & STATE_BASE_RSRV) != 0) {
	    wait_for = frs.request_base_rsrv();
	    break;
	  }

	  // case 2: the base reservation has requested the lock back
	  if((cur_state & STATE_BASE_RSRV_WAITING) != 0) {
	    // two things to do here:
	    //  a) if the read and write counts are zero, we need to release
	    //      the current grant of the base reservation so whatever else
	    //      is waiting can get a turn - normally this would be done on
	    //      an unlock, but if a rdlock loses a race with a setting of
	    //      the BASE_RSRV_WAITING bit, it'll back out its read count
	    //      and then come here
	    if((cur_state & (STATE_WRITER | STATE_READER_COUNT_MASK)) == 0) {
	      // swap RSRV_WAITING for RSRV
	      __sync_fetch_and_sub(&frs.state, STATE_BASE_RSRV_WAITING - STATE_BASE_RSRV);
	      frs.rsrv_impl->release();
	    }

	    //  b) even if we didn't do the release yet, make the next request
	    //      of the reservation (if nobody else has already), and wait
	    //      on the grant before we attempt to lock again
	    wait_for = frs.request_base_rsrv();
	    break;
	  }

	  // case 3: a current lock holder is sleeping
	  if((cur_state & STATE_SLEEPER) != 0) {
	    wait_for = frs.sleeper_event;
	    break;
	  }

          // case 4: if we're back to normal readers/writers, don't sleep
          //   after all
          if((cur_state & ~(STATE_READER_COUNT_MASK | STATE_WRITER | STATE_WRITER_WAITING)) == 0) {
            wait_for = Event::NO_EVENT;
            break;
          }

	  // other cases?
	  log_reservation.fatal() << "rdlock_slow: unexpected state = "
				  << std::hex << cur_state << std::dec;
	  assert(0);
	}

	// now that we have our event, we're done messing with internal state
	pthread_mutex_unlock(&frs.mutex);

	if(wait_for.exists()) {
	  switch(mode) {
	  case ALWAYS_SPIN:
	    // what to do?
	    assert(0);

	  case SPIN:
	  case WAIT:
	    // return event to caller to wait
	    return wait_for;

	  case EXTERNAL_WAIT:
	    // wait on event, then try again (continue is outside switch)
	    wait_for.external_wait();
	    break;
	  }
	}
      }
      // now retry acquisition
      continue;
    }
    // not reachable
    assert(0);
    return Event::NO_EVENT;
  }

  void FastReservation::unlock_slow(void)
  {
    FastRsrvState& frs = *(FastRsrvState *)opaque;

    if((frs.state & STATE_SLOW_FALLBACK) != 0) {
      //log_reservation.print() << "unlock " << (void *)this;
      assert(frs.rsrv_impl != 0);
      frs.rsrv_impl->release();
      return;
    }

    // we already tried the fast path in unlock(), so just take the lock to
    //  hold exceptional conditions still and then modify state
    pthread_mutex_lock(&frs.mutex);

    // based on the current state, decide if we're undoing a write lock or
    //  a read lock
    State cur_state = __sync_fetch_and_add(&frs.state, 0);
    if((cur_state & STATE_WRITER) != 0) {
      // neither SLEEPER nor BASE_RSRV should be set here
      assert((cur_state & (STATE_SLEEPER | STATE_BASE_RSRV)) == 0);

      // if the base reservation is waiting, give it back
      if((cur_state & STATE_BASE_RSRV_WAITING) != 0) {
	// swap RSRV_WAITING for RSRV
	__sync_fetch_and_sub(&frs.state, STATE_BASE_RSRV_WAITING - STATE_BASE_RSRV);
	frs.rsrv_impl->release();
      }

      // now we can clear the WRITER bit and finish
      __sync_fetch_and_sub(&frs.state, STATE_WRITER);
    } else {
      // we'd better be a reader then
      unsigned reader_count = (cur_state & STATE_READER_COUNT_MASK);
      assert(reader_count > 0);
      // BASE_RSRV should not be set, and SLEEPER shouldn't if we're the only
      //  remaining reader
      assert((cur_state & STATE_BASE_RSRV) == 0);
      assert((reader_count > 1) || ((cur_state & STATE_SLEEPER) == 0));

      // if the base reservation is waiting and we're the last reader,
      //  give it back
      if((cur_state & STATE_BASE_RSRV_WAITING) != 0) {
	// swap RSRV_WAITING for RSRV
	__sync_fetch_and_sub(&frs.state, STATE_BASE_RSRV_WAITING - STATE_BASE_RSRV);
	frs.rsrv_impl->release();
      }

      // finally, decrement the read count
      __sync_fetch_and_sub(&frs.state, 1);
    }

    pthread_mutex_unlock(&frs.mutex);
  }

  void FastReservation::advise_sleep_entry(UserEvent guard_event)
  {
    FastRsrvState& frs = *(FastRsrvState *)opaque;

    // can only be called while the public lock is held

    // take the private lock, update the sleeper count/event and set the
    //  sleeper bit if we're the first
    pthread_mutex_lock(&frs.mutex);
    if(frs.sleeper_count == 0) {
      assert(!frs.sleeper_event.exists());
      frs.sleeper_event = guard_event;
      // set the sleeper flag - it must not already be set
      State old_state = __sync_fetch_and_add(&frs.state, STATE_SLEEPER);
      assert((old_state & STATE_SLEEPER) == 0);
      // if the WRITER_WAITING bit is set, clear it, since it'll sleep now
      if((old_state & STATE_WRITER_WAITING) != 0)
        __sync_fetch_and_and(&frs.state, ~STATE_WRITER_WAITING);
      frs.sleeper_count = 1;
    } else {
      assert(frs.sleeper_event.exists());
      assert((frs.state & STATE_SLEEPER) != 0);
      // double-check that WRITER_WAITING isn't set
      assert((frs.state & STATE_WRITER_WAITING) == 0);
      frs.sleeper_count++;
      if(guard_event != frs.sleeper_event)
	frs.sleeper_event = Event::merge_events(frs.sleeper_event,
						guard_event);
    }
    pthread_mutex_unlock(&frs.mutex);
  }

  void FastReservation::advise_sleep_exit(void)
  {
    FastRsrvState& frs = *(FastRsrvState *)opaque;

    // can only be called while the public lock is held

    // take the private lock, decrement the sleeper count, clearing the
    //  event and the sleeper bit if we were the last
    pthread_mutex_lock(&frs.mutex);
    assert(frs.sleeper_count > 0);
    if(frs.sleeper_count == 1) {
      // clear the sleeper flag - it must already be set
      State old_state = __sync_fetch_and_sub(&frs.state, STATE_SLEEPER);
      assert((old_state & STATE_SLEEPER) != 0);
      // double-check that WRITER_WAITING isn't set
      assert((frs.state & STATE_WRITER_WAITING) == 0);
      frs.sleeper_count = 0;
      assert(frs.sleeper_event.exists());
      frs.sleeper_event = Event::NO_EVENT;
    } else {
      assert(frs.sleeper_event.exists());
      assert((frs.state & STATE_SLEEPER) != 0);
      frs.sleeper_count--;
    }
    pthread_mutex_unlock(&frs.mutex);
  }


}; // namespace Realm
