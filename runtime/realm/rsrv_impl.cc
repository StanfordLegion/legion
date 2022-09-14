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

      virtual void event_triggered(bool poisoned, TimeLimit work_until)
      {
	// if input event is poisoned, do not attempt to take the lock - simply poison
	//  the output event too
	if(poisoned) {
	  log_poison.info() << "poisoned deferred lock skipped - lock=" << lock << " after=" << after_lock;
	  GenEventImpl::trigger(after_lock, true /*poisoned*/, work_until);
	} else {
	  get_runtime()->get_lock_impl(lock)->acquire(mode, exclusive,
						      ReservationImpl::ACQUIRE_BLOCKING,
						      after_lock);
	}
	// not attached to anything, so delete ourselves when we're done
	delete this;
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

      virtual void event_triggered(bool poisoned, TimeLimit work_until)
      {
	// if input event is poisoned, do not attempt to release the lock
	// we don't have an output event here, so this may result in a hang if nobody is
	//  paying attention
	if(poisoned) {
	  log_poison.warning() << "poisoned deferred unlock skipped - POSSIBLE HANG - lock=" << lock;
	} else {
	  get_runtime()->get_lock_impl(lock)->release(work_until);
	}
	// not attached to anything, so delete ourselves when we're done
	delete this;
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

      virtual void event_triggered(bool poisoned, TimeLimit work_until)
      {
	// if input event is poisoned, do not attempt to destroy the lock
	// we don't have an output event here, so this may result in a leak if nobody is
	//  paying attention
	if(poisoned) {
	  log_poison.info() << "poisoned deferred lock destruction skipped - POSSIBLE LEAK - lock=" << lock;
	} else {
	  get_runtime()->get_lock_impl(lock)->release_reservation();
	}
	// not attached to anything, so delete ourselves when we're done
	delete this;
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
      // early out - if the event has obviously triggered (or is NO_EVENT)
      //  don't build up continuation
      if(wait_on.has_triggered()) {
	log_reservation.info() << "reservation release: rsrv=" << *this;
	get_runtime()->get_lock_impl(*this)->release(TimeLimit::responsive());
      } else {
	log_reservation.info() << "reservation release: rsrv=" << *this << " wait_on=" << wait_on;
	EventImpl::add_waiter(wait_on, new DeferredUnlockRequest(*this));
      }
    }

    // Create a new lock, destroy an existing lock
    /*static*/ Reservation Reservation::create_reservation()
    {
      // see if the freelist has an event we can reuse
      ReservationImpl *impl = get_runtime()->local_reservation_free_list->alloc_entry();
      assert(impl);
      assert(ID(impl->me).is_reservation());
      if(impl) {
	AutoLock<> al(impl->mutex);

	assert(impl->owner == Network::my_node_id);
	assert(impl->count == ReservationImpl::ZERO_COUNT);
	assert(impl->mode == ReservationImpl::MODE_EXCL);
	assert(impl->local_excl_waiters.empty());
	assert(impl->local_shared.empty());
	assert(impl->retries.empty());
        assert(impl->remote_waiter_mask.empty());
	assert(!impl->in_use);

	impl->in_use = true;

	log_reservation.info() << "reservation created: rsrv=" << impl->me;
	return impl->me;
      }
      assert(false);
      return Reservation::NO_RESERVATION;
    }

    void Reservation::destroy_reservation(Event wait_on)
    {
      log_reservation.info() << "reservation destroyed: rsrv=" << *this;

      // a lock has to be destroyed on the node that created it
      if(NodeID(ID(*this).rsrv_creator_node()) != Network::my_node_id) {
	ActiveMessage<DestroyLockMessage> amsg(ID(*this).rsrv_creator_node());
	amsg->actual = *this;
	amsg->dummy = *this;
        amsg->wait_on = wait_on;
	amsg.commit();
	return;
      }

      // to destroy a local lock, we first must lock it (exclusively)
      Event e = acquire(0/*mode*/, true/*exclusive*/, wait_on);
      if(!e.has_triggered()) {
	EventImpl::add_waiter(e, new DeferredLockDestruction(*this));
      } else {
        ReservationImpl *lock_impl = get_runtime()->get_lock_impl(*this);
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
      init(Reservation::NO_RESERVATION, (unsigned)-1);
    }

    void ReservationImpl::init(Reservation _me, unsigned _init_owner)
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
    }

    /*static*/ void LockReleaseMessage::handle_message(NodeID sender, const LockReleaseMessage &msg,
						       const void *data, size_t datalen)
    {
      assert(0);
    }


    /*static*/ void LockGrantMessage::handle_message(NodeID sender, const LockGrantMessage &args,
						     const void *data, size_t datalen,
						     TimeLimit work_until)
    {
      log_reservation.debug(          "reservation request granted: reservation=" IDFMT " mode=%d", // mask=%lx",
	       args.lock.id, args.mode); //, args.remote_waiter_mask);

      ReservationImpl::WaiterList to_wake;
      Event retry_trigger = Event::NO_EVENT;

      ReservationImpl *impl = get_runtime()->get_lock_impl(args.lock);
      {
	AutoLock<> a(impl->mutex);

	// make sure we were really waiting for this lock
	assert(impl->owner != Network::my_node_id);
	assert(impl->requested);

	// first, update our copy of the protected data (if any)
	const int *pos = (const int *)data;

	size_t waiter_count = *pos++;
	assert(datalen == ((waiter_count+1) * sizeof(int)));
	impl->remote_waiter_mask.clear();
	for(size_t i = 0; i < waiter_count; i++)
	  impl->remote_waiter_mask.add(*pos++);

	if(args.mode == 0) // take ownership if given exclusive access
	  impl->owner = Network::my_node_id;
	impl->mode = args.mode;
	impl->requested = false;

#ifndef NDEBUG
	bool any_local =
#endif
	  impl->select_local_waiters(to_wake, retry_trigger);
	assert(any_local);
      }

      if(!to_wake.empty())
	get_runtime()->event_triggerer.trigger_event_waiters(to_wake,
							     false /*!poisoned*/,
							     work_until);

      if(retry_trigger.exists())
	GenEventImpl::trigger(retry_trigger, false /*!poisoned*/, work_until);
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

      {
	AutoLock<> a(mutex); // hold mutex on lock while we check things

	// it'd be bad if somebody tried to take a lock that had been 
	//   deleted...  (info is only valid on a lock's home node)
	assert((NodeID(ID(me).rsrv_creator_node()) != Network::my_node_id) ||
	       in_use);

	// if this is just a placeholder nonblocking acquire, update the retry_count and
	//  return immediately
	if(acquire_type == ACQUIRE_NONBLOCKING_PLACEHOLDER) {
	  RetryInfo& info = retries[new_mode];
	  info.count++;
	  return Event::NO_EVENT;
	}

	if(owner == Network::my_node_id) {
	  // case 1: we own the lock
	  // can we grant it?  (don't if there is a higher priority waiter)
	  if((count == ZERO_COUNT) ||
	     ((mode == new_mode) &&
	      (mode != MODE_EXCL) &&
	      local_excl_waiters.empty() &&
	      (local_shared.empty() ||
	       local_shared.begin()->first > mode))) {
	    mode = new_mode;
	    count++;
	    log_reservation.spew("count ++(1) [%p]=%d", &count, count);
	    got_lock = true;
#ifdef DEBUG_REALM
	    // if this is a shared mode, there should be no waiters or retry
	    //  events for that mode
	    if(new_mode != MODE_EXCL) {
	      assert(local_shared.count(new_mode) == 0);
	      std::map<unsigned, RetryInfo>::iterator it = retries.find(new_mode);
	      assert((it == retries.end()) || (!it->second.event.exists()));
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
	  std::map<unsigned, RetryInfo>::iterator it = retries.find(new_mode);
	  assert((it != retries.end()) && (it->second.count > 0) &&
		 !it->second.event.exists());
	  if(it->second.count > 1) {
	    it->second.count--;
	  } else {
	    retries.erase(it);
	  }
	}

	// if we didn't get the lock, put our event on the queue of local
	//  waiters - create an event if we weren't given one to use
	if(!got_lock) {
	  switch(acquire_type) {
	  case ACQUIRE_BLOCKING:
	    {
	      GenEventImpl *after_impl;
	      if(after_lock.exists()) {
		after_impl = get_runtime()->get_genevent_impl(after_lock);
	      } else {
		after_impl = GenEventImpl::create_genevent();
		after_lock = after_impl->current_event();
	      }
	      after_impl->merger.prepare_merger(after_lock,
						false /*!ignore_faults*/, 1);
	      EventMerger::MergeEventPrecondition *p = after_impl->merger.get_next_precondition();
	      after_impl->merger.arm_merger();

	      if(new_mode == MODE_EXCL) {
		local_excl_waiters.push_back(p);
	      } else {
		LocalSharedInfo& info = local_shared[new_mode];
		info.count++;
		info.waiters.push_back(p);
	      }
	      break;
	    }

	  case ACQUIRE_NONBLOCKING:
	    {
	      // can't handle an existing after_event
	      assert(!after_lock.exists());

	      RetryInfo& info = retries[new_mode];

	      // first, record that we'll eventually see a retry of this
	      info.count++;

	      // now, make a retry event if we don't have one, or reuse an existing one
	      if(!info.event.exists())
		info.event = GenEventImpl::create_genevent()->current_event();
	      after_lock = info.event;

	      break;
	    }

	  case ACQUIRE_NONBLOCKING_RETRY:
	    {
	      // same as ACQUIRE_NONBLOCKING, but no increment of the retry count, since we
	      //  already did that on the first request

	      // can't handle an existing after_event
	      assert(!after_lock.exists());

	      RetryInfo& info = retries[new_mode];

	      if(!info.event.exists())
		info.event = GenEventImpl::create_genevent()->current_event();
	      after_lock = info.event;

	      break;
	    }

	  default:
	    assert(0);
	  }
	}
      }

      if(lock_request_target != -1)
      {
	ActiveMessage<LockRequestMessage> amsg(lock_request_target);
	amsg->node = Network::my_node_id;
	amsg->lock = me;
	amsg->mode = new_mode;
	amsg.commit();
      }

      // if we got the lock, trigger an event if we were given one
      if(got_lock && after_lock.exists()) 
	GenEventImpl::trigger(after_lock, false /*!poisoned*/);

      return after_lock;
    }

    // factored-out code to select one or more local waiters on a lock
    //  fills events to trigger into 'to_wake' and returns true if any were
    //  found - NOTE: ASSUMES LOCK IS ALREADY HELD!
    // also looks at retry_events and triggers one of those if it's higher
    //  priority than any blocking waiter
    bool ReservationImpl::select_local_waiters(WaiterList& to_wake,
					       Event& retry)
    {
      // take a single exclusive waiter if there any present
      if(!local_excl_waiters.empty()) {
	EventWaiter *w = local_excl_waiters.pop_front();
	to_wake.push_back(w);
	mode = MODE_EXCL;
	count = ZERO_COUNT + 1;
	return true;
      }

      // take local shared waiters as long as they're not lower priority
      //  than the highest retry event
      if(!local_shared.empty()) {
	std::map<unsigned, LocalSharedInfo>::iterator it = local_shared.begin();
	if(retries.empty() || (it->first <= retries.begin()->first)) {
	  mode = it->first;
	  count = ZERO_COUNT + it->second.count;
	  to_wake.swap(it->second.waiters);
	  local_shared.erase(it);
	  return true;
	}
      }

      // last choice is the highest priority retry event
      if(!retries.empty()) {
	std::map<unsigned, RetryInfo>::iterator it = retries.begin();
	retry = it->second.event;
	it->second.event = Event::NO_EVENT;
	// don't actually remove the retries entry - we need to see those
	//  retries actually happen
	return true;
      }

      return false;
    }

    void ReservationImpl::release(TimeLimit work_until)
    {
      // make a list of events that we be woken - can't do it while holding the
      //  lock's mutex (because the event we trigger might try to take the lock)
      WaiterList to_wake;
      Event retry_trigger = Event::NO_EVENT;

      int release_target = -1;
      int grant_target = -1;
      NodeSet copy_waiters;

      do {
#ifdef RSRV_DEBUG_MSGS
	log_reservation.debug(            "release: reservation=" IDFMT " count=%d mode=%d owner=%d", // share=%lx wait=%lx",
			me.id, count, mode, owner); //, remote_sharer_mask, remote_waiter_mask);
#endif
	AutoLock<> a(mutex); // hold mutex on lock for entire function

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
	if(owner != Network::my_node_id) {
	  assert(mode != MODE_EXCL);
	  mode = 0;

	  release_target = owner;
	  break;
	}

	// case 2: we own the lock, so we can give it to a local waiter (or a retry list)
	bool any_local = select_local_waiters(to_wake, retry_trigger);
	if(any_local)
	  break;

	// case 3: we can grant to a remote waiter (if any) if we don't expect any local retries
	if(!remote_waiter_mask.empty() && retries.empty()) {
	  // nobody local wants it, but another node does
          NodeID new_owner = *remote_waiter_mask.begin();
	  remote_waiter_mask.remove(new_owner);

#ifdef RSRV_DEBUG_MSGS
	  log_reservation.debug(              "reservation going to remote waiter: new=%d", // mask=%lx",
		   new_owner); //, remote_waiter_mask);
#endif

	  grant_target = new_owner;
          copy_waiters.swap(remote_waiter_mask);

	  owner = new_owner;
	}

	// nobody wants it?  just sits in available state
      } while(0);

      if(release_target != -1)
      {
	log_reservation.debug("releasing reservation " IDFMT " back to owner %d",
			      me.id, release_target);
	ActiveMessage<LockReleaseMessage> amsg(release_target);
	amsg->lock = me;
	amsg.commit();
      }

      if(grant_target != -1)
      {
        // Make a buffer for storing our waiter mask and the the local data
	size_t waiter_count = copy_waiters.size();
        size_t payload_size = ((waiter_count+1) * sizeof(int));
        int *payload = (int*)malloc(payload_size);
	int *pos = payload;
	*pos++ = waiter_count;
	for(NodeSet::const_iterator it = copy_waiters.begin();
	    it != copy_waiters.end();
	    ++it)
	  *pos++ = *it;
	ActiveMessage<LockGrantMessage> amsg(grant_target, payload_size);
	amsg->lock = me;
	amsg->mode = 0; // TODO: figure out shared cases
	amsg.add_payload(payload, payload_size);
	amsg.commit();
      }

      if(!to_wake.empty())
	get_runtime()->event_triggerer.trigger_event_waiters(to_wake,
							     false /*!poisoned*/,
							     work_until);

      if(retry_trigger.exists())
	GenEventImpl::trigger(retry_trigger, false /*!poisoned*/, work_until);
    }

    bool ReservationImpl::is_locked(unsigned check_mode, bool excl_ok)
    {
      // checking the owner can be done atomically, so doesn't need mutex
      if(owner != Network::my_node_id) return false;

      // conservative check on lock count also doesn't need mutex
      if(count == ZERO_COUNT) return false;

      // a careful check of the lock mode and count does require the mutex
      bool held;
      {
	AutoLock<> a(mutex);

	held = ((count > ZERO_COUNT) &&
		((mode == check_mode) || ((mode == 0) && excl_ok)));
      }

      return held;
    }

    void ReservationImpl::release_reservation(void)
    {
      // take the lock's mutex to sanity check it and clear the in_use field
      {
	AutoLock<> al(mutex);

	// should only get here if the current node holds an exclusive lock
	assert(owner == Network::my_node_id);
	assert(count == 1 + ZERO_COUNT);
	assert(mode == MODE_EXCL);
	assert(local_excl_waiters.empty());
	assert(local_shared.empty());
	assert(retries.empty());
	assert(in_use);
      	in_use = false;
	count = ZERO_COUNT;
      }
      log_reservation.info() << "releasing reservation: reservation=" << me;

      get_runtime()->local_reservation_free_list->free_entry(this);
    }

    /*static*/ void DestroyLockMessage::handle_message(NodeID sender,const DestroyLockMessage &args,
						       const void *data, size_t datalen)
    {
      Reservation a = args.actual;
      a.destroy_reservation(args.wait_on);
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class FastReservation

  // first define the actual internal state of a fast reservation - this
  //  is overlaid on the public state union, so must start with the state
  //  word
  struct FastRsrvState {
    // NOTE: do not access state through this alias
    char _state[sizeof(atomic<FastReservation::State>)];

    ReservationImpl *rsrv_impl;   // underlying reservation
    Mutex mutex;              // protects rest of internal state
    Event rsrv_ready;             // ready event for a pending rsrv request
    unsigned sleeper_count;
    Event sleeper_event;
    Mutex::CondVar condvar;        // for external waiters

    // pointer math to obtain FastRsrvState reference
    static FastRsrvState& get_frs(FastReservation& frsv);

    // must be called while holding mutex
    Event request_base_rsrv(FastReservation& frsv);
  };

  /*static*/ inline FastRsrvState& FastRsrvState::get_frs(FastReservation& frsv)
  {
    uintptr_t base = reinterpret_cast<uintptr_t>(frsv.opaque) - sizeof(frsv.state);
    return *reinterpret_cast<FastRsrvState *>(base);
  }

  Event FastRsrvState::request_base_rsrv(FastReservation& frsv)
  {
    // make a request of the base reservation if we haven't already
    if(!rsrv_ready.exists())
      rsrv_ready = rsrv_impl->acquire(0, true /*excl*/,
				      ReservationImpl::ACQUIRE_BLOCKING);

    // now check if event has triggered (could be satisfaction of an earlier
    //  request that we're noticing now, or immediate grant in this call)
    if(rsrv_ready.has_triggered()) {
      rsrv_ready = Event::NO_EVENT;
      FastReservation::State prev = frsv.state.fetch_sub(FastReservation::STATE_BASE_RSRV);
      assert((prev & FastReservation::STATE_BASE_RSRV) != 0);
    }

    return rsrv_ready;
  }

#ifdef REALM_DEBUG_FRSRV_HOLDERS
  Mutex frsv_debug_mutex;
  std::map<Thread *, FastReservationDebugInfo *> frsv_debug_map;

  /*static*/ FastReservationDebugInfo *FastReservationDebugInfo::lookup_debuginfo(void)
  {
    FastReservationDebugInfo **infoptr;
    {
      AutoLock<> al(frsv_debug_mutex);
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
    REALM_THREAD_LOCAL FastReservationDebugInfo *frsv_debug = 0;
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
    FastRsrvState& frs = FastRsrvState::get_frs(*this);

    if(_rsrv.exists()) {
      // if there's an underlying reservation, it initially owns the lock
      state.store(STATE_BASE_RSRV);
      frs.rsrv_impl = get_runtime()->get_lock_impl(_rsrv);
    } else {
      // it's allowed to have no underlying reservation
      state.store(0);
      frs.rsrv_impl = 0;
    }
    // mutex, condvar must be manually constructed
    new(&frs.mutex) Mutex;
    new(&frs.condvar) Mutex::CondVar(frs.mutex);
    frs.rsrv_ready = Event::NO_EVENT;
    frs.sleeper_count = 0;
    frs.sleeper_event = Event::NO_EVENT;
    if(Config::use_fast_reservation_fallback) {
      state.fetch_or(STATE_SLOW_FALLBACK);
      if(!frs.rsrv_impl)
	frs.rsrv_impl = get_runtime()->get_lock_impl(Reservation::create_reservation());
    }
  }

  FastReservation::~FastReservation(void)
  {
    FastRsrvState& frs = FastRsrvState::get_frs(*this);

    // if we have an underlying reservation and it does not hold the lock
    //  at the moment, give it back
    if(frs.rsrv_impl != 0) {
      if((state.load() & STATE_BASE_RSRV) == 0) {
	// if the SLOW_FALLBACK is set, we delete the reservation rather than
	//  just releasing it
	if((state.load() & STATE_SLOW_FALLBACK) != 0)
	  frs.rsrv_impl->me.destroy_reservation();
	else
	  frs.rsrv_impl->release(TimeLimit::responsive());
      }
    }
    // mutex, condvar must be manually destroyed
    call_destructor(&frs.condvar);
    call_destructor(&frs.mutex);
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
  // TODO: should this be per reservation though?
  static atomic<int> fallback_retry_count(0);

  // WARNING: make sure any changes to this code have corresponding changes made
  //  to trywrlock_slow() below!
  Event FastReservation::wrlock_slow(WaitMode mode)
  {
    FastRsrvState& frs = FastRsrvState::get_frs(*this);

    if((state.load() & STATE_SLOW_FALLBACK) != 0) {
      assert(frs.rsrv_impl != 0);
      ReservationImpl::AcquireType acqtype;
      int current_count;
      do {
	current_count = fallback_retry_count.load();
	if(current_count == 0) {
	  acqtype = ReservationImpl::ACQUIRE_NONBLOCKING;
	  break;
	} else
	  acqtype = ReservationImpl::ACQUIRE_NONBLOCKING_RETRY;
      } while(!fallback_retry_count.compare_exchange(current_count,
						     current_count - 1));
      Event e = frs.rsrv_impl->acquire(0, true /*excl*/, acqtype);
      if(e.exists()) {
	// attempt failed, so we'll retry later - increment count
	fallback_retry_count.fetch_add(1);
      }
      //log_reservation.print() << "wrlock " << (void *)this << " = " << e;
      return e;
    }

    // repeat until we succeed
    while(1) {
      // read the current state to see if any exceptional conditions exist
      State cur_state = state.load_acquire();

      // if there are no exceptional conditions (sleepers, base_rsrv stuff),
      //  try to clear the WRITER_WAITING (if set), set WRITER, on the
      //  assumption that the READER_COUNT is 0 (i.e. we want the CAS to fail
      //  if there are still readers)
      if((cur_state & (STATE_SLOW_FALLBACK |
		       STATE_BASE_RSRV | STATE_BASE_RSRV_WAITING |
		       STATE_SLEEPER)) == 0) {
	State prev_state = (cur_state & STATE_WRITER_WAITING);
	State new_state = STATE_WRITER;
	if(state.compare_exchange(prev_state, new_state))
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
	  state.compare_exchange(cur_state,
				 cur_state | STATE_WRITER_WAITING);

	  mm_pause();
	  continue;
	}

	// waiting is more complicated
	assert(0);
      }

      // any other transition requires holding the fast reservation's mutex
      {
	frs.mutex.lock();

	// resample the state - since we hold the lock, exceptional bits
	//  cannot change out from under us
	cur_state = state.load_acquire();

	// goal is to find (or possibly create) a condition we can wait
	//  on before trying again
	Event wait_for = Event::NO_EVENT;

	while(true) {
	  // case 1: the base reservation still owns the lock
	  if((cur_state & STATE_BASE_RSRV) != 0) {
	    wait_for = frs.request_base_rsrv(*this);
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
	frs.mutex.unlock();

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
  }

  bool FastReservation::trywrlock_slow(void)
  {
    FastRsrvState& frs = FastRsrvState::get_frs(*this);

    if((state.load() & STATE_SLOW_FALLBACK) != 0) {
      assert(frs.rsrv_impl != 0);
      ReservationImpl::AcquireType acqtype;
      int current_count;
      do {
	current_count = fallback_retry_count.load();
	if(current_count == 0) {
	  acqtype = ReservationImpl::ACQUIRE_NONBLOCKING;
	  break;
	} else
	  acqtype = ReservationImpl::ACQUIRE_NONBLOCKING_RETRY;
      } while(!fallback_retry_count.compare_exchange(current_count,
						     current_count - 1));
      Event e = frs.rsrv_impl->acquire(0, true /*excl*/, acqtype);
      if(e.exists()) {
	// attempt failed, so we'll retry later - increment count
	fallback_retry_count.fetch_add(1);
      }
      //log_reservation.print() << "wrlock " << (void *)this << " = " << e;
      return false;  // forgetting the event here causes inefficiencies!
    }

    // repeat until we succeed
    while(1) {
      // attempt to change 0 -> STATE_WRITER
      State cur_state = 0;
      if(state.compare_exchange(cur_state, STATE_WRITER))  // updates cur_state
	return true;

      // simple contention just causes us to return
      if((cur_state & (STATE_READER_COUNT_MASK |
		       STATE_WRITER |
		       STATE_WRITER_WAITING)) != 0)
	return false;

      // any other transition requires holding the fast reservation's mutex
      {
	frs.mutex.lock();

	// resample the state - since we hold the lock, exceptional bits
	//  cannot change out from under us
	cur_state = state.load_acquire();

	bool event_needed = false;

	while(true) {
	  // case 1: the base reservation still owns the lock
	  if((cur_state & STATE_BASE_RSRV) != 0) {
	    Event e = frs.request_base_rsrv(*this);
	    if(e.exists())
	      event_needed = true;  // forgetting the event here causes inefficiencies!
	    break;
	  }

	  // case 2: a current lock holder is sleeping
	  if((cur_state & STATE_SLEEPER) != 0) {
	    event_needed = true;  // forgetting the event here causes inefficiencies!
	    break;
	  }

          // case 3: if we're back to normal readers/writers, don't sleep
          //   after all
          if((cur_state & ~(STATE_READER_COUNT_MASK | STATE_WRITER | STATE_WRITER_WAITING)) == 0) {
            break;
          }

	  // other cases?
	  log_reservation.fatal() << "wrlock_slow: unexpected state = "
				  << std::hex << cur_state << std::dec;
	  assert(0);
	}

	// now we're done messing with internal state
	frs.mutex.unlock();

	// if an event trigger is required, fail the lock attempt
	if(event_needed)
	  return false;
      }

      // now retry acquisition
      continue;
    }
  }

  // WARNING: make sure any changes to this code have corresponding changes made
  //  to trywrlock_slow() below!
  Event FastReservation::rdlock_slow(WaitMode mode)
  {
    FastRsrvState& frs = FastRsrvState::get_frs(*this);

    if((state.load() & STATE_SLOW_FALLBACK) != 0) {
      assert(frs.rsrv_impl != 0);
      ReservationImpl::AcquireType acqtype;
      int current_count;
      do {
	current_count = fallback_retry_count.load();
	if(current_count == 0) {
	  acqtype = ReservationImpl::ACQUIRE_NONBLOCKING;
	  break;
	} else
	  acqtype = ReservationImpl::ACQUIRE_NONBLOCKING_RETRY;
      } while(!fallback_retry_count.compare_exchange(current_count,
						     current_count - 1));
      Event e = frs.rsrv_impl->acquire(1, false /*!excl*/, acqtype);
      if(e.exists()) {
	// attempt failed, so we'll retry later - increment count
	fallback_retry_count.fetch_add(1);
      }
      //log_reservation.print() << "rdlock " << (void *)this << " = " << e;
      return e;
    }

    // repeat until we succeed
    while(1) {
      // check the current state for things that might involve waiting
      //  before trying to increment the count
      State cur_state = state.load_acquire();

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
	  State prev_state = state.fetch_add_acqrel(1);
	  if((prev_state & ~(STATE_SLEEPER | STATE_READER_COUNT_MASK)) == 0) {
	    // no conflicts - we have the lock
	    return Event::NO_EVENT;
	  }
	  // decrement the count again if we failed
	  state.fetch_sub(1);
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
	frs.mutex.lock();

	// resample the state - since we hold the lock, exceptional bits
	//  cannot change out from under us
	cur_state = state.load_acquire();

	// goal is to find (or possibly create) a condition we can wait
	//  on before trying again
	Event wait_for = Event::NO_EVENT;

	while(true) {
	  // case 1: the base reservation still owns the lock
	  if((cur_state & STATE_BASE_RSRV) != 0) {
	    wait_for = frs.request_base_rsrv(*this);
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
	      state.fetch_sub(STATE_BASE_RSRV_WAITING - STATE_BASE_RSRV);
	      frs.rsrv_impl->release(TimeLimit::responsive());
	    }

	    //  b) even if we didn't do the release yet, make the next request
	    //      of the reservation (if nobody else has already), and wait
	    //      on the grant before we attempt to lock again
	    wait_for = frs.request_base_rsrv(*this);
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
	frs.mutex.unlock();

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
  }

  bool FastReservation::tryrdlock_slow(void)
  {
    FastRsrvState& frs = FastRsrvState::get_frs(*this);

    if((state.load() & STATE_SLOW_FALLBACK) != 0) {
      assert(frs.rsrv_impl != 0);
      ReservationImpl::AcquireType acqtype;
      int current_count;
      do {
	current_count = fallback_retry_count.load();
	if(current_count == 0) {
	  acqtype = ReservationImpl::ACQUIRE_NONBLOCKING;
	  break;
	} else
	  acqtype = ReservationImpl::ACQUIRE_NONBLOCKING_RETRY;
      } while(!fallback_retry_count.compare_exchange(current_count,
						     current_count - 1));
      Event e = frs.rsrv_impl->acquire(1, false /*!excl*/, acqtype);
      if(e.exists()) {
	// attempt failed, so we'll retry later - increment count
	fallback_retry_count.fetch_add(1);
      }
      //log_reservation.print() << "rdlock " << (void *)this << " = " << e;
      return false;  // forgetting the event here causes inefficiencies!
    }

    // repeat until we succeed
    while(1) {
      // check the current state for things that might involve waiting
      //  before trying to increment the count
      State cur_state = state.load_acquire();

      // if the only thing present is (potentially sleeping) readers, attempt to
      //  increment the count (this prevents cache-fighting with writers)
      if((cur_state & ~(STATE_SLEEPER | STATE_READER_COUNT_MASK)) == 0) {
	State prev_state = state.fetch_add_acqrel(1);
	if((prev_state & ~(STATE_SLEEPER | STATE_READER_COUNT_MASK)) == 0) {
	  // no conflicts - we have the lock
	  return true;
	}
	// decrement the count again if we failed
	cur_state = state.fetch_sub(1);
	// TODO: handle the case where BASE_RSRV_WAITING got set while
	//  we had the (erroneous) reader count increase
	assert((cur_state & STATE_BASE_RSRV_WAITING) == 0);
	return false;
      }

      // if the BASE_RSRV bit appears to be set, we probably need to request it,
      //  but take the lock first to be sure
      if((cur_state & STATE_BASE_RSRV) != 0) {
	frs.mutex.lock();

	// resample state
	cur_state = state.load_acquire();

	bool retry = false;
	if((cur_state & STATE_BASE_RSRV) != 0) {
	  Event e = frs.request_base_rsrv(*this);
	  if(!e.exists()) {
	    // we got the base reservation, so we can try the acquire again
	    retry = true;
	  }
	}

	frs.mutex.unlock();

	if(retry)
	  continue;
      }

      // any other condition will have to sort itself out - return to caller
      //  with a failure
      return false;
    }
  }

  void FastReservation::unlock_slow(void)
  {
    FastRsrvState& frs = FastRsrvState::get_frs(*this);

    if((state.load() & STATE_SLOW_FALLBACK) != 0) {
      //log_reservation.print() << "unlock " << (void *)this;
      assert(frs.rsrv_impl != 0);
      frs.rsrv_impl->release(TimeLimit::responsive());
      return;
    }

    // we already tried the fast path in unlock(), so just take the lock to
    //  hold exceptional conditions still and then modify state
    frs.mutex.lock();

    // based on the current state, decide if we're undoing a write lock or
    //  a read lock
    State cur_state = state.load_acquire();
    if((cur_state & STATE_WRITER) != 0) {
      // neither SLEEPER nor BASE_RSRV should be set here
      assert((cur_state & (STATE_SLEEPER | STATE_BASE_RSRV)) == 0);

      // if the base reservation is waiting, give it back
      if((cur_state & STATE_BASE_RSRV_WAITING) != 0) {
	// swap RSRV_WAITING for RSRV
	state.fetch_sub(STATE_BASE_RSRV_WAITING - STATE_BASE_RSRV);
	frs.rsrv_impl->release(TimeLimit::responsive());
      }

      // now we can clear the WRITER bit and finish
      state.fetch_sub_acqrel(STATE_WRITER);
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
	state.fetch_sub(STATE_BASE_RSRV_WAITING - STATE_BASE_RSRV);
	frs.rsrv_impl->release(TimeLimit::responsive());
      }

      // finally, decrement the read count
      state.fetch_sub_acqrel(1);
    }

    frs.mutex.unlock();
  }

  void FastReservation::advise_sleep_entry(UserEvent guard_event)
  {
    FastRsrvState& frs = FastRsrvState::get_frs(*this);

    // can only be called while the public lock is held

    // take the private lock, update the sleeper count/event and set the
    //  sleeper bit if we're the first
    frs.mutex.lock();
    if(frs.sleeper_count == 0) {
      assert(!frs.sleeper_event.exists());
      frs.sleeper_event = guard_event;
      // set the sleeper flag - it must not already be set
      State old_state = state.fetch_add(STATE_SLEEPER);
      assert((old_state & STATE_SLEEPER) == 0);
      // if the WRITER_WAITING bit is set, clear it, since it'll sleep now
      if((old_state & STATE_WRITER_WAITING) != 0)
	state.fetch_and(~STATE_WRITER_WAITING);
      frs.sleeper_count = 1;
    } else {
      assert(frs.sleeper_event.exists());
      assert((state.load() & STATE_SLEEPER) != 0);
      // double-check that WRITER_WAITING isn't set
      assert((state.load() & STATE_WRITER_WAITING) == 0);
      frs.sleeper_count++;
      if(guard_event != frs.sleeper_event)
	frs.sleeper_event = Event::merge_events(frs.sleeper_event,
						guard_event);
    }
    frs.mutex.unlock();
  }

  void FastReservation::advise_sleep_exit(void)
  {
    FastRsrvState& frs = FastRsrvState::get_frs(*this);

    // can only be called while the public lock is held

    // take the private lock, decrement the sleeper count, clearing the
    //  event and the sleeper bit if we were the last
    frs.mutex.lock();
    assert(frs.sleeper_count > 0);
    if(frs.sleeper_count == 1) {
      // clear the sleeper flag - it must already be set
      State old_state = state.fetch_sub(STATE_SLEEPER);
      assert((old_state & STATE_SLEEPER) != 0);
      // double-check that WRITER_WAITING isn't set
      assert((old_state & STATE_WRITER_WAITING) == 0);
      frs.sleeper_count = 0;
      assert(frs.sleeper_event.exists());
      frs.sleeper_event = Event::NO_EVENT;
    } else {
      assert(frs.sleeper_event.exists());
      assert((state.load() & STATE_SLEEPER) != 0);
      frs.sleeper_count--;
    }
    frs.mutex.unlock();
  }

  /* static */
  void LockRequestMessage::handle_message(NodeID sender, const LockRequestMessage &args,
					     const void *data, size_t datalen)
  {
    ReservationImpl *impl = get_runtime()->get_lock_impl(args.lock);

    log_reservation.debug("reservation request: reservation=" IDFMT ", node=%d, mode=%d",
			  args.lock.id, args.node, args.mode);

      // can't send messages while holding mutex, so remember args and who
      //  (if anyone) to send to
      int req_forward_target = -1;
      int grant_target = -1;
      NodeSet copy_waiters;

      do {
	AutoLock<> a(impl->mutex);

	// case 1: we don't even own the lock any more - pass the request on
	//  to whoever we think the owner is
	if(impl->owner != Network::my_node_id) {
	  // can reuse the args we were given
	  log_reservation.debug(              "forwarding reservation request: reservation=" IDFMT ", from=%d, to=%d, mode=%d",
		   args.lock.id, args.node, impl->owner, args.mode);
	  req_forward_target = impl->owner;
	  break;
	}

	// it'd be bad if somebody tried to take a lock that had been
	//   deleted...  (info is only valid on a lock's home node)
	assert((NodeID(ID(impl->me).rsrv_creator_node()) != Network::my_node_id) ||
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
	ActiveMessage<LockRequestMessage> amsg(req_forward_target);
	amsg->node = args.node;
	amsg->lock = args.lock;
	amsg->mode = args.mode;
	amsg.commit();
      }

      if(grant_target != -1)
      {
        // Make a buffer for storing our waiter mask and the the local data
	size_t waiter_count = copy_waiters.size();
        size_t payload_size = ((waiter_count+1) * sizeof(int));
        int *payload = (int*)malloc(payload_size);
	int *pos = payload;
	*pos++ = waiter_count;
	for(NodeSet::const_iterator it = copy_waiters.begin();
	    it != copy_waiters.end();
	    ++it)
	  *pos++ = *it;
	ActiveMessage<LockGrantMessage> amsg(grant_target, payload_size);
	amsg->lock = args.lock;
	amsg->mode = 0; // always grant exclusive for now
	amsg.add_payload(payload, payload_size);
	amsg.commit();
      }
  }

  ActiveMessageHandlerReg<LockRequestMessage> lock_request_message_handler;
  ActiveMessageHandlerReg<LockReleaseMessage> lock_release_message_handler;
  ActiveMessageHandlerReg<LockGrantMessage> lock_grant_message_handler;
  ActiveMessageHandlerReg<DestroyLockMessage> destroy_lock_message_handler;

}; // namespace Realm
