/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include "rsrv_impl.h"

#include "logging.h"
#include "event_impl.h"
#include "runtime_impl.h"

namespace Realm {

  Logger log_reservation("reservation");

  ////////////////////////////////////////////////////////////////////////
  //
  // class DeferredLockRequest
  //

    class DeferredLockRequest : public EventWaiter {
    public:
      DeferredLockRequest(Reservation _lock, unsigned _mode, bool _exclusive,
			  GenEventImpl *_after_lock)
	: lock(_lock), mode(_mode), exclusive(_exclusive), after_lock(_after_lock) {}

      virtual ~DeferredLockRequest(void) { }

      virtual bool event_triggered(Event e)
      {
	get_runtime()->get_lock_impl(lock)->acquire(mode, exclusive, after_lock);
        return true;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"deferred lock: lock=" IDFMT " after=" IDFMT "/%d\n",
		lock.id, after_lock->me.id(), after_lock->generation + 1);
      }

    protected:
      Reservation lock;
      unsigned mode;
      bool exclusive;
      GenEventImpl *after_lock;
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

      virtual bool event_triggered(Event e)
      {
	get_runtime()->get_lock_impl(lock)->release();
        return true;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"deferred unlock: lock=" IDFMT "\n",
	       lock.id);
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

      virtual bool event_triggered(Event e)
      {
	get_runtime()->get_lock_impl(lock)->release_reservation();
        return true;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"deferred lock destruction: lock=" IDFMT "\n", lock.id);
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
	Event e = get_runtime()->get_lock_impl(*this)->acquire(mode, exclusive);
	log_reservation.info() << "reservation acquire: rsrv=" << *this << " finish=" << e;
	//printf("(" IDFMT "/%d)\n", e.id, e.gen);
	return e;
      } else {
	GenEventImpl *after_lock = GenEventImpl::create_genevent();
	Event e = after_lock->current_event();
	log_reservation.info() << "reservation acquire: rsrv=" << *this << " finish=" << e << " wait_on=" << wait_on;
	EventImpl::add_waiter(wait_on, new DeferredLockRequest(*this, mode, exclusive, after_lock));
	//printf("*(" IDFMT "/%d)\n", after_lock.id, after_lock.gen);
	return e;
      }
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
      assert(ID(impl->me).type() == ID::ID_LOCK);
      if(impl) {
	AutoHSLLock al(impl->mutex);

	assert(impl->owner == gasnet_mynode());
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
      AutoHSLLock a(get_runtime()->nodes[gasnet_mynode()].mutex);

      std::vector<ReservationImpl>& locks = 
        get_runtime()->nodes[gasnet_mynode()].locks;

#ifdef REUSE_LOCKS
      // try to find an lock we can reuse
      for(std::vector<ReservationImpl>::iterator it = locks.begin();
	  it != locks.end();
	  it++) {
	// check the owner and in_use without taking the lock - conservative check
	if((*it).in_use || ((*it).owner != gasnet_mynode())) continue;

	// now take the lock and make sure it really isn't in use
	AutoHSLLock a((*it).mutex);
	if(!(*it).in_use && ((*it).owner == gasnet_mynode())) {
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
      Reservation r = ID(ID::ID_LOCK, gasnet_mynode(), index).convert<Reservation>();
      locks[index].init(r, gasnet_mynode());
      locks[index].in_use = true;
      get_runtime()->nodes[gasnet_mynode()].num_locks = index + 1;
      log_reservation.info() << "created new reservation: reservation=" << r;
      return r;
#endif
    }

    void Reservation::destroy_reservation()
    {
      log_reservation.info() << "reservation destroyed: rsrv=" << *this;

      // a lock has to be destroyed on the node that created it
      if(ID(*this).node() != gasnet_mynode()) {
	DestroyLockMessage::send_request(ID(*this).node(), *this);
	return;
      }

      // to destroy a local lock, we first must lock it (exclusively)
      ReservationImpl *lock_impl = get_runtime()->get_lock_impl(*this);
      Event e = lock_impl->acquire(0, true);
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

    /*static*/ void LockRequestMessage::send_request(gasnet_node_t target,
						     gasnet_node_t req_node,
						     Reservation lock,
						     unsigned mode)
    {
      RequestArgs args;

      args.node = req_node; // NOT gasnet_mynode() - may be forwarding a request
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
	if(impl->owner != gasnet_mynode()) {
	  // can reuse the args we were given
	  log_reservation.debug(              "forwarding reservation request: reservation=" IDFMT ", from=%d, to=%d, mode=%d",
		   args.lock.id, args.node, impl->owner, args.mode);
	  req_forward_target = impl->owner;
	  break;
	}

	// it'd be bad if somebody tried to take a lock that had been 
	//   deleted...  (info is only valid on a lock's home node)
	assert((ID(impl->me).node() != gasnet_mynode()) ||
	       impl->in_use);

	// case 2: we're the owner, and nobody is holding the lock, so grant
	//  it to the (original) requestor
	if((impl->count == ReservationImpl::ZERO_COUNT) && 
           (impl->remote_sharer_mask.empty())) {
          assert(impl->remote_waiter_mask.empty());

	  log_reservation.debug(              "granting reservation request: reservation=" IDFMT ", node=%d, mode=%d",
		   args.lock.id, args.node, args.mode);
	  grant_target = args.node;
          copy_waiters = impl->remote_waiter_mask;

	  impl->owner = args.node;
	  break;
	}

	// case 3: we're the owner, but we can't grant the lock right now -
	//  just set a bit saying that the node is waiting and get back to
	//  work
	log_reservation.debug(            "deferring reservation request: reservation=" IDFMT ", node=%d, mode=%d (count=%d cmode=%d)",
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

    /*static*/ void LockReleaseMessage::send_request(gasnet_node_t target,
						     Reservation lock)
    {
      RequestArgs args;

      args.node = gasnet_mynode();
      args.lock = lock;
      Message::request(target, args);
    }

    /*static*/ void LockReleaseMessage::handle_request(RequestArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      assert(0);
    }

    /*static*/ void LockGrantMessage::send_request(gasnet_node_t target,
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

      std::deque<GenEventImpl *> to_wake;

      ReservationImpl *impl = get_runtime()->get_lock_impl(args.lock);
      {
	AutoHSLLock a(impl->mutex);

	// make sure we were really waiting for this lock
	assert(impl->owner != gasnet_mynode());
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
	  impl->owner = gasnet_mynode();
	impl->mode = args.mode;
	impl->requested = false;

#ifndef NDEBUG
	bool any_local =
#endif
	  impl->select_local_waiters(to_wake);
	assert(any_local);
      }

      for(std::deque<GenEventImpl *>::iterator it = to_wake.begin();
	  it != to_wake.end();
	  it++) {
	log_reservation.debug("release trigger: reservation=" IDFMT " event=" IDFMT "/%d",
			args.lock.id, (*it)->me.id(), (*it)->generation+1);
	(*it)->trigger_current();
      }
    }

    Event ReservationImpl::acquire(unsigned new_mode, bool exclusive,
				     GenEventImpl *after_lock /* = 0*/)
    {
      Event after_lock_event = after_lock ? after_lock->current_event() : Event::NO_EVENT;

      log_reservation.debug(		      "local reservation request: reservation=" IDFMT " mode=%d excl=%d event=" IDFMT "/%d count=%d impl=%p",
		      me.id, new_mode, exclusive, 
		      after_lock_event.id,
		      after_lock_event.gen, count, this);

      // collapse exclusivity into mode
      if(exclusive) new_mode = MODE_EXCL;

      bool got_lock = false;
      int lock_request_target = -1;

      {
	AutoHSLLock a(mutex); // hold mutex on lock while we check things

	// it'd be bad if somebody tried to take a lock that had been 
	//   deleted...  (info is only valid on a lock's home node)
	assert((ID(me).node() != gasnet_mynode()) ||
	       in_use);

	if(owner == gasnet_mynode()) {
#ifdef LOCK_TRACING
          {
            LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
            item.lock_id = me.id;
            item.owner = gasnet_mynode();
            item.action = LockTraceItem::ACT_LOCAL_REQUEST;
          }
#endif
	  // case 1: we own the lock
	  // can we grant it?
	  if((count == ZERO_COUNT) || ((mode == new_mode) && (mode != MODE_EXCL))) {
	    mode = new_mode;
	    count++;
	    log_reservation.spew("count ++(1) [%p]=%d", &count, count);
	    got_lock = true;
#ifdef LOCK_TRACING
            {
              LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
              item.lock_id = me.id;
              item.owner = gasnet_mynode();
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

	// if we didn't get the lock, put our event on the queue of local
	//  waiters - create an event if we weren't given one to use
	if(!got_lock) {
	  if(!after_lock) {
	    after_lock = GenEventImpl::create_genevent();
	    after_lock_event = after_lock->current_event();
	  }
	  local_waiters[new_mode].push_back(after_lock);
	}
      }

      if(lock_request_target != -1)
      {
	LockRequestMessage::send_request(lock_request_target, gasnet_mynode(),
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
      if(got_lock && after_lock) 
	after_lock->trigger(after_lock_event.gen, gasnet_mynode());

      return after_lock_event;
    }

    // factored-out code to select one or more local waiters on a lock
    //  fills events to trigger into 'to_wake' and returns true if any were
    //  found - NOTE: ASSUMES LOCK IS ALREADY HELD!
    bool ReservationImpl::select_local_waiters(std::deque<GenEventImpl *>& to_wake)
    {
      if(local_waiters.size() == 0)
	return false;

      // favor the local waiters
      log_reservation.debug("reservation going to local waiter: size=%zd first=%d(%zd)",
	       local_waiters.size(), 
	       local_waiters.begin()->first,
	       local_waiters.begin()->second.size());
	
      // further favor exclusive waiters
      if(local_waiters.find(MODE_EXCL) != local_waiters.end()) {
	std::deque<GenEventImpl *>& excl_waiters = local_waiters[MODE_EXCL];
	to_wake.push_back(excl_waiters.front());
	excl_waiters.pop_front();
	  
	// if the set of exclusive waiters is empty, delete it
	if(excl_waiters.size() == 0)
	  local_waiters.erase(MODE_EXCL);
	  
	mode = MODE_EXCL;
	count = ZERO_COUNT + 1;
	log_reservation.spew("count <-1 [%p]=%d", &count, count);
      } else {
	// pull a whole list of waiters that want to share with the same mode
	std::map<unsigned, std::deque<GenEventImpl *> >::iterator it = local_waiters.begin();
	
	mode = it->first;
	count = ZERO_COUNT + it->second.size();
	log_reservation.spew("count <-waiters [%p]=%d", &count, count);
	assert(count > ZERO_COUNT);
	// grab the list of events wanting to share the lock
	to_wake.swap(it->second);
	local_waiters.erase(it);  // actually pull list off map!
	// TODO: can we share with any other nodes?
      }
#ifdef LOCK_TRACING
      {
        LockTraceItem &item = Tracer<LockTraceItem>::trace_item();
        item.lock_id = me.id;
        item.owner = gasnet_mynode();
        item.action = LockTraceItem::ACT_LOCAL_GRANT;
      }
#endif

      return true;
    }

    void ReservationImpl::release(void)
    {
      // make a list of events that we be woken - can't do it while holding the
      //  lock's mutex (because the event we trigger might try to take the lock)
      std::deque<GenEventImpl *> to_wake;

      int release_target = -1;
      int grant_target = -1;
      NodeSet copy_waiters;

      do {
	log_reservation.debug(            "release: reservation=" IDFMT " count=%d mode=%d owner=%d", // share=%lx wait=%lx",
			me.id, count, mode, owner); //, remote_sharer_mask, remote_waiter_mask);
	AutoHSLLock a(mutex); // hold mutex on lock for entire function

	assert(count > ZERO_COUNT);

	// if this isn't the last holder of the lock, just decrement count
	//  and return
	count--;
	log_reservation.spew("count -- [%p]=%d", &count, count);
	log_reservation.debug(            "post-release: reservation=" IDFMT " count=%d mode=%d", // share=%lx wait=%lx",
		 me.id, count, mode); //, remote_sharer_mask, remote_waiter_mask);
	if(count > ZERO_COUNT) break;

	// case 1: if we were sharing somebody else's lock, tell them we're
	//  done
	if(owner != gasnet_mynode()) {
	  assert(mode != MODE_EXCL);
	  mode = 0;

	  release_target = owner;
	  break;
	}

	// case 2: we own the lock, so we can give it to another waiter
	//  (local or remote)
	bool any_local = select_local_waiters(to_wake);
	assert(!any_local || (to_wake.size() > 0));

	if(!any_local && (!remote_waiter_mask.empty())) {
	  // nobody local wants it, but another node does
	  //HACK int new_owner = remote_waiter_mask.find_first_set();
	  // TODO: use iterator - all we need is *begin()
          int new_owner = 0;  while(!remote_waiter_mask.contains(new_owner)) new_owner++;
          remote_waiter_mask.remove(new_owner);

	  log_reservation.debug(              "reservation going to remote waiter: new=%d", // mask=%lx",
		   new_owner); //, remote_waiter_mask);

	  grant_target = new_owner;
          copy_waiters = remote_waiter_mask;

	  owner = new_owner;
          remote_waiter_mask = NodeSet();
	}
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

      for(std::deque<GenEventImpl *>::iterator it = to_wake.begin();
	  it != to_wake.end();
	  it++) {
	log_reservation.debug("release trigger: reservation=" IDFMT " event=" IDFMT "/%d",
			me.id, (*it)->me.id(), (*it)->generation + 1);
	(*it)->trigger_current();
      }
    }

    bool ReservationImpl::is_locked(unsigned check_mode, bool excl_ok)
    {
      // checking the owner can be done atomically, so doesn't need mutex
      if(owner != gasnet_mynode()) return false;

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
	assert(owner == gasnet_mynode());
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

    /*static*/ void DestroyLockMessage::send_request(gasnet_node_t target,
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

}; // namespace Realm
