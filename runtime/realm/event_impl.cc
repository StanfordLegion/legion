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

#include "event_impl.h"
#include "lowlevel_impl.h"
#include "logging.h"

#ifdef USE_CUDA
GASNETT_THREADKEY_DECLARE(gpu_thread_ptr);
#endif

namespace Realm {

  Logger log_event("event");
  Logger log_barrier("barrier");

  ////////////////////////////////////////////////////////////////////////
  //
  // class Event
  //

  /*static*/ const Event Event::NO_EVENT = { 0, 0 };
  // Take this you POS c++ type system
  /* static */ const UserEvent UserEvent::NO_USER_EVENT = 
    *(static_cast<UserEvent*>(const_cast<Event*>(&Event::NO_EVENT)));

  bool Event::has_triggered(void) const
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    if(!id) return true; // special case: NO_EVENT has always triggered
    EventImpl *e = get_runtime()->get_event_impl(*this);
    return e->has_triggered(gen);
  }

  // creates an event that won't trigger until all input events have
  /*static*/ Event Event::merge_events(const std::set<Event>& wait_for)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    return GenEventImpl::merge_events(wait_for);
  }

  /*static*/ Event Event::merge_events(Event ev1, Event ev2,
				       Event ev3 /*= NO_EVENT*/, Event ev4 /*= NO_EVENT*/,
				       Event ev5 /*= NO_EVENT*/, Event ev6 /*= NO_EVENT*/)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    return GenEventImpl::merge_events(ev1, ev2, ev3, ev4, ev5, ev6);
  }

  void Event::wait(void) const
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    if(!id) return;  // special case: never wait for NO_EVENT
    EventImpl *e = get_runtime()->get_event_impl(*this);

    // early out case too
    if(e->has_triggered(gen)) return;

    // waiting on an event does not count against the low level's time
    DetailedTimer::ScopedPush sp2(TIME_NONE);

    // are we a thread that knows how to do something useful while waiting?
    if(PreemptableThread::preemptable_sleep(*this))
      return;

    // maybe a GPU thread?
#ifdef USE_CUDA
    void *ptr = gasnett_threadkey_get(gpu_thread_ptr);
    if(ptr != 0) {
      //assert(0);
      //printf("oh, good - we're a gpu thread - we'll spin for now\n");
      //printf("waiting for " IDFMT "/%d\n", id, gen);
      while(!e->has_triggered(gen)) {
#ifdef __SSE2__
	_mm_pause();
#else
	usleep(1000);
#endif
      }
      //printf("done\n");
      return;
    }
#endif
    // we're probably screwed here - try waiting and polling gasnet while
    //  we wait
    //printf("waiting on event, polling gasnet to hopefully not die\n");
    while(!e->has_triggered(gen)) {
      // can't poll here - the GPU DMA code sometimes polls from inside an active
      //  message handler (consider turning polling back on once that's fixed)
      //do_some_polling();
#ifdef __SSE2__
      _mm_pause();
#endif
      // no sleep - we don't want an OS-scheduler-latency here
      //usleep(10000);
    }
    return;
    //assert(ptr != 0);
  }

  void Event::external_wait(void) const
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    if(!id) return;  // special case: never wait for NO_EVENT
    EventImpl *e = get_runtime()->get_event_impl(*this);

    // early out case too
    if(e->has_triggered(gen)) return;
    
    // waiting on an event does not count against the low level's time
    DetailedTimer::ScopedPush sp2(TIME_NONE);
    
    e->external_wait(gen);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class UserEvent
  //

  /*static*/ UserEvent UserEvent::create_user_event(void)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    Event e = GenEventImpl::create_genevent()->current_event();
    assert(e.id != 0);
    UserEvent u;
    u.id = e.id;
    u.gen = e.gen;
    return u;
  }

  void UserEvent::trigger(Event wait_on) const
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

    GenEventImpl *e = get_runtime()->get_genevent_impl(*this);
#ifdef EVENT_GRAPH_TRACE
    Event enclosing = find_enclosing_termination_event();
    log_event_graph.info("Event Trigger: (" IDFMT ",%d) (" IDFMT 
			 ",%d) (" IDFMT ",%d)",
			 id, gen, wait_on.id, wait_on.gen,
			 enclosing.id, enclosing.gen);
#endif
    e->trigger(gen, gasnet_mynode(), wait_on);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Barrier
  //

  /*static*/ Barrier Barrier::create_barrier(unsigned expected_arrivals,
					     ReductionOpID redop_id /*= 0*/,
					     const void *initial_value /*= 0*/,
					     size_t initial_value_size /*= 0*/)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

    BarrierImpl *impl = BarrierImpl::create_barrier(expected_arrivals, redop_id, initial_value, initial_value_size);
    Barrier b = impl->me.convert<Barrier>();
    b.gen = impl->generation + 1;
    b.timestamp = 0;

#ifdef EVENT_GRAPH_TRACE
    log_event_graph.info("Barrier Creation: " IDFMT " %d", b.id, expected_arrivals);
#endif

    return b;
  }

  void Barrier::destroy_barrier(void)
  {
    log_barrier.info("barrier destruction request: " IDFMT "/%d", id, gen);
  }

  Barrier Barrier::advance_barrier(void) const
  {
    Barrier nextgen;
    nextgen.id = id;
    nextgen.gen = gen + 1;
    nextgen.timestamp = 0;

    return nextgen;
  }

  Barrier Barrier::alter_arrival_count(int delta) const
  {
    timestamp_t timestamp = __sync_fetch_and_add(&BarrierImpl::barrier_adjustment_timestamp, 1);
#ifdef EVENT_GRAPH_TRACE
    Event enclosing = find_enclosing_termination_event();
    log_event_graph.info("Barrier Alter: (" IDFMT ",%d) (" IDFMT
			 ",%d) %d", id, gen, enclosing.id, enclosing.gen, delta);
#endif
    BarrierImpl *impl = get_runtime()->get_barrier_impl(*this);
    impl->adjust_arrival(gen, delta, timestamp, Event::NO_EVENT, 0, 0);

    Barrier with_ts;
    with_ts.id = id;
    with_ts.gen = gen;
    with_ts.timestamp = timestamp;

    return with_ts;
  }

  Barrier Barrier::get_previous_phase(void) const
  {
    Barrier result = *this;
    result.gen--;
    return result;
  }

  void Barrier::arrive(unsigned count /*= 1*/, Event wait_on /*= Event::NO_EVENT*/,
		       const void *reduce_value /*= 0*/, size_t reduce_value_size /*= 0*/) const
  {
#ifdef EVENT_GRAPH_TRACE
    Event enclosing = find_enclosing_termination_event();
    log_event_graph.info("Barrier Arrive: (" IDFMT ",%d) (" IDFMT
			 ",%d) (" IDFMT ",%d) %d",
			 id, gen, wait_on.id, wait_on.gen,
			 enclosing.id, enclosing.gen, count);
#endif
    // arrival uses the timestamp stored in this barrier object
    BarrierImpl *impl = get_runtime()->get_barrier_impl(*this);
    impl->adjust_arrival(gen, -count, timestamp, wait_on,
			 reduce_value, reduce_value_size);
  }

  bool Barrier::get_result(void *value, size_t value_size) const
  {
    BarrierImpl *impl = get_runtime()->get_barrier_impl(*this);
    return impl->get_result(gen, value, value_size);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GenEventImpl
  //

  GenEventImpl::GenEventImpl(void)
    : me((ID::IDType)-1), owner(-1)
  {
    generation = 0;
    gen_subscribed = 0;
    next_free = 0;
  }

  void GenEventImpl::init(ID _me, unsigned _init_owner)
  {
    me = _me;
    owner = _init_owner;
    generation = 0;
    gen_subscribed = 0;
    next_free = 0;
  }


    // Perform our merging events in a lock free way
    class EventMerger : public EventWaiter {
    public:
      EventMerger(GenEventImpl *_finish_event)
	: count_needed(1), finish_event(_finish_event)
      {
      }

      virtual ~EventMerger(void)
      {
      }

      void add_event(Event wait_for)
      {
	if(wait_for.has_triggered()) return; // early out
        // Increment the count and then add ourselves
        __sync_fetch_and_add(&count_needed, 1);
	// step 2: enqueue ourselves on the input event
	EventImpl::add_waiter(wait_for, this);
      }

      // arms the merged event once you're done adding input events - just
      //  decrements the count for the implicit 'init done' event
      // return a boolean saying whether it triggered upon arming (which
      //  means the caller should delete this EventMerger)
      bool arm(void)
      {
	bool nuke = event_triggered();
        return nuke;
      }

      virtual bool event_triggered(void)
      {
	// save ID and generation because we can't reference finish_event after the
	// decrement (unless last_trigger ends up being true)
	ID::IDType id = finish_event->me.id();
	Event::gen_t gen = finish_event->generation;

	int count_left = __sync_fetch_and_add(&count_needed, -1);

        // Put the logging first to avoid segfaults
        log_event.info("received trigger merged event " IDFMT "/%d (%d)",
		  id, gen, count_left);

	// count is the value before the decrement, so it was 1, it's now 0
	bool last_trigger = (count_left == 1);

	if(last_trigger) {
	  finish_event->trigger_current();
	}

        // caller can delete us if this was the last trigger
        return last_trigger;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"event merger: " IDFMT "/%d\n", finish_event->me.id(), finish_event->generation+1);
      }

    protected:
      int count_needed;
      GenEventImpl *finish_event;
    };

    // creates an event that won't trigger until all input events have
    /*static*/ Event GenEventImpl::merge_events(const std::set<Event>& wait_for)
    {
      if (wait_for.empty())
        return Event::NO_EVENT;
      // scan through events to see how many exist/haven't fired - we're
      //  interested in counts of 0, 1, or 2+ - also remember the first
      //  event we saw for the count==1 case
      int wait_count = 0;
      Event first_wait;
      for(std::set<Event>::const_iterator it = wait_for.begin();
	  (it != wait_for.end()) && (wait_count < 2);
	  it++)
	if(!(*it).has_triggered()) {
	  if(!wait_count) first_wait = *it;
	  wait_count++;
	}
      log_event.info("merging events - at least %d not triggered",
		wait_count);

      // Avoid these optimizations if we are doing event graph tracing
#ifndef EVENT_GRAPH_TRACE
      // counts of 0 or 1 don't require any merging
      if(wait_count == 0) return Event::NO_EVENT;
      if(wait_count == 1) return first_wait;
#else
      if (wait_for.size() == 1)
        return *(wait_for.begin());
#endif
      // counts of 2+ require building a new event and a merger to trigger it
      GenEventImpl *finish_event = GenEventImpl::create_genevent();
      EventMerger *m = new EventMerger(finish_event);

      // get the Event for this GenEventImpl before any triggers can occur
      Event e = finish_event->current_event();

#ifdef EVENT_GRAPH_TRACE
      log_event_graph.info("Event Merge: (" IDFMT ",%d) %ld", 
			   e.id, e.gen, wait_for.size());
#endif

      for(std::set<Event>::const_iterator it = wait_for.begin();
	  it != wait_for.end();
	  it++) {
	log_event.info("merged event " IDFMT "/%d waiting for " IDFMT "/%d",
		  finish_event->me.id(), finish_event->generation, (*it).id, (*it).gen);
	m->add_event(*it);
#ifdef EVENT_GRAPH_TRACE
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ",%d)",
                             finish_event->me.id(), finish_event->generation,
                             it->id, it->gen);
#endif
      }

      // once they're all added - arm the thing (it might go off immediately)
      if(m->arm())
        delete m;

      return e;
    }

    /*static*/ Event GenEventImpl::merge_events(Event ev1, Event ev2,
						Event ev3 /*= NO_EVENT*/, Event ev4 /*= NO_EVENT*/,
						Event ev5 /*= NO_EVENT*/, Event ev6 /*= NO_EVENT*/)
    {
      // scan through events to see how many exist/haven't fired - we're
      //  interested in counts of 0, 1, or 2+ - also remember the first
      //  event we saw for the count==1 case
      int wait_count = 0;
      Event first_wait;
      if(!ev6.has_triggered()) { first_wait = ev6; wait_count++; }
      if(!ev5.has_triggered()) { first_wait = ev5; wait_count++; }
      if(!ev4.has_triggered()) { first_wait = ev4; wait_count++; }
      if(!ev3.has_triggered()) { first_wait = ev3; wait_count++; }
      if(!ev2.has_triggered()) { first_wait = ev2; wait_count++; }
      if(!ev1.has_triggered()) { first_wait = ev1; wait_count++; }

      // Avoid these optimizations if we are doing event graph tracing
#ifndef EVENT_GRAPH_TRACE
      // counts of 0 or 1 don't require any merging
      if(wait_count == 0) return Event::NO_EVENT;
      if(wait_count == 1) return first_wait;
#else
      int existential_count = 0;
      if (ev1.exists()) existential_count++;
      if (ev2.exists()) existential_count++;
      if (ev3.exists()) existential_count++;
      if (ev4.exists()) existential_count++;
      if (ev5.exists()) existential_count++;
      if (ev6.exists()) existential_count++;
      if (existential_count == 0)
        return Event::NO_EVENT;
      if (existential_count == 1)
      {
        if (ev1.exists()) return ev1;
        if (ev2.exists()) return ev2;
        if (ev3.exists()) return ev3;
        if (ev4.exists()) return ev4;
        if (ev5.exists()) return ev5;
        if (ev6.exists()) return ev6;
      }
#endif

      // counts of 2+ require building a new event and a merger to trigger it
      GenEventImpl *finish_event = GenEventImpl::create_genevent();
      EventMerger *m = new EventMerger(finish_event);

      // get the Event for this GenEventImpl before any triggers can occur
      Event e = finish_event->current_event();

      m->add_event(ev1);
      m->add_event(ev2);
      m->add_event(ev3);
      m->add_event(ev4);
      m->add_event(ev5);
      m->add_event(ev6);

#ifdef EVENT_GRAPH_TRACE
      log_event_graph.info("Event Merge: (" IDFMT ",%d) %d",
               finish_event->me.id(), finish_event->generation, existential_count);
      if (ev1.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation, ev1.id, ev1.gen);
      if (ev2.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation, ev2.id, ev2.gen);
      if (ev3.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation, ev3.id, ev3.gen);
      if (ev4.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation, ev4.id, ev4.gen);
      if (ev5.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation, ev5.id, ev5.gen);
      if (ev6.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation, ev6.id, ev6.gen);
#endif

      // once they're all added - arm the thing (it might go off immediately)
      if(m->arm())
        delete m;

      return e;
    }

    /*static*/ GenEventImpl *GenEventImpl::create_genevent(void)
    {
      GenEventImpl *impl = get_runtime()->local_event_free_list->alloc_entry();
      assert(impl);
      assert(ID(impl->me).type() == ID::ID_EVENT);

      log_event.info("event created: event=" IDFMT "/%d", impl->me.id(), impl->generation+1);
#ifdef EVENT_TRACING
      {
	EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
	item.event_id = impl->me.id();
	item.event_gen = impl->me.gen;
	item.action = EventTraceItem::ACT_CREATE;
      }
#endif
      return impl;
    }
    
    void GenEventImpl::check_for_catchup(Event::gen_t implied_trigger_gen)
    {
      // early out before we take a lock
      if(implied_trigger_gen <= generation) return;

      // now take a lock and see if we really need to catch up
      std::vector<EventWaiter *> stale_waiters;
      {
	AutoHSLLock a(mutex);

	if(implied_trigger_gen > generation) {
	  assert(owner != gasnet_mynode());  // cannot be a local event

	  log_event.info("event catchup: " IDFMT "/%d -> %d",
			 me.id(), generation, implied_trigger_gen);
	  generation = implied_trigger_gen;
	  stale_waiters.swap(local_waiters);  // we'll actually notify them below
	}
      }

      if(!stale_waiters.empty()) {
	for(std::vector<EventWaiter *>::iterator it = stale_waiters.begin();
	    it != stale_waiters.end();
	    it++)
	  (*it)->event_triggered();
      }
    }

    bool GenEventImpl::add_waiter(Event::gen_t needed_gen, EventWaiter *waiter)
    {
#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = me->id;
        item.event_gen = needed_gen;
        item.action = EventTraceItem::ACT_WAIT;
      }
#endif
      bool trigger_now = false;

      int subscribe_owner = -1;
      Event subscribe_event;
      Event::gen_t previous_subscribe_gen = 0;
      {
	AutoHSLLock a(mutex);

	if(needed_gen > generation) {
	  log_event.debug("event not ready: event=" IDFMT "/%d owner=%d gen=%d subscr=%d",
		    me.id(), needed_gen, owner, generation, gen_subscribed);

	  // catchup code for remote events has been moved to get_genevent_impl, so
	  //  we should never be asking for a stale version here
	  assert(needed_gen == (generation + 1));

	  // do we need to subscribe?
	  if((owner != gasnet_mynode()) && (gen_subscribed < needed_gen)) {
	    previous_subscribe_gen = gen_subscribed;
	    gen_subscribed = needed_gen;
	    subscribe_owner = owner;
	    subscribe_event = me.convert<Event>();
	    subscribe_event.gen = needed_gen;
	  }

	  // now we add to the local waiter list
	  local_waiters.push_back(waiter);
	} else {
	  // event we are interested in has already triggered!
	  trigger_now = true; // actually do trigger outside of mutex
	}
      }

      if((subscribe_owner != -1))
	EventSubscribeMessage::send_request(owner, subscribe_event, previous_subscribe_gen);

      if(trigger_now) {
	bool nuke = waiter->event_triggered();
        if(nuke)
          delete waiter;
      }

      return true;  // waiter is always either enqueued or triggered right now
    }

    ///////////////////////////////////////////////////
    // Events



    void EventTriggerMessage::RequestArgs::apply(gasnet_node_t target)
    {
      EventTriggerMessage::send_request(target, event);
    }

  /*static*/ void EventTriggerMessage::send_request(gasnet_node_t target, Event event)
  {
    RequestArgs args;

    args.node = gasnet_mynode();
    args.event = event;
    Message::request(target, args);
  }

  /*static*/ void EventTriggerMessage::broadcast_request(const NodeSet& targets, Event event)
  {
    RequestArgs args;

    args.node = gasnet_mynode();
    args.event = event;
    targets.map(args);
  }

  /*static*/ void EventSubscribeMessage::send_request(gasnet_node_t target, Event event, Event::gen_t previous_gen)
  {
    RequestArgs args;

    args.node = gasnet_mynode();
    args.event = event;
    args.previous_subscribe_gen = previous_gen;
    Message::request(target, args);
  }

    // only called for generational events
    /*static*/ void EventSubscribeMessage::handle_request(EventSubscribeMessage::RequestArgs args)
    {
      log_event.debug("event subscription: node=%d event=" IDFMT "/%d",
		args.node, args.event.id, args.event.gen);

      GenEventImpl *impl = get_runtime()->get_genevent_impl(args.event);

#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = args.event.id; 
        item.event_gen = args.event.gen;
        item.action = EventTraceItem::ACT_WAIT;
      }
#endif

      // early-out case: if we can see the generation needed has already
      //  triggered, signal without taking the mutex
      unsigned stale_gen = impl->generation;
      if(stale_gen >= args.event.gen) {
	log_event.debug("event subscription early-out: node=%d event=" IDFMT "/%d (<= %d)",
		  args.node, args.event.id, args.event.gen, stale_gen);
	Event triggered = args.event;
	triggered.gen = stale_gen;
	EventTriggerMessage::send_request(args.node, triggered);
	return;
      }

      {
	AutoHSLLock a(impl->mutex);
        // first trigger any generations which are below our current generation
        if(impl->generation > (args.previous_subscribe_gen)) {
          log_event.debug("event subscription already done: node=%d event=" IDFMT "/%d (<= %d)",
		    args.node, args.event.id, args.event.gen, impl->generation);
	  Event triggered = args.event;
	  triggered.gen = impl->generation;
	  EventTriggerMessage::send_request(args.node, triggered);
        }

	// if the subscriber is asking about a generation that JUST triggered, the above trigger message
	//  is all we needed to do
	if(args.event.gen > impl->generation) {
	  // barrier logic is now separated, so we should never hear about an event generate beyond the one
	  //  that will trigger next
	  assert(args.event.gen <= (impl->generation + 1));

	  impl->remote_waiters.add(args.node);
	  log_event.debug("event subscription recorded: node=%d event=" IDFMT "/%d (> %d)",
		    args.node, args.event.id, args.event.gen, impl->generation);
	}
      }
    } 

    /*static*/ void EventTriggerMessage::handle_request(EventTriggerMessage::RequestArgs args)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      log_event.debug("Remote trigger of event " IDFMT "/%d from node %d!",
		args.event.id, args.event.gen, args.node);
      GenEventImpl *impl = get_runtime()->get_genevent_impl(args.event);
      impl->trigger(args.event.gen, args.node);
    }


  /*static*/ Barrier::timestamp_t BarrierImpl::barrier_adjustment_timestamp;



    /*static*/ bool EventImpl::add_waiter(Event needed, EventWaiter *waiter)
    {
      return get_runtime()->get_event_impl(needed)->add_waiter(needed.gen, waiter);
    }

    bool GenEventImpl::has_triggered(Event::gen_t needed_gen)
    {
#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = me.id;
        item.event_gen = needed_gen;
        item.action = EventTraceItem::ACT_QUERY;
      }
#endif
      return (needed_gen <= generation);
    }

    class PthreadCondWaiter : public EventWaiter {
    public:
      PthreadCondWaiter(GASNetCondVar &_cv)
        : cv(_cv)
      {
      }
      virtual ~PthreadCondWaiter(void) 
      {
      }

      virtual bool event_triggered(void)
      {
        // Need to hold the lock to avoid the race
        AutoHSLLock(cv.mutex);
	cv.signal();
        // we're allocated on caller's stack, so deleting would be bad
        return false;
      }
      virtual void print_info(FILE *f) { fprintf(f,"external waiter\n"); }

    public:
      GASNetCondVar &cv;
    };

    void GenEventImpl::external_wait(Event::gen_t gen_needed)
    {
      GASNetCondVar cv(mutex);
      PthreadCondWaiter w(cv);
      {
	AutoHSLLock a(mutex);

	if(gen_needed > generation) {
	  local_waiters.push_back(&w);
    
	  if((owner != gasnet_mynode()) && (gen_needed > gen_subscribed)) {
	    printf("AAAH!  Can't subscribe to another node's event in external_wait()!\n");
	    exit(1);
	  }

	  // now just sleep on the condition variable - hope we wake up
	  cv.wait();
	}
      }
    }

    class DeferredEventTrigger : public EventWaiter {
    public:
      DeferredEventTrigger(GenEventImpl *_after_event)
	: after_event(_after_event)
      {}

      virtual ~DeferredEventTrigger(void) { }

      virtual bool event_triggered(void)
      {
	log_event.info("deferred trigger occuring: " IDFMT "/%d", after_event->me.id(), after_event->generation+1);
	after_event->trigger_current();
        return true;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"deferred trigger: after=" IDFMT "/%d\n",
		after_event->me.id(), after_event->generation+1);
      }

    protected:
      GenEventImpl *after_event;
    };

    void GenEventImpl::trigger_current(void)
    {
      // wrapper triggers the next generation on the current node
      trigger(generation + 1, gasnet_mynode());
    }

    void GenEventImpl::trigger(Event::gen_t gen_triggered, int trigger_node, Event wait_on)
    {
      if(!wait_on.has_triggered()) {
	// deferred trigger
	// TODO: forward the deferred trigger to the owning node if it's remote
	log_event.info("deferring event trigger: in=" IDFMT "/%d out=" IDFMT "/%d",
		       wait_on.id, wait_on.gen, me.id(), gen_triggered);
	EventImpl::add_waiter(wait_on, new DeferredEventTrigger(this));
	return;
      }

      log_event.spew("event triggered: event=" IDFMT "/%d by node %d", 
		me.id(), gen_triggered, trigger_node);
#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = me.id;
        item.event_gen = gen_triggered;
        item.action = EventTraceItem::ACT_TRIGGER;
      }
#endif

      std::vector<EventWaiter *> to_wake;
      {
	AutoHSLLock a(mutex);

        // SJT: there is at least one unavoidable case where we'll receive
	//  duplicate trigger notifications, so if we see a triggering of
	//  an older generation, just ignore it
	if(gen_triggered <= generation) return;

	// in preparation for switching everybody over to trigger_current(), complain
	//  LOUDLY if this wouldn't actually be a triggering of the current generation
	if(gen_triggered != (generation + 1))
	  log_event.error("HELP!  non-current event generation being triggered: " IDFMT "/%d vs %d",
			  me.id(), gen_triggered, generation + 1);

        generation = gen_triggered;

	// grab whole list of local waiters - we'll trigger them once we let go of the lock
	//printf("[%d] LOCAL WAITERS: %zd\n", gasnet_mynode(), local_waiters.size());
	to_wake.swap(local_waiters);

	// notify remote waiters and/or event's actual owner
	if(owner == gasnet_mynode()) {
	  // send notifications to every other node that has subscribed
	  //  (except the one that triggered)
          if (!remote_waiters.empty())
          {
	    Event triggered = me.convert<Event>();
	    triggered.gen = gen_triggered;
	    
            NodeSet send_mask;
            send_mask.swap(remote_waiters);
	    EventTriggerMessage::broadcast_request(send_mask, triggered);
          }
	} else {
	  if(((unsigned)trigger_node) == gasnet_mynode()) {
	    // if we're not the owner, we just send to the owner and let him
	    //  do the broadcast (assuming the trigger was local)
	    //assert(remote_waiters == 0);

	    Event triggered = me.convert<Event>();
	    triggered.gen = gen_triggered;
	    EventTriggerMessage::send_request(owner, triggered);
	  }
	}
      }

      // if this is one of our events, put ourselves on the free
      //  list (we don't need our lock for this)
      if(owner == gasnet_mynode()) {
	get_runtime()->local_event_free_list->free_entry(this);
      }

      // now that we've let go of the lock, notify all the waiters who wanted
      //  this event generation (or an older one)
      {
	for(std::vector<EventWaiter *>::iterator it = to_wake.begin();
	    it != to_wake.end();
	    it++) {
	  bool nuke = (*it)->event_triggered();
          if(nuke) {
            //printf("deleting: "); (*it)->print_info(); fflush(stdout);
            delete (*it);
          }
        }
      }
    }

    /*static*/ BarrierImpl *BarrierImpl::create_barrier(unsigned expected_arrivals,
							ReductionOpID redopid,
							const void *initial_value /*= 0*/,
							size_t initial_value_size /*= 0*/)
    {
      BarrierImpl *impl = get_runtime()->local_barrier_free_list->alloc_entry();
      assert(impl);
      assert(impl->me.type() == ID::ID_BARRIER);

      // set the arrival count
      impl->base_arrival_count = expected_arrivals;

      if(redopid == 0) {
	assert(initial_value_size == 0);
	impl->redop_id = 0;
	impl->redop = 0;
	impl->initial_value = 0;
	impl->value_capacity = 0;
	impl->final_values = 0;
      } else {
	impl->redop_id = redopid;  // keep the ID too so we can share it
	impl->redop = get_runtime()->reduce_op_table[redopid];

	assert(initial_value != 0);
	assert(initial_value_size == impl->redop->sizeof_lhs);

	impl->initial_value = (char *)malloc(initial_value_size);
	memcpy(impl->initial_value, initial_value, initial_value_size);

	impl->value_capacity = 0;
	impl->final_values = 0;
      }

      // and let the barrier rearm as many times as necessary without being released
      impl->free_generation = (unsigned)-1;

      log_barrier.info("barrier created: " IDFMT "/%d base_count=%d redop=%d",
		       impl->me.id(), impl->generation, impl->base_arrival_count, redopid);
#ifdef EVENT_TRACING
      {
	EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
	item.event_id = impl->me.id();
	item.event_gen = impl->me.gen;
	item.action = EventTraceItem::ACT_CREATE;
      }
#endif
      return impl;
    }

    BarrierImpl::BarrierImpl(void)
      : me((ID::IDType)-1), owner(-1)
    {
      generation = 0;
      gen_subscribed = 0;
      first_generation = free_generation = 0;
      next_free = 0;
      remote_subscribe_gens.clear();
      remote_trigger_gens.clear();
      base_arrival_count = 0;
      redop = 0;
      initial_value = 0;
      value_capacity = 0;
      final_values = 0;
    }

    void BarrierImpl::init(ID _me, unsigned _init_owner)
    {
      me = _me;
      owner = _init_owner;
      generation = 0;
      gen_subscribed = 0;
      first_generation = free_generation = 0;
      next_free = 0;
      remote_subscribe_gens.clear();
      remote_trigger_gens.clear();
      base_arrival_count = 0;
      redop = 0;
      initial_value = 0;
      value_capacity = 0;
      final_values = 0;
    }

    /*static*/ void BarrierAdjustMessage::handle_request(RequestArgs args, const void *data, size_t datalen)
    {
      log_barrier.info("received barrier arrival: delta=%d in=" IDFMT "/%d out=" IDFMT "/%d (%llx)",
		       args.delta, args.wait_on.id, args.wait_on.gen, args.barrier.id, args.barrier.gen, args.barrier.timestamp);
      BarrierImpl *impl = get_runtime()->get_barrier_impl(args.barrier);
      impl->adjust_arrival(args.barrier.gen, args.delta, args.barrier.timestamp, args.wait_on,
			   datalen ? data : 0, datalen);
    }

    /*static*/ void BarrierAdjustMessage::send_request(gasnet_node_t target, Barrier barrier, int delta, Event wait_on,
						       const void *data, size_t datalen)
    {
      RequestArgs args;
      
      args.barrier = barrier;
      args.delta = delta;
      args.wait_on = wait_on;
      
      Message::request(target, args, data, datalen, PAYLOAD_COPY);
    }

    /*static*/ void BarrierSubscribeMessage::send_request(gasnet_node_t target, ID::IDType barrier_id, Event::gen_t subscribe_gen)
    {
      RequestArgs args;

      args.node = gasnet_mynode();
      args.barrier_id = barrier_id;
      args.subscribe_gen = subscribe_gen;
      
      Message::request(target, args);
    }

    /*static*/ void BarrierTriggerMessage::send_request(gasnet_node_t target, ID::IDType barrier_id,
							Event::gen_t trigger_gen, Event::gen_t previous_gen,
							Event::gen_t first_generation, ReductionOpID redop_id,
							const void *data, size_t datalen)
    {
      RequestArgs args;

      args.node = gasnet_mynode();
      args.barrier_id = barrier_id;
      args.trigger_gen = trigger_gen;
      args.previous_gen = previous_gen;
      args.first_generation = first_generation;
      args.redop_id = redop_id;

      Message::request(target, args, data, datalen, PAYLOAD_COPY);
    }

// like strdup, but works on arbitrary byte arrays
static void *bytedup(const void *data, size_t datalen)
{
  if(datalen == 0) return 0;
  void *dst = malloc(datalen);
  assert(dst != 0);
  memcpy(dst, data, datalen);
  return dst;
}

    class DeferredBarrierArrival : public EventWaiter {
    public:
      DeferredBarrierArrival(Barrier _barrier, int _delta, const void *_data, size_t _datalen)
	: barrier(_barrier), delta(_delta), data(bytedup(_data, _datalen)), datalen(_datalen)
      {}

      virtual ~DeferredBarrierArrival(void)
      {
	if(data)
	  free(data);
      }

      virtual bool event_triggered(void)
      {
	log_barrier.info("deferred barrier arrival: " IDFMT "/%d (%llx), delta=%d",
			 barrier.id, barrier.gen, barrier.timestamp, delta);
	BarrierImpl *impl = get_runtime()->get_barrier_impl(barrier);
	impl->adjust_arrival(barrier.gen, delta, barrier.timestamp, Event::NO_EVENT, data, datalen);
        return true;
      }

      virtual void print_info(FILE *f)
      {
	fprintf(f,"deferred arrival: barrier=" IDFMT "/%d (%llx), delta=%d datalen=%zd\n",
		barrier.id, barrier.gen, barrier.timestamp, delta, datalen);
      }

    protected:
      Barrier barrier;
      int delta;
      void *data;
      size_t datalen;
    };

    BarrierImpl::Generation::Generation(void) : unguarded_delta(0) {}
    BarrierImpl::Generation::~Generation(void)
      {
        for(std::map<int, PerNodeUpdates *>::iterator it = pernode.begin();
            it != pernode.end();
            it++)
          delete (it->second);
      }

    void BarrierImpl::Generation::handle_adjustment(Barrier::timestamp_t ts, int delta)
      {
	if(ts == 0) {
	  // simple case - apply delta directly
	  unguarded_delta += delta;
	  return;
	}

        int node = ts >> BARRIER_TIMESTAMP_NODEID_SHIFT;
        PerNodeUpdates *pn;
        std::map<int, PerNodeUpdates *>::iterator it = pernode.find(node);
        if(it != pernode.end()) {
          pn = it->second;
        } else {
          pn = new PerNodeUpdates;
          pernode[node] = pn;
        }
        if(delta > 0) {
          // TODO: really need two timestamps to properly order increments
          unguarded_delta += delta;
          pn->last_ts = ts;
          std::map<Barrier::timestamp_t, int>::iterator it2 = pn->pending.begin();
          while((it2 != pn->pending.end()) && (it2->first <= pn->last_ts)) {
            log_barrier.info("applying pending delta: %llx/%d", it2->first, it2->second);
            unguarded_delta += it2->second;
            pn->pending.erase(it2);
            it2 = pn->pending.begin();
          }
        } else {
          // if the timestamp is late enough, we can apply this directly
          if(ts <= pn->last_ts) {
            log_barrier.info("adjustment can be applied immediately: %llx/%d (%llx)",
                             ts, delta, pn->last_ts);
            unguarded_delta += delta;
          } else {
            log_barrier.info("adjustment must be deferred: %llx/%d (%llx)",
                             ts, delta, pn->last_ts);
            pn->pending[ts] += delta;
          }
        }
      }

    struct RemoteNotification {
      unsigned node;
      Event::gen_t trigger_gen, previous_gen;
    };

    // used to adjust a barrier's arrival count either up or down
    // if delta > 0, timestamp is current time (on requesting node)
    // if delta < 0, timestamp says which positive adjustment this arrival must wait for
    void BarrierImpl::adjust_arrival(Event::gen_t barrier_gen, int delta, 
				     Barrier::timestamp_t timestamp, Event wait_on,
				     const void *reduce_value, size_t reduce_value_size)
    {
      if(!wait_on.has_triggered()) {
	// deferred arrival
	Barrier b = me.convert<Barrier>();
	b.gen = barrier_gen;
	b.timestamp = timestamp;
#ifndef DEFER_ARRIVALS_LOCALLY
        if(owner != gasnet_mynode()) {
	  // let deferral happen on owner node (saves latency if wait_on event
          //   gets triggered there)
          //printf("sending deferred arrival to %d for " IDFMT "/%d (" IDFMT "/%d)\n",
          //       owner, e.id, e.gen, wait_on.id, wait_on.gen);
	  log_barrier.info("forwarding deferred barrier arrival: delta=%d in=" IDFMT "/%d out=" IDFMT "/%d (%llx)",
			   delta, wait_on.id, wait_on.gen, b.id, b.gen, b.timestamp);
	  BarrierAdjustMessage::send_request(owner, b, delta, wait_on, reduce_value, reduce_value_size);
	  return;
        }
#endif
	log_barrier.info("deferring barrier arrival: delta=%d in=" IDFMT "/%d out=" IDFMT "/%d (%llx)",
			 delta, wait_on.id, wait_on.gen, me.id(), barrier_gen, timestamp);
	EventImpl::add_waiter(wait_on, new DeferredBarrierArrival(b, delta, 
								  reduce_value, reduce_value_size));
	return;
      }

      log_barrier.info("barrier adjustment: event=" IDFMT "/%d delta=%d ts=%llx", 
		       me.id(), barrier_gen, delta, timestamp);

#ifdef DEBUG_BARRIER_REDUCTIONS
      if(reduce_value_size) {
        char buffer[129];
	for(size_t i = 0; (i < reduce_value_size) && (i < 64); i++)
	  sprintf(buffer+2*i, "%02x", ((const unsigned char *)reduce_value)[i]);
	log_barrier.info("barrier reduction: event=" IDFMT "/%d size=%zd data=%s",
	                 me.id(), barrier_gen, reduce_value_size, buffer);
      }
#endif

      if(owner != gasnet_mynode()) {
	// all adjustments handled by owner node
	Barrier b = me.convert<Barrier>();
	b.gen = barrier_gen;
	b.timestamp = timestamp;
	BarrierAdjustMessage::send_request(owner, b, delta, Event::NO_EVENT, reduce_value, reduce_value_size);
	return;
      }

      // can't actually trigger while holding the lock, so remember which generation(s),
      //  if any, to trigger and do it at the end
      Event::gen_t trigger_gen = 0;
      std::vector<EventWaiter *> local_notifications;
      std::vector<RemoteNotification> remote_notifications;
      Event::gen_t oldest_previous = 0;
      void *final_values_copy = 0;
      {
	AutoHSLLock a(mutex);

	// sanity checks - is this a valid barrier?
	assert(generation < free_generation);
	assert(base_arrival_count > 0);

	// update whatever generation we're told to
	{
	  assert(barrier_gen > generation);
	  Generation *g;
	  std::map<Event::gen_t, Generation *>::iterator it = generations.find(barrier_gen);
	  if(it != generations.end()) {
	    g = it->second;
	  } else {
	    g = new Generation;
	    generations[barrier_gen] = g;
	    log_barrier.info("added tracker for barrier " IDFMT ", generation %d",
			     me.id(), barrier_gen);
	  }

	  g->handle_adjustment(timestamp, delta);
	}

	// if the update was to the next generation, it may cause one or more generations
	//  to trigger
	if(barrier_gen == (generation + 1)) {
	  std::map<Event::gen_t, Generation *>::iterator it = generations.begin();
	  while((it != generations.end()) &&
		(it->first == (generation + 1)) &&
		((base_arrival_count + it->second->unguarded_delta) == 0)) {
	    // keep the list of local waiters to wake up once we release the lock
	    local_notifications.insert(local_notifications.end(), 
				       it->second->local_waiters.begin(), it->second->local_waiters.end());
	    trigger_gen = generation = it->first;
	    delete it->second;
	    generations.erase(it);
	    it = generations.begin();
	  }

	  // if any triggers occurred, figure out which remote nodes need notifications
	  //  (i.e. any who have subscribed)
	  if(generation >= barrier_gen) {
	    std::map<unsigned, Event::gen_t>::iterator it = remote_subscribe_gens.begin();
	    while(it != remote_subscribe_gens.end()) {
	      RemoteNotification rn;
	      rn.node = it->first;
	      if(it->second <= generation) {
		// we have fulfilled the entire subscription
		rn.trigger_gen = it->second;
		std::map<unsigned, Event::gen_t>::iterator to_nuke = it++;
		remote_subscribe_gens.erase(to_nuke);
	      } else {
		// subscription remains valid
		rn.trigger_gen = generation;
		it++;
	      }
	      // also figure out what the previous generation this node knew about was
	      {
		std::map<unsigned, Event::gen_t>::iterator it2 = remote_trigger_gens.find(rn.node);
		if(it2 != remote_trigger_gens.end()) {
		  rn.previous_gen = it2->second;
		  it2->second = rn.trigger_gen;
		} else {
		  rn.previous_gen = first_generation;
		  remote_trigger_gens[rn.node] = rn.trigger_gen;
		}
	      }
	      if(remote_notifications.empty() || (rn.previous_gen < oldest_previous))
		oldest_previous = rn.previous_gen;
	      remote_notifications.push_back(rn);
	    }
	  }
	}

	// do we have reduction data to apply?  we can do this even if the actual adjustment is
	//  being held - no need to have lots of reduce values lying around
	if(reduce_value_size > 0) {
	  assert(redop != 0);
	  assert(redop->sizeof_rhs == reduce_value_size);

	  // do we have space for this reduction result yet?
	  int rel_gen = barrier_gen - first_generation;
	  assert(rel_gen > 0);

	  if((size_t)rel_gen > value_capacity) {
	    size_t new_capacity = rel_gen;
	    final_values = (char *)realloc(final_values, new_capacity * redop->sizeof_lhs);
	    while(value_capacity < new_capacity) {
	      memcpy(final_values + (value_capacity * redop->sizeof_lhs), initial_value, redop->sizeof_lhs);
	      value_capacity += 1;
	    }
	  }

	  redop->apply(final_values + ((rel_gen - 1) * redop->sizeof_lhs), reduce_value, 1, true);
	}

	// do this AFTER we actually update the reduction value above :)
	// if any remote notifications are going to occur and we have reduction values, make a copy so
	//  we have something stable after we let go of the lock
	if(trigger_gen && redop) {
	  int rel_gen = oldest_previous + 1 - first_generation;
	  assert(rel_gen > 0);
	  int count = trigger_gen - oldest_previous;
	  final_values_copy = bytedup(final_values + ((rel_gen - 1) * redop->sizeof_lhs),
				      count * redop->sizeof_lhs);
	}
      }

      if(trigger_gen != 0) {
	log_barrier.info("barrier trigger: event=" IDFMT "/%d", 
			 me.id(), trigger_gen);

	// notify local waiters first
	for(std::vector<EventWaiter *>::const_iterator it = local_notifications.begin();
	    it != local_notifications.end();
	    it++) {
	  bool nuke = (*it)->event_triggered();
	  if(nuke)
	    delete (*it);
	}

	// now do remote notifications
	for(std::vector<RemoteNotification>::const_iterator it = remote_notifications.begin();
	    it != remote_notifications.end();
	    it++) {
	  log_barrier.info("sending remote trigger notification: " IDFMT "/%d -> %d, dest=%d",
			   me.id(), (*it).previous_gen, (*it).trigger_gen, (*it).node);
	  void *data = 0;
	  size_t datalen = 0;
	  if(final_values_copy) {
	    data = (char *)final_values_copy + (((*it).previous_gen - oldest_previous) * redop->sizeof_lhs);
	    datalen = ((*it).trigger_gen - (*it).previous_gen) * redop->sizeof_lhs;
	  }
	  BarrierTriggerMessage::send_request((*it).node, me.id(), (*it).trigger_gen, (*it).previous_gen,
					      first_generation, redop_id, data, datalen);
	}
      }

      // free our copy of the final values, if we had one
      if(final_values_copy)
	free(final_values_copy);
    }

    bool BarrierImpl::has_triggered(Event::gen_t needed_gen)
    {
      // no need to take lock to check current generation
      if(needed_gen <= generation) return true;

      // if we're not the owner, subscribe if we haven't already
      if(owner != gasnet_mynode()) {
	Event::gen_t previous_subscription;
	// take lock to avoid duplicate subscriptions
	{
	  AutoHSLLock a(mutex);
	  previous_subscription = gen_subscribed;
	  if(gen_subscribed < needed_gen)
	    gen_subscribed = needed_gen;
	}

	if(previous_subscription < needed_gen) {
	  log_barrier.info("subscribing to barrier " IDFMT "/%d", me.id(), needed_gen);
	  BarrierSubscribeMessage::send_request(owner, me.id(), needed_gen);
	}
      }

      // whether or not we subscribed, the answer for now is "no"
      return false;
    }

    void BarrierImpl::external_wait(Event::gen_t needed_gen)
    {
      assert(0);
    }

    bool BarrierImpl::add_waiter(Event::gen_t needed_gen, EventWaiter *waiter/*, bool pre_subscribed = false*/)
    {
      bool trigger_now = false;
      {
	AutoHSLLock a(mutex);

	if(needed_gen > generation) {
	  Generation *g;
	  std::map<Event::gen_t, Generation *>::iterator it = generations.find(needed_gen);
	  if(it != generations.end()) {
	    g = it->second;
	  } else {
	    g = new Generation;
	    generations[needed_gen] = g;
	    log_barrier.info("added tracker for barrier " IDFMT ", generation %d",
			     me.id(), needed_gen);
	  }
	  g->local_waiters.push_back(waiter);

	  // a call to has_triggered should have already handled the necessary subscription
	  assert((owner == gasnet_mynode()) || (gen_subscribed >= needed_gen));
	} else {
	  // needed generation has already occurred - trigger this waiter once we let go of lock
	  trigger_now = true;
	}
      }

      if(trigger_now) {
	bool nuke = waiter->event_triggered();
	if(nuke)
	  delete waiter;
      }

      return true;
    }

    /*static*/ void BarrierSubscribeMessage::handle_request(BarrierSubscribeMessage::RequestArgs args)
    {
      Barrier b;
      b.id = args.barrier_id;
      b.gen = args.subscribe_gen;
      BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

      // take the lock and add the subscribing node - notice if they need to be notified for
      //  any generations that have already triggered
      Event::gen_t trigger_gen = 0;
      Event::gen_t previous_gen = 0;
      void *final_values_copy = 0;
      size_t final_values_size = 0;
      {
	AutoHSLLock a(impl->mutex);

	// make sure the subscription is for this "lifetime" of the barrier
	assert(args.subscribe_gen > impl->first_generation);

	bool already_subscribed = false;
	{
	  std::map<unsigned, Event::gen_t>::iterator it = impl->remote_subscribe_gens.find(args.node);
	  if(it != impl->remote_subscribe_gens.end()) {
	    // a valid subscription should always be for a generation that hasn't
	    //  triggered yet
	    assert(it->second > impl->generation);
	    if(it->second >= args.subscribe_gen)
	      already_subscribed = true;
	    else
	      it->second = args.subscribe_gen;
	  } else {
	    // new subscription - don't reset remote_trigger_gens because the node may have
	    //  been subscribed in the past
	    // NOTE: remote_subscribe_gens should only hold subscriptions for
	    //  generations that haven't triggered, so if we're subscribing to 
	    //  an old generation, don't add it
	    if(args.subscribe_gen > impl->generation)
	      impl->remote_subscribe_gens[args.node] = args.subscribe_gen;
	  }
	}

	// as long as we're not already subscribed to this generation, check to see if
	//  any trigger notifications are needed
	if(!already_subscribed && (impl->generation > impl->first_generation)) {
	  std::map<unsigned, Event::gen_t>::iterator it = impl->remote_trigger_gens.find(args.node);
	  if((it == impl->remote_trigger_gens.end()) or (it->second < impl->generation)) {
	    previous_gen = ((it == impl->remote_trigger_gens.end()) ?
			      impl->first_generation :
			      it->second);
	    trigger_gen = impl->generation;
	    impl->remote_trigger_gens[args.node] = impl->generation;

	    if(impl->redop) {
	      int rel_gen = previous_gen + 1 - impl->first_generation;
	      assert(rel_gen > 0);
	      final_values_size = (trigger_gen - previous_gen) * impl->redop->sizeof_lhs;
	      final_values_copy = bytedup(impl->final_values + ((rel_gen - 1) * impl->redop->sizeof_lhs),
					  final_values_size);
	    }
	  }
	}
      }

      // send trigger message outside of lock, if needed
      if(trigger_gen > 0) {
	log_barrier.info("sending immediate barrier trigger: " IDFMT "/%d -> %d",
			 args.barrier_id, previous_gen, trigger_gen);
	BarrierTriggerMessage::send_request(args.node, args.barrier_id, trigger_gen, previous_gen,
					    impl->first_generation, impl->redop_id,
					    final_values_copy, final_values_size);
      }

      if(final_values_copy)
	free(final_values_copy);
    }

    /*static*/ void BarrierTriggerMessage::handle_request(BarrierTriggerMessage::RequestArgs args,
							  const void *data, size_t datalen)
    {
      log_barrier.info("received remote barrier trigger: " IDFMT "/%d -> %d",
		       args.barrier_id, args.previous_gen, args.trigger_gen);

      Barrier b;
      b.id = args.barrier_id;
      b.gen = args.trigger_gen;
      BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

      // we'll probably end up with a list of local waiters to notify
      std::vector<EventWaiter *> local_notifications;
      {
	AutoHSLLock a(impl->mutex);

	// it's theoretically possible for multiple trigger messages to arrive out
	//  of order, so check if this message triggers the oldest possible range
	if(args.previous_gen == impl->generation) {
	  // see if we can pick up any of the held triggers too
	  while(!impl->held_triggers.empty()) {
	    std::map<Event::gen_t, Event::gen_t>::iterator it = impl->held_triggers.begin();
	    // if it's not contiguous, we're done
	    if(it->first != args.trigger_gen) break;
	    // it is contiguous, so absorb it into this message and remove the held trigger
	    log_barrier.info("collapsing future trigger: " IDFMT "/%d -> %d -> %d",
			     args.barrier_id, args.previous_gen, args.trigger_gen, it->second);
	    args.trigger_gen = it->second;
	    impl->held_triggers.erase(it);
	  }

	  impl->generation = args.trigger_gen;

	  // now iterate through any generations up to and including the latest triggered
	  //  generation, and accumulate local waiters to notify
	  while(!impl->generations.empty()) {
	    std::map<Event::gen_t, BarrierImpl::Generation *>::iterator it = impl->generations.begin();
	    if(it->first > args.trigger_gen) break;

	    local_notifications.insert(local_notifications.end(),
				       it->second->local_waiters.begin(),
				       it->second->local_waiters.end());
	    delete it->second;
	    impl->generations.erase(it);
	  }
	} else {
	  // hold this trigger until we get messages for the earlier generation(s)
	  log_barrier.info("holding future trigger: " IDFMT "/%d (%d -> %d)",
			   args.barrier_id, impl->generation, 
			   args.previous_gen, args.trigger_gen);
	  impl->held_triggers[args.previous_gen] = args.trigger_gen;
	}

	// is there any data we need to store?
	if(datalen) {
	  assert(args.redop_id != 0);

	  // TODO: deal with invalidation of previous instance of a barrier
	  impl->redop_id = args.redop_id;
	  impl->redop = get_runtime()->reduce_op_table[args.redop_id];
	  impl->first_generation = args.first_generation;

	  int rel_gen = args.trigger_gen - impl->first_generation;
	  assert(rel_gen > 0);
	  if(impl->value_capacity < (size_t)rel_gen) {
	    size_t new_capacity = rel_gen;
	    impl->final_values = (char *)realloc(impl->final_values, new_capacity * impl->redop->sizeof_lhs);
	    // no need to initialize new entries - we'll overwrite them now or when data does show up
	    impl->value_capacity = new_capacity;
	  }
	  assert(datalen == (impl->redop->sizeof_lhs * (args.trigger_gen - args.previous_gen)));
	  memcpy(impl->final_values + ((rel_gen - 1) * impl->redop->sizeof_lhs), data, datalen);
	}
      }

      // with lock released, perform any local notifications
      for(std::vector<EventWaiter *>::const_iterator it = local_notifications.begin();
	  it != local_notifications.end();
	  it++) {
	bool nuke = (*it)->event_triggered();
	if(nuke)
	  delete (*it);
      }
    }

    bool BarrierImpl::get_result(Event::gen_t result_gen, void *value, size_t value_size)
    {
      // take the lock so we can safely see how many results (if any) are on hand
      AutoHSLLock al(mutex);

      // generation hasn't triggered yet?
      if(result_gen > generation) return false;

      // if it has triggered, we should have the data
      int rel_gen = result_gen - first_generation;
      assert(rel_gen > 0);
      assert((size_t)rel_gen <= value_capacity);

      assert(redop != 0);
      assert(value_size == redop->sizeof_lhs);
      assert(value != 0);
      memcpy(value, final_values + ((rel_gen - 1) * redop->sizeof_lhs), redop->sizeof_lhs);
      return true;
    }

}; // namespace Realm
