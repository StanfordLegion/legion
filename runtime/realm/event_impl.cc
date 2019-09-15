/* Copyright 2019 Stanford University, NVIDIA Corporation
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

#include "realm/event_impl.h"

#include "realm/proc_impl.h"
#include "realm/runtime_impl.h"
#include "realm/logging.h"
#include "realm/threads.h"
#include "realm/profiling.h"

namespace Realm {

  Logger log_event("event");
  Logger log_barrier("barrier");
  Logger log_poison("poison");

  // used in places that don't currently propagate poison but should
  static const bool POISON_FIXME = false;

  // turn nested event triggers into a list walk instead - keeps from blowing
  //  out the stack
  namespace ThreadLocal {
    __thread EventWaiter::EventWaiterList *nested_wake_list = 0;
  };

#if 0
  ////////////////////////////////////////////////////////////////////////
  //
  // class DeferredEventTrigger
  //

  class DeferredEventTrigger : public EventWaiter {
 public:
    DeferredEventTrigger(Event _after_event);

    virtual ~DeferredEventTrigger(void);
    
    virtual bool event_triggered(Event e, bool poisoned);

    virtual void print(std::ostream& os) const;

    virtual Event get_finish_event(void) const;

  protected:
    Event after_event;
  };
  
  DeferredEventTrigger::DeferredEventTrigger(Event _after_event)
    : after_event(_after_event)
  {}

  DeferredEventTrigger::~DeferredEventTrigger(void) { }

  bool DeferredEventTrigger::event_triggered(Event e, bool poisoned)
  {
    if(poisoned) {
      log_poison.info() << "poisoned deferred event: event=" << after_event;
      GenEventImpl::trigger(after_event, true /*poisoned*/);
      return true;
    }
    
    log_event.info() << "deferred trigger occuring: " << after_event;
    GenEventImpl::trigger(after_event, false /*!poisoned*/);
    return true;
  }

  void DeferredEventTrigger::print(std::ostream& os) const
  {
    os << "deferred trigger: after=" << after_event;
  }

  Event DeferredEventTrigger::get_finish_event(void) const
  {
    return after_event;
  }
#endif

  ////////////////////////////////////////////////////////////////////////
  //
  // class Event
  //

  /*static*/ const Event Event::NO_EVENT = { 0 };
  // Take this you POS c++ type system
  /* static */ const UserEvent UserEvent::NO_USER_EVENT = 
    *(static_cast<UserEvent*>(const_cast<Event*>(&Event::NO_EVENT)));

  bool Event::has_triggered(void) const
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    if(!id) return true; // special case: NO_EVENT has always triggered
    EventImpl *e = get_runtime()->get_event_impl(*this);
    bool poisoned = false;
    if(e->has_triggered(ID(id).event_generation(), poisoned)) {
      // a poisoned event causes an exception because the caller isn't prepared for it
      if(poisoned) {
#ifdef REALM_USE_EXCEPTIONS
	if(Thread::self()->exceptions_permitted()) {
	  throw PoisonedEventException(*this);
	} else
#endif
	  {
	    log_poison.fatal() << "FATAL: no handler for test of poisoned event " << *this;
	    assert(0);
	  }
      }
      return true;
    } else
      return false;
  }

  bool Event::has_triggered_faultaware(bool& poisoned) const
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    if(!id) return true; // special case: NO_EVENT has always triggered
    EventImpl *e = get_runtime()->get_event_impl(*this);
    return e->has_triggered(ID(id).event_generation(), poisoned);
  }

  // creates an event that won't trigger until all input events have
  /*static*/ Event Event::merge_events(const std::set<Event>& wait_for)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    return GenEventImpl::merge_events(wait_for, false /*!ignore faults*/);
  }

  /*static*/ Event Event::merge_events(const std::vector<Event>& wait_for)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    return GenEventImpl::merge_events(wait_for, false /*!ignore faults*/);
  }

  /*static*/ Event Event::merge_events(Event ev1, Event ev2,
				       Event ev3 /*= NO_EVENT*/, Event ev4 /*= NO_EVENT*/,
				       Event ev5 /*= NO_EVENT*/, Event ev6 /*= NO_EVENT*/)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    return GenEventImpl::merge_events(ev1, ev2, ev3, ev4, ev5, ev6);
  }

  /*static*/ Event Event::merge_events_ignorefaults(const std::set<Event>& wait_for)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    return GenEventImpl::merge_events(wait_for, true /*ignore faults*/);
  }

  /*static*/ Event Event::merge_events_ignorefaults(const std::vector<Event>& wait_for)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    return GenEventImpl::merge_events(wait_for, true /*ignore faults*/);
  }

  /*static*/ Event Event::ignorefaults(Event wait_for)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    return GenEventImpl::ignorefaults(wait_for);
  }

  class EventTriggeredCondition {
  public:
    EventTriggeredCondition(EventImpl* _event, EventImpl::gen_t _gen, 
			    ProfilingMeasurements::OperationEventWaits::WaitInterval *_interval);

    class Callback : public EventWaiter {
    public:
      Callback(const EventTriggeredCondition& _cond);
      virtual ~Callback(void);
      virtual void event_triggered(bool poisoned);
      virtual void print(std::ostream& os) const;
      virtual Event get_finish_event(void) const;
      virtual void operator()(bool poisoned) = 0;

    protected:
      const EventTriggeredCondition& cond;
#ifdef REALM_EVENT_WAITER_BACKTRACE
      mutable Backtrace backtrace;
#endif
    };

    void add_callback(Callback& cb) const;

  protected:
    EventImpl *event;
    EventImpl::gen_t gen;
    ProfilingMeasurements::OperationEventWaits::WaitInterval *interval;
  };

  EventTriggeredCondition::EventTriggeredCondition(EventImpl* _event, EventImpl::gen_t _gen, 
						   ProfilingMeasurements::OperationEventWaits::WaitInterval *_interval)
    : event(_event), gen(_gen), interval(_interval)
  {}

  void EventTriggeredCondition::add_callback(Callback& cb) const
  {
    event->add_waiter(gen, &cb);
  }

  EventTriggeredCondition::Callback::Callback(const EventTriggeredCondition& _cond)
    : cond(_cond)
  {
#ifdef REALM_EVENT_WAITER_BACKTRACE
    backtrace.capture_backtrace();
#endif
  }
  
  EventTriggeredCondition::Callback::~Callback(void)
  {
  }
  
  void EventTriggeredCondition::Callback::event_triggered(bool poisoned)
  {
    if(cond.interval)
      cond.interval->record_wait_ready();
    (*this)(poisoned);
  }

  void EventTriggeredCondition::Callback::print(std::ostream& os) const
  {
#ifdef REALM_EVENT_WAITER_BACKTRACE
    backtrace.lookup_symbols();
    os << "EventTriggeredCondition (backtrace=" << backtrace << ")";
#else
    os << "EventTriggeredCondition (thread unknown)";
#endif
  }  

  Event EventTriggeredCondition::Callback::get_finish_event(void) const
  {
    return Event::NO_EVENT;  // ideally would be the finish event of the task being suspended
  }

  void Event::wait(void) const
  {
    bool poisoned = false;
    wait_faultaware(poisoned);
    // a poisoned event causes an exception because the caller isn't prepared for it
    if(poisoned) {
#ifdef REALM_USE_EXCEPTIONS
      if(Thread::self()->exceptions_permitted()) {
	throw PoisonedEventException(*this);
      } else
#endif
      {
	log_poison.fatal() << "FATAL: no handler for test of poisoned event " << *this;
	assert(0);
      }
    }
  }

  void Event::wait_faultaware(bool& poisoned) const
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    if(!id) return;  // special case: never wait for NO_EVENT
    EventImpl *e = get_runtime()->get_event_impl(*this);
    EventImpl::gen_t gen = ID(id).event_generation();

    // early out case too
    if(e->has_triggered(gen, poisoned)) return;

    // waiting on an event does not count against the low level's time
    DetailedTimer::ScopedPush sp2(TIME_NONE);

    Thread *thread = Thread::self();
    if(thread) {
      log_event.info() << "thread blocked: thread=" << thread << " event=" << *this;
      // see if we are being asked to profile these waits
      ProfilingMeasurements::OperationEventWaits::WaitInterval *interval = 0;
      if(thread->get_operation() != 0) {
	interval = thread->get_operation()->create_wait_interval(*this);
	if(interval)
	  interval->record_wait_start();
      }
      // describe the condition we want the thread to wait on
      thread->wait_for_condition(EventTriggeredCondition(e, gen, interval), poisoned);
      if(interval)
	interval->record_wait_end();
      log_event.info() << "thread resumed: thread=" << thread << " event=" << *this << " poisoned=" << poisoned;
      return;
    }

    assert(0); // if we're not a Thread, we have a problem
    return;
    //assert(ptr != 0);
  }

  void Event::external_wait(void) const
  {
    bool poisoned = false;
    external_wait_faultaware(poisoned);
    // a poisoned event causes an exception because the caller isn't prepared for it
    if(poisoned) {
#ifdef REALM_USE_EXCEPTIONS
      if(Thread::self()->exceptions_permitted()) {
	throw PoisonedEventException(*this);
      } else
#endif
      {
	log_poison.fatal() << "FATAL: no handler for test of poisoned event " << *this;
	assert(0);
      }
    }
  }

  void Event::external_wait_faultaware(bool& poisoned) const
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    if(!id) return;  // special case: never wait for NO_EVENT
    EventImpl *e = get_runtime()->get_event_impl(*this);
    EventImpl::gen_t gen = ID(id).event_generation();

    // early out case too
    if(e->has_triggered(gen, poisoned)) return;
    
    // waiting on an event does not count against the low level's time
    DetailedTimer::ScopedPush sp2(TIME_NONE);
    
    log_event.info() << "external thread blocked: event=" << *this;
    e->external_wait(gen, poisoned);
    log_event.info() << "external thread resumed: event=" << *this;
  }

  bool Event::external_timedwait(long long max_ns) const
  {
    bool poisoned = false;
    bool triggered = external_timedwait_faultaware(poisoned, max_ns);
    if(!triggered)
      return false;

    // a poisoned event causes an exception because the caller isn't prepared for it
    if(poisoned) {
#ifdef REALM_USE_EXCEPTIONS
      if(Thread::self()->exceptions_permitted()) {
	throw PoisonedEventException(*this);
      } else
#endif
      {
	log_poison.fatal() << "FATAL: no handler for test of poisoned event " << *this;
	assert(0);
      }
    }

    return true;
  }

  bool Event::external_timedwait_faultaware(bool& poisoned,
					    long long max_ns) const
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
    if(!id) return true;  // special case: never wait for NO_EVENT
    EventImpl *e = get_runtime()->get_event_impl(*this);
    EventImpl::gen_t gen = ID(id).event_generation();

    // early out case too
    if(e->has_triggered(gen, poisoned)) return true;

    // waiting on an event does not count against the low level's time
    DetailedTimer::ScopedPush sp2(TIME_NONE);

    log_event.info() << "external thread blocked: event=" << *this;
    bool triggered = e->external_timedwait(gen, poisoned, max_ns);
    log_event.info() << "external thread resumed: event=" << *this
		     << (triggered ? "" : " (timeout)");
    return triggered;
  }

  void Event::cancel_operation(const void *reason_data, size_t reason_len) const
  {
    get_runtime()->optable.request_cancellation(*this, reason_data, reason_len);
  }

  void Event::set_operation_priority(int new_priority) const
  {
    get_runtime()->optable.set_priority(*this, new_priority);
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
    log_event.info() << "user event created: event=" << e;
    return u;
  }

  namespace Config {
    // if non-zero, eagerly checks deferred user event triggers for loops up to the
    //  specified limit
    int event_loop_detection_limit = 0;
  };

  void UserEvent::trigger(Event wait_on, bool ignore_faults) const
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

#ifdef EVENT_GRAPH_TRACE
    Event enclosing = find_enclosing_termination_event();
    log_event_graph.info("Event Trigger: (" IDFMT ",%d) (" IDFMT 
			 ",%d) (" IDFMT ",%d)",
			 id, gen, wait_on.id, wait_on.gen,
			 enclosing.id, enclosing.gen);
#endif

    bool poisoned = false;
    if(wait_on.has_triggered_faultaware(poisoned)) {
      log_event.info() << "user event trigger: event=" << *this << " wait_on=" << wait_on
		       << (poisoned ? " (poisoned)" : "");
      GenEventImpl::trigger(*this, poisoned && !ignore_faults);
    } else {
      // deferred trigger
      log_event.info() << "deferring user event trigger: event=" << *this << " wait_on=" << wait_on;
      if(Config::event_loop_detection_limit > 0) {
	// before we add the deferred trigger as a waiter, see if this is causing a cycle in the event graph
	if(EventImpl::detect_event_chain(*this, wait_on, 
					 Config::event_loop_detection_limit,
					 true /*print chain*/)) {
	  log_event.fatal() << "deferred trigger creates event loop!  event=" << *this << " wait_on=" << wait_on;
	  assert(0);
	}
      }

      // use the event's merger to wait for this precondition
      GenEventImpl *event_impl = get_genevent_impl(*this);
      event_impl->merger.prepare_merger(*this, ignore_faults, 1);
      event_impl->merger.add_precondition(wait_on);
      event_impl->merger.arm_merger();
    }
  }

  void UserEvent::cancel(void) const
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

#ifdef EVENT_GRAPH_TRACE
    // TODO: record cancellation?
    Event enclosing = find_enclosing_termination_event();
    log_event_graph.info("Event Trigger: (" IDFMT ",%d) (" IDFMT 
			 ",%d) (" IDFMT ",%d)",
			 id, gen, wait_on.id, wait_on.gen,
			 enclosing.id, enclosing.gen);
#endif

    log_event.info() << "user event cancelled: event=" << *this;
    GenEventImpl::trigger(*this, true /*poisoned*/);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Barrier
  //

  namespace {
    Barrier make_no_barrier(void)
    {
      Barrier b;
      b.id = 0;
      b.timestamp = 0;
      return b;
    }
  };

  /*static*/ const Barrier Barrier::NO_BARRIER = make_no_barrier();

  /*static*/ Barrier Barrier::create_barrier(unsigned expected_arrivals,
					     ReductionOpID redop_id /*= 0*/,
					     const void *initial_value /*= 0*/,
					     size_t initial_value_size /*= 0*/)
  {
    DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);

    BarrierImpl *impl = BarrierImpl::create_barrier(expected_arrivals, redop_id, initial_value, initial_value_size);
    Barrier b = impl->current_barrier();

#ifdef EVENT_GRAPH_TRACE
    log_event_graph.info("Barrier Creation: " IDFMT " %d", b.id, expected_arrivals);
#endif

    return b;
  }

  void Barrier::destroy_barrier(void)
  {
    log_barrier.info() << "barrier destruction request: " << *this;
  }

  Barrier Barrier::advance_barrier(void) const
  {
    ID nextid(id);
    EventImpl::gen_t gen = ID(id).barrier_generation() + 1;
    // return NO_BARRIER if the count overflows
    if(gen > nextid.barrier_generation().MAXVAL)
      return Barrier::NO_BARRIER;
    nextid.barrier_generation() = ID(id).barrier_generation() + 1;

    Barrier nextgen = nextid.convert<Barrier>();
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
    impl->adjust_arrival(ID(id).barrier_generation(), delta, timestamp, Event::NO_EVENT,
			 my_node_id, false /*!forwarded*/,
			 0, 0);

    Barrier with_ts;
    with_ts.id = id;
    with_ts.timestamp = timestamp;

    return with_ts;
  }

  Barrier Barrier::get_previous_phase(void) const
  {
    ID previd(id);
    EventImpl::gen_t gen = ID(id).barrier_generation();
    // can't back up before generation 0
    previd.barrier_generation() = ((gen > 0) ? (gen - 1) : gen);

    Barrier prevgen = previd.convert<Barrier>();
    prevgen.timestamp = 0;

    return prevgen;
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
    impl->adjust_arrival(ID(id).barrier_generation(), -count, timestamp, wait_on,
			 my_node_id, false /*!forwarded*/,
			 reduce_value, reduce_value_size);
  }

  bool Barrier::get_result(void *value, size_t value_size) const
  {
    BarrierImpl *impl = get_runtime()->get_barrier_impl(*this);
    return impl->get_result(ID(id).barrier_generation(), value, value_size);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class EventImpl
  //

  /*static*/ bool EventImpl::detect_event_chain(Event search_from, Event target,
						int max_depth, bool print_chain)
  {
    std::vector<Event> events;
    std::vector<EventWaiter *> waiters;
    std::set<Event> events_seen;
    std::vector<EventWaiter *> todo;

    events.reserve(max_depth + 1);
    waiters.reserve(max_depth + 1);

    events.push_back(search_from);
    waiters.push_back(0);

    Event e = search_from;
    while(true) {
      EventWaiter *waiters_head = 0;

      ID id(e);
      if(id.is_event()) {
	GenEventImpl *impl = get_runtime()->get_genevent_impl(e);

	{
	  AutoHSLLock al(impl->mutex);
	  gen_t gen = impl->generation.load();
	  if(gen >= id.event_generation()) {
	    // already triggered!?
	    assert(0);
	  } else if((gen + 1) == id.event_generation()) {
	    // current generation
	    waiters_head = impl->current_local_waiters.head.next;
	  } else {
	    std::map<EventImpl::gen_t, EventWaiter::EventWaiterList>::const_iterator it = impl->future_local_waiters.find(id.event_generation());
	    if(it != impl->future_local_waiters.end())
	      waiters_head = it->second.head.next;
	  }
	}
      } else if(id.is_barrier()) {
	assert(0);
	break;
      } else {
	assert(0);
      }

      // record all of these event waiters as seen before traversing, so that we find the
      //  shortest possible path
      int count = 0;
      for(EventWaiter *pos = waiters_head; pos; pos = pos->ew_list_link.next) {
	Event e2 = pos->get_finish_event();
	if(!e2.exists()) continue;
	if(e2 == target) {
	  if(print_chain) {
	    LoggerMessage msg(log_event.error());
	    if(msg.is_active()) {
	      msg << "event chain found!";
	      events.push_back(e2);
	      waiters.push_back(pos);
	      for(size_t i = 0; i < events.size(); i++) {
		msg << "\n  " << events[i];
		if(waiters[i]) {
		  msg << ": ";
		  waiters[i]->print(msg.get_stream());
		}
	      }
	    }
	  }
	  return true;
	}
	// don't search past the requested maximum depth
	if(events.size() > (size_t)max_depth)
	  continue;
	bool inserted = events_seen.insert(e2).second;
	if(inserted) {
	  if(count++ == 0)
	    todo.push_back(0); // marker so we know when to "pop" the stack
	  todo.push_back(pos);
	}
      }

      if(count == 0) {
	events.pop_back();
	waiters.pop_back();
      }

      // get next waiter
      EventWaiter *w = 0;
      while(!todo.empty()) {
	w = todo.back();
	todo.pop_back();
	if(w) break;
	assert(!events.empty());
	events.pop_back();
	waiters.pop_back();
      }
      if(!w) break;
      e = w->get_finish_event();
      events.push_back(e);
      waiters.push_back(w);
    }
    return false;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class EventMerger::MergeEventPrecondition
  //

  void EventMerger::MergeEventPrecondition::event_triggered(bool poisoned)
  {
    merger->precondition_triggered(poisoned);
  }

  void EventMerger::MergeEventPrecondition::print(std::ostream& os) const
  {
    os << "event merger: " << get_finish_event()
       << " left=" << merger->count_needed.load();
  }

  Event EventMerger::MergeEventPrecondition::get_finish_event(void) const
  {
    return merger->event_impl->make_event(merger->finish_gen);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class EventMerger
  //

#if 0
    class EventMerger {
    public:
      EventMerger(GenEventImpl *_event_impl);
      ~EventMerger(void);

      bool is_active(void) const;

      void prepare_merger(Event _finish_event, bool _ignore_faults, unsigned _max_preconditions);

      void add_precondition(Event wait_for);

      void arm_merger(void);

    protected:
      void precondition_triggered(bool poisoned);

      friend class MergeEventPrecondition;

      class MergeEventPrecondition : public EventWaiter {
      public:
	EventMerger *merger;

	virtual void event_triggered(bool poisoned)
	{
	  merger->precondition_triggered(poisoned);
	}

	virtual void print(std::ostream& os) const
	{
	  os << "event merger: " << merger->finish_event 
	     << " left=" << merger->count_needed;
	}

	virtual Event get_finish_event(void) const
	{
	  return merger->event_impl->make_event(merger->finish_gen);
	}
      };

#if 0
      virtual bool event_triggered(Event triggered, bool poisoned)
      {
	// if the input is poisoned, we propagate that poison eagerly
	if(poisoned) {
	  bool first_fault = (__sync_fetch_and_add(&faults_observed, 1) == 0);
	  if(first_fault && !ignore_faults) {
	    log_poison.info() << "event merger poisoned: after=" << finish_event;
	    GenEventImpl::trigger(finish_event, true /*poisoned*/);
	  }
	}

	int count_left = __sync_fetch_and_add(&count_needed, -1);

        // Put the logging first to avoid segfaults
	log_event.debug() << "received trigger merged event=" << finish_event << " left=" << count_left << " poisoned=" << poisoned;

	// count is the value before the decrement, so it was 1, it's now 0
	bool last_trigger = (count_left == 1);

	// trigger on the last input event, unless we did an early poison propagation
	if(last_trigger && (ignore_faults || (faults_observed == 0))) {
	  GenEventImpl::trigger(finish_event, false /*!poisoned*/);
	}

        // caller can delete us if this was the last trigger
        return last_trigger;
      }
#endif

      GenEventImpl *event_impl;
      Event::gen_t finish_gen;
      bool ignore_faults;
      atomic<int> count_needed;
      int faults_observed;

      static const size_t MAX_INLINE_PRECONDITIONS = 6;
      MergeEventPrecondition inline_preconditions[MAX_INLINE_PRECONDITIONS];
      MergeEventPrecondition *preconditions;
      unsigned num_preconditions, max_preconditions;
    };
#endif

    EventMerger::EventMerger(GenEventImpl *_event_impl)
      : event_impl(_event_impl)
      , count_needed(0)
      , preconditions(inline_preconditions)
      , max_preconditions(MAX_INLINE_PRECONDITIONS)
    {
      for(unsigned i = 0; i < MAX_INLINE_PRECONDITIONS; i++)
	inline_preconditions[i].merger = this;
    }

    EventMerger::~EventMerger(void)
    {
      // TODO: put this assert back once we've got a better handle on
      //  shutdown race conditions
      //assert(!is_active());
      if(!is_active())
	if(max_preconditions > MAX_INLINE_PRECONDITIONS)
	  delete[] preconditions;
    }

    bool EventMerger::is_active(void) const
    {
      return(count_needed.load() != 0);
    }

    void EventMerger::prepare_merger(Event _finish_event, bool _ignore_faults, unsigned _max_preconditions)
    {
      assert(!is_active());
      finish_gen = ID(_finish_event).event_generation();
      assert(event_impl->make_event(finish_gen) == _finish_event);
      ignore_faults = _ignore_faults;
      count_needed.store(1);  // this matches the subsequent call to arm()
      faults_observed = 0;
      num_preconditions = 0;
      // resize the precondition array if needed
      if(_max_preconditions > max_preconditions) {
	if(max_preconditions > MAX_INLINE_PRECONDITIONS)
	  delete[] preconditions;
	max_preconditions = _max_preconditions;
	preconditions = new MergeEventPrecondition[max_preconditions];
	for(unsigned i = 0; i < max_preconditions; i++)
	  preconditions[i].merger = this;
      }
    }

    void EventMerger::add_precondition(Event wait_for)
    {
      assert(is_active());

      bool poisoned = false;
      if(wait_for.has_triggered_faultaware(poisoned)) {
	if(poisoned) {
	  // always count faults, but don't necessarily propagate
	  bool first_fault = (__sync_fetch_and_add(&faults_observed, 1) == 0);
	  if(first_fault && !ignore_faults) {
            log_poison.info() << "event merger early poison: after=" << event_impl->make_event(finish_gen);
	    event_impl->trigger(finish_gen, my_node_id, true /*poisoned*/);
	  }
	}
	// either way we return to the caller without updating the count_needed
	return;
      }

      // figure out which precondition slot we'll use
      assert(num_preconditions < max_preconditions);
      MergeEventPrecondition *p = &preconditions[num_preconditions++];

      // increment count first, then add the waiter
      count_needed.fetch_add(1);
      EventImpl::add_waiter(wait_for, p);
    }

    void EventMerger::arm_merger(void)
    {
      assert(is_active());
      precondition_triggered(false /*!poisoned*/);
    }

    void EventMerger::precondition_triggered(bool poisoned)
    {
      // if the input is poisoned, we propagate that poison eagerly
      if(poisoned) {
	bool first_fault = (__sync_fetch_and_add(&faults_observed, 1) == 0);
	if(first_fault && !ignore_faults) {
	  log_poison.info() << "event merger poisoned: after=" << event_impl->make_event(finish_gen);
	  event_impl->trigger(finish_gen, my_node_id, true /*poisoned*/);
	}
      }

      int count_left = count_needed.fetch_add(-1);

      // Put the logging first to avoid segfaults
      log_event.debug() << "received trigger merged event=" << event_impl->make_event(finish_gen)
			<< " left=" << count_left << " poisoned=" << poisoned;

      // count is the value before the decrement, so it was 1, it's now 0
      bool last_trigger = (count_left == 1);

      // trigger on the last input event, unless we did an early poison propagation
      if(last_trigger && (ignore_faults || (faults_observed == 0))) {
	event_impl->trigger(finish_gen, my_node_id, false /*!poisoned*/);
      }

      // if the event was triggered early due to poison, its insertion on 
      //  the free list is delayed until we know that the event merger is
      //  inactive (i.e. when last_trigger is true)
      if(last_trigger)
	event_impl->perform_delayed_free_list_insertion();
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class EventImpl
  //

  EventImpl::EventImpl(void)
    : me((ID::IDType)-1), owner(-1)
  {}

  EventImpl::~EventImpl(void)
  {}

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class GenEventImpl
  //

  GenEventImpl::GenEventImpl(void)
    : generation(0)
    , gen_subscribed(0)
    , num_poisoned_generations(0)
    , merger(this)
    , has_external_waiters(false)
    , external_waiter_condvar(mutex)
  {
    next_free = 0;
    poisoned_generations = 0;
    has_local_triggers = false;
    free_list_insertion_delayed = false;
  }

  GenEventImpl::~GenEventImpl(void)
  {
#ifdef DEBUG_REALM
    AutoHSLLock a(mutex);
    if(!current_local_waiters.empty() ||
       !future_local_waiters.empty() ||
       has_external_waiters ||
       !remote_waiters.empty()) {
      log_event.fatal() << "Event " << me << " destroyed with"
			<< (current_local_waiters.empty() ? "" : " current local waiters")
			<< (future_local_waiters.empty() ? "" : " current future waiters")
			<< (has_external_waiters ? " external waiters" : "")
			<< (remote_waiters.empty() ? "" : " remote waiters");
      while(!current_local_waiters.empty()) {
	EventWaiter *ew = current_local_waiters.pop_front();
	log_event.fatal() << "  waiting on " << make_event(generation.load()) << ": " << ew;
      }
      for(std::map<gen_t, EventWaiter::EventWaiterList>::iterator it = future_local_waiters.begin();
	  it != future_local_waiters.end();
	  ++it) {
	while(!it->second.empty()) {
	  EventWaiter *ew = it->second.pop_front();
	  log_event.fatal() << "  waiting on " << make_event(it->first) << ": " << ew;
	}
      }
    }
#endif
  }

  void GenEventImpl::init(ID _me, unsigned _init_owner)
  {
    me = _me;
    owner = _init_owner;
    generation.store(0);
    gen_subscribed = 0;
    next_free = 0;
    num_poisoned_generations.store(0);
    poisoned_generations = 0;
    has_local_triggers = false;
  }


    // creates an event that won't trigger until all input events have
    /*static*/ Event GenEventImpl::merge_events(const std::set<Event>& wait_for,
						bool ignore_faults)
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
	  it++) {
	bool poisoned = false;
	if((*it).has_triggered_faultaware(poisoned)) {
          if(poisoned) {
	    // if we're not ignoring faults, we need to propagate this fault, and can do
	    //  so by just returning this poisoned event
	    if(!ignore_faults) {
	      log_poison.info() << "merging events - " << (*it) << " already poisoned";
	      return *it;
	    }
          }
	} else {
	  if(!wait_count) first_wait = *it;
	  wait_count++;
	}
      }
      log_event.debug() << "merging events - at least " << wait_count << " not triggered";

      // Avoid these optimizations if we are doing event graph tracing
      // we also cannot return an input event directly in the (wait_count == 1) case
      //  if we're ignoring faults
#ifndef EVENT_GRAPH_TRACE
      // counts of 0 or 1 don't require any merging
      if(wait_count == 0) return Event::NO_EVENT;
      if((wait_count == 1) && !ignore_faults) return first_wait;
#else
      if((wait_for.size() == 1) && !ignore_faults)
        return *(wait_for.begin());
#endif
      // counts of 2+ require building a new event and a merger to trigger it
      GenEventImpl *event_impl = GenEventImpl::create_genevent();
      Event finish_event = event_impl->current_event();

      EventMerger *m = &(event_impl->merger);
      m->prepare_merger(finish_event, ignore_faults, wait_for.size());

#ifdef EVENT_GRAPH_TRACE
      log_event_graph.info("Event Merge: (" IDFMT ",%d) %ld", 
			   finish_event.id, finish_event.gen, wait_for.size());
#endif

      for(std::set<Event>::const_iterator it = wait_for.begin();
	  it != wait_for.end();
	  it++) {
	log_event.info() << "event merging: event=" << finish_event << " wait_on=" << *it;
	m->add_precondition(*it);
#ifdef EVENT_GRAPH_TRACE
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ",%d)",
                             finish_event.id, finish_event.gen,
                             it->id, it->gen);
#endif
      }

      // once they're all added - arm the thing (it might go off immediately)
      m->arm_merger();

      return finish_event;
    }

    // creates an event that won't trigger until all input events have
    /*static*/ Event GenEventImpl::merge_events(const std::vector<Event>& wait_for,
						bool ignore_faults)
    {
      if (wait_for.empty())
        return Event::NO_EVENT;
      // scan through events to see how many exist/haven't fired - we're
      //  interested in counts of 0, 1, or 2+ - also remember the first
      //  event we saw for the count==1 case
      int wait_count = 0;
      Event first_wait;
      for(std::vector<Event>::const_iterator it = wait_for.begin();
	  (it != wait_for.end()) && (wait_count < 2);
	  it++) {
	bool poisoned = false;
	if((*it).has_triggered_faultaware(poisoned)) {
          if(poisoned) {
	    // if we're not ignoring faults, we need to propagate this fault, and can do
	    //  so by just returning this poisoned event
	    if(!ignore_faults) {
	      log_poison.info() << "merging events - " << (*it) << " already poisoned";
	      return *it;
	    }
          }
	} else {
	  if(!wait_count) first_wait = *it;
	  wait_count++;
	}
      }
      log_event.debug() << "merging events - at least " << wait_count << " not triggered";

      // Avoid these optimizations if we are doing event graph tracing
      // we also cannot return an input event directly in the (wait_count == 1) case
      //  if we're ignoring faults
#ifndef EVENT_GRAPH_TRACE
      // counts of 0 or 1 don't require any merging
      if(wait_count == 0) return Event::NO_EVENT;
      if((wait_count == 1) && !ignore_faults) return first_wait;
#else
      if((wait_for.size() == 1) && !ignore_faults)
        return *(wait_for.begin());
#endif
      // counts of 2+ require building a new event and a merger to trigger it
      GenEventImpl *event_impl = GenEventImpl::create_genevent();
      Event finish_event = event_impl->current_event();
      EventMerger *m = &(event_impl->merger);

      m->prepare_merger(finish_event, ignore_faults, wait_for.size());

#ifdef EVENT_GRAPH_TRACE
      log_event_graph.info("Event Merge: (" IDFMT ",%d) %ld", 
			   finish_event.id, finish_event.gen, wait_for.size());
#endif

      for(std::vector<Event>::const_iterator it = wait_for.begin();
	  it != wait_for.end();
	  it++) {
	log_event.info() << "event merging: event=" << finish_event << " wait_on=" << *it;
	m->add_precondition(*it);
#ifdef EVENT_GRAPH_TRACE
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ",%d)",
                             finish_event.id, finish_event.gen,
                             it->id, it->gen);
#endif
      }

      // once they're all added - arm the thing (it might go off immediately)
      m->arm_merger();

      return finish_event;
    }

    /*static*/ Event GenEventImpl::ignorefaults(Event wait_for)
    {
      bool poisoned = false;
      // poisoned or not, we return no event if it is done
      if(wait_for.has_triggered_faultaware(poisoned))
        return Event::NO_EVENT;
      GenEventImpl *event_impl = GenEventImpl::create_genevent();
      Event finish_event = event_impl->current_event();
      EventMerger *m = &(event_impl->merger);
      m->prepare_merger(finish_event, true/*ignore faults*/, 1);
#ifdef EVENT_GRAPH_TRACE
      log_event_graph.info("Event Merge: (" IDFMT ",%d) 1", 
			   finish_event.id, finish_event.gen);
#endif
      log_event.info() << "event merging: event=" << finish_event 
                       << " wait_on=" << wait_for;
      m->add_precondition(wait_for);
#ifdef EVENT_GRAPH_TRACE
      log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ",%d)",
                           finish_event.id, finish_event.gen,
                           wait_for.id, wait_for.gen);
#endif
      m->arm_merger();
      return finish_event;
    }

    /*static*/ Event GenEventImpl::merge_events(Event ev1, Event ev2,
						Event ev3 /*= NO_EVENT*/, Event ev4 /*= NO_EVENT*/,
						Event ev5 /*= NO_EVENT*/, Event ev6 /*= NO_EVENT*/)
    {
      // scan through events to see how many exist/haven't fired - we're
      //  interested in counts of 0, 1, or 2+ - also remember the first
      //  event we saw for the count==1 case
      // any poison on input events is immediately propagated (by simply returning
      //  the poisoned input event)
      int wait_count = 0;
      Event first_wait;
#define CHECK_EVENT(ev) \
      do {		       \
	bool poisoned = false; \
	if((ev).has_triggered_faultaware(poisoned)) { \
	  if(poisoned) \
	    return(ev); \
	} else { \
	  first_wait = (ev); \
	  wait_count++; \
	} \
      } while(0)
      CHECK_EVENT(ev6);
      CHECK_EVENT(ev5);
      CHECK_EVENT(ev4);
      CHECK_EVENT(ev3);
      CHECK_EVENT(ev2);
      CHECK_EVENT(ev1);

      log_event.debug() << "merging events - at least " << wait_count << " not triggered";

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
      GenEventImpl *event_impl = GenEventImpl::create_genevent();
      Event finish_event = event_impl->current_event();
      EventMerger *m = &(event_impl->merger);
      m->prepare_merger(finish_event, false/*!ignore faults*/, 6);

      if(ev1.exists()) {
	log_event.info() << "event merging: event=" << finish_event << " wait_on=" << ev1;
	m->add_precondition(ev1);
      }
      if(ev2.exists()) {
	log_event.info() << "event merging: event=" << finish_event << " wait_on=" << ev2;
	m->add_precondition(ev2);
      }
      if(ev3.exists()) {
	log_event.info() << "event merging: event=" << finish_event << " wait_on=" << ev3;
	m->add_precondition(ev3);
      }
      if(ev4.exists()) {
	log_event.info() << "event merging: event=" << finish_event << " wait_on=" << ev4;
	m->add_precondition(ev4);
      }
      if(ev5.exists()) {
	log_event.info() << "event merging: event=" << finish_event << " wait_on=" << ev5;
	m->add_precondition(ev5);
      }
      if(ev6.exists()) {
	log_event.info() << "event merging: event=" << finish_event << " wait_on=" << ev6;
	m->add_precondition(ev6);
      }

#ifdef EVENT_GRAPH_TRACE
      log_event_graph.info("Event Merge: (" IDFMT ",%d) %d",
               finish_event->me.id(), finish_event->generation.load(), existential_count);
      if (ev1.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation.load(), ev1.id, ev1.gen);
      if (ev2.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation.load(), ev2.id, ev2.gen);
      if (ev3.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation.load(), ev3.id, ev3.gen);
      if (ev4.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation.load(), ev4.id, ev4.gen);
      if (ev5.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation.load(), ev5.id, ev5.gen);
      if (ev6.exists())
        log_event_graph.info("Event Precondition: (" IDFMT ",%d) (" IDFMT ", %d)",
            finish_event->me.id(), finish_event->generation.load(), ev6.id, ev6.gen);
#endif

      // once they're all added - arm the thing (it might go off immediately)
      m->arm_merger();

      return finish_event;
    }

    /*static*/ GenEventImpl *GenEventImpl::create_genevent(void)
    {
      GenEventImpl *impl = get_runtime()->local_event_free_list->alloc_entry();
      assert(impl);
      assert(ID(impl->me).is_event());

      log_event.spew() << "event created: event=" << impl->current_event();

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

    bool GenEventImpl::add_waiter(gen_t needed_gen, EventWaiter *waiter)
    {
#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = me->id;
        item.event_gen = needed_gen;
        item.action = EventTraceItem::ACT_WAIT;
      }
#endif
      // no early check here as the caller will generally have tried has_triggered()
      //  before allocating its EventWaiter object

      bool trigger_now = false;
      bool trigger_poisoned = false;

      int subscribe_owner = -1;
      gen_t previous_subscribe_gen = 0;
      {
	AutoHSLLock a(mutex);

	// three cases below

	if(needed_gen <= generation.load()) {
	  // 1) the event has triggered and any poison information is in the poisoned generation list
	  trigger_now = true; // actually do trigger outside of mutex
	  trigger_poisoned = is_generation_poisoned(needed_gen);
	} else {
	  std::map<gen_t, bool>::const_iterator it = local_triggers.find(needed_gen);
	  if(it != local_triggers.end()) {
	    // 2) we're not the owner node, but we've locally triggered this and have correct poison info
	    assert(owner != my_node_id);
	    trigger_now = true;
	    trigger_poisoned = it->second;
	  } else {
	    // 3) we don't know of a trigger of this event, so record the waiter and subscribe if needed
	    gen_t cur_gen = generation.load();
	    log_event.debug() << "event not ready: event=" << me << "/" << needed_gen
			      << " owner=" << owner << " gen=" << cur_gen << " subscr=" << gen_subscribed;

	    // is this for the "current" next generation?
	    if(needed_gen == (cur_gen + 1)) {
	      // yes, put in the current waiter list
	      current_local_waiters.push_back(waiter);
	    } else {
	      // no, put it in an appropriate future waiter list - only allowed for non-owners
	      assert(owner != my_node_id);
	      future_local_waiters[needed_gen].push_back(waiter);
	    }

	    // do we need to subscribe to this event?
	    if((owner != my_node_id) && (gen_subscribed < needed_gen)) {
	      previous_subscribe_gen = gen_subscribed;
	      gen_subscribed = needed_gen;
	      subscribe_owner = owner;
	    }
	  }
	}
      }

      if((subscribe_owner != -1)) {
	ActiveMessage<EventSubscribeMessage> amsg(owner);
	amsg->event = make_event(needed_gen);
	amsg->previous_subscribe_gen = previous_subscribe_gen;
	amsg.commit();
      }

      if(trigger_now)
	waiter->event_triggered(trigger_poisoned);

      return true;  // waiter is always either enqueued or triggered right now
    }

    inline bool GenEventImpl::is_generation_poisoned(gen_t gen) const
    {
      // common case: no poisoned generations
      int npg_cached = num_poisoned_generations.load_acquire();
      if(__builtin_expect((npg_cached == 0), 1))
	return false;
      
      for(int i = 0; i < npg_cached; i++)
	if(poisoned_generations[i] == gen)
	  return true;
      return false;
    }


    ///////////////////////////////////////////////////
    // Events


  template <typename T>
  struct MediumBroadcastHelper : public T::RequestArgs {
    inline void apply(NodeID target)
    {
      T::Message::request(target, *this, payload, payload_size, payload_mode);
    }

    void broadcast(const NodeSet& targets,
		   const void *_payload, size_t _payload_size,
		   int _payload_mode)
    {
      payload = _payload;
      payload_size = _payload_size;
      payload_mode = _payload_mode;
      assert((payload_mode != PAYLOAD_FREE) && "cannot use PAYLOAD_FREE with broadcast!");
      targets.map(*this);
    }

    const void *payload;
    size_t payload_size;
    int payload_mode;
  };
  
    // only called for generational events
    /*static*/ void EventSubscribeMessage::handle_message(NodeID sender, const EventSubscribeMessage &args,
							  const void *data, size_t datalen)
    {
      log_event.debug() << "event subscription: node=" << sender << " event=" << args.event;

      GenEventImpl *impl = get_runtime()->get_genevent_impl(args.event);

#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = args.event.id; 
        item.event_gen = args.event.gen;
        item.action = EventTraceItem::ACT_WAIT;
      }
#endif

      // we may send a trigger message in response to the subscription
      EventImpl::gen_t subscribe_gen = ID(args.event).event_generation();
      EventImpl::gen_t trigger_gen = 0;
      bool subscription_recorded = false;

      // early-out case: if we can see the generation needed has already
      //  triggered, signal without taking the mutex
      EventImpl::gen_t stale_gen = impl->generation.load_acquire();
      if(stale_gen >= subscribe_gen) {
	trigger_gen = stale_gen;
      } else {
	AutoHSLLock a(impl->mutex);

	// look at the previously-subscribed generation from the requestor - we'll send
	//  a trigger message if anything newer has triggered
	EventImpl::gen_t cur_gen = impl->generation.load();
        if(cur_gen > args.previous_subscribe_gen)
	  trigger_gen = cur_gen;

	// are they subscribing to the current generation?
	if(subscribe_gen == (cur_gen + 1)) {
	  impl->remote_waiters.add(sender);
	  subscription_recorded = true;
	} else {
	  // should never get subscriptions newer than our current
	  assert(subscribe_gen <= cur_gen);
	}
      }

      if(subscription_recorded)
	log_event.debug() << "event subscription recorded: node=" << sender
			  << " event=" << args.event << " (> " << stale_gen << ")";

      if(trigger_gen > 0) {
	log_event.debug() << "event subscription immediate trigger: node=" << sender
			  << " event=" << args.event << " (<= " << trigger_gen << ")";
	ID trig_id(args.event);
	trig_id.event_generation() = trigger_gen;
	Event triggered = trig_id.convert<Event>();

	// it is legal to use poisoned generation info like this because it is
	// always updated before the generation - the load_acquire above makes
	// sure we read in the correct order
	int npg_cached = impl->num_poisoned_generations.load_acquire();
	ActiveMessage<EventUpdateMessage> amsg(sender, npg_cached*sizeof(EventImpl::gen_t));
	amsg->event = triggered;
	amsg.add_payload(impl->poisoned_generations, npg_cached*sizeof(EventImpl::gen_t), PAYLOAD_KEEP);
	amsg.commit();
      }
    } 

    /*static*/ void EventTriggerMessage::handle_message(NodeID sender, const EventTriggerMessage &args,
							const void *data, size_t datalen)
    {
      DetailedTimer::ScopedPush sp(TIME_LOW_LEVEL);
      log_event.debug() << "remote trigger of event " << args.event << " from node " << sender;
      GenEventImpl *impl = get_runtime()->get_genevent_impl(args.event);
      impl->trigger(ID(args.event).event_generation(), sender, args.poisoned);
    }

  template <typename T>
  struct ArrayOstreamHelper {
    ArrayOstreamHelper(const T *_base, size_t _count)
      : base(_base), count(_count)
    {}

    const T *base;
    size_t count;
  };

  template <typename T>
  std::ostream& operator<<(std::ostream& os, const ArrayOstreamHelper<T>& h)
  {
    switch(h.count) {
    case 0: return os << "0:{}";
    case 1: return os << "1:{ " << h.base[0] << " }";
    default:
      os << h.count << ":{ " << h.base[0];
      for(size_t i = 1; i < h.count; i++)
	os << ", " << h.base[i];
      os << " }";
      return os;
    }
  }


  void GenEventImpl::process_update(gen_t current_gen,
				    const gen_t *new_poisoned_generations,
				    int new_poisoned_count)
  {
    // this event had better not belong to us...
    assert(owner != my_node_id);

    // the result of the update may trigger multiple generations worth of waiters - keep their
    //  generation IDs straight (we'll look up the poison bits later)
    std::map<gen_t, EventWaiter::EventWaiterList> to_wake;

    {
      AutoHSLLock a(mutex);

#define CHECK_POISONED_GENS
#ifdef CHECK_POISONED_GENS
      if(new_poisoned_count > 0) {
	// should be an incremental update
	int old_npg = num_poisoned_generations.load();
	assert(old_npg <= new_poisoned_count);
	if(old_npg > 0)
	  assert(memcmp(poisoned_generations, new_poisoned_generations, 
			old_npg * sizeof(gen_t)) == 0);
      } else {
	// we shouldn't have any local ones either
	assert(num_poisoned_generations.load() == 0);
      }
#endif

      // this might be old news if we had subscribed to an event and then triggered it ourselves
      if(current_gen <= generation.load())
	return;

      // first thing - update the poisoned generation list
      if(new_poisoned_count > 0) {
	if(!poisoned_generations)
	  poisoned_generations = new gen_t[POISONED_GENERATION_LIMIT];
	if(num_poisoned_generations.load() < new_poisoned_count) {
	  assert(new_poisoned_count <= POISONED_GENERATION_LIMIT);
	  memcpy(poisoned_generations, new_poisoned_generations,
		 new_poisoned_count * sizeof(gen_t));
	  num_poisoned_generations.store_release(new_poisoned_count);
	}
      }

      // grab any/all waiters - start with current generation
      if(!current_local_waiters.empty())
	to_wake[generation.load() + 1].swap(current_local_waiters);

      // now any future waiters up to and including the triggered gen
      if(!future_local_waiters.empty()) {
	std::map<gen_t, EventWaiter::EventWaiterList>::iterator it = future_local_waiters.begin();
	while((it != future_local_waiters.end()) && (it->first <= current_gen)) {
	  to_wake[it->first].swap(it->second);
	  future_local_waiters.erase(it);
	  it = future_local_waiters.begin();
	}

	// and see if there's a future list that's now current
	if((it != future_local_waiters.end()) && (it->first == (current_gen + 1))) {
	  current_local_waiters.swap(it->second);
	  future_local_waiters.erase(it);
	}
      }

      // next, clear out any local triggers that have been ack'd
      if(has_local_triggers) {
	std::map<gen_t, bool>::iterator it = local_triggers.begin();
	while((it != local_triggers.end()) && (it->first <= current_gen)) {
	  assert(it->second == is_generation_poisoned(it->first));
	  local_triggers.erase(it);
	  it = local_triggers.begin();
	}
	has_local_triggers = !local_triggers.empty();
      }

      // finally, update the generation count, representing that we have complete information to that point
      generation.store_release(current_gen);

      // external waiters need to be signalled inside the lock
      if(has_external_waiters) {
	has_external_waiters = false;
	external_waiter_condvar.broadcast();
      }
    }

    // now trigger anybody that needs to be triggered
    if(!to_wake.empty()) {
      for(std::map<gen_t, EventWaiter::EventWaiterList>::iterator it = to_wake.begin();
	  it != to_wake.end();
	  it++) {
	bool poisoned = is_generation_poisoned(it->first);
	if(!poisoned) {
	  if(ThreadLocal::nested_wake_list != 0) {
	    // append our waiters for caller to handle rather than recursing
	    ThreadLocal::nested_wake_list->absorb_append(it->second);
	  } else {
	    ThreadLocal::nested_wake_list = &(it->second);  // avoid recursion
	    while(!it->second.empty()) {
	      EventWaiter *w = it->second.pop_front();
	      w->event_triggered(false /*!poisoned*/);
	    }
	    ThreadLocal::nested_wake_list = 0;
	  }
	} else {
	  while(!it->second.empty()) {
	    EventWaiter *w = it->second.pop_front();
	    w->event_triggered(true /*poisoned*/);
	  }
	}
      }
    }
  }

    /*static*/ void EventUpdateMessage::handle_message(NodeID sender, const EventUpdateMessage &args,
						       const void *data, size_t datalen)
    {
      const EventImpl::gen_t *new_poisoned_gens = (const EventImpl::gen_t *)data;
      int new_poisoned_count = datalen / sizeof(EventImpl::gen_t);
      assert((new_poisoned_count * sizeof(EventImpl::gen_t)) == datalen);  // no remainders or overflow please

      log_event.debug() << "event update: event=" << args.event
			<< " poisoned=" << ArrayOstreamHelper<EventImpl::gen_t>(new_poisoned_gens, new_poisoned_count);

      GenEventImpl *impl = get_runtime()->get_genevent_impl(args.event);
      impl->process_update(ID(args.event).event_generation(), new_poisoned_gens, new_poisoned_count);
    }


  /*static*/ Barrier::timestamp_t BarrierImpl::barrier_adjustment_timestamp;



    bool GenEventImpl::has_triggered(gen_t needed_gen, bool& poisoned)
    {
#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = me.id;
        item.event_gen = needed_gen;
        item.action = EventTraceItem::ACT_QUERY;
      }
#endif
      // lock-free check
      if(needed_gen <= generation.load_acquire()) {
	// it is safe to call is_generation_poisoned after just a load_acquire
	poisoned = is_generation_poisoned(needed_gen);
	return true;
      }

      // if the above check fails, we have to see if we have performed any local triggers -
      // if not, we can internally-consistently say that the event hasn't triggered from our
      // perspective yet
      if(!has_local_triggers) {
	poisoned = false;
	return false;
      }

      // both easy cases failed, so take the lock that lets us see which local triggers exist
      // this prevents us from ever answering "no" on the current node if the trigger occurred here
      bool locally_triggered = false;
      poisoned = false;
      {
	AutoHSLLock a(mutex);

	std::map<gen_t, bool>::const_iterator it = local_triggers.find(needed_gen);
	if(it != local_triggers.end()) {
	  locally_triggered = true;
	  poisoned = it->second;
	}
      }
      return locally_triggered;
    }

    void GenEventImpl::external_wait(gen_t gen_needed, bool& poisoned)
    {
      {
	AutoHSLLock a(mutex);

	// wait until the generation has advanced far enough
	while(gen_needed > generation.load_acquire()) {
	  has_external_waiters = true;
	  external_waiter_condvar.wait();
	}

	poisoned = is_generation_poisoned(gen_needed);
      }
    }

    bool GenEventImpl::external_timedwait(gen_t gen_needed, bool& poisoned,
					  long long max_ns)
    {
      long long deadline = Clock::current_time_in_nanoseconds() + max_ns;
      {
	AutoHSLLock a(mutex);

	// wait until the generation has advanced far enough
	while(gen_needed > generation.load_acquire()) {
	  long long now = Clock::current_time_in_nanoseconds();
	  if(now >= deadline)
	    return false;  // trigger has not occurred
	  has_external_waiters = true;
	  // we don't actually care what timedwait returns - we'll recheck
	  //  the generation ourselves
	  external_waiter_condvar.timedwait(deadline - now);
	}

	poisoned = is_generation_poisoned(gen_needed);
      }
      return true;
    }

    void GenEventImpl::trigger(gen_t gen_triggered, int trigger_node, bool poisoned)
    {
      Event e = make_event(gen_triggered);
      log_event.debug() << "event triggered: event=" << e << " by node " << trigger_node
			<< " (poisoned=" << poisoned << ")";

#ifdef EVENT_TRACING
      {
        EventTraceItem &item = Tracer<EventTraceItem>::trace_item();
        item.event_id = me.id;
        item.event_gen = gen_triggered;
        item.action = EventTraceItem::ACT_TRIGGER;
      }
#endif

      EventWaiter::EventWaiterList to_wake;

      if(my_node_id == owner) {
	// we own this event

	NodeSet to_update;
	gen_t update_gen;
	bool free_event = false;

	{
	  AutoHSLLock a(mutex);

	  // must always be the next generation
	  assert(gen_triggered == (generation.load() + 1));

	  to_wake.swap(current_local_waiters);
	  assert(future_local_waiters.empty()); // no future waiters here

	  to_update.swap(remote_waiters);
	  update_gen = gen_triggered;

	  // update poisoned generation list
	  bool max_poisons = false;
	  if(poisoned) {
	    if(!poisoned_generations)
	      poisoned_generations = new gen_t[POISONED_GENERATION_LIMIT];
	    int npg_cached = num_poisoned_generations.load();
	    assert(npg_cached < POISONED_GENERATION_LIMIT);
	    poisoned_generations[npg_cached] = gen_triggered;
	    num_poisoned_generations.store_release(npg_cached + 1);
	    if((npg_cached + 1) == POISONED_GENERATION_LIMIT)
	      max_poisons = true;
	  }

	  // update generation last, with a synchronization to make sure poisoned generation
	  // list is valid to any observer of this update
	  generation.store_release(gen_triggered);

	  // we'll free the event unless it's maxed out on poisoned generations
	  //  or generation count
	  free_event = ((gen_triggered < ((1U << ID::EVENT_GENERATION_WIDTH) - 1)) &&
			!max_poisons);
	  // special case: if the merger is still active, defer the
	  //  re-insertion until all the preconditions have triggered
	  if(free_event && merger.is_active()) {
	    free_list_insertion_delayed = true;
	    free_event = false;
	  }

	  // external waiters need to be signalled inside the lock
	  if(has_external_waiters) {
	    has_external_waiters = false;
	    external_waiter_condvar.broadcast();
	  }
	}

	// any remote nodes to notify?
	if(!to_update.empty()) {
	  int npg_cached = num_poisoned_generations.load_acquire();
	  ActiveMessage<EventUpdateMessage> amsg(to_update, npg_cached*sizeof(EventImpl::gen_t));
	  amsg->event = make_event(update_gen);
	  amsg.add_payload(poisoned_generations, npg_cached*sizeof(EventImpl::gen_t), PAYLOAD_KEEP);
	  amsg.commit();
	}

	// free event?
	if(free_event)
	  get_runtime()->local_event_free_list->free_entry(this);
      } else {
	// we're triggering somebody else's event, so the first thing to do is tell them
	assert(trigger_node == (int)my_node_id);
	// once we send this message, it's possible we get an update from the owner before
	//  we take the lock a few lines below here (assuming somebody on this node had 
	//  already subscribed), so check here that we're triggering a new generation
	// (the alternative is to not send the message until after we update local state, but
	// that adds latency for everybody else)
	assert(gen_triggered > generation.load());
	ActiveMessage<EventTriggerMessage> amsg(owner);
	amsg->event = make_event(gen_triggered);
	amsg->poisoned = poisoned;
	amsg.commit();
	// we might need to subscribe to intermediate generations
	bool subscribe_needed = false;
	gen_t previous_subscribe_gen = 0;

	// now update our version of the data structure
	{
	  AutoHSLLock a(mutex);

	  gen_t cur_gen = generation.load();
	  // is this the "next" version?
	  if(gen_triggered == (cur_gen + 1)) {
	    // yes, so we have complete information and can update the state directly
	    to_wake.swap(current_local_waiters);
	    // any future waiters?
	    if(!future_local_waiters.empty()) {
	      std::map<gen_t, EventWaiter::EventWaiterList>::iterator it = future_local_waiters.begin();
	      log_event.debug() << "future waiters non-empty: first=" << it->first << " (= " << (gen_triggered + 1) << "?)";
	      if(it->first == (gen_triggered + 1)) {
		current_local_waiters.swap(it->second);
		future_local_waiters.erase(it);
	      }
	    }
	    // if this event was poisoned, record it in the local triggers since we only
	    //  update the official poison list on owner update messages
	    if(poisoned) {
	      local_triggers[gen_triggered] = true;
	      has_local_triggers = true;
              subscribe_needed = true; // make sure we get that update
	    }

	    // update generation last, with a store_release to make sure poisoned generation
	    // list is valid to any observer of this update
	    generation.store_release(gen_triggered);
	  } else 
	    if(gen_triggered > (cur_gen + 1)) {
	      // we can't update the main state because there are generations that we know
	      //  have triggered, but we do not know if they are poisoned, so look in the
	      //  future waiter list to see who we can wake, and update the local trigger
	      //  list

	      std::map<gen_t, EventWaiter::EventWaiterList>::iterator it = future_local_waiters.find(gen_triggered);
	      if(it != future_local_waiters.end()) {
		to_wake.swap(it->second);
		future_local_waiters.erase(it);
	      }

	      local_triggers[gen_triggered] = poisoned;
	      has_local_triggers = true;

	      // TODO: this might still cause shutdown races - do we really
	      //  need to do this at all?
	      if(gen_triggered > (gen_subscribed + 1)) {
		subscribe_needed = true;
		previous_subscribe_gen = gen_subscribed;
		gen_subscribed = gen_triggered - 1;
	      }
	    }

	  // external waiters need to be signalled inside the lock
	  if(has_external_waiters) {
	    has_external_waiters = false;
	    external_waiter_condvar.broadcast();
	  }
	}

	if(subscribe_needed) {
	  ActiveMessage<EventSubscribeMessage> amsg(owner);
	  amsg->event = make_event(gen_triggered);
	  amsg->previous_subscribe_gen = previous_subscribe_gen;
	  amsg.commit();
	}
      }

      // finally, trigger any local waiters
      if(!to_wake.empty()) {
	if(!poisoned) {
	  if(ThreadLocal::nested_wake_list != 0) {
	    // append our waiters for caller to handle rather than recursing
	    ThreadLocal::nested_wake_list->absorb_append(to_wake);
	  } else {
	    ThreadLocal::nested_wake_list = &to_wake;  // avoid recursion
	    while(!to_wake.empty()) {
	      EventWaiter *ew = to_wake.pop_front();
	      ew->event_triggered(false /*!poisoned*/);
	    }
	    ThreadLocal::nested_wake_list = 0;
	  }
	} else {
	  while(!to_wake.empty()) {
	    EventWaiter *ew = to_wake.pop_front();
	    ew->event_triggered(true /*poisoned*/);
	  }
	}
      }
    }

    void GenEventImpl::perform_delayed_free_list_insertion(void)
    {
      bool free_event = false;

      {
	AutoHSLLock a(mutex);
	if(free_list_insertion_delayed) {
	  free_event = true;
	  free_list_insertion_delayed = false;
	}
      }

      if(free_event)
	get_runtime()->local_event_free_list->free_entry(this);
    }

    /*static*/ BarrierImpl *BarrierImpl::create_barrier(unsigned expected_arrivals,
							ReductionOpID redopid,
							const void *initial_value /*= 0*/,
							size_t initial_value_size /*= 0*/)
    {
      BarrierImpl *impl = get_runtime()->local_barrier_free_list->alloc_entry();
      assert(impl);
      assert(ID(impl->me).is_barrier());

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
	impl->redop = get_runtime()->reduce_op_table.get(redopid, 0);
	if(impl->redop == 0) {
	  log_event.fatal() << "no reduction op registered for ID " << redopid;
	  abort();
	}

	assert(initial_value != 0);
	assert(initial_value_size == impl->redop->sizeof_lhs);

	impl->initial_value = (char *)malloc(initial_value_size);
	memcpy(impl->initial_value, initial_value, initial_value_size);

	impl->value_capacity = 0;
	impl->final_values = 0;
      }

      // and let the barrier rearm as many times as necessary without being released
      //impl->free_generation = (unsigned)-1;

      log_barrier.info() << "barrier created: " << impl->me << "/" << impl->generation
			 << " base_count=" << impl->base_arrival_count << " redop=" << redopid;
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
      : has_external_waiters(false)
      , external_waiter_condvar(mutex)
    {
      generation = 0;
      gen_subscribed = 0;
      first_generation = /*free_generation =*/ 0;
      next_free = 0;
      remote_subscribe_gens.clear();
      remote_trigger_gens.clear();
      base_arrival_count = 0;
      redop = 0;
      initial_value = 0;
      value_capacity = 0;
      final_values = 0;
    }

    BarrierImpl::~BarrierImpl(void)
    {
      if(initial_value)
	free(initial_value);
      if(final_values)
	free(final_values);
    }

    void BarrierImpl::init(ID _me, unsigned _init_owner)
    {
      me = _me;
      owner = _init_owner;
      generation = 0;
      gen_subscribed = 0;
      first_generation = /*free_generation =*/ 0;
      next_free = 0;
      remote_subscribe_gens.clear();
      remote_trigger_gens.clear();
      base_arrival_count = 0;
      redop = 0;
      initial_value = 0;
      value_capacity = 0;
      final_values = 0;
    }

    /*static*/ void BarrierAdjustMessage::handle_message(NodeID sender, const BarrierAdjustMessage &args,
							 const void *data, size_t datalen)
    {
      log_barrier.info() << "received barrier arrival: delta=" << args.delta
			 << " in=" << args.wait_on << " out=" << args.barrier
			 << " (" << args.barrier.timestamp << ")";
      BarrierImpl *impl = get_runtime()->get_barrier_impl(args.barrier);
      EventImpl::gen_t gen = ID(args.barrier).barrier_generation();
      impl->adjust_arrival(gen, args.delta, args.barrier.timestamp, args.wait_on,
			   args.sender, args.forwarded,
			   datalen ? data : 0, datalen);
    }

    /*static*/ void BarrierAdjustMessage::send_request(NodeID target, Barrier barrier, int delta, Event wait_on,
						       NodeID sender, bool forwarded,
						       const void *data, size_t datalen)
    {
      ActiveMessage<BarrierAdjustMessage> amsg(target,datalen);
      amsg->barrier = barrier;
      amsg->delta = delta;
      amsg->wait_on = wait_on;
      amsg->sender = sender;
      amsg->forwarded = forwarded;
      amsg.add_payload(data, datalen);
      amsg.commit();
    }

    /*static*/ void BarrierSubscribeMessage::send_request(NodeID target, ID::IDType barrier_id, EventImpl::gen_t subscribe_gen,
							  NodeID subscriber, bool forwarded)
    {
      ActiveMessage<BarrierSubscribeMessage> amsg(target);
      amsg->subscriber = subscriber;
      amsg->forwarded = forwarded;
      amsg->barrier_id = barrier_id;
      amsg->subscribe_gen = subscribe_gen;
      amsg.commit();
    }

    /*static*/ void BarrierTriggerMessage::send_request(NodeID target, ID::IDType barrier_id,
							EventImpl::gen_t trigger_gen, EventImpl::gen_t previous_gen,
							EventImpl::gen_t first_generation, ReductionOpID redop_id,
							NodeID migration_target,	unsigned base_arrival_count,
							const void *data, size_t datalen)
    {
      ActiveMessage<BarrierTriggerMessage> amsg(target,datalen);
      amsg->barrier_id = barrier_id;
      amsg->trigger_gen = trigger_gen;
      amsg->previous_gen = previous_gen;
      amsg->first_generation = first_generation;
      amsg->redop_id = redop_id;
      amsg->migration_target = migration_target;
      amsg->base_arrival_count = base_arrival_count;
      amsg.add_payload(data, datalen);
      amsg.commit();
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
      DeferredBarrierArrival(Barrier _barrier, int _delta,
			     NodeID _sender, bool _forwarded,
			     const void *_data, size_t _datalen)
	: barrier(_barrier), delta(_delta),
	  sender(_sender), forwarded(_forwarded),
	  data(bytedup(_data, _datalen)), datalen(_datalen)
      {}

      virtual ~DeferredBarrierArrival(void)
      {
	if(data)
	  free(data);
      }

      virtual void event_triggered(bool poisoned)
      {
	// TODO: handle poison
	assert(poisoned == POISON_FIXME);
	log_barrier.info() << "deferred barrier arrival: " << barrier
			   << " (" << barrier.timestamp << "), delta=" << delta;
	BarrierImpl *impl = get_runtime()->get_barrier_impl(barrier);
	impl->adjust_arrival(ID(barrier).barrier_generation(), delta, barrier.timestamp, Event::NO_EVENT,
			     sender, forwarded,
			     data, datalen);
	// not attached to anything, so delete ourselves when we're done
	delete this;
      }

      virtual void print(std::ostream& os) const
      {
	os << "deferred arrival: barrier=" << barrier << " (" << barrier.timestamp << ")"
	   << ", delta=" << delta << " datalen=" << datalen;
      }

      virtual Event get_finish_event(void) const
      {
	return barrier;
      }

    protected:
      Barrier barrier;
      int delta;
      NodeID sender;
      bool forwarded;
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
      EventImpl::gen_t trigger_gen, previous_gen;
    };

    // used to adjust a barrier's arrival count either up or down
    // if delta > 0, timestamp is current time (on requesting node)
    // if delta < 0, timestamp says which positive adjustment this arrival must wait for
    void BarrierImpl::adjust_arrival(gen_t barrier_gen, int delta, 
				     Barrier::timestamp_t timestamp, Event wait_on,
				     NodeID sender, bool forwarded,
				     const void *reduce_value, size_t reduce_value_size)
    {
      Barrier b = make_barrier(barrier_gen, timestamp);
      if(!wait_on.has_triggered()) {
	// deferred arrival

	// only forward deferred arrivals if the precondition is not one that looks like it'll
	//  trigger here first
        if(owner != my_node_id) {
	  ID wait_id(wait_on);
	  int wait_node;
	  if(wait_id.is_event())
	    wait_node = wait_id.event_creator_node();
	  else
	    wait_node = wait_id.barrier_creator_node();
	  if(wait_node != (int)my_node_id) {
	    // let deferral happen on owner node (saves latency if wait_on event
	    //   gets triggered there)
	    //printf("sending deferred arrival to %d for " IDFMT "/%d (" IDFMT "/%d)\n",
	    //       owner, e.id, e.gen, wait_on.id, wait_on.gen);
	    log_barrier.info() << "forwarding deferred barrier arrival: delta=" << delta
			       << " in=" << wait_on << " out=" << b << " (" << timestamp << ")";
	    BarrierAdjustMessage::send_request(owner, b, delta, wait_on,
					       sender, (sender != my_node_id),
					       reduce_value, reduce_value_size);
	    return;
	  }
	}

	log_barrier.info() << "deferring barrier arrival: delta=" << delta << " in=" << wait_on
			   << " out=" << b << " (" << timestamp << ")";
	EventImpl::add_waiter(wait_on, new DeferredBarrierArrival(b, delta, 
								  sender, forwarded,
								  reduce_value, reduce_value_size));
	return;
      }

      log_barrier.info() << "barrier adjustment: event=" << b
			 << " delta=" << delta << " ts=" << timestamp;

#ifdef DEBUG_BARRIER_REDUCTIONS
      if(reduce_value_size) {
        char buffer[129];
	for(size_t i = 0; (i < reduce_value_size) && (i < 64); i++)
	  sprintf(buffer+2*i, "%02x", ((const unsigned char *)reduce_value)[i]);
	log_barrier.info("barrier reduction: event=" IDFMT "/%d size=%zd data=%s",
	                 me.id(), barrier_gen, reduce_value_size, buffer);
      }
#endif

      // can't actually trigger while holding the lock, so remember which generation(s),
      //  if any, to trigger and do it at the end
      gen_t trigger_gen = 0;
      EventWaiter::EventWaiterList local_notifications;
      std::vector<RemoteNotification> remote_notifications;
      gen_t oldest_previous = 0;
      void *final_values_copy = 0;
      NodeID migration_target = (NodeID) -1;
      NodeID forward_to_node = (NodeID) -1;
      NodeID inform_migration = (NodeID) -1;

      do { // so we can use 'break' from the middle
	AutoHSLLock a(mutex);

	bool generation_updated = false;

	// ownership can change, so check it inside the lock
	if(owner != my_node_id) {
	  forward_to_node = owner;
	  break;
	} else {
	  // if this message had to be forwarded to get here, tell the original sender we are the
	  //  new owner
	  if(forwarded && (sender != my_node_id))
	    inform_migration = sender;
	}

	// sanity checks - is this a valid barrier?
	//assert(generation < free_generation);
	assert(base_arrival_count > 0);

	// update whatever generation we're told to
	{
	  assert(barrier_gen > generation);
	  Generation *g;
	  std::map<gen_t, Generation *>::iterator it = generations.find(barrier_gen);
	  if(it != generations.end()) {
	    g = it->second;
	  } else {
	    g = new Generation;
	    generations[barrier_gen] = g;
	    log_barrier.info() << "added tracker for barrier " << me << ", generation " << barrier_gen;
	  }

	  g->handle_adjustment(timestamp, delta);
	}

	// if the update was to the next generation, it may cause one or more generations
	//  to trigger
	if(barrier_gen == (generation + 1)) {
	  std::map<gen_t, Generation *>::iterator it = generations.begin();
	  while((it != generations.end()) &&
		(it->first == (generation + 1)) &&
		((base_arrival_count + it->second->unguarded_delta) == 0)) {
	    // keep the list of local waiters to wake up once we release the lock
	    local_notifications.absorb_append(it->second->local_waiters);
	    trigger_gen = generation = it->first;
	    generation_updated = true;
	    delete it->second;
	    generations.erase(it);
	    it = generations.begin();
	  }

	  // if any triggers occurred, figure out which remote nodes need notifications
	  //  (i.e. any who have subscribed)
	  if(generation >= barrier_gen) {
	    std::map<unsigned, gen_t>::iterator it = remote_subscribe_gens.begin();
	    while(it != remote_subscribe_gens.end()) {
	      RemoteNotification rn;
	      rn.node = it->first;
	      if(it->second <= generation) {
		// we have fulfilled the entire subscription
		rn.trigger_gen = it->second;
		std::map<unsigned, gen_t>::iterator to_nuke = it++;
		remote_subscribe_gens.erase(to_nuke);
	      } else {
		// subscription remains valid
		rn.trigger_gen = generation;
		it++;
	      }
	      // also figure out what the previous generation this node knew about was
	      {
		std::map<unsigned, gen_t>::iterator it2 = remote_trigger_gens.find(rn.node);
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

#ifndef DISABLE_BARRIER_MIGRATION
	  // if there were zero local waiters and a single remote waiter, this barrier is an obvious
	  //  candidate for migration
          // don't migrate a barrier more than once though (i.e. only if it's on the creator node still)
	  // also, do not migrate a barrier if we have any local involvement in future generations
	  //  (either arrivals or waiters or a subscription that will become a waiter)
	  // finally (hah!), do not migrate barriers using reduction ops
	  if(local_notifications.empty() && (remote_notifications.size() == 1) &&
	     generations.empty() && (gen_subscribed <= generation) &&
	     (redop == 0) &&
             (NodeID(ID(me).barrier_creator_node()) == my_node_id)) {
	    log_barrier.info() << "barrier migration: " << me << " -> " << remote_notifications[0].node;
	    migration_target = remote_notifications[0].node;
	    owner = migration_target;
            // remember that we had up to date information up to this generation so that we don't try to
            //   subscribe to things we already know about
            gen_subscribed = generation;
	  }
#endif
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

	// external waiters need to be signalled inside the lock
	if(generation_updated && has_external_waiters) {
	  has_external_waiters = false;
	  external_waiter_condvar.broadcast();
	}
      } while(0);

      if(forward_to_node != (NodeID) -1) {
	Barrier b = make_barrier(barrier_gen, timestamp);
	BarrierAdjustMessage::send_request(forward_to_node, b, delta, Event::NO_EVENT,
					   sender, (sender != my_node_id),
					   reduce_value, reduce_value_size);
	return;
      }

      if(inform_migration != (NodeID) -1) {
	Barrier b = make_barrier(barrier_gen, timestamp);
	BarrierMigrationMessage::send_request(inform_migration, b, my_node_id);
      }

      if(trigger_gen != 0) {
	log_barrier.info() << "barrier trigger: event=" << me << "/" << trigger_gen;

	// notify local waiters first
	if(!local_notifications.empty()) {
	  if(!POISON_FIXME) {
	    if(ThreadLocal::nested_wake_list != 0) {
	      // append our waiters for caller to handle rather than recursing
	      ThreadLocal::nested_wake_list->absorb_append(local_notifications);
	    } else {
	      ThreadLocal::nested_wake_list = &local_notifications;  // avoid recursion
	      while(!local_notifications.empty()) {
		EventWaiter *ew = local_notifications.pop_front();
		ew->event_triggered(false /*!poisoned*/);
	      }
	      ThreadLocal::nested_wake_list = 0;
	    }
	  } else {
	    while(!local_notifications.empty()) {
	      EventWaiter *ew = local_notifications.pop_front();
	      ew->event_triggered(true /*poisoned*/);
	    }
	  }
	}

	// now do remote notifications
	for(std::vector<RemoteNotification>::const_iterator it = remote_notifications.begin();
	    it != remote_notifications.end();
	    it++) {
	  log_barrier.info() << "sending remote trigger notification: " << me << "/"
			     << (*it).previous_gen << " -> " << (*it).trigger_gen << ", dest=" << (*it).node;
	  void *data = 0;
	  size_t datalen = 0;
	  if(final_values_copy) {
	    data = (char *)final_values_copy + (((*it).previous_gen - oldest_previous) * redop->sizeof_lhs);
	    datalen = ((*it).trigger_gen - (*it).previous_gen) * redop->sizeof_lhs;
	  }
	  BarrierTriggerMessage::send_request((*it).node, me.id, (*it).trigger_gen, (*it).previous_gen,
					      first_generation, redop_id, migration_target, base_arrival_count,
					      data, datalen);
	}
      }

      // free our copy of the final values, if we had one
      if(final_values_copy)
	free(final_values_copy);
    }

    bool BarrierImpl::has_triggered(gen_t needed_gen, bool& poisoned)
    {
      poisoned = POISON_FIXME;

      // no need to take lock to check current generation
      if(needed_gen <= generation) return true;

      // update the subscription (even on the local node), but do a
      //  quick test first to avoid taking a lock if the subscription is
      //  clearly already done
      if(gen_subscribed < needed_gen) {
	// looks like it needs an update - take lock to avoid duplicate
	//  subscriptions
	gen_t previous_subscription;
	bool send_subscription_request = false;
        NodeID cur_owner = (NodeID) -1;
	{
	  AutoHSLLock a(mutex);
	  previous_subscription = gen_subscribed;
	  if(gen_subscribed < needed_gen) {
	    gen_subscribed = needed_gen;
	    // test ownership while holding the mutex
	    if(owner != my_node_id) {
	      send_subscription_request = true;
              cur_owner = owner;
            }
	  }
	}

	// if we're not the owner, send subscription if we haven't already
	if(send_subscription_request) {
	  log_barrier.info() << "subscribing to barrier " << make_barrier(needed_gen) << " (prev=" << previous_subscription << ")";
	  BarrierSubscribeMessage::send_request(cur_owner, me.id, needed_gen, my_node_id, false/*!forwarded*/);
	}
      }

      // whether or not we subscribed, the answer for now is "no"
      return false;
    }

    // TODO: this is identical to GenEventImpl versions - lift back up into
    //   EventImpl?
    void BarrierImpl::external_wait(gen_t gen_needed, bool& poisoned)
    {
      {
	AutoHSLLock a(mutex);

	// wait until the generation has advanced far enough
	while(gen_needed > generation) {
	  has_external_waiters = true;
	  external_waiter_condvar.wait();
	}

	poisoned = POISON_FIXME;
      }
    }

    bool BarrierImpl::external_timedwait(gen_t gen_needed, bool& poisoned,
					 long long max_ns)
    {
      long long deadline = Clock::current_time_in_nanoseconds() + max_ns;
      {
	AutoHSLLock a(mutex);

	// wait until the generation has advanced far enough
	while(gen_needed > generation) {
	  long long now = Clock::current_time_in_nanoseconds();
	  if(now >= deadline)
	    return false;  // trigger has not occurred
	  has_external_waiters = true;
	  // we don't actually care what timedwait returns - we'll recheck
	  //  the generation ourselves
	  external_waiter_condvar.timedwait(deadline - now);
	}

	poisoned = POISON_FIXME;
      }
      return true;
    }

    bool BarrierImpl::add_waiter(gen_t needed_gen, EventWaiter *waiter/*, bool pre_subscribed = false*/)
    {
      bool trigger_now = false;
      {
	AutoHSLLock a(mutex);

	if(needed_gen > generation) {
	  Generation *g;
	  std::map<gen_t, Generation *>::iterator it = generations.find(needed_gen);
	  if(it != generations.end()) {
	    g = it->second;
	  } else {
	    g = new Generation;
	    generations[needed_gen] = g;
	    log_barrier.info() << "added tracker for barrier " << make_barrier(needed_gen);
	  }
	  g->local_waiters.push_back(waiter);

	  // a call to has_triggered should have already handled the necessary subscription
	  assert((owner == my_node_id) || (gen_subscribed >= needed_gen));
	} else {
	  // needed generation has already occurred - trigger this waiter once we let go of lock
	  trigger_now = true;
	}
      }

      if(trigger_now) {
	waiter->event_triggered(POISON_FIXME);
      }

      return true;
    }

    /*static*/ void BarrierSubscribeMessage::handle_message(NodeID sender, const BarrierSubscribeMessage &args,
							    const void *data, size_t datalen)
    {
      ID id(args.barrier_id);
      id.barrier_generation() = args.subscribe_gen;
      Barrier b = id.convert<Barrier>();
      BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

      // take the lock and add the subscribing node - notice if they need to be notified for
      //  any generations that have already triggered
      EventImpl::gen_t trigger_gen = 0;
      EventImpl::gen_t previous_gen = 0;
      void *final_values_copy = 0;
      size_t final_values_size = 0;
      NodeID forward_to_node = (NodeID) -1;
      NodeID inform_migration = (NodeID) -1;
      
      do {
	AutoHSLLock a(impl->mutex);

	// first check - are we even the current owner?
	if(impl->owner != my_node_id) {
	  forward_to_node = impl->owner;
	  break;
	} else {
	  if(args.forwarded) {
	    // our own request wrapped back around can be ignored - we've already added the local waiter
	    if(args.subscriber == my_node_id) {
	      break;
	    } else {
	      inform_migration = args.subscriber;
	    }
	  }
	}

	// make sure the subscription is for this "lifetime" of the barrier
	assert(args.subscribe_gen > impl->first_generation);

	bool already_subscribed = false;
	{
	  std::map<unsigned, EventImpl::gen_t>::iterator it = impl->remote_subscribe_gens.find(args.subscriber);
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
	      impl->remote_subscribe_gens[args.subscriber] = args.subscribe_gen;
	  }
	}

	// as long as we're not already subscribed to this generation, check to see if
	//  any trigger notifications are needed
	if(!already_subscribed && (impl->generation > impl->first_generation)) {
	  std::map<unsigned, EventImpl::gen_t>::iterator it = impl->remote_trigger_gens.find(args.subscriber);
	  if((it == impl->remote_trigger_gens.end()) or (it->second < impl->generation)) {
	    previous_gen = ((it == impl->remote_trigger_gens.end()) ?
			      impl->first_generation :
			      it->second);
	    trigger_gen = impl->generation;
	    impl->remote_trigger_gens[args.subscriber] = impl->generation;

	    if(impl->redop) {
	      int rel_gen = previous_gen + 1 - impl->first_generation;
	      assert(rel_gen > 0);
	      final_values_size = (trigger_gen - previous_gen) * impl->redop->sizeof_lhs;
	      final_values_copy = bytedup(impl->final_values + ((rel_gen - 1) * impl->redop->sizeof_lhs),
					  final_values_size);
	    }
	  }
	}
      } while(0);

      if(forward_to_node != (NodeID) -1) {
	BarrierSubscribeMessage::send_request(forward_to_node, args.barrier_id, args.subscribe_gen,
					      args.subscriber, (args.subscriber != my_node_id));
      }

      if(inform_migration != (NodeID) -1) {
	BarrierMigrationMessage::send_request(inform_migration, b, my_node_id);
      }

      // send trigger message outside of lock, if needed
      if(trigger_gen > 0) {
	log_barrier.info("sending immediate barrier trigger: " IDFMT "/%d -> %d",
			 args.barrier_id, previous_gen, trigger_gen);
	BarrierTriggerMessage::send_request(args.subscriber, args.barrier_id, trigger_gen, previous_gen,
					    impl->first_generation, impl->redop_id,
					    (NodeID) -1 /*no migration*/, 0 /*dummy arrival count*/,
					    final_values_copy, final_values_size);
      }

      if(final_values_copy)
	free(final_values_copy);
    }

   /*static*/ void BarrierTriggerMessage::handle_message(NodeID sender, const BarrierTriggerMessage &args,
							 const void *data, size_t datalen)
    {
      log_barrier.info("received remote barrier trigger: " IDFMT "/%d -> %d",
		       args.barrier_id, args.previous_gen, args.trigger_gen);

      EventImpl::gen_t trigger_gen = args.trigger_gen;

      ID id(args.barrier_id);
      id.barrier_generation() = trigger_gen;
      Barrier b = id.convert<Barrier>();
      BarrierImpl *impl = get_runtime()->get_barrier_impl(b);

      // we'll probably end up with a list of local waiters to notify
      EventWaiter::EventWaiterList local_notifications;
      {
	AutoHSLLock a(impl->mutex);

	bool generation_updated = false;

	// handle migration of the barrier ownership (possibly to us)
	if(args.migration_target != (NodeID) -1) {
	  log_barrier.info() << "barrier " << b << " has migrated to " << args.migration_target;
	  impl->owner = args.migration_target;
	  impl->base_arrival_count = args.base_arrival_count;
	}

	// it's theoretically possible for multiple trigger messages to arrive out
	//  of order, so check if this message triggers the oldest possible range
	// NOTE: it's ok for previous_gen to be earlier than our current generation - this
	//  occurs with barrier migration because the new owner may not know which notifications
	//  have already been performed
	if(args.previous_gen <= impl->generation) {
	  // see if we can pick up any of the held triggers too
	  while(!impl->held_triggers.empty()) {
	    std::map<EventImpl::gen_t, EventImpl::gen_t>::iterator it = impl->held_triggers.begin();
	    // if it's not contiguous, we're done
	    if(it->first != trigger_gen) break;
	    // it is contiguous, so absorb it into this message and remove the held trigger
	    log_barrier.info("collapsing future trigger: " IDFMT "/%d -> %d -> %d",
			     args.barrier_id, args.previous_gen, trigger_gen, it->second);
	    trigger_gen = it->second;
	    impl->held_triggers.erase(it);
	  }

	  if(trigger_gen > impl->generation) {
	    impl->generation = trigger_gen;
	    generation_updated = true;
	  }

	  // now iterate through any generations up to and including the latest triggered
	  //  generation, and accumulate local waiters to notify
	  while(!impl->generations.empty()) {
	    std::map<EventImpl::gen_t, BarrierImpl::Generation *>::iterator it = impl->generations.begin();
	    if(it->first > trigger_gen) break;

	    local_notifications.absorb_append(it->second->local_waiters);
	    delete it->second;
	    impl->generations.erase(it);
	  }
	} else {
	  // hold this trigger until we get messages for the earlier generation(s)
	  log_barrier.info("holding future trigger: " IDFMT "/%d (%d -> %d)",
			   args.barrier_id, impl->generation, 
			   args.previous_gen, trigger_gen);
	  impl->held_triggers[args.previous_gen] = trigger_gen;
	}

	// is there any data we need to store?
	if(datalen) {
	  assert(args.redop_id != 0);

	  // TODO: deal with invalidation of previous instance of a barrier
	  impl->redop_id = args.redop_id;
	  impl->redop = get_runtime()->reduce_op_table.get(args.redop_id, 0);
	  if(impl->redop == 0) {
	    log_event.fatal() << "no reduction op registered for ID " << args.redop_id;
	    abort();
	  }
	  impl->first_generation = args.first_generation;

	  int rel_gen = trigger_gen - impl->first_generation;
	  assert(rel_gen > 0);
	  if(impl->value_capacity < (size_t)rel_gen) {
	    size_t new_capacity = rel_gen;
	    impl->final_values = (char *)realloc(impl->final_values, new_capacity * impl->redop->sizeof_lhs);
	    // no need to initialize new entries - we'll overwrite them now or when data does show up
	    impl->value_capacity = new_capacity;
	  }
	  assert(datalen == (impl->redop->sizeof_lhs * (trigger_gen - args.previous_gen)));
	  memcpy(impl->final_values + ((rel_gen - 1) * impl->redop->sizeof_lhs), data, datalen);
	}

	// external waiters need to be signalled inside the lock
	if(generation_updated && impl->has_external_waiters) {
	  impl->has_external_waiters = false;
	  impl->external_waiter_condvar.broadcast();
	}
      }

      // with lock released, perform any local notifications
      if(!local_notifications.empty()) {
	if(!POISON_FIXME) {
	  if(ThreadLocal::nested_wake_list != 0) {
	    // append our waiters for caller to handle rather than recursing
	    ThreadLocal::nested_wake_list->absorb_append(local_notifications);
	  } else {
	    ThreadLocal::nested_wake_list = &local_notifications;  // avoid recursion
	    while(!local_notifications.empty()) {
	      EventWaiter *ew = local_notifications.pop_front();
	      ew->event_triggered(false /*!poisoned*/);
	    }
	    ThreadLocal::nested_wake_list = 0;
	  }
	} else {
	  while(!local_notifications.empty()) {
	    EventWaiter *ew = local_notifications.pop_front();
	    ew->event_triggered(true /*poisoned*/);
	  }
	}
      }
    }

    bool BarrierImpl::get_result(gen_t result_gen, void *value, size_t value_size)
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

    /*static*/ void BarrierMigrationMessage::handle_message(NodeID sender, const BarrierMigrationMessage &args,
							    const void *data, size_t datalen)
    {
      log_barrier.info() << "received barrier migration: barrier=" << args.barrier << " owner=" << args.current_owner;
      BarrierImpl *impl = get_runtime()->get_barrier_impl(args.barrier);
      {
	AutoHSLLock a(impl->mutex);
	impl->owner = args.current_owner;
      }
    }

    /*static*/ void BarrierMigrationMessage::send_request(NodeID target, Barrier barrier, NodeID owner)
    {
      ActiveMessage<BarrierMigrationMessage> amsg(target);
      amsg->barrier = barrier;
      amsg->current_owner = owner;
      amsg.commit();
    }

  ActiveMessageHandlerReg<EventSubscribeMessage> event_subscribe_message_handler;
  ActiveMessageHandlerReg<EventTriggerMessage> event_trigger_message_handler;
  ActiveMessageHandlerReg<EventUpdateMessage> event_update_message_handler;
  ActiveMessageHandlerReg<BarrierAdjustMessage> barrier_adjust_message_handler;
  ActiveMessageHandlerReg<BarrierSubscribeMessage> barrier_subscribe_message_handler;
  ActiveMessageHandlerReg<BarrierTriggerMessage> barrier_trigger_message_handler;
  ActiveMessageHandlerReg<BarrierMigrationMessage> barrier_migration_message_handler;

}; // namespace Realm
