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
  Logger log_compqueue("compqueue");

  // used in places that don't currently propagate poison but should
  static const bool POISON_FIXME = false;

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
    if(!id) {
      poisoned = false;
      return true; // special case: NO_EVENT has always triggered
    }
    EventImpl *e = get_runtime()->get_event_impl(*this);
    return e->has_triggered(ID(id).event_generation(), poisoned);
  }

  void Event::subscribe(void) const
  {
    // early out - NO_EVENT and local generational events never require
    //  subscription
    if(!id ||
       (ID(id).is_event() &&
	(NodeID(ID(id).event_creator_node()) == Network::my_node_id)))
      return;

    EventImpl *e = get_runtime()->get_event_impl(*this);
    e->subscribe(ID(id).event_generation());
  }

  // creates an event that won't trigger until all input events have
  /*static*/ Event Event::merge_events(const std::set<Event>& wait_for)
  {
    return GenEventImpl::merge_events(wait_for, false /*!ignore faults*/);
  }

  /*static*/ Event Event::merge_events(const std::vector<Event>& wait_for)
  {
    return GenEventImpl::merge_events(wait_for, false /*!ignore faults*/);
  }

  /*static*/ Event Event::merge_events(Event ev1, Event ev2,
				       Event ev3 /*= NO_EVENT*/, Event ev4 /*= NO_EVENT*/,
				       Event ev5 /*= NO_EVENT*/, Event ev6 /*= NO_EVENT*/)
  {
    return GenEventImpl::merge_events(ev1, ev2, ev3, ev4, ev5, ev6);
  }

  /*static*/ Event Event::merge_events_ignorefaults(const std::set<Event>& wait_for)
  {
    return GenEventImpl::merge_events(wait_for, true /*ignore faults*/);
  }

  /*static*/ Event Event::merge_events_ignorefaults(const std::vector<Event>& wait_for)
  {
    return GenEventImpl::merge_events(wait_for, true /*ignore faults*/);
  }

  /*static*/ Event Event::ignorefaults(Event wait_for)
  {
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
      virtual void event_triggered(bool poisoned, TimeLimit work_until);
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
    bool remove_callback(Callback& cb) const;

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

  bool EventTriggeredCondition::remove_callback(Callback& cb) const
  {
    return event->remove_waiter(gen, &cb);
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
  
  void EventTriggeredCondition::Callback::event_triggered(bool poisoned,
							  TimeLimit work_until)
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
    if(!id) {
      poisoned = false;
      return;  // special case: never wait for NO_EVENT
    }
    EventImpl *e = get_runtime()->get_event_impl(*this);
    EventImpl::gen_t gen = ID(id).event_generation();

    // early out case too
    if(e->has_triggered(gen, poisoned)) return;

    // if not called from a task, use external_wait instead
    if(!ThreadLocal::current_processor.exists()) {
      log_event.info() << "external thread blocked: event=" << *this;
      e->external_wait(gen, poisoned);
      log_event.info() << "external thread resumed: event=" << *this;
      return;
    }

    Thread *thread = Thread::self();
    assert(thread); // all tasks had better have a thread...

    log_event.info() << "thread blocked: thread=" << thread
                     << " event=" << *this;
    // see if we are being asked to profile these waits
    ProfilingMeasurements::OperationEventWaits::WaitInterval *interval = 0;
    if(thread->get_operation() != 0) {
      interval = thread->get_operation()->create_wait_interval(*this);
      if(interval)
        interval->record_wait_start();
    }
    // describe the condition we want the thread to wait on
    thread->wait_for_condition(EventTriggeredCondition(e, gen, interval),
                               poisoned);
    if(interval)
      interval->record_wait_end();
    log_event.info() << "thread resumed: thread=" << thread
                     << " event=" << *this << " poisoned=" << poisoned;
  }

  void Event::external_wait(void) const
  {
    bool poisoned = false;
    external_wait_faultaware(poisoned);
    // a poisoned event causes an exception because the caller isn't prepared for it
    if(poisoned) {
#ifdef REALM_USE_EXCEPTIONS
      if(Thread::self() && Thread::self()->exceptions_permitted()) {
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
    if(!id) {
      poisoned = false;
      return;  // special case: never wait for NO_EVENT
    }
    EventImpl *e = get_runtime()->get_event_impl(*this);
    EventImpl::gen_t gen = ID(id).event_generation();

    // early out case too
    if(e->has_triggered(gen, poisoned)) return;
    
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
    if(!id) {
      poisoned = false;
      return true;  // special case: never wait for NO_EVENT
    }
    EventImpl *e = get_runtime()->get_event_impl(*this);
    EventImpl::gen_t gen = ID(id).event_generation();

    // early out case too
    if(e->has_triggered(gen, poisoned)) return true;

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

  /*static*/ const ::realm_event_gen_t Barrier::MAX_PHASES = (::realm_event_gen_t(1) << REALM_EVENT_GENERATION_BITS) - 1;

  /*static*/ Barrier Barrier::create_barrier(unsigned expected_arrivals,
					     ReductionOpID redop_id /*= 0*/,
					     const void *initial_value /*= 0*/,
					     size_t initial_value_size /*= 0*/)
  {
    BarrierImpl *impl = BarrierImpl::create_barrier(expected_arrivals, redop_id, initial_value, initial_value_size);
    Barrier b = impl->current_barrier();

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
#ifdef DEBUG_REALM
    assert(MAX_PHASES <= nextid.barrier_generation().MAXVAL);
#endif
    // return NO_BARRIER if the count overflows
    if(gen > MAX_PHASES)
      return Barrier::NO_BARRIER;
    nextid.barrier_generation() = ID(id).barrier_generation() + 1;

    Barrier nextgen = nextid.convert<Barrier>();
    nextgen.timestamp = 0;

    return nextgen;
  }

  Barrier Barrier::alter_arrival_count(int delta) const
  {
    timestamp_t timestamp = BarrierImpl::barrier_adjustment_timestamp.fetch_add(1);
    BarrierImpl *impl = get_runtime()->get_barrier_impl(*this);
    impl->adjust_arrival(ID(id).barrier_generation(), delta, timestamp, Event::NO_EVENT,
			 Network::my_node_id, false /*!forwarded*/,
			 0, 0, TimeLimit::responsive());

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
    // arrival uses the timestamp stored in this barrier object
    BarrierImpl *impl = get_runtime()->get_barrier_impl(*this);
    impl->adjust_arrival(ID(id).barrier_generation(), -int(count), timestamp, wait_on,
			 Network::my_node_id, false /*!forwarded*/,
			 reduce_value, reduce_value_size,
			 TimeLimit::responsive());
  }

  bool Barrier::get_result(void *value, size_t value_size) const
  {
    BarrierImpl *impl = get_runtime()->get_barrier_impl(*this);
    return impl->get_result(ID(id).barrier_generation(), value, value_size);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class EventTriggerNotifier
  //

  EventTriggerNotifier::EventTriggerNotifier()
    : BackgroundWorkItem("event triggers")
  {}

  void EventTriggerNotifier::trigger_event_waiters(EventWaiter::EventWaiterList& to_trigger,
						   bool poisoned,
						   TimeLimit trigger_until)
  {
    // we get one list from the caller, but we need a total of two
    EventWaiter::EventWaiterList second_list;
    if(poisoned) {
      if(nested_poisoned != 0) {
	nested_poisoned->absorb_append(to_trigger);
	return;
      }
      // install lists to catch any recursive triggers
      nested_poisoned = &to_trigger;
      nested_normal = &second_list;
    } else {
      if(nested_normal != 0) {
	nested_normal->absorb_append(to_trigger);
	return;
      }
      // install lists to catch any recursive triggers
      nested_normal = &to_trigger;
      nested_poisoned = &second_list;
    }

    do {
      // TODO: triggers are fast - consider doing more than one per time check?
      if(!nested_normal->empty()) {
	EventWaiter *w = nested_normal->pop_front();
	w->event_triggered(false /*!poisoned*/, trigger_until);
      } else if(!nested_poisoned->empty()) {
	EventWaiter *w = nested_poisoned->pop_front();
	w->event_triggered(true /*poisoned*/, trigger_until);
      } else {
	// list is exhausted - we can return right away (after removing
	//   trigger-catching lists)
	nested_normal = nested_poisoned = 0;
	return;
      }
    } while(!trigger_until.is_expired());

    // do we have any triggers we want to defer?
    if(!nested_normal->empty() || !nested_poisoned->empty()) {
      bool was_empty;
      {
	AutoLock<> al(mutex);
	was_empty = delayed_normal.empty() && delayed_poisoned.empty();
	delayed_normal.absorb_append(*nested_normal);
	delayed_poisoned.absorb_append(*nested_poisoned);
      }
      if(was_empty)
	make_active();
    }

    // done catching recursive event triggers
    nested_normal = nested_poisoned = 0;
  }

  bool EventTriggerNotifier::do_work(TimeLimit work_until)
  {
    // take the lock and grab both lists
    EventWaiter::EventWaiterList todo_normal, todo_poisoned;
    {
      AutoLock<> al(mutex);
      todo_normal.swap(delayed_normal);
      todo_poisoned.swap(delayed_poisoned);
    }

    // any nested triggering should append to our list instead of recurse
    nested_normal = &todo_normal;
    nested_poisoned = &todo_poisoned;

    // now trigger until we're out of time
    do {
      // TODO: triggers are fast - consider doing more than one per time check?
      if(!todo_normal.empty()) {
	EventWaiter *w = todo_normal.pop_front();
	w->event_triggered(false /*!poisoned*/, work_until);
      } else if(!todo_poisoned.empty()) {
	EventWaiter *w = todo_poisoned.pop_front();
	w->event_triggered(true /*poisoned*/, work_until);
      } else
	break;
    } while(!work_until.is_expired());

    // un-register nested trigger catchers
    nested_normal = nested_poisoned = 0;

    // if we have anything left to do, prepend (using append+swap) them to
    //  whatever got added by other threads while we were triggering stuff
    if(!todo_normal.empty() || !todo_poisoned.empty()) {
      bool was_empty;
      {
	AutoLock<> al(mutex);
	was_empty = delayed_normal.empty() && delayed_poisoned.empty();
	if(!todo_normal.empty()) {
	  if(!delayed_normal.empty())
	    todo_normal.absorb_append(delayed_normal);
	  delayed_normal.swap(todo_normal);
	}
	if(!todo_poisoned.empty()) {
	  if(!delayed_poisoned.empty())
	    todo_poisoned.absorb_append(delayed_poisoned);
	  delayed_poisoned.swap(todo_poisoned);
	}
      }
      if(was_empty)
        return true;  // request requeuing to get more work done
    }

    // no work left or already requeued
    return false;
  }

  /*static*/ REALM_THREAD_LOCAL EventWaiter::EventWaiterList *EventTriggerNotifier::nested_normal = 0;
  /*static*/ REALM_THREAD_LOCAL EventWaiter::EventWaiterList *EventTriggerNotifier::nested_poisoned = 0;


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
      Event next_barrier_gen = Event::NO_EVENT;

      ID id(e);
      if(id.is_event()) {
	GenEventImpl *impl = get_runtime()->get_genevent_impl(e);

	{
	  AutoLock<> al(impl->mutex);
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
        BarrierImpl *impl = get_runtime()->get_barrier_impl(e);

        {
          AutoLock<> al(impl->mutex);
          std::map<gen_t, BarrierImpl::Generation *>::const_iterator it = impl->generations.begin();
          // skip any generations before the one of interest
          while((it != impl->generations.end()) && (it->first < id.barrier_generation()))
            ++it;
          // take the waiter list of the exact generation of interest (if exists)
          if((it != impl->generations.end() && (it->first == id.barrier_generation()))) {
            waiters_head = it->second->local_waiters.head.next;
            ++it;
          }
          // if there's waiters on future barrier generations, they're implicitly
          //  dependent on this generation
          if(it != impl->generations.end())
            next_barrier_gen = impl->make_event(it->first);
        }
      } else {
	assert(0);
      }

      assert(!next_barrier_gen.exists());

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

  void EventMerger::MergeEventPrecondition::event_triggered(bool poisoned,
							    TimeLimit work_until)
  {
    merger->precondition_triggered(poisoned, work_until);
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
      assert(!is_active());
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
      faults_observed.store(0);
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
	  bool first_fault = (faults_observed.fetch_add(1) == 0);
	  if(first_fault && !ignore_faults) {
            log_poison.info() << "event merger early poison: after=" << event_impl->make_event(finish_gen);
	    event_impl->trigger(finish_gen, Network::my_node_id,
				true /*poisoned*/,
				TimeLimit::responsive());
	  }
	}
	// either way we return to the caller without updating the count_needed
	return;
      }

      // figure out which precondition slot we'll use
      assert(num_preconditions < max_preconditions);
      MergeEventPrecondition *p = &preconditions[num_preconditions++];

      // increment count first, then add the waiter
      count_needed.fetch_add_acqrel(1);
      EventImpl::add_waiter(wait_for, p);
    }

    // as an alternative to add_precondition, get_next_precondition can
    //  be used to get a precondition that can manually be added to a waiter
    //  list
    EventMerger::MergeEventPrecondition *EventMerger::get_next_precondition(void)
    {
      assert(is_active());
      assert(num_preconditions < max_preconditions);
      MergeEventPrecondition *p = &preconditions[num_preconditions++];
      count_needed.fetch_add(1);
      return p;
    }

    void EventMerger::arm_merger(void)
    {
      assert(is_active());
      precondition_triggered(false /*!poisoned*/, TimeLimit::responsive());
    }

    void EventMerger::precondition_triggered(bool poisoned,
					     TimeLimit work_until)
    {
      // if the input is poisoned, we propagate that poison eagerly
      if(poisoned) {
	bool first_fault = (faults_observed.fetch_add(1) == 0);
	if(first_fault && !ignore_faults) {
	  log_poison.info() << "event merger poisoned: after=" << event_impl->make_event(finish_gen);
	  event_impl->trigger(finish_gen, Network::my_node_id,
			      true /*poisoned*/, work_until);
	}
      }

      // used below, but after we're allowed to look at the object
      Event e = Event::NO_EVENT;
      if(log_event.want_debug())
        e = event_impl->make_event(finish_gen);

      // once we decrement this, if we aren't the last trigger, we can't
      //  look at *this again
      int count_left = count_needed.fetch_sub_acqrel(1);

      // Put the logging first to avoid segfaults
      log_event.debug() << "received trigger merged event=" << e
			<< " left=" << count_left << " poisoned=" << poisoned;

      // count is the value before the decrement, so it was 1, it's now 0
      bool last_trigger = (count_left == 1);

      if(last_trigger) {
	// if we dynamically allocated space for a wide merger, give that
	//  storage back - the chance that this particular event will have
	//  another wide merge isn't particularly high
	if(max_preconditions > MAX_INLINE_PRECONDITIONS) {
	  delete[] preconditions;
	  preconditions = inline_preconditions;
	  max_preconditions = MAX_INLINE_PRECONDITIONS;
	}

	// trigger on the last input event, unless we did an early poison propagation
	if(ignore_faults || (faults_observed.load() == 0))
	  event_impl->trigger(finish_gen, Network::my_node_id,
			      false /*!poisoned*/, work_until);

	// if the event was triggered early due to poison, its insertion on
	//  the free list is delayed until we know that the event merger is
	//  inactive (i.e. when last_trigger is true)
	event_impl->perform_delayed_free_list_insertion();
      }
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
    , external_waiter_condvar(external_waiter_mutex)
  {
    next_free = 0;
    poisoned_generations = 0;
    has_local_triggers = false;
    free_list_insertion_delayed = false;
  }

  GenEventImpl::~GenEventImpl(void)
  {
#ifdef DEBUG_REALM
    AutoLock<> a(mutex);
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
	log_event.fatal() << "  waiting on " << make_event(generation.load() + 1) << ": " << ew;
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
    if(poisoned_generations)
      delete[] poisoned_generations;
  }

  void GenEventImpl::init(ID _me, unsigned _init_owner)
  {
    me = _me;
    owner = _init_owner;
    generation.store(0);
    gen_subscribed.store(0);
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
      // counts of 0 or 1 don't require any merging
      if(wait_count == 0) return Event::NO_EVENT;
      if((wait_count == 1) && !ignore_faults) return first_wait;

      // counts of 2+ require building a new event and a merger to trigger it
      GenEventImpl *event_impl = GenEventImpl::create_genevent();
      Event finish_event = event_impl->current_event();

      EventMerger *m = &(event_impl->merger);
      m->prepare_merger(finish_event, ignore_faults, wait_for.size());

      for(std::set<Event>::const_iterator it = wait_for.begin();
	  it != wait_for.end();
	  it++) {
	log_event.info() << "event merging: event=" << finish_event << " wait_on=" << *it;
	m->add_precondition(*it);
      }

      // once they're all added - arm the thing (it might go off immediately)
      m->arm_merger();

      return finish_event;
    }

    // creates an event that won't trigger until all input events have
    /*static*/ Event GenEventImpl::merge_events(span<const Event> wait_for,
						bool ignore_faults)
    {
      if (wait_for.empty())
        return Event::NO_EVENT;
      // scan through events to see how many exist/haven't fired - we're
      //  interested in counts of 0, 1, or 2+ - also remember the first
      //  event we saw for the count==1 case
      int wait_count = 0;
      size_t first_wait = 0;
      for(size_t i = 0; (i < wait_for.size()) && (wait_count < 2); i++) {
	bool poisoned = false;
	if(wait_for[i].has_triggered_faultaware(poisoned)) {
          if(poisoned) {
	    // if we're not ignoring faults, we need to propagate this fault, and can do
	    //  so by just returning this poisoned event
	    if(!ignore_faults) {
	      log_poison.info() << "merging events - " << wait_for[i] << " already poisoned";
	      return wait_for[i];
	    }
          }
	} else {
	  if(!wait_count) first_wait = i;
	  wait_count++;
	}
      }
      log_event.debug() << "merging events - at least " << wait_count << " not triggered";

      // counts of 0 or 1 don't require any merging
      if(wait_count == 0) return Event::NO_EVENT;
      if((wait_count == 1) && !ignore_faults) return wait_for[first_wait];

      // counts of 2+ require building a new event and a merger to trigger it
      GenEventImpl *event_impl = GenEventImpl::create_genevent();
      Event finish_event = event_impl->current_event();
      EventMerger *m = &(event_impl->merger);

      m->prepare_merger(finish_event, ignore_faults,
			wait_for.size() - first_wait);

      for(size_t i = first_wait; i < wait_for.size(); i++) {
	log_event.info() << "event merging: event=" << finish_event << " wait_on=" << wait_for[i];
	m->add_precondition(wait_for[i]);
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
      log_event.info() << "event merging: event=" << finish_event 
                       << " wait_on=" << wait_for;
      m->add_precondition(wait_for);
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

      // counts of 0 or 1 don't require any merging
      if(wait_count == 0) return Event::NO_EVENT;
      if(wait_count == 1) return first_wait;

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
	AutoLock<> a(mutex);

	// three cases below

	if(needed_gen <= generation.load()) {
	  // 1) the event has triggered and any poison information is in the poisoned generation list
	  trigger_now = true; // actually do trigger outside of mutex
	  trigger_poisoned = is_generation_poisoned(needed_gen);
	} else {
	  std::map<gen_t, bool>::const_iterator it = local_triggers.find(needed_gen);
	  if(it != local_triggers.end()) {
	    // 2) we're not the owner node, but we've locally triggered this and have correct poison info
	    assert(owner != Network::my_node_id);
	    trigger_now = true;
	    trigger_poisoned = it->second;
	  } else {
	    // 3) we don't know of a trigger of this event, so record the waiter and subscribe if needed
	    gen_t cur_gen = generation.load();
	    log_event.debug() << "event not ready: event=" << me << "/" << needed_gen
			      << " owner=" << owner << " gen=" << cur_gen << " subscr=" << gen_subscribed.load();

	    // is this for the "current" next generation?
	    if(needed_gen == (cur_gen + 1)) {
	      // yes, put in the current waiter list
	      current_local_waiters.push_back(waiter);
	    } else {
	      // no, put it in an appropriate future waiter list - only allowed for non-owners
	      assert(owner != Network::my_node_id);
	      future_local_waiters[needed_gen].push_back(waiter);
	    }

	    // do we need to subscribe to this event?
	    if((owner != Network::my_node_id) &&
	       (gen_subscribed.load() < needed_gen)) {
	      previous_subscribe_gen = gen_subscribed.load();
	      gen_subscribed.store(needed_gen);
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
	waiter->event_triggered(trigger_poisoned, TimeLimit::responsive());

      return true;  // waiter is always either enqueued or triggered right now
    }

    bool GenEventImpl::remove_waiter(gen_t needed_gen, EventWaiter *waiter)
    {
      AutoLock<> a(mutex);

      // case 1: the event has already triggered, so nothing to remove
      // TODO: this might still be racy with delayed event waiter notification
      if(needed_gen <= generation.load())
	return false;

      // case 2: is it a local trigger we've also already dealt with?
      {
	std::map<gen_t, bool>::const_iterator it = local_triggers.find(needed_gen);
	if(it != local_triggers.end())
	  return false;
      }

      // case 3: it'd better be in a waiter list
      if(needed_gen == (generation.load() + 1)) {
	bool ok = current_local_waiters.erase(waiter) > 0;
	assert(ok);
	return true;
      } else {
	bool ok = future_local_waiters[needed_gen].erase(waiter) > 0;
	assert(ok);
	return true;
      }
    }

    inline bool GenEventImpl::is_generation_poisoned(gen_t gen) const
    {
      // common case: no poisoned generations
      int npg_cached = num_poisoned_generations.load_acquire();
      if(REALM_LIKELY(npg_cached == 0))
	return false;
      
      for(int i = 0; i < npg_cached; i++)
	if(poisoned_generations[i] == gen)
	  return true;
      return false;
    }


    ///////////////////////////////////////////////////
    // Events


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
	AutoLock<> a(impl->mutex);

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
	ActiveMessage<EventUpdateMessage> amsg(sender,
                                               impl->poisoned_generations,
                                               npg_cached*sizeof(EventImpl::gen_t));
	amsg->event = triggered;
	amsg.commit();
      }
    } 

    /*static*/ void EventTriggerMessage::handle_message(NodeID sender, const EventTriggerMessage &args,
							const void *data, size_t datalen,
							TimeLimit work_until)
    {
      log_event.debug() << "remote trigger of event " << args.event << " from node " << sender;
      GenEventImpl *impl = get_runtime()->get_genevent_impl(args.event);
      impl->trigger(ID(args.event).event_generation(), sender, args.poisoned,
		    work_until);
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
				    int new_poisoned_count,
				    TimeLimit work_until)
  {
    // this event had better not belong to us...
    assert(owner != Network::my_node_id);

    // the result of the update may trigger multiple generations worth of waiters - keep their
    //  generation IDs straight (we'll look up the poison bits later)
    std::map<gen_t, EventWaiter::EventWaiterList> to_wake;

    {
      AutoLock<> a(mutex);

      // this might be old news (due to packet reordering or if we had
      //  subscribed to an event and then triggered it ourselves)
      if(current_gen <= generation.load())
	return;

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
        // also need external waiter mutex
        AutoLock<KernelMutex> al2(external_waiter_mutex);
	external_waiter_condvar.broadcast();
      }
    }

    // now trigger anybody that needs to be triggered
    if(!to_wake.empty()) {
      for(std::map<gen_t, EventWaiter::EventWaiterList>::iterator it = to_wake.begin();
	  it != to_wake.end();
	  it++) {
	bool poisoned = is_generation_poisoned(it->first);
	get_runtime()->event_triggerer.trigger_event_waiters(it->second,
							     poisoned,
							     work_until);
      }
    }
  }

    /*static*/ void EventUpdateMessage::handle_message(NodeID sender, const EventUpdateMessage &args,
						       const void *data, size_t datalen,
						       TimeLimit work_until)
    {
      const EventImpl::gen_t *new_poisoned_gens = (const EventImpl::gen_t *)data;
      int new_poisoned_count = datalen / sizeof(EventImpl::gen_t);
      assert((new_poisoned_count * sizeof(EventImpl::gen_t)) == datalen);  // no remainders or overflow please

      log_event.debug() << "event update: event=" << args.event
			<< " poisoned=" << ArrayOstreamHelper<EventImpl::gen_t>(new_poisoned_gens, new_poisoned_count);

      GenEventImpl *impl = get_runtime()->get_genevent_impl(args.event);
      impl->process_update(ID(args.event).event_generation(),
			   new_poisoned_gens, new_poisoned_count,
			   work_until);
    }


  /*static*/ atomic<Barrier::timestamp_t> BarrierImpl::barrier_adjustment_timestamp(0);



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
	AutoLock<> a(mutex);

	std::map<gen_t, bool>::const_iterator it = local_triggers.find(needed_gen);
	if(it != local_triggers.end()) {
	  locally_triggered = true;
	  poisoned = it->second;
	}
      }
      return locally_triggered;
    }

    void GenEventImpl::subscribe(gen_t subscribe_gen)
    {
      // should never be called on a local event
      assert(owner != Network::my_node_id);
      
      // lock-free check on previous subscriptions or known triggers
      if((subscribe_gen <= gen_subscribed.load_acquire()) ||
	 (subscribe_gen <= generation.load_acquire()))
	return;

      bool subscribe_needed = false;
      gen_t previous_subscribe_gen = 0;
      {
	AutoLock<> a(mutex);

	// if the requested generation is already known to be triggered
	bool already_triggered = false;
	if(subscribe_gen <= generation.load()) {
	  already_triggered = true;
	} else 
	  if(has_local_triggers) {
	    // if we have a local trigger (poisoned or not), that counts too
	    if(local_triggers.count(subscribe_gen))
	      already_triggered = true;
	  }

	if(!already_triggered && (subscribe_gen > gen_subscribed.load())) {
	  subscribe_needed = true;
	  previous_subscribe_gen = gen_subscribed.load();
	  gen_subscribed.store(subscribe_gen);
	}
      }

      if(subscribe_needed) {
	ActiveMessage<EventSubscribeMessage> amsg(owner);
	amsg->event = make_event(subscribe_gen);
	amsg->previous_subscribe_gen = previous_subscribe_gen;
	amsg.commit();
      }
    }
  
    void GenEventImpl::external_wait(gen_t gen_needed, bool& poisoned)
    {
      // if the event is remote, make sure we've subscribed
      if(this->owner != Network::my_node_id)
	this->subscribe(gen_needed);

      {
	AutoLock<> a(mutex);

	// wait until the generation has advanced far enough
	while(gen_needed > generation.load_acquire()) {
	  has_external_waiters = true;
          // must wait on external_waiter_condvar with external_waiter_mutex
          //  but NOT with base mutex - hand-over-hand lock on the way in,
          //  and then release external_waiter mutex before retaking main
          //  mutex
          external_waiter_mutex.lock();
          mutex.unlock();
	  external_waiter_condvar.wait();
          external_waiter_mutex.unlock();
          mutex.lock();
	}

	poisoned = is_generation_poisoned(gen_needed);
      }
    }

    bool GenEventImpl::external_timedwait(gen_t gen_needed, bool& poisoned,
					  long long max_ns)
    {
      long long deadline = Clock::current_time_in_nanoseconds() + max_ns;
      {
	AutoLock<> a(mutex);

	// wait until the generation has advanced far enough
	while(gen_needed > generation.load_acquire()) {
	  long long now = Clock::current_time_in_nanoseconds();
	  if(now >= deadline)
	    return false;  // trigger has not occurred
	  has_external_waiters = true;
	  // we don't actually care what timedwait returns - we'll recheck
	  //  the generation ourselves
          // must wait on external_waiter_condvar with external_waiter_mutex
          //  but NOT with base mutex - hand-over-hand lock on the way in,
          //  and then release external_waiter mutex before retaking main
          //  mutex
          external_waiter_mutex.lock();
          mutex.unlock();
	  external_waiter_condvar.timedwait(deadline - now);
          external_waiter_mutex.unlock();
          mutex.lock();
	}

	poisoned = is_generation_poisoned(gen_needed);
      }
      return true;
    }

    void GenEventImpl::trigger(gen_t gen_triggered, int trigger_node,
			       bool poisoned, TimeLimit work_until)
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

      if(Network::my_node_id == owner) {
	// we own this event

	NodeSet to_update;
	gen_t update_gen;
	bool free_event = false;

	{
	  AutoLock<> a(mutex);

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
            // also need external waiter mutex
            AutoLock<KernelMutex> al2(external_waiter_mutex);
	    external_waiter_condvar.broadcast();
	  }
	}

	// any remote nodes to notify?
	if(!to_update.empty()) {
	  int npg_cached = num_poisoned_generations.load_acquire();
	  ActiveMessage<EventUpdateMessage> amsg(to_update,
						 poisoned_generations,
						 npg_cached*sizeof(EventImpl::gen_t));
	  amsg->event = make_event(update_gen);
	  amsg.commit();
	}

	// free event?
	if(free_event)
	  get_runtime()->local_event_free_list->free_entry(this);
      } else {
	// we're triggering somebody else's event, so the first thing to do is tell them
	assert(trigger_node == (int)Network::my_node_id);
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
	  AutoLock<> a(mutex);

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
	      if(gen_triggered > gen_subscribed.load()) {
		subscribe_needed = true; // make sure we get that update
		previous_subscribe_gen = gen_subscribed.load();
		gen_subscribed.store(gen_triggered);
	      }
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
	      if(gen_triggered > (gen_subscribed.load() + 1)) {
		subscribe_needed = true;
		previous_subscribe_gen = gen_subscribed.load();
		gen_subscribed.store(gen_triggered);
	      }
	    }

	  // external waiters need to be signalled inside the lock
	  if(has_external_waiters) {
	    has_external_waiters = false;
            // also need external waiter mutex
            AutoLock<KernelMutex> al2(external_waiter_mutex);
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
      if(!to_wake.empty())
	get_runtime()->event_triggerer.trigger_event_waiters(to_wake,
							     poisoned,
							     work_until);
    }

    void GenEventImpl::perform_delayed_free_list_insertion(void)
    {
      bool free_event = false;

      {
	AutoLock<> a(mutex);
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

      log_barrier.info() << "barrier created: " << impl->me << "/" << impl->generation.load()
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
      : generation(0)
      , gen_subscribed(0)
      , has_external_waiters(false)
      , external_waiter_condvar(external_waiter_mutex)
    {
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
      gen_subscribed.store(0);
      first_generation = /*free_generation =*/ 0;
      next_free = 0;
      remote_subscribe_gens.clear();
      remote_trigger_gens.clear();
      base_arrival_count = 0;
      redop = 0;
      initial_value = 0;
      value_capacity = 0;
      final_values = 0;
      generation.store_release(0);
    }

    /*static*/ void BarrierAdjustMessage::handle_message(NodeID sender, const BarrierAdjustMessage &args,
							 const void *data, size_t datalen,
							 TimeLimit work_until)
    {
      log_barrier.info() << "received barrier arrival: delta=" << args.delta
			 << " in=" << args.wait_on << " out=" << args.barrier
			 << " (" << args.barrier.timestamp << ")";
      BarrierImpl *impl = get_runtime()->get_barrier_impl(args.barrier);
      EventImpl::gen_t gen = ID(args.barrier).barrier_generation();
      impl->adjust_arrival(gen, args.delta, args.barrier.timestamp, args.wait_on,
			   args.sender, args.forwarded,
			   datalen ? data : 0, datalen, work_until);
    }

    /*static*/ void BarrierAdjustMessage::send_request(NodeID target, Barrier barrier, int delta, Event wait_on,
						       NodeID sender, bool forwarded,
						       const void *data, size_t datalen)
    {
      ActiveMessage<BarrierAdjustMessage> amsg(target, datalen);
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
      ActiveMessage<BarrierTriggerMessage> amsg(target, datalen);
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

      virtual void event_triggered(bool poisoned, TimeLimit work_until)
      {
	// TODO: handle poison
	assert(poisoned == POISON_FIXME);
	log_barrier.info() << "deferred barrier arrival: " << barrier
			   << " (" << barrier.timestamp << "), delta=" << delta;
	BarrierImpl *impl = get_runtime()->get_barrier_impl(barrier);
	impl->adjust_arrival(ID(barrier).barrier_generation(), delta, barrier.timestamp, Event::NO_EVENT,
			     sender, forwarded,
			     data, datalen, work_until);
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
      NodeID node;
      EventImpl::gen_t trigger_gen, previous_gen;
    };

    // used to adjust a barrier's arrival count either up or down
    // if delta > 0, timestamp is current time (on requesting node)
    // if delta < 0, timestamp says which positive adjustment this arrival must wait for
    void BarrierImpl::adjust_arrival(gen_t barrier_gen, int delta, 
				     Barrier::timestamp_t timestamp, Event wait_on,
				     NodeID sender, bool forwarded,
				     const void *reduce_value, size_t reduce_value_size,
				     TimeLimit work_until)
    {
      Barrier b = make_barrier(barrier_gen, timestamp);
      if(!wait_on.has_triggered()) {
	// deferred arrival

	// only forward deferred arrivals if the precondition is not one that looks like it'll
	//  trigger here first
        if(owner != Network::my_node_id) {
	  ID wait_id(wait_on);
	  int wait_node;
	  if(wait_id.is_event())
	    wait_node = wait_id.event_creator_node();
	  else
	    wait_node = wait_id.barrier_creator_node();
	  if(wait_node != (int)Network::my_node_id) {
	    // let deferral happen on owner node (saves latency if wait_on event
	    //   gets triggered there)
	    //printf("sending deferred arrival to %d for " IDFMT "/%d (" IDFMT "/%d)\n",
	    //       owner, e.id, e.gen, wait_on.id, wait_on.gen);
	    log_barrier.info() << "forwarding deferred barrier arrival: delta=" << delta
			       << " in=" << wait_on << " out=" << b << " (" << timestamp << ")";
	    BarrierAdjustMessage::send_request(owner, b, delta, wait_on,
					       sender, (sender != Network::my_node_id),
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
	AutoLock<> a(mutex);

	bool generation_updated = false;

	// ownership can change, so check it inside the lock
	if(owner != Network::my_node_id) {
	  forward_to_node = owner;
	  break;
	} else {
	  // if this message had to be forwarded to get here, tell the original sender we are the
	  //  new owner
	  if(forwarded && (sender != Network::my_node_id))
	    inform_migration = sender;
	}

	// sanity checks - is this a valid barrier?
	//assert(generation < free_generation);
	assert(base_arrival_count > 0);

	// update whatever generation we're told to
	{
	  assert(barrier_gen > generation.load());
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
	if(barrier_gen == (generation.load() + 1)) {
	  std::map<gen_t, Generation *>::iterator it = generations.begin();
	  while((it != generations.end()) &&
		(it->first == (generation.load() + 1)) &&
		((base_arrival_count + it->second->unguarded_delta) == 0)) {
	    // keep the list of local waiters to wake up once we release the lock
	    local_notifications.absorb_append(it->second->local_waiters);
	    trigger_gen = it->first;
            generation.store_release(it->first);
	    generation_updated = true;

	    delete it->second;
	    generations.erase(it);
	    it = generations.begin();
	  }

	  // if any triggers occurred, figure out
          //  which remote nodes need notifications (i.e. any who subscribed)
	  if(trigger_gen >= barrier_gen) {
	    std::map<unsigned, gen_t>::iterator it = remote_subscribe_gens.begin();
	    while(it != remote_subscribe_gens.end()) {
	      RemoteNotification rn;
	      rn.node = it->first;
	      if(it->second <= trigger_gen) {
		// we have fulfilled the entire subscription
		rn.trigger_gen = it->second;
		std::map<unsigned, gen_t>::iterator to_nuke = it++;
		remote_subscribe_gens.erase(to_nuke);
	      } else {
		// subscription remains valid
		rn.trigger_gen = trigger_gen;
		++it;
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
	     generations.empty() && (gen_subscribed.load() <= generation.load()) &&
	     (redop == 0) &&
             (NodeID(ID(me).barrier_creator_node()) == Network::my_node_id)) {
	    log_barrier.info() << "barrier migration: " << me << " -> " << remote_notifications[0].node;
	    migration_target = remote_notifications[0].node;
	    owner = migration_target;
            // remember that we had up to date information up to this generation so that we don't try to
            //   subscribe to things we already know about
            gen_subscribed.store(generation.load());
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

	  (redop->cpu_apply_excl_fn)(final_values + ((rel_gen - 1) * redop->sizeof_lhs), 0, reduce_value, 0, 1, redop->userdata);
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
          // also need external waiter mutex
          AutoLock<KernelMutex> al2(external_waiter_mutex);
	  external_waiter_condvar.broadcast();
	}
      } while(0);

      if(forward_to_node != (NodeID) -1) {
	Barrier b = make_barrier(barrier_gen, timestamp);
	BarrierAdjustMessage::send_request(forward_to_node, b, delta, Event::NO_EVENT,
					   sender, (sender != Network::my_node_id),
					   reduce_value, reduce_value_size);
	return;
      }

      if(inform_migration != (NodeID) -1) {
	Barrier b = make_barrier(barrier_gen, timestamp);
	BarrierMigrationMessage::send_request(inform_migration, b, Network::my_node_id);
      }

      if(trigger_gen != 0) {
	log_barrier.info() << "barrier trigger: event=" << me << "/" << trigger_gen;

	// notify local waiters first
	if(!local_notifications.empty())
	  get_runtime()->event_triggerer.trigger_event_waiters(local_notifications,
							       POISON_FIXME,
							       work_until);

	// now do remote notifications
	for(std::vector<RemoteNotification>::const_iterator it = remote_notifications.begin();
	    it != remote_notifications.end();
	    it++) {
	  // normally we'll just send a remote waiter data up to the
	  //  generation they asked for - the exception is the target of a
	  //  migration, who must get up to date data
	  gen_t tgt_trigger_gen = (*it).trigger_gen;
	  if((*it).node == migration_target)
	    tgt_trigger_gen = trigger_gen;
	  log_barrier.info() << "sending remote trigger notification: " << me << "/"
			     << (*it).previous_gen << " -> " << tgt_trigger_gen << ", dest=" << (*it).node;
	  void *data = 0;
	  size_t datalen = 0;
	  if(final_values_copy) {
	    data = (char *)final_values_copy + (((*it).previous_gen - oldest_previous) * redop->sizeof_lhs);
	    datalen = (tgt_trigger_gen - (*it).previous_gen) * redop->sizeof_lhs;
	  }
	  BarrierTriggerMessage::send_request((*it).node, me.id, tgt_trigger_gen, (*it).previous_gen,
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
      if(needed_gen <= generation.load_acquire()) return true;

#ifdef BARRIER_HAS_TRIGGERED_DOES_SUBSCRIBE
      // update the subscription (even on the local node), but do a
      //  quick test first to avoid taking a lock if the subscription is
      //  clearly already done
      if(needed_gen > gen_subscribed.load()) {
	// looks like it needs an update - take lock to avoid duplicate
	//  subscriptions
	gen_t previous_subscription;
	bool send_subscription_request = false;
        NodeID cur_owner = (NodeID) -1;
	{
	  AutoLock<> a(mutex);
	  previous_subscription = gen_subscribed.load();
	  if(needed_gen > previous_subscription) {
	    gen_subscribed.store(needed_gen);
	    // test ownership while holding the mutex
	    if(owner != Network::my_node_id) {
	      send_subscription_request = true;
              cur_owner = owner;
            }
	  }
	}

	// if we're not the owner, send subscription if we haven't already
	if(send_subscription_request) {
	  log_barrier.info() << "subscribing to barrier " << make_barrier(needed_gen) << " (prev=" << previous_subscription << ")";
	  BarrierSubscribeMessage::send_request(cur_owner, me.id, needed_gen, Network::my_node_id, false/*!forwarded*/);
	}
      }
#endif

      // whether or not we subscribed, the answer for now is "no"
      return false;
    }

    void BarrierImpl::subscribe(gen_t subscribe_gen)
    {
      // update the subscription (even on the local node), but do a
      //  quick test first to avoid taking a lock if the subscription is
      //  clearly already done
      if(subscribe_gen > gen_subscribed.load()) {
	// looks like it needs an update - take lock to avoid duplicate
	//  subscriptions
	gen_t previous_subscription;
	bool send_subscription_request = false;
        NodeID cur_owner = (NodeID) -1;
	{
	  AutoLock<> a(mutex);
	  previous_subscription = gen_subscribed.load();
	  if(previous_subscription < subscribe_gen) {
	    gen_subscribed.store(subscribe_gen);
	    // test ownership while holding the mutex
	    if(owner != Network::my_node_id) {
	      send_subscription_request = true;
              cur_owner = owner;
            }
	  }
	}

	// if we're not the owner, send subscription if we haven't already
	if(send_subscription_request) {
	  log_barrier.info() << "subscribing to barrier " << make_barrier(subscribe_gen) << " (prev=" << previous_subscription << ")";
	  BarrierSubscribeMessage::send_request(cur_owner, me.id, subscribe_gen, Network::my_node_id, false/*!forwarded*/);
	}
      }
    }
  
    void BarrierImpl::external_wait(gen_t gen_needed, bool& poisoned)
    {
      poisoned = POISON_FIXME;

      // early out for now without taking lock (TODO: fix for poisoning)
      if(gen_needed <= generation.load_acquire())
        return;

      // make sure we're subscribed to a (potentially-remote) trigger
      this->subscribe(gen_needed);

      {
	AutoLock<> a(mutex);

	// wait until the generation has advanced far enough
	while(gen_needed > generation.load()) {
	  has_external_waiters = true;
          // must wait on external_waiter_condvar with external_waiter_mutex
          //  but NOT with base mutex - hand-over-hand lock on the way in,
          //  and then release external_waiter mutex before retaking main
          //  mutex
          external_waiter_mutex.lock();
          mutex.unlock();
	  external_waiter_condvar.wait();
          external_waiter_mutex.unlock();
          mutex.lock();
	}
      }
    }

    bool BarrierImpl::external_timedwait(gen_t gen_needed, bool& poisoned,
					 long long max_ns)
    {
      poisoned = POISON_FIXME;

      // early out for now without taking lock (TODO: fix for poisoning)
      if(gen_needed <= generation.load_acquire()) return true;

      // make sure we're subscribed to a (potentially-remote) trigger
      this->subscribe(gen_needed);

      long long deadline = Clock::current_time_in_nanoseconds() + max_ns;
      {
	AutoLock<> a(mutex);

	// wait until the generation has advanced far enough
	while(gen_needed > generation.load()) {
	  long long now = Clock::current_time_in_nanoseconds();
	  if(now >= deadline)
	    return false;  // trigger has not occurred
	  has_external_waiters = true;
	  // we don't actually care what timedwait returns - we'll recheck
	  //  the generation ourselves
          // must wait on external_waiter_condvar with external_waiter_mutex
          //  but NOT with base mutex - hand-over-hand lock on the way in,
          //  and then release external_waiter mutex before retaking main
          //  mutex
          external_waiter_mutex.lock();
          mutex.unlock();
	  external_waiter_condvar.timedwait(deadline - now);
          external_waiter_mutex.unlock();
          mutex.lock();
	}
      }
      return true;
    }

    bool BarrierImpl::add_waiter(gen_t needed_gen, EventWaiter *waiter/*, bool pre_subscribed = false*/)
    {
      bool trigger_now = false;
      gen_t previous_subscription;
      bool send_subscription_request = false;
      NodeID cur_owner = (NodeID) -1;
      {
	AutoLock<> a(mutex);

	if(needed_gen > generation.load()) {
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

	  // check to see if we need to subscribe
	  if((owner != Network::my_node_id) &&
	     (gen_subscribed.load() < needed_gen)) {
	    previous_subscription = gen_subscribed.load();
	    gen_subscribed.store(needed_gen);
	    send_subscription_request = true;
	    cur_owner = owner;
	  }
	} else {
	  // needed generation has already occurred - trigger this waiter once we let go of lock
	  trigger_now = true;
	}
      }

      if(send_subscription_request) {
	log_barrier.info() << "subscribing to barrier " << make_barrier(needed_gen) << " (prev=" << previous_subscription << ")";
	BarrierSubscribeMessage::send_request(cur_owner, me.id, needed_gen, Network::my_node_id, false/*!forwarded*/);
      }

      if(trigger_now) {
	waiter->event_triggered(POISON_FIXME, TimeLimit::responsive());
      }

      return true;
    }

    bool BarrierImpl::remove_waiter(gen_t needed_gen, EventWaiter *waiter)
    {
      AutoLock<> a(mutex);

      if(needed_gen <= generation.load()) {
	// already triggered, so nothing to remove
	return false;
      }

      // find the right generation - this should not fail
      std::map<gen_t, Generation *>::iterator it = generations.find(needed_gen);
      assert(it != generations.end());
      bool ok = it->second->local_waiters.erase(waiter);
      assert(ok);
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
	AutoLock<> a(impl->mutex);

	// first check - are we even the current owner?
	if(impl->owner != Network::my_node_id) {
	  forward_to_node = impl->owner;
	  break;
	} else {
	  if(args.forwarded) {
	    // our own request wrapped back around can be ignored - we've already added the local waiter
	    if(args.subscriber == Network::my_node_id) {
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
	    assert(it->second > impl->generation.load());
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
	    if(args.subscribe_gen > impl->generation.load())
	      impl->remote_subscribe_gens[args.subscriber] = args.subscribe_gen;
	  }
	}

	// as long as we're not already subscribed to this generation, check to see if
	//  any trigger notifications are needed
	if(!already_subscribed &&
           (impl->generation.load() > impl->first_generation)) {
	  std::map<unsigned, EventImpl::gen_t>::iterator it = impl->remote_trigger_gens.find(args.subscriber);
	  if((it == impl->remote_trigger_gens.end()) ||
             (it->second < impl->generation.load())) {
	    previous_gen = ((it == impl->remote_trigger_gens.end()) ?
			      impl->first_generation :
			      it->second);
	    trigger_gen = impl->generation.load();
	    impl->remote_trigger_gens[args.subscriber] = impl->generation.load();

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
					      args.subscriber, (args.subscriber != Network::my_node_id));
      }

      if(inform_migration != (NodeID) -1) {
	BarrierMigrationMessage::send_request(inform_migration, b, Network::my_node_id);
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
							 const void *data, size_t datalen,
							 TimeLimit work_until)
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
	AutoLock<> a(impl->mutex);

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
	if(args.previous_gen <= impl->generation.load()) {
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

	  if(trigger_gen > impl->generation.load()) {
	    impl->generation.store_release(trigger_gen);
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
			   args.barrier_id, impl->generation.load(), 
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
	  assert(args.previous_gen >= impl->first_generation);
	  memcpy(impl->final_values + ((args.previous_gen -
					impl->first_generation) * impl->redop->sizeof_lhs),
		 data, datalen);
	}

	// external waiters need to be signalled inside the lock
	if(generation_updated && impl->has_external_waiters) {
	  impl->has_external_waiters = false;
          // also need external waiter mutex
          AutoLock<KernelMutex> al2(impl->external_waiter_mutex);
	  impl->external_waiter_condvar.broadcast();
	}
      }

      // with lock released, perform any local notifications
      if(!local_notifications.empty())
	get_runtime()->event_triggerer.trigger_event_waiters(local_notifications,
							     POISON_FIXME,
							     work_until);
    }

    bool BarrierImpl::get_result(gen_t result_gen, void *value, size_t value_size)
    {
      // generation hasn't triggered yet?
      if(result_gen > generation.load_acquire()) return false;

      // take the lock so we can safely see how many results (if any) are on hand
      AutoLock<> al(mutex);

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
	AutoLock<> a(impl->mutex);
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


  ////////////////////////////////////////////////////////////////////////
  //
  // class CompletionQueue
  //

  /*static*/ const CompletionQueue CompletionQueue::NO_QUEUE = { 0 };

  /*static*/ CompletionQueue CompletionQueue::create_completion_queue(size_t max_size)
  {
    CompQueueImpl *cq = get_runtime()->local_compqueue_free_list->alloc_entry();
    // sanity-check that we haven't exhausted the space of cq IDs
    if(get_runtime()->get_compqueue_impl(cq->me) != cq) {
      log_compqueue.fatal() << "completion queue ID space exhausted!";
      abort();
    }
    if(max_size > 0)
      cq->set_capacity(max_size, false /*!resizable*/);
    else
      cq->set_capacity(1024 /*no obvious way to pick this*/, true /*resizable*/);

    log_compqueue.info() << "created completion queue: cq=" << cq->me << " size=" << max_size;
    return cq->me;
  }

  // destroy a completion queue
  void CompletionQueue::destroy(Event wait_on /*= Event::NO_EVENT*/)
  {
    NodeID owner = ID(*this).compqueue_owner_node();

    log_compqueue.info() << "destroying completion queue: cq=" << *this << " wait_on=" << wait_on;

    if(owner == Network::my_node_id) {
      CompQueueImpl *cq = get_runtime()->get_compqueue_impl(*this);

      if(wait_on.has_triggered())
	cq->destroy();
      else
	cq->deferred_destroy.defer(cq, wait_on);
    } else {
      ActiveMessage<CompQueueDestroyMessage> amsg(owner);
      amsg->comp_queue = *this;
      amsg->wait_on = wait_on;
      amsg.commit();
    }
  }

  /*static*/ void CompQueueDestroyMessage::handle_message(NodeID sender,
							  const CompQueueDestroyMessage &msg,
							  const void *data, size_t datalen)
  {
    CompQueueImpl *cq = get_runtime()->get_compqueue_impl(msg.comp_queue);

    if(msg.wait_on.has_triggered())
      cq->destroy();
    else
      cq->deferred_destroy.defer(cq, msg.wait_on);
  }

  // adds an event to the completion queue (once it triggers)
  // non-faultaware version raises a fatal error if the specified 'event'
  //  is poisoned
  void CompletionQueue::add_event(Event event)
  {
    NodeID owner = ID(*this).compqueue_owner_node();

    log_compqueue.info() << "event registered with completion queue: cq=" << *this << " event=" << event;

    if(owner == Network::my_node_id) {
      CompQueueImpl *cq = get_runtime()->get_compqueue_impl(*this);

      cq->add_event(event, false /*!faultaware*/);
    } else {
      ActiveMessage<CompQueueAddEventMessage> amsg(owner);
      amsg->comp_queue = *this;
      amsg->event = event;
      amsg->faultaware = false;
      amsg.commit();
    }
  }

  void CompletionQueue::add_event_faultaware(Event event)
  {
    NodeID owner = ID(*this).compqueue_owner_node();

    log_compqueue.info() << "event registered with completion queue: cq=" << *this << " event=" << event << " (faultaware)";

    if(owner == Network::my_node_id) {
      CompQueueImpl *cq = get_runtime()->get_compqueue_impl(*this);

      cq->add_event(event, true /*faultaware*/);
    } else {
      ActiveMessage<CompQueueAddEventMessage> amsg(owner);
      amsg->comp_queue = *this;
      amsg->event = event;
      amsg->faultaware = true;
      amsg.commit();
    }
  }

  /*static*/ void CompQueueAddEventMessage::handle_message(NodeID sender,
							   const CompQueueAddEventMessage &msg,
							   const void *data, size_t datalen)
  {
    CompQueueImpl *cq = get_runtime()->get_compqueue_impl(msg.comp_queue);

    cq->add_event(msg.event, msg.faultaware);
  }

  // requests up to 'max_events' triggered events to be popped from the
  //  queue and stored in the provided 'events' array (if null, the
  //  identities of the triggered events are discarded)
  // this call returns the actual number of events popped, which may be
  //  zero (this call is nonblocking)
  // when 'add_event_faultaware' is used, any poisoning of the returned
  //  events is not signalled explicitly - the caller is expected to
  //  check via 'has_triggered_faultaware' itself
  size_t CompletionQueue::pop_events(Event *events, size_t max_events)
  {
    NodeID owner = ID(*this).compqueue_owner_node();
    size_t count;

    if(owner == Network::my_node_id) {
      CompQueueImpl *cq = get_runtime()->get_compqueue_impl(*this);

      count = cq->pop_events(events, max_events);
    } else {
      // bounce data off a temp array since we don't know if 'events' is
      //  accessible to an active message handler thread
      Event *ev_copy = 0;
      if(events) {
	ev_copy = reinterpret_cast<Event *>(malloc(max_events * sizeof(Event)));
	assert(ev_copy != 0);
      }
      CompQueueImpl::RemotePopRequest *req = new CompQueueImpl::RemotePopRequest(ev_copy, max_events);

      ActiveMessage<CompQueuePopRequestMessage> amsg(owner);
      amsg->comp_queue =  *this;
      amsg->max_to_pop = max_events;
      amsg->discard_events = (events == 0);
      amsg->request = reinterpret_cast<intptr_t>(req);
      amsg.commit();

      // now wait for a response - no real alternative to blocking here?
      {
	AutoLock<> al(req->mutex);
	while(!req->completed)
	  req->condvar.wait();
      }

      count = req->count;

      delete req;

      if(ev_copy) {
	if(count > 0)
	  memcpy(events, ev_copy, count * sizeof(Event));
	free(ev_copy);
      }
    }

    if(events != 0)
      log_compqueue.info() << "events popped: cq=" << *this << " max=" << max_events << " act=" << count << " events=" << PrettyVector<Event>(events, count);
    else
      log_compqueue.info() << "events popped: cq=" << *this << " max=" << max_events << " act=" << count << " events=(ignored)";

    return count;
  }

  /*static*/ void CompQueuePopRequestMessage::handle_message(NodeID sender,
							     const CompQueuePopRequestMessage &msg,
							     const void *data, size_t datalen)
  {
    CompQueueImpl *cq = get_runtime()->get_compqueue_impl(msg.comp_queue);

    Event *events = 0;
    size_t max_to_pop = msg.max_to_pop;
    if(!msg.discard_events) {
      // we're going to use temp space on the stack, so limit the number of
      //  events to something sane
      if(max_to_pop > 1024)
	max_to_pop = 1024;
      events = reinterpret_cast<Event *>(alloca(max_to_pop * sizeof(Event)));
      assert(events != 0);
    }

    size_t count = cq->pop_events(events, max_to_pop);

    size_t bytes = (msg.discard_events ?
		      0 :
		      count * sizeof(Event));
    ActiveMessage<CompQueuePopResponseMessage> amsg(sender, bytes);
    amsg->count = count;
    amsg->request = msg.request;
    if(bytes > 0)
      amsg.add_payload(events, bytes, PAYLOAD_COPY);
    amsg.commit();
  }

  /*static*/ void CompQueuePopResponseMessage::handle_message(NodeID sender,
							      const CompQueuePopResponseMessage &msg,
							      const void *data, size_t datalen)
  {
    CompQueueImpl::RemotePopRequest *req = reinterpret_cast<CompQueueImpl::RemotePopRequest *>(msg.request);

    {
      AutoLock<> al(req->mutex);
      assert(msg.count <= req->capacity);
      if(req->events) {
	// data expected
	assert(datalen == (msg.count * sizeof(Event)));
	if(msg.count > 0)
	  memcpy(req->events, data, datalen);
      } else {
	// no data expected
	assert(datalen == 0);
      }
      req->count = msg.count;
      req->completed = true;
      req->condvar.broadcast();
    }
  }

  // get an event that, once triggered, guarantees that (at least) one
  //  call to pop_events made since the non-empty event was requested
  //  will return a non-zero number of triggered events
  // once a call to pop_events has been made (by the caller of
  //  get_nonempty_event or anybody else), the guarantee is lost and
  //  a new non-empty event must be requested
  // note that 'get_nonempty_event().has_triggered()' is unlikely to
  //  ever return 'true' if called from a node other than the one that
  //  created the completion queue (i.e. the query at least has the
  //  round-trip network communication latency to deal with) - if polling
  //  on the completion queue is unavoidable, the loop should poll on
  //  pop_events directly
  Event CompletionQueue::get_nonempty_event(void)
  {
    NodeID owner = ID(*this).compqueue_owner_node();

    if(owner == Network::my_node_id) {
      CompQueueImpl *cq = get_runtime()->get_compqueue_impl(*this);

      Event e = cq->get_local_progress_event();
      log_compqueue.info() << "local nonempty event: cq=" << *this << " event=" << e;
      return e;
    } else {
      // we can't reuse progress events safely (because we can't see when pop
      //  calls occur), so make a fresh event each time
      GenEventImpl *ev_impl = GenEventImpl::create_genevent();
      Event e = ev_impl->current_event();

      ActiveMessage<CompQueueRemoteProgressMessage> amsg(owner);
      amsg->comp_queue = *this;
      amsg->progress = e;
      amsg.commit();

      return e;
    }
  }

  /*static*/ void CompQueueRemoteProgressMessage::handle_message(NodeID sender,
								 const CompQueueRemoteProgressMessage &msg,
								 const void *data, size_t datalen)
  {
    CompQueueImpl *cq = get_runtime()->get_compqueue_impl(msg.comp_queue);

    cq->add_remote_progress_event(msg.progress);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class CompQueueImpl
  //

  CompQueueImpl::CompQueueImpl(void)
    : next_free(0)
    , resizable(false)
    , max_events(0)
    , wr_ptr(0), rd_ptr(0), pending_events(0)
    , commit_ptr(0), consume_ptr(0)
    , cur_events(0)
    , completed_events(0)
    , has_progress_events(false)
    , local_progress_event(0)
    , first_free_waiter(0)
    , batches(0)
  {}

  CompQueueImpl::~CompQueueImpl(void)
  {
    AutoLock<> al(mutex);
    assert(pending_events.load() == 0);
    while(batches) {
      CompQueueWaiterBatch *next_batch = batches->next_batch;
      delete batches;
      batches = next_batch;
    }
    if(completed_events)
      free(completed_events);
  }

  void CompQueueImpl::init(CompletionQueue _me, int _owner)
  {
    this->me = _me;
    this->owner = _owner;
  }

  void CompQueueImpl::set_capacity(size_t _max_size, bool _resizable)
  {
    AutoLock<> al(mutex);
    if(resizable)
      assert(cur_events == 0);
    else
      assert(wr_ptr.load() == consume_ptr.load());
    wr_ptr.store(0);
    rd_ptr.store(0);
    pending_events.store(0);
    commit_ptr.store(0);
    consume_ptr.store(0);
    cur_events = 0;
    resizable = _resizable;
    // round up to a power of 2 for easy modulo arithmetic
    max_events = 1;
    while(max_events < _max_size)
      max_events <<= 1;

    void *ptr = malloc(sizeof(Event) * max_events);
    assert(ptr != 0);
    completed_events = reinterpret_cast<Event *>(ptr);
  }

  void CompQueueImpl::destroy(void)
  {
    AutoLock<> al(mutex);
    // ok to have completed events leftover, but no pending events
    assert(pending_events.load() == 0);
    max_events = 0;
    if(completed_events) {
      free(completed_events);
      completed_events = 0;
    }

    get_runtime()->local_compqueue_free_list->free_entry(this);
  }

  void CompQueueImpl::add_event(Event event, bool faultaware)
  {
    bool poisoned = false;

    // special case: NO_EVENT has no impl...
    if(!event.exists()) {
      add_completed_event(event, 0 /*no waiter*/, TimeLimit::responsive());
      return;
    }

    EventImpl *ev_impl = get_runtime()->get_event_impl(event);
    EventImpl::gen_t needed_gen = ID(event).event_generation();

    if(ev_impl->has_triggered(needed_gen, poisoned)) {
      if(poisoned && !faultaware) {
	log_compqueue.fatal() << "cannot enqueue poisoned event: cq=" << me << " event=" << event;
	abort();
      } else
	add_completed_event(event, 0 /*no waiter*/, TimeLimit::responsive());
    } else {
      // we need a free waiter - make some if needed
      CompQueueWaiter *waiter = 0;
      {
	AutoLock<> al(mutex);
	pending_events.fetch_add(1);

	// try to pop a waiter from the free list - this needs to use
	//  CAS to accomodate unsynchronized pushes to the list, but the
	//  mutex we hold prevents any other poppers (so no ABA problem)
	waiter = first_free_waiter.load_acquire();
	while(waiter) {
	  if(first_free_waiter.compare_exchange(waiter, waiter->next_free))
	    break;
	}

	if(waiter) {
	  waiter->next_free = 0;
	} else {
	  // create a batch of waiters, claim the first one and enqueue the others
	  //  in the free list
	  batches = new CompQueueWaiterBatch(this, batches);
	  waiter = &batches->waiters[0];
	  for(size_t i = 1; i < CQWAITER_BATCH_SIZE - 1; i++)
	    batches->waiters[i].next_free = &batches->waiters[i + 1];
	  // CAS for insertion into the free list
	  CompQueueWaiter *old_head = first_free_waiter.load();
	  while(true) {
	    batches->waiters[CQWAITER_BATCH_SIZE - 1].next_free = old_head;
	    if(first_free_waiter.compare_exchange(old_head,
						  &batches->waiters[1]))
	      break;
	  }
	}
      }
      // with the lock released, add the waiter
      waiter->wait_on = event;
      waiter->faultaware = faultaware;
      ev_impl->add_waiter(needed_gen, waiter);
    }
  }

  Event CompQueueImpl::get_local_progress_event(void)
  {
    // non-resizable queues can observe a non-empty queue and return NO_EVENT
    //  without taking the lock
    if(!resizable && (rd_ptr.load() < commit_ptr.load()))
      return Event::NO_EVENT;

    {
      AutoLock<> al(mutex);

      // now that we hold the lock, check emptiness consistent with progress
      //  event information
      if(resizable) {
	if(cur_events > 0) {
	  assert(local_progress_event == 0);
	  return Event::NO_EVENT;
	}
      } else {
	// before we recheck in the non-resizable case, set the
	//  'has_progress_events' flag - this ensures that any pusher we don't
	//  see when we recheck the commit pointer will see the flag and take the
	//  log to handle the progress event we're about to make
	has_progress_events.store(true);
	// commit load has to be fenced to stay after the store above
	if(rd_ptr.load() < commit_ptr.load_fenced())
	  return Event::NO_EVENT;
      }

      // we appear to be empty - get or create the progress event
      if(local_progress_event) {
	ID id(local_progress_event->me);
	id.event_generation() = local_progress_event_gen;
	return id.convert<Event>();
      } else {
	// make a new one
	local_progress_event = GenEventImpl::create_genevent();
	Event e = local_progress_event->current_event();
	// TODO: we probably don't really need this field because it's always local
	local_progress_event_gen = ID(e).event_generation();
	return e;
      }
    }
  }

  void CompQueueImpl::add_remote_progress_event(Event event)
  {
    // if queue is non-empty, we'll just immediately trigger the event
    bool immediate_trigger = false;

    // non-resizable queues can check without even taking the lock
    if(!resizable && (rd_ptr.load() < commit_ptr.load())) {
      immediate_trigger = true;
    } else {
      AutoLock<> al(mutex);

      // now that we hold the lock, check emptiness consistent with progress
      //  event information
      if(resizable) {
	if(cur_events > 0)
	  immediate_trigger = true;
      } else {
	// before we recheck in the non-resizable case, set the
	//  'has_progress_events' flag - this ensures that any pusher we don't
	//  see when we recheck the commit pointer will see the flag and take the
	//  log to handle the remote progress event we're about to add
	has_progress_events.store(true);
	// commit load has to be fenced to stay after the store above
	if(rd_ptr.load() < commit_ptr.load_fenced())
	  immediate_trigger = true;
      }

      if(!immediate_trigger)
	remote_progress_events.push_back(event);
    }

    // lock is released, so we can trigger now if needed
    if(immediate_trigger) {
      // the event is remote, but we know it's a GenEventImpl, so trigger here
      GenEventImpl::trigger(event, false /*!poisoned*/);
    }
  }

  size_t CompQueueImpl::pop_events(Event *events, size_t max_to_pop)
  {
    if(resizable) {
      AutoLock<> al(mutex);
      if((cur_events > 0) && (max_to_pop > 0)) {
	size_t count = std::min(cur_events, max_to_pop);

	// get current offset and advance pointer
	size_t rd_ofs = rd_ptr.fetch_add(count) & (max_events - 1);
	if(events) {
	  // does copy wrap around?	  
	  if((rd_ofs + count) > max_events) {
	    size_t before_wrap = max_events - rd_ofs;
	    // yes, two memcpy's needed
	    memcpy(events, completed_events + rd_ofs,
		   before_wrap * sizeof(Event));
	    memcpy(events + before_wrap, completed_events,
		   (count - before_wrap) * sizeof(Event));
	  } else {
	    // no, single memcpy does the job
	    memcpy(events, completed_events + rd_ofs, count * sizeof(Event));
	  }
	}

	cur_events -= count;

	return count;
      } else
	return 0;
    } else {
      // lock-free version for nonresizable queues
      size_t count, old_rd_ptr;

      // use CAS to move rd_ptr up to (but not past) commit_ptr
      {
	old_rd_ptr = rd_ptr.load();
	while(true) {
	  size_t old_commit = commit_ptr.load_acquire();
	  if(old_rd_ptr == old_commit) {
	    // queue is empty
	    return 0;
	  }
	  count = std::min((old_commit - old_rd_ptr), max_to_pop);	  
	  size_t new_rd_ptr = old_rd_ptr + count;
	  if(rd_ptr.compare_exchange(old_rd_ptr, new_rd_ptr))
	    break; // success
	}
      }

      // get current offset and advance pointer
      size_t rd_ofs = old_rd_ptr & (max_events - 1);

      if(events) {
	// does copy wrap around?	  
	if((rd_ofs + count) > max_events) {
	  size_t before_wrap = max_events - rd_ofs;
	  // yes, two memcpy's needed
	  memcpy(events, completed_events + rd_ofs,
		 before_wrap * sizeof(Event));
	  memcpy(events + before_wrap, completed_events,
		 (count - before_wrap) * sizeof(Event));
	} else {
	  // no, single memcpy does the job
	  memcpy(events, completed_events + rd_ofs, count * sizeof(Event));
	}
      }

      // once we've copied out our events, mark that we've consumed the
      //  entries - this has to happen in the same order as the rd_ptr
      //  bumps though
      while(consume_ptr.load() != old_rd_ptr) { /*pause?*/ }
      size_t check = consume_ptr.fetch_add_acqrel(count);
      assert(check == old_rd_ptr);

      return count;
    }
  }

  void CompQueueImpl::add_completed_event(Event event, CompQueueWaiter *waiter,
					  TimeLimit work_until)
  {
    log_compqueue.info() << "event pushed: cq=" << me << " event=" << event;
    GenEventImpl *local_trigger = 0;
    EventImpl::gen_t local_trigger_gen;
    std::vector<Event> remote_triggers;

    if(resizable) {
      AutoLock<> al(mutex);
      // check for overflow
      if(cur_events >= max_events) {
	// should detect it precisely
	assert(cur_events == max_events);
	size_t new_size = max_events * 2;
	Event *new_events = reinterpret_cast<Event *>(malloc(new_size * sizeof(Event)));
	assert(new_events != 0);
	size_t rd_ofs = rd_ptr.load() & (max_events - 1);
	if(rd_ofs > 0) {
	  // most cases wrap around
	  memcpy(new_events, completed_events + rd_ofs,
		 (cur_events - rd_ofs) * sizeof(Event));
	  memcpy(new_events + (cur_events - rd_ofs), completed_events,
		 rd_ofs * sizeof(Event));
	} else
	  memcpy(new_events, completed_events, cur_events * sizeof(Event));
	free(completed_events);
	completed_events = new_events;
	rd_ptr.store(0);
	wr_ptr.store(cur_events);
	max_events = new_size;
      }

      cur_events++;
      if(waiter != 0) {
	pending_events.fetch_sub(1);
	// add waiter to free list for reuse
	waiter->next_free = first_free_waiter.load();
	// cannot fail since we hold lock
	bool ok = first_free_waiter.compare_exchange(waiter->next_free,
						     waiter);
	assert(ok);
      }

      size_t wr_ofs = wr_ptr.fetch_add(1) & (max_events - 1);
      completed_events[wr_ofs] = event;
      
      // grab things-to-trigger so we can do it outside the lock
      local_trigger = local_progress_event;
      local_trigger_gen = local_progress_event_gen;
      local_progress_event = 0;
      remote_triggers.swap(remote_progress_events);
    } else {
      // lock-free version for non-resizable queues

      // bump the write pointer and check for overflow
      size_t old_consume = consume_ptr.load();
      size_t old_wr_ptr = wr_ptr.fetch_add(1);
      if((old_wr_ptr - old_consume) >= max_events) {
	log_compqueue.fatal() << "completion queue overflow: cq=" << me << " size=" << max_events;
	abort();
      }

      size_t wr_ofs = old_wr_ptr & (max_events - 1);
      completed_events[wr_ofs] = event;

      // bump commit pointer, but respecting order
      while(commit_ptr.load() != old_wr_ptr) { /*pause?*/ }
      size_t check = commit_ptr.fetch_add_acqrel(1);
      assert(check == old_wr_ptr);

      // lock-free insertion of waiter into free list
      if(waiter) {
	pending_events.fetch_sub(1);
	CompQueueWaiter *old_head = first_free_waiter.load();
	while(true) {
	  waiter->next_free = old_head;
	  if(first_free_waiter.compare_exchange(old_head, waiter))
	    break;
	}
      }

      // see if we need to do any triggering - fenced load to keep it after the
      //  update of commit_ptr above
      if(has_progress_events.load_fenced()) {
	// take lock for this
	AutoLock<> al(mutex);
	local_trigger = local_progress_event;
	local_trigger_gen = local_progress_event_gen;
	local_progress_event = 0;
	remote_triggers.swap(remote_progress_events);
	has_progress_events.store(false);
      }
    }

    if(local_trigger) {
      log_compqueue.debug() << "triggering local progress event: cq=" << me << " event=" << local_trigger->current_event();
      local_trigger->trigger(local_trigger_gen, Network::my_node_id,
			     false /*!poisoned*/, work_until);
    }

    if(!remote_triggers.empty())
      for(std::vector<Event>::const_iterator it = remote_triggers.begin();
	  it != remote_triggers.end();
	  ++it) {
	log_compqueue.debug() << "triggering remote progress event: cq=" << me << " event=" << (*it);
	GenEventImpl::trigger(*it, false /*!poisoned*/, work_until);
      }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class CompQueueImpl::DeferredDestroy
  //

  void CompQueueImpl::DeferredDestroy::defer(CompQueueImpl *_cq, Event wait_on)
  {
    cq = _cq;
    EventImpl::add_waiter(wait_on, this);
  }

  void CompQueueImpl::DeferredDestroy::event_triggered(bool poisoned,
						       TimeLimit work_until)
  {
    assert(!poisoned);
    cq->destroy();
  }

  void CompQueueImpl::DeferredDestroy::print(std::ostream& os) const
  {
    os << "deferred completion queue destruction: cq=" << cq->me;
  }

  Event CompQueueImpl::DeferredDestroy::get_finish_event(void) const
  {
    return Event::NO_EVENT;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class CompQueueImpl::RemotePopRequest
  //

  CompQueueImpl::RemotePopRequest::RemotePopRequest(Event *_events,
						    size_t _capacity)
    : condvar(mutex), completed(false)
    , count(0), capacity(_capacity), events(_events)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // class CompQueueImpl::CompQueueWaiter
  //

  void CompQueueImpl::CompQueueWaiter::event_triggered(bool poisoned,
						       TimeLimit work_until)
  {
    if(poisoned && !faultaware) {
      log_compqueue.fatal() << "cannot enqueue poisoned event: cq=" << cq->me << " event=" << wait_on;
      abort();
    } else
      cq->add_completed_event(wait_on, this, work_until);
  }

  void CompQueueImpl::CompQueueWaiter::print(std::ostream& os) const
  {
    os << "completion queue insertion: cq=" << cq->me << " event=" << wait_on;
  }

  Event CompQueueImpl::CompQueueWaiter::get_finish_event(void) const
  {
    return Event::NO_EVENT;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class CompQueueImpl::CompQueueWaiterBatch
  //

  CompQueueImpl::CompQueueWaiterBatch::CompQueueWaiterBatch(CompQueueImpl *cq,
							    CompQueueWaiterBatch *_next)
    : next_batch(_next)
  {
    for(size_t i = 0; i < CQWAITER_BATCH_SIZE; i++)
      waiters[i].cq = cq;
  }

  CompQueueImpl::CompQueueWaiterBatch::~CompQueueWaiterBatch(void)
  {
    // avoid recursive delete - destroyer should walk `next_batch` chain
    //delete next_batch;
  }


  ActiveMessageHandlerReg<EventSubscribeMessage> event_subscribe_message_handler;
  ActiveMessageHandlerReg<EventTriggerMessage> event_trigger_message_handler;
  ActiveMessageHandlerReg<EventUpdateMessage> event_update_message_handler;
  ActiveMessageHandlerReg<BarrierAdjustMessage> barrier_adjust_message_handler;
  ActiveMessageHandlerReg<BarrierSubscribeMessage> barrier_subscribe_message_handler;
  ActiveMessageHandlerReg<BarrierTriggerMessage> barrier_trigger_message_handler;
  ActiveMessageHandlerReg<BarrierMigrationMessage> barrier_migration_message_handler;
  ActiveMessageHandlerReg<CompQueueDestroyMessage> compqueue_destroy_message_handler;
  ActiveMessageHandlerReg<CompQueueAddEventMessage> compqueue_addevent_message_handler;
  ActiveMessageHandlerReg<CompQueueRemoteProgressMessage> compqueue_remoteprogress_message_handler;
  ActiveMessageHandlerReg<CompQueuePopRequestMessage> compqueue_poprequest_message_handler;
  ActiveMessageHandlerReg<CompQueuePopResponseMessage> compqueue_popresponse_message_handler;

}; // namespace Realm
