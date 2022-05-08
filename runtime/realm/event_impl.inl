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

// Event/UserEvent/Barrier implementations for Realm

// nop, but helps IDEs
#include "realm/event_impl.h"

// can't include runtime_impl.h because it's including us, but we need this declaration:
//include "realm/runtime_impl.h"
namespace Realm { 
  extern EventImpl *get_event_impl(Event e);
  extern GenEventImpl *get_genevent_impl(Event e);
  extern BarrierImpl *get_barrier_impl(Event e);
};

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class EventImpl

  inline Event EventImpl::make_event(gen_t gen) const
  {
    ID id(me);
    id.event_generation() = gen;
    return id.convert<Event>();
  }

  inline /*static*/ bool EventImpl::add_waiter(Event needed, EventWaiter *waiter)
  {
    return get_event_impl(needed)->add_waiter(gen_t(ID(needed).event_generation()), waiter);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GenEventImpl

  inline Event GenEventImpl::current_event(void) const
  {
    ID id(me);
    id.event_generation() = this->generation.load() + 1;
    return id.convert<Event>();
  }

  inline /*static*/ void GenEventImpl::trigger(Event e, bool poisoned)
  {
    GenEventImpl *impl = get_genevent_impl(e);
    // ideally the caller would tell us how long we can spend triggering
    //  a large fanout, but in the absence of such information, try to
    //  remain "responsive"
    impl->trigger(gen_t(ID(e).event_generation()), Network::my_node_id,
		  poisoned, TimeLimit::responsive());
  }

  inline /*static*/ void GenEventImpl::trigger(Event e, bool poisoned,
					       TimeLimit work_until)
  {
    GenEventImpl *impl = get_genevent_impl(e);
    impl->trigger(gen_t(ID(e).event_generation()), Network::my_node_id,
		  poisoned, work_until);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class BarrierImpl

  inline Barrier BarrierImpl::current_barrier(Barrier::timestamp_t timestamp /*= 0*/) const
  {
    ID id(me);
    gen_t gen = this->generation.load() + 1;
    if(gen > id.barrier_generation().MAXVAL)
      return Barrier::NO_BARRIER;
    id.barrier_generation() = gen;
    Barrier b = id.convert<Barrier>();
    b.timestamp = timestamp;
    return b;
  }

  inline Barrier BarrierImpl::make_barrier(gen_t gen,
					   Barrier::timestamp_t timestamp /*= 0*/) const
  {
    ID id(me);
    if(gen > id.barrier_generation().MAXVAL)
      return Barrier::NO_BARRIER;
    id.barrier_generation() = gen;
    Barrier b = id.convert<Barrier>();
    b.timestamp = timestamp;
    return b;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class EventWaiter

  inline std::ostream& operator<<(std::ostream& os, const EventWaiter &waiter)
  {
    waiter.print(os);
    return os;
  }


}; // namespace Realm

