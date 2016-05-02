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

// Event/UserEvent/Barrier implementations for Realm

// nop, but helps IDEs
#include "event_impl.h"

// can't include runtime_impl.h because it's including us, but we need this declaration:
//include "runtime_impl.h"
namespace Realm { 
  extern EventImpl *get_event_impl(Event e);
  extern GenEventImpl *get_genevent_impl(Event e);
  extern BarrierImpl *get_barrier_impl(Event e);
};

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class EventImpl

  inline /*static*/ bool EventImpl::add_waiter(Event needed, EventWaiter *waiter)
  {
    return get_event_impl(needed)->add_waiter(needed.gen, waiter);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GenEventImpl

  inline Event GenEventImpl::current_event(void) const
  {
    Event e = me.convert<Event>();
    e.gen = generation+1;
    return e;
  }

  inline Event GenEventImpl::make_event(Event::gen_t gen) const
  {
    Event e = me.convert<Event>();
    e.gen = gen;
    return e;
  }

  inline /*static*/ void GenEventImpl::trigger(Event e, bool poisoned)
  {
    GenEventImpl *impl = get_genevent_impl(e);
    impl->trigger(e.gen, gasnet_mynode(), poisoned);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class BarrierImpl

  inline Barrier BarrierImpl::current_barrier(Barrier::timestamp_t timestamp /*= 0*/) const
  {
    Barrier b = me.convert<Barrier>();
    b.gen = generation + 1;
    b.timestamp = timestamp;
    return b;
  }

  inline Barrier BarrierImpl::make_barrier(Event::gen_t gen,
					   Barrier::timestamp_t timestamp /*= 0*/) const
  {
    Barrier b = me.convert<Barrier>();
    b.gen = gen;
    b.timestamp = timestamp;
    return b;
  }


}; // namespace Realm

