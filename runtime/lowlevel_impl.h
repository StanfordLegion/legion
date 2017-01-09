/* Copyright 2017 Stanford University, NVIDIA Corporation
 * Copyright 2017 Los Alamos National Laboratory
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

#ifndef LOWLEVEL_IMPL_H
#define LOWLEVEL_IMPL_H

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
//
//  WARNING: Do not add any new code to this file - its days are numbered.
//   It only exists for code in lowlevel_{dma,disk,gpu}.cc.
//
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


#ifndef MAX_NUM_THREADS
#define MAX_NUM_THREADS 32
#endif

#include "lowlevel.h"

#include "realm/operation.h"
#include "realm/dynamic_table.h"
#include "realm/id.h"
#include "realm/metadata.h"

#include <assert.h>

#include "activemsg.h"

#ifdef CHECK_REENTRANT_MESSAGES
GASNETT_THREADKEY_DEFINE(in_handler);
#endif
GASNETT_THREADKEY_DECLARE(cur_thread);

#if __cplusplus >= 201103L
#define typeof decltype
#endif

namespace Realm {
  class Module;
  class Operation;
  class ProfilingRequestSet;
};

#include "realm/event_impl.h"
#include "realm/rsrv_impl.h"
#include "realm/machine_impl.h"
#include "realm/proc_impl.h"
#include "realm/mem_impl.h"
#include "realm/inst_impl.h"
#include "realm/idx_impl.h"
#include "realm/runtime_impl.h"

// namespace importing for backwards compatibility
namespace LegionRuntime {
  namespace LowLevel {
    typedef Realm::ID ID;
    typedef Realm::EventWaiter EventWaiter;
    typedef Realm::EventImpl EventImpl;
    typedef Realm::GenEventImpl GenEventImpl;
    typedef Realm::BarrierImpl BarrierImpl;
    typedef Realm::ReservationImpl ReservationImpl;
    typedef Realm::MachineImpl MachineImpl;
    typedef Realm::ProcessorImpl ProcessorImpl;
    typedef Realm::ProcessorGroup ProcessorGroup;
    typedef Realm::Task Task;
    typedef Realm::MemoryImpl MemoryImpl;
    typedef Realm::MetadataBase MetadataBase;
    typedef Realm::RegionInstanceImpl RegionInstanceImpl;
    typedef Realm::IndexSpaceImpl IndexSpaceImpl;
    typedef Realm::Node Node;
    typedef Realm::DetailedTimer DetailedTimer;

    inline Realm::RuntimeImpl *get_runtime(void)
    {
      return Realm::get_runtime();
    }
  };
};

#endif
