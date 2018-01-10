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

#ifndef RUNTIME_LOWLEVEL_H
#define RUNTIME_LOWLEVEL_H

#warning This file is deprecated - include "realm/realm.h" instead (and use namespace Realm instead of LegionRuntime::LowLevel).

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
//
//  WARNING: Do not add any new code to this file - its days are numbered.
//   New code should include realm/realm.h or some of its individual pieces.
//
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

#include "legion/arrays.h"
#include "realm/realm_c.h"
#ifdef USE_HDF
#include <hdf5.h>
#endif

// just include the new Realm header and import a bunch of stuff into the
//  old C++ namespace

#include "realm/realm.h"

namespace LegionRuntime {
  namespace LowLevel {
    // importing stuff into this namespace for backwards compatibility
    typedef Realm::ReductionOpID ReductionOpID;
    typedef Realm::ReductionOpUntyped ReductionOpUntyped;
    typedef Realm::CustomSerdezID CustomSerdezID;
    typedef Realm::CustomSerdezUntyped CustomSerdezUntyped;
    typedef Realm::Event Event;
    typedef Realm::UserEvent UserEvent;
    typedef Realm::Barrier Barrier;
    typedef Realm::Reservation Reservation;
    typedef Realm::Processor Processor;
    typedef Realm::Memory Memory;
    typedef Realm::RegionInstance RegionInstance;
    typedef Realm::Machine Machine;
    typedef Realm::Runtime Runtime;

    typedef ::realm_id_t IDType;

  }; // namespace LowLevel
}; // namespace LegionRuntime

#endif
