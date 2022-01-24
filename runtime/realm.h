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

// meta-header for Realm - includes all the individual pieces

// decide whether we want C and/or C++ bindings (default matches host language)
//
// each set of bindings has its own include-once ifdef armor, allowing the
//  second set of bindings to be loaded even if the first already has been
#if !defined(REALM_ENABLE_C_BINDINGS) && !defined(REALM_DISABLE_C_BINDINGS)
  #ifndef __cplusplus
    #define REALM_ENABLE_C_BINDINGS
  #endif
#endif
#if !defined(REALM_ENABLE_CXX_BINDINGS) && !defined(REALM_DISABLE_CXX_BINDINGS)
  #ifdef __cplusplus
    #define REALM_ENABLE_CXX_BINDINGS
  #endif
#endif

#ifdef REALM_ENABLE_C_BINDINGS
#include "realm/realm_c.h"
#endif

#ifdef REALM_ENABLE_CXX_BINDINGS
#ifndef REALM_H
#define REALM_H

#include "realm/realm_config.h"

#include "realm/profiling.h"
#include "realm/redop.h"
#include "realm/event.h"
#include "realm/reservation.h"
#include "realm/processor.h"
#include "realm/memory.h"
#include "realm/instance.h"
#include "realm/machine.h"
#include "realm/runtime.h"
#include "realm/indexspace.h"
#include "realm/codedesc.h"
#include "realm/subgraph.h"

#ifdef REALM_USE_HDF5
#include "realm/hdf5/hdf5_access.h"
#endif

#endif // ifndef REALM_H
#endif // ifdef REALM_ENABLE_CXX_BINDINGS

