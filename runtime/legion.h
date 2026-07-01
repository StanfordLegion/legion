/* Copyright 2026 Stanford University, NVIDIA Corporation
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

// decide whether we want C and/or C++ bindings (default matches host language)
//
// each set of bindings has its own include-once ifdef armor, allowing the
//  second set of bindings to be loaded even if the first already has been
#if !defined(LEGION_ENABLE_C_BINDINGS) && !defined(LEGION_DISABLE_C_BINDINGS)
  #ifndef __cplusplus
    #define LEGION_ENABLE_C_BINDINGS
  #endif
#endif
#if !defined(LEGION_ENABLE_CXX_BINDINGS) && !defined(LEGION_DISABLE_CXX_BINDINGS)
  #ifdef __cplusplus
    #define LEGION_ENABLE_CXX_BINDINGS
  #endif
#endif

#ifdef LEGION_ENABLE_C_BINDINGS
#include "legion/bindings/c_bindings.h"
#endif

#ifdef LEGION_ENABLE_CXX_BINDINGS
#ifndef __LEGION_H__
#define __LEGION_H__

/**
 * \mainpage Legion Runtime Documentation
 *
 * This is the main page of the Legion Runtime documentation.
 *
 * @see Legion::Runtime
 */

/**
 * \file legion.h
 * Legion C++ API
 */

#if __cplusplus < 201703L
#error "Legion requires C++17 as the minimum standard version"
#endif

#include "legion/api/config.h"
#include "legion/api/accessors.h"
#include "legion/api/argument_map.h"
#include "legion/api/buffers.h"
#include "legion/api/constraints.h"
#include "legion/api/data.h"
#include "legion/api/exception.h"
#include "legion/api/functors.h"
#include "legion/api/future.h"
#include "legion/api/future_map.h"
#include "legion/api/geometry.h"
#include "legion/api/interop.h"
#include "legion/api/launchers.h"
#include "legion/api/mapping.h"
#include "legion/api/output_region.h"
#include "legion/api/physical_region.h"
#include "legion/api/predicate.h"
#include "legion/api/redop.h"
#include "legion/api/registrars.h"
#include "legion/api/requirements.h"
#include "legion/api/runtime.h"
#include "legion/api/sync.h"
#include "legion/api/transforms.h"
#include "legion/api/values.h"

#endif // __LEGION_H__
#endif // defined LEGION_ENABLE_CXX_BINDINGS
