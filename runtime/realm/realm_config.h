/* Copyright 2017 Stanford University, NVIDIA Corporation
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

// configuration settings that control how Realm is built
// this is expected to become an auto-generated file at some point

#ifndef REALM_CONFIG_H
#define REALM_CONFIG_H

#ifdef REALM_USE_CMAKE
#include "realm_defines.h"
#endif

// if set, uses ucontext.h for user level thread switching, otherwise falls
//  back to POSIX threads
#if !defined(REALM_USE_NATIVE_THREADS) && !defined(__MACH__)
// clang on Mac is generating apparently-broken code in the user thread
//  scheduler, so disable this code path for now
#define REALM_USE_USER_THREADS
#endif

// if set, uses Linux's kernel-level io_submit interface, otherwise uses
//  POSIX AIO for async file I/O
#ifdef __linux__
//define REALM_USE_KERNEL_AIO
#endif

// dynamic loading via dlfcn and a not-completely standard dladdr extension
#define REALM_USE_DLFCN
#define REALM_USE_DLADDR

// can Realm use exceptions to propagate errors back to the profiling interace?
#define REALM_USE_EXCEPTIONS

// Define a prefix for annotating functions for CUDA compilation
#ifdef __CUDACC__
#define __CUDA_HD__ __host__ __device__
#else
#define __CUDA_HD__
#endif

// the Realm operation table is needed if you want to be able to cancel operations
#ifndef REALM_NO_USE_OPERATION_TABLE
#define REALM_USE_OPERATION_TABLE
#endif

// runtime configuration settings
namespace Realm {
  namespace Config {
    // if non-zero, eagerly checks deferred user event triggers for loops up to the
    //  specified limit
    extern int event_loop_detection_limit;
  };
};

#endif
