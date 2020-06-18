/* Copyright 2020 Stanford University, NVIDIA Corporation
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

// different compilers spell things different ways - here we define a bunch
//  of macros that contain the which-compiler-am-i tests

// NOTE: none of the macros definitions are really Realm-specific, but all
//  macro names are prefixed with REALM_ to avoid name conflicts

#ifndef REALM_COMPILER_SUPPORT_H
#define REALM_COMPILER_SUPPORT_H

// REALM_CXX_STANDARD - one of: 0(c++98), 11, 14, 17
//                      if set from outside, remap 98->0 to allow comparisons
//                      if not set from outside, autodetect
#ifdef REALM_CXX_STANDARD
  #if REALM_CXX_STANDARD == 98
    #undef REALM_CXX_STANDARD
    #define REALM_CXX_STANDARD 0
  #endif
#else
  #if __cplusplus >= 201703L
    #define REALM_CXX_STANDARD 17
  #elif __cplusplus >= 201402L
    #define REALM_CXX_STANDARD 14
  #elif __cplusplus >= 201103L
    #define REALM_CXX_STANDARD 11
  #else
    #define REALM_CXX_STANDARD 0
  #endif
#endif

// REALM_ON_LINUX   - defined if Realm is being built for a Linux host
// REALM_ON_MACOS   - defined if Realm is being built for a macOS host
// REALM_ON_FREEBSD - defined if Realm is being built for a FreeBSD host
#ifdef __linux__
  #define REALM_ON_LINUX
#endif
#ifdef __MACH__
  #define REALM_ON_MACOS
#endif
#ifdef __FreeBSD__
  #define REALM_ON_FREEBSD
#endif

// REALM_BIG_ENDIAN - defined if Realm is being built for a big-endian host
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__)
  #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    #define REALM_BIG_ENDIAN
  #endif
#endif

// REALM_CUDA_HD - adds __host__ __device__ iff compiling with NVCC
#ifdef __CUDACC__
#define REALM_CUDA_HD __host__ __device__
#else
#define REALM_CUDA_HD
#endif

// REALM_THREAD_LOCAL - type modifier for a thread-local variable
#define REALM_THREAD_LOCAL __thread

// REALM_ASSERT(cond, message) - abort program if 'cond' is not true
#ifdef __CUDACC__
#define REALM_ASSERT(cond, message)  assert(cond)
#else
#define REALM_ASSERT(cond, message)  assert((cond) && (message))
#endif

// REALM_LIKELY(expr) - suggest that `expr` is usually true
// REALM_UNLILELY(expr) - suggest that `expr` is usually false
// REALM_EXPECT(expr, expval) - suggest that `expr` usually evaluates to `expval`
#define REALM_LIKELY(expr)         __builtin_expect((expr), true)
#define REALM_UNLIKELY(expr)       __builtin_expect((expr), false)
#define REALM_EXPECT(expr, expval) __builtin_expect((expr), (expval))

// REALM_ATTR_UNUSED(thing) - indicate that `thing` is unused
#define REALM_ATTR_UNUSED(thing)  thing __attribute__((unused))

// REALM_ATTR_WARN_UNUSED(thing) - ask compiler to warn if result of `thing` is
//  not used
#define REALM_ATTR_WARN_UNUSED(thing)  thing __attribute__((warn_unused_result))

// REALM_ATTR_PRINTF_FORMAT(thing, stridx, first) - check printf-style args
#define REALM_ATTR_PRINTF_FORMAT(thing, stridx, first) \
                          thing __attribute__((format(printf, stridx, first)))

// REALM_ATTR_NORETURN(thing) - indicates that `thing` never returns
#define REALM_ATTR_NORETURN(thing)  thing __attribute__((noreturn))

// REALM_ATTR_DEPRECATED(msg, thing) - indicates `thing` is deprecated, printing
//                                     `msg` at compile time if possible
#ifdef __ICC
  #define REALM_ATTR_DEPRECATED(msg, thing)				\
                                    thing __attribute__((deprecated))
#else
  #define REALM_ATTR_DEPRECATED(msg, thing)				\
                                    thing __attribute__((deprecated(msg)))
#endif

// REALM_ALIGNOF(type) - returns the byte alignment required for `type`
#if REALM_CXX_STANDARD >= 11
  #define REALM_ALIGNOF(type)  alignof(type)
#else
  #define REALM_ALIGNOF(type) __alignof__(type)
#endif

// REALM_ALIGNED_TYPE_SAMEAS(newtype, origtype, reftype) - defines type `newtype`
//         that is identical to `origtype` except that it is aligned as `reftype`
#define REALM_ALIGNED_TYPE_SAMEAS(newtype, origtype, reftype) \
          typedef origtype __attribute__((aligned(__alignof__(reftype)))) newtype

// REALM_ALIGNED_TYPE_CONST(newtype, origtype, bytes) - defines type `newtype`
//          that is identical to `origtype` except it is aligned to `bytes`
#define REALM_ALIGNED_TYPE_CONST(newtype, origtype, bytes) \
          typedef origtype __attribute__((aligned(bytes))) newtype

// REALM_HAVE_CXXABI_H - defined if <cxxabi.h> is available
#define REALM_HAVE_CXXABI_H

#endif
