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

// different compilers spell things different ways - here we define a bunch
//  of macros that contain the which-compiler-am-i tests

// NOTE: none of the macros definitions are really Realm-specific, but all
//  macro names are prefixed with REALM_ to avoid name conflicts

#ifndef REALM_COMPILER_SUPPORT_H
#define REALM_COMPILER_SUPPORT_H

// REALM_COMPILER_IS_GCC   - defined if compiler is really gcc
// REALM_COMPILER_IS_CLANG - defined if compiler is clang
// REALM_COMPILER_IS_ICC   - defined if compiler is icc
// REALM_COMPILER_IS_PGI   - defined if compiler is pgcc/nvcc
// REALM_COMPILER_IS_MSVC  - defined if compiler is MSVC
// REALM_COMPILER_IS_NVCC  - defined if compiler is really nvcc
// REALM_COMPILER_IS_HIPCC - defined if compiler is hipcc
#if defined(__HIPCC__)
  #define REALM_COMPILER_IS_HIPCC
// Detect nvc++ before testing __CUDACC__
#elif defined(__PGIC__) || defined(__NVCOMPILER)
  #define REALM_COMPILER_IS_PGI
#elif defined(__CUDACC__)
  #define REALM_COMPILER_IS_NVCC
#elif defined(__ICC)
  #define REALM_COMPILER_IS_ICC
#elif defined(__clang__)
  #define REALM_COMPILER_IS_CLANG
#elif defined(_MSC_VER)
  #define REALM_COMPILER_IS_MSVC
#elif defined(__GNUC__)
  #define REALM_COMPILER_IS_GCC
#else
  // unknown compiler?
#endif

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
    #ifdef _MSC_VER
      // MSVC doesn't advertise it by default, but always supports at least C++11
      #define REALM_CXX_STANDARD 11
    #else
      #define REALM_CXX_STANDARD 0
    #endif
  #endif
#endif

// REALM_ON_LINUX   - defined if Realm is being built for a Linux host
// REALM_ON_MACOS   - defined if Realm is being built for a macOS host
// REALM_ON_FREEBSD - defined if Realm is being built for a FreeBSD host
// REALM_ON_WINDOWS - defined if Realm is being built for a Windows host
#ifdef __linux__
  #define REALM_ON_LINUX
#endif
#ifdef __MACH__
  #define REALM_ON_MACOS
#endif
#ifdef __FreeBSD__
  #define REALM_ON_FREEBSD
#endif
#ifdef _MSC_VER
  #define REALM_ON_WINDOWS
#endif

// REALM_BIG_ENDIAN - defined if Realm is being built for a big-endian host
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__)
  #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    #define REALM_BIG_ENDIAN
  #endif
#endif

// REALM_CUDA_HD - adds __host__ __device__ iff compiling with NVCC
#if defined (__CUDACC__) || defined (__HIPCC__)
#define REALM_CUDA_HD __host__ __device__
#else
#define REALM_CUDA_HD
#endif

// REALM_THREAD_LOCAL - type modifier for a thread-local variable
#ifdef _MSC_VER
  #define REALM_THREAD_LOCAL __declspec(thread)
#else
  #define REALM_THREAD_LOCAL __thread
#endif

// REALM_ASSERT(cond, message) - abort program if 'cond' is not true
#if defined (__CUDACC__) || defined (__HIPCC__)
#define REALM_ASSERT(cond, message)  assert(cond)
#else
#define REALM_ASSERT(cond, message)  assert((cond) && (message))
#endif

// REALM_LIKELY(expr) - suggest that `expr` is usually true
// REALM_UNLILELY(expr) - suggest that `expr` is usually false
// REALM_EXPECT(expr, expval) - suggest that `expr` usually evaluates to `expval`
#ifdef _MSC_VER
  // no __builtin_expect in MSVC
  #define REALM_LIKELY(expr)         (expr)
  #define REALM_UNLIKELY(expr)       (expr)
  #define REALM_EXPECT(expr, expval) (expr)
#else
  #define REALM_LIKELY(expr)         __builtin_expect((expr), true)
  #define REALM_UNLIKELY(expr)       __builtin_expect((expr), false)
  #define REALM_EXPECT(expr, expval) __builtin_expect((expr), (expval))
#endif

// REALM_ATTR_UNUSED(thing) - indicate that `thing` is unused
#define REALM_ATTR_UNUSED(thing)  thing __attribute__((unused))

// REALM_ATTR_WARN_UNUSED(thing) - ask compiler to warn if result of `thing` is
//  not used
#ifdef _MSC_VER
  // MSVC has _Check_return_, but needs it on the definition too, which makes
  //  it not a drop-in replacement
  #define REALM_ATTR_WARN_UNUSED(thing)  /*_Check_return_*/ thing
#else
  #define REALM_ATTR_WARN_UNUSED(thing)  thing __attribute__((warn_unused_result))
#endif

// REALM_ATTR_PRINTF_FORMAT(thing, stridx, first) - check printf-style args
#ifdef _MSC_VER
  #define REALM_ATTR_PRINTF_FORMAT(thing, stridx, first) thing
#else
  #define REALM_ATTR_PRINTF_FORMAT(thing, stridx, first) \
                          thing __attribute__((format(printf, stridx, first)))
#endif

// REALM_ATTR_NORETURN(thing) - indicates that `thing` never returns
#ifdef _MSC_VER
  #define REALM_ATTR_NORETURN(thing)  __declspec(noreturn) thing
#else
  #define REALM_ATTR_NORETURN(thing)  thing __attribute__((noreturn))
#endif

// REALM_ATTR_DEPRECATED(msg, thing) - indicates `thing` is deprecated, printing
//                                     `msg` at compile time if possible
// REALM_ATTR_DEPRECATED2(msg, thing1, thing2) - above when `thing` has commas in it
#ifdef __ICC
  #define REALM_ATTR_DEPRECATED(msg, thing)				\
                                    thing __attribute__((deprecated))
  #define REALM_ATTR_DEPRECATED2(msg, thing1, thing2)			\
                                    thing1, thing2 __attribute__((deprecated))
#elif defined(_MSC_VER)
  #define REALM_ATTR_DEPRECATED(msg, thing)				\
                                    __declspec(deprecated(msg)) thing
  #define REALM_ATTR_DEPRECATED2(msg, thing1, thing2)			\
                                    __declspec(deprecated(msg)) thing1, thing2
#else
  #define REALM_ATTR_DEPRECATED(msg, thing)				\
                                    thing __attribute__((deprecated(msg)))
  #define REALM_ATTR_DEPRECATED2(msg, thing1, thing2)			\
                                    thing1, thing2 __attribute__((deprecated(msg)))
#endif

// REALM_ALIGNOF(type) - returns the byte alignment required for `type`
#if REALM_CXX_STANDARD >= 11
  #define REALM_ALIGNOF(type)  alignof(type)
#else
  #define REALM_ALIGNOF(type) __alignof__(type)
#endif

// REALM_ALIGNED_TYPE_SAMEAS(newtype, origtype, reftype) - defines type `newtype`
//         that is identical to `origtype` except that it is aligned as `reftype`
#ifdef _MSC_VER
  #define REALM_ALIGNED_TYPE_SAMEAS(newtype, origtype, reftype) \
        typedef __declspec(align(__alignof(reftype))) origtype newtype
#else
  #define REALM_ALIGNED_TYPE_SAMEAS(newtype, origtype, reftype) \
        typedef origtype __attribute__((aligned(__alignof__(reftype)))) newtype
#endif

// REALM_ALIGNED_TYPE_CONST(newtype, origtype, bytes) - defines type `newtype`
//          that is identical to `origtype` except it is aligned to `bytes`
#ifdef _MSC_VER
  #define REALM_ALIGNED_TYPE_CONST(newtype, origtype, bytes) \
        typedef __declspec(align(bytes)) origtype newtype
#else
  #define REALM_ALIGNED_TYPE_CONST(newtype, origtype, bytes) \
        typedef origtype __attribute__((aligned((bytes)+0))) newtype
#endif

// REALM_HAVE_CXXABI_H - defined if <cxxabi.h> is available
#ifndef _MSC_VER
  #define REALM_HAVE_CXXABI_H
#endif

// the following defines are used to communicate which declarations are part
//  of the "official" public Realm API (as opposed to things that need to be in
//  the same header files for C++ reasons) - they also control which symbols
//  appear in the shared library:
// REALM_PUBLIC_API - class/method/variable intended for direct use
//                    in applications
// REALM_INTERNAL_API_EXTERNAL_LINKAGE - method/variable not intended for
//                    direct use in applications but used indirectly and
//                    therefore requiring external linkage
// REALM_INTERNAL_API - method/variable that is not part of the public API
//                    despite being in an otherwise-public class

#ifdef REALM_LIMIT_SYMBOL_VISIBILITY
  #define REALM_PUBLIC_API  __attribute__((visibility("default")))
  #define REALM_INTERNAL_API_EXTERNAL_LINKAGE  __attribute__((visibility("default")))
  #define REALM_INTERNAL_API  __attribute__((visibility("hidden")))
#else
  #define REALM_PUBLIC_API
  #define REALM_INTERNAL_API_EXTERNAL_LINKAGE
  #define REALM_INTERNAL_API
#endif

// compiler-specific hackery

// Microsoft Visual Studio
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS

// MSVC gives a narrowing warning on explicit assignments, which is much too agressive
#pragma warning(disable: 4244 4267)

// MSVC warns if pieces of an explicit template instantiation are split across files
#pragma warning(disable: 4661)

// windows.h by default defines macros called 'min' and 'max' - suppress that
#define NOMINMAX

// ssize_t doesn't exist, so make it
#include <basetsd.h>
typedef SSIZE_T ssize_t;
#endif

#endif
