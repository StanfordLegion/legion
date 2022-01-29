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

// clocks, timers for Realm

// nop, but helps IDEs
#include "realm/timers.h"

#include "realm/logging.h"

#include <climits>

#ifdef REALM_TIMERS_USE_RDTSC
  #if defined(__i386__) || defined(__x86_64__)
    #if defined(REALM_COMPILER_IS_NVCC)
      // old versions of nvcc have trouble with avx512 intrinsic definitions,
      //  which we cannot avoid in the include below
      // Update 1/13/2022: the issue has been observed even with CUDA 11.2.
      //  as the version check doesn't seem to safely filter out all the buggy
      //  ones, we remove the check.
      #define __rdtsc __builtin_ia32_rdtsc
    #else
      #include <x86intrin.h>
    #endif
  #endif
#endif

// we'll use (non-standard) __int128 if available (gcc, clang, icc, at least)
#if defined(REALM_COMPILER_IS_GCC) || defined(REALM_COMPILER_IS_CLANG) || defined(REALM_COMPILER_IS_ICC)
  // only available in 64-bit builds?
  #if __SIZEOF_POINTER__ >= 8
    #define REALM_HAS_INT128
  #endif
#endif

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class Clock

  inline /*static*/ double Clock::current_time(bool absolute /*= false*/)
  {
    return (current_time_in_nanoseconds(absolute) * 1e-9);
  }
  
  inline /*static*/ long long Clock::current_time_in_microseconds(bool absolute /*= false*/)
  {
    return (current_time_in_nanoseconds(absolute) / 1000);
  }
  
  inline /*static*/ long long Clock::current_time_in_nanoseconds(bool absolute /*= false*/)
  {
    uint64_t native = native_time();
    uint64_t nanoseconds = native_to_nanoseconds.convert_forward_absolute(native);
    if(!absolute)
      nanoseconds -= zero_time;
#ifdef DEBUG_REALM
    // make sure this fits in a long long
    assert(nanoseconds <= uint64_t(LLONG_MAX));
#endif
    return nanoseconds;
  }

  inline /*static*/ uint64_t Clock::native_time()
  {
#ifdef REALM_TIMERS_USE_RDTSC
    if(REALM_LIKELY(cpu_tsc_enabled))
      return raw_cpu_tsc();
#endif
    // fallback
    return native_time_slower();
  }

  // conversion between native and nanoseconds
  inline /*static*/ uint64_t Clock::native_to_nanoseconds_absolute(uint64_t native)
  {
    return native_to_nanoseconds.convert_forward_absolute(native);
  }

  inline /*static*/ uint64_t Clock::nanoseconds_to_native_absolute(uint64_t nanoseconds)
  {
    return native_to_nanoseconds.convert_reverse_absolute(nanoseconds);
  }

  inline /*static*/ int64_t Clock::native_to_nanoseconds_delta(int64_t d_native)
  {
    return native_to_nanoseconds.convert_forward_delta(d_native);
  }

  inline /*static*/ int64_t Clock::nanoseconds_to_native_delta(int64_t d_nanoseconds)
  {
    return native_to_nanoseconds.convert_reverse_delta(d_nanoseconds);
  }

#ifdef REALM_TIMERS_USE_RDTSC
  inline /*static*/ uint64_t Clock::raw_cpu_tsc()
  {
#if defined(__i386__) || defined(__x86_64__)
#ifdef __PGI
      // PGI's x86intrin doesn't define __rdtsc?
      unsigned int dummy;
      return __rdtscp(&dummy);
#else
      return __rdtsc();
#endif
#else
#error Missing cpu tsc reading code!
#endif
  }
#endif


  ////////////////////////////////////////////////////////////////////////
  //
  // class Clock::TimescaleConverter

  inline uint64_t Clock::TimescaleConverter::convert_forward_absolute(uint64_t ta)
  {
#ifdef DEBUG_REALM
    assert(ta >= a_zero);
#endif
    uint64_t rel_ta = ta - a_zero;
    // slope is 32.32 fixed point, so we want:
    //  rel_tb = (rel_ta * slope_a_to_b) >> 32
#ifdef REALM_HAS_INT128
    __int128 rel_tb_128 = (__int128(rel_ta) * slope_a_to_b + (1U << 31)) >> 32;
#ifdef DEBUG_REALM
    assert((rel_tb_128 >= 0) && (rel_tb_128 <= __int128(~uint64_t(0))));
#endif
    uint64_t rel_tb = rel_tb_128;
#else
    // without 128-bit integers, we need three multiplies to get the
    //  partial products we want - first product drops upper 32 bits
    uint64_t rel_tb = rel_ta * (slope_a_to_b >> 32);
    // second product gets upper ta with fractional slope -> integer tb
    rel_tb += ((rel_ta >> 32) * (slope_a_to_b & uint64_t(0xFFFFFFFFULL)));
    // final product computes and rounds fractional part (carefully)
    rel_tb += ((((rel_ta & uint64_t(0xFFFFFFFFULL)) *
                 (slope_a_to_b & uint64_t(0xFFFFFFFFULL))) >> 31) + 1) >> 1;
#endif
    uint64_t tb = rel_tb + b_zero;
#ifdef DEBUG_REALM
    assert(tb >= b_zero);
#endif
    return tb;
  }

  inline uint64_t Clock::TimescaleConverter::convert_reverse_absolute(uint64_t tb)
  {
#ifdef DEBUG_REALM
    assert(tb >= b_zero);
#endif
    uint64_t rel_tb = tb - b_zero;
    // slope is 32.32 fixed point, so we want:
    //  rel_ta = (rel_tb * slope_b_to_a) >> 32
#ifdef REALM_HAS_INT128
    __int128 rel_ta_128 = (__int128(rel_tb) * slope_b_to_a + (1U << 31)) >> 32;
#ifdef DEBUG_REALM
    assert((rel_ta_128 >= 0) && (rel_ta_128 <= __int128(~uint64_t(0))));
#endif
    uint64_t rel_ta = rel_ta_128;
#else
    // without 128-bit integers, we need three multiplies to get the
    //  partial products we want - first product drops upper 32 bits
    uint64_t rel_ta = rel_tb * (slope_b_to_a >> 32);
    // second product gets upper ta with fractional slope -> integer tb
    rel_ta += ((rel_tb >> 32) * (slope_b_to_a & uint64_t(0xFFFFFFFFULL)));
    // final product computes and rounds fractional part (carefully)
    rel_ta += ((((rel_tb & uint64_t(0xFFFFFFFFULL)) *
                 (slope_b_to_a & uint64_t(0xFFFFFFFFULL))) >> 31) + 1) >> 1;
#endif
    uint64_t ta = rel_ta + a_zero;
#ifdef DEBUG_REALM
    assert(ta >= a_zero);
#endif
    return ta;
  }

  inline int64_t Clock::TimescaleConverter::convert_forward_delta(int64_t da)
  {
#ifdef REALM_HAS_INT128
    __int128 rel_db_128 = (__int128(da) * slope_a_to_b + (1U << 31)) >> 32;
    assert((rel_db_128 >= LLONG_MIN) && (rel_db_128 <= LLONG_MAX));
    int64_t db = rel_db_128;
#else
    // pretend da is unsigned and then if it's negative, we fix things up
    //  by subtracting (2^64 * slope) >> 32 == (slope << 32)
    uint64_t uns_da = da;
    uint64_t uns_db = uns_da * (slope_a_to_b >> 32);
    uns_db += ((uns_da >> 32) * (slope_a_to_b & uint64_t(0xFFFFFFFFULL)));
    // final product computes and rounds fractional part (carefully)
    uns_db += ((((uns_da & uint64_t(0xFFFFFFFFULL)) *
                 (slope_a_to_b & uint64_t(0xFFFFFFFFULL))) >> 31) + 1) >> 1;
    if(da < 0)
      uns_db -= (slope_a_to_b << 32);
    int64_t db = uns_db; // is there a way to sanity-check this?
#endif
    return db;
  }

  inline int64_t Clock::TimescaleConverter::convert_reverse_delta(int64_t db)
  {
#ifdef REALM_HAS_INT128
    __int128 rel_da_128 = (__int128(db) * slope_b_to_a + (1U << 31)) >> 32;
    assert((rel_da_128 >= LLONG_MIN) && (rel_da_128 <= LLONG_MAX));
    int64_t da = rel_da_128;
#else
    // pretend db is unsigned and then if it's negative, we fix things up
    //  by subtracting (2^64 * slope) >> 32 == (slope << 32)
    uint64_t uns_db = db;
    uint64_t uns_da = uns_db * (slope_b_to_a >> 32);
    uns_da += ((uns_db >> 32) * (slope_b_to_a & uint64_t(0xFFFFFFFFULL)));
    // final product computes and rounds fractional part (carefully)
    uns_da += ((((uns_db & uint64_t(0xFFFFFFFFULL)) *
                 (slope_b_to_a & uint64_t(0xFFFFFFFFULL))) >> 31) + 1) >> 1;
    if(db < 0)
      uns_da -= (slope_b_to_a << 32);
    int64_t da = uns_da; // is there a way to sanity-check this?
#endif
    return da;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class TimeLimit
  //

  inline TimeLimit::TimeLimit()
    :
      limit_native(~uint64_t(0))
    , interrupt_flag(0)
  {}

  // these constructors describe a limit in terms of Realm's clock
  /*static*/ inline TimeLimit TimeLimit::absolute(long long absolute_time_in_nsec,
						  atomic<bool> *_interrupt_flag /*= 0*/)
  {
    TimeLimit t;
    t.limit_native = Clock::nanoseconds_to_native_absolute(absolute_time_in_nsec);
    t.interrupt_flag = _interrupt_flag;
    return t;
  }

  /*static*/ inline TimeLimit TimeLimit::relative(long long relative_time_in_nsec,
						  atomic<bool> *_interrupt_flag /*= 0*/)
  {
    TimeLimit t;
    t.limit_native = (Clock::native_time() +
                      Clock::nanoseconds_to_native_delta(relative_time_in_nsec));
    t.interrupt_flag = _interrupt_flag;
    return t;
  }

  // often the desired time limit is "idk, something responsive", so
  //  have a common way to pick a completely-made-up number
  /*static*/ inline TimeLimit TimeLimit::responsive()
  {
#ifdef REALM_RESPONSIVE_TIMELIMIT
    return TimeLimit::relative(REALM_RESPONSIVE_TIMELIMIT);
#else
    // go with 10us as a default
    return TimeLimit::relative(10000);
#endif
  }

  inline bool TimeLimit::is_expired() const
  {
    return(((interrupt_flag != 0) && interrupt_flag->load()) ||
	   (Clock::native_time() >= limit_native));
  }

  inline bool TimeLimit::will_expire(long long additional_nsec) const
  {
    return(((interrupt_flag != 0) && interrupt_flag->load()) ||
           ((Clock::native_time() +
             Clock::nanoseconds_to_native_delta(additional_nsec)) >=
	    limit_native));
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Timestamp

  inline TimeStamp::TimeStamp(const char *_message, bool _difference, Logger *_logger /*= 0*/)
    : message(_message), difference(_difference), logger(_logger)
  {
    start_native = Clock::native_time();

    if(!difference) {
      double start_time = 1e-9 * Clock::native_to_nanoseconds_absolute(start_native);
      if(logger)
	logger->info("%s %7.6f", message, start_time);
      else
	printf("%s %7.6f\n", message, start_time);
    }
  }

  inline TimeStamp::~TimeStamp(void)
  {
    if(difference) {
      int64_t interval_native = Clock::native_time() - start_native;
      double interval = 1e-9 * Clock::native_to_nanoseconds_delta(interval_native);

      if(logger)
	logger->info("%s %7.6f", message, interval);
      else
	printf("%s %7.6f\n", message, interval);
    }
  }


}; // namespace Realm

