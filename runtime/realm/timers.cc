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

#include "realm/timers.h"
#include "realm/logging.h"

#if defined(REALM_ON_LINUX) || defined(REALM_ON_FREEBSD)
#include <time.h>
#endif

#ifdef REALM_ON_MACOS
#include <mach/clock.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif

#ifdef REALM_ON_WINDOWS
#include <windows.h>
#include <sysinfoapi.h>
#endif

namespace Realm {

  Logger log_timer("timers");


  ////////////////////////////////////////////////////////////////////////
  //
  // class Clock

  // if set_zero_time() is not called, relative time will equal absolute time
  /*static*/ uint64_t Clock::zero_time = 0;

#ifdef REALM_TIMERS_USE_RDTSC
  /*static*/ bool Clock::cpu_tsc_enabled = false;
#endif

  /*static*/ Clock::TimescaleConverter Clock::native_to_nanoseconds;

  /*static*/ uint64_t Clock::native_time_slower()
  {
#if defined(REALM_ON_LINUX) || defined(REALM_ON_FREEBSD)
    // in a posix world that isn't using the cpu tsc, use CLOCK_REALTIME
    //  (CLOCK_MONOTONIC is better for intervals, but should really use tsc
    //  if you care that much)
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t t = ts.tv_sec;
    t = (t * 1000000000) + ts.tv_nsec;
#endif
#ifdef REALM_ON_MACOS
    // we're not using tsc intrinsics, but still try to use mach_absolute_time
    //  rather than the much-slower mach clock port
    uint64_t t = mach_absolute_time();
#endif
#ifdef REALM_ON_WINDOWS
    // similar to posix, we're not using tsc for some reason, so get the best
    //  non-tsc thing windows can offer
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    uint64_t t = ft.dwHighDateTime;
    t = (t << 32) + ft.dwLowDateTime;
    // windows epoch starts at 1/1/1601, which is overkill and happens to
    //  overflow int64_t, so shift to Unix epoch (1/1/1970)
    t -= uint64_t(10000000) * 86400 * (369 * 365 + 89 /* leap days*/);
    t *= 100;  // FILETIME is in 100ns units
#endif
    return t;
  }

  /*static*/ long long Clock::get_zero_time(void)
  {
    return zero_time;
  }

  /*static*/ void Clock::set_zero_time(void)
  {
    uint64_t native = native_time();
    zero_time = native_to_nanoseconds.convert_forward_absolute(native);
  }

  /*static*/ void Clock::calibrate(int use_cpu_tsc /*1=yes, 0=no, -1=dont care*/,
                                   uint64_t force_cpu_tsc_freq)
  {
#ifdef REALM_TIMERS_USE_RDTSC
    if(use_cpu_tsc != 0) {  // "yes" or "dont care"
      // we want to get two time samples spread by an interesting amount of
      //  real time (TARGET_NANOSECONDS), but we don't know the overhead of
      //  the OS time call so we do iterations with progressively more and
      //  more TSC reads for a single OS call
      uint64_t native1 = 0, native2 = 0, nanoseconds1 = 0, nanoseconds2 = 0;
      unsigned iterations = 0;
      static const uint64_t TARGET_NANOSECONDS = 10000000; // 10ms
#ifdef REALM_ON_MACOS
      clock_serv_t cclock;
      host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
#endif
      while(true) {
        // increase the number of tsc reads we do exponentially
        for(int i = 0; i < (1 << iterations); i++)
          native2 = std::max(native2, raw_cpu_tsc());

        // one OS call of the appropriate type
#if defined(REALM_ON_LINUX) || defined(REALM_ON_FREEBSD)
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        nanoseconds2 = ts.tv_sec;
        nanoseconds2 = (nanoseconds2 * 1000000000) + ts.tv_nsec;
#endif
#ifdef REALM_ON_MACOS
        mach_timespec_t ts;
        clock_get_time(cclock, &ts);
        nanoseconds2 = ts.tv_sec;
        nanoseconds2 = (nanoseconds2 * 1000000000) + ts.tv_nsec;
#endif
#ifdef REALM_ON_WINDOWS
        FILETIME ft;
        GetSystemTimePreciseAsFileTime(&ft);
        nanoseconds2 = ft.dwHighDateTime;
        nanoseconds2 = (nanoseconds2 << 32) + ft.dwLowDateTime;
        // windows epoch starts at 1/1/1601, which is overkill and happens to
        //  overflow int64_t, so shift to Unix epoch (1/1/1970)
        nanoseconds2 -= uint64_t(10000000) * 86400 * (369 * 365 + 89 /* leap days*/);
        nanoseconds2 *= 100;  // FILETIME is in 100ns units
#endif

        if(iterations == 0) {
          native1 = native2;
          nanoseconds1 = nanoseconds2;
        }
        ++iterations;

        // termination case 1 - a forced cpu tsc freq means we only need
        //  one sample
        if(force_cpu_tsc_freq > 0) {
          // supplied freq is ticks/sec = ticks / 1e9 ns
          bool ok = native_to_nanoseconds.set(native1,
                                              nanoseconds1,
                                              native1 + force_cpu_tsc_freq,
                                              nanoseconds1 + 1000000000);
          if(ok) {
            log_timer.debug() << "fixed-freq calibration: native=" << native1
                              << " nanoseconds=" << nanoseconds1
                              << " freq=" << (1e-9 * force_cpu_tsc_freq);
            cpu_tsc_enabled = true;
          } else {
            log_timer.warning() << "fixed-freq calibration failed: native=" << native1
                                << " nanoseconds=" << nanoseconds1
                                << " freq=" << (1e-9 * force_cpu_tsc_freq);
          }
          break;
        }

        // termination case 2 - at least 4 iterations and enough elapsed time
        if((iterations > 3) && (nanoseconds2 >= (nanoseconds1 + TARGET_NANOSECONDS))) {
          // estimate the frequency and rule out things that look silly
          //  (less than 100 MHz or above 100 GHz)
          double est_ghz = 1.0 * (native2 - native1) / (nanoseconds2 - nanoseconds1);
          if((est_ghz >= 0.1) && (est_ghz <= 100.0)) {
            bool ok = native_to_nanoseconds.set(native1,
                                                nanoseconds1,
                                                native2,
                                                nanoseconds2);

            if(ok) {
              log_timer.debug() << "tsc calibration: native=" << native1 << "," << native2
                                << " nanoseconds=" << nanoseconds1 << "," << nanoseconds2
                                << " freq=" << est_ghz;
              cpu_tsc_enabled = true;
            }
          } else {
            log_timer.warning() << "tsc calibration failed: native=" << native1 << "," << native2
                                << " nanoseconds=" << nanoseconds1 << "," << nanoseconds2
                                << " freq=" << est_ghz;
          }
          break;
        }

        // termination case 3 - too many iterations - assume a tsc read
        //  should never be faster than 1ns
        if((uint64_t(1) << iterations) > TARGET_NANOSECONDS) {
          log_timer.warning() << "tsc calibration too fast: native=" << native1 << "," << native2
                              << " nanoseconds=" << nanoseconds1 << "," << nanoseconds2
                              << " iterations=" << iterations;
          break;
        }

        // termination case 4 - time elapsed with too few iterations
        if(nanoseconds2 >= (nanoseconds1 + TARGET_NANOSECONDS)) {
          log_timer.warning() << "tsc calibration too slow: native=" << native1 << "," << native2
                              << " nanoseconds=" << nanoseconds1 << "," << nanoseconds2
                              << " iterations=" << iterations;
          break;
        }
      }
#ifdef REALM_ON_MACOS
      mach_port_deallocate(mach_task_self(), cclock);
#endif

      // if we've got a working tsc, we're done
      if(cpu_tsc_enabled)
        return;
    }
#else
    // fatal error if the user demanded use of the cpu tsc
    if(use_cpu_tsc > 0) {
      log_timer.fatal() << "missing support for CPU time stamp counter mode";
      abort();
    }
#endif

    // non-tsc mode - default is to use an OS-provided source of nanoseconds,
    //  requiring no initialization
#ifdef REALM_ON_MACOS
    // macos is special though - use mach apis to figure out how
    //  mach_absolute_time relates to nanoseconds
    // get a single sample from CALENDAR_CLOCK and the tsc
    mach_timespec_t ts;
    clock_serv_t cclock;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &ts);
    uint64_t abstime = mach_absolute_time();
    mach_port_deallocate(mach_task_self(), cclock);
    uint64_t caltime = ts.tv_sec;
    caltime = (caltime * 100000000) + ts.tv_nsec;
    // now ask mach for the clock ratio and we'll fake the second sample
    mach_timebase_info_data_t info;
    kern_return_t ret = mach_timebase_info(&info);
    if(ret != KERN_SUCCESS) {
      log_timer.fatal() << "unable to get mach timebase";
      abort();
    }
    // ns = numer/denom * ticks -> `denom` ns == `numer` ticks
    bool ok = native_to_nanoseconds.set(abstime,
                                        caltime,
                                        abstime + info.numer,
                                        caltime + info.denom);
    if(!ok) {
      log_timer.fatal() << "mach calibration failed: abstime=" << abstime
                        << " caltime=" << caltime
                        << " ratio=" << info.numer << "/" << info.denom;
      abort();
    }

    log_timer.debug() << "mach calibration: abstime=" << abstime
                      << " caltime=" << caltime
                      << " ratio=" << info.numer << "/" << info.denom;
#endif
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class Clock::TimescaleConverter

  Clock::TimescaleConverter::TimescaleConverter()
    : a_zero(0)
    , b_zero(0)
    , slope_a_to_b(uint64_t(1) << 32)
    , slope_b_to_a(uint64_t(1) << 32)
  {}

  bool Clock::TimescaleConverter::set(uint64_t ta1, uint64_t tb1,
                                      uint64_t ta2, uint64_t tb2)
  {
    // insist that second point is later
    if((ta2 <= ta1) || (tb2 <= tb1)) return false;

    // limit the supported ratio to 2^16 in either direction
    uint64_t a_delta = ta2 - ta1;
    uint64_t b_delta = tb2 - tb1;
    if(((a_delta >> 16) >= b_delta) || ((b_delta >> 16) >= a_delta))
      return false;

    // slopes are 32.32 fixed point - i.e. slope_a_to_b = (2^32 * b) / a
#ifdef REALM_HAS_INT128
    // easy if we have 128-bit integer math
    __int128 a_to_b_128 = (__int128(b_delta) << 32) / a_delta;
    __int128 b_to_a_128 = (__int128(a_delta) << 32) / b_delta;
#ifdef DEBUG_REALM
    assert((a_to_b_128 >= 0) && (a_to_b_128 < LLONG_MAX));
    assert((b_to_a_128 >= 0) && (b_to_a_128 < LLONG_MAX));
#endif
    slope_a_to_b = a_to_b_128;
    slope_b_to_a = b_to_a_128;
#else
    // do this by first computing the integer part (ok due to ratio check):
    //   slope_int = 2^32 * (b/a)
    // and then correcting with the fractional part (which won't overflow)
    //   slope = slope_int + ((b - a * slope_int) / a)
    slope_a_to_b = (b_delta / a_delta) << 32;
    slope_a_to_b += ((b_delta << 32) - a_delta * slope_a_to_b) / a_delta;
    slope_b_to_a = (a_delta / b_delta) << 32;
    slope_b_to_a += ((a_delta << 32) - b_delta * slope_b_to_a) / b_delta;
#endif

    a_zero = ta1;
    b_zero = tb1;

#ifdef DEBUG_REALM
    // sanity-check
    uint64_t tb3 = convert_forward_absolute(ta2);
    uint64_t ta3 = convert_reverse_absolute(tb2);
    int64_t db_plus = convert_forward_delta(a_delta);
    int64_t db_minus = convert_forward_delta(-((int64_t)a_delta));
    int64_t da_plus = convert_reverse_delta(b_delta);
    int64_t da_minus = convert_reverse_delta(-((int64_t)b_delta));
    assert((ta2 == ta3) && (tb2 == tb3) &&
           (db_plus == (int64_t)b_delta) && (-db_minus == (int64_t)b_delta) &&
           (da_plus == (int64_t)a_delta) && (-da_minus == (int64_t)a_delta));
#endif

    return true;
  }


}; // namespace Realm
