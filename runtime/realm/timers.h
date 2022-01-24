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

#ifndef REALM_TIMERS_H
#define REALM_TIMERS_H

#include "realm/realm_config.h"

#include "realm/atomics.h"

#include <cstdint>

#if defined(__i386__) || defined(__x86_64__)
#define REALM_TIMERS_USE_RDTSC
#endif

namespace Realm {

  // Clock provides (static) methods for getting the current time, which can be either:
  //  relative (default) - time since the runtime was initialized, synchronized between
  //                         all nodes
  //  absolute           - system time reported by the OS, which may not be well synchronized
  //                         between nodes
  //
  // The time may be requested in one of three units:
  //  seconds - uses a double to store fractional seconds
  //  microseconds - uses a 64-bit integer, no fractional microseconds
  //  nanoseconds - uses a 64-bit integer, no fractional nanoseconds
  //
  // Also provided is the "native" time, which is ideally super-low-overhead
  //  to query (e.g. reading a cpu time stampe counter), but isn't necessarily
  //  in units of nanoseconds, nor is it synchronized between processes
  class REALM_PUBLIC_API Clock {
  public:
    static double current_time(bool absolute = false);
    static long long current_time_in_microseconds(bool absolute = false);
    static long long current_time_in_nanoseconds(bool absolute = false);

    // the "zero time" is the offset between absolute and relative time on THIS node
    //   and is stored at nanosecond granularity
    static long long get_zero_time(void);
    // set_zero_time() should only be called by the runtime init code
    static void set_zero_time(void);

    // inlined version when we're using CPU time stamp counter
    static uint64_t native_time();

    // conversion between native and nanoseconds
    static uint64_t native_to_nanoseconds_absolute(uint64_t native);
    static uint64_t nanoseconds_to_native_absolute(uint64_t nanoseconds);
    static int64_t native_to_nanoseconds_delta(int64_t d_native);
    static int64_t nanoseconds_to_native_delta(int64_t d_nanoseconds);

    // initialization/calibration of timing
    static void calibrate(int use_cpu_tsc /*1=yes, 0=no, -1=dont care*/,
                          uint64_t force_cpu_tsc_freq);

    class TimescaleConverter {
    public:
      // defaults to identity conversion
      TimescaleConverter();

      // learns an affine translation between two timescales based on two
      //  samples from each timescale (ta1 and tb1 should be the "same time",
      //  and ta2 and tb2 should be a (later) "same time")
      // fails if the translation cannot be represented (i.e. if the
      //  time intervals differ by a factor of more than 2^32)
      bool set(uint64_t ta1, uint64_t tb1, uint64_t ta2, uint64_t tb2);

      // conversion of absolute times ("forward" = A->B, "reverse" = B-A)
      uint64_t convert_forward_absolute(uint64_t ta);
      uint64_t convert_reverse_absolute(uint64_t tb);

      // conversion of time deltas (slightly cheaper, but must fit in int64_t)
      int64_t convert_forward_delta(int64_t da);
      int64_t convert_reverse_delta(int64_t db);

    protected:
      uint64_t a_zero, b_zero, slope_a_to_b, slope_b_to_a;
    };

  protected:
#ifdef REALM_TIMERS_USE_RDTSC
    static uint64_t raw_cpu_tsc();
#endif

    // slower function-call version of native_time for platform portability
    static uint64_t native_time_slower();

    static uint64_t zero_time;
    static TimescaleConverter native_to_nanoseconds;
#ifdef REALM_TIMERS_USE_RDTSC
    static bool cpu_tsc_enabled;
#endif
  };

  // a central theme for a responsive runtime is to place limits on how long
  //  is spent doing any one task - this is described using a TimeLimit object
  class TimeLimit {
  public:
    // default constructor generates a time limit infinitely far in the future
    TimeLimit();

    // these constructors describe a limit in terms of Realm's clock
    static TimeLimit absolute(long long absolute_time_in_nsec,
			      atomic<bool> *_interrupt_flag = 0);
    static TimeLimit relative(long long relative_time_in_nsec,
			      atomic<bool> *_interrupt_flag = 0);

    // often the desired time limit is "idk, something responsive", so
    //  have a common way to pick a completely-made-up number
    static TimeLimit responsive();

    bool is_expired() const;
    bool will_expire(long long additional_nsec) const;

  protected:
    uint64_t limit_native;
    atomic<bool> *interrupt_flag;
  };

  class Logger;

  // a Timestamp is a convenient way to record timestamps or measure time spent within a
  //  given C++ scope
  class TimeStamp {
  public:
    TimeStamp(const char *_message, bool _difference, Logger *_logger = 0);
    ~TimeStamp(void);

  protected:
    const char *message;
    bool difference;
    Logger *logger;
    uint64_t start_native;
  };
  
}; // namespace Realm

#include "realm/timers.inl"

#endif // ifndef REALM_TIMERS_H

