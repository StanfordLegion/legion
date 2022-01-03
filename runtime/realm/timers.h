/* Copyright 2021 Stanford University, NVIDIA Corporation
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
  class Clock {
  public:
    static double current_time(bool absolute = false);
    static long long current_time_in_microseconds(bool absolute = false);
    static long long current_time_in_nanoseconds(bool absolute = false);

    // the "zero time" is the offset between absolute and relative time on THIS node
    //   and is stored at nanosecond granularity
    static long long get_zero_time(void);
    // set_zero_time() should only be called by the runtime init code
    static void set_zero_time(void);

  protected:
    REALM_INTERNAL_API_EXTERNAL_LINKAGE
    static long long zero_time;
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
    double start_time;
  };
  
}; // namespace Realm

#include "realm/timers.inl"

#endif // ifndef REALM_TIMERS_H

