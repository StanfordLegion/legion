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

// clocks, timers for Realm

#ifndef REALM_TIMERS_H
#define REALM_TIMERS_H

#include "realm/activemsg.h"

#include <stdio.h>
#include <map>

// outside of namespace because 50-letter-long enums are annoying
enum {
  TIME_NONE,
  TIME_KERNEL,
  TIME_COPY,
  TIME_HIGH_LEVEL,
  TIME_LOW_LEVEL,
  TIME_MAPPER,
  TIME_SYSTEM,
  TIME_AM,
};

extern pthread_key_t thread_timer_key;

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

  // DetailedTimer allows for precise timing of what a given thread was doing when (or how long)
  class DetailedTimer {
  public:
    static void init_timers(void);
#ifdef DETAILED_TIMING
    static void clear_timers(bool all_nodes = true);
    static void push_timer(int timer_kind);
    static void pop_timer(void);
    static void roll_up_timers(std::map<int, double>& timers, bool local_only);
    static void report_timers(bool local_only = false);
#else
    static void clear_timers(bool all_nodes = true) {}
    static void push_timer(int timer_kind) {}
    static void pop_timer(void) {}
    static void roll_up_timers(std::map<int, double>& timers, bool local_only) {}
    static void report_timers(bool local_only = false) {}
#endif
    class ScopedPush {
    public:
      ScopedPush(int timer_kind) { push_timer(timer_kind); }
      ~ScopedPush(void) { pop_timer(); }
    };

    static const char* stringify(int level)
    {
      switch (level)
        {
	case TIME_NONE:
	  return "NONE";
	case TIME_KERNEL:
	  return "KERNEL";
	case TIME_COPY:
	  return "COPY";
	case TIME_HIGH_LEVEL:
	  return "HIGH-LEVEL";
	case TIME_LOW_LEVEL:
	  return "LOW-LEVEL";
	case TIME_MAPPER:
	  return "MAPPER";
	case TIME_SYSTEM:
	  return "SYSTEM";
	case TIME_AM:
	  return "ACTV_MESG";
	default:
	  break;
        }
      // We only call this at the end of the run so leaking a little memory isn't too bad
      char *result = new char[16];
      sprintf(result,"%d",level);
      return result;
    }
  };

  // active messages

  struct ClearTimersMessage {
    struct RequestArgs {
      int sender;
      int dummy;
    };

    static void handle_request(RequestArgs args);
     
    typedef ActiveMessageShortNoReply<CLEAR_TIMER_MSGID,
				      RequestArgs,
				      handle_request> Message;

    static void send_request(NodeID target);
  };

  struct TimerDataRequestMessage {
    struct RequestArgs {
      int sender;
      void *rollup_ptr;
    };

    static void handle_request(RequestArgs args);
     
    typedef ActiveMessageShortNoReply<ROLL_UP_TIMER_MSGID,
				      RequestArgs,
				      handle_request> Message;

    static void send_request(NodeID target, void *rollup_ptr);
  };
  
  struct TimerDataResponseMessage {
    struct RequestArgs : public BaseMedium {
      int sender;
      void *rollup_ptr;
    };

    static void handle_request(RequestArgs args, const void *data, size_t datalen);
     
    typedef ActiveMessageMediumNoReply<ROLL_UP_TIMER_MSGID,
			 	       RequestArgs,
				       handle_request> Message;

    static void send_request(NodeID target, void *rollup_ptr,
			     const void *data, size_t datalen, int payload_mode);
  };
  
}; // namespace Realm

#include "realm/timers.inl"

#endif // ifndef REALM_TIMERS_H

