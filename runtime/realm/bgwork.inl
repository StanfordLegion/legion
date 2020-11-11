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

// manager for background work that can be performed by available threads

// NOP but useful for IDEs
#include "realm/bgwork.h"

#include "realm/timers.h"

#ifdef REALM_TIMELIMIT_USE_RDTSC
#include <x86intrin.h>
#ifdef __PGI
// PGI's x86intrin doesn't define __rdtsc?
static inline unsigned long long __rdtsc()
{
  unsigned int dummy;
  return __rdtscp(&dummy);
}
#endif
#endif

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class TimeLimit
  //

  inline TimeLimit::TimeLimit()
    :
#ifdef REALM_TIMELIMIT_USE_RDTSC
      limit_rdtsc(~uint64_t(0))
#else
      limit_time(0)
#endif
    , interrupt_flag(0)
  {}

  // these constructors describe a limit in terms of Realm's clock
  /*static*/ inline TimeLimit TimeLimit::absolute(long long absolute_time_in_nsec,
						  atomic<bool> *_interrupt_flag /*= 0*/)
  {
    TimeLimit t;
#ifdef REALM_TIMELIMIT_USE_RDTSC
    t.limit_rdtsc = (__rdtsc() +
		     (((absolute_time_in_nsec - Clock::current_time_in_nanoseconds()) *
		       rdtsc_per_64k_nanoseconds) >> 16));
#else
    t.limit_time = absolute_time_in_nsec;
#endif
    t.interrupt_flag = _interrupt_flag;
    return t;
  }

  /*static*/ inline TimeLimit TimeLimit::relative(long long relative_time_in_nsec,
						  atomic<bool> *_interrupt_flag /*= 0*/)
  {
    TimeLimit t;
#ifdef REALM_TIMELIMIT_USE_RDTSC
    t.limit_rdtsc = (__rdtsc() +
		     ((relative_time_in_nsec *
		       rdtsc_per_64k_nanoseconds) >> 16));
#else
    t.limit_time = Clock::current_time_in_nanoseconds() + relative_time_in_nsec;
#endif
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
#ifdef REALM_TIMELIMIT_USE_RDTSC
    return(((interrupt_flag != 0) && interrupt_flag->load()) ||
	   (__rdtsc() >= limit_rdtsc));
#else
    return(((interrupt_flag != 0) && interrupt_flag->load()) ||
	   ((limit_time > 0) &&
	    (Clock::current_time_in_nanoseconds() >= limit_time)));
#endif
  }

  inline bool TimeLimit::will_expire(long long additional_nsec) const
  {
#ifdef REALM_TIMELIMIT_USE_RDTSC
    return(((interrupt_flag != 0) && interrupt_flag->load()) ||
	   ((__rdtsc() +
	     ((additional_nsec * rdtsc_per_64k_nanoseconds) >> 16)) >=
	    limit_rdtsc));
#else
    return(((interrupt_flag != 0) && interrupt_flag->load()) ||
	   ((limit_time > 0) &&
	    ((Clock::current_time_in_nanoseconds() +
	      additional_nsec) >= limit_time)));
#endif
  }


}; // namespace Realm
