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

// reservations for Realm

// NOP but useful for IDEs
#include "realm/reservation.h"

//define REALM_DEBUG_FRSRV_HOLDERS
#ifdef REALM_DEBUG_FRSRV_HOLDERS
#include "realm/threads.h"
#endif

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class FastReservation

  inline Event FastReservation::lock(WaitMode mode /*= SPIN*/)
  {
    return wrlock(mode);
  }

#ifdef REALM_DEBUG_FRSRV_HOLDERS
  struct FastReservationDebugInfo {
    Thread *owner;
    std::map<FastReservation *, int> locks_held;
    std::vector<std::pair<FastReservation *, int> > locks_log;

    static FastReservationDebugInfo *lookup_debuginfo(void);
  };
  namespace ThreadLocal {
    extern __thread FastReservationDebugInfo *frsv_debug;
  };
#endif

  inline Event FastReservation::wrlock(WaitMode mode /*= SPIN*/)
  {
#ifdef REALM_DEBUG_FRSRV_HOLDERS
    if(!ThreadLocal::frsv_debug || 
       (ThreadLocal::frsv_debug->owner != Thread::self()))
      ThreadLocal::frsv_debug = FastReservationDebugInfo::lookup_debuginfo();
    assert(ThreadLocal::frsv_debug->locks_held.count(this) == 0);
    ThreadLocal::frsv_debug->locks_held[this] = -1;
    ThreadLocal::frsv_debug->locks_log.push_back(std::make_pair(this, 1));
#endif

    // the fast case for a writer lock is when it is completely uncontended
    State old_state = 0;
    State new_state = STATE_WRITER;

    bool got_lock = __sync_bool_compare_and_swap(&state, old_state, new_state);
    if(__builtin_expect(got_lock, true))
      return Event::NO_EVENT;

    // contention or some exceptional condition?  take slow path
    Event e = wrlock_slow(mode);
#ifdef REALM_DEBUG_FRSRV_HOLDERS
    if(e.exists()) {
      // didn't actually get the lock
      ThreadLocal::frsv_debug->locks_held.erase(this);
      ThreadLocal::frsv_debug->locks_log.push_back(std::make_pair(this, -1));
    }
#endif
    return e;
  }
  
  inline Event FastReservation::rdlock(WaitMode mode /*= SPIN*/)
  {
#ifdef REALM_DEBUG_FRSRV_HOLDERS
    if(!ThreadLocal::frsv_debug || 
       (ThreadLocal::frsv_debug->owner != Thread::self()))
      ThreadLocal::frsv_debug = FastReservationDebugInfo::lookup_debuginfo();
    if(ThreadLocal::frsv_debug->locks_held.count(this) != 0) {
      // uncomment next line to disallow nested read locks
      //assert(0);
      assert(ThreadLocal::frsv_debug->locks_held[this] > 0);
      ThreadLocal::frsv_debug->locks_held[this]++;
    } else
      ThreadLocal::frsv_debug->locks_held[this] = 1;
    ThreadLocal::frsv_debug->locks_log.push_back(std::make_pair(this, 2));
#endif

    // the fast case for a reader lock is to observe a lack of writers or
    //  base reservation requests, atomically increment the reader count, and
    //  still observe lack of writer/rsrv contention (we check first so that
    //  a pending writer doesn't get interference from new attempted readers)
    // note that a sleeper is ok, as long as it's a reader
    State cur_state = (volatile const State&)state;
    if(__builtin_expect((cur_state & ~(STATE_SLEEPER | STATE_READER_COUNT_MASK)) == 0, 1)) {
      State orig_state = __sync_fetch_and_add(&state, 1);
      if(__builtin_expect((orig_state & ~(STATE_SLEEPER | STATE_READER_COUNT_MASK)) == 0, 1)) {
	return Event::NO_EVENT;
      } else {
	// put the count back before we go down the slow path
	__sync_fetch_and_sub(&state, 1);
      }
    }

    // contention or some exceptional condition?  take slow path
    Event e = rdlock_slow(mode);
#ifdef REALM_DEBUG_FRSRV_HOLDERS
    if(e.exists()) {
      // didn't actually get the lock
      if(ThreadLocal::frsv_debug->locks_held[this] <= 1) {
	assert(ThreadLocal::frsv_debug->locks_held[this] != 0);
	ThreadLocal::frsv_debug->locks_held.erase(this);
      } else
	ThreadLocal::frsv_debug->locks_held[this]--;
      ThreadLocal::frsv_debug->locks_log.push_back(std::make_pair(this, -2));
    }
#endif
    return e;
  }

  inline void FastReservation::unlock(void)
  {
#ifdef REALM_DEBUG_FRSRV_HOLDERS
    if(!ThreadLocal::frsv_debug || 
       (ThreadLocal::frsv_debug->owner != Thread::self()))
      ThreadLocal::frsv_debug = FastReservationDebugInfo::lookup_debuginfo();
    assert(ThreadLocal::frsv_debug->locks_held.count(this) == 1);
    if(ThreadLocal::frsv_debug->locks_held[this] <= 1) {
      assert(ThreadLocal::frsv_debug->locks_held[this] != 0);
      ThreadLocal::frsv_debug->locks_held.erase(this);
    } else
      ThreadLocal::frsv_debug->locks_held[this]--;
    ThreadLocal::frsv_debug->locks_log.push_back(std::make_pair(this, 3));
#endif

    // two fast cases:
    // 1) if STATE_WRITER is set, this is a write lock release and we must
    //   have READER_COUNT=0, SLEEPER=0, BASE_RSRV=0 - WRITER_WAITING is a
    //   don't care (READER_COUNT or SLEEPER is an illegal condition, but
    //   asserts are on slow path)
    // 2) otherwise it's a reader lock and the fast path requires
    //   READER_COUNT>0, WRITER=0, BASE_RSRV=0 - WRITER_WAITING and SLEEPER
    //   are don't cares (READER_COUNT==0 is illegal, flagged in slow path)
    //
    // in both cases we compute the desired new state and do a compare/swap to
    //  avoid races with state changes that would take us off the fast path

    State cur_state = (volatile const State&)state;

    if((cur_state & STATE_WRITER) != 0) {
      if(__builtin_expect((cur_state & (STATE_READER_COUNT_MASK |
					STATE_SLEEPER |
					STATE_BASE_RSRV_WAITING)) == 0, 1)) {
	State new_state = cur_state - STATE_WRITER;
	bool ok = __sync_bool_compare_and_swap(&state, cur_state, new_state);
	if(__builtin_expect(ok, 1))
	  return;
      }
    } else {
      if(__builtin_expect(((cur_state & STATE_READER_COUNT_MASK) != 0) &&
			  ((cur_state & (STATE_WRITER |
					 STATE_BASE_RSRV_WAITING)) == 0), 1)) {
	State new_state = cur_state - 1;
	bool ok = __sync_bool_compare_and_swap(&state, cur_state, new_state);
	if(__builtin_expect(ok, 1))
	  return;
      }
    }
    unlock_slow();
  }

	
}; // namespace Realm

