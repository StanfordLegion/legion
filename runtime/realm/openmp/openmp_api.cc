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

#include "realm/openmp/openmp_threadpool.h"

#include "realm/logging.h"

#include <stdio.h>
#include <stdint.h>

namespace Realm {
  extern Logger log_omp;

  // referred to from openmp_module.cc to force linkage of this file
  void openmp_api_force_linkage(void)
  {}

};

// application-visible OpenMP API calls - always generated
extern "C" {

  using namespace Realm;

  REALM_PUBLIC_API
  int omp_get_num_threads(void)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(wi)
      return wi->num_threads;
    else
      return 1;
  }

  REALM_PUBLIC_API
  int omp_get_num_places(void)
  {
    // Some newer versions of the OpenMP runtime enable control over OpenMP
    // threads ploacement onto different sockets or cores. This sort of thing
    // is controlled by Realm in our implementation, so just return the number
    // of threads allocated to this worker pool.
    return omp_get_num_threads();
  }

  REALM_PUBLIC_API
  int omp_get_max_threads(void)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(wi)
      return wi->pool->get_num_workers() + 1;
    else
      return 1;
  }

  REALM_PUBLIC_API
  int omp_get_num_procs(void)
  {
    return omp_get_max_threads();
  }

  REALM_PUBLIC_API
  int omp_get_thread_num(void)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(wi)
      return wi->thread_id;
    else
      return 0;
  }

  REALM_PUBLIC_API
  int omp_get_level(void)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(wi) {
      int level = 0;
      Realm::ThreadPool::WorkItem *item = wi->work_item;
      while(item) {
	level++;
	item = item->parent_work_item;
      }
      return level;
    } else
      return 0;
  }

  REALM_PUBLIC_API
  int omp_in_parallel(void)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(wi) {
      Realm::ThreadPool::WorkItem *item = wi->work_item;
      if(item) {
	if((wi->num_threads > 1) && (item->single_winner.load() == -1))
	  return 1;
	else
	  return 0;  // single inside parallel
      } else
	return 0;
    } else
      return 0;
  }

  REALM_PUBLIC_API
  void omp_set_num_threads(int num_threads)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(wi) {
      wi->app_num_threads = num_threads; // TODO: connect this to something?
    } else {
      log_omp.warning() << "omp_set_num_threads(" << num_threads << ") called on non-OpenMP Realm proessor - ignoring";
    }
  }
};

// runtime API calls used by compiler to interact with GOMP-style runtime
#ifdef REALM_OPENMP_GOMP_SUPPORT
extern "C" {

  using namespace Realm;

  REALM_PUBLIC_API
  void GOMP_parallel_start(void (*fnptr)(void *data), void *data, int nthreads)
  {
    //printf("GOMP_parallel_start(%p, %p, %d)\n", fnptr, data, nthreads);
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(true);
    if(!wi)
      return;

    std::set<int> worker_ids;
    wi->pool->claim_workers(nthreads - 1, worker_ids);
    int act_threads = 1 + worker_ids.size();

    ThreadPool::WorkItem *work = new ThreadPool::WorkItem(act_threads);
    wi->push_work_item(work);

    wi->thread_id = 0;
    wi->num_threads = act_threads;
    int idx = 1;
    for(std::set<int>::const_iterator it = worker_ids.begin();
	it != worker_ids.end();
	++it) {
      wi->pool->start_worker(*it, idx, act_threads, fnptr, data, work);
      idx++;
    }
    // in GOMP, the master thread runs fnptr itself, so we just return
  }

  REALM_PUBLIC_API
  void GOMP_parallel_end(void)
  {
    //printf("GOMP_parallel_end()\n");
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    // if this isn't called from a openmp-enabled thread, we already
    //  complained above
    if(!wi)
      return;

    ThreadPool::WorkItem *work = wi->pop_work_item();
    assert(work != 0);
    // make sure all workers have finished
    if(work->remaining_workers.fetch_sub_acqrel(1) > 1) {
      log_omp.info() << "waiting for workers to complete";
      while(work->remaining_workers.load_acquire() > 0)
	sched_yield();
    }
    delete work;
  }

  REALM_PUBLIC_API
  void GOMP_parallel(void (*fnptr)(void *data), void *data, unsigned nthreads, unsigned int flags)
  {
    GOMP_parallel_start(fnptr, data, nthreads);
    fnptr(data);
    GOMP_parallel_end();
  }

  REALM_PUBLIC_API
  bool GOMP_single_start(void)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(true);
    if(!wi)
      return true;  // trivially the winner

    if(!wi->work_item) {
      // not inside a larger construct - treat as nop
      return true;
    }

    // try to become the "single" winner - the intent in OpenMP is that the
    //  first worker to get here should do the work
    int prev = -1;
    if(wi->work_item->single_winner.compare_exchange(prev, wi->thread_id)) {
      return true;
    } else {
      // if we were already the winner, that's ok too
      return (prev == wi->thread_id);
    }
  }

  REALM_PUBLIC_API
  void GOMP_barrier(void)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(true);
    if(!wi)
      return;

    //log_omp.print() << "barrier enter: id=" << wi->thread_id;

    if(wi->work_item && (wi->num_threads > 1)) {
      // step 1: observe that barrier is not still being exited
      int c;
      do {
	c = wi->work_item->barrier_count.load();
      } while(c >= wi->num_threads);
      // step 2: increment counter to enter
      c = wi->work_item->barrier_count.fetch_add(1) + 1;
      if(c == wi->num_threads) {
	// last arriver - reset count once all others have exited
	//   reset "single" winner too
	wi->work_item->single_winner.store(-1);
	while(true) {
	  int expval = 2 * wi->num_threads - 1;
	  if(wi->work_item->barrier_count.compare_exchange(expval, 0))
	    break;
	}
      } else {
	// step 3: observe that all threads have entered
	do {
	  c = wi->work_item->barrier_count.load();
	} while(c < wi->num_threads);
	// step 4: increment counter again to exit
	wi->work_item->barrier_count.fetch_add(1);
      }
    } else {
      // not inside a larger construct - nothing to do
    }

    //log_omp.print() << "barrier exit: id=" << wi->thread_id;
  }

  REALM_PUBLIC_API
  bool GOMP_loop_static_start(long start, long end, long incr, long chunk,
			      long *istart, long *iend)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(true);
    if(!wi) {
      // give back the whole loop and hope for the best
      *istart = start;
      *iend = end;
      return (start < end);
    }

    // loops must be inside work items
    assert(wi->work_item != 0);

    log_omp.debug() << "loop static start: start=" << start
		    << " end=" << end << " incr=" << incr
		    << " chunk=" << chunk;

    int64_t span_start, span_end;
    bool more = wi->work_item->schedule.start_static(start, end,
						     incr, chunk,
						     wi->thread_id,
						     span_start,
						     span_end);
    if(more) {
      *istart = span_start;
      *iend = span_end;
    }
    return more;
  }

  REALM_PUBLIC_API
  bool GOMP_loop_dynamic_start(long start, long end, long incr, long chunk,
			       long *istart, long *iend)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(true);
    if(!wi) {
      // give back the whole loop and hope for the best
      *istart = start;
      *iend = end;
      return ((incr > 0) ? (start < end) : (start > end));
    }

    // loops must be inside work items
    assert(wi->work_item != 0);

    log_omp.debug() << "loop dynamic start: start=" << start
		    << " end=" << end << " incr=" << incr
		    << " chunk=" << chunk;

    wi->work_item->schedule.start_dynamic(start, end, incr, chunk);
    int64_t span_start, span_end;
    int64_t stride = 0; // not used
    bool more = wi->work_item->schedule.next_dynamic(span_start, span_end, stride);
    if(more) {
      *istart = span_start;
      *iend = span_end;
    }
    return more;
  }

  REALM_PUBLIC_API
  bool GOMP_loop_ull_dynamic_start(bool up,
                                   uint64_t start, uint64_t end,
                                   uint64_t incr, uint64_t chunk,
                                   uint64_t *istart, uint64_t *iend)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(true);
    if(!wi) {
      // give back the whole loop and hope for the best
      *istart = start;
      *iend = end;
      return (up ? (start < end) : (start > end));
    }

    // loops must be inside work items
    assert(wi->work_item != 0);

    log_omp.debug() << "loop dynamic start: start=" << start
		    << " end=" << end << " incr=" << (int64_t)incr
		    << " chunk=" << chunk;

    // shift values down to int64_t rank for LoopSchedule
    int64_t start_shifted = static_cast<int64_t>(start - (uint64_t(1) << 63));
    int64_t end_shifted = static_cast<int64_t>(end - (uint64_t(1) << 63));

    wi->work_item->schedule.start_dynamic(start_shifted, end_shifted, incr, chunk);
    int64_t span_start, span_end;
    int64_t stride = 0; // not used
    bool more = wi->work_item->schedule.next_dynamic(span_start, span_end, stride);
    if(more) {
      // shift from int64_t back to uint64_t range
      *istart = static_cast<uint64_t>(span_start) + (uint64_t(1) << 63);
      *iend = static_cast<uint64_t>(span_end) + (uint64_t(1) << 63);
    }
    return more;
  }

  REALM_PUBLIC_API
  void GOMP_loop_end_nowait(void)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(!wi)
      return;  // complained already above

    // loops must be inside work items
    assert(wi->work_item != 0);

    log_omp.debug() << "loop end nowait";

    wi->work_item->schedule.end_loop(false /*!wait*/);
  }

  REALM_PUBLIC_API
  void GOMP_loop_end(void)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(!wi)
      return;  // complained already above

    // loops must be inside work items
    assert(wi->work_item != 0);

    log_omp.debug() << "loop end";

    wi->work_item->schedule.end_loop(true /*wait*/);
  }

  REALM_PUBLIC_API
  bool GOMP_loop_static_next(long *istart, long *iend)
  { 
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(!wi)
      return false;  // complained already above

    // loops must be inside work items
    assert(wi->work_item != 0);

    int64_t span_start, span_end;
    bool more = wi->work_item->schedule.next_static(span_start, span_end);

    if(more) {
      *istart = span_start;
      *iend = span_end;
      log_omp.debug() << "loop static next: start=" << *istart
		      << " end=" << *iend;
    } else
      log_omp.debug() << "loop static next: done";

    return more;
  }

  REALM_PUBLIC_API
  bool GOMP_loop_dynamic_next(long *istart, long *iend)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(!wi)
      return false;  // complained already above

    // loops must be inside work items
    assert(wi->work_item != 0);

    log_omp.debug() << "loop dynamic next: pstart=" << *istart
		    << " pend=" << *iend;

    int64_t span_start, span_end, stride;
    bool more = wi->work_item->schedule.next_dynamic(span_start, span_end,
						     stride);

    if(more) {
      *istart = span_start;
      *iend = span_end;
      log_omp.debug() << "loop dynamic next: start=" << *istart
		      << " end=" << *iend;
    } else
      log_omp.debug() << "loop dynamic next: done";

    return more;
  }

  REALM_PUBLIC_API
  bool GOMP_loop_ull_dynamic_next(uint64_t *istart, uint64_t *iend)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(!wi)
      return false;  // complained already above

    // loops must be inside work items
    assert(wi->work_item != 0);

    log_omp.debug() << "loop dynamic next: pstart=" << *istart
		    << " pend=" << *iend;

    int64_t span_start, span_end, stride;
    bool more = wi->work_item->schedule.next_dynamic(span_start, span_end,
						     stride);

    if(more) {
      // shift from int64_t back to uint64_t range
      *istart = static_cast<uint64_t>(span_start) + (uint64_t(1) << 63);
      *iend = static_cast<uint64_t>(span_end) + (uint64_t(1) << 63);
      log_omp.debug() << "loop dynamic next: start=" << *istart
		      << " end=" << *iend;
    } else
      log_omp.debug() << "loop dynamic next: done";

    return more;
  }

  static unsigned hash_gomp_critical_name(void **pptr)
  {
    uintptr_t v = reinterpret_cast<uintptr_t>(pptr);
    // two hashes pushes 24 bits worth of address into the bottom 6
    v ^= (v >> 6);
    v ^= (v >> 12);
    // return bottom 6 bits
    return (v & 63);
  }

  REALM_PUBLIC_API
  void GOMP_critical_name_start(void **pptr)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(!wi || !wi->work_item || (wi->num_threads == 1)) {
      // already single-threaded, so no further effort needed
      return;
    }

    unsigned bit = hash_gomp_critical_name(pptr);
    uint64_t mask = uint64_t(1) << bit;

    // try to set the bit in the critical flags - repeat until success
    while(true) {
      uint64_t orig = wi->work_item->critical_flags.fetch_or(mask);
      if((orig & mask) == 0) break;
      Thread::yield();
    }
  }

  REALM_PUBLIC_API
  void GOMP_critical_name_end(void **pptr)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(!wi || !wi->work_item || (wi->num_threads == 1)) {
      // already single-threaded, so no further effort needed
      return;
    }

    unsigned bit = hash_gomp_critical_name(pptr);
    uint64_t mask = uint64_t(1) << bit;

    // clear the bit - it had better already be set
    uint64_t orig = wi->work_item->critical_flags.fetch_and(~mask);
    assert((orig & mask) != 0);
  }

  REALM_PUBLIC_API
  void GOMP_critical_start(void)
  {
    GOMP_critical_name_start(0);
  }

  REALM_PUBLIC_API
  void GOMP_critical_end(void)
  {
    GOMP_critical_name_end(0);
  }

  // GOMP_atomic_{start,end} just take/release a global lock - not great for
  //  performance, but compilers seem to only use it when they have no other
  //  choice
  Mutex gomp_atomic_mutex;

  REALM_PUBLIC_API
  void GOMP_atomic_start(void)
  {
    gomp_atomic_mutex.lock();
  }

  REALM_PUBLIC_API
  void GOMP_atomic_end(void)
  {
    gomp_atomic_mutex.unlock();
  }
};
#endif

// runtime API calls used by compiler to interact with kmp(intel)-style runtime
#ifdef REALM_OPENMP_KMP_SUPPORT
namespace Realm {

  typedef int32_t kmp_int32;
  typedef uint32_t kmp_uint32;
  typedef int64_t kmp_int64;
  typedef uint64_t kmp_uint64;
  typedef void (*kmpc_micro)(kmp_int32 *global_tid, kmp_int32 *local_tid, ...);
  typedef struct ident ident_t;
  typedef void (*kmpc_reduce)(void *lhs_data, void *rhs_data);
  typedef int32_t kmp_critical_name;

  struct kmp_thunk {
    kmpc_micro microtask;
    std::vector<void *> argv;

    static void invoke_0(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid);
    }

    static void invoke_1(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0]);
    }

    static void invoke_2(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1]);
    }

    static void invoke_3(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2]);
    }

    static void invoke_4(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3]);
    }

    static void invoke_5(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4]);
    }

    static void invoke_6(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5]);
    }

    static void invoke_7(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6]);
    }

    static void invoke_8(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7]);
    }

    static void invoke_9(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8]);
    }

    static void invoke_10(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9]);
    }

    static void invoke_11(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10]);
    }

    static void invoke_12(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11]);
    }

    static void invoke_13(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12]);
    }

    static void invoke_14(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13]);
    }

    static void invoke_15(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14]);
    }

    static void invoke_16(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15]);
    }

    static void invoke_17(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15],
			 thunk->argv[16]);
    }

    static void invoke_18(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15],
			 thunk->argv[16], thunk->argv[17]);
    }

    static void invoke_19(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15],
			 thunk->argv[16], thunk->argv[17],
			 thunk->argv[18]);
    }

    static void invoke_20(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15],
			 thunk->argv[16], thunk->argv[17],
			 thunk->argv[18], thunk->argv[19]);
    }

    static void invoke_21(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15],
			 thunk->argv[16], thunk->argv[17],
			 thunk->argv[18], thunk->argv[19],
			 thunk->argv[20]);
    }

    static void invoke_22(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15],
			 thunk->argv[16], thunk->argv[17],
			 thunk->argv[18], thunk->argv[19],
			 thunk->argv[20], thunk->argv[21]);
    }

    static void invoke_23(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15],
			 thunk->argv[16], thunk->argv[17],
			 thunk->argv[18], thunk->argv[19],
			 thunk->argv[20], thunk->argv[21],
			 thunk->argv[22]);
    }

    static void invoke_24(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15],
			 thunk->argv[16], thunk->argv[17],
			 thunk->argv[18], thunk->argv[19],
			 thunk->argv[20], thunk->argv[21],
			 thunk->argv[22], thunk->argv[23]);
    }

    static void invoke_25(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15],
			 thunk->argv[16], thunk->argv[17],
			 thunk->argv[18], thunk->argv[19],
			 thunk->argv[20], thunk->argv[21],
			 thunk->argv[22], thunk->argv[23],
			 thunk->argv[24]);
    }

    static void invoke_26(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15],
			 thunk->argv[16], thunk->argv[17],
			 thunk->argv[18], thunk->argv[19],
			 thunk->argv[20], thunk->argv[21],
			 thunk->argv[22], thunk->argv[23],
			 thunk->argv[24], thunk->argv[25]);
    }

    static void invoke_27(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15],
			 thunk->argv[16], thunk->argv[17],
			 thunk->argv[18], thunk->argv[19],
			 thunk->argv[20], thunk->argv[21],
			 thunk->argv[22], thunk->argv[23],
			 thunk->argv[24], thunk->argv[25],
			 thunk->argv[26]);
    }

    static void invoke_28(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15],
			 thunk->argv[16], thunk->argv[17],
			 thunk->argv[18], thunk->argv[19],
			 thunk->argv[20], thunk->argv[21],
			 thunk->argv[22], thunk->argv[23],
			 thunk->argv[24], thunk->argv[25],
			 thunk->argv[26], thunk->argv[27]);
    }

    static void invoke_29(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15],
			 thunk->argv[16], thunk->argv[17],
			 thunk->argv[18], thunk->argv[19],
			 thunk->argv[20], thunk->argv[21],
			 thunk->argv[22], thunk->argv[23],
			 thunk->argv[24], thunk->argv[25],
			 thunk->argv[26], thunk->argv[27],
			 thunk->argv[28]);
    }

    static void invoke_30(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15],
			 thunk->argv[16], thunk->argv[17],
			 thunk->argv[18], thunk->argv[19],
			 thunk->argv[20], thunk->argv[21],
			 thunk->argv[22], thunk->argv[23],
			 thunk->argv[24], thunk->argv[25],
			 thunk->argv[26], thunk->argv[27],
			 thunk->argv[28], thunk->argv[29]);
    }

    static void invoke_31(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15],
			 thunk->argv[16], thunk->argv[17],
			 thunk->argv[18], thunk->argv[19],
			 thunk->argv[20], thunk->argv[21],
			 thunk->argv[22], thunk->argv[23],
			 thunk->argv[24], thunk->argv[25],
			 thunk->argv[26], thunk->argv[27],
			 thunk->argv[28], thunk->argv[29],
			 thunk->argv[30]);
    }

    static void invoke_32(void *data)
    {
      const kmp_thunk *thunk = (const kmp_thunk *)data;
      kmp_int32 global_tid = 44; // what does this do?
      kmp_int32 local_tid = omp_get_thread_num(); // CHECK
      (thunk->microtask)(&global_tid, &local_tid,
			 thunk->argv[0], thunk->argv[1],
			 thunk->argv[2], thunk->argv[3],
			 thunk->argv[4], thunk->argv[5],
			 thunk->argv[6], thunk->argv[7],
			 thunk->argv[8], thunk->argv[9],
			 thunk->argv[10], thunk->argv[11],
			 thunk->argv[12], thunk->argv[13],
			 thunk->argv[14], thunk->argv[15],
			 thunk->argv[16], thunk->argv[17],
			 thunk->argv[18], thunk->argv[19],
			 thunk->argv[20], thunk->argv[21],
			 thunk->argv[22], thunk->argv[23],
			 thunk->argv[24], thunk->argv[25],
			 thunk->argv[26], thunk->argv[27],
			 thunk->argv[28], thunk->argv[29],
			 thunk->argv[30], thunk->argv[31]);
    }
  };

};

extern "C" {

  using namespace Realm;

  REALM_PUBLIC_API
  void __kmpc_begin(ident_t *loc, kmp_int32 flags)
  {
    // do nothing
  }

  REALM_PUBLIC_API
  void __kmpc_end(ident_t *loc)
  {
    // do nothing
  }

  REALM_PUBLIC_API
  kmp_int32 __kmpc_ok_to_fork(ident_t *loc)
  {
    // this is supposed to always return TRUE
    return 1;
  }

  REALM_PUBLIC_API
  kmp_int32 __kmpc_global_thread_num(ident_t *loc)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(wi) {
      kmp_int32 gid = wi->thread_id;
      const ThreadPool::WorkItem *item = wi->work_item;
      if(item != 0)
	while(item->parent_work_item != 0) {
	  gid = item->prev_thread_id;
	  item = item->parent_work_item;
	}
      return gid;
    } else
      return 0;
  }

  REALM_PUBLIC_API
  void __kmpc_push_num_threads(ident_t *loc, kmp_int32 global_tid,
                               kmp_int32 num_threads)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(wi) {
      wi->app_num_threads = num_threads; // TODO: connect this to something?
    } else {
      log_omp.warning() << "__kmpc_push_num_threads(" << num_threads << ") called on non-OpenMP Realm proessor - ignoring";
    }
  }

  REALM_PUBLIC_API
  void __kmpc_fork_call(ident_t *loc, kmp_int32 argc, kmpc_micro microtask, ...)
  {
    //printf("kmpc_fork(%p, %d)\n", loc, argc);

    // capture variable length arguments into thunk
    kmp_thunk thunk;
    thunk.microtask = microtask;
    {
      va_list ap;
      va_start(ap, microtask);
      for(int i = 0; i < argc; i++)
	thunk.argv.push_back(va_arg(ap, void *));
      va_end(ap);
    }
    void (*invoker)(void *data);
    // there doesn't seem to be a limit on how many args that the compiler
    //  will want to pass directly into the microthunk, but there's also
    //  no portable way to handle a dynamic count in one invoker (because
    //  the invoker needs to look up the thread index in each thread), so
    //  we have lots of choices and hope it's enough...
    switch(argc) {
    case  0: invoker = &kmp_thunk::invoke_0; break;
    case  1: invoker = &kmp_thunk::invoke_1; break;
    case  2: invoker = &kmp_thunk::invoke_2; break;
    case  3: invoker = &kmp_thunk::invoke_3; break;
    case  4: invoker = &kmp_thunk::invoke_4; break;
    case  5: invoker = &kmp_thunk::invoke_5; break;
    case  6: invoker = &kmp_thunk::invoke_6; break;
    case  7: invoker = &kmp_thunk::invoke_7; break;
    case  8: invoker = &kmp_thunk::invoke_8; break;
    case  9: invoker = &kmp_thunk::invoke_9; break;
    case 10: invoker = &kmp_thunk::invoke_10; break;
    case 11: invoker = &kmp_thunk::invoke_11; break;
    case 12: invoker = &kmp_thunk::invoke_12; break;
    case 13: invoker = &kmp_thunk::invoke_13; break;
    case 14: invoker = &kmp_thunk::invoke_14; break;
    case 15: invoker = &kmp_thunk::invoke_15; break;
    case 16: invoker = &kmp_thunk::invoke_16; break;
    case 17: invoker = &kmp_thunk::invoke_17; break;
    case 18: invoker = &kmp_thunk::invoke_18; break;
    case 19: invoker = &kmp_thunk::invoke_19; break;
    case 20: invoker = &kmp_thunk::invoke_20; break;
    case 21: invoker = &kmp_thunk::invoke_21; break;
    case 22: invoker = &kmp_thunk::invoke_22; break;
    case 23: invoker = &kmp_thunk::invoke_23; break;
    case 24: invoker = &kmp_thunk::invoke_24; break;
    case 25: invoker = &kmp_thunk::invoke_25; break;
    case 26: invoker = &kmp_thunk::invoke_26; break;
    case 27: invoker = &kmp_thunk::invoke_27; break;
    case 28: invoker = &kmp_thunk::invoke_28; break;
    case 29: invoker = &kmp_thunk::invoke_29; break;
    case 30: invoker = &kmp_thunk::invoke_30; break;
    case 31: invoker = &kmp_thunk::invoke_31; break;
    case 32: invoker = &kmp_thunk::invoke_32; break;
    default:
      {
	fprintf(stderr, "HELP!  __kmpc_fork_call called with argc == %d\n", argc);
	assert(0);
      }
    }

    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(true);
    if(!wi) {
      // invoke thunk directly and then return
      (*invoker)(&thunk);
      return;
    }

    std::set<int> worker_ids;
    if(wi->app_num_threads != 1)
      wi->pool->claim_workers(wi->app_num_threads - 1, worker_ids);
    int act_threads = 1 + worker_ids.size();

    ThreadPool::WorkItem *work = new ThreadPool::WorkItem(act_threads);
    wi->push_work_item(work);

    wi->thread_id = 0;
    wi->num_threads = act_threads;
    int idx = 1;
    for(std::set<int>::const_iterator it = worker_ids.begin();
	it != worker_ids.end();
	++it) {
      wi->pool->start_worker(*it, idx, act_threads, invoker, &thunk, work);
      idx++;
    }

    // in kmp version, we invoke the thunk for the master ourselves
    (*invoker)(&thunk);

    // and then we immediately clean things up (c.f. GOMP_parallel_end)
    ThreadPool::WorkItem *work2 = wi->pop_work_item();
    assert(work == work2);
    // make sure all workers have finished
    if(work->remaining_workers.fetch_sub_acqrel(1) > 1) {
      log_omp.info() << "waiting for workers to complete";
      while(work->remaining_workers.load_acquire() > 0)
	sched_yield();
    }
    delete work;
  }

};

namespace Realm {

  // templated code for __kmpc_for_static_init_{4,4u,8,8u}
  template <typename T>
  static inline void kmpc_for_static_init(ident_t *loc, kmp_int32 global_tid,
					  kmp_int32 schedtype,
					  kmp_int32 *plastiter,
					  T *plower, T *pupper,
					  T *pstride,
					  typename make_signed<T>::type incr,
					  typename make_signed<T>::type chunk)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(true);
    // if running sequentially, or bounds are inside-out,
    //  just return with the bounds unchanged
    if(!wi ||
       (wi->num_threads == 1) ||
       ((incr > 0) && (*plower > *pupper)) ||
       ((incr < 0) && (*plower < *pupper))) {
      *pstride = *pupper - *plower + ((incr > 0) ? 1 : -1);
      return;
    }

    //printf("static_init(%p, %d, %d)\n", loc, global_tid, schedtype);
    switch(schedtype) {
    case 34 /* kmp_sch_static */:
    case 33 /* kmp_sch_static_chunked - (chunk ignored - TODO) */:
      {
	T iters;
	if(incr > 0) {
	  iters = 1 + (*pupper - *plower) / incr;
          *pstride = *pupper - *plower + 1;
	} else {
	  iters = 1 + (*plower - *pupper) / -incr;
          *pstride = *pupper - *plower - 1;
	}
	T whole = iters / wi->num_threads;
	T leftover = iters - (whole * wi->num_threads);
	*plower += incr * (whole * wi->thread_id +
			   ((((T)(wi->thread_id)) < leftover) ? wi->thread_id : leftover));
	*pupper = *plower + incr * (whole +
				    ((((T)(wi->thread_id)) < leftover) ? 1 : 0)) - 1;
	// special case for when some threads get no iterations at all
	*plastiter = ((whole > 0) ? (wi->thread_id == (wi->num_threads - 1)) :
		                    (((T)(wi->thread_id)) == (leftover - 1)));
        //log_omp.print() << "static: " << *plower << " " << *pupper << " " << *pstride
        //                << " " << incr << " " << *plastiter;
	return;
      }

    default: assert(false);
    }
  }

  // templated code for __kmpc_dispatch_init_{4,4u,8,8u}
  template <typename T>
  void kmpc_dispatch_init(ident_t *loc, kmp_int32 global_tid,
			  kmp_int32 schedtype,
			  T lb, T ub,
			  typename make_signed<T>::type st,
			  typename make_signed<T>::type chunk)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(true);
    if(!wi) {
      // no place to store the loop information, so this is a fatal error
      log_omp.fatal() << "OpenMP loop with dynamic scheduling on non-OpenMP Realm processor!";
      abort();
    }

    // loops must be inside work items
    assert(wi->work_item != 0);

    // kmp uses an inclusive upper bound, so add the increment to get
    //  the exclusive form
    ub += st;
      
    log_omp.debug() << "loop dynamic start: start=" << lb
		    << " end=" << ub << " incr=" << st
		    << " chunk=" << chunk;

    wi->work_item->schedule.start_dynamic(lb, ub, st, chunk);
  }

  // templated code for __kmpc_dispatch_init_{4,4u,8,8u}
  template <typename T>
  int kmpc_dispatch_next(ident_t *loc, kmp_int32 global_tid,
			 kmp_int32 *p_last,
			 T *p_lb, T *p_ub,
			 typename make_signed<T>::type *p_st)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(!wi) return false;  // complained already above

    // loops must be inside work items
    assert(wi->work_item != 0);

    int64_t span_start, span_end, stride;
    if(wi->work_item->schedule.next_dynamic(span_start, span_end,
					    stride)) {
      log_omp.debug() << "loop dynamic next: start=" << span_start
		      << " end=" << span_end;

      *p_last = 0; // no detection of last block
      *p_lb = span_start;
      *p_ub = span_end - stride;  // back to inclusive upper bound
      *p_st = stride;
      return 1;
    } else {
      log_omp.debug() << "loop dynamic next: done";

      // no explicit call to end the loop, so do it here
      // if the loop construct requires a wait, an explicit barrier is
      //  inserted by the compiler
      wi->work_item->schedule.end_loop(false /*!wait*/);
      return 0;
    }
  }
    
};

extern "C" {

  using namespace Realm;

  REALM_PUBLIC_API
  void __kmpc_for_static_init_4(ident_t *loc, kmp_int32 global_tid,
				kmp_int32 schedtype,
				kmp_int32 *plastiter,
				kmp_int32 *plower, kmp_int32 *pupper,
				kmp_int32 *pstride,
				kmp_int32 incr, kmp_int32 chunk)
  {
    kmpc_for_static_init<kmp_int32>(loc, global_tid, schedtype, plastiter,
				    plower, pupper, pstride, incr, chunk);
  }

  REALM_PUBLIC_API
  void __kmpc_for_static_init_4u(ident_t *loc, kmp_int32 global_tid,
				 kmp_int32 schedtype,
				 kmp_int32 *plastiter,
				 kmp_uint32 *plower, kmp_uint32 *pupper,
				 kmp_uint32 *pstride,
				 kmp_int32 incr, kmp_int32 chunk)
  {
    kmpc_for_static_init<kmp_uint32>(loc, global_tid, schedtype, plastiter,
				     plower, pupper, pstride, incr, chunk);
  }

  REALM_PUBLIC_API
  void __kmpc_for_static_init_8(ident_t *loc, kmp_int32 global_tid,
				kmp_int32 schedtype,
				kmp_int32 *plastiter,
				kmp_int64 *plower, kmp_int64 *pupper,
				kmp_int64 *pstride,
				kmp_int64 incr, kmp_int64 chunk)
  {
    kmpc_for_static_init<kmp_int64>(loc, global_tid, schedtype, plastiter,
				    plower, pupper, pstride, incr, chunk);
  }

  REALM_PUBLIC_API
  void __kmpc_for_static_init_8u(ident_t *loc, kmp_int32 global_tid,
				 kmp_int32 schedtype,
				 kmp_int32 *plastiter,
				 kmp_uint64 *plower, kmp_uint64 *pupper,
				 kmp_uint64 *pstride,
				 kmp_int64 incr, kmp_int64 chunk)
  {
    kmpc_for_static_init<kmp_uint64>(loc, global_tid, schedtype, plastiter,
				     plower, pupper, pstride, incr, chunk);
  }

  REALM_PUBLIC_API
  void __kmpc_for_static_fini(ident_t *loc, kmp_int32 global_tid)
  {
    //printf("static_fini(%p, %d)\n", loc, global_tid);
  }

  REALM_PUBLIC_API
  void __kmpc_dispatch_init_4(ident_t *loc, kmp_int32 global_tid,
			      kmp_int32 schedtype,
			      kmp_int32 lb, kmp_int32 ub,
			      kmp_int32 st, kmp_int32 chunk)
  {
    kmpc_dispatch_init<kmp_int32>(loc, global_tid, schedtype,
				  lb, ub, st, chunk);
  }

  REALM_PUBLIC_API
  void __kmpc_dispatch_init_4u(ident_t *loc, kmp_int32 global_tid,
			       kmp_int32 schedtype,
			       kmp_uint32 lb, kmp_uint32 ub,
			       kmp_int32 st, kmp_int32 chunk)
  {
    kmpc_dispatch_init<kmp_uint32>(loc, global_tid, schedtype,
				   lb, ub, st, chunk);
  }

  REALM_PUBLIC_API
  void __kmpc_dispatch_init_8(ident_t *loc, kmp_int32 global_tid,
			      kmp_int32 schedtype,
			      kmp_int64 lb, kmp_int64 ub,
			      kmp_int64 st, kmp_int64 chunk)
  {
    kmpc_dispatch_init<kmp_int64>(loc, global_tid, schedtype,
				  lb, ub, st, chunk);
  }

  REALM_PUBLIC_API
  void __kmpc_dispatch_init_8u(ident_t *loc, kmp_int32 global_tid,
			       kmp_int32 schedtype,
			       kmp_uint64 lb, kmp_uint64 ub,
			       kmp_int64 st, kmp_int64 chunk)
  {
    // map between uint64_t and int64_t by flipping the upper bit
    kmp_int64 lb_signed = lb ^ (uint64_t(1) << 63);
    kmp_int64 ub_signed = ub ^ (uint64_t(1) << 63);

    kmpc_dispatch_init<kmp_int64>(loc, global_tid, schedtype,
				  lb_signed, ub_signed, st, chunk);
  }

  REALM_PUBLIC_API
  int __kmpc_dispatch_next_4(ident_t *loc, kmp_int32 global_tid,
			     kmp_int32 *p_last, kmp_int32 *p_lb,
			     kmp_int32 *p_ub, kmp_int32 *p_st)
  {
    return kmpc_dispatch_next<kmp_int32>(loc, global_tid,
					 p_last, p_lb, p_ub, p_st);
  }

  REALM_PUBLIC_API
  int __kmpc_dispatch_next_4u(ident_t *loc, kmp_int32 global_tid,
			      kmp_int32 *p_last, kmp_uint32 *p_lb,
			      kmp_uint32 *p_ub, kmp_int32 *p_st)
  {
    return kmpc_dispatch_next<kmp_uint32>(loc, global_tid,
					  p_last, p_lb, p_ub, p_st);
  }

  REALM_PUBLIC_API
  int __kmpc_dispatch_next_8(ident_t *loc, kmp_int32 global_tid,
			     kmp_int32 *p_last, kmp_int64 *p_lb,
			     kmp_int64 *p_ub, kmp_int64 *p_st)
  {
    return kmpc_dispatch_next<kmp_int64>(loc, global_tid,
					 p_last, p_lb, p_ub, p_st);
  }

  REALM_PUBLIC_API
  int __kmpc_dispatch_next_8u(ident_t *loc, kmp_int32 global_tid,
			      kmp_int32 *p_last, kmp_uint64 *p_lb,
			      kmp_uint64 *p_ub, kmp_int64 *p_st)
  {
    // map between uint64_t and int64_t by flipping the upper bit
    kmp_int64 lb_signed, ub_signed;
    int ret = kmpc_dispatch_next<kmp_int64>(loc, global_tid, p_last,
					    &lb_signed, &ub_signed, p_st);
    *p_lb = lb_signed ^ (uint64_t(1) << 63);
    *p_ub = ub_signed ^ (uint64_t(1) << 63);
    return ret;
  }

  REALM_PUBLIC_API
  kmp_int32 __kmpc_reduce_nowait(ident_t *loc, kmp_int32 global_tid,
				 kmp_int32 nvars, size_t reduce_size,
				 void *reduce_data, kmpc_reduce reduce_func,
				 kmp_critical_name *lck)
  {
    // tell caller to just do it themselves in all cases
    return 2;
  }

  REALM_PUBLIC_API
  void __kmpc_end_reduce_nowait(ident_t *loc, kmp_int32 global_tid,
				kmp_critical_name *lck)
  {
    // do nothing
  }

  REALM_PUBLIC_API
  void __kmpc_serialized_parallel(ident_t *loc, kmp_int32 global_tid)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(true);
    if(!wi)
      return;

    // create a new work item that is just this thread
    ThreadPool::WorkItem *work = new ThreadPool::WorkItem(1);
    wi->push_work_item(work);
    wi->thread_id = 0;
    wi->num_threads = 1;

    // caller will actually execute loop body, so return to them
  }

  REALM_PUBLIC_API
  void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32 global_tid)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    // if this isn't called from a openmp-enabled thread, we already
    //  complained above
    if(!wi) 
      return;

    // pop the top work item and make sure we're the only worker
    ThreadPool::WorkItem *work = wi->pop_work_item();
    assert(work != 0);
    assert(work->remaining_workers.load() == 1);
    delete work;
  }

  REALM_PUBLIC_API
  kmp_int32 __kmpc_single(ident_t *loc, kmp_int32 global_tid)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(true);
    if(!wi)
      return 1;  // trivially the winner

    if(!wi->work_item) {
      // not inside a larger construct - treat as nop
      return 1;
    }

    // try to become the "single" winner - the intent in OpenMP is that the
    //  first worker to get here should do the work
    int prev = -1;
    if(wi->work_item->single_winner.compare_exchange(prev, wi->thread_id)) {
      return 1;
    } else {
      // if we were already the winner, that's ok too
      return (prev == wi->thread_id) ? 1 : 0;
    }
  }

  REALM_PUBLIC_API
  void __kmpc_end_single(ident_t *loc, kmp_int32 global_tid)
  {
    // we didn't create a team for __kmpc_single, so nothing to do here
  }

  REALM_PUBLIC_API
  void __kmpc_barrier(ident_t *loc, kmp_int32 global_tid)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(true);
    if(!wi)
      return;

    //log_omp.print() << "barrier enter: id=" << wi->thread_id;

    if(wi->work_item && (wi->num_threads > 1)) {
      // step 1: observe that barrier is not still being exited
      int c;
      do {
	c = wi->work_item->barrier_count.load();
      } while(c >= wi->num_threads);
      // step 2: increment counter to enter
      c = wi->work_item->barrier_count.fetch_add(1) + 1;
      if(c == wi->num_threads) {
	// last arriver - reset count once all others have exited
	//   reset "single" winner too
	wi->work_item->single_winner.store(-1);
	while(true) {
	  int expval = 2 * wi->num_threads - 1;
	  if(wi->work_item->barrier_count.compare_exchange(expval, 0))
	    break;
	}
      } else {
	// step 3: observe that all threads have entered
	do {
	  c = wi->work_item->barrier_count.load();
	} while(c < wi->num_threads);
	// step 4: increment counter again to exit
	wi->work_item->barrier_count.fetch_add(1);
      }
    } else {
      // not inside a larger construct - nothing to do
    }
    
    //log_omp.print() << "barrier exit: id=" << wi->thread_id;
  }

  static unsigned hash_kmp_critical_name(kmp_critical_name *lck)
  {
    uintptr_t v = reinterpret_cast<uintptr_t>(lck);
    // two hashes pushes 24 bits worth of address into the bottom 6
    v ^= (v >> 6);
    v ^= (v >> 12);
    // return bottom 6 bits
    return (v & 63);
  }

  REALM_PUBLIC_API
  void __kmpc_critical(ident_t *loc, kmp_int32 global_tid,
		       kmp_critical_name *lck)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(true);
    if(!wi || !wi->work_item || (wi->num_threads == 1)) {
      // already single-threaded, so no further effort needed
      return;
    }

    unsigned bit = hash_kmp_critical_name(lck);
    uint64_t mask = uint64_t(1) << bit;

    // try to set the bit in the critical flags - repeat until success
    while(true) {
      uint64_t orig = wi->work_item->critical_flags.fetch_or(mask);
      if((orig & mask) == 0) break;
      Thread::yield();
    }
  }

  REALM_PUBLIC_API
  void __kmpc_end_critical(ident_t *loc, kmp_int32 global_tid,
			   kmp_critical_name *lck)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info(false);
    if(!wi || !wi->work_item || (wi->num_threads == 1)) {
      // already single-threaded, so no further effort needed
      return;
    }

    unsigned bit = hash_kmp_critical_name(lck);
    uint64_t mask = uint64_t(1) << bit;

    // clear the bit - it had better already be set
    uint64_t orig = wi->work_item->critical_flags.fetch_and(~mask);
    assert((orig & mask) != 0);
  }

};
#endif
