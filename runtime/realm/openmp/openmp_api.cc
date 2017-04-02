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

#include "openmp_threadpool.h"

#include "../logging.h"

#include <stdio.h>
#include <stdint.h>

namespace Realm {
  extern Logger log_omp;

  // application-visible calls - always generated
  extern "C" {
    int omp_get_num_threads(void)
    {
      Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info();
      if(wi)
	return wi->num_threads;
      else
	return 1;
    }

    int omp_get_thread_num(void)
    {
      Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info();
      if(wi)
	return wi->thread_id;
      else
	return 0;
    }
  };

#ifdef REALM_OPENMP_GOMP_SUPPORT
  extern "C" {
    void GOMP_parallel_start(void (*fnptr)(void *data), void *data, int nthreads)
    {
      //printf("GOMP_parallel_start(%p, %p, %d)\n", fnptr, data, nthreads);
      Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info();
      if(!wi) {
	log_omp.warning() << "OpenMP-parallelized loop on non-OpenMP Realm processor!";
	return;
      }

      std::set<int> worker_ids;
      wi->pool->claim_workers(nthreads - 1, worker_ids);
      int act_threads = 1 + worker_ids.size();

      ThreadPool::WorkItem *work = new ThreadPool::WorkItem;
      work->remaining_workers = act_threads;
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

    void GOMP_parallel_end(void)
    {
      //printf("GOMP_parallel_end()\n");
      Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info();
      // if this isn't called from a openmp-enabled thread, we already
      //  complained above
      if(!wi)
	return;

      ThreadPool::WorkItem *work = wi->pop_work_item();
      assert(work != 0);
      // make sure all workers have finished
      if(__sync_sub_and_fetch(&(work->remaining_workers), 1) > 0) {
	log_omp.info() << "waiting for workers to complete";
	while(__sync_sub_and_fetch(&(work->remaining_workers), 0) > 0)
	  sched_yield();
      }
      delete work;
    }

    void GOMP_parallel(void (*fnptr)(void *data), void *data, unsigned nthreads, unsigned int flags)
    {
      GOMP_parallel_start(fnptr, data, nthreads);
      fnptr(data);
      GOMP_parallel_end();
    }
  };
#endif

#ifdef REALM_OPENMP_KMP_SUPPORT
  typedef int32_t kmp_int32;
  typedef uint32_t kmp_uint32;
  typedef int64_t kmp_int64;
  typedef uint64_t kmp_uint64;
  typedef void (*kmpc_micro)(kmp_int32 *global_tid, kmp_int32 *local_tid, ...);
  typedef struct ident ident_t;
  typedef void (*kmpc_reduce)(void *lhs_data, void *rhs_data);
  typedef int32_t kmp_critical_name;

  extern "C" {
    void __kmpc_begin(ident_t *loc, kmp_int32 flags);
    void __kmpc_end(ident_t *loc);

    kmp_int32 __kmpc_ok_to_fork(ident_t *loc);
    kmp_int32 __kmpc_global_thread_num(ident_t *loc);

    void __kmpc_fork_call(ident_t *loc, kmp_int32 argc, kmpc_micro microtask, ...);
    void __kmpc_for_static_init_4(ident_t *loc, kmp_int32 global_tid,
				  kmp_int32 schedtype,
				  kmp_int32 *plastiter,
				  kmp_int32 *plower, kmp_int32 *pupper,
				  kmp_int32 *pstride,
				  kmp_int32 incr, kmp_int32 chunk);
    void __kmpc_for_static_init_4u(ident_t *loc, kmp_int32 global_tid,
				   kmp_int32 schedtype,
				   kmp_int32 *plastiter,
				   kmp_uint32 *plower, kmp_uint32 *pupper,
				   kmp_uint32 *pstride,
				   kmp_uint32 incr, kmp_uint32 chunk);
    void __kmpc_for_static_init_8(ident_t *loc, kmp_int32 global_tid,
				  kmp_int32 schedtype,
				  kmp_int32 *plastiter,
				  kmp_int64 *plower, kmp_int64 *pupper,
				  kmp_int64 *pstride,
				  kmp_int64 incr, kmp_int64 chunk);
    void __kmpc_for_static_init_8u(ident_t *loc, kmp_int32 global_tid,
				   kmp_int32 schedtype,
				   kmp_int32 *plastiter,
				   kmp_uint64 *plower, kmp_uint64 *pupper,
				   kmp_uint64 *pstride,
				   kmp_uint64 incr, kmp_uint64 chunk);
    void __kmpc_for_static_fini(ident_t *loc, kmp_int32 global_tid);
    kmp_int32 __kmpc_reduce_nowait(ident_t *loc, kmp_int32 global_tid,
				   kmp_int32 nvars, size_t reduce_size,
				   void *reduce_data, kmpc_reduce reduce_func,
				   kmp_critical_name *lck);
    void __kmpc_end_reduce_nowait(ident_t *loc, kmp_int32 global_tid,
				  kmp_critical_name *lck);

    void __kmpc_serialized_parallel(ident_t *loc, kmp_int32 global_tid);
    void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32 global_tid);
  };

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
  };

  void __kmpc_begin(ident_t *loc, kmp_int32 flags)
  {
    // do nothing
  }

  void __kmpc_end(ident_t *loc)
  {
    // do nothing
  }

  kmp_int32 __kmpc_ok_to_fork(ident_t *loc)
  {
    // this is supposed to always return TRUE
    return 1;
  }

  kmp_int32 __kmpc_global_thread_num(ident_t *loc)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info();
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
    default:
      {
	fprintf(stderr, "HELP!  __kmpc_fork_call called with argc == %d\n", argc);
	assert(0);
      }
    }

    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info();
    if(!wi) {
      log_omp.warning() << "OpenMP-parallelized loop on non-OpenMP Realm processor!";
      // invoke thunk directly and then return
      (*invoker)(&thunk);
      return;
    }

    std::set<int> worker_ids;
    // TODO: thread limit comes from where?
    wi->pool->claim_workers(-1, worker_ids);
    int act_threads = 1 + worker_ids.size();

    ThreadPool::WorkItem *work = new ThreadPool::WorkItem;
    work->remaining_workers = act_threads;
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
    if(__sync_sub_and_fetch(&(work->remaining_workers), 1) > 0) {
      log_omp.info() << "waiting for workers to complete";
      while(__sync_sub_and_fetch(&(work->remaining_workers), 0) > 0)
	sched_yield();
    }
    delete work;
  }

  // templated code for __kmpc_for_static_init_{4,4u,8,8u}
  template <typename T>
  static inline void kmpc_for_static_init(ident_t *loc, kmp_int32 global_tid,
					  kmp_int32 schedtype,
					  kmp_int32 *plastiter,
					  T *plower, T *pupper,
					  T *pstride,
					  T incr, T chunk)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info();
    // if running sequentially, or bounds are inside-out,
    //  just return with the bounds unchanged
    if(!wi ||
       (wi->num_threads == 1) ||
       ((incr > 0) && (*plower > *pupper)) ||
       ((incr < 0) && (*plower < *pupper))) {
      return;
    }

    //printf("static_init(%p, %d, %d)\n", loc, global_tid, schedtype);
    switch(schedtype) {
    case 34 /* kmp_sch_static */:
      {
	T iters;
	if(incr > 0) {
	  iters = 1 + (*pupper - *plower) / incr;
	} else {
	  iters = 1 + (*plower - *pupper) / -incr;
	}
	T whole = iters / wi->num_threads;
	T leftover = iters - (whole * wi->num_threads);
	*plower += incr * (whole * wi->thread_id +
			   ((wi->thread_id < leftover) ? wi->thread_id : leftover));
	*pupper = *plower + incr * (whole +
				    ((wi->thread_id < leftover) ? 1 : 0)) - 1;
	// special case for when some threads get no iterations at all
	*plastiter = ((whole > 0) ? (wi->thread_id == (wi->num_threads - 1)) :
		                    (wi->thread_id == (leftover - 1)));
	//printf("static(%d, %d, %d, %d, %d)\n", *plower, *pupper, *pstride, incr, *plastiter);
	return;
      }

    default: assert(false);
    }
  }

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

  void __kmpc_for_static_init_4u(ident_t *loc, kmp_int32 global_tid,
				 kmp_int32 schedtype,
				 kmp_int32 *plastiter,
				 kmp_uint32 *plower, kmp_uint32 *pupper,
				 kmp_uint32 *pstride,
				 kmp_uint32 incr, kmp_uint32 chunk)
  {
    kmpc_for_static_init<kmp_uint32>(loc, global_tid, schedtype, plastiter,
				     plower, pupper, pstride, incr, chunk);
  }

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

  void __kmpc_for_static_init_8u(ident_t *loc, kmp_int32 global_tid,
				 kmp_int32 schedtype,
				 kmp_int32 *plastiter,
				 kmp_uint64 *plower, kmp_uint64 *pupper,
				 kmp_uint64 *pstride,
				 kmp_uint64 incr, kmp_uint64 chunk)
  {
    kmpc_for_static_init<kmp_uint64>(loc, global_tid, schedtype, plastiter,
				     plower, pupper, pstride, incr, chunk);
  }

  void __kmpc_for_static_fini(ident_t *loc, kmp_int32 global_tid)
  {
    //printf("static_fini(%p, %d)\n", loc, global_tid);
  }

  kmp_int32 __kmpc_reduce_nowait(ident_t *loc, kmp_int32 global_tid,
				 kmp_int32 nvars, size_t reduce_size,
				 void *reduce_data, kmpc_reduce reduce_func,
				 kmp_critical_name *lck)
  {
    // tell caller to just do it themselves in all cases
    return 2;
  }

  void __kmpc_end_reduce_nowait(ident_t *loc, kmp_int32 global_tid,
				kmp_critical_name *lck)
  {
    // do nothing
  }

  void __kmpc_serialized_parallel(ident_t *loc, kmp_int32 global_tid)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info();
    if(!wi) {
      log_omp.warning() << "OpenMP-parallelized loop on non-OpenMP Realm processor!";
      return;
    }

    // create a new work item that is just this thread
    ThreadPool::WorkItem *work = new ThreadPool::WorkItem;
    work->remaining_workers = 1;
    wi->push_work_item(work);
    wi->thread_id = 0;
    wi->num_threads = 1;

    // caller will actually execute loop body, so return to them
  }

  void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32 global_tid)
  {
    Realm::ThreadPool::WorkerInfo *wi = Realm::ThreadPool::get_worker_info();
    // if this isn't called from a openmp-enabled thread, we already
    //  complained above
    if(!wi) 
      return;

    // pop the top work item and make sure we're the only worker
    ThreadPool::WorkItem *work = wi->pop_work_item();
    assert(work != 0);
    assert(work->remaining_workers == 1);
    delete work;
  }
#endif

}; // namespace Realm
