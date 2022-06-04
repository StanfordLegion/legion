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

// OpenMP (or similar) thread pool for Realm
#include "realm/openmp/openmp_threadpool.h"

#include "realm/logging.h"
#include "realm/proc_impl.h"

namespace Realm {

  Logger log_pool("threadpool");

  namespace ThreadLocal {
    REALM_THREAD_LOCAL ThreadPool::WorkerInfo *threadpool_workerinfo = 0;
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // class LoopSchedule

  void LoopSchedule::initialize(int _num_workers)
  {
    num_workers = _num_workers;
    loop_pos.store(0);
    loop_barrier.store(0);
  }

  static inline uint64_t index_to_pos(int64_t index,
				      int64_t base, int64_t incr)
  {
    if(incr > 0) {
      uint64_t delta = (uint64_t)index - (uint64_t)base;
      if(incr != 1)
	delta /= incr;
      return delta;
    } else {
      uint64_t delta = (uint64_t)base - (uint64_t)index;
      if(incr != -1)
	delta /= -incr;
      return delta;
    }
  }

  static inline int64_t pos_to_index(uint64_t pos,
				     int64_t base, int64_t incr)
  {
    if(incr > 0)
      return (base + (pos * incr));
    else
      return (base - (pos * -incr));
  }
  
  bool LoopSchedule::start_static(int64_t start, int64_t end,
				  int64_t incr, int64_t chunk,
				  int thread_id,
				  int64_t& span_start, int64_t& span_end)
  {
    // make sure nobody's still on the previous loop
    while(loop_barrier.load() >= num_workers) Thread::yield();

    // compute the loop limit, dealing with the negative stride cases
    uint64_t limit;
    if(incr > 0) {
      limit = ((end >= start) ?
	         index_to_pos(end, start, incr) :
	         0);
    } else {
      limit = ((end <= start) ?
	         index_to_pos(end, start, incr) :
	         0);
    }

    // if the chunk wasn't specified, divide the work as evenly as
    //  possible into a single chunk per thread
    if(chunk <= 0) {
      chunk = limit / num_workers;
      if((uint64_t(chunk) * num_workers) < limit)
	chunk++;
    }

    // the compiler promises all threads will have the same value, so
    //  everybody can just store knowing that either they're first or
    //  they're writing the same thing as everybody else
    loop_limit.store(limit);
    loop_base.store(start);
    loop_incr.store(incr);
    loop_chunk.store(chunk);

    // signal that we're in the loop
    loop_barrier.fetch_add(1);

    // and now give this thread its first chunk
    uint64_t pos = thread_id * chunk;
    if(pos < limit) {
      uint64_t count = std::min((uint64_t)chunk,
				(limit - pos));
      span_start = pos_to_index(pos, start, incr);
      span_end = pos_to_index(pos + count, start, incr);
      return true;
    } else
      return false;
  }

  bool LoopSchedule::next_static(int64_t& span_start, int64_t& span_end)
  {
    // we use these a bunch, and it's ok to cache them
    int64_t base = loop_base.load();
    int64_t incr = loop_incr.load();
    int64_t chunk = loop_chunk.load();
    uint64_t limit = loop_limit.load();
      
    // don't need to know thread id because we just step by num_workers
    //  chunks
    uint64_t old_pos = index_to_pos(span_start, base, incr);
    uint64_t new_pos = old_pos + (num_workers * chunk);

    // if we wrap around, new_pos will be less than old_pos
    if((new_pos > old_pos) && (new_pos < limit)) {
      uint64_t count = std::min((uint64_t)chunk,
				(limit - new_pos));
      span_start = pos_to_index(new_pos, base, incr);
      span_end = pos_to_index(new_pos + count, base, incr);
      return true;
    } else
      return false;
  }

  void LoopSchedule::start_dynamic(int64_t start, int64_t end,
				   int64_t incr, int64_t chunk)
  {
    // make sure nobody's still on the previous loop
    while(loop_barrier.load() >= num_workers) Thread::yield();

    // compute the loop limit, dealing with the negative stride cases
    uint64_t limit;
    if(incr > 0) {
      limit = ((end >= start) ?
	         index_to_pos(end, start, incr) :
	         0);
    } else {
      limit = ((end <= start) ?
	         index_to_pos(end, start, incr) :
	         0);
    }

    // if the chunk wasn't specified, pick a value that aims for ~8
    //  chunks per thread to get some dynamic scheduling goodness
    if(chunk <= 0) {
      chunk = limit / (8 * num_workers);
      if(chunk == 0) chunk = 1;
    }

    // if the chunk size is so large that n-1 of the workers can
    //  cause an overshoot that wraps around, we have a problem
    uint64_t limit_overshoot = (limit + ((num_workers - 1) *
					 (uint64_t)chunk));
    assert(limit_overshoot >= limit);

    // the compiler promises all threads will have the same value, so
    //  everybody can just store knowing that either they're first or
    //  they're writing the same thing as everybody else
    // (loop_pos was reset to 0 at the end of the previous loop)
    loop_limit.store(limit);
    loop_base.store(start);
    loop_incr.store(incr);
    loop_chunk.store(chunk);

    // signal that we're in the loop
    loop_barrier.fetch_add(1);
  }

  bool LoopSchedule::next_dynamic(int64_t& span_start, int64_t& span_end,
				  int64_t& stride)
  {
    // we use these a bunch, and it's ok to cache them
    int64_t base = loop_base.load();
    int64_t incr = loop_incr.load();
    int64_t chunk = loop_chunk.load();
    uint64_t limit = loop_limit.load();
      
    // atomic increment to claim a new chunk
    uint64_t new_pos = loop_pos.fetch_add(chunk);

    if(new_pos < limit) {
      uint64_t count = std::min((uint64_t)chunk,
				(limit - new_pos));
      span_start = pos_to_index(new_pos, base, incr);
      span_end = pos_to_index(new_pos + count, base, incr);
      stride = incr;
      return true;
    } else
      return false;
  }

  void LoopSchedule::end_loop(bool wait)
  {
    // stall unless all threads have entered the current loop
    while(loop_barrier.load() < num_workers) Thread::yield();

    // increment the barrier to indicate we're done, and see if we're the
    //  last
    int prev_count = loop_barrier.fetch_add(1);
    bool last_ender = (prev_count == (2 * num_workers - 1));

    if(wait) {
      if(last_ender) {
	// we don't need to contribute again, and can return unless there was
	//  only one worker
	if(num_workers > 1) return;
      } else {
	// we need to observe the count reach 2x num_workers
	while(loop_barrier.load() < (2 * num_workers)) Thread::yield();

	// signal our observation and return if we're not the last to do so
	prev_count = loop_barrier.fetch_add(1);
	if(prev_count < (3 * num_workers - 2)) return;
      }
    } else {
      // if we weren't the last to end, we're done - no waiting
      if(!last_ender)
	return;
    }

    // exactly one thread gets here, and resets loop_pos and then the barrier
    loop_pos.store(0);
    loop_barrier.store_release(0);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ThreadPool::WorkItem

  ThreadPool::WorkItem::WorkItem(int _num_threads)
    : remaining_workers(_num_threads)
    , single_winner(-1)
    , barrier_count(0)
    , critical_flags(0)
  {
    schedule.initialize(_num_threads);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ThreadPool::WorkerInfo

  void ThreadPool::WorkerInfo::push_work_item(ThreadPool::WorkItem *new_work)
  {
    new_work->prev_thread_id = thread_id;
    new_work->prev_num_threads = num_threads;
    new_work->parent_work_item = work_item;
    work_item = new_work;
  }

  ThreadPool::WorkItem *ThreadPool::WorkerInfo::pop_work_item(void)
  {
    WorkItem *old_item = work_item;
    thread_id = old_item->prev_thread_id;
    num_threads = old_item->prev_num_threads;
    work_item = old_item->parent_work_item;
    return old_item;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ThreadPool

  ThreadPool::ThreadPool(Processor _proc,
                         int _num_workers,
			 const std::string& _name_prefix,
			 int _numa_node, size_t _stack_size,
			 CoreReservationSet& crs)
    : proc(_proc)
    , num_workers(_num_workers)
    , workers_running(false)
  {
    // create per-worker core reservations
    CoreReservationParameters params;
    params.set_num_cores(1);
    params.set_numa_domain(_numa_node);
    params.set_alu_usage(params.CORE_USAGE_EXCLUSIVE);
    params.set_fpu_usage(params.CORE_USAGE_EXCLUSIVE);
    params.set_ldst_usage(params.CORE_USAGE_SHARED);
    params.set_max_stack_size(_stack_size);

    core_rsrvs.resize(num_workers, 0);
    for(int i = 0; i < num_workers; i++)
      core_rsrvs[i] = new CoreReservation(stringbuilder() << _name_prefix << " (worker " << (i + 1) << ")",
					  crs, params);

    // these will be filled in as workers show up
    worker_threads.resize(num_workers, 0);

    worker_infos.resize(num_workers + 1);
    for(int i = 0; i <= num_workers; i++) {
      WorkerInfo& wi = worker_infos[i];
      wi.status.store(i ? WorkerInfo::WORKER_NOT_RUNNING :
		          WorkerInfo::WORKER_MASTER);
      wi.pool = this;
      wi.thread_id = 0;
      wi.num_threads = 1;
      wi.fnptr = 0;
      wi.data = 0;
      wi.work_item = 0;
    }

    log_pool.info() << "pool " << (void *)this << " started - " << num_workers << " workers";
  }
  
  ThreadPool::~ThreadPool(void)
  {
    assert(!workers_running);

    for(int i = 0; i < num_workers; i++)
      delete core_rsrvs[i];
  }

  // associates the calling thread as the master of the threadpool
  void ThreadPool::associate_as_master(void)
  {
    log_pool.debug() << "associate: " << Thread::self() << " " << (void *)(ThreadLocal::threadpool_workerinfo) << " " << (void *)(&worker_infos[0]);
    if(ThreadLocal::threadpool_workerinfo) {
      // should not already be associated with a different pool
      assert(ThreadLocal::threadpool_workerinfo == &worker_infos[0]);
    } else {
      ThreadLocal::threadpool_workerinfo = &worker_infos[0];
    }
  }

  // returns the associated thread pool, optionally warning if none exists
  /*static*/ ThreadPool *ThreadPool::get_associated_pool(bool warn_if_missing)
  {
    WorkerInfo *info = get_worker_info(warn_if_missing);
    return (info ? info->pool : 0);
  }

  // entry point for workers - does not return until thread pool is shut down
  void ThreadPool::worker_entry(void)
  {
    log_pool.info() << "new worker thread";

    // set TLS so that we respond properly to get_executing_processor
    ThreadLocal::current_processor = proc;

    // choose an ID by finding an info to change from STARTING->IDLE
    int id = 1;
    while(true) {
      int expval = WorkerInfo::WORKER_STARTING;
      if(worker_infos[id].status.compare_exchange(expval,
						  WorkerInfo::WORKER_IDLE))
	break;
      id++;
      assert(id <= num_workers);
    }

    // subtract 1 because we don't keep a Thread * for master
    assert(worker_threads[id - 1] == 0);
    worker_threads[id - 1] = Thread::self();

    WorkerInfo *wi = &worker_infos[id];
    ThreadLocal::threadpool_workerinfo = wi;
    log_pool.debug() << "worker: " << Thread::self() << " " << (void *)(ThreadLocal::threadpool_workerinfo);

    bool worker_shutdown = false;
    while(!worker_shutdown) {
      switch(wi->status.load_acquire()) {
      case WorkerInfo::WORKER_IDLE:
      case WorkerInfo::WORKER_CLAIMED:
	{
	  sched_yield();
	  break;
	}

      case WorkerInfo::WORKER_ACTIVE:
	{
	  log_pool.info() << "worker " << wi->thread_id << "/" << wi->num_threads << " executing: " << (void *)(wi->fnptr) << "(" << wi->data << ")";
	  (wi->fnptr)(wi->data);
	  log_pool.info() << "worker " << wi->thread_id << "/" << wi->num_threads << " done";
	  wi->work_item->remaining_workers.fetch_sub_acqrel(1);
	  wi->status.store(WorkerInfo::WORKER_IDLE);
	  break;
	}

      case WorkerInfo::WORKER_SHUTDOWN:
	{
	  log_pool.info() << "worker shutdown received";
	  worker_shutdown = true;
	  break;
	}

      default:
	assert(0);
      }
    }
  }

  // starts worker threads running if they weren't already
  void ThreadPool::start_worker_threads(void)
  {
    if(!workers_running) {
      for(int i = 0; i < num_workers; i++) {
	assert(worker_threads[i] == 0);
	worker_infos[i+1].status.store(WorkerInfo::WORKER_STARTING);
      }

      for(int i = 0; i < num_workers; i++) {
	// the threads we launch will assign ids themselves
	ThreadLaunchParameters tlp;
	Thread::create_kernel_thread<ThreadPool,
				     &ThreadPool::worker_entry>(this, tlp,
								*core_rsrvs[i]);
      }

      workers_running = true;
    }
  }

  // asks worker threads to shut down and waits for them to complete
  void ThreadPool::stop_worker_threads(void)
  {
    if(workers_running) {
      // tell all workers to shutdown
      for(int i = 0; i < num_workers; i++) {
	int expval = WorkerInfo::WORKER_IDLE;
	bool ok = worker_infos[i+1].status.compare_exchange(expval,
							    WorkerInfo::WORKER_SHUTDOWN);
	assert(ok);
      }

      // now join on all threads
      for(int i = 0; i < num_workers; i++) {
	worker_threads[i]->join();
	delete worker_threads[i];
	worker_threads[i] = 0;
      }

      workers_running = false;
    }
  }

  void ThreadPool::claim_workers(int count, std::set<int>& worker_ids)
  {
    // spin up threads if they're not already
    if(!workers_running)
      start_worker_threads();

    int remaining = count;
    size_t i = 0;
    while(i < worker_infos.size()) {
      // attempt atomic change from IDLE -> CLAIMED
      int expval = WorkerInfo::WORKER_IDLE;
      if(worker_infos[i].status.compare_exchange(expval,
						 WorkerInfo::WORKER_CLAIMED)) {
	worker_ids.insert(i);
	remaining -= 1;
	if(remaining == 0)
	  break;
      }
      // unless this worker is still starting up, we have to move on to the next
      //  one and try it
      if(expval != WorkerInfo::WORKER_STARTING)
	i++;
    }

    log_pool.info() << "claim_workers requested " << count << ", got " << worker_ids.size();
  }

  void ThreadPool::start_worker(int worker_id,
				int thread_id, int num_threads,
				void (*fnptr)(void *data), void *data,
				WorkItem *work_item)
  {
    assert((worker_id >= 0) && (worker_id <= num_workers));
    WorkerInfo *wi = &worker_infos[worker_id];
    
    assert(wi->status.load() == WorkerInfo::WORKER_CLAIMED);
    wi->thread_id = thread_id;
    wi->num_threads = num_threads;
    wi->app_num_threads = 0;
    wi->fnptr = fnptr;
    wi->data = data;
    wi->work_item = work_item;
    int expval = WorkerInfo::WORKER_CLAIMED;
    bool ok = wi->status.compare_exchange(expval,
					  WorkerInfo::WORKER_ACTIVE);
    assert(ok);
  }

};
