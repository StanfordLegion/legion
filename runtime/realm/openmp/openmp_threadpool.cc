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

// OpenMP (or similar) thread pool for Realm
#include "openmp_threadpool.h"

#include "../logging.h"

namespace Realm {

  Logger log_pool("threadpool");

  namespace ThreadLocal {
    __thread ThreadPool::WorkerInfo *threadpool_workerinfo = 0;
  };

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

  ThreadPool::ThreadPool(int _num_workers)
    : num_workers(_num_workers)
  {
    // these will be filled in as workers show up
    worker_threads.resize(num_workers, 0);

    worker_infos.resize(num_workers + 1);
    for(int i = 0; i <= num_workers; i++) {
      WorkerInfo& wi = worker_infos[i];
      wi.status = i ? WorkerInfo::WORKER_STARTING : WorkerInfo::WORKER_MASTER;
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
  {}

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

  // entry point for workers - does not return until thread pool is shut down
  void ThreadPool::worker_entry(void)
  {
    log_pool.info() << "new worker thread";

    // choose an ID by finding an info to change from STARTING->IDLE
    int id = 1;
    while(!__sync_bool_compare_and_swap(&(worker_infos[id].status),
					WorkerInfo::WORKER_STARTING,
					WorkerInfo::WORKER_IDLE)) {
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
      switch(wi->status) {
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
	  __sync_fetch_and_sub(&(wi->work_item->remaining_workers), 1);
	  wi->status = WorkerInfo::WORKER_IDLE;
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

  // asks worker threads to shut down and waits for them to complete
  void ThreadPool::shutdown(void)
  {
    // tell all workers to shutdown
    for(std::vector<WorkerInfo>::iterator it = worker_infos.begin();
	it != worker_infos.end();
	++it) {
      if(it->status == WorkerInfo::WORKER_MASTER) continue;
      bool ok = __sync_bool_compare_and_swap(&(it->status),
					     WorkerInfo::WORKER_IDLE,
					     WorkerInfo::WORKER_SHUTDOWN);
      assert(ok);
    }

    // now join on all threads
    for(std::vector<Thread *>::const_iterator it = worker_threads.begin();
	it != worker_threads.end();
	++it)
      (*it)->join();

    worker_threads.clear();
  }

  void ThreadPool::claim_workers(int count, std::set<int>& worker_ids)
  {
    int remaining = count;
    for(size_t i = 0; i < worker_infos.size(); i++)
      // attempt atomic change from IDLE -> CLAIMED
      if(__sync_bool_compare_and_swap(&(worker_infos[i].status),
				      WorkerInfo::WORKER_IDLE,
				      WorkerInfo::WORKER_CLAIMED)) {
	worker_ids.insert(i);
	remaining -= 1;
	if(remaining == 0)
	  break;
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
    
    assert(wi->status == WorkerInfo::WORKER_CLAIMED);
    wi->thread_id = thread_id;
    wi->num_threads = num_threads;
    wi->fnptr = fnptr;
    wi->data = data;
    wi->work_item = work_item;
    __sync_bool_compare_and_swap(&(wi->status),
				 WorkerInfo::WORKER_CLAIMED,
				 WorkerInfo::WORKER_ACTIVE);
  }

};
