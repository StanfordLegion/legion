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
#ifndef REALM_OPENMP_THREADPOOL_H
#define REALM_OPENMP_THREADPOOL_H

#include "../threads.h"

namespace Realm {

  class ThreadPool {
  public:
    ThreadPool(int _num_workers);
    ~ThreadPool(void);

    // associates the calling thread as the master of the threadpool
    void associate_as_master(void);

    // entry point for workers - does not return until thread pool is shut down
    void worker_entry(void);

    struct WorkItem {
      int prev_thread_id;
      int prev_num_threads;
      WorkItem *parent_work_item;
      int remaining_workers;
    };

    struct WorkerInfo {
      enum Status {
	WORKER_MASTER,
	WORKER_STARTING,
	WORKER_IDLE,
	WORKER_CLAIMED,
	WORKER_ACTIVE,
	WORKER_SHUTDOWN,
      };
      int /*Status*/ status; // int allows CAS primitives
      ThreadPool *pool;
      int thread_id;  // in current team
      int num_threads; // in current team
      void (*fnptr)(void *data);
      void *data;
      WorkItem *work_item;

      void push_work_item(WorkItem *new_work);
      WorkItem *pop_work_item(void);
    };
      
    // returns the WorkerInfo (if any) associated with the caller (which
    //  can be master or worker)
    static WorkerInfo *get_worker_info(void);

    // asks worker threads to shut down and waits for them to complete
    void shutdown(void);

    void claim_workers(int count, std::set<int>& worker_ids);

    void start_worker(int worker_id,
		      int thread_id, int num_threads,
		      void (*fnptr)(void *data), void *data,
		      WorkItem *work_item);

  protected:
    int num_workers;
    std::vector<Thread *> worker_threads;
    std::vector<WorkerInfo> worker_infos;
  };

}; // namespace Realm

#include "openmp_threadpool.inl"

#endif
