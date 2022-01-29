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

// manager for background work that can be performed by available threads

#ifndef REALM_BGWORK_H
#define REALM_BGWORK_H

#include "realm/atomics.h"
#include "realm/threads.h"
#include "realm/mutex.h"
#include "realm/cmdline.h"
#include "realm/timers.h"

#include <string>

namespace Realm {

  class BackgroundWorkItem;
  class BackgroundWorkThread;

  class BackgroundWorkManager {
  public:
    BackgroundWorkManager(void);
    ~BackgroundWorkManager(void);

    struct Config {
      WithDefault<unsigned, 2> generic_workers; // non-numa-specific workers
      WithDefault<unsigned, 0> per_numa_workers;
      WithDefault<bool, false> pin_generic;
      WithDefault<bool, false> pin_numa;
      WithDefault<size_t, 1024> worker_stacksize_in_kb;
      WithDefault<long long, 0> worker_spin_interval;
      WithDefault<long long, 100000> work_item_timeslice;
    };

    void configure_from_cmdline(std::vector<std::string>& cmdline);

    void start_dedicated_workers(Realm::CoreReservationSet& crs);

    void stop_dedicated_workers(void);

    typedef unsigned long long BitMask;
    static const size_t MAX_WORK_ITEMS = 256;
    static const size_t BITMASK_BITS = 8 * sizeof(BitMask);
    static const size_t BITMASK_ARRAY_SIZE = (MAX_WORK_ITEMS + BITMASK_BITS - 1) / BITMASK_BITS;

    class Worker {
    public:
      Worker(void);
      ~Worker(void);

      void set_manager(BackgroundWorkManager *_manager);

      // configuration settings impact which work items this worker can handle
      void set_max_timeslice(long long _timeslice_in_ns);
      void set_numa_domain(int _numa_domain);  // -1 == dont care

      bool do_work(long long max_time_in_ns,
		   atomic<bool> *interrupt_flag);

    protected:
      BackgroundWorkManager *manager;
      unsigned starting_slot;
      BitMask known_work_item_mask[BITMASK_ARRAY_SIZE];
      BitMask allowed_work_item_mask[BITMASK_ARRAY_SIZE];
      long long max_timeslice;
      int numa_domain;
    };
  protected:
    friend class BackgroundWorkManager::Worker;
    friend class BackgroundWorkItem;

    unsigned assign_slot(BackgroundWorkItem *item);
    void release_slot(unsigned slot);
    void advertise_work(unsigned slot);

    Config cfg;

    // mutex protects assignment of work items to slots
    Mutex mutex;
    atomic<unsigned> num_work_items;
    atomic<BitMask> active_work_item_mask[BITMASK_ARRAY_SIZE];

    atomic<int> work_item_usecounts[MAX_WORK_ITEMS];
    BackgroundWorkItem *work_items[MAX_WORK_ITEMS];

    friend class BackgroundWorkThread;

    // to manage sleeping workers, we need to stuff three things into a
    //  single atomically-updatable state variable:
    // a) the number of active work items - no worker should sleep if there are
    //     any active work items, and any increment of the active work items
    //     should wake up one sleeping worker (unless there are none)
    //     (NOTE: this counter can temporarily underflow, so needs to be the top
    //       field in the variable to avoid temporarily corrupting other fields)
    // b) the number of sleeping workers
    // c) a bit indicating if a shutdown has been requested (which should wake
    //     up all remaining workers)
    static const uint32_t STATE_SHUTDOWN_BIT = 1;
    static const uint32_t STATE_ACTIVE_ITEMS_MASK = 0xFFFF;
    static const unsigned STATE_ACTIVE_ITEMS_SHIFT = 16;
    static const uint32_t STATE_SLEEPING_WORKERS_MASK = 0xFFF;
    static const unsigned STATE_SLEEPING_WORKERS_SHIFT = 4;
    atomic<uint32_t> worker_state;

    // sleeping workers go in a doorbell list with a delegating mutex
    DelegatingMutex db_mutex;
    DoorbellList db_list;

    std::vector<BackgroundWorkThread *> dedicated_workers;
  };

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE BackgroundWorkItem {
  public:
    BackgroundWorkItem(const std::string& _name);
    virtual ~BackgroundWorkItem(void);

    void add_to_manager(BackgroundWorkManager *_manager,
			int _numa_domain = -1,
			long long _min_timeslice_needed = -1);

    // perform work, trying to respect the 'work_until' time limit - return
    //  true to request requeuing (this is more efficient than calling
    //  'make_active' at the end of 'do_work') or false if all work has been
    //  completed (or if 'make_active' has already been called)
    virtual bool do_work(TimeLimit work_until) = 0;

  protected:
    friend class BackgroundWorkManager::Worker;

    // mark this work item as active (i.e. having work to do)
    void make_active(void);

    std::string name;
    BackgroundWorkManager *manager;
    int numa_domain;
    long long min_timeslice_needed;
    unsigned index;

#ifdef DEBUG_REALM
  public:
    // in debug mode, we'll track the state of a work item to avoid
    //  duplicate activations or activations after shutdown
    enum State {
      STATE_IDLE,
      STATE_ACTIVE,
      STATE_SHUTDOWN,
    };

    void make_inactive(void);  // called immediately before 'do_work'
    void shutdown_work_item(void);

  protected:
    atomic<State> state;
#endif
  };

};

#endif
