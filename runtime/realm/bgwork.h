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

#ifndef REALM_BGWORK_H
#define REALM_BGWORK_H

#include "realm/atomics.h"
#include "realm/threads.h"
#include "realm/mutex.h"
#include "realm/cmdline.h"

#include <string>

#if defined(__i386__) || defined(__x86_64__)
#define REALM_TIMELIMIT_USE_RDTSC
#endif

namespace Realm {

  class BackgroundWorkItem;
  class BackgroundWorkThread;

  // a central theme with background workers is to place limits on how long
  //  they do any one task - this is described using a TimeLimit object
  class TimeLimit {
  public:
    // default constructor generates a time limit infinitely far in the future
    TimeLimit();

    // these constructors describe a limit in terms of Realm's clock (or
    //  RDTSC, if available)
    static TimeLimit absolute(long long absolute_time_in_nsec,
			      atomic<bool> *_interrupt_flag = 0);
    static TimeLimit relative(long long relative_time_in_nsec,
			      atomic<bool> *_interrupt_flag = 0);

    // often the desired time limit is "idk, something responsive", so
    //  have a common way to pick a completely-made-up number
    static TimeLimit responsive();

    bool is_expired() const;
    bool will_expire(long long additional_nsec) const;

#ifdef REALM_TIMELIMIT_USE_RDTSC
    static void calibrate_rdtsc();
#endif

  protected:
#ifdef REALM_TIMELIMIT_USE_RDTSC
    static uint64_t rdtsc_per_64k_nanoseconds;
    uint64_t limit_rdtsc;
#else
    long long limit_time;
#endif
    atomic<bool> *interrupt_flag;
  };

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
      unsigned known_work_items;
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
    atomic<unsigned> num_work_items;
    atomic<int> active_work_items;
    atomic<BitMask> active_work_item_mask[BITMASK_ARRAY_SIZE];

    BackgroundWorkItem *work_items[MAX_WORK_ITEMS];

    friend class BackgroundWorkThread;

    Mutex mutex;
    CondVar condvar;
    atomic<int> shutdown_flag;
    atomic<int> sleeping_workers;
    std::vector<BackgroundWorkThread *> dedicated_workers;
  };

  class BackgroundWorkItem {
  public:
    BackgroundWorkItem(const std::string& _name);
    virtual ~BackgroundWorkItem(void);

    void add_to_manager(BackgroundWorkManager *_manager,
			int _numa_domain = -1,
			long long _min_timeslice_needed = -1);

    virtual void do_work(TimeLimit work_until) = 0;

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

#include "realm/bgwork.inl"

#endif
