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

#include "realm/bgwork.h"
#include "realm/timers.h"
#include "realm/logging.h"
#include "realm/utils.h"
#include "realm/numa/numasysif.h"

static unsigned ctz(uint64_t v)
{
#ifdef REALM_ON_WINDOWS
  unsigned long index;
#ifdef _WIN64
  if(_BitScanForward64(&index, v))
    return index;
#else
  unsigned v_lo = v;
  unsigned v_hi = v >> 32;
  if(_BitScanForward(&index, v_lo))
    return index;
  else if(_BitScanForward(&index, v_hi))
    return index + 32;
#endif
  else
    return 0;
#else
  return __builtin_ctzll(v);
#endif
}

namespace Realm {

  Logger log_bgwork("bgwork");


  ////////////////////////////////////////////////////////////////////////
  //
  // class BackgroundWorkThread
  //

  class BackgroundWorkThread {
  public:
    BackgroundWorkThread(BackgroundWorkManager *_manager,
			 CoreReservationSet& crs,
			 const std::string& _name,
			 int _numa_domain,
			 bool _pin_thread, size_t _stack_size);
    ~BackgroundWorkThread(void);

    void main_loop(void);

    void join(void);

  protected:
    BackgroundWorkManager *manager;
    CoreReservation *rsrv;
    Thread *me;
    int numa_domain;
  };

  BackgroundWorkThread::BackgroundWorkThread(BackgroundWorkManager *_manager,
					     CoreReservationSet& crs,
					     const std::string& _name,
					     int _numa_domain,
					     bool _pin_thread,
					     size_t _stack_size)
    : manager(_manager)
    , numa_domain(_numa_domain)
  {
    CoreReservationParameters params;
    params.set_numa_domain(_numa_domain);
    if(_pin_thread)
      params.set_alu_usage(CoreReservationParameters::CORE_USAGE_EXCLUSIVE);

    rsrv = new CoreReservation(_name, crs, params);

    ThreadLaunchParameters tlp;
    tlp.set_stack_size(_stack_size);
    me = Thread::create_kernel_thread<BackgroundWorkThread,
				      &BackgroundWorkThread::main_loop>(this,
									tlp,
									*rsrv);
  }

  BackgroundWorkThread::~BackgroundWorkThread(void)
  {
    delete me;
    delete rsrv;
  }

  void BackgroundWorkThread::main_loop(void)
  {
    BackgroundWorkManager::Worker worker;
    worker.set_manager(manager);
    worker.set_numa_domain(numa_domain);

    log_bgwork.info() << "dedicated worker starting - worker=" << this << " numa=" << numa_domain;

    long long spin_until = -1;
    while(true) {
      uint32_t state_val = manager->worker_state.load_acquire();

      // if there's work to do, do it
      if(((state_val >> BackgroundWorkManager::STATE_ACTIVE_ITEMS_SHIFT) &
          BackgroundWorkManager::STATE_ACTIVE_ITEMS_MASK) != 0) {
        // reset spin timer
        spin_until = -1;

        // do work until there's none left
	while(worker.do_work(-1 /*max_time*/, 0 /*interrupt_flag*/)) {}

        // and then retest state variable
        continue;
      }

      // shutdown requested?
      if((state_val & BackgroundWorkManager::STATE_SHUTDOWN_BIT) != 0)
        break;

      // (potentially) spin for a bit and then sleep
      if(manager->cfg.worker_spin_interval > 0) {
        if(spin_until < 0) {
          spin_until = (manager->cfg.worker_spin_interval +
                        Clock::current_time_in_nanoseconds(true /*absolute*/));
          Thread::yield();
          continue;
        } else {
          // if we haven't exhausted the spin timer, spin more
	  if(Clock::current_time_in_nanoseconds(true /*absolute*/) < spin_until) {
	    Thread::yield();
            continue;
          }
        }
      }

      // all other options exhausted - try to increment sleeping worker
      //   count, but only if nothing has changed (false conflicts with
      //   other workers going to sleep aren't a big deal)
      {
        uint32_t expected = state_val;
        uint32_t newval = state_val + (1 << BackgroundWorkManager::STATE_SLEEPING_WORKERS_SHIFT);
        if(!manager->worker_state.compare_exchange(expected, newval))
          continue;
      }

      log_bgwork.debug() << "dedicated worker sleeping - worker=" << this;
      {
        Doorbell *db = Doorbell::get_thread_doorbell();
        db->prepare();
        if(manager->db_list.add_doorbell(db)) {
          db->wait();
        } else {
          // signal came first, so cancel our wait
          db->cancel();
        }
      }
      log_bgwork.debug() << "dedicated worker awake - worker=" << this;
    }

    log_bgwork.info() << "dedicated worker terminating - worker=" << this;
  }

  void BackgroundWorkThread::join(void)
  {
    me->join();
  }
	

  ////////////////////////////////////////////////////////////////////////
  //
  // class BackgroundWorkManager
  //

  BackgroundWorkManager::BackgroundWorkManager(void)
    : num_work_items(0)
    , worker_state(0)
  {
    for(unsigned i = 0; i < BITMASK_ARRAY_SIZE; i++)
      active_work_item_mask[i].store(0);

    for(unsigned i = 0; i < MAX_WORK_ITEMS; i++) {
      work_item_usecounts[i].store(0);
      work_items[i] = 0;
    }
  }

  BackgroundWorkManager::~BackgroundWorkManager(void)
  {
    assert(dedicated_workers.empty());
  }

  unsigned BackgroundWorkManager::assign_slot(BackgroundWorkItem *item)
  {
    AutoLock<> al(mutex);
    // TODO: reuse slots
    unsigned slot = num_work_items.load();
    assert(slot < MAX_WORK_ITEMS);
    work_items[slot] = item;
    int prev = work_item_usecounts[slot].fetch_add_acqrel(1);
    assert(prev == 0);
    (void)prev;
    num_work_items.store_release(slot + 1);
    return slot;
  }

  void BackgroundWorkManager::release_slot(unsigned slot)
  {
    // ensure slot does not have active work
    unsigned elem = slot / BITMASK_BITS;
    unsigned ofs = slot % BITMASK_BITS;
    BitMask mask = BitMask(1) << ofs;
    assert((active_work_item_mask[elem].load() & mask) == 0);

    // use count has to change from 1 (i.e. we are only remaining user)
    //  to 0 _before_ we enter critical section
    while(true) {
      int expected = 1;
      if(work_item_usecounts[slot].compare_exchange(expected, 0))
        break;
      Thread::yield();
    }

    {
      AutoLock<> al(mutex);
      // NOTE: no reuse of released slots right now
      work_items[slot] = 0;
    }
  }

  void BackgroundWorkManager::advertise_work(unsigned slot)
  {
    unsigned elem = slot / BITMASK_BITS;
    unsigned ofs = slot % BITMASK_BITS;
    
    BitMask mask = BitMask(1) << ofs;
    BitMask prev = active_work_item_mask[elem].fetch_or_acqrel(mask);
    assert((prev & mask) == 0);

    // increment the active items field in 'worker_state' and see if there are
    //  any sleeping workers that could help out
    uint32_t state_val = worker_state.fetch_add_acqrel(1 << STATE_ACTIVE_ITEMS_SHIFT);
    bool wake_worker = false;
    while(((state_val >> STATE_SLEEPING_WORKERS_SHIFT) &
           STATE_SLEEPING_WORKERS_MASK) != 0) {
      // use a CAS to decrement without underflowing - retry as needed
      if(worker_state.compare_exchange(state_val,
                                       state_val - (1 << STATE_SLEEPING_WORKERS_SHIFT))) {
        wake_worker = true;
        break;
      }
    }
    // actually waking a worker requires access to the doorbell list (or
    //   delegating to somebody else)
    if(wake_worker) {
      uint64_t tstate;
      uint64_t act_pops = db_mutex.attempt_enter(1, tstate);
      while(act_pops != 0) {
        db_list.notify_newest(act_pops, true /*prefer_spinning*/);

        // try to release mutex, but loop in case work was delegated to us
        act_pops = db_mutex.attempt_exit(tstate);
      }
    }
  }

  void BackgroundWorkManager::configure_from_cmdline(std::vector<std::string>& cmdline)
  {
    CommandLineParser cp;
    cp.add_option_int("-ll:bgwork", cfg.generic_workers.val)
      .add_option_int("-ll:bgnuma", cfg.per_numa_workers.val)
      .add_option_int("-ll:bgworkpin", cfg.pin_generic.val)
      .add_option_int("-ll:bgnumapin", cfg.pin_numa.val)
      .add_option_int("-ll:bgstack", cfg.worker_stacksize_in_kb.val)
      .add_option_int("-ll:bgspin", cfg.worker_spin_interval.val)
      .add_option_int("-ll:bgslice", cfg.work_item_timeslice.val);

    bool ok = cp.parse_command_line(cmdline);
    assert(ok);
  }

  void BackgroundWorkManager::start_dedicated_workers(Realm::CoreReservationSet& crs)
  {
    for(unsigned i = 0; i < cfg.generic_workers; i++)
      dedicated_workers.push_back(new BackgroundWorkThread(this,
							   crs,
							   stringbuilder() << "dedicated worker (generic) #" << (i + 1),
							   -1, // numa
							   cfg.pin_generic,
							   cfg.worker_stacksize_in_kb << 10));

    if(cfg.per_numa_workers > 0) {
      std::map<int, NumaNodeCpuInfo> cpuinfo;
      if(numasysif_numa_available() &&
	 numasysif_get_cpu_info(cpuinfo) &&
	 !cpuinfo.empty()) {
	for(std::map<int, NumaNodeCpuInfo>::const_iterator it = cpuinfo.begin();
	    it != cpuinfo.end();
	    ++it) {
	  const NumaNodeCpuInfo& ci = it->second;
	  // filter out any numa domains with insufficient core counts
	  int cores_needed = cfg.pin_numa ? cfg.per_numa_workers.val : 1;
	  if(ci.cores_available < cores_needed)
	    continue;

	  for(unsigned i = 0; i < cfg.per_numa_workers; i++)
	    dedicated_workers.push_back(new BackgroundWorkThread(this,
								 crs,
								 stringbuilder() << "dedicated worker (numa " << ci.node_id << ") #" << (i + 1),
								 ci.node_id,
								 cfg.pin_numa,
								 cfg.worker_stacksize_in_kb << 10));
	}
      } else {
	log_bgwork.warning() << "numa support not found (or not working)";
      }
    }
  }

  void BackgroundWorkManager::stop_dedicated_workers(void)
  {
    // set flag and signal any sleeping workers
    uint32_t prev_state = worker_state.fetch_or(STATE_SHUTDOWN_BIT);

    // use CAS to actually claim workers since work advertisers might get
    //   some of them
    while(true) {
      unsigned sleepers = ((prev_state >> STATE_SLEEPING_WORKERS_SHIFT) &
                           STATE_SLEEPING_WORKERS_MASK);
      if(sleepers == 0)
        break;

      if(worker_state.compare_exchange(prev_state,
                                       prev_state - (sleepers << STATE_SLEEPING_WORKERS_SHIFT))) {
        uint64_t tstate;
        uint64_t act_pops = db_mutex.attempt_enter(sleepers, tstate);
        while(act_pops != 0) {
          db_list.notify_newest(act_pops, true /*prefer_spinning*/);

          // try to release mutex, but loop in case work was delegated to us
          act_pops = db_mutex.attempt_exit(tstate);
        }

        break;
      }
    }

    // now join on all the threads
    for(std::vector<BackgroundWorkThread *>::iterator it = dedicated_workers.begin();
	it != dedicated_workers.end();
	++it) {
      (*it)->join();
      delete *it;
    }
    dedicated_workers.clear();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class BackgroundWorkItem
  //

  BackgroundWorkItem::BackgroundWorkItem(const std::string& _name)
    : name(_name)
    , manager(0)
    , index(0)
#ifdef DEBUG_REALM
    , state(STATE_IDLE)
#endif
  {}

  BackgroundWorkItem::~BackgroundWorkItem(void)
  {
#ifdef DEBUG_REALM
    State old_state = state.load();
    if(old_state != STATE_SHUTDOWN) {
      log_bgwork.fatal() << "invalid destruction: item=" << ((void *)this)
			 << " name='" << name
			 << "' oldstate=" << old_state;
      abort();
    }
#endif
    if(manager)
      manager->release_slot(index);
 }

  void BackgroundWorkItem::add_to_manager(BackgroundWorkManager *_manager,
					  int _numa_domain /*= -1*/,
					  long long _min_timeslice_needed /*= -1*/)
  {
    manager = _manager;
    numa_domain = _numa_domain;
    min_timeslice_needed = _min_timeslice_needed;
    index = manager->assign_slot(this);
    log_bgwork.info() << "new work item: manager=" << manager
		      << " item=" << this
		      << " slot=" << index << " name=" << name
		      << " domain=" << numa_domain
		      << " timeslice=" << min_timeslice_needed;
  }

  // mark this work item as active (i.e. having work to do)
  void BackgroundWorkItem::make_active(void)
  {
    if(!manager) return;
    log_bgwork.debug() << "work advertised: manager=" << manager
		       << " item=" << this
		       << " slot=" << index;
#ifdef DEBUG_REALM
    State old_state = state.exchange(STATE_ACTIVE);
    if(old_state != STATE_IDLE) {
      log_bgwork.fatal() << "invalid make_active: item=" << ((void *)this)
			 << " name='" << name
			 << "' oldstate=" << old_state;
      abort();
    }
#endif
    manager->advertise_work(index);
  }

#ifdef DEBUG_REALM
  // called immediately before 'do_work'
  void BackgroundWorkItem::make_inactive(void)
  {
    State old_state = state.exchange(STATE_IDLE);
    if(old_state != STATE_ACTIVE) {
      log_bgwork.fatal() << "invalid make_inactive: item=" << ((void *)this)
			 << " name='" << name
			 << "' oldstate=" << old_state;
      abort();
    }
  }

  void BackgroundWorkItem::shutdown_work_item(void)
  {
    State old_state = state.exchange(STATE_SHUTDOWN);
    if(old_state != STATE_IDLE) {
      log_bgwork.fatal() << "invalid shutdown: item=" << ((void *)this)
                         << " name='" << name
			 << "' oldstate=" << old_state;
      abort();
    }
  }
#endif


  ////////////////////////////////////////////////////////////////////////
  //
  // class BackgroundWorkManager::Worker
  //

  BackgroundWorkManager::Worker::Worker(void)
    : manager(0)
    , starting_slot(0)
    , max_timeslice(-1)
    , numa_domain(-1)
  {
    for(unsigned i = 0; i < BITMASK_ARRAY_SIZE; i++) {
      known_work_item_mask[i] = 0;
      allowed_work_item_mask[i] = 0;
    }
  }

  BackgroundWorkManager::Worker::~Worker(void)
  {}

  void BackgroundWorkManager::Worker::set_manager(BackgroundWorkManager *_manager)
  {
    manager = _manager;
    // reset our cache of allowed work items
    for(unsigned i = 0; i < BITMASK_ARRAY_SIZE; i++) {
      known_work_item_mask[i] = 0;
      allowed_work_item_mask[i] = 0;
    }
  }

  void BackgroundWorkManager::Worker::set_max_timeslice(long long _timeslice_in_ns)
  {
    max_timeslice = _timeslice_in_ns;
    // reset our cache of allowed work items
    for(unsigned i = 0; i < BITMASK_ARRAY_SIZE; i++) {
      known_work_item_mask[i] = 0;
      allowed_work_item_mask[i] = 0;
    }
  }

  void BackgroundWorkManager::Worker::set_numa_domain(int _numa_domain)
  {
    numa_domain = _numa_domain;
    // reset our cache of allowed work items
    for(unsigned i = 0; i < BITMASK_ARRAY_SIZE; i++) {
      known_work_item_mask[i] = 0;
      allowed_work_item_mask[i] = 0;
    }
  }
  
  bool BackgroundWorkManager::Worker::do_work(long long max_time_in_ns,
					      atomic<bool> *interrupt_flag)
  {
    // set our deadline for returning
    long long work_until_time = ((max_time_in_ns > 0) ?
                                   (Clock::current_time_in_nanoseconds(true /*absolute*/) +
                                    max_time_in_ns) :
				   -1);

    bool did_work = true;
    while(true) {
      // if we've exhausted the known work items, loop back around and
      //  check to see if any new work items have showed up
      if(starting_slot >= manager->num_work_items.load_acquire()) {
	// if we get here twice in a row without doing any work, return
	//  to the caller to let them spin/sleep/whatever
	if(!did_work) return false;
	did_work = false;

	starting_slot = 0;
        // TODO: if/when slots are reused, we need a way to invalidate
        //  our known/allowed masks
      }

      // look at a whole BitMask entry at once, skipping over 0's
      unsigned elem = starting_slot / BITMASK_BITS;
      unsigned ofs = starting_slot % BITMASK_BITS;
      BitMask active_mask = (manager->active_work_item_mask[elem].load() &
                             (~BitMask(0) << ofs));

      // are there any bits set that we've not seen before?
      BitMask unknown_mask = active_mask & ~known_work_item_mask[elem];
      while(unknown_mask != 0) {
        BitMask unknown_bit = (unknown_mask & ~(unknown_mask - 1));
        unsigned unknown_slot = (elem * BITMASK_BITS) + ctz(unknown_bit);

        // attempt to increment the user count for this slot so we can
        //  peek at information that tells us if we want to service it
        int prev_count = manager->work_item_usecounts[unknown_slot].fetch_add_acqrel(1);
        if(prev_count > 0) {
          // slot pointer is valid and can't change until we decrement the
          //  use count again
	  BackgroundWorkItem *item = manager->work_items[unknown_slot];
          assert(item != 0);

          bool allowed = true;

	  // don't take things whose timeslice is too long
	  if((max_timeslice > 0) && (item->min_timeslice_needed > 0) &&
	     (max_timeslice < item->min_timeslice_needed)) {
            allowed = false;
	  }

	  // don't take things that are in the wrong numa domain
	  if((numa_domain >= 0) && (item->numa_domain >= 0) &&
	     (numa_domain != item->numa_domain)) {
            allowed = false;
	  }

          log_bgwork.info() << "worker " << this << " discovered slot " << unknown_slot << " (" << item->name << ") allowed=" << allowed;

          if(allowed)
            allowed_work_item_mask[elem] |= unknown_bit;
        }
        // unconditional decrement to match the increment
        manager->work_item_usecounts[unknown_slot].fetch_sub(1);

        known_work_item_mask[elem] |= unknown_bit;
        unknown_mask &= ~unknown_bit;
      }

      // now that everything is at least known, limit the active mask to
      //  just things we're allowed to take
      BitMask allowed_mask = (active_mask & allowed_work_item_mask[elem]);

      while(allowed_mask != 0) {
	// this leaves only the least significant 1 bit set
	BitMask target_bit = allowed_mask & ~(allowed_mask - 1);
	// attempt to clear this bit
	BitMask prev = manager->active_work_item_mask[elem].fetch_and_acqrel(~target_bit);
	if(prev & target_bit) {
	  // success!

          // decrement count of active work items - temporary underflow is
          //  possible here, so no way to sanity-check state
          manager->worker_state.fetch_sub(1 << BackgroundWorkManager::STATE_ACTIVE_ITEMS_SHIFT);

	  unsigned slot = ((elem * BITMASK_BITS) + ctz(target_bit));
	  log_bgwork.debug() << "work claimed: manager=" << manager
			    << " slot=" << slot
			    << " worker=" << this;
	  long long t_start = Clock::current_time_in_nanoseconds(true /*absolute*/);
	  // don't spend more than 1ms on any single task before going on to the
	  //  next thing - TODO: pull this out as a config variable
	  long long t_quantum = (manager->cfg.work_item_timeslice + t_start);
	  if((work_until_time > 0) && (work_until_time < t_quantum))
	    t_quantum = work_until_time;

          // increase the use count for this slot - this should NEVER see
          //  an invalid slot because we have claimed a work request and
          //  not ack'd it yet
          int prev_usecount = manager->work_item_usecounts[slot].fetch_add_acqrel(1);
          assert(prev_usecount > 0);
          (void)prev_usecount;

	  BackgroundWorkItem *item = manager->work_items[slot];
#ifdef DEBUG_REALM
	  item->make_inactive();
#endif
          while(true) {
            bool requeue = item->do_work(TimeLimit::absolute(t_quantum, interrupt_flag));
            if(requeue) {
              // we can just call this item's work function again if we're not out
              //  of time and if there's nothing else to do
              uint32_t other_work_items = (manager->worker_state.load() >> BackgroundWorkManager::STATE_ACTIVE_ITEMS_SHIFT);
              if(other_work_items == 0) {
                long long now = Clock::current_time_in_nanoseconds(true /*absolute*/);
                if((work_until_time <= 0) || (work_until_time > now)) {
                  // update t_quantum and then loop back around
                  t_quantum = (manager->cfg.work_item_timeslice + now);
                  if((work_until_time > 0) && (work_until_time < t_quantum))
                    t_quantum = work_until_time;
                  continue;
                }
              }
              // if we fall through to here, we've got other stuff to do, so
              //  actually enqueue the item before going on
              item->make_active();
              break;
            } else
              break;
          }
#ifdef REALM_BGWORK_PROFILE
	  long long t_stop = Clock::current_time_in_nanoseconds(true /*absolute*/);
	  long long elapsed = t_stop - t_start;
	  long long overshoot = ((t_stop > t_quantum) ?
	                           (t_stop - t_quantum) :
	                           0);
	  log_bgwork.print() << "work: slot=" << slot << " elapsed=" << elapsed << " overshoot=" << overshoot;
#endif
          // we're done with this slot for now
          manager->work_item_usecounts[slot].fetch_sub_acqrel(1);

	  starting_slot = slot + 1;
	  did_work = true;
	  break;
	} else {
	  // loop around and try other bits
	  allowed_mask &= ~target_bit;
	}
      }

      // if we get here with a zero mask, skip ahead to next chunk of bits
      if(allowed_mask == 0)
        starting_slot = (elem + 1) * BITMASK_BITS;

      // before we loop around, see if there's been an interupt requested or
      //  we've used all the time permitted
      if(interrupt_flag != 0) {
	if(interrupt_flag->load())
	  return true;
      }
      if(work_until_time > 0) {
	long long now = Clock::current_time_in_nanoseconds(true /*absolute*/);
	if(now >= work_until_time)
	  return true;
      }
    }
  }


};
