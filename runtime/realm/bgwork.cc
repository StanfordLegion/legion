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
  // class TimeLimit
  //

#ifdef REALM_TIMELIMIT_USE_RDTSC
  // set a default assuming a 2GHz clock so that we do something sensible
  //  if calibration doesn't happen for some reason

  /*static*/ uint64_t TimeLimit::rdtsc_per_64k_nanoseconds = 131072;

  /*static*/ void TimeLimit::calibrate_rdtsc()
  {
    // measure 256k nanoseconds (i.e. 0.000256s) and see how many rdtscs we get
    uint64_t ts1 = __rdtsc();
    long long stop_at = Clock::current_time_in_nanoseconds() + 262144;
    uint64_t ts2;
    size_t loop_count = 0;
    do {
      loop_count++;
      ts2 = __rdtsc();
    } while(Clock::current_time_in_nanoseconds() < stop_at);

    uint64_t per_64k_nsec = (ts2 - ts1) >> 2;
    float freq = per_64k_nsec / 65536.0;
    // ignore values that seem nonsensical - look for a frequency between
    //   1-10 GHz and make sure we managed at least 256 loops (i.e. <= 1us/loop)
    if((freq >= 1.0) && (freq <= 10.0) && (loop_count >= 256)) {
      log_bgwork.info() << "rdtsc calibration: per_64k_nsec=" << per_64k_nsec
			<< " freq=" << freq << " count=" << loop_count;
      rdtsc_per_64k_nanoseconds = per_64k_nsec;
    } else {
      log_bgwork.warning() << "rdtsc calibration failed: per_64k_nsec=" << per_64k_nsec
			   << " freq=" << freq << " count=" << loop_count
			   << " - timeouts will be based on a 2GHz clock";
    }
  }
#endif

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

    while(manager->shutdown_flag.load() == 0) {
      // see if there is work to do
      if(manager->active_work_items.load() != 0) {
	// do work until there's nothing left
	while(worker.do_work(-1 /*max_time*/, 0 /*interrupt_flag*/)) {}
      } else {
	// (potentially) spin for a bit and then sleep
	long long spin_until = (manager->cfg.worker_spin_interval +
				Clock::current_time_in_nanoseconds());
	while(manager->active_work_items.load() == 0) {
	  if(manager->shutdown_flag.load() != 0)
	    break;
	  if(Clock::current_time_in_nanoseconds() < spin_until) {
	    Thread::yield();
	  } else {
	    log_bgwork.info() << "dedicated worker sleeping - worker=" << this;
	    {
	      AutoLock<> a(manager->mutex);

	      // re-check shutdown flag with mutex held
	      if(manager->shutdown_flag.load() != 0)
		break;

	      // indicate our desire to sleep and then check the 
	      //  active_work_items count one more time
	      int other_sleepers = manager->sleeping_workers.fetch_add(1);
	      if(manager->active_work_items.load() == 0) {
		manager->condvar.wait();
	      } else {
		manager->sleeping_workers.store(0);
		if(other_sleepers)
		  manager->condvar.broadcast();
	      }
	    }
	    log_bgwork.info() << "dedicated worker awake - worker=" << this;
	    // if we're woken up, there's probably work to do
	    break;
	  }
	}
      }
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
    , active_work_items(0)
    , condvar(mutex)
    , shutdown_flag(0)
    , sleeping_workers(0)
  {
    for(unsigned i = 0; i < BITMASK_ARRAY_SIZE; i++)
      active_work_item_mask[i].store(0);

    for(unsigned i = 0; i < MAX_WORK_ITEMS; i++)
      work_items[i] = 0;
  }

  BackgroundWorkManager::~BackgroundWorkManager(void)
  {
    assert(dedicated_workers.empty());
  }

  unsigned BackgroundWorkManager::assign_slot(BackgroundWorkItem *item)
  {
    unsigned slot = num_work_items.fetch_add(1);
    assert(slot < MAX_WORK_ITEMS);
    work_items[slot] = item;
    return slot;
  }

  void BackgroundWorkManager::release_slot(unsigned slot)
  {
    // ensure slot does not have active work
    unsigned elem = slot / BITMASK_BITS;
    unsigned ofs = slot % BITMASK_BITS;
    BitMask mask = BitMask(1) << ofs;
    assert((active_work_item_mask[elem].load() & mask) == 0);

    // NOTE: no reuse of released slots right now
    work_items[slot] = 0;
  }

  void BackgroundWorkManager::advertise_work(unsigned slot)
  {
    unsigned elem = slot / BITMASK_BITS;
    unsigned ofs = slot % BITMASK_BITS;
    
    BitMask mask = BitMask(1) << ofs;
    BitMask prev = active_work_item_mask[elem].fetch_or_acqrel(mask);
    assert((prev & mask) == 0);

    int prev_count = active_work_items.fetch_add(1);
    if((prev_count == 0) && (sleeping_workers.load() > 0)) {
      AutoLock<> a(mutex);
      // check again - somebody else may have handled this
      if(sleeping_workers.load() > 0) {
	condvar.broadcast();
	sleeping_workers.store(0);
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

#ifdef REALM_TIMELIMIT_USE_RDTSC
    TimeLimit::calibrate_rdtsc();
#endif
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
    shutdown_flag.store(1);
    {
      AutoLock<> a(mutex);
      if(sleeping_workers.load() != 0)
	condvar.broadcast();
      sleeping_workers.store(0);
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
    log_bgwork.info() << "work advertised: manager=" << manager
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
    , known_work_items(0)
    , max_timeslice(-1)
    , numa_domain(-1)
  {
    for(unsigned i = 0; i < BITMASK_ARRAY_SIZE; i++)
      allowed_work_item_mask[i] = 0;
  }

  BackgroundWorkManager::Worker::~Worker(void)
  {}

  void BackgroundWorkManager::Worker::set_manager(BackgroundWorkManager *_manager)
  {
    manager = _manager;
    // reset our cache of allowed work items
    known_work_items = 0;
    for(unsigned i = 0; i < BITMASK_ARRAY_SIZE; i++)
      allowed_work_item_mask[i] = 0;
  }

  void BackgroundWorkManager::Worker::set_max_timeslice(long long _timeslice_in_ns)
  {
    max_timeslice = _timeslice_in_ns;
    // reset our cache of allowed work items
    known_work_items = 0;
    for(unsigned i = 0; i < BITMASK_ARRAY_SIZE; i++)
      allowed_work_item_mask[i] = 0;
  }

  void BackgroundWorkManager::Worker::set_numa_domain(int _numa_domain)
  {
    numa_domain = _numa_domain;
    // reset our cache of allowed work items
    known_work_items = 0;
    for(unsigned i = 0; i < BITMASK_ARRAY_SIZE; i++)
      allowed_work_item_mask[i] = 0;
  }
  
  bool BackgroundWorkManager::Worker::do_work(long long max_time_in_ns,
					      atomic<bool> *interrupt_flag)
  {
    // set our deadline for returning
    long long work_until_time = ((max_time_in_ns > 0) ?
				   (Clock::current_time_in_nanoseconds() + max_time_in_ns) :
				   -1);

    bool did_work = true;
    while(true) {
      // if we've exhausted the known work items, loop back around and
      //  check to see if any new work items have showed up
      if(starting_slot >= known_work_items) {
	// if we get here twice in a row without doing any work, return
	//  to the caller to let them spin/sleep/whatever
	if(!did_work) return false;
	did_work = false;
	unsigned cur_work_items = manager->num_work_items.load();
	while(known_work_items < cur_work_items) {
	  BackgroundWorkItem *item = manager->work_items[known_work_items];

	  // ignore deleted items
	  if(item == 0) {
	    known_work_items++;
	    continue;
	  }

	  // don't take things whose timeslice is too long
	  if((max_timeslice > 0) && (item->min_timeslice_needed > 0) &&
	     (max_timeslice < item->min_timeslice_needed)) {
	    known_work_items++;
	    continue;
	  }

	  // don't take things that are in the wrong numa domain
	  if((numa_domain >= 0) && (item->numa_domain >= 0) &&
	     (numa_domain != item->numa_domain)) {
	    known_work_items++;
	    continue;
	  }

	  unsigned elem = known_work_items / BITMASK_BITS;
	  unsigned ofs = known_work_items % BITMASK_BITS;
	  BitMask mask = BitMask(1) << ofs;
	  allowed_work_item_mask[elem] |= mask;

	  known_work_items++;
	}
	starting_slot = 0;
      }

      // look at a whole BitMask entry at once, skipping over 0's
      unsigned elem = starting_slot / BITMASK_BITS;
      unsigned ofs = starting_slot % BITMASK_BITS;
      BitMask mask;
      mask = (manager->active_work_item_mask[elem].load() &
	      allowed_work_item_mask[elem] &
	      (~BitMask(0) << ofs));
      while(mask != 0) {
	// this leaves only the least significant 1 bit set
	BitMask target_bit = mask & ~(mask - 1);
	// attempt to clear this bit
	BitMask prev = manager->active_work_item_mask[elem].fetch_and_acqrel(~target_bit);
	if(prev & target_bit) {
	  // success!
	  manager->active_work_items.fetch_sub(1);
	  unsigned slot = ((elem * BITMASK_BITS) + ctz(target_bit));
	  log_bgwork.info() << "work claimed: manager=" << manager
			    << " slot=" << slot
			    << " worker=" << this;
	  long long t_start = Clock::current_time_in_nanoseconds();
	  // don't spend more than 1ms on any single task before going on to the
	  //  next thing - TODO: pull this out as a config variable
	  long long t_quantum = (manager->cfg.work_item_timeslice + t_start);
	  if((work_until_time > 0) && (work_until_time < t_quantum))
	    t_quantum = work_until_time;
	  BackgroundWorkItem *item = manager->work_items[slot];
#ifdef DEBUG_REALM
	  item->make_inactive();
#endif
	  item->do_work(TimeLimit::absolute(t_quantum, interrupt_flag));
#ifdef REALM_BGWORK_PROFILE
	  long long t_stop = Clock::current_time_in_nanoseconds();
	  long long elapsed = t_stop - t_start;
	  long long overshoot = ((t_stop > t_quantum) ?
	                           (t_stop - t_quantum) :
	                           0);
	  log_bgwork.print() << "work: slot=" << slot << " elapsed=" << elapsed << " overshoot=" << overshoot;
#endif
	  starting_slot = slot + 1;
	  did_work = true;
	  break;
	} else {
	  // loop around and try other bits
	  mask &= ~target_bit;
	}
      }
      // if we get here with a zero mask, skip ahead to next chunk of bits
      if(mask == 0)
	starting_slot = (elem + 1) * BITMASK_BITS;

      // before we loop around, see if there's been an interupt requested or
      //  we've used all the time permitted
      if(interrupt_flag != 0) {
	if(interrupt_flag->load())
	  return true;
      }
      if(work_until_time > 0) {
	long long now = Clock::current_time_in_nanoseconds();
	if(now >= work_until_time)
	  return true;
      }
    }
  }


};
