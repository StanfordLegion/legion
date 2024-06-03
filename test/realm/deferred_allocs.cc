/* Copyright 2024 Stanford University, NVIDIA Corporation
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

// Realm test for deferred instance allocation

#include <realm.h>
#include <realm/cmdline.h>

#include "philox.h"

#include <math.h>
#include <time.h>

#include "osdep.h"

using namespace Realm;

Logger log_app("app");

typedef Philox_2x32<> PRNG;

class PRNGSequence {
public:
  PRNGSequence(unsigned _seed1, unsigned _seed2, unsigned _ctr = 0)
    : seed1(_seed1), seed2(_seed2), ctr(_ctr)
  {}

  unsigned next_int(unsigned n)
  {
    return PRNG::rand_int(seed1, seed2, ctr++, n);
  }

protected:
  unsigned seed1, seed2, ctr;
};

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  WORKER_TASK,
  ALLOC_RESULT_TASK,
};

struct TestConfig {
  unsigned seed;
  bool directed_tests;
  int trials_per_mem;
  int trial_length;
  int buckets_min;
  int buckets_max;
  bool all_memories;
  bool check_alloc_result;
};

struct InstanceInfo {
  RegionInstance inst;
  Event create_event;
  bool alloc_result;
  UserEvent destroy_event;
  enum State {
    ALLOC_PENDING,
    ALLOC_FAILED,
    ALLOCED,
    DEST_PENDING,
    DESTROYED,
  };
  State state;

  InstanceInfo(RegionInstance _inst, Event _create_event, bool _alloc_result)
    : inst(_inst), create_event(_create_event), alloc_result(_alloc_result)
    , destroy_event(UserEvent::NO_USER_EVENT), state(ALLOC_PENDING)
  {}
};

void alloc_result_task(const void *args, size_t arglen,
		       const void *userdata, size_t userlen, Processor p)
{
  // args is a ProfilingResponse whose user_data is a UserEvent
  ProfilingResponse pr(args, arglen);
  UserEvent alloc_result_event = *static_cast<const UserEvent *>(pr.user_data());

  ProfilingMeasurements::InstanceAllocResult result;
  bool ok = pr.get_measurement(result);
  assert(ok);

  if(result.success)
    alloc_result_event.trigger();
  else
    alloc_result_event.cancel();
}

void directed_test_memory(const TestConfig& config, Memory m, Processor p,
			  const char *name, const char *testdesc)
{
  log_app.info() << "directed test: " << name << " memory=" << m;
  
  const char *pos = testdesc;
  
  // first thing is the number of buckets we need
  int buckets = strtol(pos, (char **)&pos, 10);

  FieldID fid = 1;
  size_t field_size = 32; // this avoids issues with instance alignment

  size_t bucket_size = m.capacity() / field_size / buckets;
  assert(bucket_size > 0);

  std::vector<InstanceInfo> insts;
  std::map<FieldID, size_t> field_sizes;
  field_sizes[fid] = field_size;

  while(*pos) {
    if(*pos == ' ') {
      pos++;
      continue;
    }

    char cmd = *pos++;
    int amt = strtol(pos, (char **)&pos, 10);

    switch(cmd) {
    case 'a': // allocate
      {
	size_t idx = insts.size();
	Rect<1> rect(1, amt * bucket_size);

	// we need a profiling request set that ignores failures
	ProfilingRequestSet prs;
	prs.add_request(Processor::NO_PROC, 0 /*ignore*/)
	  .add_measurement<ProfilingMeasurements::InstanceStatus>();
	UserEvent alloc_result_event = UserEvent::NO_USER_EVENT;
	if(config.check_alloc_result) {
	  alloc_result_event = UserEvent::create_user_event();
	  prs.add_request(p, ALLOC_RESULT_TASK,
			  &alloc_result_event, sizeof(alloc_result_event))
	    .add_measurement<ProfilingMeasurements::InstanceAllocResult>();
	}

	RegionInstance inst;
	Event e = RegionInstance::create_instance(inst, m, rect,
						  field_sizes, 0 /*SOA*/, prs);

	bool alloc_result = true;
	if(config.check_alloc_result) {
	  // alloc result should be delivered without any further activity
	  // use alarm since we need to yield to a profiling task
	  alarm(10);
	  bool poisoned = false;
          alloc_result_event.wait_faultaware(poisoned);
	  alarm(0);
	  alloc_result = !poisoned;
        }

	insts.push_back(InstanceInfo(inst, e, alloc_result));
	log_app.debug() << "alloc #" << idx << ": size=" << amt
			<< " inst=" << inst << " ready=" << e << " result=" << alloc_result;

	break;
      }

    case 's': // successful allocation
      {
	InstanceInfo& ii = insts[amt];
	assert(ii.state == InstanceInfo::ALLOC_PENDING);
	log_app.debug() << "success #" << amt << " inst=" << ii.inst;
	if(config.check_alloc_result)
	  assert(ii.alloc_result);
	bool poisoned = false;
	// normal apps should not call external_wait, but we do it here to
	//  detect hangs more easily and we know nothing else wants to run
	bool triggered = ii.create_event.external_timedwait_faultaware(poisoned,
								       1000000000);
	if(!triggered) {
	  log_app.fatal() << "alloc #" << amt << " inst=" << ii.inst
			  << " want=success got=hang"
			  << " test=" << name << " (" << testdesc << ")";
	  abort();
	}
	if(poisoned) {
	  log_app.fatal() << "alloc #" << amt << " inst=" << ii.inst
			  << " want=success got=failed"
			  << " test=" << name << " (" << testdesc << ")";
	  abort();
	}
	ii.state = InstanceInfo::ALLOCED;
	break;
      }

    case 'f': // failed allocation (expected)
    case 'u': // failed allocation (unexpected)
      {
	InstanceInfo& ii = insts[amt];
	assert(ii.state == InstanceInfo::ALLOC_PENDING);
	log_app.debug() << "failed #" << amt << " inst=" << ii.inst;
	if(config.check_alloc_result)
	  assert(ii.alloc_result == (cmd == 'u'));
	bool poisoned = false;
	// normal apps should not call external_wait, but we do it here to
	//  detect hangs more easily and we know nothing else wants to run
	bool triggered = ii.create_event.external_timedwait_faultaware(poisoned,
								       1000000000);
	if(!triggered) {
	  log_app.fatal() << "alloc #" << amt << " inst=" << ii.inst
			  << " want=failed got=hang"
			  << " test=" << name << " (" << testdesc << ")";
	  abort();
	}
	if(!poisoned) {
	  log_app.fatal() << "alloc #" << amt << " inst=" << ii.inst
			  << " want=failed got=success"
			  << " test=" << name << " (" << testdesc << ")";
	  abort();
	}
	ii.state = InstanceInfo::ALLOC_FAILED;
	ii.inst.destroy();
	break;
      }

    case 'n': // allocation NOT ready
      {
	InstanceInfo& ii = insts[amt];
	assert(ii.state == InstanceInfo::ALLOC_PENDING);
	log_app.debug() << "success #" << amt << " inst=" << ii.inst;
	bool poisoned = false;
	// normal apps should not call external_wait, but we do it here to
	//  detect hangs more easily and we know nothing else wants to run
	bool triggered = ii.create_event.external_timedwait_faultaware(poisoned, 10000000);
	if(triggered) {
	  log_app.fatal() << "alloc #" << amt << " inst=" << ii.inst
			  << " want=notready got=" << (poisoned ? "failed" : "success")
			  << " test=" << name << " (" << testdesc << ")";
	  abort();
	}
	break;
      }

    case 'd': // destroy
      {
	InstanceInfo& ii = insts[amt];
	assert(ii.state == InstanceInfo::ALLOCED);
	ii.destroy_event = UserEvent::create_user_event();
	log_app.debug() << "destroy #" << amt << " inst=" << ii.inst
			<< " event=" << ii.destroy_event;
	ii.inst.destroy(ii.destroy_event);
	ii.state = InstanceInfo::DEST_PENDING;
	break;
      }

    case 'i': // destroy (instant)
      {
	InstanceInfo& ii = insts[amt];
	assert(ii.state == InstanceInfo::ALLOCED);
	ii.destroy_event = UserEvent::NO_USER_EVENT;
	log_app.debug() << "destroy #" << amt << " inst=" << ii.inst
			<< " event=" << ii.destroy_event;
	ii.inst.destroy(ii.destroy_event);
	ii.state = InstanceInfo::DESTROYED;
	break;
      }

    case 't': // trigger
      {
	InstanceInfo& ii = insts[amt];
	assert(ii.state == InstanceInfo::DEST_PENDING);
	log_app.debug() << "trigger #" << amt << " inst=" << ii.inst
			<< " event=" << ii.destroy_event;
	ii.destroy_event.trigger();
	ii.state = InstanceInfo::DESTROYED;
	break;
      }

    case 'c': // cancel
      {
	InstanceInfo& ii = insts[amt];
	assert(ii.state == InstanceInfo::DEST_PENDING);
	log_app.debug() << "cancel #" << amt << " inst=" << ii.inst
			<< " event=" << ii.destroy_event;
	ii.destroy_event.cancel();
	ii.state = InstanceInfo::ALLOCED;
	break;
      }

    default: assert(0);
    }
  }

  // go through instances and clean up whatever's left
  for(std::vector<InstanceInfo>::iterator it = insts.begin();
      it != insts.end();
      ++it)
    switch(it->state) {
    case InstanceInfo::ALLOC_PENDING:
      {
	bool poisoned = false;
	bool triggered = it->create_event.external_timedwait_faultaware(poisoned, 1000000);
	assert(triggered);
	it->inst.destroy();
	break;
      }

    case InstanceInfo::ALLOCED:
      {
	it->inst.destroy();
	break;
      }

    case InstanceInfo::DEST_PENDING:
      {
	it->destroy_event.trigger();
	break;
      }

    default: break;
    }
}

void alloc_chain_test(Memory m, Processor p, int num_chains, int chain_length)
{
  log_app.info() << "alloc chain test: memory=" << m;
  
  FieldID fid = 1;
  size_t field_size = 32; // this avoids issues with instance alignment

  size_t bucket_size = m.capacity() / field_size / num_chains;
  assert(bucket_size > 0);

  std::vector<InstanceInfo> insts;
  std::map<FieldID, size_t> field_sizes;
  field_sizes[fid] = field_size;

  // an initial instance is allocated to tie up all of memory
  RegionInstance blockage;
  {
    Rect<1> rect(1, num_chains * bucket_size);
    RegionInstance::create_instance(blockage, m, rect,
				    field_sizes, 0 /*SOA*/, ProfilingRequestSet()).wait();
  }
  UserEvent e_start = UserEvent::create_user_event();

  std::vector<Event> c_events(num_chains, e_start);

  for(int i = 0; i < chain_length; i++)
    for(int c = 0; c < num_chains; c++) {
      // perform a deferred allocation for this chain
      Rect<1> rect(1, bucket_size);
      RegionInstance c_inst;
      Event e = RegionInstance::create_instance(c_inst, m, rect,
						field_sizes, 0 /*SOA*/,
						ProfilingRequestSet(),
						c_events[c]);
      // dummy task launch based on this allocation
      e = p.spawn(WORKER_TASK, 0, 0, e);
      // now delete that instance - TODO: a completion event from destroy()
      //  would be safer
      c_inst.destroy(e);
      c_events[c] = e;
    }

  Event all_chains = Event::merge_events(c_events);

  // chains all built - now let's remove the blockage
  blockage.destroy();
  e_start.trigger();

  all_chains.wait();
} 

#if RANDOM_TESTS
// random tests would be nice, but the code below isn't right
struct InstanceTracker {
  int size; // in buckets
  RegionInstance inst;
  Event create_event;
  int exp_base, act_base;
  UserEvent destroy_event;
  enum CreateStatus {
    CS_PENDING,
    CS_CREATED,
    CS_FAILED
  };
  CreateStatus cstatus;
  enum DestroyStatus {
    DS_ACTIVE,
    DS_PENDING,
    DS_TRIGGERED
  };
  DestroyStatus dstatus;

  InstanceTracker(int _size, RegionInstance _inst, Event _create_event,
		  int _exp_base)
    : size(_size), inst(_inst), create_event(_create_event)
    , exp_base(_exp_base)
    , act_base(-1)
    , destroy_event(UserEvent::NO_USER_EVENT)
    , cstatus(CS_PENDING), dstatus(DS_ACTIVE)
  {}
};

int pick_size(PRNGSequence& seq, int buckets)
{
  // bias towards lower numbers
  int size = buckets - int(sqrtf(seq.next_int(buckets * buckets)));
  return size;
}

int set_bits(unsigned& bitmask, int count, int slots, int start_at = 0)
{
  unsigned toset = (1 << count) - 1;
  for(int i = start_at; i <= (slots - count); i++)
    if((bitmask & (toset << i)) == 0) {
      bitmask |= (toset << i);
      return i;
    }
  return -1;
}

void clear_bits(unsigned& bitmask, int count, int base)
{
  unsigned toclear = (1 << count) - 1;
  assert(((bitmask >> count) & toclear) == toclear);
  bitmask -= (toclear << count);
}

void random_test_memory(Memory m, int buckets, int steps, unsigned seed1, unsigned seed2)
{
  FieldID fid = 1;
  size_t field_size = 8;

  size_t bucket_size = m.capacity() / field_size / buckets;
  assert(bucket_size > 0);

  std::map<FieldID, size_t> field_sizes;
  field_sizes[fid] = field_size;

  PRNGSequence seq(seed1, seed2);

  int step = 0;
  unsigned exp_bitmask = 0;
  std::vector<InstanceTracker> insts;
  // we need a profiling request set that ignores failures
  ProfilingRequestSet prs;
  prs.add_request(Processor::NO_PROC, 0 /*ignore*/)
    .add_measurement<ProfilingMeasurements::InstanceStatus>();
  
  while(step < steps) {
    switch(seq.next_int(3)) {
    case 0: {
      // create something
      int size = pick_size(seq, buckets);
      Rect<1> rect(1, size * bucket_size);
      RegionInstance inst;
      int exp_base = set_bits(exp_bitmask, size, buckets);
      size_t idx = insts.size();
      log_app.debug() << "alloc #" << idx << ": size=" << size
		      << " exp_base=" << exp_base
		      << " bitmask=" << std::hex << exp_bitmask << std::dec;
      Event e = RegionInstance::create_instance(inst, m, rect,
						field_sizes, 0 /*SOA*/, prs);
      insts.push_back(InstanceTracker(size, inst, e, exp_base));
      if(exp_base >= 0) {
	// expected to succeed at some point
      } else {
	// expected to fail in short order
	bool poisoned = false;
	bool trigger = e.external_timedwait_faultaware(poisoned, 1000000);
	assert(trigger && poisoned);
	insts[idx].cstatus = InstanceTracker::CS_FAILED;
      }
      // either of these counts as a step
      step++;
    }
    default: break;
    }
  }
}
#endif

void worker_task(const void *args, size_t arglen,
		 const void *userdata, size_t userlen, Processor p)
{
  log_app.debug() << "worker task running on processor " << p;
}

void top_level_task(const void *args, size_t arglen,
		    const void *userdata, size_t userlen, Processor p)
{
  const TestConfig& config = *reinterpret_cast<const TestConfig *>(args);

  log_app.print() << "deferred_allocs test: directed=" << config.directed_tests
		  << " all=" << config.all_memories;

  PRNGSequence bseq(config.seed, 0);
#ifdef RANDOM_TESTS
  unsigned seed2 = 1;
#endif
  
  Machine::MemoryQuery mq(Machine::get_machine());
  for(Machine::MemoryQuery::iterator it = mq.begin(); it != mq.end(); ++it) {
    Memory m = *it;
    if(m.capacity() == 0) continue;

    // GPU_DYNAMIC_MEM does not support deferred allocation
    if(m.kind() == Memory::GPU_DYNAMIC_MEM) continue;

    // directed tests
    if(config.directed_tests) {
      directed_test_memory(config, m, p, "simple capacity limit",
			   "3 a1 s0 a1 s1 a1 s2 a1 f3");

      directed_test_memory(config, m, p, "simple reuse",
			   "1 a1 s0 i0 a1 s1 d1 c1 d1 t1 a1 s2");

      directed_test_memory(config, m, p, "capacity limit despite pending free",
			   "3 a1 s0 a1 s1 a1 s2 d1 a2 f3");

      directed_test_memory(config, m, p, "future capacity limit",
			   "2 a1 s0 a1 s1 d1 a1 a1 f3 t1 s2");

      directed_test_memory(config, m, p, "in order triggers",
			   "2 a1 s0 a1 s1 d0 a1 t0 s2");

      directed_test_memory(config, m, p, "alloc pass with capacity",
                           "2 a1 s0 d0 a1 s1 t0");

      directed_test_memory(config, m, p, "out of order deletes",
			   "2 a1 s0 a1 s1 d0 d1 c1 d1 t1 a1 s2");

      directed_test_memory(config, m, p, "out of order triggers",
			   "2 a1 s0 a1 s1 d0 d1 a2 t1 t0 s2");

      directed_test_memory(config, m, p, "using later free",
			   "2 a1 s0 a1 s1 d0 a1 d1 t1 s2");

      directed_test_memory(config, m, p, "reordered by later free",
			   "5 a1 s0 a1 s1 a2 s2 a1 s3" // 01223
			   "  d0 a1"                   // 01223   41223
			   "  d1 a1"                   // 01223   45223
			   "  d2 a2 d3"                // 01223   4566.
			   "  t2 s4 s5"                // 01453   6645.
			   "  a1"                      // 01453   66457
			   "  t0 t1 s6 t3 s7"          // 66457
			   );

      directed_test_memory(config, m, p, "avoid fragmentation failures",
			   "3 a1 s0 a1 s1 a1 s2 d0 a1 d1 d2 a2 t1 n3 t0 s3 t2 s4");

      directed_test_memory(config, m, p, "nasty partial rollback case",
			   "5 a2 s0 a1 s1 a1 s2 a1 s3" // 00123
			   " d0 a2 n4"                 // 00123   44123
			   " d3 a1 n5"                 // 00123   44125
			   " d1 d2 a2 n6"              // 00123   44665
			   " t2 n4 n5 n6"              // 001.3   44665
			   // 4 needs to succeed, but not 5 due to fragmentation
			   " t0 s4 n5 n6"              // 441.3   44665
			   " t1 s5 n6"                 // 445.3   44566
			   " t3 s6"                    // 44566
			   );

      directed_test_memory(config, m, p, "reordered instant destroy",
			   "2 a1 s0 a1 s1 d0 a1 i1 s2");

      directed_test_memory(config, m, p, "rebuild release allocator",
			   "3 a1 s0 a1 s1 a1 s2 d0 a1 d2 d1 a2 t1 n3 t0 s3 t2 s4");

      directed_test_memory(config, m, p, "recover from benign failure",
			   "3 a1 s0 a1 s1 a1 s2 d2 d0 d1 a2 c0 t1 t2 s3");

      directed_test_memory(config, m, p, "recover with collateral damage",
			   "4 a1 s0 a1 s1 a1 s2 a1 s3"  // 0123
			   "  d0 a1"                    // 0123  4123
			   "  d1 d2 a2"                 // 0123  4553
			   "  d3 a1"                    // 0123  4556
			   "  c1 u5"                    // 0123  416.
			   "  a1"                       // 0123  4167
			   "  t0 s4 t2 s6 t3 s7"        // 4167
			   );

      directed_test_memory(config, m, p, "mixing destroys",
			   "4 a1 s0 a1 s1 a1 s2 a1 s3"  // 0123
			   "  d2 a1"                    // 0123  0143
			   "  i0 s4"                    // 4123
			   );
      
#ifdef REALM_REORDER_DEFERRED_ALLOCATIONS
      directed_test_memory(config, m, p, "out of order success",
			   "3 a1 s0 a2 s1 d0 d1 a2 a1 t1 s3 t0 s2");
#endif

      // workaround for issue 892: don't attempt alloc chain test on remote
      //  memories for now
      if(m.address_space() == p.address_space()) {
	// test queuing up of chains of allocations that are not currently
	//  possible but will be once their preconditions trigger
	alloc_chain_test(m, p, 4, 10);
      }
    }

#ifdef RANDOM_TESTS
    for(int i = 0; i < config.trials_per_mem; i++) {
      int buckets = config.buckets_min;
      if(config.buckets_max > config.buckets_min)
	buckets += bseq.next_int(config.buckets_max - config.buckets_min + 1);

      // directed tests
      random_test_memory(m, buckets, config.trial_length, config.seed, seed2++);
    }
#endif

    if(!config.all_memories) break;
  }

  Runtime::get_runtime().shutdown(Event::NO_EVENT, 0 /*success*/);
}

int main(int argc, const char **argv)
{
  Runtime rt;

  rt.init(&argc, (char ***)&argv);

  TestConfig config;
  config.seed = 12345;
  config.directed_tests = true;
  config.trials_per_mem = 1;
  config.trial_length = 10;
  config.buckets_min = 4;
  config.buckets_max = 4;
  config.all_memories = false;
  config.check_alloc_result = true;

  CommandLineParser clp;
  clp.add_option_int("-seed", config.seed);
  clp.add_option_int("-d", config.directed_tests);
  clp.add_option_int("-t", config.trials_per_mem);
  clp.add_option_int("-l", config.trial_length);
  clp.add_option_int("-min", config.buckets_min);
  clp.add_option_int("-max", config.buckets_max);
  clp.add_option_bool("-all", config.all_memories);

  bool ok = clp.parse_command_line(argc, argv);
  assert(ok);

  // try to use a cpu proc, but if that doesn't exist, take whatever we can get
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC)
    .first();
  if(!p.exists())
    p = Machine::ProcessorQuery(Machine::get_machine()).first();
  assert(p.exists());

  Processor::register_task_by_kind(p.kind(), false /*!global*/,
                                  TOP_LEVEL_TASK,
                                  CodeDescriptor(top_level_task),
                                  ProfilingRequestSet()).external_wait();

  Processor::register_task_by_kind(p.kind(), false /*!global*/,
				   WORKER_TASK,
				   CodeDescriptor(worker_task),
				   ProfilingRequestSet()).external_wait();

  Processor::register_task_by_kind(p.kind(), false /*!global*/,
				   ALLOC_RESULT_TASK,
				   CodeDescriptor(alloc_result_task),
				   ProfilingRequestSet()).external_wait();

  // collective launch of a single top level task
  rt.collective_spawn(p, TOP_LEVEL_TASK, &config, sizeof(config));

  // now sleep this thread until that shutdown actually happens
  int ret = rt.wait_for_shutdown();
  
  return ret;
}
