#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <chrono>

#include <time.h>

#include "realm.h"
#include "realm/profiling.h"

#include "osdep.h"

using namespace Realm;
using namespace Realm::ProfilingMeasurements;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  CHILD_TASK     = Processor::TASK_ID_FIRST_AVAILABLE+1,
  RESPONSE_TASK,
};

// we're going to use alarm() as a watchdog to detect hangs
void sigalrm_handler(int sig)
{
  fprintf(stderr, "HELP!  Alarm triggered - likely hang!\n");
  exit(1);
}

// some of the code in here needs the fault-tolerance stuff in Realm to show up
#define TRACK_MACHINE_UPDATES

#ifdef TRACK_MACHINE_UPDATES
class MyMachineUpdateTracker : public Machine::MachineUpdateSubscriber {
public:
  MyMachineUpdateTracker(void) {}
  virtual ~MyMachineUpdateTracker(void) {}

  virtual void processor_updated(Processor p, UpdateType update_type, 
				 const void *payload, size_t payload_size)
  {
    printf("machine processor update: " IDFMT " %s (%zd payload_bytes)\n",
	   p.id, (update_type == Machine::MachineUpdateSubscriber::THING_ADDED ? "added" :
		  update_type == Machine::MachineUpdateSubscriber::THING_REMOVED ? "removed" : "updated"),
	   payload_size);
  }

  virtual void memory_updated(Memory m, UpdateType update_type, 
			      const void *payload, size_t payload_size)
  {
    printf("machine memory update: " IDFMT " %s (%zd payload_bytes)\n",
	   m.id, (update_type == Machine::MachineUpdateSubscriber::THING_ADDED ? "added" :
		  update_type == Machine::MachineUpdateSubscriber::THING_REMOVED ? "removed" : "updated"),
	   payload_size);
  }
};

MyMachineUpdateTracker tracker;
#endif

struct ChildTaskArgs {
  bool inject_fault;
  bool hang;
  int sleep_useconds;
  Event wait_on;
};

void child_task(const void *args, size_t arglen, 
		const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "starting task on processor " << p;
  assert(arglen == sizeof(ChildTaskArgs));
  const ChildTaskArgs& cargs = *(const ChildTaskArgs *)args;
  if(cargs.wait_on.exists()) {
    cargs.wait_on.wait();
  }
  if(cargs.hang) {
    // create a user event and wait on it - hangs unless somebody cancels us
    UserEvent::create_user_event().wait();
  } else
    usleep(cargs.sleep_useconds);
#ifdef REALM_USE_EXCEPTIONS
  bool inject_fault = *(const bool *)args;
  if(inject_fault) {
    int buffer[4];
    buffer[0] = 11;
    buffer[1] = 22;
    buffer[2] = 33;
    buffer[3] = 44;
    // this causes a fatal error if Realm doesn't have exception support, so don't
    //  do it in that case
    Processor::report_execution_fault(44, buffer, 4*sizeof(int));
  }
#endif
  log_app.print() << "ending task on processor " << p;
}

Barrier response_counter;
int expected_responses_remaining = 0;

void response_task(const void *args, size_t arglen,
		   const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "profiling response task on processor " << p;
  printf("got profiling response - %zd bytes\n", arglen);
  printf("Bytes:");
  for(size_t i = 0; (i < arglen) && (i < 256); i++)
    printf(" %02x", ((unsigned char *)args)[i]);
  printf("\n");

  Realm::ProfilingResponse pr(args, arglen);

  OperationStatus::Result result = OperationStatus::COMPLETED_SUCCESSFULLY;
  if(pr.has_measurement<OperationStatus>()) {
    OperationStatus *op_status = pr.get_measurement<OperationStatus>();
    result = op_status->result;
    printf("op status = %d (code = %d, details = %zd bytes)\n",
	   (int)(op_status->result),
	   op_status->error_code,
	   op_status->error_details.size());
    delete op_status;
  } else
    printf("no status\n");

  if(pr.has_measurement<OperationTimeline>()) {
    OperationTimeline *op_timeline = pr.get_measurement<OperationTimeline>();
    printf("op timeline = %lld %lld %lld %lld (%lld %lld %lld)\n",
	   op_timeline->ready_time,
	   op_timeline->start_time,
	   op_timeline->end_time,
	   op_timeline->complete_time,
           (((op_timeline->start_time >= 0) && (op_timeline->ready_time >= 0)) ?
            (op_timeline->start_time - op_timeline->ready_time) : -1),
           (((op_timeline->end_time >= 0) && (op_timeline->start_time >= 0)) ?
            (op_timeline->end_time - op_timeline->start_time) : -1),
           (((op_timeline->complete_time >= 0) && (op_timeline->end_time >= 0)) ?
            (op_timeline->complete_time - op_timeline->end_time) : -1));
    // ready/start/end/complete should at least be ordered (if they exist)
    if(result != OperationStatus::CANCELLED) {
      assert(op_timeline->ready_time >= 0);
      assert(op_timeline->start_time >= op_timeline->ready_time);
    } else
      assert(op_timeline->start_time == OperationTimeline::INVALID_TIMESTAMP);
    if(result == OperationStatus::TERMINATED_EARLY)
      assert(op_timeline->end_time == OperationTimeline::INVALID_TIMESTAMP);
    else
      assert(op_timeline->end_time >= op_timeline->start_time);
    assert(op_timeline->complete_time >= op_timeline->end_time);
    delete op_timeline;
  } else
    printf("no timeline\n");

  if(pr.has_measurement<OperationTimelineGPU>()) {
    OperationTimelineGPU *op_timeline = pr.get_measurement<OperationTimelineGPU>();
    printf("op gpu timeline = %lld %lld (%lld)\n",
	   op_timeline->start_time,
	   op_timeline->end_time,
	   op_timeline->end_time - op_timeline->start_time);
    // start and end should at least be ordered
    if(result != OperationStatus::CANCELLED)
      assert(op_timeline->start_time >= 0);
    assert(op_timeline->end_time >= op_timeline->start_time);
    delete op_timeline;
  } else
    printf("no gpu timeline\n");

  if(pr.has_measurement<OperationEventWaits>()) {
    OperationEventWaits *op_waits = pr.get_measurement<OperationEventWaits>();
    printf("op waits = %zd", op_waits->intervals.size());
    if(!op_waits->intervals.empty()) {
      printf(" [");
      for(std::vector<OperationEventWaits::WaitInterval>::const_iterator it = op_waits->intervals.begin();
	  it != op_waits->intervals.end();
	  it++)
	printf(" (%lld %lld %lld %llx)",
	       it->wait_start, it->wait_ready, it->wait_end, it->wait_event.id);
      printf(" ]\n");
    } else
      printf("\n");
    delete op_waits;
  } else
    printf("no event wait data\n");

  if(0&&pr.has_measurement<OperationBacktrace>()) {
    OperationBacktrace *op_backtrace = pr.get_measurement<OperationBacktrace>();
    std::cout << "op backtrace = " << op_backtrace->backtrace;
    delete op_backtrace;
  }

  {
    InstanceStatus stat;
    if(pr.get_measurement(stat)) {
      std::cout << "inst status = " << stat.result << "\n";

      assert(pr.user_data_size() == sizeof(InstanceStatus::Result));
      InstanceStatus::Result exp_result = *(const InstanceStatus::Result *)(pr.user_data());
      if(exp_result != stat.result) {
	std::cout << "mismatch!  expected " << exp_result << "\n";
	exit(1);
      }
    }
  }

  {
    InstanceAllocResult result;
    if(pr.get_measurement(result)) {
      std::cout << "inst alloc success = " << result.success << "\n";
    }
  }

  {
    InstanceMemoryUsage usage;
    if(pr.get_measurement(usage))
      std::cout << "inst mem usage = " << usage.instance << " " << usage.memory << " " << usage.bytes << "\n";
  }

  if(pr.has_measurement<InstanceTimeline>()) {
    InstanceTimeline *inst_timeline = pr.get_measurement<InstanceTimeline>();
    printf("inst timeline = %lld %lld %lld (%lld %lld)\n",
	   inst_timeline->create_time,
	   inst_timeline->ready_time,
	   inst_timeline->delete_time,
	   inst_timeline->ready_time - inst_timeline->create_time,
	   inst_timeline->delete_time - inst_timeline->ready_time);
    delete inst_timeline;
  } else
    printf("no instance timeline\n");

  if(pr.user_data_size() > 0) {
    printf("user data = %zd (", pr.user_data_size());
    unsigned char *data = (unsigned char *)(pr.user_data());
    for(size_t i = 0; i < pr.user_data_size(); i++)
      printf(" %02x", data[i]);
    printf(" )\n");
  } else
    printf("no user data\n");

  if(__sync_sub_and_fetch(&expected_responses_remaining, 1) < 0) {
    printf("HELP!  Too many responses received!\n");
    exit(1);
  }

  // signal that we got a response
  response_counter.arrive();
}

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  printf("top level task - getting machine and list of CPUs\n");

  Machine machine = Machine::get_machine();
  std::vector<Processor> all_cpus;
  std::vector<Processor> all_gpus;
  {
    std::set<Processor> all_processors;
    machine.get_all_processors(all_processors);
    for(std::set<Processor>::const_iterator it = all_processors.begin();
	it != all_processors.end();
	it++) {
      if(it->kind() == Processor::LOC_PROC)
	all_cpus.push_back(*it);
      if(it->kind() == Processor::TOC_PROC)
	all_gpus.push_back(*it);
    }
  }
  bool has_gpus = !all_gpus.empty();

#ifdef TEST_FAULTS
  // touch all of the new resilience-based calls (just to test linking)
  if(arglen == 97) {  // if the compiler can DCE this, no link errors...  :(
    Processor::report_execution_fault(0, 0, 0);
    Processor::NO_PROC.report_processor_fault(0, 0, 0);
    Memory::NO_MEMORY.report_memory_fault(0, 0, 0);
    RegionInstance::NO_INST.report_instance_fault(0, 0, 0);
    RegionInstance::NO_INST.get_accessor().report_fault(ptr_t(0), 0, 0);
    RegionInstance::NO_INST.get_accessor().report_fault(DomainPoint(), 0, 0);
    RegionInstance::NO_INST.get_accessor().typeify<int>().report_fault(ptr_t(0));
    RegionInstance::NO_INST.get_accessor().typeify<int>().report_fault(DomainPoint());
    bool poisoned;
    Event::NO_EVENT.has_triggered_faultaware(poisoned);
    Event::NO_EVENT.wait_faultaware(poisoned);
    Event::NO_EVENT.external_wait_faultaware(poisoned);
    Event::NO_EVENT.cancel_operation(0, 0);
  }
#endif
  
  // launch a child task and perform some measurements on it
  // choose the last cpu/gpu, which is likely to be on a different node
  Processor profile_cpu = all_cpus.front();
  Processor task_proc = (has_gpus ? all_gpus.back() : all_cpus.back());
  ProfilingRequestSet prs;
  ProfilingRequest& pr = prs.add_request(profile_cpu, RESPONSE_TASK,
					 &task_proc, sizeof(task_proc));
  pr.add_measurement<OperationStatus>()
    .add_measurement<OperationTimeline>()
    .add_measurement<OperationEventWaits>()
    .add_measurement<OperationBacktrace>();
  if(has_gpus)
    pr.add_measurement<OperationTimelineGPU>();

  // we expect (exactly) 7 responses for tasks + 2 for instances
  // exception: gpu doesn't do the interrupt-during-wait task yet
  expected_responses_remaining = (has_gpus ? 6 : 7) + 2;
  response_counter = Barrier::create_barrier(expected_responses_remaining);

#ifndef _MSC_VER
  // give ourselves 60 seconds for the tasks, and their profiling responses, to
  //  finish - (this is excessive for an unloaded machine but can happen
  //  with heavy load)
  alarm(60);
#endif

  ChildTaskArgs cargs;
  cargs.inject_fault = false;
  cargs.sleep_useconds = 100000;
  cargs.hang = false;
  cargs.wait_on = Event::NO_EVENT;
  Event e1 = task_proc.spawn(CHILD_TASK, &cargs, sizeof(cargs), prs);

  cargs.inject_fault = true;
  Event e2 = task_proc.spawn(CHILD_TASK, &cargs, sizeof(cargs), prs, e1);
  cargs.inject_fault = false;
  Event e3 = task_proc.spawn(CHILD_TASK, &cargs, sizeof(cargs), prs, e2);

  {
    bool poisoned = false;
    e3.wait_faultaware(poisoned);
    printf("e3 done! (poisoned=%d)\n", poisoned);
  }

  // test event wait profiling
  {
    UserEvent u = UserEvent::create_user_event();
    cargs.wait_on = u;
    Event e4 = task_proc.spawn(CHILD_TASK, &cargs, sizeof(cargs), prs);
    cargs.wait_on = Event::NO_EVENT;
    cargs.sleep_useconds = 500000;
    Event e5 = task_proc.spawn(CHILD_TASK, &cargs, sizeof(cargs), prs);
    u.trigger(e5);
    e4.wait();
  }

  // test cancellation - first of a task that is "running"
  {
    cargs.sleep_useconds = 5000000;
    Event e4 = task_proc.spawn(CHILD_TASK, &cargs, sizeof(cargs), prs);
    sleep(2);
    int info = 111;
    e4.cancel_operation(&info, sizeof(info));
    bool poisoned = false;
    e4.wait_faultaware(poisoned);
    assert(poisoned);
  }

  // now cancellation of an event that is blocked on some event
  if(!has_gpus) {
    cargs.hang = true;
    Event e5 = task_proc.spawn(CHILD_TASK, &cargs, sizeof(cargs), prs);
    sleep(2);
    int info = 112;
    e5.cancel_operation(&info, sizeof(info));
    bool poisoned = false;
    e5.wait_faultaware(poisoned);
    assert(poisoned);
  }

  // instance profiling #1 - normal instance creation/deletion
  {
    Rect<1> is(0, 31);
    ProfilingRequestSet prs;
    Memory mem = Machine::MemoryQuery(machine).only_kind(Memory::SYSTEM_MEM).first();
    assert(mem.exists());
    InstanceStatus::Result exp_result = InstanceStatus::DESTROYED_SUCCESSFULLY;
    prs.add_request(profile_cpu, RESPONSE_TASK, &exp_result, sizeof(exp_result))
      .add_measurement<InstanceStatus>()
      .add_measurement<InstanceAllocResult>()
      .add_measurement<InstanceTimeline>()
      .add_measurement<InstanceMemoryUsage>();
    RegionInstance inst;
    Event e = RegionInstance::create_instance(inst, mem, is,
					      std::vector<size_t>(1, 8),
					      0, // SOA
					      prs);

    // while we've got an instance, let's try canceling some copies
    {
      // variant 1: canceling the copy before the preconditions are
      //  satisfied
      UserEvent u = UserEvent::create_user_event();
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].set_field(inst, 0, 1);
      dsts[0].set_field(inst, 0, 1);
      Event e2 = is.copy(srcs, dsts, prs, u);
      int info = 113;
      e2.cancel_operation(&info, sizeof(info));
      u.trigger(e);
      bool poisoned = false;
      e2.wait_faultaware(poisoned);
      assert(poisoned);
    }
    {
      // variant 2: propagate poison from a canceled precondition
      UserEvent u = UserEvent::create_user_event();
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].set_field(inst, 0, 1);
      dsts[0].set_field(inst, 0, 1);
      Event e2 = is.copy(srcs, dsts, prs, u);
      u.cancel();
      bool poisoned = false;
      e2.wait_faultaware(poisoned);
      assert(poisoned);
    }

    inst.destroy();
  }

  // instance profiling #2 - allocation failure
  {
    Rect<1> is(0, 1 << 21); // make sure total size is less than 4GB for 32 bit builds
    ProfilingRequestSet prs;
    Memory mem = Machine::MemoryQuery(machine).only_kind(Memory::SYSTEM_MEM).first();
    assert(mem.exists());
    InstanceStatus::Result exp_result = InstanceStatus::FAILED_ALLOCATION;
    prs.add_request(profile_cpu, RESPONSE_TASK, &exp_result, sizeof(exp_result))
      .add_measurement<InstanceStatus>()
      .add_measurement<InstanceAllocResult>()
      .add_measurement<InstanceTimeline>()
      .add_measurement<InstanceMemoryUsage>();
    RegionInstance inst;
    Event e = RegionInstance::create_instance(inst, mem, is,
					      std::vector<size_t>(1, 1024),
					      0, // SOA
					      prs);
    // a normal inst.destroy(e) would not work here, as 'e' is poisoned...
    // instead, we need to "launder" the poison in order to actually clean
    //  up the metadata for the failed allocation
    inst.destroy(Event::ignorefaults(e));
  }

  printf("waiting for profiling responses...\n");
  response_counter.wait();
  printf("all profiling responses received\n");
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  // get reference times using both C++'s steady_clock and realm timers - we'll
  //  check their correlation at the end
  std::chrono::time_point<std::chrono::steady_clock> t1_sys =
      std::chrono::steady_clock::now();
  long long t1_realm = Clock::current_time_in_nanoseconds();
  long long init_err = Clock::get_calibration_error();

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(CHILD_TASK, child_task);
  rt.register_task(RESPONSE_TASK, response_task);

#ifndef _MSC_VER
  signal(SIGALRM, sigalrm_handler);
#endif

#ifdef TRACK_MACHINE_UPDATES
  MyMachineUpdateTracker *tracker = new MyMachineUpdateTracker;
  Machine::get_machine().add_subscription(tracker);
#endif

  // select a processor to run the top level task on
  Processor p = Processor::NO_PROC;
  {
    std::set<Processor> all_procs;
    Machine::get_machine().get_all_processors(all_procs);
    for(std::set<Processor>::const_iterator it = all_procs.begin();
	it != all_procs.end();
	it++)
      if(it->kind() == Processor::LOC_PROC) {
	p = *it;
	break;
      }
  }
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // request shutdown once that task is complete
  rt.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();

  std::cout << "init calibration error = " << init_err << "\n";

  // get updated timer values and check for correlation
  std::chrono::time_point<std::chrono::steady_clock> t2_sys =
      std::chrono::steady_clock::now();
  long long t2_realm = Clock::current_time_in_nanoseconds();

  long long td_sys = std::chrono::nanoseconds(t2_sys - t1_sys).count();
  long long td_realm = t2_realm - t1_realm;

  // ask realm for its calibration error and correct accordingly
  long long cal_error = Clock::get_calibration_error();
  long long td_realm_corr = td_realm - (cal_error - init_err);
  std::cout << "sys=" << td_sys << " realm=" << td_realm << " corr=" << td_realm_corr
            << "\n";

#ifdef TRACK_MACHINE_UPDATES
  // the machine is gone at this point, so no need to remove ourselves explicitly
  //Machine::get_machine().remove_subscription(tracker);
  delete tracker;
#endif
  
  return 0;
}
