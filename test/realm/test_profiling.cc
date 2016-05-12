#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <csignal>

#include <time.h>
#include <unistd.h>

#include "realm/realm.h"
#include "realm/profiling.h"

using namespace Realm;
using namespace Realm::ProfilingMeasurements;
using namespace LegionRuntime::LowLevel;

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

  if(pr.has_measurement<OperationStatus>()) {
    OperationStatus *op_status = pr.get_measurement<OperationStatus>();
    printf("op status = %d (code = %d, details = %zd bytes)\n",
	   (int)(op_status->result),
	   op_status->error_code,
	   op_status->error_details.size());
    delete op_status;
  } else
    printf("no status\n");

  if(pr.has_measurement<OperationTimeline>()) {
    OperationTimeline *op_timeline = pr.get_measurement<OperationTimeline>();
    printf("op timeline = %llu %llu %llu %llu (%lld %lld %lld)\n",
	   op_timeline->ready_time,
	   op_timeline->start_time,
	   op_timeline->end_time,
	   op_timeline->complete_time,
	   op_timeline->start_time - op_timeline->ready_time,
	   op_timeline->end_time - op_timeline->start_time,
	   op_timeline->complete_time - op_timeline->end_time);
    delete op_timeline;
  } else
    printf("no timeline\n");

  if(pr.has_measurement<OperationEventWaits>()) {
    OperationEventWaits *op_waits = pr.get_measurement<OperationEventWaits>();
    printf("op waits = %zd", op_waits->intervals.size());
    if(!op_waits->intervals.empty()) {
      printf(" [");
      for(std::vector<OperationEventWaits::WaitInterval>::const_iterator it = op_waits->intervals.begin();
	  it != op_waits->intervals.end();
	  it++)
	printf(" (%lld %lld %lld %llx/%d)",
	       it->wait_start, it->wait_ready, it->wait_end, it->wait_event.id, it->wait_event.gen);
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

  if(pr.user_data_size() > 0) {
    printf("user data = %zd (", pr.user_data_size());
    unsigned char *data = (unsigned char *)(pr.user_data());
    for(size_t i = 0; i < pr.user_data_size(); i++)
      printf(" %02x", data[i]);
    printf(" )\n");
  } else
    printf("no user data\n");

  if(__sync_add_and_fetch(&expected_responses_remaining, -1) < 0) {
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
  {
    std::set<Processor> all_processors;
    machine.get_all_processors(all_processors);
    for(std::set<Processor>::const_iterator it = all_processors.begin();
	it != all_processors.end();
	it++)
      if(it->kind() == Processor::LOC_PROC)
	all_cpus.push_back(*it);
  }

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
  // choose the last cpu, which is likely to be on a different node
  Processor profile_cpu = all_cpus.front();
  Processor first_cpu = all_cpus.back();
  ProfilingRequestSet prs;
  prs.add_request(profile_cpu, RESPONSE_TASK, &first_cpu, sizeof(first_cpu))
    .add_measurement<OperationStatus>()
    .add_measurement<OperationTimeline>()
    .add_measurement<OperationEventWaits>()
    .add_measurement<OperationBacktrace>();

  // we expect (exactly) three responses
  expected_responses_remaining = 7;
  response_counter = Barrier::create_barrier(expected_responses_remaining);

  // give ourselves 15 seconds for the tasks, and their profiling responses, to finish
  alarm(15);

  ChildTaskArgs cargs;
  cargs.inject_fault = false;
  cargs.sleep_useconds = 100000;
  cargs.hang = false;
  cargs.wait_on = Event::NO_EVENT;
  Event e1 = first_cpu.spawn(CHILD_TASK, &cargs, sizeof(cargs), prs);

  cargs.inject_fault = true;
  Event e2 = first_cpu.spawn(CHILD_TASK, &cargs, sizeof(cargs), prs, e1);
  cargs.inject_fault = false;
  Event e3 = first_cpu.spawn(CHILD_TASK, &cargs, sizeof(cargs), prs, e2);

  {
    bool poisoned = false;
    e3.wait_faultaware(poisoned);
    printf("e3 done! (poisoned=%d)\n", poisoned);
  }

  // test event wait profiling
  {
    UserEvent u = UserEvent::create_user_event();
    cargs.wait_on = u;
    Event e4 = first_cpu.spawn(CHILD_TASK, &cargs, sizeof(cargs), prs);
    cargs.wait_on = Event::NO_EVENT;
    cargs.sleep_useconds = 500000;
    Event e5 = first_cpu.spawn(CHILD_TASK, &cargs, sizeof(cargs), prs);
    u.trigger(e5);
    e4.wait();
  }

  // test cancellation - first of a task that is "running"
  {
    cargs.sleep_useconds = 5000000;
    Event e4 = first_cpu.spawn(CHILD_TASK, &cargs, sizeof(cargs), prs);
    sleep(2);
    int info = 111;
    e4.cancel_operation(&info, sizeof(info));
    bool poisoned = false;
    e4.wait_faultaware(poisoned);
    assert(poisoned);
  }

  // now cancellatin of an event that is blocked on some event
  {
    cargs.hang = true;
    Event e5 = first_cpu.spawn(CHILD_TASK, &cargs, sizeof(cargs), prs);
    sleep(2);
    int info = 112;
    e5.cancel_operation(&info, sizeof(info));
    bool poisoned = false;
    e5.wait_faultaware(poisoned);
    assert(poisoned);
  }

  printf("waiting for profiling responses...\n");
  response_counter.wait();
  printf("all profiling responses received\n");
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(CHILD_TASK, child_task);
  rt.register_task(RESPONSE_TASK, response_task);

  signal(SIGALRM, sigalrm_handler);

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

#ifdef TRACK_MACHINE_UPDATES
  // the machine is gone at this point, so no need to remove ourselves explicitly
  //Machine::get_machine().remove_subscription(tracker);
  delete tracker;
#endif
  
  return 0;
}
