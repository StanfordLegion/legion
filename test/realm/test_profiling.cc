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
#define NO_TRACK_MACHINE_UPDATES
#define NO_TEST_FAULTS

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

void child_task(const void *args, size_t arglen, Processor p)
{
  printf("starting task on processor " IDFMT "\n", p.id);
  sleep(1);
#ifdef TEST_FAULTS
  bool inject_fault = *(const bool *)args;
  if(inject_fault) {
    int buffer[4];
    buffer[0] = 11;
    buffer[1] = 22;
    buffer[2] = 33;
    buffer[3] = 44;
    Processor::report_execution_fault(44, buffer, 4*sizeof(int));
  }
#endif
  printf("ending task on processor " IDFMT "\n", p.id);
}

Barrier response_counter;
int expected_responses_remaining;

void response_task(const void *args, size_t arglen, Processor p)
{
  printf("got profiling response - %zd bytes\n", arglen);
  printf("Bytes:");
  for(size_t i = 0; i < arglen; i++)
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
    printf("op timeline = %llu %llu %llu (%lld %lld)\n",
	   op_timeline->ready_time,
	   op_timeline->start_time,
	   op_timeline->end_time,
	   op_timeline->start_time - op_timeline->ready_time,
	   op_timeline->end_time - op_timeline->start_time);
    delete op_timeline;
  } else
    printf("no timeline\n");

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

void top_level_task(const void *args, size_t arglen, Processor p)
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
    Processor::cancel_task(Event::NO_EVENT);
    Domain::cancel_copy(Event::NO_EVENT);
    UserEvent::create_user_event().cancel();
  }
#endif
  
  // launch a child task and perform some measurements on it
  // choose the last cpu, which is likely to be on a different node
  Processor first_cpu = all_cpus.back();
  ProfilingRequestSet prs;
  prs.add_request(first_cpu, RESPONSE_TASK, &first_cpu, sizeof(first_cpu))
    .add_measurement<OperationStatus>()
    .add_measurement<OperationTimeline>();

  // we expect (exactly) three responses
  response_counter = Barrier::create_barrier(3);
  expected_responses_remaining = 3;

  bool inject_fault = false;
  Event e1 = first_cpu.spawn(CHILD_TASK, &inject_fault, sizeof(inject_fault), prs);
  inject_fault = true;
  Event e2 = first_cpu.spawn(CHILD_TASK, &inject_fault, sizeof(inject_fault), prs, e1);
  inject_fault = false;
  Event e3 = first_cpu.spawn(CHILD_TASK, &inject_fault, sizeof(inject_fault), prs, e2);

  // give ourselves 5 seconds for the tasks, and their profiling responses, to finish
  alarm(5);

  bool poisoned = false;
#ifdef TEST_FAULTS
  e3.wait_faultaware(poisoned);
#else
  e3.wait();
#endif
  printf("done! (poisoned=%d)\n", poisoned);

  printf("waiting for profiling responses...\n");
  response_counter.wait();
  printf("all profiling responses received\n");

  Runtime::get_runtime().shutdown();
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(CHILD_TASK, child_task);
  rt.register_task(RESPONSE_TASK, response_task);

  signal(SIGALRM, sigalrm_handler);

  // Start the machine running
  // Control never returns from this call
  // Note we only run the top level task on one processor
  // You can also run the top level task on all processors or one processor per node
  rt.run(TOP_LEVEL_TASK, Runtime::ONE_TASK_ONLY);

  return 0;
}
