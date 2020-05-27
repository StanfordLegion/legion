#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <unistd.h>

#include <time.h>

#include "realm.h"

using namespace Realm;

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  DELAY_TASK     = Processor::TASK_ID_FIRST_AVAILABLE+1,
};

struct DelayTaskArgs {
  int id;
  int sleep_useconds;
};

int *task_counts = 0;
Processor *task_procs = 0;
double *task_start_times = 0;
double *task_end_times = 0;

void delay_task(const void *args, size_t arglen, 
		const void *userdata, size_t userlen, Processor p)
{
  const DelayTaskArgs& d_args = *(const DelayTaskArgs *)args;

  // increment the count
  __sync_fetch_and_add(&(task_counts[d_args.id]), 1);
  task_procs[d_args.id] = p;
  task_start_times[d_args.id] = Clock::current_time();

  //printf("starting task %d on processor " IDFMT "\n", d_args.id, p.id);
  usleep(d_args.sleep_useconds);
  //printf("ending task %d on processor " IDFMT "\n", d_args.id, p.id);

  task_end_times[d_args.id] = Clock::current_time();
}

// checks that all tasks in the first group started before all tasks in the second group
static int check_task_ordering(int start1, int end1, int start2, int end2)
{
  // get the max start time of group 1 and the min start time of group 2
  double max1 = -1e100;
  for(int i = start1; i <= end1; i++)
    if(max1 < task_start_times[i])
      max1 = task_start_times[i];

  double min2 = 1e100;
  for(int i = start2; i <= end2; i++)
    if(min2 > task_start_times[i])
      min2 = task_start_times[i];

  if(max1 > min2) {
    printf("ERROR: At least one task in [%d,%d] started after a task in [%d,%d]\n",
	   start1, end1, start2, end2);
    return 1;
  }

  return 0;
}    

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  int errors = 0;
  bool top_level_proc_reused = false;

  printf("top level task - getting machine and list of CPUs\n");

  Machine machine = Machine::get_machine();
  std::vector<Processor> all_cpus;
  Machine::ProcessorQuery pq(machine);
  pq.same_address_space_as(p).only_kind(Processor::LOC_PROC);
  for(Machine::ProcessorQuery::iterator it = pq.begin(); it != pq.end(); ++it) {
    // try not to use the processor that the top level task is running on
    if(*it != p)
      all_cpus.push_back(*it);
  }
  // if this is the ONLY processor, go ahead and add it back in
  if(all_cpus.empty()) {
    all_cpus.push_back(p);
    top_level_proc_reused = true;
  }
  int num_cpus = all_cpus.size();

  printf("creating processor group for all CPUs...\n");
  ProcessorGroup pgrp = ProcessorGroup::create_group(all_cpus);
  printf("group ID is " IDFMT "\n", pgrp.id);

  // see if the member list is what we expect
  std::vector<Processor> members;
  pgrp.get_group_members(members);
  if(members == all_cpus)
    printf("member list matches\n");
  else {
    errors++;
    printf("member list MISMATCHES\n");
    printf("expected:");
    for(std::vector<Processor>::const_iterator it = all_cpus.begin();
	it != all_cpus.end();
	it++)
      printf(" " IDFMT, (*it).id);
    printf("\n");
    printf("  actual:");
    for(std::vector<Processor>::const_iterator it = members.begin();
	it != members.end();
	it++)
      printf(" " IDFMT, (*it).id);
    printf("\n");
  }

  // we're going to launch 4 groups of tasks, each with one per CPU
  // 1) task sent to individual procs, priority = 0
  // 2) tasks sent to group, priority = 0
  // 3) tasks sent to group, priority = 1
  // 4) tasks sent to individual procs, priority = 0
  //
  // execution order in Realm should be 1, 3, 4, 2
  // NOTE: the shared LLR implements processor groups differently and gets 1, 3, 2, 4
  // NOTE2: if the processor our top-level task is running on is the only one, we'll
  //   get 3, 1, 4, 2
  int expected_order[4];
#ifdef SHARED_LOWLEVEL
  expected_order[0] = 1;
  expected_order[1] = 3;
  expected_order[2] = 2;
  expected_order[3] = 4;
#else
  expected_order[0] = top_level_proc_reused ? 3 : 1;
  expected_order[1] = top_level_proc_reused ? 1 : 3;
  expected_order[2] = 4;
  expected_order[3] = 2;
#endif
  
  int total_tasks = 4 * num_cpus;
  task_counts = new int[total_tasks];
  task_procs = new Processor[total_tasks];
  task_start_times = new double[total_tasks];
  task_end_times = new double[total_tasks];
  for(int i = 0; i < total_tasks; i++)
    task_counts[i] = 0;

  std::set<Event> task_events;
  int count = 0;
  for(int batch = 0; batch < 4; batch++) {
    for(int i = 0; i < num_cpus; i++) {
      bool to_group = (batch == 1) || (batch == 2);
      int priority = (batch == 2) ? 1 : 0;

      DelayTaskArgs d_args;
      d_args.id = count++;
      d_args.sleep_useconds = 250000;
      Event e = (to_group ? pgrp : all_cpus[i]).spawn(DELAY_TASK, &d_args, sizeof(d_args),
						      Event::NO_EVENT, priority);
      task_events.insert(e);
    }
    // small delay after each batch to make sure the tasks are all enqueued
    usleep(10000);
  }
  printf("%d tasks launched\n", count);

  // merge events
  Event merged = Event::merge_events(task_events);
  printf("merged event ID is " IDFMT " - waiting on it...\n",
	 merged.id);

  merged.wait();

  // now check the results
  for(int i = 0; i < total_tasks; i++) {
    if(task_counts[i] != 1) {
      printf("ERROR: task count for %d is %d, not 1\n", i, task_counts[i]);
      errors++;
    }
  }
  std::map<Processor, int> proc_counts;
  for(int i = 0; i < total_tasks; i++)
    proc_counts[task_procs[i]] += 1;
  for(std::map<Processor, int>::iterator it = proc_counts.begin();
      it != proc_counts.end();
      it++)
    if(it->second != 4) {
      printf("ERROR: processor " IDFMT " ran %d tasks, not 4\n", it->first.id, it->second);
      errors++;
    }

  for(int i = 0; i < 3; i++) {
    int start1 = (expected_order[i] - 1) * num_cpus;
    int end1 = start1 + (num_cpus - 1);
    int start2 = (expected_order[i+1] - 1) * num_cpus;
    int end2 = start2 + (num_cpus - 1);

    errors += check_task_ordering(start1, end1, start2, end2);
  }

  if(errors) {
    printf("Raw data:\n");
    for(int i = 0; i < total_tasks; i++) {
      printf("%2d: %d " IDFMT " %4.1f %4.1f\n",
	     i, task_counts[i], task_procs[i].id, task_start_times[i], task_end_times[i]);
    }

    printf("Exiting with errors.\n");
    exit(1);
  }

  printf("done!\n");
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(DELAY_TASK, delay_task);

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
  
  return 0;
}
