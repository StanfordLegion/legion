#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include "osdep.h"

#include <time.h>

#include "realm.h"

using namespace Realm;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  DELAY_TASK     = Processor::TASK_ID_FIRST_AVAILABLE+1,
};

struct DelayTaskArgs {
  int id;
  int sleep_useconds;
  RegionInstance inst;
};

enum {
  FID_TASK_COUNT, // int
  FID_TASK_PROC,  // Processor
  FID_TASK_START, // double
  FID_TASK_END,   // double
};

long long duration_microseconds(const struct timespec from, const struct timespec to) {
  return ((long long)(to.tv_sec) - (long long)(from.tv_sec)) * 1000000 +
    ((long long)(to.tv_nsec) - (long long)(from.tv_nsec)) / 1000;
}

struct timespec add_microseconds(const struct timespec time, long long microseconds) {
  long long new_nsec = time.tv_nsec + microseconds * 1000;
  long long new_sec = time.tv_sec + new_nsec / 1000000000;
  new_nsec = new_nsec % 1000000000;

  struct timespec result;
  result.tv_sec = new_sec;
  result.tv_nsec = new_nsec;
  return result;
}

void accurate_sleep(long long microseconds) {
  // Sleep can be inaccurate depending on system configuration and load,
  // at least track that here so that we know when the sleep is accurate.

  long long init_time = Realm::Clock::current_time_in_microseconds();
  long long current_time = init_time;

#if 1
  // Attempt to do a more accurate sleep by breaking the interval into
  // smaller pieces of size `granule` microseconds.

  long long final_target_time = init_time + microseconds;

  const long long granule = 100000; // 100 ms
  while (final_target_time - current_time > granule) {
    usleep(granule);
    current_time = Realm::Clock::current_time_in_microseconds();
  }
#else
  usleep(microseconds);
  current_time = Realm::Clock::current_time_in_microseconds();
#endif

  double relative = ((double)(current_time - init_time))/microseconds;
  if (relative > 1.2) {
    log_app.warning() << "sleep took too long - goal: " << microseconds <<
      " us, actual: " << current_time - init_time << " us, relative: " <<
      relative;
  }
}

void delay_task(const void *args, size_t arglen, 
		const void *userdata, size_t userlen, Processor p)
{
  const DelayTaskArgs& d_args = *(const DelayTaskArgs *)args;

  AffineAccessor<int, 1> task_counts(d_args.inst, FID_TASK_COUNT);
  AffineAccessor<Processor, 1> task_procs(d_args.inst, FID_TASK_PROC);
  AffineAccessor<double, 1> task_start_times(d_args.inst, FID_TASK_START);
  AffineAccessor<double, 1> task_end_times(d_args.inst, FID_TASK_END);

  // increment the count
  __sync_fetch_and_add(&(task_counts[d_args.id]), 1);
  task_procs[d_args.id] = p;
  task_start_times[d_args.id] = Clock::current_time();

  //printf("starting task %d on processor " IDFMT "\n", d_args.id, p.id);
  accurate_sleep(d_args.sleep_useconds);
  //printf("ending task %d on processor " IDFMT "\n", d_args.id, p.id);

  task_end_times[d_args.id] = Clock::current_time();
}

// checks that all tasks in the first group started before all tasks in the second group
static int check_task_ordering(int start1, int end1, int start2, int end2,
			       AffineAccessor<double, 1> task_start_times)
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
    log_app.error() << "ERROR: At least one task in ["
		    << start1 << "," << end1 << "] started after a task in ["
		    << start2 << "," << end2 << "]";
    return 1;
  }

  return 0;
}    

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  int errors = 0;
  bool top_level_proc_reused = false;

  log_app.info() << "top level task - getting machine and list of CPUs";

  Machine machine = Machine::get_machine();
  Processor lastp = Processor::NO_PROC;
  {
    Machine::ProcessorQuery pq(machine);
    pq.only_kind(Processor::LOC_PROC);
    // workaround for issue 892 - use the current processor (guaranteed local)
    //  rather than trying to get a remote one
    lastp = p;
    //for(Machine::ProcessorQuery::iterator it = pq.begin(); it != pq.end(); ++it)
    //  lastp = *it;
  }
  assert(lastp.exists());

  std::vector<Processor> all_cpus;
  Machine::ProcessorQuery pq(machine);
  pq.same_address_space_as(lastp).only_kind(Processor::LOC_PROC);
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

  log_app.info() << "creating processor group for all CPUs...";
  ProcessorGroup pgrp = ProcessorGroup::create_group(all_cpus);
  log_app.info() << "group ID is " << pgrp;

  // see if the member list is what we expect
  std::vector<Processor> members;
  pgrp.get_group_members(members);
  if(members == all_cpus)
    log_app.info() << "member list matches";
  else {
    errors++;
    log_app.error() << "member list MISMATCHES: "
		    << " expected=" << PrettyVector<Processor>(all_cpus)
		    << " actual=" << PrettyVector<Processor>(members);
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
  Rect<1> is_tasks(0, total_tasks - 1);

  Memory local_mem = Machine::MemoryQuery(machine).has_affinity_to(p).first();
  Memory tgt_mem = Machine::MemoryQuery(machine).has_affinity_to(lastp).first();
  std::map<FieldID, size_t> fields;
  fields[FID_TASK_COUNT] = sizeof(int);
  fields[FID_TASK_PROC] = sizeof(Processor);
  fields[FID_TASK_START] = sizeof(double);
  fields[FID_TASK_END] = sizeof(double);

  RegionInstance local_inst, tgt_inst;
  RegionInstance::create_instance(local_inst, local_mem, is_tasks,
				  fields, 0 /*SOA*/,
				  ProfilingRequestSet()).wait();
  RegionInstance::create_instance(tgt_inst, tgt_mem, is_tasks,
				  fields, 0 /*SOA*/,
				  ProfilingRequestSet()).wait();

  // clear the task counts to 0 in the remote instance
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    srcs[0].set_fill(int(0));
    dsts[0].set_field(tgt_inst, FID_TASK_COUNT, sizeof(int));
    is_tasks.copy(srcs, dsts, ProfilingRequestSet(), Event::NO_EVENT).wait();
  }

  std::set<Event> task_events, pgrp_events;
  int count = 0;
  for(int batch = 0; batch < 4; batch++) {
    for(int i = 0; i < num_cpus; i++) {
      bool to_group = (batch == 1) || (batch == 2);
      int priority = (batch == 2) ? 1 : 0;

      DelayTaskArgs d_args;
      d_args.id = count++;
      d_args.sleep_useconds = 1000000;
      d_args.inst = tgt_inst;
      Event e = (to_group ? pgrp : all_cpus[i]).spawn(DELAY_TASK, &d_args, sizeof(d_args),
						      Event::NO_EVENT, priority);
      task_events.insert(e);
      if(to_group)
	pgrp_events.insert(e);
    }
    // small delay after each batch to make sure the tasks are all enqueued
    accurate_sleep(400000);
  }
  log_app.info() << count << " tasks launched";

  // can destroy the processor group as soon as all of its tasks are done
  pgrp.destroy(Event::merge_events(pgrp_events));

  // merge events
  Event merged = Event::merge_events(task_events);
  log_app.info() << "merged event ID is " << merged << " - waiting on it...";

  // copy results back once all tasks are done
  {
    std::vector<CopySrcDstField> srcs(4), dsts(4);
    srcs[0].set_field(tgt_inst, FID_TASK_COUNT, sizeof(int));
    srcs[1].set_field(tgt_inst, FID_TASK_PROC, sizeof(Processor));
    srcs[2].set_field(tgt_inst, FID_TASK_START, sizeof(double));
    srcs[3].set_field(tgt_inst, FID_TASK_END, sizeof(double));
    dsts[0].set_field(local_inst, FID_TASK_COUNT, sizeof(int));
    dsts[1].set_field(local_inst, FID_TASK_PROC, sizeof(Processor));
    dsts[2].set_field(local_inst, FID_TASK_START, sizeof(double));
    dsts[3].set_field(local_inst, FID_TASK_END, sizeof(double));
    is_tasks.copy(srcs, dsts, ProfilingRequestSet(), merged).wait();
  }

  AffineAccessor<int, 1> task_counts(local_inst, FID_TASK_COUNT);
  AffineAccessor<Processor, 1> task_procs(local_inst, FID_TASK_PROC);
  AffineAccessor<double, 1> task_start_times(local_inst, FID_TASK_START);
  AffineAccessor<double, 1> task_end_times(local_inst, FID_TASK_END);

  // now check the results
  for(int i = 0; i < total_tasks; i++) {
    if(task_counts[i] != 1) {
      log_app.error() << "ERROR: task count for " << i << " is " << task_counts[i] << ", not 1";
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
      log_app.error() << "ERROR: processor " << it->first << " ran " << it->second << " tasks, not 4";
      errors++;
    }

  for(int i = 0; i < 3; i++) {
    int start1 = (expected_order[i] - 1) * num_cpus;
    int end1 = start1 + (num_cpus - 1);
    int start2 = (expected_order[i+1] - 1) * num_cpus;
    int end2 = start2 + (num_cpus - 1);

    errors += check_task_ordering(start1, end1, start2, end2, task_start_times);
  }

  if(errors) {
    log_app.error() << "Raw data:";
    for(int i = 0; i < total_tasks; i++) {
      log_app.error("%2d: %d " IDFMT " %4.1f %4.1f\n",
		    i, task_counts[i], task_procs[i].id, task_start_times[i], task_end_times[i]);
    }

    log_app.error() <<  "Exiting with errors.";
    exit(1);
  }

  // simple check for now to make sure IDs are reused - create a new group
  //  and verify it gets the same ID as the one we used above
  ProcessorGroup pgrp2 = ProcessorGroup::create_group(all_cpus);
  if(pgrp != pgrp2) {
    log_app.error() << "processor group ID not reused? " << pgrp2 << " != " << pgrp;
    exit(1);
  }
  pgrp2.destroy();

  log_app.info() << "done!";
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
