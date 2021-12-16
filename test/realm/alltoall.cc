#include "realm.h"
#include "realm/cmdline.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <set>
#include <time.h>
#include "osdep.h"
#include <vector>

using namespace Realm;

enum {
  STARTUP_TASK_ID = Processor::TASK_ID_FIRST_AVAILABLE+0,
  HOOKUP_TASK_ID,
  BARRIER_TASK_ID,
};

namespace TestConfig {
  size_t size = 64 << 20; // default to 64 MB
  bool full_barrier = false;
};

Barrier hookup_barrier, execution_barrier;
UserEvent barriers_ready;
std::vector<Memory> local_memories;
std::vector<RegionInstance> local_instances;
std::vector<Event> local_events;

struct Exchange {
  RegionInstance inst;
  Rect<1,long long> bounds;
};

void startup_task(const void *args, size_t arglen, 
                  const void *userdata, size_t userlen, Processor p)
{
  // Nothing to do here
}

void hookup_task(const void *args, size_t arglen, 
                 const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == (local_memories.size() * sizeof(Exchange)));
  const Exchange *exchanges = (const Exchange*)args;
  std::vector<CopySrcDstField> srcs(1);
  std::vector<CopySrcDstField> dsts(1);
  for (unsigned m = 0; m < local_memories.size(); ++m) {
    const Exchange &exchange = exchanges[m];
    srcs.back().set_field(exchange.inst, 0, 1);
    dsts.back().set_field(local_instances[m], 0, 1);
    IndexSpace<1,long long> space = exchange.bounds;
    local_events[m] = space.copy(srcs, dsts, ProfilingRequestSet(), local_events[m]);
  }
  if (TestConfig::full_barrier) {
    std::set<Event> preconditions;
    for (unsigned idx = 0; idx < local_events.size(); ++idx) {
      preconditions.insert(local_events[idx]);
      local_events[idx] = execution_barrier;
    }
    execution_barrier.arrive(1/*count*/, Event::merge_events(preconditions));
    execution_barrier = execution_barrier.advance_barrier();
  }
}

void barrier_task(const void *args, size_t arglen, 
                  const void *userdata, size_t userlen, Processor p)
{
  assert(barriers_ready.exists());
  assert(arglen == (2 * sizeof(Barrier)));
  const Barrier *barriers = (const Barrier*)args;
  hookup_barrier = barriers[0];
  execution_barrier = barriers[1];
  barriers_ready.trigger();
}

void measure_all_to_all(Memory::Kind memkind, const char *description,
                        unsigned rank, size_t total_ranks,
                        const std::vector<Processor> &remote_procs)
{
  const size_t total_memories = Machine::MemoryQuery(Machine::get_machine())
    .only_kind(memkind)
    .has_capacity(1)
    .count();
  if (total_memories <= 1) {
    fprintf(stdout,"Skipping %s: found only %zd memories\n\n", description, total_memories);
    return;
  }

  // Find our local memories
  local_memories.clear();
  for (Machine::MemoryQuery::iterator it = 
        Machine::MemoryQuery(Machine::get_machine())
         .only_kind(memkind).local_address_space().has_capacity(1).begin(); it; ++it)
    local_memories.push_back(*it);

  // make a user event to delay the execution of this program
  const UserEvent start = UserEvent::create_user_event();
  local_events.resize(local_memories.size());
  for (unsigned idx = 0; idx < local_events.size(); ++idx)
    local_events[idx] = start;
  
  // Make our local instances
  const size_t inst_size = TestConfig::size * total_memories;
  const IndexSpace<1,long long> bounds = Rect<1,long long>(0, inst_size - 1);
  const std::vector<size_t> field_sizes(1, 1);
  local_instances.resize(local_memories.size());
  for (unsigned idx = 0; idx < local_memories.size(); ++idx) {
    RegionInstance::create_instance(local_instances[idx], local_memories[idx],
        bounds, field_sizes, 0/*nop*/, ProfilingRequestSet()).wait();
    assert(local_instances[idx].exists());
  }
  // Wait for everyone to be done making their instances
  hookup_barrier.arrive();
  hookup_barrier.wait();
  hookup_barrier = hookup_barrier.advance_barrier();

  // Run a generation for each remote rank, do this hierarchically so that 
  // we exchange between all pairs between each pair of ranks before going
  // on to the next rank
  // This code assumes the number of local memories is the same on each rank
  // First do all-to-all's between our local memories
  std::vector<CopySrcDstField> srcs(1);
  std::vector<CopySrcDstField> dsts(1);
  const size_t offset = rank * local_instances.size() * TestConfig::size; 
  if (local_memories.size() > 1) {
    for (unsigned l = 1; l < local_memories.size(); ++l) {
      for (unsigned m = 0; m < local_memories.size(); ++m) {
        unsigned m2 = (m + l) % local_memories.size();
        // Issue the copy
        srcs.back().set_field(local_instances[m], 0, 1);
        dsts.back().set_field(local_instances[m2], 0, 1);
        const size_t lower = offset + m * TestConfig::size;
        IndexSpace<1,long long> space = Rect<1,long long>(lower, lower + TestConfig::size - 1);
        local_events[m2] = space.copy(srcs, dsts, ProfilingRequestSet(), local_events[m2]);
      }
      if (TestConfig::full_barrier) {
        std::set<Event> preconditions;
        for (unsigned idx = 0; idx < local_events.size(); ++idx) {
          preconditions.insert(local_events[idx]);
          local_events[idx] = execution_barrier;
        }
        execution_barrier.arrive(1/*count*/, Event::merge_events(preconditions));
        execution_barrier = execution_barrier.advance_barrier();
      }
    }
    // Wait for all the ranks to be done
    hookup_barrier.arrive();
    hookup_barrier.wait();
    hookup_barrier = hookup_barrier.advance_barrier();
  }
  std::vector<Exchange> exchanges(local_memories.size());
  for (unsigned r = 1; r < total_ranks; r++) {
    unsigned next_rank = (rank + r) % total_ranks;
    for (unsigned l = 0; l < local_memories.size(); ++l) {
      for (unsigned m = 0; m < local_memories.size(); ++m) {
        unsigned m2 = (m + l) % local_memories.size();
        Exchange &exchange = exchanges[m2];
        exchange.inst = local_instances[m];
        const size_t lower = offset + m * TestConfig::size;
        exchange.bounds = Rect<1,long long>(lower, lower + TestConfig::size - 1);
      }
      // Send the message and trigger the event when we are done
      Event done = remote_procs[next_rank].spawn(HOOKUP_TASK_ID, 
          &exchanges.front(), exchanges.size() * sizeof(Exchange));
      hookup_barrier.arrive(1, done);
      hookup_barrier.wait();
      hookup_barrier = hookup_barrier.advance_barrier();
    }
  }
  // Hook all our finish conditions into the execution barrier
  if (!TestConfig::full_barrier) {
    std::set<Event> finished;
    for (unsigned idx = 0; idx < local_events.size(); ++idx)
      finished.insert(local_events[idx]);
    execution_barrier.arrive(1, Event::merge_events(finished));
    execution_barrier = execution_barrier.advance_barrier();
  }
  // Start the timer
  const long long t1 = Clock::current_time_in_nanoseconds();
  // Start our graph
  start.trigger();
  // Wait for everyone to arrive on the barrier
  execution_barrier.get_previous_phase().wait();
  // Stop the timer
  const long long t2 = Clock::current_time_in_nanoseconds(); 

  // Report bandwidth
  if (rank == 0) {
    size_t total_copies = total_memories * (total_memories - 1);
    size_t total_bytes = total_copies * TestConfig::size; 
    double total_gigabytes = double(total_bytes) / double(1 << 30);
    double total_bandwidth = total_gigabytes / (double(t2 - t1) / 1e9);
    double rank_bandwidth = total_bandwidth / total_ranks;
    double memory_bandwidth = total_bandwidth / total_memories;
    fprintf(stdout,"%s Total GB Moved: %.4g GB\n", description, total_gigabytes);
    fprintf(stdout,"%s Total Execution Time: %lld ns\n", description, (t2 - t1));
    fprintf(stdout,"%s Total Bandwidth: %.4g GB/s\n", description, total_bandwidth);
    fprintf(stdout,"%s Rank Bandwidth: %.4g GB/s\n", description, rank_bandwidth);
    fprintf(stdout,"%s Bandwidth: %.4g GB/s\n\n", description, memory_bandwidth);
  }
  // Clean up
  for (unsigned idx = 0; idx < local_instances.size(); ++idx)
    local_instances[idx].destroy();
  local_memories.clear();
  local_instances.clear();
  local_events.clear();

  // Wait for everyone to be done with clean up
  hookup_barrier.arrive();
  hookup_barrier.wait();
  hookup_barrier = hookup_barrier.advance_barrier();
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int_units("-size", TestConfig::size, 'M')
    .add_option_bool("-full", TestConfig::full_barrier);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  const size_t total_ranks = Machine::get_machine().get_address_space_count();
  const Processor local = Machine::ProcessorQuery(Machine::get_machine())
    .local_address_space()
    .only_kind(Processor::LOC_PROC)
    .first();
  const unsigned rank = local.address_space();
  std::vector<Processor> remote_procs(total_ranks, Processor::NO_PROC);
  for (Machine::ProcessorQuery::iterator it = 
        Machine::ProcessorQuery(Machine::get_machine())
          .only_kind(Processor::LOC_PROC).begin(); it; ++it) {
    unsigned proc_rank = it->address_space();
    if (!remote_procs[proc_rank].exists())
      remote_procs[proc_rank] = *it;
  }
  for (unsigned idx = 0; idx < total_ranks; ++idx)
    assert(remote_procs[idx].exists());

  Event reg0 = 
    Processor::register_task_by_kind(Processor::LOC_PROC, false/*global*/, 
        STARTUP_TASK_ID, CodeDescriptor(startup_task), ProfilingRequestSet(), 0, 0);
  Event reg1 = 
    Processor::register_task_by_kind(Processor::LOC_PROC, false/*global*/, 
        HOOKUP_TASK_ID, CodeDescriptor(hookup_task), ProfilingRequestSet(), 0, 0);
  Event reg2 =
    Processor::register_task_by_kind(Processor::LOC_PROC, false/*global*/, 
        BARRIER_TASK_ID, CodeDescriptor(barrier_task), ProfilingRequestSet(), 0, 0);

  if (rank > 0)
    barriers_ready = UserEvent::create_user_event();

  // make sure everyone is done registering their tasks before starting
  rt.collective_spawn_by_kind(Processor::LOC_PROC, STARTUP_TASK_ID, 0, 0, 
      true/*one per node*/, Event::merge_events(reg0, reg1, reg2)).wait();

  if (rank == 0) {
    hookup_barrier = Barrier::create_barrier(total_ranks);
    execution_barrier = Barrier::create_barrier(total_ranks);
    Barrier barriers[2] = { hookup_barrier, execution_barrier };
    for (unsigned idx = 1; idx < total_ranks; ++idx)
      remote_procs[idx].spawn(BARRIER_TASK_ID, barriers, 2 * sizeof(Barrier));
  } else {
    barriers_ready.wait();
  }

  measure_all_to_all(Memory::SYSTEM_MEM, "System Memory",
                     rank, total_ranks, remote_procs);
  measure_all_to_all(Memory::REGDMA_MEM, "Registered Memory",
                     rank, total_ranks, remote_procs);
  measure_all_to_all(Memory::GPU_FB_MEM, "Framebuffer Memory",
                     rank, total_ranks, remote_procs);

  if (rank == 0) {
    hookup_barrier.destroy_barrier();
    execution_barrier.destroy_barrier();
  }

  // request shutdown once the last task is complete
  rt.shutdown();

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();
  
  return 0;
}
