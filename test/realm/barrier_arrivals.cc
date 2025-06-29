/* Copyright 2023 NVIDIA Corporation
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

#include <assert.h>

#include "realm.h"
#include "osdep.h"
#include "realm/cmdline.h"
#include "realm/id.h"
#include "realm/network.h"

using namespace Realm;

Logger log_app("app");

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  ARRIVAL_TASK,
  WAITER_TASK,
};

struct TaskArgs {
  int num_iters;
  int num_tasks;
  int idx;
  int arrive_count;
  int arrive_value;
  int redop_result;
  Barrier barrier;
  bool do_advance;
};

namespace TestConfig {
  int slice_arrival = -1;
  int ignore_subscriber = 0;
  bool do_redop = false;
}; // namespace TestConfig

enum
{
  REDOP_ADD = 1
};

static const long long BARRIER_INITIAL_VALUE = 2;

class ReductionOpIntAdd {
public:
  typedef long long LHS;
  typedef long long RHS;

  template <bool EXCL>
  static void apply(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  // both of these are optional
  static const RHS identity;

  template <bool EXCL>
  static void fold(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }
};

const ReductionOpIntAdd::RHS ReductionOpIntAdd::identity = 0;

static void *bytedup(const void *data, size_t datalen)
{
  if(datalen == 0)
    return 0;
  void *dst = malloc(datalen);
  assert(dst != 0);
  memcpy(dst, data, datalen);
  return dst;
}

void arrival_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                  Processor p)
{
  TaskArgs task_args = *(const TaskArgs *)args;
  Barrier barrier = task_args.barrier;

  int n = 1;
  long long value = task_args.arrive_value;
  int count = task_args.arrive_count;

  if(TestConfig::slice_arrival == Network::my_node_id) {
    std::swap(count, n);
    value = count;
  }

  long long *reduce_value =
      reinterpret_cast<long long *>(bytedup(&value, sizeof(long long)));

  log_app.info() << "arrive_task N:" << Network::my_node_id << " b:" << barrier;

  for(int i = 0; i < n; i++) {
    if(TestConfig::do_redop) {
      barrier.arrive(count, Event::NO_EVENT, reduce_value, sizeof(long long));
      delete reduce_value;
    } else {
      barrier.arrive(count, Event::NO_EVENT);
    }
  }
}

void waiter_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                 Processor p)
{
  TaskArgs task_args = *(const TaskArgs *)args;
  Barrier barrier = task_args.barrier;

  barrier.wait();

  log_app.info() << "waiter_task unblocked N:" << Network::my_node_id << " b:" << barrier;

  if(TestConfig::do_redop) {
    long long result;
    bool ready = barrier.get_result(&result, sizeof(result));
    if(ready) {
      log_app.info() << "#### N:" << Network::my_node_id << " barrier=" << barrier
                     << " result=" << result << " exp:" << task_args.redop_result;
      assert(result == task_args.redop_result);
    }
  }
}

struct BarrierArrivalInfo {
  std::vector<Barrier::ParticipantInfo> participants;
  std::vector<NodeID> waiter_address_spaces;
};

void main_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
               Processor p)
{
  const int min_barrier_owners = 8;
  std::map<NodeID, Memory> memories;
  Machine::MemoryQuery mq(Machine::get_machine());
  for(Machine::MemoryQuery::iterator it = mq.begin(); it != mq.end(); ++it) {
    Memory memory = *it;
    if((memory).kind() == Memory::SYSTEM_MEM && !ID(memory).is_ib_memory()) {
      memories[ID(*it).memory_owner_node()] = memory;
    }
  }

  std::vector<Processor> reader_cpus, cpus;
  Machine machine = Machine::get_machine();
  for(const auto &memory : memories) {
    Machine::ProcessorQuery pq = Machine::ProcessorQuery(machine)
                                     .only_kind(Processor::LOC_PROC)
                                     .same_address_space_as(memory.second);
    for(Machine::ProcessorQuery::iterator it = pq.begin(); it; it++) {
      reader_cpus.push_back(*it);
      break;
    }
  }

  // Run test cases for various barrier arrival patterns

  typedef std::vector<BarrierArrivalInfo> ArrivalPatternMap;
  std::vector<ArrivalPatternMap> test_cases;

  if(Network::max_node_id + 1 < min_barrier_owners) {
    log_app.info() << "Not enough ranks/nodes avaiable:" << Network::max_node_id + 1
                   << " needed:" << min_barrier_owners;
    return;
  }

  std::vector<BarrierArrivalInfo> pattern2 = {
      BarrierArrivalInfo{{{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}, {6, 1}, {7, 1}},
                         /*waiter_address_space=*/{0, 1, 2, 3, 4, 8, 9}},

      BarrierArrivalInfo{{{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}, {6, 1}, {7, 1}},
                         /*waiter_address_space=*/{7, 8}}};

  std::vector<BarrierArrivalInfo> pattern3 = {
      BarrierArrivalInfo{{{0, 1}, {1, 1}, {3, 1}, {4, 1}, {7, 1}, {8, 1}},
                         /*waiter_address_space=*/{6, 9}},

      BarrierArrivalInfo{{{0, 1}, {1, 1}, {3, 1}, {4, 1}, {7, 1}, {8, 1}},
                         /*waiter_address_space=*/{6, 7, 8}},

      BarrierArrivalInfo{{{0, 1}, {1, 1}, {3, 1}, {4, 1}, {7, 1}, {8, 1}},
                         /*waiter_address_space=*/{0, 1, 2, 3, 4, 5, 6}},
  };

  // 5,6 should not be anywhere in patter5 but present in only
  // subscriptions of pattern3 and present in only reduction tree in
  // pattern2 then we can an edge case to address

  std::vector<BarrierArrivalInfo> pattern5 = {
      BarrierArrivalInfo{{{0, 1}, {1, 1}, {3, 1}, {4, 1}, {7, 1}, {8, 1}},
                         /*waiter_address_space=*/{0, 1}},

      BarrierArrivalInfo{{{0, 1}, {1, 1}, {3, 1}, {4, 1}, {7, 1}, {8, 1}},
                         /*waiter_address_space=*/{8, 9}},
  };

  /// std::vector<BarrierArrivalInfo> pattern4 = {
  // BarrierArrivalInfo{{{0, 1}, {1, 1}},
  //                  /*waiter_address_space=*/{0, 3}},
  // };

  test_cases.push_back(pattern2);
  // test_cases.push_back(pattern3);
  // test_cases.push_back(pattern5);

  Barrier barrier;

  if(TestConfig::do_redop) {
    barrier = Barrier::create_barrier(0, REDOP_ADD, &BARRIER_INITIAL_VALUE,
                                      sizeof(BARRIER_INITIAL_VALUE));
  } else {
    barrier = Barrier::create_barrier(0);
  }

  std::vector<Barrier> barrier_gens(1, barrier);

  int gen_offset = 0;
  for(size_t i = 0; i < test_cases.size(); i++) { // Test Case
    std::vector<Event> task_events;

    int expected_arrivals = 0;

    NodeID max_rank_id = 0;
    for(const auto &participant : test_cases[i][0].participants) {
      expected_arrivals += participant.count;
      max_rank_id = std::max(max_rank_id, static_cast<NodeID>(participant.address_space));
    }

    assert(Network::max_node_id >= max_rank_id);

    int gen_advances = 0;

    if(TestConfig::do_redop) {
      barrier =
          Barrier::create_barrier(expected_arrivals, REDOP_ADD, &BARRIER_INITIAL_VALUE,
                                  sizeof(BARRIER_INITIAL_VALUE));
    } else {
      barrier = Barrier::create_barrier(expected_arrivals);
    }
    gen_offset = 0;
    barrier_gens.clear();
    barrier_gens.push_back(barrier);
    gen_advances = test_cases[i].size() - 1;

    for(int ii = 0; ii < gen_advances; ii++) {
      barrier_gens.push_back(barrier_gens.back().advance_barrier());
    }
  }

  for(size_t i = 0; i < test_cases.size(); i++) { // Test Case
    std::vector<Event> task_events;

    int redop_result = BARRIER_INITIAL_VALUE;

    NodeID max_rank_id = 0;
    for(const auto &participant : test_cases[i][0].participants) {
      max_rank_id = std::max(max_rank_id, static_cast<NodeID>(participant.address_space));
      redop_result += static_cast<int>(participant.address_space);
    }

    for(size_t j = 0; j < test_cases[i].size(); j++) {
      assert(j + gen_offset < barrier_gens.size());
      Barrier handle = barrier_gens[j + gen_offset];

      for(int k = test_cases[i][j].participants.size() - 1; k >= 0; k--) {
        auto participant = test_cases[i][j].participants[k];
        TaskArgs task_args;
        task_args.barrier = handle;
        task_args.idx = participant.address_space;
        task_args.arrive_count = participant.count;
        task_args.arrive_value = static_cast<int>(participant.address_space);
        Event e = reader_cpus[(participant.address_space + 0) % reader_cpus.size()].spawn(
            ARRIVAL_TASK, &task_args, sizeof(TaskArgs));
        task_events.push_back(e);
      }

      for(int k = test_cases[i][j].waiter_address_spaces.size() - 1; k >= 0; k--) {
        AddressSpace address_space = test_cases[i][j].waiter_address_spaces[k];
        TaskArgs task_args;
        task_args.barrier = handle;
        task_args.idx = address_space;
        task_args.redop_result = redop_result;
        Event e = reader_cpus[(address_space + 0) % reader_cpus.size()].spawn(
            WAITER_TASK, &task_args, sizeof(TaskArgs));
        task_events.push_back(e);
      }

      // TODO(apryakhin@): Do we need to wait here or let arrivals to
      // happen for various reduction trees?
      // Event::merge_events(task_events).wait();
      // task_events.clear();
    }
    Event::merge_events(task_events).wait();
    gen_offset += test_cases[i].size();
  }
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, (char ***)&argv);

  rt.register_reduction<ReductionOpIntAdd>(REDOP_ADD);

  CommandLineParser cp;
  cp.add_option_int("-slice_arrival", TestConfig::slice_arrival);
  cp.add_option_int("-redop", TestConfig::do_redop);
  cp.add_option_int("-ignore", TestConfig::ignore_subscriber);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, MAIN_TASK,
                                   CodeDescriptor(main_task), ProfilingRequestSet())
      .external_wait();

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, ARRIVAL_TASK,
                                   CodeDescriptor(arrival_task), ProfilingRequestSet())
      .external_wait();

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, WAITER_TASK,
                                   CodeDescriptor(waiter_task), ProfilingRequestSet())
      .external_wait();

  Event e = rt.collective_spawn(p, MAIN_TASK, 0, 0);

  rt.shutdown(e);

  int ret = rt.wait_for_shutdown();

  return ret;
}
