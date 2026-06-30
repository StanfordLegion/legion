/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
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

#include <cmath>
#include <iomanip>
#include <realm.h>
#include <realm/cmdline.h>

#include <random>
#include <unistd.h>
#include <limits>

using namespace Realm;

const size_t MAX_DIM = 2;
// typedef size_t ElementType;
typedef unsigned short ElementType;

typedef Realm::IndexSpace<MAX_DIM, int> CopyIndexSpace;

Logger log_app("app");

enum
{
  BENCH_TIMING_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  UPDATE_OP_TIMING_TASK,
  TOP_LEVEL_TASK,
  REMOTE_COPY_TASK,
};

namespace TestConfig {
  bool verify = false;
  bool enable_profiling = true;
  bool enable_remote_copy = false;
  bool graphviz = false;
  int graph_type = 2; // 0 means isolated dense, 1 means concurrent dense
  size_t num_iterations = 2;
  size_t num_samples = 2;
  size_t max_ops = 4;
  size_t max_copy_fields = 32000;
  size_t num_fields = 1;
  size_t field_size = sizeof(ElementType);
  size_t size = 4ULL * 256ULL; // 1024ULL * 4ULL;//1024ULL;
};                             // namespace TestConfig

template <int N, typename T, typename FT>
inline void copy(RegionInstance src_inst, RegionInstance dst_inst,
                 const std::vector<FieldID> &fields, IndexSpace<N, T> index_space)
{
  std::vector<CopySrcDstField> srcs(fields.size()), dsts(fields.size());
  for(size_t i = 0; i < fields.size(); i++) {
    srcs[i].set_field(src_inst, fields[i], sizeof(FT));
    dsts[i].set_field(dst_inst, fields[i], sizeof(FT));
  }
  index_space.copy(srcs, dsts, ProfilingRequestSet()).wait();
}

template <int N, typename T, typename DT>
static void dump_and_verify(RegionInstance inst, RegionInstance proxy_inst,
                            const std::vector<FieldID> &fields,
                            const IndexSpace<N, T> &is, size_t row_size,
                            const std::vector<DT> &values, bool verbose = false)
{
  copy<N, T, DT>(inst, proxy_inst, fields, is);
  size_t max_fail_count = 50;
  size_t fail_count = 0;
  int index = 0;
  for(FieldID fid : fields) {
    GenericAccessor<DT, N, T> acc(proxy_inst, fid);
    size_t i = 0;
    DT value = values[index++];
    for(IndexSpaceIterator<N, T> it(is); it.valid; it.step()) {
      for(PointInRectIterator<N, T> it2(it.rect); it2.valid; it2.step()) {
        DT v = acc[it2.p];
        if(value != DT(-1) && v != value) {
          std::cout << "bad v:" << int(v) << " p:" << it2.p << " fid:" << fid
                    << " expv:" << value << std::endl;
          fail_count++;
        } else {
          // std::cout << "good v:" << int(v) << " p:" << it2.p << " fid:" << fid
          //<< std::endl;
        }
        if(verbose) {
          if((i++) % row_size == 0)
            std::cout << std::endl;
          std::cout << it2.p << ": " << v << " ";
        }
        if(fail_count >= max_fail_count)
          break;
      }
      if(verbose)
        std::cout << "\n";
    }
  }
  assert(fail_count == 0);
}

class Stat {
public:
  Stat()
    : count(0)
    , mean(0.0)
    , sum(0.0)
    , square_sum(0.0)
    , smallest(std::numeric_limits<double>::max())
    , largest(-std::numeric_limits<double>::max())
  {}
  void reset() { *this = Stat(); }
  void sample(double s)
  {
    count++;
    if(s < smallest)
      smallest = s;
    if(s > largest)
      largest = s;
    sum += s;
    double delta0 = s - mean;
    mean += delta0 / count;
    double delta1 = s - mean;
    square_sum += delta0 * delta1;
  }
  unsigned get_count() const { return count; }
  double get_average() const { return mean; }
  double get_sum() const { return sum; }
  double get_stddev() const
  {
    return get_variance() > 0.0 ? std::sqrt(get_variance()) : 0.0;
  }
  double get_variance() const { return square_sum / (count > 2 ? 1 : count - 1); }
  double get_smallest() const { return smallest; }
  double get_largest() const { return largest; }

  friend std::ostream &operator<<(std::ostream &os, const Stat &s);

private:
  unsigned count;
  double mean;
  double sum;
  double square_sum;
  double smallest;
  double largest;
};

std::ostream &operator<<(std::ostream &os, const Stat &s)
{
  return os << std::scientific << std::setprecision(2)
            << s.get_average() /*<< "(+/-" << s.get_stddev() << ')'*/
            << ", MIN=" << s.get_smallest() << ", MAX=" << s.get_largest()
            << ", N=" << s.get_count();
}

struct CopyOperation;
class TestGraphFactory;

struct RemoteCopyTaskArgs {
  CopyOperation *op;
  Realm::UserEvent profiling_event;
  Realm::UserEvent remote_copy_event;
  Realm::Event wait_on;
  CopyIndexSpace index_space;
  Realm::Processor profile_proc;
  Realm::CopySrcDstField dsts;
  Realm::CopySrcDstField srcs;
};

struct UpdateOpTimingTaskArgs {
  CopyOperation *op;
  Realm::UserEvent profiling_event;
};

static const char *mem_kind_to_string(Realm::Memory::Kind kind)
{
  switch(kind) {
  case Realm::Memory::Kind::DISK_MEM:
    return "DISK";
  case Realm::Memory::Kind::FILE_MEM:
    return "FILE";
  case Realm::Memory::Kind::GLOBAL_MEM:
    return "GLOBAL";
  case Realm::Memory::Kind::GPU_DYNAMIC_MEM:
    return "GPU_DYN";
  case Realm::Memory::Kind::GPU_FB_MEM:
    return "GPU_FB";
  case Realm::Memory::Kind::GPU_MANAGED_MEM:
    return "GPU_MANAGED";
  case Realm::Memory::Kind::HDF_MEM:
    return "HDF_MEM";
  case Realm::Memory::Kind::LEVEL1_CACHE:
    return "L1$";
  case Realm::Memory::Kind::LEVEL2_CACHE:
    return "L2$";
  case Realm::Memory::Kind::LEVEL3_CACHE:
    return "L3$";
  case Realm::Memory::Kind::REGDMA_MEM:
    return "REGDMA";
  case Realm::Memory::Kind::SOCKET_MEM:
    return "SOCKET";
  case Realm::Memory::Kind::SYSTEM_MEM:
    return "SYSTEM";
  case Realm::Memory::Kind::Z_COPY_MEM:
    return "Z_COPY";
  default:
    return "Unknown";
  }
}

static void display_memory_info(Memory m)
{
  std::vector<Machine::MemoryMemoryAffinity> affinities;
  affinities.clear();
  Realm::Machine::get_machine().get_mem_mem_affinity(affinities, m, Memory::NO_MEMORY,
                                                     false);
  log_app.print() << "Memory: " << m << " kind: " << mem_kind_to_string(m.kind())
                  << " size: " << static_cast<double>(m.capacity()) / (1024.0 * 1024.0)
                  << "MiB";
  for(Machine::MemoryMemoryAffinity &m2m : affinities) {
    log_app.print() << "\t" << m2m.m2 << " est-bw: " << std::fixed << std::setprecision(2)
                    << (m2m.bandwidth / 1000.0) << "GB/s est-lat: " << m2m.latency
                    << "ns";
  }
}

struct CopyOperation {
  CopyIndexSpace index_space;
  std::vector<Realm::RegionInstance> owned_instances;
  std::vector<Realm::CopySrcDstField> dsts;
  std::vector<Realm::CopySrcDstField> srcs;
  std::vector<CopyOperation *> dependencies;
  Realm::Memory src_mem;
  bool is_dependency = false;
  Realm::Event current_event = Realm::Event::NO_EVENT;
  Stat measured_time; // nanoseconds
  Stat lag_time;      // nanoseconds
  CopyOperation(CopyIndexSpace i, std::vector<Realm::CopySrcDstField> &d,
                std::vector<Realm::CopySrcDstField> &s, Realm::Memory src)
    : index_space(i)
    , dsts(d)
    , srcs(s)
    , src_mem(src)
  {}
  ~CopyOperation()
  {
    for(Realm::RegionInstance &inst : owned_instances) {
      inst.destroy();
    }
  }
  size_t get_total_size() const
  {
    size_t total = 0;
    for(size_t i = 0; i < dsts.size(); i++) {
      total += dsts[i].size;
    }
    return index_space.volume() * total;
  }
  void add_dependency(CopyOperation *op)
  {
    dependencies.push_back(op);
    op->is_dependency = true;
  }

  template <typename FwdIter>
  static size_t get_total_size(FwdIter begin, FwdIter end)
  {
    size_t total = 0;
    for(; begin != end; ++begin) {
      CopyOperation &op = *begin;
      total += op.get_total_size();
    }
    return total;
  }
};

class TestGraphFactory {
public:
  virtual ~TestGraphFactory() {}
  virtual void create(std::vector<CopyOperation> &graph) = 0;
  virtual void verify() = 0;
};

template <typename T>
class RandomPicker {
public:
  // Constructor takes a vector of memories to pick from
  RandomPicker(const std::vector<T> &_elements)
    : elements(_elements)
    , gen(std::random_device{}())
    , dist(0, elements.size() - 1)
  {}

  std::pair<size_t, T> operator()(size_t i)
  {
    size_t random_index = dist(gen);
    return {random_index, elements[random_index]};
  }

protected:
  std::vector<T> elements;
  std::mt19937 gen;
  std::uniform_int_distribution<> dist;
};

class MultiFieldTestGraphFactory : public TestGraphFactory {
public:
  std::vector<Realm::Memory> memories_to_test;
  size_t size;
  std::map<FieldID, size_t> fields;
  std::vector<std::map<FieldID, size_t>> inst_to_fields;
  std::vector<std::tuple<CopyIndexSpace, RegionInstance, std::vector<ElementType>,
                         std::vector<FieldID>>>
      validate_queue;

  MultiFieldTestGraphFactory(std::vector<Realm::Memory> &_mems, size_t _sz,
                             std::map<FieldID, size_t> _fields)
    : memories_to_test(_mems)
    , size(_sz)
    , fields(_fields)
  {}

  void verify() override
  {
    Realm::Machine::MemoryQuery mq(Realm::Machine::get_machine());
    mq.only_kind(Memory::SYSTEM_MEM).has_capacity(1);
    std::vector<Realm::Memory> memories(mq.begin(), mq.end());

    for(const auto &[index_space, inst, value, fields] : validate_queue) {
      RegionInstance validate_instance;
      std::map<FieldID, size_t> src_fields;
      assert(value.empty() == false);
      for(FieldID fid : fields)
        src_fields[fid] = sizeof(value.front());
      Realm::RegionInstance::create_instance(validate_instance, *mq.begin(), index_space,
                                             src_fields, 0, ProfilingRequestSet())
          .wait();
      dump_and_verify<MAX_DIM, int, ElementType>(inst, validate_instance, fields,
                                                 index_space, 1, value);
    }
  }

  /*virtual*/
  void create(std::vector<CopyOperation> &graph) override
  {
    graph.clear();

    constexpr ElementType value = 9;

    Realm::Point<MAX_DIM> start_pnt(0, 0);
    Realm::Point<MAX_DIM> end_pnt(1, TestConfig::size);

    // Realm::Point<MAX_DIM> start_pnt(0);
    // Realm::Point<MAX_DIM> end_pnt(TestConfig::size);
    CopyIndexSpace is(Rect<MAX_DIM>{start_pnt, end_pnt});

    std::map<FieldID, size_t> src_fields;
    for(const auto [field_id, field_size] : fields) {
      src_fields[field_id] = field_size;
    }

    std::map<FieldID, size_t> dst_fields;
    for(const auto [field_id, field_size] : fields) {
      dst_fields[field_id] = field_size;
    }

    assert(src_fields.empty() == false);
    assert(dst_fields.empty() == false);

    std::map<FieldID, ElementType> values;

    std::vector<Realm::RegionInstance> instances;
    std::vector<Realm::RegionInstance> src_instances;
    std::vector<Realm::RegionInstance> dst_instances;

    for(size_t i = 0; i < memories_to_test.size(); i++) {
      for(size_t j = 0; j < memories_to_test.size(); j++) {
        Realm::RegionInstance src_inst, dst_inst;
        Realm::RegionInstance::create_instance(src_inst, memories_to_test[i], is,
                                               src_fields, 0, ProfilingRequestSet())
            .wait();
        Realm::RegionInstance::create_instance(dst_inst, memories_to_test[j], is,
                                               dst_fields, 0, ProfilingRequestSet())
            .wait();

        if(i == j) {
          int index = 1;
          std::vector<Event> fill_events;
          for(const auto &[field_id, field_size] : src_fields) {
            std::vector<Realm::CopySrcDstField> srcs(1), dsts(1);
            ElementType fill_value = value + index++;
            values[field_id] = fill_value;
            srcs[0].set_fill<ElementType>(fill_value); // this needs conversion
            assert(field_size == sizeof(ElementType));
            dsts[0].set_field(src_inst, field_id, field_size);
            fill_events.emplace_back(is.copy(srcs, dsts, ProfilingRequestSet()));
          }

          Event::merge_events(fill_events).wait();
        }

        instances.push_back(src_inst);
        src_instances.push_back(src_inst);
        inst_to_fields.push_back(src_fields);
        instances.push_back(dst_inst);
        dst_instances.push_back(dst_inst);
        inst_to_fields.push_back(dst_fields);
      }
    }

    const size_t max_concurrent_ops = TestConfig::max_ops;

    RandomPicker inst_picker(instances);

    for(size_t i = 0; i < max_concurrent_ops; i++) {
      auto src_inst = src_instances[0];
      auto dst_inst =
          dst_instances[std::max(size_t(0), dst_instances.size() - 1)]; // inst_picker(i);

      // auto src_fields = src_fields;//inst_to_fields[0];
      // auto dst_fields = dst_fields;//inst_to_fields[dst_inst.first];

      const size_t max_fields =
          std::min<size_t>(TestConfig::max_copy_fields, fields.size());
      std::vector<Realm::CopySrcDstField> srcs(max_fields), dsts(max_fields);

      std::vector<FieldID> src_field_ids;
      for(const auto &[field_id, field_size] : src_fields) {
        src_field_ids.push_back(field_id);
      }

      std::vector<FieldID> dst_field_ids;
      for(const auto &[field_id, field_size] : dst_fields) {
        dst_field_ids.push_back(field_id);
      }

      // Shuffle to select random fields for source and destination
      std::shuffle(src_field_ids.begin(), src_field_ids.end(),
                   std::mt19937(std::random_device()()));
      std::shuffle(dst_field_ids.begin(), dst_field_ids.end(),
                   std::mt19937(std::random_device()()));

      std::vector<ElementType> verify_values;

      // Pick `max_fields` random fields for the source
      size_t field_index = 0;
      for(size_t i = 0; i < max_fields && field_index < src_field_ids.size(); i++) {
        srcs[i].set_field(src_inst, src_field_ids[field_index],
                          src_fields[src_field_ids[field_index]]);
        verify_values.push_back(values[src_field_ids[field_index]]);
        field_index++;
      }

      // Pick `max_fields` random fields for the destination
      field_index = 0;
      for(size_t i = 0; i < max_fields && field_index < dst_field_ids.size(); i++) {
        dsts[i].set_field(dst_inst, dst_field_ids[field_index],
                          dst_fields[dst_field_ids[field_index]]);
        field_index++;
      }

      auto sub =
          std::vector<FieldID>(dst_field_ids.begin(), dst_field_ids.begin() + max_fields);
      validate_queue.emplace_back(std::make_tuple(is, dst_inst, verify_values, sub));

      graph.emplace_back(is, dsts, srcs, src_inst.get_location());
    }

    graph[0].owned_instances = instances;
  }
};

static void display_node_data(std::vector<CopyOperation> &graph)
{
  if(TestConfig::graphviz) {
    std::cout << "digraph g {" << std::endl;
  }
  // Node information
  for(size_t i = 0; i < graph.size(); i++) {
    // Assume instances are the same across all src and all dst operations
    Memory src = graph[i].srcs[0].inst.get_location();
    Memory dst = graph[i].dsts[0].inst.get_location();
    if(!src.exists()) {
      src = dst;
    }
    std::vector<Realm::Machine::MemoryMemoryAffinity> affinity;
    if(Realm::Machine::get_machine().get_mem_mem_affinity(affinity, src, dst, false) ==
       0) {
      Realm::Machine::MemoryMemoryAffinity fake_aff;
      fake_aff.m1 = src;
      fake_aff.m2 = dst;
      fake_aff.bandwidth = 1;
      fake_aff.latency = UINT_MAX;
      affinity.push_back(fake_aff);
    }
    const double bw =
        (graph[i].get_total_size() * 1000ULL) / graph[i].measured_time.get_average();
    const double lag = graph[i].lag_time.get_average();
    if(TestConfig::graphviz) {
      std::cout << "node_" << i << "[label=<" << i << "<BR /><FONT POINT-SIZE=\"10\">";
    }
    std::cout << src << '(' << mem_kind_to_string(src.kind()) << ") : " << dst << '('
              << mem_kind_to_string(dst.kind()) << ")";
    if(TestConfig::graphviz) {
      std::cout << "<BR />";
    }
    std::cout << " sz: " << graph[i].get_total_size() / (1024ULL * 1024ULL) << "MiB"
              << " bw: " << std::fixed << std::setprecision(2) << bw / 1000.0 << "GB/s ("
              << std::fixed << 100.0 * bw / affinity[0].bandwidth << "%)"
              << " lag: " << lag << "ns";
    if(TestConfig::graphviz) {
      std::cout << "</FONT>>];";
    }
    std::cout << std::endl;
  }
  // Links
  if(TestConfig::graphviz) {
    for(size_t i = 0; i < graph.size(); i++) {
      for(size_t j = 0; j < graph[i].dependencies.size(); j++) {
        const size_t dependency_idx = graph[i].dependencies[j] - graph.data();
        std::cout << "node_" << dependency_idx << " -> node_" << i << ';' << std::endl;
      }
    }
    std::cout << '}' << std::endl;
  }
}

static void remote_copy_task(const void *args, size_t arglen, const void *userdata,
                             size_t userlen, Processor p)
{
  const RemoteCopyTaskArgs &self_args =
      *reinterpret_cast<const RemoteCopyTaskArgs *>(args);
  assert(arglen == sizeof(RemoteCopyTaskArgs));

  Realm::ProfilingRequestSet prs;
  if(TestConfig::enable_profiling) {
    UpdateOpTimingTaskArgs prof_args;
    prof_args.op = self_args.op;
    prof_args.profiling_event = self_args.profiling_event;
    prs.add_request(self_args.profile_proc, UPDATE_OP_TIMING_TASK, &prof_args,
                    sizeof(prof_args))
        .add_measurement(ProfilingMeasurements::OperationTimeline::ID);
  }
  std::vector<Realm::CopySrcDstField> srcs(1, self_args.srcs), dsts(1, self_args.dsts);
  Realm::Event event = self_args.index_space.copy(srcs, dsts, prs, self_args.wait_on);
  self_args.remote_copy_event.trigger(event);

  Realm::RegionInstance src_inst = self_args.srcs.inst;
  Realm::RegionInstance dst_inst = self_args.dsts.inst;
  log_app.debug("Remote Copy(%p) from src:%llx(%llx) to dst:%llx(%llx) is issued on "
                "processor %llx",
                self_args.op, src_inst.id, src_inst.get_location().id, dst_inst.id,
                dst_inst.get_location().id, p.id);
}

static void issue_copy_from_remote(std::vector<Realm::Event> &finish_events,
                                   CopyOperation &op, Realm::Event wait_on,
                                   Processor local_p, Processor remote_p)
{
  UserEvent profiling_event = UserEvent::NO_USER_EVENT;
  UserEvent remote_copy_event = UserEvent::create_user_event();
  if(TestConfig::enable_profiling) {
    profiling_event = UserEvent::create_user_event();
    finish_events.push_back(profiling_event);
  }

  RemoteCopyTaskArgs remote_args;
  remote_args.profiling_event = profiling_event;
  remote_args.remote_copy_event = remote_copy_event;
  remote_args.wait_on = wait_on;
  remote_args.op =
      &op; // profiling task is executed on local proc, so it is OK to pass a pointer
  remote_args.profile_proc = local_p;
  remote_args.index_space = op.index_space;
  remote_args.srcs = op.srcs[0];
  remote_args.dsts = op.dsts[0];

  Realm::Event remote_task_event =
      remote_p.spawn(REMOTE_COPY_TASK, &remote_args, sizeof(RemoteCopyTaskArgs));
  finish_events.push_back(remote_task_event);

  // TODO: Add some validation of the copy here
  // This is a dangling node (a node without children), so make sure to
  // capture it's finish event as one that marks the graph as complete
  if(!op.is_dependency) {
    finish_events.push_back(remote_copy_event);
  }
}

static void issue_copy_from_local(std::vector<Realm::Event> &finish_events,
                                  CopyOperation &op, Realm::Event wait_on, Processor p)
{
  // Queue up the copy!
  // TODO: Use the profiling request set to accumulate the times of the
  // individual copies for bandwidth verification
  Realm::ProfilingRequestSet prs;
  if(TestConfig::enable_profiling) {
    UpdateOpTimingTaskArgs prof_args;
    UserEvent profiling_event = UserEvent::create_user_event();
    finish_events.push_back(profiling_event);
    prof_args.op = &op;
    prof_args.profiling_event = profiling_event;
    prs.add_request(p, UPDATE_OP_TIMING_TASK, &prof_args, sizeof(prof_args))
        .add_measurement(ProfilingMeasurements::OperationTimeline::ID);
  }
  op.current_event = op.index_space.copy(op.srcs, op.dsts, prs, wait_on);

  Realm::RegionInstance src_inst = op.srcs[0].inst;
  Realm::RegionInstance dst_inst = op.dsts[0].inst;
  log_app.debug("Local Copy(%p) from src:%llx(%llx) to dst:%llx(%llx) is issued on "
                "processor %llx",
                &op, src_inst.id, src_inst.get_location().id, dst_inst.id,
                dst_inst.get_location().id, p.id);

  // TODO: Add some validation of the copy here
  // This is a dangling node (a node without children), so make sure to
  // capture it's finish event as one that marks the graph as complete
  if(!op.is_dependency) {
    finish_events.push_back(op.current_event);
  }
}

static Realm::Event run_graph(std::vector<CopyOperation> &graph, Realm::Event start_event,
                              Realm::Processor p)
{
  // build a map of processor
  std::map<realm_address_space_t, Realm::Processor> proc_map;
  if(TestConfig::enable_remote_copy) {
    for(realm_address_space_t i = 0;
        i < Realm::Machine::get_machine().get_address_space_count(); i++) {
      proc_map[i] = Realm::Processor::NO_PROC;
    }
    for(Machine::ProcessorQuery::iterator it =
            Realm::Machine::ProcessorQuery(Realm::Machine::get_machine())
                .only_kind(Realm::Processor::LOC_PROC)
                .begin();
        it; ++it) {
      Processor proc = *it;
      if(proc_map[proc.address_space()] == Realm::Processor::NO_PROC) {
        proc_map[proc.address_space()] = proc;
      }
    }
  }
  std::vector<Realm::Event> finish_events;
  // TODO: Add fill operations for validation purposes
  // This implementation assumes a topologically sorted graph from the graph
  // generator
  for(CopyOperation &op : graph) {
    Realm::Event wait_on = start_event;
    if(op.dependencies.size() > 0) {
      std::vector<Realm::Event> events(op.dependencies.size() + 1,
                                       Realm::Event::NO_EVENT);
      for(size_t i = 0; i < op.dependencies.size(); i++) {
        events[i] = op.dependencies[i]->current_event;
      }
      events.back() = start_event;
      wait_on = Event::merge_events(events);
    }

    realm_address_space_t src_rank = op.src_mem.address_space();
    if(!TestConfig::enable_remote_copy || src_rank == p.address_space()) {
      issue_copy_from_local(finish_events, op, wait_on, p);
    } else {
      issue_copy_from_remote(finish_events, op, wait_on, p, proc_map[src_rank]);
    }
  }
  // And the final event signaling this graph is complete
  return Realm::Event::merge_events(finish_events);
}

static void update_operation_time(const void *args, size_t arglen, const void *userdata,
                                  size_t userlen, Realm::Processor p)
{
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(UpdateOpTimingTaskArgs));
  const UpdateOpTimingTaskArgs &self_args =
      *static_cast<const UpdateOpTimingTaskArgs *>(resp.user_data());
  ProfilingMeasurements::OperationTimeline timeline;

  if(resp.get_measurement(timeline)) {
    self_args.op->measured_time.sample(timeline.complete_time - timeline.start_time);
    self_args.op->lag_time.sample(timeline.start_time - timeline.ready_time);
  } else {
    assert(0 && "Failed to get timeline measurement");
  }
  self_args.profiling_event.trigger();
  log_app.debug("Profile Copy:%p is done on processor %llx", self_args.op, p.id);
}

static void bench_timing_task(const void *args, size_t arglen, const void *userdata,
                              size_t userlen, Processor p)
{
  log_app.print("=== Memory Info ===");
  Realm::Machine::MemoryQuery mq(Realm::Machine::get_machine());
  mq.only_kind(Memory::GPU_FB_MEM).has_capacity(1);
  std::vector<Realm::Memory> memories(mq.begin(), mq.end());
  for(Memory m : memories) {
    display_memory_info(m);
  }
  log_app.print("===================");

  // mq = mq.has_capacity(pow(TestConfig::size, MAX_DIM) * sizeof(ElementType));
  memories.assign(mq.begin(), mq.end());
  if(memories.size() == 0) {
    abort();
  }

  TestGraphFactory *test_factory = nullptr;
  std::map<FieldID, size_t> fields;
  for(size_t i = 0; i < TestConfig::num_fields; i++) {
    fields[i] = TestConfig::field_size;
  }
  test_factory = new MultiFieldTestGraphFactory(memories, TestConfig::size, fields);

  std::vector<CopyOperation> graph;
  test_factory->create(graph);

  Stat graph_time;

  for(size_t sample_iter = 0; sample_iter < TestConfig::num_samples; sample_iter++) {
    Realm::UserEvent trigger_event = Realm::UserEvent::create_user_event();
    Realm::Event current_graph_event = trigger_event;
    for(size_t graph_iter = 0; graph_iter < TestConfig::num_iterations; graph_iter++) {
      current_graph_event = run_graph(graph, current_graph_event, p);
    }

    size_t start_time = Clock::current_time_in_microseconds();
    trigger_event.trigger();    // Start the graphs
    current_graph_event.wait(); // Wait for them to finish
    size_t end_time = Clock::current_time_in_microseconds();

    if(TestConfig::verify) {
      test_factory->verify();
    }

    if(sample_iter != 0) {
      graph_time.sample(double(end_time - start_time) / TestConfig::num_iterations);
    }
    log_app.info() << "\tGraph sample (us): " << end_time - start_time;
  }

  size_t total_size_bytes = CopyOperation::get_total_size(graph.begin(), graph.end());

  log_app.print() << "Graph total transfer size: "
                  << total_size_bytes / (1024ULL * 1024ULL) << "MiB";
  log_app.print() << "Graph time (us): " << graph_time.get_average();
  log_app.print() << "Graph bandwidth (GB/s): "
                  << total_size_bytes / (1000.0 * graph_time.get_average());
  if(TestConfig::enable_profiling) {
    display_node_data(graph);
  }
  graph.clear(); // Should destroy all the created instances

  delete test_factory;

  usleep(100000);
}

int main(int argc, char **argv)
{
  Runtime r;

  bool ok = r.init(&argc, &argv);
  assert(ok);

  CommandLineParser cp;
  cp.add_option_int("-profile", TestConfig::enable_profiling)
      .add_option_int("-remote-copy", TestConfig::enable_remote_copy)
      .add_option_int("-iter", TestConfig::num_iterations)
      .add_option_int("-samples", TestConfig::num_samples)
      .add_option_int("-size", TestConfig::size)
      .add_option_int("-copy_fields", TestConfig::max_copy_fields)
      .add_option_int("-num_fields", TestConfig::num_fields)
      .add_option_int("-field_size", TestConfig::field_size)
      .add_option_int("-graphviz", TestConfig::graphviz)
      .add_option_int("-max_ops", TestConfig::max_ops)
      .add_option_int("-verify", TestConfig::verify);
  ok = cp.parse_command_line(argc, (const char **)argv);
  assert(ok);

  r.register_task(BENCH_TIMING_TASK, bench_timing_task);
  r.register_task(UPDATE_OP_TIMING_TASK, update_operation_time);
  r.register_task(REMOTE_COPY_TASK, remote_copy_task);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();

  // collective launch of a single task - everybody gets the same finish event
  Event e = r.collective_spawn(p, BENCH_TIMING_TASK, 0, 0);

  // request shutdown once that task is complete
  r.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  r.wait_for_shutdown();

  return 0;
}
