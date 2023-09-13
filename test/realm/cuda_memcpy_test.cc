#include "realm/cuda/cuda_memcpy.h"

#include <time.h>

#include <cassert>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "realm.h"
#include "osdep.h"
#include "realm/cmdline.h"
#include "realm/id.h"

using namespace Realm;
using namespace Realm::Cuda;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  COPYPROF_TASK,
};

enum
{
  FID_BASE = 44,
};

// Custom version of uint4 to test 16B transfers
struct alignas(16) Integer128 {
  uint64_t hi, lo;
  Integer128() = default;
  Integer128(uint64_t v)
    : hi(v)
    , lo(v)
  {}
  bool operator==(const Integer128 &o) { return hi == o.hi && lo == o.lo; }
  bool operator!=(const Integer128 &o) { return !operator==(o); }
};

std::ostream &operator<<(std::ostream &os, const Integer128 &dt)
{
  return os << dt.hi << ',' << dt.lo;
}

struct SpeedTestArgs {
  Memory mem;
  RegionInstance inst;
  size_t elements;
  int reps;
  Machine::AffinityDetails affinity;
};

struct CopyProfResult {
  long long *nanoseconds;
  UserEvent done;
};

namespace TestConfig {
  int copy_reps = 1;            // if nonzero, average over #reps copies
  int copy_fields = 1;          // number of distinct fields to copy
  size_t sparse_chunk = 0;      // if nonzero, test sparse copies with chunk size
  size_t sparse_gap = 16;       // gap between sparse chunks (if used)
  size_t chunks = 2;            // number of sparse chunks in each rect
  size_t rects = 1;             // number of rects for the kernel
  size_t rect_increasement = 0; // width increasement between rects
  bool verify = true;          // wether to verify the result of kernel
  bool verbose = false;         // wether to print the result of kernel
  bool test_sparse = false;     // if true, dst will also be sparse
}; // namespace TestConfig

template <int N, typename T, typename FT>
void copy(RegionInstance src_inst, RegionInstance dst_inst, FieldID fid,
          IndexSpace<N, T> index_space)
{
  std::vector<CopySrcDstField> srcs(1), dsts(1);
  srcs[0].set_field(src_inst, fid, sizeof(FT));
  dsts[0].set_field(dst_inst, fid, sizeof(FT));
  index_space.copy(srcs, dsts, ProfilingRequestSet()).wait();
}

template <int N, typename T, typename DT>
void dump_and_verify(RegionInstance inst, RegionInstance proxy_inst, FieldID fid,
                     const IndexSpace<N, T> &is, size_t row_size, DT value,
                     bool verbose = false)
{
  copy<N, T, DT>(inst, proxy_inst, fid, is);
  GenericAccessor<DT, N, T> acc(proxy_inst, fid);
  size_t i = 0;
  bool fail = false;
  for(IndexSpaceIterator<N, T> it(is); it.valid; it.step()) {
    for(PointInRectIterator<N, T> it2(it.rect); it2.valid; it2.step()) {
      DT v = acc[it2.p];
      if(value != DT(-1) && v != value) {
        fail = true;
      }
      if(verbose) {
        if((i++) % row_size == 0)
          std::cout << std::endl;
        std::cout << it2.p << ": " << v << " ";
      }
    }
    if(verbose)
      std::cout << "\n";
    assert(!fail);
  }
}

template <typename T>
void run_test(Processor p)
{
  log_app.print() << "Realm run_test for type size: " << sizeof(T);

  size_t max_stride = TestConfig::sparse_chunk +
                      (TestConfig::rects - 1) * TestConfig::rect_increasement +
                      TestConfig::sparse_gap;

  size_t elements = max_stride * TestConfig::chunks * TestConfig::rects;
  IndexSpace<1> d = Rect<1>(0, elements - 1);

  std::map<FieldID, size_t> field_sizes;
  for(int i = 0; i < TestConfig::copy_fields; i++)
    field_sizes[FID_BASE + i] = sizeof(T);

  IndexSpace<1> d_sparse;
  size_t sparse_elements = 0;
  assert(TestConfig::sparse_chunk > 0);
  std::vector<Rect<1>> rects;
  for(size_t r = 0; r < TestConfig::rects; r++) {
    size_t chunk_size = TestConfig::sparse_chunk + r * TestConfig::rect_increasement;
    for(size_t chunk_idx = 0; chunk_idx < TestConfig::chunks; chunk_idx++) {
      size_t idx = r * TestConfig::chunks + chunk_idx;
      rects.push_back(Rect<1>(idx * max_stride, idx * max_stride + chunk_size - 1));
      sparse_elements += chunk_size;
    }
  }
  log_app.print() << rects.size();
  d_sparse = IndexSpace<1>(rects);
  assert(sparse_elements == d_sparse.volume());
  log_app.print() << "Construct sparse data, chunk: " << TestConfig::sparse_chunk
                  << " gap: " << TestConfig::sparse_gap
                  << " num_sparse_chunks: " << TestConfig::chunks
                  << " d_sparse.volume(): " << d_sparse.volume();

  IndexSpace<1> d_dense = Rect<1>(0, d_sparse.volume() - 1);

  Memory cpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .local_address_space()
                       .only_kind(Memory::Kind::SYSTEM_MEM)
                       .first();

  // create instances on cpu for verifications
  RegionInstance inst_src_cpu;
  RegionInstance::create_instance(inst_src_cpu, cpu_mem, d, field_sizes, 0,
                                  ProfilingRequestSet())
      .wait();
  assert(inst_src_cpu.exists());

  RegionInstance inst_dst_cpu;
  if(TestConfig::test_sparse) {
    RegionInstance::create_instance(inst_dst_cpu, cpu_mem, d, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
  } else {
    RegionInstance::create_instance(inst_dst_cpu, cpu_mem, d_dense, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
  }
  assert(inst_dst_cpu.exists());

  std::vector<Memory> gpu_mems;
  Machine machine = Machine::get_machine();
  for (Machine::MemoryQuery::iterator it =
           Machine::MemoryQuery(machine).begin();
       it; ++it) {
    Memory m = *it;
    if (!ID(m).is_ib_memory() && m.kind() == Memory::GPU_FB_MEM) {
      gpu_mems.push_back(m);
    }
  }

  for (Memory src_mem : gpu_mems) {
    RegionInstance inst_src;
    RegionInstance::create_instance(inst_src, src_mem, d, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
    assert(inst_src.exists());

    // clear the instance first - this should also take care of faulting it in
    {
      std::vector<CopySrcDstField> srcs(TestConfig::copy_fields);
      for(int i = 0; i < TestConfig::copy_fields; i++) {
        srcs[i].set_fill<T>(static_cast<T>(7));
      }
      std::vector<CopySrcDstField> dsts(TestConfig::copy_fields);
      for(int i = 0; i < TestConfig::copy_fields; i++)
        dsts[i].set_field(inst_src, FID_BASE + i, sizeof(T));

      // use sparse idx space foc copy
      d_sparse.copy(srcs, dsts, ProfilingRequestSet()).wait();
      if(TestConfig::verify) {
        dump_and_verify<1, int, T>(inst_src, inst_src_cpu, FID_BASE, d_sparse,
                                   d_sparse.volume(), 7, TestConfig::verbose);
      }
    }

    for (Memory dst_mem: gpu_mems) {
      RegionInstance inst_dst;
      if(TestConfig::test_sparse) {
        RegionInstance::create_instance(inst_dst, dst_mem, d_sparse, field_sizes, 0,
                                        ProfilingRequestSet())
            .wait();
      } else {
        RegionInstance::create_instance(inst_dst, dst_mem, d_dense, field_sizes, 0,
                                        ProfilingRequestSet())
            .wait();
      }
      assert(inst_dst.exists());

      long long sparse_total_time = 0;
      for(int rep = 0; rep < TestConfig::copy_reps; rep++) {
        IndexSpace<1> d_dst = d_dense;
        if(TestConfig::test_sparse) {
          d_dst = d_sparse;
        }

        if (!TestConfig::test_sparse) assert(0);

        std::vector<CopySrcDstField> srcs(TestConfig::copy_fields);
        for (int i = 0; i < TestConfig::copy_fields; i++)
          srcs[i].set_field(inst_src, FID_BASE + i, sizeof(T));
        std::vector<CopySrcDstField> dsts(TestConfig::copy_fields);
        for (int i = 0; i < TestConfig::copy_fields; i++)
          dsts[i].set_field(inst_dst, FID_BASE + i, sizeof(T));

        long long single_copy_time = -1;
        UserEvent single_copy_done = UserEvent::create_user_event();
        {
          CopyProfResult result;
          result.nanoseconds = &single_copy_time;
          result.done = single_copy_done;
          ProfilingRequestSet prs;
          prs.add_request(p, COPYPROF_TASK, &result, sizeof(CopyProfResult))
              .add_measurement<ProfilingMeasurements::OperationTimeline>();
          d_dst.copy(srcs, dsts, prs).wait();
        }
        single_copy_done.wait();
        sparse_total_time += single_copy_time;

        // verify kernel
        if(TestConfig::verify) {
          dump_and_verify<1, int, T>(inst_dst, inst_dst_cpu, FID_BASE, d_dst,
                                     d_dst.volume(), 7, TestConfig::verbose);
        }
      }

      inst_dst.destroy();

      double sparse_bw =
          static_cast<double>(sparse_elements) * sizeof(T) /
          (static_cast<double>(sparse_total_time) * TestConfig::copy_reps);
      double sparse_time_gpu =
          static_cast<double>(sparse_total_time) / TestConfig::copy_reps;

      log_app.print() << "Memory src: " << src_mem << " dst: " << dst_mem
                      << " buffer size(MiB): "
                      << sparse_elements * sizeof(T) / (1024.0 * 1024.0)
                      << " elements: " << elements
                      << " sparse_elements: " << sparse_elements
                      << " sparse_time_gpu: " << sparse_time_gpu
                      << " sparse_bw: " << sparse_bw;
    }
    inst_src.destroy();
  }

  inst_src_cpu.destroy();
  inst_dst_cpu.destroy();
}

void copy_profiling_task(const void *args, size_t arglen, const void *userdata,
                         size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(CopyProfResult));
  const CopyProfResult *result = static_cast<const CopyProfResult *>(resp.user_data());

  ProfilingMeasurements::OperationTimeline timeline;
  if(resp.get_measurement(timeline)) {
    *(result->nanoseconds) = timeline.complete_time - timeline.start_time;
    result->done.trigger();
  } else {
    log_app.fatal() << "no operation timeline in profiling response!";
    assert(0);
  }
}

void top_level_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  // start run_test
  run_test<uint8_t>(p);
  run_test<uint16_t>(p);
  run_test<uint32_t>(p);
  run_test<uint64_t>(p);
  run_test<Integer128>(p);

  usleep(100000);
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int("-reps", TestConfig::copy_reps)
      .add_option_int("-sparse", TestConfig::sparse_chunk)
      .add_option_int("-gap", TestConfig::sparse_gap)
      .add_option_int("-chunks", TestConfig::chunks)
      .add_option_int("-rects", TestConfig::rects)
      .add_option_int("-increase", TestConfig::rect_increasement)
      .add_option_bool("-verify", TestConfig::verify)
      .add_option_bool("-verbose", TestConfig::verbose)
      .add_option_bool("-test-sparse", TestConfig::test_sparse);

  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, COPYPROF_TASK,
                                   CodeDescriptor(copy_profiling_task),
                                   ProfilingRequestSet(), 0, 0)
      .wait();

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // request shutdown once that task is complete
  rt.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();

  return 0;
}

