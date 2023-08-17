#include <time.h>

#include <cassert>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "realm.h"
#include "osdep.h"
#include "philox.h"
#include "realm/cmdline.h"
#include "realm/cuda/cuda_memcpy.h"
#include "realm/cuda/cuda_module.h"
#include "realm/id.h"


using namespace Realm;
using namespace Realm::Cuda;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  MEMSPEED_TASK,
  COPYPROF_TASK,
};

template <int N>
struct LayoutPermutation {
  int dim_order[N];
  char name[N + 1];
};

template <int N>
struct TransposeExperiment {
  const struct LayoutPermutation<N> *src_perm;
  const struct LayoutPermutation<N> *dst_perm;
  long long nanoseconds;
};

template <int N>
void create_permutations(std::vector<LayoutPermutation<N>> &perms,
                         LayoutPermutation<N> &scratch, int pos, size_t narrow_size)
{
  static const char *dim_names = "XYZWABCDEF";

  for(int i = 0; i < N; i++) {
    bool found = false;
    for(int j = 0; j < pos; j++)
      if(scratch.dim_order[j] == i)
        found = true;
    if(found)
      continue;

    scratch.dim_order[pos] = i;
    scratch.name[pos] =
        (((i == 0) && (narrow_size > 0)) ? tolower(dim_names[i]) : dim_names[i]);

    if(pos == (N - 1))
      perms.push_back(scratch);
    else
      create_permutations<N>(perms, scratch, pos + 1, narrow_size);
  }
}

struct CopyProfResult {
  long long *nanoseconds;
  UserEvent done;
};

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

namespace TestConfig {
  size_t field_size = 32;
  size_t block_cols = 4;
  size_t block_rows = 4;
  bool verbose = false;
  bool verify = true;
  size_t buffer_size = 8192 << 4; // should be bigger than any cache in system
  size_t narrow_dim = 0;        // should one dimension be a fixed (narrow) size?
  unsigned dim_mask = 3;        // i.e. all the dimensions
  bool all_mems = true;
  size_t pad_width = 0;
  size_t max_perms = 10;
  unsigned random_seed = 12345; // used to sample permutations if needed
  bool wait_after = false;      // wait after each copy?
  bool do_unit_test = false;
};                              // namespace TestConfig

template <int N, typename T, typename DT>
void dump_and_verify(RegionInstance inst, FieldID fid,
                     const IndexSpace<N, T> &is, size_t row_size,
                     std::vector<DT> test_data, bool verbose = false,
                     bool verify = true) {
  return;
  if (verify || verbose) {
    GenericAccessor<DT, N, T> acc(inst, fid);
    size_t i = 0;
    for (IndexSpaceIterator<N, T> it(is); it.valid; it.step()) {
      for (PointInRectIterator<N, T> it2(it.rect); it2.valid; it2.step()) {
        DT v = acc[it2.p];
        if (verify) {
          if (v != test_data[i]) {
            std::cout << "Mismatch at " << it2.p << ": " << v
                      << " != " << test_data[i] << std::endl;
            assert(0);
          }
          assert(v == test_data[i]);
        }
        if (verbose) {
          if ((i) % row_size == 0) std::cout << std::endl;
          std::cout << it2.p << ": " << v << " ";
        }
        i++;
      }
      if (verbose) std::cout << "\n";
    }
  }
}

template <typename T, int N>
struct Data {
  T data[N];
  Data() {}
  Data(T _val) {
    for (int i = 0; i < N; i++) data[i] = _val + i;
  }
  operator T() const {
    T value = T(0);
    for (int i = 0; i < N; i++) {
      value += data[i];
    }
    return value;
  }
};

template <typename T, int N>
std::ostream &operator<<(std::ostream &, const Data<T, N> &);

template <typename T, int N>
std::ostream &operator<<(std::ostream &os, const Data<T, N> &data)
{
  double value = T(0);
  for (int i = 0; i < N; i++) {
    value += data.data[i];
  }
  os << value;
  return os;
}

template <typename T, size_t BYTES>
struct Pad {
  T val;
  char padding[BYTES - sizeof(T)];
  Pad() {}
  Pad(T _val) : val(_val) {}
  operator T() const { return val; }
};

template <typename T, size_t BYTES>
std::ostream &operator<<(std::ostream &, const Pad<T, BYTES> &);

template <typename T, size_t BYTES>
std::ostream &operator<<(std::ostream &os, const Pad<T, BYTES> &pad)
{
  os << pad.val;
  return os;
}

template <int N, typename T, typename FT>
void copy(RegionInstance src_inst, RegionInstance dst_inst, FieldID fid,
          IndexSpace<N, T> index_space)
{
  std::vector<CopySrcDstField> srcs(1), dsts(1);
  srcs[0].set_field(src_inst, fid, sizeof(FT));
  dsts[0].set_field(dst_inst, fid, sizeof(FT));
  index_space.copy(srcs, dsts, ProfilingRequestSet()).wait();
}

template <int N, typename T, typename DT, typename LAMBDA>
void init_index_space(RegionInstance inst, RegionInstance proxy_inst,
                      IndexSpace<N, T> index_space, FieldID fid, LAMBDA filler,
                      bool use_proxy_inst = false)
{
  GenericAccessor<DT, N, T> acc_data1(use_proxy_inst ? proxy_inst : inst, fid);
  for(IndexSpaceIterator<N, T> it(index_space); it.valid; it.step()) {
    for(PointInRectIterator<N, T> it2(it.rect); it2.valid; it2.step()) {
      acc_data1[it2.p] = filler(it2.p);
    }
  }
  if(use_proxy_inst) {
    copy<N, T, DT>(proxy_inst, inst, fid, index_space);
  }
}

template <int N, typename T, typename FT>
void do_single_dim(Memory src_mem, Memory dst_mem, int log2_size,
                   size_t narrow_size, Processor prof_proc,
                   const std::set<std::string> &src_perms,
                   const std::set<std::string> &dst_perms,
                   Rect<N, T> bounds = Rect<N, T>::make_empty(),
                   Rect<N, T> bounds_pad = Rect<N, T>::make_empty(),
                   int pad = 0) {
  std::vector<LayoutPermutation<N>> perms;
  LayoutPermutation<N> scratch;
  memset(&scratch, 0, sizeof(scratch));
  create_permutations<N>(perms, scratch, 0, narrow_size);

  if((TestConfig::max_perms > 3) && (perms.size() > TestConfig::max_perms)) {
    // mostly-randomly sample permutations in the hopes we get interesting
    //  stuff, but keep the first 3 (i.e. normal fortran order and two that
    //  are very similar to it)
    for(size_t i = 3; i < TestConfig::max_perms; i++) {
      size_t idx = (3 + (Philox_2x32<>::rand_raw(TestConfig::random_seed, N, i) %
                         (perms.size() - 3)));
      perms[i] = perms[idx];
    }
    perms.resize(TestConfig::max_perms);
  }

  if (bounds.empty()) {
    for (int i = 0; i < N; i++) {
      bounds.lo[i] = 0;
      if ((N > 1) && (narrow_size > 0)) {
        if (i == 0)
          bounds.hi[i] = narrow_size - 1;
        else {
          bounds.hi[i] = (1 << (log2_size / std::max(1, N - 1))) - 1;
          // max is for compilers that don't see this code is unreachable for
          // N==1
        }
      } else
        bounds.hi[i] = (1 << (log2_size / N)) - 1;
    }
  }

  log_app.info() << "bounds=" << bounds;
  IndexSpace<N> is(bounds);

  if (bounds_pad.empty()) {
    for (int i = 0; i < N; i++) {
      bounds_pad.lo[i] = 0;
      bounds_pad.hi[i] = bounds.hi[i] + (((narrow_size > 0) &&
                                          (narrow_size < TestConfig::pad_width))
                                             ? narrow_size
                                             : TestConfig::pad_width);
    }
  }

  log_app.info() << "bounds_pad=" << bounds_pad;
  IndexSpace<N> is_pad(bounds_pad);

  std::map<FieldID, size_t> field_sizes;
  field_sizes[0] = sizeof(FT);
  InstanceLayoutConstraints ilc(field_sizes, 1);

  std::vector<FT> test_data;
  std::vector<TransposeExperiment<N> *> experiments;

  Event wait_for = Event::NO_EVENT;
  std::vector<Event> done_events;
  for(unsigned i = 0; i < perms.size(); i++) {
    if (src_perms.find(perms[i].name) == src_perms.end()) continue;
    RegionInstance src_inst;
    {
      InstanceLayoutGeneric *ilg = InstanceLayoutGeneric::choose_instance_layout<N, T>(
          is_pad, ilc, perms[i].dim_order);
      wait_for = RegionInstance::create_instance(src_inst, src_mem, ilg,
                                                 ProfilingRequestSet(), wait_for);

      int index = 0;
      auto src_lambda = [&](Point<N, T> p) -> FT {
        test_data.push_back(index+N);
        return index += N;
      };
      init_index_space<N, T, FT>(src_inst, src_inst, is, /*field_id=*/0,
                                 src_lambda, /*use_proxy_inst=*/false);
    }

    for (unsigned j = 0; j < perms.size(); j++) {
      if (dst_perms.find(perms[j].name) == dst_perms.end()) continue;
      RegionInstance dst_inst;
      {
        InstanceLayoutGeneric *ilg =
            InstanceLayoutGeneric::choose_instance_layout<N, T>(
                is_pad, ilc, perms[j].dim_order);
        wait_for = RegionInstance::create_instance(
            dst_inst, dst_mem, ilg, ProfilingRequestSet(), wait_for);
      }

      wait_for.wait();

      TransposeExperiment<N> *exp = new TransposeExperiment<N>;
      exp->src_perm = &perms[i];
      exp->dst_perm = &perms[j];
      exp->nanoseconds = 0;

      UserEvent done = UserEvent::create_user_event();
      done_events.push_back(done);

      CopyProfResult cpr;
      cpr.nanoseconds = &(exp->nanoseconds);
      cpr.done = done;

      experiments.push_back(exp);

      ProfilingRequestSet prs;
      prs.add_request(prof_proc, COPYPROF_TASK, &cpr, sizeof(CopyProfResult))
          .add_measurement<ProfilingMeasurements::OperationTimeline>();
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].inst = src_inst;
      srcs[0].field_id = 0;
      srcs[0].size = field_sizes[0];
      dsts[0].inst = dst_inst;
      dsts[0].field_id = 0;
      dsts[0].size = field_sizes[0];

      wait_for = is.copy(srcs, dsts, prs, wait_for);
      wait_for.wait();

      dump_and_verify<N, T, FT>(src_inst, /*field_id=*/0, is,
                                is.bounds.hi[i] + 1, test_data,
                                TestConfig::verbose, TestConfig::verify);
      dump_and_verify<N, T, FT>(dst_inst, /*field_id=*/0, is,
                                is.bounds.hi[i] + 1, test_data,
                                TestConfig::verbose, TestConfig::verify);

      dst_inst.destroy(wait_for);
    }

    src_inst.destroy(wait_for);
  }

  // wait for copies to finish
  done_events.push_back(wait_for);
  Event::merge_events(done_events).wait();

  for(typename std::vector<TransposeExperiment<N> *>::const_iterator it =
          experiments.begin();
      it != experiments.end(); ++it) {
    double bw = 1.0 * is.volume() * field_sizes[0] / (*it)->nanoseconds;

    log_app.print() << "Test passed, src=" << (*it)->src_perm->name
                    << " dst=" << (*it)->dst_perm->name << " bounds=" << bounds
                    << " bounds_pad=" << bounds_pad
                    << " field_size=" << sizeof(FT)
                    << " time=" << (1e-9 * (*it)->nanoseconds) << " bw=" << bw;
    delete *it;
  }
}

template <int N, typename T>
void do_single_dim_field_size(Memory src_mem, Memory dst_mem, int log2_size,
                              size_t narrow_size, Processor prof_proc,
                              const std::set<std::string> &src_perms,
                              const std::set<std::string> &dst_perms,
                              Rect<N, T> bounds = Rect<N, T>::make_empty(),
                              Rect<N, T> bounds_pad = Rect<N, T>::make_empty(),
                              size_t field_size = 8, int pad = 0) {
  if (TestConfig::field_size == 8) {
    do_single_dim<N, T, double>(src_mem, dst_mem, log2_size,
                                TestConfig::narrow_dim, prof_proc, src_perms, dst_perms,
                                bounds, bounds_pad, pad);
  } else if (TestConfig::field_size == 16) {
    typedef Data<double, 2> FieldStruct16;
    do_single_dim<N, T, FieldStruct16>(src_mem, dst_mem, log2_size,
                                       TestConfig::narrow_dim, prof_proc, src_perms,
                                       dst_perms, bounds, bounds_pad, pad);
  } else if (TestConfig::field_size == 24) {
    typedef Data<double, 3> FieldStruct24;
    do_single_dim<N, T, FieldStruct24>(src_mem, dst_mem, log2_size,
                                       TestConfig::narrow_dim, prof_proc, src_perms,
                                       dst_perms, bounds, bounds_pad, pad);
  }
  if (TestConfig::field_size == 32) {
    typedef Data<double, 4> FieldStruct32;
    do_single_dim<N, T, FieldStruct32>(src_mem, dst_mem, log2_size,
                                       TestConfig::narrow_dim, prof_proc, src_perms,
                                       dst_perms, bounds, bounds_pad, pad);
  }
  if (TestConfig::field_size == 40) {
    typedef Data<char, 40> FieldStruct40;
    do_single_dim<N, T, FieldStruct40>(src_mem, dst_mem, log2_size,
                                       TestConfig::narrow_dim, prof_proc, src_perms,
                                       dst_perms, bounds, bounds_pad, pad);
  }
}


std::set<Processor::Kind> supported_proc_kinds;

void top_level_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p) {
  log_app.print() << "Copy/transpose test";

  size_t log2_buffer_size = 0;
  {
    size_t v = TestConfig::buffer_size / sizeof(int);
    if (TestConfig::narrow_dim > 0) v /= TestConfig::narrow_dim;
    while (v > 1) {
      v >>= 1;
      log2_buffer_size++;
    }
  }

  std::vector<Memory> src_mems, dst_mems;
  Machine::MemoryQuery mq = Machine::MemoryQuery(Machine::get_machine())
                                .has_capacity(TestConfig::buffer_size * 2)
                                .only_kind(Memory::GPU_FB_MEM);
  src_mems.assign(mq.begin(), mq.end());
  dst_mems = src_mems;

  for (std::vector<Memory>::const_iterator src_it = src_mems.begin();
       src_it != src_mems.end(); ++src_it) {
    if (NodeID(ID(*src_it).memory_owner_node()) != Network::my_node_id)
      continue;
    for (std::vector<Memory>::const_iterator dst_it = dst_mems.begin();
         dst_it != dst_mems.end(); ++dst_it) {
      log_app.print() << "srcmem=" << *src_it << " dstmem=" << *dst_it;

      if (NodeID(ID(*dst_it).memory_owner_node()) != Network::my_node_id)
        continue;
      if (TestConfig::dim_mask == 1) {
        // TODO(apryakhin@): Add support/tests for dim=1
        assert(0);
        // do_single_dim<1, FT>(*src_it, *dst_it, log2_buffer_size, 0, p);
      } else if (TestConfig::dim_mask == 2) {
        do_single_dim_field_size<2, int>(*src_it, *dst_it, log2_buffer_size,
                                         TestConfig::narrow_dim, p, {"XY"},
                                         {"YX"});
      } else if (TestConfig::dim_mask == 3) {
        if (!TestConfig::do_unit_test) {
          do_single_dim_field_size<3, int>(*src_it, *dst_it, log2_buffer_size,
                                           TestConfig::narrow_dim, p, {"XYZ"},
                                           {"YXZ"});
        } else {
          const size_t block_cols = TestConfig::block_cols;
          const size_t block_rows = TestConfig::block_rows;
          for (size_t xscale = 1; xscale < 4; xscale *= 2) {
            for (size_t yscale = 1; yscale < 4; yscale *= 2) {
              {
                size_t zscale = 1;
                for (int pad_x = -1; pad_x <= 1; pad_x++) {
                  for (int pad_y = -1; pad_y <= 1; pad_y++) {
                    do_single_dim_field_size<3>(
                        *src_it, *dst_it, log2_buffer_size,
                        TestConfig::narrow_dim, p, {"XYZ"}, {"YXZ"},
                        Rect<3>(Point<3>(0, 0, 0),
                                Point<3>((xscale * block_cols) + pad_x,
                                         (yscale * block_rows) + pad_y,
                                         zscale * block_rows)),
                        Rect<3>(Point<3>(0, 0, 0),
                                Point<3>((xscale * block_cols) + pad_x,
                                         (yscale * block_rows) + pad_y,
                                         zscale * block_rows)),
                        TestConfig::field_size);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int("-b", TestConfig::buffer_size)
      .add_option_int("-narrow", TestConfig::narrow_dim)
      .add_option_int("-dims", TestConfig::dim_mask)
      .add_option_int("-block_cols", TestConfig::block_cols)
      .add_option_int("-block_rows", TestConfig::block_rows)
      .add_option_int("-field_size", TestConfig::field_size)
      .add_option_int("-all", TestConfig::all_mems)
      .add_option_int("-pad", TestConfig::pad_width)
      .add_option_int("-perms", TestConfig::max_perms)
      .add_option_int("-seed", TestConfig::random_seed)
      .add_option_int("-verbose", TestConfig::verbose)
      .add_option_int("-verify", TestConfig::verify)
      .add_option_int("-wait", TestConfig::wait_after)
      .add_option_int("-unit_test", TestConfig::do_unit_test);
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
