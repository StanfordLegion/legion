#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "osdep.h"
#include "realm.h"
#include "realm/cmdline.h"

#ifdef ENABLE_DIRECT_TEST
#include "realm/cuda/cuda_internal.h"
#endif

#include "realm/cuda/cuda_module.h"
#include "realm/id.h"

using namespace Realm;
using namespace Realm::Cuda;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  COPYPROF_TASK,
  DYNAMIC_TASK_START,
};

enum {
  FID_PTR1 = 100,
  FID_DATA1 = 200,
  FID_DATA2,
  FID_RANGE1 = 300,
  FID_RANGE2,
};

enum {
  SERDEZ_WRAP_FLOAT = 444,
};

template <int N>
struct TransposeExperiment {
  long long nanoseconds;
};

template <typename T>
struct WrappingSerdez {
  typedef T FIELD_TYPE;
  static const size_t MAX_SERIALIZED_SIZE = sizeof(T) + sizeof(size_t);

  static size_t serialized_size(const T &val) { return MAX_SERIALIZED_SIZE; }

  static size_t serialize(const T &val, void *buffer) {
    size_t size = sizeof(T);
    memcpy(buffer, &size, sizeof(size_t));
    memcpy(static_cast<char *>(buffer) + sizeof(size_t), &val, sizeof(T));
    return MAX_SERIALIZED_SIZE;
  }

  static size_t deserialize(T &val, const void *buffer) {
    size_t size;
    memcpy(&size, buffer, sizeof(size_t));
    assert(size == sizeof(T));
    memcpy(&val, static_cast<const char *>(buffer) + sizeof(size_t), sizeof(T));
    return MAX_SERIALIZED_SIZE;
  }

  static void destroy(T &val) {}
};

struct CopyProfResult {
  long long *nanoseconds;
  UserEvent done;
};

void copy_profiling_task(const void *args, size_t arglen, const void *userdata,
                         size_t userlen, Processor p) {
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(CopyProfResult));
  const CopyProfResult *result =
      static_cast<const CopyProfResult *>(resp.user_data());

  ProfilingMeasurements::OperationTimeline timeline;
  if (resp.get_measurement(timeline)) {
    *(result->nanoseconds) = timeline.complete_time - timeline.start_time;
    result->done.trigger();
  } else {
    log_app.fatal() << "no operation timeline in profiling response!";
    assert(0);
  }
}

static const size_t MAX_TEST_DIM = 3;

namespace TestConfig {
size_t cuda_grid_x = 1;
size_t sizes1[MAX_TEST_DIM];
size_t sizes2[MAX_TEST_DIM];
// size_t size1 = 64 << 1;
// size_t size2 = 64 << 1;
size_t pieces1 = 1;
size_t pieces2 = 1;
bool skipfirst = false;
bool splitcopies = false;
bool do_scatter = true;
bool do_direct = false;
bool verbose = false;
bool verify = true;
bool ind_on_gpu = true;
bool remote_gather = false;
};  // namespace TestConfig

struct SpeedTestArgs {
  Memory mem;
  RegionInstance inst;
  size_t elements;
  int reps;
  Machine::AffinityDetails affinity;
};

typedef std::map<FieldID, size_t> FieldMap;

// maybe type - used to handle cases where expected value is not known
//  (e.g. due to aliased scatters)
template <typename T>
class Maybe {
 public:
  Maybe() : valid(false) {}
  Maybe(T _val) : valid(true), value(_val) {}
  bool has_value() const { return valid; }
  T get_value() const {
    assert(valid);
    return value;
  }

 protected:
  bool valid;
  T value;
};

template <int N, typename T, typename DT>
void dump_instance(RegionInstance inst, FieldID fid, const IndexSpace<N, T> &is,
                   size_t row_size) {
  if (!TestConfig::verbose) return;
  size_t i = 0;
  GenericAccessor<DT, N, T> acc(inst, fid);
  for (IndexSpaceIterator<N, T> it(is); it.valid; it.step()) {
    for (PointInRectIterator<N, T> it2(it.rect); it2.valid; it2.step()) {
      DT v = acc[it2.p];

      if ((i++) % (row_size + 1) == 0) std::cout << std::endl;
      std::cout << it2.p << ": " << v << " ";
    }
    std::cout << "\n";
  }
}

template <int N, typename T>
class DistributedData {
 public:
  static const int _N = N;
  typedef T _T;

  ~DistributedData();

  void add_subspaces(IndexSpace<N, T> is, size_t count,
                     size_t num_subrects = 1);
  void add_subspaces(const std::vector<IndexSpace<N, T>> &subspaces);

  template <typename LAMBDA>
  Event create_instances(const FieldMap &fields, LAMBDA mem_picker,
                         int offset = 0, bool inverse = false);

  void destroy_instances(Event wait_on);

  template <typename FT, typename LAMBDA>
  Event fill(IndexSpace<N, T> is, FieldID fid, LAMBDA filler, Event wait_on);

  template <typename FT, typename SRC>
  Event gather(IndexSpace<N, T> is, FieldID ptr_id, DistributedData<N, T> &ind,
               const SRC &src, FieldID src_id, FieldID dst_id,
               bool oor_possible, bool aliasing_possible,
               CustomSerdezID serdez_id, Event wait_on, Processor p,
               TransposeExperiment<N> *exp);

  template <typename FT, typename DST>
  Event scatter(IndexSpace<N, T> is, FieldID ptr_id, DistributedData<N, T> &ind,
                DST &dst, FieldID src_id, FieldID dst_id, bool oor_possible,
                bool aliasing_possible, CustomSerdezID serdez_id, Event wait_on,
                Processor p, TransposeExperiment<N> *exp) const;

#ifdef ENABLE_DIRECT_TEST
  template <typename FT, typename DST>
  Event direct_scatter(IndexSpace<N, T> is, FieldID ptr_id,
                       DistributedData<N, T> &ind, DST &dst, FieldID src_id,
                       FieldID dst_id, bool oor_possible,
                       bool aliasing_possible, CustomSerdezID serdez_id,
                       Event wait_on) const;
#endif

  template <typename FT>
  bool verify(IndexSpace<N, T> is, FieldID fid, Event wait_on);

 protected:
  template <int N2, typename T2>
  friend class DistributedData;

  struct Piece {
    std::vector<Rect<N, T>> subrects;
    IndexSpace<N, T> space;
    Processor proc;
    RegionInstance inst, cpu_inst;
  };
  std::vector<Piece> pieces;

  struct RefDataEntry {
    void *data;
    void (*deleter)(void *);
  };
  std::map<FieldID, RefDataEntry> ref_data;
  template <typename FT>
  std::map<Point<N, T>, Maybe<FT>> &get_ref_data(FieldID field_id);
  template <typename FT>
  const std::map<Point<N, T>, Maybe<FT>> &get_ref_data(FieldID field_id) const;
};

template <int N, typename T>
DistributedData<N, T>::~DistributedData() {
  for (typename std::map<FieldID, RefDataEntry>::const_iterator it =
           ref_data.begin();
       it != ref_data.end(); ++it)
    (it->second.deleter)(it->second.data);
  ref_data.clear();
}

template <int N, typename T>
void DistributedData<N, T>::add_subspaces(IndexSpace<N, T> is, size_t count,
                                          size_t num_subrects) {
  if (count == 1) {
    size_t idx = pieces.size();
    pieces.resize(idx + 1);
    pieces[idx].space = is;
    pieces[idx].proc = Processor::NO_PROC;
    pieces[idx].inst = RegionInstance::NO_INST;

    std::vector<IndexSpace<N, T>> index_spaces;
    is.create_equal_subspaces(/*num_pieces=*/num_subrects, /*granularity=*/1,
                              index_spaces, ProfilingRequestSet())
        .wait();
    std::vector<Rect<N, T>> subrects;
    for (size_t i = 0; i < num_subrects; i++) {
      subrects.push_back(index_spaces[i].bounds);
    }
    pieces[idx].subrects = subrects;

  } else {
    std::vector<IndexSpace<N, T>> subspaces;
    is.create_equal_subspaces(count, 1, subspaces, ProfilingRequestSet())
        .wait();
    add_subspaces(subspaces);
  }
}

template <int N, typename T>
void DistributedData<N, T>::add_subspaces(
    const std::vector<IndexSpace<N, T>> &subspaces) {
  size_t base = pieces.size();
  pieces.resize(base + subspaces.size());
  for (size_t i = 0; i < subspaces.size(); i++) {
    pieces[base + i].space = subspaces[i];
    pieces[base + i].proc = Processor::NO_PROC;
    pieces[base + i].inst = RegionInstance::NO_INST;
  }
}

template <int N, typename T>
template <typename LAMBDA>
Event DistributedData<N, T>::create_instances(const FieldMap &fields,
                                              LAMBDA mem_picker, int offset,
                                              bool inverse) {
  std::vector<Event> events;
  for (size_t i = 0; i < pieces.size(); i++) {
    Memory m = mem_picker(i + offset, pieces[i].space);
    Processor p = Machine::ProcessorQuery(Machine::get_machine())
                      .only_kind(Processor::LOC_PROC)
                      .has_affinity_to(m)
                      .first();

    // if no processor has affinity, at least pick one in same address space
    Memory cpu_mem = Memory::NO_MEMORY;
    if (!p.exists()) {
      p = Machine::ProcessorQuery(Machine::get_machine())
              .only_kind(Processor::LOC_PROC)
              .same_address_space_as(m)
              .first();
      assert(p.exists());
      cpu_mem = Machine::MemoryQuery(Machine::get_machine())
                    .only_kind(Memory::SYSTEM_MEM)
                    .has_affinity_to(p)
                    .has_capacity(1)
                    .first();
      assert(cpu_mem.exists());
    }

    pieces[i].proc = p;

    {
      int dim_order[N];
      for (int i = 0; i < N; i++) {
        dim_order[i] = i;
        // dim_order[inverse ? i : N - i - 1] = i;
      }
      InstanceLayoutConstraints ilc(fields, 1);
      InstanceLayoutGeneric *ilg =
          InstanceLayoutGeneric::choose_instance_layout<N, T>(
              pieces[i].space, pieces[i].subrects, ilc, dim_order);

      Event e = RegionInstance::create_instance(pieces[i].inst, m, ilg,
                                                ProfilingRequestSet());
      events.push_back(e);
    }

    if (cpu_mem.exists()) {
      Event e = RegionInstance::create_instance(
          pieces[i].cpu_inst, cpu_mem, pieces[i].space, fields, 0 /* SOA */,
          ProfilingRequestSet());
      events.push_back(e);
    } else
      pieces[i].cpu_inst = RegionInstance::NO_INST;
  }
  return Event::merge_events(events);
}

template <int N, typename T>
void DistributedData<N, T>::destroy_instances(Event wait_on) {
  for (typename std::vector<Piece>::iterator it = pieces.begin();
       it != pieces.end(); ++it) {
    it->inst.destroy(wait_on);
    it->inst = RegionInstance::NO_INST;
    if (it->cpu_inst.exists()) {
      it->cpu_inst.destroy(wait_on);
      it->cpu_inst = RegionInstance::NO_INST;
    }
    it->proc = Processor::NO_PROC;
  }
}

template <typename T>
static void delete_object(void *obj) {
  delete reinterpret_cast<T *>(obj);
}

template <int N, typename T>
template <typename FT>
std::map<Point<N, T>, Maybe<FT>> &DistributedData<N, T>::get_ref_data(
    FieldID field_id) {
  typename std::map<FieldID, RefDataEntry>::const_iterator it =
      ref_data.find(field_id);
  if (it != ref_data.end()) {
    return *reinterpret_cast<std::map<Point<N, T>, Maybe<FT>> *>(
        it->second.data);
  } else {
    std::map<Point<N, T>, Maybe<FT>> *newmap =
        new std::map<Point<N, T>, Maybe<FT>>;
    RefDataEntry &e = ref_data[field_id];
    e.data = reinterpret_cast<void *>(newmap);
    e.deleter = &delete_object<std::map<Point<N, T>, Maybe<FT>>>;
    return *newmap;
  }
}

template <int N, typename T>
template <typename FT>
const std::map<Point<N, T>, Maybe<FT>> &DistributedData<N, T>::get_ref_data(
    FieldID field_id) const {
  typename std::map<FieldID, RefDataEntry>::const_iterator it =
      ref_data.find(field_id);
  assert(it != ref_data.end());
  return *reinterpret_cast<const std::map<Point<N, T>, Maybe<FT>> *>(
      it->second.data);
}

template <int N, typename T, typename FT, typename LAMBDA>
class FillerTask {
 public:
  struct Args {
    Args(LAMBDA _filler) : filler(_filler) {}
    IndexSpace<N, T> space;
    RegionInstance inst;
    FieldID field_id;
    LAMBDA filler;
  };

  static void task_body(const void *argdata, size_t arglen,
                        const void *userdata, size_t userlen, Processor p) {
    assert(sizeof(Args) == arglen);
    const Args &args = *reinterpret_cast<const Args *>(argdata);
    // log_app.info() << "filler: is=" << args.space << " inst=" << args.inst;

    args.inst.fetch_metadata(p).wait();
    AffineAccessor<FT, N, T> acc(args.inst, args.field_id);
    IndexSpaceIterator<N, T> it(args.space);
    while (it.valid) {
      PointInRectIterator<N, T> pit(it.rect);
      while (pit.valid) {
        FT val = args.filler(pit.p);
        log_app.debug() << "  [" << pit.p << "] = " << val;
        acc[pit.p] = val;
        pit.step();
      }
      it.step();
    }
  }
};

Processor::TaskFuncID next_func_id = DYNAMIC_TASK_START;
std::map<const char *, Processor::TaskFuncID> task_ids;

template <typename T>
static Processor::TaskFuncID lookup_task_id() {
  const char *key = typeid(T).name();
  std::map<const char *, Processor::TaskFuncID>::const_iterator it =
      task_ids.find(key);
  if (it != task_ids.end()) return it->second;

  Processor::TaskFuncID id = next_func_id++;
  Event e = Processor::register_task_by_kind(
      Processor::LOC_PROC,
#ifdef _MSC_VER
      // no portable task registration for windows yet
      false /*!global*/,
#else
      true /*global*/,
#endif
      id, CodeDescriptor(&T::task_body), ProfilingRequestSet());
  e.wait();
  task_ids[key] = id;
  return id;
}

template <int N, typename T>
template <typename FT, typename LAMBDA>
Event DistributedData<N, T>::fill(IndexSpace<N, T> is, FieldID fid,
                                  LAMBDA filler, Event wait_on) {
  typename FillerTask<N, T, FT, LAMBDA>::Args args(filler);
  args.field_id = fid;
  Processor::TaskFuncID id = lookup_task_id<FillerTask<N, T, FT, LAMBDA>>();
  std::vector<Event> events;
  for (typename std::vector<Piece>::iterator it = pieces.begin();
       it != pieces.end(); ++it) {
    IndexSpace<N, T> isect;
    IndexSpace<N, T>::compute_intersection(is, it->space, isect,
                                           ProfilingRequestSet())
        .wait();
    args.space = isect;
    args.inst = it->cpu_inst.exists() ? it->cpu_inst : it->inst;
    Event e =
        it->proc.spawn(id, &args, sizeof(args), ProfilingRequestSet(), wait_on);

    // do a copy if we're using a proxy cpu instance
    if (it->cpu_inst.exists()) {
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].set_field(it->cpu_inst, fid, sizeof(FT));
      dsts[0].set_field(it->inst, fid, sizeof(FT));
      e = isect.copy(srcs, dsts, ProfilingRequestSet(), e);
    }

    events.push_back(e);
  }

  // update reference data
  std::map<Point<N, T>, Maybe<FT>> &ref = get_ref_data<FT>(fid);
  IndexSpaceIterator<N, T> it(is);
  while (it.valid) {
    PointInRectIterator<N, T> pit(it.rect);
    while (pit.valid) {
      ref[pit.p] = filler(pit.p);
      pit.step();
    }
    it.step();
  }

  return Event::merge_events(events);
}

#ifdef ENABLE_DIRECT_TEST
// TODO(apryakhin@): Refactor to run either gathe or scatter or
// gather-scatter.
template <int N, typename T>
template <typename FT, typename DST>
Event DistributedData<N, T>::direct_scatter(
    IndexSpace<N, T> is, FieldID ptr_id, DistributedData<N, T> &ind, DST &dst,
    FieldID src_id, FieldID dst_id, bool oor_possible, bool aliasing_possible,
    CustomSerdezID serdez_id, Event wait_on) const {
  std::vector<Event> events;
  for (typename std::vector<Piece>::const_iterator it = pieces.begin();
       it != pieces.end(); ++it) {
    for (typename std::vector<typename DST::Piece>::const_iterator it2 =
             dst.pieces.begin();
         it2 != dst.pieces.end(); ++it2) {
      RuntimeImpl *rt = Realm::get_runtime();
      auto module = rt->get_module<Realm::Cuda::CudaModule>("cuda");
      // Just suppport a single GPU test for now.
      assert(module->gpus.size() == 1);

      // Fetch affine bounds.
      auto inst_impl = get_runtime()->get_instance_impl(it2->inst);
      const InstanceLayout<DST::_N, typename DST::_T> *inst_layout =
          checked_cast<const InstanceLayout<DST::_N, typename DST::_T> *>(
              inst_impl->metadata.layout);

      const InstancePieceList<DST::_N, typename DST::_T> &piece_list =
          inst_layout->piece_lists[0];
      const InstanceLayoutPiece<DST::_N, typename DST::_T> *layout_piece =
          piece_list.find_piece(Point<DST::_N, typename DST::_T>::ZEROES());

      const AffineLayoutPiece<DST::_N, typename DST::_T> *affine =
          static_cast<const AffineLayoutPiece<DST::_N, typename DST::_T> *>(
              layout_piece);

      Realm::Cuda::MemcpyUnstructuredInfo<DST::_N> copy_info = {};

      size_t count = sizeof(FT);
      for (int i = 1; i < DST::_N; i++) {
        copy_info.dst.strides[i - 1] = affine->strides[i] / count;
        count *= copy_info.dst.strides[i - 1];
      }
      copy_info.volume = is.volume();
      copy_info.src.addr =
          reinterpret_cast<uintptr_t>(it->inst.pointer_untyped(0, 0));

      assert(ind.pieces.size() == 1);
      copy_info.src_ind = 0;
      copy_info.dst_ind =
          reinterpret_cast<uintptr_t>(ind.pieces[0].inst.pointer_untyped(0, 0));
      copy_info.dst.addr =
          reinterpret_cast<uintptr_t>(it2->inst.pointer_untyped(0, 0));
      copy_info.field_size = sizeof(FT);

      size_t field_size = std::min<size_t>(sizeof(FT), 8);

      char kernel_name[30];
      std::snprintf(kernel_name, sizeof(kernel_name),
                    "run_memcpy_indirect%uD_%lu", DST::_N, field_size << 3);

      auto gpu = module->gpus[0];
      CUfunction memcpy_fn = 0;
      CHECK_CU(CUDA_DRIVER_FNPTR(cuModuleGetFunction)(
          &memcpy_fn, gpu->device_module, kernel_name));

      void *params[] = {&copy_info};
      size_t threads_per_block = 256;
      int blocks_per_grid = 0;

      gpu->push_context();
      CHECK_CU(CUDA_DRIVER_FNPTR(cuOccupancyMaxActiveBlocksPerMultiprocessor)(
          &blocks_per_grid, memcpy_fn, threads_per_block, 0));

      blocks_per_grid = std::min(
          blocks_per_grid,
          (int)((is.volume() + threads_per_block - 1) / threads_per_block));

      auto stream = gpu->get_next_d2d_stream()->get_stream();

      CHECK_CU(CUDA_DRIVER_FNPTR(cuLaunchKernel)(memcpy_fn, blocks_per_grid, 1,
                                                 1, threads_per_block, 1, 1, 0,
                                                 stream, params, 0));

      CHECK_CU(CUDA_DRIVER_FNPTR(cuStreamSynchronize)(stream));

      gpu->pop_context();

      dump_instance<N, T, FT>(it->inst, FID_DATA1, is, is.bounds.hi[0]);

      dump_instance<N, T, Point<DST::_N, typename DST::_T>>(
          ind.pieces[0].inst, ptr_id, is, is.bounds.hi[0]);

      dump_instance<DST::_N, typename DST::_T, FT>(
          it2->inst, FID_DATA1, it2->space, it2->space.bounds.hi[0]);
    }
  }

  // update reference data
  const std::map<Point<N, T>, Maybe<Point<DST::_N, typename DST::_T>>> &ptrref =
      ind.get_ref_data<Point<DST::_N, typename DST::_T>>(ptr_id);

  const std::map<Point<N, T>, Maybe<FT>> &srcref = get_ref_data<FT>(src_id);

  std::map<Point<DST::_N, typename DST::_T>, Maybe<FT>> &dstref =
      dst.template get_ref_data<FT>(dst_id);

  std::set<Point<DST::_N, typename DST::_T>> touched;  // to detect aliasing
                                                       //
  IndexSpaceIterator<N, T> it(is);

  while (it.valid) {
    PointInRectIterator<N, T> pit(it.rect);
    while (pit.valid) {
      Point<DST::_N, typename DST::_T> p2 = ptrref.at(pit.p).get_value();
      if (dstref.count(p2) > 0) {
        if (touched.count(p2) > 0) {
          assert(aliasing_possible);
          dstref[p2] = Maybe<FT>();
        } else {
          dstref[p2] = srcref.at(pit.p);
          touched.insert(p2);
        }
      } else
        assert(oor_possible);  // make sure we didn't lie to Realm
      pit.step();
    }
    it.step();
  }
  return Event::merge_events(events);
}
#endif

template <int N, typename T>
template <typename FT, typename DST>
Event DistributedData<N, T>::scatter(IndexSpace<N, T> is, FieldID ptr_id,
                                     DistributedData<N, T> &ind, DST &dst,
                                     FieldID src_id, FieldID dst_id,
                                     bool oor_possible, bool aliasing_possible,
                                     CustomSerdezID serdez_id, Event wait_on,
                                     Processor p,
                                     TransposeExperiment<N> *exp) const {
  std::vector<Event> events;
  for (typename std::vector<Piece>::const_iterator it = pieces.begin();
       it != pieces.end(); ++it) {
    IndexSpace<N, T> isect;
    IndexSpace<N, T>::compute_intersection(is, it->space, isect,
                                           ProfilingRequestSet())
        .wait();

    typename CopyIndirection<N, T>::template Unstructured<DST::_N,
                                                          typename DST::_T>
        indirect;
    indirect.field_id = ptr_id;
    indirect.inst = ind.pieces[0].inst;
    indirect.is_ranges = false;
    indirect.subfield_offset = 0;
    indirect.oor_possible = oor_possible;
    indirect.aliasing_possible = aliasing_possible;
    indirect.next_indirection = nullptr;

    std::vector<CopySrcDstField> srcs, dsts;
    srcs.resize(1);
    dsts.resize(1);
    srcs[0].set_field(it->inst, src_id, sizeof(FT));
    dsts[0].set_indirect(0, dst_id, sizeof(FT));
    if (serdez_id != 0) {
      srcs[0].set_serdez(serdez_id);
      dsts[0].set_serdez(serdez_id);
    }

    for (typename std::vector<typename DST::Piece>::const_iterator it2 =
             dst.pieces.begin();
         it2 != dst.pieces.end(); ++it2) {
      indirect.spaces.push_back(it2->space);
      indirect.insts.push_back(it2->inst);
    }

    exp->nanoseconds = 0;

    UserEvent done = UserEvent::create_user_event();
    events.push_back(done);

    CopyProfResult cpr;
    cpr.nanoseconds = &(exp->nanoseconds);
    cpr.done = done;

    ProfilingRequestSet prs;
    prs.add_request(p, COPYPROF_TASK, &cpr, sizeof(CopyProfResult))
        .add_measurement<ProfilingMeasurements::OperationTimeline>();

    Event e = is.copy(
        srcs, dsts,
        std::vector<const typename CopyIndirection<N, T>::Base *>(1, &indirect),
        prs, wait_on);
    e.wait();

    dump_instance<N, T, FT>(it->inst, FID_DATA1, is, is.bounds.hi[0]);

    dump_instance<N, T, Point<DST::_N, typename DST::_T>>(
        ind.pieces[0].inst, ptr_id, is, is.bounds.hi[0]);

    dump_instance<DST::_N, typename DST::_T, FT>(
        indirect.insts[0], FID_DATA1, indirect.spaces[0],
        indirect.spaces[0].bounds.hi[0]);
    // indirect.spaces[0].bounds.hi[DST::_N-1]);
  }

  // update reference data
  const std::map<Point<N, T>, Maybe<Point<DST::_N, typename DST::_T>>> &ptrref =
      ind.get_ref_data<Point<DST::_N, typename DST::_T>>(ptr_id);

  const std::map<Point<N, T>, Maybe<FT>> &srcref = get_ref_data<FT>(src_id);

  std::map<Point<DST::_N, typename DST::_T>, Maybe<FT>> &dstref =
      dst.template get_ref_data<FT>(dst_id);

  std::set<Point<DST::_N, typename DST::_T>> touched;  // to detect aliasing
  IndexSpaceIterator<N, T> it(is);
  while (it.valid) {
    PointInRectIterator<N, T> pit(it.rect);
    while (pit.valid) {
      Point<DST::_N, typename DST::_T> p2 = ptrref.at(pit.p).get_value();
      if (dstref.count(p2) > 0) {
        if (touched.count(p2) > 0) {
          assert(aliasing_possible);
          dstref[p2] = Maybe<FT>();
        } else {
          dstref[p2] = srcref.at(pit.p);
          touched.insert(p2);
        }
      } else
        assert(oor_possible);  // make sure we didn't lie to Realm
      pit.step();
    }
    it.step();
  }

  return Event::merge_events(events);
}

template <int N, typename T>
template <typename FT, typename SRC>
Event DistributedData<N, T>::gather(IndexSpace<N, T> is, FieldID ptr_id,
                                    DistributedData<N, T> &ind, const SRC &src,
                                    FieldID src_id, FieldID dst_id,
                                    bool oor_possible, bool aliasing_possible,
                                    CustomSerdezID serdez_id, Event wait_on,
                                    Processor p, TransposeExperiment<N> *exp) {
  std::vector<Event> events;
  for (typename std::vector<Piece>::const_iterator it = pieces.begin();
       it != pieces.end(); ++it) {
    IndexSpace<N, T> isect;
    IndexSpace<N, T>::compute_intersection(is, it->space, isect,
                                           ProfilingRequestSet())
        .wait();

    typename CopyIndirection<N, T>::template Unstructured<SRC::_N,
                                                          typename SRC::_T>
        indirect;
    indirect.field_id = ptr_id;
    indirect.inst = ind.pieces[0].inst;
    indirect.is_ranges = false;
    indirect.subfield_offset = 0;
    indirect.oor_possible = oor_possible;
    indirect.aliasing_possible = aliasing_possible;
    indirect.next_indirection = nullptr;

    std::vector<CopySrcDstField> srcs, dsts;
    srcs.resize(1);
    dsts.resize(1);
    srcs[0].set_indirect(0, src_id, sizeof(FT));
    dsts[0].set_field(it->inst, dst_id, sizeof(FT));
    if (serdez_id != 0) {
      srcs[0].set_serdez(serdez_id);
      dsts[0].set_serdez(serdez_id);
    }

    for (typename std::vector<typename SRC::Piece>::const_iterator it2 =
             src.pieces.begin();
         it2 != src.pieces.end(); ++it2) {
      indirect.spaces.push_back(it2->space);
      indirect.insts.push_back(it2->inst);
    }

    exp->nanoseconds = 0;

    UserEvent done = UserEvent::create_user_event();
    events.push_back(done);

    CopyProfResult cpr;
    cpr.nanoseconds = &(exp->nanoseconds);
    cpr.done = done;

    ProfilingRequestSet prs;
    prs.add_request(p, COPYPROF_TASK, &cpr, sizeof(CopyProfResult))
        .add_measurement<ProfilingMeasurements::OperationTimeline>();

    Event e = is.copy(
        srcs, dsts,
        std::vector<const typename CopyIndirection<N, T>::Base *>(1, &indirect),
        prs, wait_on);
    e.wait();

    dump_instance<N, T, FT>(it->inst, FID_DATA1, is, is.bounds.hi[0]);

    dump_instance<N, T, Point<SRC::_N, typename SRC::_T>>(
        ind.pieces[0].inst, ptr_id, is, is.bounds.hi[0]);

    dump_instance<SRC::_N, typename SRC::_T, FT>(
        indirect.insts[0], FID_DATA1, indirect.spaces[0], is.bounds.hi[0]);
  }

  // update reference data
  const std::map<Point<N, T>, Maybe<Point<SRC::_N, typename SRC::_T>>> &ptrref =
      ind.get_ref_data<Point<SRC::_N, typename SRC::_T>>(ptr_id);

  const std::map<Point<SRC::_N, typename SRC::_T>, Maybe<FT>> &srcref =
      src.template get_ref_data<FT>(src_id);

  std::map<Point<N, T>, Maybe<FT>> &dstref = get_ref_data<FT>(dst_id);

  IndexSpaceIterator<N, T> it(is);
  while (it.valid) {
    PointInRectIterator<N, T> pit(it.rect);
    while (pit.valid) {
      Point<SRC::_N, typename SRC::_T> p2 = ptrref.at(pit.p).get_value();
      if (srcref.count(p2) > 0) {
        dstref[pit.p] = srcref.at(p2);
      } else
        assert(oor_possible);
      pit.step();
    }
    it.step();
  }

  return Event::merge_events(events);
}

template <int N, typename T>
template <typename FT>
bool DistributedData<N, T>::verify(IndexSpace<N, T> is, FieldID fid,
                                   Event wait_on) {
  if (!TestConfig::verify) return true;
  wait_on.wait();

  const std::map<Point<N, T>, Maybe<FT>> &ref = get_ref_data<FT>(fid);

  int errors = 0;
  for (typename std::vector<Piece>::iterator it = pieces.begin();
       it != pieces.end(); ++it) {
    IndexSpace<N, T> isect;
    IndexSpace<N, T>::compute_intersection(is, it->space, isect,
                                           ProfilingRequestSet())
        .wait();

    AffineAccessor<FT, N, T> acc;
    RegionInstance tmp_inst = RegionInstance::NO_INST;
    if (Machine::get_machine().has_affinity(
            Processor::get_executing_processor(), it->inst.get_location())) {
      // good, access this instance directly
      acc.reset(it->inst, fid);
    } else {
      // need to make a temporary instance in a memory we can access
      Memory m = Machine::MemoryQuery(Machine::get_machine())
                     .has_affinity_to(Processor::get_executing_processor())
                     .has_capacity(1)
                     .first();  // TODO: best!
      assert(m.exists());
      std::map<FieldID, size_t> tmp_fields;
      tmp_fields[fid] = sizeof(FT);
      // TODO(apryakhin): Make sure the dimensions are also inverted if
      // the original instance is inverted.
      RegionInstance::create_instance(tmp_inst, m, isect, tmp_fields, 0 /*SOA*/,
                                      ProfilingRequestSet())
          .wait();
      std::vector<CopySrcDstField> srcs, dsts;
      srcs.resize(1);
      dsts.resize(1);
      srcs[0].set_field(it->inst, fid, sizeof(FT));
      dsts[0].set_field(tmp_inst, fid, sizeof(FT));
      isect.copy(srcs, dsts, ProfilingRequestSet()).wait();
      acc.reset(tmp_inst, fid);
    }

    IndexSpaceIterator<N, T> iit(isect);
    while (iit.valid) {
      PointInRectIterator<N, T> pit(iit.rect);
      while (pit.valid) {
        Maybe<FT> exp = ref.at(pit.p);
        FT act = acc[pit.p];
        if (exp.has_value()) {
          if (exp.get_value() == act) {
            // good
            log_app.debug() << "  match at [" << pit.p
                            << "]: exp=" << exp.get_value() << " act=" << act;
          } else {
            if (errors++ < 10)
              log_app.error() << "  mismatch at [" << pit.p
                              << "]: exp=" << exp.get_value() << " act=" << act;
          }
        } else {
          log_app.debug() << "  cannot check at [" << pit.p
                          << "]: exp=??? act=" << act;
        }
        pit.step();
      }
      iit.step();
    }

    if (tmp_inst.exists()) tmp_inst.destroy();
  }

  return (errors == 0);
}

template <int N, typename T>
class RoundRobinPicker {
 public:
  RoundRobinPicker(const std::vector<Memory> &_memories, bool _reverse = false)
      : memories(_memories), reverse(_reverse) {}
  Memory operator()(size_t i, IndexSpace<N, T> is) {
    if (reverse)
      return memories[memories.size() - 1 - (i % memories.size())];
    else {
      return memories[i % memories.size()];
    }
  }

 protected:
  const std::vector<Memory> &memories;
  bool reverse;
};

template <int N, typename T, typename DT>
void dump_field(RegionInstance inst, FieldID fid, IndexSpace<N, T> is) {
  AffineAccessor<DT, N, T> acc(inst, fid);
  for (IndexSpaceIterator<N, T> it(is); it.valid; it.step())
    for (PointInRectIterator<N, T> it2(it.rect); it2.valid; it2.step()) {
      DT v = acc[it2.p];
      std::cout << it2.p << ": " << v << "\n";
    }
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
std::ostream &operator<<(std::ostream &os, const Pad<T, BYTES> &pad) {
  os << pad.val;
  return os;
}

template <int N, typename T, int N2, typename T2, typename DT>
bool scatter_gather_test(const std::vector<Memory> &sys_mems,
                         const std::vector<Memory> &gpu_mems, int pieces1,
                         int pieces2, Processor p, CustomSerdezID serdez_id = 0,
                         bool inverse = true) {
  Rect<N, T> r1;
  Rect<N2, T2> r2;
  for (int i = 0; i < N; i++) r1.lo[i] = 0;
  for (int i = 0; i < N; i++) r1.hi[i] = TestConfig::sizes1[i] - 1;
  for (int i = 0; i < N2; i++) r2.lo[i] = 0;
  for (int i = 0; i < N2; i++) {
    r2.hi[i] = TestConfig::sizes2[i] - 1;
  }

  log_app.info() << "Run testcase for N=" << N << " N2=" << N2
                 << " src_bounds=" << r1 << " dst_bounds=" << r2;

  IndexSpace<N, T> is1(r1);
  IndexSpace<N2, T2> is2(r2);

  FieldMap fields_ind;
  fields_ind[FID_PTR1] = sizeof(Point<N2, T2>);

  FieldMap fields1;
  // fields1[FID_PTR1] = sizeof(Point<N2, T2>);
  fields1[FID_DATA1] = sizeof(DT);
  // fields1[FID_DATA2] = sizeof(DT);

  std::map<FieldID, size_t> fields2;
  fields2[FID_DATA1] = sizeof(DT);
  // fields2[FID_DATA2] = sizeof(DT);

  DistributedData<N, T> region1;
  region1.add_subspaces(is1, pieces1);
  region1
      .create_instances(fields1, RoundRobinPicker<N, T>(gpu_mems),
                        /*offset=*/TestConfig::remote_gather ? 0 : 1)
      .wait();

  DistributedData<N, T> region_ind;
  region_ind.add_subspaces(is1, pieces1);
  region_ind
      .create_instances(
          fields_ind,
          RoundRobinPicker<N, T>(TestConfig::ind_on_gpu ? gpu_mems : sys_mems),
          /*offset=*/TestConfig::remote_gather ? 1 : 0)
      .wait();

  DistributedData<N2, T2> region2;
  region2.add_subspaces(is2, pieces2, /*num_subrects=*/1);
  region2
      .create_instances(fields2, RoundRobinPicker<N2, T2>(gpu_mems),
                        /*offset=*/TestConfig::remote_gather ? 1 : 0, inverse)
      .wait();

  Matrix<N2, N, T> transform;
  for (int i = 0; i < N2; i++) {
    for (int j = 0; j < N; j++) {
      transform[i][j] = (i == j ? 1 : 0);
    }
  }
  region_ind
      .template fill<Point<N2, T2>>(
          is1, FID_PTR1,
          [=](Point<N, T> p) -> Point<N2, T2> { return transform * p; },
          Event::NO_EVENT)
      .wait();

  region1
      .template fill<DT>(
          is1, FID_DATA1, [&](Point<N, T> p) -> DT { return DT(p.x); },
          Event::NO_EVENT)
      .wait();

  region2
      .template fill<DT>(
          is2, FID_DATA1, [](Point<N2, T2> p) -> DT { return DT(300 + p.x); },
          Event::NO_EVENT)
      .wait();

  TransposeExperiment<N> *exp = new TransposeExperiment<N>;

  if (TestConfig::do_scatter && !TestConfig::do_direct) {
    region1
        .template scatter<DT>(is1, FID_PTR1, region_ind, region2, FID_DATA1,
                              FID_DATA1, false /*!oor_possible*/,
                              true /*aliasing_possible*/, serdez_id,
                              Event::NO_EVENT, p, exp)
        .wait();

    if (!region2.template verify<DT>(is2, FID_DATA1, Event::NO_EVENT))
      return false;
  }

#ifdef ENABLE_DIRECT_TEST
  if (TestConfig::do_scatter && TestConfig::do_direct) {
    region1
        .template direct_scatter<DT>(
            is1, FID_PTR1, region_ind, region2, FID_DATA1, FID_DATA1,
            false /*!oor_possible*/, true /*aliasing_possible*/, serdez_id,
            Event::NO_EVENT)
        .wait();

    if (!region2.template verify<DT>(is2, FID_DATA1, Event::NO_EVENT))
      return false;
  }
#endif

  if (!TestConfig::do_scatter && !TestConfig::do_direct) {
    region1
        .template gather<DT>(is1, FID_PTR1, region_ind, region2, FID_DATA1,
                             FID_DATA1, false /*!oor_possible*/,
                             true /*aliasing_possible*/, serdez_id,
                             Event::NO_EVENT, p, exp)
        .wait();

    if (!region1.template verify<DT>(is1, FID_DATA1, Event::NO_EVENT))
      return false;

    log_app.print() << "Exp Time=" << exp->nanoseconds;
  }

  log_app.print() << "Time=" << exp->nanoseconds / (1000000.0);

  usleep(100);
  region1.destroy_instances(Event::NO_EVENT);
  region_ind.destroy_instances(Event::NO_EVENT);
  region2.destroy_instances(Event::NO_EVENT);

  return true;
}

std::set<Processor::Kind> supported_proc_kinds;

void top_level_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p) {
  log_app.print() << "Realm scatter/gather test";

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

  std::vector<Memory> sys_mems;
  Machine::MemoryQuery mq(Machine::get_machine());
  mq.only_kind(Memory::SYSTEM_MEM).has_capacity(1);
  sys_mems.assign(mq.begin(), mq.end());
  assert(!sys_mems.empty());

  bool ok = true;

  if (!scatter_gather_test<1, long long, 1, long long, int>(
          sys_mems, gpu_mems, TestConfig::pieces1, TestConfig::pieces2, p)) {
    ok = false;
  }

  if (!scatter_gather_test<1, long long, 2, long long, int>(
          sys_mems, gpu_mems, TestConfig::pieces1, TestConfig::pieces2, p)) {
    ok = false;
  }

  if (!scatter_gather_test<2, long long, 2, long long, int>(
          sys_mems, gpu_mems, TestConfig::pieces1, TestConfig::pieces2, p)) {
    ok = false;
  }

  if (!scatter_gather_test<1, long long, 3, long long, int>(
          sys_mems, gpu_mems, TestConfig::pieces1, TestConfig::pieces2, p)) {
    ok = false;
  }

  if (!scatter_gather_test<3, long long, 3, long long, int>(
          sys_mems, gpu_mems, TestConfig::pieces1, TestConfig::pieces2, p)) {
    ok = false;
  }

  typedef Pad<float, 70> BigFloat;
  if (!scatter_gather_test<3, long long, 3, long long, BigFloat>(
          sys_mems, gpu_mems, TestConfig::pieces1, TestConfig::pieces2, p)) {
    ok = false;
  }

  log_app.info() << "Scatter/gather test finished "
                 << (ok ? "successfully" : "with errors!");

  usleep(100000);
  assert(ok);

  Runtime::get_runtime().shutdown(Processor::get_current_finish_event(),
                                  ok ? 0 : 1);
}

int main(int argc, char **argv) {
  Runtime rt;

  rt.init(&argc, &argv);

  for (size_t i = 0; i < MAX_TEST_DIM; i++) {
    TestConfig::sizes1[i] = 10;
    TestConfig::sizes2[i] = TestConfig::sizes1[i];  //* 2;
  }

  CommandLineParser cp;
  // TODO(apryakhin@): Depreate some size options.
  // cp.add_option_int("-s1", TestConfig::size1);
  // cp.add_option_int("-s2", TestConfig::size2);
  cp.add_option_int("-s1x", TestConfig::sizes1[0]);
  cp.add_option_int("-s1y", TestConfig::sizes1[1]);
  cp.add_option_int("-s1z", TestConfig::sizes1[2]);
  cp.add_option_int("-s2x", TestConfig::sizes2[0]);
  cp.add_option_int("-s2y", TestConfig::sizes2[1]);
  cp.add_option_int("-s2z", TestConfig::sizes2[2]);
  cp.add_option_int("-p1", TestConfig::pieces1);
  cp.add_option_int("-p2", TestConfig::pieces2);
  cp.add_option_bool("-skipfirst", TestConfig::skipfirst);
  cp.add_option_bool("-split", TestConfig::splitcopies);
  cp.add_option_int("-scatter", TestConfig::do_scatter);
  cp.add_option_int("-verbose", TestConfig::verbose);
  cp.add_option_int("-verify", TestConfig::verify);
  cp.add_option_int("-direct", TestConfig::do_direct);
  cp.add_option_int("-ind_on_gpu", TestConfig::ind_on_gpu);
  cp.add_option_int("-remote_gather", TestConfig::remote_gather);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  Processor::register_task_by_kind(
      Processor::LOC_PROC, false /*!global*/, COPYPROF_TASK,
      CodeDescriptor(copy_profiling_task), ProfilingRequestSet(), 0, 0)
      .wait();

  rt.register_custom_serdez<WrappingSerdez<float>>(SERDEZ_WRAP_FLOAT);

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // shutdown will be requested by main task

  // now sleep this thread until that shutdown actually happens
  int result = rt.wait_for_shutdown();

  return result;
}
