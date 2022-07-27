/* Copyright 2022 Stanford University
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

#include "realm_stencil.h"

#include <algorithm>
#include <climits>

#include "realm.h"

#include "cpu_kernels.h"

using namespace Realm;

enum {
  TOP_LEVEL_TASK          = Processor::TASK_ID_FIRST_AVAILABLE+0,
  CREATE_REGION_TASK      = Processor::TASK_ID_FIRST_AVAILABLE+1,
  CREATE_REGION_DONE_TASK = Processor::TASK_ID_FIRST_AVAILABLE+2,
  SHARD_TASK              = Processor::TASK_ID_FIRST_AVAILABLE+3,
  STENCIL_TASK            = Processor::TASK_ID_FIRST_AVAILABLE+4,
  INCREMENT_TASK          = Processor::TASK_ID_FIRST_AVAILABLE+5,
  CHECK_TASK              = Processor::TASK_ID_FIRST_AVAILABLE+6,
};

enum {
  REDOP_MIN = 11,
  REDOP_MAX = 12,
};

enum {
  FID_INPUT = 101,
  FID_OUTPUT = 102,
};

template <typename K, typename V>
class DefaultMap : public std::map<K, V> {
public:
  DefaultMap(const V &v) : std::map<K, V>(), def(v) {}
  V &operator[](const K &k) {
    if (std::map<K, V>::count(k)) {
      return std::map<K, V>::operator[](k);
    } else {
      V &result = std::map<K, V>::operator[](k);
      result = def;
      return result;
    }
  }
  const V &operator[](const K &k) const {
    if (std::map<K, V>::count(k)) {
      return std::map<K, V>::operator[](k);
    } else {
      return def;
    }
  }
private:
  const V def;
};

struct AppConfig {
public:
  AppConfig()
    : nx(12), ny(12), ntx(2), nty(2), tsteps(20), tprune(5), init(1000) {}
public:
  coord_t nx;
  coord_t ny;
  coord_t ntx;
  coord_t nty;
  coord_t tsteps;
  coord_t tprune;
  coord_t init;
};

void get_optional_arg(int argc, char **argv,
                      const char *flag, coord_t &out)
{
  for (int i = 0; i < argc; ++i) {
    if (strcmp(argv[i], flag) == 0) {
      if (i+1 < argc) {
        out = atoll(argv[i+1]);
      }
      return;
    } else if (argv[i][0] == '-') {
      i++;
    }
  }
}

AppConfig parse_config(int argc, char **argv)
{
  AppConfig config;
  get_optional_arg(argc, argv, "-nx", config.nx);
  get_optional_arg(argc, argv, "-ny", config.ny);
  get_optional_arg(argc, argv, "-ntx", config.ntx);
  get_optional_arg(argc, argv, "-nty", config.nty);
  get_optional_arg(argc, argv, "-tsteps", config.tsteps);
  get_optional_arg(argc, argv, "-tprune", config.tprune);
  get_optional_arg(argc, argv, "-init", config.init);
  return config;
}

#ifdef _MSC_VER
#define __sync_bool_compare_and_swap(ptr,old,new) _InterlockedCompareExchange64(ptr,new,old)
#endif

#define DECLARE_REDUCTION(CLASS, T, U, APPLY_OP, FOLD_OP, ID) \
  class CLASS {                                                         \
  public:                                                               \
  typedef T LHS, RHS;                                                   \
  template <bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);       \
  template <bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);      \
  static const T identity;                                              \
  };                                                                    \
                                                                        \
  const T CLASS::identity = ID;                                         \
                                                                        \
  template <>                                                           \
  void CLASS::apply<true>(LHS &lhs, RHS rhs)                            \
  {                                                                     \
    lhs = APPLY_OP(lhs, rhs);                                           \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::apply<false>(LHS &lhs, RHS rhs)                           \
  {                                                                     \
    volatile U *target = (U *)&(lhs);                                   \
    union { U as_U; T as_T; } oldval, newval;                           \
    do {                                                                \
      oldval.as_U = *target;                                            \
      newval.as_T = APPLY_OP(oldval.as_T, rhs);                         \
    } while(!__sync_bool_compare_and_swap(target, oldval.as_U, newval.as_U)); \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::fold<true>(RHS &rhs1, RHS rhs2)                           \
  {                                                                     \
    rhs1 = FOLD_OP(rhs1, rhs2);                                         \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::fold<false>(RHS &rhs1, RHS rhs2)                          \
  {                                                                     \
    volatile U *target = (U *)&rhs1;                                    \
    union { U as_U; T as_T; } oldval, newval;                           \
    do {                                                                \
      oldval.as_U = *target;                                            \
      newval.as_T = FOLD_OP(oldval.as_T, rhs2);                         \
    } while(!__sync_bool_compare_and_swap(target, oldval.as_U, newval.as_U)); \
  }

DECLARE_REDUCTION(RedopMin, long long, long long, std::min, std::min, LLONG_MAX)
DECLARE_REDUCTION(RedopMax, long long, long long, std::max, std::max, LLONG_MIN)

#undef DECLARE_REDUCTION

Event fill(RegionInstance inst, FieldID fid, DTYPE value)
{
  CopySrcDstField field;
  field.inst = inst;
  field.field_id = fid;
  field.size = sizeof(DTYPE);

  std::vector<CopySrcDstField> fields;
  fields.push_back(field);

  return inst.get_indexspace<2, coord_t>().fill(fields, ProfilingRequestSet(),
                                                &value, sizeof(value));
}

Event copy(RegionInstance src_inst, RegionInstance dst_inst, FieldID fid,
           Event wait_for)
{
  CopySrcDstField src_field;
  src_field.inst = src_inst;
  src_field.field_id = fid;
  src_field.size = sizeof(DTYPE);

  std::vector<CopySrcDstField> src_fields;
  src_fields.push_back(src_field);

  CopySrcDstField dst_field;
  dst_field.inst = dst_inst;
  dst_field.field_id = fid;
  dst_field.size = sizeof(DTYPE);

  std::vector<CopySrcDstField> dst_fields;
  dst_fields.push_back(dst_field);

  return dst_inst.get_indexspace<2, coord_t>().copy(src_fields, dst_fields,
                                                    ProfilingRequestSet(),
                                                    wait_for);
}

void get_base_and_stride(RegionInstance inst, FieldID fid, DTYPE *&base, size_t &stride)
{
  AffineAccessor<DTYPE, 2, coord_t> acc = AffineAccessor<DTYPE, 2, coord_t>(inst, fid);
  base = reinterpret_cast<DTYPE *>(acc.ptr(inst.get_indexspace<2, coord_t>().bounds.lo));
  assert(acc.strides.x == sizeof(DTYPE));
  stride = acc.strides.y;
}

void dump(RegionInstance inst, FieldID fid, Rect2 bounds, const char *prefix)
{
  AffineAccessor<DTYPE, 2, coord_t> acc = AffineAccessor<DTYPE, 2, coord_t>(inst, fid);
  for (PointInRectIterator<2, coord_t> it(bounds); it.valid; it.step()) {
    printf("%s: %2lld %2lld value %8.3f\n", prefix, it.p.x, it.p.y, acc.read(it.p));
  }
}

void inline_copy(RegionInstance src_inst, RegionInstance dst_inst, FieldID fid,
                 Rect2 bounds)
{
  AffineAccessor<DTYPE, 2, coord_t> src_acc = AffineAccessor<DTYPE, 2, coord_t>(src_inst, fid);
  AffineAccessor<DTYPE, 2, coord_t> dst_acc = AffineAccessor<DTYPE, 2, coord_t>(dst_inst, fid);
  for (PointInRectIterator<2, coord_t> it(bounds); it.valid; it.step()) {
    dst_acc.write(it.p, src_acc.read(it.p));
  }
}

void inline_copy_raw(RegionInstance src_inst, RegionInstance dst_inst,
                     FieldID fid, Rect2 bounds)
{
  // FIXME: Something is still wrong in the index arithmetic here

  AffineAccessor<DTYPE, 2, coord_t> src_acc = AffineAccessor<DTYPE, 2, coord_t>(src_inst, fid);
  DTYPE *src_base;
  size_t src_stride;
  get_base_and_stride(src_inst, fid, src_base, src_stride);

  AffineAccessor<DTYPE, 2, coord_t> dst_acc = AffineAccessor<DTYPE, 2, coord_t>(dst_inst, fid);
  DTYPE *dst_base;
  size_t dst_stride;
  get_base_and_stride(dst_inst, fid, dst_base, dst_stride);

  Rect2 src_bounds = src_inst.get_indexspace<2, coord_t>().bounds;
  Point2 src_offset = bounds.lo - src_bounds.lo;

  Rect2 dst_bounds = dst_inst.get_indexspace<2, coord_t>().bounds;
  Point2 dst_offset = bounds.lo - dst_bounds.lo;

  Point2 size = bounds.hi - bounds.lo + Point2(1, 1);

  copy2D(src_base, dst_base,
         src_stride/sizeof(DTYPE),
         src_offset.x, src_offset.x + size.x,
         src_offset.y, src_offset.y + size.y,
         dst_stride/sizeof(DTYPE),
         dst_offset.x, dst_offset.y);
}

void stencil_task(const void *args, size_t arglen,
                  const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(StencilArgs));
  const StencilArgs &a = *reinterpret_cast<const StencilArgs *>(args);

  DTYPE *private_base_input, *private_base_output;
  size_t private_stride_input, private_stride_output;
  get_base_and_stride(a.private_inst, FID_INPUT, private_base_input, private_stride_input);
  get_base_and_stride(a.private_inst, FID_OUTPUT, private_base_output, private_stride_output);
  assert(private_stride_input == private_stride_output);

  Rect2 private_bounds = a.private_inst.get_indexspace<2, coord_t>().bounds;
  Point2 interior_offset = a.interior_bounds.lo - private_bounds.lo;
  Point2 interior_size = a.interior_bounds.hi - a.interior_bounds.lo + Point2(1, 1);

  if (a.xp_inst.exists())
    inline_copy(a.xp_inst, a.private_inst, FID_INPUT,
                a.xp_inst.get_indexspace<2, coord_t>().bounds);
  if (a.xm_inst.exists())
    inline_copy(a.xm_inst, a.private_inst, FID_INPUT,
                a.xm_inst.get_indexspace<2, coord_t>().bounds);
  if (a.yp_inst.exists())
    inline_copy(a.yp_inst, a.private_inst, FID_INPUT,
                a.yp_inst.get_indexspace<2, coord_t>().bounds);
  if (a.ym_inst.exists())
    inline_copy(a.ym_inst, a.private_inst, FID_INPUT,
                a.ym_inst.get_indexspace<2, coord_t>().bounds);

  DTYPE *weights = a.weights;

  stencil(private_base_input, private_base_output, weights,
          private_stride_input/sizeof(DTYPE),
          interior_offset.x,
          interior_offset.x + interior_size.x,
          interior_offset.y,
          interior_offset.y + interior_size.y);
}

void increment_task(const void *args, size_t arglen,
                    const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(IncrementArgs));
  const IncrementArgs &a = *reinterpret_cast<const IncrementArgs *>(args);

  DTYPE *private_base_input;
  size_t private_stride_input;
  get_base_and_stride(a.private_inst, FID_INPUT, private_base_input, private_stride_input);

  Rect2 private_bounds = a.private_inst.get_indexspace<2, coord_t>().bounds;
  Point2 outer_offset = a.outer_bounds.lo - private_bounds.lo;
  Point2 outer_size = a.outer_bounds.hi - a.outer_bounds.lo + Point2(1, 1);

  increment(private_base_input,
            private_stride_input/sizeof(DTYPE),
            outer_offset.x,
            outer_offset.x + outer_size.x,
            outer_offset.y,
            outer_offset.y + outer_size.y);

  if (a.xp_inst.exists())
    inline_copy(a.private_inst, a.xp_inst, FID_INPUT,
                a.xp_inst.get_indexspace<2, coord_t>().bounds);
  if (a.xm_inst.exists())
    inline_copy(a.private_inst, a.xm_inst, FID_INPUT,
                a.xm_inst.get_indexspace<2, coord_t>().bounds);
  if (a.yp_inst.exists())
    inline_copy(a.private_inst, a.yp_inst, FID_INPUT,
                a.yp_inst.get_indexspace<2, coord_t>().bounds);
  if (a.ym_inst.exists())
    inline_copy(a.private_inst, a.ym_inst, FID_INPUT,
                a.ym_inst.get_indexspace<2, coord_t>().bounds);
}

void check_task(const void *args, size_t arglen,
                const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(CheckArgs));
  const CheckArgs &a = *reinterpret_cast<const CheckArgs *>(args);

  DTYPE expect_input = a.init + a.tsteps;
  DTYPE expect_output = a.init;

  // Check input
  {
    AffineAccessor<DTYPE, 2, coord_t> acc = AffineAccessor<DTYPE, 2, coord_t>(a.private_inst, FID_INPUT);
    for (PointInRectIterator<2, coord_t> it(a.interior_bounds); it.valid; it.step()) {
      if (acc.read(it.p) != expect_input) {
        printf("bad value: got %f expected %f\n", acc.read(it.p), expect_input);
        assert(false);
        abort(); // if someone compiles with NDEBUG make sure this fails anyway
      }
    }
  }

  // Check output
  {
    AffineAccessor<DTYPE, 2, coord_t> acc = AffineAccessor<DTYPE, 2, coord_t>(a.private_inst, FID_OUTPUT);
    for (PointInRectIterator<2, coord_t> it(a.interior_bounds); it.valid; it.step()) {
      if (acc.read(it.p) != expect_output) {
        printf("bad value: got %f expected %f\n", acc.read(it.p), expect_output);
        assert(false);
        abort(); // if someone compiles with NDEBUG make sure this fails anyway
      }
    }
  }
}

void create_region_task(const void *args, size_t arglen,
                        const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(CreateRegionArgs));
  const CreateRegionArgs &a = *reinterpret_cast<const CreateRegionArgs *>(args);

  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID_INPUT] = sizeof(DTYPE);
  field_sizes[FID_OUTPUT] = sizeof(DTYPE);

  RegionInstance inst = RegionInstance::NO_INST;
  RegionInstance::create_instance(inst, a.memory,
                                  a.bounds, field_sizes,
                                  0 /*SOA*/, ProfilingRequestSet()).wait();

  // Send the instance back to the requesting node
  // Important: Don't return until this is complete
  CreateRegionDoneArgs done;
  done.inst = inst;
  done.dest_proc = a.dest_proc;
  done.dest_inst = a.dest_inst;
  a.dest_proc.spawn(CREATE_REGION_DONE_TASK, &done, sizeof(done)).wait();
}

void create_region_done_task(const void *args, size_t arglen,
                             const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(CreateRegionDoneArgs));
  const CreateRegionDoneArgs &a = *reinterpret_cast<const CreateRegionDoneArgs *>(args);

  // We had better be on the destination proc, otherwise these
  // pointer won't be valid.
  assert(a.dest_proc == p);

  *a.dest_inst = a.inst;
}

void shard_task(const void *args, size_t arglen,
                const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(ShardArgs));
  const ShardArgs &a = *reinterpret_cast<const ShardArgs *>(args);

  // Initialize
  RegionInstance private_inst = RegionInstance::NO_INST;
  RegionInstance xp_inst_out_local = RegionInstance::NO_INST;
  RegionInstance xm_inst_out_local = RegionInstance::NO_INST;
  RegionInstance yp_inst_out_local = RegionInstance::NO_INST;
  RegionInstance ym_inst_out_local = RegionInstance::NO_INST;
  {
    std::map<FieldID, size_t> field_sizes;
    field_sizes[FID_INPUT] = sizeof(DTYPE);
    field_sizes[FID_OUTPUT] = sizeof(DTYPE);

    std::vector<Event> events;
    events.push_back(
      RegionInstance::create_instance(private_inst, a.sysmem,
                                      a.exterior_bounds, field_sizes,
                                      0 /*SOA*/, ProfilingRequestSet()));
    if (a.xp_inst_out.exists()) {
      Event e = a.xp_inst_out.fetch_metadata(p);
      e.wait();
      events.push_back(
        RegionInstance::create_instance(xp_inst_out_local, a.regmem,
                                        a.xp_inst_out.get_indexspace<2, coord_t>(), field_sizes,
                                        0 /*SOA*/, ProfilingRequestSet()));
    }
    if (a.xm_inst_out.exists()) {
      Event e = a.xm_inst_out.fetch_metadata(p);
      e.wait();
      events.push_back(
        RegionInstance::create_instance(xm_inst_out_local, a.regmem,
                                        a.xm_inst_out.get_indexspace<2, coord_t>(), field_sizes,
                                        0 /*SOA*/, ProfilingRequestSet()));
    }
    if (a.yp_inst_out.exists()) {
      Event e = a.yp_inst_out.fetch_metadata(p);
      e.wait();
      events.push_back(
        RegionInstance::create_instance(yp_inst_out_local, a.regmem,
                                        a.yp_inst_out.get_indexspace<2, coord_t>(), field_sizes,
                                        0 /*SOA*/, ProfilingRequestSet()));
    }
    if (a.ym_inst_out.exists()) {
      Event e = a.ym_inst_out.fetch_metadata(p);
      e.wait();
      events.push_back(
        RegionInstance::create_instance(ym_inst_out_local, a.regmem,
                                        a.ym_inst_out.get_indexspace<2, coord_t>(), field_sizes,
                                        0 /*SOA*/, ProfilingRequestSet()));
    }
    Event::merge_events(events).wait();
  }

  {
    std::vector<Event> events;
    events.push_back(fill(private_inst, FID_INPUT,  a.init));
    events.push_back(fill(private_inst, FID_OUTPUT, a.init));
    if (a.xp_inst_in.exists()) events.push_back(fill(a.xp_inst_in, FID_INPUT,  a.init));
    if (a.xp_inst_in.exists()) events.push_back(fill(a.xp_inst_in, FID_OUTPUT, a.init));
    if (a.xm_inst_in.exists()) events.push_back(fill(a.xm_inst_in, FID_INPUT,  a.init));
    if (a.xm_inst_in.exists()) events.push_back(fill(a.xm_inst_in, FID_OUTPUT, a.init));
    if (a.yp_inst_in.exists()) events.push_back(fill(a.yp_inst_in, FID_INPUT,  a.init));
    if (a.yp_inst_in.exists()) events.push_back(fill(a.yp_inst_in, FID_OUTPUT, a.init));
    if (a.ym_inst_in.exists()) events.push_back(fill(a.ym_inst_in, FID_INPUT,  a.init));
    if (a.ym_inst_in.exists()) events.push_back(fill(a.ym_inst_in, FID_OUTPUT, a.init));
    Event::merge_events(events).wait();
  }

  // Init the weights
  size_t weights_size = (2*RADIUS + 1) * (2*RADIUS + 1);
  DTYPE *weights = (DTYPE*)malloc(sizeof(DTYPE) * weights_size);
  memset(weights, 0, sizeof(DTYPE) * weights_size);

#define WEIGHT(i, j) weights[(j + RADIUS) * (2 * RADIUS + 1) + (i + RADIUS)]
  for (coord_t i = 1; i <= RADIUS; i++) {
    WEIGHT( 0,  i) =  1.0/(2.0*i*RADIUS);
    WEIGHT( i,  0) =  1.0/(2.0*i*RADIUS);
    WEIGHT( 0, -i) = -1.0/(2.0*i*RADIUS);
    WEIGHT(-i,  0) = -1.0/(2.0*i*RADIUS);
  }
#undef WEIGHT

  // Barrier
  // Warning: If you're used to Legion barriers, please note that
  // Realm barriers DON'T WORK THE SAME WAY.
  Barrier sync = a.sync;
  sync.arrive(1);
  sync.wait();
  sync = sync.advance_barrier();

  // These are going to change, so give them mutable names
  Barrier xp_empty_in = a.xp_empty_in;
  Barrier xm_empty_in = a.xm_empty_in;
  Barrier yp_empty_in = a.yp_empty_in;
  Barrier ym_empty_in = a.ym_empty_in;

  Barrier xp_empty_out = a.xp_empty_out;
  Barrier xm_empty_out = a.xm_empty_out;
  Barrier yp_empty_out = a.yp_empty_out;
  Barrier ym_empty_out = a.ym_empty_out;

  Barrier xp_full_in = a.xp_full_in;
  Barrier xm_full_in = a.xm_full_in;
  Barrier yp_full_in = a.yp_full_in;
  Barrier ym_full_in = a.ym_full_in;

  Barrier xp_full_out = a.xp_full_out;
  Barrier xm_full_out = a.xm_full_out;
  Barrier yp_full_out = a.yp_full_out;
  Barrier ym_full_out = a.ym_full_out;

  // Main time step loop
  Event stencil_done = Event::NO_EVENT;
  Event increment_done = Event::NO_EVENT;
  Event xp_copy_done = Event::NO_EVENT;
  Event xm_copy_done = Event::NO_EVENT;
  Event yp_copy_done = Event::NO_EVENT;
  Event ym_copy_done = Event::NO_EVENT;
  for (coord_t t = 0; t < a.tsteps; t++) {
    {
      StencilArgs args;
      args.private_inst = private_inst;
      args.xp_inst = a.xp_inst_in;
      args.xm_inst = a.xm_inst_in;
      args.yp_inst = a.yp_inst_in;
      args.ym_inst = a.ym_inst_in;
      args.interior_bounds = a.interior_bounds;
      args.weights = weights;
      Event precondition = Event::merge_events(
        increment_done,
        (xp_full_in.exists() ? xp_full_in.get_previous_phase() : Event::NO_EVENT),
        (xm_full_in.exists() ? xm_full_in.get_previous_phase() : Event::NO_EVENT),
        (yp_full_in.exists() ? yp_full_in.get_previous_phase() : Event::NO_EVENT),
        (ym_full_in.exists() ? ym_full_in.get_previous_phase() : Event::NO_EVENT));
      if (t == a.tprune) {
        precondition.wait();
      }
      stencil_done = p.spawn(STENCIL_TASK, &args, sizeof(args), precondition);
      if (xp_empty_out.exists()) xp_empty_out.arrive(1, stencil_done);
      if (xm_empty_out.exists()) xm_empty_out.arrive(1, stencil_done);
      if (yp_empty_out.exists()) yp_empty_out.arrive(1, stencil_done);
      if (ym_empty_out.exists()) ym_empty_out.arrive(1, stencil_done);
    }

    {
      IncrementArgs args;
      args.private_inst = private_inst;
      args.xp_inst = xp_inst_out_local;
      args.xm_inst = xm_inst_out_local;
      args.yp_inst = yp_inst_out_local;
      args.ym_inst = ym_inst_out_local;
      args.outer_bounds = a.outer_bounds;
      Event precondition = Event::merge_events(
        stencil_done, xp_copy_done, xm_copy_done, yp_copy_done, ym_copy_done);
      increment_done = p.spawn(INCREMENT_TASK, &args, sizeof(args), precondition);
    }

    if (a.xp_inst_out.exists()) {
      xp_copy_done = copy(xp_inst_out_local, a.xp_inst_out, FID_INPUT,
                          Event::merge_events(increment_done, xp_empty_in));
      xp_full_out.arrive(1, xp_copy_done);
    }

    if (a.xm_inst_out.exists()) {
      xm_copy_done = copy(xm_inst_out_local, a.xm_inst_out, FID_INPUT,
                          Event::merge_events(increment_done, xm_empty_in));
      xm_full_out.arrive(1, xm_copy_done);
    }

    if (a.yp_inst_out.exists()) {
      yp_copy_done = copy(yp_inst_out_local, a.yp_inst_out, FID_INPUT,
                          Event::merge_events(increment_done, yp_empty_in));
      yp_full_out.arrive(1, yp_copy_done);
    }

    if (a.ym_inst_out.exists()) {
      ym_copy_done = copy(ym_inst_out_local, a.ym_inst_out, FID_INPUT,
                          Event::merge_events(increment_done, ym_empty_in));
      ym_full_out.arrive(1, ym_copy_done);
    }

    if (xp_empty_in.exists()) xp_empty_in = xp_empty_in.advance_barrier();
    if (xm_empty_in.exists()) xm_empty_in = xm_empty_in.advance_barrier();
    if (yp_empty_in.exists()) yp_empty_in = yp_empty_in.advance_barrier();
    if (ym_empty_in.exists()) ym_empty_in = ym_empty_in.advance_barrier();

    if (xp_empty_out.exists()) xp_empty_out = xp_empty_out.advance_barrier();
    if (xm_empty_out.exists()) xm_empty_out = xm_empty_out.advance_barrier();
    if (yp_empty_out.exists()) yp_empty_out = yp_empty_out.advance_barrier();
    if (ym_empty_out.exists()) ym_empty_out = ym_empty_out.advance_barrier();

    if (xp_full_in.exists()) xp_full_in = xp_full_in.advance_barrier();
    if (xm_full_in.exists()) xm_full_in = xm_full_in.advance_barrier();
    if (yp_full_in.exists()) yp_full_in = yp_full_in.advance_barrier();
    if (ym_full_in.exists()) ym_full_in = ym_full_in.advance_barrier();

    if (xp_full_out.exists()) xp_full_out = xp_full_out.advance_barrier();
    if (xm_full_out.exists()) xm_full_out = xm_full_out.advance_barrier();
    if (yp_full_out.exists()) yp_full_out = yp_full_out.advance_barrier();
    if (ym_full_out.exists()) ym_full_out = ym_full_out.advance_barrier();
  }

  // This task hasn't blocked, so no subtasks have executed yet.
  // Only time subtask execution
  long long start = Realm::Clock::current_time_in_microseconds();
  increment_done.wait();
  long long stop = Realm::Clock::current_time_in_microseconds();

  // Send start and stop times back to top level task
  a.first_start.arrive(1, Event::NO_EVENT, &start, sizeof(start));
  a.last_start.arrive(1, Event::NO_EVENT, &start, sizeof(start));
  a.first_stop.arrive(1, Event::NO_EVENT, &stop, sizeof(stop));
  a.last_stop.arrive(1, Event::NO_EVENT, &stop, sizeof(stop));

  {
    CheckArgs check_args;
    check_args.private_inst = private_inst;
    check_args.tsteps = a.tsteps;
    check_args.init = a.init;
    check_args.interior_bounds = a.interior_bounds;
    Event check_done = p.spawn(CHECK_TASK, &check_args, sizeof(check_args), increment_done);
    check_done.wait();
  }

  // Make sure all operations are done before returning
  Event::merge_events(
    xp_copy_done, xm_copy_done, yp_copy_done, ym_copy_done).wait();

  free(weights);
}

void top_level_task(const void *args, size_t arglen,
                    const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(AppConfig));
  const AppConfig &config = *reinterpret_cast<const AppConfig *>(args);
  printf("Stencil configuration:\n");
  printf("      nx %lld\n", config.nx);
  printf("      ny %lld\n", config.ny);
  printf("     ntx %lld\n", config.ntx);
  printf("     nty %lld\n", config.nty);
  printf("  tsteps %lld\n", config.tsteps);
  printf("  tprune %lld\n", config.tprune);
  printf("    init %lld\n", config.init);

  // Discover the machine topology
  Machine machine = Machine::get_machine();

  std::vector<Processor> procs;
  {
    Machine::ProcessorQuery query(machine);
    query.only_kind(Processor::LOC_PROC);
    procs.insert(procs.end(), query.begin(), query.end());
  }

  std::map<Processor, Memory> proc_sysmems;
  std::map<Processor, Memory> proc_regmems;
  {
    std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
    machine.get_proc_mem_affinity(proc_mem_affinities);

    for (size_t i = 0; i < proc_mem_affinities.size(); ++i) {
      Machine::ProcessorMemoryAffinity& affinity = proc_mem_affinities[i];

      // skip memories with no capacity for creating instances
      if(affinity.m.capacity() == 0)
        continue;

      if (affinity.p.kind() == Processor::LOC_PROC) {
        if (affinity.m.kind() == Memory::SYSTEM_MEM) {
          proc_sysmems[affinity.p] = affinity.m;
          if (proc_regmems.find(affinity.p) == proc_regmems.end())
            proc_regmems[affinity.p] = affinity.m;
        }
        else if (affinity.m.kind() == Memory::REGDMA_MEM)
          proc_regmems[affinity.p] = affinity.m;
      }
    }
  }

  // Assign shards to processors
  coord_t nshards = config.ntx * config.nty;
  assert(procs.size() >= static_cast<size_t>(nshards)); // Expect one core per shard
  Rect2 shards(Point2(0, 0), Point2(config.ntx-1, config.nty-1));
  std::map<Point2, Processor> shard_procs;
  {
    std::vector<Processor>::iterator pit = procs.begin();
    for (PointInRectIterator<2, coord_t> it(shards); it.valid; it.step()) {
      assert(pit != procs.end());
      shard_procs[it.p] = *pit++;
    }
  }

  // Size of grid excluding the border
  coord_t nx = config.nx - 2*RADIUS;
  coord_t ny = config.ny - 2*RADIUS;
  assert(nx >= config.ntx);
  assert(ny >= config.nty);

  // Choose block sizes for each shard
  std::vector<Rect1> x_blocks;
  std::vector<Rect1> y_blocks;

  for (coord_t ix = 0; ix < config.ntx; ix++) {
    x_blocks.push_back(Rect1(ix*nx/config.ntx, (ix+1)*nx/config.ntx - 1));
  }
  for (coord_t iy = 0; iy < config.nty; iy++) {
    y_blocks.push_back(Rect1(iy*ny/config.nty, (iy+1)*ny/config.nty - 1));
  }

  // Create incoming exchange buffers
  DefaultMap<Point2, RegionInstance> xp_insts(RegionInstance::NO_INST);
  DefaultMap<Point2, RegionInstance> xm_insts(RegionInstance::NO_INST);
  DefaultMap<Point2, RegionInstance> yp_insts(RegionInstance::NO_INST);
  DefaultMap<Point2, RegionInstance> ym_insts(RegionInstance::NO_INST);

  {
    std::map<FieldID, size_t> field_sizes;
    field_sizes[FID_INPUT] = sizeof(DTYPE);
    field_sizes[FID_OUTPUT] = sizeof(DTYPE);

    

    std::vector<Event> events;
    for (PointInRectIterator<2, coord_t> it(shards); it.valid; it.step()) {
      Point2 i(it.p);
      Rect2 xp_bounds(Point2(x_blocks[i.x].hi + 1,      y_blocks[i.y].lo),
                      Point2(x_blocks[i.x].hi + RADIUS, y_blocks[i.y].hi));
      Rect2 xm_bounds(Point2(x_blocks[i.x].lo - RADIUS, y_blocks[i.y].lo),
                      Point2(x_blocks[i.x].lo - 1,      y_blocks[i.y].hi));
      Rect2 yp_bounds(Point2(x_blocks[i.x].lo,          y_blocks[i.y].hi + 1),
                      Point2(x_blocks[i.x].hi,          y_blocks[i.y].hi + RADIUS));
      Rect2 ym_bounds(Point2(x_blocks[i.x].lo,          y_blocks[i.y].lo - RADIUS),
                      Point2(x_blocks[i.x].hi,          y_blocks[i.y].lo - 1));

      Processor shard_proc(shard_procs[i]);
      Memory memory(proc_regmems[shard_proc]);

      // Region allocation has to be done on the remote node
      if (i.x != shards.hi.x) {
        CreateRegionArgs args;
        args.bounds = xp_bounds;
        args.memory = memory;
        args.dest_proc = p;
        args.dest_inst = &xp_insts[i];
        events.push_back(shard_proc.spawn(CREATE_REGION_TASK, &args, sizeof(args)));
      }

      if (i.x != shards.lo.x) {
        CreateRegionArgs args;
        args.bounds = xm_bounds;
        args.memory = memory;
        args.dest_proc = p;
        args.dest_inst = &xm_insts[i];
        events.push_back(shard_proc.spawn(CREATE_REGION_TASK, &args, sizeof(args)));
      }

      if (i.y != shards.hi.y) {
        CreateRegionArgs args;
        args.bounds = yp_bounds;
        args.memory = memory;
        args.dest_proc = p;
        args.dest_inst = &yp_insts[i];
        events.push_back(shard_proc.spawn(CREATE_REGION_TASK, &args, sizeof(args)));
      }

      if (i.y != shards.lo.y) {
        CreateRegionArgs args;
        args.bounds = ym_bounds;
        args.memory = memory;
        args.dest_proc = p;
        args.dest_inst = &ym_insts[i];
        events.push_back(shard_proc.spawn(CREATE_REGION_TASK, &args, sizeof(args)));
      }
    }
    Event::merge_events(events).wait();
  }

  // Create incoming phase barriers
  DefaultMap<Point2, Barrier> xp_bars_empty(Barrier::NO_BARRIER);
  DefaultMap<Point2, Barrier> xm_bars_empty(Barrier::NO_BARRIER);
  DefaultMap<Point2, Barrier> yp_bars_empty(Barrier::NO_BARRIER);
  DefaultMap<Point2, Barrier> ym_bars_empty(Barrier::NO_BARRIER);

  DefaultMap<Point2, Barrier> xp_bars_full(Barrier::NO_BARRIER);
  DefaultMap<Point2, Barrier> xm_bars_full(Barrier::NO_BARRIER);
  DefaultMap<Point2, Barrier> yp_bars_full(Barrier::NO_BARRIER);
  DefaultMap<Point2, Barrier> ym_bars_full(Barrier::NO_BARRIER);

  for (PointInRectIterator<2, coord_t> it(shards); it.valid; it.step()) {
    Point2 i(it.p);

    if (i.x != shards.hi.x) xp_bars_empty[i] = Barrier::create_barrier(1);
    if (i.x != shards.lo.x) xm_bars_empty[i] = Barrier::create_barrier(1);
    if (i.y != shards.hi.y) yp_bars_empty[i] = Barrier::create_barrier(1);
    if (i.y != shards.lo.y) ym_bars_empty[i] = Barrier::create_barrier(1);

    if (i.x != shards.hi.x) xp_bars_full[i] = Barrier::create_barrier(1);
    if (i.x != shards.lo.x) xm_bars_full[i] = Barrier::create_barrier(1);
    if (i.y != shards.hi.y) yp_bars_full[i] = Barrier::create_barrier(1);
    if (i.y != shards.lo.y) ym_bars_full[i] = Barrier::create_barrier(1);
  }

  // Create barrier to keep shard launch synchronized
  Barrier sync_bar = Barrier::create_barrier(nshards);
  Barrier first_start_bar = Barrier::create_barrier(nshards, REDOP_MIN, &RedopMin::identity, sizeof(RedopMin::identity));
  Barrier first_stop_bar = Barrier::create_barrier(nshards, REDOP_MIN, &RedopMin::identity, sizeof(RedopMin::identity));
  Barrier last_start_bar = Barrier::create_barrier(nshards, REDOP_MAX, &RedopMax::identity, sizeof(RedopMax::identity));
  Barrier last_stop_bar = Barrier::create_barrier(nshards, REDOP_MAX, &RedopMax::identity, sizeof(RedopMax::identity));

  // Launch shard tasks
  {
    std::vector<Event> events;
    for (PointInRectIterator<2, coord_t> it(shards); it.valid; it.step()) {
      Point2 i(it.p);

      Rect2 interior_bounds(Point2(x_blocks[i.x].lo, y_blocks[i.y].lo),
                            Point2(x_blocks[i.x].hi, y_blocks[i.y].hi));
      Rect2 exterior_bounds(Point2(x_blocks[i.x].lo - RADIUS, y_blocks[i.y].lo - RADIUS),
                            Point2(x_blocks[i.x].hi + RADIUS, y_blocks[i.y].hi + RADIUS));
      // As interior, but bloated only on the outer edges
      Rect2 outer_bounds(Point2(x_blocks[i.x].lo - (i.x == shards.lo.x ? RADIUS : 0),
                                y_blocks[i.y].lo - (i.y == shards.lo.y ? RADIUS : 0)),
                         Point2(x_blocks[i.x].hi + (i.x == shards.hi.x ? RADIUS : 0),
                                y_blocks[i.y].hi + (i.y == shards.hi.y ? RADIUS : 0)));

      // Pack arguments
      ShardArgs args;
      args.xp_inst_in = xp_insts[i];
      args.xm_inst_in = xm_insts[i];
      args.yp_inst_in = yp_insts[i];
      args.ym_inst_in = ym_insts[i];

      args.xp_inst_out = xm_insts[i + Point2( 1,  0)];
      args.xm_inst_out = xp_insts[i + Point2(-1,  0)];
      args.yp_inst_out = ym_insts[i + Point2( 0,  1)];
      args.ym_inst_out = yp_insts[i + Point2( 0, -1)];

      args.xp_empty_in = xp_bars_empty[i];
      args.xm_empty_in = xm_bars_empty[i];
      args.yp_empty_in = yp_bars_empty[i];
      args.ym_empty_in = ym_bars_empty[i];

      args.xp_empty_out = xm_bars_empty[i + Point2( 1,  0)];
      args.xm_empty_out = xp_bars_empty[i + Point2(-1,  0)];
      args.yp_empty_out = ym_bars_empty[i + Point2( 0,  1)];
      args.ym_empty_out = yp_bars_empty[i + Point2( 0, -1)];

      args.xp_full_in = xp_bars_full[i];
      args.xm_full_in = xm_bars_full[i];
      args.yp_full_in = yp_bars_full[i];
      args.ym_full_in = ym_bars_full[i];

      args.xp_full_out = xm_bars_full[i + Point2( 1,  0)];
      args.xm_full_out = xp_bars_full[i + Point2(-1,  0)];
      args.yp_full_out = ym_bars_full[i + Point2( 0,  1)];
      args.ym_full_out = yp_bars_full[i + Point2( 0, -1)];

      args.sync = sync_bar;
      args.first_start = first_start_bar;
      args.last_start = last_start_bar;
      args.first_stop = first_stop_bar;
      args.last_stop = last_stop_bar;

      args.tsteps = config.tsteps + config.tprune;
      args.tprune = config.tprune;
      args.init = config.init;

      args.point = i;
      args.interior_bounds = interior_bounds;
      args.exterior_bounds = exterior_bounds;
      args.outer_bounds = outer_bounds;

      args.sysmem = proc_sysmems[shard_procs[i]];
      args.regmem = proc_regmems[shard_procs[i]];

      // Sanity checks
      assert(exterior_bounds.contains(outer_bounds));
      assert(outer_bounds.contains(interior_bounds));

      assert(args.xp_inst_in.exists() == args.xp_inst_out.exists());
      assert(args.xm_inst_in.exists() == args.xm_inst_out.exists());
      assert(args.yp_inst_in.exists() == args.yp_inst_out.exists());
      assert(args.ym_inst_in.exists() == args.ym_inst_out.exists());

      {
        std::vector<Event> events;

        if (args.xp_inst_in.exists()) events.push_back(args.xp_inst_in.fetch_metadata(p));
        if (args.xm_inst_in.exists()) events.push_back(args.xm_inst_in.fetch_metadata(p));
        if (args.yp_inst_in.exists()) events.push_back(args.yp_inst_in.fetch_metadata(p));
        if (args.ym_inst_in.exists()) events.push_back(args.ym_inst_in.fetch_metadata(p));

        if (args.xp_inst_out.exists()) events.push_back(args.xp_inst_out.fetch_metadata(p));
        if (args.xm_inst_out.exists()) events.push_back(args.xm_inst_out.fetch_metadata(p));
        if (args.yp_inst_out.exists()) events.push_back(args.yp_inst_out.fetch_metadata(p));
        if (args.ym_inst_out.exists()) events.push_back(args.ym_inst_out.fetch_metadata(p));
        Event::merge_events(events).wait();
      }

      if (args.xp_inst_in.exists()) assert(exterior_bounds.contains(args.xp_inst_in.get_indexspace<2, coord_t>().bounds));
      if (args.xm_inst_in.exists()) assert(exterior_bounds.contains(args.xm_inst_in.get_indexspace<2, coord_t>().bounds));
      if (args.yp_inst_in.exists()) assert(exterior_bounds.contains(args.yp_inst_in.get_indexspace<2, coord_t>().bounds));
      if (args.ym_inst_in.exists()) assert(exterior_bounds.contains(args.ym_inst_in.get_indexspace<2, coord_t>().bounds));

      if (args.xp_inst_in.exists()) assert(!interior_bounds.contains(args.xp_inst_in.get_indexspace<2, coord_t>().bounds));
      if (args.xm_inst_in.exists()) assert(!interior_bounds.contains(args.xm_inst_in.get_indexspace<2, coord_t>().bounds));
      if (args.yp_inst_in.exists()) assert(!interior_bounds.contains(args.yp_inst_in.get_indexspace<2, coord_t>().bounds));
      if (args.ym_inst_in.exists()) assert(!interior_bounds.contains(args.ym_inst_in.get_indexspace<2, coord_t>().bounds));

      if (args.xp_inst_out.exists()) assert(interior_bounds.contains(args.xp_inst_out.get_indexspace<2, coord_t>().bounds));
      if (args.xm_inst_out.exists()) assert(interior_bounds.contains(args.xm_inst_out.get_indexspace<2, coord_t>().bounds));
      if (args.yp_inst_out.exists()) assert(interior_bounds.contains(args.yp_inst_out.get_indexspace<2, coord_t>().bounds));
      if (args.ym_inst_out.exists()) assert(interior_bounds.contains(args.ym_inst_out.get_indexspace<2, coord_t>().bounds));

      assert(args.xp_inst_in.exists() == args.xp_empty_in.exists());
      assert(args.xm_inst_in.exists() == args.xm_empty_in.exists());
      assert(args.yp_inst_in.exists() == args.yp_empty_in.exists());
      assert(args.ym_inst_in.exists() == args.ym_empty_in.exists());

      assert(args.xp_inst_in.exists() == args.xp_empty_out.exists());
      assert(args.xm_inst_in.exists() == args.xm_empty_out.exists());
      assert(args.yp_inst_in.exists() == args.yp_empty_out.exists());
      assert(args.ym_inst_in.exists() == args.ym_empty_out.exists());

      assert(args.xp_inst_in.exists() == args.xp_full_in.exists());
      assert(args.xm_inst_in.exists() == args.xm_full_in.exists());
      assert(args.yp_inst_in.exists() == args.yp_full_in.exists());
      assert(args.ym_inst_in.exists() == args.ym_full_in.exists());

      assert(args.xp_inst_in.exists() == args.xp_full_out.exists());
      assert(args.xm_inst_in.exists() == args.xm_full_out.exists());
      assert(args.yp_inst_in.exists() == args.yp_full_out.exists());
      assert(args.ym_inst_in.exists() == args.ym_full_out.exists());

      // Launch task
      events.push_back(shard_procs[i].spawn(SHARD_TASK, &args, sizeof(args)));
    }
    Event::merge_events(events).wait();
  }

  // Collect start and stop times
  long long first_start;
  first_start_bar.wait();
  assert(first_start_bar.get_result(&first_start, sizeof(first_start)));

  long long last_start;
  last_start_bar.wait();
  assert(last_start_bar.get_result(&last_start, sizeof(last_start)));

  long long first_stop;
  first_stop_bar.wait();
  assert(first_stop_bar.get_result(&first_stop, sizeof(first_stop)));

  long long last_stop;
  last_stop_bar.wait();
  assert(last_stop_bar.get_result(&last_stop, sizeof(last_stop)));

  long long start = first_start;
  long long stop = last_stop;

  printf("\n");
  printf("Elapsed time: %e seconds\n", (stop - start)/1e6);
  printf("Iterations: %lld\n", config.tsteps);
  printf("Time per iteration: %e seconds\n",
         (stop - start)/1e6/config.tsteps);
  printf("Start skew: %e seconds\n", (last_start - first_start)/1e6);
  printf("Stop skew: %e seconds\n", (last_stop - first_stop)/1e6);
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(CREATE_REGION_TASK, create_region_task);
  rt.register_task(CREATE_REGION_DONE_TASK, create_region_done_task);
  rt.register_task(SHARD_TASK, shard_task);
  rt.register_task(STENCIL_TASK, stencil_task);
  rt.register_task(INCREMENT_TASK, increment_task);
  rt.register_task(CHECK_TASK, check_task);

  rt.register_reduction<RedopMin>(REDOP_MIN);
  rt.register_reduction<RedopMax>(REDOP_MAX);

  // select a processor to run the top level task on
  Processor p = Processor::NO_PROC;
  {
    Machine::ProcessorQuery query(Machine::get_machine());
    query.only_kind(Processor::LOC_PROC);
    p = query.first();
  }
  assert(p.exists());

  AppConfig config = parse_config(argc, argv);

  // collective launch of a single task - everybody gets the same finish event
  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, &config, sizeof(config));

  // request shutdown once that task is complete
  rt.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();

  return 0;
}
