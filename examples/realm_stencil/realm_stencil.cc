/* Copyright 2017 Stanford University
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

#include "realm.h"

#include "cpu_kernels.h"

using namespace Realm;

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  SHARD_TASK     = Processor::TASK_ID_FIRST_AVAILABLE+1,
  STENCIL_TASK   = Processor::TASK_ID_FIRST_AVAILABLE+2,
  INCREMENT_TASK = Processor::TASK_ID_FIRST_AVAILABLE+3,
  CHECK_TASK     = Processor::TASK_ID_FIRST_AVAILABLE+4,
};

enum {
  FID_INPUT = 101,
  FID_OUTPUT = 102,
  FID_WEIGHT = 103,
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
    : nx(12), ny(12), ntx(4), nty(4), tsteps(20), tprune(5), init(1000) {}
public:
  size_t nx;
  size_t ny;
  size_t ntx;
  size_t nty;
  size_t tsteps;
  size_t tprune;
  size_t init;
};

void get_optional_arg(int argc, char **argv,
                      const char *flag, size_t &out)
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

Event fill(RegionInstance inst, FieldID fid, DTYPE value)
{
  CopySrcDstField field;
  field.inst = inst;
  field.field_id = fid;
  field.size = sizeof(DTYPE);

  std::vector<CopySrcDstField> fields;
  fields.push_back(field);

  return inst.get_indexspace<2>().fill(fields, ProfilingRequestSet(),
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

  return dst_inst.get_indexspace<2>().copy(src_fields, dst_fields,
                                           ProfilingRequestSet(),
                                           wait_for);
}

void get_base_and_stride(RegionInstance inst, FieldID fid, DTYPE *&base, size_t &stride)
{
  AffineAccessor<DTYPE, 2> acc = AffineAccessor<DTYPE, 2>(inst, fid);
  base = reinterpret_cast<DTYPE *>(acc.ptr(inst.get_indexspace<2>().bounds.lo));
  assert(acc.strides.x == sizeof(DTYPE));
  stride = acc.strides.y;
}

void dump(RegionInstance inst, FieldID fid, Rect<2> bounds, const char *prefix)
{
  AffineAccessor<DTYPE, 2> acc = AffineAccessor<DTYPE, 2>(inst, fid);
  for (PointInRectIterator<2, int> it(bounds); it.valid; it.step()) {
    printf("%s: %2d %2d value %8.3f\n", prefix, it.p.x, it.p.y, acc.read(it.p));
  }
}

DTYPE *get_weights()
{
  static bool init = false;
  static DTYPE weights[(2*RADIUS + 1) * (2*RADIUS + 1)] = {0};

  if (!init) {
#define WEIGHT(i, j) weights[(j + RADIUS) * (2 * RADIUS + 1) + (i + RADIUS)]
    for (coord_t i = 1; i <= RADIUS; i++) {
      WEIGHT( 0,  i) =  1.0/(2.0*i*RADIUS);
      WEIGHT( i,  0) =  1.0/(2.0*i*RADIUS);
      WEIGHT( 0, -i) = -1.0/(2.0*i*RADIUS);
      WEIGHT(-i,  0) = -1.0/(2.0*i*RADIUS);
    }
    init = true;
#undef WEIGHT
  }

  return weights;
}

void inline_copy(RegionInstance src_inst, RegionInstance dst_inst, FieldID fid,
                 Rect<2> bounds)
{
  AffineAccessor<DTYPE, 2> src_acc = AffineAccessor<DTYPE, 2>(src_inst, fid);
  AffineAccessor<DTYPE, 2> dst_acc = AffineAccessor<DTYPE, 2>(dst_inst, fid);
  for (PointInRectIterator<2, int> it(bounds); it.valid; it.step()) {
    dst_acc.write(it.p, src_acc.read(it.p));
  }
}

void inline_copy_raw(RegionInstance src_inst, RegionInstance dst_inst,
                     FieldID fid, Rect<2> bounds)
{
  // FIXME: Something is still wrong in the index arithmetic here

  AffineAccessor<DTYPE, 2> src_acc = AffineAccessor<DTYPE, 2>(src_inst, fid);
  DTYPE *src_base;
  size_t src_stride;
  get_base_and_stride(src_inst, fid, src_base, src_stride);

  AffineAccessor<DTYPE, 2> dst_acc = AffineAccessor<DTYPE, 2>(dst_inst, fid);
  DTYPE *dst_base;
  size_t dst_stride;
  get_base_and_stride(dst_inst, fid, dst_base, dst_stride);

  Rect<2> src_bounds = src_inst.get_indexspace<2>().bounds;
  Point<2> src_offset = bounds.lo - src_bounds.lo;

  Rect<2> dst_bounds = dst_inst.get_indexspace<2>().bounds;
  Point<2> dst_offset = bounds.lo - dst_bounds.lo;

  Point<2> size = bounds.hi - bounds.lo + Point<2>(1, 1);

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

  Rect<2> private_bounds = a.private_inst.get_indexspace<2>().bounds;
  Point<2> interior_offset = a.interior_bounds.lo - private_bounds.lo;
  Point<2> interior_size = a.interior_bounds.hi - a.interior_bounds.lo + Point<2>(1, 1);

  if (a.xp_inst.exists()) {
    inline_copy(a.xp_inst, a.private_inst, FID_INPUT,
                a.xp_inst.get_indexspace<2>().bounds);
  }

  if (a.xm_inst.exists()) {
    inline_copy(a.xm_inst, a.private_inst, FID_INPUT,
                a.xm_inst.get_indexspace<2>().bounds);
  }

  if (a.yp_inst.exists()) {
    inline_copy(a.yp_inst, a.private_inst, FID_INPUT,
                a.yp_inst.get_indexspace<2>().bounds);
  }

  if (a.ym_inst.exists()) {
    inline_copy(a.ym_inst, a.private_inst, FID_INPUT,
                a.ym_inst.get_indexspace<2>().bounds);
  }

  DTYPE *weights = get_weights();

  printf("private base %p\n", private_base_input);
  printf("private stride %lu (%lu elements)\n", private_stride_input, private_stride_input/sizeof(DTYPE));

  printf("private bounds %d %d to %d %d\n",
         private_bounds.lo.x, private_bounds.lo.y,
         private_bounds.hi.x, private_bounds.hi.y);

  printf("interior bounds %d %d to %d %d\n",
         a.interior_bounds.lo.x, a.interior_bounds.lo.y,
         a.interior_bounds.hi.x, a.interior_bounds.hi.y);

  stencil(private_base_input, private_base_output, weights,
          private_stride_input/sizeof(DTYPE),
          interior_offset.x,
          interior_offset.x + interior_size.x,
          interior_offset.y,
          interior_offset.y + interior_size.y);

  // dump(a.private_inst, FID_INPUT,  a.interior_bounds, " input");
  // dump(a.private_inst, FID_OUTPUT, a.interior_bounds, "output");
}

void increment_task(const void *args, size_t arglen,
                    const void *userdata, size_t userlen, Processor p)
{
  printf("increment\n");
  assert(arglen == sizeof(IncrementArgs));
  const IncrementArgs &a = *reinterpret_cast<const IncrementArgs *>(args);

  DTYPE *private_base_input;
  size_t private_stride_input;
  get_base_and_stride(a.private_inst, FID_INPUT, private_base_input, private_stride_input);

  Rect<2> private_bounds = a.private_inst.get_indexspace<2>().bounds;
  Point<2> outer_offset = a.outer_bounds.lo - private_bounds.lo;
  Point<2> outer_size = a.outer_bounds.hi - a.outer_bounds.lo + Point<2>(1, 1);

  increment(private_base_input,
            private_stride_input/sizeof(DTYPE),
            outer_offset.x,
            outer_offset.x + outer_size.x,
            outer_offset.y,
            outer_offset.y + outer_size.y);

  if (a.xp_inst.exists()) {
    inline_copy(a.private_inst, a.xp_inst, FID_INPUT,
                a.xp_inst.get_indexspace<2>().bounds);
  }

  if (a.xm_inst.exists()) {
    inline_copy(a.private_inst, a.xm_inst, FID_INPUT,
                a.xm_inst.get_indexspace<2>().bounds);
  }

  if (a.yp_inst.exists()) {
    inline_copy(a.private_inst, a.yp_inst, FID_INPUT,
                a.yp_inst.get_indexspace<2>().bounds);
  }

  if (a.ym_inst.exists()) {
    inline_copy(a.private_inst, a.ym_inst, FID_INPUT,
                a.ym_inst.get_indexspace<2>().bounds);
  }

  // dump(a.private_inst, FID_INPUT,  a.outer_bounds, " input");
}

void check_task(const void *args, size_t arglen,
                const void *userdata, size_t userlen, Processor p)
{
  printf("check\n");
  assert(arglen == sizeof(CheckArgs));
  const CheckArgs &a = *reinterpret_cast<const CheckArgs *>(args);

  DTYPE expect_input = a.init + a.tsteps;
  DTYPE expect_output = a.init;

  // Check input
  {
    AffineAccessor<DTYPE, 2> acc = AffineAccessor<DTYPE, 2>(a.private_inst, FID_INPUT);
    for (PointInRectIterator<2, int> it(a.interior_bounds); it.valid; it.step()) {
      if (acc.read(it.p) != expect_input) {
        printf("bad value: got %f expected %f\n", acc.read(it.p), expect_input);
        assert(false);
        abort(); // if someone compiles with NDEBUG make sure this fails anyway
      }
    }
  }

  // Check output
  {
    AffineAccessor<DTYPE, 2> acc = AffineAccessor<DTYPE, 2>(a.private_inst, FID_OUTPUT);
    for (PointInRectIterator<2, int> it(a.interior_bounds); it.valid; it.step()) {
      if (acc.read(it.p) != expect_output) {
        printf("bad value: got %f expected %f\n", acc.read(it.p), expect_output);
        assert(false);
        abort(); // if someone compiles with NDEBUG make sure this fails anyway
      }
    }
  }
}

void shard_task(const void *args, size_t arglen,
                const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(ShardArgs));
  const ShardArgs &a = *reinterpret_cast<const ShardArgs *>(args);
  printf("shard %d %d running on processor " IDFMT "\n", a.point.x, a.point.y, p.id);

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
    if (a.xp_inst_out.exists())
      events.push_back(
        RegionInstance::create_instance(xp_inst_out_local, a.regmem,
                                        a.xp_inst_out.get_indexspace<2>(), field_sizes,
                                        0 /*SOA*/, ProfilingRequestSet()));
    if (a.xm_inst_out.exists())
      events.push_back(
        RegionInstance::create_instance(xm_inst_out_local, a.regmem,
                                        a.xm_inst_out.get_indexspace<2>(), field_sizes,
                                        0 /*SOA*/, ProfilingRequestSet()));
    if (a.yp_inst_out.exists())
      events.push_back(
        RegionInstance::create_instance(yp_inst_out_local, a.regmem,
                                        a.yp_inst_out.get_indexspace<2>(), field_sizes,
                                        0 /*SOA*/, ProfilingRequestSet()));
    if (a.ym_inst_out.exists())
      events.push_back(
        RegionInstance::create_instance(ym_inst_out_local, a.regmem,
                                        a.ym_inst_out.get_indexspace<2>(), field_sizes,
                                        0 /*SOA*/, ProfilingRequestSet()));
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
  for (size_t t = 0; t < a.tsteps; t++) {
    {
      StencilArgs args;
      args.private_inst = private_inst;
      args.xp_inst = a.xp_inst_in;
      args.xm_inst = a.xm_inst_in;
      args.yp_inst = a.yp_inst_in;
      args.ym_inst = a.ym_inst_in;
      args.print_ts = t == a.tprune;
      args.interior_bounds = a.interior_bounds;
      Event precondition = Event::merge_events(
        increment_done,
        (xp_full_in.exists() ? xp_full_in.get_previous_phase() : Event::NO_EVENT),
        (xm_full_in.exists() ? xm_full_in.get_previous_phase() : Event::NO_EVENT),
        (yp_full_in.exists() ? yp_full_in.get_previous_phase() : Event::NO_EVENT),
        (ym_full_in.exists() ? ym_full_in.get_previous_phase() : Event::NO_EVENT));
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
      args.print_ts = t == a.tsteps - a.tprune - 1;
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

  Event check_done = Event::NO_EVENT;
  {
    CheckArgs check_args;
    check_args.private_inst = private_inst;
    check_args.tsteps = a.tsteps;
    check_args.init = a.init;
    check_args.interior_bounds = a.interior_bounds;
    check_done = p.spawn(CHECK_TASK, &check_args, sizeof(check_args), increment_done);
  }

  Event complete = check_done;
  complete.wait();
}

void top_level_task(const void *args, size_t arglen,
                    const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(AppConfig));
  const AppConfig &config = *reinterpret_cast<const AppConfig *>(args);
  printf("Stencil configuration:\n");
  printf("      nx %lu\n", config.nx);
  printf("      ny %lu\n", config.ny);
  printf("     ntx %lu\n", config.ntx);
  printf("     nty %lu\n", config.nty);
  printf("  tsteps %lu\n", config.tsteps);
  printf("  tprune %lu\n", config.tprune);
  printf("    init %lu\n", config.init);

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
  assert(procs.size() >= config.ntx*config.nty); // Expect one core per shard
  Rect<2> shards(Point<2>(0, 0), Point<2>(config.ntx-1, config.nty-1));
  std::map<Point<2>, Processor> shard_procs;
  {
    std::vector<Processor>::iterator pit = procs.begin();
    for (PointInRectIterator<2, int> it(shards); it.valid; it.step()) {
      assert(pit != procs.end());
      shard_procs[it.p] = *pit++;
    }
  }

  // Size of grid excluding the border
  size_t nx = config.nx - 2*RADIUS;
  size_t ny = config.ny - 2*RADIUS;
  assert(nx >= config.ntx);
  assert(ny >= config.nty);

  // Choose block sizes for each shard
  std::vector<Rect<1> > x_blocks;
  std::vector<Rect<1> > y_blocks;

  for (size_t ix = 0; ix < config.ntx; ix++) {
    x_blocks.push_back(Rect<1>(ix*nx/config.ntx, (ix+1)*nx/config.ntx - 1));
  }
  for (size_t iy = 0; iy < config.nty; iy++) {
    y_blocks.push_back(Rect<1>(iy*ny/config.nty, (iy+1)*ny/config.nty - 1));
  }

  // Create incoming exchange buffers
  DefaultMap<Point<2>, RegionInstance> xp_insts(RegionInstance::NO_INST);
  DefaultMap<Point<2>, RegionInstance> xm_insts(RegionInstance::NO_INST);
  DefaultMap<Point<2>, RegionInstance> yp_insts(RegionInstance::NO_INST);
  DefaultMap<Point<2>, RegionInstance> ym_insts(RegionInstance::NO_INST);

  {
    std::map<FieldID, size_t> field_sizes;
    field_sizes[FID_INPUT] = sizeof(DTYPE);
    field_sizes[FID_OUTPUT] = sizeof(DTYPE);

    std::vector<Event> events;
    for (PointInRectIterator<2, int> it(shards); it.valid; it.step()) {
      Point<2> i(it.p);
      Rect<2> xp_bounds(Point<2>(x_blocks[i.x].hi + 1,      y_blocks[i.y].lo),
                        Point<2>(x_blocks[i.x].hi + RADIUS, y_blocks[i.y].hi));
      Rect<2> xm_bounds(Point<2>(x_blocks[i.x].lo - RADIUS, y_blocks[i.y].lo),
                        Point<2>(x_blocks[i.x].lo - 1,      y_blocks[i.y].hi));
      Rect<2> yp_bounds(Point<2>(x_blocks[i.x].lo,          y_blocks[i.y].hi + 1),
                        Point<2>(x_blocks[i.x].hi,          y_blocks[i.y].hi + RADIUS));
      Rect<2> ym_bounds(Point<2>(x_blocks[i.x].lo,          y_blocks[i.y].lo - RADIUS),
                        Point<2>(x_blocks[i.x].hi,          y_blocks[i.y].lo - 1));

      Memory memory(proc_regmems[shard_procs[i]]);
      if (i.x != shards.hi.x)
        events.push_back(
          RegionInstance::create_instance(xp_insts[i], memory,
                                          xp_bounds, field_sizes,
                                          0 /*SOA*/, ProfilingRequestSet()));
      if (i.x != shards.lo.x)
        events.push_back(
          RegionInstance::create_instance(xm_insts[i], memory,
                                          xm_bounds, field_sizes,
                                          0 /*SOA*/, ProfilingRequestSet()));
      if (i.y != shards.hi.y)
        events.push_back(
          RegionInstance::create_instance(yp_insts[i], memory,
                                          yp_bounds, field_sizes,
                                          0 /*SOA*/, ProfilingRequestSet()));
      if (i.y != shards.lo.y)
        events.push_back(
          RegionInstance::create_instance(ym_insts[i], memory,
                                          ym_bounds, field_sizes,
                                          0 /*SOA*/, ProfilingRequestSet()));

      printf("block %d %d\n", i.x, i.y);
      printf("  bounds %d %d to %d %d\n",
             x_blocks[i.x].lo.x, y_blocks[i.y].lo.x,
             x_blocks[i.x].hi.x, y_blocks[i.y].hi.x);
      printf("  xp %d %d to %d %d\n",
             xp_bounds.lo.x, xp_bounds.lo.y,
             xp_bounds.hi.x, xp_bounds.hi.y);
      printf("  xm %d %d to %d %d\n",
             xm_bounds.lo.x, xm_bounds.lo.y,
             xm_bounds.hi.x, xm_bounds.hi.y);
      printf("  yp %d %d to %d %d\n",
             yp_bounds.lo.x, yp_bounds.lo.y,
             yp_bounds.hi.x, yp_bounds.hi.y);
      printf("  ym %d %d to %d %d\n",
             ym_bounds.lo.x, ym_bounds.lo.y,
             ym_bounds.hi.x, ym_bounds.hi.y);
    }
    Event::merge_events(events).wait();
  }

  // Create incoming phase barriers
  DefaultMap<Point<2>, Barrier> xp_bars_empty(Barrier::NO_BARRIER);
  DefaultMap<Point<2>, Barrier> xm_bars_empty(Barrier::NO_BARRIER);
  DefaultMap<Point<2>, Barrier> yp_bars_empty(Barrier::NO_BARRIER);
  DefaultMap<Point<2>, Barrier> ym_bars_empty(Barrier::NO_BARRIER);

  DefaultMap<Point<2>, Barrier> xp_bars_full(Barrier::NO_BARRIER);
  DefaultMap<Point<2>, Barrier> xm_bars_full(Barrier::NO_BARRIER);
  DefaultMap<Point<2>, Barrier> yp_bars_full(Barrier::NO_BARRIER);
  DefaultMap<Point<2>, Barrier> ym_bars_full(Barrier::NO_BARRIER);

  for (PointInRectIterator<2, int> it(shards); it.valid; it.step()) {
    Point<2> i(it.p);
    printf("block %d %d\n", i.x, i.y);
    printf("  xp bars expect %d\n", i.x != shards.hi.x ? 1 : 0);
    printf("  xm bars expect %d\n", i.x != shards.lo.x ? 1 : 0);
    printf("  yp bars expect %d\n", i.y != shards.hi.y ? 1 : 0);
    printf("  ym bars expect %d\n", i.y != shards.lo.y ? 1 : 0);

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
  Barrier sync_bar = Barrier::create_barrier(config.ntx * config.nty);
  printf("sync bar expects %lu arrivals\n", config.ntx * config.nty);

  // Launch shard tasks
  {
    std::vector<Event> events;
    for (PointInRectIterator<2, int> it(shards); it.valid; it.step()) {
      Point<2> i(it.p);

      Rect<2> interior_bounds(Point<2>(x_blocks[i.x].lo, y_blocks[i.y].lo),
                              Point<2>(x_blocks[i.x].hi, y_blocks[i.y].hi));
      Rect<2> exterior_bounds(Point<2>(x_blocks[i.x].lo - RADIUS, y_blocks[i.y].lo - RADIUS),
                              Point<2>(x_blocks[i.x].hi + RADIUS, y_blocks[i.y].hi + RADIUS));
      // As interior, but bloated only on the outer edges
      Rect<2> outer_bounds(Point<2>(x_blocks[i.x].lo - (i.x == shards.lo.x ? RADIUS : 0),
                                    y_blocks[i.y].lo - (i.y == shards.lo.y ? RADIUS : 0)),
                           Point<2>(x_blocks[i.x].hi + (i.x == shards.hi.x ? RADIUS : 0),
                                    y_blocks[i.y].hi + (i.y == shards.hi.y ? RADIUS : 0)));

      // Pack arguments
      ShardArgs args;
      args.xp_inst_in = xp_insts[i];
      args.xm_inst_in = xm_insts[i];
      args.yp_inst_in = yp_insts[i];
      args.ym_inst_in = ym_insts[i];

      args.xp_inst_out = xm_insts[i + Point<2>( 1,  0)];
      args.xm_inst_out = xp_insts[i + Point<2>(-1,  0)];
      args.yp_inst_out = ym_insts[i + Point<2>( 0,  1)];
      args.ym_inst_out = yp_insts[i + Point<2>( 0, -1)];

      args.xp_empty_in = xp_bars_empty[i];
      args.xm_empty_in = xm_bars_empty[i];
      args.yp_empty_in = yp_bars_empty[i];
      args.ym_empty_in = ym_bars_empty[i];

      args.xp_empty_out = xm_bars_empty[i + Point<2>( 1,  0)];
      args.xm_empty_out = xp_bars_empty[i + Point<2>(-1,  0)];
      args.yp_empty_out = ym_bars_empty[i + Point<2>( 0,  1)];
      args.ym_empty_out = yp_bars_empty[i + Point<2>( 0, -1)];

      args.xp_full_in = xp_bars_full[i];
      args.xm_full_in = xm_bars_full[i];
      args.yp_full_in = yp_bars_full[i];
      args.ym_full_in = ym_bars_full[i];

      args.xp_full_out = xm_bars_full[i + Point<2>( 1,  0)];
      args.xm_full_out = xp_bars_full[i + Point<2>(-1,  0)];
      args.yp_full_out = ym_bars_full[i + Point<2>( 0,  1)];
      args.ym_full_out = yp_bars_full[i + Point<2>( 0, -1)];

      args.sync = sync_bar;

      args.tsteps = config.tsteps;
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

      if (args.xp_inst_in.exists()) assert(exterior_bounds.contains(args.xp_inst_in.get_indexspace<2>().bounds));
      if (args.xm_inst_in.exists()) assert(exterior_bounds.contains(args.xm_inst_in.get_indexspace<2>().bounds));
      if (args.yp_inst_in.exists()) assert(exterior_bounds.contains(args.yp_inst_in.get_indexspace<2>().bounds));
      if (args.ym_inst_in.exists()) assert(exterior_bounds.contains(args.ym_inst_in.get_indexspace<2>().bounds));

      if (args.xp_inst_in.exists()) assert(!interior_bounds.contains(args.xp_inst_in.get_indexspace<2>().bounds));
      if (args.xm_inst_in.exists()) assert(!interior_bounds.contains(args.xm_inst_in.get_indexspace<2>().bounds));
      if (args.yp_inst_in.exists()) assert(!interior_bounds.contains(args.yp_inst_in.get_indexspace<2>().bounds));
      if (args.ym_inst_in.exists()) assert(!interior_bounds.contains(args.ym_inst_in.get_indexspace<2>().bounds));

      if (args.xp_inst_out.exists()) assert(interior_bounds.contains(args.xp_inst_out.get_indexspace<2>().bounds));
      if (args.xm_inst_out.exists()) assert(interior_bounds.contains(args.xm_inst_out.get_indexspace<2>().bounds));
      if (args.yp_inst_out.exists()) assert(interior_bounds.contains(args.yp_inst_out.get_indexspace<2>().bounds));
      if (args.ym_inst_out.exists()) assert(interior_bounds.contains(args.ym_inst_out.get_indexspace<2>().bounds));

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
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(SHARD_TASK, shard_task);
  rt.register_task(STENCIL_TASK, stencil_task);
  rt.register_task(INCREMENT_TASK, increment_task);
  rt.register_task(CHECK_TASK, check_task);

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
