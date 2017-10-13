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

Event fill(RegionInstance inst, FieldID fid, DTYPE value) {
  CopySrcDstField field;
  field.inst = inst;
  field.field_id = fid;
  field.size = sizeof(DTYPE);

  std::vector<CopySrcDstField> fields;
  fields.push_back(field);

  return inst.get_indexspace<2>().fill(fields, ProfilingRequestSet(),
                                       &value, sizeof(value));
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

  DTYPE *weights = get_weights();

  Rect<2> private_bounds = a.private_inst.get_indexspace<2>().bounds;

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
          a.interior_bounds.lo.x - private_bounds.lo.x,
          a.interior_bounds.hi.x - private_bounds.lo.x + 1,
          a.interior_bounds.lo.y - private_bounds.lo.y,
          a.interior_bounds.hi.y - private_bounds.lo.y + 1);

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

  increment(private_base_input,
          private_stride_input/sizeof(DTYPE),
          a.outer_bounds.lo.x - private_bounds.lo.x,
          a.outer_bounds.hi.x - private_bounds.lo.x + 1,
          a.outer_bounds.lo.y - private_bounds.lo.y,
          a.outer_bounds.hi.y - private_bounds.lo.y + 1);

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
  RegionInstance private_inst;
  {
    Machine::MemoryQuery query(Machine::get_machine());
    query.has_affinity_to(p);
    query.only_kind(Memory::SYSTEM_MEM);
    Memory memory = query.first();

    std::map<FieldID, size_t> field_sizes;
    field_sizes[FID_INPUT] = sizeof(DTYPE);
    field_sizes[FID_OUTPUT] = sizeof(DTYPE);

    RegionInstance::create_instance(private_inst, memory,
                                    a.exterior_bounds, field_sizes,
                                    0 /*SOA*/, ProfilingRequestSet()).wait();
    assert(private_inst.exists());
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

  // Main time step loop
  Event stencil_done;
  Event increment_done;
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
      stencil_done = p.spawn(STENCIL_TASK, &args, sizeof(args), increment_done);
    }

    {
      IncrementArgs args;
      args.private_inst = private_inst;
      args.xp_inst = a.xp_inst_in;
      args.xm_inst = a.xm_inst_in;
      args.yp_inst = a.yp_inst_in;
      args.ym_inst = a.ym_inst_in;
      args.print_ts = t == a.tsteps - a.tprune - 1;
      args.outer_bounds = a.outer_bounds;
      increment_done = p.spawn(INCREMENT_TASK, &args, sizeof(args), stencil_done);
    }
  }

  Event check_done;
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
  std::map<Point<2>, RegionInstance> xp_insts;
  std::map<Point<2>, RegionInstance> xm_insts;
  std::map<Point<2>, RegionInstance> yp_insts;
  std::map<Point<2>, RegionInstance> ym_insts;

  {
    std::map<FieldID, size_t> field_sizes;
    field_sizes[FID_INPUT] = sizeof(DTYPE);
    field_sizes[FID_OUTPUT] = sizeof(DTYPE);

    std::vector<Event> events;
    for (PointInRectIterator<2, int> it(shards); it.valid; it.step()) {
      Point<2> i(it.p);
      Rect<2> xp_bounds(Point<2>(x_blocks[i.x].hi - RADIUS + 1, y_blocks[i.y].lo),
                        Point<2>(x_blocks[i.x].hi,              y_blocks[i.y].hi));
      Rect<2> xm_bounds(Point<2>(x_blocks[i.x].lo,              y_blocks[i.y].lo),
                        Point<2>(x_blocks[i.x].lo + RADIUS - 1, y_blocks[i.y].hi));
      Rect<2> yp_bounds(Point<2>(x_blocks[i.x].lo,              y_blocks[i.y].hi - RADIUS + 1),
                        Point<2>(x_blocks[i.x].hi,              y_blocks[i.y].hi));
      Rect<2> ym_bounds(Point<2>(x_blocks[i.x].lo,              y_blocks[i.y].lo),
                        Point<2>(x_blocks[i.x].hi,              y_blocks[i.y].lo + RADIUS - 1));

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
  std::map<Point<2>, Barrier> xp_bars_empty;
  std::map<Point<2>, Barrier> xm_bars_empty;
  std::map<Point<2>, Barrier> yp_bars_empty;
  std::map<Point<2>, Barrier> ym_bars_empty;

  std::map<Point<2>, Barrier> xp_bars_full;
  std::map<Point<2>, Barrier> xm_bars_full;
  std::map<Point<2>, Barrier> yp_bars_full;
  std::map<Point<2>, Barrier> ym_bars_full;

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

      // Sanity checks
      assert(exterior_bounds.contains(outer_bounds));
      assert(outer_bounds.contains(interior_bounds));

      assert(args.xp_inst_in.exists() == args.xp_inst_out.exists());
      assert(args.xm_inst_in.exists() == args.xm_inst_out.exists());
      assert(args.yp_inst_in.exists() == args.yp_inst_out.exists());
      assert(args.ym_inst_in.exists() == args.ym_inst_out.exists());

      if (args.xp_inst_in.exists()) assert(interior_bounds.contains(args.xp_inst_in.get_indexspace<2>().bounds));
      if (args.xm_inst_in.exists()) assert(interior_bounds.contains(args.xm_inst_in.get_indexspace<2>().bounds));
      if (args.yp_inst_in.exists()) assert(interior_bounds.contains(args.yp_inst_in.get_indexspace<2>().bounds));
      if (args.ym_inst_in.exists()) assert(interior_bounds.contains(args.ym_inst_in.get_indexspace<2>().bounds));

      if (args.xp_inst_out.exists()) assert(exterior_bounds.contains(args.xp_inst_out.get_indexspace<2>().bounds));
      if (args.xm_inst_out.exists()) assert(exterior_bounds.contains(args.xm_inst_out.get_indexspace<2>().bounds));
      if (args.yp_inst_out.exists()) assert(exterior_bounds.contains(args.yp_inst_out.get_indexspace<2>().bounds));
      if (args.ym_inst_out.exists()) assert(exterior_bounds.contains(args.ym_inst_out.get_indexspace<2>().bounds));

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
