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

using namespace Realm;

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
};

enum {
  FID_INPUT = 101,
  FID_OUTPUT = 102,
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
      Rect<2> xp_bounds(Point<2>(x_blocks[i.x].hi,              y_blocks[i.y].lo),
                        Point<2>(x_blocks[i.x].hi + RADIUS - 1, y_blocks[i.y].hi));
      Rect<2> xm_bounds(Point<2>(x_blocks[i.x].lo - RADIUS,     y_blocks[i.y].lo),
                        Point<2>(x_blocks[i.x].lo - 1,          y_blocks[i.y].hi));
      Rect<2> yp_bounds(Point<2>(x_blocks[i.x].lo,              y_blocks[i.y].hi),
                        Point<2>(x_blocks[i.x].hi,              y_blocks[i.y].hi + RADIUS + 1));
      Rect<2> ym_bounds(Point<2>(x_blocks[i.x].lo,              y_blocks[i.y].lo - RADIUS),
                        Point<2>(x_blocks[i.x].hi,              y_blocks[i.y].lo - 1));
      Memory memory(proc_regmems[shard_procs[i]]);
      events.push_back(
        RegionInstance::create_instance(xp_insts[i], memory,
                                        xp_bounds, field_sizes,
                                        0 /*SOA*/, ProfilingRequestSet()));
      events.push_back(
        RegionInstance::create_instance(xm_insts[i], memory,
                                        xm_bounds, field_sizes,
                                        0 /*SOA*/, ProfilingRequestSet()));
      events.push_back(
        RegionInstance::create_instance(yp_insts[i], memory,
                                        yp_bounds, field_sizes,
                                        0 /*SOA*/, ProfilingRequestSet()));
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
    for (std::vector<Event>::iterator it = events.begin(), ie = events.end(); it != ie; ++it) {
      it->wait();
    }
  }

  // Create phase barriers
  std::map<Point<2>, Barrier> xp_bars_empty_in;
  std::map<Point<2>, Barrier> xm_bars_empty_in;
  std::map<Point<2>, Barrier> yp_bars_empty_in;
  std::map<Point<2>, Barrier> ym_bars_empty_in;

  std::map<Point<2>, Barrier> xp_bars_empty_out;
  std::map<Point<2>, Barrier> xm_bars_empty_out;
  std::map<Point<2>, Barrier> yp_bars_empty_out;
  std::map<Point<2>, Barrier> ym_bars_empty_out;

  std::map<Point<2>, Barrier> xp_bars_full_in;
  std::map<Point<2>, Barrier> xm_bars_full_in;
  std::map<Point<2>, Barrier> yp_bars_full_in;
  std::map<Point<2>, Barrier> ym_bars_full_in;

  std::map<Point<2>, Barrier> xp_bars_full_out;
  std::map<Point<2>, Barrier> xm_bars_full_out;
  std::map<Point<2>, Barrier> yp_bars_full_out;
  std::map<Point<2>, Barrier> ym_bars_full_out;

  for (PointInRectIterator<2, int> it(shards); it.valid; it.step()) {

  }

  // Launch shard tasks
  for (PointInRectIterator<2, int> it(shards); it.valid; it.step()) {
    ShardArgs args;
  }
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

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

  AppConfig config = parse_config(argc, argv);

  // collective launch of a single task - everybody gets the same finish event
  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, &config, sizeof(config));

  // request shutdown once that task is complete
  rt.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();

  return 0;
}
