#include <realm.h>
#include <realm/cmdline.h>

using namespace Realm;

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
};

enum
{
  FID1 = 101,
  FID2 = 102,
};

Logger log_app("app");

struct CreateRegionArgs {
  RegionInstance *inst;
  Memory memory;
  Rect<1, int> bounds;
};

struct InstanceLogicalLayout1 {
  int x;
  float y;
};

struct InstanceLogicalLayout2 {
  long long x;
  double y;
};

template <typename FT, typename T0, typename T1>
void update(RegionInstance inst, Rect<1, int> bounds, FieldID fid, int add)
{
  AffineAccessor<FT, 1, int> accessor(inst, fid);
  PointInRectIterator<1, int> pit(bounds);
  while(pit.valid) {
    accessor[pit.p].x = static_cast<T0>(pit.p.x + add);
    accessor[pit.p].y = static_cast<T1>(pit.p.x + add + 1);
    pit.step();
  }
}

template <typename FT, typename T0, typename T1>
void verify(RegionInstance inst, Rect<1, int> bounds, FieldID fid, int add)
{
  AffineAccessor<FT, 1, int> accessor(inst, fid);
  PointInRectIterator<1, int> pit(bounds);
  while(pit.valid) {
    assert(accessor[pit.p].x == static_cast<T0>(pit.p.x + add));
    assert(accessor[pit.p].y == static_cast<T1>(pit.p.x + add + 1));
    log_app.info() << "p=" << pit.p << " x=" << accessor[pit.p].x
                   << " y=" << accessor[pit.p].y;
    pit.step();
  }
}

void top_level_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p)
{
  Rect<1, int> bounds;
  bounds.lo = Point<1, int>(0);
  bounds.hi = Point<1, int>(7);

  std::vector<Memory> memories;
  Machine::MemoryQuery mq(Machine::get_machine());
  mq.only_kind(Memory::SYSTEM_MEM).has_capacity(1);
  memories.assign(mq.begin(), mq.end());
  assert(!memories.empty());

  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID1] = sizeof(InstanceLogicalLayout1);
  field_sizes[FID2] = sizeof(InstanceLogicalLayout2);

  RegionInstance inst = RegionInstance::NO_INST;
  RegionInstance::create_instance(inst, *memories.begin(), bounds, field_sizes,
                                  /*SOA=*/0, ProfilingRequestSet())
      .wait();

  update<InstanceLogicalLayout1, int, float>(inst, bounds, FID1, /*add=*/1);
  update<InstanceLogicalLayout2, long long, double>(inst, bounds, FID2,
                                                    /*add=*/2);

  verify<InstanceLogicalLayout1, int, float>(inst, bounds, FID1, /*add=*/1);
  verify<InstanceLogicalLayout2, long long, double>(inst, bounds, FID2,
                                                    /*add=*/2);

  Runtime::get_runtime().shutdown(Event::NO_EVENT, 0 /*success*/);
}

int main(int argc, const char **argv)
{
  Runtime rt;

  rt.init(&argc, (char ***)&argv);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();

  if(!p.exists()) {
    p = Machine::ProcessorQuery(Machine::get_machine()).first();
  }

  assert(p.exists());

  Processor::register_task_by_kind(p.kind(), false /*!global*/, TOP_LEVEL_TASK,
                                   CodeDescriptor(top_level_task),
                                   ProfilingRequestSet())
      .external_wait();

  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  int ret = rt.wait_for_shutdown();

  return ret;
}
