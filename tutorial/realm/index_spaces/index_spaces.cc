#include "realm.h"
#include "realm/id.h"
#include "realm/cmdline.h"

#include <cassert>

using namespace Realm;

Logger log_app("app");

enum {
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
};

enum {
  FID_BASE = 44,
};

namespace SampleConfig {
  size_t src_buffer_size = 64 << 20;
  size_t dst_buffer_size = 16 << 20;
};

Event fill(RegionInstance inst, IndexSpace<1, int> is, int fill_value,
           Event wait_on = Event::NO_EVENT) {
  std::vector<CopySrcDstField> dsts(1);
  dsts[0].set_field(inst, FID_BASE, sizeof(int));
  return is.fill(dsts, ProfilingRequestSet(), &fill_value, sizeof(fill_value),
                 wait_on);
}

void main_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p) {
  std::vector<Memory> memories;
  Machine::MemoryQuery mq(Machine::get_machine());
  mq.only_kind(Memory::SYSTEM_MEM).has_capacity(1);
  memories.assign(mq.begin(), mq.end());

  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID_BASE] = sizeof(int);

  size_t src_elements = SampleConfig::src_buffer_size / sizeof(int);
  size_t dst_elements = SampleConfig::dst_buffer_size / sizeof(int);

  IndexSpace<1> src_index_space = IndexSpace<1>(Rect<1>(0, src_elements - 1));
  IndexSpace<1> dst_index_space = IndexSpace<1>(Rect<1>(0, dst_elements - 1));

  size_t mem_idx = 0;

  RegionInstance inst1;
  Event ev1 = RegionInstance::create_instance(
      inst1, memories[mem_idx++ % memories.size()], src_index_space, field_sizes,
      /*block_size=*/0, ProfilingRequestSet());

  RegionInstance inst2;
  Event ev2 = RegionInstance::create_instance(
      inst2, memories[mem_idx++ % memories.size()], dst_index_space,
      field_sizes, /*block_size=*/0, ProfilingRequestSet());

  Event fill_event =
      Event::merge_events(fill(inst1, src_index_space, /*fill_value=*/7, ev1),
                          fill(inst2, dst_index_space, /*fill_value=*/8, ev2));

  IndexSpace<1> isect;
  Event isect_event = IndexSpace<1>::compute_intersection(
      src_index_space, dst_index_space, isect, ProfilingRequestSet(),
      fill_event);

  std::vector<CopySrcDstField> srcs(1), dsts(1);
  srcs[0].set_field(inst1, FID_BASE, sizeof(int));
  dsts[0].set_field(inst2, FID_BASE, sizeof(int));
  Event ev3 = isect.copy(srcs, dsts, ProfilingRequestSet(),
                         Event::merge_events(fill_event, isect_event));

  ev3.wait();

  GenericAccessor<int, 1, int> acc(inst2, FID_BASE);
  for (IndexSpaceIterator<1, int> it(isect); it.valid; it.step()) {
    for (PointInRectIterator<1, int> it2(it.rect); it2.valid; it2.step()) {
      assert(acc[it2.p] == 7);
    }
  }

  inst2.destroy();
  inst1.destroy();
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int_units("-bsrc", SampleConfig::src_buffer_size, 'M'),
  cp.add_option_int_units("-bdst", SampleConfig::dst_buffer_size, 'M');
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();

  if (!p.exists()) {
    p = Machine::ProcessorQuery(Machine::get_machine()).first();
  }

  assert(p.exists());

  Processor::register_task_by_kind(p.kind(), false /*!global*/, MAIN_TASK,
                                   CodeDescriptor(main_task),
                                   ProfilingRequestSet())
      .external_wait();

  Event e = rt.collective_spawn(p, MAIN_TASK, 0, 0);
  rt.shutdown(e);
  rt.wait_for_shutdown();
  return 0;
}




