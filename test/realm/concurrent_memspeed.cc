#include "realm.h"
#include "realm/cmdline.h"
#include "realm/id.h"
#include <cassert>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <time.h>

#include "osdep.h"

using namespace Realm;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  COPYPROF_TASK,
};

enum {
  FID_BASE = 44,
};

struct MemspeedExperiment {
  long long copy_start_time = -1;
  long long copy_end_time = -1;
  long long copied_bytes = 0;
};

struct CopyProfResult {
  long long *copy_start_time;
  long long *copy_end_time;
  long long *copied_bytes;
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
    *(result->copy_start_time) =
        (*(result->copy_start_time) == -1)
            ? timeline.start_time
            : std::min<long long>(*(result->copy_start_time),
                                  timeline.start_time);
    *(result->copy_end_time) = (*(result->copy_end_time)) == -1
                                   ? timeline.complete_time
                                   : ::max<long long>(*(result->copy_end_time),
                                                      timeline.complete_time);

    result->done.trigger();
  } else {
    log_app.fatal() << "no operation timeline in profiling response!";
    assert(0);
  }
  ProfilingMeasurements::OperationMemoryUsage copied_bytes;
  if (resp.get_measurement(copied_bytes)) {
    *(result->copied_bytes) += copied_bytes.size;
  } else {
    log_app.fatal() << "no operation memory usage in profiling response!";
    assert(0);
  }
}

namespace TestConfig {
int dimensions = 1;
size_t x_size = 2000;
size_t y_size = 2000;
size_t z_size = 2000;
size_t x_copy_size_lo = 0;
size_t x_copy_size_hi = 0;
size_t y_copy_size_lo = 0;
size_t y_copy_size_hi = 0;
size_t z_copy_size_lo = 0;
size_t z_copy_size_hi = 0;
size_t max_src_memories = 1;
size_t max_dst_memories = 1;
size_t buffer_size = 64 << 20; // should be bigger than any cache in system
int copy_reps = 1;             // if nonzero, average over #reps copies
int copy_fields = 1;           // number of distinct fields to copy
bool copy_aos = false;         // if true, use an AOS memory layout
};                             // namespace TestConfig

std::set<Processor::Kind> supported_proc_kinds;

template <int N>
Rect<N>
create_boundaries(int dims,
                  const std::vector<std::pair<size_t, size_t>> &dims_sizes) {
  assert(dims <= static_cast<int>(dims_sizes.size()));
  Rect<N> boundaries;
  for (int i = 0; i < dims; i++) {
    assert(dims_sizes[i].second > 0);
    boundaries.lo[i] = dims_sizes[i].first;
    boundaries.hi[i] = dims_sizes[i].second;
  }
  return boundaries;
}

template <int N>
void do_copies(Processor p, const std::vector<Memory> &memories) {
  std::map<FieldID, size_t> field_sizes;
  for (int i = 0; i < TestConfig::copy_fields; i++) {
    field_sizes[FID_BASE + i] = sizeof(void *);
  }

  // TODO(artempriakhin): Add more sanity checks for dimensions.
  Rect<N> boundaries =
      create_boundaries<N>(TestConfig::dimensions, {{0, TestConfig::x_size},
                                                    {0, TestConfig::y_size},
                                                    {0, TestConfig::z_size}});

  Rect<N> copy_boundaries = create_boundaries<N>(
      TestConfig::dimensions,
      {{TestConfig::x_copy_size_lo, TestConfig::x_copy_size_hi > 0
                                        ? TestConfig::x_copy_size_hi
                                        : TestConfig::x_size},
       {TestConfig::y_copy_size_lo, TestConfig::x_copy_size_hi > 0
                                        ? TestConfig::x_copy_size_hi
                                        : TestConfig::x_size},
       {TestConfig::z_copy_size_lo, TestConfig::z_copy_size_hi > 0
                                        ? TestConfig::z_copy_size_hi
                                        : TestConfig::z_size}});

  std::vector<Event> done_events;
  std::vector<RegionInstance> src_instances, dst_instances;
  std::vector<IndexSpace<N>> index_spaces;
  std::vector<std::vector<CopySrcDstField>> src_fields, dst_fields;
  std::vector<ProfilingRequestSet> profile_requests;

  std::vector<MemspeedExperiment> memspeed_experiments(memories.size());
  std::vector<std::thread> threads;

  size_t src_memories = 0;
  for (size_t i = 0; i < memories.size(); i++) {
    Memory m1 = memories[i];

    IndexSpace<N> index_space(boundaries);
    src_instances.push_back(RegionInstance());
    RegionInstance::create_instance(src_instances.back(), m1, index_space,
                                    field_sizes, (TestConfig::copy_aos ? 1 : 0),
                                    ProfilingRequestSet())
        .wait();
    assert(src_instances.back().exists());

    {
      void *fill_value = 0;
      std::vector<CopySrcDstField> srcs(TestConfig::copy_fields);
      for (int i = 0; i < TestConfig::copy_fields; i++)
        srcs[i].set_fill(fill_value);
      std::vector<CopySrcDstField> dsts(TestConfig::copy_fields);
      for (int i = 0; i < TestConfig::copy_fields; i++)
        dsts[i].set_field(src_instances.back(), FID_BASE + i, sizeof(void *));

      index_space.copy(srcs, dsts, ProfilingRequestSet()).wait();
    }

    size_t dst_memories = 0;
    for (size_t j = 0; j < memories.size(); j++) {
      if (i == j) {
        continue;
      }
      Memory m2 = memories[j];

      dst_instances.push_back(RegionInstance());
      RegionInstance::create_instance(
          dst_instances.back(), m2, index_space, field_sizes,
          (TestConfig::copy_aos ? 1 : 0), ProfilingRequestSet())
          .wait();
      assert(dst_instances.back().exists());

      // TODO(artempriakhin): Support various buffer sizes.
      for (int k = 0; k < TestConfig::copy_reps; k++) {

        std::vector<CopySrcDstField> srcs(TestConfig::copy_fields);
        for (int i = 0; i < TestConfig::copy_fields; i++)
          srcs[i].set_field(src_instances.back(), FID_BASE + i, sizeof(void *));
        src_fields.push_back(srcs);
        std::vector<CopySrcDstField> dsts(TestConfig::copy_fields);
        for (int i = 0; i < TestConfig::copy_fields; i++)
          dsts[i].set_field(dst_instances.back(), FID_BASE + i, sizeof(void *));
        dst_fields.push_back(dsts);

        UserEvent done = UserEvent::create_user_event();

        done_events.push_back(done);
        {
          CopyProfResult result;
          result.copy_start_time = &memspeed_experiments[i].copy_start_time;
          result.copy_end_time = &memspeed_experiments[i].copy_end_time;
          result.copied_bytes = &memspeed_experiments[i].copied_bytes;
          result.done = done;
          ProfilingRequestSet prs;
          prs.add_request(p, COPYPROF_TASK, &result, sizeof(CopyProfResult))
              .add_measurement<ProfilingMeasurements::OperationTimeline>()
              .add_measurement<ProfilingMeasurements::OperationMemoryUsage>();
          profile_requests.push_back(prs);
        }
        index_spaces.push_back(IndexSpace<N>(copy_boundaries));
      }

      if (++dst_memories >= TestConfig::max_dst_memories)
        break;
    }
    if (++src_memories >= TestConfig::max_src_memories)
      break;
  }

  // We should probably find a better way to lauch tasks concurrently.
  for (size_t i = 0; i < index_spaces.size(); i++) {
    threads.push_back(std::thread([&, i] {
      index_spaces[i].copy(src_fields[i], dst_fields[i], profile_requests[i]);
    }));
    threads.back().detach();
  }

  Event::merge_events(done_events).wait();

  log_app.info() << "Memspeed Results";
  for (size_t i = 0; i < memspeed_experiments.size(); ++i) {
    long long copy_duration = (memspeed_experiments[i].copy_end_time -
                               memspeed_experiments[i].copy_start_time);
    if (copy_duration == 0)
      continue;
    log_app.print() << "Node=" << memories[i]
                    << " copy_duration=" << copy_duration
                    << " copied_bytes=" << memspeed_experiments[i].copied_bytes
                    << " bandwidth="
                    << (double)memspeed_experiments[i].copied_bytes /
                           copy_duration;
  }

  for (auto &instance : dst_instances) {
    instance.destroy();
  }
  for (auto &instance : src_instances) {
    instance.destroy();
  }
}

void top_level_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p) {
  log_app.print() << "Realm concurrent memory speed test";

  // build the list of memories that we want to test
  std::vector<Memory> memories;
  Machine machine = Machine::get_machine();

  for (Machine::MemoryQuery::iterator it =
           Machine::MemoryQuery(machine).begin();
       it; ++it) {
    Memory m = *it;
    if (m.kind() == Memory::GPU_FB_MEM) {
      memories.push_back(m);
    }
  }

  if (TestConfig::dimensions == 1)
    do_copies<1>(p, memories);
  else if (TestConfig::dimensions == 2)
    do_copies<2>(p, memories);
  else if (TestConfig::dimensions == 3)
    do_copies<3>(p, memories);

  usleep(100000);
}

int main(int argc, char **argv) {
  Runtime rt;

  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int_units("-b", TestConfig::buffer_size, 'M')
      .add_option_int("-reps", TestConfig::copy_reps)
      .add_option_int("-dims", TestConfig::dimensions)
      .add_option_int("-fields", TestConfig::copy_fields)
      .add_option_int("-aos", TestConfig::copy_aos)
      .add_option_int("-x_size", TestConfig::x_size)
      .add_option_int("-y_size", TestConfig::y_size)
      .add_option_int("-z_size", TestConfig::y_size)
      .add_option_int("-x_copy_size_lo", TestConfig::x_copy_size_lo)
      .add_option_int("-x_copy_size_hi", TestConfig::x_copy_size_hi)
      .add_option_int("-y_copy_size_lo", TestConfig::y_copy_size_lo)
      .add_option_int("-y_copy_size_hi", TestConfig::y_copy_size_hi)
      .add_option_int("-z_copy_size_lo", TestConfig::y_copy_size_lo)
      .add_option_int("-z_copy_size_hi", TestConfig::y_copy_size_hi)
      .add_option_int("-max_src_memories", TestConfig::max_src_memories)
      .add_option_int("-max_dst_memories", TestConfig::max_dst_memories);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  Processor::register_task_by_kind(
      Processor::LOC_PROC, false /*!global*/, COPYPROF_TASK,
      CodeDescriptor(copy_profiling_task), ProfilingRequestSet(), 0, 0)
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
