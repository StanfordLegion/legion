/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#include <realm.h>
#include <realm/cmdline.h>

using namespace Realm;

enum {
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  READER_TASK,
  WRITER_TASK,
};

enum {
  FID1 = 101,
  FID2 = 102,
};

Logger log_app("app");

struct InstanceLogicalLayout1 {
  int x;
  float y;
};

struct InstanceLogicalLayout2 {
  long long x;
  double y;
};

#define MAX_BATCH_COUNT 2

namespace ExampleConfig {
size_t num_batches = MAX_BATCH_COUNT;
size_t batch_inst_bytes = 1 << 20;
};  // namespace ExampleConfig

struct TaskArguments {
  Rect<1> bounds;
  RegionInstance inst;
};

template <typename FT, typename T0, typename T1>
void update(RegionInstance inst, Rect<1, int> bounds, FieldID fid, int add) {
  AffineAccessor<FT, 1, int> accessor(inst, fid);
  PointInRectIterator<1, int> pit(bounds);
  while (pit.valid) {
    accessor[pit.p].x = static_cast<T0>(pit.p[0] + add);
    accessor[pit.p].y = static_cast<T1>(pit.p[0] + add + 1);
    pit.step();
  }
}

template <typename FT, typename T0, typename T1>
void verify(RegionInstance inst, Rect<1, int> bounds, FieldID fid, int add) {
  AffineAccessor<FT, 1, int> accessor(inst, fid);
  PointInRectIterator<1, int> pit(bounds);
  while (pit.valid) {
    assert(accessor[pit.p].x == static_cast<T0>(pit.p[0] + add));
    assert(accessor[pit.p].y == static_cast<T1>(pit.p[0] + add + 1));
    log_app.info() << "p=" << pit.p << " x=" << accessor[pit.p].x
                   << " y=" << accessor[pit.p].y;
    pit.step();
  }
}

void reader_task(const void *args, size_t arglen, const void *userdata,
                 size_t userlen, Processor p) {
  const TaskArguments &task_args =
      *reinterpret_cast<const TaskArguments *>(args);
  verify<InstanceLogicalLayout1, int, float>(task_args.inst, task_args.bounds,
                                             FID1, /*add=*/1);
  verify<InstanceLogicalLayout2, long long, double>(task_args.inst,
                                                    task_args.bounds, FID2,
                                                    /*add=*/2);
}

void writer_task(const void *args, size_t arglen, const void *userdata,
                 size_t userlen, Processor p) {
  const TaskArguments &task_args =
      *reinterpret_cast<const TaskArguments *>(args);
  update<InstanceLogicalLayout1, int, float>(task_args.inst, task_args.bounds,
                                             FID1, /*add=*/1);
  update<InstanceLogicalLayout2, long long, double>(
      task_args.inst, task_args.bounds, FID2, /*add=*/2);
}

void main_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p) {
  std::vector<Memory> memories;
  Machine::MemoryQuery mq(Machine::get_machine());
  mq.only_kind(Memory::SYSTEM_MEM).has_capacity(1).has_affinity_to(p);
  memories.assign(mq.begin(), mq.end());
  assert(!memories.empty());

  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID1] = sizeof(InstanceLogicalLayout1);
  field_sizes[FID2] = sizeof(InstanceLogicalLayout2);

  size_t inst_elems =
      ExampleConfig::batch_inst_bytes / (field_sizes[FID1] + field_sizes[FID2]);
  Rect<1> bounds = Rect<1>(Point<1>(0), Point<1>(inst_elems));

  UserEvent start_event = UserEvent::create_user_event();

  Event prev_event = start_event;
  for (size_t batch_idx = 0; batch_idx < ExampleConfig::num_batches;
       batch_idx++) {
    RegionInstance inst;
    Event create_event = RegionInstance::create_instance(
        inst, memories.front(), bounds,
        field_sizes, 1, ProfilingRequestSet(), prev_event);

    TaskArguments task_args{bounds, inst};
    prev_event = p.spawn(
        READER_TASK, &task_args, sizeof(TaskArguments),
        p.spawn(WRITER_TASK, &task_args, sizeof(TaskArguments), create_event));
    inst.destroy(prev_event);
  }

  start_event.trigger();
  prev_event.wait();
  Runtime::get_runtime().shutdown(Processor::get_current_finish_event(),
                                  /*success=*/0);
}

int main(int argc, const char **argv) {
  Runtime rt;
  rt.init(&argc, (char ***)&argv);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();

  Processor::register_task_by_kind(p.kind(), false /*!global*/, MAIN_TASK,
                                   CodeDescriptor(main_task),
                                   ProfilingRequestSet())
      .external_wait();
  Processor::register_task_by_kind(p.kind(), false /*!global*/, READER_TASK,
                                   CodeDescriptor(reader_task),
                                   ProfilingRequestSet())
      .external_wait();
  Processor::register_task_by_kind(p.kind(), false /*!global*/, WRITER_TASK,
                                   CodeDescriptor(writer_task),
                                   ProfilingRequestSet())
      .external_wait();

  rt.collective_spawn(p, MAIN_TASK, 0, 0);
  return rt.wait_for_shutdown();
}
