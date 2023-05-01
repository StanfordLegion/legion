/* Copyright 2023 NVIDIA Corporation
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

#include "realm.h"
#include "realm/cmdline.h"

using namespace Realm;

Logger log_app("app");

enum {
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  WRITER_TASK1,
  WRITER_TASK2,
  WRITER_INNER_TASK,
  READER_TASK,
};

enum {
  FID_DATA = 100,
};

namespace TestConfig {
  int num_elements = 10;
  int num_readers = 10;
  int num_writers = 10;
  bool distributed_reservation = false; // use reservation across processes
};

struct WriterTaskArgs {
  int idx;
  IndexSpace<1> is;
  RegionInstance inst;
  Reservation reservation;
  Event precond;
};

struct ReaderTaskArgs {
  int idx;
  IndexSpace<1> is;
  RegionInstance inst;
  Barrier reader_barrier;
};

void writer_inner_task(const void *args, size_t arglen, const void *userdata,
                       size_t userlen, Processor p) 
{
  const WriterTaskArgs& task_args = *reinterpret_cast<const WriterTaskArgs *>(args);
  RegionInstance inst;
  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID_DATA] = sizeof(int);
  std::vector<CopySrcDstField> srcs(1), dsts(1);

  if (task_args.inst.address_space() != p.address_space()) {
    Memory cpu_mem = Machine::MemoryQuery(Machine::get_machine()).local_address_space().only_kind(Memory::Kind::SYSTEM_MEM).first();
    assert(cpu_mem.exists());
    Event e1 = task_args.inst.fetch_metadata(Processor::get_executing_processor());
    Event e2 = RegionInstance::create_instance(inst, cpu_mem, task_args.is, field_sizes,
                                               0 /*block_size=SOA*/,
                                               ProfilingRequestSet());
    Event e3 = Event::merge_events(e1, e2);
    srcs[0].set_field(task_args.inst, FID_DATA, sizeof(int));
    dsts[0].set_field(inst, FID_DATA, sizeof(int));
    task_args.is.copy(srcs, dsts, ProfilingRequestSet(), e3).wait();
  } else {
    inst = task_args.inst;
  }
  
  AffineAccessor<int, 1> acc(inst, FID_DATA);
  for(IndexSpaceIterator<1> it(task_args.is); it.valid; it.step()) {
    for(PointInRectIterator<1> it2(it.rect); it2.valid; it2.step()) {
      acc[it2.p] += task_args.idx;
    }
  }

  if (task_args.inst.address_space() != p.address_space()) {
    srcs[0].set_field(inst, FID_DATA, sizeof(int));
    dsts[0].set_field(task_args.inst, FID_DATA, sizeof(int));
    task_args.is.copy(srcs, dsts, ProfilingRequestSet()).wait();
  }
}

void writer_task1(const void *args, size_t arglen, const void *userdata,
                 size_t userlen, Processor p) 
{
  const WriterTaskArgs& task_args = *reinterpret_cast<const WriterTaskArgs *>(args);
  log_app.print("writer1 %d on proc %llx", task_args.idx, p.id);
  
  // this is not task spawner
  writer_inner_task(args, arglen, userdata, userlen, p);
}

void writer_task2(const void *args, size_t arglen, const void *userdata,
                 size_t userlen, Processor p) 
{
  const WriterTaskArgs& task_args = *reinterpret_cast<const WriterTaskArgs *>(args);
  log_app.print("writer2 %d on proc %llx", task_args.idx, p.id);
  
  // acquire the exclusive ownership
  Event e1 = task_args.reservation.acquire();
  Event e2 = p.spawn(WRITER_INNER_TASK, &task_args, sizeof(WriterTaskArgs), e1);
  task_args.reservation.release(e2);
  e2.wait();
}

void reader_task(const void *args, size_t arglen, const void *userdata,
                 size_t userlen, Processor p) 
{
  const ReaderTaskArgs& task_args = *reinterpret_cast<const ReaderTaskArgs *>(args);
  Barrier reader_barrier = task_args.reader_barrier;

  log_app.print("reader %d on proc %llx", task_args.idx, p.id);

  // we use the barrier here to make sure all reader tasks have successfully acquired
  // the reservation
  reader_barrier.arrive(1);
  reader_barrier.wait();

  RegionInstance inst;
  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID_DATA] = sizeof(int);
  std::vector<CopySrcDstField> srcs(1), dsts(1);

  if (task_args.inst.address_space() != p.address_space()) {
    Memory cpu_mem = Machine::MemoryQuery(Machine::get_machine()).local_address_space().only_kind(Memory::Kind::SYSTEM_MEM).first();
    assert(cpu_mem.exists());
    Event e1 = task_args.inst.fetch_metadata(Processor::get_executing_processor());
    Event e2 = RegionInstance::create_instance(inst, cpu_mem, task_args.is, field_sizes,
                                               0 /*block_size=SOA*/,
                                               ProfilingRequestSet());
    Event e3 = Event::merge_events(e1, e2);
    srcs[0].set_field(task_args.inst, FID_DATA, sizeof(int));
    dsts[0].set_field(inst, FID_DATA, sizeof(int));
    task_args.is.copy(srcs, dsts, ProfilingRequestSet(), e3).wait();
  } else {
    inst = task_args.inst;
  }
  
  // expected1 is the case the reader tasks run before the writer task,
  // expected2 is the case the reader tasks run after the writer task.
  int expected1 = (TestConfig::num_writers - 1) * TestConfig::num_writers / 2;
  int expected2 = expected1 + TestConfig::num_writers;
  int expected = 0;
  AffineAccessor<int, 1> acc(inst, FID_DATA);
  for(IndexSpaceIterator<1> it(task_args.is); it.valid; it.step()) {
    for(PointInRectIterator<1> it2(it.rect); it2.valid; it2.step()) {
      // set the expected value, since reader tasks hold the reservation, 
      // the expected value should be fixed. 
      if (!expected) {
        if (acc[it2.p] == expected1) {
          expected = expected1;
        } else if (acc[it2.p] == expected2) {
          expected = expected2;
        }
      } else {
        if (acc[it2.p] != expected) {
          log_app.error("error result actual: %d, expected: %d\n", acc[it2.p], expected);
          assert (0);
        }
      }
    }
  }
}

void main_task(const void *args, size_t arglen, const void *userdata,
               size_t userlen, Processor p) 
{
  // create an instance
  Rect<1> rect(0, TestConfig::num_elements);
  IndexSpace<1> is(rect);
  RegionInstance inst;
  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID_DATA] = sizeof(int);
  Memory cpu_mem = Machine::MemoryQuery(Machine::get_machine()).local_address_space().only_kind(Memory::Kind::SYSTEM_MEM).first();
  assert(cpu_mem.exists());
  RegionInstance::create_instance(inst, cpu_mem, is, field_sizes,
                                  0 /*block_size=SOA*/,
                                  ProfilingRequestSet()).wait();
  // init the instance with 0
  int fill_value = 0;
  std::vector<CopySrcDstField> sdf(1);
  sdf[0].inst = inst;
  sdf[0].field_id = FID_DATA;
  sdf[0].size = sizeof(int);
  is.fill(sdf, ProfilingRequestSet(), &fill_value, sizeof(fill_value)).wait();

  // create a list of cpus for launch tasks
  Machine::ProcessorQuery pq = Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  std::vector<Processor> cpus;
  for (Machine::ProcessorQuery::iterator it = pq.begin(); it; ++it) {
    Processor p = *it;
    cpus.push_back(p);
  }

  // create reservations
  Reservation reservation = Reservation::create_reservation();

  UserEvent start_event = UserEvent::create_user_event();
  
  // launch writer tasks
  std::vector<Event> task_events;
  for (int i = 0; i < TestConfig::num_writers; i++) {
    WriterTaskArgs task_args;
    task_args.idx = i;
    task_args.is = is;
    task_args.inst = inst;
    Event e;
    if (TestConfig::distributed_reservation) {
      task_args.reservation = reservation;
      e = cpus[i % cpus.size()].spawn(WRITER_TASK2, &task_args, sizeof(WriterTaskArgs), start_event);
    } else {
      Event e1 = reservation.acquire(0, true, start_event);
      e = cpus[i % cpus.size()].spawn(WRITER_TASK1, &task_args, sizeof(WriterTaskArgs), e1);
      reservation.release(e);
    }
    task_events.push_back(e);
  }
  Event writer_event = Event::merge_events(task_events);
  
  // launch reader task
  task_events.clear();
  Barrier reader_barrier = Barrier::create_barrier(TestConfig::num_readers);
  for (int i = 0; i < TestConfig::num_readers; i++) {
    ReaderTaskArgs task_args;
    task_args.idx = i;
    task_args.is = is;
    task_args.inst = inst;
    task_args.reader_barrier = reader_barrier;
    Event e1 = reservation.acquire(1, false, writer_event);
    Event e2 = cpus[i % cpus.size()].spawn(READER_TASK, &task_args, sizeof(ReaderTaskArgs), e1);
    reservation.release(e2);
    task_events.push_back(e2);
  }
  // launch another writer task to compete reservation with reader tasks
  {
    WriterTaskArgs task_args;
    task_args.idx = TestConfig::num_readers;
    task_args.is = is;
    task_args.inst = inst;
    Event e1 = reservation.acquire(0, true, writer_event);
    Event e2 = cpus[TestConfig::num_readers % cpus.size()].spawn(WRITER_TASK1, &task_args, sizeof(WriterTaskArgs), e1);
    reservation.release(e2);
    task_events.push_back(e2);
  }
  Event reader_event = Event::merge_events(task_events);
  start_event.trigger();
  reader_event.wait();

  // check results
  int expected = 0;
  expected = (TestConfig::num_writers + 1) * TestConfig::num_writers / 2;
  AffineAccessor<int, 1> acc(inst, FID_DATA);
  for(IndexSpaceIterator<1> it(is); it.valid; it.step()) {
    for(PointInRectIterator<1> it2(it.rect); it2.valid; it2.step()) {
      if (acc[it2.p] != expected) {
        log_app.error("error result actual: %d, expected: %d\n", acc[it2.p], expected);
        assert (0);
      }
    }
  }
  log_app.print("Success");

  // clean up
  inst.destroy();
  reservation.destroy_reservation();
}

int main(int argc, char **argv) 
{
  Runtime rt;
  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int("-ne", TestConfig::num_elements);
  cp.add_option_int("-nw", TestConfig::num_writers);
  cp.add_option_int("-nr", TestConfig::num_readers);
  cp.add_option_bool("-d", TestConfig::distributed_reservation);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);
  
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   MAIN_TASK,
                                   CodeDescriptor(main_task),
                                   ProfilingRequestSet(),
                                   0, 0).wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   WRITER_TASK1,
                                   CodeDescriptor(writer_task1),
                                   ProfilingRequestSet(),
                                   0, 0).wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   WRITER_TASK2,
                                   CodeDescriptor(writer_task2),
                                   ProfilingRequestSet(),
                                   0, 0).wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   WRITER_INNER_TASK,
                                   CodeDescriptor(writer_inner_task),
                                   ProfilingRequestSet(),
                                   0, 0).wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   READER_TASK,
                                   CodeDescriptor(reader_task),
                                   ProfilingRequestSet(),
                                   0, 0).wait();

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC)
    .first();
  assert(p.exists());

  Event e = rt.collective_spawn(p, MAIN_TASK, 0, 0);
  rt.shutdown(e);
  int ret = rt.wait_for_shutdown();

  return ret;
}
