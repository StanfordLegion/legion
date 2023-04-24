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

#include <assert.h>
#include <unistd.h>
#include <syscall.h>

using namespace Realm;

Logger log_app("app");

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  WRITER_TASK,
  READER_TASK,
};

enum { REDOP_ADD = 1 };

class ReductionOpIntAdd {
public:
  typedef int LHS;
  typedef int RHS;

  template <bool EXCL>
  static void apply(LHS& lhs, RHS rhs) { lhs += rhs; }

  static const RHS identity;

  template <bool EXCL>
  static void fold(RHS& rhs1, RHS rhs2) { rhs1 += rhs2; }
};

const ReductionOpIntAdd::RHS ReductionOpIntAdd::identity = 0;

struct TaskArgs {
  int num_iters;
  int num_tasks;
  int idx;
  Barrier writer_barrier;
  Barrier reader_barrier;
};

namespace TestConfig {
  int num_writers = 4;
  int num_readers = 3;
  int num_iters = 4;
};


void writer_task(const void *args, size_t arglen, 
                 const void *userdata, size_t userlen, Processor p)
{
  TaskArgs task_args = *(const TaskArgs *)args;
  int idx = task_args.idx;
  int num_iters = task_args.num_iters;
  Barrier writer_b = task_args.writer_barrier;
  Barrier reader_b = task_args.reader_barrier;
  log_app.print("start writer task %d on Processor %llx, tid %ld", idx, p.id, syscall(SYS_gettid));
  for (int i = 0; i < num_iters; i++) {
    usleep(10000);
    int reduce_val = (i + 1) * idx;
    writer_b.arrive(1, Event::NO_EVENT, &reduce_val, sizeof(reduce_val));
    log_app.info("writer %d finishes iter %d", idx, i);
    reader_b.wait();
    writer_b = writer_b.advance_barrier();
    reader_b = reader_b.advance_barrier();
  }
}

void reader_task(const void *args, size_t arglen, 
                 const void *userdata, size_t userlen, Processor p)
{
  TaskArgs task_args = *(const TaskArgs *)args;
  int idx = task_args.idx;
  int num_iters = task_args.num_iters;
  int num_tasks = task_args.num_tasks;
  Barrier writer_b = task_args.writer_barrier;
  Barrier reader_b = task_args.reader_barrier;
  log_app.print("start reader task %d on Processor %llx, tid %ld", idx, p.id, syscall(SYS_gettid));
  for (int i = 0; i < num_iters; i++) {
    writer_b.wait();
    int result = 0;
    bool ready = writer_b.get_result(&result, sizeof(result));
    assert(ready);
    int expected_result = (i + 1) * ((num_tasks - 1) * num_tasks / 2);
    assert (expected_result == result);
    log_app.info("reader idx %d, iter %d, result %d", idx, i, result);
    reader_b.arrive(1);
    writer_b = writer_b.advance_barrier();
    reader_b = reader_b.advance_barrier();
  }
}

void main_task(const void *args, size_t arglen, 
                    const void *userdata, size_t userlen, Processor p)
{
  log_app.print("start top task on Processor %llx, tid %ld", p.id, syscall(SYS_gettid));

  Machine machine = Machine::get_machine();
  Machine::ProcessorQuery pq = Machine::ProcessorQuery(machine).only_kind(Processor::LOC_PROC);
  std::vector<Processor> writer_cpus, reader_cpus, cpus;
  if (pq.count() >= static_cast<size_t>(2 + TestConfig::num_readers)) {
    // we reserve the 1st for top level task and 2nd to 2nd + TestConfig::num_readers for reader tasks
    // and the rest of them for writer tasks
    int idx = 0;
    for (Machine::ProcessorQuery::iterator it = pq.begin(); it; it++) {
      if (*it == p) continue;
      if (idx < TestConfig::num_readers) {
        reader_cpus.push_back(*it);
      } else {
        writer_cpus.push_back(*it);
      }
      idx ++;
    }
  } else {
    log_app.warning("The number of CPU processor required = number of readers + writers + 1, please specify it through -ll:cpu");
    reader_cpus.push_back(p);
    writer_cpus.push_back(p);
  }

  double t_start = Clock::current_time();

  // create barriers
  int init_value = 0;
  Barrier writer_barrier = Barrier::create_barrier(TestConfig::num_writers, REDOP_ADD,
                                                   &init_value, sizeof(init_value));
  Barrier reader_barrier = Barrier::create_barrier(TestConfig::num_readers);

  // spawn writer tasks
  std::vector<Event> task_events;
  for (int i = 0; i < TestConfig::num_writers; i++) {
    TaskArgs task_args;
    task_args.writer_barrier = writer_barrier;
    task_args.reader_barrier = reader_barrier;
    task_args.idx = i;
    task_args.num_iters = TestConfig::num_iters;
    task_args.num_tasks = TestConfig::num_writers;
    Event e = writer_cpus[i % writer_cpus.size()].spawn(WRITER_TASK, &task_args, sizeof(TaskArgs));
    task_events.push_back(e);
  }

  // spawn reader tasks
  for (int i = 0; i < TestConfig::num_readers; i++) {
    TaskArgs task_args;
    task_args.writer_barrier = writer_barrier;
    task_args.reader_barrier = reader_barrier;
    task_args.idx = i;
    task_args.num_iters = TestConfig::num_iters;
    task_args.num_tasks = TestConfig::num_writers;
    Event e = reader_cpus[i % reader_cpus.size()].spawn(READER_TASK, &task_args, sizeof(TaskArgs));
    task_events.push_back(e);
  }

  Event e = Event::merge_events(task_events);
  e.wait();
  double t_end = Clock::current_time();
  log_app.print("Total time %f(s)", t_end - t_start);
}

int main(int argc, char **argv) 
{
  Runtime rt;

  rt.init(&argc, (char ***)&argv);
  
  CommandLineParser cp;
  cp.add_option_int("-nw", TestConfig::num_writers);
  cp.add_option_int("-nr", TestConfig::num_readers);
  cp.add_option_int("-ni", TestConfig::num_iters);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  rt.register_reduction<ReductionOpIntAdd>(REDOP_ADD);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, MAIN_TASK,
                                   CodeDescriptor(main_task),
                                   ProfilingRequestSet()).external_wait();

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, WRITER_TASK,
                                  CodeDescriptor(writer_task),
                                  ProfilingRequestSet()).external_wait();

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, READER_TASK,
                                  CodeDescriptor(reader_task),
                                  ProfilingRequestSet()).external_wait();

  Event e = rt.collective_spawn(p, MAIN_TASK, 0, 0);

  rt.shutdown(e);

  int ret = rt.wait_for_shutdown();

  return ret;
}
