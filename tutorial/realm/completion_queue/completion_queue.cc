/* Copyright 2024 NVIDIA Corporation
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

#include <unistd.h>

using namespace Realm;

Logger log_app("app");

enum {
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  WORKER_TASK,
  CLEANCQ_TASK,
};

namespace TestConfig {
  int num_tasks = 100;
};

struct CleanCQTaskArgs {
  int num_tasks;
  unsigned int completion_queue_creation_node;
  CompletionQueue completion_queue;
};

struct WorkerTaskArgs {
  int idx;
};

void worker_task(const void *args, size_t arglen, const void *userdata,
                 size_t userlen, Processor p) 
{
  const WorkerTaskArgs& task_args = *reinterpret_cast<const WorkerTaskArgs *>(args);
  log_app.info("worker %d on proc %llx", task_args.idx, p.id);
  usleep(10000);
}

void cleancq_task(const void *args, size_t arglen, const void *userdata,
                  size_t userlen, Processor p) 
{
  const CleanCQTaskArgs& task_args = *reinterpret_cast<const CleanCQTaskArgs *>(args);
  log_app.print("cleanup on proc %llx", p.id);

  CompletionQueue completion_queue = task_args.completion_queue;

  // the assert fails if CLEANCQ_TASK is run on the rank that is not the one
  // where the completion_queue is created.
  if (task_args.completion_queue_creation_node == p.address_space()) {
    Event nonempty = completion_queue.get_nonempty_event();
    assert(nonempty == Event::NO_EVENT);
  }

  int num_finished_task = 0;
  std::vector<Event> popped(task_args.num_tasks, Event::NO_EVENT);
  while (num_finished_task < task_args.num_tasks) {
    // try to pop an event
    size_t count = completion_queue.pop_events(&popped[0], task_args.num_tasks);
    if(count > 0) {
      log_app.info("clean up %zu tasks", count);
      num_finished_task += count;
    } else {
	    // instead of hammering cq, ask for a nonempty event and wait on it
	    //  before trying again
      Event nonempty = completion_queue.get_nonempty_event();
      nonempty.wait();
    }
  }
}

void main_task(const void *args, size_t arglen, const void *userdata,
               size_t userlen, Processor p) 
{
  Machine machine = Machine::get_machine();
  Machine::ProcessorQuery pq(machine);
  pq = pq.only_kind(p.kind());
  if (pq.count() < 2) {
    log_app.warning("It is better to run this program with at least 2 CPU processors, please specify it through -ll:cpu");
  }

  CompletionQueue completion_queue = CompletionQueue::create_completion_queue(TestConfig::num_tasks);

  // first, test NO_EVENT returned by get_nonempty_event
  Event nonempty = completion_queue.get_nonempty_event();
  WorkerTaskArgs worker_task_args;
  worker_task_args.idx = 0;
  Event e = p.spawn(WORKER_TASK, &worker_task_args, sizeof(WorkerTaskArgs));
  completion_queue.add_event(e);
  nonempty.wait();

  Event nonempty_test = completion_queue.get_nonempty_event();
  assert(nonempty_test == Event::NO_EVENT);
  std::vector<Event> popped(1, Event::NO_EVENT);
  completion_queue.pop_events(&popped[0], 1);
  nonempty_test = completion_queue.get_nonempty_event();
  assert(nonempty_test != Event::NO_EVENT);

  // second, demonstrate how to use CompletionQueue across tasks
  // pick processors for tasks
  Processor cleancq_proc = Processor::NO_PROC;
  std::vector<Processor> worker_procs;
  if (pq.count() < 2) {
    cleancq_proc = p;
    worker_procs.push_back(p);
  } else {
    if (machine.get_address_space_count() == 1) {
      // pick the next processor after p for the CLEANCQ_TASK
      cleancq_proc = *(pq.begin().operator++());
    } else {
      // pick a remote processor for the CLEANCQ_TASK, this is used to demostrate get_nonempty_event on non-owner node
      for(Machine::ProcessorQuery::iterator it = pq.begin(); it; it++) {
        if (it->address_space() != p.address_space()) {
          cleancq_proc = *it;
          break;
        } 
      }
    }
    // use processors other than cleancq_proc for WORKER_TASKs
    for(Machine::ProcessorQuery::iterator it = pq.begin(); it; it++) {
      if ((*it) != cleancq_proc) {
        worker_procs.push_back(*it);
      } 
    }
  }

  // launch worker tasks
  for (int i = 0; i < TestConfig::num_tasks; i++) {
    WorkerTaskArgs worker_task_args;
    worker_task_args.idx = i;
    Event e = worker_procs[i % worker_procs.size()].spawn(WORKER_TASK, &worker_task_args, sizeof(WorkerTaskArgs));
    completion_queue.add_event(e);
  }

  // launch clean completion queue task
  nonempty = completion_queue.get_nonempty_event();
  CleanCQTaskArgs cq_task_args;
  cq_task_args.num_tasks = TestConfig::num_tasks;
  cq_task_args.completion_queue = completion_queue;
  cq_task_args.completion_queue_creation_node = p.address_space();
  Event cleancq_e = cleancq_proc.spawn(CLEANCQ_TASK, &cq_task_args, sizeof(CleanCQTaskArgs), nonempty);

  completion_queue.destroy(cleancq_e);
}

int main(int argc, char **argv) 
{
  Runtime rt;
  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int("-nt", TestConfig::num_tasks);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);
  
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   MAIN_TASK,
                                   CodeDescriptor(main_task),
                                   ProfilingRequestSet(),
                                   0, 0).wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   WORKER_TASK,
                                   CodeDescriptor(worker_task),
                                   ProfilingRequestSet(),
                                   0, 0).wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   CLEANCQ_TASK,
                                   CodeDescriptor(cleancq_task),
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
