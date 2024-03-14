/* Copyright 2024 Stanford University
 * Copyright 2024 NVIDIA Corp
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

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <realm.h>
#include <realm/cmdline.h>

using namespace Realm;

Logger log_app("app");

enum
{
  BENCH_LAUNCHER_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  BENCH_TIMING_TASK,
  DUMMY_TASK_LAUNCHER,
  DUMMY_TASK
};

struct BenchTimingTaskArgs {
  size_t num_launcher_tasks = 1;
  size_t num_dummy_tasks = 100;
  size_t num_samples = 1;
  size_t arg_size = 0;
  bool chain = false;
  bool test_gpu = false;
  bool use_proc_group = false;
};

struct DummyTaskLauncherArgs {
  UserEvent dummy_task_trigger_event;
  UserEvent dummy_task_wait_event;
  size_t num_child_tasks;
  size_t arg_size;
  bool chain;
};

class Stat {
public:
  Stat()
    : count(0)
    , mean(0.0)
    , sum(0.0)
    , square_sum(0.0)
    , smallest(std::numeric_limits<double>::max())
    , largest(-std::numeric_limits<double>::max())
  {}
  void reset() { *this = Stat(); }
  void sample(double s)
  {
    count++;
    if(s < smallest)
      smallest = s;
    if(s > largest)
      largest = s;
    sum += s;
    double delta0 = s - mean;
    mean += delta0 / count;
    double delta1 = s - mean;
    square_sum += delta0 * delta1;
  }
  unsigned get_count() const { return count; }
  double get_average() const { return mean; }
  double get_sum() const { return sum; }
  double get_stddev() const
  {
    return get_variance() > 0.0 ? std::sqrt(get_variance()) : 0.0;
  }
  double get_variance() const { return square_sum / (count > 2 ? 1 : count - 1); }
  double get_smallest() const { return smallest; }
  double get_largest() const { return largest; }

  friend std::ostream &operator<<(std::ostream &os, const Stat &s);

private:
  unsigned count;
  double mean;
  double sum;
  double square_sum;
  double smallest;
  double largest;
};

std::ostream &operator<<(std::ostream &os, const Stat &s)
{
  return os << std::scientific << std::setprecision(2)
            << s.get_average() /*<< "(+/-" << s.get_stddev() << ')'*/
            << ", MIN=" << s.get_smallest() << ", MAX=" << s.get_largest()
            << ", N=" << s.get_count();
}

#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
void dummy_gpu_task(const void *args, size_t arglen, 
		                const void *userdata, size_t userlen, Processor p);
#endif

static void display_processor_info(Processor p) {}

static void dummy_task(const void *args, size_t arglen, const void *userdata,
                       size_t userlen, Processor p)
{}

static void dummy_task_launcher(const void *args, size_t arglen, const void *userdata,
                                size_t userlen, Processor p)
{
  // Launches N dummy tasks, waiting for the given user event and signaling another user
  // event once complete
  const DummyTaskLauncherArgs &self_args =
      *reinterpret_cast<const DummyTaskLauncherArgs *>(args);
  assert(arglen == sizeof(DummyTaskLauncherArgs));
  std::vector<char> task_args(self_args.arg_size);
  std::vector<Event> task_events(self_args.num_child_tasks, Event::NO_EVENT);
  UserEvent trigger_event = UserEvent::create_user_event();
  Event depends_event = trigger_event;
  for(size_t i = 0; i < task_events.size(); i++) {
    task_events[i] =
        p.spawn(DUMMY_TASK, task_args.data(), task_args.size(), depends_event);
    if(self_args.chain) {
      depends_event = task_events[i];
    }
  }
  self_args.dummy_task_wait_event.trigger(
      self_args.chain ? task_events.back() : Event::merge_events(task_events));
  trigger_event.trigger(self_args.dummy_task_trigger_event);
}

static void bench_timing_task(const void *args, size_t arglen, const void *userdata,
                              size_t userlen, Processor p)
{
  const BenchTimingTaskArgs &self_args =
      *reinterpret_cast<const BenchTimingTaskArgs *>(args);
  assert(arglen == sizeof(BenchTimingTaskArgs));

  Stat spawn_time, completion_time;

  DummyTaskLauncherArgs launcher_args;
  launcher_args.arg_size = self_args.arg_size;
  launcher_args.num_child_tasks = self_args.num_dummy_tasks;
  launcher_args.chain = self_args.chain;

  Processor::Kind proc_kind = Processor::Kind::NO_KIND;
  if (self_args.test_gpu) {
    proc_kind = Processor::TOC_PROC;
  } else {
    proc_kind = Processor::LOC_PROC;
  }

  std::vector<Processor> processors;
  size_t proc_num = 0;

  {
    Machine::ProcessorQuery processors_to_test =
        Machine::ProcessorQuery(Machine::get_machine()).only_kind(proc_kind);

    processors.assign(processors_to_test.begin(), processors_to_test.end());
    proc_num = processors.size();

    if (self_args.use_proc_group) {
      ProcessorGroup proc_group = ProcessorGroup::create_group(processors);
      processors.clear();
      processors.push_back(proc_group);
    }
  }

  std::vector<Event> task_events(self_args.num_launcher_tasks * proc_num,
                                 Event::NO_EVENT);
  std::vector<Event> child_task_events(self_args.num_launcher_tasks * proc_num,
                                       Event::NO_EVENT);

  for(size_t s = 0; s < self_args.num_samples + 1; s++) {
    UserEvent trigger_event = UserEvent::create_user_event();
    launcher_args.dummy_task_trigger_event = UserEvent::create_user_event();

    for(size_t p = 0; p < proc_num; p++) {
      Processor target_processor = processors[p % processors.size()];
      for(size_t t = 0; t < self_args.num_launcher_tasks; t++) {
        launcher_args.dummy_task_wait_event = UserEvent::create_user_event();
        task_events[p * self_args.num_launcher_tasks + t] = target_processor.spawn(
            DUMMY_TASK_LAUNCHER, &launcher_args, sizeof(launcher_args), trigger_event);
        child_task_events[p * self_args.num_launcher_tasks + t] =
            launcher_args.dummy_task_wait_event;
      }
    }

    // Make sure the launcher tasks have completed (their child tasks are all queued)
    Event wait_event = Event::merge_events(task_events);
    {
      // Time the spawn
      size_t start_time = Clock::current_time_in_microseconds();
      trigger_event.trigger();
      wait_event.wait();
      size_t end_time = Clock::current_time_in_microseconds();
      spawn_time.sample(double(proc_num * self_args.num_launcher_tasks *
                               self_args.num_dummy_tasks * 1e6) /
                        double(end_time - start_time));
      log_app.info() << "Spawn sample (us): " << end_time - start_time;
    }

    wait_event = Event::merge_events(child_task_events);
    {
      // Time the completion of the dummy tasks
      size_t start_time = Clock::current_time_in_microseconds();
      launcher_args.dummy_task_trigger_event.trigger();
      wait_event.wait();
      size_t end_time = Clock::current_time_in_microseconds();
      completion_time.sample(double(proc_num * self_args.num_launcher_tasks *
                                    self_args.num_dummy_tasks * 1e6) /
                             double(end_time - start_time));
      log_app.info() << "Completion sample (us): " << end_time - start_time;
    }
    // Warm-up
    if(s == 0) {
      spawn_time.reset();
      completion_time.reset();
    }
  }

  log_app.print() << "Spawn rate (tasks/s): " << spawn_time;
  log_app.print() << "Completion rate (tasks/s): " << completion_time;
  Runtime::get_runtime().shutdown();
}

int main(int argc, char **argv)
{
  Runtime r;
  CommandLineParser cp;

  bool ok = r.init(&argc, &argv);
  assert(ok);

  r.register_task(BENCH_TIMING_TASK, bench_timing_task);
  Processor::register_task_by_kind(Processor::LOC_PROC,
				   false /*!global*/,
				   DUMMY_TASK_LAUNCHER,
				   CodeDescriptor(dummy_task_launcher),
				   ProfilingRequestSet()).wait();
  Processor::register_task_by_kind(Processor::LOC_PROC,
				   false /*!global*/,
				   DUMMY_TASK,
				   CodeDescriptor(dummy_task),
				   ProfilingRequestSet()).wait();
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
  Processor::register_task_by_kind(Processor::TOC_PROC,
				   false /*!global*/,
				   DUMMY_TASK_LAUNCHER,
				   CodeDescriptor(dummy_task_launcher),
				   ProfilingRequestSet()).wait();
  Processor::register_task_by_kind(Processor::TOC_PROC,
				   false /*!global*/,
				   DUMMY_TASK,
				   CodeDescriptor(dummy_gpu_task),
				   ProfilingRequestSet()).wait();
#endif

  BenchTimingTaskArgs args;

  cp.add_option_int("-a", args.arg_size);
  cp.add_option_int("-s", args.num_samples);
  cp.add_option_int("-tpp", args.num_launcher_tasks);
  cp.add_option_int("-n", args.num_dummy_tasks);
  cp.add_option_bool("-c", args.chain);
  cp.add_option_bool("-gpu", args.test_gpu);
  cp.add_option_bool("-g", args.use_proc_group);

  ok = cp.parse_command_line(argc, (const char **)argv);
  assert(ok);

  Machine::ProcessorQuery processors_to_test =
      Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  for(Processor p : processors_to_test) {
    display_processor_info(p);
  }

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();

  r.collective_spawn(p, BENCH_TIMING_TASK, &args, sizeof(args));

  return r.wait_for_shutdown();
}
