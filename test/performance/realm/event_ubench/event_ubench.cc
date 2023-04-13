/* Copyright 2023 Stanford University
 * Copyright 2023 NVIDIA Corp
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

#include <iomanip>
#include <iostream>
#include <realm.h>
#include <realm/cmdline.h>

using namespace Realm;

Logger log_app("app");

enum
{
  BENCH_LAUNCHER_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  BENCH_SETUP_FAN_TASK,
  BENCH_SETUP_CHAIN_TASK,
  BENCH_TIMING_TASK
};

enum TestFlags
{
  EVENT_TEST = 1 << 0,
  FAN_TEST = 1 << 1,
  CHAIN_TEST = 1 << 2
};

struct BenchLauncherTaskArgs {
  uint64_t enabled_tests = 0;
  bool measure_latency = false;
  size_t min_usecs = 1000000;
  size_t num_samples = 0;
  size_t min_test_size = 1024;
  size_t max_test_size = 1024;
};

struct BenchSetupFanTaskArgs {
  size_t num_samples;
  size_t num_events_per_sample;
  struct SampleEventPair {
    Event start_event;
    UserEvent wait_event;
  } events[1];
};

struct BenchSetupChainTaskArgs {
  size_t num_samples;
  size_t num_events_per_sample;
  UserEvent chain_events[1];
};

struct BenchTimingTaskArgs {
  uint64_t enabled_tests;
  bool measure_latency;
  size_t min_usecs;
  size_t num_samples;
  size_t min_num_events;
  size_t max_num_events;
  Processor dst_proc;
};

static void display_processor_info(Processor p) {}

// We want the fanned out events to be allocated on the remote processor for every sample,
// so setup the measurement DAGs and the start and end event for each sample DAG, but then
// spawn the fan eetup task to fill in the fan-out events.  This allows us to measure how
// fanned out events communicate across processors (it should be just one active message
// per sample if the processor is not in the local address space)
static void setup_fan_test(const BenchTimingTaskArgs &src_args, size_t num_samples,
                           Processor current_processor, size_t num_events,
                           UserEvent &trigger_event, Event &wait_event)
{
  std::vector<char> task_arg_buffer(sizeof(BenchSetupFanTaskArgs) +
                                    (num_samples - 1) *
                                        sizeof(BenchSetupFanTaskArgs::SampleEventPair));
  BenchSetupFanTaskArgs &task_args =
      *reinterpret_cast<BenchSetupFanTaskArgs *>(task_arg_buffer.data());
  std::vector<Event> event_array(num_samples, Event::NO_EVENT);

  trigger_event = UserEvent::create_user_event();
  Event sample_trigger_event = trigger_event;

  task_args.num_samples = num_samples;
  task_args.num_events_per_sample = num_events;
  for(size_t j = 0; j < num_samples; j++) {
    UserEvent start_sample_event = UserEvent::create_user_event();
    task_args.events[j].start_event = start_sample_event;
    task_args.events[j].wait_event = UserEvent::create_user_event();

    start_sample_event.trigger(sample_trigger_event);
    event_array[j] = task_args.events[j].wait_event;
    if(src_args.measure_latency)
      sample_trigger_event = task_args.events[j].wait_event;
  }
  wait_event = Event::merge_events(event_array);

  // Pass all the sample events to the setup task and have it set up the fan DAG
  src_args.dst_proc.spawn(BENCH_SETUP_FAN_TASK, &task_args, task_arg_buffer.size())
      .wait();
}

static void setup_chain_test(const BenchTimingTaskArgs &src_args, size_t num_samples,
                             Processor current_processor, size_t chain_length,
                             UserEvent &trigger_event, Event &wait_event)
{
  std::vector<char> task_arg_buffer(sizeof(BenchSetupChainTaskArgs) +
                                    (src_args.num_samples * chain_length - 1) *
                                        sizeof(UserEvent));
  BenchSetupChainTaskArgs &task_args =
      *reinterpret_cast<BenchSetupChainTaskArgs *>(task_arg_buffer.data());

  trigger_event = UserEvent::create_user_event();
  Event sample_trigger_event = trigger_event;

  std::vector<Event> end_of_samples_events(num_samples, Event::NO_EVENT);

  task_args.num_samples = num_samples;
  task_args.num_events_per_sample = chain_length;

  for(size_t i = 0; i < num_samples; i++) {
    for(size_t j = 0; j < chain_length; j++) {
      task_args.chain_events[i * chain_length + j] = UserEvent::create_user_event();
    }

    task_args.chain_events[i * chain_length].trigger(sample_trigger_event);
    end_of_samples_events[i] = task_args.chain_events[(i + 1) * chain_length - 1];
    if(src_args.measure_latency)
      sample_trigger_event = task_args.chain_events[(i + 1) * chain_length - 1];
  }
  wait_event = Event::merge_events(end_of_samples_events);

  // Pass all the sample events to the setup task and have it set up the fan DAG
  src_args.dst_proc.spawn(BENCH_SETUP_CHAIN_TASK, &task_args, task_arg_buffer.size())
      .wait();
}

static void report_timing(float usecs, bool measure_latency, size_t num_samples,
                          size_t num_events_per_sample)
{
  if(measure_latency)
    log_app.print() << '\t' << (usecs) / num_samples << " us (" << num_samples
                    << " samples)";
  else
    log_app.print() << '\t' << std::scientific << std::setprecision(2)
                    << (num_samples * num_events_per_sample * 1e6) / usecs
                    << " events/s (" << num_samples << " samples, total=" << usecs << " us)";
}

static double time_dag(UserEvent &trigger_event, Event &wait_event)
{
  double start_time = Clock::current_time_in_microseconds();
  trigger_event.trigger();
  wait_event.wait();
  double end_time = Clock::current_time_in_microseconds();
  return end_time - start_time;
}

//
// Each sample is set up as follows:
/* clang-format off */
// FAN: (for num_events_per_sample number of inner_events)
//                      +--> inner_event --+
// sample_start_event --+--> inner_event --+--> sample_finish_event
//                      +--> inner_event --+
// CHAIN: (where N is the chain depth)
// start_sample_event --> local_event1 --> remote_event1 --> local_event2 --> remote_event2 --> ... -> local_eventN sample_finish_event
/* clang-format on */

// This task just sets up the triggering events and spawns the task to set up the DAG,
// then performs the measurement by triggering and waiting for the DAG to complete and
// aggregates the result.
// We want to spawn a task to setup the DAGs in order to ensure the sub-DAG events are
// local to the target remote processor rather than the local one. The following layout is
// how each test's measurement should be setup

/* clang-format off */
// Bandwidth: (for num_samples number of samples)
//                 +--> SAMPLE --+
// trigger_event --+--> SAMPLE --+--> wait_event
//                 +--> SAMPLE --+
//
// Latency: (for num_samples number of samples)
// trigger_event --> SAMPLE --> SAMPLE --> SAMPLE --> wait_event
/* clang-format on */

static void bench_timing_task(const void *args, size_t arglen, const void *userdata,
                              size_t userlen, Processor p)
{
  assert(arglen == sizeof(BenchTimingTaskArgs));
  const BenchTimingTaskArgs &src_args = *static_cast<const BenchTimingTaskArgs *>(args);
  UserEvent trigger_event;
  Event wait_event;
  size_t num_samples = src_args.num_samples;

  if(src_args.enabled_tests & FAN_TEST) {
    for(size_t i = src_args.min_num_events; i <= src_args.max_num_events; i<<=1) {
      double usecs = 0.0;
      if(src_args.num_samples == 0) {
        if(src_args.min_usecs > 0) {
          setup_fan_test(src_args, 10, p, i, trigger_event, wait_event);
          usecs = time_dag(trigger_event, wait_event);
          if(usecs < src_args.min_usecs) {
            // Dynamically figure out the number of samples to fill the minimum time for
            // this test
            num_samples = std::max(10.0, (10.0 * src_args.min_usecs) / usecs);
          }
        }
      }
      setup_fan_test(src_args, num_samples, p, i, trigger_event, wait_event);
      usecs = time_dag(trigger_event, wait_event);
      log_app.print() << "Fan test " << p << "->" << src_args.dst_proc;
      report_timing(usecs, src_args.measure_latency, num_samples, i + 2);
    }
  } else if(src_args.enabled_tests & CHAIN_TEST) {
    for(size_t i = src_args.min_num_events; i <= src_args.max_num_events; i<<=1) {
      setup_chain_test(src_args, num_samples, p, i, trigger_event, wait_event);
      double usecs = time_dag(trigger_event, wait_event);
      log_app.print() << "Chain test " << p << "->" << src_args.dst_proc;
      report_timing(usecs, src_args.measure_latency, num_samples, 2 * i + 2);
    }
  }
}

static void bench_setup_fan_task(const void *args, size_t arglen, const void *userdata,
                                 size_t userlen, Processor p)
{
  const BenchSetupFanTaskArgs &src_args =
      *static_cast<const BenchSetupFanTaskArgs *>(args);
  assert(arglen ==
         sizeof(src_args) + (src_args.num_samples - 1) * sizeof(src_args.events));
  std::vector<Event> events(src_args.num_events_per_sample, Event::NO_EVENT);
  for(size_t i = 0; i < src_args.num_samples; i++) {
    for(size_t j = 0; j < src_args.num_events_per_sample; j++) {
      UserEvent e = UserEvent::create_user_event();
      e.trigger(src_args.events[i].start_event);
      events[j] = e;
    }
    src_args.events[i].wait_event.trigger(Event::merge_events(events));
  }
}

static void bench_setup_chain_task(const void *args, size_t arglen, const void *userdata,
                                   size_t userlen, Processor p)
{
  const BenchSetupChainTaskArgs &src_args =
      *static_cast<const BenchSetupChainTaskArgs *>(args);
  assert(arglen ==
         sizeof(src_args) + (src_args.num_samples * src_args.num_events_per_sample - 1) *
                                sizeof(src_args.chain_events));
  for(size_t i = 0; i < src_args.num_samples; i++) {
    for(size_t j = 0; j < src_args.num_events_per_sample - 1; j++) {
      UserEvent chain_link_event = UserEvent::create_user_event();
      chain_link_event.trigger(
          src_args.chain_events[i * src_args.num_events_per_sample + j]);
      src_args.chain_events[i * src_args.num_events_per_sample + j + 1].trigger(
          chain_link_event);
    }
  }
}

// This task launches a task for each pair of processors in the machine.
static void bench_launcher(const void *args, size_t arglen, const void *userdata,
                           size_t userlen, Processor p)
{
  BenchTimingTaskArgs task_args;
  Event e = Event::NO_EVENT;
  const BenchLauncherTaskArgs &src_args =
      *static_cast<const BenchLauncherTaskArgs *>(args);

  assert(arglen == sizeof(BenchLauncherTaskArgs));

  Machine::ProcessorQuery q1 =
      Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  for(Processor p : q1) {
    display_processor_info(p);
  }

  task_args.enabled_tests = src_args.enabled_tests;
  task_args.num_samples = src_args.num_samples;
  task_args.min_usecs = src_args.min_usecs;
  task_args.min_num_events = src_args.min_test_size;
  task_args.max_num_events = src_args.max_test_size;
  task_args.measure_latency = src_args.measure_latency;

  q1 = Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  Machine::ProcessorQuery q2 =
      Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  for(Processor p1 : q1) {
    for(Processor p2 : q2) {
      task_args.dst_proc = p2;
      e = p1.spawn(BENCH_TIMING_TASK, &task_args, sizeof(task_args), e);
    }
  }

  Runtime::get_runtime().shutdown(e);
}

int main(int argc, char **argv)
{
  Runtime r;
  CommandLineParser cp;

  bool ok = r.init(&argc, &argv);
  assert(ok);

  r.register_task(BENCH_LAUNCHER_TASK, bench_launcher);
  r.register_task(BENCH_SETUP_FAN_TASK, bench_setup_fan_task);
  r.register_task(BENCH_SETUP_CHAIN_TASK, bench_setup_chain_task);
  r.register_task(BENCH_TIMING_TASK, bench_timing_task);

  BenchLauncherTaskArgs args;
  std::vector<std::string> enabled_tests;
  std::string only_kind;

  cp.add_option_int("-s", args.num_samples);
  cp.add_option_stringlist("-t", enabled_tests);
  cp.add_option_int("-m", args.min_test_size);
  cp.add_option_int("-n", args.max_test_size);
  cp.add_option_bool("-L", args.measure_latency);
  ok = cp.parse_command_line(argc, (const char **)argv);

  if (args.min_test_size > args.max_test_size) {
    args.max_test_size = args.min_test_size;
  }

  if(enabled_tests.size() == 0) {
    args.enabled_tests = ~0ULL;
  } else {
    for(size_t i = 0; i < enabled_tests.size(); i++) {
      if(enabled_tests[i] == "FAN")
        args.enabled_tests |= (uint64_t)FAN_TEST;
      else if(enabled_tests[i] == "CHAIN")
        args.enabled_tests |= (uint64_t)CHAIN_TEST;
      else if(enabled_tests[i] == "EVENT")
        args.enabled_tests |= (uint64_t)EVENT_TEST;
      else
        abort();
    }
  }

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();

  r.collective_spawn(p, BENCH_LAUNCHER_TASK, &args, sizeof(args));

  return r.wait_for_shutdown();
}
