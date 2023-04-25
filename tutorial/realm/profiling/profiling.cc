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

using namespace Realm;

Logger log_app("app");

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  COMPUTE_PROF_TASK,
  COMPUTE_TASK,
  COPY_PROF_TASK,
  INST_PROF_TASK,
};

enum {
  FID_BASE = 44,
};

namespace TestConfig {
  int num_iterations = 2;
  int num_tasks = 50;
};

struct ComputeTaskArgs {
  int cpu_idx;
  int idx;
};

struct ComputeProfResult {
  // performance metrics from OperationTimeline
  long long ready_time;
  long long start_time;
  long long complete_time;
  long long wait_time;
  // performance metrics from OperationProcessorUsage
  Processor proc;
};

struct ComputeProfResultWrapper {
  struct ComputeProfResult *metrics;
  int cpu_idx;
  int idx;
  UserEvent done;
};

struct CopyProfResult {
  // performance metrics from OperationTimeline
  long long start_time;
  long long complete_time;
  // performance metrics from OperationCopyInfo
  unsigned long long src_inst_id;
  unsigned long long dst_inst_id;
  int request_type;
  unsigned int num_hops;
  // performance metrics from OperationMemoryUsage
  Memory src_mem;
  Memory dst_mem;
  size_t size;
};

struct CopyProfResultWrapper {
  struct CopyProfResult *metrics;
  UserEvent done;
};

struct InstProfResult {
  // performance metrics from InstanceTimeline
  long long create_time;
  long long ready_time;
  long long delete_time;
  // performance metrics from InstanceMemoryUsage
  Memory memory;
  size_t bytes;
};

struct InstProfResultWrapper {
  struct InstProfResult *metrics;
  UserEvent done;
};

void compute_prof_task(const void *args, size_t arglen, 
                       const void *userdata, size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(ComputeProfResultWrapper));
  const ComputeProfResultWrapper *result = static_cast<const ComputeProfResultWrapper *>(resp.user_data());

  ComputeProfResult* const metrics = result->metrics;

  ProfilingMeasurements::OperationTimeline timeline;
  if(resp.get_measurement(timeline)) {
    metrics->ready_time = timeline.ready_time;
    metrics->start_time = timeline.start_time;
    metrics->complete_time = timeline.complete_time;
  }
  ProfilingMeasurements::OperationProcessorUsage processor_usage;
  if(resp.get_measurement(processor_usage)) {
    metrics->proc = processor_usage.proc;
  }
  result->done.trigger();
}

void copy_prof_task(const void *args, size_t arglen, 
                    const void *userdata, size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(CopyProfResultWrapper));
  const CopyProfResultWrapper *result = static_cast<const CopyProfResultWrapper *>(resp.user_data());

  CopyProfResult* const metrics = result->metrics;

  ProfilingMeasurements::OperationTimeline timeline;
  if(resp.get_measurement(timeline)) {
    metrics->start_time = timeline.start_time;
    metrics->complete_time = timeline.complete_time;
  }
  ProfilingMeasurements::OperationCopyInfo copy_info;
  if (resp.get_measurement(copy_info)) {
    metrics->src_inst_id = copy_info.inst_info[0].src_inst_id.id;
    metrics->dst_inst_id = copy_info.inst_info[0].dst_inst_id.id;
    metrics->request_type = copy_info.inst_info[0].request_type;
    metrics->num_hops = copy_info.inst_info[0].num_hops;
  }
  ProfilingMeasurements::OperationMemoryUsage memory_usage;
  if (resp.get_measurement(memory_usage)) {
    metrics->src_mem = memory_usage.source;
    metrics->dst_mem = memory_usage.target;
    metrics->size = memory_usage.size;
  }
  result->done.trigger();
}

void inst_prof_task(const void *args, size_t arglen, 
                    const void *userdata, size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(InstProfResultWrapper));
  const InstProfResultWrapper *result = static_cast<const InstProfResultWrapper *>(resp.user_data());

  InstProfResult* const metrics = result->metrics;

  ProfilingMeasurements::InstanceTimeline timeline;
  if(resp.get_measurement(timeline)) {
    metrics->create_time = timeline.create_time;
    metrics->ready_time = timeline.ready_time;
    metrics->delete_time = timeline.delete_time;
  }
  ProfilingMeasurements::InstanceMemoryUsage memory_usage;
  if(resp.get_measurement(memory_usage)) {
    metrics->memory = memory_usage.memory;
    metrics->bytes = memory_usage.bytes;
  }
  result->done.trigger();
}

void compute_task(const void *args, size_t arglen, 
                  const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(ComputeTaskArgs));
  const ComputeTaskArgs *task_args = static_cast<const ComputeTaskArgs *>(args);
  log_app.info("compute_task %d is executed on Processor %llx, idx %d", task_args->idx, p.id, task_args->cpu_idx);
  usleep(task_args->idx % 4 * 1000);
}

int find_the_best_cpu(std::vector<std::pair<int, double>> &cpu_status, std::vector<double> &batch_task_runtime, int task_idx, bool use_profile)
{
  // if no profile is used, we use round-robin
  int idx = task_idx % cpu_status.size();
  double lowest_cpu_runtime = 0;
  if (use_profile) {
    lowest_cpu_runtime = cpu_status[0].second;
    idx = 0;
    for (size_t i = 1; i < cpu_status.size(); i++) {
      if (cpu_status[i].second < lowest_cpu_runtime) {
        lowest_cpu_runtime = cpu_status[i].second;
        idx = i;
      }
    }
    log_app.info("Decision: task %d, cpu %d, cpu runtime time %f", task_idx, idx, lowest_cpu_runtime);
    cpu_status[idx].second += batch_task_runtime[task_idx];
    cpu_status[idx].first ++;
  }
  return idx;
}

void main_task(const void *args, size_t arglen, 
               const void *userdata, size_t userlen, Processor p)
{
  // set the processors for the profiling and worker tasks.
  Machine::ProcessorQuery pq = Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  Processor profile_proc = Processor::NO_PROC;
  std::vector<Processor> worker_procs;
  if (pq.count() < 3) {
    log_app.fatal("It is better to run this program with at least 3 CPU processors and 2 per rank, please specify it through -ll:cpu");
    profile_proc = p;
    worker_procs.push_back(p);
  } else {
    profile_proc = pq.next(p);
    assert (profile_proc.address_space() == p.address_space());
    for(Machine::ProcessorQuery::iterator it = pq.begin(); it; ++it) {
      if ((*it) != p && (*it) != profile_proc) {
        worker_procs.push_back(*it);
      }
    }
  }

  // first, profile a task
  ComputeProfResult task_metrics;
  UserEvent compute_done = UserEvent::create_user_event();
  ComputeProfResultWrapper task_result;
  task_result.metrics = &task_metrics;
  task_result.cpu_idx = 0;
  task_result.done = compute_done;
  ProfilingRequestSet task_prs;
  task_prs.add_request(profile_proc, COMPUTE_PROF_TASK, &task_result, sizeof(ComputeProfResultWrapper))
    .add_measurement<ProfilingMeasurements::OperationTimeline>()
    .add_measurement<ProfilingMeasurements::OperationProcessorUsage>();
  ComputeTaskArgs compute_task_args;
  compute_task_args.idx = 0;
  compute_task_args.cpu_idx = 0;
  worker_procs[0].spawn(COMPUTE_TASK, &compute_task_args, sizeof(ComputeTaskArgs), task_prs).wait();

  // second, profile a copy and instance creation
  size_t elements = 1024;
  IndexSpace<1> ispace = Rect<1>(0, elements - 1);
  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID_BASE] = sizeof(int);
  
  // create instances on cpu memory
  Machine::MemoryQuery mq = Machine::MemoryQuery(Machine::get_machine())
      .local_address_space().only_kind(Memory::SYSTEM_MEM);
  Memory cpu_mem = mq.first();
  assert (cpu_mem.kind() != Memory::NO_MEMKIND);
  RegionInstance src_inst, dst_inst;

  // profile inst creation
  UserEvent src_inst_done = UserEvent::create_user_event();
  InstProfResult src_inst_metrics;
  InstProfResultWrapper src_inst_result;
  src_inst_result.metrics = &src_inst_metrics;
  src_inst_result.done = src_inst_done;
  ProfilingRequestSet src_inst_prs;
  src_inst_prs.add_request(profile_proc, INST_PROF_TASK, &src_inst_result, sizeof(InstProfResultWrapper))
    .add_measurement<ProfilingMeasurements::InstanceTimeline>()
    .add_measurement<ProfilingMeasurements::InstanceMemoryUsage>();
  RegionInstance::create_instance(src_inst, cpu_mem, ispace, field_sizes,
		  0, src_inst_prs).wait();

  UserEvent dst_inst_done = UserEvent::create_user_event();
  InstProfResult dst_inst_metrics;
  InstProfResultWrapper dst_inst_result;
  dst_inst_result.metrics = &dst_inst_metrics;
  dst_inst_result.done = dst_inst_done;
  ProfilingRequestSet dst_inst_prs;
  dst_inst_prs.add_request(profile_proc, INST_PROF_TASK, &dst_inst_result, sizeof(InstProfResultWrapper))
    .add_measurement<ProfilingMeasurements::InstanceTimeline>()
    .add_measurement<ProfilingMeasurements::InstanceMemoryUsage>();
  RegionInstance::create_instance(dst_inst, cpu_mem, ispace, field_sizes,
		  0, dst_inst_prs).wait();

  // fill the instance with some data for verification
  {
    int fill_value = 10;
    std::vector<CopySrcDstField> srcs(1);
    srcs[0].set_fill(fill_value);
    std::vector<CopySrcDstField> dsts(1);
    dsts[0].set_field(src_inst, FID_BASE, sizeof(int));
    ispace.copy(srcs, dsts, ProfilingRequestSet()).wait();
  }

  // launch a copy from src to dst and profile the copy
  std::vector<CopySrcDstField> srcs(1);
  srcs[0].set_field(src_inst, FID_BASE, sizeof(int));
  std::vector<CopySrcDstField> dsts(1);
  dsts[0].set_field(dst_inst, FID_BASE, sizeof(int));

  CopyProfResult copy_metrics;
  UserEvent copy_done = UserEvent::create_user_event();
  CopyProfResultWrapper copy_result;
  copy_result.metrics = &copy_metrics;
  copy_result.done = copy_done;
  ProfilingRequestSet copy_prs;
  copy_prs.add_request(profile_proc, COPY_PROF_TASK, &copy_result, sizeof(CopyProfResultWrapper))
    .add_measurement<ProfilingMeasurements::OperationTimeline>()
    .add_measurement<ProfilingMeasurements::OperationCopyInfo>()
    .add_measurement<ProfilingMeasurements::OperationMemoryUsage>();
  ispace.copy(srcs, dsts, copy_prs).wait();

  // verify the result of copy
  AffineAccessor<int, 1> acc(dst_inst, FID_BASE);
  for(IndexSpaceIterator<1, int> it(ispace); it.valid; it.step()) {
    for(PointInRectIterator<1, int> it2(it.rect); it2.valid; it2.step()) {
      int act = acc[it2.p];
      if(act != 10) {
        log_app.error() << "mismatch: [" << it2.p << "] = " << act << " (expected 1)";
      }
    }
  }

  src_inst.destroy();
  dst_inst.destroy();

  // collect profiling results
  task_result.done.wait();
  long long task_total_time = task_metrics.complete_time - task_metrics.start_time;
  long long task_wait_time = task_metrics.start_time - task_metrics.ready_time;
  log_app.print("Task on processor %llx ready at %lld, start at %lld, complete at %lld, total time %lld ns, wait time %lld ns",
                task_metrics.proc.id,
                task_metrics.ready_time, task_metrics.start_time, task_metrics.complete_time, task_total_time, task_wait_time);

  copy_result.done.wait();
  long long copy_total_time = copy_metrics.complete_time - copy_metrics.start_time;
  log_app.print("Copy start at %lld, complete at %lld, total time %lld ns, "
                "src inst %llx on memory %llx, dst inst %llx on memory %llx, size %zu (B), num hops %u",  
                copy_metrics.start_time, copy_metrics.complete_time, copy_total_time,
                copy_metrics.src_inst_id, copy_metrics.src_mem.id, copy_metrics.dst_inst_id, copy_metrics.dst_mem.id,
                copy_metrics.size, copy_metrics.num_hops);

  src_inst_done.wait();
  log_app.print("Src instance on memory %llx, size %zu (B), created at %lld, ready at %lld, destroyed at %lld",
                src_inst_metrics.memory.id, src_inst_metrics.bytes,
                src_inst_metrics.create_time, src_inst_metrics.ready_time, src_inst_metrics.delete_time);

  dst_inst_done.wait();
  log_app.print("Dst instance on memory %llx, size %zu (B), created at %lld, ready at %lld, destroyed at %lld",
                dst_inst_metrics.memory.id, dst_inst_metrics.bytes,
                dst_inst_metrics.create_time, dst_inst_metrics.ready_time, dst_inst_metrics.delete_time);

  // third, use profiling results to infer load balance task launching
  // create a vector to track the tasks distributed to processors
  std::vector<std::pair<int, double>> cpu_status(worker_procs.size(), {0,0});
  std::vector<UserEvent> prof_events;
  std::vector<Event> worker_events;
  std::vector<ComputeProfResult> batch_task_metrics(TestConfig::num_tasks);
  std::vector<double> batch_task_runtime(TestConfig::num_tasks, 0);
  for (int i = 0; i < TestConfig::num_iterations; i++) {
    UserEvent start_event = UserEvent::create_user_event();
    for (int j = 0; j < TestConfig::num_tasks; j++) {
      UserEvent compute_done = UserEvent::create_user_event();
      ComputeProfResultWrapper task_result;
      ComputeTaskArgs compute_task_args;
      compute_task_args.idx = j;
      bool use_profile = i == 0 ? false : true;
      compute_task_args.cpu_idx = find_the_best_cpu(cpu_status, batch_task_runtime, j, use_profile);
      task_result.cpu_idx = compute_task_args.cpu_idx;
      task_result.idx = j;
      task_result.metrics = &batch_task_metrics[j];
      task_result.done = compute_done;
      ProfilingRequestSet task_prs;
      task_prs.add_request(profile_proc, COMPUTE_PROF_TASK, &task_result, sizeof(ComputeProfResultWrapper))
        .add_measurement<ProfilingMeasurements::OperationTimeline>();
      Event e = worker_procs[compute_task_args.cpu_idx].spawn(COMPUTE_TASK, &compute_task_args, sizeof(ComputeTaskArgs), task_prs, start_event);
      prof_events.push_back(compute_done);
      worker_events.push_back(e);
    }
    Event done_event = Event::merge_events(worker_events);
    long long t1 = Clock::current_time_in_nanoseconds();
    start_event.trigger();
    done_event.wait();
    long long t2 = Clock::current_time_in_nanoseconds();
    if (i == 0) {
      log_app.print("With round-robin %lld us", (t2 - t1)/1000);
    } else {
      log_app.print("With profiling %lld us", (t2 - t1)/1000);
    }
    // collect profiling results
    for (int j = 0; j < TestConfig::num_tasks; j++) {
      prof_events[j].wait();
      double runtime = (batch_task_metrics[j].complete_time - batch_task_metrics[j].start_time) * 1e-9;
      batch_task_runtime[j] = (batch_task_runtime[j] * i + runtime) / (i + 1);
      log_app.info("task %d, runtime %f", j, batch_task_runtime[j]);
    }
    prof_events.clear();
  }
}

int main(int argc, char **argv) {
  Runtime rt;

  rt.init(&argc, (char ***)&argv);

  CommandLineParser cp;
  cp.add_option_int("-ni", TestConfig::num_iterations);
  cp.add_option_int("-nt", TestConfig::num_tasks);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, MAIN_TASK,
                                   CodeDescriptor(main_task),
                                   ProfilingRequestSet()).external_wait();

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, COMPUTE_TASK,
                                  CodeDescriptor(compute_task),
                                  ProfilingRequestSet()).external_wait();

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, COMPUTE_PROF_TASK,
                                  CodeDescriptor(compute_prof_task),
                                  ProfilingRequestSet()).external_wait();

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, COPY_PROF_TASK,
                                  CodeDescriptor(copy_prof_task),
                                  ProfilingRequestSet()).external_wait();

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, INST_PROF_TASK,
                                  CodeDescriptor(inst_prof_task),
                                  ProfilingRequestSet()).external_wait();

  Event e = rt.collective_spawn(p, MAIN_TASK, 0, 0);

  rt.shutdown(e);

  int ret = rt.wait_for_shutdown();

  return ret;
}
