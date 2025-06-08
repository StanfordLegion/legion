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

#include "prealm.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <zlib.h>
// Need this to know which version of Legion Prof we're using
static constexpr unsigned LEGION_PROF_VERSION =
// We hard code this since (P)Realm is about to be split from
// Legion and we won't be able to rely on this dependence anymore
// which means we'll have to manually update PRealm when the
// logging interface with Legion changes
#if 0
#include "legion/legion_profiling_version.h"
#else
    1008 // the current Legion Prof version we work with
#endif
    ;
// pr_fopen expects filename to be a std::string
#define pr_fopen(filename, mode) gzopen(filename.c_str(), mode)
#define pr_fwrite(f, data, num_bytes) gzwrite(f, data, num_bytes)
#define pr_fflush(f, mode) gzflush(f, mode)
#define pr_fclose(f) gzclose(f)

namespace PRealm {

const Event Event::NO_EVENT = Realm::Event::NO_EVENT;
const UserEvent UserEvent::NO_USER_EVENT = Realm::UserEvent::NO_USER_EVENT;
const Barrier Barrier::NO_BARRIER = Realm::Barrier::NO_BARRIER;
const ::realm_event_gen_t Barrier::MAX_PHASES = Realm::Barrier::MAX_PHASES;
const CompletionQueue CompletionQueue::NO_QUEUE =
    Realm::CompletionQueue::NO_QUEUE;
const Reservation Reservation::NO_RESERVATION =
    Realm::Reservation::NO_RESERVATION;
const Processor Processor::NO_PROC = Realm::Processor::NO_PROC;
const ProcessorGroup ProcessorGroup::NO_PROC_GROUP =
    Realm::ProcessorGroup::NO_PROC_GROUP;
const RegionInstance RegionInstance::NO_INST = RegionInstance();

Realm::Logger log_pr("PRealm");
thread_local ThreadProfiler *thread_profiler = nullptr;

class AutoLock {
public:
  inline AutoLock(Realm::FastReservation &r, bool excl = true)
    : local_lock(r)
    , exclusive(excl)
    , held(true)
  {
    if(exclusive) {
      Realm::Event ready = local_lock.wrlock();
      while(ready.exists()) {
        ready.wait();
        ready = local_lock.wrlock();
      }
    } else {
      Realm::Event ready = local_lock.rdlock();
      while(ready.exists()) {
        ready.wait();
        ready = local_lock.rdlock();
      }
    }
  }

public:
  AutoLock(AutoLock &&rhs) = delete;
  AutoLock(const AutoLock &rhs) = delete;
  inline ~AutoLock(void)
  {
    if(held)
      local_lock.unlock();
  }

public:
  AutoLock &operator=(AutoLock &&rhs) = delete;
  AutoLock &operator=(const AutoLock &rhs) = delete;

public:
  inline void release(void)
  {
    assert(held);
    local_lock.unlock();
    held = false;
  }
  inline void reacquire(void)
  {
    assert(!held);
    if(exclusive) {
      Realm::Event ready = local_lock.wrlock();
      while(ready.exists()) {
        ready.wait();
        ready = local_lock.wrlock();
      }
    } else {
      Realm::Event ready = local_lock.rdlock();
      while(ready.exists()) {
        ready.wait();
        ready = local_lock.rdlock();
      }
    }
    held = true;
  }

protected:
  Realm::FastReservation &local_lock;
  const bool exclusive;
  bool held;
};

class Profiler {
public:
  struct WrapperArgs {
    Event wait_on;
    size_t arglen;
    Processor::TaskFuncID task_id;
    int priority;
    Realm::Event fevent;
    ThreadProfiler::timestamp_t spawn_time;
    // Option for continuation wrapper
    Realm::UserEvent after;
    Realm::Processor origin;
  };
  struct ShutdownArgs {
    Event precondition;
    int code;
  };

public:
  Profiler(void);
  Profiler(const Profiler &rhs) = delete;
  Profiler &operator=(const Profiler &rhs) = delete;
  ~Profiler(void);

public:
  inline Processor get_local_processor(void) const { return local_proc; }
  Processor get_remote_processor(AddressSpace target);
  Processor get_implicit_processor(void);
  void parse_command_line(int argc, char **argv);
  void parse_command_line(std::vector<std::string> &cmdline, bool remove_args);
  void initialize(void);
  void defer_shutdown(Event precondition, int return_code);
  void wait_for_shutdown(void);
  void perform_shutdown(void);
  void finalize(void);
  Realm::Event get_remote_done(void);
  void record_thread_profiler(ThreadProfiler *profiler);
  unsigned long long find_backtrace_id(Backtrace &bt);
  unsigned long long find_provenance_id(const std::string_view &provenance);
  static Profiler &get_profiler(void);
  static void callback(const void *args, size_t arglen, const void *user_args,
                       size_t user_arglen, Realm::Processor p);
  static void wrapper(const void *args, size_t arglen, const void *user_args,
                      size_t user_arglen, Realm::Processor p);
  static void trigger(const void *args, size_t arglen, const void *user_args,
                      size_t user_arglen, Realm::Processor p);
  static void external(const void *args, size_t arglen, const void *user_args,
                       size_t user_arglen, Realm::Processor p);
  static void shutdown(const void *args, size_t arglen, const void *user_args,
                       size_t user_arglen, Realm::Processor p);
  static constexpr Realm::Processor::TaskFuncID CALLBACK_TASK_ID =
      Realm::Processor::TASK_ID_FIRST_AVAILABLE;
  static constexpr Realm::Processor::TaskFuncID WRAPPER_TASK_ID =
      Realm::Processor::TASK_ID_FIRST_AVAILABLE + 1;
  static constexpr Realm::Processor::TaskFuncID TRIGGER_TASK_ID =
      Realm::Processor::TASK_ID_FIRST_AVAILABLE + 2;
  static constexpr Realm::Processor::TaskFuncID EXTERNAL_TASK_ID =
      Realm::Processor::TASK_ID_FIRST_AVAILABLE + 3;
  static constexpr Realm::Processor::TaskFuncID SHUTDOWN_TASK_ID =
      Realm::Processor::TASK_ID_FIRST_AVAILABLE + 4;
  static_assert((CALLBACK_TASK_ID + 5) == Processor::TASK_ID_FIRST_AVAILABLE);
  static constexpr int CALLBACK_TASK_PRIORITY = std::numeric_limits<int>::min();

public:
#ifdef DEBUG_REALM
  void increment_total_outstanding_requests(ThreadProfiler::ProfKind kind);
  void decrement_total_outstanding_requests(ThreadProfiler::ProfKind kind);
#else
  void increment_total_outstanding_requests(void);
  void decrement_total_outstanding_requests(void);
#endif
  void record_task(Processor::TaskFuncID task_id, const std::string_view &task_name);
  void record_variant(Processor::TaskFuncID task_id, Processor::Kind kind);
  void record_memory(const Memory &m);
  void record_processor(const Processor &p);
  void record_affinities(std::vector<Memory> &memories_to_log);
  void record_remote_notification(Realm::Event notified);
  void record_implicit(long long start_time, Realm::Event fevent,
                       const std::vector<ThreadProfiler::WaitInfo> &wait_intervals);
  void update_footprint(size_t footprint, ThreadProfiler *profiler);

public:
  void serialize(const ThreadProfiler::MemDesc &info) const;
  void serialize(const ThreadProfiler::ProcDesc &info) const;
  void serialize(const ThreadProfiler::ProcMemDesc &info) const;
  void serialize(const ThreadProfiler::EventWaitInfo &info) const;
  void serialize(const ThreadProfiler::EventMergerInfo &info) const;
  void serialize(const ThreadProfiler::EventTriggerInfo &info) const;
  void serialize(const ThreadProfiler::EventPoisonInfo &info) const;
  void serialize(const ThreadProfiler::ExternalEventInfo &info) const;
  void serialize(const ThreadProfiler::BarrierArrivalInfo &info) const;
  void serialize(const ThreadProfiler::ReservationAcquireInfo &info) const;
  void serialize(const ThreadProfiler::InstanceReadyInfo &info) const;
  void serialize(const ThreadProfiler::InstanceUsageInfo &info) const;
  void serialize(const ThreadProfiler::FillInfo &info) const;
  void serialize(const ThreadProfiler::CopyInfo &info) const;
  void serialize(const ThreadProfiler::TaskInfo &info) const;
  void serialize(const ThreadProfiler::GPUTaskInfo &info) const;
  void serialize(const ThreadProfiler::InstTimelineInfo &info) const;
  void serialize(const ThreadProfiler::ProfTaskInfo &info) const;
  void serialize(const ThreadProfiler::ApplicationInfo &info) const;
  void serialize(const ThreadProfiler::SpawnInfo &info) const;

#ifdef REALM_USE_CUDA
public:
  Cuda::CudaModule *find_or_create_cuda_module(Realm::Cuda::CudaModule *mod);
#endif

private:
  void log_preamble(void) const;
  void log_configuration(Machine &machine, Processor local) const;
  void log_builtin_tasks(void) const;

private:
  Realm::FastReservation profiler_lock;
  std::vector<ThreadProfiler *> thread_profilers;
  std::deque<Realm::Event> remote_notifications;
  std::map<AddressSpace, Processor> remote_processors;

private:
#ifdef DEBUG_REALM
  unsigned total_outstanding_requests[ThreadProfiler::LAST_PROF];
#else
  std::atomic<unsigned> total_outstanding_requests;
#endif
private:
  gzFile f;
  Processor local_proc;
  Processor implicit_proc;
  std::string file_name;
  size_t output_footprint_threshold;
  size_t target_latency; // in us
  Realm::UserEvent done_event;
  std::vector<Memory> recorded_memories;
  std::vector<Processor> recorded_processors;
  std::unordered_set<Processor::TaskFuncID> registered_tasks;
  std::map<uintptr_t, unsigned long long> backtrace_ids;
  std::unordered_set<unsigned long long> provenance_ids;
  std::atomic<size_t> total_memory_footprint;
  unsigned total_address_spaces;
  Event shutdown_precondition;
  Realm::UserEvent shutdown_wait;
  int return_code;
  bool has_shutdown;
#ifdef REALM_USE_CUDA
  Cuda::CudaModule *cuda_module;
#endif

public:
  bool enabled;
  bool self_profile;
  bool no_critical_paths;
};

enum
{
  PROC_DESC_ID,
  MAX_DIM_DESC_ID,
  MACHINE_DESC_ID,
  CALIBRATION_ERR_ID,
  ZERO_TIME_ID,
  MEM_DESC_ID,
  PROC_MEM_DESC_ID,
  PHYSICAL_INST_REGION_ID,
  PHYSICAL_INST_LAYOUT_ID,
  PHYSICAL_INST_LAYOUT_DIM_ID,
  PHYSICAL_INST_USAGE_ID,
  TASK_KIND_ID,
  TASK_VARIANT_ID,
  TASK_WAIT_INFO_ID,
  TASK_INFO_ID,
  IMPLICIT_TASK_INFO_ID,
  GPU_TASK_INFO_ID,
  COPY_INFO_ID,
  COPY_INST_INFO_ID,
  FILL_INFO_ID,
  FILL_INST_INFO_ID,
  INST_TIMELINE_INFO_ID,
  PARTITION_INFO_ID,
  BACKTRACE_DESC_ID,
  EVENT_WAIT_INFO_ID,
  EVENT_MERGER_INFO_ID,
  EVENT_TRIGGER_INFO_ID,
  EVENT_POISON_INFO_ID,
  EXTERNAL_EVENT_INFO_ID,
  BARRIER_ARRIVAL_INFO_ID,
  RESERVATION_ACQUIRE_INFO_ID,
  INSTANCE_READY_INFO_ID,
  COMPLETION_QUEUE_INFO_ID,
  PROFTASK_INFO_ID,
  APPLICATION_INFO_ID,
  PROVENANCE_ID,
  SPAWN_INFO_ID,
};

enum DepPartOpKind {
  DEP_PART_UNION = 0,                  // a single union
  DEP_PART_UNIONS = 1,                 // many parallel unions
  DEP_PART_UNION_REDUCTION = 2,        // union reduction to a single space
  DEP_PART_INTERSECTION = 3,           // a single intersection
  DEP_PART_INTERSECTIONS = 4,          // many parallel intersections
  DEP_PART_INTERSECTION_REDUCTION = 5, // intersection reduction to a space
  DEP_PART_DIFFERENCE = 6,             // a single difference
  DEP_PART_DIFFERENCES = 7,            // many parallel differences
  DEP_PART_EQUAL = 8,                  // an equal partition operation
  DEP_PART_BY_FIELD = 9,               // create a partition from a field
  DEP_PART_BY_IMAGE = 10,              // create partition by image
  DEP_PART_BY_IMAGE_RANGE = 11,        // create partition by image range
  DEP_PART_BY_PREIMAGE = 12,           // create partition by preimage
  DEP_PART_BY_PREIMAGE_RANGE = 13,     // create partition by preimage range
  DEP_PART_ASSOCIATION = 14,           // create an association
  DEP_PART_WEIGHTS = 15,               // create partition by weights
};

void prealm_task_name(Realm::Processor::TaskFuncID task_id,
                      const std::string_view &task_name)
{
  Profiler::get_profiler().record_task(task_id, task_name);
}

ThreadProfiler::ThreadProfiler(Processor local, Realm::Event implicit)
    : local_proc(local), implicit_fevent(implicit),
      start_time(Realm::Clock::current_time_in_nanoseconds()) {
  assert(local_proc.exists());
  if (is_implicit()) {
    // fevent should always be on the same node as the local processor
    // to keep Legion Prof happy
    const Realm::ID id(implicit.id);
    assert(local_proc.address_space() == id.event_creator_node());
  }
}

Event ThreadProfiler::get_fevent(void) const {
  if (is_implicit())
    return implicit_fevent;
  const Event result = Processor::get_current_finish_event();
  // fevent should always be on the same node as the local processor
  // to keep Legion Prof happy
  const Realm::ID id(result.id);
  assert(local_proc.address_space() == id.event_creator_node());
  return result;
}

Processor ThreadProfiler::get_callback_processor(void) const {
  if (is_implicit())
    return Profiler::get_profiler().get_local_processor();
  else
    return local_proc;
}

void ThreadProfiler::NameClosure::add_instance(const RegionInstance &inst) {
  for (std::vector<RegionInstance>::const_iterator it = instances.begin();
       it != instances.end(); it++)
    if ((*it) == inst)
      return;
  instances.push_back(inst);
}

Event ThreadProfiler::NameClosure::find_instance_name(
    Realm::RegionInstance inst) const {
  for (std::vector<RegionInstance>::const_iterator it = instances.begin();
       it != instances.end(); it++)
    if (it->id == inst.id)
      return it->unique_event;
  std::abort();
}

void ThreadProfiler::process_proc_desc(const Processor &p) {
  if (std::binary_search(proc_ids.begin(), proc_ids.end(), p.id))
    return;
  proc_ids.push_back(p.id);
  std::sort(proc_ids.begin(), proc_ids.end());
  Profiler::get_profiler().record_processor(p);
}

void ThreadProfiler::process_mem_desc(const Memory &m) {
  if (m == Memory::NO_MEMORY)
    return;
  if (std::binary_search(mem_ids.begin(), mem_ids.end(), m.id))
    return;
  mem_ids.push_back(m.id);
  std::sort(mem_ids.begin(), mem_ids.end());
  Profiler::get_profiler().record_memory(m);
}

void ThreadProfiler::add_fill_request(ProfilingRequestSet &requests,
                                      const std::vector<CopySrcDstField> &dsts,
                                      Event critical) {
  Profiler &profiler = Profiler::get_profiler();
  if (!profiler.enabled)
    return;
#ifdef DEBUG_REALM
  profiler.increment_total_outstanding_requests(FILL_PROF);
#else
  profiler.increment_total_outstanding_requests();
#endif
  ProfilingArgs args(FILL_PROF);
  args.critical = critical;
  NameClosure *closure = new NameClosure;
  for (std::vector<CopySrcDstField>::const_iterator it = dsts.begin();
       it != dsts.end(); it++)
    closure->add_instance(it->inst);
  args.id.closure = closure;
  args.provenance = get_fevent();
  Processor current = get_callback_processor();
  Realm::ProfilingRequest &req =
      requests.add_request(current, Profiler::CALLBACK_TASK_ID, &args,
                           sizeof(args), Profiler::CALLBACK_TASK_PRIORITY);
  req.add_measurement<ProfilingMeasurements::OperationTimeline>();
  req.add_measurement<ProfilingMeasurements::OperationMemoryUsage>();
  req.add_measurement<ProfilingMeasurements::OperationCopyInfo>();
  req.add_measurement<ProfilingMeasurements::OperationFinishEvent>();
}

void ThreadProfiler::add_copy_request(ProfilingRequestSet &requests,
                                      const std::vector<CopySrcDstField> &srcs,
                                      const std::vector<CopySrcDstField> &dsts,
                                      Event critical) {
  Profiler &profiler = Profiler::get_profiler();
  if (!profiler.enabled)
    return;
#ifdef DEBUG_REALM
  profiler.increment_total_outstanding_requests(COPY_PROF);
#else
  profiler.increment_total_outstanding_requests();
#endif
  ProfilingArgs args(COPY_PROF);
  args.critical = critical;
  NameClosure *closure = new NameClosure;
  for (std::vector<CopySrcDstField>::const_iterator it = srcs.begin();
       it != srcs.end(); it++)
    closure->add_instance(it->inst);
  for (std::vector<CopySrcDstField>::const_iterator it = dsts.begin();
       it != dsts.end(); it++)
    closure->add_instance(it->inst);
  args.id.closure = closure;
  args.provenance = get_fevent();
  Processor current = get_callback_processor();
  Realm::ProfilingRequest &req =
      requests.add_request(current, Profiler::CALLBACK_TASK_ID, &args,
                           sizeof(args), Profiler::CALLBACK_TASK_PRIORITY);
  req.add_measurement<ProfilingMeasurements::OperationTimeline>();
  req.add_measurement<ProfilingMeasurements::OperationMemoryUsage>();
  req.add_measurement<ProfilingMeasurements::OperationCopyInfo>();
  req.add_measurement<ProfilingMeasurements::OperationFinishEvent>();
}

void ThreadProfiler::add_task_request(ProfilingRequestSet &requests,
                                      Processor::TaskFuncID task_id,
                                      Event critical, Event fevent,
                                      timestamp_t spawn_time) {
  Profiler &profiler = Profiler::get_profiler();
  if (!profiler.enabled)
    return;
#ifdef DEBUG_REALM
  profiler.increment_total_outstanding_requests(TASK_PROF);
#else
  profiler.increment_total_outstanding_requests();
#endif
  ProfilingArgs args(TASK_PROF, spawn_time);
  args.id.task = task_id;
  args.critical = critical;
  args.provenance = fevent;
  Processor current = get_callback_processor();
  Realm::ProfilingRequest &req =
      requests.add_request(current, Profiler::CALLBACK_TASK_ID, &args,
                           sizeof(args), Profiler::CALLBACK_TASK_PRIORITY);
  req.add_measurement<ProfilingMeasurements::OperationTimeline>();
  req.add_measurement<ProfilingMeasurements::OperationProcessorUsage>();
  req.add_measurement<ProfilingMeasurements::OperationEventWaits>();
  req.add_measurement<ProfilingMeasurements::OperationFinishEvent>();
  if (!is_implicit() && (local_proc.kind() == Processor::TOC_PROC))
    req.add_measurement<ProfilingMeasurements::OperationTimelineGPU>();
}

Event ThreadProfiler::add_inst_request(ProfilingRequestSet &requests,
                                       Event critical) {
  Profiler &profiler = Profiler::get_profiler();
  if (!profiler.enabled)
    return Event::NO_EVENT;
#ifdef DEBUG_REALM
  profiler.increment_total_outstanding_requests(INST_PROF);
#else
  profiler.increment_total_outstanding_requests();
#endif
  // Make a unique event to name this event
  const Realm::UserEvent unique_name = Realm::UserEvent::create_user_event();
  unique_name.trigger();
  ProfilingArgs args(INST_PROF);
  args.id.inst = unique_name;
  args.critical = critical;
  args.provenance = get_fevent();
  Processor current = get_callback_processor();
  Realm::ProfilingRequest &req =
      requests.add_request(current, Profiler::CALLBACK_TASK_ID, &args,
                           sizeof(args), Profiler::CALLBACK_TASK_PRIORITY);
  req.add_measurement<ProfilingMeasurements::InstanceMemoryUsage>();
  req.add_measurement<ProfilingMeasurements::InstanceTimeline>();
  return unique_name;
}

void ThreadProfiler::record_event_wait(Event wait_on, Backtrace &bt,
                                       long long start, long long stop) {
  Profiler &profiler = Profiler::get_profiler();
  if (!profiler.enabled)
    return;
  unsigned long long backtrace_id = profiler.find_backtrace_id(bt);
  event_wait_infos.emplace_back(
      EventWaitInfo{local_proc.id, get_fevent(), wait_on, backtrace_id});
  if (wait_on.is_barrier())
    record_barrier_use(wait_on);
  profiler.update_footprint(sizeof(EventWaitInfo), this);
  if (is_implicit()) {
    WaitInfo &info = implicit_waits.emplace_back(WaitInfo());
    info.wait_start = start;
    info.wait_ready = stop; // assume it's ready immediately
    info.wait_end = stop;
    info.wait_event = wait_on;
  }
}

void ThreadProfiler::record_event_trigger(Event result, Event pre) {
  Profiler &profiler = Profiler::get_profiler();
  if (!profiler.enabled || profiler.no_critical_paths)
    return;
  EventTriggerInfo &info = event_trigger_infos.emplace_back(EventTriggerInfo());
  info.performed = Realm::Clock::current_time_in_nanoseconds();
  info.result = result;
  info.precondition = pre;
  if (pre.is_barrier())
    record_barrier_use(pre);
  info.provenance = get_fevent();
  // See if we're triggering this event on the same node where it was made
  // If not we need to eventually notify the node where it was made that
  // it was triggered here and what the fevent was for it
  const Realm::ID id(result.id);
  const AddressSpace creator_node = id.event_creator_node();
  if (creator_node != local_proc.address_space()) {
    Realm::Processor remote = profiler.get_remote_processor(creator_node);
    const Realm::Event notified =
        remote.spawn(Profiler::TRIGGER_TASK_ID, &info, sizeof(info));
    profiler.record_remote_notification(notified);
  }
  profiler.update_footprint(sizeof(info), this);
}

void ThreadProfiler::record_event_poison(Event result) {
  Profiler &profiler = Profiler::get_profiler();
  if (!profiler.enabled || profiler.no_critical_paths)
    return;
  EventPoisonInfo &info = event_poison_infos.emplace_back(EventPoisonInfo());
  info.performed = Realm::Clock::current_time_in_nanoseconds();
  info.result = result;
  info.provenance = get_fevent();
  // See if we're poisoning this event on the same node where it was made
  // If not we need to eventually notify the node where it was made that
  // it was triggered here and what the fevent was for it
  const Realm::ID id(result.id);
  const AddressSpace creator_node = id.event_creator_node();
  if (creator_node != local_proc.address_space()) {
    // Encode this in an event trigger struct but put the actual
    // poisoned event in the precondition so the receiver knows
    // that was a poison and not a trigger
    Realm::Processor remote = profiler.get_remote_processor(creator_node);
    const EventTriggerInfo tinfo{Event::NO_EVENT, info.provenance, info.result,
                                 info.performed};
    const Realm::Event notified =
        remote.spawn(Profiler::TRIGGER_TASK_ID, &tinfo, sizeof(tinfo));
    profiler.record_remote_notification(notified);
  }
  profiler.update_footprint(sizeof(info), this);
}

void ThreadProfiler::record_barrier_use(Event bar) {
  // TODO: Right now we're doing the same as the "all-critical-arrivals" path
  // as Legion Prof since we don't know if and when it is safe to added in a
  // reduction operator to the barriers. This however means that if we only load
  // a subset of the data files we might be getting an incomplete picture of the
  // critical path for barrier arrivals.
  // Note: technically since we can see when we make the barriers, we can see if
  // they have a reduction operator and if they don't then substitute the
  // arrival timing reduction operator similar to what Legion does, although
  // then if we have mixed barriers some with reduction and some without we'll
  // need some way to be able to tell them apart.
}

void ThreadProfiler::record_barrier_arrival(Event result, Event precondition) {
  Profiler &profiler = Profiler::get_profiler();
  if (!profiler.enabled || profiler.no_critical_paths)
    return;
  BarrierArrivalInfo &info =
      barrier_arrival_infos.emplace_back(BarrierArrivalInfo());
  info.performed = Realm::Clock::current_time_in_nanoseconds();
  info.result = result;
  info.precondition = precondition;
  if (precondition.is_barrier())
    record_barrier_use(precondition);
  info.provenance = get_fevent();
  profiler.update_footprint(sizeof(info), this);
}

void ThreadProfiler::record_event_merger(Event result,
                                         const Event *preconditions,
                                         size_t num_events) {
  if (!result.exists())
    return;
  Profiler &profiler = Profiler::get_profiler();
  if (!profiler.enabled || profiler.no_critical_paths)
    return;
  // Realm can return one of the preconditions as the result of
  // an event merger as an optimization, to handle that we check
  // if the result is the same as any of the preconditions, if it
  // is then there is nothing needed for us to record
  for (unsigned idx = 0; idx < num_events; idx++)
    if (preconditions[idx] == result)
      return;
  EventMergerInfo &info = event_merger_infos.emplace_back(EventMergerInfo());
  // Take the timing measurement of when this happened first
  info.performed = Realm::Clock::current_time_in_nanoseconds();
  info.result = result;
  info.preconditions.resize(num_events);
  for (unsigned idx = 0; idx < num_events; idx++) {
    info.preconditions[idx] = preconditions[idx];
    if (preconditions[idx].is_barrier())
      record_barrier_use(preconditions[idx]);
  }
  info.provenance = get_fevent();
  profiler.update_footprint(sizeof(info) + num_events * sizeof(Event), this);
}

void ThreadProfiler::record_external_event(Realm::Event event,
                                           const std::string_view &prov)
{
  Profiler &profiler = Profiler::get_profiler();
  if(!profiler.enabled || profiler.no_critical_paths)
    return;
  // Check to see if this has already triggered or not
  if(event.has_triggered()) {
    // Take the timing measurement of when this happened first
    ExternalEventInfo &info = external_event_infos.emplace_back(ExternalEventInfo());
    info.triggered = Realm::Clock::current_time_in_nanoseconds();
    // Doing created after triggered makes it look like the event was
    // ready before it was created, which it probably was since it
    // triggered almost immediately after it was created, so while
    // not strictly accurate it reflects what actually happened so
    // the profiler can give good feedback
    info.created = Realm::Clock::current_time_in_nanoseconds();
    info.external = event;
    info.fevent = get_fevent();
    info.provenance = profiler.find_provenance_id(prov);
    profiler.update_footprint(sizeof(info), this);
  } else {
    // Spawn a no-op task with a profiling request that will measure when
    // the external event has triggered, we'll use that to determine
    // the trigger time for the event
    const ExternalTriggerArgs args{event, get_fevent(),
                                   profiler.find_provenance_id(prov)};
    // Need to dispatch this on a base realm processor to avoid
    // profiling ourselves when we do this
    const Realm::Processor local = get_callback_processor();
// Increment the number of pending callbacks we expect to see
#ifdef DEBUG_REALM
      profiler.increment_total_outstanding_requests(EXTERNAL_PROF);
#else
      profiler.increment_total_outstanding_requests();
#endif
      Realm::ProfilingRequestSet requests;
      Realm::ProfilingRequest &request =
          requests.add_request(local, Profiler::EXTERNAL_TASK_ID, &args, sizeof(args),
                               Profiler::CALLBACK_TASK_PRIORITY);
      request.add_measurement<ProfilingMeasurements::OperationTimeline>();
      // Make sure to protect the event in case it is poisoned
      local.spawn(Processor::TASK_ID_PROCESSOR_NOP, nullptr, 0, requests,
                  Realm::Event::ignorefaults(event), Profiler::CALLBACK_TASK_PRIORITY);
    }
  }

  void ThreadProfiler::record_reservation_acquire(Reservation r, Event result,
                                                  Event precondition)
  {
    Profiler &profiler = Profiler::get_profiler();
    if(!profiler.enabled || profiler.no_critical_paths)
      return;
    ReservationAcquireInfo &info =
        reservation_acquire_infos.emplace_back(ReservationAcquireInfo());
    info.performed = Realm::Clock::current_time_in_nanoseconds();
    info.result = result;
    info.precondition = precondition;
    if(precondition.is_barrier())
      record_barrier_use(precondition);
    info.reservation = r;
    info.provenance = get_fevent();
    profiler.update_footprint(sizeof(info), this);
  }

  Event ThreadProfiler::record_instance_ready(RegionInstance inst, Event result,
                                              Event precondition)
  {
    Profiler &profiler = Profiler::get_profiler();
    if(!profiler.enabled || profiler.no_critical_paths)
      return result;
    if(!result.exists()) {
      Realm::UserEvent rename = Realm::UserEvent::create_user_event();
      rename.trigger();
      result = rename;
    }
    InstanceReadyInfo &info = instance_ready_infos.emplace_back(InstanceReadyInfo());
    info.performed = Realm::Clock::current_time_in_nanoseconds();
    info.result = result;
    info.unique = inst.unique_event;
    info.precondition = precondition;
    if(precondition.is_barrier())
      record_barrier_use(precondition);
    profiler.update_footprint(sizeof(info), this);
    return result;
  }

  void ThreadProfiler::record_instance_usage(RegionInstance inst, FieldID field)
  {
    Profiler &profiler = Profiler::get_profiler();
    if(!profiler.enabled)
      return;
    InstanceUsageInfo &info = instance_usage_infos.emplace_back(InstanceUsageInfo());
    info.inst_event = inst.unique_event;
    info.op_id = get_fevent().id;
    info.field = field;
    profiler.update_footprint(sizeof(info), this);
  }

  void ThreadProfiler::process_response(ProfilingResponse &response)
  {
    Profiler &profiler = Profiler::get_profiler();
    long long start = 0;
    if(profiler.self_profile)
      start = Realm::Clock::current_time_in_nanoseconds();
    assert(sizeof(ProfilingArgs) == response.user_data_size());
    const ProfilingArgs *args = static_cast<const ProfilingArgs *>(response.user_data());
    Event finish_event;
    typedef ProfilingMeasurements::OperationCopyInfo::InstInfo InstInfo;
    switch(args->kind) {
    case FILL_PROF:
    {
      ProfilingMeasurements::OperationMemoryUsage usage;
      if(!response.get_measurement(usage))
        std::abort();
      ProfilingMeasurements::OperationCopyInfo cpinfo;
      if(!response.get_measurement(cpinfo))
        std::abort();
      ProfilingMeasurements::OperationTimeline timeline;
      if(!response.get_measurement(timeline))
        std::abort();
      assert(timeline.is_valid());
      ProfilingMeasurements::OperationFinishEvent fevent;
      if(!response.get_measurement(fevent))
        std::abort();

      process_mem_desc(usage.target);

      FillInfo &info = fill_infos.emplace_back(FillInfo());
      info.size = usage.size;
      info.create = timeline.create_time;
      info.ready = timeline.ready_time;
      info.start = timeline.start_time;
      // use complete_time instead of end_time to include async work
      info.stop = timeline.complete_time;
      info.fevent = fevent.finish_event;
      info.creator = args->provenance;
      info.critical = args->critical;
      if(args->critical.is_barrier())
        record_barrier_use(args->critical);
      for(std::vector<InstInfo>::const_iterator it = cpinfo.inst_info.begin();
          it != cpinfo.inst_info.end(); it++) {
#ifdef DEBUG_REALM
      assert(!it->dst_fields.empty());
      assert(it->dst_insts.size() == 1);
#endif
      Realm::RegionInstance instance = it->dst_insts.front();
      Memory location = instance.get_location();

      Event name = args->id.closure->find_instance_name(instance);
      unsigned offset = info.inst_infos.size();
      info.inst_infos.resize(offset + it->dst_fields.size());
      for (unsigned idx = 0; idx < it->dst_fields.size(); idx++) {
        FillInstInfo &inst_info = info.inst_infos[offset + idx];
        inst_info.dst = location.id;
        inst_info.fid = it->dst_fields[idx];
        inst_info.dst_inst_uid = name;
      }
      }
    profiler.update_footprint(
        sizeof(FillInfo) + info.inst_infos.size() * sizeof(FillInstInfo), this);
    finish_event = fevent.finish_event;
    delete args->id.closure;
    break;
    }
  case COPY_PROF: {
    ProfilingMeasurements::OperationMemoryUsage usage;
    if (!response.get_measurement(usage))
      std::abort();
    ProfilingMeasurements::OperationCopyInfo cpinfo;
    if (!response.get_measurement(cpinfo))
      std::abort();
    ProfilingMeasurements::OperationTimeline timeline;
    if (!response.get_measurement(timeline))
      std::abort();
    assert(timeline.is_valid());
    ProfilingMeasurements::OperationFinishEvent fevent;
    if (!response.get_measurement(fevent))
      std::abort();

    process_mem_desc(usage.source);
    process_mem_desc(usage.target);

    CopyInfo &info = copy_infos.emplace_back(CopyInfo());
    info.size = usage.size;
    info.create = timeline.create_time;
    info.ready = timeline.ready_time;
    info.start = timeline.start_time;
    // use complete_time instead of end_time to include async work
    info.stop = timeline.complete_time;
    info.fevent = fevent.finish_event;
    info.creator = args->provenance;
    info.critical = args->critical;
    if (args->critical.is_barrier())
      record_barrier_use(args->critical);
    assert(!cpinfo.inst_info.empty());
    NameClosure *closure = args->id.closure;
    for (std::vector<InstInfo>::const_iterator it = cpinfo.inst_info.begin();
         it != cpinfo.inst_info.end(); it++) {
      assert(it->src_fields.size() == it->dst_fields.size());
      if (it->src_indirection_inst.exists() ||
          it->dst_indirection_inst.exists()) {
        // Apparently we have to do the full cross-product of
        // everything here. I don't really understand so just
        // log what the Realm developers say and redirect any
        // questions from the profiler back to Realm
        unsigned offset = info.inst_infos.size();
        info.inst_infos.resize(offset +
                               (it->src_insts.size() * it->src_fields.size() *
                                it->dst_insts.size() * it->dst_fields.size()) +
                               1 /*extra for indirection*/);
        // Finally log the indirection instance(s)
        CopyInstInfo &indirect = info.inst_infos[offset++];
        indirect.indirect = true;
        indirect.num_hops = it->num_hops;
        if (it->src_indirection_inst.exists()) {
          indirect.src = it->src_indirection_inst.get_location().id;
          indirect.src_fid = it->src_indirection_field;
          indirect.src_inst_uid =
              closure->find_instance_name(it->src_indirection_inst);
        } else {
          indirect.src = 0;
          indirect.src_fid = 0;
          indirect.src_inst_uid = Event::NO_EVENT;
        }
        if (it->dst_indirection_inst.exists()) {
          indirect.dst = it->dst_indirection_inst.get_location().id;
          indirect.dst_fid = it->dst_indirection_field;
          indirect.dst_inst_uid =
              closure->find_instance_name(it->dst_indirection_inst);
        } else {
          indirect.dst = 0;
          indirect.dst_fid = 0;
          indirect.dst_inst_uid = Event::NO_EVENT;
        }
        for (unsigned idx1 = 0; idx1 < it->src_insts.size(); idx1++) {
          Realm::RegionInstance src_inst = it->src_insts[idx1];
          Memory src_location = src_inst.get_location();
          Event src_name = closure->find_instance_name(src_inst);
          for (unsigned idx2 = 0; idx2 < it->dst_insts.size(); idx2++) {
            Realm::RegionInstance dst_inst = it->dst_insts[idx2];
            Memory dst_location = dst_inst.get_location();
            Event dst_name = closure->find_instance_name(dst_inst);
            for (unsigned idx3 = 0; idx3 < it->src_fields.size(); idx3++) {
              const FieldID src_fid = it->src_fields[idx3];
              for (unsigned idx4 = 0; idx4 < it->dst_fields.size(); idx4++) {
                const FieldID dst_fid = it->dst_fields[idx4];
                CopyInstInfo &inst_info = info.inst_infos[offset++];
                inst_info.src = src_location.id;
                inst_info.dst = dst_location.id;
                inst_info.src_fid = src_fid;
                inst_info.dst_fid = dst_fid;
                inst_info.src_inst_uid = src_name;
                inst_info.dst_inst_uid = dst_name;
                inst_info.num_hops = it->num_hops;
                inst_info.indirect = false;
              }
            }
          }
        }
      } else {
        // Ask the Realm developers about why these assertions are true
        // because I still don't completely understand the logic
        assert(it->src_insts.size() == 1);
        assert(it->dst_insts.size() == 1);
        Realm::RegionInstance src_inst = it->src_insts.front();
        Realm::RegionInstance dst_inst = it->dst_insts.front();
        Memory src_location = src_inst.get_location();
        Memory dst_location = dst_inst.get_location();
        Event src_name = closure->find_instance_name(src_inst);
        Event dst_name = closure->find_instance_name(dst_inst);
        const unsigned offset = info.inst_infos.size();
        info.inst_infos.resize(offset + it->src_fields.size());
        for (unsigned idx = 0; idx < it->src_fields.size(); idx++) {
          CopyInstInfo &inst_info = info.inst_infos[offset + idx];
          inst_info.src = src_location.id;
          inst_info.dst = dst_location.id;
          inst_info.src_fid = it->src_fields[idx];
          inst_info.dst_fid = it->dst_fields[idx];
          inst_info.src_inst_uid = src_name;
          inst_info.dst_inst_uid = dst_name;
          inst_info.num_hops = it->num_hops;
          inst_info.indirect = false;
        }
      }
    }
    profiler.update_footprint(
        sizeof(CopyInfo) + info.inst_infos.size() * sizeof(CopyInstInfo), this);
    finish_event = fevent.finish_event;
    delete args->id.closure;
    break;
  }
  case TASK_PROF: {
    ProfilingMeasurements::OperationProcessorUsage usage;
    if (!response.get_measurement(usage))
      std::abort();
    ProfilingMeasurements::OperationTimeline timeline;
    if (!response.get_measurement(timeline))
      std::abort();
    ProfilingMeasurements::OperationEventWaits waits;
    if (!response.get_measurement(waits))
      std::abort();
    ProfilingMeasurements::OperationFinishEvent finish;
    if (!response.get_measurement(finish))
      std::abort();
    assert(timeline.is_valid());

    process_proc_desc(usage.proc);

    if (args->critical.is_barrier())
      record_barrier_use(args->critical);
    ProfilingMeasurements::OperationTimelineGPU timeline_gpu;
    if (response.get_measurement(timeline_gpu)) {
      assert(timeline_gpu.is_valid());

      GPUTaskInfo &info = gpu_task_infos.emplace_back(GPUTaskInfo());
      info.task_id = args->id.task;
      info.proc = usage.proc;
      size_t spawn_size = 0;
      // This happens if we had a wrapper task
      if(args->create_time != 0) {
        // See if this is cross-node task
        const Realm::ID id(args->provenance.id);
        if(usage.proc.address_space() != id.event_creator_node()) {
          // This is a cross-node task so record the spawn time
          SpawnInfo &spawn = spawn_infos.emplace_back(SpawnInfo());
          spawn.fevent = finish.finish_event;
          spawn.spawn = args->create_time;
          spawn_size = sizeof(spawn);
          info.create = timeline.create_time;
        } else
          info.create = args->create_time;
      } else
        info.create = timeline.create_time;
      info.ready = timeline.ready_time;
      info.start = timeline.start_time;
      info.stop = timeline.end_time;

      // record gpu time
      info.gpu_start = timeline_gpu.start_time;
      info.gpu_stop = timeline_gpu.end_time;

      unsigned num_intervals = waits.intervals.size();
      if (num_intervals > 0) {
        for (unsigned idx = 0; idx < num_intervals; ++idx) {
          WaitInfo &wait_info = info.wait_intervals.emplace_back(WaitInfo());
          wait_info.wait_start = waits.intervals[idx].wait_start;
          wait_info.wait_ready = waits.intervals[idx].wait_ready;
          wait_info.wait_end = waits.intervals[idx].wait_end;
          wait_info.wait_event = waits.intervals[idx].wait_event;
        }
      }
      info.creator = args->provenance;
      info.critical = args->critical;

      info.finish_event = finish.finish_event;
      profiler.update_footprint(
          sizeof(info) + spawn_size + num_intervals * sizeof(WaitInfo), this);
    } else {
      TaskInfo &info = task_infos.emplace_back(TaskInfo());
      info.task_id = args->id.task;
      info.proc = usage.proc;
      size_t spawn_size = 0;
      // This happens if we had a wrapper task
      if(args->create_time != 0) {
        // See if this is cross-node task
        const Realm::ID id(args->provenance.id);
        if(usage.proc.address_space() != id.event_creator_node()) {
          // This is a cross-node task so record the spawn time
          SpawnInfo &spawn = spawn_infos.emplace_back(SpawnInfo());
          spawn.fevent = finish.finish_event;
          spawn.spawn = args->create_time;
          spawn_size = sizeof(spawn);
          info.create = timeline.create_time;
        } else
          info.create = args->create_time;
      } else
        info.create = timeline.create_time;
      info.ready = timeline.ready_time;
      info.start = timeline.start_time;
      // use complete_time instead of end_time to include async work
      info.stop = timeline.complete_time;
      unsigned num_intervals = waits.intervals.size();
      if (num_intervals > 0) {
        for (unsigned idx = 0; idx < num_intervals; ++idx) {
          WaitInfo &wait_info = info.wait_intervals.emplace_back(WaitInfo());
          wait_info.wait_start = waits.intervals[idx].wait_start;
          wait_info.wait_ready = waits.intervals[idx].wait_ready;
          wait_info.wait_end = waits.intervals[idx].wait_end;
          wait_info.wait_event = waits.intervals[idx].wait_event;
        }
      }
      info.creator = args->provenance;
      info.critical = args->critical;
      info.finish_event = finish.finish_event;
      profiler.update_footprint(
          sizeof(info) + spawn_size + num_intervals * sizeof(WaitInfo), this);
    }
    finish_event = finish.finish_event;
    break;
  }
  case INST_PROF: {
    ProfilingMeasurements::InstanceTimeline timeline;
    if (!response.get_measurement(timeline))
      std::abort();
    ProfilingMeasurements::InstanceMemoryUsage usage;
    if (!response.get_measurement(usage))
      std::abort();

    process_mem_desc(usage.memory);

    InstTimelineInfo &info =
        inst_timeline_infos.emplace_back(InstTimelineInfo());
    info.inst_uid = args->id.inst;
    info.inst_id = usage.instance.id;
    info.mem_id = usage.memory.id;
    info.size = usage.bytes;
    info.create = timeline.create_time;
    info.ready = timeline.ready_time;
    info.destroy = timeline.delete_time;
    info.creator = args->provenance;
    profiler.update_footprint(sizeof(info), this);
    finish_event = args->id.inst;
    break;
  }
  case PART_PROF: {
    // TODO:
    std::abort();
  }
  default:
    std::abort();
  }
  if (profiler.self_profile) {
    const long long stop = Realm::Clock::current_time_in_nanoseconds();
    ProfTaskInfo &info = prof_task_infos.emplace_back(ProfTaskInfo());
    info.proc_id = local_proc.id;
    info.start = start;
    info.stop = stop;
    info.creator = finish_event;
    info.finish_event = Processor::get_current_finish_event();
    profiler.update_footprint(sizeof(info), this);
  }
#ifdef DEBUG_REALM
  profiler.decrement_total_outstanding_requests(args->kind);
#else
  profiler.decrement_total_outstanding_requests();
#endif
  }

void ThreadProfiler::process_trigger(const void *args, size_t arglen) {
  assert(arglen == sizeof(EventTriggerInfo));
  EventTriggerInfo info;
  std::memcpy(&info, args, sizeof(EventTriggerInfo));
  Profiler &profiler = Profiler::get_profiler();
  // Check to see if this was poisoned or not
  if (info.result.exists()) {
    // Normal event trigger
    event_trigger_infos.push_back(info);
    profiler.update_footprint(sizeof(info), this);
  } else {
    // This was poisoned
    EventPoisonInfo &pinfo = event_poison_infos.emplace_back(EventPoisonInfo());
    pinfo.result = info.precondition;
    assert(pinfo.result.exists());
    pinfo.provenance = info.provenance;
    pinfo.performed = info.performed;
    profiler.update_footprint(sizeof(pinfo), this);
  }
}

void ThreadProfiler::process_external(ProfilingResponse &response)
{
  Profiler &profiler = Profiler::get_profiler();
  long long start = 0;
  if(profiler.self_profile)
    start = Realm::Clock::current_time_in_nanoseconds();
  assert(sizeof(ExternalTriggerArgs) == response.user_data_size());
  const ExternalTriggerArgs *args =
      static_cast<const ExternalTriggerArgs *>(response.user_data());
  ProfilingMeasurements::OperationTimeline timeline;
  if(!response.get_measurement(timeline))
    std::abort();

  ExternalEventInfo &info = external_event_infos.emplace_back(ExternalEventInfo());
  info.created = timeline.create_time;
  info.triggered = timeline.ready_time;
  info.external = args->external;
  info.fevent = args->fevent;
  info.provenance = args->provenance;
  profiler.update_footprint(sizeof(info), this);

  if(profiler.self_profile) {
    const long long stop = Realm::Clock::current_time_in_nanoseconds();
    ProfTaskInfo &info = prof_task_infos.emplace_back(ProfTaskInfo());
    info.proc_id = local_proc.id;
    info.start = start;
    info.stop = stop;
    info.creator = args->fevent;
    info.finish_event = Processor::get_current_finish_event();
    profiler.update_footprint(sizeof(info), this);
  }
#ifdef DEBUG_REALM
  profiler.decrement_total_outstanding_requests(EXTERNAL_PROF);
#else
  profiler.decrement_total_outstanding_requests();
#endif
}

void ThreadProfiler::record_time_range(long long start, const std::string_view &name)
{
  ApplicationInfo &info = application_infos.emplace_back(ApplicationInfo());
  info.stop = Realm::Clock::current_time_in_nanoseconds();
  info.start = start;
  info.fevent = get_fevent();
  info.proc_id = local_proc.id;
  Profiler &profiler = Profiler::get_profiler();
  info.provenance = profiler.find_provenance_id(name);
  profiler.update_footprint(sizeof(info), this);
}

size_t ThreadProfiler::dump_inter(long long target_latency) {
  Profiler &profiler = Profiler::get_profiler();
  // Start the timing so we know how long we are taking
  const long long t_start = Realm::Clock::current_time_in_microseconds();
  // Scale our latency by how much we are over the space limit
  const long long t_stop = t_start + target_latency;
  size_t diff = 0;
  while (!event_wait_infos.empty()) {
    EventWaitInfo &info = event_wait_infos.front();
    profiler.serialize(info);
    diff += sizeof(info);
    event_wait_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if (t_curr >= t_stop)
      return diff;
  }
  while (!event_merger_infos.empty()) {
    EventMergerInfo &info = event_merger_infos.front();
    profiler.serialize(info);
    diff += (sizeof(info) + (info.preconditions.size() * sizeof(Event)));
    event_merger_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if (t_curr >= t_stop)
      return diff;
  }
  while (!event_trigger_infos.empty()) {
    EventTriggerInfo &info = event_trigger_infos.front();
    profiler.serialize(info);
    diff += sizeof(info);
    event_trigger_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if (t_curr >= t_stop)
      return diff;
  }
  while (!event_poison_infos.empty()) {
    EventPoisonInfo &info = event_poison_infos.front();
    profiler.serialize(info);
    diff += sizeof(info);
    event_poison_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if (t_curr >= t_stop)
      return diff;
  }
  while(!external_event_infos.empty()) {
    ExternalEventInfo &info = external_event_infos.front();
    profiler.serialize(info);
    diff += sizeof(info);
    external_event_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if(t_curr >= t_stop)
      return diff;
  }
  while (!barrier_arrival_infos.empty()) {
    BarrierArrivalInfo &info = barrier_arrival_infos.front();
    profiler.serialize(info);
    diff += sizeof(info);
    barrier_arrival_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if (t_curr >= t_stop)
      return diff;
  }
  while (!reservation_acquire_infos.empty()) {
    ReservationAcquireInfo &info = reservation_acquire_infos.front();
    profiler.serialize(info);
    diff += sizeof(info);
    reservation_acquire_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if (t_curr >= t_stop)
      return diff;
  }
  while (!instance_ready_infos.empty()) {
    InstanceReadyInfo &info = instance_ready_infos.front();
    profiler.serialize(info);
    diff += sizeof(info);
    instance_ready_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if (t_curr >= t_stop)
      return diff;
  }
  while (!instance_usage_infos.empty()) {
    InstanceUsageInfo &info = instance_usage_infos.front();
    profiler.serialize(info);
    diff += sizeof(info);
    instance_usage_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if (t_curr >= t_stop)
      return diff;
  }
  while (!fill_infos.empty()) {
    FillInfo &info = fill_infos.front();
    profiler.serialize(info);
    diff += sizeof(info);
    fill_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if (t_curr >= t_stop)
      return diff;
  }
  while (!copy_infos.empty()) {
    CopyInfo &info = copy_infos.front();
    profiler.serialize(info);
    diff += sizeof(info);
    copy_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if (t_curr >= t_stop)
      return diff;
  }
  while (!task_infos.empty()) {
    TaskInfo &info = task_infos.front();
    profiler.serialize(info);
    diff += sizeof(info) + info.wait_intervals.size() * sizeof(WaitInfo);
    task_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if (t_curr >= t_stop)
      return diff;
  }
  while (!gpu_task_infos.empty()) {
    GPUTaskInfo &info = gpu_task_infos.front();
    profiler.serialize(info);
    diff += sizeof(info) + info.wait_intervals.size() * sizeof(WaitInfo);
    gpu_task_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if (t_curr >= t_stop)
      return diff;
  }
  while (!inst_timeline_infos.empty()) {
    InstTimelineInfo &info = inst_timeline_infos.front();
    profiler.serialize(info);
    diff += sizeof(info);
    inst_timeline_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if (t_curr >= t_stop)
      return diff;
  }
  while (!prof_task_infos.empty()) {
    ProfTaskInfo &info = prof_task_infos.front();
    profiler.serialize(info);
    diff += sizeof(info);
    prof_task_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if (t_curr >= t_stop)
      return diff;
  }
  while(!application_infos.empty()) {
    ApplicationInfo &info = application_infos.front();
    profiler.serialize(info);
    diff += sizeof(info);
    application_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if(t_curr >= t_stop)
      return diff;
  }
  while(!spawn_infos.empty()) {
    SpawnInfo &info = spawn_infos.front();
    profiler.serialize(info);
    diff += sizeof(info);
    spawn_infos.pop_front();
    const long long t_curr = Realm::Clock::current_time_in_microseconds();
    if(t_curr >= t_stop)
      return diff;
  }
  return diff;
}

void ThreadProfiler::finalize(void) {
  Profiler &profiler = Profiler::get_profiler();
  for (unsigned idx = 0; idx < event_wait_infos.size(); idx++)
    profiler.serialize(event_wait_infos[idx]);
  for (unsigned idx = 0; idx < event_merger_infos.size(); idx++)
    profiler.serialize(event_merger_infos[idx]);
  for (unsigned idx = 0; idx < event_trigger_infos.size(); idx++)
    profiler.serialize(event_trigger_infos[idx]);
  for (unsigned idx = 0; idx < event_poison_infos.size(); idx++)
    profiler.serialize(event_poison_infos[idx]);
  for(unsigned idx = 0; idx < external_event_infos.size(); idx++)
    profiler.serialize(external_event_infos[idx]);
  for (unsigned idx = 0; idx < barrier_arrival_infos.size(); idx++)
    profiler.serialize(barrier_arrival_infos[idx]);
  for (unsigned idx = 0; idx < reservation_acquire_infos.size(); idx++)
    profiler.serialize(reservation_acquire_infos[idx]);
  for (unsigned idx = 0; idx < instance_ready_infos.size(); idx++)
    profiler.serialize(instance_ready_infos[idx]);
  for (unsigned idx = 0; idx < instance_usage_infos.size(); idx++)
    profiler.serialize(instance_usage_infos[idx]);
  for (unsigned idx = 0; idx < fill_infos.size(); idx++)
    profiler.serialize(fill_infos[idx]);
  for (unsigned idx = 0; idx < copy_infos.size(); idx++)
    profiler.serialize(copy_infos[idx]);
  for (unsigned idx = 0; idx < task_infos.size(); idx++)
    profiler.serialize(task_infos[idx]);
  for (unsigned idx = 0; idx < gpu_task_infos.size(); idx++)
    profiler.serialize(gpu_task_infos[idx]);
  for (unsigned idx = 0; idx < inst_timeline_infos.size(); idx++)
    profiler.serialize(inst_timeline_infos[idx]);
  for (unsigned idx = 0; idx < prof_task_infos.size(); idx++)
    profiler.serialize(prof_task_infos[idx]);
  for(unsigned idx = 0; idx < application_infos.size(); idx++)
    profiler.serialize(application_infos[idx]);
  for(unsigned idx = 0; idx < spawn_infos.size(); idx++)
    profiler.serialize(spawn_infos[idx]);
  if (is_implicit())
    profiler.record_implicit(start_time, implicit_fevent, implicit_waits);
}

/*static*/ ThreadProfiler &ThreadProfiler::get_thread_profiler(void) {
  if (thread_profiler == nullptr) {
    Processor proc = Processor::get_executing_processor();
    Profiler &profiler = Profiler::get_profiler();
    if (!proc.exists()) {
      proc = profiler.get_implicit_processor();
      const Realm::UserEvent fevent = Realm::UserEvent::create_user_event();
      fevent.trigger();
      thread_profiler = new ThreadProfiler(proc, fevent);
    } else {
      thread_profiler = new ThreadProfiler(proc, Realm::Event::NO_EVENT);
    }
    profiler.record_thread_profiler(thread_profiler);
  }
  return *thread_profiler;
}

Profiler::Profiler(void)
    : local_proc(Processor::NO_PROC), implicit_proc(Processor::NO_PROC),
      output_footprint_threshold(128 << 20 /*128MB*/),
      target_latency(100 /*us*/), total_memory_footprint(0),
      shutdown_wait(Realm::UserEvent::NO_USER_EVENT), return_code(0),
      has_shutdown(false), enabled(false), self_profile(false),
      no_critical_paths(false) {
#ifdef DEBUG_REALM
  for (unsigned idx = 0; idx < ThreadProfiler::LAST_PROF; idx++)
    total_outstanding_requests[idx] = 0;
#endif
#ifdef REALM_USE_CUDA
  cuda_module = nullptr;
#endif
}

Profiler::~Profiler(void)
{
#ifdef REALM_USE_CUDA
  if(cuda_module != nullptr)
    delete cuda_module;
#endif
}

void Profiler::parse_command_line(int argc, char **argv) {
  Realm::CommandLineParser cp;
  cp.add_option_string("-pr:logfile", file_name, true)
      .add_option_int("-pr:footprint", output_footprint_threshold, true)
      .add_option_int("-pr:latency", target_latency, true)
      .add_option_bool("-pr:self", self_profile, true)
      .add_option_bool("-pr:no-critical", no_critical_paths, true)
      .parse_command_line(argc, argv);
  if (file_name.empty()) {
    log_pr.warning() << "PRealm did not find a file name specified with "
                        "-pr:logfile. No profiling will be logged.";
  } else {
    enabled = true;
  }
}

void Profiler::parse_command_line(std::vector<std::string> &cmdline,
                                  bool remove_args) {
  Realm::CommandLineParser cp;
  cp.add_option_string("-pr:logfile", file_name, !remove_args)
      .add_option_int("-pr:footprint", output_footprint_threshold, !remove_args)
      .add_option_int("-pr:latency", target_latency, !remove_args)
      .add_option_bool("-pr:self", self_profile, !remove_args)
      .add_option_bool("-pr:no-critical", no_critical_paths, !remove_args)
      .parse_command_line(cmdline);
  if (file_name.empty()) {
    log_pr.warning() << "PRealm did not find a file name specified with "
                        "-pr:logfile. No profiling will be logged.";
  } else {
    enabled = true;
  }
}

void Profiler::initialize(void) {
  Machine machine = Machine::get_machine();
  total_address_spaces = machine.get_address_space_count();
  Realm::Machine::ProcessorQuery local_procs(machine);
  local_procs.local_address_space();
  assert(!local_proc.exists());
  local_proc = local_procs.first();
  assert(local_proc.exists());
  if (!enabled)
    return;
  size_t pct = file_name.find_first_of('%', 0);
  if (pct != std::string::npos) {
    std::stringstream ss;
    ss << file_name.substr(0, pct) << local_proc.address_space()
       << file_name.substr(pct + 1);
    file_name = ss.str();
  } else if (total_address_spaces > 1) {
    log_pr.error()
        << "When running in multi-process configurations PRealm requires "
        << "that all filenames contain a '%%' delimiter so each process will "
           "be "
        << "writing to an independent file. The specified file name '"
        << file_name << "' does not contain a '%%' delimiter.";
    exit(1);
  }

  // Create the logfile
  f = pr_fopen(file_name, "wb");
  if (!f) {
    log_pr.error() << "PRealm is unable to open file " << file_name
                   << " for writing";
    exit(1);
  }
  // Log the preamble
  log_preamble();
  // Log the machine description
  log_configuration(machine, local_proc);
  // Record all the default task IDs and variants
  log_builtin_tasks();
}

void Profiler::log_preamble(void) const {
  typedef Realm::Processor::TaskFuncID TaskID;
  typedef Realm::Processor::TaskFuncID VariantID;
  typedef Realm::Processor::Kind ProcKind;
  typedef Realm::Memory::Kind MemKind;
  typedef ::realm_id_t ProcID;
  typedef ::realm_id_t MemID;
  typedef ::realm_id_t InstID;
  typedef ::realm_id_t IDType;
  typedef ::realm_id_t UniqueID;
  typedef long long timestamp_t;
  std::stringstream ss;
  ss << "FileType: BinaryLegionProf v: 1.0" << std::endl;

  const std::string delim = ", ";

  ss << "ProcDesc {"
     << "id:" << PROC_DESC_ID << delim << "proc_id:ProcID:" << sizeof(ProcID)
     << delim << "kind:ProcKind:" << sizeof(ProcKind) << delim
     << "uuid_size:uuid_size:" << sizeof(unsigned) << delim
     << "cuda_device_uuid:uuid:" << sizeof(char) << "}" << std::endl;

  ss << "MaxDimDesc {"
     << "id:" << MAX_DIM_DESC_ID << delim
     << "max_dim:maxdim:" << sizeof(unsigned) << "}" << std::endl;

  ss << "MachineDesc {"
     << "id:" << MACHINE_DESC_ID << delim
     << "node_id:unsigned:" << sizeof(unsigned) << delim
     << "num_nodes:unsigned:" << sizeof(unsigned) << delim
     << "version:unsigned:" << sizeof(unsigned) << delim << "hostname:string:"
     << "-1" << delim
     << "host_id:unsigned long long:" << sizeof(unsigned long long) << delim
     << "process_id:unsigned:" << sizeof(unsigned) << "}" << std::endl;

  ss << "CalibrationErr {"
     << "id:" << CALIBRATION_ERR_ID << delim
     << "calibration_err:long long:" << sizeof(long long) << "}" << std::endl;

  ss << "ZeroTime {"
     << "id:" << ZERO_TIME_ID << delim
     << "zero_time:long long:" << sizeof(long long) << "}" << std::endl;

  ss << "MemDesc {"
     << "id:" << MEM_DESC_ID << delim << "mem_id:MemID:" << sizeof(MemID)
     << delim << "kind:MemKind:" << sizeof(MemKind) << delim
     << "capacity:unsigned long long:" << sizeof(unsigned long long) << "}"
     << std::endl;

  ss << "ProcMDesc {"
     << "id:" << PROC_MEM_DESC_ID << delim
     << "proc_id:ProcID:" << sizeof(ProcID) << delim
     << "mem_id:MemID:" << sizeof(MemID) << delim
     << "bandwidth:unsigned:" << sizeof(unsigned) << delim
     << "latency:unsigned:" << sizeof(unsigned) << "}" << std::endl;

  ss << "PhysicalInstRegionDesc {"
     << "id:" << PHYSICAL_INST_REGION_ID << delim
     << "inst_uid:unsigned long long:" << sizeof(Event) << delim
     << "ispace_id:IDType:" << sizeof(IDType) << delim
     << "fspace_id:unsigned:" << sizeof(unsigned) << delim
     << "tree_id:unsigned:" << sizeof(unsigned) << "}" << std::endl;

  ss << "PhysicalInstLayoutDesc {"
     << "id:" << PHYSICAL_INST_LAYOUT_ID << delim
     << "inst_uid:unsigned long long:" << sizeof(Event) << delim
     << "field_id:unsigned:" << sizeof(unsigned) << delim
     << "fspace_id:unsigned:" << sizeof(unsigned) << delim
     << "has_align:bool:" << sizeof(bool) << delim
     << "eqk:unsigned:" << sizeof(unsigned) << delim
     << "align_desc:unsigned:" << sizeof(unsigned) << "}" << std::endl;

  ss << "PhysicalInstDimOrderDesc {"
     << "id:" << PHYSICAL_INST_LAYOUT_DIM_ID << delim
     << "inst_uid:unsigned long long:" << sizeof(Event) << delim
     << "dim:unsigned:" << sizeof(unsigned) << delim
     << "dim_kind:unsigned:" << sizeof(unsigned) << "}" << std::endl;

  ss << "PhysicalInstanceUsage {"
     << "id:" << PHYSICAL_INST_USAGE_ID << delim
     << "inst_uid:unsigned long long:" << sizeof(Event) << delim
     << "op_id:UniqueID:" << sizeof(UniqueID) << delim
     << "index_id:unsigned:" << sizeof(unsigned) << delim
     << "field_id:unsigned:" << sizeof(unsigned) << "}" << std::endl;

  ss << "TaskKind {"
     << "id:" << TASK_KIND_ID << delim << "task_id:TaskID:" << sizeof(TaskID)
     << delim << "name:string:"
     << "-1" << delim << "overwrite:bool:" << sizeof(bool) << "}" << std::endl;

  ss << "TaskVariant {"
     << "id:" << TASK_VARIANT_ID << delim << "task_id:TaskID:" << sizeof(TaskID)
     << delim << "variant_id:VariantID:" << sizeof(VariantID) << delim
     << "name:string:"
     << "-1"
     << "}" << std::endl;

  ss << "TaskWaitInfo {"
     << "id:" << TASK_WAIT_INFO_ID << delim
     << "op_id:UniqueID:" << sizeof(UniqueID) << delim
     << "task_id:TaskID:" << sizeof(TaskID) << delim
     << "variant_id:VariantID:" << sizeof(VariantID) << delim
     << "wait_start:timestamp_t:" << sizeof(timestamp_t) << delim
     << "wait_ready:timestamp_t:" << sizeof(timestamp_t) << delim
     << "wait_end:timestamp_t:" << sizeof(timestamp_t) << delim
     << "wait_event:unsigned long long:" << sizeof(Event) << "}" << std::endl;

  ss << "TaskInfo {"
     << "id:" << TASK_INFO_ID << delim << "op_id:UniqueID:" << sizeof(UniqueID)
     << delim << "task_id:TaskID:" << sizeof(TaskID) << delim
     << "variant_id:VariantID:" << sizeof(VariantID) << delim
     << "proc_id:ProcID:" << sizeof(ProcID) << delim
     << "create:timestamp_t:" << sizeof(timestamp_t) << delim
     << "ready:timestamp_t:" << sizeof(timestamp_t) << delim
     << "start:timestamp_t:" << sizeof(timestamp_t) << delim
     << "stop:timestamp_t:" << sizeof(timestamp_t) << delim
     << "creator:unsigned long long:" << sizeof(Event) << delim
     << "critical:unsigned long long:" << sizeof(Event) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << "}" << std::endl;

  ss << "ImplicitTaskInfo {"
     << "id:" << IMPLICIT_TASK_INFO_ID << delim
     << "op_id:UniqueID:" << sizeof(UniqueID) << delim
     << "task_id:TaskID:" << sizeof(TaskID) << delim
     << "variant_id:VariantID:" << sizeof(VariantID) << delim
     << "proc_id:ProcID:" << sizeof(ProcID) << delim
     << "create:timestamp_t:" << sizeof(timestamp_t) << delim
     << "ready:timestamp_t:" << sizeof(timestamp_t) << delim
     << "start:timestamp_t:" << sizeof(timestamp_t) << delim
     << "stop:timestamp_t:" << sizeof(timestamp_t) << delim
     << "creator:unsigned long long:" << sizeof(Event) << delim
     << "critical:unsigned long long:" << sizeof(Event) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << "}" << std::endl;

  ss << "GPUTaskInfo {"
     << "id:" << GPU_TASK_INFO_ID << delim
     << "op_id:UniqueID:" << sizeof(UniqueID) << delim
     << "task_id:TaskID:" << sizeof(TaskID) << delim
     << "variant_id:VariantID:" << sizeof(VariantID) << delim
     << "proc_id:ProcID:" << sizeof(ProcID) << delim
     << "create:timestamp_t:" << sizeof(timestamp_t) << delim
     << "ready:timestamp_t:" << sizeof(timestamp_t) << delim
     << "start:timestamp_t:" << sizeof(timestamp_t) << delim
     << "stop:timestamp_t:" << sizeof(timestamp_t) << delim
     << "gpu_start:timestamp_t:" << sizeof(timestamp_t) << delim
     << "gpu_stop:timestamp_t:" << sizeof(timestamp_t) << delim
     << "creator:unsigned long long:" << sizeof(Event) << delim
     << "critical:unsigned long long:" << sizeof(Event) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << "}" << std::endl;

  ss << "CopyInfo {"
     << "id:" << COPY_INFO_ID << delim << "op_id:UniqueID:" << sizeof(UniqueID)
     << delim << "size:unsigned long long:" << sizeof(unsigned long long)
     << delim << "create:timestamp_t:" << sizeof(timestamp_t) << delim
     << "ready:timestamp_t:" << sizeof(timestamp_t) << delim
     << "start:timestamp_t:" << sizeof(timestamp_t) << delim
     << "stop:timestamp_t:" << sizeof(timestamp_t) << delim
     << "creator:unsigned long long:" << sizeof(Event) << delim
     << "critical:unsigned long long:" << sizeof(Event) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << delim
     << "collective:unsigned:" << sizeof(unsigned) << "}" << std::endl;

  ss << "CopyInstInfo {"
     << "id:" << COPY_INST_INFO_ID << delim << "src:MemID:" << sizeof(MemID)
     << delim << "dst:MemID:" << sizeof(MemID) << delim
     << "src_fid:unsigned:" << sizeof(FieldID) << delim
     << "dst_fid:unsigned:" << sizeof(FieldID) << delim
     << "src_inst:unsigned long long:" << sizeof(Event) << delim
     << "dst_inst:unsigned long long:" << sizeof(Event) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << delim
     << "num_hops:unsigned:" << sizeof(unsigned) << delim
     << "indirect:bool:" << sizeof(bool) << "}" << std::endl;

  ss << "FillInfo {"
     << "id:" << FILL_INFO_ID << delim << "op_id:UniqueID:" << sizeof(UniqueID)
     << delim << "size:unsigned long long:" << sizeof(unsigned long long)
     << delim << "create:timestamp_t:" << sizeof(timestamp_t) << delim
     << "ready:timestamp_t:" << sizeof(timestamp_t) << delim
     << "start:timestamp_t:" << sizeof(timestamp_t) << delim
     << "stop:timestamp_t:" << sizeof(timestamp_t) << delim
     << "creator:unsigned long long:" << sizeof(Event) << delim
     << "critical:unsigned long long:" << sizeof(Event) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << "}" << std::endl;

  ss << "FillInstInfo {"
     << "id:" << FILL_INST_INFO_ID << delim << "dst:MemID:" << sizeof(MemID)
     << delim << "fid:unsigned:" << sizeof(FieldID) << delim
     << "dst_inst:unsigned long long:" << sizeof(Event) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << "}" << std::endl;

  ss << "InstTimelineInfo {"
     << "id:" << INST_TIMELINE_INFO_ID << delim
     << "inst_uid:unsigned long long:" << sizeof(Event) << delim
     << "inst_id:InstID:" << sizeof(InstID) << delim
     << "mem_id:MemID:" << sizeof(MemID) << delim
     << "size:unsigned long long:" << sizeof(unsigned long long) << delim
     << "op_id:UniqueID:" << sizeof(UniqueID) << delim
     << "create:timestamp_t:" << sizeof(timestamp_t) << delim
     << "ready:timestamp_t:" << sizeof(timestamp_t) << delim
     << "destroy:timestamp_t:" << sizeof(timestamp_t) << delim
     << "creator:unsigned long long:" << sizeof(Event) << "}" << std::endl;

  ss << "PartitionInfo {"
     << "id:" << PARTITION_INFO_ID << delim
     << "op_id:UniqueID:" << sizeof(UniqueID) << delim
     << "part_op:DepPartOpKind:" << sizeof(DepPartOpKind) << delim
     << "create:timestamp_t:" << sizeof(timestamp_t) << delim
     << "ready:timestamp_t:" << sizeof(timestamp_t) << delim
     << "start:timestamp_t:" << sizeof(timestamp_t) << delim
     << "stop:timestamp_t:" << sizeof(timestamp_t) << delim
     << "creator:unsigned long long:" << sizeof(Event) << delim
     << "critical:unsigned long long:" << sizeof(Event) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << "}" << std::endl;

  ss << "BacktraceDesc {"
     << "id:" << BACKTRACE_DESC_ID << delim
     << "backtrace_id:unsigned long long:" << sizeof(unsigned long long)
     << delim << "backtrace:string:"
     << "-1"
     << "}" << std::endl;

  ss << "EventWaitInfo {"
     << "id:" << EVENT_WAIT_INFO_ID << delim
     << "proc_id:ProcID:" << sizeof(ProcID) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << delim
     << "wait_event:unsigned long long:" << sizeof(Event) << delim
     << "backtrace_id:unsigned long long:" << sizeof(unsigned long long) << "}"
     << std::endl;

  ss << "EventMergerInfo {"
     << "id:" << EVENT_MERGER_INFO_ID << delim
     << "result:unsigned long long:" << sizeof(Event) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << delim
     << "performed:timestamp_t:" << sizeof(timestamp_t) << delim
     << "pre0:unsigned long long:" << sizeof(Event) << delim
     << "pre1:unsigned long long:" << sizeof(Event) << delim
     << "pre2:unsigned long long:" << sizeof(Event) << delim
     << "pre3:unsigned long long:" << sizeof(Event) << "}" << std::endl;

  ss << "EventTriggerInfo {"
     << "id:" << EVENT_TRIGGER_INFO_ID << delim
     << "result:unsigned long long:" << sizeof(Event) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << delim
     << "precondition:unsigned long long:" << sizeof(Event) << delim
     << "performed:timestamp_t:" << sizeof(timestamp_t) << "}" << std::endl;

  ss << "EventPoisonInfo {"
     << "id:" << EVENT_POISON_INFO_ID << delim
     << "result:unsigned long long:" << sizeof(Event) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << delim
     << "performed:timestamp_t:" << sizeof(timestamp_t) << "}" << std::endl;

  ss << "ExternalEventInfo {"
     << "id:" << EXTERNAL_EVENT_INFO_ID << delim
     << "external:unsigned long long:" << sizeof(Event) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << delim
     << "created:timestamp_t:" << sizeof(timestamp_t) << delim
     << "trigger:timestamp_t:" << sizeof(timestamp_t) << delim
     << "prov:unsigned long long:" << sizeof(unsigned long long) << "}" << std::endl;

  ss << "BarrierArrivalInfo {"
     << "id:" << BARRIER_ARRIVAL_INFO_ID << delim
     << "result:unsigned long long:" << sizeof(Event) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << delim
     << "precondition:unsigned long long:" << sizeof(Event) << delim
     << "performed:timestamp_t:" << sizeof(timestamp_t) << "}" << std::endl;

  ss << "ReservationAcquireInfo {"
     << "id:" << RESERVATION_ACQUIRE_INFO_ID << delim
     << "result:unsigned long long:" << sizeof(Event) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << delim
     << "precondition:unsigned long long:" << sizeof(Event) << delim
     << "performed:timestamp_t:" << sizeof(timestamp_t) << delim
     << "reservation:unsigned long long:" << sizeof(Reservation) << "}"
     << std::endl;

  ss << "InstanceReadyInfo {"
     << "id:" << INSTANCE_READY_INFO_ID << delim
     << "result:unsigned long long:" << sizeof(Event) << delim
     << "precondition:unsigned long long:" << sizeof(Event) << delim
     << "inst_uid:unsigned long long:" << sizeof(Event) << delim
     << "performed:timestamp_t:" << sizeof(timestamp_t) << "}" << std::endl;

  ss << "CompletionQueueInfo {"
     << "id:" << COMPLETION_QUEUE_INFO_ID << delim
     << "result:unsigned long long:" << sizeof(Event) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << delim
     << "performed:timestamp_t:" << sizeof(timestamp_t) << delim
     << "pre0:unsigned long long:" << sizeof(Event) << delim
     << "pre1:unsigned long long:" << sizeof(Event) << delim
     << "pre2:unsigned long long:" << sizeof(Event) << delim
     << "pre3:unsigned long long:" << sizeof(Event) << "}" << std::endl;

  ss << "ProfTaskInfo {"
     << "id:" << PROFTASK_INFO_ID << delim
     << "proc_id:ProcID:" << sizeof(ProcID) << delim
     << "op_id:UniqueID:" << sizeof(UniqueID) << delim
     << "start:timestamp_t:" << sizeof(timestamp_t) << delim
     << "stop:timestamp_t:" << sizeof(timestamp_t) << delim
     << "creator:unsigned long long:" << sizeof(Event) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << delim
     << "completion:bool:" << sizeof(bool) << "}" << std::endl;

  ss << "ApplicationCallInfo {"
     << "id:" << APPLICATION_INFO_ID << delim
     << "provenance:unsigned long long:" << sizeof(unsigned long long) << delim
     << "start:timestamp_t:" << sizeof(timestamp_t) << delim
     << "stop:timestamp_t:" << sizeof(timestamp_t) << delim
     << "proc_id:ProcID:" << sizeof(ProcID) << delim
     << "fevent:unsigned long long:" << sizeof(Event) << "}" << std::endl;

  ss << "Provenance {"
     << "id:" << PROVENANCE_ID << delim
     << "provenance:unsigned long long:" << sizeof(unsigned long long) << delim
     << "prov:string:-1}" << std::endl;

  ss << "SpawnInfo {"
     << "id:" << SPAWN_INFO_ID << delim << "fevent:unsigned long long:" << sizeof(Event)
     << delim << "spawn:timestamp_t:" << sizeof(timestamp_t) << delim
     << "create:timestamp_t:" << sizeof(timestamp_t) << "}" << std::endl;

  // An empty line indicates the end of the preamble.
  ss << std::endl;
  std::string preamble = ss.str();

  pr_fwrite(f, preamble.c_str(), strlen(preamble.c_str()));
}

void Profiler::log_configuration(Machine &machine, Processor local) const {
  // Log the machine configuration
  int ID = MACHINE_DESC_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  unsigned node_id = local.address_space();
  pr_fwrite(f, (char *)&(node_id), sizeof(node_id));
  pr_fwrite(f, (char *)&(total_address_spaces), sizeof(total_address_spaces));
  unsigned version = LEGION_PROF_VERSION;
  pr_fwrite(f, (char *)&(version), sizeof(version));
  Machine::ProcessInfo process_info;
  machine.get_process_info(local, &process_info);
  pr_fwrite(f, process_info.hostname, strlen(process_info.hostname) + 1);
  pr_fwrite(f, (char *)&(process_info.hostid), sizeof(process_info.hostid));
  pr_fwrite(f, (char *)&(process_info.processid),
            sizeof(process_info.processid));
  // Log the zero time
  ID = ZERO_TIME_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  long long zero_time = Realm::Clock::get_zero_time();
  pr_fwrite(f, (char *)&(zero_time), sizeof(zero_time));
  // Log the maximum dimensions
  ID = MAX_DIM_DESC_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  unsigned max_dim = REALM_MAX_DIM;
  pr_fwrite(f, (char *)&(max_dim), sizeof(max_dim));
}

void Profiler::log_builtin_tasks(void) const {
  for (Processor::TaskFuncID task_id = Processor::TASK_ID_PROCESSOR_NOP;
       task_id <= Processor::TASK_ID_PROCESSOR_SHUTDOWN; task_id++) {
    const char *task_name = nullptr;
    switch (task_id) {
    case Processor::TASK_ID_PROCESSOR_NOP: {
      task_name = "NOP Task";
      break;
    }
    case Processor::TASK_ID_PROCESSOR_INIT: {
      task_name = "Initialization Task";
      break;
    }
    case Processor::TASK_ID_PROCESSOR_SHUTDOWN: {
      task_name = "Shutdown Task";
      break;
    }
    default:
      std::abort();
    }
    int ID = TASK_KIND_ID;
    pr_fwrite(f, (char *)&ID, sizeof(ID));
    pr_fwrite(f, (char *)&(task_id), sizeof(task_id));
    pr_fwrite(f, task_name, strlen(task_name) + 1);
    bool overwrite = false;
    pr_fwrite(f, (char *)&(overwrite), sizeof(overwrite));
    for (unsigned variant_id = Processor::NO_KIND + 1;
         variant_id <= Processor::PY_PROC; variant_id++) {
      ID = TASK_VARIANT_ID;
      pr_fwrite(f, (char *)&ID, sizeof(ID));
      pr_fwrite(f, (char *)&(task_id), sizeof(task_id));
      pr_fwrite(f, (char *)&(variant_id), sizeof(variant_id));
      pr_fwrite(f, task_name, strlen(task_name) + 1);
    }
  }
  // Also record a special variant for external threads
  const char *task_name = "External Thread";
  int ID = TASK_VARIANT_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  const Processor::TaskFuncID task_id = Processor::TASK_ID_PROCESSOR_NOP;
  pr_fwrite(f, (char *)&(task_id), sizeof(task_id));
  unsigned variant_id = Processor::NO_KIND;
  pr_fwrite(f, (char *)&(variant_id), sizeof(variant_id));
  pr_fwrite(f, task_name, strlen(task_name) + 1);
}

void Profiler::serialize(const ThreadProfiler::ProcDesc &proc_desc) const {
  int ID = PROC_DESC_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&(proc_desc.proc_id), sizeof(proc_desc.proc_id));
  pr_fwrite(f, (char *)&(proc_desc.kind), sizeof(proc_desc.kind));
#ifdef REALM_USE_CUDA
  unsigned uuid_size = Realm::Cuda::UUID_SIZE;
  pr_fwrite(f, (char *)&(uuid_size), sizeof(uuid_size));
  for (size_t i = 0; i < Realm::Cuda::UUID_SIZE; i++) {
    pr_fwrite(f, (char *)&(proc_desc.cuda_device_uuid[i]), sizeof(char));
  }
#else
  unsigned uuid_size = 16;
  pr_fwrite(f, (char *)&(uuid_size), sizeof(uuid_size));
  char uuid_str[16] = {0};
  for (size_t i = 0; i < uuid_size; i++) {
    pr_fwrite(f, (char *)&(uuid_str[i]), sizeof(char));
  }
#endif
}

void Profiler::serialize(const ThreadProfiler::MemDesc &mem_desc) const {
  int ID = MEM_DESC_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&(mem_desc.mem_id), sizeof(mem_desc.mem_id));
  pr_fwrite(f, (char *)&(mem_desc.kind), sizeof(mem_desc.kind));
  pr_fwrite(f, (char *)&(mem_desc.capacity), sizeof(mem_desc.capacity));
}

void Profiler::serialize(const ThreadProfiler::ProcMemDesc &pm) const {
  int ID = PROC_MEM_DESC_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&(pm.proc_id), sizeof(pm.proc_id));
  pr_fwrite(f, (char *)&(pm.mem_id), sizeof(pm.mem_id));
  pr_fwrite(f, (char *)&(pm.bandwidth), sizeof(pm.bandwidth));
  pr_fwrite(f, (char *)&(pm.latency), sizeof(pm.latency));
}

void Profiler::serialize(const ThreadProfiler::EventWaitInfo &info) const {
  int ID = EVENT_WAIT_INFO_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&info.proc_id, sizeof(info.proc_id));
  pr_fwrite(f, (char *)&info.fevent.id, sizeof(info.fevent.id));
  pr_fwrite(f, (char *)&info.event.id, sizeof(info.event.id));
  pr_fwrite(f, (char *)&info.backtrace_id, sizeof(info.backtrace_id));
}

void Profiler::serialize(const ThreadProfiler::EventMergerInfo &info) const {
  int ID = EVENT_MERGER_INFO_ID;
  for (unsigned offset = 0; offset < info.preconditions.size(); offset += 4) {
    pr_fwrite(f, (char *)&ID, sizeof(ID));
    pr_fwrite(f, (char *)&info.result.id, sizeof(info.result.id));
    pr_fwrite(f, (char *)&info.provenance.id, sizeof(info.provenance.id));
    pr_fwrite(f, (char *)&info.performed, sizeof(info.performed));
    pr_fwrite(f, (char *)&info.preconditions[offset].id,
              sizeof(info.preconditions[offset].id));
    for (unsigned idx = 1; idx < 4; idx++) {
      if ((offset + idx) < info.preconditions.size())
        pr_fwrite(f, (char *)&info.preconditions[offset + idx].id,
                  sizeof(info.preconditions[offset + idx].id));
      else
        pr_fwrite(f, (char *)&Event::NO_EVENT, sizeof(Event::NO_EVENT));
    }
  }
}

void Profiler::serialize(const ThreadProfiler::EventTriggerInfo &info) const {
  int ID = EVENT_TRIGGER_INFO_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&info.result.id, sizeof(info.result.id));
  pr_fwrite(f, (char *)&info.provenance.id, sizeof(info.provenance.id));
  pr_fwrite(f, (char *)&info.precondition.id, sizeof(info.precondition.id));
  pr_fwrite(f, (char *)&info.performed, sizeof(info.performed));
}

void Profiler::serialize(const ThreadProfiler::EventPoisonInfo &info) const {
  int ID = EVENT_POISON_INFO_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&info.result.id, sizeof(info.result.id));
  pr_fwrite(f, (char *)&info.provenance.id, sizeof(info.provenance.id));
  pr_fwrite(f, (char *)&info.performed, sizeof(info.performed));
}

void Profiler::serialize(const ThreadProfiler::ExternalEventInfo &info) const
{
  int ID = EXTERNAL_EVENT_INFO_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&info.external.id, sizeof(info.external.id));
  pr_fwrite(f, (char *)&info.fevent.id, sizeof(info.fevent.id));
  pr_fwrite(f, (char *)&info.created, sizeof(info.triggered));
  pr_fwrite(f, (char *)&info.triggered, sizeof(info.triggered));
  pr_fwrite(f, (char *)&info.provenance, sizeof(info.provenance));
}

void Profiler::serialize(const ThreadProfiler::BarrierArrivalInfo &info) const {
  int ID = BARRIER_ARRIVAL_INFO_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&info.result.id, sizeof(info.result.id));
  pr_fwrite(f, (char *)&info.provenance.id, sizeof(info.provenance.id));
  pr_fwrite(f, (char *)&info.precondition.id, sizeof(info.precondition.id));
  pr_fwrite(f, (char *)&info.performed, sizeof(info.performed));
}

void Profiler::serialize(
    const ThreadProfiler::ReservationAcquireInfo &info) const {
  int ID = RESERVATION_ACQUIRE_INFO_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&info.result.id, sizeof(info.result.id));
  pr_fwrite(f, (char *)&info.provenance.id, sizeof(info.provenance.id));
  pr_fwrite(f, (char *)&info.precondition.id, sizeof(info.precondition.id));
  pr_fwrite(f, (char *)&info.performed, sizeof(info.performed));
  pr_fwrite(f, (char *)&info.reservation.id, sizeof(info.reservation.id));
}

void Profiler::serialize(const ThreadProfiler::InstanceReadyInfo &info) const {
  int ID = INSTANCE_READY_INFO_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&info.result.id, sizeof(info.result.id));
  pr_fwrite(f, (char *)&info.precondition.id, sizeof(info.precondition.id));
  pr_fwrite(f, (char *)&info.unique.id, sizeof(info.unique.id));
  pr_fwrite(f, (char *)&info.performed, sizeof(info.performed));
}

void Profiler::serialize(const ThreadProfiler::InstanceUsageInfo &info) const {
  int ID = PHYSICAL_INST_USAGE_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&(info.inst_event.id), sizeof(info.inst_event.id));
  pr_fwrite(f, (char *)&(info.op_id), sizeof(info.op_id));
  unsigned index = 0; // All these have the same "region requirement for now
  pr_fwrite(f, (char *)&(index), sizeof(index));
  pr_fwrite(f, (char *)&(info.field), sizeof(info.field));
}

void Profiler::serialize(const ThreadProfiler::FillInfo &fill_info) const {
  int ID = FILL_INFO_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  unsigned long long op_id = fill_info.creator.id;
  pr_fwrite(f, (char *)&(op_id), sizeof(op_id));
  pr_fwrite(f, (char *)&(fill_info.size), sizeof(fill_info.size));
  pr_fwrite(f, (char *)&(fill_info.create), sizeof(fill_info.create));
  pr_fwrite(f, (char *)&(fill_info.ready), sizeof(fill_info.ready));
  pr_fwrite(f, (char *)&(fill_info.start), sizeof(fill_info.start));
  pr_fwrite(f, (char *)&(fill_info.stop), sizeof(fill_info.stop));
  pr_fwrite(f, (char *)&(fill_info.creator), sizeof(fill_info.creator));
  pr_fwrite(f, (char *)&(fill_info.critical), sizeof(fill_info.critical));
  pr_fwrite(f, (char *)&(fill_info.fevent), sizeof(fill_info.fevent.id));
  ID = FILL_INST_INFO_ID;
  for (std::vector<ThreadProfiler::FillInstInfo>::const_iterator it =
           fill_info.inst_infos.begin();
       it != fill_info.inst_infos.end(); it++) {
    const ThreadProfiler::FillInstInfo &fill_inst = *it;
    pr_fwrite(f, (char *)&ID, sizeof(ID));
    pr_fwrite(f, (char *)&(fill_inst.dst), sizeof(fill_inst.dst));
    pr_fwrite(f, (char *)&(fill_inst.fid), sizeof(fill_inst.fid));
    pr_fwrite(f, (char *)&(fill_inst.dst_inst_uid.id),
              sizeof(fill_inst.dst_inst_uid.id));
    pr_fwrite(f, (char *)&(fill_info.fevent), sizeof(fill_info.fevent.id));
  }
}

void Profiler::serialize(const ThreadProfiler::CopyInfo &copy_info) const {
  int ID = COPY_INFO_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  unsigned long long op_id = copy_info.creator.id;
  pr_fwrite(f, (char *)&(op_id), sizeof(op_id));
  pr_fwrite(f, (char *)&(copy_info.size), sizeof(copy_info.size));
  pr_fwrite(f, (char *)&(copy_info.create), sizeof(copy_info.create));
  pr_fwrite(f, (char *)&(copy_info.ready), sizeof(copy_info.ready));
  pr_fwrite(f, (char *)&(copy_info.start), sizeof(copy_info.start));
  pr_fwrite(f, (char *)&(copy_info.stop), sizeof(copy_info.stop));
  pr_fwrite(f, (char *)&(copy_info.creator), sizeof(copy_info.creator));
  pr_fwrite(f, (char *)&(copy_info.critical), sizeof(copy_info.critical));
  pr_fwrite(f, (char *)&(copy_info.fevent), sizeof(copy_info.fevent.id));
  unsigned collective = 0; // no collective copies ehre
  pr_fwrite(f, (char *)&(collective), sizeof(collective));
  ID = COPY_INST_INFO_ID;
  for (std::vector<ThreadProfiler::CopyInstInfo>::const_iterator it =
           copy_info.inst_infos.begin();
       it != copy_info.inst_infos.end(); it++) {
    const ThreadProfiler::CopyInstInfo &copy_inst = *it;
    pr_fwrite(f, (char *)&ID, sizeof(ID));
    pr_fwrite(f, (char *)&(copy_inst.src), sizeof(copy_inst.src));
    pr_fwrite(f, (char *)&(copy_inst.dst), sizeof(copy_inst.dst));
    pr_fwrite(f, (char *)&(copy_inst.src_fid), sizeof(copy_inst.src_fid));
    pr_fwrite(f, (char *)&(copy_inst.dst_fid), sizeof(copy_inst.dst_fid));
    pr_fwrite(f, (char *)&(copy_inst.src_inst_uid.id),
              sizeof(copy_inst.src_inst_uid.id));
    pr_fwrite(f, (char *)&(copy_inst.dst_inst_uid.id),
              sizeof(copy_inst.dst_inst_uid.id));
    pr_fwrite(f, (char *)&(copy_info.fevent), sizeof(copy_info.fevent.id));
    pr_fwrite(f, (char *)&(copy_inst.num_hops), sizeof(copy_inst.num_hops));
    pr_fwrite(f, (char *)&(copy_inst.indirect), sizeof(copy_inst.indirect));
  }
}

void Profiler::serialize(const ThreadProfiler::TaskInfo &task_info) const {
  int ID = TASK_INFO_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  unsigned long long op_id = task_info.finish_event.id;
  pr_fwrite(f, (char *)&(op_id), sizeof(op_id));
  pr_fwrite(f, (char *)&(task_info.task_id), sizeof(task_info.task_id));
  // Use the processor kind as the variant
  unsigned variant_id = task_info.proc.kind();
  pr_fwrite(f, (char *)&(variant_id), sizeof(variant_id));
  pr_fwrite(f, (char *)&(task_info.proc.id), sizeof(task_info.proc.id));
  pr_fwrite(f, (char *)&(task_info.create), sizeof(task_info.create));
  pr_fwrite(f, (char *)&(task_info.ready), sizeof(task_info.ready));
  pr_fwrite(f, (char *)&(task_info.start), sizeof(task_info.start));
  pr_fwrite(f, (char *)&(task_info.stop), sizeof(task_info.stop));
  pr_fwrite(f, (char *)&(task_info.creator), sizeof(task_info.creator));
  pr_fwrite(f, (char *)&(task_info.critical), sizeof(task_info.critical));
  pr_fwrite(f, (char *)&(task_info.finish_event),
            sizeof(task_info.finish_event));
  ID = TASK_WAIT_INFO_ID;
  for (std::vector<ThreadProfiler::WaitInfo>::const_iterator it =
           task_info.wait_intervals.begin();
       it != task_info.wait_intervals.end(); it++) {
    const ThreadProfiler::WaitInfo &wait_info = *it;
    pr_fwrite(f, (char *)&ID, sizeof(ID));
    pr_fwrite(f, (char *)&(op_id), sizeof(op_id));
    pr_fwrite(f, (char *)&(task_info.task_id), sizeof(task_info.task_id));
    pr_fwrite(f, (char *)&(variant_id), sizeof(variant_id));
    pr_fwrite(f, (char *)&(wait_info.wait_start), sizeof(wait_info.wait_start));
    pr_fwrite(f, (char *)&(wait_info.wait_ready), sizeof(wait_info.wait_ready));
    pr_fwrite(f, (char *)&(wait_info.wait_end), sizeof(wait_info.wait_end));
    pr_fwrite(f, (char *)&(wait_info.wait_event), sizeof(wait_info.wait_event));
  }
}

void Profiler::serialize(const ThreadProfiler::GPUTaskInfo &task_info) const {
  int ID = GPU_TASK_INFO_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  unsigned long long op_id = task_info.finish_event.id;
  pr_fwrite(f, (char *)&(op_id), sizeof(op_id));
  pr_fwrite(f, (char *)&(task_info.task_id), sizeof(task_info.task_id));
  // Use the processor kind as the variant
  unsigned variant_id = task_info.proc.kind();
  pr_fwrite(f, (char *)&(variant_id), sizeof(variant_id));
  pr_fwrite(f, (char *)&(task_info.proc.id), sizeof(task_info.proc.id));
  pr_fwrite(f, (char *)&(task_info.create), sizeof(task_info.create));
  pr_fwrite(f, (char *)&(task_info.ready), sizeof(task_info.ready));
  pr_fwrite(f, (char *)&(task_info.start), sizeof(task_info.start));
  pr_fwrite(f, (char *)&(task_info.stop), sizeof(task_info.stop));
  pr_fwrite(f, (char *)&(task_info.gpu_start), sizeof(task_info.gpu_start));
  pr_fwrite(f, (char *)&(task_info.gpu_stop), sizeof(task_info.gpu_stop));
  pr_fwrite(f, (char *)&(task_info.creator), sizeof(task_info.creator));
  pr_fwrite(f, (char *)&(task_info.critical), sizeof(task_info.critical));
  pr_fwrite(f, (char *)&(task_info.finish_event),
            sizeof(task_info.finish_event));
  ID = TASK_WAIT_INFO_ID;
  for (std::vector<ThreadProfiler::WaitInfo>::const_iterator it =
           task_info.wait_intervals.begin();
       it != task_info.wait_intervals.end(); it++) {
    const ThreadProfiler::WaitInfo &wait_info = *it;
    pr_fwrite(f, (char *)&ID, sizeof(ID));
    pr_fwrite(f, (char *)&(op_id), sizeof(op_id));
    pr_fwrite(f, (char *)&(task_info.task_id), sizeof(task_info.task_id));
    pr_fwrite(f, (char *)&(variant_id), sizeof(variant_id));
    pr_fwrite(f, (char *)&(wait_info.wait_start), sizeof(wait_info.wait_start));
    pr_fwrite(f, (char *)&(wait_info.wait_ready), sizeof(wait_info.wait_ready));
    pr_fwrite(f, (char *)&(wait_info.wait_end), sizeof(wait_info.wait_end));
    pr_fwrite(f, (char *)&(wait_info.wait_event), sizeof(wait_info.wait_event));
  }
}

void Profiler::serialize(
    const ThreadProfiler::InstTimelineInfo &inst_timeline_info) const {
  int ID = INST_TIMELINE_INFO_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&(inst_timeline_info.inst_uid.id),
            sizeof(inst_timeline_info.inst_uid.id));
  pr_fwrite(f, (char *)&(inst_timeline_info.inst_id),
            sizeof(inst_timeline_info.inst_id));
  pr_fwrite(f, (char *)&(inst_timeline_info.mem_id),
            sizeof(inst_timeline_info.mem_id));
  pr_fwrite(f, (char *)&(inst_timeline_info.size),
            sizeof(inst_timeline_info.size));
  unsigned long long op_id = inst_timeline_info.creator.id;
  pr_fwrite(f, (char *)&op_id, sizeof(op_id));
  pr_fwrite(f, (char *)&(inst_timeline_info.create),
            sizeof(inst_timeline_info.create));
  pr_fwrite(f, (char *)&(inst_timeline_info.ready),
            sizeof(inst_timeline_info.ready));
  pr_fwrite(f, (char *)&(inst_timeline_info.destroy),
            sizeof(inst_timeline_info.destroy));
  pr_fwrite(f, (char *)&(inst_timeline_info.creator),
            sizeof(inst_timeline_info.creator));
}

void Profiler::serialize(
    const ThreadProfiler::ProfTaskInfo &proftask_info) const {
  int ID = PROFTASK_INFO_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&(proftask_info.proc_id), sizeof(proftask_info.proc_id));
  unsigned long long op_id = proftask_info.creator.id;
  pr_fwrite(f, (char *)&(op_id), sizeof(op_id));
  pr_fwrite(f, (char *)&(proftask_info.start), sizeof(proftask_info.start));
  pr_fwrite(f, (char *)&(proftask_info.stop), sizeof(proftask_info.stop));
  pr_fwrite(f, (char *)&(proftask_info.creator), sizeof(proftask_info.creator));
  pr_fwrite(f, (char *)&(proftask_info.finish_event),
            sizeof(proftask_info.finish_event));
  bool completion = true; // always a true completion here
  pr_fwrite(f, (char *)&(completion), sizeof(completion));
}

void Profiler::serialize(const ThreadProfiler::ApplicationInfo &info) const
{
  int ID = APPLICATION_INFO_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&info.provenance, sizeof(info.provenance));
  pr_fwrite(f, (char *)&(info.start), sizeof(info.start));
  pr_fwrite(f, (char *)&(info.stop), sizeof(info.stop));
  pr_fwrite(f, (char *)&(info.proc_id), sizeof(info.proc_id));
  pr_fwrite(f, (char *)&info.fevent.id, sizeof(info.fevent.id));
}

void Profiler::serialize(const ThreadProfiler::SpawnInfo &info) const
{
  int ID = SPAWN_INFO_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&info.fevent.id, sizeof(info.fevent.id));
  pr_fwrite(f, (char *)&(info.spawn), sizeof(info.spawn));
}

void Profiler::finalize(void) {
  {
    AutoLock p_lock(profiler_lock);
#ifdef DEBUG_REALM
    bool done = true;
    for(unsigned idx = 0; idx < ThreadProfiler::LAST_PROF; idx++) {
      if(total_outstanding_requests[idx] == 0)
        continue;
      done = false;
      break;
    }
    if(!done)
      done_event = Realm::UserEvent::create_user_event();
#else
    if(total_outstanding_requests.load() > 0)
      done_event = Realm::UserEvent::create_user_event();
#endif
  }
  if (done_event.exists())
    done_event.wait();
  // Finalize all the instances
  for (std::vector<ThreadProfiler *>::const_iterator it =
           thread_profilers.begin();
       it != thread_profilers.end(); it++)
    (*it)->finalize();
  if (enabled) {
    // Get the calibration error
    const long long calibration_error = Realm::Clock::get_calibration_error();
    int ID = CALIBRATION_ERR_ID;
    pr_fwrite(f, (char *)&ID, sizeof(ID));
    pr_fwrite(f, (char *)&(calibration_error), sizeof(calibration_error));
    // Close the file
    pr_fflush(f, Z_FULL_FLUSH);
    pr_fclose(f);
  }
}

void Profiler::defer_shutdown(Event precondition, int code) {
  // If we're on node 0 then we can do the work
  if (local_proc.address_space() == 0) {
    AutoLock p_lock(profiler_lock);
    shutdown_precondition = precondition;
    return_code = code;
    has_shutdown = true;
    if (shutdown_wait.exists())
      // Protect from application level poison
      shutdown_wait.trigger(precondition);
  } else {
    // Send a message to node 0 informing it that we received the shutdown
    ShutdownArgs args{precondition, code};
    // Find a processor on node 0
    Realm::Machine::ProcessorQuery proc_finder(Machine::get_machine());
    for (Realm::Machine::ProcessorQuery::iterator it = proc_finder.begin();
         it != proc_finder.end(); it++) {
      if (it->address_space() != 0)
        continue;
      const Realm::ProfilingRequestSet no_requests;
      it->spawn(SHUTDOWN_TASK_ID, &args, sizeof(args), no_requests);
      return;
    }
    std::abort(); // should never get here
  }
}

void Profiler::wait_for_shutdown(void) {
  assert(local_proc.address_space() == 0);
  AutoLock p_lock(profiler_lock);
  if (!has_shutdown) {
    shutdown_wait = Realm::UserEvent::create_user_event();
    p_lock.release();
    bool ignore; // ignore poison from the application
    shutdown_wait.wait_faultaware(ignore);
  } else {
    p_lock.release();
    bool ignore; // ignore poison from the application
    shutdown_precondition.wait_faultaware(ignore);
  }
}

void Profiler::perform_shutdown(void) {
  assert(local_proc.address_space() == 0);
  assert(has_shutdown);
  Realm::Runtime::get_runtime().shutdown(shutdown_precondition, return_code);
}

#ifdef DEBUG_REALM
void Profiler::increment_total_outstanding_requests(
    ThreadProfiler::ProfKind kind) {
  AutoLock p_lock(profiler_lock);
  total_outstanding_requests[kind]++;
}

void Profiler::decrement_total_outstanding_requests(
    ThreadProfiler::ProfKind kind) {
  AutoLock p_lock(profiler_lock);
  assert(total_outstanding_requests[kind] > 0);
  if(--total_outstanding_requests[kind] > 0)
    return;
  for (unsigned idx = 0; idx < ThreadProfiler::LAST_PROF; idx++) {
    if (idx == kind)
      continue;
    if (total_outstanding_requests[idx] == 0)
      continue;
    return;
  }
  Realm::UserEvent to_trigger = done_event;
  p_lock.release();
  if (to_trigger.exists())
    to_trigger.trigger(shutdown_precondition);
}
#else
void Profiler::increment_total_outstanding_requests(void) {
  total_outstanding_requests.fetch_add(1);
}

void Profiler::decrement_total_outstanding_requests(void) {
  unsigned previous = total_outstanding_requests.fetch_sub(1);
  assert(previous > 0);
  if (previous == 1) {
    Realm::UserEvent to_trigger = Realm::UserEvent::NO_USER_EVENT;
    AutoLock p_lock(profiler_lock);
    if ((total_outstanding_requests.load() == 0) && done_event.exists()) {
      to_trigger = done_event;
    }
    p_lock.release();
    if (to_trigger.exists())
      to_trigger.trigger(shutdown_precondition);
  }
}
#endif

Processor Profiler::get_remote_processor(AddressSpace target) {
  {
    AutoLock p_lock(profiler_lock, false /*exclusive*/);
    std::map<AddressSpace, Processor>::const_iterator finder =
        remote_processors.find(target);
    if(finder != remote_processors.end())
      return finder->second;
  }
  Realm::Machine::ProcessorQuery remote_procs(Machine::get_machine());
  remote_procs.only_kind(Processor::LOC_PROC);
  for (Realm::Machine::ProcessorQuery::iterator it = remote_procs.begin();
       it != remote_procs.end(); it++) {
    if (it->address_space() != target)
      continue;
    AutoLock p_lock(profiler_lock);
    remote_processors.emplace(std::make_pair(target, *it));
    return *it;
  }
  // should never get here
  std::abort();
}

Processor Profiler::get_implicit_processor(void) {
  AutoLock p_lock(profiler_lock);
  if (!implicit_proc.exists()) {
    // Synthesize an implicit processor name
    // Count how many local processors are on this node
    Machine machine = Machine::get_machine();
    Realm::Machine::ProcessorQuery local_procs(machine);
    local_procs.local_address_space();
    const size_t count = local_procs.count();
    assert(count > 0);
    implicit_proc.id =
        Realm::ID::make_processor(local_proc.address_space(), count).id;
    // Record the processor descriptor
    ThreadProfiler::ProcDesc proc_desc;
    proc_desc.proc_id = implicit_proc.id;
    proc_desc.kind = Processor::IO_PROC;
    serialize(proc_desc);
    recorded_processors.push_back(implicit_proc);
    std::sort(recorded_processors.begin(), recorded_processors.end());
  }
  const Processor result = implicit_proc;
  return result;
}

void Profiler::record_thread_profiler(ThreadProfiler *profiler) {
  AutoLock p_lock(profiler_lock);
  thread_profilers.push_back(profiler);
}

unsigned long long Profiler::find_backtrace_id(Backtrace &bt) {
  const uintptr_t hash = bt.hash();
  {
    AutoLock p_lock(profiler_lock, false /*exclusive*/);
    std::map<uintptr_t, unsigned long long>::const_iterator finder =
        backtrace_ids.find(hash);
    if(finder != backtrace_ids.end())
      return finder->second;
  }
  // First time seeing this backtrace so capture the symbols
  std::stringstream ss;
  ss << bt;
  const std::string str = ss.str();
  // Compute the backtrace based on the symbols using a deterministic
  // hash function, we can't use std::hash because it is not
  // guaranteed to be the same across processes
  std::vector<std::string> symbols;
  bt.print_symbols(symbols);
  unsigned long long result = 0;
  constexpr unsigned long long large_prime = 18446744073709551557;
  for(std::vector<std::string>::const_iterator it = symbols.begin(); it != symbols.end();
      it++) {
    for(unsigned idx = 0; idx < it->size(); idx++) {
      const unsigned long long c = it->at(idx);
      result = ((result << (8 * sizeof(char))) + c) % large_prime;
    }
  }
  // Now retake the lock and see if we lost the race
  AutoLock p_lock(profiler_lock);
  std::map<uintptr_t, unsigned long long>::const_iterator finder =
      backtrace_ids.find(hash);
  if (finder != backtrace_ids.end()) {
    assert(result == finder->second);
    return result;
  }
  // Save the backtrace into the file
  int ID = BACKTRACE_DESC_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&result, sizeof(result));
  pr_fwrite(f, str.c_str(), str.size() + 1);
  backtrace_ids[hash] = result;
  return result;
}

unsigned long long Profiler::find_provenance_id(const std::string_view &provenance)
{
  // Compute a simple deterministic hash of this string, we can't use std::hash
  // becasue it is not guaranteed to be the same across processes and we want
  // this to be the same across all processes
  unsigned long long hash = 0;
  constexpr unsigned long long large_prime = 18446744073709551557;
  for(unsigned idx = 0; idx < provenance.size(); idx++) {
    const unsigned long long c = provenance[idx];
    hash = ((hash << (8 * sizeof(char))) + c) % large_prime;
  }
  {
    AutoLock p_lock(profiler_lock, false /*exclusive*/);
    std::unordered_set<unsigned long long>::const_iterator finder =
        provenance_ids.find(hash);
    if(finder != provenance_ids.end())
      return hash;
  }
  AutoLock p_lock(profiler_lock);
  if(provenance_ids.insert(hash).second) {
    // Need to save the provenance string now
    int ID = PROVENANCE_ID;
    pr_fwrite(f, (char *)&ID, sizeof(ID));
    pr_fwrite(f, (char *)&hash, sizeof(hash));
    pr_fwrite(f, provenance.data(), provenance.size());
    char null = '\0';
    pr_fwrite(f, &null, sizeof(null));
  }
  return hash;
}

void Profiler::record_task(Processor::TaskFuncID task_id,
                           const std::string_view &task_name)
{
  AutoLock p_lock(profiler_lock);
  // Check to see if we've registered this task ID before
  if(!task_name.empty()) {
    // Custom name from the user, always save it and overwrite
    int ID = TASK_KIND_ID;
    pr_fwrite(f, (char *)&ID, sizeof(ID));
    pr_fwrite(f, (char *)&(task_id), sizeof(task_id));
    pr_fwrite(f, (char *)task_name.data(), task_name.size());
    char null = '\0';
    pr_fwrite(f, &null, sizeof(null));
    bool overwrite = true; // always overwrite
    pr_fwrite(f, (char *)&(overwrite), sizeof(overwrite));
    registered_tasks.insert(task_id);
  } else if(registered_tasks.find(task_id) == registered_tasks.end()) {
    std::string name;
    name += "Task ";
    name += std::to_string(task_id);
    int ID = TASK_KIND_ID;
    pr_fwrite(f, (char *)&ID, sizeof(ID));
    pr_fwrite(f, (char *)&task_id, sizeof(task_id));
    pr_fwrite(f, name.c_str(), name.size() + 1);
    bool overwrite = false;
    pr_fwrite(f, (char *)&(overwrite), sizeof(overwrite));
    registered_tasks.insert(task_id);
  }
}

void Profiler::record_variant(Processor::TaskFuncID task_id,
                              Processor::Kind kind) {
  static const char *proc_names[] = {
#define PROC_NAMES(name, desc) #name,
      REALM_PROCESSOR_KINDS(PROC_NAMES)
#undef PROC_NAMES
  };
  char name[128];
  snprintf(name, sizeof(name), "%s Variant of Task %d", proc_names[kind],
           task_id);
  AutoLock p_lock(profiler_lock);
  int ID = TASK_VARIANT_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  pr_fwrite(f, (char *)&(task_id), sizeof(task_id));
  unsigned variant_id = kind;
  pr_fwrite(f, (char *)&(variant_id), sizeof(variant_id));
  pr_fwrite(f, name, strlen(name) + 1);
}

void Profiler::record_memory(const Memory &m) {
  {
    AutoLock p_lock(profiler_lock, false /*exclusive*/);
    if(std::binary_search(recorded_memories.begin(), recorded_memories.end(), m))
      return;
  }
  AutoLock p_lock(profiler_lock);
  if(std::binary_search(recorded_memories.begin(), recorded_memories.end(), m))
    return;
  // Also log all the affinities for this memory
  std::vector<Memory> memories_to_log(1, m);
  record_affinities(memories_to_log);
}

void Profiler::record_processor(const Processor &p) {
  {
    AutoLock p_lock(profiler_lock, false /*exclusive*/);
    if(std::binary_search(recorded_processors.begin(), recorded_processors.end(), p))
      return;
  }
  AutoLock p_lock(profiler_lock);
  if(std::binary_search(recorded_processors.begin(), recorded_processors.end(), p))
    return;
  // Record the processor descriptor
  ThreadProfiler::ProcDesc proc_desc;
  proc_desc.proc_id = p.id;
  proc_desc.kind = p.kind();
#ifdef REALM_USE_CUDA
  if (!Realm::Cuda::get_cuda_device_uuid(p, &proc_desc.cuda_device_uuid))
    proc_desc.cuda_device_uuid[0] = 0;
#endif
  serialize(proc_desc);
  recorded_processors.push_back(p);
  std::sort(recorded_processors.begin(), recorded_processors.end());
  std::vector<Memory> memories_to_log;
  std::vector<Machine::ProcessorMemoryAffinity> affinities;
  Machine::get_machine().get_proc_mem_affinity(affinities, p);
  for (std::vector<Machine::ProcessorMemoryAffinity>::const_iterator pit =
           affinities.begin();
       pit != affinities.end(); pit++)
    if (!std::binary_search(recorded_memories.begin(), recorded_memories.end(),
                            pit->m))
      memories_to_log.push_back(pit->m);
  record_affinities(memories_to_log);
}

void Profiler::record_affinities(std::vector<Memory> &memories_to_log) {
  while (!memories_to_log.empty()) {
    const Memory m = memories_to_log.back();
    memories_to_log.pop_back();
    // Eagerly log the processor description to the logging file so
    // that it appears before anything that needs it
    const ThreadProfiler::MemDesc mem = {m.id, m.kind(), m.capacity()};
    serialize(mem);
    recorded_memories.push_back(m);
    std::sort(recorded_memories.begin(), recorded_memories.end());
    std::vector<Machine::ProcessorMemoryAffinity> memory_affinities;
    Machine machine = Machine::get_machine();
    machine.get_proc_mem_affinity(memory_affinities, Processor::NO_PROC, m);
    for (std::vector<Machine::ProcessorMemoryAffinity>::const_iterator mit =
             memory_affinities.begin();
         mit != memory_affinities.end(); mit++) {
      if (!std::binary_search(recorded_processors.begin(),
                              recorded_processors.end(), mit->p)) {
        ThreadProfiler::ProcDesc proc;
        proc.proc_id = mit->p.id;
        proc.kind = mit->p.kind();
#ifdef REALM_USE_CUDA
        if (!Realm::Cuda::get_cuda_device_uuid(mit->p, &proc.cuda_device_uuid))
          proc.cuda_device_uuid[0] = 0;
#endif
        serialize(proc);
        recorded_processors.push_back(mit->p);
        std::sort(recorded_processors.begin(), recorded_processors.end());
        std::vector<Machine::ProcessorMemoryAffinity> processor_affinities;
        machine.get_proc_mem_affinity(processor_affinities, mit->p);
        for (std::vector<Machine::ProcessorMemoryAffinity>::const_iterator pit =
                 processor_affinities.begin();
             pit != processor_affinities.end(); pit++)
          if (!std::binary_search(recorded_memories.begin(),
                                  recorded_memories.end(), pit->m))
            memories_to_log.push_back(pit->m);
      }
      const ThreadProfiler::ProcMemDesc info = {mit->p.id, m.id, mit->bandwidth,
                                                mit->latency};
      serialize(info);
    }
  }
}

void Profiler::record_implicit(
    long long start_time, Realm::Event fevent,
    const std::vector<ThreadProfiler::WaitInfo> &wait_intervals) {
  long long stop_time = Realm::Clock::current_time_in_nanoseconds();
  int ID = IMPLICIT_TASK_INFO_ID;
  pr_fwrite(f, (char *)&ID, sizeof(ID));
  const unsigned long long op_id = fevent.id;
  pr_fwrite(f, (char *)&(op_id), sizeof(op_id));
  // Use the NOP task ID for external threads
  const Processor::TaskFuncID task_id = Processor::TASK_ID_PROCESSOR_NOP;
  pr_fwrite(f, (char *)&(task_id), sizeof(task_id));
  // this is the external thread variant
  unsigned variant_id = Processor::NO_KIND;
  pr_fwrite(f, (char *)&(variant_id), sizeof(variant_id));
  assert(implicit_proc.exists());
  pr_fwrite(f, (char *)&(implicit_proc.id), sizeof(implicit_proc.id));
  pr_fwrite(f, (char *)&(start_time), sizeof(start_time));
  pr_fwrite(f, (char *)&(start_time), sizeof(start_time));
  pr_fwrite(f, (char *)&(start_time), sizeof(start_time));
  pr_fwrite(f, (char *)&(stop_time), sizeof(stop_time));
  pr_fwrite(f, (char *)&(Event::NO_EVENT), sizeof(Event::NO_EVENT));
  pr_fwrite(f, (char *)&(Event::NO_EVENT), sizeof(Event::NO_EVENT));
  pr_fwrite(f, (char *)&(fevent), sizeof(fevent));
  ID = TASK_WAIT_INFO_ID;
  for (std::vector<ThreadProfiler::WaitInfo>::const_iterator it =
           wait_intervals.begin();
       it != wait_intervals.end(); it++) {
    const ThreadProfiler::WaitInfo &wait_info = *it;
    pr_fwrite(f, (char *)&ID, sizeof(ID));
    pr_fwrite(f, (char *)&(op_id), sizeof(op_id));
    pr_fwrite(f, (char *)&(task_id), sizeof(task_id));
    pr_fwrite(f, (char *)&(variant_id), sizeof(variant_id));
    pr_fwrite(f, (char *)&(wait_info.wait_start), sizeof(wait_info.wait_start));
    pr_fwrite(f, (char *)&(wait_info.wait_ready), sizeof(wait_info.wait_ready));
    pr_fwrite(f, (char *)&(wait_info.wait_end), sizeof(wait_info.wait_end));
    pr_fwrite(f, (char *)&(wait_info.wait_event), sizeof(wait_info.wait_event));
  }
}

void Profiler::record_remote_notification(Realm::Event notified) {
  AutoLock p_lock(profiler_lock);
  while (!remote_notifications.empty() &&
         remote_notifications.front().has_triggered())
    remote_notifications.pop_front();
  remote_notifications.push_back(notified);
}

Realm::Event Profiler::get_remote_done(void) {
  // No need for a lock, shouldn't be racing with anything
  while (!remote_notifications.empty() &&
         remote_notifications.front().has_triggered())
    remote_notifications.pop_front();
  if (remote_notifications.empty())
    return Realm::Event::NO_EVENT;
  std::vector<Realm::Event> done(remote_notifications.begin(),
                                 remote_notifications.end());
  return Realm::Event::merge_events(done);
}

void Profiler::update_footprint(size_t diff, ThreadProfiler *profiler) {
  size_t footprint = total_memory_footprint.fetch_add(diff) + diff;
  if (footprint > output_footprint_threshold) {
    // An important bit of logic here, if we're over the threshold then
    // we want to have a little bit of a feedback loop so the more over
    // the limit we are then the more time we give the profiler to dump
    // out things to the output file. We'll try to make this continuous
    // so there are no discontinuities in performance. If the threshold
    // is zero we'll just choose an arbitrarily large scale factor to
    // ensure that things work properly.
    double over_scale =
        output_footprint_threshold == 0
            ? double(1 << 20)
            : double(footprint) / double(output_footprint_threshold);
    // Let's actually make this quadratic so it's not just linear
    if (output_footprint_threshold > 0)
      over_scale *= over_scale;
    // Need a lock to protect the file
    AutoLock p_lock(profiler_lock);
    diff = profiler->dump_inter(over_scale * target_latency);
#ifndef NDEBUG
    footprint =
#endif
        total_memory_footprint.fetch_sub(diff);
    assert(footprint >= diff); // check for wrap-around
  }
}

/*static*/ void Profiler::callback(const void *args, size_t arglen,
                                   const void *user_args, size_t user_arglen,
                                   Realm::Processor p) {
  Realm::ProfilingResponse response(args, arglen);
  ThreadProfiler &thread_profiler = ThreadProfiler::get_thread_profiler();
  thread_profiler.process_response(response);
}

/*static*/ void Profiler::wrapper(const void *args, size_t arglen,
                                  const void *user_args, size_t user_arglen,
                                  Realm::Processor p) {
  const WrapperArgs *wargs = static_cast<const WrapperArgs *>(args);
  assert(arglen == (sizeof(WrapperArgs) + wargs->arglen));
  assert(wargs->fevent.exists());
  assert(wargs->spawn_time != 0);
  ProfilingRequestSet requests;
  ThreadProfiler::get_thread_profiler().add_task_request(
      requests, wargs->task_id, wargs->wait_on, wargs->fevent, wargs->spawn_time);

  const Realm::Event done = p.spawn(
      wargs->task_id, (wargs->arglen > 0) ? (void *)(wargs + 1) : nullptr,
      wargs->arglen, requests, wargs->wait_on, wargs->priority);
  if (wargs->after.exists()) {
    wargs->after.trigger(done);
    assert(wargs->origin.exists());
    // Send a trigger message back to the origin
    const ThreadProfiler::EventTriggerInfo info{wargs->after, wargs->fevent,
                                                done, wargs->spawn_time};
    const Realm::Event notified =
        wargs->origin.spawn(Profiler::TRIGGER_TASK_ID, &info, sizeof(info));
    Profiler::get_profiler().record_remote_notification(notified);
  } else
    done.wait();
}

/*static*/ void Profiler::trigger(const void *args, size_t arglen,
                                  const void *user_args, size_t user_arglen,
                                  Realm::Processor p) {
  ThreadProfiler &thread_profiler = ThreadProfiler::get_thread_profiler();
  thread_profiler.process_trigger(args, arglen);
}

/*static*/ void Profiler::external(const void *args, size_t arglen, const void *user_args,
                                   size_t user_arglen, Realm::Processor p)
{
  Realm::ProfilingResponse response(args, arglen);
  ThreadProfiler &thread_profiler = ThreadProfiler::get_thread_profiler();
  thread_profiler.process_external(response);
}

/*static*/ void Profiler::shutdown(const void *args, size_t arglen,
                                   const void *user_args, size_t user_arglen,
                                   Realm::Processor p) {
  assert(arglen == sizeof(ShutdownArgs));
  const ShutdownArgs *sargs = static_cast<const ShutdownArgs *>(args);
  Profiler::get_profiler().defer_shutdown(sargs->precondition, sargs->code);
}

/*static*/ Profiler &Profiler::get_profiler(void) {
  static Profiler singleton;
  return singleton;
}

void Runtime::parse_command_line(int argc, char **argv) {
  Profiler::get_profiler().parse_command_line(argc, argv);
  Realm::Runtime::parse_command_line(argc, argv);
}

void Runtime::parse_command_line(std::vector<std::string> &cmdline,
                                 bool remove_args) {
  Profiler::get_profiler().parse_command_line(cmdline, remove_args);
  Realm::Runtime::parse_command_line(cmdline, remove_args);
}

bool Runtime::configure_from_command_line(int argc, char **argv) {
  Profiler::get_profiler().parse_command_line(argc, argv);
  return Realm::Runtime::configure_from_command_line(argc, argv);
}

bool Runtime::configure_from_command_line(std::vector<std::string> &cmdline,
                                          bool remove_args) {
  Profiler::get_profiler().parse_command_line(cmdline, remove_args);
  return Realm::Runtime::configure_from_command_line(cmdline, remove_args);
}

void Runtime::start(void) {
  Profiler::get_profiler().initialize();
  Realm::Runtime::start();
  // Register our profiling callback function with all the local processors
  Machine machine = Machine::get_machine();
  Realm::Machine::ProcessorQuery local_procs(machine);
  local_procs.local_address_space();
  std::vector<Realm::Event> registered;
  const Realm::ProfilingRequestSet no_requests;
  const CodeDescriptor callback(Profiler::callback);
  const CodeDescriptor wrapper(Profiler::wrapper);
  const CodeDescriptor trigger(Profiler::trigger);
  const CodeDescriptor external(Profiler::external);
  const CodeDescriptor shutdown(Profiler::shutdown);
  for (Realm::Machine::ProcessorQuery::iterator it = local_procs.begin();
       it != local_procs.end(); it++) {
    Realm::Event done =
        it->register_task(Profiler::CALLBACK_TASK_ID, callback, no_requests);
    if (done.exists())
      registered.push_back(done);
    done = it->register_task(Profiler::WRAPPER_TASK_ID, wrapper, no_requests);
    if (done.exists())
      registered.push_back(done);
    done = it->register_task(Profiler::TRIGGER_TASK_ID, trigger, no_requests);
    if(done.exists())
      registered.push_back(done);
    done = it->register_task(Profiler::EXTERNAL_TASK_ID, external, no_requests);
    if (done.exists())
      registered.push_back(done);
    done = it->register_task(Profiler::SHUTDOWN_TASK_ID, shutdown, no_requests);
    if (done.exists())
      registered.push_back(done);
  }
  if (!registered.empty())
    Realm::Event::merge_events(registered).wait();
}

bool Runtime::init(int *argc, char ***argv) {
  // if we get null pointers for argc and argv, use a local version so
  //  any changes from network_init are seen in configure_from_command_line
  int my_argc = 0;
  char **my_argv = 0;
  if (!argc)
    argc = &my_argc;
  if (!argv)
    argv = &my_argv;

  if (!Realm::Runtime::network_init(argc, argv))
    return false;
  if (!Realm::Runtime::create_configs(*argc, *argv))
    return false;
  if (!configure_from_command_line(*argc, *argv))
    return false;
  start();
  return true;
}

Event Processor::register_task(TaskFuncID task_id,
                               const CodeDescriptor &codedesc,
                               const ProfilingRequestSet &prs,
                               const void *user_data,
                               size_t user_data_len) const {
  Profiler &profiler = Profiler::get_profiler();
  profiler.record_task(task_id, std::string_view());
  profiler.record_variant(task_id, kind());
  return Realm::Processor::register_task(task_id, codedesc, prs, user_data,
                                         user_data_len);
}

/*static*/ Event
Processor::register_task_by_kind(Kind kind, bool global, TaskFuncID task_id,
                                 const CodeDescriptor &codedesc,
                                 const ProfilingRequestSet &prs,
                                 const void *user_data, size_t user_data_len) {
  Profiler &profiler = Profiler::get_profiler();
  profiler.record_task(task_id, std::string_view());
  if (kind == Processor::NO_KIND) {
    Realm::Machine::ProcessorQuery local_procs(Machine::get_machine());
    local_procs.local_address_space();
    std::vector<Processor::Kind> kinds;
    for (Realm::Machine::ProcessorQuery::iterator it = local_procs.begin();
         it != local_procs.end(); it++) {
      const Processor::Kind kind = it->kind();
      if (std::binary_search(kinds.begin(), kinds.end(), kind))
        continue;
      profiler.record_variant(task_id, kind);
      kinds.push_back(kind);
      std::sort(kinds.begin(), kinds.end());
    }
  } else {
    profiler.record_variant(task_id, kind);
  }
  return Realm::Processor::register_task_by_kind(
      kind, global, task_id, codedesc, prs, user_data, user_data_len);
}

Event Processor::spawn(TaskFuncID func_id, const void *args, size_t arglen,
                       const ProfilingRequestSet &requests, Event wait_on,
                       int priority) const {
  ThreadProfiler &profiler = ThreadProfiler::get_thread_profiler();
  if (profiler.local_proc.address_space() != address_space()) {
    // If this processor is not local we need a wrapper task so that it
    // looks like the fevent is local to the current node
    if (requests.empty()) {
      // Make a user event that will be triggered by the wrapper task
      const Realm::UserEvent after = Realm::UserEvent::create_user_event();
      // Lie aobut the trigger time so that it looks like it is always
      // before the spawn so it never appears like it is on the critical path
      const ThreadProfiler::timestamp_t spawn_time =
          Realm::Clock::current_time_in_nanoseconds();
      if (arglen > 0) {
        const size_t buffer_size = sizeof(Profiler::WrapperArgs) + arglen;
        Profiler::WrapperArgs *wrapper_args =
            (Profiler::WrapperArgs *)std::malloc(buffer_size);
        wrapper_args->wait_on = wait_on;
        wrapper_args->arglen = arglen;
        wrapper_args->task_id = func_id;
        wrapper_args->priority = priority;
        wrapper_args->fevent = profiler.get_fevent();
        wrapper_args->spawn_time = spawn_time;
        wrapper_args->after = after;
        wrapper_args->origin = profiler.get_callback_processor();
        std::memcpy(wrapper_args + 1, args, arglen);
        Realm::Processor::spawn(Profiler::WRAPPER_TASK_ID, wrapper_args,
                                buffer_size, Event::NO_EVENT, priority);
        std::free(wrapper_args);
      } else {
        Profiler::WrapperArgs wrapper_args{
            wait_on,  0 /*arglen*/,          func_id,
            priority, profiler.get_fevent(), spawn_time,
            after,    profiler.local_proc};
        Realm::Processor::spawn(Profiler::WRAPPER_TASK_ID, &wrapper_args,
                                sizeof(wrapper_args), Event::NO_EVENT,
                                priority);
      }
      return after;
    } else {
      // TODO: need to serialize up requests and move them remotely
      std::abort();
    }
  } else {
    // Can spawn this directly locally
    ProfilingRequestSet alt_requests(requests);
    profiler.add_task_request(alt_requests, func_id, wait_on,
                              profiler.get_fevent());
    return Realm::Processor::spawn(func_id, args, arglen, alt_requests, wait_on,
                                   priority);
  }
}

bool Runtime::register_task(Processor::TaskFuncID task_id,
                            Processor::TaskFuncPtr taskptr) {
  // since processors are the same size we can just cast the function pointer
  Realm::Processor::TaskFuncPtr altptr =
      reinterpret_cast<Realm::Processor::TaskFuncPtr>(taskptr);
  Profiler &profiler = Profiler::get_profiler();
  // Record that we have a task with this task ID
  profiler.record_task(task_id, std::string_view());
  // Get all the local processors and record that we have a variant for each
  // kind of local processor
  Realm::Machine::ProcessorQuery local_procs(Machine::get_machine());
  local_procs.local_address_space();
  std::vector<Processor::Kind> kinds;
  for (Realm::Machine::ProcessorQuery::iterator it = local_procs.begin();
       it != local_procs.end(); it++) {
    const Processor::Kind kind = it->kind();
    if (std::binary_search(kinds.begin(), kinds.end(), kind))
      continue;
    profiler.record_variant(task_id, kind);
    kinds.push_back(kind);
    std::sort(kinds.begin(), kinds.end());
  }
  return Realm::Runtime::register_task(task_id, altptr);
}

Event Runtime::collective_spawn(Processor target_proc,
                                Processor::TaskFuncID task_id, const void *args,
                                size_t arglen, Event wait_on, int priority) {
  // Launch a wrapper task that will actually spawn the task on the processor
  // with extra profiling
  const Event fevent = ThreadProfiler::get_thread_profiler().get_fevent();
  const ThreadProfiler::timestamp_t spawn_time =
      Realm::Clock::current_time_in_nanoseconds();
  if (arglen > 0) {
    const size_t buffer_size = sizeof(Profiler::WrapperArgs) + arglen;
    Profiler::WrapperArgs *wrapper_args =
        (Profiler::WrapperArgs *)std::malloc(buffer_size);
    wrapper_args->wait_on = wait_on;
    wrapper_args->arglen = arglen;
    wrapper_args->task_id = task_id;
    wrapper_args->priority = priority;
    wrapper_args->fevent = fevent;
    wrapper_args->spawn_time = spawn_time;
    wrapper_args->after = Realm::UserEvent::NO_USER_EVENT;
    wrapper_args->origin = Realm::Processor::NO_PROC;
    std::memcpy(wrapper_args + 1, args, arglen);
    const Event result = Realm::Runtime::collective_spawn(
        target_proc, Profiler::WRAPPER_TASK_ID, wrapper_args, buffer_size,
        wait_on, priority);
    std::free(wrapper_args);
    return result;
  } else {
    const Profiler::WrapperArgs wrapper_args{wait_on,
                                             0 /*arglen*/,
                                             task_id,
                                             priority,
                                             fevent,
                                             spawn_time,
                                             Realm::UserEvent::NO_USER_EVENT,
                                             Realm::Processor::NO_PROC};
    return Realm::Runtime::collective_spawn(
        target_proc, Profiler::WRAPPER_TASK_ID, &wrapper_args,
        sizeof(wrapper_args), wait_on, priority);
  }
}

Event Runtime::collective_spawn_by_kind(Processor::Kind target_kind,
                                        Processor::TaskFuncID task_id,
                                        const void *args, size_t arglen,
                                        bool one_per_node, Event wait_on,
                                        int priority) {
  // Launch a wrapper task that will actually spawn the tasks on the processor
  // with extra profiling
  const Event fevent = ThreadProfiler::get_thread_profiler().get_fevent();
  const ThreadProfiler::timestamp_t spawn_time =
      Realm::Clock::current_time_in_nanoseconds();
  if (arglen > 0) {
    const size_t buffer_size = sizeof(Profiler::WrapperArgs) + arglen;
    Profiler::WrapperArgs *wrapper_args =
        (Profiler::WrapperArgs *)std::malloc(buffer_size);
    wrapper_args->wait_on = wait_on;
    wrapper_args->arglen = arglen;
    wrapper_args->task_id = task_id;
    wrapper_args->priority = priority;
    wrapper_args->fevent = fevent;
    wrapper_args->spawn_time = spawn_time;
    wrapper_args->after = Realm::UserEvent::NO_USER_EVENT;
    wrapper_args->origin = Realm::Processor::NO_PROC;
    std::memcpy(wrapper_args + 1, args, arglen);
    const Event result = Realm::Runtime::collective_spawn_by_kind(
        target_kind, Profiler::WRAPPER_TASK_ID, wrapper_args, buffer_size,
        one_per_node, wait_on, priority);
    std::free(wrapper_args);
    return result;
  } else {
    const Profiler::WrapperArgs wrapper_args{wait_on,
                                             0 /*arglen*/,
                                             task_id,
                                             priority,
                                             fevent,
                                             spawn_time,
                                             Realm::UserEvent::NO_USER_EVENT,
                                             Realm::Processor::NO_PROC};
    return Realm::Runtime::collective_spawn_by_kind(
        target_kind, Profiler::WRAPPER_TASK_ID, &wrapper_args,
        sizeof(wrapper_args), one_per_node, wait_on, priority);
  }
}

void Runtime::shutdown(Event wait_on, int result_code) {
  // Buffer this until we know all the profiling is done
  Profiler::get_profiler().defer_shutdown(wait_on, result_code);
}

int Runtime::wait_for_shutdown(void) {
  Profiler &profiler = Profiler::get_profiler();
  const AddressSpace local_space =
      profiler.get_local_processor().address_space();
  // Make sure node 0 has received the shutdown notification before we try
  // to finalize any of the profilers
  if (local_space == 0)
    profiler.wait_for_shutdown();
  // Then do a barrier to notify all the other nodes that the shutdown has
  // been received and they can try to finalize their profiler
  Realm::Runtime::collective_spawn_by_kind(Processor::Kind::NO_KIND,
                                           Processor::TASK_ID_PROCESSOR_NOP,
                                           nullptr, 0, true /*one per process*/)
      .wait();
  // Now we need to do another barrier across all the nodes to make sure that
  // all of our asynchronous updates are done. Note it's not safe to do this
  // until we know that all Realm operations are issues which we only get after
  // the previous barrier.
  Realm::Runtime::collective_spawn_by_kind(
      Processor::Kind::NO_KIND, Processor::TASK_ID_PROCESSOR_NOP, nullptr, 0,
      true /*one per process*/, profiler.get_remote_done())
      .wait();
  // Now we can finalize the profiler
  profiler.finalize();
  // Do a barrier to make sure that everyone is done reporting their profiling
  Realm::Runtime::collective_spawn_by_kind(Processor::Kind::NO_KIND,
                                           Processor::TASK_ID_PROCESSOR_NOP,
                                           nullptr, 0, true /*one per process*/)
      .wait();
  // If we're node 0 now we can actually perform the shutdown
  if (local_space == 0)
    profiler.perform_shutdown();
  return Realm::Runtime::wait_for_shutdown();
}

/*static*/ Runtime Runtime::get_runtime(void) {
  return Realm::Runtime::get_runtime();
}

Module *Runtime::get_module_untyped(const char *name)
{
  Module *result = Realm::Runtime::get_module_untyped(name);
#ifdef REALM_USE_CUDA
  if(strcmp(name, "cuda") == 0) {
    Realm::Cuda::CudaModule *cuda_module =
        dynamic_cast<Realm::Cuda::CudaModule *>(result);
    assert(cuda_module != nullptr);
    Profiler &profiler = Profiler::get_profiler();
    return profiler.find_or_create_cuda_module(cuda_module);
  }
#endif
  return result;
}

#ifdef REALM_USE_CUDA
Cuda::CudaModule *Profiler::find_or_create_cuda_module(Realm::Cuda::CudaModule *mod)
{
  AutoLock p_lock(profiler_lock);
  if(cuda_module == nullptr)
    cuda_module = new Cuda::CudaModule(mod);
  return cuda_module;
}

namespace Cuda {
  CudaModule::CudaModule(Realm::Cuda::CudaModule *intern)
    : Realm::Module(intern->get_name())
    , internal(intern)
  {}

  CudaModule::~CudaModule(void) {}

  Event CudaModule::make_realm_event(CUevent_st *cuda_event)
  {
    const Realm::Event result = internal->make_realm_event(cuda_event);
    ThreadProfiler::get_thread_profiler().record_external_event(result, "CUDA Event");
    return result;
  }

  Event CudaModule::make_realm_event(CUstream_st *cuda_stream)
  {
    const Realm::Event result = internal->make_realm_event(cuda_stream);
    ThreadProfiler::get_thread_profiler().record_external_event(result, "CUDA Stream");
    return result;
  }
} // namespace Cuda
#endif

} // namespace PRealm
