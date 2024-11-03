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
// Need this so we know which version of legion prof we're working with
#include "legion/legion_profiling_serializer.h"
#include <zlib.h>
// pr_fopen expects filename to be a std::string
#define pr_fopen(filename, mode)      gzopen(filename.c_str(),mode)
#define pr_fwrite(f, data, num_bytes) gzwrite(f,data,num_bytes)
#define pr_fflush(f, mode)            gzflush(f,mode)
#define pr_fclose(f)                  gzclose(f)

namespace PRealm {

  const Event Event::NO_EVENT = Realm::Event::NO_EVENT;
  const UserEvent UserEvent::NO_USER_EVENT = Realm::UserEvent::NO_USER_EVENT;
  const Barrier Barrier::NO_BARRIER = Realm::Barrier::NO_BARRIER;
  const CompletionQueue CompletionQueue::NO_QUEUE = Realm::CompletionQueue::NO_QUEUE;
  const Reservation Reservation::NO_RESERVATION = Realm::Reservation::NO_RESERVATION;
  const Processor Processor::NO_PROC = Realm::Processor::NO_PROC;
  const ProcessorGroup ProcessorGroup::NO_PROC_GROUP = Realm::ProcessorGroup::NO_PROC_GROUP;
  const RegionInstance RegionInstance::NO_INST = RegionInstance();

  Realm::Logger log_pr("PRealm");
  thread_local ThreadProfiler *thread_profiler = nullptr;

  enum ProfKind {
    FILL_PROF,
    COPY_PROF,
    TASK_PROF,
    INST_PROF,
    PART_PROF,
    LAST_PROF, // must be last
  };

  struct ProfilingArgs {
  public:
    inline ProfilingArgs(ProfKind k) : kind(k) { }
  public:
    Event critical;
    Event provenance;
    union {
      Realm::Event inst;
      Processor::TaskFuncID task;
    } id;
    ProfKind kind;
  }; 

  class Profiler {
  public:
    Profiler(void);
    Profiler(const Profiler &rhs) = delete;
    Profiler& operator=(const Profiler &rhs) = delete;
  public:
    inline Processor get_local_processor(void) const { return local_proc; }
    void parse_command_line(int argc, char **argv);
    void parse_command_line(std::vector<std::string> &cmdline, bool remove_args);
    void initialize(void);
    void defer_shutdown(Event precondition, int return_code);
    void release_shutdown(void);
    void finalize(void);
    void record_thread_profiler(ThreadProfiler *profiler);
    unsigned long long find_backtrace_id(Backtrace &bt);
    static Profiler& get_profiler(void);
    static Processor::TaskFuncPtr callback; 
    static constexpr Realm::Processor::TaskFuncID CALLBACK_TASK_ID = 
      Realm::Processor::TASK_ID_FIRST_AVAILABLE;
    static_assert((CALLBACK_TASK_ID+1) == Processor::TASK_ID_FIRST_AVAILABLE);
    static constexpr int CALLBACK_TASK_PRIORITY = std::numeric_limits<int>::min();
  public:
#ifdef DEBUG_REALM
    void increment_total_outstanding_requests(ProfKind kind);
    void decrement_total_outstanding_requests(ProfKind kind);
#else
    void increment_total_outstanding_requests(void);
    void decrement_total_outstanding_requests(void);
#endif
    void update_footprint(size_t footprint, ThreadProfiler *profiler);
  private:
    void log_preamble(void) const;
    void log_configuration(Machine &machine, Processor local) const;
  private:
    Realm::FastReservation profiler_lock;
    std::vector<ThreadProfiler*> thread_profilers;
  private:
#ifdef DEBUG_REALM
    unsigned total_outstanding_requests[LAST_PROF];
#else
    std::atomic<unsigned> total_outstanding_requests;
#endif
  private:
    gzFile f;
    Processor local_proc;
    std::string file_name;
    size_t output_footprint_threshold;
    size_t target_latency; // in us
    const Realm::UserEvent done_event;
    std::map<uintptr_t,unsigned long long> backtrace_ids;
    unsigned long long next_backtrace_id;
    std::atomic<size_t> total_memory_footprint;
    Event shutdown_precondition;
    int return_code;
    unsigned total_address_spaces;
    bool has_shutdown;
  public:
    bool self_profile;
    bool no_critical_paths;
  };

  enum {
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
    TASK_VARIANT_ID,
    TASK_WAIT_INFO_ID,
    TASK_INFO_ID,
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
    BARRIER_ARRIVAL_INFO_ID,
    RESERVATION_ACQUIRE_INFO_ID,
    INSTANCE_READY_INFO_ID,
    COMPLETION_QUEUE_INFO_ID,
    PROFTASK_INFO_ID,
  };

  enum DepPartOpKind {
    DEP_PART_UNION = 0, // a single union
    DEP_PART_UNIONS = 1, // many parallel unions
    DEP_PART_UNION_REDUCTION = 2, // union reduction to a single space
    DEP_PART_INTERSECTION = 3, // a single intersection
    DEP_PART_INTERSECTIONS = 4, // many parallel intersections
    DEP_PART_INTERSECTION_REDUCTION = 5, // intersection reduction to a space
    DEP_PART_DIFFERENCE = 6, // a single difference
    DEP_PART_DIFFERENCES = 7, // many parallel differences
    DEP_PART_EQUAL = 8, // an equal partition operation
    DEP_PART_BY_FIELD = 9, // create a partition from a field
    DEP_PART_BY_IMAGE = 10, // create partition by image
    DEP_PART_BY_IMAGE_RANGE = 11, // create partition by image range
    DEP_PART_BY_PREIMAGE = 12, // create partition by preimage
    DEP_PART_BY_PREIMAGE_RANGE = 13, // create partition by preimage range
    DEP_PART_ASSOCIATION = 14, // create an association
    DEP_PART_WEIGHTS = 15, // create partition by weights
  };

  ThreadProfiler::ThreadProfiler(Processor local)
    : local_proc(local)
  {
  }

  void ThreadProfiler::add_fill_request(ProfilingRequestSet &requests,
        const std::vector<CopySrcDstField> &dsts, Event critical) 
  {
    Profiler &profiler = Profiler::get_profiler();
#ifdef DEBUG_REALM
    profiler.increment_total_outstanding_requests(FILL_PROF);
#else
    profiler.increment_total_outstanding_requests();
#endif
    ProfilingArgs args(FILL_PROF);
    args.critical = critical;
    Processor current = local_proc;
    if (!current.exists()) {
      current = profiler.get_local_processor();
    } else {
      args.provenance = Processor::get_current_finish_event();
    }
    Realm::ProfilingRequest &req = requests.add_request(current,
        Profiler::CALLBACK_TASK_ID, &args, sizeof(args), Profiler::CALLBACK_TASK_PRIORITY);
    req.add_measurement<
              ProfilingMeasurements::OperationTimeline>();
    req.add_measurement<
              ProfilingMeasurements::OperationMemoryUsage>();
    req.add_measurement<
              ProfilingMeasurements::OperationCopyInfo>();
    req.add_measurement<
              ProfilingMeasurements::OperationFinishEvent>();
  }

  void ThreadProfiler::add_copy_request(ProfilingRequestSet &requests,
        const std::vector<CopySrcDstField> &srcs,
        const std::vector<CopySrcDstField> &dsts, Event critical) 
  {
    Profiler &profiler = Profiler::get_profiler();
#ifdef DEBUG_REALM
    profiler.increment_total_outstanding_requests(COPY_PROF);
#else
    profiler.increment_total_outstanding_requests();
#endif
    ProfilingArgs args(COPY_PROF);
    args.critical = critical;
    Processor current = local_proc;
    if (!current.exists()) {
      current = profiler.get_local_processor();
    } else {
      args.provenance = Processor::get_current_finish_event();
    }
    Realm::ProfilingRequest &req = requests.add_request(current,
        Profiler::CALLBACK_TASK_ID, &args, sizeof(args), Profiler::CALLBACK_TASK_PRIORITY);
    req.add_measurement<
              ProfilingMeasurements::OperationTimeline>();
    req.add_measurement<
              ProfilingMeasurements::OperationMemoryUsage>();
    req.add_measurement<
              ProfilingMeasurements::OperationCopyInfo>();
    req.add_measurement<
              ProfilingMeasurements::OperationFinishEvent>();
  }

  void ThreadProfiler::add_task_request(ProfilingRequestSet &requests,
        Processor::TaskFuncID task_id, Event critical)
  {
    Profiler &profiler = Profiler::get_profiler();
#ifdef DEBUG_REALM
    profiler.increment_total_outstanding_requests(TASK_PROF);
#else
    profiler.increment_total_outstanding_requests();
#endif
    ProfilingArgs args(TASK_PROF);
    args.id.task = task_id;
    args.critical = critical;
    Processor current = local_proc;
    if (!current.exists()) {
      current = profiler.get_local_processor();
    } else {
      args.provenance = Processor::get_current_finish_event();
    }
    Realm::ProfilingRequest &req = requests.add_request(current,
        Profiler::CALLBACK_TASK_ID, &args, sizeof(args), Profiler::CALLBACK_TASK_PRIORITY);
    req.add_measurement<
              ProfilingMeasurements::OperationTimeline>();
    req.add_measurement<
              ProfilingMeasurements::OperationProcessorUsage>();
    req.add_measurement<
              ProfilingMeasurements::OperationEventWaits>();
    req.add_measurement<
              ProfilingMeasurements::OperationFinishEvent>();
  }

  Event ThreadProfiler::add_inst_request(ProfilingRequestSet &requests,
        const InstanceLayoutGeneric *ilg, Event critical)
  {
    Profiler &profiler = Profiler::get_profiler();
#ifdef DEBUG_REALM
    profiler.increment_total_outstanding_requests(TASK_PROF);
#else
    profiler.increment_total_outstanding_requests();
#endif
    // Make a unique event to name this event
    const Realm::UserEvent unique_name = Realm::UserEvent::create_user_event();
    unique_name.trigger();
    ProfilingArgs args(TASK_PROF);
    args.id.inst = unique_name;
    args.critical = critical;
    Processor current = local_proc;
    if (!current.exists()) {
      current = profiler.get_local_processor();
    } else {
      args.provenance = Processor::get_current_finish_event();
    }
    Realm::ProfilingRequest &req = requests.add_request(current,
        Profiler::CALLBACK_TASK_ID, &args, sizeof(args), Profiler::CALLBACK_TASK_PRIORITY);   
    req.add_measurement<
               ProfilingMeasurements::InstanceMemoryUsage>();
    req.add_measurement<
               ProfilingMeasurements::InstanceTimeline>();
    return unique_name;
  }

  void ThreadProfiler::record_event_wait(Event wait_on, Backtrace &bt)
  {
    if (!local_proc.exists())
      return;
    Profiler &profiler = Profiler::get_profiler();
    unsigned long long backtrace_id = profiler.find_backtrace_id(bt);
    event_wait_infos.emplace_back(EventWaitInfo{local_proc.id, 
        Processor::get_current_finish_event(), wait_on, backtrace_id});
    if (wait_on.is_barrier())
      record_barrier_usage(wait_on);
    profiler.update_footprint(sizeof(EventWaitInfo), this);
  }

  void ThreadProfiler::record_event_trigger(Event result, Event pre)
  {
    Profiler &profiler = Profiler::get_profiler();
    if (profiler.no_critical_paths)
      return;
    EventTriggerInfo &info = event_trigger_infos.emplace_back(EventTriggerInfo());
    info.performed = Realm::Clock::current_time_in_nanoseconds();
    info.result = result;
    info.precondition = pre;
    if (pre.is_barrier())
      record_barrier_usage(pre);
    if (local_proc.exists())
      info.provenance = Processor::get_current_finish_event();
    // See if we're triggering this event on the same node where it was made
    // If not we need to eventually notify the node where it was made that
    // it was triggered here and what the fevent was for it
    const Realm::ID id(result.id);
    const AddressSpace creator_node = id.event_creator_node();
    // TODO: handle sending the message to the creator node to let it know the provenance
    assert(creator_node == profiler.get_local_processor().address_space());
    profiler.update_footprint(sizeof(info), this);
  }

  void ThreadProfiler::record_event_poison(Event result)
  {
    Profiler &profiler = Profiler::get_profiler();
    if (profiler.no_critical_paths)
      return;
    EventPoisonInfo &info = event_poison_infos.emplace_back(EventPoisonInfo());
    info.performed = Realm::Clock::current_time_in_nanoseconds();
    info.result = result;
    if (local_proc.exists())
      info.provenance = Processor::get_current_finish_event();
    // See if we're poisoning this event on the same node where it was made
    // If not we need to eventually notify the node where it was made that
    // it was triggered here and what the fevent was for it
    const Realm::ID id(result.id);
    const AddressSpace creator_node = id.event_creator_node();
    // TODO: handle sending the message to the creator node to let it know the provenance
    assert(creator_node == profiler.get_local_processor().address_space());
    profiler.update_footprint(sizeof(info), this);
  }
  
  void ThreadProfiler::record_event_merger(Event result, const Event *preconditions, size_t num_events)
  {
    Profiler &profiler = Profiler::get_profiler();
    if (profiler.no_critical_paths)
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
        record_barrier_usage(preconditions[idx]);
    }
    if (local_proc.exists())
      info.provenance = Processor::get_current_finish_event();
    profiler.update_footprint(sizeof(info) + num_events * sizeof(Event), this); 
  }

  /*static*/ ThreadProfiler& ThreadProfiler::get_thread_profiler(void)
  {
    if (thread_profiler == nullptr) {
      thread_profiler = new ThreadProfiler(Processor::get_executing_processor()); 
      Profiler::get_profiler().record_thread_profiler(thread_profiler);
    }
    return *thread_profiler;
  }

  Profiler::Profiler(void) : local_proc(Processor::NO_PROC),
    output_footprint_threshold(128 << 20/*128MB*/), target_latency(100/*us*/),
    done_event(Realm::UserEvent::create_user_event()), 
    total_memory_footprint(0), return_code(0),
    has_shutdown(false), self_profile(false), no_critical_paths(false)
  {
#ifdef DEBUG_REALM
    for (unsigned idx = 0; idx < LAST_PROF; idx++)
      total_outstanding_requests[idx] = 0;
    total_outstanding_requests[TASK_PROF] = 1; // guard
#else
    total_outstanding_requests.store(1); // guard
#endif
  }

  void Profiler::parse_command_line(int argc, char **argv)
  {
    Realm::CommandLineParser cp;
    cp.add_option_string("-pr:logfile", file_name, true)
      .add_option_int("-pr:footprint", output_footprint_threshold, true)
      .add_option_int("-pr:latency", target_latency, true)
      .add_option_bool("-pr:self", self_profile, true)
      .add_option_bool("-pr:no-critical", no_critical_paths, true)
      .parse_command_line(argc, argv);
    if (file_name.empty()) {
      log_pr.error() << "PRealm must have a file name specified with -pr:logfile";
      std::abort();
    }
  }

  void Profiler::parse_command_line(std::vector<std::string> &cmdline, bool remove_args)
  {
    Realm::CommandLineParser cp;
    cp.add_option_string("-pr:logfile", file_name, !remove_args)
      .add_option_int("-pr:footprint", output_footprint_threshold, !remove_args)
      .add_option_int("-pr:latency", target_latency, !remove_args)
      .add_option_bool("-pr:self", self_profile, !remove_args)
      .add_option_bool("-pr:no-critical", no_critical_paths, !remove_args)
      .parse_command_line(cmdline);
    if (file_name.empty()) {
      log_pr.error() << "PRealm must have a file name specified with -pr:logfile";
      std::abort();
    }
  }

  void Profiler::initialize(void)
  {
    Machine machine = Machine::get_machine();
    total_address_spaces = machine.get_address_space_count();
    Realm::Machine::ProcessorQuery local_procs(machine);
    local_procs.local_address_space();
    assert(!local_proc.exists());
    local_proc = local_procs.first();
    next_backtrace_id = local_proc.address_space();
    assert(local_proc.exists());
    size_t pct = file_name.find_first_of('%', 0);
    if (pct != std::string::npos) {
      std::stringstream ss;
      ss << file_name.substr(0, pct) << local_proc.address_space() << file_name.substr(pct+1);
      file_name = ss.str();
    }
    else if (total_address_spaces > 1) {
      log_pr.error() << "When running in multi-process configurations PRealm requires "
        << "that all filenames contain a '\%' delimiter so each process will be "
        << "writing to an independent file. The specified file name '" << file_name
        << "' does not contain a '\%' delimiter.";
      std::abort();
    }

    // Create the logfile
    f = pr_fopen(file_name, "wb");
    if (!f) {
      log_pr.error() << "PRealm is unable to open file " << file_name << " for writing";
      std::abort();
    }
    // Log the preamble
    log_preamble();
    // Log the machine description
    log_configuration(machine, local_proc);
  }

  void Profiler::log_preamble(void) const
  {
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
       << "id:" << PROC_DESC_ID                     << delim
       << "proc_id:ProcID:"     << sizeof(ProcID)   << delim
       << "kind:ProcKind:"      << sizeof(ProcKind) << delim
       << "uuid_size:uuid_size:" << sizeof(unsigned) << delim
       << "cuda_device_uuid:uuid:"          << sizeof(char)
       << "}" << std::endl;

    ss << "MaxDimDesc {"
       << "id:" << MAX_DIM_DESC_ID                 << delim
       << "max_dim:maxdim:" << sizeof(unsigned)
       << "}" << std::endl;

    ss << "MachineDesc {"
       << "id:" << MACHINE_DESC_ID                  << delim
       << "node_id:unsigned:"   << sizeof(unsigned) << delim
       << "num_nodes:unsigned:" << sizeof(unsigned) << delim
       << "version:unsigned:" << sizeof(unsigned)   << delim
       << "hostname:string:"    << "-1"             << delim
       << "host_id:unsigned long long:" << sizeof(unsigned long long) << delim
       << "process_id:unsigned:" << sizeof(unsigned)
       << "}" << std::endl;

    ss << "CalibrationErr {"
       << "id:" << CALIBRATION_ERR_ID                << delim
       << "calibration_err:long long:" << sizeof(long long)
       << "}" << std::endl;

    ss << "ZeroTime {"
       << "id:" << ZERO_TIME_ID  << delim
       << "zero_time:long long:" << sizeof(long long)
       << "}" << std::endl;

    ss << "MemDesc {" 
       << "id:" << MEM_DESC_ID                               << delim
       << "mem_id:MemID:"                << sizeof(MemID)    << delim
       << "kind:MemKind:"                << sizeof(MemKind)  << delim
       << "capacity:unsigned long long:" << sizeof(unsigned long long)
       << "}" << std::endl;

    ss << "ProcMDesc {"
       << "id:" << PROC_MEM_DESC_ID                          << delim
       << "proc_id:ProcID:"              << sizeof(ProcID)   << delim
       << "mem_id:MemID:"                << sizeof(MemID)    << delim
       << "bandwidth:unsigned:"          << sizeof(unsigned) << delim
       << "latency:unsigned:"            << sizeof(unsigned)
       << "}" << std::endl;

    ss << "PhysicalInstRegionDesc {"
       << "id:" << PHYSICAL_INST_REGION_ID                      << delim
       << "inst_uid:unsigned long long:" << sizeof(Event)     << delim
       << "ispace_id:IDType:"            << sizeof(IDType)      << delim
       << "fspace_id:unsigned:"          << sizeof(unsigned)    << delim
       << "tree_id:unsigned:"            << sizeof(unsigned)
       << "}" << std::endl;

    ss << "PhysicalInstLayoutDesc {"
       << "id:" << PHYSICAL_INST_LAYOUT_ID                  << delim
       << "inst_uid:unsigned long long:" << sizeof(Event) << delim
       << "field_id:unsigned:"        << sizeof(unsigned)   << delim
       << "fspace_id:unsigned:"       << sizeof(unsigned)   << delim
       << "has_align:bool:"           << sizeof(bool)       << delim
       << "eqk:unsigned:"             << sizeof(unsigned)   << delim
       << "align_desc:unsigned:"      << sizeof(unsigned)
       << "}" << std::endl;

    ss << "PhysicalInstDimOrderDesc {"
       << "id:" << PHYSICAL_INST_LAYOUT_DIM_ID              << delim
       << "inst_uid:unsigned long long:" << sizeof(Event) << delim
       << "dim:unsigned:"             << sizeof(unsigned)   << delim
       << "dim_kind:unsigned:"        << sizeof(unsigned)
       << "}" << std::endl;

    ss << "PhysicalInstanceUsage {"
       << "id:" << PHYSICAL_INST_USAGE_ID                   << delim
       << "inst_uid:unsigned long long:" << sizeof(Event) << delim
       << "op_id:UniqueID:"           << sizeof(UniqueID)   << delim
       << "index_id:unsigned:"        << sizeof(unsigned)   << delim
       << "field_id:unsigned:"        << sizeof(unsigned)
       << "}" << std::endl;

    ss << "TaskVariant {"
       << "id:" << TASK_VARIANT_ID                     << delim
       << "task_id:TaskID:"       << sizeof(TaskID)    << delim
       << "variant_id:VariantID:" << sizeof(VariantID) << delim
       << "name:string:"          << "-1"
       << "}" << std::endl;

    ss << "TaskWaitInfo {"
       << "id:" << TASK_WAIT_INFO_ID                       << delim
       << "op_id:UniqueID:"         << sizeof(UniqueID)    << delim
       << "task_id:TaskID:"         << sizeof(TaskID)      << delim
       << "variant_id:VariantID:"   << sizeof(VariantID)   << delim
       << "wait_start:timestamp_t:" << sizeof(timestamp_t) << delim
       << "wait_ready:timestamp_t:" << sizeof(timestamp_t) << delim
       << "wait_end:timestamp_t:"   << sizeof(timestamp_t) << delim
       << "wait_event:unsigned long long:" << sizeof(Event)
       << "}" << std::endl;

    ss << "TaskInfo {"
       << "id:" << TASK_INFO_ID                         << delim
       << "op_id:UniqueID:"      << sizeof(UniqueID)    << delim
       << "task_id:TaskID:"      << sizeof(TaskID)      << delim
       << "variant_id:VariantID:"<< sizeof(VariantID)   << delim
       << "proc_id:ProcID:"      << sizeof(ProcID)      << delim
       << "create:timestamp_t:"  << sizeof(timestamp_t) << delim
       << "ready:timestamp_t:"   << sizeof(timestamp_t) << delim
       << "start:timestamp_t:"   << sizeof(timestamp_t) << delim
       << "stop:timestamp_t:"    << sizeof(timestamp_t) << delim
       << "creator:unsigned long long:" << sizeof(Event) << delim
       << "critical:unsigned long long:" << sizeof(Event) << delim
       << "fevent:unsigned long long:" << sizeof(Event)
       << "}" << std::endl;

    ss << "GPUTaskInfo {"
       << "id:" << GPU_TASK_INFO_ID                       << delim
       << "op_id:UniqueID:"        << sizeof(UniqueID)    << delim
       << "task_id:TaskID:"        << sizeof(TaskID)      << delim
       << "variant_id:VariantID:"   << sizeof(VariantID)    << delim
       << "proc_id:ProcID:"        << sizeof(ProcID)      << delim
       << "create:timestamp_t:"    << sizeof(timestamp_t) << delim
       << "ready:timestamp_t:"     << sizeof(timestamp_t) << delim
       << "start:timestamp_t:"     << sizeof(timestamp_t) << delim
       << "stop:timestamp_t:"      << sizeof(timestamp_t) << delim
       << "gpu_start:timestamp_t:" << sizeof(timestamp_t) << delim
       << "gpu_stop:timestamp_t:"  << sizeof(timestamp_t) << delim
       << "creator:unsigned long long:" << sizeof(Event) << delim
       << "critical:unsigned long long:" << sizeof(Event) << delim
       << "fevent:unsigned long long:" << sizeof(Event)
       << "}" << std::endl;

    ss << "CopyInfo {"
       << "id:" << COPY_INFO_ID                                    << delim
       << "op_id:UniqueID:"          << sizeof(UniqueID)           << delim
       << "size:unsigned long long:" << sizeof(unsigned long long) << delim
       << "create:timestamp_t:"      << sizeof(timestamp_t)        << delim
       << "ready:timestamp_t:"       << sizeof(timestamp_t)        << delim
       << "start:timestamp_t:"       << sizeof(timestamp_t)        << delim
       << "stop:timestamp_t:"        << sizeof(timestamp_t)        << delim
       << "creator:unsigned long long:" << sizeof(Event)      << delim
       << "critical:unsigned long long:" << sizeof(Event)        << delim
       << "fevent:unsigned long long:" << sizeof(Event)          << delim
       << "collective:unsigned:"     << sizeof(unsigned)
       << "}" << std::endl;

    ss << "CopyInstInfo {"
       << "id:" << COPY_INST_INFO_ID                           << delim
       << "src:MemID:"                 << sizeof(MemID)        << delim
       << "dst:MemID:"                 << sizeof(MemID)        << delim
       << "src_fid:unsigned:"          << sizeof(FieldID)      << delim
       << "dst_fid:unsigned:"          << sizeof(FieldID)      << delim
       << "src_inst:unsigned long long:"  << sizeof(Event)   << delim
       << "dst_inst:unsigned long long:"  << sizeof(Event)   << delim
       << "fevent:unsigned long long:" << sizeof(Event)      << delim
       << "num_hops:unsigned:"       << sizeof(unsigned)           << delim
       << "indirect:bool:"             << sizeof(bool)
       << "}" << std::endl;

    ss << "FillInfo {"
       << "id:" << FILL_INFO_ID                        << delim
       << "op_id:UniqueID:"     << sizeof(UniqueID)    << delim
       << "size:unsigned long long:" << sizeof(unsigned long long) << delim
       << "create:timestamp_t:" << sizeof(timestamp_t) << delim
       << "ready:timestamp_t:"  << sizeof(timestamp_t) << delim
       << "start:timestamp_t:"  << sizeof(timestamp_t) << delim
       << "stop:timestamp_t:"   << sizeof(timestamp_t) << delim
       << "creator:unsigned long long:" << sizeof(Event) << delim
       << "critical:unsigned long long:" << sizeof(Event) << delim
       << "fevent:unsigned long long:" << sizeof(Event)
       << "}" << std::endl;

    ss << "FillInstInfo {"
       << "id:" << FILL_INST_INFO_ID                           << delim
       << "dst:MemID:"                    << sizeof(MemID)     << delim
       << "fid:unsigned:"                 << sizeof(FieldID)   << delim
       << "dst_inst:unsigned long long:"  << sizeof(Event)   << delim
       << "fevent:unsigned long long:" << sizeof(Event)
       << "}" << std::endl;

    ss << "InstTimelineInfo {"
       << "id:" << INST_TIMELINE_INFO_ID                << delim
       << "inst_uid:unsigned long long:" << sizeof(Event) << delim
       << "inst_id:InstID:"          << sizeof(InstID)   << delim
       << "mem_id:MemID:"            << sizeof(MemID)    << delim
       << "size:unsigned long long:" << sizeof(unsigned long long) << delim
       << "op_id:UniqueID:"       << sizeof(UniqueID) << delim
       << "create:timestamp_t:"  << sizeof(timestamp_t) << delim
       << "ready:timestamp_t:"  << sizeof(timestamp_t) << delim
       << "destroy:timestamp_t:" << sizeof(timestamp_t) << delim
       << "creator:unsigned long long:" << sizeof(Event)
       << "}" << std::endl;

    ss << "PartitionInfo {"
       << "id:" << PARTITION_INFO_ID                          << delim
       << "op_id:UniqueID:"         << sizeof(UniqueID)       << delim
       << "part_op:DepPartOpKind:"  << sizeof(DepPartOpKind)  << delim
       << "create:timestamp_t:"     << sizeof(timestamp_t)    << delim
       << "ready:timestamp_t:"      << sizeof(timestamp_t)    << delim
       << "start:timestamp_t:"      << sizeof(timestamp_t)    << delim
       << "stop:timestamp_t:"       << sizeof(timestamp_t)    << delim
       << "creator:unsigned long long:" << sizeof(Event)    << delim
       << "critical:unsigned long long:" << sizeof(Event)   << delim
       << "fevent:unsigned long long:" << sizeof(Event)
       << "}" << std::endl;

    ss << "BacktraceDesc {"
       << "id:" << BACKTRACE_DESC_ID                                       << delim
       << "backtrace_id:unsigned long long:" << sizeof(unsigned long long) << delim
       << "backtrace:string:" << "-1"
       << "}" << std::endl;

    ss << "EventWaitInfo {"
       << "id:" << EVENT_WAIT_INFO_ID                         << delim
       << "proc_id:ProcID:" << sizeof(ProcID)                 << delim
       << "fevent:unsigned long long:" << sizeof(Event)     << delim
       << "wait_event:unsigned long long:" << sizeof(Event) << delim
       << "backtrace_id:unsigned long long:" << sizeof(unsigned long long)
       << "}" << std::endl;

    ss << "EventMergerInfo {"
       << "id:" << EVENT_MERGER_INFO_ID                       << delim
       << "result:unsigned long long:" << sizeof(Event)     << delim
       << "fevent:unsigned long long:" << sizeof(Event)     << delim
       << "performed:timestamp_t:" << sizeof(timestamp_t)     << delim
       << "pre0:unsigned long long:" << sizeof(Event)       << delim
       << "pre1:unsigned long long:" << sizeof(Event)       << delim
       << "pre2:unsigned long long:" << sizeof(Event)       << delim
       << "pre3:unsigned long long:" << sizeof(Event)
       << "}" << std::endl;

    ss << "EventTriggerInfo {"
       << "id:" << EVENT_TRIGGER_INFO_ID                      << delim
       << "result:unsigned long long:" << sizeof(Event)     << delim
       << "fevent:unsigned long long:" << sizeof(Event)     << delim
       << "precondition:unsigned long long:" << sizeof(Event) << delim
       << "performed:timestamp_t:" << sizeof(timestamp_t)
       << "}" << std::endl;

    ss << "EventPoisonInfo {"
       << "id:" << EVENT_POISON_INFO_ID                       << delim
       << "result:unsigned long long:" << sizeof(Event)     << delim
       << "fevent:unsigned long long:" << sizeof(Event)     << delim
       << "performed:timestamp_t:" << sizeof(timestamp_t)
       << "}" << std::endl;

    ss << "BarrierArrivalInfo {"
       << "id:" << BARRIER_ARRIVAL_INFO_ID                    << delim
       << "result:unsigned long long:" << sizeof(Event)     << delim
       << "fevent:unsigned long long:" << sizeof(Event)     << delim
       << "precondition:unsigned long long:" << sizeof(Event) << delim
       << "performed:timestamp_t:" << sizeof(timestamp_t)
       << "}" << std::endl;

    ss << "ReservationAcquireInfo {"
       << "id:" << RESERVATION_ACQUIRE_INFO_ID                << delim
       << "result:unsigned long long:" << sizeof(Event)     << delim
       << "fevent:unsigned long long:" << sizeof(Event)     << delim
       << "precondition:unsigned long long:" << sizeof(Event) << delim
       << "performed:timestamp_t:" << sizeof(timestamp_t)     << delim
       << "reservation:unsigned long long:" << sizeof(Reservation)
       << "}" << std::endl;

    ss << "InstanceReadyInfo {"
       << "id:" << INSTANCE_READY_INFO_ID                       << delim
       << "result:unsigned long long:" << sizeof(Event)       << delim
       << "precondition:unsigned long long:" << sizeof(Event) << delim
       << "inst_uid:unsigned long long:" << sizeof(Event)     << delim
       << "performed:timestamp_t:" << sizeof(timestamp_t)
       << "}" << std::endl;

    ss << "CompletionQueueInfo {"
       << "id:" << COMPLETION_QUEUE_INFO_ID                   << delim
       << "result:unsigned long long:" << sizeof(Event)     << delim
       << "fevent:unsigned long long:" << sizeof(Event)     << delim
       << "performed:timestamp_t:" << sizeof(timestamp_t)     << delim
       << "pre0:unsigned long long:" << sizeof(Event)       << delim
       << "pre1:unsigned long long:" << sizeof(Event)       << delim
       << "pre2:unsigned long long:" << sizeof(Event)       << delim
       << "pre3:unsigned long long:" << sizeof(Event)
       << "}" << std::endl;

    ss << "ProfTaskInfo {"
       << "id:" << PROFTASK_INFO_ID                        << delim
       << "proc_id:ProcID:"         << sizeof(ProcID)      << delim
       << "op_id:UniqueID:"         << sizeof(UniqueID)    << delim
       << "start:timestamp_t:"      << sizeof(timestamp_t) << delim
       << "stop:timestamp_t:"       << sizeof(timestamp_t) << delim
       << "creator:unsigned long long:" << sizeof(Event) << delim
       << "fevent:unsigned long long:" << sizeof(Event)  << delim
       << "completion:bool:" << sizeof(bool)
       << "}" << std::endl;

    // An empty line indicates the end of the preamble.
    ss << std::endl;
    std::string preamble = ss.str();

    pr_fwrite(f, preamble.c_str(), strlen(preamble.c_str()));
  }

  void Profiler::log_configuration(Machine &machine, Processor local) const
  {
    // Log the machine configuration
    int ID = MACHINE_DESC_ID;
    pr_fwrite(f, (char*)&ID, sizeof(ID));
    unsigned node_id = local.address_space();
    pr_fwrite(f, (char*)&(node_id), sizeof(node_id));
    pr_fwrite(f, (char*)&(total_address_spaces), sizeof(total_address_spaces));
    unsigned version = LEGION_PROF_VERSION;
    pr_fwrite(f, (char*)&(version), sizeof(version));
    Machine::ProcessInfo process_info;
    machine.get_process_info(local, &process_info);
    pr_fwrite(f, process_info.hostname, strlen(process_info.hostname) + 1);
    pr_fwrite(f, (char*)&(process_info.hostid), sizeof(process_info.hostid));
    pr_fwrite(f, (char*)&(process_info.processid), sizeof(process_info.processid));
    // Log the zero time
    ID = ZERO_TIME_ID;
    pr_fwrite(f, (char*)&ID, sizeof(ID));
    long long zero_time = Realm::Clock::get_zero_time();
    pr_fwrite(f, (char*)&(zero_time), sizeof(zero_time));
    // Log the maximum dimensions
    ID = MAX_DIM_DESC_ID;
    pr_fwrite(f, (char*)&ID, sizeof(ID));
    unsigned max_dim = REALM_MAX_DIM;
    pr_fwrite(f, (char*)&(max_dim), sizeof(max_dim));
  }

  void Profiler::finalize(void)
  {
    // Remove our guard outstanding request that added in the constructor
#ifdef DEBUG_REALM
    decrement_total_outstanding_requests(TASK_PROF); 
#else
    decrement_total_outstanding_requests();
#endif
    if (!done_event.has_triggered())
      done_event.wait();
    // Finalize all the instances
    for (std::vector<ThreadProfiler*>::const_iterator it =
          thread_profilers.begin(); it != thread_profilers.end(); it++)
      (*it)->finalize();
    // Get the calibration error
    const long long calibration_error = Realm::Clock::get_calibration_error();
    int ID = CALIBRATION_ERR_ID;
    pr_fwrite(f, (char*)&ID, sizeof(ID));
    pr_fwrite(f, (char*)&(calibration_error), sizeof(calibration_error));
    // Close the file
    pr_fflush(f, Z_FULL_FLUSH);
    pr_fclose(f);
  }

  void Profiler::defer_shutdown(Event precondition, int code)
  {
    assert(!has_shutdown);
    has_shutdown = true;
    shutdown_precondition = precondition;
    return_code = code;
  }

#ifdef DEBUG_REALM
  void Profiler::increment_total_outstanding_requests(ProfKind kind)
  {
    profiler_lock.wrlock().wait();
    total_outstanding_requests[kind]++;
    profiler_lock.unlock();
  }

  void Profiler::decrement_total_outstanding_requests(ProfKind kind)
  {
    profiler_lock.wrlock().wait();
    assert(total_outstanding_requests[kind] > 0);
    if (--total_outstanding_requests[kind] > 0) {
      profiler_lock.unlock();
      return;
    }
    for (unsigned idx = 0; idx < LAST_PROF; idx++) {
      if (idx == kind)
        continue;
      if (total_outstanding_requests[idx] == 0)
        continue;
      profiler_lock.unlock();
      return;
    }
    profiler_lock.unlock();
    assert(!done_event.has_triggered());
    done_event.trigger();
  }
#else
  void Profiler::increment_total_outstanding_requests(void)
  {
    total_outstanding_requests.fetch_add(1);
  }

  void Profiler::decrement_total_outstanding_requests(void)
  {
    unsigned previous = total_outstanding_requests.fetch_sub(1);
    assert(previous > 0);
    if (previous == 1) {
      assert(!done_event.has_triggered());
      done_event.trigger();
    }
  }
#endif

  void Profiler::release_shutdown(void)
  {
    if (has_shutdown)
      Realm::Runtime::get_runtime().shutdown(shutdown_precondition, return_code);
  }

  void Profiler::record_thread_profiler(ThreadProfiler *profiler)
  {
    profiler_lock.wrlock().wait();
    thread_profilers.push_back(profiler);
    profiler_lock.unlock();
  }

  unsigned long long Profiler::find_backtrace_id(Backtrace &bt)
  {
    const uintptr_t hash = bt.hash();
    profiler_lock.rdlock().wait();
    std::map<uintptr_t,unsigned long long>::const_iterator finder =
      backtrace_ids.find(hash);
    if (finder != backtrace_ids.end()) {
      unsigned long long result = finder->second;
      profiler_lock.unlock();
      return result;
    }
    profiler_lock.unlock();
    // First time seeing this backtrace so capture the symbols
    bt.lookup_symbols();
    std::stringstream ss;
    ss << bt;
    const std::string str = ss.str();
    // Now retake the lock and see if we lost the race
    profiler_lock.wrlock().wait();
    finder = backtrace_ids.find(hash);
    if (finder != backtrace_ids.end()) {
      unsigned long long result = finder->second;
      profiler_lock.unlock();
      return result;
    }
    // Didn't lose the race so generate a new ID for this backtrace
    unsigned long long result = next_backtrace_id;
    next_backtrace_id += total_address_spaces;
    // Save the backtrace into the file
    int ID = BACKTRACE_DESC_ID;
    pr_fwrite(f, (char*)&ID, sizeof(ID));
    pr_fwrite(f, (char*)&result, sizeof(result));
    pr_fwrite(f, str.c_str(), str.size() + 1);
    backtrace_ids[hash] = result;
    profiler_lock.unlock();
    return result;
  }

  void Profiler::update_footprint(size_t diff, ThreadProfiler *profiler)
  {
    size_t footprint = total_memory_footprint.fetch_add(diff) + diff;
    if (footprint > output_footprint_threshold)
    {
      // An important bit of logic here, if we're over the threshold then
      // we want to have a little bit of a feedback loop so the more over
      // the limit we are then the more time we give the profiler to dump
      // out things to the output file. We'll try to make this continuous
      // so there are no discontinuities in performance. If the threshold
      // is zero we'll just choose an arbitrarily large scale factor to 
      // ensure that things work properly.
      double over_scale = output_footprint_threshold == 0 ? double(1 << 20) :
                      double(footprint) / double(output_footprint_threshold);
      // Let's actually make this quadratic so it's not just linear
      if (output_footprint_threshold > 0)
        over_scale *= over_scale;
      // Need a lock to protect the file
      profiler_lock.wrlock().wait();
      diff = profiler->dump_inter(over_scale);
      profiler_lock.unlock();
#ifndef NDEBUG
      footprint = 
#endif
        total_memory_footprint.fetch_sub(diff);
      assert(footprint >= diff); // check for wrap-around
    }
  }

  /*static*/ Profiler& Profiler::get_profiler(void)
  {
    static Profiler singleton;
    return singleton;
  }

  void Runtime::parse_command_line(int argc, char **argv)
  {
    Profiler::get_profiler().parse_command_line(argc, argv);
    Realm::Runtime::parse_command_line(argc, argv);
  }

  void Runtime::parse_command_line(std::vector<std::string> &cmdline, bool remove_args)
  {
    Profiler::get_profiler().parse_command_line(cmdline, remove_args);
    Realm::Runtime::parse_command_line(cmdline, remove_args);
  }

  bool Runtime::configure_from_command_line(int argc, char **argv)
  {
    Profiler::get_profiler().parse_command_line(argc, argv);
    return Realm::Runtime::configure_from_command_line(argc, argv);
  }

  bool Runtime::configure_from_command_line(std::vector<std::string> &cmdline, bool remove_args)
  {
    Profiler::get_profiler().parse_command_line(cmdline, remove_args);
    return Realm::Runtime::configure_from_command_line(cmdline, remove_args);
  }

  void Runtime::start(void)
  {
    Profiler::get_profiler().initialize();
    Realm::Runtime::start();
    // Register our profiling callback function with all the local processors 
    Machine machine = Machine::get_machine();
    Realm::Machine::ProcessorQuery local_procs(machine);
    local_procs.local_address_space();
    std::vector<Realm::Event> registered;
    const Realm::ProfilingRequestSet no_requests;
    const CodeDescriptor callback(Profiler::callback);
    for (Realm::Machine::ProcessorQuery::iterator it = 
          local_procs.begin(); it != local_procs.end(); it++)
    {
      const Realm::Event done = it->register_task(
          Profiler::CALLBACK_TASK_ID, callback, no_requests);
      if (done.exists())
        registered.push_back(done);
    }
    if (!registered.empty())
      Realm::Event::merge_events(registered).wait();
  }

  bool Runtime::init(int *argc, char ***argv)
  {
    // if we get null pointers for argc and argv, use a local version so
    //  any changes from network_init are seen in configure_from_command_line
    int my_argc = 0;
    char **my_argv = 0;
    if(!argc) argc = &my_argc;
    if(!argv) argv = &my_argv;

    if(!Realm::Runtime::network_init(argc, argv)) return false;
    if(!Realm::Runtime::create_configs(*argc, *argv))
      return false;
    if(!configure_from_command_line(*argc, *argv)) return false;
    start();
    return true;
  }

  bool Runtime::register_task(Processor::TaskFuncID task_id, Processor::TaskFuncPtr taskptr)
  {
    // since processors are the same size we can just cast the function pointer
    Realm::Processor::TaskFuncPtr altptr = reinterpret_cast<Realm::Processor::TaskFuncPtr>(taskptr);
    return Realm::Runtime::register_task(task_id, altptr);
  }

  void Runtime::shutdown(Event wait_on, int result_code)
  {
    // Buffer this until we know all the profiling is done
    Profiler::get_profiler().defer_shutdown(wait_on, result_code);
  }

  int Runtime::wait_for_shutdown(void)
  {
    Profiler &profiler = Profiler::get_profiler();
    profiler.finalize();
    // Do a barrier to make sure that everyone is done reporting their profiling
    Realm::Runtime::collective_spawn_by_kind(Processor::Kind::NO_KIND, 
        Processor::TASK_ID_PROCESSOR_NOP, nullptr, 0, true/*one per process*/).wait();
    // If we're the node the buffered the shutdown do that now
    profiler.release_shutdown();
    return Realm::Runtime::wait_for_shutdown();
  }
}
