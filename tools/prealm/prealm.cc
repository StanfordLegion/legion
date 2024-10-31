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

  class Profiler {
  public:
    Profiler(void);
    Profiler(const Profiler &rhs) = delete;
    Profiler& operator=(const Profiler &rhs) = delete;
  public:
    void parse_command_line(int argc, char **argv);
    void parse_command_line(std::vector<std::string> &cmdline, bool remove_args);
    void initialize(void);
    void record_shutdown(Event shutdown_precondition, int return_code);
    void finalize(void);
    void record_thread_profiler(ThreadProfiler *profiler);
    static Profiler& get_profiler(void);
  private:
    void log_preamble(void) const;
    void log_configuration(void) const;
  private:
    Realm::FastReservation profiler_lock;
    std::vector<ThreadProfiler*> thread_profilers;
  private:
    gzFile f;
    AddressSpace local_space;
    std::string file_name;
    size_t max_footprint;
    size_t target_latency; // in us
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

  /*static*/ ThreadProfiler& ThreadProfiler::get_thread_profiler(void)
  {
    if (thread_profiler == nullptr) {
      thread_profiler = new ThreadProfiler(Processor::get_executing_processor()); 
      Profiler::get_profiler().record_thread_profiler(thread_profiler);
    }
    return *thread_profiler;
  }

  Profiler::Profiler(void) : max_footprint(128 << 20/*128MB*/), 
    target_latency(100/*us*/), self_profile(false), no_critical_paths(false)
  {
  }

  void Profiler::parse_command_line(int argc, char **argv)
  {
    Realm::CommandLineParser cp;
    cp.add_option_string("-pr:logfile", file_name, true)
      .add_option_int("-pr:footprint", max_footprint, true)
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
      .add_option_int("-pr:footprint", max_footprint, !remove_args)
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
    Realm::Machine::ProcessorQuery local_procs(machine);
    local_procs.local_address_space();
    Processor local = local_procs.first();
    assert(local.exists());
    size_t pct = file_name.find_first_of('%', 0);
    if (pct != std::string::npos) {
      std::stringstream ss;
      ss << file_name.substr(0, pct) << local.address_space() << file_name.substr(pct+1);
      file_name = ss.str();
    }
    else if (machine.get_address_space_count() > 1) {
      log_pr.error() << "When running in multi-process configurations PRealm requires "
        << "that all filenames contain a '\%' delimiter so each process will be "
        << "writing to an independent file. The specified file name '" << file_name
        << "' contains no '\%' delimiter.";
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
    log_configuration();
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

  void Profiler::record_thread_profiler(ThreadProfiler *profiler)
  {
    profiler_lock.wrlock().wait();
    thread_profilers.push_back(profiler);
    profiler_lock.unlock();
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
    Profiler::get_profiler().record_shutdown(wait_on, result_code);
  }

  int Runtime::wait_for_shutdown(void)
  {
    Profiler::get_profiler().finalize();
    return Realm::Runtime::wait_for_shutdown();
  }
}
