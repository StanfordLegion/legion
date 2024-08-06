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

#ifndef __LEGION_PROFILING_H__
#define __LEGION_PROFILING_H__

#include "realm.h"
#include "legion/legion_types.h"
#include "legion/legion_utilities.h"
#include "legion/garbage_collection.h"
#include "realm/profiling.h"

#ifdef LEGION_USE_CUDA
#include "realm/cuda/cuda_module.h"
#endif

#include <assert.h>
#include <deque>
#include <algorithm>
#include <sstream>

#ifdef DETAILED_LEGION_PROF
#define DETAILED_PROFILER(runtime, call) \
  DetailedProfiler __detailed_profiler(runtime, call)
#else
#define DETAILED_PROFILER(runtime, call) // Nothing
#endif

// This version tracks the compabilitity of the Legion Prof logging
// format. Whenver you make changes to the logging format, increment the number
// stored in legion_profiling_version.h to track the change.
constexpr unsigned LEGION_PROF_VERSION =
#include "legion/legion_profiling_version.h"
;

namespace Legion {
  namespace Internal { 

    // XXX: Make sure these typedefs are consistent with Realm
    typedef long long timestamp_t;
    typedef Realm::Processor::Kind ProcKind;
    typedef Realm::Memory::Kind MemKind;
    typedef ::realm_id_t ProcID;
    typedef ::realm_id_t MemID;
    typedef ::realm_id_t InstID;
    typedef ::realm_id_t IDType;

    class LegionProfSerializer; // forward declaration
    // A small interface class for handling profiling responses
    class ProfilingResponseHandler {
    public:
      // Return true if we should profile this profiling response
      virtual bool handle_profiling_response(
          const Realm::ProfilingResponse &response,
          const void *orig, size_t orig_length) = 0;
    };

    struct ProfilingResponseBase {
    public:
      ProfilingResponseBase(ProfilingResponseHandler *h, UniqueID op);
    public:
      ProfilingResponseHandler *const handler;
      const UniqueID op_id;
      const LgEvent creator;
    };

    /*
     * This class provides an interface for mapping physical instance names
     * back to their unique event names for the profiler
     */
    class InstanceNameClosure : public Collectable {
    public:
      virtual ~InstanceNameClosure(void) { }
    public:
      virtual LgEvent find_instance_name(PhysicalInstance inst) const = 0;
    };

    /*
     * This class provides an instantiation for a fixed number of names
     * Currently we just instantiate it for sizes of 1 and 2 for 
     * fills and normal copies respectively
     */
    template<size_t ENTRIES>
    class SmallNameClosure : public InstanceNameClosure {
    public:
      SmallNameClosure(void);
      SmallNameClosure(const SmallNameClosure &rhs) = delete;
      virtual ~SmallNameClosure(void) { }
    public:
      SmallNameClosure& operator=(const SmallNameClosure &rhs) = delete;
    public:
      void record_instance_name(PhysicalInstance inst, LgEvent name);
      virtual LgEvent find_instance_name(PhysicalInstance inst) const;
    private:
      static_assert(ENTRIES > 0);
      // Optimize for the common case of there being one or two entries
      PhysicalInstance instances[ENTRIES];
      LgEvent names[ENTRIES];
    };

    class LegionProfMarker {
    public:
      LegionProfMarker(const char* name);
      ~LegionProfMarker();
      void mark_stop();
    private:
      const char* name;
      bool stopped;
      Processor proc;
      timestamp_t start, stop;
    };

    class LegionProfDesc {
    public:
      struct ProcDesc {
      public:
        ProcID proc_id;
        ProcKind kind;
#ifdef LEGION_USE_CUDA
        Realm::Cuda::Uuid cuda_device_uuid;
#endif
      };
      struct MemDesc {
      public:
        MemID mem_id;
        MemKind kind;
        unsigned long long capacity;
      };
      struct ProcMemDesc {
      public:
        ProcID proc_id;
        MemID mem_id;
        unsigned bandwidth;
        unsigned latency;
      };
      struct TaskKind {
      public:
        TaskID task_id;
        const char *name;
        bool overwrite;
      };
      struct TaskVariant {
      public:
        TaskID task_id;
        VariantID variant_id;
        const char *name;
      };
      struct MapperName {
        MapperID mapper_id;
        ProcID mapper_proc;
        const char *name;
      };
      struct MapperCallDesc {
      public:
        unsigned kind;
        const char *name;
      };
      struct RuntimeCallDesc {
      public:
        unsigned kind;
        const char *name;
      };
      struct MetaDesc {
      public:
        unsigned kind;
        bool message;
        bool ordered_vc;
        const char *name;
      };
      struct OpDesc {
      public:
        unsigned kind;
        const char *name;
      };
      struct MaxDimDesc {
	unsigned max_dim;
      };
      struct RuntimeConfig {
        bool debug;
        bool spy;
        bool gc;
        bool inorder;
        bool safe_mapper;
        bool safe_runtime;
        bool safe_ctrlrepl;
        bool part_checks;
        bool bounds_checks;
        bool resilient;
      };
      struct MachineDesc {
        unsigned node_id;
        unsigned num_nodes;
        unsigned version;
        Machine::ProcessInfo process_info;
      };
      struct CalibrationErr {
      public:
        long long calibration_err;
      };
      struct ZeroTime {
      public:
        long long zero_time;
      };
      struct Provenance {
      public:
        ProvenanceID pid;
        const char *provenance;
        size_t size;
      };
      struct Backtrace {
      public:
        unsigned long long id;
        const char *backtrace;
      };
    };

    class LegionProfInstance {
    public:
      struct OperationInstance {
      public:
        UniqueID op_id;
        UniqueID parent_id;
        unsigned kind;
        ProvenanceID provenance;
      };
      struct MultiTask {
      public:
        UniqueID op_id;
        TaskID task_id;
      };
      struct SliceOwner {
      public:
        UniqueID parent_id;
        UniqueID op_id;
      };
      struct WaitInfo {
      public:
        timestamp_t wait_start, wait_ready, wait_end;
        LgEvent wait_event;
      };
      struct TaskInfo {
      public:
        UniqueID op_id;
        TaskID task_id;
        VariantID variant_id;
        ProcID proc_id;
        timestamp_t create, ready, start, stop;
        std::deque<WaitInfo> wait_intervals;
        LgEvent creator;
        LgEvent critical;
        LgEvent finish_event;
      };
      struct GPUTaskInfo {
      public:
        UniqueID op_id;
        TaskID task_id;
        VariantID variant_id;
        ProcID proc_id;
        timestamp_t create, ready, start, stop;
        timestamp_t gpu_start, gpu_stop;
        std::deque<WaitInfo> wait_intervals;
        LgEvent creator;
        LgEvent critical;
        LgEvent finish_event;
      };
      struct IndexSpacePointDesc {
      public:
	IDType unique_id;
	unsigned dim;
        long long points[LEGION_MAX_DIM];
      };
      struct IndexSpaceEmptyDesc {
      public:
	IDType unique_id;
      };
      struct IndexSpaceRectDesc {
      public:
	IDType unique_id;
        long long rect_lo[LEGION_MAX_DIM];
        long long rect_hi[LEGION_MAX_DIM];
	unsigned dim;
      };
      struct FieldDesc {
      public:
	UniqueID unique_id;
	unsigned field_id;
	unsigned long long size;
	const char *name;
      };
      struct FieldSpaceDesc {
      public:
	UniqueID unique_id;
	const char *name;
      };
      struct IndexPartDesc {
      public:
        UniqueID unique_id;
        const char *name;
      };
      struct IndexSpaceDesc {
      public:
	UniqueID unique_id;
	const char *name;
      };
      struct IndexPartitionDesc {
      public:
        IDType parent_id;
        IDType unique_id;
        bool disjoint;
        LegionColor point;
      };
      struct IndexSubSpaceDesc {
      public:
	IDType parent_id;
	IDType unique_id;
      };
      struct LogicalRegionDesc {
      public:
	IDType ispace_id;
	unsigned fspace_id;
	unsigned tree_id;
	const char *name;
      };
      struct PhysicalInstRegionDesc {
      public:
        LgEvent inst_uid;
	IDType ispace_id;
	unsigned fspace_id;
	unsigned tree_id;
      };
      struct PhysicalInstLayoutDesc {
      public:
        LgEvent inst_uid;
	unsigned field_id;
	unsigned fspace_id;
        EqualityKind eqk;
        bool has_align;
        unsigned alignment;
      };
      struct PhysicalInstDimOrderDesc {
      public:
        LgEvent inst_uid;
        unsigned dim;
        DimensionKind k;
      };
      struct PhysicalInstanceUsage {
      public:
        LgEvent inst_uid;
        UniqueID op_id;
        unsigned index;
        unsigned field;
      };
      struct IndexSpaceSizeDesc {
      public:
        UniqueID id;
        unsigned long long dense_size, sparse_size;
        bool is_sparse;
      };
      struct MetaInfo {
      public:
        UniqueID op_id;
        unsigned lg_id;
        ProcID proc_id;
        timestamp_t create, ready, start, stop;
        std::deque<WaitInfo> wait_intervals;
        LgEvent creator;
        LgEvent critical;
        LgEvent finish_event;
      };
      struct MessageInfo : public MetaInfo {
      public:
        // Spawn is recorded on the creator node while
        // create is recorded on the destination node
        // We use that to detect network congestion and
        // cases of timing skew
        timestamp_t spawn;
      };
      struct CopyInstInfo {
      public:
        MemID src, dst;
        FieldID src_fid, dst_fid;
        LgEvent src_inst_uid, dst_inst_uid;
        unsigned num_hops;
        bool indirect;
      };
      struct CopyInfo {
      public:
        UniqueID op_id;
        unsigned long long size;
        timestamp_t create, ready, start, stop;
        LgEvent fevent;
        LgEvent creator;
        LgEvent critical;
        CollectiveKind collective;
        std::vector<CopyInstInfo> inst_infos;
      };
      struct FillInstInfo {
      public:
        MemID dst;
        FieldID fid;
        LgEvent dst_inst_uid;
      };
      struct FillInfo {
      public:
        UniqueID op_id;
        unsigned long long size;
        timestamp_t create, ready, start, stop;
        LgEvent fevent;
        LgEvent creator;
        LgEvent critical;
        CollectiveKind collective;
        std::vector<FillInstInfo> inst_infos;
      };
      struct InstTimelineInfo {
      public:
        LgEvent inst_uid;
        InstID inst_id;
        MemID mem_id;
        unsigned long long size;
        UniqueID op_id; // creator op for the instance
        timestamp_t create, ready, destroy;
        LgEvent creator;
      };
      struct PartitionInfo {
      public:
        UniqueID op_id;
        DepPartOpKind part_op;
        unsigned long long create, ready, start, stop;
        LgEvent fevent;
        LgEvent creator;
        LgEvent critical;
      };
      struct MapperCallInfo {
      public:
        MapperID mapper;
        ProcID mapper_proc;
        MappingCallKind kind;
        UniqueID op_id;
        timestamp_t start, stop;
        ProcID proc_id;
        LgEvent finish_event;
      };
      struct RuntimeCallInfo {
      public:
        RuntimeCallKind kind;
        timestamp_t start, stop;
        ProcID proc_id;
        LgEvent finish_event;
      };
      struct ApplicationCallInfo {
        ProvenanceID pid;
        timestamp_t start, stop;
        ProcID proc_id;
        LgEvent finish_event;
      }; 
      struct EventWaitInfo {
      public:
        ProcID proc_id;
        LgEvent fevent;
        LgEvent event;
        unsigned long long backtrace_id;
      };
      struct ProfTaskInfo {
      public:
        ProcID proc_id;
        UniqueID op_id;
        timestamp_t start, stop;
        LgEvent creator;
        LgEvent finish_event;
      };
      struct EventMergerInfo {
      public:
        LgEvent result;
        LgEvent fevent;
        timestamp_t performed;
        std::vector<LgEvent> preconditions; 
      };
      struct EventTriggerInfo {
      public:
        LgEvent result;
        LgEvent fevent;
        LgEvent precondition;
        timestamp_t performed;
      };
      struct EventPoisonInfo {
      public:
        LgEvent result;
        LgEvent fevent;
        timestamp_t performed;
      };
      struct BarrierArrivalInfo {
      public:
        LgEvent result;
        LgEvent fevent;
        LgEvent precondition;
        timestamp_t performed;
      };
      struct ReservationAcquireInfo {
      public:
        LgEvent result;
        LgEvent fevent;
        LgEvent precondition;
        timestamp_t performed;
        Reservation reservation;
      };
      struct ProfilingInfo : public ProfilingResponseBase {
      public:
        ProfilingInfo(ProfilingResponseHandler *h, UniqueID uid);
      public:
        size_t id; 
        union {
          size_t id2;
          InstanceNameClosure *closure;
          long long spawn_time;
        } extra;
        LgEvent creator;
        LgEvent critical;
      };
    public:
      LegionProfInstance(LegionProfiler *owner);
      LegionProfInstance(const LegionProfInstance &rhs);
      ~LegionProfInstance(void);
    public:
      LegionProfInstance& operator=(const LegionProfInstance &rhs);
    public: 
      void register_operation(Operation *op);
      void register_multi_task(Operation *op, TaskID kind);
      void register_slice_owner(UniqueID pid, UniqueID id);
      void register_index_space_rect(IndexSpaceRectDesc
				     &ispace_rect_desc);
      void register_index_space_point(IndexSpacePointDesc
				      &ispace_point_desc);
      template <int DIM, typename T>
      void record_index_space_point(IDType handle, const Point<DIM, T> &point);
      template<int DIM, typename T>
      void record_index_space_rect(IDType handle, const Rect<DIM, T> &rect);
      void register_empty_index_space(IDType handle);
      void register_field(UniqueID unique_id, unsigned field_id,
			  size_t size, const char* name);
      void register_field_space(UniqueID unique_id, const char* name);
      void register_index_part(UniqueID unique_id, const char* name);
      void register_index_space(UniqueID unique_id, const char* name);
      void register_index_subspace(IDType parent_id, IDType unique_id,
				   const DomainPoint &point);
      void register_index_partition(IDType parent_id, IDType unique_id,
				    bool disjoint, LegionColor point);
      void register_logical_region(IDType index_space,
				   unsigned field_space, unsigned tree_id,
				   const char* name);
      void register_physical_instance_region(LgEvent inst_uid,
					     LogicalRegion handle);
      void register_physical_instance_layout(LgEvent unique_event,
                                             FieldSpace fs,
                                             const LayoutConstraintSet &lc);
      void register_physical_instance_field(LgEvent inst_uid,
                                            unsigned field_id,
                                            unsigned fspace,
                                            unsigned align,
                                            bool has_align,
                                            EqualityKind eqk);
      void register_physical_instance_dim_order(LgEvent inst_uid,
                                                unsigned dim,
                                                DimensionKind k);
      void register_physical_instance_use(LgEvent inst_uid,
                                          UniqueID op_id,
                                          unsigned index,
                                          const std::vector<FieldID> &fields);
      void register_index_space_size(UniqueID id,
                                     unsigned long long
                                     dense_size,
                                     unsigned long long
                                     sparse_size,
                                     bool is_sparse);
    public:
      void record_event_merger(LgEvent result, 
          const LgEvent *preconditions, size_t count);
      void record_event_trigger(LgEvent result, LgEvent precondition);
      void record_event_poison(LgEvent result);
      void record_barrier_arrival(LgEvent result, LgEvent precondition);
      void record_reservation_acquire(Reservation r, LgEvent result,
          LgEvent precondition);
    public:
      void process_task(const ProfilingInfo *info,
            const Realm::ProfilingResponse &response,
            const Realm::ProfilingMeasurements::OperationProcessorUsage &usage);
      void process_meta(const ProfilingInfo *info,
            const Realm::ProfilingResponse &response,
            const Realm::ProfilingMeasurements::OperationProcessorUsage &usage);
      void process_message(const ProfilingInfo *info,
            const Realm::ProfilingResponse &response,
            const Realm::ProfilingMeasurements::OperationProcessorUsage &usage);
      void process_copy(const ProfilingInfo *info,
            const Realm::ProfilingResponse &response,
            const Realm::ProfilingMeasurements::OperationMemoryUsage &usage);
      void process_fill(const ProfilingInfo *info,
            const Realm::ProfilingResponse &response,
            const Realm::ProfilingMeasurements::OperationMemoryUsage &usage);
      void process_inst_timeline(const ProfilingInfo *info,
            const Realm::ProfilingResponse &response,
            const Realm::ProfilingMeasurements::InstanceMemoryUsage &usage,
            const Realm::ProfilingMeasurements::InstanceTimeline &timeline);
      void process_partition(const ProfilingInfo *info,
                             const Realm::ProfilingResponse &response);
      void process_implicit(UniqueID op_id, TaskID tid, Processor proc,
          long long start, long long stop, std::deque<WaitInfo> &waits,
          LgEvent finish_event);
      void process_mem_desc(const Memory &m);
      void process_proc_desc(const Processor &p);
      void process_proc_mem_aff_desc(const Memory &m);
      void process_proc_mem_aff_desc(const Processor &p);
    public:
      void record_mapper_call(MapperID mapper, Processor mapper_proc,
       MappingCallKind kind, UniqueID uid, timestamp_t start, timestamp_t stop);
      void record_runtime_call(RuntimeCallKind kind, timestamp_t start,
                               timestamp_t stop);
      void record_application_range(ProvenanceID pid,
                                    timestamp_t start, timestamp_t stop);
      void record_event_wait(LgEvent event, Realm::Backtrace &bt);
    public:
      void record_proftask(Processor p, UniqueID op_id, timestamp_t start,
          timestamp_t stop, LgEvent creator, LgEvent finish_event);
    public:
      void dump_state(LegionProfSerializer *serializer);
      size_t dump_inter(LegionProfSerializer *serializer, const double over);
    private:
      LegionProfiler *const owner;
      std::deque<OperationInstance> operation_instances;
      std::deque<MultiTask>         multi_tasks;
      std::deque<SliceOwner>        slice_owners;
    private:
      std::deque<TaskInfo> task_infos;
      std::deque<GPUTaskInfo> gpu_task_infos;
      std::deque<IndexSpaceRectDesc> ispace_rect_desc;
      std::deque<IndexSpacePointDesc> ispace_point_desc;
      std::deque<IndexSpaceEmptyDesc> ispace_empty_desc;
      std::deque<FieldDesc> field_desc;
      std::deque<FieldSpaceDesc> field_space_desc;
      std::deque<IndexPartDesc> index_part_desc;
      std::deque<IndexSpaceDesc> index_space_desc;
      std::deque<IndexSubSpaceDesc> index_subspace_desc;
      std::deque<IndexPartitionDesc> index_partition_desc;
      std::deque<LogicalRegionDesc> lr_desc;
      std::deque<PhysicalInstRegionDesc> phy_inst_rdesc;
      std::deque<PhysicalInstLayoutDesc> phy_inst_layout_rdesc;
      std::deque<PhysicalInstDimOrderDesc> phy_inst_dim_order_rdesc;
      std::deque<PhysicalInstanceUsage> phy_inst_usage;
      std::deque<IndexSpaceSizeDesc> index_space_size_desc;
      std::deque<MetaInfo> meta_infos;
      std::deque<MessageInfo> message_infos;
      std::deque<CopyInfo> copy_infos;
      std::deque<FillInfo> fill_infos;
      std::deque<InstTimelineInfo> inst_timeline_infos;
      std::deque<PartitionInfo> partition_infos;
      std::deque<MapperCallInfo> mapper_call_infos;
      std::deque<RuntimeCallInfo> runtime_call_infos;
      std::deque<ApplicationCallInfo> application_call_infos;
      std::deque<EventWaitInfo> event_wait_infos;
      std::deque<EventMergerInfo> event_merger_infos;
      std::deque<EventTriggerInfo> event_trigger_infos;
      std::deque<EventPoisonInfo> event_poison_infos;
      std::deque<BarrierArrivalInfo> barrier_arrival_infos;
      std::deque<ReservationAcquireInfo> reservation_acquire_infos;
      // keep track of MemIDs/ProcIDs to avoid duplicate entries
      std::vector<MemID> mem_ids;
      std::vector<ProcID> proc_ids;
    private:
      std::deque<ProfTaskInfo> prof_task_infos;
    };

    class LegionProfiler : public ProfilingResponseHandler {
    public:
      enum ProfilingKind {
        LEGION_PROF_TASK,
        LEGION_PROF_META,
        LEGION_PROF_MESSAGE,
        LEGION_PROF_COPY,
        LEGION_PROF_FILL,
        LEGION_PROF_INST,
        LEGION_PROF_PARTITION,
        LEGION_PROF_LAST,
      };
      struct ProfilingInfo : public LegionProfInstance::ProfilingInfo {
      public:
        ProfilingInfo(LegionProfiler *p, ProfilingKind k, UniqueID uid)
          : LegionProfInstance::ProfilingInfo(p, uid), kind(k) { }
        ProfilingInfo(LegionProfiler *p, ProfilingKind k, Operation *op);
      public:
        ProfilingKind kind;
      };
    public:
      // Statically known information passed through the constructor
      // so that it can be deduplicated
      LegionProfiler(Processor target_proc, const Machine &machine,
                     Runtime *rt, unsigned num_meta_tasks,
                     const char *const *const meta_task_descriptions,
                     unsigned num_message_kinds,
                     const char *const *const message_decriptions,
                     unsigned num_operation_kinds,
                     const char *const *const operation_kind_descriptions,
                     const char *serializer_type,
                     const char *prof_logname,
                     const size_t total_runtime_instances,
                     const size_t footprint_threshold,
                     const size_t target_latency,
                     const size_t minimum_call_threshold,
                     const bool slow_config_ok,
                     const bool self_profile,
                     const bool no_critical);
      LegionProfiler(const LegionProfiler &rhs) = delete;
      virtual ~LegionProfiler(void);
    public:
      LegionProfiler& operator=(const LegionProfiler &rhs) = delete;
    public:
      void register_task_kind(TaskID task_id, const char *name, bool overwrite);
      void register_task_variant(TaskID task_id,
                                 VariantID variant_id, 
                                 const char *variant_name);
      unsigned long long find_backtrace_id(Realm::Backtrace &bt);
    public:
      void record_memory(Memory m);
      void record_processor(Processor p);
      void record_affinities(std::vector<Memory> &memories_to_log);
    public:
      void add_task_request(Realm::ProfilingRequestSet &requests, TaskID tid, 
                            VariantID vid, UniqueID task_uid, Processor p, 
                            LgEvent critical);
      void add_meta_request(Realm::ProfilingRequestSet &requests,
                            LgTaskID tid, Operation *op, LgEvent critical);
      void add_copy_request(Realm::ProfilingRequestSet &requests, 
                            InstanceNameClosure *closure, Operation *op,
                            LgEvent critical, unsigned count = 1, 
                            CollectiveKind collective = COLLECTIVE_NONE);
      void add_fill_request(Realm::ProfilingRequestSet &requests,
                            InstanceNameClosure *closure, Operation *op,
                            LgEvent critical,
                            CollectiveKind collective = COLLECTIVE_NONE);
      void add_inst_request(Realm::ProfilingRequestSet &requests,
                            Operation *op, LgEvent unique_event);
      void handle_failed_instance_allocation(void);
      void add_partition_request(Realm::ProfilingRequestSet &requests,
                                 Operation *op, DepPartOpKind part_op,
                                 LgEvent critical);
      // Adding a message profiling request is a static method
      // because we might not have a profiler on the local node
      static void add_message_request(Realm::ProfilingRequestSet &requests,
                                      MessageKind kind,Processor remote_target,
                                      LgEvent critical);
    public:
      // Alternate versions of the one above with op ids
      void add_task_request(Realm::ProfilingRequestSet &requests, TaskID tid,
                            VariantID vid, UniqueID uid, LgEvent critical);
      void add_gpu_task_request(Realm::ProfilingRequestSet &requests,
                            TaskID tid, VariantID vid, UniqueID uid,
                            LgEvent critical);
      void add_meta_request(Realm::ProfilingRequestSet &requests,
                            LgTaskID tid, UniqueID uid, LgEvent critical);
      void add_copy_request(Realm::ProfilingRequestSet &requests,
                            InstanceNameClosure *closure, UniqueID uid,
                            LgEvent critical, unsigned count = 1,
                            CollectiveKind collective = COLLECTIVE_NONE);
      void add_fill_request(Realm::ProfilingRequestSet &requests,
                            InstanceNameClosure *closure, UniqueID uid,
                            LgEvent critical,
                            CollectiveKind collective = COLLECTIVE_NONE);
      void add_inst_request(Realm::ProfilingRequestSet &requests,
                            UniqueID uid, LgEvent unique_event);
      void add_partition_request(Realm::ProfilingRequestSet &requests,
                                 UniqueID uid, DepPartOpKind part_op,
                                 LgEvent critical);
    public:
      // Process low-level runtime profiling results
      virtual bool handle_profiling_response(
          const Realm::ProfilingResponse &response,
          const void *orig, size_t orig_length);
    public:
      // Dump all the results
      void finalize(void);
    public:
      void record_mapper_name(MapperID mapper, Processor p, const char *name);
      void record_mapper_call_kinds(const char *const *const mapper_call_names,
                                    unsigned int num_mapper_call_kinds);
      
      void record_runtime_call_kinds(const char *const *const runtime_calls,
                                     unsigned int num_runtime_call_kinds);
      void record_provenance(ProvenanceID pid, 
                             const char *provenance, size_t size);
    public:
#ifdef DEBUG_LEGION
      void increment_total_outstanding_requests(ProfilingKind kind,
                                                unsigned cnt = 1);
      void decrement_total_outstanding_requests(ProfilingKind kind,
                                                unsigned cnt = 1);
#else
      void increment_total_outstanding_requests(unsigned cnt = 1);
      void decrement_total_outstanding_requests(unsigned cnt = 1);
#endif
    public:
      void update_footprint(size_t diff, LegionProfInstance *inst);
    public:
      void issue_default_mapper_warning(Operation *op, const char *call_name);
    public:
      LegionProfInstance* find_or_create_profiling_instance(void);
    public:
      Runtime *const runtime;
      // Event to trigger once the profiling is actually done
      const RtUserEvent done_event;
      // Minimum duration of mapper and runtime calls for logging in ns
      const long long minimum_call_threshold;
      // Size in bytes of the footprint before we start dumping
      const size_t output_footprint_threshold;
      // The goal size in microseconds of the output tasks
      const long long output_target_latency;
      // Target processor on which to launch jobs
      const Processor target_proc;
      // Whether we are self-profiling
      const bool self_profile;
      // Whether we are profiling for critical path
      const bool no_critical_paths;
    private:
      LegionProfSerializer* serializer;
      mutable LocalLock profiler_lock;
      std::vector<LegionProfInstance*> instances;
      std::map<Processor,LegionProfInstance*> processor_instances;
      std::map<uintptr_t,unsigned long long> backtrace_ids;
      unsigned long long next_backtrace_id;
      std::vector<Memory> recorded_memories;
      std::vector<Processor> recorded_processors;
#ifdef DEBUG_LEGION
      unsigned total_outstanding_requests[LEGION_PROF_LAST];
#else
      std::atomic<unsigned> total_outstanding_requests;
#endif
    private:
      // For knowing when we need to start dumping early
      std::atomic<size_t> total_memory_footprint;
    private:
      // Issue the default mapper warning
      std::atomic<bool> need_default_mapper_warning; 
    };

    class DetailedProfiler {
    public:
      DetailedProfiler(Runtime *runtime, RuntimeCallKind call);
      DetailedProfiler(const DetailedProfiler &rhs);
      ~DetailedProfiler(void);
    public:
      DetailedProfiler& operator=(const DetailedProfiler &rhs);
    private:
      LegionProfiler *const profiler;
      const RuntimeCallKind call_kind;
      timestamp_t start_time;
    };

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline void LegionProfInstance::record_index_space_point(IDType handle,
                                                      const Point<DIM,T> &point)
    //--------------------------------------------------------------------------
    {
      IndexSpacePointDesc ispace_point_desc;
      ispace_point_desc.unique_id = handle;
      ispace_point_desc.dim = (unsigned)DIM;
#define DIMFUNC(D2) \
      ispace_point_desc.points[D2-1] = (D2<=DIM) ? (long long)point[D2-1] : 0;
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      register_index_space_point(ispace_point_desc);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline void LegionProfInstance::record_index_space_rect(IDType handle,
                                                        const Rect<DIM,T> &rect)
    //--------------------------------------------------------------------------
    {
      IndexSpaceRectDesc ispace_rect_desc;
      ispace_rect_desc.unique_id = handle;
      ispace_rect_desc.dim = DIM;
#define DIMFUNC(D2) \
      ispace_rect_desc.rect_lo[D2-1] = (D2<=DIM) ? (long long)rect.lo[D2-1]:0; \
      ispace_rect_desc.rect_hi[D2-1] = (D2<=DIM) ? (long long)rect.hi[D2-1]:0;
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      register_index_space_rect(ispace_rect_desc);
    }

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_PROFILING_H__

