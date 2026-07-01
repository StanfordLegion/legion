/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_PROFILER_H__
#define __LEGION_PROFILER_H__

#include "legion/kernel/garbage_collection.h"
#include "legion/managers/message.h"
#include "legion/tools/types.h"
#ifdef LEGION_USE_CUDA
#include "realm/cuda/cuda_module.h"
#endif

// This version tracks the compabilitity of the Legion Prof logging
// format. Whenver you make changes to the logging format, increment the number
// stored in legion_profiling_version.h to track the change.
constexpr unsigned LEGION_PROF_VERSION =
#include "legion/tools/profiler_version.h"
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

    // This class helps us profile barriers by allowing us to
    // find the latest barrier arrival to trigger
    struct ArrivalInfo {
    public:
      ArrivalInfo(void);
      ArrivalInfo(const ArrivalInfo& rhs);
      ArrivalInfo(LgEvent precondition);
      ArrivalInfo(
          timestamp_t arrival, timestamp_t trigger, LgEvent precondition,
          LgEvent fevent);
    public:
      timestamp_t arrival_time;
      std::atomic<timestamp_t> trigger_time;
      LgEvent arrival_precondition;
      LgEvent fevent;
    };

    // This reduction is used for profiling the arrival of barriers
    class BarrierArrivalReduction {
    public:
      typedef ArrivalInfo RHS;
      typedef ArrivalInfo LHS;
      static const ArrivalInfo identity;
      static constexpr ReductionOpID REDOP = LEGION_MAX_APPLICATION_REDOP_ID;
      static constexpr timestamp_t SENTINEL =
          std::numeric_limits<timestamp_t>::max();

      template<bool EXCLUSIVE>
      static void apply(LHS& lhs, const RHS& rhs);
      template<bool EXCLUSIVE>
      static void fold(RHS& rhs1, const RHS& rhs2);
    };

    class LegionProfSerializer;  // forward declaration
    // A small interface class for handling profiling responses
    class ProfilingResponseHandler {
    public:
      // Return true if we should profile this profiling response
      virtual bool handle_profiling_response(
          const Realm::ProfilingResponse& response, const void* orig,
          size_t orig_length, LgEvent& fevent, bool& failed_alloc) = 0;
    };

    struct ProfilingResponseBase {
    public:
      ProfilingResponseBase(
          ProfilingResponseHandler* h, UniqueID op, bool complete = true)
        : handler(h), op_id(op), completion(complete)
      { }
    public:
      ProfilingResponseHandler* const handler;
      const UniqueID op_id;
      // Whether this profiling response happens at the completion
      // of the operation or at the initiation
      const bool completion;
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
      virtual DistributedID find_instance_subspace(
          PhysicalInstance inst) const = 0;
      virtual DistributedID find_copy_expression(void) const = 0;
      virtual ReductionOpID find_redop(void) const = 0;
    };

    /*
     * This class provides an instantiation for a fixed number of names
     * Currently we just instantiate it for sizes of 1 and 2 for
     * fills and normal copies respectively
     */
    template<size_t ENTRIES>
    class SmallNameClosure : public InstanceNameClosure {
    public:
      SmallNameClosure(IndexSpaceExpression* expr, ReductionOpID redop = 0);
      SmallNameClosure(const SmallNameClosure& rhs) = delete;
      virtual ~SmallNameClosure(void) { }
    public:
      SmallNameClosure& operator=(const SmallNameClosure& rhs) = delete;
    public:
      void record_instance_name(
          PhysicalInstance inst, LgEvent name, DistributedID subspace);
      virtual LgEvent find_instance_name(PhysicalInstance inst) const override;
      virtual DistributedID find_instance_subspace(
          PhysicalInstance inst) const override;
      virtual DistributedID find_copy_expression(void) const override
      {
        return copy_expr;
      }
      virtual ReductionOpID find_redop(void) const override { return redop; }
    private:
      static_assert(ENTRIES > 0);
      const DistributedID copy_expr;
      const ReductionOpID redop;
      // Optimize for the common case of there being one or two entries
      PhysicalInstance instances[ENTRIES];
      LgEvent names[ENTRIES];
      DistributedID subspaces[ENTRIES];
    };

    /*
     * An instantiation of the instance name closure for a variable number of
     * entries
     */
    class LargeNameClosure : public InstanceNameClosure {
    public:
      LargeNameClosure(
          IndexSpaceExpression* expr, FieldID fid, size_t expected = 0,
          ReductionOpID redop = 0);
      LargeNameClosure(const LargeNameClosure& rhs) = delete;
      virtual ~LargeNameClosure(void) { }
    public:
      LargeNameClosure& operator=(const LargeNameClosure& rhs) = delete;
    public:
      void record_instance_name(
          PhysicalInstance inst, LgEvent name, DistributedID subspace);
      virtual LgEvent find_instance_name(PhysicalInstance inst) const override;
      virtual DistributedID find_instance_subspace(
          PhysicalInstance inst) const override;
      virtual DistributedID find_copy_expression(void) const override
      {
        return copy_expr;
      }
      virtual ReductionOpID find_redop(void) const override { return redop; }
    public:
      const DistributedID copy_expr;
      const ReductionOpID redop;
      const FieldID fid;
      struct Entry {
        PhysicalInstance instance;
        LgEvent name;
        DistributedID subspace;
      };
      std::vector<Entry> entries;
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
        DistributedID unique_id;
        unsigned dim;
        long long points[LEGION_MAX_DIM];
      };
      struct IndexSpaceEmptyDesc {
      public:
        DistributedID unique_id;
      };
      struct IndexSpaceRectDesc {
      public:
        DistributedID unique_id;
        long long rect_lo[LEGION_MAX_DIM];
        long long rect_hi[LEGION_MAX_DIM];
        unsigned dim;
      };
      struct FieldDesc {
      public:
        UniqueID unique_id;
        unsigned field_id;
        unsigned long long size;
        const char* name;
      };
      struct FieldSpaceDesc {
      public:
        UniqueID unique_id;
        const char* name;
      };
      struct IndexPartDesc {
      public:
        UniqueID unique_id;
        const char* name;
      };
      struct IndexSpaceDesc {
      public:
        UniqueID unique_id;
        const char* name;
      };
      struct IndexPartitionDesc {
      public:
        DistributedID parent_id;
        DistributedID unique_id;
        bool disjoint;
        LegionColor point;
      };
      struct IndexSubSpaceDesc {
      public:
        DistributedID parent_id;
        DistributedID unique_id;
      };
      struct LogicalRegionDesc {
      public:
        DistributedID ispace_id;
        DistributedID fspace_id;
        unsigned tree_id;
        const char* name;
      };
      struct PhysicalInstRegionDesc {
      public:
        LgEvent inst_uid;
        DistributedID ispace_id;
        DistributedID fspace_id;
        unsigned tree_id;
      };
      struct PhysicalInstLayoutDesc {
      public:
        LgEvent inst_uid;
        unsigned field_id;
        DistributedID fspace_id;
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
      struct PhysicalInstanceSpaces {
      public:
        LgEvent inst_uid;
        DistributedID union_space;
        DistributedID piece_space;
      };
      struct PhysicalInstanceUsage {
      public:
        LgEvent inst_uid;
        LgEvent fevent;
        UniqueID op_id;
        timestamp_t start;
        timestamp_t stop;
        DistributedID index_expr;
        PrivilegeMode mode;
        ReductionOpID redop;
        FieldID field;
        int32_t index;
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
        DistributedID src_expr, dst_expr;
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
        ReductionOpID redop;
        DistributedID copy_expr;
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
        DistributedID fill_expr;
        std::vector<FillInstInfo> inst_infos;
      };
      struct InstTimelineInfo {
      public:
        LgEvent inst_uid;
        InstID inst_id;
        MemID mem_id;
        unsigned long long size;
        UniqueID op_id;  // creator op for the instance
        timestamp_t create, ready, destroy;
        LgEvent creator;
      };
      struct PartInstInfo {
      public:
        MemID src;
        FieldID fid;
        LgEvent src_inst_uid;
        DistributedID src_expr;
      };
      struct PartitionInfo {
      public:
        UniqueID op_id;
        DepPartOpKind part_op;
        unsigned long long create, ready, start, stop;
        LgEvent fevent;
        LgEvent creator;
        LgEvent critical;
        std::vector<PartInstInfo> inst_infos;
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
      struct AsyncEffectInfo {
        LgEvent external;
        LgEvent fevent;
        timestamp_t created;
        timestamp_t triggered;
        ProvenanceID pid;
      };
      struct EventWaitInfo {
      public:
        ProcID proc_id;
        LgEvent fevent;
        LgEvent event;
        ProvenanceID provenance_id;
        unsigned long long backtrace_id;
      };
      struct ProfTaskInfo {
      public:
        ProcID proc_id;
        UniqueID op_id;
        timestamp_t start, stop;
        LgEvent creator;
        LgEvent finish_event;
        bool completion;
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
      struct InstanceReadyInfo {
      public:
        LgEvent result;
        LgEvent precondition;
        LgEvent unique;
        timestamp_t performed;
      };
      struct InstanceRedistrictInfo {
      public:
        LgEvent result;
        LgEvent precondition;
        LgEvent previous;
        LgEvent next;
        timestamp_t performed;
      };
      struct MakeValidInfo {
      public:
        LgEvent result;
        LgEvent fevent;
        DistributedID space;
        timestamp_t created;
        timestamp_t triggered;
      };
      struct FetchMetadataInfo {
      public:
        LgEvent result;
        LgEvent fevent;
        LgEvent inst_uid;
        timestamp_t created;
        timestamp_t triggered;
      };
      struct CompletionQueueInfo {
      public:
        LgEvent result;
        LgEvent fevent;
        timestamp_t performed;
        std::vector<LgEvent> preconditions;
      };
      struct ProfilingInfo : public ProfilingResponseBase {
      public:
        ProfilingInfo(ProfilingResponseHandler* h, UniqueID uid);
      public:
        size_t id;
        union {
          size_t id2;
          InstanceNameClosure* closure;
          long long spawn_time;
        } extra;
        LgEvent creator;
        LgEvent critical;
      };
    public:
      LegionProfInstance(
          LegionProfiler* owner, Processor local, LgEvent external);
    private:
      LegionProfInstance(
          LegionProfiler* owner, Processor local, LgEvent external,
          long long start);
    public:
      LegionProfInstance(const LegionProfInstance& rhs) = delete;
      ~LegionProfInstance(void);
    public:
      LegionProfInstance& operator=(const LegionProfInstance& rhs) = delete;
    public:
      inline bool is_external_thread(void) const
      {
        return external_fevent.exists();
      }
    public:
      LegionProfInstance* dump(void);
      void register_operation(Operation* op);
      void register_multi_task(Operation* op, TaskID kind);
      void register_slice_owner(UniqueID pid, UniqueID id);
      void register_index_space_rect(IndexSpaceRectDesc& ispace_rect_desc);
      void register_index_space_point(IndexSpacePointDesc& ispace_point_desc);
      template<int DIM, typename T>
      void record_index_space_point(
          DistributedID handle, const Point<DIM, T>& point);
      template<int DIM, typename T>
      void record_index_space_rect(
          DistributedID handle, const Rect<DIM, T>& rect);
      void register_empty_index_space(DistributedID handle);
      void register_field(
          UniqueID unique_id, unsigned field_id, size_t size, const char* name);
      void register_field_space(UniqueID unique_id, const char* name);
      void register_index_part(UniqueID unique_id, const char* name);
      void register_index_space(UniqueID unique_id, const char* name);
      void register_index_subspace(
          DistributedID parent_id, DistributedID unique_id,
          const DomainPoint& point);
      void register_index_partition(
          DistributedID parent_id, DistributedID unique_id, bool disjoint,
          LegionColor point);
      void register_logical_region(
          DistributedID index_space, DistributedID field_space,
          unsigned tree_id, const char* name);
      void register_physical_instance_region(
          LgEvent inst_uid, LogicalRegion handle);
      void register_physical_instance_layout(
          LgEvent unique_event, FieldSpace fs, const LayoutConstraintSet& lc);
      void register_physical_instance_field(
          LgEvent inst_uid, unsigned field_id, DistributedID fspace,
          unsigned align, bool has_align, EqualityKind eqk);
      void register_physical_instance_dim_order(
          LgEvent inst_uid, unsigned dim, DimensionKind k);
      void register_physical_instance_spaces(
          LgEvent inst_uid, DistributedID union_space,
          DistributedID piece_space);
      void register_physical_instance_use(
          LgEvent inst_uid, UniqueID op_id, DistributedID index_expr,
          FieldID field, PrivilegeMode mode, ReductionOpID redop,
          timestamp_t start_time, timestamp_t stop_time, int index = -1);
      void register_index_space_size(
          UniqueID id, unsigned long long dense_size,
          unsigned long long sparse_size, bool is_sparse);
    public:
      void record_event_merger(
          LgEvent result, const LgEvent* preconditions, size_t count);
      void record_event_trigger(LgEvent result, LgEvent precondition);
      void record_event_poison(LgEvent result);
      void record_barrier_arrival(LgEvent barrier, LgEvent precondition);
      void record_barrier_use(LgEvent barrier, UniqueID uid);
      void record_reservation_acquire(
          Reservation r, LgEvent result, LgEvent precondition);
      void record_instance_ready(
          LgEvent result, LgEvent unique_event,
          LgEvent precondition = LgEvent::NO_LG_EVENT);
      void record_instance_redistrict(
          LgEvent& result, LgEvent prev_unique_event, LgEvent next_unique_event,
          LgEvent precondition = LgEvent::NO_LG_EVENT);
      void record_completion_queue_event(
          LgEvent result, LgEvent fevent, timestamp_t timestamp,
          const LgEvent* preconditions, size_t count);
      void record_make_valid(LgEvent event, DistributedID space = 0);
      void record_fetch_metadata(LgEvent event, LgEvent inst_uid);
    public:
      void process_task(
          const ProfilingInfo* info, const Realm::ProfilingResponse& response,
          const Realm::ProfilingMeasurements::OperationProcessorUsage& usage);
      void process_meta(
          const ProfilingInfo* info, const Realm::ProfilingResponse& response,
          const Realm::ProfilingMeasurements::OperationProcessorUsage& usage);
      void process_message(
          const ProfilingInfo* info, const Realm::ProfilingResponse& response,
          const Realm::ProfilingMeasurements::OperationProcessorUsage& usage);
      void process_copy(
          const ProfilingInfo* info, const Realm::ProfilingResponse& response,
          const Realm::ProfilingMeasurements::OperationMemoryUsage& usage);
      void process_fill(
          const ProfilingInfo* info, const Realm::ProfilingResponse& response,
          const Realm::ProfilingMeasurements::OperationMemoryUsage& usage);
      void process_inst_timeline(
          const ProfilingInfo* info, const Realm::ProfilingResponse& response,
          const Realm::ProfilingMeasurements::InstanceMemoryUsage& usage,
          const Realm::ProfilingMeasurements::InstanceTimeline& timeline);
      void process_partition(
          const ProfilingInfo* info, const Realm::ProfilingResponse& response);
      void process_arrival(
          const ProfilingInfo* info,
          const Realm::ProfilingMeasurements::OperationTimeline& timeline);
      void process_async_effect(
          const ProfilingInfo* info,
          const Realm::ProfilingMeasurements::OperationTimeline& timeline);
      void process_make_valid(
          const ProfilingInfo* info,
          const Realm::ProfilingMeasurements::OperationTimeline& timeline);
      void process_fetch_metadata(
          const ProfilingInfo* info,
          const Realm::ProfilingMeasurements::OperationTimeline& timeline);
      void process_implicit(
          UniqueID op_id, TaskID tid, long long start, long long stop,
          std::deque<WaitInfo>& waits, LgEvent finish_event);
      void process_mem_desc(const Memory& m);
      void process_proc_desc(const Processor& p);
      void process_proc_mem_aff_desc(const Memory& m);
      void process_proc_mem_aff_desc(const Processor& p);
      void process_event_trigger(Deserializer& derez);
      void process_event_poison(Deserializer& derez);
    public:
      void record_mapper_call(
          MapperID mapper, Processor mapper_proc, MappingCallKind kind,
          UniqueID uid, timestamp_t start, timestamp_t stop);
      void record_runtime_call(
          RuntimeCallKind kind, timestamp_t start, timestamp_t stop);
      void record_application_range(
          ProvenanceID pid, timestamp_t start, timestamp_t stop);
      void record_async_effect(ApEvent effect, const char* provenance);
      void record_event_wait(
          LgEvent event, ProvenanceID pid, Realm::Backtrace& bt);
      void begin_external_wait(LgEvent event);
      void end_external_wait(LgEvent event);
    public:
      void record_proftask(
          Processor p, UniqueID op_id, timestamp_t start, timestamp_t stop,
          LgEvent creator, LgEvent finish_event, bool creator_complete);
    public:
      void dump_state(LegionProfSerializer* serializer);
      bool dump_inter(LegionProfSerializer* serializer, const long long t_stop);
    public:
      // If this profiler instance is associated with an external thread
      // then it will have an external fevent that will eventually be
      // rendered in Legion Prof as an implicit top-level task
      const LgEvent external_fevent;
      const Processor local_proc;  // might be fake
      const long long external_start;
    public:
      // Use this for creating lock-free linked lists of profiler instances
      std::atomic<LegionProfInstance*> next = nullptr;
      size_t footprint = 0;
    private:
      LegionProfiler* const owner;
      std::deque<OperationInstance> operation_instances;
      std::deque<MultiTask> multi_tasks;
      std::deque<SliceOwner> slice_owners;
    private:
      std::deque<TaskInfo> task_infos;
      std::deque<TaskInfo> implicit_infos;
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
      std::deque<PhysicalInstanceSpaces> phy_inst_spaces;
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
      std::deque<AsyncEffectInfo> async_effect_infos;
      std::deque<EventWaitInfo> event_wait_infos;
      std::deque<EventMergerInfo> event_merger_infos;
      std::deque<EventTriggerInfo> event_trigger_infos;
      std::deque<EventPoisonInfo> event_poison_infos;
      std::deque<BarrierArrivalInfo> barrier_arrival_infos;
      std::deque<ReservationAcquireInfo> reservation_acquire_infos;
      std::deque<InstanceReadyInfo> instance_ready_infos;
      std::deque<InstanceRedistrictInfo> instance_redistrict_infos;
      std::deque<CompletionQueueInfo> completion_queue_infos;
      std::deque<MakeValidInfo> make_valid_infos;
      std::deque<FetchMetadataInfo> fetch_metadata_infos;
      std::vector<WaitInfo> external_wait_infos;
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
        LEGION_PROF_ARRIVAL,
        LEGION_PROF_BARRIER,
        LEGION_PROF_TRIGGER,
        LEGION_PROF_LAST,
      };
      struct ProfilingInfo : public LegionProfInstance::ProfilingInfo {
      public:
        ProfilingInfo(LegionProfiler* p, ProfilingKind k, UniqueID uid)
          : LegionProfInstance::ProfilingInfo(p, uid), kind(k)
        { }
        ProfilingInfo(LegionProfiler* p, ProfilingKind k, Operation* op);
      public:
        ProfilingKind kind;
      };
      struct ProfilerDumpArgs : public LgTaskArgs<ProfilerDumpArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_PROFILER_DUMP_TASK_ID;
      public:
        ProfilerDumpArgs(void) = default;
        ProfilerDumpArgs(LegionProfiler* prof, RtUserEvent done)
          : LgTaskArgs<ProfilerDumpArgs>(true, true), profiler(prof),
            dump_event(done)
        { }
      public:
        void execute(void) const;
      public:
        LegionProfiler* profiler;
        RtUserEvent dump_event;
      };
      struct MapperCallDesc {
      public:
        unsigned kind;
        const char* name;
      };
      struct RuntimeCallDesc {
      public:
        unsigned kind;
        const char* name;
      };
      struct MetaDesc {
      public:
        unsigned kind;
        bool message;
        bool ordered_vc;
        const char* name;
      };
      struct OpDesc {
      public:
        unsigned kind;
        const char* name;
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
        std::string name;
        bool overwrite;
      };
      struct TaskVariant {
      public:
        TaskID task_id;
        VariantID variant_id;
        std::string name;
      };
      struct MapperName {
        MapperID mapper_id;
        ProcID mapper_proc;
        std::string name;
      };
      struct Provenance {
      public:
        ProvenanceID pid;
        std::string provenance;
      };
      struct Backtrace {
      public:
        unsigned long long id;
        std::string backtrace;
      };
      struct PendingBacktrace : public Backtrace {
        uintptr_t hash;
        PendingBacktrace* next;
      };
    public:
      // Statically known information passed through the constructor
      // so that it can be deduplicated
      LegionProfiler(
          Processor target_proc, const Machine& machine,
          unsigned num_meta_tasks,
          const char* const * const meta_task_descriptions,
          unsigned num_message_kinds,
          const char* const * const message_decriptions,
          unsigned num_operation_kinds,
          const char* const * const operation_kind_descriptions,
          const char* serializer_type, const char* prof_logname,
          const size_t total_runtime_instances,
          const size_t footprint_threshold, const size_t target_latency,
          const size_t minimum_call_threshold, const bool slow_config_ok,
          const bool self_profile, const bool no_critical,
          const bool all_arrivals);
      LegionProfiler(const LegionProfiler& rhs) = delete;
      virtual ~LegionProfiler(void);
    public:
      LegionProfiler& operator=(const LegionProfiler& rhs) = delete;
    public:
      void register_task_kind(TaskID task_id, const char* name, bool overwrite);
      void register_task_variant(
          TaskID task_id, VariantID variant_id, const char* variant_name);
      unsigned long long find_backtrace_id(Realm::Backtrace& bt);
      void drain_pending_backtraces(bool track_diff);
    public:
      void record_memory(Memory m);
      void record_processor(Processor p);
      void record_affinities(std::vector<Memory>& memories_to_log);
      // We make a custom processor rendering any implicit top-level tasks
      // because we need to render them separately from other tasks since
      // they might be running concurrently on different threads
      // (Note also that the same implicit top-level task doesn't even
      // need to stay on the same external thread for its whole lifespan.)
      ProcID get_implicit_processor(void);
      TaskID get_external_implicit_task(void);
    public:
      void add_task_request(
          Realm::ProfilingRequestSet& requests, TaskID tid, VariantID vid,
          UniqueID task_uid, Processor p, LgEvent critical);
      void add_meta_request(
          Realm::ProfilingRequestSet& requests, LgTaskID tid, Operation* op,
          LgEvent critical);
      void add_copy_request(
          Realm::ProfilingRequestSet& requests, InstanceNameClosure* closure,
          Operation* op, LgEvent critical, unsigned count = 1,
          CollectiveKind collective = COLLECTIVE_NONE);
      void add_fill_request(
          Realm::ProfilingRequestSet& requests, InstanceNameClosure* closure,
          Operation* op, LgEvent critical,
          CollectiveKind collective = COLLECTIVE_NONE);
      void add_inst_request(
          Realm::ProfilingRequestSet& requests, Operation* op,
          LgEvent unique_event);
      void add_partition_request(
          Realm::ProfilingRequestSet& requests, Operation* op,
          DepPartOpKind part_op, LgEvent critical,
          LargeNameClosure* closure = nullptr);
      // Adding a message profiling request is a static method
      // because we might not have a profiler on the local node
      static void add_message_request(
          Realm::ProfilingRequestSet& requests, MessageKind kind,
          Processor remote_target, LgEvent critical);
    public:
      // Alternate versions of the one above with op ids
      void add_task_request(
          Realm::ProfilingRequestSet& requests, TaskID tid, VariantID vid,
          UniqueID uid, LgEvent critical);
      void add_gpu_task_request(
          Realm::ProfilingRequestSet& requests, TaskID tid, VariantID vid,
          UniqueID uid, LgEvent critical);
      void add_meta_request(
          Realm::ProfilingRequestSet& requests, LgTaskID tid, UniqueID uid,
          LgEvent critical);
      void add_copy_request(
          Realm::ProfilingRequestSet& requests, InstanceNameClosure* closure,
          UniqueID uid, LgEvent critical, unsigned count = 1,
          CollectiveKind collective = COLLECTIVE_NONE);
      void add_fill_request(
          Realm::ProfilingRequestSet& requests, InstanceNameClosure* closure,
          UniqueID uid, LgEvent critical,
          CollectiveKind collective = COLLECTIVE_NONE);
      void add_inst_request(
          Realm::ProfilingRequestSet& requests, UniqueID uid,
          LgEvent unique_event);
      void add_partition_request(
          Realm::ProfilingRequestSet& requests, UniqueID uid,
          DepPartOpKind part_op, LgEvent critical,
          LargeNameClosure* closure = nullptr);
    public:
      void profile_barrier_arrival(
          Realm::Barrier bar, size_t count, LgEvent precondition,
          Realm::Event protected_precondition);
      void profile_barrier_trigger(Realm::Barrier bar, UniqueID uid);
      bool update_previous_recorded_barrier(
          Realm::Barrier bar, Realm::Barrier& previous);
    public:
      // Process low-level runtime profiling results
      virtual bool handle_profiling_response(
          const Realm::ProfilingResponse& response, const void* orig,
          size_t orig_length, LgEvent& fevent, bool& failed_alloc) override;
    public:
      // Dump all the results
      void finalize(void);
      void dump_instances(RtUserEvent dump_event);
    public:
      void record_mapper_name(MapperID mapper, Processor p, const char* name);
      void record_mapper_call_kinds(
          const char* const * const mapper_call_names,
          unsigned int num_mapper_call_kinds);
      void record_runtime_call_kinds(
          const char* const * const runtime_calls,
          unsigned int num_runtime_call_kinds);
      void record_provenance(
          ProvenanceID pid, const char* provenance, size_t size);
    public:
      void increment_outstanding_message_request(void);
      LgEvent find_message_fevent(LgEvent original_fevent, bool remove);
    protected:
      void increment_total_outstanding_requests(
          ProfilingKind kind, unsigned cnt = 1);
      void decrement_total_outstanding_requests(
          ProfilingKind kind, unsigned cnt = 1);
    public:
      enum EffectKind {
        ASYNC_EFFECT,
        MAKE_VALID_EFFECT,
        FETCH_METADATA_EFFECT,
      };
      void measure_event_trigger(
          LgEvent effect, EffectKind kind, LgEvent fevent, size_t extra_data,
          const char* prov = nullptr);
    public:
      void update_footprint(size_t diff, LegionProfInstance* inst);
    protected:
      void update_footprint(size_t diff);
    public:
      void issue_default_mapper_warning(Operation* op, const char* call_name);
    public:
      void instantiate_profiling_instance(void);
    public:
      // Event to trigger once the profiling is actually done
      const Realm::UserEvent done_event;
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
      // Whether we are recording all the critical barrier arrivals
      // or we are doing a reduction with the barrier to compute it
      const bool all_critical_arrivals;
    private:
      LegionProfSerializer* serializer;
      mutable LocalLock profiler_lock;
      std::atomic<LegionProfInstance*> instances = nullptr;
      std::map<uintptr_t, unsigned long long> backtrace_ids;
      std::vector<Memory> recorded_memories;
      std::vector<Processor> recorded_processors;
      std::map<LgEvent, LgEvent> message_fevents;
      std::map<std::pair<unsigned, unsigned>, unsigned> recorded_barriers;
#ifdef LEGION_DEBUG
      unsigned total_outstanding_requests[LEGION_PROF_LAST];
#else
      std::atomic<unsigned> total_outstanding_requests;
#endif
    private:
      // For knowing when we need to start dumping early
      std::atomic<size_t> total_memory_footprint;
    private:
      std::atomic<ProcID> implicit_top_level_task_proc;
      std::optional<TaskID> external_implicit_task;
    private:
      // Issue the default mapper warning
      std::atomic<bool> need_default_mapper_warning;
    private:
      std::deque<ProcDesc> processor_descriptions;
      std::deque<MemDesc> memory_descriptions;
      std::deque<ProcMemDesc> procmem_affinities;
      std::deque<TaskKind> task_kinds;
      std::deque<TaskVariant> task_variants;
      std::deque<Backtrace> backtraces;
      std::deque<MapperName> mapper_names;
      std::deque<Provenance> provenances;
      std::atomic<PendingBacktrace*> pending_backtraces = nullptr;
    private:
      // Special sentinel value for indicating that we have
      // launched a dump profile task for things which we
      // need to dump immediately, but is not a valid instances
      static constexpr uintptr_t DUMP_NOW = 0xbeebbeeb;
      std::atomic<LegionProfInstance*> dump_list = nullptr;
      RtEvent last_dump_task;
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/tools/profiler.inl"

#endif  // __LEGION_PROFILER_H__
