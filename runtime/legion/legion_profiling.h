/* Copyright 2022 Stanford University, NVIDIA Corporation
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
#include "realm/profiling.h"

#include <assert.h>
#include <deque>
#include <algorithm>
#include <sstream>

#define LEGION_PROF_SELF_PROFILE

#ifdef DETAILED_LEGION_PROF
#define DETAILED_PROFILER(runtime, call) \
  DetailedProfiler __detailed_profiler(runtime, call)
#else
#define DETAILED_PROFILER(runtime, call) // Nothing
#endif

namespace Legion {
  namespace Internal { 

    // XXX: Make sure these typedefs are consistent with Realm
    typedef ::realm_barrier_timestamp_t timestamp_t;
    typedef Realm::Processor::Kind ProcKind;
    typedef Realm::Memory::Kind MemKind;
    typedef ::realm_id_t ProcID;
    typedef ::realm_id_t MemID;
    typedef ::realm_id_t InstID;
    typedef ::realm_id_t IDType;

    class LegionProfSerializer; // forward declaration

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
    };

    class LegionProfInstance {
    public:
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
      struct OperationInstance {
      public:
        UniqueID op_id;
        UniqueID parent_id;
        unsigned kind;
        const char *provenance;
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
      };
      struct TaskInfo {
      public:
        UniqueID op_id;
        TaskID task_id;
        VariantID variant_id;
        ProcID proc_id;
        timestamp_t create, ready, start, stop;
        std::deque<WaitInfo> wait_intervals;
#ifdef LEGION_PROF_PROVENANCE
        LgEvent provenance, finish_event;
#endif
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
#ifdef LEGION_PROF_PROVENANCE
        LgEvent provenance, finish_event;
#endif
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
	UniqueID op_id;
	IDType inst_id;
	IDType ispace_id;
	unsigned fspace_id;
	unsigned tree_id;
      };
      struct PhysicalInstLayoutDesc {
      public:
	UniqueID op_id;
	IDType inst_id;
	unsigned field_id;
	unsigned fspace_id;
        EqualityKind eqk;
        bool has_align;
        unsigned alignment;
      };
      struct PhysicalInstDimOrderDesc {
      public:
        UniqueID op_id;
        IDType inst_id;
        unsigned dim;
        DimensionKind k;
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
#ifdef LEGION_PROF_PROVENANCE
        LgEvent provenance, finish_event;
#endif
      };
      struct CopyInstInfo {
      public:
        InstID src_inst_id;
        InstID dst_inst_id;
        unsigned num_fields;
        unsigned num_hops;
        unsigned request_type;
      };
      struct CopyInfo {
      public:
        UniqueID op_id;
        MemID src, dst;
        unsigned long long size;
        timestamp_t create, ready, start, stop;
        LgEvent fevent;
        unsigned num_requests;
        std::deque<CopyInstInfo> requests;
#ifdef LEGION_PROF_PROVENANCE
        LgEvent provenance;
#endif
      };
      struct FillInfo {
      public:
        UniqueID op_id;
        MemID dst;
        timestamp_t create, ready, start, stop;
#ifdef LEGION_PROF_PROVENANCE
        LgEvent provenance;
#endif
      };
      struct InstCreateInfo {
      public:
        UniqueID op_id;
        InstID inst_id;
        timestamp_t create; // time of HLR creation request
#ifdef LEGION_PROF_PROVENANCE
        LgEvent provenance;
#endif
      };
      struct InstUsageInfo {
      public:
        UniqueID op_id;
        InstID inst_id;
        MemID mem_id;
        unsigned long long size;
      };
      struct InstTimelineInfo {
      public:
        UniqueID op_id;
        InstID inst_id;
        timestamp_t create, ready, destroy;
      };
      struct PartitionInfo {
      public:
        UniqueID op_id;
        DepPartOpKind part_op;
        unsigned long long create, ready, start, stop;
#ifdef LEGION_PROF_PROVENANCE
        LgEvent provenance;
#endif
      };
      struct MapperCallInfo {
      public:
        MappingCallKind kind;
        UniqueID op_id;
        timestamp_t start, stop;
        ProcID proc_id;
      };
      struct RuntimeCallInfo {
      public:
        RuntimeCallKind kind;
        timestamp_t start, stop;
        ProcID proc_id;
      };
      struct ProcDesc {
      public:
        ProcID proc_id;
        ProcKind kind;
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
#ifdef LEGION_PROF_SELF_PROFILE
      struct ProfTaskInfo {
      public:
        ProcID proc_id;
        UniqueID op_id;
        timestamp_t start, stop;
      };
#endif
      struct ProfilingInfo : public ProfilingResponseBase {
      public:
        ProfilingInfo(ProfilingResponseHandler *h) 
          : ProfilingResponseBase(h) 
#ifdef LEGION_PROF_PROVENANCE
          , provenance(!Processor::get_executing_processor.exists() ?
          LgEvent::NO_LG_EVENT : LgEvent(Processor::get_current_finish_event()))
#endif
        { }
      public:
        size_t id, id2;
        UniqueID op_id;
#ifdef LEGION_PROF_PROVENANCE
        LgEvent provenance;
#endif
      };
    public:
      LegionProfInstance(LegionProfiler *owner);
      LegionProfInstance(const LegionProfInstance &rhs);
      ~LegionProfInstance(void);
    public:
      LegionProfInstance& operator=(const LegionProfInstance &rhs);
    public:
      void register_task_kind(TaskID task_id, const char *name, bool overwrite);
      void register_task_variant(TaskID task_id,
                                 VariantID variant_id, 
                                 const char *variant_name);
      void register_operation(Operation *op);
      void register_multi_task(Operation *op, TaskID kind);
      void register_slice_owner(UniqueID pid, UniqueID id);
      void register_index_space_rect(IndexSpaceRectDesc
				     &ispace_rect_desc);
      void register_index_space_point(IndexSpacePointDesc
				      &ispace_point_desc);
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
      void register_physical_instance_region(UniqueID op_id, IDType inst_id,
					     LogicalRegion handle);
      void register_physical_instance_field(UniqueID op_id, IDType inst_id, 
                                            unsigned field_id,
                                            unsigned fspace,
                                            unsigned align,
                                            bool has_align,
                                            EqualityKind eqk);
      void register_physical_instance_dim_order(UniqueID op_id,
                                                IDType inst_id,
                                                unsigned dim,
                                                DimensionKind k);
      void register_index_space_size(UniqueID id,
                                     unsigned long long
                                     dense_size,
                                     unsigned long long
                                     sparse_size,
                                     bool is_sparse);
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
      void process_inst_create(UniqueID op_id, PhysicalInstance inst,
                               timestamp_t create);
      void process_inst_usage(const ProfilingInfo *info,
            const Realm::ProfilingResponse &response,
            const Realm::ProfilingMeasurements::InstanceMemoryUsage &usage);
      void process_inst_timeline(const ProfilingInfo *info,
            const Realm::ProfilingResponse &response,
            const Realm::ProfilingMeasurements::InstanceTimeline &timeline);
      void process_partition(const ProfilingInfo *info,
                             const Realm::ProfilingResponse &response);
      void process_mem_desc(const Memory &m);
      void process_proc_desc(const Processor &p);
      void process_proc_mem_aff_desc(const Memory &m);
      void process_proc_mem_aff_desc(const Processor &p);

    public:
      void record_mapper_call(Processor proc, MappingCallKind kind, 
                              UniqueID uid, timestamp_t start,
                              timestamp_t stop);
      void record_runtime_call(Processor proc, RuntimeCallKind kind,
                               timestamp_t start, timestamp_t stop);
#ifdef LEGION_PROF_SELF_PROFILE
    public:
      void record_proftask(Processor p, UniqueID op_id, timestamp_t start,
                           timestamp_t stop);
#endif
    public:
      void dump_state(LegionProfSerializer *serializer);
      size_t dump_inter(LegionProfSerializer *serializer, const double over);
    private:
      LegionProfiler *const owner;
      std::deque<TaskKind>          task_kinds;
      std::deque<TaskVariant>       task_variants;
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
      std::deque<IndexSpaceSizeDesc> index_space_size_desc;
      std::deque<MetaInfo> meta_infos;
      std::deque<CopyInfo> copy_infos;
      std::deque<FillInfo> fill_infos;
      std::deque<InstCreateInfo> inst_create_infos;
      std::deque<InstUsageInfo> inst_usage_infos;
      std::deque<InstTimelineInfo> inst_timeline_infos;
      std::deque<PartitionInfo> partition_infos;
      std::deque<MapperCallInfo> mapper_call_infos;
      std::deque<RuntimeCallInfo> runtime_call_infos;
      std::deque<MemDesc> mem_desc_infos;
      std::deque<ProcDesc> proc_desc_infos;
      std::deque<ProcMemDesc> proc_mem_aff_desc_infos;
      // keep track of MemIDs/ProcIDs to avoid duplicate entries
      std::vector<MemID> mem_ids;
      std::vector<ProcID> proc_ids;
#ifdef LEGION_PROF_SELF_PROFILE
    private:
      std::deque<ProfTaskInfo> prof_task_infos;
#endif
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
        ProfilingInfo(LegionProfiler *p, ProfilingKind k)
          : LegionProfInstance::ProfilingInfo(p), kind(k) { }
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
                     const bool slow_config_ok);
      LegionProfiler(const LegionProfiler &rhs);
      virtual ~LegionProfiler(void);
    public:
      LegionProfiler& operator=(const LegionProfiler &rhs);
    public:
      // Dynamically created things must be registered at runtime
      // Tasks
      void register_task_kind(TaskID task_id, const char *task_name, 
                              bool overwrite);
      void register_task_variant(TaskID task_id,
                                 VariantID var_id,
                                 const char *variant_name);
      // Operations
      void register_operation(Operation *op);
      void register_multi_task(Operation *op, TaskID task_id);
      void register_slice_owner(UniqueID pid, UniqueID id);
    public:
      void add_task_request(Realm::ProfilingRequestSet &requests, TaskID tid, 
                            VariantID vid, UniqueID task_uid, Processor p);
      void add_meta_request(Realm::ProfilingRequestSet &requests,
                            LgTaskID tid, Operation *op);
      void add_copy_request(Realm::ProfilingRequestSet &requests, 
                            Operation *op, unsigned count = 1);
      void add_fill_request(Realm::ProfilingRequestSet &requests,
                            Operation *op);
      void add_inst_request(Realm::ProfilingRequestSet &requests,
                            Operation *op);
      void handle_failed_instance_allocation(void);
      void add_partition_request(Realm::ProfilingRequestSet &requests,
                                 Operation *op, DepPartOpKind part_op);
      // Adding a message profiling request is a static method
      // because we might not have a profiler on the local node
      static void add_message_request(Realm::ProfilingRequestSet &requests,
                                      MessageKind kind,Processor remote_target);
    public:
      // Alternate versions of the one above with op ids
      void add_task_request(Realm::ProfilingRequestSet &requests, 
                            TaskID tid, VariantID vid, UniqueID uid);
      void add_gpu_task_request(Realm::ProfilingRequestSet &requests,
                            TaskID tid, VariantID vid, UniqueID uid);
      void add_meta_request(Realm::ProfilingRequestSet &requests,
                            LgTaskID tid, UniqueID uid);
      void add_copy_request(Realm::ProfilingRequestSet &requests, 
                            UniqueID uid, unsigned count = 1);
      void add_fill_request(Realm::ProfilingRequestSet &requests,
                            UniqueID uid);
      void add_inst_request(Realm::ProfilingRequestSet &requests,
                            UniqueID uid);
      void add_partition_request(Realm::ProfilingRequestSet &requests,
                                 UniqueID uid, DepPartOpKind part_op);
    public:
      // Process low-level runtime profiling results
      virtual void handle_profiling_response(const ProfilingResponseBase *base,
                                      const Realm::ProfilingResponse &response,
                                      const void *orig, size_t orig_length);
    public:
      // Dump all the results
      void finalize(void);
    public:
      void record_instance_creation(PhysicalInstance inst, Memory memory,
                                    UniqueID op_id, timestamp_t create);
      void record_empty_index_space(IDType handle);
      void record_field_space(UniqueID uid, const char* name);
      void record_field(UniqueID unique_id,
			unsigned field_id,
			size_t size,
			const char* name);
      void record_index_space(UniqueID uid, const char* name);
      void record_index_subspace(IDType parent_id, IDType unique_id,
				 const DomainPoint &point);
      void record_logical_region(IDType index_space, unsigned field_space,
				 unsigned tree_id, const char* name);
      void record_physical_instance_region(UniqueID op_id, IDType inst_id, 
                                           LogicalRegion handle);
      void record_physical_instance_fields(UniqueID op_id, IDType inst_id, 
                                  FieldSpace fs, std::vector<FieldID>& fields);
      void record_physical_instance_layout(UniqueID op_id,
                                           IDType inst_id, FieldSpace fs,
                                           const LayoutConstraints *lc);
      void record_index_part(UniqueID id, const char* name);
      void record_index_partition(UniqueID parent_id, UniqueID id, 
                                  bool disjoint, LegionColor c);
      void record_index_space_size(UniqueID unique_id,
                                   unsigned long long
                                   dense_size,
                                   unsigned long long
                                   sparse_size,
                                   bool is_sparse);
    public:
      void record_mapper_call_kinds(const char *const *const mapper_call_names,
                                    unsigned int num_mapper_call_kinds);
      void record_mapper_call(MappingCallKind kind, UniqueID uid,
                              timestamp_t start, timestamp_t stop);
    public:
      void record_runtime_call_kinds(const char *const *const runtime_calls,
                                     unsigned int num_runtime_call_kinds);
      void record_runtime_call(RuntimeCallKind kind, timestamp_t start,
                               timestamp_t stop);
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
    private:
      void create_thread_local_profiling_instance(void);
    public:
      Runtime *const runtime;
      // Event to trigger once the profiling is actually done
      const RtUserEvent done_event;
      // Size in bytes of the footprint before we start dumping
      const size_t output_footprint_threshold;
      // The goal size in microseconds of the output tasks
      const long long output_target_latency;
      // Target processor on which to launch jobs
      const Processor target_proc;
    private:
      LegionProfSerializer* serializer;
      mutable LocalLock profiler_lock;
      std::vector<LegionProfInstance*> instances;
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
    public:
      void record_index_space_point_desc(
          LegionProfInstance::IndexSpacePointDesc &i);
      void record_index_space_rect_desc(
          LegionProfInstance::IndexSpaceRectDesc &i);
      template <int DIM, typename T>
      void record_index_space_point(IDType handle, const Point<DIM, T> &point);
      template<int DIM, typename T>
      void record_index_space_rect(IDType handle, const Rect<DIM, T> &rect);
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
    inline void LegionProfiler::record_index_space_point(IDType handle,
                                                      const Point<DIM,T> &point)
    //--------------------------------------------------------------------------
    {
      LegionProfInstance::IndexSpacePointDesc ispace_point_desc;
      ispace_point_desc.unique_id = handle;
      ispace_point_desc.dim = (unsigned)DIM;
#define DIMFUNC(D2) \
      ispace_point_desc.points[D2-1] = (D2<=DIM) ? (long long)point[D2-1] : 0;
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      record_index_space_point_desc(ispace_point_desc);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline void LegionProfiler::record_index_space_rect(IDType handle,
                                                        const Rect<DIM,T> &rect)
    //--------------------------------------------------------------------------
    {
      LegionProfInstance::IndexSpaceRectDesc ispace_rect_desc;
      ispace_rect_desc.unique_id = handle;
      ispace_rect_desc.dim = DIM;
#define DIMFUNC(D2) \
      ispace_rect_desc.rect_lo[D2-1] = (D2<=DIM) ? (long long)rect.lo[D2-1]:0; \
      ispace_rect_desc.rect_hi[D2-1] = (D2<=DIM) ? (long long)rect.hi[D2-1]:0;
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      record_index_space_rect_desc(ispace_rect_desc);
    }

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_PROFILING_H__

