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

#ifndef __MAPPER_MANAGER_H__
#define __MAPPER_MANAGER_H__

#include "legion/legion_types.h"
#include "legion/legion_mapping.h"
#include "legion/legion_instances.h"

namespace Legion {
  namespace Internal { 

    /**
     * \class MapperManager
     * This is the base class for a bunch different kinds of mapper
     * managers. Some calls into this manager from the mapper will
     * be handled right away, while other we may need to defer and
     * possibly preempt.  This later class of calls are the ones that
     * are made virtual so that the 
     */
    class MapperManager : public InstanceDeletionSubscriber {
    public:
      struct AcquireStatus {
      public:
        std::set<PhysicalManager*> instances;
        std::vector<bool> results;
      }; 
      struct DeferMessageArgs : public LgTaskArgs<DeferMessageArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_MAPPER_MESSAGE_TASK_ID;
      public:
        DeferMessageArgs(MapperManager *man, Processor p, unsigned k,
                         void *m, size_t s, bool b)
          : LgTaskArgs<DeferMessageArgs>(implicit_provenance),
            manager(man), sender(p), kind(k), 
            message(m), size(s), broadcast(b) { }
      public:
        MapperManager *const manager;
        const Processor sender;
        const unsigned kind;
        void *const message;
        const size_t size;
        const bool broadcast;
      };
      struct DeferInstanceCollectionArgs :
        public LgTaskArgs<DeferInstanceCollectionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_MAPPER_COLLECTION_TASK_ID;
      public:
        DeferInstanceCollectionArgs(MapperManager *man, PhysicalManager *inst)
          : LgTaskArgs<DeferInstanceCollectionArgs>(implicit_provenance),
            manager(man), instance(inst) { }
      public:
        MapperManager *const manager;
        PhysicalManager *const instance;
      };
    public:
      MapperManager(Runtime *runtime, Mapping::Mapper *mapper, 
                    MapperID map_id, Processor p, bool is_default);
      virtual ~MapperManager(void);
    public:
      const char* get_mapper_name(void);
    public: // Task mapper calls
      void invoke_select_task_options(TaskOp *task, Mapper::TaskOptions &output,
                                      bool prioritize);
      void invoke_premap_task(TaskOp *task, Mapper::PremapTaskInput &input,
                              Mapper::PremapTaskOutput &output); 
      void invoke_slice_task(TaskOp *task, Mapper::SliceTaskInput &input,
                               Mapper::SliceTaskOutput &output); 
      void invoke_map_task(TaskOp *task, Mapper::MapTaskInput &input,
                           Mapper::MapTaskOutput &output); 
      void invoke_replicate_task(TaskOp *task, 
                                 Mapper::ReplicateTaskInput &input,
                                 Mapper::ReplicateTaskOutput &output);
      void invoke_select_task_variant(TaskOp *task, 
                                      Mapper::SelectVariantInput &input,
                                      Mapper::SelectVariantOutput &output);
      void invoke_post_map_task(TaskOp *task, Mapper::PostMapInput &input,
                                Mapper::PostMapOutput &output);
      void invoke_select_task_sources(TaskOp *task, 
                                      Mapper::SelectTaskSrcInput &input,
                                      Mapper::SelectTaskSrcOutput &output);
      void invoke_select_task_sources(RemoteTaskOp *task, 
                                      Mapper::SelectTaskSrcInput &input,
                                      Mapper::SelectTaskSrcOutput &output);
      void invoke_task_report_profiling(TaskOp *task, 
                                        Mapper::TaskProfilingInfo &input);
      void invoke_task_select_sharding_functor(TaskOp *task,
                              Mapper::SelectShardingFunctorInput &input,
                              Mapper::SelectShardingFunctorOutput &output);
    public: // Inline mapper calls
      void invoke_map_inline(MapOp *op, Mapper::MapInlineInput &input,
                             Mapper::MapInlineOutput &output); 
      void invoke_select_inline_sources(MapOp *op, 
                                        Mapper::SelectInlineSrcInput &input,
                                        Mapper::SelectInlineSrcOutput &output);
      void invoke_select_inline_sources(RemoteMapOp *op, 
                                        Mapper::SelectInlineSrcInput &input,
                                        Mapper::SelectInlineSrcOutput &output);
      void invoke_inline_report_profiling(MapOp *op, 
                                          Mapper::InlineProfilingInfo &input);
    public: // Copy mapper calls
      void invoke_map_copy(CopyOp *op,
                           Mapper::MapCopyInput &input,
                           Mapper::MapCopyOutput &output);
      void invoke_select_copy_sources(CopyOp *op,
                                      Mapper::SelectCopySrcInput &input,
                                      Mapper::SelectCopySrcOutput &output);
      void invoke_select_copy_sources(RemoteCopyOp *op,
                                      Mapper::SelectCopySrcInput &input,
                                      Mapper::SelectCopySrcOutput &output);
      void invoke_copy_report_profiling(CopyOp *op,
                                        Mapper::CopyProfilingInfo &input);
      void invoke_copy_select_sharding_functor(CopyOp *op,
                              Mapper::SelectShardingFunctorInput &input,
                              Mapper::SelectShardingFunctorOutput &output);
    public: // Close mapper calls
      void invoke_select_close_sources(CloseOp *op,
                                       Mapper::SelectCloseSrcInput &input,
                                       Mapper::SelectCloseSrcOutput &output);
      void invoke_select_close_sources(RemoteCloseOp *op,
                                       Mapper::SelectCloseSrcInput &input,
                                       Mapper::SelectCloseSrcOutput &output);
      void invoke_close_report_profiling(CloseOp *op,
                                         Mapper::CloseProfilingInfo &input);
      void invoke_close_select_sharding_functor(CloseOp *op,
                              Mapper::SelectShardingFunctorInput &input,
                              Mapper::SelectShardingFunctorOutput &output);
    public: // Acquire mapper calls
      void invoke_map_acquire(AcquireOp *op,
                              Mapper::MapAcquireInput &input,
                              Mapper::MapAcquireOutput &output);
      void invoke_acquire_report_profiling(AcquireOp *op,
                                           Mapper::AcquireProfilingInfo &input);
      void invoke_acquire_select_sharding_functor(AcquireOp *op,
                              Mapper::SelectShardingFunctorInput &input,
                              Mapper::SelectShardingFunctorOutput &output);
    public: // Release mapper calls
      void invoke_map_release(ReleaseOp *op,
                              Mapper::MapReleaseInput &input,
                              Mapper::MapReleaseOutput &output);
      void invoke_select_release_sources(ReleaseOp *op,
                                         Mapper::SelectReleaseSrcInput &input,
                                         Mapper::SelectReleaseSrcOutput &output);
      void invoke_select_release_sources(RemoteReleaseOp *op,
                                         Mapper::SelectReleaseSrcInput &input,
                                         Mapper::SelectReleaseSrcOutput &output);
      void invoke_release_report_profiling(ReleaseOp *op,
                                           Mapper::ReleaseProfilingInfo &input);
      void invoke_release_select_sharding_functor(ReleaseOp *op,
                              Mapper::SelectShardingFunctorInput &input,
                              Mapper::SelectShardingFunctorOutput &output);
    public: // Partition mapper calls
      void invoke_select_partition_projection(DependentPartitionOp *op,
                          Mapper::SelectPartitionProjectionInput &input,
                          Mapper::SelectPartitionProjectionOutput &output);
      void invoke_map_partition(DependentPartitionOp *op,
                          Mapper::MapPartitionInput &input,
                          Mapper::MapPartitionOutput &output);
      void invoke_select_partition_sources(DependentPartitionOp *op,
                          Mapper::SelectPartitionSrcInput &input,
                          Mapper::SelectPartitionSrcOutput &output);
      void invoke_select_partition_sources(RemotePartitionOp *op,
                          Mapper::SelectPartitionSrcInput &input,
                          Mapper::SelectPartitionSrcOutput &output);
      void invoke_partition_report_profiling(DependentPartitionOp *op,
                          Mapper::PartitionProfilingInfo &input);
      void invoke_partition_select_sharding_functor(DependentPartitionOp *op,
                              Mapper::SelectShardingFunctorInput &input,
                              Mapper::SelectShardingFunctorOutput &output);
    public: // Fill mapper calls
      void invoke_fill_select_sharding_functor(FillOp *op,
                              Mapper::SelectShardingFunctorInput &input,
                              Mapper::SelectShardingFunctorOutput &output);
    public: // All reduce 
      void invoke_map_future_map_reduction(AllReduceOp *op,
                              Mapper::FutureMapReductionInput &input,
                              Mapper::FutureMapReductionOutput &output);
    public: // Task execution mapper calls
      void invoke_configure_context(TaskOp *task,
                                    Mapper::ContextConfigOutput &output);
      void invoke_select_tunable_value(TaskOp *task,
                                       Mapper::SelectTunableInput &input,
                                       Mapper::SelectTunableOutput &output);
    public: // must epoch and graph mapper calls
      void invoke_must_epoch_select_sharding_functor(MustEpochOp *op,
                              Mapper::SelectShardingFunctorInput &input,
                              Mapper::MustEpochShardingFunctorOutput &output);
      void invoke_map_must_epoch(MustEpochOp *op,
                                 Mapper::MapMustEpochInput &input,
                                 Mapper::MapMustEpochOutput &output);
      void invoke_map_dataflow_graph(Mapper::MapDataflowGraphInput &input,
                                     Mapper::MapDataflowGraphOutput &output);
    public: // memoization mapper calls
      void invoke_memoize_operation(Mappable *mappable,
                                    Mapper::MemoizeInput &input,
                                    Mapper::MemoizeOutput &output);
    public: // scheduling and stealing mapper calls
      void invoke_select_tasks_to_map(Mapper::SelectMappingInput &input,
                                      Mapper::SelectMappingOutput &output);
      void invoke_select_steal_targets(Mapper::SelectStealingInput &input,
                                       Mapper::SelectStealingOutput &output);
      void invoke_permit_steal_request(Mapper::StealRequestInput &input,
                                       Mapper::StealRequestOutput &output);
    public: // handling mapper calls
      void invoke_handle_message(Mapper::MapperMessage *message);
      void invoke_handle_task_result(Mapper::MapperTaskResult &result);
    public:
      // Instance deletion subscriber methods
      virtual void notify_instance_deletion(PhysicalManager *manager);
      virtual void add_subscriber_reference(PhysicalManager *manager);
      virtual bool remove_subscriber_reference(PhysicalManager *manager);
    public:
      virtual bool is_locked(MappingCallInfo *info) = 0;
      virtual void lock_mapper(MappingCallInfo *info, bool read_only) = 0;
      virtual void unlock_mapper(MappingCallInfo *info) = 0;
    public:
      virtual bool is_reentrant(MappingCallInfo *info) = 0;
      virtual void enable_reentrant(MappingCallInfo *info) = 0;
      virtual void disable_reentrant(MappingCallInfo *info) = 0;
    protected:
      friend class Runtime;
      friend class MappingCallInfo;
      friend class Mapping::AutoLock;
      virtual void begin_mapper_call(MappingCallInfo *info,
                                     bool prioritize = false) = 0;
      virtual void pause_mapper_call(MappingCallInfo *info) = 0;
      virtual void resume_mapper_call(MappingCallInfo *info,
                                      RuntimeCallKind kind) = 0;
      virtual void finish_mapper_call(MappingCallInfo *info) = 0;
    public:
      virtual bool is_safe_for_unbounded_pools(void) = 0;
      virtual void report_unsafe_allocation_in_unbounded_pool(
          const MappingCallInfo *info, Memory memory, RuntimeCallKind kind) = 0;
    public:
      static const char* get_mapper_call_name(MappingCallKind kind);
    public:
      static void handle_deferred_message(const void *args);
      static void handle_deferred_collection(const void *args);
    public:
      // For stealing
      void process_advertisement(Processor advertiser); 
      void perform_stealing(std::multimap<Processor,MapperID> &targets);
    public:
      // For advertising
      void process_failed_steal(Processor thief);
      void perform_advertisements(std::set<Processor> &failed_waiters);
    public:
      Runtime *const runtime;
      Mapping::Mapper *const mapper;
      const MapperID mapper_id;
      const Processor processor;
      const bool profile_mapper;
      const bool request_valid_instances;
      const bool is_default_mapper;
      const bool initially_reentrant;
    protected:
      mutable LocalLock mapper_lock;
    protected: // Steal request information
      // Mappers on other processors that we've tried to steal from and failed
      std::set<Processor> steal_blacklist;
      // Mappers that have tried to steal from us and which we
      // should advertise work when we have it
      std::set<Processor> failed_thiefs;
    };

    /**
     * \class SerializingManager
     * In this class at most one mapper call can be running at 
     * a time. Mapper calls that invoke expensive runtime operations
     * can be pre-empted and it is up to the mapper to control
     * whether additional mapper calls when the call is blocked.
     */
    class SerializingManager : public MapperManager {
    public:
      SerializingManager(Runtime *runtime, Mapping::Mapper *mapper,
         MapperID map_id, Processor p, bool reentrant, bool is_default = false);
      SerializingManager(const SerializingManager &rhs);
      virtual ~SerializingManager(void);
    public:
      SerializingManager& operator=(const SerializingManager &rhs);
    public:
      virtual bool is_locked(MappingCallInfo *info);
      virtual void lock_mapper(MappingCallInfo *info, bool read_only);
      virtual void unlock_mapper(MappingCallInfo *info);
    public:
      virtual bool is_reentrant(MappingCallInfo *info);
      virtual void enable_reentrant(MappingCallInfo *info);
      virtual void disable_reentrant(MappingCallInfo *info);
    protected:
      virtual void begin_mapper_call(MappingCallInfo *info,
                                     bool prioritize = false);
      virtual void pause_mapper_call(MappingCallInfo *info);
      virtual void resume_mapper_call(MappingCallInfo *info,
                                      RuntimeCallKind kind);
      virtual void finish_mapper_call(MappingCallInfo *info);
    public:
      virtual bool is_safe_for_unbounded_pools(void);
      virtual void report_unsafe_allocation_in_unbounded_pool(
          const MappingCallInfo *info, Memory memory, RuntimeCallKind kind);
    protected:
      // Must be called while holding the mapper reservation
      RtUserEvent complete_pending_pause_mapper_call(void);
      RtUserEvent complete_pending_finish_mapper_call(void);
    protected:
      // The one executing call if any otherwise NULL
      MappingCallInfo *executing_call; 
      // Calls yet to start running
      std::deque<MappingCallInfo*> pending_calls; 
      // Calls that are ready to resume after runtime work
      std::deque<MappingCallInfo*> ready_calls;
      // Number of calls paused due to runtime work
      unsigned paused_calls;
      // Whether this mapper supports reentrant mapper calls
      const bool allow_reentrant;
      // Whether or not we are currently supporting reentrant calls
      bool permit_reentrant;
      // A flag checking whether we have a pending paused mapper call
      std::atomic<bool> pending_pause_call;
      // A flag checking whether we have a pending finished call
      std::atomic<bool> pending_finish_call;
    };

    /**
     * \class ConcurrentManager
     * In this class many mapper calls can be running concurrently.
     * It is upper to the mapper to lock itself when necessary to 
     * protect internal state. Mappers can be locked in exclusive
     * or non-exclusive modes.
     */
    class ConcurrentManager : public MapperManager {
    public:
      enum LockState {
        UNLOCKED_STATE,
        READ_ONLY_STATE,
        EXCLUSIVE_STATE,
      };
    public:
      ConcurrentManager(Runtime *runtime, Mapping::Mapper *mapper,
                        MapperID map_id, Processor p, bool is_default = false);
      ConcurrentManager(const ConcurrentManager &rhs);
      virtual ~ConcurrentManager(void);
    public:
      ConcurrentManager& operator=(const ConcurrentManager &rhs);
    public:
      virtual bool is_locked(MappingCallInfo *info);
      virtual void lock_mapper(MappingCallInfo *info, bool read_only);
      virtual void unlock_mapper(MappingCallInfo *info);
    public:
      virtual bool is_reentrant(MappingCallInfo *info);
      virtual void enable_reentrant(MappingCallInfo *info);
      virtual void disable_reentrant(MappingCallInfo *info);
    protected:
      virtual void begin_mapper_call(MappingCallInfo *info,
                                     bool prioritize = false);
      virtual void pause_mapper_call(MappingCallInfo *info);
      virtual void resume_mapper_call(MappingCallInfo *info,
                                      RuntimeCallKind kind);
      virtual void finish_mapper_call(MappingCallInfo *info);
    public:
      virtual bool is_safe_for_unbounded_pools(void);
      virtual void report_unsafe_allocation_in_unbounded_pool(
          const MappingCallInfo *info, Memory memory, RuntimeCallKind kind);
    protected:
      // Must be called while holding the lock
      void release_lock(std::vector<RtUserEvent> &to_trigger); 
    protected:
      LockState lock_state;
      std::set<MappingCallInfo*> current_holders;
      std::deque<MappingCallInfo*> read_only_waiters;
      std::deque<MappingCallInfo*> exclusive_waiters;
    };

    class MappingCallInfo {
    public:
      MappingCallInfo(MapperManager *man, MappingCallKind k, 
                      Operation *op, bool prioritize = false);
      ~MappingCallInfo(void);
    public:
      inline void pause_mapper_call(void)
        { manager->pause_mapper_call(this); }
      inline void resume_mapper_call(RuntimeCallKind kind)
        { manager->resume_mapper_call(this, kind); }
      inline const char* get_mapper_name(void) const
        { return manager->get_mapper_name(); }
      inline const char* get_mapper_call_name(void) const
        { return manager->get_mapper_call_name(kind); }
      inline bool is_locked(void)
        { return manager->is_locked(this); }
      inline void lock_mapper(bool read_only)
        { manager->lock_mapper(this, read_only); }
      inline void unlock_mapper(void)
        { manager->unlock_mapper(this); }
      inline bool is_reentrant(void)
        { return manager->is_reentrant(this); }
      inline void enable_reentrant(void)
        { manager->enable_reentrant(this); }
      inline void disable_reentrant(void)
        { manager->disable_reentrant(this); }
      inline void report_unsafe_allocation_in_unbounded_pool(
          Memory m, RuntimeCallKind k)
        { manager->report_unsafe_allocation_in_unbounded_pool(this, m, k); }
      void record_acquired_instance(InstanceManager *manager, bool created);
      void release_acquired_instance(InstanceManager *manager);
      bool perform_acquires(
          const std::vector<MappingInstance> &instances,
          std::vector<unsigned> *to_erase = NULL, 
          bool filter_acquired_instances = false);
    public:
      MapperManager*const               manager;
      RtUserEvent                       resume;
      const MappingCallKind             kind;
      Operation*const                   operation;
      std::map<PhysicalManager*,unsigned/*count*/>* acquired_instances;
      long long                         start_time;
      long long                         pause_time;
      bool                              reentrant;
    };

  };
};

#endif // __MAPPER_MANAGER_H__
