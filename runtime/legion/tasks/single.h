/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_SINGLE_TASK_H__
#define __LEGION_SINGLE_TASK_H__

#include "legion/tasks/task.h"
#include "legion/operations/collective.h"
#include "legion/utilities/instance_set.h"

namespace Legion {
  namespace Internal {

    /**
     * \class SingleTask
     * This is the parent type for each of the single class
     * kinds of classes.  It also serves as the type that
     * represents a context for each application level task.
     */
    class SingleTask : public TaskOp {
    public:
      struct MispredicationTaskArgs
        : public LgTaskArgs<MispredicationTaskArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_MISPREDICATION_TASK_ID;
      public:
        MispredicationTaskArgs(void) = default;
        MispredicationTaskArgs(SingleTask* t)
          : LgTaskArgs<MispredicationTaskArgs>(false, false), task(t)
        { }
        inline void execute(void) const { task->handle_mispredication(); }
      public:
        SingleTask* task;
      };
      struct OrderConcurrentLaunchArgs
        : public LgTaskArgs<OrderConcurrentLaunchArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_ORDER_CONCURRENT_LAUNCH_TASK_ID;
      public:
        OrderConcurrentLaunchArgs(void) = default;
        OrderConcurrentLaunchArgs(
            SingleTask* t, Processor p, ApEvent s, VariantID v)
          : LgTaskArgs<OrderConcurrentLaunchArgs>(false, false), task(t),
            processor(p), vid(v), start(s),
            ready(Runtime::create_ap_user_event(nullptr))
        { }
        void execute(void) const;
      public:
        SingleTask* task;
        Processor processor;
        VariantID vid;
        ApEvent start;
        ApUserEvent ready;
      };
    public:
      SingleTask(void);
      virtual ~SingleTask(void);
    public:
      virtual void trigger_dependence_analysis(void) override = 0;
    public:
      // These two functions are only safe to call after
      // the task has had its variant selected
      bool is_leaf(void) const;
      bool is_inner(void) const;
      inline bool is_concurrent(void) const { return concurrent_task; }
      bool is_created_region(unsigned index) const;
      void update_no_access_regions(void);
      void clone_single_from(SingleTask* task);
    public:
      inline void clone_virtual_mapped(std::vector<bool>& target) const
      {
        target = virtual_mapped;
      }
      inline void clone_parent_req_indexes(std::vector<unsigned>& target) const
      {
        target = parent_req_indexes;
      }
      inline const std::deque<InstanceSet>& get_physical_instances(void) const
      {
        return physical_instances;
      }
      inline const std::vector<bool>& get_no_access_regions(void) const
      {
        return no_access_regions;
      }
      inline VariantID get_selected_variant(void) const
      {
        return selected_variant;
      }
      inline const std::set<RtEvent>& get_map_applied_conditions(void) const
      {
        return map_applied_conditions;
      }
      inline RtEvent get_profiling_reported(void) const
      {
        return profiling_reported;
      }
      virtual ContextCoordinate get_task_tree_coordinate(void) const override
      {
        return ContextCoordinate(context_index, index_point);
      }
    public:
      void enqueue_ready_task(
          bool use_target_processor, RtEvent wait_on = RtEvent::NO_RT_EVENT);
      RtEvent perform_versioning_analysis(const bool post_mapper);
      virtual bool replicate_task(void);
      virtual void initialize_map_task_input(
          Mapper::MapTaskInput& input, Mapper::MapTaskOutput& output,
          MustEpochOp* must_epoch_owner);
      virtual bool finalize_map_task_output(
          Mapper::MapTaskInput& input, Mapper::MapTaskOutput& output,
          MustEpochOp* must_epoch_owner);
      void handle_post_mapped(RtEvent pre = RtEvent::NO_RT_EVENT);
    protected:
      void prepare_output_instance(
          unsigned index, InstanceSet& instance_set,
          const RegionRequirement& req, Memory target,
          const LayoutConstraintSet& constraints);
    public:
      virtual InnerContext* create_implicit_context(void);
      void configure_execution_context(Mapper::ContextConfigOutput& config);
      void set_shard_manager(ShardManager* manager);
    protected:  // mapper helper call
      void validate_target_processors(const std::vector<Processor>& prcs) const;
    protected:
      bool invoke_mapper(MustEpochOp* must_epoch_owner);
      bool map_all_regions(
          MustEpochOp* must_epoch_owner, const DeferMappingArgs* defer_args);
      void perform_post_mapping(const TraceInfo& trace_info);
      void check_future_return_bounds(FutureInstance* instance) const;
      void create_leaf_memory_pools(
          VariantImpl* impl,
          std::map<std::pair<Memory, bool>, PoolBounds>& dynamic_pool_bounds);
    public:
      bool acquire_leaf_memory_pool(
          Memory memory, const PoolBounds& bounds, bool escaping,
          bool optimistic, RtEvent* safe_for_unbounded_pools);
      void release_leaf_memory_pool(Memory memory, bool escaping);
    protected:
      void pack_single_task(Serializer& rez, AddressSpaceID target);
      void unpack_single_task(
          Deserializer& derez, std::set<RtEvent>& ready_events);
    public:
      virtual void pack_profiling_requests(
          Serializer& rez, std::set<RtEvent>& applied) const override;
      virtual int add_copy_profiling_request(
          const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& requests,
          bool fill, unsigned count = 1) override;
      virtual bool handle_profiling_response(
          const Realm::ProfilingResponse& response, const void* orig,
          size_t orig_length, LgEvent& fevent, bool& failed_alloc) override;
      virtual void handle_profiling_update(int count) override;
      void finalize_single_task_profiling(void);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual bool is_top_level_task(void) const { return false; }
      virtual bool is_shard_task(void) const { return false; }
      virtual SingleTask* get_origin_task(void) const = 0;
    public:
      virtual void predicate_false(void) override = 0;
      virtual void launch_task(bool inline_task = false) override;
      virtual bool distribute_task(void) override = 0;
      virtual bool perform_mapping(
          MustEpochOp* owner = nullptr,
          const DeferMappingArgs* args = nullptr) override = 0;
      virtual void handle_future_size(
          size_t return_type_size, std::set<RtEvent>& applied_events) = 0;
      virtual uint64_t order_collectively_mapped_unbounded_pools(
          uint64_t lamport_clock, bool need_result)
      {
        std::abort();
      }
      virtual ApEvent order_concurrent_launch(ApEvent start, VariantImpl* impl)
      {
        std::abort();
      }
      virtual void record_output_extent(
          unsigned idx, const DomainPoint& color, const DomainPoint& extents)
      {
        std::abort();
      }
      virtual void record_output_registered(
          RtEvent registered, std::set<RtEvent>& applied_events)
      {
        std::abort();
      }
      virtual void trigger_replay(void) override;
      // For tasks that are sharded off by control replication
      virtual void shard_off(RtEvent mapped_precondition);
      virtual bool is_stealable(void) const override = 0;
    public:
      virtual TaskKind get_task_kind(void) const override = 0;
    public:
      // Override these methods from operation class
      virtual void trigger_mapping(void) override;
    protected:
      friend class ShardManager;
      virtual void trigger_task_commit(void) override = 0;
    public:
      virtual bool send_task(
          Processor target, std::vector<SingleTask*>& others) = 0;
      virtual bool pack_task(
          Serializer& rez, AddressSpaceID target) override = 0;
      virtual bool unpack_task(
          Deserializer& derez, Processor current, AddressSpaceID source,
          std::set<RtEvent>& ready_events) override = 0;
      virtual void perform_inlining(
          VariantImpl* variant,
          const std::deque<InstanceSet>& parent_regions) override;
    public:
      virtual void handle_future(
          ApEvent effects, FutureInstance* instance, const void* metadata,
          size_t metasize, FutureFunctor* functor, Processor future_proc,
          bool own_functor) = 0;
      virtual void handle_mispredication(void) = 0;
    public:
      virtual void perform_concurrent_task_barrier(void) = 0;
    public:
      // From Memoizable
      virtual ApEvent replay_mapping(void) override;
    public:
      virtual void perform_replicate_collective_versioning(
          unsigned index, unsigned parent_req_index,
          op::map<LogicalRegion, CollectiveVersioningBase::RegionVersioning>&
              to_perform);
      virtual void convert_replicate_collective_views(
          const CollectiveViewCreatorBase::RendezvousKey& key,
          std::map<
              LogicalRegion, CollectiveViewCreatorBase::CollectiveRendezvous>&
              rendezvous);
    public:
      void handle_remote_profiling_response(Deserializer& derez);
    public:
      virtual void concurrent_allreduce(
          ProcessorManager* manager, uint64_t lamport_clock, VariantID vid,
          bool poisoned) = 0;
      void record_inner_termination(ApEvent termination_event);
    protected:
      virtual TaskContext* create_execution_context(
          VariantImpl* v, std::set<ApEvent>& launch_events, bool inline_task,
          bool leaf_task);
    protected:
      // Boolean for each region saying if it is virtual mapped
      std::vector<bool> virtual_mapped;
      // Regions which are NO_ACCESS or have no privilege fields
      std::vector<bool> no_access_regions;
    protected:
      std::vector<Processor> target_processors;
      // Hold the result of the mapping
      std::deque<InstanceSet> physical_instances;
      std::vector<ApEvent> region_preconditions;
      std::vector<std::vector<PhysicalManager*> > source_instances;
      std::vector<Memory> future_memories;
      std::map<std::pair<Memory, bool /*escaping*/>, MemoryPool*>
          leaf_memory_pools;
    protected:  // Mapper choices
      std::vector<unsigned> untracked_valid_regions;
      VariantID selected_variant;
      TaskPriority task_priority;
      bool perform_postmap;
    protected:
      // Events that must be triggered before we are done mapping
      std::set<RtEvent> map_applied_conditions;
      // The single task termination event encapsulates the exeuction of the
      // task being done and all child operations and their effects being done
      // It does NOT encapsulate the 'effects_complete' of this task
      // Only the actual operation completion event captures that
      ApUserEvent single_task_termination;
    protected:
      TaskContext* execution_context;
      RemoteTraceRecorder* remote_trace_recorder;
      // For replication of this task
      ShardManager* shard_manager;
    protected:
      mutable std::atomic<bool> leaf_cached, is_leaf_result;
      mutable std::atomic<bool> inner_cached, is_inner_result;
    protected:
      std::vector<ProfilingMeasurementID> task_profiling_requests;
      std::vector<ProfilingMeasurementID> copy_profiling_requests;
      RtUserEvent profiling_reported;
      int profiling_priority;
      int copy_fill_priority;
      std::atomic<int> outstanding_profiling_requests;
      std::atomic<int> outstanding_profiling_reported;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_SINGLE_TASK_H__
