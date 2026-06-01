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

#ifndef __LEGION_TASK_H__
#define __LEGION_TASK_H__

#include "legion/analysis/versioning.h"
#include "legion/api/mapping.h"
#include "legion/api/requirements.h"
#include "legion/operations/predicate.h"
#include "legion/utilities/buffers.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ExternalTask
     * An extention of the external-facing Task to help
     * with packing and unpacking them
     */
    class ExternalTask : public Task,
                         public ExternalMappable {
    public:
      ExternalTask(void);
    public:
      void pack_external_task(Serializer& rez, AddressSpaceID target) const;
      void unpack_external_task(Deserializer& derez);
    public:
      static void pack_output_requirement(
          const OutputRequirement& req, Serializer& rez);
    public:
      static void unpack_output_requirement(
          OutputRequirement& req, Deserializer& derez);
    public:
      virtual void set_context_index(uint64_t index) = 0;
    protected:
      BufferManager<Task, OPERATION_LIFETIME> arg_manager;
      BufferManager<Task, OPERATION_LIFETIME> local_arg_manager;
    };

    /**
     * \class TaskRegions
     * This is a helper class for accessing the region requirements of a task
     */
    class TaskRequirements {
    public:
      TaskRequirements(Task& t) : task(t) { }
    public:
      inline size_t size(void) const
      {
        return task.regions.size() + task.output_regions.size();
      }
      inline bool is_output_created(unsigned idx) const
      {
        if (idx < task.regions.size())
          return false;
        return (
            task.output_regions[idx - task.regions.size()].flags &
            LEGION_CREATED_OUTPUT_REQUIREMENT_FLAG);
      }
      inline RegionRequirement& operator[](unsigned idx)
      {
        return (idx < task.regions.size()) ?
                   task.regions[idx] :
                   task.output_regions[idx - task.regions.size()];
      }
      inline const RegionRequirement& operator[](unsigned idx) const
      {
        return (idx < task.regions.size()) ?
                   task.regions[idx] :
                   task.output_regions[idx - task.regions.size()];
      }
    private:
      Task& task;
    };

    /**
     * \class TaskOp
     * This is the base task operation class for all
     * kinds of tasks in the system.
     */
    class TaskOp : public ExternalTask,
                   public PredicatedOp {
    public:
      enum TaskKind {
        INDIVIDUAL_TASK_KIND,
        POINT_TASK_KIND,
        INDEX_TASK_KIND,
        SLICE_TASK_KIND,
        SHARD_TASK_KIND,
      };
      class OutputOptions {
      public:
        OutputOptions(void) : store(0) { }
        OutputOptions(bool global, bool bounded, bool grouped)
          : store((global ? 1 : 0) | (bounded ? 2 : 0) | (grouped ? 4 : 0))
        { }
      public:
        inline bool global_indexing(void) const { return (store & 1); }
        inline bool bounded_requirement(void) const { return (store & 2); }
        inline bool grouped_fields(void) const { return (store & 4); }
      private:
        uint8_t store;
      };
    public:
      struct TriggerTaskArgs : public LgTaskArgs<TriggerTaskArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_TRIGGER_TASK_ID;
      public:
        TriggerTaskArgs(void) = default;
        TriggerTaskArgs(TaskOp* t, DistributedID parent_ctx_did)
          : LgTaskArgs<TriggerTaskArgs>(false, false), op(t)
        {
          enclosing_context = parent_ctx_did;
          unique_op_id = t->get_unique_op_id();
        }
        inline void execute(void) const
        {
          implicit_operation = op;
          op->trigger_mapping();
        }
      public:
        TaskOp* op;
      };
      struct DeferMappingArgs : public LgTaskArgs<DeferMappingArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_PERFORM_MAPPING_TASK_ID;
      public:
        DeferMappingArgs(void) = default;
        DeferMappingArgs(
            TaskOp* op, MustEpochOp* owner, unsigned cnt,
            std::vector<unsigned>* performed, std::vector<ApEvent>* eff)
          : LgTaskArgs<DeferMappingArgs>(false, false), proxy_this(op),
            must_op(owner), invocation_count(cnt), performed_regions(performed),
            effects(eff)
        { }
        void execute(void) const;
      public:
        TaskOp* proxy_this;
        MustEpochOp* must_op;
        unsigned invocation_count;
        std::vector<unsigned>* performed_regions;
        std::vector<ApEvent>* effects;
      };
      struct FinalizeOutputEqKDTreeArgs
        : public LgTaskArgs<FinalizeOutputEqKDTreeArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_FINALIZE_OUTPUT_TREE_TASK_ID;
      public:
        FinalizeOutputEqKDTreeArgs(void) = default;
        FinalizeOutputEqKDTreeArgs(TaskOp* owner)
          : LgTaskArgs<FinalizeOutputEqKDTreeArgs>(false, false),
            proxy_this(owner)
        { }
        inline void execute(void) const
        {
          proxy_this->finalize_output_region_trees();
        }
      public:
        TaskOp* proxy_this;
      };
      struct DeferTriggerChildrenCommitArgs
        : public LgTaskArgs<DeferTriggerChildrenCommitArgs> {
      public:
        static constexpr LgTaskID TASK_ID =
            LG_DEFER_TRIGGER_CHILDREN_COMMIT_TASK_ID;
      public:
        DeferTriggerChildrenCommitArgs(void) = default;
        DeferTriggerChildrenCommitArgs(TaskOp* t)
          : LgTaskArgs<DeferTriggerChildrenCommitArgs>(false, false), task(t)
        { }
        inline void execute(void) const { task->trigger_children_committed(); }
      public:
        TaskOp* task;
      };
    public:
      TaskOp(void);
      virtual ~TaskOp(void);
    public:
      virtual UniqueID get_unique_id(void) const override;
      virtual uint64_t get_context_index(void) const override;
      virtual void set_context_index(uint64_t index) override;
      virtual bool has_parent_task(void) const override;
      virtual const Task* get_parent_task(void) const override;
      virtual const std::string_view& get_provenance_string(
          bool human = true) const override;
      virtual const char* get_task_name(void) const override;
      virtual bool is_reducing_future(void) const;
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
      virtual void pack_profiling_requests(
          Serializer& rez, std::set<RtEvent>& applied) const;
    public:
      bool is_remote(void) const;
      bool is_forward_progress_task(void);
      inline bool is_stolen(void) const { return (steal_count > 0); }
      inline bool is_origin_mapped(void) const { return map_origin; }
      inline bool is_replicable(void) const { return replicate; }
      int get_depth(void) const;
    public:
      void set_current_proc(Processor current);
      inline void set_origin_mapped(bool origin) { map_origin = origin; }
      inline void set_replicated(bool repl) { replicate = repl; }
      inline void set_target_proc(Processor next) { target_proc = next; }
    public:
      void set_must_epoch(
          MustEpochOp* epoch, unsigned index, bool do_registration);
    public:
      void pack_base_task(Serializer& rez, AddressSpaceID target);
      void unpack_base_task(
          Deserializer& derez, std::set<RtEvent>& ready_events);
      void pack_base_external_task(Serializer& rez, AddressSpaceID target);
      void unpack_base_external_task(Deserializer& derez);
    public:
      void mark_stolen(void);
      void initialize_base_task(
          InnerContext* ctx, const Predicate& p, Processor::TaskFuncID tid,
          Provenance* provenance);
    public:
      bool select_task_options(bool prioritize);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual size_t get_region_count(void) const override;
      virtual Mappable* get_mappable(void) override;
      virtual bool invalidates_physical_trace_template(
          bool& exec_fence) const override
      {
        exec_fence = false;
        return !regions.empty();
      }
    public:
      virtual void trigger_dependence_analysis(void) override = 0;
      virtual void trigger_commit(void) override;
    public:
      virtual void predicate_false(void) override = 0;
    public:
      virtual void select_sources(
          const unsigned index, PhysicalManager* target,
          const std::vector<InstanceView*>& sources,
          std::vector<unsigned>& ranking,
          std::map<unsigned, PhysicalManager*>& points) override;
      virtual void update_atomic_locks(
          const unsigned index, Reservation lock, bool exclusive) override;
      virtual unsigned find_parent_index(unsigned idx) override;
      virtual VersionInfo& get_version_info(unsigned idx);
      virtual const VersionInfo& get_version_info(unsigned idx) const;
      virtual std::map<PhysicalManager*, unsigned>* get_acquired_instances_ref(
          void) override;
    public:
      virtual bool distribute_task(void) = 0;
      virtual bool perform_mapping(
          MustEpochOp* owner = nullptr,
          const DeferMappingArgs* args = nullptr) = 0;
      virtual void launch_task(bool inline_task = false) = 0;
      virtual bool is_stealable(void) const = 0;
      virtual bool is_output_global(unsigned idx) const { return false; }
      virtual bool is_output_bounded(unsigned idx) const { return false; }
      virtual bool is_output_grouped(unsigned idx) const { return false; }
    public:
      virtual TaskKind get_task_kind(void) const = 0;
    public:
      // Returns true if the task should be deactivated
      virtual bool pack_task(Serializer& rez, AddressSpaceID target) = 0;
      virtual bool unpack_task(
          Deserializer& derez, Processor current, AddressSpaceID source,
          std::set<RtEvent>& ready_events) = 0;
      virtual void perform_inlining(
          VariantImpl* variant,
          const std::deque<InstanceSet>& parent_regions) = 0;
    public:
      bool defer_perform_mapping(
          RtEvent precondition, MustEpochOp* op, unsigned invocation_count = 0,
          std::vector<unsigned>* performed = nullptr,
          std::vector<ApEvent>* effects = nullptr);
    public:
      // Tell the parent context that this task is in a ready queue
      void activate_outstanding_task(void);
      void deactivate_outstanding_task(void);
    public:
      void clone_task_op_from(TaskOp* rhs, Processor p, bool stealable);
      void update_grants(const std::vector<Grant>& grants);
      void update_arrival_barriers(const std::vector<PhaseBarrier>& barriers);
      void finalize_output_region_trees(void);
    public:
      void compute_parent_indexes(bool force);
    public:
      // From Memoizable
      virtual const RegionRequirement& get_requirement(
          unsigned idx) const override
      {
        return logical_regions[idx];
      }
      virtual unsigned get_output_offset() const override
      {
        return regions.size();
      }
    public:  // helper for mapping, here because of inlining
      void validate_variant_selection(
          MapperManager* local_mapper, VariantImpl* impl, Processor::Kind kind,
          const std::deque<InstanceSet>& physical_instances,
          const char* call_name) const;
    public:
      // These methods get called once the task has executed
      // and all the children have either mapped, completed,
      // or committed.
      void trigger_children_committed(RtEvent pre = RtEvent::NO_RT_EVENT);
    protected:
      // Tasks have two requirements to commit:
      // - all commit dependences must be satisfied (trigger_commit)
      // - all children must commit (children_committed)
      virtual void trigger_task_commit(void) = 0;
    protected:
      TaskRequirements logical_regions;
      // Region requirements to check for collective behavior
      std::vector<unsigned> check_collective_regions;
      // A map of any locks that we need to take for this task
      std::map<Reservation, bool /*exclusive*/> atomic_locks;
      // Set of acquired instances for this task
      std::map<PhysicalManager*, unsigned /*ref count*/> acquired_instances;
    protected:
      std::vector<unsigned> parent_req_indexes;
      // The version infos for this task
      op::vector<VersionInfo> version_infos;
    protected:
      // Whether we have an optional future return value
      std::optional<size_t> future_return_size;
    protected:
      bool commit_received;
    protected:
      bool options_selected;
      bool memoize_selected;
      bool map_origin;
      bool request_valid_instances;
      bool elide_future_return;
      bool replicate;
    private:
      mutable bool is_local;
      mutable bool local_cached;
    protected:
      bool children_commit;
    protected:
      MapperManager* mapper;
    public:
      // Index for this must epoch op
      unsigned must_epoch_index;
    public:
      static void log_requirement(
          UniqueID uid, unsigned idx, const RegionRequirement& req);
    };

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const TaskOp& op)
    //--------------------------------------------------------------------------
    {
      os << op.get_task_name() << " (UID: " << op.get_unique_op_id() << ")";
      return os;
    }

    /**
     * \class TaskImpl
     * This class is used for storing all the meta-data associated
     * with a logical task
     */
    class TaskImpl : public Heapify<TaskImpl, RUNTIME_LIFETIME> {
    public:
      struct SemanticRequestArgs : public LgTaskArgs<SemanticRequestArgs> {
      public:
        static constexpr LgTaskID TASK_ID =
            LG_TASK_IMPL_SEMANTIC_INFO_REQ_TASK_ID;
      public:
        SemanticRequestArgs(void) = default;
        SemanticRequestArgs(TaskImpl* proxy, SemanticTag t, AddressSpaceID src)
          : LgTaskArgs<SemanticRequestArgs>(false, false), proxy_this(proxy),
            tag(t), source(src)
        { }
        void execute(void) const;
      public:
        TaskImpl* proxy_this;
        SemanticTag tag;
        AddressSpaceID source;
      };
    public:
      TaskImpl(TaskID tid, const char* name = nullptr);
      TaskImpl(const TaskImpl& rhs) = delete;
      ~TaskImpl(void);
    public:
      TaskImpl& operator=(const TaskImpl& rhs) = delete;
    public:
      VariantID get_unique_variant_id(void);
      void add_variant(VariantImpl* impl);
      VariantImpl* find_variant_impl(VariantID variant_id, bool can_fail);
      void find_valid_variants(
          std::vector<VariantID>& valid_variants, Processor::Kind kind) const;
    public:
      const char* get_name(bool needs_lock = true);
      void attach_semantic_information(
          SemanticTag tag, AddressSpaceID source, const void* buffer,
          size_t size, bool is_mutable, bool send_to_owner);
      bool retrieve_semantic_information(
          SemanticTag tag, const void*& buffer, size_t& size, bool can_fail,
          bool wait_until);
      void send_semantic_info(
          AddressSpaceID target, SemanticTag tag, const void* value,
          size_t size, bool is_mutable,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT);
      void send_semantic_request(
          AddressSpaceID target, SemanticTag tag, bool can_fail,
          bool wait_until, RtUserEvent ready);
      void process_semantic_request(
          SemanticTag tag, AddressSpaceID target, bool can_fail,
          bool wait_until, RtUserEvent ready);
    public:
      inline AddressSpaceID get_owner_space(void) const
      {
        return get_owner_space(task_id);
      }
      static AddressSpaceID get_owner_space(TaskID task_id);
    public:
      const TaskID task_id;
      char* const initial_name;
    private:
      mutable LocalLock task_lock;
      std::map<VariantID, VariantImpl*> variants;
      // VariantIDs that we've handed out but haven't registered yet
      std::set<VariantID> pending_variants;
      std::map<SemanticTag, SemanticInfo> semantic_infos;
      // Track whether all these variants are idempotent or not
      bool all_idempotent;
    };

    /**
     * \class VariantImpl
     * This class is used for storing all the meta-data associated
     * with a particular variant implementation of a task
     */
    class VariantImpl : public Heapify<VariantImpl, RUNTIME_LIFETIME> {
    public:
      VariantImpl(
          VariantID vid, TaskImpl* owner, const TaskVariantRegistrar& registrar,
          size_t return_type_size, bool has_return_type_size,
          const CodeDescriptor& realm_desc, const void* user_data = nullptr,
          size_t user_data_size = 0);
      VariantImpl(const VariantImpl& rhs) = delete;
      ~VariantImpl(void);
    public:
      VariantImpl& operator=(const VariantImpl& rhs) = delete;
    public:
      inline bool is_leaf(void) const { return leaf_variant; }
      inline bool is_inner(void) const { return inner_variant; }
      inline bool is_idempotent(void) const { return idempotent_variant; }
      inline bool is_replicable(void) const { return replicable_variant; }
      inline bool is_concurrent(void) const { return concurrent_variant; }
      inline bool needs_barrier(void) const { return concurrent_barrier; }
      inline const char* get_name(void) const { return variant_name; }
      inline const ExecutionConstraintSet& get_execution_constraints(void) const
      {
        return execution_constraints;
      }
      inline const TaskLayoutConstraintSet& get_layout_constraints(void) const
      {
        return layout_constraints;
      }
    public:
      bool is_no_access_region(unsigned idx) const;
    public:
      ApEvent dispatch_task(
          Processor target, SingleTask* task, TaskContext* ctx,
          ApEvent precondition, int priority,
          Realm::ProfilingRequestSet& requests);
    public:
      bool can_use(Processor::Kind kind, bool warn) const;
    public:
      void broadcast_variant(
          RtUserEvent done, AddressSpaceID origin, AddressSpaceID local);
      void find_padded_locks(
          SingleTask* task, const std::vector<RegionRequirement>& regions,
          const std::deque<InstanceSet>& physical_instances) const;
      void record_padded_fields(
          const std::vector<RegionRequirement>& regions,
          const std::vector<PhysicalRegion>& physical_regions) const;
    public:
      static bool check_padding(const TaskLayoutConstraintSet& constraints);
      static std::map<std::pair<Memory::Kind, bool>, PoolBounds>
          make_pool_bounds(
              TaskImpl* impl, VariantID v,
              const TaskVariantRegistrar& registrar);
    public:
      const VariantID vid;
      TaskImpl* const owner;
      const bool global;  // globally valid variant
      const bool needs_padding;
      const bool has_return_type_size;
      const size_t return_type_size;
    public:
      const CodeDescriptorID descriptor_id;
      CodeDescriptor realm_descriptor;
    public:
      const ExecutionConstraintSet execution_constraints;
      const TaskLayoutConstraintSet layout_constraints;
      const std::map<std::pair<Memory::Kind, bool>, PoolBounds>
          leaf_pool_bounds;
    private:
      void* user_data;
      size_t user_data_size;
      ApEvent ready_event;
    private:  // properties
      const bool leaf_variant;
      const bool inner_variant;
      const bool idempotent_variant;
      const bool replicable_variant;
      const bool concurrent_variant;
      const bool concurrent_barrier;
    private:
      char* variant_name;
    };

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const VariantImpl& impl)
    //--------------------------------------------------------------------------
    {
      os << impl.get_name();
      return os;
    }

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_TASK_H__
