/* Copyright 2013 Stanford University
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


#ifndef __LEGION_OPS_H__
#define __LEGION_OPS_H__

#include "legion_types.h"
#include "legion.h"
#include "region_tree.h"

namespace LegionRuntime {
  namespace HighLevel {

    /////////////////////////////////////////////////////////////
    // Generalized Operation 
    /////////////////////////////////////////////////////////////
    /**
     * A class for representing all operations.  Has a common
     * interface for doing mapping dependence analysis as well
     * as operation activation and deactivation.
     */
    class GeneralizedOperation : public Lockable, public Mappable { // include Lockable for fine-grained locking inside object
    public:
      GeneralizedOperation(HighLevelRuntime *rt);
      virtual ~GeneralizedOperation(void);
    public:
      bool activate_base(GeneralizedOperation *parent);
      void deactivate_base(void);
      // Always make sure to lock the context before acquiring
      // our own lock if need be
      void lock_context(bool exclusive = true) const;
      void unlock_context(void) const;
#ifdef DEBUG_HIGH_LEVEL
      void assert_context_locked(void) const; // Assert that the lock has been taken
      void assert_context_not_locked(void) const; // Assert that the lock has not been taken
#endif
#ifdef LEGION_SPY
      UniqueID get_unique_id(int gen = -1) const;
#else
      UniqueID get_unique_id(void) const;
#endif
    public:
      // Mapping dependence operations
      bool is_ready();
      void notify(void);
      // These prevent race conditions when doing dependence analysis
      // by increasing the outstanding dependence count by 1 during analysis
      // so we can't trigger prematurely.  This also ensure we catch the case
      // where there are no dependences.
      inline void start_analysis(void) { outstanding_dependences++; }
      inline void finish_analysis(void) { notify(); }
      inline GenerationID get_gen(void) const { return generation; }
      virtual void add_mapping_dependence(unsigned idx, const LogicalUser &prev, DependenceType dtype) = 0;
      virtual bool add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen) = 0;
    public:
      virtual bool activate(GeneralizedOperation *parent = NULL) = 0;
      virtual void deactivate(void) = 0; 
      virtual bool perform_dependence_analysis(void) = 0; // return true if op can be deactivated
      virtual bool perform_operation(void) = 0; // return true if operation succeeded
      virtual MapperID get_mapper_id(void) const = 0;
      virtual bool has_mapped(GenerationID gen) = 0; // not precise
      virtual Event get_termination_event(void) const = 0;
    protected:
      // Called once the task is ready to map
      virtual void trigger(void) = 0;
    protected:
      void clone_generalized_operation_from(GeneralizedOperation *rhs);
    protected:
      LegionErrorType verify_requirement(const RegionRequirement &req, 
                                         FieldID &bad_field, size_t &bad_size, unsigned &bad_idx);
    protected:
      size_t compute_operation_size(void);
      void pack_operation(Serializer &rez);
      void unpack_operation(Deserializer &derez);
    protected:
      bool active;
      bool context_owner;
      UniqueID unique_id;
      RegionTreeForest *forest_ctx;
      GenerationID generation;
      unsigned outstanding_dependences;
      HighLevelRuntime *const runtime;
#ifdef LEGION_SPY
      std::vector<UniqueID> previous_ids;
#endif
    };

    /////////////////////////////////////////////////////////////
    // Epoch Operation 
    /////////////////////////////////////////////////////////////
    /**
     * This is a dummy operation that acts as a summary of a bunch
     * of other operations in a logical state.  As soon as it
     * triggers it automatically marks itself as mapped so that
     * anyone else that was waiting on it can map.  The idea
     * is that it delineates the boundaries between epochs.
     */
    class EpochOperation : public GeneralizedOperation {
    public:
      EpochOperation(HighLevelRuntime *rt);
      ~EpochOperation(void);
    public:
      virtual void add_mapping_dependence(unsigned idx, const LogicalUser &prev, DependenceType dtype);
      virtual bool add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen);
    public:
      virtual bool activate(GeneralizedOperation *parent = NULL);
      virtual void deactivate(void);
      virtual bool perform_dependence_analysis(void);
      virtual bool perform_operation(void);
      virtual MapperID get_mapper_id(void) const;
      virtual bool has_mapped(GenerationID gen);
      virtual Event get_termination_event(void) const;
    protected:
      virtual void trigger(void);
    public:
      inline void increment_dependence_count(void) { outstanding_dependences++; }
      inline void start_down_sample(void) { lock(); }
      inline void end_down_sample(void) { unlock(); }
    protected:
      Context parent_ctx; 
      std::set<GeneralizedOperation*> map_dependent_waiters;
    };

    /////////////////////////////////////////////////////////////
    // Deferred Operation 
    /////////////////////////////////////////////////////////////
    /**
     * This is an intermediate class that will be the base
     * of any operations which are just deferrals of operations
     * that are going to be done at mapping dependence time.
     */
    class DeferredOperation : public GeneralizedOperation {
    public:
      DeferredOperation(HighLevelRuntime *rt);
      virtual ~DeferredOperation(void);
    public:
      // Methods which are left virtual to be filled in
      virtual bool activate(GeneralizedOperation *parent = NULL) = 0;
      virtual void deactivate(void) = 0;
    public:
      // Methods which should never be called
      virtual void add_mapping_dependence(unsigned idx, const LogicalUser &prev, DependenceType dtype);
      virtual bool add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen);
      virtual bool perform_dependence_analysis(void);
      virtual bool perform_operation(void);
      virtual void trigger(void);
      virtual MapperID get_mapper_id(void) const;
      virtual bool has_mapped(GenerationID gen);
      virtual Event get_termination_event(void) const;
    public:
      // Method to be implemented by base class
      virtual void perform_deferred(void) = 0;
    };

    /////////////////////////////////////////////////////////////
    // Mapping Operation 
    /////////////////////////////////////////////////////////////
    /**
     * A class for inline mapping operations.
     */
    class MappingOperation : public GeneralizedOperation {
    public:
      MappingOperation(HighLevelRuntime *rt);
      virtual ~MappingOperation(void);
    public:
      void initialize(Context ctx, const RegionRequirement &req, MapperID id, MappingTagID tag, bool check_priv);
      void initialize(Context ctx, unsigned idx, MapperID id, MappingTagID tag, bool check_priv);
    public:
      bool is_valid(GenerationID gen_id) const;
      void wait_until_valid(GenerationID gen_id);
      LogicalRegion get_logical_region(GenerationID gen_id) const; 
      PhysicalInstance get_physical_instance(GenerationID gen_id) const;
      //bool has_accessor(GenerationID gen_id, AccessorType at) const;
      Accessor::RegionAccessor<Accessor::AccessorType::Generic> get_accessor(GenerationID gen_id) const;
      Accessor::RegionAccessor<Accessor::AccessorType::Generic> get_field_accessor(GenerationID gen_id, FieldID fid) const;
      PhysicalRegion get_physical_region(void);
      Event get_map_event(void) const;
      Event get_unmap_event(void) const;
      bool has_region_idx(void) const;
      unsigned get_region_idx(void) const;
    public:
      // Functions from GenerlizedOperation
      virtual bool activate(GeneralizedOperation *parent = NULL);
      virtual void deactivate(void);
      virtual void add_mapping_dependence(unsigned idx, const LogicalUser &prev, DependenceType dtype);
      virtual bool add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen);
      virtual bool perform_dependence_analysis(void);
      virtual bool perform_operation(void);
      virtual void trigger(void);
      virtual MapperID get_mapper_id(void) const;
      virtual bool has_mapped(GenerationID gen);
      virtual Event get_termination_event(void) const;
    private:
      void check_privilege(void);
    private:
      Context parent_ctx;
      RegionRequirement requirement;
      UserEvent mapped_event;
      Event ready_event;
      InstanceRef physical_instance;
      UserEvent unmapped_event;
      int region_idx; // for tracking the region in the original task arguments that we're mapping
    private:
      std::set<GeneralizedOperation*> map_dependent_waiters;
      std::vector<InstanceRef> source_copy_instances;
      MappingTagID tag;
    };

    /////////////////////////////////////////////////////////////
    // Unmap Operation 
    /////////////////////////////////////////////////////////////
    /**
     * A class for defferring the unmapping of regions.
     */
    class UnmapOperation : public DeferredOperation {
    public:
      UnmapOperation(HighLevelRuntime *rt);
      virtual ~UnmapOperation(void);
    public:
      void initialize(Context parent, const PhysicalRegion &reg);
    public:
      // Functions from GenerlizedOperation
      virtual bool activate(GeneralizedOperation *parent = NULL);
      virtual void deactivate(void);
    protected:
      virtual void perform_deferred(void);
    private:
      Context parent_ctx;
      PhysicalRegion region;
    };

    /////////////////////////////////////////////////////////////
    // Deletion Operation 
    /////////////////////////////////////////////////////////////
    /**
     * A class for deferrring deletions until all mapping tasks
     * that require the given resources have finished using them.
     * Deletions are performed in two steps: the first step frees
     * up the contexts in the calling task and all enclosing contexts;
     * the second step actually changes the structure of the region
     * tree to prune out branches that have been deleted.  The second
     * step can only occur after any tasks that may be using the deleted
     * resource in the calling context have terminated.
     *
     * Deferring both of these steps makes for an interesting state
     * machine for the deletion operation.  Deletions will be notified
     * at two points which may occur in arbitrary order:
     *
     *   a) when all mapping dependences are satisified, stage 1 can be done
     *   b) when all dependent tasks in the enclosing space have terminated,
     *      stage 1 and stage 2 can both be done
     *
     * We do stage 1 and stage 2 as soon as possible, but wait to reclaim
     * the deletion operation until all both calls have been performed.
     */
    class DeletionOperation : public GeneralizedOperation {
    public:
      DeletionOperation(HighLevelRuntime *rt);
      virtual ~DeletionOperation(void);
    public:
      void initialize_index_space_deletion(Context parent, IndexSpace space);
      void initialize_index_partition_deletion(Context parent, IndexPartition part);
      void initialize_field_space_deletion(Context parent, FieldSpace space);
      void initialize_field_deletion(Context parent, FieldSpace space, const std::set<FieldID> &to_free);
      void initialize_region_deletion(Context parent, LogicalRegion handle);
      void initialize_partition_deletion(Context parent, LogicalPartition handle);
    public:
      // Functions from GeneralizedOperation
      virtual bool activate(GeneralizedOperation *parent = NULL);
      virtual void deactivate(void);
      virtual void add_mapping_dependence(unsigned idx, const LogicalUser &prev, DependenceType dtype);
      virtual bool add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen);
      virtual bool perform_dependence_analysis(void);
      virtual bool perform_operation(void);
      virtual void trigger(void);
      virtual MapperID get_mapper_id(void) const;
      virtual bool has_mapped(GenerationID gen);
    public:
      Event get_map_event(void) const { return mapping_event; }
      Event get_termination_event(void) const { return termination_event; }
      void finalize(void);
    private:
      void launch_finalize(Event precondition);
      void perform_internal(bool finalize);
    private:
      enum DeletionKind {
        DESTROY_INDEX_SPACE,
        DESTROY_INDEX_PARTITION,
        DESTROY_FIELD_SPACE,
        DESTROY_FIELD,
        DESTROY_REGION,
        DESTROY_PARTITION,
      };
    private:
      Context parent_ctx;
      union {
        IndexSpace space;
        IndexPartition partition;
      } index;
      FieldSpace field_space; 
      DeletionKind handle_tag;
      std::set<FieldID> free_fields;
      LogicalRegion region;
      LogicalPartition partition;
      bool stage1_done;
      bool stage2_done;
      UserEvent mapping_event; // stage 1 done
      UserEvent termination_event; // stage 2 done
      std::set<Event> finalize_events;
    };

    /////////////////////////////////////////////////////////////
    // Creation Operation 
    /////////////////////////////////////////////////////////////
    /**
     * A class for deferring creation operations so that we don't
     * need to update the RegionTreeForest right away.  Note that
     * unlike deletions these operations aren't truly deferred
     * and never have any mapping dependences so we actually 
     * perform the operations when we normally would do dependence
     * analysis.
     */
    class CreationOperation : public DeferredOperation {
    public:
      CreationOperation(HighLevelRuntime *rt);
      virtual ~CreationOperation(void); 
    public:
      virtual bool activate(GeneralizedOperation *parent = NULL);
      virtual void deactivate(void);
    public:
      void initialize_index_space_creation(Context parent, Domain domain);
      void initialize_index_partition_creation(Context parent, IndexPartition pid,
            IndexSpace space, bool disjoint, int part_color, const std::map<Color,Domain> &new_spaces,
            Domain color_space = Domain::NO_DOMAIN);
      void initialize_field_space_creation(Context parent, FieldSpace space);
      void initialize_field_creation(Context parent, FieldSpace space, FieldID fid, size_t field_size);
      void initialize_field_creation(Context parent, FieldSpace space, const std::map<FieldID,size_t> &fids);
      void initialize_region_creation(Context parent, LogicalRegion handle);
    public:
      void initialize_get_logical_partition(Context parent, LogicalRegion reg_handle, IndexPartition index_handle);
      void initialize_get_logical_partition_by_color(Context parent, LogicalRegion handle, Color c);
      void initialize_get_logical_partition_by_tree(Context parent, IndexPartition handle, FieldSpace space, RegionTreeID tid);
      void initialize_get_logical_subregion(Context parent, LogicalPartition handle, IndexSpace index_handle);
      void initialize_get_logical_subregion_by_color(Context parent, LogicalPartition handle, Color c);
      void initialize_get_logical_subregion_by_tree(Context parent, IndexSpace handle, FieldSpace space, RegionTreeID tid);
    public:
      bool get_index_partition(Context ctx, IndexSpace parent, Color color, IndexPartition &result);
      bool get_index_subspace(Context ctx, IndexPartition parent, Color color, IndexSpace &result);
      bool get_index_space_domain(Context ctx, IndexSpace handle, Domain &result);
      bool get_index_partition_color_space(Context ctx, IndexPartition p, Domain &result);
      bool get_logical_partition_by_color(Context ctx, LogicalRegion parent, Color c, LogicalPartition &result);
      bool get_logical_subregion_by_color(Context ctx, LogicalPartition parent, Color c, LogicalRegion &result);
    protected:
      virtual void perform_deferred(void);
    private:
      enum CreationKind {
        CREATE_INDEX_SPACE,
        CREATE_INDEX_PARTITION,
        CREATE_FIELD_SPACE,
        CREATE_FIELD,
        CREATE_LOGICAL_REGION,
        TOUCH_LOGICAL_PARTITION,
        TOUCH_LOGICAL_PARTITION_BY_COLOR,
        TOUCH_LOGICAL_PARTITION_BY_TREE,
        TOUCH_LOGICAL_SUBREGION,
        TOUCH_LOGICAL_SUBREGION_BY_COLOR,
        TOUCH_LOGICAL_SUBREGION_BY_TREE,
        CREATION_KIND_NONE,
      };
    private:
      Context parent_ctx;
      CreationKind creation_kind;
      Domain domain;
      IndexSpace index_space;
      FieldSpace field_space;
      IndexPartition index_part;
      bool disjoint;
      int part_color;
      Color color;
      RegionTreeID tree_id;
      LogicalRegion new_region;
      LogicalPartition new_partition;
      std::map<Color,Domain> new_subspaces;
      std::map<FieldID,size_t> new_fields;
    };

    /////////////////////////////////////////////////////////////
    // Start Operation 
    /////////////////////////////////////////////////////////////
    /**
     * A class for deferring operations associated with
     * starting a task.
     */
    class StartOperation : public DeferredOperation {
    public:
      StartOperation(HighLevelRuntime *rt);
      virtual ~StartOperation(void);
    public:
      virtual bool activate(GeneralizedOperation *parent = NULL);
      virtual void deactivate(void);
      virtual void perform_deferred(void);
    public:
      void initialize(Context ctx);
    protected:
      Context task_ctx;
    };

    /////////////////////////////////////////////////////////////
    // Complete Operation 
    /////////////////////////////////////////////////////////////
    /**
     * A class for deferring operations associate with
     * completing a task.
     */
    class CompleteOperation : public DeferredOperation {
    public:
      CompleteOperation(HighLevelRuntime *rt);
      virtual ~CompleteOperation(void);
    public:
      virtual bool activate(GeneralizedOperation *parent = NULL);
      virtual void deactivate(void);
      virtual void perform_deferred(void);
    public:
      void initialize(Context ctx);
    protected:
      Context task_ctx;
    };

    /////////////////////////////////////////////////////////////
    // Task Context
    /////////////////////////////////////////////////////////////
    /**
     * A general class for representing all kinds of tasks
     */
    class TaskContext : public Task, public GeneralizedOperation {
    public:
      TaskContext(HighLevelRuntime *rt, ContextID id);
      virtual ~TaskContext(void);
    public:
      bool activate_task(GeneralizedOperation *parent);
      void deactivate_task(void);
    public:
      void initialize_task(Context parent, Processor::TaskFuncID tid,
                      void *args, size_t arglen, bool index_space,
                      const Predicate &predicate,
                      MapperID mid, MappingTagID tag);
      void set_requirements(const std::vector<IndexSpaceRequirement> &indexes,
                            const std::vector<FieldSpaceRequirement> &fields,
                            const std::vector<RegionRequirement> &regions, bool perform_checks);
    public:
      // Functions from GeneralizedOperation
      virtual void add_mapping_dependence(unsigned idx, const LogicalUser &prev, DependenceType dtype);
      virtual bool add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen) = 0;
      virtual bool activate(GeneralizedOperation *parent = NULL) = 0;
      virtual void deactivate(void) = 0;
      virtual bool perform_dependence_analysis(void);
      virtual bool perform_operation(void) = 0;
      virtual void trigger(void) = 0;
      virtual MapperID get_mapper_id(void) const;
      virtual bool has_mapped(GenerationID gen) = 0;
    public:
      virtual bool is_single(void) const = 0;
      virtual bool is_distributed(void) = 0;
      virtual bool is_locally_mapped(void) = 0;
      virtual bool is_stealable(void) = 0;
      virtual bool is_remote(void) = 0;
      virtual bool is_partially_unpacked(void) = 0;
    public:
      virtual bool distribute_task(void) = 0; // Return true if still local
      virtual bool perform_mapping(void) = 0; // Return if mapping was successful
      virtual void launch_task(void) = 0;
      virtual bool prepare_steal(void) = 0;
      virtual bool sanitize_region_forest(void) = 0;
      virtual void initialize_subtype_fields(void) = 0; 
      virtual Event get_map_event(void) const = 0;
      virtual Event get_termination_event(void) const = 0;
      virtual ContextID get_enclosing_physical_context(unsigned idx) = 0;
    public:
      virtual void remote_start(const void *args, size_t arglen) = 0;
      virtual void remote_children_mapped(const void *args, size_t arglen) = 0;
      virtual void remote_finish(const void *args, size_t arglen) = 0;
    public:
      virtual size_t compute_task_size(void) = 0;
      virtual void pack_task(Serializer &rez) = 0;
      virtual void unpack_task(Deserializer &derez) = 0;
      virtual void finish_task_unpack(void) = 0;
    public:
      // Functions from Task
      virtual UniqueID get_unique_task_id(void) const { return get_unique_id(); }
    public:
      inline bool is_leaf(void) const { return variants->leaf; }
    public:
      // For returning privileges (stored in create_* lists)
      void return_privileges(const std::set<IndexSpace> &new_indexes,
                             const std::set<FieldSpace> &new_fields,
                             const std::set<LogicalRegion> &new_regions,
                             const std::map<FieldID,FieldSpace> &new_field_ids);
      bool has_created_index_space(IndexSpace handle) const;
      bool has_created_field_space(FieldSpace handle) const;
      bool has_created_region(LogicalRegion handle) const;
      bool has_created_field(FieldID fid) const; 
    protected:
      size_t compute_task_context_size(void);
      void pack_task_context(Serializer &rez);
      void unpack_task_context(Deserializer &derez);
    protected:
      size_t compute_privileges_return_size(void);
      void pack_privileges_return(Serializer &rez);
      size_t unpack_privileges_return(Deserializer &derez); // return number of new regions
    protected:
      size_t compute_deletions_return_size(void);
      void pack_deletions_return(Serializer &rez);
      void unpack_deletions_return(Deserializer &derez);
    protected:
      bool invoke_mapper_locally_mapped(void);
      bool invoke_mapper_stealable(void);
      bool invoke_mapper_map_region_virtual(unsigned idx, Processor target);
      bool invoke_mapper_profile_task(Processor target);
      Processor invoke_mapper_select_target_proc(void);
      Processor::TaskFuncID invoke_mapper_select_variant(Processor target);
      void invoke_mapper_failed_mapping(unsigned idx, Processor target);
      void invoke_mapper_notify_profiling(Processor target, const Mapper::ExecutionProfile &profile);
    protected:
      void clone_task_context_from(TaskContext *rhs);
    protected:
      // Compute the enclosing contexts for this task
      void find_enclosing_contexts(std::vector<ContextID> &contexts);
    protected:
      const ContextID ctx_id;
      // Remember some fields are here already from the Task class
      Context parent_ctx;
      Predicate task_pred;
    protected:
      friend class SingleTask;
      friend class GeneralizedOperation;
      friend class MappingOperation;
      friend class DeletionOperation;
      // Keep track of created objects that we have privileges for
      std::set<IndexSpace>            created_index_spaces;
      std::set<FieldSpace>            created_field_spaces;
      std::set<LogicalRegion>         created_regions;
      std::map<FieldID,FieldSpace>    created_fields;
      // Keep track of deletions which have been performed
      std::vector<LogicalRegion>      deleted_regions; 
      std::vector<LogicalPartition>   deleted_partitions;
      std::map<FieldID,FieldSpace>    deleted_fields;
      // Which deleted regions and partitions need invalidations
      std::vector<LogicalRegion>      needed_region_invalidations;
      std::vector<LogicalPartition>   needed_partition_invalidations;
    protected:
      // Any other conditions needed for launching the task
      std::set<Event> launch_preconditions;
      // Additional conditions prior to this task being considered
      // mapped, primarily come from virtual walks for remote tasks
      std::set<Event> mapped_preconditions;
      // The waiters for each region to be mapped
      std::vector<std::set<GeneralizedOperation*> > map_dependent_waiters;
    };

    /////////////////////////////////////////////////////////////
    // Single Task 
    /////////////////////////////////////////////////////////////
    /**
     * A class for representing tasks which will only contain
     * a single point.  Serves as the interface for calling
     * contexts as well.
     */
    class SingleTask : public TaskContext {
    public:
      SingleTask(HighLevelRuntime *rt, ContextID id);
      virtual ~SingleTask(void);
    public:
      bool activate_single(GeneralizedOperation *parent);
      void deactivate_single(void);
    public:
      // Functions from GeneralizedOperation
      virtual bool add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen) = 0;
      virtual bool perform_operation(void);
      virtual void trigger(void) = 0;
      virtual bool activate(GeneralizedOperation *parent = NULL) = 0;
      virtual void deactivate(void) = 0;
      virtual bool has_mapped(GenerationID gen) = 0;
    public:
      // Functions from TaskContext
      virtual bool is_single(void) const { return true; }
      virtual bool is_distributed(void) = 0;
      virtual bool is_locally_mapped(void) = 0;
      virtual bool is_stealable(void) = 0;
      virtual bool is_remote(void) = 0;
      virtual bool is_partially_unpacked(void) = 0;
    public:
      // Functions from TaskContext
      virtual bool distribute_task(void) = 0; // Return true if still local
      virtual bool perform_mapping(void) = 0;
      virtual void launch_task(void);
      virtual bool prepare_steal(void);
      virtual bool sanitize_region_forest(void) = 0;
      virtual void initialize_subtype_fields(void) = 0;
      virtual Event get_map_event(void) const = 0;
      virtual Event get_termination_event(void) const = 0;
      virtual ContextID get_enclosing_physical_context(unsigned idx) = 0;
    public:
      // Functions from TaskContext
      virtual size_t compute_task_size(void) = 0;
      virtual void pack_task(Serializer &rez) = 0;
      virtual void unpack_task(Deserializer &derez) = 0;
      virtual void finish_task_unpack(void) = 0;
    public:
      ContextID find_enclosing_physical_context(LogicalRegion parent);
      ContextID find_outermost_physical_context(void) const;
    public:
      void register_child_task(TaskContext *child);
      void register_child_map(MappingOperation *op, int idx = -1);
      void register_child_deletion(DeletionOperation *op);
    public:
      void unregister_child_task(TaskContext *child);
      void unregister_child_map(MappingOperation *op);
      void unregister_child_deletion(DeletionOperation *op);
    public:
      // Operations on index space trees
      void create_index_space(Domain domain);
      void destroy_index_space(IndexSpace space, bool finalize); 
      Color create_index_partition(IndexPartition pid, IndexSpace parent, bool disjoint, int color,
                                  const std::map<Color,Domain> &coloring, Domain color_space = Domain::NO_DOMAIN);
      void destroy_index_partition(IndexPartition pid, bool finalize);
      IndexPartition get_index_partition(IndexSpace parent, Color color, bool can_create);
      IndexSpace get_index_subspace(IndexPartition p, Color color, bool can_create);
      Domain get_index_space_domain(IndexSpace handle, bool can_create);
      Domain get_index_partition_color_space(IndexPartition p, bool can_create);
    public:
      // Operations on field spaces
      void create_field_space(FieldSpace space);
      void destroy_field_space(FieldSpace space, bool finalize);
      void allocate_fields(FieldSpace space, const std::map<FieldID,size_t> &field_allocations);
      void free_fields(FieldSpace space, const std::set<FieldID> &to_free);
    public:
      // Operations on region trees
      void create_region(LogicalRegion handle);  
      void destroy_region(LogicalRegion handle, bool finalize);
      void destroy_partition(LogicalPartition handle, bool finalize);
      LogicalPartition get_region_partition(LogicalRegion parent, IndexPartition handle);
      LogicalRegion get_partition_subregion(LogicalPartition parent, IndexSpace handle);
      LogicalPartition get_region_subcolor(LogicalRegion parent, Color c, bool can_create);
      LogicalRegion get_partition_subcolor(LogicalPartition parent, Color c, bool can_create);
      LogicalPartition get_partition_subtree(IndexPartition handle, FieldSpace space, RegionTreeID tid);
      LogicalRegion get_region_subtree(IndexSpace handle, FieldSpace space, RegionTreeID tid);
    public:
      void unmap_physical_region(PhysicalRegion region);
    public:
      IndexSpace get_index_space(LogicalRegion handle);
      FieldSpace get_field_space(LogicalRegion handle);
    public:
      // Methods for checking privileges
      LegionErrorType check_privilege(const IndexSpaceRequirement &req) const;
      LegionErrorType check_privilege(const FieldSpaceRequirement &req) const;
      LegionErrorType check_privilege(const RegionRequirement &req, FieldID &bad_field) const;
    public:
      void start_task(std::vector<PhysicalRegion> &physical_regions);
      inline bool needs_pre_start(void) { return !is_remote(); }
      void pre_start(void);
    public:
      void complete_task(const void *result, size_t result_size, std::vector<PhysicalRegion> &physical_regions, bool owned);
      void post_complete_task(void);
      virtual const void* get_local_args(DomainPoint &point, size_t &local_size) = 0;
      virtual void handle_future(const void *result, size_t result_size, Event ready_event, bool owner) = 0;
    public:
      virtual void children_mapped(void) = 0;
      virtual void finish_task(void) = 0;
      virtual void remote_start(const void *args, size_t arglen) = 0;
      virtual void remote_children_mapped(const void *args, size_t arglen) = 0;
      virtual void remote_finish(const void *args, size_t arglen) = 0;
    public:
      const RegionRequirement& get_region_requirement(unsigned idx);
      Processor get_executing_processor(void) const { return executing_processor; }
    public:
      void register_deletion(LogicalRegion handle);
      void register_deletion(LogicalPartition handle);
      void register_deletion(const std::map<FieldID,FieldSpace> &fields);
      void return_deletions(const std::vector<LogicalRegion> &handles);
      void return_deletions(const std::vector<LogicalPartition> &handles);
      void return_deletions(const std::map<FieldID,FieldSpace> &fields);
      void invalidate_matches(LogicalRegion handle, const std::map<FieldID,FieldSpace> &fields);
    public:
      // These five functions handle the job of returning state for fields that
      // are created in region trees that existed before the start of the task.
      // They have to merge the state back into the enclosing task's context.
      void return_created_field_contexts(SingleTask *enclosing);
      void return_field_context(LogicalRegion handle, ContextID inner_ctx, const std::vector<FieldID> &fields);
      size_t compute_return_created_contexts(void);
      void pack_return_created_contexts(Serializer &rez);
      void unpack_return_created_contexts(Deserializer &derez);
    public:
      size_t compute_source_copy_instances_return(void);
      void pack_source_copy_instances_return(Serializer &derez);
      static void unpack_source_copy_instances_return(Deserializer &derez, RegionTreeForest *forest, UniqueID uid);
    public:
      size_t compute_reference_return(void);
      void pack_reference_return(Serializer &rez);
      static void unpack_reference_return(Deserializer &derez, RegionTreeForest *forest, UniqueID uid);
    protected:
      size_t compute_single_task_size(void);
      void pack_single_task(Serializer &rez);
      void unpack_single_task(Deserializer &derez);
    protected:
      bool map_all_regions(Processor target, Event single_term, Event multi_term);
      virtual InstanceRef find_premapped_region(unsigned idx) = 0;
      void initialize_region_tree_contexts(void);
    protected:
      void release_source_copy_instances(void);
      void issue_restoring_copies(std::set<Event> &wait_on_events, Event single, Event multi);
      void invalidate_owned_contexts(void);
    protected:
      // The processor on which this task is executing
      Processor executing_processor;
      Processor::TaskFuncID low_id; 
      unsigned unmapped; // number of regions still unmapped
      bool notify_runtime;
      std::vector<bool> non_virtual_mapped_region;
      // This vector is filled in by perform_operation which does the mapping
      std::vector<InstanceRef> physical_instances;
      // This vector contains references to clone references in the task's context
      std::vector<std::pair<InstanceRef,bool/*still hold reference*/> > clone_instances;
      // A vector for capturing the copies required to launch the task
      std::vector<InstanceRef> source_copy_instances;
      // A vector for capturing the close copies required to finish the task
      std::vector<InstanceRef> close_copy_instances;
      // This vector describes the physical ContextID for each region's mapping
      std::vector<ContextID> physical_contexts;
      // This vector just stores the physical region implementations for the task's duration
      std::vector<PhysicalRegionImpl*> physical_region_impls;
      // The set of child task's created when running this task
      std::set<TaskContext*> child_tasks;
      std::set<MappingOperation*> child_maps;
      std::set<DeletionOperation*> child_deletions;
      // The lock for managing the above queues
#ifdef LOW_LEVEL_LOCKS
      Lock child_lock;
#else
      ImmovableLock child_lock;
#endif
      // For packing up return created fields
      std::map<unsigned,std::vector<FieldID> > need_pack_created_fields;
      // For doing profiling of a task's execution
      bool profile_task;
      Mapper::ExecutionProfile exec_profile;
    };

    /////////////////////////////////////////////////////////////
    // Multi Task 
    /////////////////////////////////////////////////////////////
    /**
     * Abstract class for representing all tasks which contain
     * multiple tasks.
     */
    class MultiTask : public TaskContext {
    public:
      MultiTask(HighLevelRuntime *rt, ContextID id);
      virtual ~MultiTask(void);
    public:
      bool activate_multi(GeneralizedOperation *parent);
      void deactivate_multi(void);
    public:
      // Functions from GeneralizedOperation
      virtual bool perform_operation(void);
      virtual void trigger(void) = 0;
      virtual bool activate(GeneralizedOperation *parent = NULL) = 0;
      virtual void deactivate(void) = 0;
      virtual bool add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen) = 0;
      virtual bool has_mapped(GenerationID gen) = 0;
    public:
      // Functions from TaskContext
      virtual bool is_single(void) const { return false; }
      virtual bool is_distributed(void) = 0;
      virtual bool is_locally_mapped(void) = 0;
      virtual bool is_stealable(void) = 0;
      virtual bool is_remote(void) = 0;
      virtual bool is_partially_unpacked(void) = 0;
    public:
      // Functions from TaskContext
      virtual bool distribute_task(void) = 0; // Return true if still local
      virtual bool perform_mapping(void) = 0;
      virtual void launch_task(void) = 0;
      virtual bool prepare_steal(void) = 0;
      virtual bool sanitize_region_forest(void) = 0;
      virtual void initialize_subtype_fields(void) = 0;
      virtual Event get_map_event(void) const = 0;
      virtual Event get_termination_event(void) const = 0;
      virtual ContextID get_enclosing_physical_context(unsigned idx) = 0;
    public:
      // Functions from TaskContext
      virtual size_t compute_task_size(void) = 0;
      virtual void pack_task(Serializer &rez) = 0;
      virtual void unpack_task(Deserializer &derez) = 0;
      virtual void finish_task_unpack(void) = 0;
    public:
      // We have a separate operation for fusing map and launch for
      // multi-tasks so when we enumerate the index space we can map
      // and launch one task, and then go onto the next point.  This
      // allows us to overlap mapping on the utility processor with
      // tasks running on the computation processor.
      virtual bool map_and_launch(void) = 0;
    public:
      virtual void remote_start(const void *args, size_t arglen) = 0;
      virtual void remote_children_mapped(const void *args, size_t arglen) = 0;
      virtual void remote_finish(const void *args, size_t arglen) = 0; 
    public:
      void return_deletions(const std::vector<LogicalRegion> &region_handles,
                            const std::vector<LogicalPartition> &partition_handles,
                            const std::map<FieldID,FieldSpace> &fields);
    protected:
      // New functions for slicing that need to be done for multi-tasks
      bool is_sliced(void);
      bool slice_index_space(void);
      virtual bool pre_slice(void) = 0;
      virtual bool post_slice(void) = 0; // What to do after slicing
      virtual SliceTask *clone_as_slice_task(Domain new_domain, Processor target_proc, 
                                             bool recurse, bool stealable) = 0;
      virtual void handle_future(const AnyPoint &point, const void *result, size_t result_size, 
                                 Event ready_event, bool owner) = 0;
      void clone_multi_from(MultiTask *rhs, Domain new_domain, bool recurse);
    protected:
      size_t compute_multi_task_size(void);
      void pack_multi_task(Serializer &derez);
      void unpack_multi_task(Deserializer &derez);
    protected:
      friend class PointTask;
      // index_space from Task
      bool sliced;
      // The slices made of this task
      std::list<SliceTask*> slices;
      // For knowing whether we are doing reductions are keeping all futures
      bool has_reduction;
      ReductionOpID redop_id;
      void *reduction_state;
      size_t reduction_state_size;
      Barrier must_barrier; // for use with must parallelism
      // Argument Map for index space arguments
      ArgumentMapImpl *arg_map_impl;
      // Pre-mapped regions for index space tasks
      std::map<unsigned/*idx*/,InstanceRef> premapped_regions;
    };

    /////////////////////////////////////////////////////////////
    // Individual Task 
    /////////////////////////////////////////////////////////////
    /**
     * A class for representing single task launches.
     */
    class IndividualTask : public SingleTask {
    public:
      friend class HighLevelRuntime;
      IndividualTask(HighLevelRuntime *rt, ContextID id);
      virtual ~IndividualTask(void);
    public:
      // Functions from GeneralizedOperation
      virtual void trigger(void);
      virtual bool activate(GeneralizedOperation *parent = NULL);
      virtual void deactivate(void);
      virtual bool add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen);
      virtual bool has_mapped(GenerationID gen);
    public:
      // Functions from TaskContext
      virtual bool is_distributed(void);
      virtual bool is_locally_mapped(void);
      virtual bool is_stealable(void);
      virtual bool is_remote(void);
      virtual bool is_partially_unpacked(void);
    public:
      // Functions from TaskContext
      virtual bool distribute_task(void); // Return true if still local
      virtual bool perform_mapping(void);
      virtual bool sanitize_region_forest(void);
      virtual void initialize_subtype_fields(void);
      virtual Event get_map_event(void) const;
      virtual Event get_termination_event(void) const;
      virtual ContextID get_enclosing_physical_context(unsigned idx);
    public:
      // Functions from TaskContext
      virtual size_t compute_task_size(void);
      virtual void pack_task(Serializer &rez);
      virtual void unpack_task(Deserializer &derez);
      virtual void finish_task_unpack(void);
    public:
      // Functions from SingleTask
      virtual InstanceRef find_premapped_region(unsigned idx);
      virtual void children_mapped(void);
      virtual void finish_task(void);
      virtual void remote_start(const void *args, size_t arglen);
      virtual void remote_children_mapped(const void *args, size_t arglen);
      virtual void remote_finish(const void *args, size_t arglen);
      virtual const void* get_local_args(DomainPoint &point, size_t &local_size);
      virtual void handle_future(const void *result, size_t result_size, Event ready_event, bool owner);
    public:
      Future get_future(void);
    private:
      Processor current_proc;
      // Keep track of both whether the value has been set as well
      // as what its value is if it has
      bool distributed;
      bool locally_set;
      bool locally_mapped;
      bool stealable_set;
      bool stealable;
      bool remote;
      bool top_level_task;
      UserEvent mapped_event;
      UserEvent termination_event;
      FutureImpl *future;
    private:
      // For remote versions
      void *remote_future;
      size_t remote_future_len;
      // orig_proc from task
      Context orig_ctx;
      Event remote_start_event;
      Event remote_mapped_event;
      bool partially_unpacked;
      void *remaining_buffer;
      size_t remaining_bytes;
    };

    /////////////////////////////////////////////////////////////
    // Point Task 
    /////////////////////////////////////////////////////////////
    /**
     * A class for representing single tasks that are part
     * of a large index space of tasks.
     */
    class PointTask : public SingleTask {
    public:
      friend class SliceTask;
      PointTask(HighLevelRuntime *rt, ContextID id); 
      virtual ~PointTask(void);
    public:
      // Functions from GeneralizedOperation
      virtual void trigger(void);
      virtual bool activate(GeneralizedOperation *parent = NULL);
      virtual void deactivate(void);
      virtual bool add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen);
      virtual bool has_mapped(GenerationID gen);
    public:
      // Functions from TaskContext
      virtual bool is_distributed(void);
      virtual bool is_locally_mapped(void);
      virtual bool is_stealable(void);
      virtual bool is_remote(void);
      virtual bool is_partially_unpacked(void);
    public:
      // Functions from TaskContext
      virtual bool distribute_task(void); // Return true if still local
      virtual bool perform_mapping(void);
      virtual bool sanitize_region_forest(void);
      virtual void initialize_subtype_fields(void);
      virtual Event get_map_event(void) const;
      virtual Event get_termination_event(void) const;
      virtual ContextID get_enclosing_physical_context(unsigned idx);
    public:
      // Functions from TaskContext
      virtual size_t compute_task_size(void);
      virtual void pack_task(Serializer &rez);
      virtual void unpack_task(Deserializer &derez);
      virtual void finish_task_unpack(void);
    public:
      // Functions from SingleTask
      virtual InstanceRef find_premapped_region(unsigned idx);
      virtual void children_mapped(void);
      virtual void finish_task(void);
      virtual void remote_start(const void *args, size_t arglen);
      virtual void remote_children_mapped(const void *args, size_t arglen);
      virtual void remote_finish(const void *args, size_t arglen);
      virtual const void* get_local_args(DomainPoint &point, size_t &local_size);
      virtual void handle_future(const void *result, size_t result_size, Event ready_event, bool owner);
    public:
      void unmap_all_regions(void);
      void update_requirements(const std::vector<RegionRequirement> &reqs);
      void update_argument(const ArgumentMapImpl *impl);
    private:
      SliceTask *slice_owner;
      UserEvent point_termination_event;
      // The local argument for this particular point
      void *local_point_argument;
      size_t local_point_argument_len;
    };

    /////////////////////////////////////////////////////////////
    // Index Task 
    /////////////////////////////////////////////////////////////
    /**
     * A multi-task that is the top-level task object for
     * all index space launches.
     */
    class IndexTask : public MultiTask {
    public:
      IndexTask(HighLevelRuntime *rt, ContextID id); 
      virtual ~IndexTask(void);
    public:
      // Functions from GeneralizedOperation
      virtual void trigger(void);
      virtual bool activate(GeneralizedOperation *parent = NULL);
      virtual void deactivate(void);
      virtual bool add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen);
      virtual bool has_mapped(GenerationID gen);
    public:
      // Functions from TaskContext
      virtual bool is_distributed(void);
      virtual bool is_locally_mapped(void);
      virtual bool is_stealable(void);
      virtual bool is_remote(void);
      virtual bool is_partially_unpacked(void);
    public:
      // Functions from TaskContext
      virtual bool distribute_task(void); // Return true if still local
      virtual bool perform_mapping(void);
      virtual void launch_task(void);
      virtual bool prepare_steal(void);
      virtual bool sanitize_region_forest(void);
      virtual void initialize_subtype_fields(void);
      virtual Event get_map_event(void) const;
      virtual Event get_termination_event(void) const;
      virtual ContextID get_enclosing_physical_context(unsigned idx);
    public:
      // Functions from TaskContext
      virtual size_t compute_task_size(void);
      virtual void pack_task(Serializer &rez);
      virtual void unpack_task(Deserializer &derez);
      virtual void finish_task_unpack(void);
    public:
      virtual void remote_start(const void *args, size_t arglen);
      virtual void remote_children_mapped(const void *args, size_t arglen);
      virtual void remote_finish(const void *args, size_t arglen);
    public:
      // Function from MultiTask
      virtual bool map_and_launch(void);
      virtual SliceTask *clone_as_slice_task(Domain new_domain, Processor target_proc, 
                                             bool recurse, bool stealable);
      virtual bool pre_slice(void);
      virtual bool post_slice(void);
      virtual void handle_future(const AnyPoint &point, const void *result, size_t result_size, Event ready_event, bool owner);
    public:
      void set_index_domain(Domain space, const ArgumentMap &map, size_t num_regions, bool must);
      void set_reduction_args(ReductionOpID redop, const TaskArgument &initial_value);
      Future get_future(void);
      FutureMap get_future_map(void);
    public:
      // Functions called from slices at different points during execution
      void slice_start(unsigned long denominator, size_t points, const std::vector<unsigned> &non_virtual_mapped);
      void slice_mapped(const std::vector<unsigned> &virtual_mapped);
      void slice_finished(size_t points);
    public:
#ifdef DEBUG_HIGH_LEVEL
      void check_overlapping_slices(unsigned idx, const std::set<LogicalRegion> &used_regions);
#endif
      // This function pairs with pack_tree_state_return in SliceTask
      void unpack_tree_state_return(unsigned idx, ContextID ctx, RegionTreeForest::SendingMode mode, Deserializer &derez);
    private:
      friend class SliceTask;
      bool locally_set;
      bool locally_mapped; 
      UserEvent mapped_event;
      UserEvent termination_event;
      std::pair<unsigned long,unsigned long> frac_index_space;
      size_t num_total_points;
      size_t num_finished_points;
      // Keep track of the number of points that have mapped this index space
      std::vector<unsigned> mapped_points;
      unsigned unmapped; // number of unmapped regions
      FutureMapImpl *future_map;
      FutureImpl *reduction_future;
      // Vector for tracking source copy instances when performing sanitization
      std::vector<InstanceRef> source_copy_instances;
      // Map for checking for overlapping slices on writes
#ifdef DEBUG_HIGH_LEVEL
      std::map<unsigned/*idx*/,std::set<LogicalRegion> > slice_overlap;
#endif
    };

    /////////////////////////////////////////////////////////////
    // Slice Task 
    /////////////////////////////////////////////////////////////
    /**
     * A task for representing slices of index spaces created by
     * the user.  Slices enumerated their slice of the index
     * space into single Point Tasks.
     */
    class SliceTask : public MultiTask {
    protected:
      struct FutureResult {
      public:
        FutureResult(void)
          : buffer(NULL), buffer_size(0), ready_event(Event::NO_EVENT) { }
        FutureResult(void *b, size_t s, Event r)
          : buffer(b), buffer_size(s), ready_event(r) { }
      public:
        void *buffer;
        size_t buffer_size;
        Event ready_event;
      };
    public:
      SliceTask(HighLevelRuntime *rt, ContextID id);
      virtual ~SliceTask(void);
    public:
      // Functions from GeneralizedOperation
      virtual void trigger(void);
      virtual bool activate(GeneralizedOperation *parent = NULL);
      virtual void deactivate(void);
      virtual bool add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen);
      virtual bool has_mapped(GenerationID gen);
    public:
      // Functions from TaskContext
      virtual bool is_distributed(void);
      virtual bool is_locally_mapped(void);
      virtual bool is_stealable(void);
      virtual bool is_remote(void);
      virtual bool is_partially_unpacked(void);
    public:
      // Functions from TaskContext
      virtual bool distribute_task(void); // Return true if still local
      virtual bool perform_mapping(void);
      virtual void launch_task(void);
      virtual bool prepare_steal(void);
      virtual bool sanitize_region_forest(void);
      virtual void initialize_subtype_fields(void);
      virtual Event get_map_event(void) const;
      virtual Event get_termination_event(void) const;
      virtual ContextID get_enclosing_physical_context(unsigned idx);
    public:
      // Functions from TaskContext
      virtual size_t compute_task_size(void);
      virtual void pack_task(Serializer &rez);
      virtual void unpack_task(Deserializer &derez);
      virtual void finish_task_unpack(void);
    public:
      virtual void remote_start(const void *args, size_t arglen);
      virtual void remote_children_mapped(const void *args, size_t arglen);
      virtual void remote_finish(const void *args, size_t arglen);
    public:
      // Functions from MultiTask
      virtual bool map_and_launch(void);
      virtual SliceTask *clone_as_slice_task(Domain new_domain, Processor target_proc, 
                                             bool recurse, bool stealable);
      virtual bool pre_slice(void);
      virtual bool post_slice(void);
      virtual void handle_future(const AnyPoint &point, const void *result, size_t result_size, 
                                 Event ready_event, bool owner);
    protected:
      PointTask* clone_as_point_task(bool new_point);
    public:
      void set_denominator(unsigned long denom, unsigned long split);
      void point_task_mapped(PointTask *point);
      void point_task_finished(PointTask *point);
    private:
      // Methods to be run once all the slice's points have finished a phase
      void post_slice_start(void);
      void post_slice_mapped(void);
      void post_slice_finished(void);
    private:
      size_t compute_state_return_size(unsigned idx, ContextID ctx, RegionTreeForest::SendingMode mode);
      void pack_tree_state_return(unsigned idx, ContextID ctx, RegionTreeForest::SendingMode mode, Serializer &rez);
    protected:
      friend class PointTask;
      friend class IndexTask;
      // The following set of fields are set when a slice is cloned
      bool distributed; 
      bool locally_mapped;
      bool stealable;
      bool remote;
      Event termination_event;
      Processor current_proc;
      std::vector<PointTask*> points;
      // For remote slices
      // orig_proc from Task
      IndexTask *index_owner;
      Event remote_start_event;
      Event remote_mapped_event;
      bool partially_unpacked;
      void *remaining_buffer;
      size_t remaining_bytes;
      // For storing futures when remote, the slice owns the result values
      // but the AnyPoint buffers are owned by the points themselves which
      // we know are live throughout the life of the SliceTask.
      std::map<AnyPoint,FutureResult> future_results;
      // (1/denominator indicates fraction of index space in this slice)
      unsigned long denominator; // Set explicitly, no need to copy
      // The split factor is the multiple difference between the current denominator
      // and the denominator when the region tree is finally unpacked.  It allows
      // the InstanceManagers to rescale themselves to know exactly what fraction
      // they actually represent after being unpacked since the index space
      // of tasks may have been split up many times since they were packed.
      unsigned long split_factor; // Set explicitly, no need to copy
      bool enumerating; // Set to true when we're enumerating the slice
      //LowLevel::ElementMask::Enumerator *enumerator;
      //int remaining_enumerated;
      Domain::DomainPointIterator *domain_iterator;
      unsigned num_unmapped_points;
      unsigned num_unfinished_points;
      // Keep track of the number of non-virtual mappings for point tasks
      std::vector<unsigned> non_virtual_mappings;
    };

  }; // namespace HighLevel
}; // namespace LegionRuntime

#endif // __LEGION_OPS_H__
