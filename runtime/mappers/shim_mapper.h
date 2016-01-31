/* Copyright 2016 Stanford University, NVIDIA Corporation
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


#ifndef __SHIM_MAPPER_H__
#define __SHIM_MAPPER_H__

#include "legion.h"
#include "mapping_utilities.h"
#include "default_mapper.h"
#include <cstdlib>
#include <cassert>
#include <algorithm>

namespace Legion {
  namespace Mapping {

    /**
     * \struct RegionRequirement
     * Region requirements are the objects used to name the logical regions
     * that are used by tasks, copies, and inline mapping operations.  Region
     * requirements can name either logical regions or logical partitions in
     * for index space launches.  In addition to placing logical upper bounds
     * on the privileges required for an operation, region requirements also
     * specify the privileges and coherence modes associated with the needed
     * logical region/partition.  Region requirements have a series of
     * constructors for different scenarios.  All fields in region requirements
     * are publicly visible so applications can mutate them freely including
     * configuring region requirements in ways not supported with the default
     * set of constructors.
     */
    struct RegionRequirement {
    public: 
      RegionRequirement(void);
      /**
       * Standard region requirement constructor for logical region
       */
      RegionRequirement(LogicalRegion _handle,
                        const std::set<FieldID> &privilege_fields,
                        const std::vector<FieldID> &instance_fields,
                        PrivilegeMode _priv, CoherenceProperty _prop, 
                        LogicalRegion _parent, MappingTagID _tag = 0, 
                        bool _verified = false);
      /**
       * Partition region requirement with projection function
       */
      RegionRequirement(LogicalPartition pid, ProjectionID _proj,
                        const std::set<FieldID> &privilege_fields,
                        const std::vector<FieldID> &instance_fields,
                        PrivilegeMode _priv, CoherenceProperty _prop,
                        LogicalRegion _parent, MappingTagID _tag = 0, 
                        bool _verified = false);
      /**
       * Region requirement with projection function
       */
      RegionRequirement(LogicalRegion _handle, ProjectionID _proj,
                        const std::set<FieldID> &privilege_fields,
                        const std::vector<FieldID> &instance_fields,
                        PrivilegeMode _priv, CoherenceProperty _prop,
                        LogicalRegion _parent, MappingTagID _tag = 0,
                        bool _verified = false);
      /**
       * Standard reduction region requirement.  Note no privilege
       * is passed, but instead a reduction operation ID is specified.
       */
      RegionRequirement(LogicalRegion _handle,
                        const std::set<FieldID> &privilege_fields,
                        const std::vector<FieldID> &instance_fields,
                        ReductionOpID op, CoherenceProperty _prop, 
                        LogicalRegion _parent, MappingTagID _tag = 0, 
                        bool _verified = false);
      /**
       * Partition region requirement for reduction.
       */
      RegionRequirement(LogicalPartition pid, ProjectionID _proj, 
                        const std::set<FieldID> &privilege_fields,
                        const std::vector<FieldID> &instance_fields,
                        ReductionOpID op, CoherenceProperty _prop,
                        LogicalRegion _parent, MappingTagID _tag = 0, 
                        bool _verified = false);
      /**
       * Projection logical region requirement for reduction
       */
      RegionRequirement(LogicalRegion _handle, ProjectionID _proj,
                        const std::set<FieldID> &privilege_fields,
                        const std::vector<FieldID> &instance_fields,
                        ReductionOpID op, CoherenceProperty _prop, 
                        LogicalRegion _parent, MappingTagID _tag = 0, 
                        bool _verified = false);
    public:
      // Analogous constructors without the privilege and instance fields
      RegionRequirement(LogicalRegion _handle, PrivilegeMode _priv, 
                        CoherenceProperty _prop, LogicalRegion _parent,
			MappingTagID _tag = 0, bool _verified = false);
      RegionRequirement(LogicalPartition pid, ProjectionID _proj,
                        PrivilegeMode _priv, CoherenceProperty _prop,
                        LogicalRegion _parent, MappingTagID _tag = 0, 
                        bool _verified = false);
      RegionRequirement(LogicalRegion _handle, ProjectionID _proj,
                        PrivilegeMode _priv, CoherenceProperty _prop, 
                        LogicalRegion _parent, MappingTagID _tag = 0, 
                        bool _verified = false);
      RegionRequirement(LogicalRegion _handle, ReductionOpID op, 
                        CoherenceProperty _prop, LogicalRegion _parent,
			MappingTagID _tag = 0, bool _verified = false);
      RegionRequirement(LogicalPartition pid, ProjectionID _proj, 
                        ReductionOpID op, CoherenceProperty _prop,
                        LogicalRegion _parent, MappingTagID _tag = 0, 
                        bool _verified = false);
      RegionRequirement(LogicalRegion _handle, ProjectionID _proj,
                        ReductionOpID op, CoherenceProperty _prop, 
                        LogicalRegion _parent, MappingTagID _tag = 0, 
                        bool _verified = false);
    public:
      bool operator==(const RegionRequirement &req) const;
      bool operator<(const RegionRequirement &req) const;
    public:
      /**
       * Method for adding a field to region requirements
       * @param fid field ID to add
       * @param instance indicate whether to add to instance fields
       */
      inline RegionRequirement& add_field(FieldID fid, bool instance = true);
      inline RegionRequirement& add_fields(const std::vector<FieldID>& fids, 
                                           bool instance = true);

      inline RegionRequirement& add_flags(RegionFlags new_flags);
    public:
#ifdef PRIVILEGE_CHECKS
      unsigned get_accessor_privilege(void) const;
#endif
      bool has_field_privilege(FieldID fid) const;
      void copy_without_mapping_info(const RegionRequirement &rhs);
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      void initialize_mapping_fields(void);
    public:
      // Fields used for controlling task launches
      LogicalRegion region; /**< mutually exclusive with partition*/
      LogicalPartition partition; /**< mutually exclusive with region*/
      std::set<FieldID> privilege_fields; /**< unique set of privilege fields*/
      std::vector<FieldID> instance_fields; /**< physical instance fields*/
      PrivilegeMode privilege; /**< region privilege mode*/
      CoherenceProperty prop; /**< region coherence mode*/
      LogicalRegion parent; /**< parent region to derive privileges from*/
      ReductionOpID redop; /**<reduction operation (default 0)*/
      MappingTagID tag; /**< mapping tag for this region requirement*/
      RegionFlags flags; /**< optional flags set for region requirements*/
      HandleType handle_type; /**< region or partition requirement*/
      ProjectionID projection; /**< projection function for index space tasks*/
    public:
      // These are fields that are set by the runtime as part
      // of mapping calls that are passed to the mapper.  See
      // the Mapper interface of how these fields are used. They
      // are only valid if the premapped flag is true.
      bool premapped;
      bool must_early_map;
      bool restricted;
      size_t max_blocking_factor;
      std::map<Memory,bool> current_instances;
    public:
      // These are fields that a Mapper class can set as part
      // of mapping calls for controlling the mapping of a task
      // containing a given region requirement.  See the Mapper
      // interface for how these fields are used.
      bool virtual_map;
      bool early_map;
      bool enable_WAR_optimization;
      bool reduction_list;
      bool make_persistent;
      size_t blocking_factor;
      // TODO: hardness factor
      std::vector<Memory> target_ranking;
      std::set<FieldID> additional_fields;
    public:
      // These are fields set by the runtime to inform the
      // Mapper about the result of mapping decisions.
      bool mapping_failed;
      Memory selected_memory;
    };

    //==========================================================================
    //                            Mapping Classes
    //==========================================================================
    
    /**
     * \class Mapable 
     * The Mappable class serves as the abstract base class for
     * represeting operations such as tasks, copies, and inline mappings
     * which can be mapped.  In some cases the mapper will be
     * invoked with a general mappable operation and the mapper
     * can decide whether to specialize on the kind of operation
     * being performed or not.
     */
    class Mappable {
    public:
      enum MappableKind {
        TASK_MAPPABLE,
        COPY_MAPPABLE,
        INLINE_MAPPABLE,
        ACQUIRE_MAPPABLE,
        RELEASE_MAPPABLE,
      };
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      Mappable(void);
    public:
      MapperID                            map_id;
      MappingTagID                        tag;
    public:
      virtual MappableKind get_mappable_kind(void) const = 0;
      virtual Task* as_mappable_task(void) const = 0;
      virtual Copy* as_mappable_copy(void) const = 0;
      virtual Inline* as_mappable_inline(void) const = 0;
      virtual Acquire* as_mappable_acquire(void) const = 0;
      virtual Release* as_mappable_release(void) const = 0;
      virtual UniqueID get_unique_mappable_id(void) const = 0;
      virtual unsigned get_depth(void) const = 0;
    };

    /**
     * \class Task
     * Task objects provide an interface to the arguments to
     * a task as well as some of the meta-data available from
     * the runtime about the task.  Pointers to task objects
     * are used in two places: as an argument to task 
     * implementations, and as arguments to the mapper object
     * associated with a task.  In many cases the referenced
     * pointers are annotated const so that this data cannot
     * be corrupted by the application.
     */
    class Task : public Mappable {
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      Task(void);
    public:
      // Task argument information
      Processor::TaskFuncID               task_id; 
      std::vector<IndexSpaceRequirement>  indexes;
      std::vector<RegionRequirement>      regions;
      std::vector<Future>                 futures;
      std::vector<Grant>                  grants;
      std::vector<PhaseBarrier>           wait_barriers;
      std::vector<PhaseBarrier>           arrive_barriers;
      void                               *args;                         
      size_t                              arglen;
    public:
      // Index task argument information
      bool                                is_index_space;
      bool                                must_parallelism; 
      Domain                              index_domain;
      DomainPoint                         index_point;
      void                               *local_args;
      size_t                              local_arglen;
    public:
      // Meta data information from the runtime
      Processor                           orig_proc;
      Processor                           current_proc;
      unsigned                            steal_count;
      unsigned                            depth;  
      bool                                speculated;
      bool                                premapped;
      TaskVariantCollection              *variants;
    public:
      // Values set by the runtime for controlling
      // scheduling and variant selection
      VariantID                           selected_variant;
      bool                                schedule;
    public:
      // Task options that can be set by a mapper in
      // the set_task_options call.  See the Mapper 
      // interface for description of these fields.
      Processor                           target_proc;
      std::set<Processor>                 additional_procs;
      TaskPriority                        task_priority;
      bool                                inline_task;
      bool                                spawn_task;
      bool                                map_locally;
      bool                                profile_task;
      bool                                post_map_task;
    public:
      // Options for configuring this task's context
      int                                 max_window_size;
      unsigned                            hysteresis_percentage;
      int                                 max_outstanding_frames;
      unsigned                            min_tasks_to_schedule;
      unsigned                            min_frames_to_schedule;
      unsigned                            max_directory_size;
    public:
      // Profiling information for the task
      unsigned long long                  start_time;
      unsigned long long                  stop_time;
    public:
      inline UniqueID get_unique_task_id(void) const;
    public:
      virtual MappableKind get_mappable_kind(void) const = 0;
      virtual Task* as_mappable_task(void) const = 0;
      virtual Copy* as_mappable_copy(void) const = 0;
      virtual Inline* as_mappable_inline(void) const = 0;
      virtual Acquire* as_mappable_acquire(void) const = 0;
      virtual Release* as_mappable_release(void) const = 0;
      virtual UniqueID get_unique_mappable_id(void) const = 0;
      virtual const char* get_task_name(void) const = 0;
      virtual unsigned get_depth(void) const;
    };

    /**
     * \class Copy
     * Copy objects provide an interface to the arguments
     * from a copy operation call.  Copy objects are passed
     * as arguments to mapper calls that need to decide
     * how best to map a copy operation.
     */
    class Copy : public Mappable {
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      Copy(void);
    public:
      // Copy Launcher arguments
      std::vector<RegionRequirement>    src_requirements;
      std::vector<RegionRequirement>    dst_requirements;
      std::vector<Grant>                grants;
      std::vector<PhaseBarrier>         wait_barriers;
      std::vector<PhaseBarrier>         arrive_barriers;
    public:
      // Parent task for the copy operation
      Task                              *parent_task;
    public:
      inline UniqueID get_unique_copy_id(void) const;
    public:
      virtual MappableKind get_mappable_kind(void) const = 0;
      virtual Task* as_mappable_task(void) const = 0;
      virtual Copy* as_mappable_copy(void) const = 0;
      virtual Inline* as_mappable_inline(void) const = 0;
      virtual Acquire* as_mappable_acquire(void) const = 0;
      virtual Release* as_mappable_release(void) const = 0;
      virtual UniqueID get_unique_mappable_id(void) const = 0;
      virtual unsigned get_depth(void) const;
    };

    /**
     * \class Inline
     * Inline mapping objects present an interface to
     * the arguments from an inline mapping call.
     * Inline objects are passed to mapper calls to
     * decide how to best map the inline operation.
     */
    class Inline : public Mappable {
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      Inline(void);
    public:
      // Inline Launcher arguments
      RegionRequirement                 requirement;
    public:
      // Parent task for the inline operation
      Task                              *parent_task;
    public:
      inline UniqueID get_unique_inline_id(void) const;
    public:
      virtual MappableKind get_mappable_kind(void) const = 0;
      virtual Task* as_mappable_task(void) const = 0;
      virtual Copy* as_mappable_copy(void) const = 0;
      virtual Inline* as_mappable_inline(void) const = 0;
      virtual Acquire* as_mappable_acquire(void) const = 0;
      virtual Release* as_mappable_release(void) const = 0;
      virtual UniqueID get_unique_mappable_id(void) const = 0;
      virtual unsigned get_depth(void) const;
    };

    /**
     * \class Acquire
     * Acquire objects present an interface to the 
     * arguments from a user-level software coherence 
     * acquire call.  Acquire objects are passed to 
     * mapper calls that need to decide how to best 
     * map an acquire.
     */
    class Acquire : public Mappable {
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      Acquire(void);
    public:
      // Acquire Launcher arguments
      LogicalRegion                     logical_region;
      LogicalRegion                     parent_region;
      std::set<FieldID>                 fields;
      PhysicalRegion                    region;
      std::vector<Grant>                grants;
      std::vector<PhaseBarrier>         wait_barriers;
      std::vector<PhaseBarrier>         arrive_barriers;
    public:
      // Parent task for the acquire operation
      Task                              *parent_task;
    public:
      inline UniqueID get_unique_acquire_id(void) const;
    public:
      virtual MappableKind get_mappable_kind(void) const = 0;
      virtual Task* as_mappable_task(void) const = 0;
      virtual Copy* as_mappable_copy(void) const = 0;
      virtual Inline* as_mappable_inline(void) const = 0;
      virtual Acquire* as_mappable_acquire(void) const = 0;
      virtual Release* as_mappable_release(void) const = 0;
      virtual UniqueID get_unique_mappable_id(void) const = 0;
      virtual unsigned get_depth(void) const;
    };

    /**
     * \class Release
     * Release objects present an interface to the
     * arguments from a user-level software coherence
     * release call.  Release objects are passed to 
     * mapper calls that need to decide how best
     * to map a release.
     */
    class Release : public Mappable {
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      Release(void);
    public:
      // Release Launcher arguments
      LogicalRegion                     logical_region;
      LogicalRegion                     parent_region;
      std::set<FieldID>                 fields;
      PhysicalRegion                    region;
      std::vector<Grant>                grants;
      std::vector<PhaseBarrier>         wait_barriers;
      std::vector<PhaseBarrier>         arrive_barriers;
    public:
      // Parent task for the release operation
      Task                              *parent_task;
    public:
      inline UniqueID get_unique_release_id(void) const;
    public:
      virtual MappableKind get_mappable_kind(void) const = 0;
      virtual Task* as_mappable_task(void) const = 0;
      virtual Copy* as_mappable_copy(void) const = 0;
      virtual Inline* as_mappable_inline(void) const = 0;
      virtual Acquire* as_mappable_acquire(void) const = 0;
      virtual Release* as_mappable_release(void) const = 0;
      virtual UniqueID get_unique_mappable_id(void) const = 0;
      virtual unsigned get_depth(void) const;
    };

    /**
     * \class TaskVariantCollection
     * THIS IS NOT AN APPLICATION LEVEL OBJECT! Instead it
     * provides an interface for mapper objects to know what
     * kinds of variants have been registered with the runtime
     * of a given kind of task.  This allows the mapper to make
     * intelligent decisions about where best to send a task.
     */
    class TaskVariantCollection {
    public:
      class Variant {
      public:
        Processor::TaskFuncID low_id;
        Processor::Kind proc_kind;
        bool single_task; /**< supports single tasks*/
        bool index_space; /**< supports index tasks*/
        bool inner;
        bool leaf;
        VariantID vid;
      public:
        Variant(void)
          : low_id(0) { }
        Variant(Processor::TaskFuncID id, Processor::Kind k, 
                bool single, bool index, 
                bool in, bool lf,
                VariantID v)
          : low_id(id), proc_kind(k), 
            single_task(single), index_space(index), 
            inner(in), leaf(lf), vid(v) { }
      };
    protected:
      // Only the runtime should be able to make these
      FRIEND_ALL_RUNTIME_CLASSES
      TaskVariantCollection(Processor::TaskFuncID uid, 
                            const char *n,
                            const bool idem, size_t ret)
        : user_id(uid), name(n), 
          idempotent(idem), return_size(ret) { }
      void add_variant(Processor::TaskFuncID low_id, 
                       Processor::Kind kind, 
                       bool single, bool index,
                       bool inner, bool leaf,
                       VariantID vid);
      const Variant& select_variant(bool single, bool index, 
                                    Processor::Kind kind);
    public:
      /**
       * Check to see if a collection of variants has one
       * that meets the specific criteria.
       * @param kind the kind of processor to support
       * @param single whether the variants supports single tasks
       * @param index_space whether the variant supports index space launches
       * @return true if a variant exists, false otherwise
       */
      bool has_variant(Processor::Kind kind, 
                       bool single, bool index_space);
      /**
       * Return the variant ID for a variant that
       * meets the given qualifications
       * @param kind the kind of processor to support
       * @param single whether the variant supports single tasks
       * @param index_space whether the variant supports index space launches
       * @return the variant ID if one exists
       */
      VariantID get_variant(Processor::Kind kind, 
                            bool single, bool index_space);
      /**
       * Check to see if a collection has a variant with
       * the given ID.
       * @param vid the variant ID
       * @return true if the collection has a variant with the given ID
       */
      bool has_variant(VariantID vid);
      /**
       * Find the variant with a given ID.
       * @param vid the variant ID
       * @return a const reference to the variant if it exists
       */
      const Variant& get_variant(VariantID vid);

      const std::map<VariantID,Variant>& get_all_variants(void) const
        { return variants; }
    public:
      const Processor::TaskFuncID user_id;
      const char *name;
      const bool idempotent;
      const size_t return_size;
    protected:
      std::map<VariantID,Variant> variants;
    };

    /**
     * \interface Mapper
     * This is the interface definition that must be implemented
     * by all mapping objects.  It defines the set of queries that
     * the runtime will make in order to map an application on to
     * a particular architecture.  The mapping interface has no
     * impact on correctness.  Decisions made by any mapper will
     * only effect performance.  Legion provides a default mapper
     * with heuristics for performing mapping.  A good strategy
     * for tuning an application is to implement a mapper that
     * extends the default mapper and gradually override methods
     * needed for customizing an application's mapping for a
     * particular target architecture.
     */
    class Mapper {
    public:
      /**
       * \struct DomainSplit
       * A domain split object specifies a decomposition
       * of a domain so that different parts of the domain
       * can be mapped and run in parallel on different
       * processors.  The domain for a DomainSplit specifies
       * the sub-domain described by this DomainSplit.
       * Processor gives the target processor this domain
       * split should be sent to.  Recurse indicates whether
       * this sub-domain should be recursively sub-divided
       * by the mapper on the target processor.  Stealable
       * indicates whether this DomainSplit object is
       * eligible for stealing by other mappers.
       */
      struct DomainSplit {
      public:
        DomainSplit(Domain d, Processor p, 
                   bool rec, bool steal)
          : domain(d), proc(p), 
            recurse(rec), stealable(steal) { }
      public:
        Domain domain;
        Processor proc;
        bool recurse;
        bool stealable;
      };
    public:
      /**
       * \struct MappingConstraint
       * A mapping constraint object captures constraints on
       * two different operations which needs to be satisfied 
       * in order for the tasks to be executed in parallel.
       * We use these as part of the must parallelism call.
       */
      struct MappingConstraint {
      public:
        MappingConstraint(Task *one, unsigned id1,
                          Task *two, unsigned id2,
                          DependenceType d)
          : t1(one), idx1(id1), t2(two), idx2(id2), dtype(d) { }
      public:
        Task *t1;
        unsigned idx1;
        Task *t2;
        unsigned idx2;
        DependenceType dtype;
      };
    public:
      Mapper(Runtime *rt) 
        : runtime(rt) { }
      virtual ~Mapper(void) { }
    protected:
      Runtime *const runtime;
    public:
      /**
       * ----------------------------------------------------------------------
       *  Select Task Options
       * ----------------------------------------------------------------------
       * This mapper call happens immediately after the task is launched
       * and before any other operations are performed.  The mapper
       * then has the option of mutating the following fields on
       * the task object from any of their default values. 
       *
       * target_proc = local processor
       * inline_task = false
       * spawn_task  = false 
       * profile_task= false
       *
       * target_proc - this only applies to single task launches
       *               and allows the mapper to specify the target
       *               processor where the task should be sent
       *               prior to having any operations performed.
       *               Note that if stealing is disabled then this is
       *               the processor on which the task will be executed
       *               unless it is redirected due to a processor failure.
       *               For index space tasks, they will be broken
       *               up using the slice_domain mapper call.
       * inline_task - Specify whether this task should be inlined
       *               locally using the parent task's mapped regions.
       *               If the regions are not already mapped they
       *               will be re-mapped, the task will be
       *               executed locally, and then the regions unmapped.
       *               If this option is selected, the mapper should also
       *               select the task variant to be used by setting
       *               the 'selected_variant' field.
       * spawn_task  - This field is inspired by Cilk and has equivalent
       *               semantics.  If a task is spawned it becomes eligible
       *               for stealing, otherwise it will traverse the mapping
       *               process without being stolen.  The one deviation from 
       *               Cilk stealing is that stealing in Legion is managed by
       *               mappers instead of the Legion runtime.
       * map_locally - Tasks have the option of either being mapped on 
       *               the processor on which they were created or being mapped
       *               on their ultimate destination processor.  Mapping on the
       *               local processor where the task was created can be
       *               more efficient in some cases since it requires less
       *               meta-data movement by the runtime, but can also be
       *               subject to having an incomplete view of the destination
       *               memories during the mapping process.  In general a task
       *               should only be mapped locally if it is a leaf task as
       *               the runtime will need to move the meta-data for a task
       *               anyway if it is going to launch sub-tasks.  Note that
       *               deciding to map a task locally disqualifies that task
       *               from being stolen as it will have already been mapped
       *               once it enters the ready queue.
       * profile_task- Decide whether profiling information should be collected
       *               for this task.  If set to true, then the mapper will
       *               be notified after the task has finished executing.
       */
      virtual void select_task_options(Task *task) = 0; 

      /**
       * ----------------------------------------------------------------------
       *  Select Tasks to Schedule 
       * ----------------------------------------------------------------------
       * Select which tasks should be scheduled onto the processor
       * managed by this mapper.  The mapper is given a list of tasks
       * that are ready to be mapped with their 'schedule' field set to false.  
       * The mapper can set 'schedule' to true to map the corresponding tasks
       * in the list.  This method gives the mapper the prerogative to
       * re-order how tasks are mapped to give tasks on an application's
       * critical path priority.  By leaving tasks on the list of tasks
       * ready to map, the mapper can make additional tasks available
       * for stealing longer.  It is acceptable for the mapper to 
       * indicate that no tasks should currently be mapped if it
       * determines that it wants to wait longer before mapping any
       * additional tasks.
       *
       * In addition to choosing the tasks to schedule, the mapper can
       * also change the value of the 'target_proc' field to another
       * processor.  If 'target_proc' is changed the task will be 
       * redistributed to the new target processor.  This gives the
       * mapper a push operation instead of having to wait for stealing
       * when many tasks end up on a single processor.
       *
       * Note that if both 'schedule' is set to true, and a new target
       * processor is specified, the runtime will first try to map 
       * the task on the current processor, and if that fails, it will
       * then send the task to the new target processor.
       * @param ready_tasks the list of tasks that are ready to map
       */
      virtual void select_tasks_to_schedule(
                      const std::list<Task*> &ready_tasks) = 0;

      /**
       * ----------------------------------------------------------------------
       *  Target Task Steal 
       * ----------------------------------------------------------------------
       * Select a target processor from which to attempt a task
       * steal.  The runtime provides a list of processors that have
       * had previous steal requests fail and are therefore
       * blacklisted.  Any attempts to send a steal request to a
       * blacklisted processor will not be performed.  Note the runtime
       * advertises when new works is available on a processor which
       * will then remove processors from the blacklist.
       * @param blacklist list of processors that are blacklisted
       * @param set of processors to target for stealing
       */
      virtual void target_task_steal(const std::set<Processor> &blacklist,
                                     std::set<Processor> &targets) = 0;

      /**
       * ----------------------------------------------------------------------
       *  Permit Task Steal 
       * ----------------------------------------------------------------------
       * Unlike Cilk where stealing occurred automatically, Legion
       * places stealing under control of the mapper because of 
       * the extremely high cost of moving data in distributed
       * memory architectures.  As a result, when a steal request
       * is received from a mapper on a remote node, the Legion
       * runtime asks the mapper at the current node whether it
       * wants to permit stealing of any of the tasks that it
       * currently is managing.  The mapper is told which processor
       * is attempting the steal as well as a lists of tasks that
       * are eligible to be stolen (i.e. were spawned).  The
       * mapper then returns a (potentially empty) set of tasks
       * that will be given to the thief processor.
       * @param thief the processor that send the steal request
       * @param tasks the list of tasks eligible for stealing
       * @param to_steal tasks to be considered stolen
       */
      virtual void permit_task_steal(Processor thief, 
                                const std::vector<const Task*> &tasks,
                                std::set<const Task*> &to_steal) = 0;

      /**
       * ----------------------------------------------------------------------
       *  Slice Domain 
       * ----------------------------------------------------------------------
       * Instead of needing to map an index space of tasks on a single
       * domain, Legion allows index space of tasks to be decomposed
       * into smaller sets of tasks that are mapped in parallel on
       * different processors.  To achieve this, the domain of the
       * index space task launch must be sliced into subsets of points
       * and distributed to the different processors which will actually
       * run the tasks.  Decomposing arbitrary domains in a way that
       * matches the target architecture is clearly a mapping decision.
       * Slicing the domain can be done recursively to match the 
       * hierarchical nature of modern machines.  By setting the
       * 'recurse' field on a DomainSplit struct to true, the runtime
       * will invoke slice_domain again on the destination node.
       * It is acceptable to return a single slice consisting of the
       * entire domain, but this will guarantee that all points in 
       * an index space will map on the same node.  Note that if
       * the slicing procedure doesn't slice the domain into sub-domains
       * that totally cover the original domain, the missing tasks
       * will not run (the only place where mapper correctness is important).
       * @param task the task being considered
       * @param domain the domain to be sliced
       * @param slices the slices of the domain that were made
       */
      virtual void slice_domain(const Task *task, const Domain &domain,
                                std::vector<DomainSplit> &slices) = 0;

      /**
       * ----------------------------------------------------------------------
       *  Pre-Map Task 
       * ----------------------------------------------------------------------
       * Prior to the task being moved or mapped onto a node, it undergoes
       * a pre-mapping phase.  The Mapper has an option of pre-mapping 
       * any of the regions prior to moving the task.  If the 'must_early_map'
       * field on a RegionRequirement is set, then the mapper must specify
       * the necessary fields for the runtime to perform the mapping.  These
       * fields are covered in detail in documentation of the map_task
       * mapper call.  Also note that once a task pre-maps any regions then
       * it will no longer be eligible for stealing as the runtime checks
       * that the instances selected for the pre-map regions are visible
       * from the target processor.
       *
       * Note also that if a task decides to pre-map regions or is required
       * to by the runtime, it must also specify the blocking factor
       * for each region requirement since these fields are normally not
       * set until the select task variant call which only occurs once
       * a task has been bound to a processor.
       * @param task the task being premapped
       * @return should the runtime notify the mapper of a successful premapping
       */
      virtual bool pre_map_task(Task *task) = 0;

      /**
       * ----------------------------------------------------------------------
       *  Select Task Variant 
       * ----------------------------------------------------------------------
       * Legion supports having multiple functionally equivalent variants
       * for a task.  This call allows the mapper to select the variant
       * best matched to the target processor by setting the value
       * of the 'selected_variant' field.  In addition to specifying
       * the variant to be used, the mapper must fill in the 
       * 'blocking_factor' field on each region requirement that is to
       * be mapped telling the runtime the required layout for the region.
       * 
       * The selected blocking factor must be between 1 and the value
       * contained in the 'max_blocking_factor' field for the given
       * region requirement.  Data can be laid out in a variety of
       * ways by specifying the blocking factor.  For example,
       * consider an instance with 2 fields (A and B) and four
       * values for each field.  A blocking factor of 1 will 
       * allocate 1 value of each field at a time before going
       * onto the next value of each field.  This corresponds to
       * an array-of-structs (AOS) layout of the data and is well
       * suited to linear walks of memory by sequential un-vectorizied
       * CPU kernels.
       * ---------------------------------------------------
       * AOS layout (blocking factor = 1)
       * ---------------------------------------------------
       * A B A B A B A B
       * ---------------------------------------------------
       * Another possible layout would be to use the max_blocking_factor
       * (in this case 4) to indicate that all the A values should be
       * laid out before all the B values.  This corresponds to a
       * struct-of-arrays (SOA) layout of the data and is well
       * suited to achieving coalesced loads on GPUs.
       * ---------------------------------------------------
       * SOA layout (blocking factor = max_blocking_factor = 4)
       * ---------------------------------------------------
       * A A A A B B B B
       * ---------------------------------------------------
       * However, there is a third option, which is to choose a
       * hybrid blocking factor which is useful for vector loads
       * for vectorized CPU kernels.  Consider if A and B are
       * both double-precision values, and the task being executed
       * is an SSE-vector kernel that wants to load 2 A values
       * into an SSE register followed by 2 B values into an 
       * SSE register using SSE vector loads.  In this case,
       * a blocking factor of 2 will work best as it supports
       * vector loads, while still maintaining a linear walk
       * through memory for the prefetchers.
       * ---------------------------------------------------
       * Hybrid layout (blocking factor = 2)
       * ---------------------------------------------------
       * A A B B A A B B
       * ---------------------------------------------------
       * Examples of useful hybrid blocking factors 
       * to be considered are 4 and 2 for single- and double-precision
       * SSE vector loads respectively, 8 and 4 for single- and 
       * double-precision AVX vector loads, and 16 and 8 for 
       * single- and double-precision vector loads on the Xeon Phi.
       *
       * For reduction instances, instead of selecting the layout of
       * the instance by providing a blocking factor, the mapper can
       * control whether to use a reduction-fold instance or a
       * reduction-list instance.  Reduction-fold instances allocate
       * a single value in memory for each point in the index space
       * and fold multiple reduction values into the same location.
       * Reduction-list instances will instead buffer up all the 
       * reductions in a list and replay them when the buffer is copied
       * to its destination.  Reduction-fold instances work better for
       * dense reductions while reduction-list instances work better
       * for sparse reductions.  This choice is determined by setting
       * the 'reduction_list' flag on the region requirement for
       * the reduction region.
       * @param task the task being considered
       */
      virtual void select_task_variant(Task *task) = 0; 

      /**
       * ----------------------------------------------------------------------
       *  Map Task 
       * ----------------------------------------------------------------------
       * After a task has been bound to a processor to run, it must decided
       * where to place the physical instances for each of the logical regions
       * that is requested.  This call is used to decide where to place
       * those physical regions.  All information is contained within the task
       * object and the fields on the region requirements for each region.
       *
       * On each region requirement, the runtime specifies the memories which
       * contain at least one field with valid data for the requirement and a
       * boolean indicating if that memory contains an instance with all the
       * necessary valid fields ('current_instances' in RegionRequirement)
       * and whether it is laid out in a manner consistent with the selected
       * blocking factor chosen in the select_task_variant mapping call.
       * The Mapper then fills in the 'target_ranking' vector on each region
       * requirement specifying the a list of memories in which to try to 
       * create a physical instance.  For each memory, the
       * runtime will first try to find a physical instance in that 
       * memory that contains all the needed data for the region 
       * requirement.  If such a physical instance cannot be found, the
       * runtime will attempt to make an instance and then issue the
       * necessary copies to have valid data before the task begins.
       * If the memory is full, the runtime continues on to the next
       * memory in the list.  If all the memories fail, the runtime
       * will report back to the mapper that the mapping failed.
       *
       * In addition to specifying a list of memories, there are two other
       * fields that can help control the mapping process.  The Mapper
       * can also specify a list of additional fields that to be allocated 
       * in any created physical instances of a logical region 
       * ('additional_fields' for each RegionRequirement).  This is useful 
       * if the mapper knows that it wants to allocate larger physical 
       * instances in a specific memory that can be used for many future 
       * tasks.  The final value is the 'enable_WAR_optimization' field.  
       * A write-after-read optimization means that if the runtime 
       * determines that an instance exists in a given memory with 
       * valid data, but using such an instance would require the current 
       * task to wait for a previous task that is reading the region to 
       * finish, then a new copy of the instance in the same memory should 
       * be made so the two tasks can be run in parallel.  This optimization 
       * comes at the price of requiring additional space in the target 
       * memory.  By default the WAR optimization is disabled.
       *
       * For advanced mapping strategies, Legion supports the ability to
       * virtually map a logical region by setting the 'virtual_map'
       * field on a region requirement to true.
       * While Legion requires that privileges be passed following the
       * task hierarchy, it is not required that tasks actually make
       * physical instances for the logical regions that they have 
       * requested as part of their region requirements.  If this is
       * the case, then the task can opt to map the region virtually
       * which means that no physical instance will be made for the 
       * particular region.  Instead all of the task's sub-tasks must
       * finish mapping at which point the resulting state will flow
       * back out of the specified task's context into the context in
       * which the task was initially being mapped.  This can allow
       * for more detailed mapping information, but also adds length
       * to the mapping path.  Mapping a region virtually is a trade-off
       * that is made based on the amount of computation in the task
       * and its subtasks and the precision of mapping information.
       *
       * Design note: Legion intentionally hides the names and information
       * about the physical instances that actually exist within the 
       * memory hierarchy and instead performs mappings based on memories.
       * This hides a level of detail from the mapper, but also makes
       * the interface significantly simpler.  Furthermore, to reduce 
       * the number of calls to the mapper, we encourage the listing of
       * memories so the mapper can express its full range of choices
       * in one call.  The allows the runtime to have to keep querying
       * the mapper to see if it would like to continue attempting
       * to map a region.  The mapper interface is the aspect of the 
       * interface that is currently in the greatest flux and the easiest to
       * change.  We are interested in hearing from real users about
       * whether they prefer to use memories or actual names of physical
       * instances when performing mapping decisions.
       * @param task the task to be mapped
       * @return should the runtime notify the mapper of a successful mapping
       */
      virtual bool map_task(Task *task) = 0;

      /**
       * ----------------------------------------------------------------------
       *  Post-Map Task 
       * ----------------------------------------------------------------------
       *  This is a temporary addition to the mapper interface that will
       *  change significantly in the new mapper interface. If the 
       *  'post_map_task' flag is set for a task in select_task_options
       *  mapper call then this call will be invoked after map_task succeeds.
       *  It works very similar to map_task in that the mapper can ask
       *  for the creation of new instances for region requirements to be
       *  done after the task has completed by filling in the target_ranking
       *  field of a region requirement with memories in which to place 
       *  additional instances after the task has finished executing. 
       */
      virtual void post_map_task(Task *task) = 0;

      /**
       * ----------------------------------------------------------------------
       *  Map Inline 
       * ----------------------------------------------------------------------
       * This call has a identical semantics to the map_task call, with the
       * exception that there is only a single region requirement in the
       * inline operation and virtual mappings are not permitted.  Otherwise
       * all the fields in the region requirement must be filled out by the
       * mapper to enable mapping.
       *
       * In addition, the mapper must also set the 'blocking_factor' field or
       * the 'reduction_list' field in the case of reduction copies as there
       * is no equivalent selection of task variants for an inline mapping.
       * @param inline_operation inline operation to be mapped
       * @return should the runtime notify the mapper of a successful mapping
       */
      virtual bool map_inline(Inline *inline_operation) = 0;

      /**
       * ----------------------------------------------------------------------
       *  Map Copy
       * ----------------------------------------------------------------------
       * Similar to the map_task call, this call asks the mapper to fill
       * in the necessary fields in the destination region requirements for
       * this copy operation.  Note that ONLY THE DESTINATION region
       * requirements have to have their mapping fields set as the source
       * instances will be used to find existing physical instances containing
       * valid data and will not actually be mapped. Users have the option
       * of still filling in mapping requests for the src region requirements
       * to create an explicit instance for performing the copy, but it 
       * is not required.
       *
       * In addition, the mapper must also set the 'blocking_factor' field or
       * the 'reduction_list' field in the case of reduction copies as there
       * is no equivalent selection of a task variant for a copy.  Unlike
       * mapping a task, copies are not permitted to request virtual mappings.
       * @param copy copy operation to be mapped
       * @return should the runtime notify the mapper of a successful mapping
       */
      virtual bool map_copy(Copy *copy) = 0; 

      /**
       * ----------------------------------------------------------------------
       *  Map Must Epoch 
       * ----------------------------------------------------------------------
       *  This is the mapping call for must epochs where a collection of tasks
       *  are all being requested to run in a must-parallelism mode so that
       *  they can synchronize.  The same mapping operations must be performed
       *  for each of the Task objects as in 'map_task', with the additional
       *  constraints that the mappings for the region requirements listed
       *  in the 'constraints' vector be satisifed to ensure the tasks 
       *  actually can run in parallel.
       *  @param tasks the tasks to be mapped
       *  @param constraints the constraints on the mapping
       *  @return should the runtime notify the tasks of a successful mapping
       */
      virtual bool map_must_epoch(const std::vector<Task*> &tasks,
                            const std::vector<MappingConstraint> &constraints,
                            MappingTagID tag) = 0;

      /**
       * ----------------------------------------------------------------------
       *  Notify Mapping Result 
       * ----------------------------------------------------------------------
       * If the mapper returned true to any map request, these calls
       * are used to return the mapping result back to the mapper if
       * the mapping succeeded.  The result memory for each of the 
       * instances that are to be used is set in the 'selected_memory'
       * field of each RegionRequirement.  If a virtual mapping was
       * selected then the memory will be a NO_MEMORY.
       */
      virtual void notify_mapping_result(const Mappable *mappable) = 0;

      /**
       * ----------------------------------------------------------------------
       *  Notify Mapping Failed 
       * ----------------------------------------------------------------------
       * In the case that a mapping operation failed (i.e. no physical
       * instances could be found or made in the list of memories), then
       * the runtime notifies the mapper with this call.  Region requirements
       * that caused the task to fail to map will have the 'mapping_failed'
       * field in their region requirement set to true.  The task is
       * then placed back onto the mapper's ready queue.
       */
      virtual void notify_mapping_failed(const Mappable *mappable) = 0;

      /**
       * ----------------------------------------------------------------------
       *  Rank Copy Targets 
       * ----------------------------------------------------------------------
       * This mapper call is invoked when a non-leaf task is launched in order
       * to set up the configuration of the context for this task to manage
       * how far deferred execution can progress. There are two components
       * to managing this: first controlling the number of outstanding 
       * operations in the context, and second controlling how many of 
       * these need to be mapped before the scheduler stops being invoked
       * by the runtime. By setting the following fields, the mapper 
       * can control both of these characteristics.
       *
       * max_window_size - set the maximum number of operations that can
       *                   be outstanding in a context. The default value
       *                   is set to either 1024 or whatever value was
       *                   passed on the command line to the -hl:window flag.
       *                   Setting the value less than or equal to zero will
       *                   disable the maximum.
       * hysteresis_percentage - set the percentage of the maximum task window
       *                   that should be outstanding before a context starts
       *                   issuing tasks again. Hysteresis avoids jitter and
       *                   enables a more efficient execution at the potential
       *                   cost of latency in a single task. The default value
       *                   is 75% indicating that a context will start 
       *                   launching new sub-tasks after the current number of
       *                   tasks drains to 75% of the max_window_size.  If the
       *                   max_window_size is disabled then this parameter has
       *                   no effect on the execution.
       * max_outstanding_frames - instead of specifying the maximum window size
       *                   applications can also launch frames corresponding
       *                   to application specific groups of tasks (see the
       *                   'complete_frame' runtime call for more information). 
       *                   The 'max_outstanding_frames' field allows mappers 
       *                   to specify the maximum number of outstanding frames 
       *                   instead. Setting this parameter subsumes the 
       *                   max_window_size.
       * min_tasks_to_schedule - specify the minimum number of pending mapped
       *                   tasks that must be issued before calls to 
       *                   select_tasks_to_schedule are stopped for this 
       *                   context. The default is set to 32 or the value
       *                   passed on the command line to the -hl:sched flag.
       *                   Any value of 0 or less will disable the check
       *                   causing select_tasks_to_schedule to be polled as
       *                   long as there are tasks to map.
       * max_directory_size - specifying the maximum number of leaf entries
       *                   in the region tree state directory for which the
       *                   runtime should maintain precise information. Beyond
       *                   this number the runtime may introduce imprecision
       *                   that results in unnecessary invalidation messages
       *                   in order to minimize the data structure size.
       */
      virtual void configure_context(Task *task) = 0;

      /**
       * ----------------------------------------------------------------------
       *  Rank Copy Targets 
       * ----------------------------------------------------------------------
       * If this call occurs the runtime is rebuilding a physical instance
       * of a logical region because it is closing up one or more partitions
       * of a logical region so it can open a new one.  The runtime
       * provides the mapper with all the same information for other mapper
       * calls as well as the logical region that is being rebuild.  The
       * runtime also tells the mapper where the current physical instances
       * exist that have enough space to rebuild the instance.  The mapper
       * then specifies an optional list of memories to_reuse and make
       * as a new valid copy of the region.  The mapper can also specify
       * an additional list of memories in which to try to create a new
       * valid physical instance in which to rebuild the instance.  The
       * create_one flag specifies whether the runtime should attempt to
       * make a single new instance from the create list or if false to try
       * all of them.  In many cases, it can be beneficial to scatter 
       * valid copies of rebuilt-region in different memories to be 
       * available for future tasks.
       *
       * Another option available to the mapper if the 'complete' flag
       * is set to true is to leave both lists empty and to create 
       * a virtual instance which represents the data as an agglomeration
       * of the existing physical instances instead of explicitly 
       * rebuilding a physical instance.  When copies are made from the
       * virtual instance the runtime will perform a more expensive
       * copy analysis.  In some cases this approach results in higher 
       * performance if the cost of the analysis is saved by the reduced 
       * amount of data moved.  This is solely a performance decision and 
       * is therefore left to the mapper to decide.
       * @param mappable the mappable object that is causing the close
       * @param rebuild_region the logical region being rebuilt
       * @param current_instances memories which contain physical instances
       *    with enough space to rebuild the instance
       * @param complete whether the node we are closing is complete
       *    and therefore able to make composite instances
       * @param max_blocking_factor the maximum blocking factor possible
       *    for the instances to be made
       * @param to_reuse memories with physical instances to re-use
       * @param to_create memories to try and make physical instances in
       * @param create_one only try to create one new physical instance
       *    or try to create as many as possible if false
       * @param blocking_factor the chosen blocking factor for the
       *    instances to be made
       * @return whether to make a composite instance
       */
      virtual bool rank_copy_targets(const Mappable *mappable,
                                     LogicalRegion rebuild_region,
                                     const std::set<Memory> &current_instances,
                                     bool complete,
                                     size_t max_blocking_factor,
                                     std::set<Memory> &to_reuse,
                                     std::vector<Memory> &to_create,
                                     bool &create_one,
                                     size_t &blocking_factor) = 0;
      /**
       * ----------------------------------------------------------------------
       *  Rank Copy Sources 
       * ----------------------------------------------------------------------
       * To perform a copy operation, the runtime often has a choice
       * of source locations from where to copy data from to create
       * a physical instance in a destination memory.  If a choice
       * is available, the runtime uses this call to ask the mapper
       * to select an ordering for the memories to use when issuing
       * the multiple copies necessary to update all the fields in
       * a physical instance.  If the returned vector is empty, the
       * runtime will issue the copies in an arbitrary order until
       * all the field contain valid data.
       * @param mappable the mappable object for which the copy is occuring
       * @param current_instances memories containing valid data
       * @param dst_mem the target memory containing the physical instance
       * @param chosen_order the order from which to issue copies
       */
      virtual void rank_copy_sources(const Mappable *mappable,
                      const std::set<Memory> &current_instances,
                      Memory dst_mem, 
                      std::vector<Memory> &chosen_order) = 0;
      /**
       * ----------------------------------------------------------------------
       *  Notify Profiling Info 
       * ----------------------------------------------------------------------
       * Report back the profiling information for a task
       * after the task has finished executing.
       * @param task the task that was profiled
       * @param profiling the profiling information for the task
       */
      virtual void notify_profiling_info(const Task *task) = 0;

      // temporary helper for old profiling code
      struct ExecutionProfile {
	unsigned long long start_time; // microseconds since program start
	unsigned long long stop_time; // microseconds since program start
      };

      /**
       * ----------------------------------------------------------------------
       *  Speculate on Predicate 
       * ----------------------------------------------------------------------
       * Ask the mapper if it would like to speculate on the
       * value of a boolean predicate.  If the call returns
       * yes, then the mapper should set the spec_value to 
       * indicate what it thinks the speculative value should
       * be.  If the call returns false, then the result of
       * spec_value will be ignored by the runtime and 
       * anything depending on the predicate will block until
       * it resolves.
       * @param op the op that is predicated
       * @param spec_value the speculative value to be
       *    set if the mapper is going to speculate
       * @return true if the mapper is speculating, false otherwise
       */
      virtual bool speculate_on_predicate(const Mappable *mappable,
                                          bool &spec_value) = 0;

      /**
       * ----------------------------------------------------------------------
       *  Get Tunable Value
       * ----------------------------------------------------------------------
       * Ask the mapper to specify the value for a tunable variable.
       * This operation is invoked whenever a call to 'get_tunable_value'
       * is made by a task.  Currently all tunable variables are integers
       * so the value returned from this method will be passed back directly
       * as the resulting value for the tunable variable.
       * @param task the task that is asking for the tunable variable
       * @param tid the ID of the tunable variable (e.g. name)
       * @param tag the context specific tag for the tunable request
       * @return the resulting value for the tunable variable
       */
      virtual int get_tunable_value(const Task *task, 
                                    TunableID tid,
                                    MappingTagID tag) = 0;

      /**
       * ----------------------------------------------------------------------
       *  Handle Message
       * ----------------------------------------------------------------------
       * Handle a message sent from one of our adjacent mappers of the same
       * kind on a different processor.
       * @param source the processor whose mapper sent the message
       * @param message buffer containing the message
       * @param length size of the message in bytes
       */
      virtual void handle_message(Processor source,
                                  const void *message, size_t length) = 0;

      /**
       * ----------------------------------------------------------------------
       *  Handle Mapper Task Result 
       * ----------------------------------------------------------------------
       * Handle the result of the mapper task with the corresponding task
       * token that was launched by a call to 'launch_mapper_task'.
       * @param event the event identifying the task that was launched
       * @param result buffer containing the result of the task
       * @param result_size size of the result buffer in bytes
       */
      virtual void handle_mapper_task_result(MapperEvent event,
                                             const void *result, 
                                             size_t result_size) = 0;

      //------------------------------------------------------------------------
      // All methods below here are methods that are already implemented
      // and serve as an interface for inheriting mapper classes to 
      // introspect the Legion runtime.  They also provide interfaces
      // for directing the runtime to perform operations like sending
      // messages to other mappers.  We provide these methods here in
      // order to scope who is able to access them.  We only want mapper
      // objects to have access to them and hence they are provided here
      // as methods that will be inherited by sub-type mappers.
      //------------------------------------------------------------------------
    protected:
      //------------------------------------------------------------------------
      // Methods for communication with other mappers
      //------------------------------------------------------------------------

      /**
       * Send a message to our corresponding mapper for a different processor.
       * @param target the processor whose mapper we are sending the message
       * @param message a pointer to a buffer containing the message
       * @param the size of the message to be sent in bytes
       */
      void send_message(Processor target, const void *message, size_t length); 

      /**
       * Broadcast a message to all other mappers of the same kind. Mappers
       * can also control the fan-out radix for the broadcast message.
       */
      void broadcast_message(const void *message, size_t length, int radix = 4);
    protected:
      //------------------------------------------------------------------------
      // Methods for launching asynchronous mapper tasks 
      //------------------------------------------------------------------------

      /**
       * Launch an asychronous task to compute a value for a mapper to use
       * in the future. Note that because mapper calls are not allowed to 
       * block, we don't return a future for these tasks.  Instead we 
       * return a mapper event that can be used to track when the result 
       * of the task is passed back to the mapper.
       */
      MapperEvent launch_mapper_task(Processor::TaskFuncID tid,
                                     const TaskArgument &arg);

      /**
       * We can invoke this call during any mapping call that will defer a
       * mapping call until a specific mapper event has triggered.
       */
      void defer_mapper_call(MapperEvent event);

      /**
       * Merge a collection of mapper events together to create a new
       * mapper event that will trigger when all preconditions have triggered.
       */
      MapperEvent merge_mapper_events(const std::set<MapperEvent> &events);
    protected:
      //------------------------------------------------------------------------
      // Methods for introspecting index space trees 
      // For documentation see methods of the same name in Runtime
      //------------------------------------------------------------------------
      IndexPartition get_index_partition(IndexSpace parent, Color color) const;

      IndexSpace get_index_subspace(IndexPartition p, Color c) const;
      IndexSpace get_index_subspace(IndexPartition p, 
                                    const DomainPoint &color) const;

      bool has_multiple_domains(IndexSpace handle) const;

      Domain get_index_space_domain(IndexSpace handle) const;

      void get_index_space_domains(IndexSpace handle,
                                   std::vector<Domain> &domains) const;

      Domain get_index_partition_color_space(IndexPartition p) const;

      void get_index_space_partition_colors(IndexSpace sp, 
                                            std::set<Color> &colors) const;

      bool is_index_partition_disjoint(IndexPartition p) const;

      template<unsigned DIM>
      IndexSpace get_index_subspace(IndexPartition p, 
                          LegionRuntime::Arrays::Point<DIM> &color_point) const;

      Color get_index_space_color(IndexSpace handle) const;

      Color get_index_partition_color(IndexPartition handle) const;

      IndexSpace get_parent_index_space(IndexPartition handle) const;

      bool has_parent_index_partition(IndexSpace handle) const;
      
      IndexPartition get_parent_index_partition(IndexSpace handle) const;
    protected:
      //------------------------------------------------------------------------
      // Methods for introspecting field spaces 
      // For documentation see methods of the same name in Runtime
      //------------------------------------------------------------------------
      size_t get_field_size(FieldSpace handle, FieldID fid) const;

      void get_field_space_fields(FieldSpace handle, std::set<FieldID> &fields);
    protected:
      //------------------------------------------------------------------------
      // Methods for introspecting logical region trees
      //------------------------------------------------------------------------
      LogicalPartition get_logical_partition(LogicalRegion parent, 
                                             IndexPartition handle) const;

      LogicalPartition get_logical_partition_by_color(LogicalRegion parent,
                                                      Color color) const;

      LogicalPartition get_logical_partition_by_tree(IndexPartition handle,
                                                     FieldSpace fspace,
                                                     RegionTreeID tid) const;

      LogicalRegion get_logical_subregion(LogicalPartition parent,
                                          IndexSpace handle) const;

      LogicalRegion get_logical_subregion_by_color(LogicalPartition parent,
                                                   Color color) const;
      
      LogicalRegion get_logical_subregion_by_tree(IndexSpace handle,
                                                  FieldSpace fspace,
                                                  RegionTreeID tid) const;

      Color get_logical_region_color(LogicalRegion handle) const;

      Color get_logical_partition_color(LogicalPartition handle) const;

      LogicalRegion get_parent_logical_region(LogicalPartition handle) const;

      bool has_parent_logical_partition(LogicalRegion handle) const;

      LogicalPartition get_parent_logical_partition(LogicalRegion handle) const;
    protected:
      //------------------------------------------------------------------------
      // Methods for introspecting the state of machine resources
      //------------------------------------------------------------------------
      
      /**
       * Take a sample of the amount of space allocated in a
       * specific memory. Note this is just a sample and may 
       * return different values even in consecutive calls.
       * Also note that this value is imprecise and is only 
       * based on allocations known to the local address space.
       * @param m the memory to be sampled
       * @return size in bytes of all the instances allocated in the memory
       */
      size_t sample_allocated_space(Memory m) const;

      /**
       * Take a sample of the amount of free memory available
       * in a specific memory. Note that this is just a sample
       * and may return different values even in consecutive calls.
       * Also note that this value is imprecise and is only based
       * on allocations known to the local address space.
       * @param m the memory to be sampled
       * @return size in bytes of all the free space in the memory
       */
      size_t sample_free_space(Memory m) const;

      /**
       * Take a sample of the number of instances allocated in
       * a specific memory. Note that this is just a sample and
       * may return different values even in consecutive calls.
       * Also note that this value is imprecise and is only based
       * on allocations known to the local address space.
       * @param m the memory to be sampled
       * @return number of instances allocated in the memory
       */
      unsigned sample_allocated_instances(Memory m) const;

      /**
       * Take a sample of the number of unmapped tasks which are
       * currently assigned to the processor, but are unmapped.
       * This sample is only valid for processors in the local
       * address space.
       * @param p the processor to be sampled
       * @return the count of the tasks assigned to the processor but unmapped
       */
      unsigned sample_unmapped_tasks(Processor p) const;
    };

    /**
     * \class ShimMapper
     * The ShimMapper class provides backwards compatibility with an earlier
     * version of the mapping interface.  The new mapper calls are implemented
     * as functions of the earlier mapper calls.  Old mappers can use the
     * new Mapper interface simply by extending the ShimMapper instead
     * of the old DefaultMapper
     */
    class ShimMapper : public DefaultMapper {
    public:
      ShimMapper(Machine machine, HighLevelRuntime *rt, Processor local);
      ShimMapper(const ShimMapper &rhs);
      virtual ~ShimMapper(void);
    public:
      ShimMapper& operator=(const ShimMapper &rhs);
    public:
      // The new mapping calls
      virtual void select_task_options(Task *task);
      virtual void select_tasks_to_schedule(
                      const std::list<Task*> &ready_tasks);
      virtual void target_task_steal(
                            const std::set<Processor> &blacklist,
                            std::set<Processor> &targets);
      // No need to override permit_task_steal as the interface is unchanged
      //virtual void permit_task_steal(Processor thief, 
      //                          const std::vector<const Task*> &tasks,
      //                          std::set<const Task*> &to_steal);
      // No need to override slice_domain as the interface is unchanged
      //virtual void slice_domain(const Task *task, const Domain &domain,
      //                          std::vector<DomainSplit> &slices);
      virtual bool pre_map_task(Task *task);
      virtual void select_task_variant(Task *task);
      virtual bool map_task(Task *task);
      virtual bool map_copy(Copy *copy);
      virtual bool map_inline(Inline *inline_operation);
      virtual void notify_mapping_result(const Mappable *mappable);
      virtual void notify_mapping_failed(const Mappable *mappable);
      virtual bool rank_copy_targets(const Mappable *mappable,
                                     LogicalRegion rebuild_region,
                                     const std::set<Memory> &current_instances,
                                     bool complete,
                                     size_t max_blocking_factor,
                                     std::set<Memory> &to_reuse,
                                     std::vector<Memory> &to_create,
                                     bool &create_one,
                                     size_t &blocking_factor);
      virtual void rank_copy_sources(const Mappable *mappable,
                      const std::set<Memory> &current_instances,
                      Memory dst_mem, 
                      std::vector<Memory> &chosen_order);
      virtual void notify_profiling_info(const Task *task);
      virtual bool speculate_on_predicate(const Mappable *mappable,
                                          bool &spec_value);
    protected:
      // Old-style mapping methods
      virtual bool spawn_task(const Task *task);
      virtual bool map_task_locally(const Task *task);
      virtual Processor select_target_processor(const Task *task);
      virtual bool map_region_virtually(const Task *task, Processor target,
					const RegionRequirement &req, 
                                        unsigned index);
      virtual bool map_task_region(const Task *task, Processor target, 
                                   MappingTagID tag, bool inline_mapping, 
                                   bool pre_mapping, 
                                   const RegionRequirement &req, unsigned index,
        const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
				   std::vector<Memory> &target_ranking,
				   std::set<FieldID> &additional_fields,
				   bool &enable_WAR_optimization);
      virtual size_t select_region_layout(const Task *task, Processor target,
					  const RegionRequirement &req, 
                                          unsigned index, 
                                          const Memory &chosen_mem, 
                                          size_t max_blocking_factor);
      virtual bool select_reduction_layout(const Task *task, 
                                           const Processor target,
					   const RegionRequirement &req, 
                                           unsigned index, 
                                           const Memory &chosen_mem);
      virtual void select_tasks_to_schedule(const std::list<Task*> &ready_tasks,
					    std::vector<bool> &ready_mask);
      virtual Processor target_task_steal(const std::set<Processor> &blacklist);
      virtual VariantID select_task_variant(const Task *task, Processor target);
      virtual void notify_mapping_result(const Task *task, Processor target,
					 const RegionRequirement &req,
					 unsigned index, bool inline_mapping, 
                                         Memory result);
      virtual void notify_failed_mapping(const Task *task, Processor target,
					 const RegionRequirement &req,
					 unsigned index, bool inline_mapping);
      virtual void rank_copy_sources(const std::set<Memory> &current_instances,
				     const Memory &dst, 
                                     std::vector<Memory> &chosen_order);
      virtual void rank_copy_targets(const Task *task, Processor target,
                                   MappingTagID tag, bool inline_mapping,
                                   const RegionRequirement &req, unsigned index,
                                   const std::set<Memory> &current_instances,
                                   std::set<Memory> &to_reuse,
                                   std::vector<Memory> &to_create,
                                   bool &create_one);
      virtual bool profile_task_execution(const Task *task, Processor target);
      virtual void notify_profiling_info(const Task *task, Processor target,
					 const ExecutionProfile &profiling);
      virtual bool speculate_on_predicate(MappingTagID tag, 
                                          bool &speculative_value);
    };

  };
};

// For backwards compatibility
namespace LegionRuntime {
  namespace HighLevel {
    typedef Legion::Mapping::ShimMapper ShimMapper;
  };
};

#endif // __SHIM_MAPPER_H__

// EOF

