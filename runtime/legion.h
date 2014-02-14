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


#ifndef __LEGION_RUNTIME_H__
#define __LEGION_RUNTIME_H__

#include "legion_types.h"

// temporary helper macro to turn link errors into runtime errors
#define UNIMPLEMENTED_METHOD(retval) do { assert(0); return retval; } while(0)

/**
 * \namespace LegionRuntime
 * Namespace for all Legion runtime objects
 */
namespace LegionRuntime {
  /**
   * \namespace HighLevel
   * Namespace specifically devoted to objects of the Legion high-level
   * runtime system (see LowLevel namespace for low-level objects). 
   */
  namespace HighLevel {

    //==========================================================================
    //                       Data Description Classes
    //==========================================================================

    /**
     * \class FieldSpace
     * Field spaces define the objects used for managing the fields or
     * "columns" of a logical region.  Only the Legion runtime is able
     * to create non-empty field spaces.  Fields within a field space
     * are allocated using field space allocators
     *
     * @see FieldSpaceAllocator
     */
    class FieldSpace {
    public:
      static const FieldSpace NO_SPACE; /**< empty field space handle*/
    protected:
      // Only the runtime should be allowed to make these
      FRIEND_ALL_RUNTIME_CLASSES
      FieldSpace(FieldSpaceID id);
    public:
      FieldSpace(void);
      FieldSpace(const FieldSpace &rhs);
    public:
      inline FieldSpace& operator=(const FieldSpace &rhs);
      inline bool operator==(const FieldSpace &rhs) const;
      inline bool operator!=(const FieldSpace &rhs) const;
      inline bool operator<(const FieldSpace &rhs) const;
      inline FieldSpaceID get_id(void) const { return id; }
    private:
      FieldSpaceID id;
    };

    /**
     * \class LogicalRegion
     * Logical region objects define handles to the actual logical regions
     * maintained by the runtime.  Logical regions are defined by a triple
     * consisting of the index space, field space, and region tree ID of
     * the logical region.  These three values are used to uniquely name
     * every logical region created in a Legion program.
     *
     * Logical region objects can be copied by value and stored in data 
     * structures.  Only the Legion runtime is able to create logical region 
     * objects that are non-empty.
     *
     * @see FieldSpace
     */
    class LogicalRegion {
    public:
      static const LogicalRegion NO_REGION; /**< empty logical region handle*/
    protected:
      // Only the runtime should be allowed to make these
      FRIEND_ALL_RUNTIME_CLASSES
      LogicalRegion(RegionTreeID tid, IndexSpace index, FieldSpace field);
    public:
      LogicalRegion(void);
      LogicalRegion(const LogicalRegion &rhs);
    public:
      inline LogicalRegion& operator=(const LogicalRegion &rhs);
      inline bool operator==(const LogicalRegion &rhs) const;
      inline bool operator!=(const LogicalRegion &rhs) const;
      inline bool operator<(const LogicalRegion &rhs) const;
    public:
      inline IndexSpace get_index_space(void) const { return index_space; }
      inline FieldSpace get_field_space(void) const { return field_space; }
      inline RegionTreeID get_tree_id(void) const { return tree_id; }
    private:
      // These are private so the user can't just arbitrarily change them
      RegionTreeID tree_id;
      IndexSpace index_space;
      FieldSpace field_space;
    };

    /**
     * \class LogicalPartition
     * Logical partition objects defines handles to the actual logical
     * partitions maintained by the runtime.  Logical partitions are
     * defined by a triple consisting of the index partition, field
     * space, and region tree ID of the logical partition.  These three
     * values are sufficient to name every logical partition created
     * in a Legion program.
     *
     * Logical partition objects can be copied by values and stored in 
     * data structures.  Only the Legion runtime is able to create 
     * non-empty logical partitions.
     *
     * @see FieldSpace
     */
    class LogicalPartition {
    public:
      static const LogicalPartition NO_PART; /**< empty logical partition */
    protected:
      // Only the runtime should be allowed to make these
      FRIEND_ALL_RUNTIME_CLASSES
      LogicalPartition(RegionTreeID tid, IndexPartition pid, FieldSpace field);
    public:
      LogicalPartition(void);
      LogicalPartition(const LogicalPartition &rhs);
    public:
      inline LogicalPartition& operator=(const LogicalPartition &rhs);
      inline bool operator==(const LogicalPartition &rhs) const;
      inline bool operator!=(const LogicalPartition &rhs) const;
      inline bool operator<(const LogicalPartition &rhs) const;
    public:
      inline IndexPartition get_index_partition(void) const 
        { return index_partition; }
      inline FieldSpace get_field_space(void) const { return field_space; }
      inline RegionTreeID get_tree_id(void) const { return tree_id; }
    private:
      // These are private so the user can't just arbitrary change them
      RegionTreeID tree_id;
      IndexPartition index_partition;
      FieldSpace field_space;
    };

    //==========================================================================
    //                       Data Allocation Classes
    //==========================================================================

    /**
     * \class IndexAllocator
     * Index allocators provide objects for doing allocation on
     * index spaces.  They must be explicitly created by the
     * runtime so that they can be linked back to the runtime.
     * Index allocators can be passed by value to functions
     * and stored in data structures, but should not escape 
     * the enclosing context in which they were created.
     *
     * Index space allocators operate on a single index space
     * which is immutable.  Separate index space allocators
     * must be made to perform allocations on different index
     * spaces.
     *
     * @see HighLevelRuntime
     */
    class IndexAllocator {
    public:
      IndexAllocator(void);
      IndexAllocator(const IndexAllocator &allocator);
      ~IndexAllocator(void);
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      // Only the HighLevelRuntime should be able to make these
      IndexAllocator(IndexSpace space, IndexSpaceAllocator *allocator);
    public:
      IndexAllocator& operator=(const IndexAllocator &allocator);
      inline bool operator<(const IndexAllocator &rhs) const;
      inline bool operator==(const IndexAllocator &rhs) const;
    public:
      /**
       * @param num_elements number of elements to allocate
       * @return pointer to the first element in the allocated block
       */
      inline ptr_t alloc(unsigned num_elements = 1);
      /**
       * @param ptr pointer to the first element to free
       * @param num_elements number of elements to be freed
       */
      inline void free(ptr_t ptr, unsigned num_elements = 1);
      /**
       * @return the index space associated with this allocator
       */
      inline IndexSpace get_index_space(void) const { return index_space; }
    private:
      IndexSpace index_space;
      IndexSpaceAllocator *allocator;
    };

    /**
     * \class FieldAllocator
     * Field allocators provide objects for performing allocation on
     * field spaces.  They must be explicitly created by the runtime so
     * that they can be linked back to the runtime.  Field allocators
     * can be passed by value to functions and stored in data structures,
     * but they should never escape the enclosing context in which they
     * were created.
     *
     * Field space allocators operate on a single field space which
     * is immutable.  Separate field space allocators must be made
     * to perform allocations on different field spaces.
     *
     * @see FieldSpace
     * @see HighLevelRuntime
     */
    class FieldAllocator {
    public:
      FieldAllocator(void);
      FieldAllocator(const FieldAllocator &allocator);
      ~FieldAllocator(void);
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      // Only the HighLevelRuntime should be able to make these
      FieldAllocator(FieldSpace f, Context p, HighLevelRuntime *rt);
    public:
      FieldAllocator& operator=(const FieldAllocator &allocator);
      inline bool operator<(const FieldAllocator &rhs) const;
      inline bool operator==(const FieldAllocator &rhs) const;
    public:
      /**
       * Allocate a field with a given size. Optionally specify
       * the field ID to be assigned.  Note if you use
       * AUTO_GENERATE_ID, then all fields for the field space
       * should be generated this way or field names may be
       * deduplicated as the runtime will not check against
       * user assigned field names when generating its own.
       * @param field_size size of the field to be allocated
       * @param desired_fieldid field ID to be assigned to the
       *   field or AUTO_GENERATE_ID to specify that the runtime
       *   should assign a fresh field ID
       * @return field ID for the allocated field
       */
      inline FieldID allocate_field(size_t field_size, 
              FieldID desired_fieldid = AUTO_GENERATE_ID);
      /**
       * Deallocate the specified field from the field space.
       * @param fid the field ID to be deallocated
       */
      inline void free_field(FieldID fid);

      /**
       * Same as allocate field, but this field will only
       * be available locally on the node on which it is
       * created and not on remote nodes.  It will then be
       * implicitly destroyed once the task in which it is 
       * allocated completes.
       */
      inline FieldID allocate_local_field(size_t field_size,
              FieldID desired_fieldid = AUTO_GENERATE_ID);
      /**
       * Allocate a collection of fields with the specified sizes.
       * Optionally pass in a set of field IDs to use when allocating
       * the fields otherwise the vector should be empty or the
       * same size as field_sizes with AUTO_GENERATE_ID set as the
       * value for each of the resulting_field IDs.  The length of 
       * the resulting_fields vector must be less than or equal to 
       * the length of field_sizes.  Upon return it will be the same 
       * size with field IDs specified for all the allocated fields
       * @param field_sizes size in bytes of the fields to be allocated
       * @param resulting_fields optional field names for allocated fields
       * @return resulting_fields vector with length equivalent to
       *    the length of field_sizes with field IDs specified
       */
      inline void allocate_fields(const std::vector<size_t> &field_sizes,
                                  std::vector<FieldID> &resulting_fields);
      /**
       * Free a collection of field IDs
       * @param to_free set of field IDs to be freed
       */
      inline void free_fields(const std::set<FieldID> &to_free);
      /**
       * Same as allocate_fields but the fields that are allocated
       * will only be available locally on the node on which 
       * this call is made and not on remote nodes.  The fields
       * will be implicitly destroyed once the task in which
       * they were created completes.
       */
      inline void allocate_local_fields(const std::vector<size_t> &field_sizes,
                                        std::vector<FieldID> &resulting_fields);
      /**
       * @return field space associated with this allocator
       */
      inline FieldSpace get_field_space(void) const { return field_space; }
    private:
      FieldSpace field_space;
      Context parent;
      HighLevelRuntime *runtime;
    };

    //==========================================================================
    //                    Pass-By-Value Argument Classes
    //==========================================================================

    /**
     * \class TaskArgument
     * A class for describing an untyped task argument.  Note that task
     * arguments do not make copies of the data they point to.  Copies
     * are only made upon calls to the runtime to avoid double copying.
     * It is up to the user to make sure that the the data described by
     * a task argument is valid throughout the duration of its lifetime.
     */
    class TaskArgument {
      public:
      TaskArgument(void) : args(NULL), arglen(0) { }
      TaskArgument(const void *arg, size_t argsize)
        : args(const_cast<void*>(arg)), arglen(argsize) { }
      TaskArgument(const TaskArgument &rhs)
        : args(rhs.args), arglen(rhs.arglen) { }
    public:
      inline size_t get_size(void) const { return arglen; }
      inline void*  get_ptr(void) const { return args; }
    public:
      inline bool operator==(const TaskArgument &arg) const
        { return (args == arg.args) && (arglen == arg.arglen); }
      inline bool operator<(const TaskArgument &arg) const
        { return (args < arg.args) && (arglen < arg.arglen); }
      inline TaskArgument& operator=(const TaskArgument &rhs)
        { args = rhs.args; arglen = rhs.arglen; return *this; }
    private:
      void *args;
      size_t arglen;
    };

    /**
     * \class ArgumentMap
     * Argument maps provide a data structure for storing the task
     * arguments that are to be associated with different points in
     * an index space launch.  Argument maps are light-weight handle
     * to the actual implementation that uses a versioning system
     * to make it efficient to re-use argument maps over many task
     * calls, especially if there are very few changes applied to
     * the map between task call launches.
     */
    class ArgumentMap {
    public:
      ArgumentMap(void);
      ArgumentMap(const ArgumentMap &rhs);
      ~ArgumentMap(void);
    public:
      ArgumentMap& operator=(const ArgumentMap &rhs);
      inline bool operator==(const ArgumentMap &rhs) const
        { return (impl == rhs.impl); }
      inline bool operator<(const ArgumentMap &rhs) const
        { return (impl < rhs.impl); }
    public:
      /**
       * Check to see if a point has an argument set
       * @param point the point to check
       * @return true if the point has a value already set
       */
      bool has_point(const DomainPoint &point);
      /**
       * Associate an argument with a domain point
       * @param point the point to associate with the task argument
       * @param arg the task argument
       * @param replace specify whether to overwrite an existing value
       */
      void set_point(const DomainPoint &point, const TaskArgument &arg,
                     bool replace = true);
      /**
       * Remove a point from the argument map
       * @param point the point to be removed
       * @return true if the point was removed
       */
      bool remove_point(const DomainPoint &point);
      /**
       * Get the task argument for a point if it exists, otherwise
       * return an empty task argument.
       * @param point the point to retrieve
       * @return a task argument if the point exists otherwise
       *    an empty task argument
       */
      TaskArgument get_point(const DomainPoint &point) const;
    public:
      /**
       * An older method for setting the point argument in
       * an argument map.
       * @param point the point to associate the task argument
       * @param arg the argument
       * @param replace specify if the value should overwrite
       *    the existing value if it already exists
       */
      template<typename PT, unsigned DIM>
      inline void set_point_arg(const PT point[DIM], const TaskArgument &arg, 
                                bool replace = false);
      /**
       * An older method for removing a point argument from
       * an argument map.
       * @param point the point to remove from the map
       */
      template<typename PT, unsigned DIM>
      inline bool remove_point(const PT point[DIM]);
    private:
      FRIEND_ALL_RUNTIME_CLASSES
      class Impl;
      Impl *impl;
    private:
      ArgumentMap(Impl *i);
    };

    //==========================================================================
    //                           Predicate Classes
    //==========================================================================

    /**
     * \class Predicate
     * Predicate values are used for performing speculative 
     * execution within an application.  They are lightweight handles
     * that can be passed around by value and stored in data
     * structures.  However, they should not escape the context of
     * the task in which they are created as they will be garbage
     * collected by the runtime.  Except for predicates with constant
     * value, all other predicates should be created by the runtime.
     */
    class Predicate {
    public:
      static const Predicate TRUE_PRED;
      static const Predicate FALSE_PRED;
    public:
      Predicate(void);
      Predicate(const Predicate &p);
      explicit Predicate(bool value);
      ~Predicate(void);
    public:
      class Impl;
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      Impl *impl;
      // Only the runtime should be allowed to make these
      Predicate(Impl *impl);
    public:
      Predicate& operator=(const Predicate &p);
      inline bool operator==(const Predicate &p) const;
      inline bool operator<(const Predicate &p) const;
      inline bool operator!=(const Predicate &p) const;
    private:
      bool const_value;
    };

    //==========================================================================
    //             Simultaneous Coherence Synchronization Classes
    //==========================================================================

    /**
     * \class Lock 
     * NOTE THIS IS NOT A NORMAL LOCK!
     * A lock is an atomicity mechanism for use with regions acquired
     * with simultaneous coherence in a deferred execution model.  
     * Locks are light-weight handles that are created in a parent 
     * task and can be passed to child tasks to guaranteeing atomic access 
     * to a region in simultaneous mode.  Lock can be used to request 
     * access in either exclusive (mode 0) or non-exclusive mode (any number 
     * other than zero).  Non-exclusive modes are mutually-exclusive from each 
     * other. While locks can be passed down the task tree, they should
     * never escape the context in which they are created as they will be 
     * garbage collected when the task in which they were created is complete.
     *
     * There are two ways to use locks.  The first is to use the blocking
     * acquire and release methods on the lock directly.  Acquire
     * guarantees that the application will hold the lock when it 
     * returns, but may result in stalls while some other task is holding the 
     * lock.  The recommended way of using locks is to request
     * grants of a lock through the runtime interface and then pass
     * grants to launcher objects.  This ensures that the lock will be
     * held during the course of the operation while still avoiding
     * any stalls in the task's execution.
     * @see TaskLauncher
     * @see IndexLauncher
     * @see CopyLauncher
     * @see InlineLauncher
     * @see HighLevelRuntime
     */
    class Lock {
    public:
      Lock(void);
    protected:
      // Only the runtime is allowed to make non-empty locks 
      FRIEND_ALL_RUNTIME_CLASSES
      Lock(Reservation r);
    public:
      bool operator<(const Lock &rhs) const;
      bool operator==(const Lock &rhs) const;
    public:
      void acquire(unsigned mode = 0, bool exclusive = true);
      void release(void);
    private:
      Reservation reservation_lock;
    };

    /**
     * \struct LockRequest
     * This is a helper class for requesting grants.  It
     * specifies the locks that are needed, what mode they
     * should be acquired in, and whether or not they 
     * should be acquired in exclusive mode or not.
     */
    struct LockRequest {
    public:
      LockRequest(Lock l, unsigned mode = 0, bool exclusive = true);
    public:
      Lock lock;
      unsigned mode;
      bool exclusive;
    };

    /**
     * \class Grant
     * Grants are ways of naming deferred acquisitions and releases
     * of locks.  This allows the application to defer a lock 
     * acquire but still be able to use it to specify which tasks
     * must run while holding the this particular grant of the lock.
     * Grants are created through the runtime call 'acquire_grant'.
     * Once a grant has been used for all necessary tasks, the
     * application can defer a grant release using the runtime
     * call 'release_grant'.
     * @see HighLevelRuntime
     */
    class Grant {
    public:
      Grant(void);
      Grant(const Grant &g);
      ~Grant(void);
    public:
      class Impl;
    protected:
      // Only the runtime is allowed to make non-empty grants
      FRIEND_ALL_RUNTIME_CLASSES
      Grant(Impl *impl);
    public:
      bool operator==(const Grant &g) const
        { return impl == g.impl; }
      bool operator<(const Grant &g) const
        { return impl < g.impl; }
      Grant& operator=(const Grant &g);
    protected:
      Impl *impl;
    };

    /**
     * \class PhaseBarrier
     * Phase barriers are a synchronization mechanism for use with
     * regions acquired with simultaneous coherence in a deferred
     * execution model.  Phase barriers allow the application to
     * guarantee that a collection of tasks are all executing their
     * sub-tasks all within the same phase of computation.  Phase
     * barriers are light-weight handles that can be passed by value
     * or stored in data structures.  Phase barriers are made in
     * a parent task and can be passed down to any sub-tasks.  However,
     * phase barriers should not escape the context in which they
     * were created as the task that created them will garbage collect
     * their resources.
     *
     * Note that there are two ways to use phase barriers.  The first
     * is to use the blocking operations to wait for a phase to begin
     * and to indicate that the task has arrived at the current phase.
     * These operations may stall and block current task execution.
     * The preferred method for using phase barriers is to pass them
     * in as wait and arrive barriers for launcher objects which will
     * perform the necessary operations on barriers before an after
     * the operation is executed.
     * @see TaskLauncher
     * @see IndexLauncher
     * @see CopyLauncher
     * @see InlineLauncher
     * @see HighLevelRuntime
     */
    class PhaseBarrier {
    public:
      PhaseBarrier(void);
    protected:
      // Only the runtime is allowed to make non-empty phase barriers
      FRIEND_ALL_RUNTIME_CLASSES
      PhaseBarrier(Barrier b, unsigned participants);
    public:
      bool operator<(const PhaseBarrier &rhs) const;
      bool operator==(const PhaseBarrier &rhs) const;
    public:
      void arrive(unsigned count = 0);
      void wait(void);
    public:
      // for debug
      Barrier get_barrier(void) const { return phase_barrier; }
      unsigned participant_count(void) const { return participants; }
    private:
      Barrier phase_barrier;
      unsigned participants;
    };

    //==========================================================================
    //                    Operation Requirement Classes
    //==========================================================================

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
      inline void add_field(FieldID fid, bool instance = true);
    public:
#ifdef PRIVILEGE_CHECKS
      AccessorPrivilege get_accessor_privilege(void) const;
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
      unsigned flags; /**< optional flags set for region requirements*/
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

    /**
     * \struct IndexSpaceRequirement
     * Index space requirements are used to specify allocation and
     * deallocation privileges on logical regions.  Just like region
     * privileges, index space privileges must be inherited from a
     * region on which the parent task had an equivalent privilege.
     */
    struct IndexSpaceRequirement {
    public:
      IndexSpace    handle;
      AllocateMode  privilege;
      IndexSpace    parent;
      bool          verified;
    public:
      IndexSpaceRequirement(void);
      IndexSpaceRequirement(IndexSpace _handle,
                            AllocateMode _priv,
                            IndexSpace _parent,
                            bool _verified = false);
    public:
      bool operator<(const IndexSpaceRequirement &req) const;
      bool operator==(const IndexSpaceRequirement &req) const;
    };

    /**
     * \struct FieldSpaceRequirement
     * @deprecated
     * Field space requirements can be used to specify that a task
     * requires additional privileges on a field spaces such as
     * the ability to allocate and free fields.
     *
     * This class is maintained for backwards compatibility with
     * Legion applications written to previous versions of this
     * interface and can safely be ignored for newer programs.
     */
    struct FieldSpaceRequirement {
    public:
      FieldSpace   handle;
      AllocateMode privilege;
      bool         verified;
    public:
      FieldSpaceRequirement(void);
      FieldSpaceRequirement(FieldSpace _handle,
                            AllocateMode _priv,
                            bool _verified = false);
    public:
      bool operator<(const FieldSpaceRequirement &req) const;
      bool operator==(const FieldSpaceRequirement &req) const;
    };

    //==========================================================================
    //                    Operation Launcher Classes
    //==========================================================================

    /**
     * \struct TaskLauncher
     * Task launchers are objects that describe a launch
     * configuration to the runtime.  They can be re-used
     * and safely modified between calls to task launches.
     * @see HighLevelRuntime
     */
    struct TaskLauncher {
    public:
      TaskLauncher(void);
      TaskLauncher(Processor::TaskFuncID tid, 
                   TaskArgument arg,
                   Predicate pred = Predicate::TRUE_PRED,
                   MapperID id = 0,
                   MappingTagID tag = 0);
    public:
      inline void add_index_requirement(const IndexSpaceRequirement &req);
      inline void add_region_requirement(const RegionRequirement &req);
      inline void add_field(unsigned idx, FieldID fid, bool inst = true);
    public:
      inline void add_future(Future f);
      inline void add_grant(Grant g);
      inline void add_wait_barrier(PhaseBarrier bar);
      inline void add_arrival_barrier(PhaseBarrier bar);
    public:
      Processor::TaskFuncID              task_id;
      std::vector<IndexSpaceRequirement> index_requirements;
      std::vector<RegionRequirement>     region_requirements;
      std::vector<Future>                futures;
      std::vector<Grant>                 grants;
      std::vector<PhaseBarrier>          wait_barriers;
      std::vector<PhaseBarrier>          arrive_barriers;
      TaskArgument                       argument;
      Predicate                          predicate;
      MapperID                           map_id;
      MappingTagID                       tag;
      DomainPoint                        point;
    };

    /**
     * \struct IndexLauncher
     * Index launchers are objects that describe the launch
     * of an index space of tasks to the runtime.  They can
     * be re-used and safely modified between calls to 
     * index space launches.
     * @see HighLevelRuntime
     */
    struct IndexLauncher {
    public:
      IndexLauncher(void);
      IndexLauncher(Processor::TaskFuncID tid,
                    Domain domain,
                    TaskArgument global_arg,
                    ArgumentMap map,
                    Predicate pred = Predicate::TRUE_PRED,
                    bool must = false,
                    MapperID id = 0,
                    MappingTagID tag = 0);
    public:
      inline void add_index_requirement(const IndexSpaceRequirement &req);
      inline void add_region_requirement(const RegionRequirement &req);
      inline void add_field(unsigned idx, FieldID fid, bool inst = true);
    public:
      inline void add_future(Future f);
      inline void add_grant(Grant g);
      inline void add_wait_barrier(PhaseBarrier bar);
      inline void add_arrival_barrier(PhaseBarrier bar);
    public:
      Processor::TaskFuncID              task_id;
      Domain                             launch_domain;
      std::vector<IndexSpaceRequirement> index_requirements;
      std::vector<RegionRequirement>     region_requirements;
      std::vector<Future>                futures;
      std::vector<Grant>                 grants;
      std::vector<PhaseBarrier>          wait_barriers;
      std::vector<PhaseBarrier>          arrive_barriers;
      TaskArgument                       global_arg;
      ArgumentMap                        argument_map;
      Predicate                          predicate;
      bool                               must_parallelism;
      MapperID                           map_id;
      MappingTagID                       tag;
    };

    /**
     * \struct InlineLauncher
     * Inline launchers are objects that describe the launch
     * of an inline mapping operation to the runtime.  They
     * can be re-used and safely modified between calls to
     * inline mapping operations.
     * @see HighLevelRuntime
     */
    struct InlineLauncher {
    public:
      InlineLauncher(void);
      InlineLauncher(const RegionRequirement &req,
                     MapperID id = 0,
                     MappingTagID tag = 0);
    public:
      inline void add_field(FieldID fid, bool inst = true);
    public:
      RegionRequirement               requirement;
      MapperID                        map_id;
      MappingTagID                    tag;
    };

    /**
     * \struct CopyLauncher
     * Copy launchers are objects that can be used to issue
     * copies between two regions including regions that are
     * not of the same region tree.  Copy operations specify
     * an arbitrary number of pairs of source and destination 
     * region requirements.  Each region requirement must have 
     * exactly one field with fields of a pair being the same
     * size (unless a reduction copy is being performed).  The
     * source region requirements must be READ_ONLY, while the
     * destination requirements must be either READ_WRITE,
     * WRITE_ONLY, or REDUCE with a reduction function.  While
     * the regions in a source and a destination pair do not have
     * to be in the same region tree, they must share an index space
     * tree.  Furthermore, the index space of the source region
     * requirement will determine the data copied, and therefore
     * the source index space must be the same or a sub-space of
     * the index space of the destination region.
     * @see HighLevelRuntime
     */
    struct CopyLauncher {
    public:
      CopyLauncher(Predicate pred = Predicate::TRUE_PRED,
                   MapperID id = 0, MappingTagID tag = 0);
    public:
      inline void add_copy_requirements(const RegionRequirement &src,
					const RegionRequirement &dst);
      inline void add_src_field(unsigned idx, FieldID fid, bool inst = true);
      inline void add_dst_field(unsigned idx, FieldID fid, bool inst = true);
    public:
      inline void add_grant(Grant g);
      inline void add_wait_barrier(PhaseBarrier bar);
      inline void add_arrival_barrier(PhaseBarrier bar);
    public:
      std::vector<RegionRequirement>  src_requirements;
      std::vector<RegionRequirement>  dst_requirements;
      std::vector<Grant>              grants;
      std::vector<PhaseBarrier>       wait_barriers;
      std::vector<PhaseBarrier>       arrive_barriers;
      Predicate                       predicate;
      MapperID                        map_id;
      MappingTagID                    tag;
    };

    //==========================================================================
    //                          Future Value Classes
    //==========================================================================

    /**
     * \class Future
     * Futures are the objects returned from asynchronous task
     * launches.  Applications can wait on futures to get their values,
     * pass futures as arguments and preconditions to other tasks,
     * or use them to create predicates if they are boolean futures.
     * Futures are lightweight handles that can be passed by value
     * or stored in data structures.  However, futures should not
     * escape the context in which they are created as the runtime
     * garbage collects them after the enclosing task context
     * completes execution.
     *
     * Since futures can be the result of predicated tasks we also
     * provide a mechanism for checking whether the future contains
     * an empty result.  An empty future will be returned for all
     * futures which come from tasks which predicates that resolve
     * to false.
     */
    class Future {
    public:
      Future(void);
      Future(const Future &f);
      ~Future(void);
    public:
      class Impl;
    private:
      Impl *impl;
    protected:
      // Only the runtime should be allowed to make these
      FRIEND_ALL_RUNTIME_CLASSES
      Future(Impl *impl);
    public:
      bool operator==(const Future &f) const
        { return impl == f.impl; }
      bool operator<(const Future &f) const
        { return impl < f.impl; }
      Future& operator=(const Future &f);
    public:
      /**
       * Wait on the result of this future.  Return
       * the value of the future as the specified 
       * template type.
       * @return the value of the future cast as the template type
       */
      template<typename T> inline T get_result(void);
      /**
       * Block until the future completes.
       */
      void get_void_result(void);
      /**
       * Check to see if the future is empty.  The
       * user can specify whether to block and wait
       * for the future to complete first before
       * returning.  If the non-blocking version
       * of the call will return true, until
       * the future actually completes.
       */
      bool is_empty(bool block = false);
    public:
      /**
       * Return a const reference to the future.
       * WARNING: these method is unsafe as the underlying
       * buffer containing the future result can be deleted
       * if the Future handle is lost even a reference
       * to the underlying buffer is maitained.  This
       * scenario can lead to seg-faults.  Use at your
       * own risk.  Note also that this call will not
       * properly deserialize buffers that were serialized
       * with a 'legion_serialize' method.
       */
      template<typename T> inline const T& get_reference(void);
      /**
       * Return an untyped pointer to the 
       * future result.  WARNING: this
       * method is unsafe for the same reasons
       * as get_reference.  It also will not 
       * deserialize anything serialized with a 
       * legion_serialize method.
       */
      inline const void* get_untyped_pointer(void);
    private:
      void* get_untyped_result(void); 
    };

    /**
     * \class FutureMap
     * Future maps are the values returned from asynchronous index space
     * task launches.  Future maps store futures for each of the points
     * in the index space launch.  The application can either wait for
     * a point or choose to extract a future for the given point which 
     * will be filled in when the task for that point completes.
     *
     * Future maps are handles that can be passes by value or stored in
     * data structures.  However, future maps should not escape the
     * context in which they are created as the runtime garbage collects
     * them after the enclosing task context completes execution.
     */
    class FutureMap {
    public:
      FutureMap(void);
      FutureMap(const FutureMap &map);
      ~FutureMap(void);
    private:
      class Impl;
      Impl *impl;
    protected:
      // Only the runtime should be allowed to make these
      FRIEND_ALL_RUNTIME_CLASSES
      FutureMap(Impl *impl);
    public:
      bool operator==(const FutureMap &f) const
        { return impl == f.impl; }
      bool operator<(const FutureMap &f) const
        { return impl < f.impl; }
      FutureMap& operator=(const FutureMap &f);
    public:
      /**
       * Block until we can return the result for the
       * task executing for the given domain point.
       * @param point the point task to wait for
       * @return the return value of the task
       */
      template<typename T>
        inline T get_result(const DomainPoint &point);
      /**
       * Non-blocking call that will return a future that
       * will contain the value from the given index task
       * point when it completes.
       * @param point the point task to wait for
       * @return a future for the index task point
       */
      Future get_future(const DomainPoint &point);
      /**
       * Blocking call that will return one the point
       * in the index space task has executed.
       * @param point the point task to wait for
       */
      void get_void_result(const DomainPoint &point);
    public:
      /**
       * An older method for getting the result of
       * a point in an index space launch that is
       * maintained for backwards compatibility.
       * @param point the index task point to get the return value from
       * @return the return value of the index task point
       */
      template<typename RT, typename PT, unsigned DIM> 
        inline RT get_result(const PT point[DIM]);
      /**
       * An older method for getting a future corresponding
       * to a point in an index task launch.  This call is
       * non-blocking and actually waiting for the task to
       * complete will necessitate waiting on the future.
       * @param point the index task point to get the future for
       * @return a future for the point in the index task launch
       */
      template<typename PT, unsigned DIM>
        inline Future get_future(const PT point[DIM]);
      /**
       * An older method for performing a blocking wait
       * for a point in an index task launch.
       * @param point the point in the index task launch to wait for
       */
      template<typename PT, unsigned DIM>
        inline void get_void_result(const PT point[DIM]);
    public:
      /**
       * Wait for all the tasks in the index space launch of
       * tasks to complete before returning.
       */
      void wait_all_results(void); 
    }; 

    //==========================================================================
    //                          Physical Data Classes
    //==========================================================================

    /**
     * \class PhysicalRegion
     * Physical region objects are used to manage access to the
     * physical instances that hold data.  They are lightweight
     * handles that can be stored in data structures and passed
     * by value.  They should never escape the context in which
     * they are created.
     */
    class PhysicalRegion {
    public:
      PhysicalRegion(void);
      PhysicalRegion(const PhysicalRegion &rhs);
      ~PhysicalRegion(void);
    private:
      class Impl;
      Impl *impl;
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      PhysicalRegion(Impl *impl);
    public:
      PhysicalRegion& operator=(const PhysicalRegion &rhs);
      inline bool operator==(const PhysicalRegion &reg) const
        { return (impl == reg.impl); }
      inline bool operator<(const PhysicalRegion &reg) const
        { return (impl < reg.impl); }
    public:
      /**
       * Check to see if this represents a mapped physical region. 
       */
      inline bool is_mapped(void) const;
      /**
       * For physical regions returned as the result of an
       * inline mapping, this call will block until the physical
       * instance has a valid copy of the data.
       */
      void wait_until_valid(void);
      /**
       * For physical regions returned from inline mappings,
       * this call will query if the instance contains valid
       * data yet.
       * @return whether the region has valid data
       */
      bool is_valid(void) const;
      /**
       * @return the logical region for this physical region
       */
      LogicalRegion get_logical_region(void) const;
      /**
       * Return a generic accessor for the entire physical region.
       */
      Accessor::RegionAccessor<Accessor::AccessorType::Generic> 
        get_accessor(void) const;
      /**
       * Return a field accessor for a specific field within the region.
       */
      Accessor::RegionAccessor<Accessor::AccessorType::Generic> 
        get_field_accessor(FieldID field) const; 
    };
 
    /**
     * \class IndexIterator
     * This is a helper class for iterating over the points within
     * an index space or the index space of a given logical region.
     * It should never be copied and will assert fail if a copy is
     * made of it.
     */
    class IndexIterator {
    public:
      explicit IndexIterator(IndexSpace space);
      explicit IndexIterator(LogicalRegion handle);
      IndexIterator(const IndexIterator &rhs);
      ~IndexIterator(void);
    public:
      /**
       * Check to see if the iterator has a next point
       */
      inline bool has_next(void) const;
      /**
       * Get the current point in the iterator.  Advances
       * the iterator to the next point.
       */
      inline ptr_t next(void);
    public:
      IndexIterator& operator=(const IndexIterator &rhs);
    private:
      Enumerator *const enumerator;
      bool finished;
      int current_pointer;
      int remaining_elmts;
    };

    //==========================================================================
    //                      Software Coherence Classes
    //==========================================================================

    /**
     * \struct AcquireLauncher
     * An AcquireLauncher is a class that is used for supporting user-level
     * software coherence when working with logical regions held in 
     * simultaneous coherence mode.  By default simultaneous mode requires
     * all users to use the same physical instance.  By acquiring coherence
     * on the physical region, a parent task can launch sub-tasks which
     * are not required to use the same physical instance.  Synchronization
     * primitives are allowed to specify what must occur before the
     * acquire operation is performed.
     */
    struct AcquireLauncher {
    public:
      AcquireLauncher(LogicalRegion logical_region, 
                      LogicalRegion parent_region,
                      PhysicalRegion physical_region,
                      Predicate pred = Predicate::TRUE_PRED,
                      MapperID id = 0, MappingTagID tag = 0);
    public:
      inline void add_field(FieldID f);
      inline void add_grant(Grant g);
      inline void add_wait_barrier(PhaseBarrier pb);
      inline void add_arrival_barrier(PhaseBarrier pb);
    public:
      LogicalRegion                   logical_region;
      LogicalRegion                   parent_region;
      std::set<FieldID>               fields;
    public:
      PhysicalRegion                  physical_region;
    public:
      std::vector<Grant>              grants;
      std::vector<PhaseBarrier>       wait_barriers;
      std::vector<PhaseBarrier>       arrive_barriers;
      Predicate                       predicate;
      MapperID                        map_id;
      MappingTagID                    tag;
    };

    /**
     * \struct ReleaseLauncher
     * A ReleaseLauncher supports the complementary operation to acquire
     * for performing user-level software coherence when dealing with
     * regions in simultaneous coherence mode.  
     */
    struct ReleaseLauncher {
    public:
      ReleaseLauncher(LogicalRegion logical_region, 
                      LogicalRegion parent_region,
                      PhysicalRegion physical_region,
                      Predicate pred = Predicate::TRUE_PRED,
                      MapperID id = 0, MappingTagID tag = 0);
    public:
      inline void add_field(FieldID f);
      inline void add_grant(Grant g);
      inline void add_wait_barrier(PhaseBarrier pb);
      inline void add_arrival_barrier(PhaseBarrier pb);
    public:
      LogicalRegion                   logical_region;
      LogicalRegion                   parent_region;
      std::set<FieldID>               fields;
    public:
      PhysicalRegion                  physical_region;
    public:
      std::vector<Grant>              grants;
      std::vector<PhaseBarrier>       wait_barriers;
      std::vector<PhaseBarrier>       arrive_barriers;
      Predicate                       predicate;
      MapperID                        map_id;
      MappingTagID                    tag;
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
      Processor::TaskFuncID task_id; 
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
      bool                                inline_task;
      bool                                spawn_task;
      bool                                map_locally;
      bool                                profile_task;
      TaskPriority                        task_priority;
    public:
      // Profiling information for the task
      unsigned long long                  start_time;
      unsigned long long                  stop_time;
      unsigned long long                  complete_time;
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

      const std::map<VariantID,Variant>& get_all_variants(void) const { return variants; }
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
      Mapper(void) { }
      virtual ~Mapper(void) { }
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
       * valid data and will not actually be mapped.
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
       * @param task the task that generated the future for this predicate
       * @param spec_value the speculative value to be
       *    set if the mapper is going to speculate
       * @return true if the mapper is speculating, false otherwise
       */
      virtual bool speculate_on_predicate(const Task *task,
                                          bool &spec_value) = 0;
    };

    //==========================================================================
    //                           Runtime Classes
    //==========================================================================

    /**
     * \struct ColoredPoints
     * Colored points struct for describing colorings.
     */
    template<typename T>
    struct ColoredPoints {
    public:
      std::set<T> points;
      std::set<std::pair<T,T> > ranges;
    };

    /**
     * \struct InputArgs
     * Input arguments helper struct for passing in
     * the command line arguments to the runtime.
     */
    struct InputArgs {
    public:
      char **argv;
      int argc;
    };

    /**
     * \struct TaskConfigOptions
     * A class for describing the configuration options
     * for a task being registered with the runtime.  
     * Leaf tasks must not contain any calls to the runtime.
     * Inner tasks must never touch any of the data for 
     * which they have privileges which is identical to
     * the Sequoia definition of an inner task.
     * Idempotent tasks must have no side-effects outside
     * of the kind that Legion can analyze (i.e. writing
     * regions).
     */
    struct TaskConfigOptions {
    public:
      TaskConfigOptions(bool leaf = false,
                        bool inner = false,
                        bool idempotent = false);
    public:
      bool leaf;
      bool inner;
      bool idempotent;
    };

    /**
     * \class HighLevelRuntime
     * The HighLevelRuntime class is the primary interface for
     * Legion.  Every task is given a reference to the runtime as
     * part of its arguments.  All Legion operations are then
     * performed by directing the runtime to perform them through
     * the methods of this class.  The methods in HighLevelRuntime
     * are broken into three categories.  The first group of
     * calls are the methods that can be used by application
     * tasks during runtime.  The second group contains calls
     * for initializing the runtime during start-up callback.
     * The final section of calls are static methods that are
     * used to configure the runtime prior to starting it up.
     */
    class HighLevelRuntime {
    protected:
      // The HighLevelRuntime bootstraps itself and should
      // never need to be explicitly created.
      friend class Runtime;
      HighLevelRuntime(Runtime *rt);
    public:
      //------------------------------------------------------------------------
      // Index Space Operations
      //------------------------------------------------------------------------
      /**
       * Create a new index space
       * @param ctx the enclosing task context
       * @param max_num_elmts maximum number of elements in the index space
       * @return the handle for the new index space
       */
      IndexSpace create_index_space(Context ctx, size_t max_num_elmts);
      /**
       * Create a new index space based on a domain
       * @param ctx the enclosing task context
       * @param domain the domain for the new index space
       * @return the handle for the new index space
       */
      IndexSpace create_index_space(Context ctx, Domain domain);
      /**
       * Destroy an existing index space
       * @param ctx the enclosing task context
       * @param handle the index space to destroy
       */
      void destroy_index_space(Context ctx, IndexSpace handle);
    public:
      //------------------------------------------------------------------------
      // Index Partition Operations
      //------------------------------------------------------------------------
      /**
       * Create an index partition.
       * @param ctx the enclosing task context
       * @param parent index space being partitioned
       * @param coloring the coloring of the parent index space
       * @param disjoint whether the partitioning is disjoint or not
       * @param part_color optional color name for the partition
       * @return handle for the next index partition
       */
      IndexPartition create_index_partition(Context ctx, IndexSpace parent, 
                                            const Coloring &coloring, 
                                            bool disjoint, 
                                            int part_color = -1);

      /**
       * Create an index partition from a domain color space and coloring.
       * @param ctx the enclosing task context
       * @param parent index space being partitioned
       * @param color_space the domain of colors 
       * @param coloring the domain coloring of the parent index space
       * @param disjoint whether the partitioning is disjoint or not
       * @param part_color optional color name for the partition
       * @return handle for the next index partition
       */
      IndexPartition create_index_partition(Context ctx, IndexSpace parent, 
					    Domain color_space, 
                                            const DomainColoring &coloring,
					    bool disjoint,
                                            int part_color = -1);

      /**
       * Create an index partitioning from a typed mapping.
       * @param ctx the enclosing task context
       * @param parent index space being partitioned
       * @param mapping the mapping of points to colors
       * @param part_color optional color name for the partition
       * @return handle for the next index partition
       */
      template <typename T>
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
					    const T& mapping,
					    int part_color = -1);

      /**
       * Create an index partitioning from an existing field
       * in a physical instance.  This requires that the field
       * accessor be valid for the entire parent index space.  By definition
       * colors are always non-negative.  The runtime will iterate over the
       * field accessor and interpret values as signed integers.  Any
       * locations less than zero will be ignored.  Values greater than or
       * equal to zero will be colored and placed in the appropriate
       * subregion.  By definition this partitioning mechanism has to 
       * disjoint since each pointer value has at most one color.
       * @param ctx the enclosing task context
       * @param field_accessor field accessor for the coloring field
       * @param disjoint whether the partitioning is disjoint or not
       * @param complete whether the partitioning is complete or not
       * @return handle for the next index partition
       */
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
       Accessor::RegionAccessor<Accessor::AccessorType::Generic> field_accessor, 
                                            int part_color = -1);

      /**
       * Destroy an index partition
       * @param ctx the enclosing task context
       * @param handle index partition to be destroyed
       */
      void destroy_index_partition(Context ctx, IndexPartition handle);
    public:
      //------------------------------------------------------------------------
      // Index Tree Traversal Operations
      //------------------------------------------------------------------------
      /**
       * Return the index partitioning of an index space 
       * with the assigned color.
       * @param ctx enclosing task context
       * @param parent index space
       * @param color of index partition
       * @return handle for the index partition with the specified color
       */
      IndexPartition get_index_partition(Context ctx, IndexSpace parent, 
                                         Color color);

      /**
       * Return the index subspace of an index partitioning
       * with the specified color.
       * @param ctx enclosing task context
       * @param p parent index partitioning
       * @param color of the index sub-space
       * @return handle for the index space with the specified color
       */
      IndexSpace get_index_subspace(Context ctx, IndexPartition p, 
                                    Color color); 

      /**
       * Return the domain corresponding to the
       * specified index space if it exists
       * @param ctx enclosing task context
       * @param handle index space handle
       * @return the domain corresponding to the index space
       */
      Domain get_index_space_domain(Context ctx, IndexSpace handle);

      /**
       * Return a domain that represents the color space
       * for the specified partition.
       * @param ctx enclosing task context
       * @param p handle for the index partition
       * @return a domain for the color space of the specified partition
       */
      Domain get_index_partition_color_space(Context ctx, IndexPartition p);

      /**
       * Return a set that contains the colors of all
       * the partitions of the index space.  It is unlikely
       * the colors are numerically dense which precipitates
       * the need for a set.
       * @param ctx enclosing task context
       * @param sp handle for the index space
       * @param colors reference to the set object in which to place the colors
       */
      void get_index_space_partition_colors(Context ctx, IndexSpace sp,
                                            std::set<Color> &colors);

      /**
       * Return whether a given index partition is disjoint
       * @param ctx enclosing task context
       * @param p index partition handle
       * @return whether the index partition is disjoint
       */
      bool is_index_partition_disjoint(Context ctx, IndexPartition p);

      /**
       * Get an index subspace from a partition with a given
       * color point.
       * @param ctx enclosing task context
       * @param p parent index partition handle
       * @param color_point point containing color value of index subspace
       * @return the corresponding index space to the specified color point
       */
      template <unsigned DIM>
      IndexSpace get_index_subspace(Context ctx, IndexPartition p, 
                                    Arrays::Point<DIM> color_point);

      /**
       * Return the color for the corresponding index space in
       * its member partition.  If it is a top-level index space
       * then zero will be returned.
       * @param ctx enclosing task context
       * @param handle the index space for which to find the color
       * @return the color for the index space
       */
      Color get_index_space_color(Context ctx, IndexSpace handle);

      /**
       * Return the color for the corresponding index partition in
       * in relation to its parent logical region.
       * @param ctx enclosing task context
       * @param handle the index partition for which to find the color
       * @return the color for the index partition
       */
      Color get_index_partition_color(Context ctx, IndexPartition handle);
    public:
      //------------------------------------------------------------------------
      // Safe Cast Operations
      //------------------------------------------------------------------------
      /**
       * Safe cast a pointer down to a target region.  If the pointer
       * is not in the target region, then a nil pointer is returned.
       * @param ctx enclosing task context
       * @param pointer the pointer to be case
       * @param region the target logical region
       * @return the same pointer if it can be safely cast, otherwise nil
       */
      ptr_t safe_cast(Context ctx, ptr_t pointer, LogicalRegion region);

      /**
       * Safe case a domain point down to a target region.  If the point
       * is not in the target region, then an empty domain point
       * is returned.
       * @param ctx enclosing task context
       * @param point the domain point to be cast
       * @param region the target logical region
       * @return the same point if it can be safely cast, otherwise empty
       */
      DomainPoint safe_cast(Context ctx, DomainPoint point, 
                            LogicalRegion region);
    public:
      //------------------------------------------------------------------------
      // Field Space Operations
      //------------------------------------------------------------------------
      /**
       * Create a new field space.
       * @param ctx enclosing task context
       * @return handle for the new field space
       */
      FieldSpace create_field_space(Context ctx);
      /**
       * Destroy an existing field space.
       * @param ctx enclosing task context
       * @param handle of the field space to be destroyed
       */
      void destroy_field_space(Context ctx, FieldSpace handle);
    public:
      //------------------------------------------------------------------------
      // Logical Region Operations
      //------------------------------------------------------------------------
      /**
       * Create a new logical region tree from the given index
       * space and field space.  It is important to note that every
       * call to this function will return a new logical region with
       * a new tree ID.  Only the triple of an index space, a field
       * space, and a tree ID uniquely define a logical region.
       * @param ctx enclosing task context
       * @param index handle for the index space of the logical region
       * @param fields handle for the field space of the logical region
       * @return handle for the logical region created
       */
      LogicalRegion create_logical_region(Context ctx, IndexSpace index, 
                                          FieldSpace fields);
      /**
       * Destroy a logical region and all of its logical sub-regions.
       * @param ctx enclosing task context
       * @param handle logical region handle to destroy
       */
      void destroy_logical_region(Context ctx, LogicalRegion handle);

      /**
       * Destroy a logical partition and all of it is logical sub-regions.
       * @param ctx enclosing task context
       * @param handle logical partition handle to destroy
       */
      void destroy_logical_partition(Context ctx, LogicalPartition handle);
    public:
      //------------------------------------------------------------------------
      // Logical Region Tree Traversal Operations
      //------------------------------------------------------------------------
      /**
       * Return the logical partition instance of the given index partition
       * in the region tree for the parent logical region.
       * @param ctx enclosing task context
       * @param parent the logical region parent
       * @param handle index partition handle
       * @return corresponding logical partition in the same tree 
       *    as the parent region
       */
      LogicalPartition get_logical_partition(Context ctx, LogicalRegion parent, 
                                             IndexPartition handle);
      
      /**
       * Return the logical partition of the logical region parent with
       * the specified color.
       * @param ctx enclosing task context
       * @param parent logical region
       * @param color for the specified logical partition
       * @return the logical partition for the specified color
       */
      LogicalPartition get_logical_partition_by_color(Context ctx, 
                                                      LogicalRegion parent, 
                                                      Color c);
      
      /**
       * Return the logical partition identified by the triple of index
       * partition, field space, and region tree ID.
       * @param ctx enclosing task context
       * @param handle index partition handle
       * @param fspace field space handle
       * @param tid region tree ID
       * @return the corresponding logical partition
       */
      LogicalPartition get_logical_partition_by_tree(Context ctx, 
                                                     IndexPartition handle, 
                                                     FieldSpace fspace, 
                                                     RegionTreeID tid); 

      /**
       * Return the logical region instance of the given index space 
       * in the region tree for the parent logical partition.
       * @param ctx enclosing task context
       * @param parent the logical partition parent
       * @param handle index space handle
       * @return corresponding logical region in the same tree 
       *    as the parent partition 
       */
      LogicalRegion get_logical_subregion(Context ctx, LogicalPartition parent, 
                                          IndexSpace handle);

      /**
       * Return the logical region of the logical partition parent with
       * the specified color.
       * @param ctx enclosing task context
       * @param parent logical partition 
       * @param color for the specified logical region 
       * @return the logical region for the specified color
       */
      LogicalRegion get_logical_subregion_by_color(Context ctx, 
                                                   LogicalPartition parent, 
                                                   Color c);

      /**
       * Return the logical partition identified by the triple of index
       * space, field space, and region tree ID.
       * @param ctx enclosing task context
       * @param handle index space handle
       * @param fspace field space handle
       * @param tid region tree ID
       * @return the corresponding logical region
       */
      LogicalRegion get_logical_subregion_by_tree(Context ctx, 
                                                  IndexSpace handle, 
                                                  FieldSpace fspace, 
                                                  RegionTreeID tid);

      /**
       * Return the color for the logical region corresponding to
       * its location in the parent partition.  If the region is a
       * top-level region then zero is returned.
       * @param ctx enclosing task context
       * @param handle the logical region for which to find the color
       * @return the color for the logical region
       */
      Color get_logical_region_color(Context ctx, LogicalRegion handle);

      /**
       * Return the color for the logical partition corresponding to
       * its location relative to the parent logical region.
       * @param ctx enclosing task context
       * @param handle the logical partition handle for which to find the color
       * @return the color for the logical partition
       */
      Color get_logical_partition_color(Context ctx, LogicalPartition handle);
    public:
      //------------------------------------------------------------------------
      // Allocator and Argument Map Operations 
      //------------------------------------------------------------------------
      /**
       * Create an index allocator object for a given index space
       * @param ctx enclosing task context
       * @param handle for the index space to create an allocator
       * @return a new index space allocator for the given index space
       */
      IndexAllocator create_index_allocator(Context ctx, IndexSpace handle);

      /**
       * Create a field space allocator object for the given field space
       * @param ctx enclosing task context
       * @param handle for the field space to create an allocator
       * @return a new field space allocator for the given field space
       */
      FieldAllocator create_field_allocator(Context ctx, FieldSpace handle);

      /**
       * @deprectated
       * Create an argument map in the given context.  This method
       * is deprecated as argument maps can now be created directly
       * by a simple declaration.
       * @param ctx enclosing task context
       * @return a new argument map
       */
      ArgumentMap create_argument_map(Context ctx);
    public:
      //------------------------------------------------------------------------
      // Task Launch Operations
      //------------------------------------------------------------------------
      /**
       * Launch a single task with arguments specified by
       * the configuration of the task launcher.
       * @see TaskLauncher
       * @param ctx enclosing task context
       * @param launcher the task launcher configuration
       * @return a future for the return value of the task
       */
      Future execute_task(Context ctx, const TaskLauncher &launcher);

      /**
       * Launch an index space of tasks with arguments specified
       * by the index launcher configuration.
       * @see IndexLauncher
       * @param ctx enclosing task context
       * @param launcher the task launcher configuration
       * @return a future map for return values of the points
       *    in the index space of tasks
       */
      FutureMap execute_index_space(Context ctx, const IndexLauncher &launcher);

      /**
       * Launch an index space of tasks with arguments specified
       * by the index launcher configuration.  Reduce all the
       * return values into a single value using the specified
       * reduction operator into a single future value.  The reduction
       * operation must be a foldable reduction.
       * @see IndexLauncher
       * @param ctx enclosing task context
       * @param launcher the task launcher configuration
       * @param redop ID for the reduction op to use for reducing return values
       * @return a future result representing the reduction of
       *    all the return values from the index space of tasks
       */
      Future execute_index_space(Context ctx, const IndexLauncher &launcher,
                                 ReductionOpID redop);

      /**
       * @deprecated
       * An older method for launching a single task maintained for backwards
       * compatibility with older Legion programs.  
       * @param ctx enclosing task context
       * @param task_id the ID of the task to launch
       * @param indexes the index space requirements for the task
       * @param fields the field space requirements for the task
       * @param regions the region requirements for the task
       * @param arg untyped arguments passed by value to the task
       * @param predicate for controlling speculation
       * @param id of the mapper to associate with the task
       * @param tag mapping tag to be passed to any mapping calls
       * @return future representing return value of the task
       */
      Future execute_task(Context ctx, 
                          Processor::TaskFuncID task_id,
                          const std::vector<IndexSpaceRequirement> &indexes,
                          const std::vector<FieldSpaceRequirement> &fields,
                          const std::vector<RegionRequirement> &regions,
                          const TaskArgument &arg, 
                          const Predicate &predicate = Predicate::TRUE_PRED,
                          MapperID id = 0, 
                          MappingTagID tag = 0);

      /**
       * @deprecated
       * An older method for launching an index space of tasks maintained
       * for backwards compatibility with older Legion programs.
       * @param ctx enclosing task context
       * @param task_id the ID of the task to launch
       * @param domain for the set of points in the index space to create
       * @param indexes the index space requirements for the tasks
       * @param fields the field space requirements for the tasks
       * @param regions the region requirements for the tasks
       * @param global_arg untyped arguments passed by value to all tasks
       * @param arg_map argument map containing point arguments for tasks
       * @param predicate for controlling speculation
       * @param must_parallelism are tasks required to be run concurrently
       * @param id of the mapper to associate with the task
       * @param tag mapping tag to be passed to any mapping calls
       * @return future map containing results for all tasks
       */
      FutureMap execute_index_space(Context ctx, 
                          Processor::TaskFuncID task_id,
                          const Domain domain,
                          const std::vector<IndexSpaceRequirement> &indexes,
                          const std::vector<FieldSpaceRequirement> &fields,
                          const std::vector<RegionRequirement> &regions,
                          const TaskArgument &global_arg, 
                          const ArgumentMap &arg_map,
                          const Predicate &predicate = Predicate::TRUE_PRED,
                          bool must_paralleism = false, 
                          MapperID id = 0, 
                          MappingTagID tag = 0);

      /**
       * @deprecated
       * An older method for launching an index space of tasks that reduce
       * all of their values by a reduction operation down to a single
       * future.  Maintained for backwards compatibility with older
       * Legion programs.
       * @param ctx enclosing task context
       * @param task_id the ID of the task to launch
       * @param domain for the set of points in the index space to create
       * @param indexes the index space requirements for the tasks
       * @param fields the field space requirements for the tasks
       * @param regions the region requirements for the tasks
       * @param global_arg untyped arguments passed by value to all tasks
       * @param arg_map argument map containing point arguments for tasks
       * @param reduction operation to be used for reducing return values
       * @param predicate for controlling speculation
       * @param must_parallelism are tasks required to be run concurrently
       * @param id of the mapper to associate with the task
       * @param tag mapping tag to be passed to any mapping calls
       * @return future containing reduced return value of all tasks
       */
      Future execute_index_space(Context ctx, 
                          Processor::TaskFuncID task_id,
                          const Domain domain,
                          const std::vector<IndexSpaceRequirement> &indexes,
                          const std::vector<FieldSpaceRequirement> &fields,
                          const std::vector<RegionRequirement> &regions,
                          const TaskArgument &global_arg, 
                          const ArgumentMap &arg_map,
                          ReductionOpID reduction, 
                          const TaskArgument &initial_value,
                          const Predicate &predicate = Predicate::TRUE_PRED,
                          bool must_parallelism = false, 
                          MapperID id = 0, 
                          MappingTagID tag = 0);
    public:
      //------------------------------------------------------------------------
      // Inline Mapping Operations
      //------------------------------------------------------------------------
      /**
       * Perform an inline mapping operation from the given inline
       * operation configuration.  Note the application must wait for
       * the resulting physical region to become valid before using it.
       * @see InlineLauncher
       * @param ctx enclosing task context
       * @param launcher inline launcher object
       * @return a physical region for the resulting data
       */
      PhysicalRegion map_region(Context ctx, const InlineLauncher &launcher);

      /**
       * Perform an inline mapping operation which returns a physical region
       * object for the requested region requirement.  Note the application 
       * must wait for the resulting physical region to become valid before 
       * using it.
       * @param ctx enclosing task context
       * @param req the region requirement for the inline mapping
       * @param id the mapper ID to associate with the operation
       * @param tag the mapping tag to pass to any mapping calls
       * @return a physical region for the resulting data
       */
      PhysicalRegion map_region(Context ctx, const RegionRequirement &req, 
                                MapperID id = 0, MappingTagID tag = 0);

      /**
       * Perform an inline mapping operation that re-maps a physical region
       * that was initially mapped when the task began.
       * @param ctx enclosing task context
       * @param idx index of the region requirement from the enclosing task
       * @param id the mapper ID to associate with the operation
       * @param the mapping tag to pass to any mapping calls 
       * @return a physical region for the resulting data 
       */
      PhysicalRegion map_region(Context ctx, unsigned idx, 
                                MapperID id = 0, MappingTagID tag = 0);

      /**
       * Remap a region from an existing physical region.  It will
       * still be necessary for the application to wait until the
       * physical region is valid again before using it.
       * @param ctx enclosing task context
       * @param region the physical region to be remapped
       */
      void remap_region(Context ctx, PhysicalRegion region);

      /**
       * Unmap a physical region.  This can unmap either a previous
       * inline mapping physical region or a region initially mapped
       * as part of the task's launch.
       * @param ctx enclosing task context
       * @param region physical region to be unmapped
       */
      void unmap_region(Context ctx, PhysicalRegion region);

      /**
       * Map all the regions originally requested for a context (if they
       * haven't already been mapped).
       * @param ctx enclosing task context
       */
      void map_all_regions(Context ctx);

      /**
       * Unmap all the regions originally requested for a context (if
       * they haven't already been unmapped).
       * @param ctx enclosing task context
       */
      void unmap_all_regions(Context ctx);
    public:
      //------------------------------------------------------------------------
      // Copy Operations
      //------------------------------------------------------------------------
      /**
       * Launch a copy operation from the given configuration of
       * the given copy launcher.  Return a void future that can
       * be used for indicating when the copy has completed.
       * @see CopyLauncher
       * @param ctx enclosing task context
       * @param launcher copy launcher object
       * @return future for when the copy is finished
       */
      void issue_copy_operation(Context ctx, const CopyLauncher &launcher);
    public:
      //------------------------------------------------------------------------
      // Predicate Operations
      //------------------------------------------------------------------------
      /**
       * Create a new predicate value from a future.  The future passed
       * must be a boolean future.
       * @param f future value to convert to a predicate
       * @return predicate value wrapping the future
       */
      Predicate create_predicate(Context ctx, const Future &f);

      /**
       * Create a new predicate value that is the logical 
       * negation of another predicate value.
       * @param p predicate value to logically negate
       * @return predicate value logically negating previous predicate
       */
      Predicate predicate_not(Context ctx, const Predicate &p);

      /**
       * Create a new predicate value that is the logical
       * conjunction of two other predicate values.
       * @param p1 first predicate to logically and 
       * @param p2 second predicate to logically and
       * @return predicate value logically and-ing two predicates
       */
      Predicate predicate_and(Context ctx, const Predicate &p1, 
                                           const Predicate &p2);

      /**
       * Create a new predicate value that is the logical
       * disjunction of two other predicate values.
       * @param p1 first predicate to logically or
       * @param p2 second predicate to logically or
       * @return predicate value logically or-ing two predicates
       */
      Predicate predicate_or(Context ctx, const Predicate &p1, 
                                          const Predicate &p2);
    public:
      //------------------------------------------------------------------------
      // Lock Operations
      //------------------------------------------------------------------------
      /**
       * Create a new lock.
       * @param ctx enclosing task context
       * @return a new lock handle
       */
      Lock create_lock(Context ctx);

      /**
       * Destroy a lock.  This operation will
       * defer the lock destruction until the
       * completion of the task in which the destruction
       * is performed so the user does not need to worry
       * about races with child operations which may
       * be using the lock.
       * @param ctx enclosing task context
       * @param r lock to be destroyed
       */
      void destroy_lock(Context ctx, Lock l);

      /**
       * Acquire one or more locks in the given mode.  Returns
       * a grant object which can be passed to many kinds
       * of launchers for indicating that the operations
       * must be performed while the grant his held.
       * Note that the locks will be acquired in the order specified
       * by the in the vector which may be necessary for
       * applications to avoid deadlock.
       * @param ctx the enclosing task context
       * @param requests vector of locks to acquire
       * @return a grant object
       */
      Grant acquire_grant(Context ctx, 
                          const std::vector<LockRequest> &requests);

      /**
       * Release the grant object indicating that no more
       * operations will be launched that require the 
       * grant object.  Once this is done and all the tasks
       * using the grant complete the runtime can release
       * the lock.
       * @param ctx the enclosing task context
       * @param grant the grant object to release
       */
      void release_grant(Context ctx, Grant grant);
    public:
      //------------------------------------------------------------------------
      // Phase Barrier operations
      //------------------------------------------------------------------------
      /**
       * Create a new phase barrier with a given number of 
       * participants.  Note that this number of participants
       * is only the number of participants that are advancing the
       * phases from one to the next.  An unlimited number of tasks
       * can be run by each participant within a phase.  Furthermore
       * all the participants can run an unlimited number of phases
       * provided that they all run the same number.
       * @param ctx enclosing task context
       * @param participants number of participant tasks
       *    that will be performing calls to advance the barrier
       * @return a new phase barrier handle
       */
      PhaseBarrier create_phase_barrier(Context ctx, unsigned participants);

      /**
       * Destroy a phase barrier.  This operation will 
       * defer the phase barrier destruciton until the
       * completion of the task in which in the destruction
       * is performed so the the user does not need to
       * worry about races with child operations which
       * may still be using the phase barrier.
       * @param ctx enclosing task context
       * @param pb phase barrier to be destroyed
       */
      void destroy_phase_barrier(Context ctx, PhaseBarrier pb);

      /**
       * Advance an existing phase barrier to the next
       * phase.  Note this is NOT arrive which indicates
       * an actual arrival at the next phase.  Instead this
       * allows tasks launched with the returned phase
       * barrier to indicate that they should be executed
       * in the next phase of the computation.
       * @param ctx enclosing task context
       * @param pb the phase barrier to be advanced
       * @return an updated phase barrier used for the next phase
       */
      PhaseBarrier advance_phase_barrier(Context ctx, PhaseBarrier pb);
    public:
      //------------------------------------------------------------------------
      // User-Managed Software Coherence 
      //------------------------------------------------------------------------
      /**
       * Issue an acquire operation on the specified physical region
       * provided by the acquire launcher.  This call should be matched
       * by a release call later in the same context on the same 
       * physical region.
       */
      void issue_acquire(Context ctx, const AcquireLauncher &launcher);

      /**
       * Issue a release operation on the specified physical region
       * provided by the release launcher.  This call should be preceded
       * by an acquire call earlier in teh same context on the same
       * physical region.
       */
      void issue_release(Context ctx, const ReleaseLauncher &launcher);
    public:
      //------------------------------------------------------------------------
      // Fence Operations 
      //------------------------------------------------------------------------
      /**
       * Issue a Legion mapping fence in the current context.  A 
       * Legion mapping fence guarantees that all of the tasks issued
       * in the context prior to the fence will finish mapping
       * before the tasks after the fence begin to map.  This can be
       * useful as a performance optimization to minimize the
       * number of mapping independence tests required.
       */
      void issue_mapping_fence(Context ctx);

      /**
       * Issue a Legion execution fence in the current context.  A 
       * Legion execution fence guarantees that all of the tasks issued
       * in the context prior to the fence will finish running
       * before the tasks after the fence begin to map.  This 
       * will allow the necessary propagation of Legion meta-data
       * such as modifications to the region tree made prior
       * to the fence visible to tasks issued after the fence.
       */
      void issue_execution_fence(Context ctx); 
    public:
      //------------------------------------------------------------------------
      // Tracing Operations 
      //------------------------------------------------------------------------
      /**
       * Start a new trace of legion operations. Tracing enables
       * the runtime to memoize the dynamic logical dependence
       * analysis for these operations.  Future executions of
       * the trace will no longer need to perform the dynamic
       * dependence analysis, reducing overheads and improving
       * the parallelism available in the physical analysis.
       * The trace ID need only be local to the enclosing context.
       * Traces are currently not permitted to be nested.
       */
      void begin_trace(Context ctx, TraceID tid);
      /**
       * Mark the end of trace that was being performed.
       */
      void end_trace(Context ctx, TraceID tid);
    public:
      //------------------------------------------------------------------------
      // Miscellaneous Operations
      //------------------------------------------------------------------------
      /**
       * Retrieve the mapper at the given mapper ID associated
       * with the processor in which this task is executing.  This
       * call allows applications to make calls into mappers that
       * they have created to inform that mapper of important
       * application level information.
       * @param ctx the enclosing task context
       * @param id the mapper ID for which mapper to locate
       * @return a pointer to the specified mapper object
       */
      Mapper* get_mapper(Context ctx, MapperID id);
      
      /**
       * Return the processor on which the current task is
       * being executed.
       * @param ctx enclosing task context
       * @return the processor on which the task is running
       */
      Processor get_executing_processor(Context ctx);

      /**
       * Indicate that data in a particular physical region
       * appears to be incorrect for whatever reason.  This
       * will cause the runtime to trap into an error handler
       * and may result in the task being re-executed if the 
       * fault is determined to be recoverable.  Control
       * will never return from this call.  The application can also
       * indicate whether it believes that this particular instance
       * is invalid (nuclear=false) or whether it believes that all
       * instances contain invalid data (nuclear=true).  If all 
       * instances are bad the runtime will nuke all copies of the
       * data and restart the tasks necessary to generate them.
       * @param ctx enclosing task context
       * @param region physical region which contains bad data
       * @param nuclear whether the single instance is invalid or all are
       */
      void raise_region_exception(Context ctx, PhysicalRegion region, 
                                  bool nuclear);
    public:
      //------------------------------------------------------------------------
      // Registration Callback Operations
      //------------------------------------------------------------------------
      /**
       * Add a mapper at the given mapper ID for the runtime
       * to use when mapping tasks.  Note that this call should
       * only be used in the mapper registration callback function
       * that occurs before tasks begin executing.  It can have
       * undefined results if used during regular runtime execution.
       * @param map_id the mapper ID to associate with the mapper
       * @param mapper pointer to the mapper object
       * @param proc the processor to associate the mapper with
       */
      void add_mapper(MapperID map_id, Mapper *mapper, Processor proc);
      
      /**
       * Replace the default mapper for a given processor with
       * a new mapper.  Note that this call should only be used
       * in the mapper registration callback function that occurs
       * before tasks begin executing.  It can have undefined
       * results if used during regular runtime execution.
       * @param mapper pointer to the mapper object to use
       *    as the new default mapper
       * @param proc the processor to associate the mapper with
       */
      void replace_default_mapper(Mapper *mapper, Processor proc);
    public:
      //------------------------------------------------------------------------
      // Start-up Operations
      // Everything below here is a static function that is used to configure
      // the runtime prior to calling the start method to begin execution.
      //------------------------------------------------------------------------
    public:
      /**
       * After configuring the runtime object this method should be called
       * to start the runtime running.  The runtime will then launch
       * the specified top-level task on one of the processors in the
       * machine.  Note if background is set to false, control will
       * never return from this call.  An integer is returned since
       * this is routinely the last call at the end of 'main' in a
       * program and it is nice to return an integer from 'main' to
       * satisfy compiler type checkers.
       *
       * In addition to the arguments passed to the application, there
       * are also several flags that can be passed to the runtime
       * to control execution.
       * 
       * -------------
       *  Stealing
       * -------------
       * -hl:nosteal  Disable any stealing in the runtime.  The runtime
       *              will never query any mapper about stealing.
       * ------------------------
       *  Out-of-order Execution
       * ------------------------
       * -hl:window <int> Specify the maximum number of child tasks
       *              allowed in a given task context at a time.  A call
       *              to launch more tasks than the allotted window
       *              will stall the parent task until child tasks
       *              begin completing.  The default is 1024.
       * -hl:sched <int> The run-ahead factor for the runtime.  How many
       *              outstanding tasks ready to run should be on each
       *              processor before backing off the mapping procedure.
       * -hl:width <int> Scheduling granularity when handling dependence
       *              analysis and issuing operations.  Effectively the
       *              Legion runtime superscalar width.
       * -hl:outorder Execute operations out-of-order when the runtime
       *              has been compiled with the macro INORDER_EXECUTION.
       *              By default when compiling with INORDER_EXECUTION
       *              all applications will run in program order.
       * -------------
       *  Messaging
       * -------------
       * -hl:message <int> Maximum size in bytes of the active messages
       *              to be sent between instances of the high-level 
       *              runtime.  This can help avoid the use of expensive
       *              per-pair-of-node RDMA buffers in the low-level
       *              runtime.  Default value is 4K which should guarantee
       *              medium sized active messages on Infiniband clusters.
       * ---------------------
       *  Dependence Analysis
       * ---------------------
       * -hl:filter <int> Maximum number of tasks allowed in logical
       *              or physical epochs.  Default value is 32.
       * -hl:no_dyn   Disable dynamic disjointness tests when the runtime
       *              has been compiled with macro DYNAMIC_TESTS defined
       *              which enables dynamic disjointness testing.
       * ---------------------
       *  Resiliency
       * ---------------------
       * -hl:resilient Enable features that make the runtime resilient
       *              including deferred commit that can be controlled
       *              by the next two flags.  By default this is off
       *              for performance reasons.  Once resiliency mode
       *              is enabled, then the user can control when 
       *              operations commit using the next two flags.
       * -------------
       *  Debugging
       * ------------- 
       * -hl:tree     Dump intermediate physical region tree states before
       *              and after every operation.  The runtime must be
       *              compiled in debug mode with the DEBUG_HIGH_LEVEL
       *              macro defined.
       * -hl:disjointness Verify the specified disjointness of 
       *              partitioning operations.  The runtime must be
       *              compiled with the DEBUG_HIGH_LEVEL macro defined.
       * -hl:separate Indicate that separate instances of the high
       *              level runtime should be made for each processor.
       *              The default is one runtime instance per node.
       *              This is primarily useful for debugging purposes
       *              to force messages to be sent between runtime 
       *              instances on the same node.
       * -------------
       *  Profiling
       * -------------
       * -hl:prof <int> Specify the number of nodes on which to enable
       *              profiling information to be collected.  By default
       *              all nodes are enabled.  Zero will disable all
       *              profiling while each number greater than zero will
       *              profile on that number of nodes.
       *
       * @param argc the number of input arguments
       * @param argv pointer to an array of string arguments of size argc
       * @param background whether to execute the runtime in the background
       * @return only if running in background, otherwise never
       */
      static int start(int argc, char **argv, bool background = false);

      /**
       * Blocking call to wait for the runtime to shutdown when
       * running in background mode.  Otherwise this call should
       * never be used.
       */
      static void wait_for_shutdown(void);
      
      /**
       * Set the top-level task ID for the runtime to use when beginning
       * an application.  This should be set before calling start.  Otherwise
       * the runtime will default to a value of zero.  Note the top-level
       * task must be a single task and not an index space task.
       * @param top_id ID of the top level task to be run
       */
      static void set_top_level_task_id(Processor::TaskFuncID top_id);

      /**
       * Register a reduction operation with the runtime.  Note that the
       * reduction operation template type must conform to the specification
       * for a reduction operation outlined in the low-level runtime 
       * interface.  Reduction operations can be used either for reduction
       * privileges on a region or for performing reduction of values across
       * index space task launches.  The reduction operation ID zero is
       * reserved for runtime use.
       * @param redop_id ID at which to register the reduction operation
       */
      template<typename REDOP>
      static void register_reduction_op(ReductionOpID redop_id);

      /**
       * Return a pointer to a given reduction operation object.
       * @param redop_id ID of the reduction operation to find
       * @return a pointer to the reduction operation object if it exists
       */
      static const ReductionOp* get_reduction_op(ReductionOpID redop_id);

      /**
       * Register a region projection function that can be used to map
       * from an upper bound of a logical region down to a specific
       * logical sub-region for a given domain point during index
       * task execution.  The projection ID zero is reserved for runtime
       * use. The user can pass in AUTO_GENERATE_ID to
       * have the runtime automatically generate an ID.
       * @param handle the projection ID to register the function at
       * @return ID where the function was registered
       */
      template<LogicalRegion (*PROJ_PTR)(LogicalRegion, const DomainPoint&,
                                         HighLevelRuntime*)>
      static ProjectionID register_region_function(ProjectionID handle);

      /**
       * Register a partition projection function that can be used to
       * map from an upper bound of a logical partition down to a specific
       * logical sub-region for a given domain point during index task
       * execution.  The projection ID zero is reserved for runtime use.
       * The user can pass in AUTO_GENERATE_ID to have the runtime
       * automatically generate an ID.
       * @param handle the projection ID to register the function at
       * @return ID where the function was registered
       */
      template<LogicalRegion (*PROJ_PTR)(LogicalPartition, const DomainPoint&,
                                         HighLevelRuntime*)>
      static ProjectionID register_partition_function(ProjectionID handle);
    public:
      /**
       * This call allows the application to register a callback function
       * that will be run prior to beginning any task execution on every
       * runtime in the system.  It can be used to register or update the
       * mapping between mapper IDs and mappers, register reductions,
       * register projection function, register coloring functions, or
       * configure any other static runtime variables prior to beginning
       * the application.
       * @param callback function pointer to the callback function to be run
       */
      static void set_registration_callback(RegistrationCallbackFnptr callback);

      /**
       * This method can be used to retrieve the default arguments passed into
       * the runtime at the start call from any point in the machine.
       * @return a reference to the input arguments passed in at start-up
       */
      static const InputArgs& get_input_args(void);
    public:
      /**
       * Enable recording of profiling information.
       */
      static void enable_profiling(void);
      /**
       * Disable recording of profiling information.
       */
      static void disable_profiling(void);
      /**
       * Dump the current profiling information to file.
       */
      static void dump_profiling(void);
    public:
      //------------------------------------------------------------------------
      // Task Registration Operations
      //------------------------------------------------------------------------
      /**
       * Register a task with a template return type for the given
       * kind of processor.
       * @param id the ID at which to assign the task
       * @param kind the processor kind on which the task can run
       * @param single whether the task can be run as a single task
       * @param index whether the task can be run as an index space task
       * @param vid the variant ID to assign to the task
       * @param options the task configuration options
       * @param task_name string name for the task
       * @return the ID where the task was assigned
       */
      template<typename T,
        T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, HighLevelRuntime*)>
      static TaskID register_legion_task(TaskID id, Processor::Kind proc_kind,
                                         bool single, bool index, 
                                         VariantID vid = AUTO_GENERATE_ID,
                              TaskConfigOptions options = TaskConfigOptions(),
                                         const char *task_name = NULL);
      /**
       * Register a task with a void return type for the given
       * kind of processor.
       * @param id the ID at which to assign the task
       * @param kind the processor kind on which the task can run
       * @param single whether the task can be run as a single task
       * @param index whether the task can be run as an index space task
       * @param vid the variant ID to assign to the task
       * @param options the task configuration options
       * @param task_name string name for the task
       * @return the ID where the task was assigned
       */
      template<
        void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                         Context, HighLevelRuntime*)>
      static TaskID register_legion_task(TaskID id, Processor::Kind proc_kind,
                                         bool single, bool index,
                                         VariantID vid = AUTO_GENERATE_ID,
                             TaskConfigOptions options = TaskConfigOptions(),
                                         const char *task_name = NULL);
      /**
       * @deprecated
       * Register a single task with a template return type for the given
       * task ID and the processor kind.  Optionally specify whether the
       * task is a leaf task or give it a name for debugging error messages.
       * @param id the ID at which to assign the task
       * @param kind the processor kind on which the task can run
       * @param leaf whether the task is a leaf task (makes no runtime calls)
       * @param name for the task in error messages
       * @param vid the variant ID to assign to the task
       * @param inner whether the task is an inner task
       * @param idempotent whether the task is idempotent
       * @return the ID where the task was assigned
       */
      template<typename T,
        T (*TASK_PTR)(const void*,size_t,
                      const std::vector<RegionRequirement>&,
                      const std::vector<PhysicalRegion>&,
                      Context,HighLevelRuntime*)>
      static TaskID register_single_task(TaskID id, Processor::Kind proc_kind,
                                         bool leaf = false,
                                         const char *name = NULL,
                                         VariantID vid = AUTO_GENERATE_ID,
                                         bool inner = false,
                                         bool idempotent = false);
      /**
       * @deprecated
       * Register a single task with a void return type for the given
       * task ID and the processor kind.  Optionally specify whether the
       * task is a leaf task or give it a name for debugging error messages.
       * @param id the ID at which to assign the task
       * @param kind the processor kind on which the task can run
       * @param leaf whether the task is a leaf task (makes no runtime calls)
       * @param name for the task in error messages
       * @param vid the variant ID to assign to the task
       * @param inner whether the task is an inner task
       * @param idempotent whether the task is idempotent
       * @return the ID where the task was assigned
       */
      template<
        void (*TASK_PTR)(const void*,size_t,
                         const std::vector<RegionRequirement>&,
                         const std::vector<PhysicalRegion>&,
                         Context,HighLevelRuntime*)>
      static TaskID register_single_task(TaskID id, Processor::Kind proc_kind,
                                         bool leaf = false,
                                         const char *name = NULL,
                                         VariantID vid = AUTO_GENERATE_ID,
                                         bool inner = false,
                                         bool idempotent = false);
      /**
       * @deprecated
       * Register an index space task with a template return type
       * for the given task ID and processor kind.  Optionally specify
       * whether the task is a leaf task or give it a name for 
       * debugging error messages.
       * @param id the ID at which to assign the task
       * @param kind the processor kind on which the task can run
       * @param leaf whether the task is a leaf task (makes no runtime calls)
       * @param name for the task in error messages
       * @param vid the variant ID to assign to the task
       * @param inner whether the task is an inner task
       * @param idempotent whether the task is idempotent
       * @return the ID where the task was assigned
       */
      template<typename T,
        T (*TASK_PTR)(const void*,size_t,const void*,size_t,const DomainPoint&,
                      const std::vector<RegionRequirement>&,
                      const std::vector<PhysicalRegion>&,
                      Context,HighLevelRuntime*)>
      static TaskID register_index_task(TaskID id, Processor::Kind proc_kind,
                                        bool leaf = false,
                                        const char *name = NULL,
                                        VariantID vid = AUTO_GENERATE_ID,
                                        bool inner = false,
                                        bool idempotent = false); 
      /**
       * @deprecated
       * Register an index space task with a void return type
       * for the given task ID and processor kind.  Optionally specify
       * whether the task is a leaf task or give it a name for 
       * debugging error messages.
       * @param id the ID at which to assign the task
       * @param kind the processor kind on which the task can run
       * @param leaf whether the task is a leaf task (makes no runtime calls)
       * @param name for the task in error messages
       * @param vid the variant ID to assign to the task
       * @param inner whether the task is an inner task
       * @param idempotent whether the task is idempotent
       * @return the ID where the task was assigned
       */
      template<
        void (*TASK_PTR)(const void*,size_t,const void*,size_t,
                         const DomainPoint&,
                         const std::vector<RegionRequirement>&,
                         const std::vector<PhysicalRegion>&,
                         Context,HighLevelRuntime*)>
      static TaskID register_index_task(TaskID id, Processor::Kind proc_kind,
                                        bool leaf = false,
                                        const char *name = NULL,
                                        VariantID vid = AUTO_GENERATE_ID,
                                        bool inner = false,
                                        bool idempotent = false);
    public:
      /**
       * Provide a mechanism for finding the high-level runtime
       * pointer for a processor wrapper tasks that are starting
       * a new application level task.
       * @param processor the task will run on
       * @return the high-level runtime pointer for the specified processor
       */
      static HighLevelRuntime* get_runtime(Processor p);
    private:
      friend class FieldAllocator;
      FieldID allocate_field(Context ctx, FieldSpace space, 
                             size_t field_size, FieldID fid, bool local);
      void free_field(Context ctx, FieldSpace space, FieldID fid);
      void allocate_fields(Context ctx, FieldSpace space, 
                           const std::vector<size_t> &sizes,
                           std::vector<FieldID> &resulting_fields, bool local);
      void free_fields(Context ctx, FieldSpace space, 
                       const std::set<FieldID> &to_free);
    private:
      // Methods for the wrapper functions to get information from the runtime
      friend class LegionTaskWrapper;
      friend class LegionSerialization;
      const std::vector<PhysicalRegion>& begin_task(Context ctx);
      void end_task(Context ctx, const void *result, size_t result_size,
                    bool owned = false);
      const void* get_local_args(Context ctx, DomainPoint &point, 
                                 size_t &local_size);
    private:
      static ProjectionID register_region_projection_function(
                                    ProjectionID handle, void *func_ptr);
      static ProjectionID register_partition_projection_function(
                                    ProjectionID handle, void *func_ptr);
      static TaskID update_collection_table(
                      LowLevelFnptr low_ptr, InlineFnptr inline_ptr,
                      TaskID uid, Processor::Kind proc_kind, 
                      bool single_task, bool index_space_task,
                      VariantID vid, size_t return_size,
                      const TaskConfigOptions &options,
                      const char *task_name);
      static LowLevel::ReductionOpTable& get_reduction_table(void);
    private:
      Runtime *runtime;
    };

    //==========================================================================
    //                        Compiler Helper Classes
    //==========================================================================

    /**
     * \class ColoringSerializer
     * This is a decorator class that helps the Legion compiler
     * with returning colorings as the result of task calls.
     */
    class ColoringSerializer {
    public:
      ColoringSerializer(void) { }
      ColoringSerializer(const Coloring &c);
    public:
      size_t legion_buffer_size(void) const;
      size_t legion_serialize(void *buffer) const;
      size_t legion_deserialize(const void *buffer);
    public:
      inline Coloring& ref(void) { return coloring; }
    private:
      Coloring coloring;
    };

    /**
     * \class DomainColoringSerializer
     * This is a decorator class that helps the Legion compiler
     * with returning domain colorings as the result of task calls.
     */
    class DomainColoringSerializer {
    public:
      DomainColoringSerializer(void) { }
      DomainColoringSerializer(const DomainColoring &c);
    public:
      size_t legion_buffer_size(void) const;
      size_t legion_serialize(void *buffer) const;
      size_t legion_deserialize(const void *buffer);
    public:
      inline DomainColoring& ref(void) { return coloring; }
    private:
      DomainColoring coloring;
    };

    //-------------------------------------------------------------------------
    // Everything below this point is either a template definition to make
    // sure that that the necessary templates get instantiated or inline
    // definitions to make sure everything gets inlined properly.
    // Proceed at your own risk: template meta-programming ahead.
    //-------------------------------------------------------------------------

    /**
     * \class LegionSerialization
     * The Legion serialization class provides template meta-programming
     * help for returning complex data types from task calls.  If the 
     * types have three special methods defined on them then we know
     * how to serialize the type for the runtime rather than just doing
     * a dumb bit copy.  This is especially useful for types which 
     * require deep copies instead of shallow bit copies.  The three
     * methods which must be defined are:
     * size_t legion_buffer_size(void)
     * void legion_serialize(void *buffer)
     * void legion_deserialize(const void *buffer)
     */
    class LegionSerialization {
    public:
      // A helper method for getting access to the runtime's
      // end_task method with private access
      static inline void end_helper(HighLevelRuntime *rt, Context ctx,
          const void *result, size_t result_size, bool owned)
      {
        rt->end_task(ctx, result, result_size, owned);
      }

      // WARNING: There are two levels of SFINAE (substitution failure is 
      // not an error) here.  Proceed at your own risk. First we have to 
      // check to see if the type is a struct.  If it is then we check to 
      // see if it has a 'legion_serialize' method.  We assume if there is 
      // a 'legion_serialize' method there are also 'legion_buffer_size'
      // and 'legion_deserialize' methods.
      
      template<typename T, bool HAS_SERIALIZE>
      struct NonPODSerializer {
        static inline void end_task(HighLevelRuntime *rt, Context ctx, 
                                    T *result)
        {
          size_t buffer_size = result->legion_buffer_size();
          void *buffer = malloc(buffer_size);
          result->legion_serialize(buffer);
          end_helper(rt, ctx, buffer, buffer_size, true/*owned*/);
          // No need to free the buffer, the Legion runtime owns it now
        }
        static inline T unpack(const void *result)
        {
          T derez;
          derez.legion_deserialize(result);
          return derez;
        }
      };

      template<typename T>
      struct NonPODSerializer<T,false> {
        static inline void end_task(HighLevelRuntime *rt, Context ctx, 
                                    T *result)
        {
          end_helper(rt, ctx, (void*)result, sizeof(T), false/*owned*/);
        }
        static inline T unpack(const void *result)
        {
          return (*((const T*)result));
        }
      };

      template<typename T>
      struct HasSerialize {
        typedef char no[1];
        typedef char yes[2];

        struct Fallback { void legion_serialize(void *); };
        struct Derived : T, Fallback { };

        template<typename U, U> struct Check;

        template<typename U>
        static no& test_for_serialize(
                  Check<void (Fallback::*)(void*), &U::legion_serialize> *);

        template<typename U>
        static yes& test_for_serialize(...);

        static const bool value = 
          (sizeof(test_for_serialize<Derived>(0)) == sizeof(yes));
      };

      template<typename T, bool IS_STRUCT>
      struct StructHandler {
        static inline void end_task(HighLevelRuntime *rt, 
                                    Context ctx, T *result)
        {
          // Otherwise this is a struct, so see if it has serialization methods 
          NonPODSerializer<T,HasSerialize<T>::value>::end_task(rt, ctx, 
                                                               result);
        }
        static inline T unpack(const void *result)
        {
          return NonPODSerializer<T,HasSerialize<T>::value>::unpack(result); 
        }
      };
      // False case of template specialization
      template<typename T>
      struct StructHandler<T,false> {
        static inline void end_task(HighLevelRuntime *rt, Context ctx, 
                                    T *result)
        {
          end_helper(rt, ctx, (void*)result, sizeof(T), false/*owned*/);
        }
        static inline T unpack(const void *result)
        {
          return (*((const T*)result));
        }
      };

      template<typename T>
      struct IsAStruct {
        typedef char no[1];
        typedef char yes[2];
        
        template <typename U> static yes& test_for_struct(int U:: *x);
        template <typename U> static no& test_for_struct(...);

        static const bool value = 
                        (sizeof(test_for_struct<T>(0)) == sizeof(yes));
      };

      // Figure out whether this is a struct or not 
      // and call the appropriate Finisher
      template<typename T>
      static inline void end_task(HighLevelRuntime *rt, Context ctx, T *result)
      {
        StructHandler<T,IsAStruct<T>::value>::end_task(rt, ctx, 
                                                       result); 
      }

      template<typename T>
      static inline T unpack(const void *result)
      {
        return StructHandler<T,IsAStruct<T>::value>::unpack(result);
      }
    };
    
    //--------------------------------------------------------------------------
    inline FieldSpace& FieldSpace::operator=(const FieldSpace &rhs)
    //--------------------------------------------------------------------------
    {
      id = rhs.id;
      return *this;
    }

    //--------------------------------------------------------------------------
    inline bool FieldSpace::operator==(const FieldSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      return (id == rhs.id);
    }

    //--------------------------------------------------------------------------
    inline bool FieldSpace::operator!=(const FieldSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      return (id != rhs.id);
    }

    //--------------------------------------------------------------------------
    inline bool FieldSpace::operator<(const FieldSpace &rhs) const
    //--------------------------------------------------------------------------
    {
      return (id < rhs.id);
    }

    //--------------------------------------------------------------------------
    inline LogicalRegion& LogicalRegion::operator=(const LogicalRegion &rhs) 
    //--------------------------------------------------------------------------
    {
      tree_id = rhs.tree_id;
      index_space = rhs.index_space;
      field_space = rhs.field_space;
      return *this;
    }
    
    //--------------------------------------------------------------------------
    inline bool LogicalRegion::operator==(const LogicalRegion &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((tree_id == rhs.tree_id) && (index_space == rhs.index_space) 
              && (field_space == rhs.field_space));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalRegion::operator!=(const LogicalRegion &rhs) const
    //--------------------------------------------------------------------------
    {
      return (!((*this) == rhs));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalRegion::operator<(const LogicalRegion &rhs) const
    //--------------------------------------------------------------------------
    {
      if (tree_id < rhs.tree_id)
        return true;
      else if (tree_id > rhs.tree_id)
        return false;
      else
      {
        if (index_space < rhs.index_space)
          return true;
        else if (index_space != rhs.index_space) // therefore greater than
          return false;
        else
          return field_space < rhs.field_space;
      }
    }

    //--------------------------------------------------------------------------
    inline LogicalPartition& LogicalPartition::operator=(
                                                    const LogicalPartition &rhs)
    //--------------------------------------------------------------------------
    {
      tree_id = rhs.tree_id;
      index_partition = rhs.index_partition;
      field_space = rhs.field_space;
      return *this;
    }

    //--------------------------------------------------------------------------
    inline bool LogicalPartition::operator==(const LogicalPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((tree_id == rhs.tree_id) && 
              (index_partition == rhs.index_partition) && 
              (field_space == rhs.field_space));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalPartition::operator!=(const LogicalPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      return (!((*this) == rhs));
    }

    //--------------------------------------------------------------------------
    inline bool LogicalPartition::operator<(const LogicalPartition &rhs) const
    //--------------------------------------------------------------------------
    {
      if (tree_id < rhs.tree_id)
        return true;
      else if (tree_id > rhs.tree_id)
        return false;
      else
      {
        if (index_partition < rhs.index_partition)
          return true;
        else if (index_partition > rhs.index_partition)
          return false;
        else
          return (field_space < rhs.field_space);
      }
    }

    //--------------------------------------------------------------------------
    inline bool IndexAllocator::operator==(const IndexAllocator &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((index_space == rhs.index_space) && (allocator == rhs.allocator));
    }

    //--------------------------------------------------------------------------
    inline bool IndexAllocator::operator<(const IndexAllocator &rhs) const
    //--------------------------------------------------------------------------
    {
      if (allocator < rhs.allocator)
        return true;
      else if (allocator > rhs.allocator)
        return false;
      else
        return (index_space < rhs.index_space);
    }

    //--------------------------------------------------------------------------
    inline ptr_t IndexAllocator::alloc(unsigned num_elements /*= 1*/)
    //--------------------------------------------------------------------------
    {
      ptr_t result(allocator->alloc(num_elements));
      return result;
    }

    //--------------------------------------------------------------------------
    inline void IndexAllocator::free(ptr_t ptr, unsigned num_elements /*= 1*/)
    //--------------------------------------------------------------------------
    {
      allocator->free(ptr.value,num_elements);
    }

    //--------------------------------------------------------------------------
    inline bool FieldAllocator::operator==(const FieldAllocator &rhs) const
    //--------------------------------------------------------------------------
    {
      return ((field_space == rhs.field_space) && (runtime == rhs.runtime));
    }

    //--------------------------------------------------------------------------
    inline bool FieldAllocator::operator<(const FieldAllocator &rhs) const
    //--------------------------------------------------------------------------
    {
      if (runtime < rhs.runtime)
        return true;
      else if (runtime > rhs.runtime)
        return false;
      else
        return (field_space < rhs.field_space);
    }

    //--------------------------------------------------------------------------
    inline FieldID FieldAllocator::allocate_field(size_t field_size, 
                                FieldID desired_fieldid /*= AUTO_GENERATE_ID*/)
    //--------------------------------------------------------------------------
    {
      return runtime->allocate_field(parent, field_space, 
                                 field_size, desired_fieldid, false/*local*/); 
    }

    //--------------------------------------------------------------------------
    inline void FieldAllocator::free_field(FieldID id)
    //--------------------------------------------------------------------------
    {
      runtime->free_field(parent, field_space, id);
    }

    //--------------------------------------------------------------------------
    inline FieldID FieldAllocator::allocate_local_field(size_t field_size,
                                FieldID desired_fieldid /*= AUTO_GENERATE_ID*/)
    //--------------------------------------------------------------------------
    {
      return runtime->allocate_field(parent, field_space,
                                field_size, desired_fieldid, true/*local*/);
    }

    //--------------------------------------------------------------------------
    inline void FieldAllocator::allocate_fields(
        const std::vector<size_t> &field_sizes,
        std::vector<FieldID> &resulting_fields)
    //--------------------------------------------------------------------------
    {
      runtime->allocate_fields(parent, field_space, 
                               field_sizes, resulting_fields, false/*local*/);
    }

    //--------------------------------------------------------------------------
    inline void FieldAllocator::free_fields(const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      runtime->free_fields(parent, field_space, to_free);
    }

    //--------------------------------------------------------------------------
    inline void FieldAllocator::allocate_local_fields(
        const std::vector<size_t> &field_sizes,
        std::vector<FieldID> &resulting_fields)
    //--------------------------------------------------------------------------
    {
      runtime->allocate_fields(parent, field_space, 
                               field_sizes, resulting_fields, true/*local*/);
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline void ArgumentMap::set_point_arg(const PT point[DIM], 
                                           const TaskArgument &arg, 
                                           bool replace/*= false*/)
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT(DIM <= DomainPoint::MAX_POINT_DIM);  
      DomainPoint dp;
      dp.dim = DIM;
      for (unsigned idx = 0; idx < DIM; idx++)
        dp.point_data[idx] = point[idx];
      set_point(dp, arg, replace);
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline bool ArgumentMap::remove_point(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT(DIM <= DomainPoint::MAX_POINT_DIM);
      DomainPoint dp;
      dp.dim = DIM;
      for (unsigned idx = 0; idx < DIM; idx++)
        dp.point_data[idx] = point[idx];
      return remove_point(dp);
    }

    //--------------------------------------------------------------------------
    inline bool Predicate::operator==(const Predicate &p) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
      {
        if (p.impl == NULL)
          return (const_value == p.const_value);
        else
          return false;
      }
      else
        return (impl == p.impl);
    }

    //--------------------------------------------------------------------------
    inline bool Predicate::operator<(const Predicate &p) const
    //--------------------------------------------------------------------------
    {
      if (impl == NULL)
      {
        if (p.impl == NULL)
          return (const_value < p.const_value);
        else
          return true;
      }
      else
        return (impl < p.impl);
    }

    //--------------------------------------------------------------------------
    inline bool Predicate::operator!=(const Predicate &p) const
    //--------------------------------------------------------------------------
    {
      return !(*this == p);
    }

    //--------------------------------------------------------------------------
    inline void RegionRequirement::add_field(FieldID fid, 
                                             bool instance/*= true*/)
    //--------------------------------------------------------------------------
    {
      privilege_fields.insert(fid);
      if (instance)
        instance_fields.push_back(fid);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_index_requirement(
                                              const IndexSpaceRequirement &req)
    //--------------------------------------------------------------------------
    {
      index_requirements.push_back(req);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_region_requirement(
                                                  const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      region_requirements.push_back(req);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_field(unsigned idx, FieldID fid, bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < region_requirements.size());
#endif
      region_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_future(Future f)
    //--------------------------------------------------------------------------
    {
      futures.push_back(f);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void TaskLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void IndexLauncher::add_index_requirement(
                                              const IndexSpaceRequirement &req)
    //--------------------------------------------------------------------------
    {
      index_requirements.push_back(req);
    }

    //--------------------------------------------------------------------------
    inline void IndexLauncher::add_region_requirement(
                                                  const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      region_requirements.push_back(req);
    }

    //--------------------------------------------------------------------------
    inline void IndexLauncher::add_field(unsigned idx, FieldID fid, bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < region_requirements.size());
#endif
      region_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void IndexLauncher::add_future(Future f)
    //--------------------------------------------------------------------------
    {
      futures.push_back(f);
    }

    //--------------------------------------------------------------------------
    inline void IndexLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void IndexLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void IndexLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void InlineLauncher::add_field(FieldID fid, bool inst)
    //--------------------------------------------------------------------------
    {
      requirement.add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_copy_requirements(
                     const RegionRequirement &src, const RegionRequirement &dst)
    //--------------------------------------------------------------------------
    {
      src_requirements.push_back(src);
      dst_requirements.push_back(dst);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_src_field(unsigned idx,FieldID fid,bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < src_requirements.size());
#endif
      src_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_dst_field(unsigned idx,FieldID fid,bool inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < dst_requirements.size());
#endif
      dst_requirements[idx].add_field(fid, inst);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void CopyLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_field(FieldID f)
    //--------------------------------------------------------------------------
    {
      fields.insert(f);
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void AcquireLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_field(FieldID f)
    //--------------------------------------------------------------------------
    {
      fields.insert(f);
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      grants.push_back(g);
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_wait_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      wait_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    inline void ReleaseLauncher::add_arrival_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      arrive_barriers.push_back(bar);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T Future::get_result(void)
    //--------------------------------------------------------------------------
    {
      // Unpack the value using LegionSerialization in case
      // the type has an alternative method of unpacking
      return LegionSerialization::unpack<T>(get_untyped_result());
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline const T& Future::get_reference(void)
    //--------------------------------------------------------------------------
    {
      return *((const T*)get_untyped_result());
    }

    //--------------------------------------------------------------------------
    inline const void* Future::get_untyped_pointer(void)
    //--------------------------------------------------------------------------
    {
      return get_untyped_result();
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T FutureMap::get_result(const DomainPoint &dp)
    //--------------------------------------------------------------------------
    {
      Future f = get_future(dp);
      return f.get_result<T>();
    }

    //--------------------------------------------------------------------------
    template<typename RT, typename PT, unsigned DIM>
    inline RT FutureMap::get_result(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT(DIM <= DomainPoint::MAX_POINT_DIM);
      DomainPoint dp;
      dp.dim = DIM;
      for (unsigned idx = 0; idx < DIM; idx++)
        dp.point_data[idx] = point[idx];
      Future f = get_future(dp);
      return f.get_result<RT>();
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline Future FutureMap::get_future(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT(DIM <= DomainPoint::MAX_POINT_DIM);
      DomainPoint dp;
      dp.dim = DIM;
      for (unsigned idx = 0; idx < DIM; idx++)
        dp.point_data[idx] = point[idx];
      return get_future(dp);
    }

    //--------------------------------------------------------------------------
    template<typename PT, unsigned DIM>
    inline void FutureMap::get_void_result(const PT point[DIM])
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT(DIM <= DomainPoint::MAX_POINT_DIM);
      DomainPoint dp;
      dp.dim = DIM;
      for (unsigned idx = 0; idx < DIM; idx++)
        dp.point_data[idx] = point[idx];
      Future f = get_future(dp);
      return f.get_void_result();
    }

    //--------------------------------------------------------------------------
    inline bool PhysicalRegion::is_mapped(void) const
    //--------------------------------------------------------------------------
    {
      return (impl != NULL);
    }

    //--------------------------------------------------------------------------
    inline bool IndexIterator::has_next(void) const
    //--------------------------------------------------------------------------
    {
      return (!finished);
    }
    
    //--------------------------------------------------------------------------
    inline ptr_t IndexIterator::next(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!finished);
#endif
      ptr_t result = current_pointer;
      remaining_elmts--;
      if (remaining_elmts > 0)
      {
        current_pointer++;
      }
      else
      {
        finished = !(enumerator->get_next(current_pointer, remaining_elmts));
      }
      return result;
    }

    //--------------------------------------------------------------------------
    inline UniqueID Task::get_unique_task_id(void) const
    //--------------------------------------------------------------------------
    {
      return get_unique_mappable_id();
    }

    //--------------------------------------------------------------------------
    inline UniqueID Copy::get_unique_copy_id(void) const
    //--------------------------------------------------------------------------
    {
      return get_unique_mappable_id();
    }

    //--------------------------------------------------------------------------
    inline UniqueID Inline::get_unique_inline_id(void) const
    //--------------------------------------------------------------------------
    {
      return get_unique_mappable_id();
    }

    //--------------------------------------------------------------------------
    inline UniqueID Acquire::get_unique_acquire_id(void) const
    //--------------------------------------------------------------------------
    {
      return get_unique_mappable_id();
    }

    //--------------------------------------------------------------------------
    inline UniqueID Release::get_unique_release_id(void) const
    //--------------------------------------------------------------------------
    {
      return get_unique_mappable_id();
    }

    //--------------------------------------------------------------------------
    template<typename T>
    IndexPartition HighLevelRuntime::create_index_partition(Context ctx,
        IndexSpace parent, const T& mapping, int part_color /*= -1*/)
    //--------------------------------------------------------------------------
    {
      Arrays::Rect<T::IDIM> parent_rect = 
        get_index_space_domain(ctx, parent).get_rect<T::IDIM>();
      Arrays::Rect<T::ODIM> color_space = mapping.image_convex(parent_rect);
      Arrays::CArrayLinearization<T::ODIM> color_space_lin(color_space);
      DomainColoring c;
      for (typename T::PointInOutputRectIterator pir(color_space); 
          pir; pir++) 
      {
        Arrays::Rect<T::IDIM> preimage = mapping.preimage(pir.p);
#ifdef DEBUG_HIGH_LEVEL
        assert(mapping.preimage_is_dense(pir.p));
#endif
        c[color_space_lin.image(pir.p)] = Domain::from_rect<T::IDIM>(preimage);
      }
      IndexPartition result = create_index_partition(ctx, parent, 
              Domain::from_rect<T::ODIM>(color_space), c, 
              true/*disjoint*/, part_color);
#ifdef DEBUG_HIGH_LEVEL
      // We don't actually know if we're supposed to check disjointness
      // so if we're in debug mode then just do it.
      {
        std::set<Color> current_colors;  
        for (DomainColoring::const_iterator it1 = c.begin();
              it1 != c.end(); it1++)
        {
          current_colors.insert(it1->first);
          for (DomainColoring::const_iterator it2 = c.begin();
                it2 != c.end(); it2++)
          {
            if (current_colors.find(it2->first) != current_colors.end())
              continue;
            Arrays::Rect<T::IDIM> rect1 = it1->second.get_rect<T::IDIM>();
            Arrays::Rect<T::IDIM> rect2 = it2->second.get_rect<T::IDIM>();
            if (rect1.overlaps(rect2))
            {
              fprintf(stderr,"ERROR: colors %d and %d of partition %d are " 
                             "not disjoint rectangles as they should be!",
                              it1->first, it2->first, result);
              assert(false);
              exit(ERROR_DISJOINTNESS_TEST_FAILURE);
            }
          }
        }
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    template<unsigned DIM>
    IndexSpace HighLevelRuntime::get_index_subspace(Context ctx, 
                              IndexPartition p, Arrays::Point<DIM> color_point)
    //--------------------------------------------------------------------------
    {
      Arrays::Rect<DIM> color_space = 
        get_index_partition_color_space(ctx, p).get_rect<DIM>();
      Arrays::CArrayLinearization<DIM> color_space_lin(color_space);
      return get_index_subspace(ctx, p, 
                                  (Color)(color_space_lin.image(color_point)));
    }

    //--------------------------------------------------------------------------
    template<typename REDOP>
    /*static*/ void HighLevelRuntime::register_reduction_op(
                                                        ReductionOpID redop_id)
    //--------------------------------------------------------------------------
    {
      if (redop_id == 0)
      {
        fprintf(stderr,"ERROR: ReductionOpID zero is reserved.\n");
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_RESERVED_REDOP_ID);
      }
      LowLevel::ReductionOpTable &red_table = 
          HighLevelRuntime::get_reduction_table(); 
      // Check to make sure we're not overwriting a prior reduction op 
      if (red_table.find(redop_id) != red_table.end())
      {
        fprintf(stderr,"ERROR: ReductionOpID %d has already been used " 
                       "in the reduction table\n",redop_id);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_DUPLICATE_REDOP_ID);
      }
      red_table[redop_id] = 
        LowLevel::ReductionOpUntyped::create_reduction_op<REDOP>(); 
    }

    //--------------------------------------------------------------------------
    template<LogicalRegion (*PROJ_PTR)(LogicalRegion, const DomainPoint&,
                                       HighLevelRuntime*)>
    /*static*/ ProjectionID HighLevelRuntime::register_region_function(
                                                    ProjectionID handle)
    //--------------------------------------------------------------------------
    {
      return HighLevelRuntime::register_region_projection_function(
                                  handle, reinterpret_cast<void *>(PROJ_PTR));
    }

    //--------------------------------------------------------------------------
    template<LogicalRegion (*PROJ_PTR)(LogicalPartition, const DomainPoint&,
                                       HighLevelRuntime*)>
    /*static*/ ProjectionID HighLevelRuntime::register_partition_function(
                                                    ProjectionID handle)
    //--------------------------------------------------------------------------
    {
      return HighLevelRuntime::register_partition_projection_function(
                                  handle, reinterpret_cast<void *>(PROJ_PTR));
    }
    
    //--------------------------------------------------------------------------
    // Wrapper functions for high-level tasks
    //--------------------------------------------------------------------------

    /**
     * \class LegionTaskWrapper
     * This is a helper class that has static template methods for 
     * wrapping Legion application tasks.  For all tasks we can make
     * wrappers both for normal execution and also for inline execution.
     */
    class LegionTaskWrapper {
    public: 
      // Non-void return type for new legion task types
      template<typename T,
        T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, HighLevelRuntime*)>
      static void legion_task_wrapper(const void*, size_t, Processor);
    public:
      // Void return type for new legion task types
      template<
        void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, HighLevelRuntime*)>
      static void legion_task_wrapper(const void*, size_t, Processor);
    public:
      // Non-void single task wrapper
      template<typename T,
      T (*TASK_PTR)(const void*,size_t,const std::vector<RegionRequirement>&,
                  const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
      static void high_level_task_wrapper(const void*, size_t, Processor);
    public:
      // Void single task wrapper
      template<
      void (*TASK_PTR)(const void*,size_t,const std::vector<RegionRequirement>&,
                 const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
      static void high_level_task_wrapper(const void*, size_t, Processor);
    public:
      // Non-void index task wrapper
      template<typename RT,
      RT (*TASK_PTR)(const void*,size_t,const void*,size_t,const DomainPoint&,
                 const std::vector<RegionRequirement>&,
                 const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
      static void high_level_index_task_wrapper(const void*,size_t,Processor);
    public:
      // Void index task wrapper
      template< 
      void (*TASK_PTR)(const void*,size_t,const void*,size_t,const DomainPoint&, 
                 const std::vector<RegionRequirement>&,
                 const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
      static void high_level_index_task_wrapper(const void*,size_t,Processor);
    public: // INLINE VERSIONS OF THE ABOVE METHODS
      template<typename T,
        T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, HighLevelRuntime*)>
      static void inline_task_wrapper(const Task*, 
          const std::vector<PhysicalRegion>&, Context, HighLevelRuntime*,
          void*&, size_t&);
    public:
      template<
        void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                         Context, HighLevelRuntime*)>
      static void inline_task_wrapper(const Task*,
          const std::vector<PhysicalRegion>&, Context, HighLevelRuntime*,
          void*&, size_t&);
    public:
      template<typename T,
      T (*TASK_PTR)(const void*,size_t,const std::vector<RegionRequirement>&,
                  const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
      static void high_level_inline_task_wrapper(const Task*,
          const std::vector<PhysicalRegion>&, Context, HighLevelRuntime*,
          void*&, size_t&);
    public:
      template<
      void (*TASK_PTR)(const void*,size_t,const std::vector<RegionRequirement>&,
                 const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
      static void high_level_inline_task_wrapper(const Task*,
          const std::vector<PhysicalRegion>&, Context, HighLevelRuntime*,
          void*&, size_t&);
    public:
      template<typename RT,
      RT (*TASK_PTR)(const void*,size_t,const void*,size_t,const DomainPoint&,
                 const std::vector<RegionRequirement>&,
                 const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
      static void high_level_inline_index_task_wrapper(const Task*,
          const std::vector<PhysicalRegion>&, Context, HighLevelRuntime*,
          void*&, size_t&);
    public:
      template< 
      void (*TASK_PTR)(const void*,size_t,const void*,size_t,const DomainPoint&, 
                 const std::vector<RegionRequirement>&,
                 const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
      static void high_level_inline_index_task_wrapper(const Task*,
          const std::vector<PhysicalRegion>&, Context, HighLevelRuntime*,
          void*&, size_t&);
    };
    
    //--------------------------------------------------------------------------
    template<typename T,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, HighLevelRuntime*)>
    void LegionTaskWrapper::legion_task_wrapper(const void *args, 
                                                size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Assert that we are returning Futures or FutureMaps
      LEGION_STATIC_ASSERT((LegionTypeInequality<T,Future>::value));
      LEGION_STATIC_ASSERT((LegionTypeInequality<T,FutureMap>::value));
      // Assert that the return type size is within the required size
      LEGION_STATIC_ASSERT(sizeof(T) <= MAX_RETURN_SIZE);
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      assert(arglen == sizeof(Context));
#endif
      const std::vector<PhysicalRegion> &regions = runtime->begin_task(ctx);

      // Invoke the task with the given context
      T return_value = 
        (*TASK_PTR)(reinterpret_cast<Task*>(ctx),regions,ctx,runtime);

      // Send the return value back
      LegionSerialization::end_task<T>(runtime, ctx, &return_value);
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, HighLevelRuntime*)>
    void LegionTaskWrapper::legion_task_wrapper(const void *args, 
                                                size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      assert(arglen == sizeof(Context));
#endif
      const std::vector<PhysicalRegion> &regions = runtime->begin_task(ctx);

      (*TASK_PTR)(reinterpret_cast<Task*>(ctx), regions, ctx, runtime);

      // Send an empty return value back
      runtime->end_task(ctx, NULL, 0);
    }

    //--------------------------------------------------------------------------
    template<typename T,
      T (*TASK_PTR)(const void*,size_t,const std::vector<RegionRequirement>&,
                  const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    void LegionTaskWrapper::high_level_task_wrapper(const void *args, 
                                                    size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Assert that we aren't returning Futures or FutureMaps
      LEGION_STATIC_ASSERT((LegionTypeInequality<T,Future>::value));
      LEGION_STATIC_ASSERT((LegionTypeInequality<T,FutureMap>::value));
      // Assert that the return type size is within the required size
      LEGION_STATIC_ASSERT(sizeof(T) <= MAX_RETURN_SIZE);
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      assert(arglen == sizeof(Context));
#endif
      // Get the arguments associated with the context
      const std::vector<PhysicalRegion> &regions = runtime->begin_task(ctx);
      Task *task = reinterpret_cast<Task*>(ctx);

      // Invoke the task with the given context
      T return_value = (*TASK_PTR)(task->args, task->arglen, task->regions,
                                   regions, ctx, runtime);

      // Send the return value back
      LegionSerialization::end_task<T>(runtime, ctx, &return_value);
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const void*,size_t,const std::vector<RegionRequirement>&,
                 const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    void LegionTaskWrapper::high_level_task_wrapper(const void *args, 
                                                    size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      assert(arglen == sizeof(Context));
#endif
      // Get the arguments associated with the context
      const std::vector<PhysicalRegion> &regions = runtime->begin_task(ctx);
      Task *task = reinterpret_cast<Task*>(ctx);

      // Invoke the task with the given context
      (*TASK_PTR)(task->args, task->arglen, task->regions, 
                                    regions, ctx, runtime);

      // Send an empty return value back
      runtime->end_task(ctx, NULL, 0);
    }

    //--------------------------------------------------------------------------
    template<typename RT,
      RT (*TASK_PTR)(const void*,size_t,const void*,size_t,const DomainPoint&,
                 const std::vector<RegionRequirement>&,
                 const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    void LegionTaskWrapper::high_level_index_task_wrapper(const void *args, 
                                                     size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Assert that we aren't returning Futures or FutureMaps
      LEGION_STATIC_ASSERT((LegionTypeInequality<RT,Future>::value));
      LEGION_STATIC_ASSERT((LegionTypeInequality<RT,FutureMap>::value));
      // Assert that the return type size is within the required size
      LEGION_STATIC_ASSERT(sizeof(RT) <= MAX_RETURN_SIZE);
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      assert(arglen == sizeof(Context));
#endif
      // Get the arguments associated with the context
      const std::vector<PhysicalRegion> &regions = runtime->begin_task(ctx);
      Task *task = reinterpret_cast<Task*>(ctx);
      
      // Invoke the task with the given context
      RT return_value = (*TASK_PTR)(task->args, task->arglen, task->local_args, 
                                    task->local_arglen, task->index_point, 
                                    task->regions, regions, ctx, runtime);

      // Send the return value back
      LegionSerialization::end_task<RT>(runtime, ctx, &return_value); 
    }

    //--------------------------------------------------------------------------
    template< 
      void (*TASK_PTR)(const void*,size_t,const void*,size_t,const DomainPoint&, 
                 const std::vector<RegionRequirement>&,
                 const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    void LegionTaskWrapper::high_level_index_task_wrapper(const void *args, 
                                                     size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Get the high level runtime
      HighLevelRuntime *runtime = HighLevelRuntime::get_runtime(p);

      // Read the context out of the buffer
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      assert(arglen == sizeof(Context));
#endif
      // Get the arguments associated with the context
      const std::vector<PhysicalRegion> &regions = runtime->begin_task(ctx); 
      Task *task = reinterpret_cast<Task*>(ctx);

      // Invoke the task with the given context
      (*TASK_PTR)(task->args, task->arglen, task->local_args, 
                  task->local_arglen, task->index_point, 
                  task->regions, regions, ctx, runtime);

      // Send an empty return value back
      runtime->end_task(ctx, NULL, 0); 
    }

    //--------------------------------------------------------------------------
    template<typename T,
        T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, HighLevelRuntime*)>
    void LegionTaskWrapper::inline_task_wrapper(const Task *task, 
          const std::vector<PhysicalRegion> &regions, 
          Context ctx, HighLevelRuntime *runtime, 
          void *&return_addr, size_t &return_size)
    //--------------------------------------------------------------------------
    {
      // Assert that we aren't returning Futures or FutureMaps
      LEGION_STATIC_ASSERT((LegionTypeInequality<T,Future>::value));
      LEGION_STATIC_ASSERT((LegionTypeInequality<T,FutureMap>::value));
      // Assert that the return type size is within the required size
      LEGION_STATIC_ASSERT(sizeof(T) <= MAX_RETURN_SIZE);

      T return_value = (*TASK_PTR)(task, regions, ctx, runtime);

      // Send the return value back, no need to pack it
      return_size = sizeof(return_value);
      return_addr = malloc(return_size);
      memcpy(return_addr,&return_value,return_size);
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                       Context, HighLevelRuntime*)>
    void LegionTaskWrapper::inline_task_wrapper(const Task *task,
          const std::vector<PhysicalRegion> &regions, 
          Context ctx, HighLevelRuntime *runtime,
          void *&return_addr, size_t &return_size)
    //--------------------------------------------------------------------------
    {
      (*TASK_PTR)(task, regions, ctx, runtime);

      return_size = 0;
      return_addr = 0;
    }

    //--------------------------------------------------------------------------
    template<typename T,
      T (*TASK_PTR)(const void*,size_t,const std::vector<RegionRequirement>&,
                  const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    void LegionTaskWrapper::high_level_inline_task_wrapper(const Task *task,
          const std::vector<PhysicalRegion> &regions,
          Context ctx, HighLevelRuntime *runtime,
          void *&return_addr, size_t &return_size)
    //--------------------------------------------------------------------------
    {
      // Assert that we aren't returning Futures or FutureMaps
      LEGION_STATIC_ASSERT((LegionTypeInequality<T,Future>::value));
      LEGION_STATIC_ASSERT((LegionTypeInequality<T,FutureMap>::value));
      // Assert that the return type size is within the required size
      LEGION_STATIC_ASSERT(sizeof(T) <= MAX_RETURN_SIZE);

      T return_value = (*TASK_PTR)(task->args, task->arglen,
                                   task->regions, regions, ctx, runtime);

      return_size = sizeof(return_value);
      return_addr = malloc(return_size);
      memcpy(return_addr,&return_value,return_size);
    }

    //--------------------------------------------------------------------------
    template<
    void (*TASK_PTR)(const void*,size_t,const std::vector<RegionRequirement>&,
                 const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    void LegionTaskWrapper::high_level_inline_task_wrapper(const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx, HighLevelRuntime *runtime,
        void *&return_addr, size_t &return_size)
    //--------------------------------------------------------------------------
    {
      (*TASK_PTR)(task->args, task->arglen, 
                  task->regions, regions, ctx, runtime);

      return_size = 0;
      return_addr = NULL;
    }

    //--------------------------------------------------------------------------
    template<typename RT,
      RT (*TASK_PTR)(const void*,size_t,const void*,size_t,const DomainPoint&,
                 const std::vector<RegionRequirement>&,
                 const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    void LegionTaskWrapper::high_level_inline_index_task_wrapper(
          const Task *task, const std::vector<PhysicalRegion> &regions,
          Context ctx, HighLevelRuntime *runtime,
          void *&return_addr, size_t &return_size)
    //--------------------------------------------------------------------------
    {
      // Assert that we aren't returning Futures or FutureMaps
      LEGION_STATIC_ASSERT((LegionTypeInequality<RT,Future>::value));
      LEGION_STATIC_ASSERT((LegionTypeInequality<RT,FutureMap>::value));
      // Assert that the return type size is within the required size
      LEGION_STATIC_ASSERT(sizeof(RT) <= MAX_RETURN_SIZE);

      RT return_value = (*TASK_PTR)(task->args, task->arglen,
                                    task->local_args, task->local_arglen,
                                    task->index_point, task->regions,
                                    regions, ctx, runtime);

      return_size = sizeof(return_value);
      return_addr = malloc(return_size);
      memcpy(return_addr,&return_value,return_size);
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const void*,size_t,const void*,size_t,const DomainPoint&, 
                 const std::vector<RegionRequirement>&,
                 const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    void LegionTaskWrapper::high_level_inline_index_task_wrapper(
        const Task *task, const std::vector<PhysicalRegion> &regions,
        Context ctx, HighLevelRuntime *runtime,
        void *&return_addr, size_t &return_size)
    //--------------------------------------------------------------------------
    {
      (*TASK_PTR)(task->args, task->arglen, 
                  task->local_args, task->local_arglen,
                  task->index_point, task->regions,
                  regions, ctx, runtime);
                              
      return_size = 0;
      return_addr = NULL;
    }

    //--------------------------------------------------------------------------
    template<typename T,
      T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, HighLevelRuntime*)>
    /*static*/ TaskID HighLevelRuntime::register_legion_task(TaskID id,
                                                    Processor::Kind proc_kind,
                                                    bool single, bool index,
                                                    VariantID vid,
                                                    TaskConfigOptions options,
                                                    const char *task_name)
    //--------------------------------------------------------------------------
    {
      if (task_name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(32*sizeof(char));
        sprintf(buffer,"%d",id);
        task_name = buffer;
      }
      return HighLevelRuntime::update_collection_table(
        LegionTaskWrapper::legion_task_wrapper<T,TASK_PTR>, 
        LegionTaskWrapper::inline_task_wrapper<T,TASK_PTR>, id, proc_kind, 
                            single, index, vid, sizeof(T), options, task_name);
    }

    //--------------------------------------------------------------------------
    template<
      void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                    Context, HighLevelRuntime*)>
    /*static*/ TaskID HighLevelRuntime::register_legion_task(TaskID id,
                                                    Processor::Kind proc_kind,
                                                    bool single, bool index,
                                                    VariantID vid,
                                                    TaskConfigOptions options,
                                                    const char *task_name)
    //--------------------------------------------------------------------------
    {
      if (task_name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(32*sizeof(char));
        sprintf(buffer,"%d",id);
        task_name = buffer;
      }
      else
        task_name = strdup(task_name);
      return HighLevelRuntime::update_collection_table(
        LegionTaskWrapper::legion_task_wrapper<TASK_PTR>, 
        LegionTaskWrapper::inline_task_wrapper<TASK_PTR>, id, proc_kind, 
                            single, index, vid, 0/*size*/, options, task_name);
    }

    //--------------------------------------------------------------------------
    template<typename T,
        T (*TASK_PTR)(const void*,size_t,const std::vector<RegionRequirement>&, 
                  const std::vector<PhysicalRegion>&,Context,HighLevelRuntime*)>
    /*static*/ TaskID HighLevelRuntime::register_single_task(TaskID id, 
                                                    Processor::Kind proc_kind, 
                                                    bool leaf/*= false*/, 
                                                    const char *name/*= NULL*/,
                                                    VariantID vid/*= AUTO*/,
                                                    bool inner/*= false*/,
                                                    bool idempotent/*= false*/)
    //--------------------------------------------------------------------------
    {
      if (name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(32*sizeof(char));
        sprintf(buffer,"%d",id);
        name = buffer;
      }
      else
        name = strdup(name);
      return HighLevelRuntime::update_collection_table(
              LegionTaskWrapper::high_level_task_wrapper<T,TASK_PTR>,  
              LegionTaskWrapper::high_level_inline_task_wrapper<T,TASK_PTR>,
              id, proc_kind, true/*single*/, false/*index space*/, vid,
              sizeof(T), TaskConfigOptions(leaf, inner, idempotent), name);
    }

    //--------------------------------------------------------------------------
    template<
        void (*TASK_PTR)(const void*,size_t,
                         const std::vector<RegionRequirement>&,
                         const std::vector<PhysicalRegion>&,
                         Context,HighLevelRuntime*)>
    /*static*/ TaskID HighLevelRuntime::register_single_task(TaskID id, 
                                                     Processor::Kind proc_kind, 
                                                     bool leaf/*= false*/, 
                                                     const char *name/*= NULL*/,
                                                     VariantID vid/*= AUTO*/,
                                                     bool inner/*= false*/,
                                                     bool idempotent/*= false*/)
    //--------------------------------------------------------------------------
    {
      if (name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(32*sizeof(char));
        sprintf(buffer,"%d",id);
        name = buffer;
      }
      else
        name = strdup(name);
      return HighLevelRuntime::update_collection_table(
              LegionTaskWrapper::high_level_task_wrapper<TASK_PTR>,
              LegionTaskWrapper::high_level_inline_task_wrapper<TASK_PTR>,
              id, proc_kind, true/*single*/, false/*index space*/, vid,
              0/*size*/, TaskConfigOptions(leaf, inner, idempotent), name);
    }

    //--------------------------------------------------------------------------
    template<typename RT/*return type*/,
        RT (*TASK_PTR)(const void*,size_t,const void*,size_t,const DomainPoint&,
                       const std::vector<RegionRequirement>&,
                       const std::vector<PhysicalRegion>&,
                       Context,HighLevelRuntime*)>
    /*static*/ TaskID HighLevelRuntime::register_index_task(TaskID id, 
                                                    Processor::Kind proc_kind, 
                                                    bool leaf/*= false*/, 
                                                    const char *name/*= NULL*/,
                                                    VariantID vid/*= AUTO*/,
                                                    bool inner/*= false*/,
                                                    bool idempotent/*= false*/)
    //--------------------------------------------------------------------------
    {
      if (name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(32*sizeof(char));
        sprintf(buffer,"%d",id);
        name = buffer;
      }
      else
        name = strdup(name);
      return HighLevelRuntime::update_collection_table(
          LegionTaskWrapper::high_level_index_task_wrapper<RT,TASK_PTR>,  
          LegionTaskWrapper::high_level_inline_index_task_wrapper<RT,TASK_PTR>,
          id, proc_kind, false/*single*/, true/*index space*/, vid,
          sizeof(RT), TaskConfigOptions(leaf, inner, idempotent), name);
    }

    //--------------------------------------------------------------------------
    template<
        void (*TASK_PTR)(const void*,size_t,const void*,size_t,
                         const DomainPoint&,
                         const std::vector<RegionRequirement>&,
                         const std::vector<PhysicalRegion>&,
                         Context,HighLevelRuntime*)>
    /*static*/ TaskID HighLevelRuntime::register_index_task(TaskID id, 
                                                    Processor::Kind proc_kind, 
                                                    bool leaf/*= false*/, 
                                                    const char *name/*= NULL*/,
                                                    VariantID vid/*= AUTO*/,
                                                    bool inner/*= false*/,
                                                    bool idempotent/*= false*/)
    //--------------------------------------------------------------------------
    {
      if (name == NULL)
      {
        // Has no name, so just call it by its number
        char *buffer = (char*)malloc(32*sizeof(char));
        sprintf(buffer,"%d",id);
        name = buffer;
      }
      else
        name = strdup(name);
      return HighLevelRuntime::update_collection_table(
              LegionTaskWrapper::high_level_index_task_wrapper<TASK_PTR>, 
              LegionTaskWrapper::high_level_inline_index_task_wrapper<TASK_PTR>,
              id, proc_kind, false/*single*/, true/*index space*/, vid,
              0/*size*/, TaskConfigOptions(leaf, inner, idempotent), name);
    }


  }; // namespace HighLevel
}; // namespace LegionRuntime

#endif // __LEGION_RUNTIME_H__

// EOF

