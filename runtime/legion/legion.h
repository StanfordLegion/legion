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


#ifndef __LEGION_RUNTIME_H__
#define __LEGION_RUNTIME_H__

/**
 * \file legion.h
 * Legion C++ API
 */

#include "legion_types.h"
#include "legion_constraint.h"

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
     * \class IndexSpace 
     * Index spaces are handles that reference a collection of 
     * points. These collections of points are used to define the
     * "rows" of a logical region. Only the Legion runtime is able
     * to create non-empty index spaces. 
     */
    class IndexSpace {
    public:
      static const IndexSpace NO_SPACE; /**< empty index space handle*/
    protected:
      // Only the runtime should be allowed to make these
      FRIEND_ALL_RUNTIME_CLASSES
      IndexSpace(IndexSpaceID id, IndexTreeID tid);
    public:
      IndexSpace(void);
      IndexSpace(const IndexSpace &rhs);
    public:
      inline IndexSpace& operator=(const IndexSpace &rhs);
      inline bool operator==(const IndexSpace &rhs) const;
      inline bool operator!=(const IndexSpace &rhs) const;
      inline bool operator<(const IndexSpace &rhs) const;
      inline bool operator>(const IndexSpace &rhs) const;
      inline IndexSpaceID get_id(void) const { return id; }
      inline IndexTreeID get_tree_id(void) const { return tid; }
      inline bool exists(void) const { return (id != 0); }
    private:
      friend std::ostream& operator<<(std::ostream& os, const IndexSpace& is);
      IndexSpaceID id;
      IndexTreeID tid;
    };

    /**
     * \class IndexPartition
     * Index partitions are handles that name partitions of an
     * index space into various subspaces. Only the Legion runtime
     * is able to create non-empty index partitions.
     */
    class IndexPartition {
    public:
      static const IndexPartition NO_PART;
    protected:
      // Only the runtime should be allowed to make these
      FRIEND_ALL_RUNTIME_CLASSES
      IndexPartition(IndexPartitionID id, IndexTreeID tid);
    public:
      IndexPartition(void);
      IndexPartition(const IndexPartition &rhs);
    public:
      inline IndexPartition& operator=(const IndexPartition &rhs);
      inline bool operator==(const IndexPartition &rhs) const;
      inline bool operator!=(const IndexPartition &rhs) const;
      inline bool operator<(const IndexPartition &rhs) const;
      inline bool operator>(const IndexPartition &rhs) const;
      inline IndexPartitionID get_id(void) const { return id; }
      inline IndexTreeID get_tree_id(void) const { return tid; }
      inline bool exists(void) const { return (id != 0); }
    public:
      IndexPartitionID id;
      IndexTreeID tid;
    };

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
      inline bool operator>(const FieldSpace &rhs) const;
      inline FieldSpaceID get_id(void) const { return id; }
      inline bool exists(void) const { return (id != 0); }
    private:
      friend std::ostream& operator<<(std::ostream& os, const FieldSpace& fs);
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
      inline bool exists(void) const { return (tree_id != 0); } 
    private:
      friend std::ostream& operator<<(std::ostream& os, const LogicalRegion& lr);
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
      inline bool exists(void) const { return (tree_id != 0); }
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
     * @see Runtime
     */
    class IndexAllocator {
    public:
      IndexAllocator(void);
      IndexAllocator(const IndexAllocator &allocator);
      ~IndexAllocator(void);
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      // Only the Runtime should be able to make these
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
     * @see Runtime
     */
    class FieldAllocator {
    public:
      FieldAllocator(void);
      FieldAllocator(const FieldAllocator &allocator);
      ~FieldAllocator(void);
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      // Only the Runtime should be able to make these
      FieldAllocator(FieldSpace f, Context p, Runtime *rt);
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
       * @param serdez_id optional parameter for specifying a
       *   custom serdez object for serializing and deserializing
       *   a field when it is moved.
       * @return field ID for the allocated field
       */
      inline FieldID allocate_field(size_t field_size, 
              FieldID desired_fieldid = AUTO_GENERATE_ID,
              CustomSerdezID serdez_id = 0);
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
              FieldID desired_fieldid = AUTO_GENERATE_ID,
              CustomSerdezID serdez_id = 0);
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
                                  std::vector<FieldID> &resulting_fields,
                                  CustomSerdezID serdez_id = 0);
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
                                        std::vector<FieldID> &resulting_fields,
                                        CustomSerdezID serdez_id = 0);
      /**
       * @return field space associated with this allocator
       */
      inline FieldSpace get_field_space(void) const { return field_space; }
    private:
      FieldSpace field_space;
      Context parent;
      Runtime *runtime;
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
     * @see Runtime
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
     * @see Runtime
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
     * @see Runtime
     */
    class PhaseBarrier {
    public:
      PhaseBarrier(void);
    protected:
      // Only the runtime is allowed to make non-empty phase barriers
      FRIEND_ALL_RUNTIME_CLASSES
      PhaseBarrier(Barrier b);
    public:
      bool operator<(const PhaseBarrier &rhs) const;
      bool operator==(const PhaseBarrier &rhs) const;
      bool operator!=(const PhaseBarrier &rhs) const;
    public:
      void arrive(unsigned count = 1);
      void wait(void);
      void alter_arrival_count(int delta);
      Barrier get_barrier(void) const { return phase_barrier; }
    protected:
      Barrier phase_barrier;
      friend std::ostream& operator<<(std::ostream& os, const PhaseBarrier& pb);
    };

   /**
     * \class Collective
     * A DynamicCollective object is a special kind of PhaseBarrier
     * that is created with an associated reduction operation.
     * Arrivals on a dynamic collective can contribute a value to
     * each generation of the collective, either in the form of a
     * value or in the form of a future. The reduction operation is used
     * to reduce all the contributed values (which all must be of the same 
     * type) to a common value. This value is returned in the form of
     * a future which applications can use as a normal future. Note
     * that unlike MPI collectives, collectives in Legion can
     * have different sets of producers and consumers and not
     * all producers need to contribute a value.
     */
    class DynamicCollective : public PhaseBarrier {
    public:
      DynamicCollective(void);
    protected:
      // Only the runtime is allowed to make non-empty dynamic collectives
      FRIEND_ALL_RUNTIME_CLASSES
      DynamicCollective(Barrier b, ReductionOpID redop);
    public:
      // All the same operations as a phase barrier
      void arrive(const void *value, size_t size, unsigned count = 1);
    protected:
      ReductionOpID redop;
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
      inline RegionRequirement& add_field(FieldID fid, bool instance = true);
      inline RegionRequirement& add_fields(const std::vector<FieldID>& fids, 
                                           bool instance = true);

      inline RegionRequirement& add_flags(RegionFlags new_flags);
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
      void get_void_result(void) const;
      /**
       * Check to see if the future is empty.  The
       * user can specify whether to block and wait
       * for the future to complete first before
       * returning.  If the non-blocking version
       * of the call will return true, until
       * the future actually completes.
       */
      bool is_empty(bool block = false) const;
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
    public:
      /**
       * Allow users to generate their own futures. These
       * futures are guaranteed to always have completed
       * and to always have concrete values.
       */
      template<typename T>
      static inline Future from_value(Runtime *rt, const T &value);

      /**
       * Generates a future from an untyped pointer.  No
       * serialization is performed.
       */
      static inline Future from_untyped_pointer(Runtime *rt,
						const void *buffer,
						size_t bytes);
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
    //                    Operation Launcher Classes
    //==========================================================================

    /**
     * \struct TaskLauncher
     * Task launchers are objects that describe a launch
     * configuration to the runtime.  They can be re-used
     * and safely modified between calls to task launches.
     * @see Runtime
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
      inline IndexSpaceRequirement&
              add_index_requirement(const IndexSpaceRequirement &req);
      inline RegionRequirement&
              add_region_requirement(const RegionRequirement &req);
      inline void add_field(unsigned idx, FieldID fid, bool inst = true);
    public:
      inline void add_future(Future f);
      inline void add_grant(Grant g);
      inline void add_wait_barrier(PhaseBarrier bar);
      inline void add_arrival_barrier(PhaseBarrier bar);
    public:
      inline void set_predicate_false_future(Future f);
      inline void set_predicate_false_result(TaskArgument arg);
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
    public:
      // If the predicate is set to anything other than
      // Predicate::TRUE_PRED, then the application must 
      // specify a value for the future in the case that
      // the predicate resolves to false. TaskArgument(NULL,0)
      // can be used if the task's return type is void.
      Future                             predicate_false_future;
      TaskArgument                       predicate_false_result;
    };

    /**
     * \struct IndexLauncher
     * Index launchers are objects that describe the launch
     * of an index space of tasks to the runtime.  They can
     * be re-used and safely modified between calls to 
     * index space launches.
     * @see Runtime
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
      inline IndexSpaceRequirement&
                  add_index_requirement(const IndexSpaceRequirement &req);
      inline RegionRequirement&
                  add_region_requirement(const RegionRequirement &req);
      inline void add_field(unsigned idx, FieldID fid, bool inst = true);
    public:
      inline void add_future(Future f);
      inline void add_grant(Grant g);
      inline void add_wait_barrier(PhaseBarrier bar);
      inline void add_arrival_barrier(PhaseBarrier bar);
    public:
      inline void set_predicate_false_future(Future f);
      inline void set_predicate_false_result(TaskArgument arg);
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
    public:
      // If the predicate is set to anything other than
      // Predicate::TRUE_PRED, then the application must 
      // specify a value for the future in the case that
      // the predicate resolves to false. TaskArgument(NULL,0)
      // can be used if the task's return type is void.
      Future                             predicate_false_future;
      TaskArgument                       predicate_false_result;
    };

    /**
     * \struct InlineLauncher
     * Inline launchers are objects that describe the launch
     * of an inline mapping operation to the runtime.  They
     * can be re-used and safely modified between calls to
     * inline mapping operations.
     * @see Runtime
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
     * region requirements.  The source region requirements 
     * must be READ_ONLY, while the destination requirements 
     * must be either READ_WRITE, WRITE_ONLY, or REDUCE with 
     * a reduction function.  While the regions in a source 
     * and a destination pair do not have to be in the same 
     * region tree, one of the following two conditions must hold: 
     * 1. The two regions share an index space tree and the
     *    source region's index space is an ancestor of the
     *    destination region's index space.
     * 2. The source and destination index spaces must be
     *    of the same kind (either dimensions match or number
     *    of elements match in the element mask) and the source
     *    region's index space must dominate the destination
     *    region's index space.
     * If either of these two conditions does not hold then
     * the runtime will issue an error.
     * @see Runtime
     */
    struct CopyLauncher {
    public:
      CopyLauncher(Predicate pred = Predicate::TRUE_PRED,
                   MapperID id = 0, MappingTagID tag = 0);
    public:
      inline unsigned add_copy_requirements(const RegionRequirement &src,
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

    /**
     * \struct FillLauncher
     * Fill launchers are objects that describe the parameters
     * for issuing a fill operation.
     * @see Runtime
     */
    struct FillLauncher {
    public:
      FillLauncher(void);
      FillLauncher(LogicalRegion handle, LogicalRegion parent,
                   TaskArgument arg, Predicate pred = Predicate::TRUE_PRED);
      FillLauncher(LogicalRegion handle, LogicalRegion parent,
                   Future f, Predicate pred = Predicate::TRUE_PRED);
    public:
      inline void set_argument(TaskArgument arg);
      inline void set_future(Future f);
      inline void add_field(FieldID fid);
      inline void add_grant(Grant g);
      inline void add_wait_barrier(PhaseBarrier bar);
      inline void add_arrival_barrier(PhaseBarrier bar);
    public:
      LogicalRegion                   handle;
      LogicalRegion                   parent;
      TaskArgument                    argument;
      Future                          future;
      Predicate                       predicate;
      std::set<FieldID>               fields;
      std::vector<Grant>              grants;
      std::vector<PhaseBarrier>       wait_barriers;
      std::vector<PhaseBarrier>       arrive_barriers;
    };

    //==========================================================================
    //                          Task Variant Registrars 
    //==========================================================================

    /**
     * \struct LayoutConstraintRegistrar
     * A layout description registrar is the mechanism by which application
     * can register a set of constraints with a specific ID. This ID can
     * then be used globally to refer to this set of constraints. All 
     * constraint sets are associated with a specifid field space which
     * contains the FieldIDs used in the constraints. This can be a 
     * NO_SPACE if there are no field constraints. All the rest of the 
     * constraints can be optionally specified.
     */
    struct LayoutConstraintRegistrar {
    public:
      LayoutConstraintRegistrar(void);
      LayoutConstraintRegistrar(FieldSpace handle, 
                                 const char *layout_name = NULL);
    public:
      inline LayoutConstraintRegistrar&
        add_constraint(const SpecializedConstraint &constraint);
      inline LayoutConstraintRegistrar&
        add_constraint(const MemoryConstraint &constraint);
      inline LayoutConstraintRegistrar&
        add_constraint(const OrderingConstraint &constraint);
      inline LayoutConstraintRegistrar&
        add_constraint(const SplittingConstraint &constraint);
      inline LayoutConstraintRegistrar&
        add_constraint(const FieldConstraint &constraint);
      inline LayoutConstraintRegistrar&
        add_constraint(const DimensionConstraint &constraint);
      inline LayoutConstraintRegistrar&
        add_constraint(const AlignmentConstraint &constraint);
      inline LayoutConstraintRegistrar&
        add_constraint(const OffsetConstraint &constraint);
      inline LayoutConstraintRegistrar&
        add_constraint(const PointerConstraint &constraint); 
    public:
      FieldSpace                                handle;
      LayoutConstraintSet                       layout_constraints;
      const char*                               layout_name;
    };

    /**
     * \struct TaskVariantRegistrar
     * This structure captures all the meta-data information necessary for
     * describing a task variant including the logical task ID, the execution
     * constraints, the layout constraints, and any properties of the task.
     * This structure is used for registering task variants and is also
     * the output type for variants created by task variant generators.
     */
    struct TaskVariantRegistrar {
    public:
      TaskVariantRegistrar(void);
      TaskVariantRegistrar(TaskID task_id, bool global = true,
                           GeneratorContext ctx = NULL,
                           const char *variant_name = NULL);
      TaskVariantRegistrar(TaskID task_id, const char *variant_name,
			   bool global = true,
                           GeneratorContext ctx = NULL);
    public: // Add execution constraints
      inline TaskVariantRegistrar& 
        add_constraint(const ISAConstraint &constraint);
      inline TaskVariantRegistrar&
        add_constraint(const ProcessorConstraint &constraint);
      inline TaskVariantRegistrar& 
        add_constraint(const ResourceConstraint &constraint);
      inline TaskVariantRegistrar&
        add_constraint(const LaunchConstraint &constraint);
      inline TaskVariantRegistrar&
        add_constraint(const ColocationConstraint &constraint);
    public: // Add layout constraint sets
      inline TaskVariantRegistrar&
        add_layout_constraint_set(unsigned index, LayoutConstraintID desc);
    public: // Set properties
      inline void set_leaf(bool is_leaf = true);
      inline void set_inner(bool is_inner = true);
      inline void set_idempotent(bool is_idempotent = true);
    public:
      TaskID                            task_id;
      GeneratorContext                  generator;
      bool                              global_registration;
      const char*                       task_variant_name;
    public: // constraints
      ExecutionConstraintSet            execution_constraints; 
      TaskLayoutConstraintSet           layout_constraints;
    public: // properties
      bool                              leaf_variant;
      bool                              inner_variant;
      bool                              idempotent_variant;
    };

    //==========================================================================
    //                          Task Generators 
    //==========================================================================

    /**
     * \struct TaskGeneratorArguments
     * This structure defines the arguments that will be passed to a 
     * task generator function. The task generator function will then
     * be expected to generate one or more variants and register them
     * with the runtime.
     */
    struct TaskGeneratorArguments {
    public:
      TaskID                            task_id;
    public:
      const void*                       user_data;
      size_t                            user_data_size;
    public:
      MapperID                          mapper_id;
      const void*                       mapper_args;
      size_t                            mapper_args_size;
    public:
      std::vector<Processor>            target_procs;
      // TODO: std::vector<PhysicalInstance> valid_instances;
    };

    /**
     * \struct TaskGeneratorRegistrar
     * The TaskGeneratorRegistrar records the arguments for registering
     * a new task generator function for a specific task_id with the runtime. 
     * The application must specify a unique generator ID to associate with 
     * the generator function pointer for the particular task ID, but the
     * same generator function pointer can be registered for multiple
     * task IDs as long as the generator IDs are all different. The 
     * application can also provide optional constraints on the kinds of
     * task variants the generator can emit. All other arguments are optional 
     * and can be filled in at the application's discretion.
     */
    struct TaskGeneratorRegistrar {
    public:
      TaskGeneratorRegistrar(void);
      TaskGeneratorRegistrar(GeneratorID id, TaskID task_id, GeneratorFnptr fn,
                             const void *udata = NULL, size_t udata_size = 0,
                             const char *name = NULL);
    public: // Add execution constraints
      inline TaskGeneratorRegistrar& 
        add_constraint(const ISAConstraint &constraint);
      inline TaskGeneratorRegistrar&
        add_constraint(const ProcessorConstraint &constraint);
      inline TaskGeneratorRegistrar& 
        add_constraint(const ResourceConstraint &constraint);
      inline TaskGeneratorRegistrar&
        add_constraint(const LaunchConstraint &constraint);
      inline TaskGeneratorRegistrar&
        add_constraint(const ColocationConstraint &constraint);
    public: // Add layout constraint sets
      inline TaskGeneratorRegistrar&
        add_layout_constraint_set(unsigned index, LayoutConstraintID desc);
    public:
      GeneratorID                       generator_id;
      TaskID                            task_id;
      GeneratorFnptr                    generator;
    public: // constraints
      ExecutionConstraintSet            execution_constraints;
      TaskLayoutConstraintSet           layout_constraints;
    public:
      const void*                       user_data;
      size_t                            user_data_size;
    public:
      const char*                       name;
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
      /**
       * Return the memories where the underlying physical instances locate.
       */
      void get_memories(std::set<Memory>& memories) const;
      /**
       * Return a list of fields that the physical region contains.
       */
      void get_fields(std::vector<FieldID>& fields) const;
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
      IndexIterator(const Domain &dom, ptr_t start = ptr_t());
      IndexIterator(Runtime *rt, Context ctx, IndexSpace space, ptr_t start = ptr_t());
      IndexIterator(Runtime *rt, Context ctx, LogicalRegion lr, ptr_t start = ptr_t());
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
      /**
       * Get the current point in the iterator and up to 'req_count'
       * additional points in the index space.  Returns the actual
       * count of contiguous points in 'act_count'.
       */
      inline ptr_t next_span(size_t& act_count, 
                             size_t req_count = (size_t)-1LL);
    public:
      IndexIterator& operator=(const IndexIterator &rhs);
    private:
      Enumerator *enumerator;
      bool finished;
      off_t current_pointer;
      size_t remaining_elmts;
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
    //                        Must Parallelism Classes
    //==========================================================================
    
    /**
     * \struct MustEpochLauncher
     * This is a meta-launcher object which contains other launchers.  The
     * purpose of this meta-launcher is to guarantee that all of the operations
     * specified in this launcher be guaranteed to run simultaneously.  This
     * enables the use of synchronization mechanisms such as phase barriers
     * and reservations between these operations without concern for deadlock.
     * If any condition is detected that will prevent simultaneous 
     * parallel execution of the operations the runtime will report an error.
     * These conditions include true data dependences on regions as well
     * as cases where mapping decisions artificially serialize operations
     * such as two tasks being mapped to the same processor.
     */
    struct MustEpochLauncher {
    public:
      MustEpochLauncher(MapperID id = 0, MappingTagID tag = 0);
    public:
      inline void add_single_task(const DomainPoint &point,
                                  const TaskLauncher &launcher);
      inline void add_index_task(const IndexLauncher &launcher);
    public:
      MapperID                        map_id;
      MappingTagID               mapping_tag;
      std::vector<TaskLauncher> single_tasks;
      std::vector<IndexLauncher> index_tasks;
    };

    //==========================================================================
    //                     MPI Interoperability Classes
    //==========================================================================

    class MPILegionHandshake {
    public:
      MPILegionHandshake(void);
      MPILegionHandshake(const MPILegionHandshake &rhs);
      ~MPILegionHandshake(void);
    public:
      class Impl;
    private:
      Impl *impl;
    protected:
      // Only the runtime should be able to make these
      FRIEND_ALL_RUNTIME_CLASSES
      MPILegionHandshake(Impl *impl);
    public:
      bool operator==(const MPILegionHandshake &h) const
        { return impl == h.impl; }
      bool operator<(const MPILegionHandshake &h) const
        { return impl < h.impl; }
      MPILegionHandshake& operator=(const MPILegionHandshake &rhs);
    public:
      /**
       * Non-blocking call to signal to Legion that this participant
       * is ready to pass control to Legion.
       */
      void mpi_handoff_to_legion(void);
      /**
       * A blocking call that will cause this participant to wait
       * for all Legion participants to hand over control to MPI.
       */
      void mpi_wait_on_legion(void);
    public:
      /**
       * A non-blocking call to signal to MPI that this participant
       * is ready to pass control to MPI.
       */
      void legion_handoff_to_mpi(void);
      /**
       * A blocking call that will cause this participant to wait
       * for all MPI participants to hand over control to Legion.
       */
      void legion_wait_on_mpi(void);
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
                                    Arrays::Point<DIM> &color_point) const;

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
     * \interface ProjectionFunctor
     * This defines an interface for objects that need to be
     * able to handle projection requests for an application.
     * Whenever index space tasks are launched with projection
     * region requirements, instances of this object are used
     * to handle the lowering down to individual regions for
     * specific task instances in the index space of task.
     * No more than one query of this interface will be made
     * per object at a time.
     *
     * Note also that the interface inherits from the 
     * RegionTreeInspector class which gives it access to 
     * all of the functions in that class for discovering
     * the shape of index space trees, field spaces, and
     * logical region trees.
     */
    class ProjectionFunctor {
    public:
      ProjectionFunctor(Runtime *rt);
      virtual ~ProjectionFunctor(void);
    public:
      /**
       * Compute the projection for a logical region projection
       * requirement down to a specific logical region.
       * @param ctx the context for this projection
       * @param task the task for the requested projection
       * @param index which region requirement we are projecting
       * @param upper_bound the upper bound logical region
       * @param point the point of the task in the index space
       * @return logical region to be used by the child task
       */
      virtual LogicalRegion project(Context ctx, Task *task, 
                                    unsigned index,
                                    LogicalRegion upper_bound,
                                    const DomainPoint &point) = 0;
      /**
       * Compute the projection for a logical partition projection
       * requirement down to a specific logical region.
       * @param ctx the context for this projection
       * @param task the task for the requested projection
       * @param index which region requirement we are projecting
       * @param upper_bound the upper bound logical partition
       * @param point the point of the task in the index space
       * @return logical region to be used by the child task
       */
      virtual LogicalRegion project(Context ctx, Task *task, 
                                    unsigned index,
                                    LogicalPartition upper_bound,
                                    const DomainPoint &point) = 0;
    protected:
      Runtime *const runtime;
    };

    /**
     * \class Runtime
     * The Runtime class is the primary interface for
     * Legion.  Every task is given a reference to the runtime as
     * part of its arguments.  All Legion operations are then
     * performed by directing the runtime to perform them through
     * the methods of this class.  The methods in Runtime
     * are broken into three categories.  The first group of
     * calls are the methods that can be used by application
     * tasks during runtime.  The second group contains calls
     * for initializing the runtime during start-up callback.
     * The final section of calls are static methods that are
     * used to configure the runtime prior to starting it up.
     */
    class Runtime {
    protected:
      // The Runtime bootstraps itself and should
      // never need to be explicitly created.
      friend class Internal;
      friend class Future;
      Runtime(Internal *rt);
    public:
      //------------------------------------------------------------------------
      // Index Space Operations
      //------------------------------------------------------------------------
      /**
       * Create a new unstructured index space
       * @param ctx the enclosing task context
       * @param max_num_elmts maximum number of elements in the index space
       * @return the handle for the new index space
       */
      IndexSpace create_index_space(Context ctx, size_t max_num_elmts);
      /**
       * Create a new structured index space based on a domain
       * @param ctx the enclosing task context
       * @param domain the domain for the new index space
       * @return the handle for the new index space
       */
      IndexSpace create_index_space(Context ctx, Domain domain);
      /**
       * Create a new structured index space based on a set of domains
       * @param ctx the enclosing task context
       * @param domains the set of domains
       * @return the handle for the new index space
       */
      IndexSpace create_index_space(Context ctx, 
                                    const std::set<Domain> &domains);
      /**
       * Destroy an existing index space
       * @param ctx the enclosing task context
       * @param handle the index space to destroy
       */
      void destroy_index_space(Context ctx, IndexSpace handle);
    public:
      //------------------------------------------------------------------------
      // Index Partition Operations Based on Coloring
      //------------------------------------------------------------------------
      /**
       * Create an index partition from a point coloring
       * @param ctx the enclosing task context
       * @param parent index space being partitioned
       * @param color_space space of colors for the partition
       * @param coloring the coloring of the parent index space
       * @param part_kind the kind of partition or whether to compute it
       * @param color optional color for the new partition
       * @param allocable whether the child index spaces are allocable
       * @return handle for the new index partition
       */
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                        const Domain &color_space,
                                        const PointColoring &coloring,
                                        PartitionKind part_kind = COMPUTE_KIND,
                                        int color = AUTO_GENERATE_ID,
                                        bool allocable = true);
      /**
       * @deprecated
       * See the previous create_index_partition call
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
       * Create an index partition from a domain point coloring
       * @param ctx the enclosing task context
       * @param parent the index space being partitioned
       * @param color_space space of colors for the partition
       * @param coloring the coloring of the parent index space
       * @param part_kind the kind of partition or whether to compute it
       * @param color optional color for the new partition
       * @return handle for the new index partition
       */
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                        const Domain &color_space,
                                        const DomainPointColoring &coloring,
                                        PartitionKind part_kind = COMPUTE_KIND,
                                        int color = AUTO_GENERATE_ID);
      /**
       * @deprecated
       * See the previous create index partition call
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
       * Create an index partition from a multi-domain point coloring
       * @param ctx the enclosing task context
       * @param parent the index space being partitioned
       * @param color_space space of colors for the partition
       * @param coloring the coloring of the parent index space
       * @param part_kind the kind of partition or whether to compute it
       * @param color optional color for the new partition
       * @return handle for the new index partition
       */
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                      const Domain &color_space,
                                      const MultiDomainPointColoring &coloring,
                                      PartitionKind part_kind = COMPUTE_KIND,
                                      int color = AUTO_GENERATE_ID);
      /**
       * @deprecated
       * See the previous create index partition call
       * Create an index partitiong from a domain color space and
       * a multi-domain coloring which allows multiple domains to
       * be associated with each color.
       * @param ctx the enclosing task context
       * @param parent index space being partitioned
       * @param color_space the domain of colors
       * @param coloring the multi-domain coloring
       * @param disjoint whether the partitioning is disjoint or not
       * @param part_color optional color name for the partition
       * @return handle for the next index partition
       */
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                            Domain color_space,
                                            const MultiDomainColoring &coloring,
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
					    int part_color = AUTO_GENERATE_ID);

      /**
       * @deprectated 
       * @see create_partition_by_field instead
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
      // Functional Index Partition Operations
      //------------------------------------------------------------------------

      /**
       * Create 'color_space' index subspaces (one for each point) in a 
       * common partition of the 'parent' index space. By definition the 
       * resulting partition will be disjoint. Users can also specify a 
       * minimum 'granularity' for the size of the index subspaces. Users 
       * can specify an optional color for the index partition. The 
       * 'allocable' field indicates whether dynamic allocation and 
       * freeing are supported on the index subspaces.
       * @param ctx the enclosing task context
       * @param parent index space of the partition to be made
       * @param color_space space of colors to create 
       * @param granularity the minimum size of the index subspaces
       * @param color optional color paramter for the partition
       * @param allocable whether dynamic allocation of pointers is permitted
       * @return name of the created index partition
       */
      IndexPartition create_equal_partition(Context ctx, IndexSpace parent,
                                            Domain color_space, 
                                            size_t granularity = 1,
                                            int color = AUTO_GENERATE_ID,
                                            bool allocable = false);

      /**
       * Create 'color_space' index subspaces (one for each point) in a 
       * common partition of the 'parent' index space. By definition the 
       * resulting partition will be disjoint. Users must also specify 
       * weights for the number of entries to put in each index spaces. 
       * The size of the 'weights' vector must be of the same as the
       * volume of the 'color_space' domain. Users can also specify a 
       * minimum 'granularity' of the size of the index subspaces. The 
       * 'color' field gives users optional control over the color of the 
       * created partition. The 'allocable' field indicates if dynamic 
       * allocation and freeing of pointers is supported on the index subspaces.
       * @param ctx the enclosing task context
       * @param parent index space of the partition to be made
       * @param color_space space of colors to create
       * @param weights map from colors to weights
       * @param granularity the minimum size of the index subspaces
       * @param color optional color paramter for the partition
       * @param allocable whether dynamic allocation of pointers is permitted
       * @return name of the created index partition
       */
      IndexPartition create_weighted_partition(Context ctx, IndexSpace parent,
                                       Domain color_space,
                                       const std::map<DomainPoint,int> &weights,
                                       size_t granularity = 1,
                                       int color = AUTO_GENERATE_ID,
                                       bool allocable = false); 

      /**
       * This function zips a union operation over all the index subspaces
       * in two different partitions. The zip operation is only applied 
       * to the points contained in the intersection of the two color
       * spaces. The corresponding pairs of index spaces are unioned
       * together and assigned to the same color in the new index
       * partition. The resulting partition is created off the 'parent'
       * index space. In order to be sound the parent must be an 
       * ancestor of both index partitions. The kind of partition 
       * (e.g. disjoint or aliased) can be specified with the 'part_kind'
       * argument. This argument can also be used to request that the 
       * runtime compute the kind of partition. The user can assign
       * a color to the new partition by the 'color' argument. The
       * 'allocable' argument controls if the new index subspaces
       * support dynamic allocation and freeing of pointers. Note that
       * this call only works for unstructured index spaces.
       * @param ctx the enclosing task context
       * @param parent the parent index space for the new partition
       * @param handle1 first index partition
       * @param handle2 second index partition
       * @param part_kind indicate the kind of partition
       * @param color the new color for the index partition
       * @param allocable whether dynamic allocation of pointers is permitted
       * @return name of the created index partition
       */
      IndexPartition create_partition_by_union(Context ctx,
                                       IndexSpace parent,
                                       IndexPartition handle1,
                                       IndexPartition handle2,
                                       PartitionKind part_kind = COMPUTE_KIND,
                                       int color = AUTO_GENERATE_ID,
                                       bool allocable = false);

      /**
       * This function zips an intersection operation over all the index 
       * subspaces in two different partitions. The zip operation is only
       * applied to points contained in the intersection of the two 
       * color spaces. The corresponding pairs of index spaces from each
       * partition are intersected together and assigned to the same
       * color in the new index partition. The resulting partition is
       * created off the 'parent' index space. In order to be sound both
       * index partitions must come from the same index tree as the
       * parent and at least one must have the 'parent' index space as
       * an ancestor. The user can say whether the partition is disjoint
       * or not or ask the runtime to compute the result using the 
       * 'part_kind' argument. The user can assign a color to the new 
       * partition by the 'color' argument. The 'allocable' argument 
       * controls if the new index subspaces support dynamic allocation 
       * and freeing of pointers.
       * @param ctx the enclosing task context
       * @param parent the parent index space for the new partition
       * @param handle1 first index partition
       * @param handle2 second index partition
       * @param part_kind indicate the kind of partition
       * @param color the new color for the index partition
       * @param allocable whether dynamic allocation of pointers is permitted
       * @return name of the created index partition
       */
      IndexPartition create_partition_by_intersection(Context ctx,
                                        IndexSpace parent,
                                        IndexPartition handle1,
                                        IndexPartition handle2,
                                        PartitionKind part_kind = COMPUTE_KIND,
                                        int color = AUTO_GENERATE_ID,
                                        bool allocable = false);

      /**
       * This function zips a set difference operation over all the index 
       * subspaces in two different partitions. The zip operation is only
       * applied to the points contained in the intersection of the two
       * color spaces. The difference is taken from the corresponding 
       * pairs of index spaces from each partition. The resulting partition
       * is created off the 'parent' index space. In order to be sound,
       * both index partitions must be from the same index tree and the
       * first index partition must have the 'parent' index space as an
       * ancestor. The user can say whether the partition is disjoint or
       * not or ask the runtime to compute the result using the 'part_kind'
       * argument. The user can assign a color to the new partition by
       * the 'color' argument. The 'allocable' argument controls if the
       * new index subspaces support dynamic allocation and freeing of 
       * pointers. Note that this call only works for unstructured
       * index spaces.
       * @param ctx the enclosing task context
       * @param parent the parent index space for the new partition
       * @param handle1 first index partition
       * @param handle2 second index partition
       * @param part_kind indicate the kind of partition
       * @param color the new color for the index partition
       * @param allocable whether dynamic allocation of pointers is permitted
       * @return name of the created index partition
       */
      IndexPartition create_partition_by_difference(Context ctx,
                                        IndexSpace parent,
                                        IndexPartition handle1,
                                        IndexPartition handle2,
                                        PartitionKind part_kind = COMPUTE_KIND,
                                        int color = AUTO_GENERATE_ID,
                                        bool allocable = false);

      /**
       * This performs a cross product between two different index
       * partitions. For every index subspace in the first index 
       * partition the runtime will create a new index partition
       * of that index space with each of the different index subspaces
       * in the second index partition. As a result, whole set of new
       * index partitions will be created. The user can request which
       * partition names to return by specifying a map of domain points
       * from the color space of the first index partition. If the map
       * is empty, no index partitions will be returned. The user can
       * can say what kind the partitions are using the 'part_kind'
       * argument. The user can also specify a color for the new partitions
       * using the 'color' argument. If a specific color is specified, it
       * must be available for a partition in each of the index subspaces
       * in the first index partition. The user can specify whether the new 
       * index partitions support dynamic allocation and freeing of pointers 
       * with the 'allocable' argument.
       * @param ctx the enclosing task context
       * @param handle1 the first index partition
       * @param handle2 the second index partition
       * @param handle optional map for new partitions (can be empty)
       * @param part_kind indicate the kinds for the partitions
       * @param color optional color for each of the new partitions
       * @param allocable whether dynamic allocation of pointers is permitted
       */
      void create_cross_product_partitions(Context ctx,
                                  IndexPartition handle1,
                                  IndexPartition handle2,
                                  std::map<DomainPoint,IndexPartition> &handles,
                                  PartitionKind part_kind = COMPUTE_KIND,
                                  int color = AUTO_GENERATE_ID,
                                  bool allocable = false);

      /**
       * Create partition by field uses an existing field in a logical
       * region to perform a partitioning operation. The field type
       * must be of 'Color' type (or int type) so that the runtime can 
       * interpret the field as an enumerated function from 
       * ptr_t -> Color. Pointers are assigned into index subspaces
       * based on their assigned color. Pointers with negative entries
       * will not be assigned into any index subspace. The resulting
       * index partition is a partition of the index space of the logical 
       * region over which the operation is performed. By definition
       * this partition is disjoint. The 'color' argument can be used
       * to specify an optional color for the index partition. The user 
       * can control whether the index subspaces support dynamic allocation 
       * and freeing of pointers using the 'allocable' argument. This
       * operation is illegal to perform on structured index spaces.
       * Note that this call only works for unstructured index spaces.
       * @param ctx the enclosing task context
       * @param handle logical region handle containing the chosen
       *               field and of which a partition will be created
       * @param parent the parent region from which privileges are derived
       * @param fid the field ID of the logical region containing the coloring
       * @param color_space space of colors for the partition
       * @param color optional new color for the index partition
       * @param allocable whether dynamic allocation of pointers is permitted
       * @return a new index partition of the index space of the logical region
       */
      IndexPartition create_partition_by_field(Context ctx,
                                               LogicalRegion handle,
                                               LogicalRegion parent,
                                               FieldID fid,
                                               const Domain &color_space,
                                               int color = AUTO_GENERATE_ID,
                                               bool allocable = false);

      /**
       * Create partition by image creates a new index partition from an
       * existing field that represents an enumerated function from 
       * pointers into the logical region containing the field 'fid'
       * to pointers in the 'handle' index space. The function the field
       * represents therefore has type ptr_t@projection -> ptr_t@handle.
       * We can therefore create a new index partition of 'handle' by
       * mapping each of the pointers in the index subspaces in the
       * index partition of the 'projection' logical partition to get
       * pointers into the 'handle' index space and assigning them to
       * a corresponding index subspace. The runtime will automatically
       * compute if the resulting partition is disjoint or not. The
       * user can give the new partition a color by specifying the 
       * 'color' argument. The user can also control whether dynamic
       * allocation and freeing of pointers is available on the new
       * index subspaces using the 'allocable' field. This method is
       * illegal to perform on structured index spaces. Note that this
       * call only works for unstructured index spaces.
       * @param ctx the enclosing task context
       * @param handle the parent index space of the new index partition
       *               and the one in which all the ptr_t contained in
       *               'fid' must point
       * @param projection the logical partition of which we are creating
       *                   a projected version of through the field
       * @param parent the parent region from which privileges are derived
       * @param fid the field ID of the 'projection' logical partition
       *            we are reading which contains ptr_t@handle
       * @param part_kind specify the kind of partition
       * @param color optional new color for the index partition
       * @param allocable whether dynamic allocation of pointers is permitted
       * @return a new index partition of the 'handle' index space
       */
      IndexPartition create_partition_by_image(Context ctx,
                                         IndexSpace handle,
                                         LogicalPartition projection,
                                         LogicalRegion parent,
                                         FieldID fid,
                                         const Domain &color_space,
                                         PartitionKind part_kind = COMPUTE_KIND,
                                         int color = AUTO_GENERATE_ID,
                                         bool allocable = false);
                                               
      /**
       * Create partition by premimage performs the opposite operation
       * of create partition by image. The function will create a new
       * index partition of the index space of 'handle' by projecting it
       * through another index space 'projection'. The field contained in
       * 'fid' of the logical region 'handle' must contain pointers into
       * 'projection' index space. For each 'pointer' in the index space
       * of 'handle', this function will compute its equivalent pointer
       * into the index space tree of 'projection' by looking it up in
       * the field 'fid'. The input pointer will be assigned to analogous
       * index subspaces for each of the index subspaces in 'projection'
       * that its projected pointer belonged to. The runtime will compute
       * if the resulting partition is disjoint. The user can also assign
       * a color to the new index partition with the 'color' argument.
       * The 'allocable' argument controls whether dynamic pointer allocation
       * and freeing is supported on the new index subspaces. This
       * method is illegal to perform on structured index spaces.
       * Note that this call only works for unstructured index spaces.
       * @param ctx the enclosing task context
       * @param projection the index partition being projected
       * @param handle logical region over which to evaluate the function
       * @param parent the parent region from which privileges are derived
       * @param fid the field ID of the 'handle' logical region containing
       *            the function being evaluated
       * @param part_kind specify the kind of partition
       * @param color optional new color for the index partition
       * @param allocable wehther dynamic allocation of pointers is permitted
       * @return a new index partition of the index space of 'handle'
       */
      IndexPartition create_partition_by_preimage(Context ctx, 
                                        IndexPartition projection,
                                        LogicalRegion handle,
                                        LogicalRegion parent,
                                        FieldID fid,
                                        const Domain &color_space,
                                        PartitionKind part_kind = COMPUTE_KIND,
                                        int color = AUTO_GENERATE_ID,
                                        bool allocable = false);
    public:
      //------------------------------------------------------------------------
      // Computed Index Spaces and Partitions 
      //------------------------------------------------------------------------

      /**
       * Create a new index partition in which the individual sub-regions
       * will computed by one of the following calls:
       *  - create_index_space_union (2 variants)
       *  - create_index_space_intersection (2 variants)
       *  - create_index_space_difference
       * Once this call is made the application can immediately retrieve
       * the names of the new sub-regions corresponding to each the different
       * colors in the color space. However, it is the responsibility of
       * the application to ensure that it actually computes an index space
       * for each of the colors. Undefined behavior will result if the 
       * application tries to assign to a color of an index space more 
       * than once. If the runtime is asked to compute the disjointness,
       * the application must assign values to each of the different subspace
       * colors before querying the disjointness or deadlock will likely
       * result (unless a different task is guaranteed to compute any
       * remaining index subspaces). Note that this call only works for
       * unstructured index spaces.
       * @param ctx the enclosing task context
       * @param parent the parent index space for the partition
       * @param color_space the color space for the new partition
       * @param part_kind optionally specify the partition kind
       * @param color optionally assign a color to the partition
       * @param allocable specify if the sub-regions can allocate
       * @return handle of the new index partition
       */
      IndexPartition create_pending_partition(Context ctx,
                                              IndexSpace parent,
                                              const Domain &color_space,
                                  PartitionKind part_kind = COMPUTE_KIND, 
                                  int color = AUTO_GENERATE_ID,
                                  bool allocable = false);

      /**
       * Create a new index space by unioning together a bunch of index spaces 
       * from a common index space tree. The resulting index space is assigned
       * to be the index space corresponding to 'color' of the 'parent' index
       * partition. It is illegal to invoke this method with a 'parent' index
       * partition that was not created by a 'create_pending_partition' call.
       * All of the index spaces being unioned together must come from the
       * same index space tree. Note that this call only works for 
       * unstructured index spaces.
       * @param ctx the enclosing task context
       * @param parent the parent index partition 
       * @param color the color to assign the index space to in the parent
       * @param handles a vector index space handles to union
       * @return handle of the new index space
       */
      IndexSpace create_index_space_union(Context ctx, IndexPartition parent, 
                                        const DomainPoint &color,
                                        const std::vector<IndexSpace> &handles);

      /**
       * This method is the same as the one above, except the index
       * spaces all come from a common partition specified by 'handle'.
       * The resulting index space will be a union of all the index
       * spaces of 'handle'.
       * @param ctx the enlcosing task context
       * @param parent the parent index partition 
       * @param color the color to assign the index space to in the parent
       * @param handle an index partition to union together
       * @return handle of the new index space
       */
      IndexSpace create_index_space_union(Context ctx, IndexPartition parent,
                                          const DomainPoint &color,
                                          IndexPartition handle);

      /**
       * Create a new index space by intersecting together a bunch of index
       * spaces from a common index space tree. The resulting index space is
       * assigned to be the index space corresponding to 'color' of the 
       * 'parent' index partition. It is illegal to invoke this method with
       * a 'parent' index partition that was not created by a call to 
       * 'create_pending_partition'. All of the index spaces being 
       * intersected together must come from the same index space tree.
       * Note that this call only works for unstructured index spaces.
       * @param ctx the enclosing task context
       * @param parent the parent index partition
       * @param color the color to assign the index space to in the parent
       * @param handles a vector index space handles to intersect 
       * @return handle of the new index space
       */
      IndexSpace create_index_space_intersection(Context ctx, 
                                                 IndexPartition parent,
                                                 const DomainPoint &color,
                                       const std::vector<IndexSpace> &handles);

      /**
       * This method is the same as the one above, except the index
       * spaces all come from a common partition specified by 'handle'.
       * The resulting index space will be an intersection of all the index
       * spaces of 'handle'. Note that this call only works for unstructured
       * index spaces.
       * @param ctx the enlcosing task context
       * @param parent the parent index partition
       * @param color the color to assign the index space to in the parent
       * @param handle an index partition to intersect together
       * @return handle of the new index space
       */
      IndexSpace create_index_space_intersection(Context ctx, 
                                                 IndexPartition parent,
                                                 const DomainPoint &color,
                                                 IndexPartition handle);

      /**
       * Create a new index space by taking the set difference of
       * an existing index space with a set of other index spaces.
       * The resulting index space is assigned to be the index space
       * corresponding to 'color' of the 'parent' index partition.
       * It is illegal to invoke this method with a 'parent' index
       * partition that was not created by a call to 'create_index_partition'.
       * The 'initial' index space is the index space from which 
       * differences will be performed, and each of the index spaces in 
       * 'handles' will be subsequently subtracted from the 'initial' index
       * space. All of the index spaces in 'handles' as well as 'initial'
       * must come from the same index space tree. Note that this call
       * only works for unstructured index spaces.
       * @param ctx the enclosing task context
       * @param parent the parent index partition
       * @param color the color to assign the index space to in the parent
       * @param initial the starting index space
       * @param handles a vector index space handles to subtract 
       * @return handle of the new index space
       */
      IndexSpace create_index_space_difference(Context ctx, 
                                               IndexPartition parent,
                                               const DomainPoint &color,
                                               IndexSpace initial,
                                        const std::vector<IndexSpace> &handles);
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
      IndexPartition get_index_partition(Context ctx, IndexSpace parent,
                                         const DomainPoint &color);
      // Context free versions
      IndexPartition get_index_partition(IndexSpace parent, Color color);
      IndexPartition get_index_partition(IndexSpace parent, 
                                         const DomainPoint &color);
      /**
       * Return true if the index space has an index partition
       * with the specified color.
       * @param ctx enclosing task context
       * @param parent index space
       * @param color of index partition
       * @return true if an index partition exists with the specified color
       */
      bool has_index_partition(Context ctx, IndexSpace parent,
                               const DomainPoint &color);
      // Context free
      bool has_index_partition(IndexSpace parent,
                               const DomainPoint &color);

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
      IndexSpace get_index_subspace(Context ctx, IndexPartition p,
                                    const DomainPoint &color);
      // Context free versions
      IndexSpace get_index_subspace(IndexPartition p, Color color);
      IndexSpace get_index_subspace(IndexPartition p,
                                    const DomainPoint &color);

      /**
       * Return true if the index partition has an index subspace
       * with the specified color.
       * @param ctx enclosing task context
       * @param p parent index partitioning
       * @param color of the index sub-space
       * @return true if an index space exists with the specified color
       */
      bool has_index_subspace(Context ctx, IndexPartition p,
                              const DomainPoint &color);
      // Context free
      bool has_index_subspace(IndexPartition p,
                              const DomainPoint &color);

      /**
       * Return if the given index space is represented by 
       * multiple domains or just a single one. If multiple
       * domains represent the index space then 'get_index_space_domains'
       * should be used for getting the set of domains.
       * @param ctx enclosing task context
       * @param handle index space handle
       * @return true if the index space has multiple domains
       */
      bool has_multiple_domains(Context ctx, IndexSpace handle);
      // Context free
      bool has_multiple_domains(IndexSpace handle);

      /**
       * Return the domain corresponding to the
       * specified index space if it exists
       * @param ctx enclosing task context
       * @param handle index space handle
       * @return the domain corresponding to the index space
       */
      Domain get_index_space_domain(Context ctx, IndexSpace handle);
      // Context free
      Domain get_index_space_domain(IndexSpace handle);

      /**
       * Return the domains that represent the index space.
       * While the previous call only works when there is a
       * single domain for the index space, this call will
       * work in all circumstances.
       * @param ctx enclosing task context
       * @param handle index space handle
       * @param vector to populate with domains
       */
      void get_index_space_domains(Context ctx, IndexSpace handle,
                                   std::vector<Domain> &domains);
      // Context free
      void get_index_space_domains(IndexSpace handle,
                                   std::vector<Domain> &domains);

      /**
       * Return a domain that represents the color space
       * for the specified partition.
       * @param ctx enclosing task context
       * @param p handle for the index partition
       * @return a domain for the color space of the specified partition
       */
      Domain get_index_partition_color_space(Context ctx, IndexPartition p);
      // Context free
      Domain get_index_partition_color_space(IndexPartition p);

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
      void get_index_space_partition_colors(Context ctx, IndexSpace sp,
                                            std::set<DomainPoint> &colors);
      // Context free versions
      void get_index_space_partition_colors(IndexSpace sp,
                                            std::set<Color> &colors);
      void get_index_space_partition_colors(IndexSpace sp,
                                            std::set<DomainPoint> &colors);

      /**
       * Return whether a given index partition is disjoint
       * @param ctx enclosing task context
       * @param p index partition handle
       * @return whether the index partition is disjoint
       */
      bool is_index_partition_disjoint(Context ctx, IndexPartition p);
      // Context free
      bool is_index_partition_disjoint(IndexPartition p);

      /**
       * Return whether a given index partition is complete
       * @param ctx enclosing task context
       * @param p index partition handle
       * @return whether the index partition is complete
       */
      bool is_index_partition_complete(Context ctx, IndexPartition p);
      // Context free
      bool is_index_partition_complete(IndexPartition p);

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
      DomainPoint get_index_space_color_point(Context ctx, IndexSpace handle);
      // Context free
      Color get_index_space_color(IndexSpace handle);
      DomainPoint get_index_space_color_point(IndexSpace handle);

      /**
       * Return the color for the corresponding index partition in
       * in relation to its parent logical region.
       * @param ctx enclosing task context
       * @param handle the index partition for which to find the color
       * @return the color for the index partition
       */
      Color get_index_partition_color(Context ctx, IndexPartition handle);
      DomainPoint get_index_partition_color_point(Context ctx,
                                                  IndexPartition handle);
      // Context free
      Color get_index_partition_color(IndexPartition handle);
      DomainPoint get_index_partition_color_point(IndexPartition handle);

      /**
       * Return the index space parent for the given index partition.
       * @param ctx enclosing task context
       * @param handle for the index partition
       * @return index space for the parent
       */
      IndexSpace get_parent_index_space(Context ctx, IndexPartition handle);
      // Context free
      IndexSpace get_parent_index_space(IndexPartition handle);

      /**
       * Returns true if the given index space has a parent partition.
       * @param ctx enclosing task context
       * @param handle for the index space
       * @return true if there is a parent index partition
       */
      bool has_parent_index_partition(Context ctx, IndexSpace handle);
      // Context free
      bool has_parent_index_partition(IndexSpace handle);

      /**
       * Returns the parent partition for the given index space.
       * Use the previous call to check to see if a parent actually exists.
       * @param ctx enclosing task context
       * @param handle for the index space
       * @return the parent index partition
       */
      IndexPartition get_parent_index_partition(Context ctx, IndexSpace handle);
      // Context free
      IndexPartition get_parent_index_partition(IndexSpace handle);
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

      /**
       * Get the size of a specific field within field space.
       * @param ctx enclosing task context
       * @param handle field space handle
       * @param fid field ID for which to find the size
       * @return the size of the field in bytes
       */
      size_t get_field_size(Context ctx, FieldSpace handle, FieldID fid);
      // Context free
      size_t get_field_size(FieldSpace handle, FieldID fid);

      /**
       * Get the IDs of the fields currently allocated in a field space.
       * @param ctx enclosing task context
       * @param handle field space handle
       * @param set in which to place the field IDs
       */
      void get_field_space_fields(Context ctx, FieldSpace handle,
                                  std::set<FieldID> &fields);
      // Context free
      void get_field_space_fields(FieldSpace handle,
                                  std::set<FieldID> &fields);
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
      // Context free
      LogicalPartition get_logical_partition(LogicalRegion parent,
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
      LogicalPartition get_logical_partition_by_color(Context ctx,
                                                      LogicalRegion parent,
                                                      const DomainPoint &c);
      // Context free
      LogicalPartition get_logical_partition_by_color(LogicalRegion parent, 
                                                      Color c);
      LogicalPartition get_logical_partition_by_color(LogicalRegion parent,
                                                      const DomainPoint &c);

      /**
       * Return true if the logical region has a logical partition with
       * the specified color.
       * @param ctx enclosing task context
       * @param parent logical region
       * @param color for the specified logical partition
       * @return true if the logical partition exists with the specified color
       */
      bool has_logical_partition_by_color(Context ctx,
                                          LogicalRegion parent,
                                          const DomainPoint &c);
      // Context free
      bool has_logical_partition_by_color(LogicalRegion parent,
                                          const DomainPoint &c);
      
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
      // Context free
      LogicalPartition get_logical_partition_by_tree(IndexPartition handle, 
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
      // Context free
      LogicalRegion get_logical_subregion(LogicalPartition parent, 
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
      LogicalRegion get_logical_subregion_by_color(Context ctx,
                                                   LogicalPartition parent,
                                                   const DomainPoint &c);
      // Context free
      LogicalRegion get_logical_subregion_by_color(LogicalPartition parent, 
                                                   Color c);
      LogicalRegion get_logical_subregion_by_color(LogicalPartition parent,
                                                   const DomainPoint &c);

      /**
       * Return true if the logical partition has a logical region with
       * the specified color.
       * @param ctx enclosing task context
       * @param parent logical partition
       * @param color for the specified logical region
       * @return true if a logical region exists with the specified color
       */
      bool has_logical_subregion_by_color(Context ctx,
                                          LogicalPartition parent,
                                          const DomainPoint &c);
      // Context free
      bool has_logical_subregion_by_color(LogicalPartition parent,
                                          const DomainPoint &c);

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
      // Context free
      LogicalRegion get_logical_subregion_by_tree(IndexSpace handle, 
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
      DomainPoint get_logical_region_color_point(Context ctx, 
                                                 LogicalRegion handle);
      // Context free versions
      Color get_logical_region_color(LogicalRegion handle);
      DomainPoint get_logical_region_color_point(LogicalRegion handle);

      /**
       * Return the color for the logical partition corresponding to
       * its location relative to the parent logical region.
       * @param ctx enclosing task context
       * @param handle the logical partition handle for which to find the color
       * @return the color for the logical partition
       */
      Color get_logical_partition_color(Context ctx, LogicalPartition handle);
      DomainPoint get_logical_partition_color_point(Context ctx,
                                                    LogicalPartition handle);
      // Context free versions
      Color get_logical_partition_color(LogicalPartition handle);
      DomainPoint get_logical_partition_color_point(LogicalPartition handle);

      /**
       * Return the parent logical region for a given logical partition.
       * @param ctx enclosing task context
       * @param handle the logical partition handle for which to find a parent
       * @return the parent logical region
       */
      LogicalRegion get_parent_logical_region(Context ctx, 
                                              LogicalPartition handle);
      // Context free
      LogicalRegion get_parent_logical_region(LogicalPartition handle);

      /**
       * Return true if the logical region has a parent logical partition.
       * @param ctx enclosing task context
       * @param handle for the logical region for which to check for a parent
       * @return true if a parent exists
       */
      bool has_parent_logical_partition(Context ctx, LogicalRegion handle);
      // Context free
      bool has_parent_logical_partition(LogicalRegion handle);

      /**
       * Return the parent logical partition for a logical region.
       * @param ctx enclosing task context
       * @param handle for the logical region for which to find a parent
       * @return the parent logical partition
       */
      LogicalPartition get_parent_logical_partition(Context ctx, 
                                                    LogicalRegion handle);
      // Context free
      LogicalPartition get_parent_logical_partition(LogicalRegion handle);
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
       * Unmap all the regions originally requested for a context (if
       * they haven't already been unmapped). WARNING: this call will
       * invalidate all accessors currently valid in the enclosing
       * parent task context.
       * @param ctx enclosing task context
       */
      void unmap_all_regions(Context ctx);
    public:
      //------------------------------------------------------------------------
      // Fill Field Operations
      //------------------------------------------------------------------------
      /**
       * Fill the specified field by setting all the entries in the index
       * space from the given logical region to a specified value. Importantly
       * this operation is done lazily so that the writes only need to happen
       * the next time the field is used and therefore it is a very 
       * inexpensive operation to perform. This operation requires read-write
       * privileges on the requested field.
       * @param ctx enclosing task context
       * @param handle the logical region on which to fill the field
       * @param parent the parent region from which privileges are derived
       * @param fid the field to fill 
       * @param value the value to assign to all the entries
       * @param pred the predicate for this operation
       */
      template<typename T>
      void fill_field(Context ctx, LogicalRegion handle, LogicalRegion parent, 
                      FieldID fid, const T &value, 
                      Predicate pred = Predicate::TRUE_PRED);

      /**
       * This version of fill field is exactly the same as the one above,
       * but is untyped and allows the value to be specified as a buffer
       * with a size. The runtime will make a copy of the buffer. This
       * operation requires read-write privileges on the field.
       * @param ctx enclosing task context
       * @param handle the logical region on which to fill the field
       * @param parent the parent region from which privileges are derived
       * @param fid the field to fill 
       * @param value pointer to the buffer containing the value to be used
       * @param value_size size of the buffer in bytes
       * @param pred the predicate for this operation
       */
      void fill_field(Context ctx, LogicalRegion handle, LogicalRegion parent,
                      FieldID fid, const void *value, size_t value_size,
                      Predicate pred = Predicate::TRUE_PRED);

      /**
       * This version of fill field is exactly the same as the one above,
       * but uses a future value. This operation requires read-write privileges 
       * on the field.
       * @param ctx enclosing task context
       * @param handle the logical region on which to fill the field
       * @param parent the parent region from which privileges are derived
       * @param fid the field to fill 
       * @param value pointer to the buffer containing the value to be used
       * @param value_size size of the buffer in bytes
       * @param pred the predicate for this operation
       */
      void fill_field(Context ctx, LogicalRegion handle, LogicalRegion parent,
                      FieldID fid, Future f, 
                      Predicate pred = Predicate::TRUE_PRED);

      /**
       * Fill multiple fields of a logical region with the same value.
       * This operation requires read-write privileges on the fields.
       * @param ctx enclosing task context
       * @param handle the logical region on which to fill the field
       * @param parent the parent region from which privileges are derived
       * @param fields the set of fields to fill 
       * @param value the value to assign to all the entries
       * @param pred the predicate for this operation
       */
      template<typename T>
      void fill_fields(Context ctx, LogicalRegion handle, LogicalRegion parent,
                        const std::set<FieldID> &fields, const T &value,
                        Predicate pred = Predicate::TRUE_PRED);

      /**
       * Fill multiple fields of a logical region with the same value.
       * The runtime will make a copy of the buffer passed. This operation
       * requires read-write privileges on the fields.
       * @param ctx enclosing task context
       * @param handle the logical region on which to fill the field
       * @param parent the parent region from which privileges are derived
       * @param fields the set of fields to fill
       * @param value pointer to the buffer containing the value to be used
       * @param value_size size of the buffer in bytes
       * @param pred the predicate for this operation
       */
      void fill_fields(Context ctx, LogicalRegion handle, LogicalRegion parent,
                       const std::set<FieldID> &fields, 
                       const void *value, size_t value_size,
                       Predicate pred = Predicate::TRUE_PRED);

      /**
       * Fill multiple fields of a logical region with the same future value.
       * This operation requires read-write privileges on the fields.
       * @param ctx enclosing task context
       * @param handle the logical region on which to fill the field
       * @param parent the parent region from which privileges are derived
       * @param fields the set of fields to fill
       * @param future the future value to use for filling the fields
       * @param pred the predicate for this operation
       */
      void fill_fields(Context ctx, LogicalRegion handle, LogicalRegion parent,
                       const std::set<FieldID> &fields,
                       Future f, Predicate pred = Predicate::TRUE_PRED);

      /**
       * Perform a fill operation using a launcher which specifies
       * all of the parameters of the launch.
       * @param ctx enclosing task context
       * @param launcher the launcher that describes the fill operation
       */
      void fill_fields(Context ctx, const FillLauncher &launcher);
    public:
      //------------------------------------------------------------------------
      // File Operations
      //------------------------------------------------------------------------
      /**
       * Attach an HDF5 file as a physical region. The file must already 
       * exist. Legion will defer the attach operation until all other
       * operations on the logical region are finished. After the attach
       * operation succeeds, then all other physical instances for the 
       * logical region will be invalidated, making the physical instance
       * the only valid version of the logical region. The resulting physical 
       * instance is attached with restricted coherence (the same as logical 
       * regions mapped with simultaneous coherence). All operations using 
       * the logical region will be required to use the physical instance
       * until the restricted coherence is removed using an acquire 
       * operation. The restricted coherence can be reinstated by
       * performing a release operation. Just like other physical regions,
       * the HDF5 file can be both mapped and unmapped after it is created.
       * The runtime will report an error for an attempt to attach an file
       * to a logical region which is already mapped in the enclosing
       * parent task's context. The runtime will also report an error if
       * the task launching the attach operation does not have the 
       * necessary privileges (read-write) on the logical region.
       * The resulting physical region is unmapped, but can be mapped
       * using the standard inline mapping calls.
       * @param ctx enclosing task context
       * @param file_name the path to an existing HDF5 file
       * @param handle the logical region with which to associate the file
       * @param parent the parent logical region containing privileges
       * @param field_map mapping for field IDs to HDF5 dataset names
       * @param mode the access mode for attaching the file
       * @return a new physical instance corresponding to the HDF5 file
       */
      PhysicalRegion attach_hdf5(Context ctx, const char *file_name,
                                 LogicalRegion handle, LogicalRegion parent, 
                                 const std::map<FieldID,const char*> &field_map,
                                 LegionFileMode mode);

      /**
       * Detach an HDF5 file. This can only be performed on a physical
       * region that was created by calling attach_hdf5. The runtime
       * will properly defer the detach call until all other operations
       * on the logical region are complete. It is the responsibility of
       * the user to perform the necessary operations to flush any data
       * back to the physical instance before detaching (e.g. releasing
       * coherence, etc). If the physical region is still mapped when
       * this function is called, then it will be unmapped by this call.
       * Note that this file may not actually get detached until much 
       * later in the execution of the program due to Legion's deferred 
       * execution model.
       * @param ctx enclosing task context 
       * @param region the physical region for an HDF5 file to detach
       */
      void detach_hdf5(Context ctx, PhysicalRegion region);

      /**
       * Attach an normal file as a physical region. This attach is similar to
       * attach_hdf5 operation, except that the file has exact same data format
       * as in-memory physical region. Data lays out as SOA in file.
       */
      PhysicalRegion attach_file(Context ctx, const char *file_name,
                                 LogicalRegion handle, LogicalRegion parent,
                                 const std::vector<FieldID> &field_vec,
                                 LegionFileMode mode);

      /**
       * Detach an normal file. THis detach operation is similar to
       * detach_hdf5
       */
      void detach_file(Context ctx, PhysicalRegion region);
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
       * Create a new phase barrier with an expected number of 
       * arrivals.  Note that this number of arrivals 
       * is the number of arrivals performed on each generation
       * of the phase barrier and cannot be changed.
       * @param ctx enclosing task context
       * @param arrivals number of arrivals on the barrier 
       * @return a new phase barrier handle
       */
      PhaseBarrier create_phase_barrier(Context ctx, unsigned arrivals);

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
      // Dynamic Collective operations
      //------------------------------------------------------------------------
      /**
       * A dynamic collective is a special type of phase barrier that 
       * is also associated with a reduction operation that allows arrivals
       * to contribute a value to a generation of the barrier. The runtime
       * reduces down all the applied values to a common value for each
       * generation of the phase barrier. The number of arrivals gives a
       * default number of expected arrivals for each generation.
       * @param ctx enclosing task context
       * @param arrivals default number of expected arrivals 
       * @param redop the associated reduction operation
       * @param init_value the inital value for each generation
       * @param init_size the size in bytes of the initial value
       * @return a new dynamic collective handle
       */
      DynamicCollective create_dynamic_collective(Context ctx, 
                                                  unsigned arrivals,
                                                  ReductionOpID redop,
                                                  const void *init_value,
                                                  size_t init_size);

      /**
       * Destroy a dynamic collective operation. It has the
       * same semantics as the destruction of a phase barrier.
       * @param ctx enclosing task context
       * @param dc dynamic collective to destroy
       */
      void destroy_dynamic_collective(Context ctx, DynamicCollective dc);

      /**
       * Arrive on a dynamic collective immediately with a value
       * stored in an untyped buffer.
       * @param ctx enclosing task context
       * @param dc dynamic collective on which to arrive
       * @param buffer pointer to an untyped buffer
       * @param size size of the buffer in bytes
       * @param count arrival count on the barrier
       */
      void arrive_dynamic_collective(Context ctx,
                                     DynamicCollective dc,
                                     const void *buffer, 
                                     size_t size, unsigned count = 1);

      /**
       * Perform a deferred arrival on a dynamic collective dependent
       * upon a future value.  The runtime will automatically pipe the
       * future value through to the dynamic collective.
       * @param ctx enclosing task context
       * @param dc dynamic collective on which to arrive
       * @param f future to use for performing the arrival
       * @param count total arrival count
       */
      void defer_dynamic_collective_arrival(Context ctx, 
                                            DynamicCollective dc,
                                            Future f, unsigned count = 1);

      /**
       * This will return the value of a dynamic collective in
       * the form of a future. Applications can then use this 
       * future just like all other futures.
       * @param ctx enclosing task context
       * @param dc dynamic collective on which to get the result
       * @return future value that contains the result of the collective
       */
      Future get_dynamic_collective_result(Context ctx, DynamicCollective dc); 

      /**
       * Advance an existing dynamic collective to the next
       * phase.  It has the same semantics as the equivalent
       * call for phase barriers.
       * @param ctx enclosing task context
       * @param dc the dynamic collective to be advanced
       * @return an updated dynamic collective used for the next phase
       */
      DynamicCollective advance_dynamic_collective(Context ctx, 
                                                   DynamicCollective dc);
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
      // Frame Operations 
      //------------------------------------------------------------------------
      /**
       * Frames are a very simple way to control the number of 
       * outstanding operations in a task context. By default, mappers
       * have control over this by saying how many outstanding operations
       * each task context can have using the 'configure_context' mapper
       * call. However, in many cases, it is easier for custom mappers to
       * reason about how many iterations or some other application-specific
       * set of operations are in flight. To facilitate this, applications can 
       * create 'frames' of tasks. Using the 'configure_context' mapper
       * call, custom mappers can specify the maximum number of outstanding
       * frames that make up the operation window. It is best to place these
       * calls at the end of a frame of tasks.
       */
      void complete_frame(Context ctx);
    public:
      //------------------------------------------------------------------------
      // Must Parallelism 
      //------------------------------------------------------------------------
      /**
       * Launch a collection of operations that all must be guaranteed to 
       * execute in parallel.  This construct is necessary for ensuring the
       * correctness of tasks which use simultaneous coherence and perform
       * synchronization between different physical instances (e.g. using
       * phase barriers or reservations).  
       */
      FutureMap execute_must_epoch(Context ctx, 
                                   const MustEpochLauncher &launcher);
    public:
      //------------------------------------------------------------------------
      // Tunable Variables 
      //------------------------------------------------------------------------
      /**
       * Similar to Legion's ancestral predecessor Sequoia, Legion supports
       * tunable variables which are integers supplied by the mapper for 
       * individual task contexts.  The idea is that there are some parameters
       * which should be considered parameters determined by the underlying
       * hardware.  To make these parameters explicit, we express them as
       * tunables which are filled in at runtime by mapper objects.  Right
       * now all tunables are integers, however, if users develop compelling
       * cases for other types we would be interested in knowing about them.
       */
      int get_tunable_value(Context ctx, TunableID tid, 
                            MapperID mapper = 0, MappingTagID tag = 0);
    public:
      //------------------------------------------------------------------------
      // Timing Operations 
      //------------------------------------------------------------------------
      /**
       * Issue an operation into the stream to record the current time in
       * seconds. The resulting future should be interpreted as a 'double'
       * that represents the absolute time when this measurement was taken.
       * The operation can be given an optional future which will not be 
       * interpreted, but will be used as a precondition to ensure that the 
       * measurement will not be taken until the precondition is complete.
       */
      Future get_current_time(Context ctx, Future precondition = Future());

      /**
       * Issue an operation into the stream to record the current time in 
       * microseconds. The resulting future should be interpreted as a 
       * 'long long' with no fractional microseconds. The operation can be
       * givien an optional future precondition which will not be interpreted,
       * but ill be used as a precondition to ensure that the measurement
       * will not be taken until the precondition is complete.
       */
      Future get_current_time_in_microseconds(Context ctx, 
                                              Future precondition = Future());

      /**
       * Issue an operation into the stream to record the current time in 
       * nanoseconds. The resulting future should be interpreted as a 
       * 'long long' with no fractional nanoseconds. The operation can be
       * givien an optional future precondition which will not be interpreted,
       * but ill be used as a precondition to ensure that the measurement
       * will not be taken until the precondition is complete.
       */
      Future get_current_time_in_nanoseconds(Context ctx,
                                             Future precondition = Future());

      /**
       * Return the base time in nanoseconds on THIS node with which all 
       * other aboslute timings can be compared. This value will not change 
       * during the course of the lifetime of a Legion application and may 
       * therefore be safely cached.
       */
      static long long get_zero_time(void);
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
       * @param target processor if any, if none specified then
       *               the executing processor for the current
       *               context is used, if specified processor
       *               must be local to the address space
       * @return a pointer to the specified mapper object
       */
      Mapper* get_mapper(Context ctx, MapperID id, 
                         Processor target = Processor::NO_PROC);
      
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
      // MPI Interoperability 
      //------------------------------------------------------------------------
      /**
       * Return a reference to the mapping from MPI ranks to address spaces.
       * This method is only valid if the static initialization method
       * 'configure_MPI_interoperability' was called on all nodes before 
       * starting the runtime with the static 'start' method.
       * @return a const reference to the forward map
       */
      const std::map<int/*rank*/,AddressSpace>& find_forward_MPI_mapping(void);

      /**
       * Return a reference to the reverse mapping from address spaces to
       * MPI ranks. This method is only valid if the static initialization
       * method 'configure_MPI_interoperability' was called on all nodes
       * before starting the runtime with the static 'start' method.
       * @return a const reference to the reverse map
       */
      const std::map<AddressSpace,int/*rank*/>& find_reverse_MPI_mapping(void);
    public:
      //------------------------------------------------------------------------
      // Semantic Information 
      //------------------------------------------------------------------------
      /**
       * Attach semantic information to a logical task
       * @param handle task_id the ID of the task
       * @param tag the semantic tag
       * @param buffer pointer to a buffer
       * @param size size of the buffer to save
       * @param is_mutable can the tag value be changed later
       */
      void attach_semantic_information(TaskID task_id, SemanticTag tag,
                     const void *buffer, size_t size, bool is_mutable = false);

      /**
       * Attach semantic information to an index space
       * @param handle index space handle
       * @param tag semantic tag
       * @param buffer pointer to a buffer
       * @param size size of the buffer to save
       * @param is_mutable can the tag value be changed later
       */
      void attach_semantic_information(IndexSpace handle, SemanticTag tag,
                     const void *buffer, size_t size, bool is_mutable = false);

      /**
       * Attach semantic information to an index partition 
       * @param handle index partition handle
       * @param tag semantic tag
       * @param buffer pointer to a buffer
       * @param size size of the buffer to save
       * @param is_mutable can the tag value be changed later
       */
      void attach_semantic_information(IndexPartition handle, SemanticTag tag,
                     const void *buffer, size_t size, bool is_mutable = false);

      /**
       * Attach semantic information to a field space
       * @param handle field space handle
       * @param tag semantic tag
       * @param buffer pointer to a buffer
       * @param size size of the buffer to save
       * @param is_mutable can the tag value be changed later
       */
      void attach_semantic_information(FieldSpace handle, SemanticTag tag,
                     const void *buffer, size_t size, bool is_mutable = false);

      /**
       * Attach semantic information to a specific field 
       * @param handle field space handle
       * @param fid field ID
       * @param tag semantic tag
       * @param buffer pointer to a buffer
       * @param size size of the buffer to save
       * @param is_mutable can the tag value be changed later
       */
      void attach_semantic_information(FieldSpace handle, FieldID fid, 
                                       SemanticTag tag, const void *buffer, 
                                       size_t size, bool is_mutable = false);

      /**
       * Attach semantic information to a logical region 
       * @param handle logical region handle
       * @param tag semantic tag
       * @param buffer pointer to a buffer
       * @param size size of the buffer to save
       * @param is_mutable can the tag value be changed later
       */
      void attach_semantic_information(LogicalRegion handle, SemanticTag tag,
                     const void *buffer, size_t size, bool is_mutable = false);
      
      /**
       * Attach semantic information to a logical partition 
       * @param handle logical partition handle
       * @param tag semantic tag
       * @param buffer pointer to a buffer
       * @param size size of the buffer to save
       * @param is_mutable can the tag value be changed later
       */
      void attach_semantic_information(LogicalPartition handle, 
                                       SemanticTag tag, const void *buffer, 
                                       size_t size, bool is_mutable = false);

      /**
       * Attach a name to a task
       * @param task_id the ID of the task
       * @param name pointer to the name
       * @param is_mutable can the name be changed later
       */
      void attach_name(TaskID task_id, const char *name, 
                       bool is_mutable = false);

      /**
       * Attach a name to an index space
       * @param handle index space handle
       * @param name pointer to a name
       * @param is_mutable can the name be changed later
       */
      void attach_name(IndexSpace handle, const char *name,
                       bool is_mutable = false);

      /**
       * Attach a name to an index partition
       * @param handle index partition handle
       * @param name pointer to a name
       * @param is_mutable can the name be changed later
       */
      void attach_name(IndexPartition handle, const char *name,
                       bool is_mutable = false);

      /**
       * Attach a name to a field space
       * @param handle field space handle
       * @param name pointer to a name
       * @param is_mutable can the name be changed later
       */
      void attach_name(FieldSpace handle, const char *name,
                       bool is_mutable = false);

      /**
       * Attach a name to a specific field
       * @param handle field space handle
       * @param fid field ID
       * @param name pointer to a name
       * @param is_mutable can the name be changed later
       */
      void attach_name(FieldSpace handle, FieldID fid, 
                       const char *name, bool is_mutable = false);

      /**
       * Attach a name to a logical region
       * @param handle logical region handle
       * @param name pointer to a name
       * @param is_mutable can the name be changed later
       */
      void attach_name(LogicalRegion handle, const char *name,
                       bool is_mutable = false);

      /**
       * Attach a name to a logical partition
       * @param handle logical partition handle
       * @param name pointer to a name
       * @param is_mutable can the name be changed later
       */
      void attach_name(LogicalPartition handle, const char *name,
                       bool is_mutable = false);

      /**
       * Retrieve semantic information for a task
       * @param task_id the ID of the task
       * @param tag semantic tag
       * @param result pointer to assign to the semantic buffer
       * @param size where to write the size of the semantic buffer
       * @param can_fail query allowed to fail
       * @param wait_until_ready wait indefinitely for the tag
       * @return true if the query succeeds
       */
      bool retrieve_semantic_information(TaskID task_id, SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail = false,
                                         bool wait_until_ready = false);

      /**
       * Retrieve semantic information for an index space
       * @param handle index space handle
       * @param tag semantic tag
       * @param result pointer to assign to the semantic buffer
       * @param size where to write the size of the semantic buffer
       * @param can_fail query allowed to fail
       * @param wait_until_ready wait indefinitely for the tag
       * @return true if the query succeeds
       */
      bool retrieve_semantic_information(IndexSpace handle, SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail = false,
                                         bool wait_until_ready = false);

      /**
       * Retrieve semantic information for an index partition 
       * @param handle index partition handle
       * @param tag semantic tag
       * @param result pointer to assign to the semantic buffer
       * @param size where to write the size of the semantic buffer
       * @param can_fail query allowed to fail
       * @param wait_until_ready wait indefinitely for the tag
       * @return true if the query succeeds
       */
      bool retrieve_semantic_information(IndexPartition handle, SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail = false,
                                         bool wait_until_ready = false);

      /**
       * Retrieve semantic information for a field space
       * @param handle field space handle
       * @param tag semantic tag
       * @param result pointer to assign to the semantic buffer
       * @param size where to write the size of the semantic buffer
       * @param can_fail query allowed to fail
       * @param wait_until_ready wait indefinitely for the tag
       * @return true if the query succeeds
       */
      bool retrieve_semantic_information(FieldSpace handle, SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail = false,
                                         bool wait_until_ready = false);

      /**
       * Retrieve semantic information for a specific field 
       * @param handle field space handle
       * @param fid field ID
       * @param tag semantic tag
       * @param result pointer to assign to the semantic buffer
       * @param size where to write the size of the semantic buffer
       * @param can_fail query allowed to fail
       * @param wait_until_ready wait indefinitely for the tag
       * @return true if the query succeeds
       */
      bool retrieve_semantic_information(FieldSpace handle, FieldID fid, 
                                         SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail = false,
                                         bool wait_until_ready = false);

      /**
       * Retrieve semantic information for a logical region 
       * @param handle logical region handle
       * @param tag semantic tag
       * @param result pointer to assign to the semantic buffer
       * @param size where to write the size of the semantic buffer
       * @param can_fail query allowed to fail
       * @param wait_until_ready wait indefinitely for the tag
       * @return true if the query succeeds
       */
      bool retrieve_semantic_information(LogicalRegion handle, SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail = false,
                                         bool wait_until_ready = false);

      /**
       * Retrieve semantic information for a logical partition
       * @param handle logical partition handle
       * @param tag semantic tag
       * @param result pointer to assign to the semantic buffer
       * @param size where to write the size of the semantic buffer
       * @param can_fail query allowed to fail
       * @param wait_until_ready wait indefinitely for the tag
       * @return true if the query succeeds
       */
      bool retrieve_semantic_information(LogicalPartition handle, 
                                         SemanticTag tag,
                                         const void *&result, size_t &size,
                                         bool can_fail = false,
                                         bool wait_until_ready = false);

      /**
       * Retrieve the name of a task
       * @param task_id the ID of the task
       * @param result pointer to assign to the name
       */
      void retrieve_name(TaskID task_id, const char *&result);

      /**
       * Retrieve the name of an index space
       * @param handle index space handle
       * @param result pointer to assign to the name
       */
      void retrieve_name(IndexSpace handle, const char *&result);

      /**
       * Retrieve the name of an index partition
       * @param handle index partition handle
       * @param result pointer to assign to the name
       */
      void retrieve_name(IndexPartition handle, const char *&result);

      /**
       * Retrieve the name of a field space
       * @param handle field space handle
       * @param result pointer to assign to the name
       */
      void retrieve_name(FieldSpace handle, const char *&result);

      /**
       * Retrieve the name of a specific field
       * @param handle field space handle
       * @param fid field ID
       * @param result pointer to assign to the name
       */
      void retrieve_name(FieldSpace handle, FieldID fid, const char *&result);

      /**
       * Retrieve the name of a logical region
       * @param handle logical region handle
       * @param result pointer to assign to the name
       */
      void retrieve_name(LogicalRegion handle, const char *&result);

      /**
       * Retrieve the name of a logical partition
       * @param handle logical partition handle
       * @param result pointer to assign to the name
       */
      void retrieve_name(LogicalPartition handle, const char *&result);
    public:
      //------------------------------------------------------------------------
      // Registration Callback Operations
      // All of these calls must be made while in the registration
      // function called before start-up.  This function is specified
      // by calling the 'set_registration_callback' static method.
      //------------------------------------------------------------------------

      /**
       * Dynamically generate a unique Mapper ID for use across the machine
       * @return a Mapper ID that is globally unique across the machine
       */
      MapperID generate_dynamic_mapper_id(void);

      /**
       * Statically generate a unique Mapper ID for use across the machine.
       * This can only be called prior to the runtime starting. It must
       * be invoked symmetrically across all nodes in the machine prior
       * to starting the rutnime.
       * @return a MapperID that is globally unique across the machine
       */
      static MapperID generate_static_mapper_id(void);

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

      /**
       * Register a projection functor for handling projection
       * queries. The ProjectionID must be non-zero because 
       * zero is the identity projection. Unlike mappers which
       * require a separate instance per processor, only
       * one of these must be registered per projection ID.
       * @param pid the projection ID to use for the registration
       * @param functor the object to register for handle projections
       */
      void register_projection_functor(ProjectionID pid, 
                                       ProjectionFunctor *functor);
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
       * -hl:no_dyn   Disable dynamic disjointness tests when the runtime
       *              has been compiled with macro DYNAMIC_TESTS defined
       *              which enables dynamic disjointness testing.
       * -hl:epoch <int> Change the size of garbage collection epochs. The
       *              default value is 64. Increasing it adds latency to
       *              the garbage collection but makes it more efficient.
       *              Decreasing the value reduces latency, but adds
       *              inefficiency to the collection.
       * -hl:unsafe_launch Tell the runtime to skip any checks for 
       *              checking for deadlock between a parent task and
       *              the sub-operations that it is launching. Note
       *              that this is unsafe for a reason. The application
       *              can and will deadlock if any currently mapped
       *              regions conflict with those requested by a child
       *              task or other operation.
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
       * -hl:registration Record the mapping from low-level task IDs to
       *              task variant names for debugging low-level runtime
       *              error messages.
       * -------------
       *  Profiling
       * -------------
       * -hl:spy      Enable light-weight logging for Legion Spy which
       *              is valuable for understanding properties of an
       *              application such as the shapes of region trees
       *              and the kinds of tasks/operations that are created.
       *              Checking of the runtime with Legion Spy will still
       *              require the runtime to be compiled with -DLEGION_SPY.
       * -hl:prof <int> Specify the number of nodes on which to enable
       *              profiling information to be collected.  By default
       *              all nodes are disabled. Zero will disable all
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
       * running in background mode.  Otherwise it is illegal to 
       * invoke this method.
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
       * Configre the runtime for interoperability with MPI. This call
       * should be made once in each MPI process before invoking the 
       * 'start' function when running Legion within the same process 
       * as MPI. As a result of this call the 'find_forward_MPI_mapping' 
       * and 'find_reverse_MPI_mapping' methods on a runtime instance will 
       * return a map which associates an AddressSpace with each MPI rank.
       * @param rank the integer naming this MPI rank
       */
      static void configure_MPI_interoperability(int rank);

      /**
       * Create a handshake object for exchanging control between MPI
       * and Legion. We make this a static method so that it can be
       * created before the Legion runtime is initialized.
       * @param init_in_MPI who owns initial control of the handshake,
       *                    by default it is MPI
       * @param mpi_participants number of calls that need to be made to 
       *                    the handshake to pass control from MPI to Legion
       * @param legion_participants number of calls that need to be made to
       *                    the handshake to pass control from Legion to MPI
       */
      static MPILegionHandshake create_handshake(bool init_in_MPI = true,
                                                 int mpi_participants = 1,
                                                 int legion_participants = 1);

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
       * Register custom serialize/deserialize operation with the
       * runtime. This can be used for providing custom serialization
       * and deserialization method for fields that are not trivially
       * copied (e.g. byte-wise copies). The type being registered
       * must conform to the Realm definition of a CustomSerdez
       * object (see realm/custom_serdez.h).
       */
      template<typename SERDEZ>
      static void register_custom_serdez_op(CustomSerdezID serdez_id);

      /**
       * Return a pointer to the given custom serdez operation object.
       * @param serdez_id ID of the serdez operation to find
       * @return a pointer to the serdez operation object if it exists
       */
      static const SerdezOp* get_serdez_op(CustomSerdezID serdez_id);

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
                                         Runtime*)>
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
                                         Runtime*)>
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
      // Layout Registration Operations
      //------------------------------------------------------------------------
      /**
       * Register a new layout description with the runtime. The runtime will
       * return an ID that is a globally unique name for this set of 
       * constraints and can be used anywhere in the machine. Once this set 
       * of constraints is set, it cannot be changed.
       * @param registrar a layout description registrar 
       * @return a unique layout ID assigned to this set of constraints 
       */
      LayoutConstraintID register_layout(
                                    const LayoutConstraintRegistrar &registrar);

      /**
       * Release the set of constraints associated the given layout ID.
       * This promises that this set of constraints will never be used again.
       * @param layout_id the name for the set of constraints to release
       */
      void release_layout(LayoutConstraintID layout_id);

      /**
       * A static version of the method above to register layout
       * descriptions prior to the runtime starting. Attempting to
       * use this method after the runtime starts will result in a 
       * failure. All of the calls to this method must specifiy layout
       * descriptions that are not associated with a field space.
       * This call must be made symmetrically across all nodes.
       * @param registrar a layout description registrar
       * @param layout_id the ID to associate with the description
       * @return the layout id assigned to the set of constraints
       */
      static LayoutConstraintID preregister_layout(
                               const LayoutConstraintRegistrar &registrar,
                               LayoutConstraintID layout_id = AUTO_GENERATE_ID);

      /**
       * Get the field space for a specific layout description
       * @param layout_id the layout ID for which to obtain the field space
       * @return the field space handle for the layout description
       */
      FieldSpace get_layout_constraint_field_space(
                                  LayoutConstraintID layout_id);

      /**
       * Get the constraints for a specific layout description
       * @param layout_id the layout ID for which to obtain the constraints
       * @param layout_constraints a LayoutConstraintSet to populate
       */
      void get_layout_constraints(LayoutConstraintID layout_id,
                                  LayoutConstraintSet &layout_constraints);

      /**
       * Get the name associated with a particular layout description
       * @param layout_id the layout ID for which to obtain the name
       * @return a pointer to a string of the name of the layou description
       */
      const char* get_layout_constraints_name(LayoutConstraintID layout_id);
    public:
      //------------------------------------------------------------------------
      // Task Registration Operations
      //------------------------------------------------------------------------
      
      /**
       * Dynamically generate a unique Task ID for use across the machine
       * @return a Task ID that is globally unique across the machine
       */
      TaskID generate_dynamic_task_id(void);

      /**
       * Statically generate a unique Task ID for use across the machine.
       * This can only be called prior to the runtime starting. It must
       * be invoked symmetrically across all nodes in the machine prior
       * to starting the runtime.
       * @return a TaskID that is globally unique across the machine
       */
      static TaskID generate_static_task_id(void);

      /**
       * Dynamically register a new task variant with the runtime with
       * a non-void return type.
       * @param registrar the task variant registrar for describing the task
       * @return variant ID for the task
       */
      template<typename T,
        T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, Runtime*)>
      VariantID register_task_variant(const TaskVariantRegistrar &registrar);

      /**
       * Dynamically register a new task variant with the runtime with
       * a non-void return type and user data.
       * @param registrar the task variant registrar for describing the task
       * @param user_data the user data to associate with the task variant
       * @return variant ID for the task
       */
      template<typename T, typename UDT,
        T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, Runtime*, const UDT&)>
      VariantID register_task_variant(const TaskVariantRegistrar &registrar,
                                      const UDT &user_data);

      /**
       * Dynamically register a new task variant with the runtime with
       * a void return type.
       * @param registrar the task variant registrar for describing the task
       * @return variant ID for the task
       */
      template<
        void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                         Context, Runtime*)>
      VariantID register_task_variant(const TaskVariantRegistrar &registrar);

      /**
       * Dynamically register a new task variant with the runtime with
       * a void return type and user data.
       * @param registrar the task variant registrar for describing the task
       * @param user_data the user data to associate with the task variant
       * @return variant ID for the task
       */
      template<typename UDT,
        void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                         Context, Runtime*, const UDT&)>
      VariantID register_task_variant(const TaskVariantRegistrar &registrar,
                                      const UDT &user_data);

      /**
       * Dynamically register a new task variant with the runtime that
       * has already built in the necessary preamble/postamble (i.e.
       * calls to LegionTaskWrapper::legion_task_{pre,post}amble)
       * @param registrar the task variant registrar for describing the task
       * @param codedesc the code descriptor for the pre-wrapped task
       * @param user_data pointer to optional user data to associate with the
       * task variant
       * @param user_len size of optional user_data in bytes
       * @return variant ID for the task
       */
      VariantID register_task_variant(const TaskVariantRegistrar &registrar,
				      const CodeDescriptor &codedesc,
				      const void *user_data = NULL,
				      size_t user_len = 0);

      /**
       * Statically register a new task variant with the runtime with
       * a non-void return type prior to the runtime starting. This call
       * must be made on all nodes and it will fail if done after the
       * Runtime::start method has been invoked.
       * @param registrar the task variant registrar for describing the task
       * @param task_name an optional name to assign to the logical task
       * @return variant ID for the task
       */
      template<typename T,
        T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, Runtime*)>
      static VariantID preregister_task_variant(
                                    const TaskVariantRegistrar &registrar,
                                    const char *task_name = NULL);

      /**
       * Statically register a new task variant with the runtime with
       * a non-void return type and userd data prior to the runtime 
       * starting. This call must be made on all nodes and it will 
       * fail if done after the Runtime::start method has been invoked.
       * @param registrar the task variant registrar for describing the task
       * @param user_data the user data to associate with the task variant
       * @param task_name an optional name to assign to the logical task
       * @return variant ID for the task
       */
      template<typename T, typename UDT,
        T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, Runtime*, const UDT&)>
      static VariantID preregister_task_variant(
                      const TaskVariantRegistrar &registrar, 
                      const UDT &user_data, const char *task_name = NULL);
       
      /**
       * Statically register a new task variant with the runtime with
       * a void return type prior to the runtime starting. This call
       * must be made on all nodes and it will fail if done after the
       * Runtime::start method has been invoked.
       * @param registrar the task variant registrar for describing the task
       * @param an optional name to assign to the logical task
       * @return variant ID for the task
       */
      template<
        void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                         Context, Runtime*)>
      static VariantID preregister_task_variant(
                                    const TaskVariantRegistrar &registrar,
                                    const char *task_name = NULL);

      /**
       * Statically register a new task variant with the runtime with
       * a void return type and user data prior to the runtime starting. 
       * This call must be made on all nodes and it will fail if done 
       * after the Runtime::start method has been invoked.
       * @param registrar the task variant registrar for describing the task
       * @param user_data the user data to associate with the task variant
       * @param an optional name to assign to the logical task
       * @return variant ID for the task
       */
      template<typename UDT,
        void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                         Context, Runtime*, const UDT&)>
      static VariantID preregister_task_variant(
              const TaskVariantRegistrar &registrar, 
              const UDT &user_data, const char *task_name = NULL);
    public:
      //------------------------------------------------------------------------
      // Task Generator Registration Operations
      //------------------------------------------------------------------------
     
      /**
       * Dynamically generate a unique generator ID for use across the machine.
       * @return a Generator ID that is globally unique across the machine
       */
      GeneratorID generate_dynamic_generator_id(void);

      /**
       * Statically generate a unique Generator ID for use across the machine.
       * This can only be called prior to the runtime starting. It must be 
       * invoked symmetrically across all nodes in the machine prior to 
       * starting the runtime.
       * @return a Generator ID that is globally unique across the machine
       */
      static GeneratorID generate_static_generator_id(void);

      /**
       * Dynamically register a new task generator function with the runtime
       * @param registrar the task generator registrar
       */
      void register_task_generator(const TaskGeneratorRegistrar &registrar);

      /**
       * Statically register a new task generator function with the runtime.
       * This call must be made on all nodes and it will fail if done after
       * the runtime has been started.
       * @param registrar the task generator registrar
       */
      static void preregister_task_generator(
                                   const TaskGeneratorRegistrar &registrar);
                                   

      /**
       * Statically register a new task variant with the runtime that
       * has already built in the necessary preamble/postamble (i.e.
       * calls to LegionTaskWrapper::legion_task_{pre,post}amble).
       * This call must be made on all nodes and it will fail if done 
       * after the Runtime::start method has been invoked.
       * @param registrar the task variant registrar for describing the task
       * @param codedesc the code descriptor for the pre-wrapped task
       * @param user_data pointer to optional user data to associate with the
       * task variant
       * @param user_len size of optional user_data in bytes
       * @return variant ID for the task
       */
      static VariantID preregister_task_variant(
              const TaskVariantRegistrar &registrar,
	      const CodeDescriptor &codedesc,
	      const void *user_data = NULL,
	      size_t user_len = 0,
	      const char *task_name = NULL);

    public:
      // ------------------ Deprecated task registration -----------------------
      /**
       * @deprecated
       * Register a task with a template return type for the given
       * kind of processor.
       * @param id the ID to assign to the task
       * @param proc_kind the processor kind on which the task can run
       * @param single whether the task can be run as a single task
       * @param index whether the task can be run as an index space task
       * @param vid the variant ID to assign to the task
       * @param options the task configuration options
       * @param task_name string name for the task
       * @return the ID the task was assigned
       */
      template<typename T,
        T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, Runtime*)>
      static TaskID register_legion_task(TaskID id, Processor::Kind proc_kind,
                                         bool single, bool index, 
                                         VariantID vid = AUTO_GENERATE_ID,
                              TaskConfigOptions options = TaskConfigOptions(),
                                         const char *task_name = NULL);
      /**
       * @deprecated
       * Register a task with a void return type for the given
       * kind of processor.
       * @param id the ID to assign to the task 
       * @param proc_kind the processor kind on which the task can run
       * @param single whether the task can be run as a single task
       * @param index whether the task can be run as an index space task
       * @param vid the variant ID to assign to the task
       * @param options the task configuration options
       * @param task_name string name for the task
       * @return the ID the task was assigned
       */
      template<
        void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                         Context, Runtime*)>
      static TaskID register_legion_task(TaskID id, Processor::Kind proc_kind,
                                         bool single, bool index,
                                         VariantID vid = AUTO_GENERATE_ID,
                             TaskConfigOptions options = TaskConfigOptions(),
                                         const char *task_name = NULL);
      /**
       * @deprecated
       * Same as the register_legion_task above, but allow for users to
       * pass some static data that will be passed as an argument to
       * all invocations of the function.
       * @param id the ID at which to assign the task
       * @param proc_kind the processor kind on which the task can run
       * @param single whether the task can be run as a single task
       * @param index whether the task can be run as an index space task
       * @param user_data user data type to pass to all invocations of the task
       * @param vid the variant ID to assign to the task
       * @param options the task configuration options
       * @param task_name string name for the task
       * @return the ID the task was assigned
       */
      template<typename T, typename UDT,
        T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, Runtime*, const UDT&)>
      static TaskID register_legion_task(TaskID id, Processor::Kind proc_kind,
                                         bool single, bool index,
                                         const UDT &user_data,
                                         VariantID vid = AUTO_GENERATE_ID,
                              TaskConfigOptions options = TaskConfigOptions(),
                                         const char *task_name = NULL);
      /**
       * @deprecated
       * Same as the register_legion_task above, but allow for users to
       * pass some static data that will be passed as an argument to
       * all invocations of the function.
       * @param id the ID at which to assign the task
       * @param proc_kind the processor kind on which the task can run
       * @param single whether the task can be run as a single task
       * @param index whether the task can be run as an index space task
       * @param user_data user data type to pass to all invocations of the task
       * @param vid the variant ID to assign to the task
       * @param options the task configuration options
       * @param task_name string name for the task
       * @return the ID the task was assigned
       */
      template<typename UDT,
        void (*TASK_PTR)(const Task*,const std::vector<PhysicalRegion>&,
                         Context, Runtime*, const UDT&)>
      static TaskID register_legion_task(TaskID id, Processor::Kind proc_kind,
                                         bool single, bool index,
                                         const UDT &user_data,
                                         VariantID vid = AUTO_GENERATE_ID,
                              TaskConfigOptions options = TaskConfigOptions(),
                                         const char *task_name = NULL);
    public:
      /**
       * Provide a mechanism for finding the high-level runtime
       * pointer for a processor wrapper tasks that are starting
       * a new application level task.
       * @param processor the task will run on
       * @return the high-level runtime pointer for the specified processor
       */
      static Runtime* get_runtime(Processor p);
    private:
      friend class FieldAllocator;
      FieldID allocate_field(Context ctx, FieldSpace space, 
                             size_t field_size, FieldID fid, bool local,
                             CustomSerdezID serdez_id);
      void free_field(Context ctx, FieldSpace space, FieldID fid);
      void allocate_fields(Context ctx, FieldSpace space, 
                           const std::vector<size_t> &sizes,
                           std::vector<FieldID> &resulting_fields, 
                           bool local, CustomSerdezID serdez_id);
      void free_fields(Context ctx, FieldSpace space, 
                       const std::set<FieldID> &to_free);
    private:
      // Methods for the wrapper functions to get information from the runtime
      friend class LegionTaskWrapper;
      friend class LegionSerialization;
      const std::vector<PhysicalRegion>& begin_task(Context ctx);
      void end_task(Context ctx, const void *result, size_t result_size,
                    bool owned = false);
      Future from_value(const void *value, size_t value_size, bool owned);
    private:
      static ProjectionID register_region_projection_function(
                                    ProjectionID handle, void *func_ptr);
      static ProjectionID register_partition_projection_function(
                                    ProjectionID handle, void *func_ptr);
    private:
      VariantID register_variant(const TaskVariantRegistrar &registrar,bool ret,
                                 const void *user_data, size_t user_data_size,
                                 CodeDescriptor *realm);
      static VariantID preregister_variant(const TaskVariantRegistrar &reg,
                                 const void *user_data, size_t user_data_size,
                                 CodeDescriptor *realm,
                                 bool has_return, const char *task_name,
                                 bool check_task_id = true);
    private:
      static ReductionOpTable& get_reduction_table(void);
      static SerdezOpTable& get_serdez_table(void);
      static SerdezRedopTable& get_serdez_redop_table(void);
    private:
      friend class Mapper;
      Internal *runtime;
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

  }; // namespace HighLevel
}; // namespace LegionRuntime

#include "legion.inl"

#endif // __LEGION_RUNTIME_H__

// EOF

