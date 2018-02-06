/* Copyright 2018 Stanford University, NVIDIA Corporation
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

// decide whether we want C and/or C++ bindings (default matches host language)
//
// each set of bindings has its own include-once ifdef armor, allowing the
//  second set of bindings to be loaded even if the first already has been
#if !defined(LEGION_ENABLE_C_BINDINGS) && !defined(LEGION_DISABLE_C_BINDINGS)
  #ifndef __cplusplus
    #define LEGION_ENABLE_C_BINDINGS
  #endif
#endif
#if !defined(LEGION_ENABLE_CXX_BINDINGS) && !defined(LEGION_DISABLE_CXX_BINDINGS)
  #ifdef __cplusplus
    #define LEGION_ENABLE_CXX_BINDINGS
  #endif
#endif

#ifdef LEGION_ENABLE_C_BINDINGS
#include "legion/legion_c.h"
#endif

#ifdef LEGION_ENABLE_CXX_BINDINGS
#ifndef __LEGION_RUNTIME_H__
#define __LEGION_RUNTIME_H__

/**
 * \mainpage Legion Runtime Documentation
 *
 * This is the main page of the Legion Runtime documentation.
 *
 * @see Legion::Runtime
 */

/**
 * \file legion.h
 * Legion C++ API
 */

#include "legion/legion_types.h"
#include "legion/legion_domain.h"
#include "legion/legion_constraint.h"

// temporary helper macro to turn link errors into runtime errors
#define UNIMPLEMENTED_METHOD(retval) do { assert(0); return retval; } while(0)

// There will be a new implementation of this for control replication
#define LEGION_PRINT_ONCE(runtime, ctx, file, fmt, ...)   \
  fprintf(file, fmt, ##__VA_ARGS__);            

/**
 * \namespace Legion
 * Namespace for all Legion runtime objects
 */
namespace Legion {

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
      IndexSpace(IndexSpaceID id, IndexTreeID tid, TypeTag tag);
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
      inline TypeTag get_type_tag(void) const { return type_tag; }
      inline int get_dim(void) const;
    protected:
      friend std::ostream& operator<<(std::ostream& os, 
                                      const IndexSpace& is);
      IndexSpaceID id;
      IndexTreeID tid;
      TypeTag type_tag;
    };

    /**
     * \class IndexSpaceT
     * A templated index space that captures the dimension
     * and coordinate type of an index space as template
     * parameters for enhanced type checking and efficiency.
     */
    template<int DIM, typename COORD_T = coord_t>
    class IndexSpaceT : public IndexSpace {
    protected:
      // Only the runtime should be allowed to make these
      FRIEND_ALL_RUNTIME_CLASSES
      IndexSpaceT(IndexSpaceID id, IndexTreeID tid);
    public:
      IndexSpaceT(void);
      IndexSpaceT(const IndexSpaceT &rhs);  
      explicit IndexSpaceT(const IndexSpace &rhs);
    public:
      inline IndexSpaceT& operator=(const IndexSpace &rhs);
      inline IndexSpaceT& operator=(const IndexSpaceT &rhs);
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
      IndexPartition(IndexPartitionID id, IndexTreeID tid, TypeTag tag);
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
      inline TypeTag get_type_tag(void) const { return type_tag; }
      inline int get_dim(void) const;
    protected:
      friend std::ostream& operator<<(std::ostream& os, 
                                      const IndexPartition& ip);
      IndexPartitionID id;
      IndexTreeID tid;
      TypeTag type_tag;
    };

    /**
     * \class IndexPartitionT
     * A templated index partition that captures the dimension
     * and coordinate type of an index partition as template
     * parameters for enhanced type checking and efficiency
     */
    template<int DIM, typename COORD_T = coord_t>
    class IndexPartitionT : public IndexPartition {
    protected:
      // Only the runtime should be allowed to make these
      FRIEND_ALL_RUNTIME_CLASSES
      IndexPartitionT(IndexPartitionID id, IndexTreeID tid);
    public:
      IndexPartitionT(void);
      IndexPartitionT(const IndexPartitionT &rhs);
      explicit IndexPartitionT(const IndexPartition &rhs);
    public:
      inline IndexPartitionT& operator=(const IndexPartition &rhs);
      inline IndexPartitionT& operator=(const IndexPartitionT &rhs);
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
      inline TypeTag get_type_tag(void) const 
        { return index_space.get_type_tag(); }
      inline int get_dim(void) const { return index_space.get_dim(); }
    protected:
      friend std::ostream& operator<<(std::ostream& os, 
                                      const LogicalRegion& lr);
      // These are private so the user can't just arbitrarily change them
      RegionTreeID tree_id;
      IndexSpace index_space;
      FieldSpace field_space;
    };

    /**
     * \class LogicalRegionT
     * A templated logical region that captures the dimension
     * and coordinate type of a logical region as template
     * parameters for enhanced type checking and efficiency.
     */
    template<int DIM, typename COORD_T = coord_t>
    class LogicalRegionT : public LogicalRegion {
    protected:
      // Only the runtime should be allowed to make these
      FRIEND_ALL_RUNTIME_CLASSES
      LogicalRegionT(RegionTreeID tid, IndexSpace index, FieldSpace field);
    public:
      LogicalRegionT(void);
      LogicalRegionT(const LogicalRegionT &rhs);
      explicit LogicalRegionT(const LogicalRegion &rhs);
    public:
      inline LogicalRegionT& operator=(const LogicalRegion &rhs);
      inline LogicalRegionT& operator=(const LogicalRegionT &rhs);
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
      inline TypeTag get_type_tag(void) const 
        { return index_partition.get_type_tag(); }
      inline int get_dim(void) const { return index_partition.get_dim(); }
    protected:
      friend std::ostream& operator<<(std::ostream& os, 
                                      const LogicalPartition& lp);
      // These are private so the user can't just arbitrary change them
      RegionTreeID tree_id;
      IndexPartition index_partition;
      FieldSpace field_space;
    };

    /**
     * \class LogicalPartitionT
     * A templated logical partition that captures the dimension
     * and coordinate type of an logical partition as template
     * parameters for enhanced type checking and efficiency
     */
    template<int DIM, typename COORD_T = coord_t>
    class LogicalPartitionT : public LogicalPartition {
    protected:
      // Only the runtime should be allowed to make these
      FRIEND_ALL_RUNTIME_CLASSES
      LogicalPartitionT(RegionTreeID tid, IndexPartition pid, FieldSpace field);
    public:
      LogicalPartitionT(void);
      LogicalPartitionT(const LogicalPartitionT &rhs);
      explicit LogicalPartitionT(const LogicalPartition &rhs);
    public:
      inline LogicalPartitionT& operator=(const LogicalPartition &rhs);
      inline LogicalPartitionT& operator=(const LogicalPartitionT &rhs);
    };

    //==========================================================================
    //                       Data Allocation Classes
    //==========================================================================

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    /**
     * \class IndexIterator
     * This is a helper class for iterating over the points within
     * an index space or the index space of a given logical region.
     * It should never be copied and will assert fail if a copy is
     * made of it.
     */
    class LEGION_DEPRECATED("Use DomainPointIterator instead") IndexIterator {
    public:
      IndexIterator(void);
      IndexIterator(const Domain &dom, ptr_t start = ptr_t());
      IndexIterator(Runtime *rt, Context ctx, 
                    IndexSpace space, ptr_t start = ptr_t());
      IndexIterator(Runtime *rt, Context ctx, 
                    LogicalRegion lr, ptr_t start = ptr_t());
      IndexIterator(Runtime *rt,
                    IndexSpace space, ptr_t start = ptr_t());
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
      Realm::IndexSpaceIterator<1,coord_t> is_iterator;
      Realm::PointInRectIterator<1,coord_t> rect_iterator;
    };

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
    class LEGION_DEPRECATED("Dynamic IndexAllocators are no longer supported")
      IndexAllocator : public Unserializable<IndexAllocator> {
    public:
      IndexAllocator(void);
      IndexAllocator(const IndexAllocator &allocator);
      ~IndexAllocator(void);
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      // Only the Runtime should be able to make these
      IndexAllocator(IndexSpace space, IndexIterator iterator);
    public:
      IndexAllocator& operator=(const IndexAllocator &allocator);
      inline bool operator<(const IndexAllocator &rhs) const;
      inline bool operator==(const IndexAllocator &rhs) const;
    public:
      /**
       * @param num_elements number of elements to allocate
       * @return pointer to the first element in the allocated block
       */
      LEGION_DEPRECATED("Dynamic allocation is no longer supported")
      ptr_t alloc(unsigned num_elements = 1);
      /**
       * @param ptr pointer to the first element to free
       * @param num_elements number of elements to be freed
       */
      LEGION_DEPRECATED("Dynamic allocation is no longer supported")
      void free(ptr_t ptr, unsigned num_elements = 1);
      /**
       * @return the index space associated with this allocator
       */
      inline IndexSpace get_index_space(void) const { return index_space; }
    private:
      IndexSpace index_space;
      IndexIterator iterator;
    };
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif

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
    class FieldAllocator : public Unserializable<FieldAllocator> {
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
    class TaskArgument : public Unserializable<TaskArgument> {
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
    class ArgumentMap : public Unserializable<ArgumentMap> {
    public:
      ArgumentMap(void);
      ArgumentMap(const FutureMap &rhs);
      ArgumentMap(const ArgumentMap &rhs);
      ~ArgumentMap(void);
    public:
      ArgumentMap& operator=(const FutureMap &rhs);
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
      Internal::ArgumentMapImpl *impl;
    private:
      explicit ArgumentMap(Internal::ArgumentMapImpl *i);
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
    class Predicate : public Unserializable<Predicate> {
    public:
      static const Predicate TRUE_PRED;
      static const Predicate FALSE_PRED;
    public:
      Predicate(void);
      Predicate(const Predicate &p);
      explicit Predicate(bool value);
      ~Predicate(void);
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      Internal::PredicateImpl *impl;
      // Only the runtime should be allowed to make these
      explicit Predicate(Internal::PredicateImpl *impl);
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
     * @see IndexTaskLauncher
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
    protected:
      // Only the runtime is allowed to make non-empty grants
      FRIEND_ALL_RUNTIME_CLASSES
      explicit Grant(Internal::GrantImpl *impl);
    public:
      bool operator==(const Grant &g) const
        { return impl == g.impl; }
      bool operator<(const Grant &g) const
        { return impl < g.impl; }
      Grant& operator=(const Grant &g);
    protected:
      Internal::GrantImpl *impl;
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
     * @see IndexTaskLauncher
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
      PhaseBarrier(Internal::ApBarrier b);
    public:
      bool operator<(const PhaseBarrier &rhs) const;
      bool operator==(const PhaseBarrier &rhs) const;
      bool operator!=(const PhaseBarrier &rhs) const;
    public:
      void arrive(unsigned count = 1);
      void wait(void);
      void alter_arrival_count(int delta);
      Realm::Barrier get_barrier(void) const { return phase_barrier; }
      bool exists(void) const;
    protected:
      Internal::ApBarrier phase_barrier;
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
      DynamicCollective(Internal::ApBarrier b, ReductionOpID redop);
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
      inline bool is_verified(void) const 
        { return (flags & VERIFIED_FLAG); }
      inline bool is_no_access(void) const 
        { return (flags & NO_ACCESS_FLAG); }
      inline bool is_restricted(void) const 
        { return (flags & RESTRICTED_FLAG); }
      inline bool must_premap(void) const
        { return (flags & MUST_PREMAP_FLAG); }
    public:
#ifdef PRIVILEGE_CHECKS
      unsigned get_accessor_privilege(void) const;
#endif
      bool has_field_privilege(FieldID fid) const;
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
      ProjectionType handle_type; /**< region or partition requirement*/
      ProjectionID projection; /**< projection function for index space tasks*/
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
    class Future : public Unserializable<Future> {
    public:
      Future(void);
      Future(const Future &f);
      ~Future(void);
    private:
      Internal::FutureImpl *impl;
    protected:
      // Only the runtime should be allowed to make these
      FRIEND_ALL_RUNTIME_CLASSES
      explicit Future(Internal::FutureImpl *impl,
                      bool need_reference = true);
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
       * @param silence_warnings silence any warnings for this blocking call
       * @return the value of the future cast as the template type
       */
      template<typename T> 
        inline T get_result(bool silence_warnings = false) const;
      /**
       * Block until the future completes.
       * @param silence_warnings silence any warnings for this blocking call
       */
      void get_void_result(bool silence_warnings = false) const;
      /**
       * Check to see if the future is empty.  The
       * user can specify whether to block and wait
       * for the future to complete first before
       * returning.  If the non-blocking version
       * of the call will return true, until
       * the future actually completes.
       * @param block indicate whether to block for the result
       * @param silence_warnings silence any warnings for this blocking call
       */
      bool is_empty(bool block = false, bool silence_warnings = false) const;
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
       * @param silence_warnings silence any warnings for this blocking call
       */
      template<typename T> 
        inline const T& get_reference(bool silence_warnings = false);
      /**
       * Return an untyped pointer to the 
       * future result.  WARNING: this
       * method is unsafe for the same reasons
       * as get_reference.  It also will not 
       * deserialize anything serialized with a 
       * legion_serialize method.
       * @param silence_warnings silence any warnings for this blocking call
       */
      inline const void* get_untyped_pointer(bool silence_warnings = false);
      /**
       * Return the number of bytes contained in the future.
       */
      size_t get_untyped_size(void);
    public:
      // These methods provide partial support the C++ future interface
      template<typename T>
      inline T get(void);

      inline bool valid(void) const;

      inline void wait(void) const;
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
      void* get_untyped_result(bool silence_warnings) const; 
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
    class FutureMap : public Unserializable<FutureMap> {
    public:
      FutureMap(void);
      FutureMap(const FutureMap &map);
      ~FutureMap(void);
    private:
      Internal::FutureMapImpl *impl;
    protected:
      // Only the runtime should be allowed to make these
      FRIEND_ALL_RUNTIME_CLASSES
      explicit FutureMap(Internal::FutureMapImpl *impl,
                         bool need_reference = true);
    public:
      inline bool operator==(const FutureMap &f) const
        { return impl == f.impl; }
      inline bool operator<(const FutureMap &f) const
        { return impl < f.impl; }
      inline Future operator[](const DomainPoint &point)
        { return get_future(point); }
      FutureMap& operator=(const FutureMap &f);
    public:
      /**
       * Block until we can return the result for the
       * task executing for the given domain point.
       * @param point the point task to wait for
       * @param silence_warnings silence any warnings for this blocking call
       * @return the return value of the task
       */
      template<typename T>
        inline T get_result(const DomainPoint &point,
                            bool silence_warnings = false);
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
       * @param silience_warnings silence any warnings for this blocking call
       */
      void get_void_result(const DomainPoint &point,
                           bool silence_warnings = false);
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
       * @param silence_warnings silience warnings for this blocking call
       */
      void wait_all_results(bool silence_warnings = false); 
    }; 


    //==========================================================================
    //                    Operation Launcher Classes
    //==========================================================================

    /**
     * \struct StaticDependence
     * This is a helper class for specifying static dependences 
     * between operations launched by an application. Operations
     * can optionally specify these dependences to aid in reducing
     * runtime overhead. These static dependences need only
     * be specified for dependences based on region requirements.
     */
    struct StaticDependence : public Unserializable<StaticDependence> {
    public:
      StaticDependence(void);
      StaticDependence(unsigned previous_offset,
                       unsigned previous_req_index,
                       unsigned current_req_index,
                       DependenceType dtype,
                       bool validates = false);
    public:
      inline void add_field(FieldID fid);
    public:
      // The relative offset from this operation to 
      // previous operation in the stream of operations
      // (e.g. 1 is the operation launched immediately before)
      unsigned                      previous_offset;
      // Region requirement of the previous operation for the dependence
      unsigned                      previous_req_index;
      // Region requirement of the current operation for the dependence
      unsigned                      current_req_index;
      // The type of the dependence
      DependenceType                dependence_type;
      // Whether this requirement validates the previous writer
      bool                          validates;
      // Fields that have the dependence
      std::set<FieldID>             dependent_fields;
    };

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
      inline void add_wait_handshake(MPILegionHandshake handshake);
      inline void add_arrival_handshake(MPILegionHandshake handshake);
    public:
      inline void set_predicate_false_future(Future f);
      inline void set_predicate_false_result(TaskArgument arg);
    public:
      inline void set_independent_requirements(bool independent);
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
    public:
      // Inform the runtime about any static dependences
      // These will be ignored outside of static traces
      const std::vector<StaticDependence> *static_dependences;
    public:
      // Users can tell the runtime this task is eligible
      // for inlining by the mapper. This will invoke the 
      // select_task_options call inline as part of the launch
      // logic for this task to allow the mapper to decide
      // whether to inline the task or not. Note that if the
      // mapper pre-empts during execution then resuming it
      // may take a long time if another long running task
      // gets scheduled on the processor that launched this task.
      bool                               enable_inlining;
    public:
      // Users can inform the runtime that all region requirements
      // are independent of each other in this task. Independent
      // means that either field sets are independent or region
      // requirements are disjoint based on the region tree.
      bool                               independent_requirements;
    public:
      bool                               silence_warnings;
    };

    /**
     * \struct IndexTaskLauncher
     * Index launchers are objects that describe the launch
     * of an index space of tasks to the runtime.  They can
     * be re-used and safely modified between calls to 
     * index space launches.
     * @see Runtime
     */
    struct IndexTaskLauncher {
    public:
      IndexTaskLauncher(void);
      IndexTaskLauncher(Processor::TaskFuncID tid,
                        Domain domain,
                        TaskArgument global_arg,
                        ArgumentMap map,
                        Predicate pred = Predicate::TRUE_PRED,
                        bool must = false,
                        MapperID id = 0,
                        MappingTagID tag = 0);
      IndexTaskLauncher(Processor::TaskFuncID tid,
                        IndexSpace launch_space,
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
      inline void add_wait_handshake(MPILegionHandshake handshake);
      inline void add_arrival_handshake(MPILegionHandshake handshake);
    public:
      inline void set_predicate_false_future(Future f);
      inline void set_predicate_false_result(TaskArgument arg);
    public:
      inline void set_independent_requirements(bool independent);
    public:
      Processor::TaskFuncID              task_id;
      Domain                             launch_domain;
      IndexSpace                         launch_space;
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
    public:
      // Inform the runtime about any static dependences
      // These will be ignored outside of static traces
      const std::vector<StaticDependence> *static_dependences;
    public:
      // Users can tell the runtime this task is eligible
      // for inlining by the mapper. This will invoke the 
      // select_task_options call inline as part of the launch
      // logic for this task to allow the mapper to decide
      // whether to inline the task or not. Note that if the
      // mapper pre-empts during execution then resuming it
      // may take a long time if another long running task
      // gets scheduled on the processor that launched this task.
      bool                               enable_inlining;
    public:
      // Users can inform the runtime that all region requirements
      // are independent of each other in this task. Independent
      // means that either field sets are independent or region
      // requirements are disjoint based on the region tree.
      bool                               independent_requirements;
    public:
      bool                               silence_warnings;
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
                     MappingTagID tag = 0,
                     LayoutConstraintID layout_id = 0);
    public:
      inline void add_field(FieldID fid, bool inst = true);
    public:
      inline void add_grant(Grant g);
      inline void add_wait_barrier(PhaseBarrier bar);
      inline void add_arrival_barrier(PhaseBarrier bar);
      inline void add_wait_handshake(MPILegionHandshake handshake);
      inline void add_arrival_handshake(MPILegionHandshake handshake);
    public:
      RegionRequirement               requirement;
      std::vector<Grant>              grants;
      std::vector<PhaseBarrier>       wait_barriers;
      std::vector<PhaseBarrier>       arrive_barriers;
      MapperID                        map_id;
      MappingTagID                    tag;
    public:
      LayoutConstraintID              layout_constraint_id;
    public:
      // Inform the runtime about any static dependences
      // These will be ignored outside of static traces
      const std::vector<StaticDependence> *static_dependences;
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
      // Specify gather/scatter region requirements (must have exactly 1 field)
      inline void add_gather_field(const RegionRequirement &gather_req,
                                   FieldID gather_fid, bool inst = true);
      inline void add_scatter_field(const RegionRequirement &scatter_req,
                                    FieldID scatter_fid, bool inst = true);
    public:
      inline void add_grant(Grant g);
      inline void add_wait_barrier(PhaseBarrier bar);
      inline void add_arrival_barrier(PhaseBarrier bar);
      inline void add_wait_handshake(MPILegionHandshake handshake);
      inline void add_arrival_handshake(MPILegionHandshake handshake); 
    public:
      std::vector<RegionRequirement>  src_requirements;
      std::vector<RegionRequirement>  dst_requirements;
      std::vector<RegionRequirement>  gather_requirements;
      std::vector<RegionRequirement>  scatter_requirements;
      std::vector<Grant>              grants;
      std::vector<PhaseBarrier>       wait_barriers;
      std::vector<PhaseBarrier>       arrive_barriers;
      Predicate                       predicate;
      MapperID                        map_id;
      MappingTagID                    tag;
      DomainPoint                     point;
    public:
      // Inform the runtime about any static dependences
      // These will be ignored outside of static traces
      const std::vector<StaticDependence> *static_dependences;
    public:
      bool                            silence_warnings;
    };

    /**
     * \struct IndexCopyLauncher
     * An index copy launcher is the same as a normal copy launcher
     * but it supports the ability to launch multiple copies all 
     * over a single index space domain. This means that region 
     * requirements can use projection functions the same as with
     * index task launches.
     * @see CopyLauncher
     * @see Runtime
     */
    struct IndexCopyLauncher {
    public:
      IndexCopyLauncher(void);
      IndexCopyLauncher(Domain domain, Predicate pred = Predicate::TRUE_PRED,
                        MapperID id = 0, MappingTagID tag = 0);
      IndexCopyLauncher(IndexSpace space, Predicate pred = Predicate::TRUE_PRED,
                        MapperID id = 0, MappingTagID tag = 0);
    public:
      inline unsigned add_copy_requirements(const RegionRequirement &src,
					    const RegionRequirement &dst);
      inline void add_src_field(unsigned idx, FieldID fid, bool inst = true);
      inline void add_dst_field(unsigned idx, FieldID fid, bool inst = true);
    public:
      // Specify gather/scatter region requirements (must have exactly 1 field)
      inline void add_gather_field(const RegionRequirement &gather_req,
                                   FieldID gather_fid, bool inst = true);
      inline void add_scatter_field(const RegionRequirement &scatter_req,
                                    FieldID scatter_fid, bool inst = true);
    public:
      inline void add_grant(Grant g);
      inline void add_wait_barrier(PhaseBarrier bar);
      inline void add_arrival_barrier(PhaseBarrier bar);
      inline void add_wait_handshake(MPILegionHandshake handshake);
      inline void add_arrival_handshake(MPILegionHandshake handshake);
    public:
      std::vector<RegionRequirement>  src_requirements;
      std::vector<RegionRequirement>  dst_requirements;
      std::vector<RegionRequirement>  gather_requirements;
      std::vector<RegionRequirement>  scatter_requirements;
      std::vector<Grant>              grants;
      std::vector<PhaseBarrier>       wait_barriers;
      std::vector<PhaseBarrier>       arrive_barriers;
      Domain                          launch_domain;
      IndexSpace                      launch_space;
      Predicate                       predicate;
      MapperID                        map_id;
      MappingTagID                    tag;
    public:
      // Inform the runtime about any static dependences
      // These will be ignored outside of static traces
      const std::vector<StaticDependence> *static_dependences;
    public:
      bool                            silence_warnings;
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
                   TaskArgument arg, Predicate pred = Predicate::TRUE_PRED,
                   MapperID id = 0, MappingTagID tag = 0);
      FillLauncher(LogicalRegion handle, LogicalRegion parent,
                   Future f, Predicate pred = Predicate::TRUE_PRED,
                   MapperID id = 0, MappingTagID tag = 0);
    public:
      inline void set_argument(TaskArgument arg);
      inline void set_future(Future f);
      inline void add_field(FieldID fid);
      inline void add_grant(Grant g);
      inline void add_wait_barrier(PhaseBarrier bar);
      inline void add_arrival_barrier(PhaseBarrier bar);
      inline void add_wait_handshake(MPILegionHandshake handshake);
      inline void add_arrival_handshake(MPILegionHandshake handshake);
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
      MapperID                        map_id;
      MappingTagID                    tag;
      DomainPoint                     point;
    public:
      // Inform the runtime about any static dependences
      // These will be ignored outside of static traces
      const std::vector<StaticDependence> *static_dependences;
    public:
      bool                            silence_warnings;
    };

    /**
     * \struct IndexFillLauncher
     * Index fill launchers are objects that are used to describe
     * a fill over a particular domain. They can be used with 
     * projeciton functions to describe a fill over an arbitrary
     * set of logical regions.
     * @see FillLauncher
     * @see Runtime
     */
    struct IndexFillLauncher {
    public:
      IndexFillLauncher(void);
      // Region projection
      IndexFillLauncher(Domain domain, LogicalRegion handle, 
                        LogicalRegion parent, TaskArgument arg,
                        ProjectionID projection = 0,
                        Predicate pred = Predicate::TRUE_PRED,
                        MapperID id = 0, MappingTagID tag = 0);
      IndexFillLauncher(Domain domain, LogicalRegion handle, 
                        LogicalRegion parent, Future f,
                        ProjectionID projection = 0,
                        Predicate pred = Predicate::TRUE_PRED,
                        MapperID id = 0, MappingTagID tag = 0);
      IndexFillLauncher(IndexSpace space, LogicalRegion handle, 
                        LogicalRegion parent, TaskArgument arg,
                        ProjectionID projection = 0,
                        Predicate pred = Predicate::TRUE_PRED,
                        MapperID id = 0, MappingTagID tag = 0);
      IndexFillLauncher(IndexSpace space, LogicalRegion handle, 
                        LogicalRegion parent, Future f,
                        ProjectionID projection = 0,
                        Predicate pred = Predicate::TRUE_PRED,
                        MapperID id = 0, MappingTagID tag = 0);
      // Partition projection
      IndexFillLauncher(Domain domain, LogicalPartition handle, 
                        LogicalRegion parent, TaskArgument arg,
                        ProjectionID projection = 0,
                        Predicate pred = Predicate::TRUE_PRED,
                        MapperID id = 0, MappingTagID tag = 0);
      IndexFillLauncher(Domain domain, LogicalPartition handle, 
                        LogicalRegion parent, Future f,
                        ProjectionID projection = 0,
                        Predicate pred = Predicate::TRUE_PRED,
                        MapperID id = 0, MappingTagID tag = 0);
      IndexFillLauncher(IndexSpace space, LogicalPartition handle, 
                        LogicalRegion parent, TaskArgument arg,
                        ProjectionID projection = 0,
                        Predicate pred = Predicate::TRUE_PRED,
                        MapperID id = 0, MappingTagID tag = 0);
      IndexFillLauncher(IndexSpace space, LogicalPartition handle, 
                        LogicalRegion parent, Future f,
                        ProjectionID projection = 0,
                        Predicate pred = Predicate::TRUE_PRED,
                        MapperID id = 0, MappingTagID tag = 0);
    public:
      inline void set_argument(TaskArgument arg);
      inline void set_future(Future f);
      inline void add_field(FieldID fid);
      inline void add_grant(Grant g);
      inline void add_wait_barrier(PhaseBarrier bar);
      inline void add_arrival_barrier(PhaseBarrier bar);
      inline void add_wait_handshake(MPILegionHandshake handshake);
      inline void add_arrival_handshake(MPILegionHandshake handshake);
    public:
      Domain                          launch_domain;
      IndexSpace                      launch_space;
      LogicalRegion                   region;
      LogicalPartition                partition;
      LogicalRegion                   parent;
      ProjectionID                    projection;
      TaskArgument                    argument;
      Future                          future;
      Predicate                       predicate;
      std::set<FieldID>               fields;
      std::vector<Grant>              grants;
      std::vector<PhaseBarrier>       wait_barriers;
      std::vector<PhaseBarrier>       arrive_barriers;
      MapperID                        map_id;
      MappingTagID                    tag;
    public:
      // Inform the runtime about any static dependences
      // These will be ignored outside of static traces
      const std::vector<StaticDependence> *static_dependences;
    public:
      bool                            silence_warnings;
    };

    /**
     * \struct AttachLauncher
     * Attach launchers are used for attaching existing physical resources
     * outside of a Legion application to a specific logical region.
     * This can include attaching files or arrays from inter-operating
     * programs. We provide a generic attach launcher than can handle
     * all kinds of attachments. Each attach launcher should be used
     * for attaching only one kind of resource.
     * @see Runtime
     */
    struct AttachLauncher {
    public:
      AttachLauncher(ExternalResource resource, 
                     LogicalRegion handle, LogicalRegion parent);
    public:
      inline void attach_file(const char *file_name,
                              const std::vector<FieldID> &fields,
                              LegionFileMode mode);
      inline void attach_hdf5(const char *file_name,
                              const std::map<FieldID,const char*> &field_map,
                              LegionFileMode mode);
      // Helper methods for AOS and SOA arrays, but it is totally 
      // acceptable to fill in the layout constraint set manually
      inline void attach_array_aos(void *base, bool column_major,
                                   const std::vector<FieldID> &fields,
                                   Memory mem, size_t alignment = 16);
      inline void attach_array_soa(void *base, bool column_major,
                                   const std::vector<FieldID> &fields,
                                   Memory mem, size_t alignment = 16);
    public:
      ExternalResource                              resource;
      LogicalRegion                                 handle;
      LogicalRegion                                 parent;
    public:
      // Data for files
      const char                                    *file_name;
      LegionFileMode                                mode;
      std::vector<FieldID>                          file_fields; // normal files
      std::map<FieldID,/*file name*/const char*>    field_files; // hdf5 files
    public:
      // Data for external instances
      LayoutConstraintSet                           constraints;
      std::set<FieldID>                             privilege_fields;
    public:
      // Inform the runtime about any static dependences
      // These will be ignored outside of static traces
      const std::vector<StaticDependence>           *static_dependences;
    };

    /**
     * \struct PredicateLauncher
     * Predicate launchers are used for merging several predicates
     * into a new predicate either by an 'AND' or an 'OR' operation.
     * @see Runtime
     */
    struct PredicateLauncher {
    public:
      PredicateLauncher(bool and_op = true);
    public:
      inline void add_predicate(const Predicate &pred); 
    public:
      bool                            and_op; // if not 'and' then 'or'
      std::vector<Predicate>          predicates;
    };

    /**
     * \struct TimingLauncher
     * Timing launchers are used for issuing a timing measurement.
     * @see Runtime
     */
    struct TimingLauncher {
    public:
      TimingLauncher(TimingMeasurement measurement);
    public:
      inline void add_precondition(const Future &f);
    public:
      TimingMeasurement               measurement;
      std::set<Future>                preconditions;
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
     * the output type for variants created by generator tasks.
     */
    struct TaskVariantRegistrar {
    public:
      TaskVariantRegistrar(void);
      TaskVariantRegistrar(TaskID task_id, bool global = true,
                           const char *variant_name = NULL);
      TaskVariantRegistrar(TaskID task_id, const char *variant_name,
			   bool global = true);
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
      inline void set_replicable(bool is_replicable = true);
    public: // Generator Task IDs
      inline void add_generator_task(TaskID tid);
    public:
      TaskID                            task_id;
      bool                              global_registration;
      const char*                       task_variant_name;
    public: // constraints
      ExecutionConstraintSet            execution_constraints; 
      TaskLayoutConstraintSet           layout_constraints;
    public:
      // TaskIDs for which this variant can serve as a generator
      std::set<TaskID>                  generator_tasks;
    public: // properties
      bool                              leaf_variant;
      bool                              inner_variant;
      bool                              idempotent_variant;
      bool                              replicable_variant;
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
    class PhysicalRegion : public Unserializable<PhysicalRegion> {
    public:
      PhysicalRegion(void);
      PhysicalRegion(const PhysicalRegion &rhs);
      ~PhysicalRegion(void);
    private:
      Internal::PhysicalRegionImpl *impl;
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      explicit PhysicalRegion(Internal::PhysicalRegionImpl *impl);
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
      bool is_mapped(void) const;
      /**
       * For physical regions returned as the result of an
       * inline mapping, this call will block until the physical
       * instance has a valid copy of the data. You can silence
       * warnings about this blocking call with the 
       * 'silence_warnings' parameter.
       */
      void wait_until_valid(bool silence_warnings = false);
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
       * @deprecated
       * Return a generic accessor for the entire physical region.
       * This method is now deprecated. Please use the 'get_field_accessor'
       * method instead. You can silence warnings about this blocking
       * call with the 'silence_warnings' parameter.
       */
      LEGION_DEPRECATED("All accessors in the LegionRuntime::Accessor "
                        "namespace are now deprecated. FieldAccessor "
                        "from the Legion namespace should be used now.")
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic> 
          get_accessor(bool silence_warnings = false) const;
      /**
       * @deprecated
       * You should be able to create accessors by passing this
       * object directly to the constructor of an accessor
       * Return a field accessor for a specific field within the region.
       * You can silence warnings regarding this blocking call with
       * the 'silence_warnings' parameter.
       */
      LEGION_DEPRECATED("All accessors in the LegionRuntime::Accessor "
                        "namespace are now deprecated. FieldAccessor "
                        "from the Legion namespace should be used now.")
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic> 
          get_field_accessor(FieldID field, 
                             bool silence_warnings = false) const;
      /**
       * Return the memories where the underlying physical instances locate.
       */
      void get_memories(std::set<Memory>& memories) const;
      /**
       * Return a list of fields that the physical region contains.
       */
      void get_fields(std::vector<FieldID>& fields) const;
    public:
      template<int DIM, typename COORD_T>
      DomainT<DIM,COORD_T> get_bounds(void) const;
      // We'll also allow this to implicitly cast to a realm index space
      // so that users can easily iterate over the points
      template<int DIM, typename COORD_T>
      operator DomainT<DIM,COORD_T>(void) const;
      // They can implicitly cast to a rectangle if there is no
      // sparsity map, runtime will check for this
      template<int DIM, typename COORD_T>
      operator Rect<DIM,COORD_T>(void) const;
    protected:
      // These methods can only be accessed by the FieldAccessor class
      template<PrivilegeMode, typename, int, typename, typename, bool>
      friend class FieldAccessor;
      template<typename, int, typename, typename>
      friend class UnsafeFieldAccessor;
      Realm::RegionInstance get_instance_info(PrivilegeMode mode, 
                                              FieldID fid, size_t field_size,
                                              void *realm_is, TypeTag type_tag,
                                              bool silence_warnings,
                                              bool generic_accessor,
                                              bool check_field_size,
                                              ReductionOpID redop = 0) const;
      void fail_bounds_check(DomainPoint p, FieldID fid,
                             PrivilegeMode mode) const;
      void fail_bounds_check(Domain d, FieldID fid,
                             PrivilegeMode mode) const;
    protected:
      void get_bounds(void *realm_is, TypeTag type_tag) const;
    };

    /**
     * \class FieldAccessor
     * A field accessor is a class used to get access to the data 
     * inside of a PhysicalRegion object for a specific field. The
     * default version of this class is empty, but the following
     * specializations of this class with different privilege modes
     * will provide different methods specific to that privilege type
     *
     * READ_ONLY
     *  - FT read(const Point<N,T>&) const
     *  - const FT* ptr(const Point<N,T>&) const (Affine Accessor only)
     *  - const FT* ptr(const Rect<N,T>&) const (Affine Accessor only)
     *  - const FT& operator[](const Point<N,T>&) const (Affine Accessor only)
     *
     * READ_WRITE
     *  - FT read(const Point<N,T>&) const
     *  - void write(const Point<N,T>&, FT val) const
     *  - FT* ptr(const Point<N,T>&) const (Affine Accessor only)
     *  - FT* ptr(const Rect<N,T>&) const (Affine Accessor only)
     *  - FT& operator[](const Point<N,T>&) const (Affine Accessor only)
     *  - template<typename REDOP> void reduce(const Point<N,T>&, REDOP::RHS);
     *    (Affine Accessor only)
     *
     *  WRITE_DISCARD
     *  - void write(const Point<N,T>&, FT val) const
     *  - FT* ptr(const Point<N,T>&) const (Affine Accessor only)
     *  - FT* ptr(const Rect<N,T>&) const (Affine Accessor only)
     *  - FT& operator[](const Point<N,T>&) const (Affine Accessor only)
     *
     * REDUCE
     *  - template<typename REDOP> void reduce(const Point<N,T>&, REDOP::RHS)
     *    (Affine Accessor only)
     */
    template<PrivilegeMode MODE, typename FT, int N, typename COORD_T = coord_t,
             typename A = Realm::GenericAccessor<FT,N,COORD_T>,
#ifdef BOUNDS_CHECKS
             bool CHECK_BOUNDS = true>
#else
             bool CHECK_BOUNDS = false>
#endif
    class FieldAccessor {
    public:
      FieldAccessor(void) { }
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    // The actual field size in case it is different from the 
                    // one being used in FT and we still want to check it
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false) { }
      // For Realm::AffineAccessor specializations there are additional
      // methods for creating accessors with limited bounding boxes and
      // affine transformations for using alternative coordinates spaces
      // Specify a specific bounds rectangle to use for the accessor
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const Rect<N,COORD_T> bounds,
                    // The actual field size in case it is different from the 
                    // one being used in FT and we still want to check it
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false) { }
      // Specify a specific Affine transform to use for interpreting points
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,COORD_T> transform,
                    // The actual field size in case it is different from the 
                    // one being used in FT and we still want to check it
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false) { }
      // Specify both a transform and a bounds to use
      template<int M>
      FieldAccessor(const PhysicalRegion &region, FieldID fid,
                    const AffineTransform<M,N,COORD_T> transform,
                    const Rect<N,COORD_T> bounds,
                    // The actual field size in case it is different from the 
                    // one being used in FT and we still want to check it
                    size_t actual_field_size = sizeof(FT),
#ifdef DEBUG_LEGION
                    bool check_field_size = true,
#else
                    bool check_field_size = false,
#endif
                    bool silence_warnings = false) { }
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
                      PhysicalRegion physical_region = PhysicalRegion(),
                      Predicate pred = Predicate::TRUE_PRED,
                      MapperID id = 0, MappingTagID tag = 0);
    public:
      inline void add_field(FieldID f);
      inline void add_grant(Grant g);
      inline void add_wait_barrier(PhaseBarrier pb);
      inline void add_arrival_barrier(PhaseBarrier pb);
      inline void add_wait_handshake(MPILegionHandshake handshake);
      inline void add_arrival_handshake(MPILegionHandshake handshake);
    public:
      LogicalRegion                   logical_region;
      LogicalRegion                   parent_region;
      std::set<FieldID>               fields;
    public:
      // This field is now optional
      PhysicalRegion                  physical_region;
    public:
      std::vector<Grant>              grants;
      std::vector<PhaseBarrier>       wait_barriers;
      std::vector<PhaseBarrier>       arrive_barriers;
      Predicate                       predicate;
      MapperID                        map_id;
      MappingTagID                    tag;
    public:
      // Inform the runtime about any static dependences
      // These will be ignored outside of static traces
      const std::vector<StaticDependence> *static_dependences;
    public:
      bool                            silence_warnings;
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
                      PhysicalRegion physical_region = PhysicalRegion(),
                      Predicate pred = Predicate::TRUE_PRED,
                      MapperID id = 0, MappingTagID tag = 0);
    public:
      inline void add_field(FieldID f);
      inline void add_grant(Grant g);
      inline void add_wait_barrier(PhaseBarrier pb);
      inline void add_arrival_barrier(PhaseBarrier pb);
      inline void add_wait_handshake(MPILegionHandshake handshake);
      inline void add_arrival_handshake(MPILegionHandshake handshake);
    public:
      LogicalRegion                   logical_region;
      LogicalRegion                   parent_region;
      std::set<FieldID>               fields;
    public:
      // This field is now optional
      PhysicalRegion                  physical_region;
    public:
      std::vector<Grant>              grants;
      std::vector<PhaseBarrier>       wait_barriers;
      std::vector<PhaseBarrier>       arrive_barriers;
      Predicate                       predicate;
      MapperID                        map_id;
      MappingTagID                    tag;
    public:
      // Inform the runtime about any static dependences
      // These will be ignored outside of static traces
      const std::vector<StaticDependence> *static_dependences;
    public:
      bool                            silence_warnings;
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
      inline void add_index_task(const IndexTaskLauncher &launcher);
    public:
      MapperID                       map_id;
      MappingTagID                   mapping_tag;
      std::vector<TaskLauncher>      single_tasks;
      std::vector<IndexTaskLauncher> index_tasks;
    public:
      bool                           silence_warnings;
    };

    //==========================================================================
    //                     MPI Interoperability Classes
    //==========================================================================

    class MPILegionHandshake : public Unserializable<MPILegionHandshake> {
    public:
      MPILegionHandshake(void);
      MPILegionHandshake(const MPILegionHandshake &rhs);
      ~MPILegionHandshake(void);
    private:
      Internal::MPILegionHandshakeImpl *impl;
    protected:
      // Only the runtime should be able to make these
      FRIEND_ALL_RUNTIME_CLASSES
      explicit MPILegionHandshake(Internal::MPILegionHandshakeImpl *impl);
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
      void mpi_handoff_to_legion(void) const;
      /**
       * A blocking call that will cause this participant to wait
       * for all Legion participants to hand over control to MPI.
       */
      void mpi_wait_on_legion(void) const;
    public:
      /**
       * A non-blocking call to signal to MPI that this participant
       * is ready to pass control to MPI.
       */
      void legion_handoff_to_mpi(void) const;
      /**
       * A blocking call that will cause this participant to wait
       * for all MPI participants to hand over control to Legion.
       */
      void legion_wait_on_mpi(void) const;
    public:
      /*
       * For asynchronous Legion execution, you can use these
       * methods to get a phase barrier associated with the 
       * handshake object instead of blocking on the legion side
       */
      /**
       * Get the Legion phase barrier associated with waiting on the handshake 
       */
      PhaseBarrier get_legion_wait_phase_barrier(void) const;
      /**
       * Get the Legion phase barrier associated with arriving on the handshake
       */
      PhaseBarrier get_legion_arrive_phase_barrier(void) const;
      /**
       * Advance the handshake associated with the Legion side
       */
      void advance_legion_handshake(void) const;
    };

    //==========================================================================
    //                           Operation Classes
    //==========================================================================

    /**
     * \class Mappable
     * The mappable class provides a base class for all 
     * the different types which can be passed to represent 
     * an operation to a mapping call.
     */
    class Mappable {
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      Mappable(void);
    public:
      // Return a globally unique ID for this operation
      virtual UniqueID get_unique_id(void) const = 0;
      // Return the number of operations that came before
      // this operation in the same context (close operations
      // return number of previous close operations)
      virtual unsigned get_context_index(void) const = 0;
      // Return the depth of this operation in the task tree
      virtual int get_depth(void) const = 0;
    public:
      enum MappableType {
        TASK_MAPPABLE,
        COPY_MAPPABLE,
        INLINE_MAPPABLE,
        ACQUIRE_MAPPABLE,
        RELEASE_MAPPABLE,
        CLOSE_MAPPABLE,
        FILL_MAPPABLE,
        PARTITION_MAPPABLE,
      };
      virtual MappableType get_mappable_type(void) const = 0;
      virtual const Task* as_task(void) const = 0;
      virtual const Copy* as_copy(void) const = 0;
      virtual const InlineMapping* as_inline(void) const = 0;
      virtual const Acquire* as_acquire(void) const = 0;
      virtual const Release* as_release(void) const = 0;
      virtual const Close* as_close(void) const = 0;
      virtual const Fill* as_fill(void) const = 0;
      virtual const Partition* as_partition(void) const = 0;
    public:
      MapperID                                  map_id;
      MappingTagID                              tag;
    public:
      // Mapper annotated data 
      void*                                     mapper_data;
      size_t                                    mapper_data_size;
    };

    /**
     * \class Task
     * This class contains all the information from a task
     * launch for either an individual or an index space
     * task. It also provides information about the current
     * state of the task from the runtime perspective so 
     * that mappers can make informed decisions.
     */
    class Task : public Mappable {
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      Task(void);
    public:
      virtual const char* get_task_name(void) const = 0;
    public:
      virtual MappableType get_mappable_type(void) const 
        { return TASK_MAPPABLE; }
      virtual const Task* as_task(void) const { return this; }
      virtual const Copy* as_copy(void) const { return NULL; }
      virtual const InlineMapping* as_inline(void) const { return NULL; }
      virtual const Acquire* as_acquire(void) const { return NULL; }
      virtual const Release* as_release(void) const { return NULL; }
      virtual const Close* as_close(void) const { return NULL; }
      virtual const Fill* as_fill(void) const { return NULL; }
      virtual const Partition* as_partition(void) const { return NULL; }
    public:
      // Task argument information
      Processor::TaskFuncID task_id; 
      std::vector<IndexSpaceRequirement>  indexes;
      std::vector<RegionRequirement>      regions;
      std::vector<Future>                 futures;
      std::vector<Grant>                  grants;
      std::vector<PhaseBarrier>           wait_barriers;
      std::vector<PhaseBarrier>           arrive_barriers;
      void*                               args;                         
      size_t                              arglen;
    public:
      // Index task argument information
      bool                                is_index_space;
      bool                                must_epoch_task; 
      Domain                              index_domain;
      DomainPoint                         index_point;
      void*                               local_args;
      size_t                              local_arglen;
    public:
      // Meta data information from the runtime
      Processor                           orig_proc;
      Processor                           current_proc;
      Processor                           target_proc;
      unsigned                            steal_count;
      bool                                stealable;
      bool                                speculated;
    public:
      // Parent task (only guaranteed to be good for one recursion)
      const Task*                         parent_task;
    };

    /**
     * \class Copy
     * This class contains all the information about an
     * explicit copy region-to-region copy operation.
     */
    class Copy : public Mappable {
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      Copy(void);
    public:
      virtual MappableType get_mappable_type(void) const 
        { return COPY_MAPPABLE; }
      virtual const Task* as_task(void) const { return NULL; }
      virtual const Copy* as_copy(void) const { return this; }
      virtual const InlineMapping* as_inline(void) const { return NULL; }
      virtual const Acquire* as_acquire(void) const { return NULL; }
      virtual const Release* as_release(void) const { return NULL; }
      virtual const Close* as_close(void) const { return NULL; }
      virtual const Fill* as_fill(void) const { return NULL; }
      virtual const Partition* as_partition(void) const { return NULL; }
    public:
      // Copy Launcher arguments
      std::vector<RegionRequirement>    src_requirements;
      std::vector<RegionRequirement>    dst_requirements;
      std::vector<Grant>                grants;
      std::vector<PhaseBarrier>         wait_barriers;
      std::vector<PhaseBarrier>         arrive_barriers;
    public:
      // Index copy argument information
      bool                              is_index_space;
      Domain                            index_domain;
      DomainPoint                       index_point;
    public:
      // Parent task for the copy operation
      const Task*                       parent_task;
    };

    /**
     * \class InlineMapping
     * This class contains all the information about an
     * inline mapping operation from its launcher
     */
    class InlineMapping : public Mappable {
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      InlineMapping(void);
    public:
      virtual MappableType get_mappable_type(void) const 
        { return INLINE_MAPPABLE; }
      virtual const Task* as_task(void) const { return NULL; }
      virtual const Copy* as_copy(void) const { return NULL; }
      virtual const InlineMapping* as_inline(void) const { return this; }
      virtual const Acquire* as_acquire(void) const { return NULL; }
      virtual const Release* as_release(void) const { return NULL; }
      virtual const Close* as_close(void) const { return NULL; }
      virtual const Fill* as_fill(void) const { return NULL; }
      virtual const Partition* as_partition(void) const { return NULL; }
    public:
      // Inline Launcher arguments
      RegionRequirement                 requirement;
      std::vector<Grant>                grants;
      std::vector<PhaseBarrier>         wait_barriers;
      std::vector<PhaseBarrier>         arrive_barriers;
      LayoutConstraintID                layout_constraint_id; 
    public:
      // Parent task for the inline operation
      const Task*                       parent_task;
    };

    /**
     * \class Acquire
     * This class contains all the information about an
     * acquire operation from the original launcher.
     */
    class Acquire : public Mappable {
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      Acquire(void);
    public:
      virtual MappableType get_mappable_type(void) const 
        { return ACQUIRE_MAPPABLE; }
      virtual const Task* as_task(void) const { return NULL; }
      virtual const Copy* as_copy(void) const { return NULL; }
      virtual const InlineMapping* as_inline(void) const { return NULL; }
      virtual const Acquire* as_acquire(void) const { return this; }
      virtual const Release* as_release(void) const { return NULL; }
      virtual const Close* as_close(void) const { return NULL; }
      virtual const Fill* as_fill(void) const { return NULL; }
      virtual const Partition* as_partition(void) const { return NULL; }
    public:
      // Acquire Launcher arguments
      LogicalRegion                     logical_region;
      LogicalRegion                     parent_region;
      std::set<FieldID>                 fields;
      std::vector<Grant>                grants;
      std::vector<PhaseBarrier>         wait_barriers;
      std::vector<PhaseBarrier>         arrive_barriers;
    public:
      // Parent task for the acquire operation
      const Task*                       parent_task;
    };

    /**
     * \class Release
     * This class contains all the information about a
     * release operation from the original launcher.
     */
    class Release : public Mappable {
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      Release(void);
    public:
      virtual MappableType get_mappable_type(void) const 
        { return RELEASE_MAPPABLE; }
      virtual const Task* as_task(void) const { return NULL; }
      virtual const Copy* as_copy(void) const { return NULL; }
      virtual const InlineMapping* as_inline(void) const { return NULL; }
      virtual const Acquire* as_acquire(void) const { return NULL; }
      virtual const Release* as_release(void) const { return this; }
      virtual const Close* as_close(void) const { return NULL; }
      virtual const Fill* as_fill(void) const { return NULL; }
      virtual const Partition* as_partition(void) const { return NULL; }
    public:
      // Release Launcher arguments
      LogicalRegion                     logical_region;
      LogicalRegion                     parent_region;
      std::set<FieldID>                 fields;
      std::vector<Grant>                grants;
      std::vector<PhaseBarrier>         wait_barriers;
      std::vector<PhaseBarrier>         arrive_barriers;
    public:
      // Parent task for the release operation
      const Task*                       parent_task;
    };

    /**
     * \class Close
     * This class represents a close operation that has
     * been requested by the runtime. The region requirement
     * for this operation is synthesized by the runtime
     * but will name the logical region and fields being
     * closed.  The privileges and coherence will always
     * be READ_WRITE EXCLUSIVE.
     */
    class Close : public Mappable {
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      Close(void);
    public:
      virtual MappableType get_mappable_type(void) const 
        { return CLOSE_MAPPABLE; }
      virtual const Task* as_task(void) const { return NULL; }
      virtual const Copy* as_copy(void) const { return NULL; }
      virtual const InlineMapping* as_inline(void) const { return NULL; }
      virtual const Acquire* as_acquire(void) const { return NULL; }
      virtual const Release* as_release(void) const { return NULL; }
      virtual const Close* as_close(void) const { return this; }
      virtual const Fill* as_fill(void) const { return NULL; }
      virtual const Partition* as_partition(void) const { return NULL; }
    public:
      // Synthesized region requirement
      RegionRequirement                 requirement;
    public:
      // Parent task for the inline operation
      const Task*                       parent_task;
    };

    /**
     * \class Fill
     * This class represents a fill operation for the
     * original launcher. See the fill launcher for
     * more information.
     */
    class Fill : public Mappable {
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      Fill(void);
    public:
      virtual MappableType get_mappable_type(void) const 
        { return FILL_MAPPABLE; }
      virtual const Task* as_task(void) const { return NULL; }
      virtual const Copy* as_copy(void) const { return NULL; }
      virtual const InlineMapping* as_inline(void) const { return NULL; }
      virtual const Acquire* as_acquire(void) const { return NULL; }
      virtual const Release* as_release(void) const { return NULL; }
      virtual const Close* as_close(void) const { return NULL; }
      virtual const Fill* as_fill(void) const { return this; }
      virtual const Partition* as_partition(void) const { return NULL; }
    public:
      // Synthesized region requirement
      RegionRequirement               requirement;
      std::vector<Grant>              grants;
      std::vector<PhaseBarrier>       wait_barriers;
      std::vector<PhaseBarrier>       arrive_barriers;
    public:
      // Index fill argument information
      bool                              is_index_space;
      Domain                            index_domain;
      DomainPoint                       index_point;
    public:
      // Parent task for the fill operation
      const Task*                       parent_task;
    };

    /**
     * \class Partition
     * This class represents a dependent partition 
     * operation that is being performed by the
     * runtime. These will be crated by calls to
     * the runtime such as 'create_partition_by_field'.
     */
    class Partition : public Mappable {
    protected:
      FRIEND_ALL_RUNTIME_CLASSES
      Partition(void);
    public:
      virtual MappableType get_mappable_type(void) const 
        { return PARTITION_MAPPABLE; }
      virtual const Task* as_task(void) const { return NULL; }
      virtual const Copy* as_copy(void) const { return NULL; }
      virtual const InlineMapping* as_inline(void) const { return NULL; }
      virtual const Acquire* as_acquire(void) const { return NULL; }
      virtual const Release* as_release(void) const { return NULL; }
      virtual const Close* as_close(void) const { return NULL; }
      virtual const Fill* as_fill(void) const { return NULL; }
      virtual const Partition* as_partition(void) const { return this; }
    public:
      enum PartitionKind {
        BY_FIELD, // create partition by field
        BY_IMAGE, // create partition by image
        BY_IMAGE_RANGE, // create partition by image range
        BY_PREIMAGE, // create partition by preimage
        BY_PREIMAGE_RANGE,  // create partition by preimage range
        BY_ASSOCIATION,  // create partition by association
      };
      virtual PartitionKind get_partition_kind(void) const = 0;
    public:
      // Synthesized region requirement
      RegionRequirement                   requirement;
    public:
      // Index partition argument information
      bool                                is_index_space;
      Domain                              index_domain;
      DomainPoint                         index_point;
    public:
      // Parent task for the partition operation
      const Task*                         parent_task;
    };

    //==========================================================================
    //                           Runtime Classes
    //==========================================================================

    /**
     * @deprecated 
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
      ProjectionFunctor(void);
      ProjectionFunctor(Runtime *rt);
      virtual ~ProjectionFunctor(void);
    public:
      /**
       * This is the more general implementation of projection
       * functions that work for all kinds of operations. 
       * Implementations can switch on the mappable type to
       * figure out the kind of the operation that is requesting
       * the projection. The default implementation of this method
       * calls the deprecated version of this method for tasks and
       * fails for all other kinds of operations. Note that this
       * method is not passed a context, because it should only
       * be invoking context free runtime methods.
       * @param mappable the operation requesting the projection
       * @param index the index of the region requirement being projected
       * @param upper_bound the upper bound logical region
       * @param point the point being projected
       * @return logical region result
       */
      virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                    LogicalRegion upper_bound,
                                    const DomainPoint &point);
      /**
       * Same method as above, but with a partition as an upper bound
       * @param mappable the operation requesting the projection
       * @param index the index of the region requirement being projected
       * @param upper_bound the upper bound logical partition
       * @param point the point being projected
       * @return logical region result
       */
      virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                    LogicalPartition upper_bound,
                                    const DomainPoint &point);

      /**
       * @deprecated
       * Compute the projection for a logical region projection
       * requirement down to a specific logical region.
       * @param ctx the context for this projection
       * @param task the task for the requested projection
       * @param index which region requirement we are projecting
       * @param upper_bound the upper bound logical region
       * @param point the point of the task in the index space
       * @return logical region to be used by the child task
       */
      LEGION_DEPRECATED("The interface for projection functors has been "
                        "updated. Please use the new 'project' methods.")
      virtual LogicalRegion project(Context ctx, Task *task,
                                    unsigned index,
                                    LogicalRegion upper_bound,
                                    const DomainPoint &point);
      /**
       * @deprecated
       * Compute the projection for a logical partition projection
       * requirement down to a specific logical region.
       * @param ctx the context for this projection
       * @param task the task for the requested projection
       * @param index which region requirement we are projecting
       * @param upper_bound the upper bound logical partition
       * @param point the point of the task in the index space
       * @return logical region to be used by the child task
       */
      LEGION_DEPRECATED("The interface for projection functors has been "
                        "updated. Please use the new 'project' methods.")
      virtual LogicalRegion project(Context ctx, Task *task, 
                                    unsigned index,
                                    LogicalPartition upper_bound,
                                    const DomainPoint &point);
      
      /**
       * Indicate whether calls to this projection functor
       * must be serialized or can be performed in parallel.
       * Usually they must be exclusive if this functor contains
       * state for memoizing results.
       */
      virtual bool is_exclusive(void) const { return false; }

      /**
       * Specify the depth which this projection function goes
       * for all the points in an index space launch from 
       * the upper bound node in the region tree. Depth is
       * defined as the number of levels of the region tree
       * crossed from the upper bound logical region or partition.
       * So depth 0 for a REG_PROJECTION means the same region
       * while depth 0 for a PART_PROJECTION means a subregion
       * in the immediate partition. Depth 0 is the default
       * for the identity projection function.
       */
      virtual unsigned get_depth(void) const = 0;
    private:
      friend class Internal::Runtime;
      // For pre-registered projection functors the runtime will
      // use this to initialize the runtime pointer
      inline void set_runtime(Runtime *rt) { runtime = rt; }
    protected:
      Runtime *runtime;
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
     *
     * A note on context free functions: context free functions 
     * have equivalent functionality to their non-context-free 
     * couterparts. However, context free functions can be 
     * safely used in a leaf task while any runtime function 
     * that requires a context cannot be used in a leaf task. 
     * If your task variant only uses context free functions
     * as part of its implementation then it is safe for you 
     * to annotate it as a leaf task variant.
     */
    class Runtime {
    protected:
      // The Runtime bootstraps itself and should
      // never need to be explicitly created.
      friend class Internal::Runtime;
      friend class Future;
      Runtime(Internal::Runtime *rt);
    public:
      //------------------------------------------------------------------------
      // Index Space Operations
      //------------------------------------------------------------------------
      ///@{
      /**
       * Create a new top-level index space based on the given domain bounds
       * @param ctx the enclosing task context
       * @param bounds the bounds for the new index space
       * @return the handle for the new index space
       */
      IndexSpace create_index_space(Context ctx, Domain bounds);
      // Template version
      template<int DIM, typename COORD_T>
      IndexSpaceT<DIM,COORD_T> create_index_space(Context ctx,
                                                  Rect<DIM,COORD_T> bounds);
      ///@}
      ///@{
      /**
       * Create a new top-level index space from a vector of points
       * @param ctx the enclosing task context
       * @param points a vector of points to have in the index space
       * @return the handle for the new index space
       */
      IndexSpace create_index_space(Context ctx, 
                                    const std::vector<DomainPoint> &points);
      // Template version
      template<int DIM, typename COORD_T>
      IndexSpaceT<DIM,COORD_T> create_index_space(Context ctx,
                      const std::vector<Point<DIM,COORD_T> > &points);
      ///@}
      ///@{
      /**
       * Create a new top-level index space from a vector of rectangles
       * @param ctx the enclosing task context
       * @param rects a vector of rectangles to have in the index space
       * @return the handle for the new index space
       */
      IndexSpace create_index_space(Context ctx,
                                    const std::vector<Domain> &rects);
      // Template version
      template<int DIM, typename COORD_T>
      IndexSpaceT<DIM,COORD_T> create_index_space(Context ctx,
                      const std::vector<Rect<DIM,COORD_T> > &rects);
      ///@}
      ///@{
      /**
       * Create a new top-level index space by unioning together 
       * several existing index spaces
       * @param ctx the enclosing task context
       * @param spaces the index spaces to union together
       * @return the handle for the new index space
       */
      IndexSpace union_index_spaces(Context ctx, 
                                    const std::vector<IndexSpace> &spaces);
      // Template version
      template<int DIM, typename COORD_T>
      IndexSpaceT<DIM,COORD_T> union_index_spaces(Context ctx,
                     const std::vector<IndexSpaceT<DIM,COORD_T> > &spaces);
      ///@}
      ///@{
      /**
       * Create a new top-level index space by intersecting 
       * several existing index spaces
       * @param ctx the enclosing task context
       * @param spaces the index spaces to intersect
       * @return the handle for the new index space
       */
      IndexSpace intersect_index_spaces(Context ctx,
                                        const std::vector<IndexSpace> &spaces);
      // Template version
      template<int DIM, typename COORD_T>
      IndexSpaceT<DIM,COORD_T> intersect_index_spaces(Context ctx,
                         const std::vector<IndexSpaceT<DIM,COORD_T> > &spaces);
      ///@}
      ///@{
      /**
       * Create a new top-level index space by taking the
       * set difference of two different index spaces
       */
      IndexSpace subtract_index_spaces(Context ctx, 
                                       IndexSpace left, IndexSpace right);
      // Template version
      template<int DIM, typename COORD_T>
      IndexSpaceT<DIM,COORD_T> subtract_index_spaces(Context ctx,
           IndexSpaceT<DIM,COORD_T> left, IndexSpaceT<DIM,COORD_T> right);
      ///@}
      /**
       * @deprecated
       * Create a new top-level index space with the maximum number of elements
       * @param ctx the enclosing task context
       * @param max_num_elmts maximum number of elements in the index space
       * @return the handle for the new index space
       */
      LEGION_DEPRECATED("Use the new index space creation routines with a "
                        "single domain or rectangle.")
      IndexSpace create_index_space(Context ctx, size_t max_num_elmts);
      /**
       * @deprecated
       * Create a new top-level index space based on a set of domains
       * @param ctx the enclosing task context
       * @param domains the set of domains
       * @return the handle for the new index space
       */
      LEGION_DEPRECATED("Use the new index space creation routines with a "
                        "single domain or rectangle.")
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
      // (These are deprecated, use the dependent partitioning calls instead)
      //------------------------------------------------------------------------
      /**
       * @deprecated
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
      LEGION_DEPRECATED("Use the new dependent partitioning API calls instead.")
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                        const Domain &color_space,
                                        const PointColoring &coloring,
                                        PartitionKind part_kind = COMPUTE_KIND,
                                        Color color = AUTO_GENERATE_ID,
                                        bool allocable = false);
      /**
       * @deprecated
       * See the previous create_index_partition call
       * Create an index partition.
       * @param ctx the enclosing task context
       * @param parent index space being partitioned
       * @param coloring the coloring of the parent index space
       * @param disjoint whether the partitioning is disjoint or not
       * @param color optional color name for the partition
       * @return handle for the next index partition
       */
      LEGION_DEPRECATED("Use the new dependent partitioning API calls instead.")
      IndexPartition create_index_partition(Context ctx, IndexSpace parent, 
                                            const Coloring &coloring, 
                                            bool disjoint, 
                                            Color color = AUTO_GENERATE_ID);

      /**
       * @deprecated
       * Create an index partition from a domain point coloring
       * @param ctx the enclosing task context
       * @param parent the index space being partitioned
       * @param color_space space of colors for the partition
       * @param coloring the coloring of the parent index space
       * @param part_kind the kind of partition or whether to compute it
       * @param color optional color for the new partition
       * @return handle for the new index partition
       */
      LEGION_DEPRECATED("Use the new dependent partitioning API calls instead.")
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                        const Domain &color_space,
                                        const DomainPointColoring &coloring,
                                        PartitionKind part_kind = COMPUTE_KIND,
                                        Color color = AUTO_GENERATE_ID);
      /**
       * @deprecated
       * See the previous create index partition call
       * Create an index partition from a domain color space and coloring.
       * @param ctx the enclosing task context
       * @param parent index space being partitioned
       * @param color_space the domain of colors 
       * @param coloring the domain coloring of the parent index space
       * @param disjoint whether the partitioning is disjoint or not
       * @param color optional color name for the partition
       * @return handle for the next index partition
       */
      LEGION_DEPRECATED("Use the new dependent partitioning API calls instead.")
      IndexPartition create_index_partition(Context ctx, IndexSpace parent, 
					    Domain color_space, 
                                            const DomainColoring &coloring,
					    bool disjoint,
                                            Color color = AUTO_GENERATE_ID);

      /**
       * @deprecated
       * Create an index partition from a multi-domain point coloring
       * @param ctx the enclosing task context
       * @param parent the index space being partitioned
       * @param color_space space of colors for the partition
       * @param coloring the coloring of the parent index space
       * @param part_kind the kind of partition or whether to compute it
       * @param color optional color for the new partition
       * @return handle for the new index partition
       */
      LEGION_DEPRECATED("Use the new dependent partitioning API calls instead.")
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                      const Domain &color_space,
                                      const MultiDomainPointColoring &coloring,
                                      PartitionKind part_kind = COMPUTE_KIND,
                                      Color color = AUTO_GENERATE_ID);
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
       * @param color optional color name for the partition
       * @return handle for the next index partition
       */
      LEGION_DEPRECATED("Use the new dependent partitioning API calls instead.")
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                            Domain color_space,
                                            const MultiDomainColoring &coloring,
                                            bool disjoint,
                                            Color color = AUTO_GENERATE_ID);
      /**
       * @deprecated
       * Create an index partitioning from a typed mapping.
       * @param ctx the enclosing task context
       * @param parent index space being partitioned
       * @param mapping the mapping of points to colors
       * @param color optional color name for the partition
       * @return handle for the next index partition
       */
      template <typename T>
      LEGION_DEPRECATED("Use the new dependent partitioning API calls instead.")
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
					    const T& mapping,
					    Color color = AUTO_GENERATE_ID);

      /**
       * @deprecated 
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
      LEGION_DEPRECATED("Use the new dependent partitioning API calls instead.")
      IndexPartition create_index_partition(Context ctx, IndexSpace parent,
       LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic> field_accessor,
                                        Color color = AUTO_GENERATE_ID);

      /**
       * Destroy an index partition
       * @param ctx the enclosing task context
       * @param handle index partition to be destroyed
       */
      void destroy_index_partition(Context ctx, IndexPartition handle);
    public:
      //------------------------------------------------------------------------
      // Dependent Partitioning Operations
      //------------------------------------------------------------------------
      ///@{
      /**
       * Create 'color_space' index subspaces (one for each point) in a 
       * common partition of the 'parent' index space. By definition the 
       * resulting partition will be disjoint. Users can also specify a 
       * minimum 'granularity' for the size of the index subspaces. Users 
       * can specify an optional color for the index partition.
       * @param ctx the enclosing task context
       * @param parent index space of the partition to be made
       * @param color_space space of colors to create 
       * @param granularity the minimum size of the index subspaces
       * @param color optional color paramter for the partition
       * @return name of the created index partition
       */
      IndexPartition create_equal_partition(Context ctx, IndexSpace parent,
                                            IndexSpace color_space, 
                                            size_t granularity = 1,
                                            Color color = AUTO_GENERATE_ID);
      template<int DIM, typename COORD_T, 
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexPartitionT<DIM,COORD_T> create_equal_partition(Context ctx,
                        IndexSpaceT<DIM,COORD_T> parent,
                        IndexSpaceT<COLOR_DIM,COLOR_COORD_T> color_space,
                        size_t granularity = 1, Color color = AUTO_GENERATE_ID);
      ///@}
      ///@{
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
       * a color to the new partition by the 'color' argument.
       * @param ctx the enclosing task context
       * @param parent the parent index space for the new partition
       * @param handle1 first index partition
       * @param handle2 second index partition
       * @param color_space space of colors to zip over
       * @param part_kind indicate the kind of partition
       * @param color the new color for the index partition
       * @return name of the created index partition
       */
      IndexPartition create_partition_by_union(Context ctx,
                                       IndexSpace parent,
                                       IndexPartition handle1,
                                       IndexPartition handle2,
                                       IndexSpace color_space,
                                       PartitionKind part_kind = COMPUTE_KIND,
                                       Color color = AUTO_GENERATE_ID);
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexPartitionT<DIM,COORD_T> create_partition_by_union(Context ctx,
                              IndexSpaceT<DIM,COORD_T> parent,
                              IndexPartitionT<DIM,COORD_T> handle1,
                              IndexPartitionT<DIM,COORD_T> handle2,
                              IndexSpaceT<COLOR_DIM,COLOR_COORD_T> color_space,
                              PartitionKind part_kind = COMPUTE_KIND,
                              Color color = AUTO_GENERATE_ID);
      ///@}
      ///@{
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
       * partition by the 'color' argument.
       * @param ctx the enclosing task context
       * @param parent the parent index space for the new partition
       * @param handle1 first index partition
       * @param handle2 second index partition
       * @param color_space space of colors to zip over
       * @param part_kind indicate the kind of partition
       * @param color the new color for the index partition
       * @return name of the created index partition
       */
      IndexPartition create_partition_by_intersection(Context ctx,
                                        IndexSpace parent,
                                        IndexPartition handle1,
                                        IndexPartition handle2,
                                        IndexSpace color_space,
                                        PartitionKind part_kind = COMPUTE_KIND,
                                        Color color = AUTO_GENERATE_ID);
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexPartitionT<DIM,COORD_T> create_partition_by_intersection(
                              Context ctx,
                              IndexSpaceT<DIM,COORD_T> parent,
                              IndexPartitionT<DIM,COORD_T> handle1,
                              IndexPartitionT<DIM,COORD_T> handle2,
                              IndexSpaceT<COLOR_DIM,COLOR_COORD_T> color_space,
                              PartitionKind part_kind = COMPUTE_KIND,
                              Color color = AUTO_GENERATE_ID);
      ///@}
      ///@{
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
       * the 'color' argument.
       * index spaces.
       * @param ctx the enclosing task context
       * @param parent the parent index space for the new partition
       * @param handle1 first index partition
       * @param handle2 second index partition
       * @param color_space space of colors to zip over
       * @param part_kind indicate the kind of partition
       * @param color the new color for the index partition
       * @return name of the created index partition
       */
      IndexPartition create_partition_by_difference(Context ctx,
                                        IndexSpace parent,
                                        IndexPartition handle1,
                                        IndexPartition handle2,
                                        IndexSpace color_space,
                                        PartitionKind part_kind = COMPUTE_KIND,
                                        Color color = AUTO_GENERATE_ID);
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexPartitionT<DIM,COORD_T> create_partition_by_difference(Context ctx,
                              IndexSpaceT<DIM,COORD_T> parent,
                              IndexPartitionT<DIM,COORD_T> handle1,
                              IndexPartitionT<DIM,COORD_T> handle2,
                              IndexSpaceT<COLOR_DIM,COLOR_COORD_T> color_space,
                              PartitionKind part_kind = COMPUTE_KIND,
                              Color color = AUTO_GENERATE_ID);
      ///@}
      ///@{
      /**
       * This performs a cross product between two different index
       * partitions. For every index subspace in the first index 
       * partition the runtime will create a new index partition
       * of that index space by intersecting each of the different index 
       * subspaces in the second index partition. As a result, whole set 
       * of new index partitions will be created. The user can request which
       * partition names to return by specifying a map of domain points
       * from the color space of the first index partition. If the map
       * is empty, no index partitions will be returned. The user can
       * can say what kind the partitions are using the 'part_kind'
       * argument. The user can also specify a color for the new partitions
       * using the 'color' argument. If a specific color is specified, it
       * must be available for a partition in each of the index subspaces
       * in the first index partition.
       * @param ctx the enclosing task context
       * @param handle1 the first index partition
       * @param handle2 the second index partition
       * @param handle optional map for new partitions (can be empty)
       * @param part_kind indicate the kinds for the partitions
       * @param color optional color for each of the new partitions
       * @return the color used for each of the partitions
       */
      Color create_cross_product_partitions(Context ctx,
                                  IndexPartition handle1,
                                  IndexPartition handle2,
                                  std::map<IndexSpace,IndexPartition> &handles,
                                  PartitionKind part_kind = COMPUTE_KIND,
                                  Color color = AUTO_GENERATE_ID);
      template<int DIM, typename COORD_T, 
               int COLOR_DIM, typename COLOR_COORD_T>
      Color create_cross_product_partitions(Context ctx,
                                  IndexPartitionT<DIM,COORD_T> handle1,
                                  IndexPartitionT<DIM,COORD_T> handle2,
                                  typename std::map<
                                    IndexSpaceT<DIM,COORD_T>,
                                    IndexPartitionT<DIM,COORD_T> > &handles,
                                  PartitionKind part_kind = COMPUTE_KIND,
                                  Color color = AUTO_GENERATE_ID);
      ///@}
      ///@{
      /**
       * Create association will construct an injective mapping between
       * the points of two different index spaces. The mapping will 
       * be constructed in a field of the domain logical region so that
       * there is one entry for each point in the index space of the
       * domain logical region. If the cardinality of domain index
       * space is larger than the cardinality of the range index space
       * then some entries in the field may not be written. It is the
       * responsiblity of the user to have initialized the field with
       * a "null" value to detect such cases. Users wishing to create
       * a bi-directional mapping between index spaces can also use
       * the versions of this method that take a logical region on
       * the range as well.
       * @param ctx the enclosing task context
       * @param domain the region for the results and source index space
       * @param domain_parent the region from which privileges are derived
       * @param fid the field of domain in which to place the results
       * @param range the index space to serve as the range of the mapping
       * @param id the ID of the mapper to use for mapping the fields
       * @param tag the tag to pass to the mapper for context
       */
      void create_association(Context ctx,
                              LogicalRegion domain,
                              LogicalRegion domain_parent,
                              FieldID domain_fid,
                              IndexSpace range,
                              MapperID id = 0,
                              MappingTagID tag = 0);
      void create_bidirectional_association(Context ctx,
                                            LogicalRegion domain,
                                            LogicalRegion domain_parent,
                                            FieldID domain_fid,
                                            LogicalRegion range,
                                            LogicalRegion range_parent,
                                            FieldID range_fid,
                                            MapperID id = 0,
                                            MappingTagID tag = 0);
      // Template versions
      template<int DIM1, typename COORD_T1, int DIM2, typename COORD_T2>
      void create_association(Context ctx,
                              LogicalRegionT<DIM1,COORD_T1> domain,
                              LogicalRegionT<DIM1,COORD_T1> domain_parent,
                              FieldID domain_fid, // type: Point<DIM2,COORD_T2>
                              IndexSpaceT<DIM2,COORD_T2> range,
                              MapperID id = 0,
                              MappingTagID tag = 0);
      template<int DIM1, typename COORD_T1, int DIM2, typename COORD_T2>
      void create_bidirectional_association(Context ctx,
                              LogicalRegionT<DIM1,COORD_T1> domain,
                              LogicalRegionT<DIM1,COORD_T1> domain_parent,
                              FieldID domain_fid, // type: Point<DIM2,COORD_T2>
                              LogicalRegionT<DIM2,COORD_T2> range,
                              LogicalRegionT<DIM2,COORD_T2> range_parent,
                              FieldID range_fid, // type: Point<DIM1,COORD_T1>
                              MapperID id = 0,
                              MappingTagID tag = 0);
      ///@}
      /**
       * Create partition by restriction will make a new partition of a
       * logical region by computing new restriction bounds for each 
       * of the different subregions. All the sub-regions will have
       * the same 'extent' (e.g. contain the same number of initial points).
       * The particular location of the extent for each sub-region is
       * determined by taking a point in the color space and transforming
       * it by multiplying it by a 'transform' matrix to compute a 
       * 'delta' for the particular subregion. This 'delta' is then added
       * to the bounds of the 'extent' rectangle to generate a new bounding
       * rectangle for the subregion of the given color. The runtime will
       * also automatically intersect the resulting bounding rectangle with 
       * the original bounds of the parent region to ensure proper containment.
       * This may result in empty subregions.
       * @param ctx the enclosing task context
       * @param parent the parent index space to be partitioned
       * @param color_space the color space of the partition
       * @param transform a matrix transformation to be performed on each color
       * @param extent the rectangle shape of each of the bounds
       * @param part_kind the specify the partition kind
       * @param color optional new color for the index partition
       * @return a new index partition of the parent index space
       */
      IndexPartition create_partition_by_restriction(Context ctx,
                                        IndexSpace parent,
                                        IndexSpace color_space,
                                        DomainTransform transform,
                                        Domain extent,
                                        PartitionKind part_kind = COMPUTE_KIND,
                                        Color color = AUTO_GENERATE_ID);
      // Template version
      template<int DIM, int COLOR_DIM, typename COORD_T>
      IndexPartitionT<DIM,COORD_T> create_partition_by_restriction(Context ctx,
                                IndexSpaceT<DIM,COORD_T> parent,
                                IndexSpaceT<COLOR_DIM,COORD_T> color_space,
                                Transform<DIM,COLOR_DIM,COORD_T> transform,
                                Rect<DIM,COORD_T> extent,
                                PartitionKind part_kind = COMPUTE_KIND,
                                Color color = AUTO_GENERATE_ID);

      /**
       * Create partition by blockify is a special (but common) case of 
       * create partition by restriction, that is guaranteed to create a 
       * disjoint partition given the blocking factor specified for each 
       * dimension. This call will also create an implicit color space
       * for the partition that is the caller's responsibility to reclaim.
       * This assumes an origin of (0)* for all dimensions of the extent.
       * @param ctx the enclosing task context
       * @param parent the parent index space to be partitioned
       * @param blocking factor the blocking factors for each dimension
       * @param color optional new color for the index partition
       * @return a new index partition of the parent index space
       */
      IndexPartition create_partition_by_blockify(Context ctx,
                                        IndexSpace parent,
                                        DomainPoint blocking_factor,
                                        Color color = AUTO_GENERATE_ID);
      // Template version
      template<int DIM, typename COORD_T>
      IndexPartitionT<DIM,COORD_T> create_partition_by_blockify(Context ctx,
                                    IndexSpaceT<DIM,COORD_T> parent,
                                    Point<DIM,COORD_T> blocking_factor,
                                    Color color = AUTO_GENERATE_ID);
      /**
       * An alternate version of create partition by blockify that also
       * takes an origin to use for the computation of the extent.
       * @param ctx the enclosing task context
       * @param parent the parent index space to be partitioned
       * @param blocking factor the blocking factors for each dimension
       * @param origin the origin to use for computing the extent
       * @param color optional new color for the index partition
       * @return a new index partition of the parent index space
       */
      IndexPartition create_partition_by_blockify(Context ctx,
                                        IndexSpace parent,
                                        DomainPoint blockify_factor,
                                        DomainPoint origin,
                                        Color color = AUTO_GENERATE_ID);
      // Template version
      template<int DIM, typename COORD_T>
      IndexPartitionT<DIM,COORD_T> create_partition_by_blockify(Context ctx,
                                    IndexSpaceT<DIM,COORD_T> parent,
                                    Point<DIM,COORD_T> blocking_factor,
                                    Point<DIM,COORD_T> origin,
                                    Color color = AUTO_GENERATE_ID);
      ///@{
      /**
       * Create partition by field uses an existing field in a logical
       * region to perform a partitioning operation. The field type
       * must be of 'Point<COLOR_DIM,COLOR_COORD_T>' type so that the 
       * runtime can interpret the field as an enumerated function from 
       * Point<DIM,COORD_TT> -> Point<COLOR_DIM,COLOR_COORD_T>. Pointers 
       * are assigned into index subspaces based on their assigned color. 
       * Pointers with negative entries will not be assigned into any 
       * index subspace. The resulting index partition is a partition 
       * of the index space of the logical region over which the 
       * operation is performed. By definition this partition is 
       * disjoint. The 'color' argument can be used to specify an 
       * optional color for the index partition.
       * @param ctx the enclosing task context
       * @param handle logical region handle containing the chosen
       *               field and of which a partition will be created
       * @param parent the parent region from which privileges are derived
       * @param fid the field ID of the logical region containing the coloring
       * @param color_space space of colors for the partition
       * @param color optional new color for the index partition
       * @param id the ID of the mapper to use for mapping the fields
       * @param tag the context tag to pass to the mapper
       * @return a new index partition of the index space of the logical region
       */
      IndexPartition create_partition_by_field(Context ctx,
                                               LogicalRegion handle,
                                               LogicalRegion parent,
                                               FieldID fid, 
                                               IndexSpace color_space,
                                               Color color = AUTO_GENERATE_ID,
                                               MapperID id = 0,
                                               MappingTagID tag = 0);
      template<int DIM, typename COORD_T, 
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexPartitionT<DIM,COORD_T> create_partition_by_field(Context ctx,
                          LogicalRegionT<DIM,COORD_T> handle,
                          LogicalRegionT<DIM,COORD_T> parent,
                          FieldID fid, // type: Point<COLOR_DIM,COLOR_COORD_T>
                          IndexSpaceT<COLOR_DIM,COLOR_COORD_T> color_space,
                          Color color = AUTO_GENERATE_ID,
                          MapperID id = 0, MappingTagID tag = 0);
      ///@}
      ///@{
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
       * 'color' argument.
       * @param ctx the enclosing task context
       * @param handle the parent index space of the new index partition
       *               and the one in which all the ptr_t contained in
       *               'fid' must point
       * @param projection the logical partition of which we are creating
       *                   a projected version of through the field
       * @param parent the parent region from which privileges are derived
       * @param fid the field ID of the 'projection' logical partition
       *            we are reading which contains ptr_t@handle
       * @param color_space the index space of potential colors
       * @param part_kind specify the kind of partition
       * @param color optional new color for the index partition
       * @param id the ID of the mapper to use for mapping field
       * @param tag the mapper tag to provide context to the mapper
       * @return a new index partition of the 'handle' index space
       */
      IndexPartition create_partition_by_image(Context ctx,
                                         IndexSpace handle,
                                         LogicalPartition projection,
                                         LogicalRegion parent,
                                         FieldID fid,
                                         IndexSpace color_space,
                                         PartitionKind part_kind = COMPUTE_KIND,
                                         Color color = AUTO_GENERATE_ID,
                                         MapperID id = 0, MappingTagID tag = 0);
      template<int DIM1, typename COORD_T1, 
               int DIM2, typename COORD_T2, 
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexPartitionT<DIM2,COORD_T2> create_partition_by_image(Context ctx,
                              IndexSpaceT<DIM2,COORD_T2> handle,
                              LogicalPartitionT<DIM1,COORD_T1> projection,
                              LogicalRegionT<DIM1,COORD_T1> parent,
                              FieldID fid, // type: Point<DIM2,COORD_T2>
                              IndexSpaceT<COLOR_DIM,COLOR_COORD_T> color_space,
                              PartitionKind part_kind = COMPUTE_KIND,
                              Color color = AUTO_GENERATE_ID,
                              MapperID id = 0, MappingTagID tag = 0);
      // Range versions of image
      IndexPartition create_partition_by_image_range(Context ctx,
                                         IndexSpace handle,
                                         LogicalPartition projection,
                                         LogicalRegion parent,
                                         FieldID fid,
                                         IndexSpace color_space,
                                         PartitionKind part_kind = COMPUTE_KIND,
                                         Color color = AUTO_GENERATE_ID,
                                         MapperID id = 0, MappingTagID tag = 0);
      template<int DIM1, typename COORD_T1, 
               int DIM2, typename COORD_T2, 
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexPartitionT<DIM2,COORD_T2> create_partition_by_image_range(
                              Context ctx,
                              IndexSpaceT<DIM2,COORD_T2> handle,
                              LogicalPartitionT<DIM1,COORD_T1> projection,
                              LogicalRegionT<DIM1,COORD_T1> parent,
                              FieldID fid, // type: Rect<DIM2,COORD_T2>
                              IndexSpaceT<COLOR_DIM,COLOR_COORD_T> color_space,
                              PartitionKind part_kind = COMPUTE_KIND,
                              Color color = AUTO_GENERATE_ID,
                              MapperID id = 0, MappingTagID tag = 0);
      ///@}                                    
      ///@{
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
       * @param ctx the enclosing task context
       * @param projection the index partition being projected
       * @param handle logical region over which to evaluate the function
       * @param parent the parent region from which privileges are derived
       * @param fid the field ID of the 'handle' logical region containing
       *            the function being evaluated
       * @param color_space the space of colors for the partition
       * @param part_kind specify the kind of partition
       * @param color optional new color for the index partition
       * @param id the ID of the mapper to use for mapping field
       * @param tag the mapper tag to provide context to the mapper
       * @return a new index partition of the index space of 'handle'
       */
      IndexPartition create_partition_by_preimage(Context ctx, 
                                        IndexPartition projection,
                                        LogicalRegion handle,
                                        LogicalRegion parent,
                                        FieldID fid,
                                        IndexSpace color_space,
                                        PartitionKind part_kind = COMPUTE_KIND,
                                        Color color = AUTO_GENERATE_ID,
                                        MapperID id = 0, MappingTagID tag = 0);
      template<int DIM1, typename COORD_T1,
               int DIM2, typename COORD_T2,
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexPartitionT<DIM1,COORD_T1> create_partition_by_preimage(Context ctx,
                              IndexPartitionT<DIM2,COORD_T2> projection,
                              LogicalRegionT<DIM1,COORD_T1> handle,
                              LogicalRegionT<DIM1,COORD_T1> parent,
                              FieldID fid, // type: Point<DIM2,COORD_T2>
                              IndexSpaceT<COLOR_DIM,COLOR_COORD_T> color_space,
                              PartitionKind part_kind = COMPUTE_KIND,
                              Color color = AUTO_GENERATE_ID,
                              MapperID id = 0, MappingTagID tag = 0);
      // Range versions of preimage 
      IndexPartition create_partition_by_preimage_range(Context ctx, 
                                        IndexPartition projection,
                                        LogicalRegion handle,
                                        LogicalRegion parent,
                                        FieldID fid,
                                        IndexSpace color_space,
                                        PartitionKind part_kind = COMPUTE_KIND,
                                        Color color = AUTO_GENERATE_ID,
                                        MapperID id = 0, MappingTagID tag = 0);
      template<int DIM1, typename COORD_T1,
               int DIM2, typename COORD_T2,
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexPartitionT<DIM1,COORD_T1> create_partition_by_preimage_range(
                              Context ctx,
                              IndexPartitionT<DIM2,COORD_T2> projection,
                              LogicalRegionT<DIM1,COORD_T1> handle,
                              LogicalRegionT<DIM1,COORD_T1> parent,
                              FieldID fid, // type: Rect<DIM2,COORD_T2>
                              IndexSpaceT<COLOR_DIM,COLOR_COORD_T> color_space,
                              PartitionKind part_kind = COMPUTE_KIND,
                              Color color = AUTO_GENERATE_ID,
                              MapperID id = 0, MappingTagID tag = 0);
      ///@}
    public:
      //------------------------------------------------------------------------
      // Computed Index Spaces and Partitions 
      //------------------------------------------------------------------------
      ///@{
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
       * remaining index subspaces).
       * @param ctx the enclosing task context
       * @param parent the parent index space for the partition
       * @param color_space the color space for the new partition
       * @param part_kind optionally specify the partition kind
       * @param color optionally assign a color to the partition
       * @return handle of the new index partition
       */
      IndexPartition create_pending_partition(Context ctx,
                                              IndexSpace parent,
                                              IndexSpace color_space,
                                  PartitionKind part_kind = COMPUTE_KIND, 
                                  Color color = AUTO_GENERATE_ID);
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexPartitionT<DIM,COORD_T> create_pending_partition(Context ctx,
                              IndexSpaceT<DIM,COORD_T> parent,
                              IndexSpaceT<COLOR_DIM,COLOR_COORD_T> color_space,
                              PartitionKind part_kind = COMPUTE_KIND,
                              Color color = AUTO_GENERATE_ID);
      ///@}
      ///@{
      /**
       * Create a new index space by unioning together a bunch of index spaces 
       * from a common index space tree. The resulting index space is assigned
       * to be the index space corresponding to 'color' of the 'parent' index
       * partition. It is illegal to invoke this method with a 'parent' index
       * partition that was not created by a 'create_pending_partition' call.
       * All of the index spaces being unioned together must come from the
       * same index space tree.
       * @param ctx the enclosing task context
       * @param parent the parent index partition 
       * @param color the color to assign the index space to in the parent
       * @param handles a vector index space handles to union
       * @return handle of the new index space
       */
      IndexSpace create_index_space_union(Context ctx, IndexPartition parent, 
                                        const DomainPoint &color,
                                        const std::vector<IndexSpace> &handles);
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexSpaceT<DIM,COORD_T> create_index_space_union(Context ctx,
                                IndexPartitionT<DIM,COORD_T> parent,
                                Point<COLOR_DIM,COLOR_COORD_T> color,
                                const typename std::vector<
                                  IndexSpaceT<DIM,COORD_T> > &handles);
      ///@}
      ///@{
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
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexSpaceT<DIM,COORD_T> create_index_space_union(Context ctx,
                                IndexPartitionT<DIM,COORD_T> parent,
                                Point<COLOR_DIM,COLOR_COORD_T> color,
                                IndexPartitionT<DIM,COORD_T> handle);
      ///@}
      ///@{
      /**
       * Create a new index space by intersecting together a bunch of index
       * spaces from a common index space tree. The resulting index space is
       * assigned to be the index space corresponding to 'color' of the 
       * 'parent' index partition. It is illegal to invoke this method with
       * a 'parent' index partition that was not created by a call to 
       * 'create_pending_partition'. All of the index spaces being 
       * intersected together must come from the same index space tree.
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
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexSpaceT<DIM,COORD_T> create_index_space_intersection(Context ctx,
                                IndexPartitionT<DIM,COORD_T> parent,
                                Point<COLOR_DIM,COLOR_COORD_T> color,
                                const typename std::vector<
                                  IndexSpaceT<DIM,COORD_T> > &handles);
      ///@}
      ///@{
      /**
       * This method is the same as the one above, except the index
       * spaces all come from a common partition specified by 'handle'.
       * The resulting index space will be an intersection of all the index
       * spaces of 'handle'.
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
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexSpaceT<DIM,COORD_T> create_index_space_intersection(Context ctx,
                                IndexPartitionT<DIM,COORD_T> parent,
                                Point<COLOR_DIM,COLOR_COORD_T> color,
                                IndexPartitionT<DIM,COORD_T> handle);
      ///@}
      ///@{
      /**
       * Create a new index space by taking the set difference of
       * an existing index space with a set of other index spaces.
       * The resulting index space is assigned to be the index space
       * corresponding to 'color' of the 'parent' index partition.
       * It is illegal to invoke this method with a 'parent' index
       * partition that was not created by a call to 'create_pending_partition'.
       * The 'initial' index space is the index space from which 
       * differences will be performed, and each of the index spaces in 
       * 'handles' will be subsequently subtracted from the 'initial' index
       * space. All of the index spaces in 'handles' as well as 'initial'
       * must come from the same index space tree.
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
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexSpaceT<DIM,COORD_T> create_index_space_difference(Context ctx,
                                IndexPartitionT<DIM,COORD_T> parent,
                                Point<COLOR_DIM,COLOR_COORD_T> color,
                                IndexSpaceT<DIM,COORD_T> initial,
                                const typename std::vector<
                                  IndexSpaceT<DIM,COORD_T> > &handles);
      ///@}
    public:
      //------------------------------------------------------------------------
      // Index Tree Traversal Operations
      //------------------------------------------------------------------------
      ///@{
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
      // Template version
      template<int DIM, typename COORD_T>
      IndexPartitionT<DIM,COORD_T> get_index_partition(
                                  IndexSpaceT<DIM,COORD_T> parent, Color color);

      ///@}
      
      ///@{
      /**
       * Return true if the index space has an index partition
       * with the specified color.
       * @param ctx enclosing task context
       * @param parent index space
       * @param color of index partition
       * @return true if an index partition exists with the specified color
       */
      bool has_index_partition(Context ctx, IndexSpace parent, Color color);
      bool has_index_partition(Context ctx, IndexSpace parent,
                               const DomainPoint &color);
      // Context free
      bool has_index_partition(IndexSpace parent, Color color);
      bool has_index_partition(IndexSpace parent,
                               const DomainPoint &color);
      // Template version
      template<int DIM, typename COORD_T>
      bool has_index_partition(IndexSpaceT<DIM,COORD_T> parent, Color color);
      ///@}

      ///@{
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
      // Template version
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexSpaceT<DIM,COORD_T> get_index_subspace(
                                  IndexPartitionT<DIM,COORD_T> p,
                                  Point<COLOR_DIM,COLOR_COORD_T> color);
      ///@}

      ///@{
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
      // Template version
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      bool has_index_subspace(IndexPartitionT<DIM,COORD_T> p,
                              Point<COLOR_DIM,COLOR_COORD_T> color);
      ///@}

      ///@{
      /**
       * @deprecated
       * Return if the given index space is represented by 
       * multiple domains or just a single one. If multiple
       * domains represent the index space then 'get_index_space_domains'
       * should be used for getting the set of domains.
       * @param ctx enclosing task context
       * @param handle index space handle
       * @return true if the index space has multiple domains
       */
      LEGION_DEPRECATED("Multiple domains are no longer supported.")
      bool has_multiple_domains(Context ctx, IndexSpace handle);
      // Context free
      LEGION_DEPRECATED("Multiple domains are no longer supported.")
      bool has_multiple_domains(IndexSpace handle);
      ///@}

      ///@{
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
      // Template version
      template<int DIM, typename COORD_T>
      DomainT<DIM,COORD_T> get_index_space_domain(
                                        IndexSpaceT<DIM,COORD_T> handle);
      ///@}

      ///@{
      /**
       * @deprecated
       * Return the domains that represent the index space.
       * While the previous call only works when there is a
       * single domain for the index space, this call will
       * work in all circumstances.
       * @param ctx enclosing task context
       * @param handle index space handle
       * @param vector to populate with domains
       */
      LEGION_DEPRECATED("Multiple domains are no longer supported.")
      void get_index_space_domains(Context ctx, IndexSpace handle,
                                   std::vector<Domain> &domains);
      // Context free
      LEGION_DEPRECATED("Multiple domains are no longer supported.")
      void get_index_space_domains(IndexSpace handle,
                                   std::vector<Domain> &domains);
      ///@}

      ///@{
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
      // Template version
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      DomainT<COLOR_DIM,COLOR_COORD_T>
             get_index_partition_color_space(IndexPartitionT<DIM,COORD_T> p);
      ///@}

      ///@{
      /**
       * Return the name of the color space for a partition
       * @param ctx enclosing task context
       * @param p handle for the index partition
       * @return the name of the color space of the specified partition
       */
      IndexSpace get_index_partition_color_space_name(Context ctx, 
                                                      IndexPartition p);
      // Context free
      IndexSpace get_index_partition_color_space_name(IndexPartition p);
      // Template version
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      IndexSpaceT<COLOR_DIM,COLOR_COORD_T> 
           get_index_partition_color_space_name(IndexPartitionT<DIM,COORD_T> p);
      ///@}

      ///@{
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
      ///@}

      ///@{
      /**
       * Return whether a given index partition is disjoint
       * @param ctx enclosing task context
       * @param p index partition handle
       * @return whether the index partition is disjoint
       */
      bool is_index_partition_disjoint(Context ctx, IndexPartition p);
      // Context free
      bool is_index_partition_disjoint(IndexPartition p);
      ///@}

      ///@{
      /**
       * Return whether a given index partition is complete
       * @param ctx enclosing task context
       * @param p index partition handle
       * @return whether the index partition is complete
       */
      bool is_index_partition_complete(Context ctx, IndexPartition p);
      // Context free
      bool is_index_partition_complete(IndexPartition p);
      ///@}

      /**
       * @deprecated
       * Get an index subspace from a partition with a given color point.
       * @param ctx enclosing task context
       * @param p parent index partition handle
       * @param color_point point containing color value of index subspace
       * @return the corresponding index space to the specified color point
       */
      template <unsigned DIM>
      LEGION_DEPRECATED("Use the new templated methods for geting a subspace.")
      IndexSpace get_index_subspace(Context ctx, IndexPartition p, 
                                LegionRuntime::Arrays::Point<DIM> color_point);

      ///@{
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
      // Template version
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      Point<COLOR_DIM,COLOR_COORD_T>
            get_index_space_color(IndexSpaceT<DIM,COORD_T> handle);
      ///@}

      ///@{
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
      ///@}

      ///@{
      /**
       * Return the index space parent for the given index partition.
       * @param ctx enclosing task context
       * @param handle for the index partition
       * @return index space for the parent
       */
      IndexSpace get_parent_index_space(Context ctx, IndexPartition handle);
      // Context free
      IndexSpace get_parent_index_space(IndexPartition handle);
      // Template version
      template<int DIM, typename COORD_T>
      IndexSpaceT<DIM,COORD_T> get_parent_index_space(
                                      IndexPartitionT<DIM,COORD_T> handle);
      ///@}

      ///@{
      /**
       * Returns true if the given index space has a parent partition.
       * @param ctx enclosing task context
       * @param handle for the index space
       * @return true if there is a parent index partition
       */
      bool has_parent_index_partition(Context ctx, IndexSpace handle);
      // Context free
      bool has_parent_index_partition(IndexSpace handle);
      ///@}

      ///@{
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
      // Template version
      template<int DIM, typename COORD_T>
      IndexPartitionT<DIM,COORD_T> get_parent_index_partition(
                                              IndexSpaceT<DIM,COORD_T> handle);
      ///@}

      ///@{
      /**
       * Return the depth in the index space tree of the given index space.
       * @param ctx enclosing task context
       * @param handle the index space
       * @return depth in the index space tree of the index space
       */
      unsigned get_index_space_depth(Context ctx, IndexSpace handle);
      // Context free
      unsigned get_index_space_depth(IndexSpace handle);
      ///@}

      ///@{
      /**
       * Return the depth in the index space tree of the given index partition.
       * @param ctx enclosing task context
       * @param handle the index partition
       * @return depth in the index space tree of the index partition
       */
      unsigned get_index_partition_depth(Context ctx, IndexPartition handle);
      // Context free
      unsigned get_index_partition_depth(IndexPartition handle);
      ///@}
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
      
      /**
       * Safe case a domain point down to a target region.  If the point
       * is not in the target region, returns false, otherwise returns true
       * @param ctx enclosing task context
       * @param point the domain point to be cast
       * @param region the target logical region
       * @return if the point is in the logical region
       */
      template<int DIM, typename COORD_T>
      bool safe_cast(Context ctx, 
                     Point<DIM,COORD_T> point,
                     LogicalRegionT<DIM,COORD_T> region);
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

      ///@{
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
      ///@}

      ///@{
      /**
       * Get the IDs of the fields currently allocated in a field space.
       * @param ctx enclosing task context
       * @param handle field space handle
       * @param set in which to place the field IDs
       */
      void get_field_space_fields(Context ctx, FieldSpace handle,
                                  std::vector<FieldID> &fields);
      // Context free
      void get_field_space_fields(FieldSpace handle,
                                  std::vector<FieldID> &fields);
      ///@}

      ///@{
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
      ///@}
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
      // Template version
      template<int DIM, typename COORD_T>
      LogicalRegionT<DIM,COORD_T> create_logical_region(Context ctx,
                                      IndexSpaceT<DIM,COORD_T> index,
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
      ///@{
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
      // Template version
      template<int DIM, typename COORD_T>
      LogicalPartitionT<DIM,COORD_T> get_logical_partition(
                                      LogicalRegionT<DIM,COORD_T> parent,
                                      IndexPartitionT<DIM,COORD_T> handle);
      ///@}
      
      ///@{
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
      // Template version
      template<int DIM, typename COORD_T>
      LogicalPartitionT<DIM,COORD_T> get_logical_partition_by_color(
                                        LogicalRegionT<DIM,COORD_T> parent,
                                        Color c);
      ///@}

      ///@{
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
      ///@}
      
      ///@{
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
      // Template version
      template<int DIM, typename COORD_T>
      LogicalPartitionT<DIM,COORD_T> get_logical_partition_by_tree(
                                      IndexPartitionT<DIM,COORD_T> handle,
                                      FieldSpace fspace, RegionTreeID tid);
      ///@}

      ///@{
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
      // Template version
      template<int DIM, typename COORD_T>
      LogicalRegionT<DIM,COORD_T> get_logical_subregion(
                                          LogicalPartitionT<DIM,COORD_T> parent,
                                          IndexSpaceT<DIM,COORD_T> handle);
      ///@}

      ///@{
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
      // Template version
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      LogicalRegionT<DIM,COORD_T> get_logical_subregion_by_color(
                                  LogicalPartitionT<DIM,COORD_T> parent,
                                  Point<COLOR_DIM,COLOR_COORD_T> color);
      ///@}

      ///@{
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
      // Template version
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      bool has_logical_subregion_by_color(LogicalPartitionT<DIM,COORD_T> parent,
                                  Point<COLOR_DIM,COLOR_COORD_T> color);
      ///@}

      ///@{
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
      // Template version
      template<int DIM, typename COORD_T>
      LogicalRegionT<DIM,COORD_T> get_logical_subregion_by_tree(
                                    IndexSpaceT<DIM,COORD_T> handle,
                                    FieldSpace space, RegionTreeID tid);
      ///@}

      ///@{
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
      // Template version
      template<int DIM, typename COORD_T,
               int COLOR_DIM, typename COLOR_COORD_T>
      Point<COLOR_DIM,COLOR_COORD_T>
        get_logical_region_color_point(LogicalRegionT<DIM,COORD_T> handle);
      ///@}

      ///@{
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
      ///@}

      ///@{
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
      // Template version
      template<int DIM, typename COORD_T>
      LogicalRegionT<DIM,COORD_T> get_parent_logical_region(
                                    LogicalPartitionT<DIM,COORD_T> handle);
      ///@}

      ///@{
      /**
       * Return true if the logical region has a parent logical partition.
       * @param ctx enclosing task context
       * @param handle for the logical region for which to check for a parent
       * @return true if a parent exists
       */
      bool has_parent_logical_partition(Context ctx, LogicalRegion handle);
      // Context free
      bool has_parent_logical_partition(LogicalRegion handle);
      ///@}

      ///@{
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
      // Template version
      template<int DIM, typename COORD_T>
      LogicalPartitionT<DIM,COORD_T> get_parent_logical_partition(
                                          LogicalRegionT<DIM,COORD_T> handle);
      ///@}
    public:
      //------------------------------------------------------------------------
      // Allocator and Argument Map Operations 
      //------------------------------------------------------------------------
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
      /**
       * @deprecated
       * Create an index allocator object for a given index space
       * This method is deprecated becasue index spaces no longer support
       * dynamic allocation. This will still work only if there is exactly
       * one allocator made for the index space throughout the duration
       * of its lifetime.
       * @param ctx enclosing task context
       * @param handle for the index space to create an allocator
       * @return a new index space allocator for the given index space
       */
      LEGION_DEPRECATED("Dynamic index allocation is no longer supported.")
      IndexAllocator create_index_allocator(Context ctx, IndexSpace handle);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif

      /**
       * Create a field space allocator object for the given field space
       * @param ctx enclosing task context
       * @param handle for the field space to create an allocator
       * @return a new field space allocator for the given field space
       */
      FieldAllocator create_field_allocator(Context ctx, FieldSpace handle);

      /**
       * @deprecated
       * Create an argument map in the given context.  This method
       * is deprecated as argument maps can now be created directly
       * by a simple declaration.
       * @param ctx enclosing task context
       * @return a new argument map
       */
      LEGION_DEPRECATED("ArgumentMap can be constructed directly.")
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
       * @see IndexTaskLauncher
       * @param ctx enclosing task context
       * @param launcher the task launcher configuration
       * @return a future map for return values of the points
       *    in the index space of tasks
       */
      FutureMap execute_index_space(Context ctx, 
                                    const IndexTaskLauncher &launcher);

      /**
       * Launch an index space of tasks with arguments specified
       * by the index launcher configuration.  Reduce all the
       * return values into a single value using the specified
       * reduction operator into a single future value.  The reduction
       * operation must be a foldable reduction.
       * @see IndexTaskLauncher
       * @param ctx enclosing task context
       * @param launcher the task launcher configuration
       * @param redop ID for the reduction op to use for reducing return values
       * @return a future result representing the reduction of
       *    all the return values from the index space of tasks
       */
      Future execute_index_space(Context ctx, const IndexTaskLauncher &launcher,
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
      LEGION_DEPRECATED("Launching tasks should be done with the new task "
                        "launcher interface.")
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
      LEGION_DEPRECATED("Launching tasks should be done with the new task "
                        "launcher interface.")
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
      LEGION_DEPRECATED("Launching tasks should be done with the new task "
                        "launcher interface.")
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

      /**
       * Perform an index fill operation using a launcher which
       * specifies all the parameters of the launch.
       * @param ctx enclosing task context
       * @param launcher the launcher that describes the index fill operation
       */
      void fill_fields(Context ctx, const IndexFillLauncher &launcher);
    public:
      //------------------------------------------------------------------------
      // Attach Operations
      //------------------------------------------------------------------------

      /**
       * Attach an external resource to a logical region
       * @param ctx enclosing task context
       * @param launcher the attach launcher that describes the resource
       * @return the physical region for the external resource
       */
      PhysicalRegion attach_external_resource(Context ctx, 
                                              const AttachLauncher &launcher);

      /**
       * Detach an external resource from a logical region
       * @param ctx enclosing task context
       * @param the physical region for the external resource
       */
      void detach_external_resource(Context ctx, PhysicalRegion region);

      /**
       * @deprecated
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
      LEGION_DEPRECATED("Attaching specific HDF5 file type is deprecated "
                        "in favor of generic attach launcher interface.")
      PhysicalRegion attach_hdf5(Context ctx, const char *file_name,
                                 LogicalRegion handle, LogicalRegion parent, 
                                 const std::map<FieldID,const char*> &field_map,
                                 LegionFileMode mode);

      /**
       * @deprecated
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
      LEGION_DEPRECATED("Detaching specific HDF5 file type is deprecated "
                        "in favor of generic detach interface.")
      void detach_hdf5(Context ctx, PhysicalRegion region);

      /**
       * @deprecated
       * Attach an normal file as a physical region. This attach is similar to
       * attach_hdf5 operation, except that the file has exact same data format
       * as in-memory physical region. Data lays out as SOA in file.
       */
      LEGION_DEPRECATED("Attaching generic file type is deprecated "
                        "in favor of generic attach launcher interface.")
      PhysicalRegion attach_file(Context ctx, const char *file_name,
                                 LogicalRegion handle, LogicalRegion parent,
                                 const std::vector<FieldID> &field_vec,
                                 LegionFileMode mode);

      /**
       * @deprecated
       * Detach an normal file. THis detach operation is similar to
       * detach_hdf5
       */
      LEGION_DEPRECATED("Detaching generic file type is deprecated "
                        "in favor of generic detach interface.")
      void detach_file(Context ctx, PhysicalRegion region);
    public:
      //------------------------------------------------------------------------
      // Copy Operations
      //------------------------------------------------------------------------
      /**
       * Launch a copy operation from the given configuration of
       * the given copy launcher.
       * @see CopyLauncher
       * @param ctx enclosing task context
       * @param launcher copy launcher object
       */
      void issue_copy_operation(Context ctx, const CopyLauncher &launcher);

      /**
       * Launch an index copy operation from the given configuration
       * of the given copy launcher
       * @see IndexCopyLauncher
       * @param ctx enclosing task context
       * @param launcher index copy launcher object
       */
      void issue_copy_operation(Context ctx, const IndexCopyLauncher &launcher);
    public:
      //------------------------------------------------------------------------
      // Predicate Operations
      //------------------------------------------------------------------------
      /**
       * Create a new predicate value from a future.  The future passed
       * must be a boolean future.
       * @param ctx enclosing task context
       * @param f future value to convert to a predicate
       * @return predicate value wrapping the future
       */
      Predicate create_predicate(Context ctx, const Future &f);

      /**
       * Create a new predicate value that is the logical 
       * negation of another predicate value.
       * @param ctx enclosing task context
       * @param p predicate value to logically negate
       * @return predicate value logically negating previous predicate
       */
      Predicate predicate_not(Context ctx, const Predicate &p);

      /**
       * Create a new predicate value that is the logical
       * conjunction of two other predicate values.
       * @param ctx enclosing task context
       * @param p1 first predicate to logically and 
       * @param p2 second predicate to logically and
       * @return predicate value logically and-ing two predicates
       */
      Predicate predicate_and(Context ctx, const Predicate &p1, 
                                           const Predicate &p2);

      /**
       * Create a new predicate value that is the logical
       * disjunction of two other predicate values.
       * @param ctx enclosing task context
       * @param p1 first predicate to logically or
       * @param p2 second predicate to logically or
       * @return predicate value logically or-ing two predicates
       */
      Predicate predicate_or(Context ctx, const Predicate &p1, 
                                          const Predicate &p2);

      /**
       * Generic predicate constructor for an arbitrary number of predicates
       * @param ctx enclosing task context
       * @param launcher the predicate launcher
       * #return predicate value of combining other predicates
       */
      Predicate create_predicate(Context ctx,const PredicateLauncher &launcher);

      /**
       * Get a future value that will be completed when the predicate triggers
       * @param ctx enclosing task context
       * @param pred the predicate for which to get a future
       * @return a boolean future with the result of the predicate
       */
      Future get_predicate_future(Context ctx, const Predicate &p);
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
                                            const Future &f,
                                            unsigned count = 1);

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
      /**
       * Start a new static trace of operations. Inside of this trace
       * it is the application's responsibility to specify any dependences
       * that would normally have existed between each operation being
       * launched and any prior operations in the trace (there is no need
       * to specify dependences on anything outside of the trace). The
       * application can optionally specify a set of region trees for 
       * which it will be supplying dependences, with all other region
       * trees being left to the runtime to handle. If no such set is
       * specified then the runtime will operate under the assumption
       * that the application is specifying dependences for all region trees.
       * @param ctx the enclosing task context
       * @param managed optional list of region trees managed by the application
       */
      void begin_static_trace(Context ctx, 
                              const std::set<RegionTreeID> *managed = NULL);
      /**
       * Finish a static trace of operations
       * @param ctx the enclosing task context
       */
      void end_static_trace(Context ctx);
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
       * tunables which are filled in at runtime by mapper objects. This
       * method will return asynchronously with a future that will be set
       * once the mapper fills in the value for the future. It is the 
       * responsibility of the application to maintain consistency on the
       * expected types for a given tunable between the application and
       * the mapper.
       */
      Future select_tunable_value(Context ctx, TunableID tid,
                                  MapperID mapper = 0, MappingTagID tag = 0);

      /**
       * @deprecated
       * This is the old method for asking the mapper to specify a 
       * tunable value. It will assume that the resulting tunable 
       * future can be interpreted as an integer.
       */
      LEGION_DEPRECATED("Tunable values should now be obtained via the "
                        "generic interface that returns a future result.")
      int get_tunable_value(Context ctx, TunableID tid, 
                            MapperID mapper = 0, MappingTagID tag = 0);
    public:
      //------------------------------------------------------------------------
      // Task Local Interface
      //------------------------------------------------------------------------
      /**
       * Get a reference to the task for the current context.
       * @param the enclosing task context 
       * @return a pointer to the task for the context
       */
      const Task* get_local_task(Context ctx);

      /**
       * Get the value of a task-local variable named by the ID. This
       * variable only has the lifetime of the task
       * @param ctx the enclosing task context
       * @param id the ID of the task-local variable to return
       * @return pointer to the value of the variable if any
       */
      void* get_local_task_variable_untyped(Context ctx, LocalVariableID id);
      template<typename T>
      T* get_local_task_variable(Context ctx, LocalVariableID id);

      /**
       * Set the value of a task-local variable named by ID. This 
       * variable will only have the lifetime of the task. The user
       * can also specify an optional destructor function which will
       * implicitly be called at the end the task's execution
       * @param ctx the enclosing task context
       * @param id the ID of the task-local variable to set
       * @param value the value to set the variable to
       * @param destructor optional method to delete the value
       */
      void set_local_task_variable_untyped(Context ctx, LocalVariableID id, 
          const void *value, void (*destructor)(void*) = NULL);
      template<typename T>
      void set_local_task_variable(Context ctx, LocalVariableID id,
          const T* value, void (*destructor)(void*) = NULL);
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
       * Issue a timing measurement operation configured with a launcher.
       * The above methods are just common special cases. This allows for
       * the general case of an arbitrary measurement with an arbitrary
       * number of preconditions.
       */
      Future issue_timing_measurement(Context ctx, 
                                      const TimingLauncher &launcher);

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
      Mapping::Mapper* get_mapper(Context ctx, MapperID id, 
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

      /**
       * Return the local MPI rank ID for the current Legion runtime
       */
      int find_local_MPI_rank(void);
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
       * Get the mapper runtime for passing to a newly created mapper.
       * @return a pointer to the mapper runtime for this Legion instance
       */
      Mapping::MapperRuntime* get_mapper_runtime(void);

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
       * to use when mapping tasks. If a specific processor is passed
       * to the call then the mapper instance will only be registered
       * on that processor. Alternatively, if no processor is passed,
       * then the mapper will be registered with all processors on
       * the local node.
       * @param map_id the mapper ID to associate with the mapper
       * @param mapper pointer to the mapper object
       * @param proc the processor to associate the mapper with
       */
      void add_mapper(MapperID map_id, Mapping::Mapper *mapper, 
                      Processor proc = Processor::NO_PROC);
      
      /**
       * Replace the default mapper for a given processor with
       * a new mapper.  If a specific processor is passed to the call
       * then the mapper instance will only be registered on that
       * processor. Alternatively, if no processor is passed, then
       * the mapper will be registered with all processors on 
       * the local node.
       * @param mapper pointer to the mapper object to use
       *    as the new default mapper
       * @param proc the processor to associate the mapper with
       */
      void replace_default_mapper(Mapping::Mapper *mapper, 
                                  Processor proc = Processor::NO_PROC);

    public:
      /**
       * Dynamically generate a unique projection ID for use across the machine
       * @reutrn a ProjectionID that is globally unique across the machine
       */
      ProjectionID generate_dynamic_projection_id(void);

      /**
       * Statically generate a unique Projection ID for use across the machine.
       * This can only be called prior to the runtime starting. It must be
       * invoked symmetrically across all the nodes in the machine prior
       * to starting the runtime.
       * @return a ProjectionID that is globally unique across the machine
       */
      static ProjectionID generate_static_projection_id(void);

      /**
       * Register a projection functor for handling projection
       * queries. The ProjectionID must be non-zero because 
       * zero is the identity projection. Unlike mappers which
       * require a separate instance per processor, only
       * one of these must be registered per projection ID.
       * The runtime takes ownership for deleting the projection 
       * functor after the application has finished executing.
       * @param pid the projection ID to use for the registration
       * @param functor the object to register for handling projections
       */
      void register_projection_functor(ProjectionID pid, 
                                       ProjectionFunctor *functor);

      /**
       * Register a projection functor before the runtime has started only.
       * The runtime will update the projection functor so that it has 
       * contains a valid runtime pointer prior to the projection functor
       * ever being invoked. The runtime takes ownership for deleting the
       * projection functor after the application has finished executing.
       * @param pid the projection ID to use for the registration
       * @param functor the objecto register for handling projections
       */
      static void preregister_projection_functor(ProjectionID pid,
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
       * -lg:nosteal  Disable any stealing in the runtime.  The runtime
       *              will never query any mapper about stealing.
       * ------------------------
       *  Out-of-order Execution
       * ------------------------
       * -lg:window <int> Specify the maximum number of child tasks
       *              allowed in a given task context at a time.  A call
       *              to launch more tasks than the allotted window
       *              will stall the parent task until child tasks
       *              begin completing.  The default is 1024.
       * -lg:sched <int> The run-ahead factor for the runtime.  How many
       *              outstanding tasks ready to run should be on each
       *              processor before backing off the mapping procedure.
       * -lg:inorder  Execute operations in strict propgram order. This
       *              flag will actually run the entire operation through
       *              the pipeline and wait for it to complete before
       *              permitting the next operation to start.
       * -------------
       *  Messaging
       * -------------
       * -lg:message <int> Maximum size in bytes of the active messages
       *              to be sent between instances of the Legion
       *              runtime.  This can help avoid the use of expensive
       *              per-pair-of-node RDMA buffers in the low-level
       *              runtime.  Default value is 4K which should guarantee
       *              medium sized active messages on Infiniband clusters.
       * ---------------------
       *  Configuration Flags 
       * ---------------------
       * -lg:no_dyn   Disable dynamic disjointness tests when the runtime
       *              has been compiled with macro DYNAMIC_TESTS defined
       *              which enables dynamic disjointness testing.
       * -lg:epoch <int> Change the size of garbage collection epochs. The
       *              default value is 64. Increasing it adds latency to
       *              the garbage collection but makes it more efficient.
       *              Decreasing the value reduces latency, but adds
       *              inefficiency to the collection.
       * -lg:unsafe_launch Tell the runtime to skip any checks for 
       *              checking for deadlock between a parent task and
       *              the sub-operations that it is launching. Note
       *              that this is unsafe for a reason. The application
       *              can and will deadlock if any currently mapped
       *              regions conflict with those requested by a child
       *              task or other operation.
       * -lg:unsafe_mapper Tell the runtime to skip any checks for 
       *              validating the correctness of the results from 
       *              mapper calls. Turning this off may result in 
       *              internal crashes in the runtime if the mapper
       *              provides invalid output from any mapper call.
       *              (Default: false in debug mode, true in release mode.)
       * -lg:safe_mapper Tell the runtime to perform all correctness
       *              checks on mapper calls regardless of the 
       *              optimization level. (Default: true in debug mode,
       *              false in release mode.)
       * -lg:local <int> Specify the maximum number of local fields
       *              permitted in any field space within a context.
       * ---------------------
       *  Resiliency
       * ---------------------
       * -lg:resilient Enable features that make the runtime resilient
       *              including deferred commit that can be controlled
       *              by the next two flags.  By default this is off
       *              for performance reasons.  Once resiliency mode
       *              is enabled, then the user can control when 
       *              operations commit using the next two flags.
       * -------------
       *  Debugging
       * ------------- 
       * -lg:warn     Enable all verbose runtime warnings
       * -lg:ldb <replay_file> Replay the execution of the application
       *              with the associated replay file generted by LegionSpy. 
       *              This will run the application in the Legion debugger.
       * -lg:replay <replay_file> Rerun the execution of the application with
       *              the associated replay file generated by LegionSpy.
       * -lg:tree     Dump intermediate physical region tree states before
       *              and after every operation.  The runtime must be
       *              compiled in debug mode with the DEBUG_LEGION
       *              macro defined.
       * -lg:disjointness Verify the specified disjointness of 
       *              partitioning operations.  The runtime must be
       *              compiled with the DEBUG_LEGION macro defined.
       * -lg:separate Indicate that separate instances of the high
       *              level runtime should be made for each processor.
       *              The default is one runtime instance per node.
       *              This is primarily useful for debugging purposes
       *              to force messages to be sent between runtime 
       *              instances on the same node.
       * -lg:registration Record the mapping from low-level task IDs to
       *              task variant names for debugging low-level runtime
       *              error messages.
       * -lg:test     Replace the default mapper with the test mapper
       *              which will generate sound but random mapping 
       *              decision in order to stress-test the runtime.
       * -lg:delay <sec> Delay the start of the runtime by 'sec' seconds.
       *              This is often useful for attaching debuggers on 
       *              one or more nodes prior to an application beginning.
       * -------------
       *  Profiling
       * -------------
       * -lg:spy      Enable light-weight logging for Legion Spy which
       *              is valuable for understanding properties of an
       *              application such as the shapes of region trees
       *              and the kinds of tasks/operations that are created.
       *              Checking of the runtime with Legion Spy will still
       *              require the runtime to be compiled with -DLEGION_SPY.
       * -lg:prof <int> Specify the number of nodes on which to enable
       *              profiling information to be collected.  By default
       *              all nodes are disabled. Zero will disable all
       *              profiling while each number greater than zero will
       *              profile on that number of nodes.
       * -lg:serializer <string> Specify the kind of serializer to use:
       *              'ascii' or 'binary'. The default is 'binary'.
       * -lg:prof_logfile <filename> If using a binary serializer the
       *              name of the output file to write to.
       * -lg:prof_footprint <int> The maximum goal size of Legion Prof 
       *              footprint during runtime in MBs. If the total data 
       *              captured by the profiler exceeds this footprint, the 
       *              runtime will begin dumping data out to the output file 
       *              in a minimally invasive way while the application is 
       *              still running. The default is 512 (MB).
       * -lg:prof_latency <int> The goal latency in microseconds of 
       *              intermediate profiling tasks to be writing to output
       *              files if the maximum footprint size is exceeded.
       *              This allows control over the granularity so they
       *              can be made small enough to interleave with other
       *              runtime work. The default is 100 (us).
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
       * use.
       * @param handle the projection ID to register the function at
       * @return ID where the function was registered
       */
      template<LogicalRegion (*PROJ_PTR)(LogicalRegion, const DomainPoint&,
                                         Runtime*)>
      LEGION_DEPRECATED("Projection functions should now be specified "
                        "using projection functor objects")
      static ProjectionID register_region_function(ProjectionID handle);

      /**
       * Register a partition projection function that can be used to
       * map from an upper bound of a logical partition down to a specific
       * logical sub-region for a given domain point during index task
       * execution.  The projection ID zero is reserved for runtime use.
       * @param handle the projection ID to register the function at
       * @return ID where the function was registered
       */
      template<LogicalRegion (*PROJ_PTR)(LogicalPartition, const DomainPoint&,
                                         Runtime*)>
      LEGION_DEPRECATED("Projection functions should now be specified "
                        "using projection functor objects")
      static ProjectionID register_partition_function(ProjectionID handle);
    public:
      /**
       * This call allows the application to add a callback function
       * that will be run prior to beginning any task execution on every
       * runtime in the system.  It can be used to register or update the
       * mapping between mapper IDs and mappers, register reductions,
       * register projection function, register coloring functions, or
       * configure any other static runtime variables prior to beginning
       * the application.
       * @param callback function pointer to the callback function to be run
       */
      static void add_registration_callback(RegistrationCallbackFnptr callback);
      /**
       * @deprecated
       * This call allows the application to register a callback function
       * that will be run prior to beginning any task execution on every
       * runtime in the system.  It can be used to register or update the
       * mapping between mapper IDs and mappers, register reductions,
       * register projection function, register coloring functions, or
       * configure any other static runtime variables prior to beginning
       * the application.
       * @param callback function pointer to the callback function to be run
       */
      LEGION_DEPRECATED("Legion now supports multiple registration callbacks "
                        "added via the add_registration_callback method.") 
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
       * @param vid optional static variant ID
       * @return variant ID for the task
       */
      template<typename T,
        T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, Runtime*)>
      static VariantID preregister_task_variant(
                                    const TaskVariantRegistrar &registrar,
                                    const char *task_name = NULL,
                                    VariantID vid = AUTO_GENERATE_ID);

      /**
       * Statically register a new task variant with the runtime with
       * a non-void return type and userd data prior to the runtime 
       * starting. This call must be made on all nodes and it will 
       * fail if done after the Runtime::start method has been invoked.
       * @param registrar the task variant registrar for describing the task
       * @param user_data the user data to associate with the task variant
       * @param task_name an optional name to assign to the logical task
       * @param vid optional static variant ID
       * @return variant ID for the task
       */
      template<typename T, typename UDT,
        T (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                      Context, Runtime*, const UDT&)>
      static VariantID preregister_task_variant(
                      const TaskVariantRegistrar &registrar, 
                      const UDT &user_data,
                      const char *task_name = NULL,
                      VariantID vid = AUTO_GENERATE_ID);
       
      /**
       * Statically register a new task variant with the runtime with
       * a void return type prior to the runtime starting. This call
       * must be made on all nodes and it will fail if done after the
       * Runtime::start method has been invoked.
       * @param registrar the task variant registrar for describing the task
       * @param an optional name to assign to the logical task
       * @param vid optional static variant ID
       * @return variant ID for the task
       */
      template<
        void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                         Context, Runtime*)>
      static VariantID preregister_task_variant(
                                    const TaskVariantRegistrar &registrar,
                                    const char *task_name = NULL,
                                    VariantID vid = AUTO_GENERATE_ID);

      /**
       * Statically register a new task variant with the runtime with
       * a void return type and user data prior to the runtime starting. 
       * This call must be made on all nodes and it will fail if done 
       * after the Runtime::start method has been invoked.
       * @param registrar the task variant registrar for describing the task
       * @param user_data the user data to associate with the task variant
       * @param an optional name to assign to the logical task
       * @param vid optional static variant ID
       * @return variant ID for the task
       */
      template<typename UDT,
        void (*TASK_PTR)(const Task*, const std::vector<PhysicalRegion>&,
                         Context, Runtime*, const UDT&)>
      static VariantID preregister_task_variant(
              const TaskVariantRegistrar &registrar, const UDT &user_data, 
              const char *task_name = NULL, VariantID vid = AUTO_GENERATE_ID);

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

      /**
       * This is the necessary preamble call to use when registering a 
       * task variant with an explicit CodeDescriptor. It takes the base 
       * Realm task arguments and will return the equivalent Legion task 
       * arguments from the runtime.
       * @param data pointer to the Realm task data
       * @param datalen size of the Realm task data in bytes
       * @param p Realm processor on which the task is running
       * @param task reference to the Task pointer to be set
       * @param regionsptr pointer to the vector of regions reference to set
       * @param ctx the context to set
       * @param runtime the runtime pointer to set
       */
      static void legion_task_preamble(const void *data, size_t datalen,
                                       Processor p, const Task *& task,
                                       const std::vector<PhysicalRegion> *& reg,
                                       Context& ctx, Runtime *& runtime);

      /**
       * This is the necessary postamble call to use when registering a task
       * variant with an explicit CodeDescriptor. It passes back the task
       * return value and completes the task. It should be the last thing
       * called before the task finishes.
       * @param runtime the runtime pointer
       * @param ctx the context for the task
       * @param retvalptr pointer to the return value
       * @param retvalsize the size of the return value in bytes
       */
      static void legion_task_postamble(Runtime *runtime, Context ctx,
                                        const void *retvalptr = NULL,
                                        size_t retvalsize = 0);
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
      LEGION_DEPRECATED("Task registration should be done with "
                        "a TaskVariantRegistrar") 
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
      LEGION_DEPRECATED("Task registration should be done with "
                        "a TaskVariantRegistrar")
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
      LEGION_DEPRECATED("Task registration should be done with "
                        "a TaskVariantRegistrar")
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
      LEGION_DEPRECATED("Task registration should be done with "
                        "a TaskVariantRegistrar")
      static TaskID register_legion_task(TaskID id, Processor::Kind proc_kind,
                                         bool single, bool index,
                                         const UDT &user_data,
                                         VariantID vid = AUTO_GENERATE_ID,
                              TaskConfigOptions options = TaskConfigOptions(),
                                         const char *task_name = NULL);
    public:
      /**
       * Provide a mechanism for finding the Legion runtime
       * pointer for a processor wrapper tasks that are starting
       * a new application level task.
       * @param processor the task will run on
       * @return the Legion runtime pointer for the specified processor
       */
      static Runtime* get_runtime(Processor p = Processor::NO_PROC);

      /**
       * Get the context for the currently executing task this must
       * be called inside of an actual Legion task. Calling it outside
       * of a Legion task will result in undefined behavior
       * @return the context for the enclosing task in which we are executing
       */
      static Context get_context(void);
    private:
      // Helper methods for templates
      IndexSpace create_index_space_internal(Context ctx, const void *realm_is,
                                             TypeTag type_tag);
      IndexPartition create_restricted_partition(Context ctx,
                                      IndexSpace parent,
                                      IndexSpace color_space,
                                      const void *transform,
                                      size_t transform_size,
                                      const void *extent, 
                                      size_t extent_size,
                                      PartitionKind part_kind, Color color);
      IndexSpace create_index_space_union_internal(Context ctx,
                                      IndexPartition parent,
                                      const void *realm_color, TypeTag type_tag,
                                      const std::vector<IndexSpace> &handles);
      IndexSpace create_index_space_union_internal(Context ctx, 
                                      IndexPartition parent, 
                                      const void *realm_color, TypeTag type_tag,
                                      IndexPartition handle);
      IndexSpace create_index_space_intersection_internal(Context ctx,
                                      IndexPartition parent,
                                      const void *realm_color, TypeTag type_tag,
                                      const std::vector<IndexSpace> &handles);
      IndexSpace create_index_space_intersection_internal(Context ctx, 
                                      IndexPartition parent, 
                                      const void *realm_color, TypeTag type_tag,
                                      IndexPartition handle);
      IndexSpace create_index_space_difference_internal(Context ctx,
                                      IndexPartition paretn,
                                      const void *realm_color, TypeTag type_tag,
                                      IndexSpace initial,
                                      const std::vector<IndexSpace> &handles);
      IndexSpace get_index_subspace_internal(IndexPartition handle, 
                                      const void *realm_color,TypeTag type_tag);
      bool has_index_subspace_internal(IndexPartition handle,
                                      const void *realm_color,TypeTag type_tag);
      void get_index_partition_color_space_internal(IndexPartition handle,
                                      void *realm_is, TypeTag type_tag);
      void get_index_space_domain_internal(IndexSpace handle, 
                                      void *realm_is, TypeTag type_tag);
      void get_index_space_color_internal(IndexSpace handle,
                                      void *realm_color, TypeTag type_tag);
      bool safe_cast_internal(Context ctx, LogicalRegion region,
                                      const void *realm_point,TypeTag type_tag);
      LogicalRegion get_logical_subregion_by_color_internal(
                                      LogicalPartition parent,
                                      const void *realm_color,TypeTag type_tag);
      bool has_logical_subregion_by_color_internal(
                                      LogicalPartition parent,
                                      const void *realm_color,TypeTag type_tag);
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
      Future from_value(const void *value, size_t value_size, bool owned);
    private:
      VariantID register_variant(const TaskVariantRegistrar &registrar,bool ret,
                                 const void *user_data, size_t user_data_size,
                                 CodeDescriptor *realm);
      static VariantID preregister_variant(const TaskVariantRegistrar &reg,
                                 const void *user_data, size_t user_data_size,
                                 CodeDescriptor *realm,
                                 bool has_return, const char *task_name,
                                 VariantID vid, bool check_task_id = true);
    private:
      static ReductionOpTable& get_reduction_table(void);
      static SerdezOpTable& get_serdez_table(void);
      static SerdezRedopTable& get_serdez_redop_table(void);
    private:
      friend class Mapper;
      Internal::Runtime *runtime;
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

}; // namespace Legion

#include "legion/legion.inl"
// Include this here so we get the mapper interface in the header file
// We have to put it here though since the mapper interface depends
// on the rest of the legion interface
#include "legion/legion_mapping.h"

#endif // __LEGION_RUNTIME_H__
#endif // defined LEGION_ENABLE_CXX_BINDINGS

// EOF

