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

#ifndef __LEGION_DATA_H__
#define __LEGION_DATA_H__

#include "legion/api/types.h"

namespace Legion {

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
    IndexSpace(DistributedID id, IndexTreeID tid, TypeTag tag);
  public:
    IndexSpace(void);
  public:
    inline bool operator==(const IndexSpace& rhs) const;
    inline bool operator!=(const IndexSpace& rhs) const;
    inline bool operator<(const IndexSpace& rhs) const;
    inline bool operator>(const IndexSpace& rhs) const;
    inline std::size_t hash(void) const;
    inline DistributedID get_id(bool filter = true) const;
    inline IndexTreeID get_tree_id(void) const { return tid; }
    inline bool exists(void) const { return (did != 0); }
    inline TypeTag get_type_tag(void) const { return type_tag; }
    inline int get_dim(void) const;
    bool valid(void) const;
  protected:
    friend std::ostream& operator<<(std::ostream& os, const IndexSpace& is);
    DistributedID did;
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
  private:
    static_assert(DIM > 0, "DIM must be positive");
    static_assert(DIM <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<COORD_T>::value, "must be integral type");
  protected:
    // Only the runtime should be allowed to make these
    FRIEND_ALL_RUNTIME_CLASSES
    IndexSpaceT(DistributedID id, IndexTreeID tid);
  public:
    IndexSpaceT(void);
    explicit IndexSpaceT(const IndexSpace& rhs);
  public:
    inline IndexSpaceT& operator=(const IndexSpace& rhs);
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
    IndexPartition(DistributedID id, IndexTreeID tid, TypeTag tag);
  public:
    IndexPartition(void);
  public:
    inline bool operator==(const IndexPartition& rhs) const;
    inline bool operator!=(const IndexPartition& rhs) const;
    inline bool operator<(const IndexPartition& rhs) const;
    inline bool operator>(const IndexPartition& rhs) const;
    inline std::size_t hash(void) const;
    inline DistributedID get_id(bool filter = true) const;
    inline IndexTreeID get_tree_id(void) const { return tid; }
    inline bool exists(void) const { return (did != 0); }
    inline TypeTag get_type_tag(void) const { return type_tag; }
    inline int get_dim(void) const;
    bool valid(void) const;
  protected:
    friend std::ostream& operator<<(std::ostream& os, const IndexPartition& ip);
    DistributedID did;
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
  private:
    static_assert(DIM > 0, "DIM must be positive");
    static_assert(DIM <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<COORD_T>::value, "must be integral type");
  protected:
    // Only the runtime should be allowed to make these
    FRIEND_ALL_RUNTIME_CLASSES
    IndexPartitionT(DistributedID id, IndexTreeID tid);
  public:
    IndexPartitionT(void);
    explicit IndexPartitionT(const IndexPartition& rhs);
  public:
    inline IndexPartitionT& operator=(const IndexPartition& rhs);
  };

  /**
   * \class FieldSpace
   * Field spaces define the objects used for managing the fields or
   * "columns" of a logical region.  Only the Legion runtime is able
   * to create non-empty field spaces.  Fields within a field space
   * are allocated using field space allocators
   *
   * @see FieldAllocator
   */
  class FieldSpace {
  public:
    static const FieldSpace NO_SPACE; /**< empty field space handle*/
  protected:
    // Only the runtime should be allowed to make these
    FRIEND_ALL_RUNTIME_CLASSES
    FieldSpace(DistributedID id);
  public:
    FieldSpace(void);
  public:
    inline bool operator==(const FieldSpace& rhs) const;
    inline bool operator!=(const FieldSpace& rhs) const;
    inline bool operator<(const FieldSpace& rhs) const;
    inline bool operator>(const FieldSpace& rhs) const;
    inline std::size_t hash(void) const;
    inline DistributedID get_id(bool filter = true) const;
    inline bool exists(void) const { return (did != 0); }
    bool valid(void) const;
  private:
    friend std::ostream& operator<<(std::ostream& os, const FieldSpace& fs);
    DistributedID did;
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
    LogicalRegion(DistributedID tid, IndexSpace index, FieldSpace field);
  public:
    LogicalRegion(void);
  public:
    inline bool operator==(const LogicalRegion& rhs) const;
    inline bool operator!=(const LogicalRegion& rhs) const;
    inline bool operator<(const LogicalRegion& rhs) const;
    std::size_t hash(void) const;
  public:
    inline IndexSpace get_index_space(void) const { return index_space; }
    inline FieldSpace get_field_space(void) const { return field_space; }
    inline RegionTreeID get_tree_id(bool filter = true) const;
    inline bool exists(void) const { return (tree_did != 0); }
    inline TypeTag get_type_tag(void) const
    {
      return index_space.get_type_tag();
    }
    inline int get_dim(void) const { return index_space.get_dim(); }
    bool valid(void) const;
  protected:
    friend std::ostream& operator<<(std::ostream& os, const LogicalRegion& lr);
    // These are private so the user can't just arbitrarily change them
    DistributedID tree_did;
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
  private:
    static_assert(DIM > 0, "DIM must be positive");
    static_assert(DIM <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<COORD_T>::value, "must be integral type");
  protected:
    // Only the runtime should be allowed to make these
    FRIEND_ALL_RUNTIME_CLASSES
    LogicalRegionT(DistributedID tid, IndexSpace index, FieldSpace field);
  public:
    LogicalRegionT(void);
    explicit LogicalRegionT(const LogicalRegion& rhs);
  public:
    inline LogicalRegionT& operator=(const LogicalRegion& rhs);
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
    LogicalPartition(DistributedID tid, IndexPartition pid, FieldSpace field);
  public:
    LogicalPartition(void);
  public:
    inline bool operator==(const LogicalPartition& rhs) const;
    inline bool operator!=(const LogicalPartition& rhs) const;
    inline bool operator<(const LogicalPartition& rhs) const;
    std::size_t hash(void) const;
  public:
    inline IndexPartition get_index_partition(void) const
    {
      return index_partition;
    }
    inline FieldSpace get_field_space(void) const { return field_space; }
    inline RegionTreeID get_tree_id(bool filter = true) const;
    inline bool exists(void) const { return (tree_did != 0); }
    inline TypeTag get_type_tag(void) const
    {
      return index_partition.get_type_tag();
    }
    inline int get_dim(void) const { return index_partition.get_dim(); }
    bool valid(void) const;
  protected:
    friend std::ostream& operator<<(
        std::ostream& os, const LogicalPartition& lp);
    // These are private so the user can't just arbitrary change them
    DistributedID tree_did;
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
  private:
    static_assert(DIM > 0, "DIM must be positive");
    static_assert(DIM <= LEGION_MAX_DIM, "DIM must be <= LEGION_MAX_DIM");
    static_assert(std::is_integral<COORD_T>::value, "must be integral type");
  protected:
    // Only the runtime should be allowed to make these
    FRIEND_ALL_RUNTIME_CLASSES
    LogicalPartitionT(DistributedID tid, IndexPartition pid, FieldSpace field);
  public:
    LogicalPartitionT(void);
    explicit LogicalPartitionT(const LogicalPartition& rhs);
  public:
    inline LogicalPartitionT& operator=(const LogicalPartition& rhs);
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
  class FieldAllocator : public Unserializable {
  public:
    FieldAllocator(void);
    FieldAllocator(const FieldAllocator& allocator);
    FieldAllocator(FieldAllocator&& allocator) noexcept;
    ~FieldAllocator(void);
  protected:
    FRIEND_ALL_RUNTIME_CLASSES
    // Only the Runtime should be able to make these
    FieldAllocator(Internal::FieldAllocatorImpl* impl);
  public:
    FieldAllocator& operator=(const FieldAllocator& allocator);
    FieldAllocator& operator=(FieldAllocator&& allocator) noexcept;
    inline bool operator<(const FieldAllocator& rhs) const;
    inline bool operator==(const FieldAllocator& rhs) const;
    inline bool exists(void) const { return (impl != nullptr); }
  public:
    ///@{
    /**
     * Allocate a field with a given size. Optionally specify
     * the field ID to be assigned.  Note if you use
     * LEGION_AUTO_GENERATE_ID, then all fields for the field space
     * should be generated this way or field names may be
     * deduplicated as the runtime will not check against
     * user assigned field names when generating its own.
     * @param field_size size of the field to be allocated
     * @param desired_fieldid field ID to be assigned to the
     *   field or LEGION_AUTO_GENERATE_ID to specify that the runtime
     *   should assign a fresh field ID
     * @param serdez_id optional parameter for specifying a
     *   custom serdez object for serializing and deserializing
     *   a field when it is moved.
     * @param local_field whether this is a local field or not
     * @param provenance an optional string describing the provenance
     *                   information for this index space
     * @return field ID for the allocated field
     */
    FieldID allocate_field(
        size_t field_size, FieldID desired_fieldid = LEGION_AUTO_GENERATE_ID,
        CustomSerdezID serdez_id = 0, bool local_field = false,
        const char* provenance = nullptr);
    FieldID allocate_field(
        const Future& field_size,
        FieldID desired_fieldid = LEGION_AUTO_GENERATE_ID,
        CustomSerdezID serdez_id = 0, bool local_field = false,
        const char* provenance = nullptr);
    ///@}
    /**
     * Deallocate the specified field from the field space.
     * @param fid the field ID to be deallocated
     * @param unordered set to true if this is performed by a different
     *          thread than the one for the task (e.g a garbage collector)
     * @param provenance an optional string describing the provenance
     *                   information for this index space
     */
    LEGION_DEPRECATED(
        "We are considering removing support for freeing fields"
        "in a future Legion release. Please contact the Legion developer's "
        "list if field deletion is important for your application.")
    void free_field(
        FieldID fid, const bool unordered = false,
        const char* provenance = nullptr);

    /**
     * Same as allocate field, but this field will only
     * be available locally on the node on which it is
     * created and not on remote nodes.  It will then be
     * implicitly destroyed once the task in which it is
     * allocated completes.
     */
    FieldID allocate_local_field(
        size_t field_size, FieldID desired_fieldid = LEGION_AUTO_GENERATE_ID,
        CustomSerdezID serdez_id = 0, const char* provenance = nullptr);
    ///@{
    /**
     * Allocate a collection of fields with the specified sizes.
     * Optionally pass in a set of field IDs to use when allocating
     * the fields otherwise the vector should be empty or the
     * same size as field_sizes with LEGION_AUTO_GENERATE_ID set as the
     * value for each of the resulting_field IDs.  The length of
     * the resulting_fields vector must be less than or equal to
     * the length of field_sizes.  Upon return it will be the same
     * size with field IDs specified for all the allocated fields
     * @param field_sizes size in bytes of the fields to be allocated
     * @param resulting_fields optional field names for allocated fields
     * @param local_fields whether these should be local fields or not
     * @param provenance an optional string describing the provenance
     *                   information for this index space
     * @return resulting_fields vector with length equivalent to
     *    the length of field_sizes with field IDs specified
     */
    void allocate_fields(
        const std::vector<size_t>& field_sizes,
        std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id = 0,
        bool local_fields = false, const char* provenance = nullptr);
    void allocate_fields(
        const std::vector<Future>& field_sizes,
        std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id = 0,
        bool local_fields = false, const char* provenance = nullptr);
    ///@}
    /**
     * Free a collection of field IDs
     * @param to_free set of field IDs to be freed
     * @param unordered set to true if this is performed by a different
     *          thread than the one for the task (e.g a garbage collector)
     * @param provenance an optional string describing the provenance
     *                   information for this index space
     */
    LEGION_DEPRECATED(
        "We are considering removing support for freeing fields"
        "in a future Legion release. Please contact the Legion developer's "
        "list if field deletion is important for your application.")
    void free_fields(
        const std::set<FieldID>& to_free, const bool unordered = false,
        const char* provenance = nullptr);
    /**
     * Same as allocate_fields but the fields that are allocated
     * will only be available locally on the node on which
     * this call is made and not on remote nodes.  The fields
     * will be implicitly destroyed once the task in which
     * they were created completes.
     */
    void allocate_local_fields(
        const std::vector<size_t>& field_sizes,
        std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id = 0,
        const char* provenance = nullptr);
    /**
     * @return field space associated with this allocator
     */
    FieldSpace get_field_space(void) const;
  private:
    Internal::FieldAllocatorImpl* impl;
  };

}  // namespace Legion

#include "legion/api/data.inl"

#endif  // __LEGION_DATA_H__
