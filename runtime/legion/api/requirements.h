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

#ifndef __LEGION_REQUIREMENTS_H__
#define __LEGION_REQUIREMENTS_H__

#include "legion/api/data.h"

namespace Legion {

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
    RegionRequirement(
        LogicalRegion _handle, const std::set<FieldID>& privilege_fields,
        const std::vector<FieldID>& instance_fields, PrivilegeMode _priv,
        CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag = 0,
        bool _verified = false);
    /**
     * Partition region requirement with projection function
     */
    RegionRequirement(
        LogicalPartition pid, ProjectionID _proj,
        const std::set<FieldID>& privilege_fields,
        const std::vector<FieldID>& instance_fields, PrivilegeMode _priv,
        CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag = 0,
        bool _verified = false);
    /**
     * Region requirement with projection function
     */
    RegionRequirement(
        LogicalRegion _handle, ProjectionID _proj,
        const std::set<FieldID>& privilege_fields,
        const std::vector<FieldID>& instance_fields, PrivilegeMode _priv,
        CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag = 0,
        bool _verified = false);
    /**
     * Standard reduction region requirement.  Note no privilege
     * is passed, but instead a reduction operation ID is specified.
     */
    RegionRequirement(
        LogicalRegion _handle, const std::set<FieldID>& privilege_fields,
        const std::vector<FieldID>& instance_fields, ReductionOpID op,
        CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag = 0,
        bool _verified = false);
    /**
     * Partition region requirement for reduction.
     */
    RegionRequirement(
        LogicalPartition pid, ProjectionID _proj,
        const std::set<FieldID>& privilege_fields,
        const std::vector<FieldID>& instance_fields, ReductionOpID op,
        CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag = 0,
        bool _verified = false);
    /**
     * Projection logical region requirement for reduction
     */
    RegionRequirement(
        LogicalRegion _handle, ProjectionID _proj,
        const std::set<FieldID>& privilege_fields,
        const std::vector<FieldID>& instance_fields, ReductionOpID op,
        CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag = 0,
        bool _verified = false);
  public:
    // Analogous constructors without the privilege and instance fields
    RegionRequirement(
        LogicalRegion _handle, PrivilegeMode _priv, CoherenceProperty _prop,
        LogicalRegion _parent, MappingTagID _tag = 0, bool _verified = false);
    RegionRequirement(
        LogicalPartition pid, ProjectionID _proj, PrivilegeMode _priv,
        CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag = 0,
        bool _verified = false);
    RegionRequirement(
        LogicalRegion _handle, ProjectionID _proj, PrivilegeMode _priv,
        CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag = 0,
        bool _verified = false);
    RegionRequirement(
        LogicalRegion _handle, ReductionOpID op, CoherenceProperty _prop,
        LogicalRegion _parent, MappingTagID _tag = 0, bool _verified = false);
    RegionRequirement(
        LogicalPartition pid, ProjectionID _proj, ReductionOpID op,
        CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag = 0,
        bool _verified = false);
    RegionRequirement(
        LogicalRegion _handle, ProjectionID _proj, ReductionOpID op,
        CoherenceProperty _prop, LogicalRegion _parent, MappingTagID _tag = 0,
        bool _verified = false);
  public:
    RegionRequirement(const RegionRequirement& rhs);
    RegionRequirement(RegionRequirement&& rhs) noexcept;
    ~RegionRequirement(void);
    RegionRequirement& operator=(const RegionRequirement& req);
    RegionRequirement& operator=(RegionRequirement&& rhs) noexcept;
  public:
    bool operator==(const RegionRequirement& req) const;
    bool operator<(const RegionRequirement& req) const;
  public:
    /**
     * Method for adding a field to region requirements
     * @param fid field ID to add
     * @param instance indicate whether to add to instance fields
     */
    inline RegionRequirement& add_field(FieldID fid, bool instance = true);
    inline RegionRequirement& add_fields(
        const std::vector<FieldID>& fids, bool instance = true);

    inline RegionRequirement& add_flags(RegionFlags new_flags);
  public:
    inline bool is_verified(void) const
    {
      return (flags & LEGION_VERIFIED_FLAG);
    }
    inline bool is_no_access(void) const
    {
      return (flags & LEGION_NO_ACCESS_FLAG);
    }
    inline bool is_restricted(void) const
    {
      return (flags & LEGION_RESTRICTED_FLAG);
    }
    LEGION_DEPRECATED("Premapping regions is no longer supported.")
    inline bool must_premap(void) const { return false; }
  public:
    const void* get_projection_args(size_t* size) const;
    void set_projection_args(const void* args, size_t size, bool own = false);
  public:
    bool has_field_privilege(FieldID fid) const;
  public:
    // Fields used for controlling task launches
    LogicalRegion region;               /**< mutually exclusive with partition*/
    LogicalPartition partition;         /**< mutually exclusive with region*/
    std::set<FieldID> privilege_fields; /**< unique set of privilege fields*/
    std::vector<FieldID> instance_fields; /**< physical instance fields*/
    PrivilegeMode privilege;              /**< region privilege mode*/
    CoherenceProperty prop;               /**< region coherence mode*/
    LogicalRegion parent; /**< parent region to derive privileges from*/
    ReductionOpID redop;  /**<reduction operation (default 0)*/
    MappingTagID tag;     /**< mapping tag for this region requirement*/
    RegionFlags flags;    /**< optional flags set for region requirements*/
  public:
    ProjectionType handle_type; /**< region or partition requirement*/
    ProjectionID projection;    /**< projection function for index space tasks*/
  public:
    void* projection_args;       /**< projection arguments buffer*/
    size_t projection_args_size; /**< projection arguments buffer size*/
  };

  /**
   * \struct OutputRequirement
   * Output requirements are a special kind of region requirement to inform
   * the runtime that the task will be producing new instances as part of its
   * execution that will be attached to the logical region at the end of the
   * task, and are therefore not mapped ahead of the task's execution.
   *
   * Output region requirements come in two flavors: those that are already
   * bounded region requirements and those which are going to produce variable
   * sized outputs. Bounded region requirements behave like normal region
   * requirements except they will not be mapped by the task. Alternatively,
   * for variable-sized output region requirements the runtime
   * will create fresh region and partition names for output requirements
   * right after the task is launched. Output requirements still pick
   * field IDs and the field space for the output regions.
   *
   * In case of individual task launch, the dimension of an output region
   * is chosen by the `dim` argument to the output requirement , and
   * and no partitions will be created by the runtime. For index space
   * launches, the runtime gives back a fresh region and partition,
   * whose construction is controlled by the indexing mode specified
   * the output requirement:
   *
   * 0) For either indexing mode, the output partition is always a disjoint
   *    complete partition. The color space of the partition is identical to
   *    to the launch domain by default, but must be explicitly specified
   *    if the output requirement uses a non-identity projection functor.
   *    (see `set_projection`) Any projection functor associated with an
   *    output requirement must be bijective.
   *
   * 1) When the global indexing is requested, the dimension of the output
   *    region must be the same as the color space. The index space is
   *    constructed such that the extent of each dimension is a sum of
   *    that dimension's extents of the outputs produced by point tasks;
   *    i.e., the range of the i-th subregion on dimension k
   *    is [S, S+n), where S is the sum of the previous i-1 subregions'
   *    extents on the k dimension and n is the extent of the output of
   *    the i-th point task on the k dimension. Outputs are well-formed
   *    only when their extents are aligned with their neighbors'. For
   *    example, outputs of extents (3, 4) and (5, 4), respectively,
   *    are valid if the producers' points are (0, 0) and (1, 0),
   *    respectively, whereas they are not well-formed if the colors
   *    are (0, 0) and (0, 1); for the former, the bounds of the output
   *    subregions are ([0, 2], [0, 3]) and ([3, 7], [0, 3]),
   *    respectively.
   *
   * 2) With the local indexing, the output region has an (N+k)-D index
   *    space for an N-D launch domain, where k is the dimension chosen
   *    by the output requirement. The range of the subregion produced
   *    by the point task p (where p is a point in an N-D space) is
   *    [<p,lo>, <p,hi>] where [lo, hi] is the bounds of the point task p
   *    and <v1,v2> denotes a concatenation of points v1 and v2.
   *    The root index space is simply a union of all subspaces.
   *
   * 3) In the case of local indexing, the output region can either have a
   *    "loose" convex hull parent index space or a "tight" index space that
   *    contains exactly the points in the child space. With the convex hull,
   *    the runtime computes an upper bound rectangle with as many rows as
   *    children and as many columns as the extent of the larges child space.
   *    If convex_hull is set to false, the runtime will compute a more
   *    expensive sparse index space containing exactly the children points.
   *
   * Note that the global indexing has performance consequences since
   * the runtime needs to perform a global prefix sum to compute the ranges
   * of subspaces. Similarly the "tight" bounds can be expensive to compute
   * due to the cost of building the sparsity data structure.
   *
   */
  struct OutputRequirement : public RegionRequirement {
  public:
    OutputRequirement(bool bounded = false);
    OutputRequirement(const RegionRequirement& req);
    OutputRequirement(
        FieldSpace field_space, const std::set<FieldID>& fields, int dim = 1,
        bool global_indexing = false);
  public:
    OutputRequirement(const OutputRequirement& rhs);
    ~OutputRequirement(void);
    OutputRequirement& operator=(const RegionRequirement& req);
    OutputRequirement& operator=(const OutputRequirement& req);
  public:
    bool operator==(const OutputRequirement& req) const;
    bool operator<(const OutputRequirement& req) const;
  public:
    template<int DIM, typename COORD_T>
    void set_type_tag();
    // Specifies a projection functor id for this requirement.
    // For a projection output requirement, a color space must be specified.
    // The projection functor must be a bijective mapping from the launch
    // domain to the color space. This implies that the launch domain's
    // volume must be the same as the color space's.
    void set_projection(ProjectionID projection, IndexSpace color_space);
  public:
    TypeTag type_tag;
    FieldSpace field_space;   /**< field space for the output region */
    bool global_indexing;     /**< global indexing is used when true */
    bool bounded_requirement; /**< indicate requirement is bounded */
    IndexSpace color_space;   /**< color space for the output partition */
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
    IndexSpace handle;
    AllocateMode privilege;
    IndexSpace parent;
    bool verified;
  public:
    IndexSpaceRequirement(void);
    IndexSpaceRequirement(
        IndexSpace _handle, AllocateMode _priv, IndexSpace _parent,
        bool _verified = false);
  public:
    bool operator<(const IndexSpaceRequirement& req) const;
    bool operator==(const IndexSpaceRequirement& req) const;
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
    FieldSpace handle;
    AllocateMode privilege;
    bool verified;
  public:
    FieldSpaceRequirement(void);
    FieldSpaceRequirement(
        FieldSpace _handle, AllocateMode _priv, bool _verified = false);
  public:
    bool operator<(const FieldSpaceRequirement& req) const;
    bool operator==(const FieldSpaceRequirement& req) const;
  };

}  // namespace Legion

#include "legion/api/requirements.inl"

#endif  // __LEGION_REQUIREMENTS_H__
