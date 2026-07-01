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

#ifndef __LEGION_RUNTIME_H__
#define __LEGION_RUNTIME_H__

#include "legion/api/buffers.h"
#include "legion/api/functors.h"
#include "legion/api/future.h"
#include "legion/api/launchers.h"
#include "legion/api/registrars.h"
#include "legion/api/transforms.h"
#include "legion/api/values.h"

#define LEGION_PRINT_ONCE(runtime, ctx, file, fmt, ...) \
  {                                                     \
    char message[4096];                                 \
    snprintf(message, 4096, fmt, ##__VA_ARGS__);        \
    runtime->print_once(ctx, file, message);            \
  }

namespace Legion {

  /**
   * \struct InputArgs
   * Input arguments helper struct for passing in
   * the command line arguments to the runtime.
   */
  struct InputArgs {
  public:
    char** argv;
    int argc;
  };

  /**
   * \struct RegistrationCallbackArgs
   * A struct containing arguments for a registration callback
   */
  struct RegistrationCallbackArgs {
    Machine machine;
    Runtime* runtime;
    std::set<Processor> local_procs;
    UntypedBuffer buffer;
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
    TaskConfigOptions(
        bool leaf = false, bool inner = false, bool idempotent = false);
  public:
    bool leaf;
    bool inner;
    bool idempotent;
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
    Runtime(Internal::Runtime* rt);
  public:
    //------------------------------------------------------------------------
    // Index Space Operations
    //------------------------------------------------------------------------
    ///@{
    /**
     * Create a new top-level index space based on the given domain bounds
     * If the bounds contains a Realm index space then Legion will take
     * ownership of any sparsity maps.
     * @param ctx the enclosing task context
     * @param bounds the bounds for the new index space
     * @param type_tag optional type tag to use for the index space
     * @param provenance an optional string describing the provenance
     *                   information for this index space
     * @param take_ownership whether Legion should take ownership of the
     *                       sparsity map or not, if not then Legion will
     *                       add its own reference
     * @return the handle for the new index space
     */
    IndexSpace create_index_space(
        Context ctx, const Domain& bounds, TypeTag type_tag = 0,
        const char* provenance = nullptr, const bool take_ownership = false);
    // Template version
    template<int DIM, typename COORD_T>
    IndexSpaceT<DIM, COORD_T> create_index_space(
        Context ctx, const Rect<DIM, COORD_T>& bounds,
        const char* provenance = nullptr);
    template<int DIM, typename COORD_T>
    IndexSpaceT<DIM, COORD_T> create_index_space(
        Context ctx, const DomainT<DIM, COORD_T>& bounds,
        const char* provenance = nullptr, const bool take_ownership = false);
    ///@}
    ///@{
    /**
     * Create a new top-level index space from a future which contains
     * a Domain object. If the Domain conaints a Realm index space then
     * Legion will take ownership of any sparsity maps.
     * @param ctx the enclosing task context
     * @param dimensions number of dimensions for the created space
     * @param future the future value containing the bounds
     * @param type_tag optional type tag to use for the index space
     *                 defaults to 'coord_t'
     * @param provenance an optional string describing the provenance
     *                   information for this index space
     * @return the handle for the new index space
     */
    IndexSpace create_index_space(
        Context ctx, size_t dimensions, const Future& f, TypeTag type_tag = 0,
        const char* provenance = nullptr);
    template<int DIM, typename COORD_T>
    IndexSpaceT<DIM, COORD_T> create_index_space(
        Context ctx, const Future& f, const char* provenance = nullptr);
    ///@}
    ///@{
    /**
     * Create a new top-level index space from a vector of points
     * @param ctx the enclosing task context
     * @param points a vector of points to have in the index space
     * @param provenance an optional string describing the provenance
     *                   information for this index space
     * @return the handle for the new index space
     */
    IndexSpace create_index_space(
        Context ctx, const std::vector<DomainPoint>& points,
        const char* provenance = nullptr);
    // Template version
    template<int DIM, typename COORD_T>
    IndexSpaceT<DIM, COORD_T> create_index_space(
        Context ctx, const std::vector<Point<DIM, COORD_T> >& points,
        const char* provenance = nullptr);
    ///@}
    ///@{
    /**
     * Create a new top-level index space from a vector of rectangles
     * @param ctx the enclosing task context
     * @param rects a vector of rectangles to have in the index space
     * @param provenance an optional string describing the provenance
     *                   information for this index space
     * @return the handle for the new index space
     */
    IndexSpace create_index_space(
        Context ctx, const std::vector<Domain>& rects,
        const char* provenance = nullptr);
    // Template version
    template<int DIM, typename COORD_T>
    IndexSpaceT<DIM, COORD_T> create_index_space(
        Context ctx, const std::vector<Rect<DIM, COORD_T> >& rects,
        const char* provenance = nullptr);
    ///@}
    ///@{
    /**
     * Create a new top-level index space by unioning together
     * several existing index spaces
     * @param ctx the enclosing task context
     * @param spaces the index spaces to union together
     * @param provenance an optional string describing the provenance
     *                   information for this index space
     * @return the handle for the new index space
     */
    IndexSpace union_index_spaces(
        Context ctx, const std::vector<IndexSpace>& spaces,
        const char* provenance = nullptr);
    // Template version
    template<int DIM, typename COORD_T>
    IndexSpaceT<DIM, COORD_T> union_index_spaces(
        Context ctx, const std::vector<IndexSpaceT<DIM, COORD_T> >& spaces,
        const char* provenance = nullptr);
    ///@}
    ///@{
    /**
     * Create a new top-level index space by intersecting
     * several existing index spaces
     * @param ctx the enclosing task context
     * @param spaces the index spaces to intersect
     * @param provenance an optional string describing the provenance
     *                   information for this index space
     * @return the handle for the new index space
     */
    IndexSpace intersect_index_spaces(
        Context ctx, const std::vector<IndexSpace>& spaces,
        const char* provenance = nullptr);
    // Template version
    template<int DIM, typename COORD_T>
    IndexSpaceT<DIM, COORD_T> intersect_index_spaces(
        Context ctx, const std::vector<IndexSpaceT<DIM, COORD_T> >& spaces,
        const char* provenance = nullptr);
    ///@}
    ///@{
    /**
     * Create a new top-level index space by taking the
     * set difference of two different index spaces
     * @param provenance an optional string describing the provenance
     *                   information for this index space
     */
    IndexSpace subtract_index_spaces(
        Context ctx, IndexSpace left, IndexSpace right,
        const char* provenance = nullptr);
    // Template version
    template<int DIM, typename COORD_T>
    IndexSpaceT<DIM, COORD_T> subtract_index_spaces(
        Context ctx, IndexSpaceT<DIM, COORD_T> left,
        IndexSpaceT<DIM, COORD_T> right, const char* provenance = nullptr);
    ///@}
    /**
     * @deprecated
     * Create a new top-level index space with the maximum number of elements
     * @param ctx the enclosing task context
     * @param max_num_elmts maximum number of elements in the index space
     * @return the handle for the new index space
     */
    LEGION_DEPRECATED(
        "Use the new index space creation routines with a "
        "single domain or rectangle.")
    IndexSpace create_index_space(Context ctx, size_t max_num_elmts);
    /**
     * @deprecated
     * Create a new top-level index space based on a set of domains
     * @param ctx the enclosing task context
     * @param domains the set of domains
     * @return the handle for the new index space
     */
    LEGION_DEPRECATED(
        "Use the new index space creation routines with a "
        "single domain or rectangle.")
    IndexSpace create_index_space(Context ctx, const std::set<Domain>& domains);
    /**
     * Create a new shared ownership of a top-level index space to prevent it
     * from being destroyed by other potential owners. Every call to this
     * method that succeeds must be matched with a corresponding call
     * to destroy the index space in order for the index space to
     * actually be deleted. The index space must not have been destroyed
     * prior to this call being performed.
     * @param ctx the enclosing task context
     * @param handle for top-level index space to request ownership for
     */
    void create_shared_ownership(Context ctx, IndexSpace handle);
    /**
     * Destroy an existing index space
     * @param ctx the enclosing task context
     * @param handle the index space to destroy
     * @param unordered set to true if this is performed by a different
     *          thread than the one for the task (e.g a garbage collector)
     * @param recurse delete the full index tree
     * @param provenance an optional string describing the provenance
     *                   information for this index space
     */
    void destroy_index_space(
        Context ctx, IndexSpace handle, const bool unordered = false,
        const bool recurse = true, const char* provenance = nullptr);
  public:
    /**
     * Create a new shared ownership of an index partition to prevent it
     * from being destroyed by other potential owners. Every call to this
     * method that succeeds must be matched with a corresponding call
     * to destroy the index partition in order for the index partition to
     * actually be deleted. The index partition must not have been destroyed
     * prior to this call being performed.
     * @param ctx the enclosing task context
     * @param handle for index partition to request ownership for
     */
    void create_shared_ownership(Context ctx, IndexPartition handle);
    /**
     * Destroy an index partition
     * @param ctx the enclosing task context
     * @param handle index partition to be destroyed
     * @param unordered set to true if this is performed by a different
     *          thread than the one for the task (e.g a garbage collector)
     * @param recurse destroy the full sub-tree below this partition
     * @param provenance an optional string describing the provenance
     *                   information for this index space
     */
    void destroy_index_partition(
        Context ctx, IndexPartition handle, const bool unordered = false,
        const bool recurse = true, const char* provenance = nullptr);
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
     * can specify an optional color for the index partition. Note: for
     * multi-dimensional cases, this implementation will currently only
     * split across the first dimension. This is useful for providing an
     * initial equal partition, but is unlikely to be an ideal partition
     * for long repetitive use. Do NOT rely on this behavior as the runtime
     * reserves the right to change the implementation in the future.
     * @param ctx the enclosing task context
     * @param parent index space of the partition to be made
     * @param color_space space of colors to create
     * @param granularity the minimum size of the index subspaces
     * @param color optional color paramter for the partition
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return name of the created index partition
     */
    IndexPartition create_equal_partition(
        Context ctx, IndexSpace parent, IndexSpace color_space,
        size_t granularity = 1, Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexPartitionT<DIM, COORD_T> create_equal_partition(
        Context ctx, IndexSpaceT<DIM, COORD_T> parent,
        IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
        size_t granularity = 1, Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    ///@}
    ///@{
    /**
     * Create 'color_space' index spaces (one for each point) to partition
     * the parent 'parent' index space using the 'weights' to proportionally
     * size the resulting subspaces. By definition the resulting partition
     * will be disjoint. Users can also specify a minimum 'granularity' for
     * the size of the index subspaces. Users can specify an optional
     * 'color' for the name of the created index partition.
     * @param ctx the enclosing task context
     * @param parent index space of the partition to be made
     * @param weights per-color weights for sizing output regions
     * @param color_space space of the colors to create
     * @param granularity the minimum size of the index subspaces
     * @param color optional color parameter for the partition
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return name of the created index partition
     */
    IndexPartition create_partition_by_weights(
        Context ctx, IndexSpace parent,
        const std::map<DomainPoint, int>& weights, IndexSpace color_space,
        size_t granularity = 1, Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexPartitionT<DIM, COORD_T> create_partition_by_weights(
        Context ctx, IndexSpaceT<DIM, COORD_T> parent,
        const std::map<Point<COLOR_DIM, COLOR_COORD_T>, int>& weights,
        IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
        size_t granularity = 1, Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    // 64-bit versions
    IndexPartition create_partition_by_weights(
        Context ctx, IndexSpace parent,
        const std::map<DomainPoint, size_t>& weights, IndexSpace color_space,
        size_t granularity = 1, Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexPartitionT<DIM, COORD_T> create_partition_by_weights(
        Context ctx, IndexSpaceT<DIM, COORD_T> parent,
        const std::map<Point<COLOR_DIM, COLOR_COORD_T>, size_t>& weights,
        IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
        size_t granularity = 1, Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    // Alternate versions of the above method that take a future map where
    // the values in the future map will be interpretted as integer weights
    // You can use this method with both 32 and 64 bit weights
    IndexPartition create_partition_by_weights(
        Context ctx, IndexSpace parent, const FutureMap& weights,
        IndexSpace color_space, size_t granularity = 1,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexPartitionT<DIM, COORD_T> create_partition_by_weights(
        Context ctx, IndexSpaceT<DIM, COORD_T> parent, const FutureMap& weights,
        IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
        size_t granularity = 1, Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
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
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return name of the created index partition
     */
    IndexPartition create_partition_by_union(
        Context ctx, IndexSpace parent, IndexPartition handle1,
        IndexPartition handle2, IndexSpace color_space,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexPartitionT<DIM, COORD_T> create_partition_by_union(
        Context ctx, IndexSpaceT<DIM, COORD_T> parent,
        IndexPartitionT<DIM, COORD_T> handle1,
        IndexPartitionT<DIM, COORD_T> handle2,
        IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
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
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return name of the created index partition
     */
    IndexPartition create_partition_by_intersection(
        Context ctx, IndexSpace parent, IndexPartition handle1,
        IndexPartition handle2, IndexSpace color_space,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexPartitionT<DIM, COORD_T> create_partition_by_intersection(
        Context ctx, IndexSpaceT<DIM, COORD_T> parent,
        IndexPartitionT<DIM, COORD_T> handle1,
        IndexPartitionT<DIM, COORD_T> handle2,
        IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    ///@}
    ///@{
    /**
     * This version of create partition by intersection will intersect an
     * existing partition with a parent index space in order to generate
     * a new partition where each subregion is the intersection of the
     * parent with the corresponding subregion in the original partition.
     * We require that the partition and the parent index space both have
     * the same dimensionality and coordinate type, but they can be
     * otherwise unrelated. The application can also optionally indicate
     * that the parent will dominate all the subregions in the partition
     * which will allow the runtime to elide the intersection test and
     * turn this into a partition copy operation.
     * @param ctx the enclosing task context
     * @param parent the new parent index space for the mirrored partition
     * @param partition the partition to mirror
     * @param part_kind optinally specify the completenss of the partition
     * @param color optional new color for the mirrored partition
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @param dominates whether the parent dominates the partition
     */
    IndexPartition create_partition_by_intersection(
        Context ctx, IndexSpace parent, IndexPartition partition,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID, bool dominates = false,
        const char* provenance = nullptr);
    template<int DIM, typename COORD_T>
    IndexPartitionT<DIM, COORD_T> create_partition_by_intersection(
        Context ctx, IndexSpaceT<DIM, COORD_T> parent,
        IndexPartitionT<DIM, COORD_T> partition,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID, bool dominates = false,
        const char* provenance = nullptr);
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
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return name of the created index partition
     */
    IndexPartition create_partition_by_difference(
        Context ctx, IndexSpace parent, IndexPartition handle1,
        IndexPartition handle2, IndexSpace color_space,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexPartitionT<DIM, COORD_T> create_partition_by_difference(
        Context ctx, IndexSpaceT<DIM, COORD_T> parent,
        IndexPartitionT<DIM, COORD_T> handle1,
        IndexPartitionT<DIM, COORD_T> handle2,
        IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
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
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return the color used for each of the partitions
     */
    Color create_cross_product_partitions(
        Context ctx, IndexPartition handle1, IndexPartition handle2,
        std::map<IndexSpace, IndexPartition>& handles,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    Color create_cross_product_partitions(
        Context ctx, IndexPartitionT<DIM, COORD_T> handle1,
        IndexPartitionT<DIM, COORD_T> handle2,
        typename std::map<
            IndexSpaceT<DIM, COORD_T>, IndexPartitionT<DIM, COORD_T> >& handles,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
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
     * @param map_arg an untyped buffer for the mapper data of the Partition
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     */
    void create_association(
        Context ctx, LogicalRegion domain, LogicalRegion domain_parent,
        FieldID domain_fid, IndexSpace range, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = nullptr);
    void create_bidirectional_association(
        Context ctx, LogicalRegion domain, LogicalRegion domain_parent,
        FieldID domain_fid, LogicalRegion range, LogicalRegion range_parent,
        FieldID range_fid, MapperID id = 0, MappingTagID tag = 0,
        UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = nullptr);
    // Template versions
    template<int DIM1, typename COORD_T1, int DIM2, typename COORD_T2>
    void create_association(
        Context ctx, LogicalRegionT<DIM1, COORD_T1> domain,
        LogicalRegionT<DIM1, COORD_T1> domain_parent,
        FieldID domain_fid,  // type: Point<DIM2,COORD_T2>
        IndexSpaceT<DIM2, COORD_T2> range, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = nullptr);
    template<int DIM1, typename COORD_T1, int DIM2, typename COORD_T2>
    void create_bidirectional_association(
        Context ctx, LogicalRegionT<DIM1, COORD_T1> domain,
        LogicalRegionT<DIM1, COORD_T1> domain_parent,
        FieldID domain_fid,  // type: Point<DIM2,COORD_T2>
        LogicalRegionT<DIM2, COORD_T2> range,
        LogicalRegionT<DIM2, COORD_T2> range_parent,
        FieldID range_fid,  // type: Point<DIM1,COORD_T1>
        MapperID id = 0, MappingTagID tag = 0,
        UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = nullptr);
    ///@}
    ///@{
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
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return a new index partition of the parent index space
     */
    IndexPartition create_partition_by_restriction(
        Context ctx, IndexSpace parent, IndexSpace color_space,
        DomainTransform transform, Domain extent,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    // Template version
    template<int DIM, int COLOR_DIM, typename COORD_T>
    IndexPartitionT<DIM, COORD_T> create_partition_by_restriction(
        Context ctx, IndexSpaceT<DIM, COORD_T> parent,
        IndexSpaceT<COLOR_DIM, COORD_T> color_space,
        Transform<DIM, COLOR_DIM, COORD_T> transform, Rect<DIM, COORD_T> extent,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    ///@}
    ///@{
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
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return a new index partition of the parent index space
     */
    IndexPartition create_partition_by_blockify(
        Context ctx, IndexSpace parent, DomainPoint blocking_factor,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    // Template version
    template<int DIM, typename COORD_T>
    IndexPartitionT<DIM, COORD_T> create_partition_by_blockify(
        Context ctx, IndexSpaceT<DIM, COORD_T> parent,
        Point<DIM, COORD_T> blocking_factor,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    /**
     * An alternate version of create partition by blockify that also
     * takes an origin to use for the computation of the extent.
     * @param ctx the enclosing task context
     * @param parent the parent index space to be partitioned
     * @param blocking factor the blocking factors for each dimension
     * @param origin the origin to use for computing the extent
     * @param color optional new color for the index partition
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return a new index partition of the parent index space
     */
    IndexPartition create_partition_by_blockify(
        Context ctx, IndexSpace parent, DomainPoint blockify_factor,
        DomainPoint origin, Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    // Template version
    template<int DIM, typename COORD_T>
    IndexPartitionT<DIM, COORD_T> create_partition_by_blockify(
        Context ctx, IndexSpaceT<DIM, COORD_T> parent,
        Point<DIM, COORD_T> blocking_factor, Point<DIM, COORD_T> origin,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    ///@}
    ///@{
    /**
     * Create partition by domain allows users to specify an explicit
     * Domain object to use for one or more subregions directly.
     * This is similar to the old (deprecated) coloring APIs.
     * However, instead of specifying colors for each element, we
     * encourage users to create domains that express as few dense
     * rectangles in them as necessary for expressing the index space.
     * The runtime will not attempt to coalesce the rectangles in
     * each domain further.
     * @param ctx the enclosing task context
     * @param parent the parent index space to be partitioned
     * @param domains map of points in the color space points to domains
     * @param color_space the color space for the partition
     * @param perform_intersections intersect domains with parent space
     * @param part_kind specify the partition kind or ask to compute it
     * @param color the color of the result of the partition
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @param take_ownership whether Legion should take ownership of the
     *                       domains or not
     * @return a new index partition of the parent index space
     */
    IndexPartition create_partition_by_domain(
        Context ctx, IndexSpace parent,
        const std::map<DomainPoint, Domain>& domains, IndexSpace color_space,
        bool perform_intersections = true,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID, const char* provenance = nullptr,
        bool take_ownership = false);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexPartitionT<DIM, COORD_T> create_partition_by_domain(
        Context ctx, IndexSpaceT<DIM, COORD_T> parent,
        const std::map<Point<COLOR_DIM, COLOR_COORD_T>, DomainT<DIM, COORD_T> >&
            domains,
        IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
        bool perform_intersections = true,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID, const char* provenance = nullptr,
        bool take_ownership = false);
    /**
     * This is an alternate version of create_partition_by_domain that
     * instead takes a future map for the list of domains to be used.
     * The runtime will automatically interpret the results in the
     * individual futures as domains for creating the partition. This
     * allows users to create this partition without blocking.
     * @param ctx the enclosing task context
     * @param parent the parent index space to be partitioned
     * @param domains future map of points to domains
     * @param color_space the color space for the partition
     * @param perform_intersections intersect domains with parent space
     * @param part_kind specify the partition kind or ask to compute it
     * @param color the color of the result of the partition
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return a new index partition of the parent index space
     */
    IndexPartition create_partition_by_domain(
        Context ctx, IndexSpace parent, const FutureMap& domain_future_map,
        IndexSpace color_space, bool perform_intersections = true,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexPartitionT<DIM, COORD_T> create_partition_by_domain(
        Context ctx, IndexSpaceT<DIM, COORD_T> parent,
        const FutureMap& domain_future_map,
        IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
        bool perform_intersections = true,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    ///@}
    ///@{
    /**
     * Create partition by rectangles is a special case of partition by domain
     * that will create a partition from a list of rectangles supplied for
     * each point in the color space.
     * @param ctx the enclosing task context
     * @param parent the parent index space to be partitioned
     * @param rectangles map of rectangle lists for each point
     * @param color_space the color space for the partition
     * @param perform_intersections intersect domains with parent space
     * @param part_kind specify the partition kind or ask to compute it
     * @param color the color of the result of the partition
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @param collective whether shards from a control replicated context
     *                   should work collectively to construct the map
     * @return a new index partition of the parent index space
     */
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexPartitionT<DIM, COORD_T> create_partition_by_rectangles(
        Context ctx, IndexSpaceT<DIM, COORD_T> parent,
        const std::map<
            Point<COLOR_DIM, COLOR_COORD_T>, std::vector<Rect<DIM, COORD_T> > >&
            rectangles,
        IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
        bool perform_intersections = true,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID, const char* provenance = nullptr,
        bool collective = false);
    ///@}
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
     * @param part_kind the kind of the partition
     * @param map_arg an untyped buffer for the mapper data of the Partition
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return a new index partition of the index space of the logical region
     */
    IndexPartition create_partition_by_field(
        Context ctx, LogicalRegion handle, LogicalRegion parent, FieldID fid,
        IndexSpace color_space, Color color = LEGION_AUTO_GENERATE_ID,
        MapperID id = 0, MappingTagID tag = 0,
        PartitionKind part_kind = LEGION_DISJOINT_KIND,
        UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = nullptr);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexPartitionT<DIM, COORD_T> create_partition_by_field(
        Context ctx, LogicalRegionT<DIM, COORD_T> handle,
        LogicalRegionT<DIM, COORD_T> parent,
        FieldID fid,  // type: Point<COLOR_DIM,COLOR_COORD_T>
        IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
        Color color = LEGION_AUTO_GENERATE_ID, MapperID id = 0,
        MappingTagID tag = 0, PartitionKind part_kind = LEGION_DISJOINT_KIND,
        UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = nullptr);
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
     * @param map_arg an untyped buffer for the mapper data of the Partition
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return a new index partition of the 'handle' index space
     */
    IndexPartition create_partition_by_image(
        Context ctx, IndexSpace handle, LogicalPartition projection,
        LogicalRegion parent, FieldID fid, IndexSpace color_space,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = nullptr);
    template<
        int DIM1, typename COORD_T1, int DIM2, typename COORD_T2, int COLOR_DIM,
        typename COLOR_COORD_T>
    IndexPartitionT<DIM2, COORD_T2> create_partition_by_image(
        Context ctx, IndexSpaceT<DIM2, COORD_T2> handle,
        LogicalPartitionT<DIM1, COORD_T1> projection,
        LogicalRegionT<DIM1, COORD_T1> parent,
        FieldID fid,  // type: Point<DIM2,COORD_T2>
        IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = nullptr);
    // Range versions of image
    IndexPartition create_partition_by_image_range(
        Context ctx, IndexSpace handle, LogicalPartition projection,
        LogicalRegion parent, FieldID fid, IndexSpace color_space,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = nullptr);
    template<
        int DIM1, typename COORD_T1, int DIM2, typename COORD_T2, int COLOR_DIM,
        typename COLOR_COORD_T>
    IndexPartitionT<DIM2, COORD_T2> create_partition_by_image_range(
        Context ctx, IndexSpaceT<DIM2, COORD_T2> handle,
        LogicalPartitionT<DIM1, COORD_T1> projection,
        LogicalRegionT<DIM1, COORD_T1> parent,
        FieldID fid,  // type: Rect<DIM2,COORD_T2>
        IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = nullptr);
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
     * @param map_arg an untyped buffer for the mapper data of the Partition
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return a new index partition of the index space of 'handle'
     */
    IndexPartition create_partition_by_preimage(
        Context ctx, IndexPartition projection, LogicalRegion handle,
        LogicalRegion parent, FieldID fid, IndexSpace color_space,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = nullptr);
    template<
        int DIM1, typename COORD_T1, int DIM2, typename COORD_T2, int COLOR_DIM,
        typename COLOR_COORD_T>
    IndexPartitionT<DIM1, COORD_T1> create_partition_by_preimage(
        Context ctx, IndexPartitionT<DIM2, COORD_T2> projection,
        LogicalRegionT<DIM1, COORD_T1> handle,
        LogicalRegionT<DIM1, COORD_T1> parent,
        FieldID fid,  // type: Point<DIM2,COORD_T2>
        IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = nullptr);
    // Range versions of preimage
    IndexPartition create_partition_by_preimage_range(
        Context ctx, IndexPartition projection, LogicalRegion handle,
        LogicalRegion parent, FieldID fid, IndexSpace color_space,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = nullptr);
    template<
        int DIM1, typename COORD_T1, int DIM2, typename COORD_T2, int COLOR_DIM,
        typename COLOR_COORD_T>
    IndexPartitionT<DIM1, COORD_T1> create_partition_by_preimage_range(
        Context ctx, IndexPartitionT<DIM2, COORD_T2> projection,
        LogicalRegionT<DIM1, COORD_T1> handle,
        LogicalRegionT<DIM1, COORD_T1> parent,
        FieldID fid,  // type: Rect<DIM2,COORD_T2>
        IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = nullptr);
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
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return handle of the new index partition
     */
    IndexPartition create_pending_partition(
        Context ctx, IndexSpace parent, IndexSpace color_space,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexPartitionT<DIM, COORD_T> create_pending_partition(
        Context ctx, IndexSpaceT<DIM, COORD_T> parent,
        IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
        PartitionKind part_kind = LEGION_COMPUTE_KIND,
        Color color = LEGION_AUTO_GENERATE_ID,
        const char* provenance = nullptr);
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
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return handle of the new index space
     */
    IndexSpace create_index_space_union(
        Context ctx, IndexPartition parent, const DomainPoint& color,
        const std::vector<IndexSpace>& handles,
        const char* provenance = nullptr);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexSpaceT<DIM, COORD_T> create_index_space_union(
        Context ctx, IndexPartitionT<DIM, COORD_T> parent,
        Point<COLOR_DIM, COLOR_COORD_T> color,
        const typename std::vector<IndexSpaceT<DIM, COORD_T> >& handles,
        const char* provenance = nullptr);
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
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return handle of the new index space
     */
    IndexSpace create_index_space_union(
        Context ctx, IndexPartition parent, const DomainPoint& color,
        IndexPartition handle, const char* provenance = nullptr);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexSpaceT<DIM, COORD_T> create_index_space_union(
        Context ctx, IndexPartitionT<DIM, COORD_T> parent,
        Point<COLOR_DIM, COLOR_COORD_T> color,
        IndexPartitionT<DIM, COORD_T> handle, const char* provenance = nullptr);
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
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return handle of the new index space
     */
    IndexSpace create_index_space_intersection(
        Context ctx, IndexPartition parent, const DomainPoint& color,
        const std::vector<IndexSpace>& handles,
        const char* provenance = nullptr);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexSpaceT<DIM, COORD_T> create_index_space_intersection(
        Context ctx, IndexPartitionT<DIM, COORD_T> parent,
        Point<COLOR_DIM, COLOR_COORD_T> color,
        const typename std::vector<IndexSpaceT<DIM, COORD_T> >& handles,
        const char* provenance = nullptr);
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
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return handle of the new index space
     */
    IndexSpace create_index_space_intersection(
        Context ctx, IndexPartition parent, const DomainPoint& color,
        IndexPartition handle, const char* provenannce = nullptr);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexSpaceT<DIM, COORD_T> create_index_space_intersection(
        Context ctx, IndexPartitionT<DIM, COORD_T> parent,
        Point<COLOR_DIM, COLOR_COORD_T> color,
        IndexPartitionT<DIM, COORD_T> handle, const char* provenance = nullptr);
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
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return handle of the new index space
     */
    IndexSpace create_index_space_difference(
        Context ctx, IndexPartition parent, const DomainPoint& color,
        IndexSpace initial, const std::vector<IndexSpace>& handles,
        const char* provenancne = nullptr);
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexSpaceT<DIM, COORD_T> create_index_space_difference(
        Context ctx, IndexPartitionT<DIM, COORD_T> parent,
        Point<COLOR_DIM, COLOR_COORD_T> color,
        IndexSpaceT<DIM, COORD_T> initial,
        const typename std::vector<IndexSpaceT<DIM, COORD_T> >& handles,
        const char* provenance = nullptr);
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
    IndexPartition get_index_partition(
        Context ctx, IndexSpace parent, Color color);
    IndexPartition get_index_partition(
        Context ctx, IndexSpace parent, const DomainPoint& color);
    // Context free versions
    IndexPartition get_index_partition(IndexSpace parent, Color color);
    IndexPartition get_index_partition(
        IndexSpace parent, const DomainPoint& color);
    // Template version
    template<int DIM, typename COORD_T>
    IndexPartitionT<DIM, COORD_T> get_index_partition(
        IndexSpaceT<DIM, COORD_T> parent, Color color);

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
    bool has_index_partition(
        Context ctx, IndexSpace parent, const DomainPoint& color);
    // Context free
    bool has_index_partition(IndexSpace parent, Color color);
    bool has_index_partition(IndexSpace parent, const DomainPoint& color);
    // Template version
    template<int DIM, typename COORD_T>
    bool has_index_partition(IndexSpaceT<DIM, COORD_T> parent, Color color);
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
    IndexSpace get_index_subspace(Context ctx, IndexPartition p, Color color);
    IndexSpace get_index_subspace(
        Context ctx, IndexPartition p, const DomainPoint& color);
    // Context free versions
    IndexSpace get_index_subspace(IndexPartition p, Color color);
    IndexSpace get_index_subspace(IndexPartition p, const DomainPoint& color);
    // Template version
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexSpaceT<DIM, COORD_T> get_index_subspace(
        IndexPartitionT<DIM, COORD_T> p, Point<COLOR_DIM, COLOR_COORD_T> color);
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
    bool has_index_subspace(
        Context ctx, IndexPartition p, const DomainPoint& color);
    // Context free
    bool has_index_subspace(IndexPartition p, const DomainPoint& color);
    // Template version
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    bool has_index_subspace(
        IndexPartitionT<DIM, COORD_T> p, Point<COLOR_DIM, COLOR_COORD_T> color);
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
    DomainT<DIM, COORD_T> get_index_space_domain(
        IndexSpaceT<DIM, COORD_T> handle);
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
    void get_index_space_domains(
        Context ctx, IndexSpace handle, std::vector<Domain>& domains);
    // Context free
    LEGION_DEPRECATED("Multiple domains are no longer supported.")
    void get_index_space_domains(
        IndexSpace handle, std::vector<Domain>& domains);
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
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    DomainT<COLOR_DIM, COLOR_COORD_T> get_index_partition_color_space(
        IndexPartitionT<DIM, COORD_T> p);
    ///@}

    ///@{
    /**
     * Return the name of the color space for a partition
     * @param ctx enclosing task context
     * @param p handle for the index partition
     * @return the name of the color space of the specified partition
     */
    IndexSpace get_index_partition_color_space_name(
        Context ctx, IndexPartition p);
    // Context free
    IndexSpace get_index_partition_color_space_name(IndexPartition p);
    // Template version
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    IndexSpaceT<COLOR_DIM, COLOR_COORD_T> get_index_partition_color_space_name(
        IndexPartitionT<DIM, COORD_T> p);
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
    void get_index_space_partition_colors(
        Context ctx, IndexSpace sp, std::set<Color>& colors);
    void get_index_space_partition_colors(
        Context ctx, IndexSpace sp, std::set<DomainPoint>& colors);
    // Context free versions
    void get_index_space_partition_colors(
        IndexSpace sp, std::set<Color>& colors);
    void get_index_space_partition_colors(
        IndexSpace sp, std::set<DomainPoint>& colors);
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
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    Point<COLOR_DIM, COLOR_COORD_T> get_index_space_color(
        IndexSpaceT<DIM, COORD_T> handle);
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
    DomainPoint get_index_partition_color_point(
        Context ctx, IndexPartition handle);
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
    IndexSpaceT<DIM, COORD_T> get_parent_index_space(
        IndexPartitionT<DIM, COORD_T> handle);
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
    IndexPartitionT<DIM, COORD_T> get_parent_index_partition(
        IndexSpaceT<DIM, COORD_T> handle);
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
     * Safe case a domain point down to a target region.  If the point
     * is not in the target region, then an empty domain point
     * is returned.
     * @param ctx enclosing task context
     * @param point the domain point to be cast
     * @param region the target logical region
     * @return the same point if it can be safely cast, otherwise empty
     */
    DomainPoint safe_cast(Context ctx, DomainPoint point, LogicalRegion region);

    /**
     * Safe case a domain point down to a target region.  If the point
     * is not in the target region, returns false, otherwise returns true
     * @param ctx enclosing task context
     * @param point the domain point to be cast
     * @param region the target logical region
     * @return if the point is in the logical region
     */
    template<int DIM, typename COORD_T>
    bool safe_cast(
        Context ctx, Point<DIM, COORD_T> point,
        LogicalRegionT<DIM, COORD_T> region);
  public:
    //------------------------------------------------------------------------
    // Field Space Operations
    //------------------------------------------------------------------------
    /**
     * Create a new field space.
     * @param ctx enclosing task context
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return handle for the new field space
     */
    FieldSpace create_field_space(
        Context ctx, const char* provenance = nullptr);
    /**
     * Create a new field space with an existing set of fields
     * @param ctx enclosing task context
     * @param field_sizes initial field sizes for fields
     * @param resulting_fields optional field snames for fields
     * @param serdez_id optional parameter for specifying a custom
     *        serdez object for serializing and deserializing fields
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return handle for the new field space
     */
    FieldSpace create_field_space(
        Context ctx, const std::vector<size_t>& field_sizes,
        std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id = 0,
        const char* provenance = nullptr);
    /**
     * Create a new field space with an existing set of fields
     * @param ctx enclosing task context
     * @param field_sizes initial field sizes for fields
     * @param resulting_fields optional field snames for fields
     * @param serdez_id optional parameter for specifying a custom
     *        serdez object for serializing and deserializing fields
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return handle for the new field space
     */
    FieldSpace create_field_space(
        Context ctx, const std::vector<Future>& field_sizes,
        std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id = 0,
        const char* provenance = nullptr);
    /**
     * Create a new shared ownership of a field space to prevent it
     * from being destroyed by other potential owners. Every call to this
     * method that succeeds must be matched with a corresponding call
     * to destroy the field space in order for the field space to
     * actually be deleted. The field space must not have been destroyed
     * prior to this call being performed.
     * @param ctx the enclosing task context
     * @param handle for field space to request ownership for
     */
    void create_shared_ownership(Context ctx, FieldSpace handle);
    /**
     * Destroy an existing field space.
     * @param ctx enclosing task context
     * @param handle of the field space to be destroyed
     * @param unordered set to true if this is performed by a different
     *          thread than the one for the task (e.g a garbage collector)
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     */
    void destroy_field_space(
        Context ctx, FieldSpace handle, const bool unordered = false,
        const char* provenance = nullptr);

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
    void get_field_space_fields(
        Context ctx, FieldSpace handle, std::vector<FieldID>& fields);
    // Context free
    void get_field_space_fields(
        FieldSpace handle, std::vector<FieldID>& fields);
    ///@}

    ///@{
    /**
     * Get the IDs of the fields currently allocated in a field space.
     * @param ctx enclosing task context
     * @param handle field space handle
     * @param set in which to place the field IDs
     */
    void get_field_space_fields(
        Context ctx, FieldSpace handle, std::set<FieldID>& fields);
    // Context free
    void get_field_space_fields(FieldSpace handle, std::set<FieldID>& fields);
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
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return handle for the logical region created
     */
    LogicalRegion create_logical_region(
        Context ctx, IndexSpace index, FieldSpace fields,
        bool task_local = false, const char* provenance = nullptr);
    // Template version
    template<int DIM, typename COORD_T>
    LogicalRegionT<DIM, COORD_T> create_logical_region(
        Context ctx, IndexSpaceT<DIM, COORD_T> index, FieldSpace fields,
        bool task_local = false, const char* provenance = nullptr);
    /**
     * Create a new shared ownership of a top-level logical region to prevent
     * it  from being destroyed by other potential owners. Every call to this
     * method that succeeds must be matched with a corresponding call
     * to destroy the logical region in order for the logical region to
     * actually be deleted. The logical region must not have been destroyed
     * prior to this call being performed.
     * @param ctx the enclosing task context
     * @param handle for top-level logical region to request ownership for
     */
    void create_shared_ownership(Context ctx, LogicalRegion handle);
    /**
     * Destroy a logical region and all of its logical sub-regions.
     * @param ctx enclosing task context
     * @param handle logical region handle to destroy
     * @param unordered set to true if this is performed by a different
     *          thread than the one for the task (e.g a garbage collector)
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     */
    void destroy_logical_region(
        Context ctx, LogicalRegion handle, const bool unordered = false,
        const char* provenance = nullptr);

    /**
     * Destroy a logical partition and all of it is logical sub-regions.
     * @param ctx enclosing task context
     * @param handle logical partition handle to destroy
     * @param unordered set to true if this is performed by a different
     *          thread than the one for the task (e.g a garbage collector)
     */
    LEGION_DEPRECATED(
        "Destruction of logical partitions are no-ops now."
        "Logical partitions are automatically destroyed when their root "
        "logical region or their index spartition are destroyed.")
    void destroy_logical_partition(
        Context ctx, LogicalPartition handle, const bool unordered = false);

    /**
     * Internally the runtime creates "equivalence sets" which are
     * subsets of logical regions that it uses for performing its analyses.
     * In general, these equivalence sets are established on a first touch
     * basis and then altered using runtime heuristics. However, you can
     * influence their selection using this API call which will reset the
     * equivalence sets for certain fields on a arbitrary region in the
     * region tree (note you must have privileges on this region). The
     * next task to use this region or any overlapping regions will create
     * new equivalence sets. Therefore it is useful to use this to inform
     * the runtime when switching from one partition to a new partition.
     * Note that this method will only impact your performance and has no
     * bearing on the correctness of your application.
     * @param ctx enclosing task context
     * @param parent the logical region where privileges are derived from
     * @param region the region to reset the equivalence sets for
     * @param fields the fields for which these should apply
     */
    void reset_equivalence_sets(
        Context ctx, LogicalRegion parent, LogicalRegion region,
        const std::set<FieldID>& fields);
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
    LogicalPartition get_logical_partition(
        Context ctx, LogicalRegion parent, IndexPartition handle);
    // Context free
    LogicalPartition get_logical_partition(
        LogicalRegion parent, IndexPartition handle);
    // Template version
    template<int DIM, typename COORD_T>
    LogicalPartitionT<DIM, COORD_T> get_logical_partition(
        LogicalRegionT<DIM, COORD_T> parent,
        IndexPartitionT<DIM, COORD_T> handle);
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
    LogicalPartition get_logical_partition_by_color(
        Context ctx, LogicalRegion parent, Color c);
    LogicalPartition get_logical_partition_by_color(
        Context ctx, LogicalRegion parent, const DomainPoint& c);
    // Context free
    LogicalPartition get_logical_partition_by_color(
        LogicalRegion parent, Color c);
    LogicalPartition get_logical_partition_by_color(
        LogicalRegion parent, const DomainPoint& c);
    // Template version
    template<int DIM, typename COORD_T>
    LogicalPartitionT<DIM, COORD_T> get_logical_partition_by_color(
        LogicalRegionT<DIM, COORD_T> parent, Color c);
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
    bool has_logical_partition_by_color(
        Context ctx, LogicalRegion parent, const DomainPoint& c);
    // Context free
    bool has_logical_partition_by_color(
        LogicalRegion parent, const DomainPoint& c);
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
    LogicalPartition get_logical_partition_by_tree(
        Context ctx, IndexPartition handle, FieldSpace fspace,
        RegionTreeID tid);
    // Context free
    LogicalPartition get_logical_partition_by_tree(
        IndexPartition handle, FieldSpace fspace, RegionTreeID tid);
    // Template version
    template<int DIM, typename COORD_T>
    LogicalPartitionT<DIM, COORD_T> get_logical_partition_by_tree(
        IndexPartitionT<DIM, COORD_T> handle, FieldSpace fspace,
        RegionTreeID tid);
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
    LogicalRegion get_logical_subregion(
        Context ctx, LogicalPartition parent, IndexSpace handle);
    // Context free
    LogicalRegion get_logical_subregion(
        LogicalPartition parent, IndexSpace handle);
    // Template version
    template<int DIM, typename COORD_T>
    LogicalRegionT<DIM, COORD_T> get_logical_subregion(
        LogicalPartitionT<DIM, COORD_T> parent,
        IndexSpaceT<DIM, COORD_T> handle);
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
    LogicalRegion get_logical_subregion_by_color(
        Context ctx, LogicalPartition parent, Color c);
    LogicalRegion get_logical_subregion_by_color(
        Context ctx, LogicalPartition parent, const DomainPoint& c);
    // Context free
    LogicalRegion get_logical_subregion_by_color(
        LogicalPartition parent, Color c);
    LogicalRegion get_logical_subregion_by_color(
        LogicalPartition parent, const DomainPoint& c);
    // Template version
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    LogicalRegionT<DIM, COORD_T> get_logical_subregion_by_color(
        LogicalPartitionT<DIM, COORD_T> parent,
        Point<COLOR_DIM, COLOR_COORD_T> color);
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
    bool has_logical_subregion_by_color(
        Context ctx, LogicalPartition parent, const DomainPoint& c);
    // Context free
    bool has_logical_subregion_by_color(
        LogicalPartition parent, const DomainPoint& c);
    // Template version
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    bool has_logical_subregion_by_color(
        LogicalPartitionT<DIM, COORD_T> parent,
        Point<COLOR_DIM, COLOR_COORD_T> color);
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
    LogicalRegion get_logical_subregion_by_tree(
        Context ctx, IndexSpace handle, FieldSpace fspace, RegionTreeID tid);
    // Context free
    LogicalRegion get_logical_subregion_by_tree(
        IndexSpace handle, FieldSpace fspace, RegionTreeID tid);
    // Template version
    template<int DIM, typename COORD_T>
    LogicalRegionT<DIM, COORD_T> get_logical_subregion_by_tree(
        IndexSpaceT<DIM, COORD_T> handle, FieldSpace space, RegionTreeID tid);
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
    DomainPoint get_logical_region_color_point(
        Context ctx, LogicalRegion handle);
    // Context free versions
    Color get_logical_region_color(LogicalRegion handle);
    DomainPoint get_logical_region_color_point(LogicalRegion handle);
    // Template version
    template<int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
    Point<COLOR_DIM, COLOR_COORD_T> get_logical_region_color_point(
        LogicalRegionT<DIM, COORD_T> handle);
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
    DomainPoint get_logical_partition_color_point(
        Context ctx, LogicalPartition handle);
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
    LogicalRegion get_parent_logical_region(
        Context ctx, LogicalPartition handle);
    // Context free
    LogicalRegion get_parent_logical_region(LogicalPartition handle);
    // Template version
    template<int DIM, typename COORD_T>
    LogicalRegionT<DIM, COORD_T> get_parent_logical_region(
        LogicalPartitionT<DIM, COORD_T> handle);
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
    LogicalPartition get_parent_logical_partition(
        Context ctx, LogicalRegion handle);
    // Context free
    LogicalPartition get_parent_logical_partition(LogicalRegion handle);
    // Template version
    template<int DIM, typename COORD_T>
    LogicalPartitionT<DIM, COORD_T> get_parent_logical_partition(
        LogicalRegionT<DIM, COORD_T> handle);
    ///@}
  public:
    //------------------------------------------------------------------------
    // Allocator and Argument Map Operations
    //------------------------------------------------------------------------
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
     * @param outputs optional output requirements
     * @return a future for the return value of the task
     */
    Future execute_task(
        Context ctx, const TaskLauncher& launcher,
        std::vector<OutputRequirement>* outputs = nullptr);

    /**
     * Launch an index space of tasks with arguments specified
     * by the index launcher configuration.
     * @see IndexTaskLauncher
     * @param ctx enclosing task context
     * @param launcher the task launcher configuration
     * @param outputs optional output requirements
     * @return a future map for return values of the points
     *    in the index space of tasks
     */
    FutureMap execute_index_space(
        Context ctx, const IndexTaskLauncher& launcher,
        std::vector<OutputRequirement>* outputs = nullptr);

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
     * @param ordered request that the reduced future value be computed
     *        in an ordered way so that all shards see a consistent result,
     *        this is more expensive than unordered which allows for the
     *        creation of a butterfly all-reduce network across the shards,
     *        integer reductions should be safe to use with the unordered
     *        mode and still give the same result on each shard whereas
     *        floating point reductions may produce different answers on
     *        each shard if ordered is set to false
     * @param outputs optional output requirements
     * @return a future result representing the reduction of
     *    all the return values from the index space of tasks
     */
    Future execute_index_space(
        Context ctx, const IndexTaskLauncher& launcher, ReductionOpID redop,
        bool ordered = true, std::vector<OutputRequirement>* outputs = nullptr);

    /**
     * Reduce a future map down to a single future value using
     * a specified reduction operator. This assumes that all the values
     * in the future map are instance of the reduction operator's RHS
     * type and the resulting future will also be an RHS type.
     * @param ctx enclosing task context
     * @param future_map the future map to reduct the value
     * @param redop ID for the reduction op to use for reducing values
     * @param ordered request that the reduced future value be computed
     *        in an ordered way so that all shards see a consistent result,
     *        this is more expensive than unordered which allows for the
     *        creation of a butterfly all-reduce network across the shards,
     *        integer reductions should be safe to use with the unordered
     *        mode and still give the same result on each shard whereas
     *        floating point reductions may produce different answers on
     *        each shard if ordered is set to false
     * @param map_id mapper to use for deciding where to map the output future
     * @param tag pass-through value to the mapper for application context
     * @param provenance an optional string for describing the provenance
     *        of this invocation
     * @return a future result representing the the reduction of all the
     *         values in the future map
     */
    Future reduce_future_map(
        Context ctx, const FutureMap& future_map, ReductionOpID redop,
        bool ordered = true, MapperID map_id = 0, MappingTagID tag = 0,
        const char* provenance = nullptr, Future initial_value = Future());

    /**
     * Construct a future map from a collection of buffers. The user must
     * also specify the domain of the future map and there must be one buffer
     * for every point in the domain. In the case of 'collective=true' the
     * runtime supports different shards in a control-replicated context
     * to work collectively to construct the future map. The runtime will
     * not detect if points are missing or if points are duplicated and
     * undefined behavior will occur. If 'collective=true', the application
     * must provide a sharding function that describes assignment of points
     * to shards for the runtime to use. The runtime will verify this
     * sharding function accurately describes all the points passed in.
     * If the task is not control-replicated then the 'collective' flag
     * will not have any effect.
     * @param ctx enclosing task context
     * @param domain the index space that names all points in the future map
     * @param data the set of futures from which to create the future map
     * @param collective whether shards from a control replicated context
     *                   should work collectively to construct the map
     * @param sid the sharding function ID that describes the sharding
     *                   pattern if collective=true
     * @param implicit_sharding if collective=true this says whether the
     *                   sharding should be implicitly handled by the
     *                   runtime and the sharding function ID ignored
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return a new future map containing all the futures
     */
    FutureMap construct_future_map(
        Context ctx, IndexSpace domain,
        const std::map<DomainPoint, UntypedBuffer>& data,
        bool collective = false, ShardingID sid = 0,
        bool implicit_sharding = false, const char* provenance = nullptr);
    LEGION_DEPRECATED("Use the version that takes an IndexSpace instead")
    FutureMap construct_future_map(
        Context ctx, const Domain& domain,
        const std::map<DomainPoint, UntypedBuffer>& data,
        bool collective = false, ShardingID sid = 0,
        bool implicit_sharding = false);

    /**
     * Construct a future map from a collection of futures. The user must
     * also specify the domain of the futures and there must be one future
     * for every point in the domain. In the case of 'collective=true' the
     * runtime supports different shards in a control-replicated context
     * to work collectively to construct the future map. The runtime will
     * not detect if points are missing or if points are duplicated and
     * undefined behavior will occur. If the task is not control-replicated
     * then the 'collective' flag will not have any effect.
     * @param ctx enclosing task context
     * @param domain the index space that names all points in the future map
     * @param futures the set of futures from which to create the future map
     * @param collective whether shards from a control replicated context
     *                   should work collectively to construct the map
     * @param sid the sharding function ID that describes the sharding
     *                   pattern if collective=true
     * @param implicit_sharding if collective=true this says whether the
     *                   sharding should be implicitly handled by the
     *                   runtime and the sharding function ID ignored
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return a new future map containing all the futures
     */
    FutureMap construct_future_map(
        Context ctx, IndexSpace domain,
        const std::map<DomainPoint, Future>& futures, bool collective = false,
        ShardingID sid = 0, bool implicit_sharding = false,
        const char* provenance = nullptr);
    LEGION_DEPRECATED("Use the version that takes an IndexSpace instead")
    FutureMap construct_future_map(
        Context ctx, const Domain& domain,
        const std::map<DomainPoint, Future>& futures, bool collective = false,
        ShardingID sid = 0, bool implicit_sharding = false);

    /**
     * Apply a transform to a FutureMap. All points that access the
     * FutureMap will be transformed by the 'transform' function before
     * accessing the backing future map. Note that multiple transforms
     * can be composed this way to create new FutureMaps. This version
     * takes a function pointer which must take a domain point and the
     * range of the original future map and returns a new domain point
     * that must fall within the range.
     * @param ctx enclosing task context
     * @param fm future map to apply a new coordinate space to
     * @param new_domain an index space to describe the domain of points
     *        for the transformed future map
     * @param fnptr a function pointer to call to transform points
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return a new future map with the coordinate space transformed
     */
    using PointTransformFunc = std::function<DomainPoint(
        const DomainPoint&, const Domain&, const Domain&)>;
    FutureMap transform_future_map(
        Context ctx, const FutureMap& fm, IndexSpace new_domain,
        PointTransformFunc func, const char* provenance = nullptr);
    /**
     * Apply a transform to a FutureMap. All points that access the
     * FutureMap will be transformed by the 'transform' function before
     * accessing the backing future map. Note that multiple transforms
     * can be composed this way to create new FutureMaps. This version
     * takes a pointer PointTransform functor object to invoke to
     * transform the coordinate spaces of the points.
     * @param ctx enclosing task context
     * @param fm future map to apply a new coordinate space to
     * @param new_domain an index space to describe the domain of points
     *        for the transformed future map
     * @param functor pointer to a functor to transform points
     * @param take_ownership whether the runtime should delete the functor
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return a new future map with the coordinate space transformed
     */
    FutureMap transform_future_map(
        Context ctx, const FutureMap& fm, IndexSpace new_domain,
        PointTransformFunctor* functor, bool take_ownership = false,
        const char* provenance = nullptr);

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
    LEGION_DEPRECATED(
        "Launching tasks should be done with the new task "
        "launcher interface.")
    Future execute_task(
        Context ctx, TaskID task_id,
        const std::vector<IndexSpaceRequirement>& indexes,
        const std::vector<FieldSpaceRequirement>& fields,
        const std::vector<RegionRequirement>& regions, const UntypedBuffer& arg,
        const Predicate& predicate = Predicate::TRUE_PRED, MapperID id = 0,
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
    LEGION_DEPRECATED(
        "Launching tasks should be done with the new task "
        "launcher interface.")
    FutureMap execute_index_space(
        Context ctx, TaskID task_id, const Domain domain,
        const std::vector<IndexSpaceRequirement>& indexes,
        const std::vector<FieldSpaceRequirement>& fields,
        const std::vector<RegionRequirement>& regions,
        const UntypedBuffer& global_arg, const ArgumentMap& arg_map,
        const Predicate& predicate = Predicate::TRUE_PRED,
        bool must_paralleism = false, MapperID id = 0, MappingTagID tag = 0);

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
    LEGION_DEPRECATED(
        "Launching tasks should be done with the new task "
        "launcher interface.")
    Future execute_index_space(
        Context ctx, TaskID task_id, const Domain domain,
        const std::vector<IndexSpaceRequirement>& indexes,
        const std::vector<FieldSpaceRequirement>& fields,
        const std::vector<RegionRequirement>& regions,
        const UntypedBuffer& global_arg, const ArgumentMap& arg_map,
        ReductionOpID reduction, const UntypedBuffer& initial_value,
        const Predicate& predicate = Predicate::TRUE_PRED,
        bool must_parallelism = false, MapperID id = 0, MappingTagID tag = 0);
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
    PhysicalRegion map_region(Context ctx, const InlineLauncher& launcher);

    /**
     * Perform an inline mapping operation which returns a physical region
     * object for the requested region requirement.  Note the application
     * must wait for the resulting physical region to become valid before
     * using it.
     * @param ctx enclosing task context
     * @param req the region requirement for the inline mapping
     * @param id the mapper ID to associate with the operation
     * @param tag the mapping tag to pass to any mapping calls
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return a physical region for the resulting data
     */
    PhysicalRegion map_region(
        Context ctx, const RegionRequirement& req, MapperID id = 0,
        MappingTagID tag = 0, const char* provenance = nullptr);

    /**
     * Perform an inline mapping operation that re-maps a physical region
     * that was initially mapped when the task began.
     * @param ctx enclosing task context
     * @param idx index of the region requirement from the enclosing task
     * @param id the mapper ID to associate with the operation
     * @param the mapping tag to pass to any mapping calls
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return a physical region for the resulting data
     */
    PhysicalRegion map_region(
        Context ctx, unsigned idx, MapperID id = 0, MappingTagID tag = 0,
        const char* provenance = nullptr);

    /**
     * Remap a region from an existing physical region.  It will
     * still be necessary for the application to wait until the
     * physical region is valid again before using it.
     * @param ctx enclosing task context
     * @param region the physical region to be remapped
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     */
    void remap_region(
        Context ctx, PhysicalRegion region, const char* provenance = nullptr);

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
    // Output Region Operations
    //------------------------------------------------------------------------
    /**
     * Return a single output region of a task.
     * @param ctx enclosing task context
     * @param index the output region index to query
     */
    OutputRegion get_output_region(Context ctx, unsigned index);

    /**
     * Return all output regions of a task.
     * @param ctx enclosing task context
     * @param regions a vector to which output regions are returned
     */
    void get_output_regions(Context ctx, std::vector<OutputRegion>& regions);
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
    void fill_field(
        Context ctx, LogicalRegion handle, LogicalRegion parent, FieldID fid,
        const T& value, Predicate pred = Predicate::TRUE_PRED);

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
    void fill_field(
        Context ctx, LogicalRegion handle, LogicalRegion parent, FieldID fid,
        const void* value, size_t value_size,
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
    void fill_field(
        Context ctx, LogicalRegion handle, LogicalRegion parent, FieldID fid,
        Future f, Predicate pred = Predicate::TRUE_PRED);

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
    void fill_fields(
        Context ctx, LogicalRegion handle, LogicalRegion parent,
        const std::set<FieldID>& fields, const T& value,
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
    void fill_fields(
        Context ctx, LogicalRegion handle, LogicalRegion parent,
        const std::set<FieldID>& fields, const void* value, size_t value_size,
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
    void fill_fields(
        Context ctx, LogicalRegion handle, LogicalRegion parent,
        const std::set<FieldID>& fields, Future f,
        Predicate pred = Predicate::TRUE_PRED);

    /**
     * Perform a fill operation using a launcher which specifies
     * all of the parameters of the launch.
     * @param ctx enclosing task context
     * @param launcher the launcher that describes the fill operation
     */
    void fill_fields(Context ctx, const FillLauncher& launcher);

    /**
     * Perform an index fill operation using a launcher which
     * specifies all the parameters of the launch.
     * @param ctx enclosing task context
     * @param launcher the launcher that describes the index fill operation
     */
    void fill_fields(Context ctx, const IndexFillLauncher& launcher);

    /**
     * Discard the data inside the fields of a particular logical region
     * @param ctx enclosing task context
     * @param launcher the launcher that describes the discard operation
     */
    void discard_fields(Context ctx, const DiscardLauncher& launcher);
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
    PhysicalRegion attach_external_resource(
        Context ctx, const AttachLauncher& launcher);

    /**
     * Detach an external resource from a logical region
     * @param ctx enclosing task context
     * @param region the physical region for the external resource
     * @param flush copy out data to the physical region before detaching
     * @param unordered set to true if this is performed by a different
     *          thread than the one for the task (e.g a garbage collector)
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return an empty future indicating when the resource is detached
     */
    Future detach_external_resource(
        Context ctx, PhysicalRegion region, const bool flush = true,
        const bool unordered = false, const char* provenance = nullptr);

    /**
     * Attach multiple external resources to logical regions using an
     * index space attach operation. In a control replicated context
     * this method is an unusual "collective" method meaning that
     * different shards are allowed to pass in different arguments
     * and the runtime will interpret them as different sub-operations
     * coming from different shards. All the physical regions returned
     * from this method must be detached together as well using the
     * 'detach_external_resources' method and cannot be detached
     * individually using the 'detach_external_resource' method.
     * @param ctx enclosing task context
     * @param launcher the index attach launcher describing the external
     *        resources to be attached
     * @return an external resources objects containing the physical
     *        regions for the attached resources with regions in the
     *        same order as specified in the launcher
     */
    ExternalResources attach_external_resources(
        Context ctx, const IndexAttachLauncher& launcher);

    /**
     * Detach multiple external resources that were all created by
     * a common call to 'attach_external_resources'.
     * @param ctx enclosing task context
     * @param external the external resources to detach
     * @param flush copy out data to the physical region before detaching
     * @param unordered set to true if this is performed by a different
     *          thread than the one for the task (e.g a garbage collector)
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return an empty future indicating when the resources are detached
     */
    Future detach_external_resources(
        Context ctx, ExternalResources external, const bool flush = true,
        const bool unordered = false, const char* provenance = nullptr);

    /**
     * Force progress on unordered operations. After performing one
     * of these calls then all outstanding unordered operations that
     * have been issued are guaranteed to be in the task stream.
     * @param ctx enclosing task context
     */
    void progress_unordered_operations(Context ctx);

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
    LEGION_DEPRECATED(
        "Attaching specific HDF5 file type is deprecated "
        "in favor of generic attach launcher interface.")
    PhysicalRegion attach_hdf5(
        Context ctx, const char* file_name, LogicalRegion handle,
        LogicalRegion parent, const std::map<FieldID, const char*>& field_map,
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
    LEGION_DEPRECATED(
        "Detaching specific HDF5 file type is deprecated "
        "in favor of generic detach interface.")
    void detach_hdf5(Context ctx, PhysicalRegion region);

    /**
     * @deprecated
     * Attach an normal file as a physical region. This attach is similar to
     * attach_hdf5 operation, except that the file has exact same data format
     * as in-memory physical region. Data lays out as SOA in file.
     */
    LEGION_DEPRECATED(
        "Attaching generic file type is deprecated "
        "in favor of generic attach launcher interface.")
    PhysicalRegion attach_file(
        Context ctx, const char* file_name, LogicalRegion handle,
        LogicalRegion parent, const std::vector<FieldID>& field_vec,
        LegionFileMode mode);

    /**
     * @deprecated
     * Detach an normal file. THis detach operation is similar to
     * detach_hdf5
     */
    LEGION_DEPRECATED(
        "Detaching generic file type is deprecated "
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
    void issue_copy_operation(Context ctx, const CopyLauncher& launcher);

    /**
     * Launch an index copy operation from the given configuration
     * of the given copy launcher
     * @see IndexCopyLauncher
     * @param ctx enclosing task context
     * @param launcher index copy launcher object
     */
    void issue_copy_operation(Context ctx, const IndexCopyLauncher& launcher);
  public:
    //------------------------------------------------------------------------
    // Predicate Operations
    //------------------------------------------------------------------------
    /**
     * Create a new predicate value from a future.  The future passed
     * must be a boolean future.
     * @param ctx enclosing task context
     * @param f future value to convert to a predicate
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return predicate value wrapping the future
     */
    Predicate create_predicate(
        Context ctx, const Future& f, const char* provenance = nullptr);

    /**
     * Create a new predicate value that is the logical
     * negation of another predicate value.
     * @param ctx enclosing task context
     * @param p predicate value to logically negate
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return predicate value logically negating previous predicate
     */
    Predicate predicate_not(
        Context ctx, const Predicate& p, const char* provenance = nullptr);

    /**
     * Create a new predicate value that is the logical
     * conjunction of two other predicate values.
     * @param ctx enclosing task context
     * @param p1 first predicate to logically and
     * @param p2 second predicate to logically and
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return predicate value logically and-ing two predicates
     */
    Predicate predicate_and(
        Context ctx, const Predicate& p1, const Predicate& p2,
        const char* provenance = nullptr);

    /**
     * Create a new predicate value that is the logical
     * disjunction of two other predicate values.
     * @param ctx enclosing task context
     * @param p1 first predicate to logically or
     * @param p2 second predicate to logically or
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return predicate value logically or-ing two predicates
     */
    Predicate predicate_or(
        Context ctx, const Predicate& p1, const Predicate& p2,
        const char* provenance = nullptr);

    /**
     * Generic predicate constructor for an arbitrary number of predicates
     * @param ctx enclosing task context
     * @param launcher the predicate launcher
     * #return predicate value of combining other predicates
     */
    Predicate create_predicate(Context ctx, const PredicateLauncher& launcher);

    /**
     * Get a future value that will be completed when the predicate triggers
     * @param ctx enclosing task context
     * @param pred the predicate for which to get a future
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return a boolean future with the result of the predicate
     */
    Future get_predicate_future(
        Context ctx, const Predicate& p, const char* provenance = nullptr);
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
    Grant acquire_grant(Context ctx, const std::vector<LockRequest>& requests);

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
     * in the next phase of the computation. Note that once
     * a phase barrier exhausts its total number of generations
     * then it will fail the 'exists' method test. It is the
     * responsibility of the application to detect this case
     * and handle it correctly by making a new PhaseBarrier.
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
    DynamicCollective create_dynamic_collective(
        Context ctx, unsigned arrivals, ReductionOpID redop,
        const void* init_value, size_t init_size);

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
    void arrive_dynamic_collective(
        Context ctx, DynamicCollective dc, const void* buffer, size_t size,
        unsigned count = 1);

    /**
     * Perform a deferred arrival on a dynamic collective dependent
     * upon a future value.  The runtime will automatically pipe the
     * future value through to the dynamic collective.
     * @param ctx enclosing task context
     * @param dc dynamic collective on which to arrive
     * @param f future to use for performing the arrival
     * @param count total arrival count
     */
    void defer_dynamic_collective_arrival(
        Context ctx, DynamicCollective dc, const Future& f, unsigned count = 1);

    /**
     * This will return the value of a dynamic collective in
     * the form of a future. Applications can then use this
     * future just like all other futures.
     * @param ctx enclosing task context
     * @param dc dynamic collective on which to get the result
     * @param provenance an optional string describing the provenance
     *                   information for this operation
     * @return future value that contains the result of the collective
     */
    Future get_dynamic_collective_result(
        Context ctx, DynamicCollective dc, const char* provenance = nullptr);

    /**
     * Advance an existing dynamic collective to the next
     * phase.  It has the same semantics as the equivalent
     * call for phase barriers.
     * @param ctx enclosing task context
     * @param dc the dynamic collective to be advanced
     * @return an updated dynamic collective used for the next phase
     */
    DynamicCollective advance_dynamic_collective(
        Context ctx, DynamicCollective dc);
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
    void issue_acquire(Context ctx, const AcquireLauncher& launcher);

    /**
     * Issue a release operation on the specified physical region
     * provided by the release launcher.  This call should be preceded
     * by an acquire call earlier in teh same context on the same
     * physical region.
     */
    void issue_release(Context ctx, const ReleaseLauncher& launcher);
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
    Future issue_mapping_fence(Context ctx, const char* provenance = nullptr);

    /**
     * Issue a Legion execution fence in the current context.  A
     * Legion execution fence guarantees that all of the tasks issued
     * in the context prior to the fence will finish running
     * before the tasks after the fence begin to map.  This
     * will allow the necessary propagation of Legion meta-data
     * such as modifications to the region tree made prior
     * to the fence visible to tasks issued after the fence.
     */
    Future issue_execution_fence(Context ctx, const char* provenance = nullptr);
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
     * Traces are currently not permitted to be nested. In general,
     * the runtime will capture all dependence information for the
     * trace. However, in some cases, compilers may want to pass
     * information along for the logical dependence analysis as a
     * static trace. Inside of a static trace it is the application's
     * responsibility to specify any dependences that would normally
     * have existed between each operation being launched and any prior
     * operations in the trace (there is no need to specify dependences
     * on anything outside of the trace). The application can optionally
     * specify a set of region trees for which it will be supplying
     * dependences, with all other region trees being left to the runtime
     * to handle. If no such set is specified then the runtime will operate
     * under the assumption that the application is specifying dependences
     * for all region trees.
     * @param ctx the enclosing task context
     * @param tid the trace ID of the trace to be captured
     * @param logical_only whether physical tracing is permitted
     * @param static_trace whether this is a static trace
     * @param managed specific region trees the application will handle
     *                in the case of a static trace
     */
    void begin_trace(
        Context ctx, TraceID tid, bool logical_only = false,
        bool static_trace = false,
        const std::set<RegionTreeID>* managed = nullptr,
        const char* provenance = nullptr);
    /**
     * Mark the end of trace that was being performed.
     */
    void end_trace(Context ctx, TraceID tid, const char* provenance = nullptr);
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
    LEGION_DEPRECATED("Use begin_trace with static_trace=true")
    void begin_static_trace(
        Context ctx, const std::set<RegionTreeID>* managed = nullptr);
    /**
     * Finish a static trace of operations
     * @param ctx the enclosing task context
     */
    LEGION_DEPRECATED("Use end_trace")
    void end_static_trace(Context ctx);

    /**
     * Dynamically generate a unique TraceID for use across the machine
     * @return a TraceID that is globally unique across the machine
     */
    TraceID generate_dynamic_trace_id(void);

    /**
     * Generate a contiguous set of TraceIDs for use by a library.
     * This call will always generate the same answer for the same library
     * name no many how many times it is called or on how many nodes it
     * is called. If the count passed in to this method differs for the
     * same library name the runtime will raise an error.
     * @param name a unique null-terminated string that names the library
     * @param count the number of trace IDs that should be generated
     * @return the first TraceID that is allocated to the library
     */
    TraceID generate_library_trace_ids(const char* name, size_t count);

    /**
     * Statically generate a unique Trace ID for use across the machine.
     * This can only be called prior to the runtime starting. It must be
     * invoked symmetrically across all the nodes in the machine prior
     * to starting the runtime.
     * @return a TraceID that is globally unique across the machine
     */
    static TraceID generate_static_trace_id(void);
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
    void complete_frame(Context ctx, const char* provenance = nullptr);
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
    FutureMap execute_must_epoch(
        Context ctx, const MustEpochLauncher& launcher);
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
    Future select_tunable_value(
        Context ctx, TunableID tid, MapperID mapper = 0, MappingTagID tag = 0,
        const void* args = nullptr, size_t argsize = 0);
    Future select_tunable_value(Context ctx, const TunableLauncher& launcher);

    /**
     * @deprecated
     * This is the old method for asking the mapper to specify a
     * tunable value. It will assume that the resulting tunable
     * future can be interpreted as an integer.
     */
    LEGION_DEPRECATED(
        "Tunable values should now be obtained via the "
        "generic interface that returns a future result.")
    int get_tunable_value(
        Context ctx, TunableID tid, MapperID mapper = 0, MappingTagID tag = 0);
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
    void set_local_task_variable_untyped(
        Context ctx, LocalVariableID id, const void* value,
        void (*destructor)(void*) = nullptr);
    template<typename T>
    void set_local_task_variable(
        Context ctx, LocalVariableID id, const T* value,
        void (*destructor)(void*) = nullptr);
  public:
    //------------------------------------------------------------------------
    // Deferred Buffer/Value Creation/Destruction
    //------------------------------------------------------------------------
    /**
     * Try to allocate a deferred value given the parameters in
     * the deferred value request. If the allocation fails then
     * the value will not exist. Note that you can  implicitly convert
     * the untyped version of the value to a typed version after
     * you unpack the optional.
     * @param ctx the enclosing task context
     * @param request struct describing the value parameters
     * @return an UntypedDeferredValue object
     */
    UntypedDeferredValue allocate_deferred_value(
        Context ctx, const DeferredValueRequest& request);

    /**
     * Destroy a deferred value eagerly before the end of the
     * task. Note this is not required as the runtime will clean
     * up this value at the end of the task anyway and indeed
     * not encouraged but we support it in case users want to;
     * @param ctx the enclosing task context
     * @param value the deferred value to be destroyed
     * @param precondition event to defer the deletion on if any
     */
    void destroy_deferred_value(
        Context ctx, UntypedDeferredValue& value,
        Realm::Event precondition = Realm::Event::NO_EVENT);

    /**
     * Try to allocate a deferred buffer given the parameters in
     * the deferred buffer request. If the allocation fails then
     * the buffer will not exist. Note that you can implicitly
     * convert the untyped version of the value to a typed version
     * after you unpack the optional.
     * @param ctx the enclosing task context
     * @param request struct describing the buffer parameters
     * @return an UntypedDeferredBuffer struct
     */
    template<typename COORD_T = coord_t>
    UntypedDeferredBuffer<COORD_T> allocate_deferred_buffer(
        Context ctx, const DeferredBufferRequest& request);

    /**
     * Destroy a deferred buffer eagerly before the end of the
     * task. Note this is not required as the runtime will clean
     * up this value at the end of the task anyway and indeed
     * not encouraged but we support it in case users want to;
     * @param ctx the enclosing task context
     * @param buffer the deferred buffer to be destroyed
     * @param precondition event to defer the deletion on if any
     */
    template<typename COORD_T = coord_t>
    void destroy_deferred_buffer(
        Context ctx, UntypedDeferredBuffer<COORD_T>& buffer,
        Realm::Event precondition = Realm::Event::NO_EVENT);
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
    Future get_current_time_in_microseconds(
        Context ctx, Future precondition = Future());

    /**
     * Issue an operation into the stream to record the current time in
     * nanoseconds. The resulting future should be interpreted as a
     * 'long long' with no fractional nanoseconds. The operation can be
     * givien an optional future precondition which will not be interpreted,
     * but ill be used as a precondition to ensure that the measurement
     * will not be taken until the precondition is complete.
     */
    Future get_current_time_in_nanoseconds(
        Context ctx, Future precondition = Future());

    /**
     * Issue a timing measurement operation configured with a launcher.
     * The above methods are just common special cases. This allows for
     * the general case of an arbitrary measurement with an arbitrary
     * number of preconditions.
     */
    Future issue_timing_measurement(
        Context ctx, const TimingLauncher& launcher);

    /**
     * Return the base time in nanoseconds on THIS node with which all
     * other aboslute timings can be compared. This value will not change
     * during the course of the lifetime of a Legion application and may
     * therefore be safely cached.
     */
    static long long get_zero_time(void);
  public:
    //------------------------------------------------------------------------
    // Exception Handling
    //------------------------------------------------------------------------

    /**
     * Push an exception handler on to the stack for the current task.
     */
    void push_exception_handler(Context ctx, ExceptionHandlerID handler);

    /**
     * Pop an exception handler off the stack of the current task and return
     * a future. If there was an unhandled exception then it will be stored
     * in the future, otherwise the future will be empty.
     */
    Future pop_exception_handler(Context ctx, const char* provenance = nullptr);
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
    Mapping::Mapper* get_mapper(
        Context ctx, MapperID id, Processor target = Processor::NO_PROC);

    /**
     * Start a mapper call from the application side. This will create
     * a mapper context to use during the mapper call. The creation of
     * this mapper context will ensure appropriate synchronization with
     * other mapper calls consistent with the mapper synchronization model.
     * @param ctx the enclosing task context
     * @param id the mapper ID for which mapper to locate
     * @param target processor if any, if none specified then
     *               the executing processor for the current
     *               context is used, if specified processor
     *               must be local to the address space
     * @return a fresh mapper context to use for the mapper call
     */
    Mapping::MapperContext begin_mapper_call(
        Context ctx, MapperID id, Processor target = Processor::NO_PROC);

    /**
     * End a mapper call from the application side. This must be done for
     * all mapper contexts created by calls into begin_mapper_call.
     * @param ctx mapper context to end
     */
    void end_mapper_call(Mapping::MapperContext ctx);

    /**
     * Return the processor on which the current task is
     * being executed.
     * @param ctx enclosing task context
     * @return the processor on which the task is running
     */
    Processor get_executing_processor(Context ctx);

    /**
     * Return a pointer to the task object for the currently
     * executing task.
     * @param ctx enclosing task context
     * @return a pointer to the Task object for the current task
     */
    const Task* get_current_task(Context ctx);

    /**
     * Query the space available to this task in a given memory.
     * This is an instantaneous value and may be subject to change.
     * If the mapper has provided an upper bound for a pool in this
     * memory then it will reflect how much space is left available
     * in that pool, otherwise it will reflect the space left in the
     * actual memory. Note that the space available does not imply
     * that you can create an instance of this size as the memory
     * may be fragmented and the largest hole might be much smaller
     * than the size returned by this function.
     * @param ctx enclosing task context
     * @param target the memory being queried
     * @param escaping query the escaping or non-escaping pool
     * @return the instantaneous remaining size in the target memory
     */
    size_t query_available_memory(
        Context ctx, Memory target, bool escaping = true);

    /**
     * Inform the runtime that a task is done performing memory
     * allocations in a given memory ahead of the completion of
     * the task. This will allow the runtime to free up the memory
     * pool for additional allocations earlier than waiting for
     * the completion of the task. Note that you can only release
     * escaping pools of memories as the non-escaping pools have
     * already appeared to have freed up their memory.
     * @param ctx enclosing task context
     * @param memory the memory in which allocations are finished
     */
    void release_memory_pool(Context ctx, Memory target);

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
    void raise_region_exception(
        Context ctx, PhysicalRegion region, bool nuclear);

    /**
     * Yield the task to allow other tasks on the processor. In most
     * Legion programs calling this should never be necessary. However,
     * sometimes an application may want to put its own polling loop
     * inside a task. If it does it may need to yield the processor that
     * it is running on to allow other tasks to run on that processor.
     * This can be accomplished by invoking this method. The task will
     * be pre-empted and other eligible tasks will be permitted to run on
     * this processor.
     */
    void yield(Context ctx);

    /**
     * Record an asynchronous completion effect for this task. An
     * "asynchronous effect" can be whatever you want it to be as long
     * as Realm permits you to make an event for it. For example it
     * could be a CUDA event or a CUDA stream that you have converted to
     * a Realm event and you want to make sure any asynchronous GPU work
     * represented by one of those primitives is reflected in the
     * completion of this task eventhough you're going to exit the
     * host-side execution of this task before it is finished. Legion
     * will make sure this event is reflected in the complete of the
     * task meaning the task won't be considered done until all its
     * effects are done as well. You technically don't have to use this
     * API for correctness; you can invoke the Realm version of this
     * API directly as well, but it will not work with implicit top-level
     * tasks and you also will not get support for analyzing these efffects
     * as part of Legion's critical path infrastructure without using it.
     */
    void record_asynchronous_effect(
        Context ctx, Realm::Event effect, const char* provenance = nullptr);

    /**
     * This method provides a mechanism for performing a blocking barrier
     * inside the point tasks of concurrent index space task launch. This
     * may seem very un-Legion-like and indeed it is. However, there is one
     * very important use case that we've identified where it is imperative
     * that we have such a feature and there's really no good way to work
     * around the issue at the moment other than to provide this feature.
     *
     * Launching collective kernels on a GPU is currently an unsafe thing
     * to do (see this section of the CUDA programming guide:
     * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams
     * "for example, inter-kernel communication is undefined") which means
     * that technically using collective kernel libraries such as NCCL is
     * illegal in CUDA programs. To help make this safer, we recommend putting
     * a barrier both before and after every single collective kernel launch
     * performed in a concurrent index space task launch. Yes, we know this
     * sucks and it will probably hurt your performance. Please raise issues
     * with the NVIDIA CUDA team.
     *
     * To use this method all variants selected by each point task must have
     * set the 'concurrent_barrier' flag when they were registered which tells
     * the runtime to make this barrier available. The runtime will check that
     * all variants in the concurrent index space task launch have this set.
     * It will raise an error if any of the selected variants do not have this
     * set (as this probably means you have some points that are going to
     * expect to arrive on the barrier while others will not). This method will
     * perform a barrier across the N tasks in the concurrent index space task
     * launch. The expected cost of this barrier is O(log N) in the number of
     * tasks N in the collective index space task launch.
     *
     * When using this barrier to address the CUDA issue described above: it
     * is the user's responsibility to make sure that one barrier is
     * performed before the kernel is launched and one is performed after the
     * launch in order to be avoid deadlocks. Furthermore, it is the user's
     * responsibility to make sure that the kernel has actually been issued
     * to GPU driver (be very careful with non-blocking communicators).
     */
    void concurrent_task_barrier(Context ctx);

    /**
     * Start an application profiling range in this context. This must be
     * matched with a corresponding stop application profiling range
     * call in the same context. You can nest application profiling ranges
     * but it is up to the user to make sure that no ABAB patterns occur
     * or the boxes will not appear the way the user intended. If profiling
     * is not enabled then these calls will be no-ops. The provenance string
     * passed to the stop call is a normal provenance string and can have
     * both a human and machine components separated by a delimiter that will
     * be parsed by the profiler.
     */
    void start_profiling_range(Context ctx);
    void stop_profiling_range(Context ctx, const char* provenance);
  public:
    //------------------------------------------------------------------------
    // MPI Interoperability
    //------------------------------------------------------------------------
    /**
     * @return true if the MPI interop has been established
     */
    bool is_MPI_interop_configured(void);

    /**
     * Return a reference to the mapping from MPI ranks to address spaces.
     * This method is only valid if the static initialization method
     * 'configure_MPI_interoperability' was called on all nodes before
     * starting the runtime with the static 'start' method.
     * @return a const reference to the forward map
     */
    const std::map<int /*rank*/, AddressSpace>& find_forward_MPI_mapping(void);

    /**
     * Return a reference to the reverse mapping from address spaces to
     * MPI ranks. This method is only valid if the static initialization
     * method 'configure_MPI_interoperability' was called on all nodes
     * before starting the runtime with the static 'start' method.
     * @return a const reference to the reverse map
     */
    const std::map<AddressSpace, int /*rank*/>& find_reverse_MPI_mapping(void);

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
     * @param local_only attach the name only on this node
     */
    void attach_semantic_information(
        TaskID task_id, SemanticTag tag, const void* buffer, size_t size,
        bool is_mutable = false, bool local_only = false);

    /**
     * Attach semantic information to an index space
     * @param handle index space handle
     * @param tag semantic tag
     * @param buffer pointer to a buffer
     * @param size size of the buffer to save
     * @param is_mutable can the tag value be changed later
     */
    void attach_semantic_information(
        IndexSpace handle, SemanticTag tag, const void* buffer, size_t size,
        bool is_mutable = false);

    /**
     * Attach semantic information to an index partition
     * @param handle index partition handle
     * @param tag semantic tag
     * @param buffer pointer to a buffer
     * @param size size of the buffer to save
     * @param is_mutable can the tag value be changed later
     */
    void attach_semantic_information(
        IndexPartition handle, SemanticTag tag, const void* buffer, size_t size,
        bool is_mutable = false);

    /**
     * Attach semantic information to a field space
     * @param handle field space handle
     * @param tag semantic tag
     * @param buffer pointer to a buffer
     * @param size size of the buffer to save
     * @param is_mutable can the tag value be changed later
     */
    void attach_semantic_information(
        FieldSpace handle, SemanticTag tag, const void* buffer, size_t size,
        bool is_mutable = false);

    /**
     * Attach semantic information to a specific field
     * @param handle field space handle
     * @param fid field ID
     * @param tag semantic tag
     * @param buffer pointer to a buffer
     * @param size size of the buffer to save
     * @param is_mutable can the tag value be changed later
     */
    void attach_semantic_information(
        FieldSpace handle, FieldID fid, SemanticTag tag, const void* buffer,
        size_t size, bool is_mutable = false);

    /**
     * Attach semantic information to a logical region
     * @param handle logical region handle
     * @param tag semantic tag
     * @param buffer pointer to a buffer
     * @param size size of the buffer to save
     * @param is_mutable can the tag value be changed later
     */
    void attach_semantic_information(
        LogicalRegion handle, SemanticTag tag, const void* buffer, size_t size,
        bool is_mutable = false);

    /**
     * Attach semantic information to a logical partition
     * @param handle logical partition handle
     * @param tag semantic tag
     * @param buffer pointer to a buffer
     * @param size size of the buffer to save
     * @param is_mutable can the tag value be changed later
     */
    void attach_semantic_information(
        LogicalPartition handle, SemanticTag tag, const void* buffer,
        size_t size, bool is_mutable = false);

    /**
     * Attach a name to a task
     * @param task_id the ID of the task
     * @param name pointer to the name
     * @param is_mutable can the name be changed later
     * @param local_only attach the name only on the local node
     */
    void attach_name(
        TaskID task_id, const char* name, bool is_mutable = false,
        bool local_only = false);

    /**
     * Attach a name to an index space
     * @param handle index space handle
     * @param name pointer to a name
     * @param is_mutable can the name be changed later
     */
    void attach_name(
        IndexSpace handle, const char* name, bool is_mutable = false);

    /**
     * Attach a name to an index partition
     * @param handle index partition handle
     * @param name pointer to a name
     * @param is_mutable can the name be changed later
     */
    void attach_name(
        IndexPartition handle, const char* name, bool is_mutable = false);

    /**
     * Attach a name to a field space
     * @param handle field space handle
     * @param name pointer to a name
     * @param is_mutable can the name be changed later
     */
    void attach_name(
        FieldSpace handle, const char* name, bool is_mutable = false);

    /**
     * Attach a name to a specific field
     * @param handle field space handle
     * @param fid field ID
     * @param name pointer to a name
     * @param is_mutable can the name be changed later
     */
    void attach_name(
        FieldSpace handle, FieldID fid, const char* name,
        bool is_mutable = false);

    /**
     * Attach a name to a logical region
     * @param handle logical region handle
     * @param name pointer to a name
     * @param is_mutable can the name be changed later
     */
    void attach_name(
        LogicalRegion handle, const char* name, bool is_mutable = false);

    /**
     * Attach a name to a logical partition
     * @param handle logical partition handle
     * @param name pointer to a name
     * @param is_mutable can the name be changed later
     */
    void attach_name(
        LogicalPartition handle, const char* name, bool is_mutable = false);

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
    bool retrieve_semantic_information(
        TaskID task_id, SemanticTag tag, const void*& result, size_t& size,
        bool can_fail = false, bool wait_until_ready = false);

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
    bool retrieve_semantic_information(
        IndexSpace handle, SemanticTag tag, const void*& result, size_t& size,
        bool can_fail = false, bool wait_until_ready = false);

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
    bool retrieve_semantic_information(
        IndexPartition handle, SemanticTag tag, const void*& result,
        size_t& size, bool can_fail = false, bool wait_until_ready = false);

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
    bool retrieve_semantic_information(
        FieldSpace handle, SemanticTag tag, const void*& result, size_t& size,
        bool can_fail = false, bool wait_until_ready = false);

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
    bool retrieve_semantic_information(
        FieldSpace handle, FieldID fid, SemanticTag tag, const void*& result,
        size_t& size, bool can_fail = false, bool wait_until_ready = false);

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
    bool retrieve_semantic_information(
        LogicalRegion handle, SemanticTag tag, const void*& result,
        size_t& size, bool can_fail = false, bool wait_until_ready = false);

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
    bool retrieve_semantic_information(
        LogicalPartition handle, SemanticTag tag, const void*& result,
        size_t& size, bool can_fail = false, bool wait_until_ready = false);

    /**
     * Retrieve the name of a task
     * @param task_id the ID of the task
     * @param result pointer to assign to the name
     */
    void retrieve_name(TaskID task_id, const char*& result);

    /**
     * Retrieve the name of an index space
     * @param handle index space handle
     * @param result pointer to assign to the name
     */
    void retrieve_name(IndexSpace handle, const char*& result);

    /**
     * Retrieve the name of an index partition
     * @param handle index partition handle
     * @param result pointer to assign to the name
     */
    void retrieve_name(IndexPartition handle, const char*& result);

    /**
     * Retrieve the name of a field space
     * @param handle field space handle
     * @param result pointer to assign to the name
     */
    void retrieve_name(FieldSpace handle, const char*& result);

    /**
     * Retrieve the name of a specific field
     * @param handle field space handle
     * @param fid field ID
     * @param result pointer to assign to the name
     */
    void retrieve_name(FieldSpace handle, FieldID fid, const char*& result);

    /**
     * Retrieve the name of a logical region
     * @param handle logical region handle
     * @param result pointer to assign to the name
     */
    void retrieve_name(LogicalRegion handle, const char*& result);

    /**
     * Retrieve the name of a logical partition
     * @param handle logical partition handle
     * @param result pointer to assign to the name
     */
    void retrieve_name(LogicalPartition handle, const char*& result);
  public:
    //------------------------------------------------------------------------
    // Printing operations, these are useful for only generating output
    // from a single task if the task has been replicated (either directly
    // or as part of control replication).
    //------------------------------------------------------------------------
    /**
     * Print the string to the given C file (may also be stdout/stderr)
     * exactly once regardless of the replication status of the task.
     * @param ctx the enclosing task context
     * @param file the file to be written to
     * @param message pointer to the C string to be written
     */
    void print_once(Context ctx, FILE* f, const char* message);

    /**
     * Print the logger message exactly once regardless of the control
     * replication status of the task.
     * @param ctx the enclosing task context
     * @param message the Realm Logger Message to be logged
     */
    void log_once(Context ctx, Realm::LoggerMessage& message);
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
     * Generate a contiguous set of MapperIDs for use by a library.
     * This call will always generate the same answer for the same library
     * name no many how many times it is called or on how many nodes it
     * is called. If the count passed in to this method differs for the
     * same library name the runtime will raise an error.
     * @param name a unique null-terminated string that names the library
     * @param count the number of mapper IDs that should be generated
     * @return the first mapper ID that is allocated to the library
     */
    MapperID generate_library_mapper_ids(const char* name, size_t count);

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
    void add_mapper(
        MapperID map_id, Mapping::Mapper* mapper,
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
    void replace_default_mapper(
        Mapping::Mapper* mapper, Processor proc = Processor::NO_PROC);
  public:
    /**
     * Dynamically generate a unique projection ID for use across the machine
     * @reutrn a ProjectionID that is globally unique across the machine
     */
    ProjectionID generate_dynamic_projection_id(void);

    /**
     * Generate a contiguous set of ProjectionIDs for use by a library.
     * This call will always generate the same answer for the same library
     * name no many how many times it is called or on how many nodes it
     * is called. If the count passed in to this method differs for the
     * same library name the runtime will raise an error.
     * @param name a unique null-terminated string that names the library
     * @param count the number of projection IDs that should be generated
     * @return the first projection ID that is allocated to the library
     */
    ProjectionID generate_library_projection_ids(
        const char* name, size_t count);

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
     * @param silence_warnings disable warnings about dynamic registration
     * @param warning_string a string to be reported with any warnings
     */
    void register_projection_functor(
        ProjectionID pid, ProjectionFunctor* functor,
        bool silence_warnings = false, const char* warning_string = nullptr);

    /**
     * Register a projection functor before the runtime has started only.
     * The runtime will update the projection functor so that it has
     * contains a valid runtime pointer prior to the projection functor
     * ever being invoked. The runtime takes ownership for deleting the
     * projection functor after the application has finished executing.
     * @param pid the projection ID to use for the registration
     * @param functor the object to register for handling projections
     */
    static void preregister_projection_functor(
        ProjectionID pid, ProjectionFunctor* functor);

    /**
     * Return a pointer to a given projection functor object.
     * The runtime retains ownership of this object.
     * @param pid ID of the projection functor to find
     * @return a pointer to the projection functor if it exists
     */
    static ProjectionFunctor* get_projection_functor(ProjectionID pid);
  public:
    /**
     * Dynamically generate a unique sharding ID for use across the machine
     * @return a ShardingID that is globally unique across the machine
     */
    ShardingID generate_dynamic_sharding_id(void);

    /**
     * Generate a contiguous set of ShardingIDs for use by a library.
     * This call will always generate the same answer for the same library
     * name no many how many times it is called or on how many nodes it
     * is called. If the count passed in to this method differs for the
     * same library name the runtime will raise an error.
     * @param name a unique null-terminated string that names the library
     * @param count the number of sharding IDs that should be generated
     * @return the first sharding ID that is allocated to the library
     */
    ShardingID generate_library_sharding_ids(const char* name, size_t count);

    /**
     * Statically generate a unique Sharding ID for use across the machine.
     * This can only be called prior to the runtime starting. It must be
     * invoked symmetrically across all the nodes in the machine prior
     * to starting the runtime.
     * @return ShardingID that is globally unique across the machine
     */
    static ShardingID generate_static_sharding_id(void);

    /**
     * Register a sharding functor for handling control replication
     * queries about which shard owns which a given point in an
     * index space launch. The ShardingID must be non-zero because
     * zero is the special "round-robin" sharding functor. The
     * runtime takes ownership of for deleting the sharding functor
     * after the application has finished executing.
     * @param sid the sharding ID to use for the registration
     * @param functor the object to register for handling sharding requests
     * @param silence_warnings disable warnings about dynamic registration
     * @param warning_string a string to be reported with any warnings
     */
    void register_sharding_functor(
        ShardingID sid, ShardingFunctor* functor, bool silence_warnings = false,
        const char* warning_string = nullptr);

    /**
     * Register a sharding functor before the runtime has
     * started only. The sharding functor will be invoked to
     * handle queries during control replication about which
     * shard owns a given point in an index space launch. The
     * runtime takes ownership for deleting the sharding functor
     * after the application has finished executing.
     * @param sid the sharding ID to use for the registration
     * @param functor the object too register for handling sharding
     */
    static void preregister_sharding_functor(
        ShardingID sid, ShardingFunctor* functor);

    /**
     * Return a pointer to a given sharding functor object.
     * The runtime retains ownership of this object.
     * @param sid ID of the sharding functor to find
     * @return a pointer o the sharding functor if it exists
     */
    static ShardingFunctor* get_sharding_functor(ShardingID sid);
  public:
    /**
     * Dynamically generate a unique concurrent ID for use across the machine
     * @return a ConcurrentID that is globally unique across the machine
     */
    ConcurrentID generate_dynamic_concurrent_id(void);

    /**
     * Generate a contiguous set of ConcurrentIDs for use by a library.
     * This call will always generate the same answer for the same library
     * name no many how many times it is called or on how many nodes it
     * is called. If the count passed in to this method differs for the
     * same library name the runtime will raise an error.
     * @param name a unique null-terminated string that names the library
     * @param count the number of concurrent IDs that should be generated
     * @return the first concurrent ID that is allocated to the library
     */
    ConcurrentID generate_library_concurrent_ids(
        const char* name, size_t count);

    /**
     * Statically generate a unique Concurrent ID for use across the machine.
     * This can only be called prior to the runtime starting. It must be
     * invoked symmetrically across all the nodes in the machine prior
     * to starting the runtime.
     * @return ConcurrentID that is globally unique across the machine
     */
    static ConcurrentID generate_static_concurrent_id(void);

    /**
     * Register a concurrent coloring functor for handling grouping of
     * concurrent index space task launches into subsets of points that
     * can execute concurrently. The ConcurrentID must be non-zero because
     * zero is the special built-in "map all points to the same color"
     * functor which should be the default for most concurrent index
     * space task launches. The runtime takes ownership of the functor and
     * will delete it upon runtime shutdown.
     * @param cid the concurrent ID to use for the registration
     * @param functor the object to register for handling concurrent grouping
     * @param silence_warnings disable warnings about dynamic registration
     * @param warning_string a string to be reported with any warnings
     */
    void register_concurrent_coloring_functor(
        ConcurrentID cid, ConcurrentColoringFunctor* functor,
        bool silence_warnings = false, const char* warning_string = nullptr);

    /**
     * Register a concurrent coloring functor before the runtime has
     * started only. The concurrent coloring functor will be invoked to
     * group points in a concurrent index space task launch into
     * subsets of points that can be executed concurrently. The runtime
     * take ownership for the functor and will delete it upon shutdown.
     * @param cid the concurrent ID to use for the registration
     * @param functor the object to register for handling concurrent grouping
     */
    static void preregister_concurrent_coloring_functor(
        ConcurrentID cid, ConcurrentColoringFunctor* functor);

    /**
     * Return a pointer to a given concurrent coloring functor object.
     * The runtime retains ownership of this object.
     * @param cid ID of the concurrent coloring functor to find
     * @return a pointer to the concurrent coloring functor if it exists
     */
    static ConcurrentColoringFunctor* get_concurrent_coloring_functor(
        ConcurrentID cid);
  public:
    /**
     * Dynamically generate a unique exception handler ID for use across the
     * machine
     * @return a ExceptionHandlerID that is globally unique across the machine
     */
    ExceptionHandlerID generate_dynamic_exception_handler_id(void);

    /**
     * Generate a contiguous set of ExceptionHandlerIDs for use by a library.
     * This call will always generate the same answer for the same library
     * name no many how many times it is called or on how many nodes it
     * is called. If the count passed in to this method differs for the
     * same library name the runtime will raise an error.
     * @param name a unique null-terminated string that names the library
     * @param count the number of exception handler IDs that should be generated
     * @return the first exception handler ID that is allocated to the library
     */
    ExceptionHandlerID generate_library_exception_handler_ids(
        const char* name, size_t count);

    /**
     * Statically generate a unique ExceptionHandler ID for use across the
     * machine. This can only be called prior to the runtime starting. It must
     * be invoked symmetrically across all the nodes in the machine prior to
     * starting the runtime.
     * @return ExceptionHandlerID that is globally unique across the machine
     */
    static ExceptionHandlerID generate_static_exception_handler_id(void);

    /**
     * Register a exception handler for handling exceptions from tasks.
     * @param hid the exception handler ID to use for the registration
     * @param handler the object to register for handling exceptions
     */
    void register_exception_handler(
        ExceptionHandlerID hid, ExceptionHandler* handler);

    /**
     * Register an exception handler before the runtime has
     * started only. The runtime take ownership for the handler
     * and will delete it upon shutdown.
     * @param hid the exception handler ID to use for the registration
     * @param handler the object to register for handling exceptions
     */
    static void preregister_exception_handler(
        ExceptionHandlerID cid, ExceptionHandler* handler);

    /**
     * Return a pointer to a given exception handler object.
     * The runtime retains ownership of this object.
     * @param hid ID of the exception handler to find
     * @return a pointer to the exception handler if it exists
     */
    static ExceptionHandler* get_exception_handler(ExceptionHandlerID hid);
  public:
    /**
     * Dynamically generate a unique reduction ID for use across the machine
     * @return a ReductionOpID that is globally unique across the machine
     */
    ReductionOpID generate_dynamic_reduction_id(void);

    /**
     * Generate a contiguous set of ReductionOpIDs for use by a library.
     * This call will always generate the same answer for the same library
     * name no many how many times it is called or on how many nodes it
     * is called. If the count passed in to this method differs for the
     * same library name the runtime will raise an error.
     * @param name a unique null-terminated string that names the library
     * @param count the number of reduction IDs that should be generated
     * @return the first reduction ID that is allocated to the library
     */
    ReductionOpID generate_library_reduction_ids(
        const char* name, size_t count);

    /**
     * Statically generate a unique reduction ID for use across the machine.
     * This can only be called prior to the runtime starting. It must be
     * invoked symmetrically across all the nodes in the machine prior
     * to starting the runtime.
     * @return a ReductionOpID that is globally unique across the machine
     */
    static ReductionOpID generate_static_reduction_id(void);

    /**
     * Register a reduction operation with the runtime. Note that the
     * reduction operation template type must conform to the specification
     * for a reduction operation outlined in the Realm runtime
     * interface.  Reduction operations can be used either for reduction
     * privileges on a region or for performing reduction of values across
     * index space task launches. The reduction operation ID zero is
     * reserved for runtime use. Note that even though this method is
     * a static method it can be called either before or after the runtime
     * has been started safely.
     * @param redop_id ID at which to register the reduction operation
     * @param permit_duplicates will allow a duplicate registration to
     *        be successful if this reduction ID has been used before
     */
    template<typename REDOP>
    static void register_reduction_op(
        ReductionOpID redop_id, bool permit_duplicates = false);

    /**
     * Register an untyped reduction operation with the runtime. Note
     * that the reduction operation template type must conform to the
     * specification for a reduction operation outlined in the Realm runtime
     * interface. Reduction operations can be used either for reduction
     * privileges on a region or for performing reduction of values across
     * index space task launches. The reduction operation ID zero is
     * reserved for runtime use. The runtime will take ownership of this
     * operation and delete it at the end of the program. Note that eventhough
     * this is a static method it can be called either before or after the
     * runtime has been started safely.
     * @param redop_id ID at which to register the reduction opeation
     * @param op the untyped reduction operator (legion claims ownership)
     * @param init_fnptr optional function for initializing the reduction
     *        type of this reduction operator if they also support compression
     * @pram fold_fnptr optional function for folding reduction types of this
     *        reduction operator if they also support compression
     * @param permit_duplicates will allow a duplicate registration to
     *        be successful if this reduction ID has been used before
     */
    static void register_reduction_op(
        ReductionOpID redop_id, ReductionOp* op,
        SerdezInitFunc init_fnptr = nullptr,
        SerdezFoldFunc fold_fnptr = nullptr, bool permit_duplicates = false);

    /**
     * Return a pointer to a given reduction operation object.
     * @param redop_id ID of the reduction operation to find
     * @return a pointer to the reduction operation object if it exists
     */
    static const ReductionOp* get_reduction_op(ReductionOpID redop_id);

#ifdef LEGION_GPU_REDUCTIONS
    template<typename REDOP>
    LEGION_DEPRECATED("Use register_reduction_op instead")
    static void preregister_gpu_reduction_op(ReductionOpID redop_id);
#endif
  public:
    /**
     * Dynamically generate a unique serdez ID for use across the machine
     * @return a CustomSerdezID that is globally unique across the machine
     */
    CustomSerdezID generate_dynamic_serdez_id(void);

    /**
     * Generate a contiguous set of CustomSerdezIDs for use by a library.
     * This call will always generate the same answer for the same library
     * name no many how many times it is called or on how many nodes it
     * is called. If the count passed in to this method differs for the
     * same library name the runtime will raise an error.
     * @param name a unique null-terminated string that names the library
     * @param count the number of serdez IDs that should be generated
     * @return the first serdez ID that is allocated to the library
     */
    CustomSerdezID generate_library_serdez_ids(const char* name, size_t count);

    /**
     * Statically generate a unique serdez ID for use across the machine.
     * This can only be called prior to the runtime starting. It must be
     * invoked symmetrically across all the nodes in the machine prior
     * to starting the runtime.
     * @return a CustomSerdezID that is globally unique across the machine
     */
    static CustomSerdezID generate_static_serdez_id(void);

    /**
     * Register custom serialize/deserialize operation with the
     * runtime. This can be used for providing custom serialization
     * and deserialization method for fields that are not trivially
     * copied (e.g. byte-wise copies). The type being registered
     * must conform to the Realm definition of a CustomSerdez
     * object (see realm/custom_serdez.h). Note that eventhough
     * this is a static method you can safely call it both before
     * and after the runtime has been started.
     * @param serdez_id ID at which to register the serdez operator
     * @param permit_duplicates will allow a duplicate registration to
     *        be successful if this serdez ID has been used before
     */
    template<typename SERDEZ>
    static void register_custom_serdez_op(
        CustomSerdezID serdez_id, bool permit_duplicates = false);

    /**
     * Register custom serialize/deserialize operation with the
     * runtime. This can be used for providing custom serialization
     * and deserialization method for fields that are not trivially
     * copied (e.g. byte-wise copies). Note that eventhough this is
     * a static method you can safely call it both before and after
     * the runtime has been started.
     * @param serdez_id ID at which to register the serdez operator
     * @param serdez_op The functor for the serdez op
     * @param permit_duplicates will allow a duplicate registration to
     *        be successful if this serdez ID has been used before
     */
    static void register_custom_serdez_op(
        CustomSerdezID serdez_id, SerdezOp* serdez_op,
        bool permit_duplicates = false);

    /**
     * Return a pointer to the given custom serdez operation object.
     * @param serdez_id ID of the serdez operation to find
     * @return a pointer to the serdez operation object if it exists
     */
    static const SerdezOp* get_serdez_op(CustomSerdezID serdez_id);
  public:
    //------------------------------------------------------------------------
    // Start-up Operations
    // Everything below here is a static function that is used to configure
    // the runtime prior to calling the start method to begin execution.
    //------------------------------------------------------------------------
  public:
    /**
     * Return a string representing the Legion version. This string
     * can be compared in application code against the LEGION_VERSION
     * macro defined by legion.h to detect header/library mismatches.
     */
    static const char* get_legion_version(void);

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
     * -lg:vector <int> Set the initial vectorization option for fusing
     *              together important runtime meta tasks in the mapper.
     *              The default is 16.
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
     * -lg:safe_ctrlrepl <level> Perform dynamic checks to verify the
     *              correctness of control replication. This will compute a
     *              hash of all the arguments to each call into the runtime
     *              and perform a collective to compare it across
     *              the shards to see if they all align.
     *              Level 0: no checks
     *              Level 1: sound but incomplete checks (no false positives)
     *              Level 2: unsound but complete checks (no false negatives)
     * -lg:safe_tracing Request that the runtime check the invariants
     *              required for using tracing.
     * -lg:local <int> Specify the maximum number of local fields
     *              permitted in any field space within a context.
     * ---------------------
     *  Tracing
     * ---------------------
     * -lg:no_tracing Disable execution with tracing of any kind. All calls
     *              to begin and end traces will be ignored.
     * -lg:no_physical_tracing Disable physical tracing. All calls to begin
     *              and end traces will only be performed with regards to
     *              logical dependence analysis.
     * -lg:no_auto_tracing Disable auto-tracing by default so that Legion
     *              is not trying to find traces inside tasks. Mappers can
     *              still override this default value and set it to true
     *              to opt-in to auto-tracing
     * -lg:no_trace_optimization Turn off all trace optimizations
     * -lg:no_fence_elision Turn off the fence elision optimization that
     *              improves the performance of back-to-back idempotent
     *              trace replays
     * -lg:no_transitive_reduction Turn off the transitive reduction
     *              optimization that is performed on captured traces
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
     * -lg:warn_backtrace Print a backtrace for each warning
     * -lg:werror   Promote all legion warnings into errors
     * -lg:leaks    Report information about resource leaks
     * -lg:ldb <replay_file> Replay the execution of the application
     *              with the associated replay file generted by LegionSpy.
     *              This will run the application in the Legion debugger.
     * -lg:replay <replay_file> Rerun the execution of the application with
     *              the associated replay file generated by LegionSpy.
     * -lg:disjointness Verify the specified disjointness of partitioning
     *              operations. This flag is now a synonym for -lg:partcheck
     * -lg:partcheck This flag will ask the runtime to dynamically verify
     *              that all correctness properties for partitions are
     *              upheld. This includes checking that the parent region
     *              dominates all subregions and that all annotations of
     *              disjointness and completeness from the user are correct.
     *              This is an expensive test and users should expect a
     *              significant slow-down of their application when using it.
     * -lg:registration Record the mapping from Realm task IDs to
     *              task variant names for debugging Realm runtime
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
     * -lg:spy <int> Enable Legion Spy logging for off-load postprocessing
     *              by the Legion Spy script. Legion Spy is valuable for
     *              understanding properties of an application such as the
     *              shapes of region trees and the kinds of tasks/operations
     *              that are created. Formal verification of the Legion
     *              runtime can also be performed. Different logging levels
     *              are required for different properites.
     *              Level 0: No logging.
     *              Level 1: Light-weight logging sufficient for generating
     *                       pictures of execution graphs and region trees
     *              Level 2: Heavy-weight logging needed for formal
     *                       verification of the runtime itself but will
     *                       not require any Legion Spy emulation.
     *              Level 3: Logging of equivalence sets for analysis of
     *                       how well the runtime heurisitics for
     *                       dependence analysis are working.
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
     * -lg:prof_call_threshold <int> The minimum size of runtime and
     *              mapper calls in order for them to be logged by the
     *              profiler in microseconds. All runtime and mapper calls
     *              that are less than this threshold will be discarded
     *              and will not be recorded in the profiling logs. The
     *              default value is 1 (us) so all calls that take less
     *              than 1 us are elided.
     * -lg:prof_self Perform self-profiling so that the profiling
     *              response meta-tasks are also recorded in the profile.
     *              In general these are tiny and not worth profiling,
     *              but you might still want to see them. They are not
     *              recorded by default.
     * -lg:prof_critical_paths Enable critical path logging which will
     *              capture Realm event graph structures. This may add
     *              50% overhead to execution and increase profile log
     *              sizes by a multiplicative factor.
     * -lg:prof_no_critical_paths Disable logging for performing critial
     *              path analysis as it is can greatly increase the size
     *              of the Legion Prof log files
     *
     * @param argc the number of input arguments
     * @param argv pointer to an array of string arguments of size argc
     * @param background whether to execute the runtime in the background
     * @param supply_default_mapper whether the runtime should initialize
     *              the default mapper for use by the application
     * @param filter filter legion and realm command line arguments
     * @param realm_initialize_network invoke Realm's network initialization
     * @param realm_configure_machine finish configuring the Realm machine model
     *              from the command line if not previously configured. If you
     *              set this to false then you must ensure that the Realm
     *              machine model is fully configured before invoking the
     *              Legion start method.
     * @return only if running in background, otherwise never
     */
    static int start(
        int argc, char** argv, bool background = false,
        bool supply_default_mapper = true, bool filter = false,
        bool realm_initialize_network = true,
        bool realm_configure_machine = true);

    /**
     * This 'initialize' method is an optional method that provides
     * users a way to look at the command line arguments before they
     * actually start the Legion runtime. Users will still need to
     * call 'start' in order to actually start the Legion runtime but
     * this way they can do some static initialization and use their
     * own command line parameters to initialize the runtime prior
     * to actually starting it. The resulting 'argc' and 'argv' should
     * be passed into the 'start' method or undefined behavior will occur.
     * @param argc pointer to an integer in which to store the argument count
     * @param argv pointer to array of strings for storing command line args
     * @param filter remove any legion and realm command line arguments
     * @param parse parse any runtime command line arguments during this call
     *              (if set to false parsing happens during start method)
     * @param realm_initialize_network invoke Realm's network initialization
     * @param realm_configure_machine configure the Realm machine model
     *              from the command line if not previously configured. Note
     *              that if you set this to false you must do any Realm
     *              machine model configuration calls yourself.
     */
    static void initialize(
        int* argc, char*** argv, bool filter = false, bool parse = true,
        bool realm_initialize_network = true,
        bool realm_create_configure_machine = true);

    /**
     * Blocking call to wait for the runtime to shutdown when
     * running in background mode.  Otherwise it is illegal to
     * invoke this method. Returns the exit code for the application.
     */
    static int wait_for_shutdown(void);

    /**
     * Set the return code for the application from Legion.
     * This will be returned as the result from 'start' or
     * 'wait_for_shutdown'. The default is zero. If multiple
     * non-zero values are set then at least one of the non-zero
     * values will be returned.
     */
    static void set_return_code(int return_code);

    /**
     * Set the top-level task ID for the runtime to use when beginning
     * an application.  This should be set before calling start. If no
     * top-level task ID is set then the runtime will not start running
     * any tasks at start-up.
     * @param top_id ID of the top level task to be run
     */
    static void set_top_level_task_id(TaskID top_id);

    /**
     * Set the mapper ID for the runtime to use when starting the
     * top-level task. This can be called either before the runtime
     * is started, or during the registration callback, but will
     * have no effect after the top-level task is started.
     */
    static void set_top_level_task_mapper_id(MapperID mapper_id);

    /**
     * After the runtime is started, users can launch as many top-level
     * tasks as they want using this method. Each one will start a new
     * top-level task and returns values with a future. Currently we
     * only permit this to be called from threads not managed by Legion.
     */
    Future launch_top_level_task(const TaskLauncher& launcher);

    /**
     * In addition to launching top-level tasks from outside the runtime,
     * applications can bind external threads as new implicit top-level
     * tasks to the runtime. This will tell the runtime that this external
     * thread should now function as new top-level task that is executing.
     * After this call the trhead will be treated as through it is a
     * top-level task running on a specific kind of processor. Users can
     * also mark that this implicit top-level task is control replicable
     * for supporting implicit top-level tasks for multi-node runs. For
     * the control replicable case we expect to see the same number of
     * calls from every address space. This number is controlled by
     * shard_per_address_space and defaults to one. The application can
     * also optionally specify the shard ID for every implicit top level
     * task for control replication settings. If it is specified, then
     * the application must specify it for all shards. Otherwise the
     * runtime will allocate shard IDs contiguously on each node before
     * proceeding to the next node.
     */
    Context begin_implicit_task(
        TaskID top_task_id, MapperID mapper_id,
        Processor::Kind proc_kind = Processor::NO_KIND,
        const char* task_name = nullptr, bool control_replicable = false,
        unsigned shard_per_address_space = 1, int shard_id = -1,
        DomainPoint shard_point = DomainPoint());

    /**
     * Unbind an implicit context from the external thread it is
     * currently associated with. It is the user's responsibility
     * to make sure that no more than one external thread is bound
     * to an implicit task's context at a time or undefined
     * behavior will occur.
     */
    void unbind_implicit_task_from_external_thread(Context ctx);

    /**
     * Bind an implicit context to an external thread.
     * It is the user's responsibility to make sure that no more
     * than one external thread is bound to an implicit task's context
     * at a time or undefined behavior will occur.
     */
    void bind_implicit_task_to_external_thread(Context ctx);

    /**
     * This is the final method for marking the end of an
     * implicit top-level task. If there are any asynchronous effects
     * that were launched during the implicit top-level task (such as
     * a CUDA kernel launch) then users are required to capture all
     * those effects as a Realm event to tell Legion when all those
     * effects are completed. Finishing an implicit top-level task
     * still requires waiting explicitly for the runtime to shutdown.
     * The Context object is no longer valid after this call.
     */
    void finish_implicit_task(
        Context ctx, Realm::Event effects = Realm::Event::NO_EVENT);

    /**
     * Return the maximum number of dimensions that Legion was
     * configured to support in this build.
     * @return the maximum number of dimensions that Legion supports
     */
    static size_t get_maximum_dimension(void);

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
     * Create a handshake object for exchanging control between an
     * external application and Legion. We make this a static method so that
     * it can be created before the Legion runtime is initialized.
     * @param init_in_ext who owns initial control of the handshake,
     *                    by default it is the external application
     * @param ext_participants number of calls that need to be made to the
     *                    handshake to pass control from the external
     *                    application to Legion
     * @param legion_participants number of calls that need to be made to
     *                    the handshake to pass control from Legion to the
     *                    external application
     */
    static LegionHandshake create_external_handshake(
        bool init_in_ext = true, int ext_participants = 1,
        int legion_participants = 1);

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
    static MPILegionHandshake create_handshake(
        bool init_in_MPI = true, int mpi_participants = 1,
        int legion_participants = 1);

    /**
     * @deprecated
     * Register a region projection function that can be used to map
     * from an upper bound of a logical region down to a specific
     * logical sub-region for a given domain point during index
     * task execution.  The projection ID zero is reserved for runtime
     * use.
     * @param handle the projection ID to register the function at
     * @return ID where the function was registered
     */
    template<
        LogicalRegion (*PROJ_PTR)(LogicalRegion, const DomainPoint&, Runtime*)>
    LEGION_DEPRECATED(
        "Projection functions should now be specified "
        "using projection functor objects")
    static ProjectionID register_region_function(ProjectionID handle);

    /**
     * @deprecated
     * Register a partition projection function that can be used to
     * map from an upper bound of a logical partition down to a specific
     * logical sub-region for a given domain point during index task
     * execution.  The projection ID zero is reserved for runtime use.
     * @param handle the projection ID to register the function at
     * @return ID where the function was registered
     */
    template<LogicalRegion (*PROJ_PTR)(
        LogicalPartition, const DomainPoint&, Runtime*)>
    LEGION_DEPRECATED(
        "Projection functions should now be specified "
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
     * @param buffer optional argument buffer to pass to the callback
     * @param dedup whether to deduplicate this with other registration
     *              callbacks for the same function
     * @param dedup_tag a tag to use for deduplication in the case where
     *              applications may want to deduplicate across multiple
     *              callbacks with the same function pointer
     */
    static void add_registration_callback(
        RegistrationCallback callback, bool dedup = true, size_t dedup_tag = 0);
    static void add_registration_callback(
        RegistrationWithArgsCallback callback, const UntypedBuffer& buffer,
        bool dedup = true, size_t dedup_tag = 0);

    /**
     * This call allows applications to request a registration callback
     * be performed after the runtime has started. The application can
     * select whether this registration is performed locally (e.g. once
     * on the local node) or globally across all nodes in the machine.
     * The method will not return until the registration has been performed
     * on all the target address spaces. All function pointers passed into
     * this method with 'global' set to true must "portable", meaning that
     * we can lookup their shared object name and symbol name. This means
     * they either need to originate with a shared object or the binary
     * must be linked with '-rdynamic'. It's up the user to guarantee this
     * or Legion will raise an error about a non-portable function pointer.
     * For any given function pointer all calls must be made with the same
     * value of 'global' or hangs can occur.
     * @param ctx enclosing task context
     * @param global whether this registration needs to be performed
     *               in all address spaces or just the local one
     * @param buffer optional buffer of data to pass to callback
     * @param dedup whether to deduplicate this with other registration
     *              callbacks for the same function
     * @param dedup_tag a tag to use for deduplication in the case where
     *              applications may want to deduplicate across multiple
     *              callbacks with the same function pointer
     */
    static void perform_registration_callback(
        RegistrationCallback callback, bool global, bool deduplicate = true,
        size_t dedup_tag = 0);
    static void perform_registration_callback(
        RegistrationWithArgsCallback callback, const UntypedBuffer& buffer,
        bool global, bool deduplicate = true, size_t dedup_tag = 0);

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
    LEGION_DEPRECATED(
        "Legion now supports multiple registration callbacks "
        "added via the add_registration_callback method.")
    static void set_registration_callback(RegistrationCallback callback);

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
        const LayoutConstraintRegistrar& registrar);

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
        const LayoutConstraintRegistrar& registrar,
        LayoutConstraintID layout_id = LEGION_AUTO_GENERATE_ID);

    /**
     * Get the field space for a specific layout description
     * @param layout_id the layout ID for which to obtain the field space
     * @return the field space handle for the layout description
     */
    FieldSpace get_layout_constraint_field_space(LayoutConstraintID layout_id);

    /**
     * Get the constraints for a specific layout description
     * @param layout_id the layout ID for which to obtain the constraints
     * @param layout_constraints a LayoutConstraintSet to populate
     */
    void get_layout_constraints(
        LayoutConstraintID layout_id, LayoutConstraintSet& layout_constraints);

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
     * Generate a contiguous set of TaskIDs for use by a library.
     * This call will always generate the same answer for the same library
     * name no many how many times it is called or on how many nodes it
     * is called. If the count passed in to this method differs for the
     * same library name the runtime will raise an error.
     * @param name a unique null-terminated string that names the library
     * @param count the number of task IDs that should be generated
     * @return the first task ID that is allocated to the library
     */
    TaskID generate_library_task_ids(const char* name, size_t count);

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
     * @param vid optional variant ID to use
     * @return variant ID for the task
     */
    template<
        typename T,
        T (*TASK_PTR)(
            const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*)>
    VariantID register_task_variant(
        const TaskVariantRegistrar& registrar,
        VariantID vid = LEGION_AUTO_GENERATE_ID);

    /**
     * Dynamically register a new task variant with the runtime with
     * a non-void return type and user data.
     * @param registrar the task variant registrar for describing the task
     * @param user_data the user data to associate with the task variant
     * @param vid optional variant ID to use
     * @return variant ID for the task
     */
    template<
        typename T, typename UDT,
        T (*TASK_PTR)(
            const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*,
            const UDT&)>
    VariantID register_task_variant(
        const TaskVariantRegistrar& registrar, const UDT& user_data,
        VariantID vid = LEGION_AUTO_GENERATE_ID);

    /**
     * Dynamically register a new task variant with the runtime with
     * a void return type.
     * @param registrar the task variant registrar for describing the task
     * @param vid optional variant ID to use
     * @return variant ID for the task
     */
    template<void (*TASK_PTR)(
        const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*)>
    VariantID register_task_variant(
        const TaskVariantRegistrar& registrar,
        VariantID vid = LEGION_AUTO_GENERATE_ID);

    /**
     * Dynamically register a new task variant with the runtime with
     * a void return type and user data.
     * @param registrar the task variant registrar for describing the task
     * @param user_data the user data to associate with the task variant
     * @param vid optional variant ID to use
     * @return variant ID for the task
     */
    template<
        typename UDT, void (*TASK_PTR)(
                          const Task*, const std::vector<PhysicalRegion>&,
                          Context, Runtime*, const UDT&)>
    VariantID register_task_variant(
        const TaskVariantRegistrar& registrar, const UDT& user_data,
        VariantID vid = LEGION_AUTO_GENERATE_ID);

    /**
     * Dynamically register a new task variant with the runtime that
     * has already built in the necessary preamble/postamble (i.e.
     * calls to LegionTaskWrapper::legion_task_{pre,post}amble)
     * @param registrar the task variant registrar for describing the task
     * @param codedesc the code descriptor for the pre-wrapped task
     * @param user_data pointer to optional user data to associate with the
     * task variant
     * @param user_len size of optional user_data in bytes
     * @param return_type_size size in bytes of the maximum return type
     *                         produced by this task variant
     * @param vid optional variant ID to use
     * @param has_return_type_size boolean indicating whether the max
     *                         return_type_size is valid or not, in cases
     *                         with unbounded output futures this should
     *                         be set to false but will come with a
     *                         significant performance penalty
     * @return variant ID for the task
     */
    VariantID register_task_variant(
        const TaskVariantRegistrar& registrar, const CodeDescriptor& codedesc,
        const void* user_data = nullptr, size_t user_len = 0,
        size_t return_type_size = LEGION_MAX_RETURN_SIZE,
        VariantID vid = LEGION_AUTO_GENERATE_ID,
        bool has_return_type_size = true);

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
    template<
        typename T,
        T (*TASK_PTR)(
            const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*)>
    static VariantID preregister_task_variant(
        const TaskVariantRegistrar& registrar, const char* task_name = nullptr,
        VariantID vid = LEGION_AUTO_GENERATE_ID);

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
    template<
        typename T, typename UDT,
        T (*TASK_PTR)(
            const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*,
            const UDT&)>
    static VariantID preregister_task_variant(
        const TaskVariantRegistrar& registrar, const UDT& user_data,
        const char* task_name = nullptr,
        VariantID vid = LEGION_AUTO_GENERATE_ID);

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
    template<void (*TASK_PTR)(
        const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*)>
    static VariantID preregister_task_variant(
        const TaskVariantRegistrar& registrar, const char* task_name = nullptr,
        VariantID vid = LEGION_AUTO_GENERATE_ID);

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
    template<
        typename UDT, void (*TASK_PTR)(
                          const Task*, const std::vector<PhysicalRegion>&,
                          Context, Runtime*, const UDT&)>
    static VariantID preregister_task_variant(
        const TaskVariantRegistrar& registrar, const UDT& user_data,
        const char* task_name = nullptr,
        VariantID vid = LEGION_AUTO_GENERATE_ID);

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
     * @param return_type_size size in bytes of the maximum return type
     *                         produced by this task variant
     * @param has_return_type_size boolean indicating whether the max
     *                         return_type_size is valid or not, in cases
     *                         with unbounded output futures this should
     *                         be set to false but will come with a
     *                         significant performance penalty
     * @param check_task_id verify validity of the task ID
     * @return variant ID for the task
     */
    static VariantID preregister_task_variant(
        const TaskVariantRegistrar& registrar, const CodeDescriptor& codedesc,
        const void* user_data = nullptr, size_t user_len = 0,
        const char* task_name = nullptr,
        VariantID vid = LEGION_AUTO_GENERATE_ID,
        size_t return_type_size = LEGION_MAX_RETURN_SIZE,
        bool has_return_type_size = true, bool check_task_id = true);

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
    static void legion_task_preamble(
        const void* data, size_t datalen, Processor p, const Task*& task,
        const std::vector<PhysicalRegion>*& reg, Context& ctx,
        Runtime*& runtime);

    /**
     * This is the necessary postamble call to use when registering a task
     * variant with an explicit CodeDescriptor. It passes back the task
     * return value and completes the task. It should be the last thing
     * called before the task finishes. Note that if the return value is
     * not backed by an instance, then it must be in host-visible memory.
     * @param ctx the context for the task
     * @param retvalptr pointer to the return value
     * @param retvalsize the size of the return value in bytes
     * @param owned whether the runtime takes ownership of this result
     * @param inst optional Realm instance containing the data that
     *              Legion should take ownership of
     * @param metadataptr a pointer to host memory that contains metadata
     *              for the future. The runtime will always make a copy
     *              of this data if it is not nullptr.
     * @param metadatasize the size of the metadata buffer if non-nullptr
     */
    static void legion_task_postamble(
        Context ctx, const void* retvalptr = nullptr, size_t retvalsize = 0,
        bool owned = false,
        Realm::RegionInstance inst = Realm::RegionInstance::NO_INST,
        const void* metadataptr = nullptr, size_t metadatasize = 0);

    /**
     * This variant of the Legion task postamble allows clients to
     * return data in arbitrary memory locations as a future result.
     * Realm::ExternalInstanceResource objects provide ways of describing
     * all kinds of external allocations that Legion can understand
     * @param ctx the context for the task
     * @param retvalptr raw pointer for the allocation (can be nullptr)
     * @param retvalsize the size of the return value in bytes
     * @param owned whether the runtime takes ownership of this result
     * @param allocation an external instance resource description of
     *                   the future result data
     * @param freefunc optional function pointer to invoke to free the
     *                 resources associated with an external resource
     * @param metadataptr a pointer to host memory that contains metadata
     *              for the future. The runtime will always make a copy
     *              of this data if it is not nullptr.
     * @param metadatasize the size of the metadata buffer if non-nullptr
     */
    static void legion_task_postamble(
        Context ctx, const void* retvalptr, size_t retvalsize, bool owned,
        const Realm::ExternalInstanceResource& allocation,
        void (*freefunc)(const Realm::ExternalInstanceResource&) = nullptr,
        const void* metadataptr = nullptr, size_t metadatasize = 0);

    /**
     * This variant of the Legion task postamble allows users to pass in
     * a future functor object to serve as a callback interface for Legion
     * to query so that it is only invoked in the case where futures actually
     * need to be serialized.
     * @param ctx the context for the task
     * @param callback_functor pointer to the callback object
     * @param owned whether Legion should take ownership of the object
     */
    static void legion_task_postamble(
        Context ctx, FutureFunctor* callback_functor, bool owned = false);

    /**
     * This is a special variant of Legion task postamble for returning
     * a Domain as a value from a task. Clients can specify whether Legion
     * should take ownership of the sparsity map of the domain or not. If
     * not, then Legion will add its own reference to keep the sparsity
     * map for the domain alive if necessary.
     * @param ctx the context for the task
     * @param domain the domain to return as a reuslt
     * @param take_ownership whether Legion takes ownership of the domain
     * @param metadataptr a pointer to host memory that contains metadata
     *              for the future. The runtime will always make a copy
     *              of this data if it is not nullptr.
     * @param metadatasize the size of the metadata buffer if non-nullptr
     */
    static void legion_task_postamble(
        Context ctx, const Domain& domain, bool take_ownership,
        const void* metadataptr = nullptr, size_t metadatasize = 0);
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
    template<
        typename T,
        T (*TASK_PTR)(
            const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*)>
    LEGION_DEPRECATED(
        "Task registration should be done with "
        "a TaskVariantRegistrar")
    static TaskID register_legion_task(
        TaskID id, Processor::Kind proc_kind, bool single, bool index,
        VariantID vid = LEGION_AUTO_GENERATE_ID,
        TaskConfigOptions options = TaskConfigOptions(),
        const char* task_name = nullptr);
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
    template<void (*TASK_PTR)(
        const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*)>
    LEGION_DEPRECATED(
        "Task registration should be done with "
        "a TaskVariantRegistrar")
    static TaskID register_legion_task(
        TaskID id, Processor::Kind proc_kind, bool single, bool index,
        VariantID vid = LEGION_AUTO_GENERATE_ID,
        TaskConfigOptions options = TaskConfigOptions(),
        const char* task_name = nullptr);
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
    template<
        typename T, typename UDT,
        T (*TASK_PTR)(
            const Task*, const std::vector<PhysicalRegion>&, Context, Runtime*,
            const UDT&)>
    LEGION_DEPRECATED(
        "Task registration should be done with "
        "a TaskVariantRegistrar")
    static TaskID register_legion_task(
        TaskID id, Processor::Kind proc_kind, bool single, bool index,
        const UDT& user_data, VariantID vid = LEGION_AUTO_GENERATE_ID,
        TaskConfigOptions options = TaskConfigOptions(),
        const char* task_name = nullptr);
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
    template<
        typename UDT, void (*TASK_PTR)(
                          const Task*, const std::vector<PhysicalRegion>&,
                          Context, Runtime*, const UDT&)>
    LEGION_DEPRECATED(
        "Task registration should be done with "
        "a TaskVariantRegistrar")
    static TaskID register_legion_task(
        TaskID id, Processor::Kind proc_kind, bool single, bool index,
        const UDT& user_data, VariantID vid = LEGION_AUTO_GENERATE_ID,
        TaskConfigOptions options = TaskConfigOptions(),
        const char* task_name = nullptr);
  public:
    /**
     * Provide a method to test whether the Legion runtime has been
     * started yet or not. Note that this method simply queries at a
     * single point in time and can race with a call to Runtime::start
     * performed by a different thread.
     */
    static bool has_runtime(void);

    /**
     * Provide a mechanism for finding the Legion runtime
     * pointer for a processor wrapper tasks that are starting
     * a new application level task.
     * @param processor the task will run on
     * @return the Legion runtime pointer for the specified processor
     */
    static Runtime* get_runtime(Processor p = Processor::NO_PROC);

    /**
     * Test whether we are inside of a Legion task and therefore
     * have a context available. This can be used to see if it
     * is safe to call 'Runtime::get_context'.
     * @return boolean indicating if we are inside of a Legion task
     */
    static bool has_context(void);

    /**
     * Get the context for the currently executing task this must
     * be called inside of an actual Legion task. Calling it outside
     * of a Legion task will result in undefined behavior
     * @return the context for the enclosing task in which we are executing
     */
    static Context get_context(void);

    /**
     * Get the task object associated with a context
     * @param ctx enclosing processor context
     * @return the task representation of the context
     */
    static const Task* get_context_task(Context ctx);
  private:
    IndexPartition create_restricted_partition(
        Context ctx, IndexSpace parent, IndexSpace color_space,
        const void* transform, size_t transform_size, const void* extent,
        size_t extent_size, PartitionKind part_kind, Color color,
        const char* provenance, const char* func_name, bool blockify);
    IndexSpace create_index_space_union_internal(
        Context ctx, IndexPartition parent, const void* realm_color,
        size_t color_size, TypeTag type_tag, const char* provenance,
        const char* func, const std::vector<IndexSpace>& handles);
    IndexSpace create_index_space_union_internal(
        Context ctx, IndexPartition parent, const void* realm_color,
        size_t color_size, TypeTag type_tag, const char* provenance,
        const char* func, IndexPartition handle);
    IndexSpace create_index_space_intersection_internal(
        Context ctx, IndexPartition parent, const void* realm_color,
        size_t color_size, TypeTag type_tag, const char* provenance,
        const char* func, const std::vector<IndexSpace>& handles);
    IndexSpace create_index_space_intersection_internal(
        Context ctx, IndexPartition parent, const void* realm_color,
        size_t color_size, TypeTag type_tag, const char* provenance,
        const char* func, IndexPartition handle);
    IndexSpace create_index_space_difference_internal(
        Context ctx, IndexPartition paretn, const void* realm_color,
        size_t color_size, TypeTag type_tag, const char* provenance,
        const char* func, IndexSpace initial,
        const std::vector<IndexSpace>& handles);
    IndexSpace get_index_subspace_internal(
        IndexPartition handle, const void* realm_color, TypeTag type_tag);
    bool has_index_subspace_internal(
        IndexPartition handle, const void* realm_color, TypeTag type_tag);
    void get_index_partition_color_space_internal(
        IndexPartition handle, void* realm_is, TypeTag type_tag);
    void get_index_space_domain_internal(
        IndexSpace handle, void* realm_is, TypeTag type_tag);
    void get_index_space_color_internal(
        IndexSpace handle, void* realm_color, TypeTag type_tag);
    bool safe_cast_internal(
        Context ctx, LogicalRegion region, const void* realm_point,
        TypeTag type_tag, const char* func_name);
    LogicalRegion get_logical_subregion_by_color_internal(
        LogicalPartition parent, const void* realm_color, TypeTag type_tag);
    bool has_logical_subregion_by_color_internal(
        LogicalPartition parent, const void* realm_color, TypeTag type_tag);
  private:
    // Methods for the wrapper functions to get information from the runtime
    friend class LegionTaskWrapper;
    friend class LegionSerialization;
  public:
    // This method is hidden down here and not publicly documented because
    // users shouldn't really need it for anything, however there are some
    // reasonable cases where it might be utilitized for things like doing
    // file I/O or printf that people might want it for so we've got it
    ShardID get_shard_id(Context ctx, bool I_know_what_I_am_doing = false);
    // We'll also allow users to get the total number of shards in the context
    // if they also ar willing to attest they know what they are doing
    size_t get_num_shards(Context ctx, bool I_know_what_I_am_doing = false);
    // This is another hidden method for control replication because it's
    // still somewhat experimental. In some cases there are unavoidable
    // sources of randomness that can mess with the needed invariants for
    // control replication (e.g. garbage collectors). This method will
    // allow the application to pass in an array of elements from each shard
    // and the runtime will fill in an output buffer with an ordered array of
    // elements that were passed in by every shard. Each shard will get the
    // same elements that were present in all the other shards in the same
    // order in the output array. The number of elements in the output buffer
    // is returned as a future (of type size_t) as the runtime will return
    // immediately and the application can continue running ahead. The
    // application must keep the input and output buffers allocated until
    // the future resolves. By definition the output buffer need be no bigger
    // than the input buffer since only elements that are in the input buffer
    // on any shard can appear in the output buffer. Note you can use this
    // method safely in contexts that are not control replicated as well:
    // the input will just be mem-copied to the output and num_elements
    // returned as the future result.
    Future consensus_match(
        Context ctx, const void* input, void* output, size_t num_elements,
        size_t element_size, const char* provenance = nullptr);
  private:
    friend class Mapper;
    Internal::Runtime* runtime;
  };

}  // namespace Legion

#include "legion/api/runtime.inl"

#endif  // __LEGION_RUNTIME_H__
