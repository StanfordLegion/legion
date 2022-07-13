/* Copyright 2023 Stanford University, NVIDIA Corporation
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

// index spaces for Realm

#ifndef REALM_INDEXSPACE_H
#define REALM_INDEXSPACE_H

#include "realm/event.h"
#include "realm/memory.h"
#include "realm/point.h"
#define REALM_SKIP_INLINES
#include "realm/instance.h"
#undef REALM_SKIP_INLINES

#include "realm/dynamic_templates.h"
#include "realm/realm_c.h"
#include "realm/realm_config.h"
#include "realm/sparsity.h"

#include "realm/custom_serdez.h"

/**
 * \file indexspace.h
 * This file provides a C++ interface to Realm's index spaces.
 */

namespace Realm {
  // NOTE: all these interfaces are templated, which means partitions.cc is going
  //  to have to somehow know which ones to instantiate - this is controlled by the
  //  following type lists, using a bunch of helper stuff from dynamic_templates.h

  typedef DynamicTemplates::IntList<1, REALM_MAX_DIM> DIMCOUNTS;
  typedef DynamicTemplates::TypeList<int, unsigned int, long long>::TL DIMTYPES;
  typedef DynamicTemplates::TypeList<int, bool>::TL FLDTYPES;

  class ProfilingRequestSet;
  class CodeDescriptor;

  /**
   * \class CopySrcDstField
   * A class used to describe a single field of a source or destination
   * instance for a copy operation.
   */
  struct CopySrcDstField {
  public:
    CopySrcDstField(void);
    CopySrcDstField(const CopySrcDstField &copy_from);
    CopySrcDstField &operator=(const CopySrcDstField &copy_from);
    ~CopySrcDstField(void);
    CopySrcDstField &set_field(RegionInstance _inst, FieldID _field_id, size_t _size,
                               size_t _subfield_offset = 0);
    CopySrcDstField &set_indirect(int _indirect_index, FieldID _field_id, size_t _size,
                                  size_t _subfield_offset = 0);
    CopySrcDstField &set_redop(ReductionOpID _redop_id, bool _is_fold,
                               bool exclusive = false);
    CopySrcDstField &set_serdez(CustomSerdezID _serdez_id);
    CopySrcDstField &set_fill(const void *_data, size_t _size);
    template <typename T>
    CopySrcDstField &set_fill(T value);

  public:
    RegionInstance inst;
    FieldID field_id;
    size_t size;
    ReductionOpID redop_id;
    bool red_fold;
    bool red_exclusive;
    CustomSerdezID serdez_id;
    size_t subfield_offset;
    int indirect_index;
    static const size_t MAX_DIRECT_SIZE = 8;
    union {
      char direct[8];
      void *indirect;
    } fill_data;
  };

  std::ostream &operator<<(std::ostream &os, const CopySrcDstField &sd);

  template <int N, typename T = int>
  struct IndexSpaceIterator;
  template <int N, typename T = int>
  class SparsityMap;

  /**
   * \class FieldDataDescriptor
   * A template class used to describe field data provided for partitioning
   * operations. IS is the type of the index space that defines the domain
   * dimensionality and base type.
   */
  template <typename IS, typename FT>
  struct FieldDataDescriptor {
    IS index_space;
    RegionInstance inst;
    size_t field_offset;
  };

  /**
   * \class TranslationTransform
   * A translation transform is a special case of an affine transform
   * that only has a translation component. It is used to transform
   * points in one coordinate space into points in another coordinate
   * space by adding a constant offset to each coordinate.
   */
  template <int N, typename T = int>
  class REALM_PUBLIC_API TranslationTransform {
   public:
    TranslationTransform(void) = default;
    TranslationTransform(const Point<N, T>& _offset);

    /**
     * Apply the translation transform to a point in the domain
     * source coordinate space to produce a point in the destination
     * coordinate space. Adds the offset to the point.
     * \param point The point in the source coordinate space.
     * \return The point in the destination coordinate space.
     */
    template <typename T2>
    Realm::Point<N, T> operator[](const Realm::Point<N, T2>& point) const;

    Point<N, T> offset;
  };

  /**
   * \class AffineTransform
   * An affine transform is used to transform points in one
   * coordinate space into points in another coordinate space
   * using the basic Ax + b transformation, where A is a
   * transform matrix and b is an offset vector
   */
  template <int M, int N, typename T = int>
  class REALM_PUBLIC_API AffineTransform {
   public:
    AffineTransform(void) = default;
    AffineTransform(const Realm::Matrix<M, N, T>& _transform,
                    const Point<M, T>& _offset);

    /**
     * Apply the affine transform to a point in the domain
     * source coordinate space to produce a point in the destination
     * coordinate space.
     * \param point The point in the source coordinate space.
     * \return The point in the destination coordinate space.
     */
    template <typename T2>
    Realm::Point<M, T> operator[](const Realm::Point<N, T2>& point) const;

    Realm::Matrix<M, N, T> transform;
    Point<M, T> offset;
  };

  /**
   * \class StructuredTransform
   * A structured transform is used to represent both affine and
   * translation transforms. It is templated on the dimensionality
   * (N) and base type (T) of the index space that defines the
   * domain over which the transform is defined, and the dimensionality
   * (N2) and base type (T2) of the index space that defines the
   * range of the transform.
   */
  template <int N, typename T, int N2, typename T2>
  class REALM_PUBLIC_API StructuredTransform {
   public:
    StructuredTransform(void) = default;
    StructuredTransform(const AffineTransform<N, N2, T2>& _transform);
    StructuredTransform(const TranslationTransform<N, T2>& _transform);

    enum StructuredTransformType {
      NONE = 0,
      AFFINE = 1,
      TRANSLATION = 2,
    };

    /**
     * Apply the structured transform to a point in the domain
     * source coordinate space to produce a point in the destination
     * coordinate space.
     * \param point The point in the source coordinate space.
     * \return The point in the destination coordinate space.
     */
    Point<N, T> operator[](const Point<N2, T>& point) const;

    // protected:
    Realm::Matrix<N, N2, T2> transform_matrix;
    Point<N, T2> offset;
    StructuredTransformType type = StructuredTransformType::NONE;
  };

  /**
   * \class DomainTransform
   * A domain transform is used to represent a transform from one
   * index space to another (both structured and unstructured).
   * It is templated on the dimensionality (N) and base type (T) of
   * the index space that defines the domain over which the transform
   * is defined, and the dimensionality (N2) and base type (T2) of the
   * index space that defines the range of the transform.
   */
  template <int N, typename T, int N2, typename T2>
  class REALM_PUBLIC_API DomainTransform {
   public:
    DomainTransform(void) = default;
    DomainTransform(const StructuredTransform<N, T, N2, T2>& _transform);
    DomainTransform(
        const std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Point<N, T>>>&
            _field_data);
    DomainTransform(
        const std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Rect<N, T>>>&
            _field_data);

    enum DomainTransformType {
      NONE = 0,
      STRUCTURED = 1,
      UNSTRUCTURED_PTR = 2,
      UNSTRUCTURED_RANGE = 3,
    };

    // protected:
    StructuredTransform<N, T, N2, T2> structured_transform;
    std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Point<N, T>>> ptr_data;
    std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Rect<N, T>>> range_data;
    DomainTransformType type = DomainTransformType::NONE;
  };

  class IndirectionInfo;

  /**
   * \class CopyIndirection
   * A copy indirection is a container class that is used to describe
   * an indirection that is used to transform points in one index space
   * into points in another index space. It is templated on the dimensionality
   * (N) and base type (T) of source index space, and the dimensionality
   * (N2) and base type (T2) of the destination index space.
   */
  template <int N, typename T = int>
  class REALM_PUBLIC_API CopyIndirection {
  public:
    class Base {
    public:
      virtual ~Base(void) {}
      REALM_INTERNAL_API_EXTERNAL_LINKAGE
      virtual IndirectionInfo *create_info(const IndexSpace<N, T> &is) const = 0;
    };

    template <int N2, typename T2 = int>
    class Affine : public CopyIndirection<N, T>::Base {
    public:
      virtual ~Affine(void) {}

      // Defines the next indirection to avoid a 3-way templating.
      typename CopyIndirection<N2, T2>::Base *next_indirection;

      Matrix<N, N2, T2> transform;
      Point<N2, T2> offset_lo, offset_hi;
      Point<N2, T2> divisor;
      Rect<N2, T2> wrap;
      std::vector<IndexSpace<N2, T2>> spaces;
      std::vector<RegionInstance> insts;

      REALM_INTERNAL_API_EXTERNAL_LINKAGE
      virtual IndirectionInfo *create_info(const IndexSpace<N, T> &is) const;
    };

    template <int N2, typename T2 = int>
    class Unstructured : public CopyIndirection<N, T>::Base {
    public:
      virtual ~Unstructured(void) {}

      typename CopyIndirection<N2, T2>::Base *next_indirection;

      FieldID field_id;
      RegionInstance inst;
      bool is_ranges;
      bool oor_possible;      // can any pointers fall outside all the target spaces?
      bool aliasing_possible; // can multiple pointers go to the same element?
      size_t subfield_offset;
      std::vector<IndexSpace<N2, T2>> spaces;
      std::vector<RegionInstance> insts;

      REALM_INTERNAL_API_EXTERNAL_LINKAGE
      virtual IndirectionInfo *create_info(const IndexSpace<N, T> &is) const;
    };
  };


  /**
   * \class IndexSpace
   * An IndexSpace is a POD type that contains a bounding rectangle and an
   * optional SparsityMap - the contents of the IndexSpace are the intersection
   * of the bounding rectangle's volume and the optional SparsityMap's contents
   * - application code may directly manipulate the bounding rectangle
   * - this will be common for structured index spaces.
   */
  template <int N, typename T>
  struct REALM_PUBLIC_API IndexSpace {
    Rect<N, T> bounds;
    SparsityMap<N, T> sparsity;

    IndexSpace(void); // results in an empty index space
    IndexSpace(const Rect<N, T> &_bounds);
    IndexSpace(const Rect<N, T> &_bounds, SparsityMap<N, T> _sparsity);

    ///@{
    /**
     * Construct an index space from a list of points or rects
     *  this construction can be significantly faster if the caller promises
     *  that all of the 'points' or 'rects' are disjoint.
     *  \param points list of points to include in the index space.
     *  \param disjoint true if the points are guaranteed to be disjoint.
     */
    explicit IndexSpace(const std::vector<Point<N,T> >& points,
                        bool disjoint = false);

    explicit IndexSpace(const std::vector<Rect<N,T> >& rects,
                        bool disjoint = false);
    ///@}

    ///@{
    /**
     * Construct a guaranteed-empty index space.
     * \return an empty index space.
     */
    static IndexSpace<N,T> make_empty(void);
    ///@}

    ///@{
    /**
     * Reclaim any physical resources associated with this index space.
     * It will clear the sparsity map of this index space if it exists.
     * \param wait_on event to wait on before destroying the index space.
     */

    void destroy(Event wait_on = Event::NO_EVENT);
    ///@}

    ///@{
    /**
     * Retrun true if there are no points in the space
     * (may be imprecise due to lazy loading of sparsity data).
     * \return true if the index space is empty.
     */
    bool empty(void) const;

    ///@}

    ///@{
    /**
     * Return true if there is no sparsity map (i.e. the bounds fully
     * define the domain).
     * \return true if the index space is dense.
     */
    REALM_CUDA_HD
    bool dense(void) const;
    ///@}

    ///@{
    /**
     * Start any operation needed to get detailed sparsity information.
     * This function starts an operation to compute detailed sparsity
     * information for the index space. The operation may take some
     * time to complete, especially for complicated index spaces.
     * By default, the sparsity map is computed precisely, but if
     * performance is a concern, an approximate map can be computed 
     * by setting the \p precise parameter to \c false.
     * \param precise false is the sparsity map may be preserved even
     * for dense spaces.
     * \return event to wait on before using the sparsity map.
     */
    Event make_valid(bool precise = true) const;

    bool is_valid(bool precise = true) const;
    ///@}

    ///@{
    /**
     * Return the tightest description possible of the index space.
     * \param precise false is the sparsity map may be preserved even
     * for dense spaces.
     * \return the tightest index space possible.
     */
    IndexSpace<N,T> tighten(bool precise = true) const;
    ///@}

    ///@{
    /**
     * Query for individual points or rectangles.
     * \param p point to query
     * \param r rectangle to query
     * \return true if the point or rectangle is contained in the index space.
     */
    bool contains(const Point<N,T>& p) const;
    bool contains_all(const Rect<N,T>& r) const;
    bool contains_any(const Rect<N,T>& r) const;
    ///@}

    bool overlaps(const IndexSpace<N, T> &other) const;

    ///@{
    /**
     * Return actual number of points in index space (may be less than
     * volume of bounding box).
     * \return the number of points in the index space.
     */
    size_t volume(void) const;

    ///@}

    ///@{
    /**
     * Query whether a point is approximately contained in the index
     * space. This function returns \c true if the point \p p is 
     * approximately contained in the index space. The approximation 
     * is guaranteed to be a superset, so if \c contains_approx
     * returns \c false, then \c contains.
     * \param p point to query
     * \param r rectangle to query
     * \return true if the point or rectangle is contained in the index space.
     */
    bool contains_approx(const Point<N,T>& p) const;
    bool contains_all_approx(const Rect<N,T>& r) const;
    bool contains_any_approx(const Rect<N,T>& r) const;
    ///@}

    bool overlaps_approx(const IndexSpace<N, T> &other) const;

    ///@{
    /**
     * Approximate number of points in index space (may be less than
     * volume of bounding box, but larger than actual volume).
     * \return the approximate number of points in the index space.
     */

    size_t volume_approx(void) const;
    ///@}

    ///@{
    /**
     * Attempt to compute a set of covering rectangles for the index
     * space with the following properties:
     *
     * a) Every point in the index space is included in (exactly) one rectangle.
     * b) None of the resulting rectangles overlap each other.
     * c) No more than 'max_rects' rectangles are used (0 = no limit).
     * d) The relative storage overhead (%) is less than 'max_overhead', i.e.
     * 100 * (volume(covering) / volume(space) - 1) <= max_overhead.
     *
     * If successful, this function returns true and fills in the 'covering'
     * vector. If unsuccessful, it returns false and leaves 'covering'
     * unchanged.
     *
     * For N=1 (i.e., 1-D index spaces), this function is optimal, returning a
     * zero-overhead covering using a minimal number of rectangles if that
     * satisfies the 'max_rects' bound, or a covering (using max_rects
     * rectangles) with minimal overhead if that overhead is acceptable, or
     * fails if no covering exists.
     *
     * For N>1, heuristics are used, and the guarantees are much weaker:
     *
     * a) A request with 'max_rects'==1 will precisely compute the overhead and
     * succeed/fail appropriately.
     *
     * b) A request with 'max_rects'==0 (no limit) will always succeed with zero
     * overhead, although the number of rectangles used may not be minimal.
     *
     * c) The computational complexity of the attempt will be bounded at:
     * O(nm log m + nmk^2), where:
     *
     *    - n = dimension of index space.
     *    - m = size of exact internal representation (which itself is computed
     * by heuristics and may not be optimal for some dependent-partitioning
     * results).
     *    - k = maximum output rectangles.
     *
     * This allows for sorting the inputs and/or outputs, as well as dynamic
     * programming approaches, but precludes more "heroic" optimizations. A use
     * case that requires better results and/or admits specific optimizations
     * will need to compute its own coverings.
     *
     * \param max_rects Maximum number of rectangles to use (0 = no limit).
     * \param max_overhead Maximum relative storage overhead (0 = no limit).
     * \param covering Vector to fill in with covering rectangles.
     *
     * \return True if successful, false if not.
     */
    bool compute_covering(size_t max_rects, int max_overhead,
                          std::vector<Rect<N, T>>& covering) const;
    ///@}

    ///@{
    /**
     * Iterates internally over points and calls the specified callable or
     * lambda for each point in each subrectangle, providing an alternative to
     * using IndexSpaceIterator's.
     *
     * \param lambda A callable or lambda function that takes a Rect<N,T> and a
     * Point<N,T> as input parameters.
     * \param restriction An optional restriction on the subrectangles to
     * iterate over. Only points and subrectangles within this restriction will
     * be visited.
     */
    template <typename LAMBDA>
    void foreach_subrect(LAMBDA lambda);
    template <typename LAMBDA>
    void foreach_subrect(LAMBDA lambda, const Rect<N, T>& restriction);
    ///@}

    // instance creation
#if 0
    RegionInstance create_instance(Memory memory,
				   const std::vector<size_t>& field_sizes,
				   size_t block_size,
				   const ProfilingRequestSet& reqs) const;
#endif

    ///@{
    /**
     * Fill values into the specified destination fields in the index
     * space.
     *
     * \param dsts Vector of CopySrcDstField's describing the destination
     * fields.
     * \param requests Set of profiling requests.
     * \param fill_value Pointer to the fill value.
     * \param fill_value_size Size of the fill value in bytes.
     * \param wait_on Event to wait on before performing the fill operation.
     * \param priority Task priority.
     * \return Event representing the fill operation.
     *
     * Note: This version of fill() does not support indirection. Use explicit
     * arguments for fill values and reduction operation info.
     */
    Event fill(const std::vector<CopySrcDstField>& dsts,
               const ProfilingRequestSet& requests, const void* fill_value,
               size_t fill_value_size, Event wait_on = Event::NO_EVENT,
               int priority = 0) const;
    ///@}

    ///@{
    /**
     * Copy data from source fields to destination fields in the index
     * space.
     *
     * \param srcs Vector of CopySrcDstField's describing the source fields.
     * \param dsts Vector of CopySrcDstField's describing the destination
     * fields.
     * \param requests Set of profiling requests.
     * \param wait_on Event to wait on before performing the copy operation.
     * \param priority Task priority.
     * \return Event representing the copy operation.
     */
    Event copy(const std::vector<CopySrcDstField>& srcs,
               const std::vector<CopySrcDstField>& dsts,
               const ProfilingRequestSet& requests,
               Event wait_on = Event::NO_EVENT, int priority = 0) const;
    ///@}

    ///@{
    /**
     * Copy data from source fields to destination fields in the index
     * space with indirection.
     *
     * \param srcs Vector of CopySrcDstField's describing the source fields.
     * \param dsts Vector of CopySrcDstField's describing the destination
     * fields.
     * \param indirects Vector of pointers to the base class of the
     * indirections.
     * \param requests Set of profiling requests.
     * \param wait_on Event to wait on before performing the copy operation.
     * \param priority Task priority.
     * \return Event representing the copy operation.
     */
    Event copy(const std::vector<CopySrcDstField>& srcs,
               const std::vector<CopySrcDstField>& dsts,
               const std::vector<const typename CopyIndirection<N, T>::Base*>&
                   indirects,
               const ProfilingRequestSet& requests,
               Event wait_on = Event::NO_EVENT, int priority = 0) const;
    ///@}

    // Partitioning operations

    ///@{
    /**
     * Create equal index spaces from this index space.
     *
     * \param count Number of subspaces to create.
     * \param granularity Granularity of the subspaces.
     * \param index Index of the subspace to create.
     * \param subspace Index space to fill in with the result.
     * \param reqs Set of profiling requests.
     * \param wait_on Event to wait on before performing the partitioning
     * operation.
     * \return Event representing the partitioning operation.
     * */
    Event create_equal_subspace(size_t count, size_t granularity,
                                unsigned index, IndexSpace<N,T> &subspace,
                                const ProfilingRequestSet &reqs,
                                Event wait_on = Event::NO_EVENT) const;

    Event create_equal_subspaces(size_t count, size_t granularity,
				 std::vector<IndexSpace<N,T> >& subspaces,
				 const ProfilingRequestSet &reqs,
				 Event wait_on = Event::NO_EVENT) const;
    ///@}

    ///@{
    /**
     * Create index spaces from this index space based on the specified weights.
     *
     * \param count Number of subspaces to create.
     * \param granularity Granularity of the subspaces.
     * \param weights Vector of weights for each subspace.
     * \param subspace Index space to fill in with the result.
     * \param reqs Set of profiling requests.
     * \param wait_on Event to wait on before performing the partitioning
     * operation.
     * \return Event representing the partitioning operation.
     * */
    Event create_weighted_subspaces(size_t count, size_t granularity,
                                    const std::vector<int> &weights,
                                    std::vector<IndexSpace<N, T>> &subspaces,
                                    const ProfilingRequestSet &reqs,
                                    Event wait_on = Event::NO_EVENT) const;

    Event create_weighted_subspaces(size_t count, size_t granularity,
				    const std::vector<size_t>& weights,
				    std::vector<IndexSpace<N,T> >& subspaces,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;
    ///@}

    // Field-based partitioning operations

    template <typename FT>
    Event create_subspace_by_field(
        const std::vector<FieldDataDescriptor<IndexSpace<N, T>, FT>> &field_data,
        FT color, IndexSpace<N, T> &subspace, const ProfilingRequestSet &reqs,
        Event wait_on = Event::NO_EVENT) const;

    template <typename FT>
    Event create_subspaces_by_field(
        const std::vector<FieldDataDescriptor<IndexSpace<N, T>, FT>> &field_data,
        const std::vector<FT> &colors, std::vector<IndexSpace<N, T>> &subspaces,
        const ProfilingRequestSet &reqs, Event wait_on = Event::NO_EVENT) const;

    ///@{
    /**
     * Allows the "function" described by the field to be composed with a
     * second (computable) function before matching the colors - the second
     * function is provided via a CodeDescriptor object and should have the type
     * FT->FT2.
     * \param field_data the field data to use for the computation
     * \param codedesc the code descriptor for the second function
     * \param color the color to match
     * \param subspace the resulting subspace
     * \param reqs profiling requests
     * \param wait_on event to wait on before starting.
     * \return an event that will trigger when the operation is
     * complete.
     */
    template <typename FT, typename FT2>
    Event create_subspace_by_field(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,FT> >& field_data,
				   const CodeDescriptor& codedesc,
				   FT2 color,
				   IndexSpace<N,T>& subspace,
				   const ProfilingRequestSet &reqs,
				   Event wait_on = Event::NO_EVENT) const;
    ///@}

    template <typename FT, typename FT2>
    Event create_subspaces_by_field(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,FT> >& field_data,
				    const CodeDescriptor& codedesc,
				    const std::vector<FT2>& colors,
				    std::vector<IndexSpace<N,T> >& subspaces,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;

    ///@{
    /**
     * Computes subspaces of this index space by determining what subsets are
     * reachable from subsets of some other index space via a
     * transformation. The transformed data points from source index
     * spaces to ours are used to compute the images. - i.e. upon
     * return (and waiting for the finish event), the following
     * invariant holds: images[i] = { y | exists x, x in
     * transform(sources[i]) = y }
     *  \param transform the transformation to apply
     *  \param source the source index space
     *  \param image the resulting image index space
     *  \param reqs profiling requests
     *  \param wait_on event to wait on before starting
     *  \return an event that will trigger when the operation is
     *  complete
     */
    template <int N2, typename T2, typename TRANSFORM>
    Event create_subspace_by_image(const TRANSFORM& transform,
                                   const IndexSpace<N2, T2>& source,
                                   const IndexSpace<N, T>& image,
                                   const ProfilingRequestSet& reqs,
                                   Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2, typename TRANSFORM>
    Event create_subspaces_by_image(
        const TRANSFORM& transform,
        const std::vector<IndexSpace<N2, T2>>& sources,
        std::vector<IndexSpace<N, T>>& images, const ProfilingRequestSet& reqs,
        Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2>
    Event create_subspaces_by_image(
        const DomainTransform<N, T, N2, T2>& domain_transform,
        const std::vector<IndexSpace<N2, T2>>& sources,
        std::vector<IndexSpace<N, T>>& images, const ProfilingRequestSet& reqs,
        Event wait_on = Event::NO_EVENT) const;
    ///@}

    ///@{
    /**
     * Computes subspaces of this index space by determining what subsets are
     * reachable from subsets of some other index space - the field data points
     * from the other index space to ours and is used to compute the image of
     * each source - i.e. upon return (and waiting for the finish event), the
     * following invariant holds: images[i] = { y | exists x, x in sources[i] ^
     * field_data(x) = y
     *  }
     *  \param field_data the field data to use for the computation
     *  \param source the source index space
     *  \param image the resulting image index space
     *  \param reqs profiling requests
     *  \param wait_on event to wait on before starting.
     *  \return an event that will trigger when the operation is
     */
    template <int N2, typename T2>
    Event create_subspace_by_image(
        const std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Point<N, T>>>
            &field_data,
        const IndexSpace<N2, T2> &source, IndexSpace<N, T> &image,
        const ProfilingRequestSet &reqs, Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2>
    Event create_subspaces_by_image(
        const std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Point<N, T>>>&
            field_data,
        const std::vector<IndexSpace<N2, T2>>& sources,
        std::vector<IndexSpace<N, T>>& images, const ProfilingRequestSet& reqs,
        Event wait_on = Event::NO_EVENT) const;

    // range versions
    template <int N2, typename T2>
    Event create_subspace_by_image(
        const std::vector<FieldDataDescriptor<IndexSpace<N2, T2>, Rect<N, T>>>
            &field_data,
        const IndexSpace<N2, T2> &source, IndexSpace<N, T> &image,
        const ProfilingRequestSet &reqs, Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2>
    Event create_subspaces_by_image(const std::vector<FieldDataDescriptor<IndexSpace<N2,T2>,Rect<N,T> > >& field_data,
				    const std::vector<IndexSpace<N2,T2> >& sources,
				    std::vector<IndexSpace<N,T> >& images,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;
    ///@}

    ///@{
    /**
     * Computes subspaces of this index space by determining what subsets are
     * reachable from subsets of some other index space taking into
     * account that the computed image may be restricted by an intersection
     * or difference operation. It can often be much faster to filter
     * the projected points before stuffing them into an index space.
     *  \param field_data the field data to use for the computation
     *  \param sources the source index spaces
     *  \param diff_rhs the index spaces to use for the difference operation
     *  \param images the resulting image index spaces
     *  \param reqs profiling requests
     *  \param wait_on event to wait on before starting.
     *  \return an event that will trigger when the operation is
     *  complete.
     */
    template <int N2, typename T2>
    Event create_subspaces_by_image_with_difference(const std::vector<FieldDataDescriptor<IndexSpace<N2,T2>,Point<N,T> > >& field_data,
				    const std::vector<IndexSpace<N2,T2> >& sources,
				    const std::vector<IndexSpace<N,T> >& diff_rhs,
				    std::vector<IndexSpace<N,T> >& images,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2>
    Event create_subspaces_by_image_with_difference(
        const DomainTransform<N, T, N2, T2>& domain_transform,
        const std::vector<IndexSpace<N2, T2>>& sources,
        const std::vector<IndexSpace<N, T>>& diff_rhs,
        std::vector<IndexSpace<N, T>>& images, const ProfilingRequestSet& reqs,
        Event wait_on = Event::NO_EVENT) const;
    ///@}

    ///@{
    /**
     * Computes subspaces of this index space by determining what subsets can
     * reach subsets of some other index space via a transformation.
     * The transormed data points from this index space to the other
     * and is used to compute the preimage of each
     * target - i.e. upon return (and waiting for the finish event), the
     * following invariant holds: preimages[i] = { x | transform(x) in
     * targets[i] }
     * \param transform the transformation to use for the computation
     * \param target the target index space
     * \param preimage the resulting preimage index space
     * \param reqs profiling requests
     * \param wait_on event to wait on before starting.
     * \return an event that will trigger when the operation is
     */
    template <int N2, typename T2, typename TRANSFORM>
    Event create_subspace_by_preimage(const TRANSFORM& transform,
                                      const IndexSpace<N2, T2>& target,
                                      IndexSpace<N, T>& preimage,
                                      const ProfilingRequestSet& reqs,
                                      Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2, typename TRANSFORM>
    Event create_subspaces_by_preimage(
        const TRANSFORM& transform,
        const std::vector<IndexSpace<N2, T2>>& targets,
        std::vector<IndexSpace<N, T>>& preimages,
        const ProfilingRequestSet& reqs, Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2>
    Event create_subspaces_by_preimage(
        const DomainTransform<N2, T2, N, T>& domain_transform,
        const std::vector<IndexSpace<N2, T2>>& targets,
        std::vector<IndexSpace<N, T>>& preimages,
        const ProfilingRequestSet& reqs, Event wait_on = Event::NO_EVENT) const;
    ///@}

    ///@{
    /**
     * Computes subspaces of this index space by determining what subsets can
     * reach subsets of some other index space - the field data points from this
     * index space to the other and is used to compute the preimage of each
     * target - i.e. upon return (and waiting for the finish event), the
     * following invariant holds: preimages[i] = { x | field_data(x) in
     * targets[i] }
     * \param field_data the field data to use for the computation
     * \param target the target index space
     * \param preimage the resulting preimage index space
     * \param reqs profiling requests
     * \param wait_on event to wait on before starting.
     * \return an event that will trigger when the operation is
     */
    template <int N2, typename T2>
    Event create_subspace_by_preimage(
        const std::vector<FieldDataDescriptor<IndexSpace<N, T>, Point<N2, T2>>>
            &field_data,
        const IndexSpace<N2, T2> &target, IndexSpace<N, T> &preimage,
        const ProfilingRequestSet &reqs, Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2>
    Event create_subspaces_by_preimage(
        const std::vector<FieldDataDescriptor<IndexSpace<N, T>, Point<N2, T2>>>
            &field_data,
        const std::vector<IndexSpace<N2, T2>> &targets,
        std::vector<IndexSpace<N, T>> &preimages, const ProfilingRequestSet &reqs,
        Event wait_on = Event::NO_EVENT) const;
    // range versions
    template <int N2, typename T2>
    Event create_subspace_by_preimage(
        const std::vector<FieldDataDescriptor<IndexSpace<N, T>, Rect<N2, T2>>>
            &field_data,
        const IndexSpace<N2, T2> &target, IndexSpace<N, T> &preimage,
        const ProfilingRequestSet &reqs, Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2>
    Event create_subspaces_by_preimage(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,
				                         Rect<N2,T2> > >& field_data,
				       const std::vector<IndexSpace<N2,T2> >& targets,
				       std::vector<IndexSpace<N,T> >& preimages,
				       const ProfilingRequestSet &reqs,
				       Event wait_on = Event::NO_EVENT) const;
    ///@}

    ///@{
    /**
     * Create an association, fill in instances described by 'field_data' with a
     * mapping from this index space to the 'range' index space.
     * \param field_data the field data to use for the computation
     * \param range the range index space
     * \param reqs profiling requests
     * \param wait_on event to wait on before starting.
     * \return an event that will trigger when the operation is
     */
    template <int N2, typename T2>
    Event create_association(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,
                                                      Point<N2,T2> > >& field_data,
                             const IndexSpace<N2,T2> &range,
                             const ProfilingRequestSet &reqs,
                             Event wait_on = Event::NO_EVENT) const;
    ///@}
    // set operations

    // three basic operations (union, intersection, difference) are provided in 4 forms:
    //  IS op IS     -> result
    //  IS[] op IS[] -> result[] (zip over two inputs, which must be same length)
    //  IS op IS[]   -> result[] (first input applied to each element of second array)
    //  IS[] op IS   -> result[] (each element of first array applied to second input)

    ///@{
    /**
     * Compute the union of two index spaces.
     * \param lhs the left hand side of the union
     * \param rhs the right hand side of the union
     * \param result the resulting index space
     * \param reqs profiling requests
     * \param wait_on event to wait on before starting.
     * \return an event that will trigger when the operation is
     * complete
     */
    static Event compute_union(const IndexSpace<N,T>& lhs,
				    const IndexSpace<N,T>& rhs,
				    IndexSpace<N,T>& result,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT);

    static Event compute_unions(const std::vector<IndexSpace<N,T> >& lhss,
        const std::vector<IndexSpace<N,T> >& rhss,
        std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);

    static Event compute_unions(const IndexSpace<N,T>& lhs,
				     const std::vector<IndexSpace<N,T> >& rhss,
				     std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);

    static Event compute_unions(const std::vector<IndexSpace<N,T> >& lhss,
				     const IndexSpace<N,T>& rhs,
				     std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);
    ///@}

    ///@{
    /**
     * Compute the intersection of two index spaces.
     * \param lhs the left hand side of the intersection
     * \param rhs the right hand side of the intersection
     * \param result the resulting index space
     * \param reqs profiling requests
     * \param wait_on event to wait on before starting
     * \return an event that will trigger when the operation is complete
     */
    static Event compute_intersection(const IndexSpace<N,T>& lhs,
				    const IndexSpace<N,T>& rhs,
				    IndexSpace<N,T>& result,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT);

    static Event compute_intersections(const std::vector<IndexSpace<N,T> >& lhss,
				     const std::vector<IndexSpace<N,T> >& rhss,
				     std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);

    static Event compute_intersections(const IndexSpace<N,T>& lhs,
				     const std::vector<IndexSpace<N,T> >& rhss,
				     std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);

    static Event compute_intersections(const std::vector<IndexSpace<N,T> >& lhss,
				     const IndexSpace<N,T>& rhs,
				     std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);
    ///@}

    ///@{
    /**
     * Compute the difference of two index spaces.
     * \param lhs the left hand side of the difference
     * \param rhs the right hand side of the difference
     * \param result the resulting index space
     * \param reqs profiling requests
     * \param wait_on event to wait on before starting
     * \return an event that will trigger when the operation is complete
     */
    static Event compute_difference(const IndexSpace<N,T>& lhs,
				    const IndexSpace<N,T>& rhs,
				    IndexSpace<N,T>& result,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT);

    static Event compute_differences(const std::vector<IndexSpace<N,T> >& lhss,
				     const std::vector<IndexSpace<N,T> >& rhss,
				     std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);

    static Event compute_differences(const IndexSpace<N,T>& lhs,
				     const std::vector<IndexSpace<N,T> >& rhss,
				     std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);

    static Event compute_differences(const std::vector<IndexSpace<N,T> >& lhss,
				     const IndexSpace<N,T>& rhs,
				     std::vector<IndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);
    ///@}

    // set reduction operations (union and intersection)

    static Event compute_union(const std::vector<IndexSpace<N, T>> &subspaces,
                               IndexSpace<N, T> &result, const ProfilingRequestSet &reqs,
                               Event wait_on = Event::NO_EVENT);

    static Event compute_intersection(const std::vector<IndexSpace<N, T>> &subspaces,
                                      IndexSpace<N, T> &result,
                                      const ProfilingRequestSet &reqs,
                                      Event wait_on = Event::NO_EVENT);
  };

  template <int N, typename T>
  REALM_PUBLIC_API std::ostream &operator<<(std::ostream &os, const IndexSpace<N, T> &p);

  class IndexSpaceGenericImpl;

  /**
   * \class IndexSpaceGeneric
   * A type-erased IndexSpace that can be used to avoid template explosion
   * at the cost of run-time indirection - avoid using this in
   * performance-critical code.
   */
  class REALM_PUBLIC_API IndexSpaceGeneric {
  public:
    IndexSpaceGeneric();
    IndexSpaceGeneric(const IndexSpaceGeneric &copy_from);

    template <int N, typename T>
    IndexSpaceGeneric(const IndexSpace<N,T>& copy_from);
    template <int N, typename T>
    IndexSpaceGeneric(const Rect<N,T>& copy_from);

    ~IndexSpaceGeneric();

    IndexSpaceGeneric &operator=(const IndexSpaceGeneric &copy_from);

    template <int N, typename T>
    IndexSpaceGeneric& operator=(const IndexSpace<N,T>& copy_from);
    template <int N, typename T>
    IndexSpaceGeneric& operator=(const Rect<N,T>& copy_from);

    template <int N, typename T>
    const IndexSpace<N, T> &as_index_space() const;

    // only IndexSpace method exposed directly is copy
    Event copy(const std::vector<CopySrcDstField> &srcs,
	       const std::vector<CopySrcDstField> &dsts,
	       const ProfilingRequestSet &requests,
	       Event wait_on = Event::NO_EVENT,
	       int priority = 0) const;

    template <int N, typename T>
    Event copy(const std::vector<CopySrcDstField> &srcs,
	       const std::vector<CopySrcDstField> &dsts,
	       const std::vector<const typename CopyIndirection<N,T>::Base *> &indirects,
	       const ProfilingRequestSet &requests,
	       Event wait_on = Event::NO_EVENT,
	       int priority = 0) const;

    // "public" but not useful to application code
    IndexSpaceGenericImpl *impl;

  protected:
    // would like to use sizeof(IndexSpace<REALM_MAX_DIM, size_t>) here,
    //  but that requires the specializations that are defined in the
    //  include of indexspace.inl below...
    static constexpr size_t MAX_TYPE_SIZE = DIMTYPES::MaxSize::value;
    static constexpr size_t STORAGE_BYTES = (2*REALM_MAX_DIM + 2) * MAX_TYPE_SIZE;

    typedef char Storage_unaligned[STORAGE_BYTES];
    REALM_ALIGNED_TYPE_SAMEAS(Storage_aligned, Storage_unaligned, DIMTYPES::MaxSizeType<MAX_TYPE_SIZE>::TYPE);
    Storage_aligned raw_storage;
  };

  template <int N, typename T> class LinearizedIndexSpace;

  /**
   * \class LinearizedIndexSpaceIntfc
   * Instances are based around the concept of a "linearization" of some index
   * space, which is responsible for mapping (valid) points in the index space
   * into a hopefully-fairly-dense subset of [0,size) (for some size) because
   * index spaces can have arbitrary dimensionality, this description is wrapped
   * in an abstract interface - all implementations must inherit from the
   * approriate LinearizedIndexSpace<N,T> intermediate.
   */
  class LinearizedIndexSpaceIntfc {
  protected:
    // cannot be created directly
    LinearizedIndexSpaceIntfc(int _dim, int _idxtype);

  public:
    virtual ~LinearizedIndexSpaceIntfc(void);

    virtual LinearizedIndexSpaceIntfc *clone(void) const = 0;

    // returns the size of the linearized space
    virtual size_t size(void) const = 0;

    // check and conversion routines to get a dimension-aware intermediate
    template <int N, typename T>
    bool check_dim(void) const;
    template <int N, typename T>
    LinearizedIndexSpace<N, T> &as_dim(void);
    template <int N, typename T>
    const LinearizedIndexSpace<N, T> &as_dim(void) const;

    int dim, idxtype;
  };

  template <int N, typename T>
  class LinearizedIndexSpace : public LinearizedIndexSpaceIntfc {
  protected:
    // still can't be created directly
    LinearizedIndexSpace(const IndexSpace<N, T> &_indexspace);

  public:
    // generic way to linearize a point
    virtual size_t linearize(const Point<N, T> &p) const = 0;

    IndexSpace<N, T> indexspace;
  };

  /**
   * \class AffineLinearizedIndexSpace
   * The simplest possible linearization of an index space is to use an affine
   * transformation to map the bounding box of the index space to the
   * range [0, volume)
   */
  template <int N, typename T>
  class AffineLinearizedIndexSpace : public LinearizedIndexSpace<N, T> {
  public:
    // "fortran order" has the smallest stride in the first dimension
    explicit AffineLinearizedIndexSpace(const IndexSpace<N, T> &_indexspace,
                                        bool fortran_order = true);

    virtual LinearizedIndexSpaceIntfc *clone(void) const;

    virtual size_t size(void) const;

    virtual size_t linearize(const Point<N, T> &p) const;

    size_t volume, offset;
    Point<N, ptrdiff_t> strides;
    Rect<N, T> dbg_bounds;
  };

  template <int N, typename T>
  class SparsityMapPublicImpl;

  /**
   * \class IndexSpaceIterator
   * An IndexSpaceIterator iterates over the valid points in an IndexSpace,
   * rectangles at a time.
   */
  template <int N, typename T>
  struct REALM_PUBLIC_API IndexSpaceIterator {
    Rect<N, T> rect;
    IndexSpace<N, T> space;
    Rect<N, T> restriction;
    bool valid;
    // for iterating over SparsityMap's
    SparsityMapPublicImpl<N, T> *s_impl;
    size_t cur_entry;

    IndexSpaceIterator(void);
    IndexSpaceIterator(const IndexSpace<N, T> &_space);
    IndexSpaceIterator(const IndexSpace<N, T> &_space, const Rect<N, T> &_restrict);

    void reset(const IndexSpace<N, T> &_space);
    void reset(const IndexSpace<N, T> &_space, const Rect<N, T> &_restrict);

    // steps to the next subrect, returning true if a next subrect exists
    bool step(void);
  };

}; // namespace Realm

// specializations of std::less<T> for IndexSpace<N,T> allow
//  them to be used in STL containers
namespace std {
  template <int N, typename T>
  struct less<Realm::IndexSpace<N, T>> {
    bool operator()(const Realm::IndexSpace<N, T> &is1,
                    const Realm::IndexSpace<N, T> &is2) const;
  };
}; // namespace std

#include "realm/indexspace.inl"

#endif // ifndef REALM_INDEXSPACE_H
