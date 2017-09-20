/* Copyright 2017 Stanford University, NVIDIA Corporation
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

#include "event.h"
#include "memory.h"
#define REALM_SKIP_INLINES
#include "instance.h"
#undef REALM_SKIP_INLINES

#include "lowlevel_config.h"
#include "realm_config.h"
#include "arrays.h"
#include "sparsity.h"
#include "dynamic_templates.h"

// we need intptr_t - make it if needed
#if __cplusplus >= 201103L
#include <cstdint>
#else
typedef ptrdiff_t intptr_t;
#endif

#include "custom_serdez.h"

namespace Realm {
  typedef ::legion_lowlevel_coord_t coord_t;

  // NOTE: all these interfaces are templated, which means partitions.cc is going
  //  to have to somehow know which ones to instantiate - this is controlled by the
  //  following type lists, using a bunch of helper stuff from dynamic_templates.h

  typedef DynamicTemplates::IntList<1, 3> DIMCOUNTS;
  typedef DynamicTemplates::TypeList<int, unsigned int, long long, coord_t>::TL DIMTYPES;
  typedef DynamicTemplates::TypeList<int, bool>::TL FLDTYPES;

  class ProfilingRequestSet;
  class CodeDescriptor;
  
  struct CopySrcDstField {
  public:
  CopySrcDstField(void) 
    : inst(RegionInstance::NO_INST), field_id((FieldID)-1), size(0), 
      serdez_id(0), subfield_offset(0) { }
  public:
    RegionInstance inst;
    FieldID field_id;
    size_t size;
    CustomSerdezID serdez_id;
    size_t subfield_offset;
  };

  // new stuff here - the "Z" prefix will go away once we delete the old stuff

  template <int N, typename T = int> struct ZPoint;
  template <int N, typename T = int> struct ZRect;
  template <int M, int N, typename T = int> struct ZMatrix;
  template <int N, typename T = int> struct ZIndexSpace;
  template <int N, typename T = int> struct ZIndexSpaceIterator;
  template <int N, typename T = int> class SparsityMap;

  // a Point is a tuple describing a point in an N-dimensional space - the default "base type"
  //  for each dimension is int, but 64-bit indices are supported as well

  // only a few methods exist directly on a Point<N,T>:
  // 1) trivial constructor
  // 2) [for N <= 4] constructor taking N arguments of type T
  // 3) default copy constructor
  // 4) default assignment operator
  // 5) operator[] to access individual components

  // specializations for N <= 4 defined in indexspace.inl
  template <int N, typename T>
  struct ZPoint {
    T x, y, z, w;  T rest[N - 4];

    __CUDA_HD__
    ZPoint(void);
    __CUDA_HD__
    explicit ZPoint(const T vals[N]);
    // copies allow type coercion (assuming the underlying type does)
    template <typename T2> __CUDA_HD__
    ZPoint(const ZPoint<N, T2>& copy_from);
    template <typename T2> __CUDA_HD__
    ZPoint<N,T>& operator=(const ZPoint<N, T2>& copy_from);

    __CUDA_HD__
    T& operator[](int index);
    __CUDA_HD__
    const T& operator[](int index) const;

    template <typename T2> __CUDA_HD__
    T dot(const ZPoint<N, T2>& rhs) const;
  };

  template <int N, typename T>
  std::ostream& operator<<(std::ostream& os, const ZPoint<N,T>& p);

  // component-wise operators defined on Point<N,T> (with optional coercion)
  template <int N, typename T, typename T2> __CUDA_HD__
  bool operator==(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  bool operator!=(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs);

  template <int N, typename T, typename T2> __CUDA_HD__
  ZPoint<N,T> operator+(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  ZPoint<N,T>& operator+=(ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  ZPoint<N,T> operator-(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  ZPoint<N,T>& operator-=(ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  ZPoint<N,T> operator*(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  ZPoint<N,T>& operator*=(ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  ZPoint<N,T> operator/(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  ZPoint<N,T>& operator/=(ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  ZPoint<N,T> operator%(const ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  ZPoint<N,T>& operator%=(ZPoint<N,T>& lhs, const ZPoint<N,T2>& rhs);

  // a Rect is a pair of points defining the lower and upper bounds of an N-D rectangle
  //  the bounds are INCLUSIVE

  template <int N, typename T>
  struct ZRect {
    ZPoint<N,T> lo, hi;

    __CUDA_HD__
    ZRect(void);
    __CUDA_HD__
    ZRect(const ZPoint<N,T>& _lo, const ZPoint<N,T>& _hi);
    // copies allow type coercion (assuming the underlying type does)
    template <typename T2> __CUDA_HD__
    ZRect(const ZRect<N, T2>& copy_from);
    template <typename T2> __CUDA_HD__
    ZRect<N,T>& operator=(const ZRect<N, T2>& copy_from);

    // constructs a guaranteed-empty rectangle
    static ZRect<N,T> make_empty(void);

    __CUDA_HD__
    bool empty(void) const;
    __CUDA_HD__
    size_t volume(void) const;

    __CUDA_HD__
    bool contains(const ZPoint<N,T>& p) const;

    // true if all points in other are in this rectangle
    __CUDA_HD__
    bool contains(const ZRect<N,T>& other) const;
    __CUDA_HD__
    bool contains(const ZIndexSpace<N,T>& is) const;

    // true if there are any points in the intersection of the two rectangles
    __CUDA_HD__
    bool overlaps(const ZRect<N,T>& other) const;

    __CUDA_HD__
    ZRect<N,T> intersection(const ZRect<N,T>& other) const;

    // returns the _bounding box_ of the union of two rectangles (the actual union
    //  might not be a rectangle)
    __CUDA_HD__
    ZRect<N,T> union_bbox(const ZRect<N,T>& other) const;

    // copy and fill operations (wrappers for ZIndexSpace versions)
    Event fill(const std::vector<CopySrcDstField> &dsts,
               const ProfilingRequestSet &requests,
               const void *fill_value, size_t fill_value_size,
               Event wait_on = Event::NO_EVENT) const;

    Event copy(const std::vector<CopySrcDstField> &srcs,
               const std::vector<CopySrcDstField> &dsts,
               const ProfilingRequestSet &requests,
               Event wait_on = Event::NO_EVENT,
               ReductionOpID redop_id = 0, bool red_fold = false) const;

    Event copy(const std::vector<CopySrcDstField> &srcs,
               const std::vector<CopySrcDstField> &dsts,
               const ZIndexSpace<N,T> &mask,
               const ProfilingRequestSet &requests,
               Event wait_on = Event::NO_EVENT,
               ReductionOpID redop_id = 0, bool red_fold = false) const;
  };

  template <int N, typename T>
  std::ostream& operator<<(std::ostream& os, const ZRect<N,T>& p);

  template <int N, typename T, typename T2> __CUDA_HD__
  bool operator==(const ZRect<N,T>& lhs, const ZRect<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  bool operator!=(const ZRect<N,T>& lhs, const ZRect<N,T2>& rhs);

  // rectangles may be displaced by a vector (i.e. point)
  template <int N, typename T, typename T2> __CUDA_HD__
  ZRect<N,T> operator+(const ZRect<N,T>& lhs, const ZPoint<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  ZRect<N,T>& operator+=(ZRect<N,T>& lhs, const ZPoint<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  ZRect<N,T> operator-(const ZRect<N,T>& lhs, const ZPoint<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  ZRect<N,T>& operator-=(ZRect<N,T>& lhs, const ZRect<N,T2>& rhs);

  template <int M, int N, typename T>
  struct ZMatrix {
    ZPoint<N,T> rows[M];

    __CUDA_HD__
    ZMatrix(void);
    // copies allow type coercion (assuming the underlying type does)
    template <typename T2> __CUDA_HD__
    ZMatrix(const ZMatrix<M, N, T2>& copy_from);
    template <typename T2> __CUDA_HD__
    ZMatrix<M, N, T>& operator=(const ZMatrix<M, N, T2>& copy_from);

    __CUDA_HD__
    ZPoint<N,T>& operator[](int index);
    __CUDA_HD__
    const ZPoint<N,T>& operator[](int index) const;
  };

  template <int M, int N, typename T, typename T2> __CUDA_HD__
  ZPoint<M, T> operator*(const ZMatrix<M, N, T>& m, const ZPoint<N, T2>& p);

  template <int N, typename T>
  class ZPointInRectIterator {
  public:
    ZPoint<N,T> p;
    bool valid;
    ZRect<N,T> rect;
    bool fortran_order;

    __CUDA_HD__
    ZPointInRectIterator(void);
    __CUDA_HD__
    ZPointInRectIterator(const ZRect<N,T>& _r, bool _fortran_order = true);
    __CUDA_HD__
    bool step(void);
  };

  // a FieldDataDescriptor is used to describe field data provided for partitioning
  //  operations - it is templated on the dimensionality (N) and base type (T) of the
  //  index space that defines the domain over which the data is defined, and the
  //  type of the data contained in the field (FT)
  template <typename IS, typename FT>
  struct FieldDataDescriptor {
    IS index_space;
    RegionInstance inst;
    size_t field_offset;
  };

  // an IndexSpace is a POD type that contains a bounding rectangle and an optional SparsityMap - the
  //  contents of the IndexSpace are the intersection of the bounding rectangle's volume and the
  //  optional SparsityMap's contents
  // application code may directly manipulate the bounding rectangle - this will be common for structured
  //  index spaces
  template <int N, typename T>
  struct ZIndexSpace {
    ZRect<N,T> bounds;
    SparsityMap<N,T> sparsity;

    ZIndexSpace(void);  // results in an empty index space
    ZIndexSpace(const ZRect<N,T>& _bounds);
    ZIndexSpace(const ZRect<N,T>& _bounds, SparsityMap<N,T> _sparsity);

    // construct an index space from a list of points or rects
    ZIndexSpace(const std::vector<ZPoint<N,T> >& points);
    ZIndexSpace(const std::vector<ZRect<N,T> >& rects);

    // constructs a guaranteed-empty index space
    static ZIndexSpace<N,T> make_empty(void);

    // reclaim any physical resources associated with this index space
    //  will clear the sparsity map of this index space if it exists
    void destroy(Event wait_on = Event::NO_EVENT);

    // true if we're SURE that there are no points in the space (may be imprecise due to
    //  lazy loading of sparsity data)
    bool empty(void) const;
    
    // true if there is no sparsity map (i.e. the bounds fully define the domain)
    __CUDA_HD__
    bool dense(void) const;

    // kicks off any operation needed to get detailed sparsity information - asking for
    //  approximate data can be a lot quicker for complicated index spaces
    Event make_valid(bool precise = true) const;
    bool is_valid(bool precise = true) const;

    // returns the tightest description possible of the index space
    // if 'precise' is false, the sparsity map may be preserved even for dense
    //  spaces
    ZIndexSpace<N,T> tighten(bool precise = true) const;

    // queries for individual points or rectangles
    bool contains(const ZPoint<N,T>& p) const;
    bool contains_all(const ZRect<N,T>& r) const;
    bool contains_any(const ZRect<N,T>& r) const;

    bool overlaps(const ZIndexSpace<N,T>& other) const;

    // actual number of points in index space (may be less than volume of bounding box)
    size_t volume(void) const;

    // approximate versions of the above queries - the approximation is guaranteed to be a supserset,
    //  so if contains_approx returns false, contains would too
    bool contains_approx(const ZPoint<N,T>& p) const;
    bool contains_all_approx(const ZRect<N,T>& r) const;
    bool contains_any_approx(const ZRect<N,T>& r) const;

    bool overlaps_approx(const ZIndexSpace<N,T>& other) const;

    // approximage number of points in index space (may be less than volume of bounding box, but larger than
    //   actual volume)
    size_t volume_approx(void) const;


    // as an alternative to IndexSpaceIterator's, this will internally iterate over rectangles
    //  and call your callable/lambda for each subrectangle
    template <typename LAMBDA>
    void foreach_subrect(LAMBDA lambda);
    template <typename LAMBDA>
    void foreach_subrect(LAMBDA lambda, const ZRect<N,T>& restriction);

    // instance creation
#if 0
    RegionInstance create_instance(Memory memory,
				   const std::vector<size_t>& field_sizes,
				   size_t block_size,
				   const ProfilingRequestSet& reqs) const;
#endif

    // copy and fill operations

    Event fill(const std::vector<CopySrcDstField> &dsts,
               const ProfilingRequestSet &requests,
               const void *fill_value, size_t fill_value_size,
               Event wait_on = Event::NO_EVENT) const;

    Event copy(const std::vector<CopySrcDstField> &srcs,
               const std::vector<CopySrcDstField> &dsts,
               const ProfilingRequestSet &requests,
               Event wait_on = Event::NO_EVENT,
               ReductionOpID redop_id = 0, bool red_fold = false) const;

    Event copy(const std::vector<CopySrcDstField> &srcs,
               const std::vector<CopySrcDstField> &dsts,
               const ZIndexSpace<N,T> &mask,
               const ProfilingRequestSet &requests,
               Event wait_on = Event::NO_EVENT,
               ReductionOpID redop_id = 0, bool red_fold = false) const;

    // partitioning operations

    // index-based:
    Event create_equal_subspace(size_t count, size_t granularity,
                                unsigned index, ZIndexSpace<N,T> &subspace,
                                const ProfilingRequestSet &reqs,
                                Event wait_on = Event::NO_EVENT) const;

    Event create_equal_subspaces(size_t count, size_t granularity,
				 std::vector<ZIndexSpace<N,T> >& subspaces,
				 const ProfilingRequestSet &reqs,
				 Event wait_on = Event::NO_EVENT) const;

    Event create_weighted_subspaces(size_t count, size_t granularity,
				    const std::vector<int>& weights,
				    std::vector<ZIndexSpace<N,T> >& subspaces,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;

    // field-based:

    template <typename FT>
    Event create_subspace_by_field(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,FT> >& field_data,
				   FT color,
				   ZIndexSpace<N,T>& subspace,
				   const ProfilingRequestSet &reqs,
				   Event wait_on = Event::NO_EVENT) const;

    template <typename FT>
    Event create_subspaces_by_field(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,FT> >& field_data,
				    const std::vector<FT>& colors,
				    std::vector<ZIndexSpace<N,T> >& subspaces,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;

    // this version allows the "function" described by the field to be composed with a
    //  second (computable) function before matching the colors - the second function
    //  is provided via a CodeDescriptor object and should have the type FT->FT2
    template <typename FT, typename FT2>
    Event create_subspace_by_field(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,FT> >& field_data,
				   const CodeDescriptor& codedesc,
				   FT2 color,
				   ZIndexSpace<N,T>& subspace,
				   const ProfilingRequestSet &reqs,
				   Event wait_on = Event::NO_EVENT) const;

    template <typename FT, typename FT2>
    Event create_subspaces_by_field(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,FT> >& field_data,
				    const CodeDescriptor& codedesc,
				    const std::vector<FT2>& colors,
				    std::vector<ZIndexSpace<N,T> >& subspaces,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;

    // computes subspaces of this index space by determining what subsets are reachable from
    //  subsets of some other index space - the field data points from the other index space to
    //  ours and is used to compute the image of each source - i.e. upon return (and waiting
    //  for the finish event), the following invariant holds:
    //    images[i] = { y | exists x, x in sources[i] ^ field_data(x) = y }
    template <int N2, typename T2>
    Event create_subspace_by_image(const std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZPoint<N,T> > >& field_data,
				   const ZIndexSpace<N2,T2>& source,
				   ZIndexSpace<N,T>& image,
				   const ProfilingRequestSet &reqs,
				   Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2>
    Event create_subspaces_by_image(const std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZPoint<N,T> > >& field_data,
				    const std::vector<ZIndexSpace<N2,T2> >& sources,
				    std::vector<ZIndexSpace<N,T> >& images,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;
    // range versions
    template <int N2, typename T2>
    Event create_subspace_by_image(const std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZRect<N,T> > >& field_data,
				   const ZIndexSpace<N2,T2>& source,
				   ZIndexSpace<N,T>& image,
				   const ProfilingRequestSet &reqs,
				   Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2>
    Event create_subspaces_by_image(const std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZRect<N,T> > >& field_data,
				    const std::vector<ZIndexSpace<N2,T2> >& sources,
				    std::vector<ZIndexSpace<N,T> >& images,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;

    // a common case that is worth optimizing is when the computed image is
    //  going to be restricted by an intersection or difference operation - it
    //  can often be much faster to filter the projected points before stuffing
    //  them into an index space

    template <int N2, typename T2>
    Event create_subspaces_by_image_with_difference(const std::vector<FieldDataDescriptor<ZIndexSpace<N2,T2>,ZPoint<N,T> > >& field_data,
				    const std::vector<ZIndexSpace<N2,T2> >& sources,
				    const std::vector<ZIndexSpace<N,T> >& diff_rhs,
				    std::vector<ZIndexSpace<N,T> >& images,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;

    // computes subspaces of this index space by determining what subsets can reach subsets
    //  of some other index space - the field data points from this index space to the other
    //  and is used to compute the preimage of each target - i.e. upon return (and waiting
    //  for the finish event), the following invariant holds:
    //    preimages[i] = { x | field_data(x) in targets[i] }
    template <int N2, typename T2>
    Event create_subspace_by_preimage(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,
				                        ZPoint<N2,T2> > >& field_data,
				      const ZIndexSpace<N2,T2>& target,
				      ZIndexSpace<N,T>& preimage,
				      const ProfilingRequestSet &reqs,
				      Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2>
    Event create_subspaces_by_preimage(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,
				                         ZPoint<N2,T2> > >& field_data,
				       const std::vector<ZIndexSpace<N2,T2> >& targets,
				       std::vector<ZIndexSpace<N,T> >& preimages,
				       const ProfilingRequestSet &reqs,
				       Event wait_on = Event::NO_EVENT) const;
    // range versions
    template <int N2, typename T2>
    Event create_subspace_by_preimage(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,
				                        ZRect<N2,T2> > >& field_data,
				      const ZIndexSpace<N2,T2>& target,
				      ZIndexSpace<N,T>& preimage,
				      const ProfilingRequestSet &reqs,
				      Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2>
    Event create_subspaces_by_preimage(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,
				                         ZRect<N2,T2> > >& field_data,
				       const std::vector<ZIndexSpace<N2,T2> >& targets,
				       std::vector<ZIndexSpace<N,T> >& preimages,
				       const ProfilingRequestSet &reqs,
				       Event wait_on = Event::NO_EVENT) const;

    // create association
    // fill in the instances described by 'field_data' with a mapping
    // from this index space to the 'range' index space
    template <int N2, typename T2>
    Event create_association(const std::vector<FieldDataDescriptor<ZIndexSpace<N,T>,
                                                      ZPoint<N2,T2> > >& field_data,
                             const ZIndexSpace<N2,T2> &range,
                             const ProfilingRequestSet &reqs,
                             Event wait_on = Event::NO_EVENT) const;

    // set operations

    // three basic operations (union, intersection, difference) are provided in 4 forms:
    //  IS op IS     -> result
    //  IS[] op IS[] -> result[] (zip over two inputs, which must be same length)
    //  IS op IS[]   -> result[] (first input applied to each element of second array)
    //  IS[] op IS   -> result[] (each element of first array applied to second input)

    static Event compute_union(const ZIndexSpace<N,T>& lhs,
				    const ZIndexSpace<N,T>& rhs,
				    ZIndexSpace<N,T>& result,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT);

    static Event compute_unions(const std::vector<ZIndexSpace<N,T> >& lhss,
				     const std::vector<ZIndexSpace<N,T> >& rhss,
				     std::vector<ZIndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);

    static Event compute_unions(const ZIndexSpace<N,T>& lhs,
				     const std::vector<ZIndexSpace<N,T> >& rhss,
				     std::vector<ZIndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);

    static Event compute_unions(const std::vector<ZIndexSpace<N,T> >& lhss,
				     const ZIndexSpace<N,T>& rhs,
				     std::vector<ZIndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);

    static Event compute_intersection(const ZIndexSpace<N,T>& lhs,
				    const ZIndexSpace<N,T>& rhs,
				    ZIndexSpace<N,T>& result,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT);

    static Event compute_intersections(const std::vector<ZIndexSpace<N,T> >& lhss,
				     const std::vector<ZIndexSpace<N,T> >& rhss,
				     std::vector<ZIndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);

    static Event compute_intersections(const ZIndexSpace<N,T>& lhs,
				     const std::vector<ZIndexSpace<N,T> >& rhss,
				     std::vector<ZIndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);

    static Event compute_intersections(const std::vector<ZIndexSpace<N,T> >& lhss,
				     const ZIndexSpace<N,T>& rhs,
				     std::vector<ZIndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);

    static Event compute_difference(const ZIndexSpace<N,T>& lhs,
				    const ZIndexSpace<N,T>& rhs,
				    ZIndexSpace<N,T>& result,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT);

    static Event compute_differences(const std::vector<ZIndexSpace<N,T> >& lhss,
				     const std::vector<ZIndexSpace<N,T> >& rhss,
				     std::vector<ZIndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);

    static Event compute_differences(const ZIndexSpace<N,T>& lhs,
				     const std::vector<ZIndexSpace<N,T> >& rhss,
				     std::vector<ZIndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);

    static Event compute_differences(const std::vector<ZIndexSpace<N,T> >& lhss,
				     const ZIndexSpace<N,T>& rhs,
				     std::vector<ZIndexSpace<N,T> >& results,
				     const ProfilingRequestSet &reqs,
				     Event wait_on = Event::NO_EVENT);

    // set reduction operations (union and intersection)

    static Event compute_union(const std::vector<ZIndexSpace<N,T> >& subspaces,
			       ZIndexSpace<N,T>& result,
			       const ProfilingRequestSet &reqs,
			       Event wait_on = Event::NO_EVENT);
				     
    static Event compute_intersection(const std::vector<ZIndexSpace<N,T> >& subspaces,
				      ZIndexSpace<N,T>& result,
				      const ProfilingRequestSet &reqs,
				      Event wait_on = Event::NO_EVENT);
  };

  template <int N, typename T>
  std::ostream& operator<<(std::ostream& os, const ZIndexSpace<N,T>& p);

  // instances are based around the concept of a "linearization" of some index space, which is
  //  responsible for mapping (valid) points in the index space into a hopefully-fairly-dense
  //  subset of [0,size) (for some size)
  //
  // because index spaces can have arbitrary dimensionality, this description is wrapped in an
  //  abstract interface - all implementations must inherit from the approriate LinearizedIndexSpace<N,T>
  //  intermediate

  template <int N, typename T> class LinearizedIndexSpace;

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
    template <int N, typename T> bool check_dim(void) const;
    template <int N, typename T> LinearizedIndexSpace<N,T>& as_dim(void);
    template <int N, typename T> const LinearizedIndexSpace<N,T>& as_dim(void) const;

    int dim, idxtype;
  };

  template <int N, typename T>
  class LinearizedIndexSpace : public LinearizedIndexSpaceIntfc {
  protected:
    // still can't be created directly
    LinearizedIndexSpace(const ZIndexSpace<N,T>& _indexspace);

  public:
    // generic way to linearize a point
    virtual size_t linearize(const ZPoint<N,T>& p) const = 0;

    ZIndexSpace<N,T> indexspace;
  };

  // the simplest way to linearize an index space is build an affine translation from its
  //  bounding box to the range [0, volume)
  template <int N, typename T>
  class AffineLinearizedIndexSpace : public LinearizedIndexSpace<N,T> {
  public:
    // "fortran order" has the smallest stride in the first dimension
    explicit AffineLinearizedIndexSpace(const ZIndexSpace<N,T>& _indexspace, bool fortran_order = true);

    virtual LinearizedIndexSpaceIntfc *clone(void) const;
    
    virtual size_t size(void) const;

    virtual size_t linearize(const ZPoint<N,T>& p) const;

    size_t volume, offset;
    ZPoint<N, ptrdiff_t> strides;
    ZRect<N,T> dbg_bounds;
  };

  template <int N, typename T>
  class SparsityMapPublicImpl;

  // an IndexSpaceIterator iterates over the valid points in an IndexSpace, rectangles at a time
  template <int N, typename T>
  struct ZIndexSpaceIterator {
    ZRect<N,T> rect;
    ZIndexSpace<N,T> space;
    ZRect<N,T> restriction;
    bool valid;
    // for iterating over SparsityMap's
    SparsityMapPublicImpl<N,T> *s_impl;
    size_t cur_entry;

    ZIndexSpaceIterator(void);
    ZIndexSpaceIterator(const ZIndexSpace<N,T>& _space);
    ZIndexSpaceIterator(const ZIndexSpace<N,T>& _space, const ZRect<N,T>& _restrict);

    void reset(const ZIndexSpace<N,T>& _space);
    void reset(const ZIndexSpace<N,T>& _space, const ZRect<N,T>& _restrict);

    // steps to the next subrect, returning true if a next subrect exists
    bool step(void);
  };

}; // namespace Realm

// specializations of std::less<T> for ZPoint/ZRect/ZIndexSpace<N,T> allow
//  them to be used in STL containers
namespace std {
  template<int N, typename T>
  struct less<Realm::ZPoint<N,T> > {
    bool operator()(const Realm::ZPoint<N,T>& p1, const Realm::ZPoint<N,T>& p2) const;
  };

  template<int N, typename T>
  struct less<Realm::ZRect<N,T> > {
    bool operator()(const Realm::ZRect<N,T>& r1, const Realm::ZRect<N,T>& r2) const;
  };

  template<int N, typename T>
  struct less<Realm::ZIndexSpace<N,T> > {
    bool operator()(const Realm::ZIndexSpace<N,T>& is1, const Realm::ZIndexSpace<N,T>& is2) const;
  };
};

#include "indexspace.inl"

#endif // ifndef REALM_INDEXSPACE_H
