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

// index spaces for Realm

#ifndef REALM_INDEXSPACE_H
#define REALM_INDEXSPACE_H

#include "realm/event.h"
#include "realm/memory.h"
#define REALM_SKIP_INLINES
#include "realm/instance.h"
#undef REALM_SKIP_INLINES

#include "realm/realm_c.h"
#include "realm/realm_config.h"
#include "realm/sparsity.h"
#include "realm/dynamic_templates.h"

// we need intptr_t - make it if needed
#if __cplusplus >= 201103L
#include <stdint.h>
#else
typedef ptrdiff_t intptr_t;
#endif

#include "realm/custom_serdez.h"

namespace Realm {
  // NOTE: all these interfaces are templated, which means partitions.cc is going
  //  to have to somehow know which ones to instantiate - this is controlled by the
  //  following type lists, using a bunch of helper stuff from dynamic_templates.h

  typedef DynamicTemplates::IntList<1, 3> DIMCOUNTS;
  typedef DynamicTemplates::TypeList<int, unsigned int, long long>::TL DIMTYPES;
  typedef DynamicTemplates::TypeList<int, bool>::TL FLDTYPES;

  class ProfilingRequestSet;
  class CodeDescriptor;
  
  struct CopySrcDstField {
  public:
    CopySrcDstField(void);
    ~CopySrcDstField(void);
    CopySrcDstField &set_field(RegionInstance _inst, FieldID _field_id,
			       size_t _size, size_t _subfield_offset = 0);
    CopySrcDstField &set_indirect(int _indirect_index, FieldID _field_id,
				  size_t _size, size_t _subfield_offset = 0);
    CopySrcDstField &set_redop(ReductionOpID _redop_id, bool _is_fold);
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
    CustomSerdezID serdez_id;
    size_t subfield_offset;
    int indirect_index;
    static const size_t MAX_DIRECT_SIZE = 8;
    union {
      char direct[8];
      void *indirect;
    } fill_data;
  };

  // new stuff here - the "Z" prefix will go away once we delete the old stuff

  template <int N, typename T = int> struct Point;
  template <int N, typename T = int> struct Rect;
  template <int M, int N, typename T = int> struct Matrix;
  template <int N, typename T = int> struct IndexSpace;
  template <int N, typename T = int> struct IndexSpaceIterator;
  template <int N, typename T = int> class SparsityMap;

  template <int N, typename T = int>
  class CopyIndirection {
  public:
    class Base {
      IndexSpace<N,T> target;
    };

    template <int N2, typename T2 = int>
    class Affine : public CopyIndirection<N,T>::Base {
    public:
      Matrix<N,N2,T2> transform;
      Point<N2,T2> offset_lo, offset_hi;
      Point<N2,T2> divisor;
      std::vector<IndexSpace<N2,T2> > spaces;
      std::vector<RegionInstance> insts;
    };

    template <int N2, typename T2 = int>
    class Unstructured : public CopyIndirection<N,T>::Base {
    public:
      FieldID field_id;
      RegionInstance inst;
      bool is_ranges;
      size_t subfield_offset;
      std::vector<IndexSpace<N2,T2> > spaces;
      std::vector<RegionInstance> insts;
    };
  };

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
  struct Point {
    T x, y, z, w;  T rest[N - 4];

    __CUDA_HD__
    Point(void);
    __CUDA_HD__
    explicit Point(const T vals[N]);
    // copies allow type coercion (assuming the underlying type does)
    template <typename T2> __CUDA_HD__
    Point(const Point<N, T2>& copy_from);
    template <typename T2> __CUDA_HD__
    Point<N,T>& operator=(const Point<N, T2>& copy_from);

    __CUDA_HD__
    T& operator[](int index);
    __CUDA_HD__
    const T& operator[](int index) const;

    template <typename T2> __CUDA_HD__
    T dot(const Point<N, T2>& rhs) const;
  };

  template <int N, typename T>
  std::ostream& operator<<(std::ostream& os, const Point<N,T>& p);

  // component-wise operators defined on Point<N,T> (with optional coercion)
  template <int N, typename T, typename T2> __CUDA_HD__
  bool operator==(const Point<N,T>& lhs, const Point<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  bool operator!=(const Point<N,T>& lhs, const Point<N,T2>& rhs);

  template <int N, typename T, typename T2> __CUDA_HD__
  Point<N,T> operator+(const Point<N,T>& lhs, const Point<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  Point<N,T>& operator+=(Point<N,T>& lhs, const Point<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  Point<N,T> operator-(const Point<N,T>& lhs, const Point<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  Point<N,T>& operator-=(Point<N,T>& lhs, const Point<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  Point<N,T> operator*(const Point<N,T>& lhs, const Point<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  Point<N,T>& operator*=(Point<N,T>& lhs, const Point<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  Point<N,T> operator/(const Point<N,T>& lhs, const Point<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  Point<N,T>& operator/=(Point<N,T>& lhs, const Point<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  Point<N,T> operator%(const Point<N,T>& lhs, const Point<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  Point<N,T>& operator%=(Point<N,T>& lhs, const Point<N,T2>& rhs);

  // a Rect is a pair of points defining the lower and upper bounds of an N-D rectangle
  //  the bounds are INCLUSIVE

  template <int N, typename T>
  struct Rect {
    Point<N,T> lo, hi;

    __CUDA_HD__
    Rect(void);
    __CUDA_HD__
    Rect(const Point<N,T>& _lo, const Point<N,T>& _hi);
    // copies allow type coercion (assuming the underlying type does)
    template <typename T2> __CUDA_HD__
    Rect(const Rect<N, T2>& copy_from);
    template <typename T2> __CUDA_HD__
    Rect<N,T>& operator=(const Rect<N, T2>& copy_from);

    // constructs a guaranteed-empty rectangle
    __CUDA_HD__
    static Rect<N,T> make_empty(void);

    __CUDA_HD__
    bool empty(void) const;
    __CUDA_HD__
    size_t volume(void) const;

    __CUDA_HD__
    bool contains(const Point<N,T>& p) const;

    // true if all points in other are in this rectangle
    __CUDA_HD__
    bool contains(const Rect<N,T>& other) const;
    __CUDA_HD__
    bool contains(const IndexSpace<N,T>& is) const;

    // true if there are any points in the intersection of the two rectangles
    __CUDA_HD__
    bool overlaps(const Rect<N,T>& other) const;

    __CUDA_HD__
    Rect<N,T> intersection(const Rect<N,T>& other) const;

    // returns the _bounding box_ of the union of two rectangles (the actual union
    //  might not be a rectangle)
    __CUDA_HD__
    Rect<N,T> union_bbox(const Rect<N,T>& other) const;

    // copy and fill operations (wrappers for IndexSpace versions)
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
               const IndexSpace<N,T> &mask,
               const ProfilingRequestSet &requests,
               Event wait_on = Event::NO_EVENT,
               ReductionOpID redop_id = 0, bool red_fold = false) const;
  };

  template <int N, typename T>
  std::ostream& operator<<(std::ostream& os, const Rect<N,T>& p);

  template <int N, typename T, typename T2> __CUDA_HD__
  bool operator==(const Rect<N,T>& lhs, const Rect<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  bool operator!=(const Rect<N,T>& lhs, const Rect<N,T2>& rhs);

  // rectangles may be displaced by a vector (i.e. point)
  template <int N, typename T, typename T2> __CUDA_HD__
  Rect<N,T> operator+(const Rect<N,T>& lhs, const Point<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  Rect<N,T>& operator+=(Rect<N,T>& lhs, const Point<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  Rect<N,T> operator-(const Rect<N,T>& lhs, const Point<N,T2>& rhs);
  template <int N, typename T, typename T2> __CUDA_HD__
  Rect<N,T>& operator-=(Rect<N,T>& lhs, const Rect<N,T2>& rhs);

  template <int M, int N, typename T>
  struct Matrix {
    Point<N,T> rows[M];

    __CUDA_HD__
    Matrix(void);
    // copies allow type coercion (assuming the underlying type does)
    template <typename T2> __CUDA_HD__
    Matrix(const Matrix<M, N, T2>& copy_from);
    template <typename T2> __CUDA_HD__
    Matrix<M, N, T>& operator=(const Matrix<M, N, T2>& copy_from);

    __CUDA_HD__
    Point<N,T>& operator[](int index);
    __CUDA_HD__
    const Point<N,T>& operator[](int index) const;
  };

  template <int M, int N, typename T, typename T2> __CUDA_HD__
  Point<M, T> operator*(const Matrix<M, N, T>& m, const Point<N, T2>& p);

  template <int N, typename T>
  class PointInRectIterator {
  public:
    Point<N,T> p;
    bool valid;
    Rect<N,T> rect;
    bool fortran_order;

    __CUDA_HD__
    PointInRectIterator(void);
    __CUDA_HD__
    PointInRectIterator(const Rect<N,T>& _r, bool _fortran_order = true);
    __CUDA_HD__
    void reset(const Rect<N,T>& _r, bool _fortran_order = true);
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
  struct IndexSpace {
    Rect<N,T> bounds;
    SparsityMap<N,T> sparsity;

    IndexSpace(void);  // results in an empty index space
    IndexSpace(const Rect<N,T>& _bounds);
    IndexSpace(const Rect<N,T>& _bounds, SparsityMap<N,T> _sparsity);

    // construct an index space from a list of points or rects
    IndexSpace(const std::vector<Point<N,T> >& points);
    IndexSpace(const std::vector<Rect<N,T> >& rects);

    // constructs a guaranteed-empty index space
    static IndexSpace<N,T> make_empty(void);

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
    IndexSpace<N,T> tighten(bool precise = true) const;

    // queries for individual points or rectangles
    bool contains(const Point<N,T>& p) const;
    bool contains_all(const Rect<N,T>& r) const;
    bool contains_any(const Rect<N,T>& r) const;

    bool overlaps(const IndexSpace<N,T>& other) const;

    // actual number of points in index space (may be less than volume of bounding box)
    size_t volume(void) const;

    // approximate versions of the above queries - the approximation is guaranteed to be a supserset,
    //  so if contains_approx returns false, contains would too
    bool contains_approx(const Point<N,T>& p) const;
    bool contains_all_approx(const Rect<N,T>& r) const;
    bool contains_any_approx(const Rect<N,T>& r) const;

    bool overlaps_approx(const IndexSpace<N,T>& other) const;

    // approximage number of points in index space (may be less than volume of bounding box, but larger than
    //   actual volume)
    size_t volume_approx(void) const;


    // as an alternative to IndexSpaceIterator's, this will internally iterate over rectangles
    //  and call your callable/lambda for each subrectangle
    template <typename LAMBDA>
    void foreach_subrect(LAMBDA lambda);
    template <typename LAMBDA>
    void foreach_subrect(LAMBDA lambda, const Rect<N,T>& restriction);

    // instance creation
#if 0
    RegionInstance create_instance(Memory memory,
				   const std::vector<size_t>& field_sizes,
				   size_t block_size,
				   const ProfilingRequestSet& reqs) const;
#endif

    // copy and fill operations

    // old versions do not support indirection, use explicit arguments for
    //   fill values, reduction op info
    Event fill(const std::vector<CopySrcDstField> &dsts,
               const ProfilingRequestSet &requests,
               const void *fill_value, size_t fill_value_size,
               Event wait_on = Event::NO_EVENT) const;

    Event copy(const std::vector<CopySrcDstField> &srcs,
               const std::vector<CopySrcDstField> &dsts,
               const ProfilingRequestSet &requests,
               Event wait_on,
               ReductionOpID redop_id, bool red_fold = false) const;

    Event copy(const std::vector<CopySrcDstField> &srcs,
	       const std::vector<CopySrcDstField> &dsts,
	       const ProfilingRequestSet &requests,
	       Event wait_on = Event::NO_EVENT) const;

    Event copy(const std::vector<CopySrcDstField> &srcs,
	       const std::vector<CopySrcDstField> &dsts,
	       const std::vector<const typename CopyIndirection<N,T>::Base *> &indirects,
	       const ProfilingRequestSet &requests,
	       Event wait_on = Event::NO_EVENT) const;

    // partitioning operations

    // index-based:
    Event create_equal_subspace(size_t count, size_t granularity,
                                unsigned index, IndexSpace<N,T> &subspace,
                                const ProfilingRequestSet &reqs,
                                Event wait_on = Event::NO_EVENT) const;

    Event create_equal_subspaces(size_t count, size_t granularity,
				 std::vector<IndexSpace<N,T> >& subspaces,
				 const ProfilingRequestSet &reqs,
				 Event wait_on = Event::NO_EVENT) const;

    Event create_weighted_subspaces(size_t count, size_t granularity,
				    const std::vector<int>& weights,
				    std::vector<IndexSpace<N,T> >& subspaces,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;

    // field-based:

    template <typename FT>
    Event create_subspace_by_field(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,FT> >& field_data,
				   FT color,
				   IndexSpace<N,T>& subspace,
				   const ProfilingRequestSet &reqs,
				   Event wait_on = Event::NO_EVENT) const;

    template <typename FT>
    Event create_subspaces_by_field(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,FT> >& field_data,
				    const std::vector<FT>& colors,
				    std::vector<IndexSpace<N,T> >& subspaces,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;

    // this version allows the "function" described by the field to be composed with a
    //  second (computable) function before matching the colors - the second function
    //  is provided via a CodeDescriptor object and should have the type FT->FT2
    template <typename FT, typename FT2>
    Event create_subspace_by_field(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,FT> >& field_data,
				   const CodeDescriptor& codedesc,
				   FT2 color,
				   IndexSpace<N,T>& subspace,
				   const ProfilingRequestSet &reqs,
				   Event wait_on = Event::NO_EVENT) const;

    template <typename FT, typename FT2>
    Event create_subspaces_by_field(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,FT> >& field_data,
				    const CodeDescriptor& codedesc,
				    const std::vector<FT2>& colors,
				    std::vector<IndexSpace<N,T> >& subspaces,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;

    // computes subspaces of this index space by determining what subsets are reachable from
    //  subsets of some other index space - the field data points from the other index space to
    //  ours and is used to compute the image of each source - i.e. upon return (and waiting
    //  for the finish event), the following invariant holds:
    //    images[i] = { y | exists x, x in sources[i] ^ field_data(x) = y }
    template <int N2, typename T2>
    Event create_subspace_by_image(const std::vector<FieldDataDescriptor<IndexSpace<N2,T2>,Point<N,T> > >& field_data,
				   const IndexSpace<N2,T2>& source,
				   IndexSpace<N,T>& image,
				   const ProfilingRequestSet &reqs,
				   Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2>
    Event create_subspaces_by_image(const std::vector<FieldDataDescriptor<IndexSpace<N2,T2>,Point<N,T> > >& field_data,
				    const std::vector<IndexSpace<N2,T2> >& sources,
				    std::vector<IndexSpace<N,T> >& images,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;
    // range versions
    template <int N2, typename T2>
    Event create_subspace_by_image(const std::vector<FieldDataDescriptor<IndexSpace<N2,T2>,Rect<N,T> > >& field_data,
				   const IndexSpace<N2,T2>& source,
				   IndexSpace<N,T>& image,
				   const ProfilingRequestSet &reqs,
				   Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2>
    Event create_subspaces_by_image(const std::vector<FieldDataDescriptor<IndexSpace<N2,T2>,Rect<N,T> > >& field_data,
				    const std::vector<IndexSpace<N2,T2> >& sources,
				    std::vector<IndexSpace<N,T> >& images,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;

    // a common case that is worth optimizing is when the computed image is
    //  going to be restricted by an intersection or difference operation - it
    //  can often be much faster to filter the projected points before stuffing
    //  them into an index space

    template <int N2, typename T2>
    Event create_subspaces_by_image_with_difference(const std::vector<FieldDataDescriptor<IndexSpace<N2,T2>,Point<N,T> > >& field_data,
				    const std::vector<IndexSpace<N2,T2> >& sources,
				    const std::vector<IndexSpace<N,T> >& diff_rhs,
				    std::vector<IndexSpace<N,T> >& images,
				    const ProfilingRequestSet &reqs,
				    Event wait_on = Event::NO_EVENT) const;

    // computes subspaces of this index space by determining what subsets can reach subsets
    //  of some other index space - the field data points from this index space to the other
    //  and is used to compute the preimage of each target - i.e. upon return (and waiting
    //  for the finish event), the following invariant holds:
    //    preimages[i] = { x | field_data(x) in targets[i] }
    template <int N2, typename T2>
    Event create_subspace_by_preimage(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,
				                        Point<N2,T2> > >& field_data,
				      const IndexSpace<N2,T2>& target,
				      IndexSpace<N,T>& preimage,
				      const ProfilingRequestSet &reqs,
				      Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2>
    Event create_subspaces_by_preimage(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,
				                         Point<N2,T2> > >& field_data,
				       const std::vector<IndexSpace<N2,T2> >& targets,
				       std::vector<IndexSpace<N,T> >& preimages,
				       const ProfilingRequestSet &reqs,
				       Event wait_on = Event::NO_EVENT) const;
    // range versions
    template <int N2, typename T2>
    Event create_subspace_by_preimage(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,
				                        Rect<N2,T2> > >& field_data,
				      const IndexSpace<N2,T2>& target,
				      IndexSpace<N,T>& preimage,
				      const ProfilingRequestSet &reqs,
				      Event wait_on = Event::NO_EVENT) const;

    template <int N2, typename T2>
    Event create_subspaces_by_preimage(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,
				                         Rect<N2,T2> > >& field_data,
				       const std::vector<IndexSpace<N2,T2> >& targets,
				       std::vector<IndexSpace<N,T> >& preimages,
				       const ProfilingRequestSet &reqs,
				       Event wait_on = Event::NO_EVENT) const;

    // create association
    // fill in the instances described by 'field_data' with a mapping
    // from this index space to the 'range' index space
    template <int N2, typename T2>
    Event create_association(const std::vector<FieldDataDescriptor<IndexSpace<N,T>,
                                                      Point<N2,T2> > >& field_data,
                             const IndexSpace<N2,T2> &range,
                             const ProfilingRequestSet &reqs,
                             Event wait_on = Event::NO_EVENT) const;

    // set operations

    // three basic operations (union, intersection, difference) are provided in 4 forms:
    //  IS op IS     -> result
    //  IS[] op IS[] -> result[] (zip over two inputs, which must be same length)
    //  IS op IS[]   -> result[] (first input applied to each element of second array)
    //  IS[] op IS   -> result[] (each element of first array applied to second input)

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

    // set reduction operations (union and intersection)

    static Event compute_union(const std::vector<IndexSpace<N,T> >& subspaces,
			       IndexSpace<N,T>& result,
			       const ProfilingRequestSet &reqs,
			       Event wait_on = Event::NO_EVENT);
				     
    static Event compute_intersection(const std::vector<IndexSpace<N,T> >& subspaces,
				      IndexSpace<N,T>& result,
				      const ProfilingRequestSet &reqs,
				      Event wait_on = Event::NO_EVENT);
  };

  template <int N, typename T>
  std::ostream& operator<<(std::ostream& os, const IndexSpace<N,T>& p);

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
    LinearizedIndexSpace(const IndexSpace<N,T>& _indexspace);

  public:
    // generic way to linearize a point
    virtual size_t linearize(const Point<N,T>& p) const = 0;

    IndexSpace<N,T> indexspace;
  };

  // the simplest way to linearize an index space is build an affine translation from its
  //  bounding box to the range [0, volume)
  template <int N, typename T>
  class AffineLinearizedIndexSpace : public LinearizedIndexSpace<N,T> {
  public:
    // "fortran order" has the smallest stride in the first dimension
    explicit AffineLinearizedIndexSpace(const IndexSpace<N,T>& _indexspace, bool fortran_order = true);

    virtual LinearizedIndexSpaceIntfc *clone(void) const;
    
    virtual size_t size(void) const;

    virtual size_t linearize(const Point<N,T>& p) const;

    size_t volume, offset;
    Point<N, ptrdiff_t> strides;
    Rect<N,T> dbg_bounds;
  };

  template <int N, typename T>
  class SparsityMapPublicImpl;

  // an IndexSpaceIterator iterates over the valid points in an IndexSpace, rectangles at a time
  template <int N, typename T>
  struct IndexSpaceIterator {
    Rect<N,T> rect;
    IndexSpace<N,T> space;
    Rect<N,T> restriction;
    bool valid;
    // for iterating over SparsityMap's
    SparsityMapPublicImpl<N,T> *s_impl;
    size_t cur_entry;

    IndexSpaceIterator(void);
    IndexSpaceIterator(const IndexSpace<N,T>& _space);
    IndexSpaceIterator(const IndexSpace<N,T>& _space, const Rect<N,T>& _restrict);

    void reset(const IndexSpace<N,T>& _space);
    void reset(const IndexSpace<N,T>& _space, const Rect<N,T>& _restrict);

    // steps to the next subrect, returning true if a next subrect exists
    bool step(void);
  };

}; // namespace Realm

// specializations of std::less<T> for Point/Rect/IndexSpace<N,T> allow
//  them to be used in STL containers
namespace std {
  template<int N, typename T>
  struct less<Realm::Point<N,T> > {
    bool operator()(const Realm::Point<N,T>& p1, const Realm::Point<N,T>& p2) const;
  };

  template<int N, typename T>
  struct less<Realm::Rect<N,T> > {
    bool operator()(const Realm::Rect<N,T>& r1, const Realm::Rect<N,T>& r2) const;
  };

  template<int N, typename T>
  struct less<Realm::IndexSpace<N,T> > {
    bool operator()(const Realm::IndexSpace<N,T>& is1, const Realm::IndexSpace<N,T>& is2) const;
  };
};

#include "realm/indexspace.inl"

#endif // ifndef REALM_INDEXSPACE_H
