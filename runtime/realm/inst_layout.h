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

// Layout descriptors for Realm RegionInstances

#ifndef REALM_INST_LAYOUT_H
#define REALM_INST_LAYOUT_H

#include "realm/indexspace.h"
#include "realm/serialize.h"

#include <vector>
#include <map>
#include <iostream>

namespace Realm {

  class InstanceLayoutConstraints {
  public:
    InstanceLayoutConstraints(void) { }
    InstanceLayoutConstraints(const std::map<FieldID, size_t>& field_sizes,
			      size_t block_size);
    InstanceLayoutConstraints(const std::vector<size_t>& field_sizes,
			      size_t block_size);
#ifdef REALM_USE_LEGION_LAYOUT_CONSTRAINTS
    InstanceLayoutConstraints(const Legion::LayoutConstraintSet& lcs);
#endif

    struct FieldInfo {
      FieldID field_id;
      int offset;
      int size;
      int alignment;
    };
    typedef std::vector<FieldInfo> FieldGroup;

    std::vector<FieldGroup> field_groups;
  };


  // instance layouts are templated on the type of the IndexSpace used to
  //  index them, but they all inherit from a generic version
  class InstanceLayoutGeneric {
  protected:
    // cannot be created directly
    InstanceLayoutGeneric(void);

  public:
    template <typename S>
    static InstanceLayoutGeneric *deserialize_new(S& deserializer);

    virtual ~InstanceLayoutGeneric(void);

    // adjusts offsets of all pieces by 'adjust_amt'
    virtual void relocate(size_t adjust_amt) = 0;

    virtual void print(std::ostream& os) const = 0;

    template <int N, typename T>
    static InstanceLayoutGeneric *choose_instance_layout(IndexSpace<N,T> is,
							 const InstanceLayoutConstraints& ilc,
                                                         const int dim_order[N]);

    size_t bytes_used;
    size_t alignment_reqd;

    // we optimize for fields being laid out similarly, and have fields
    //  indirectly reference a piece list
    struct FieldLayout {
      int list_idx;
      int rel_offset;
      int size_in_bytes;
    };

    std::map<FieldID, FieldLayout> fields;
  };

  std::ostream& operator<<(std::ostream& os, const InstanceLayoutGeneric& ilg);

  // users that wish to handle instances as simple blocks of bits may use
  //  an InstanceLayoutOpaque to just request a contiguous range of bytes with
  //  a specified alignment
  class InstanceLayoutOpaque : public InstanceLayoutGeneric {
  public:
    InstanceLayoutOpaque(size_t _bytes_used, size_t _alignment_reqd);
  };

  template <int N, typename T>
  class InstanceLayoutPiece {
  public:
    enum LayoutType {
      InvalidLayoutType,
      AffineLayoutType,
      HDF5LayoutType,
    };

    InstanceLayoutPiece(void);
    InstanceLayoutPiece(LayoutType _layout_type);

    template <typename S>
    static InstanceLayoutPiece<N,T> *deserialize_new(S& deserializer);

    virtual ~InstanceLayoutPiece(void);

    virtual size_t calculate_offset(const Point<N,T>& p) const = 0;

    virtual void relocate(size_t base_offset) = 0;

    virtual void print(std::ostream& os) const = 0;

    LayoutType layout_type;
    Rect<N,T> bounds;
  };

  template <int N, typename T>
  std::ostream& operator<<(std::ostream& os, const InstanceLayoutPiece<N,T>& ilp);

  template <int N, typename T>
  class AffineLayoutPiece : public InstanceLayoutPiece<N,T> {
  public:
    AffineLayoutPiece(void);

    template <typename S>
    static InstanceLayoutPiece<N,T> *deserialize_new(S& deserializer);

    virtual size_t calculate_offset(const Point<N,T>& p) const;

    virtual void relocate(size_t base_offset);

    virtual void print(std::ostream& os) const;

    static Serialization::PolymorphicSerdezSubclass<InstanceLayoutPiece<N,T>, AffineLayoutPiece<N,T> > serdez_subclass;

    template <typename S>
    bool serialize(S& serializer) const;

    Point<N, size_t> strides;
    size_t offset;
  };

  template <int N, typename T>
  class InstancePieceList {
  public:
    InstancePieceList(void);
    ~InstancePieceList(void);

    const InstanceLayoutPiece<N,T> *find_piece(Point<N,T> p) const;

    void relocate(size_t base_offset);

    template <typename S>
    bool serialize(S& serializer) const;
    template <typename S>
    bool deserialize(S& deserializer);

    std::vector<InstanceLayoutPiece<N,T> *> pieces;
    // placeholder for lookup structure (e.g. K-D tree)
  };

  template <int N, typename T>
  std::ostream& operator<<(std::ostream& os, const InstancePieceList<N,T>& ipl);

  template <int N, typename T>
  class InstanceLayout : public InstanceLayoutGeneric {
  public:
    InstanceLayout(void);

    template <typename S>
    static InstanceLayoutGeneric *deserialize_new(S& deserializer);

    virtual ~InstanceLayout(void);

    // adjusts offsets of pieces to start from 'base_offset'
    virtual void relocate(size_t base_offset);

    virtual void print(std::ostream& os) const;

    // computes the offset of the specified field for an element - this
    //  is generally much less efficient than using a layout-specific accessor
    size_t calculate_offset(Point<N,T> p, FieldID fid) const;

    IndexSpace<N,T> space;
    std::vector<InstancePieceList<N,T> > piece_lists;

    static Serialization::PolymorphicSerdezSubclass<InstanceLayoutGeneric, InstanceLayout<N,T> > serdez_subclass;

    template <typename S>
    bool serialize(S& serializer) const;
  };

  // accessor stuff should eventually move to a different header file I think

  template <typename FT>
  class AccessorRefHelper {
  public:
    AccessorRefHelper(RegionInstance _inst, size_t _offset);

    // "read"
    operator FT(void) const;

    // "write"
    AccessorRefHelper<FT>& operator=(const FT& newval);
    AccessorRefHelper<FT>& operator=(const AccessorRefHelper<FT>& rhs);

  protected:
    RegionInstance inst;
    size_t offset;
  };

  // a generic accessor that works (slowly) for any instance layout
  template <typename FT, int N, typename T = int>
  class GenericAccessor {
  public:
    GenericAccessor(void);

    // GenericAccessor constructors always work, but the is_compatible(...)
    //  calls are still available for templated code

    // implicitly tries to cover the entire instance's domain
    GenericAccessor(RegionInstance inst,
		   FieldID field_id, size_t subfield_offset = 0);

    // limits domain to a subrectangle
    GenericAccessor(RegionInstance inst,
		   FieldID field_id, const Rect<N,T>& subrect,
		   size_t subfield_offset = 0);

    ~GenericAccessor(void);

    static bool is_compatible(RegionInstance inst, ptrdiff_t field_offset);
    static bool is_compatible(RegionInstance inst, ptrdiff_t field_offset, const Rect<N,T>& subrect);

    template <typename INST>
    static bool is_compatible(const INST &instance, unsigned field_id);
    template <typename INST>
    static bool is_compatible(const INST &instance, unsigned field_id, const Rect<N,T>& subrect);

    // GenericAccessor does not support returning a pointer to an element
    //FT *ptr(const Point<N,T>& p) const;

    FT read(const Point<N,T>& p);
    void write(const Point<N,T>& p, FT newval);

    // this returns a "reference" that knows how to do a read via operator FT
    //  or a write via operator=
    AccessorRefHelper<FT> operator[](const Point<N,T>& p);

    // instead of storing the top-level layout - we narrow down to just the
    //  piece list and relative offset of the field we're interested in
    RegionInstance inst;
    const InstancePieceList<N,T> *piece_list;
    int rel_offset;
    // cache the most recently-used piece
    const InstanceLayoutPiece<N,T> *prev_piece;

  //protected:
    // not a const method because of the piece caching
    size_t get_offset(const Point<N,T>& p);
  };
  
  template <typename FT, int N, typename T>
  std::ostream& operator<<(std::ostream& os, const GenericAccessor<FT,N,T>& a);


  // an instance accessor based on an affine linearization of an index space
  template <typename FT, int N, typename T = int>
  class AffineAccessor {
  public:
    // Note: All constructors except the default one must currently be called 
    // on the host so there are no __CUDA_HD__ qualifiers

    // TODO: Sean check if this is safe for a default constructor
    __CUDA_HD__
    AffineAccessor(void);
    // NOTE: these constructors will die horribly if the conversion is not
    //  allowed - call is_compatible(...) first if you're not sure

    // implicitly tries to cover the entire instance's domain
    AffineAccessor(RegionInstance inst,
		   FieldID field_id, size_t subfield_offset = 0);

    // limits domain to a subrectangle
    AffineAccessor(RegionInstance inst,
		   FieldID field_id, const Rect<N,T>& subrect,
		   size_t subfield_offset = 0);

    // these two constructors build accessors that incorporate an
    //  affine coordinate transform before the lookup in the actual instance
    template <int N2, typename T2>
    AffineAccessor(RegionInstance inst,
		   const Matrix<N2, N, T2>& transform,
		   const Point<N2, T2>& offset,
		   FieldID field_id, size_t subfield_offset = 0);

    // note that the subrect here is in in the accessor's indexspace
    //  (from which the corresponding subrectangle in the instance can be
    //  easily determined)
    template <int N2, typename T2>
    AffineAccessor(RegionInstance inst,
		   const Matrix<N2, N, T2>& transform,
		   const Point<N2, T2>& offset,
		   FieldID field_id, const Rect<N,T>& subrect,
		   size_t subfield_offset = 0);

    __CUDA_HD__
    ~AffineAccessor(void);

    static bool is_compatible(RegionInstance inst, FieldID field_id);
    static bool is_compatible(RegionInstance inst, FieldID field_id, const Rect<N,T>& subrect);
    template <int N2, typename T2>
    static bool is_compatible(RegionInstance inst,
			      const Matrix<N2, N, T2>& transform,
			      const Point<N2, T2>& offset,
			      FieldID field_id);
    template <int N2, typename T2>
    static bool is_compatible(RegionInstance inst,
			      const Matrix<N2, N, T2>& transform,
			      const Point<N2, T2>& offset,
			      FieldID field_id, const Rect<N,T>& subrect);

    __CUDA_HD__
    FT *ptr(const Point<N,T>& p) const;
    __CUDA_HD__
    FT read(const Point<N,T>& p) const;
    __CUDA_HD__
    void write(const Point<N,T>& p, FT newval) const;

    __CUDA_HD__
    FT& operator[](const Point<N,T>& p) const;

    __CUDA_HD__
    bool is_dense_arbitrary(const Rect<N,T> &bounds) const; // any dimension ordering
    __CUDA_HD__
    bool is_dense_col_major(const Rect<N,T> &bounds) const; // Fortran dimension ordering
    __CUDA_HD__
    bool is_dense_row_major(const Rect<N,T> &bounds) const; // C dimension ordering

  //protected:
  //friend
  // std::ostream& operator<<(std::ostream& os, const AffineAccessor<FT,N,T>& a);
//#define REALM_ACCESSOR_DEBUG
#ifdef REALM_ACCESSOR_DEBUG
    RegionInstance dbg_inst;
    Rect<N,T> dbg_bounds;
#ifdef __CUDACC__
#error "REALM_ACCESSOR_DEBUG macro for AffineAccessor not supported for GPU code"
#endif
#endif
    intptr_t base;
    Point<N, ptrdiff_t> strides;
  protected:
    __CUDA_HD__
    FT* get_ptr(const Point<N,T>& p) const;
  };

  template <typename FT, int N, typename T>
  std::ostream& operator<<(std::ostream& os, const AffineAccessor<FT,N,T>& a);

}; // namespace Realm

#include "realm/inst_layout.inl"

#endif // ifndef REALM_INST_LAYOUT_H


