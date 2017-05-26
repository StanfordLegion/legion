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

// Layout descriptors for Realm RegionInstances

#ifndef REALM_INST_LAYOUT_H
#define REALM_INST_LAYOUT_H

#include "indexspace.h"

#include <vector>
#include <map>
#include <iostream>

namespace Realm {

  typedef int FieldID;

  class InstanceLayoutConstraints {
  public:
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


  // instance layouts are templated on the type of the ZIndexSpace used to
  //  index them, but they all inherit from a generic version
  class InstanceLayoutGeneric {
  protected:
    // cannot be created directly
    InstanceLayoutGeneric(void);

  public:
    virtual ~InstanceLayoutGeneric(void);

    // adjusts offsets of all pieces by 'adjust_amt'
    virtual void relocate(size_t adjust_amt) = 0;

    virtual void print(std::ostream& os) const = 0;

    template <int N, typename T>
    static InstanceLayoutGeneric *choose_instance_layout(ZIndexSpace<N,T> is,
							 const InstanceLayoutConstraints& ilc);

    size_t bytes_used;
    size_t alignment_reqd;

    // we optimize for fields being laid out similarly, and have fields
    //  indirectly reference a piece list
    struct FieldLayout {
      int list_idx;
      int rel_offset;
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
    };

    InstanceLayoutPiece(void);
    InstanceLayoutPiece(LayoutType _layout_type);
    virtual ~InstanceLayoutPiece(void);

    virtual void relocate(size_t base_offset) = 0;

    virtual void print(std::ostream& os) const = 0;

    LayoutType layout_type;
    ZRect<N,T> bounds;
  };

  template <int N, typename T>
  std::ostream& operator<<(std::ostream& os, const InstanceLayoutPiece<N,T>& ilp);

  template <int N, typename T>
  class AffineLayoutPiece : public InstanceLayoutPiece<N,T> {
  public:
    AffineLayoutPiece(void);

    virtual void relocate(size_t base_offset);

    virtual void print(std::ostream& os) const;

    ZPoint<N, size_t> strides;
    size_t offset;
  };

  template <int N, typename T>
  class InstancePieceList {
  public:
    InstancePieceList(void);
    ~InstancePieceList(void);

    const InstanceLayoutPiece<N,T> *find_piece(ZPoint<N,T> p) const;

    void relocate(size_t base_offset);

    std::vector<InstanceLayoutPiece<N,T> *> pieces;
    // placeholder for lookup structure (e.g. K-D tree)
  };

  template <int N, typename T>
  std::ostream& operator<<(std::ostream& os, const InstancePieceList<N,T>& ipl);

  template <int N, typename T>
  class InstanceLayout : public InstanceLayoutGeneric {
  public:
    InstanceLayout(void);
    virtual ~InstanceLayout(void);

    // adjusts offsets of pieces to start from 'base_offset'
    virtual void relocate(size_t base_offset);

    virtual void print(std::ostream& os) const;

    // computes the offset of the specified field for an element - this
    //  is generally much less efficient than using a layout-specific accessor
    size_t calculate_offset(ZPoint<N,T> p, FieldID fid) const;

    ZIndexSpace<N,T> space;
    std::vector<InstancePieceList<N,T> > piece_lists;
  };

  // accessor stuff should eventually move to a different header file I think

  // Privileges for using an accessor
  enum AccessorPrivilege {
    ACCESSOR_PRIV_NONE   = 0x00000000,
    ACCESSOR_PRIV_READ   = 0x00000001,
    ACCESSOR_PRIV_WRITE  = 0x00000002,
    ACCESSOR_PRIV_REDUCE = 0x00000004,
    ACCESSOR_PRIV_ALL    = 0x00000007,
  };

  // an instance accessor based on an affine linearization of an index space
  template <typename FT, int N, typename T = int>
  class AffineAccessor {
  public:
    // Note: All constructors except the default one must currently be called 
    // on the host so there are no __CUDA_HD__ qualifiers

    // TODO: Sean check if this is safe for a default constructor
    __CUDA_HD__
    AffineAccessor(void) : base(0) { }
    // NOTE: these constructors will die horribly if the conversion is not
    //  allowed - call is_compatible(...) first if you're not sure

    // implicitly tries to cover the entire instance's domain
    AffineAccessor(RegionInstance inst, ptrdiff_t field_offset);

    // limits domain to a subrectangle
    AffineAccessor(RegionInstance inst, ptrdiff_t field_offset, const ZRect<N,T>& subrect);

    // for higher-level interfaces to use, the INST type must implement the following methods
    // - RegionInstance get_instance(unsigned field_id, ptrdiff_t &field_offset)
    // - ZIndexSpace<N,T> get_bounds(void) -- for bounds checks
    // - AccessorPrivilege get_accessor_privileges(void) -- for privilege checks
    template <typename INST>
    AffineAccessor(const INST &instance, unsigned field_id);
    template <typename INST>
    AffineAccessor(const INST &instance, unsigned field_id, const ZRect<N,T>& subrect);

    __CUDA_HD__
    ~AffineAccessor(void);

    static bool is_compatible(RegionInstance inst, ptrdiff_t field_offset);
    static bool is_compatible(RegionInstance inst, ptrdiff_t field_offset, const ZRect<N,T>& subrect);

    template <typename INST>
    static bool is_compatible(const INST &instance, unsigned field_id);
    template <typename INST>
    static bool is_compatible(const INST &instance, unsigned field_id, const ZRect<N,T>& subrect);

    __CUDA_HD__
    FT *ptr(const ZPoint<N,T>& p) const;
    __CUDA_HD__
    FT read(const ZPoint<N,T>& p) const;
    __CUDA_HD__
    void write(const ZPoint<N,T>& p, FT newval) const;

    __CUDA_HD__
    FT& operator[](const ZPoint<N,T>& p);
    __CUDA_HD__
    const FT& operator[](const ZPoint<N,T>& p) const;

  //protected:
  //friend
  // std::ostream& operator<<(std::ostream& os, const AffineAccessor<FT,N,T>& a);
#define REALM_ACCESSOR_DEBUG
#ifdef REALM_ACCESSOR_DEBUG
    RegionInstance dbg_inst;
    ZRect<N,T> dbg_bounds;
#ifdef __CUDACC__
#error "REALM_ACCESSOR_DEBUG macro for AffineAccessor not supported for GPU code"
#endif
#endif
    intptr_t base;
    ZPoint<N, ptrdiff_t> strides;
#ifdef PRIVILEGE_CHECKS
    AccessorPrivilege privileges;
#endif
#ifdef BOUNDS_CHECKS
    ZIndexSpace<N,T> bounds;
#ifdef __CUDACC__
#error "BOUNDS_CHECKS macro for AffineAccessor not supported for GPU code"
#endif
#endif
  protected:
    __CUDA_HD__
    FT* get_ptr(const ZPoint<N,T>& p) const;
  };

  template <typename FT, int N, typename T>
  std::ostream& operator<<(std::ostream& os, const AffineAccessor<FT,N,T>& a);

}; // namespace Realm

#include "inst_layout.inl"

#endif // ifndef REALM_INST_LAYOUT_H


