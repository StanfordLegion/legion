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

// NOP but useful for IDE's
#include "indexspace.h"

namespace Realm {

  // TODO: move to helper include file
  template <typename T, typename T2>
  inline T roundup(T val, T2 step)
  {
    T rem = val % step;
    if(rem == 0)
      return val;
    else
      return val + (step - rem);
  }

  template <typename T>
  inline T max(T a, T b)
  {
    return((a > b) ? a : b);
  }

  template <typename T>
  inline T gcd(T a, T b)
  {
    while(a != b) {
      if(a > b)
	a -= b;
      else
	b -= a;
    }
    return a;
  }

  template <typename T>
  inline T lcm(T a, T b)
  {
    // TODO: more efficient way?
    return(a * b / gcd(a, b));
  }

  namespace PPrint {

    template <typename T>
    struct VectorOstreamHelper {
      VectorOstreamHelper(const std::vector<T>& _vec, const char *_sep,
			  const char *_pre, const char *_post)
	: vec(_vec), sep(_sep), pre(_pre), post(_post)
      {}

      const std::vector<T>& vec;
      const char *sep, *pre, *post;
    };

    template <typename T>
    std::ostream& operator<<(std::ostream& os, VectorOstreamHelper<T> voh)
    {
      if(!voh.vec.empty()) {
	if(voh.pre) os << voh.pre;
	typename std::vector<T>::const_iterator it = voh.vec.begin();
	os << *it++;
	while(it != voh.vec.end()) {
	  if(voh.sep) os << voh.sep;
	  os << *it++;
	}
	if(voh.post) os << voh.post;
      }
      return os;
    }

    template <typename T>
    VectorOstreamHelper<T> pprint(const std::vector<T>& vec,
				  const char *sep = 0,
				  const char *pre = 0,
				  const char *post = 0)
    {
      return VectorOstreamHelper<T>(vec, sep, pre, post);
    }
  };
  

  ////////////////////////////////////////////////////////////////////////
  //
  // class InstanceLayoutConstraints

  inline InstanceLayoutConstraints::InstanceLayoutConstraints(const std::vector<size_t>& field_sizes,
							      size_t block_size)
  {
    // use the field sizes to generate "offsets" as unique IDs
    switch(block_size) {
    case 0:
      {
	// SOA - each field is its own "group"
	field_groups.resize(field_sizes.size());
	size_t offset = 0;
	for(size_t i = 0; i < field_sizes.size(); i++) {
	  field_groups[i].resize(1);
	  field_groups[i][0].field_id = offset;
	  field_groups[i][0].offset = -1;
	  field_groups[i][0].size = field_sizes[i];
	  field_groups[i][0].alignment = field_sizes[i]; // natural alignment 
	  offset += field_sizes[i];
	}
	break;
      }

    case 1:
      {
	// AOS - all field_groups in same group
	field_groups.resize(1);
	field_groups[0].resize(field_sizes.size());
	size_t offset = 0;
	for(size_t i = 0; i < field_sizes.size(); i++) {
	  field_groups[0][i].field_id = offset;
	  field_groups[0][i].offset = -1;
	  field_groups[0][i].size = field_sizes[i];
	  field_groups[0][i].alignment = field_sizes[i]; // natural alignment 
	  offset += field_sizes[i];
	}
	break;
      }

    default:
      {
	// hybrid - blech
	assert(0);
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class InstanceLayoutGeneric

  inline InstanceLayoutGeneric::InstanceLayoutGeneric(void)
  {}

  inline InstanceLayoutGeneric::~InstanceLayoutGeneric(void)
  {}

  template <int N, typename T>
  inline /*static*/ InstanceLayoutGeneric *InstanceLayoutGeneric::choose_instance_layout(ZIndexSpace<N,T> is,
											 const InstanceLayoutConstraints& ilc)
  {
    InstanceLayout<N,T> *layout = new InstanceLayout<N,T>;
    layout->bytes_used = 0;
    layout->alignment_reqd = 1;
    layout->space = is;

    std::vector<ZRect<N,T> > piece_bounds;
    if(is.dense()) {
      // dense case is nice and simple
      piece_bounds.push_back(is.bounds);
    } else {
      // we need precise data for non-dense index spaces (the original
      //  'bounds' on the ZIndexSpace is often VERY conservative)
      SparsityMapPublicImpl<N,T> *impl = is.sparsity.impl();
      const std::vector<SparsityMapEntry<N,T> >& entries = impl->get_entries();
      if(!entries.empty()) {
	// TODO: set some sort of threshold for merging entries
	typename std::vector<SparsityMapEntry<N,T> >::const_iterator it = entries.begin();
	ZRect<N,T> bbox = it->bounds;
	while(it != entries.end()) {
	  ++it;
	  bbox = bbox.union_bbox(it->bounds);
	}
	piece_bounds.push_back(bbox);
      }
    }

    // TODO: merge identical piece lists
    layout->piece_lists.resize(ilc.field_groups.size());
    for(size_t li = 0; li < ilc.field_groups.size(); li++) {
      const InstanceLayoutConstraints::FieldGroup& fg = ilc.field_groups[li];
      InstancePieceList<N,T>& pl = layout->piece_lists[li];
      pl.pieces.reserve(piece_bounds.size());

      // figure out layout of fields in group - this is constant across the
      //  pieces
      size_t gsize = 0;
      size_t galign = 1;
      // we can't set field offsets in a single pass because we don't know
      //  the whole group's alignment until we look at every field
      std::map<FieldID, int> field_offsets;
      for(std::vector<InstanceLayoutConstraints::FieldInfo>::const_iterator it2 = fg.begin();
	  it2 != fg.end();
	  ++it2) {
	size_t offset;
	if(it2->offset >= 0) {
	  offset = it2->offset;
	} else {
	  // if not specified, field goes at the end of all known fields
	  //  (or a bit past if alignment is a concern)
	  offset = gsize;
	  if(it2->alignment > 1)
	    offset = roundup(offset, it2->alignment);
	}
	// increase size and alignment if needed
	gsize = max(gsize, offset + it2->size);
	if((it2->alignment > 1) && ((galign % it2->alignment) != 0))
	  galign = lcm(galign, (size_t)(it2->alignment));
	
	field_offsets[it2->field_id] = offset;
      }
      if(galign > 1) {
	// group size needs to be rounded up to match group alignment
	gsize = roundup(gsize, galign);

	// overall instance alignment layout must be compatible with group
	layout->alignment_reqd = lcm(layout->alignment_reqd, galign);
      }
      // now that the group offset is properly aligned, we can set the
      //  actual field offsets
      for(std::map<FieldID, int>::const_iterator it2 = field_offsets.begin();
	  it2 != field_offsets.end();
	  ++it2) {
	// should not have seen this field before
	assert(layout->fields.count(it2->first) == 0);
	InstanceLayoutGeneric::FieldLayout& fl = layout->fields[it2->first];
	fl.list_idx = li;
	fl.rel_offset = /*group_offset +*/ it2->second;
      }

      for(typename std::vector<ZRect<N,T> >::const_iterator it = piece_bounds.begin();
	  it != piece_bounds.end();
	  ++it) {
	ZRect<N,T> bbox = *it;
	// TODO: bloat bbox for block size if desired
	ZRect<N,T> bloated = bbox;

	// always create an affine piece for now
	AffineLayoutPiece<N,T> *piece = new AffineLayoutPiece<N,T>;
	piece->bounds = bbox;

	// starting point for piece is first galign-aligned location above
	//  existing pieces
	size_t piece_start = roundup(layout->bytes_used, galign);
	piece->offset = piece_start;
	// always do fortran order for now
	size_t stride = gsize;
	for(int i = 0; i < N; i++) {
	  piece->strides[i] = stride;
	  piece->offset -= bloated.lo[i] * stride;
	  stride *= (bloated.hi[i] - bloated.lo[i] + 1);
	}

	// final value of stride is total bytes used by piece - use that
	//  to set new instance footprint
	layout->bytes_used = piece_start + stride;

	pl.pieces.push_back(piece);
      }
    }

    return layout;
  }

  inline std::ostream& operator<<(std::ostream& os, const InstanceLayoutGeneric& ilg)
  {
    ilg.print(os);
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class InstanceLayoutPiece<N,T>

  template <int N, typename T>
  inline InstanceLayoutPiece<N,T>::InstanceLayoutPiece(void)
    : layout_type(InvalidLayoutType)
  {}

  template <int N, typename T>
  inline InstanceLayoutPiece<N,T>::InstanceLayoutPiece(LayoutType _layout_type)
    : layout_type(_layout_type)
  {}

  template <int N, typename T>
  inline InstanceLayoutPiece<N,T>::~InstanceLayoutPiece(void)
  {}

  template <int N, typename T>
  inline std::ostream& operator<<(std::ostream& os, const InstanceLayoutPiece<N,T>& ilp)
  {
    ilp.print(os);
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class AffineLayoutPiece<N,T>

  template <int N, typename T>
  inline AffineLayoutPiece<N,T>::AffineLayoutPiece(void)
    : InstanceLayoutPiece<N,T>(InstanceLayoutPiece<N,T>::AffineLayoutType)
  {}

  template <int N, typename T>
  inline void AffineLayoutPiece<N,T>::relocate(size_t base_offset)
  {
    offset += base_offset;
  }

  template <int N, typename T>
  void AffineLayoutPiece<N,T>::print(std::ostream& os) const
  {
    os << this->bounds << "->affine(" << strides << "+" << offset << ")";
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class InstancePieceList<N,T>

  template <int N, typename T>
  inline InstancePieceList<N,T>::InstancePieceList(void)
  {}

  template <int N, typename T>
  inline InstancePieceList<N,T>::~InstancePieceList(void)
  {
    for(typename std::vector<InstanceLayoutPiece<N,T> *>::const_iterator it = pieces.begin();
	it != pieces.end();
	++it)
      delete *it;
  }

  template <int N, typename T>
  inline const InstanceLayoutPiece<N,T> *InstancePieceList<N,T>::find_piece(ZPoint<N,T> p) const
  {
    for(typename std::vector<InstanceLayoutPiece<N,T> *>::const_iterator it = pieces.begin();
	it != pieces.end();
	++it)
      if((*it)->bounds.contains(p))
	return *it;
    return 0;
  }

  template <int N, typename T>
  inline void InstancePieceList<N,T>::relocate(size_t base_offset)
  {
    for(typename std::vector<InstanceLayoutPiece<N,T> *>::const_iterator it = pieces.begin();
	it != pieces.end();
	++it)
      (*it)->relocate(base_offset);
  }

  template <int N, typename T>
  inline std::ostream& operator<<(std::ostream& os, const InstancePieceList<N,T>& ipl)
  {
    os << '[';
    bool first = true;
    for(typename std::vector<InstanceLayoutPiece<N,T> *>::const_iterator it = ipl.pieces.begin();
	it != ipl.pieces.end();
	++it) {
      if(!first) os << ", ";
      first = false;
      os << *(*it);
    }
    os << ']';
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class InstanceLayout<N,T>

  template <int N, typename T>
  inline InstanceLayout<N,T>::InstanceLayout(void)
  {}

  template <int N, typename T>
  InstanceLayout<N,T>::~InstanceLayout(void)
  {}

  // adjusts offsets of pieces to start from 'base_offset'
  template <int N, typename T>
  void InstanceLayout<N,T>::relocate(size_t base_offset)
  {
    for(typename std::vector<InstancePieceList<N,T> >::iterator it = piece_lists.begin();
	it != piece_lists.end();
	++it)
      it->relocate(base_offset);
  }

  template <int N, typename T>
  void InstanceLayout<N,T>::print(std::ostream& os) const
  {
    os << "Layout(bytes=" << bytes_used << ", align=" << alignment_reqd
       << ", fields={";
    bool first = true;
    for(std::map<FieldID, FieldLayout>::const_iterator it = fields.begin();
	it != fields.end();
	++it) {
      if(!first) os << ", ";
      first = false;
      os << it->first << "=" << it->second.list_idx << "+" << it->second.rel_offset;
    }
    os << "}, lists=[";
    {
      bool first = true;
      for(typename std::vector<InstancePieceList<N,T> >::const_iterator it = piece_lists.begin();
	  it != piece_lists.end();
	  ++it) {
	if(!first) os << ", ";
	first = false;
	os << *it;
      }
    }
    os << "])";
  }

  // computes the offset of the specified field for an element - this
  //  is generally much less efficient than using a layout-specific accessor
  template <int N, typename T>
  inline size_t InstanceLayout<N,T>::calculate_offset(ZPoint<N,T> p, FieldID fid) const
  {
    // first look up the field to see which piece list it uses (and get offset)
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = fields.find(fid);
    assert(it != fields.end());

    const InstanceLayoutPiece<N,T> *ilp = piece_lists[it->second.list_idx].find_piece(p);
    assert(ilp != 0);
    size_t offset = 0;
    switch(ilp->layout_type) {
    case InstanceLayoutPiece<1,coord_t>::AffineLayoutType:
      {
	const AffineLayoutPiece<N,T> *alp = static_cast<const AffineLayoutPiece<N,T> *>(ilp);
	offset = alp->offset + alp->strides.dot(p) + it->second.rel_offset;
	break;
      }
    default:
      assert(0);
    }
    return offset;
  }
      

  ////////////////////////////////////////////////////////////////////////
  //
  // class AffineAccessor<FT,N,T>

  // NOTE: these constructors will die horribly if the conversion is not
  //  allowed - call is_compatible(...) first if you're not sure

  // implicitly tries to cover the entire instance's domain
  template <typename FT, int N, typename T>
  inline AffineAccessor<FT,N,T>::AffineAccessor(RegionInstance inst, ptrdiff_t field_offset)
  {
    const InstanceLayout<N,T> *layout = dynamic_cast<const InstanceLayout<N,T> *>(inst.get_layout());
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_offset);
    assert(it != layout->fields.end());
    const InstancePieceList<N,T>& ipl = layout->piece_lists[it->second.list_idx];
    
    // this constructor only works if there's exactly one piece and it's affine
    assert(ipl.pieces.size() == 1);
    const InstanceLayoutPiece<N,T> *ilp = ipl.pieces[0];
    assert((ilp->layout_type == InstanceLayoutPiece<N,T>::AffineLayoutType));
    const AffineLayoutPiece<N,T> *alp = static_cast<const AffineLayoutPiece<N,T> *>(ilp);
    base = reinterpret_cast<intptr_t>(inst.pointer_untyped(0,
							   layout->bytes_used));
    assert(base != 0);
    base += alp->offset + it->second.rel_offset;
    strides = alp->strides;
#ifdef REALM_ACCESSOR_DEBUG
    dbg_inst = inst;
    dbg_bounds = alp->bounds;
#endif
  }

  template <typename FT, int N, typename T>
  template <typename INST>
  inline AffineAccessor<FT,N,T>::AffineAccessor(const INST &instance, unsigned field_id)
  {
    ptrdiff_t field_offset = 0;
    RegionInstance inst = instance.get_instance(field_id, field_offset);
    const InstanceLayout<N,T> *layout = dynamic_cast<const InstanceLayout<N,T> *>(inst.get_layout());
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_offset);
    assert(it != layout->fields.end());
    const InstancePieceList<N,T>& ipl = layout->piece_lists[it->second.list_idx];
    
    // this constructor only works if there's exactly one piece and it's affine
    assert(ipl.pieces.size() == 1);
    const InstanceLayoutPiece<N,T> *ilp = ipl.pieces[0];
    assert((ilp->layout_type == InstanceLayoutPiece<N,T>::AffineLayoutType));
    const AffineLayoutPiece<N,T> *alp = static_cast<const AffineLayoutPiece<N,T> *>(ilp);
    base = reinterpret_cast<intptr_t>(inst.pointer_untyped(0,
							   layout->bytes_used));
    assert(base != 0);
    base += alp->offset + it->second.rel_offset;
    strides = alp->strides;
#ifdef REALM_ACCESSOR_DEBUG
    dbg_inst = inst;
    dbg_bounds = alp->bounds;
#endif
#ifdef PRIVILEGE_CHECKS
    privileges = instance.get_accessor_privileges();
#endif
#ifdef BOUNDS_CHECKS
    bounds = instance.template get_bounds<N,T>();
#endif
  }

  // limits domain to a subrectangle
  template <typename FT, int N, typename T>
  AffineAccessor<FT,N,T>::AffineAccessor(RegionInstance inst, ptrdiff_t field_offset, const ZRect<N,T>& subrect)
  {
    const InstanceLayout<N,T> *layout = dynamic_cast<const InstanceLayout<N,T> *>(inst.get_layout());
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_offset);
    assert(it != layout->fields.end());
    const InstancePieceList<N,T>& ipl = layout->piece_lists[it->second.list_idx];
    
    // find the piece that holds the lo corner of the subrect and insist it
    //  exists, covers the whole subrect, and is affine
    const InstanceLayoutPiece<N,T> *ilp = ipl.find_piece(subrect.lo);
    assert(ilp && ilp->bounds.contains(subrect));
    assert((ilp->layout_type == InstanceLayoutPiece<N,T>::AffineLayoutType));
    const AffineLayoutPiece<N,T> *alp = static_cast<const AffineLayoutPiece<N,T> *>(ilp);
    base = reinterpret_cast<intptr_t>(inst.pointer_untyped(0,
							   layout->bytes_used));
    assert(base != 0);
    base += alp->offset + it->second.rel_offset;
    strides = alp->strides;
#ifdef REALM_ACCESSOR_DEBUG
    dbg_inst = inst;
    dbg_bounds = alp->bounds;
#endif
  }

  template <typename FT, int N, typename T>
  template <typename INST>
  inline AffineAccessor<FT,N,T>::AffineAccessor(const INST &instance, unsigned field_id, const ZRect<N,T>& subrect)
  {
    ptrdiff_t field_offset = 0;
    RegionInstance inst = instance.get_instance(field_id, field_offset);
    const InstanceLayout<N,T> *layout = dynamic_cast<const InstanceLayout<N,T> *>(inst.get_layout());
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_offset);
    assert(it != layout->fields.end());
    const InstancePieceList<N,T>& ipl = layout->piece_lists[it->second.list_idx];
    
    // find the piece that holds the lo corner of the subrect and insist it
    //  exists, covers the whole subrect, and is affine
    const InstanceLayoutPiece<N,T> *ilp = ipl.find_piece(subrect.lo);
    assert(ilp && ilp->bounds.contains(subrect));
    assert((ilp->layout_type == InstanceLayoutPiece<N,T>::AffineLayoutType));
    const AffineLayoutPiece<N,T> *alp = static_cast<const AffineLayoutPiece<N,T> *>(ilp);
    base = reinterpret_cast<intptr_t>(inst.pointer_untyped(0,
							   layout->bytes_used));
    assert(base != 0);
    base += alp->offset + it->second.rel_offset;
    strides = alp->strides;
#ifdef REALM_ACCESSOR_DEBUG
    dbg_inst = inst;
    dbg_bounds = alp->bounds;
#endif
#ifdef PRIVILEGE_CHECKS
    privileges = instance.get_accessor_privileges();
#endif
#ifdef BOUNDS_CHECKS
    // TODO: verify here that subrect is wholly contained in 'instance'?
    bounds = subrect;
#endif
  }

  template <typename FT, int N, typename T>
  inline AffineAccessor<FT,N,T>::~AffineAccessor(void)
  {}

  template <typename FT, int N, typename T>
  inline /*static*/ bool AffineAccessor<FT,N,T>::is_compatible(RegionInstance inst, ptrdiff_t field_offset)
  {
    const InstanceLayout<N,T> *layout = dynamic_cast<const InstanceLayout<N,T> *>(inst.get_layout());
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_offset);
    if(it == layout->fields.end())
      return false;
    const InstancePieceList<N,T>& ipl = layout->piece_lists[it->second.list_idx];
    
    // this constructor only works if there's exactly one piece and it's affine
    if(ipl.pieces.size() != 1)
      return false;
    const InstanceLayoutPiece<N,T> *ilp = ipl.pieces[0];
    if(ilp->layout_type != InstanceLayoutPiece<N,T>::AffineLayoutType)
      return false;
    void *base = inst.pointer_untyped(0, layout->bytes_used);
    if(base == 0)
      return false;

    // all checks passed!
    return true;
  }

  template <typename FT, int N, typename T>
  inline /*static*/ bool AffineAccessor<FT,N,T>::is_compatible(RegionInstance inst, ptrdiff_t field_offset, const ZRect<N,T>& subrect)
  {
    const InstanceLayout<N,T> *layout = dynamic_cast<const InstanceLayout<N,T> *>(inst.get_layout());
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_offset);
    if(it == layout->fields.end())
      return false;
    const InstancePieceList<N,T>& ipl = layout->piece_lists[it->second.list_idx];
    
    // find the piece that holds the lo corner of the subrect and insist it
    //  exists, covers the whole subrect, and is affine
    const InstanceLayoutPiece<N,T> *ilp = ipl.find_piece(subrect.lo);
    if(!(ilp && ilp->bounds.contains(subrect)))
      return false;
    if(ilp->layout_type != InstanceLayoutPiece<N,T>::AffineLayoutType)
      return false;
    void *base = inst.pointer_untyped(0, layout->bytes_used);
    if(base == 0)
      return false;

    // all checks passed!
    return true;
  }

  template <typename FT, int N, typename T>
  template <typename INST>
  inline /*static*/ bool AffineAccessor<FT,N,T>::is_compatible(const INST &instance, unsigned field_id)
  {
    ptrdiff_t field_offset = 0;
    RegionInstance inst = instance.get_instance(field_id, field_offset);
    return is_compatible(inst, field_offset);
  }

  template <typename FT, int N, typename T>
  template <typename INST>
  inline /*static*/ bool AffineAccessor<FT,N,T>::is_compatible(const INST &instance, unsigned field_id, const ZRect<N,T>& subrect)
  {
    ptrdiff_t field_offset = 0;
    RegionInstance inst = instance.get_instance(field_id, field_offset);
    return is_compatible(inst, field_offset, subrect);
  }

  template <typename FT, int N, typename T>
  inline FT *AffineAccessor<FT,N,T>::ptr(const ZPoint<N,T>& p) const
  {
#ifdef PRIVILEGE_CHECKS
    assert(privileges & ACCESSOR_PRIV_ALL);
#endif
#ifdef BOUNDS_CHECKS
    assert(bounds.contains(p));
#endif
    intptr_t rawptr = base;
    for(int i = 0; i < N; i++) rawptr += p[i] * strides[i];
    return reinterpret_cast<FT *>(rawptr);
  }

  template <typename FT, int N, typename T>
  inline FT AffineAccessor<FT,N,T>::read(const ZPoint<N,T>& p) const
  {
#ifdef PRIVILEGE_CHECKS
    assert(privileges & ACCESSOR_PRIV_READ);
#endif
#ifdef BOUNDS_CHECKS
    assert(bounds.contains(p));
#endif
    return *(this->ptr(p));
  }

  template <typename FT, int N, typename T>
  inline void AffineAccessor<FT,N,T>::write(const ZPoint<N,T>& p, FT newval) const
  {
#ifdef PRIVILEGE_CHECKS
    assert(privileges & ACCESSOR_PRIV_WRITE);
#endif
#ifdef BOUNDS_CHECKS
    assert(bounds.contains(p));
#endif
    *(ptr(p)) = newval;
  }

  template <typename FT, int N, typename T>
  inline std::ostream& operator<<(std::ostream& os, const AffineAccessor<FT,N,T>& a)
  {
    os << "AffineAccessor{ base=" << std::hex << a.base << std::dec << " strides=" << a.strides;
#ifdef REALM_ACCESSOR_DEBUG
    os << " inst=" << a.dbg_inst;
    os << " bounds=" << a.dbg_bounds;
    os << "->[" << std::hex << a.ptr(a.dbg_bounds.lo) << "," << a.ptr(a.dbg_bounds.hi)+1 << std::dec << "]";
#endif
    os << " }";
    return os;
  }

}; // namespace Realm
