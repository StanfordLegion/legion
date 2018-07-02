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

// NOP but useful for IDE's
#include "realm/indexspace.h"

namespace Realm {

  // TODO: move to helper include file
  template <typename T, typename T2>
  inline T round_up(T val, T2 step)
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
  

  ////////////////////////////////////////////////////////////////////////
  //
  // class InstanceLayoutGeneric

  inline InstanceLayoutGeneric::InstanceLayoutGeneric(void)
  {}

  template <typename S>
  /*static*/ inline InstanceLayoutGeneric *InstanceLayoutGeneric::deserialize_new(S& deserializer)
  {
    return Serialization::PolymorphicSerdezHelper<InstanceLayoutGeneric>::deserialize_new(deserializer);
  }

  inline InstanceLayoutGeneric::~InstanceLayoutGeneric(void)
  {}

  template <int N, typename T>
  inline /*static*/ InstanceLayoutGeneric *InstanceLayoutGeneric::choose_instance_layout(IndexSpace<N,T> is,
											 const InstanceLayoutConstraints& ilc,
                                                                                         const int dim_order[N])
  {
    InstanceLayout<N,T> *layout = new InstanceLayout<N,T>;
    layout->bytes_used = 0;
    // require 32B alignment of each instance piece for vectorizing goodness
    layout->alignment_reqd = 32;
    layout->space = is;

    std::vector<Rect<N,T> > piece_bounds;
    if(is.dense()) {
      // dense case is nice and simple
      if(!is.bounds.empty())
	piece_bounds.push_back(is.bounds);
    } else {
      // we need precise data for non-dense index spaces (the original
      //  'bounds' on the IndexSpace is often VERY conservative)
      SparsityMapPublicImpl<N,T> *impl = is.sparsity.impl();
      const std::vector<SparsityMapEntry<N,T> >& entries = impl->get_entries();
      if(!entries.empty()) {
	// TODO: set some sort of threshold for merging entries
	typename std::vector<SparsityMapEntry<N,T> >::const_iterator it = entries.begin();
	Rect<N,T> bbox = is.bounds.intersection(it->bounds);
	while(++it != entries.end())
	  bbox = bbox.union_bbox(is.bounds.intersection(it->bounds));
	if(!bbox.empty())
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
      std::map<FieldID, int> field_sizes;
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
	    offset = round_up(offset, it2->alignment);
	}
	// increase size and alignment if needed
	gsize = max(gsize, offset + it2->size);
	if((it2->alignment > 1) && ((galign % it2->alignment) != 0))
	  galign = lcm(galign, size_t(it2->alignment));
	field_offsets[it2->field_id] = offset;
	field_sizes[it2->field_id] = it2->size;
      }
      if(galign > 1) {
	// group size needs to be rounded up to match group alignment
	gsize = round_up(gsize, galign);

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
	fl.size_in_bytes = field_sizes[it2->first];
      }

      for(typename std::vector<Rect<N,T> >::const_iterator it = piece_bounds.begin();
	  it != piece_bounds.end();
	  ++it) {
	Rect<N,T> bbox = *it;
	// TODO: bloat bbox for block size if desired
	Rect<N,T> bloated = bbox;

	// always create an affine piece for now
	AffineLayoutPiece<N,T> *piece = new AffineLayoutPiece<N,T>;
	piece->bounds = bbox;

	// starting point for piece is first galign-aligned location above
	//  existing pieces
	size_t piece_start = round_up(layout->bytes_used, galign);
	piece->offset = piece_start;
	size_t stride = gsize;
	for(int i = 0; i < N; i++) {
          const int dim = dim_order[i];
          assert((0 <= dim) && (dim < N));
	  piece->strides[dim] = stride;
	  piece->offset -= bloated.lo[dim] * stride;
	  stride *= (bloated.hi[dim] - bloated.lo[dim] + 1);
	}

	// final value of stride is total bytes used by piece - use that
	//  to set new instance footprint
	layout->bytes_used = piece_start + stride;

	pl.pieces.push_back(piece);
      }
    }

    return layout;
  }

  template <typename S>
  inline bool serialize(S& serializer, const InstanceLayoutGeneric& ilg)
  {
    return Serialization::PolymorphicSerdezHelper<InstanceLayoutGeneric>::serialize(serializer, ilg);
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
  template <typename S>
  /*static*/ inline InstanceLayoutPiece<N,T> *InstanceLayoutPiece<N,T>::deserialize_new(S& deserializer)
  {
    return Serialization::PolymorphicSerdezHelper<InstanceLayoutPiece<N,T> >::deserialize_new(deserializer);
  }

  template <int N, typename T>
  inline InstanceLayoutPiece<N,T>::~InstanceLayoutPiece(void)
  {}

  template <int N, typename T, typename S>
  inline bool serialize(S& serializer, const InstanceLayoutPiece<N,T>& ilp)
  {
    return Serialization::PolymorphicSerdezHelper<InstanceLayoutPiece<N,T> >::serialize(serializer, ilp);
  }

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
  template <typename S>
  /*static*/ inline InstanceLayoutPiece<N,T> *AffineLayoutPiece<N,T>::deserialize_new(S& s)
  {
    AffineLayoutPiece<N,T> *alp = new AffineLayoutPiece<N,T>;
    if((s >> alp->bounds) &&
       (s >> alp->strides) &&
       (s >> alp->offset)) {
      return alp;
    } else {
      delete alp;
      return 0;
    }
  }

  template <int N, typename T>
  inline size_t AffineLayoutPiece<N,T>::calculate_offset(const Point<N,T>& p) const
  {
    return offset + strides.dot(p);
  }

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

  template <int N, typename T>
  template <typename S>
  inline bool AffineLayoutPiece<N,T>::serialize(S& s) const
  {
    return ((s << this->bounds) &&
	    (s << strides) &&
	    (s << offset));
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
  inline const InstanceLayoutPiece<N,T> *InstancePieceList<N,T>::find_piece(Point<N,T> p) const
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

  template <int N, typename T, typename S>
  inline bool serialize(S& s, const InstancePieceList<N,T>& ipl)
  {
    return ipl.serialize(s);
  }

  template <int N, typename T, typename S>
  inline bool deserialize(S& s, InstancePieceList<N,T>& ipl)
  {
    return ipl.deserialize(s);
  }

  template <int N, typename T>
  template <typename S>
  inline bool InstancePieceList<N,T>::serialize(S& s) const
  {
    size_t len = pieces.size();
    if(!(s << len)) return false;
    for(size_t i = 0; i < len; i++)
      if(!(s << *(pieces[i]))) return false;
    return true;
  }
    
  template <int N, typename T>
  template <typename S>
  inline bool InstancePieceList<N,T>::deserialize(S& s)
  {
    size_t len;
    if(!(s >> len)) return false;
    pieces.resize(len);
    for(size_t i = 0; i < len; i++) {
      InstanceLayoutPiece<N,T> *ilp = InstanceLayoutPiece<N,T>::deserialize_new(s);
      if(!ilp)
	return false;
      pieces[i] = ilp;
    }
    return true;
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
  template <typename S>
  /*static*/ InstanceLayoutGeneric *InstanceLayout<N,T>::deserialize_new(S& s)
  {
    InstanceLayout<N,T> *il = new InstanceLayout<N,T>;
    if((s >> il->bytes_used) &&
       (s >> il->alignment_reqd) &&
       (s >> il->fields) &&
       (s >> il->space) &&
       (s >> il->piece_lists)) {
      return il;
    } else {
      delete il;
      return 0;
    }
  }

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
  inline size_t InstanceLayout<N,T>::calculate_offset(Point<N,T> p, FieldID fid) const
  {
    // first look up the field to see which piece list it uses (and get offset)
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = fields.find(fid);
    assert(it != fields.end());

    const InstanceLayoutPiece<N,T> *ilp = piece_lists[it->second.list_idx].find_piece(p);
    assert(ilp != 0);
    size_t offset = ilp->calculate_offset(p);
    // add in field's offset
    offset += it->second.rel_offset;
    return offset;
  }

  template <int N, typename T>
  template <typename S>
  inline bool InstanceLayout<N,T>::serialize(S& s) const
  {
    return ((s << bytes_used) &&
	    (s << alignment_reqd) &&
	    (s << fields) &&
	    (s << space) &&
	    (s << piece_lists));
  }
      

  ////////////////////////////////////////////////////////////////////////
  //
  // class AccessorRefHelper<FT>

  template <typename FT>
  inline AccessorRefHelper<FT>::AccessorRefHelper(RegionInstance _inst,
						  size_t _offset)
    : inst(_inst)
    , offset(_offset)
  {}

  // "read"
  template <typename FT>
  inline AccessorRefHelper<FT>::operator FT(void) const
  {
    FT val;
    inst.read_untyped(offset, &val, sizeof(FT));
    return val;
  }

  // "write"
  template <typename FT>
  inline AccessorRefHelper<FT>& AccessorRefHelper<FT>::operator=(const FT& newval)
  {
    inst.write_untyped(offset, &newval, sizeof(FT));
    return *this;
  }

  template <typename FT>
  inline AccessorRefHelper<FT>& AccessorRefHelper<FT>::operator=(
                                  const AccessorRefHelper<FT>& rhs)
  {
    const FT newval = rhs; 
    inst.write_untyped(offset, &newval, sizeof(FT));
    return *this;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class GenericAccessor<FT,N,T>

  template <typename FT, int N, typename T>
  inline GenericAccessor<FT,N,T>::GenericAccessor(void)
    : inst(RegionInstance::NO_INST)
    , piece_list(0)
    , rel_offset(0)
    , prev_piece(0)
  {}

  // GenericAccessor constructors always work, but the is_compatible(...)
  //  calls are still available for templated code

  // implicitly tries to cover the entire instance's domain
  template <typename FT, int N, typename T>
  inline GenericAccessor<FT,N,T>::GenericAccessor(RegionInstance inst,
						  FieldID field_id,
						  size_t subfield_offset /*= 0*/)
  {
    this->inst = inst;

    // find the right piece list for our field
    const InstanceLayout<N,T> *layout = dynamic_cast<const InstanceLayout<N,T> *>(inst.get_layout());
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_id);
    assert(it != layout->fields.end());

    this->piece_list = &layout->piece_lists[it->second.list_idx];
    this->rel_offset = it->second.rel_offset + subfield_offset;
    this->prev_piece = 0;
  }

  // limits domain to a subrectangle (doesn't matter for generic accessor)
  template <typename FT, int N, typename T>
  inline GenericAccessor<FT,N,T>::GenericAccessor(RegionInstance inst,
						  FieldID field_id,
						  const Rect<N,T>& subrect,
						  size_t subfield_offset /*= 0*/)
  {
    this->inst = inst;

    // find the right piece list for our field
    const InstanceLayout<N,T> *layout = dynamic_cast<const InstanceLayout<N,T> *>(inst.get_layout());
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_id);
    assert(it != layout->fields.end());

    this->piece_list = &layout->piece_lists[it->second.list_idx];
    this->rel_offset = it->second.rel_offset + subfield_offset;
    this->prev_piece = 0;
  }

  template <typename FT, int N, typename T>
  inline GenericAccessor<FT,N,T>::~GenericAccessor(void)
  {}

  template <typename FT, int N, typename T>
  inline /*static*/ bool GenericAccessor<FT,N,T>::is_compatible(RegionInstance inst,
								ptrdiff_t field_offset)
  {
    return true;
  }

  template <typename FT, int N, typename T>
  inline /*static*/ bool GenericAccessor<FT,N,T>::is_compatible(RegionInstance inst,
								ptrdiff_t field_offset,
								const Rect<N,T>& subrect)
  {
    return true;
  }

  template <typename FT, int N, typename T>
  template <typename INST>
  inline /*static*/ bool GenericAccessor<FT,N,T>::is_compatible(const INST &instance,
								unsigned field_id)
  {
    return true;
  }

  template <typename FT, int N, typename T>
  template <typename INST>
  inline /*static*/ bool GenericAccessor<FT,N,T>::is_compatible(const INST &instance,
								unsigned field_id,
								const Rect<N,T>& subrect)
  {
    return true;
  }

  template <typename FT, int N, typename T>
  inline FT GenericAccessor<FT,N,T>::read(const Point<N,T>& p)
  {
    size_t offset = this->get_offset(p);
    FT val;
    inst.read_untyped(offset, &val, sizeof(FT));
    return val;
  }

  template <typename FT, int N, typename T>
  inline void GenericAccessor<FT,N,T>::write(const Point<N,T>& p, FT newval)
  {
    size_t offset = this->get_offset(p);
    inst.write_untyped(offset, &newval, sizeof(FT));
  }

  // this returns a "reference" that knows how to do a read via operator FT
  //  or a write via operator=
  template <typename FT, int N, typename T>
  inline AccessorRefHelper<FT> GenericAccessor<FT,N,T>::operator[](const Point<N,T>& p)
  {
    size_t offset = this->get_offset(p);
    return AccessorRefHelper<FT>(inst, offset);
  }

  // not a const method because of the piece caching
  template <typename FT, int N, typename T>
  inline size_t GenericAccessor<FT,N,T>::get_offset(const Point<N,T>& p)
  {
    const InstanceLayoutPiece<N,T> *mypiece = prev_piece;
    if(!mypiece || !mypiece->bounds.contains(p)) {
      mypiece = piece_list->find_piece(p);
      assert(mypiece);
      prev_piece = mypiece;
    }
    size_t offset = mypiece->calculate_offset(p);
    // add in field (or subfield) adjustment
    offset += rel_offset;
    return offset;
  }

  template <typename FT, int N, typename T>
  inline std::ostream& operator<<(std::ostream& os, const GenericAccessor<FT,N,T>& a)
  {
    os << "GenericAccessor{ offset=" << a.rel_offset << " list=" << *a.piece_list << " }";
    return os;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class AffineAccessor<FT,N,T>

  template <typename FT, int N, typename T>
  __CUDA_HD__
  inline AffineAccessor<FT,N,T>::AffineAccessor(void)
    : base(0)
  {}  

  // NOTE: these constructors will die horribly if the conversion is not
  //  allowed - call is_compatible(...) first if you're not sure

  // implicitly tries to cover the entire instance's domain
  template <typename FT, int N, typename T>
  inline AffineAccessor<FT,N,T>::AffineAccessor(RegionInstance inst,
						FieldID field_id,
						size_t subfield_offset /*= 0*/)
  {
    const InstanceLayout<N,T> *layout = dynamic_cast<const InstanceLayout<N,T> *>(inst.get_layout());
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_id);
    assert(it != layout->fields.end());
    const InstancePieceList<N,T>& ipl = layout->piece_lists[it->second.list_idx];
    
    // Special case for empty instances
    if(ipl.pieces.empty()) {
      base = 0;
      for(int i = 0; i < N; i++) strides[i] = 0;
      return;
    }

    // this constructor only works if there's exactly one piece and it's affine
    assert(ipl.pieces.size() == 1);
    const InstanceLayoutPiece<N,T> *ilp = ipl.pieces[0];
    assert((ilp->layout_type == InstanceLayoutPiece<N,T>::AffineLayoutType));
    const AffineLayoutPiece<N,T> *alp = static_cast<const AffineLayoutPiece<N,T> *>(ilp);
    base = reinterpret_cast<intptr_t>(inst.pointer_untyped(0,
							   layout->bytes_used));
    assert(base != 0);
    base += alp->offset + it->second.rel_offset + subfield_offset;
    strides = alp->strides;
#ifdef REALM_ACCESSOR_DEBUG
    dbg_inst = inst;
    dbg_bounds = alp->bounds;
#endif
  }

  // limits domain to a subrectangle
  template <typename FT, int N, typename T>
  AffineAccessor<FT,N,T>::AffineAccessor(RegionInstance inst,
					 FieldID field_id, const Rect<N,T>& subrect,
					 size_t subfield_offset /*= 0*/)
  {
    const InstanceLayout<N,T> *layout = dynamic_cast<const InstanceLayout<N,T> *>(inst.get_layout());
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_id);
    assert(it != layout->fields.end());
    const InstancePieceList<N,T>& ipl = layout->piece_lists[it->second.list_idx];

    // special case for empty regions
    if(subrect.empty()) {
      base = 0;
      for(int i = 0; i < N; i++) strides[i] = 0;
    } else {
      // find the piece that holds the lo corner of the subrect and insist it
      //  exists, covers the whole subrect, and is affine
      const InstanceLayoutPiece<N,T> *ilp = ipl.find_piece(subrect.lo);
      assert(ilp && ilp->bounds.contains(subrect));
      assert((ilp->layout_type == InstanceLayoutPiece<N,T>::AffineLayoutType));
      const AffineLayoutPiece<N,T> *alp = static_cast<const AffineLayoutPiece<N,T> *>(ilp);
      base = reinterpret_cast<intptr_t>(inst.pointer_untyped(0,
							     layout->bytes_used));
      assert(base != 0);
      base += alp->offset + it->second.rel_offset + subfield_offset;
      strides = alp->strides;
    }
#ifdef REALM_ACCESSOR_DEBUG
    dbg_inst = inst;
    dbg_bounds = subrect;  // stay inside the subrect we were given
#endif
  }

  // these two constructors build accessors that incorporate an
  //  affine coordinate transform before the lookup in the actual instance
  template <typename FT, int N, typename T>
  template <int N2, typename T2>
  inline AffineAccessor<FT,N,T>::AffineAccessor(RegionInstance inst,
						const Matrix<N2, N, T2>& transform,
						const Point<N2, T2>& offset,
						FieldID field_id,
						size_t subfield_offset /*= 0*/)
  {
    // instance's dimensionality should be <N2,T2>
    const InstanceLayout<N2,T2> *layout = dynamic_cast<const InstanceLayout<N2,T2> *>(inst.get_layout());
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_id);
    assert(it != layout->fields.end());
    const InstancePieceList<N2,T2>& ipl = layout->piece_lists[it->second.list_idx];
    
    // Special case for empty instances
    if(ipl.pieces.empty()) {
      base = 0;
      for(int i = 0; i < N; i++) strides[i] = 0;
      return;
    }

    // this constructor only works if there's exactly one piece and it's affine
    assert(ipl.pieces.size() == 1);
    const InstanceLayoutPiece<N2,T2> *ilp = ipl.pieces[0];
    assert((ilp->layout_type == InstanceLayoutPiece<N2,T2>::AffineLayoutType));
    const AffineLayoutPiece<N2,T2> *alp = static_cast<const AffineLayoutPiece<N2,T2> *>(ilp);
    base = reinterpret_cast<intptr_t>(inst.pointer_untyped(0,
							   layout->bytes_used));
    assert(base != 0);
    base += alp->offset + it->second.rel_offset + subfield_offset;
    // to get the effect of transforming every accessed x to Ax+b, we
    //  add strides.b to the offset and left-multiply s'*A to get strides
    //  that go directly from the view's space to the element's offset
    //  in the instance
    base += alp->strides.dot(offset);
    for(int i = 0; i < N; i++) {
      strides[i] = 0;
      for(int j = 0; j < N2; j++)
	strides[i] += alp->strides[j] * transform.rows[j][i];
    }
#ifdef REALM_ACCESSOR_DEBUG
    dbg_inst = inst;
    // bounds are not inferrable here!
    dbg_bounds = Rect<N,T>();
#endif
  }

  // note that the subrect here is in in the accessor's indexspace
  //  (from which the corresponding subrectangle in the instance can be
  //  easily determined)
  template <typename FT, int N, typename T>
  template <int N2, typename T2>
  inline AffineAccessor<FT,N,T>::AffineAccessor(RegionInstance inst,
						const Matrix<N2, N, T2>& transform,
						const Point<N2, T2>& offset,
						FieldID field_id,
						const Rect<N,T>& subrect,
						size_t subfield_offset /*= 0*/)
  {
    // instance's dimensionality should be <N2,T2>
    const InstanceLayout<N2,T2> *layout = dynamic_cast<const InstanceLayout<N2,T2> *>(inst.get_layout());
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_id);
    assert(it != layout->fields.end());
    const InstancePieceList<N2,T2>& ipl = layout->piece_lists[it->second.list_idx];

    // special case for empty regions
    if(subrect.empty()) {
      base = 0;
      for(int i = 0; i < N; i++) strides[i] = 0;
    } else {
      // if the subrect isn't empty, compute the bounding box of the image
      //  of the subrectangle through the transform - this is a bit ugly
      //  to account for negative elements in the matrix
      Rect<N2,T2> subrect_image(offset, offset);
      for(int i = 0; i < N2; i++)
	for(int j = 0; j < N; j++) {
	  T2 e = transform.rows[i][j];
	  if(e > 0) {
	    subrect_image.lo[i] += e * subrect.lo[j];
	    subrect_image.hi[i] += e * subrect.hi[j];
	  }
	  if(e < 0) {
	    subrect_image.lo[i] += e * subrect.hi[j];
	    subrect_image.hi[i] += e * subrect.lo[j];
	  }
	}
    
      // find the piece that holds the lo corner of the subrect and insist it
      //  exists, covers the whole subrect, and is affine
      const InstanceLayoutPiece<N2,T2> *ilp = ipl.find_piece(subrect_image.lo);
      assert(ilp && ilp->bounds.contains(subrect_image));
      assert((ilp->layout_type == InstanceLayoutPiece<N2,T2>::AffineLayoutType));
      const AffineLayoutPiece<N2,T2> *alp = static_cast<const AffineLayoutPiece<N2,T2> *>(ilp);
      base = reinterpret_cast<intptr_t>(inst.pointer_untyped(0,
							     layout->bytes_used));
      assert(base != 0);
      base += alp->offset + it->second.rel_offset + subfield_offset;
      // to get the effect of transforming every accessed x to Ax+b, we
      //  add strides.b to the offset and left-multiply s'*A to get strides
      //  that go directly from the view's space to the element's offset
      //  in the instance
      base += alp->strides.dot(offset);
      for(int i = 0; i < N; i++) {
	strides[i] = 0;
	for(int j = 0; j < N2; j++)
	  strides[i] += alp->strides[j] * transform.rows[j][i];
      }
    }
#ifdef REALM_ACCESSOR_DEBUG
    dbg_inst = inst;
    dbg_bounds = subrect;  // stay inside the subrect we were given
#endif
  }

  template <typename FT, int N, typename T> __CUDA_HD__
  inline AffineAccessor<FT,N,T>::~AffineAccessor(void)
  {}

  template <typename FT, int N, typename T>
  inline /*static*/ bool AffineAccessor<FT,N,T>::is_compatible(RegionInstance inst, FieldID field_id)
  {
    const InstanceLayout<N,T> *layout = dynamic_cast<const InstanceLayout<N,T> *>(inst.get_layout());
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_id);
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
  inline /*static*/ bool AffineAccessor<FT,N,T>::is_compatible(RegionInstance inst, FieldID field_id, const Rect<N,T>& subrect)
  {
    const InstanceLayout<N,T> *layout = dynamic_cast<const InstanceLayout<N,T> *>(inst.get_layout());
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_id);
    if(it == layout->fields.end())
      return false;
    const InstancePieceList<N,T>& ipl = layout->piece_lists[it->second.list_idx];

    // as long as we had the right field, we're always compatible with an
    //  empty subrect
    if(subrect.empty())
      return true;
    
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
  template <int N2, typename T2>
  inline /*static*/ bool AffineAccessor<FT,N,T>::is_compatible(RegionInstance inst,
							       const Matrix<N2, N, T2>& transform,
							       const Point<N2, T2>& offset,
							       FieldID field_id)
  {
    // instance's dimensionality should be <N2,T2>
    const InstanceLayout<N2,T2> *layout = dynamic_cast<const InstanceLayout<N2,T2> *>(inst.get_layout());
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_id);
    if(it == layout->fields.end())
      return false;
    const InstancePieceList<N2,T2>& ipl = layout->piece_lists[it->second.list_idx];
    
    // this constructor only works if there's exactly one piece and it's affine
    if(ipl.pieces.size() != 1)
      return false;
    const InstanceLayoutPiece<N2,T2> *ilp = ipl.pieces[0];
    if(ilp->layout_type != InstanceLayoutPiece<N2,T2>::AffineLayoutType)
      return false;
    void *base = inst.pointer_untyped(0, layout->bytes_used);
    if(base == 0)
      return false;

    // all checks passed!
    return true;
  }

  template <typename FT, int N, typename T>
  template <int N2, typename T2>
  inline /*static*/ bool AffineAccessor<FT,N,T>::is_compatible(RegionInstance inst,
							       const Matrix<N2, N, T2>& transform,
							       const Point<N2, T2>& offset,
							       FieldID field_id, const Rect<N,T>& subrect)
  {
    // instance's dimensionality should be <N2,T2>
    const InstanceLayout<N2,T2> *layout = dynamic_cast<const InstanceLayout<N2,T2> *>(inst.get_layout());
    std::map<FieldID, InstanceLayoutGeneric::FieldLayout>::const_iterator it = layout->fields.find(field_id);
    if(it == layout->fields.end())
      return false;
    const InstancePieceList<N2,T2>& ipl = layout->piece_lists[it->second.list_idx];

    // as long as we had the right field, we're always compatible with an
    //  empty subrect
    if(subrect.empty())
      return true;

    // if the subrect isn't empty, compute the bounding box of the image
    //  of the subrectangle through the transform - this is a bit ugly
    //  to account for negative elements in the matrix
    Rect<N2,T2> subrect_image(offset, offset);
    for(int i = 0; i < N2; i++)
      for(int j = 0; j < N; j++) {
	T2 e = transform.rows[i][j];
	if(e > 0) {
	  subrect_image.lo[i] += e * subrect.lo[j];
	  subrect_image.hi[i] += e * subrect.hi[j];
	}
	if(e < 0) {
	  subrect_image.lo[i] += e * subrect.hi[j];
	  subrect_image.hi[i] += e * subrect.lo[j];
	}
      }
    
    // find the piece that holds the lo corner of the subrect and insist it
    //  exists, covers the whole subrect, and is affine
    const InstanceLayoutPiece<N2,T2> *ilp = ipl.find_piece(subrect_image.lo);
    if(!(ilp && ilp->bounds.contains(subrect_image)))
      return false;
    if(ilp->layout_type != InstanceLayoutPiece<N2,T2>::AffineLayoutType)
      return false;
    void *base = inst.pointer_untyped(0, layout->bytes_used);
    if(base == 0)
      return false;

    // all checks passed!
    return true;
  }

  template <typename FT, int N, typename T>
  inline FT *AffineAccessor<FT,N,T>::ptr(const Point<N,T>& p) const
  {
    return this->get_ptr(p);
  }

  template <typename FT, int N, typename T> __CUDA_HD__
  inline FT AffineAccessor<FT,N,T>::read(const Point<N,T>& p) const
  {
    return *(this->get_ptr(p));
  }

  template <typename FT, int N, typename T> __CUDA_HD__
  inline void AffineAccessor<FT,N,T>::write(const Point<N,T>& p, FT newval) const
  {
    *(this->get_ptr(p)) = newval;
  }

  template <typename FT, int N, typename T> __CUDA_HD__
  inline FT& AffineAccessor<FT,N,T>::operator[](const Point<N,T>& p) const
  {
    return *(this->get_ptr(p));
  }

  template <typename FT, int N, typename T> __CUDA_HD__
  inline bool AffineAccessor<FT,N,T>::is_dense_arbitrary(const Rect<N,T> &bounds) const
  {
    ptrdiff_t exp_offset = sizeof(FT);
    int used_mask = 0; // keep track of which dimensions we've already matched
    for (int i = 0; i < N; i++) {
      bool found = false;
      for (int j = 0; j < N; j++) {
        if ((used_mask >> j) & 1) continue;
        if (strides[j] != exp_offset) continue;
        found = true;
        used_mask |= (1 << j);
        exp_offset *= (bounds.hi[j] - bounds.lo[j] + 1);
        break;
      }
      if (!found)
        return false;
    }
    return true;
  }

  template <typename FT, int N, typename T> __CUDA_HD__
  inline bool AffineAccessor<FT,N,T>::is_dense_col_major(const Rect<N,T> &bounds) const
  {
    ptrdiff_t exp_offset = sizeof(FT);
    for (int i = 0; i < N; i++) {
      if (strides[i] != exp_offset) return false;
      exp_offset *= (bounds.hi[i] - bounds.lo[i] + 1);
    }
    return true;
  }

  template <typename FT, int N, typename T> __CUDA_HD__
  inline bool AffineAccessor<FT,N,T>::is_dense_row_major(const Rect<N,T> &bounds) const
  {
    ptrdiff_t exp_offset = sizeof(FT);
    for (int i = N-1; i >= 0; i--) {
      if (strides[i] != exp_offset) return false;
      exp_offset *= (bounds.hi[i] - bounds.lo[i] + 1);
    }
    return true;
  }

  template <typename FT, int N, typename T> __CUDA_HD__
  inline FT *AffineAccessor<FT,N,T>::get_ptr(const Point<N,T>& p) const
  {
    intptr_t rawptr = base;
    for(int i = 0; i < N; i++) rawptr += p[i] * strides[i];
    return reinterpret_cast<FT *>(rawptr);
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
