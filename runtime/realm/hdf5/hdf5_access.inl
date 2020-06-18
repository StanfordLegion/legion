/* Copyright 2020 Stanford University, NVIDIA Corporation
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

// HDF5-specific instance layouts and accessors

// NOP but useful for IDE's
#include "realm/hdf5/hdf5_access.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class HDF5LayoutPiece<N,T>

  template <int N, typename T>
  inline HDF5LayoutPiece<N,T>::HDF5LayoutPiece(void)
    : InstanceLayoutPiece<N,T>(InstanceLayoutPiece<N,T>::HDF5LayoutType)
  {}

  template <int N, typename T>
  template <typename S>
  /*static*/ inline InstanceLayoutPiece<N,T> *HDF5LayoutPiece<N,T>::deserialize_new(S& s)
  {
    HDF5LayoutPiece<N,T> *hlp = new HDF5LayoutPiece<N,T>;
    if((s >> hlp->bounds) &&
       (s >> hlp->filename) &&
       (s >> hlp->dsetname) &&
       (s >> hlp->offset) &&
       (s >> hlp->dim_order) &&
       (s >> hlp->read_only)) {
      return hlp;
    } else {
      delete hlp;
      return 0;
    }
  }

  template <int N, typename T>
  inline InstanceLayoutPiece<N,T> *HDF5LayoutPiece<N,T>::clone(void) const
  {
    HDF5LayoutPiece<N,T> *copy = new HDF5LayoutPiece<N,T>;
    copy->bounds = this->bounds;
    copy->filename = this->filename;
    copy->dsetname = this->dsetname;
    copy->offset = this->offset;
    for(int i = 0; i < N; i++)
      copy->dim_order[i] = this->dim_order[i];
    copy->read_only = this->read_only;
    return copy;
  }

  template <int N, typename T>
  inline size_t HDF5LayoutPiece<N,T>::calculate_offset(const Point<N,T>& p) const
  {
    assert(0);
    return 0;
  }

  template <int N, typename T>
  inline void HDF5LayoutPiece<N,T>::relocate(size_t base_offset)
  {
  }

  template <int N, typename T>
  void HDF5LayoutPiece<N,T>::print(std::ostream& os) const
  {
    os << this->bounds << "->hdf5(" << filename << "," << dsetname << "+" << offset << ")";
  }

  template <int N, typename T>
  template <typename S>
  inline bool HDF5LayoutPiece<N,T>::serialize(S& s) const
  {
    return ((s << this->bounds) &&
	    (s << filename) &&
	    (s << dsetname) &&
	    (s << offset) &&
	    (s << dim_order) &&
	    (s << read_only));
  }


  namespace PieceLookup {

    ////////////////////////////////////////////////////////////////////////
    //
    // class PieceLookup::HDF5Piece<N,T>

    template <int N, typename T>
    unsigned HDF5Piece<N,T>::delta() const
    {
      return (data >> 8);
    }

    template <int N, typename T>
    const char *HDF5Piece<N,T>::filename() const
    {
      return (reinterpret_cast<const char *>(this) +
	      sizeof(HDF5Piece<N,T>));
    }

    template <int N, typename T>
    const char *HDF5Piece<N,T>::dsetname() const
    {
      return (reinterpret_cast<const char *>(this) +
	      sizeof(HDF5Piece<N,T>) +
	      this->filename_len + 1);
    }

    template <int N, typename T>
    const Instruction *HDF5Piece<N,T>::next() const
    {
      return this->skip(sizeof(HDF5Piece<N,T>) +
			this->filename_len +
			this->dsetname_len + 2);
    }

  }; // namespace PieceLookup

}; // namespace Realm
