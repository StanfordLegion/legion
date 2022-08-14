/* Copyright 2021 Stanford University, NVIDIA Corporation
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

// CUDA-specific instance layouts and accessors

// NOP but useful for IDE's
#include "realm/cuda/cuda_access.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class CudaArrayLayoutPiece<N,T>

  template <int N, typename T>
  inline CudaArrayLayoutPiece<N,T>::CudaArrayLayoutPiece(void)
    : InstanceLayoutPiece<N,T>(PieceLayoutTypes::CudaArrayLayoutType)
  {
    for(int i = 0; i < 3; i++)
      offset[i] = 0;
  }

  template <int N, typename T>
  template <typename S>
  /*static*/ inline InstanceLayoutPiece<N,T> *CudaArrayLayoutPiece<N,T>::deserialize_new(S& s)
  {
    CudaArrayLayoutPiece<N,T> *clp = new CudaArrayLayoutPiece<N,T>;
    if((s >> clp->bounds) &&
       (s >> clp->offset[0]) &&
       (s >> clp->offset[1]) &&
       (s >> clp->offset[2])) {
      return clp;
    } else {
      delete clp;
      return 0;
    }
  }

  template <int N, typename T>
  inline InstanceLayoutPiece<N,T> *CudaArrayLayoutPiece<N,T>::clone(void) const
  {
    CudaArrayLayoutPiece<N,T> *copy = new CudaArrayLayoutPiece<N,T>;
    copy->bounds = this->bounds;
    for(int i = 0; i < 3; i++)
      copy->offset[i] = offset[i];
    return copy;
  }

  template <int N, typename T>
  inline size_t CudaArrayLayoutPiece<N,T>::calculate_offset(const Point<N,T>& p) const
  {
    assert(0);
    return 0;
  }

  template <int N, typename T>
  inline void CudaArrayLayoutPiece<N,T>::relocate(size_t base_offset)
  {
  }

  template <int N, typename T>
  void CudaArrayLayoutPiece<N,T>::print(std::ostream& os) const
  {
    os << this->bounds << "->cudaarray(<" << offset[0];
    if(N > 1) os << ',' << offset[1];
    if(N > 2) os << ',' << offset[2];
    os << ">)";
  }

  template <int N, typename T>
  size_t CudaArrayLayoutPiece<N,T>::lookup_inst_size() const
  {
    return sizeof(PieceLookup::CudaArrayPiece<N,T>);
  }

  template <int N, typename T>
  PieceLookup::Instruction *CudaArrayLayoutPiece<N,T>::create_lookup_inst(void *ptr, unsigned next_delta) const
  {
    PieceLookup::CudaArrayPiece<N,T> *cp = new(ptr) PieceLookup::CudaArrayPiece<N,T>(next_delta);
    cp->array = 0;
    cp->offset[0] = 0;
    cp->offset[1] = 0;
    cp->offset[2] = 0;
    // TODO: surface/texture objects
    return cp;
  }

  template <int N, typename T>
  template <typename S>
  inline bool CudaArrayLayoutPiece<N,T>::serialize(S& s) const
  {
    return ((s << this->bounds) &&
            (s << this->offset[0]) &&
            (s << this->offset[1]) &&
            (s << this->offset[2]));
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class PieceLookup::CudaArrayPiece<N,T>

  namespace PieceLookup {

    template <int N, typename T>
    CudaArrayPiece<N,T>::CudaArrayPiece(unsigned next_delta)
      : Instruction(PieceLookup::Opcodes::OP_CUDA_ARRAY_PIECE + (next_delta << 8))
    {}

    template <int N, typename T>
    REALM_CUDA_HD
    unsigned CudaArrayPiece<N,T>::delta() const
    {
      return (data >> 8);
    }

    template <int N, typename T>
    REALM_CUDA_HD
    const Instruction *CudaArrayPiece<N,T>::next() const
    {
      return this->skip(sizeof(CudaArrayPiece<N,T>));
    }

  }; // namespace PieceLookup


  ////////////////////////////////////////////////////////////////////////
  //
  // class ExternalCudaMemoryResource

  template <typename S>
  bool ExternalCudaMemoryResource::serialize(S& s) const
  {
    return ((s << cuda_device_id) &&
            (s << base) &&
            (s << size_in_bytes) &&
	    (s << read_only));
  }

  template <typename S>
  /*static*/ ExternalInstanceResource *ExternalCudaMemoryResource::deserialize_new(S& s)
  {
    int cuda_device_id;
    uintptr_t base;
    size_t size_in_bytes;
    bool read_only;
    if((s >> cuda_device_id) &&
       (s >> base) &&
       (s >> size_in_bytes) &&
       (s >> read_only))
      return new ExternalCudaMemoryResource(cuda_device_id,
                                            base, size_in_bytes, read_only);
    else
      return 0;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ExternalCudaArrayResource

  template <typename S>
  bool ExternalCudaArrayResource::serialize(S& s) const
  {
    return ((s << cuda_device_id) &&
	    (s << array));
  }

  template <typename S>
  /*static*/ ExternalInstanceResource *ExternalCudaArrayResource::deserialize_new(S& s)
  {
    int cuda_device_id;
    uintptr_t array;
    if((s >> cuda_device_id) &&
       (s >> array))
      return new ExternalCudaArrayResource(cuda_device_id,
                                           reinterpret_cast<CUarray_st *>(array));
    else
      return 0;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class ExternalCudaPinnedHostResource

  template <typename S>
  bool ExternalCudaPinnedHostResource::serialize(S& s) const
  {
    return ((s << base) &&
            (s << size_in_bytes) &&
	    (s << read_only));
  }

  template <typename S>
  /*static*/ ExternalInstanceResource *ExternalCudaPinnedHostResource::deserialize_new(S& s)
  {
    uintptr_t base;
    size_t size_in_bytes;
    bool read_only;
    if((s >> base) &&
       (s >> size_in_bytes) &&
       (s >> read_only))
      return new ExternalCudaPinnedHostResource(base, size_in_bytes, read_only);
    else
      return 0;
  }


}; // namespace Realm
