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

#ifndef REALM_CUDA_ACCESS_H
#define REALM_CUDA_ACCESS_H

#include "realm/inst_layout.h"

// CUDA driver/runtime opaque structs for arrays (convertable with explicit cast)
struct CUarray_st;
struct cudaArray;

namespace Realm {

  namespace PieceLayoutTypes {
    static const LayoutType CudaArrayLayoutType = 3;
  };

  // CUDA arrays are fundamentally limited to 3D
  struct CudaArrayPieceInfo {
    int offset[3];
  };

  template <int N, typename T>
  class REALM_PUBLIC_API CudaArrayLayoutPiece :
    public InstanceLayoutPiece<N,T>, public CudaArrayPieceInfo {
  public:
    CudaArrayLayoutPiece(void);

    template <typename S>
    static InstanceLayoutPiece<N,T> *deserialize_new(S& deserializer);

    virtual InstanceLayoutPiece<N,T> *clone(void) const;

    virtual size_t calculate_offset(const Point<N,T>& p) const;

    virtual void relocate(size_t base_offset);

    virtual void print(std::ostream& os) const;

    virtual size_t lookup_inst_size() const;
    virtual PieceLookup::Instruction *create_lookup_inst(void *ptr,
							 unsigned next_delta) const ;

    static Serialization::PolymorphicSerdezSubclass<InstanceLayoutPiece<N,T>, CudaArrayLayoutPiece<N,T> > serdez_subclass;

    template <typename S>
    bool serialize(S& serializer) const;
  };


  namespace PieceLookup {

    namespace Opcodes {
      static const Opcode OP_CUDA_ARRAY_PIECE = 4;  // this is an CudaArrayPiece<N,T>
    }

    static const unsigned ALLOW_CUDA_ARRAY_PIECE = 1U << Opcodes::OP_CUDA_ARRAY_PIECE;

    template <int N, typename T>
    struct CudaArrayPiece : public Instruction {
      // data is: { delta[23:0], opcode[7:0] }
      // top 24 bits of data is jump delta
      CudaArrayPiece(unsigned next_delta);

      unsigned delta() const;

      CUarray_st *array;
      int offset[3];

      const Instruction *next() const;
    };

  };

  class REALM_PUBLIC_API ExternalCudaMemoryResource : public ExternalInstanceResource {
  public:
    ExternalCudaMemoryResource(int _cuda_device_id, uintptr_t _base,
                               size_t _size_in_bytes, bool _read_only);
    ExternalCudaMemoryResource(int _cuda_device_id, void *_base,
                               size_t _size_in_bytes);
    ExternalCudaMemoryResource(int _cuda_device_id, const void *_base,
                               size_t _size_in_bytes);

    // returns the suggested memory in which this resource should be created
    Memory suggested_memory() const;

    virtual ExternalInstanceResource *clone(void) const;

    template <typename S>
    bool serialize(S& serializer) const;

    template <typename S>
    static ExternalInstanceResource *deserialize_new(S& deserializer);

  protected:
    ExternalCudaMemoryResource();

    static Serialization::PolymorphicSerdezSubclass<ExternalInstanceResource, ExternalCudaMemoryResource> serdez_subclass;

    virtual void print(std::ostream& os) const;

  public:
    int cuda_device_id;
    uintptr_t base;
    size_t size_in_bytes;
    bool read_only;
  };

  class REALM_PUBLIC_API ExternalCudaArrayResource : public ExternalInstanceResource {
  public:
    ExternalCudaArrayResource(int _cuda_device_id, CUarray_st *_array);
    ExternalCudaArrayResource(int _cuda_device_id, cudaArray *_array);

    // returns the suggested memory in which this resource should be created
    Memory suggested_memory() const;

    virtual ExternalInstanceResource *clone(void) const;

    template <typename S>
    bool serialize(S& serializer) const;

    template <typename S>
    static ExternalInstanceResource *deserialize_new(S& deserializer);

  protected:
    ExternalCudaArrayResource();

    static Serialization::PolymorphicSerdezSubclass<ExternalInstanceResource, ExternalCudaArrayResource> serdez_subclass;

    virtual void print(std::ostream& os) const;

  public:
    int cuda_device_id;
    uintptr_t array;
  };

  class REALM_PUBLIC_API ExternalCudaPinnedHostResource : public ExternalMemoryResource {
  public:
    ExternalCudaPinnedHostResource(uintptr_t _base, size_t _size_in_bytes, bool _read_only);
    ExternalCudaPinnedHostResource(void *_base, size_t _size_in_bytes);
    ExternalCudaPinnedHostResource(const void *_base, size_t _size_in_bytes);

    // returns the suggested memory in which this resource should be created
    Memory suggested_memory() const;

    virtual ExternalInstanceResource *clone(void) const;

    template <typename S>
    bool serialize(S& serializer) const;

    template <typename S>
    static ExternalInstanceResource *deserialize_new(S& deserializer);

  protected:
    ExternalCudaPinnedHostResource();

    static Serialization::PolymorphicSerdezSubclass<ExternalInstanceResource, ExternalCudaPinnedHostResource> serdez_subclass;

    virtual void print(std::ostream& os) const;
  };

}; // namespace Realm

#include "realm/cuda/cuda_access.inl"

#endif // ifndef REALM_CUDA_ACCESS_H
