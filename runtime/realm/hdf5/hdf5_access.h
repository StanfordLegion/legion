/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#ifndef REALM_HDF5_ACCESS_H
#define REALM_HDF5_ACCESS_H

#include "realm/inst_layout.h"
#include <string>
#include <vector>

namespace Realm {

  // we avoid including hdf5.h here, but we need a definition for something that'll
  //  be compatible with HDF5's hsize_t
  typedef unsigned long long hdf5_size_t;

  namespace PieceLayoutTypes {
    static const LayoutType HDF5LayoutType = 2;
  };

  // dimension-agnostic form for piece info allows us to get to it without
  //  having to know the template parameters of the HDF5LayoutPiece holding
  //  it
  struct REALM_PUBLIC_API HDF5PieceInfo {
    std::string dsetname;
    // TODO: small vectors
    // this is the offset within the hdf5 dataset, uses its dimensionality
    std::vector<hdf5_size_t> offset;
    // this maps from realm dimensions to hdf5 dimensions - the
    //  dimensionalities may differ - use '-1' for realm dimensions that
    //  do not correspond to an hdf5 dimension
    std::vector<int> dim_order;
    bool read_only;
  };

  template <int N, typename T>
    class REALM_PUBLIC_API HDF5LayoutPiece : public InstanceLayoutPiece<N,T>, public HDF5PieceInfo {
  public:
    HDF5LayoutPiece(void);

    template <typename S>
    static InstanceLayoutPiece<N,T> *deserialize_new(S& deserializer);

    virtual InstanceLayoutPiece<N,T> *clone(void) const;

    virtual size_t calculate_offset(const Point<N,T>& p) const;

    virtual void relocate(size_t base_offset);

    virtual void print(std::ostream& os) const;

    virtual size_t lookup_inst_size() const;
    virtual PieceLookup::Instruction *create_lookup_inst(void *ptr,
							 unsigned next_delta) const ;

    static Serialization::PolymorphicSerdezSubclass<InstanceLayoutPiece<N,T>, HDF5LayoutPiece<N,T> > serdez_subclass;

    template <typename S>
    bool serialize(S& serializer) const;
  };


  namespace PieceLookup {

    namespace Opcodes {
      static const Opcode OP_HDF5_PIECE = 3;  // this is an HDF5Piece<N,T>
    }

    static const unsigned ALLOW_HDF5_PIECE = 1U << Opcodes::OP_HDF5_PIECE;

    template <int N, typename T>
    struct HDF5Piece : public Instruction {
      // data is: { delta[23:0], opcode[7:0] }
      // top 24 bits of data is jump delta
      HDF5Piece(unsigned next_delta);

      unsigned delta() const;

      unsigned short dsetname_len;
      const char *dsetname() const;

      Rect<N,T> bounds;
      Point<N, hdf5_size_t> offset;
      int dim_order[N];
      bool read_only;

      const Instruction *next() const;
    };

  };


  class REALM_PUBLIC_API ExternalHDF5Resource : public ExternalInstanceResource {
  public:
    ExternalHDF5Resource(const std::string& _filename, bool _read_only);

    // returns the suggested memory in which this resource should be created
    Memory suggested_memory() const;

    virtual ExternalInstanceResource *clone(void) const;

    template <typename S>
    bool serialize(S& serializer) const;

    template <typename S>
    static ExternalInstanceResource *deserialize_new(S& deserializer);

  protected:
    ExternalHDF5Resource();

    static Serialization::PolymorphicSerdezSubclass<ExternalInstanceResource, ExternalHDF5Resource> serdez_subclass;

    virtual void print(std::ostream& os) const;

  public:
    std::string filename;
    bool read_only;
  };

}; // namespace Realm

#include "realm/hdf5/hdf5_access.inl"

#endif // ifndef REALM_HDF5_ACCESS_H
