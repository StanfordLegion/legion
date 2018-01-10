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

// HDF5-specific instance layouts and accessors

#ifndef REALM_HDF5_ACCESS_H
#define REALM_HDF5_ACCESS_H

#include "realm/inst_layout.h"
#include <string>

#include <hdf5.h>

namespace Realm {

  template <int N, typename T>
  class HDF5LayoutPiece : public InstanceLayoutPiece<N,T> {
  public:
    HDF5LayoutPiece(void);

    template <typename S>
    static InstanceLayoutPiece<N,T> *deserialize_new(S& deserializer);

    virtual size_t calculate_offset(const Point<N,T>& p) const;

    virtual void relocate(size_t base_offset);

    virtual void print(std::ostream& os) const;

    static Serialization::PolymorphicSerdezSubclass<InstanceLayoutPiece<N,T>, HDF5LayoutPiece<N,T> > serdez_subclass;

    template <typename S>
    bool serialize(S& serializer) const;

    std::string filename, dsetname;
    Point<N, hsize_t> offset;
  };

}; // namespace Realm

#include "realm/hdf5/hdf5_access.inl"

#endif // ifndef REALM_HDF5_ACCESS_H
