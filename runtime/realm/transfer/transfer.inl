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

// data transfer (a.k.a. dma) engine for Realm

// nop, but useful for IDEs
#include "realm/transfer/transfer.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferIterator
  //

  template <typename S>
  bool serialize(S& serializer, const TransferIterator& ti)
  {
    return Serialization::PolymorphicSerdezHelper<TransferIterator>::serialize(serializer, ti);
  }

  template <typename S>
  /*static*/ TransferIterator *TransferIterator::deserialize_new(S& deserializer)
  {
    return Serialization::PolymorphicSerdezHelper<TransferIterator>::deserialize_new(deserializer);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class TransferDomain
  //

  inline std::ostream& operator<<(std::ostream& os, const TransferDomain& td)
  {
    td.print(os);
    return os;
  }

  template <typename S>
  bool serialize(S& serializer, const TransferDomain& ci)
  {
    return Serialization::PolymorphicSerdezHelper<TransferDomain>::serialize(serializer, ci);
  }

  template <typename S>
  /*static*/ TransferDomain *TransferDomain::deserialize_new(S& deserializer)
  {
    return Serialization::PolymorphicSerdezHelper<TransferDomain>::deserialize_new(deserializer);
  }


}; // namespace Realm
