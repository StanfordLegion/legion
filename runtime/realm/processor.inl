/* Copyright 2024 Stanford University, NVIDIA Corporation
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

// processors for Realm

// nop, but helps IDEs
#include "realm/processor.h"

#include "realm/serialize.h"
TYPE_IS_SERIALIZABLE(Realm::Processor);
TYPE_IS_SERIALIZABLE(Realm::Processor::Kind);

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class Processor

inline void
Processor::get_group_members(std::vector<Processor> &member_list) const {
  size_t num_members = 0;
  get_group_members(nullptr, num_members);
  member_list.resize(num_members);
  get_group_members(member_list.data(), num_members);
}

inline std::ostream &operator<<(std::ostream &os, Realm::Processor p) {
  return os << std::hex << p.id << std::dec;
}

/*static*/ inline Processor
Processor::create_group(const span<const Processor> &members) {
  return ProcessorGroup::create_group(members);
}

#if defined(REALM_USE_KOKKOS)
  // Kokkos execution policies will accept an "execution instance" to
  //  capture task parallelism - provide those here
  inline Processor::KokkosExecInstance::KokkosExecInstance(Processor _p)
    : p (_p)
  {}

  inline Processor::KokkosExecInstance Processor::kokkos_work_space(void) const
  {
    return KokkosExecInstance(*this);
  }
#endif

}; // namespace Realm  
