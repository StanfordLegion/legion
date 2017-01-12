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

// instances for Realm

// nop, but helps IDEs
#include "instance.h"

#include "serialize.h"
TYPE_IS_SERIALIZABLE(Realm::RegionInstance);

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class RegionInstance

  inline bool RegionInstance::operator<(const RegionInstance& rhs) const
  {
    return id < rhs.id;
  }

  inline bool RegionInstance::operator==(const RegionInstance& rhs) const
  {
    return id == rhs.id;
  }

  inline bool RegionInstance::operator!=(const RegionInstance& rhs) const
  {
    return id != rhs.id;
  }

  inline bool RegionInstance::exists(void) const
  {
    return id != 0;
  }

  inline std::ostream& operator<<(std::ostream& os, RegionInstance r)
  {
    return os << std::hex << r.id << std::dec;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RegionInstance::DestroyedField


  inline RegionInstance::DestroyedField::DestroyedField(void) 
    : offset(0), size(0), serdez_id(0)
  { }

  inline RegionInstance::DestroyedField::DestroyedField(unsigned o, unsigned s, CustomSerdezID sid)
    : offset(o), size(s), serdez_id(sid)
  { }


}; // namespace Realm  
