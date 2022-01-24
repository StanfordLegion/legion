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

// NOP, but helpful for IDEs
#include "realm/llvmjit/llvmjit.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class LLVMIRImplementation

  template <typename S>
  bool LLVMIRImplementation::serialize(S& serializer) const
  {
    return ((serializer << ir) &&
	    (serializer << entry_symbol));
  }

  template <typename S>
  /*static*/ CodeImplementation *LLVMIRImplementation::deserialize_new(S& deserializer)
  {
    LLVMIRImplementation *i = new LLVMIRImplementation;
    if((deserializer >> i->ir) &&
       (deserializer >> i->entry_symbol)) {
      return i;
    } else {
      delete i;
      return 0;
    }
  }

}; // namespace Realm
