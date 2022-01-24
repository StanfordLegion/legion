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

// helper defines/data structures for fault reporting/handling in Realm

// nop, but helps IDEs
#include "realm/faults.h"

namespace Realm {

    ////////////////////////////////////////////////////////////////////////
    //
    // class Backtrace
    //

    template <typename S>
    bool serdez(S& serdez, const Backtrace& b)
    {
      return ((serdez & b.pc_hash) &&
	      (serdez & b.pcs) &&
	      (serdez & b.symbols));
    }


}; // namespace Realm
