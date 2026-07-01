/* Copyright 2026 Stanford University, NVIDIA Corporation
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

// Included from sync.h - do not include this directly

// Useful for IDEs
#include "legion/api/sync.h"

namespace Legion {

  //--------------------------------------------------------------------------
  inline std::ostream& operator<<(std::ostream& os, const PhaseBarrier& pb)
  //--------------------------------------------------------------------------
  {
    os << "PhaseBarrier(" << pb.phase_barrier << ")";
    return os;
  }

}  // namespace Legion
