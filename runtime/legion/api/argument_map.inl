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

// Included from argument_map.h - do not include this directly

// Useful for IDEs
#include "legion/api/argument_map.h"

namespace Legion {

  //--------------------------------------------------------------------------
  template<typename PT, unsigned DIM>
  inline void ArgumentMap::set_point_arg(
      const PT point[DIM], const UntypedBuffer& arg, bool replace /*= false*/)
  //--------------------------------------------------------------------------
  {
    static_assert(
        DIM <= DomainPoint::MAX_POINT_DIM,
        "ArgumentMap DIM is larger than LEGION_MAX_DIM");
    DomainPoint dp;
    dp.dim = DIM;
    for (unsigned idx = 0; idx < DIM; idx++) dp.point_data[idx] = point[idx];
    set_point(dp, arg, replace);
  }

  //--------------------------------------------------------------------------
  template<typename PT, unsigned DIM>
  inline bool ArgumentMap::remove_point(const PT point[DIM])
  //--------------------------------------------------------------------------
  {
    static_assert(
        DIM <= DomainPoint::MAX_POINT_DIM,
        "ArgumentMap DIM is larger than LEGION_MAX_DIM");
    DomainPoint dp;
    dp.dim = DIM;
    for (unsigned idx = 0; idx < DIM; idx++) dp.point_data[idx] = point[idx];
    return remove_point(dp);
  }

}  // namespace Legion
