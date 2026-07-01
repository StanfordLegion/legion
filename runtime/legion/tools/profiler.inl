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

// Included from profiler.h - do not include this directly

// Useful for IDEs
#include "legion/tools/profiler.h"

namespace Legion {
  namespace Internal {

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline void LegionProfInstance::record_index_space_point(
        DistributedID handle, const Point<DIM, T>& point)
    //--------------------------------------------------------------------------
    {
      IndexSpacePointDesc ispace_point_desc;
      ispace_point_desc.unique_id = handle;
      ispace_point_desc.dim = (unsigned)DIM;
#define DIMFUNC(D2) \
  ispace_point_desc.points[D2 - 1] = (D2 <= DIM) ? (long long)point[D2 - 1] : 0;
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      register_index_space_point(ispace_point_desc);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    inline void LegionProfInstance::record_index_space_rect(
        DistributedID handle, const Rect<DIM, T>& rect)
    //--------------------------------------------------------------------------
    {
      IndexSpaceRectDesc ispace_rect_desc;
      ispace_rect_desc.unique_id = handle;
      ispace_rect_desc.dim = DIM;
#define DIMFUNC(D2)                                 \
  ispace_rect_desc.rect_lo[D2 - 1] =                \
      (D2 <= DIM) ? (long long)rect.lo[D2 - 1] : 0; \
  ispace_rect_desc.rect_hi[D2 - 1] =                \
      (D2 <= DIM) ? (long long)rect.hi[D2 - 1] : 0;
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      register_index_space_rect(ispace_rect_desc);
    }

  }  // namespace Internal
}  // namespace Legion
