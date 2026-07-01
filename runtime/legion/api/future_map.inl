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

// Included from future_map.h - do not include this directly

// Useful for IDEs
#include "legion/api/future_map.h"

namespace Legion {

  //--------------------------------------------------------------------------
  template<typename T>
  inline T FutureMap::get_result(
      const DomainPoint& dp, bool silence_warnings,
      const char* warning_string) const
  //--------------------------------------------------------------------------
  {
    Future f = get_future(dp);
    return f.get_result<T>(silence_warnings, warning_string);
  }

  //--------------------------------------------------------------------------
  template<typename RT, typename PT, unsigned DIM>
  inline RT FutureMap::get_result(const PT point[DIM]) const
  //--------------------------------------------------------------------------
  {
    static_assert(
        DIM <= DomainPoint::MAX_POINT_DIM,
        "FutureMap DIM is larger than LEGION_MAX_DIM");
    DomainPoint dp;
    dp.dim = DIM;
    for (unsigned idx = 0; idx < DIM; idx++) dp.point_data[idx] = point[idx];
    Future f = get_future(dp);
    return f.get_result<RT>();
  }

  //--------------------------------------------------------------------------
  template<typename PT, unsigned DIM>
  inline Future FutureMap::get_future(const PT point[DIM]) const
  //--------------------------------------------------------------------------
  {
    static_assert(
        DIM <= DomainPoint::MAX_POINT_DIM,
        "FutureMap DIM is larger than LEGION_MAX_DIM");
    DomainPoint dp;
    dp.dim = DIM;
    for (unsigned idx = 0; idx < DIM; idx++) dp.point_data[idx] = point[idx];
    return get_future(dp);
  }

  //--------------------------------------------------------------------------
  template<typename PT, unsigned DIM>
  inline void FutureMap::get_void_result(const PT point[DIM]) const
  //--------------------------------------------------------------------------
  {
    static_assert(
        DIM <= DomainPoint::MAX_POINT_DIM,
        "FutureMap DIM is larger than LEGION_MAX_DIM");
    DomainPoint dp;
    dp.dim = DIM;
    for (unsigned idx = 0; idx < DIM; idx++) dp.point_data[idx] = point[idx];
    Future f = get_future(dp);
    return f.get_void_result();
  }

}  // namespace Legion

namespace std {

#define LEGION_DEFINE_HASHABLE(__TYPE_NAME__)                     \
  template<>                                                      \
  struct hash<__TYPE_NAME__> {                                    \
    inline std::size_t operator()(const __TYPE_NAME__& obj) const \
    {                                                             \
      return obj.hash();                                          \
    }                                                             \
  };

  LEGION_DEFINE_HASHABLE(Legion::FutureMap);

#undef LEGION_DEFINE_HASHABLE

}  // namespace std
