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

#ifndef __LEGION_TEMPLATE_HELP_H__
#define __LEGION_TEMPLATE_HELP_H__

/**
 * \file legion_template_help.h
 */

#include "legion_config.h"

namespace Legion {

  /**
   * \struct LegionStaticAssert
   * Help with static assertions.
   */
  template<bool> struct LegionStaticAssert;
  template<> struct LegionStaticAssert<true> { };
#define LEGION_STATIC_ASSERT(condition) \
  do { LegionStaticAssert<(condition)>(); } while (0)

  /**
   * \struct LegionTypeEquality
   * Help with checking equality of types.
   */
  template<typename T, typename U>
  struct LegionTypeInequality {
  public:
    static const bool value = true;
  };
  template<typename T>
  struct LegionTypeInequality<T,T> {
  public:
    static const bool value = false;
  };

}; // namespace Legion

#endif // __LEGION_TEMPLATE_HELP_H__

