/* Copyright 2017 Stanford University
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

#ifndef __REALM_STENCIL__
#define __REALM_STENCIL__

#include "realm.h"

#define RADIUS 2
#define DTYPE double
#define RESTRICT __restrict__

struct ShardArgs {
public:
  Realm::RegionInstance xp_inst, xm_inst, yp_inst, ym_inst;
  size_t tsteps, init;
  Realm::Rect<2> interior_bounds, exterior_bounds;
};

struct StencilArgs {
public:
  Realm::RegionInstance private_inst, xp_inst, xm_inst, yp_inst, ym_inst;
  bool print_ts;
  Realm::Rect<2> interior_bounds;
};

struct IncrementArgs {
public:
  Realm::RegionInstance private_inst, xp_inst, xm_inst, yp_inst, ym_inst;
  bool print_ts;
  Realm::Rect<2> exterior_bounds;
};

struct CheckArgs {
public:
  Realm::RegionInstance private_inst;
  size_t tsteps, init;
  Realm::Rect<2> interior_bounds;
};

#endif
