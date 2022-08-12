/* Copyright 2022 Stanford University
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

#ifndef DTYPE
#error DTYPE must be defined
#endif

#ifndef RESTRICT
#error RESTRICT must be defined
#endif

#ifndef RADIUS
#error RADIUS must be defined
#endif

#include "realm.h"

#include "cpu_kernels.h" // for coord_t

typedef Realm::Point<1, coord_t> Point1;
typedef Realm::Point<2, coord_t> Point2;
typedef Realm::Rect<1, coord_t> Rect1;
typedef Realm::Rect<2, coord_t> Rect2;

struct CreateRegionArgs {
public:
  Rect2 bounds;
  Realm::Memory memory;
  Realm::Processor dest_proc;
  // Warning: Pointers live on dest_proc
  Realm::RegionInstance *dest_inst;
};

struct CreateRegionDoneArgs {
public:
  Realm::RegionInstance inst;
  Realm::Processor dest_proc;
  // Warning: Pointers live on dest_proc
  Realm::RegionInstance *dest_inst;
};

struct ShardArgs {
public:
  Realm::RegionInstance xp_inst_in, xm_inst_in, yp_inst_in, ym_inst_in;
  Realm::RegionInstance xp_inst_out, xm_inst_out, yp_inst_out, ym_inst_out;
  Realm::Barrier xp_empty_in, xm_empty_in, yp_empty_in, ym_empty_in;
  Realm::Barrier xp_empty_out, xm_empty_out, yp_empty_out, ym_empty_out;
  Realm::Barrier xp_full_in, xm_full_in, yp_full_in, ym_full_in;
  Realm::Barrier xp_full_out, xm_full_out, yp_full_out, ym_full_out;
  Realm::Barrier sync, first_start, last_start, first_stop, last_stop;
  coord_t tsteps, tprune, init;
  Point2 point;
  Rect2 interior_bounds, exterior_bounds, outer_bounds;
  Realm::Memory sysmem, regmem;
};

struct StencilArgs {
public:
  Realm::RegionInstance private_inst, xp_inst, xm_inst, yp_inst, ym_inst;
  Rect2 interior_bounds;
  DTYPE *weights;
};

struct IncrementArgs {
public:
  Realm::RegionInstance private_inst, xp_inst, xm_inst, yp_inst, ym_inst;
  Rect2 outer_bounds;
};

struct CheckArgs {
public:
  Realm::RegionInstance private_inst;
  coord_t tsteps, init;
  Rect2 interior_bounds;
};

#endif
