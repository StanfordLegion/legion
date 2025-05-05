/* Copyright 2024 Stanford University
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

#include "realm/prealm/prealm.h"

#include "cpu_kernels.h" // for coord_t

typedef PRealm::Point<1, coord_t> Point1;
typedef PRealm::Point<2, coord_t> Point2;
typedef PRealm::Rect<1, coord_t> Rect1;
typedef PRealm::Rect<2, coord_t> Rect2;

struct CreateRegionArgs {
public:
  Rect2 bounds;
  PRealm::Memory memory;
  PRealm::Processor dest_proc;
  // Warning: Pointers live on dest_proc
  PRealm::RegionInstance *dest_inst;
};

struct CreateRegionDoneArgs {
public:
  PRealm::RegionInstance inst;
  PRealm::Processor dest_proc;
  // Warning: Pointers live on dest_proc
  PRealm::RegionInstance *dest_inst;
};

struct ShardArgs {
public:
  PRealm::RegionInstance xp_inst_in, xm_inst_in, yp_inst_in, ym_inst_in;
  PRealm::RegionInstance xp_inst_out, xm_inst_out, yp_inst_out, ym_inst_out;
  PRealm::Barrier xp_empty_in, xm_empty_in, yp_empty_in, ym_empty_in;
  PRealm::Barrier xp_empty_out, xm_empty_out, yp_empty_out, ym_empty_out;
  PRealm::Barrier xp_full_in, xm_full_in, yp_full_in, ym_full_in;
  PRealm::Barrier xp_full_out, xm_full_out, yp_full_out, ym_full_out;
  PRealm::Barrier sync, first_start, last_start, first_stop, last_stop;
  coord_t tsteps, tprune, init;
  Point2 point;
  Rect2 interior_bounds, exterior_bounds, outer_bounds;
  PRealm::Memory sysmem, regmem;
};

struct StencilArgs {
public:
  PRealm::RegionInstance private_inst, xp_inst, xm_inst, yp_inst, ym_inst;
  Rect2 interior_bounds;
  DTYPE *weights;
};

struct IncrementArgs {
public:
  PRealm::RegionInstance private_inst, xp_inst, xm_inst, yp_inst, ym_inst;
  Rect2 outer_bounds;
};

struct CheckArgs {
public:
  PRealm::RegionInstance private_inst;
  coord_t tsteps, init;
  Rect2 interior_bounds;
};

#endif
