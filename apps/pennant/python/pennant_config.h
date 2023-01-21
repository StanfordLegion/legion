/* Copyright 2023 Stanford University
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

#ifndef __PENNANT_CONFIG_H__
#define __PENNANT_CONFIG_H__

#ifndef LEGION_USE_PYTHON_CFFI
#include <stdbool.h>
#include "legion.h"
#endif // LEGION_USE_PYTHON_CFFI

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mesh_colorings {
  legion_coloring_t rz_all_c;
  legion_coloring_t rz_spans_c;
  legion_coloring_t rp_all_c;
  legion_coloring_t rp_all_private_c;
  legion_coloring_t rp_all_ghost_c;
  legion_coloring_t rp_all_shared_c;
  legion_coloring_t rp_spans_c;
  legion_coloring_t rs_all_c;
  legion_coloring_t rs_spans_c;
  int64_t nspans_zones;
  int64_t nspans_points;
} mesh_colorings;

typedef struct mesh_partitions {
  legion_logical_partition_t rz_all_p;
  legion_logical_partition_t rp_all_p;
  legion_logical_partition_t rp_all_private_p;
  legion_logical_partition_t rp_all_ghost_p;
  legion_logical_partition_t rp_all_shared_p;
  legion_logical_partition_t rs_all_p;
  int64_t nspans_zones;
  int64_t nspans_points;
} mesh_partitions;

typedef struct config {
  double alfa;
  double bcx[2];
  int64_t bcx_n;
  double bcy[2];
  int64_t bcy_n;
  double cfl;
  double cflv;
  int64_t chunksize;
  int64_t cstop;
  double dtfac;
  double dtinit;
  double dtmax;
  double dtreport;
  double einit;
  double einitsub;
  double gamma;
  double meshscale;
  double q1;
  double q2;
  double qgamma;
  double rinit;
  double rinitsub;
  double ssmin;
  double subregion[4];
  int64_t subregion_n;
  double tstop;
  double uinitradial;
  double meshparams[4];
  int64_t meshparams_n;
  int64_t meshtype;
  int64_t nzx;
  int64_t nzy;
  int64_t numpcx;
  int64_t numpcy;
  double lenx;
  double leny;
  int64_t nz;
  int64_t np;
  int64_t ns;
  int64_t maxznump;
  int64_t npieces;
  bool par_init;
  bool seq_init;
  bool print_ts;
  bool enable;
  bool warmup;
  int64_t prune;
  bool compact;
  bool internal;
  bool interior;
  int64_t stripsize;
  int64_t spansize;
  int64_t nspans_zones;
  int64_t nspans_points;
} config;

#ifdef __cplusplus
}
#endif

#endif // __PENNANT_CONFIG_H__
