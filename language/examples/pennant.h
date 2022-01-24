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

#ifndef __PENNANT_H__
#define __PENNANT_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  MULTICOLOR = -1,
};

void generate_mesh_raw(
  int64_t conf_np,
  int64_t conf_nz,
  int64_t conf_nzx,
  int64_t conf_nzy,
  double conf_lenx,
  double conf_leny,
  int64_t conf_numpcx,
  int64_t conf_numpcy,
  int64_t conf_npieces,
  int64_t conf_meshtype,
  bool conf_compact,
  int64_t conf_stripsize,
  int64_t conf_spansize,
  double *pointpos_x, size_t *pointpos_x_size,
  double *pointpos_y, size_t *pointpos_y_size,
  int64_t *pointcolors, size_t *pointcolors_size,
  uint64_t *pointmcolors, size_t *pointmcolors_size,
  int64_t *pointspancolors, size_t *pointspancolors_size,
  int64_t *zonestart, size_t *zonestart_size,
  int64_t *zonesize, size_t *zonesize_size,
  int64_t *zonepoints, size_t *zonepoints_size,
  int64_t *zonecolors, size_t *zonecolors_size,
  int64_t *zonespancolors, size_t *zonespancolors_size,
  int64_t *nspans_zones,
  int64_t *nspans_points);

void print_rusage(const char *);

void register_mappers();

#ifdef __cplusplus
}
#endif

#endif // __PENNANT_H__
