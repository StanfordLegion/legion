/* Copyright 2013 Stanford University and Los Alamos National Security, LLC
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

#ifndef _PENNANT_H
#define _PENNANT_H

#include "stdint.h"

// FIXME: This currently breaks C++ import.
/* typedef intptr_t ctx_t; */
/* typedef intptr_t lg_int_t; */

///
/// Configuration
///
struct config
{
  config();

  // Configuration variables.
  double alfa;
  double bcx_0, bcx_1;
  double bcy_0, bcy_1;
  intptr_t bcx_n, bcy_n;
  double cfl;
  double cflv;
  intptr_t chunksize;
  intptr_t cstop;
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
  double subregion_0, subregion_1, subregion_2, subregion_3;
  double tstop;
  double uinitradial;

  // Mesh variables.
  intptr_t nz;
  intptr_t np;
  intptr_t ns;
  intptr_t maxznump;
};

///
/// I/O
///

extern intptr_t read_input();
extern void write_output(intptr_t ctx);
extern void validate_output(intptr_t ctx);
extern void start_timer(intptr_t ctx);
extern void stop_timer(intptr_t ctx);
extern void print_timer(intptr_t ctx);
extern config get_config(intptr_t ctx);
extern intptr_t get_zone_znump(intptr_t ctx, intptr_t z);
extern intptr_t get_zone_mapzp(intptr_t ctx, intptr_t z, intptr_t p);
extern double get_point_pxx(intptr_t ctx, intptr_t p);
extern double get_point_pxy(intptr_t ctx, intptr_t p);
extern void put_zone_zr(intptr_t raw_ctx, intptr_t z, double zr);
extern void put_zone_ze(intptr_t raw_ctx, intptr_t z, double ze);
extern void put_zone_zp(intptr_t raw_ctx, intptr_t z, double zp);

///
/// Vectors
///

struct vec2
{
  double x;
  double y;
};

inline vec2 add(vec2 a, vec2 b)
{
  vec2 result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  return result;
}

inline vec2 sub(vec2 a, vec2 b)
{
  vec2 result;
  result.x = a.x - b.x;
  result.y = a.y - b.y;
  return result;
}

inline vec2 scale(vec2 a, double b)
{
  vec2 result;
  result.x = a.x * b;
  result.y = a.y * b;
  return result;
}

inline double dot(vec2 a, vec2 b)
{
  return a.x*b.x + a.y*b.y;
}

inline double cross(vec2 a, vec2 b)
{
  return a.x*b.y - a.y*b.x;
}

extern double length(vec2 a);

inline vec2 rotateCCW(vec2 a)
{
  vec2 result;
  result.x = -a.y;
  result.y = a.x;
  return result;
}

inline vec2 project(vec2 a, vec2 b)
{
  return sub(a, scale(b, dot(a, b)));
}

#endif
