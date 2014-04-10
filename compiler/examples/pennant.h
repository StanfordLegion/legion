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

#include "legion.h"

using namespace LegionRuntime::HighLevel;

// FIXME: This currently breaks C++ import.
/* typedef intptr_t ctx_t; */
/* typedef intptr_t lg_int_t; */

enum {
  MESH_PIE,
  MESH_RECT,
  MESH_HEX,
};

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

  // Mesh generator variables.
  intptr_t meshtype;
  double meshparams_0, meshparams_1, meshparams_2, meshparams_3;
  intptr_t meshparams_n;
  intptr_t nzx;
  intptr_t nzy;
  intptr_t numpcx;
  intptr_t numpcy;
  double lenx;
  double leny;

  // Mesh variables.
  intptr_t nz;
  intptr_t np;
  intptr_t ns;
  intptr_t maxznump;

  // Command-line parameters.
  intptr_t npieces;
  bool use_foreign;
};

///
/// I/O
///

extern config read_config();
extern void foreign_read_input(HighLevelRuntime *runtime,
                               Context ctx,
                               config conf,
                               PhysicalRegion rz_all[1],
                               PhysicalRegion rp_all[1],
                               PhysicalRegion rs_all[1],
                               PhysicalRegion rm_all[1],
                               PhysicalRegion pcolor_a[1],
                               PhysicalRegion pcolors_a[1],
                               PhysicalRegion pcolor_shared_a[1]);
extern void write_output(HighLevelRuntime *runtime,
                         Context ctx,
                         config conf,
                         PhysicalRegion rz_all[1],
                         PhysicalRegion rp_all[1],
                         PhysicalRegion rs_all[1]);
extern void foreign_validate_output(HighLevelRuntime *runtime,
                                    Context ctx,
                                    config conf,
                                    PhysicalRegion rz_all[1],
                                    PhysicalRegion rp_all[1],
                                    PhysicalRegion rs_all[1]);
extern double get_abs_time();
extern void print_global_elapsed_time(double start_time, double stop_time);
extern void print_simulation_start();
extern void print_simulation_loop(intptr_t cycle, double time, double dt,
                                  double start_time, double last_time,
                                  double current_time, intptr_t interval);

///
/// Coloring
///

extern Coloring foreign_all_zones_coloring(HighLevelRuntime *runtime,
                                           Context ctx,
                                           config conf,
                                           PhysicalRegion rm_all[1]);

extern Coloring foreign_all_points_coloring(HighLevelRuntime *runtime,
                                            Context ctx,
                                            config conf,
                                            PhysicalRegion pcolor_a[1]);

extern Coloring foreign_private_points_coloring(HighLevelRuntime *runtime,
                                                Context ctx,
                                                config conf,
                                                PhysicalRegion pcolor_a[1]);

extern Coloring foreign_ghost_points_coloring(HighLevelRuntime *runtime,
                                              Context ctx,
                                              config conf,
                                              PhysicalRegion pcolor_a[1],
                                              PhysicalRegion pcolors_a[1]);

extern Coloring foreign_shared_points_coloring(HighLevelRuntime *runtime,
                                               Context ctx,
                                               config conf,
                                               PhysicalRegion pcolor_shared_a[1]);

extern Coloring foreign_all_sides_coloring(HighLevelRuntime *runtime,
                                           Context ctx,
                                           config conf,
                                           PhysicalRegion rm_all[1]);

///
/// Kernels
///

extern double calc_global_dt(double dt, double dtfac, double dtinit,
                             double dtmax, double dthydro,
                             double time, double tstop, intptr_t cycle);

extern void foreign_init_step_zones(intptr_t zstart,
                                    intptr_t zend,
                                    PhysicalRegion rz[2]);

extern void foreign_calc_centers(intptr_t sstart,
                                 intptr_t send,
                                 PhysicalRegion rz[2],
                                 PhysicalRegion rpp[1],
                                 PhysicalRegion rpg[1],
                                 PhysicalRegion rs[2]);

extern void foreign_calc_volumes(intptr_t sstart,
                                 intptr_t send,
                                 PhysicalRegion rz[2],
                                 PhysicalRegion rpp[1],
                                 PhysicalRegion rpg[1],
                                 PhysicalRegion rs[2]);

extern void foreign_calc_surface_vecs(intptr_t sstart,
                                      intptr_t send,
                                      PhysicalRegion rz[1],
                                      PhysicalRegion rs[2]);

extern void foreign_calc_edge_len(intptr_t sstart,
                                  intptr_t send,
                                  PhysicalRegion rpp[1],
                                  PhysicalRegion rpg[1],
                                  PhysicalRegion rs[2]);

extern void foreign_calc_char_len(intptr_t sstart,
                                  intptr_t send,
                                  PhysicalRegion rz[1],
                                  PhysicalRegion rs[2]);

extern void foreign_calc_rho_half(intptr_t zstart,
                                  intptr_t zend,
                                  PhysicalRegion rz[2]);

extern void foreign_sum_point_mass(intptr_t sstart,
                                   intptr_t send,
                                   PhysicalRegion rz[1],
                                   PhysicalRegion rpp[1],
                                   PhysicalRegion rpg[1],
                                   PhysicalRegion rs[1]);

extern void foreign_calc_state_at_half(double gamma,
                                       double ssmin,
                                       double dt,
                                       intptr_t zstart,
                                       intptr_t zend,
                                       PhysicalRegion rz[2]);

extern void foreign_calc_force_pgas(intptr_t sstart,
                                    intptr_t send,
                                    PhysicalRegion rz[1],
                                    PhysicalRegion rs[2]);

extern void foreign_calc_force_tts(double afla,
                                   double ssmin,
                                   intptr_t sstart,
                                   intptr_t send,
                                   PhysicalRegion rz[1],
                                   PhysicalRegion rs[2]);

extern void foreign_qcs_zone_center_velocity(intptr_t sstart,
                                             intptr_t send,
                                             PhysicalRegion rz[2],
                                             PhysicalRegion rpp[1],
                                             PhysicalRegion rpg[1],
                                             PhysicalRegion rs[1]);

extern void foreign_qcs_corner_divergence(intptr_t sstart,
                                          intptr_t send,
                                          PhysicalRegion rz[1],
                                          PhysicalRegion rpp[1],
                                          PhysicalRegion rpg[1],
                                          PhysicalRegion rs[2]);

extern void foreign_qcs_qcn_force(double gamma,
                                  double q1,
                                  double q2,
                                  intptr_t sstart,
                                  intptr_t send,
                                  PhysicalRegion rz[1],
                                  PhysicalRegion rpp[1],
                                  PhysicalRegion rpg[1],
                                  PhysicalRegion rs[2]);

extern void foreign_qcs_force(intptr_t sstart,
                              intptr_t send,
                              PhysicalRegion rs[2]);

extern void foreign_sum_point_force(intptr_t sstart,
                                    intptr_t send,
                                    PhysicalRegion rpp[2],
                                    PhysicalRegion rpg[2],
                                    PhysicalRegion rs[1]);

extern void foreign_calc_centers_full(intptr_t sstart,
                                      intptr_t send,
                                      PhysicalRegion rz[2],
                                      PhysicalRegion rpp[1],
                                      PhysicalRegion rpg[1],
                                      PhysicalRegion rs[2]);

extern void foreign_calc_volumes_full(intptr_t sstart,
                                      intptr_t send,
                                      PhysicalRegion rz[2],
                                      PhysicalRegion rpp[1],
                                      PhysicalRegion rpg[1],
                                      PhysicalRegion rs[2]);

extern void foreign_calc_work(double dt,
                              intptr_t sstart,
                              intptr_t send,
                              PhysicalRegion rz[1],
                              PhysicalRegion rpp[1],
                              PhysicalRegion rpg[1],
                              PhysicalRegion rs[1]);

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
