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

#include "pennant.h"
#include "pennant.lg.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "default_mapper.h"
#include "utilities.h"

typedef intptr_t ctx_t;
typedef intptr_t lg_int_t;

struct state
{
  std::string dir;
  std::map<std::string, std::vector<std::string> > params;
  config conf;
  double start_time, end_time;
  std::vector<double> pxx, pxy;
  std::vector<double> zr, ze, zp;
  std::vector<std::vector<lg_int_t> > mapzp;
};

config::config()
  : alfa(0.5)
  , bcx_0(0.0), bcx_1(0.0), bcy_0(0.0), bcy_1(0.0)
  , bcx_n(0), bcy_n(0)
  , cfl(0.6)
  , cflv(0.1)
  , cstop(999999)
  , tstop(1e99)
  , chunksize(99999999)
  , dtfac(1.2)
  , dtinit(1e99)
  , dtmax(1e99)
  , dtreport(10)
  , einit(0.0)
  , einitsub(0.0)
  , gamma(5.0 / 3.0)
  , meshscale(1.0)
  , q1(0.0)
  , q2(2.0)
  , qgamma(5.0 / 3.0)
  , rinit(1.0)
  , rinitsub(1.0)
  , ssmin(0.0)
  , subregion_0(0.0), subregion_1(0.0), subregion_2(0.0), subregion_3(0.0)
  , uinitradial(0.0)
{}


static void extract_param(state *ctx, const char *name, int idx, double &var)
{
  std::string key(name);
  if (ctx->params.count(key)) {
    assert(ctx->params[key].size() > idx);
    var = stod(ctx->params[key][idx]);
  }
}

static void extract_param(state *ctx, const char *name, int idx, intptr_t &var)
{
  std::string key(name);
  if (ctx->params.count(key)) {
    assert(ctx->params[key].size() > idx);
    var = (intptr_t)stoll(ctx->params[key][idx]);
  }
}

ctx_t read_input()
{
  state *ctx = new state;

  ctx->dir = "pennant.tests/sedovsmall/";

  {
    std::string pnt_filename = ctx->dir + "sedovsmall.pnt";
    std::ifstream pnt_file(pnt_filename.c_str(), std::ios::in);

    // Abort if file not valid.
    if (!pnt_file) {
      assert(0 && "input file does not exist");
    }

    // Read all lines.
    while (pnt_file) {
      std::string line;
      getline(pnt_file, line);

      std::stringstream split(line, std::ios::in);

      std::string name;
      split >> name;

      // Skip blank lines.
      if (name.size() == 0) {
        continue;
      }

      // Slurp rest of line.
      std::vector<std::string> data;
      while (split) {
        std::string datum;
        split >> datum;
        if (datum.size() > 0) {
          data.push_back(datum);
        }
      }

      ctx->params[name] = data;
    }
  }

  extract_param(ctx, "cstop", 0, ctx->conf.cstop);
  extract_param(ctx, "tstop", 0, ctx->conf.tstop);
  extract_param(ctx, "meshscale", 0, ctx->conf.meshscale);
  extract_param(ctx, "subregion", 0, ctx->conf.subregion_0);
  extract_param(ctx, "subregion", 1, ctx->conf.subregion_1);
  extract_param(ctx, "subregion", 2, ctx->conf.subregion_2);
  extract_param(ctx, "subregion", 3, ctx->conf.subregion_3);
  extract_param(ctx, "cfl", 0, ctx->conf.cfl);
  extract_param(ctx, "cflv", 0, ctx->conf.cflv);
  extract_param(ctx, "rinit", 0, ctx->conf.rinit);
  extract_param(ctx, "einit", 0, ctx->conf.einit);
  extract_param(ctx, "rinitsub", 0, ctx->conf.rinitsub);
  extract_param(ctx, "einitsub", 0, ctx->conf.einitsub);
  extract_param(ctx, "uinitradial", 0, ctx->conf.uinitradial);
  extract_param(ctx, "bcx", 0, ctx->conf.bcx_0);
  extract_param(ctx, "bcx", 1, ctx->conf.bcx_1);
  extract_param(ctx, "bcy", 0, ctx->conf.bcy_0);
  extract_param(ctx, "bcy", 1, ctx->conf.bcy_1);
  ctx->conf.bcx_n = ctx->params["bcx"].size();
  ctx->conf.bcy_n = ctx->params["bcy"].size();
  extract_param(ctx, "ssmin", 0, ctx->conf.ssmin);
  extract_param(ctx, "q1", 0, ctx->conf.q1);
  extract_param(ctx, "q2", 0, ctx->conf.q2);
  extract_param(ctx, "dtinit", 0, ctx->conf.dtinit);
  extract_param(ctx, "chunksize", 0, ctx->conf.chunksize);

  {
    std::string meshfile("meshfile");
    assert(ctx->params.count(meshfile));
    assert(ctx->params[meshfile].size() == 1);
    std::string gmv_filename = ctx->dir + ctx->params[meshfile][0];
    std::ifstream gmv_file(gmv_filename.c_str(), std::ios::in);

    lg_int_t np = 0;
    lg_int_t nz = 0;
    lg_int_t ns = 0;
    lg_int_t maxznump = 0;

    std::vector<double> &pxx = ctx->pxx, &pxy = ctx->pxy;
    std::vector<std::vector<lg_int_t> > &mapzp = ctx->mapzp;

    // Abort if file not valid.
    if (!gmv_file) {
      assert(0 && "input file does not exist");
    }

    {
      std::string line;
      getline(gmv_file, line);
      assert(line == "gmvinput ascii");
    }

    {
      std::string token;
      gmv_file >> token;
      assert(token == "nodes");
      gmv_file >> np;
    }

    for (int ip = 0; ip < np; ip++) {
      double x;
      gmv_file >> x;
      pxx.push_back(x);
    }

    for (int ip = 0; ip < np; ip++) {
      double y;
      gmv_file >> y;
      pxy.push_back(y);
    }

    for (int ip = 0; ip < np; ip++) {
      double z;
      gmv_file >> z;
      // Throw away z coordinates.
    }

    {
      std::string token;
      gmv_file >> token;
      assert(token == "cells");
      gmv_file >> nz;
    }

    for (int iz = 0; iz < nz; iz++) {
      {
        std::string token;
        gmv_file >> token;
        assert(token == "general");
        double nf;
        gmv_file >> nf;
        assert(nf == 1);
      }

      double znump;
      gmv_file >> znump;

      if (znump > maxznump) {
        maxznump = znump;
      }
      ns += znump;

      std::vector<lg_int_t> points;
      for (int zip = 0; zip < znump; zip++) {
        lg_int_t ip;
        gmv_file >> ip;
        points.push_back(ip - 1);
      }
      mapzp.push_back(points);
    }

    ctx->conf.nz = nz;
    ctx->conf.np = np;
    ctx->conf.ns = ns;
    ctx->conf.maxznump = maxznump;
  }

  ctx->zr.resize(ctx->conf.nz);
  ctx->ze.resize(ctx->conf.nz);
  ctx->zp.resize(ctx->conf.nz);

  return (ctx_t)ctx;
}

void write_output(ctx_t raw_ctx)
{
  state *ctx = (state *)raw_ctx;
  printf("#%4s\n", "zr");
  for (int iz = 0; iz < ctx->conf.nz; iz++) {
    printf("%5d%18.8e\n", iz + 1, ctx->zr[iz]);
  }
  printf("#  ze\n");
  for (int iz = 0; iz < ctx->conf.nz; iz++) {
    printf("%5d%18.8e\n", iz + 1, ctx->ze[iz]);
  }
  printf("#  zp\n");
  for (int iz = 0; iz < ctx->conf.nz; iz++) {
    printf("%5d%18.8e\n", iz + 1, ctx->zp[iz]);
  }
}

void validate_output(ctx_t raw_ctx)
{
  state *ctx = (state *)raw_ctx;

  std::vector<double> sol_zr, sol_ze, sol_zp;

  {
    std::string xy_filename = ctx->dir + "sedovsmall.xy.std";
    std::ifstream xy_file(xy_filename.c_str(), std::ios::in);
    if (!xy_file) {
      assert(0 && "solution file does not exist");
    }

    {
      std::string line;
      getline(xy_file, line);
      assert(line == "#  zr");
    }

    for (int i = 0; i < ctx->conf.nz; i++) {
      int iz;
      double zr;
      xy_file >> iz;
      xy_file >> zr;
      sol_zr.push_back(zr);
    }

    {
      std::string ignore;
      getline(xy_file, ignore);
      std::string line;
      getline(xy_file, line);
      assert(line == "#  ze");
    }

    for (int i = 0; i < ctx->conf.nz; i++) {
      int iz;
      double ze;
      xy_file >> iz;
      xy_file >> ze;
      sol_ze.push_back(ze);
    }

    {
      std::string ignore;
      getline(xy_file, ignore);
      std::string line;
      getline(xy_file, line);
      assert(line == "#  zp");
    }

    for (int i = 0; i < ctx->conf.nz; i++) {
      int iz;
      double zp;
      xy_file >> iz;
      xy_file >> zp;
      sol_zp.push_back(zp);
    }
  }

  double absolute_eps = 1e-12;
  double relative_eps = 1e-8;
  for (int iz = 0; iz < ctx->conf.nz; iz++) {
    double ck = ctx->zr[iz];
    double sol = sol_zr[iz];
    if (fabs(ck - sol) < absolute_eps) {
      continue;
    }
    if (fabs(ck - sol) / sol < relative_eps) {
      continue;
    }
    assert(0 && "zr value out of bounds");
  }

  for (int iz = 0; iz < ctx->conf.nz; iz++) {
    double ck = ctx->ze[iz];
    double sol = sol_ze[iz];
    if (fabs(ck - sol) < absolute_eps) {
      continue;
    }
    if (fabs(ck - sol) / sol < relative_eps) {
      continue;
    }
    assert(0 && "ze value out of bounds");
  }

  for (int iz = 0; iz < ctx->conf.nz; iz++) {
    double ck = ctx->zp[iz];
    double sol = sol_zp[iz];
    if (fabs(ck - sol) < absolute_eps) {
      continue;
    }
    if (fabs(ck - sol) / sol < relative_eps) {
      continue;
    }
    assert(0 && "zp value out of bounds");
  }
}

void start_timer(intptr_t raw_ctx)
{
  state *ctx = (state *)raw_ctx;
  LegionRuntime::LowLevel::DetailedTimer::clear_timers();
  ctx->start_time = LegionRuntime::LowLevel::Clock::abs_time();
}

void stop_timer(intptr_t raw_ctx)
{
  state *ctx = (state *)raw_ctx;
  ctx->end_time = LegionRuntime::LowLevel::Clock::abs_time();
}

void print_timer(intptr_t raw_ctx)
{
  state *ctx = (state *)raw_ctx;
  double delta_time = ctx->end_time - ctx->start_time;
  printf("\n**************************************\n");
  printf("total problem run time=%15.6e\n", delta_time);
  printf("**************************************\n\n");
  LegionRuntime::LowLevel::DetailedTimer::report_timers();
}

config get_config(ctx_t raw_ctx)
{
  state *ctx = (state *)raw_ctx;
  return ctx->conf;
}

lg_int_t get_zone_znump(ctx_t raw_ctx, lg_int_t z)
{
  state *ctx = (state *)raw_ctx;
  return ctx->mapzp[z].size();
}

lg_int_t get_zone_mapzp(ctx_t raw_ctx, lg_int_t z, lg_int_t p)
{
  state *ctx = (state *)raw_ctx;
  return ctx->mapzp[z][p];
}

double get_point_pxx(ctx_t raw_ctx, lg_int_t p)
{
  state *ctx = (state *)raw_ctx;
  return ctx->pxx[p];
}

double get_point_pxy(ctx_t raw_ctx, lg_int_t p)
{
  state *ctx = (state *)raw_ctx;
  return ctx->pxy[p];
}

void put_zone_zr(ctx_t raw_ctx, lg_int_t z, double zr)
{
  state *ctx = (state *)raw_ctx;
  ctx->zr[z] = zr;
}

void put_zone_ze(ctx_t raw_ctx, lg_int_t z, double ze)
{
  state *ctx = (state *)raw_ctx;
  ctx->ze[z] = ze;
}

void put_zone_zp(ctx_t raw_ctx, lg_int_t z, double zp)
{
  state *ctx = (state *)raw_ctx;
  ctx->zp[z] = zp;
}

double length(vec2 a)
{
  return sqrt(dot(a, a));
}

///
/// Mapper
///

class PennantMapper : public DefaultMapper
{
public:
  PennantMapper(Machine *machine, HighLevelRuntime *rt, Processor local);
  virtual Processor select_target_processor(const Task *task);
};

PennantMapper::PennantMapper(Machine *machine, HighLevelRuntime *rt, Processor local)
  : DefaultMapper(machine, rt, local)
{
}

Processor PennantMapper::select_target_processor(const Task *task)
{
  switch (task->task_id) {
  // case TASK_INIT_MESH_ZONES:
  //   {
  //     LogicalRegion param = task->regions[0]->region;
  //     return;
  //   }
  default:
    return DefaultMapper::select_target_processor(task);
  }
}

void create_mappers(Machine *machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    runtime->replace_default_mapper(new PennantMapper(machine, runtime, *it), *it);
  }
}

///
/// Main
///

int main(int argc, char **argv)
{
  HighLevelRuntime::set_registration_callback(create_mappers);
  init_pennant_lg();
  HighLevelRuntime::set_top_level_task_id(TASK_TOPLEVEL);

  return HighLevelRuntime::start(argc, argv);
}
