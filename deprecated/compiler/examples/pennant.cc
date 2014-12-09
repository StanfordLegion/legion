/* Copyright 2014 Stanford University and Los Alamos National Security, LLC
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
#include "legion_tasks.h"
#include "mapping_utilities.h"
#include "utilities.h"

///
/// Configuration
///

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

///
/// Constants
///

// typedef AccessorType::AOS<0> AT;
typedef AccessorType::SOA<0> AT;

typedef AccessorType::SOA<1> AT_SOA_1;
typedef AccessorType::SOA<4> AT_SOA_4;
typedef AccessorType::SOA<8> AT_SOA_8;

typedef AccessorType::ReductionFold<reduction_plus_double> AT_RED_PLUS;

///
/// Logging
///

LegionRuntime::Logger::Category log_mapper("mapper");

///
/// Command-line defaults
///

const std::string default_input_filename = "pennant.tests/sedovsmall/sedovsmall.pnt";
const intptr_t default_npieces = 1;
const bool default_use_foreign = true;

///
/// Input helpers
///

static std::string get_input_filename()
{
  const InputArgs &args = HighLevelRuntime::get_input_args();

  std::string filename = default_input_filename;
  for (int i = 1; i < args.argc; i++) {
    std::string arg = args.argv[i];

    // Skip options.
    if (arg.find('-') == 0) {
      i++;
      continue;
    }

    // Grab positional parameter.
    filename = arg;
    break;
  }
  return filename;
}

static std::string get_solution_filename()
{
  std::string input_filename = get_input_filename();

  size_t sep = input_filename.find_last_of('.');
  if (sep != std::string::npos) {
    return input_filename.substr(0, sep) + ".xy.std";
  } else {
    return input_filename + ".xy.std";
  }
}

static std::string get_directory(const std::string &filename)
{
  size_t sep = filename.find_last_of('/');
  if (sep != std::string::npos) {
    return filename.substr(0, sep + 1);
  } else {
    return ".";
  }
}

static intptr_t get_npieces()
{
  const InputArgs &args = HighLevelRuntime::get_input_args();

  for (int i = 1; i < args.argc; i++) {
    std::string arg = args.argv[i];

    // Grab npieces parameter.
    if (arg == "-npieces") {
      i++;
      return std::stoll(args.argv[i]);
    }

    // Skip other options.
    if (arg.find('-') == 0) {
      i++;
      continue;
    }
  }
  return default_npieces;
}

static bool get_use_foreign()
{
  const InputArgs &args = HighLevelRuntime::get_input_args();

  for (int i = 1; i < args.argc; i++) {
    std::string arg = args.argv[i];

    // Grab npieces parameter.
    if (arg == "-foreign") {
      i++;
      return bool(std::stoll(args.argv[i]));
    }

    // Skip other options.
    if (arg.find('-') == 0) {
      i++;
      continue;
    }
  }
  return default_use_foreign;
}

static void extract_param(const std::map<std::string,
                                         std::vector<std::string> > &params,
                          const std::string &key, int idx, std::string &var,
                          bool required)
{
  if (params.count(key)) {
    assert(params.at(key).size() > idx);
    var = params.at(key)[idx];
  } else {
    assert(!required);
  }
}

static void extract_param(const std::map<std::string,
                                         std::vector<std::string> > &params,
                          const std::string &key, int idx, double &var,
                          bool required)
{
  if (params.count(key)) {
    assert(params.at(key).size() > idx);
    var = stod(params.at(key)[idx]);
  } else {
    assert(!required);
  }
}

static void extract_param(const std::map<std::string,
                                         std::vector<std::string> > params,
                          const std::string &key, int idx, intptr_t &var,
                          bool required)
{
  if (params.count(key)) {
    assert(params.at(key).size() > idx);
    var = (intptr_t)stoll(params.at(key)[idx]);
  } else {
    assert(!required);
  }
}

static void read_params(const std::string &pnt_filename,
                        std::map<std::string, std::vector<std::string> > &params)
{
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

    params[name] = data;
  }
}

///
/// Mesh Generator
///

static void init_mesh(config &conf)
{
  conf.nzx = conf.meshparams_0;
  conf.nzy = (conf.meshparams_n >= 2 ? conf.meshparams_1 : conf.nzx);
  if (conf.meshtype != MESH_PIE) {
    conf.lenx = (conf.meshparams_n >= 3 ? conf.meshparams_2 : 1.0);
  } else {
    // convention:  x = theta, y = r
    conf.lenx = (conf.meshparams_n >= 3 ? conf.meshparams_2 : 90.0)
      * M_PI / 180.0;
  }
  conf.leny = (conf.meshparams_n >= 4 ? conf.meshparams_3 : 1.0);

  if (conf.nzx <= 0 || conf.nzy <= 0 || conf.lenx <= 0. || conf.leny <= 0. ) {
    assert(0 && "meshparams values must be positive");
  }
  if (conf.meshtype == MESH_PIE && conf.lenx >= 2. * M_PI) {
    assert(0 && "meshparams theta must be < 360");
  }

  // Calculate approximate nz, np, ns for region size upper bound.
  conf.nz = conf.nzx * conf.nzy;
  conf.np = (conf.nzx + 1) * (conf.nzy + 1);
  if (conf.meshtype != MESH_HEX) {
    conf.maxznump = 4;
  } else {
    conf.maxznump = 6;
  }
  conf.ns = conf.nz * conf.maxznump;
}

const intptr_t MULTICOLOR = -1;

static void generate_mesh_rect(config &conf,
                               std::vector<double> &pointpos_x,
                               std::vector<double> &pointpos_y,
                               std::vector<intptr_t> &pointcolors,
                               std::map<intptr_t, std::vector<intptr_t> > &pointmcolors,
                               std::vector<intptr_t> &zonestart,
                               std::vector<intptr_t> &zonesize,
                               std::vector<intptr_t> &zonepoints,
                               std::vector<intptr_t> &zonecolors,
                               std::vector<intptr_t> &zxbounds,
                               std::vector<intptr_t> &zybounds)
{
  intptr_t &nz = conf.nz;
  intptr_t &np = conf.np;

  nz = conf.nzx * conf.nzy;
  const int npx = conf.nzx + 1;
  const int npy = conf.nzy + 1;
  np = npx * npy;

  // generate point coordinates
  pointpos_x.reserve(np);
  pointpos_y.reserve(np);
  double dx = conf.lenx / (double) conf.nzx;
  double dy = conf.leny / (double) conf.nzy;
  int pcy = 0;
  for (int j = 0; j < npy; ++j) {
    if (j >= zybounds[pcy+1]) pcy += 1;
    double y = dy * (double) j;
    int pcx = 0;
    for (int i = 0; i < npx; ++i) {
      if (i >= zxbounds[pcx+1]) pcx += 1;
      double x = dx * (double) i;
      pointpos_x.push_back(x);
      pointpos_y.push_back(y);
      int c = pcy * conf.numpcx + pcx;
      if (i != zxbounds[pcx] && j != zybounds[pcy])
        pointcolors.push_back(c);
      else {
        int p = pointpos_x.size() - 1;
        pointcolors.push_back(MULTICOLOR);
        std::vector<intptr_t> &pmc = pointmcolors[p];
        if (i == zxbounds[pcx] && j == zybounds[pcy])
          pmc.push_back(c - conf.numpcx - 1);
        if (j == zybounds[pcy]) pmc.push_back(c - conf.numpcx);
        if (i == zxbounds[pcx]) pmc.push_back(c - 1);
        pmc.push_back(c);
      }
    }
  }

  // generate zone adjacency lists
  zonestart.reserve(nz);
  zonesize.reserve(nz);
  zonepoints.reserve(4 * nz);
  zonecolors.reserve(nz);
  pcy = 0;
  for (int j = 0; j < conf.nzy; ++j) {
    if (j >= zybounds[pcy+1]) pcy += 1;
    int pcx = 0;
    for (int i = 0; i < conf.nzx; ++i) {
      if (i >= zxbounds[pcx+1]) pcx += 1;
      zonestart.push_back(zonepoints.size());
      zonesize.push_back(4);
      int p0 = j * npx + i;
      zonepoints.push_back(p0);
      zonepoints.push_back(p0 + 1);
      zonepoints.push_back(p0 + npx + 1);
      zonepoints.push_back(p0 + npx);
      zonecolors.push_back(pcy * conf.numpcx + pcx);
    }
  }
}

static void generate_mesh_pie(config &conf,
                              std::vector<double> &pointpos_x,
                              std::vector<double> &pointpos_y,
                              std::vector<intptr_t> &pointcolors,
                              std::map<intptr_t, std::vector<intptr_t> > &pointmcolors,
                              std::vector<intptr_t> &zonestart,
                              std::vector<intptr_t> &zonesize,
                              std::vector<intptr_t> &zonepoints,
                              std::vector<intptr_t> &zonecolors,
                              std::vector<intptr_t> &zxbounds,
                              std::vector<intptr_t> &zybounds)
{
  intptr_t &nz = conf.nz;
  intptr_t &np = conf.np;

  nz = conf.nzx * conf.nzy;
  const int npx = conf.nzx + 1;
  const int npy = conf.nzy + 1;
  np = npx * (npy - 1) + 1;

  // generate point coordinates
  pointpos_x.reserve(np);
  pointpos_y.reserve(np);
  double dth = conf.lenx / (double) conf.nzx;
  double dr  = conf.leny / (double) conf.nzy;
  int pcy = 0;
  for (int j = 0; j < npy; ++j) {
    if (j >= zybounds[pcy+1]) pcy += 1;
    if (j == 0) {
      // special case:  "row" at origin only contains
      // one point, shared by all pieces in row
      pointpos_x.push_back(0.);
      pointpos_y.push_back(0.);
      if (conf.numpcx == 1)
        pointcolors.push_back(0);
      else {
        pointcolors.push_back(MULTICOLOR);
        std::vector<intptr_t> &pmc = pointmcolors[0];
        for (int c = 0; c < conf.numpcx; ++c)
          pmc.push_back(c);
      }
      continue;
    }
    double r = dr * (double) j;
    int pcx = 0;
    for (int i = 0; i < npx; ++i) {
      if (i >= zxbounds[pcx+1]) pcx += 1;
      double th = dth * (double) (conf.nzx - i);
      double x = r * cos(th);
      double y = r * sin(th);
      pointpos_x.push_back(x);
      pointpos_y.push_back(y);
      int c = pcy * conf.numpcx + pcx;
      if (i != zxbounds[pcx] && j != zybounds[pcy])
        pointcolors.push_back(c);
      else {
        int p = pointpos_x.size() - 1;
        pointcolors.push_back(MULTICOLOR);
        std::vector<intptr_t> &pmc = pointmcolors[p];
        if (i == zxbounds[pcx] && j == zybounds[pcy])
          pmc.push_back(c - conf.numpcx - 1);
        if (j == zybounds[pcy]) pmc.push_back(c - conf.numpcx);
        if (i == zxbounds[pcx]) pmc.push_back(c - 1);
        pmc.push_back(c);
      }
    }
  }

  // generate zone adjacency lists
  zonestart.reserve(nz);
  zonesize.reserve(nz);
  zonepoints.reserve(4 * nz);
  zonecolors.reserve(nz);
  pcy = 0;
  for (int j = 0; j < conf.nzy; ++j) {
    if (j >= zybounds[pcy+1]) pcy += 1;
    int pcx = 0;
    for (int i = 0; i < conf.nzx; ++i) {
      if (i >= zxbounds[pcx+1]) pcx += 1;
      zonestart.push_back(zonepoints.size());
      int p0 = j * npx + i - (npx - 1);
      if (j == 0) {
        zonesize.push_back(3);
        zonepoints.push_back(0);
      }
      else {
        zonesize.push_back(4);
        zonepoints.push_back(p0);
        zonepoints.push_back(p0 + 1);
      }
      zonepoints.push_back(p0 + npx + 1);
      zonepoints.push_back(p0 + npx);
      zonecolors.push_back(pcy * conf.numpcx + pcx);
    }
  }
}

static void generate_mesh_hex(config &conf,
                              std::vector<double> &pointpos_x,
                              std::vector<double> &pointpos_y,
                              std::vector<intptr_t> &pointcolors,
                              std::map<intptr_t, std::vector<intptr_t> > &pointmcolors,
                              std::vector<intptr_t> &zonestart,
                              std::vector<intptr_t> &zonesize,
                              std::vector<intptr_t> &zonepoints,
                              std::vector<intptr_t> &zonecolors,
                              std::vector<intptr_t> &zxbounds,
                              std::vector<intptr_t> &zybounds)
{
  intptr_t &nz = conf.nz;
  intptr_t &np = conf.np;

  nz = conf.nzx * conf.nzy;
  const int npx = conf.nzx + 1;
  const int npy = conf.nzy + 1;

  // generate point coordinates
  pointpos_x.resize(2 * npx * npy);  // upper bound
  pointpos_y.resize(2 * npx * npy);  // upper bound
  double dx = conf.lenx / (double) (conf.nzx - 1);
  double dy = conf.leny / (double) (conf.nzy - 1);

  std::vector<intptr_t> pbase(npy);
  int p = 0;
  int pcy = 0;
  for (int j = 0; j < npy; ++j) {
    if (j >= zybounds[pcy+1]) pcy += 1;
    pbase[j] = p;
    double y = dy * ((double) j - 0.5);
    y = std::max(0., std::min(conf.leny, y));
    int pcx = 0;
    for (int i = 0; i < npx; ++i) {
      if (i >= zxbounds[pcx+1]) pcx += 1;
      double x = dx * ((double) i - 0.5);
      x = std::max(0., std::min(conf.lenx, x));
      int c = pcy * conf.numpcx + pcx;
      if (i == 0 || i == conf.nzx || j == 0 || j == conf.nzy) {
        pointpos_x[p] = x;
        pointpos_y[p++] = y;
        if (i != zxbounds[pcx] && j != zybounds[pcy])
          pointcolors.push_back(c);
        else {
          int p1 = p - 1;
          pointcolors.push_back(MULTICOLOR);
          std::vector<intptr_t> &pmc = pointmcolors[p1];
          if (j == zybounds[pcy]) pmc.push_back(c - conf.numpcx);
          if (i == zxbounds[pcx]) pmc.push_back(c - 1);
          pmc.push_back(c);
        }
      }
      else {
        pointpos_x[p] = x - dx / 6.;
        pointpos_y[p++] = y + dy / 6.;
        pointpos_x[p] = x + dx / 6.;
        pointpos_y[p++] = y - dy / 6.;
        if (i != zxbounds[pcx] && j != zybounds[pcy]) {
          pointcolors.push_back(c);
          pointcolors.push_back(c);
        }
        else {
          int p1 = p - 2;
          int p2 = p - 1;
          pointcolors.push_back(MULTICOLOR);
          pointcolors.push_back(MULTICOLOR);
          std::vector<intptr_t> &pmc1 = pointmcolors[p1];
          std::vector<intptr_t> &pmc2 = pointmcolors[p2];
          if (i == zxbounds[pcx] && j == zybounds[pcy]) {
            pmc1.push_back(c - conf.numpcx - 1);
            pmc2.push_back(c - conf.numpcx - 1);
            pmc1.push_back(c - 1);
            pmc2.push_back(c - conf.numpcx);
          }
          else if (j == zybounds[pcy]) {
            pmc1.push_back(c - conf.numpcx);
            pmc2.push_back(c - conf.numpcx);
          }
          else {  // i == zxbounds[pcx]
            pmc1.push_back(c - 1);
            pmc2.push_back(c - 1);
          }
          pmc1.push_back(c);
          pmc2.push_back(c);
        }
      }
    } // for i
  } // for j
  np = p;
  pointpos_x.resize(np);
  pointpos_y.resize(np);

  // generate zone adjacency lists
  zonestart.resize(nz);
  zonesize.resize(nz);
  zonepoints.reserve(6 * nz);  // upper bound
  zonecolors.reserve(nz);
  pcy = 0;
  for (int j = 0; j < conf.nzy; ++j) {
    if (j >= zybounds[pcy+1]) pcy += 1;
    int pbasel = pbase[j];
    int pbaseh = pbase[j+1];
    int pcx = 0;
    for (int i = 0; i < conf.nzx; ++i) {
      if (i >= zxbounds[pcx+1]) pcx += 1;
      int z = j * conf.nzx + i;
      std::vector<intptr_t> v(6);
      v[1] = pbasel + 2 * i;
      v[0] = v[1] - 1;
      v[2] = v[1] + 1;
      v[5] = pbaseh + 2 * i;
      v[4] = v[5] + 1;
      v[3] = v[4] + 1;
      if (j == 0) {
        v[0] = pbasel + i;
        v[2] = v[0] + 1;
        if (i == conf.nzx - 1) v.erase(v.begin()+3);
        v.erase(v.begin()+1);
      } // if j
      else if (j == conf.nzy - 1) {
        v[5] = pbaseh + i;
        v[3] = v[5] + 1;
        v.erase(v.begin()+4);
        if (i == 0) v.erase(v.begin()+0);
      } // else if j
      else if (i == 0)
        v.erase(v.begin()+0);
      else if (i == conf.nzx - 1)
        v.erase(v.begin()+3);
      zonestart[z] = zonepoints.size();
      zonesize[z] = v.size();
      zonepoints.insert(zonepoints.end(), v.begin(), v.end());
      zonecolors.push_back(pcy * conf.numpcx + pcx);
    } // for i
  } // for j
}

static void calc_mesh_num_pieces(config &conf)
{
    // pick numpcx, numpcy such that pieces are as close to square
    // as possible
    // we would like:  nzx / numpcx == nzy / numpcy,
    // where numpcx * numpcy = npieces (total number of pieces)
    // this solves to:  numpcx = sqrt(npieces * nzx / nzy)
    // we compute this, assuming nzx <= nzy (swap if necessary)
    double nx = static_cast<double>(conf.nzx);
    double ny = static_cast<double>(conf.nzy);
    bool swapflag = (nx > ny);
    if (swapflag) std::swap(nx, ny);
    double n = sqrt(conf.npieces * nx / ny);
    // need to constrain n to be an integer with npieces % n == 0
    // try rounding n both up and down
    int n1 = floor(n + 1.e-12);
    n1 = std::max(n1, 1);
    while (conf.npieces % n1 != 0) --n1;
    int n2 = ceil(n - 1.e-12);
    while (conf.npieces % n2 != 0) ++n2;
    // pick whichever of n1 and n2 gives blocks closest to square,
    // i.e. gives the shortest long side
    double longside1 = std::max(nx / n1, ny / (conf.npieces/n1));
    double longside2 = std::max(nx / n2, ny / (conf.npieces/n2));
    conf.numpcx = (longside1 <= longside2 ? n1 : n2);
    conf.numpcy = conf.npieces / conf.numpcx;
    if (swapflag) std::swap(conf.numpcx, conf.numpcy);
}

static void generate_mesh(config &conf,
                          std::vector<double> &pointpos_x,
                          std::vector<double> &pointpos_y,
                          std::vector<intptr_t> &pointcolors,
                          std::map<intptr_t, std::vector<intptr_t> > &pointmcolors,
                          std::vector<std::vector<intptr_t> > &mapzp,
                          // std::vector<intptr_t> &zonestart,
                          // std::vector<intptr_t> &zonesize,
                          // std::vector<intptr_t> &zonepoints,
                          std::vector<intptr_t> &zonecolors)
{
  // Do calculations common to all mesh types:
  std::vector<intptr_t> zxbounds;
  std::vector<intptr_t> zybounds;
  calc_mesh_num_pieces(conf);
  zxbounds.push_back(-1);
  for (int pcx = 1; pcx < conf.numpcx; ++pcx)
    zxbounds.push_back(pcx * conf.nzx / conf.numpcx);
  zxbounds.push_back(conf.nzx + 1);
  zybounds.push_back(-1);
  for (int pcy = 1; pcy < conf.numpcy; ++pcy)
    zybounds.push_back(pcy * conf.nzy / conf.numpcy);
  zybounds.push_back(0x7FFFFFFF);

  // Mesh type-specific calculations:
  std::vector<intptr_t> zonestart;
  std::vector<intptr_t> zonesize;
  std::vector<intptr_t> zonepoints;
  if (conf.meshtype == MESH_PIE) {
    generate_mesh_pie(conf, pointpos_x, pointpos_y, pointcolors, pointmcolors,
                      zonestart, zonesize, zonepoints, zonecolors,
                      zxbounds, zybounds);
  } else if (conf.meshtype == MESH_RECT) {
    generate_mesh_rect(conf, pointpos_x, pointpos_y, pointcolors, pointmcolors,
                       zonestart, zonesize, zonepoints, zonecolors,
                       zxbounds, zybounds);
  } else if (conf.meshtype == MESH_HEX) {
    generate_mesh_hex(conf, pointpos_x, pointpos_y, pointcolors, pointmcolors,
                      zonestart, zonesize, zonepoints, zonecolors,
                      zxbounds, zybounds);
  }

  // Convert zone ajancency lists to mapzp format.
  mapzp.resize(conf.nz);
  for (intptr_t z = 0; z < conf.nz; z++) {
    intptr_t p0 = zonestart[z];
    intptr_t znump = zonesize[z];
    for (intptr_t p = p0; p < p0 + znump; p++) {
      mapzp[z].push_back(zonepoints[p]);
    }
  }
}

///
/// I/O
///

config read_config()
{
  std::string pnt_filename = get_input_filename();
  std::string dir = get_directory(pnt_filename);

  printf("Reading %s\n", pnt_filename.c_str());

  std::map<std::string, std::vector<std::string> > params;
  read_params(pnt_filename, params);

  config conf;
  {
    std::string meshtype;
    extract_param(params, "meshtype", 0, meshtype, true);
    if (meshtype == "pie") {
      conf.meshtype = MESH_PIE;
    } else if (meshtype == "rect") {
      conf.meshtype = MESH_RECT;
    } else if (meshtype == "hex") {
      conf.meshtype = MESH_HEX;
    } else {
      assert(0 && "invalid meshtype");
    }

    std::vector<double> meshparams;
    for (int i = 0; i < params["meshparams"].size(); i++) {
      meshparams.push_back(stod(params["meshparams"][i]));
    }

    extract_param(params, "meshparams", 0, conf.meshparams_0, true);
    extract_param(params, "meshparams", 1, conf.meshparams_1, false);
    extract_param(params, "meshparams", 2, conf.meshparams_2, false);
    extract_param(params, "meshparams", 3, conf.meshparams_3, false);
    conf.meshparams_n = params["meshparams"].size();

    init_mesh(conf);
  }

  conf.npieces = get_npieces();
  conf.use_foreign = get_use_foreign();

  printf("Using npieces %ld\n", conf.npieces);

  extract_param(params, "cstop", 0, conf.cstop, false);
  extract_param(params, "tstop", 0, conf.tstop, false);
  extract_param(params, "meshscale", 0, conf.meshscale, false);
  extract_param(params, "subregion", 0, conf.subregion_0, false);
  extract_param(params, "subregion", 1, conf.subregion_1, false);
  extract_param(params, "subregion", 2, conf.subregion_2, false);
  extract_param(params, "subregion", 3, conf.subregion_3, false);
  extract_param(params, "cfl", 0, conf.cfl, false);
  extract_param(params, "cflv", 0, conf.cflv, false);
  extract_param(params, "rinit", 0, conf.rinit, false);
  extract_param(params, "einit", 0, conf.einit, false);
  extract_param(params, "rinitsub", 0, conf.rinitsub, false);
  extract_param(params, "einitsub", 0, conf.einitsub, false);
  extract_param(params, "uinitradial", 0, conf.uinitradial, false);
  extract_param(params, "bcx", 0, conf.bcx_0, false);
  extract_param(params, "bcx", 1, conf.bcx_1, false);
  extract_param(params, "bcy", 0, conf.bcy_0, false);
  extract_param(params, "bcy", 1, conf.bcy_1, false);
  conf.bcx_n = params["bcx"].size();
  conf.bcy_n = params["bcy"].size();
  extract_param(params, "ssmin", 0, conf.ssmin, false);
  extract_param(params, "q1", 0, conf.q1, false);
  extract_param(params, "q2", 0, conf.q2, false);
  extract_param(params, "dtinit", 0, conf.dtinit, false);
  extract_param(params, "chunksize", 0, conf.chunksize, false);

  return conf;
}

const intptr_t nocolors = -1;
const intptr_t manycolors = -2;
const intptr_t pcolors_bits = 64;

void foreign_read_input(HighLevelRuntime *runtime,
                        Context ctx,
                        config conf,
                        PhysicalRegion rz_all[1],
                        PhysicalRegion rp_all[1],
                        PhysicalRegion rs_all[1],
                        PhysicalRegion rm_all[1],
                        PhysicalRegion pcolor_a[1],
                        PhysicalRegion pcolors_a[1],
                        PhysicalRegion pcolor_shared_a[1])
{
  // Read mesh.
  std::vector<double> px_x;
  std::vector<double> px_y;
  std::vector<std::vector<intptr_t> > mapzp;
  {
    // Toss colorings and regenerate them.
    std::vector<intptr_t> pcs;
    std::map<intptr_t, std::vector<intptr_t> > pcm;
    std::vector<intptr_t> zc;
    generate_mesh(conf, px_x, px_y, pcs, pcm, mapzp, zc);
  }

  // Allocate mesh.
  {
    IndexAllocator rz_all_alloc =
      runtime->create_index_allocator(ctx, rz_all[0].get_logical_region().get_index_space());
    IndexAllocator rp_all_alloc =
      runtime->create_index_allocator(ctx, rp_all[0].get_logical_region().get_index_space());
    IndexAllocator rs_all_alloc =
      runtime->create_index_allocator(ctx, rs_all[0].get_logical_region().get_index_space());
    IndexAllocator rm_all_alloc =
      runtime->create_index_allocator(ctx, rm_all[0].get_logical_region().get_index_space());

    rz_all_alloc.alloc(conf.nz);
    rp_all_alloc.alloc(conf.np);
    rs_all_alloc.alloc(conf.ns);
    rm_all_alloc.alloc(conf.npieces);
  }

  // Initialize zones.
  std::vector<intptr_t> m_zstart(conf.npieces);
  std::vector<intptr_t> m_zend(conf.npieces);
  std::vector<intptr_t> m_sstart(conf.npieces);
  std::vector<intptr_t> m_send(conf.npieces);
  {
    RegionAccessor<AT_SOA_8, intptr_t> accessor_znump =
      rz_all[0].get_field_accessor(FIELD_ZNUMP).typeify<intptr_t>().convert<AT_SOA_8>();

    intptr_t zones_per_color = (conf.nz + conf.npieces - 1) / conf.npieces;

    intptr_t zstart = 0, zend = 0, sstart = 0, send = 0;
    for (intptr_t m = 0; m < conf.npieces; m++) {
      zstart = zend;
      zend = std::min(zend + zones_per_color, conf.nz);

      sstart = send;

      for (intptr_t z = zstart; z < zend; z++) {
        intptr_t znump = mapzp[z].size();
        accessor_znump.write(z, znump);
        send += znump;
      }

      m_zstart[m] = zstart;
      m_zend[m] = zend;

      m_sstart[m] = sstart;
      m_send[m] = send;
    }
  }

  // Initialize points.
  {
    RegionAccessor<AT_SOA_8, double> accessor_px_x =
      rp_all[0].get_field_accessor(FIELD_PX_X).typeify<double>().convert<AT_SOA_8>();
    RegionAccessor<AT_SOA_8, double> accessor_px_y =
      rp_all[0].get_field_accessor(FIELD_PX_Y).typeify<double>().convert<AT_SOA_8>();
    RegionAccessor<AT_SOA_1, bool> accessor_has_bcx_0 =
      rp_all[0].get_field_accessor(FIELD_HAS_BCX_0).typeify<bool>().convert<AT_SOA_1>();
    RegionAccessor<AT_SOA_1, bool> accessor_has_bcx_1 =
      rp_all[0].get_field_accessor(FIELD_HAS_BCX_1).typeify<bool>().convert<AT_SOA_1>();
    RegionAccessor<AT_SOA_1, bool> accessor_has_bcy_0 =
      rp_all[0].get_field_accessor(FIELD_HAS_BCY_0).typeify<bool>().convert<AT_SOA_1>();
    RegionAccessor<AT_SOA_1, bool> accessor_has_bcy_1 =
      rp_all[0].get_field_accessor(FIELD_HAS_BCY_1).typeify<bool>().convert<AT_SOA_1>();

    RegionAccessor<AT_SOA_8, intptr_t> accessor_pcolor =
      pcolor_a[0].get_accessor().typeify<intptr_t>().convert<AT_SOA_8>();

    RegionAccessor<AT_SOA_8, uint64_t> accessor_pcolors =
      pcolors_a[0].get_accessor().typeify<uint64_t>().convert<AT_SOA_8>();

    RegionAccessor<AT_SOA_8, intptr_t> accessor_pcolor_shared =
      pcolor_shared_a[0].get_accessor().typeify<intptr_t>().convert<AT_SOA_8>();

    const double eps = 1e-12;
    for (intptr_t p = 0; p < conf.np; p++) {
      accessor_px_x.write(p, px_x[p] * conf.meshscale);
      accessor_px_y.write(p, px_y[p] * conf.meshscale);

      accessor_has_bcx_0.write(p, conf.bcx_n > 0 && fabs(px_x[p] - conf.bcx_0) < eps);
      accessor_has_bcx_1.write(p, conf.bcx_n > 1 && fabs(px_x[p] - conf.bcx_1) < eps);
      accessor_has_bcy_0.write(p, conf.bcy_n > 0 && fabs(px_y[p] - conf.bcy_0) < eps);
      accessor_has_bcy_1.write(p, conf.bcy_n > 1 && fabs(px_y[p] - conf.bcy_1) < eps);
    }

    intptr_t pcolors_words = (conf.npieces + pcolors_bits - 1)/pcolors_bits;
    std::vector<intptr_t> pcolors(conf.np, nocolors);
    for (intptr_t m = 0; m < conf.npieces; m++) {
      intptr_t zstart = m_zstart[m], zend = m_zend[m];
      intptr_t zcolor = m; // zone color is the same as mesh piece
      for (intptr_t z = zstart; z < zend; z++) {
        for (std::vector<intptr_t>::iterator ip = mapzp[z].begin(), ep = mapzp[z].end(); ip != ep; ip++) {
          intptr_t p = *ip;

          if (pcolors[p] == nocolors || pcolors[p] == zcolor) {
            pcolors[p] = zcolor;
          } else {
            pcolors[p] = manycolors;
          }

          intptr_t word = p*pcolors_words + zcolor/pcolors_bits;
          intptr_t bit = zcolor%pcolors_bits;
          accessor_pcolors.write(word, accessor_pcolors.read(word) | static_cast<uint64_t>(1 << bit));
        }
      }
    }

    for (intptr_t p = 0; p < conf.np; p++) {
      intptr_t c = pcolors[p];
      assert(c != nocolors);
      accessor_pcolor.write(p, pcolors[p]);
    }

    for (intptr_t p = 0; p < conf.np; p++) {
      accessor_pcolor_shared.write(p, nocolors);
    }

    std::vector<bool> hascolor(conf.np, false);
    bool completely_exhausted = false;
    while (!completely_exhausted) {
      completely_exhausted = true;
      for (intptr_t m = 0; m < conf.npieces; m++) {
        intptr_t zstart = m_zstart[m], zend = m_zend[m];
        intptr_t zcolor = m; // zone color is the same as mesh piece
        bool has_new_point = false;
        for (intptr_t z = zstart; z < zend && !has_new_point; z++) {
          for (std::vector<intptr_t>::iterator ip = mapzp[z].begin(), ep = mapzp[z].end(); ip != ep; ip++) {
            intptr_t p = *ip;

            if (pcolors[p] == manycolors) {
              if (!hascolor[p]) {
                accessor_pcolor_shared.write(p, zcolor);
                hascolor[p] = true;
                has_new_point = true;
                completely_exhausted = false;
                break;
              }
            }
          }
        }
      }
    }
  }

  // Initialize sides.
  {
    RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsz =
      rs_all[0].get_field_accessor(FIELD_MAPSZ).typeify<ptr_t>().convert<AT_SOA_4>();
    RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp1_pointer =
      rs_all[0].get_field_accessor(FIELD_MAPSP1_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
    RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp1_region =
      rs_all[0].get_field_accessor(FIELD_MAPSP1_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
    RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp2_pointer =
      rs_all[0].get_field_accessor(FIELD_MAPSP2_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
    RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp2_region =
      rs_all[0].get_field_accessor(FIELD_MAPSP2_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
    RegionAccessor<AT_SOA_4, ptr_t> accessor_mapss3 =
      rs_all[0].get_field_accessor(FIELD_MAPSS3).typeify<ptr_t>().convert<AT_SOA_4>();
    RegionAccessor<AT_SOA_4, ptr_t> accessor_mapss4 =
      rs_all[0].get_field_accessor(FIELD_MAPSS4).typeify<ptr_t>().convert<AT_SOA_4>();

    for (intptr_t m = 0; m < conf.npieces; m++) {
      intptr_t zstart = m_zstart[m], zend = m_zend[m];
      intptr_t sstart = m_sstart[m];
      for (intptr_t z = zstart; z < zend; z++) {
        intptr_t znump = mapzp[z].size();
        for (intptr_t is = 0; is < znump; is++) {
          intptr_t is3 = (is + znump - 1)%znump;
          intptr_t is4 = (is + 1)%znump;

          intptr_t s = is + sstart;
          intptr_t s3 = is3 + sstart;
          intptr_t s4 = is4 + sstart;
          intptr_t p1 = mapzp[z][is];
          intptr_t p2 = mapzp[z][is4];

          accessor_mapsz.write(s, z);
          accessor_mapsp1_pointer.write(s, p1);
          accessor_mapsp1_region.write(s, 0);
          accessor_mapsp2_pointer.write(s, p2);
          accessor_mapsp2_region.write(s, 0);
          accessor_mapss3.write(s, s3);
          accessor_mapss4.write(s, s4);
        }
        sstart += znump;
      }
    }
  }

  // Initialize mesh pieces.
  {
    RegionAccessor<AT_SOA_8, intptr_t> accessor_mcolor =
      rm_all[0].get_field_accessor(FIELD_MCOLOR).typeify<intptr_t>().convert<AT_SOA_8>();
    RegionAccessor<AT_SOA_8, intptr_t> accessor_zstart =
      rm_all[0].get_field_accessor(FIELD_ZSTART).typeify<intptr_t>().convert<AT_SOA_8>();
    RegionAccessor<AT_SOA_8, intptr_t> accessor_zend =
      rm_all[0].get_field_accessor(FIELD_ZEND).typeify<intptr_t>().convert<AT_SOA_8>();
    RegionAccessor<AT_SOA_8, intptr_t> accessor_sstart =
      rm_all[0].get_field_accessor(FIELD_SSTART).typeify<intptr_t>().convert<AT_SOA_8>();
    RegionAccessor<AT_SOA_8, intptr_t> accessor_send =
      rm_all[0].get_field_accessor(FIELD_SEND).typeify<intptr_t>().convert<AT_SOA_8>();

    for (intptr_t m = 0; m < conf.npieces; m++) {
      accessor_mcolor.write(m, m);
      accessor_zstart.write(m, m_zstart[m]);
      accessor_zend.write(m, m_zend[m]);
      accessor_sstart.write(m, m_sstart[m]);
      accessor_send.write(m, m_send[m]);
    }
  }

}

void write_output(HighLevelRuntime *runtime,
                  Context ctx,
                  config conf,
                  PhysicalRegion rz_all[1],
                  PhysicalRegion rp_all[1],
                  PhysicalRegion rs_all[1])
{
  // state *ctx = (state *)raw_ctx;
  // printf("#%4s\n", "zr");
  // for (int iz = 0; iz < ctx->conf.nz; iz++) {
  //   printf("%5d%18.8e\n", iz + 1, ctx->zr[iz]);
  // }
  // printf("#  ze\n");
  // for (int iz = 0; iz < ctx->conf.nz; iz++) {
  //   printf("%5d%18.8e\n", iz + 1, ctx->ze[iz]);
  // }
  // printf("#  zp\n");
  // for (int iz = 0; iz < ctx->conf.nz; iz++) {
  //   printf("%5d%18.8e\n", iz + 1, ctx->zp[iz]);
  // }
}

void foreign_validate_output(HighLevelRuntime *runtime,
                             Context ctx,
                             config conf,
                             PhysicalRegion rz_all[1],
                             PhysicalRegion rp_all[1],
                             PhysicalRegion rs_all[1])
{
  std::vector<double> sol_zr, sol_ze, sol_zp;

  {
    std::string xy_filename = get_solution_filename();
    std::ifstream xy_file(xy_filename.c_str(), std::ios::in);
    if (!xy_file) {
      assert(0 && "solution file does not exist");
    }

    {
      std::string line;
      getline(xy_file, line);
      assert(line == "#  zr");
    }

    for (int i = 0; i < conf.nz; i++) {
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

    for (int i = 0; i < conf.nz; i++) {
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

    for (int i = 0; i < conf.nz; i++) {
      int iz;
      double zp;
      xy_file >> iz;
      xy_file >> zp;
      sol_zp.push_back(zp);
    }
  }

  RegionAccessor<AT_SOA_8, double> accessor_zr =
    rz_all[0].get_field_accessor(FIELD_ZR).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_ze =
    rz_all[0].get_field_accessor(FIELD_ZE).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zp =
    rz_all[0].get_field_accessor(FIELD_ZP).typeify<double>().convert<AT_SOA_8>();

  double absolute_eps = 1e-12;
  double relative_eps = 1e-8;
  for (intptr_t iz = 0; iz < conf.nz; iz++) {
    double ck = accessor_zr.read(iz);
    double sol = sol_zr[iz];
    if (fabs(ck - sol) < absolute_eps) {
      continue;
    }
    if (fabs(ck - sol) / sol < relative_eps) {
      continue;
    }
    assert(0 && "zr value out of bounds");
  }

  for (intptr_t iz = 0; iz < conf.nz; iz++) {
    double ck = accessor_ze.read(iz);
    double sol = sol_ze[iz];
    if (fabs(ck - sol) < absolute_eps) {
      continue;
    }
    if (fabs(ck - sol) / sol < relative_eps) {
      continue;
    }
    assert(0 && "ze value out of bounds");
  }

  for (intptr_t iz = 0; iz < conf.nz; iz++) {
    double ck = accessor_zp.read(iz);
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

double get_abs_time()
{
  return LegionRuntime::LowLevel::Clock::abs_time();
}

void print_global_elapsed_time(double start_time, double end_time)
{
  double delta_time = end_time - start_time;
  printf("\n**************************************\n");
  printf("total problem run time=%15.6e\n", delta_time);
  printf("**************************************\n\n");
}

void print_simulation_start()
{
  printf("Starting simulation\n");
}

void print_simulation_loop(intptr_t cycle, double time, double dt,
                           double start_time, double last_time,
                           double current_time, intptr_t interval)
{
  printf("cycle %4ld    sim time %.3e    dt %.3e    time %.3e (per iteration) %.3e (total)\n",
         cycle, time, dt, (current_time - last_time)/interval, current_time - start_time);
}

///
/// Coloring
///

Coloring foreign_all_zones_coloring(HighLevelRuntime *runtime,
                                    Context ctx,
                                    config conf,
                                    PhysicalRegion rm_all[1])
{
  Coloring result;

  RegionAccessor<AT_SOA_8, intptr_t> accessor_zstart =
    rm_all[0].get_field_accessor(FIELD_ZSTART).typeify<intptr_t>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, intptr_t> accessor_zend =
    rm_all[0].get_field_accessor(FIELD_ZEND).typeify<intptr_t>().convert<AT_SOA_8>();

  for (intptr_t m = 0; m < conf.npieces; m++) {
    intptr_t zstart = accessor_zstart.read(m);
    intptr_t zend = accessor_zend.read(m);
    result[m].ranges.insert(std::pair<ptr_t, ptr_t>(zstart, zend - 1));
  }

  return result;
}

Coloring foreign_all_points_coloring(HighLevelRuntime *runtime,
                                     Context ctx,
                                     config conf,
                                     PhysicalRegion pcolor_a[1])
{
  Coloring result;

  // Force all colors to exist, even if empty.
  result[0];
  result[1];

  RegionAccessor<AT_SOA_8, intptr_t> accessor_pcolor =
    pcolor_a[0].get_accessor().typeify<intptr_t>().convert<AT_SOA_8>();

  intptr_t start_p = -1, start_c = nocolors;
  intptr_t p;
  for (p = 0; p < conf.np; p++) {
    intptr_t c = accessor_pcolor.read(p);

    if (start_p >= 0) {
      if (start_c == c) {
        continue;
      }

      if (start_c == manycolors) {
        result[1].ranges.insert(std::pair<ptr_t, ptr_t>(start_p, p - 1));
      } else {
        result[0].ranges.insert(std::pair<ptr_t, ptr_t>(start_p, p - 1));
      }
    }

    start_p = p;
    start_c = c;
  }

  if (start_p >= 0) {
    if (start_c == manycolors) {
      result[1].ranges.insert(std::pair<ptr_t, ptr_t>(start_p, p - 1));
    } else {
      result[0].ranges.insert(std::pair<ptr_t, ptr_t>(start_p, p - 1));
    }
  }

  return result;
}

Coloring foreign_private_points_coloring(HighLevelRuntime *runtime,
                                         Context ctx,
                                         config conf,
                                         PhysicalRegion pcolor_a[1])
{
  Coloring result;

  // Force all colors to exist even if empty.
  for (intptr_t c = 0; c < conf.npieces; c++) {
    result[c];
  }

  RegionAccessor<AT_SOA_8, intptr_t> accessor_pcolor =
    pcolor_a[0].get_accessor().typeify<intptr_t>().convert<AT_SOA_8>();

  intptr_t start_p = -1, start_c = nocolors;
  intptr_t p;
  for (p = 0; p < conf.np; p++) {
    intptr_t c = accessor_pcolor.read(p);
    assert(c != nocolors);

    if (start_p >= 0) {
      if (start_c == c) {
        continue;
      }

      if (start_c != manycolors) {
        result[start_c].ranges.insert(std::pair<ptr_t, ptr_t>(start_p, p - 1));
      }
    }

    start_p = p;
    start_c = c;
  }

  if (start_p >= 0 && start_c != manycolors) {
    result[start_c].ranges.insert(std::pair<ptr_t, ptr_t>(start_p, p - 1));
  }

  return result;
}

Coloring foreign_ghost_points_coloring(HighLevelRuntime *runtime,
                                       Context ctx,
                                       config conf,
                                       PhysicalRegion pcolor_a[1],
                                       PhysicalRegion pcolors_a[1])
{
  Coloring result;

  // Force all colors to exist even if empty.
  for (intptr_t c = 0; c < conf.npieces; c++) {
    result[c];
  }

  RegionAccessor<AT_SOA_8, intptr_t> accessor_pcolor =
    pcolor_a[0].get_accessor().typeify<intptr_t>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, uint64_t> accessor_pcolors =
    pcolors_a[0].get_accessor().typeify<uint64_t>().convert<AT_SOA_8>();

  intptr_t pcolors_words = (conf.npieces + pcolors_bits - 1)/pcolors_bits;
  for (intptr_t p = 0; p < conf.np; p++) {
    intptr_t pcolors = accessor_pcolor.read(p);
    if (pcolors == manycolors) {
      for (intptr_t w = 0; w < pcolors_words; w++) {
        uint64_t word = accessor_pcolors.read(p*pcolors_words + w);
        for (intptr_t bit = 0; bit < pcolors_bits; bit++) {
          intptr_t c = w*pcolors_bits + bit;
          if (c >= conf.npieces) {
            break;
          }
          if (word & static_cast<uint64_t>(1 << bit)) {
            result[c].points.insert(p);
          }
        }
      }
    }
  }

  return result;
}

Coloring foreign_shared_points_coloring(HighLevelRuntime *runtime,
                                        Context ctx,
                                        config conf,
                                        PhysicalRegion pcolor_shared_a[1])
{
  Coloring result;

  // Force all colors to exist even if empty.
  for (intptr_t c = 0; c < conf.npieces; c++) {
    result[c];
  }

  RegionAccessor<AT_SOA_8, intptr_t> accessor_pcolor_shared =
    pcolor_shared_a[0].get_accessor().typeify<intptr_t>().convert<AT_SOA_8>();

  for (intptr_t p = 0; p < conf.np; p++) {
    intptr_t c = accessor_pcolor_shared.read(p);
    if (c != nocolors) {
      // printf("shared point %ld color %ld\n", p, c);
      result[c].points.insert(p);
    }
  }

  return result;
}

Coloring foreign_all_sides_coloring(HighLevelRuntime *runtime,
                                 Context ctx,
                                 config conf,
                                 PhysicalRegion rm_all[1])
{
  Coloring result;

  RegionAccessor<AT_SOA_8, intptr_t> accessor_sstart =
    rm_all[0].get_field_accessor(FIELD_SSTART).typeify<intptr_t>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, intptr_t> accessor_send =
    rm_all[0].get_field_accessor(FIELD_SEND).typeify<intptr_t>().convert<AT_SOA_8>();

  for (intptr_t m = 0; m < conf.npieces; m++) {
    intptr_t sstart = accessor_sstart.read(m);
    intptr_t send = accessor_send.read(m);
    result[m].ranges.insert(std::pair<ptr_t, ptr_t>(sstart, send - 1));
  }

  return result;
}

///
/// Kernels
///

double calc_global_dt(double dt, double dtfac, double dtinit,
                      double dtmax, double dthydro,
                      double time, double tstop, intptr_t cycle)
{
  double dtlast = dt;

  dt = dtmax;

  if (cycle == 0) {
    dt = fmin(dt, dtinit);
  } else {
    double dtrecover = dtfac * dtlast;
    dt = fmin(dt, dtrecover);
  }

  dt = fmin(dt, tstop - time);
  dt = fmin(dt, dthydro);

  return dt;
}

void foreign_init_step_zones(intptr_t zstart,
                             intptr_t zend,
                             PhysicalRegion rz[2])
{
  RegionAccessor<AT_SOA_8, double> accessor_zvol =
    rz[0].get_field_accessor(FIELD_ZVOL).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zxp_x =
    rz[1].get_field_accessor(FIELD_ZXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zxp_y =
    rz[1].get_field_accessor(FIELD_ZXP_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zx_x =
    rz[1].get_field_accessor(FIELD_ZX_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zx_y =
    rz[1].get_field_accessor(FIELD_ZX_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zareap =
    rz[1].get_field_accessor(FIELD_ZAREAP).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zarea =
    rz[1].get_field_accessor(FIELD_ZAREA).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zvol0 =
    rz[1].get_field_accessor(FIELD_ZVOL0).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zvolp =
    rz[1].get_field_accessor(FIELD_ZVOLP).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zdl =
    rz[1].get_field_accessor(FIELD_ZDL).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zw =
    rz[1].get_field_accessor(FIELD_ZW).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zuc_x =
    rz[1].get_field_accessor(FIELD_ZUC_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zuc_y =
    rz[1].get_field_accessor(FIELD_ZUC_Y).typeify<double>().convert<AT_SOA_8>();

  for (intptr_t z = zstart; z < zend; z++) {
    accessor_zxp_x.write(z, 0.0);
    accessor_zxp_y.write(z, 0.0);
    accessor_zx_x.write(z, 0.0);
    accessor_zx_y.write(z, 0.0);
    accessor_zareap.write(z, 0.0);
    accessor_zarea.write(z, 0.0);
    accessor_zvol0.write(z, accessor_zvol.read(z));
    accessor_zvolp.write(z, 0.0);
    accessor_zvol.write(z, 0.0);
    accessor_zdl.write(z, 1e99);
    accessor_zw.write(z, 0.0);
    accessor_zuc_x.write(z, 0.0);
    accessor_zuc_y.write(z, 0.0);
  }
}

void foreign_calc_centers(intptr_t sstart,
                          intptr_t send,
                          PhysicalRegion rz[2],
                          PhysicalRegion rpp[1],
                          PhysicalRegion rpg[1],
                          PhysicalRegion rs[2])
{
  RegionAccessor<AT_SOA_8, intptr_t> accessor_znump =
    rz[0].get_field_accessor(FIELD_ZNUMP).typeify<intptr_t>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zxp_x =
    rz[1].get_field_accessor(FIELD_ZXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zxp_y =
    rz[1].get_field_accessor(FIELD_ZXP_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpp_pxp_x =
    rpp[0].get_field_accessor(FIELD_PXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpp_pxp_y =
    rpp[0].get_field_accessor(FIELD_PXP_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpg_pxp_x =
    rpg[0].get_field_accessor(FIELD_PXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpg_pxp_y =
    rpg[0].get_field_accessor(FIELD_PXP_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsz =
    rs[0].get_field_accessor(FIELD_MAPSZ).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp1_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP1_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp1_region =
    rs[0].get_field_accessor(FIELD_MAPSP1_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp2_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP2_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp2_region =
    rs[0].get_field_accessor(FIELD_MAPSP2_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_8, double> accessor_exp_x =
    rs[1].get_field_accessor(FIELD_EXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_exp_y =
    rs[1].get_field_accessor(FIELD_EXP_Y).typeify<double>().convert<AT_SOA_8>();

  for (intptr_t s = sstart; s < send; s++) {
    ptr_t z = accessor_mapsz.read(s);
    ptr_t p1_pointer = accessor_mapsp1_pointer.read(s);
    uint32_t p1_region = accessor_mapsp1_region.read(s);
    ptr_t p2_pointer = accessor_mapsp2_pointer.read(s);
    uint32_t p2_region = accessor_mapsp2_region.read(s);

    vec2 p1_pxp;
    switch (p1_region) {
    case 1:
      p1_pxp.x = accessor_rpp_pxp_x.read(p1_pointer);
      p1_pxp.y = accessor_rpp_pxp_y.read(p1_pointer);
      break;
    case 2:
      p1_pxp.x = accessor_rpg_pxp_x.read(p1_pointer);
      p1_pxp.y = accessor_rpg_pxp_y.read(p1_pointer);
      break;
    }

    vec2 p2_pxp;
    switch (p2_region) {
    case 1:
      p2_pxp.x = accessor_rpp_pxp_x.read(p2_pointer);
      p2_pxp.y = accessor_rpp_pxp_y.read(p2_pointer);
      break;
    case 2:
      p2_pxp.x = accessor_rpg_pxp_x.read(p2_pointer);
      p2_pxp.y = accessor_rpg_pxp_y.read(p2_pointer);
      break;
    }

    vec2 exp = scale(add(p1_pxp, p2_pxp), 0.5);
    accessor_exp_x.write(s, exp.x);
    accessor_exp_y.write(s, exp.y);

    double znump = static_cast<double>(accessor_znump.read(z));
    accessor_zxp_x.write(z, accessor_zxp_x.read(z) + p1_pxp.x/znump);
    accessor_zxp_y.write(z, accessor_zxp_y.read(z) + p1_pxp.y/znump);
  }
}

void foreign_calc_volumes(intptr_t sstart,
                          intptr_t send,
                          PhysicalRegion rz[2],
                          PhysicalRegion rpp[1],
                          PhysicalRegion rpg[1],
                          PhysicalRegion rs[2])
{
  RegionAccessor<AT_SOA_8, double> accessor_zxp_x =
    rz[0].get_field_accessor(FIELD_ZXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zxp_y =
    rz[0].get_field_accessor(FIELD_ZXP_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zareap =
    rz[1].get_field_accessor(FIELD_ZAREAP).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zvolp =
    rz[1].get_field_accessor(FIELD_ZVOLP).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpp_pxp_x =
    rpp[0].get_field_accessor(FIELD_PXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpp_pxp_y =
    rpp[0].get_field_accessor(FIELD_PXP_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpg_pxp_x =
    rpg[0].get_field_accessor(FIELD_PXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpg_pxp_y =
    rpg[0].get_field_accessor(FIELD_PXP_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsz =
    rs[0].get_field_accessor(FIELD_MAPSZ).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp1_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP1_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp1_region =
    rs[0].get_field_accessor(FIELD_MAPSP1_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp2_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP2_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp2_region =
    rs[0].get_field_accessor(FIELD_MAPSP2_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_8, double> accessor_sareap =
    rs[1].get_field_accessor(FIELD_SAREAP).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_svolp =
    rs[1].get_field_accessor(FIELD_SVOLP).typeify<double>().convert<AT_SOA_8>();

  for (intptr_t s = sstart; s < send; s++) {
    ptr_t z = accessor_mapsz.read(s);
    ptr_t p1_pointer = accessor_mapsp1_pointer.read(s);
    uint32_t p1_region = accessor_mapsp1_region.read(s);
    ptr_t p2_pointer = accessor_mapsp2_pointer.read(s);
    uint32_t p2_region = accessor_mapsp2_region.read(s);

    vec2 zxp;
    zxp.x = accessor_zxp_x.read(z);
    zxp.y = accessor_zxp_y.read(z);

    vec2 p1_pxp;
    switch (p1_region) {
    case 1:
      p1_pxp.x = accessor_rpp_pxp_x.read(p1_pointer);
      p1_pxp.y = accessor_rpp_pxp_y.read(p1_pointer);
      break;
    case 2:
      p1_pxp.x = accessor_rpg_pxp_x.read(p1_pointer);
      p1_pxp.y = accessor_rpg_pxp_y.read(p1_pointer);
      break;
    }

    vec2 p2_pxp;
    switch (p2_region) {
    case 1:
      p2_pxp.x = accessor_rpp_pxp_x.read(p2_pointer);
      p2_pxp.y = accessor_rpp_pxp_y.read(p2_pointer);
      break;
    case 2:
      p2_pxp.x = accessor_rpg_pxp_x.read(p2_pointer);
      p2_pxp.y = accessor_rpg_pxp_y.read(p2_pointer);
      break;
    }

    double sa = 0.5 * cross(sub(p2_pxp, p1_pxp), sub(zxp, p1_pxp));
    double sv = (1.0 / 3.0) * sa * (p1_pxp.x + p2_pxp.x + zxp.x);

    accessor_sareap.write(s, sa);
    accessor_svolp.write(s, sv);
    accessor_zareap.write(z, accessor_zareap.read(z) + sa);
    accessor_zvolp.write(z, accessor_zvolp.read(z) + sv);
    assert(sv > 0.0);
  }
}

void foreign_calc_surface_vecs(intptr_t sstart,
                               intptr_t send,
                               PhysicalRegion rz[1],
                               PhysicalRegion rs[2])
{
  RegionAccessor<AT_SOA_8, double> accessor_zxp_x =
    rz[0].get_field_accessor(FIELD_ZXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zxp_y =
    rz[0].get_field_accessor(FIELD_ZXP_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsz =
    rs[0].get_field_accessor(FIELD_MAPSZ).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_8, double> accessor_exp_x =
    rs[0].get_field_accessor(FIELD_EXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_exp_y =
    rs[0].get_field_accessor(FIELD_EXP_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_ssurfp_x =
    rs[1].get_field_accessor(FIELD_SSURFP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_ssurfp_y =
    rs[1].get_field_accessor(FIELD_SSURFP_Y).typeify<double>().convert<AT_SOA_8>();

  for (intptr_t s = sstart; s < send; s++) {
    ptr_t z = accessor_mapsz.read(s);

    vec2 exp;
    exp.x = accessor_exp_x.read(s);
    exp.y = accessor_exp_y.read(s);

    vec2 zxp;
    zxp.x = accessor_zxp_x.read(z);
    zxp.y = accessor_zxp_y.read(z);

    vec2 ssurfp = rotateCCW(sub(exp, zxp));
    accessor_ssurfp_x.write(s, ssurfp.x);
    accessor_ssurfp_y.write(s, ssurfp.y);
  }
}

void foreign_calc_edge_len(intptr_t sstart,
                           intptr_t send,
                           PhysicalRegion rpp[1],
                           PhysicalRegion rpg[1],
                           PhysicalRegion rs[2])
{
  RegionAccessor<AT_SOA_8, double> accessor_rpp_pxp_x =
    rpp[0].get_field_accessor(FIELD_PXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpp_pxp_y =
    rpp[0].get_field_accessor(FIELD_PXP_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpg_pxp_x =
    rpg[0].get_field_accessor(FIELD_PXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpg_pxp_y =
    rpg[0].get_field_accessor(FIELD_PXP_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp1_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP1_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp1_region =
    rs[0].get_field_accessor(FIELD_MAPSP1_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp2_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP2_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp2_region =
    rs[0].get_field_accessor(FIELD_MAPSP2_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_8, double> accessor_elen =
    rs[1].get_field_accessor(FIELD_ELEN).typeify<double>().convert<AT_SOA_8>();

  for (intptr_t s = sstart; s < send; s++) {
    ptr_t p1_pointer = accessor_mapsp1_pointer.read(s);
    uint32_t p1_region = accessor_mapsp1_region.read(s);
    ptr_t p2_pointer = accessor_mapsp2_pointer.read(s);
    uint32_t p2_region = accessor_mapsp2_region.read(s);

    vec2 p1_pxp;
    switch (p1_region) {
    case 1:
      p1_pxp.x = accessor_rpp_pxp_x.read(p1_pointer);
      p1_pxp.y = accessor_rpp_pxp_y.read(p1_pointer);
      break;
    case 2:
      p1_pxp.x = accessor_rpg_pxp_x.read(p1_pointer);
      p1_pxp.y = accessor_rpg_pxp_y.read(p1_pointer);
      break;
    }

    vec2 p2_pxp;
    switch (p2_region) {
    case 1:
      p2_pxp.x = accessor_rpp_pxp_x.read(p2_pointer);
      p2_pxp.y = accessor_rpp_pxp_y.read(p2_pointer);
      break;
    case 2:
      p2_pxp.x = accessor_rpg_pxp_x.read(p2_pointer);
      p2_pxp.y = accessor_rpg_pxp_y.read(p2_pointer);
      break;
    }

    double elen = length(sub(p2_pxp, p1_pxp));

    accessor_elen.write(s, elen);
  }
}

void foreign_calc_char_len(intptr_t sstart,
                           intptr_t send,
                           PhysicalRegion rz[2],
                           PhysicalRegion rs[1])
{
  RegionAccessor<AT_SOA_8, intptr_t> accessor_znump =
    rz[0].get_field_accessor(FIELD_ZNUMP).typeify<intptr_t>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zdl =
    rz[1].get_field_accessor(FIELD_ZDL).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsz =
    rs[0].get_field_accessor(FIELD_MAPSZ).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_8, double> accessor_sareap =
    rs[0].get_field_accessor(FIELD_SAREAP).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_elen =
    rs[0].get_field_accessor(FIELD_ELEN).typeify<double>().convert<AT_SOA_8>();

  for (intptr_t s = sstart; s < send; s++) {
    ptr_t z = accessor_mapsz.read(s);

    intptr_t znump = accessor_znump.read(z);

    double area = accessor_sareap.read(s);
    double base = accessor_elen.read(s);
    double fac;
    if (znump == 3) {
      fac = 3.0;
    } else {
      fac = 4.0;
    }
    double sdl = fac * area / base;
    accessor_zdl.write(z, fmin(accessor_zdl.read(z), sdl));
  }
}

void foreign_calc_rho_half(intptr_t zstart,
                           intptr_t zend,
                           PhysicalRegion rz[2])
{
  RegionAccessor<AT_SOA_8, double> accessor_zvolp =
    rz[0].get_field_accessor(FIELD_ZVOLP).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zm =
    rz[0].get_field_accessor(FIELD_ZM).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zrp =
    rz[1].get_field_accessor(FIELD_ZRP).typeify<double>().convert<AT_SOA_8>();

  for (intptr_t z = zstart; z < zend; z++) {
    accessor_zrp.write(z, accessor_zm.read(z) / accessor_zvolp.read(z));
  }
}

void foreign_sum_point_mass(intptr_t sstart,
                            intptr_t send,
                            PhysicalRegion rz[1],
                            PhysicalRegion rpp[1],
                            PhysicalRegion rpg[1],
                            PhysicalRegion rs[1])
{
  RegionAccessor<AT_SOA_8, double> accessor_zareap =
    rz[0].get_field_accessor(FIELD_ZAREAP).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zrp =
    rz[0].get_field_accessor(FIELD_ZRP).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpp_pmaswt =
    rpp[0].get_field_accessor(FIELD_PMASWT).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpg_pmaswt =
    rpg[0].get_accessor().typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsz =
    rs[0].get_field_accessor(FIELD_MAPSZ).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp1_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP1_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp1_region =
    rs[0].get_field_accessor(FIELD_MAPSP1_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapss3 =
    rs[0].get_field_accessor(FIELD_MAPSS3).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_8, double> accessor_smf =
    rs[0].get_field_accessor(FIELD_SMF).typeify<double>().convert<AT_SOA_8>();

  for (intptr_t s = sstart; s < send; s++) {
    ptr_t z = accessor_mapsz.read(s);
    ptr_t p1_pointer = accessor_mapsp1_pointer.read(s);
    uint32_t p1_region = accessor_mapsp1_region.read(s);
    ptr_t s3 = accessor_mapss3.read(s);

    double m = accessor_zrp.read(z) * accessor_zareap.read(z) *
      0.5 * (accessor_smf.read(s) + accessor_smf.read(s3));

    switch (p1_region) {
    case 1:
      accessor_rpp_pmaswt.write(p1_pointer, accessor_rpp_pmaswt.read(p1_pointer) + m);
      break;
    case 2:
      accessor_rpg_pmaswt.reduce<reduction_plus_double>(p1_pointer, m);
      break;
    }
  }
}

void foreign_calc_state_at_half(double gamma,
                                double ssmin,
                                double dt,
                                intptr_t zstart,
                                intptr_t zend,
                                PhysicalRegion rz[2])
{
  RegionAccessor<AT_SOA_8, double> accessor_zvol0 =
    rz[0].get_field_accessor(FIELD_ZVOL0).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zvolp =
    rz[0].get_field_accessor(FIELD_ZVOLP).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zm =
    rz[0].get_field_accessor(FIELD_ZM).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zr =
    rz[0].get_field_accessor(FIELD_ZR).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_ze =
    rz[0].get_field_accessor(FIELD_ZE).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zwrate =
    rz[0].get_field_accessor(FIELD_ZWRATE).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zp =
    rz[1].get_field_accessor(FIELD_ZP).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zss =
    rz[1].get_field_accessor(FIELD_ZSS).typeify<double>().convert<AT_SOA_8>();

  double gm1 = gamma - 1.0;
  double ss2 = fmax(ssmin * ssmin, 1e-99);
  double dth = 0.5 * dt;

  for (intptr_t z = zstart; z < zend; z++) {
    double zm = accessor_zm.read(z);
    double zr = accessor_zr.read(z);
    double ze = accessor_ze.read(z);
    double zvol0 = accessor_zvol0.read(z);
    double zvolp = accessor_zvolp.read(z);
    double zwrate = accessor_zwrate.read(z);

    double rx = zr;
    double ex = fmax(ze, 0.0);
    double px = gm1 * rx * ex;
    double prex = gm1 * ex;
    double perx = gm1 * rx;
    double csqd = fmax(ss2, prex + perx * px / (rx * rx));
    double z0per = perx;
    double zss = sqrt(csqd);
    accessor_zss.write(z, zss);

    double zminv = 1.0 / zm;
    double dv = (zvolp - zvol0) * zminv;
    double bulk = zr * zss * zss;
    double denom = 1.0 + 0.5 * z0per * dv;
    double src = zwrate * dth * zminv;
    accessor_zp.write(z, px + (z0per * src - zr * bulk * dv) / denom);
  }
}

void foreign_calc_force_pgas(intptr_t sstart,
                             intptr_t send,
                             PhysicalRegion rz[1],
                             PhysicalRegion rs[2])
{
  RegionAccessor<AT_SOA_8, double> accessor_zp =
    rz[0].get_field_accessor(FIELD_ZP).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsz =
    rs[0].get_field_accessor(FIELD_MAPSZ).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_8, double> accessor_ssurfp_x =
    rs[0].get_field_accessor(FIELD_SSURFP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_ssurfp_y =
    rs[0].get_field_accessor(FIELD_SSURFP_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_sfp_x =
    rs[1].get_field_accessor(FIELD_SFP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_sfp_y =
    rs[1].get_field_accessor(FIELD_SFP_Y).typeify<double>().convert<AT_SOA_8>();

  for (intptr_t s = sstart; s < send; s++) {
    ptr_t z = accessor_mapsz.read(s);

    vec2 ssurfp;
    ssurfp.x = accessor_ssurfp_x.read(s);
    ssurfp.y = accessor_ssurfp_y.read(s);

    double zp = accessor_zp.read(z);
    vec2 sfx = scale(ssurfp, -zp);
    accessor_sfp_x.write(s, sfx.x);
    accessor_sfp_y.write(s, sfx.y);
  }
}

void foreign_calc_force_tts(double alfa,
                            double ssmin,
                            intptr_t sstart,
                            intptr_t send,
                            PhysicalRegion rz[1],
                            PhysicalRegion rs[2])
{
  RegionAccessor<AT_SOA_8, double> accessor_zareap =
    rz[0].get_field_accessor(FIELD_ZAREAP).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zrp =
    rz[0].get_field_accessor(FIELD_ZRP).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zss =
    rz[0].get_field_accessor(FIELD_ZSS).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsz =
    rs[0].get_field_accessor(FIELD_MAPSZ).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_8, double> accessor_sareap =
    rs[0].get_field_accessor(FIELD_SAREAP).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_smf =
    rs[0].get_field_accessor(FIELD_SMF).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_ssurfp_x =
    rs[0].get_field_accessor(FIELD_SSURFP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_ssurfp_y =
    rs[0].get_field_accessor(FIELD_SSURFP_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_sft_x =
    rs[1].get_field_accessor(FIELD_SFT_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_sft_y =
    rs[1].get_field_accessor(FIELD_SFT_Y).typeify<double>().convert<AT_SOA_8>();

  for (intptr_t s = sstart; s < send; s++) {
    ptr_t z = accessor_mapsz.read(s);

    double zareap = accessor_zareap.read(z);
    double zrp = accessor_zrp.read(z);
    double zss = accessor_zss.read(z);
    double sareap = accessor_sareap.read(s);
    double smf = accessor_smf.read(s);
    vec2 ssurfp;
    ssurfp.x = accessor_ssurfp_x.read(s);
    ssurfp.y = accessor_ssurfp_y.read(s);

    double svfacinv = zareap / sareap;
    double srho = zrp * smf * svfacinv;
    double sstmp = fmax(zss, ssmin);
    sstmp = alfa * sstmp * sstmp;
    double sdp = sstmp * (srho - zrp);
    vec2 sqq = scale(ssurfp, -sdp);
    accessor_sft_x.write(s, sqq.x);
    accessor_sft_y.write(s, sqq.y);
  }
}

void foreign_qcs_zone_center_velocity(intptr_t sstart,
                                      intptr_t send,
                                      PhysicalRegion rz[2],
                                      PhysicalRegion rpp[1],
                                      PhysicalRegion rpg[1],
                                      PhysicalRegion rs[1])
{
  RegionAccessor<AT_SOA_8, intptr_t> accessor_znump =
    rz[0].get_field_accessor(FIELD_ZNUMP).typeify<intptr_t>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zuc_x =
    rz[1].get_field_accessor(FIELD_ZUC_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zuc_y =
    rz[1].get_field_accessor(FIELD_ZUC_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpp_pu_x =
    rpp[0].get_field_accessor(FIELD_PU_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpp_pu_y =
    rpp[0].get_field_accessor(FIELD_PU_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpg_pu_x =
    rpg[0].get_field_accessor(FIELD_PU_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpg_pu_y =
    rpg[0].get_field_accessor(FIELD_PU_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsz =
    rs[0].get_field_accessor(FIELD_MAPSZ).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp1_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP1_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp1_region =
    rs[0].get_field_accessor(FIELD_MAPSP1_REGION).typeify<uint32_t>().convert<AT_SOA_4>();

  for (intptr_t s = sstart; s < send; s++) {
    ptr_t z = accessor_mapsz.read(s);
    ptr_t p1_pointer = accessor_mapsp1_pointer.read(s);
    uint32_t p1_region = accessor_mapsp1_region.read(s);

    vec2 pu;
    switch (p1_region) {
    case 1:
      pu.x = accessor_rpp_pu_x.read(p1_pointer);
      pu.y = accessor_rpp_pu_y.read(p1_pointer);
      break;
    case 2:
      pu.x = accessor_rpg_pu_x.read(p1_pointer);
      pu.y = accessor_rpg_pu_y.read(p1_pointer);
      break;
    }

    intptr_t znump = accessor_znump.read(z);

    vec2 zuc = scale(pu, 1.0 / static_cast<double>(znump));
    accessor_zuc_x.write(z, accessor_zuc_x.read(z) + zuc.x);
    accessor_zuc_y.write(z, accessor_zuc_y.read(z) + zuc.y);
  }
}

void foreign_qcs_corner_divergence(intptr_t sstart,
                                   intptr_t send,
                                   PhysicalRegion rz[1],
                                   PhysicalRegion rpp[1],
                                   PhysicalRegion rpg[1],
                                   PhysicalRegion rs[2])
{
  RegionAccessor<AT_SOA_8, double> accessor_zxp_x =
    rz[0].get_field_accessor(FIELD_ZXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zxp_y =
    rz[0].get_field_accessor(FIELD_ZXP_Y).typeify<double>().convert<AT_SOA_8>();
 RegionAccessor<AT_SOA_8, double> accessor_zuc_x =
    rz[0].get_field_accessor(FIELD_ZUC_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zuc_y =
    rz[0].get_field_accessor(FIELD_ZUC_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpp_pxp_x =
    rpp[0].get_field_accessor(FIELD_PXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpp_pxp_y =
    rpp[0].get_field_accessor(FIELD_PXP_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpp_pu_x =
    rpp[0].get_field_accessor(FIELD_PU_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpp_pu_y =
    rpp[0].get_field_accessor(FIELD_PU_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpg_pxp_x =
    rpg[0].get_field_accessor(FIELD_PXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpg_pxp_y =
    rpg[0].get_field_accessor(FIELD_PXP_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpg_pu_x =
    rpg[0].get_field_accessor(FIELD_PU_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpg_pu_y =
    rpg[0].get_field_accessor(FIELD_PU_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsz =
    rs[0].get_field_accessor(FIELD_MAPSZ).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp1_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP1_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp1_region =
    rs[0].get_field_accessor(FIELD_MAPSP1_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp2_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP2_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp2_region =
    rs[0].get_field_accessor(FIELD_MAPSP2_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapss3 =
    rs[0].get_field_accessor(FIELD_MAPSS3).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_8, double> accessor_exp_x =
    rs[0].get_field_accessor(FIELD_EXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_exp_y =
    rs[0].get_field_accessor(FIELD_EXP_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_elen =
    rs[0].get_field_accessor(FIELD_ELEN).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_carea =
    rs[1].get_field_accessor(FIELD_CAREA).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_ccos =
    rs[1].get_field_accessor(FIELD_CCOS).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_cdiv =
    rs[1].get_field_accessor(FIELD_CDIV).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_cevol =
    rs[1].get_field_accessor(FIELD_CEVOL).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_cdu =
    rs[1].get_field_accessor(FIELD_CDU).typeify<double>().convert<AT_SOA_8>();

  for (intptr_t s2 = sstart; s2 < send; s2++) {
    // intptr_t c = s2;
    ptr_t s = accessor_mapss3.read(s2);
    ptr_t z = accessor_mapsz.read(s);
    ptr_t p_pointer = accessor_mapsp2_pointer.read(s);
    uint32_t p_region = accessor_mapsp2_region.read(s);
    ptr_t p1_pointer = accessor_mapsp1_pointer.read(s);
    uint32_t p1_region = accessor_mapsp1_region.read(s);
    ptr_t p2_pointer = accessor_mapsp2_pointer.read(s2);
    uint32_t p2_region = accessor_mapsp2_region.read(s2);
    // ptr_t e1 = s;
    // intptr_t e2 = s2;

    vec2 p_pu, p_pxp;
    switch (p_region) {
    case 1:
      p_pu.x = accessor_rpp_pu_x.read(p_pointer);
      p_pu.y = accessor_rpp_pu_y.read(p_pointer);
      p_pxp.x = accessor_rpp_pxp_x.read(p_pointer);
      p_pxp.y = accessor_rpp_pxp_y.read(p_pointer);
      break;
    case 2:
      p_pu.x = accessor_rpg_pu_x.read(p_pointer);
      p_pu.y = accessor_rpg_pu_y.read(p_pointer);
      p_pxp.x = accessor_rpg_pxp_x.read(p_pointer);
      p_pxp.y = accessor_rpg_pxp_y.read(p_pointer);
      break;
    }

    vec2 p1_pu;
    switch (p1_region) {
    case 1:
      p1_pu.x = accessor_rpp_pu_x.read(p1_pointer);
      p1_pu.y = accessor_rpp_pu_y.read(p1_pointer);
      break;
    case 2:
      p1_pu.x = accessor_rpg_pu_x.read(p1_pointer);
      p1_pu.y = accessor_rpg_pu_y.read(p1_pointer);
      break;
    }

    vec2 p2_pu;
    switch (p2_region) {
    case 1:
      p2_pu.x = accessor_rpp_pu_x.read(p2_pointer);
      p2_pu.y = accessor_rpp_pu_y.read(p2_pointer);
      break;
    case 2:
      p2_pu.x = accessor_rpg_pu_x.read(p2_pointer);
      p2_pu.y = accessor_rpg_pu_y.read(p2_pointer);
      break;
    }

    vec2 zuc;
    zuc.x = accessor_zuc_x.read(z);
    zuc.y = accessor_zuc_y.read(z);

    vec2 zxp;
    zxp.x = accessor_zxp_x.read(z);
    zxp.y = accessor_zxp_y.read(z);

    vec2 e1_exp;
    e1_exp.x = accessor_exp_x.read(s);
    e1_exp.y = accessor_exp_y.read(s);

    vec2 e2_exp;
    e2_exp.x = accessor_exp_x.read(s2);
    e2_exp.y = accessor_exp_y.read(s2);

    vec2 up0 = p_pu;
    vec2 xp0 = p_pxp;
    vec2 up1 = scale(add(p_pu, p2_pu), 0.5);
    vec2 xp1 = e2_exp;
    vec2 up2 = zuc;
    vec2 xp2 = zxp;
    vec2 up3 = scale(add(p1_pu, p_pu), 0.5);
    vec2 xp3 = e1_exp;

    double cvolume = 0.5 * cross(sub(xp2, xp0), sub(xp3, xp1));
    accessor_carea.write(s2, cvolume);

    double e1_elen = accessor_elen.read(s);
    double e2_elen = accessor_elen.read(s2);

    vec2 v1 = sub(xp3, xp0);
    vec2 v2 = sub(xp1, xp0);
    double de1 = e1_elen;
    double de2 = e2_elen;
    double minelen = fmin(de1, de2);
    if (minelen < 1e-12) {
      accessor_ccos.write(s2, 0.0);
    } else {
      accessor_ccos.write(s2, 4.0 * dot(v1, v2) / (de1 * de2));
    }

    double cdiv = (cross(sub(up2, up0), sub(xp3, xp1)) - cross(sub(up3, up1), sub(xp2, xp0)))/(2.0 * cvolume);
    accessor_cdiv.write(s2, cdiv);

    vec2 dxx1 = scale(sub(sub(add(xp1, xp2), xp0), xp3), 0.5);
    vec2 dxx2 = scale(sub(sub(add(xp2, xp3), xp0), xp1), 0.5);
    double dx1 = length(dxx1);
    double dx2 = length(dxx2);

    vec2 duav = scale(add(add(add(up0, up1), up2), up3), 0.25);

    double test1 = fabs(dot(dxx1, duav) * dx2);
    double test2 = fabs(dot(dxx2, duav) * dx1);
    double num = 0.0;
    double den = 0.0;
    if (test1 > test2) {
      num = dx1;
      den = dx2;
    } else {
      num = dx2;
      den = dx1;
    }
    double r = num / den;
    double evol = fmin(sqrt(4.0 * cvolume * r), 2.0 * minelen);

    double dv1 = length(sub(sub(add(up1, up2), up0), up3));
    double dv2 = length(sub(sub(add(up2, up3), up0), up1));
    double du = fmax(dv1, dv2);
    if (cdiv < 0.0) {
      accessor_cevol.write(s2, evol);
      accessor_cdu.write(s2, du);
    } else {
      accessor_cevol.write(s2, 0.0);
      accessor_cdu.write(s2, 0.0);
    }
  }
}

void foreign_qcs_qcn_force(double gamma,
                           double q1,
                           double q2,
                           intptr_t sstart,
                           intptr_t send,
                           PhysicalRegion rz[1],
                           PhysicalRegion rpp[1],
                           PhysicalRegion rpg[1],
                           PhysicalRegion rs[2])
{
  RegionAccessor<AT_SOA_8, double> accessor_zrp =
    rz[0].get_field_accessor(FIELD_ZRP).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zss =
    rz[0].get_field_accessor(FIELD_ZSS).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpp_pu_x =
    rpp[0].get_field_accessor(FIELD_PU_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpp_pu_y =
    rpp[0].get_field_accessor(FIELD_PU_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpg_pu_x =
    rpg[0].get_field_accessor(FIELD_PU_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpg_pu_y =
    rpg[0].get_field_accessor(FIELD_PU_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsz =
    rs[0].get_field_accessor(FIELD_MAPSZ).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp1_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP1_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp1_region =
    rs[0].get_field_accessor(FIELD_MAPSP1_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp2_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP2_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp2_region =
    rs[0].get_field_accessor(FIELD_MAPSP2_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapss3 =
    rs[0].get_field_accessor(FIELD_MAPSS3).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_8, double> accessor_elen =
    rs[0].get_field_accessor(FIELD_ELEN).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_cdiv =
    rs[0].get_field_accessor(FIELD_CDIV).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_cevol =
    rs[0].get_field_accessor(FIELD_CEVOL).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_cdu =
    rs[0].get_field_accessor(FIELD_CDU).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_cqe1_x =
    rs[1].get_field_accessor(FIELD_CQE1_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_cqe1_y =
    rs[1].get_field_accessor(FIELD_CQE1_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_cqe2_x =
    rs[1].get_field_accessor(FIELD_CQE2_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_cqe2_y =
    rs[1].get_field_accessor(FIELD_CQE2_Y).typeify<double>().convert<AT_SOA_8>();

  double gammap1 = gamma + 1.0;

  for (intptr_t s4 = sstart; s4 < send; s4++) {
    // intptr_t c = s4;
    ptr_t z = accessor_mapsz.read(s4);
    ptr_t s = accessor_mapss3.read(s4);
    ptr_t p_pointer = accessor_mapsp2_pointer.read(s);
    uint32_t p_region = accessor_mapsp2_region.read(s);
    ptr_t p1_pointer = accessor_mapsp1_pointer.read(s);
    uint32_t p1_region = accessor_mapsp1_region.read(s);
    ptr_t p2_pointer = accessor_mapsp2_pointer.read(s4);
    uint32_t p2_region = accessor_mapsp2_region.read(s4);
    // ptr_t e1 = s;
    // intptr_t e2 = s4;

    double zrp = accessor_zrp.read(z);
    double zss = accessor_zss.read(z);
    double cdiv = accessor_cdiv.read(s4);
    double cevol = accessor_cevol.read(s4);
    double cdu = accessor_cdu.read(s4);
    double e1_elen = accessor_elen.read(s);
    double e2_elen = accessor_elen.read(s4);

    vec2 p_pu;
    switch (p_region) {
    case 1:
      p_pu.x = accessor_rpp_pu_x.read(p_pointer);
      p_pu.y = accessor_rpp_pu_y.read(p_pointer);
      break;
    case 2:
      p_pu.x = accessor_rpg_pu_x.read(p_pointer);
      p_pu.y = accessor_rpg_pu_y.read(p_pointer);
      break;
    }

    vec2 p1_pu;
    switch (p1_region) {
    case 1:
      p1_pu.x = accessor_rpp_pu_x.read(p1_pointer);
      p1_pu.y = accessor_rpp_pu_y.read(p1_pointer);
      break;
    case 2:
      p1_pu.x = accessor_rpg_pu_x.read(p1_pointer);
      p1_pu.y = accessor_rpg_pu_y.read(p1_pointer);
      break;
    }

    vec2 p2_pu;
    switch (p2_region) {
    case 1:
      p2_pu.x = accessor_rpp_pu_x.read(p2_pointer);
      p2_pu.y = accessor_rpp_pu_y.read(p2_pointer);
      break;
    case 2:
      p2_pu.x = accessor_rpg_pu_x.read(p2_pointer);
      p2_pu.y = accessor_rpg_pu_y.read(p2_pointer);
      break;
    }

    double ztmp2 = q2 * 0.25 * gammap1 * cdu;
    double ztmp1 = q1 * zss;
    double zkur = ztmp2 + sqrt(ztmp2 * ztmp2 + ztmp1 * ztmp1);
    double rmu = zkur * zrp * cevol;
    if (cdiv > 0.0) {
      rmu = 0.0;
    }

    vec2 cqe1 = scale(sub(p_pu, p1_pu), rmu / e1_elen);
    vec2 cqe2 = scale(sub(p2_pu, p_pu), rmu / e2_elen);

    accessor_cqe1_x.write(s4, cqe1.x);
    accessor_cqe1_y.write(s4, cqe1.y);
    accessor_cqe2_x.write(s4, cqe2.x);
    accessor_cqe2_y.write(s4, cqe2.y);
  }
}

void foreign_qcs_force(intptr_t sstart,
                       intptr_t send,
                       PhysicalRegion rs[2])
{
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapss4 =
    rs[0].get_field_accessor(FIELD_MAPSS4).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_8, double> accessor_elen =
    rs[0].get_field_accessor(FIELD_ELEN).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_carea =
    rs[0].get_field_accessor(FIELD_CAREA).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_ccos =
    rs[0].get_field_accessor(FIELD_CCOS).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_cqe1_x =
    rs[0].get_field_accessor(FIELD_CQE1_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_cqe1_y =
    rs[0].get_field_accessor(FIELD_CQE1_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_cqe2_x =
    rs[0].get_field_accessor(FIELD_CQE2_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_cqe2_y =
    rs[0].get_field_accessor(FIELD_CQE2_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_sfq_x =
    rs[1].get_field_accessor(FIELD_SFQ_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_sfq_y =
    rs[1].get_field_accessor(FIELD_SFQ_Y).typeify<double>().convert<AT_SOA_8>();

  for (intptr_t s = sstart; s < send; s++) {
    // intptr_t c1 = s;
    ptr_t c2 = accessor_mapss4.read(s);
    // intptr_t e = s;

    double el = accessor_elen.read(s);
    double c1_carea = accessor_carea.read(s);
    double c2_carea = accessor_carea.read(c2);
    double c1_ccos = accessor_ccos.read(s);
    double c2_ccos = accessor_ccos.read(c2);

    vec2 c1_cqe1;
    c1_cqe1.x = accessor_cqe1_x.read(s);
    c1_cqe1.y = accessor_cqe1_y.read(s);

    vec2 c1_cqe2;
    c1_cqe2.x = accessor_cqe2_x.read(s);
    c1_cqe2.y = accessor_cqe2_y.read(s);

    vec2 c2_cqe1;
    c2_cqe1.x = accessor_cqe1_x.read(c2);
    c2_cqe1.y = accessor_cqe1_y.read(c2);

    vec2 c2_cqe2;
    c2_cqe2.x = accessor_cqe2_x.read(c2);
    c2_cqe2.y = accessor_cqe2_y.read(c2);

    double c1sin2 = 1.0 - c1_ccos * c1_ccos;
    double c1w = 0.0;
    double c1cos = 0.0;
    if (c1sin2 >= 1e-4) {
      c1w = c1_carea / c1sin2;
      c1cos = c1_ccos;
    }

    double c2sin2 = 1.0 - c2_ccos * c2_ccos;
    double c2w = 0.0;
    double c2cos = 0.0;
    if (c2sin2 >= 1e-4) {
      c2w = c2_carea / c2sin2;
      c2cos = c2_ccos;
    }

    vec2 sfq = scale(add(scale(add(c1_cqe2, scale(c1_cqe1, c1cos)), c1w),
                         scale(add(c2_cqe1, scale(c2_cqe2, c2cos)), c2w)),
                     1.0 / el);
    accessor_sfq_x.write(s, sfq.x);
    accessor_sfq_y.write(s, sfq.y);
  }
}

void foreign_sum_point_force(intptr_t sstart,
                             intptr_t send,
                             PhysicalRegion rpp[2],
                             PhysicalRegion rpg[2],
                             PhysicalRegion rs[1])
{
  RegionAccessor<AT_SOA_8, double> accessor_rpp_pf_y =
    rpp[0].get_field_accessor(FIELD_PF_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpp_pf_x =
    rpp[0].get_field_accessor(FIELD_PF_X).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpg_pf_y =
    rpg[0].get_accessor().typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpg_pf_x =
    rpg[1].get_accessor().typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsz =
    rs[0].get_field_accessor(FIELD_MAPSZ).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp1_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP1_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp1_region =
    rs[0].get_field_accessor(FIELD_MAPSP1_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapss3 =
    rs[0].get_field_accessor(FIELD_MAPSS3).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_8, double> accessor_sfp_x =
    rs[0].get_field_accessor(FIELD_SFP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_sfp_y =
    rs[0].get_field_accessor(FIELD_SFP_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_sfq_x =
    rs[0].get_field_accessor(FIELD_SFQ_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_sfq_y =
    rs[0].get_field_accessor(FIELD_SFQ_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_sft_x =
    rs[0].get_field_accessor(FIELD_SFT_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_sft_y =
    rs[0].get_field_accessor(FIELD_SFT_Y).typeify<double>().convert<AT_SOA_8>();

  for (intptr_t s = sstart; s < send; s++) {
    ptr_t z = accessor_mapsz.read(s);
    ptr_t p1_pointer = accessor_mapsp1_pointer.read(s);
    uint32_t p1_region = accessor_mapsp1_region.read(s);
    ptr_t s3 = accessor_mapss3.read(s);

    vec2 s_sfp;
    s_sfp.x = accessor_sfp_x.read(s);
    s_sfp.y = accessor_sfp_y.read(s);

    vec2 s3_sfp;
    s3_sfp.x = accessor_sfp_x.read(s3);
    s3_sfp.y = accessor_sfp_y.read(s3);

    vec2 s_sfq;
    s_sfq.x = accessor_sfq_x.read(s);
    s_sfq.y = accessor_sfq_y.read(s);

    vec2 s3_sfq;
    s3_sfq.x = accessor_sfq_x.read(s3);
    s3_sfq.y = accessor_sfq_y.read(s3);

    vec2 s_sft;
    s_sft.x = accessor_sft_x.read(s);
    s_sft.y = accessor_sft_y.read(s);

    vec2 s3_sft;
    s3_sft.x = accessor_sft_x.read(s3);
    s3_sft.y = accessor_sft_y.read(s3);

    vec2 f = sub(add(s_sfp, add(s_sfq, s_sft)),
                 add(s3_sfp, add(s3_sfq, s3_sft)));

    switch (p1_region) {
    case 1:
      accessor_rpp_pf_x.write(p1_pointer, accessor_rpp_pf_x.read(p1_pointer) + f.x);
      accessor_rpp_pf_y.write(p1_pointer, accessor_rpp_pf_y.read(p1_pointer) + f.y);
      break;
    case 2:
      accessor_rpg_pf_x.reduce<reduction_plus_double>(p1_pointer, f.x);
      accessor_rpg_pf_y.reduce<reduction_plus_double>(p1_pointer, f.y);
      break;
    }
  }
}

void foreign_calc_centers_full(intptr_t sstart,
                               intptr_t send,
                               PhysicalRegion rz[2],
                               PhysicalRegion rpp[1],
                               PhysicalRegion rpg[1],
                               PhysicalRegion rs[2])
{
  RegionAccessor<AT_SOA_8, intptr_t> accessor_znump =
    rz[0].get_field_accessor(FIELD_ZNUMP).typeify<intptr_t>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zx_x =
    rz[1].get_field_accessor(FIELD_ZX_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zx_y =
    rz[1].get_field_accessor(FIELD_ZX_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpp_px_x =
    rpp[0].get_field_accessor(FIELD_PX_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpp_px_y =
    rpp[0].get_field_accessor(FIELD_PX_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpg_px_x =
    rpg[0].get_field_accessor(FIELD_PX_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpg_px_y =
    rpg[0].get_field_accessor(FIELD_PX_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsz =
    rs[0].get_field_accessor(FIELD_MAPSZ).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp1_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP1_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp1_region =
    rs[0].get_field_accessor(FIELD_MAPSP1_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp2_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP2_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp2_region =
    rs[0].get_field_accessor(FIELD_MAPSP2_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_8, double> accessor_ex_x =
    rs[1].get_field_accessor(FIELD_EX_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_ex_y =
    rs[1].get_field_accessor(FIELD_EX_Y).typeify<double>().convert<AT_SOA_8>();

  for (intptr_t s = sstart; s < send; s++) {
    ptr_t z = accessor_mapsz.read(s);
    ptr_t p1_pointer = accessor_mapsp1_pointer.read(s);
    uint32_t p1_region = accessor_mapsp1_region.read(s);
    ptr_t p2_pointer = accessor_mapsp2_pointer.read(s);
    uint32_t p2_region = accessor_mapsp2_region.read(s);

    vec2 p1_px;
    switch (p1_region) {
    case 1:
      p1_px.x = accessor_rpp_px_x.read(p1_pointer);
      p1_px.y = accessor_rpp_px_y.read(p1_pointer);
      break;
    case 2:
      p1_px.x = accessor_rpg_px_x.read(p1_pointer);
      p1_px.y = accessor_rpg_px_y.read(p1_pointer);
      break;
    }

    vec2 p2_px;
    switch (p2_region) {
    case 1:
      p2_px.x = accessor_rpp_px_x.read(p2_pointer);
      p2_px.y = accessor_rpp_px_y.read(p2_pointer);
      break;
    case 2:
      p2_px.x = accessor_rpg_px_x.read(p2_pointer);
      p2_px.y = accessor_rpg_px_y.read(p2_pointer);
      break;
    }

    vec2 ex = scale(add(p1_px, p2_px), 0.5);
    accessor_ex_x.write(s, ex.x);
    accessor_ex_y.write(s, ex.y);

    double znump = static_cast<double>(accessor_znump.read(z));
    accessor_zx_x.write(z, accessor_zx_x.read(z) + p1_px.x/znump);
    accessor_zx_y.write(z, accessor_zx_y.read(z) + p1_px.y/znump);
  }
}

void foreign_calc_volumes_full(intptr_t sstart,
                               intptr_t send,
                               PhysicalRegion rz[2],
                               PhysicalRegion rpp[1],
                               PhysicalRegion rpg[1],
                               PhysicalRegion rs[2])
{
  RegionAccessor<AT_SOA_8, double> accessor_zx_x =
    rz[0].get_field_accessor(FIELD_ZX_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zx_y =
    rz[0].get_field_accessor(FIELD_ZX_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zarea =
    rz[1].get_field_accessor(FIELD_ZAREA).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zvol =
    rz[1].get_field_accessor(FIELD_ZVOL).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpp_px_x =
    rpp[0].get_field_accessor(FIELD_PX_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpp_px_y =
    rpp[0].get_field_accessor(FIELD_PX_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpg_px_x =
    rpg[0].get_field_accessor(FIELD_PX_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpg_px_y =
    rpg[0].get_field_accessor(FIELD_PX_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsz =
    rs[0].get_field_accessor(FIELD_MAPSZ).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp1_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP1_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp1_region =
    rs[0].get_field_accessor(FIELD_MAPSP1_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp2_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP2_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp2_region =
    rs[0].get_field_accessor(FIELD_MAPSP2_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_8, double> accessor_sarea =
    rs[1].get_field_accessor(FIELD_SAREA).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_svol =
    rs[1].get_field_accessor(FIELD_SVOL).typeify<double>().convert<AT_SOA_8>();

  for (intptr_t s = sstart; s < send; s++) {
    ptr_t z = accessor_mapsz.read(s);
    ptr_t p1_pointer = accessor_mapsp1_pointer.read(s);
    uint32_t p1_region = accessor_mapsp1_region.read(s);
    ptr_t p2_pointer = accessor_mapsp2_pointer.read(s);
    uint32_t p2_region = accessor_mapsp2_region.read(s);

    vec2 zx;
    zx.x = accessor_zx_x.read(z);
    zx.y = accessor_zx_y.read(z);

    vec2 p1_px;
    switch (p1_region) {
    case 1:
      p1_px.x = accessor_rpp_px_x.read(p1_pointer);
      p1_px.y = accessor_rpp_px_y.read(p1_pointer);
      break;
    case 2:
      p1_px.x = accessor_rpg_px_x.read(p1_pointer);
      p1_px.y = accessor_rpg_px_y.read(p1_pointer);
      break;
    }

    vec2 p2_px;
    switch (p2_region) {
    case 1:
      p2_px.x = accessor_rpp_px_x.read(p2_pointer);
      p2_px.y = accessor_rpp_px_y.read(p2_pointer);
      break;
    case 2:
      p2_px.x = accessor_rpg_px_x.read(p2_pointer);
      p2_px.y = accessor_rpg_px_y.read(p2_pointer);
      break;
    }

    double sa = 0.5 * cross(sub(p2_px, p1_px), sub(zx, p1_px));
    double sv = (1.0 / 3.0) * sa * (p1_px.x + p2_px.x + zx.x);

    accessor_sarea.write(s, sa);
    accessor_svol.write(s, sv);
    accessor_zarea.write(z, accessor_zarea.read(z) + sa);
    accessor_zvol.write(z, accessor_zvol.read(z) + sv);
    assert(sv > 0.0);
  }
}

void foreign_calc_work(double dt,
                       intptr_t sstart,
                       intptr_t send,
                       PhysicalRegion rz[1],
                       PhysicalRegion rpp[1],
                       PhysicalRegion rpg[1],
                       PhysicalRegion rs[1])
{
  RegionAccessor<AT_SOA_8, double> accessor_zw =
    rz[0].get_field_accessor(FIELD_ZW).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_zetot =
    rz[0].get_field_accessor(FIELD_ZETOT).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpp_pxp_x =
    rpp[0].get_field_accessor(FIELD_PXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpp_pxp_y =
    rpp[0].get_field_accessor(FIELD_PXP_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpp_pu0_x =
    rpp[0].get_field_accessor(FIELD_PU0_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpp_pu0_y =
    rpp[0].get_field_accessor(FIELD_PU0_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpp_pu_x =
    rpp[0].get_field_accessor(FIELD_PU_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpp_pu_y =
    rpp[0].get_field_accessor(FIELD_PU_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_8, double> accessor_rpg_pxp_x =
    rpg[0].get_field_accessor(FIELD_PXP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpg_pxp_y =
    rpg[0].get_field_accessor(FIELD_PXP_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpg_pu0_x =
    rpg[0].get_field_accessor(FIELD_PU0_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpg_pu0_y =
    rpg[0].get_field_accessor(FIELD_PU0_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpg_pu_x =
    rpg[0].get_field_accessor(FIELD_PU_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_rpg_pu_y =
    rpg[0].get_field_accessor(FIELD_PU_Y).typeify<double>().convert<AT_SOA_8>();

  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsz =
    rs[0].get_field_accessor(FIELD_MAPSZ).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp1_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP1_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp1_region =
    rs[0].get_field_accessor(FIELD_MAPSP1_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, ptr_t> accessor_mapsp2_pointer =
    rs[0].get_field_accessor(FIELD_MAPSP2_POINTER).typeify<ptr_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_4, uint32_t> accessor_mapsp2_region =
    rs[0].get_field_accessor(FIELD_MAPSP2_REGION).typeify<uint32_t>().convert<AT_SOA_4>();
  RegionAccessor<AT_SOA_8, double> accessor_sfp_x =
    rs[0].get_field_accessor(FIELD_SFP_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_sfp_y =
    rs[0].get_field_accessor(FIELD_SFP_Y).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_sfq_x =
    rs[0].get_field_accessor(FIELD_SFQ_X).typeify<double>().convert<AT_SOA_8>();
  RegionAccessor<AT_SOA_8, double> accessor_sfq_y =
    rs[0].get_field_accessor(FIELD_SFQ_Y).typeify<double>().convert<AT_SOA_8>();

  for (intptr_t s = sstart; s < send; s++) {
    ptr_t z = accessor_mapsz.read(s);
    ptr_t p1_pointer = accessor_mapsp1_pointer.read(s);
    uint32_t p1_region = accessor_mapsp1_region.read(s);
    ptr_t p2_pointer = accessor_mapsp2_pointer.read(s);
    uint32_t p2_region = accessor_mapsp2_region.read(s);

    vec2 sfp;
    sfp.x = accessor_sfp_x.read(s);
    sfp.y = accessor_sfp_y.read(s);

    vec2 sfq;
    sfq.x = accessor_sfq_x.read(s);
    sfq.y = accessor_sfq_y.read(s);

    vec2 p1_pu0;
    switch (p1_region) {
    case 1:
      p1_pu0.x = accessor_rpp_pu0_x.read(p1_pointer);
      p1_pu0.y = accessor_rpp_pu0_y.read(p1_pointer);
      break;
    case 2:
      p1_pu0.x = accessor_rpg_pu0_x.read(p1_pointer);
      p1_pu0.y = accessor_rpg_pu0_y.read(p1_pointer);
      break;
    }

    vec2 p1_pu;
    switch (p1_region) {
    case 1:
      p1_pu.x = accessor_rpp_pu_x.read(p1_pointer);
      p1_pu.y = accessor_rpp_pu_y.read(p1_pointer);
      break;
    case 2:
      p1_pu.x = accessor_rpg_pu_x.read(p1_pointer);
      p1_pu.y = accessor_rpg_pu_y.read(p1_pointer);
      break;
    }

    vec2 p2_pu0;
    switch (p2_region) {
    case 1:
      p2_pu0.x = accessor_rpp_pu0_x.read(p2_pointer);
      p2_pu0.y = accessor_rpp_pu0_y.read(p2_pointer);
      break;
    case 2:
      p2_pu0.x = accessor_rpg_pu0_x.read(p2_pointer);
      p2_pu0.y = accessor_rpg_pu0_y.read(p2_pointer);
      break;
    }

    vec2 p2_pu;
    switch (p2_region) {
    case 1:
      p2_pu.x = accessor_rpp_pu_x.read(p2_pointer);
      p2_pu.y = accessor_rpp_pu_y.read(p2_pointer);
      break;
    case 2:
      p2_pu.x = accessor_rpg_pu_x.read(p2_pointer);
      p2_pu.y = accessor_rpg_pu_y.read(p2_pointer);
      break;
    }

    double p1_pxp_x;
    switch (p1_region) {
    case 1:
      p1_pxp_x = accessor_rpp_pxp_x.read(p1_pointer);
      break;
    case 2:
      p1_pxp_x = accessor_rpg_pxp_x.read(p1_pointer);
      break;
    }

    double p2_pxp_x;
    switch (p2_region) {
    case 1:
      p2_pxp_x = accessor_rpp_pxp_x.read(p2_pointer);
      break;
    case 2:
      p2_pxp_x = accessor_rpg_pxp_x.read(p2_pointer);
      break;
    }

    vec2 sftot = add(sfp, sfq);
    double sd1 = dot(sftot, add(p1_pu0, p1_pu));
    double sd2 = dot(scale(sftot, -1.0), add(p2_pu0, p2_pu));
    double dwork = -0.5 * dt * (sd1 * p1_pxp_x + sd2 * p2_pxp_x);

    accessor_zetot.write(z, accessor_zetot.read(z) + dwork);
    accessor_zw.write(z, accessor_zw.read(z) + dwork);
  }
}

///
/// Vectors
///

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
  virtual void select_task_options(Task *task);
  virtual void select_task_variant(Task *task);
  virtual bool map_task(Task *task);
  virtual bool map_inline(Inline *inline_operation);
  virtual void notify_mapping_failed(const Mappable *mappable);
private:
  Color get_task_color_by_region(Task *task, LogicalRegion region);
private:
  std::map<Processor::Kind, std::vector<Processor> > all_processors;
  Memory local_sysmem;
  Memory local_regmem;
};

PennantMapper::PennantMapper(Machine *machine, HighLevelRuntime *rt, Processor local)
  : DefaultMapper(machine, rt, local)
{
  const std::set<Processor> &procs = machine->get_all_processors();
  for (std::set<Processor>::const_iterator it = procs.begin();
       it != procs.end(); it++) {
    Processor::Kind kind = machine->get_processor_kind(*it);
    all_processors[kind].push_back(*it);
  }

  local_sysmem =
    machine_interface.find_memory_kind(local_proc, Memory::SYSTEM_MEM);
  local_regmem =
    machine_interface.find_memory_kind(local_proc, Memory::REGDMA_MEM);
  if(!local_regmem.exists()) {
    local_regmem = local_sysmem;
  }
}

void PennantMapper::select_task_options(Task *task)
{
  switch (task->task_id) {
  case TASK_INIT_POINTERS:
  case TASK_INIT_MESH_ZONES:
  case TASK_INIT_SIDE_FRACS:
  case TASK_INIT_HYDRO:
  case TASK_INIT_RADIAL_VELOCITY:
  case TASK_INIT_STEP_POINTS:
  case TASK_ADV_POS_HALF:
  case TASK_INIT_STEP_ZONES:
  case TASK_CALC_CENTERS:
  case TASK_CALC_VOLUMES:
  case TASK_CALC_SURFACE_VECS:
  case TASK_CALC_EDGE_LEN:
  case TASK_CALC_CHAR_LEN:
  case TASK_CALC_RHO_HALF:
  case TASK_SUM_POINT_MASS:
  case TASK_CALC_STATE_AT_HALF:
  case TASK_CALC_FORCE_PGAS:
  case TASK_CALC_FORCE_TTS:
  case TASK_QCS_ZONE_CENTER_VELOCITY:
  case TASK_QCS_CORNER_DIVERGENCE:
  case TASK_QCS_QCN_FORCE:
  case TASK_QCS_FORCE:
  case TASK_CALC_FORCE_QCS:
  case TASK_SUM_POINT_FORCE:
  case TASK_APPLY_BOUNDARY_CONDITIONS:
  case TASK_CALC_ACCEL:
  case TASK_ADV_POS_FULL:
  case TASK_CALC_CENTERS_FULL:
  case TASK_CALC_VOLUMES_FULL:
  case TASK_CALC_WORK:
  case TASK_CALC_WORK_RATE:
  case TASK_CALC_ENERGY:
  case TASK_CALC_RHO_FULL:
  case TASK_CALC_DT_COURANT:
  case TASK_CALC_DT_VOLUME:
  case TASK_CALC_DT_HYDRO:
    {
      assert(task->regions.size() >= 1);
      LogicalRegion region = task->regions[0].region;
      Color color = get_task_color_by_region(task, region);

      // Task options:
      task->inline_task = false;
      task->spawn_task = false;
      task->map_locally = false;
      task->profile_task = false;

      // Processor (round robin by piece of graph):
      std::vector<Processor> &procs = all_processors[Processor::LOC_PROC];
      task->target_proc = procs[color % procs.size()];
    }
    return;
  default:
    DefaultMapper::select_task_options(task);
    return;
  }
}

void PennantMapper::select_task_variant(Task *task)
{
  // Use the SOA variant for all tasks.
  task->selected_variant = VARIANT_SOA;

  std::vector<RegionRequirement> &regions = task->regions;
  for (std::vector<RegionRequirement>::iterator it = regions.begin();
        it != regions.end(); it++) {
    RegionRequirement &req = *it;

    // Select SOA layout for all regions.
    req.blocking_factor = req.max_blocking_factor;
  }
}

bool PennantMapper::map_task(Task *task)
{
  assert(task->target_proc == local_proc);

  std::vector<RegionRequirement> &regions = task->regions;
  for (std::vector<RegionRequirement>::iterator it = regions.begin();
        it != regions.end(); it++) {
    RegionRequirement &req = *it;

    // Region options:
    req.virtual_map = false;
    req.enable_WAR_optimization = false;
    req.reduction_list = false;

    // Place all regions in local system memory.
    req.target_ranking.push_back(local_sysmem);
  }

  return false;
}

bool PennantMapper::map_inline(Inline *inline_operation)
{
  RegionRequirement &req = inline_operation->requirement;

  // Region options:
  req.virtual_map = false;
  req.enable_WAR_optimization = false;
  req.reduction_list = false;
  req.blocking_factor = req.max_blocking_factor;

  // Place all regions in global memory.
  req.target_ranking.push_back(local_sysmem);

  log_mapper.debug(
    "inline mapping region (%d,%d,%d) target ranking front %d (size %lu)",
    req.region.get_index_space().id,
    req.region.get_field_space().get_id(),
    req.region.get_tree_id(),
    req.target_ranking[0].id,
    req.target_ranking.size());

  return false;
}

void PennantMapper::notify_mapping_failed(const Mappable *mappable)
{
  switch (mappable->get_mappable_kind()) {
  case Mappable::TASK_MAPPABLE:
    {
      log_mapper.warning("mapping failed on task");
      break;
    }
  case Mappable::COPY_MAPPABLE:
    {
      log_mapper.warning("mapping failed on copy");
      break;
    }
  case Mappable::INLINE_MAPPABLE:
    {
      Inline *_inline = mappable->as_mappable_inline();
      RegionRequirement &req = _inline->requirement;
      LogicalRegion region = req.region;
      log_mapper.warning(
        "mapping %s on inline region (%d,%d,%d) memory %d",
        (req.mapping_failed ? "failed" : "succeeded"),
        region.get_index_space().id,
        region.get_field_space().get_id(),
        region.get_tree_id(),
        req.selected_memory.id);
      break;
    }
  case Mappable::ACQUIRE_MAPPABLE:
    {
      log_mapper.warning("mapping failed on acquire");
      break;
    }
  case Mappable::RELEASE_MAPPABLE:
    {
      log_mapper.warning("mapping failed on release");
      break;
    }
  }
  assert(0 && "mapping failed");
}

Color PennantMapper::get_task_color_by_region(Task *task, LogicalRegion region)
{
  return get_logical_region_color(region);
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
