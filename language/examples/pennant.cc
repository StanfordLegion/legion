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

#include "pennant.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <map>
#include <vector>

#include "mappers/default_mapper.h"

#include <sys/time.h>
#include <sys/resource.h>
void print_rusage(const char *message)
{
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0) return;
  printf("%s: %ld MB\n", message, usage.ru_maxrss / 1024);
}

using namespace Legion;
using namespace Legion::Mapping;

struct config {
  int64_t np;
  int64_t nz;
  int64_t nzx;
  int64_t nzy;
  double lenx;
  double leny;
  int64_t numpcx;
  int64_t numpcy;
  int64_t npieces;
  int64_t meshtype;
  bool compact;
  int64_t stripsize;
  int64_t spansize;
};

enum {
  MESH_PIE = 0,
  MESH_RECT = 1,
  MESH_HEX = 2,
};

///
/// Mesh Generator
///

/*
 * Some portions of the following code are derived from the
 * open-source release of PENNANT:
 *
 * https://github.com/losalamos/PENNANT
 */

static void generate_mesh_rect(config &conf,
                               std::vector<double> &pointpos_x,
                               std::vector<double> &pointpos_y,
                               std::vector<int64_t> &pointcolors,
                               std::map<int64_t, std::vector<int64_t> > &pointmcolors,
                               std::vector<int64_t> &zonestart,
                               std::vector<int64_t> &zonesize,
                               std::vector<int64_t> &zonepoints,
                               std::vector<int64_t> &zonecolors,
                               std::vector<int64_t> &zxbounds,
                               std::vector<int64_t> &zybounds)
{
  int64_t &nz = conf.nz;
  int64_t &np = conf.np;

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
        std::vector<int64_t> &pmc = pointmcolors[p];
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
                              std::vector<int64_t> &pointcolors,
                              std::map<int64_t, std::vector<int64_t> > &pointmcolors,
                              std::vector<int64_t> &zonestart,
                              std::vector<int64_t> &zonesize,
                              std::vector<int64_t> &zonepoints,
                              std::vector<int64_t> &zonecolors,
                              std::vector<int64_t> &zxbounds,
                              std::vector<int64_t> &zybounds)
{
  int64_t &nz = conf.nz;
  int64_t &np = conf.np;

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
        std::vector<int64_t> &pmc = pointmcolors[0];
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
        std::vector<int64_t> &pmc = pointmcolors[p];
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
                              std::vector<int64_t> &pointcolors,
                              std::map<int64_t, std::vector<int64_t> > &pointmcolors,
                              std::vector<int64_t> &zonestart,
                              std::vector<int64_t> &zonesize,
                              std::vector<int64_t> &zonepoints,
                              std::vector<int64_t> &zonecolors,
                              std::vector<int64_t> &zxbounds,
                              std::vector<int64_t> &zybounds)
{
  int64_t &nz = conf.nz;
  int64_t &np = conf.np;

  nz = conf.nzx * conf.nzy;
  const int npx = conf.nzx + 1;
  const int npy = conf.nzy + 1;

  // generate point coordinates
  pointpos_x.resize(2 * npx * npy);  // upper bound
  pointpos_y.resize(2 * npx * npy);  // upper bound
  double dx = conf.lenx / (double) (conf.nzx - 1);
  double dy = conf.leny / (double) (conf.nzy - 1);

  std::vector<int64_t> pbase(npy);
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
          std::vector<int64_t> &pmc = pointmcolors[p1];
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
          std::vector<int64_t> &pmc1 = pointmcolors[p1];
          std::vector<int64_t> &pmc2 = pointmcolors[p2];
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
      std::vector<int64_t> v(6);
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
                          std::vector<int64_t> &pointcolors,
                          std::map<int64_t, std::vector<int64_t> > &pointmcolors,
                          std::vector<int64_t> &zonestart,
                          std::vector<int64_t> &zonesize,
                          std::vector<int64_t> &zonepoints,
                          std::vector<int64_t> &zonecolors)
{
  // Do calculations common to all mesh types:
  std::vector<int64_t> zxbounds;
  std::vector<int64_t> zybounds;
  if (conf.numpcx <= 0 || conf.numpcy <= 0) {
    calc_mesh_num_pieces(conf);
  }
  zxbounds.push_back(-1);
  for (int pcx = 1; pcx < conf.numpcx; ++pcx)
    zxbounds.push_back(pcx * conf.nzx / conf.numpcx);
  zxbounds.push_back(conf.nzx + 1);
  zybounds.push_back(-1);
  for (int pcy = 1; pcy < conf.numpcy; ++pcy)
    zybounds.push_back(pcy * conf.nzy / conf.numpcy);
  zybounds.push_back(0x7FFFFFFF);

  // Mesh type-specific calculations:
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
}

static void sort_zones_by_color(const config &conf,
                                const std::vector<int64_t> &zonecolors,
                                std::vector<int64_t> &zones_inverse_map,
                                std::vector<int64_t> &zones_map)
{
  // Sort zones by color.
  assert(int64_t(zonecolors.size()) == conf.nz);
  std::map<int64_t, std::vector<int64_t> > zones_by_color;
  for (int64_t z = 0; z < conf.nz; z++) {
    zones_by_color[zonecolors[z]].push_back(z);
  }
  for (int64_t c = 0; c < conf.npieces; c++) {
    std::vector<int64_t> &zones = zones_by_color[c];
    for (std::vector<int64_t>::iterator zt = zones.begin(), ze = zones.end();
         zt != ze; ++zt) {
      int64_t z = *zt;
      assert(zones_map[z] == -1ll);
      zones_map[z] = zones_inverse_map.size();
      zones_inverse_map.push_back(z);
    }
  }
}

static std::set<int64_t> zone_point_set(
  int64_t z,
  const std::vector<int64_t> &zonestart,
  const std::vector<int64_t> &zonesize,
  const std::vector<int64_t> &zonepoints)
{
  std::set<int64_t> points;
  for (int64_t z_start = zonestart[z], z_size = zonesize[z],
         z_point = z_start; z_point < z_start + z_size; z_point++) {
    points.insert(zonepoints[z_point]);
  }
  return points;
}

static void sort_zones_by_color_strip(const config &conf,
                                      const std::vector<double> &pointpos_x,
                                      const std::vector<double> &pointpos_y,
                                      const std::vector<int64_t> &zonestart,
                                      const std::vector<int64_t> &zonesize,
                                      const std::vector<int64_t> &zonepoints,
                                      const std::vector<int64_t> &zonecolors,
                                      std::vector<int64_t> &zones_inverse_map,
                                      std::vector<int64_t> &zones_map)
{
  int64_t stripsize = conf.stripsize;

  // Sort zones by color. Within each color, make strips of zones of
  // size stripsize.

  std::vector<std::vector<int64_t> > strips;
  assert(int64_t(zonecolors.size()) == conf.nz);
  for (int64_t c = 0; c < conf.npieces; c++) {
    strips.assign(strips.size(), std::vector<int64_t>());

    int64_t z_start = -1ll;
    std::set<int64_t> z_start_points;
    for (int64_t z = 0; z < conf.nz; z++) {
      if (zonecolors[z] == c) {
        if (z_start >= 0) {
          if (z > z_start + 1) {
            bool intersect = false;
            for (int64_t z_start = zonestart[z], z_size = zonesize[z],
                   z_point = z_start; z_point < z_start + z_size; z_point++) {
              if (z_start_points.count(zonepoints[z_point])) {
                intersect = true;
                break;
              }
            }
            if (intersect) {
              z_start = z;
              z_start_points =
                zone_point_set(z_start, zonestart, zonesize, zonepoints);
            }
          }
        } else {
          z_start = z;
          z_start_points =
            zone_point_set(z_start, zonestart, zonesize, zonepoints);
        }

        int64_t strip = (z - z_start)/stripsize;
        if (strip + 1 > int64_t(strips.size())) {
          strips.resize(strip+1);
        }
        strips[strip].push_back(z);
      }
    }

    for (std::vector<std::vector<int64_t> >::iterator st = strips.begin(),
           se = strips.end(); st != se; ++st) {
      for (std::vector<int64_t>::iterator zt = st->begin(), ze = st->end();
           zt != ze; ++zt) {
        int64_t z = *zt;
        assert(zones_map[z] == -1ll);
        zones_map[z] = zones_inverse_map.size();
        zones_inverse_map.push_back(z);
      }
    }
  }
}

static void sort_points_by_color(
  const config &conf,
  const std::vector<int64_t> &pointcolors,
  std::map<int64_t, std::vector<int64_t> > &pointmcolors,
  std::vector<int64_t> &points_inverse_map,
  std::vector<int64_t> &points_map)
{
  // Sort points by color; sort multi-color points by first color.
  assert(int64_t(pointcolors.size()) == conf.np);
  std::map<int64_t, std::vector<int64_t> > points_by_color;
  std::map<int64_t, std::vector<int64_t> > points_by_multicolor;
  for (int64_t p = 0; p < conf.np; p++) {
    if (pointcolors[p] == MULTICOLOR) {
      points_by_multicolor[pointmcolors[p][0]].push_back(p);
    } else {
      points_by_color[pointcolors[p]].push_back(p);
    }
  }
  for (int64_t c = 0; c < conf.npieces; c++) {
    std::vector<int64_t> &points = points_by_multicolor[c];
    for (std::vector<int64_t>::iterator pt = points.begin(), pe = points.end();
         pt != pe; ++pt) {
      int64_t p = *pt;
      assert(points_map[p] == -1ll);
      points_map[p] = points_inverse_map.size();
      points_inverse_map.push_back(p);
    }
  }
  for (int64_t c = 0; c < conf.npieces; c++) {
    std::vector<int64_t> &points = points_by_color[c];
    for (std::vector<int64_t>::iterator pt = points.begin(), pe = points.end();
         pt != pe; ++pt) {
      int64_t p = *pt;
      assert(points_map[p] == -1ll);
      points_map[p] = points_inverse_map.size();
      points_inverse_map.push_back(p);
    }
  }
}

static void compact_mesh(const config &conf,
                         std::vector<double> &pointpos_x,
                         std::vector<double> &pointpos_y,
                         std::vector<int64_t> &pointcolors,
                         std::map<int64_t, std::vector<int64_t> > &pointmcolors,
                         std::vector<int64_t> &zonestart,
                         std::vector<int64_t> &zonesize,
                         std::vector<int64_t> &zonepoints,
                         std::vector<int64_t> &zonecolors)
{
  // This stage is responsible for compacting the mesh so that each of
  // the pieces is dense (in the global coordinate space). This
  // involves sorting the various elements by color and then rewriting
  // the various pointers to be internally consistent again.

  // Sort zones by color.
  std::vector<int64_t> zones_inverse_map;
  std::vector<int64_t> zones_map(conf.nz, -1ll);
  if (conf.stripsize > 0) {
    sort_zones_by_color_strip(conf, pointpos_x, pointpos_y,
                              zonestart, zonesize, zonepoints, zonecolors,
                              zones_inverse_map, zones_map);
  } else {
    sort_zones_by_color(conf, zonecolors, zones_inverse_map, zones_map);
  }
  assert(int64_t(zones_inverse_map.size()) == conf.nz);

  // Sort points by color; sort multi-color points by first color.
  std::vector<int64_t> points_inverse_map;
  std::vector<int64_t> points_map(conf.np, -1ll);
  sort_points_by_color(conf, pointcolors, pointmcolors,
                       points_inverse_map, points_map);
  assert(int64_t(points_inverse_map.size()) == conf.np);

  // Various sanity checks.
#if 0
  for (int64_t z = 0; z < conf.nz; z++) {
    printf("zone old %ld new %ld color %ld\n", z, zones_map[z], zonecolors[z]);
  }

  printf("\n");

  for (int64_t newz = 0; newz < conf.nz; newz++) {
    int64_t oldz = zones_inverse_map[newz];
    printf("zone new %ld old %ld color %ld\n", newz, oldz, zonecolors[oldz]);
  }

  printf("\n");

  for (int64_t p = 0; p < conf.np; p++) {
    printf("point old %ld new %ld color %ld\n", p, points_map[p], pointcolors[p]);
  }

  printf("\n");

  for (int64_t newp = 0; newp < conf.np; newp++) {
    int64_t oldp = points_inverse_map[newp];
    printf("point new %ld old %ld color %ld\n", newp, oldp, pointcolors[oldp]);
  }
#endif

  // Project zones through the zones map.
  {
    std::vector<int64_t> old_zonestart = zonestart;
    for (int64_t newz = 0; newz < conf.nz; newz++) {
      int64_t oldz = zones_inverse_map[newz];
      zonestart[newz] = old_zonestart[oldz];
    }
  }
  {
    std::vector<int64_t> old_zonesize = zonesize;
    for (int64_t newz = 0; newz < conf.nz; newz++) {
      int64_t oldz = zones_inverse_map[newz];
      zonesize[newz] = old_zonesize[oldz];
    }
  }
  {
    std::vector<int64_t> old_zonepoints = zonepoints;
    int64_t nzp = zonepoints.size();
    for (int64_t zp = 0; zp < nzp; zp++) {
      zonepoints[zp] = points_map[old_zonepoints[zp]];
    }
  }
  {
    std::vector<int64_t> old_zonecolors = zonecolors;
    for (int64_t newz = 0; newz < conf.nz; newz++) {
      int64_t oldz = zones_inverse_map[newz];
      zonecolors[newz] = old_zonecolors[oldz];
    }
  }

  // Project points through the points map.
  {
    std::vector<double> old_pointpos_x = pointpos_x;
    for (int64_t newp = 0; newp < conf.np; newp++) {
      int64_t oldp = points_inverse_map[newp];
      pointpos_x[newp] = old_pointpos_x[oldp];
    }
  }
  {
    std::vector<double> old_pointpos_y = pointpos_y;
    for (int64_t newp = 0; newp < conf.np; newp++) {
      int64_t oldp = points_inverse_map[newp];
      pointpos_y[newp] = old_pointpos_y[oldp];
    }
  }
  {
    std::vector<int64_t> old_pointcolors = pointcolors;
    for (int64_t newp = 0; newp < conf.np; newp++) {
      int64_t oldp = points_inverse_map[newp];
      pointcolors[newp] = old_pointcolors[oldp];
    }
  }
  {
    std::map<int64_t, std::vector<int64_t> > old_pointmcolors = pointmcolors;
    for (int64_t newp = 0; newp < conf.np; newp++) {
      int64_t oldp = points_inverse_map[newp];
      pointmcolors[newp] = old_pointmcolors[oldp];
    }
  }

}

static void
color_spans(const config &conf,
            const std::vector<double> &pointpos_x,
            const std::vector<double> &pointpos_y,
            const std::vector<int64_t> &pointcolors,
            std::map<int64_t, std::vector<int64_t> > &pointmcolors,
            const std::vector<int64_t> &zonestart,
            const std::vector<int64_t> &zonesize,
            const std::vector<int64_t> &zonepoints,
            const std::vector<int64_t> &zonecolors,
            std::vector<int64_t> &zonespancolors_vec,
            std::vector<int64_t> &pointspancolors_vec,
            int64_t &nspans_zones,
            int64_t &nspans_points)
{
  {
    // Compute zone spans.
    std::vector<std::vector<std::vector<int64_t> > > spans(conf.npieces);
    std::vector<int64_t> span_size(conf.npieces, conf.spansize);
    for (int64_t z = 0; z < conf.nz; z++) {
      int64_t c = zonecolors[z];
      if (span_size[c] + zonesize[c] > conf.spansize) {
        spans[c].resize(spans[c].size() + 1);
        span_size[c] = 0;
      }
      spans[c][spans[c].size() - 1].push_back(z);
      span_size[c] += zonesize[z];
    }

    // Color zones by span.
    nspans_zones = 0;
    zonespancolors_vec.assign(conf.nz, -1ll);
    for (int64_t c = 0; c < conf.npieces; c++) {
      std::vector<std::vector<int64_t> > &color_spans = spans[c];
      int64_t nspans = color_spans.size();
      nspans_zones = std::max(nspans_zones, nspans);
      for (int64_t ispan = 0; ispan < nspans; ispan++) {
        std::vector<int64_t> &span = color_spans[ispan];
        for (std::vector<int64_t>::iterator zt = span.begin(), ze = span.end();
             zt != ze; ++zt) {
          int64_t z = *zt;
          zonespancolors_vec[z] = ispan;
        }
      }
    }
    for (int64_t z = 0; z < conf.nz; z++) {
      assert(zonespancolors_vec[z] != -1ll);
    }
  }

  {
    // Compute point spans.
    std::vector<std::vector<std::vector<int64_t> > > spans(conf.npieces);
    std::vector<std::vector<std::vector<int64_t> > > mspans(conf.npieces);
    std::vector<int64_t> span_size(conf.npieces, conf.spansize);
    std::vector<int64_t> mspan_size(conf.npieces, conf.spansize);
    for (int64_t p = 0; p < conf.np; p++) {
      int64_t c = pointcolors[p];
      if (c != MULTICOLOR) {
        if (span_size[c] >= conf.spansize) {
          spans[c].resize(spans[c].size() + 1);
          span_size[c] = 0;
        }
        spans[c][spans[c].size() - 1].push_back(p);
        span_size[c]++;
      } else {
        c = pointmcolors[p][0];
        if (mspan_size[c] >= conf.spansize) {
          mspans[c].resize(mspans[c].size() + 1);
          mspan_size[c] = 0;
        }
        mspans[c][mspans[c].size() - 1].push_back(p);
        mspan_size[c]++;
      }
    }

    // Color points by span.
    nspans_points = 0;
    pointspancolors_vec.assign(conf.np, -1ll);
    for (int64_t c = 0; c < conf.npieces; c++) {
      std::vector<std::vector<int64_t> > &color_spans = spans[c];
      int64_t nspans = color_spans.size();
      nspans_points = std::max(nspans_points, nspans);
      for (int64_t ispan = 0; ispan < nspans; ispan++) {
        std::vector<int64_t> &span = color_spans[ispan];
        for (std::vector<int64_t>::iterator pt = span.begin(), pe = span.end();
             pt != pe; ++pt) {
          int64_t p = *pt;
          pointspancolors_vec[p] = ispan;
        }
      }
    }
    for (int64_t c = 0; c < conf.npieces; c++) {
      std::vector<std::vector<int64_t> > &color_spans = mspans[c];
      int64_t nspans = color_spans.size();
      nspans_points = std::max(nspans_points, nspans);
      for (int64_t ispan = 0; ispan < nspans; ispan++) {
        std::vector<int64_t> &span = color_spans[ispan];
        for (std::vector<int64_t>::iterator pt = span.begin(), pe = span.end();
             pt != pe; ++pt) {
          int64_t p = *pt;
          pointspancolors_vec[p] = ispan;
        }
      }
    }
    for (int64_t p = 0; p < conf.np; p++) {
      assert(pointspancolors_vec[p] != -1ll);
    }
  }
}

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
  int64_t *nspans_points)
{
  config conf;
  conf.np = conf_np;
  conf.nz = conf_nz;
  conf.nzx = conf_nzx;
  conf.nzy = conf_nzy;
  conf.lenx = conf_lenx;
  conf.leny = conf_leny;
  conf.numpcx = conf_numpcx;
  conf.numpcy = conf_numpcy;
  conf.npieces = conf_npieces;
  conf.meshtype = conf_meshtype;
  conf.compact = conf_compact;
  conf.stripsize = conf_stripsize;
  conf.spansize = conf_spansize;

  std::vector<double> pointpos_x_vec;
  std::vector<double> pointpos_y_vec;
  std::vector<int64_t> pointcolors_vec;
  std::map<int64_t, std::vector<int64_t> > pointmcolors_map;
  std::vector<int64_t> zonestart_vec;
  std::vector<int64_t> zonesize_vec;
  std::vector<int64_t> zonepoints_vec;
  std::vector<int64_t> zonecolors_vec;

  generate_mesh(conf,
                pointpos_x_vec,
                pointpos_y_vec,
                pointcolors_vec,
                pointmcolors_map,
                zonestart_vec,
                zonesize_vec,
                zonepoints_vec,
                zonecolors_vec);

  if (conf.compact) {
    compact_mesh(conf,
                 pointpos_x_vec,
                 pointpos_y_vec,
                 pointcolors_vec,
                 pointmcolors_map,
                 zonestart_vec,
                 zonesize_vec,
                 zonepoints_vec,
                 zonecolors_vec);
  }

  std::vector<int64_t> zonespancolors_vec;
  std::vector<int64_t> pointspancolors_vec;

  color_spans(conf,
              pointpos_x_vec,
              pointpos_y_vec,
              pointcolors_vec,
              pointmcolors_map,
              zonestart_vec,
              zonesize_vec,
              zonepoints_vec,
              zonecolors_vec,
              zonespancolors_vec,
              pointspancolors_vec,
              *nspans_zones,
              *nspans_points);

  int64_t color_words = int64_t(ceil(conf_npieces/64.0));

  assert(pointpos_x_vec.size() <= *pointpos_x_size);
  assert(pointpos_y_vec.size() <= *pointpos_y_size);
  assert(pointcolors_vec.size() <= *pointcolors_size);
  assert(pointcolors_vec.size()*color_words <= *pointmcolors_size);
  assert(pointspancolors_vec.size() <= *pointspancolors_size);
  assert(zonestart_vec.size() <= *zonestart_size);
  assert(zonesize_vec.size() <= *zonesize_size);
  assert(zonepoints_vec.size() <= *zonepoints_size);
  assert(zonecolors_vec.size() <= *zonecolors_size);
  assert(zonespancolors_vec.size() <= *zonespancolors_size);

  memcpy(pointpos_x, pointpos_x_vec.data(), pointpos_x_vec.size()*sizeof(double));
  memcpy(pointpos_y, pointpos_y_vec.data(), pointpos_y_vec.size()*sizeof(double));
  memcpy(pointcolors, pointcolors_vec.data(), pointcolors_vec.size()*sizeof(int64_t));
  memcpy(pointspancolors, pointspancolors_vec.data(), pointspancolors_vec.size()*sizeof(int64_t));
  memcpy(zonestart, zonestart_vec.data(), zonestart_vec.size()*sizeof(int64_t));
  memcpy(zonesize, zonesize_vec.data(), zonesize_vec.size()*sizeof(int64_t));
  memcpy(zonepoints, zonepoints_vec.data(), zonepoints_vec.size()*sizeof(int64_t));
  memcpy(zonecolors, zonecolors_vec.data(), zonecolors_vec.size()*sizeof(int64_t));
  memcpy(zonespancolors, zonespancolors_vec.data(), zonespancolors_vec.size()*sizeof(int64_t));

  memset(pointmcolors, 0, (*pointmcolors_size)*sizeof(uint64_t));
  for (std::map<int64_t, std::vector<int64_t> >::iterator it = pointmcolors_map.begin(),
         ie = pointmcolors_map.end(); it != ie; ++it) {
    int64_t p = it->first;
    for (std::vector<int64_t>::iterator ct = it->second.begin(),
           ce = it->second.end(); ct != ce; ++ct) {
      int64_t word = (*ct) / 64.0;
      int64_t bit = (*ct) % 64;
      pointmcolors[p + word] |= (1 << bit);
    }
  }

  *pointpos_x_size = pointpos_x_vec.size();
  *pointpos_y_size = pointpos_y_vec.size();
  *pointcolors_size = pointcolors_vec.size();
  *pointmcolors_size = pointcolors_vec.size()*color_words;
  *pointspancolors_size = pointspancolors_vec.size();
  *zonestart_size = zonestart_vec.size();
  *zonesize_size = zonesize_vec.size();
  *zonepoints_size = zonepoints_vec.size();
  *zonecolors_size = zonecolors_vec.size();
  *zonespancolors_size = zonespancolors_vec.size();
}

///
/// Mapper
///

#define SPMD_SHARD_USE_IO_PROC 0

static LegionRuntime::Logger::Category log_pennant("pennant");

class PennantMapper : public DefaultMapper
{
public:
  PennantMapper(MapperRuntime *rt, Machine machine, Processor local,
                const char *mapper_name,
                std::vector<Processor>* procs_list,
                std::vector<Memory>* sysmems_list,
                std::map<Memory, std::vector<Processor> >* sysmem_local_procs,
#if SPMD_SHARD_USE_IO_PROC
                std::map<Memory, std::vector<Processor> >* sysmem_local_io_procs,
#endif
                std::map<Processor, Memory>* proc_sysmems,
                std::map<Processor, Memory>* proc_regmems);
  virtual void default_policy_rank_processor_kinds(
                                    MapperContext ctx, const Task &task,
                                    std::vector<Processor::Kind> &ranking);
  virtual Processor default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task);
  virtual void default_policy_select_target_processors(
                                    MapperContext ctx,
                                    const Task &task,
                                    std::vector<Processor> &target_procs);
  virtual LogicalRegion default_policy_select_instance_region(
                                    MapperContext ctx, Memory target_memory,
                                    const RegionRequirement &req,
                                    const LayoutConstraintSet &constraints,
                                    bool force_new_instances,
                                    bool meets_constraints);
  virtual void map_copy(const MapperContext ctx,
                        const Copy &copy,
                        const MapCopyInput &input,
                        MapCopyOutput &output);
  virtual void map_must_epoch(const MapperContext           ctx,
                              const MapMustEpochInput&      input,
                                    MapMustEpochOutput&     output);
  template<bool IS_SRC>
  void pennant_create_copy_instance(MapperContext ctx, const Copy &copy,
                                    const RegionRequirement &req, unsigned index,
                                    std::vector<PhysicalInstance> &instances);
private:
  std::vector<Processor>& procs_list;
  std::vector<Memory>& sysmems_list;
  std::map<Memory, std::vector<Processor> >& sysmem_local_procs;
#if SPMD_SHARD_USE_IO_PROC
  std::map<Memory, std::vector<Processor> >& sysmem_local_io_procs;
#endif
  std::map<Processor, Memory>& proc_sysmems;
  // std::map<Processor, Memory>& proc_regmems;
};

PennantMapper::PennantMapper(MapperRuntime *rt, Machine machine, Processor local,
                             const char *mapper_name,
                             std::vector<Processor>* _procs_list,
                             std::vector<Memory>* _sysmems_list,
                             std::map<Memory, std::vector<Processor> >* _sysmem_local_procs,
#if SPMD_SHARD_USE_IO_PROC
                             std::map<Memory, std::vector<Processor> >* _sysmem_local_io_procs,
#endif
                             std::map<Processor, Memory>* _proc_sysmems,
                             std::map<Processor, Memory>* _proc_regmems)
  : DefaultMapper(rt, machine, local, mapper_name),
    procs_list(*_procs_list),
    sysmems_list(*_sysmems_list),
    sysmem_local_procs(*_sysmem_local_procs),
#if SPMD_SHARD_USE_IO_PROC
    sysmem_local_io_procs(*_sysmem_local_io_procs),
#endif
    proc_sysmems(*_proc_sysmems)// ,
    // proc_regmems(*_proc_regmems)
{
}

void PennantMapper::default_policy_rank_processor_kinds(MapperContext ctx,
                        const Task &task, std::vector<Processor::Kind> &ranking)
{
#if SPMD_SHARD_USE_IO_PROC
  const char* task_name = task.get_task_name();
  const char* prefix = "shard_";
  if (strncmp(task_name, prefix, strlen(prefix)) == 0) {
    // Put shard tasks on IO processors.
    ranking.resize(5);
    ranking[0] = Processor::TOC_PROC;
    ranking[1] = Processor::PROC_SET;
    ranking[2] = Processor::IO_PROC;
    ranking[3] = Processor::LOC_PROC;
    ranking[4] = Processor::PY_PROC;
  } else {
#endif
    ranking.resize(5);
    ranking[0] = Processor::TOC_PROC;
    ranking[1] = Processor::PROC_SET;
    ranking[2] = Processor::LOC_PROC;
    ranking[3] = Processor::IO_PROC;
    ranking[4] = Processor::PY_PROC;
#if SPMD_SHARD_USE_IO_PROC
  }
#endif
}

Processor PennantMapper::default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task)
{
  if (!task.regions.empty()) {
    if (task.regions[0].handle_type == SINGULAR) {
      Color index = runtime->get_logical_region_color(ctx, task.regions[0].region);
#define NO_SPMD 0
#if NO_SPMD
      return procs_list[index % procs_list.size()];
#else
      std::vector<Processor> &local_procs =
        sysmem_local_procs[proc_sysmems[local_proc]];
      if (local_procs.size() > 1) {
#define SPMD_RESERVE_SHARD_PROC 0
#if SPMD_RESERVE_SHARD_PROC
        return local_procs[(index % (local_procs.size() - 1)) + 1];
#else
        return local_procs[index % local_procs.size()];
#endif
      } else if (local_procs.size() > 0) { // FIXME: This check seems to be required when using Python processors
        return local_procs[0];
      }
#endif
    }
  }

  return DefaultMapper::default_policy_select_initial_processor(ctx, task);
}

void PennantMapper::default_policy_select_target_processors(
                                    MapperContext ctx,
                                    const Task &task,
                                    std::vector<Processor> &target_procs)
{
  target_procs.push_back(task.target_proc);
}

LogicalRegion PennantMapper::default_policy_select_instance_region(
                                MapperContext ctx, Memory target_memory,
                                const RegionRequirement &req,
                                const LayoutConstraintSet &layout_constraints,
                                bool force_new_instances,
                                bool meets_constraints)
{
  return req.region;
}

//--------------------------------------------------------------------------
template<bool IS_SRC>
void PennantMapper::pennant_create_copy_instance(MapperContext ctx,
                     const Copy &copy, const RegionRequirement &req,
                     unsigned idx, std::vector<PhysicalInstance> &instances)
//--------------------------------------------------------------------------
{
  // This method is identical to the default version except that it
  // chooses an intelligent memory based on the destination of the
  // copy.

  // See if we have all the fields covered
  std::set<FieldID> missing_fields = req.privilege_fields;
  for (std::vector<PhysicalInstance>::const_iterator it =
        instances.begin(); it != instances.end(); it++)
  {
    it->remove_space_fields(missing_fields);
    if (missing_fields.empty())
      break;
  }
  if (missing_fields.empty())
    return;
  // If we still have fields, we need to make an instance
  // We clearly need to take a guess, let's see if we can find
  // one of our instances to use.

  // ELLIOTT: Get the remote node here.
  Color index = runtime->get_logical_region_color(ctx, copy.src_requirements[idx].region);
// #if SPMD_RESERVE_SHARD_PROC
//   size_t sysmem_index = index / (std::max(sysmem_local_procs.begin()->second.size() - 1, (size_t)1));
// #else
//   size_t sysmem_index = index / sysmem_local_procs.begin()->second.size();
// #endif
//   assert(sysmem_index < sysmems_list.size());
//   Memory target_memory = sysmems_list[sysmem_index];
  Memory target_memory = default_policy_select_target_memory(ctx,
                           procs_list[index % procs_list.size()],
                           req);
  log_pennant.spew("Building instance for copy of a region with index %u to be in memory %llx",
                      index, target_memory.id);
  bool force_new_instances = false;
  LayoutConstraintID our_layout_id =
   default_policy_select_layout_constraints(ctx, target_memory,
                                            req, COPY_MAPPING,
                                            true/*needs check*/,
                                            force_new_instances);
  LayoutConstraintSet creation_constraints =
              runtime->find_layout_constraints(ctx, our_layout_id);
  creation_constraints.add_constraint(
      FieldConstraint(missing_fields,
                      false/*contig*/, false/*inorder*/));
  instances.resize(instances.size() + 1);
  if (!default_make_instance(ctx, target_memory,
        creation_constraints, instances.back(),
        COPY_MAPPING, force_new_instances, true/*meets*/, req))
  {
    // If we failed to make it that is bad
    log_pennant.error("Pennant mapper failed allocation for "
                   "%s region requirement %d of explicit "
                   "region-to-region copy operation in task %s "
                   "(ID %lld) in memory " IDFMT " for processor "
                   IDFMT ". This means the working set of your "
                   "application is too big for the allotted "
                   "capacity of the given memory under the default "
                   "mapper's mapping scheme. You have three "
                   "choices: ask Realm to allocate more memory, "
                   "write a custom mapper to better manage working "
                   "sets, or find a bigger machine. Good luck!",
                   IS_SRC ? "source" : "destination", idx,
                   copy.parent_task->get_task_name(),
                   copy.parent_task->get_unique_id(),
		       target_memory.id,
		       copy.parent_task->current_proc.id);
    assert(false);
  }
}

void PennantMapper::map_copy(const MapperContext ctx,
                             const Copy &copy,
                             const MapCopyInput &input,
                             MapCopyOutput &output)
{
  log_pennant.spew("Pennant mapper map_copy");
  for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++)
  {
    // Use a virtual instance for the source unless source is
    // restricted or we'd applying a reduction.
    output.src_instances[idx].clear();
    if (copy.src_requirements[idx].is_restricted()) {
      // If it's restricted, just take the instance. This will only
      // happen inside the shard task.
      output.src_instances[idx] = input.src_instances[idx];
      if (!output.src_instances[idx].empty())
        runtime->acquire_and_filter_instances(ctx,
                                output.src_instances[idx]);
    } else if (copy.dst_requirements[idx].privilege == REDUCE) {
      // Use the default here. This will place the instance on the
      // current node.
      default_create_copy_instance<true/*is src*/>(ctx, copy,
                copy.src_requirements[idx], idx, output.src_instances[idx]);
    } else {
      output.src_instances[idx].push_back(
        PhysicalInstance::get_virtual_instance());
    }

    // Place the destination instance on the remote node.
    output.dst_instances[idx].clear();
    if (!copy.dst_requirements[idx].is_restricted()) {
      // Call a customized method to create an instance on the desired node.
      pennant_create_copy_instance<false/*is src*/>(ctx, copy,
        copy.dst_requirements[idx], idx, output.dst_instances[idx]);
    } else {
      // If it's restricted, just take the instance. This will only
      // happen inside the shard task.
      output.dst_instances[idx] = input.dst_instances[idx];
      if (!output.dst_instances[idx].empty())
        runtime->acquire_and_filter_instances(ctx,
                                output.dst_instances[idx]);
    }
  }
}
void PennantMapper::map_must_epoch(const MapperContext           ctx,
                                   const MapMustEpochInput&      input,
                                         MapMustEpochOutput&     output)
{
  size_t num_nodes = sysmems_list.size();
  size_t num_tasks = input.tasks.size();
  size_t num_shards_per_node =
    num_nodes < input.tasks.size() ? (num_tasks + num_nodes - 1) / num_nodes : 1;
  std::map<const Task*, size_t> task_indices;
  for (size_t idx = 0; idx < num_tasks; ++idx) {
    size_t node_idx = idx / num_shards_per_node;
    size_t proc_idx = idx % num_shards_per_node;
    assert(node_idx < sysmems_list.size());
#if SPMD_SHARD_USE_IO_PROC
    assert(proc_idx < sysmem_local_io_procs[sysmems_list[node_idx]].size());
    output.task_processors[idx] = sysmem_local_io_procs[sysmems_list[node_idx]][proc_idx];
#else
    assert(proc_idx < sysmem_local_procs[sysmems_list[node_idx]].size());
    output.task_processors[idx] = sysmem_local_procs[sysmems_list[node_idx]][proc_idx];
#endif

    task_indices[input.tasks[idx]] = node_idx;
  }

  for (size_t idx = 0; idx < input.constraints.size(); ++idx) {
    const MappingConstraint& constraint = input.constraints[idx];
    int owner_id = -1;

    for (unsigned i = 0; i < constraint.constrained_tasks.size(); ++i) {
      const RegionRequirement& req =
        constraint.constrained_tasks[i]->regions[
          constraint.requirement_indexes[i]];
      if (req.is_no_access()) continue;
      assert(owner_id == -1);
      owner_id = static_cast<int>(i);
    }
    assert(owner_id != -1);

    const Task* task = constraint.constrained_tasks[owner_id];
    const RegionRequirement& req =
      task->regions[constraint.requirement_indexes[owner_id]];
    Memory target_memory = sysmems_list[task_indices[task]];
    LayoutConstraintSet layout_constraints;
    default_policy_select_constraints(ctx, layout_constraints, target_memory, req);
    layout_constraints.add_constraint(
      FieldConstraint(req.privilege_fields, false /*!contiguous*/));

    PhysicalInstance inst;
    bool created;
    bool ok = runtime->find_or_create_physical_instance(ctx, target_memory,
        layout_constraints, std::vector<LogicalRegion>(1, req.region),
        inst, created, true /*acquire*/);
    assert(ok);
    output.constraint_mappings[idx].push_back(inst);
  }
}

static void create_mappers(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs)
{
  std::vector<Processor>* procs_list = new std::vector<Processor>();
  std::vector<Memory>* sysmems_list = new std::vector<Memory>();
  std::map<Memory, std::vector<Processor> >* sysmem_local_procs =
    new std::map<Memory, std::vector<Processor> >();
#if SPMD_SHARD_USE_IO_PROC
  std::map<Memory, std::vector<Processor> >* sysmem_local_io_procs =
    new std::map<Memory, std::vector<Processor> >();
#endif
  std::map<Processor, Memory>* proc_sysmems = new std::map<Processor, Memory>();
  std::map<Processor, Memory>* proc_regmems = new std::map<Processor, Memory>();


  std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
  machine.get_proc_mem_affinity(proc_mem_affinities);

  for (unsigned idx = 0; idx < proc_mem_affinities.size(); ++idx) {
    Machine::ProcessorMemoryAffinity& affinity = proc_mem_affinities[idx];

    // skip memories with no capacity for creating instances
    if(affinity.m.capacity() == 0)
      continue;

    if (affinity.p.kind() == Processor::LOC_PROC ||
        affinity.p.kind() == Processor::IO_PROC) {
      if (affinity.m.kind() == Memory::SYSTEM_MEM) {
        (*proc_sysmems)[affinity.p] = affinity.m;
        if (proc_regmems->find(affinity.p) == proc_regmems->end())
          (*proc_regmems)[affinity.p] = affinity.m;
      }
      else if (affinity.m.kind() == Memory::REGDMA_MEM)
        (*proc_regmems)[affinity.p] = affinity.m;
    }
  }

  for (std::map<Processor, Memory>::iterator it = proc_sysmems->begin();
       it != proc_sysmems->end(); ++it) {
    if (it->first.kind() == Processor::LOC_PROC) {
      procs_list->push_back(it->first);
      (*sysmem_local_procs)[it->second].push_back(it->first);
    }
#if SPMD_SHARD_USE_IO_PROC
    else if (it->first.kind() == Processor::IO_PROC) {
      (*sysmem_local_io_procs)[it->second].push_back(it->first);
    }
#endif
  }

  for (std::map<Memory, std::vector<Processor> >::iterator it =
        sysmem_local_procs->begin(); it != sysmem_local_procs->end(); ++it)
    sysmems_list->push_back(it->first);

  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    PennantMapper* mapper = new PennantMapper(runtime->get_mapper_runtime(),
                                              machine, *it, "pennant_mapper",
                                              procs_list,
                                              sysmems_list,
                                              sysmem_local_procs,
#if SPMD_SHARD_USE_IO_PROC
                                              sysmem_local_io_procs,
#endif
                                              proc_sysmems,
                                              proc_regmems);
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  Runtime::add_registration_callback(create_mappers);
}
