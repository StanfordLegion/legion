/* Copyright 2016 Stanford University
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

#include "default_mapper.h"

#include <sys/time.h>
#include <sys/resource.h>
void print_rusage(const char *message)
{
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0) return;
  printf("%s: %ld MB\n", message, usage.ru_maxrss / 1024);
}

using namespace LegionRuntime::HighLevel;

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

static LegionRuntime::Logger::Category log_pennant("pennant");

class PennantMapper : public DefaultMapper
{
public:
  PennantMapper(Machine machine, HighLevelRuntime *rt, Processor local);
  virtual void select_task_options(Task *task);
  virtual void select_task_variant(Task *task);
  virtual bool map_task(Task *task);
  virtual bool map_inline(Inline *inline_operation);
  virtual bool map_copy(Copy *copy);
  virtual void notify_mapping_failed(const Mappable *mappable);
  virtual bool map_must_epoch(const std::vector<Task*> &tasks,
                              const std::vector<MappingConstraint> &constraints,
                              MappingTagID tag);
  //virtual bool rank_copy_targets(const Mappable *mappable,
  //                               LogicalRegion rebuild_region,
  //                               const std::set<Memory> &current_instances,
  //                               bool complete,
  //                               size_t max_blocking_factor,
  //                               std::set<Memory> &to_reuse,
  //                               std::vector<Memory> &to_create,
  //                               bool &create_one,
  //                               size_t &blocking_factor);
  //virtual void rank_copy_sources(const Mappable *mappable,
  //                               const std::set<Memory> &current_instances,
  //                               Memory dst_mem,
  //                               std::vector<Memory> &chosen_order);
private:
  Color get_task_color_by_region(Task *task, const RegionRequirement &requirement);
  LogicalRegion get_root_region(LogicalRegion handle);
  LogicalRegion get_root_region(LogicalPartition handle);
private:
  std::vector<Processor> procs_list;
  std::vector<Memory> sysmems_list;
  std::map<Memory, std::vector<Processor> > sysmem_local_procs;
  std::map<Processor, Memory> proc_sysmems;
  std::map<Processor, Memory> proc_regmems;
};

PennantMapper::PennantMapper(Machine machine, HighLevelRuntime *rt, Processor local)
  : DefaultMapper(machine, rt, local)
{
  Machine::ProcessorQuery procs =
    Machine::ProcessorQuery(machine).only_kind(Processor::LOC_PROC);
  procs_list.assign(procs.begin(), procs.end());

  Machine::MemoryQuery sysmems =
    Machine::MemoryQuery(machine).only_kind(Memory::SYSTEM_MEM);
  sysmems_list.assign(sysmems.begin(), sysmems.end());
  Machine::MemoryQuery regmems =
    Machine::MemoryQuery(machine).only_kind(Memory::REGDMA_MEM);

  for (Machine::ProcessorQuery::iterator it = procs.begin();
       it != procs.end(); ++it) {
    Memory sysmem = Machine::MemoryQuery(sysmems).has_affinity_to(*it).first();
    assert(sysmem.exists());
    proc_sysmems[*it] = sysmem;
    Memory regmem = Machine::MemoryQuery(regmems).has_affinity_to(*it).first();
    if (!regmem.exists()) regmem = sysmem;
    proc_regmems[*it] = regmem;
  }

  for (Machine::MemoryQuery::iterator it = sysmems.begin();
       it != sysmems.end(); ++it) {
    Machine::ProcessorQuery local_procs =
      Machine::ProcessorQuery(procs).has_affinity_to(*it);
    sysmem_local_procs[*it].assign(local_procs.begin(), local_procs.end());
  }
}

void PennantMapper::select_task_options(Task *task)
{
  // Task options:
  task->inline_task = false;
  task->spawn_task = false;
  task->map_locally = true;
  task->profile_task = false;

  if (!task->regions.empty()) {
    if (task->regions[0].handle_type == SINGULAR) {
      Color index = get_logical_region_color(task->regions[0].region);
      std::vector<Processor> &local_procs =
        sysmem_local_procs[proc_sysmems[task->target_proc]];
      task->target_proc = local_procs[(index % (local_procs.size() - 1)) + 1];
    }
  }
}

void PennantMapper::select_task_variant(Task *task)
{
  // Use the SOA variant for all tasks.
  // task->selected_variant = VARIANT_SOA;
  DefaultMapper::select_task_variant(task);

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
  task->additional_procs.clear();
  // std::vector<Processor> &local_procs = sysmem_local_procs[proc_sysmems[task->target_proc]];
  // task->additional_procs.insert(local_procs.begin(), local_procs.end());

  Memory sysmem = proc_sysmems[task->target_proc];
  assert(sysmem.exists());
  std::vector<RegionRequirement> &regions = task->regions;
  for (std::vector<RegionRequirement>::iterator it = regions.begin();
        it != regions.end(); it++) {
    RegionRequirement &req = *it;

    // Region options:
    req.virtual_map = false;
    req.enable_WAR_optimization = false;
    req.reduction_list = false;

    // Place all regions in local system memory.
    req.target_ranking.push_back(sysmem);
    {
      LogicalRegion root;

      if (req.handle_type == SINGULAR || req.handle_type == REG_PROJECTION) {
        root = get_root_region(req.region);
      } else {
        assert(req.handle_type == PART_PROJECTION);
        root = get_root_region(req.partition);
      }

      const char *name_;
      runtime->retrieve_name(root, name_);
      assert(name_);
      std::string name(name_);

      int num_fields = 0;
      if (name == "rz_all") {
        num_fields = 24;
      } else if (name == "rp_all") {
        num_fields = 17;
      } else if (name == "rs_all") {
        num_fields = 34;
      } else if (name == "rz_spans") {
        num_fields = 2;
      } else if (name == "rs_spans") {
        num_fields = 2;
      } else if (name == "rp_spans_private") {
        num_fields = 2;
      } else if (name == "rp_spans_shared") {
        num_fields = 2;
      } else {
        assert(false);
      }

      const int base = 101;
      for (int i = base; i < base + num_fields; i++) {
        req.additional_fields.insert(i);
      }
    }
  }

  return false;
}

bool PennantMapper::map_copy(Copy *copy)
{
  for (unsigned idx = 0; idx < copy->src_requirements.size(); ++idx) {
    RegionRequirement& src_req = copy->src_requirements[idx];
    RegionRequirement& dst_req = copy->dst_requirements[idx];
    Color src_color = get_logical_region_color(src_req.region);
    Color dst_color = get_logical_region_color(dst_req.region);
    Processor src_proc = procs_list[src_color % procs_list.size()];
    Processor dst_proc = procs_list[dst_color % procs_list.size()];

#if 0
    // If the source is a composite instance, then the runtime will
    // crash here if we decline to map the source.
    if (!src_req.restricted) {
#endif
      src_req.target_ranking.clear();
      for (std::map<Memory, bool>::iterator it = src_req.current_instances.begin(),
             ie = src_req.current_instances.end(); it != ie; ++it) {
        if (it->second) {
          src_req.target_ranking.push_back(it->first);
        }
      }
      if (src_req.target_ranking.empty()) {
        src_req.target_ranking.push_back(proc_sysmems[src_proc]);
      }
#if 0
    }
#endif

    if (!dst_req.restricted) {
      dst_req.target_ranking.clear();
      dst_req.target_ranking.push_back(proc_sysmems[dst_proc]);
    }

    src_req.blocking_factor = src_req.max_blocking_factor;
    dst_req.blocking_factor = dst_req.max_blocking_factor;
  }
  return false;
}

bool PennantMapper::map_inline(Inline *inline_operation)
{
  Memory sysmem = proc_sysmems[local_proc];
  RegionRequirement &req = inline_operation->requirement;

  // Region options:
  req.virtual_map = false;
  req.enable_WAR_optimization = false;
  req.reduction_list = false;
  req.blocking_factor = req.max_blocking_factor;

  // Place all regions in global memory.
  req.target_ranking.push_back(sysmem);

  log_pennant.debug(
    "inline mapping region (%d,%d,%d) target ranking front " IDFMT " (size %zu)",
    req.region.get_index_space().get_id(),
    req.region.get_field_space().get_id(),
    req.region.get_tree_id(),
    req.target_ranking[0].id,
    req.target_ranking.size());

  return false;
}

bool PennantMapper::map_must_epoch(const std::vector<Task*> &tasks,
                                    const std::vector<MappingConstraint> &constraints,
                                    MappingTagID tag)
{
  unsigned tasks_per_sysmem = (tasks.size() + sysmems_list.size() - 1) / sysmems_list.size();
  for (unsigned i = 0; i < tasks.size(); ++i)
  {
    Task* task = tasks[i];
    unsigned index = task->index_point.point_data[0];
    assert(index / tasks_per_sysmem < sysmems_list.size());
    Memory sysmem = sysmems_list[index / tasks_per_sysmem];
    unsigned subindex = index % tasks_per_sysmem;
    assert(subindex < sysmem_local_procs[sysmem].size());
    task->target_proc = sysmem_local_procs[sysmem][subindex];
    map_task(task);
    task->additional_procs.clear();
  }

  typedef std::map<LogicalRegion, Memory> Mapping;
  Mapping mappings;
  for (unsigned i = 0; i < constraints.size(); ++i)
  {
    const MappingConstraint& c = constraints[i];
    if (c.t1->regions[c.idx1].flags & NO_ACCESS_FLAG &&
        c.t2->regions[c.idx2].flags & NO_ACCESS_FLAG)
      continue;

    Memory regmem;
    if (c.t2->regions[c.idx2].flags & NO_ACCESS_FLAG)
      regmem = proc_regmems[c.t1->target_proc]; // proc_sysmems[c.t1->target_proc];
    else if (c.t1->regions[c.idx1].flags & NO_ACCESS_FLAG)
      regmem = proc_regmems[c.t2->target_proc]; // proc_sysmems[c.t2->target_proc];
    else
      assert(0);
    c.t1->regions[c.idx1].target_ranking.clear();
    c.t1->regions[c.idx1].target_ranking.push_back(regmem);
    c.t2->regions[c.idx2].target_ranking.clear();
    c.t2->regions[c.idx2].target_ranking.push_back(regmem);
    mappings[c.t1->regions[c.idx1].region] = regmem;
  }

  for (unsigned i = 0; i < constraints.size(); ++i)
  {
    const MappingConstraint& c = constraints[i];
    if (c.t1->regions[c.idx1].flags & NO_ACCESS_FLAG &&
        c.t2->regions[c.idx2].flags & NO_ACCESS_FLAG)
    {
      Mapping::iterator it =
        mappings.find(c.t1->regions[c.idx1].region);
      assert(it != mappings.end());
      Memory regmem = it->second;
      c.t1->regions[c.idx1].target_ranking.clear();
      c.t1->regions[c.idx1].target_ranking.push_back(regmem);
      c.t2->regions[c.idx2].target_ranking.clear();
      c.t2->regions[c.idx2].target_ranking.push_back(regmem);
    }
  }

  //for (unsigned i = 0; i < constraints.size(); ++i)
  //{
  //  const MappingConstraint& c = constraints[i];
  //  fprintf(stderr,
  //      "task %s (UID: %llu) point %d region %u (%x,%d,%d)\n",
  //      c.t1->get_task_name(), c.t1->get_unique_mappable_id(),
  //      c.t1->index_point.point_data[0],
  //      c.idx1,
  //      c.t1->regions[c.idx1].region.get_index_space().get_id(),
  //      c.t1->regions[c.idx1].region.get_field_space().get_id(),
  //      c.t1->regions[c.idx1].region.get_tree_id());
  //  fprintf(stderr,
  //      "task %s (UID: %llu) point %d region %u (%x,%d,%d)\n",
  //      c.t2->get_task_name(), c.t2->get_unique_mappable_id(),
  //      c.t2->index_point.point_data[0],
  //      c.idx2,
  //      c.t2->regions[c.idx2].region.get_index_space().get_id(),
  //      c.t2->regions[c.idx2].region.get_field_space().get_id(),
  //      c.t2->regions[c.idx2].region.get_tree_id());
  //  fprintf(stderr,
  //      "task %s (UID: %llu) region %u -> task %s (UID: %llu) region %u, mapped to memory %llx, %llx\n",
  //      c.t1->get_task_name(), c.t1->get_unique_mappable_id(), c.idx1,
  //      c.t2->get_task_name(), c.t2->get_unique_mappable_id(), c.idx2,
  //      c.t1->regions[c.idx1].target_ranking.begin()->id,
  //      c.t2->regions[c.idx2].target_ranking.begin()->id);
  //}

  return false;
}

void PennantMapper::notify_mapping_failed(const Mappable *mappable)
{
  switch (mappable->get_mappable_kind()) {
  case Mappable::TASK_MAPPABLE:
    {
      Task *task = mappable->as_mappable_task();
      log_pennant.warning("mapping failed on task %s", task->variants->name);
      break;
    }
  case Mappable::COPY_MAPPABLE:
    {
      log_pennant.warning("mapping failed on copy");
      break;
    }
  case Mappable::INLINE_MAPPABLE:
    {
      Inline *_inline = mappable->as_mappable_inline();
      RegionRequirement &req = _inline->requirement;
      LogicalRegion region = req.region;
      log_pennant.warning(
        "mapping %s on inline region (%d,%d,%d) memory " IDFMT,
        (req.mapping_failed ? "failed" : "succeeded"),
        region.get_index_space().get_id(),
        region.get_field_space().get_id(),
        region.get_tree_id(),
        req.selected_memory.id);
      break;
    }
  case Mappable::ACQUIRE_MAPPABLE:
    {
      log_pennant.warning("mapping failed on acquire");
      break;
    }
  case Mappable::RELEASE_MAPPABLE:
    {
      log_pennant.warning("mapping failed on release");
      break;
    }
  }
  assert(0 && "mapping failed");
}

//bool PennantMapper::rank_copy_targets(const Mappable *mappable,
//                                      LogicalRegion rebuild_region,
//                                      const std::set<Memory> &current_instances,
//                                      bool complete,
//                                      size_t max_blocking_factor,
//                                      std::set<Memory> &to_reuse,
//                                      std::vector<Memory> &to_create,
//                                      bool &create_one,
//                                      size_t &blocking_factor)
//{
//  // DefaultMapper::rank_copy_targets(mappable, rebuild_region, current_instances,
//  //                                  complete, max_blocking_factor, to_reuse,
//  //                                  to_create, create_one, blocking_factor);
//  // if (create_one) {
//  //   blocking_factor = max_blocking_factor;
//  // }
//  return true;
//}

//void PennantMapper::rank_copy_sources(const Mappable *mappable,
//                                      const std::set<Memory> &current_instances,
//                                      Memory dst_mem,
//                                      std::vector<Memory> &chosen_order)
//{
//  // Elliott: This is to fix a bug in the default mapper which throws
//  // an error with composite instances.
//
//  // Handle the simple case of having the destination
//  // memory in the set of instances
//  if (current_instances.find(dst_mem) != current_instances.end())
//  {
//    chosen_order.push_back(dst_mem);
//    return;
//  }
//
//  machine_interface.find_memory_stack(dst_mem,
//                                      chosen_order, true/*latency*/);
//  if (chosen_order.empty())
//  {
//    // This is the multi-hop copy because none
//    // of the memories had an affinity
//    // SJT: just send the first one
//    if(current_instances.size() > 0) {
//      chosen_order.push_back(*(current_instances.begin()));
//    } else {
//      // Elliott: This is a composite instance.
//      //assert(false);
//    }
//  }
//}

Color PennantMapper::get_task_color_by_region(Task *task, const RegionRequirement &requirement)
{
  if (requirement.handle_type == SINGULAR) {
    return get_logical_region_color(requirement.region);
  }
  return 0;
}

LogicalRegion PennantMapper::get_root_region(LogicalRegion handle)
{
  if (has_parent_logical_partition(handle)) {
    return get_root_region(get_parent_logical_partition(handle));
  }
  return handle;
}

LogicalRegion PennantMapper::get_root_region(LogicalPartition handle)
{
  return get_root_region(get_parent_logical_region(handle));
}

static void create_mappers(Machine machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    runtime->replace_default_mapper(new PennantMapper(machine, runtime, *it), *it);
  }
}

void register_mappers()
{
  HighLevelRuntime::set_registration_callback(create_mappers);
}
