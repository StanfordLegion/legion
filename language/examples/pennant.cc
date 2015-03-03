/* Copyright 2015 Stanford University
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
                          // std::vector<std::vector<int64_t> > &mapzp,
                          std::vector<int64_t> &zonestart,
                          std::vector<int64_t> &zonesize,
                          std::vector<int64_t> &zonepoints,
                          std::vector<int64_t> &zonecolors)
{
  // Do calculations common to all mesh types:
  std::vector<int64_t> zxbounds;
  std::vector<int64_t> zybounds;
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
  // std::vector<int64_t> zonestart;
  // std::vector<int64_t> zonesize;
  // std::vector<int64_t> zonepoints;
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
  // mapzp.resize(conf.nz);
  // for (int64_t z = 0; z < conf.nz; z++) {
  //   int64_t p0 = zonestart[z];
  //   int64_t znump = zonesize[z];
  //   for (int64_t p = p0; p < p0 + znump; p++) {
  //     mapzp[z].push_back(zonepoints[p]);
  //   }
  // }
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
  double *pointpos_x, size_t *pointpos_x_size,
  double *pointpos_y, size_t *pointpos_y_size,
  int64_t *pointcolors, size_t *pointcolors_size,
  uint64_t *pointmcolors, size_t *pointmcolors_size,
  int64_t *zonestart, size_t *zonestart_size,
  int64_t *zonesize, size_t *zonesize_size,
  int64_t *zonepoints, size_t *zonepoints_size,
  int64_t *zonecolors, size_t *zonecolors_size)
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

  int64_t color_words = int64_t(ceil(conf_npieces/64.0));

  assert(pointpos_x_vec.size() <= *pointpos_x_size);
  assert(pointpos_y_vec.size() <= *pointpos_y_size);
  assert(pointcolors_vec.size() <= *pointcolors_size);
  assert(pointcolors_vec.size()*color_words <= *pointmcolors_size);
  assert(zonestart_vec.size() <= *zonestart_size);
  assert(zonesize_vec.size() <= *zonesize_size);
  assert(zonepoints_vec.size() <= *zonepoints_size);
  assert(zonecolors_vec.size() <= *zonecolors_size);

  memcpy(pointpos_x, pointpos_x_vec.data(), pointpos_x_vec.size()*sizeof(double));
  memcpy(pointpos_y, pointpos_y_vec.data(), pointpos_y_vec.size()*sizeof(double));
  memcpy(pointcolors, pointcolors_vec.data(), pointcolors_vec.size()*sizeof(int64_t));
  memcpy(zonestart, zonestart_vec.data(), zonestart_vec.size()*sizeof(int64_t));
  memcpy(zonesize, zonesize_vec.data(), zonesize_vec.size()*sizeof(int64_t));
  memcpy(zonepoints, zonepoints_vec.data(), zonepoints_vec.size()*sizeof(int64_t));
  memcpy(zonecolors, zonecolors_vec.data(), zonecolors_vec.size()*sizeof(int64_t));

  memset(pointmcolors, (*pointmcolors_size)*sizeof(uint64_t), 0);
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
  *zonestart_size = zonestart_vec.size();
  *zonesize_size = zonesize_vec.size();
  *zonepoints_size = zonepoints_vec.size();
  *zonecolors_size = zonecolors_vec.size();
}

///
/// Mapper
///

LegionRuntime::Logger::Category log_mapper("mapper");

class PennantMapper : public DefaultMapper
{
public:
  PennantMapper(Machine machine, HighLevelRuntime *rt, Processor local);
  virtual void select_task_options(Task *task);
  virtual void select_task_variant(Task *task);
  virtual bool map_task(Task *task);
  virtual bool map_inline(Inline *inline_operation);
  virtual void notify_mapping_failed(const Mappable *mappable);
private:
  Color get_task_color_by_region(Task *task, const RegionRequirement &requirement);
private:
  Memory local_sysmem;
  Memory local_regmem;
  std::set<Processor> local_procs;
};

PennantMapper::PennantMapper(Machine machine, HighLevelRuntime *rt, Processor local)
  : DefaultMapper(machine, rt, local)
{
  local_sysmem =
    machine_interface.find_memory_kind(local_proc, Memory::SYSTEM_MEM);
  local_regmem =
    machine_interface.find_memory_kind(local_proc, Memory::REGDMA_MEM);
  if(!local_regmem.exists()) {
    local_regmem = local_sysmem;
  }

  machine.get_shared_processors(local_sysmem, local_procs);
  if (!local_procs.empty()) {
    machine_interface.filter_processors(machine, Processor::LOC_PROC, local_procs);
  }
}

void PennantMapper::select_task_options(Task *task)
{
  // Task options:
  task->inline_task = false;
  task->spawn_task = false;
  task->map_locally = false;
  task->profile_task = false;

  task->additional_procs = local_procs;
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

Color PennantMapper::get_task_color_by_region(Task *task, const RegionRequirement &requirement)
{
  if (requirement.handle_type == SINGULAR) {
    return get_logical_region_color(requirement.region);
  }
  return 0;
}

void create_mappers(Machine machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
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
