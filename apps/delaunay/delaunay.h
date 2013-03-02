/* Copyright 2012 Stanford University
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


#ifndef __LEGION_DELAUNAY__
#define __LEGION_DELAUNAY__

#include "legion.h"
#include "accessor.h"
#include "legion_vector.h"
#include "serial_stl.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionStd;

enum TaskNames {
  LEGION_MAIN,
  REFINE_MESH,
  DELAUNAY_RECURSE,
  COMPUTE_CAVITY,
  PROCESS_CAVITY,
};

typedef unsigned gen_t;

struct Point {
public:
  Point(void)
  {
    coords[0] = 0.0f; coords[1] = 0.0f; coords[2] = 0.0f;
  }
  Point(float x, float y, float z)
  {
    coords[0] = x; coords[1] = y; coords[2] = z;
  }
public:
  inline bool operator==(const Point &rhs) const
  {
    for (int i = 0; i < 3; i++)
      if (coords[i] != rhs.coords[i])
        return false;
    return true;
  }
  inline bool operator!=(const Point &rhs) const
  {
    return !(*this == rhs);
  }
public:
  float coords[3];
};

struct BadTriangle {
public:
  BadTriangle(void) : ptr(ptr_t::nil()), gen(0) { }
  BadTriangle(ptr_t p, gen_t g) : ptr(p), gen(g) { }
public:
  ptr_t ptr;
  gen_t gen;
};

struct BorderEdge {
public:
  ptr_t update_triangle;
  unsigned neighbor_idx;
  Point edge[2];
};

struct Triangle {
public:
  Triangle(void) { }
  Triangle(const Point &p1, const Point &p2);
  Triangle(const Point &p1, const Point &p2, const Point &p3);
public:
  inline int find_index(const Point &p) const
  {
    if (p == points[0]) return 0;
    if (p == points[1]) return 1;
    if ((dim==3) && (p == points[2])) return 2;
    return -1;
  }
  void set_adjacent(const Point &p1, const Point &p2, ptr_t adjacent);
  inline bool is_bad_triangle(void) const { return bad; }
  float get_angle(unsigned idx) const;
  inline bool is_obtuse(void) const { return obtuse; }
  inline ptr_t get_obtuse_opposite(void) const { return neighbors[obtuse_idx]; }
  bool in_circle(const Point &point) const;
public:
  static float compute_angle(const Point &p1, const Point &p2, const Point &p3);
  BorderEdge compute_border_edge(const Triangle &other) const;
public:
  bool bad, obtuse;
  unsigned dim, obtuse_idx; // either 2 or 3
  ptr_t my_ptr;
  gen_t generation;
  unsigned num_neighbors; // either 1 or 3
  ptr_t neighbors[3];
  Point points[3];
  Point center;
  float radius_squared;
};


struct Cavity {
public:
  struct CavityTriangle {
  public:
    CavityTriangle(void)
      : ptr(ptr_t::nil()), gen(0), inside(false) { }
    CavityTriangle(ptr_t p, gen_t g, bool in)
      : ptr(p), gen(g), inside(in) { }
  public:
    ptr_t ptr;
    gen_t gen;
    bool inside;
  };
public:
  Cavity(void);
public:
  void initialize(ptr_t start_ptr, LegionVector<Triangle> &triangles);
  void build(LogicalRegion log_triangles, LegionVector<Triangle> &triangles,
             HighLevelRuntime *rt, Context ctx);
  bool is_member(const Point &center, const Triangle &tri) const;
  void update(LegionVector<Triangle> &triangles, SerializableVector<BadTriangle> &new_bad_triangles);
  bool is_valid_cavity(BadTriangle init, LegionVector<Triangle> &triangles);
public:
  size_t legion_buffer_size(void) const;
  void legion_serialize(void *buffer);
  void legion_deserialize(const void *buffer);
public:
  bool valid_cavity;
  LogicalRegion needed_triangles;
  LogicalPartition needed_part;
  ptr_t center_ptr;
  BadTriangle init_bad;
  std::vector<CavityTriangle> cavity_tris;
  std::vector<BorderEdge> border;
  mutable size_t buffer_size;
};

#endif // __LEGION_DELAUNAY__
