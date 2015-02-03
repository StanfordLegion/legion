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


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <time.h>

#include "delaunay.h"
#include "legion.h"
#include "legion_vector.h"
#include "serial_stl.h"
#include "alt_mappers.h"

#include <smmintrin.h>

#define PI 3.14159265359
#define MINANGLE  30.0f

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionStd;

LegionRuntime::Logger::Category log_delaunay("delaunay");

void load_triangles(const char *base_name, LegionVector<Triangle> &triangles);
void find_bad_triangles(LegionVector<Triangle> &triangles, LegionVector<BadTriangle> &bad_triangles);

void legion_main(const void *args, size_t arglen,
                 const std::vector<RegionRequirement> &logical_regions,
                 const std::vector<PhysicalRegion> &physical_regions,
                 Context ctx, HighLevelRuntime *runtime)
{
  const char *base_name = NULL;
  size_t max_triangles = 1048576;
  size_t max_values[2];
  max_values[0] = 8192; // max intial bad triangles
  max_values[1] = 1; // max chunk size
  {
    InputArgs *inputs = (InputArgs*)args;
    for (int i = 1; i < inputs->argc; i++)
    {
      if (!strcmp(inputs->argv[i],"-m"))
      {
        max_triangles = atoi(inputs->argv[++i]);
        continue;
      }
      if (!strcmp(inputs->argv[i],"-b"))
      {
        max_values[0] = atoi(inputs->argv[++i]);
        continue;
      }
      if (!strcmp(inputs->argv[i],"-chunks"))
      {
        max_values[1] = atoi(inputs->argv[++i]);
        continue;
      }
      if (!strcmp(inputs->argv[i],"-f"))
      {
        base_name = inputs->argv[++i];
        continue;
      }
    }
  }
  if (base_name == NULL)
  {
    log_delaunay(LEVEL_ERROR,"ERROR: No input file specified");
    exit(1);
  }
  log_delaunay(LEVEL_PRINT,"Loading experiment %s", base_name);
  LegionVector<Triangle> triangles(runtime, ctx, max_triangles);   
  triangles.map(runtime, ctx, READ_WRITE, EXCLUSIVE);
  load_triangles(base_name, triangles);
  log_delaunay(LEVEL_PRINT,"Loaded %ld triangles", triangles.size());
  triangles.unmap(runtime, ctx);

  {
    std::vector<IndexSpaceRequirement> index_reqs;
    std::vector<FieldSpaceRequirement> field_reqs;
    std::vector<RegionRequirement>     region_reqs;
    region_reqs.push_back(triangles.get_region_requirement(READ_WRITE,EXCLUSIVE));
    index_reqs.push_back(triangles.get_index_requirement(MUTABLE));

    Future f = runtime->execute_task(ctx, REFINE_MESH, index_reqs, field_reqs,
                                     region_reqs, TaskArgument(max_values,2*sizeof(size_t)));
    f.get_void_result();
  }
  triangles.destroy(runtime, ctx);
}

void refine_mesh(const void *args, size_t arglen,
                 const std::vector<RegionRequirement> &logical_regions,
                 const std::vector<PhysicalRegion> &physical_regions,
                 Context ctx, HighLevelRuntime *runtime)
{
  size_t max_bad = *((size_t*)args);
  size_t max_chunk = *((size_t*)(((char*)args)+sizeof(size_t)));
  LegionVector<Triangle> triangles(runtime, ctx, logical_regions[0].region, 
                  physical_regions[0], logical_regions[0].instance_fields[0]);
  LegionVector<BadTriangle> bad_triangles(runtime, ctx, max_bad);
  bad_triangles.map(runtime, ctx, READ_WRITE, EXCLUSIVE);
  find_bad_triangles(triangles, bad_triangles);
  log_delaunay(LEVEL_PRINT,"Found %ld initial bad triangles", bad_triangles.size());
  triangles.unmap(runtime, ctx);
  std::vector<IndexSpaceRequirement> index_reqs;
  std::vector<FieldSpaceRequirement> field_reqs;
  std::vector<RegionRequirement>     region_reqs;
  region_reqs.push_back(triangles.get_region_requirement(READ_WRITE,SIMULTANEOUS));
  index_reqs.push_back(triangles.get_index_requirement(ALLOCABLE));
  LegionVector<BadTriangle>::iterator it = bad_triangles.begin();
  while (it != bad_triangles.end())
  {
    SerializableVector<BadTriangle> chunk_tris; 
    for (unsigned idx = 0; (idx < max_chunk) && (it != bad_triangles.end()); idx++,it++)
    {
      chunk_tris.impl.push_back(*it);
    }
    size_t buffer_size = sizeof(size_t) + chunk_tris.legion_buffer_size();
    void *buffer = malloc(buffer_size);
    *((size_t*)buffer) = max_chunk;
    chunk_tris.legion_serialize((void*)(((char*)buffer)+sizeof(size_t)));
    runtime->execute_task(ctx, DELAUNAY_RECURSE, index_reqs, field_reqs,
                          region_reqs, TaskArgument(buffer,buffer_size));
    free(buffer);
  }
  bad_triangles.unmap(runtime, ctx);
  bad_triangles.destroy(runtime, ctx);
}

void delaunay_recurse(const void *args, size_t arglen,
                      const std::vector<RegionRequirement> &logical_regions,
                      const std::vector<PhysicalRegion> &physical_regions,
                      Context ctx, HighLevelRuntime *runtime)
{
  size_t max_bad = *((const size_t*)args);
  SerializableVector<BadTriangle> unpack_bad_triangles;
  unpack_bad_triangles.legion_deserialize((void*)(((char*)args)+sizeof(size_t)));
  std::vector<BadTriangle> &bad_triangles = unpack_bad_triangles.impl; 

  LegionVector<Triangle> triangles(runtime, ctx, logical_regions[0].region, 
                  physical_regions[0], logical_regions[0].instance_fields[0]);
  triangles.unmap(runtime, ctx);

  std::vector<IndexSpaceRequirement> index_reqs;
  std::vector<FieldSpaceRequirement> field_reqs;
  std::vector<RegionRequirement>    region_reqs;
  std::vector<Future>            cavity_futures;
  std::vector<Future>               bad_futures;

  log_delaunay(LEVEL_INFO,"Starting delaunay recurse with %ld bad triangles %p", bad_triangles.size(), ctx);

  while (!bad_triangles.empty())
  {
    index_reqs.clear();
    region_reqs.clear();
    region_reqs.push_back(triangles.get_region_requirement(READ_ONLY, ATOMIC));
    for (std::vector<BadTriangle>::const_iterator it = bad_triangles.begin();
          it != bad_triangles.end(); it++)
    {
      BadTriangle bad_tri = *it;
      cavity_futures.push_back(runtime->execute_task(ctx, COMPUTE_CAVITY, index_reqs, field_reqs,
                                                     region_reqs, TaskArgument(&bad_tri,sizeof(BadTriangle))));
    }
    bad_triangles.clear();

    region_reqs[0].privilege = READ_WRITE;
    index_reqs.push_back(triangles.get_index_requirement(ALLOCABLE));
    // Now wait for each of our cavities to come back
    for (std::vector<Future>::iterator it = cavity_futures.begin();
          it != cavity_futures.end(); it++)
    {
      Cavity cavity = it->get_result<Cavity>(); 
      if (cavity.valid_cavity)
      {
        region_reqs[0].region = cavity.needed_triangles;
        size_t buffer_size = cavity.legion_buffer_size();
        void *buffer = malloc(buffer_size);
        cavity.legion_serialize(buffer);
        bad_futures.push_back(runtime->execute_task(ctx, PROCESS_CAVITY, index_reqs, field_reqs,
                                                    region_reqs, TaskArgument(buffer, buffer_size)));
        free(buffer);
        runtime->destroy_index_partition(ctx, cavity.needed_part.get_index_partition());
      }
    }
    cavity_futures.clear();

    // See if we have any more bad triangles to handle
    for (std::vector<Future>::iterator it = bad_futures.begin();
          it != bad_futures.end(); it++)
    {
      SerializableVector<BadTriangle> more_bad = it->get_result<SerializableVector<BadTriangle> >(); 
      bad_triangles.insert(bad_triangles.end(),more_bad.impl.begin(),more_bad.impl.end()); 
    }
    bad_futures.clear();

    // Check to see how many bad triangles we have, if it is more than the chunk size,
    // spread it out into additional tasks
    region_reqs.clear();
    region_reqs.push_back(triangles.get_region_requirement(READ_WRITE,SIMULTANEOUS));
    log_delaunay(LEVEL_INFO,"Delaunay recurse now has %ld bad triangles %p", bad_triangles.size(), ctx);
    while (bad_triangles.size() > max_bad)
    {
      unsigned to_pull = (bad_triangles.size() >= (2*max_bad)) ? max_bad : (bad_triangles.size() - max_bad);
      SerializableVector<BadTriangle> to_send;
      for (unsigned idx = 0; idx < to_pull; idx++)
      {
        to_send.impl.push_back(bad_triangles.back());
        bad_triangles.pop_back();
      }
      size_t buffer_size = sizeof(size_t) + to_send.legion_buffer_size();
      void *buffer = malloc(buffer_size);
      *((size_t*)buffer) = max_bad;
      to_send.legion_serialize((void*)(((char*)buffer)+sizeof(size_t)));
      runtime->execute_task(ctx, DELAUNAY_RECURSE, index_reqs, field_reqs,
                            region_reqs, TaskArgument(buffer, buffer_size));
    }
    log_delaunay(LEVEL_INFO,"Continuing delaunay recurse with %ld bad triangles %p", bad_triangles.size(), ctx);
  }
}

/**
 * Compute the cavity for this bad triangle and create a partition
 * for handling it.  Then return the one logical region in the partition
 * for the cavity.
 */
Cavity compute_cavity(const void *args, size_t arglen,
                      const std::vector<RegionRequirement> &logical_regions,
                      const std::vector<PhysicalRegion> &physical_regions,
                      Context ctx, HighLevelRuntime *runtime)
{
  BadTriangle bad_tri = *((BadTriangle*)args);
  LegionVector<Triangle> triangles(runtime, ctx, logical_regions[0].region, 
                      physical_regions[0], logical_regions[0].instance_fields[0]);
  Cavity result;
  // Only need to initialize this cavity if it is still a bad triangle
  if (result.is_valid_cavity(bad_tri,triangles))
  {
    result.initialize(bad_tri.ptr, triangles);
    result.build(logical_regions[0].region, triangles, runtime, ctx);
  }
  return result;
}

/**
 * Check to see if all the triangles in the cavity are still valid.
 * If not we're done.  Otherwise, fix the cavity and allocate new
 * triangles.  Put any new bad triangles in the queue of bad triangles.
 */
SerializableVector<BadTriangle> process_cavity(const void *args, size_t arglen,
                                  const std::vector<RegionRequirement> &logical_regions,
                                  const std::vector<PhysicalRegion> &physical_regions,
                                  Context ctx, HighLevelRuntime *runtime)
{
  Cavity cavity;
  cavity.legion_deserialize(args);
  //printf("Processing cavity at %d\n",cavity.center_ptr.value);
  LegionVector<Triangle> needed_triangles(runtime, ctx, logical_regions[0].region,
                      physical_regions[0], logical_regions[0].instance_fields[0]);
  SerializableVector<BadTriangle> new_bad_triangles;
  cavity.update(needed_triangles, new_bad_triangles); 
  return new_bad_triangles;
}

// Implementation of methods from Cavity 
Cavity::Cavity(void)
  : needed_triangles(LogicalRegion::NO_REGION), needed_part(LogicalPartition::NO_PART),
    center_ptr(ptr_t::nil()) { }

void Cavity::initialize(ptr_t start_ptr, LegionVector<Triangle> &triangles)
{
  cavity_tris.clear();
  border.clear();
  center_ptr = start_ptr;
  Triangle *center = &triangles[center_ptr];
  if (center->is_obtuse())
  {
    center_ptr = center->get_obtuse_opposite();
    center = &triangles[center_ptr];
  }
}

void Cavity::build(LogicalRegion log_triangles, LegionVector<Triangle> &triangles,
                   HighLevelRuntime *rt, Context ctx)
{
  std::set<ptr_t> frontier;
  frontier.insert(center_ptr);

  Coloring cavity_coloring;
  ColoredPoints<ptr_t> &cavity_points = cavity_coloring[0];
  cavity_points.points.insert(center_ptr);

  const Triangle &center = triangles[center_ptr];
  while (!frontier.empty())
  {
    ptr_t curr_ptr;
    {
      std::set<ptr_t>::iterator curr_it = frontier.begin();
      curr_ptr = *curr_it;
      frontier.erase(curr_it);
    }
    cavity_points.points.insert(curr_ptr);
    const Triangle &current = triangles[curr_ptr];
    cavity_tris.push_back(CavityTriangle(curr_ptr,current.generation,true/*inside*/));
    for (unsigned idx = 0; idx < current.num_neighbors; idx++)
    {
      ptr_t next_ptr = current.neighbors[idx];
      const Triangle &next = triangles[next_ptr];
      if ((!((center.dim == 2) && (next.dim == 2) && (next_ptr != center_ptr))) && is_member(center.center, next))
      {
        if ((next.dim == 2) && (center.dim != 2))
        {
          initialize(next_ptr, triangles);
          build(log_triangles, triangles, rt, ctx);
          return;
        }
        else
        {
          if (cavity_points.points.find(next_ptr) == cavity_points.points.end())
            frontier.insert(next_ptr);
        }
      }
      else
      {
        // Otherwise it's on the border so we still need the point
        cavity_points.points.insert(next_ptr);
        border.push_back(next.compute_border_edge(current));
        cavity_tris.push_back(CavityTriangle(next_ptr,next.generation,false/*inside*/));
      }
    }
  }
  // Now make the partition for this cavity
  {
    //printf("Cavity has %ld triangles\n", cavity_points.points.size());
    IndexPartition ip = rt->create_index_partition(ctx, log_triangles.get_index_space(),
                                                   cavity_coloring, true/*disjoint*/);
    this->needed_part = rt->get_logical_partition(ctx, log_triangles, ip);
    this->needed_triangles = rt->get_logical_subregion_by_color(ctx, this->needed_part, 0/*region color*/);
  }
}

bool Cavity::is_member(const Point &center, const Triangle &tri) const
{
  return tri.in_circle(center);
}

void Cavity::update(LegionVector<Triangle> &triangles, SerializableVector<BadTriangle> &new_bad_triangles)
{
  // First check to see if we can fix the cavity or whether any of triangles
  // have been invalidated
  bool all_valid = true;
  for (std::vector<CavityTriangle>::const_iterator it = cavity_tris.begin();
        it != cavity_tris.end(); it++)
  {
    const Triangle &tri = triangles[it->ptr];
    if (tri.generation > it->gen)
    {
      all_valid = false;
      break;
    }
  }
  if (!all_valid)
  {
    // Check to see if our initial bad_ptr is still a valid triangle,
    // if it is, put it back on the list, otherwise we're done
    const Triangle &init_tri = triangles[init_bad.ptr];
    if (init_tri.generation == init_bad.gen)
      new_bad_triangles.impl.push_back(init_bad);
    return;
  }
  // Otherwise, we can now fix this cavity, make new triangles
  Point center_point;
  unsigned center_dim;
  Point p[2]; // for cases of dim2
  // Forget the center tri after this point since it is going
  // to get overwritten
  {
    const Triangle &center_tri = triangles[center_ptr];
    center_point = center_tri.center;
    center_dim = center_tri.dim;
    if (center_dim == 2)
    {
      p[0] = center_tri.points[0];
      p[1] = center_tri.points[1];
    }
  }
  unsigned search_idx = 0; 
  std::vector<Triangle> new_triangles(border.size());
  for (unsigned idx = 0; idx < border.size(); idx++)
  {
    new_triangles[idx] = Triangle(center_point, border[idx].edge[0], border[idx].edge[1]); 
    // Figure out where to write it.  See if we can overwrite on
    // of the triangles we just destroyed, otherwise, allocate
    // a new triangle.
    int index = -1;
    for ( /*nothing*/; search_idx < cavity_tris.size(); search_idx++)
    {
      if (cavity_tris[search_idx].inside)
      {
        index = search_idx++;
        break;
      }
    }
    if (index != -1)
    {
      // Re-use an existing triangle
      new_triangles[idx].my_ptr = cavity_tris[index].ptr;
      new_triangles[idx].generation = cavity_tris[index].gen + 1; // increment the generation
    }
    else
    {
      // Allocate a new spot for the triangle
      ptr_t dest_ptr = triangles.push_back(new_triangles[idx]);
      new_triangles[idx].my_ptr = dest_ptr;
      new_triangles[idx].generation = 0;
    }
  } 
  // If we built around an edge, need to make new triangles
  if (center_dim == 2)
  {
    for (int i = 0; i < 2; i++)
    {
      Triangle edge_triangle(center_point,p[i]);
      int index = -1;
      for (/*nothing*/; search_idx < cavity_tris.size(); search_idx++)
      {
        if (cavity_tris[search_idx].inside)
        {
          index = search_idx++;
          break;
        }
      }
      if (index != -1)
      {
        edge_triangle.my_ptr = cavity_tris[index].ptr;
        edge_triangle.generation = cavity_tris[index].gen + 1;
      }
      else
      {
        ptr_t dest_ptr = triangles.push_back(edge_triangle);
        edge_triangle.my_ptr = dest_ptr;
        edge_triangle.generation = 0;
      }
      // Find which of the other new triangles we are edges with
      for (unsigned idx = 0; idx < new_triangles.size(); idx++)
      {
        if (new_triangles[idx].find_index(p[i]) != -1)
        {
          edge_triangle.neighbors[0] = new_triangles[idx].my_ptr;
          break;
        }
      }
      triangles[edge_triangle.my_ptr] = edge_triangle;
      new_triangles.push_back(edge_triangle);
    }
  }
  // now that we have got everyone's pointer, we need to hook up all the nieghbor pointers
  for (unsigned idx = 0; idx < new_triangles.size(); idx++)
  {
    // No need to do this for dim2 triangles, they've already been handled
    if (new_triangles[idx].dim == 2)
      continue; 
#ifdef DEBUG_DELAUNAY
    assert(border[idx].update_triangle != ptr_t::nil());
#endif
    Triangle &border_tri = triangles[border[idx].update_triangle];
#ifdef DEBUG_DELAUNAY
    assert(new_triangles[idx].my_ptr != ptr_t::nil());
    assert(border[idx].update_triangle != new_triangles[idx].my_ptr);
#endif
    border_tri.neighbors[border[idx].neighbor_idx] = new_triangles[idx].my_ptr;
    // Update the border triangle generation since it's been changed
    border_tri.generation++;
    // We know from how we called the constructor before that the neighbor pointer
    // for the border triangle is location 0 = (1+2)%3
    new_triangles[idx].neighbors[0] = border[idx].update_triangle;
    // Now get the neighbors for the other two edges
    {
      const Point &search_point = new_triangles[idx].points[1];
      ptr_t neighbor_ptr = ptr_t::nil();
      for (unsigned nidx = 0; nidx < new_triangles.size(); nidx++)
      {
        if (nidx == idx) 
          continue;
        if (new_triangles[nidx].find_index(search_point) != -1)
        {
          neighbor_ptr = new_triangles[nidx].my_ptr;
          break;
        }
      }
#ifdef DEBUG_DELAUNAY
      assert(neighbor_ptr != ptr_t::nil());
      assert(new_triangles[idx].my_ptr != neighbor_ptr);
#endif
      new_triangles[idx].neighbors[1] = neighbor_ptr;
    }
    // Do it again for the other point
    {
      const Point &search_point = new_triangles[idx].points[2];
      ptr_t neighbor_ptr = ptr_t::nil();
      for (unsigned nidx = 0; nidx < new_triangles.size(); nidx++)
      {
        if (nidx == idx) 
          continue;
        if (new_triangles[nidx].find_index(search_point) != -1)
        {
          neighbor_ptr = new_triangles[nidx].my_ptr;
          break;
        }
      }
#ifdef DEBUG_DELAUNAY
      assert(neighbor_ptr != ptr_t::nil());
      assert(new_triangles[idx].my_ptr != neighbor_ptr);
#endif
      new_triangles[idx].neighbors[2] = neighbor_ptr;
    }
    // now we can write the triangle into the triangle region
    ptr_t dest_ptr = new_triangles[idx].my_ptr;
    triangles[dest_ptr] = new_triangles[idx];
    // Finally check to see if the new triangle we just created
    // is a bad triangle in which case add it to the list of bad triangles
    if (new_triangles[idx].is_bad_triangle())
      new_bad_triangles.impl.push_back(BadTriangle(new_triangles[idx].my_ptr,new_triangles[idx].generation));
  }
#ifdef DEBUG_DELAUNAY
  if (center_dim == 2)
  {
    size_t num_new = new_triangles.size();
    assert(triangles[new_triangles[num_new-2].my_ptr].neighbors[0] != ptr_t::nil());
    assert(triangles[new_triangles[num_new-1].my_ptr].neighbors[0] != ptr_t::nil());
  }
#endif
}

bool Cavity::is_valid_cavity(BadTriangle init, LegionVector<Triangle> &triangles)
{
  init_bad = init;
  const Triangle &tri = triangles[init_bad.ptr];
  if (tri.generation == init_bad.gen)
    valid_cavity = true;
  else
    valid_cavity = false;
  return valid_cavity;
}

size_t Cavity::legion_buffer_size(void) const
{
  buffer_size = 0;
  buffer_size += sizeof(valid_cavity);
  buffer_size += sizeof(needed_triangles);
  buffer_size += sizeof(needed_part);
  buffer_size += sizeof(center_ptr);
  buffer_size += sizeof(init_bad);
  buffer_size += sizeof(size_t); // number of cavity tris
  buffer_size += (cavity_tris.size() * sizeof(CavityTriangle));
  buffer_size += sizeof(size_t); // number of BorderElements
  buffer_size += (border.size() * sizeof(BorderEdge));
  return buffer_size;
}

void Cavity::legion_serialize(void *buffer)
{
  char *ptr = (char*)buffer;
  *((bool*)ptr) = valid_cavity;
  ptr += sizeof(bool);
  *((LogicalRegion*)ptr) = needed_triangles;
  ptr += sizeof(LogicalRegion);
  *((LogicalPartition*)ptr) = needed_part;
  ptr += sizeof(LogicalPartition);
  *((ptr_t*)ptr) = center_ptr;
  ptr += sizeof(ptr_t);
  *((BadTriangle*)ptr) = init_bad;
  ptr += sizeof(BadTriangle);
  *((size_t*)ptr) = cavity_tris.size();
  ptr += sizeof(size_t);
  for (std::vector<CavityTriangle>::const_iterator it = cavity_tris.begin();
        it != cavity_tris.end(); it++)
  {
    *((CavityTriangle*)ptr) = *it;
    ptr += sizeof(CavityTriangle);
  }
  *((size_t*)ptr) = border.size();
  ptr += sizeof(size_t);
  for (std::vector<BorderEdge>::const_iterator it = border.begin();
        it != border.end(); it++)
  {
    *((BorderEdge*)ptr) = *it;
    ptr += sizeof(BorderEdge);
  }
}

void Cavity::legion_deserialize(const void *buffer)
{
  const char *ptr = (const char*)buffer;
  valid_cavity = *((const bool*)ptr);
  ptr += sizeof(bool);
  needed_triangles = *((const LogicalRegion*)ptr);
  ptr += sizeof(LogicalRegion);
  needed_part = *((const LogicalPartition*)ptr);
  ptr += sizeof(LogicalPartition);
  center_ptr = *((const ptr_t*)ptr);
  ptr += sizeof(ptr_t);
  init_bad = *((const BadTriangle*)ptr);
  ptr += sizeof(BadTriangle);
  size_t num_tris = *((const size_t*)ptr);
  ptr += sizeof(size_t);
  for (unsigned idx = 0; idx < num_tris; idx++)
  {
    cavity_tris.push_back(*((const CavityTriangle*)ptr));
    ptr += sizeof(CavityTriangle);
  }
  size_t num_border = *((const size_t*)ptr);
  ptr += sizeof(size_t);
  for (unsigned idx = 0; idx < num_border; idx++)
  {
    border.push_back(*((const BorderEdge*)ptr));
    ptr += sizeof(BorderEdge);
  }
}

class DelaunayMapper : public SharedMapper {
public:
  DelaunayMapper(Machine *machine, HighLevelRuntime *rt, Processor proc)
    : SharedMapper(machine, rt, proc) { }
public:
  virtual void permit_task_steal(Processor thief, const std::vector<const Task*> &tasks,
                                      std::set<const Task*> &to_steal)
  {
    // Only steal dealunay recurse tasks
    if (stealing_enabled)
    {
      if (max_steals_per_theft == 0)
        return;
      unsigned total_stolen = 0;
      // Go through and see if we can find any delaunay recurse tasks to steal
      for (std::vector<const Task*>::const_iterator it = tasks.begin();
            it != tasks.end(); it++)
      {
        if (((*it)->task_id == DELAUNAY_RECURSE) && ((*it)->steal_count < max_steal_count))
        {
          to_steal.insert(*it);
          total_stolen++;
          if (total_stolen == max_steals_per_theft)
            break;
        }
      }
    }
  }
};

void registration_func(Machine *machine, HighLevelRuntime *rt, 
                       const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(new DelaunayMapper(machine, rt, *it), *it);
  }
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_registration_callback(registration_func);

  HighLevelRuntime::register_single_task<legion_main>
    (LEGION_MAIN, Processor::LOC_PROC, false/*leaf*/, "legion_main");
  HighLevelRuntime::register_single_task<refine_mesh>
    (REFINE_MESH, Processor::LOC_PROC, false/*leaf*/, "refine_mesh");
  HighLevelRuntime::register_single_task<delaunay_recurse>
    (DELAUNAY_RECURSE, Processor::LOC_PROC, false/*leaf*/, "delaunay_recurse");
  HighLevelRuntime::register_single_task<Cavity,compute_cavity>
    (COMPUTE_CAVITY, Processor::LOC_PROC, false/*leaf*/, "compute_cavity");
  HighLevelRuntime::register_single_task<SerializableVector<BadTriangle>,process_cavity>
    (PROCESS_CAVITY, Processor::LOC_PROC, true/*leaf*/, "process_cavity");

  HighLevelRuntime::set_top_level_task_id(LEGION_MAIN);

  return HighLevelRuntime::start(argc, argv);
}

// Helper method for Triangles
static inline float distance_squared(const Point &p1, const Point &p2)
{
  __m128 point1 = _mm_set_ps(0.0f,p1.coords[2],p1.coords[1],p1.coords[0]);
  __m128 point2 = _mm_set_ps(0.0f,p2.coords[2],p2.coords[1],p2.coords[0]);
  __m128 diff = _mm_sub_ps(point1,point2);
  __m128 dotp = _mm_dp_ps(diff,diff,113);
  return _mm_cvtss_f32(dotp);
}

// Function for Triangle
Triangle::Triangle(const Point &p1, const Point &p2)
  : bad(false), obtuse(false), dim(2), 
    my_ptr(ptr_t::nil()), generation(0), num_neighbors(1)
{
#ifdef DEBUG_DELAUNAY
  assert(p1 != p2);
#endif
  points[0] = p1;
  points[1] = p2;
  neighbors[0] = ptr_t::nil();
  for (int idx = 0; idx < 3; idx++)
    center.coords[idx] = 0.5 * (p1.coords[idx] + p2.coords[idx]);
  radius_squared = distance_squared(center, p1);
}

Triangle::Triangle(const Point &p1, const Point &p2, const Point &p3)
  : bad(false), obtuse(false), dim(3), 
    my_ptr(ptr_t::nil()), generation(0), num_neighbors(3)
{
#ifdef DEBUG_DELAUNAY
  assert(p1 != p2);
  assert(p2 != p3);
  assert(p1 != p3);
#endif
  points[0] = p1;
  points[1] = p2;
  points[2] = p3;
  neighbors[0] = ptr_t::nil();
  neighbors[1] = ptr_t::nil();
  neighbors[2] = ptr_t::nil();
  for (unsigned idx = 0; idx < 3; idx++)
  {
    float angle = get_angle(idx);
    if (angle > 90.1)
    {
      obtuse = true;
      obtuse_idx = idx;
    }
    else if (angle < MINANGLE)
    {
      bad = true;
    }
  }
#if 0
  for (unsigned idx = 0; idx < 3; idx++)
    printf("Point %d: %f %f %f\n",idx,points[idx].coords[0],points[idx].coords[1],points[idx].coords[2]);
  {
    Point x(p2.coords[0]-p1.coords[0],p2.coords[1]-p1.coords[1],p2.coords[2]-p1.coords[2]);
    Point y(p3.coords[0]-p1.coords[0],p3.coords[1]-p1.coords[1],p3.coords[2]-p1.coords[2]);
    double xlen = sqrt(x.coords[0]*x.coords[0] + x.coords[1]*x.coords[1] + x.coords[2]*x.coords[2]);
    double ylen = sqrt(y.coords[0]*y.coords[0] + y.coords[1]*y.coords[1] + y.coords[2]*y.coords[2]);
    double cosine = (x.coords[0]*y.coords[0] + x.coords[1]*y.coords[1] + x.coords[2]*y.coords[2])/ (xlen * ylen);
    double sine_sq = 1.0 - cosine * cosine;
    double plen = ylen / xlen;
    printf("cosine: %lf  sine_sq: %lf  plen: %lf\n",cosine, sine_sq, plen);
    double s = plen * cosine;
    double t = plen * sine_sq;
    double wp = (plen - cosine) / (2 * t);
    double wb = 0.5 - (wp * s);
    double tmp = 1 - wb - wp;
    printf("tmp: %f\n", tmp);
    Point t1(p1.coords[0]*tmp,p1.coords[1]*tmp,p1.coords[2]*tmp);
    Point t2(t1.coords[0] + wb*p2.coords[0], t1.coords[1] + wb*p2.coords[1], t1.coords[2] + wb*p2.coords[2]);
    Point cent(t2.coords[0] + wp*p3.coords[0], t2.coords[1] + wp*p3.coords[1], t2.coords[2] + wp*p3.coords[2]);
    printf("Center %f %f %f\n", cent.coords[0],cent.coords[1],cent.coords[2]);
    Point diff(cent.coords[0]-p1.coords[0],cent.coords[1]-p1.coords[1],cent.coords[2]-p1.coords[2]);
    double distance_sq = (diff.coords[0]*diff.coords[0] + diff.coords[1]*diff.coords[1] + diff.coords[2]*diff.coords[2]);
    printf("Distance sq: %f\n", distance_sq);
  }
#endif
  __m128 point1 = _mm_set_ps(0.0f,p1.coords[2],p1.coords[1],p1.coords[0]);
  __m128 point2 = _mm_set_ps(0.0f,p2.coords[2],p2.coords[1],p2.coords[0]);
  __m128 point3 = _mm_set_ps(0.0f,p3.coords[2],p3.coords[1],p3.coords[0]);
  __m128 x = _mm_sub_ps(point2,point1);
  __m128 y = _mm_sub_ps(point3,point1);
  __m128 xlensq = _mm_dp_ps(x,x,113);
  __m128 ylensq = _mm_dp_ps(y,y,113);
  __m128 cosine = _mm_mul_ss(_mm_dp_ps(x,y,113),_mm_rsqrt_ss(_mm_mul_ss(xlensq,ylensq)));
  __m128 sine_sq = _mm_sub_ss(_mm_set1_ps(1.0),_mm_mul_ss(cosine,cosine));
  __m128 plen = _mm_rsqrt_ss(_mm_div_ss(xlensq,ylensq));
  //printf("cosine: %f  sine_sq: %f  plen: %f\n", _mm_cvtss_f32(cosine), _mm_cvtss_f32(sine_sq), _mm_cvtss_f32(plen));
  __m128 s = _mm_mul_ss(plen,cosine);
  __m128 t = _mm_mul_ss(plen,sine_sq);
  __m128 wp = _mm_div_ss(_mm_sub_ss(plen,cosine),_mm_mul_ss(_mm_set1_ps(2.0),t));
  __m128 wb = _mm_sub_ss(_mm_set1_ps(0.5),_mm_mul_ss(wp,s));
  __m128 tmp = _mm_sub_ss(_mm_sub_ss(_mm_set1_ps(1.0),wb),wp);
  //printf("tmp: %f\n",_mm_cvtss_f32(tmp));
  tmp = _mm_shuffle_ps(tmp,tmp,0);
  //printf("tmp: %f %f %f\n", _mm_cvtss_f32(tmp), _mm_cvtss_f32(_mm_shuffle_ps(tmp,tmp,1)), _mm_cvtss_f32(_mm_shuffle_ps(tmp,tmp,2)));
  tmp = _mm_mul_ps(point1,tmp);
  tmp = _mm_add_ps(tmp,_mm_mul_ps(point2,_mm_shuffle_ps(wb,wb,0)));
  tmp = _mm_add_ps(tmp,_mm_mul_ps(point3,_mm_shuffle_ps(wp,wp,0)));
  center.coords[0] = _mm_cvtss_f32(tmp);
  center.coords[1] = _mm_cvtss_f32(_mm_shuffle_ps(tmp,tmp,1));
  center.coords[2] = _mm_cvtss_f32(_mm_shuffle_ps(tmp,tmp,2));
  //printf("center %f %f %f\n",center.coords[0],center.coords[1],center.coords[2]);
  __m128 diff = _mm_sub_ps(tmp,point1);
  __m128 dotp = _mm_dp_ps(diff,diff,113);
  radius_squared = _mm_cvtss_f32(dotp);
  //printf("Raidus squared: %f\n", radius_squared);
}

void Triangle::set_adjacent(const Point &p1, const Point &p2, ptr_t adjacent)
{
  if (dim == 2)
    neighbors[0] = adjacent;
  else
  {
    int idx1 = find_index(p1);
    int idx2 = find_index(p2);
#ifdef DEBUG_DELAUNAY
    assert(idx1 != -1);
    assert(idx2 != -1);
#endif
    int idx = (idx1 + idx2) % 3;
    neighbors[idx] = adjacent;
  }
}

float Triangle::get_angle(unsigned idx) const
{
  float result = 0.0f;
  switch (idx)
  {
    case 0:
      result = compute_angle(points[1],points[0],points[2]);
      break;
    case 1:
      result = compute_angle(points[1],points[2],points[0]);
      break;
    case 2:
      result = compute_angle(points[0],points[1],points[2]);
      break;
    default:
      assert(false);
  }
  return result;
}

/*static*/ float Triangle::compute_angle(const Point &p1, const Point &p2, const Point &p3)
{
  __m128 point1 = _mm_set_ps(0.0f,p1.coords[2],p1.coords[1],p1.coords[0]);
  __m128 point2 = _mm_set_ps(0.0f,p2.coords[2],p2.coords[1],p2.coords[0]);
  __m128 point3 = _mm_set_ps(0.0f,p3.coords[2],p3.coords[1],p3.coords[0]);
  __m128 diffa = _mm_sub_ps(point1,point2);
  __m128 diffc = _mm_sub_ps(point3,point2);
  __m128 dotp1 = _mm_dp_ps(diffa,diffc,113);
  __m128 dotp2 = _mm_dp_ps(diffa,diffa,113);
  __m128 dotp3 = _mm_dp_ps(diffc,diffc,113);
  __m128 d = _mm_mul_ss(dotp1,_mm_rsqrt_ss(_mm_mul_ss(dotp2,dotp3)));
  float result = _mm_cvtss_f32(d);
  return (180.0f/PI) * acos(result);
}

bool Triangle::in_circle(const Point &point) const
{
  return (distance_squared(point,center) <= radius_squared); 
}

BorderEdge Triangle::compute_border_edge(const Triangle &other) const
{
  BorderEdge result;
  result.update_triangle = my_ptr; // this is the triangle that needs to be updated
  // figure out which two points we share in common and the neighbor idx
  int idx1 = -1, idx2 = -1;
  for (unsigned i = 0; i < other.dim; i++)
  {
    int idx = find_index(other.points[i]); 
    if (idx != -1)
    {
      if (idx1 == -1)
        idx1 = idx;
      else
      {
#ifdef DEBUG_DELAUNAY
        assert(idx2 == -1);
#endif
        idx2 = idx;
      }
    }
  }
#ifdef DEBUG_DELAUNAY
  assert(idx1 != -1);
  assert(idx2 != -1);
#endif
  result.edge[0] = points[idx1];
  result.edge[1] = points[idx2];
  if (dim == 3)
    result.neighbor_idx = (idx1 + idx2) % 3;
  else
    result.neighbor_idx = 0;
  return result;
}

// Find bad triangles
void find_bad_triangles(LegionVector<Triangle> &triangles, LegionVector<BadTriangle> &bad_triangles)
{
  for (LegionVector<Triangle>::iterator it = triangles.begin();
        it != triangles.end(); it++)
  {
    const Triangle &tri = *it;
    if (tri.is_bad_triangle())
      bad_triangles.push_back(BadTriangle(tri.my_ptr,tri.generation));
  }
}

// Functions for loading the input 
static inline void add_adjacent(std::map<std::pair<unsigned,unsigned>,std::pair<Triangle*,Triangle*> > &adjacent,
                                int n1, int n2, Triangle *tri)
{
  std::map<std::pair<unsigned,unsigned>,std::pair<Triangle*,Triangle*> >::iterator finder 
    = adjacent.find(std::pair<unsigned,unsigned>(n1,n2));
  if (finder != adjacent.end())
  {
#ifdef DEBUG_DELAUNAY
    assert(finder->second.second == NULL);
#endif
    finder->second.second = tri;
  }
  else
  {
    finder = adjacent.find(std::pair<unsigned,unsigned>(n2,n1));
    if (finder != adjacent.end())
    {
#ifdef DEBUG_DELAUNAY
      assert(finder->second.second == NULL);
#endif
      finder->second.second = tri;
    }
    else
    {
      adjacent[std::pair<unsigned,unsigned>(n1,n2)] = std::pair<Triangle*,Triangle*>(tri,NULL);
    }
  }
}

void load_triangles(const char *base_name, LegionVector<Triangle> &triangles)
{
  Point *points = NULL;
  // Read the points first
  {
    char file_name[128];
    strcpy(file_name,base_name);
    strcat(file_name,".node");
    FILE *point_file = fopen(file_name,"r");
    assert(point_file != NULL);
    int ntups, other1, other2, other3;
    assert(fscanf(point_file,"%d %d %d %d", &ntups, &other1, &other2, &other3) > 0);
    points = (Point*)malloc(ntups * sizeof(Point));
    for (int i = 0; i < ntups; i++)
    {
      int index;
      float x, y, z;
      assert(fscanf(point_file,"%d %f %f %f", &index, &x, &y, &z) > 0);
      points[index].coords[0] = x;
      points[index].coords[1] = y;
      points[index].coords[2] = 0.0f;
    }
    assert(fclose(point_file) == 0);
  }
  std::map<std::pair<unsigned,unsigned>,std::pair<Triangle*,Triangle*> > adjacent;
  // Read the elements
  {
    char file_name[128];
    strcpy(file_name,base_name);
    strcat(file_name,".ele");
    FILE *elem_file = fopen(file_name,"r");
    assert(elem_file != NULL);
    int nelems, other1, other2;
    assert(fscanf(elem_file,"%d %d %d", &nelems, &other1, &other2) > 0);
    for (int i = 0; i < nelems; i++)
    {
      int index, n1, n2, n3;
      assert(fscanf(elem_file,"%d %d %d %d", &index, &n1, &n2, &n3) > 0);
      ptr_t ptr = triangles.push_back(Triangle(points[n1], points[n2], points[n3]));
      triangles[ptr].my_ptr = ptr;
      add_adjacent(adjacent,n1,n2,&(triangles[ptr]));
      add_adjacent(adjacent,n2,n3,&(triangles[ptr]));
      add_adjacent(adjacent,n3,n1,&(triangles[ptr]));
    }
    assert(fclose(elem_file) == 0);
  }
  // Read the poly edges
  {
    char file_name[128];
    strcpy(file_name,base_name);
    strcat(file_name,".poly");
    FILE *seg_file = fopen(file_name,"r");
    assert(seg_file != NULL);
    int nsegs, other1, other2, other3, other4, other5;
    assert(fscanf(seg_file,"%d %d %d %d %d %d", &other1, &other2, &other3, &other4, &nsegs, &other5) > 0);
    for (int i = 0; i < nsegs; i++)
    {
      int index, n1, n2, other;
      assert(fscanf(seg_file,"%d %d %d %d", &index, &n1, &n2, &other) > 0);
      ptr_t ptr = triangles.push_back(Triangle(points[n1],points[n2]));
      triangles[ptr].my_ptr = ptr;
      add_adjacent(adjacent,n1,n2,&(triangles[ptr]));
    }
    assert(fclose(seg_file) == 0);
  }
  // Sanity check, everything in the adjacency map should have two things
  // and then update the triangles with the adjacency information
  for (std::map<std::pair<unsigned,unsigned>,std::pair<Triangle*,Triangle*> >::const_iterator it
        = adjacent.begin(); it != adjacent.end(); it++)
  {
    assert(it->first.first != it->first.second);
    assert(it->second.first != NULL);
    assert(it->second.second != NULL);
    it->second.first->set_adjacent(points[it->first.first],points[it->first.second],it->second.second->my_ptr);
    it->second.second->set_adjacent(points[it->first.first],points[it->first.second],it->second.first->my_ptr);
  }
  for (LegionVector<Triangle>::iterator it = triangles.begin();
        it != triangles.end(); it++)
  {
    const Triangle &tri = *it;
    for (unsigned idx = 0; idx < tri.num_neighbors; idx++)
      assert(tri.neighbors[idx] != ptr_t::nil());
  }
  // now we can delete the points
  free(points);
}

// EOF

