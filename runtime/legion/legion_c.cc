/* Copyright 2018 Stanford University
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

#include "legion.h"
#include "legion/legion_c.h"
#include "legion/legion_c_util.h"
#ifdef REALM_USE_LLVM
#include "realm/llvmjit/llvmjit.h"
#endif
#ifdef REALM_USE_PYTHON
#include "realm/python/python_source.h"
#endif

// Disable deprecated warnings in this file since we are also
// trying to maintain backwards compatibility support for older
// interfaces here in the C API
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wdeprecated"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef __clang__
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

using namespace Legion;
using namespace Legion::Mapping;
using namespace Legion::Mapping::Utilities;
typedef Point<1,coord_t> Point1D;
typedef Point<2,coord_t> Point2D;
typedef Point<3,coord_t> Point3D;
typedef Rect<1,coord_t> Rect1D;
typedef Rect<2,coord_t> Rect2D;
typedef Rect<3,coord_t> Rect3D;
typedef Transform<1,1,coord_t> Transform1x1;
typedef Transform<1,2,coord_t> Transform1x2;
typedef Transform<1,3,coord_t> Transform1x3;
typedef Transform<2,1,coord_t> Transform2x1;
typedef Transform<2,2,coord_t> Transform2x2;
typedef Transform<2,3,coord_t> Transform2x3;
typedef Transform<3,1,coord_t> Transform3x1;
typedef Transform<3,2,coord_t> Transform3x2;
typedef Transform<3,3,coord_t> Transform3x3;
typedef AffineTransform<1,1,coord_t> AffineTransform1x1;
typedef AffineTransform<1,2,coord_t> AffineTransform1x2;
typedef AffineTransform<1,3,coord_t> AffineTransform1x3;
typedef AffineTransform<2,1,coord_t> AffineTransform2x1;
typedef AffineTransform<2,2,coord_t> AffineTransform2x2;
typedef AffineTransform<2,3,coord_t> AffineTransform2x3;
typedef AffineTransform<3,1,coord_t> AffineTransform3x1;
typedef AffineTransform<3,2,coord_t> AffineTransform3x2;
typedef AffineTransform<3,3,coord_t> AffineTransform3x3;

// -----------------------------------------------------------------------
// Pointer Operations
// -----------------------------------------------------------------------

legion_ptr_t
legion_ptr_nil(void)
{
  legion_ptr_t ptr;
  ptr.value = -1LL;
  return ptr;
}

bool
legion_ptr_is_null(legion_ptr_t ptr)
{
  return ptr.value == -1LL;
}

legion_ptr_t
legion_ptr_safe_cast(legion_runtime_t runtime_,
                     legion_context_t ctx_,
                     legion_ptr_t pointer_,
                     legion_logical_region_t region_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  ptr_t pointer = CObjectWrapper::unwrap(pointer_);
  LogicalRegion region = CObjectWrapper::unwrap(region_);

  ptr_t result = runtime->safe_cast(ctx, pointer, region);
  return CObjectWrapper::wrap(result);
}

// -----------------------------------------------------------------------
// Domain Operations
// -----------------------------------------------------------------------

legion_domain_t
legion_domain_from_rect_1d(legion_rect_1d_t r_)
{
  Rect1D r = CObjectWrapper::unwrap(r_);

  return CObjectWrapper::wrap(Domain(r));
}

legion_domain_t
legion_domain_from_rect_2d(legion_rect_2d_t r_)
{
  Rect2D r = CObjectWrapper::unwrap(r_);

  return CObjectWrapper::wrap(Domain(r));
}

legion_domain_t
legion_domain_from_rect_3d(legion_rect_3d_t r_)
{
  Rect3D r = CObjectWrapper::unwrap(r_);

  return CObjectWrapper::wrap(Domain(r));
}

legion_domain_t
legion_domain_from_index_space(legion_runtime_t runtime_,
                               legion_index_space_t is_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexSpace is = CObjectWrapper::unwrap(is_);

  return CObjectWrapper::wrap(runtime->get_index_space_domain(is));
}

legion_rect_1d_t
legion_domain_get_rect_1d(legion_domain_t d_)
{
  Domain d = CObjectWrapper::unwrap(d_);
  Rect1D r = d;

  return CObjectWrapper::wrap(r);
}

legion_rect_2d_t
legion_domain_get_rect_2d(legion_domain_t d_)
{
  Domain d = CObjectWrapper::unwrap(d_);
  Rect2D r = d;

  return CObjectWrapper::wrap(r);
}

legion_rect_3d_t
legion_domain_get_rect_3d(legion_domain_t d_)
{
  Domain d = CObjectWrapper::unwrap(d_);
  Rect3D r = d;

  return CObjectWrapper::wrap(r);
}

bool
legion_domain_is_dense(legion_domain_t d_)
{
  Domain d = CObjectWrapper::unwrap(d_);

  return d.dense();
}

legion_rect_1d_t
legion_domain_get_bounds_1d(legion_domain_t d_)
{
  Domain d = CObjectWrapper::unwrap(d_);
  DomainT<1,coord_t> space = d;

  return CObjectWrapper::wrap(space.bounds);
}

legion_rect_2d_t
legion_domain_get_bounds_2d(legion_domain_t d_)
{
  Domain d = CObjectWrapper::unwrap(d_);
  DomainT<2,coord_t> space = d;

  return CObjectWrapper::wrap(space.bounds);
}

legion_rect_3d_t
legion_domain_get_bounds_3d(legion_domain_t d_)
{
  Domain d = CObjectWrapper::unwrap(d_);
  DomainT<3,coord_t> space = d;

  return CObjectWrapper::wrap(space.bounds);
}

size_t
legion_domain_get_volume(legion_domain_t d_)
{
  Domain d = CObjectWrapper::unwrap(d_);

  return d.get_volume();
}

// -----------------------------------------------------------------------
// Domain Transform Operations
// -----------------------------------------------------------------------

legion_domain_transform_t
legion_domain_transform_from_1x1(legion_transform_1x1_t t_)
{
  Transform1x1 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainTransform(t));
}

legion_domain_transform_t
legion_domain_transform_from_1x2(legion_transform_1x2_t t_)
{
  Transform1x2 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainTransform(t));
}

legion_domain_transform_t
legion_domain_transform_from_1x3(legion_transform_1x3_t t_)
{
  Transform1x3 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainTransform(t));
}

legion_domain_transform_t
legion_domain_transform_from_2x1(legion_transform_2x1_t t_)
{
  Transform2x1 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainTransform(t));
}

legion_domain_transform_t
legion_domain_transform_from_2x2(legion_transform_2x2_t t_)
{
  Transform2x2 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainTransform(t));
}

legion_domain_transform_t
legion_domain_transform_from_2x3(legion_transform_2x3_t t_)
{
  Transform2x3 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainTransform(t));
}

legion_domain_transform_t
legion_domain_transform_from_3x1(legion_transform_3x1_t t_)
{
  Transform3x1 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainTransform(t));
}

legion_domain_transform_t
legion_domain_transform_from_3x2(legion_transform_3x2_t t_)
{
  Transform3x2 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainTransform(t));
}

legion_domain_transform_t
legion_domain_transform_from_3x3(legion_transform_3x3_t t_)
{
  Transform3x3 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainTransform(t));
}

legion_domain_affine_transform_t
legion_domain_affine_transform_from_1x1(legion_affine_transform_1x1_t t_)
{
  AffineTransform1x1 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainAffineTransform(t));
}

legion_domain_affine_transform_t
legion_domain_affine_transform_from_1x2(legion_affine_transform_1x2_t t_)
{
  AffineTransform1x2 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainAffineTransform(t));
}

legion_domain_affine_transform_t
legion_domain_affine_transform_from_1x3(legion_affine_transform_1x3_t t_)
{
  AffineTransform1x3 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainAffineTransform(t));
}

legion_domain_affine_transform_t
legion_domain_affine_transform_from_2x1(legion_affine_transform_2x1_t t_)
{
  AffineTransform2x1 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainAffineTransform(t));
}

legion_domain_affine_transform_t
legion_domain_affine_transform_from_2x2(legion_affine_transform_2x2_t t_)
{
  AffineTransform2x2 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainAffineTransform(t));
}

legion_domain_affine_transform_t
legion_domain_affine_transform_from_2x3(legion_affine_transform_2x3_t t_)
{
  AffineTransform2x3 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainAffineTransform(t));
}

legion_domain_affine_transform_t
legion_domain_affine_transform_from_3x1(legion_affine_transform_3x1_t t_)
{
  AffineTransform3x1 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainAffineTransform(t));
}

legion_domain_affine_transform_t
legion_domain_affine_transform_from_3x2(legion_affine_transform_3x2_t t_)
{
  AffineTransform3x2 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainAffineTransform(t));
}

legion_domain_affine_transform_t
legion_domain_affine_transform_from_3x3(legion_affine_transform_3x3_t t_)
{
  AffineTransform3x3 t = CObjectWrapper::unwrap(t_);

  return CObjectWrapper::wrap(DomainAffineTransform(t));
}

// -----------------------------------------------------------------------
// Domain Point Operations
// -----------------------------------------------------------------------

legion_domain_point_t
legion_domain_point_from_point_1d(legion_point_1d_t p_)
{
  Point1D p = CObjectWrapper::unwrap(p_);

  return CObjectWrapper::wrap(DomainPoint(p));
}

legion_domain_point_t
legion_domain_point_from_point_2d(legion_point_2d_t p_)
{
  Point2D p = CObjectWrapper::unwrap(p_);

  return CObjectWrapper::wrap(DomainPoint(p));
}

legion_domain_point_t
legion_domain_point_from_point_3d(legion_point_3d_t p_)
{
  Point3D p = CObjectWrapper::unwrap(p_);

  return CObjectWrapper::wrap(DomainPoint(p));
}

legion_point_1d_t
legion_domain_point_get_point_1d(legion_domain_point_t p_)
{
  DomainPoint d = CObjectWrapper::unwrap(p_);
  Point1D p = d;

  return CObjectWrapper::wrap(p);
}

legion_point_2d_t
legion_domain_point_get_point_2d(legion_domain_point_t p_)
{
  DomainPoint d = CObjectWrapper::unwrap(p_);
  Point2D p = d;

  return CObjectWrapper::wrap(p);
}

legion_point_3d_t
legion_domain_point_get_point_3d(legion_domain_point_t p_)
{
  DomainPoint d = CObjectWrapper::unwrap(p_);
  Point3D p = d;

  return CObjectWrapper::wrap(p);
}

legion_domain_point_t
legion_domain_point_nil()
{
  return CObjectWrapper::wrap(DomainPoint::nil());
}

bool
legion_domain_point_is_null(legion_domain_point_t point_)
{
  DomainPoint point = CObjectWrapper::unwrap(point_);

  return point.is_null();
}

legion_domain_point_t
legion_domain_point_safe_cast(legion_runtime_t runtime_,
                              legion_context_t ctx_,
                              legion_domain_point_t point_,
                              legion_logical_region_t region_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  DomainPoint point = CObjectWrapper::unwrap(point_);
  LogicalRegion region = CObjectWrapper::unwrap(region_);

  DomainPoint result = runtime->safe_cast(ctx, point, region);
  return CObjectWrapper::wrap(result);
}

// -----------------------------------------------------------------------
// Domain Point Iterator
// -----------------------------------------------------------------------

legion_domain_point_iterator_t
legion_domain_point_iterator_create(legion_domain_t handle_)
{
  Domain handle = CObjectWrapper::unwrap(handle_);

  Domain::DomainPointIterator *it = new Domain::DomainPointIterator(handle);
  return CObjectWrapper::wrap(it);
}

void
legion_domain_point_iterator_destroy(legion_domain_point_iterator_t handle_)
{
  Domain::DomainPointIterator *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

bool
legion_domain_point_iterator_has_next(legion_domain_point_iterator_t handle_)
{
  Domain::DomainPointIterator *handle = CObjectWrapper::unwrap(handle_);

  return *handle;
}

legion_domain_point_t
legion_domain_point_iterator_next(legion_domain_point_iterator_t handle_)
{
  Domain::DomainPointIterator *handle = CObjectWrapper::unwrap(handle_);

  DomainPoint next = DomainPoint::nil();
  if (handle) {
    next = handle->p;
    (*handle)++;
  }
  return CObjectWrapper::wrap(next);
}

// -------------------------------------------------------
// Coloring Operations
// -------------------------------------------------------

legion_coloring_t
legion_coloring_create(void)
{
  Coloring *coloring = new Coloring();

  return CObjectWrapper::wrap(coloring);
}

void
legion_coloring_destroy(legion_coloring_t handle_)
{
  Coloring *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

void
legion_coloring_ensure_color(legion_coloring_t handle_,
                             legion_color_t color)
{
  Coloring *handle = CObjectWrapper::unwrap(handle_);

  (*handle)[color];
}

void
legion_coloring_add_point(legion_coloring_t handle_,
                          legion_color_t color,
                          legion_ptr_t point_)
{
  Coloring *handle = CObjectWrapper::unwrap(handle_);
  ptr_t point = CObjectWrapper::unwrap(point_);

  (*handle)[color].points.insert(point);
}

void
legion_coloring_delete_point(legion_coloring_t handle_,
                             legion_color_t color,
                             legion_ptr_t point_)
{
  Coloring *handle = CObjectWrapper::unwrap(handle_);
  ptr_t point = CObjectWrapper::unwrap(point_);

  (*handle)[color].points.erase(point);
}

bool
legion_coloring_has_point(legion_coloring_t handle_,
                          legion_color_t color,
                          legion_ptr_t point_)
{
  Coloring *handle = CObjectWrapper::unwrap(handle_);
  ptr_t point = CObjectWrapper::unwrap(point_);
  std::set<ptr_t>& points = (*handle)[color].points;

  return points.find(point) != points.end();
}

void
legion_coloring_add_range(legion_coloring_t handle_,
                          legion_color_t color,
                          legion_ptr_t start_,
                          legion_ptr_t end_)
{
  Coloring *handle = CObjectWrapper::unwrap(handle_);
  ptr_t start = CObjectWrapper::unwrap(start_);
  ptr_t end = CObjectWrapper::unwrap(end_);

  (*handle)[color].ranges.insert(std::pair<ptr_t, ptr_t>(start, end));
}

// -----------------------------------------------------------------------
// Domain Coloring Operations
// -----------------------------------------------------------------------

legion_domain_coloring_t
legion_domain_coloring_create(void)
{
  return CObjectWrapper::wrap(new DomainColoring());
}

void
legion_domain_coloring_destroy(legion_domain_coloring_t handle_)
{
  delete CObjectWrapper::unwrap(handle_);
}

void
legion_domain_coloring_color_domain(legion_domain_coloring_t dc_,
                                    legion_color_t color,
                                    legion_domain_t domain_)
{
  DomainColoring *dc = CObjectWrapper::unwrap(dc_);
  Domain domain = CObjectWrapper::unwrap(domain_);
  (*dc)[color] = domain;
}

legion_domain_t
legion_domain_coloring_get_color_space(legion_domain_coloring_t handle_)
{
  DomainColoring *handle = CObjectWrapper::unwrap(handle_);
  Color color_min = (Color)-1, color_max = 0;
  for(std::map<Color,Domain>::iterator it = handle->begin(),
        ie = handle->end(); it != ie; it++) {
    color_min = std::min(color_min, it->first);
    color_max = std::max(color_max, it->first);
  }
  Domain domain = Rect1D(Point1D(color_min), Point1D(color_max));
  return CObjectWrapper::wrap(domain);
}

// -----------------------------------------------------------------------
// Point Coloring Operations
// -----------------------------------------------------------------------

legion_point_coloring_t
legion_point_coloring_create(void)
{
  return CObjectWrapper::wrap(new PointColoring());
}

void
legion_point_coloring_destroy(
  legion_point_coloring_t handle_)
{
  delete CObjectWrapper::unwrap(handle_);
}

void
legion_point_coloring_add_point(legion_point_coloring_t handle_,
                                legion_domain_point_t color_,
                                legion_ptr_t point_)
{
  PointColoring *handle = CObjectWrapper::unwrap(handle_);
  DomainPoint color = CObjectWrapper::unwrap(color_);
  ptr_t point = CObjectWrapper::unwrap(point_);

  (*handle)[color].points.insert(point);
}

void
legion_point_coloring_add_range(legion_point_coloring_t handle_,
                                legion_domain_point_t color_,
                                legion_ptr_t start_,
                                legion_ptr_t end_ /**< inclusive */)
{
  PointColoring *handle = CObjectWrapper::unwrap(handle_);
  DomainPoint color = CObjectWrapper::unwrap(color_);
  ptr_t start = CObjectWrapper::unwrap(start_);
  ptr_t end = CObjectWrapper::unwrap(end_);

  (*handle)[color].ranges.insert(std::pair<ptr_t, ptr_t>(start, end));
}

// -----------------------------------------------------------------------
// Domain Point Coloring Operations
// -----------------------------------------------------------------------

legion_domain_point_coloring_t
legion_domain_point_coloring_create(void)
{
  return CObjectWrapper::wrap(new DomainPointColoring());
}

void
legion_domain_point_coloring_destroy(
  legion_domain_point_coloring_t handle_)
{
  delete CObjectWrapper::unwrap(handle_);
}

void
legion_domain_point_coloring_color_domain(
  legion_domain_point_coloring_t handle_,
  legion_domain_point_t color_,
  legion_domain_t domain_)
{
  DomainPointColoring *handle = CObjectWrapper::unwrap(handle_);
  DomainPoint color = CObjectWrapper::unwrap(color_);
  Domain domain = CObjectWrapper::unwrap(domain_);
  assert(handle->count(color) == 0);
  (*handle)[color] = domain;
}

// -----------------------------------------------------------------------
// Multi-Domain Coloring Operations
// -----------------------------------------------------------------------

legion_multi_domain_point_coloring_t
legion_multi_domain_point_coloring_create(void)
{
  return CObjectWrapper::wrap(new MultiDomainPointColoring());
}

void
legion_multi_domain_point_coloring_destroy(
  legion_multi_domain_point_coloring_t handle_)
{
  delete CObjectWrapper::unwrap(handle_);
}

void
legion_multi_domain_point_coloring_color_domain(
  legion_multi_domain_point_coloring_t handle_,
  legion_domain_point_t color_,
  legion_domain_t domain_)
{
  MultiDomainPointColoring *handle = CObjectWrapper::unwrap(handle_);
  DomainPoint color = CObjectWrapper::unwrap(color_);
  Domain domain = CObjectWrapper::unwrap(domain_);
  (*handle)[color].insert(domain);
}

// -------------------------------------------------------
// Index Space Operations
// -------------------------------------------------------

legion_index_space_t
legion_index_space_create(legion_runtime_t runtime_,
                          legion_context_t ctx_,
                          size_t max_num_elmts)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  IndexSpace is = runtime->create_index_space(ctx, max_num_elmts);
  return CObjectWrapper::wrap(is);
}

legion_index_space_t
legion_index_space_create_domain(legion_runtime_t runtime_,
                                 legion_context_t ctx_,
                                 legion_domain_t domain_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  Domain domain = CObjectWrapper::unwrap(domain_);

  IndexSpace is = runtime->create_index_space(ctx, domain);
  return CObjectWrapper::wrap(is);
}

legion_index_space_t
legion_index_space_union(legion_runtime_t runtime_,
                         legion_context_t ctx_,
                         const legion_index_space_t *spaces_,
                         size_t num_spaces)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  std::vector<IndexSpace> spaces;
  for (size_t i = 0; i < num_spaces; i++) {
    spaces.push_back(CObjectWrapper::unwrap(spaces_[i]));
  }

  IndexSpace is = runtime->union_index_spaces(ctx, spaces);
  return CObjectWrapper::wrap(is);
}

legion_index_space_t
legion_index_space_intersection(legion_runtime_t runtime_,
                                legion_context_t ctx_,
                                const legion_index_space_t *spaces_,
                                size_t num_spaces)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  std::vector<IndexSpace> spaces;
  for (size_t i = 0; i < num_spaces; i++) {
    spaces.push_back(CObjectWrapper::unwrap(spaces_[i]));
  }

  IndexSpace is = runtime->intersect_index_spaces(ctx, spaces);
  return CObjectWrapper::wrap(is);
}

void
legion_index_space_destroy(legion_runtime_t runtime_,
                           legion_context_t ctx_,
                           legion_index_space_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->destroy_index_space(ctx, handle);
}

bool
legion_index_space_has_multiple_domains(legion_runtime_t runtime_,
                                        legion_index_space_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  return runtime->has_multiple_domains(handle);
}

legion_domain_t
legion_index_space_get_domain(legion_runtime_t runtime_,
                              legion_index_space_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  return CObjectWrapper::wrap(runtime->get_index_space_domain(handle));
}

legion_index_partition_t
legion_index_space_get_parent_index_partition(legion_runtime_t runtime_,
                                              legion_index_space_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  return CObjectWrapper::wrap(runtime->get_parent_index_partition(handle));
}

void
legion_index_space_attach_name(legion_runtime_t runtime_,
                               legion_index_space_t handle_,
                               const char *name,
                               bool is_mutable /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->attach_name(handle, name, is_mutable);
}

void
legion_index_space_retrieve_name(legion_runtime_t runtime_,
                                 legion_index_space_t handle_,
                                 const char **result)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->retrieve_name(handle, *result);
}

//------------------------------------------------------------------------
// Index Partition Operations
//------------------------------------------------------------------------

legion_index_partition_t
legion_index_partition_create_coloring(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_coloring_t coloring_,
  bool disjoint,
  int part_color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  Coloring *coloring = CObjectWrapper::unwrap(coloring_);

  IndexPartition ip =
    runtime->create_index_partition(ctx, parent, *coloring, disjoint,
                                    part_color);
  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_domain_coloring(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_domain_t color_space_,
  legion_domain_coloring_t coloring_,
  bool disjoint,
  int part_color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  Domain color_space = CObjectWrapper::unwrap(color_space_);
  DomainColoring *coloring = CObjectWrapper::unwrap(coloring_);

  // Ensure all colors exist in coloring.
  for(Domain::DomainPointIterator c(color_space); c; c++) {
    assert(c.p.get_dim() <= 1);
    (*coloring)[c.p[0]];
  }

  IndexPartition ip =
    runtime->create_index_partition(ctx, parent, color_space, *coloring,
                                    disjoint, part_color);
  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_point_coloring(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_domain_t color_space_,
  legion_point_coloring_t coloring_,
  legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
  int color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  Domain color_space = CObjectWrapper::unwrap(color_space_);
  PointColoring *coloring = CObjectWrapper::unwrap(coloring_);

  // Ensure all colors exist in coloring.
  for(Domain::DomainPointIterator c(color_space); c; c++) {
    (*coloring)[c.p];
  }

  IndexPartition ip =
    runtime->create_index_partition(ctx, parent, color_space, *coloring,
                                    part_kind, color);
  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_domain_point_coloring(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_domain_t color_space_,
  legion_domain_point_coloring_t coloring_,
  legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
  int color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  Domain color_space = CObjectWrapper::unwrap(color_space_);
  DomainPointColoring *coloring = CObjectWrapper::unwrap(coloring_);

  // Ensure all colors exist in coloring.
  for(Domain::DomainPointIterator c(color_space); c; c++) {
    if (!(*coloring).count(c.p)) {
      switch (c.p.get_dim()) {
        case 1:
          {
            (*coloring)[c.p] = Domain(Rect1D(0, -1)); 
            break;
          }
      case 2:
          {
            (*coloring)[c.p] =
              Domain(Rect2D(Point2D(0, 0), Point2D(-1, -1)));
            break;
          }
      case 3:
          {
            (*coloring)[c.p] =
              Domain(Rect3D(Point3D(0, 0, 0), Point3D(-1, -1, -1)));
            break;
          }
      default:
        break;
      }
    }
  }

  IndexPartition ip =
    runtime->create_index_partition(ctx, parent, color_space, *coloring,
                                    part_kind, color);
  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_multi_domain_point_coloring(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_domain_t color_space_,
  legion_multi_domain_point_coloring_t coloring_,
  legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
  int color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  Domain color_space = CObjectWrapper::unwrap(color_space_);
  MultiDomainPointColoring *coloring = CObjectWrapper::unwrap(coloring_);

  // Ensure all colors exist in coloring.
  for(Domain::DomainPointIterator c(color_space); c; c++) {
    if ((*coloring)[c.p].empty()) {
      switch (c.p.get_dim()) {
        case 1:
          {
            (*coloring)[c.p].insert(Domain(Rect1D(0, -1)));
            break;
          }
      case 2:
          {
            (*coloring)[c.p].insert(
                Domain(Rect2D(Point2D(0, 0), Point2D(-1, -1))));
            break;
          }
      case 3:
          {
            (*coloring)[c.p].insert(
                Domain(Rect3D(Point3D(0, 0, 0), Point3D(-1, -1, -1))));
            break;
          }
      default:
        break;
      }
    }
  }

  IndexPartition ip =
    runtime->create_index_partition(ctx, parent, color_space, *coloring,
                                    part_kind, color);
  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_blockify_1d(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_blockify_1d_t blockify_,
  int part_color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  CObjectWrapper::Blockify<1> blockify = CObjectWrapper::unwrap(blockify_);

  IndexPartition ip =
    runtime->create_partition_by_blockify(ctx, IndexSpaceT<1,coord_t>(parent),
        blockify.block_size, blockify.offset, part_color);
  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_blockify_2d(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_blockify_2d_t blockify_,
  int part_color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  CObjectWrapper::Blockify<2> blockify = CObjectWrapper::unwrap(blockify_);

  IndexPartition ip =
    runtime->create_partition_by_blockify(ctx, IndexSpaceT<2,coord_t>(parent),
        blockify.block_size, blockify.offset, part_color);
  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_blockify_3d(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_blockify_3d_t blockify_,
  int part_color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  CObjectWrapper::Blockify<3> blockify = CObjectWrapper::unwrap(blockify_);

  IndexPartition ip =
    runtime->create_partition_by_blockify(ctx, IndexSpaceT<3,coord_t>(parent),
        blockify.block_size, blockify.offset, part_color);
  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_equal(legion_runtime_t runtime_,
                                    legion_context_t ctx_,
                                    legion_index_space_t parent_,
                                    legion_index_space_t color_space_,
                                    size_t granularity,
                                    int color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_equal_partition(ctx, parent, color_space, granularity,
                                    color);
  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_by_union(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_index_partition_t handle1_,
  legion_index_partition_t handle2_,
  legion_index_space_t color_space_,
  legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
  int color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  IndexPartition handle1 = CObjectWrapper::unwrap(handle1_);
  IndexPartition handle2 = CObjectWrapper::unwrap(handle2_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_partition_by_union(ctx, parent, handle1, handle2,
                                       color_space, part_kind, color);
  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_by_intersection(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_index_partition_t handle1_,
  legion_index_partition_t handle2_,
  legion_index_space_t color_space_,
  legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
  int color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  IndexPartition handle1 = CObjectWrapper::unwrap(handle1_);
  IndexPartition handle2 = CObjectWrapper::unwrap(handle2_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_partition_by_intersection(ctx, parent, handle1, handle2,
                                              color_space, part_kind, color);
  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_by_difference(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_index_partition_t handle1_,
  legion_index_partition_t handle2_,
  legion_index_space_t color_space_,
  legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
  int color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  IndexPartition handle1 = CObjectWrapper::unwrap(handle1_);
  IndexPartition handle2 = CObjectWrapper::unwrap(handle2_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_partition_by_difference(ctx, parent, handle1, handle2,
                                            color_space, part_kind, color);
  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_by_field(legion_runtime_t runtime_,
                                       legion_context_t ctx_,
                                       legion_logical_region_t handle_,
                                       legion_logical_region_t parent_,
                                       legion_field_id_t fid,
                                       legion_index_space_t color_space_,
                                       int color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_partition_by_field(ctx, handle, parent, fid, color_space,
                                       color);

  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_by_image(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t handle_,
  legion_logical_partition_t projection_,
  legion_logical_region_t parent_,
  legion_field_id_t fid,
  legion_index_space_t color_space_,
  legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
  int color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace handle = CObjectWrapper::unwrap(handle_);
  LogicalPartition projection = CObjectWrapper::unwrap(projection_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_partition_by_image(
      ctx, handle, projection, parent, fid, color_space, part_kind, color);

  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_by_preimage(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_partition_t projection_,
  legion_logical_region_t handle_,
  legion_logical_region_t parent_,
  legion_field_id_t fid,
  legion_index_space_t color_space_,
  legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
  int color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexPartition projection = CObjectWrapper::unwrap(projection_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_partition_by_preimage(
      ctx, projection, handle, parent, fid, color_space, part_kind, color);

  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_by_image_range(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t handle_,
  legion_logical_partition_t projection_,
  legion_logical_region_t parent_,
  legion_field_id_t fid,
  legion_index_space_t color_space_,
  legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
  int color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace handle = CObjectWrapper::unwrap(handle_);
  LogicalPartition projection = CObjectWrapper::unwrap(projection_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_partition_by_image_range(
      ctx, handle, projection, parent, fid, color_space, part_kind, color);

  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_by_preimage_range(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_partition_t projection_,
  legion_logical_region_t handle_,
  legion_logical_region_t parent_,
  legion_field_id_t fid,
  legion_index_space_t color_space_,
  legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
  int color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexPartition projection = CObjectWrapper::unwrap(projection_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_partition_by_preimage_range(
      ctx, projection, handle, parent, fid, color_space, part_kind, color);

  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_by_restriction(
    legion_runtime_t runtime_,
    legion_context_t ctx_,
    legion_index_space_t parent_,
    legion_index_space_t color_space_,
    legion_domain_transform_t transform_,
    legion_domain_t extent_,
    legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    int color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);
  DomainTransform transform = CObjectWrapper::unwrap(transform_);
  Domain extent = CObjectWrapper::unwrap(extent_);

  IndexPartition ip = 
    runtime->create_partition_by_restriction(
        ctx, parent, color_space, transform, extent, part_kind, color);

  return CObjectWrapper::wrap(ip);
}

bool
legion_index_partition_is_disjoint(legion_runtime_t runtime_,
                                   legion_index_partition_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  return runtime->is_index_partition_disjoint(handle);
}

bool
legion_index_partition_is_complete(legion_runtime_t runtime_,
                                   legion_index_partition_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  return runtime->is_index_partition_complete(handle);
}

legion_index_space_t
legion_index_partition_get_index_subspace(legion_runtime_t runtime_,
                                          legion_index_partition_t handle_,
                                          legion_color_t color)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  IndexSpace is = runtime->get_index_subspace(handle, color);

  return CObjectWrapper::wrap(is);
}

legion_index_space_t
legion_index_partition_get_index_subspace_domain_point(
  legion_runtime_t runtime_,
  legion_index_partition_t handle_,
  legion_domain_point_t color_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);
  DomainPoint color = CObjectWrapper::unwrap(color_);

  IndexSpace is = runtime->get_index_subspace(handle, color);

  return CObjectWrapper::wrap(is);
}

bool
legion_index_partition_has_index_subspace_domain_point(
  legion_runtime_t runtime_,
  legion_index_partition_t handle_,
  legion_domain_point_t color_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);
  DomainPoint color = CObjectWrapper::unwrap(color_);

  return runtime->has_index_subspace(handle, color);
}

legion_index_space_t
legion_index_partition_get_color_space(legion_runtime_t runtime_,
                                       legion_index_partition_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  IndexSpace is = runtime->get_index_partition_color_space_name(handle);

  return CObjectWrapper::wrap(is);
}

legion_color_t
legion_index_partition_get_color(legion_runtime_t runtime_,
                                 legion_index_partition_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  Color c = runtime->get_index_partition_color(handle);

  return c;
}

legion_index_space_t
legion_index_partition_get_parent_index_space(legion_runtime_t runtime_,
                                              legion_index_partition_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  IndexSpace is = runtime->get_parent_index_space(handle);

  return CObjectWrapper::wrap(is);
}

void
legion_index_partition_destroy(legion_runtime_t runtime_,
                               legion_context_t ctx_,
                               legion_index_partition_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  runtime->destroy_index_partition(ctx, handle);
}

void
legion_index_partition_attach_name(legion_runtime_t runtime_,
                                   legion_index_partition_t handle_,
                                   const char *name,
                                   bool is_mutable /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  runtime->attach_name(handle, name, is_mutable);
}

void
legion_index_partition_retrieve_name(legion_runtime_t runtime_,
                                     legion_index_partition_t handle_,
                                     const char **result)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  runtime->retrieve_name(handle, *result);
}

// -------------------------------------------------------
// Field Space Operations
// -------------------------------------------------------

legion_field_space_t
legion_field_space_create(legion_runtime_t runtime_,
                          legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  FieldSpace fs = runtime->create_field_space(ctx);
  return CObjectWrapper::wrap(fs);
}

void
legion_field_space_destroy(legion_runtime_t runtime_,
                           legion_context_t ctx_,
                           legion_field_space_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->destroy_field_space(ctx, handle);
}

void
legion_field_space_attach_name(legion_runtime_t runtime_,
                               legion_field_space_t handle_,
                               const char *name,
                               bool is_mutable /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->attach_name(handle, name, is_mutable);
}

void
legion_field_space_retrieve_name(legion_runtime_t runtime_,
                                 legion_field_space_t handle_,
                                 const char **result)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->retrieve_name(handle, *result);
}

void
legion_field_id_attach_name(legion_runtime_t runtime_,
                            legion_field_space_t handle_,
                            legion_field_id_t id,
                            const char *name,
                            bool is_mutable /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->attach_name(handle, id, name, is_mutable);
}

void
legion_field_id_retrieve_name(legion_runtime_t runtime_,
                              legion_field_space_t handle_,
                              legion_field_id_t id,
                              const char **result)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->retrieve_name(handle, id, *result);
}

// -------------------------------------------------------
// Logical Region Operations
// -------------------------------------------------------

legion_logical_region_t
legion_logical_region_create(legion_runtime_t runtime_,
                             legion_context_t ctx_,
                             legion_index_space_t index_,
                             legion_field_space_t fields_,
                             bool task_local)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace index = CObjectWrapper::unwrap(index_);
  FieldSpace fields = CObjectWrapper::unwrap(fields_);

  LogicalRegion r =
    runtime->create_logical_region(ctx, index, fields, task_local);
  return CObjectWrapper::wrap(r);
}

void
legion_logical_region_destroy(legion_runtime_t runtime_,
                              legion_context_t ctx_,
                              legion_logical_region_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);

  runtime->destroy_logical_region(ctx, handle);
}

legion_color_t
legion_logical_region_get_color(legion_runtime_t runtime_,
                                legion_logical_region_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);

  return runtime->get_logical_region_color(handle);
}

legion_domain_point_t
legion_logical_region_get_color_domain_point(legion_runtime_t runtime_,
                                             legion_logical_region_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);

  return CObjectWrapper::wrap(runtime->get_logical_region_color_point(handle));
}

bool
legion_logical_region_has_parent_logical_partition(
  legion_runtime_t runtime_,
  legion_logical_region_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);

  return runtime->has_parent_logical_partition(handle);
}

legion_logical_partition_t
legion_logical_region_get_parent_logical_partition(
  legion_runtime_t runtime_,
  legion_logical_region_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);

  LogicalPartition p = runtime->get_parent_logical_partition(handle);
  return CObjectWrapper::wrap(p);
}

void
legion_logical_region_attach_name(legion_runtime_t runtime_,
                                  legion_logical_region_t handle_,
                                  const char *name,
                                  bool is_mutable /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);

  runtime->attach_name(handle, name, is_mutable);
}

void
legion_logical_region_retrieve_name(legion_runtime_t runtime_,
                                    legion_logical_region_t handle_,
                                    const char **result)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);

  runtime->retrieve_name(handle, *result);
}

// -----------------------------------------------------------------------
// Logical Region Tree Traversal Operations
// -----------------------------------------------------------------------

legion_logical_partition_t
legion_logical_partition_create(legion_runtime_t runtime_,
                                legion_context_t ctx_,
                                legion_logical_region_t parent_,
                                legion_index_partition_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  LogicalPartition r = runtime->get_logical_partition(ctx, parent, handle);
  return CObjectWrapper::wrap(r);
}

legion_logical_partition_t
legion_logical_partition_create_by_tree(legion_runtime_t runtime_,
                                        legion_context_t ctx_,
                                        legion_index_partition_t handle_,
                                        legion_field_space_t fspace_,
                                        legion_region_tree_id_t tid)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  FieldSpace fspace = CObjectWrapper::unwrap(fspace_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  LogicalPartition r =
    runtime->get_logical_partition_by_tree(ctx, handle, fspace, tid);
  return CObjectWrapper::wrap(r);
}

void
legion_logical_partition_destroy(legion_runtime_t runtime_,
                                 legion_context_t ctx_,
                                 legion_logical_partition_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);

  runtime->destroy_logical_partition(ctx, handle);
}

legion_logical_region_t
legion_logical_partition_get_logical_subregion(
  legion_runtime_t runtime_,
  legion_logical_partition_t parent_,
  legion_index_space_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalPartition parent = CObjectWrapper::unwrap(parent_);
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  LogicalRegion r = runtime->get_logical_subregion(parent, handle);
  return CObjectWrapper::wrap(r);
}

legion_logical_region_t
legion_logical_partition_get_logical_subregion_by_color(
  legion_runtime_t runtime_,
  legion_logical_partition_t parent_,
  legion_color_t c)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalPartition parent = CObjectWrapper::unwrap(parent_);

  LogicalRegion r = runtime->get_logical_subregion_by_color(parent, c);
  return CObjectWrapper::wrap(r);
}

legion_logical_region_t
legion_logical_partition_get_logical_subregion_by_color_domain_point(
  legion_runtime_t runtime_,
  legion_logical_partition_t parent_,
  legion_domain_point_t c_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalPartition parent = CObjectWrapper::unwrap(parent_);
  DomainPoint c = CObjectWrapper::unwrap(c_);

  LogicalRegion r = runtime->get_logical_subregion_by_color(parent, c);
  return CObjectWrapper::wrap(r);
}

bool
legion_logical_partition_has_logical_subregion_by_color_domain_point(
  legion_runtime_t runtime_,
  legion_logical_partition_t parent_,
  legion_domain_point_t c_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalPartition parent = CObjectWrapper::unwrap(parent_);
  DomainPoint c = CObjectWrapper::unwrap(c_);

  return runtime->has_logical_subregion_by_color(parent, c);
}

legion_logical_region_t
legion_logical_partition_get_logical_subregion_by_tree(
  legion_runtime_t runtime_,
  legion_index_space_t handle_,
  legion_field_space_t fspace_,
  legion_region_tree_id_t tid)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexSpace handle = CObjectWrapper::unwrap(handle_);
  FieldSpace fspace = CObjectWrapper::unwrap(fspace_);

  LogicalRegion r = runtime->get_logical_subregion_by_tree(handle, fspace, tid);
  return CObjectWrapper::wrap(r);
}

legion_logical_region_t
legion_logical_partition_get_parent_logical_region(
  legion_runtime_t runtime_,
  legion_logical_partition_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);

  LogicalRegion r = runtime->get_parent_logical_region(handle);
  return CObjectWrapper::wrap(r);
}

void
legion_logical_partition_attach_name(legion_runtime_t runtime_,
                                     legion_logical_partition_t handle_,
                                     const char *name,
                                     bool is_mutable /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);

  runtime->attach_name(handle, name, is_mutable);
}

void
legion_logical_partition_retrieve_name(legion_runtime_t runtime_,
                                       legion_logical_partition_t handle_,
                                       const char **result)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);

  runtime->retrieve_name(handle, *result);
}

// -----------------------------------------------------------------------
// Region Requirement Operations
// -----------------------------------------------------------------------

legion_logical_region_t
legion_region_requirement_get_region(legion_region_requirement_t req_)
{
  RegionRequirement *req = CObjectWrapper::unwrap(req_);

  return CObjectWrapper::wrap(req->region);
}

legion_logical_region_t
legion_region_requirement_get_parent(legion_region_requirement_t req_)
{
  RegionRequirement *req = CObjectWrapper::unwrap(req_);

  return CObjectWrapper::wrap(req->parent);
}

legion_logical_partition_t
legion_region_requirement_get_partition(legion_region_requirement_t req_)
{
  RegionRequirement *req = CObjectWrapper::unwrap(req_);

  return CObjectWrapper::wrap(req->partition);
}

unsigned
legion_region_requirement_get_privilege_fields_size(
    legion_region_requirement_t req_)
{
  RegionRequirement *req = CObjectWrapper::unwrap(req_);

  return req->privilege_fields.size();
}

template<typename DST, typename SRC>
static void copy_n(DST dst, SRC src, size_t n)
{
  for(size_t i = 0; i < n; ++i)
    *dst++ = *src++;
}

void
legion_region_requirement_get_privilege_fields(
    legion_region_requirement_t req_,
    legion_field_id_t* fields,
    unsigned fields_size)
{
  RegionRequirement *req = CObjectWrapper::unwrap(req_);

  copy_n(fields, req->privilege_fields.begin(),
         std::min(req->privilege_fields.size(),
                  static_cast<size_t>(fields_size)));
}


legion_field_id_t
legion_region_requirement_get_privilege_field(
    legion_region_requirement_t req_,
    unsigned idx)
{
  RegionRequirement *req = CObjectWrapper::unwrap(req_);
  assert(idx >= 0 && idx < req->instance_fields.size());

  std::set<FieldID>::iterator itr = req->privilege_fields.begin();
  for (unsigned i = 0; i < idx; ++i, ++itr);
  return *itr;
}

unsigned
legion_region_requirement_get_instance_fields_size(
    legion_region_requirement_t req_)
{
  RegionRequirement *req = CObjectWrapper::unwrap(req_);

  return req->instance_fields.size();
}

void
legion_region_requirement_get_instance_fields(
    legion_region_requirement_t req_,
    legion_field_id_t* fields,
    unsigned fields_size)
{
  RegionRequirement *req = CObjectWrapper::unwrap(req_);

  copy_n(fields, req->instance_fields.begin(),
         std::min(req->instance_fields.size(),
                  static_cast<size_t>(fields_size)));
}

legion_field_id_t
legion_region_requirement_get_instance_field(
    legion_region_requirement_t req_,
    unsigned idx)
{
  RegionRequirement *req = CObjectWrapper::unwrap(req_);

  assert(idx >= 0 && idx < req->instance_fields.size());
  return req->instance_fields[idx];
}

legion_privilege_mode_t
legion_region_requirement_get_privilege(legion_region_requirement_t req_)
{
  RegionRequirement *req = CObjectWrapper::unwrap(req_);

  return req->privilege;
}

legion_coherence_property_t
legion_region_requirement_get_prop(legion_region_requirement_t req_)
{
  RegionRequirement *req = CObjectWrapper::unwrap(req_);

  return req->prop;
}

legion_reduction_op_id_t
legion_region_requirement_get_redop(legion_region_requirement_t req_)
{
  RegionRequirement *req = CObjectWrapper::unwrap(req_);

  return req->redop;
}

legion_mapping_tag_id_t
legion_region_requirement_get_tag(legion_region_requirement_t req_)
{
  RegionRequirement *req = CObjectWrapper::unwrap(req_);

  return req->tag;
}

legion_handle_type_t
legion_region_requirement_get_handle_type(legion_region_requirement_t req_)
{
  RegionRequirement *req = CObjectWrapper::unwrap(req_);

  return req->handle_type;
}

legion_projection_id_t
legion_region_requirement_get_projection(legion_region_requirement_t req_)
{
  RegionRequirement *req = CObjectWrapper::unwrap(req_);

  return req->projection;
}

// -------------------------------------------------------
// Allocator and Argument Map Operations
// -------------------------------------------------------

legion_field_allocator_t
legion_field_allocator_create(legion_runtime_t runtime_,
                              legion_context_t ctx_,
                              legion_field_space_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  FieldAllocator *fsa = new FieldAllocator(runtime->create_field_allocator(ctx, handle));
  return CObjectWrapper::wrap(fsa);
}

void
legion_field_allocator_destroy(legion_field_allocator_t handle_)
{
  FieldAllocator *handle = CObjectWrapper::unwrap(handle_);
  delete handle;
  // Destructor is a nop anyway.
}

legion_field_id_t
legion_field_allocator_allocate_field(legion_field_allocator_t allocator_,
                                      size_t field_size,
                                      legion_field_id_t desired_fieldid)
{
  FieldAllocator *allocator = CObjectWrapper::unwrap(allocator_);
  return allocator->allocate_field(field_size, desired_fieldid);
}

void
legion_field_allocator_free_field(legion_field_allocator_t allocator_,
                                  legion_field_id_t fid)
{
  FieldAllocator *allocator = CObjectWrapper::unwrap(allocator_);
  allocator->free_field(fid);
}

legion_field_id_t
legion_field_allocator_allocate_local_field(legion_field_allocator_t allocator_,
                                            size_t field_size,
                                            legion_field_id_t desired_fieldid)
{
  FieldAllocator *allocator = CObjectWrapper::unwrap(allocator_);
  return allocator->allocate_local_field(field_size, desired_fieldid);
}

legion_argument_map_t
legion_argument_map_create()
{
  return CObjectWrapper::wrap(new ArgumentMap());
}

void
legion_argument_map_set_point(legion_argument_map_t map_,
                              legion_domain_point_t dp_,
                              legion_task_argument_t arg_,
                              bool replace)
{
  ArgumentMap *map = CObjectWrapper::unwrap(map_);
  DomainPoint dp = CObjectWrapper::unwrap(dp_);
  TaskArgument arg = CObjectWrapper::unwrap(arg_);

  map->set_point(dp, arg, replace);
}

void
legion_argument_map_destroy(legion_argument_map_t map_)
{
  ArgumentMap *map = CObjectWrapper::unwrap(map_);

  delete map;
}

//------------------------------------------------------------------------
// Predicate Operations
//------------------------------------------------------------------------

void
legion_predicate_destroy(legion_predicate_t handle_)
{
  Predicate *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

const legion_predicate_t
legion_predicate_true(void)
{
  return CObjectWrapper::wrap_const(&Predicate::TRUE_PRED);
}

const legion_predicate_t
legion_predicate_false(void)
{
  return CObjectWrapper::wrap_const(&Predicate::FALSE_PRED);
}

// -----------------------------------------------------------------------
// Phase Barrier Operations
// -----------------------------------------------------------------------

legion_phase_barrier_t
legion_phase_barrier_create(legion_runtime_t runtime_,
                            legion_context_t ctx_,
                            unsigned arrivals)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  PhaseBarrier result = runtime->create_phase_barrier(ctx, arrivals);
  return CObjectWrapper::wrap(result);
}

void
legion_phase_barrier_destroy(legion_runtime_t runtime_,
                             legion_context_t ctx_,
                             legion_phase_barrier_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  PhaseBarrier handle = CObjectWrapper::unwrap(handle_);

  runtime->destroy_phase_barrier(ctx, handle);
}

legion_phase_barrier_t
legion_phase_barrier_alter_arrival_count(legion_runtime_t runtime_,
                                         legion_context_t ctx_,
                                         legion_phase_barrier_t handle_,
                                         int delta)
{
  PhaseBarrier handle = CObjectWrapper::unwrap(handle_);

  handle.alter_arrival_count(delta); // This modifies handle.
  return CObjectWrapper::wrap(handle);
}

void
legion_phase_barrier_arrive(legion_runtime_t runtime_,
                            legion_context_t ctx_,
                            legion_phase_barrier_t handle_,
                            unsigned count /* = 1 */)
{
  PhaseBarrier handle = CObjectWrapper::unwrap(handle_);

  handle.arrive(count);
}

void
legion_phase_barrier_wait(legion_runtime_t runtime_,
                          legion_context_t ctx_,
                          legion_phase_barrier_t handle_)
{
  PhaseBarrier handle = CObjectWrapper::unwrap(handle_);

  handle.wait();
}

legion_phase_barrier_t
legion_phase_barrier_advance(legion_runtime_t runtime_,
                             legion_context_t ctx_,
                             legion_phase_barrier_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  PhaseBarrier handle = CObjectWrapper::unwrap(handle_);

  PhaseBarrier result = runtime->advance_phase_barrier(ctx, handle);
  return CObjectWrapper::wrap(result);
}

// -----------------------------------------------------------------------
// Dynamic Collective Operations
// -----------------------------------------------------------------------

legion_dynamic_collective_t
legion_dynamic_collective_create(legion_runtime_t runtime_,
                                 legion_context_t ctx_,
                                 unsigned arrivals,
                                 legion_reduction_op_id_t redop,
                                 const void *init_value,
                                 size_t init_size)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  DynamicCollective result =
    runtime->create_dynamic_collective(ctx, arrivals, redop,
                                       init_value, init_size);
  return CObjectWrapper::wrap(result);
}

void
legion_dynamic_collective_destroy(legion_runtime_t runtime_,
                                  legion_context_t ctx_,
                                  legion_dynamic_collective_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  DynamicCollective handle = CObjectWrapper::unwrap(handle_);

  runtime->destroy_dynamic_collective(ctx, handle);
}

void
legion_dynamic_collective_arrive(legion_runtime_t runtime_,
                                 legion_context_t ctx_,
                                 legion_dynamic_collective_t handle_,
                                 const void *buffer,
                                 size_t size,
                                 unsigned count /* = 1 */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  DynamicCollective handle = CObjectWrapper::unwrap(handle_);

  runtime->arrive_dynamic_collective(ctx, handle, buffer, size, count);
}

legion_dynamic_collective_t
legion_dynamic_collective_alter_arrival_count(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_dynamic_collective_t handle_,
  int delta)
{
  DynamicCollective handle = CObjectWrapper::unwrap(handle_);

  handle.alter_arrival_count(delta); // This modifies handle.
  return CObjectWrapper::wrap(handle);
}

void
legion_dynamic_collective_defer_arrival(legion_runtime_t runtime_,
                                        legion_context_t ctx_,
                                        legion_dynamic_collective_t handle_,
                                        legion_future_t f_,
                                        unsigned count /* = 1 */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  DynamicCollective handle = CObjectWrapper::unwrap(handle_);
  Future *f = CObjectWrapper::unwrap(f_);

  runtime->defer_dynamic_collective_arrival(ctx, handle, *f, count);
}

legion_future_t
legion_dynamic_collective_get_result(legion_runtime_t runtime_,
                                     legion_context_t ctx_,
                                     legion_dynamic_collective_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  DynamicCollective handle = CObjectWrapper::unwrap(handle_);

  Future f = runtime->get_dynamic_collective_result(ctx, handle);
  return CObjectWrapper::wrap(new Future(f));
}

legion_dynamic_collective_t
legion_dynamic_collective_advance(legion_runtime_t runtime_,
                                  legion_context_t ctx_,
                                  legion_dynamic_collective_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  DynamicCollective handle = CObjectWrapper::unwrap(handle_);

  DynamicCollective result = runtime->advance_dynamic_collective(ctx, handle);
  return CObjectWrapper::wrap(result);
}

//------------------------------------------------------------------------
// Future Operations
//------------------------------------------------------------------------

legion_future_t
legion_future_from_untyped_pointer(legion_runtime_t runtime_,
                                   const void *buffer,
                                   size_t size)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);

  Future *result = new Future(
    Future::from_untyped_pointer(runtime, buffer, size));
  return CObjectWrapper::wrap(result);
}

legion_future_t
legion_future_copy(legion_future_t handle_)
{
  Future *handle = CObjectWrapper::unwrap(handle_);

  Future *result = new Future(*handle);
  return CObjectWrapper::wrap(result);
}

void
legion_future_destroy(legion_future_t handle_)
{
  Future *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

void
legion_future_get_void_result(legion_future_t handle_)
{
  Future *handle = CObjectWrapper::unwrap(handle_);

  handle->get_void_result();
}

bool
legion_future_is_empty(legion_future_t handle_,
                       bool block /* = false */)
{
  Future *handle = CObjectWrapper::unwrap(handle_);

  return handle->is_empty(block);
}

bool
legion_future_is_ready(legion_future_t handle_)
{
  Future *handle = CObjectWrapper::unwrap(handle_);

  return handle->is_ready();
}

const void *
legion_future_get_untyped_pointer(legion_future_t handle_)
{
  Future *handle = CObjectWrapper::unwrap(handle_);

  return handle->get_untyped_pointer();
}

size_t
legion_future_get_untyped_size(legion_future_t handle_)
{
  Future *handle = CObjectWrapper::unwrap(handle_);
  return handle->get_untyped_size();
}

// -----------------------------------------------------------------------
// Future Map Operations
// -----------------------------------------------------------------------

legion_future_map_t
legion_future_map_copy(legion_future_map_t handle_)
{
  FutureMap *handle = CObjectWrapper::unwrap(handle_);

  FutureMap *result = new FutureMap(*handle);
  return CObjectWrapper::wrap(result);
}

void
legion_future_map_destroy(legion_future_map_t fm_)
{
  FutureMap *fm = CObjectWrapper::unwrap(fm_);

  delete fm;
}

void
legion_future_map_wait_all_results(legion_future_map_t fm_)
{
  FutureMap *fm = CObjectWrapper::unwrap(fm_);

  fm->wait_all_results();
}

legion_future_t
legion_future_map_get_future(legion_future_map_t fm_,
                             legion_domain_point_t dp_)
{
  FutureMap *fm = CObjectWrapper::unwrap(fm_);
  DomainPoint dp = CObjectWrapper::unwrap(dp_);

  return CObjectWrapper::wrap(new Future(fm->get_future(dp)));
}

//------------------------------------------------------------------------
// Task Launch Operations
//------------------------------------------------------------------------

legion_task_launcher_t
legion_task_launcher_create(
  legion_task_id_t tid,
  legion_task_argument_t arg_,
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t tag /* = 0 */)
{
  TaskArgument arg = CObjectWrapper::unwrap(arg_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  TaskLauncher *launcher = new TaskLauncher(tid, arg, *pred, id, tag);
  return CObjectWrapper::wrap(launcher);
}

void
legion_task_launcher_destroy(legion_task_launcher_t launcher_)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  delete launcher;
}

legion_future_t
legion_task_launcher_execute(legion_runtime_t runtime_,
                             legion_context_t ctx_,
                             legion_task_launcher_t launcher_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  Future f = runtime->execute_task(ctx, *launcher);
  return CObjectWrapper::wrap(new Future(f));
}

unsigned
legion_task_launcher_add_region_requirement_logical_region(
  legion_task_launcher_t launcher_,
  legion_logical_region_t handle_,
  legion_privilege_mode_t priv,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool verified /* = false*/)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->region_requirements.size();
  launcher->add_region_requirement(
    RegionRequirement(handle, priv, prop, parent, tag, verified));
  return idx;
}

unsigned
legion_task_launcher_add_region_requirement_logical_region_reduction(
  legion_task_launcher_t launcher_,
  legion_logical_region_t handle_,
  legion_reduction_op_id_t redop,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool verified /* = false*/)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->region_requirements.size();
  launcher->add_region_requirement(
    RegionRequirement(handle, redop, prop, parent, tag, verified));
  return idx;
}

void
legion_task_launcher_add_field(legion_task_launcher_t launcher_,
                               unsigned idx,
                               legion_field_id_t fid,
                               bool inst /* = true */)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->add_field(idx, fid, inst);
}

void
legion_task_launcher_add_flags(legion_task_launcher_t launcher_,
                               unsigned idx,
                               enum legion_region_flags_t flags)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->region_requirements[idx].add_flags(flags);
}

void
legion_task_launcher_intersect_flags(legion_task_launcher_t launcher_,
                                     unsigned idx,
                                     enum legion_region_flags_t flags)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->region_requirements[idx].flags &= flags;
}

unsigned
legion_task_launcher_add_index_requirement(
  legion_task_launcher_t launcher_,
  legion_index_space_t handle_,
  legion_allocate_mode_t priv,
  legion_index_space_t parent_,
  bool verified /* = false*/)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  IndexSpace handle = CObjectWrapper::unwrap(handle_);
  IndexSpace parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->index_requirements.size();
  launcher->add_index_requirement(
    IndexSpaceRequirement(handle, priv, parent, verified));
  return idx;
}

void
legion_task_launcher_add_future(legion_task_launcher_t launcher_,
                                legion_future_t future_)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  Future *future = CObjectWrapper::unwrap(future_);

  launcher->add_future(*future);
}

void
legion_task_launcher_add_wait_barrier(legion_task_launcher_t launcher_,
                                      legion_phase_barrier_t bar_)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  PhaseBarrier bar = CObjectWrapper::unwrap(bar_);

  launcher->add_wait_barrier(bar);
}

void
legion_task_launcher_add_arrival_barrier(legion_task_launcher_t launcher_,
                                         legion_phase_barrier_t bar_)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  PhaseBarrier bar = CObjectWrapper::unwrap(bar_);

  launcher->add_arrival_barrier(bar);
}

legion_index_launcher_t
legion_index_launcher_create(
  legion_task_id_t tid,
  legion_domain_t domain_,
  legion_task_argument_t global_arg_,
  legion_argument_map_t map_,
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  bool must /* = false */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t tag /* = 0 */)
{
  Domain domain = CObjectWrapper::unwrap(domain_);
  TaskArgument global_arg = CObjectWrapper::unwrap(global_arg_);
  ArgumentMap *map = CObjectWrapper::unwrap(map_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  IndexTaskLauncher *launcher =
    new IndexTaskLauncher(tid, domain, global_arg, *map, *pred, must, id, tag);
  return CObjectWrapper::wrap(launcher);
}

void
legion_index_launcher_destroy(legion_index_launcher_t launcher_)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  delete launcher;
}

legion_future_map_t
legion_index_launcher_execute(legion_runtime_t runtime_,
                             legion_context_t ctx_,
                             legion_index_launcher_t launcher_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  FutureMap f = runtime->execute_index_space(ctx, *launcher);
  return CObjectWrapper::wrap(new FutureMap(f));
}

legion_future_t
legion_index_launcher_execute_reduction(legion_runtime_t runtime_,
                                        legion_context_t ctx_,
                                        legion_index_launcher_t launcher_,
                                        legion_reduction_op_id_t redop)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  Future f = runtime->execute_index_space(ctx, *launcher, redop);
  return CObjectWrapper::wrap(new Future(f));
}

unsigned
legion_index_launcher_add_region_requirement_logical_region(
  legion_index_launcher_t launcher_,
  legion_logical_region_t handle_,
  legion_projection_id_t proj /* = 0 */,
  legion_privilege_mode_t priv,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool verified /* = false*/)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->region_requirements.size();
  launcher->add_region_requirement(
    RegionRequirement(handle, proj, priv, prop, parent, tag, verified));
  return idx;
}

unsigned
legion_index_launcher_add_region_requirement_logical_partition(
  legion_index_launcher_t launcher_,
  legion_logical_partition_t handle_,
  legion_projection_id_t proj /* = 0 */,
  legion_privilege_mode_t priv,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool verified /* = false*/)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->region_requirements.size();
  launcher->add_region_requirement(
    RegionRequirement(handle, proj, priv, prop, parent, tag, verified));
  return idx;
}

unsigned
legion_index_launcher_add_region_requirement_logical_region_reduction(
  legion_index_launcher_t launcher_,
  legion_logical_region_t handle_,
  legion_projection_id_t proj /* = 0 */,
  legion_reduction_op_id_t redop,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool verified /* = false*/)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->region_requirements.size();
  launcher->add_region_requirement(
    RegionRequirement(handle, proj, redop, prop, parent, tag, verified));
  return idx;
}

unsigned
legion_index_launcher_add_region_requirement_logical_partition_reduction(
  legion_index_launcher_t launcher_,
  legion_logical_partition_t handle_,
  legion_projection_id_t proj /* = 0 */,
  legion_reduction_op_id_t redop,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool verified /* = false*/)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->region_requirements.size();
  launcher->add_region_requirement(
    RegionRequirement(handle, proj, redop, prop, parent, tag, verified));
  return idx;
}

void
legion_index_launcher_add_field(legion_index_launcher_t launcher_,
                               unsigned idx,
                               legion_field_id_t fid,
                               bool inst /* = true */)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->add_field(idx, fid, inst);
}

void
legion_index_launcher_add_flags(legion_index_launcher_t launcher_,
                                unsigned idx,
                                enum legion_region_flags_t flags)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->region_requirements[idx].add_flags(flags);
}

void
legion_index_launcher_intersect_flags(legion_index_launcher_t launcher_,
                                      unsigned idx,
                                      enum legion_region_flags_t flags)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->region_requirements[idx].flags &= flags;
}

unsigned
legion_index_launcher_add_index_requirement(
  legion_index_launcher_t launcher_,
  legion_index_space_t handle_,
  legion_allocate_mode_t priv,
  legion_index_space_t parent_,
  bool verified /* = false*/)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  IndexSpace handle = CObjectWrapper::unwrap(handle_);
  IndexSpace parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->index_requirements.size();
  launcher->add_index_requirement(
    IndexSpaceRequirement(handle, priv, parent, verified));
  return idx;
}

void
legion_index_launcher_add_future(legion_index_launcher_t launcher_,
                                 legion_future_t future_)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  Future *future = CObjectWrapper::unwrap(future_);

  launcher->add_future(*future);
}

void
legion_index_launcher_add_wait_barrier(legion_index_launcher_t launcher_,
                                      legion_phase_barrier_t bar_)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  PhaseBarrier bar = CObjectWrapper::unwrap(bar_);

  launcher->add_wait_barrier(bar);
}

void
legion_index_launcher_add_arrival_barrier(legion_index_launcher_t launcher_,
                                         legion_phase_barrier_t bar_)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  PhaseBarrier bar = CObjectWrapper::unwrap(bar_);

  launcher->add_arrival_barrier(bar);
}

// -----------------------------------------------------------------------
// Inline Mapping Operations
// -----------------------------------------------------------------------

legion_inline_launcher_t
legion_inline_launcher_create_logical_region(
  legion_logical_region_t handle_,
  legion_privilege_mode_t priv,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t region_tag /* = 0 */,
  bool verified /* = false*/,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t launcher_tag /* = 0 */)
{
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  InlineLauncher *launcher = new InlineLauncher(
    RegionRequirement(handle, priv, prop, parent, region_tag, verified),
    id,
    launcher_tag);
  return CObjectWrapper::wrap(launcher);
}

void
legion_inline_launcher_destroy(legion_inline_launcher_t handle_)
{
  InlineLauncher *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

legion_physical_region_t
legion_inline_launcher_execute(legion_runtime_t runtime_,
                               legion_context_t ctx_,
                               legion_inline_launcher_t launcher_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  InlineLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  PhysicalRegion r = runtime->map_region(ctx, *launcher);
  return CObjectWrapper::wrap(new PhysicalRegion(r));
}

void
legion_inline_launcher_add_field(legion_inline_launcher_t launcher_,
                                 legion_field_id_t fid,
                                 bool inst /* = true */)
{
  InlineLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->add_field(fid, inst);
}

void
legion_runtime_remap_region(legion_runtime_t runtime_,
                            legion_context_t ctx_,
                            legion_physical_region_t region_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  PhysicalRegion *region = CObjectWrapper::unwrap(region_);

  runtime->remap_region(ctx, *region);
}

void
legion_runtime_unmap_region(legion_runtime_t runtime_,
                            legion_context_t ctx_,
                            legion_physical_region_t region_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  PhysicalRegion *region = CObjectWrapper::unwrap(region_);

  runtime->unmap_region(ctx, *region);
}

void
legion_runtime_unmap_all_regions(legion_runtime_t runtime_,
                                 legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  runtime->unmap_all_regions(ctx);
}

// -----------------------------------------------------------------------
// Fill Field Operations
// -----------------------------------------------------------------------

void
legion_runtime_fill_field(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_logical_region_t handle_,
  legion_logical_region_t parent_,
  legion_field_id_t fid,
  const void *value,
  size_t value_size,
  legion_predicate_t pred_ /* = legion_predicate_true() */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  runtime->fill_field(ctx, handle, parent, fid, value, value_size, *pred);
}

void
legion_runtime_fill_field_future(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_logical_region_t handle_,
  legion_logical_region_t parent_,
  legion_field_id_t fid,
  legion_future_t f_,
  legion_predicate_t pred_ /* = legion_predicate_true() */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  Future *f = CObjectWrapper::unwrap(f_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  runtime->fill_field(ctx, handle, parent, fid, *f, *pred);
}

// -----------------------------------------------------------------------
// File Operations
// -----------------------------------------------------------------------

legion_field_map_t
legion_field_map_create()
{
  std::map<FieldID, const char *> *result =
    new std::map<FieldID, const char *>();

  return CObjectWrapper::wrap(result);
}

void
legion_field_map_destroy(legion_field_map_t handle_)
{
  std::map<FieldID, const char *> *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

void
legion_field_map_insert(legion_field_map_t handle_,
                        legion_field_id_t key,
                        const char *value)
{
  std::map<FieldID, const char *> *handle = CObjectWrapper::unwrap(handle_);

  handle->insert(std::pair<FieldID, const char *>(key, value));
}

legion_physical_region_t
legion_runtime_attach_hdf5(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  const char *filename,
  legion_logical_region_t handle_,
  legion_logical_region_t parent_,
  legion_field_map_t field_map_,
  legion_file_mode_t mode)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  std::map<FieldID, const char *> *field_map =
    CObjectWrapper::unwrap(field_map_);

  AttachLauncher launcher(EXTERNAL_HDF5_FILE, handle, parent);
  launcher.attach_hdf5(filename, *field_map, mode);

  PhysicalRegion result = runtime->attach_external_resource(ctx, launcher);

  return CObjectWrapper::wrap(new PhysicalRegion(result));
}

void
legion_runtime_detach_hdf5(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_physical_region_t region_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  PhysicalRegion *region = CObjectWrapper::unwrap(region_);

  runtime->detach_external_resource(ctx, *region);
}

// -----------------------------------------------------------------------
// Copy Operations
// -----------------------------------------------------------------------

legion_copy_launcher_t
legion_copy_launcher_create(
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t launcher_tag /* = 0 */)
{
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  CopyLauncher *launcher = new CopyLauncher(*pred, id, launcher_tag);
  return CObjectWrapper::wrap(launcher);
}

void
legion_copy_launcher_destroy(legion_copy_launcher_t handle_)
{
  CopyLauncher *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

void
legion_copy_launcher_execute(legion_runtime_t runtime_,
                             legion_context_t ctx_,
                             legion_copy_launcher_t launcher_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  CopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  runtime->issue_copy_operation(ctx, *launcher);
}

unsigned
legion_copy_launcher_add_src_region_requirement_logical_region(
  legion_copy_launcher_t launcher_,
  legion_logical_region_t handle_,
  legion_privilege_mode_t priv,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool verified /* = false*/)
{
  CopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->src_requirements.size();
  launcher->src_requirements.push_back(
    RegionRequirement(handle, priv, prop, parent, tag, verified));
  return idx;
}

unsigned
legion_copy_launcher_add_dst_region_requirement_logical_region(
  legion_copy_launcher_t launcher_,
  legion_logical_region_t handle_,
  legion_privilege_mode_t priv,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool verified /* = false*/)
{
  CopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->dst_requirements.size();
  launcher->dst_requirements.push_back(
    RegionRequirement(handle, priv, prop, parent, tag, verified));
  return idx;
}

unsigned
legion_copy_launcher_add_dst_region_requirement_logical_region_reduction(
  legion_copy_launcher_t launcher_,
  legion_logical_region_t handle_,
  legion_reduction_op_id_t redop,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool verified /* = false*/)
{
  CopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->dst_requirements.size();
  launcher->dst_requirements.push_back(
    RegionRequirement(handle, redop, prop, parent, tag, verified));
  return idx;
}

void
legion_copy_launcher_add_src_field(legion_copy_launcher_t launcher_,
                                   unsigned idx,
                                   legion_field_id_t fid,
                                   bool inst /* = true */)
{
  CopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->add_src_field(idx, fid, inst);
}

void
legion_copy_launcher_add_dst_field(legion_copy_launcher_t launcher_,
                                   unsigned idx,
                                   legion_field_id_t fid,
                                   bool inst /* = true */)
{
  CopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->add_dst_field(idx, fid, inst);
}

void
legion_copy_launcher_add_wait_barrier(legion_copy_launcher_t launcher_,
                                      legion_phase_barrier_t bar_)
{
  CopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  PhaseBarrier bar = CObjectWrapper::unwrap(bar_);

  launcher->add_wait_barrier(bar);
}

void
legion_copy_launcher_add_arrival_barrier(legion_copy_launcher_t launcher_,
                                         legion_phase_barrier_t bar_)
{
  CopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  PhaseBarrier bar = CObjectWrapper::unwrap(bar_);

  launcher->add_arrival_barrier(bar);
}

// -----------------------------------------------------------------------
// Index Copy Operations
// -----------------------------------------------------------------------

legion_index_copy_launcher_t
legion_index_copy_launcher_create(
  legion_domain_t domain_,
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t launcher_tag /* = 0 */)
{
  Domain domain = CObjectWrapper::unwrap(domain_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  IndexCopyLauncher *launcher = new IndexCopyLauncher(domain, *pred, id, launcher_tag);
  return CObjectWrapper::wrap(launcher);
}

void
legion_index_copy_launcher_destroy(legion_index_copy_launcher_t handle_)
{
  IndexCopyLauncher *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

void
legion_index_copy_launcher_execute(legion_runtime_t runtime_,
                                   legion_context_t ctx_,
                                   legion_index_copy_launcher_t launcher_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  runtime->issue_copy_operation(ctx, *launcher);
}

unsigned
legion_index_copy_launcher_add_src_region_requirement_logical_region(
  legion_index_copy_launcher_t launcher_,
  legion_logical_region_t handle_,
  legion_projection_id_t proj /* = 0 */,
  legion_privilege_mode_t priv,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool verified /* = false*/)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->src_requirements.size();
  launcher->src_requirements.push_back(
    RegionRequirement(handle, proj, priv, prop, parent, tag, verified));
  return idx;
}

unsigned
legion_index_copy_launcher_add_dst_region_requirement_logical_region(
  legion_index_copy_launcher_t launcher_,
  legion_logical_region_t handle_,
  legion_projection_id_t proj /* = 0 */,
  legion_privilege_mode_t priv,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool verified /* = false*/)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->dst_requirements.size();
  launcher->dst_requirements.push_back(
    RegionRequirement(handle, proj, priv, prop, parent, tag, verified));
  return idx;
}

unsigned
legion_index_copy_launcher_add_src_region_requirement_logical_partition(
  legion_index_copy_launcher_t launcher_,
  legion_logical_partition_t handle_,
  legion_projection_id_t proj /* = 0 */,
  legion_privilege_mode_t priv,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool verified /* = false*/)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->src_requirements.size();
  launcher->src_requirements.push_back(
    RegionRequirement(handle, proj, priv, prop, parent, tag, verified));
  return idx;
}

unsigned
legion_index_copy_launcher_add_dst_region_requirement_logical_partition(
  legion_index_copy_launcher_t launcher_,
  legion_logical_partition_t handle_,
  legion_projection_id_t proj /* = 0 */,
  legion_privilege_mode_t priv,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool verified /* = false*/)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->dst_requirements.size();
  launcher->dst_requirements.push_back(
    RegionRequirement(handle, proj, priv, prop, parent, tag, verified));
  return idx;
}

unsigned
legion_index_copy_launcher_add_dst_region_requirement_logical_region_reduction(
  legion_index_copy_launcher_t launcher_,
  legion_logical_region_t handle_,
  legion_projection_id_t proj /* = 0 */,
  legion_reduction_op_id_t redop,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool verified /* = false*/)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->dst_requirements.size();
  launcher->dst_requirements.push_back(
    RegionRequirement(handle, redop, proj, prop, parent, tag, verified));
  return idx;
}

unsigned
legion_index_copy_launcher_add_dst_region_requirement_logical_partition_reduction(
  legion_index_copy_launcher_t launcher_,
  legion_logical_partition_t handle_,
  legion_projection_id_t proj /* = 0 */,
  legion_reduction_op_id_t redop,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool verified /* = false*/)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->dst_requirements.size();
  launcher->dst_requirements.push_back(
    RegionRequirement(handle, redop, proj, prop, parent, tag, verified));
  return idx;
}

void
legion_index_copy_launcher_add_src_field(legion_index_copy_launcher_t launcher_,
                                         unsigned idx,
                                         legion_field_id_t fid,
                                         bool inst /* = true */)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->add_src_field(idx, fid, inst);
}

void
legion_index_copy_launcher_add_dst_field(legion_index_copy_launcher_t launcher_,
                                         unsigned idx,
                                         legion_field_id_t fid,
                                         bool inst /* = true */)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->add_dst_field(idx, fid, inst);
}

void
legion_index_copy_launcher_add_wait_barrier(legion_index_copy_launcher_t launcher_,
                                            legion_phase_barrier_t bar_)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  PhaseBarrier bar = CObjectWrapper::unwrap(bar_);

  launcher->add_wait_barrier(bar);
}

void
legion_index_copy_launcher_add_arrival_barrier(legion_index_copy_launcher_t launcher_,
                                               legion_phase_barrier_t bar_)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  PhaseBarrier bar = CObjectWrapper::unwrap(bar_);

  launcher->add_arrival_barrier(bar);
}

// -----------------------------------------------------------------------
// Acquire Operations
// -----------------------------------------------------------------------

legion_acquire_launcher_t
legion_acquire_launcher_create(
  legion_logical_region_t logical_region_,
  legion_logical_region_t parent_region_,
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t tag /* = 0 */)
{
  LogicalRegion logical_region = CObjectWrapper::unwrap(logical_region_);
  LogicalRegion parent_region = CObjectWrapper::unwrap(parent_region_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  AcquireLauncher *launcher =
    new AcquireLauncher(logical_region, parent_region, PhysicalRegion(),
                        *pred, id, tag);
  return CObjectWrapper::wrap(launcher);
}

void
legion_acquire_launcher_destroy(legion_acquire_launcher_t handle_)
{
  AcquireLauncher *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

void
legion_acquire_launcher_execute(legion_runtime_t runtime_,
                                legion_context_t ctx_,
                                legion_acquire_launcher_t launcher_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  AcquireLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  runtime->issue_acquire(ctx, *launcher);
}

void
legion_acquire_launcher_add_field(legion_acquire_launcher_t launcher_,
                                  legion_field_id_t fid)
{
  AcquireLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->add_field(fid);
}

void
legion_acquire_launcher_add_wait_barrier(legion_acquire_launcher_t launcher_,
                                         legion_phase_barrier_t bar_)
{
  AcquireLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  PhaseBarrier bar = CObjectWrapper::unwrap(bar_);

  launcher->add_wait_barrier(bar);
}

void
legion_acquire_launcher_add_arrival_barrier(
  legion_acquire_launcher_t launcher_,
  legion_phase_barrier_t bar_)
{
  AcquireLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  PhaseBarrier bar = CObjectWrapper::unwrap(bar_);

  launcher->add_arrival_barrier(bar);
}

// -----------------------------------------------------------------------
// Release Operations
// -----------------------------------------------------------------------

legion_release_launcher_t
legion_release_launcher_create(
  legion_logical_region_t logical_region_,
  legion_logical_region_t parent_region_,
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t tag /* = 0 */)
{
  LogicalRegion logical_region = CObjectWrapper::unwrap(logical_region_);
  LogicalRegion parent_region = CObjectWrapper::unwrap(parent_region_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  ReleaseLauncher *launcher =
    new ReleaseLauncher(logical_region, parent_region, PhysicalRegion(),
                        *pred, id, tag);
  return CObjectWrapper::wrap(launcher);
}

void
legion_release_launcher_destroy(legion_release_launcher_t handle_)
{
  ReleaseLauncher *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

void
legion_release_launcher_execute(legion_runtime_t runtime_,
                                legion_context_t ctx_,
                                legion_release_launcher_t launcher_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  ReleaseLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  runtime->issue_release(ctx, *launcher);
}

void
legion_release_launcher_add_field(legion_release_launcher_t launcher_,
                                  legion_field_id_t fid)
{
  ReleaseLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->add_field(fid);
}

void
legion_release_launcher_add_wait_barrier(legion_release_launcher_t launcher_,
                                         legion_phase_barrier_t bar_)
{
  ReleaseLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  PhaseBarrier bar = CObjectWrapper::unwrap(bar_);

  launcher->add_wait_barrier(bar);
}

void
legion_release_launcher_add_arrival_barrier(
  legion_release_launcher_t launcher_,
  legion_phase_barrier_t bar_)
{
  ReleaseLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  PhaseBarrier bar = CObjectWrapper::unwrap(bar_);

  launcher->add_arrival_barrier(bar);
}

// -----------------------------------------------------------------------
// Attach/Detach Operations
// -----------------------------------------------------------------------

legion_attach_launcher_t
legion_attach_launcher_create(legion_logical_region_t logical_region_,
                              legion_logical_region_t parent_region_,
                              legion_external_resource_t resource)
{
  LogicalRegion logical_region = CObjectWrapper::unwrap(logical_region_);
  LogicalRegion parent_region = CObjectWrapper::unwrap(parent_region_);

  AttachLauncher *launcher = 
    new AttachLauncher(resource, logical_region, parent_region);
  return CObjectWrapper::wrap(launcher);
}

void
legion_attach_launcher_destroy(legion_attach_launcher_t handle_)
{
  AttachLauncher *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

legion_physical_region_t
legion_attach_launcher_execute(legion_runtime_t runtime_,
                               legion_context_t ctx_,
                               legion_attach_launcher_t launcher_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  AttachLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  PhysicalRegion region = runtime->attach_external_resource(ctx, *launcher);
  return CObjectWrapper::wrap(new PhysicalRegion(region));
}

void
legion_attach_launcher_add_cpu_soa_field(legion_attach_launcher_t launcher_,
                                         legion_field_id_t fid,
                                         void *base_ptr,
                                         bool column_major)
{
  AttachLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  std::vector<FieldID> fields(1, fid);
  // Find the memory that we are using
  const Memory local_sysmem = Machine::MemoryQuery(Machine::get_machine())
      .has_affinity_to(Processor::get_executing_processor())
      .only_kind(Memory::SYSTEM_MEM)
      .first();
  launcher->attach_array_soa(base_ptr, column_major, fields, local_sysmem);
}

legion_future_t
legion_detach_external_resource(legion_runtime_t runtime_,
                                legion_context_t ctx_,
                                legion_physical_region_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);

  Future *result = new Future(
      runtime->detach_external_resource(ctx, *handle));
  return CObjectWrapper::wrap(result);
}

// -----------------------------------------------------------------------
// Must Epoch Operations
// -----------------------------------------------------------------------

legion_must_epoch_launcher_t
legion_must_epoch_launcher_create(
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t launcher_tag /* = 0 */)
{
  MustEpochLauncher *launcher = new MustEpochLauncher(id, launcher_tag);
  return CObjectWrapper::wrap(launcher);
}

void
legion_must_epoch_launcher_destroy(legion_must_epoch_launcher_t handle_)
{
  MustEpochLauncher *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

legion_future_map_t
legion_must_epoch_launcher_execute(legion_runtime_t runtime_,
                                   legion_context_t ctx_,
                                   legion_must_epoch_launcher_t launcher_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  MustEpochLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  FutureMap f = runtime->execute_must_epoch(ctx, *launcher);
  return CObjectWrapper::wrap(new FutureMap(f));
}

void
legion_must_epoch_launcher_add_single_task(
  legion_must_epoch_launcher_t launcher_,
  legion_domain_point_t point_,
  legion_task_launcher_t handle_)
{
  MustEpochLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  DomainPoint point = CObjectWrapper::unwrap(point_);
  {
    TaskLauncher *handle = CObjectWrapper::unwrap(handle_);
    launcher->add_single_task(point, *handle);
  }

  // Destroy handle.
  legion_task_launcher_destroy(handle_);
}

void
legion_must_epoch_launcher_add_index_task(
  legion_must_epoch_launcher_t launcher_,
  legion_index_launcher_t handle_)
{
  MustEpochLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  {
    IndexTaskLauncher *handle = CObjectWrapper::unwrap(handle_);
    launcher->add_index_task(*handle);
  }

  // Destroy handle.
  legion_index_launcher_destroy(handle_);
}

// -----------------------------------------------------------------------
// Tracing Operations
// -----------------------------------------------------------------------

void
legion_runtime_begin_trace(legion_runtime_t runtime_,
                           legion_context_t ctx_,
                           legion_trace_id_t tid)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  runtime->begin_trace(ctx, tid);
}

void
legion_runtime_end_trace(legion_runtime_t runtime_,
                         legion_context_t ctx_,
                         legion_trace_id_t tid)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  runtime->end_trace(ctx, tid);
}

// -----------------------------------------------------------------------
// Fence Operations
// -----------------------------------------------------------------------

void
legion_runtime_issue_mapping_fence(legion_runtime_t runtime_,
                                   legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  runtime->issue_mapping_fence(ctx);
}

void
legion_runtime_issue_execution_fence(legion_runtime_t runtime_,
                                     legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  runtime->issue_execution_fence(ctx);
}

// -----------------------------------------------------------------------
// Tunable Variables
// -----------------------------------------------------------------------

legion_future_t
legion_runtime_select_tunable_value(legion_runtime_t runtime_,
				    legion_context_t ctx_,
				    legion_tunable_id_t tid,
				    legion_mapper_id_t mapper /* = 0 */,
				    legion_mapping_tag_id_t tag /* = 0 */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  Future f = runtime->select_tunable_value(ctx, tid, mapper, tag);
  return CObjectWrapper::wrap(new Future(f));
}

// -----------------------------------------------------------------------
// Miscellaneous Operations
// -----------------------------------------------------------------------

legion_runtime_t
legion_runtime_get_runtime()
{
  Runtime *runtime = Runtime::get_runtime();
  return CObjectWrapper::wrap(runtime);
}

legion_processor_t
legion_runtime_get_executing_processor(legion_runtime_t runtime_,
                                       legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  Processor proc = runtime->get_executing_processor(ctx);
  return CObjectWrapper::wrap(proc);
}

// -----------------------------------------------------------------------
// Physical Data Operations
// -----------------------------------------------------------------------

void
legion_physical_region_destroy(legion_physical_region_t handle_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

bool
legion_physical_region_is_mapped(legion_physical_region_t handle_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);

  return handle->is_mapped();
}

void
legion_physical_region_wait_until_valid(legion_physical_region_t handle_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);

  handle->wait_until_valid();
}

bool
legion_physical_region_is_valid(legion_physical_region_t handle_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);

  return handle->is_valid();
}

legion_logical_region_t
legion_physical_region_get_logical_region(legion_physical_region_t handle_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);

  LogicalRegion region = handle->get_logical_region();
  return CObjectWrapper::wrap(region);
}

size_t
legion_physical_region_get_field_count(legion_physical_region_t handle_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);
  std::vector<FieldID> fields;
  handle->get_fields(fields);
  return fields.size();
}

legion_field_id_t
legion_physical_region_get_field_id(legion_physical_region_t handle_, size_t index)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);
  std::vector<FieldID> fields;
  handle->get_fields(fields);
  assert((index >= 0) && (index < fields.size()));
  return fields[index];
}

legion_accessor_array_1d_t
legion_physical_region_get_field_accessor_array_1d(
  legion_physical_region_t handle_,
  legion_field_id_t fid)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);
  UnsafeFieldAccessor<char,1,coord_t,Realm::AffineAccessor<char,1,coord_t> >
    *accessor = new UnsafeFieldAccessor<char,1,coord_t,
                      Realm::AffineAccessor<char,1,coord_t> >(*handle, fid);

  return CObjectWrapper::wrap(accessor);
}

void
legion_accessor_array_1d_destroy(legion_accessor_array_1d_t handle_)
{
  UnsafeFieldAccessor<char,1,coord_t,Realm::AffineAccessor<char,1,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

legion_accessor_array_2d_t
legion_physical_region_get_field_accessor_array_2d(
  legion_physical_region_t handle_,
  legion_field_id_t fid)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);
  UnsafeFieldAccessor<char,2,coord_t,Realm::AffineAccessor<char,2,coord_t> >
    *accessor = new UnsafeFieldAccessor<char,2,coord_t,
                      Realm::AffineAccessor<char,2,coord_t> >(*handle, fid);

  return CObjectWrapper::wrap(accessor);
}

void
legion_accessor_array_2d_destroy(legion_accessor_array_2d_t handle_)
{
  UnsafeFieldAccessor<char,2,coord_t,Realm::AffineAccessor<char,2,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

legion_accessor_array_3d_t
legion_physical_region_get_field_accessor_array_3d(
  legion_physical_region_t handle_,
  legion_field_id_t fid)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);
  UnsafeFieldAccessor<char,3,coord_t,Realm::AffineAccessor<char,3,coord_t> >
    *accessor = new UnsafeFieldAccessor<char,3,coord_t,
                      Realm::AffineAccessor<char,3,coord_t> >(*handle, fid);

  return CObjectWrapper::wrap(accessor);
}

void
legion_accessor_array_3d_destroy(legion_accessor_array_3d_t handle_)
{
  UnsafeFieldAccessor<char,3,coord_t,Realm::AffineAccessor<char,3,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

void *
legion_accessor_array_1d_raw_rect_ptr(legion_accessor_array_1d_t handle_,
                                      legion_rect_1d_t rect_,
                                      legion_rect_1d_t *subrect_,
                                      legion_byte_offset_t *offsets_)
{
  UnsafeFieldAccessor<char,1,coord_t,Realm::AffineAccessor<char,1,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);
  Rect1D rect = CObjectWrapper::unwrap(rect_);

  void *data = handle->ptr(rect.lo);
  *subrect_ = CObjectWrapper::wrap(rect); // no checks
  offsets_[0] = CObjectWrapper::wrap(handle->accessor.strides[0]);
  return data;
}

void *
legion_accessor_array_2d_raw_rect_ptr(legion_accessor_array_2d_t handle_,
                                      legion_rect_2d_t rect_,
                                      legion_rect_2d_t *subrect_,
                                      legion_byte_offset_t *offsets_)
{
  UnsafeFieldAccessor<char,2,coord_t,Realm::AffineAccessor<char,2,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);
  Rect2D rect = CObjectWrapper::unwrap(rect_);

  void *data = handle->ptr(rect.lo);
  *subrect_ = CObjectWrapper::wrap(rect); // no checks
  offsets_[0] = CObjectWrapper::wrap(handle->accessor.strides[0]);
  offsets_[1] = CObjectWrapper::wrap(handle->accessor.strides[1]);
  return data;
}

void *
legion_accessor_array_3d_raw_rect_ptr(legion_accessor_array_3d_t handle_,
                                      legion_rect_3d_t rect_,
                                      legion_rect_3d_t *subrect_,
                                      legion_byte_offset_t *offsets_)
{
  UnsafeFieldAccessor<char,3,coord_t,Realm::AffineAccessor<char,3,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);
  Rect3D rect = CObjectWrapper::unwrap(rect_);

  void *data = handle->ptr(rect.lo);
  *subrect_ = CObjectWrapper::wrap(rect); // no checks
  offsets_[0] = CObjectWrapper::wrap(handle->accessor.strides[0]);
  offsets_[1] = CObjectWrapper::wrap(handle->accessor.strides[1]);
  offsets_[2] = CObjectWrapper::wrap(handle->accessor.strides[2]);
  return data;
}

void
legion_accessor_array_1d_read(legion_accessor_array_1d_t handle_,
                              legion_ptr_t ptr_,
                              void *dst, size_t bytes)
{
  UnsafeFieldAccessor<char,1,coord_t,Realm::AffineAccessor<char,1,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);
  ptr_t ptr = CObjectWrapper::unwrap(ptr_);

  memcpy(dst, handle->ptr(ptr.value), bytes);
}

void
legion_accessor_array_1d_read_point(legion_accessor_array_1d_t handle_,
                                    legion_point_1d_t point_,
                                    void *dst, size_t bytes)
{
  UnsafeFieldAccessor<char,1,coord_t,Realm::AffineAccessor<char,1,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);
  Point1D point = CObjectWrapper::unwrap(point_);

  memcpy(dst, handle->ptr(point), bytes);
}

void
legion_accessor_array_2d_read_point(legion_accessor_array_2d_t handle_,
                                    legion_point_2d_t point_,
                                    void *dst, size_t bytes)
{
  UnsafeFieldAccessor<char,2,coord_t,Realm::AffineAccessor<char,2,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);
  Point2D point = CObjectWrapper::unwrap(point_);

  memcpy(dst, handle->ptr(point), bytes);
}

void
legion_accessor_array_3d_read_point(legion_accessor_array_3d_t handle_,
                                    legion_point_3d_t point_,
                                    void *dst, size_t bytes)
{
  UnsafeFieldAccessor<char,3,coord_t,Realm::AffineAccessor<char,3,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);
  Point3D point = CObjectWrapper::unwrap(point_);

  memcpy(dst, handle->ptr(point), bytes);
}

void
legion_accessor_array_1d_write(legion_accessor_array_1d_t handle_,
                               legion_ptr_t ptr_,
                               const void *src, size_t bytes)
{
  UnsafeFieldAccessor<char,1,coord_t,Realm::AffineAccessor<char,1,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);
  ptr_t ptr = CObjectWrapper::unwrap(ptr_);

  memcpy(handle->ptr(ptr.value), src, bytes); 
}

void
legion_accessor_array_1d_write_point(legion_accessor_array_1d_t handle_,
                                     legion_point_1d_t point_,
                                     const void *src, size_t bytes)
{
  UnsafeFieldAccessor<char,1,coord_t,Realm::AffineAccessor<char,1,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);
  Point1D point = CObjectWrapper::unwrap(point_);

  memcpy(handle->ptr(point), src, bytes);
}

void
legion_accessor_array_2d_write_point(legion_accessor_array_2d_t handle_,
                                     legion_point_2d_t point_,
                                     const void *src, size_t bytes)
{
  UnsafeFieldAccessor<char,2,coord_t,Realm::AffineAccessor<char,2,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);
  Point2D point = CObjectWrapper::unwrap(point_);

  memcpy(handle->ptr(point), src, bytes);
}

void
legion_accessor_array_3d_write_point(legion_accessor_array_3d_t handle_,
                                     legion_point_3d_t point_,
                                     const void *src, size_t bytes)
{
  UnsafeFieldAccessor<char,3,coord_t,Realm::AffineAccessor<char,3,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);
  Point3D point = CObjectWrapper::unwrap(point_);

  memcpy(handle->ptr(point), src, bytes);
}

void *
legion_accessor_array_1d_ref(legion_accessor_array_1d_t handle_,
                             legion_ptr_t ptr_)
{
  UnsafeFieldAccessor<char,1,coord_t,Realm::AffineAccessor<char,1,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);
  ptr_t ptr = CObjectWrapper::unwrap(ptr_);

  return handle->ptr(ptr.value);
}

void *
legion_accessor_array_1d_ref_point(legion_accessor_array_1d_t handle_,
                                   legion_point_1d_t point_)
{
  UnsafeFieldAccessor<char,1,coord_t,Realm::AffineAccessor<char,1,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);
  Point1D point = CObjectWrapper::unwrap(point_);

  return handle->ptr(point);
}

void *
legion_accessor_array_2d_ref_point(legion_accessor_array_2d_t handle_,
                                   legion_point_2d_t point_)
{
  UnsafeFieldAccessor<char,2,coord_t,Realm::AffineAccessor<char,2,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);
  Point2D point = CObjectWrapper::unwrap(point_);

  return handle->ptr(point);
}

void *
legion_accessor_array_3d_ref_point(legion_accessor_array_3d_t handle_,
                                   legion_point_3d_t point_)
{
  UnsafeFieldAccessor<char,3,coord_t,Realm::AffineAccessor<char,3,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);
  Point3D point = CObjectWrapper::unwrap(point_);

  return handle->ptr(point);
}

legion_index_iterator_t
legion_index_iterator_create(legion_runtime_t runtime_,
                             legion_context_t ctx_,
                             legion_index_space_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  IndexIterator *iterator = new IndexIterator(runtime, ctx, handle);
  return CObjectWrapper::wrap(iterator);
}

void
legion_index_iterator_destroy(legion_index_iterator_t handle_)
{
  IndexIterator *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

bool
legion_index_iterator_has_next(legion_index_iterator_t handle_)
{
  IndexIterator *handle = CObjectWrapper::unwrap(handle_);

  return handle->has_next();
}

legion_ptr_t
legion_index_iterator_next(legion_index_iterator_t handle_)
{
  IndexIterator *handle = CObjectWrapper::unwrap(handle_);

  return CObjectWrapper::wrap(handle->next());
}

legion_ptr_t
legion_index_iterator_next_span(legion_index_iterator_t handle_,
                                size_t *act_count,
                                size_t req_count)
{
  IndexIterator *handle = CObjectWrapper::unwrap(handle_);

  return CObjectWrapper::wrap(handle->next_span(*act_count, req_count));
}

//------------------------------------------------------------------------
// Task Operations
//------------------------------------------------------------------------

legion_unique_id_t
legion_context_get_unique_id(legion_context_t ctx_)
{
  Task* task =
    reinterpret_cast<Task*>(CObjectWrapper::unwrap(ctx_)->context());
  return task->get_unique_id();
}

legion_unique_id_t
legion_task_get_unique_id(legion_task_t task_)
{
  return CObjectWrapper::unwrap(task_)->get_unique_id();
}

int
legion_task_get_depth(legion_task_t task_)
{
  return CObjectWrapper::unwrap(task_)->get_depth();
}

legion_mapping_tag_id_t
legion_task_get_tag(legion_task_t task_)
{
  return CObjectWrapper::unwrap(task_)->tag;
}

void
legion_task_id_attach_name(legion_runtime_t runtime_,
                           legion_task_id_t task_id,
                           const char *name,
                           bool is_mutable /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);

  runtime->attach_name(task_id, name, is_mutable);
}

void
legion_task_id_retrieve_name(legion_runtime_t runtime_,
                             legion_task_id_t task_id,
                             const char **result)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);

  runtime->retrieve_name(task_id, *result);
}

void *
legion_task_get_args(legion_task_t task_)
{
  Task *task = CObjectWrapper::unwrap(task_);

  return task->args;
}

size_t
legion_task_get_arglen(legion_task_t task_)
{
  Task *task = CObjectWrapper::unwrap(task_);

  return task->arglen;
}

legion_domain_t
legion_task_get_index_domain(legion_task_t task_)
{
  Task *task = CObjectWrapper::unwrap(task_);

  return CObjectWrapper::wrap(task->index_domain);
}

legion_domain_point_t
legion_task_get_index_point(legion_task_t task_)
{
  Task *task = CObjectWrapper::unwrap(task_);

  return CObjectWrapper::wrap(task->index_point);
}

bool
legion_task_get_is_index_space(legion_task_t task_)
{
  Task *task = CObjectWrapper::unwrap(task_);

  return task->is_index_space;
}

void *
legion_task_get_local_args(legion_task_t task_)
{
  Task *task = CObjectWrapper::unwrap(task_);

  return task->local_args;
}

size_t
legion_task_get_local_arglen(legion_task_t task_)
{
  Task *task = CObjectWrapper::unwrap(task_);

  return task->local_arglen;
}

unsigned
legion_task_get_regions_size(legion_task_t task_)
{
  Task *task = CObjectWrapper::unwrap(task_);

  return task->regions.size();
}

legion_region_requirement_t
legion_task_get_region(legion_task_t task_, unsigned idx)
{
  Task *task = CObjectWrapper::unwrap(task_);

  return CObjectWrapper::wrap(&task->regions[idx]);
}

unsigned
legion_task_get_futures_size(legion_task_t task_)
{
  Task *task = CObjectWrapper::unwrap(task_);

  return task->futures.size();
}

legion_future_t
legion_task_get_future(legion_task_t task_, unsigned idx)
{
  Task *task = CObjectWrapper::unwrap(task_);
  Future future = task->futures[idx];

  return CObjectWrapper::wrap(new Future(future));
}

legion_task_id_t
legion_task_get_task_id(legion_task_t task_)
{
  Task *task = CObjectWrapper::unwrap(task_);

  return task->task_id;
}

legion_processor_t
legion_task_get_target_proc(legion_task_t task_)
{
  Task *task = CObjectWrapper::unwrap(task_);

  return CObjectWrapper::wrap(task->target_proc);
}

const char *
legion_task_get_name(legion_task_t task_)
{
  Task *task = CObjectWrapper::unwrap(task_);

  return task->get_task_name();
}

// -----------------------------------------------------------------------
// Inline Operations
// -----------------------------------------------------------------------

legion_region_requirement_t
legion_inline_get_requirement(legion_inline_t inline_operation_)
{
  InlineMapping *inline_operation = 
    CObjectWrapper::unwrap(inline_operation_);

  return CObjectWrapper::wrap(&inline_operation->requirement);
}

//------------------------------------------------------------------------
// Execution Constraints
//------------------------------------------------------------------------

legion_execution_constraint_set_t
legion_execution_constraint_set_create(void)
{
  ExecutionConstraintSet *constraints = new ExecutionConstraintSet();

  return CObjectWrapper::wrap(constraints);
}

void
legion_execution_constraint_set_destroy(
  legion_execution_constraint_set_t handle_)
{
  ExecutionConstraintSet *constraints = CObjectWrapper::unwrap(handle_);

  delete constraints;
}

void
legion_execution_constraint_set_add_isa_constraint(
  legion_execution_constraint_set_t handle_,
  uint64_t prop)
{
  ExecutionConstraintSet *constraints = CObjectWrapper::unwrap(handle_);

  constraints->add_constraint(ISAConstraint(prop));
}

void
legion_execution_constraint_set_add_processor_constraint(
  legion_execution_constraint_set_t handle_,
  legion_processor_kind_t proc_kind_)
{
  ExecutionConstraintSet *constraints = CObjectWrapper::unwrap(handle_);
  Processor::Kind proc_kind = CObjectWrapper::unwrap(proc_kind_);

  constraints->add_constraint(ProcessorConstraint(proc_kind));
}

void
legion_execution_constraint_set_add_resource_constraint(
  legion_execution_constraint_set_t handle_,
  legion_resource_constraint_t resource,
  legion_equality_kind_t eq,
  size_t value)
{
  ExecutionConstraintSet *constraints = CObjectWrapper::unwrap(handle_);

  constraints->add_constraint(ResourceConstraint(resource, eq, value));
}

void
legion_execution_constraint_set_add_launch_constraint(
  legion_execution_constraint_set_t handle_,
  legion_launch_constraint_t kind,
  size_t value)
{
  ExecutionConstraintSet *constraints = CObjectWrapper::unwrap(handle_);

  constraints->add_constraint(LaunchConstraint(kind, value));
}

void
legion_execution_constraint_set_add_launch_constraint_multi_dim(
  legion_execution_constraint_set_t handle_,
  legion_launch_constraint_t kind,
  const size_t *values,
  int dims)
{
  ExecutionConstraintSet *constraints = CObjectWrapper::unwrap(handle_);

  constraints->add_constraint(LaunchConstraint(kind, values, dims));
}

void
legion_execution_constraint_set_add_colocation_constraints(
  legion_execution_constraint_set_t handle_,
  const unsigned *indexes,
  size_t num_indexes,
  const legion_field_id_t *fields,
  size_t num_fields)
{
  ExecutionConstraintSet *constraints = CObjectWrapper::unwrap(handle_);
  std::vector<unsigned> actual_indexes(num_indexes);
  for (unsigned idx = 0; idx < num_indexes; idx++)
    actual_indexes[idx] = indexes[idx];
  std::set<FieldID> all_fields;
  for (unsigned idx = 0; idx < num_fields; idx++)
    all_fields.insert(fields[idx]);

  constraints->add_constraint(ColocationConstraint(actual_indexes, all_fields));
}

//------------------------------------------------------------------------
// Layout Constraints
//------------------------------------------------------------------------

legion_layout_constraint_set_t
legion_layout_constraint_set_create(void)
{
  LayoutConstraintSet *constraints = new LayoutConstraintSet();

  return CObjectWrapper::wrap(constraints);
}

void
legion_layout_constraint_set_destroy(legion_layout_constraint_set_t handle_)
{
  LayoutConstraintSet *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

legion_layout_constraint_id_t
legion_layout_constraint_set_register(
  legion_runtime_t runtime_,
  legion_field_space_t fspace_,
  legion_layout_constraint_set_t handle_,
  const char *layout_name)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  FieldSpace fspace = CObjectWrapper::unwrap(fspace_);
  LayoutConstraintSet *constraints = CObjectWrapper::unwrap(handle_);

  LayoutConstraintRegistrar registrar(fspace, layout_name);
  registrar.layout_constraints = *constraints;

  return runtime->register_layout(registrar);
}

legion_layout_constraint_id_t
legion_layout_constraint_set_preregister(
  legion_layout_constraint_set_t handle_,
  const char *set_name)
{
  LayoutConstraintSet *constraints = CObjectWrapper::unwrap(handle_);

  LayoutConstraintRegistrar registrar(FieldSpace::NO_SPACE, set_name);
  registrar.layout_constraints = *constraints;

  return Runtime::preregister_layout(registrar);
}

void
legion_layout_constraint_set_release(
  legion_runtime_t runtime_,
  legion_layout_constraint_id_t handle)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);

  runtime->release_layout(handle);
}

void
legion_layout_constraint_set_add_specialized_constraint(
  legion_layout_constraint_set_t handle_,
  legion_specialized_constraint_t specialized,
  legion_reduction_op_id_t redop)
{
  LayoutConstraintSet *constraints = CObjectWrapper::unwrap(handle_);

  constraints->add_constraint(SpecializedConstraint(specialized, redop));
}

void
legion_layout_constraint_set_add_memory_constraint(
  legion_layout_constraint_set_t handle_,
  legion_memory_kind_t kind_)
{
  LayoutConstraintSet *constraints = CObjectWrapper::unwrap(handle_);
  Memory::Kind kind = CObjectWrapper::unwrap(kind_);

  constraints->add_constraint(MemoryConstraint(kind));
}

void
legion_layout_constraint_set_add_field_constraint(
  legion_layout_constraint_set_t handle_,
  const legion_field_id_t *fields, size_t num_fields,
  bool contiguous,
  bool inorder)
{
  LayoutConstraintSet *constraints = CObjectWrapper::unwrap(handle_);
  std::vector<FieldID> field_ids(num_fields);
  for (unsigned idx = 0; idx < num_fields; idx++)
    field_ids[idx] = fields[idx];
  
  constraints->add_constraint(FieldConstraint(field_ids, contiguous, inorder));
}

void
legion_layout_constraint_set_add_ordering_constraint(
 legion_layout_constraint_set_t handle_,
 const legion_dimension_kind_t *dims,
 size_t num_dims,
 bool contiguous)
{
  LayoutConstraintSet *constraints = CObjectWrapper::unwrap(handle_);
  std::vector<DimensionKind> ordering(num_dims);
  for (unsigned idx = 0; idx < num_dims; idx++)
    ordering[idx] = dims[idx];

  constraints->add_constraint(OrderingConstraint(ordering, contiguous));
}

void
legion_layout_constraint_set_add_splitting_constraint(
  legion_layout_constraint_set_t handle_,
  legion_dimension_kind_t dim)
{
  LayoutConstraintSet *constraints = CObjectWrapper::unwrap(handle_);

  constraints->add_constraint(SplittingConstraint(dim));
}

void
legion_layout_constraint_set_add_full_splitting_constraint(
  legion_layout_constraint_set_t handle_,
  legion_dimension_kind_t dim,
  size_t value)
{
  LayoutConstraintSet *constraints = CObjectWrapper::unwrap(handle_);

  constraints->add_constraint(SplittingConstraint(dim, value));
}

void
legion_layout_constraint_set_add_dimension_constraint(
  legion_layout_constraint_set_t handle_,
  legion_dimension_kind_t dim,
  legion_equality_kind_t eq, size_t value)
{
  LayoutConstraintSet *constraints = CObjectWrapper::unwrap(handle_);

  constraints->add_constraint(DimensionConstraint(dim, eq, value));
}

void
legion_layout_constraint_set_add_alignment_constraint(
  legion_layout_constraint_set_t handle_,
  legion_field_id_t field,
  legion_equality_kind_t eq,
  size_t byte_boundary)
{
  LayoutConstraintSet *constraints = CObjectWrapper::unwrap(handle_);

  constraints->add_constraint(AlignmentConstraint(field, eq, byte_boundary));
}

void
legion_layout_constraint_set_add_offset_constraint(
  legion_layout_constraint_set_t handle_,
  legion_field_id_t field,
  size_t offset)
{
  LayoutConstraintSet *constraints = CObjectWrapper::unwrap(handle_);

  constraints->add_constraint(OffsetConstraint(field, offset));
}

void
legion_layout_constraint_set_add_pointer_constraint(
  legion_layout_constraint_set_t handle_,
  legion_memory_t mem_,
  uintptr_t ptr)
{
  LayoutConstraintSet *constraints = CObjectWrapper::unwrap(handle_);
  Memory mem = CObjectWrapper::unwrap(mem_);

  constraints->add_constraint(PointerConstraint(mem, ptr)); 
}

// -----------------------------------------------------------------------
// Task Layout Constraints
// -----------------------------------------------------------------------

legion_task_layout_constraint_set_t
legion_task_layout_constraint_set_create(void)
{
  TaskLayoutConstraintSet *constraints = new TaskLayoutConstraintSet();

  return CObjectWrapper::wrap(constraints);
}

void
legion_task_layout_constraint_set_destroy(
  legion_task_layout_constraint_set_t handle_)
{
  TaskLayoutConstraintSet *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

void
legion_task_layout_constraint_set_add_layout_constraint(
  legion_task_layout_constraint_set_t handle_,
  unsigned idx,
  legion_layout_constraint_id_t layout)
{
  TaskLayoutConstraintSet *handle = CObjectWrapper::unwrap(handle_);

  handle->add_layout_constraint(idx, layout);
}

//------------------------------------------------------------------------
// Start-up Operations
//------------------------------------------------------------------------

int
legion_runtime_start(int argc,
                     char **argv,
                     bool background /* = false */)
{
  return Runtime::start(argc, argv, background);
}

void
legion_runtime_wait_for_shutdown(void)
{
  Runtime::wait_for_shutdown();
}

void
legion_runtime_set_top_level_task_id(legion_task_id_t top_id)
{
  Runtime::set_top_level_task_id(top_id);
}

const legion_input_args_t
legion_runtime_get_input_args(void)
{
  return CObjectWrapper::wrap_const(Runtime::get_input_args());
}

// List of callbacks registered.
static std::vector<legion_registration_callback_pointer_t> callbacks;

void
registration_callback_wrapper(Machine machine,
                              Runtime *rt,
                              const std::set<Processor> &local_procs)
{
  legion_machine_t machine_ = CObjectWrapper::wrap(&machine);
  legion_runtime_t rt_ = CObjectWrapper::wrap(rt);
  legion_processor_t local_procs_[local_procs.size()];

  unsigned idx = 0;
  for (std::set<Processor>::iterator itr = local_procs.begin();
      itr != local_procs.end(); ++itr)
  {
    const Processor& proc = *itr;
    local_procs_[idx++] = CObjectWrapper::wrap(proc);
  }

  for (std::vector<legion_registration_callback_pointer_t>::iterator itr = callbacks.begin();
      itr != callbacks.end(); ++itr)
  {
    (*itr)(machine_, rt_, local_procs_, idx);
  }
}

void
legion_runtime_add_registration_callback(
  legion_registration_callback_pointer_t callback_)
{
  static bool registered = false;
  if (!registered) {
    Runtime::add_registration_callback(registration_callback_wrapper);
    registered = true;
  }
  callbacks.push_back(callback_);
}

legion_mapper_id_t
legion_runtime_generate_library_mapper_ids(
    legion_runtime_t runtime_,
    const char *library_name,
    size_t count)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);

  return runtime->generate_library_mapper_ids(library_name, count);
}

void
legion_runtime_replace_default_mapper(
  legion_runtime_t runtime_,
  legion_mapper_t mapper_,
  legion_processor_t proc_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Mapper *mapper = CObjectWrapper::unwrap(mapper_);
  Processor proc = CObjectWrapper::unwrap(proc_);

  runtime->replace_default_mapper(mapper, proc);
}

class FunctorWrapper : public ProjectionFunctor {
public:
  FunctorWrapper(unsigned dep,
                 legion_projection_functor_logical_region_t region_fn,
                 legion_projection_functor_logical_partition_t partition_fn)
    : ProjectionFunctor()
    , depth(dep)
    , region_functor(region_fn)
    , partition_functor(partition_fn)
  {
  }

  FunctorWrapper(Runtime *rt, unsigned dep,
                 legion_projection_functor_logical_region_t region_fn,
                 legion_projection_functor_logical_partition_t partition_fn)
    : ProjectionFunctor(rt)
    , depth(dep)
    , region_functor(region_fn)
    , partition_functor(partition_fn)
  {
  }

  virtual LogicalRegion project(const Mappable *mappable,
                                unsigned index,
                                LogicalRegion upper_bound,
                                const DomainPoint &point)
  {
    legion_runtime_t runtime_ = CObjectWrapper::wrap(runtime);
    const legion_mappable_t mappable_ = CObjectWrapper::wrap_const(mappable);
    legion_logical_region_t upper_bound_ = CObjectWrapper::wrap(upper_bound);
    legion_domain_point_t point_ = CObjectWrapper::wrap(point);

    assert(region_functor);
    legion_logical_region_t result =
      region_functor(runtime_, mappable_, index, upper_bound_, point_);
    return CObjectWrapper::unwrap(result);
  }

  virtual LogicalRegion project(const Mappable *mappable,
                                unsigned index,
                                LogicalPartition upper_bound,
                                const DomainPoint &point)
  {
    legion_runtime_t runtime_ = CObjectWrapper::wrap(runtime);
    legion_mappable_t mappable_ = CObjectWrapper::wrap_const(mappable);
    legion_logical_partition_t upper_bound_ = CObjectWrapper::wrap(upper_bound);
    legion_domain_point_t point_ = CObjectWrapper::wrap(point);

    assert(partition_functor);
    legion_logical_region_t result =
      partition_functor(runtime_, mappable_, index, upper_bound_, point_);
    return CObjectWrapper::unwrap(result);
  }

  unsigned get_depth(void) const { return depth; }

private:
  const unsigned depth;
  legion_projection_functor_logical_region_t region_functor;
  legion_projection_functor_logical_partition_t partition_functor;
};

legion_projection_id_t
legion_runtime_generate_library_projection_ids(
    legion_runtime_t runtime_,
    const char *library_name,
    size_t count)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);

  return runtime->generate_library_projection_ids(library_name, count);
}

void
legion_runtime_preregister_projection_functor(
  legion_projection_id_t id,
  unsigned depth,
  legion_projection_functor_logical_region_t region_functor,
  legion_projection_functor_logical_partition_t partition_functor)
{
  FunctorWrapper *functor =
    new FunctorWrapper(depth, region_functor, partition_functor);
  Runtime::preregister_projection_functor(id, functor);
}

void
legion_runtime_register_projection_functor(
  legion_runtime_t runtime_,
  legion_projection_id_t id,
  unsigned depth,
  legion_projection_functor_logical_region_t region_functor,
  legion_projection_functor_logical_partition_t partition_functor)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);

  FunctorWrapper *functor =
    new FunctorWrapper(runtime, depth, region_functor, partition_functor);
  runtime->register_projection_functor(id, functor);
}

legion_task_id_t
legion_runtime_generate_library_task_ids(
    legion_runtime_t runtime_,
    const char *library_name,
    size_t count)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);

  return runtime->generate_library_task_ids(library_name, count);
}

legion_task_id_t
legion_runtime_register_task_variant_fnptr(
  legion_runtime_t runtime_,
  legion_task_id_t id /* = AUTO_GENERATE_ID */,
  const char *task_name /* = NULL*/,
  bool global,
  legion_execution_constraint_set_t execution_constraints_,
  legion_task_layout_constraint_set_t layout_constraints_,
  legion_task_config_options_t options,
  legion_task_pointer_wrapped_t wrapped_task_pointer,
  const void *userdata,
  size_t userlen)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  ExecutionConstraintSet *execution_constraints =
    CObjectWrapper::unwrap(execution_constraints_);
  TaskLayoutConstraintSet *layout_constraints =
    CObjectWrapper::unwrap(layout_constraints_);

  if (id == AUTO_GENERATE_ID)
    id = runtime->generate_dynamic_task_id();

  TaskVariantRegistrar registrar(id, task_name, global);
  registrar.set_leaf(options.leaf);
  registrar.set_inner(options.inner);
  registrar.set_idempotent(options.idempotent);
  if (layout_constraints)
    registrar.layout_constraints = *layout_constraints;
  if (execution_constraints)
    registrar.execution_constraints = *execution_constraints;

  CodeDescriptor code_desc(Realm::Type::from_cpp_type<Processor::TaskFuncPtr>());
  code_desc.add_implementation(new Realm::FunctionPointerImplementation((void(*)())wrapped_task_pointer));

  /*VariantID vid =*/ runtime->register_task_variant(
    registrar, code_desc, userdata, userlen);

  if (task_name)
    runtime->attach_name(id, task_name);
  return id;
}

legion_task_id_t
legion_runtime_preregister_task_variant_fnptr(
  legion_task_id_t id /* = AUTO_GENERATE_ID */,
  const char *task_name /* = NULL*/,
  legion_execution_constraint_set_t execution_constraints_,
  legion_task_layout_constraint_set_t layout_constraints_,
  legion_task_config_options_t options,
  legion_task_pointer_wrapped_t wrapped_task_pointer,
  const void *userdata,
  size_t userlen)
{
  ExecutionConstraintSet *execution_constraints =
    CObjectWrapper::unwrap(execution_constraints_);
  TaskLayoutConstraintSet *layout_constraints =
    CObjectWrapper::unwrap(layout_constraints_);

  if (id == AUTO_GENERATE_ID)
    id = Runtime::generate_static_task_id();

  TaskVariantRegistrar registrar(id, task_name);
  registrar.set_leaf(options.leaf);
  registrar.set_inner(options.inner);
  registrar.set_idempotent(options.idempotent);
  if (layout_constraints)
    registrar.layout_constraints = *layout_constraints;
  if (execution_constraints)
    registrar.execution_constraints = *execution_constraints;

  CodeDescriptor code_desc(Realm::Type::from_cpp_type<Processor::TaskFuncPtr>());
  code_desc.add_implementation(new Realm::FunctionPointerImplementation((void(*)())wrapped_task_pointer));

  /*VariantID vid =*/ Runtime::preregister_task_variant(
    registrar, code_desc, userdata, userlen, task_name);

  return id;
}

#ifdef REALM_USE_LLVM
legion_task_id_t
legion_runtime_register_task_variant_llvmir(
  legion_runtime_t runtime_,
  legion_task_id_t id /* = AUTO_GENERATE_ID */,
  const char *task_name /* = NULL*/,
  bool global,
  legion_execution_constraint_set_t execution_constraints_,
  legion_task_layout_constraint_set_t layout_constraints_,
  legion_task_config_options_t options,
  const char *llvmir,
  const char *entry_symbol,
  const void *userdata,
  size_t userlen)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  ExecutionConstraintSet *execution_constraints =
    CObjectWrapper::unwrap(execution_constraints_);
  TaskLayoutConstraintSet *layout_constraints =
    CObjectWrapper::unwrap(layout_constraints_);

  if (id == AUTO_GENERATE_ID)
    id = runtime->generate_dynamic_task_id();

  TaskVariantRegistrar registrar(id, task_name, global);
  registrar.set_leaf(options.leaf);
  registrar.set_inner(options.inner);
  registrar.set_idempotent(options.idempotent);
  if (layout_constraints)
    registrar.layout_constraints = *layout_constraints;
  if (execution_constraints)
    registrar.execution_constraints = *execution_constraints;

  CodeDescriptor code_desc(Realm::Type::from_cpp_type<Processor::TaskFuncPtr>());
  code_desc.add_implementation(new Realm::LLVMIRImplementation(llvmir, strlen(llvmir), entry_symbol));

  /*VariantID vid =*/ runtime->register_task_variant(
    registrar, code_desc, userdata, userlen);

  if (task_name)
    runtime->attach_name(id, task_name);
  return id;
}

legion_task_id_t
legion_runtime_preregister_task_variant_llvmir(
  legion_task_id_t id /* = AUTO_GENERATE_ID */,
  const char *task_name /* = NULL*/,
  legion_execution_constraint_set_t execution_constraints_,
  legion_task_layout_constraint_set_t layout_constraints_,
  legion_task_config_options_t options,
  const char *llvmir,
  const char *entry_symbol,
  const void *userdata,
  size_t userlen)
{
  ExecutionConstraintSet *execution_constraints =
    CObjectWrapper::unwrap(execution_constraints_);
  TaskLayoutConstraintSet *layout_constraints =
    CObjectWrapper::unwrap(layout_constraints_);

  if (id == AUTO_GENERATE_ID)
    id = Runtime::generate_static_task_id();

  TaskVariantRegistrar registrar(id, task_name);
  registrar.set_leaf(options.leaf);
  registrar.set_inner(options.inner);
  registrar.set_idempotent(options.idempotent);
  if (layout_constraints)
    registrar.layout_constraints = *layout_constraints;
  if (execution_constraints)
    registrar.execution_constraints = *execution_constraints;

  CodeDescriptor code_desc(Realm::Type::from_cpp_type<Processor::TaskFuncPtr>());
  code_desc.add_implementation(new Realm::LLVMIRImplementation(llvmir, strlen(llvmir), entry_symbol));

  /*VariantID vid =*/ Runtime::preregister_task_variant(
    registrar, code_desc, userdata, userlen, task_name);
  return id;
}
#endif

#ifdef REALM_USE_PYTHON
legion_task_id_t
legion_runtime_register_task_variant_python_source(
  legion_runtime_t runtime_,
  legion_task_id_t id /* = AUTO_GENERATE_ID */,
  const char *task_name /* = NULL*/,
  bool global,
  legion_execution_constraint_set_t execution_constraints_,
  legion_task_layout_constraint_set_t layout_constraints_,
  legion_task_config_options_t options,
  const char *module_name,
  const char *function_name,
  const void *userdata,
  size_t userlen)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  ExecutionConstraintSet *execution_constraints =
    CObjectWrapper::unwrap(execution_constraints_);
  TaskLayoutConstraintSet *layout_constraints =
    CObjectWrapper::unwrap(layout_constraints_);

  if (id == AUTO_GENERATE_ID)
    id = runtime->generate_dynamic_task_id();

  TaskVariantRegistrar registrar(id, task_name, global);
  registrar.set_leaf(options.leaf);
  registrar.set_inner(options.inner);
  registrar.set_idempotent(options.idempotent);
  if (layout_constraints)
    registrar.layout_constraints = *layout_constraints;
  if (execution_constraints)
    registrar.execution_constraints = *execution_constraints;

  CodeDescriptor code_desc(Realm::Type::from_cpp_type<Processor::TaskFuncPtr>());
  code_desc.add_implementation(new Realm::PythonSourceImplementation(module_name, function_name));

  /*VariantID vid =*/ runtime->register_task_variant(
    registrar, code_desc, userdata, userlen);

  if (task_name)
    runtime->attach_name(id, task_name);
  return id;
}
#endif

void
legion_task_preamble(
  const void *data,
  size_t datalen,
  legion_proc_id_t proc_id,
  legion_task_t *taskptr,
  const legion_physical_region_t **regionptr,
  unsigned * num_regions_ptr,
  legion_context_t * ctxptr,
  legion_runtime_t * runtimeptr)
{
  Processor p;
  p.id = proc_id;
  const Task *task;
  const std::vector<PhysicalRegion> *regions;
  Context ctx;
  Runtime *runtime;

  Runtime::legion_task_preamble(data,
				datalen,
				p,
				task,
				regions,
				ctx,
				runtime);

  CContext *cctx = new CContext(ctx, *regions);
  *taskptr = CObjectWrapper::wrap_const(task);
  *regionptr = cctx->regions();
  *num_regions_ptr = cctx->num_regions();
  *ctxptr = CObjectWrapper::wrap(cctx);
  *runtimeptr = CObjectWrapper::wrap(runtime);
}

void
legion_task_postamble(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  const void *retval,
  size_t retsize)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  CContext *cctx = CObjectWrapper::unwrap(ctx_);
  Context ctx = cctx->context();
  delete cctx;

  Runtime::legion_task_postamble(runtime,
				 ctx,
				 retval,
				 retsize);
}

// -----------------------------------------------------------------------
// Timing Operations
// -----------------------------------------------------------------------

unsigned long long
legion_get_current_time_in_micros(void)
{
  return Realm::Clock::current_time_in_microseconds();
}

unsigned long long
legion_get_current_time_in_nanos(void)
{
  return Realm::Clock::current_time_in_nanoseconds();
}

legion_future_t
legion_issue_timing_op_seconds(legion_runtime_t runtime_,
                               legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  TimingLauncher launcher(MEASURE_SECONDS);
  Future f = runtime->issue_timing_measurement(ctx, launcher);  
  return CObjectWrapper::wrap(new Future(f));
}

legion_future_t
legion_issue_timing_op_microseconds(legion_runtime_t runtime_,
                                    legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  TimingLauncher launcher(MEASURE_MICRO_SECONDS);
  Future f = runtime->issue_timing_measurement(ctx, launcher);  
  return CObjectWrapper::wrap(new Future(f));
}

legion_future_t
legion_issue_timing_op_nanoseconds(legion_runtime_t runtime_,
                                   legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  TimingLauncher launcher(MEASURE_NANO_SECONDS);
  Future f = runtime->issue_timing_measurement(ctx, launcher);  
  return CObjectWrapper::wrap(new Future(f));
}

// -----------------------------------------------------------------------
// Machine Operations
// -----------------------------------------------------------------------

legion_machine_t
legion_machine_create()
{
  Machine *result = new Machine(Machine::get_machine());

  return CObjectWrapper::wrap(result);
}

void
legion_machine_destroy(legion_machine_t handle_)
{
  Machine *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

void
legion_machine_get_all_processors(
  legion_machine_t machine_,
  legion_processor_t *processors_,
  size_t processors_size)
{
  Machine *machine = CObjectWrapper::unwrap(machine_);

  std::set<Processor> pset;
  machine->get_all_processors(pset);
  std::set<Processor>::iterator itr = pset.begin();

  size_t num_to_copy = std::min(pset.size(), processors_size);

  for (unsigned i = 0; i < num_to_copy; ++i) {
    processors_[i] = CObjectWrapper::wrap(*itr++);
  }
}

size_t
legion_machine_get_all_processors_size(legion_machine_t machine_)
{
  Machine *machine = CObjectWrapper::unwrap(machine_);

  std::set<Processor> pset;
  machine->get_all_processors(pset);
  return pset.size();
}

void
legion_machine_get_all_memories(
  legion_machine_t machine_,
  legion_memory_t *memories_,
  size_t memories_size)
{
  Machine *machine = CObjectWrapper::unwrap(machine_);

  std::set<Memory> mset;
  machine->get_all_memories(mset);
  std::set<Memory>::iterator itr = mset.begin();

  size_t num_to_copy = std::min(mset.size(), memories_size);

  for (size_t i = 0; i < num_to_copy; ++i) {
    memories_[i] = CObjectWrapper::wrap(*itr++);
  }
}

size_t
legion_machine_get_all_memories_size(legion_machine_t machine_)
{
  Machine *machine = CObjectWrapper::unwrap(machine_);

  std::set<Memory> mset;
  machine->get_all_memories(mset);
  return mset.size();
}

// -----------------------------------------------------------------------
// Processor Operations
// -----------------------------------------------------------------------

legion_processor_kind_t
legion_processor_kind(legion_processor_t proc_)
{
  Processor proc = CObjectWrapper::unwrap(proc_);

  return CObjectWrapper::wrap(proc.kind());
}

legion_address_space_t
legion_processor_address_space(legion_processor_t proc_)
{
  Processor proc = CObjectWrapper::unwrap(proc_);

  return proc.address_space();
}

// -----------------------------------------------------------------------
// Memory Operations
// -----------------------------------------------------------------------

legion_memory_kind_t
legion_memory_kind(legion_memory_t mem_)
{
  Memory mem = CObjectWrapper::unwrap(mem_);

  return CObjectWrapper::wrap(mem.kind());
}

legion_address_space_t
legion_memory_address_space(legion_memory_t mem_)
{
  Memory mem = CObjectWrapper::unwrap(mem_);

  return mem.address_space();
}

// -----------------------------------------------------------------------
// Processor Query Operations
// -----------------------------------------------------------------------

legion_processor_query_t
legion_processor_query_create(legion_machine_t machine_)
{
  Machine *machine = CObjectWrapper::unwrap(machine_);

  Machine::ProcessorQuery *result = new Machine::ProcessorQuery(*machine);
  return CObjectWrapper::wrap(result);
}

legion_processor_query_t
legion_processor_query_create_copy(legion_processor_query_t query_)
{
  Machine::ProcessorQuery *query = CObjectWrapper::unwrap(query_);

  Machine::ProcessorQuery *result = new Machine::ProcessorQuery(*query);
  return CObjectWrapper::wrap(result);
}

void
legion_processor_query_destroy(legion_processor_query_t handle_)
{
  Machine::ProcessorQuery *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

void
legion_processor_query_only_kind(legion_processor_query_t query_,
                                 legion_processor_kind_t kind_)
{
  Machine::ProcessorQuery *query = CObjectWrapper::unwrap(query_);
  Processor::Kind kind = CObjectWrapper::unwrap(kind_);

  query->only_kind(kind);
}

void
legion_processor_query_local_address_space(legion_processor_query_t query_)
{
  Machine::ProcessorQuery *query = CObjectWrapper::unwrap(query_);

  query->local_address_space();
}

void
legion_processor_query_same_address_space_as_processor(legion_processor_query_t query_,
                                                       legion_processor_t proc_)
{
  Machine::ProcessorQuery *query = CObjectWrapper::unwrap(query_);
  Processor proc = CObjectWrapper::unwrap(proc_);

  query->same_address_space_as(proc);
}

void
legion_processor_query_same_address_space_as_memory(legion_processor_query_t query_,
                                                    legion_memory_t mem_)
{
  Machine::ProcessorQuery *query = CObjectWrapper::unwrap(query_);
  Memory mem = CObjectWrapper::unwrap(mem_);

  query->same_address_space_as(mem);
}

void
legion_processor_query_has_affinity_to_memory(legion_processor_query_t query_,
                                              legion_memory_t mem_,
                                              unsigned min_bandwidth /* = 0 */,
                                              unsigned max_latency /* = 0 */)
{
  Machine::ProcessorQuery *query = CObjectWrapper::unwrap(query_);
  Memory mem = CObjectWrapper::unwrap(mem_);

  query->has_affinity_to(mem, min_bandwidth, max_latency);
}

void
legion_processor_query_best_affinity_to_memory(legion_processor_query_t query_,
                                               legion_memory_t mem_,
                                               int bandwidth_weight /* = 0 */,
                                               int latency_weight /* = 0 */)
{
  Machine::ProcessorQuery *query = CObjectWrapper::unwrap(query_);
  Memory mem = CObjectWrapper::unwrap(mem_);

  query->best_affinity_to(mem, bandwidth_weight, latency_weight);
}

size_t
legion_processor_query_count(legion_processor_query_t query_)
{
  Machine::ProcessorQuery *query = CObjectWrapper::unwrap(query_);

  return query->count();
}

legion_processor_t
legion_processor_query_first(legion_processor_query_t query_)
{
  Machine::ProcessorQuery *query = CObjectWrapper::unwrap(query_);

  Processor result = query->first();
  return CObjectWrapper::wrap(result);
}

legion_processor_t
legion_processor_query_next(legion_processor_query_t query_,
                           legion_processor_t after_)
{
  Machine::ProcessorQuery *query = CObjectWrapper::unwrap(query_);
  Processor after = CObjectWrapper::unwrap(after_);

  Processor result = query->next(after);
  return CObjectWrapper::wrap(result);
}

legion_processor_t
legion_processor_query_random(legion_processor_query_t query_)
{
  Machine::ProcessorQuery *query = CObjectWrapper::unwrap(query_);

  Processor result = query->random();
  return CObjectWrapper::wrap(result);
}

// -----------------------------------------------------------------------
// Memory Query Operations
// -----------------------------------------------------------------------

legion_memory_query_t
legion_memory_query_create(legion_machine_t machine_)
{
  Machine *machine = CObjectWrapper::unwrap(machine_);

  Machine::MemoryQuery *result = new Machine::MemoryQuery(*machine);
  return CObjectWrapper::wrap(result);
}

legion_memory_query_t
legion_memory_query_create_copy(legion_memory_query_t query_)
{
  Machine::MemoryQuery *query = CObjectWrapper::unwrap(query_);

  Machine::MemoryQuery *result = new Machine::MemoryQuery(*query);
  return CObjectWrapper::wrap(result);
}

void
legion_memory_query_destroy(legion_memory_query_t handle_)
{
  Machine::MemoryQuery *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

void
legion_memory_query_only_kind(legion_memory_query_t query_,
                              legion_memory_kind_t kind_)
{
  Machine::MemoryQuery *query = CObjectWrapper::unwrap(query_);
  Memory::Kind kind = CObjectWrapper::unwrap(kind_);

  query->only_kind(kind);
}

void
legion_memory_query_local_address_space(legion_memory_query_t query_)
{
  Machine::MemoryQuery *query = CObjectWrapper::unwrap(query_);

  query->local_address_space();
}

void
legion_memory_query_same_address_space_as_processor(legion_memory_query_t query_,
                                                    legion_processor_t proc_)
{
  Machine::MemoryQuery *query = CObjectWrapper::unwrap(query_);
  Processor proc = CObjectWrapper::unwrap(proc_);

  query->same_address_space_as(proc);
}

void
legion_memory_query_same_address_space_as_memory(legion_memory_query_t query_,
                                                 legion_memory_t mem_)
{
  Machine::MemoryQuery *query = CObjectWrapper::unwrap(query_);
  Memory mem = CObjectWrapper::unwrap(mem_);

  query->same_address_space_as(mem);
}

void
legion_memory_query_has_affinity_to_processor(legion_memory_query_t query_,
                                              legion_processor_t proc_,
                                              unsigned min_bandwidth /* = 0 */,
                                              unsigned max_latency /* = 0 */)
{
  Machine::MemoryQuery *query = CObjectWrapper::unwrap(query_);
  Processor proc = CObjectWrapper::unwrap(proc_);

  query->has_affinity_to(proc, min_bandwidth, max_latency);
}

void
legion_memory_query_has_affinity_to_memory(legion_memory_query_t query_,
                                           legion_memory_t mem_,
                                           unsigned min_bandwidth /* = 0 */,
                                           unsigned max_latency /* = 0 */)
{
  Machine::MemoryQuery *query = CObjectWrapper::unwrap(query_);
  Memory mem = CObjectWrapper::unwrap(mem_);

  query->has_affinity_to(mem, min_bandwidth, max_latency);
}

void
legion_memory_query_best_affinity_to_processor(legion_memory_query_t query_,
                                               legion_processor_t proc_,
                                               int bandwidth_weight /* = 0 */,
                                               int latency_weight /* = 0 */)
{
  Machine::MemoryQuery *query = CObjectWrapper::unwrap(query_);
  Processor proc = CObjectWrapper::unwrap(proc_);

  query->best_affinity_to(proc, bandwidth_weight, latency_weight);
}

void
legion_memory_query_best_affinity_to_memory(legion_memory_query_t query_,
                                            legion_memory_t mem_,
                                            int bandwidth_weight /* = 0 */,
                                            int latency_weight /* = 0 */)
{
  Machine::MemoryQuery *query = CObjectWrapper::unwrap(query_);
  Memory mem = CObjectWrapper::unwrap(mem_);

  query->best_affinity_to(mem, bandwidth_weight, latency_weight);
}

size_t
legion_memory_query_count(legion_memory_query_t query_)
{
  Machine::MemoryQuery *query = CObjectWrapper::unwrap(query_);

  return query->count();
}

legion_memory_t
legion_memory_query_first(legion_memory_query_t query_)
{
  Machine::MemoryQuery *query = CObjectWrapper::unwrap(query_);

  Memory result = query->first();
  return CObjectWrapper::wrap(result);
}

legion_memory_t
legion_memory_query_next(legion_memory_query_t query_,
                         legion_memory_t after_)
{
  Machine::MemoryQuery *query = CObjectWrapper::unwrap(query_);
  Memory after = CObjectWrapper::unwrap(after_);

  Memory result = query->next(after);
  return CObjectWrapper::wrap(result);
}

legion_memory_t
legion_memory_query_random(legion_memory_query_t query_)
{
  Machine::MemoryQuery *query = CObjectWrapper::unwrap(query_);

  Memory result = query->random();
  return CObjectWrapper::wrap(result);
}

// -----------------------------------------------------------------------
// Physical Instance Operations
// -----------------------------------------------------------------------

void
legion_physical_instance_destroy(legion_physical_instance_t instance_)
{
  delete CObjectWrapper::unwrap(instance_);
}

// -----------------------------------------------------------------------
// Slice Task Output
// -----------------------------------------------------------------------

void
legion_slice_task_output_slices_add(
    legion_slice_task_output_t output_,
    legion_task_slice_t slice_)
{
  Mapper::SliceTaskOutput* output = CObjectWrapper::unwrap(output_);
  Mapper::TaskSlice slice = CObjectWrapper::unwrap(slice_);
  output->slices.push_back(slice);
}

void
legion_slice_task_output_verify_correctness_set(
    legion_slice_task_output_t output_,
    bool verify_correctness)
{
  CObjectWrapper::unwrap(output_)->verify_correctness = verify_correctness;
}

// -----------------------------------------------------------------------
// Map Task Input/Output
// -----------------------------------------------------------------------

void
legion_map_task_output_chosen_instances_clear_all(
    legion_map_task_output_t output_)
{
  Mapper::MapTaskOutput* output = CObjectWrapper::unwrap(output_);
  output->chosen_instances.clear();
}

void
legion_map_task_output_chosen_instances_clear_each(
    legion_map_task_output_t output_,
    size_t idx_)
{
  Mapper::MapTaskOutput* output = CObjectWrapper::unwrap(output_);
  output->chosen_instances[idx_].clear();
}

void
legion_map_task_output_chosen_instances_add(
    legion_map_task_output_t output_,
    legion_physical_instance_t *instances_,
    size_t instances_size_)
{
  Mapper::MapTaskOutput* output = CObjectWrapper::unwrap(output_);
  output->chosen_instances.push_back(std::vector<PhysicalInstance>());
  std::vector<PhysicalInstance>& chosen_instances =
    output->chosen_instances.back();
  for (size_t i = 0; i < instances_size_; ++i)
    chosen_instances.push_back(*CObjectWrapper::unwrap(instances_[i]));
}

void
legion_map_task_output_chosen_instances_set(
    legion_map_task_output_t output_,
    size_t idx_,
    legion_physical_instance_t *instances_,
    size_t instances_size_)
{
  Mapper::MapTaskOutput* output = CObjectWrapper::unwrap(output_);
  std::vector<PhysicalInstance>& chosen_instances =
    output->chosen_instances[idx_];
  chosen_instances.clear();
  for (size_t i = 0; i < instances_size_; ++i)
    chosen_instances.push_back(*CObjectWrapper::unwrap(instances_[i]));
}

void
legion_map_task_output_target_procs_clear(
    legion_map_task_output_t output_)
{
  Mapper::MapTaskOutput* output = CObjectWrapper::unwrap(output_);
  output->target_procs.clear();
}

void
legion_map_task_output_target_procs_add(
    legion_map_task_output_t output_,
    legion_processor_t proc_)
{
  Mapper::MapTaskOutput* output = CObjectWrapper::unwrap(output_);
  output->target_procs.push_back(CObjectWrapper::unwrap(proc_));
}

legion_processor_t
legion_map_task_output_target_procs_get(
    legion_map_task_output_t output_,
    size_t idx_)
{
  return CObjectWrapper::wrap(
      CObjectWrapper::unwrap(output_)->target_procs[idx_]);
}

void
legion_map_task_output_task_priority_set(
    legion_map_task_output_t output_,
    legion_task_priority_t priority_)
{
  Mapper::MapTaskOutput* output = CObjectWrapper::unwrap(output_);
  output->task_priority = priority_;
}

// -----------------------------------------------------------------------
// MapperRuntime Operations
// -----------------------------------------------------------------------

bool
legion_mapper_runtime_create_physical_instance_layout_constraint(
    legion_mapper_runtime_t runtime_,
    legion_mapper_context_t ctx_,
    legion_memory_t target_memory_,
    legion_layout_constraint_set_t constraints_,
    const legion_logical_region_t *regions_,
    size_t regions_size_,
    legion_physical_instance_t *result_,
    bool acquire_,
    legion_garbage_collection_priority_t priority_)
{
  MapperRuntime* runtime = CObjectWrapper::unwrap(runtime_);
  MapperContext ctx = CObjectWrapper::unwrap(ctx_);
  Memory memory = CObjectWrapper::unwrap(target_memory_);
  LayoutConstraintSet* constraints = CObjectWrapper::unwrap(constraints_);
  std::vector<LogicalRegion> regions;
  regions.reserve(regions_size_);
  for (size_t idx = 0; idx < regions_size_; ++idx)
    regions.push_back(CObjectWrapper::unwrap(regions_[idx]));

  PhysicalInstance* result = new PhysicalInstance;
  bool ret =
    runtime->create_physical_instance(
        ctx, memory, *constraints, regions, *result, acquire_, priority_);
  *result_ = CObjectWrapper::wrap(result);
  return ret;
}

bool
legion_mapper_runtime_create_physical_instance_layout_constraint_id(
    legion_mapper_runtime_t runtime_,
    legion_mapper_context_t ctx_,
    legion_memory_t target_memory_,
    legion_layout_constraint_id_t layout_id_,
    const legion_logical_region_t *regions_,
    size_t regions_size_,
    legion_physical_instance_t *result_,
    bool acquire_,
    legion_garbage_collection_priority_t priority_)
{
  MapperRuntime* runtime = CObjectWrapper::unwrap(runtime_);
  MapperContext ctx = CObjectWrapper::unwrap(ctx_);
  Memory memory = CObjectWrapper::unwrap(target_memory_);
  std::vector<LogicalRegion> regions;
  regions.reserve(regions_size_);
  for (size_t idx = 0; idx < regions_size_; ++idx)
    regions.push_back(CObjectWrapper::unwrap(regions_[idx]));

  PhysicalInstance* result = new PhysicalInstance;
  bool ret =
    runtime->create_physical_instance(
        ctx, memory, layout_id_, regions, *result, acquire_, priority_);
  *result_ = CObjectWrapper::wrap(result);
  return ret;
}

bool
legion_mapper_runtime_find_or_create_physical_instance_layout_constraint(
    legion_mapper_runtime_t runtime_,
    legion_mapper_context_t ctx_,
    legion_memory_t target_memory_,
    legion_layout_constraint_set_t constraints_,
    const legion_logical_region_t *regions_,
    size_t regions_size_,
    legion_physical_instance_t *result_,
    bool *created_,
    bool acquire_,
    legion_garbage_collection_priority_t priority_,
    bool tight_region_bounds_)
{
  MapperRuntime* runtime = CObjectWrapper::unwrap(runtime_);
  MapperContext ctx = CObjectWrapper::unwrap(ctx_);
  Memory memory = CObjectWrapper::unwrap(target_memory_);
  LayoutConstraintSet* constraints = CObjectWrapper::unwrap(constraints_);
  std::vector<LogicalRegion> regions;
  regions.reserve(regions_size_);
  for (size_t idx = 0; idx < regions_size_; ++idx)
    regions.push_back(CObjectWrapper::unwrap(regions_[idx]));

  PhysicalInstance* result = new PhysicalInstance;
  bool ret =
    runtime->find_or_create_physical_instance(
        ctx, memory, *constraints, regions, *result, *created_,
        acquire_, priority_, tight_region_bounds_);
  *result_ = CObjectWrapper::wrap(result);
  return ret;
}

bool
legion_mapper_runtime_find_or_create_physical_instance_layout_constraint_id(
    legion_mapper_runtime_t runtime_,
    legion_mapper_context_t ctx_,
    legion_memory_t target_memory_,
    legion_layout_constraint_id_t layout_id_,
    const legion_logical_region_t *regions_,
    size_t regions_size_,
    legion_physical_instance_t *result_,
    bool *created_,
    bool acquire_,
    legion_garbage_collection_priority_t priority_,
    bool tight_region_bounds_)
{
  MapperRuntime* runtime = CObjectWrapper::unwrap(runtime_);
  MapperContext ctx = CObjectWrapper::unwrap(ctx_);
  Memory memory = CObjectWrapper::unwrap(target_memory_);
  std::vector<LogicalRegion> regions;
  regions.reserve(regions_size_);
  for (size_t idx = 0; idx < regions_size_; ++idx)
    regions.push_back(CObjectWrapper::unwrap(regions_[idx]));

  PhysicalInstance* result = new PhysicalInstance;
  bool ret =
    runtime->find_or_create_physical_instance(
        ctx, memory, layout_id_, regions, *result, *created_,
        acquire_, priority_, tight_region_bounds_);
  *result_ = CObjectWrapper::wrap(result);
  return ret;
}

bool
legion_mapper_runtime_find_physical_instance_layout_constraint(
    legion_mapper_runtime_t runtime_,
    legion_mapper_context_t ctx_,
    legion_memory_t target_memory_,
    legion_layout_constraint_set_t constraints_,
    const legion_logical_region_t *regions_,
    size_t regions_size_,
    legion_physical_instance_t *result_,
    bool acquire_,
    bool tight_region_bounds_)
{
  MapperRuntime* runtime = CObjectWrapper::unwrap(runtime_);
  MapperContext ctx = CObjectWrapper::unwrap(ctx_);
  Memory memory = CObjectWrapper::unwrap(target_memory_);
  LayoutConstraintSet* constraints = CObjectWrapper::unwrap(constraints_);
  std::vector<LogicalRegion> regions;
  regions.reserve(regions_size_);
  for (size_t idx = 0; idx < regions_size_; ++idx)
    regions.push_back(CObjectWrapper::unwrap(regions_[idx]));

  PhysicalInstance* result = new PhysicalInstance;
  bool ret =
    runtime->find_physical_instance(
        ctx, memory, *constraints, regions, *result,
        acquire_, tight_region_bounds_);
  *result_ = CObjectWrapper::wrap(result);
  return ret;
}

bool
legion_mapper_runtime_find_physical_instance_layout_constraint_id(
    legion_mapper_runtime_t runtime_,
    legion_mapper_context_t ctx_,
    legion_memory_t target_memory_,
    legion_layout_constraint_id_t layout_id_,
    const legion_logical_region_t *regions_,
    size_t regions_size_,
    legion_physical_instance_t *result_,
    bool acquire_,
    bool tight_region_bounds_)
{
  MapperRuntime* runtime = CObjectWrapper::unwrap(runtime_);
  MapperContext ctx = CObjectWrapper::unwrap(ctx_);
  Memory memory = CObjectWrapper::unwrap(target_memory_);
  std::vector<LogicalRegion> regions;
  regions.reserve(regions_size_);
  for (size_t idx = 0; idx < regions_size_; ++idx)
    regions.push_back(CObjectWrapper::unwrap(regions_[idx]));

  PhysicalInstance* result = new PhysicalInstance;
  bool ret =
    runtime->find_physical_instance(
        ctx, memory, layout_id_, regions, *result,
        acquire_, tight_region_bounds_);
  *result_ = CObjectWrapper::wrap(result);
  return ret;
}

bool
legion_mapper_runtime_acquire_instance(
    legion_mapper_runtime_t runtime_,
    legion_mapper_context_t ctx_,
    legion_physical_instance_t instance_)
{
  MapperRuntime* runtime = CObjectWrapper::unwrap(runtime_);
  MapperContext ctx = CObjectWrapper::unwrap(ctx_);
  PhysicalInstance* instance = CObjectWrapper::unwrap(instance_);
  return runtime->acquire_instance(ctx, *instance);
}

bool
legion_mapper_runtime_acquire_instances(
    legion_mapper_runtime_t runtime_,
    legion_mapper_context_t ctx_,
    legion_physical_instance_t *instances_,
    size_t instances_size)
{
  MapperRuntime* runtime = CObjectWrapper::unwrap(runtime_);
  MapperContext ctx = CObjectWrapper::unwrap(ctx_);
  std::vector<PhysicalInstance> instances;
  for (size_t idx = 0; idx < instances_size; ++idx)
    instances.push_back(*CObjectWrapper::unwrap(instances_[idx]));
  return runtime->acquire_instances(ctx, instances);
}
