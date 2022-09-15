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
#ifdef __PGIC__
#pragma diag_suppress 816
#pragma diag_suppress 1445
#endif

using namespace Legion;
using namespace Legion::Mapping;
using namespace Legion::Mapping::Utilities;
#define TYPEDEF_POINT(DIM) \
  typedef Point<DIM,coord_t> Point##DIM##D;
LEGION_FOREACH_N(TYPEDEF_POINT)
#undef TYPEDEF_POINT
#define TYPEDEF_RECT(DIM) \
  typedef Rect<DIM,coord_t> Rect##DIM##D;
LEGION_FOREACH_N(TYPEDEF_RECT)
#undef TYPEDEF_RECT
#define TYPEDEF_TRANSFORM(D1,D2) \
  typedef Transform<D1,D2,coord_t> Transform##D1##x##D2;
LEGION_FOREACH_NN(TYPEDEF_TRANSFORM)
#undef TYPEDEF_TRANSFORM
#define TYPEDEF_AFFINE(D1,D2) \
  typedef AffineTransform<D1,D2,coord_t> AffineTransform##D1##x##D2;
LEGION_FOREACH_NN(TYPEDEF_AFFINE)
#undef TYPEDEF_AFFINE
#define TYPEDEF_BUFFER(DIM) \
  typedef DeferredBuffer<char,DIM> DeferredBufferChar##DIM##D;
LEGION_FOREACH_N(TYPEDEF_BUFFER)
#undef TYPEDEF_BUFFER

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
legion_domain_empty(unsigned dim)
{
  legion_domain_t domain;
  domain.dim = dim;
  domain.is_id = 0;
  for (unsigned i = 0; i < dim; i++)
    domain.rect_data[i] = 1;
  for (unsigned i = 0; i < dim; i++)
    domain.rect_data[dim+i] = 0;
  for (unsigned i = 2*dim; i < (2*LEGION_MAX_DIM); i++)
    domain.rect_data[i] = 0;
  return domain;
}

#define FROM_RECT(DIM) \
legion_domain_t \
legion_domain_from_rect_##DIM##d(legion_rect_##DIM##d_t r_) \
{ \
  Rect##DIM##D r = CObjectWrapper::unwrap(r_); \
 \
  return CObjectWrapper::wrap(Domain(r)); \
}
LEGION_FOREACH_N(FROM_RECT)
#undef FROM_RECT

legion_domain_t
legion_domain_from_index_space(legion_runtime_t runtime_,
                               legion_index_space_t is_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexSpace is = CObjectWrapper::unwrap(is_);

  return CObjectWrapper::wrap(runtime->get_index_space_domain(is));
}

#define GET_RECT(DIM) \
legion_rect_##DIM##d_t \
legion_domain_get_rect_##DIM##d(legion_domain_t d_) \
{ \
  Domain d = CObjectWrapper::unwrap(d_); \
  Rect##DIM##D r = d; \
\
  return CObjectWrapper::wrap(r); \
}
LEGION_FOREACH_N(GET_RECT)
#undef GET_RECT

bool
legion_domain_is_dense(legion_domain_t d_)
{
  Domain d = CObjectWrapper::unwrap(d_);

  return d.dense();
}

#define GET_BOUNDS(DIM) \
legion_rect_##DIM##d_t \
legion_domain_get_bounds_##DIM##d(legion_domain_t d_) \
{ \
  Domain d = CObjectWrapper::unwrap(d_); \
  DomainT<DIM,coord_t> space = d; \
 \
  return CObjectWrapper::wrap(space.bounds); \
}
LEGION_FOREACH_N(GET_BOUNDS)
#undef GET_BOUNDS

bool
legion_domain_contains(legion_domain_t d_, legion_domain_point_t p_)
{
  Domain d = CObjectWrapper::unwrap(d_);
  DomainPoint p = CObjectWrapper::unwrap(p_);
  return d.contains(p);
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
legion_domain_transform_identity(unsigned m, unsigned n)
{
  legion_domain_transform_t result;
  result.m = m;
  result.n = n;
  for (unsigned i = 0; i < m; i++)
    for (unsigned j = 0; j < n; j++)
      if (i == j)
        result.matrix[i*n+j] = 1;
      else
        result.matrix[i*n+j] = 0;
  for (unsigned i = m*n; i < (LEGION_MAX_DIM * LEGION_MAX_DIM); i++)
    result.matrix[i] = 0;
  return result;
}

#define FROM_TRANSFORM(D1,D2) \
legion_domain_transform_t \
legion_domain_transform_from_##D1##x##D2(legion_transform_##D1##x##D2##_t t_) \
{ \
  Transform##D1##x##D2 t = CObjectWrapper::unwrap(t_); \
 \
  return CObjectWrapper::wrap(DomainTransform(t)); \
}
LEGION_FOREACH_NN(FROM_TRANSFORM)
#undef FROM_TRANSFORM

legion_domain_affine_transform_t
legion_domain_affine_transform_identity(unsigned m, unsigned n)
{
  legion_domain_affine_transform_t result;
  result.transform.m = m;
  result.transform.n = n;
  for (unsigned i = 0; i < m; i++)
    for (unsigned j = 0; j < n; j++)
      if (i == j)
        result.transform.matrix[i*n+j] = 1;
      else
        result.transform.matrix[i*n+j] = 0;
  for (unsigned i = 0; i < (LEGION_MAX_DIM * LEGION_MAX_DIM); i++)
    result.transform.matrix[i] = 0;
  result.offset.dim = m;
  for (unsigned i = 0; i < LEGION_MAX_DIM; i++)
    result.offset.point_data[i] = 0;
  return result;
}

#define FROM_AFFINE(D1,D2) \
legion_domain_affine_transform_t \
legion_domain_affine_transform_from_##D1##x##D2(legion_affine_transform_##D1##x##D2##_t t_) \
{ \
  AffineTransform##D1##x##D2 t = CObjectWrapper::unwrap(t_); \
 \
  return CObjectWrapper::wrap(DomainAffineTransform(t)); \
}
LEGION_FOREACH_NN(FROM_AFFINE)
#undef FROM_AFFINE

// -----------------------------------------------------------------------
// Domain Point Operations
// -----------------------------------------------------------------------

#define FROM_POINT(DIM) \
legion_domain_point_t \
legion_domain_point_from_point_##DIM##d(legion_point_##DIM##d_t p_) \
{ \
  Point##DIM##D p = CObjectWrapper::unwrap(p_); \
 \
  return CObjectWrapper::wrap(DomainPoint(p)); \
}
LEGION_FOREACH_N(FROM_POINT)
#undef FROM_POINT

#define GET_POINT(DIM) \
legion_point_##DIM##d_t \
legion_domain_point_get_point_##DIM##d(legion_domain_point_t p_) \
{ \
  DomainPoint d = CObjectWrapper::unwrap(p_); \
  Point##DIM##D p = d; \
 \
  return CObjectWrapper::wrap(p); \
}
LEGION_FOREACH_N(GET_POINT)
#undef GET_POINT

legion_domain_point_t
legion_domain_point_origin(unsigned dim)
{
  legion_domain_point_t result;
  result.dim = dim;
  for (unsigned i = 0; i < LEGION_MAX_DIM; i++)
    result.point_data[i] = 0;
  return result;
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

// -----------------------------------------------------------------------
// Rect in Domain Iterator
// -----------------------------------------------------------------------

#define DEFINE_RECT_IN_DOMAIN_ITERATOR(N)                                      \
                                                                               \
legion_rect_in_domain_iterator_##N##d_t                                        \
legion_rect_in_domain_iterator_create_##N##d(legion_domain_t handle_)          \
{                                                                              \
  Domain domain = CObjectWrapper::unwrap(handle_);                             \
  assert(domain.dim == N);                                                     \
  RectInDomainIterator<N,coord_t> *itr =                                       \
    new RectInDomainIterator<N,coord_t>(domain);                               \
  return CObjectWrapper::wrap(itr);                                            \
}                                                                              \
                                                                               \
void                                                                           \
legion_rect_in_domain_iterator_destroy_##N##d(                                 \
                              legion_rect_in_domain_iterator_##N##d_t handle_) \
{                                                                              \
  RectInDomainIterator<N,coord_t> *itr = CObjectWrapper::unwrap(handle_);      \
  delete itr;                                                                  \
}                                                                              \
                                                                               \
bool                                                                           \
legion_rect_in_domain_iterator_valid_##N##d(                                   \
                              legion_rect_in_domain_iterator_##N##d_t handle_) \
{                                                                              \
  RectInDomainIterator<N,coord_t> *itr = CObjectWrapper::unwrap(handle_);      \
  return itr->valid();                                                         \
}                                                                              \
                                                                               \
bool                                                                           \
legion_rect_in_domain_iterator_step_##N##d(                                    \
                              legion_rect_in_domain_iterator_##N##d_t handle_) \
{                                                                              \
  RectInDomainIterator<N,coord_t> *itr = CObjectWrapper::unwrap(handle_);      \
  return itr->step();                                                          \
}                                                                              \
                                                                               \
legion_rect_##N##d_t                                                           \
legion_rect_in_domain_iterator_get_rect_##N##d(                                \
                              legion_rect_in_domain_iterator_##N##d_t handle_) \
{                                                                              \
  RectInDomainIterator<N,coord_t> *itr = CObjectWrapper::unwrap(handle_);      \
  return CObjectWrapper::wrap(**itr);                                          \
}

LEGION_FOREACH_N(DEFINE_RECT_IN_DOMAIN_ITERATOR)
#undef DEFINE_RECT_IN_DOMAIN_ITERATOR

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
legion_index_space_create_future(legion_runtime_t runtime_,
                                 legion_context_t ctx_,
                                 size_t dimensions,
                                 legion_future_t future_,
                                 legion_type_tag_t type_tag)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  Future *future = CObjectWrapper::unwrap(future_);

  IndexSpace is = 
    runtime->create_index_space(ctx, dimensions, *future, type_tag);
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

legion_index_space_t
legion_index_space_subtraction(legion_runtime_t runtime_,
                               legion_context_t ctx_,
                               legion_index_space_t left_,
                               legion_index_space_t right_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace left = CObjectWrapper::unwrap(left_);
  IndexSpace right = CObjectWrapper::unwrap(right_);

  return CObjectWrapper::wrap(
      runtime->subtract_index_spaces(ctx, left, right));
}

void
legion_index_space_create_shared_ownership(legion_runtime_t runtime_,
                                           legion_context_t ctx_,
                                           legion_index_space_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->create_shared_ownership(ctx, handle);
}

void
legion_index_space_destroy(legion_runtime_t runtime_,
                           legion_context_t ctx_,
                           legion_index_space_t handle_)
{
  legion_index_space_destroy_unordered(runtime_, ctx_, handle_, false);
}

void
legion_index_space_destroy_unordered(legion_runtime_t runtime_,
                                     legion_context_t ctx_,
                                     legion_index_space_t handle_,
                                     bool unordered)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->destroy_index_space(ctx, handle, unordered);
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

bool
legion_index_space_has_parent_index_partition(legion_runtime_t runtime_,
                                              legion_index_space_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  return runtime->has_parent_index_partition(handle);
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
legion_index_space_attach_semantic_information(legion_runtime_t runtime_,
                                               legion_index_space_t handle_,
                                               legion_semantic_tag_t tag,
                                               const void *buffer,
                                               size_t size,
                                               bool is_mutable /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->attach_semantic_information(handle, tag, buffer, size, is_mutable);
}

bool
legion_index_space_retrieve_semantic_information(
                                         legion_runtime_t runtime_,
                                         legion_index_space_t handle_,
                                         legion_semantic_tag_t tag,
                                         const void **result,
                                         size_t *size,
                                         bool can_fail /* = false */,
                                         bool wait_until_ready /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  return runtime->retrieve_semantic_information(
                       handle, tag, *result, *size, can_fail, wait_until_ready);
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

int
legion_index_space_get_dim(legion_index_space_t handle_)
{
  return CObjectWrapper::unwrap(handle_).get_dim();
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
  legion_color_t part_color /* = AUTO_GENERATE_ID */)
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
  legion_color_t part_color /* = AUTO_GENERATE_ID */)
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
  legion_color_t color /* = AUTO_GENERATE_ID */)
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
  legion_color_t color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  Domain color_space = CObjectWrapper::unwrap(color_space_);
  DomainPointColoring *coloring = CObjectWrapper::unwrap(coloring_);

  // Ensure all colors exist in coloring.
  for(Domain::DomainPointIterator c(color_space); c; c++) {
    if (!(*coloring).count(c.p)) {
      switch (parent.get_dim()) {
        case 1:
          {
            (*coloring)[c.p] = Domain(Rect1D(0, -1)); 
            break;
          }
#if LEGION_MAX_DIM >= 2
      case 2:
          {
            (*coloring)[c.p] =
              Domain(Rect2D(Point2D(0, 0), Point2D(-1, -1)));
            break;
          }
#endif
#if LEGION_MAX_DIM >= 3
      case 3:
          {
            (*coloring)[c.p] =
              Domain(Rect3D(Point3D(0, 0, 0), Point3D(-1, -1, -1)));
            break;
          }
#endif
#if LEGION_MAX_DIM >= 4
      case 4:
          {
            const coord_t lovals[4] = { 0, 0, 0, 0 };
            const coord_t hivals[4] = {-1, -1, -1, -1 };
            (*coloring)[c.p] =
              Domain(Rect4D(Point4D(lovals), Point4D(hivals)));
            break;
          }
#endif
#if LEGION_MAX_DIM >= 5
      case 5:
          {
            const coord_t lovals[5] = { 0, 0, 0, 0, 0 };
            const coord_t hivals[5] = {-1, -1, -1, -1, -1 };
            (*coloring)[c.p] =
              Domain(Rect5D(Point5D(lovals), Point5D(hivals)));
            break;
          }
#endif
#if LEGION_MAX_DIM >= 6
      case 6:
          {
            const coord_t lovals[6] = { 0, 0, 0, 0, 0, 0 };
            const coord_t hivals[6] = {-1, -1, -1, -1, -1, -1 };
            (*coloring)[c.p] =
              Domain(Rect6D(Point6D(lovals), Point6D(hivals)));
            break;
          }
#endif
#if LEGION_MAX_DIM >= 7
      case 7:
          {
            const coord_t lovals[7] = { 0, 0, 0, 0, 0, 0, 0 };
            const coord_t hivals[7] = {-1, -1, -1, -1, -1, -1, -1 };
            (*coloring)[c.p] =
              Domain(Rect7D(Point7D(lovals), Point7D(hivals)));
            break;
          }
#endif
#if LEGION_MAX_DIM >= 8
      case 8:
          {
            const coord_t lovals[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            const coord_t hivals[8] = {-1, -1, -1, -1, -1, -1, -1, -1 };
            (*coloring)[c.p] =
              Domain(Rect8D(Point8D(lovals), Point8D(hivals)));
            break;
          }
#endif
#if LEGION_MAX_DIM >= 9
      case 9:
          {
            const coord_t lovals[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            const coord_t hivals[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1 };
            (*coloring)[c.p] =
              Domain(Rect9D(Point9D(lovals), Point9D(hivals)));
            break;
          }
#endif
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
  legion_color_t color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  Domain color_space = CObjectWrapper::unwrap(color_space_);
  MultiDomainPointColoring *coloring = CObjectWrapper::unwrap(coloring_);

  // Ensure all colors exist in coloring.
  for(Domain::DomainPointIterator c(color_space); c; c++) {
    if ((*coloring)[c.p].empty()) {
      switch (parent.get_dim()) {
        case 1:
          {
            (*coloring)[c.p].insert(Domain(Rect1D(0, -1)));
            break;
          }
#if LEGION_MAX_DIM >= 2
      case 2:
          {
            (*coloring)[c.p].insert(
                Domain(Rect2D(Point2D(0, 0), Point2D(-1, -1))));
            break;
          }
#endif
#if LEGION_MAX_DIM >= 3
      case 3:
          {
            (*coloring)[c.p].insert(
                Domain(Rect3D(Point3D(0, 0, 0), Point3D(-1, -1, -1))));
            break;
          }
#endif
#if LEGION_MAX_DIM >= 4
      case 4:
          {
            const coord_t lovals[4] = { 0, 0, 0, 0 };
            const coord_t hivals[4] = {-1, -1, -1, -1 };
            (*coloring)[c.p].insert(
                Domain(Rect4D(Point4D(lovals), Point4D(hivals))));
            break;
          }
#endif
#if LEGION_MAX_DIM >= 5
      case 5:
          {
            const coord_t lovals[5] = { 0, 0, 0, 0, 0 };
            const coord_t hivals[5] = {-1, -1, -1, -1, -1 };
            (*coloring)[c.p].insert(
                Domain(Rect5D(Point5D(lovals), Point5D(hivals))));
            break;
          }
#endif
#if LEGION_MAX_DIM >= 6
      case 6:
          {
            const coord_t lovals[6] = { 0, 0, 0, 0, 0, 0 };
            const coord_t hivals[6] = {-1, -1, -1, -1, -1, -1 };
            (*coloring)[c.p].insert(
                Domain(Rect6D(Point6D(lovals), Point6D(hivals))));
            break;
          }
#endif
#if LEGION_MAX_DIM >= 7
      case 7:
          {
            const coord_t lovals[7] = { 0, 0, 0, 0, 0, 0, 0 };
            const coord_t hivals[7] = {-1, -1, -1, -1, -1, -1, -1 };
            (*coloring)[c.p].insert(
                Domain(Rect7D(Point7D(lovals), Point7D(hivals))));
            break;
          }
#endif
#if LEGION_MAX_DIM >= 8
      case 8:
          {
            const coord_t lovals[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            const coord_t hivals[8] = {-1, -1, -1, -1, -1, -1, -1, -1 };
            (*coloring)[c.p].insert(
                Domain(Rect8D(Point8D(lovals), Point8D(hivals))));
            break;
          }
#endif
#if LEGION_MAX_DIM >= 9
      case 9:
          {
            const coord_t lovals[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            const coord_t hivals[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1 };
            (*coloring)[c.p].insert(
                Domain(Rect9D(Point9D(lovals), Point9D(hivals))));
            break;
          }
#endif
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

#define CREATE_BLOCKIFY(DIM) \
legion_index_partition_t \
legion_index_partition_create_blockify_##DIM##d( \
  legion_runtime_t runtime_, \
  legion_context_t ctx_, \
  legion_index_space_t parent_, \
  legion_blockify_##DIM##d_t blockify_, \
  legion_color_t part_color /* = AUTO_GENERATE_ID */) \
{ \
  Runtime *runtime = CObjectWrapper::unwrap(runtime_); \
  Context ctx = CObjectWrapper::unwrap(ctx_)->context(); \
  IndexSpace parent = CObjectWrapper::unwrap(parent_); \
  CObjectWrapper::Blockify<DIM> blockify = CObjectWrapper::unwrap(blockify_); \
 \
  IndexPartition ip = \
    runtime->create_partition_by_blockify(ctx, IndexSpaceT<DIM,coord_t>(parent), \
        blockify.block_size, blockify.offset, part_color); \
  return CObjectWrapper::wrap(ip); \
}
LEGION_FOREACH_N(CREATE_BLOCKIFY)
#undef CREATE_BLOCKIFY

legion_index_partition_t
legion_index_partition_create_equal(legion_runtime_t runtime_,
                                    legion_context_t ctx_,
                                    legion_index_space_t parent_,
                                    legion_index_space_t color_space_,
                                    size_t granularity,
                                    legion_color_t color /* = AUTO_GENERATE_ID */)
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
legion_index_partition_create_by_weights(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_domain_point_t *colors_,
  int *weights_,
  size_t num_colors,
  legion_index_space_t color_space_,
  size_t granularity /* = 1 */,
  legion_color_t color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);
  std::map<DomainPoint,int> weights;
  for (unsigned idx = 0; idx < num_colors; idx++)
    weights[CObjectWrapper::unwrap(colors_[idx])] = weights_[idx]; 

  IndexPartition ip =
    runtime->create_partition_by_weights(ctx, parent, weights, 
                              color_space, granularity, color);
  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_by_weights_future_map(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_future_map_t future_map_,
  legion_index_space_t color_space_,
  size_t granularity /* = 1 */,
  legion_color_t color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  FutureMap *future_map = CObjectWrapper::unwrap(future_map_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_partition_by_weights(ctx, parent, *future_map,
                                  color_space, granularity, color);
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
  legion_color_t color /* = AUTO_GENERATE_ID */)
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
  legion_color_t color /* = AUTO_GENERATE_ID */)
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
legion_index_partition_create_by_intersection_mirror(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_index_partition_t handle_,
  legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
  legion_color_t color /* = AUTO_GENERATE_ID */,
  bool dominates /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  IndexPartition ip =
    runtime->create_partition_by_intersection(ctx, parent, handle, part_kind,
                                              color, dominates);
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
  legion_color_t color /* = AUTO_GENERATE_ID */)
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
legion_index_partition_create_by_domain(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_domain_point_t *colors_,
  legion_domain_t *domains_,
  size_t num_color_domains,
  legion_index_space_t color_space_,
  bool perform_intersections /* = true */,
  legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
  legion_color_t color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);
  std::map<DomainPoint,Domain> domains;
  for (unsigned idx = 0; idx < num_color_domains; idx++)
    domains[CObjectWrapper::unwrap(colors_[idx])] = 
      CObjectWrapper::unwrap(domains_[idx]);

  IndexPartition ip =
    runtime->create_partition_by_domain(ctx, parent, domains, 
        color_space, perform_intersections, part_kind, color);
  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_by_domain_future_map(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_future_map_t future_map_,
  legion_index_space_t color_space_,
  bool perform_intersections,
  legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
  legion_color_t color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  FutureMap *future_map = CObjectWrapper::unwrap(future_map_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_partition_by_domain(ctx, parent, *future_map,
        color_space, perform_intersections, part_kind, color);
  return CObjectWrapper::wrap(ip);
}

legion_index_partition_t
legion_index_partition_create_by_field(legion_runtime_t runtime_,
                                       legion_context_t ctx_,
                                       legion_logical_region_t handle_,
                                       legion_logical_region_t parent_,
                                       legion_field_id_t fid,
                                       legion_index_space_t color_space_,
                                       legion_color_t color /* = AUTO_GENERATE_ID */,
                                       legion_mapper_id_t id /* = 0 */,
                                       legion_mapping_tag_id_t tag /* = 0 */,
                                       legion_partition_kind_t part_kind /* = DISJOINT_KIND */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_partition_by_field(ctx, handle, parent, fid, color_space,
                                       color, id, tag, part_kind);

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
  legion_color_t color /* = AUTO_GENERATE_ID */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t tag /* = 0 */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace handle = CObjectWrapper::unwrap(handle_);
  LogicalPartition projection = CObjectWrapper::unwrap(projection_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_partition_by_image(
      ctx, handle, projection, parent, fid, color_space, part_kind, color, id, tag);

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
  legion_color_t color /* = AUTO_GENERATE_ID */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t tag /* = 0 */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexPartition projection = CObjectWrapper::unwrap(projection_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_partition_by_preimage(
      ctx, projection, handle, parent, fid, color_space, part_kind, color, id, tag);

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
  legion_color_t color /* = AUTO_GENERATE_ID */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t tag /* = 0 */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace handle = CObjectWrapper::unwrap(handle_);
  LogicalPartition projection = CObjectWrapper::unwrap(projection_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_partition_by_image_range(
      ctx, handle, projection, parent, fid, color_space, part_kind, color, id, tag);

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
  legion_color_t color /* = AUTO_GENERATE_ID */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t tag /* = 0 */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexPartition projection = CObjectWrapper::unwrap(projection_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_partition_by_preimage_range(
      ctx, projection, handle, parent, fid, color_space, part_kind, color, id, tag);

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
    legion_color_t color /* = AUTO_GENERATE_ID */)
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

legion_index_partition_t
legion_index_partition_create_pending_partition(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_index_space_t color_space_,
  legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
  legion_color_t color /* = AUTO_GENERATE_ID */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  IndexSpace color_space = CObjectWrapper::unwrap(color_space_);

  IndexPartition ip =
    runtime->create_pending_partition(
        ctx, parent, color_space, part_kind, color);

  return CObjectWrapper::wrap(ip);
}

legion_index_space_t
legion_index_partition_create_index_space_union_spaces(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_partition_t parent_,
  legion_domain_point_t color_,
  const legion_index_space_t *spaces_,
  size_t num_spaces)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexPartition parent = CObjectWrapper::unwrap(parent_);
  DomainPoint color = CObjectWrapper::unwrap(color_);

  std::vector<IndexSpace> handles;
  for (size_t idx = 0; idx < num_spaces; ++idx)
    handles.push_back(CObjectWrapper::unwrap(spaces_[idx]));

  return CObjectWrapper::wrap(
      runtime->create_index_space_union(ctx, parent, color, handles));
}

legion_index_space_t
legion_index_partition_create_index_space_union_partition(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_partition_t parent_,
  legion_domain_point_t color_,
  legion_index_partition_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexPartition parent = CObjectWrapper::unwrap(parent_);
  DomainPoint color = CObjectWrapper::unwrap(color_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  return CObjectWrapper::wrap(
      runtime->create_index_space_union(ctx, parent, color, handle));
}

legion_index_space_t
legion_index_partition_create_index_space_intersection_spaces(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_partition_t parent_,
  legion_domain_point_t color_,
  const legion_index_space_t *spaces_,
  size_t num_spaces)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexPartition parent = CObjectWrapper::unwrap(parent_);
  DomainPoint color = CObjectWrapper::unwrap(color_);

  std::vector<IndexSpace> handles;
  for (size_t idx = 0; idx < num_spaces; ++idx)
    handles.push_back(CObjectWrapper::unwrap(spaces_[idx]));

  return CObjectWrapper::wrap(
      runtime->create_index_space_intersection(ctx, parent, color, handles));
}

legion_index_space_t
legion_index_partition_create_index_space_intersection_partition(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_partition_t parent_,
  legion_domain_point_t color_,
  legion_index_partition_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexPartition parent = CObjectWrapper::unwrap(parent_);
  DomainPoint color = CObjectWrapper::unwrap(color_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  return CObjectWrapper::wrap(
      runtime->create_index_space_intersection(ctx, parent, color, handle));
}

legion_index_space_t
legion_index_partition_create_index_space_difference(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_partition_t parent_,
  legion_domain_point_t color_,
  legion_index_space_t initial_,
  const legion_index_space_t *spaces_,
  size_t num_spaces)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexPartition parent = CObjectWrapper::unwrap(parent_);
  DomainPoint color = CObjectWrapper::unwrap(color_);
  IndexSpace initial = CObjectWrapper::unwrap(initial_);

  std::vector<IndexSpace> handles;
  for (size_t idx = 0; idx < num_spaces; ++idx)
    handles.push_back(CObjectWrapper::unwrap(spaces_[idx]));

  return CObjectWrapper::wrap(
      runtime->create_index_space_difference(
        ctx, parent, color, initial, handles));
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
legion_index_partition_create_shared_ownership(legion_runtime_t runtime_,
                                               legion_context_t ctx_,
                                               legion_index_partition_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  runtime->create_shared_ownership(ctx, handle);
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
legion_index_partition_destroy_unordered(legion_runtime_t runtime_,
                                         legion_context_t ctx_,
                                         legion_index_partition_t handle_,
                                         bool unordered, bool recurse)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  runtime->destroy_index_partition(ctx, handle, unordered, recurse);
}

void
legion_index_partition_attach_semantic_information(
                                              legion_runtime_t runtime_,
                                              legion_index_partition_t handle_,
                                              legion_semantic_tag_t tag,
                                              const void *buffer,
                                              size_t size,
                                              bool is_mutable /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  runtime->attach_semantic_information(handle, tag, buffer, size, is_mutable);
}

bool
legion_index_partition_retrieve_semantic_information(
                                         legion_runtime_t runtime_,
                                         legion_index_partition_t handle_,
                                         legion_semantic_tag_t tag,
                                         const void **result,
                                         size_t *size,
                                         bool can_fail /* = false */,
                                         bool wait_until_ready /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  return runtime->retrieve_semantic_information(
                       handle, tag, *result, *size, can_fail, wait_until_ready);
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

legion_field_space_t
legion_field_space_create_with_fields(legion_runtime_t runtime_,
                                      legion_context_t ctx_,
                                      size_t *field_sizes,
                                      legion_field_id_t *field_ids,
                                      size_t num_fields,
                                      legion_custom_serdez_id_t serdez)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  std::vector<size_t> sizes(num_fields);
  std::vector<FieldID> ids(num_fields);
  for (unsigned idx = 0; idx < num_fields; idx++)
  {
    sizes[idx] = field_sizes[idx];
    ids[idx] = field_ids[idx];
  }
  FieldSpace fs = runtime->create_field_space(ctx, sizes, ids, serdez);
  for (unsigned idx = 0; idx < num_fields; idx++)
    field_ids[idx] = ids[idx];
  return CObjectWrapper::wrap(fs);
}

legion_field_space_t
legion_field_space_create_with_futures(legion_runtime_t runtime_,
                                       legion_context_t ctx_,
                                       legion_future_t *field_sizes,
                                       legion_field_id_t *field_ids,
                                       size_t num_fields,
                                       legion_custom_serdez_id_t serdez)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  std::vector<Future> sizes(num_fields);
  std::vector<FieldID> ids(num_fields);
  for (unsigned idx = 0; idx < num_fields; idx++)
  {
    sizes[idx] = *CObjectWrapper::unwrap(field_sizes[idx]);
    ids[idx] = field_ids[idx];
  }
  FieldSpace fs = runtime->create_field_space(ctx, sizes, ids, serdez);
  for (unsigned idx = 0; idx < num_fields; idx++)
    field_ids[idx] = ids[idx];
  return CObjectWrapper::wrap(fs);
}

void
legion_field_space_create_shared_ownership(legion_runtime_t runtime_,
                                           legion_context_t ctx_,
                                           legion_field_space_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->create_shared_ownership(ctx, handle);
}

legion_field_space_t
legion_field_space_no_space()
{
  return CObjectWrapper::wrap(FieldSpace::NO_SPACE);
}

void
legion_field_space_destroy(legion_runtime_t runtime_,
                           legion_context_t ctx_,
                           legion_field_space_t handle_)
{
  legion_field_space_destroy_unordered(runtime_, ctx_, handle_, false);
}

void
legion_field_space_destroy_unordered(legion_runtime_t runtime_,
                                     legion_context_t ctx_,
                                     legion_field_space_t handle_,
                                     bool unordered)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->destroy_field_space(ctx, handle, unordered);
}

void
legion_field_space_attach_semantic_information(
                                              legion_runtime_t runtime_,
                                              legion_field_space_t handle_,
                                              legion_semantic_tag_t tag,
                                              const void *buffer,
                                              size_t size,
                                              bool is_mutable /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->attach_semantic_information(handle, tag, buffer, size, is_mutable);
}

bool
legion_field_space_retrieve_semantic_information(
                                         legion_runtime_t runtime_,
                                         legion_field_space_t handle_,
                                         legion_semantic_tag_t tag,
                                         const void **result,
                                         size_t *size,
                                         bool can_fail /* = false */,
                                         bool wait_until_ready /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  return runtime->retrieve_semantic_information(
                       handle, tag, *result, *size, can_fail, wait_until_ready);
}

legion_field_id_t *
legion_field_space_get_fields(legion_runtime_t runtime_,
                              legion_context_t ctx_,
                              legion_field_space_t handle_,
                              size_t *size)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  std::vector<FieldID> fields;
  runtime->get_field_space_fields(ctx, handle, fields);
  legion_field_id_t *result = (legion_field_id_t *)malloc(sizeof(legion_field_id_t) * fields.size());
  std::copy(fields.begin(), fields.end(), result);
  *size = fields.size();
  return result;
}

bool
legion_field_space_has_fields(legion_runtime_t runtime_,
                              legion_context_t ctx_,
                              legion_field_space_t handle_,
                              const legion_field_id_t *fields_,
                              size_t fields_size)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  std::set<FieldID> fields;
  runtime->get_field_space_fields(ctx, handle, fields);
  for (size_t idx = 0; idx < fields_size; ++idx)
    if (fields.find(fields_[idx]) == fields.end()) return false;
  return true;
}

void
legion_field_id_attach_semantic_information(legion_runtime_t runtime_,
                                            legion_field_space_t handle_,
                                            legion_field_id_t id,
                                            legion_semantic_tag_t tag,
                                            const void *buffer,
                                            size_t size,
                                            bool is_mutable /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->attach_semantic_information(
                                     handle, id, tag, buffer, size, is_mutable);
}

bool
legion_field_id_retrieve_semantic_information(
                                         legion_runtime_t runtime_,
                                         legion_field_space_t handle_,
                                         legion_field_id_t id,
                                         legion_semantic_tag_t tag,
                                         const void **result,
                                         size_t *size,
                                         bool can_fail /* = false */,
                                         bool wait_until_ready /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  return runtime->retrieve_semantic_information(
                   handle, id, tag, *result, *size, can_fail, wait_until_ready);
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

size_t
legion_field_id_get_size(legion_runtime_t runtime_,
                         legion_context_t ctx_,
                         legion_field_space_t handle_,
                         legion_field_id_t id)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  return runtime->get_field_size(ctx, handle, id);
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
legion_logical_region_create_shared_ownership(legion_runtime_t runtime_,
                                              legion_context_t ctx_,
                                              legion_logical_region_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);

  runtime->create_shared_ownership(ctx, handle);
}

void
legion_logical_region_destroy(legion_runtime_t runtime_,
                              legion_context_t ctx_,
                              legion_logical_region_t handle_)
{
  legion_logical_region_destroy_unordered(runtime_, ctx_, handle_, false);
}

void
legion_logical_region_destroy_unordered(legion_runtime_t runtime_,
                                        legion_context_t ctx_,
                                        legion_logical_region_t handle_,
                                        bool unordered)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);

  runtime->destroy_logical_region(ctx, handle, unordered);
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
legion_logical_region_attach_semantic_information(
                                              legion_runtime_t runtime_,
                                              legion_logical_region_t handle_,
                                              legion_semantic_tag_t tag,
                                              const void *buffer,
                                              size_t size,
                                              bool is_mutable /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);

  runtime->attach_semantic_information(handle, tag, buffer, size, is_mutable);
}

bool
legion_logical_region_retrieve_semantic_information(
                                         legion_runtime_t runtime_,
                                         legion_logical_region_t handle_,
                                         legion_semantic_tag_t tag,
                                         const void **result,
                                         size_t *size,
                                         bool can_fail /* = false */,
                                         bool wait_until_ready /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);

  return runtime->retrieve_semantic_information(
                       handle, tag, *result, *size, can_fail, wait_until_ready);
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
                                legion_logical_region_t parent_,
                                legion_index_partition_t handle_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  IndexPartition handle = CObjectWrapper::unwrap(handle_);

  LogicalPartition r = runtime->get_logical_partition(parent, handle);
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
  legion_logical_partition_destroy_unordered(runtime_, ctx_, handle_, false);
}

void
legion_logical_partition_destroy_unordered(legion_runtime_t runtime_,
                                           legion_context_t ctx_,
                                           legion_logical_partition_t handle_,
                                           bool unordered)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);

  runtime->destroy_logical_partition(ctx, handle, unordered);
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
legion_logical_partition_attach_semantic_information(
                                              legion_runtime_t runtime_,
                                              legion_logical_partition_t handle_,
                                              legion_semantic_tag_t tag,
                                              const void *buffer,
                                              size_t size,
                                              bool is_mutable /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);

  runtime->attach_semantic_information(handle, tag, buffer, size, is_mutable);
}

bool
legion_logical_partition_retrieve_semantic_information(
                                         legion_runtime_t runtime_,
                                         legion_logical_partition_t handle_,
                                         legion_semantic_tag_t tag,
                                         const void **result,
                                         size_t *size,
                                         bool can_fail /* = false */,
                                         bool wait_until_ready /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);

  return runtime->retrieve_semantic_information(
                       handle, tag, *result, *size, can_fail, wait_until_ready);
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

legion_region_requirement_t
legion_region_requirement_create_logical_region(
  legion_logical_region_t handle_,
  legion_privilege_mode_t priv,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag,
  bool verified)
{
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  return CObjectWrapper::wrap(
      new RegionRequirement(handle, priv, prop, parent, tag, verified));
}

legion_region_requirement_t
legion_region_requirement_create_logical_region_projection(
  legion_logical_region_t handle_,
  legion_projection_id_t proj,
  legion_privilege_mode_t priv,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag,
  bool verified)
{
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  return CObjectWrapper::wrap(
      new RegionRequirement(handle, proj, priv, prop, parent, tag, verified));
}

legion_region_requirement_t
legion_region_requirement_create_logical_partition(
  legion_logical_partition_t handle_,
  legion_projection_id_t proj,
  legion_privilege_mode_t priv,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag,
  bool verified)
{
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  return CObjectWrapper::wrap(
      new RegionRequirement(handle, proj, priv, prop, parent, tag, verified));
}

void
legion_region_requirement_destroy(legion_region_requirement_t handle_)
{
  delete CObjectWrapper::unwrap(handle_);
}

void
legion_region_requirement_add_field(legion_region_requirement_t req_,
                                    legion_field_id_t field,
                                    bool instance)
{
  CObjectWrapper::unwrap(req_)->add_field(field, instance);
}

void
legion_region_requirement_add_flags(legion_region_requirement_t req_,
                                    legion_region_flags_t flags)
{
  CObjectWrapper::unwrap(req_)->add_flags(flags);
}

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
  assert(idx < req->instance_fields.size());

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

  assert(idx < req->instance_fields.size());
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
legion_auto_generate_id(void)
{
  return LEGION_AUTO_GENERATE_ID;
}

legion_field_id_t
legion_field_allocator_allocate_field(legion_field_allocator_t allocator_,
                                      size_t field_size,
                                      legion_field_id_t desired_fieldid)
{
  FieldAllocator *allocator = CObjectWrapper::unwrap(allocator_);
  return allocator->allocate_field(field_size, desired_fieldid);
}

legion_field_id_t
legion_field_allocator_allocate_field_future(legion_field_allocator_t allocator_,
                                             legion_future_t field_size_,
                                             legion_field_id_t desired_fieldid)
{
  FieldAllocator *allocator = CObjectWrapper::unwrap(allocator_);
  Future *field_size = CObjectWrapper::unwrap(field_size_);
  return allocator->allocate_field(*field_size, desired_fieldid);
}

void
legion_field_allocator_free_field(legion_field_allocator_t allocator_,
                                  legion_field_id_t fid)
{
  legion_field_allocator_free_field_unordered(allocator_, fid, false);
}

void
legion_field_allocator_free_field_unordered(legion_field_allocator_t allocator_,
                                            legion_field_id_t fid,
                                            bool unordered)
{
  FieldAllocator *allocator = CObjectWrapper::unwrap(allocator_);
  allocator->free_field(fid, unordered);
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

legion_argument_map_t
legion_argument_map_from_future_map(legion_future_map_t map_)
{
  FutureMap *map = CObjectWrapper::unwrap(map_);

  return CObjectWrapper::wrap(new ArgumentMap(*map));
}

void
legion_argument_map_set_point(legion_argument_map_t map_,
                              legion_domain_point_t dp_,
                              legion_untyped_buffer_t arg_,
                              bool replace)
{
  ArgumentMap *map = CObjectWrapper::unwrap(map_);
  DomainPoint dp = CObjectWrapper::unwrap(dp_);
  UntypedBuffer arg = CObjectWrapper::unwrap(arg_);

  map->set_point(dp, arg, replace);
}

void
legion_argument_map_set_future(legion_argument_map_t map_,
                               legion_domain_point_t dp_,
                               legion_future_t future_,
                               bool replace)
{
  ArgumentMap *map = CObjectWrapper::unwrap(map_);
  DomainPoint dp = CObjectWrapper::unwrap(dp_);
  Future *future = CObjectWrapper::unwrap(future_);

  map->set_point(dp, *future, replace);
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

legion_predicate_t
legion_predicate_create(legion_runtime_t runtime_,
                        legion_context_t ctx_,
                        legion_future_t f_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  Future *f = CObjectWrapper::unwrap(f_);

  Predicate result = runtime->create_predicate(ctx, *f);
  return CObjectWrapper::wrap(new Predicate(result));
}

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

void
legion_future_wait(legion_future_t handle_,
                   bool silence_warnings,
                   const char *warning_string)
{
  Future *handle = CObjectWrapper::unwrap(handle_);

  handle->get_void_result(silence_warnings, warning_string);
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

bool
legion_future_is_ready_subscribe(legion_future_t handle_, bool subscribe)
{
  Future *handle = CObjectWrapper::unwrap(handle_);

  return handle->is_ready(subscribe);
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

legion_future_t
legion_future_map_reduce(legion_runtime_t runtime_,
                         legion_context_t ctx_,
                         legion_future_map_t fm_,
                         legion_reduction_op_id_t redop,
                         bool deterministic)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  FutureMap *fm = CObjectWrapper::unwrap(fm_);

  return CObjectWrapper::wrap(new Future(
        runtime->reduce_future_map(ctx, *fm, redop, deterministic)));
}

legion_future_map_t
legion_future_map_construct_from_buffers(legion_runtime_t runtime_,
                                         legion_context_t ctx_,
                                         legion_domain_t domain_,
                                         legion_domain_point_t *points_,
                                         legion_untyped_buffer_t *data_,
                                         size_t num_points,
                                         bool collective,
                                         legion_sharding_id_t sid,
                                         bool implicit_sharding)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  Domain domain = CObjectWrapper::unwrap(domain_);
  std::map<DomainPoint,UntypedBuffer> data;
  for (unsigned idx = 0; idx < num_points; idx++)
  {
    DomainPoint point = CObjectWrapper::unwrap(points_[idx]);
    data[point] = CObjectWrapper::unwrap(data_[idx]);
  }
  return CObjectWrapper::wrap(new FutureMap(
    runtime->construct_future_map(ctx, domain, data, collective, sid,
                                  implicit_sharding)));
}

legion_future_map_t
legion_future_map_construct_from_futures(legion_runtime_t runtime_,
                                         legion_context_t ctx_,
                                         legion_domain_t domain_,
                                         legion_domain_point_t *points_,
                                         legion_future_t *futures_,
                                         size_t num_futures,
                                         bool collective,
                                         legion_sharding_id_t sid,
                                         bool implicit_sharding)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  Domain domain = CObjectWrapper::unwrap(domain_);
  std::map<DomainPoint,Future> futures;
  for (unsigned idx = 0; idx < num_futures; idx++)
  {
    DomainPoint point = CObjectWrapper::unwrap(points_[idx]);
    futures[point] = *(CObjectWrapper::unwrap(futures_[idx]));
  }
  return CObjectWrapper::wrap(new FutureMap(
    runtime->construct_future_map(ctx, domain, futures, collective, sid,
                                  implicit_sharding)));
}

// -----------------------------------------------------------------------
// Deferred Buffer Operations
// -----------------------------------------------------------------------

#define CREATE_BUFFER(DIM) \
legion_deferred_buffer_char_##DIM##d_t \
legion_deferred_buffer_char_##DIM##d_create( \
    legion_rect_##DIM##d_t bounds_, \
    legion_memory_kind_t kind_, \
    char *initial_value) \
{ \
  Rect##DIM##D bounds = CObjectWrapper::unwrap(bounds_); \
  Memory::Kind kind = CObjectWrapper::unwrap(kind_); \
 \
  return CObjectWrapper::wrap( \
      new DeferredBufferChar##DIM##D(bounds, kind, initial_value)); \
}
LEGION_FOREACH_N(CREATE_BUFFER)
#undef CREATE_BUFFER

#define BUFFER_PTR(DIM) \
char* \
legion_deferred_buffer_char_##DIM##d_ptr( \
    legion_deferred_buffer_char_##DIM##d_t buffer_, \
    legion_point_##DIM##d_t p_) \
{ \
  DeferredBufferChar##DIM##D *buffer = CObjectWrapper::unwrap(buffer_); \
  Point##DIM##D p = CObjectWrapper::unwrap(p_); \
  return buffer->ptr(p); \
}
LEGION_FOREACH_N(BUFFER_PTR)
#undef BUFFER_PTR

#define BUFFER_DESTROY(DIM) \
void \
legion_deferred_buffer_char_##DIM##d_destroy( \
    legion_deferred_buffer_char_##DIM##d_t buffer_) \
{ \
  DeferredBufferChar##DIM##D *buffer = CObjectWrapper::unwrap(buffer_); \
  delete buffer; \
}
LEGION_FOREACH_N(BUFFER_DESTROY)
#undef BUFFER_DESTROY

//------------------------------------------------------------------------
// Task Launch Operations
//------------------------------------------------------------------------

legion_task_launcher_t
legion_task_launcher_create(
  legion_task_id_t tid,
  legion_untyped_buffer_t arg_,
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t tag /* = 0 */)
{
  UntypedBuffer arg = CObjectWrapper::unwrap(arg_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  TaskLauncher *launcher = new TaskLauncher(tid, arg, *pred, id, tag);
  return CObjectWrapper::wrap(launcher);
}

legion_task_launcher_t
legion_task_launcher_create_from_buffer(
  legion_task_id_t tid,
  const void *buffer,
  size_t buffer_size,
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t tag /* = 0 */)
{
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  TaskLauncher *launcher = new TaskLauncher(tid, 
      UntypedBuffer(buffer, buffer_size), *pred, id, tag);
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
  if (launcher->elide_future_return)
  {
    legion_future_t result_;
    result_.impl = nullptr;
    return result_;
  }
  else
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
legion_task_launcher_set_region_requirement_logical_region(
  legion_task_launcher_t launcher_,
  unsigned idx,
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

  if (idx >= launcher->region_requirements.size()) {
    launcher->region_requirements.resize(idx + 1);
  }
  launcher->region_requirements[idx] =
    RegionRequirement(handle, priv, prop, parent, tag, verified);
}

void
legion_task_launcher_set_region_requirement_logical_region_reduction(
  legion_task_launcher_t launcher_,
  unsigned idx,
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


  if (idx >= launcher->region_requirements.size()) {
    launcher->region_requirements.resize(idx + 1);
  }
  launcher->region_requirements[idx] =
    RegionRequirement(handle, redop, prop, parent, tag, verified);
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

const void*
legion_index_launcher_get_projection_args(legion_region_requirement_t requirement_,
					  size_t *size)
{
  return CObjectWrapper::unwrap(requirement_)->get_projection_args(size);
}

void
legion_index_launcher_set_projection_args(legion_index_launcher_t launcher_,
					  unsigned idx,
					  const void *args,
					  size_t size,
					  bool own)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->region_requirements[idx].set_projection_args(args, size, own);
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

void
legion_task_launcher_set_argument(legion_task_launcher_t launcher_,
                                  legion_untyped_buffer_t arg_)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  UntypedBuffer arg = CObjectWrapper::unwrap(arg_);

  launcher->argument = arg;
}

void
legion_task_launcher_set_point(legion_task_launcher_t launcher_,
                               legion_domain_point_t point_)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  DomainPoint point = CObjectWrapper::unwrap(point_);

  launcher->point = point;
}

void
legion_task_launcher_set_sharding_space(legion_task_launcher_t launcher_,
                                        legion_index_space_t is_)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  IndexSpace is = CObjectWrapper::unwrap(is_);

  launcher->sharding_space = is;
}

void
legion_task_launcher_set_predicate_false_future(legion_task_launcher_t launcher_,
                                                legion_future_t f_)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  Future *f = CObjectWrapper::unwrap(f_);

  launcher->predicate_false_future = *f;
}

void
legion_task_launcher_set_predicate_false_result(legion_task_launcher_t launcher_,
                                                legion_untyped_buffer_t arg_)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  UntypedBuffer arg = CObjectWrapper::unwrap(arg_);

  launcher->predicate_false_result = arg;
}

void
legion_task_launcher_set_mapper(legion_task_launcher_t launcher_,
                                legion_mapper_id_t mapper_id)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->map_id = mapper_id;
}

void
legion_task_launcher_set_mapping_tag(legion_task_launcher_t launcher_,
                                     legion_mapping_tag_id_t tag)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->tag = tag;
}

void
legion_task_launcher_set_mapper_arg(legion_task_launcher_t launcher_,
                                    legion_untyped_buffer_t arg_)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  UntypedBuffer arg = CObjectWrapper::unwrap(arg_);

  launcher->map_arg = arg;
}

void
legion_task_launcher_set_enable_inlining(legion_task_launcher_t launcher_,
                                         bool enable_inlining)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->enable_inlining = enable_inlining;
}

void
legion_task_launcher_set_local_function_task(legion_task_launcher_t launcher_,
                                             bool local_function_task)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->local_function_task = local_function_task;
}

void
legion_task_launcher_set_elide_future_return(legion_task_launcher_t launcher_,
                                             bool elide_future_return)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->elide_future_return = elide_future_return;
}

void
legion_task_launcher_set_provenance(legion_task_launcher_t launcher_,
                                    const char *provenance)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->provenance = provenance;
}

legion_index_launcher_t
legion_index_launcher_create(
  legion_task_id_t tid,
  legion_domain_t domain_,
  legion_untyped_buffer_t global_arg_,
  legion_argument_map_t map_,
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  bool must /* = false */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t tag /* = 0 */)
{
  Domain domain = CObjectWrapper::unwrap(domain_);
  UntypedBuffer global_arg = CObjectWrapper::unwrap(global_arg_);
  ArgumentMap *map = CObjectWrapper::unwrap(map_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  IndexTaskLauncher *launcher =
    new IndexTaskLauncher(tid, domain, global_arg, *map, *pred, must, id, tag);
  return CObjectWrapper::wrap(launcher);
}

legion_index_launcher_t
legion_index_launcher_create_from_buffer(
  legion_task_id_t tid,
  legion_domain_t domain_,
  const void *buffer,
  size_t buffer_size,
  legion_argument_map_t map_,
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  bool must /* = false */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t tag /* = 0 */)
{
  Domain domain = CObjectWrapper::unwrap(domain_);
  ArgumentMap *map = CObjectWrapper::unwrap(map_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  IndexTaskLauncher *launcher = new IndexTaskLauncher(tid, domain,
      UntypedBuffer(buffer, buffer_size), *map, *pred, must, id, tag);
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
  if (launcher->elide_future_return)
  {
    legion_future_map_t result_;
    result_.impl = nullptr;
    return result_;
  }
  else
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
  if (launcher->elide_future_return)
  {
    legion_future_t result_;
    result_.impl = nullptr;
    return result_;
  }
  else
    return CObjectWrapper::wrap(new Future(f));
}

legion_future_t
legion_index_launcher_execute_deterministic_reduction(
                                        legion_runtime_t runtime_,
                                        legion_context_t ctx_,
                                        legion_index_launcher_t launcher_,
                                        legion_reduction_op_id_t redop,
                                        bool deterministic)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  Future f = runtime->execute_index_space(ctx, *launcher, redop, deterministic);
  if (launcher->elide_future_return)
  {
    legion_future_t result_;
    result_.impl = nullptr;
    return result_;
  }
  else
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
legion_index_launcher_set_region_requirement_logical_region(
  legion_index_launcher_t launcher_,
  unsigned idx,
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

  if (idx >= launcher->region_requirements.size()) {
    launcher->region_requirements.resize(idx + 1);
  }
  launcher->region_requirements[idx] =
    RegionRequirement(handle, proj, priv, prop, parent, tag, verified);
}

void
legion_index_launcher_set_region_requirement_logical_partition(
  legion_index_launcher_t launcher_,
  unsigned idx,
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

  if (idx >= launcher->region_requirements.size()) {
    launcher->region_requirements.resize(idx + 1);
  }
  launcher->region_requirements[idx] =
    RegionRequirement(handle, proj, priv, prop, parent, tag, verified);
}

void
legion_index_launcher_set_region_requirement_logical_region_reduction(
  legion_index_launcher_t launcher_,
  unsigned idx,
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

  if (idx >= launcher->region_requirements.size()) {
    launcher->region_requirements.resize(idx + 1);
  }
  launcher->region_requirements[idx] =
    RegionRequirement(handle, proj, redop, prop, parent, tag, verified);
}

void
legion_index_launcher_set_region_requirement_logical_partition_reduction(
  legion_index_launcher_t launcher_,
  unsigned idx,
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

  if (idx >= launcher->region_requirements.size()) {
    launcher->region_requirements.resize(idx + 1);
  }
  launcher->region_requirements[idx] =
    RegionRequirement(handle, proj, redop, prop, parent, tag, verified);
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

void
legion_index_launcher_add_point_future(legion_index_launcher_t launcher_,
                                       legion_argument_map_t map_)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  ArgumentMap *map = CObjectWrapper::unwrap(map_);

  launcher->point_futures.push_back(*map);
}

void
legion_index_launcher_set_global_arg(legion_index_launcher_t launcher_,
                                     legion_untyped_buffer_t global_arg_)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  UntypedBuffer global_arg = CObjectWrapper::unwrap(global_arg_);

  launcher->global_arg = global_arg;
}

void
legion_index_launcher_set_sharding_space(legion_index_launcher_t launcher_,
                                         legion_index_space_t is_)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  IndexSpace is = CObjectWrapper::unwrap(is_);

  launcher->sharding_space = is;
}

void
legion_index_launcher_set_mapper(legion_index_launcher_t launcher_,
                                 legion_mapper_id_t mapper_id)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->map_id = mapper_id;
}

void
legion_index_launcher_set_mapping_tag(legion_index_launcher_t launcher_,
                                      legion_mapping_tag_id_t tag)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->tag = tag;
}

void
legion_index_launcher_set_mapper_arg(legion_index_launcher_t launcher_,
                                     legion_untyped_buffer_t arg_)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  UntypedBuffer arg = CObjectWrapper::unwrap(arg_);

  launcher->map_arg = arg;
}

void
legion_index_launcher_set_elide_future_return(legion_index_launcher_t launcher_,
                                              bool elide_future_return)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->elide_future_return = elide_future_return;
}

void
legion_index_launcher_set_provenance(legion_index_launcher_t launcher_,
                                     const char *provenance)
{
  IndexTaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->provenance = provenance;
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
legion_inline_launcher_set_mapper_arg(legion_inline_launcher_t launcher_,
                                      legion_untyped_buffer_t arg_)
{
  InlineLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  UntypedBuffer arg = CObjectWrapper::unwrap(arg_);

  launcher->map_arg = arg;
}

void
legion_inline_launcher_set_provenance(legion_inline_launcher_t launcher_,
                                      const char *provenance)
{
  InlineLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->provenance = provenance;
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

legion_fill_launcher_t
legion_fill_launcher_create(
  legion_logical_region_t handle_,
  legion_logical_region_t parent_,
  legion_field_id_t fid,
  const void *value,
  size_t value_size,
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t tag /* = 0 */)
{
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);
  
  FillLauncher *launcher = new FillLauncher(handle, parent,
      UntypedBuffer(value, value_size), *pred, id, tag); 
  launcher->add_field(fid);
  return CObjectWrapper::wrap(launcher);
}

legion_fill_launcher_t
legion_fill_launcher_create_from_future(
    legion_logical_region_t handle_,
    legion_logical_region_t parent_,
    legion_field_id_t fid,
    legion_future_t future_,
    legion_predicate_t pred_ /* = legion_predicate_true() */,
    legion_mapper_id_t id /* = 0 */,
    legion_mapping_tag_id_t tag /* = 0 */)
{
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  Future *future = CObjectWrapper::unwrap(future_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  FillLauncher *launcher = 
    new FillLauncher(handle, parent, *future, *pred, id, tag);
  launcher->add_field(fid);
  return CObjectWrapper::wrap(launcher);
}

void
legion_fill_launcher_destroy(legion_fill_launcher_t handle_)
{
  FillLauncher *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

void
legion_fill_launcher_add_field(legion_fill_launcher_t launcher_,
                               legion_field_id_t fid)
{
  FillLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->add_field(fid);
}

void
legion_fill_launcher_execute(legion_runtime_t runtime_,
                             legion_context_t ctx_,
                             legion_fill_launcher_t launcher_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  FillLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  runtime->fill_fields(ctx, *launcher);
}

void
legion_fill_launcher_set_point(legion_fill_launcher_t launcher_,
                               legion_domain_point_t point_)
{
  FillLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  DomainPoint point = CObjectWrapper::unwrap(point_);

  launcher->point = point;
}

void
legion_fill_launcher_set_sharding_space(legion_fill_launcher_t launcher_,
                                        legion_index_space_t space_)
{
  FillLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  IndexSpace is = CObjectWrapper::unwrap(space_);

  launcher->sharding_space = is;
}

void
legion_fill_launcher_set_mapper_arg(legion_fill_launcher_t launcher_,
                                    legion_untyped_buffer_t arg_)
{
  FillLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  UntypedBuffer arg = CObjectWrapper::unwrap(arg_);

  launcher->map_arg = arg;
}

void
legion_fill_launcher_set_provenance(legion_fill_launcher_t launcher_,
                                    const char *provenance)
{
  FillLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->provenance = provenance;
}

// -----------------------------------------------------------------------
// Index Fill Field Operations
// -----------------------------------------------------------------------

void
legion_runtime_index_fill_field(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_logical_partition_t handle_,
  legion_logical_region_t parent_,
  legion_field_id_t fid,
  const void *value,
  size_t value_size,
  legion_projection_id_t proj /* = 0 */,
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t launcher_tag /* = 0 */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  IndexFillLauncher launcher(
      runtime->get_index_partition_color_space_name(handle.get_index_partition()),
      handle, parent, UntypedBuffer(value, value_size), proj,
      *pred, id, launcher_tag);
  launcher.add_field(fid);
  runtime->fill_fields(ctx, launcher);
}

void
legion_runtime_index_fill_field_with_space(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t space_,
  legion_logical_partition_t handle_,
  legion_logical_region_t parent_,
  legion_field_id_t fid,
  const void *value,
  size_t value_size,
  legion_projection_id_t proj /* = 0 */,
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t launcher_tag /* = 0 */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace space = CObjectWrapper::unwrap(space_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  IndexFillLauncher launcher(
      space, handle, parent, UntypedBuffer(value, value_size), proj,
      *pred, id, launcher_tag);
  launcher.add_field(fid);
  runtime->fill_fields(ctx, launcher);
}

void
legion_runtime_index_fill_field_with_domain(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_domain_t domain_,
  legion_logical_partition_t handle_,
  legion_logical_region_t parent_,
  legion_field_id_t fid,
  const void *value,
  size_t value_size,
  legion_projection_id_t proj /* = 0 */,
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t launcher_tag /* = 0 */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  Domain domain = CObjectWrapper::unwrap(domain_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  IndexFillLauncher launcher(
      domain, handle, parent, UntypedBuffer(value, value_size), proj,
      *pred, id, launcher_tag);
  launcher.add_field(fid);
  runtime->fill_fields(ctx, launcher);
}

void
legion_runtime_index_fill_field_future(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_logical_partition_t handle_,
  legion_logical_region_t parent_,
  legion_field_id_t fid,
  legion_future_t f_,
  legion_projection_id_t proj /* = 0 */,
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t launcher_tag /* = 0 */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  Future *f = CObjectWrapper::unwrap(f_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  IndexFillLauncher launcher(
      runtime->get_index_partition_color_space_name(handle.get_index_partition()),
      handle, parent, *f, proj, *pred, id, launcher_tag);
  launcher.add_field(fid);
  runtime->fill_fields(ctx, launcher);
}

void
legion_runtime_index_fill_field_future_with_space(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t space_,
  legion_logical_partition_t handle_,
  legion_logical_region_t parent_,
  legion_field_id_t fid,
  legion_future_t f_,
  legion_projection_id_t proj /* = 0 */,
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t launcher_tag /* = 0 */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexSpace space = CObjectWrapper::unwrap(space_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  Future *f = CObjectWrapper::unwrap(f_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  IndexFillLauncher launcher(
      space, handle, parent, *f, proj, *pred, id, launcher_tag);
  launcher.add_field(fid);
  runtime->fill_fields(ctx, launcher);
}

void
legion_runtime_index_fill_field_future_with_domain(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_domain_t domain_,
  legion_logical_partition_t handle_,
  legion_logical_region_t parent_,
  legion_field_id_t fid,
  legion_future_t f_,
  legion_projection_id_t proj /* = 0 */,
  legion_predicate_t pred_ /* = legion_predicate_true() */,
  legion_mapper_id_t id /* = 0 */,
  legion_mapping_tag_id_t launcher_tag /* = 0 */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  Domain domain = CObjectWrapper::unwrap(domain_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  Future *f = CObjectWrapper::unwrap(f_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  IndexFillLauncher launcher(
      domain, handle, parent, *f, proj, *pred, id, launcher_tag);
  launcher.add_field(fid);
  runtime->fill_fields(ctx, launcher);
}

legion_index_fill_launcher_t
legion_index_fill_launcher_create_with_space(
    legion_index_space_t space_,
    legion_logical_partition_t handle_,
    legion_logical_region_t parent_,
    legion_field_id_t fid,
    const void *value,
    size_t value_size,
    legion_projection_id_t proj /* = 0 */,
    legion_predicate_t pred_ /* = legion_predicate_true() */,
    legion_mapper_id_t id /* = 0 */,
    legion_mapping_tag_id_t launcher_tag /* = 0 */)
{
  IndexSpace space = CObjectWrapper::unwrap(space_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  IndexFillLauncher *launcher = new IndexFillLauncher(space, handle, parent,
      UntypedBuffer(value, value_size), proj, *pred, id, launcher_tag);
  launcher->add_field(fid);
  return CObjectWrapper::wrap(launcher);
}

legion_index_fill_launcher_t
legion_index_fill_launcher_create_with_domain(
    legion_domain_t domain_,
    legion_logical_partition_t handle_,
    legion_logical_region_t parent_,
    legion_field_id_t fid,
    const void *value,
    size_t value_size,
    legion_projection_id_t proj /* = 0 */,
    legion_predicate_t pred_ /* = legion_predicate_true() */,
    legion_mapper_id_t id /* = 0 */,
    legion_mapping_tag_id_t launcher_tag /* = 0 */)
{
  Domain domain = CObjectWrapper::unwrap(domain_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  IndexFillLauncher *launcher = new IndexFillLauncher(domain, handle, parent,
      UntypedBuffer(value, value_size), proj, *pred, id, launcher_tag);
  launcher->add_field(fid);
  return CObjectWrapper::wrap(launcher);
}

legion_index_fill_launcher_t
legion_index_fill_launcher_create_from_future_with_space(
    legion_index_space_t space_,
    legion_logical_partition_t handle_,
    legion_logical_region_t parent_,
    legion_field_id_t fid,
    legion_future_t future_,
    legion_projection_id_t proj /* = 0 */,
    legion_predicate_t pred_ /* = legion_predicate_true() */,
    legion_mapper_id_t id /* = 0 */,
    legion_mapping_tag_id_t launcher_tag /* = 0 */)
{
  IndexSpace space = CObjectWrapper::unwrap(space_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  Future *future = CObjectWrapper::unwrap(future_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  IndexFillLauncher *launcher = new IndexFillLauncher(space, handle,
      parent, *future, proj, *pred, id, launcher_tag);
  launcher->add_field(fid);
  return CObjectWrapper::wrap(launcher);
}

legion_index_fill_launcher_t
legion_index_fill_launcher_create_from_future_with_domain(
    legion_domain_t domain_,
    legion_logical_partition_t handle_,
    legion_logical_region_t parent_,
    legion_field_id_t fid,
    legion_future_t future_,
    legion_projection_id_t proj /* = 0 */,
    legion_predicate_t pred_ /* = legion_predicate_true() */,
    legion_mapper_id_t id /* = 0 */,
    legion_mapping_tag_id_t launcher_tag /* = 0 */)
{
  Domain domain = CObjectWrapper::unwrap(domain_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);
  Future *future = CObjectWrapper::unwrap(future_);
  Predicate *pred = CObjectWrapper::unwrap(pred_);

  IndexFillLauncher *launcher = new IndexFillLauncher(domain,
      handle, parent, *future, proj, *pred, id, launcher_tag);
  launcher->add_field(fid);
  return CObjectWrapper::wrap(launcher);
}

void
legion_index_fill_launcher_destroy(legion_index_fill_launcher_t handle_)
{
  IndexFillLauncher *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

void
legion_index_fill_launcher_add_field(legion_index_fill_launcher_t launcher_,
                                     legion_field_id_t fid)
{
  IndexFillLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->add_field(fid);
}

void
legion_index_fill_launcher_execute(legion_runtime_t runtime_,
                                   legion_context_t ctx_,
                                   legion_index_fill_launcher_t launcher_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexFillLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  runtime->fill_fields(ctx, *launcher);
}

void
legion_index_fill_launcher_set_sharding_space(legion_index_fill_launcher_t launcher_,
                                              legion_index_space_t space_)
{
  IndexFillLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  IndexSpace is = CObjectWrapper::unwrap(space_);

  launcher->sharding_space = is;
}

void
legion_index_fill_launcher_set_mapper_arg(legion_index_fill_launcher_t launcher_,
                                          legion_untyped_buffer_t arg_)
{
  IndexFillLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  UntypedBuffer arg = CObjectWrapper::unwrap(arg_);

  launcher->map_arg = arg;
}

void
legion_index_fill_launcher_set_provenance(legion_index_fill_launcher_t launcher_,
                                          const char *provenance)
{
  IndexFillLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->provenance = provenance;
}

legion_region_requirement_t
legion_fill_get_requirement(legion_fill_t fill_)
{
  Fill *fill = CObjectWrapper::unwrap(fill_);

  return CObjectWrapper::wrap(&fill->requirement);
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

  AttachLauncher launcher(LEGION_EXTERNAL_HDF5_FILE, handle, parent);
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

unsigned
legion_copy_launcher_add_src_indirect_region_requirement_logical_region(
  legion_copy_launcher_t launcher_,
  legion_logical_region_t handle_,
  legion_field_id_t fid,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool is_range_indirection /* = false */,
  bool verified /* = false*/)
{
  CopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->src_requirements.size();
  launcher->add_src_indirect_field(fid,
    RegionRequirement(handle, LEGION_READ_ONLY, prop, parent, tag, verified),
    is_range_indirection);
  return idx;
}

unsigned
legion_copy_launcher_add_dst_indirect_region_requirement_logical_region(
  legion_copy_launcher_t launcher_,
  legion_logical_region_t handle_,
  legion_field_id_t fid,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool is_range_indirection /* = false */,
  bool verified /* = false*/)
{
  CopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->dst_requirements.size();
  launcher->add_dst_indirect_field(fid,
    RegionRequirement(handle, LEGION_READ_ONLY, prop, parent, tag, verified),
    is_range_indirection);
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

void
legion_copy_launcher_set_possible_src_indirect_out_of_range(
    legion_copy_launcher_t launcher_, bool flag)
{
  CopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  launcher->possible_src_indirect_out_of_range = flag;
}

void
legion_copy_launcher_set_possible_dst_indirect_out_of_range(
    legion_copy_launcher_t launcher_, bool flag)
{
  CopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  launcher->possible_dst_indirect_out_of_range = flag;
}

void
legion_copy_launcher_set_point(legion_copy_launcher_t launcher_,
                               legion_domain_point_t point_)
{
  CopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  DomainPoint point = CObjectWrapper::unwrap(point_);

  launcher->point = point;
}

void
legion_copy_launcher_set_sharding_space(legion_copy_launcher_t launcher_,
                                        legion_index_space_t space_)
{
  CopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  IndexSpace is = CObjectWrapper::unwrap(space_);

  launcher->sharding_space = is;
}

void
legion_copy_launcher_set_mapper_arg(legion_copy_launcher_t launcher_,
                                    legion_untyped_buffer_t arg_)
{
  CopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  UntypedBuffer arg = CObjectWrapper::unwrap(arg_);

  launcher->map_arg = arg;
}

void
legion_copy_launcher_set_provenance(legion_copy_launcher_t launcher_,
                                    const char *provenance)
{
  CopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->provenance = provenance;
}

legion_region_requirement_t
legion_copy_get_requirement(legion_copy_t copy_, unsigned idx)
{
  Copy *copy = CObjectWrapper::unwrap(copy_);

  if (idx < copy->src_requirements.size())
    return CObjectWrapper::wrap(&copy->src_requirements[idx]);
  else
    idx -= copy->src_requirements.size();
  if (idx < copy->dst_requirements.size())
    return CObjectWrapper::wrap(&copy->dst_requirements[idx]);
  else
    idx -= copy->dst_requirements.size();
  if (idx < copy->src_indirect_requirements.size())
    return CObjectWrapper::wrap(&copy->src_indirect_requirements[idx]);
  else
    idx -= copy->src_indirect_requirements.size();
  assert(idx < copy->dst_indirect_requirements.size());
  return CObjectWrapper::wrap(&copy->dst_indirect_requirements[idx]);
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

unsigned
legion_index_copy_launcher_add_src_indirect_region_requirement_logical_region(
  legion_index_copy_launcher_t launcher_,
  legion_logical_region_t handle_,
  legion_projection_id_t proj /* = 0 */,
  legion_field_id_t fid,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool is_range_indirection /* = false */,
  bool verified /* = false*/)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->src_requirements.size();
  launcher->add_src_indirect_field(fid,
    RegionRequirement(handle, proj, LEGION_READ_ONLY, prop, parent, tag, verified),
    is_range_indirection);
  return idx;
}

unsigned
legion_index_copy_launcher_add_dst_indirect_region_requirement_logical_region(
  legion_index_copy_launcher_t launcher_,
  legion_logical_region_t handle_,
  legion_projection_id_t proj /* = 0 */,
  legion_field_id_t fid,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool is_range_indirection /* = false */,
  bool verified /* = false*/)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->dst_requirements.size();
  launcher->add_dst_indirect_field(fid,
    RegionRequirement(handle, proj, LEGION_READ_ONLY, prop, parent, tag, verified),
    is_range_indirection);
  return idx;
}

unsigned
legion_index_copy_launcher_add_src_indirect_region_requirement_logical_partition(
  legion_index_copy_launcher_t launcher_,
  legion_logical_partition_t handle_,
  legion_projection_id_t proj /* = 0 */,
  legion_field_id_t fid,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool is_range_indirection /* = false */,
  bool verified /* = false*/)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->src_requirements.size();
  launcher->add_src_indirect_field(fid,
    RegionRequirement(handle, proj, LEGION_READ_ONLY, prop, parent, tag, verified),
    is_range_indirection);
  return idx;
}

unsigned
legion_index_copy_launcher_add_dst_indirect_region_requirement_logical_partition(
  legion_index_copy_launcher_t launcher_,
  legion_logical_partition_t handle_,
  legion_projection_id_t proj /* = 0 */,
  legion_field_id_t fid,
  legion_coherence_property_t prop,
  legion_logical_region_t parent_,
  legion_mapping_tag_id_t tag /* = 0 */,
  bool is_range_indirection /* = false */,
  bool verified /* = false*/)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  unsigned idx = launcher->dst_requirements.size();
  launcher->add_dst_indirect_field(fid,
    RegionRequirement(handle, proj, LEGION_READ_ONLY, prop, parent, tag, verified),
    is_range_indirection);
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

void
legion_index_copy_launcher_set_possible_src_indirect_out_of_range(
    legion_index_copy_launcher_t launcher_, bool flag)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  launcher->possible_src_indirect_out_of_range = flag;
}

void
legion_index_copy_launcher_set_possible_dst_indirect_out_of_range(
    legion_index_copy_launcher_t launcher_, bool flag)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  launcher->possible_dst_indirect_out_of_range = flag;
}

void
legion_index_copy_launcher_set_sharding_space(legion_index_copy_launcher_t launcher_,
                                              legion_index_space_t space_)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  IndexSpace is = CObjectWrapper::unwrap(space_);

  launcher->sharding_space = is;
}

void
legion_index_copy_launcher_set_mapper_arg(legion_index_copy_launcher_t launcher_,
                                          legion_untyped_buffer_t arg_)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  UntypedBuffer arg = CObjectWrapper::unwrap(arg_);

  launcher->map_arg = arg;
}

void
legion_index_copy_launcher_set_provenance(legion_index_copy_launcher_t launcher_,
                                          const char *provenance)
{
  IndexCopyLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->provenance = provenance;
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

void
legion_acquire_launcher_set_mapper_arg(legion_acquire_launcher_t launcher_,
                                       legion_untyped_buffer_t arg_)
{
  AcquireLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  UntypedBuffer arg = CObjectWrapper::unwrap(arg_);

  launcher->map_arg = arg;
}

void
legion_acquire_launcher_set_provenance(legion_acquire_launcher_t launcher_,
                                       const char *provenance)
{
  AcquireLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->provenance = provenance;
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

void
legion_release_launcher_set_mapper_arg(legion_release_launcher_t launcher_,
                                       legion_untyped_buffer_t arg_)
{
  ReleaseLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  UntypedBuffer arg = CObjectWrapper::unwrap(arg_);

  launcher->map_arg = arg;
}

void
legion_release_launcher_set_provenance(legion_release_launcher_t launcher_,
                                       const char *provenance)
{
  ReleaseLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->provenance = provenance;
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
legion_attach_launcher_attach_hdf5(legion_attach_launcher_t handle_,
                                   const char *filename,
                                   legion_field_map_t field_map_,
                                   legion_file_mode_t mode)
{
  AttachLauncher *handle = CObjectWrapper::unwrap(handle_);

  std::map<FieldID, const char *> *field_map =
    CObjectWrapper::unwrap(field_map_);

  handle->attach_hdf5(filename, *field_map, mode);
}

void
legion_attach_launcher_set_restricted(legion_attach_launcher_t handle_,
                                      bool restricted)
{
  AttachLauncher *handle = CObjectWrapper::unwrap(handle_);

  handle->restricted = restricted;
}

void
legion_attach_launcher_set_mapped(legion_attach_launcher_t handle_,
                                  bool mapped)
{
  AttachLauncher *handle = CObjectWrapper::unwrap(handle_);

  handle->mapped = mapped;
}

void
legion_attach_launcher_set_provenance(legion_attach_launcher_t handle_,
                                      const char *provenance)
{
  AttachLauncher *handle = CObjectWrapper::unwrap(handle_);

  handle->provenance = provenance;
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
  launcher->attach_array_soa(base_ptr, column_major, fields);
}

legion_future_t
legion_detach_external_resource(legion_runtime_t runtime_,
                                legion_context_t ctx_,
                                legion_physical_region_t handle_)
{
  return legion_unordered_detach_external_resource(runtime_, ctx_, handle_, true, false);
}

legion_future_t
legion_flush_detach_external_resource(legion_runtime_t runtime_,
                                      legion_context_t ctx_,
                                      legion_physical_region_t handle_,
                                      bool flush)
{
  return legion_unordered_detach_external_resource(runtime_, ctx_, handle_, flush, false);
}

legion_future_t
legion_unordered_detach_external_resource(legion_runtime_t runtime_,
                                          legion_context_t ctx_,
                                          legion_physical_region_t handle_,
                                          bool flush, bool unordered)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);

  Future *result = new Future(
      runtime->detach_external_resource(ctx, *handle, flush, unordered));
  return CObjectWrapper::wrap(result);
}

void
legion_context_progress_unordered_operations(legion_runtime_t runtime_,
                                             legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  runtime->progress_unordered_operations(ctx);
}

// -----------------------------------------------------------------------
// Index Attach/Detach Operations
// -----------------------------------------------------------------------

legion_index_attach_launcher_t
legion_index_attach_launcher_create(
    legion_logical_region_t parent_region_,
    legion_external_resource_t resource,
    bool restricted)
{
  LogicalRegion parent_region = CObjectWrapper::unwrap(parent_region_);

  IndexAttachLauncher *launcher =
    new IndexAttachLauncher(resource, parent_region, restricted);
  return CObjectWrapper::wrap(launcher);
}

void
legion_index_attach_launcher_set_restricted(
    legion_index_attach_launcher_t handle_,
    bool restricted)
{
  IndexAttachLauncher *handle = CObjectWrapper::unwrap(handle_);

  handle->restricted = restricted;
}

void
legion_index_attach_launcher_set_provenance(
    legion_index_attach_launcher_t handle_, const char *provenance)
{
  IndexAttachLauncher *handle = CObjectWrapper::unwrap(handle_);

  handle->provenance = provenance;
}

void
legion_index_attach_launcher_set_deduplicate_across_shards(
    legion_index_attach_launcher_t handle_,
    bool deduplicate)
{
  IndexAttachLauncher *handle = CObjectWrapper::unwrap(handle_);

  handle->deduplicate_across_shards = deduplicate;
}

void
legion_index_attach_launcher_attach_file(legion_index_attach_launcher_t handle_,
                                         legion_logical_region_t region_,
                                         const char *filename,
                                         const legion_field_id_t *fields_,
                                         size_t num_fields,
                                         legion_file_mode_t mode)
{
  IndexAttachLauncher *handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion region = CObjectWrapper::unwrap(region_);

  std::vector<FieldID> fields(num_fields);
  for (unsigned idx = 0; idx < num_fields; idx++)
    fields[idx] = fields_[idx];

  handle->attach_file(region, filename, fields, mode);
}

void
legion_index_attach_launcher_attach_hdf5(legion_index_attach_launcher_t handle_,
                                         legion_logical_region_t region_,
                                         const char *filename,
                                         legion_field_map_t field_map_,
                                         legion_file_mode_t mode)
{
  IndexAttachLauncher *handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion region = CObjectWrapper::unwrap(region_);

  std::map<FieldID, const char *> *field_map =
    CObjectWrapper::unwrap(field_map_);

  handle->attach_hdf5(region, filename, *field_map, mode);
}

void
legion_index_attach_launcher_attach_array_soa(legion_index_attach_launcher_t handle_,
                                         legion_logical_region_t region_,
                                         void *base_ptr, bool column_major,
                                         const legion_field_id_t *fields_,
                                         size_t num_fields,
                                         legion_memory_t memory_)
{
  IndexAttachLauncher *handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion region = CObjectWrapper::unwrap(region_);
  Memory memory = CObjectWrapper::unwrap(memory_);

  std::vector<FieldID> fields(num_fields);
  for (unsigned idx = 0; idx < num_fields; idx++)
    fields[idx] = fields_[idx];

  handle->attach_array_soa(region, base_ptr, column_major, fields, memory);
}

void
legion_index_attach_launcher_attach_array_aos(legion_index_attach_launcher_t handle_,
                                         legion_logical_region_t region_,
                                         void *base_ptr, bool column_major,
                                         const legion_field_id_t *fields_,
                                         size_t num_fields,
                                         legion_memory_t memory_)
{
  IndexAttachLauncher *handle = CObjectWrapper::unwrap(handle_);
  LogicalRegion region = CObjectWrapper::unwrap(region_);
  Memory memory = CObjectWrapper::unwrap(memory_);

  std::vector<FieldID> fields(num_fields);
  for (unsigned idx = 0; idx < num_fields; idx++)
    fields[idx] = fields_[idx];

  handle->attach_array_aos(region, base_ptr, column_major, fields, memory);
}

void
legion_index_attach_launcher_destroy(legion_index_attach_launcher_t handle_)
{
  IndexAttachLauncher *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

legion_external_resources_t
legion_attach_external_resources(legion_runtime_t runtime_,
                                 legion_context_t ctx_,
                                 legion_index_attach_launcher_t launcher_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  IndexAttachLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  ExternalResources resources = 
    runtime->attach_external_resources(ctx, *launcher);
  return CObjectWrapper::wrap(new ExternalResources(resources));
}

legion_future_t
legion_detach_external_resources(legion_runtime_t runtime_,
                                 legion_context_t ctx_,
                                 legion_external_resources_t handle_,
                                 bool flush, bool unordered)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  ExternalResources *handle = CObjectWrapper::unwrap(handle_);

  Future *result = new Future(
      runtime->detach_external_resources(ctx, *handle, flush, unordered));
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

void
legion_must_epoch_launcher_set_launch_domain(
  legion_must_epoch_launcher_t launcher_,
  legion_domain_t domain_)
{
  MustEpochLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  Domain domain = CObjectWrapper::unwrap(domain_);

  launcher->launch_domain = domain;
}

void
legion_must_epoch_launcher_set_launch_space(
  legion_must_epoch_launcher_t launcher_,
  legion_index_space_t is_)
{
  MustEpochLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  IndexSpace is = CObjectWrapper::unwrap(is_);

  launcher->launch_space = is;
}

void
legion_must_epoch_launcher_set_provenance(
  legion_must_epoch_launcher_t launcher_, const char *provenance)
{
  MustEpochLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->provenance = provenance;
}

// -----------------------------------------------------------------------
// Tracing Operations
// -----------------------------------------------------------------------

void
legion_runtime_begin_trace(legion_runtime_t runtime_,
                           legion_context_t ctx_,
                           legion_trace_id_t tid,
                           bool logical_only)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  runtime->begin_trace(ctx, tid, logical_only);
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
// Frame Operations
// -----------------------------------------------------------------------

void
legion_runtime_complete_frame(legion_runtime_t runtime_,
                              legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  runtime->complete_frame(ctx);
}


// -----------------------------------------------------------------------
// Fence Operations
// -----------------------------------------------------------------------

legion_future_t
legion_runtime_issue_mapping_fence(legion_runtime_t runtime_,
                                   legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  Future *result = new Future(runtime->issue_mapping_fence(ctx));
  return CObjectWrapper::wrap(result);
}

legion_future_t
legion_runtime_issue_execution_fence(legion_runtime_t runtime_,
                                     legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  Future *result = new Future(runtime->issue_execution_fence(ctx));
  return CObjectWrapper::wrap(result);
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

bool
legion_runtime_has_runtime()
{
  return Runtime::has_runtime();
}

legion_runtime_t
legion_runtime_get_runtime()
{
  Runtime *runtime = Runtime::get_runtime();
  return CObjectWrapper::wrap(runtime);
}

bool
legion_runtime_has_context()
{
  return Runtime::has_context();
}

legion_context_t
legion_runtime_get_context()
{
  Context ctx = Runtime::get_context();
  CContext *cctx = new CContext(ctx);
  return CObjectWrapper::wrap(cctx);
}

void
legion_context_destroy(legion_context_t cctx_)
{
  CContext *cctx = CObjectWrapper::unwrap(cctx_);
  assert(cctx->num_regions() == 0 && "do not manually destroy automatically created contexts");
  delete cctx;
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

void
legion_runtime_yield(legion_runtime_t runtime_, legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  runtime->yield(ctx);
}

legion_shard_id_t
legion_runtime_local_shard(legion_runtime_t runtime_, legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  return runtime->local_shard(ctx);
}

size_t
legion_runtime_total_shards(legion_runtime_t runtime_, legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  return runtime->total_shards(ctx);
}

void
legion_runtime_enable_scheduler_lock()
{
  Processor::enable_scheduler_lock();
}

void
legion_runtime_disable_scheduler_lock()
{
  Processor::disable_scheduler_lock();
}

void
legion_runtime_print_once(legion_runtime_t runtime_,
                          legion_context_t ctx_,
                          FILE *f,
                          const char *message)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  runtime->print_once(ctx, f, message);
}

void
legion_runtime_print_once_fd(legion_runtime_t runtime_,
                             legion_context_t ctx_,
                             int fd, const char *mode,
                             const char *message)
{
  FILE *f = fdopen(fd, mode);
  legion_runtime_print_once(runtime_, ctx_, f, message);
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

legion_physical_region_t
legion_physical_region_copy(legion_physical_region_t handle_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);

  PhysicalRegion *result = new PhysicalRegion(*handle);
  return CObjectWrapper::wrap(result);
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
  assert((index < fields.size()));
  return fields[index];
}

size_t
legion_physical_region_get_memory_count(legion_physical_region_t handle_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);
  std::set<Memory> memories;
  handle->get_memories(memories);
  return memories.size();
}

legion_memory_t
legion_physical_region_get_memory(legion_physical_region_t handle_, size_t index)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);
  std::set<Memory> memories;
  handle->get_memories(memories);
  std::set<Memory>::iterator it = memories.begin();
  for (size_t i = 0; i < index; i++, it++) {
    assert(it != memories.end());
  }
  return CObjectWrapper::wrap(*it);
}

#define GET_ACCESSOR(DIM) \
legion_accessor_array_##DIM##d_t \
legion_physical_region_get_field_accessor_array_##DIM##d( \
  legion_physical_region_t handle_, \
  legion_field_id_t fid) \
{ \
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_); \
  UnsafeFieldAccessor<char,DIM,coord_t,Realm::AffineAccessor<char,DIM,coord_t> > \
    *accessor = new UnsafeFieldAccessor<char,DIM,coord_t, \
                      Realm::AffineAccessor<char,DIM,coord_t> >(*handle, fid); \
 \
  return CObjectWrapper::wrap(accessor); \
}
LEGION_FOREACH_N(GET_ACCESSOR)
#undef GET_ACCESSOR

#define DESTROY_ACCESSOR(DIM) \
void \
legion_accessor_array_##DIM##d_destroy(legion_accessor_array_##DIM##d_t handle_) \
{ \
  UnsafeFieldAccessor<char,DIM,coord_t,Realm::AffineAccessor<char,DIM,coord_t> > \
    *handle = CObjectWrapper::unwrap(handle_); \
 \
  delete handle; \
}
LEGION_FOREACH_N(DESTROY_ACCESSOR)
#undef DESTROY_ACCESSOR

#define RAW_RECT_PTR(DIM) \
void *                    \
legion_accessor_array_##DIM##d_raw_rect_ptr(legion_accessor_array_##DIM##d_t handle_, \
                                            legion_rect_##DIM##d_t rect_, \
                                            legion_rect_##DIM##d_t *subrect_, \
                                            legion_byte_offset_t *offsets_) \
{ \
  UnsafeFieldAccessor<char,DIM,coord_t,Realm::AffineAccessor<char,DIM,coord_t> > \
    *handle = CObjectWrapper::unwrap(handle_); \
  Rect##DIM##D rect = CObjectWrapper::unwrap(rect_); \
  \
  void *data = handle->ptr(rect.lo); \
  *subrect_ = CObjectWrapper::wrap(rect); \
  for (int i = 0; i < DIM; i++) \
    offsets_[i] = CObjectWrapper::wrap(handle->accessor.strides[i]); \
  return data; \
}
LEGION_FOREACH_N(RAW_RECT_PTR)
#undef RAW_RECT_PTR

#if LEGION_MAX_DIM >= 1
legion_accessor_array_1d_t
legion_physical_region_get_field_accessor_array_1d_with_transform(
  legion_physical_region_t handle_,
  legion_field_id_t fid,
  legion_domain_affine_transform_t transform_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);
  DomainAffineTransform domtrans = CObjectWrapper::unwrap(transform_);
  UnsafeFieldAccessor<char,1,coord_t,Realm::AffineAccessor<char,1,coord_t> >
    *accessor = NULL;
  assert(domtrans.transform.n == 1);
  switch (domtrans.transform.m)
  {
#define AFFINE(DIM) \
    case DIM: \
      { \
        const AffineTransform<DIM,1,coord_t> transform = domtrans; \
        accessor = new UnsafeFieldAccessor<char,1,coord_t, \
                 Realm::AffineAccessor<char,1,coord_t> >(*handle, fid, transform); \
        break; \
      }
    LEGION_FOREACH_N(AFFINE)
#undef AFFINE
    default:
      assert(false);
  }

  return CObjectWrapper::wrap(accessor);
}
#endif

#if LEGION_MAX_DIM >= 2
legion_accessor_array_2d_t
legion_physical_region_get_field_accessor_array_2d_with_transform(
  legion_physical_region_t handle_,
  legion_field_id_t fid,
  legion_domain_affine_transform_t transform_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);
  DomainAffineTransform domtrans = CObjectWrapper::unwrap(transform_);
  UnsafeFieldAccessor<char,2,coord_t,Realm::AffineAccessor<char,2,coord_t> >
    *accessor = NULL;
  assert(domtrans.transform.n == 2);
  switch (domtrans.transform.m)
  {
#define AFFINE(DIM) \
    case DIM: \
      { \
        const AffineTransform<DIM,2,coord_t> transform = domtrans; \
        accessor = new UnsafeFieldAccessor<char,2,coord_t, \
                 Realm::AffineAccessor<char,2,coord_t> >(*handle, fid, transform); \
        break; \
      }
    LEGION_FOREACH_N(AFFINE)
#undef AFFINE
    default:
      assert(false);
  }

  return CObjectWrapper::wrap(accessor);
}
#endif

#if LEGION_MAX_DIM >= 3
legion_accessor_array_3d_t
legion_physical_region_get_field_accessor_array_3d_with_transform(
  legion_physical_region_t handle_,
  legion_field_id_t fid,
  legion_domain_affine_transform_t transform_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);
  DomainAffineTransform domtrans = CObjectWrapper::unwrap(transform_);
  UnsafeFieldAccessor<char,3,coord_t,Realm::AffineAccessor<char,3,coord_t> >
    *accessor = NULL;
  assert(domtrans.transform.n == 3);
  switch (domtrans.transform.m)
  {
#define AFFINE(DIM) \
    case DIM: \
      { \
        const AffineTransform<DIM,3,coord_t> transform = domtrans; \
        accessor = new UnsafeFieldAccessor<char,3,coord_t, \
                 Realm::AffineAccessor<char,3,coord_t> >(*handle, fid, transform); \
        break; \
      }
    LEGION_FOREACH_N(AFFINE)
#undef AFFINE
    default:
      assert(false);
  }

  return CObjectWrapper::wrap(accessor);
}
#endif

#if LEGION_MAX_DIM >= 4
legion_accessor_array_4d_t
legion_physical_region_get_field_accessor_array_4d_with_transform(
  legion_physical_region_t handle_,
  legion_field_id_t fid,
  legion_domain_affine_transform_t transform_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);
  DomainAffineTransform domtrans = CObjectWrapper::unwrap(transform_);
  UnsafeFieldAccessor<char,4,coord_t,Realm::AffineAccessor<char,4,coord_t> >
    *accessor = NULL;
  assert(domtrans.transform.n == 4);
  switch (domtrans.transform.m)
  {
#define AFFINE(DIM) \
    case DIM: \
      { \
        const AffineTransform<DIM,4,coord_t> transform = domtrans; \
        accessor = new UnsafeFieldAccessor<char,4,coord_t, \
                 Realm::AffineAccessor<char,4,coord_t> >(*handle, fid, transform); \
        break; \
      }
    LEGION_FOREACH_N(AFFINE)
#undef AFFINE
    default:
      assert(false);
  }

  return CObjectWrapper::wrap(accessor);
}
#endif

#if LEGION_MAX_DIM >= 5
legion_accessor_array_5d_t
legion_physical_region_get_field_accessor_array_5d_with_transform(
  legion_physical_region_t handle_,
  legion_field_id_t fid,
  legion_domain_affine_transform_t transform_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);
  DomainAffineTransform domtrans = CObjectWrapper::unwrap(transform_);
  UnsafeFieldAccessor<char,5,coord_t,Realm::AffineAccessor<char,5,coord_t> >
    *accessor = NULL;
  assert(domtrans.transform.n == 5);
  switch (domtrans.transform.m)
  {
#define AFFINE(DIM) \
    case DIM: \
      { \
        const AffineTransform<DIM,5,coord_t> transform = domtrans; \
        accessor = new UnsafeFieldAccessor<char,5,coord_t, \
                 Realm::AffineAccessor<char,5,coord_t> >(*handle, fid, transform); \
        break; \
      }
    LEGION_FOREACH_N(AFFINE)
#undef AFFINE
    default:
      assert(false);
  }

  return CObjectWrapper::wrap(accessor);
}
#endif

#if LEGION_MAX_DIM >= 6
legion_accessor_array_6d_t
legion_physical_region_get_field_accessor_array_6d_with_transform(
  legion_physical_region_t handle_,
  legion_field_id_t fid,
  legion_domain_affine_transform_t transform_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);
  DomainAffineTransform domtrans = CObjectWrapper::unwrap(transform_);
  UnsafeFieldAccessor<char,6,coord_t,Realm::AffineAccessor<char,6,coord_t> >
    *accessor = NULL;
  assert(domtrans.transform.n == 6);
  switch (domtrans.transform.m)
  {
#define AFFINE(DIM) \
    case DIM: \
      { \
        const AffineTransform<DIM,6,coord_t> transform = domtrans; \
        accessor = new UnsafeFieldAccessor<char,6,coord_t, \
                 Realm::AffineAccessor<char,6,coord_t> >(*handle, fid, transform); \
        break; \
      }
    LEGION_FOREACH_N(AFFINE)
#undef AFFINE
    default:
      assert(false);
  }

  return CObjectWrapper::wrap(accessor);
}
#endif

#if LEGION_MAX_DIM >= 7
legion_accessor_array_7d_t
legion_physical_region_get_field_accessor_array_7d_with_transform(
  legion_physical_region_t handle_,
  legion_field_id_t fid,
  legion_domain_affine_transform_t transform_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);
  DomainAffineTransform domtrans = CObjectWrapper::unwrap(transform_);
  UnsafeFieldAccessor<char,7,coord_t,Realm::AffineAccessor<char,7,coord_t> >
    *accessor = NULL;
  assert(domtrans.transform.n == 7);
  switch (domtrans.transform.m)
  {
#define AFFINE(DIM) \
    case DIM: \
      { \
        const AffineTransform<DIM,7,coord_t> transform = domtrans; \
        accessor = new UnsafeFieldAccessor<char,7,coord_t, \
                 Realm::AffineAccessor<char,7,coord_t> >(*handle, fid, transform); \
        break; \
      }
    LEGION_FOREACH_N(AFFINE)
#undef AFFINE
    default:
      assert(false);
  }

  return CObjectWrapper::wrap(accessor);
}
#endif

#if LEGION_MAX_DIM >= 8
legion_accessor_array_8d_t
legion_physical_region_get_field_accessor_array_8d_with_transform(
  legion_physical_region_t handle_,
  legion_field_id_t fid,
  legion_domain_affine_transform_t transform_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);
  DomainAffineTransform domtrans = CObjectWrapper::unwrap(transform_);
  UnsafeFieldAccessor<char,8,coord_t,Realm::AffineAccessor<char,8,coord_t> >
    *accessor = NULL;
  assert(domtrans.transform.n == 8);
  switch (domtrans.transform.m)
  {
#define AFFINE(DIM) \
    case DIM: \
      { \
        const AffineTransform<DIM,8,coord_t> transform = domtrans; \
        accessor = new UnsafeFieldAccessor<char,8,coord_t, \
                 Realm::AffineAccessor<char,8,coord_t> >(*handle, fid, transform); \
        break; \
      }
    LEGION_FOREACH_N(AFFINE)
#undef AFFINE
    default:
      assert(false);
  }

  return CObjectWrapper::wrap(accessor);
}
#endif

#if LEGION_MAX_DIM >= 9
legion_accessor_array_9d_t
legion_physical_region_get_field_accessor_array_9d_with_transform(
  legion_physical_region_t handle_,
  legion_field_id_t fid,
  legion_domain_affine_transform_t transform_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);
  DomainAffineTransform domtrans = CObjectWrapper::unwrap(transform_);
  UnsafeFieldAccessor<char,9,coord_t,Realm::AffineAccessor<char,9,coord_t> >
    *accessor = NULL;
  assert(domtrans.transform.n == 9);
  switch (domtrans.transform.m)
  {
#define AFFINE(DIM) \
    case DIM: \
      { \
        const AffineTransform<DIM,9,coord_t> transform = domtrans; \
        accessor = new UnsafeFieldAccessor<char,9,coord_t, \
                 Realm::AffineAccessor<char,9,coord_t> >(*handle, fid, transform); \
        break; \
      }
    LEGION_FOREACH_N(AFFINE)
#undef AFFINE
    default:
      assert(false);
  }

  return CObjectWrapper::wrap(accessor);
}
#endif

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

#define READ_POINT(DIM) \
void \
legion_accessor_array_##DIM##d_read_point(legion_accessor_array_##DIM##d_t handle_, \
                                          legion_point_##DIM##d_t point_, \
                                          void *dst, size_t bytes) \
{ \
  UnsafeFieldAccessor<char,DIM,coord_t,Realm::AffineAccessor<char,DIM,coord_t> > \
    *handle = CObjectWrapper::unwrap(handle_); \
  Point##DIM##D point = CObjectWrapper::unwrap(point_); \
 \
  memcpy(dst, handle->ptr(point), bytes); \
}
LEGION_FOREACH_N(READ_POINT)
#undef READ_POINT

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

#define WRITE_POINT(DIM) \
void \
legion_accessor_array_##DIM##d_write_point(legion_accessor_array_##DIM##d_t handle_, \
                                           legion_point_##DIM##d_t point_, \
                                           const void *src, size_t bytes) \
{ \
  UnsafeFieldAccessor<char,DIM,coord_t,Realm::AffineAccessor<char,DIM,coord_t> > \
    *handle = CObjectWrapper::unwrap(handle_); \
  Point##DIM##D point = CObjectWrapper::unwrap(point_); \
 \
  memcpy(handle->ptr(point), src, bytes); \
}
LEGION_FOREACH_N(WRITE_POINT)
#undef WRITE_POINT

void *
legion_accessor_array_1d_ref(legion_accessor_array_1d_t handle_,
                             legion_ptr_t ptr_)
{
  UnsafeFieldAccessor<char,1,coord_t,Realm::AffineAccessor<char,1,coord_t> >
    *handle = CObjectWrapper::unwrap(handle_);
  ptr_t ptr = CObjectWrapper::unwrap(ptr_);

  return handle->ptr(ptr.value);
}

#define REF_POINT(DIM) \
void * \
legion_accessor_array_##DIM##d_ref_point(legion_accessor_array_##DIM##d_t handle_, \
                                         legion_point_##DIM##d_t point_) \
{ \
  UnsafeFieldAccessor<char,DIM,coord_t,Realm::AffineAccessor<char,DIM,coord_t> > \
    *handle = CObjectWrapper::unwrap(handle_); \
  Point##DIM##D point = CObjectWrapper::unwrap(point_); \
 \
  return handle->ptr(point); \
}
LEGION_FOREACH_N(REF_POINT)
#undef REF_POINT

// -----------------------------------------------------------------------
// External Resource Operations
// -----------------------------------------------------------------------

void
legion_external_resources_destroy(legion_external_resources_t handle_)
{
  ExternalResources *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

size_t
legion_external_resources_size(legion_external_resources_t handle_)
{
  ExternalResources *handle = CObjectWrapper::unwrap(handle_);

  return handle->size();
}

legion_physical_region_t
legion_external_resources_get_region(legion_external_resources_t handle_,
                                     unsigned index)
{
  ExternalResources *handle = CObjectWrapper::unwrap(handle_);

  PhysicalRegion result = (*handle)[index];
  return CObjectWrapper::wrap(new PhysicalRegion(result));
}

//------------------------------------------------------------------------
// Mappable Operations
//------------------------------------------------------------------------

enum legion_mappable_type_id_t
legion_mappable_get_type(legion_mappable_t mappable_)
{
  Mappable *mappable = CObjectWrapper::unwrap(mappable_);

  return mappable->get_mappable_type();
}

legion_task_t
legion_mappable_as_task(legion_mappable_t mappable_)
{
  Mappable *mappable = CObjectWrapper::unwrap(mappable_);
  Task* task = const_cast<Task*>(mappable->as_task());
  assert(task != NULL);

  return CObjectWrapper::wrap(task);
}

legion_copy_t
legion_mappable_as_copy(legion_mappable_t mappable_)
{
  Mappable *mappable = CObjectWrapper::unwrap(mappable_);
  Copy* copy = const_cast<Copy*>(mappable->as_copy());
  assert(copy != NULL);

  return CObjectWrapper::wrap(copy);
}

legion_fill_t
legion_mappable_as_fill(legion_mappable_t mappable_)
{
  Mappable *mappable = CObjectWrapper::unwrap(mappable_);
  Fill* fill = const_cast<Fill*>(mappable->as_fill());
  assert(fill != NULL);

  return CObjectWrapper::wrap(fill);
}

legion_inline_t
legion_mappable_as_inline_mapping(legion_mappable_t mappable_)
{
  Mappable *mappable = CObjectWrapper::unwrap(mappable_);
  InlineMapping* mapping = const_cast<InlineMapping*>(mappable->as_inline());
  assert(mapping != NULL);

  return CObjectWrapper::wrap(mapping);
}

//------------------------------------------------------------------------
// Task Operations
//------------------------------------------------------------------------

legion_unique_id_t
legion_context_get_unique_id(legion_context_t ctx_)
{
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  const Task *task = Runtime::get_context_task(ctx);
  return task->get_unique_id();
}

legion_task_mut_t
legion_task_create_empty()
{
  TaskMut *task = new TaskMut();
  return CObjectWrapper::wrap(task);
}

void
legion_task_destroy(legion_task_mut_t handle_)
{
  TaskMut *handle = CObjectWrapper::unwrap(handle_);
  delete handle;
}

legion_task_t
legion_task_mut_as_task(legion_task_mut_t task_)
{
  TaskMut *task = CObjectWrapper::unwrap(task_);
  return CObjectWrapper::wrap(static_cast<Task *>(task));
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

legion_mapper_id_t
legion_task_get_mapper(legion_task_t task_)
{
  return CObjectWrapper::unwrap(task_)->map_id;
}

legion_mapping_tag_id_t
legion_task_get_tag(legion_task_t task_)
{
  return CObjectWrapper::unwrap(task_)->tag;
}

void
legion_task_id_attach_semantic_information(legion_runtime_t runtime_,
                                           legion_task_id_t task_id,
                                           legion_semantic_tag_t tag,
                                           const void *buffer,
                                           size_t size,
                                           bool is_mutable /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);

  runtime->attach_semantic_information(task_id, tag, buffer, size, is_mutable);
}

bool
legion_task_id_retrieve_semantic_information(
                                         legion_runtime_t runtime_,
                                         legion_task_id_t task_id,
                                         legion_semantic_tag_t tag,
                                         const void **result,
                                         size_t *size,
                                         bool can_fail /* = false */,
                                         bool wait_until_ready /* = false */)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);

  return runtime->retrieve_semantic_information(
                      task_id, tag, *result, *size, can_fail, wait_until_ready);
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

void
legion_task_set_args(legion_task_mut_t task_, void *args)
{
  TaskMut *task = CObjectWrapper::unwrap(task_);

  task->args = args;
}

size_t
legion_task_get_arglen(legion_task_t task_)
{
  Task *task = CObjectWrapper::unwrap(task_);

  return task->arglen;
}

void
legion_task_set_arglen(legion_task_mut_t task_, size_t arglen)
{
  TaskMut *task = CObjectWrapper::unwrap(task_);

  task->arglen = arglen;
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
legion_task_get_requirement(legion_task_t task_, unsigned idx)
{
  Task *task = CObjectWrapper::unwrap(task_);
  assert(idx < task->regions.size());

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

void
legion_task_add_future(legion_task_mut_t task_, legion_future_t future_)
{
  TaskMut *task = CObjectWrapper::unwrap(task_);
  Future *future = CObjectWrapper::unwrap(future_);

  task->futures.push_back(*future);
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
legion_execution_constraint_set_add_colocation_constraint(
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

void
legion_runtime_initialize(int *argc,
                          char ***argv,
                          bool filter /* = false */)
{
  return Runtime::initialize(argc, argv, filter);
}

int
legion_runtime_start(int argc,
                     char **argv,
                     bool background /* = false */)
{
  return Runtime::start(argc, argv, background);
}

int
legion_runtime_wait_for_shutdown(void)
{
  return Runtime::wait_for_shutdown();
}

void
legion_runtime_set_return_code(int return_code)
{
  Runtime::set_return_code(return_code);
}

void
legion_runtime_set_top_level_task_id(legion_task_id_t top_id)
{
  Runtime::set_top_level_task_id(top_id);
}

size_t
legion_runtime_get_maximum_dimension(void)
{
  return Runtime::get_maximum_dimension();
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
  FunctorWrapper(bool exc, bool func, unsigned dep,
                 legion_projection_functor_logical_region_t region_fn,
                 legion_projection_functor_logical_partition_t partition_fn,
                 legion_projection_functor_logical_region_mappable_t region_fn_mappable,
                 legion_projection_functor_logical_partition_mappable_t partition_fn_mappable)
    : ProjectionFunctor()
    , exclusive(exc)
    , functional(func)
    , depth(dep)
    , region_functor(region_fn)
    , partition_functor(partition_fn)
    , region_functor_mappable(region_fn_mappable)
    , partition_functor_mappable(partition_fn_mappable)
  {
    if (functional) {
      assert(!region_functor_mappable);
      assert(!partition_functor_mappable);
    } else {
      assert(!region_functor);
      assert(!partition_functor);
    }
  }

  FunctorWrapper(Runtime *rt,
                 bool exc, bool func, unsigned dep,
                 legion_projection_functor_logical_region_t region_fn,
                 legion_projection_functor_logical_partition_t partition_fn,
                 legion_projection_functor_logical_region_mappable_t region_fn_mappable,
                 legion_projection_functor_logical_partition_mappable_t partition_fn_mappable)
    : ProjectionFunctor(rt)
    , exclusive(exc)
    , functional(func)
    , depth(dep)
    , region_functor(region_fn)
    , partition_functor(partition_fn)
    , region_functor_mappable(region_fn_mappable)
    , partition_functor_mappable(partition_fn_mappable)
  {
    if (functional) {
      assert(!region_functor_mappable);
      assert(!partition_functor_mappable);
    } else {
      assert(!region_functor);
      assert(!partition_functor);
    }
  }

  using ProjectionFunctor::project;

  virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                LogicalRegion upper_bound,
                                const DomainPoint &point)
  {
    legion_runtime_t runtime_ = CObjectWrapper::wrap(runtime);
    legion_mappable_t mappable_ = CObjectWrapper::wrap_const(mappable);
    legion_logical_region_t upper_bound_ = CObjectWrapper::wrap(upper_bound);
    legion_domain_point_t point_ = CObjectWrapper::wrap(point);

    if (region_functor_mappable) {
      legion_logical_region_t result =
        region_functor_mappable(runtime_, mappable_, index, upper_bound_, point_);
      return CObjectWrapper::unwrap(result);
    }

    // Hack: This fallback is needed on because pre-control
    // replication doesn't know how to call is_functional().
    return ProjectionFunctor::project(mappable, index, upper_bound, point);
  }

  virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                LogicalPartition upper_bound,
                                const DomainPoint &point)
  {
    legion_runtime_t runtime_ = CObjectWrapper::wrap(runtime);
    legion_mappable_t mappable_ = CObjectWrapper::wrap_const(mappable);
    legion_logical_partition_t upper_bound_ = CObjectWrapper::wrap(upper_bound);
    legion_domain_point_t point_ = CObjectWrapper::wrap(point);

    if (partition_functor_mappable) {
      legion_logical_region_t result =
        partition_functor_mappable(runtime_, mappable_, index, upper_bound_, point_);
      return CObjectWrapper::unwrap(result);
    }

    // Hack: This fallback is needed on because pre-control
    // replication doesn't know how to call is_functional().
    return ProjectionFunctor::project(mappable, index, upper_bound, point);
  }

  virtual LogicalRegion project(LogicalRegion upper_bound,
                                const DomainPoint &point,
                                const Domain &launch_domain)
  {
    legion_runtime_t runtime_ = CObjectWrapper::wrap(runtime);
    legion_logical_region_t upper_bound_ = CObjectWrapper::wrap(upper_bound);
    legion_domain_point_t point_ = CObjectWrapper::wrap(point);
    legion_domain_t launch_domain_ = CObjectWrapper::wrap(launch_domain);

    assert(region_functor);
    legion_logical_region_t result =
      region_functor(runtime_, upper_bound_, point_, launch_domain_);
    return CObjectWrapper::unwrap(result);
  }

  virtual LogicalRegion project(LogicalPartition upper_bound,
                                const DomainPoint &point,
                                const Domain &launch_domain)
  {
    legion_runtime_t runtime_ = CObjectWrapper::wrap(runtime);
    legion_logical_partition_t upper_bound_ = CObjectWrapper::wrap(upper_bound);
    legion_domain_point_t point_ = CObjectWrapper::wrap(point);
    legion_domain_t launch_domain_ = CObjectWrapper::wrap(launch_domain);

    assert(partition_functor);
    legion_logical_region_t result =
      partition_functor(runtime_, upper_bound_, point_, launch_domain_);
    return CObjectWrapper::unwrap(result);
  }

  virtual bool is_exclusive(void) const { return exclusive; }

  virtual bool is_functional(void) const { return functional; }

  virtual unsigned get_depth(void) const { return depth; }

private:
  const bool exclusive;
  const bool functional;
  const unsigned depth;
  legion_projection_functor_logical_region_t region_functor;
  legion_projection_functor_logical_partition_t partition_functor;
  legion_projection_functor_logical_region_mappable_t region_functor_mappable;
  legion_projection_functor_logical_partition_mappable_t partition_functor_mappable;
};

legion_projection_id_t
legion_runtime_generate_static_projection_id()
{
  return Runtime::generate_static_projection_id();
}

legion_projection_id_t
legion_runtime_generate_library_projection_ids(
    legion_runtime_t runtime_,
    const char *library_name,
    size_t count)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);

  return runtime->generate_library_projection_ids(library_name, count);
}

legion_sharding_id_t
legion_runtime_generate_library_sharding_ids(
    legion_runtime_t runtime_,
    const char *library_name,
    size_t count)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);

  return runtime->generate_library_sharding_ids(library_name, count);
}

legion_reduction_op_id_t
legion_runtime_generate_library_reduction_ids(
    legion_runtime_t runtime_,
    const char *library_name,
    size_t count)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);

  return runtime->generate_library_reduction_ids(library_name, count);
}

void
legion_runtime_preregister_projection_functor(
  legion_projection_id_t id,
  bool exclusive,
  unsigned depth,
  legion_projection_functor_logical_region_t region_functor,
  legion_projection_functor_logical_partition_t partition_functor)
{
  FunctorWrapper *functor =
    new FunctorWrapper(exclusive, true, depth,
                       region_functor, partition_functor,
                       NULL, NULL);
  Runtime::preregister_projection_functor(id, functor);
}

void
legion_runtime_preregister_projection_functor_mappable(
  legion_projection_id_t id,
  bool exclusive,
  unsigned depth,
  legion_projection_functor_logical_region_mappable_t region_functor,
  legion_projection_functor_logical_partition_mappable_t partition_functor)
{
  FunctorWrapper *functor =
    new FunctorWrapper(exclusive, false, depth,
                       NULL, NULL,
                       region_functor, partition_functor);
  Runtime::preregister_projection_functor(id, functor);
}

void
legion_runtime_register_projection_functor(
  legion_runtime_t runtime_,
  legion_projection_id_t id,
  bool exclusive,
  unsigned depth,
  legion_projection_functor_logical_region_t region_functor,
  legion_projection_functor_logical_partition_t partition_functor)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);

  FunctorWrapper *functor =
    new FunctorWrapper(runtime, exclusive, true, depth,
                       region_functor, partition_functor,
                       NULL, NULL);
  runtime->register_projection_functor(id, functor);
}

void
legion_runtime_register_projection_functor_mappable(
  legion_runtime_t runtime_,
  legion_projection_id_t id,
  bool exclusive,
  unsigned depth,
  legion_projection_functor_logical_region_mappable_t region_functor,
  legion_projection_functor_logical_partition_mappable_t partition_functor)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);

  FunctorWrapper *functor =
    new FunctorWrapper(runtime, exclusive, false, depth,
                       NULL, NULL,
                       region_functor, partition_functor);
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
  const char *variant_name /* = NULL*/,
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

  if (id == LEGION_AUTO_GENERATE_ID)
    id = runtime->generate_dynamic_task_id();

  TaskVariantRegistrar registrar(id, variant_name, global);
  registrar.set_leaf(options.leaf);
  registrar.set_inner(options.inner);
  registrar.set_idempotent(options.idempotent);
  registrar.set_replicable(options.replicable);
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
  legion_variant_id_t variant_id /* = AUTO_GENERATE_ID */,
  const char *task_name /* = NULL*/,
  const char *variant_name /* = NULL*/,
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

  if (id == LEGION_AUTO_GENERATE_ID)
    id = Runtime::generate_static_task_id();

  TaskVariantRegistrar registrar(id, variant_name);
  registrar.set_leaf(options.leaf);
  registrar.set_inner(options.inner);
  registrar.set_idempotent(options.idempotent);
  registrar.set_replicable(options.replicable);
  if (layout_constraints)
    registrar.layout_constraints = *layout_constraints;
  if (execution_constraints)
    registrar.execution_constraints = *execution_constraints;

  CodeDescriptor code_desc(Realm::Type::from_cpp_type<Processor::TaskFuncPtr>());
  code_desc.add_implementation(new Realm::FunctionPointerImplementation((void(*)())wrapped_task_pointer));

  /*VariantID vid =*/ Runtime::preregister_task_variant(
    registrar, code_desc, userdata, userlen, task_name, variant_id);

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
  registrar.set_replicable(options.replicable);
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
  legion_variant_id_t variant_id /* = AUTO_GENERATE_ID */,
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
  registrar.set_replicable(options.replicable);
  if (layout_constraints)
    registrar.layout_constraints = *layout_constraints;
  if (execution_constraints)
    registrar.execution_constraints = *execution_constraints;

  CodeDescriptor code_desc(Realm::Type::from_cpp_type<Processor::TaskFuncPtr>());
  code_desc.add_implementation(new Realm::LLVMIRImplementation(llvmir, strlen(llvmir), entry_symbol));

  /*VariantID vid =*/ Runtime::preregister_task_variant(
    registrar, code_desc, userdata, userlen, task_name, variant_id);
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
  return legion_runtime_register_task_variant_python_source_qualname(
    runtime_, id, task_name, global,
    execution_constraints_, layout_constraints_, options,
    module_name, &function_name, 1,
    userdata, userlen);
}

legion_task_id_t
legion_runtime_register_task_variant_python_source_qualname(
  legion_runtime_t runtime_,
  legion_task_id_t id /* = AUTO_GENERATE_ID */,
  const char *task_name /* = NULL*/,
  bool global,
  legion_execution_constraint_set_t execution_constraints_,
  legion_task_layout_constraint_set_t layout_constraints_,
  legion_task_config_options_t options,
  const char *module_name,
  const char **function_qualname_,
  size_t function_qualname_len,
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
  registrar.set_replicable(options.replicable);
  if (layout_constraints)
    registrar.layout_constraints = *layout_constraints;
  if (execution_constraints)
    registrar.execution_constraints = *execution_constraints;

  std::vector<std::string> function_qualname(function_qualname_, function_qualname_ + function_qualname_len);
  CodeDescriptor code_desc(Realm::Type::from_cpp_type<Processor::TaskFuncPtr>());
  code_desc.add_implementation(new Realm::PythonSourceImplementation(module_name, function_qualname));

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

  TimingLauncher launcher(LEGION_MEASURE_SECONDS);
  Future f = runtime->issue_timing_measurement(ctx, launcher);  
  return CObjectWrapper::wrap(new Future(f));
}

legion_future_t
legion_issue_timing_op_microseconds(legion_runtime_t runtime_,
                                    legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  TimingLauncher launcher(LEGION_MEASURE_MICRO_SECONDS);
  Future f = runtime->issue_timing_measurement(ctx, launcher);  
  return CObjectWrapper::wrap(new Future(f));
}

legion_future_t
legion_issue_timing_op_nanoseconds(legion_runtime_t runtime_,
                                   legion_context_t ctx_)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  TimingLauncher launcher(LEGION_MEASURE_NANO_SECONDS);
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

legion_shard_id_t
legion_context_get_shard_id(legion_runtime_t runtime_,
                            legion_context_t ctx_,
                            bool I_know_what_I_am_doing)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  return runtime->get_shard_id(ctx, I_know_what_I_am_doing);
}

size_t
legion_context_get_num_shards(legion_runtime_t runtime_,
                              legion_context_t ctx_,
                              bool I_know_what_I_am_doing)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  return runtime->get_num_shards(ctx, I_know_what_I_am_doing);
}

legion_physical_region_t
legion_get_physical_region_by_id(
    legion_physical_region_t *regionptr, 
    int id, 
    int num_regions)
{
  assert(id < num_regions);
  return regionptr[id];
}

legion_index_space_t
legion_logical_region_get_index_space(legion_logical_region_t lr_)
{
  LogicalRegion lr = CObjectWrapper::unwrap(lr_);
  return CObjectWrapper::wrap(lr.get_index_space());
}

void
legion_task_cxx_to_c(
  const Task *task,
  const std::vector<PhysicalRegion> *regions,
  Context ctx, 
  Runtime *runtime,
  legion_task_t *taskptr,
  const legion_physical_region_t **regionptr,
  unsigned * num_regions_ptr,
  legion_context_t * ctxptr,
  legion_runtime_t * runtimeptr)
{
  CContext *cctx = new CContext(ctx, *regions);
  *taskptr = CObjectWrapper::wrap_const(task);
  *regionptr = cctx->regions();
  *num_regions_ptr = cctx->num_regions();
  *ctxptr = CObjectWrapper::wrap(cctx);
  *runtimeptr = CObjectWrapper::wrap(runtime);
}
