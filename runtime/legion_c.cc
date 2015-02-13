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

#include "legion.h"
#include "legion_c.h"
#include "legion_c_util.h"
#include "utilities.h"

using namespace LegionRuntime;
using namespace LegionRuntime::HighLevel;
typedef CObjectWrapper::Generic Generic;
typedef CObjectWrapper::SOA SOA;
typedef CObjectWrapper::AccessorGeneric AccessorGeneric;
typedef CObjectWrapper::AccessorArray AccessorArray;

// -----------------------------------------------------------------------
// Pointer Operations
// -----------------------------------------------------------------------

legion_ptr_t
legion_ptr_safe_cast(legion_runtime_t runtime_,
                     legion_context_t ctx_,
                     legion_ptr_t pointer_,
                     legion_logical_region_t region_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
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
  Rect<1> r = CObjectWrapper::unwrap(r_);

  return CObjectWrapper::wrap(Domain::from_rect<1>(r));
}

legion_domain_t
legion_domain_from_rect_2d(legion_rect_2d_t r_)
{
  Rect<2> r = CObjectWrapper::unwrap(r_);

  return CObjectWrapper::wrap(Domain::from_rect<2>(r));
}

legion_domain_t
legion_domain_from_rect_3d(legion_rect_3d_t r_)
{
  Rect<3> r = CObjectWrapper::unwrap(r_);

  return CObjectWrapper::wrap(Domain::from_rect<3>(r));
}

legion_rect_1d_t
legion_domain_get_rect_1d(legion_domain_t d_)
{
  Domain d = CObjectWrapper::unwrap(d_);

  return CObjectWrapper::wrap(d.get_rect<1>());
}

legion_rect_2d_t
legion_domain_get_rect_2d(legion_domain_t d_)
{
  Domain d = CObjectWrapper::unwrap(d_);

  return CObjectWrapper::wrap(d.get_rect<2>());
}

legion_rect_3d_t
legion_domain_get_rect_3d(legion_domain_t d_)
{
  Domain d = CObjectWrapper::unwrap(d_);

  return CObjectWrapper::wrap(d.get_rect<3>());
}

legion_domain_t
legion_domain_from_index_space(legion_index_space_t is_)
{
  IndexSpace is = CObjectWrapper::unwrap(is_);

  return CObjectWrapper::wrap(Domain(is));
}

// -----------------------------------------------------------------------
// Domain Point Operations
// -----------------------------------------------------------------------

legion_domain_point_t
legion_domain_point_from_point_1d(legion_point_1d_t p_)
{
  Point<1> p = CObjectWrapper::unwrap(p_);

  return CObjectWrapper::wrap(DomainPoint::from_point<1>(p));
}

legion_domain_point_t
legion_domain_point_from_point_2d(legion_point_2d_t p_)
{
  Point<2> p = CObjectWrapper::unwrap(p_);

  return CObjectWrapper::wrap(DomainPoint::from_point<2>(p));
}

legion_domain_point_t
legion_domain_point_from_point_3d(legion_point_3d_t p_)
{
  Point<3> p = CObjectWrapper::unwrap(p_);

  return CObjectWrapper::wrap(DomainPoint::from_point<3>(p));
}

legion_domain_point_t
legion_domain_point_safe_cast(legion_runtime_t runtime_,
                              legion_context_t ctx_,
                              legion_domain_point_t point_,
                              legion_logical_region_t region_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  DomainPoint point = CObjectWrapper::unwrap(point_);
  LogicalRegion region = CObjectWrapper::unwrap(region_);

  DomainPoint result = runtime->safe_cast(ctx, point, region);
  return CObjectWrapper::wrap(result);
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
legion_domain_coloring_destroy(legion_domain_coloring_t handle)
{
  delete CObjectWrapper::unwrap(handle);
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


// -------------------------------------------------------
// Index Space Operations
// -------------------------------------------------------

legion_index_space_t
legion_index_space_create(legion_runtime_t runtime_,
                          legion_context_t ctx_,
                          size_t max_num_elmts)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);

  IndexSpace is = runtime->create_index_space(ctx, max_num_elmts);
  return CObjectWrapper::wrap(is);
}

legion_index_space_t
legion_index_space_create_domain(legion_runtime_t runtime_,
                                 legion_context_t ctx_,
                                 legion_domain_t domain_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  Domain domain = CObjectWrapper::unwrap(domain_);

  IndexSpace is = runtime->create_index_space(ctx, domain);
  return CObjectWrapper::wrap(is);
}

void
legion_index_space_destroy(legion_runtime_t runtime_,
                           legion_context_t ctx_,
                           legion_index_space_t handle_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->destroy_index_space(ctx, handle);
}

legion_domain_t
legion_index_space_get_domain(legion_runtime_t runtime_,
                              legion_context_t ctx_,
                              legion_index_space_t handle_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  return CObjectWrapper::wrap(runtime->get_index_space_domain(ctx, handle));
}

//------------------------------------------------------------------------
// Index Partition Operations
//------------------------------------------------------------------------

legion_index_partition_t
legion_index_partition_create_coloring(legion_runtime_t runtime_,
                                       legion_context_t ctx_,
                                       legion_index_space_t parent_,
                                       legion_coloring_t coloring_,
                                       bool disjoint,
                                       int part_color /* = -1 */)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  Coloring *coloring = CObjectWrapper::unwrap(coloring_);

  IndexPartition ip =
    runtime->create_index_partition(ctx, parent, *coloring, disjoint,
                                    part_color);
  return ip;
}

legion_index_partition_t
legion_index_partition_create_blockify_1d(legion_runtime_t runtime_,
                                          legion_context_t ctx_,
                                          legion_index_space_t parent_,
                                          legion_blockify_1d_t blockify_,
                                          int part_color /* = -1 */)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  Blockify<1> blockify = CObjectWrapper::unwrap(blockify_);

  IndexPartition ip =
    runtime->create_index_partition(ctx, parent, blockify, part_color);

  return ip;
}

legion_index_partition_t
legion_index_partition_create_blockify_2d(legion_runtime_t runtime_,
                                          legion_context_t ctx_,
                                          legion_index_space_t parent_,
                                          legion_blockify_2d_t blockify_,
                                          int part_color /* = -1 */)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  Blockify<2> blockify = CObjectWrapper::unwrap(blockify_);

  IndexPartition ip =
    runtime->create_index_partition(ctx, parent, blockify, part_color);

  return ip;
}

legion_index_partition_t
legion_index_partition_create_blockify_3d(legion_runtime_t runtime_,
                                          legion_context_t ctx_,
                                          legion_index_space_t parent_,
                                          legion_blockify_3d_t blockify_,
                                          int part_color /* = -1 */)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  Blockify<3> blockify = CObjectWrapper::unwrap(blockify_);

  IndexPartition ip =
    runtime->create_index_partition(ctx, parent, blockify, part_color);

  return ip;
}

legion_index_partition_t
legion_index_partition_create_domain_coloring(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_index_space_t parent_,
  legion_domain_t color_space_,
  legion_domain_coloring_t coloring_,
  bool disjoint,
  int part_color /* = -1 */)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  IndexSpace parent = CObjectWrapper::unwrap(parent_);
  Domain color_space = CObjectWrapper::unwrap(color_space_);
  DomainColoring *coloring = CObjectWrapper::unwrap(coloring_);

  IndexPartition ip =
    runtime->create_index_partition(ctx, parent, color_space, *coloring,
                                    disjoint, part_color);
  return ip;
}

legion_index_space_t
legion_index_partition_get_index_subspace(legion_runtime_t runtime_,
                                          legion_context_t ctx_,
                                          legion_index_partition_t handle,
                                          legion_color_t color)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);

  IndexSpace is = runtime->get_index_subspace(ctx, handle, color);

  return CObjectWrapper::wrap(is);
}

void
legion_index_partition_destroy(legion_runtime_t runtime_,
                               legion_context_t ctx_,
                               legion_index_partition_t handle)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);

  runtime->destroy_index_partition(ctx, handle);
}

// -------------------------------------------------------
// Field Space Operations
// -------------------------------------------------------

legion_field_space_t
legion_field_space_create(legion_runtime_t runtime_,
                          legion_context_t ctx_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);

  FieldSpace fs = runtime->create_field_space(ctx);
  return CObjectWrapper::wrap(fs);
}

void
legion_field_space_destroy(legion_runtime_t runtime_,
                           legion_context_t ctx_,
                           legion_field_space_t handle_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  runtime->destroy_field_space(ctx, handle);
}

// -------------------------------------------------------
// Logical Region Operations
// -------------------------------------------------------

legion_logical_region_t
legion_logical_region_create(legion_runtime_t runtime_,
                             legion_context_t ctx_,
                             legion_index_space_t index_,
                             legion_field_space_t fields_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  IndexSpace index = CObjectWrapper::unwrap(index_);
  FieldSpace fields = CObjectWrapper::unwrap(fields_);

  LogicalRegion r = runtime->create_logical_region(ctx, index, fields);
  return CObjectWrapper::wrap(r);
}

void
legion_logical_region_destroy(legion_runtime_t runtime_,
                              legion_context_t ctx_,
                              legion_logical_region_t handle_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  LogicalRegion handle = CObjectWrapper::unwrap(handle_);

  runtime->destroy_logical_region(ctx, handle);
}

// -----------------------------------------------------------------------
// Logical Region Tree Traversal Operations
// -----------------------------------------------------------------------

legion_logical_partition_t
legion_logical_partition_create(legion_runtime_t runtime_,
                                legion_context_t ctx_,
                                legion_logical_region_t parent_,
                                legion_index_partition_t handle)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  LogicalRegion parent = CObjectWrapper::unwrap(parent_);

  LogicalPartition r = runtime->get_logical_partition(ctx, parent, handle);
  return CObjectWrapper::wrap(r);
}

legion_logical_partition_t
legion_logical_partition_create_by_tree(legion_runtime_t runtime_,
                                        legion_context_t ctx_,
                                        legion_index_partition_t handle,
                                        legion_field_space_t fspace_,
                                        legion_region_tree_id_t tid)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  FieldSpace fspace = CObjectWrapper::unwrap(fspace_);

  LogicalPartition r =
    runtime->get_logical_partition_by_tree(ctx, handle, fspace, tid);
  return CObjectWrapper::wrap(r);
}

void
legion_logical_partition_destroy(legion_runtime_t runtime_,
                                 legion_context_t ctx_,
                                 legion_logical_partition_t handle_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  LogicalPartition handle = CObjectWrapper::unwrap(handle_);

  runtime->destroy_logical_partition(ctx, handle);
}

legion_logical_region_t
legion_logical_partition_get_logical_subregion(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_logical_partition_t parent_,
  legion_index_space_t handle_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  LogicalPartition parent = CObjectWrapper::unwrap(parent_);
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  LogicalRegion r = runtime->get_logical_subregion(ctx, parent, handle);
  return CObjectWrapper::wrap(r);
}

legion_logical_region_t
legion_logical_partition_get_logical_subregion_by_color(
  legion_runtime_t runtime_,
  legion_context_t ctx_,
  legion_logical_partition_t parent_,
  legion_color_t c)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  LogicalPartition parent = CObjectWrapper::unwrap(parent_);

  LogicalRegion r = runtime->get_logical_subregion_by_color(ctx, parent, c);
  return CObjectWrapper::wrap(r);
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

legion_index_allocator_t
legion_index_allocator_create(legion_runtime_t runtime_,
                              legion_context_t ctx_,
                              legion_index_space_t handle_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  IndexAllocator isa = runtime->create_index_allocator(ctx, handle);
  return CObjectWrapper::wrap(isa);
}

void
legion_index_allocator_destroy(legion_index_allocator_t handle_)
{
  IndexAllocator handle = CObjectWrapper::unwrap(handle_);
  // Destructor is a nop anyway.
}

legion_ptr_t
legion_index_allocator_alloc(legion_index_allocator_t allocator_,
                             unsigned num_elements)
{
  IndexAllocator allocator = CObjectWrapper::unwrap(allocator_);
  ptr_t ptr = allocator.alloc(num_elements);
  return CObjectWrapper::wrap(ptr);
}

void
legion_index_allocator_free(legion_index_allocator_t allocator_,
                            legion_ptr_t ptr_,
                            unsigned num_elements)
{
  IndexAllocator allocator = CObjectWrapper::unwrap(allocator_);
  ptr_t ptr = CObjectWrapper::unwrap(ptr_);
  allocator.free(ptr, num_elements);
}

legion_field_allocator_t
legion_field_allocator_create(legion_runtime_t runtime_,
                              legion_context_t ctx_,
                              legion_field_space_t handle_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  FieldSpace handle = CObjectWrapper::unwrap(handle_);

  FieldAllocator fsa = runtime->create_field_allocator(ctx, handle);
  return CObjectWrapper::wrap(fsa);
}

void
legion_field_allocator_destroy(legion_field_allocator_t handle_)
{
  FieldAllocator handle = CObjectWrapper::unwrap(handle_);
  // Destructor is a nop anyway.
}

legion_field_id_t
legion_field_allocator_allocate_field(legion_field_allocator_t allocator_,
                                      size_t field_size,
                                      legion_field_id_t desired_fieldid)
{
  FieldAllocator allocator = CObjectWrapper::unwrap(allocator_);
  return allocator.allocate_field(field_size, desired_fieldid);
}

void
legion_field_allocator_free_field(legion_field_allocator_t allocator_,
                                  legion_field_id_t fid)
{
  FieldAllocator allocator = CObjectWrapper::unwrap(allocator_);
  allocator.free_field(fid);
}

legion_field_id_t
legion_field_allocator_allocate_local_field(legion_field_allocator_t allocator_,
                                            size_t field_size,
                                            legion_field_id_t desired_fieldid)
{
  FieldAllocator allocator = CObjectWrapper::unwrap(allocator_);
  return allocator.allocate_local_field(field_size, desired_fieldid);
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

//------------------------------------------------------------------------
// Future Operations
//------------------------------------------------------------------------

legion_future_t
legion_future_from_buffer(legion_runtime_t runtime_,
                          const void *buffer,
                          size_t size)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);

  Future *result = new Future(Future::from_buffer(runtime, buffer, size));
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

legion_task_result_t
legion_future_get_result(legion_future_t handle_)
{
  Future *handle = CObjectWrapper::unwrap(handle_);

  TaskResult result = handle->get_result<TaskResult>();
  return CObjectWrapper::wrap(result);
}

bool
legion_future_is_empty(legion_future_t handle_,
                       bool block /* = false */)
{
  Future *handle = CObjectWrapper::unwrap(handle_);

  return handle->is_empty(block);
}

const void *
legion_future_get_untyped_pointer(legion_future_t handle_)
{
  Future *handle = CObjectWrapper::unwrap(handle_);

  return handle->get_untyped_pointer();
}

legion_task_result_t
legion_task_result_create(const void *handle, size_t size)
{
  legion_task_result_t result;
  result.value = malloc(size);
  assert(result.value);
  memcpy(result.value, handle, size);
  result.value_size = size;
  return result;
}

void
legion_task_result_destroy(legion_task_result_t handle)
{
  free(handle.value);
}

// -----------------------------------------------------------------------
// Future Map Operations
// -----------------------------------------------------------------------

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

legion_task_result_t
legion_future_map_get_result(legion_future_map_t fm_,
                             legion_domain_point_t dp_)
{
  FutureMap *fm = CObjectWrapper::unwrap(fm_);
  DomainPoint dp = CObjectWrapper::unwrap(dp_);

  TaskResult result = fm->get_result<TaskResult>(dp);
  return CObjectWrapper::wrap(result);
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
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
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
legion_task_launcher_add_future(legion_task_launcher_t launcher_,
                                legion_future_t future_)
{
  TaskLauncher *launcher = CObjectWrapper::unwrap(launcher_);
  Future *future = CObjectWrapper::unwrap(future_);

  launcher->add_future(*future);
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

  IndexLauncher *launcher =
    new IndexLauncher(tid, domain, global_arg, *map, *pred, must, id, tag);
  return CObjectWrapper::wrap(launcher);
}

void
legion_index_launcher_destroy(legion_index_launcher_t launcher_)
{
  IndexLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  delete launcher;
}

legion_future_map_t
legion_index_launcher_execute(legion_runtime_t runtime_,
                             legion_context_t ctx_,
                             legion_index_launcher_t launcher_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  IndexLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  FutureMap f = runtime->execute_index_space(ctx, *launcher);
  return CObjectWrapper::wrap(new FutureMap(f));
}

legion_future_t
legion_index_launcher_execute_reduction(legion_runtime_t runtime_,
                                        legion_context_t ctx_,
                                        legion_index_launcher_t launcher_,
                                        legion_reduction_op_id_t redop)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  IndexLauncher *launcher = CObjectWrapper::unwrap(launcher_);

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
  IndexLauncher *launcher = CObjectWrapper::unwrap(launcher_);
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
  IndexLauncher *launcher = CObjectWrapper::unwrap(launcher_);
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
  IndexLauncher *launcher = CObjectWrapper::unwrap(launcher_);
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
  IndexLauncher *launcher = CObjectWrapper::unwrap(launcher_);
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
  IndexLauncher *launcher = CObjectWrapper::unwrap(launcher_);

  launcher->add_field(idx, fid, inst);
}

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
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
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
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  PhysicalRegion *region = CObjectWrapper::unwrap(region_);

  runtime->remap_region(ctx, *region);
}

void
legion_runtime_unmap_region(legion_runtime_t runtime_,
                            legion_context_t ctx_,
                            legion_physical_region_t region_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);
  PhysicalRegion *region = CObjectWrapper::unwrap(region_);

  runtime->unmap_region(ctx, *region);
}

void
legion_runtime_map_all_regions(legion_runtime_t runtime_,
                               legion_context_t ctx_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);

  runtime->map_all_regions(ctx);
}

void
legion_runtime_unmap_all_regions(legion_runtime_t runtime_,
                                 legion_context_t ctx_)
{
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_);

  runtime->unmap_all_regions(ctx);
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

legion_accessor_generic_t
legion_physical_region_get_accessor_generic(legion_physical_region_t handle_)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);

  AccessorGeneric *accessor =
    new AccessorGeneric(handle->get_accessor());
  return CObjectWrapper::wrap(accessor);
}

legion_accessor_generic_t
legion_physical_region_get_field_accessor_generic(
  legion_physical_region_t handle_,
  legion_field_id_t fid)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);

  AccessorGeneric *accessor =
    new AccessorGeneric(handle->get_field_accessor(fid));
  return CObjectWrapper::wrap(accessor);
}

legion_accessor_array_t
legion_physical_region_get_field_accessor_array(
  legion_physical_region_t handle_,
  legion_field_id_t fid)
{
  PhysicalRegion *handle = CObjectWrapper::unwrap(handle_);

  AccessorArray *accessor =
    new AccessorArray(
      handle->get_field_accessor(fid).typeify<char>().convert<SOA>());
  return CObjectWrapper::wrap(accessor);
}

void
legion_accessor_generic_destroy(legion_accessor_generic_t handle_)
{
  AccessorGeneric *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

void
legion_accessor_generic_read(legion_accessor_generic_t handle_,
                             legion_ptr_t ptr_,
                             void *dst,
                             size_t bytes)
{
  AccessorGeneric *handle = CObjectWrapper::unwrap(handle_);
  ptr_t ptr = CObjectWrapper::unwrap(ptr_);

  handle->read_untyped(ptr, dst, bytes);
}

void
legion_accessor_generic_write(legion_accessor_generic_t handle_,
                              legion_ptr_t ptr_,
                              const void *src,
                              size_t bytes)
{
  AccessorGeneric *handle = CObjectWrapper::unwrap(handle_);
  ptr_t ptr = CObjectWrapper::unwrap(ptr_);

  handle->write_untyped(ptr, src, bytes);
}

void
legion_accessor_generic_read_domain_point(legion_accessor_generic_t handle_,
                                          legion_domain_point_t dp_,
                                          void *dst,
                                          size_t bytes)
{
  AccessorGeneric *handle = CObjectWrapper::unwrap(handle_);
  DomainPoint dp = CObjectWrapper::unwrap(dp_);

  handle->read_untyped(dp, dst, bytes);
}

void
legion_accessor_generic_write_domain_point(legion_accessor_generic_t handle_,
                                           legion_domain_point_t dp_,
                                           const void *src,
                                           size_t bytes)
{
  AccessorGeneric *handle = CObjectWrapper::unwrap(handle_);
  DomainPoint dp = CObjectWrapper::unwrap(dp_);

  handle->write_untyped(dp, src, bytes);
}

void *
legion_accessor_generic_raw_rect_ptr_1d(legion_accessor_generic_t handle_,
                                        legion_rect_1d_t rect_,
                                        legion_rect_1d_t *subrect_,
                                        legion_byte_offset_t *offsets_)
{
  AccessorGeneric *handle = CObjectWrapper::unwrap(handle_);
  Rect<1> rect = CObjectWrapper::unwrap(rect_);

  Rect<1> subrect;
  Accessor::ByteOffset offsets[1];
  void *data = handle->raw_rect_ptr<1>(rect, subrect, &offsets[0]);
  *subrect_ = CObjectWrapper::wrap(subrect);
  offsets_[0] = CObjectWrapper::wrap(offsets[0]);
  return data;
}

void *
legion_accessor_generic_raw_rect_ptr_2d(legion_accessor_generic_t handle_,
                                        legion_rect_2d_t rect_,
                                        legion_rect_2d_t *subrect_,
                                        legion_byte_offset_t *offsets_)
{
  AccessorGeneric *handle = CObjectWrapper::unwrap(handle_);
  Rect<2> rect = CObjectWrapper::unwrap(rect_);

  Rect<2> subrect;
  Accessor::ByteOffset offsets[2];
  void *data = handle->raw_rect_ptr<2>(rect, subrect, &offsets[0]);
  *subrect_ = CObjectWrapper::wrap(subrect);
  offsets_[0] = CObjectWrapper::wrap(offsets[0]);
  offsets_[1] = CObjectWrapper::wrap(offsets[1]);
  return data;
}

void *
legion_accessor_generic_raw_rect_ptr_3d(legion_accessor_generic_t handle_,
                                        legion_rect_3d_t rect_,
                                        legion_rect_3d_t *subrect_,
                                        legion_byte_offset_t *offsets_)
{
  AccessorGeneric *handle = CObjectWrapper::unwrap(handle_);
  Rect<3> rect = CObjectWrapper::unwrap(rect_);

  Rect<3> subrect;
  Accessor::ByteOffset offsets[3];
  void *data = handle->raw_rect_ptr<3>(rect, subrect, &offsets[0]);
  *subrect_ = CObjectWrapper::wrap(subrect);
  offsets_[0] = CObjectWrapper::wrap(offsets[0]);
  offsets_[1] = CObjectWrapper::wrap(offsets[1]);
  offsets_[2] = CObjectWrapper::wrap(offsets[2]);
  return data;
}

void
legion_accessor_array_destroy(legion_accessor_array_t handle_)
{
  AccessorArray *handle = CObjectWrapper::unwrap(handle_);

  delete handle;
}

void
legion_accessor_array_read(legion_accessor_array_t handle_,
                           legion_ptr_t ptr_,
                           void *dst,
                           size_t bytes)
{
  AccessorArray *handle = CObjectWrapper::unwrap(handle_);
  ptr_t ptr = CObjectWrapper::unwrap(ptr_);

  char *data = &(handle->ref(ptr));
  std::copy(data, data + bytes, static_cast<char *>(dst));
}

void
legion_accessor_array_write(legion_accessor_array_t handle_,
                            legion_ptr_t ptr_,
                            const void *src,
                            size_t bytes)
{
  AccessorArray *handle = CObjectWrapper::unwrap(handle_);
  ptr_t ptr = CObjectWrapper::unwrap(ptr_);

  char *data = &(handle->ref(ptr));
  std::copy(static_cast<const char *>(src),
            static_cast<const char *>(src) + bytes,
            data);
}

void *
legion_accessor_array_ref(legion_accessor_array_t handle_,
                          legion_ptr_t ptr_)
{
  AccessorArray *handle = CObjectWrapper::unwrap(handle_);
  ptr_t ptr = CObjectWrapper::unwrap(ptr_);

  char *data = &(handle->ref(ptr));
  return static_cast<void *>(data);
}

legion_index_iterator_t
legion_index_iterator_create(legion_index_space_t handle_)
{
  IndexSpace handle = CObjectWrapper::unwrap(handle_);

  IndexIterator *iterator = new IndexIterator(handle);
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

//------------------------------------------------------------------------
// Task Operations
//------------------------------------------------------------------------

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

//------------------------------------------------------------------------
// Start-up Operations
//------------------------------------------------------------------------

int
legion_runtime_start(int argc,
                     char **argv,
                     bool background /* = false */)
{
  return HighLevelRuntime::start(argc, argv, background);
}

void
legion_runtime_wait_for_shutdown(void)
{
  HighLevelRuntime::wait_for_shutdown();
}

void
legion_runtime_set_top_level_task_id(legion_task_id_t top_id)
{
  HighLevelRuntime::set_top_level_task_id(top_id);
}

const legion_input_args_t
legion_runtime_get_input_args(void)
{
  return CObjectWrapper::wrap_const(HighLevelRuntime::get_input_args());
}

// a pointer to the callback function that is last registered
static legion_registration_callback_pointer_t callback;

void
registration_callback_wrapper(Machine machine,
                              HighLevelRuntime *rt,
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
    local_procs_[idx++] = CObjectWrapper::wrap_const(&proc);
  }

  callback(machine_, rt_, local_procs_, idx);
}

void
legion_runtime_set_registration_callback(
  legion_registration_callback_pointer_t callback_)
{
  callback = callback_;
  HighLevelRuntime::set_registration_callback(registration_callback_wrapper);
}

void
task_wrapper_void(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx,
                  HighLevelRuntime *runtime,
                  const legion_task_pointer_void_t &task_pointer)
{
  const legion_task_t task_ = CObjectWrapper::wrap_const(task);
  std::vector<legion_physical_region_t> regions_;
  for (size_t i = 0; i < regions.size(); i++) {
    regions_.push_back(CObjectWrapper::wrap_const(&regions[i]));
  }
  legion_physical_region_t *regions_ptr = NULL;
  if (regions.size() > 0) {
    regions_ptr = &regions_[0];
  }
  unsigned num_regions = regions_.size();
  legion_context_t ctx_ = CObjectWrapper::wrap(ctx);
  legion_runtime_t runtime_ = CObjectWrapper::wrap(runtime);

  task_pointer(task_, regions_ptr, num_regions, ctx_, runtime_);
}

legion_task_id_t
legion_runtime_register_task_void(
  legion_task_id_t id,
  legion_processor_kind_t proc_kind_,
  bool single,
  bool index,
  legion_variant_id_t vid /* = AUTO_GENERATE_ID */,
  legion_task_config_options_t options_,
  const char *task_name /* = NULL*/,
  legion_task_pointer_void_t task_pointer)
{
  Processor::Kind proc_kind = CObjectWrapper::unwrap(proc_kind_);
  TaskConfigOptions options = CObjectWrapper::unwrap(options_);

  return HighLevelRuntime::register_legion_task<
    legion_task_pointer_void_t, task_wrapper_void>(
    id, proc_kind, single, index, task_pointer, vid, options, task_name);
}

TaskResult
task_wrapper(const Task *task,
             const std::vector<PhysicalRegion> &regions,
             Context ctx,
             HighLevelRuntime *runtime,
             const legion_task_pointer_t &task_pointer)
{
  const legion_task_t task_ = CObjectWrapper::wrap_const(task);
  std::vector<legion_physical_region_t> regions_;
  for (size_t i = 0; i < regions.size(); i++) {
    regions_.push_back(CObjectWrapper::wrap_const(&regions[i]));
  }
  legion_physical_region_t *regions_ptr = NULL;
  if (regions_.size() > 0) {
    regions_ptr = &regions_[0];
  }
  unsigned num_regions = regions_.size();
  legion_context_t ctx_ = CObjectWrapper::wrap(ctx);
  legion_runtime_t runtime_ = CObjectWrapper::wrap(runtime);

  legion_task_result_t result_ =
    task_pointer(task_, regions_ptr, num_regions, ctx_, runtime_);

  TaskResult result = CObjectWrapper::unwrap(result_);
  legion_task_result_destroy(result_);
  return result;
}

legion_task_id_t
legion_runtime_register_task(
  legion_task_id_t id,
  legion_processor_kind_t proc_kind_,
  bool single,
  bool index,
  legion_variant_id_t vid /* = AUTO_GENERATE_ID */,
  legion_task_config_options_t options_,
  const char *task_name /* = NULL*/,
  legion_task_pointer_t task_pointer)
{
  Processor::Kind proc_kind = CObjectWrapper::unwrap(proc_kind_);
  TaskConfigOptions options = CObjectWrapper::unwrap(options_);

  return HighLevelRuntime::register_legion_task<
    TaskResult, legion_task_pointer_t, task_wrapper>(
    id, proc_kind, single, index, task_pointer, vid, options, task_name);
}

// -----------------------------------------------------------------------
// Timing Operations
// -----------------------------------------------------------------------

unsigned long long
legion_get_current_time_in_micros(void)
{
  return TimeStamp::get_current_time_in_micros();
}
