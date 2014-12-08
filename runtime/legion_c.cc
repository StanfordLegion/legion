/* Copyright 2014 Stanford University
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

legion_domain_t
legion_domain_from_index_space(legion_index_space_t is_)
{
  IndexSpace is = CObjectWrapper::unwrap(is_);

  return CObjectWrapper::wrap(Domain(is));
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

// Caller must have ownership of parameter `handle`.
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
legion_runtime_map_all_region(legion_runtime_t runtime_,
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

void
task_wrapper_void(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx,
                  HighLevelRuntime *runtime,
                  const legion_task_pointer_void_t &task_pointer)
{
  const legion_task_t task_ = CObjectWrapper::wrap_const(task);
  std::vector<legion_physical_region_t> regions_;
  for (int i = 0; i < regions.size(); i++) {
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
  for (int i = 0; i < regions.size(); i++) {
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
