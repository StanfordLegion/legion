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


#ifndef __LEGION_C_H__
#define __LEGION_C_H__

/**
 * \file legion_c.h
 * Legion C API
 */

// ******************** IMPORTANT **************************
//
// This file is PURE C, **NOT** C++.
//
// ******************** IMPORTANT **************************

#include "legion_config.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

  // -----------------------------------------------------------------------
  // Proxy Types
  // -----------------------------------------------------------------------

// #define NEW_OPAQUE_TYPE(T) typedef void * T
#define NEW_OPAQUE_TYPE(T) typedef struct T { void *impl; } T
  NEW_OPAQUE_TYPE(legion_runtime_t);
  NEW_OPAQUE_TYPE(legion_context_t);
  NEW_OPAQUE_TYPE(legion_coloring_t);
  NEW_OPAQUE_TYPE(legion_domain_coloring_t);
  NEW_OPAQUE_TYPE(legion_index_space_allocator_t);
  NEW_OPAQUE_TYPE(legion_argument_map_t);
  NEW_OPAQUE_TYPE(legion_predicate_t);
  NEW_OPAQUE_TYPE(legion_future_t);
  NEW_OPAQUE_TYPE(legion_future_map_t);
  NEW_OPAQUE_TYPE(legion_task_launcher_t);
  NEW_OPAQUE_TYPE(legion_index_launcher_t);
  NEW_OPAQUE_TYPE(legion_inline_launcher_t);
  NEW_OPAQUE_TYPE(legion_physical_region_t);
  NEW_OPAQUE_TYPE(legion_accessor_generic_t);
  NEW_OPAQUE_TYPE(legion_accessor_array_t);
  NEW_OPAQUE_TYPE(legion_index_iterator_t);
  NEW_OPAQUE_TYPE(legion_task_t);
#undef NEW_OPAQUE_TYPE

  /**
   * @see ptr_t
   */
  typedef struct legion_ptr_t {
    unsigned value;
  } legion_ptr_t;

#define NEW_POINT_TYPE(T, DIM) typedef struct T { int x[DIM]; } T
  NEW_POINT_TYPE(legion_point_1d_t, 1);
  NEW_POINT_TYPE(legion_point_2d_t, 2);
  NEW_POINT_TYPE(legion_point_3d_t, 3);
#undef NEW_POINT_TYPE

#define NEW_RECT_TYPE(T, PT) typedef struct T { PT lo, hi; } T
  NEW_RECT_TYPE(legion_rect_1d_t, legion_point_1d_t);
  NEW_RECT_TYPE(legion_rect_2d_t, legion_point_2d_t);
  NEW_RECT_TYPE(legion_rect_3d_t, legion_point_3d_t);
#undef NEW_RECT_TYPE

  /**
   * @see LegionRuntime::HighLevel::Domain
   */
  typedef struct legion_domain_t {
    legion_lowlevel_id_t is_id;
    int dim;
    int rect_data[2 * MAX_RECT_DIM];
  } legion_domain_t;

  /**
   * @see LegionRuntime::HighLevel::IndexSpace
   */
  typedef struct legion_index_space_t {
    legion_lowlevel_id_t id;
  } legion_index_space_t;

  /**
   * @see LegionRuntime::HighLevel::IndexAllocator
   */
  typedef struct legion_index_allocator_t {
    legion_index_space_t index_space;
    legion_index_space_allocator_t allocator;
  } legion_index_allocator_t;

  /**
   * @see LegionRuntime::HighLevel::FieldSpace
   */
  typedef struct legion_field_space_t {
    legion_field_space_id_t id;
  } legion_field_space_t;

  /**
   * @see LegionRuntime::HighLevel::FieldAllocator
   */
  typedef struct legion_field_allocator_t {
    legion_field_space_t field_space;
    legion_context_t parent;
    legion_runtime_t runtime;
  } legion_field_allocator_t;

  /**
   * @see LegionRuntime::HighLevel::LogicalRegion
   */
  typedef struct legion_logical_region_t {
    legion_region_tree_id_t tree_id;
    legion_index_space_t index_space;
    legion_field_space_t field_space;
  } legion_logical_region_t;

  /**
   * @see LegionRuntime::HighLevel::LogicalPartition
   */
  typedef struct legion_logical_partition_t {
    legion_region_tree_id_t tree_id;
    legion_index_partition_t index_partition;
    legion_field_space_t field_space;
  } legion_logical_partition_t;
 
  /**
   * @see LegionRuntime::HighLevel::TaskArgument
   */
  typedef struct legion_task_argument_t {
    void *args;
    size_t arglen;
  } legion_task_argument_t;

  /**
   * @see LegionRuntime::HighLevel::TaskResult
   */
  typedef struct legion_task_result_t {
    void *value;
    size_t value_size;
  } legion_task_result_t;

  /**
   * @see LegionRuntime::Accessor::ByteOffset
   */
  typedef struct legion_byte_offset_t {
    int offset;
  } legion_byte_offset_t;

  /**
   * @see LegionRuntime::HighLevel::InputArgs
   */
  typedef struct legion_input_args_t {
    char **argv;
    int argc;
  } legion_input_args_t;

  /**
   * @see LegionRuntime::HighLevel::TaskConfigOptions
   */
  typedef struct legion_task_config_options_t {
    bool leaf /* = false */;
    bool inner /* = false */;
    bool idempotent /* = false */;
  }  legion_task_config_options_t;

  /**
   * Interface for a Legion C task returning void.
   */
  typedef
    void (*legion_task_pointer_void_t)(
      const legion_task_t /* task */,
      const legion_physical_region_t * /* regions */,
      unsigned /* num_regions */,
      legion_context_t /* ctx */,
      legion_runtime_t /* runtime */);

  /**
   * Interface for a Legion C task returning void.
   */
  typedef
    legion_task_result_t (*legion_task_pointer_t)(
      const legion_task_t /* task */,
      const legion_physical_region_t * /* regions */,
      unsigned /* num_regions */,
      legion_context_t /* ctx */,
      legion_runtime_t /* runtime */);

  // -----------------------------------------------------------------------
  // Pointer Operations
  // -----------------------------------------------------------------------

  /**
   * @see ptr_t::nil()
   */
  inline legion_ptr_t
  legion_ptr_nil(void)
  {
    legion_ptr_t ptr;
    ptr.value = (unsigned)-1;
    return ptr;
  }

  /**
   * @see ptr_t::is_null()
   */
  inline bool
  legion_ptr_is_null(legion_ptr_t ptr)
  {
    return ptr.value == (unsigned)-1;
  }

  // -----------------------------------------------------------------------
  // Domain Operations
  // -----------------------------------------------------------------------

  /**
   * @see LegionRuntime::HighLevel::Domain::from_rect()
   */
  legion_domain_t
  legion_domain_from_rect_1d(legion_rect_1d_t r);

  /**
   * @see LegionRuntime::HighLevel::Domain::from_rect()
   */
  legion_domain_t
  legion_domain_from_rect_2d(legion_rect_2d_t r);

  /**
   * @see LegionRuntime::HighLevel::Domain::from_rect()
   */
  legion_domain_t
  legion_domain_from_rect_3d(legion_rect_3d_t r);

  /**
   * @see LegionRuntime::HighLevel::Domain::Domain(
   *        LegionRuntime::HighLevel::IndexSpace)
   */
  legion_domain_t
  legion_domain_from_index_space(legion_index_space_t is);

  // -----------------------------------------------------------------------
  // Coloring Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::Coloring
   */
  legion_coloring_t
  legion_coloring_create(void);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::Coloring
   */
  void
  legion_coloring_destroy(legion_coloring_t handle);

  /**
   * @see LegionRuntime::HighLevel::Coloring
   */
  void
  legion_coloring_add_point(legion_coloring_t handle,
                            legion_color_t color,
                            legion_ptr_t point);

  /**
   * @see LegionRuntime::HighLevel::Coloring
   */
  void
  legion_coloring_add_range(legion_coloring_t handle,
                            legion_color_t color,
                            legion_ptr_t start,
                            legion_ptr_t end /**< inclusive */);

  // -----------------------------------------------------------------------
  // Index Space Operations
  // ----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::create_index_space(Context, size_t)
   */
  legion_index_space_t
  legion_index_space_create(legion_runtime_t runtime,
                            legion_context_t ctx,
                            size_t max_num_elmts);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::create_index_space(Context, Domain)
   */
  legion_index_space_t
  legion_index_space_create_domain(legion_runtime_t runtime,
                                   legion_context_t ctx,
                                   legion_domain_t domain);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::destroy_index_space()
   */
  void
  legion_index_space_destroy(legion_runtime_t runtime,
                             legion_context_t ctx,
                             legion_index_space_t handle);

  // -----------------------------------------------------------------------
  // Index Partition Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::create_index_partition(
   *        Context, IndexSpace, Coloring, bool, int)
   */
  legion_index_partition_t
  legion_index_partition_create_coloring(legion_runtime_t runtime,
                                         legion_context_t ctx,
                                         legion_index_space_t parent,
                                         legion_coloring_t coloring,
                                         bool disjoint,
                                         int part_color /* = -1 */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::create_index_partition(
   *        Context, IndexSpace, Domain, DomainColoring, bool, int)
   */
  legion_index_partition_t
  legion_index_partition_create_domain_coloring(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_index_space_t parent,
    legion_domain_t color_space,
    legion_domain_coloring_t coloring,
    bool disjoint,
    int part_color /* = -1 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::destroy_index_space()
   */
  void
  legion_index_partition_destroy(legion_runtime_t runtime,
                                legion_context_t ctx,
                                 legion_index_partition_t handle);

  // -----------------------------------------------------------------------
  // Field Space Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::create_field_space()
   */
  legion_field_space_t
  legion_field_space_create(legion_runtime_t runtime,
                            legion_context_t ctx);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::destroy_field_space()
   */
  void
  legion_field_space_destroy(legion_runtime_t runtime,
                             legion_context_t ctx,
                             legion_field_space_t handle);

  // -----------------------------------------------------------------------
  // Logical Region Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::create_logical_region()
   */
  legion_logical_region_t
  legion_logical_region_create(legion_runtime_t runtime,
                               legion_context_t ctx,
                               legion_index_space_t index,
                               legion_field_space_t fields);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::destroy_logical_region()
   */
  void
  legion_logical_region_destroy(legion_runtime_t runtime,
                                legion_context_t ctx,
                                legion_logical_region_t handle);

  // -----------------------------------------------------------------------
  // Logical Region Tree Traversal Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::get_logical_partition()
   */
  legion_logical_partition_t
  legion_logical_partition_create(legion_runtime_t runtime,
                                  legion_context_t ctx,
                                  legion_logical_region_t parent,
                                  legion_index_partition_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::destroy_logical_partition()
   */
  void
  legion_logical_partition_destroy(legion_runtime_t runtime,
                                   legion_context_t ctx,
                                   legion_logical_partition_t handle);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::get_logical_subregion()
   */
  legion_logical_region_t
  legion_logical_partition_get_logical_subregion(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_logical_partition_t parent,
    legion_index_space_t handle);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::get_logical_subregion_by_color()
   */
  legion_logical_region_t
  legion_logical_partition_get_logical_subregion_by_color(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_logical_partition_t parent,
    legion_color_t c);

  // -----------------------------------------------------------------------
  // Allocator and Argument Map Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::create_index_allocator()
   */
  legion_index_allocator_t
  legion_index_allocator_create(legion_runtime_t runtime,
                                legion_context_t ctx,
                                legion_index_space_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::IndexAllocator::~IndexAllocator()
   */
  void
  legion_index_allocator_destroy(legion_index_allocator_t handle);

  /**
   * @see LegionRuntime::HighLevel::IndexAllocator::alloc()
   */
  legion_ptr_t
  legion_index_allocator_alloc(legion_index_allocator_t allocator,
                               unsigned num_elements /* = 1 */);

  /**
   * @see LegionRuntime::HighLevel::IndexAllocator::free()
   */
  void
  legion_index_allocator_free(legion_index_allocator_t allocator,
                              legion_ptr_t ptr,
                              unsigned num_elements /* = 1 */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::create_field_allocator()
   */
  legion_field_allocator_t
  legion_field_allocator_create(legion_runtime_t runtime,
                                legion_context_t ctx,
                                legion_field_space_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::FieldAllocator::~FieldAllocator()
   */
  void
  legion_field_allocator_destroy(legion_field_allocator_t handle);

  /**
   * @see LegionRuntime::HighLevel::FieldAllocator::allocate_field()
   */
  legion_field_id_t
  legion_field_allocator_allocate_field(
    legion_field_allocator_t allocator,
    size_t field_size,
    legion_field_id_t desired_fieldid /* = AUTO_GENERATE_ID */);

  /**
   * @see LegionRuntime::HighLevel::FieldAllocator::free_field()
   */
  void
  legion_field_allocator_free_field(legion_field_allocator_t allocator,
                                    legion_field_id_t fid);

  /**
   * @see LegionRuntime::HighLevel::FieldAllocator::allocate_local_field()
   */
  legion_field_id_t
  legion_field_allocator_allocate_local_field(
    legion_field_allocator_t allocator,
    size_t field_size,
    legion_field_id_t desired_fieldid /* = AUTO_GENERATE_ID */);

  // -----------------------------------------------------------------------
  // Predicate Operations
  // -----------------------------------------------------------------------

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::Predicate::~Predicate()
   */
  void
  legion_predicate_destroy(legion_predicate_t handle);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see LegionRuntime::HighLevel::Predicate::TRUE_PRED
   */
  const legion_predicate_t
  legion_predicate_true(void);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see LegionRuntime::HighLevel::Predicate::FALSE_PRED
   */
  const legion_predicate_t
  legion_predicate_false(void);

  // -----------------------------------------------------------------------
  // Future Operations
  // -----------------------------------------------------------------------

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::Future::~Future()
   */
  void
  legion_future_destroy(legion_future_t handle);

  /**
   * @see LegionRuntime::HighLevel::Future::get_void_result()
   */
  void
  legion_future_get_void_result(legion_future_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::Future::get_result()
   */
  legion_task_result_t
  legion_future_get_result(legion_future_t handle);

  /**
   * @see LegionRuntime::HighLevel::Future::is_empty()
   */
  bool
  legion_future_is_empty(legion_future_t handle,
                         bool block /* = false */);

  /**
   * @see LegionRuntime::HighLevel::Future::get_untyped_pointer()
   */
  const void *
  legion_future_get_untyped_pointer(legion_future_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::execute_task()
   */
  legion_task_result_t
  legion_task_result_create(const void *handle, size_t size);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::execute_task()
   */
  void
  legion_task_result_destroy(legion_task_result_t handle);

  // -----------------------------------------------------------------------
  // Task Launch Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::TaskLauncher::TaskLauncher()
   */
  legion_task_launcher_t
  legion_task_launcher_create(
    legion_task_id_t tid,
    legion_task_argument_t arg,
    legion_predicate_t pred /* = legion_predicate_true() */,
    legion_mapper_id_t id /* = 0 */,
    legion_mapping_tag_id_t tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::TaskLauncher::~TaskLauncher()
   */
  void
  legion_task_launcher_destroy(legion_task_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::execute_task()
   */
  legion_future_t
  legion_task_launcher_execute(legion_runtime_t runtime,
                               legion_context_t ctx,
                               legion_task_launcher_t launcher);

  /**
   * @see LegionRuntime::HighLevel::TaskLauncher::add_region_requirement()
   */
  unsigned
  legion_task_launcher_add_region_requirement_logical_region(
    legion_task_launcher_t launcher,
    legion_logical_region_t handle,
    legion_privilege_mode_t priv,
    legion_coherence_property_t prop,
    legion_logical_region_t parent,
    legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see LegionRuntime::HighLevel::TaskLauncher::add_field()
   */
  void
  legion_task_launcher_add_field(legion_task_launcher_t launcher,
                                 unsigned idx,
                                 legion_field_id_t fid,
                                 bool inst /* = true */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::IndexLauncher::IndexLauncher()
   */
  legion_index_launcher_t
  legion_index_launcher_create(
    legion_task_id_t tid,
    legion_domain_t domain,
    legion_task_argument_t global_arg,
    legion_argument_map_t map,
    legion_predicate_t pred /* = legion_predicate_true() */,
    bool must /* = false */,
    legion_mapper_id_t id /* = 0 */,
    legion_mapping_tag_id_t tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::IndexLauncher::~IndexLauncher()
   */
  void
  legion_index_launcher_destroy(legion_index_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::execute_index_space(Context, const IndexLauncher &)
   */
  legion_future_map_t
  legion_index_launcher_execute(legion_runtime_t runtime,
                               legion_context_t ctx,
                               legion_index_launcher_t launcher);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::execute_index_space(Context, const IndexLauncher &, ReductionOpID)
   */
  legion_future_t
  legion_index_launcher_execute_reduction(legion_runtime_t runtime,
                                          legion_context_t ctx,
                                          legion_index_launcher_t launcher,
                                          legion_reduction_op_id_t redop);

  /**
   * @see LegionRuntime::HighLevel::IndexLauncher::add_region_requirement()
   */
  unsigned
  legion_index_launcher_add_region_requirement_logical_region(
    legion_index_launcher_t launcher,
    legion_logical_region_t handle,
    legion_projection_id_t proj /* = 0 */,
    legion_privilege_mode_t priv,
    legion_coherence_property_t prop,
    legion_logical_region_t parent,
    legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see LegionRuntime::HighLevel::IndexLauncher::add_region_requirement()
   */
  unsigned
  legion_index_launcher_add_region_requirement_logical_partition(
    legion_index_launcher_t launcher,
    legion_logical_partition_t handle,
    legion_projection_id_t proj /* = 0 */,
    legion_privilege_mode_t priv,
    legion_coherence_property_t prop,
    legion_logical_region_t parent,
    legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see LegionRuntime::HighLevel::IndexLaunchxer::add_field()
   */
  void
  legion_index_launcher_add_field(legion_index_launcher_t launcher,
                                 unsigned idx,
                                 legion_field_id_t fid,
                                 bool inst /* = true */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::InlineLauncher::InlineLauncher()
   */
  legion_inline_launcher_t
  legion_inline_launcher_create_logical_region(
    legion_logical_region_t handle,
    legion_privilege_mode_t priv,
    legion_coherence_property_t prop,
    legion_logical_region_t parent,
    legion_mapping_tag_id_t region_tag /* = 0 */,
    bool verified /* = false*/,
    legion_mapper_id_t id /* = 0 */,
    legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::InlineLauncher::~InlineLauncher()
   */
  void
  legion_inline_launcher_destroy(legion_inline_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::HighLevelRuntime::map_region()
   */
  legion_physical_region_t
  legion_inline_launcher_execute(legion_runtime_t runtime,
                                 legion_context_t ctx,
                                 legion_inline_launcher_t launcher);

  /**
   * @see LegionRuntime::HighLevel::InlineLauncher::add_field()
   */
  void
  legion_inline_launcher_add_field(legion_inline_launcher_t launcher,
                                   legion_field_id_t fid,
                                   bool inst /* = true */);

  /**
   * @see LegionRuntime::HighLevel::HighLevelRuntime::remap_region()
   */
  void
  legion_runtime_remap_region(legion_runtime_t runtime,
                              legion_context_t ctx,
                              legion_physical_region_t region);

  /**
   * @see LegionRuntime::HighLevel::HighLevelRuntime::unmap_region()
   */
  void
  legion_runtime_unmap_region(legion_runtime_t runtime,
                              legion_context_t ctx,
                              legion_physical_region_t region);

  /**
   * @see LegionRuntime::HighLevel::HighLevelRuntime::map_all_regions()
   */
  void
  legion_runtime_map_all_region(legion_runtime_t runtime,
                                legion_context_t ctx);

  /**
   * @see LegionRuntime::HighLevel::HighLevelRuntime::unmap_all_regions()
   */
  void
  legion_runtime_unmap_all_regions(legion_runtime_t runtime,
                                   legion_context_t ctx);

  // -----------------------------------------------------------------------
  // Physical Data Operations
  // -----------------------------------------------------------------------

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::PhysicalRegion::~PhysicalRegion()
   */
  void
  legion_physical_region_destroy(legion_physical_region_t handle);

  /**
   * @see LegionRuntime::HighLevel::PhysicalRegion::is_mapped()
   */
  bool
  legion_physical_region_is_mapped(legion_physical_region_t handle);

  /**
   * @see LegionRuntime::HighLevel::PhysicalRegion::wait_until_valid()
   */
  void
  legion_physical_region_wait_until_valid(legion_physical_region_t handle);

  /**
   * @see LegionRuntime::HighLevel::PhysicalRegion::is_valid()
   */
  bool
  legion_physical_region_is_valid(legion_physical_region_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::PhysicalRegion::get_field_accessor()
   */
  legion_accessor_generic_t
  legion_physical_region_get_field_accessor_generic(
    legion_physical_region_t handle,
    legion_field_id_t fid);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::PhysicalRegion::get_field_accessor()
   */
  legion_accessor_array_t
  legion_physical_region_get_field_accessor_array(
    legion_physical_region_t handle,
    legion_field_id_t fid);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   */
  void
  legion_accessor_generic_destroy(legion_accessor_generic_t handle);

  /**
   * @see LegionRuntime::Accessor::Generic::Untyped::read_untyped()
   */
  void
  legion_accessor_generic_read(legion_accessor_generic_t handle,
                               legion_ptr_t ptr,
                               void *dst,
                               size_t bytes);

  /**
   * @see LegionRuntime::Accessor::Generic::Untyped::write_untyped()
   */
  void
  legion_accessor_generic_write(legion_accessor_generic_t handle,
                                legion_ptr_t ptr,
                                const void *src,
                                size_t bytes);

  /**
   * @see LegionRuntime::Accessor::Generic::Untyped::raw_rect_ptr()
   */
  void *
  legion_accessor_generic_raw_rect_ptr_1d(legion_accessor_generic_t handle,
                                          legion_rect_1d_t rect,
                                          legion_rect_1d_t *subrect,
                                          legion_byte_offset_t *offsets);

  /**
   * @see LegionRuntime::Accessor::Generic::Untyped::raw_rect_ptr()
   */
  void *
  legion_accessor_generic_raw_rect_ptr_2d(legion_accessor_generic_t handle,
                                          legion_rect_2d_t rect,
                                          legion_rect_2d_t *subrect,
                                          legion_byte_offset_t *offsets);

  /**
   * @see LegionRuntime::Accessor::Generic::Untyped::raw_rect_ptr()
   */
  void *
  legion_accessor_generic_raw_rect_ptr_3d(legion_accessor_generic_t handle,
                                          legion_rect_3d_t rect,
                                          legion_rect_3d_t *subrect,
                                          legion_byte_offset_t *offsets);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   */
  void
  legion_accessor_array_destroy(legion_accessor_array_t handle);

  /**
   * @see LegionRuntime::Accessor::SOA::Untyped::elem_ptr()
   */
  void
  legion_accessor_array_read(legion_accessor_array_t handle,
                             legion_ptr_t ptr,
                             void *dst,
                             size_t bytes);

  /**
   * @see LegionRuntime::Accessor::SOA::Untyped::elem_ptr()
   */
  void
  legion_accessor_array_write(legion_accessor_array_t handle,
                              legion_ptr_t ptr,
                              const void *src,
                              size_t bytes);

  /**
   * @see LegionRuntime::Accessor::SOA::Untyped::elem_ptr()
   */
  void *
  legion_accessor_array_ref(legion_accessor_array_t handle,
                            legion_ptr_t ptr);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see LegionRuntime::HighLevel::IndexIterator::IndexIterator()
   */
  legion_index_iterator_t
  legion_index_iterator_create(legion_index_space_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see LegionRuntime::HighLevel::IndexIterator::~IndexIterator()
   */
  void
  legion_index_iterator_destroy(legion_index_iterator_t handle);

  /**
   * @see LegionRuntime::HighLevel::IndexIterator::has_next()
   */
  bool
  legion_index_iterator_has_next(legion_index_iterator_t handle);

  /**
   * @see LegionRuntime::HighLevel::IndexIterator::next()
   */
  legion_ptr_t
  legion_index_iterator_next(legion_index_iterator_t handle);

  // -----------------------------------------------------------------------
  // Task Operations
  // -----------------------------------------------------------------------

  /**
   * @see LegionRuntime::HighLevel::Task::args
   */
  void *
  legion_task_get_args(legion_task_t task);

  /**
   * @see LegionRuntime::HighLevel::Task::arglen
   */
  size_t
  legion_task_get_arglen(legion_task_t task);

  /**
   * @see LegionRuntime::HighLevel::Task::is_index_space
   */
  bool
  legion_task_get_is_index_space(legion_task_t task);

  /**
   * @see LegionRuntime::HighLevel::Task::local_args
   */
  void *
  legion_task_get_local_args(legion_task_t task);

  /**
   * @see LegionRuntime::HighLevel::Task::local_arglen
   */
  size_t
  legion_task_get_local_arglen(legion_task_t task);

  // -----------------------------------------------------------------------
  // Start-up Operations
  // -----------------------------------------------------------------------

  /**
   * @see LegionRuntime::HighLevel::HighLevelRuntime::start()
   */
  int
  legion_runtime_start(int argc,
                       char **argv,
                       bool background /* = false */);

  /**
   * @see LegionRuntime::HighLevel::HighLevelRuntime::wait_for_shutdown()
   */
  void
  legion_runtime_wait_for_shutdown(void);

  /**
   * @see LegionRuntime::HighLevel::HighLevelRuntime::set_top_level_task_id()
   */
  void
  legion_runtime_set_top_level_task_id(legion_task_id_t top_id);

  /**
   * @see LegionRuntime::HighLevel::HighLevelRuntime::get_input_args()
   */
  const legion_input_args_t
  legion_runtime_get_input_args(void);

  /**
   * @see LegionRuntime::HighLevel::HighLevelRuntime::register_legion_task()
   */
  legion_task_id_t
  legion_runtime_register_task_void(
    legion_task_id_t id,
    legion_processor_kind_t proc_kind,
    bool single,
    bool index,
    legion_variant_id_t vid /* = AUTO_GENERATE_ID */,
    legion_task_config_options_t options,
    const char *task_name /* = NULL*/,
    legion_task_pointer_void_t task_pointer);

  /**
   * @see LegionRuntime::HighLevel::HighLevelRuntime::register_legion_task()
   */
  legion_task_id_t
  legion_runtime_register_task(
    legion_task_id_t id,
    legion_processor_kind_t proc_kind,
    bool single,
    bool index,
    legion_variant_id_t vid /* = AUTO_GENERATE_ID */,
    legion_task_config_options_t options,
    const char *task_name /* = NULL*/,
    legion_task_pointer_t task_pointer);

#ifdef __cplusplus
}
#endif

#endif // __LEGION_C_H__
